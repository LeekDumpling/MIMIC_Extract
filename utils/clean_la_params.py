# -*- coding: utf-8 -*-
"""
左心房参数清洗模块 — HFpEF 队列分析（步骤：LA 参数预处理）

功能
----
读取由 EchoGraphs 模块生成的三张左心房（LA）参数文件：
  - la_morphology_results.csv   形态学参数长表
  - la_kinematic_stats.csv      运动学参数长表（含 sub_item）
  - la_analysis_qc.csv          质量控制与元数据表

对这三张表进行：
  1. QC 过滤   — 依据 la_analysis_qc.csv 中的 status 字段，按可配置策略剔除
  2. 异常值检查 — 基于生理合理范围（与 MIMIC clean_cohort_csvs.py 逻辑一致）
  3. 缺失值处理 — 参考 impute_normalize.py 策略；影像参数算法失败时直接剔除行而非填补
  4. Z-score 标准化 — 连续变量（偏态列先 log1p 变换）
  5. 长表 → 宽表 pivot
  6. 保存输出文件

输出（默认 csv/la_params/）
  la_morphology_wide.csv     形态学参数宽表（主键：video_prefix + subject_id）
  la_kinematic_wide.csv      运动学参数宽表（主键：video_prefix + subject_id）
  la_params_qc_filtered.csv  经过 QC 过滤的 QC 表（供下游追溯）
  la_params_missingness.csv  各参数缺失率报告
  la_params_feature_decisions.csv  特征决策（保留/剔除/填补）

语义约定（务必遵守）
  - max_idx → LAVmax；min_idx → LAVmin
  - 所有 value 空字符串均表示"当前不可用"，绝不能按 0 处理
  - 速率类参数依赖真实 fps；BSA 参数依赖外部 clinical CSV
  - 当前参数均来自模型连续 28 点轮廓，非人工真值

用法（从仓库根目录或 utils/ 子目录均可运行）
  python utils/clean_la_params.py \\
      --morphology  path/to/la_morphology_results.csv \\
      --kinematic   path/to/la_kinematic_stats.csv \\
      --qc          path/to/la_analysis_qc.csv \\
      --output_dir  csv/la_params

  # 仅生成缺失率报告，不写输出文件：
  python utils/clean_la_params.py --report_only
"""

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np

    _blocked = [m for m in ("numexpr", "bottleneck") if m not in sys.modules]
    for _m in _blocked:
        sys.modules[_m] = None  # type: ignore[assignment]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            import pandas as pd
    finally:
        for _m in _blocked:
            sys.modules.pop(_m, None)
except (ImportError, AttributeError, ValueError):
    raise ImportError(
        "numpy 和 pandas 是必需依赖。\n请执行：pip install numpy pandas"
    ) from None

try:
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn 未安装；将跳过 Z-score 标准化。\n"
        "安装命令：pip install scikit-learn",
        UserWarning,
    )

# ---------------------------------------------------------------------------
# 生理范围约束：每个 LA 形态/运动学参数的合理上下界
# 参考：ASE/EACVI 指南及超声心动图参考值
# 影像算法极端值（由分割失败或轮廓跟踪异常导致）使用可视化检查补充
# ---------------------------------------------------------------------------
LA_PHYSIO_RANGES: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    # 形态学参数
    "LAVmax":             (10.0,  200.0),   # mL
    "LAVI":               (5.0,   120.0),   # mL/m²
    "LAVmin":             (5.0,   150.0),   # mL
    "LAVmin-i":           (2.0,   90.0),    # mL/m²
    "LAEF":               (10.0,  95.0),    # %
    "LAD-long":           (1.5,   8.0),     # cm
    "LAD-trans":          (1.0,   7.0),     # cm
    "3D LA sphericity":   (None,  None),    # 当前未实现，跳过
    "LA ellipticity":     (0.0,   5.0),     # 无量纲
    "LA circularity":     (0.0,   2.0),     # 无量纲
    "LA sphericity index":(0.0,   2.0),     # 无量纲
    "LA eccentricity index": (0.0, 1.0),    # 无量纲
    "MAT area":           (1.0,   20.0),    # cm²
    "TGAR":               (0.0,   5.0),     # 无量纲
    # 运动学参数（参数名来自 la_kinematic_stats.csv 的 parameter_name 列）
    "LASr":               (-5.0,  60.0),    # %
    "LASrR":              (-10.0, 10.0),    # s^-1
    "Time to peak LASrR": (0.0,   100.0),   # %cycle
    "LASct":              (-60.0, 5.0),     # %
    "GCS":                (-60.0, 5.0),     # %
    "GCSR":               (-10.0, 10.0),    # s^-1
    "LS":                 (-60.0, 5.0),     # %
    "LSR":                (-10.0, 10.0),    # s^-1
    "AS":                 (-60.0, 5.0),     # %
    "ASR":                (-10.0, 10.0),    # s^-1
    "4CH ellipticity rate":    (-5.0, 5.0), # 无量纲
    "4CH circularity rate":    (-5.0, 5.0), # 无量纲
    "Sphericity index rate":   (-5.0, 5.0), # 无量纲
    "MAT area":            (0.0,  20.0),    # cm²（运动学）
    "TGAR":                (0.0,   5.0),    # 无量纲（运动学）
    "Annular expansion rate":  (-10.0, 10.0),  # cm/s
    "Longitudinal stretching rate": (-10.0, 10.0),  # cm/s
}

# QC status 字段中"硬过滤"状态码（含其中任意一个 → 该视频完全剔除）
QC_HARD_FILTER_CODES = {"missing_spacing", "analysis_error"}

# QC status 字段中"软过滤"状态码（影响特定参数，由 _should_drop_param 函数处理）
QC_SOFT_FILTER_MAP = {
    "missing_bsa":        {"LAVI", "LAVmin-i"},
    "missing_fps":        None,    # None 表示影响所有速率类参数，由列名模式判断
    "no_valid_cycle":     None,    # 影响所有运动学参数
    "not_supported_single_view": {"3D LA sphericity"},
}

# 缺失率阈值（与 impute_normalize.py 保持一致）
THRESH_DROP: float = 0.30   # ≥ 30% 缺失 → 剔除该参数列（论文约定）
THRESH_FLAG: float = 0.20   # ≥ 20% 缺失 → 填补 + 添加 _missing_flag 列

# 不参与标准化的列
_NO_NORM_COLS = {"video_prefix", "subject_id", "study_id", "source_group",
                 "keyframe_method"}

# 正态性判断：|偏度| ≤ 此阈值 → 近正态（与 la_analysis.py 保持一致）
_SKEW_THRESH: float = 1.0


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _parse_status_set(status_val) -> set:
    """将 status 字段（可能是 ';' 拼接的多个状态码）解析为集合。"""
    if pd.isna(status_val) or str(status_val).strip() == "":
        return set()
    return {s.strip() for s in str(status_val).split(";")}


def _is_binary_col(series: pd.Series) -> bool:
    """判断列是否为二值型（仅含 0/1 及 NaN）。"""
    uniq = set(series.dropna().unique())
    return uniq.issubset({0, 1, 0.0, 1.0, True, False})


def _resolve_path(path: str) -> str:
    """将相对路径解析为绝对路径（支持从 utils/ 子目录启动）。"""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


# ---------------------------------------------------------------------------
# 步骤 1：加载与基础验证
# ---------------------------------------------------------------------------

def load_la_files(
    morphology_path: str,
    kinematic_path: str,
    qc_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载三张 LA 参数文件，检查必需列是否存在，统一处理 value 列类型。

    返回
    ----
    df_morph, df_kine, df_qc
    """
    required_morph = {"video_prefix", "subject_id", "study_id", "source_group",
                      "parameter_name", "value", "unit", "frame_idx", "status"}
    required_kine  = {"video_prefix", "subject_id", "study_id", "source_group",
                      "parameter_name", "sub_item", "value", "unit",
                      "cycle_count", "fps", "status"}
    required_qc    = {"video_prefix", "source_group", "subject_id", "study_id",
                      "status", "spacing_found", "fps", "valid_cycle_count",
                      "keyframe_method"}

    def _load(path: str, required: set, label: str) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{label} 缺少必需列：{missing}\n文件路径：{path}"
            )
        # value 列：空字符串 → NaN（保留 None 语义）
        if "value" in df.columns:
            df["value"] = df["value"].replace("", np.nan)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    df_morph = _load(morphology_path, required_morph, "la_morphology_results")
    df_kine  = _load(kinematic_path,  required_kine,  "la_kinematic_stats")
    df_qc    = _load(qc_path,         required_qc,    "la_analysis_qc")

    print(f"  [加载] 形态表  {df_morph.shape[0]:>5} 行 × {df_morph.shape[1]} 列")
    print(f"  [加载] 运动表  {df_kine.shape[0]:>5} 行 × {df_kine.shape[1]} 列")
    print(f"  [加载] QC 表   {df_qc.shape[0]:>5} 行 × {df_qc.shape[1]} 列")
    return df_morph, df_kine, df_qc


# ---------------------------------------------------------------------------
# 步骤 2：QC 过滤
# ---------------------------------------------------------------------------

def apply_qc_filter(
    df_morph: pd.DataFrame,
    df_kine: pd.DataFrame,
    df_qc: pd.DataFrame,
    hard_filter_codes: set = QC_HARD_FILTER_CODES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    按 QC 表的 status 字段过滤视频。

    硬过滤（含 missing_spacing 或 analysis_error）→ 完全从所有表中移除。

    返回
    ----
    df_morph_filtered, df_kine_filtered, df_qc_filtered, removed_prefixes
    """
    # 解析每个视频的状态码集合
    df_qc = df_qc.copy()
    df_qc["_status_set"] = df_qc["status"].apply(_parse_status_set)

    # 找出需要硬过滤的视频前缀
    mask_remove = df_qc["_status_set"].apply(
        lambda s: bool(s & hard_filter_codes)
    )
    removed_prefixes = df_qc.loc[mask_remove, "video_prefix"].tolist()

    if removed_prefixes:
        print(f"\n  [QC 硬过滤] 移除 {len(removed_prefixes)} 个视频"
              f"（含 {hard_filter_codes} 之一）")

    df_qc_filtered = df_qc.loc[~mask_remove].drop(columns=["_status_set"])
    valid_prefixes = set(df_qc_filtered["video_prefix"])

    df_morph_filtered = df_morph[df_morph["video_prefix"].isin(valid_prefixes)].copy()
    df_kine_filtered  = df_kine[df_kine["video_prefix"].isin(valid_prefixes)].copy()

    print(f"  [QC 过滤后] 形态表 {len(df_morph_filtered)} 行，"
          f"运动表 {len(df_kine_filtered)} 行，"
          f"有效视频 {len(valid_prefixes)} 个")
    return df_morph_filtered, df_kine_filtered, df_qc_filtered, removed_prefixes


# ---------------------------------------------------------------------------
# 步骤 3：行级状态码过滤（软过滤）
# ---------------------------------------------------------------------------

def _should_set_nan_for_param(param_name: str, status_codes: set) -> bool:
    """
    判断给定参数名在给定状态码集合下是否应被置为 NaN（而非 0 填补）。

    逻辑：
    - not_supported_single_view → 仅 3D LA sphericity
    - missing_bsa               → LAVI、LAVmin-i
    - no_valid_cycle            → 所有运动学参数（无法判断归属时全部置 NaN）
    - missing_fps               → 速率类（s^-1 或 /s 单位）暂时全置 NaN
    - analysis_error / missing_spacing → 已在硬过滤阶段处理，这里不再重复
    """
    if not status_codes or status_codes == {"ok"}:
        return False

    for code, affected in QC_SOFT_FILTER_MAP.items():
        if code not in status_codes:
            continue
        if affected is None:
            return True
        if param_name in affected:
            return True
    return False


def apply_row_status_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    对形态表或运动表中每行的 status，若表明该参数不可用，则将 value 置为 NaN。

    不同于硬过滤（整视频移除），这里只针对具体参数行。
    """
    df = df.copy()
    for idx, row in df.iterrows():
        codes = _parse_status_set(row["status"])
        if codes and codes != {"ok"}:
            if _should_set_nan_for_param(row["parameter_name"], codes):
                df.at[idx, "value"] = np.nan
    return df


# ---------------------------------------------------------------------------
# 步骤 4：生理范围检查（影像参数）
# ---------------------------------------------------------------------------

def apply_physio_range_check(
    df: pd.DataFrame,
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = LA_PHYSIO_RANGES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对 value 列按 parameter_name 施加生理范围约束。

    超出范围的值被置为 NaN（不直接剔除行，以保留该视频其他参数）。
    参考论文：影像学参数通过可视化分布图检查，识别由分割失败或
    轮廓跟踪异常造成的极端值。

    返回
    ----
    df_checked : pd.DataFrame  （已处理）
    outlier_log : pd.DataFrame （被置 NaN 的记录日志）
    """
    df = df.copy()
    log_rows = []

    for idx, row in df.iterrows():
        pname = row["parameter_name"]
        val = row["value"]
        if pd.isna(val):
            continue
        if pname not in ranges:
            continue
        lo, hi = ranges[pname]
        out_of_range = False
        if lo is not None and val < lo:
            out_of_range = True
        if hi is not None and val > hi:
            out_of_range = True
        if out_of_range:
            log_rows.append({
                "video_prefix":  row.get("video_prefix", ""),
                "parameter_name": pname,
                "original_value": val,
                "range_lo": lo,
                "range_hi": hi,
                "action": "set_nan",
            })
            df.at[idx, "value"] = np.nan

    outlier_log = pd.DataFrame(log_rows)
    if len(outlier_log):
        print(f"  [生理范围] 置 NaN 共 {len(outlier_log)} 个超范围值")
    else:
        print("  [生理范围] 未发现超范围值")
    return df, outlier_log


# ---------------------------------------------------------------------------
# 步骤 5：长表 → 宽表 pivot
# ---------------------------------------------------------------------------

def _safe_col_name(name: str) -> str:
    """
    将参数名转换为程序安全的列名（与论文宽表命名规则一致）。

      空格 → _
      /    → _
      %    → pct
      s^-1 → s_inv
      -    → 保留（仅 LAVmin-i 等有意义的连字符）
    """
    name = str(name)
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace("%", "pct")
    name = name.replace("s^-1", "s_inv")
    return name


def pivot_morphology(df: pd.DataFrame) -> pd.DataFrame:
    """
    形态表长 → 宽。

    主键：video_prefix
    列名：parameter_name（安全化）
    附加元数据列：subject_id, study_id, source_group
    """
    id_cols = ["video_prefix", "subject_id", "study_id", "source_group"]
    id_cols = [c for c in id_cols if c in df.columns]

    pivot = df.pivot_table(
        index=id_cols,
        columns="parameter_name",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None
    # 安全化列名（非 id_cols 部分）
    pivot.columns = [
        c if c in id_cols else _safe_col_name(c)
        for c in pivot.columns
    ]
    return pivot


def pivot_kinematic(df: pd.DataFrame) -> pd.DataFrame:
    """
    运动表长 → 宽。

    主键：video_prefix
    列名：parameter_name + "__" + sub_item（安全化）
    附加元数据列：subject_id, study_id, source_group
    """
    id_cols = ["video_prefix", "subject_id", "study_id", "source_group"]
    id_cols = [c for c in id_cols if c in df.columns]

    df = df.copy()
    df["_col"] = (
        df["parameter_name"].apply(_safe_col_name)
        + "__"
        + df["sub_item"].astype(str)
    )

    pivot = df.pivot_table(
        index=id_cols,
        columns="_col",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None
    return pivot


# ---------------------------------------------------------------------------
# 步骤 6：缺失值分析与处理（宽表）
# ---------------------------------------------------------------------------

def analyse_missingness_wide(
    df: pd.DataFrame,
    skip_cols: List[str],
    thresh_drop: float = THRESH_DROP,
) -> pd.DataFrame:
    """
    对宽表计算每列缺失率，返回缺失率报告。
    """
    rows = []
    for col in df.columns:
        if col in skip_cols:
            continue
        n_total = len(df)
        n_miss = int(df[col].isnull().sum())
        pct = n_miss / n_total if n_total else 0.0
        is_bin = _is_binary_col(df[col])

        if pct >= thresh_drop:
            decision = "DROP"
            note = f"{pct*100:.1f}% 缺失 ≥ {thresh_drop*100:.0f}% 阈值"
        elif pct >= THRESH_FLAG:
            decision = "IMPUTE+FLAG"
            note = f"{pct*100:.1f}% 缺失 → 填补 + _missing_flag"
        elif pct > 0:
            decision = "IMPUTE"
            note = f"{pct*100:.1f}% 缺失 → 中位数/众数填补"
        else:
            decision = "KEEP"
            note = "无缺失"

        rows.append({
            "column":      col,
            "n_total":     n_total,
            "n_missing":   n_miss,
            "pct_missing": round(pct * 100, 2),
            "dtype":       str(df[col].dtype),
            "is_binary":   is_bin,
            "decision":    decision,
            "note":        note,
        })
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


def impute_wide(
    df: pd.DataFrame,
    miss_report: pd.DataFrame,
    drop_imaging_failures: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    对宽表应用缺失值处理决策。

    影像参数（drop_imaging_failures=True）：
      缺失率超过阈值的列直接剔除；低阈值缺失则删除该行（对应参数）而非填补。
      这与论文规定一致：影像参数算法失败时剔除样本，不进行数值填补。

    注意：由于宽表中 LA 参数列均为影像导出值，一旦某列缺失率超过阈值则整列剔除。
    缺失率未超阈值但有个别缺失的行：若 drop_imaging_failures=True，则直接置 NaN
    保留（不填补），由下游分析时的 dropna(subset=[...]) 处理。

    返回
    ----
    df_imputed, dropped_cols, added_flag_cols
    """
    df = df.copy()
    dropped_cols: List[str] = []
    added_flag_cols: List[str] = []

    for _, row in miss_report.iterrows():
        col = row["column"]
        decision = row["decision"]
        if col not in df.columns:
            continue

        if decision == "DROP":
            df.drop(columns=[col], inplace=True)
            dropped_cols.append(col)

        elif decision in ("IMPUTE", "IMPUTE+FLAG") and not drop_imaging_failures:
            # 只在非影像参数模式下才填补
            add_flag = (decision == "IMPUTE+FLAG")
            if bool(row["is_binary"]):
                fill_val = df[col].mode(dropna=True)
                fill_val = fill_val.iloc[0] if len(fill_val) else 0
            else:
                fill_val = df[col].median()
            if add_flag:
                flag_col = f"{col}_missing_flag"
                df[flag_col] = df[col].isnull().astype(int)
                added_flag_cols.append(flag_col)
            df[col] = df[col].fillna(fill_val)

        # 影像参数模式（drop_imaging_failures=True）或 KEEP：保留 NaN，不填补

    return df, dropped_cols, added_flag_cols


# ---------------------------------------------------------------------------
# 步骤 7：Z-score 标准化（宽表）
# ---------------------------------------------------------------------------

def normalise_wide(
    df: pd.DataFrame,
    skip_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对宽表中的连续 LA 参数列进行 Z-score 标准化（偏态列先 log1p）。

    返回
    ----
    df_norm, norm_log
    """
    if not _SKLEARN_AVAILABLE:
        warnings.warn("scikit-learn 不可用，跳过标准化。", UserWarning)
        return df, pd.DataFrame()

    df = df.copy()
    log_rows = []
    no_norm = set(skip_cols) | _NO_NORM_COLS
    no_norm |= {c for c in df.columns if c.endswith("_missing_flag")}

    for col in df.columns:
        if col in no_norm:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if _is_binary_col(df[col]):
            continue

        vals = df[col].dropna().values.astype(float)
        if len(vals) < 2:
            continue
        skew = float(pd.Series(vals).skew())

        col_vals = df[col].values.astype(float)
        valid_mask = ~np.isnan(col_vals)

        if abs(skew) > _SKEW_THRESH:
            transform = "log1p + z-score"
            col_vals[valid_mask] = np.log1p(np.clip(col_vals[valid_mask], 0, None))
        else:
            transform = "z-score"

        scaler = StandardScaler()
        col_vals[valid_mask] = scaler.fit_transform(
            col_vals[valid_mask].reshape(-1, 1)
        ).ravel()
        df[col] = col_vals

        log_rows.append({
            "column":    col,
            "skewness_before": round(skew, 3),
            "transform": transform,
        })

    return df, pd.DataFrame(log_rows)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def process_la_params(
    morphology_path: str,
    kinematic_path: str,
    qc_path: str,
    output_dir: str,
    report_only: bool = False,
    hard_filter_codes: Optional[set] = None,
    thresh_drop: float = THRESH_DROP,
    no_normalise: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    完整 LA 参数清洗流程。

    参数
    ----
    morphology_path   : la_morphology_results.csv 路径
    kinematic_path    : la_kinematic_stats.csv 路径
    qc_path           : la_analysis_qc.csv 路径
    output_dir        : 输出目录
    report_only       : 仅生成缺失率报告，不写输出文件
    hard_filter_codes : 硬过滤状态码集合（默认使用 QC_HARD_FILTER_CODES）
    thresh_drop       : 列缺失率超过此值则剔除该列（默认 0.30）
    no_normalise      : True → 跳过 Z-score 标准化

    返回
    ----
    dict with keys:
      "morphology_wide", "kinematic_wide", "qc_filtered",
      "missingness_morph", "missingness_kine",
      "outlier_log_morph", "outlier_log_kine"
    """
    if hard_filter_codes is None:
        hard_filter_codes = QC_HARD_FILTER_CODES

    print(f"\n{'='*60}")
    print("LA 参数清洗流程开始")
    print(f"{'='*60}")

    # 1. 加载
    df_morph, df_kine, df_qc = load_la_files(
        morphology_path, kinematic_path, qc_path
    )

    # 2. QC 硬过滤
    df_morph, df_kine, df_qc, removed = apply_qc_filter(
        df_morph, df_kine, df_qc, hard_filter_codes=hard_filter_codes
    )

    # 3. 行级软过滤（将不可用参数值置 NaN）
    df_morph = apply_row_status_filter(df_morph)
    df_kine  = apply_row_status_filter(df_kine)

    # 4. 生理范围检查
    df_morph, outlier_morph = apply_physio_range_check(df_morph)
    df_kine,  outlier_kine  = apply_physio_range_check(df_kine)

    # 5. 宽表 pivot
    morph_wide = pivot_morphology(df_morph)
    kine_wide  = pivot_kinematic(df_kine)
    print(f"  [pivot] 形态宽表 {morph_wide.shape}，运动宽表 {kine_wide.shape}")

    # ID 列（不参与缺失分析和标准化）
    id_cols = ["video_prefix", "subject_id", "study_id", "source_group"]

    # 6. 缺失分析
    miss_morph = analyse_missingness_wide(morph_wide, skip_cols=id_cols,
                                          thresh_drop=thresh_drop)
    miss_kine  = analyse_missingness_wide(kine_wide,  skip_cols=id_cols,
                                          thresh_drop=thresh_drop)

    print("\n  --- 形态表缺失率（缺失 > 0 的列）---")
    vis = miss_morph[miss_morph["pct_missing"] > 0]
    if len(vis):
        print(vis[["column", "pct_missing", "decision"]].to_string(index=False))

    print("\n  --- 运动表缺失率（缺失 > 0 的列）---")
    vis = miss_kine[miss_kine["pct_missing"] > 0]
    if len(vis):
        print(vis[["column", "pct_missing", "decision"]].to_string(index=False))

    if report_only:
        return {
            "morphology_wide":  morph_wide,
            "kinematic_wide":   kine_wide,
            "qc_filtered":      df_qc,
            "missingness_morph": miss_morph,
            "missingness_kine": miss_kine,
            "outlier_log_morph": outlier_morph,
            "outlier_log_kine": outlier_kine,
        }

    # 7. 缺失值处理（影像参数不填补，仅剔除超阈值列）
    morph_wide, dropped_m, flags_m = impute_wide(morph_wide, miss_morph,
                                                  drop_imaging_failures=True)
    kine_wide,  dropped_k, flags_k = impute_wide(kine_wide,  miss_kine,
                                                  drop_imaging_failures=True)

    print(f"\n  形态表：剔除列 {dropped_m or 'none'}，新增 flag 列 {flags_m or 'none'}")
    print(f"  运动表：剔除列 {dropped_k or 'none'}，新增 flag 列 {flags_k or 'none'}")

    # 8. 标准化
    if not no_normalise:
        morph_wide, norm_log_m = normalise_wide(morph_wide, skip_cols=id_cols)
        kine_wide,  norm_log_k = normalise_wide(kine_wide,  skip_cols=id_cols)
        if len(norm_log_m):
            print("\n  形态表标准化日志（前 5 行）：")
            print(norm_log_m.head().to_string(index=False))
        if len(norm_log_k):
            print("\n  运动表标准化日志（前 5 行）：")
            print(norm_log_k.head().to_string(index=False))
    else:
        norm_log_m = pd.DataFrame()
        norm_log_k = pd.DataFrame()

    # 9. 保存
    os.makedirs(output_dir, exist_ok=True)

    morph_path = os.path.join(output_dir, "la_morphology_wide.csv")
    kine_path  = os.path.join(output_dir, "la_kinematic_wide.csv")
    qc_out_path = os.path.join(output_dir, "la_params_qc_filtered.csv")
    miss_m_path = os.path.join(output_dir, "la_params_missingness_morph.csv")
    miss_k_path = os.path.join(output_dir, "la_params_missingness_kine.csv")
    feat_path   = os.path.join(output_dir, "la_params_feature_decisions.csv")

    morph_wide.to_csv(morph_path, index=False)
    kine_wide.to_csv(kine_path,   index=False)
    df_qc.to_csv(qc_out_path,     index=False)
    miss_morph.to_csv(miss_m_path, index=False)
    miss_kine.to_csv(miss_k_path,  index=False)

    # 汇总特征决策
    feat_all = pd.concat([
        miss_morph.assign(table="morphology"),
        miss_kine.assign(table="kinematic"),
    ], ignore_index=True)
    feat_all.to_csv(feat_path, index=False)

    print(f"\n  输出文件：")
    for p in [morph_path, kine_path, qc_out_path, miss_m_path,
              miss_k_path, feat_path]:
        print(f"    {p}")

    return {
        "morphology_wide":   morph_wide,
        "kinematic_wide":    kine_wide,
        "qc_filtered":       df_qc,
        "missingness_morph": miss_morph,
        "missingness_kine":  miss_kine,
        "outlier_log_morph": outlier_morph,
        "outlier_log_kine":  outlier_kine,
    }


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--morphology",
        default="csv/la_params/la_morphology_results.csv",
        help="la_morphology_results.csv 路径",
    )
    ap.add_argument(
        "--kinematic",
        default="csv/la_params/la_kinematic_stats.csv",
        help="la_kinematic_stats.csv 路径",
    )
    ap.add_argument(
        "--qc",
        default="csv/la_params/la_analysis_qc.csv",
        help="la_analysis_qc.csv 路径",
    )
    ap.add_argument(
        "--output_dir",
        default="csv/la_params/processed",
        help="输出目录（默认：csv/la_params/processed）",
    )
    ap.add_argument(
        "--report_only",
        action="store_true",
        help="仅生成缺失率报告，不写输出文件",
    )
    ap.add_argument(
        "--drop_threshold",
        type=float,
        default=THRESH_DROP,
        help=f"列缺失率阈值：超过此值则剔除（默认：{THRESH_DROP}）",
    )
    ap.add_argument(
        "--no_normalise",
        action="store_true",
        help="跳过 Z-score 标准化步骤",
    )
    args = ap.parse_args()

    args.morphology  = _resolve_path(args.morphology)
    args.kinematic   = _resolve_path(args.kinematic)
    args.qc          = _resolve_path(args.qc)
    args.output_dir  = _resolve_path(args.output_dir)

    process_la_params(
        morphology_path=args.morphology,
        kinematic_path=args.kinematic,
        qc_path=args.qc,
        output_dir=args.output_dir,
        report_only=args.report_only,
        thresh_drop=args.drop_threshold,
        no_normalise=args.no_normalise,
    )

    print("\n完成。")


if __name__ == "__main__":
    main()
