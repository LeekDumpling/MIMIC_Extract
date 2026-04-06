# -*- coding: utf-8 -*-
"""
左心房参数清洗模块 — HFpEF 队列分析（步骤：LA 参数预处理）

功能
----
读取由 EchoGraphs 模块生成的三张左心房（LA）参数文件：
  - final_morphology_results.csv   形态学参数长表
  - final_kinematic_stats.csv      运动学参数长表（含 sub_item；充盈段/排空段分阶段统计）
  - final_qc.csv                   质量控制与元数据表

对这三张表进行：
  1. QC 过滤       — 依据 la_analysis_qc.csv 中的 status 字段，按可配置策略剔除
  2. 异常值检查（两层）：
       第一层（自动剔除）— 仅去除"几何/数学不可能"或"明显测量失败"的值，
         目标是不误删真实患者；超出范围的值置为 NaN，保留行。
         • 单列阈值：LA_AUTO_REMOVE_RANGES（在长表上逐行检查）
         • 跨列约束：LAVmin ≤ LAVmax 等，在宽表上执行（apply_cross_checks_wide）
       第二层（人工复核标记）— 不删除，仅在宽表中添加 {col}_review_flag 列；
         宽松范围主要抓分割错误、单位错误、追踪失败、相位错配。
         使用 LA_REVIEW_RANGES 定义范围（apply_review_flags_wide）。
       未收录参数 — 不进行自动过滤；改为计算 IQR 四分位异常值统计，
         输出 CSV 报告及可视化图表供人工参考（compute_iqr_outlier_stats）。
  3. 缺失值处理   — 参考 impute_normalize.py 策略；影像参数算法失败时保留 NaN，不填补
  4. Z-score 标准化 — 连续变量（偏态列先 log1p 变换）
  5. 长表 → 宽表 pivot
  6. 保存输出文件

输出（默认 csv/la_params/processed/）
  la_morphology_wide.csv              形态学参数宽表（主键：video_prefix + subject_id）
  la_kinematic_wide.csv               运动学参数宽表（主键：video_prefix + subject_id）
  la_params_qc_filtered.csv           经过 QC 过滤的 QC 表（供下游追溯）
  la_params_missingness_morph.csv     形态表缺失率报告
  la_params_missingness_kine.csv      运动表缺失率报告
  la_params_feature_decisions.csv     特征决策（保留/剔除/填补）
  la_params_cross_check_log.csv       跨列约束违规记录（第一层）
  la_params_review_log_morph.csv      形态表第二层复核标记汇总
  la_params_review_log_kine.csv       运动表第二层复核标记汇总
  la_params_iqr_outliers_morph.csv    形态表未收录参数 IQR 异常值统计
  la_params_iqr_outliers_kine.csv     运动表未收录参数 IQR 异常值统计
  la_params_iqr_outliers_morph.png    形态表 IQR 异常值可视化（需要 matplotlib）
  la_params_iqr_outliers_kine.png     运动表 IQR 异常值可视化（需要 matplotlib）
  la_params_review_priority_summary.csv  按复核标记总数排序的视频汇总（一行一视频）
  la_params_review_priority_detail.csv   每处异常的具体参数值（一行一处异常）

语义约定（务必遵守）
  - max_idx → LAVmax；min_idx → LAVmin
  - 所有 value 空字符串均表示"当前不可用"，绝不能按 0 处理
  - 速率类参数依赖真实 fps；BSA 参数依赖外部 clinical CSV
  - 当前参数均来自模型连续 28 点轮廓，非人工真值
  - 运动学时相定义：fill = LAVmin → next LAVmax；empty = LAVmax → LAVmin
    sub_item 格式为 fill_peak / fill_mean / empty_drop / empty_mean_drop /
    empty_trough / empty_mean / empty_min / empty_mean（视参数而定）；
    Time to peak LASrR 使用 from_LAVmin 子项
  - 勿将 sub_item 映射为旧 ED/ES 语义；勿依赖已废弃的整周期子项
    （peak / mean / range / min / peak_expansion 等）

用法（从仓库根目录或 utils/ 子目录均可运行）
  python utils/clean_la_params.py \\
      --morphology  path/to/final_morphology_results.csv \\
      --kinematic   path/to/final_kinematic_stats.csv \\
      --qc          path/to/final_qc.csv \\
      --output_dir  csv/la_params/processed

  # 仅生成缺失率报告，不写输出文件：
  python utils/clean_la_params.py --report_only

  # 跳过 IQR 图表生成（无 matplotlib 环境）：
  python utils/clean_la_params.py --no_plots
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
# 第一层：自动剔除阈值
# 仅去除"几何/数学不可能"或"明显测量失败"的值，目标是不误删真实患者。
# 超出范围的值置为 NaN（保留行，保留该视频其他参数）。
# ---------------------------------------------------------------------------
LA_AUTO_REMOVE_RANGES: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    # 形态学参数
    "LAVmax":                       (0.0,    None),   # > 0 mL
    "LAVI":                         (0.0,    None),   # > 0 mL/m²
    "LAVmin":                       (0.0,    None),   # > 0 mL（跨列约束见 apply_cross_checks_wide）
    "LAVmin-i":                     (0.0,    None),   # > 0 mL/m²
    "LAEF":                         (0.0,   100.0),   # 0–100 %
    "LAD-long":                     (0.0,    None),   # > 0 cm
    "LAD-trans":                    (0.0,    None),   # > 0 cm
    "3D LA sphericity":             (0.0,    None),   # > 0
    "LA ellipticity":               (0.0,     1.0),   # 0–1（短轴/长轴定义）
    "LA circularity":               (0.0,     1.0),   # 0–1
    "LA sphericity index":          (0.0,    None),   # > 0
    "LA eccentricity index":        (0.0,     1.0),   # 0–1
    "MAT area":                     (0.0,    None),   # > 0 cm²
    "TGAR":                         (0.0,     1.0),   # 0–1（三角形/整体面积）
    # 运动学参数
    # "LASr":                         (-100.0, 100.0),  # %
    # "LASrR":                        (-20.0,   20.0),  # s^-1
    "Time to peak LASrR":           (0.0,    100.0),  # %cycle
    # "LASct-proxy":                  (-100.0, 100.0),  # %
    # "GCS":                          (-100.0, 100.0),  # %
    # "GCSR":                         (-20.0,   20.0),  # s^-1
    # "LS":                           (-100.0, 100.0),  # %
    # "LSR":                          (-20.0,   20.0),  # s^-1
    # "AS":                           (-100.0, 100.0),  # %
    # "ASR":                          (-20.0,   20.0),  # s^-1
    # "4CH ellipticity rate":         (-20.0,   20.0),  # 无量纲
    # "4CH circularity rate":         (-20.0,   20.0),  # 无量纲
    # "Sphericity index rate":        (-20.0,   20.0),  # 无量纲
    # "Annular expansion rate":       (-20.0,   20.0),  # cm/s
    # "Longitudinal stretching rate": (-20.0,   20.0),  # cm/s
}

# ---------------------------------------------------------------------------
# 第二层：人工复核阈值
# 不自动删除，仅在宽表中添加 {col}_review_flag 标志列（0=正常，1=需复核）。
# 宽松范围主要抓分割错误、单位错误、追踪失败、相位错配等。
# ---------------------------------------------------------------------------
LA_REVIEW_RANGES: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    # 形态学参数
    "LAVmax":                       (5.0,   250.0),  # mL
    "LAVI":                         (3.0,   150.0),  # mL/m²
    "LAVmin":                       (3.0,   200.0),  # mL
    "LAVmin-i":                     (1.0,   120.0),  # mL/m²
    "LAEF":                         (5.0,    90.0),  # %
    "LAD-long":                     (2.0,     9.0),  # cm
    "LAD-trans":                    (1.0,     8.0),  # cm
    "3D LA sphericity":             (0.1,     2.0),  # 无量纲
    "LA ellipticity":               (0.2,     1.0),  # 无量纲
    "LA circularity":               (0.2,     1.0),  # 无量纲
    "LA sphericity index":          (0.2,     1.5),  # 无量纲
    "LA eccentricity index":        (0.0,     1.0),  # 无量纲
    "MAT area":                     (0.5,    30.0),  # cm²
    "TGAR":                         (0.05,    1.0),  # 无量纲
    # 运动学参数
    # "LASr":                         (-20.0,  80.0),  # %
    # "LASrR":                        (-8.0,    8.0),  # s^-1
    "Time to peak LASrR":           (0.0,   100.0),  # %cycle
    # "LASct-proxy":                  (-80.0,  20.0),  # %
    # "GCS":                          (-80.0,  20.0),  # %
    # "GCSR":                         (-8.0,    8.0),  # s^-1
    # "LS":                           (-80.0,  20.0),  # %
    # "LSR":                          (-8.0,    8.0),  # s^-1
    # "AS":                           (-80.0,  20.0),  # %
    # "ASR":                          (-8.0,    8.0),  # s^-1
    "4CH ellipticity rate":         (-8.0,    8.0),  # 无量纲
    "4CH circularity rate":         (-8.0,    8.0),  # 无量纲
    "Sphericity index rate":        (-8.0,    8.0),  # 无量纲
    "Annular expansion rate":       (-8.0,    8.0),  # cm/s
    "Longitudinal stretching rate": (-8.0,    8.0),  # cm/s
}

# 跨列一致性约束（第一层补充，在宽表上执行）
# 格式：(大值参数名, 小值参数名)，若 小值列 > 大值列 则将 小值列 置为 NaN
LA_CROSS_CHECKS: List[Tuple[str, str]] = [
    ("LAVmax",   "LAVmin"),     # LAVmin ≤ LAVmax
    ("LAVI",     "LAVmin-i"),   # LAVmin-i ≤ LAVI
    ("LAD-long", "LAD-trans"),  # LAD-trans ≤ LAD-long（垂直短轴定义下）
]

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
# 运动学参数 schema（充盈段/排空段分阶段统计，final_kinematic_stats.csv）
# 时相定义：fill = LAVmin → next LAVmax；empty = LAVmax → LAVmin
# ---------------------------------------------------------------------------

# 当前版本的合法 sub_item 集合（用于加载时检测废弃子项）
KINE_KNOWN_SCHEMA: Dict[str, List[str]] = {
    "LASr":                        ["fill_peak"],
    "LASrR":                       ["fill_peak", "fill_mean"],
    "Time to peak LASrR":          ["from_LAVmin"],
    "LASct-proxy":                 ["empty_drop"],
    "GCS":                         ["fill_peak", "fill_mean", "empty_drop", "empty_mean_drop"],
    "LS":                          ["fill_peak", "fill_mean", "empty_drop", "empty_mean_drop"],
    "AS":                          ["fill_peak", "fill_mean", "empty_drop", "empty_mean_drop"],
    "GCSR":                        ["fill_peak", "fill_mean", "empty_trough", "empty_mean"],
    "LSR":                         ["fill_peak", "fill_mean", "empty_trough", "empty_mean"],
    "ASR":                         ["fill_peak", "fill_mean", "empty_trough", "empty_mean"],
    "4CH ellipticity rate":        ["fill_peak", "fill_mean", "empty_trough", "empty_mean"],
    "4CH circularity rate":        ["fill_peak", "fill_mean", "empty_trough", "empty_mean"],
    "Sphericity index rate":       ["fill_peak", "fill_mean", "empty_trough", "empty_mean"],
    "MAT area":                    ["fill_peak", "fill_mean", "empty_min",    "empty_mean"],
    "TGAR":                        ["fill_peak", "fill_mean", "empty_min",    "empty_mean"],
    "Annular expansion rate":      ["fill_peak", "fill_mean", "empty_trough", "empty_mean"],
    "Longitudinal stretching rate":["fill_peak", "fill_mean", "empty_trough", "empty_mean"],
}

# 已废弃的整周期 sub_item 值（若在输入文件中检测到则发出警告）
_DEPRECATED_KINE_SUB_ITEMS: frozenset = frozenset({
    "peak", "mean", "range", "min",
    "peak_expansion", "peak_contraction",
    "peak_stretch", "peak_compression",
})


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

    df_morph = _load(morphology_path, required_morph, "final_morphology_results")
    df_kine  = _load(kinematic_path,  required_kine,  "final_kinematic_stats")
    df_qc    = _load(qc_path,         required_qc,    "final_qc")

    # 检测废弃的整周期 sub_item（若发现说明文件版本与预期不符）
    found_deprecated = (
        set(df_kine["sub_item"].dropna().astype(str).unique())
        & _DEPRECATED_KINE_SUB_ITEMS
    )
    if found_deprecated:
        warnings.warn(
            f"final_kinematic_stats 中检测到已废弃的整周期 sub_item：{sorted(found_deprecated)}\n"
            "当前版本仅支持分阶段子项（fill_peak / fill_mean / empty_drop 等）。\n"
            "请确认输入文件是否为旧版本格式。",
            UserWarning,
        )

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
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = LA_AUTO_REMOVE_RANGES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    第一层自动剔除：对 value 列按 parameter_name 施加单列绝对阈值约束。

    超出范围的值被置为 NaN（不直接剔除行，以保留该视频其他参数）。
    未收录在 ranges 中的参数名直接跳过（不过滤），留给 IQR 统计处理。

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
        print(f"  [第一层自动剔除] 置 NaN 共 {len(outlier_log)} 个超范围值")
    else:
        print("  [第一层自动剔除] 未发现超范围值")
    return df, outlier_log


# ---------------------------------------------------------------------------
# 步骤 4b：跨列一致性约束（第一层补充，在宽表上执行）
# ---------------------------------------------------------------------------

def apply_cross_checks_wide(
    df: pd.DataFrame,
    cross_checks: List[Tuple[str, str]] = LA_CROSS_CHECKS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    在宽表上执行跨列一致性约束（第一层自动修正补充）。

    对每对 (bigger_param, smaller_param)，若 smaller_col > bigger_col，
    则将 smaller_col 置为 NaN（保留 bigger_col 及其他列）。

    返回
    ----
    df_checked, cross_check_log
    """
    df = df.copy()
    log_rows = []

    for big_param, small_param in cross_checks:
        big_col   = _safe_col_name(big_param)
        small_col = _safe_col_name(small_param)

        if big_col not in df.columns or small_col not in df.columns:
            continue

        both_valid = df[big_col].notna() & df[small_col].notna()
        violation  = both_valid & (df[small_col] > df[big_col])
        n_viol = int(violation.sum())

        if n_viol > 0:
            prefix_col = "video_prefix" if "video_prefix" in df.columns else None
            for idx in df.index[violation]:
                log_rows.append({
                    "video_prefix": df.at[idx, prefix_col] if prefix_col else idx,
                    "constraint":   f"{small_param} ≤ {big_param}",
                    "small_value":  float(df.at[idx, small_col]),
                    "big_value":    float(df.at[idx, big_col]),
                    "action":       "set_nan_small_col",
                })
            df.loc[violation, small_col] = np.nan
            print(f"  [跨列约束] {small_param} ≤ {big_param}："
                  f"修正 {n_viol} 个违规值 → NaN")

    cross_log = pd.DataFrame(log_rows)
    if cross_log.empty:
        print("  [跨列约束] 未发现跨列违规值")
    return df, cross_log


# ---------------------------------------------------------------------------
# 步骤 4c：第二层人工复核标记（在宽表上执行）
# ---------------------------------------------------------------------------

def apply_review_flags_wide(
    df: pd.DataFrame,
    skip_cols: List[str],
    review_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = LA_REVIEW_RANGES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    第二层：在宽表中为超出复核范围的值添加 {col}_review_flag 列（只标记，不删除）。

    flag = 0  → 值在复核范围内
    flag = 1  → 值超出复核范围，建议人工检查
    flag = NaN → 对应原值为 NaN，无法判断

    运动学列格式为 safe_param__sub_item，按双下划线前部分匹配复核范围。

    返回
    ----
    df_flagged, review_log
    """
    df = df.copy()
    log_rows = []
    skip_set = set(skip_cols)

    # safe_param_name → (lo, hi)
    safe_review: Dict[str, Tuple[Optional[float], Optional[float]]] = {
        _safe_col_name(k): v for k, v in review_ranges.items()
    }

    for col in list(df.columns):
        if col in skip_set:
            continue
        if col.endswith("_review_flag") or col.endswith("_missing_flag"):
            continue
        if df[col].dtype.kind not in ("f", "i", "u"):
            continue

        base = col.split("__")[0] if "__" in col else col
        if base not in safe_review:
            continue

        lo, hi = safe_review[base]
        flag_col = f"{col}_review_flag"

        valid_mask = df[col].notna()
        vals = df.loc[valid_mask, col]
        is_outside = pd.Series(False, index=vals.index)
        if lo is not None:
            is_outside = is_outside | (vals < lo)
        if hi is not None:
            is_outside = is_outside | (vals > hi)

        # Build nullable Int8 flag: True→1, False→0, NaN rows→pd.NA
        df[flag_col] = is_outside.astype("Int8").reindex(df.index)

        n_flagged = int(is_outside.sum())
        if n_flagged > 0:
            log_rows.append({
                "column":      col,
                "review_lo":   lo,
                "review_hi":   hi,
                "n_flagged":   n_flagged,
                "pct_flagged": round(n_flagged / max(int(valid_mask.sum()), 1) * 100, 2),
            })

    if log_rows:
        n_cols = len(log_rows)
        total  = sum(r["n_flagged"] for r in log_rows)
        print(f"  [第二层复核] {n_cols} 个参数列共 {total} 处超复核范围（已添加 _review_flag 列）")
    else:
        print("  [第二层复核] 所有参数均在复核范围内，未添加 _review_flag 列")

    return df, pd.DataFrame(log_rows)


# ---------------------------------------------------------------------------
# 步骤 4d：IQR 异常值统计（未收录参数）
# ---------------------------------------------------------------------------

def compute_iqr_outlier_stats(
    df: pd.DataFrame,
    skip_cols: List[str],
    defined_params: Optional[set] = None,
) -> pd.DataFrame:
    """
    对宽表中"未收录在预定义范围内"的参数列计算 IQR 四分位异常值统计。

    异常值定义（Tukey 栅栏）：
      < Q1 − 1.5 × IQR  或  > Q3 + 1.5 × IQR

    参数
    ----
    df             : 宽表（标准化之前）
    skip_cols      : 不参与统计的列（id 列等）
    defined_params : 已定义自动剔除 / 复核范围的参数 safe_name 集合；
                     在此集合中的列跳过 IQR 统计。
                     若为 None 则对所有合格数值列计算。

    返回
    ----
    DataFrame（按 pct_outlier 降序），每行对应一个参数列。
    """
    if defined_params is None:
        defined_params = set()

    rows = []
    skip_set = set(skip_cols)

    for col in df.columns:
        if col in skip_set:
            continue
        if col.endswith("_review_flag") or col.endswith("_missing_flag"):
            continue
        if df[col].dtype.kind not in ("f", "i", "u"):
            continue
        if _is_binary_col(df[col]):
            continue

        base = col.split("__")[0] if "__" in col else col
        if base in defined_params:
            continue  # 已有预定义范围，跳过

        vals = df[col].dropna()
        if len(vals) < 4:
            continue

        q1  = float(vals.quantile(0.25))
        q3  = float(vals.quantile(0.75))
        iqr = q3 - q1
        lo_bound = q1 - 1.5 * iqr
        hi_bound = q3 + 1.5 * iqr
        n_lo = int((vals < lo_bound).sum())
        n_hi = int((vals > hi_bound).sum())

        rows.append({
            "column":          col,
            "n":               len(vals),
            "mean":            round(float(vals.mean()),    4),
            "std":             round(float(vals.std()),     4),
            "min":             round(float(vals.min()),     4),
            "Q1":              round(q1,                    4),
            "median":          round(float(vals.median()),  4),
            "Q3":              round(q3,                    4),
            "max":             round(float(vals.max()),     4),
            "IQR":             round(iqr,                   4),
            "iqr_lo_bound":    round(lo_bound,              4),
            "iqr_hi_bound":    round(hi_bound,              4),
            "n_outlier_lo":    n_lo,
            "n_outlier_hi":    n_hi,
            "n_outlier_total": n_lo + n_hi,
            "pct_outlier":     round((n_lo + n_hi) / len(vals) * 100, 2),
        })

    if not rows:
        print("  [IQR 统计] 无未收录参数列（或所有列均有预定义范围）")
        return pd.DataFrame()

    result = (
        pd.DataFrame(rows)
        .sort_values("pct_outlier", ascending=False)
        .reset_index(drop=True)
    )
    print(f"  [IQR 统计] {len(result)} 个未收录参数列，"
          f"最高异常值比例 {result['pct_outlier'].iloc[0]:.1f}%")
    return result


def _setup_cjk_font() -> None:
    """
    配置 matplotlib 以正确渲染中文字符。

    按优先级尝试常见 CJK 字体（Windows / macOS / Linux），
    若均未找到则发出警告并使用系统默认字体（中文可能显示为方块）。
    同时禁用 unicode_minus 以避免负号乱码。
    """
    import matplotlib
    import matplotlib.font_manager as fm

    cjk_candidates = [
        "SimHei", "Microsoft YaHei", "SimSun", "FangSong", "NSimSun",  # Windows
        "STHeiti", "PingFang SC", "Heiti SC", "Hiragino Sans GB",        # macOS
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",                      # Linux
        "Noto Sans CJK SC", "Source Han Sans CN", "Droid Sans Fallback", # 通用
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in cjk_candidates if f in available), None)

    if chosen:
        matplotlib.rcParams["font.family"] = [chosen, "DejaVu Sans"]
    else:
        warnings.warn(
            "未找到 CJK 字体，图表中文可能显示为方块。\n"
            "Windows 用户请确认已安装 SimHei 或 Microsoft YaHei；\n"
            "Linux 用户请安装：apt-get install fonts-wqy-microhei\n"
            "或执行：pip install fonttools && fc-cache -fv",
            UserWarning,
        )
    matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 步骤 4e：复核优先级汇总（按视频聚合 review_flag）
# ---------------------------------------------------------------------------

def build_review_priority_table(
    morph_wide: pd.DataFrame,
    kine_wide: pd.DataFrame,
    id_cols: List[str],
    review_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = LA_REVIEW_RANGES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于宽表中的 *_review_flag 列，按视频聚合复核优先级。

    返回
    ----
    summary_df : 每视频一行，按 n_flags_total 降序排列
        列：id_cols + n_flags_morphology + n_flags_kinematic + n_flags_total
               + flagged_params（逗号分隔的超范围参数名列表）
    detail_df  : 每处异常一行（长格式）
        列：id_cols + table + param + value + review_lo + review_hi
    """
    safe_review: Dict[str, Tuple[Optional[float], Optional[float]]] = {
        _safe_col_name(k): v for k, v in review_ranges.items()
    }

    def _process_table(
        df: pd.DataFrame, table_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        id_present = [c for c in id_cols if c in df.columns]
        flag_cols = [c for c in df.columns if c.endswith("_review_flag")]

        # Per-video flag counts: use == 1 with fillna to handle nullable Int8
        flag_vals = df[flag_cols].apply(lambda s: (s == 1).fillna(False).astype(int), axis=0)
        per_video = df[id_present].copy()
        per_video[f"n_flags_{table_name}"] = flag_vals.sum(axis=1).values

        # Flagged parameter list per video
        def _flagged_list(row: "pd.Series") -> str:
            names = [
                fc[: -len("_review_flag")]
                for fc in flag_cols
                if row[fc] is True or row[fc] == 1
            ]
            return ", ".join(names)

        per_video["flagged_params"] = flag_vals.apply(_flagged_list, axis=1)

        # Detail rows: flag_vals already has int 0/1, so use it for masking
        detail_rows = []
        for fc in flag_cols:
            param = fc[: -len("_review_flag")]
            base = param.split("__")[0] if "__" in param else param
            lo, hi = safe_review.get(base, (None, None))
            flagged_mask = flag_vals[fc] == 1
            for _, row in df[flagged_mask].iterrows():
                detail_rows.append(
                    {
                        **{c: row[c] for c in id_present},
                        "table":     table_name,
                        "param":     param,
                        "value":     row.get(param, np.nan),
                        "review_lo": lo,
                        "review_hi": hi,
                    }
                )

        return per_video, pd.DataFrame(detail_rows)

    summary_m, detail_m = _process_table(morph_wide, "morphology")
    summary_k, detail_k = _process_table(kine_wide,  "kinematic")

    merge_keys = [c for c in ["video_prefix", "subject_id"]
                  if c in summary_m.columns and c in summary_k.columns]
    extra_m = [c for c in id_cols if c in summary_m.columns and c not in merge_keys]
    extra_k = [c for c in id_cols if c in summary_k.columns and c not in merge_keys]

    left  = summary_m[merge_keys + extra_m + ["n_flags_morphology", "flagged_params"]]
    right = summary_k[merge_keys + extra_k + ["n_flags_kinematic",  "flagged_params"]]

    summary = left.merge(right, on=merge_keys, how="outer", suffixes=("_morph", "_kine"))
    summary["n_flags_morphology"] = summary["n_flags_morphology"].fillna(0).astype(int)
    summary["n_flags_kinematic"]  = summary["n_flags_kinematic"].fillna(0).astype(int)
    summary["n_flags_total"]      = (
        summary["n_flags_morphology"] + summary["n_flags_kinematic"]
    )
    # Merge flagged_params columns
    fp_m = summary.pop("flagged_params_morph").fillna("")
    fp_k = summary.pop("flagged_params_kine").fillna("")
    summary["flagged_params"] = (
        (fp_m + ", " + fp_k)
        .str.strip(", ")
        .str.replace(r",\s*,", ",", regex=True)
        .str.strip(", ")
    )
    summary = summary.sort_values("n_flags_total", ascending=False).reset_index(drop=True)

    detail = pd.concat([detail_m, detail_k], ignore_index=True)

    n_flagged_videos = int((summary["n_flags_total"] > 0).sum())
    max_flags = int(summary["n_flags_total"].iloc[0]) if len(summary) else 0
    print(
        f"  [复核优先级] {n_flagged_videos} 个视频存在复核标记，"
        f"单视频最多 {max_flags} 处"
    )
    return summary, detail


def plot_iqr_outliers(
    iqr_stats: pd.DataFrame,
    output_path: str,
    title: str = "IQR 异常值统计（未收录参数）",
) -> Optional[str]:
    """
    生成 IQR 异常值汇总水平条形图并保存为 PNG。

    需要 matplotlib；若未安装则发出警告并返回 None。

    参数
    ----
    iqr_stats   : compute_iqr_outlier_stats 返回的 DataFrame
    output_path : PNG 输出路径（目录不存在时自动创建）
    title       : 图表标题

    返回
    ----
    output_path（成功）或 None（matplotlib 不可用 / 无数据）
    """
    if iqr_stats is None or len(iqr_stats) == 0:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "matplotlib 未安装，跳过 IQR 图表生成。\n安装命令：pip install matplotlib",
            UserWarning,
        )
        return None

    _setup_cjk_font()

    df = iqr_stats.copy()
    n_params = len(df)
    fig_height = max(4.0, n_params * 0.45)

    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = ["#d62728" if p > 10 else "#1f77b4" for p in df["pct_outlier"]]
    ax.barh(df["column"], df["pct_outlier"], color=colors)
    ax.axvline(x=5, color="orange", linestyle="--", linewidth=0.8, label="5% 参考线")
    ax.set_xlabel("IQR 异常值比例 (%)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["pct_outlier"] + 0.2,
            i,
            f'{int(row["n_outlier_total"])} / {int(row["n"])}',
            va="center",
            fontsize=7,
        )

    plt.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [IQR 图表] 已保存：{output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 步骤 5：长表 → 宽表 pivot
# ---------------------------------------------------------------------------

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

    sub_item 格式（充盈段/排空段分阶段，final_kinematic_stats.csv 新 schema）：
      fill_peak / fill_mean / empty_drop / empty_mean_drop /
      empty_trough / empty_mean / empty_min / from_LAVmin
    示例列名：LASr__fill_peak, GCS__empty_drop, Time_to_peak_LASrR__from_LAVmin
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
    no_norm |= {c for c in df.columns if c.endswith(("_missing_flag", "_review_flag"))}

    for col in df.columns:
        if col in no_norm:
            continue
        if df[col].dtype.kind not in ("f", "i", "u"):
            continue
        if _is_binary_col(df[col]):
            continue

        vals = df[col].dropna().to_numpy(dtype=float)
        if len(vals) < 2:
            continue
        skew = float(pd.Series(vals).skew())

        col_vals = df[col].to_numpy(dtype=float, na_value=np.nan).copy()
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
    no_plots: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    完整 LA 参数清洗流程（两层异常值过滤 + IQR 统计）。

    参数
    ----
    morphology_path   : la_morphology_results.csv 路径
    kinematic_path    : la_kinematic_stats.csv 路径
    qc_path           : la_analysis_qc.csv 路径
    output_dir        : 输出目录
    report_only       : 仅生成报告，不写输出文件
    hard_filter_codes : 硬过滤状态码集合（默认使用 QC_HARD_FILTER_CODES）
    thresh_drop       : 列缺失率超过此值则剔除该列（默认 0.30）
    no_normalise      : True → 跳过 Z-score 标准化
    no_plots          : True → 跳过 IQR 图表生成（无 matplotlib 环境时使用）

    返回
    ----
    dict with keys:
      "morphology_wide", "kinematic_wide", "qc_filtered",
      "missingness_morph", "missingness_kine",
      "outlier_log_morph", "outlier_log_kine",
      "cross_check_log",
      "review_log_morph", "review_log_kine",
      "iqr_stats_morph",  "iqr_stats_kine"
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

    # 4. 第一层自动剔除：单列绝对阈值（长表）
    df_morph, outlier_morph = apply_physio_range_check(df_morph)
    df_kine,  outlier_kine  = apply_physio_range_check(df_kine)

    # 5. 宽表 pivot
    morph_wide = pivot_morphology(df_morph)
    kine_wide  = pivot_kinematic(df_kine)
    print(f"  [pivot] 形态宽表 {morph_wide.shape}，运动宽表 {kine_wide.shape}")

    # 5b. 第一层补充：跨列一致性约束（宽表，仅形态表）
    print("\n  --- 跨列一致性约束 ---")
    morph_wide, cross_check_log = apply_cross_checks_wide(morph_wide)

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
            "morphology_wide":   morph_wide,
            "kinematic_wide":    kine_wide,
            "qc_filtered":       df_qc,
            "missingness_morph": miss_morph,
            "missingness_kine":  miss_kine,
            "outlier_log_morph": outlier_morph,
            "outlier_log_kine":  outlier_kine,
            "cross_check_log":   cross_check_log,
            "review_log_morph":  pd.DataFrame(),
            "review_log_kine":   pd.DataFrame(),
            "iqr_stats_morph":   pd.DataFrame(),
            "iqr_stats_kine":    pd.DataFrame(),
        }

    # 7. 缺失值处理（影像参数不填补，仅剔除超阈值列）
    morph_wide, dropped_m, flags_m = impute_wide(morph_wide, miss_morph,
                                                  drop_imaging_failures=True)
    kine_wide,  dropped_k, flags_k = impute_wide(kine_wide,  miss_kine,
                                                  drop_imaging_failures=True)

    print(f"\n  形态表：剔除列 {dropped_m or 'none'}，新增 flag 列 {flags_m or 'none'}")
    print(f"  运动表：剔除列 {dropped_k or 'none'}，新增 flag 列 {flags_k or 'none'}")

    # 7b. 第二层：人工复核标记（宽表，在标准化之前）
    print("\n  --- 第二层人工复核标记（形态表）---")
    morph_wide, review_log_morph = apply_review_flags_wide(morph_wide, skip_cols=id_cols)
    print("\n  --- 第二层人工复核标记（运动表）---")
    kine_wide,  review_log_kine  = apply_review_flags_wide(kine_wide,  skip_cols=id_cols)

    # 7c. IQR 异常值统计（未收录参数，在标准化之前）
    safe_defined = {
        _safe_col_name(k)
        for k in set(LA_AUTO_REMOVE_RANGES) | set(LA_REVIEW_RANGES)
    }
    print("\n  --- IQR 异常值统计（未收录参数，形态表）---")
    iqr_stats_morph = compute_iqr_outlier_stats(
        morph_wide, skip_cols=id_cols, defined_params=safe_defined
    )
    print("\n  --- IQR 异常值统计（未收录参数，运动表）---")
    iqr_stats_kine = compute_iqr_outlier_stats(
        kine_wide, skip_cols=id_cols, defined_params=safe_defined
    )

    # 7d. 复核优先级汇总（按视频聚合 review_flag，在标准化之前）
    print("\n  --- 复核优先级汇总 ---")
    review_priority_summary, review_priority_detail = build_review_priority_table(
        morph_wide, kine_wide, id_cols
    )

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

    morph_path        = os.path.join(output_dir, "la_morphology_wide.csv")
    kine_path         = os.path.join(output_dir, "la_kinematic_wide.csv")
    qc_out_path       = os.path.join(output_dir, "la_params_qc_filtered.csv")
    miss_m_path       = os.path.join(output_dir, "la_params_missingness_morph.csv")
    miss_k_path       = os.path.join(output_dir, "la_params_missingness_kine.csv")
    feat_path         = os.path.join(output_dir, "la_params_feature_decisions.csv")
    cross_log_path    = os.path.join(output_dir, "la_params_cross_check_log.csv")
    review_m_path     = os.path.join(output_dir, "la_params_review_log_morph.csv")
    review_k_path     = os.path.join(output_dir, "la_params_review_log_kine.csv")
    iqr_m_csv_path    = os.path.join(output_dir, "la_params_iqr_outliers_morph.csv")
    iqr_k_csv_path    = os.path.join(output_dir, "la_params_iqr_outliers_kine.csv")
    iqr_m_png_path    = os.path.join(output_dir, "la_params_iqr_outliers_morph.png")
    iqr_k_png_path    = os.path.join(output_dir, "la_params_iqr_outliers_kine.png")
    priority_sum_path = os.path.join(output_dir, "la_params_review_priority_summary.csv")
    priority_det_path = os.path.join(output_dir, "la_params_review_priority_detail.csv")

    morph_wide.to_csv(morph_path,  index=False)
    kine_wide.to_csv(kine_path,    index=False)
    df_qc.to_csv(qc_out_path,      index=False)
    miss_morph.to_csv(miss_m_path, index=False)
    miss_kine.to_csv(miss_k_path,  index=False)
    cross_check_log.to_csv(cross_log_path, index=False)
    review_log_morph.to_csv(review_m_path, index=False)
    review_log_kine.to_csv(review_k_path,  index=False)
    iqr_stats_morph.to_csv(iqr_m_csv_path, index=False)
    iqr_stats_kine.to_csv(iqr_k_csv_path,  index=False)
    review_priority_summary.to_csv(priority_sum_path, index=False)
    review_priority_detail.to_csv(priority_det_path,  index=False)

    # 汇总特征决策
    feat_all = pd.concat([
        miss_morph.assign(table="morphology"),
        miss_kine.assign(table="kinematic"),
    ], ignore_index=True)
    feat_all.to_csv(feat_path, index=False)

    # IQR 图表（需要 matplotlib）
    if not no_plots:
        plot_iqr_outliers(
            iqr_stats_morph, iqr_m_png_path,
            title="IQR 异常值统计 — 形态表未收录参数",
        )
        plot_iqr_outliers(
            iqr_stats_kine, iqr_k_png_path,
            title="IQR 异常值统计 — 运动表未收录参数",
        )

    # 打印 Top-10 最需复核视频
    top10 = review_priority_summary[review_priority_summary["n_flags_total"] > 0].head(10)
    if len(top10):
        print("\n  --- Top-10 最需复核视频（按总标记数降序）---")
        display_cols = [c for c in ["video_prefix", "subject_id", "n_flags_morphology",
                                     "n_flags_kinematic", "n_flags_total", "flagged_params"]
                        if c in top10.columns]
        print(top10[display_cols].to_string(index=False))

    saved_files = [
        morph_path, kine_path, qc_out_path,
        miss_m_path, miss_k_path, feat_path,
        cross_log_path, review_m_path, review_k_path,
        iqr_m_csv_path, iqr_k_csv_path,
        priority_sum_path, priority_det_path,
    ]
    print(f"\n  输出文件：")
    for p in saved_files:
        print(f"    {p}")

    return {
        "morphology_wide":          morph_wide,
        "kinematic_wide":           kine_wide,
        "qc_filtered":              df_qc,
        "missingness_morph":        miss_morph,
        "missingness_kine":         miss_kine,
        "outlier_log_morph":        outlier_morph,
        "outlier_log_kine":         outlier_kine,
        "cross_check_log":          cross_check_log,
        "review_log_morph":         review_log_morph,
        "review_log_kine":          review_log_kine,
        "iqr_stats_morph":          iqr_stats_morph,
        "iqr_stats_kine":           iqr_stats_kine,
        "review_priority_summary":  review_priority_summary,
        "review_priority_detail":   review_priority_detail,
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
        default="csv/la_params/final_morphology_results.csv",
        help="final_morphology_results.csv 路径",
    )
    ap.add_argument(
        "--kinematic",
        default="csv/la_params/final_kinematic_stats.csv",
        help="final_kinematic_stats.csv 路径（充盈段/排空段分阶段 schema）",
    )
    ap.add_argument(
        "--qc",
        default="csv/la_params/final_qc.csv",
        help="final_qc.csv 路径",
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
    ap.add_argument(
        "--no_plots",
        action="store_true",
        help="跳过 IQR 图表生成（无 matplotlib 环境时使用）",
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
        no_plots=args.no_plots,
    )

    print("\n完成。")


if __name__ == "__main__":
    main()
