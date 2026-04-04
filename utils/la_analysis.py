# -*- coding: utf-8 -*-
"""
左心房参数分析模块 — HFpEF 队列 MIMIC 临床数据 × LA 影像参数联合分析

功能
----
将经过清洗的 MIMIC 临床宽表（来自 impute_normalize.py 输出）与
LA 影像参数宽表（来自 clean_la_params.py 输出）合并，执行：

  1. 合并         — 按 subject_id（+ study_id 可选）关联临床表与影像表
  2. 描述性统计   — 连续变量均值±SD 或中位数[IQR]；分类变量频次/%
  3. 相关性分析   — Pearson（近正态）/ Spearman（偏态或离群）
                    输出相关矩阵 CSV + 热图 PNG
  4. FDR 校正     — 多重检验 p 值采用 Benjamini-Hochberg 方法校正
  5. 共线性检验   — VIF（statsmodels）；VIF > 10 标记为高共线性
  6. 组间比较     — 连续变量：t 检验 / Mann-Whitney U / ANOVA / Kruskal-Wallis
                    分类变量：χ² / Fisher 精确检验
  7. 输出汇总报告 CSV 及可视化图形

输出（默认 csv/la_analysis/）
  merged_dataset.csv              合并后数据集
  descriptive_stats.csv           描述性统计
  correlation_la_clinical.csv     LA参数 × 临床变量相关矩阵
  correlation_la_clinical.png     相关热图
  vif_report.csv                  VIF 共线性报告
  group_comparison.csv            组间比较结果（含 FDR 校正 p 值）

语义约定
  - 参数名遵循 clean_la_params.py 的安全化规则（空格→_ / %→pct / s^-1→s_inv）
  - max_idx → LAVmax；min_idx → LAVmin（绝不按 ED/ES 假设映射）
  - 所有空值均表示"不可用"，绝不按 0 处理

用法（从仓库根目录或 utils/ 子目录均可运行）
  python utils/la_analysis.py \\
      --clinical_csv  csv/processed/hfpef_cohort_win_hadm_processed.csv \\
      --morph_csv     csv/la_params/processed/la_morphology_wide.csv \\
      --kine_csv      csv/la_params/processed/la_kinematic_wide.csv \\
      --qc_csv        csv/la_params/processed/la_params_qc_filtered.csv \\
      --output_dir    csv/la_analysis \\
      --group_col     died_inhosp

  # 指定多个组别对比列：
  python utils/la_analysis.py --group_col died_inhosp os_event
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

import json

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    warnings.warn("scipy 未安装；统计检验将不可用。\npip install scipy", UserWarning)

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.multitest import multipletests
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False
    warnings.warn(
        "statsmodels 未安装；VIF 及 FDR 校正将不可用。\npip install statsmodels",
        UserWarning,
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

# ---------------------------------------------------------------------------
# 列分类常量
# ---------------------------------------------------------------------------

# 临床表中的 ID / 时间 / 结局列（不参与分析，但用于分组）
_CLINICAL_ID_COLS = {
    "subject_id", "hadm_id", "index_study_id", "index_study_datetime",
    "index_admittime", "index_dischtime", "death_date", "last_dischtime",
    "censor_date", "a4c_dicom_filepath",
}
_OUTCOME_COLS = {
    "hospital_expire_flag", "died_inhosp", "died_post_dc",
    "days_survived_post_dc", "died_30d", "died_90d", "died_1yr",
    "os_event", "os_days",
}

# LA 宽表中的元数据列（不参与参数分析）
_LA_META_COLS = {"video_prefix", "subject_id", "study_id", "source_group",
                 "keyframe_method"}

# 正态性判断：|偏度| ≤ 此阈值 → 近正态
_SKEW_THRESH = 1.0

# VIF 阈值：超过此值视为高共线性
_VIF_THRESH = 10.0

# 统计显著性水平
_ALPHA = 0.05


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


def _is_binary_col(series: pd.Series) -> bool:
    uniq = set(series.dropna().unique())
    return uniq.issubset({0, 1, 0.0, 1.0, True, False})


def _is_likely_normal(series: pd.Series) -> bool:
    """判断序列是否近似正态（|偏度| ≤ _SKEW_THRESH）。"""
    clean = series.dropna()
    if len(clean) < 3:
        return False
    return abs(float(clean.skew())) <= _SKEW_THRESH


def _safe_col_name(name: str) -> str:
    name = str(name)
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace("%", "pct")
    name = name.replace("s^-1", "s_inv")
    return name


# ---------------------------------------------------------------------------
# 步骤 1：合并数据集
# ---------------------------------------------------------------------------

def merge_datasets(
    df_clinical: pd.DataFrame,
    df_morph: pd.DataFrame,
    df_kine: pd.DataFrame,
    df_qc: Optional[pd.DataFrame] = None,
    join_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    将临床表与 LA 形态宽表、运动宽表按 subject_id（可含 study_id）合并。

    合并策略：inner join（仅保留同时存在于临床表和影像表的患者）。
    若 df_qc 提供，附加 QC 元数据列（spacing_found, fps, valid_cycle_count,
    keyframe_method, dense_kpts_source, gt_kpts_available）。

    参数
    ----
    join_keys : 默认 ["subject_id"]；若含 study_id 则 ["subject_id", "study_id"]
    """
    if join_keys is None:
        join_keys = ["subject_id"]

    # 确保 join_keys 存在
    for key in join_keys:
        for tbl, name in [(df_clinical, "clinical"), (df_morph, "morphology"),
                          (df_kine, "kinematic")]:
            if key not in tbl.columns:
                raise ValueError(f"合并键 '{key}' 不存在于 {name} 表中")

    # 合并形态 + 运动（按 video_prefix + join_keys）
    morph_keys = ["video_prefix"] + [k for k in join_keys if k in df_morph.columns]
    kine_keys  = ["video_prefix"] + [k for k in join_keys if k in df_kine.columns]

    # 形态表与运动表按 video_prefix 合并，运动表列若与形态表重名则加 _kine 后缀
    morph_cols = set(df_morph.columns) - set(morph_keys)
    kine_cols  = set(df_kine.columns)  - set(kine_keys)
    overlap    = morph_cols & kine_cols
    if overlap:
        df_kine = df_kine.rename(columns={c: c + "_kine" for c in overlap})

    df_img = pd.merge(df_morph, df_kine, on=morph_keys, how="outer",
                      suffixes=("", "_kine"))

    # 影像表与临床表合并
    df_merged = pd.merge(df_clinical, df_img, on=join_keys, how="inner")

    # 可选：附加 QC 元数据
    if df_qc is not None:
        qc_extra_cols = [c for c in [
            "video_prefix", "spacing_found", "clinical_found",
            "fps", "valid_cycle_count", "keyframe_method",
            "dense_kpts_source", "gt_kpts_available",
        ] if c in df_qc.columns]
        if "video_prefix" in df_qc.columns and "video_prefix" in df_merged.columns:
            df_qc_sub = df_qc[qc_extra_cols].drop_duplicates(subset=["video_prefix"])
            df_merged = pd.merge(df_merged, df_qc_sub, on="video_prefix",
                                 how="left", suffixes=("", "_qc"))

    print(f"\n  [合并] 合并后数据集 {df_merged.shape[0]} 行 × {df_merged.shape[1]} 列")
    return df_merged


# ---------------------------------------------------------------------------
# 步骤 2：描述性统计
# ---------------------------------------------------------------------------

def descriptive_statistics(
    df: pd.DataFrame,
    cols: List[str],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    对指定列计算描述性统计。

    - 近正态连续变量：均值 ± 标准差
    - 偏态连续变量：中位数 [Q1, Q3]
    - 二值/分类变量：频次（百分比）

    若提供 group_col，则分组后分别计算。

    返回
    ----
    pd.DataFrame  每行一个变量（× 每个分组）
    """
    rows = []
    groups = [None]
    if group_col and group_col in df.columns:
        groups = sorted(df[group_col].dropna().unique().tolist())

    for col in cols:
        if col not in df.columns:
            continue
        row: Dict = {"variable": col}

        for grp in groups:
            sub = df if grp is None else df[df[group_col] == grp]
            series = sub[col].dropna()
            prefix = "" if grp is None else f"grp_{grp}_"

            if _is_binary_col(series):
                n = len(series)
                cnt = int(series.sum())
                row[f"{prefix}n"] = n
                row[f"{prefix}count_1"] = cnt
                row[f"{prefix}pct_1"] = round(cnt / n * 100, 1) if n else np.nan
                row[f"{prefix}type"] = "binary"
            elif pd.api.types.is_numeric_dtype(series):
                n = len(series)
                row[f"{prefix}n"] = n
                if _is_likely_normal(series):
                    row[f"{prefix}mean"] = round(float(series.mean()), 3)
                    row[f"{prefix}sd"]   = round(float(series.std()), 3)
                    row[f"{prefix}type"] = "continuous_normal"
                else:
                    row[f"{prefix}median"] = round(float(series.median()), 3)
                    row[f"{prefix}q1"]     = round(float(series.quantile(0.25)), 3)
                    row[f"{prefix}q3"]     = round(float(series.quantile(0.75)), 3)
                    row[f"{prefix}type"]   = "continuous_skewed"
            else:
                row[f"{prefix}type"] = "categorical"

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 步骤 3：相关性分析
# ---------------------------------------------------------------------------

def correlation_analysis(
    df: pd.DataFrame,
    la_cols: List[str],
    clinical_cols: List[str],
    alpha: float = _ALPHA,
    fdr_correct: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算 LA 参数与临床变量之间的相关系数矩阵。

    对于每对 (LA参数, 临床变量)：
      - 两列均近似正态 → Pearson
      - 否则 → Spearman

    若 fdr_correct=True，对所有 p 值进行 Benjamini-Hochberg FDR 校正。

    返回
    ----
    corr_long : pd.DataFrame  长表（每行一对变量的相关结果）
                  cols: la_col, clinical_col, method, r, p_raw,
                        p_fdr (若 fdr_correct), significant_raw, significant_fdr
    corr_matrix : pd.DataFrame  以 la_col 为行、clinical_col 为列的 r 值宽矩阵
    """
    if not _SCIPY_AVAILABLE:
        warnings.warn("scipy 不可用，跳过相关性分析。", UserWarning)
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for la in la_cols:
        if la not in df.columns:
            continue
        for clin in clinical_cols:
            if clin not in df.columns:
                continue
            sub = df[[la, clin]].dropna()
            if len(sub) < 5:
                continue

            both_normal = _is_likely_normal(sub[la]) and _is_likely_normal(sub[clin])
            if both_normal:
                r, p = _scipy_stats.pearsonr(sub[la], sub[clin])
                method = "pearson"
            else:
                r, p = _scipy_stats.spearmanr(sub[la], sub[clin])
                method = "spearman"

            rows.append({
                "la_col":       la,
                "clinical_col": clin,
                "method":       method,
                "r":            round(float(r), 4),
                "p_raw":        float(p),
                "n":            len(sub),
            })

    corr_long = pd.DataFrame(rows)
    if corr_long.empty:
        return corr_long, pd.DataFrame()

    # FDR 校正
    if fdr_correct and _STATSMODELS_AVAILABLE and len(corr_long):
        _, p_fdr, _, _ = multipletests(
            corr_long["p_raw"].values, alpha=alpha, method="fdr_bh"
        )
        corr_long["p_fdr"] = p_fdr
        corr_long["significant_fdr"] = p_fdr < alpha
    corr_long["significant_raw"] = corr_long["p_raw"] < alpha

    # 宽矩阵（r 值）
    corr_matrix = corr_long.pivot(
        index="la_col", columns="clinical_col", values="r"
    )

    return corr_long, corr_matrix


# ---------------------------------------------------------------------------
# 步骤 4：VIF 共线性检验
# ---------------------------------------------------------------------------

def vif_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    vif_thresh: float = _VIF_THRESH,
) -> pd.DataFrame:
    """
    对指定特征列计算方差膨胀因子（VIF）。

    迭代删除 VIF 最大且超过阈值的特征（每轮只删一个），
    直到所有特征的 VIF ≤ vif_thresh 或特征数不足。

    返回
    ----
    pd.DataFrame  每行一个特征，列：
      feature, vif, retained, note
    """
    if not _STATSMODELS_AVAILABLE:
        warnings.warn("statsmodels 不可用，跳过 VIF 分析。", UserWarning)
        return pd.DataFrame()

    avail = [c for c in feature_cols if c in df.columns
             and pd.api.types.is_numeric_dtype(df[c])]
    if len(avail) < 2:
        warnings.warn("VIF 分析需要至少 2 个数值特征列。", UserWarning)
        return pd.DataFrame()

    # 仅保留无 NaN 的行
    sub = df[avail].dropna()
    if len(sub) < len(avail) + 1:
        warnings.warn("样本量不足（< 特征数 + 1），跳过 VIF 分析。", UserWarning)
        return pd.DataFrame()

    vif_records: Dict[str, Dict] = {c: {"feature": c, "vif": np.nan,
                                         "retained": True, "note": ""} for c in avail}
    remaining = list(avail)

    while len(remaining) >= 2:
        X = sub[remaining].values.astype(float)
        vifs = []
        for i in range(X.shape[1]):
            try:
                v = variance_inflation_factor(X, i)
            except Exception:
                v = np.nan
            vifs.append(v)

        max_vif = max((v for v in vifs if not np.isnan(v)), default=0.0)
        for i, col in enumerate(remaining):
            vif_records[col]["vif"] = round(float(vifs[i]), 3) \
                if not np.isnan(vifs[i]) else np.nan

        if max_vif <= vif_thresh:
            break

        # 剔除 VIF 最大的特征
        max_idx = int(np.nanargmax(vifs))
        drop_col = remaining[max_idx]
        vif_records[drop_col]["retained"] = False
        vif_records[drop_col]["note"] = (
            f"VIF={round(float(vifs[max_idx]), 2)} > {vif_thresh}，已剔除"
        )
        remaining.remove(drop_col)

    result = pd.DataFrame(list(vif_records.values()))
    result = result.sort_values("vif", ascending=False)
    high_count = int((result["vif"] > vif_thresh).sum())
    print(f"  [VIF] 分析 {len(avail)} 个特征，"
          f"{high_count} 个 VIF > {vif_thresh}（标记为不保留）")
    return result


# ---------------------------------------------------------------------------
# 步骤 5：组间比较
# ---------------------------------------------------------------------------

def group_comparison(
    df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
    alpha: float = _ALPHA,
    fdr_correct: bool = True,
) -> pd.DataFrame:
    """
    对每个特征列按 group_col 分组进行统计检验。

    两组：
      连续近正态 → 独立样本 t 检验（Levene 方差齐性预检）
      连续偏态   → Mann-Whitney U 检验
      分类/二值  → χ² 检验（理论频数 < 5 时改用 Fisher 精确检验）

    三组及以上：
      连续近正态 → 单因素方差分析（ANOVA）
      连续偏态   → Kruskal-Wallis 检验
      分类/二值  → χ² 检验

    若 fdr_correct=True，对所有 p 值进行 BH-FDR 校正。

    返回
    ----
    pd.DataFrame  每行一个特征，列：
      feature, n_groups, test, statistic, p_raw, p_fdr, significant_fdr,
      group_stats (JSON 字符串，每组均值/中位数等)
    """
    if not _SCIPY_AVAILABLE:
        warnings.warn("scipy 不可用，跳过组间比较。", UserWarning)
        return pd.DataFrame()

    if group_col not in df.columns:
        warnings.warn(f"分组列 '{group_col}' 不存在于数据集中。", UserWarning)
        return pd.DataFrame()

    group_vals = sorted(df[group_col].dropna().unique().tolist())
    n_groups = len(group_vals)
    if n_groups < 2:
        warnings.warn(f"分组列 '{group_col}' 仅有 {n_groups} 个唯一值，跳过。", UserWarning)
        return pd.DataFrame()

    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        samples = [df.loc[df[group_col] == g, col].dropna().values
                   for g in group_vals]
        # 过滤空组
        valid_groups = [(g, s) for g, s in zip(group_vals, samples) if len(s) >= 2]
        if len(valid_groups) < 2:
            continue

        g_labels, g_samples = zip(*valid_groups)
        all_vals = np.concatenate(g_samples)
        is_bin = _is_binary_col(pd.Series(all_vals))
        is_norm = _is_likely_normal(pd.Series(all_vals)) and not is_bin

        stat, p = np.nan, np.nan
        test_name = ""

        try:
            if is_bin or not pd.api.types.is_numeric_dtype(
                df[col].dropna().iloc[:1]
            ):
                # χ² 或 Fisher
                ct = pd.crosstab(df[group_col].dropna(), df[col].dropna())
                if ct.shape == (2, 2):
                    _, p_fish = _scipy_stats.fisher_exact(ct.values)
                    chi2, p_chi, *_ = _scipy_stats.chi2_contingency(ct)
                    # 理论频数 < 5 时用 Fisher
                    expected = _scipy_stats.chi2_contingency(ct)[3]
                    if (expected < 5).any():
                        stat, p, test_name = np.nan, p_fish, "fisher_exact"
                    else:
                        stat, p, test_name = chi2, p_chi, "chi2"
                else:
                    chi2, p, *_ = _scipy_stats.chi2_contingency(ct)
                    stat, test_name = chi2, "chi2"

            elif n_groups == 2 and len(valid_groups) == 2:
                s0, s1 = g_samples[0], g_samples[1]
                if is_norm:
                    _, p_lev = _scipy_stats.levene(s0, s1)
                    equal_var = p_lev >= 0.05
                    stat, p = _scipy_stats.ttest_ind(s0, s1, equal_var=equal_var)
                    test_name = "t_test" if equal_var else "welch_t"
                else:
                    stat, p = _scipy_stats.mannwhitneyu(s0, s1, alternative="two-sided")
                    test_name = "mann_whitney_u"

            else:
                # 三组及以上
                if is_norm:
                    stat, p = _scipy_stats.f_oneway(*g_samples)
                    test_name = "anova"
                else:
                    stat, p = _scipy_stats.kruskal(*g_samples)
                    test_name = "kruskal_wallis"

        except Exception as exc:
            test_name = f"error({exc})"

        # 每组简要统计
        grp_stats = {}
        for glbl, gsmp in zip(g_labels, g_samples):
            gseries = pd.Series(gsmp)
            if is_norm:
                grp_stats[str(glbl)] = {
                    "n": len(gsmp),
                    "mean": round(float(gseries.mean()), 3),
                    "sd":   round(float(gseries.std()), 3),
                }
            else:
                grp_stats[str(glbl)] = {
                    "n":      len(gsmp),
                    "median": round(float(gseries.median()), 3),
                    "q1":     round(float(gseries.quantile(0.25)), 3),
                    "q3":     round(float(gseries.quantile(0.75)), 3),
                }

        rows.append({
            "feature":    col,
            "group_col":  group_col,
            "n_groups":   len(valid_groups),
            "test":       test_name,
            "statistic":  round(float(stat), 4) if not np.isnan(stat) else np.nan,
            "p_raw":      float(p) if not np.isnan(p) else np.nan,
            "group_stats": json.dumps(grp_stats, ensure_ascii=False),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # FDR 校正
    if fdr_correct and _STATSMODELS_AVAILABLE:
        valid_p = result["p_raw"].notna()
        if valid_p.any():
            _, p_fdr, _, _ = multipletests(
                result.loc[valid_p, "p_raw"].values, alpha=alpha, method="fdr_bh"
            )
            result.loc[valid_p, "p_fdr"] = p_fdr
            result["significant_fdr"] = result["p_fdr"] < alpha
    result["significant_raw"] = result["p_raw"] < alpha

    sig_count = int(result.get("significant_fdr", result["significant_raw"]).sum())
    print(f"  [组间比较] 检验 {len(result)} 个特征，"
          f"FDR 校正后显著 {sig_count} 个（α={alpha}）")
    return result


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _setup_chinese_font() -> bool:
    """尝试配置 matplotlib 使用 CJK 字体（与 fit_cox_model.py 逻辑一致）。"""
    if not _MPL_AVAILABLE:
        return False
    from matplotlib import font_manager as _fm
    _candidates = [
        "SimHei", "Microsoft YaHei", "PingFang SC", "Heiti SC", "STHeiti",
        "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei",
        "Source Han Sans CN", "FangSong",
    ]
    _available = {f.name for f in _fm.fontManager.ttflist}
    for _font in _candidates:
        if _font in _available:
            matplotlib.rcParams["font.sans-serif"] = (
                [_font] + list(matplotlib.rcParams.get("font.sans-serif", []))
            )
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: str,
    title: str = "LA Parameters × Clinical Variables Correlation",
) -> None:
    """绘制相关系数热图并保存。"""
    if not _MPL_AVAILABLE or corr_matrix.empty:
        return

    _setup_chinese_font()
    n_rows, n_cols = corr_matrix.shape
    fig_w = max(10, n_cols * 0.7)
    fig_h = max(6,  n_rows * 0.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    data = corr_matrix.values.astype(float)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.6)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(corr_matrix.columns.tolist(), rotation=45,
                        ha="right", fontsize=8)
    ax.set_yticklabels(corr_matrix.index.tolist(), fontsize=8)

    # 在格子中显示数值
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [可视化] 相关热图 → {output_path}")


def plot_distribution(
    df: pd.DataFrame,
    cols: List[str],
    output_dir: str,
    max_cols_per_fig: int = 16,
) -> None:
    """
    对指定列绘制分布直方图（含 KDE），保存至 output_dir/distributions.png。

    与论文一致：通过可视化分布图检查影像参数中的极端值。
    """
    if not _MPL_AVAILABLE:
        return

    _setup_chinese_font()
    avail = [c for c in cols if c in df.columns
             and pd.api.types.is_numeric_dtype(df[c])
             and not _is_binary_col(df[c])]
    if not avail:
        return

    avail = avail[:max_cols_per_fig]
    n = len(avail)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 3),
                              squeeze=False)
    for idx, col in enumerate(avail):
        r, c_ = divmod(idx, ncols)
        ax = axes[r][c_]
        data = df[col].dropna()
        ax.hist(data, bins=20, density=True, alpha=0.6, color="steelblue",
                edgecolor="white")
        try:
            kde = _scipy_stats.gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 200)
            ax.plot(xs, kde(xs), color="darkblue", linewidth=1.5)
        except Exception:
            pass
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)

    # 隐藏多余子图
    for idx in range(n, nrows * ncols):
        r, c_ = divmod(idx, ncols)
        axes[r][c_].set_visible(False)

    plt.suptitle("LA Parameter Distributions", fontsize=11, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "la_distributions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [可视化] 分布图 → {out_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_la_analysis(
    clinical_csv: str,
    morph_csv: str,
    kine_csv: str,
    qc_csv: Optional[str],
    output_dir: str,
    group_cols: Optional[List[str]] = None,
    join_keys: Optional[List[str]] = None,
    no_plots: bool = False,
    alpha: float = _ALPHA,
    vif_thresh: float = _VIF_THRESH,
) -> Dict[str, pd.DataFrame]:
    """
    完整 LA × MIMIC 临床参数联合分析流程。

    参数
    ----
    clinical_csv : 临床宽表 CSV（impute_normalize.py 输出）
    morph_csv    : 形态学宽表 CSV（clean_la_params.py 输出）
    kine_csv     : 运动学宽表 CSV（clean_la_params.py 输出）
    qc_csv       : QC 元数据 CSV（可选，用于附加 QC 字段）
    output_dir   : 输出目录
    group_cols   : 用于组间比较的分组列名列表（如 ['died_inhosp', 'os_event']）
    join_keys    : 合并键，默认 ['subject_id']
    no_plots     : True → 跳过所有图形输出
    alpha        : 统计显著性水平
    vif_thresh   : VIF 高共线性阈值

    返回
    ----
    dict with keys:
      "merged", "descriptive", "corr_long", "corr_matrix",
      "vif_report", "group_comparisons"
    """
    if join_keys is None:
        join_keys = ["subject_id"]
    if group_cols is None:
        group_cols = []

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("LA 参数分析流程开始")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # 加载
    # ------------------------------------------------------------------ #
    df_clin  = pd.read_csv(clinical_csv,  low_memory=False)
    df_morph = pd.read_csv(morph_csv,     low_memory=False)
    df_kine  = pd.read_csv(kine_csv,      low_memory=False)
    df_qc    = pd.read_csv(qc_csv,        low_memory=False) if qc_csv else None

    print(f"  [加载] 临床表   {df_clin.shape[0]} 行 × {df_clin.shape[1]} 列")
    print(f"  [加载] 形态宽表 {df_morph.shape[0]} 行 × {df_morph.shape[1]} 列")
    print(f"  [加载] 运动宽表 {df_kine.shape[0]} 行 × {df_kine.shape[1]} 列")

    # ------------------------------------------------------------------ #
    # 合并
    # ------------------------------------------------------------------ #
    df_merged = merge_datasets(df_clin, df_morph, df_kine, df_qc,
                                join_keys=join_keys)

    merged_path = os.path.join(output_dir, "merged_dataset.csv")
    df_merged.to_csv(merged_path, index=False)
    print(f"  [保存] 合并数据集 → {merged_path}")

    # ------------------------------------------------------------------ #
    # 列分类
    # ------------------------------------------------------------------ #
    skip_cols = (
        _CLINICAL_ID_COLS | _OUTCOME_COLS | _LA_META_COLS
        | {c for c in df_merged.columns if c.endswith("_missing_flag")}
        | set(group_cols)
    )
    all_cols = [c for c in df_merged.columns if c not in skip_cols]

    # LA 参数列（来自形态 / 运动宽表）
    morph_feature_cols = [c for c in df_morph.columns
                          if c not in _LA_META_COLS and c in df_merged.columns]
    kine_feature_cols  = [c for c in df_kine.columns
                          if c not in _LA_META_COLS and c in df_merged.columns]
    la_cols = list(dict.fromkeys(morph_feature_cols + kine_feature_cols))

    # 临床变量列（数值型）
    clinical_feature_cols = [
        c for c in df_clin.columns
        if c not in skip_cols and c in df_merged.columns
        and pd.api.types.is_numeric_dtype(df_merged[c])
    ]

    print(f"\n  LA 参数列：{len(la_cols)} 个")
    print(f"  临床特征列：{len(clinical_feature_cols)} 个")

    # ------------------------------------------------------------------ #
    # 描述性统计
    # ------------------------------------------------------------------ #
    print("\n--- 描述性统计 ---")
    desc_df = descriptive_statistics(
        df_merged, cols=all_cols,
        group_col=group_cols[0] if group_cols else None,
    )
    desc_path = os.path.join(output_dir, "descriptive_stats.csv")
    desc_df.to_csv(desc_path, index=False)
    print(f"  [保存] 描述性统计 → {desc_path}")

    # ------------------------------------------------------------------ #
    # 分布可视化（影像参数）
    # ------------------------------------------------------------------ #
    if not no_plots and la_cols:
        plot_distribution(df_merged, la_cols, output_dir)

    # ------------------------------------------------------------------ #
    # 相关性分析
    # ------------------------------------------------------------------ #
    print("\n--- 相关性分析 ---")
    corr_long, corr_matrix = correlation_analysis(
        df_merged, la_cols=la_cols, clinical_cols=clinical_feature_cols,
        alpha=alpha, fdr_correct=True,
    )
    if not corr_long.empty:
        corr_long_path = os.path.join(output_dir, "correlation_la_clinical.csv")
        corr_long.to_csv(corr_long_path, index=False)
        print(f"  [保存] 相关系数长表 → {corr_long_path}")

        if not corr_matrix.empty:
            corr_mat_path = os.path.join(output_dir, "correlation_matrix.csv")
            corr_matrix.to_csv(corr_mat_path)
            print(f"  [保存] 相关矩阵 → {corr_mat_path}")

            if not no_plots:
                heatmap_path = os.path.join(output_dir, "correlation_la_clinical.png")
                plot_correlation_heatmap(corr_matrix, heatmap_path)
    else:
        print("  相关性分析：无有效列对，跳过。")

    # ------------------------------------------------------------------ #
    # VIF 共线性检验（LA 参数内部）
    # ------------------------------------------------------------------ #
    print("\n--- VIF 共线性检验（LA 参数）---")
    vif_la = vif_analysis(df_merged, feature_cols=la_cols, vif_thresh=vif_thresh)
    if not vif_la.empty:
        vif_path = os.path.join(output_dir, "vif_report_la.csv")
        vif_la.to_csv(vif_path, index=False)
        print(f"  [保存] LA 参数 VIF 报告 → {vif_path}")

    # 临床变量 VIF（可选；作为协变量预筛选参考）
    if clinical_feature_cols:
        print("\n--- VIF 共线性检验（临床变量）---")
        vif_clin = vif_analysis(df_merged, feature_cols=clinical_feature_cols,
                                vif_thresh=vif_thresh)
        if not vif_clin.empty:
            vif_clin_path = os.path.join(output_dir, "vif_report_clinical.csv")
            vif_clin.to_csv(vif_clin_path, index=False)
            print(f"  [保存] 临床变量 VIF 报告 → {vif_clin_path}")

    # ------------------------------------------------------------------ #
    # 组间比较
    # ------------------------------------------------------------------ #
    group_results: Dict[str, pd.DataFrame] = {}
    for gcol in group_cols:
        print(f"\n--- 组间比较：{gcol} ---")
        gc_df = group_comparison(
            df_merged,
            feature_cols=la_cols + clinical_feature_cols,
            group_col=gcol,
            alpha=alpha,
            fdr_correct=True,
        )
        if not gc_df.empty:
            gc_path = os.path.join(output_dir, f"group_comparison_{gcol}.csv")
            gc_df.to_csv(gc_path, index=False)
            print(f"  [保存] 组间比较结果 → {gc_path}")
            group_results[gcol] = gc_df

    print(f"\n{'='*60}")
    print(f"LA 参数分析完成。输出目录：{output_dir}")
    print(f"{'='*60}\n")

    return {
        "merged":            df_merged,
        "descriptive":       desc_df,
        "corr_long":         corr_long,
        "corr_matrix":       corr_matrix,
        "vif_la":            vif_la,
        "group_comparisons": group_results,
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
        "--clinical_csv",
        default="csv/processed/hfpef_cohort_win_hadm_processed.csv",
        help="临床处理后宽表 CSV 路径",
    )
    ap.add_argument(
        "--morph_csv",
        default="csv/la_params/processed/la_morphology_wide.csv",
        help="形态学参数宽表 CSV 路径",
    )
    ap.add_argument(
        "--kine_csv",
        default="csv/la_params/processed/la_kinematic_wide.csv",
        help="运动学参数宽表 CSV 路径",
    )
    ap.add_argument(
        "--qc_csv",
        default="csv/la_params/processed/la_params_qc_filtered.csv",
        help="QC 过滤后表 CSV 路径（可选，留空则跳过）",
    )
    ap.add_argument(
        "--output_dir",
        default="csv/la_analysis",
        help="输出目录（默认：csv/la_analysis）",
    )
    ap.add_argument(
        "--group_col",
        nargs="+",
        default=[],
        metavar="COL",
        help="组间比较分组列（可多个，如 --group_col died_inhosp os_event）",
    )
    ap.add_argument(
        "--join_key",
        nargs="+",
        default=["subject_id"],
        metavar="KEY",
        help="临床表与影像表的合并键（默认：subject_id）",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=_ALPHA,
        help=f"统计显著性水平（默认：{_ALPHA}）",
    )
    ap.add_argument(
        "--vif_thresh",
        type=float,
        default=_VIF_THRESH,
        help=f"VIF 高共线性阈值（默认：{_VIF_THRESH}）",
    )
    ap.add_argument(
        "--no_plots",
        action="store_true",
        help="跳过所有图形输出",
    )
    args = ap.parse_args()

    args.clinical_csv = _resolve_path(args.clinical_csv)
    args.morph_csv    = _resolve_path(args.morph_csv)
    args.kine_csv     = _resolve_path(args.kine_csv)
    if args.qc_csv:
        args.qc_csv   = _resolve_path(args.qc_csv)
    args.output_dir   = _resolve_path(args.output_dir)

    qc_csv = args.qc_csv if (args.qc_csv and os.path.exists(args.qc_csv)) else None

    run_la_analysis(
        clinical_csv=args.clinical_csv,
        morph_csv=args.morph_csv,
        kine_csv=args.kine_csv,
        qc_csv=qc_csv,
        output_dir=args.output_dir,
        group_cols=args.group_col,
        join_keys=args.join_key,
        no_plots=args.no_plots,
        alpha=args.alpha,
        vif_thresh=args.vif_thresh,
    )


if __name__ == "__main__":
    main()
