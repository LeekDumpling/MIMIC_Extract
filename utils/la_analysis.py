# -*- coding: utf-8 -*-
"""
左心房参数表现分析脚本 - 单窗口分析器

用途
----
对单个临床窗口的 LA 原始清洗宽表执行论文所需的参数表现分析，并
输出 Cox 建模候选池。该脚本只负责单窗口，三窗口批量编排由
``utils/la_pipeline.py`` 完成。

输入
----
- 临床表：``csv/comorbidity/hfpef_cohort_win_<window>_comorbidity.csv``
- LA 原始宽表：
  - ``csv/la_params/processed/la_morphology_wide_raw.csv``
  - ``csv/la_params/processed/la_kinematic_wide_raw.csv``
- LA 元数据：
  - ``csv/la_params/processed/la_feature_catalog.csv``
  - ``csv/la_params/processed/la_params_feature_decisions.csv``

输出
----
- merged_dataset.csv
- availability_summary.csv
- distribution_stats_raw.csv
- correlation_la_clinical.csv
- correlation_la_internal.csv
- high_corr_pairs_la.csv
- vif_report_la.csv
- la_candidates_for_cox.csv
- analysis_summary.json
- figures/
  - availability_bar.png
  - distribution_morphology.png
  - distribution_kinematic_p01.png, p02 ...
  - correlation_la_clinical.png
  - correlation_la_morphology.png
  - correlation_la_kinematic.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    raise ImportError("numpy 和 pandas 是必需依赖。\n请执行：pip install numpy pandas") from None

try:
    from scipy import stats as _scipy_stats

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    _utils_dir = os.path.dirname(os.path.abspath(__file__))
    if _utils_dir not in sys.path:
        sys.path.insert(0, _utils_dir)
    from ph_viz import setup_chinese_font as _shared_setup_chinese_font  # type: ignore
except ImportError:
    _shared_setup_chinese_font = None


_MORPH_META_COLS = {"video_prefix", "subject_id", "study_id", "source_group"}
_CLINICAL_ID_COLS = {
    "subject_id",
    "hadm_id",
    "index_study_id",
    "index_study_datetime",
    "index_admittime",
    "index_dischtime",
    "death_date",
    "last_dischtime",
    "censor_date",
    "a4c_dicom_filepath",
}
_OUTCOME_COLS = {
    "hospital_expire_flag",
    "died_inhosp",
    "died_post_dc",
    "days_survived_post_dc",
    "died_30d",
    "died_90d",
    "died_1yr",
    "os_event",
    "os_days",
}
_FIXED_CLINICAL_COLS = [
    "ntprobnp",
    "troponin_t",
    "crp",
    "albumin",
    "creatinine",
]
_HIGH_CORR_THRESHOLD = 0.80
_SKEW_THRESH = 1.0
_ALPHA = 0.05
_VIF_THRESH = 10.0
_DPI = 300
_MAX_HEATMAP_ANNOTATE_CELLS = 400


def _resolve_path(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


def _infer_window_name(path: str) -> str:
    basename = os.path.basename(path)
    match = re.search(r"hfpef_cohort_win_(.+?)_", basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]


def _is_binary_col(series: pd.Series) -> bool:
    uniq = set(series.dropna().unique())
    return uniq.issubset({0, 1, 0.0, 1.0, True, False})


def _is_likely_normal(series: pd.Series) -> bool:
    clean = series.dropna()
    if len(clean) < 3:
        return False
    return abs(float(clean.skew())) <= _SKEW_THRESH


def _prepare_join_columns(df: pd.DataFrame, study_col: str) -> pd.DataFrame:
    out = df.copy()
    out["subject_id"] = pd.to_numeric(out["subject_id"], errors="coerce").astype("Int64")
    out["_study_key"] = pd.to_numeric(out[study_col], errors="coerce").astype("Int64")
    return out


def _build_flag_column(feature: str) -> str:
    return f"{feature}_review_flag"


def _load_required_table(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} 不存在：{path}")
    return pd.read_csv(path, low_memory=False)


def _load_feature_catalog(path: str) -> pd.DataFrame:
    df = _load_required_table(path, "LA 特征目录")
    required = {"source_table", "parameter_name", "sub_item", "safe_name", "unit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"la_feature_catalog.csv 缺少字段：{sorted(missing)}")
    return df.copy()


def _load_feature_decisions(path: str) -> pd.DataFrame:
    df = _load_required_table(path, "LA 特征决策表")
    required = {"column", "decision", "table"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"la_params_feature_decisions.csv 缺少字段：{sorted(missing)}")
    out = df.copy()
    out = out.rename(columns={"column": "safe_name", "table": "source_table"})
    return out


def merge_window_dataset(
    clinical_csv: str,
    morph_csv: str,
    kine_csv: str,
    qc_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Merge one clinical window with raw LA morphology and kinematic tables."""
    df_clin = _load_required_table(clinical_csv, "临床表")
    df_morph = _load_required_table(morph_csv, "LA 形态原始宽表")
    df_kine = _load_required_table(kine_csv, "LA 运动原始宽表")
    df_qc = _load_required_table(qc_csv, "LA QC 表") if qc_csv else None

    clin = _prepare_join_columns(df_clin, "index_study_id")
    morph = _prepare_join_columns(df_morph, "study_id")
    kine = _prepare_join_columns(df_kine, "study_id")

    morph_keys = ["video_prefix", "subject_id", "study_id", "source_group", "_study_key"]
    kine_keys = ["video_prefix", "subject_id", "study_id", "source_group", "_study_key"]

    overlap = (
        set(df_morph.columns) - _MORPH_META_COLS
    ) & (
        set(df_kine.columns) - _MORPH_META_COLS
    )
    if overlap:
        kine = kine.rename(columns={col: f"{col}_kine" for col in overlap})

    df_img = pd.merge(morph, kine, on=morph_keys, how="outer", suffixes=("", "_kine"))
    df_merged = pd.merge(
        clin,
        df_img,
        on=["subject_id", "_study_key"],
        how="inner",
        suffixes=("", "_img"),
    )

    if df_qc is not None and "video_prefix" in df_qc.columns and "video_prefix" in df_merged.columns:
        qc_cols = [
            c for c in [
                "video_prefix",
                "spacing_found",
                "clinical_found",
                "fps",
                "valid_cycle_count",
                "keyframe_method",
                "dense_kpts_source",
                "gt_kpts_available",
                "status",
            ] if c in df_qc.columns
        ]
        qc_sub = df_qc[qc_cols].drop_duplicates(subset=["video_prefix"])
        df_merged = pd.merge(df_merged, qc_sub, on="video_prefix", how="left", suffixes=("", "_qc"))

    if "study_id" in df_merged.columns and "index_study_id" in df_merged.columns:
        df_merged["study_id"] = df_merged["index_study_id"]
    if "_study_key" in df_merged.columns:
        df_merged.drop(columns=["_study_key"], inplace=True)

    return df_merged


def build_feature_metadata(
    feature_catalog: pd.DataFrame,
    feature_decisions: pd.DataFrame,
) -> pd.DataFrame:
    meta = feature_catalog.copy()
    meta = pd.merge(
        meta,
        feature_decisions[["safe_name", "source_table", "decision", "pct_missing", "note"]],
        on=["safe_name", "source_table"],
        how="left",
    )
    return meta.sort_values(["source_table", "safe_name"]).reset_index(drop=True)


def availability_summary(
    df_merged: pd.DataFrame,
    feature_meta: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    n_total = len(df_merged)

    for _, row in feature_meta.iterrows():
        feature = str(row["safe_name"])
        flag_col = _build_flag_column(feature)
        present = feature in df_merged.columns

        n_non_missing = int(df_merged[feature].notna().sum()) if present else np.nan
        pct_available = round(n_non_missing / n_total * 100, 2) if present and n_total else np.nan
        if flag_col in df_merged.columns:
            n_review = int((df_merged[flag_col] == 1).sum())
            pct_review = round(n_review / n_total * 100, 2) if n_total else np.nan
        else:
            n_review = np.nan if not present else 0
            pct_review = np.nan if not present else 0.0

        rows.append({
            "source_table": row["source_table"],
            "parameter_name": row["parameter_name"],
            "sub_item": row["sub_item"],
            "safe_name": feature,
            "unit": row["unit"],
            "decision_a1": row.get("decision", ""),
            "present_in_dataset": bool(present),
            "n_total": n_total,
            "n_non_missing": n_non_missing,
            "pct_available": pct_available,
            "n_review_flag": n_review,
            "pct_review_flag": pct_review,
            "a1_pct_missing": row.get("pct_missing", np.nan),
            "a1_note": row.get("note", ""),
        })

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["present_in_dataset", "pct_available", "source_table", "safe_name"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def distribution_statistics(
    df_merged: pd.DataFrame,
    feature_meta: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for _, meta_row in feature_meta.iterrows():
        feature = str(meta_row["safe_name"])
        if feature not in df_merged.columns:
            continue
        series = pd.to_numeric(df_merged[feature], errors="coerce").dropna()
        if len(series) == 0:
            continue
        rows.append({
            "source_table": meta_row["source_table"],
            "parameter_name": meta_row["parameter_name"],
            "sub_item": meta_row["sub_item"],
            "safe_name": feature,
            "unit": meta_row["unit"],
            "n_non_missing": int(series.notna().sum()),
            "mean": round(float(series.mean()), 4),
            "sd": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "q1": round(float(series.quantile(0.25)), 4),
            "median": round(float(series.median()), 4),
            "q3": round(float(series.quantile(0.75)), 4),
            "max": round(float(series.max()), 4),
            "skewness": round(float(series.skew()), 4),
        })

    return pd.DataFrame(rows).sort_values(["source_table", "safe_name"]).reset_index(drop=True)


def correlation_against_clinical(
    df_merged: pd.DataFrame,
    la_features: Sequence[str],
    clinical_features: Sequence[str],
    alpha: float = _ALPHA,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not _SCIPY_AVAILABLE:
        raise RuntimeError("scipy 未安装，无法执行相关性分析。")

    rows: List[Dict[str, object]] = []
    for la_col in la_features:
        if la_col not in df_merged.columns:
            continue
        for clin_col in clinical_features:
            if clin_col not in df_merged.columns:
                continue
            sub = df_merged[[la_col, clin_col]].dropna()
            if len(sub) < 5:
                continue

            both_normal = _is_likely_normal(sub[la_col]) and _is_likely_normal(sub[clin_col])
            if both_normal:
                stat, p_value = _scipy_stats.pearsonr(sub[la_col], sub[clin_col])
                method = "pearson"
            else:
                stat, p_value = _scipy_stats.spearmanr(sub[la_col], sub[clin_col])
                method = "spearman"

            rows.append({
                "la_feature": la_col,
                "clinical_feature": clin_col,
                "method": method,
                "r": round(float(stat), 4),
                "p_raw": float(p_value),
                "n": int(len(sub)),
            })

    corr_long = pd.DataFrame(rows)
    if corr_long.empty:
        return corr_long, pd.DataFrame()

    if not _STATSMODELS_AVAILABLE:
        raise RuntimeError("statsmodels 未安装，无法执行 FDR 校正。")

    _, p_fdr, _, _ = multipletests(corr_long["p_raw"].values, alpha=alpha, method="fdr_bh")
    corr_long["p_fdr"] = p_fdr
    corr_long["significant_fdr"] = corr_long["p_fdr"] < alpha
    corr_long["significant_raw"] = corr_long["p_raw"] < alpha

    corr_matrix = corr_long.pivot(index="la_feature", columns="clinical_feature", values="r")
    return corr_long.sort_values(["p_fdr", "p_raw", "la_feature"]).reset_index(drop=True), corr_matrix


def internal_correlation_analysis(
    df_merged: pd.DataFrame,
    features: Sequence[str],
    panel: str,
    alpha: float = _ALPHA,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not _SCIPY_AVAILABLE:
        raise RuntimeError("scipy 未安装，无法执行相关性分析。")

    rows: List[Dict[str, object]] = []
    features = [f for f in features if f in df_merged.columns]
    for i, left in enumerate(features):
        for right in features[i + 1:]:
            sub = df_merged[[left, right]].dropna()
            if len(sub) < 5:
                continue
            both_normal = _is_likely_normal(sub[left]) and _is_likely_normal(sub[right])
            if both_normal:
                stat, p_value = _scipy_stats.pearsonr(sub[left], sub[right])
                method = "pearson"
            else:
                stat, p_value = _scipy_stats.spearmanr(sub[left], sub[right])
                method = "spearman"
            rows.append({
                "panel": panel,
                "feature_left": left,
                "feature_right": right,
                "method": method,
                "r": round(float(stat), 4),
                "p_raw": float(p_value),
                "n": int(len(sub)),
            })

    corr_long = pd.DataFrame(rows)
    if corr_long.empty:
        return corr_long, pd.DataFrame(index=features, columns=features, dtype=float)

    if not _STATSMODELS_AVAILABLE:
        raise RuntimeError("statsmodels 未安装，无法执行 FDR 校正。")

    _, p_fdr, _, _ = multipletests(corr_long["p_raw"].values, alpha=alpha, method="fdr_bh")
    corr_long["p_fdr"] = p_fdr
    corr_long["significant_fdr"] = corr_long["p_fdr"] < alpha
    corr_long["significant_raw"] = corr_long["p_raw"] < alpha

    matrix = pd.DataFrame(np.eye(len(features)), index=features, columns=features, dtype=float)
    for _, row in corr_long.iterrows():
        matrix.loc[row["feature_left"], row["feature_right"]] = row["r"]
        matrix.loc[row["feature_right"], row["feature_left"]] = row["r"]

    return corr_long.sort_values(["p_fdr", "p_raw", "feature_left"]).reset_index(drop=True), matrix


def vif_analysis(
    df_merged: pd.DataFrame,
    features: Sequence[str],
    threshold: float = _VIF_THRESH,
) -> pd.DataFrame:
    if not _STATSMODELS_AVAILABLE:
        raise RuntimeError("statsmodels 未安装，无法执行 VIF 分析。")

    features = [f for f in features if f in df_merged.columns]
    if len(features) < 2:
        return pd.DataFrame(columns=["feature", "vif", "retained", "note"])

    sub = df_merged[list(features)].dropna()
    if len(sub) < len(features) + 1:
        return pd.DataFrame(columns=["feature", "vif", "retained", "note"])

    records: Dict[str, Dict[str, object]] = {
        feature: {"feature": feature, "vif": np.nan, "retained": True, "note": ""}
        for feature in features
    }
    remaining = list(features)

    while len(remaining) >= 2:
        X = sub[remaining].astype(float).values
        vifs: List[float] = []
        for i in range(X.shape[1]):
            try:
                value = float(variance_inflation_factor(X, i))
            except Exception:
                value = np.nan
            vifs.append(value)

        max_vif = max((v for v in vifs if not np.isnan(v)), default=0.0)
        for idx, feature in enumerate(remaining):
            records[feature]["vif"] = round(vifs[idx], 4) if not np.isnan(vifs[idx]) else np.nan

        if max_vif <= threshold:
            break

        drop_idx = int(np.nanargmax(vifs))
        drop_feature = remaining[drop_idx]
        records[drop_feature]["retained"] = False
        records[drop_feature]["note"] = f"VIF={round(float(vifs[drop_idx]), 3)} > {threshold}"
        remaining.remove(drop_feature)

    result = pd.DataFrame(list(records.values()))
    if result.empty:
        return result
    return result.sort_values("vif", ascending=False, na_position="last").reset_index(drop=True)


def high_corr_pairs(
    corr_internal: pd.DataFrame,
    threshold: float = _HIGH_CORR_THRESHOLD,
) -> pd.DataFrame:
    if corr_internal.empty:
        return corr_internal
    out = corr_internal.loc[corr_internal["r"].abs() >= threshold].copy()
    return out.sort_values("r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def select_la_candidates_for_cox(
    df_merged: pd.DataFrame,
    feature_meta: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, meta_row in feature_meta.iterrows():
        feature = str(meta_row["safe_name"])
        decision = str(meta_row.get("decision", ""))
        if decision == "DROP":
            continue
        if feature not in df_merged.columns:
            continue

        series = pd.to_numeric(df_merged[feature], errors="coerce")
        n_non_missing = int(series.notna().sum())
        unique_non_null = int(series.dropna().nunique())
        if n_non_missing <= 0 or unique_non_null <= 1:
            continue

        rows.append({
            "source_table": meta_row["source_table"],
            "parameter_name": meta_row["parameter_name"],
            "sub_item": meta_row["sub_item"],
            "safe_name": feature,
            "unit": meta_row["unit"],
            "decision_a1": decision,
            "n_non_missing": n_non_missing,
            "n_unique_non_null": unique_non_null,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["source_table", "safe_name"]).reset_index(drop=True)


def _require_plotting(font_family: Optional[str]) -> None:
    if not _MPL_AVAILABLE:
        raise RuntimeError("matplotlib 未安装，无法生成 A2 图表。")
    if _shared_setup_chinese_font is None:
        raise RuntimeError("ph_viz.py 不可用，无法统一配置中文字体。")
    _shared_setup_chinese_font(font_family=font_family, strict=True)


def plot_availability_bar(
    availability_df: pd.DataFrame,
    output_path: str,
    title: str,
) -> None:
    plot_df = availability_df[availability_df["present_in_dataset"]].copy()
    plot_df = plot_df.sort_values("pct_available", ascending=True)
    if plot_df.empty:
        return

    fig_h = max(6.0, 0.28 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(plot_df["safe_name"], plot_df["pct_available"], color="#2a6f97")
    ax.set_xlabel("可用率 (%)", fontsize=10)
    ax.set_ylabel("LA 参数", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 100)
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            min(float(row["pct_available"]) + 1.2, 99.5),
            idx,
            f'{int(row["n_non_missing"])}/{int(row["n_total"])}',
            va="center",
            fontsize=7,
        )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_pages(
    df_merged: pd.DataFrame,
    feature_meta: pd.DataFrame,
    source_table: str,
    output_dir: str,
    title_prefix: str,
    base_filename: str,
    max_per_page: int = 16,
) -> List[str]:
    meta = feature_meta[feature_meta["source_table"] == source_table].copy()
    features = [f for f in meta["safe_name"].tolist() if f in df_merged.columns]
    if not features:
        return []

    saved: List[str] = []
    for page_idx, start in enumerate(range(0, len(features), max_per_page), start=1):
        chunk = features[start:start + max_per_page]
        n = len(chunk)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 3.8, nrows * 3.2),
            squeeze=False,
        )
        for idx, feature in enumerate(chunk):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            data = pd.to_numeric(df_merged[feature], errors="coerce").dropna()
            ax.hist(
                data,
                bins=20,
                density=True,
                alpha=0.65,
                color="#669bbc",
                edgecolor="white",
            )
            if len(data) >= 5 and _SCIPY_AVAILABLE:
                try:
                    kde = _scipy_stats.gaussian_kde(data)
                    xs = np.linspace(float(data.min()), float(data.max()), 200)
                    ax.plot(xs, kde(xs), color="#1d3557", linewidth=1.5)
                except Exception:
                    pass
            ax.set_title(feature, fontsize=8)
            ax.tick_params(labelsize=7)

        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].set_visible(False)

        suffix = "" if (source_table == "morphology" and page_idx == 1) else f" 第 {page_idx} 页"
        fig.suptitle(f"{title_prefix}{suffix}", fontsize=12, y=1.01)
        plt.tight_layout()

        if source_table == "morphology":
            out_path = os.path.join(output_dir, f"{base_filename}.png")
        else:
            out_path = os.path.join(output_dir, f"{base_filename}_p{page_idx:02d}.png")
        plt.savefig(out_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


def plot_heatmap(
    matrix: pd.DataFrame,
    output_path: str,
    title: str,
) -> None:
    if matrix.empty:
        return

    n_rows, n_cols = matrix.shape
    fig_w = max(8.0, n_cols * 0.38)
    fig_h = max(6.0, n_rows * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    data = matrix.astype(float).values

    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index.tolist(), fontsize=8)
    ax.set_title(title, fontsize=12, pad=12)

    if n_rows * n_cols <= _MAX_HEATMAP_ANNOTATE_CELLS:
        for i in range(n_rows):
            for j in range(n_cols):
                value = data[i, j]
                if np.isnan(value):
                    continue
                color = "white" if abs(value) >= 0.55 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def run_la_analysis(
    clinical_csv: str,
    morph_csv: str,
    kine_csv: str,
    feature_catalog_csv: str,
    feature_decisions_csv: str,
    output_dir: str,
    qc_csv: Optional[str] = None,
    window_name: Optional[str] = None,
    alpha: float = _ALPHA,
    vif_thresh: float = _VIF_THRESH,
    font_family: Optional[str] = None,
    no_plots: bool = False,
) -> Dict[str, object]:
    if not _SCIPY_AVAILABLE:
        raise RuntimeError("scipy 未安装，无法运行 A2 参数表现分析。")
    if not _STATSMODELS_AVAILABLE:
        raise RuntimeError("statsmodels 未安装，无法运行 A2 参数表现分析。")
    if not no_plots:
        _require_plotting(font_family)

    clinical_csv = _resolve_path(clinical_csv)
    morph_csv = _resolve_path(morph_csv)
    kine_csv = _resolve_path(kine_csv)
    feature_catalog_csv = _resolve_path(feature_catalog_csv)
    feature_decisions_csv = _resolve_path(feature_decisions_csv)
    if qc_csv:
        qc_csv = _resolve_path(qc_csv)
    output_dir = _resolve_path(output_dir)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    if not no_plots:
        os.makedirs(figures_dir, exist_ok=True)

    window_name = window_name or _infer_window_name(clinical_csv)
    print(f"\n{'=' * 60}")
    print(f"LA 参数表现分析开始 - {window_name}")
    print(f"{'=' * 60}")

    df_merged = merge_window_dataset(
        clinical_csv=clinical_csv,
        morph_csv=morph_csv,
        kine_csv=kine_csv,
        qc_csv=qc_csv,
    )
    feature_catalog = _load_feature_catalog(feature_catalog_csv)
    feature_decisions = _load_feature_decisions(feature_decisions_csv)
    feature_meta = build_feature_metadata(feature_catalog, feature_decisions)

    morph_features = [
        col
        for col in feature_meta.loc[feature_meta["source_table"] == "morphology", "safe_name"].tolist()
        if col in df_merged.columns
    ]
    kine_features = [
        col
        for col in feature_meta.loc[feature_meta["source_table"] == "kinematic", "safe_name"].tolist()
        if col in df_merged.columns
    ]
    real_la_features = morph_features + kine_features
    clinical_features = [
        col for col in _FIXED_CLINICAL_COLS
        if col in df_merged.columns and pd.api.types.is_numeric_dtype(df_merged[col])
    ]

    print(f"  合并后样本数：{len(df_merged)}")
    print(f"  形态学参数列：{len(morph_features)}")
    print(f"  运动学参数列：{len(kine_features)}")
    print(f"  心功能相关指标：{clinical_features}")

    availability_df = availability_summary(df_merged, feature_meta)
    distribution_df = distribution_statistics(df_merged, feature_meta)
    corr_clin_long, corr_clin_matrix = correlation_against_clinical(
        df_merged,
        la_features=real_la_features,
        clinical_features=clinical_features,
        alpha=alpha,
    )
    corr_morph_long, corr_morph_matrix = internal_correlation_analysis(
        df_merged,
        features=morph_features,
        panel="morphology",
        alpha=alpha,
    )
    corr_kine_long, corr_kine_matrix = internal_correlation_analysis(
        df_merged,
        features=kine_features,
        panel="kinematic",
        alpha=alpha,
    )
    corr_internal = pd.concat([corr_morph_long, corr_kine_long], ignore_index=True)
    high_corr_df = high_corr_pairs(corr_internal, threshold=_HIGH_CORR_THRESHOLD)
    vif_df = vif_analysis(df_merged, real_la_features, threshold=vif_thresh)
    candidate_df = select_la_candidates_for_cox(df_merged, feature_meta)

    merged_path = os.path.join(output_dir, "merged_dataset.csv")
    availability_path = os.path.join(output_dir, "availability_summary.csv")
    distribution_path = os.path.join(output_dir, "distribution_stats_raw.csv")
    corr_clin_path = os.path.join(output_dir, "correlation_la_clinical.csv")
    corr_internal_path = os.path.join(output_dir, "correlation_la_internal.csv")
    high_corr_path = os.path.join(output_dir, "high_corr_pairs_la.csv")
    vif_path = os.path.join(output_dir, "vif_report_la.csv")
    candidates_path = os.path.join(output_dir, "la_candidates_for_cox.csv")
    summary_path = os.path.join(output_dir, "analysis_summary.json")

    df_merged.to_csv(merged_path, index=False)
    availability_df.to_csv(availability_path, index=False)
    distribution_df.to_csv(distribution_path, index=False)
    corr_clin_long.to_csv(corr_clin_path, index=False)
    corr_internal.to_csv(corr_internal_path, index=False)
    high_corr_df.to_csv(high_corr_path, index=False)
    vif_df.to_csv(vif_path, index=False)
    candidate_df.to_csv(candidates_path, index=False)

    saved_figures: List[str] = []
    if not no_plots:
        availability_fig = os.path.join(figures_dir, "availability_bar.png")
        plot_availability_bar(
            availability_df,
            availability_fig,
            title=f"左心房参数可用率 - {window_name}",
        )
        saved_figures.append(availability_fig)

        saved_figures.extend(
            plot_distribution_pages(
                df_merged,
                feature_meta,
                source_table="morphology",
                output_dir=figures_dir,
                title_prefix=f"左心房形态学参数原始分布 - {window_name}",
                base_filename="distribution_morphology",
                max_per_page=16,
            )
        )
        saved_figures.extend(
            plot_distribution_pages(
                df_merged,
                feature_meta,
                source_table="kinematic",
                output_dir=figures_dir,
                title_prefix=f"左心房运动学参数原始分布 - {window_name}",
                base_filename="distribution_kinematic",
                max_per_page=16,
            )
        )

        corr_clin_fig = os.path.join(figures_dir, "correlation_la_clinical.png")
        plot_heatmap(
            corr_clin_matrix,
            corr_clin_fig,
            title=f"左心房参数与心功能相关指标的相关性 - {window_name}",
        )
        saved_figures.append(corr_clin_fig)

        corr_morph_fig = os.path.join(figures_dir, "correlation_la_morphology.png")
        plot_heatmap(
            corr_morph_matrix,
            corr_morph_fig,
            title=f"左心房形态学参数内部相关性 - {window_name}",
        )
        saved_figures.append(corr_morph_fig)

        corr_kine_fig = os.path.join(figures_dir, "correlation_la_kinematic.png")
        plot_heatmap(
            corr_kine_matrix,
            corr_kine_fig,
            title=f"左心房运动学参数内部相关性 - {window_name}",
        )
        saved_figures.append(corr_kine_fig)

    summary = {
        "window_name": window_name,
        "n_merged": int(len(df_merged)),
        "n_morph_features_present": int(len(morph_features)),
        "n_kine_features_present": int(len(kine_features)),
        "n_la_candidates_for_cox": int(len(candidate_df)),
        "clinical_indicators_used": clinical_features,
        "missing_expected_echo_fields": [
            "LVEF",
            "diastolic_echo_indices",
        ],
        "saved_figures": saved_figures,
        "output_files": [
            merged_path,
            availability_path,
            distribution_path,
            corr_clin_path,
            corr_internal_path,
            high_corr_path,
            vif_path,
            candidates_path,
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  输出目录：{output_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--clinical_csv",
        default="csv/comorbidity/hfpef_cohort_win_hadm_comorbidity.csv",
        help="单窗口临床表路径（默认：csv/comorbidity/hfpef_cohort_win_hadm_comorbidity.csv）",
    )
    ap.add_argument(
        "--morph_csv",
        default="csv/la_params/processed/la_morphology_wide_raw.csv",
        help="LA 形态学原始宽表路径",
    )
    ap.add_argument(
        "--kine_csv",
        default="csv/la_params/processed/la_kinematic_wide_raw.csv",
        help="LA 运动学原始宽表路径",
    )
    ap.add_argument(
        "--feature_catalog_csv",
        default="csv/la_params/processed/la_feature_catalog.csv",
        help="LA 特征目录路径",
    )
    ap.add_argument(
        "--feature_decisions_csv",
        default="csv/la_params/processed/la_params_feature_decisions.csv",
        help="LA 特征决策表路径",
    )
    ap.add_argument(
        "--qc_csv",
        default="csv/la_params/processed/la_params_qc_filtered.csv",
        help="LA QC 表路径（可留空）",
    )
    ap.add_argument(
        "--output_dir",
        default="csv/la_analysis/hadm",
        help="单窗口分析输出目录",
    )
    ap.add_argument(
        "--window_name",
        default=None,
        help="窗口名称；默认从临床文件名推断",
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
        help=f"VIF 阈值（默认：{_VIF_THRESH}）",
    )
    ap.add_argument(
        "--font_family",
        default=None,
        help="显式指定中文字体名；若启用绘图且字体不存在，将直接报错",
    )
    ap.add_argument(
        "--no_plots",
        action="store_true",
        help="跳过全部图表输出",
    )
    args = ap.parse_args()

    qc_csv = args.qc_csv.strip() if isinstance(args.qc_csv, str) else ""
    qc_csv = qc_csv if qc_csv else None

    run_la_analysis(
        clinical_csv=args.clinical_csv,
        morph_csv=args.morph_csv,
        kine_csv=args.kine_csv,
        feature_catalog_csv=args.feature_catalog_csv,
        feature_decisions_csv=args.feature_decisions_csv,
        output_dir=args.output_dir,
        qc_csv=qc_csv,
        window_name=args.window_name,
        alpha=args.alpha,
        vif_thresh=args.vif_thresh,
        font_family=args.font_family,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
