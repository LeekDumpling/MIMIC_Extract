# -*- coding: utf-8 -*-
"""
Cox PH 模型拟合脚本 — HFpEF 队列（步骤 8）

功能
----
读取特征选择汇总 JSON（csv/feature_selection/selection_summary.json）
及生存终点 CSV（csv/survival/hfpef_cohort_win_*_survival.csv），
对每个「时间窗 × 终点」组合中特征选择通过的特征集，拟合全量
无正则化 Cox 比例风险模型（lifelines.CoxPHFitter），
输出 HR 表、模型统计量和汇总 JSON。

计划调整评估
------------
基于特征选择汇总，以下情况无需调整原计划，直接跳过即可：
  - M3 特征数为 0 的组合（30d/90d ICU 窗口）——无特征可建模
  - 事件数 < MIN_EVENTS 的组合（已在特征选择阶段过滤）
其余 7 个非空特征组合（见下方汇总）进入 Cox 拟合：
  48h24h/1yr, 48h24h/any
  48h48h/any
  hadm/30d, hadm/90d, hadm/1yr, hadm/any

模型选择
--------
在 LASSO 已筛选特征基础上拟合全量无正则化 Cox 模型（penalizer=0），
以获得无偏 HR 估计和标准误。
- Breslow 基线风险估计（lifelines 默认）
- 若收敛失败，自动尝试小惩罚因子（0.05）兜底，并在结果中标注
- 对每个模型计算 Harrell C-index（一致性指数）

输出文件
--------
  csv/cox_models/{window}/cox_results_{endpoint}.csv  — HR 结果表
  csv/cox_models/figures/forest_{window}_{endpoint}.png     — Forest plot
  csv/cox_models/figures/baseline_survival_{window}_{endpoint}.png — 基线生存曲线
  csv/cox_models/figures/cindex_summary.png                 — C-index 汇总条形图
  csv/cox_models/cox_summary.json                     — 全组合汇总 JSON

HR 结果表列
-----------
feature, coef, HR, hr_lo95, hr_hi95, se_coef, z, p, concordance,
n_obs, n_events, penalizer, converged

用法（从仓库根目录或 utils/ 子目录均可运行）
    python utils/fit_cox_model.py
    python utils/fit_cox_model.py --summary_json csv/feature_selection/selection_summary.json
    python utils/fit_cox_model.py --window hadm --endpoint any
    python utils/fit_cox_model.py --dry_run
    python utils/fit_cox_model.py --no_plots   # 跳过图形输出
"""

import argparse
import glob
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Any

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
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError
except ImportError:
    raise ImportError(
        "lifelines 是必需依赖。\n请执行：pip install lifelines"
    ) from None

# matplotlib — 可选；不存在时仅禁用可视化
try:
    import matplotlib
    matplotlib.use("Agg")          # 非交互后端，适合服务器/脚本运行
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ---------------------------------------------------------------------------
# 中文字体配置（matplotlib）
# ---------------------------------------------------------------------------

def _setup_chinese_font() -> bool:
    """
    尝试配置 matplotlib 使用系统 CJK 字体。

    按优先级依次尝试常见中文字体（Windows / macOS / Linux）；
    若均不可用则返回 False，调用方回退为英文标签。
    """
    if not _HAS_MPL:
        return False
    from matplotlib import font_manager as _fm
    # 候选字体：Windows / macOS / Linux 常见 CJK 字体
    _candidates = [
        "SimHei",               # Windows 黑体
        "Microsoft YaHei",      # Windows 微软雅黑
        "PingFang SC",          # macOS 苹方
        "Heiti SC",             # macOS 黑体-简
        "STHeiti",              # macOS 华文黑体
        "Noto Sans CJK SC",     # Linux（Google Noto）
        "Noto Sans SC",
        "WenQuanYi Micro Hei",  # Linux 文泉驿
        "Source Han Sans CN",   # Adobe 思源黑体
        "FangSong",             # Windows 仿宋
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


_CJK_AVAILABLE: bool = _setup_chinese_font()


def _t(zh: str, en: str) -> str:
    """根据 CJK 字体可用性返回中文或英文字符串。"""
    return zh if _CJK_AVAILABLE else en

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------

# 终点定义：(event_col, time_col, 描述)
ENDPOINTS: Dict[str, Tuple[str, str, str]] = {
    "30d":  ("event_30d",  "time_30d",  "出院后 30 天死亡"),
    "90d":  ("event_90d",  "time_90d",  "出院后 90 天死亡"),
    "1yr":  ("event_1yr",  "time_1yr",  "出院后 1 年死亡"),
    "any":  ("event_any",  "time_any",  "出院后任意时间死亡（受删失约束）"),
}

# 有限时间截止点（天）及对应列名后缀（用于内联构建衍生终点）
HORIZONS: List[tuple] = [
    (30,   "30d"),
    (90,   "90d"),
    (365,  "1yr"),
    (None, "any"),
]

# 不参与建模的列（与 feature_selection.py 保持一致）
NON_FEATURE_COLS = {
    "subject_id", "hadm_id", "index_study_id", "index_study_datetime",
    "index_admittime", "index_dischtime", "a4c_dicom_filepath",
    "death_date", "last_dischtime", "censor_date",
    "hospital_expire_flag", "died_inhosp", "died_post_dc",
    "days_survived_post_dc", "died_30d", "died_90d", "died_1yr",
    "os_event", "os_days",
    "event_30d", "time_30d",
    "event_90d", "time_90d",
    "event_1yr", "time_1yr",
    "event_any", "time_any",
}

# 兜底惩罚因子（仅收敛失败时使用）
FALLBACK_PENALIZER = 0.05

MIN_EVENTS = 10

# ---------------------------------------------------------------------------
# 变量显示名映射（CSV 列名 → 规范临床名称）
# 图表 y 轴使用此名称，避免直接暴露原始表头
# ---------------------------------------------------------------------------

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    # ── 人口学 ──────────────────────────────────────────────────────────────
    "gender":                      "Sex (0=F, 1=M)",
    "anchor_age":                  "Age (years)",
    "omr_bmi":                     "BMI (kg/m\u00b2)",
    "omr_weight_kg":               "Body Weight (kg)",

    # ── 生命体征（住院）──────────────────────────────────────────────────────
    "heart_rate":                  "Heart Rate (bpm)",
    "sbp":                         "Systolic BP – ICU (mmHg)",
    "dbp":                         "Diastolic BP – ICU (mmHg)",
    "mbp":                         "Mean BP – ICU (mmHg)",
    "omr_sbp":                     "Systolic BP – Outpatient (mmHg)",
    "omr_dbp":                     "Diastolic BP – Outpatient (mmHg)",
    "resp_rate":                   "Respiratory Rate (br/min)",
    "temperature_c":               "Temperature (\u00b0C)",
    "spo2":                        "SpO\u2082 (%)",

    # ── 血常规 ───────────────────────────────────────────────────────────────
    "hemoglobin":                  "Hemoglobin (g/dL)",
    "hematocrit":                  "Hematocrit (%)",
    "wbc":                         "WBC (10\u00b3/\u03bcL)",
    "platelet":                    "Platelet Count (10\u00b3/\u03bcL)",

    # ── 凝血 ─────────────────────────────────────────────────────────────────
    "pt":                          "Prothrombin Time (s)",
    "ptt":                         "Partial Thromboplastin Time (s)",
    "inr":                         "INR",

    # ── 生化（基础代谢）──────────────────────────────────────────────────────
    "sodium":                      "Sodium (mEq/L)",
    "potassium":                   "Potassium (mEq/L)",
    "chloride":                    "Chloride (mEq/L)",
    "bicarbonate":                 "Bicarbonate (mEq/L)",
    "bun":                         "BUN (mg/dL)",
    "creatinine":                  "Creatinine (mg/dL)",
    "glucose_lab":                 "Glucose, Lab (mg/dL)",
    "calcium":                     "Calcium (mg/dL)",
    "albumin":                     "Albumin (g/dL)",
    "aniongap":                    "Anion Gap (mEq/L)",

    # ── 心脏 / 炎症标志物 ───────────────────────────────────────────────────
    "troponin_t":                  "Troponin T (ng/mL)",
    "ntprobnp":                    "NT-proBNP (pg/mL)",
    "crp":                         "CRP (mg/L)",

    # ── 合并症（Charlson）────────────────────────────────────────────────────
    "charlson_score":              "Charlson Comorbidity Index",
    "myocardial_infarct":          "Myocardial Infarction",
    "peripheral_vascular_disease": "Peripheral Vascular Disease",
    "chronic_pulmonary_disease":   "Chronic Pulmonary Disease",
    "renal_disease":               "Renal Disease",
    "mild_liver_disease":          "Mild Liver Disease",
    "severe_liver_disease":        "Severe Liver Disease",
    "malignant_cancer":            "Malignant Cancer",
    "diabetes_with_cc":            "Diabetes with Complications",
    "diabetes_without_cc":         "Diabetes without Complications",
    "cerebrovascular_disease":     "Cerebrovascular Disease",
    "hypertension":                "Hypertension",
    "atrial_fibrillation":         "Atrial Fibrillation",
}


def _display_name(col: str) -> str:
    """返回特征列的规范显示名称（未定义则原样返回列名）。"""
    return FEATURE_DISPLAY_NAMES.get(col, col)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    """
    解析相对路径。

    若路径相对于当前工作目录不存在，则尝试相对于仓库根目录解析，
    允许脚本从 utils/ 子目录直接启动。
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


def _build_survival_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 os_event / os_days 列内联构建衍生生存终点列。

    与 build_survival_endpoint.build_endpoints() 逻辑相同，
    用于在生存 CSV 不可用时从原始队列 CSV 直接读取。
    """
    df = df.copy()
    os_event = pd.to_numeric(df["os_event"], errors="coerce")
    os_days  = pd.to_numeric(df["os_days"],  errors="coerce")
    inhosp_mask = os_event.isna()

    for horizon, suffix in HORIZONS:
        if horizon is None:
            event_col_vals = os_event.copy()
            time_col_vals  = os_days.copy()
        else:
            event_col_vals = pd.Series(np.nan, index=df.index, dtype=float)
            time_col_vals  = pd.Series(np.nan, index=df.index, dtype=float)
            alive_mask = ~inhosp_mask
            event_col_vals[alive_mask] = (
                (os_event[alive_mask] == 1) & (os_days[alive_mask] <= horizon)
            ).astype(float)
            time_col_vals[alive_mask] = os_days[alive_mask].clip(upper=horizon)

        df[f"event_{suffix}"] = event_col_vals
        df[f"time_{suffix}"]  = time_col_vals.round(1)

    return df


def _load_survival_df(
    window: str,
    survival_dir: str,
    fallback_dir: str,
) -> Optional[pd.DataFrame]:
    """
    按优先级加载生存数据 DataFrame。

    1. 尝试 {survival_dir}/hfpef_cohort_win_{window}_survival.csv
    2. 若不存在，尝试 {fallback_dir}/hfpef_cohort_win_{window}.csv
       并内联构建衍生终点列。
    3. 两者均不存在则返回 None。
    """
    survival_path = os.path.join(survival_dir, f"hfpef_cohort_win_{window}_survival.csv")
    if os.path.exists(survival_path):
        df = pd.read_csv(survival_path, low_memory=False)
        return df

    # 回退到原始 CSV
    raw_path = os.path.join(fallback_dir, f"hfpef_cohort_win_{window}.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, low_memory=False)
        # 检查是否已有衍生终点
        if "event_30d" not in df.columns:
            df = _build_survival_endpoints(df)
        return df

    return None


# ---------------------------------------------------------------------------
# 核心拟合函数
# ---------------------------------------------------------------------------

def fit_cox(
    df_cox: pd.DataFrame,
    features: List[str],
    event_col: str,
    time_col: str,
    label: str = "",
) -> Dict:
    """
    在 df_cox 上对给定特征集拟合全量 Cox PH 模型。

    优先无惩罚拟合（penalizer=0）；若收敛失败，
    自动改用 penalizer=FALLBACK_PENALIZER 并在结果中标注。

    返回包含以下字段的字典：
      coef_df      : pd.DataFrame — HR 结果表
      concordance  : float
      n_obs        : int
      n_events     : int
      penalizer    : float
      converged    : bool
      error        : str（仅拟合彻底失败时非空）
    """
    cols = features + [time_col, event_col]
    missing_feats = [c for c in features if c not in df_cox.columns]
    if missing_feats:
        print(f"  [{label}] 警告：以下特征在数据中不存在，将跳过：{missing_feats}")
        features = [f for f in features if f in df_cox.columns]
        if not features:
            return {
                "coef_df": pd.DataFrame(),
                "concordance": np.nan,
                "n_obs": len(df_cox),
                "n_events": int(pd.to_numeric(df_cox.get(event_col), errors="coerce").sum()),
                "penalizer": 0.0,
                "converged": False,
                "error": f"所有特征均不在数据中: {missing_feats}",
            }

    df_model = df_cox[features + [time_col, event_col]].copy().dropna()

    # 将字符串/分类列编码为数值（仅二值列；不改原 df）
    for col in features:
        if df_model[col].dtype == object:
            uniq = df_model[col].dropna().unique()
            if len(uniq) == 2:
                # 将两个类别映射为 0/1（字典序较大的为 1）
                mapping = {v: i for i, v in enumerate(sorted(uniq))}
                df_model[col] = df_model[col].map(mapping)
                print(f"  [{label}] 编码字符串列 '{col}'：{mapping}")
            else:
                print(f"  [{label}] 警告：列 '{col}' 为非数值且非二值，跳过该列")
                features = [f for f in features if f != col]

    df_model = df_model[features + [time_col, event_col]].dropna()
    n_obs    = len(df_model)
    n_events = int(df_model[event_col].sum())

    def _try_fit(penalizer: float) -> Tuple[Optional[CoxPHFitter], bool]:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                cph = CoxPHFitter(penalizer=penalizer,
                                  baseline_estimation_method="breslow")
                cph.fit(df_model,
                        duration_col=time_col,
                        event_col=event_col,
                        show_progress=False)
            return cph, True
        except (ConvergenceError, Exception):
            return None, False

    # 首选：无惩罚
    cph, ok = _try_fit(0.0)
    penalizer_used = 0.0
    converged = ok

    # 兜底：小惩罚
    if not ok:
        print(f"  [{label}] 无惩罚拟合失败，改用 penalizer={FALLBACK_PENALIZER}")
        cph, ok = _try_fit(FALLBACK_PENALIZER)
        penalizer_used = FALLBACK_PENALIZER
        converged = ok

    if cph is None:
        return {
            "coef_df": pd.DataFrame(),
            "concordance": np.nan,
            "n_obs": n_obs,
            "n_events": n_events,
            "penalizer": penalizer_used,
            "converged": False,
            "error": "Cox 模型收敛失败（含兜底 penalizer）",
        }

    # 整理系数表
    summary = cph.summary.copy().reset_index()
    summary = summary.rename(columns={
        "covariate":            "feature",
        "coef":                 "coef",
        "exp(coef)":            "HR",
        "exp(coef) lower 95%":  "hr_lo95",
        "exp(coef) upper 95%":  "hr_hi95",
        "se(coef)":             "se_coef",
        "z":                    "z",
        "p":                    "p",
    })
    keep_cols = ["feature", "coef", "HR", "hr_lo95", "hr_hi95",
                 "se_coef", "z", "p"]
    for col in keep_cols:
        if col not in summary.columns:
            summary[col] = np.nan
    summary = summary[keep_cols].copy()
    summary["concordance"] = cph.concordance_index_
    summary["n_obs"]       = n_obs
    summary["n_events"]    = n_events
    summary["penalizer"]   = penalizer_used
    summary["converged"]   = converged

    return {
        "coef_df":    summary,
        "concordance": cph.concordance_index_,
        "n_obs":       n_obs,
        "n_events":    n_events,
        "penalizer":   penalizer_used,
        "converged":   converged,
        "cph_obj":     cph,
        "error":       "",
    }


# ---------------------------------------------------------------------------
# 可视化函数
# ---------------------------------------------------------------------------

def plot_forest(
    coef_df: "pd.DataFrame",
    window: str,
    endpoint: str,
    out_path: str,
    concordance: float,
) -> None:
    """
    绘制 Forest Plot（HR ± 95% CI）并保存为 PNG。

    特征按 HR 从大到小排列；p < 0.05 的特征以红色显示，其余用蓝色。
    """
    if not _HAS_MPL or coef_df.empty:
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        _plot_forest_inner(coef_df, window, endpoint, out_path, concordance)


def _plot_forest_inner(
    coef_df: "pd.DataFrame",
    window: str,
    endpoint: str,
    out_path: str,
    concordance: float,
) -> None:
    """内部绘图逻辑（已在 warnings.catch_warnings 上下文中调用）。"""
    df = coef_df.sort_values("HR", ascending=True).reset_index(drop=True)
    n = len(df)

    # 将原始列名替换为规范显示名称
    display_labels = df["feature"].map(_display_name).tolist()

    fig_h = max(3.5, 0.5 * n + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y_pos = np.arange(n)
    colors = ["#d62728" if p < 0.05 else "#1f77b4"
              for p in df["p"].fillna(1.0)]

    # 水平误差线（HR 点 + 95%CI）
    for i, row in df.iterrows():
        lo = row["hr_lo95"] if pd.notna(row["hr_lo95"]) else row["HR"]
        hi = row["hr_hi95"] if pd.notna(row["hr_hi95"]) else row["HR"]
        ax.plot([lo, hi], [y_pos[i], y_pos[i]], color=colors[i],
                linewidth=1.8, zorder=2)
        ax.scatter(row["HR"], y_pos[i], color=colors[i],
                   s=50, zorder=3)

    # HR=1 参考线
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)

    # 轴标签（中文优先，无 CJK 字体时退回英文）
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.set_xlabel(_t("风险比 HR（95% CI）", "Hazard Ratio HR (95% CI)"), fontsize=10)

    n_obs    = int(df["n_obs"].iloc[0])
    n_events = int(df["n_events"].iloc[0])
    ax.set_title(
        f"Forest Plot — {window}/{endpoint}\n"
        f"C-index = {concordance:.4f}  |  "
        + _t(f"n = {n_obs}，事件 = {n_events}",
             f"n = {n_obs},  events = {n_events}"),
        fontsize=10,
    )

    # 图例
    sig_patch   = mpatches.Patch(color="#d62728",
                                 label=_t("p < 0.05（显著）", "p < 0.05"))
    nosig_patch = mpatches.Patch(color="#1f77b4",
                                 label=_t("p ≥ 0.05（不显著）", "p \u2265 0.05"))
    ax.legend(handles=[sig_patch, nosig_patch], fontsize=8,
              loc="lower right")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Forest plot \u2192 {out_path}")


def plot_baseline_survival(
    cph: Any,
    window: str,
    endpoint: str,
    out_path: str,
) -> None:
    """
    绘制 Cox 模型的基线生存函数 S₀(t) 并保存为 PNG。
    """
    if not _HAS_MPL or cph is None:
        return

    try:
        bsf = cph.baseline_survival_
        if bsf is None or bsf.empty:
            return
    except Exception:
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        fig, ax = plt.subplots(figsize=(6, 4))
        t = bsf.index.values
        s = bsf.iloc[:, 0].values

        ax.step(t, s, where="post", color="#2ca02c", linewidth=2)
        ax.fill_between(t, s, step="post", alpha=0.12, color="#2ca02c")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(_t("随访时间（天）", "Follow-up Time (days)"), fontsize=10)
        ax.set_ylabel(_t("基线生存概率 S\u2080(t)", "Baseline Survival S\u2080(t)"), fontsize=10)
        ax.set_title(
            _t(f"基线生存函数 — {window}/{endpoint}",
               f"Baseline Survival Function — {window}/{endpoint}"),
            fontsize=10,
        )
        ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    print(f"  \u57fa\u7ebf\u751f\u5b58\u66f2\u7ebf \u2192 {out_path}")


def plot_cindex_summary(
    summaries: List[Dict],
    out_path: str,
) -> None:
    """
    绘制各 window×endpoint 组合的 C-index 条形图并保存。

    仅包含 skipped=False 的组合。
    """
    if not _HAS_MPL:
        return

    rows = [s for s in summaries if not s.get("skipped")]
    if not rows:
        return

    labels = [f"{s['window']}/{s['endpoint']}" for s in rows]
    c_vals = [s.get("concordance", float("nan")) for s in rows]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
        x = np.arange(len(labels))
        bars = ax.bar(x, c_vals, color="#ff7f0e", edgecolor="white", width=0.6)

        for bar, val in zip(bars, c_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.005, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=8)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0,
                   label=_t("随机基准（0.5）", "Random (0.5)"))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Harrell C-index", fontsize=10)
        ax.set_title(
            _t("Cox PH 模型 C-index 汇总", "Cox PH Model — C-index Summary"),
            fontsize=11,
        )
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    print(f"\nC-index \u6c47\u603b\u56fe \u2192 {out_path}")


# ---------------------------------------------------------------------------
# 每个 window × endpoint 的处理入口
# ---------------------------------------------------------------------------

def process_combination(
    df: pd.DataFrame,
    window: str,
    endpoint: str,
    final_features: List[str],
    output_dir: str,
    dry_run: bool = False,
    no_plots: bool = False,
) -> Dict:
    """
    对单个时间窗 + 终点组合拟合 Cox PH 模型，写出结果文件。

    返回包含关键统计的 dict（供 JSON 汇总使用）。
    """
    event_col, time_col, ep_desc = ENDPOINTS[endpoint]
    label = f"{window}/{endpoint}"

    # 过滤：仅院外患者且当前终点非缺失
    df_cox = df[df["os_event"].notna()].copy()
    df_cox = df_cox[df_cox[event_col].notna() & df_cox[time_col].notna()].copy()
    df_cox = df_cox.reset_index(drop=True)

    n_total  = len(df_cox)
    n_events = int(df_cox[event_col].sum())

    print(f"\n  [{label}] n={n_total}, 事件={n_events}, 特征数={len(final_features)}")

    if n_events < MIN_EVENTS:
        print(f"  [{label}] 跳过：事件数 < {MIN_EVENTS}")
        return {
            "window": window, "endpoint": endpoint, "endpoint_desc": ep_desc,
            "n_total": n_total, "n_events": n_events,
            "skipped": True, "reason": "too_few_events",
            "final_features": final_features,
        }

    if not final_features:
        print(f"  [{label}] 跳过：无最终特征（特征选择 M3=0）")
        return {
            "window": window, "endpoint": endpoint, "endpoint_desc": ep_desc,
            "n_total": n_total, "n_events": n_events,
            "skipped": True, "reason": "no_selected_features",
            "final_features": [],
        }

    print(f"  [{label}] 拟合 Cox PH 模型（特征：{final_features}）")
    result = fit_cox(df_cox, final_features, event_col, time_col, label=label)

    if result.get("error"):
        print(f"  [{label}] 错误：{result['error']}")
        return {
            "window": window, "endpoint": endpoint, "endpoint_desc": ep_desc,
            "n_total": n_total, "n_events": n_events,
            "skipped": True, "reason": result["error"],
            "final_features": final_features,
        }

    print(f"  [{label}] 收敛={result['converged']}, "
          f"penalizer={result['penalizer']}, "
          f"C-index={result['concordance']:.4f}")

    if not dry_run and not result["coef_df"].empty:
        win_dir = os.path.join(output_dir, window)
        os.makedirs(win_dir, exist_ok=True)

        # CSV 结果表
        out_path = os.path.join(win_dir, f"cox_results_{endpoint}.csv")
        result["coef_df"].to_csv(out_path, index=False)
        print(f"  [{label}] 结果已写出 → {out_path}")

        # 可视化
        if not no_plots and _HAS_MPL:
            fig_dir = os.path.join(output_dir, "figures")
            os.makedirs(fig_dir, exist_ok=True)

            plot_forest(
                result["coef_df"],
                window, endpoint,
                os.path.join(fig_dir, f"forest_{window}_{endpoint}.png"),
                result["concordance"],
            )
            plot_baseline_survival(
                result.get("cph_obj"),
                window, endpoint,
                os.path.join(fig_dir, f"baseline_survival_{window}_{endpoint}.png"),
            )

    return {
        "window":        window,
        "endpoint":      endpoint,
        "endpoint_desc": ep_desc,
        "n_total":       n_total,
        "n_events":      n_events,
        "skipped":       False,
        "concordance":   round(float(result["concordance"]), 4),
        "penalizer":     result["penalizer"],
        "converged":     result["converged"],
        "final_features": final_features,
        "n_features":    len(final_features),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--summary_json", default="csv/feature_selection/selection_summary.json",
                    help="特征选择汇总 JSON 路径（默认：csv/feature_selection/selection_summary.json）")
    ap.add_argument("--survival_dir", default="csv/survival",
                    help="生存终点 CSV 所在目录（默认：csv/survival）")
    ap.add_argument("--raw_dir", default="csv",
                    help="原始队列 CSV 目录（survival_dir 不存在时的回退，默认：csv）")
    ap.add_argument("--output_dir", default="csv/cox_models",
                    help="Cox 模型结果输出目录（默认：csv/cox_models）")
    ap.add_argument("--endpoint", default=None,
                    choices=list(ENDPOINTS.keys()),
                    help="仅处理指定终点（默认：全部）")
    ap.add_argument("--window", default=None,
                    help="仅处理指定时间窗关键词（如 hadm、48h24h）")
    ap.add_argument("--dry_run", action="store_true",
                    help="仅打印摘要，不写出任何文件")
    ap.add_argument("--no_plots", action="store_true",
                    help="跳过可视化图形输出（默认：生成图形）")
    args = ap.parse_args()

    args.summary_json = _resolve_path(args.summary_json)
    args.survival_dir = _resolve_path(args.survival_dir)
    args.raw_dir      = _resolve_path(args.raw_dir)
    args.output_dir   = _resolve_path(args.output_dir)

    # ---- 读取特征选择汇总 ------------------------------------------------
    if not os.path.exists(args.summary_json):
        print(f"特征选择汇总 JSON 不存在：{args.summary_json}")
        print("请先运行：python utils/feature_selection.py")
        sys.exit(1)

    with open(args.summary_json, "r", encoding="utf-8") as f:
        all_fs = json.load(f)

    print(f"读入特征选择汇总：{len(all_fs)} 条记录")

    # ---- 过滤 window / endpoint ------------------------------------------
    if args.window:
        all_fs = [s for s in all_fs if s.get("window") == args.window]
    if args.endpoint:
        all_fs = [s for s in all_fs if s.get("endpoint") == args.endpoint]

    if not all_fs:
        print("过滤后无匹配记录，退出。")
        return

    # ---- 按 window 分组读取 CSV ------------------------------------------
    # 将 all_fs 按 window 分组，减少重复读取
    from collections import defaultdict
    by_window: Dict[str, list] = defaultdict(list)
    for entry in all_fs:
        by_window[entry["window"]].append(entry)

    all_summaries = []

    for window, entries in sorted(by_window.items()):
        print(f"\n{'='*60}")
        print(f"时间窗：{window}")
        print(f"{'='*60}")

        df = _load_survival_df(window, args.survival_dir, args.raw_dir)
        if df is None:
            print(f"  找不到时间窗 '{window}' 的数据文件，跳过。")
            for e in entries:
                all_summaries.append({
                    "window": window, "endpoint": e.get("endpoint"),
                    "skipped": True, "reason": "data_file_not_found",
                    "final_features": [],
                })
            continue

        print(f"  读入 {len(df)} 行 × {len(df.columns)} 列")

        for entry in entries:
            ep = entry.get("endpoint")
            if ep not in ENDPOINTS:
                continue
            if entry.get("skipped"):
                print(f"\n  [{window}/{ep}] 特征选择已跳过（{entry.get('reason', '')}），跳过 Cox 拟合")
                all_summaries.append({
                    "window": window, "endpoint": ep,
                    "skipped": True, "reason": f"fs_skipped:{entry.get('reason', '')}",
                    "final_features": [],
                })
                continue

            final_features = entry.get("final_features", [])
            summary = process_combination(
                df, window, ep, final_features,
                args.output_dir,
                dry_run=args.dry_run,
                no_plots=args.no_plots,
            )
            all_summaries.append(summary)

    # ---- 写出全局 JSON 汇总 ----------------------------------------------
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "cox_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n全局汇总 JSON → {summary_path}")

        # C-index 汇总条形图
        if not args.no_plots and _HAS_MPL:
            fig_dir = os.path.join(args.output_dir, "figures")
            os.makedirs(fig_dir, exist_ok=True)
            plot_cindex_summary(
                all_summaries,
                os.path.join(fig_dir, "cindex_summary.png"),
            )

    # ---- 打印汇总表 -------------------------------------------------------
    print(f"\n{'='*60}")
    print("Cox PH 模型拟合汇总")
    print(f"{'='*60}")
    hdr = f"{'窗口+终点':<20} {'n':>6} {'事件':>6} {'特征数':>6} {'C-index':>9} {'收敛':>5} 状态"
    print(hdr)
    print("-" * 70)
    for s in all_summaries:
        tag = f"{s.get('window','')}/{s.get('endpoint','')}"
        if s.get("skipped"):
            reason = s.get("reason", "")
            print(f"  {tag:<20} {'—':>6} {'—':>6} {'—':>6} {'—':>9} {'—':>5} 跳过（{reason}）")
        else:
            c_idx = f"{s.get('concordance', float('nan')):.4f}"
            conv  = "✓" if s.get("converged") else "✗"
            pen   = f"pen={s.get('penalizer', 0)}" if s.get("penalizer", 0) > 0 else ""
            print(f"  {tag:<20} {s.get('n_total',0):>6} {s.get('n_events',0):>6} "
                  f"{s.get('n_features',0):>6} {c_idx:>9} {conv:>5} {pen}")

    print(f"\n完成。结果已保存至：{args.output_dir}/")


if __name__ == "__main__":
    main()
