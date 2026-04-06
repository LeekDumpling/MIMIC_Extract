# -*- coding: utf-8 -*-
"""
Model Evaluation Script — HFpEF Cohort (Step 10)

Purpose
-------
Perform internal validation and comprehensive model evaluation for the final
Cox PH model, including:

  1. **Bootstrap C-index** (Harrell's optimism-correction method, default B=1000)
     Quantifies over-fitting by comparing apparent (training) C-index with the
     model's performance when applied to the original sample.

  2. **Calibration curve** (observed vs. predicted risk at a given time horizon)
     Patients are grouped by predicted risk into quintiles; within each quintile
     the Kaplan–Meier estimator provides the observed event probability.

  3. **Decision Curve Analysis (DCA)**
     Net benefit vs. threshold probability curve — quantifies the clinical
     utility of the model relative to two simple strategies (treat all /
     treat none) at the chosen time horizon.

Output files
------------
  csv/model_eval/{window}_{endpoint}/bootstrap_cindex.json
      Bootstrap concordance summary (apparent, optimism, corrected, 95% CI)

  csv/model_eval/{window}_{endpoint}/calibration.csv
      Group-level calibration table

  csv/model_eval/{window}_{endpoint}/dca.csv
      Net benefit table across thresholds

  csv/model_eval/{window}_{endpoint}/figures/
      bootstrap_cindex.png   — bootstrap C-index distribution histogram
      calibration.png        — observed vs. predicted calibration plot
      dca.png                — decision curve

Usage (run from repo root or utils/ sub-directory)
    python utils/model_evaluation.py
    python utils/model_evaluation.py --window hadm --endpoint 1yr
    python utils/model_evaluation.py --n_bootstrap 1000 --t_eval 365
    python utils/model_evaluation.py --no_plots

References
----------
  Harrell FE et al. (1996) Multivariable prognostic models: issues in
    developing models, evaluating assumptions and adequacy, and measuring
    and reducing errors. Statistics in Medicine, 15(4):361–387.

  Vickers AJ & Elkin EB (2006) Decision curve analysis: a novel method for
    evaluating prediction models. Medical Decision Making, 26(6):565–574.
"""

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Required imports
# ---------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    print("numpy and pandas are required.\n  pip install numpy pandas")
    sys.exit(1)

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
    from lifelines.exceptions import ConvergenceError
except ImportError:
    print("lifelines is required.\n  pip install lifelines")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ---------------------------------------------------------------------------
# Font / display name helpers (shared logic with fit_cox_model.py)
# ---------------------------------------------------------------------------
_DISPLAY_NAMES_CACHE: Optional[Dict[str, str]] = None


def _display_name(col: str) -> str:
    """Return canonical clinical display name for a feature column."""
    global _DISPLAY_NAMES_CACHE
    if _DISPLAY_NAMES_CACHE is None:
        try:
            _utils_dir = os.path.dirname(os.path.abspath(__file__))
            if _utils_dir not in sys.path:
                sys.path.insert(0, _utils_dir)
            from fit_cox_model import FEATURE_DISPLAY_NAMES  # type: ignore
            _DISPLAY_NAMES_CACHE = dict(FEATURE_DISPLAY_NAMES)
        except Exception:
            _DISPLAY_NAMES_CACHE = {}
    if col.endswith("_x_logt"):
        base = col[: -len("_x_logt")]
        return f"{_DISPLAY_NAMES_CACHE.get(base, base)} \u00d7 log(t)"
    return _DISPLAY_NAMES_CACHE.get(col, col)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENDPOINTS: Dict[str, Tuple[str, str, str]] = {
    "30d":  ("event_30d",  "time_30d",  "30-day mortality"),
    "90d":  ("event_90d",  "time_90d",  "90-day mortality"),
    "1yr":  ("event_1yr",  "time_1yr",  "1-year mortality"),
    "any":  ("event_any",  "time_any",  "all-cause mortality"),
}

FALLBACK_PENALIZER = 0.05

# Default time evaluation horizon (days) for calibration + DCA
DEFAULT_T_EVAL: Dict[str, float] = {
    "30d":  30.0,
    "90d":  90.0,
    "1yr":  365.0,
    "any":  365.0,
}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _resolve_path(p: str) -> str:
    if os.path.isabs(p) or os.path.exists(p):
        return p
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, p)
    return candidate if os.path.exists(candidate) else p


# ---------------------------------------------------------------------------
# Data loading (mirrors ph_assumption_test.py)
# ---------------------------------------------------------------------------
def _build_survival_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _utils_dir = os.path.dirname(os.path.abspath(__file__))
    if _utils_dir not in sys.path:
        sys.path.insert(0, _utils_dir)
    from fit_cox_model import HORIZONS  # type: ignore
    for days, suffix in HORIZONS:
        ecol = f"event_{suffix}"
        tcol = f"time_{suffix}"
        if ecol not in df.columns or tcol not in df.columns:
            if days is not None:
                df[ecol] = ((df["os_event"] == 1) & (df["os_days"] <= days)).astype(int)
                df[tcol]  = df["os_days"].clip(upper=days)
            else:
                df[ecol] = df["os_event"].astype(int)
                df[tcol]  = df["os_days"]
    return df


def _load_survival_df(
    window: str,
    survival_dir: str,
    raw_dir: str,
) -> Optional[pd.DataFrame]:
    primary = os.path.join(survival_dir, f"hfpef_cohort_win_{window}_survival.csv")
    if os.path.exists(primary):
        return pd.read_csv(primary)
    fallback = os.path.join(raw_dir, f"hfpef_cohort_win_{window}.csv")
    if os.path.exists(fallback):
        df = pd.read_csv(fallback)
        if "os_event" in df.columns and "os_days" in df.columns:
            return _build_survival_endpoints(df)
    return None


# ---------------------------------------------------------------------------
# Model fitting helper
# ---------------------------------------------------------------------------
def _fit_cox(
    df_model: pd.DataFrame,
    features: List[str],
    event_col: str,
    time_col: str,
    penalizer: float = 0.0,
    strata: Optional[List[str]] = None,
) -> Optional["CoxPHFitter"]:
    """Fit Cox model; falls back to FALLBACK_PENALIZER on convergence failure.

    Parameters
    ----------
    strata : column names to pass as ``strata=`` to lifelines (stratified Cox).
             When provided the model estimates a separate baseline hazard per
             stratum; the violating covariate is absorbed into the strata
             structure rather than as a proportional-effect covariate.
    """
    strata_list: List[str] = strata if strata else []

    def _try(pen: float) -> Optional[CoxPHFitter]:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                cph = CoxPHFitter(penalizer=pen,
                                  baseline_estimation_method="breslow")
                cols = features + strata_list + [time_col, event_col]
                fit_kwargs: Dict = {
                    "duration_col": time_col,
                    "event_col": event_col,
                    "show_progress": False,
                }
                if strata_list:
                    fit_kwargs["strata"] = strata_list
                cph.fit(df_model[cols], **fit_kwargs)
            return cph
        except Exception:
            return None
    return _try(penalizer) or _try(FALLBACK_PENALIZER)


def _prepare_df_model(
    df: pd.DataFrame,
    features: List[str],
    event_col: str,
    time_col: str,
    strata_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop missing, encode binary strings, return clean (df, features).

    ``strata_cols`` are retained in the returned DataFrame so that
    stratified Cox models can access them during fitting and prediction.
    """
    strata_cols = strata_cols or []
    feat = [f for f in features if f in df.columns]
    extra = [c for c in strata_cols if c in df.columns and c not in feat]
    df_m = df[feat + extra + [time_col, event_col]].copy().dropna()
    for col in list(feat):
        if df_m[col].dtype == object:
            uniq = df_m[col].dropna().unique()
            if len(uniq) == 2:
                mapping = {v: i for i, v in enumerate(sorted(uniq))}
                df_m[col] = df_m[col].map(mapping)
            else:
                feat = [f for f in feat if f != col]
    extra = [c for c in strata_cols if c in df_m.columns and c not in feat]
    df_m = df_m[feat + extra + [time_col, event_col]].dropna().reset_index(drop=True)
    return df_m, feat


def _rebuild_strata_columns(
    df: pd.DataFrame,
    violating_vars: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Recreate stratification columns from the original violating covariates."""
    df_m = df.copy()
    strata_cols: List[str] = []

    for var in violating_vars:
        if var not in df_m.columns:
            continue
        is_continuous = df_m[var].nunique(dropna=True) > 5
        if is_continuous:
            strata_col = f"{var}_q4"
            df_m[strata_col] = pd.qcut(
                df_m[var],
                q=4,
                labels=False,
                duplicates="drop",
            ).astype(int)
            strata_cols.append(strata_col)
        else:
            strata_cols.append(var)

    return df_m, strata_cols


# ---------------------------------------------------------------------------
# Bootstrap C-index (Harrell's optimism-correction method)
# ---------------------------------------------------------------------------
def bootstrap_cindex(
    df_model: pd.DataFrame,
    features: List[str],
    event_col: str,
    time_col: str,
    n_boot: int = 1000,
    random_state: int = 42,
    strata: Optional[List[str]] = None,
) -> Dict:
    """
    Compute bootstrap-optimism-corrected concordance index.

    Algorithm (Harrell 1996):
      1. Fit full model → apparent C-index.
      2. For each bootstrap sample b = 1 … B:
         a. Draw sample with replacement (n = original n).
         b. Fit Cox model on bootstrap sample → C_boot_train.
         c. Apply bootstrap model to original data → C_boot_test.
         d. optimism_b = C_boot_train − C_boot_test.
      3. mean_optimism = mean(optimism_b over successful bootstrap runs).
      4. Corrected C-index = apparent − mean_optimism.

    Parameters
    ----------
    df_model     : Clean model DataFrame (features + strata_cols + time + event).
    features     : List of covariate column names.
    event_col    : Binary event indicator column.
    time_col     : Observed time column.
    n_boot       : Number of bootstrap resamples.
    random_state : Seed for reproducibility.
    strata       : Strata column names for a stratified Cox model (optional).
                   When provided each bootstrap resample also fits a stratified
                   model, ensuring the evaluation is consistent with the
                   corrected (stratified) model used in production.

    Returns
    -------
    dict with keys: apparent_cindex, mean_optimism, corrected_cindex,
                    ci_lower_95, ci_upper_95, n_successful.
    """
    rng = np.random.default_rng(random_state)
    n   = len(df_model)

    # Apparent C-index on full data
    cph_full = _fit_cox(df_model, features, event_col, time_col, strata=strata)
    if cph_full is None:
        raise RuntimeError("Full model failed to fit — cannot run bootstrap.")
    apparent = float(cph_full.concordance_index_)

    optimisms: List[float] = []
    boot_train_cindices: List[float] = []

    for _b in range(n_boot):
        idx       = rng.integers(0, n, size=n)
        df_boot   = df_model.iloc[idx].reset_index(drop=True)

        cph_boot = _fit_cox(df_boot, features, event_col, time_col, strata=strata)
        if cph_boot is None:
            continue

        c_train = float(cph_boot.concordance_index_)

        # Evaluate bootstrap model on original data.
        # predict_partial_hazard uses covariate columns only (not strata),
        # so the partial hazard is a consistent measure of relative risk
        # across strata for the purposes of the optimism correction.
        try:
            risk_orig = cph_boot.predict_partial_hazard(
                df_model[features]
            ).values
            c_test = float(
                concordance_index(
                    df_model[time_col].values,
                    -risk_orig,
                    df_model[event_col].values,
                )
            )
        except Exception:
            continue

        optimisms.append(c_train - c_test)
        boot_train_cindices.append(c_train)

    n_ok = len(optimisms)
    mean_opt = float(np.mean(optimisms)) if n_ok > 0 else 0.0
    corrected = apparent - mean_opt

    # 95% percentile CI from the bootstrap training distribution
    ci_lo = float(np.percentile(boot_train_cindices, 2.5)) if n_ok > 0 else np.nan
    ci_hi = float(np.percentile(boot_train_cindices, 97.5)) if n_ok > 0 else np.nan

    return {
        "apparent_cindex":  apparent,
        "mean_optimism":    mean_opt,
        "corrected_cindex": corrected,
        "ci_lower_95":      ci_lo,
        "ci_upper_95":      ci_hi,
        "n_successful":     n_ok,
        "n_total_boot":     n_boot,
    }


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------
def calibration_table(
    cph: "CoxPHFitter",
    df_model: pd.DataFrame,
    features: List[str],
    t_eval: float,
    n_groups: int = 5,
) -> pd.DataFrame:
    """
    Compute calibration data at time ``t_eval``.

    Patients are split into ``n_groups`` equal-sized bins by predicted risk.
    Within each bin the Kaplan–Meier estimator gives the observed event
    probability.

    Returns
    -------
    DataFrame with columns: group, n, pred_risk_mean, obs_risk.
    """
    time_col  = cph.duration_col
    event_col = cph.event_col

    # Predicted survival → predicted risk at t_eval.
    # Pass the full df_model so that stratified Cox models can find their
    # strata columns; lifelines ignores columns not used during fitting.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        surv_fn  = cph.predict_survival_function(
            df_model, times=[t_eval]
        )
    pred_surv = surv_fn.loc[t_eval].values
    pred_risk = 1.0 - pred_surv

    df_cal            = df_model[[time_col, event_col]].copy()
    df_cal["pred_risk"] = pred_risk
    df_cal["group"]     = pd.qcut(
        df_cal["pred_risk"], q=n_groups, labels=False, duplicates="drop"
    )

    rows = []
    for g in sorted(df_cal["group"].dropna().unique()):
        mask  = df_cal["group"] == g
        df_g  = df_cal[mask]
        pred_mean = float(df_g["pred_risk"].mean())

        kmf = KaplanMeierFitter()
        try:
            kmf.fit(df_g[time_col], df_g[event_col])
            km_surv  = float(
                kmf.survival_function_at_times([t_eval]).iloc[0]
            )
            obs_risk = 1.0 - km_surv
        except Exception:
            obs_risk = float("nan")

        rows.append({
            "group":          int(g),
            "n":              int(mask.sum()),
            "pred_risk_mean": pred_mean,
            "obs_risk":       float(obs_risk),
        })

    return pd.DataFrame(rows)


def _plot_calibration(
    cal_df: pd.DataFrame,
    t_eval: float,
    out_path: str,
    window: str = "",
    endpoint: str = "",
) -> None:
    """Save calibration plot: observed vs predicted risk."""
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.scatter(cal_df["pred_risk_mean"], cal_df["obs_risk"],
               s=60, color="#1f77b4", zorder=3, label="Risk quintile")
    ax.plot(cal_df["pred_risk_mean"], cal_df["obs_risk"],
            color="#1f77b4", linewidth=1.5, alpha=0.7)

    # Annotate n per group
    for _, row in cal_df.iterrows():
        ax.annotate(f"n={row['n']}",
                    (row["pred_risk_mean"], row["obs_risk"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_xlabel("Predicted risk", fontsize=11)
    ax.set_ylabel("Observed risk (Kaplan–Meier)", fontsize=11)
    t_label = f"{int(t_eval)}d" if t_eval < 365 else f"{t_eval/365:.0f}yr"
    tag = f" \u2014 {window}/{endpoint}" if window else ""
    ax.set_title(f"Calibration Curve at {t_label}{tag}", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Calibration plot \u2192 {out_path}")


# ---------------------------------------------------------------------------
# Decision Curve Analysis
# ---------------------------------------------------------------------------
def dca_table(
    cph: "CoxPHFitter",
    df_model: pd.DataFrame,
    features: List[str],
    t_eval: float,
    thresh_lo: float = 0.01,
    thresh_hi: float = 0.50,
    n_thresh: int = 100,
) -> pd.DataFrame:
    """
    Decision Curve Analysis at time ``t_eval``.

    Net benefit at threshold pt:
      NB(pt) = TP(pt)/N − FP(pt)/N × pt/(1−pt)

    where TP/FP are defined by predicted risk > pt and binary outcome
    (event before t_eval).  Censored patients with time < t_eval are treated
    as non-events (conservative approach appropriate when censoring is
    non-informative).

    Returns
    -------
    DataFrame with columns: threshold, net_benefit_model,
    net_benefit_treat_all, net_benefit_treat_none.

    Reference
    ---------
    Vickers & Elkin (2006) Medical Decision Making 26:565–574.
    """
    time_col  = cph.duration_col
    event_col = cph.event_col
    n         = len(df_model)

    # Predicted risk at t_eval.
    # Pass the full df_model so that stratified Cox models can find their
    # strata columns; lifelines ignores columns not used during fitting.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        surv_fn   = cph.predict_survival_function(
            df_model, times=[t_eval]
        )
    pred_risk = 1.0 - surv_fn.loc[t_eval].values

    # Binary outcome: event before t_eval
    y = ((df_model[event_col] == 1) &
         (df_model[time_col] <= t_eval)).astype(int).values

    thresholds = np.linspace(thresh_lo, thresh_hi, n_thresh)
    rows = []
    for pt in thresholds:
        if pt >= 1.0:
            continue
        pos_mask = pred_risk > pt
        tp = int((pos_mask & (y == 1)).sum())
        fp = int((pos_mask & (y == 0)).sum())
        nb_model = tp / n - fp / n * pt / (1.0 - pt)

        # Treat-all baseline
        tp_all = int(y.sum())
        fp_all = int((y == 0).sum())
        nb_all = tp_all / n - fp_all / n * pt / (1.0 - pt)

        rows.append({
            "threshold":             float(pt),
            "net_benefit_model":     float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": 0.0,
        })

    return pd.DataFrame(rows)


def _plot_dca(
    dca_df: pd.DataFrame,
    t_eval: float,
    out_path: str,
    window: str = "",
    endpoint: str = "",
) -> None:
    """Save Decision Curve Analysis plot."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(dca_df["threshold"], dca_df["net_benefit_model"],
            color="#1f77b4", linewidth=2, label="Cox model")
    ax.plot(dca_df["threshold"], dca_df["net_benefit_treat_all"],
            color="#ff7f0e", linewidth=1.5, linestyle="--", label="Treat all")
    ax.axhline(0, color="#2ca02c", linewidth=1.5, linestyle=":", label="Treat none")

    ax.set_xlabel("Threshold probability", fontsize=11)
    ax.set_ylabel("Net benefit", fontsize=11)
    t_label = f"{int(t_eval)}d" if t_eval < 365 else f"{t_eval/365:.0f}yr"
    tag = f" \u2014 {window}/{endpoint}" if window else ""
    ax.set_title(f"Decision Curve Analysis at {t_label}{tag}", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(dca_df["threshold"].min(), dca_df["threshold"].max())
    ax.set_ylim(
        max(-0.05, dca_df["net_benefit_model"].min() - 0.02),
        dca_df["net_benefit_model"].max() + 0.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  DCA plot \u2192 {out_path}")


# ---------------------------------------------------------------------------
# Bootstrap distribution histogram
# ---------------------------------------------------------------------------
def _plot_bootstrap_hist(
    boot_summary: Dict,
    n_boot: int,
    out_path: str,
    window: str = "",
    endpoint: str = "",
) -> None:
    """Histogram of bootstrap C-indices with apparent and corrected lines."""
    # We only stored summary statistics, not the raw distribution.
    # Plot a simple bar chart of the key metrics instead.
    fig, ax = plt.subplots(figsize=(5, 4))

    labels  = ["Apparent", "Corrected\n(optimism adj.)"]
    values  = [boot_summary["apparent_cindex"],
               boot_summary["corrected_cindex"]]
    colors  = ["#1f77b4", "#d62728"]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black",
                  linewidth=0.8)

    # Add 95% CI range bracket on the apparent bar (from bootstrap)
    ci_lo = boot_summary.get("ci_lower_95", np.nan)
    ci_hi = boot_summary.get("ci_upper_95", np.nan)
    if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
        ax.errorbar(0, values[0],
                    yerr=[[values[0] - ci_lo], [ci_hi - values[0]]],
                    fmt="none", color="black", capsize=6, linewidth=1.5,
                    label=f"Bootstrap 95% CI\n[{ci_lo:.4f}, {ci_hi:.4f}]")
        ax.legend(fontsize=8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
               label="Random (0.5)")
    ax.set_ylabel("Concordance Index (C-index)", fontsize=10)
    tag = f" \u2014 {window}/{endpoint}" if window else ""
    ax.set_title(
        f"Bootstrap C-index ({n_boot} resamples){tag}\n"
        f"Optimism = {boot_summary['mean_optimism']:.4f}",
        fontsize=10,
    )
    ax.set_ylim(0.45, min(1.0, max(values) + 0.08))
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Bootstrap C-index plot \u2192 {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation pipeline for one model
# ---------------------------------------------------------------------------
def run_evaluation(
    df_model: pd.DataFrame,
    features: List[str],
    event_col: str,
    time_col: str,
    window: str,
    endpoint: str,
    out_dir: str,
    n_boot: int = 1000,
    t_eval: Optional[float] = None,
    no_plots: bool = False,
    n_groups: int = 5,
    random_state: int = 42,
    strata: Optional[List[str]] = None,
) -> Dict:
    """
    Run the full evaluation pipeline for one window × endpoint model.

    Steps:
      1. Bootstrap C-index (Harrell optimism correction).
      2. Calibration curve at ``t_eval``.
      3. Decision Curve Analysis at ``t_eval``.

    Parameters
    ----------
    strata : Strata column names for a stratified Cox model (optional).
             When the PH assumption was corrected via quartile stratification,
             pass the strata columns here so that the evaluation uses exactly
             the same model structure as the corrected production model.

    Returns a summary dict.
    """
    label = f"{window}/{endpoint}"
    if t_eval is None:
        t_eval = DEFAULT_T_EVAL.get(endpoint, 365.0)

    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")

    # ---- 1. Fit full model --------------------------------------------------
    cph = _fit_cox(df_model, features, event_col, time_col, strata=strata)
    if cph is None:
        print(f"  [{label}] Model fit failed. Skipping evaluation.")
        return {"window": window, "endpoint": endpoint, "error": "fit_failed"}

    cindex_apparent = float(cph.concordance_index_)
    print(f"  [{label}] Apparent C-index = {cindex_apparent:.4f}")

    # ---- 2. Bootstrap -------------------------------------------------------
    print(f"  [{label}] Running {n_boot} bootstrap resamples …")
    boot = bootstrap_cindex(
        df_model, features, event_col, time_col,
        n_boot=n_boot, random_state=random_state, strata=strata,
    )
    print(
        f"  [{label}] Bootstrap: apparent={boot['apparent_cindex']:.4f}, "
        f"optimism={boot['mean_optimism']:.4f}, "
        f"corrected={boot['corrected_cindex']:.4f}"
    )
    boot_path = os.path.join(out_dir, "bootstrap_cindex.json")
    with open(boot_path, "w", encoding="utf-8") as f:
        json.dump({**boot, "window": window, "endpoint": endpoint,
                   "t_eval": t_eval, "n_boot": n_boot},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"  [{label}] Bootstrap summary \u2192 {boot_path}")

    if not no_plots and _HAS_MPL:
        _plot_bootstrap_hist(
            boot, n_boot,
            os.path.join(fig_dir, "bootstrap_cindex.png"),
            window=window, endpoint=endpoint,
        )

    # ---- 3. Calibration -----------------------------------------------------
    cal_df = calibration_table(cph, df_model, features, t_eval, n_groups)
    cal_path = os.path.join(out_dir, "calibration.csv")
    cal_df.to_csv(cal_path, index=False)
    print(f"  [{label}] Calibration table (t={t_eval:.0f}d) \u2192 {cal_path}")

    if not no_plots and _HAS_MPL:
        _plot_calibration(
            cal_df, t_eval,
            os.path.join(fig_dir, "calibration.png"),
            window=window, endpoint=endpoint,
        )

    # ---- 4. DCA -------------------------------------------------------------
    dca_df = dca_table(cph, df_model, features, t_eval)
    dca_path = os.path.join(out_dir, "dca.csv")
    dca_df.to_csv(dca_path, index=False)
    print(f"  [{label}] DCA table \u2192 {dca_path}")

    if not no_plots and _HAS_MPL:
        _plot_dca(
            dca_df, t_eval,
            os.path.join(fig_dir, "dca.png"),
            window=window, endpoint=endpoint,
        )

    return {
        "window":   window,
        "endpoint": endpoint,
        "t_eval":   t_eval,
        "n_obs":    len(df_model),
        "n_events": int(df_model[event_col].sum()),
        "apparent_cindex":  boot["apparent_cindex"],
        "corrected_cindex": boot["corrected_cindex"],
        "mean_optimism":    boot["mean_optimism"],
        "ci_lower_95":      boot["ci_lower_95"],
        "ci_upper_95":      boot["ci_upper_95"],
        "n_boot_ok":        boot["n_successful"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cox_summary",
                    default="csv/cox_models/cox_summary.json",
                    help="Path to Cox summary JSON from step 8.")
    ap.add_argument("--tvc_summary",
                    default="csv/cox_models/ph_test/stratified_correction_summary.json",
                    help="Optional path to stratification correction summary JSON "
                         "(produced by ph_assumption_test.py "
                         "--correct_violations). When present, evaluates the "
                         "stratification-corrected model for violating windows. "
                         "Default: csv/cox_models/ph_test/stratified_correction_summary.json")
    ap.add_argument("--survival_dir", default="csv/survival",
                    help="Directory with survival endpoint CSVs.")
    ap.add_argument("--raw_dir", default="csv",
                    help="Fallback directory for raw cohort CSVs.")
    ap.add_argument("--output_dir", default="csv/model_eval",
                    help="Output directory for evaluation results.")
    ap.add_argument("--window", default=None,
                    help="Only evaluate the specified time window (e.g. hadm).")
    ap.add_argument("--endpoint", default=None,
                    choices=list(ENDPOINTS.keys()),
                    help="Only evaluate the specified endpoint.")
    ap.add_argument("--n_bootstrap", type=int, default=1000,
                    help="Number of bootstrap resamples (default: 1000).")
    ap.add_argument("--t_eval", type=float, default=None,
                    help="Time horizon in days for calibration and DCA. "
                         "Defaults to endpoint-specific value "
                         "(30d=30, 90d=90, 1yr=365, any=365).")
    ap.add_argument("--n_groups", type=int, default=5,
                    help="Number of risk groups for calibration (default: 5).")
    ap.add_argument("--no_plots", action="store_true",
                    help="Skip plot generation.")
    ap.add_argument("--min_epv", type=float, default=10.0,
                    help="Minimum EPV to include a model (default: 10).")
    ap.add_argument("--random_state", type=int, default=42,
                    help="Random seed for reproducibility (default: 42).")
    args = ap.parse_args()

    for attr in ("cox_summary", "survival_dir", "raw_dir", "output_dir"):
        setattr(args, attr, _resolve_path(getattr(args, attr)))

    # ---- Load Cox summary --------------------------------------------------
    if not os.path.exists(args.cox_summary):
        print(f"Cox summary not found: {args.cox_summary}")
        print("Run step 8 first:  python utils/fit_cox_model.py")
        sys.exit(1)

    with open(args.cox_summary, "r", encoding="utf-8") as f:
        cox_summaries = json.load(f)

    # ---- Load optional stratification-correction summary -------------------
    tvc_map: Dict[str, Dict] = {}  # key: f"{window}/{endpoint}"
    tvc_summary_path = _resolve_path(args.tvc_summary)
    if os.path.exists(tvc_summary_path):
        with open(tvc_summary_path, "r", encoding="utf-8") as f:
            tvc_list = json.load(f)
        for rec in tvc_list:
            if rec.get("tvc_run"):
                key = f"{rec['window']}/{rec['endpoint']}"
                tvc_map[key] = rec
        print(f"Loaded {len(tvc_map)} stratification-corrected model(s) from "
              f"{tvc_summary_path}")

    # ---- Filter models ------------------------------------------------------
    candidates = [
        s for s in cox_summaries
        if not s.get("skipped", True)
        and s.get("final_features")
        and s.get("endpoint") in ENDPOINTS
    ]

    epv_ok = []
    for s in candidates:
        n_ev = s.get("n_events", 0)
        n_f  = len(s.get("final_features", []))
        epv  = n_ev / n_f if n_f > 0 else float("inf")
        if epv >= args.min_epv:
            epv_ok.append(s)

    if args.window:
        epv_ok = [s for s in epv_ok if s.get("window") == args.window]
    if args.endpoint:
        epv_ok = [s for s in epv_ok if s.get("endpoint") == args.endpoint]

    print(f"Models to evaluate: {len(epv_ok)}")

    all_summaries: List[Dict] = []

    for entry in epv_ok:
        window    = entry["window"]
        endpoint  = entry["endpoint"]
        label     = f"{window}/{endpoint}"
        features  = entry["final_features"]
        event_col, time_col, _ = ENDPOINTS[endpoint]
        t_eval = args.t_eval or DEFAULT_T_EVAL.get(endpoint, 365.0)

        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        df = _load_survival_df(window, args.survival_dir, args.raw_dir)
        if df is None:
            print(f"  Data not found for window '{window}'. Skipping.")
            continue

        df_base = df[df["os_event"].notna()].copy()
        df_base = df_base[
            df_base[event_col].notna() & df_base[time_col].notna()
        ].reset_index(drop=True)

        # Determine features and strata: use stratification-corrected if available.
        # NOTE: The old ``x * log(t)`` TVC interaction approach is intentionally
        # NOT supported here.  That method constructs a predictor from the
        # observed outcome time (``feature * log(time_col)``), which creates
        # direct data leakage — the model can trivially recover the outcome
        # variable from its own predictors, spuriously inflating the C-index
        # by 15–30 percentage points.  The correct approach is quartile
        # stratification, which is what ``ph_assumption_test.py`` now produces.
        tvc_rec = tvc_map.get(label)
        eval_features: List[str] = list(features)
        eval_strata: List[str] = []
        violating_vars: List[str] = []
        df_input = df_base
        if tvc_rec:
            method = tvc_rec.get("correction_method", "")
            if method == "stratification":
                # Violating covariate was moved to strata; use remaining features
                eval_features = list(tvc_rec.get("remaining_features", features))
                eval_strata   = list(tvc_rec.get("strata_cols", []))
                violating_vars = list(tvc_rec.get("violating_covariates", []))
                if eval_strata and not set(eval_strata).issubset(df_input.columns) and violating_vars:
                    df_input, rebuilt_strata = _rebuild_strata_columns(df_input, violating_vars)
                    if rebuilt_strata:
                        eval_strata = rebuilt_strata
                print(f"  Stratification-corrected model: "
                      f"features={eval_features}, strata={eval_strata}")
            else:
                # Unknown correction type (e.g. legacy _x_logt): ignore.
                # Using outcome time as a predictor (endogenous interaction)
                # inflates C-index and is methodologically invalid.
                print(f"  WARNING: Ignoring unrecognised correction method "
                      f"'{method}' for {label} — using base features to avoid "
                      f"data leakage.")

        df_model, eval_features = _prepare_df_model(
            df_input, eval_features, event_col, time_col,
            strata_cols=eval_strata,
        )

        out_dir = os.path.join(args.output_dir, f"{window}_{endpoint}")
        summary = run_evaluation(
            df_model=df_model,
            features=eval_features,
            event_col=event_col,
            time_col=time_col,
            window=window,
            endpoint=endpoint,
            out_dir=out_dir,
            n_boot=args.n_bootstrap,
            t_eval=t_eval,
            no_plots=args.no_plots,
            n_groups=args.n_groups,
            random_state=args.random_state,
            strata=eval_strata,
        )
        all_summaries.append(summary)

    # ---- Write overall summary JSON ----------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nEvaluation summary \u2192 {summary_path}")

    # ---- Print summary table -----------------------------------------------
    print(f"\n{'='*80}")
    print("Model Evaluation Summary")
    print(f"{'='*80}")
    hdr = (f"{'Window+Endpoint':<22} {'n':>5} {'Events':>7} "
           f"{'Apparent C':>11} {'Corrected C':>12} {'Optimism':>10} "
           f"{'95% CI':>18}")
    print(hdr)
    print("-" * 88)
    for s in all_summaries:
        if "error" in s:
            tag = f"{s.get('window','')}/{s.get('endpoint','')}"
            print(f"  {tag:<22} — ({s['error']})")
            continue
        tag  = f"{s['window']}/{s['endpoint']}"
        ci   = f"[{s['ci_lower_95']:.4f}, {s['ci_upper_95']:.4f}]"
        print(
            f"  {tag:<22} {s['n_obs']:>5} {s['n_events']:>7} "
            f"{s['apparent_cindex']:>11.4f} {s['corrected_cindex']:>12.4f} "
            f"{s['mean_optimism']:>10.4f} {ci:>18}"
        )

    print(f"\nDone. Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
