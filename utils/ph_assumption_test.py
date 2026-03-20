# -*- coding: utf-8 -*-
"""
Cox PH Assumption Test Script — HFpEF Cohort (Step 9)

Purpose
-------
Test the proportional hazards (PH) assumption for each Cox model
that satisfies the minimum EPV threshold (events-per-variable >= 10,
per Peduzzi et al. 1995).

The PH assumption is tested using scaled Schoenfeld residuals
(Grambsch & Therneau 1994), implemented in lifelines via
`lifelines.statistics.proportional_hazard_test`.

  - H0: the coefficient for a given covariate is constant over time
    (PH assumption holds).
  - Significant p-value (< 0.05) indicates a time-varying effect
    and potential PH violation for that covariate.

This script re-reads the raw data and re-fits the same models
described in the Cox summary JSON produced by step 8 (fit_cox_model.py).

Output files
------------
  csv/cox_models/ph_test/ph_test_{window}_{endpoint}.csv
      — Per-covariate Schoenfeld test statistic and p-value

  csv/cox_models/ph_test/figures/schoenfeld_{window}_{endpoint}.png
      — Schoenfeld residual time-trend plot for each covariate

  csv/cox_models/ph_test/ph_test_summary.json
      — Overall summary: whether each model passes the global PH test

Usage (run from repo root or utils/ sub-directory)
    python utils/ph_assumption_test.py
    python utils/ph_assumption_test.py --window hadm --endpoint 1yr
    python utils/ph_assumption_test.py --no_plots
"""

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    print("numpy and pandas are required.\n  pip install numpy pandas")
    sys.exit(1)

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    from lifelines.exceptions import ConvergenceError
except ImportError:
    print("lifelines is required.\n  pip install lifelines")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ---------------------------------------------------------------------------
# Font setup (re-use logic from fit_cox_model)
# ---------------------------------------------------------------------------
def _setup_chinese_font() -> bool:
    """Try to configure a CJK font for matplotlib labels."""
    if not _HAS_MPL:
        return False
    _CJK_CANDIDATES = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
        "Heiti SC", "PingFang SC", "Noto Sans CJK SC",
        "Source Han Sans CN", "AR PL UMing CN",
    ]
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False

_CJK_AVAILABLE: bool = _setup_chinese_font()

# ---------------------------------------------------------------------------
# Feature display-name mapping
# ---------------------------------------------------------------------------
# Lazily imported from fit_cox_model.py the first time _display_name() is called.
_DISPLAY_NAMES_CACHE: Optional[Dict[str, str]] = None


def _display_name(col: str) -> str:
    """Return the canonical clinical display name for a feature column.

    Interaction terms produced by the TVC correction (e.g. ``wbc_x_logt``)
    are formatted as ``<base display> × log(t)`` automatically.
    Falls back to the raw column name when the mapping is unavailable.
    """
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

    # Handle TVC interaction terms: e.g. "wbc_x_logt" → "WBC (x10^3/uL) × log(t)"
    if col.endswith("_x_logt"):
        base = col[: -len("_x_logt")]
        return f"{_DISPLAY_NAMES_CACHE.get(base, base)} \u00d7 log(t)"

    return _DISPLAY_NAMES_CACHE.get(col, col)


# ---------------------------------------------------------------------------
# Shared constants (must match fit_cox_model.py)
# ---------------------------------------------------------------------------
MIN_EPV = 10

ENDPOINTS: Dict[str, Tuple[str, str, str]] = {
    "30d":  ("event_30d",  "time_30d",  "30-day mortality"),
    "90d":  ("event_90d",  "time_90d",  "90-day mortality"),
    "1yr":  ("event_1yr",  "time_1yr",  "1-year mortality"),
    "any":  ("event_any",  "time_any",  "all-cause mortality"),
}

FALLBACK_PENALIZER = 0.05

# ---------------------------------------------------------------------------
# Path helper (same logic as fit_cox_model.py)
# ---------------------------------------------------------------------------
def _resolve_path(p: str) -> str:
    if os.path.isabs(p) or os.path.exists(p):
        return p
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, p)
    return candidate if os.path.exists(candidate) else p


# ---------------------------------------------------------------------------
# Data loading (mirrors fit_cox_model.py)
# ---------------------------------------------------------------------------
def _build_survival_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Inline-build derived survival endpoint columns from os_event/os_days."""
    df = df.copy()
    from fit_cox_model import HORIZONS  # reuse the same horizon definitions
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
    """Load survival DataFrame for the given window."""
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
# Refit helper
# ---------------------------------------------------------------------------
def _refit_cox(
    df: pd.DataFrame,
    features: List[str],
    event_col: str,
    time_col: str,
    label: str = "",
) -> Optional[Tuple["CoxPHFitter", pd.DataFrame]]:
    """
    Re-fit Cox PH model (same logic as fit_cox_model.fit_cox).
    Returns (fitted_cph, df_model) or None on failure.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [{label}] Warning: features not in data, skipping: {missing}")
        features = [f for f in features if f in df.columns]
    if not features:
        return None

    df_model = df[features + [time_col, event_col]].copy().dropna()

    # Encode binary string columns
    for col in features:
        if df_model[col].dtype == object:
            uniq = df_model[col].dropna().unique()
            if len(uniq) == 2:
                mapping = {v: i for i, v in enumerate(sorted(uniq))}
                df_model[col] = df_model[col].map(mapping)
            else:
                print(f"  [{label}] Skipping non-binary string column '{col}'")
                features = [f for f in features if f != col]

    df_model = df_model[features + [time_col, event_col]].dropna()

    def _try(pen):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                cph = CoxPHFitter(penalizer=pen,
                                  baseline_estimation_method="breslow")
                cph.fit(df_model, duration_col=time_col,
                        event_col=event_col, show_progress=False)
            return cph
        except Exception:
            return None

    cph = _try(0.0) or _try(FALLBACK_PENALIZER)
    if cph is None:
        print(f"  [{label}] Cox refit failed.")
        return None
    return cph, df_model


# ---------------------------------------------------------------------------
# PH test + plots
# ---------------------------------------------------------------------------
def run_ph_test(
    cph: "CoxPHFitter",
    df_model: pd.DataFrame,
    window: str,
    endpoint: str,
    out_dir: str,
    no_plots: bool = False,
    label_suffix: str = "",
) -> Dict:
    """
    Run Schoenfeld residuals PH test for one window × endpoint combination.

    Saves per-covariate CSV and optional plots:
      - ``covariate_effects_{window}_{endpoint}.png``  — survival curves stratified by covariate
      - ``schoenfeld_residuals_{window}_{endpoint}.png`` — scaled Schoenfeld residuals with LOWESS

    Returns a summary dict including a list of violating covariate names.
    """
    label = f"{window}/{endpoint}"

    try:
        result = proportional_hazard_test(
            cph, df_model, time_transform="rank"
        )
        ph_df = result.summary.copy()
        ph_df.index.name = "feature"
        ph_df = ph_df.reset_index()
        # Standardise column names produced by different lifelines versions
        ph_df.columns = [c.lower().replace(" ", "_") for c in ph_df.columns]
        if "p" not in ph_df.columns:
            p_col = next((c for c in ph_df.columns if c.startswith("p")), None)
            if p_col and p_col != "p":
                ph_df = ph_df.rename(columns={p_col: "p"})

        n_violations = int((ph_df["p"] < 0.05).sum()) if "p" in ph_df.columns else -1
        global_ok = n_violations == 0

        violating = (
            ph_df.loc[ph_df["p"] < 0.05, "feature"].tolist()
            if "p" in ph_df.columns
            else []
        )

        os.makedirs(out_dir, exist_ok=True)
        csv_fname = (
            f"ph_test_{window}_{endpoint}"
            + (f"_{label_suffix}" if label_suffix else "")
            + ".csv"
        )
        csv_path = os.path.join(out_dir, csv_fname)
        ph_df.to_csv(csv_path, index=False)
        print(f"  [{label}] PH test saved \u2192 {csv_path}")
        print(f"  [{label}] Covariates violating PH (p<0.05): "
              f"{n_violations}/{len(ph_df)}"
              + (f" — {violating}" if violating else ""))

        if not no_plots and _HAS_MPL:
            fig_dir = os.path.join(out_dir, "figures")
            _plot_covariate_effects(cph, df_model, ph_df, window, endpoint,
                                    fig_dir)
            _plot_schoenfeld_residuals(cph, df_model, ph_df, window, endpoint,
                                       fig_dir, label_suffix=label_suffix)

        return {
            "window": window, "endpoint": endpoint,
            "n_covariates": len(ph_df),
            "n_violations": n_violations,
            "global_ph_ok": global_ok,
            "violating_covariates": violating,
            "ph_test_run": True,
            "label_suffix": label_suffix,
        }

    except Exception as exc:
        print(f"  [{label}] PH test failed: {exc}")
        return {
            "window": window, "endpoint": endpoint,
            "ph_test_run": False,
            "violating_covariates": [],
            "error": str(exc),
        }


def _plot_covariate_effects(
        cph: "CoxPHFitter",
        df_model: pd.DataFrame,
        ph_df: pd.DataFrame,
        window: str,
        endpoint: str,
        fig_dir: str,
) -> None:
    """Plot survival curves stratified by covariate value (effect plots).

    Continuous variables use 10th / 90th percentiles; binary variables use
    0 / 1.  Display names from the canonical mapping are used in titles and
    legends.
    """
    os.makedirs(fig_dir, exist_ok=True)
    covariates = ph_df["feature"].tolist() if "feature" in ph_df.columns else []
    if not covariates:
        return

    # Build p-value lookup for border colours
    pval_lookup: Dict[str, float] = {}
    if "feature" in ph_df.columns and "p" in ph_df.columns:
        for _, row in ph_df.iterrows():
            pval_lookup[row["feature"]] = float(row["p"])

    n = len(covariates)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, cov in enumerate(covariates):
        ax = axes[idx // ncols][idx % ncols]
        p_val = pval_lookup.get(cov)
        violation = p_val is not None and p_val < 0.05
        display = _display_name(cov)
        p_str = f" (p={p_val:.3f})" if p_val is not None else ""
        flag = " \u2717" if violation else " \u2713"

        try:
            if df_model[cov].dtype in [np.float64, np.int64] and df_model[cov].nunique() > 2:
                low = df_model[cov].quantile(0.10)
                high = df_model[cov].quantile(0.90)
                if low == high:
                    low = df_model[cov].mean() - df_model[cov].std()
                    high = df_model[cov].mean() + df_model[cov].std()
                values = [low, high]
                legend_labels = [
                    f"{display} = {low:.2f}",
                    f"{display} = {high:.2f}",
                ]
            else:
                values = [0, 1]
                legend_labels = [f"{display} = 0", f"{display} = 1"]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                cph.plot_covariate_groups(cov, values=values, ax=ax)
                if ax.get_legend():
                    ax.legend(legend_labels, fontsize=7)

        except Exception:
            ax.text(0.5, 0.5, f"{display}\n(plot unavailable)",
                    ha="center", va="center", transform=ax.transAxes)

        ax.set_title(
            f"{display}{p_str}{flag}",
            fontsize=8,
            color="#d62728" if violation else "black",
        )
        ax.tick_params(labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d62728" if violation else "#cccccc")
            spine.set_linewidth(2.0 if violation else 0.8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    import matplotlib.patches as _mpatches
    fig.legend(
        handles=[
            _mpatches.Patch(color="#1f77b4", label="PH satisfied (p \u2265 0.05)"),
            _mpatches.Patch(color="#d62728", label="PH violated (p < 0.05)"),
        ],
        loc="lower center", ncol=2, fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"Covariate Effect Plots \u2014 {window}/{endpoint}\n"
        "(Refer to Schoenfeld residual plot for formal PH test)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = os.path.join(fig_dir, f"covariate_effects_{window}_{endpoint}.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [{window}/{endpoint}] Covariate effect plot \u2192 {out}")


def _plot_schoenfeld_residuals(
        cph: "CoxPHFitter",
        df_model: pd.DataFrame,
        ph_df: pd.DataFrame,
        window: str,
        endpoint: str,
        fig_dir: str,
        label_suffix: str = "",
) -> None:
    """Plot scaled Schoenfeld residuals vs. time for each covariate.

    A flat LOWESS curve indicates PH holds; a trend indicates a time-varying
    effect.  Subplots with a significant p-value (< 0.05) are highlighted
    with a red border and title.
    """
    os.makedirs(fig_dir, exist_ok=True)

    try:
        resid = cph.compute_residuals(df_model, kind="scaled_schoenfeld")
    except Exception as exc:
        print(f"  [{window}/{endpoint}] Cannot compute Schoenfeld residuals: {exc}")
        return

    event_col = cph.event_col
    time_col  = cph.duration_col

    event_mask = df_model[event_col].astype(bool)
    resid_ev   = resid.loc[event_mask]
    times_ev   = df_model.loc[event_mask, time_col].values

    covariates = resid.columns.tolist()
    if not covariates:
        return

    # p-value lookup
    pval_lookup: Dict[str, float] = {}
    if "feature" in ph_df.columns and "p" in ph_df.columns:
        for _, row in ph_df.iterrows():
            pval_lookup[row["feature"]] = float(row["p"])

    n = len(covariates)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, cov in enumerate(covariates):
        ax   = axes[idx // ncols][idx % ncols]
        p_val     = pval_lookup.get(cov)
        violation = p_val is not None and p_val < 0.05
        display   = _display_name(cov)

        x = times_ev
        y = resid_ev[cov].values if cov in resid_ev.columns else np.full(len(x), np.nan)

        color = "#d62728" if violation else "#1f77b4"
        ax.scatter(x, y, alpha=0.35, s=14, color=color, zorder=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)

        # LOWESS smoother
        if len(x) >= 6:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                frac = min(0.75, 20.0 / len(x))
                smoothed = lowess(y, x, frac=frac, it=0)
                ax.plot(smoothed[:, 0], smoothed[:, 1],
                        color="#d62728" if violation else "#2ca02c",
                        linewidth=2, zorder=3)
            except Exception:
                pass  # statsmodels not installed — skip smooth

        p_str   = f" (p={p_val:.3f})" if p_val is not None else ""
        flag    = " \u2717" if violation else " \u2713"
        ax.set_title(
            f"{display}{p_str}{flag}",
            fontsize=8,
            color="#d62728" if violation else "black",
        )
        ax.set_xlabel("Time (days)", fontsize=7)
        ax.set_ylabel("Schoenfeld\nResidual", fontsize=7)
        ax.tick_params(labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d62728" if violation else "#cccccc")
            spine.set_linewidth(2.0 if violation else 0.8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    import matplotlib.patches as _mpatches
    fig.legend(
        handles=[
            _mpatches.Patch(color="#1f77b4", label="PH satisfied (p \u2265 0.05)"),
            _mpatches.Patch(color="#d62728", label="PH violated (p < 0.05)"),
        ],
        loc="lower center", ncol=2, fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    suffix_str = f" [{label_suffix}]" if label_suffix else ""
    fig.suptitle(
        f"Scaled Schoenfeld Residuals \u2014 {window}/{endpoint}{suffix_str}\n"
        "(Flat LOWESS = PH holds; Trend = violation)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fname = (f"schoenfeld_residuals_{window}_{endpoint}"
             + (f"_{label_suffix}" if label_suffix else "") + ".png")
    out = os.path.join(fig_dir, fname)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [{window}/{endpoint}] Schoenfeld residual plot \u2192 {out}")

# ---------------------------------------------------------------------------
# Time-varying coefficient (TVC) correction
# ---------------------------------------------------------------------------
def _fit_tvc_corrected(
    df: pd.DataFrame,
    features: List[str],
    event_col: str,
    time_col: str,
    violating_vars: List[str],
    label: str = "",
) -> Optional[Tuple["CoxPHFitter", pd.DataFrame, List[str]]]:
    """Fit a corrected Cox model that adds log(t) interaction terms for
    covariates that violated the PH assumption.

    For each violating variable ``x``, an interaction column
    ``x_x_logt = x * log(t)`` is appended to the covariate set.  The main
    effect ``x`` is retained so that the base HR is still interpretable.

    Note
    ----
    Using the observed event/censoring time ``t`` as a proxy for the
    time-axis in the interaction is the "Grambsch–Therneau" approach used
    diagnostically.  The resulting model is **not** a true counting-process
    time-varying model; it is a practical approximation appropriate for
    checking whether a log-linear time trend absorbs the PH violation
    (Grambsch & Therneau 1994, Statistics in Medicine).

    Parameters
    ----------
    df             : DataFrame containing all required columns.
    features       : Original feature list (must be present in df).
    event_col      : Binary event indicator column.
    time_col       : Observed time column.
    violating_vars : Subset of *features* whose PH test was significant.
    label          : Label for console output.

    Returns
    -------
    (fitted_cph, df_model, corrected_features) or None on failure.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [{label}] TVC: skipping missing columns: {missing}")
        features = [f for f in features if f in df.columns]
    if not features:
        return None

    df_model = df[features + [time_col, event_col]].copy().dropna()

    # Encode binary string columns (same as _refit_cox)
    for col in list(features):
        if df_model[col].dtype == object:
            uniq = df_model[col].dropna().unique()
            if len(uniq) == 2:
                mapping = {v: i for i, v in enumerate(sorted(uniq))}
                df_model[col] = df_model[col].map(mapping)
            else:
                features = [f for f in features if f != col]

    df_model = df_model[features + [time_col, event_col]].dropna()

    corrected_features = list(features)
    added: List[str] = []
    for var in violating_vars:
        if var not in df_model.columns:
            continue
        interaction_col = f"{var}_x_logt"
        # log(t) clipped at 0.5 days to avoid log(0) / log(negative)
        df_model[interaction_col] = (
            df_model[var] * np.log(df_model[time_col].clip(lower=0.5))
        )
        if interaction_col not in corrected_features:
            corrected_features.append(interaction_col)
            added.append(interaction_col)

    if not added:
        print(f"  [{label}] TVC: no interaction terms added (violating vars "
              f"not found in features: {violating_vars})")
        return None

    print(f"  [{label}] TVC: added interaction terms: {added}")

    def _try(pen: float) -> Optional["CoxPHFitter"]:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                cph = CoxPHFitter(penalizer=pen,
                                  baseline_estimation_method="breslow")
                cph.fit(df_model[corrected_features + [time_col, event_col]],
                        duration_col=time_col, event_col=event_col,
                        show_progress=False)
            return cph
        except Exception:
            return None

    cph = _try(0.0) or _try(FALLBACK_PENALIZER)
    if cph is None:
        print(f"  [{label}] TVC corrected model fit failed.")
        return None

    return cph, df_model, corrected_features


def run_tvc_correction(
    df_cox: pd.DataFrame,
    features: List[str],
    event_col: str,
    time_col: str,
    window: str,
    endpoint: str,
    violating_vars: List[str],
    cindex_before: float,
    out_dir: str,
    no_plots: bool = False,
) -> Dict:
    """Fit TVC-corrected model, re-run PH test, and return a comparison dict.

    Parameters
    ----------
    df_cox         : Survival DataFrame (pre-filtered for this endpoint).
    features       : Original feature list for the model.
    event_col      : Binary event indicator column.
    time_col       : Observed time column.
    window         : Time-window label (e.g. ``"hadm"``).
    endpoint       : Endpoint label (e.g. ``"1yr"``).
    violating_vars : Covariates that violated PH in the original model.
    cindex_before  : C-index from the original (uncorrected) model.
    out_dir        : Output directory for corrected PH test CSVs and plots.
    no_plots       : If True, skip plot generation.

    Returns
    -------
    Dict with keys: window, endpoint, violating_vars, cindex_before,
    cindex_after, ph_ok_before, ph_ok_after, tvc_features, n_violations_after.
    """
    label = f"{window}/{endpoint}"

    fit_result = _fit_tvc_corrected(
        df_cox, features, event_col, time_col, violating_vars, label
    )
    if fit_result is None:
        return {
            "window": window, "endpoint": endpoint,
            "tvc_run": False, "error": "tvc_fit_failed",
            "violating_covariates": violating_vars,
        }

    cph_corr, df_model_corr, corrected_features = fit_result
    cindex_after = float(cph_corr.concordance_index_)

    print(f"  [{label}] TVC corrected C-index: "
          f"{cindex_before:.4f} \u2192 {cindex_after:.4f}")

    ph_result = run_ph_test(
        cph_corr, df_model_corr, window, endpoint, out_dir,
        no_plots=no_plots, label_suffix="tvc_corrected",
    )

    # Save corrected model coefficients
    try:
        coef_df = cph_corr.summary.copy()
        coef_df.index.name = "feature"
        coef_df = coef_df.reset_index()
        coef_path = os.path.join(out_dir,
                                 f"tvc_corrected_{window}_{endpoint}.csv")
        os.makedirs(out_dir, exist_ok=True)
        coef_df.to_csv(coef_path, index=False)
        print(f"  [{label}] TVC corrected coefficients \u2192 {coef_path}")
    except Exception as exc:
        print(f"  [{label}] Could not save TVC coefficients: {exc}")

    return {
        "window": window, "endpoint": endpoint,
        "tvc_run": True,
        "violating_covariates": violating_vars,
        "tvc_features": corrected_features,
        "cindex_before": cindex_before,
        "cindex_after": cindex_after,
        "ph_ok_before": False,
        "ph_ok_after": ph_result.get("global_ph_ok", False),
        "n_violations_after": ph_result.get("n_violations", -1),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cox_summary",
                    default="csv/cox_models/cox_summary.json",
                    help="Path to Cox summary JSON from step 8 (default: "
                         "csv/cox_models/cox_summary.json)")
    ap.add_argument("--survival_dir", default="csv/survival",
                    help="Directory with survival endpoint CSVs (default: csv/survival)")
    ap.add_argument("--raw_dir", default="csv",
                    help="Fallback directory for raw cohort CSVs (default: csv)")
    ap.add_argument("--output_dir", default="csv/cox_models/ph_test",
                    help="Output directory for PH test results (default: "
                         "csv/cox_models/ph_test)")
    ap.add_argument("--window", default=None,
                    help="Only process specified time window (e.g. hadm)")
    ap.add_argument("--endpoint", default=None,
                    choices=list(ENDPOINTS.keys()),
                    help="Only process specified endpoint")
    ap.add_argument("--no_plots", action="store_true",
                    help="Skip Schoenfeld residual plots")
    ap.add_argument("--min_epv", type=float, default=float(MIN_EPV),
                    help=f"Minimum EPV to include a model (default: {MIN_EPV})")
    ap.add_argument("--correct_violations", action="store_true",
                    help="For models with PH violations, fit TVC-corrected "
                         "model (adds log(t) interaction terms) and re-test PH")
    args = ap.parse_args()

    args.cox_summary  = _resolve_path(args.cox_summary)
    args.survival_dir = _resolve_path(args.survival_dir)
    args.raw_dir      = _resolve_path(args.raw_dir)
    args.output_dir   = _resolve_path(args.output_dir)

    # ---- Load Cox summary --------------------------------------------------
    if not os.path.exists(args.cox_summary):
        print(f"Cox summary JSON not found: {args.cox_summary}")
        print("Run step 8 first:  python utils/fit_cox_model.py")
        sys.exit(1)

    with open(args.cox_summary, "r", encoding="utf-8") as f:
        cox_summaries = json.load(f)

    # Filter to models that are not skipped
    candidates = [
        s for s in cox_summaries
        if not s.get("skipped", True)
        and s.get("final_features")
        and s.get("endpoint") in ENDPOINTS
    ]

    # EPV filter
    epv_ok = []
    epv_fail = []
    for s in candidates:
        n_events  = s.get("n_events", 0)
        n_feats   = len(s.get("final_features", []))
        epv       = n_events / n_feats if n_feats > 0 else float("inf")
        if epv >= args.min_epv:
            epv_ok.append(s)
        else:
            epv_fail.append(s)

    print(f"Cox summary loaded: {len(cox_summaries)} records")
    print(f"Models with EPV >= {args.min_epv}: {len(epv_ok)}")
    if epv_fail:
        print(f"Models skipped (low EPV < {args.min_epv}):")
        for s in epv_fail:
            n_e = s.get("n_events", 0)
            n_f = len(s.get("final_features", []))
            epv = n_e / n_f if n_f else float("inf")
            print(f"  {s['window']}/{s['endpoint']}  "
                  f"events={n_e}, features={n_f}, EPV={epv:.1f}")

    # Window / endpoint filters from CLI
    if args.window:
        epv_ok = [s for s in epv_ok if s.get("window") == args.window]
    if args.endpoint:
        epv_ok = [s for s in epv_ok if s.get("endpoint") == args.endpoint]

    if not epv_ok:
        print("No models remaining after filters. Exiting.")
        return

    # ---- Group by window to avoid re-loading the same CSV ------------------
    from collections import defaultdict
    by_window: Dict[str, list] = defaultdict(list)
    for s in epv_ok:
        by_window[s["window"]].append(s)

    all_ph_summaries = []
    # Store (window, ep, features, event_col, time_col, df_cox,
    #        fitted_cph, cindex, ph_result) for TVC correction
    _refit_store: List[Dict] = []

    for window, entries in sorted(by_window.items()):
        print(f"\n{'='*60}")
        print(f"Window: {window}")
        print(f"{'='*60}")

        df = _load_survival_df(window, args.survival_dir, args.raw_dir)
        if df is None:
            print(f"  Data file not found for window '{window}'. Skipping.")
            for e in entries:
                all_ph_summaries.append({
                    "window": window, "endpoint": e["endpoint"],
                    "ph_test_run": False, "error": "data_file_not_found",
                })
            continue

        # Filter to discharged-alive patients as in fit_cox_model.py
        df_base = df[df["os_event"].notna()].copy()

        for entry in entries:
            ep             = entry["endpoint"]
            event_col, time_col, _ = ENDPOINTS[ep]
            features       = entry["final_features"]
            label          = f"{window}/{ep}"

            df_cox = df_base[
                df_base[event_col].notna() & df_base[time_col].notna()
            ].copy().reset_index(drop=True)

            n_events = int(df_cox[event_col].sum())
            epv      = n_events / len(features) if features else float("inf")
            print(f"\n  [{label}] n={len(df_cox)}, events={n_events}, "
                  f"features={len(features)}, EPV={epv:.1f}")

            fit_result = _refit_cox(df_cox, features, event_col, time_col, label)
            if fit_result is None:
                all_ph_summaries.append({
                    "window": window, "endpoint": ep,
                    "ph_test_run": False, "error": "refit_failed",
                })
                continue

            fitted_cph, df_model = fit_result
            cindex = float(fitted_cph.concordance_index_)
            print(f"  [{label}] Refitted: C-index={cindex:.4f}")

            ph_result = run_ph_test(
                fitted_cph, df_model, window, ep,
                args.output_dir,
                no_plots=args.no_plots,
            )
            ph_result["cindex"] = cindex
            all_ph_summaries.append(ph_result)

            # Store for TVC correction (needed even if PH is OK)
            _refit_store.append({
                "window": window, "endpoint": ep,
                "features": features, "event_col": event_col,
                "time_col": time_col, "df_cox": df_cox,
                "cindex": cindex, "ph_result": ph_result,
            })

    # ---- Write summary JSON ------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "ph_test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_ph_summaries, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nPH test summary JSON -> {summary_path}")

    # ---- TVC correction for violating models --------------------------------
    tvc_summaries: List[Dict] = []
    if args.correct_violations:
        violating_models = [
            s for s in _refit_store
            if s["ph_result"].get("violating_covariates")
        ]
        if violating_models:
            print(f"\n{'='*60}")
            print("TVC Correction — adding log(t) interaction terms")
            print(f"{'='*60}")
            for rec in violating_models:
                viol = rec["ph_result"]["violating_covariates"]
                tvc_dir = os.path.join(args.output_dir, "tvc_corrected")
                tvc_res = run_tvc_correction(
                    df_cox=rec["df_cox"],
                    features=rec["features"],
                    event_col=rec["event_col"],
                    time_col=rec["time_col"],
                    window=rec["window"],
                    endpoint=rec["endpoint"],
                    violating_vars=viol,
                    cindex_before=rec["cindex"],
                    out_dir=tvc_dir,
                    no_plots=args.no_plots,
                )
                tvc_summaries.append(tvc_res)

            # Write TVC summary JSON
            tvc_summary_path = os.path.join(args.output_dir, "tvc_summary.json")
            with open(tvc_summary_path, "w", encoding="utf-8") as f:
                json.dump(tvc_summaries, f, ensure_ascii=False, indent=2,
                          default=str)
            print(f"\nTVC correction summary JSON -> {tvc_summary_path}")
        else:
            print("\nNo models with PH violations — TVC correction skipped.")

    # ---- Print summary table -----------------------------------------------
    print(f"\n{'='*70}")
    print("Cox PH Assumption Test Summary (Schoenfeld residuals)")
    print(f"{'='*70}")
    hdr = (f"{'Window+Endpoint':<22} {'C-index':>8} {'Covariates':>11} "
           f"{'Violations':>11} {'PH OK?':>10}")
    print(hdr)
    print("-" * 66)
    for s in all_ph_summaries:
        tag = f"{s.get('window','')}/{s.get('endpoint','')}"
        ci_str = f"{s['cindex']:.4f}" if "cindex" in s else "—"
        if not s.get("ph_test_run"):
            err = s.get("error", "unknown")
            print(f"  {tag:<22} {ci_str:>8} {'—':>11} {'—':>11} {'—':>10}  ({err})")
        else:
            n_cov  = s.get("n_covariates", "—")
            n_viol = s.get("n_violations", "—")
            ok_str = "YES ✓" if s.get("global_ph_ok") else "NO ✗"
            print(f"  {tag:<22} {ci_str:>8} {str(n_cov):>11} {str(n_viol):>11} {ok_str:>10}")

    if tvc_summaries:
        print(f"\n{'='*70}")
        print("TVC Correction Results")
        print(f"{'='*70}")
        hdr2 = (f"{'Window+Endpoint':<22} {'C-index before':>14} "
                f"{'C-index after':>13} {'PH OK after':>12}")
        print(hdr2)
        print("-" * 63)
        for s in tvc_summaries:
            tag = f"{s.get('window','')}/{s.get('endpoint','')}"
            if not s.get("tvc_run"):
                print(f"  {tag:<22} {'—':>14} {'—':>13} {'—':>12}  "
                      f"({s.get('error', 'unknown')})")
            else:
                cb = f"{s['cindex_before']:.4f}"
                ca = f"{s['cindex_after']:.4f}"
                ok = "YES ✓" if s.get("ph_ok_after") else "NO ✗"
                print(f"  {tag:<22} {cb:>14} {ca:>13} {ok:>12}")

    print(f"\nDone. Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
