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
) -> Dict:
    """
    Run Schoenfeld residuals PH test for one window × endpoint combination.
    Saves per-covariate CSV and optional residual plots.
    Returns a summary dict.
    """
    label = f"{window}/{endpoint}"
    event_col, time_col, _ = ENDPOINTS[endpoint]

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
            # older lifelines uses 'p' directly; newer may rename
            p_col = next((c for c in ph_df.columns if c.startswith("p")), None)
            if p_col and p_col != "p":
                ph_df = ph_df.rename(columns={p_col: "p"})

        n_violations = int((ph_df["p"] < 0.05).sum()) if "p" in ph_df.columns else -1
        global_ok = n_violations == 0

        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"ph_test_{window}_{endpoint}.csv")
        ph_df.to_csv(csv_path, index=False)
        print(f"  [{label}] PH test saved → {csv_path}")
        print(f"  [{label}] Covariates violating PH (p<0.05): "
              f"{n_violations}/{len(ph_df)}")

        if not no_plots and _HAS_MPL:
            _plot_schoenfeld(cph, df_model, ph_df, window, endpoint,
                             os.path.join(out_dir, "figures"))

        return {
            "window": window, "endpoint": endpoint,
            "n_covariates": len(ph_df),
            "n_violations": n_violations,
            "global_ph_ok": global_ok,
            "ph_test_run": True,
        }

    except Exception as exc:
        print(f"  [{label}] PH test failed: {exc}")
        return {
            "window": window, "endpoint": endpoint,
            "ph_test_run": False,
            "error": str(exc),
        }


def _plot_schoenfeld(
        cph: "CoxPHFitter",
        df_model: pd.DataFrame,
        ph_df: pd.DataFrame,
        window: str,
        endpoint: str,
        fig_dir: str,
) -> None:
    """Plot scaled Schoenfeld residuals for each covariate.

    Modified to show meaningful values for continuous variables:
    - For binary/categorical variables: values = [0, 1]
    - For continuous variables: values = [10th percentile, 90th percentile]
    """
    os.makedirs(fig_dir, exist_ok=True)
    covariates = ph_df["feature"].tolist() if "feature" in ph_df.columns else []
    if not covariates:
        return

    n = len(covariates)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, cov in enumerate(covariates):
        ax = axes[idx // ncols][idx % ncols]
        try:
            # Determine appropriate values to plot based on variable type
            if df_model[cov].dtype in [np.float64, np.int64] and df_model[cov].nunique() > 2:
                # Continuous variable: use 10th and 90th percentiles
                low = df_model[cov].quantile(0.10)
                high = df_model[cov].quantile(0.90)
                # Guard against equal percentiles (e.g., constant variable)
                if low == high:
                    low = df_model[cov].mean() - df_model[cov].std()
                    high = df_model[cov].mean() + df_model[cov].std()
                values = [low, high]
                legend_labels = [f"{cov} = {low:.2f}", f"{cov} = {high:.2f}"]
            else:
                # Binary/categorical variable
                values = [0, 1]
                legend_labels = [f"{cov}=0", f"{cov}=1"]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # Plot the covariate groups; the function returns the same axes
                cph.plot_covariate_groups(cov, values=values, ax=ax)
                # Update legend with meaningful labels
                if ax.get_legend():
                    ax.legend(legend_labels, fontsize=8)
                ax.set_title(cov, fontsize=9)
        except Exception as e:
            # Fallback: just annotate that the plot failed
            ax.text(0.5, 0.5, f"{cov}\n(plot unavailable)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cov, fontsize=9)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"Covariate Effect Plots — {window}/{endpoint}\n"
        f"(Use Schoenfeld residuals CSV for formal PH test p-values)",
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(fig_dir, f"schoenfeld_{window}_{endpoint}.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [{window}/{endpoint}] Schoenfeld plot -> {out}")

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
            print(f"  [{label}] Refitted: C-index={fitted_cph.concordance_index_:.4f}")

            ph_result = run_ph_test(
                fitted_cph, df_model, window, ep,
                args.output_dir,
                no_plots=args.no_plots,
            )
            all_ph_summaries.append(ph_result)

    # ---- Write summary JSON ------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "ph_test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_ph_summaries, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nPH test summary JSON -> {summary_path}")

    # ---- Print summary table -----------------------------------------------
    print(f"\n{'='*70}")
    print("Cox PH Assumption Test Summary (Schoenfeld residuals)")
    print(f"{'='*70}")
    hdr = (f"{'Window+Endpoint':<22} {'Covariates':>12} "
           f"{'Violations':>12} {'PH OK?':>8}")
    print(hdr)
    print("-" * 58)
    for s in all_ph_summaries:
        tag = f"{s.get('window','')}/{s.get('endpoint','')}"
        if not s.get("ph_test_run"):
            err = s.get("error", "unknown")
            print(f"  {tag:<22} {'—':>12} {'—':>12} {'—':>8}  ({err})")
        else:
            n_cov  = s.get("n_covariates", "—")
            n_viol = s.get("n_violations", "—")
            ok_str = "YES" if s.get("global_ph_ok") else "NO (see CSV)"
            print(f"  {tag:<22} {str(n_cov):>12} {str(n_viol):>12} {ok_str:>8}")

    print(f"\nDone. Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
