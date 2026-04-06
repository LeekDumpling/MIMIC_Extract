# -*- coding: utf-8 -*-
"""
ph_viz.py — Centralised visualisation helpers for the HFpEF Cox pipeline

All matplotlib-based plot functions are collected here so that
``fit_cox_model.py`` and ``ph_assumption_test.py`` share a single
implementation and downstream notebooks can re-use the same code without
duplication.

Public API
----------
setup_chinese_font()
plot_covariate_effects(cph, df_model, ph_df, window, endpoint, fig_dir, ...)
plot_schoenfeld_residuals(cph, df_model, ph_df, window, endpoint, fig_dir, ...)
plot_forest(coef_df, window, endpoint, out_path, concordance, ...)
plot_baseline_survival(cph, window, endpoint, out_path, ...)
plot_cindex_summary(summaries, out_path, ...)
"""

from __future__ import annotations

import os
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import matplotlib.ticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ---------------------------------------------------------------------------
# CJK font setup (shared)
# ---------------------------------------------------------------------------

_CJK_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans CN",
    "FangSong",
    "Heiti SC",
    "STHeiti",
    "WenQuanYi Micro Hei",
    "AR PL UMing CN",
]


def setup_chinese_font(
    font_family: Optional[str] = None,
    strict: bool = False,
) -> bool:
    """Configure matplotlib to use a Chinese-capable font.

    Parameters
    ----------
    font_family:
        Optional explicit font family name. When provided, only this font is
        accepted.
    strict:
        When True, raise ``RuntimeError`` if no usable font is available.

    Returns
    -------
    bool
        True when a font was successfully configured, otherwise False.
    """
    if not _HAS_MPL:
        if strict:
            raise RuntimeError("matplotlib is unavailable; cannot configure Chinese fonts.")
        return False
    from matplotlib import font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    candidates = [font_family] if font_family else list(_CJK_CANDIDATES)
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = (
                [name] + list(matplotlib.rcParams.get("font.sans-serif", []))
            )
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    if strict:
        if font_family:
            raise RuntimeError(f"Requested Chinese font is not installed: {font_family}")
        raise RuntimeError(
            "No supported Chinese font was found. Install one of: "
            + ", ".join(_CJK_CANDIDATES[:7])
        )
    return False


CJK_AVAILABLE: bool = setup_chinese_font()


# ---------------------------------------------------------------------------
# Covariate effect plots
# ---------------------------------------------------------------------------

def plot_covariate_effects(
    cph,
    df_model: pd.DataFrame,
    ph_df: pd.DataFrame,
    window: str,
    endpoint: str,
    fig_dir: str,
    label_suffix: str = "",
    display_name_fn: Optional[Callable[[str], str]] = None,
) -> None:
    """Plot partial-effects survival curves stratified by covariate value.

    Design decisions for binary vs. continuous variables
    ----------------------------------------------------
    **Continuous variables** (``nunique > 5``): Plotted at the 10th and 90th
    percentile of the observed range.  The lifelines ``plot_baseline``
    (all-covariates-at-median) is shown as a reference curve, labelled
    "Baseline (mean)".

    **Binary / low-cardinality variables** (``nunique <= 5``): Plotted at the
    actual observed category values retrieved from the data — *not* the
    hardcoded ``[0, 1]``.  This matters when features have been z-score
    normalised: after normalisation the binary categories sit at some
    ``{z_lo, z_hi}`` and the lifelines central value (median) equals
    ``z_lo`` or ``z_hi`` — always one of the two actual values —- so the
    baseline curve would perfectly coincide with one value curve every time.
    The fix is to suppress the baseline (``plot_baseline=False``) for binary
    variables; the median of a binary indicator is not a real clinical state
    and adding it creates visual ambiguity rather than information.

    Bug in baseline linestyle detection (prior versions)
    ----------------------------------------------------
    The lifelines ``plot_partial_effects_on_outcome`` function draws the
    baseline curve with ``ls=":"`` (dotted) and ``color="k"`` (black).
    Previous code looked for ``ls="--"`` (dashed) and therefore never
    detected the baseline, causing it to be silently mislabelled as the
    first value curve.  This version checks both ``":"``/``"dotted"`` and
    ``"--"``/``"dashed"`` for robustness across lifelines versions.

    Parameters
    ----------
    cph : fitted CoxPHFitter
    df_model : DataFrame used to fit the model (must contain all feature columns)
    ph_df : DataFrame with columns ``feature`` and ``p`` (Schoenfeld test output)
    window : time-window label (e.g. ``"hadm"``)
    endpoint : endpoint label (e.g. ``"1yr"``)
    fig_dir : output directory for the PNG file
    label_suffix : optional suffix appended to the filename (e.g. "stratified")
    display_name_fn : optional callable mapping column name → display label
    """
    if not _HAS_MPL:
        return
    os.makedirs(fig_dir, exist_ok=True)

    if display_name_fn is None:
        display_name_fn = lambda x: x  # noqa: E731

    covariates = ph_df["feature"].tolist() if "feature" in ph_df.columns else []
    if not covariates:
        return

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
        display = display_name_fn(cov)
        p_str = f" (p={p_val:.3f})" if p_val is not None else ""
        flag = " (FAIL)" if violation else " (OK)"

        try:
            is_continuous = (
                df_model[cov].dtype in [np.float64, np.int64]
                and df_model[cov].nunique() > 5
            )

            if is_continuous:
                low = df_model[cov].quantile(0.10)
                high = df_model[cov].quantile(0.90)
                if low == high:
                    low = df_model[cov].mean() - df_model[cov].std()
                    high = df_model[cov].mean() + df_model[cov].std()
                values = [low, high]
                value_labels = [
                    f"{display} P10 ({low:.2f})",
                    f"{display} P90 ({high:.2f})",
                ]
                # For continuous variables, keep the all-covariate-means
                # baseline so the reader can see the reference trajectory.
                use_baseline = True
            else:
                # Binary / low-cardinality variable.
                # -------------------------------------------------------
                # Use the ACTUAL observed values from the data, not the
                # hardcoded [0, 1].  After z-score normalisation, binary
                # categories sit at non-zero positions (e.g. {-0.66, 1.53}).
                # The lifelines central value (median) of a binary column is
                # always exactly one of those two values, so the baseline
                # curve ALWAYS coincides with one of the value curves — making
                # two lines indistinguishable regardless of the variable's
                # prevalence.  Fix: suppress the baseline entirely.
                actual_vals = sorted(df_model[cov].dropna().unique().tolist())
                if len(actual_vals) >= 2:
                    lo, hi = actual_vals[0], actual_vals[-1]
                else:
                    lo, hi = 0.0, 1.0
                values = [lo, hi]
                use_baseline = False  # baseline not meaningful for binary vars

                # Choose human-readable labels.
                tol = 0.02
                if abs(lo) <= tol and abs(hi - 1.0) <= tol:
                    # Unnormalised binary: values are exactly 0 and 1.
                    value_labels = [f"{display} = No (0)",
                                    f"{display} = Yes (1)"]
                else:
                    # Normalised binary: show the category indices.
                    value_labels = [f"{display} = cat-0",
                                    f"{display} = cat-1"]

            n_lines_before = len(ax.lines)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # plot_baseline kwarg supported in lifelines >= 0.25;
                # the deprecated alias plot_covariate_groups is called here
                # for compatibility but the real function is
                # plot_partial_effects_on_outcome.
                cph.plot_covariate_groups(
                    cov, values=values, ax=ax, plot_baseline=use_baseline,
                )

            # Rebuild the per-axes legend.
            # lifelines draws the baseline with ls=":" (dotted) color="k";
            # previous code incorrectly looked for "--" (dashed) and never
            # matched, causing the baseline to be silently mislabelled.
            new_lines = ax.lines[n_lines_before:]
            if new_lines:
                handles, labels_list = [], []
                val_idx = 0
                for line in new_lines:
                    ls = line.get_linestyle()
                    color = str(line.get_color()).lower()
                    is_baseline_line = (
                        # lifelines >= 0.25 uses ":" (dotted) + black
                        ls in (":", "dotted")
                        or (0, (1.0, 1.0)) == ls  # normalised dotted tuple
                        # older lifelines may use "--" dashed + black
                        or ls in ("--", "dashed", (0, (5.0, 5.0)))
                    ) and color in ("k", "black", "#000000", "#0a0a0a",
                                    "#000000ff")
                    handles.append(line)
                    if is_baseline_line and use_baseline:
                        labels_list.append("Baseline (mean)")
                    else:
                        if val_idx < len(value_labels):
                            labels_list.append(value_labels[val_idx])
                            val_idx += 1
                        else:
                            labels_list.append(line.get_label())
                ax.legend(handles, labels_list, fontsize=7)
            elif ax.get_legend():
                ax.legend(value_labels, fontsize=7)

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

    fig.legend(
        handles=[
            mpatches.Patch(color="#1f77b4", label="PH satisfied (p >= 0.05)"),
            mpatches.Patch(color="#d62728", label="PH violated (p < 0.05)"),
        ],
        loc="lower center", ncol=2, fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    suffix_str = f" [{label_suffix}]" if label_suffix else ""
    fig.suptitle(
        f"Covariate Effect Plots \u2014 {window}/{endpoint}{suffix_str}\n"
        "(Refer to Schoenfeld residual plot for formal PH test)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fname = (f"covariate_effects_{window}_{endpoint}"
             + (f"_{label_suffix}" if label_suffix else "") + ".png")
    out = os.path.join(fig_dir, fname)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [{window}/{endpoint}] Covariate effect plot -> {out}")


# ---------------------------------------------------------------------------
# Schoenfeld residual plots
# ---------------------------------------------------------------------------

def plot_schoenfeld_residuals(
    cph,
    df_model: pd.DataFrame,
    ph_df: pd.DataFrame,
    window: str,
    endpoint: str,
    fig_dir: str,
    label_suffix: str = "",
    display_name_fn: Optional[Callable[[str], str]] = None,
) -> None:
    """Plot scaled Schoenfeld residuals vs. time for each covariate.

    A flat LOWESS trend indicates the PH assumption holds; a systematic slope
    indicates a time-varying hazard ratio (PH violation).  Subplots for
    covariates with p < 0.05 are highlighted with a red border and title.

    Parameters
    ----------
    cph : fitted CoxPHFitter
    df_model : DataFrame used to fit the model
    ph_df : Schoenfeld test results (columns: ``feature``, ``p``)
    window, endpoint : labels for titles and filenames
    fig_dir : output directory
    label_suffix : optional filename suffix (e.g. ``"stratified"``)
    display_name_fn : optional mapping column name -> display label
    """
    if not _HAS_MPL:
        return
    os.makedirs(fig_dir, exist_ok=True)

    if display_name_fn is None:
        display_name_fn = lambda x: x  # noqa: E731

    try:
        resid = cph.compute_residuals(df_model, kind="scaled_schoenfeld")
    except Exception as exc:
        print(f"  [{window}/{endpoint}] Cannot compute Schoenfeld residuals: {exc}")
        return

    event_col = cph.event_col
    time_col = cph.duration_col
    event_mask = df_model[event_col].astype(bool)
    resid_ev = resid.loc[event_mask]
    times_ev = df_model.loc[event_mask, time_col].values

    covariates = resid.columns.tolist()
    if not covariates:
        return

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
        display = display_name_fn(cov)

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

        p_str = f" (p={p_val:.3f})" if p_val is not None else ""
        flag = " (FAIL)" if violation else " (OK)"
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

    fig.legend(
        handles=[
            mpatches.Patch(color="#1f77b4", label="PH satisfied (p >= 0.05)"),
            mpatches.Patch(color="#d62728", label="PH violated (p < 0.05)"),
            mlines.Line2D([], [], color="gray", linestyle="--",
                          linewidth=1.0, label="Reference (y = 0)"),
        ],
        loc="lower center", ncol=3, fontsize=8,
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
    print(f"  [{window}/{endpoint}] Schoenfeld residual plot -> {out}")


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

def plot_forest(
    coef_df: pd.DataFrame,
    window: str,
    endpoint: str,
    out_path: str,
    concordance: float,
    display_name_fn: Optional[Callable[[str], str]] = None,
    lang_fn: Optional[Callable[[str, str], str]] = None,
) -> None:
    """Plot a Forest Plot (HR +/- 95% CI) and save as PNG.

    Parameters
    ----------
    coef_df : DataFrame with columns ``feature``, ``HR``, ``hr_lo95``,
              ``hr_hi95``, ``p``, ``n_obs``, ``n_events``.
    window, endpoint : labels
    out_path : output file path
    concordance : Harrell C-index to show in the title
    display_name_fn : optional column-name -> label mapping
    lang_fn : optional ``(zh, en) -> str`` for bilingual labels;
              defaults to English
    """
    if not _HAS_MPL or coef_df.empty:
        return
    if display_name_fn is None:
        display_name_fn = lambda x: x  # noqa: E731
    if lang_fn is None:
        lang_fn = lambda zh, en: en  # noqa: E731

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        _plot_forest_inner(coef_df, window, endpoint, out_path, concordance,
                           display_name_fn, lang_fn)


def _plot_forest_inner(
    coef_df: pd.DataFrame,
    window: str,
    endpoint: str,
    out_path: str,
    concordance: float,
    display_name_fn: Callable[[str], str],
    lang_fn: Callable[[str, str], str],
) -> None:
    df = coef_df.sort_values("HR", ascending=True).reset_index(drop=True)
    n = len(df)

    display_labels = df["feature"].map(display_name_fn).tolist()

    fig_h = max(3.5, 0.5 * n + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y_pos = np.arange(n)
    colors = ["#d62728" if p < 0.05 else "#1f77b4"
              for p in df["p"].fillna(1.0)]

    for i, row in df.iterrows():
        lo = row["hr_lo95"] if pd.notna(row["hr_lo95"]) else row["HR"]
        hi = row["hr_hi95"] if pd.notna(row["hr_hi95"]) else row["HR"]
        ax.plot([lo, hi], [y_pos[i], y_pos[i]], color=colors[i],
                linewidth=1.8, zorder=2)
        ax.scatter(row["HR"], y_pos[i], color=colors[i], s=50, zorder=3)

    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.0, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.set_xlabel(lang_fn("风险比 HR（95% CI）", "Hazard Ratio HR (95% CI)"),
                  fontsize=10)

    n_obs = int(df["n_obs"].iloc[0])
    n_events = int(df["n_events"].iloc[0])
    ax.set_title(
        f"Forest Plot \u2014 {window}/{endpoint}\n"
        f"C-index = {concordance:.4f}  |  "
        + lang_fn(f"n = {n_obs}\uff0c\u4e8b\u4ef6 = {n_events}",
                  f"n = {n_obs},  events = {n_events}"),
        fontsize=10,
    )

    sig_patch = mpatches.Patch(color="#d62728",
                               label=lang_fn("p < 0.05\uff08\u663e\u8457\uff09",
                                             "p < 0.05"))
    nosig_patch = mpatches.Patch(color="#1f77b4",
                                 label=lang_fn("p>=0.05\uff08\u4e0d\u663e\u8457\uff09",
                                               "p >= 0.05"))
    ax.legend(handles=[sig_patch, nosig_patch], fontsize=8, loc="lower right")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Forest plot -> {out_path}")


# ---------------------------------------------------------------------------
# Baseline survival curve
# ---------------------------------------------------------------------------

def plot_baseline_survival(
    cph,
    window: str,
    endpoint: str,
    out_path: str,
    lang_fn: Optional[Callable[[str, str], str]] = None,
) -> None:
    """Plot the Cox model baseline survival function S0(t) and save as PNG.

    Parameters
    ----------
    cph : fitted CoxPHFitter
    window, endpoint : labels
    out_path : output file path
    lang_fn : optional bilingual label function
    """
    if not _HAS_MPL or cph is None:
        return
    if lang_fn is None:
        lang_fn = lambda zh, en: en  # noqa: E731

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
        ax.set_xlabel(
            lang_fn("\u968f\u8bbf\u65f6\u95f4\uff08\u5929\uff09",
                    "Follow-up Time (days)"),
            fontsize=10,
        )
        ax.set_ylabel(
            lang_fn("\u57fa\u7ebf\u751f\u5b58\u6982\u7387 S0(t)",
                    "Baseline Survival S0(t)"),
            fontsize=10,
        )
        ax.set_title(
            lang_fn(f"\u57fa\u7ebf\u751f\u5b58\u51fd\u6570 \u2014 {window}/{endpoint}",
                    f"Baseline Survival Function \u2014 {window}/{endpoint}"),
            fontsize=10,
        )
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    print(f"  Baseline survival -> {out_path}")


# ---------------------------------------------------------------------------
# C-index summary bar chart
# ---------------------------------------------------------------------------

def plot_cindex_summary(
    summaries: List[Dict],
    out_path: str,
    lang_fn: Optional[Callable[[str, str], str]] = None,
) -> None:
    """Plot a bar chart of C-index values across all window/endpoint combos.

    Parameters
    ----------
    summaries : list of dicts, each with keys ``window``, ``endpoint``,
                ``concordance``, ``skipped``.
    out_path : output file path
    lang_fn : optional bilingual label function
    """
    if not _HAS_MPL:
        return
    if lang_fn is None:
        lang_fn = lambda zh, en: en  # noqa: E731

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
                   label=lang_fn("\u968f\u673a\u57fa\u51c6\uff080.5\uff09",
                                 "Random (0.5)"))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Harrell C-index", fontsize=10)
        ax.set_title(
            lang_fn("Cox PH \u6a21\u578b C-index \u6c47\u603b",
                    "Cox PH Model \u2014 C-index Summary"),
            fontsize=11,
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    print(f"\nC-index summary -> {out_path}")
