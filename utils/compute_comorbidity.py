# -*- coding: utf-8 -*-
"""
Comorbidity index computation for HFpEF cohort CSV files.

Input : csv/cleaned/hfpef_cohort_win_*_cleaned.csv
Output: csv/comorbidity/hfpef_cohort_win_*_comorbidity.csv
        csv/comorbidity/comorbidity_summary.csv  (group-level statistics)

Processing steps
----------------
1. Recompute a partial Charlson Comorbidity Index (``cci_from_flags``) from
   the available binary ICD-derived columns.  Because the full ICD code list
   is not stored in the CSV, this is a *lower-bound* CCI that should match
   the provided ``charlson_score`` for the conditions that ARE captured; any
   additional diagnoses in the original query will cause the provided score
   to be higher.  All patients receive an implicit +1 for heart failure
   (inherent in the cohort definition).

   Charlson weights used for available columns:
     myocardial_infarct          +1
     CHF (all patients)          +1  (implicit; HFpEF cohort)
     peripheral_vascular_disease +1
     cerebrovascular_disease     +1
     chronic_pulmonary_disease   +1
     mild_liver_disease          +1
     diabetes_without_cc         +1  (only when diabetes_with_cc == 0)
     diabetes_with_cc            +2  (mutually exclusive with _without_cc)
     renal_disease               +2
     malignant_cancer            +2
     severe_liver_disease        +3

   Note: hypertension and atrial_fibrillation are NOT part of the original
   Charlson index but are tracked separately as HF-specific risk factors.

2. Add HF-specific prognostic composite features
   (grounded in HFpEF outcomes literature — MAGGIC, CHARM-Preserved, etc.):

     hf_any_diabetes        : diabetes_without_cc OR diabetes_with_cc
     hf_cardiorenal         : renal_disease (cardiorenal syndrome marker)
     hf_met_syndrome_proxy  : hypertension AND hf_any_diabetes
                              (metabolic syndrome surrogate)
     hf_af_ckd              : atrial_fibrillation AND renal_disease
     hf_high_risk_triad     : atrial_fibrillation AND renal_disease
                              AND hf_any_diabetes
                              (high-risk cluster strongly associated with
                               poor HFpEF prognosis)
     hf_competing_risk      : malignant_cancer OR severe_liver_disease
                              (conditions that may dominate mortality
                               independently of HF trajectory)
     hf_comorbidity_burden  : 0 = low  (charlson_score <  5)
                              1 = medium (5 – 8)
                              2 = high  (≥ 9)
     hf_comorb_score_custom : weighted sum of HF-relevant comorbidities:
                              hypertension(1) + atrial_fibrillation(2) +
                              renal_disease(2) + hf_any_diabetes(1) +
                              malignant_cancer(3) + severe_liver_disease(3) +
                              cerebrovascular_disease(1)

3. Save one output CSV per input file plus a cross-cohort summary table.

Usage
-----
    python utils/compute_comorbidity.py \\
        --input_dir  csv/cleaned \\
        --output_dir csv/comorbidity

    # Dry-run (print summary only, do not write files):
    python utils/compute_comorbidity.py --dry_run
"""

import argparse
import os
import glob
import sys
import warnings
from typing import List

try:
    import numpy as np
    _blocked = [m for m in ('numexpr', 'bottleneck') if m not in sys.modules]
    for _m in _blocked:
        sys.modules[_m] = None  # type: ignore[assignment]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            import pandas as pd
    finally:
        for _m in _blocked:
            sys.modules.pop(_m, None)
except (ImportError, AttributeError, ValueError) as _err:
    raise ImportError(
        "numpy and pandas are required.\n"
        "Install with: pip install numpy pandas"
    ) from None


# ---------------------------------------------------------------------------
# Charlson Comorbidity Index (CCI) – weights for available binary flags
# ---------------------------------------------------------------------------
# Standard weights from Charlson et al. (1987) and Deyo adaptation (1992).
# HF (+1) is added implicitly for all patients (entire cohort has HFpEF).
CCI_WEIGHTS = {
    "myocardial_infarct":          1,
    "peripheral_vascular_disease": 1,
    "cerebrovascular_disease":     1,
    "chronic_pulmonary_disease":   1,
    "mild_liver_disease":          1,
    # diabetes: mutually exclusive categories — handled separately below
    "renal_disease":               2,
    "malignant_cancer":            2,
    "severe_liver_disease":        3,
}

# Columns that should be present in every input file
_REQUIRED_COLS: List[str] = [
    "hypertension",
    "atrial_fibrillation",
    "myocardial_infarct",
    "diabetes_without_cc",
    "diabetes_with_cc",
    "renal_disease",
    "chronic_pulmonary_disease",
    "peripheral_vascular_disease",
    "cerebrovascular_disease",
    "malignant_cancer",
    "mild_liver_disease",
    "severe_liver_disease",
    "charlson_score",
]


def _to_int(series: pd.Series) -> pd.Series:
    """Convert a potentially nullable-integer Series to plain int (fill NaN→0)."""
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def compute_comorbidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CCI-from-flags and HF-specific composite features.

    Parameters
    ----------
    df : pd.DataFrame
        One of the cleaned cohort DataFrames.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with new columns appended.
    """
    df = df.copy()
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required comorbidity columns not found: {missing}"
        )

    # ------------------------------------------------------------------ #
    # Helper – integer versions of binary flags (NaN → 0)
    # ------------------------------------------------------------------ #
    mi    = _to_int(df["myocardial_infarct"])
    pvd   = _to_int(df["peripheral_vascular_disease"])
    cvd   = _to_int(df["cerebrovascular_disease"])
    copd  = _to_int(df["chronic_pulmonary_disease"])
    mild_l = _to_int(df["mild_liver_disease"])
    sev_l  = _to_int(df["severe_liver_disease"])
    dm_wo  = _to_int(df["diabetes_without_cc"])
    dm_w   = _to_int(df["diabetes_with_cc"])
    renal  = _to_int(df["renal_disease"])
    cancer = _to_int(df["malignant_cancer"])
    htn    = _to_int(df["hypertension"])
    af     = _to_int(df["atrial_fibrillation"])

    # ------------------------------------------------------------------ #
    # 1. Partial CCI (lower bound; +1 implicit for CHF in HFpEF cohort)
    # ------------------------------------------------------------------ #
    # Diabetes: CCI treats DM-without-cc and DM-with-cc as mutually
    # exclusive (+1 vs +2).  Use the higher weight when both flags are
    # somehow set to 1 (data artefact).
    dm_cci = np.where(dm_w == 1, 2, np.where(dm_wo == 1, 1, 0))

    cci_from_flags = (
        1           # CHF implicit (all HFpEF patients)
        + mi   * CCI_WEIGHTS["myocardial_infarct"]
        + pvd  * CCI_WEIGHTS["peripheral_vascular_disease"]
        + cvd  * CCI_WEIGHTS["cerebrovascular_disease"]
        + copd * CCI_WEIGHTS["chronic_pulmonary_disease"]
        + mild_l * CCI_WEIGHTS["mild_liver_disease"]
        + dm_cci
        + renal  * CCI_WEIGHTS["renal_disease"]
        + cancer * CCI_WEIGHTS["malignant_cancer"]
        + sev_l  * CCI_WEIGHTS["severe_liver_disease"]
    )
    df["cci_from_flags"] = cci_from_flags.astype(int)

    # ------------------------------------------------------------------ #
    # 2. HF-specific prognostic composite features
    # ------------------------------------------------------------------ #

    # --- a) Any diabetes (either type) ---
    df["hf_any_diabetes"] = ((dm_wo + dm_w) > 0).astype(int)

    # --- b) Cardiorenal syndrome marker ---
    df["hf_cardiorenal"] = (renal > 0).astype(int)

    # --- c) Metabolic syndrome proxy: hypertension + diabetes ---
    df["hf_met_syndrome_proxy"] = (
        (htn > 0) & (df["hf_any_diabetes"] == 1)
    ).astype(int)

    # --- d) AF + CKD co-occurrence ---
    df["hf_af_ckd"] = ((af > 0) & (renal > 0)).astype(int)

    # --- e) High-risk triad: AF + CKD + diabetes ---
    df["hf_high_risk_triad"] = (
        (af > 0) & (renal > 0) & (df["hf_any_diabetes"] == 1)
    ).astype(int)

    # --- f) Competing mortality risk ---
    df["hf_competing_risk"] = (
        (cancer > 0) | (sev_l > 0)
    ).astype(int)

    # --- g) Comorbidity burden categories ---
    cci = pd.to_numeric(df["charlson_score"], errors="coerce").fillna(0)
    df["hf_comorbidity_burden"] = pd.cut(
        cci,
        bins=[-1, 4, 8, 999],
        labels=[0, 1, 2],
    ).astype(int)  # 0=low, 1=medium, 2=high

    # --- h) Custom HF comorbidity score ---
    # Weighted sum of comorbidities most strongly linked to HFpEF prognosis
    # (weights informed by published adjusted hazard ratios):
    #   hypertension              +1  (common, moderate independent effect)
    #   atrial_fibrillation       +2  (major independent predictor)
    #   renal_disease             +2  (cardiorenal → 2× mortality risk)
    #   any diabetes              +1  (moderate HR ~1.3–1.5)
    #   malignant_cancer          +3  (large competing risk / short survival)
    #   severe_liver_disease      +3  (very high short-term mortality)
    #   cerebrovascular_disease   +1  (stroke history → worse functional status)
    df["hf_comorb_score_custom"] = (
        htn    * 1
        + af   * 2
        + renal * 2
        + df["hf_any_diabetes"].values * 1
        + cancer  * 3
        + sev_l   * 3
        + cvd     * 1
    ).astype(int)

    return df


def print_comorbidity_summary(df: pd.DataFrame, label: str = "") -> None:
    """Print a readable summary of comorbidity features."""
    hdr = f"  {label}" if label else "  Summary"
    print(f"\n{hdr}  (n={len(df)})")
    print(f"  {'Column':<30} {'Mean':>7}  {'Sum':>6}  {'% positive':>10}")
    print(f"  {'-'*60}")

    cols_to_show = [
        "charlson_score", "cci_from_flags",
        "hypertension", "atrial_fibrillation", "myocardial_infarct",
        "diabetes_without_cc", "diabetes_with_cc", "hf_any_diabetes",
        "renal_disease", "chronic_pulmonary_disease",
        "peripheral_vascular_disease", "cerebrovascular_disease",
        "malignant_cancer", "mild_liver_disease", "severe_liver_disease",
        "hf_cardiorenal", "hf_met_syndrome_proxy", "hf_af_ckd",
        "hf_high_risk_triad", "hf_competing_risk",
        "hf_comorbidity_burden", "hf_comorb_score_custom",
    ]
    for col in cols_to_show:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if col in ("charlson_score", "cci_from_flags",
                   "hf_comorbidity_burden", "hf_comorb_score_custom"):
            print(f"  {col:<30} {vals.mean():>7.2f}  {vals.sum():>6.0f}  {'(continuous)':>10}")
        else:
            pct = vals.mean() * 100
            print(f"  {col:<30} {vals.mean():>7.3f}  {vals.sum():>6.0f}  {pct:>9.1f}%")


def save_summary_table(all_results: dict, output_dir: str) -> None:
    """Save a cross-cohort comorbidity summary CSV."""
    rows = []
    flag_cols = [
        "hypertension", "atrial_fibrillation", "myocardial_infarct",
        "diabetes_without_cc", "diabetes_with_cc", "hf_any_diabetes",
        "renal_disease", "chronic_pulmonary_disease",
        "peripheral_vascular_disease", "cerebrovascular_disease",
        "malignant_cancer", "mild_liver_disease", "severe_liver_disease",
        "hf_cardiorenal", "hf_met_syndrome_proxy", "hf_af_ckd",
        "hf_high_risk_triad", "hf_competing_risk",
    ]
    score_cols = [
        "charlson_score", "cci_from_flags",
        "hf_comorbidity_burden", "hf_comorb_score_custom",
    ]
    for window, df in all_results.items():
        row: dict = {"window": window, "n": len(df)}
        for col in flag_cols:
            if col in df.columns:
                row[f"{col}_pct"] = round(
                    pd.to_numeric(df[col], errors="coerce").mean() * 100, 1
                )
        for col in score_cols:
            if col in df.columns:
                row[f"{col}_mean"] = round(
                    pd.to_numeric(df[col], errors="coerce").mean(), 2
                )
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "comorbidity_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved comorbidity summary → {out_path}")
    print(summary_df.to_string(index=False))


def process_file(
    input_path: str,
    output_path: str,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Read, enrich, and optionally save a single cohort CSV."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"{'='*60}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")

    df = compute_comorbidity_features(df)

    new_cols = [
        "cci_from_flags",
        "hf_any_diabetes", "hf_cardiorenal", "hf_met_syndrome_proxy",
        "hf_af_ckd", "hf_high_risk_triad", "hf_competing_risk",
        "hf_comorbidity_burden", "hf_comorb_score_custom",
    ]
    print(f"  Added columns: {', '.join(new_cols)}")

    print_comorbidity_summary(df, label=os.path.basename(input_path))

    if not dry_run:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n  Saved → {output_path}")
    else:
        print("\n  [dry-run] file not written")

    return df


def _resolve_path(path: str) -> str:
    """
    Resolve a (possibly relative) path.

    When *path* is relative and does not exist under the current working
    directory, the function retries relative to the repository root
    (the directory that contains this script's parent folder).  This
    allows the scripts to be launched from inside the ``utils/``
    sub-directory (e.g. via PyCharm "Run") without requiring explicit
    absolute paths.
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input_dir", default="csv/cleaned",
        help="Directory containing cleaned cohort CSVs (default: csv/cleaned)",
    )
    ap.add_argument(
        "--output_dir", default="csv/comorbidity",
        help="Directory for output CSVs (default: csv/comorbidity)",
    )
    ap.add_argument(
        "--pattern", default="hfpef_cohort_win_*_cleaned.csv",
        help="Glob pattern for input files",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Print summaries without writing output files",
    )
    args = ap.parse_args()

    args.input_dir  = _resolve_path(args.input_dir)
    args.output_dir = _resolve_path(args.output_dir)

    pattern = os.path.join(args.input_dir, args.pattern)
    input_files = sorted(glob.glob(pattern))
    if not input_files:
        print(f"No files matched: {pattern}")
        return

    print(f"Found {len(input_files)} file(s):")
    for f in input_files:
        print(f"  {f}")

    all_results: dict = {}
    for input_path in input_files:
        basename = os.path.basename(input_path)
        # e.g. hfpef_cohort_win_hadm_cleaned.csv
        #   → hfpef_cohort_win_hadm_comorbidity.csv
        stem = basename.replace("_cleaned", "").replace(".csv", "")
        window = stem.replace("hfpef_cohort_win_", "")
        output_path = os.path.join(
            args.output_dir, f"{stem}_comorbidity.csv"
        )
        df = process_file(input_path, output_path, dry_run=args.dry_run)
        all_results[window] = df

    if not args.dry_run:
        save_summary_table(all_results, args.output_dir)

    print(f"\nDone. Output written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
