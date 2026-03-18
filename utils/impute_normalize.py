# -*- coding: utf-8 -*-
"""
Missing-value imputation and feature normalisation for HFpEF cohort CSVs.

Input : csv/comorbidity/hfpef_cohort_win_*_comorbidity.csv
Output: csv/processed/hfpef_cohort_win_*_processed.csv
        csv/processed/missingness_report.csv   (per-variable analysis)
        csv/processed/feature_decisions.csv    (keep / drop / impute decision)

Decision rules
--------------
Variables (columns)
  ≥ 60 % missing  → dropped; noted in feature_decisions.csv
  20 – 59 % missing → median (continuous) / mode (binary) imputation
                      plus a companion ``<col>_missing_flag`` binary column
  5 – 19 % missing  → median / mode imputation (no extra flag)
  < 5  % missing    → median / mode imputation

Special columns that are NEVER imputed or normalised:
  - ID / datetime columns (subject_id, hadm_id, index_study_id,
    index_study_datetime, index_admittime, index_dischtime, death_date,
    a4c_dicom_filepath)
  - Outcome columns (hospital_expire_flag, died_inhosp, died_post_dc,
    died_30d, died_90d, died_1yr, days_survived_post_dc)
  Note: ``days_survived_post_dc`` is structurally NaN for censored (still-
  alive) patients and must NOT be imputed; the Cox model must treat it
  separately (see README_HFpEF_Cox.md §5 for endpoint construction).

Cases (rows)
  Patients where > 60 % of *retained* feature columns are missing are
  removed; their IDs are written to feature_decisions.csv.

Normalisation (applied after imputation to retained continuous features)
  |skewness| ≤ 1.0  → StandardScaler  (z-score)
  |skewness| > 1.0  → log1p transform, then StandardScaler

Binary columns (0/1) and all categorical/ordinal comorbidity columns are
left as integers and are not normalised.

Usage
-----
    python utils/impute_normalize.py \\
        --input_dir  csv/comorbidity \\
        --output_dir csv/processed

    # Only write the missingness report (no imputation / normalisation):
    python utils/impute_normalize.py --report_only
"""

import argparse
import os
import glob
import sys
import warnings
from typing import Dict, List, Tuple

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

try:
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not found; normalisation will be skipped.\n"
        "Install with: pip install scikit-learn",
        UserWarning,
    )


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------
# These columns are never imputed, never normalised, and are excluded from
# the per-row missingness check used to decide whether to drop a patient.

ID_COLS: List[str] = [
    "subject_id",
    "hadm_id",
    "index_study_id",
    "index_study_datetime",
    "index_admittime",
    "index_dischtime",
    "death_date",
    "a4c_dicom_filepath",
]

OUTCOME_COLS: List[str] = [
    "hospital_expire_flag",
    "died_inhosp",
    "died_post_dc",
    "days_survived_post_dc",   # structural NaN for censored patients
    "died_30d",
    "died_90d",
    "died_1yr",
]

# Binary columns: imputed with mode if missing, but never normalised
BINARY_COLS: List[str] = [
    "gender",          # will be encoded below if present as M/F string
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
    # new comorbidity flags
    "hf_any_diabetes",
    "hf_cardiorenal",
    "hf_met_syndrome_proxy",
    "hf_af_ckd",
    "hf_high_risk_triad",
    "hf_competing_risk",
]

# Ordinal / score columns: imputed with median if missing, not normalised
ORDINAL_COLS: List[str] = [
    "charlson_score",
    "cci_from_flags",
    "hf_comorbidity_burden",
    "hf_comorb_score_custom",
]

# Thresholds for missingness decisions
THRESH_DROP: float = 0.60    # ≥ 60 % → drop variable
THRESH_FLAG: float = 0.20    # ≥ 20 % (and < THRESH_DROP) → add _missing_flag
THRESH_ROW_DROP: float = 0.60  # patient has > 60 % retained features missing → drop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_binary_col(series: pd.Series) -> bool:
    """Return True if *series* contains only 0, 1 (and NaN)."""
    uniq = set(series.dropna().unique())
    return uniq.issubset({0, 1, 0.0, 1.0, True, False})


def _encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'gender' string (M/F) to binary integer 1/0 (male=1)."""
    if "gender" in df.columns and df["gender"].dtype == object:
        df = df.copy()
        df["gender"] = (
            df["gender"].str.strip().str.upper()
            .map({"M": 1, "F": 0})
        )
    return df


def analyse_missingness(
    df: pd.DataFrame,
    skip_cols: List[str],
) -> pd.DataFrame:
    """
    Compute per-column missingness statistics for all feature columns.

    Parameters
    ----------
    df : pd.DataFrame
    skip_cols : list of str
        Columns to exclude from the analysis (IDs, outcomes, etc.)

    Returns
    -------
    pd.DataFrame  with columns:
        column, n_total, n_missing, pct_missing, dtype,
        is_binary, decision, note
    """
    rows = []
    for col in df.columns:
        if col in skip_cols:
            continue
        n_total = len(df)
        n_miss = int(df[col].isnull().sum())
        pct = n_miss / n_total if n_total else 0.0
        is_bin = _is_binary_col(df[col])

        if pct >= THRESH_DROP:
            decision = "DROP"
            note = f"{pct*100:.1f}% missing ≥ {THRESH_DROP*100:.0f}% threshold"
        elif pct >= THRESH_FLAG:
            decision = "IMPUTE+FLAG"
            note = (f"{pct*100:.1f}% missing → impute + add _missing_flag column")
        elif pct > 0:
            decision = "IMPUTE"
            note = f"{pct*100:.1f}% missing → simple imputation"
        else:
            decision = "KEEP"
            note = "no missing values"

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


def _impute_column(
    df: pd.DataFrame,
    col: str,
    is_binary: bool,
    add_flag: bool,
) -> pd.DataFrame:
    """
    Impute a single column in-place in *df* and optionally add a missing flag.

    Binary columns are filled with mode; continuous with median.
    """
    if df[col].isnull().sum() == 0:
        return df

    if is_binary or col in BINARY_COLS or col in ORDINAL_COLS:
        fill_val = df[col].mode(dropna=True)
        fill_val = fill_val.iloc[0] if len(fill_val) else 0
    else:
        fill_val = df[col].median()

    if add_flag:
        flag_col = f"{col}_missing_flag"
        df[flag_col] = df[col].isnull().astype(int)

    df[col] = df[col].fillna(fill_val)
    return df


def impute(
    df: pd.DataFrame,
    missingness_report: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Apply imputation decisions from *missingness_report* to *df*.

    Returns
    -------
    df_imputed : pd.DataFrame
    dropped_cols : list of str  (columns dropped)
    added_flag_cols : list of str  (new _missing_flag columns)
    """
    df = df.copy()
    dropped_cols: List[str] = []
    added_flag_cols: List[str] = []

    for _, row in missingness_report.iterrows():
        col = row["column"]
        decision = row["decision"]

        if col not in df.columns:
            continue

        if decision == "DROP":
            df.drop(columns=[col], inplace=True)
            dropped_cols.append(col)

        elif decision in ("IMPUTE", "IMPUTE+FLAG"):
            add_flag = (decision == "IMPUTE+FLAG")
            df = _impute_column(df, col, is_binary=bool(row["is_binary"]),
                                add_flag=add_flag)
            if add_flag:
                added_flag_cols.append(f"{col}_missing_flag")

        # KEEP: nothing to do

    return df, dropped_cols, added_flag_cols


def drop_high_missing_rows(
    df: pd.DataFrame,
    skip_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove patients where more than THRESH_ROW_DROP fraction of the retained
    feature columns are missing.

    Returns
    -------
    df_clean : pd.DataFrame  (retained patients)
    dropped_rows : pd.DataFrame  (removed patients with their IDs)
    """
    feature_cols = [c for c in df.columns if c not in skip_cols]
    if not feature_cols:
        return df, pd.DataFrame()

    miss_frac = df[feature_cols].isnull().mean(axis=1)
    mask_drop = miss_frac > THRESH_ROW_DROP

    dropped = df.loc[mask_drop, [c for c in ID_COLS if c in df.columns]].copy()
    dropped["pct_features_missing"] = (miss_frac[mask_drop] * 100).round(1).values

    df_clean = df.loc[~mask_drop].copy()
    return df_clean, dropped


def normalise(
    df: pd.DataFrame,
    skip_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalise retained continuous feature columns.

    Normalisation strategy:
      |skewness| ≤ 1 → z-score (StandardScaler)
      |skewness| > 1 → log1p transform then z-score

    Parameters
    ----------
    df : pd.DataFrame
    skip_cols : list of str
        Columns to leave untouched.

    Returns
    -------
    df_norm : pd.DataFrame
    norm_log : pd.DataFrame  (log of transformations applied per column)
    """
    if not _SKLEARN_AVAILABLE:
        warnings.warn(
            "scikit-learn not available; skipping normalisation.",
            UserWarning,
        )
        return df, pd.DataFrame()

    df = df.copy()
    log_rows = []

    # Columns to normalise: numeric, not binary, not ordinal, not skip
    no_norm = set(skip_cols) | set(BINARY_COLS) | set(ORDINAL_COLS)
    # Also skip the new _missing_flag columns
    no_norm |= {c for c in df.columns if c.endswith("_missing_flag")}

    for col in df.columns:
        if col in no_norm:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if _is_binary_col(df[col]):
            continue

        vals = df[col].values.astype(float)
        skew = float(pd.Series(vals).skew())

        if abs(skew) > 1.0:
            transform = "log1p + z-score"
            vals = np.log1p(np.clip(vals, 0, None))
        else:
            transform = "z-score"

        scaler = StandardScaler()
        df[col] = scaler.fit_transform(vals.reshape(-1, 1)).ravel()

        log_rows.append({
            "column":      col,
            "skewness_before": round(skew, 3),
            "transform":   transform,
            "mean_before": round(float(pd.Series(vals).mean()), 4),
            "std_before":  round(float(pd.Series(vals).std()), 4),
        })

    norm_log = pd.DataFrame(log_rows)
    return df, norm_log


def process_file(
    input_path: str,
    output_path: str,
    report_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read one comorbidity-enriched CSV, analyse, impute, normalise, and save.

    Returns
    -------
    (miss_report, feat_decisions_df, df_processed)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"{'='*60}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")

    # Encode gender if stored as string
    df = _encode_gender(df)

    skip = [c for c in ID_COLS + OUTCOME_COLS if c in df.columns]

    # ------------------------------------------------------------------ #
    # Missingness analysis
    # ------------------------------------------------------------------ #
    miss_report = analyse_missingness(df, skip_cols=skip)
    print("\n  --- Missingness report (variables with missing values) ---")
    visible = miss_report[miss_report["pct_missing"] > 0]
    print(visible[["column", "pct_missing", "decision", "note"]].to_string(index=False))

    if report_only:
        return miss_report, pd.DataFrame(), df

    # ------------------------------------------------------------------ #
    # Row-level missingness check (before imputation)
    # ------------------------------------------------------------------ #
    df, dropped_rows = drop_high_missing_rows(df, skip_cols=skip)
    if len(dropped_rows):
        print(f"\n  Dropped {len(dropped_rows)} patient(s) with >{THRESH_ROW_DROP*100:.0f}% features missing:")
        print(dropped_rows.to_string(index=False))
    else:
        print(f"\n  No patients dropped (all have ≤{THRESH_ROW_DROP*100:.0f}% features missing).")

    # ------------------------------------------------------------------ #
    # Imputation
    # ------------------------------------------------------------------ #
    df, dropped_cols, added_flags = impute(df, miss_report)
    print(f"\n  Dropped columns : {dropped_cols if dropped_cols else 'none'}")
    print(f"  Added flag cols : {added_flags if added_flags else 'none'}")

    # ------------------------------------------------------------------ #
    # Normalisation
    # ------------------------------------------------------------------ #
    df, norm_log = normalise(df, skip_cols=skip)
    if len(norm_log):
        print("\n  --- Normalisation log ---")
        print(norm_log.to_string(index=False))
    else:
        print("\n  Normalisation: scikit-learn unavailable or no columns to normalise.")

    # ------------------------------------------------------------------ #
    # Feature decisions summary
    # ------------------------------------------------------------------ #
    feat_decisions = miss_report[["column", "pct_missing", "decision", "note"]].copy()
    if dropped_rows is not None and len(dropped_rows):
        feat_decisions = pd.concat([
            feat_decisions,
            pd.DataFrame([{
                "column": f"ROWS DROPPED (n={len(dropped_rows)})",
                "pct_missing": "",
                "decision": "ROW_DROP",
                "note": f">{THRESH_ROW_DROP*100:.0f}% retained features missing",
            }])
        ], ignore_index=True)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  Saved processed file → {output_path}")
    print(f"  Final shape: {df.shape}")

    return miss_report, feat_decisions, df


def main() -> None:
    global THRESH_DROP, THRESH_FLAG  # noqa: PLW0603
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input_dir", default="csv/comorbidity",
        help="Directory with comorbidity-enriched CSVs (default: csv/comorbidity)",
    )
    ap.add_argument(
        "--output_dir", default="csv/processed",
        help="Directory for output (default: csv/processed)",
    )
    ap.add_argument(
        "--pattern", default="hfpef_cohort_win_*_comorbidity.csv",
        help="Glob pattern for input files",
    )
    ap.add_argument(
        "--report_only", action="store_true",
        help="Print missingness report only; do not write processed files",
    )
    ap.add_argument(
        "--drop_threshold", type=float, default=THRESH_DROP,
        help=f"Variable drop threshold (default: {THRESH_DROP})",
    )
    ap.add_argument(
        "--flag_threshold", type=float, default=THRESH_FLAG,
        help=f"Missing-flag threshold (default: {THRESH_FLAG})",
    )
    args = ap.parse_args()

    THRESH_DROP = args.drop_threshold
    THRESH_FLAG = args.flag_threshold

    pattern = os.path.join(args.input_dir, args.pattern)
    input_files = sorted(glob.glob(pattern))
    if not input_files:
        print(f"No files matched: {pattern}")
        return

    print(f"Found {len(input_files)} file(s):")
    for f in input_files:
        print(f"  {f}")

    all_miss: List[pd.DataFrame] = []
    all_feat: List[pd.DataFrame] = []

    for input_path in input_files:
        basename = os.path.basename(input_path)
        stem = basename.replace("_comorbidity", "").replace(".csv", "")
        output_path = os.path.join(
            args.output_dir, f"{stem}_processed.csv"
        )
        miss, feat, _ = process_file(
            input_path, output_path, report_only=args.report_only
        )
        window = stem.replace("hfpef_cohort_win_", "")
        miss.insert(0, "window", window)
        feat.insert(0, "window", window)
        all_miss.append(miss)
        all_feat.append(feat)

    if not args.report_only:
        os.makedirs(args.output_dir, exist_ok=True)

        miss_all = pd.concat(all_miss, ignore_index=True)
        miss_path = os.path.join(args.output_dir, "missingness_report.csv")
        miss_all.to_csv(miss_path, index=False)
        print(f"\nSaved combined missingness report → {miss_path}")

        feat_all = pd.concat(all_feat, ignore_index=True)
        feat_path = os.path.join(args.output_dir, "feature_decisions.csv")
        feat_all.to_csv(feat_path, index=False)
        print(f"Saved feature decisions → {feat_path}")

    print(f"\nDone. Output written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
