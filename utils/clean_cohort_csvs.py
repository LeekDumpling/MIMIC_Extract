# -*- coding: utf-8 -*-
"""
Batch data cleaning for HFpEF cohort CSV files exported from MIMIC-IV hosp module.

Usage:
    python utils/clean_cohort_csvs.py \
        --input_dir csv \
        --output_dir csv/cleaned \
        --resource_path resources

Cleans all CSV files matching the HFpEF cohort schema in --input_dir and
saves cleaned copies to --output_dir with a '_cleaned' suffix.
Original files are never modified.

Cleaning steps applied:
  1. Datetime parsing (index_study_datetime, index_admittime, index_dischtime,
     death_date)
  2. Type enforcement for binary flag columns (int 0/1)
  3. Lab / vital outlier removal and valid-range clipping using
     resources/variable_ranges.csv thresholds (same logic as
     mimic_direct_extract.apply_variable_limits)

Also emits two reference artifacts to --output_dir:
  - variable_name_mapping.csv  : CSV-column → MIMIC-Extract LEVEL2 mapping
  - variables_not_in_mimic_extract.csv : columns with no MIMIC-Extract threshold
"""

import argparse
import os
import glob
import sys
import warnings
from typing import Dict, Tuple

try:
    import numpy as np
    # When NumPy 2.x is installed alongside optional pandas speed-up
    # libraries (numexpr, bottleneck) compiled for NumPy 1.x, importing
    # those libraries triggers numpy's C-level compatibility shim which
    # prints a full exception traceback directly via fprintf(stderr) –
    # BEFORE emitting a Python RuntimeWarning.  warnings.filterwarnings()
    # only intercepts the Python-level warning; it cannot suppress the
    # C-level fprintf output.
    #
    # Fix: insert a None sentinel into sys.modules for each problematic
    # optional dependency that has not been successfully imported yet.
    # The Python import machinery raises ModuleNotFoundError immediately
    # without executing the C extension, so numpy's compat shim is never
    # triggered.  pandas.import_optional_dependency(errors="warn")
    # catches the ImportError gracefully and falls back to pure-Python
    # implementations.  The sentinels are removed in the finally block
    # so no other code in this process is affected.
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
except (ImportError, AttributeError, ValueError) as _import_err:
    raise ImportError(
        "{}: {}\n\n"
        "numpy and pandas are required, and they must be compatible with each other.\n"
        "A common cause of this error is numpy 2.x installed alongside pandas < 2.0\n"
        "(e.g. numpy 2.0.2 + pandas 1.4.4 will produce a binary-incompatibility\n"
        "ValueError: 'numpy.dtype size changed').\n\n"
        "Recommended fixes (choose one):\n"
        "  Upgrade pandas:   pip install \"pandas>=2.0\"\n"
        "  Downgrade numpy:  pip install \"numpy<2.0\"\n"
        "  Install all deps: pip install -r utils/requirements.txt\n"
        "  Use the conda env: conda activate mimic_extract\n\n"
        "Check what is installed:\n"
        "    python -m pip show numpy pandas".format(
            type(_import_err).__name__, _import_err)
    ) from None


# ---------------------------------------------------------------------------
# Column mapping: CSV column name  →  MIMIC-Extract LEVEL2 name
# (only columns that have a corresponding entry in variable_ranges.csv)
# ---------------------------------------------------------------------------
VARIABLE_NAME_MAPPING = {
    "creatinine":   "Creatinine",
    "bun":          "Blood urea nitrogen",
    "sodium":       "Sodium",
    "potassium":    "Potassium",
    "chloride":     "Chloride",
    "bicarbonate":  "Bicarbonate",
    "calcium":      "Calcium",
    "glucose_lab":  "Glucose",
    "aniongap":     "Anion Gap",
    "albumin":      "Albumin",
    "hemoglobin":   "Hemoglobin",
    "hematocrit":   "Hematocrit",
    "wbc":          "White blood cell count",
    "platelet":     "Platelets",
    "troponin_t":   "Troponin-T",
    "pt":           "Prothrombin time",
    "ptt":          "Partial thromboplastin time",
    "heart_rate":   "Heart rate",
    "sbp":          "Systolic blood pressure",
    "dbp":          "Diastolic blood pressure",
    "mbp":          "Mean blood pressure",
    "resp_rate":    "Respiratory rate",
    "spo2":         "Oxygen saturation",
    "temperature_c":"Temperature",
    "omr_weight_kg":"Weight",
}

# ---------------------------------------------------------------------------
# Variables present in the CSV but NOT in MIMIC-Extract variable_ranges.csv
# (no outlier / valid-range thresholds; kept as-is but documented)
# ---------------------------------------------------------------------------
VARIABLES_NOT_IN_MIMIC_EXTRACT = [
    "ntprobnp",   # NT-proBNP: cardiac biomarker, not in MIMIC-Extract range table
    "crp",        # C-reactive protein: inflammatory marker, no defined range
    "inr",        # International Normalized Ratio: not listed separately in range table
    "omr_bmi",    # BMI from OMR (outpatient records): not in MIMIC-Extract
    "omr_sbp",    # Outpatient SBP from OMR: distinct from in-hospital chartevents SBP
    "omr_dbp",    # Outpatient DBP from OMR: distinct from in-hospital chartevents DBP
]

# ---------------------------------------------------------------------------
# Datetime columns
# ---------------------------------------------------------------------------
DATETIME_COLS = [
    "index_study_datetime",
    "index_admittime",
    "index_dischtime",
    "death_date",
]

# ---------------------------------------------------------------------------
# Binary flag columns (should be integer 0/1, may be stored as float)
# ---------------------------------------------------------------------------
BINARY_FLAG_COLS = [
    "hospital_expire_flag",
    "died_inhosp",
    "died_post_dc",
    "died_30d",
    "died_90d",
    "died_1yr",
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
]


def load_variable_ranges(resource_path: str) -> pd.DataFrame:
    """Load MIMIC-Extract variable_ranges.csv, indexed by LEVEL2 name (lower)."""
    path = os.path.join(resource_path, "variable_ranges.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"variable_ranges.csv not found at: {path}")
    vr = pd.read_csv(path)
    vr.columns = [c.strip() for c in vr.columns]
    vr["LEVEL2"] = vr["LEVEL2"].str.strip()
    vr = vr.set_index("LEVEL2")
    return vr


def apply_outlier_limits(df: pd.DataFrame, var_ranges: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply outlier removal and valid-range clipping for all mapped lab/vital columns.

    Logic (identical to mimic_direct_extract.apply_variable_limits):
      - value < OUTLIER_LOW  or  value > OUTLIER_HIGH  → NaN
      - value < VALID_LOW    (but >= OUTLIER_LOW)       → clip to VALID_LOW
      - value > VALID_HIGH   (but <= OUTLIER_HIGH)      → clip to VALID_HIGH

    Returns:
        df_clean: cleaned copy of df
        cleaning_log: dict {col: {n_outlier, n_valid_low, n_valid_high}}
    """
    df_clean = df.copy()
    cleaning_log = {}

    for csv_col, mimic_name in VARIABLE_NAME_MAPPING.items():
        if csv_col not in df_clean.columns:
            continue

        # Look up thresholds; skip if not in range table or thresholds are NaN
        if mimic_name not in var_ranges.index:
            continue

        row = var_ranges.loc[mimic_name]
        try:
            outlier_low  = float(row["OUTLIER LOW"])
            outlier_high = float(row["OUTLIER HIGH"])
            valid_low    = float(row["VALID LOW"])
            valid_high   = float(row["VALID HIGH"])
        except (ValueError, TypeError):
            # Threshold row has NaN values – skip cleaning for this variable
            continue

        if any(np.isnan([outlier_low, outlier_high, valid_low, valid_high])):
            continue

        col = df_clean[csv_col].astype(float)
        # Write the float-cast column back so that NaN assignment and
        # non-integer clip values (e.g. VALID_LOW=9.9) are stored correctly.
        # Without this, an all-integer column stays int64 and the .loc
        # assignment raises TypeError or silently truncates the clipped value.
        df_clean[csv_col] = col
        non_null = col.notna()

        outlier_mask   = non_null & ((col < outlier_low) | (col > outlier_high))
        valid_low_mask = non_null & ~outlier_mask & (col < valid_low)
        valid_hi_mask  = non_null & ~outlier_mask & (col > valid_high)

        df_clean.loc[outlier_mask,   csv_col] = np.nan
        df_clean.loc[valid_low_mask, csv_col] = valid_low
        df_clean.loc[valid_hi_mask,  csv_col] = valid_high

        n_out = int(outlier_mask.sum())
        n_vlo = int(valid_low_mask.sum())
        n_vhi = int(valid_hi_mask.sum())

        if n_out + n_vlo + n_vhi > 0:
            cleaning_log[csv_col] = {
                "mimic_extract_name": mimic_name,
                "n_total_non_null": int(non_null.sum()),
                "n_outlier_set_nan": n_out,
                "n_clipped_to_valid_low": n_vlo,
                "n_clipped_to_valid_high": n_vhi,
            }
            print(
                f"  {csv_col} ({mimic_name}): "
                f"{n_out} outliers→NaN, "
                f"{n_vlo} clipped to VALID_LOW={valid_low}, "
                f"{n_vhi} clipped to VALID_HIGH={valid_high}"
            )

    return df_clean, cleaning_log


def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce expected column types:
      - binary flags → Int8 (nullable integer, preserves NaN)
      - datetime columns → datetime64
      - anchor_age → float
      - charlson_score → Int16 (nullable)
      - days_survived_post_dc → float (can be NaN)
    """
    df = df.copy()

    # Binary flags
    for col in BINARY_FLAG_COLS:
        if col in df.columns:
            df[col] = pd.array(df[col], dtype="Int8")

    # Datetimes
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Charlson score
    if "charlson_score" in df.columns:
        df["charlson_score"] = pd.array(df["charlson_score"], dtype="Int16")

    return df


def clean_single_file(input_path: str, output_path: str, var_ranges: pd.DataFrame) -> None:
    """Read one CSV, clean it, and save to output_path."""
    print(f"\n{'='*60}")
    print(f"Cleaning: {os.path.basename(input_path)}")
    print(f"{'='*60}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded {len(df)} rows x {len(df.columns)} columns")

    # Step 1 – type enforcement
    df = enforce_types(df)

    # Step 2 – outlier removal + valid-range clipping
    df, log = apply_outlier_limits(df, var_ranges)
    if not log:
        print("  No outliers found – data is within all defined ranges.")

    # Step 3 – save
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved cleaned file → {output_path}")


def save_reference_tables(output_dir: str, var_ranges: pd.DataFrame) -> None:
    """Save variable_name_mapping.csv and variables_not_in_mimic_extract.csv."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Mapping table ---
    mapping_rows = []
    for csv_col, mimic_name in VARIABLE_NAME_MAPPING.items():
        has_ranges = mimic_name in var_ranges.index
        if has_ranges:
            row = var_ranges.loc[mimic_name]
            try:
                outlier_low  = float(row["OUTLIER LOW"])
                outlier_high = float(row["OUTLIER HIGH"])
                valid_low    = float(row["VALID LOW"])
                valid_high   = float(row["VALID HIGH"])
                ranges_defined = not any(np.isnan([outlier_low, outlier_high,
                                                   valid_low, valid_high]))
            except (ValueError, TypeError):
                ranges_defined = False
        else:
            ranges_defined = False

        mapping_rows.append({
            "csv_column_name": csv_col,
            "mimic_extract_level2_name": mimic_name,
            "cleaning_thresholds_defined": ranges_defined,
            "outlier_low":  var_ranges.loc[mimic_name, "OUTLIER LOW"]  if has_ranges else "",
            "valid_low":    var_ranges.loc[mimic_name, "VALID LOW"]    if has_ranges else "",
            "valid_high":   var_ranges.loc[mimic_name, "VALID HIGH"]   if has_ranges else "",
            "outlier_high": var_ranges.loc[mimic_name, "OUTLIER HIGH"] if has_ranges else "",
        })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_path = os.path.join(output_dir, "variable_name_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)
    print(f"\nSaved variable name mapping → {mapping_path}")
    print(mapping_df.to_string(index=False))

    # --- Variables NOT in MIMIC-Extract ---
    not_in_mimic = pd.DataFrame([
        {
            "csv_column_name": col,
            "reason": {
                "ntprobnp":   "NT-proBNP: cardiac biomarker not included in MIMIC-Extract variable_ranges.csv",
                "crp":        "C-reactive protein: not included in MIMIC-Extract variable_ranges.csv",
                "inr":        "INR: not listed as a separate entry in MIMIC-Extract variable_ranges.csv",
                "omr_bmi":    "BMI from OMR outpatient records: not in MIMIC-Extract (ICU-focused tool)",
                "omr_sbp":    "Outpatient SBP from OMR: separate from in-hospital chartevents SBP",
                "omr_dbp":    "Outpatient DBP from OMR: separate from in-hospital chartevents DBP",
            }.get(col, "No corresponding entry in MIMIC-Extract variable_ranges.csv"),
        }
        for col in VARIABLES_NOT_IN_MIMIC_EXTRACT
    ])
    not_in_path = os.path.join(output_dir, "variables_not_in_mimic_extract.csv")
    not_in_mimic.to_csv(not_in_path, index=False)
    print(f"\nSaved variables-not-in-MIMIC-Extract list → {not_in_path}")
    print(not_in_mimic.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir",    type=str, default="csv",
                    help="Directory containing raw cohort CSV files")
    ap.add_argument("--output_dir",   type=str, default="csv/cleaned",
                    help="Directory to write cleaned CSV files (created if needed)")
    ap.add_argument("--resource_path", type=str, default="resources",
                    help="Path to MIMIC-Extract resources/ folder (contains variable_ranges.csv)")
    ap.add_argument("--pattern",      type=str, default="hfpef_cohort_*.csv",
                    help="Glob pattern for input CSV files")
    args = ap.parse_args()

    var_ranges = load_variable_ranges(args.resource_path)

    # Save reference tables first (independent of individual files)
    save_reference_tables(args.output_dir, var_ranges)

    # Discover input files
    pattern = os.path.join(args.input_dir, args.pattern)
    input_files = sorted(glob.glob(pattern))
    if not input_files:
        print(f"\nNo files matched pattern: {pattern}")
        return

    print(f"\nFound {len(input_files)} file(s) to clean:")
    for f in input_files:
        print(f"  {f}")

    for input_path in input_files:
        basename = os.path.basename(input_path)
        stem, ext = os.path.splitext(basename)
        output_filename = stem + "_cleaned" + ext
        output_path = os.path.join(args.output_dir, output_filename)
        clean_single_file(input_path, output_path, var_ranges)

    print(f"\nDone. Cleaned files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
