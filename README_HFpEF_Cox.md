# HFpEF Cohort — Cox Proportional Hazards Survival Analysis

## Project Overview

This project builds a **Cox proportional hazards (Cox PH) model** to predict
post-discharge mortality for patients hospitalised with **Heart Failure with
Preserved Ejection Fraction (HFpEF)** using data from the
[MIMIC-IV](https://physionet.org/content/mimiciv/) database and the
[MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract) feature
extraction pipeline.

**Primary endpoint**: All-cause mortality after hospital discharge.

**Cohort**: 530 HFpEF index admissions from MIMIC-IV with a paired
transthoracic echocardiogram study (A4C view).

---

## Repository Layout (project-specific files)

```
SQL_Queries/             # Step 1 – cohort SQL (MIMIC-IV)
csv/                     # raw cohort CSVs (three time windows)
csv/cleaned/             # Step 3 – outlier-cleaned CSVs
csv/comorbidity/         # Step 4 – comorbidity-enriched CSVs
csv/processed/           # Step 5 – imputed + normalised CSVs
resources/               # MIMIC-Extract variable_ranges.csv etc.
utils/
  clean_cohort_csvs.py   # Step 3
  compute_comorbidity.py # Step 4
  impute_normalize.py    # Step 5
README_HFpEF_Cox.md      # ← this file
```

---

## Progress Checklist

| # | Step | Status | Script / File |
|---|------|--------|---------------|
| 1 | Cohort definition (SQL) | ✅ Done | `SQL_Queries/` |
| 2 | Feature extraction (MIMIC-Extract) | ✅ Done | MIMIC-Extract pipeline |
| 3 | Data cleaning (outlier removal) | ✅ Done | `utils/clean_cohort_csvs.py` |
| 4 | Comorbidity processing | ✅ Done | `utils/compute_comorbidity.py` |
| 5 | Imputation & normalisation | ✅ Done | `utils/impute_normalize.py` |
| 6 | Survival endpoint construction | ⬜ TODO | — |
| 7 | Feature selection | ⬜ TODO | — |
| 8 | Cox PH model fitting | ⬜ TODO | — |
| 9 | Assumption testing | ⬜ TODO | — |
| 10 | Model evaluation | ⬜ TODO | — |
| 11 | Visualisation | ⬜ TODO | — |

---

## Detailed Step Descriptions

### Step 1 — Cohort Definition (SQL)

**File**: `SQL_Queries/codes.sql`, `SQL_Queries/statics.sql`

**Method**: MIMIC-IV PostgreSQL queries selecting all adult hospital
admissions where:
- A primary or secondary ICD-9/ICD-10 code for HF with preserved EF was
  recorded (ICD-10: `I50.30`, `I50.31`, `I50.32`, `I50.33`, `I50.40`,
  `I50.41`, `I50.42`, `I50.43`, `I50.9`; ICD-9: `428.x` family).
- A transthoracic echocardiogram study (A4C view DICOM) was performed
  during the same admission.
- Age ≥ 18 years.

The query extracts:
- Patient demographics (`subject_id`, `hadm_id`, `gender`, `anchor_age`).
- Admission / discharge timestamps and vital status.
- **Survival outcomes**: `died_inhosp`, `died_post_dc`,
  `days_survived_post_dc`, `died_30d`, `died_90d`, `died_1yr`.
- **ICD-derived comorbidity binary flags** (Deyo/Charlson mapping) and
  a raw `charlson_score` via the MIMIC-IV `comorbidity_charlson` concept.

**Output**: Three CSV files, one per feature-aggregation time window
(see Step 2).

---

### Step 2 — Feature Extraction (MIMIC-Extract)

**Framework**: [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract)
aggregates ICU chartevents, labevents, and OMR data.

**Three time windows** are produced for every patient:

| Window | Description |
|--------|-------------|
| `win_hadm` | Entire index admission (admission → discharge) |
| `win_48h24h` | 48 h to 24 h **before** the echocardiogram study |
| `win_48h48h` | 48 h before to 48 h **after** the echocardiogram study |

**Features extracted per window**:

| Category | Columns |
|----------|---------|
| Lab values | creatinine, BUN, sodium, potassium, chloride, bicarbonate, calcium, glucose, anion gap, albumin, haemoglobin, haematocrit, WBC, platelets, NT-proBNP, troponin-T, CRP, INR, PT, PTT |
| Vital signs | heart rate, SBP, DBP, MBP, respiratory rate, SpO₂, temperature |
| OMR (outpatient) | weight, BMI, SBP, DBP |

Each feature is the **mean** of all measurements within the window.

---

### Step 3 — Data Cleaning (`utils/clean_cohort_csvs.py`)

**Method**:
1. **Datetime parsing** — `index_study_datetime`, `index_admittime`,
   `index_dischtime`, `death_date` → `datetime64`.
2. **Binary flag coercion** — comorbidity / outcome flags cast to nullable
   `Int8` (preserves `NaN` vs 0).
3. **Outlier removal and valid-range clipping** — using thresholds from
   `resources/variable_ranges.csv` (same logic as
   `mimic_direct_extract.apply_variable_limits`):
   - value < `OUTLIER_LOW` or > `OUTLIER_HIGH` → `NaN`
   - value < `VALID_LOW` (but within outlier bounds) → clipped to `VALID_LOW`
   - value > `VALID_HIGH` (but within outlier bounds) → clipped to `VALID_HIGH`

**Output**: `csv/cleaned/hfpef_cohort_win_*_cleaned.csv`

---

### Step 4 — Comorbidity Processing (`utils/compute_comorbidity.py`)

**Method**:

#### 4a. Partial Charlson Comorbidity Index (CCI) recomputation

A `cci_from_flags` column is computed from the available binary ICD-derived
flags as a **lower-bound verification** of the SQL-derived `charlson_score`.
All patients receive an implicit +1 for heart failure.

| Flag column | CCI weight |
|-------------|-----------|
| myocardial_infarct | +1 |
| CHF (all patients, implicit) | +1 |
| peripheral_vascular_disease | +1 |
| cerebrovascular_disease | +1 |
| chronic_pulmonary_disease | +1 |
| mild_liver_disease | +1 |
| diabetes_without_cc | +1 (only when diabetes_with_cc = 0) |
| diabetes_with_cc | +2 (mutually exclusive with _without_cc) |
| renal_disease | +2 |
| malignant_cancer | +2 |
| severe_liver_disease | +3 |

Note: `hypertension` and `atrial_fibrillation` are **not** part of the
original Charlson index but are tracked as HF-specific risk factors.

Mean `cci_from_flags` = **4.02** vs mean `charlson_score` = **6.61**,
consistent with the lower-bound expectation (additional diagnoses not
captured in available flag columns account for the ~2.6-point gap).

#### 4b. HF-specific prognostic composite features

Eight new binary/ordinal columns are added (supported by MAGGIC,
CHARM-Preserved, and other HFpEF outcome literature):

| New Column | Definition | Rationale |
|-----------|-----------|-----------|
| `hf_any_diabetes` | `diabetes_without_cc` OR `diabetes_with_cc` | Unified DM flag; DM raises HFpEF mortality HR ~1.3–1.5 |
| `hf_cardiorenal` | `renal_disease` | Cardiorenal syndrome; CKD doubles HF mortality risk |
| `hf_met_syndrome_proxy` | hypertension AND `hf_any_diabetes` | Metabolic syndrome surrogate; common HFpEF aetiology cluster |
| `hf_af_ckd` | AF AND renal disease | Co-occurrence strongly predicts adverse outcomes |
| `hf_high_risk_triad` | AF AND renal disease AND diabetes | High-risk cluster: all three together confer worst prognosis |
| `hf_competing_risk` | malignant_cancer OR severe_liver_disease | Non-cardiac mortality may dominate; affects Cox model interpretation |
| `hf_comorbidity_burden` | 0=low (CCI<5), 1=medium (5–8), 2=high (≥9) | Ordinal burden category from charlson_score |
| `hf_comorb_score_custom` | Weighted sum (HTN×1 + AF×2 + renal×2 + DM×1 + cancer×3 + severe liver×3 + CVD×1) | HFpEF-specific risk score |

**Key cohort statistics** (all three windows are identical for comorbidity
features as these come from ICD codes, not time windows):

| Comorbidity | Prevalence |
|-------------|-----------|
| Hypertension | 92.1 % |
| Atrial fibrillation | 53.8 % |
| Renal disease | 49.6 % |
| Any diabetes | 49.1 % |
| Chronic pulmonary disease | 37.0 % |
| Metabolic syndrome proxy | 46.4 % |
| AF + CKD | 27.7 % |
| High-risk triad | 14.9 % |
| Competing risk (cancer/liver) | 12.5 % |

**Output**: `csv/comorbidity/hfpef_cohort_win_*_comorbidity.csv` and
`csv/comorbidity/comorbidity_summary.csv`.

---

### Step 5 — Imputation & Normalisation (`utils/impute_normalize.py`)

#### 5a. Missing-value analysis and variable-drop decisions

Decision thresholds applied to **all feature columns** (ID, outcome, and
datetime columns are excluded):

| Missing % | Decision |
|-----------|---------|
| ≥ 60 % | **DROP** variable — too sparse for reliable imputation |
| 20 – 59 % | **IMPUTE + FLAG** — impute with median/mode; add companion `<col>_missing_flag` binary column |
| 5 – 19 % | **IMPUTE** — median (continuous) or mode (binary) imputation |
| < 5 % | **IMPUTE** — median / mode imputation |

**Dropped variables** (≥ 60 % missing in all windows):

| Variable | Missing % (hadm) | Missing % (48h24h) | Missing % (48h48h) | Reason |
|---------|-----------------|-------------------|-------------------|--------|
| `crp` | 89.6 % | 96.2 % | 95.5 % | CRP infrequently ordered in MIMIC-IV ICU context |
| `ntprobnp` | 75.1 % | 80.2 % | 79.6 % | NT-proBNP: not routinely measured in all admissions |
| `troponin_t` | 64.9 % | 70.9 % | 70.2 % | Serial troponin not universal in HFpEF workup |
| `albumin` | 52.3 %* | 65.5 % | 62.8 % | ICU-focused measurement; low capture in ward patients |
| `temperature_c` | 68.1 % | 74.5 % | 74.2 % | MIMIC-Extract ICU aggregation; ~26 % had ICU stay |
| `heart_rate` | 68.1 % | 74.5 % | 74.2 % | Same — chartevents-based vitals absent for non-ICU patients |
| `sbp`, `dbp`, `mbp` | 68.1 % | 74.5 % | 74.2 % | Same |
| `resp_rate`, `spo2` | 68.1 % | 74.5 % | 74.2 % | Same |

\* `albumin` is below the 60 % threshold in the `hadm` window and is
therefore retained with imputation + flag in that window.

**Variables with missing-flag columns added** (20–60 % missing):

| Variable | Window | Missing % |
|---------|--------|----------|
| `albumin` | hadm | 52.3 % |
| `pt`, `inr` | 48h24h, 48h48h | ~25–27 % |
| `ptt` | 48h24h, 48h48h | ~28–30 % |

#### 5b. Row-level completeness check

Patients where > 60 % of *retained* feature columns are missing are
removed. **No patients were dropped** in any window after variable dropping.

#### 5c. Imputation strategy

- **Continuous variables**: median imputation (robust to outliers and
  skewed distributions).
- **Binary / ordinal variables**: mode imputation (most frequent category).

Simple (single) imputation is used at this stage. For final model fitting,
**multiple imputation** (e.g., MICE via `sklearn.impute.IterativeImputer`)
should be considered for variables with 5–60 % missingness to avoid
underestimation of standard errors.

#### 5d. Normalisation

Applied **after** imputation to all retained continuous features:

| Condition | Transform |
|-----------|----------|
| \|skewness\| ≤ 1.0 | z-score (`StandardScaler`) |
| \|skewness\| > 1.0 | log₁₊₁ then z-score |

Typically log1p-transformed (right-skewed): creatinine, BUN, glucose,
WBC, platelets, INR, PT, PTT, troponin-T, NT-proBNP.

Binary columns and all comorbidity/ordinal columns are left as 0/1 integers
and are **not** normalised.

**Output**: `csv/processed/hfpef_cohort_win_*_processed.csv`,
`csv/processed/missingness_report.csv`,
`csv/processed/feature_decisions.csv`.

---

### Step 6 — Survival Endpoint Construction *(TODO)*

**Plan**:

The Cox model requires a (time, event) pair for every patient:

| Patient type | Event (`died_post_dc`) | Time variable |
|-------------|----------------------|---------------|
| Died after discharge | 1 | `days_survived_post_dc` |
| Died in hospital | 0 (exclude or treat separately) | — |
| Alive at last follow-up | 0 (censored) | Must be derived |

For **censored patients** (`died_post_dc = 0`, `died_inhosp = 0`),
`days_survived_post_dc` is `NaN`. The censoring time needs to be constructed
as: `last_known_alive_date − index_dischtime`. The last-known-alive date can
be approximated from MIMIC-IV `patients.dod` (date of death, if available)
or from the study end-date of MIMIC-IV (~2019-12-31 for MIMIC-IV v2.2).

**Code to add** (`utils/build_survival_endpoint.py`):
1. For `died_post_dc = 1`: `time = days_survived_post_dc`, `event = 1`.
2. For `died_inhosp = 1`: exclude from post-discharge analysis or model
   as a competing risk.
3. For censored: `time = (mimic_study_end − index_dischtime).days`,
   `event = 0`.
4. Additional landmark analysis: for 1-year mortality (`died_1yr`),
   truncate follow-up at 365 days.

---

### Step 7 — Feature Selection *(TODO)*

**Planned methods**:
1. **Univariate Cox screening**: Wald test p-value < 0.05 per feature
   (to remove obviously uninformative variables).
2. **Variance Inflation Factor (VIF)**: Drop one of each highly-collinear
   pair (VIF > 10), e.g. `creatinine` / `bun` / `cci_from_flags` /
   `charlson_score`.
3. **Clinical expert review**: Retain clinically important variables
   regardless of univariate significance.
4. **LASSO-penalised Cox** as an alternative automated selection approach
   (`lifelines.fitters.CRCSplineFitter` or `scikit-survival`).

---

### Step 8 — Cox PH Model Fitting *(TODO)*

**Planned libraries**: `lifelines` (primary) or `statsmodels`.

**Model variants to evaluate**:
| Model | Features | Purpose |
|-------|---------|---------|
| Model 1 (base) | Age + sex + charlson_score | Clinical reference |
| Model 2 (lab) | Model 1 + lab panel | Lab-enriched |
| Model 3 (comorbidity) | Model 1 + HF-specific composite flags | Comorbidity-enriched |
| Model 4 (full) | All retained features | Maximal prediction |

**Implementation sketch**:
```python
from lifelines import CoxPHFitter
cph = CoxPHFitter(penalizer=0.1)
cph.fit(df_model, duration_col='time_days', event_col='died_post_dc')
cph.print_summary()
```

---

### Step 9 — Proportional Hazards Assumption Testing *(TODO)*

**Method**: Schoenfeld residual test (global and per-covariate).
- `lifelines.statistics.proportional_hazard_test(cph, df_model)`
- For variables that violate PH: consider time-varying coefficients or
  stratification (e.g. `strata=['hf_comorbidity_burden']`).

---

### Step 10 — Model Evaluation *(TODO)*

**Metrics**:
- **Concordance index (C-index)**: primary discrimination metric
  (equivalent to AUC-ROC for survival data).
- **Integrated Brier Score (IBS)**: calibration over the full
  follow-up period.
- **Calibration plots**: observed vs. predicted survival at
  30 d / 90 d / 1 yr.
- **Cross-validation**: 5-fold CV to obtain unbiased C-index estimate.

---

### Step 11 — Visualisation *(TODO)*

**Planned plots**:
1. Kaplan–Meier curves stratified by `hf_comorbidity_burden`
   (low / medium / high).
2. KM curves stratified by `hf_high_risk_triad`.
3. Forest plot of Cox model hazard ratios with 95 % CI.
4. Partial effects plot for continuous predictors (age, creatinine).
5. Log-log (log[−log S(t)] vs. log t) plots for PH assumption assessment.

---

## Data Dictionary (key columns after Step 5)

### Identifiers (not used in modelling)
| Column | Description |
|--------|-------------|
| `subject_id` | MIMIC-IV patient identifier |
| `hadm_id` | Hospital admission identifier |
| `index_study_id` | Echo study identifier |
| `index_study_datetime` | Date-time of the index echocardiogram |
| `a4c_dicom_filepath` | Path to A4C DICOM file |

### Demographics
| Column | Type after Step 5 | Description |
|--------|------------------|-------------|
| `gender` | Binary (1=M, 0=F) | Patient sex |
| `anchor_age` | Float, z-scored | Age at admission |

### Outcomes (not imputed, not normalised)
| Column | Type | Description |
|--------|------|-------------|
| `hospital_expire_flag` | Binary | Died during index admission |
| `died_inhosp` | Binary | Died in hospital (same as above) |
| `died_post_dc` | Binary | **Cox event indicator** (died after discharge) |
| `days_survived_post_dc` | Float | Days alive post-discharge (NaN = censored) |
| `died_30d` | Binary | Died within 30 days of discharge |
| `died_90d` | Binary | Died within 90 days of discharge |
| `died_1yr` | Binary | Died within 1 year of discharge |

### Comorbidity features (binary unless noted)
| Column | Description |
|--------|-------------|
| `charlson_score` | SQL-derived CCI (ordinal) |
| `cci_from_flags` | Partial CCI recomputed from flags (lower bound, ordinal) |
| `hf_any_diabetes` | Any diabetes (with or without complications) |
| `hf_cardiorenal` | Renal disease present (cardiorenal syndrome marker) |
| `hf_met_syndrome_proxy` | Hypertension + diabetes (metabolic syndrome proxy) |
| `hf_af_ckd` | Atrial fibrillation + renal disease |
| `hf_high_risk_triad` | AF + renal disease + diabetes |
| `hf_competing_risk` | Cancer or severe liver disease present |
| `hf_comorbidity_burden` | 0=low, 1=medium, 2=high (from charlson_score) |
| `hf_comorb_score_custom` | Weighted HF-specific comorbidity score (ordinal) |

### Lab values (continuous, imputed + normalised after Step 5)
creatinine, bun, sodium, potassium, chloride, bicarbonate, calcium,
glucose_lab, aniongap, hemoglobin, hematocrit, wbc, platelet, inr, pt, ptt

### OMR / anthropometric
omr_weight_kg, omr_bmi, omr_sbp, omr_dbp

### Missing-indicator flags (binary, added in Step 5 for 20–60% missing vars)
albumin_missing_flag (hadm window), ptt_missing_flag, inr_missing_flag,
pt_missing_flag (48h windows)

---

## Reproducing the Pipeline

```bash
# 1. Clean raw CSVs
python utils/clean_cohort_csvs.py \
    --input_dir csv \
    --output_dir csv/cleaned \
    --resource_path resources

# 2. Compute comorbidity features
python utils/compute_comorbidity.py \
    --input_dir csv/cleaned \
    --output_dir csv/comorbidity

# 3. Impute and normalise
python utils/impute_normalize.py \
    --input_dir csv/comorbidity \
    --output_dir csv/processed

# Steps 6–11: TODO
```

**Dependencies**: `numpy`, `pandas`, `scikit-learn`
(`pip install numpy pandas scikit-learn`)

---

## References

1. Charlson ME et al. *A new method of classifying prognostic comorbidity
   in longitudinal studies.* J Chronic Dis. 1987;40(5):373–383.
2. Deyo RA et al. *Adapting a clinical comorbidity index for use with ICD-9-CM
   administrative databases.* J Clin Epidemiol. 1992;45(6):613–619.
3. Pocock SJ et al. *Predictors of mortality in patients with chronic heart
   failure: incremental value of the MAGGIC risk score.* Eur Heart J.
   2013;34(23):1757–1766.
4. Yusuf S et al. *Effects of candesartan in patients with chronic heart
   failure and preserved left-ventricular ejection fraction (CHARM-Preserved).*
   Lancet. 2003;362(9386):777–781.
5. Johnson AEW et al. *MIMIC-IV, a freely accessible electronic health record
   dataset.* Sci Data. 2023;10:1.
6. Wang EW et al. *MIMIC-Extract: A data extraction, preprocessing, and
   representation pipeline for MIMIC-III.* CHIL 2020.
