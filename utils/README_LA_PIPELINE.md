# LA Pipeline

## Summary

This repository now separates LA-related work into two branches:

- `A2` parameter-performance analysis:
  uses raw cleaned LA values and runs all three clinical windows:
  `hadm`, `24h24h`, `48h48h`
- `A3` LA-fused Cox modeling:
  uses standardized LA values and merges them into the existing HFpEF survival pipeline

The main entrypoint is `utils/la_pipeline.py`.

## Current Status

The current LA branch has been implemented and run end-to-end on the latest
available `csv/la_params/` source in this workspace.

Current progress:

- `A1` LA cleaning:
  completed with dual outputs (raw analysis tables + standardized modeling tables)
- `A2` three-window parameter-performance analysis:
  completed for `hadm`, `24h24h`, `48h48h`
- `A3` LA-fused Cox pipeline:
  completed through feature selection, Cox fitting, PH testing, and model evaluation

The latest validated LA cleaning output used for the current downstream results is:

- `csv/la_params/processed_20260407_0732_rerun/`

The latest downstream outputs for the current re-window comparison are:

- `csv/la_analysis_rewindow/`
- `csv/la_fusion_rewindow/feature_selection/`
- `csv/la_fusion_rewindow/cox_models/`
- `csv/la_fusion_rewindow/model_eval/`

## Primary Model

The current primary model is:

- `48h48h / any`

Reason:

- it is currently the preferred main model for reporting
- it passed the present EPV gate used by the downstream scripts
- it completed full internal validation
- it passed the current PH test without requiring stratification correction
- its calibration pattern is visibly more stable than `24h24h / any`
- it retains both morphology and kinematic LA variables

Current primary-model discrimination:

- apparent C-index: `0.6454`
- optimism-corrected C-index: `0.6263`
- internal-validation sample size: `n=450`
- internal-validation event count: `233`

Current primary-model variables:

- clinical:
  `hemoglobin`, `peripheral_vascular_disease`, `bun`, `hf_cardiorenal`,
  `hf_competing_risk`, `myocardial_infarct`, `chronic_pulmonary_disease`
- morphology:
  `LAVmin-i`, `LA_eccentricity_index`
- kinematic:
  `Time_to_peak_LASrR__from_LAVmin`

Current primary-model LA effects:

- `LAVmin-i`
  HR `1.2330`, 95% CI `1.0498–1.4482`, p=`0.0107`
- `LA_eccentricity_index`
  HR `1.2447`, 95% CI `1.0746–1.4417`, p=`0.0035`
- `Time_to_peak_LASrR__from_LAVmin`
  HR `0.8721`, 95% CI `0.7639–0.9956`, p=`0.0429`

Current primary-model calibration / PH status:

- calibration table:
  risk groups increase monotonically from `0.1000` to `0.4889`
- PH test:
  currently passes without stratification correction
- DCA:
  model net benefit is consistently above or comparable to treat-all at low-to-mid threshold ranges

Primary-model interpretation for the paper:

- this model provides the best balance between discrimination, calibration, and PH stability
- compared with `24h24h / any`, its corrected C-index is slightly lower, but the calibration behaviour is materially more plausible
- compared with `hadm / any`, it avoids relying on a stratified correction path and remains easier to present as a primary reportable model
- for the thesis main text, `24h24h / any` should be described as the higher-discrimination comparator rather than the primary reportable model

Paper-ready summary:

- primary reporting window / endpoint:
  `48h48h / any`
- recommended wording:
  this model is preferred because it preserves independent LA structural and kinematic information while maintaining acceptable discrimination, monotonic calibration, and a clean PH result
- LA contribution in the final multivariable Cox model:
  `LAVmin-i` and `LA_eccentricity_index` act as independent adverse markers, whereas `Time_to_peak_LASrR__from_LAVmin` contributes additional inverse-direction kinematic information

Relevant result files:

- `csv/la_fusion_rewindow/cox_models/48h48h/cox_results_any.csv`
- `csv/la_fusion_rewindow/cox_models/ph_test/ph_test_48h48h_any.csv`
- `csv/la_fusion_rewindow/model_eval/48h48h_any/bootstrap_cindex.json`
- `csv/la_fusion_rewindow/model_eval/48h48h_any/calibration.csv`
- `csv/la_fusion_rewindow/model_eval/48h48h_any/dca.csv`

## Current Evaluated Models

At the moment, the models that completed internal validation are:

- `24h24h / any`
  corrected C-index `0.6614`
- `48h48h / 1yr`
  corrected C-index `0.6054`
- `48h48h / any`
  corrected C-index `0.6263`
- `hadm / any`
  corrected C-index `0.6130`

Notes:

- not every fitted Cox model is treated as a final reportable model
- the current evaluation stage only proceeds for models that satisfy the
  script's EPV gate
- `48h24h` is now treated as a deprecated historical window and is no longer used in the active three-window comparison
- the active comparison set is `hadm`, `24h24h`, and `48h48h`
- although `24h24h / any` achieves the highest corrected C-index, it is not the preferred main-text model because of poor calibration shape and unresolved PH issues after stratification
- `48h48h / any` is the recommended main-text model at this stage

## Inputs

Required LA cleaning outputs in `csv/la_params/processed/`:

- `la_morphology_wide.csv`
- `la_kinematic_wide.csv`
- `la_morphology_wide_raw.csv`
- `la_kinematic_wide_raw.csv`
- `la_feature_catalog.csv`
- `la_params_feature_decisions.csv`
- `la_params_qc_filtered.csv`

Required clinical inputs:

- `csv/comorbidity/hfpef_cohort_win_hadm_comorbidity.csv`
- `csv/comorbidity/hfpef_cohort_win_24h24h_comorbidity.csv`
- `csv/comorbidity/hfpef_cohort_win_48h48h_comorbidity.csv`
- `csv/processed/hfpef_cohort_win_hadm_processed.csv`
- `csv/processed/hfpef_cohort_win_24h24h_processed.csv`
- `csv/processed/hfpef_cohort_win_48h48h_processed.csv`

## A1 refresh

If the raw LA wide tables and feature catalog are missing, rerun the LA cleaning step first:

```bat
python utils/clean_la_params.py --morphology csv/la_params/final_morphology_results.csv --kinematic csv/la_params/final_kinematic_stats.csv --qc csv/la_params/final_qc.csv --output_dir csv/la_params/processed
```

This step now writes both modeling tables and analysis tables:

- modeling tables:
  `la_morphology_wide.csv`, `la_kinematic_wide.csv`
- raw analysis tables:
  `la_morphology_wide_raw.csv`, `la_kinematic_wide_raw.csv`
- metadata:
  `la_feature_catalog.csv`

## A2 analysis

Run all three windows:

```bat
python utils/la_pipeline.py analysis
```

Optional Chinese font:

```bat
python utils/la_pipeline.py analysis --font_family "Microsoft YaHei"
```

Outputs:

- per-window:
  `csv/la_analysis/hadm/`
  `csv/la_analysis/24h24h/`
  `csv/la_analysis/48h48h/`
- cross-window summary:
  `csv/la_analysis/summary/`

Each window writes:

- `merged_dataset.csv`
- `availability_summary.csv`
- `distribution_stats_raw.csv`
- `correlation_la_clinical.csv`
- `correlation_la_internal.csv`
- `high_corr_pairs_la.csv`
- `vif_report_la.csv`
- `la_candidates_for_cox.csv`
- `analysis_summary.json`
- `figures/availability_bar.png`
- `figures/distribution_morphology.png`
- `figures/distribution_kinematic_p01.png`, `p02...`
- `figures/correlation_la_clinical.png`
- `figures/correlation_la_morphology.png`
- `figures/correlation_la_kinematic.png`

## A3 fusion

Build LA-fused processed inputs:

```bat
python utils/la_pipeline.py fusion
```

Outputs:

- `csv/la_fusion/processed/hfpef_cohort_win_hadm_processed.csv`
- `csv/la_fusion/processed/hfpef_cohort_win_24h24h_processed.csv`
- `csv/la_fusion/processed/hfpef_cohort_win_48h48h_processed.csv`
- `csv/la_fusion/fusion_summary.csv`

Fusion rules:

- join key:
  `subject_id + index_study_id == subject_id + study_id`
- only LA columns listed in each window's `la_candidates_for_cox.csv` are added
- `video_prefix`, `source_group`, `study_id`, `*_review_flag` are not carried into the fused modeling tables

## Full pipeline

Run the full LA branch:

```bat
python utils/la_pipeline.py full
```

This executes:

1. `A2` three-window LA analysis
2. LA fusion into `csv/la_fusion/processed/`
3. survival endpoint generation into `csv/la_fusion/survival/`
4. feature selection into `csv/la_fusion/feature_selection/`
5. Cox fitting into `csv/la_fusion/cox_models/`
6. PH testing into `csv/la_fusion/cox_models/ph_test/`
7. model evaluation into `csv/la_fusion/model_eval/`

Use a specific Chinese font for the A2 plots:

```bat
python utils/la_pipeline.py full --font_family "Microsoft YaHei"
```

Skip all plots:

```bat
python utils/la_pipeline.py full --no_plots
```

## Chinese visualization

Chinese plotting is handled centrally through `utils/ph_viz.py`.

Font priority:

1. `Microsoft YaHei`
2. `SimHei`
3. `PingFang SC`
4. `Noto Sans CJK SC`
5. `Noto Sans SC`
6. `Source Han Sans CN`
7. `FangSong`

Rules:

- `matplotlib.rcParams["axes.unicode_minus"] = False`
- if plotting is enabled and no supported Chinese font is available, the analysis stops with an explicit error
- no garbled Chinese plots are written

## Current scope limits

The current repository does not include the following fields in the analysis-ready clinical tables:

- `LVEF`
- diastolic echo indices

Therefore the "LA parameters vs. cardiac function indicators" section currently uses only the variables that
exist in the window-level `comorbidity` tables:

- `ntprobnp`
- `troponin_t`
- `crp`
- `albumin`
- `creatinine`
