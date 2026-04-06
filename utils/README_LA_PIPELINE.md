# LA Pipeline

## Summary

This repository now separates LA-related work into two branches:

- `A2` parameter-performance analysis:
  uses raw cleaned LA values and runs all three clinical windows:
  `hadm`, `48h24h`, `48h48h`
- `A3` LA-fused Cox modeling:
  uses standardized LA values and merges them into the existing HFpEF survival pipeline

The main entrypoint is `utils/la_pipeline.py`.

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
- `csv/comorbidity/hfpef_cohort_win_48h24h_comorbidity.csv`
- `csv/comorbidity/hfpef_cohort_win_48h48h_comorbidity.csv`
- `csv/processed/hfpef_cohort_win_hadm_processed.csv`
- `csv/processed/hfpef_cohort_win_48h24h_processed.csv`
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
  `csv/la_analysis/48h24h/`
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
- `csv/la_fusion/processed/hfpef_cohort_win_48h24h_processed.csv`
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
