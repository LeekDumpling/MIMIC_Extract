# -*- coding: utf-8 -*-
"""
LA 参数表现分析与 Cox 融合总编排脚本。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas 是必需依赖。\n请执行：pip install pandas") from None

try:
    _utils_dir = os.path.dirname(os.path.abspath(__file__))
    if _utils_dir not in sys.path:
        sys.path.insert(0, _utils_dir)
    from la_analysis import run_la_analysis  # type: ignore
except ImportError:
    raise ImportError("无法导入 la_analysis.py，请确认 utils 目录完整。") from None


WINDOWS: List[str] = ["hadm", "48h24h", "48h48h"]


def _resolve_path(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _prepare_join_columns(df: pd.DataFrame, study_col: str) -> pd.DataFrame:
    out = df.copy()
    out["subject_id"] = pd.to_numeric(out["subject_id"], errors="coerce").astype("Int64")
    out["_study_key"] = pd.to_numeric(out[study_col], errors="coerce").astype("Int64")
    return out


def _run_python(script_rel: str, args: Sequence[str]) -> None:
    cmd = [sys.executable, os.path.join(_repo_root(), script_rel), *args]
    print("执行：", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, cwd=_repo_root(), env=env, check=True)


def _standardized_la_table(morph_csv: str, kine_csv: str) -> pd.DataFrame:
    morph = pd.read_csv(_resolve_path(morph_csv), low_memory=False)
    kine = pd.read_csv(_resolve_path(kine_csv), low_memory=False)

    morph = _prepare_join_columns(morph, "study_id")
    kine = _prepare_join_columns(kine, "study_id")

    overlap = (
        set(morph.columns) - {"video_prefix", "subject_id", "study_id", "source_group", "_study_key"}
    ) & (
        set(kine.columns) - {"video_prefix", "subject_id", "study_id", "source_group", "_study_key"}
    )
    if overlap:
        kine = kine.rename(columns={col: f"{col}_kine" for col in overlap})

    df_img = pd.merge(
        morph,
        kine,
        on=["video_prefix", "subject_id", "study_id", "source_group", "_study_key"],
        how="outer",
        suffixes=("", "_kine"),
    )
    return df_img


def run_analysis_stage(
    windows: Sequence[str],
    analysis_root: str,
    la_processed_dir: str,
    font_family: Optional[str],
    no_plots: bool,
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    combined_candidates: List[pd.DataFrame] = []

    for window in windows:
        out_dir = os.path.join(analysis_root, window)
        summary = run_la_analysis(
            clinical_csv=f"csv/comorbidity/hfpef_cohort_win_{window}_comorbidity.csv",
            morph_csv=os.path.join(la_processed_dir, "la_morphology_wide_raw.csv"),
            kine_csv=os.path.join(la_processed_dir, "la_kinematic_wide_raw.csv"),
            feature_catalog_csv=os.path.join(la_processed_dir, "la_feature_catalog.csv"),
            feature_decisions_csv=os.path.join(la_processed_dir, "la_params_feature_decisions.csv"),
            output_dir=out_dir,
            qc_csv=os.path.join(la_processed_dir, "la_params_qc_filtered.csv"),
            window_name=window,
            font_family=font_family,
            no_plots=no_plots,
        )
        summaries.append(summary)

        candidate_path = os.path.join(out_dir, "la_candidates_for_cox.csv")
        candidate_df = pd.read_csv(candidate_path)
        if not candidate_df.empty:
            candidate_df.insert(0, "window", window)
            combined_candidates.append(candidate_df)

    summary_dir = os.path.join(analysis_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    overview = pd.DataFrame([
        {
            "window": item["window_name"],
            "n_merged": item["n_merged"],
            "n_morph_features_present": item["n_morph_features_present"],
            "n_kine_features_present": item["n_kine_features_present"],
            "n_la_candidates_for_cox": item["n_la_candidates_for_cox"],
            "clinical_indicators_used": ",".join(item["clinical_indicators_used"]),
        }
        for item in summaries
    ])
    overview.to_csv(os.path.join(summary_dir, "window_overview.csv"), index=False)

    combined = pd.concat(combined_candidates, ignore_index=True) if combined_candidates else pd.DataFrame()
    combined.to_csv(os.path.join(summary_dir, "combined_la_candidates_for_cox.csv"), index=False)

    with open(os.path.join(summary_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    return summaries


def run_fusion_stage(
    windows: Sequence[str],
    analysis_root: str,
    la_processed_dir: str,
    fusion_processed_dir: str,
) -> List[Dict[str, object]]:
    standardized_la = _standardized_la_table(
        morph_csv=os.path.join(la_processed_dir, "la_morphology_wide.csv"),
        kine_csv=os.path.join(la_processed_dir, "la_kinematic_wide.csv"),
    )
    os.makedirs(fusion_processed_dir, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    for window in windows:
        candidate_path = os.path.join(analysis_root, window, "la_candidates_for_cox.csv")
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(f"候选池文件不存在：{candidate_path}")
        candidates = pd.read_csv(candidate_path)
        if candidates.empty:
            raise RuntimeError(f"{window} 的 la_candidates_for_cox.csv 为空，无法构建 LA 融合输入。")

        selected_cols = candidates["safe_name"].dropna().astype(str).tolist()
        missing = [col for col in selected_cols if col not in standardized_la.columns]
        if missing:
            raise RuntimeError(f"{window} 候选 LA 列在标准化宽表中缺失：{missing}")

        clinical_path = _resolve_path(f"csv/processed/hfpef_cohort_win_{window}_processed.csv")
        clinical = pd.read_csv(clinical_path, low_memory=False)
        clinical = _prepare_join_columns(clinical, "index_study_id")

        la_sub = standardized_la[["subject_id", "_study_key", *selected_cols]].copy()
        dup_count = int(la_sub.duplicated(subset=["subject_id", "_study_key"]).sum())
        if dup_count > 0:
            raise RuntimeError(f"标准化 LA 宽表存在重复 subject_id + study_id 键：{dup_count} 条")

        merged = pd.merge(
            clinical,
            la_sub,
            on=["subject_id", "_study_key"],
            how="inner",
            suffixes=("", "_la"),
        )
        merged.drop(columns=["_study_key"], inplace=True)

        output_path = os.path.join(fusion_processed_dir, f"hfpef_cohort_win_{window}_processed.csv")
        merged.to_csv(output_path, index=False)

        summaries.append({
            "window": window,
            "n_clinical_input": int(len(clinical)),
            "n_la_rows": int(len(la_sub)),
            "n_merged": int(len(merged)),
            "n_features_added": int(len(selected_cols)),
            "output_path": output_path,
        })

    pd.DataFrame(summaries).to_csv(
        os.path.join(os.path.dirname(fusion_processed_dir), "fusion_summary.csv"),
        index=False,
    )
    with open(os.path.join(os.path.dirname(fusion_processed_dir), "fusion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    return summaries


def run_full_stage(
    la_processed_dir: str,
    font_family: Optional[str],
    no_plots: bool,
) -> None:
    analysis_root = _resolve_path("csv/la_analysis")
    fusion_root = _resolve_path("csv/la_fusion")
    fusion_processed_dir = os.path.join(fusion_root, "processed")
    fusion_survival_dir = os.path.join(fusion_root, "survival")
    fusion_feature_dir = os.path.join(fusion_root, "feature_selection")
    fusion_cox_dir = os.path.join(fusion_root, "cox_models")
    fusion_ph_dir = os.path.join(fusion_cox_dir, "ph_test")
    fusion_eval_dir = os.path.join(fusion_root, "model_eval")

    run_analysis_stage(
        WINDOWS,
        analysis_root=analysis_root,
        la_processed_dir=la_processed_dir,
        font_family=font_family,
        no_plots=no_plots,
    )
    run_fusion_stage(
        WINDOWS,
        analysis_root=analysis_root,
        la_processed_dir=la_processed_dir,
        fusion_processed_dir=fusion_processed_dir,
    )

    _run_python("utils/build_survival_endpoint.py", [
        "--input_dir", fusion_processed_dir,
        "--output_dir", fusion_survival_dir,
    ])
    _run_python("utils/feature_selection.py", [
        "--input_dir", fusion_survival_dir,
        "--output_dir", fusion_feature_dir,
    ])

    fit_args = [
        "--summary_json", os.path.join(fusion_feature_dir, "selection_summary.json"),
        "--survival_dir", fusion_survival_dir,
        "--output_dir", fusion_cox_dir,
    ]
    if no_plots:
        fit_args.append("--no_plots")
    _run_python("utils/fit_cox_model.py", fit_args)

    ph_args = [
        "--cox_summary", os.path.join(fusion_cox_dir, "cox_summary.json"),
        "--survival_dir", fusion_survival_dir,
        "--output_dir", fusion_ph_dir,
        "--correct_violations",
    ]
    if no_plots:
        ph_args.append("--no_plots")
    _run_python("utils/ph_assumption_test.py", ph_args)

    eval_args = [
        "--cox_summary", os.path.join(fusion_cox_dir, "cox_summary.json"),
        "--survival_dir", fusion_survival_dir,
        "--output_dir", fusion_eval_dir,
        "--tvc_summary", os.path.join(fusion_ph_dir, "stratified_correction_summary.json"),
    ]
    if no_plots:
        eval_args.append("--no_plots")
    _run_python("utils/model_evaluation.py", eval_args)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    ap_analysis = sub.add_parser("analysis", help="运行三窗口 A2 参数表现分析")
    ap_analysis.add_argument("--analysis_root", default="csv/la_analysis", help="A2 输出根目录")
    ap_analysis.add_argument("--la_processed_dir", default="csv/la_params/processed", help="LA 清洗输出目录")
    ap_analysis.add_argument("--font_family", default=None, help="显式指定中文字体")
    ap_analysis.add_argument("--no_plots", action="store_true", help="跳过 A2 图表输出")

    ap_fusion = sub.add_parser("fusion", help="构建 LA 融合后的 processed 输入")
    ap_fusion.add_argument("--analysis_root", default="csv/la_analysis", help="A2 输出根目录")
    ap_fusion.add_argument("--la_processed_dir", default="csv/la_params/processed", help="LA 清洗输出目录")
    ap_fusion.add_argument("--fusion_processed_dir", default="csv/la_fusion/processed", help="融合后的 processed 目录")

    ap_full = sub.add_parser("full", help="串行执行 analysis -> fusion -> survival -> feature_selection -> Cox -> PH -> evaluation")
    ap_full.add_argument("--la_processed_dir", default="csv/la_params/processed", help="LA 清洗输出目录")
    ap_full.add_argument("--font_family", default=None, help="显式指定中文字体")
    ap_full.add_argument("--no_plots", action="store_true", help="跳过所有可视化输出")

    args = ap.parse_args()

    if args.command == "analysis":
        run_analysis_stage(
            WINDOWS,
            analysis_root=_resolve_path(args.analysis_root),
            la_processed_dir=_resolve_path(args.la_processed_dir),
            font_family=args.font_family,
            no_plots=args.no_plots,
        )
    elif args.command == "fusion":
        run_fusion_stage(
            WINDOWS,
            analysis_root=_resolve_path(args.analysis_root),
            la_processed_dir=_resolve_path(args.la_processed_dir),
            fusion_processed_dir=_resolve_path(args.fusion_processed_dir),
        )
    elif args.command == "full":
        run_full_stage(
            la_processed_dir=_resolve_path(args.la_processed_dir),
            font_family=args.font_family,
            no_plots=args.no_plots,
        )


if __name__ == "__main__":
    main()
