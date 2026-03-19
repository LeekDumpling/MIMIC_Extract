# -*- coding: utf-8 -*-
"""
Cox PH 模型拟合脚本 — HFpEF 队列（步骤 8）

功能
----
读取特征选择汇总 JSON（csv/feature_selection/selection_summary.json）
及生存终点 CSV（csv/survival/hfpef_cohort_win_*_survival.csv），
对每个「时间窗 × 终点」组合中特征选择通过的特征集，拟合全量
无正则化 Cox 比例风险模型（lifelines.CoxPHFitter），
输出 HR 表、模型统计量和汇总 JSON。

计划调整评估
------------
基于特征选择汇总，以下情况无需调整原计划，直接跳过即可：
  - M3 特征数为 0 的组合（30d/90d ICU 窗口）——无特征可建模
  - 事件数 < MIN_EVENTS 的组合（已在特征选择阶段过滤）
其余 7 个非空特征组合（见下方汇总）进入 Cox 拟合：
  48h24h/1yr, 48h24h/any
  48h48h/any
  hadm/30d, hadm/90d, hadm/1yr, hadm/any

模型选择
--------
在 LASSO 已筛选特征基础上拟合全量无正则化 Cox 模型（penalizer=0），
以获得无偏 HR 估计和标准误。
- Breslow 基线风险估计（lifelines 默认）
- 若收敛失败，自动尝试小惩罚因子（0.05）兜底，并在结果中标注
- 对每个模型计算 Harrell C-index（一致性指数）

输出文件
--------
  csv/cox_models/{window}/cox_results_{endpoint}.csv  — HR 结果表
  csv/cox_models/cox_summary.json                     — 全组合汇总 JSON

HR 结果表列
-----------
feature, coef, HR, hr_lo95, hr_hi95, se_coef, z, p, concordance,
n_obs, n_events, penalizer, converged

用法（从仓库根目录或 utils/ 子目录均可运行）
    python utils/fit_cox_model.py
    python utils/fit_cox_model.py --summary_json csv/feature_selection/selection_summary.json
    python utils/fit_cox_model.py --window hadm --endpoint any
    python utils/fit_cox_model.py --dry_run
"""

import argparse
import glob
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np

    _blocked = [m for m in ("numexpr", "bottleneck") if m not in sys.modules]
    for _m in _blocked:
        sys.modules[_m] = None  # type: ignore[assignment]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            import pandas as pd
    finally:
        for _m in _blocked:
            sys.modules.pop(_m, None)
except (ImportError, AttributeError, ValueError):
    raise ImportError(
        "numpy 和 pandas 是必需依赖。\n请执行：pip install numpy pandas"
    ) from None

try:
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError
except ImportError:
    raise ImportError(
        "lifelines 是必需依赖。\n请执行：pip install lifelines"
    ) from None

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------

# 终点定义：(event_col, time_col, 描述)
ENDPOINTS: Dict[str, Tuple[str, str, str]] = {
    "30d":  ("event_30d",  "time_30d",  "出院后 30 天死亡"),
    "90d":  ("event_90d",  "time_90d",  "出院后 90 天死亡"),
    "1yr":  ("event_1yr",  "time_1yr",  "出院后 1 年死亡"),
    "any":  ("event_any",  "time_any",  "出院后任意时间死亡（受删失约束）"),
}

# 有限时间截止点（天）及对应列名后缀（用于内联构建衍生终点）
HORIZONS: List[tuple] = [
    (30,   "30d"),
    (90,   "90d"),
    (365,  "1yr"),
    (None, "any"),
]

# 不参与建模的列（与 feature_selection.py 保持一致）
NON_FEATURE_COLS = {
    "subject_id", "hadm_id", "index_study_id", "index_study_datetime",
    "index_admittime", "index_dischtime", "a4c_dicom_filepath",
    "death_date", "last_dischtime", "censor_date",
    "hospital_expire_flag", "died_inhosp", "died_post_dc",
    "days_survived_post_dc", "died_30d", "died_90d", "died_1yr",
    "os_event", "os_days",
    "event_30d", "time_30d",
    "event_90d", "time_90d",
    "event_1yr", "time_1yr",
    "event_any", "time_any",
}

# 兜底惩罚因子（仅收敛失败时使用）
FALLBACK_PENALIZER = 0.05

MIN_EVENTS = 10


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    """
    解析相对路径。

    若路径相对于当前工作目录不存在，则尝试相对于仓库根目录解析，
    允许脚本从 utils/ 子目录直接启动。
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


def _build_survival_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 os_event / os_days 列内联构建衍生生存终点列。

    与 build_survival_endpoint.build_endpoints() 逻辑相同，
    用于在生存 CSV 不可用时从原始队列 CSV 直接读取。
    """
    df = df.copy()
    os_event = pd.to_numeric(df["os_event"], errors="coerce")
    os_days  = pd.to_numeric(df["os_days"],  errors="coerce")
    inhosp_mask = os_event.isna()

    for horizon, suffix in HORIZONS:
        if horizon is None:
            event_col_vals = os_event.copy()
            time_col_vals  = os_days.copy()
        else:
            event_col_vals = pd.Series(np.nan, index=df.index, dtype=float)
            time_col_vals  = pd.Series(np.nan, index=df.index, dtype=float)
            alive_mask = ~inhosp_mask
            event_col_vals[alive_mask] = (
                (os_event[alive_mask] == 1) & (os_days[alive_mask] <= horizon)
            ).astype(float)
            time_col_vals[alive_mask] = os_days[alive_mask].clip(upper=horizon)

        df[f"event_{suffix}"] = event_col_vals
        df[f"time_{suffix}"]  = time_col_vals.round(1)

    return df


def _load_survival_df(
    window: str,
    survival_dir: str,
    fallback_dir: str,
) -> Optional[pd.DataFrame]:
    """
    按优先级加载生存数据 DataFrame。

    1. 尝试 {survival_dir}/hfpef_cohort_win_{window}_survival.csv
    2. 若不存在，尝试 {fallback_dir}/hfpef_cohort_win_{window}.csv
       并内联构建衍生终点列。
    3. 两者均不存在则返回 None。
    """
    survival_path = os.path.join(survival_dir, f"hfpef_cohort_win_{window}_survival.csv")
    if os.path.exists(survival_path):
        df = pd.read_csv(survival_path, low_memory=False)
        return df

    # 回退到原始 CSV
    raw_path = os.path.join(fallback_dir, f"hfpef_cohort_win_{window}.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, low_memory=False)
        # 检查是否已有衍生终点
        if "event_30d" not in df.columns:
            df = _build_survival_endpoints(df)
        return df

    return None


# ---------------------------------------------------------------------------
# 核心拟合函数
# ---------------------------------------------------------------------------

def fit_cox(
    df_cox: pd.DataFrame,
    features: List[str],
    event_col: str,
    time_col: str,
    label: str = "",
) -> Dict:
    """
    在 df_cox 上对给定特征集拟合全量 Cox PH 模型。

    优先无惩罚拟合（penalizer=0）；若收敛失败，
    自动改用 penalizer=FALLBACK_PENALIZER 并在结果中标注。

    返回包含以下字段的字典：
      coef_df      : pd.DataFrame — HR 结果表
      concordance  : float
      n_obs        : int
      n_events     : int
      penalizer    : float
      converged    : bool
      error        : str（仅拟合彻底失败时非空）
    """
    cols = features + [time_col, event_col]
    missing_feats = [c for c in features if c not in df_cox.columns]
    if missing_feats:
        print(f"  [{label}] 警告：以下特征在数据中不存在，将跳过：{missing_feats}")
        features = [f for f in features if f in df_cox.columns]
        if not features:
            return {
                "coef_df": pd.DataFrame(),
                "concordance": np.nan,
                "n_obs": len(df_cox),
                "n_events": int(pd.to_numeric(df_cox.get(event_col), errors="coerce").sum()),
                "penalizer": 0.0,
                "converged": False,
                "error": f"所有特征均不在数据中: {missing_feats}",
            }

    df_model = df_cox[features + [time_col, event_col]].copy().dropna()

    # 将字符串/分类列编码为数值（仅二值列；不改原 df）
    for col in features:
        if df_model[col].dtype == object:
            uniq = df_model[col].dropna().unique()
            if len(uniq) == 2:
                # 将两个类别映射为 0/1（字典序较大的为 1）
                mapping = {v: i for i, v in enumerate(sorted(uniq))}
                df_model[col] = df_model[col].map(mapping)
                print(f"  [{label}] 编码字符串列 '{col}'：{mapping}")
            else:
                print(f"  [{label}] 警告：列 '{col}' 为非数值且非二值，跳过该列")
                features = [f for f in features if f != col]

    df_model = df_model[features + [time_col, event_col]].dropna()
    n_obs    = len(df_model)
    n_events = int(df_model[event_col].sum())

    def _try_fit(penalizer: float) -> Tuple[Optional[CoxPHFitter], bool]:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                cph = CoxPHFitter(penalizer=penalizer,
                                  baseline_estimation_method="breslow")
                cph.fit(df_model,
                        duration_col=time_col,
                        event_col=event_col,
                        show_progress=False)
            return cph, True
        except (ConvergenceError, Exception):
            return None, False

    # 首选：无惩罚
    cph, ok = _try_fit(0.0)
    penalizer_used = 0.0
    converged = ok

    # 兜底：小惩罚
    if not ok:
        print(f"  [{label}] 无惩罚拟合失败，改用 penalizer={FALLBACK_PENALIZER}")
        cph, ok = _try_fit(FALLBACK_PENALIZER)
        penalizer_used = FALLBACK_PENALIZER
        converged = ok

    if cph is None:
        return {
            "coef_df": pd.DataFrame(),
            "concordance": np.nan,
            "n_obs": n_obs,
            "n_events": n_events,
            "penalizer": penalizer_used,
            "converged": False,
            "error": "Cox 模型收敛失败（含兜底 penalizer）",
        }

    # 整理系数表
    summary = cph.summary.copy().reset_index()
    summary = summary.rename(columns={
        "covariate":            "feature",
        "coef":                 "coef",
        "exp(coef)":            "HR",
        "exp(coef) lower 95%":  "hr_lo95",
        "exp(coef) upper 95%":  "hr_hi95",
        "se(coef)":             "se_coef",
        "z":                    "z",
        "p":                    "p",
    })
    keep_cols = ["feature", "coef", "HR", "hr_lo95", "hr_hi95",
                 "se_coef", "z", "p"]
    for col in keep_cols:
        if col not in summary.columns:
            summary[col] = np.nan
    summary = summary[keep_cols].copy()
    summary["concordance"] = cph.concordance_index_
    summary["n_obs"]       = n_obs
    summary["n_events"]    = n_events
    summary["penalizer"]   = penalizer_used
    summary["converged"]   = converged

    return {
        "coef_df":    summary,
        "concordance": cph.concordance_index_,
        "n_obs":       n_obs,
        "n_events":    n_events,
        "penalizer":   penalizer_used,
        "converged":   converged,
        "error":       "",
    }


# ---------------------------------------------------------------------------
# 每个 window × endpoint 的处理入口
# ---------------------------------------------------------------------------

def process_combination(
    df: pd.DataFrame,
    window: str,
    endpoint: str,
    final_features: List[str],
    output_dir: str,
    dry_run: bool = False,
) -> Dict:
    """
    对单个时间窗 + 终点组合拟合 Cox PH 模型，写出结果文件。

    返回包含关键统计的 dict（供 JSON 汇总使用）。
    """
    event_col, time_col, ep_desc = ENDPOINTS[endpoint]
    label = f"{window}/{endpoint}"

    # 过滤：仅院外患者且当前终点非缺失
    df_cox = df[df["os_event"].notna()].copy()
    df_cox = df_cox[df_cox[event_col].notna() & df_cox[time_col].notna()].copy()
    df_cox = df_cox.reset_index(drop=True)

    n_total  = len(df_cox)
    n_events = int(df_cox[event_col].sum())

    print(f"\n  [{label}] n={n_total}, 事件={n_events}, 特征数={len(final_features)}")

    if n_events < MIN_EVENTS:
        print(f"  [{label}] 跳过：事件数 < {MIN_EVENTS}")
        return {
            "window": window, "endpoint": endpoint, "endpoint_desc": ep_desc,
            "n_total": n_total, "n_events": n_events,
            "skipped": True, "reason": "too_few_events",
            "final_features": final_features,
        }

    if not final_features:
        print(f"  [{label}] 跳过：无最终特征（特征选择 M3=0）")
        return {
            "window": window, "endpoint": endpoint, "endpoint_desc": ep_desc,
            "n_total": n_total, "n_events": n_events,
            "skipped": True, "reason": "no_selected_features",
            "final_features": [],
        }

    print(f"  [{label}] 拟合 Cox PH 模型（特征：{final_features}）")
    result = fit_cox(df_cox, final_features, event_col, time_col, label=label)

    if result.get("error"):
        print(f"  [{label}] 错误：{result['error']}")
        return {
            "window": window, "endpoint": endpoint, "endpoint_desc": ep_desc,
            "n_total": n_total, "n_events": n_events,
            "skipped": True, "reason": result["error"],
            "final_features": final_features,
        }

    print(f"  [{label}] 收敛={result['converged']}, "
          f"penalizer={result['penalizer']}, "
          f"C-index={result['concordance']:.4f}")

    if not dry_run and not result["coef_df"].empty:
        win_dir = os.path.join(output_dir, window)
        os.makedirs(win_dir, exist_ok=True)
        out_path = os.path.join(win_dir, f"cox_results_{endpoint}.csv")
        result["coef_df"].to_csv(out_path, index=False)
        print(f"  [{label}] 结果已写出 → {out_path}")

    return {
        "window":        window,
        "endpoint":      endpoint,
        "endpoint_desc": ep_desc,
        "n_total":       n_total,
        "n_events":      n_events,
        "skipped":       False,
        "concordance":   round(float(result["concordance"]), 4),
        "penalizer":     result["penalizer"],
        "converged":     result["converged"],
        "final_features": final_features,
        "n_features":    len(final_features),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--summary_json", default="csv/feature_selection/selection_summary.json",
                    help="特征选择汇总 JSON 路径（默认：csv/feature_selection/selection_summary.json）")
    ap.add_argument("--survival_dir", default="csv/survival",
                    help="生存终点 CSV 所在目录（默认：csv/survival）")
    ap.add_argument("--raw_dir", default="csv",
                    help="原始队列 CSV 目录（survival_dir 不存在时的回退，默认：csv）")
    ap.add_argument("--output_dir", default="csv/cox_models",
                    help="Cox 模型结果输出目录（默认：csv/cox_models）")
    ap.add_argument("--endpoint", default=None,
                    choices=list(ENDPOINTS.keys()),
                    help="仅处理指定终点（默认：全部）")
    ap.add_argument("--window", default=None,
                    help="仅处理指定时间窗关键词（如 hadm、48h24h）")
    ap.add_argument("--dry_run", action="store_true",
                    help="仅打印摘要，不写出任何文件")
    args = ap.parse_args()

    args.summary_json = _resolve_path(args.summary_json)
    args.survival_dir = _resolve_path(args.survival_dir)
    args.raw_dir      = _resolve_path(args.raw_dir)
    args.output_dir   = _resolve_path(args.output_dir)

    # ---- 读取特征选择汇总 ------------------------------------------------
    if not os.path.exists(args.summary_json):
        print(f"特征选择汇总 JSON 不存在：{args.summary_json}")
        print("请先运行：python utils/feature_selection.py")
        sys.exit(1)

    with open(args.summary_json, "r", encoding="utf-8") as f:
        all_fs = json.load(f)

    print(f"读入特征选择汇总：{len(all_fs)} 条记录")

    # ---- 过滤 window / endpoint ------------------------------------------
    if args.window:
        all_fs = [s for s in all_fs if s.get("window") == args.window]
    if args.endpoint:
        all_fs = [s for s in all_fs if s.get("endpoint") == args.endpoint]

    if not all_fs:
        print("过滤后无匹配记录，退出。")
        return

    # ---- 按 window 分组读取 CSV ------------------------------------------
    # 将 all_fs 按 window 分组，减少重复读取
    from collections import defaultdict
    by_window: Dict[str, list] = defaultdict(list)
    for entry in all_fs:
        by_window[entry["window"]].append(entry)

    all_summaries = []

    for window, entries in sorted(by_window.items()):
        print(f"\n{'='*60}")
        print(f"时间窗：{window}")
        print(f"{'='*60}")

        df = _load_survival_df(window, args.survival_dir, args.raw_dir)
        if df is None:
            print(f"  找不到时间窗 '{window}' 的数据文件，跳过。")
            for e in entries:
                all_summaries.append({
                    "window": window, "endpoint": e.get("endpoint"),
                    "skipped": True, "reason": "data_file_not_found",
                    "final_features": [],
                })
            continue

        print(f"  读入 {len(df)} 行 × {len(df.columns)} 列")

        for entry in entries:
            ep = entry.get("endpoint")
            if ep not in ENDPOINTS:
                continue
            if entry.get("skipped"):
                print(f"\n  [{window}/{ep}] 特征选择已跳过（{entry.get('reason', '')}），跳过 Cox 拟合")
                all_summaries.append({
                    "window": window, "endpoint": ep,
                    "skipped": True, "reason": f"fs_skipped:{entry.get('reason', '')}",
                    "final_features": [],
                })
                continue

            final_features = entry.get("final_features", [])
            summary = process_combination(
                df, window, ep, final_features,
                args.output_dir, dry_run=args.dry_run,
            )
            all_summaries.append(summary)

    # ---- 写出全局 JSON 汇总 ----------------------------------------------
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "cox_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n全局汇总 JSON → {summary_path}")

    # ---- 打印汇总表 -------------------------------------------------------
    print(f"\n{'='*60}")
    print("Cox PH 模型拟合汇总")
    print(f"{'='*60}")
    hdr = f"{'窗口+终点':<20} {'n':>6} {'事件':>6} {'特征数':>6} {'C-index':>9} {'收敛':>5} 状态"
    print(hdr)
    print("-" * 70)
    for s in all_summaries:
        tag = f"{s.get('window','')}/{s.get('endpoint','')}"
        if s.get("skipped"):
            reason = s.get("reason", "")
            print(f"  {tag:<20} {'—':>6} {'—':>6} {'—':>6} {'—':>9} {'—':>5} 跳过（{reason}）")
        else:
            c_idx = f"{s.get('concordance', float('nan')):.4f}"
            conv  = "✓" if s.get("converged") else "✗"
            pen   = f"pen={s.get('penalizer', 0)}" if s.get("penalizer", 0) > 0 else ""
            print(f"  {tag:<20} {s.get('n_total',0):>6} {s.get('n_events',0):>6} "
                  f"{s.get('n_features',0):>6} {c_idx:>9} {conv:>5} {pen}")

    print(f"\n完成。结果已保存至：{args.output_dir}/")


if __name__ == "__main__":
    main()
