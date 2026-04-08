# -*- coding: utf-8 -*-
"""
特征选择脚本 — HFpEF 队列 Cox 比例风险分析（步骤 7）

功能
----
读取步骤 6 生成的生存终点 CSV（csv/survival/hfpef_cohort_win_*_survival.csv），
对每个时间窗 × 终点组合依次执行三步特征选择方法：

方法 1 — 单变量 Cox 筛选
    对每个特征单独拟合 Cox 模型（lifelines.CoxPHFitter），
    记录 HR / 95%CI / p 值 / 一致性指数（Harrell C-index）。
    保留 Wald 检验 p < 0.10 的特征（宽松阈值）。

方法 2 — 共线性检查（VIF）
    对方法 1 通过的候选特征计算方差膨胀因子（statsmodels VIF）。
    迭代删除 VIF 最大且 > VIF_THRESHOLD（默认 10）的特征，
    按临床优先规则选择保留项（creatinine 优先于 bun 等）。
    重复直到所有 VIF ≤ VIF_THRESHOLD 或候选特征 ≤ 2。

方法 3 — LASSO 惩罚 Cox（lifelines，l1_ratio=1.0）
    在方法 2 通过的候选特征上，对一组 penalizer 候选值进行 k 折交叉验证，
    选择最优 penalizer（对数偏似然最大化）。
    在最优 penalizer 下重新拟合全量数据，系数非零的特征构成最终特征集。

报告
----
  csv/feature_selection/{window}/method1_univariate_{endpoint}.csv  — 全特征单变量结果
  csv/feature_selection/{window}/method1_candidates_{endpoint}.csv  — p<0.10 候选集
  csv/feature_selection/{window}/method2_vif_{endpoint}.csv         — VIF 计算全程记录
  csv/feature_selection/{window}/method2_candidates_{endpoint}.csv  — VIF 过滤后候选集
  csv/feature_selection/{window}/method3_cv_{endpoint}.csv          — 交叉验证 penalizer 结果
  csv/feature_selection/{window}/method3_lasso_{endpoint}.csv       — LASSO 最终系数
  csv/feature_selection/{window}/final_features_{endpoint}.csv      — 三步汇总表
  csv/feature_selection/selection_summary.json                       — 所有 window×endpoint 汇总 JSON

用法（从仓库根目录或 utils/ 子目录均可运行）
    python utils/feature_selection.py
    python utils/feature_selection.py --input_dir csv/survival --output_dir csv/feature_selection
    python utils/feature_selection.py --endpoint any --window hadm
    python utils/feature_selection.py --dry_run
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

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif_raw
except ImportError:
    raise ImportError(
        "statsmodels 是必需依赖。\n请执行：pip install statsmodels"
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

# 不参与建模的列
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

# VIF 去除时的临床优先级（列表中靠前者优先保留）
# 当多个特征 VIF 超阈值时，当前策略倾向于：
#   1. 先删临床变量，尽量保留 LA 参数
#   2. 在临床变量内部，再按下表从后往前删（越靠后越先删）
VIF_PRIORITY = [
    # 实验室——保留肌酐，去除 BUN（重叠度高）
    "creatinine",
    "bun",
    # 血细胞——保留血红蛋白，去除血细胞比容
    "hemoglobin",
    "hematocrit",
    # 凝血——保留 INR（单一综合指标）
    "inr",
    "pt",
    "ptt",
    # 合并症评分——保留 charlson_score（标准量表）
    "charlson_score",
    "cci_from_flags",
    # 糖尿病亚型——保留综合标志
    "hf_any_diabetes",
    "diabetes_without_cc",
    "diabetes_with_cc",
    # 心肾——保留综合标志
    "hf_cardiorenal",
    "renal_disease",
    # 代谢综合征代理——保留综合标志
    "hf_met_syndrome_proxy",
    "hypertension",
    # 复合综合征——保留综合标志
    "hf_af_ckd",
    "atrial_fibrillation",
    # BMI 和体重
    "omr_bmi",
    "omr_weight_kg",
    # 电解质
    "chloride",
    "sodium",
    "bicarbonate",
    "aniongap",
    # 葡萄糖和钙
    "glucose_lab",
    "calcium",
    # 血压
    "omr_sbp",
    "omr_dbp",
    # 血小板和 WBC
    "platelet",
    "wbc",
]

FEATURE_GROUPS = ("clinical", "morphology", "kinematic")


def _is_la_feature(feature: str) -> bool:
    """
    Identify LA-derived imaging features by naming convention.

    This is used only in VIF filtering to prefer keeping LA variables when
    they are collinear with clinical covariates.
    """
    feature = str(feature)
    return (
        feature.startswith("LA_")
        or feature.startswith("LAV")
        or feature.startswith("LAD-")
        or feature.startswith("MAT_")
        or feature.startswith("TGAR")
        or "__" in feature
    )


def _feature_group(feature: str) -> str:
    """Classify a feature into clinical / morphology / kinematic."""
    feature = str(feature)
    if "__" in feature or feature.startswith("MAT_") or feature.startswith("TGAR"):
        return "kinematic"
    if feature.startswith("LA_") or feature.startswith("LAV") or feature.startswith("LAD-"):
        return "morphology"
    return "clinical"


def _feature_rank_key(
    feature: str,
    m1_lookup: Dict[str, Dict],
    df_cox: pd.DataFrame,
) -> tuple:
    """Ranking key where smaller is better for representative selection."""
    info = m1_lookup.get(feature, {})
    p_val = info.get("p", np.nan)
    p_rank = float(p_val) if pd.notna(p_val) else float("inf")
    n_obs = int(info.get("n_obs", 0) or 0)
    missing_rate = float(df_cox[feature].isna().mean()) if feature in df_cox.columns else 1.0
    return (p_rank, -n_obs, missing_rate, str(feature))


def _with_feature_groups(df: pd.DataFrame, feature_col: str = "feature") -> pd.DataFrame:
    """Attach feature_group column if feature column exists."""
    if feature_col in df.columns and "feature_group" not in df.columns:
        df = df.copy()
        df["feature_group"] = df[feature_col].astype(str).map(_feature_group)
    return df


def _select_branch_aware_m1_candidates(
    results_df: pd.DataFrame,
    df_cox: pd.DataFrame,
    p_thresholds: Dict[str, float],
    min_candidates: Dict[str, int],
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply branch-aware thresholds and minimum branch quotas after M1."""
    out = _with_feature_groups(results_df)
    out["m1_threshold"] = out["feature_group"].map(p_thresholds).astype(float)
    out["selected_by_branch_minimum"] = False
    out["significant"] = (
        out["converged"].fillna(False)
        & out["p"].notna()
        & (out["p"] < out["m1_threshold"])
    )

    selected = out.loc[out["significant"], "feature"].astype(str).tolist()
    m1_lookup = out.set_index("feature").to_dict("index")

    for group, min_n in min_candidates.items():
        current = [f for f in selected if _feature_group(f) == group]
        if len(current) >= min_n:
            continue
        extras_df = out[
            (out["feature_group"] == group)
            & out["converged"].fillna(False)
            & (~out["feature"].isin(selected))
        ].copy()
        extras_df["sort_key"] = extras_df["feature"].map(
            lambda feat: _feature_rank_key(str(feat), m1_lookup, df_cox)
        )
        extras_df = extras_df.sort_values("sort_key")
        needed = max(0, min_n - len(current))
        chosen = extras_df["feature"].astype(str).head(needed).tolist()
        if chosen:
            selected.extend(chosen)
            out.loc[out["feature"].isin(chosen), "selected_by_branch_minimum"] = True

    selected = list(dict.fromkeys(selected))
    out["m1_selected"] = out["feature"].isin(selected)
    return out, selected


def _compress_branch_candidates(
    df_cox: pd.DataFrame,
    candidate_features: List[str],
    m1_df: pd.DataFrame,
    corr_threshold: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compress highly correlated features within morphology/kinematic branches."""
    m1_lookup = m1_df.set_index("feature").to_dict("index") if not m1_df.empty else {}
    log_rows: List[Dict] = []
    retained: List[str] = []
    cluster_id = 0

    for group in FEATURE_GROUPS:
        group_feats = [f for f in candidate_features if _feature_group(f) == group]
        if len(group_feats) <= 1 or group == "clinical":
            for feat in group_feats:
                retained.append(feat)
                log_rows.append({
                    "iteration": 0,
                    "feature": feat,
                    "feature_group": group,
                    "vif": np.nan,
                    "action": "kept_pre_vif",
                    "cluster_id": np.nan,
                    "cluster_repr": feat,
                })
            continue

        corr = df_cox[group_feats].corr(method="spearman").abs()
        visited = set()

        for feat in group_feats:
            if feat in visited:
                continue
            stack = [feat]
            component: List[str] = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                component.append(cur)
                neighbours = corr.index[(corr.loc[cur] >= corr_threshold) & (corr.index != cur)].tolist()
                for nb in neighbours:
                    if nb not in visited:
                        stack.append(nb)

            component = sorted(set(component))
            cluster_id += 1
            if len(component) == 1:
                rep = component[0]
            else:
                rep = min(component, key=lambda f: _feature_rank_key(f, m1_lookup, df_cox))
            retained.append(rep)
            for member in component:
                log_rows.append({
                    "iteration": 0,
                    "feature": member,
                    "feature_group": group,
                    "vif": np.nan,
                    "action": (
                        f"kept_corr_cluster_repr>={corr_threshold:.2f}"
                        if member == rep and len(component) > 1
                        else f"removed_corr>={corr_threshold:.2f}"
                        if member != rep
                        else "kept_pre_vif"
                    ),
                    "cluster_id": cluster_id,
                    "cluster_repr": rep,
                })

    retained = list(dict.fromkeys(retained))
    return pd.DataFrame(log_rows), retained

VIF_THRESHOLD = 10.0
# 惩罚系数候选范围：从极小值（宽松正则化）到较大值（强正则化），确保在事件数较少时
# 仍能找到保留至少一个特征的惩罚强度
LASSO_PENALIZERS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
CV_FOLDS = 5
UNIVARIATE_P_THRESHOLD = 0.10
MIN_EVENTS = 10  # 最少事件数，低于此数不拟合 Cox 模型


# ---------------------------------------------------------------------------
# 路径辅助
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    """当相对路径在当前目录下不存在时，自动回退到仓库根目录。"""
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    candidate = os.path.join(repo_root, path)
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# 方法 1 — 单变量 Cox 筛选
# ---------------------------------------------------------------------------

def method1_univariate(
    df_cox: pd.DataFrame,
    feature_cols: List[str],
    event_col: str,
    time_col: str,
    p_threshold: float = UNIVARIATE_P_THRESHOLD,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    对 df_cox 中每个特征单独拟合 Cox 模型，记录全部统计量。

    返回
    ----
    (results_df, candidate_features)
    results_df       : 每行一个特征，含 coef/HR/CI/p/c_index
    candidate_features : p < threshold 的特征名列表
    """
    records = []
    for feat in feature_cols:
        sub_df = df_cox[[feat, time_col, event_col]].dropna()
        col_data = sub_df[feat]
        # 跳过方差为零的列（常数列）
        if col_data.nunique() <= 1:
            records.append({
                "feature": feat,
                "coef": np.nan,
                "exp_coef": np.nan,
                "exp_coef_lower_95": np.nan,
                "exp_coef_upper_95": np.nan,
                "se_coef": np.nan,
                "z": np.nan,
                "p": np.nan,
                "concordance": np.nan,
                "n_obs": len(sub_df),
                "n_events": int(sub_df[event_col].sum()) if len(sub_df) else 0,
                "converged": False,
                "note": "constant_column",
            })
            continue
        if len(sub_df) < 3:
            records.append({
                "feature": feat,
                "coef": np.nan,
                "exp_coef": np.nan,
                "exp_coef_lower_95": np.nan,
                "exp_coef_upper_95": np.nan,
                "se_coef": np.nan,
                "z": np.nan,
                "p": np.nan,
                "concordance": np.nan,
                "n_obs": len(sub_df),
                "n_events": int(sub_df[event_col].sum()) if len(sub_df) else 0,
                "converged": False,
                "note": "too_few_complete_cases",
            })
            continue

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                cph = CoxPHFitter()
                cph.fit(
                    sub_df,
                    duration_col=time_col,
                    event_col=event_col,
                    show_progress=False,
                )
            s = cph.summary
            row = {
                "feature": feat,
                "coef": float(s.loc[feat, "coef"]),
                "exp_coef": float(s.loc[feat, "exp(coef)"]),
                "exp_coef_lower_95": float(s.loc[feat, "exp(coef) lower 95%"]),
                "exp_coef_upper_95": float(s.loc[feat, "exp(coef) upper 95%"]),
                "se_coef": float(s.loc[feat, "se(coef)"]),
                "z": float(s.loc[feat, "z"]),
                "p": float(s.loc[feat, "p"]),
                "concordance": float(cph.concordance_index_),
                "n_obs": int(cph.event_observed.shape[0]),
                "n_events": int(cph.event_observed.sum()),
                "converged": True,
                "note": "",
            }
        except (ConvergenceError, Exception) as exc:
            row = {
                "feature": feat,
                "coef": np.nan, "exp_coef": np.nan,
                "exp_coef_lower_95": np.nan, "exp_coef_upper_95": np.nan,
                "se_coef": np.nan, "z": np.nan, "p": np.nan,
                "concordance": np.nan,
                "n_obs": len(sub_df),
                "n_events": int(sub_df[event_col].sum()) if len(sub_df) else 0,
                "converged": False,
                "note": str(exc)[:120],
            }
        records.append(row)

    results_df = pd.DataFrame(records).sort_values("p").reset_index(drop=True)
    results_df["significant"] = results_df["p"] < p_threshold
    results_df["rank"] = range(1, len(results_df) + 1)

    candidates = (
        results_df.loc[results_df["significant"] & results_df["converged"], "feature"]
        .tolist()
    )
    return results_df, candidates


# ---------------------------------------------------------------------------
# 方法 2 — VIF 共线性过滤
# ---------------------------------------------------------------------------

def _compute_vif(X: pd.DataFrame) -> pd.Series:
    """计算设计矩阵 X 中每列的 VIF 值。"""
    arr = X.values.astype(float)
    vif_vals = {}
    for i, col in enumerate(X.columns):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                v = _vif_raw(arr, i)
        except Exception:
            v = np.nan
        vif_vals[col] = v
    return pd.Series(vif_vals)


def method2_vif(
    df_cox: pd.DataFrame,
    candidate_features: List[str],
    m1_df: pd.DataFrame,
    vif_threshold: float = VIF_THRESHOLD,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    迭代计算 VIF，逐步剔除最高 VIF 超阈值的特征，按临床优先级保留更重要一方。

    返回
    ----
    (vif_log_df, retained_features)
    vif_log_df       : 全过程 VIF 记录（轮次 / 特征 / VIF / 动作）
    retained_features: 最终保留特征列表
    """
    if len(candidate_features) < 2:
        log_rows = [{"iteration": 0, "feature": f, "vif": np.nan, "action": "kept_single"}
                    for f in candidate_features]
        return _with_feature_groups(pd.DataFrame(log_rows)), list(candidate_features)

    remaining = list(candidate_features)
    log_rows = []
    iteration = 0

    while True:
        iteration += 1
        X = df_cox[remaining].dropna()
        if X.shape[0] < 3 or X.shape[1] < 2:
            break
        # 删除常数列（VIF 无法定义）
        non_const = [c for c in X.columns if X[c].nunique() > 1]
        const_cols = [c for c in X.columns if c not in non_const]
        for c in const_cols:
            log_rows.append({"iteration": iteration, "feature": c, "vif": np.nan,
                              "action": "removed_constant"})
            remaining.remove(c)
        if const_cols:
            continue

        vif_series = _compute_vif(X[remaining])
        max_vif = vif_series.max()
        for feat, v in vif_series.items():
            log_rows.append({
                "iteration": iteration,
                "feature": feat,
                "vif": round(float(v), 4) if not np.isnan(v) else np.nan,
                "action": "computed",
            })

        if np.isnan(max_vif) or max_vif <= vif_threshold:
            break  # 所有特征 VIF 均达标

        # 找出需要删除的特征；如果超阈值有多个，按优先级决定删除哪个。
        # 规则：优先删除临床变量，尽量保留 LA 参数进入后续 LASSO。
        over_threshold = vif_series[vif_series > vif_threshold].index.tolist()

        m1_lookup = m1_df.set_index("feature").to_dict("index") if not m1_df.empty else {}

        def _delete_priority(f: str) -> tuple:
            group = _feature_group(f)
            group_rank = {
                "clinical": 2,
                "morphology": 1,
                "kinematic": 0,
            }[group]
            info = m1_lookup.get(f, {})
            p_val = info.get("p", np.nan)
            p_rank = float(p_val) if pd.notna(p_val) else float("inf")
            missing_rate = float(df_cox[f].isna().mean()) if f in df_cox.columns else 1.0
            return (group_rank, p_rank, missing_rate, str(f))

        # 在超阈值集合中删除优先级最低者：
        # clinical 先删，morphology 次后删，kinematic 最后删；
        # 同组内删单变量 p 更差 / 缺失率更高 / 名字更靠后者。
        to_remove = max(over_threshold, key=lambda f: _delete_priority(f))
        remaining.remove(to_remove)
        log_rows.append({
            "iteration": iteration,
            "feature": to_remove,
            "feature_group": _feature_group(to_remove),
            "vif": round(float(vif_series[to_remove]), 4),
            "action": f"removed_vif>{vif_threshold:.0f}",
        })

        if len(remaining) < 2:
            break

    # 标记最终保留状态
    for row in log_rows:
        if row["action"] == "computed" and row["feature"] in remaining:
            # 最后一轮 computed 且仍在 remaining 中
            pass
    final_log = pd.DataFrame(log_rows)
    final_log = _with_feature_groups(final_log)
    final_log["retained"] = final_log["feature"].isin(remaining)
    return final_log, remaining


# ---------------------------------------------------------------------------
# 方法 3 — LASSO 惩罚 Cox（交叉验证选择 penalizer）
# ---------------------------------------------------------------------------

def _fit_lasso_cox(
    df_model: pd.DataFrame,
    candidate_features: List[str],
    time_col: str,
    event_col: str,
    penalizer: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    在全量数据上以指定 penalizer 拟合 LASSO Cox 模型。

    返回
    ----
    (coef_df, final_features)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=1.0)
        cph.fit(
            df_model,
            duration_col=time_col,
            event_col=event_col,
            show_progress=False,
        )
    coef_df = cph.summary.copy().reset_index()
    coef_df = coef_df.rename(columns={"covariate": "feature"})
    coef_df["penalizer"] = penalizer
    coef_df["nonzero"] = coef_df["coef"].abs() > 1e-4
    final_features = coef_df.loc[coef_df["nonzero"], "feature"].tolist()
    return coef_df, final_features


def method3_lasso(
    df_cox: pd.DataFrame,
    candidate_features: List[str],
    event_col: str,
    time_col: str,
    penalizers: List[float] = LASSO_PENALIZERS,
    cv_folds: int = CV_FOLDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    对 candidate_features 集合拟合 LASSO 惩罚 Cox 模型，
    使用 k 折交叉验证对数偏似然（log-partial-likelihood）选择最优 penalizer。

    回退策略（fallback）
    --------------------
    若 CV 最优 penalizer 的全量拟合结果中所有特征系数均为零，则从 penalizers
    列表由小至大依次尝试，直到找到保留 ≥1 个非零特征的最小 penalizer。
    回退时在 coef_df 中添加 ``fallback=True`` 列以供下游区分。

    返回
    ----
    (cv_df, coef_df, final_features)
    cv_df         : CV 结果（penalizer / mean_cv_ll / std_cv_ll）
    coef_df       : 全量数据最优模型的系数表
    final_features: LASSO 非零系数特征列表
    """
    if len(candidate_features) == 0:
        return pd.DataFrame(), pd.DataFrame(), []

    cols = candidate_features + [time_col, event_col]
    df_model = df_cox[cols].dropna()
    n = len(df_model)
    n_events = int(df_model[event_col].sum())

    if n < cv_folds or n_events < MIN_EVENTS:
        # 数据量太少，跳过 CV，直接用默认 penalizer 拟合
        best_pen = 0.1
        cv_rows = [{"penalizer": best_pen, "mean_cv_ll": np.nan, "std_cv_ll": np.nan,
                    "n_obs": n, "note": "skipped_cv_few_events"}]
        cv_df = pd.DataFrame(cv_rows)
    else:
        # k 折 CV
        indices = np.arange(n)
        rng = np.random.default_rng(42)
        rng.shuffle(indices)
        folds = np.array_split(indices, cv_folds)

        cv_rows = []
        for pen in penalizers:
            fold_lls = []
            for k, val_idx in enumerate(folds):
                train_idx = np.concatenate([folds[j] for j in range(cv_folds) if j != k])
                df_train = df_model.iloc[train_idx]
                df_val = df_model.iloc[val_idx]
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        cph = CoxPHFitter(penalizer=pen, l1_ratio=1.0)
                        cph.fit(
                            df_train,
                            duration_col=time_col,
                            event_col=event_col,
                            show_progress=False,
                        )
                        # 对数偏似然（验证集）
                        ll = cph.score(df_val, scoring_method="log_likelihood")
                        fold_lls.append(ll)
                except Exception:
                    fold_lls.append(np.nan)

            arr = np.array(fold_lls, dtype=float)
            cv_rows.append({
                "penalizer": pen,
                "mean_cv_ll": float(np.nanmean(arr)),
                "std_cv_ll": float(np.nanstd(arr)),
                "n_obs": n,
                "note": "",
            })

        cv_df = pd.DataFrame(cv_rows)
        best_pen = float(cv_df.loc[cv_df["mean_cv_ll"].idxmax(), "penalizer"])

    # 全量数据最优 penalizer 拟合
    fallback_used = False
    try:
        coef_df, final_features = _fit_lasso_cox(
            df_model, candidate_features, time_col, event_col, best_pen
        )
    except Exception as exc:
        coef_df = pd.DataFrame({"feature": candidate_features,
                                "coef": np.nan, "penalizer": best_pen,
                                "nonzero": False, "note": str(exc)[:120]})
        final_features = []

    # 回退策略：若最优 penalizer 使所有特征归零，则按从小到大顺序尝试其他 penalizer
    if len(final_features) == 0 and len(candidate_features) > 0:
        sorted_pens = sorted(penalizers)
        for fallback_pen in sorted_pens:
            if fallback_pen >= best_pen:
                continue  # 仅尝试比 best_pen 更小的值（更宽松正则化）
            try:
                coef_df_fb, final_fb = _fit_lasso_cox(
                    df_model, candidate_features, time_col, event_col, fallback_pen
                )
                if len(final_fb) > 0:
                    coef_df = coef_df_fb
                    final_features = final_fb
                    fallback_used = True
                    break
            except Exception:
                continue

    coef_df["fallback"] = fallback_used
    return cv_df, coef_df, final_features


def _augment_with_kinematic(
    df_cox: pd.DataFrame,
    base_features: List[str],
    kinematic_candidates: List[str],
    m1_df: pd.DataFrame,
    event_col: str,
    time_col: str,
    topk: int,
    p_threshold: float,
) -> Tuple[pd.DataFrame, List[str], bool, Optional[str]]:
    """
    Soft-preserve one kinematic feature after LASSO if none survived.
    """
    if not kinematic_candidates:
        return pd.DataFrame(), list(base_features), False, None

    m1_lookup = m1_df.set_index("feature").to_dict("index") if not m1_df.empty else {}
    ranked = sorted(
        [f for f in kinematic_candidates if f not in base_features],
        key=lambda f: _feature_rank_key(f, m1_lookup, df_cox),
    )[:topk]
    if not ranked:
        return pd.DataFrame(), list(base_features), False, None

    best_row = None
    best_feature = None
    rows = []

    for candidate in ranked:
        features = list(dict.fromkeys(base_features + [candidate]))
        cols = features + [time_col, event_col]
        df_model = df_cox[cols].dropna()
        if len(df_model) < 3 or int(df_model[event_col].sum()) < MIN_EVENTS:
            rows.append({
                "feature": candidate,
                "coef": np.nan,
                "p": np.nan,
                "penalizer": np.nan,
                "nonzero": False,
                "fallback": False,
                "augmented": True,
                "selected_via": "kinematic_augmentation",
                "note": "too_few_complete_cases",
            })
            continue

        fitted = None
        used_pen = 0.0
        for pen in (0.0, 0.05):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    cph = CoxPHFitter(penalizer=pen)
                    cph.fit(
                        df_model,
                        duration_col=time_col,
                        event_col=event_col,
                        show_progress=False,
                    )
                fitted = cph
                used_pen = pen
                break
            except Exception:
                continue

        if fitted is None or candidate not in fitted.summary.index:
            rows.append({
                "feature": candidate,
                "coef": np.nan,
                "p": np.nan,
                "penalizer": np.nan,
                "nonzero": False,
                "fallback": False,
                "augmented": True,
                "selected_via": "kinematic_augmentation",
                "note": "fit_failed",
            })
            continue

        p_val = float(fitted.summary.loc[candidate, "p"])
        coef = float(fitted.summary.loc[candidate, "coef"])
        row = {
            "feature": candidate,
            "coef": coef,
            "p": p_val,
            "penalizer": used_pen,
            "nonzero": abs(coef) > 1e-4,
            "fallback": used_pen > 0,
            "augmented": True,
            "selected_via": "kinematic_augmentation",
            "note": "",
        }
        rows.append(row)

        if p_val < p_threshold:
            if best_row is None or (p_val, -len(df_model), candidate) < (
                float(best_row["p"]),
                -int(best_row.get("n_obs", 0)),
                str(best_row["feature"]),
            ):
                row["n_obs"] = len(df_model)
                best_row = row
                best_feature = candidate

    if best_feature is not None and best_row is not None:
        final_features = list(dict.fromkeys(base_features + [best_feature]))
        return pd.DataFrame(rows), final_features, True, best_feature

    return pd.DataFrame(rows), list(base_features), False, None


# ---------------------------------------------------------------------------
# 汇总辅助
# ---------------------------------------------------------------------------

def _build_final_table(
    all_features: List[str],
    m1_df: pd.DataFrame,
    m2_retained: List[str],
    m3_coef_df: pd.DataFrame,
    m3_final: List[str],
    selected_via: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """将三步筛选结果合并为单张汇总表。"""
    rows = []
    m1_dict = m1_df.set_index("feature").to_dict("index") if not m1_df.empty else {}
    m3_dict = (
        m3_coef_df.set_index("feature")["coef"].to_dict()
        if "feature" in m3_coef_df.columns else {}
    )
    selected_via = selected_via or {}

    for feat in all_features:
        m1_info = m1_dict.get(feat, {})
        row = {
            "feature": feat,
            "feature_group": _feature_group(feat),
            # 方法 1
            "m1_coef":    m1_info.get("coef", np.nan),
            "m1_hr":      m1_info.get("exp_coef", np.nan),
            "m1_hr_lo95": m1_info.get("exp_coef_lower_95", np.nan),
            "m1_hr_hi95": m1_info.get("exp_coef_upper_95", np.nan),
            "m1_p":       m1_info.get("p", np.nan),
            "m1_pass":    bool(m1_info.get("m1_selected", False)),
            # 方法 2
            "m2_pass":    feat in m2_retained,
            # 方法 3
            "m3_coef":    m3_dict.get(feat, np.nan),
            "m3_pass":    feat in m3_final,
            # 最终：通过方法 3 LASSO 非零
            "final_selected": feat in m3_final,
            "selected_via": selected_via.get(feat, "lasso" if feat in m3_final else ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["final_selected", "m1_p"],
                                        ascending=[False, True]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 核心处理函数
# ---------------------------------------------------------------------------

def process_window_endpoint(
    df: pd.DataFrame,
    window: str,
    endpoint: str,
    output_dir: str,
    m1_p_thresholds: Dict[str, float],
    min_branch_candidates: Dict[str, int],
    branch_corr_threshold: float,
    kinematic_augment_topk: int,
    kinematic_augment_p: float,
    vif_threshold: float,
    cv_folds: int,
    dry_run: bool = False,
) -> Dict:
    """
    对单个时间窗 + 终点组合执行三步特征选择，写出所有报告。

    返回：包含关键统计的 dict（供 JSON 汇总使用）。
    """
    event_col, time_col, ep_desc = ENDPOINTS[endpoint]

    # 过滤：仅院外患者（os_event 非 NULL）且当前终点无缺失
    df_cox = df[df["os_event"].notna()].copy()
    df_cox = df_cox[df_cox[event_col].notna() & df_cox[time_col].notna()].copy()
    df_cox = df_cox.reset_index(drop=True)

    n_total = len(df_cox)
    n_events = int(df_cox[event_col].sum())
    label = f"{window}/{endpoint}"

    print(f"\n  [{label}] n={n_total}, 事件={n_events}")

    if n_events < MIN_EVENTS:
        print(f"  [{label}] 跳过：事件数 < {MIN_EVENTS}")
        return {"window": window, "endpoint": endpoint, "n_total": n_total,
                "n_events": n_events, "skipped": True, "reason": "too_few_events",
                "m1_candidates": 0, "m2_candidates": 0, "m3_final": 0,
                "final_features": []}

    # 确定可用特征列
    feature_cols = [
        c for c in df_cox.columns
        if c not in NON_FEATURE_COLS and df_cox[c].notna().any()
        and df_cox[c].nunique() > 1
    ]

    # ---- 方法 1 ----
    print(
        f"  [{label}] 方法 1：{len(feature_cols)} 个特征单变量筛选 "
        f"(clinical<{m1_p_thresholds['clinical']}, "
        f"morphology<{m1_p_thresholds['morphology']}, "
        f"kinematic<{m1_p_thresholds['kinematic']})"
    )
    m1_raw_df, _ = method1_univariate(df_cox, feature_cols, event_col, time_col)
    m1_df, m1_candidates = _select_branch_aware_m1_candidates(
        m1_raw_df,
        df_cox,
        p_thresholds=m1_p_thresholds,
        min_candidates=min_branch_candidates,
    )
    print(
        f"  [{label}] 方法 1 通过：{len(m1_candidates)} 个特征 "
        f"(kinematic={sum(_feature_group(f) == 'kinematic' for f in m1_candidates)})"
    )

    # ---- M1 -> M2 分支内高相关压缩 ----
    corr_log, branch_reduced = _compress_branch_candidates(
        df_cox,
        m1_candidates,
        m1_df,
        corr_threshold=branch_corr_threshold,
    )
    print(
        f"  [{label}] 分支内压缩后：{len(branch_reduced)} 个特征 "
        f"(kinematic={sum(_feature_group(f) == 'kinematic' for f in branch_reduced)})"
    )

    # ---- 方法 2 ----
    print(f"  [{label}] 方法 2：VIF 共线性过滤 (阈值={vif_threshold})")
    m2_vif_log, m2_retained = method2_vif(df_cox, branch_reduced, m1_df, vif_threshold=vif_threshold)
    m2_log = pd.concat([corr_log, m2_vif_log], ignore_index=True, sort=False)
    print(
        f"  [{label}] 方法 2 通过：{len(m2_retained)} 个特征 "
        f"(kinematic={sum(_feature_group(f) == 'kinematic' for f in m2_retained)})"
    )

    # ---- 方法 3 ----
    print(f"  [{label}] 方法 3：LASSO Cox CV ({cv_folds} 折, {len(LASSO_PENALIZERS)} 个 penalizer)")
    m3_cv, m3_coef, m3_final = method3_lasso(df_cox, m2_retained, event_col, time_col, cv_folds=cv_folds)
    m3_fallback = bool(m3_coef["fallback"].any()) if not m3_coef.empty and "fallback" in m3_coef.columns else False
    if m3_fallback:
        print(f"  [{label}] 方法 3 回退：CV最优penalizer全归零，已切换至最小有效penalizer")
    kinematic_augmented = False
    kinematic_augmented_feature = None
    selected_via = {feat: "lasso" for feat in m3_final}
    if not any(_feature_group(feat) == "kinematic" for feat in m3_final):
        kine_candidates = [feat for feat in m2_retained if _feature_group(feat) == "kinematic"]
        aug_df, augmented_final, kinematic_augmented, kinematic_augmented_feature = _augment_with_kinematic(
            df_cox=df_cox,
            base_features=m3_final,
            kinematic_candidates=kine_candidates,
            m1_df=m1_df,
            event_col=event_col,
            time_col=time_col,
            topk=kinematic_augment_topk,
            p_threshold=kinematic_augment_p,
        )
        if not aug_df.empty:
            m3_coef = pd.concat([m3_coef, aug_df], ignore_index=True, sort=False)
        if kinematic_augmented and kinematic_augmented_feature:
            m3_final = augmented_final
            selected_via[kinematic_augmented_feature] = "kinematic_augmentation"
            print(f"  [{label}] 软保留运动学特征：{kinematic_augmented_feature}")
    print(
        f"  [{label}] 方法 3 最终特征：{len(m3_final)} 个 "
        f"(kinematic={sum(_feature_group(f) == 'kinematic' for f in m3_final)}) → {m3_final}"
    )

    # ---- 汇总表 ----
    final_tbl = _build_final_table(
        feature_cols,
        m1_df,
        m2_retained,
        m3_coef,
        m3_final,
        selected_via=selected_via,
    )

    # ---- 写出报告 ----
    if not dry_run:
        win_dir = os.path.join(output_dir, window)
        os.makedirs(win_dir, exist_ok=True)

        m1_df.to_csv(
            os.path.join(win_dir, f"method1_univariate_{endpoint}.csv"), index=False)
        m1_cand = m1_df[m1_df["m1_selected"]].copy()
        m1_cand.to_csv(
            os.path.join(win_dir, f"method1_candidates_{endpoint}.csv"), index=False)

        m2_log.to_csv(
            os.path.join(win_dir, f"method2_vif_{endpoint}.csv"), index=False)
        pd.DataFrame({
            "feature": m2_retained,
            "feature_group": [_feature_group(feat) for feat in m2_retained],
        }).to_csv(
            os.path.join(win_dir, f"method2_candidates_{endpoint}.csv"), index=False)

        if not m3_cv.empty:
            m3_cv.to_csv(
                os.path.join(win_dir, f"method3_cv_{endpoint}.csv"), index=False)
        if not m3_coef.empty:
            m3_coef.to_csv(
                os.path.join(win_dir, f"method3_lasso_{endpoint}.csv"), index=False)

        final_tbl.to_csv(
            os.path.join(win_dir, f"final_features_{endpoint}.csv"), index=False)

        print(f"  [{label}] 报告已写出 → {win_dir}/")

    return {
        "window": window,
        "endpoint": endpoint,
        "endpoint_desc": ep_desc,
        "n_total": n_total,
        "n_events": n_events,
        "n_features_input": len(feature_cols),
        "skipped": False,
        "m1_candidates": len(m1_candidates),
        "m2_candidates": len(m2_retained),
        "m3_final": len(m3_final),
        "m1_kinematic": sum(_feature_group(f) == "kinematic" for f in m1_candidates),
        "m2_kinematic": sum(_feature_group(f) == "kinematic" for f in m2_retained),
        "m3_kinematic": sum(_feature_group(f) == "kinematic" for f in m3_final),
        "kinematic_augmented": kinematic_augmented,
        "kinematic_augmented_feature": kinematic_augmented_feature,
        "m3_fallback": m3_fallback,
        "m3_zero_reason": (
            "lasso_shrinkage" if len(m3_final) == 0 and len(m2_retained) > 0 else
            "no_m2_candidates" if len(m2_retained) == 0 else ""
        ),
        "final_features": m3_final,
        "best_lasso_penalizer": float(m3_coef["penalizer"].iloc[0]) if not m3_coef.empty else None,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input_dir", default="csv/survival",
                    help="生存终点 CSV 所在目录（默认：csv/survival）")
    ap.add_argument("--output_dir", default="csv/feature_selection",
                    help="特征选择报告输出目录（默认：csv/feature_selection）")
    ap.add_argument("--pattern", default="hfpef_cohort_win_*_survival.csv",
                    help="输入文件 glob 匹配模式")
    ap.add_argument("--endpoint", default=None,
                    choices=list(ENDPOINTS.keys()),
                    help="仅处理指定终点（默认：全部）")
    ap.add_argument("--window", default=None,
                    help="仅处理指定时间窗关键词（如 hadm、48h24h）")
    ap.add_argument("--p_threshold", type=float, default=None,
                    help="兼容旧版的统一 M1 p 值阈值；若提供，则同时覆盖 clinical / morphology / kinematic")
    ap.add_argument("--m1_p_clinical", type=float, default=0.10,
                    help="clinical 分支单变量 Cox 阈值（默认：0.10）")
    ap.add_argument("--m1_p_morphology", type=float, default=0.10,
                    help="morphology 分支单变量 Cox 阈值（默认：0.10）")
    ap.add_argument("--m1_p_kinematic", type=float, default=0.20,
                    help="kinematic 分支单变量 Cox 阈值（默认：0.20）")
    ap.add_argument("--min_morph_candidates", type=int, default=2,
                    help="morphology 分支的 M1 最低保留数（默认：2）")
    ap.add_argument("--min_kine_candidates", type=int, default=3,
                    help="kinematic 分支的 M1 最低保留数（默认：3）")
    ap.add_argument("--branch_corr_threshold", type=float, default=0.95,
                    help="分支内高相关压缩阈值 |r|（默认：0.95）")
    ap.add_argument("--kinematic_augment_topk", type=int, default=2,
                    help="LASSO 后运动学软保留候选数（默认：2）")
    ap.add_argument("--kinematic_augment_p", type=float, default=0.10,
                    help="运动学软保留联合 Cox 的 p 值阈值（默认：0.10）")
    ap.add_argument("--vif_threshold", type=float, default=VIF_THRESHOLD,
                    help=f"方法 2 VIF 阈值（默认：{VIF_THRESHOLD}）")
    ap.add_argument("--cv_folds", type=int, default=CV_FOLDS,
                    help=f"方法 3 交叉验证折数（默认：{CV_FOLDS}）")
    ap.add_argument("--dry_run", action="store_true",
                    help="仅打印摘要，不写出任何文件")
    args = ap.parse_args()

    args.input_dir  = _resolve_path(args.input_dir)
    args.output_dir = _resolve_path(args.output_dir)

    pattern = os.path.join(args.input_dir, args.pattern)
    input_files = sorted(glob.glob(pattern))
    if not input_files:
        print(f"未找到匹配文件：{pattern}")
        return

    # 过滤时间窗
    if args.window:
        input_files = [f for f in input_files if args.window in os.path.basename(f)]
    if not input_files:
        print(f"时间窗 '{args.window}' 下无匹配文件")
        return

    endpoints = [args.endpoint] if args.endpoint else list(ENDPOINTS.keys())
    if args.p_threshold is not None:
        args.m1_p_clinical = args.p_threshold
        args.m1_p_morphology = args.p_threshold
        args.m1_p_kinematic = args.p_threshold

    m1_p_thresholds = {
        "clinical": float(args.m1_p_clinical),
        "morphology": float(args.m1_p_morphology),
        "kinematic": float(args.m1_p_kinematic),
    }
    min_branch_candidates = {
        "morphology": int(args.min_morph_candidates),
        "kinematic": int(args.min_kine_candidates),
    }

    print(f"找到 {len(input_files)} 个文件，处理终点：{endpoints}")

    all_summaries = []

    for input_path in input_files:
        basename = os.path.basename(input_path)
        # 从文件名提取时间窗标识：hfpef_cohort_win_<window>_survival.csv
        window = basename.replace("hfpef_cohort_win_", "").replace("_survival.csv", "")
        print(f"\n{'='*60}")
        print(f"时间窗：{window}  ({basename})")
        print(f"{'='*60}")

        df = pd.read_csv(input_path, low_memory=False)
        print(f"  读入 {len(df)} 行 × {len(df.columns)} 列")

        for ep in endpoints:
            summary = process_window_endpoint(
                df, window, ep, args.output_dir,
                m1_p_thresholds=m1_p_thresholds,
                min_branch_candidates=min_branch_candidates,
                branch_corr_threshold=float(args.branch_corr_threshold),
                kinematic_augment_topk=int(args.kinematic_augment_topk),
                kinematic_augment_p=float(args.kinematic_augment_p),
                vif_threshold=float(args.vif_threshold),
                cv_folds=int(args.cv_folds),
                dry_run=args.dry_run,
            )
            all_summaries.append(summary)

    # 写出全局 JSON 汇总
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "selection_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n全局汇总 JSON → {summary_path}")

    # 打印汇总表
    print(f"\n{'='*60}")
    print("特征选择汇总")
    print(f"{'='*60}")
    hdr = f"{'窗口+终点':<20} {'输入':>6} {'M1':>5} {'M2':>5} {'M3':>5} 最终特征"
    print(hdr)
    print("-" * 80)
    for s in all_summaries:
        if s.get("skipped"):
            print(f"  {s['window']}/{s['endpoint']:<16} {'—':>6} {'—':>5} {'—':>5} {'—':>5} 跳过（事件数不足）")
        else:
            tag = f"{s['window']}/{s['endpoint']}"
            if s["final_features"]:
                feats = ", ".join(s["final_features"])
                fallback_note = " [回退]" if s.get("m3_fallback") else ""
                feat_str = f"{feats}{fallback_note}"
            else:
                reason = s.get("m3_zero_reason", "")
                reason_note = f"（{reason}）" if reason else "（无）"
                feat_str = reason_note
            print(f"  {tag:<20} {s['n_features_input']:>6} "
                  f"{s['m1_candidates']:>5} {s['m2_candidates']:>5} {s['m3_final']:>5} {feat_str}")

    print(f"\n完成。报告已保存至：{args.output_dir}/")


if __name__ == "__main__":
    main()
