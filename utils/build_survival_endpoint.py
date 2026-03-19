# -*- coding: utf-8 -*-
"""
生存终点构建脚本 — HFpEF 队列 Cox 比例风险分析

功能
----
读取经过插补/标准化的处理后 CSV（csv/processed/hfpef_cohort_win_*_processed.csv），
为每个时间窗生成生存分析就绪的 CSV，输出至 csv/survival/。

每个输出 CSV 包含：
  - 原始特征列（来自处理后文件）
  - 四组生存终点（每组含 time_ 和 event_ 两列）：
      time_30d  / event_30d   — 30 天截止点
      time_90d  / event_90d   — 90 天截止点
      time_1yr  / event_1yr   — 365 天截止点
      time_any  / event_any   — 不限时间（最长 365 天管理截尾）
  - censored_at_admin：1 = 该患者随访被截尾至 365 天行政上限

同时将院内死亡患者单独保存至 csv/survival/hfpef_cohort_win_*_inhosp.csv。

MIMIC-IV DOD（死亡日期）说明
--------------------------
死亡日期来源于医院记录与州政府记录，优先采用医院记录。
州政府记录通过姓名、出生日期及社会保障号的自定义规则匹配算法进行关联。
MIMIC-IV 中所有记录在最后一例患者出院两年后才完成采集，以降低上报延迟影响。
作为去标识化处理的一部分，出院超过 1 年的死亡日期被截尾（dod = NULL）。
因此每位患者的最长可观测随访期严格为末次出院后 1 年（365 天）。
  - 若患者在末次出院后 365 天内死亡且被记录捕获 → dod 填充去标识化后的死亡日期
  - 若患者在末次出院后存活至少 1 年 → dod = NULL（在本脚本中视为行政截尾）

患者分类（院外随访分析）
----------------------
  died_inhosp = 1  → 排除出院后分析；写入独立的院内死亡文件
  died_post_dc = 1 → 事件患者；time = days_survived_post_dc（精确死亡时间）
  其余患者        → 删失患者；time = 365（行政截尾上限）

用法
----
    python utils/build_survival_endpoint.py
    python utils/build_survival_endpoint.py --input_dir csv/processed --output_dir csv/survival
"""

import argparse
import os
import glob
import sys
import warnings
from typing import List

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
        "numpy 和 pandas 是必需依赖。\n"
        "请执行：pip install numpy pandas"
    ) from None

# 行政截尾上限（天）— MIMIC-IV 去标识化规则：出院超过 1 年的死亡日期被截尾
ADMIN_CENSOR_DAYS: int = 365

# 各终点时间窗（天）及对应列名后缀
HORIZONS: List[tuple] = [
    (30,   "30d"),
    (90,   "90d"),
    (365,  "1yr"),
    (None, "any"),   # None 表示不限时间，仍受 ADMIN_CENSOR_DAYS 约束
]

# 不得用于建模的列（标识符、时间戳、结局列）
_OUTCOME_COLS = {
    "hospital_expire_flag", "died_inhosp", "died_post_dc",
    "days_survived_post_dc", "died_30d", "died_90d", "died_1yr",
    "death_date",
}
_ID_COLS = {
    "subject_id", "hadm_id", "index_study_id",
    "index_study_datetime", "index_admittime", "index_dischtime",
    "a4c_dicom_filepath",
}


def _resolve_path(path: str) -> str:
    """
    解析相对路径。

    若路径相对于当前工作目录不存在，则尝试相对于仓库根目录解析。
    允许脚本从 utils/ 子目录（如 PyCharm 默认运行目录）直接启动，
    无需手动指定绝对路径。
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


def _safe_int(series: pd.Series) -> pd.Series:
    """将列转换为数值型 Int8（可为 NaN）。"""
    return pd.to_numeric(series, errors="coerce").astype("Int8")


def build_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    为单个时间窗的 DataFrame 构建所有生存终点列。

    参数
    ----
    df : 包含 died_inhosp、died_post_dc、days_survived_post_dc 等结局列的 DataFrame。
         注意：院内死亡行应在调用此函数前已被排除。

    返回
    ----
    带有新生存终点列的 DataFrame（原地修改副本）。
    """
    df = df.copy()

    # 将关键结局列转为数值
    died_post_dc = pd.to_numeric(df.get("died_post_dc", 0), errors="coerce").fillna(0)
    days_raw = pd.to_numeric(df.get("days_survived_post_dc", np.nan), errors="coerce")

    # 行政截尾标志：删失患者（days_raw 为 NaN）视为在 ADMIN_CENSOR_DAYS 天时截尾
    censored_mask = died_post_dc == 0  # 包括 days_raw 为 NaN 的删失患者

    # censored_at_admin：若为删失患者，则标记为行政截尾
    df["censored_at_admin"] = censored_mask.astype(int)

    # 为删失患者将 days 填充为行政截尾天数
    days_filled = days_raw.copy()
    days_filled[censored_mask] = ADMIN_CENSOR_DAYS

    for horizon, suffix in HORIZONS:
        if horizon is None:
            # 不限时间终点：仍受行政截尾约束
            time_col = days_filled.clip(upper=ADMIN_CENSOR_DAYS)
            event_col = (died_post_dc == 1).astype(int)
        else:
            # 有限时间终点：在 horizon 天内死亡才算事件
            event_col = ((died_post_dc == 1) & (days_raw <= horizon)).astype(int)
            # 时间 = min(实际死亡天数或行政截尾天数, horizon)
            time_col = days_filled.clip(upper=horizon)

        df[f"time_{suffix}"] = time_col.round(1)
        df[f"event_{suffix}"] = event_col

    return df


def process_file(
    input_path: str,
    output_postdc_path: str,
    output_inhosp_path: str,
    dry_run: bool = False,
) -> tuple:
    """
    处理单个时间窗的处理后 CSV，生成生存终点文件。

    返回
    ----
    (df_postdc, df_inhosp) — 院外随访 DataFrame 和院内死亡 DataFrame。
    """
    print(f"\n{'='*60}")
    print(f"处理文件：{os.path.basename(input_path)}")
    print(f"{'='*60}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"  读入 {len(df)} 行 × {len(df.columns)} 列")

    # ---- 分离院内死亡患者 ----------------------------------------
    died_inhosp = _safe_int(df.get("died_inhosp", pd.Series(0, index=df.index)))
    inhosp_mask = died_inhosp == 1
    df_inhosp = df[inhosp_mask].copy().reset_index(drop=True)
    df_postdc = df[~inhosp_mask].copy().reset_index(drop=True)

    print(f"  院内死亡（排除）：{inhosp_mask.sum()} 例 → 写入院内死亡文件")
    print(f"  院外随访分析：{len(df_postdc)} 例")

    # ---- 构建生存终点列 ------------------------------------------
    df_postdc = build_endpoints(df_postdc)

    # ---- 打印终点摘要 -------------------------------------------
    print(f"\n  {'终点':<12} {'事件数':>8} {'事件率':>8} {'中位随访(天)':>14}")
    print(f"  {'-'*46}")
    for _, suffix in HORIZONS:
        ecol = f"event_{suffix}"
        tcol = f"time_{suffix}"
        if ecol in df_postdc.columns:
            n_events = int(df_postdc[ecol].sum())
            pct = n_events / len(df_postdc) * 100 if len(df_postdc) > 0 else 0
            med_t = df_postdc[tcol].median()
            print(f"  {suffix:<12} {n_events:>8} {pct:>7.1f}% {med_t:>14.1f}")

    admin_censored = int(df_postdc.get("censored_at_admin", pd.Series(0)).sum())
    print(f"\n  行政截尾（≥365 天）患者：{admin_censored} 例")

    if not dry_run:
        os.makedirs(os.path.dirname(output_postdc_path), exist_ok=True)
        df_postdc.to_csv(output_postdc_path, index=False)
        print(f"\n  院外随访文件 → {output_postdc_path}")
        if len(df_inhosp) > 0:
            df_inhosp.to_csv(output_inhosp_path, index=False)
            print(f"  院内死亡文件 → {output_inhosp_path}")
    else:
        print("\n  [dry-run] 文件未写出")

    return df_postdc, df_inhosp


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input_dir", default="csv/processed",
        help="包含处理后 CSV 的目录（默认：csv/processed）",
    )
    ap.add_argument(
        "--output_dir", default="csv/survival",
        help="输出生存终点 CSV 的目录（默认：csv/survival）",
    )
    ap.add_argument(
        "--pattern", default="hfpef_cohort_win_*_processed.csv",
        help="输入文件的 glob 匹配模式",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="仅打印摘要，不写出文件",
    )
    args = ap.parse_args()

    args.input_dir  = _resolve_path(args.input_dir)
    args.output_dir = _resolve_path(args.output_dir)

    pattern = os.path.join(args.input_dir, args.pattern)
    input_files = sorted(glob.glob(pattern))
    if not input_files:
        print(f"未找到匹配文件：{pattern}")
        return

    print(f"找到 {len(input_files)} 个文件：")
    for f in input_files:
        print(f"  {f}")

    for input_path in input_files:
        basename = os.path.basename(input_path)
        # hfpef_cohort_win_hadm_processed.csv → hfpef_cohort_win_hadm
        stem = basename.replace("_processed", "").replace(".csv", "")
        output_postdc_path = os.path.join(args.output_dir, f"{stem}_survival.csv")
        output_inhosp_path = os.path.join(args.output_dir, f"{stem}_inhosp.csv")
        process_file(input_path, output_postdc_path, output_inhosp_path, args.dry_run)

    print(f"\n完成。生存终点文件已保存至：{args.output_dir}/")


if __name__ == "__main__":
    main()
