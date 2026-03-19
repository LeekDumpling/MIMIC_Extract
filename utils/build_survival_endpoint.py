# -*- coding: utf-8 -*-
"""
生存终点构建脚本 — HFpEF 队列 Cox 比例风险分析

功能
----
读取经过插补/标准化的处理后 CSV（csv/processed/hfpef_cohort_win_*_processed.csv），
为每个时间窗生成生存分析就绪的 CSV，输出至 csv/survival/。

主要生存终点列（来自原始表格，已在 SQL 中计算完毕）
  os_event  : Cox 事件指标
                1  = 出院后死亡（事件）
                0  = 删失（存活且随访期结束）
                NULL = 院内死亡（不参与出院后分析，过滤条件：WHERE died_inhosp=0）
  os_days   : Cox 时间变量（仅院外患者非 NULL）
                事件患者 : dod − index_dischtime（天）
                删失患者 : censor_date − index_dischtime（天）
                           其中 censor_date = last_dischtime + 1 年
                           last_dischtime = 患者在 MIMIC-IV 中的末次住院出院时间
                院内死亡 : NULL（排除）

MIMIC-IV DOD 及删失规则说明
--------------------------
- 删失时间从末次住院出院时间（last_dischtime）而非指数住院出院时间（index_dischtime）计算。
- 若患者在指数住院后还有后续住院，其可观测随访窗口将延伸至 last_dischtime + 1 年。
- 因此 os_days 可超过 365（多次住院患者的有效随访期更长）。
- 仅当末次出院后存活满 1 年时，dod 才被截尾（os_event=0）。

衍生时间窗终点（由本脚本计算）
  对于 H ∈ {30, 90, 365}（天），从 os_event / os_days 衍生：
    time_<H>  = min(os_days, H)
    event_<H> = 1  如果 os_event=1 且 os_days ≤ H
              = 0  否则（含 os_event=0 的删失患者）
    对于院内死亡患者（os_event=NULL）：time_<H> 和 event_<H> 均为 NULL，与院外终点隔离。

输出文件
  csv/survival/hfpef_cohort_win_*_survival.csv  — 所有患者（含院内死亡的 NULL 行）
  csv/survival/hfpef_cohort_win_*_inhosp.csv    — 仅院内死亡患者（独立文件，供院内死亡分析）

用法（Windows 单行命令）
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

# 有限时间截止点（天）及对应列名后缀
# None 表示"不限时间"（直接使用 os_days，不截断）
HORIZONS: List[tuple] = [
    (30,   "30d"),
    (90,   "90d"),
    (365,  "1yr"),
    (None, "any"),
]


def _resolve_path(path: str) -> str:
    """
    解析相对路径。

    若路径相对于当前工作目录不存在，则尝试相对于仓库根目录解析。
    允许脚本从 utils/ 子目录（如 PyCharm 默认运行目录）直接启动。
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


def build_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    为 DataFrame 添加衍生生存终点列。

    直接使用原始 os_event / os_days 列（已在 SQL 中正确计算），
    在此基础上衍生各时间窗终点。

    院内死亡患者（os_event=NaN、os_days=NaN）的所有衍生终点均保持 NaN，
    以便与出院后分析明确隔离。

    参数
    ----
    df : 包含 os_event、os_days 列的 DataFrame。

    返回
    ----
    带有新衍生终点列的 DataFrame 副本。
    """
    df = df.copy()

    os_event = pd.to_numeric(df["os_event"], errors="coerce")  # 1/0/NaN
    os_days  = pd.to_numeric(df["os_days"],  errors="coerce")  # float/NaN

    # 院内死亡掩码：os_event 为 NaN 即院内死亡
    inhosp_mask = os_event.isna()

    for horizon, suffix in HORIZONS:
        if horizon is None:
            # 不限时间：直接使用 os_event / os_days
            event_col = os_event.copy()
            time_col  = os_days.copy()
        else:
            # 时间窗终点：在 horizon 天内死亡才算事件
            # 对院内死亡患者，计算结果保持 NaN
            event_col = pd.Series(np.nan, index=df.index, dtype=float)
            time_col  = pd.Series(np.nan, index=df.index, dtype=float)

            alive_mask = ~inhosp_mask
            event_col[alive_mask] = (
                (os_event[alive_mask] == 1) & (os_days[alive_mask] <= horizon)
            ).astype(float)
            time_col[alive_mask] = os_days[alive_mask].clip(upper=horizon)

        df[f"event_{suffix}"] = event_col
        df[f"time_{suffix}"]  = time_col.round(1)

    return df


def process_file(
    input_path: str,
    output_all_path: str,
    output_inhosp_path: str,
    output_dir: str,
    dry_run: bool = False,
) -> tuple:
    """
    处理单个时间窗的处理后 CSV，生成生存终点文件。

    返回
    ----
    (df_all, df_inhosp) — 全量 DataFrame（含院内死亡 NULL 行）和仅院内死亡 DataFrame。
    """
    print(f"\n{'='*60}")
    print(f"处理文件：{os.path.basename(input_path)}")
    print(f"{'='*60}")

    df = pd.read_csv(input_path, low_memory=False)
    print(f"  读入 {len(df)} 行 × {len(df.columns)} 列")

    # ---- 统计院内死亡（os_event 为 NaN）-----------------------------
    os_event = pd.to_numeric(df.get("os_event"), errors="coerce")
    inhosp_mask = os_event.isna()
    df_inhosp = df[inhosp_mask].copy().reset_index(drop=True)
    n_inhosp = int(inhosp_mask.sum())
    n_postdc = int((~inhosp_mask).sum())

    print(f"  院内死亡（os_event=NULL）：{n_inhosp} 例")
    print(f"  院外患者（参与出院后分析）：{n_postdc} 例")
    print(f"    其中事件（os_event=1）：{int((os_event == 1).sum())} 例")
    print(f"    其中删失（os_event=0）：{int((os_event == 0).sum())} 例")

    # ---- 构建衍生终点列 ------------------------------------------
    df_all = build_endpoints(df)

    # ---- 打印衍生终点摘要（仅院外患者）-----------------------------
    df_postdc = df_all[~inhosp_mask]
    print(f"\n  {'终点':<12} {'事件数':>8} {'事件率':>8} {'中位时间(天)':>14}")
    print(f"  {'-'*46}")
    for _, suffix in HORIZONS:
        ecol = f"event_{suffix}"
        tcol = f"time_{suffix}"
        if ecol in df_postdc.columns:
            e_vals = pd.to_numeric(df_postdc[ecol], errors="coerce").dropna()
            t_vals = pd.to_numeric(df_postdc[tcol], errors="coerce").dropna()
            n_events = int(e_vals.sum())
            pct = n_events / len(e_vals) * 100 if len(e_vals) > 0 else 0
            med_t = t_vals.median()
            print(f"  {suffix:<12} {n_events:>8} {pct:>7.1f}% {med_t:>14.1f}")

    os_days_postdc = pd.to_numeric(df_postdc["os_days"], errors="coerce")
    print(f"\n  os_days 范围（院外患者）：{os_days_postdc.min():.0f} – {os_days_postdc.max():.0f} 天")
    print(f"  os_days > 365 的患者数：{(os_days_postdc > 365).sum()} 例（末次出院晚于指数出院）")

    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        df_all.to_csv(output_all_path, index=False)
        print(f"\n  全量生存文件（含 NULL 行）→ {output_all_path}")
        if n_inhosp > 0:
            df_inhosp.to_csv(output_inhosp_path, index=False)
            print(f"  院内死亡独立文件 → {output_inhosp_path}")
    else:
        print("\n  [dry-run] 文件未写出")

    return df_all, df_inhosp


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
        stem = basename.replace("_processed", "").replace(".csv", "")
        output_all_path    = os.path.join(args.output_dir, f"{stem}_survival.csv")
        output_inhosp_path = os.path.join(args.output_dir, f"{stem}_inhosp.csv")
        process_file(input_path, output_all_path, output_inhosp_path, args.output_dir, args.dry_run)

    print(f"\n完成。生存终点文件已保存至：{args.output_dir}/")


if __name__ == "__main__":
    main()
