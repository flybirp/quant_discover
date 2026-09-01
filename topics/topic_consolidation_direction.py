#!/usr/bin/env python3
"""专题：超长期横盘震荡后的突破方向特征预测。

join output/catch_consol_scanner.csv（21 特征列）与
output/an_long_consolidation.csv（maxup/maxdn_120d），按旧口径分类：

  up:    maxup_120d >= +30%
  down:  maxdn_120d <= -30%
  range: 其余

输出：方向分布、特征分组中位数对比（区分度排序）、
PE/PB 分区间的突破概率表。

用法: python3 topics/topic_consolidation_direction.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FEATURES = [
    "vol_shrink_ratio", "vol_trend", "cum_turnover",
    "atr_pct", "atr_converge", "bb_width_median", "annual_vol",
    "ma_tangle", "ma_bull_ratio",
    "zx_cross_count", "zx_tangle",
    "dif_above_zero", "dif_abs_pct",
    "pe_median", "pe_range", "pb_median",
    "mv_median", "mv_change",
    "range_pct", "trend_pct", "duration_days",
]


def main():
    zones = pd.read_csv(ROOT / "output/catch_consol_scanner.csv",
                        encoding="utf-8-sig", dtype={"code": str})
    an = pd.read_csv(ROOT / "output/an_long_consolidation.csv",
                     encoding="utf-8-sig", dtype={"code": str})
    an["end_date"] = an["base_date"]  # 契约: base_date = 区间末日
    keys = ["code", "end_date"]
    df = zones.merge(an[keys + ["maxup_120d", "maxdn_120d"]], on=keys, how="inner")
    df = df.dropna(subset=["maxup_120d", "maxdn_120d"])
    print(f"可判定样本: {len(df)} / {len(zones)}")

    # 方向分类
    up = df["maxup_120d"] >= 30
    down = df["maxdn_120d"] <= -30
    df["direction"] = np.where(up, "up", np.where(down, "down", "range"))

    vc = df["direction"].value_counts()
    print("\n--- 突破方向分布（120 交易日窗口）---")
    for d, label in [("up", "向上突破(≥+30%)"), ("down", "向下突破(≤-30%)"), ("range", "继续震荡")]:
        n = int(vc.get(d, 0))
        print(f"  {label}: {n} ({n/len(df)*100:.1f}%)")

    # 特征区分度
    print("\n--- 特征中位数对比（按 |up-down|/整体std 降序）---")
    rows = []
    for c in FEATURES:
        if c not in df.columns:
            continue
        u = df.loc[df["direction"] == "up", c].median()
        d_ = df.loc[df["direction"] == "down", c].median()
        r = df.loc[df["direction"] == "range", c].median()
        std = df[c].std()
        diff = abs(u - d_) / std if std and std > 0 else 0
        rows.append((c, u, d_, r, diff))
    rows.sort(key=lambda x: x[4], reverse=True)
    print(f"{'特征':<20} {'向上':>10} {'向下':>10} {'震荡':>10} {'区分度':>7}")
    for c, u, d_, r, diff in rows:
        stars = "★" * min(5, int(diff * 2)) if diff > 0.1 else ""
        print(f"  {c:<18} {u:>10.2f} {d_:>10.2f} {r:>10.2f} {stars}")

    # PE 分区间突破概率
    print("\n--- PE 区间 × 突破方向概率 ---")
    pe = df[df["pe_median"] > 0].copy()
    if len(pe):
        pe["pe_bin"] = pd.cut(pe["pe_median"], [0, 15, 25, 40, 60, 100, 10000],
                              labels=["0-15", "15-25", "25-40", "40-60", "60-100", "100+"])
        g = pe.groupby("pe_bin", observed=True)["direction"]
        t = pd.DataFrame({
            "样本": g.size(),
            "向上%": (g.apply(lambda s: (s == "up").mean() * 100)).round(1),
            "向下%": (g.apply(lambda s: (s == "down").mean() * 100)).round(1),
        })
        print(t.to_string())

    # PB 分区间
    print("\n--- PB 区间 × 突破方向概率 ---")
    pb = df[df["pb_median"] > 0].copy()
    if len(pb):
        pb["pb_bin"] = pd.cut(pb["pb_median"], [0, 1, 2, 3, 5, 10, 100],
                              labels=["0-1", "1-2", "2-3", "3-5", "5-10", "10+"])
        g = pb.groupby("pb_bin", observed=True)["direction"]
        t = pd.DataFrame({
            "样本": g.size(),
            "向上%": (g.apply(lambda s: (s == "up").mean() * 100)).round(1),
            "向下%": (g.apply(lambda s: (s == "down").mean() * 100)).round(1),
        })
        print(t.to_string())


if __name__ == "__main__":
    main()
