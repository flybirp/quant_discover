#!/usr/bin/env python3
"""专题：日历效应（周内 / 月度 / 春节前后）。

数据：中证1000 指数（市场层面）+ 全市场个股截面均值（每 5 日采样）。
输出：三张效应表（次 5 日收益的均值/胜率/中位）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.market import load_index
from core.engine import DEFAULT_DATA_DIR, load_data

# 2014-2026 春节日期（正月初一）
CNY = ["2014-01-31", "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16",
       "2019-02-05", "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22",
       "2024-02-10", "2025-01-29", "2026-02-17"]


def main():
    idx = load_index("zz1000")
    idx_df = pd.DataFrame({"close": idx})
    idx_df["fwd5"] = idx_df["close"].shift(-5) / idx_df["close"] - 1
    idx_df["weekday"] = idx_df.index.dayofweek
    idx_df["month"] = idx_df.index.month

    print("=== 一、周内效应（中证1000，次 5 日收益）===")
    wd_names = ["周一", "周二", "周三", "周四", "周五"]
    for wd, name in enumerate(wd_names):
        v = idx_df[idx_df.weekday == wd]["fwd5"].dropna() * 100
        print(f"  {name}: n={len(v)} E={v.mean():+.3f}% med={v.median():+.3f}% win={(v > 0).mean() * 100:.1f}%")

    print("\n=== 二、月度效应（中证1000，次月收益，每月首交易日采样）===")
    month_first = idx_df.index.to_series().groupby(idx_df.index.to_period("M")).min()
    for m in range(1, 13):
        rets = []
        for d in month_first.values:
            d = pd.Timestamp(d)
            if d.month != m:
                continue
            loc = idx_df.index.get_loc(d)
            if loc + 21 < len(idx_df):
                rets.append((idx_df["close"].iloc[loc + 21] / idx_df["close"].iloc[loc] - 1) * 100)
        v = pd.Series(rets)
        if len(v):
            print(f"  {m:>2} 月: n={len(v)} E={v.mean():+.2f}% med={v.median():+.2f}% win={(v > 0).mean() * 100:.0f}%")

    print("\n=== 三、春节效应（中证1000，节前 5 日 / 节后 5 日）===")
    for cny in CNY:
        cny_d = pd.Timestamp(cny)
        pre = idx_df[(idx_df.index < cny_d)].tail(5)
        post = idx_df[(idx_df.index >= cny_d)].head(5)
        if len(pre) < 3 or len(post) < 3:
            continue
        pre_ret = (pre["close"].iloc[-1] / pre["close"].iloc[0] - 1) * 100 if len(pre) else np.nan
        post_ret = (post["close"].iloc[-1] / post["close"].iloc[0] - 1) * 100 if len(post) else np.nan
        print(f"  {cny}: 节前5日 {pre_ret:+.2f}% | 节后5日 {post_ret:+.2f}%")
    pre_all, post_all = [], []
    for cny in CNY:
        cny_d = pd.Timestamp(cny)
        pre = idx_df[(idx_df.index < cny_d)].tail(5)
        post = idx_df[(idx_df.index >= cny_d)].head(5)
        if len(pre) >= 3 and len(post) >= 3:
            pre_all.append((pre["close"].iloc[-1] / pre["close"].iloc[0] - 1) * 100)
            post_all.append((post["close"].iloc[-1] / post["close"].iloc[0] - 1) * 100)
    print(f"  均值: 节前 {np.mean(pre_all):+.2f}% | 节后 {np.mean(post_all):+.2f}% "
          f"(n={len(pre_all)} 年, 节后为正年份 {(np.array(post_all) > 0).mean() * 100:.0f}%)")


if __name__ == "__main__":
    main()
