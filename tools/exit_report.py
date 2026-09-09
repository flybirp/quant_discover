#!/usr/bin/env python3
"""exit 引擎结果报告：分年 / 波动率分层 / 规则曲面总览（R1 等）。

输入 run.py exit 产出的明细 CSV（含 ruleset/base_date/vol60/post_*）。

用法:
    python3 tools/exit_report.py output/exit_r1_ma_break.csv --ruleset ma20_x2
    python3 tools/exit_report.py output/exit_r1_ma_break.csv --surface
"""
import argparse
import sys

import numpy as np
import pandas as pd


def per_year(df: pd.DataFrame) -> None:
    df = df.copy()
    df["year"] = df["base_date"].astype(str).str[:4]
    print(f"{'年':<6} {'n':>7} {'E净':>7} {'胜率':>6} {'卖飞+5%':>8} "
          f"{'卖对':>6} {'再入场':>7} {'平均天数':>8}")
    print("-" * 62)
    for y, s in df.groupby("year"):
        pm = s["post_max60"].dropna()
        re = s["reenter"].dropna()
        print(f"{y:<6} {len(s):>7} {s['ret_net'].mean():>+6.2f} "
              f"{(s['ret_net'] > 0).mean() * 100:>5.1f}% "
              f"{(pm >= 5).mean() * 100:>7.1f}% "
              f"{(s['post_below60'] == True).mean() * 100:>5.1f}% "
              f"{(re == True).mean() * 100 if len(re) else float('nan'):>6.1f}% "
              f"{s['days'].mean():>7.1f}")


def by_vol(df: pd.DataFrame) -> None:
    df = df.copy()
    df = df[df["vol60"].notna()]
    if not len(df):
        print("（无 vol60 列）")
        return
    df["波动层"] = pd.qcut(df["vol60"], 3, labels=["低波动", "中波动", "高波动"])
    print(f"{'波动层':<6} {'vol60中位':>9} {'n':>7} {'E净':>7} {'胜率':>6} "
          f"{'卖飞+5%':>8} {'再入场':>7}")
    print("-" * 58)
    for g, s in df.groupby("波动层", observed=True):
        pm = s["post_max60"].dropna()
        re = s["reenter"].dropna()
        print(f"{g:<6} {s['vol60'].median():>8.2f} {len(s):>7} "
              f"{s['ret_net'].mean():>+6.2f} "
              f"{(s['ret_net'] > 0).mean() * 100:>5.1f}% "
              f"{(pm >= 5).mean() * 100:>7.1f}% "
              f"{(re == True).mean() * 100 if len(re) else float('nan'):>6.1f}%")


def surface(df: pd.DataFrame) -> None:
    print(f"{'规则集':<10} {'n':>7} {'E净':>7} {'胜率':>6} {'平均天数':>8} "
          f"{'卖飞+5%':>8} {'卖对':>6} {'再入场':>7}")
    print("-" * 66)
    for rs, s in df.groupby("ruleset"):
        pm = s["post_max60"].dropna()
        re = s["reenter"].dropna()
        print(f"{rs:<10} {len(s):>7} {s['ret_net'].mean():>+6.2f} "
              f"{(s['ret_net'] > 0).mean() * 100:>5.1f}% "
              f"{s['days'].mean():>7.1f} "
              f"{(pm >= 5).mean() * 100:>7.1f}% "
              f"{(s['post_below60'] == True).mean() * 100:>5.1f}% "
              f"{(re == True).mean() * 100 if len(re) else float('nan'):>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--ruleset", help="分年/波动率分层的规则集名")
    ap.add_argument("--surface", action="store_true", help="全规则集曲面总览")
    a = ap.parse_args()

    df = pd.read_csv(a.input, encoding="utf-8-sig", dtype={"code": str},
                     low_memory=False)
    if a.surface:
        surface(df)
        return
    if not a.ruleset:
        sys.exit("指定 --ruleset 或 --surface")
    sub = df[df["ruleset"] == a.ruleset]
    if not len(sub):
        sys.exit(f"无规则集 {a.ruleset}（可选: {df['ruleset'].unique()}）")
    print(f"== {a.input} | ruleset={a.ruleset} | n={len(sub)}\n")
    print("-- 分年")
    per_year(sub)
    print("\n-- 波动率分层（vol60 三分位，附录六）")
    by_vol(sub)


if __name__ == "__main__":
    main()
