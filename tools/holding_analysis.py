#!/usr/bin/env python3
"""持有期静态分析（discover_24 复现工具，A 类信号）。

子命令:
    curve  持有期衰减曲线：horizon × (n/E/中位/胜率/盈亏比/平均MaxDD)
    mae    止损扫描（附录六"止损扫描(静态近似)"口径）
    mfe    止盈扫描（附录六"止盈扫描(静态近似)"口径）

用法:
    python3 tools/holding_analysis.py curve output/an_spike.csv
    python3 tools/holding_analysis.py curve output/an_base_break_days.csv \
        --query "group=='C' and volume_type!='缩量'"
    python3 tools/holding_analysis.py mae output/an_spike.csv \
        --horizon 90 --levels -5,-8,-10,-15
    python3 tools/holding_analysis.py mfe output/an_spike.csv \
        --horizon 90 --levels 10,20,30

口径（附录）:
    E/中位/胜率/盈亏比/平均MaxDD    附录一
    maxup/maxdn（MFE/MAE）          附录一（含基准日、high/low 口径）
    止损/止盈扫描静态近似            附录六
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd


def load(fp: str, query: str | None) -> pd.DataFrame:
    df = pd.read_csv(fp, encoding="utf-8-sig", dtype={"code": str})
    if query:
        df = df.query(query)
    return df


def horizons_of(df: pd.DataFrame) -> list[int]:
    """自动检测 ret_Nd 列，返回 N 升序。"""
    hs = [int(m.group(1)) for c in df.columns
          if (m := re.fullmatch(r"ret_(\d+)d", c))]
    return sorted(hs)


def curve(df: pd.DataFrame) -> None:
    print(f"{'持有期':<8} {'n':>7} {'E':>8} {'中位':>8} {'胜率':>7} "
          f"{'盈亏比':>7} {'平均MaxDD':>10}")
    print("-" * 60)
    for n in horizons_of(df):
        ret, mdd = df[f"ret_{n}d"], df[f"maxdn_{n}d"]
        ok = ret.notna()
        r, m = ret[ok], mdd[ok]
        win = r[r > 0]
        lose = r[r <= 0]
        plr = (win.mean() / abs(lose.mean())
               if len(win) and len(lose) and lose.mean() != 0 else np.nan)
        print(f"{n:>4}d  {len(r):>7} {r.mean():>+7.2f}% {r.median():>+7.2f}% "
              f"{(r > 0).mean() * 100:>6.1f}% {plr:>7.2f} {m.mean():>+9.2f}%")


def mae(df: pd.DataFrame, h: int, levels: list[float]) -> None:
    """附录六：触发 = maxdn_Nd <= s；触发记 s，未触发记 ret_Nd。"""
    ret, mdd = df[f"ret_{h}d"], df[f"maxdn_{h}d"]
    ok = ret.notna() & mdd.notna()
    r, m = ret[ok], mdd[ok]
    e_raw = r.mean()
    print(f"窗口 {h}d | n={len(r)} | 原始 E={e_raw:+.2f}%")
    print(f"{'止损档':>8} {'触发率':>8} {'误伤率':>8} {'误伤代价':>10} "
          f"{'E_stop':>9} {'ΔE':>8}")
    print("-" * 58)
    for s in levels:
        trig = m <= s
        p = trig.mean() * 100
        hurt = r[trig]
        hurt_win = hurt[hurt > 0]
        hurt_cost = hurt_win.mean() if len(hurt_win) else np.nan
        e_stop = np.where(trig, s, r).mean()
        print(f"{s:>7.0f}% {p:>7.1f}% "
              f"{(len(hurt_win) / trig.sum() * 100 if trig.sum() else 0):>7.1f}% "
              f"{hurt_cost:>+9.2f}% {e_stop:>+8.2f}% {e_stop - e_raw:>+7.2f}")


def mfe(df: pd.DataFrame, h: int, levels: list[float]) -> None:
    """附录六：触发 = maxup_Nd >= t；触发记 t，未触发记 ret_Nd。"""
    ret, mup = df[f"ret_{h}d"], df[f"maxup_{h}d"]
    ok = ret.notna() & mup.notna()
    r, u = ret[ok], mup[ok]
    e_raw = r.mean()
    print(f"窗口 {h}d | n={len(r)} | 原始 E={e_raw:+.2f}%")
    print(f"{'止盈档':>8} {'触发率':>8} {'右尾代价':>10} {'E_tp':>9} {'ΔE':>8}")
    print("-" * 50)
    for t in levels:
        trig = u >= t
        p = trig.mean() * 100
        tail = r[trig].mean() - t if trig.sum() else np.nan
        e_tp = np.where(trig, t, r).mean()
        print(f"{t:>+7.0f}% {p:>7.1f}% {tail:>+9.2f}% "
              f"{e_tp:>+8.2f}% {e_tp - e_raw:>+7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["curve", "mae", "mfe"])
    ap.add_argument("input", help="an 明细 CSV 路径")
    ap.add_argument("--query", help="pandas query 过滤（如 group=='C'）")
    ap.add_argument("--horizon", type=int, default=60, help="mae/mfe 窗口日数")
    ap.add_argument("--levels", default=None,
                    help="档位逗号分隔（mae 默认 -5,-8,-10,-15；mfe 默认 10,20,30）")
    a = ap.parse_args()

    df = load(a.input, a.query)
    if not len(df):
        sys.exit("过滤后无样本")
    tag = f" | query: {a.query}" if a.query else ""
    print(f"== {a.input} | n={len(df)}{tag}\n")

    if a.cmd == "curve":
        curve(df)
    elif a.cmd == "mae":
        levels = [float(x) for x in (a.levels or "-5,-8,-10,-15").split(",")]
        mae(df, a.horizon, levels)
    else:
        levels = [float(x) for x in (a.levels or "10,20,30").split(",")]
        mfe(df, a.horizon, levels)


if __name__ == "__main__":
    main()
