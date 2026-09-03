#!/usr/bin/env python3
"""周期级绝望期研究 —— discover_23 主口径的正式复现工具。

定义:
    绝望日   = 当日下跌家数占比 >= 80%（有效截面 >= 1500 只）
    绝望期初期 T0 = 指数(zz1000) 20 日累计 <= -10% 且过去 20 日内
               绝望日 >= 2 个（周线级大跌 + 多日广度极端）
    低点 L   = T0 起至企稳确认的指数最低收盘日
    企稳 T1  = 自 L 反弹 >= 7% 且 L 后 >= 10 个交易日（排除单日反抽）
    主跌段   = 个股自 T0-20 交易日（下跌起点）至 L 的累计收益——
               分组基准，刨除 L->T1 企稳反弹段（弹性股会冒充抗跌者）
    买入     = T1 收盘，统计 20/60 日收益/超额/MaxDD

用法:
    python3 tools/despair_cycle.py
    python3 tools/despair_cycle.py --drop20 -8 --reb 5   # 放宽阈值
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.engine import load_data    # noqa: E402
from core.market import load_index   # noqa: E402

DATA_DIR = Path("/Users/flybirp/Documents/mainland_data_2014")


def pass1_breadth() -> dict[str, list[int]]:
    breadth: dict[str, list[int]] = {}
    files = sorted(DATA_DIR.glob("*.csv"))
    for i, f in enumerate(files, 1):
        df = load_data(f, 180)
        if df is None:
            continue
        ds = df["date"].dt.strftime("%Y-%m-%d").values
        cl = df["close"].values
        pct = (cl[1:] / cl[:-1] - 1) * 100
        for d, p in zip(ds[1:], pct):
            b = breadth.get(d)
            if b is None:
                b = breadth[d] = [0, 0]
            if p > 0:
                b[0] += 1
            elif p < 0:
                b[1] += 1
        if i % 1000 == 0:
            print(f"  pass1 {i}/{len(files)}", file=sys.stderr)
    return breadth


def main(drop20: float, n_desp: int, reb: float, hold: int):
    breadth = pass1_breadth()
    ratio = {d: v[1] / (v[0] + v[1]) for d, v in breadth.items()
             if v[0] + v[1] >= 1500}
    desp = {d for d, r in ratio.items() if r >= 0.80}

    idx = load_index("zz1000")
    dates = [d.strftime("%Y-%m-%d") for d in idx.index]
    icl = idx.values
    n = len(icl)
    dcum = np.concatenate(
        [[0], np.cumsum([1 if d in desp else 0 for d in dates])])

    periods = []  # (t0, lo_i, t1)
    t = 25
    while t < n:
        r20 = (icl[t] / icl[t - 20] - 1) * 100
        if r20 <= drop20 and dcum[t + 1] - dcum[max(0, t - 19)] >= n_desp:
            lo, lo_i = icl[t], t
            t1 = None
            for s in range(t + 1, min(t + 260, n)):
                if icl[s] < lo:
                    lo, lo_i = icl[s], s
                if (icl[s] / lo - 1) * 100 >= reb and s - lo_i >= hold:
                    t1 = s
                    break
            if t1 is not None:
                periods.append((t, lo_i, t1))
                t = t1 + 1
            else:
                t += 21
        else:
            t += 1

    print(f"\n绝望期 {len(periods)} 个:")
    for t0, lo_i, t1 in periods:
        print(f"  确认 {dates[t0]} | 低点 {dates[lo_i]} "
              f"({(icl[lo_i] / icl[t0 - 20] - 1) * 100:.1f}%) | "
              f"企稳 {dates[t1]} | 反弹段 {(icl[t1] / icl[lo_i] - 1) * 100:+.1f}%")
    if not periods:
        sys.exit("无绝望期（尝试放宽 --drop20/--reb）")

    pmap = {k: (dates[t0], dates[lo_i], dates[t1])
            for k, (t0, lo_i, t1) in enumerate(periods)}
    idx_pos = {d: i for i, d in enumerate(dates)}

    def idx_fwd(D, h):
        j = idx_pos[D] + h
        return (icl[j] / icl[idx_pos[D]] - 1) * 100 if j < n else np.nan

    rows = []
    files = sorted(DATA_DIR.glob("*.csv"))
    for i, f in enumerate(files, 1):
        df = load_data(f, 180)
        if df is None:
            continue
        ds = list(df["date"].dt.strftime("%Y-%m-%d"))
        cl_s = df["close"].values
        low_s = df["low"].values
        pos = {d: k for k, d in enumerate(ds)}
        for k, (d0, dl, d1) in pmap.items():
            k0, kl, k1 = pos.get(d0), pos.get(dl), pos.get(d1)
            if (k0 is None or kl is None or k1 is None
                    or k0 < 20 or k1 + 60 >= len(cl_s)):
                continue
            ks = k0 - 20  # 主跌段起点 = 下跌起点
            seg = (cl_s[kl] / cl_s[ks] - 1) * 100
            r20 = (cl_s[k1 + 20] / cl_s[k1] - 1) * 100
            r60 = (cl_s[k1 + 60] / cl_s[k1] - 1) * 100
            mdd = (low_s[k1 + 1:k1 + 61].min() / cl_s[k1] - 1) * 100
            reb_s = (cl_s[k1] / cl_s[kl] - 1) * 100
            rows.append([f.stem, k, seg, reb_s, r20, r60, mdd])
        if i % 1000 == 0:
            print(f"  pass2 {i}/{len(files)}", file=sys.stderr)

    df = pd.DataFrame(rows, columns=["code", "per", "seg", "reb",
                                     "r20", "r60", "mdd"])
    df["ex60"] = [r["r60"] - idx_fwd(pmap[r["per"]][2], 60)
                  if not pd.isna(r["r60"]) else np.nan
                  for _, r in df.iterrows()]

    def grp(s):
        if s > 0:
            return "1逆行者(>0)"
        if s >= -5:
            return "2抗跌者(-5~0)"
        if s > -15:
            return "3承压(-15~-5)"
        return "4大跌(<=-15)"

    df["g"] = df["seg"].apply(grp)
    df = df.dropna(subset=["r60"])
    print(f"\n样本 {len(df)}（{df['per'].nunique()} 期）")

    t = df.groupby("g").agg(
        n=("r60", "size"),
        主跌段中位=("seg", lambda s: round(s.median(), 1)),
        反弹段中位=("reb", lambda s: round(s.median(), 1)),
        E20=("r20", lambda s: round(s.mean(), 2)),
        E60=("r60", lambda s: round(s.mean(), 2)),
        ex60=("ex60", lambda s: round(s.mean(), 2)),
        win60=("r60", lambda s: round((s > 0).mean() * 100, 1)),
        mdd=("mdd", lambda s: round(s.mean(), 2)))
    print(t.to_string())

    print("\n-- 分期（各组 60日胜率% / E60）")
    for k in sorted(df["per"].unique()):
        s = df[df["per"] == k]
        line = f"  {pmap[k][0]} (n={len(s)}): "
        for gname in sorted(s["g"].unique()):
            gg = s[s["g"] == gname]
            if len(gg) >= 20:
                line += (f"{gname.split('(')[0]}"
                         f"{(gg['r60'] > 0).mean() * 100:.0f}%/"
                         f"{gg['r60'].mean():+.1f} ")
        print(line)

    print("\n-- 污染检查：各组反弹段涨幅分布（大跌组显著大 -> "
          "刨除反弹段的必要性）")
    print(df.groupby("g")["reb"].describe()[["25%", "50%", "75%"]]
          .round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop20", type=float, default=-10.0,
                    help="初期: 指数20日累计跌幅阈值%%（默认 -10）")
    ap.add_argument("--ndesp", type=int, default=2,
                    help="初期: 20日内绝望日最少个数（默认 2）")
    ap.add_argument("--reb", type=float, default=7.0,
                    help="企稳: 自低点反弹阈值%%（默认 7）")
    ap.add_argument("--hold", type=int, default=10,
                    help="企稳: 低点后最少交易日（默认 10）")
    a = ap.parse_args()
    main(a.drop20, a.ndesp, a.reb, a.hold)
