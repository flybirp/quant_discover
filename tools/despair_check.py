#!/usr/bin/env python3
"""绝望日（市场宽度极端恶化）幸存者研究 —— discover_23 的正式复现工具。

Pass1: 全市场逐日涨跌家数 -> breadth 序列
Pass2: 绝望日截面按当日涨跌幅分组（逆行者/抗跌者/承压/大跌），
       统计末日收盘买入的 20/60 日收益、相对中证1000 超额、平均MaxDD。

口径:
    绝望日   = 当日下跌家数占比 >= 80%（有效截面 >= 1500 只，近年约 4000/5000 只下跌）
    episode  = 交易日历上相邻的绝望日串；连跌 >= 2 天的 episode 为"用户场景"
    逆行者   = 当日涨幅 > 0（红盘）
    抗跌者   = -1% ~ 0%
    承压     = -5% ~ -1%
    大跌     = <= -5%

用法:
    python3 tools/despair_check.py                 # 默认阈值 80%
    python3 tools/despair_check.py --ratio 0.9     # 深度绝望
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
HORIZONS = (20, 60)


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


def split_episodes(desp: list[str], cal: list[str]) -> list[list[str]]:
    pos = {d: i for i, d in enumerate(cal)}
    eps, cur = [], [desp[0]]
    for d in desp[1:]:
        if pos[d] - pos[cur[-1]] == 1:
            cur.append(d)
        else:
            eps.append(cur)
            cur = [d]
    eps.append(cur)
    return eps


def main(ratio_th: float):
    breadth = pass1_breadth()
    cal = sorted(breadth.keys())
    ratio = {d: breadth[d][1] / (breadth[d][0] + breadth[d][1])
             for d in cal if breadth[d][0] + breadth[d][1] >= 1500}
    desp = [d for d in cal if d in ratio and ratio[d] >= ratio_th]
    if not desp:
        sys.exit("该阈值下无绝望日")
    eps = split_episodes(desp, cal)
    ep_end = {e[-1] for e in eps}
    ep_multi = {e[-1] for e in eps if len(e) >= 2}
    print(f"\n有效交易日 {len(ratio)} | 绝望日 {len(desp)} | "
          f"episode {len(eps)} 个（最长 {max(len(e) for e in eps)} 天）| "
          f"连跌>=2天 {len(ep_multi)} 个")
    ys = pd.Series([e[-1][:4] for e in eps]).value_counts().sort_index()
    print("episode 末日按年:", {k: int(v) for k, v in ys.items()})

    # 指数同期收益
    idx = load_index("zz1000")
    idx_ds = {d.strftime("%Y-%m-%d"): v for d, v in idx.items()}
    idx_dates = [d.strftime("%Y-%m-%d") for d in idx.index]
    idx_pos = {d: i for i, d in enumerate(idx_dates)}
    fwd_cache = {}
    for D in set(desp):
        fwd_cache[D] = {}
        j = idx_pos.get(D)
        for h in HORIZONS:
            if j is None or j + h >= len(idx_dates):
                fwd_cache[D][h] = np.nan
            else:
                fwd_cache[D][h] = (idx_ds[idx_dates[j + h]] / idx_ds[D] - 1) * 100

    rows = []
    files = sorted(DATA_DIR.glob("*.csv"))
    for i, f in enumerate(files, 1):
        df = load_data(f, 180)
        if df is None:
            continue
        ds = df["date"].dt.strftime("%Y-%m-%d").values
        cl = df["close"].values
        low = df["low"].values
        pos = {d: k for k, d in enumerate(ds)}
        for D in desp:
            k = pos.get(D)
            if k is None or k < 1:
                continue
            p = (cl[k] / cl[k - 1] - 1) * 100
            k60 = k + 60
            if k60 >= len(cl):
                r20 = r60 = mdd = np.nan
            else:
                r20 = (cl[k + 20] / cl[k] - 1) * 100
                r60 = (cl[k60] / cl[k] - 1) * 100
                mdd = (low[k + 1:k60 + 1].min() / cl[k] - 1) * 100
            rows.append([f.stem, D, p, r20, r60, mdd])
        if i % 1000 == 0:
            print(f"  pass2 {i}/{len(files)}", file=sys.stderr)

    df = pd.DataFrame(rows, columns=["code", "d", "pct", "ret20", "ret60", "mdd60"])
    for h in HORIZONS:
        df[f"ex{h}"] = [
            r[f"ret{h}"] - fwd_cache[r["d"]][h]
            if not pd.isna(r[f"ret{h}"]) else np.nan
            for _, r in df.iterrows()]

    def grp(p):
        if p > 0:
            return "1逆行者(>0%)"
        if p >= -1:
            return "2抗跌者(-1~0%)"
        if p > -5:
            return "3承压(-5~-1%)"
        return "4大跌(<=-5%)"

    df["g"] = df["pct"].apply(grp)
    df["year"] = df["d"].str[:4]

    def report(sub, title):
        sub = sub.dropna(subset=["ret60"])
        print(f"\n== {title}  事件日数={sub['d'].nunique()}  样本={len(sub)}")
        t = sub.groupby("g").agg(
            n=("ret60", "size"),
            E20=("ret20", lambda s: round(s.mean(), 2)),
            E60=("ret60", lambda s: round(s.mean(), 2)),
            ex60=("ex60", lambda s: round(s.mean(), 2)),
            win60=("ret60", lambda s: round((s > 0).mean() * 100, 1)),
            mdd=("mdd60", lambda s: round(s.mean(), 2)))
        print(t.to_string())

    report(df, "全部绝望日（含窗口重叠）")
    report(df[df["d"].isin(ep_end)], "episode 末日（独立事件）")
    ep = df[df["d"].isin(ep_multi)]
    report(ep, "连跌>=2天 episode 末日")
    ep = ep.dropna(subset=["ret60"])
    print("\n-- 连跌>=2天 episode 末日：分年（各组 60日胜率%）")
    for y, s in ep.groupby("year"):
        line = f"   {y} ({s['d'].nunique()}ep): "
        for gname in sorted(s["g"].unique()):
            gg = s[s["g"] == gname]
            if len(gg) >= 30:
                label = gname.split("(")[0]
                line += f"{label}{(gg['ret60'] > 0).mean() * 100:.0f}% "
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.80,
                    help="绝望日下跌家数占比阈值（默认 0.80）")
    main(ap.parse_args().ratio)
