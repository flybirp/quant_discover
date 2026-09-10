#!/usr/bin/env python3
"""大级别单顶特征研究（discover_28 复现工具，R2 节口径）。

输入: catch_major_top.csv（major_top detector 输出，含后验标签）
输出:
  1) 大顶 vs 假顶 特征区分度表（AUC + 两组中位）
  2) 事前可用性二次检验: 特征 -> 顶点后 60/120 日收益的 RankIC
     + 五分位组收益（不经大顶标签，附录六 R2 节）
  3) 年度去聚类复核（2015 vs 非 2015）

特征全部为顶点日事前可得（附录一量比默认口径、真自然周）；
is_major/is_fake/single_top 仅用于分组，禁作特征。
收益 = close[顶点+60]/顶点收盘-1（day horizon 口径，附录一）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.engine import load_data  # noqa: E402

DATA = Path("/Users/flybirp/Documents/mainland_data_2014")

FEATURES = [
    "day_ret", "day_range", "day_body", "upper_shadow", "is_red",
    "vol_ratio", "vol_lead20", "vol_lead60", "dd_250",
    "week_ret", "week_vol_ratio",
]


def extract(row_group: pd.DataFrame, df: pd.DataFrame) -> list[dict]:
    """对单只股票的全部顶点行提取特征 + 后续收益。"""
    ds = df["date"].dt.strftime("%Y-%m-%d").values
    close = df["close"].values.astype(float)
    open_ = df["open"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    vol = df["volume"].values.astype(float)
    periods = df["date"].dt.to_period("W").values
    idx = {d: k for k, d in enumerate(ds)}
    n = len(close)

    out = []
    for _, r in row_group.iterrows():
        i = idx.get(str(r["base_date"]))
        if i is None or i < 60:
            continue
        rng = high[i] - low[i]
        w20 = slice(i - 19, i + 1)
        w60 = slice(i - 59, i + 1)
        # 量峰领先 = 窗口内 argmax(volume) 距顶点的交易日距离（正=量峰在前）
        lead20 = 19 - int(np.argmax(vol[w20]))
        lead60 = 59 - int(np.argmax(vol[w60]))
        # 过去 250 日最大回撤（close 口径 running peak）
        seg = close[i - 250:i + 1] if i >= 250 else close[:i + 1]
        rp = np.maximum.accumulate(seg)
        dd250 = float(((rp - seg) / rp).max() * 100)
        # 当周（真自然周）
        wk = periods[i]
        wk_idx = np.flatnonzero(periods[max(0, i - 10):i + 1] == wk) + max(0, i - 10)
        prev_wk = periods[wk_idx[0] - 1] if wk_idx[0] > 0 else None
        prev_idx = np.flatnonzero(periods[:wk_idx[0]] == prev_wk) if prev_wk else np.array([])
        week_ret = ((close[wk_idx[-1]] / close[prev_idx[-1]] - 1) * 100
                    if len(prev_idx) else np.nan)
        # 当周量 / 前 8 周均量
        week_vol = float(vol[wk_idx].sum())
        vols8 = []
        w_end = wk_idx[0] - 1
        for _ in range(8):
            if w_end < 0:
                break
            pw = periods[w_end]
            pj = np.flatnonzero(periods[:w_end + 1] == pw)
            vols8.append(vol[pj].sum())
            w_end = pj[0] - 1
        week_vol_ratio = week_vol / np.mean(vols8) if vols8 else np.nan

        rec = {
            "code": r["code"], "base_date": r["base_date"],
            "is_major": r["is_major"], "is_fake": r["is_fake"],
            "single_top": r["single_top"], "prior_mult": r["prior_mult"],
            "day_ret": (close[i] / close[i - 1] - 1) * 100,
            "day_range": rng / close[i - 1] * 100,
            "day_body": abs(close[i] - open_[i]) / rng * 100 if rng else np.nan,
            "upper_shadow": (high[i] - max(open_[i], close[i])) / rng * 100 if rng else np.nan,
            "is_red": close[i] > open_[i],
            "vol_ratio": vol[i] / vol[w20].mean(),
            "vol_lead20": lead20,
            "vol_lead60": lead60,
            "dd_250": dd250,
            "week_ret": week_ret,
            "week_vol_ratio": week_vol_ratio,
        }
        for h in (60, 120):
            rec[f"ret_{h}d"] = ((close[i + h] / close[i] - 1) * 100
                                if i + h < n else np.nan)
        out.append(rec)
    return out


def auc(a: pd.Series, b: pd.Series) -> float:
    """Mann-Whitney AUC：a 组的 rank 均值比例（>0.5 = a 组偏大）。"""
    s = pd.concat([a, b])
    r = s.rank()
    n1, n2 = len(a), len(b)
    if not n1 or not n2:
        return np.nan
    return (r.iloc[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def main():
    sig = pd.read_csv(ROOT / "output/catch_major_top.csv",
                      encoding="utf-8-sig", dtype={"code": str}, low_memory=False)
    sig["code"] = sig["code"].str.replace(r"\.0$", "", regex=True)
    rows = []
    files = {c: DATA / f"{c}.csv" for c in sig["code"].unique()}
    for k, (code, g) in enumerate(sig.groupby("code"), 1):
        fp = files[code]
        if not fp.exists():
            continue
        df = load_data(fp, 600)
        if df is None:
            continue
        rows.extend(extract(g, df))
        if k % 1000 == 0:
            print(f"  {k}/{len(files)}", file=sys.stderr)

    d = pd.DataFrame(rows)
    d["year"] = d["base_date"].astype(str).str[:4]
    d.to_csv(ROOT / "output/top_profile_features.csv",
             index=False, encoding="utf-8-sig")
    maj, fake = d[d["is_major"]], d[d["is_fake"]]
    print(f"\n候选 {len(d)} | 大顶 {len(maj)} | 假顶 {len(fake)}\n")

    print("== 一、大顶 vs 假顶 特征区分度（AUC>0.5 = 大顶组偏大）")
    print(f"{'特征':<15} {'大顶中位':>9} {'假顶中位':>9} {'AUC':>6}")
    print("-" * 46)
    aucs = {}
    for f in FEATURES:
        a, b = maj[f].dropna(), fake[f].dropna()
        v = auc(a, b)
        aucs[f] = v
        print(f"{f:<15} {a.median():>8.2f} {b.median():>8.2f} {v:>6.3f}")

    print("\n== 二、事前可用性二次检验（特征 -> 顶点后收益，不经标签）")
    print(f"{'特征':<15} {'IC60':>7} {'IC120':>7} "
          f"{'Q1胜率':>7} {'Q5胜率':>7} {'Q5-Q1的E120':>11}")
    print("-" * 60)
    dd = d.dropna(subset=["ret_120d"])
    for f in FEATURES:
        ok = dd[f].notna()
        sub = dd[ok]
        ic60 = sub[f].corr(sub["ret_60d"], method="spearman")
        ic120 = sub[f].corr(sub["ret_120d"], method="spearman")
        q = pd.qcut(sub[f], 5, labels=False, duplicates="drop")
        if q.nunique() < 5:
            continue
        q1 = sub[q == 0]["ret_120d"]
        q5 = sub[q == q.max()]["ret_120d"]
        print(f"{f:<15} {ic60:>+7.3f} {ic120:>+7.3f} "
              f"{(q1 > 0).mean() * 100:>6.1f}% {(q5 > 0).mean() * 100:>6.1f}% "
              f"{q5.mean() - q1.mean():>+10.2f}")

    print("\n== 三、年度去聚类复核（2015 vs 非 2015 的 AUC 方向）")
    for f in FEATURES:
        if f == "is_red":
            continue
        a15 = auc(maj[maj["year"] == "2015"][f].dropna(),
                  fake[fake["year"] == "2015"][f].dropna())
        anon = auc(maj[maj["year"] != "2015"][f].dropna(),
                   fake[fake["year"] != "2015"][f].dropna())
        flag = "" if (a15 - 0.5) * (anon - 0.5) > 0 else "  ⚠方向翻转"
        print(f"  {f:<15} 2015 AUC {a15:.3f} | 非2015 {anon:.3f}{flag}")


if __name__ == "__main__":
    main()
