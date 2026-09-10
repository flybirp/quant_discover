#!/usr/bin/env python3
"""尖顶腰斩的"下跌形态"特征研究（S5，discover_31）。

背景: discover_30 证明"更深出清"被 2015 绑架（聚类驱动）；
本工具改验**下跌的方式**（速度/斜率/途中反弹），而非深度。
特征全部在腰斩日（=买点）事前可得。

特征（附录口径: 交易日=数组下标差）:
    down_days    顶点 -> 腰斩点的交易日数（已有列，用日线重算核对）
    slope        日均跌幅 = (half/peak - 1) / down_days × 100（%/日）
    rebound_cnt  途中自阶段低点反弹 ≥10% 的次数（每次计数后重置基准）
    max_rebound  途中最大反弹幅度（自阶段低点，%）
    up_days      低点 -> 顶点交易日数（上涨耗时）

检验: 每特征三分位分组的 90 日 E/胜率 + Spearman IC +
      **分年方向稳定性与聚类贡献**（tools/year_robustness 组件，
      >0.5 = 聚类驱动，不得采用）。

用法:
    python3 tools/spike_profile.py                       # 默认 90d
    python3 tools/spike_profile.py --metric ret_120d
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.engine import load_data, DEFAULT_DATA_DIR  # noqa: E402
from tools.year_robustness import stability, cluster_index, year_table  # noqa

FEATURES = ["down_days", "slope", "rebound_cnt", "max_rebound", "up_days"]
REBOUND_PCT = 10.0


def path_features(c: np.ndarray, m: int, j: int) -> dict:
    """顶点 m -> 腰斩点 j 之间的下跌形态特征。"""
    seg = c[m:j + 1]
    days = j - m
    if days <= 0:
        return {}
    slope = (c[j] / c[m] - 1) / days * 100
    # 途中反弹次数与最大反弹（自阶段低点）
    run_low = seg[0]
    cnt, mx = 0, 0.0
    for x in seg[1:]:
        if x < run_low:
            run_low = x
        else:
            r = (x / run_low - 1) * 100
            if r > mx:
                mx = r
            if r >= REBOUND_PCT:
                cnt += 1
                run_low = x      # 重置基准，计下一次反弹
    return {"down_days": days, "slope": round(slope, 3),
            "rebound_cnt": cnt, "max_rebound": round(mx, 2)}


def main(metric: str):
    sig = pd.read_csv(ROOT / "output/catch_spike_top.csv",
                      encoding="utf-8-sig", dtype={"code": str})
    sig["code"] = sig["code"].str.replace(r"\.0$", "", regex=True)
    an = pd.read_csv(ROOT / "output/an_spike.csv", encoding="utf-8-sig",
                     dtype={"code": str}, low_memory=False)
    an["code"] = an["code"].str.replace(r"\.0$", "", regex=True)

    rows = []
    files = sorted(DEFAULT_DATA_DIR.glob("*.csv"))
    for k, f in enumerate(files, 1):
        code = f.stem
        sub = sig[sig["code"] == code]
        if not len(sub):
            continue
        df = load_data(f, 110)
        if df is None:
            continue
        ds = df["date"].dt.strftime("%Y-%m-%d").values
        c = df["close"].values.astype(float)
        pos = {d: i for i, d in enumerate(ds)}
        for _, r in sub.iterrows():
            m = pos.get(str(r["peak_date"]))
            j = pos.get(str(r["half_date"]))
            i0 = pos.get(str(r["low_date"]))
            if m is None or j is None or i0 is None or j <= m:
                continue
            fe = path_features(c, m, j)
            if not fe:
                continue
            fe.update({"code": code, "half_date": r["half_date"],
                       "up_days": m - i0})
            rows.append(fe)
        if k % 1000 == 0:
            print(f"  {k}/{len(files)}", file=sys.stderr)

    d = pd.DataFrame(rows)
    d = d.merge(an[["code", "base_date", metric]],
                left_on=["code", "half_date"], right_on=["code", "base_date"],
                how="inner")
    d = d.dropna(subset=[metric])
    d["year"] = d["half_date"].astype(str).str[:4]
    d["grp"] = d["half_date"].astype(str)      # 占位（分组按特征）
    out = ROOT / "output/spike_profile_features.csv"
    d.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n样本 {len(d)}（腰斩事件 × 收益匹配）\n")
    print("== 特征分组（三分位，低→高）与 90d 表现 + 稳健性")
    print(f"{'特征':<13} {'档':<4} {'n':>5} {'E':>8} {'胜率':>6} "
          f"{'IC':>7} {'方向p':>7} {'聚类指数':>8} {'判定':>10}")
    print("-" * 74)
    for f in FEATURES:
        ok = d[f].notna() & np.isfinite(d[f])
        sub = d[ok]
        ic = sub[f].corr(sub[metric], method="spearman")
        try:
            q = pd.qcut(sub[f], 3, labels=["低", "中", "高"], duplicates="drop")
        except ValueError:
            continue
        for gname in ("低", "中", "高"):
            s = sub[q == gname]
            if not len(s):
                continue
            st = stability(year_table(s, metric), "E")
            ci = cluster_index(s, None, metric)
            print(f"{f:<13} {gname:<4} {len(s):>5} {s[metric].mean():>+7.2f} "
                  f"{(s[metric] > 0).mean() * 100:>5.1f}% {ic:>+7.3f} "
                  f"{st['p_value']:>7.3f} {ci['cluster_index']:>8} "
                  f"{ci['verdict']:>10}")
        print(f"{'':13} {'全部':<4} {len(sub):>5} {sub[metric].mean():>+7.2f} "
              f"{(sub[metric] > 0).mean() * 100:>5.1f}% {ic:>+7.3f}")
        print("-" * 74)
    print(f"\n特征明细已保存: {out}")
    print("\n判读: 采用标准 = ① 组间 E 单调且差距大；② 方向 p<0.05；"
          "③ 聚类指数 ≤0.5（非聚类驱动）。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="ret_90d")
    a = ap.parse_args()
    main(a.metric)
