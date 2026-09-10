#!/usr/bin/env python3
"""RPS（相对强度）增强作用检验 —— discover_15 的正式复现工具。

对指定形态的 an 明细，按 base_date 当日全市场截面计算 rel_ret_120d
百分位（RPS 0~99），输出：分组收益表（E/中位/平均MaxDD/胜率）、
逐日截面 RankIC 均值、分年 <20 vs >=80 胜率对照、年度样本分布。

用法:
    python3 tools/rps_check.py                          # 默认四形态
    python3 tools/rps_check.py zfzs consolidation       # 指定形态
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.engine import load_data          # noqa: E402
from core.market import relative_return    # noqa: E402

DATA_DIR = Path("/Users/flybirp/Documents/mainland_data_2014")
INDEX = "zz1000"
PERIOD = 120

# 形态 -> an 明细（需含 code/base_date/ret_60d/maxdn_60d）
PATTERNS = {
    "base_break": "an_base_break_days.csv",
    "zfzs": "an_zfzs.csv",
    "spike_top": "an_spike.csv",
    "consolidation": "an_consolidation.csv",
}


def rps_group(r: float) -> str | None:
    if r < 20:
        return "<20"
    if r >= 80:
        return ">=80"
    if 50 <= r < 80:
        return "50-80"
    return None


def build_market_pct(need_dates: set[str]) -> dict[str, dict[str, float]]:
    """全市场逐股计算 rel_ret_120d，返回 {date -> {code: 截面百分位 0~99}}。"""
    rel_by_date: dict[str, dict[str, float]] = {d: {} for d in need_dates}
    files = sorted(DATA_DIR.glob("*.csv"))
    for i, f in enumerate(files, 1):
        df = load_data(f, 180)
        if df is None:
            continue
        rel = relative_return(df, period=PERIOD, index_name=INDEX)
        ds = df["date"].dt.strftime("%Y-%m-%d")
        for d, v in zip(ds.values, rel.values):
            if d in rel_by_date and not pd.isna(v):
                rel_by_date[d][f.stem] = float(v)
        if i % 1000 == 0:
            print(f"  进度 {i}/{len(files)}", file=sys.stderr)

    pct: dict[str, dict[str, float]] = {}
    for d, cd in rel_by_date.items():
        if len(cd) >= 50:
            s = pd.Series(cd)
            pct[d] = dict(zip(cd.keys(),
                              ((s.rank() - 1) / len(s) * 100).to_numpy()))
    return pct, rel_by_date


def main(names: list[str]):
    frames = {}
    need_dates = set()
    for name in names:
        fp = ROOT / "output" / PATTERNS[name]
        a = pd.read_csv(fp, encoding="utf-8-sig",
                        usecols=["code", "base_date", "ret_60d", "maxdn_60d"])
        frames[name] = a
        need_dates.update(a["base_date"].dropna().unique())
    print(f"截面日期数: {len(need_dates)}", file=sys.stderr)

    pct_by_date, rel_by_date = build_market_pct(need_dates)
    print(f"有效截面日: {len(pct_by_date)}", file=sys.stderr)

    for name in names:
        a = frames[name]
        rps = [pct_by_date.get(d, {}).get(str(c).zfill(6))
               for c, d in zip(a["code"], a["base_date"])]
        rel = [rel_by_date.get(d, {}).get(str(c).zfill(6))
               for c, d in zip(a["code"], a["base_date"])]
        a = a.assign(rps=rps, rel=rel).dropna(subset=["rps", "ret_60d"])
        a["grp"] = a["rps"].apply(rps_group)
        a["year"] = a["base_date"].str[:4]
        a = a.dropna(subset=["grp"])

        ics = [spearmanr(sub["rel"], sub["ret_60d"]).statistic
               for _, sub in a.groupby("base_date") if len(sub) >= 30]
        ics = [x for x in ics if not pd.isna(x)]
        rank_ic = sum(ics) / len(ics) if ics else float("nan")

        print(f"\n{'=' * 70}")
        print(f"== {name}  n={len(a)}  RankIC={rank_ic:+.3f} "
              f"(逐日截面均值, {len(ics)} 个截面)")
        g = a.groupby("grp").agg(
            E=("ret_60d", lambda s: round(s.mean(), 2)),
            med=("ret_60d", lambda s: round(s.median(), 2)),
            maxdd=("maxdn_60d", lambda s: round(s.mean(), 2)),
            win=("ret_60d", lambda s: round((s > 0).mean() * 100, 1)),
            n=("ret_60d", "size"),
        ).reindex(["<20", "50-80", ">=80"])
        print(g.to_string())

        print("-- 分年 (<20 vs >=80 胜率%)")
        for y, sub in a.groupby("year"):
            lo, hi = sub[sub["grp"] == "<20"], sub[sub["grp"] == ">=80"]
            if len(lo) >= 30 and len(hi) >= 30:
                wl = (lo["ret_60d"] > 0).mean() * 100
                wh = (hi["ret_60d"] > 0).mean() * 100
                print(f"   {y}: <20 {wl:5.1f} (n={len(lo):5}) | "
                      f">=80 {wh:5.1f} (n={len(hi):5}) | "
                      f"{'高强' if wh > wl else '低强'}")

        # 聚类贡献（统一组件 tools/year_robustness）
        from tools.year_robustness import cluster_index
        ci = cluster_index(a, "grp", "ret_60d")
        print(f"   聚类贡献指数 {ci['cluster_index']}（组间差 "
              f"{ci['effect_all']} / 剔除{ci['top_year']}年 "
              f"{ci['effect_ex_top_year']}）→ {ci['verdict']}")

        yc = a["year"].value_counts(normalize=True).sort_index()
        top = yc.head(3).round(3)
        print(f"-- 年度集中度: " + ", ".join(f"{y} {p*100:.0f}%" for y, p in top.items()))


if __name__ == "__main__":
    args = sys.argv[1:] or list(PATTERNS)
    unknown = [x for x in args if x not in PATTERNS]
    if unknown:
        sys.exit(f"未知形态: {unknown}，可选: {list(PATTERNS)}")
    main(args)
