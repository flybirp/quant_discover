"""前期状态（底部纠缠）阈值敏感性网格 3×3×3 = 27 格。

主指标 = Δ涨≥10%（20 日）/ Δ涨≥20%（60 日）：用户假设"底部纠缠 + 放量长阳
涨幅最猛"，故以**大涨概率**而非均值为判据。Δ = 通过组 − 未通过组。

用法:
    python3 run.py catch patterns/pullback_zone_grid.yaml
    python3 run.py analyze analyzers/pullback_zone_grid.yaml
    python3 tools/pullback_hist_grid.py [--good-only]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "output" / "an_pullback_zone_grid.csv"
ENTS = [60, 70, 80]
SLOPES = [5, 10, 15]
CVS = [6, 8, 10]


def _col(e: int, s: int, c: int) -> str:
    return f"hf_e{e}_s{s}_c{c}"


def _rate(g: pd.DataFrame) -> dict:
    return {
        "涨≥10%": round((g["ret_20d"] >= 10).mean() * 100, 1),
        "涨≥20%": round((g["ret_60d"] >= 20).mean() * 100, 1),
        "胜20": round((g["ret_20d"] > 0).mean() * 100, 1),
        "中位60": round(g["ret_60d"].median(), 2),
        "E60": round(g["ret_60d"].mean(), 2),
        "MaxDD60": round(g["maxdn_60d"].mean(), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--good-only", action="store_true",
                    help="只统计 bk_good（优质启动）样本")
    args = ap.parse_args()
    if not AN.exists():
        print(f"缺少 {AN}", file=sys.stderr)
        return 1

    d = pd.read_csv(AN, encoding="utf-8-sig")
    if args.good_only:
        d = d[d["bk_good"]]
    print(f"样本 n={len(d):,}" + ("（仅 bk_good 优质启动）" if args.good_only else "（全体）"))
    base = _rate(d)
    print("基线: " + "  ".join(f"{k}={v}" for k, v in base.items()))

    rows = []
    for e in ENTS:
        for s in SLOPES:
            for c in CVS:
                col = _col(e, s, c)
                if col not in d.columns:
                    continue
                a, b = d[d[col]], d[~d[col]]
                if len(a) < 100:
                    continue
                ra = _rate(a)
                rows.append({
                    "纠缠≥": e, "斜率≤": s / 100, "CV≤": c / 10,
                    "n": len(a), "占比%": round(len(a) / len(d) * 100, 1),
                    **ra,
                    "Δ涨≥10%": round(ra["涨≥10%"] - (b["ret_20d"] >= 10).mean() * 100, 1),
                    "Δ涨≥20%": round(ra["涨≥20%"] - (b["ret_60d"] >= 20).mean() * 100, 1),
                    "Δ胜20": round(ra["胜20"] - (b["ret_20d"] > 0).mean() * 100, 1),
                    "Δ中位60": round(ra["中位60"] - b["ret_60d"].median(), 2),
                })
    out = pd.DataFrame(rows)
    print("\n=== 27 格（按 Δ涨≥10% 排序）===")
    print(out.sort_values("Δ涨≥10%", ascending=False).to_string(index=False))

    print("\n=== 单调性：Δ涨≥10%（行=纠缠, 列=CV, 块=斜率）===")
    for s in SLOPES:
        sub = out[out["斜率≤"] == s / 100]
        if len(sub):
            print(f"\n-- 斜率 ≤ {s / 100}%/日 --")
            print(sub.pivot(index="纠缠≥", columns="CV≤", values="Δ涨≥10%").to_string())
    print("\n=== 单调性：Δ涨≥20% ===")
    for s in SLOPES:
        sub = out[out["斜率≤"] == s / 100]
        if len(sub):
            print(f"\n-- 斜率 ≤ {s / 100}%/日 --")
            print(sub.pivot(index="纠缠≥", columns="CV≤", values="Δ涨≥20%").to_string())

    pos = out[(out["Δ涨≥10%"] > 0) | (out["Δ涨≥20%"] > 0)]
    print(f"\n存在正向增量的格子: {len(pos)}/{len(out)}")
    if len(pos):
        print(pos.sort_values("Δ涨≥10%", ascending=False).to_string(index=False))
    else:
        print("→ 27 格全部为负：放宽阈值也无法让'底部纠缠'变成正向过滤器")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
