"""前期状态（底部纠缠）检验：长期走平 + 量稳 + 放量长阳 = 最猛？

hist 窗口 = 启动确认窗口**之前**的 hist_lookback(120) 个交易日：
  hist_flat = 期间 zf 在 zs 下方/纠缠占比 ≥ 80%
              且 |zf 斜率| ≤ 0.05%/日、|zs 斜率| ≤ 0.05%/日
              且 量能变异系数 CV ≤ 0.60
（口径见 docs/appendix.md 09b「前期状态」）

分组：hist_flat × bk_good(优质启动) 四象限，并按波动率五分位分层复核
（承接 discover_09 波动率分层的教训：不做分层会得辛普森悖论结论）。

用法:
    python3 run.py catch patterns/pullback_zone_grid.yaml
    python3 run.py analyze analyzers/pullback_zone_grid.yaml
    python3 tools/pullback_hist_flat.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "output" / "an_pullback_zone_grid.csv"


def _stat(g: pd.DataFrame) -> dict:
    if len(g) < 30:
        return {"n": len(g)}
    return {
        "n": len(g),
        "E20": round(g["ret_20d"].mean(), 2),
        "中位20": round(g["ret_20d"].median(), 2),
        "胜20": round((g["ret_20d"] > 0).mean() * 100, 1),
        "涨≥10%": round((g["ret_20d"] >= 10).mean() * 100, 1),
        "E60": round(g["ret_60d"].mean(), 2),
        "中位60": round(g["ret_60d"].median(), 2),
        "涨≥20%": round((g["ret_60d"] >= 20).mean() * 100, 1),
        "新高60": round(g["new_high_60d"].mean() * 100, 1),
        "MaxDD60": round(g["maxdn_60d"].mean(), 2),
    }


def _show(title: str, rows: list) -> None:
    print(f"\n=== {title} ===")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> int:
    if not AN.exists():
        print(f"缺少 {AN}", file=sys.stderr)
        return 1
    d = pd.read_csv(AN, encoding="utf-8-sig")
    d = d[d["vol_60d"].notna()].copy()
    print(f"样本 n={len(d):,}")
    print(f"hist_flat 占比 {d['hist_flat'].mean() * 100:.1f}%  "
          f"(n={int(d['hist_flat'].sum()):,})")
    print(f"bk_good 占比 {d['bk_good'].mean() * 100:.1f}%")

    quads = {
        "① 底部纠缠 + 优质启动": d["hist_flat"] & d["bk_good"],
        "② 底部纠缠 + 非优质": d["hist_flat"] & ~d["bk_good"],
        "③ 非纠缠 + 优质启动": ~d["hist_flat"] & d["bk_good"],
        "④ 非纠缠 + 非优质": ~d["hist_flat"] & ~d["bk_good"],
    }
    _show("四象限（用户假设：① 应最猛）",
          [{"组": k, **_stat(d[m])} for k, m in quads.items()])

    # 波动率分层复核（必须做：上节辛普森悖论教训）
    d["pct"] = d.groupby("base_date")["vol_60d"].rank(pct=True)
    d["q"] = pd.cut(d["pct"], [0, .2, .4, .6, .8, 1.0],
                    labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    rows = []
    for q, g in d.groupby("q", observed=True):
        gm = {
            "① 底部纠缠+优质": g[g["hist_flat"] & g["bk_good"]],
            "② 底部纠缠+非优": g[g["hist_flat"] & ~g["bk_good"]],
            "③ 非纠缠+优质": g[~g["hist_flat"] & g["bk_good"]],
            "④ 非纠缠+非优": g[~g["hist_flat"] & ~g["bk_good"]],
        }
        for k, gg in gm.items():
            s = _stat(gg)
            if s.get("n", 0) >= 100:
                rows.append({"波动率层": q, "组": k, **s})
    _show("波动率分层 × 四象限", rows)

    # 层内 ① vs ④ 的净差
    print("\n=== 层内净差（① 底部纠缠+优质  −  ④ 非纠缠+非优质）===")
    for q, g in d.groupby("q", observed=True):
        a = g[g["hist_flat"] & g["bk_good"]]
        b = g[~g["hist_flat"] & ~g["bk_good"]]
        if len(a) >= 100 and len(b) >= 100:
            print(f"{q}: Δ胜20={(a['ret_20d'] > 0).mean() * 100 - (b['ret_20d'] > 0).mean() * 100:+.1f}pct  "
                  f"Δ中位60={a['ret_60d'].median() - b['ret_60d'].median():+.2f}pct  "
                  f"Δ涨≥20%={(a['ret_60d'] >= 20).mean() * 100 - (b['ret_60d'] >= 20).mean() * 100:+.1f}pct  "
                  f"(①n={len(a)}, ④n={len(b)})")

    # 分年稳健性
    d["yr"] = d["base_date"].astype(str).str[:4]
    print("\n=== 分年：① 组 vs 全体（60 日中位 / 胜20）===")
    for yr, g in d.groupby("yr"):
        a = g[g["hist_flat"] & g["bk_good"]]
        if len(a) >= 50:
            print(f"{yr}: ① n={len(a):>5}  中位60={a['ret_60d'].median():+.2f}%  "
                  f"胜20={(a['ret_20d'] > 0).mean() * 100:.1f}%  |  全体 n={len(g):>6}  "
                  f"中位60={g['ret_60d'].median():+.2f}%  胜20={(g['ret_20d'] > 0).mean() * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
