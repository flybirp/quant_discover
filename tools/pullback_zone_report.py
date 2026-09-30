"""回踩三区 × 次数 的收益交叉表（discover_09b）。

用法:
    python3 run.py catch patterns/pullback_zone.yaml
    python3 run.py analyze analyzers/pullback_zone.yaml
    python3 tools/pullback_zone_report.py

口径: 见 docs/appendix.md "09b pullback_zone 回踩三区"。
  - 收益分组只使用 zone（事前口径），zone_deepest 仅作对照列输出。
  - merged（两线未分离, gap<min_gap_pct）不参与分区统计，单列报告。
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "output" / "an_pullback_zone.csv"
ZONES = ["zf", "mid", "zs"]
NOS = ["第1次", "第2次", "第3次"]


def _stats(g: pd.DataFrame, h: int) -> dict:
    r, d = f"ret_{h}d", f"maxdn_{h}d"
    nh = f"new_high_{h}d"
    return {
        "n": len(g),
        "E": round(g[r].mean(), 2),
        "中位": round(g[r].median(), 2),
        "平均MaxDD": round(g[d].mean(), 2),
        "胜率": round((g[r] > 0).mean() * 100, 1),
        "创新高率": round(g[nh].mean() * 100, 1) if nh in g else float("nan"),
    }


def _block(df: pd.DataFrame, h: int, title: str) -> None:
    print(f"\n=== {title}（{h} 日, close 基准 = 回踩日收盘） ===")
    rows = []
    for no in NOS:
        for z in ZONES:
            g = df[(df["no_bin"] == no) & (df["zone"] == z)]
            if len(g) < 200:
                rows.append({"次数": no, "区": z, "n": len(g),
                             **{k: None for k in ["E", "中位", "平均MaxDD", "胜率", "创新高率"]}})
                continue
            rows.append({"次数": no, "区": z, **_stats(g, h)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    print(f"\n-- 边际：按区（{h} 日） --")
    for z in ZONES:
        g = df[df["zone"] == z]
        if len(g):
            print(f"{z:4} n={len(g):>7}  " + "  ".join(
                f"{k}={v}" for k, v in _stats(g, h).items() if k != "n"))
    print(f"-- 边际：按次数（{h} 日） --")
    for no in NOS:
        g = df[df["no_bin"] == no]
        if len(g):
            print(f"{no:6} n={len(g):>7}  " + "  ".join(
                f"{k}={v}" for k, v in _stats(g, h).items() if k != "n"))


def main() -> int:
    if not AN.exists():
        print(f"缺少 {AN}，请先跑 catch + analyze", file=sys.stderr)
        return 1
    df = pd.read_csv(AN, encoding="utf-8-sig")
    print(f"总样本 n={len(df)}")
    print("\nzone 分布:\n" + df["zone"].value_counts().to_string())
    print("\nzone_deepest 分布（事后标签，仅对照）:\n"
          + df["zone_deepest"].value_counts().to_string())

    part = df[df["zone"].isin(ZONES)].copy()
    print(f"\n参与分区统计 n={len(part)}（已剔除 merged={int((df['zone'] == 'merged').sum())}）")

    _block(part, 20, "回踩三区 × 次数")
    _block(part, 60, "回踩三区 × 次数")

    # 分年稳健性（第1次 + 各区的 20 日创新高率）
    part["year"] = part["base_date"].astype(str).str[:4]
    print("\n=== 分年：20 日创新高率（%）===")
    pv = part.pivot_table(index="year", columns="zone", values="new_high_20d",
                          aggfunc=lambda s: round(s.mean() * 100, 1))
    pn = part.pivot_table(index="year", columns="zone", values="new_high_20d", aggfunc="size")
    print(pv.to_string())
    print("\n-- 各格子样本数 --")
    print(pn.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
