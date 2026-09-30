"""启动确认参数敏感性网格（discover_09 参数敏感性节）。

单次扫描即完成：`pullback_zone_grid` 为每个回踩所属的金叉波段标记
12 个格子（实体 {1,3,5}% × 量比 {1.5,2,3,4}×）的通过与否，
本脚本按格子过滤并比较「通过组 vs 未通过组」的收益差。

用法:
    python3 run.py catch patterns/pullback_zone_grid.yaml
    python3 run.py analyze analyzers/pullback_zone_grid.yaml
    python3 tools/pullback_grid.py [--zone zf]

口径见 docs/appendix.md "09b"；Δ 列 = 通过组 − 未通过组（正 = 该格有正向筛选价值）。
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "output" / "an_pullback_zone_grid.csv"
BODIES = [1.0, 3.0, 5.0]
VOLRS = [1.5, 2.0, 3.0, 4.0]


def _col(x: float, y: float) -> str:
    return f"pass_b{int(x * 10)}_v{int(y * 10)}"


def _stat(g: pd.DataFrame, h: int) -> dict:
    r, nh, dd = f"ret_{h}d", f"new_high_{h}d", f"maxdn_{h}d"
    return {
        "n": len(g),
        "E": round(g[r].mean(), 2),
        "胜率": round((g[r] > 0).mean() * 100, 1),
        "新高": round(g[nh].mean() * 100, 1),
        "MaxDD": round(g[dd].mean(), 2),
    }


def _row(df: pd.DataFrame, x: float, y: float) -> dict:
    c = _col(x, y)
    if c not in df.columns:
        return {"实体≥": x, "量比≥": y, "n": None}
    a = df[df[c]]                      # 通过该格
    b = df[~df[c]]                     # 未通过（被剔除）
    sa, sb = _stat(a, 20), _stat(b, 20)
    s60, s60b = _stat(a, 60), _stat(b, 60)
    return {
        "实体≥": x, "量比≥": y,
        "通过n": sa["n"], "保留%": round(sa["n"] / len(df) * 100, 1),
        "E20": sa["E"], "新高20": sa["新高"], "胜20": sa["胜率"],
        "Δ新高20": round(sa["新高"] - sb["新高"], 1),
        "ΔE20": round(sa["E"] - sb["E"], 2),
        "E60": s60["E"], "新高60": s60["新高"],
        "Δ新高60": round(s60["新高"] - s60b["新高"], 1),
        "ΔE60": round(s60["E"] - s60b["E"], 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default=None, help="只统计指定落点区（zf/mid/zs）")
    args = ap.parse_args()

    if not AN.exists():
        print(f"缺少 {AN}，请先跑 catch + analyze", file=sys.stderr)
        return 1
    df = pd.read_csv(AN, encoding="utf-8-sig")
    if args.zone:
        df = df[df["zone"] == args.zone]
    print(f"样本 n={len(df):,}" + (f" (zone={args.zone})" if args.zone else "（全体，含 merged）"))
    base = _stat(df, 20)
    b60 = _stat(df, 60)
    print(f"基线（无过滤）: 20日 E{base['E']:+}% 胜{base['胜率']}% 新高{base['新高']}% | "
          f"60日 E{b60['E']:+}% 新高{b60['新高']}%")

    rows = [_row(df, x, y) for x in BODIES for y in VOLRS]
    out = pd.DataFrame(rows)
    print("\n=== 网格：通过组的表现 与 相对未通过组的增量 ===")
    print(out.to_string(index=False))

    print("\n=== 单调性检查（Δ新高60，越小/越负 = 该条件越有害）===")
    piv = out.pivot(index="实体≥", columns="量比≥", values="Δ新高60")
    print(piv.to_string())
    piv20 = out.pivot(index="实体≥", columns="量比≥", values="Δ新高20")
    print("\n-- Δ新高20 --")
    print(piv20.to_string())

    pos = out[(out["Δ新高20"] > 0) | (out["Δ新高60"] > 0)]
    print(f"\n存在正向增量的格子: {len(pos)}/12")
    if len(pos):
        print(pos.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
