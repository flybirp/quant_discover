"""启动 K 线质量分组检验（discover_09 启动质量节）。

把"金叉前有放量长阳"进一步拆成三类（用户洞察：长上影 + 窗口最高点 = 出货）：

  G0 无达标启动    : 窗口内无 实体≥min_body 且 量比≥vol_ratio 的 K 线
  G2 优质启动(good) : 达标 且 上影占比 ≤ max_upper_shadow 且 非窗口最高点
  G1 出货嫌疑(bad)  : 达标 且 上影占比 ≥ bad_upper_shadow 且 是窗口最高点
  G3 其余(plain)    : 达标但不属于以上两类

用法:
    python3 run.py catch patterns/pullback_zone_grid.yaml
    python3 run.py analyze analyzers/pullback_zone_grid.yaml
    python3 tools/pullback_bk_quality.py [--zone zf]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "output" / "an_pullback_zone_grid.csv"


def _stat(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {}
    return {
        "n": len(g),
        "E20": round(g["ret_20d"].mean(), 2),
        "胜20": round((g["ret_20d"] > 0).mean() * 100, 1),
        "新高20": round(g["new_high_20d"].mean() * 100, 1),
        "涨≥10%": round((g["ret_20d"] >= 10).mean() * 100, 1),
        "跌≥10%": round((g["ret_20d"] <= -10).mean() * 100, 1),
        "E60": round(g["ret_60d"].mean(), 2),
        "新高60": round(g["new_high_60d"].mean() * 100, 1),
        "涨≥20%": round((g["ret_60d"] >= 20).mean() * 100, 1),
        "MaxDD60": round(g["maxdn_60d"].mean(), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default=None)
    args = ap.parse_args()
    if not AN.exists():
        print(f"缺少 {AN}", file=sys.stderr)
        return 1

    d = pd.read_csv(AN, encoding="utf-8-sig")
    if args.zone:
        d = d[d["zone"] == args.zone]
    print(f"样本 n={len(d):,}" + (f" (zone={args.zone})" if args.zone else "（全体）"))

    groups = {
        "G0 无达标启动": ~(d["bk_good"] | d["bk_bad"] | d["bk_plain"]),
        "G2 优质启动(短上影·非窗口高)": d["bk_good"],
        "G1 出货嫌疑(长上影·窗口最高)": d["bk_bad"] & ~d["bk_good"],
        "G3 其余达标": d["bk_plain"] & ~d["bk_good"] & ~d["bk_bad"],
    }
    rows = []
    for name, mask in groups.items():
        s = _stat(d[mask])
        if s:
            rows.append({"组": name, **s})
    print("\n=== 启动 K 线质量分组 ===")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 分年 60 日创新高率（%）===")
    d["yr"] = d["base_date"].astype(str).str[:4]
    piv = pd.DataFrame({
        name.split()[0]: d[mask].groupby("yr")["new_high_60d"].mean() * 100
        for name, mask in groups.items()
    }).round(1)
    cnt = pd.DataFrame({
        name.split()[0]: d[mask].groupby("yr").size() for name, mask in groups.items()
    })
    print(piv.to_string())
    print("\n-- 各年样本数 --")
    print(cnt.to_string())

    print("\n=== 上影占比分档（达标启动样本内，20 日）===")
    q = d[d["bk_good"] | d["bk_bad"] | d["bk_plain"]].copy()
    q["us_bin"] = pd.cut(q["bk_upper_shadow_pct"], [-0.1, 10, 20, 30, 40, 60, 100],
                         labels=["≤10%", "10-20%", "20-30%", "30-40%", "40-60%", ">60%"])
    for b, g in q.groupby("us_bin", observed=True):
        if len(g):
            s = _stat(g)
            print(f"上影{b:>7} n={s['n']:>6}  E20={s['E20']:+.2f}%  胜{s['胜20']}%  "
                  f"新高20={s['新高20']}%  新高60={s['新高60']}%  E60={s['E60']:+.2f}%")

    print("\n=== 是否窗口最高点（达标启动样本内）===")
    for flag, g in q.groupby("bk_is_window_high"):
        s = _stat(g)
        print(f"窗口最高={str(flag):5} n={s['n']:>6}  E20={s['E20']:+.2f}%  胜{s['胜20']}%  "
              f"新高20={s['新高20']}%  新高60={s['新高60']}%  E60={s['E60']:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
