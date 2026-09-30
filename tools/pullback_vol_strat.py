"""波动率分层检验：上影线/窗口最高点的效应是否只是波动率的投影。

背景：上影占比 = 上影 / 全幅，与个股波动率天然正相关。若"上影越大越好"
在**波动率层内**消失，则该效应只是高波动股右尾更肥的投影，上影本身无信息。

分层变量：vol_60d = 回踩日前 60 个交易日日收益 std（%，事前口径，detector 产出）。
两种分层：
  cross  —— 按 base_date 的**横截面分位**（每日全市场排名，剔除市场状态差异，主口径）
  pool   —— 全样本分位（辅助）

用法:
    python3 run.py catch patterns/pullback_zone_grid.yaml
    python3 run.py analyze analyzers/pullback_zone_grid.yaml
    python3 tools/pullback_vol_strat.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "output" / "an_pullback_zone_grid.csv"
LBL = ["Q1最低", "Q2", "Q3", "Q4", "Q5最高"]


def _line(tag: str, g: pd.DataFrame) -> str:
    if len(g) < 100:
        return f"  {tag:26} n={len(g):>6}  (样本不足)"
    return (f"  {tag:26} n={len(g):>6}  中位20={g['ret_20d'].median():+.2f}%  "
            f"中位60={g['ret_60d'].median():+.2f}%  胜20={(g['ret_20d'] > 0).mean() * 100:.1f}%  "
            f"E60={g['ret_60d'].mean():+.2f}%  新高60={g['new_high_60d'].mean() * 100:.1f}%")


def _strat(d: pd.DataFrame, by: str, title: str) -> None:
    print(f"\n########## {title} ##########")
    for q, g in d.groupby(by, observed=True):
        print(f"\n--- 波动率 {q}  vol_60d中位={g['vol_60d'].median():.2f}%  n={len(g):,} ---")
        print(_line("G0 无达标启动", g[~(g["bk_good"] | g["bk_bad"] | g["bk_plain"])]))
        print(_line("G2 优质(短上影·非高)", g[g["bk_good"]]))
        print(_line("G1 出货(长上影·窗口高)", g[g["bk_bad"] & ~g["bk_good"]]))
        q2 = g[g["bk_good"] | g["bk_bad"] | g["bk_plain"]].copy()
        if len(q2) >= 500:
            q2["us"] = pd.cut(q2["bk_upper_shadow_pct"], [-0.1, 20, 40, 100],
                              labels=["上影≤20%", "20-40%", ">40%"])
            print("  -- 层内上影分档（达标样本）--")
            for b, gg in q2.groupby("us", observed=True):
                print(_line(f"    {b}", gg))
            print("  -- 层内窗口最高点（达标样本）--")
            for f, gg in q2.groupby("bk_is_window_high"):
                print(_line(f"    窗口最高={f}", gg))


def main() -> int:
    if not AN.exists():
        print(f"缺少 {AN}", file=sys.stderr)
        return 1
    d = pd.read_csv(AN, encoding="utf-8-sig")
    d = d[d["vol_60d"].notna()].copy()
    print(f"样本 n={len(d):,}（已剔除 vol_60d 缺失）")

    # 主口径：base_date 横截面分位（每日全市场排名）
    d["pct"] = d.groupby("base_date")["vol_60d"].rank(pct=True)
    d["cross"] = pd.cut(d["pct"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=LBL)
    # 辅助：全样本分位
    d["pool"] = pd.qcut(d["vol_60d"], 5, labels=LBL)

    print("\n=== 波动率与启动特征的混淆程度（相关系数）===")
    q = d[d["bk_good"] | d["bk_bad"] | d["bk_plain"]]
    print(f"  上影占比 vs vol_60d  Spearman = {q['bk_upper_shadow_pct'].corr(q['vol_60d'], method='spearman'):.3f}")
    print(f"  窗口最高 vs vol_60d  (点二列) = "
          f"{q['bk_is_window_high'].astype(float).corr(q['vol_60d']):.3f}")
    print(f"  vol_60d vs ret_60d   Spearman = {q['vol_60d'].corr(q['ret_60d'], method='spearman'):.3f}")

    _strat(d, "cross", "主口径：每日横截面波动率五分位")
    _strat(d, "pool", "辅助：全样本波动率五分位")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
