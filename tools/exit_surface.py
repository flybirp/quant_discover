#!/usr/bin/env python3
"""退出规则参数曲面分析（S2）：成本敏感性 + 市场类型交互。

1. 成本敏感性: 同一曲面配置按往返成本 0.15%/0.30%/0.60% 各跑一次，
   汇总每规则集在不同成本下的 E净/胜率/平均持仓/单位时间收益。
2. 市场类型交互: 结果明细按 base_date 前 120 交易日 zz1000 收益
   打标签（牛/震荡/熊，三分位），统计规则集 × 市场类型。

用法:
    python3 tools/exit_surface.py exit_rules/zfzs_surface.yaml
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exit.runner import run_exit          # noqa: E402
from core.market import load_index        # noqa: E402

COSTS = (0.15, 0.30, 0.60)


def market_type(idx: pd.Series, dates: list[str], window: int = 120) -> dict:
    """每个 base_date 前的指数 120 日收益 -> 三分位标签（牛/震荡/熊）。"""
    vals = idx.values.astype(float)
    r = {}
    n = len(vals)
    for i, d in enumerate(dates):
        if i >= window:
            r[d] = (vals[i] / vals[i - window] - 1) * 100
        else:
            r[d] = np.nan
    s = pd.Series(r).dropna()
    q = s.quantile([1 / 3, 2 / 3])
    def lab(v):
        if pd.isna(v):
            return np.nan
        return "熊" if v <= q.iloc[0] else ("震荡" if v <= q.iloc[1] else "牛")
    return {k: lab(v) for k, v in r.items()}


def main(cfg_path: str):
    base = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    name = base["name"]

    frames = []
    for c in COSTS:
        cfg = dict(base)
        cfg["cost_roundtrip"] = c
        cfg["name"] = f"{name}_c{int(c * 100)}"
        out = run_exit(cfg, out_path=ROOT / f"output/exit_{cfg['name']}.csv")
        df = pd.read_csv(out, encoding="utf-8-sig", low_memory=False)
        df["cost"] = c
        frames.append(df)
        print(f"  ✅ 成本 {c:.2f}% 完成（{len(df)} 行）")

    print("\n== 一、参数曲面（成本 0.30% 基准）")
    d = frames[1]
    t = d.groupby("ruleset").agg(
        n=("ret_net", "size"),
        E净=("ret_net", lambda s: round(s.mean(), 2)),
        胜率=("win", lambda s: round(s.mean() * 100, 1)),
        天数=("days", lambda s: round(s.mean(), 1)),
        单位时间=("ret_net", lambda s: round(s.mean() / max(s.index.size, 1), 4))) if False else None
    rows = []
    for rs, s in d.groupby("ruleset"):
        days = s["days"].replace(0, np.nan).mean()
        rows.append({"ruleset": rs, "n": len(s),
                     "E净": round(s["ret_net"].mean(), 2),
                     "胜率": round(s["win"].mean() * 100, 1),
                     "天数": round(s["days"].mean(), 1),
                     "单位时间": round(s["ret_net"].mean() / days, 4)
                     if days and not np.isnan(days) else np.nan})
    t = pd.DataFrame(rows).sort_values("E净", ascending=False)
    print(t.to_string(index=False))

    print("\n== 二、成本敏感性（各规则集 E净 / 单位时间）")
    piv = []
    for rs in t["ruleset"]:
        row = {"ruleset": rs}
        for c, df in zip(COSTS, frames):
            s = df[df["ruleset"] == rs]
            row[f"E@{c}"] = round(s["ret_net"].mean(), 2)
        piv.append(row)
    pv = pd.DataFrame(piv)
    pv["E衰减(0.15→0.60)"] = (pv["E@0.6"] - pv["E@0.15"]).round(2)
    print(pv.to_string(index=False))

    print("\n== 三、市场类型交互（牛/震荡/熊 × 规则集，成本 0.30%）")
    idx = load_index("zz1000")
    dates = [d.strftime("%Y-%m-%d") for d in idx.index]
    mt = market_type(idx, dates)
    d2 = d.copy()
    d2["mt"] = d2["base_date"].map(mt)
    d2 = d2.dropna(subset=["mt"])
    rows = []
    for rs, s in d2.groupby("ruleset"):
        row = {"ruleset": rs}
        for m in ("牛", "震荡", "熊"):
            sub = s[s["mt"] == m]
            row[f"{m}E"] = round(sub["ret_net"].mean(), 2) if len(sub) else np.nan
            row[f"{m}胜率"] = (round(sub["win"].mean() * 100, 1)
                               if len(sub) else np.nan)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n注: 市场类型 = base_date 前 120 交易日中证1000 收益三分位"
          "（熊/震荡/牛），附录一指数口径。")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "exit_rules/zfzs_surface.yaml")
