#!/usr/bin/env python3
"""持有期状态 IC（discover_26 前置，3.4 节口径）。

对每路径在观察点 t ∈ {20,40,60} 取状态，检验状态对
t 后 20/60 日收益的预测力（Spearman IC + 分组）：
    分类状态: 放量滞涨 / 缩量新高 / 普通（量比默认口径，附录一）
    连续状态: 量比 / 持有期回撤 (peak_t-close_t)/peak_t / 浮盈
观测截面 n>=30 才计 IC（附录一 RankIC 惯例）。

用法:
    python3 tools/exit_state.py output/catch_spike_top.csv
    python3 tools/exit_state.py output/catch_pv_divergence_bottom.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exit.replay import load_paths  # noqa: E402

OBS = (20, 40, 60)


def states_of(p) -> list[dict]:
    """单路径在各观察点的状态行 + 未来收益。"""
    rows = []
    cl, vol = p.close, p.vol
    if vol is None:
        raise SystemExit("路径未含成交量（load_paths include_vol）")
    for t in OBS:
        if t + 60 >= len(cl):
            continue
        peak_t = cl[:t + 1].max()
        vr = vol[t] / vol[max(0, t - 19):t + 1].mean()
        day_ret = (cl[t] / cl[t - 1] - 1) * 100 if t >= 1 else np.nan
        is_new_high = cl[t] >= peak_t
        if vr >= 1.5 and day_ret < 0:
            kind = "放量滞涨"
        elif vr <= 0.7 and is_new_high:
            kind = "缩量新高"
        else:
            kind = "普通"
        # 持有期超额（附录六）：个股收益 - 同期基准收益（百分点）
        if p.bench is not None and not np.isnan(p.bench[t]):
            ex = ((cl[t] / p.entry - 1) -
                  (p.bench[t] / p.bench[0] - 1)) * 100
        else:
            ex = np.nan
        rows.append({
            "code": p.code, "base_date": p.base_date, "t": t,
            "obs_date": (p.dates[t] if p.dates is not None else None),
            "kind": kind, "vol_ratio": vr, "day_ret": day_ret,
            "dd": (peak_t - cl[t]) / peak_t * 100,
            "pnl": (cl[t] / p.entry - 1) * 100,
            "excess": ex,
            "ret20": (cl[t + 20] / cl[t] - 1) * 100,
            "ret60": (cl[t + 60] / cl[t] - 1) * 100,
        })
    return rows


def main(fp: str, rps_fp: str | None = None):
    paths = load_paths(ROOT / fp, ma_periods=(), max_horizon=130,
                       include_vol=True, bench="zz1000")
    rows = []
    for p in paths:
        rows.extend(states_of(p))
    d = pd.DataFrame(rows)
    d["year"] = d["base_date"].astype(str).str[:4]

    # RPS 持有版：查全市场每日分位表（附录六），计算 RPS 衰减
    if rps_fp:
        rt = pd.read_parquet(rps_fp)
        d = d.merge(rt.rename(columns={"date": "obs_date", "rps": "rps_t"}),
                    on=["code", "obs_date"], how="left")
        d = d.merge(rt.rename(columns={"date": "base_date", "rps": "rps0"}),
                    on=["code", "base_date"], how="left")
        d["rps_decay"] = d["rps_t"] - d["rps0"]
        print(f"  RPS 表已接入: 匹配 t 点 "
              f"{d['rps_t'].notna().mean() * 100:.1f}% | 入场点 "
              f"{d['rps0'].notna().mean() * 100:.1f}%")
    print(f"== {fp} | 路径 {len(paths)} | 状态观测 {len(d)} 行\n")

    print("-- 分类状态（放量滞涨 vs 缩量新高 vs 普通）")
    t = d.groupby("kind").agg(
        n=("ret60", "size"),
        E20=("ret20", lambda s: round(s.mean(), 2)),
        E60=("ret60", lambda s: round(s.mean(), 2)),
        win60=("ret60", lambda s: round((s > 0).mean() * 100, 1)))
    print(t.to_string())

    print("\n-- 连续状态 IC（对 t 后 60 日收益的 Spearman）")
    print(f"{'状态列':<10} {'IC':>7} {'n':>7} {'低档E60':>8} {'高档E60':>8}")
    print("-" * 46)
    extra = [c for c in ("rps_t", "rps_decay")
             if c in d.columns and d[c].notna().sum() > 100]
    for f in ("vol_ratio", "dd", "pnl", "excess", "day_ret") + tuple(extra):
        ok = d[f].notna() & d["ret60"].notna()
        sub = d[ok]
        if len(sub) < 30:
            continue
        ic = sub[f].corr(sub["ret60"], method="spearman")
        q = pd.qcut(sub[f], 5, labels=False, duplicates="drop")
        lo = sub[q == 0]["ret60"].mean()
        hi = sub[q == q.max()]["ret60"].mean()
        print(f"{f:<10} {ic:>+7.3f} {len(sub):>7} "
              f"{lo:>+7.2f} {hi:>+7.2f}")

    print("\n-- 分年方向（IC 符号稳定性，按年计算 vol_ratio/dd/pnl 的 IC60）")
    cols = [c for c in ("vol_ratio", "dd", "pnl", "excess") + tuple(extra)
            if d[c].notna().sum() > 100]
    print(f"{'年':<6}" + "".join(f"{c:>12}" for c in cols))
    for y, s in d.groupby("year"):
        line = f"{y:<6}"
        for c in cols:
            ok = s[c].notna() & s["ret60"].notna()
            if ok.sum() >= 30:
                ic = s[c][ok].corr(s["ret60"][ok], method="spearman")
                line += f"{ic:>+12.3f}"
            else:
                line += f"{'—':>12}"
        print(line)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?",
                    default="output/catch_spike_top.csv")
    ap.add_argument("--rps", default=None,
                    help="RPS 每日分位表 parquet（tools/rps_table.py 产出）")
    a = ap.parse_args()
    main(a.input, a.rps)
