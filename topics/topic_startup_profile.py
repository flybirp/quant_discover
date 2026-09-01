#!/usr/bin/env python3
"""专题：超大波段启动前半年的量价画像。

读取 output/catch_main_dd50.csv，按波段涨幅分层
（超大>500% / 普通100-300% / 小50-100%），统计每个波段起点前
120 个交易日的量价演变：

  - 整体：前半年涨跌幅 / 最大回撤 / 年化波动率
  - 分段：每 30 天一段的涨跌幅与均量（中位）
  - 末段：最后 5 日逐日涨跌幅与量比（中位）、启动日量比/振幅

用法: python3 topics/topic_startup_profile.py [--n-top N]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.engine import load_data, DEFAULT_DATA_DIR

GROUPS = [("超大(>500%)", 500, 1e9), ("普通(100-300%)", 100, 300), ("小波段(50-100%)", 50, 100)]


def load_returns(start_date: str, df: pd.DataFrame):
    """返回波段起点在日线中的索引，找不到返回 None。"""
    date_to_idx = {d: k for k, d in enumerate(df["date"].dt.strftime("%Y-%m-%d").values)}
    return date_to_idx.get(start_date)


def analyze_group(waves: pd.DataFrame, data_dir: Path) -> dict:
    stats = {"seg_ret": [[] for _ in range(4)], "seg_vol": [[] for _ in range(4)],
             "pre_ret": [], "pre_dd": [], "pre_vol": [],
             "d5": [[] for _ in range(5)], "d5vr": [[] for _ in range(5)],
             "day1_vr": [], "day1_amp": []}
    for code, group in waves.groupby("code"):
        df = load_data(data_dir / f"{code}.csv", 200)
        if df is None:
            continue
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        vol = df["volume"].values.astype(float)
        vol_ma20 = pd.Series(vol).rolling(20).mean().values
        for _, w in group.iterrows():
            si = load_returns(w["start_date"], df)
            if si is None or si < 130:
                continue
            s, e = si - 120, si  # 启动前窗口 [s, e)
            seg = close[s:e]
            stats["pre_ret"].append((seg[-1] / seg[0] - 1) * 100)
            peak = np.maximum.accumulate(seg)
            stats["pre_dd"].append(((peak - seg) / peak * 100).max())
            rets = np.diff(seg) / seg[:-1]
            stats["pre_vol"].append(np.std(rets) * np.sqrt(250) * 100)
            for k in range(4):
                a, b = s + k * 30, s + (k + 1) * 30
                stats["seg_ret"][k].append((close[b] / close[a] - 1) * 100)
                stats["seg_vol"][k].append(np.median(vol[a:b]))
            for d in range(5):
                i = e - 5 + d
                if i - 1 >= 0 and close[i - 1] > 0:
                    stats["d5"][d].append((close[i] / close[i - 1] - 1) * 100)
                if vol_ma20[i] and not np.isnan(vol_ma20[i]) and vol_ma20[i] > 0:
                    stats["d5vr"][d].append(vol[i] / vol_ma20[i])
            i = e - 1
            if vol_ma20[i] and not np.isnan(vol_ma20[i]) and vol_ma20[i] > 0 and i - 5 >= 0:
                stats["day1_vr"].append(vol[i] / np.mean(vol[i - 5:i]))
            if low[i] > 0:
                stats["day1_amp"].append((high[i] - low[i]) / low[i] * 100)
    return stats


def main():
    waves = pd.read_csv(ROOT / "output/catch_main_dd50.csv",
                        encoding="utf-8-sig", dtype={"code": str})
    print(f"主升浪波段总数: {len(waves)}")
    print("分组: 超大(>500%) / 普通(100-300%) / 小波段(50-100%)")
    print()

    all_stats = {}
    for name, lo, hi in GROUPS:
        sub = waves[(waves["return_pct"] > lo) & (waves["return_pct"] <= hi)]
        print(f"--- {name}: {len(sub)} 个波段，分析启动前 120 天 ---")
        st = analyze_group(sub, DEFAULT_DATA_DIR)
        all_stats[name] = st
        n = len(st["pre_ret"])
        if n == 0:
            continue
        print(f"  有效样本 {n}")
        print(f"  整体: 涨跌幅中位 {np.median(st['pre_ret']):+.1f}% | "
              f"最大回撤中位 {np.median(st['pre_dd']):.1f}% | "
              f"年化波动率中位 {np.median(st['pre_vol']):.1f}%")
        labels = ["120-90天前", "90-60天前", "60-30天前", "最后30天"]
        ret_line = "  分段涨跌幅中位: " + "  ".join(
            f"{l} {np.median(st['seg_ret'][k]):+.1f}%" for k, l in enumerate(labels))
        vol_line = "  分段均量中位:   " + "  ".join(
            f"{l} {np.median(st['seg_vol'][k]):.0f}" for k, l in enumerate(labels))
        print(ret_line)
        print(vol_line)
        d5 = "  ".join(f"D-{5-d} {np.median(st['d5'][d]):+.2f}%/vr{np.median(st['d5vr'][d]):.2f}"
                       for d in range(5))
        print(f"  最后5日(涨跌幅/量比中位): {d5}")
        print(f"  启动日: 量比(vs前5日) {np.median(st['day1_vr']):.2f} | "
              f"振幅 {np.median(st['day1_amp']):.2f}%")
        print()

    # 末期杀跌占比：最后30天跌幅 / 前120天总跌幅
    print("--- 末期杀跌占比（|最后30天跌幅| / |前120天总跌幅| 中位）---")
    for name in all_stats:
        st = all_stats[name]
        if not st["pre_ret"]:
            continue
        ratios = []
        for r_last, r_all in zip(st["seg_ret"][3], st["pre_ret"]):
            if r_all < 0:
                ratios.append(abs(r_last) / abs(r_all))
        if ratios:
            print(f"  {name}: {np.median(ratios)*100:.0f}%")


if __name__ == "__main__":
    main()
