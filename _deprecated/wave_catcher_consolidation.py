#!/usr/bin/env python3
"""
长期横盘捕捉器

扫描全市场个股，识别"长期（60 天以上）横盘且价格波动在均价正负 10% 以内"的区间。

横盘区间定义（滑动贪婪扩展）：
  - 以某一天为起点，逐日向后扩展窗口；
  - 窗口均价 avg = mean(close[窗口])；
  - 约束：窗口内最高价 high ≤ avg × (1 + band_pct%)，
          最低价 low  ≥ avg × (1 - band_pct%)；
  - 一旦某天突破约束即停止扩展，得到该起点的最长横盘区间；
  - 若区间天数 ≥ min_days，记录该区间并跳过，避免重复计数。

与已有脚本的关系：
  - consolidation_scanner.py：400-600 天、振幅 ≤50% 的超长期宽幅横盘；
  - find_flat_zone_after_a_main_dropping.py：≥15 天、波幅 ≤5% 且必须紧跟"加速下跌"；
  - 本脚本：≥60 天、均价 ±10% 的通用中长期窄幅横盘，不限定前置形态。

用法:
    python3 wave_catcher_consolidation.py
    python3 wave_catcher_consolidation.py --min-days 60 --band-pct 10
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_DIR = Path.home() / "Documents" / "mainland_data_2014"


def load_data(filepath: Path, min_days: int = 200) -> pd.DataFrame | None:
    """加载个股日线数据"""
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < min_days:
            return None
        return df
    except Exception:
        return None


def detect_consolidation_zones(df: pd.DataFrame, code: str,
                               min_days: int = 60,
                               band_pct: float = 10.0) -> list[dict]:
    """
    识别"长期横盘"区间。

    对每个起点 i，向后贪婪扩展窗口，只要窗口内：
      - 最高价 ≤ 窗口均价 × (1 + band_pct%)
      - 最低价 ≥ 窗口均价 × (1 - band_pct%)
    就继续纳入，否则停止，得到该起点的最长横盘区间。
    区间天数 ≥ min_days 则记录，并跳过整个区间避免重复。
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    dates = df['date'].values
    n = len(close)

    zones = []
    i = 0
    while i < n:
        # 从 i 起贪婪扩展：增量维护区间内最高/最低价与累加和
        hi = high[i]
        lo = low[i]
        s = 0.0
        cnt = 0
        end = i
        for k in range(i, n):
            cnt += 1
            s += close[k]
            if high[k] > hi:
                hi = high[k]
            if low[k] < lo:
                lo = low[k]
            avg = s / cnt
            if avg <= 0:
                break
            up = (hi - avg) / avg * 100
            down = (avg - lo) / avg * 100
            if up > band_pct or down > band_pct:
                break
            end = k

        days = end - i + 1
        if days >= min_days:
            seg_close = close[i:end + 1]
            seg_high = high[i:end + 1]
            seg_low = low[i:end + 1]
            avg = float(np.mean(seg_close))
            hi = float(np.max(seg_high))
            lo = float(np.min(seg_low))
            zones.append({
                'code': code,
                'start_date': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
                'end_date': pd.Timestamp(dates[end]).strftime('%Y-%m-%d'),
                'duration_days': days,
                'avg_price': round(avg, 2),
                'high_price': round(hi, 2),
                'low_price': round(lo, 2),
                'up_pct': round((hi - avg) / avg * 100, 2),
                'down_pct': round((avg - lo) / avg * 100, 2),
                'range_pct': round((hi - lo) / avg * 100, 2),
                'start_price': round(float(close[i]), 2),
                'end_price': round(float(close[end]), 2),
                'trend_pct': round((close[end] / close[i] - 1) * 100, 2) if close[i] > 0 else 0.0,
            })
            i = end + 1
        else:
            i += 1

    return zones


def main():
    min_days = 60
    band_pct = 10.0

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--min-days' and i + 1 < len(argv):
            min_days = int(argv[i + 1]); i += 2
        elif argv[i] == '--band-pct' and i + 1 < len(argv):
            band_pct = float(argv[i + 1]); i += 2
        else:
            i += 1

    files = sorted(DATA_DIR.glob("*.csv"))
    total = len(files)

    print("=" * 72)
    print(f"  长期横盘捕捉器")
    print(f"  条件: 持续 ≥ {min_days} 天 | 均价 ±{band_pct:.0f}% 波动")
    print(f"  (区间最高价 ≤ 均价×{1 + band_pct / 100:.3f}，最低价 ≥ 均价×{1 - band_pct / 100:.3f})")
    print("=" * 72)
    print(f"\n  共 {total} 只个股待扫描\n")

    min_data_days = min_days + 30
    all_zones = []
    scanned = 0
    skipped = 0

    for f in files:
        code = f.stem
        scanned += 1
        df = load_data(f, min_data_days)
        if df is None:
            skipped += 1
            continue

        zones = detect_consolidation_zones(df, code, min_days, band_pct)
        all_zones.extend(zones)

        if scanned % 500 == 0:
            print(f"  进度: {scanned}/{total} | 已发现 {len(all_zones)} 个横盘区间")

    print(f"  完成: {scanned}/{total} 扫描 | {skipped} 跳过 | "
          f"共发现 {len(all_zones)} 个横盘区间")

    if not all_zones:
        print("\n  未发现任何符合条件的横盘区间。")
        return

    results_df = pd.DataFrame(all_zones)
    results_df = results_df.sort_values('end_date', ascending=False).reset_index(drop=True)

    # 概览统计
    print(f"\n  --- 横盘区间概览 ---")
    print(f"  区间总数: {len(results_df)} | 涉及股票: {results_df['code'].nunique()} 只")
    print(f"  持续天数: P50={results_df['duration_days'].median():.0f}  "
          f"P75={results_df['duration_days'].quantile(0.75):.0f}  "
          f"P90={results_df['duration_days'].quantile(0.90):.0f}  "
          f"最长={results_df['duration_days'].max():.0f}")
    print(f"  总振幅:   P50={results_df['range_pct'].median():.2f}%  "
          f"P75={results_df['range_pct'].quantile(0.75):.2f}%  "
          f"P90={results_df['range_pct'].quantile(0.90):.2f}%")
    print(f"  上偏幅度: P50={results_df['up_pct'].median():.2f}%  "
          f"P90={results_df['up_pct'].quantile(0.90):.2f}%")
    print(f"  下偏幅度: P50={results_df['down_pct'].median():.2f}%  "
          f"P90={results_df['down_pct'].quantile(0.90):.2f}%")
    print(f"  起终点涨跌: P50={results_df['trend_pct'].median():.2f}%  "
          f"P25={results_df['trend_pct'].quantile(0.25):.2f}%  "
          f"P75={results_df['trend_pct'].quantile(0.75):.2f}%")

    # 输出表（最近 50 个）
    print(f"\n{'=' * 116}")
    print(f"{'#':<5} {'代码':<9} {'开始':<12} {'结束':<12} {'天数':>5} {'均价':>8} "
          f"{'最高':>8} {'最低':>8} {'上偏%':>7} {'下偏%':>7} {'振幅%':>7} {'趋势%':>7}")
    print("-" * 116)
    for idx, row in results_df.head(50).iterrows():
        print(f"{idx+1:<5} {row['code']:<9} {row['start_date']:<12} {row['end_date']:<12} "
              f"{row['duration_days']:>5.0f} {row['avg_price']:>8.2f} {row['high_price']:>8.2f} "
              f"{row['low_price']:>8.2f} {row['up_pct']:>6.2f}% {row['down_pct']:>6.2f}% "
              f"{row['range_pct']:>6.2f}% {row['trend_pct']:>6.2f}%")

    if len(results_df) > 50:
        print(f"\n  ... 共 {len(results_df)} 个区间，仅显示最近 50 个")

    output_path = Path(__file__).parent / "wave_catcher_consolidation.csv"
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
