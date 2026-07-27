#!/usr/bin/env python3
"""
加速下跌后的横盘平台扫描器

采用项目中已论证的"加速下跌（末期杀跌 / capitulation）"波段逻辑
（见 wave_catcher.py 的 oversold 模式与 discovers/discover.md 结论：
 跌幅越集中在末期，后续反转力度越强），寻找符合以下形态的波段：

  1. 加速下跌段：
     - 前 lookback_decline 天（默认 40）的平均每日跌速为 A
     - 最近 lookback_accel 天（默认 20）的平均每日跌速为 B
     - 当 A、B 均为负，且 B < A、B / A > accel_threshold（默认 1.5）时，
       判定为"加速下跌"——即最近的下跌明显快于前期（末期杀跌）。
  2. 横盘段：加速下跌结束后，紧接着出现一个持续 > 15 日的横盘区间，
     区间波幅（(最高价 - 最低价) / 区间均价）在 5% 以内。

对每只股票，找到不重叠的满足上述形态的横盘平台，记录加速下跌信息、
横盘信息、横盘均价（后续作为买点）。

用法:
    python3 find_flat_zone_after_a_main_dropping.py
    python3 find_flat_zone_after_a_main_dropping.py --lookback-decline 40 \
        --lookback-accel 20 --accel-threshold 1.5 --min-flat-days 15 --flat-range 5
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


def detect_flat_after_accel_drop(df: pd.DataFrame, code: str,
                                 lookback_decline: int = 40,
                                 lookback_accel: int = 20,
                                 accel_threshold: float = 1.5,
                                 min_flat_days: int = 15,
                                 flat_range_pct_max: float = 5.0) -> list[dict]:
    """
    识别"加速下跌之后的横盘平台"。

    加速下跌判定（与 wave_catcher.oversold 一致）：
      - 前期窗口 [i-decline-accel, i-accel) 的平均日收益 avg_prev
      - 最近窗口 [i-accel, i) 的平均日收益 avg_recent
      - avg_prev < -0.001 且 avg_recent < avg_prev 且 avg_recent/avg_prev > accel_threshold

    参数：
      lookback_decline:    前期窗口（交易日），默认 40
      lookback_accel:      最近窗口（交易日），默认 20
      accel_threshold:     加速倍数阈值，默认 1.5
      min_flat_days:       横盘区间最小天数
      flat_range_pct_max:  横盘区间最大波幅（%）
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    dates = df['date'].values
    n = len(close)

    total_lookback = lookback_decline + lookback_accel
    results = []

    i = total_lookback
    while i < n:
        # 前期 / 最近窗口（截止到横盘起点 i 之前）
        idx_prev_start = i - total_lookback
        idx_prev_end = i - lookback_accel
        idx_recent_start = i - lookback_accel
        idx_recent_end = i

        prev_seg = close[idx_prev_start:idx_prev_end]
        recent_seg = close[idx_recent_start:idx_recent_end]
        if len(prev_seg) < 2 or len(recent_seg) < 2:
            i += 1
            continue

        avg_prev_ret = float(np.mean(np.diff(prev_seg) / prev_seg[:-1]))
        avg_recent_ret = float(np.mean(np.diff(recent_seg) / recent_seg[:-1]))

        # 加速下跌条件
        if not (avg_prev_ret < -0.001 and avg_recent_ret < avg_prev_ret
                and avg_prev_ret != 0
                and avg_recent_ret / avg_prev_ret > accel_threshold):
            i += 1
            continue

        # 加速下跌成立：记录跌幅（从下跌段高点到横盘起点）
        window_close = close[idx_prev_start:i]
        window_high = float(np.max(window_close)) if len(window_close) > 0 else close[i]
        decline_pct = (window_high - close[i]) / window_high * 100 if window_high > 0 else 0.0

        # 从 i 开始贪婪扩展横盘区间，波幅需 <= flat_range_pct_max
        end = i  # 横盘最后一天（含），初始仅 i 一天
        k = i + 1
        while k < n:
            seg_high = high[i:k + 1]
            seg_low = low[i:k + 1]
            seg_close = close[i:k + 1]
            mean_close = float(np.mean(seg_close))
            if mean_close <= 0:
                break
            rng = (float(np.max(seg_high)) - float(np.min(seg_low))) / mean_close * 100
            if rng > flat_range_pct_max:
                break
            end = k
            k += 1

        flat_days = end - i + 1
        if flat_days >= min_flat_days:
            seg_close = close[i:end + 1]
            seg_high = high[i:end + 1]
            seg_low = low[i:end + 1]
            avg_price = float(np.mean(seg_close))
            flat_range = (float(np.max(seg_high)) - float(np.min(seg_low))) / avg_price * 100

            results.append({
                'code': code,
                'accel_start_date': pd.Timestamp(dates[idx_prev_start]).strftime('%Y-%m-%d'),
                'drop_high_price': round(window_high, 2),
                'decline_pct': round(decline_pct, 1),
                'avg_daily_ret_before': round(avg_prev_ret * 100, 3),
                'avg_daily_ret_accel': round(avg_recent_ret * 100, 3),
                'accel_ratio': round(avg_recent_ret / avg_prev_ret, 2),
                'flat_start_date': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
                'flat_end_date': pd.Timestamp(dates[end]).strftime('%Y-%m-%d'),
                'flat_days': flat_days,
                'flat_avg_price': round(avg_price, 2),
                'flat_range_pct': round(flat_range, 2),
            })
            # 跳过整个横盘区间，避免重复
            i = end + 1
        else:
            i += 1

    return results


def main():
    lookback_decline = 40
    lookback_accel = 20
    accel_threshold = 1.5
    min_flat_days = 15
    flat_range_pct_max = 5.0

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--lookback-decline' and i + 1 < len(argv):
            lookback_decline = int(argv[i + 1]); i += 2
        elif argv[i] == '--lookback-accel' and i + 1 < len(argv):
            lookback_accel = int(argv[i + 1]); i += 2
        elif argv[i] == '--accel-threshold' and i + 1 < len(argv):
            accel_threshold = float(argv[i + 1]); i += 2
        elif argv[i] == '--min-flat-days' and i + 1 < len(argv):
            min_flat_days = int(argv[i + 1]); i += 2
        elif argv[i] == '--flat-range' and i + 1 < len(argv):
            flat_range_pct_max = float(argv[i + 1]); i += 2
        else:
            i += 1

    files = sorted(DATA_DIR.glob("*.csv"))
    total = len(files)

    print("=" * 72)
    print(f"  加速下跌后横盘平台扫描器")
    print(f"  加速下跌: 最近{lookback_accel}天跌速 > 前{lookback_decline}天 × {accel_threshold}倍")
    print(f"  横盘:     持续≥{min_flat_days}天 | 波幅≤{flat_range_pct_max:.0f}%")
    print("=" * 72)
    print(f"\n  共 {total} 只个股待扫描\n")

    min_data_days = lookback_decline + lookback_accel + min_flat_days + 30

    all_results = []
    scanned = 0
    skipped = 0

    for f in files:
        code = f.stem
        scanned += 1
        df = load_data(f, min_data_days)
        if df is None:
            skipped += 1
            continue

        res = detect_flat_after_accel_drop(df, code, lookback_decline, lookback_accel,
                                           accel_threshold, min_flat_days, flat_range_pct_max)
        all_results.extend(res)

        if scanned % 500 == 0:
            print(f"  进度: {scanned}/{total} | 已发现 {len(all_results)} 个横盘平台")

    print(f"  完成: {scanned}/{total} 扫描 | {skipped} 跳过 | "
          f"共发现 {len(all_results)} 个横盘平台")

    if not all_results:
        print("\n  未发现任何符合条件的横盘平台。")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('flat_start_date', ascending=False).reset_index(drop=True)

    # 统计
    print(f"\n  --- 形态概览 ---")
    print(f"  加速前跌幅: P25={results_df['decline_pct'].quantile(0.25):.1f}%  "
          f"P50={results_df['decline_pct'].median():.1f}%  "
          f"P75={results_df['decline_pct'].quantile(0.75):.1f}%")
    print(f"  加速倍数:   P50={results_df['accel_ratio'].median():.2f}  "
          f"P75={results_df['accel_ratio'].quantile(0.75):.2f}  "
          f"最大={results_df['accel_ratio'].max():.2f}")
    print(f"  前期跌速:   均值={results_df['avg_daily_ret_before'].mean():.3f}%/天")
    print(f"  加速跌速:   均值={results_df['avg_daily_ret_accel'].mean():.3f}%/天")
    print(f"  横盘天数:   P50={results_df['flat_days'].median():.0f}  "
          f"P75={results_df['flat_days'].quantile(0.75):.0f}  "
          f"最长={results_df['flat_days'].max():.0f}")
    print(f"  横盘波幅:   P50={results_df['flat_range_pct'].median():.2f}%  "
          f"P75={results_df['flat_range_pct'].quantile(0.75):.2f}%")

    # 输出表（最近 50 个）
    print(f"\n{'=' * 116}")
    print(f"{'#':<5} {'代码':<10} {'加速起点':<12} {'跌幅':>7} {'加速比':>7} "
          f"{'加速跌速':>9} {'横盘起':<12} {'横盘止':<12} {'天数':>5} {'均价':>8} {'波幅':>7}")
    print("-" * 116)
    for idx, row in results_df.head(50).iterrows():
        print(f"{idx+1:<5} {row['code']:<10} {row['accel_start_date']:<12} "
              f"{row['decline_pct']:>6.1f}% {row['accel_ratio']:>7.2f} "
              f"{row['avg_daily_ret_accel']:>8.3f}% "
              f"{row['flat_start_date']:<12} {row['flat_end_date']:<12} "
              f"{row['flat_days']:>5.0f} {row['flat_avg_price']:>8.2f} {row['flat_range_pct']:>6.2f}%")

    if len(results_df) > 50:
        print(f"\n  ... 共 {len(results_df)} 个平台，仅显示最近 50 个")

    output_path = Path(__file__).parent / "flat_zone_after_drop.csv"
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
