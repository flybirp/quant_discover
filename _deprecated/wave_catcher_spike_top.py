#!/usr/bin/env python3
"""
尖顶形态捕捉器（翻倍 → 单顶 → 腰斩）

扫描全市场个股，识别三段式"倒 V / 尖顶"形态：

  1. 首段：从一个局部最低点 x 出发，上涨到最高点 max_x ≥ 2x（翻倍）。
  2. 中段：周线级别只有一个顶点 —— 从低点到腰斩点的整个区间内，
     周 K 只出现一个峰，没有第二个接近顶部的峰（排除双顶 / M 顶）。
  3. 尾段：从最高点 max_x 严格单边回落，收盘价跌破 max_x / 2（腰斩）。

判定细节：
  - 起点 x：收盘价是前后各 local_n 天内的最低价（局部底部）。
  - 顶点 max_x：从 x 上涨创新高后，自最高点回落 ≥ peak_confirm_pct% 确认见顶，
    该最高点即顶点（要求 max_x ≥ mult × x）。
  - 严格单边：从顶点到腰斩点途中，任何反弹幅度 ≤ rebound_pct%，否则不认定。
  - 腰斩点：顶点之后首个收盘价 ≤ max_x / 2 的交易日。
  - 单顶：将区间按每 5 个交易日聚合为"周 K"，除主峰外，其余局部峰的周高
    必须 ≤ 主峰 × peak_ratio（默认 0.9），否则视为有二次冲高。

用法:
    python3 wave_catcher_spike_top.py
    python3 wave_catcher_spike_top.py --local-n 30 --mult 2.0 \
        --peak-confirm-pct 20 --rebound-pct 20 --peak-ratio 0.9
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


def find_local_lows(close: np.ndarray, local_n: int) -> list[int]:
    """找出所有局部最低点：收盘价是前后各 local_n 天内的最低价。"""
    n = len(close)
    s = pd.Series(close)
    left_min = s.rolling(window=local_n + 1, min_periods=1).min()
    right_min = s.iloc[::-1].rolling(window=local_n + 1, min_periods=1).min().iloc[::-1]
    is_low = (s == left_min) & (s == right_min)
    # 对连续相等的低点平台只保留第一个，避免相邻同值低点重复计数
    lows = []
    prev = False
    for idx in range(n):
        cur = bool(is_low.iloc[idx])
        if cur and not prev:
            lows.append(idx)
        prev = cur
    return lows


def strictly_declining(close: np.ndarray, m: int, j: int, rebound_pct: float) -> bool:
    """从顶点 m 到腰斩点 j，途中任何反弹幅度不得超过 rebound_pct%。"""
    running_low = close[m]
    for k in range(m + 1, j + 1):
        if close[k] < running_low:
            running_low = close[k]
        else:
            rebound = (close[k] - running_low) / running_low * 100
            if rebound > rebound_pct:
                return False
    return True


def single_weekly_peak(high: np.ndarray, i: int, j: int, peak_ratio: float) -> bool:
    """
    判断区间 [i, j] 在周线级别是否只有一个顶点（无二次冲高）。

    将区间按每 5 个交易日聚合为"周"，取每周最高价作为周高；
    找出所有局部峰（严格高于前后周），除主峰外其余峰必须 ≤ 主峰 × peak_ratio。
    """
    seg = high[i:j + 1]
    weekly = [float(seg[s:s + 5].max()) for s in range(0, len(seg), 5)]
    L = len(weekly)
    if L < 3:
        # 区间过短，不足以形成双顶，视为通过
        return True

    main_peak = max(weekly)
    if main_peak <= 0:
        return False

    for t in range(1, L - 1):
        # 局部峰
        if weekly[t] > weekly[t - 1] and weekly[t] > weekly[t + 1]:
            if weekly[t] != main_peak and weekly[t] > main_peak * peak_ratio:
                return False
    return True


def find_spike_patterns(df: pd.DataFrame, code: str,
                        local_n: int = 30,
                        mult: float = 2.0,
                        peak_confirm_pct: float = 20.0,
                        rebound_pct: float = 20.0,
                        peak_ratio: float = 0.9,
                        max_up_days: int = 500) -> list[dict]:
    """
    识别"翻倍 → 单顶 → 腰斩"尖顶形态。
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    dates = df['date'].values
    n = len(close)

    patterns = []
    low_indices = find_local_lows(close, local_n)

    for i in low_indices:
        x = close[i]
        if x <= 0:
            continue

        # ---- 首段：上涨找顶点，回落确认见顶 ----
        max_x = x
        m = i
        topped = False
        upper = min(n, i + max_up_days + 1)
        for k in range(i + 1, upper):
            if close[k] > max_x:
                max_x = close[k]
                m = k
            # 翻倍之后才允许见顶确认
            if max_x >= mult * x and (max_x - close[k]) / max_x * 100 >= peak_confirm_pct:
                topped = True
                break

        if not topped or max_x < mult * x:
            continue

        # ---- 尾段：顶点之后找首个收盘价 <= max_x/2 ----
        j = -1
        for k in range(m + 1, n):
            if close[k] <= max_x / 2:
                j = k
                break
        if j == -1:
            continue

        # ---- 严格单边下跌验证 ----
        if not strictly_declining(close, m, j, rebound_pct):
            continue

        # ---- 周线单顶验证（全形态区间 [i, j]）----
        if not single_weekly_peak(high, i, j, peak_ratio):
            continue

        up_days = m - i
        down_days = j - m
        patterns.append({
            'code': code,
            'low_date': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
            'low_price': round(float(x), 2),
            'peak_date': pd.Timestamp(dates[m]).strftime('%Y-%m-%d'),
            'peak_price': round(float(max_x), 2),
            'half_date': pd.Timestamp(dates[j]).strftime('%Y-%m-%d'),
            'half_price': round(float(close[j]), 2),
            'up_pct': round((max_x / x - 1) * 100, 2),
            'down_pct': round((close[j] / max_x - 1) * 100, 2),
            'up_days': up_days,
            'down_days': down_days,
            'total_days': up_days + down_days,
        })

    return patterns


def main():
    local_n = 30
    mult = 2.0
    peak_confirm_pct = 20.0
    rebound_pct = 20.0
    peak_ratio = 0.9
    max_up_days = 500

    argv = sys.argv[1:]
    a = 0
    while a < len(argv):
        if argv[a] == '--local-n' and a + 1 < len(argv):
            local_n = int(argv[a + 1]); a += 2
        elif argv[a] == '--mult' and a + 1 < len(argv):
            mult = float(argv[a + 1]); a += 2
        elif argv[a] == '--peak-confirm-pct' and a + 1 < len(argv):
            peak_confirm_pct = float(argv[a + 1]); a += 2
        elif argv[a] == '--rebound-pct' and a + 1 < len(argv):
            rebound_pct = float(argv[a + 1]); a += 2
        elif argv[a] == '--peak-ratio' and a + 1 < len(argv):
            peak_ratio = float(argv[a + 1]); a += 2
        elif argv[a] == '--max-up-days' and a + 1 < len(argv):
            max_up_days = int(argv[a + 1]); a += 2
        else:
            a += 1

    files = sorted(DATA_DIR.glob("*.csv"))
    total = len(files)

    print("=" * 72)
    print("  尖顶形态捕捉器（翻倍 → 单顶 → 腰斩）")
    print(f"  起点: 局部低点(前后各{local_n}天) | 翻倍: ≥{mult:.1f}x | 见顶确认回落: ≥{peak_confirm_pct:.0f}%")
    print(f"  单顶: 次峰≤主峰×{peak_ratio:.2f} | 尾段反弹上限: {rebound_pct:.0f}% | 腰斩: 收盘≤顶点/2")
    print("=" * 72)
    print(f"\n  共 {total} 只个股待扫描\n")

    all_patterns = []
    scanned = 0
    skipped = 0

    for f in files:
        code = f.stem
        scanned += 1
        df = load_data(f, local_n * 2 + 50)
        if df is None:
            skipped += 1
            continue

        patterns = find_spike_patterns(
            df, code, local_n, mult, peak_confirm_pct, rebound_pct, peak_ratio, max_up_days
        )
        all_patterns.extend(patterns)

        if scanned % 500 == 0:
            print(f"  进度: {scanned}/{total} | 已发现 {len(all_patterns)} 个形态")

    print(f"  完成: {scanned}/{total} 扫描 | {skipped} 跳过 | 共发现 {len(all_patterns)} 个形态(去重前)")

    # 去重：同一只股票、同一"顶点→腰斩"事件可能被多个历史局部低点共享，
    # 只保留起点最早（低点最低、翻倍最完整）的一条，避免后续分析重复计数。
    best = {}
    for p in all_patterns:
        key = (p['code'], p['peak_date'], p['half_date'])
        if key not in best or p['low_date'] < best[key]['low_date']:
            best[key] = p
    all_patterns = list(best.values())
    print(f"  去重后: {len(all_patterns)} 个形态")

    if not all_patterns:
        print("\n  未发现任何符合条件的尖顶形态。")
        return

    res = pd.DataFrame(all_patterns)
    res = res.sort_values('peak_date', ascending=False).reset_index(drop=True)

    print(f"\n  --- 形态概览 ---")
    print(f"  形态总数: {len(res)} | 涉及股票: {res['code'].nunique()} 只")
    print(f"  首段涨幅: 中位 {res['up_pct'].median():.0f}% | P25 {res['up_pct'].quantile(0.25):.0f}% | P75 {res['up_pct'].quantile(0.75):.0f}%")
    print(f"  首段天数: 中位 {res['up_days'].median():.0f} | P75 {res['up_days'].quantile(0.75):.0f}")
    print(f"  尾段天数: 中位 {res['down_days'].median():.0f} | P75 {res['down_days'].quantile(0.75):.0f}")
    print(f"  全程天数: 中位 {res['total_days'].median():.0f} | P75 {res['total_days'].quantile(0.75):.0f}")

    print(f"\n{'=' * 122}")
    print(f"{'#':<5} {'代码':<9} {'低点日':<12} {'低点价':>9} {'顶点日':<12} {'顶点价':>9} "
          f"{'腰斩日':<12} {'腰斩价':>9} {'首段涨幅':>9} {'首段天':>6} {'尾段天':>6}")
    print("-" * 122)
    for idx, row in res.head(50).iterrows():
        print(f"{idx+1:<5} {row['code']:<9} {row['low_date']:<12} {row['low_price']:>9.2f} "
              f"{row['peak_date']:<12} {row['peak_price']:>9.2f} {row['half_date']:<12} "
              f"{row['half_price']:>9.2f} {row['up_pct']:>8.1f}% {row['up_days']:>6.0f} {row['down_days']:>6.0f}")

    if len(res) > 50:
        print(f"\n  ... 共 {len(res)} 个形态，仅显示最近 50 个")

    output_path = Path(__file__).parent / "wave_catcher_spike_top.csv"
    res.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
