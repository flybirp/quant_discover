#!/usr/bin/env python3
"""
主升浪波段扫描器

通过技术指标 + 状态逻辑，锁定市场中符合"主升浪"情形的个股及其时间段。

两种模式：
  drawdown (默认) — 纯价格状态机：持续上涨过程中 max drawdown ≤ 50% 的连续波段
  trendline       — 趋势线模式：沿趋势线运行，不连续2日跌破，涨幅>100%，>200天

用法:
    python3 find_main_waves.py                       # 默认 drawdown 模式
    python3 find_main_waves.py --max-dd 30           # 自定义回撤阈值
    python3 find_main_waves.py --mode trendline       # 切换到趋势线模式
    python3 find_main_waves.py --mode trendline --indicator ma60
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from indicator import sma

DATA_DIR = Path.home() / "Documents" / "mainland_data_2014"


# ============================================================
# 公共
# ============================================================

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


# ============================================================
# 模式 1: 回撤状态机 (默认)
# ============================================================

def find_uptrend_waves(df: pd.DataFrame, code: str,
                       max_drawdown_pct: float = 50.0,
                       min_days: int = 50,
                       min_return_pct: float = 50.0) -> list[dict]:
    """
    找出所有"持续上涨"波段 — 回撤状态机算法。

    定义：
      从一个绝对低点开始，股价持续走高，途中从任何局部高点的回撤
      都不超过 max_drawdown_pct，直到某次回撤突破阈值，波段结束。

    算法：
      维护 running_peak（历史最高价）。每次价格创新高，更新 peak。
      实时计算 drawdown = (peak - price) / peak。
      一旦 drawdown > max_drawdown_pct → 波段结束，从该点之后重新开始。

    参数：
      max_drawdown_pct: 波段内允许的最大回撤百分比 (默认 50)
      min_days:         波段最短持续天数 (过滤噪声)
      min_return_pct:   波段最低涨幅百分比 (过滤无意义的小波段)
    """
    close = df['close'].values
    dates = df['date'].values
    n = len(close)

    if n < min_days:
        return []

    waves = []
    # 波段起点：第一个数据点（或上一个波段结束后的新低点）
    wave_start_idx = 0
    running_peak = close[0]
    running_peak_idx = 0

    for i in range(1, n):
        price = close[i]

        if price > running_peak:
            # 创新高，更新峰值
            running_peak = price
            running_peak_idx = i
        else:
            # 计算当前回撤
            drawdown_pct = (running_peak - price) / running_peak * 100

            if drawdown_pct > max_drawdown_pct:
                # 回撤超阈值 → 波段结束
                # 波段的绝对低点 = wave_start_idx 到当前之间的最低价
                seg_close = close[wave_start_idx:i]
                abs_low_idx_local = np.argmin(seg_close)
                abs_low_idx = wave_start_idx + abs_low_idx_local
                abs_low_price = close[abs_low_idx]

                duration = i - 1 - abs_low_idx + 1  # 到突破阈值前一天
                wave_end_idx = i - 1

                if duration >= min_days:
                    ret = (close[wave_end_idx] / abs_low_price - 1) * 100
                    if ret >= min_return_pct:
                        # 波段内实际峰值和回撤
                        seg_peak = float(np.max(close[abs_low_idx:wave_end_idx + 1]))
                        seg_peak_idx = abs_low_idx + np.argmax(close[abs_low_idx:wave_end_idx + 1])
                        # 从波段峰值到波段终点之间的最大回撤
                        after_seg_peak = close[seg_peak_idx:wave_end_idx + 1]
                        seg_min_after_peak = float(np.min(after_seg_peak)) if len(after_seg_peak) > 0 else seg_peak
                        actual_dd = (seg_peak - seg_min_after_peak) / seg_peak * 100

                        waves.append({
                            'code': code,
                            'start_date': pd.Timestamp(dates[abs_low_idx]).strftime('%Y-%m-%d'),
                            'end_date': pd.Timestamp(dates[wave_end_idx]).strftime('%Y-%m-%d'),
                            'duration_days': duration,
                            'wave_start_close': round(float(abs_low_price), 2),
                            'peak_close': round(seg_peak, 2),
                            'wave_end_close': round(float(close[wave_end_idx]), 2),
                            'return_pct': round(ret, 2),
                            'max_drawdown_pct': round(actual_dd, 1),
                        })

                # 重置：从当前位置开始寻找下一个波段
                wave_start_idx = i
                running_peak = price
                running_peak_idx = i

    # 处理最后一个波段（如果到数据末尾还没触发回撤阈值）
    if wave_start_idx < n - 1:
        seg_close = close[wave_start_idx:]
        abs_low_idx_local = np.argmin(seg_close)
        abs_low_idx = wave_start_idx + abs_low_idx_local
        abs_low_price = close[abs_low_idx]
        duration = n - 1 - abs_low_idx + 1

        if duration >= min_days:
            ret = (close[n - 1] / abs_low_price - 1) * 100
            if ret >= min_return_pct:
                # 波段内实际峰值（从 abs_low_idx 到末尾）
                wave_seg = close[abs_low_idx:n]
                wave_peak_val = float(np.max(wave_seg))
                wave_peak_idx = abs_low_idx + int(np.argmax(wave_seg))
                # 从峰值到末尾的最大回撤
                after_wave_peak = close[wave_peak_idx:n]
                min_after_peak = float(np.min(after_wave_peak)) if len(after_wave_peak) > 0 else wave_peak_val
                actual_max_dd = (wave_peak_val - min_after_peak) / wave_peak_val * 100

                waves.append({
                    'code': code,
                    'start_date': pd.Timestamp(dates[abs_low_idx]).strftime('%Y-%m-%d'),
                    'end_date': pd.Timestamp(dates[n - 1]).strftime('%Y-%m-%d'),
                    'duration_days': duration,
                    'wave_start_close': round(float(abs_low_price), 2),
                    'peak_close': round(wave_peak_val, 2),
                    'wave_end_close': round(float(close[n - 1]), 2),
                    'return_pct': round(ret, 2),
                    'max_drawdown_pct': round(actual_max_dd, 1),
                })

    return waves


# ============================================================
# 模式 2: 趋势线 (原有逻辑)
# ============================================================

def make_indicator(name: str):
    """根据名称返回 (指标函数, 最小数据天数, 显示名)"""
    if name == 'ma60':
        def fn(close):
            return sma(close, 60)
        return fn, 60, 'MA60'
    elif name == 'zhixing_slow':
        def fn(close):
            ma14 = sma(close, 14)
            ma28 = sma(close, 28)
            ma57 = sma(close, 57)
            ma114 = sma(close, 114)
            return (ma14 + ma28 + ma57 + ma114) / 4
        return fn, 114, 'zhixing_slow'
    else:
        raise ValueError(f"未知指标: {name}")


def find_trendline_waves(df: pd.DataFrame, code: str, indicator_fn) -> list[dict]:
    """
    趋势线模式：沿趋势线运行，不连续2日跌破，涨幅>100%，>200天。
    """
    close = df['close'].values
    dates = df['date'].values
    line = indicator_fn(df['close']).values

    first_valid = 0
    while first_valid < len(line) and pd.isna(line[first_valid]):
        first_valid += 1
    if first_valid >= len(line):
        return []

    # 找清洁段
    clean_segments = []
    in_segment = False
    seg_start = None
    consecutive_below = 0

    for i in range(first_valid, len(close)):
        below = close[i] < line[i]
        if below:
            consecutive_below += 1
        else:
            consecutive_below = 0

        if not in_segment:
            if not below:
                in_segment = True
                seg_start = i
        else:
            if consecutive_below >= 2:
                in_segment = False
                clean_segments.append((seg_start, i - 1))

    if in_segment:
        clean_segments.append((seg_start, len(close) - 1))

    if not clean_segments:
        return []

    all_waves = []
    for seg_start, seg_end in clean_segments:
        seg_duration = seg_end - seg_start + 1
        if seg_duration <= 200:
            continue

        local_close = close[seg_start:seg_end + 1]
        local_dates = dates[seg_start:seg_end + 1]
        local_line = line[seg_start:seg_end + 1]
        seg_len = len(local_close)

        running_max = np.zeros(seg_len)
        running_max_idx = np.zeros(seg_len, dtype=int)
        running_max[-1] = local_close[-1]
        running_max_idx[-1] = seg_len - 1
        for j in range(seg_len - 2, -1, -1):
            if local_close[j] > running_max[j + 1]:
                running_max[j] = local_close[j]
                running_max_idx[j] = j
            else:
                running_max[j] = running_max[j + 1]
                running_max_idx[j] = running_max_idx[j + 1]

        best = None
        best_return = 0

        for i in range(seg_len - 200):
            if local_close[i] < local_line[i]:
                continue
            start_val = local_close[i]
            min_end = i + 200
            if min_end >= seg_len:
                break

            max_val_after = running_max[min_end]
            ret_pct = (max_val_after / start_val - 1) * 100

            if ret_pct > 100 and ret_pct > best_return:
                end_offset = running_max_idx[min_end]
                duration = end_offset - i + 1
                best_return = ret_pct
                best = {
                    'code': code,
                    'start_date': pd.Timestamp(local_dates[i]).strftime('%Y-%m-%d'),
                    'end_date': pd.Timestamp(local_dates[end_offset]).strftime('%Y-%m-%d'),
                    'duration_days': duration,
                    'start_close': round(float(start_val), 2),
                    'end_close': round(float(max_val_after), 2),
                    'return_pct': round(ret_pct, 2),
                }

        if best:
            all_waves.append(best)

    return all_waves


# ============================================================
# CLI
# ============================================================

def main():
    # 解析参数
    mode = 'drawdown'
    max_dd = 50.0
    min_days = 50
    min_return = 50.0
    indicator_name = 'zhixing_slow'

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--mode' and i + 1 < len(argv):
            mode = argv[i + 1]; i += 2
        elif argv[i] == '--max-dd' and i + 1 < len(argv):
            max_dd = float(argv[i + 1]); i += 2
        elif argv[i] == '--min-days' and i + 1 < len(argv):
            min_days = int(argv[i + 1]); i += 2
        elif argv[i] == '--min-return' and i + 1 < len(argv):
            min_return = float(argv[i + 1]); i += 2
        elif argv[i] == '--indicator' and i + 1 < len(argv):
            indicator_name = argv[i + 1]; i += 2
        else:
            i += 1

    files = sorted(DATA_DIR.glob("*.csv"))
    total = len(files)

    if mode == 'drawdown':
        output_name = f"main_waves_dd{int(max_dd)}.csv"

        print("=" * 72)
        print(f"  主升浪扫描器 (回撤状态机)")
        print(f"  条件: 回撤 ≤ {max_dd}% | 持续 ≥ {min_days}天 | 涨幅 ≥ {min_return}%")
        print("=" * 72)
        print(f"\n  共 {total} 只个股待扫描\n")

        all_results = []
        scanned = 0
        skipped = 0

        for f in files:
            code = f.stem
            scanned += 1
            df = load_data(f, min_days)
            if df is None:
                skipped += 1
                continue

            waves = find_uptrend_waves(df, code, max_dd, min_days, min_return)
            all_results.extend(waves)

            if scanned % 500 == 0:
                print(f"  进度: {scanned}/{total} | 已发现 {len(all_results)} 个波段")

        print(f"  完成: {scanned}/{total} 扫描 | {skipped} 跳过 | "
              f"共发现 {len(all_results)} 个波段")

        print(f"\n  --- 波段统计 ---")
        if all_results:
            rets = [w['return_pct'] for w in all_results]
            dds = [w['max_drawdown_pct'] for w in all_results]
            durs = [w['duration_days'] for w in all_results]
            print(f"  数量: {len(all_results)}")
            print(f"  涨幅: 均值 {np.mean(rets):.0f}% | 中位 {np.median(rets):.0f}% | 最大 {np.max(rets):.0f}%")
            print(f"  回撤: 均值 {np.mean(dds):.1f}% | 中位 {np.median(dds):.1f}% | 最大 {np.max(dds):.1f}%")
            print(f"  天数: 均值 {np.mean(durs):.0f} | 中位 {np.median(durs):.0f} | 最长 {np.max(durs)}")

    elif mode == 'trendline':
        indicator_fn, min_days_tl, display_name = make_indicator(indicator_name)
        output_name = f"main_waves_{indicator_name}.csv"

        print("=" * 72)
        print(f"  主升浪扫描器 (趋势线)")
        print(f"  趋势线: {display_name} | 条件: >200天 | 无连续2日跌破 | 涨幅>100%")
        print("=" * 72)
        print(f"\n  共 {total} 只个股待扫描\n")

        all_results = []
        scanned = 0
        skipped = 0

        for f in files:
            code = f.stem
            scanned += 1
            df = load_data(f, min_days_tl + 200)
            if df is None:
                skipped += 1
                continue

            waves = find_trendline_waves(df, code, indicator_fn)
            all_results.extend(waves)

            if scanned % 500 == 0:
                print(f"  进度: {scanned}/{total} | 已发现 {len(all_results)} 个波段")

        print(f"  完成: {scanned}/{total} 扫描 | {skipped} 跳过 | "
              f"共发现 {len(all_results)} 个波段")
    else:
        print(f"未知模式: {mode}，支持 drawdown / trendline")
        sys.exit(1)

    # 输出结果表
    print("\n" + "=" * 72)
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('return_pct', ascending=False)
        results_df = results_df.reset_index(drop=True)

        cols = [c for c in results_df.columns if c not in ('max_drawdown_pct',)]
        has_dd = 'max_drawdown_pct' in results_df.columns

        # 表头
        header = f"{'#':<5} {'代码':<8} {'起始日':<12} {'结束日':<12} " \
                 f"{'天数':<6} {'低点':>8} {'峰值':>8} {'终点':>8} {'涨幅':>8}"
        if has_dd:
            header += f" {'回撤':>6}"
        print(header)
        print("-" * len(header))

        for idx, row in results_df.iterrows():
            line = f"{idx+1:<5} {row['code']:<8} {row['start_date']:<12} {row['end_date']:<12} " \
                   f"{row['duration_days']:<6} {row.get('wave_start_close', row.get('start_close', 0)):>8.2f} " \
                   f"{row.get('peak_close', row.get('end_close', 0)):>8.2f} " \
                   f"{row.get('wave_end_close', row.get('end_close', 0)):>8.2f} " \
                   f"{row['return_pct']:>7.1f}%"
            if has_dd:
                line += f" {row['max_drawdown_pct']:>5.1f}%"
            print(line)

        output_path = Path(__file__).parent / output_name
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至: {output_path}")
        print(f"共 {len(results_df)} 个波段")
    else:
        print("  未发现任何符合条件的波段。")


if __name__ == "__main__":
    main()
