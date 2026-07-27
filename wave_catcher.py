#!/usr/bin/env python3
"""
波段识别器

通过技术指标信号锁定个股的上涨波段及其时间段。支持三种模式：

模式1 zfzs（默认）：
  入场：zhixing_fast 金叉 zhixing_slow
  离场：MA20 死叉 MA60

模式2 volume：
  入场：长期下跌60天且跌>20% + 放量>2倍均量 + 站上zhixing_fast
  离场：跌破入场前最低价（止损） or 涨幅达100%（止盈）

模式3 oversold：
  入场：识别加速下跌段（最近20天平均跌速 > 前40天平均跌速），触底反弹至少30%
  离场：反弹不足30% or 反弹后再跌破底部（止损）

与 find_main_waves.py 互补：
  - find_main_waves: 用回撤阈值或趋势线定义波段，侧重"事后归纳"
  - wave_catcher:   用指标信号定义波段，侧重"实时识别"

用法:
    python3 wave_catcher.py                           # 默认 zfzs 模式
    python3 wave_catcher.py --mode volume             # 放量突破模式
    python3 wave_catcher.py --mode volume --lookback 90 --decline 30
    python3 wave_catcher.py --mode oversold           # 超跌反弹模式
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from indicator import sma, zhixing_fast, zhixing_slow

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


def detect_waves(df: pd.DataFrame, code: str,
                 min_days: int = 20) -> list[dict]:
    """
    对单只股票识别波段。

    入场：zhixing_fast[t-1] <= zhixing_slow[t-1] 且 zhixing_fast[t] > zhixing_slow[t]
    离场：MA20[t-1] >= MA60[t-1] 且 MA20[t] < MA60[t]

    在入场和离场之间，记录波段的峰值和回撤。
    """
    close = df['close'].values
    dates = df['date'].values
    n = len(close)

    z_fast = zhixing_fast(df['close']).values
    z_slow = zhixing_slow(df['close']).values
    ma20 = sma(df['close'], 20).values
    ma60 = sma(df['close'], 60).values

    waves = []
    in_position = False
    entry_idx = None
    entry_price = None
    peak_price = None
    peak_idx = None

    for i in range(1, n):
        if np.isnan(z_fast[i]) or np.isnan(z_slow[i]) or np.isnan(ma20[i]) or np.isnan(ma60[i]):
            continue

        if not in_position:
            if z_fast[i - 1] <= z_slow[i - 1] and z_fast[i] > z_slow[i]:
                in_position = True
                entry_idx = i
                entry_price = close[i]
                peak_price = close[i]
                peak_idx = i
        else:
            if close[i] > peak_price:
                peak_price = close[i]
                peak_idx = i

            if ma20[i - 1] >= ma60[i - 1] and ma20[i] < ma60[i]:
                exit_idx = i
                exit_price = close[i]
                duration = exit_idx - entry_idx
                ret_pct = (exit_price / entry_price - 1) * 100

                seg = close[peak_idx:exit_idx + 1]
                if len(seg) > 0:
                    min_after_peak = float(np.min(seg))
                    max_dd = (peak_price - min_after_peak) / peak_price * 100
                else:
                    max_dd = 0.0

                if duration >= min_days:
                    waves.append({
                        'code': code,
                        'entry_date': pd.Timestamp(dates[entry_idx]).strftime('%Y-%m-%d'),
                        'exit_date': pd.Timestamp(dates[exit_idx]).strftime('%Y-%m-%d'),
                        'duration_days': duration,
                        'entry_price': round(float(entry_price), 2),
                        'peak_price': round(float(peak_price), 2),
                        'exit_price': round(float(exit_price), 2),
                        'return_pct': round(ret_pct, 2),
                        'max_drawdown_pct': round(max_dd, 1),
                    })

                in_position = False

    # 持仓中（未触发离场信号）
    if in_position:
        exit_idx = n - 1
        exit_price = close[-1]
        duration = exit_idx - entry_idx
        ret_pct = (exit_price / entry_price - 1) * 100

        seg = close[peak_idx:exit_idx + 1]
        if len(seg) > 0:
            min_after_peak = float(np.min(seg))
            max_dd = (peak_price - min_after_peak) / peak_price * 100
        else:
            max_dd = 0.0

        if duration >= min_days:
            waves.append({
                'code': code,
                'entry_date': pd.Timestamp(dates[entry_idx]).strftime('%Y-%m-%d'),
                'exit_date': pd.Timestamp(dates[exit_idx]).strftime('%Y-%m-%d'),
                'duration_days': duration,
                'entry_price': round(float(entry_price), 2),
                'peak_price': round(float(peak_price), 2),
                'exit_price': round(float(exit_price), 2),
                'return_pct': round(ret_pct, 2),
                'max_drawdown_pct': round(max_dd, 1),
                'holding': True,
            })

    return waves


def detect_volume_waves(df: pd.DataFrame, code: str,
                        lookback: int = 60,
                        decline_pct: float = 20.0,
                        vol_multi: float = 2.0,
                        min_days: int = 1) -> list[dict]:
    """
    识别"长期下跌后放量突破 zhixing_fast"波段。

    入场条件（同时满足）：
      1. 长期下跌：过去 lookback 天内收盘价有 70% 以上时间在 zhixing_slow 下方，
         且从 lookback 天最高点跌幅超过 decline_pct%
      2. 放量：当日成交量 > 20日均量 × vol_multi
      3. 突破：当日收盘价站上 zhixing_fast

    离场条件（任一触发）：
      1. 止损：收盘价跌破入场前 lookback 天内的最低价
      2. 止盈：收盘价达到入场价的 2 倍（涨幅 100%）
    """
    close = df['close'].values
    dates = df['date'].values
    low = df['low'].values
    volume = df['volume'].values
    n = len(close)

    z_fast = zhixing_fast(df['close']).values
    z_slow = zhixing_slow(df['close']).values
    vol_ma20 = sma(df['volume'], 20).values

    waves = []
    in_position = False
    entry_idx = None
    entry_price = None
    peak_price = None
    stop_price = None  # 入场前最低价（止损线）
    take_profit = None  # 入场价 × 2（止盈线）

    for i in range(lookback + 20, n):
        if np.isnan(z_fast[i]) or np.isnan(z_slow[i]) or np.isnan(vol_ma20[i]):
            continue

        if not in_position:
            # === 入场条件检测 ===

            # 条件1：长期下跌——过去lookback天大部分时间在zs下方
            window_close = close[i - lookback:i]
            window_zs = z_slow[i - lookback:i]
            if np.any(np.isnan(window_zs)):
                continue
            below_ratio = np.sum(window_close < window_zs) / len(window_close)
            if below_ratio < 0.7:
                continue

            # 条件1补充：从lookback天高点跌幅超过decline_pct%
            window_high = np.max(window_close)
            decline = (window_high - close[i]) / window_high * 100
            if decline < decline_pct:
                continue

            # 条件2：放量 > 2倍均量
            if volume[i] <= vol_ma20[i] * vol_multi:
                continue

            # 条件3：站上zhixing_fast
            if close[i] <= z_fast[i]:
                continue

            # === 入场 ===
            in_position = True
            entry_idx = i
            entry_price = close[i]
            peak_price = close[i]
            # 止损线：入场前lookback天内的最低价
            stop_price = float(np.min(low[i - lookback:i]))
            # 止盈线：入场价 × 2
            take_profit = entry_price * 2

        else:
            # 更新峰值
            if close[i] > peak_price:
                peak_price = close[i]

            exit_reason = None
            exit_price = None

            # 止损：收盘跌破入场前最低价
            if close[i] < stop_price:
                exit_reason = 'stop'
                exit_price = close[i]

            # 止盈：达到 100% 涨幅
            elif close[i] >= take_profit:
                exit_reason = 'target'
                exit_price = close[i]

            if exit_reason:
                duration = i - entry_idx
                ret_pct = (exit_price / entry_price - 1) * 100

                # 计算波段内最大回撤
                seg_close = close[entry_idx:i + 1]
                peak_idx_val = entry_idx + int(np.argmax(seg_close))
                after_peak = close[peak_idx_val:i + 1]
                min_after_peak = float(np.min(after_peak)) if len(after_peak) > 0 else exit_price
                max_dd = (peak_price - min_after_peak) / peak_price * 100

                if duration >= min_days:
                    waves.append({
                        'code': code,
                        'entry_date': pd.Timestamp(dates[entry_idx]).strftime('%Y-%m-%d'),
                        'exit_date': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
                        'duration_days': duration,
                        'entry_price': round(float(entry_price), 2),
                        'peak_price': round(float(peak_price), 2),
                        'exit_price': round(float(exit_price), 2),
                        'return_pct': round(ret_pct, 2),
                        'max_drawdown_pct': round(max_dd, 1),
                        'exit_reason': exit_reason,
                    })

                in_position = False

    # 持仓中
    if in_position:
        i = n - 1
        exit_price = close[i]
        duration = i - entry_idx
        ret_pct = (exit_price / entry_price - 1) * 100

        peak_idx_val = entry_idx + int(np.argmax(close[entry_idx:i + 1]))
        seg2 = close[peak_idx_val:i + 1]
        min_after_peak = float(np.min(seg2)) if len(seg2) > 0 else exit_price
        max_dd = (peak_price - min_after_peak) / peak_price * 100

        if duration >= min_days:
            waves.append({
                'code': code,
                'entry_date': pd.Timestamp(dates[entry_idx]).strftime('%Y-%m-%d'),
                'exit_date': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
                'duration_days': duration,
                'entry_price': round(float(entry_price), 2),
                'peak_price': round(float(peak_price), 2),
                'exit_price': round(float(exit_price), 2),
                'return_pct': round(ret_pct, 2),
                'max_drawdown_pct': round(max_dd, 1),
                'exit_reason': 'holding',
                'holding': True,
            })

    return waves


def detect_oversold_bounce_waves(df: pd.DataFrame, code: str,
                                 lookback_decline: int = 40,
                                 lookback_accel: int = 20,
                                 accel_threshold: float = 1.5,
                                 bounce_target_pct: float = 30.0) -> list[dict]:
    """
    识别"超跌反弹"形态 — 只记录数据点，不做交易决策。
    
    检测逻辑：
      1. 扫描历史数据，找出"加速下跌"阶段
         - 前 lookback_decline 天平均每日回撤 = A
         - 最近 lookback_accel 天平均每日回撤 = B
         - 当 B / A > accel_threshold 且都是负值，标记为加速下跌
      
      2. 在加速下跌段内找最低点
      
      3. 从最低点之后扫描，当反弹 >= bounce_target_pct% 时
         - 记录这个形态（高点、低点、反弹点、幅度）
    
    参数：
      lookback_decline:  前期窗口（天）
      lookback_accel:    最近窗口（天）
      accel_threshold:   加速倍数阈值
      bounce_target_pct: 反弹目标幅度(%)
    """
    close = df['close'].values
    dates = df['date'].values
    n = len(close)
    
    if n < lookback_decline + lookback_accel + 20:
        return []
    
    patterns = []
    last_processed_idx = 0  # 最后处理的底部位置，避免重复
    
    # 固定窗口扫描：每次向前移动一天
    for i in range(lookback_decline + lookback_accel, n):
        # 前期和最近的数据窗口
        idx_prev_start = i - lookback_decline - lookback_accel
        idx_prev_end = i - lookback_accel
        idx_recent_start = i - lookback_accel
        idx_recent_end = i
        
        # 计算前期平均每日回撤
        prev_returns = []
        for j in range(idx_prev_start + 1, idx_prev_end):
            daily_ret = (close[j] - close[j-1]) / close[j-1]
            prev_returns.append(daily_ret)
        avg_prev_ret = np.mean(prev_returns) if prev_returns else 0.0
        
        # 计算最近平均每日回撤
        recent_returns = []
        for j in range(idx_recent_start + 1, idx_recent_end):
            daily_ret = (close[j] - close[j-1]) / close[j-1]
            recent_returns.append(daily_ret)
        avg_recent_ret = np.mean(recent_returns) if recent_returns else 0.0
        
        # 检测加速下跌
        if (avg_prev_ret < -0.001 and avg_recent_ret < avg_prev_ret and 
            avg_prev_ret != 0 and avg_recent_ret / avg_prev_ret > accel_threshold):
            
            # 在更长的范围内找最低点（从当前窗口向后扩展）
            search_end = min(idx_recent_end + 30, n)  # 向后扩展30天
            search_start = idx_recent_start
            if search_start >= search_end:
                continue
            
            # 在扩展范围内找最低点
            bottom_idx = search_start + int(np.argmin(close[search_start:search_end]))
            
            # 只处理新的底部（避免重复）
            if bottom_idx > last_processed_idx:
                bottom_price = float(close[bottom_idx])
                
                # 从底部之后扫描反弹，最多向前120个交易日
                bounce_found = False
                for j in range(bottom_idx + 1, min(bottom_idx + 120, n)):
                    bounce_pct = (close[j] - bottom_price) / bottom_price * 100
                    
                    if bounce_pct >= bounce_target_pct:
                        # 反弹成功！找到前期高点
                        peak_idx = idx_prev_start + int(np.argmax(close[idx_prev_start:idx_prev_end]))
                        peak_price = float(close[peak_idx])
                        decline_pct = (peak_price - bottom_price) / peak_price * 100
                        
                        patterns.append({
                            'code': code,
                            'peak_before_date': pd.Timestamp(dates[peak_idx]).strftime('%Y-%m-%d'),
                            'peak_before_price': round(float(peak_price), 2),
                            'bottom_date': pd.Timestamp(dates[bottom_idx]).strftime('%Y-%m-%d'),
                            'bottom_price': round(float(bottom_price), 2),
                            'decline_pct': round(decline_pct, 2),
                            'bounce_date': pd.Timestamp(dates[j]).strftime('%Y-%m-%d'),
                            'bounce_price': round(float(close[j]), 2),
                            'bounce_pct': round(bounce_pct, 2),
                            'decay_days': bottom_idx - peak_idx,
                            'recover_days': j - bottom_idx,
                            'avg_daily_decline_before': round(avg_prev_ret * 100, 3),
                            'avg_daily_decline_accel': round(avg_recent_ret * 100, 3),
                        })
                        bounce_found = True
                        break
                
                # 已处理过这个底部
                last_processed_idx = bottom_idx
    
    return patterns


def main():
    mode = 'zfzs'
    min_days = 1
    lookback = 60
    decline_pct = 20.0
    vol_multi = 2.0
    # oversold 模式参数
    lookback_decline = 40
    lookback_accel = 20
    accel_threshold = 1.5
    bounce_target_pct = 30.0

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--mode' and i + 1 < len(argv):
            mode = argv[i + 1]; i += 2
        elif argv[i] == '--min-days' and i + 1 < len(argv):
            min_days = int(argv[i + 1]); i += 2
        elif argv[i] == '--lookback' and i + 1 < len(argv):
            lookback = int(argv[i + 1]); i += 2
        elif argv[i] == '--decline' and i + 1 < len(argv):
            decline_pct = float(argv[i + 1]); i += 2
        elif argv[i] == '--vol-multi' and i + 1 < len(argv):
            vol_multi = float(argv[i + 1]); i += 2
        elif argv[i] == '--lookback-decline' and i + 1 < len(argv):
            lookback_decline = int(argv[i + 1]); i += 2
        elif argv[i] == '--lookback-accel' and i + 1 < len(argv):
            lookback_accel = int(argv[i + 1]); i += 2
        elif argv[i] == '--accel-threshold' and i + 1 < len(argv):
            accel_threshold = float(argv[i + 1]); i += 2
        elif argv[i] == '--bounce-target' and i + 1 < len(argv):
            bounce_target_pct = float(argv[i + 1]); i += 2
        else:
            i += 1

    files = sorted(DATA_DIR.glob("*.csv"))
    total = len(files)

    if mode == 'zfzs':
        min_days = max(min_days, 20) if min_days == 1 else min_days
        print("=" * 72)
        print(f"  波段识别器 — zhixing 金叉模式")
        print(f"  入场: zhixing_fast 金叉 zhixing_slow")
        print(f"  离场: MA20 死叉 MA60")
        print(f"  过滤: 波段 ≥ {min_days}天")
        print("=" * 72)
        output_name = "wave_catcher_zfzs.csv"
    elif mode == 'volume':
        print("=" * 72)
        print(f"  波段识别器 — 放量突破模式")
        print(f"  入场: 长期下跌{lookback}天且跌>{decline_pct}% + 放量>{vol_multi}倍 + 站上zhixing_fast")
        print(f"  离场: 跌破入场前最低价 or 涨幅100%")
        print(f"  过滤: 波段 ≥ {min_days}天")
        print("=" * 72)
        output_name = "wave_catcher_volume.csv"
    elif mode == 'oversold':
        print("=" * 72)
        print(f"  波段识别器 — 超跌反弹模式")
        print(f"  检测: 加速下跌（最近{lookback_accel}天平均跌速 > 前{lookback_decline}天 × {accel_threshold}倍）")
        print(f"  识别: 从触底点反弹 ≥ {bounce_target_pct}%")
        print(f"  记录: 超跌反弹形态的所有数据点")
        print("=" * 72)
        output_name = "wave_catcher_oversold.csv"
    else:
        print(f"未知模式: {mode}，支持 zfzs / volume / oversold")
        sys.exit(1)

    print(f"\n  共 {total} 只个股待扫描\n")

    all_waves = []
    scanned = 0
    skipped = 0
    
    # 根据模式确定最小数据要求
    if mode == 'zfzs':
        min_data_days = 200
    elif mode == 'volume':
        min_data_days = lookback + 30
    else:  # oversold
        min_data_days = lookback_decline + lookback_accel + 50

    for f in files:
        code = f.stem
        scanned += 1
        df = load_data(f, min_data_days)
        if df is None:
            skipped += 1
            continue

        if mode == 'zfzs':
            waves = detect_waves(df, code, min_days)
        elif mode == 'volume':
            waves = detect_volume_waves(df, code, lookback, decline_pct, vol_multi, min_days)
        else:  # oversold
            waves = detect_oversold_bounce_waves(df, code, lookback_decline, lookback_accel, 
                                                accel_threshold, bounce_target_pct)
        all_waves.extend(waves)

        if scanned % 500 == 0:
            print(f"  进度: {scanned}/{total} | 已发现 {len(all_waves)} 个波段")

    print(f"  完成: {scanned}/{total} 扫描 | {skipped} 跳过 | "
          f"共发现 {len(all_waves)} 个波段")

    if not all_waves:
        print("\n  未发现任何波段。")
        return

    results_df = pd.DataFrame(all_waves)
    
    # 根据模式选择排序字段
    if mode == 'oversold':
        sort_key = 'bounce_date'
    else:
        sort_key = 'entry_date'
    
    results_df = results_df.sort_values(sort_key, ascending=False)
    results_df = results_df.reset_index(drop=True)

    # 统计
    if mode == 'oversold':
        closed = pd.DataFrame()  # oversold 模式不区分 closed/holding
        holding = pd.DataFrame()
    else:
        closed = results_df[~results_df.get('holding', pd.Series(False)).fillna(False)]
        holding = results_df[results_df.get('holding', pd.Series(False)).fillna(False)]

    print(f"\n  --- 波段概览 ---")
    
    if mode != 'oversold':
        print(f"  已结束: {len(closed)}  持仓中: {len(holding)}")
        rets = closed['return_pct']
        print(f"  涨幅: P10={np.percentile(rets,10):.0f}%  P25={np.percentile(rets,25):.0f}%  "
              f"P50={np.median(rets):.0f}%  P75={np.percentile(rets,75):.0f}%  P90={np.percentile(rets,90):.0f}%")
        print(f"  回撤: P50={closed['max_drawdown_pct'].median():.1f}%  P90={np.percentile(closed['max_drawdown_pct'],90):.1f}%")
        print(f"  天数: P50={closed['duration_days'].median():.0f}  P90={np.percentile(closed['duration_days'],90):.0f}")

    # volume模式额外统计
    if mode == 'volume' and 'exit_reason' in closed.columns:
        print(f"\n  --- 离场原因 ---")
        for reason, label in [('stop', '止损(跌破前低)'), ('target', '止盈(涨100%)')]:
            n_reason = (closed['exit_reason'] == reason).sum()
            if n_reason > 0:
                sub = closed[closed['exit_reason'] == reason]
                print(f"  {label}: {n_reason} 个 ({n_reason/len(closed)*100:.1f}%)  "
                      f"涨幅中位={sub['return_pct'].median():.0f}%  天数中位={sub['duration_days'].median():.0f}")
    
    # oversold模式额外统计
    if mode == 'oversold':
        print(f"\n  --- 超跌反弹形态统计 ---")
        print(f"  下跌幅度: P25={results_df['decline_pct'].quantile(0.25):.1f}%  "
              f"P50={results_df['decline_pct'].median():.1f}%  "
              f"P75={results_df['decline_pct'].quantile(0.75):.1f}%")
        print(f"  反弹幅度: P25={results_df['bounce_pct'].quantile(0.25):.1f}%  "
              f"P50={results_df['bounce_pct'].median():.1f}%  "
              f"P75={results_df['bounce_pct'].quantile(0.75):.1f}%")
        print(f"  下跌周期: P50={results_df['decay_days'].median():.0f}天  "
              f"P75={results_df['decay_days'].quantile(0.75):.0f}天")
        print(f"  反弹周期: P50={results_df['recover_days'].median():.0f}天  "
              f"P75={results_df['recover_days'].quantile(0.75):.0f}天")
        print(f"  平均跌速(前期): {results_df['avg_daily_decline_before'].mean():.3f}%/天")
        print(f"  平均跌速(加速): {results_df['avg_daily_decline_accel'].mean():.3f}%/天")

    # 输出表
    if mode == 'oversold':
        print(f"\n{'=' * 130}")
        print(f"{'#':<5} {'代码':<8} {'下跌高点':<12} {'高点价':>8} "
              f"{'触底日':<12} {'底价':>8} {'下跌%':>7} {'反弹日':<12} {'反弹价':>8} {'反弹%':>7} {'下跌天':>5} {'反弹天':>5}")
        print("-" * 130)
        
        for idx, row in results_df.head(50).iterrows():
            print(f"{idx+1:<5} {row['code']:<8} {row['peak_before_date']:<12} {row['peak_before_price']:>8.2f} "
                  f"{row['bottom_date']:<12} {row['bottom_price']:>8.2f} {row['decline_pct']:>6.1f}% "
                  f"{row['bounce_date']:<12} {row['bounce_price']:>8.2f} {row['bounce_pct']:>6.1f}% "
                  f"{row['decay_days']:>5.0f} {row['recover_days']:>5.0f}")
    else:
        has_reason = 'exit_reason' in results_df.columns
        extra = f" {'离场':<6}" if has_reason else ""
        print(f"\n{'=' * 100}")
        print(f"{'#':<5} {'代码':<8} {'入场日':<12} {'离场日':<12} "
              f"{'天数':<6} {'入场价':>8} {'峰值':>8} {'离场价':>8} {'涨幅':>8} {'回撤':>6}{extra} {'状态':<4}")
        print("-" * 100)

        for idx, row in results_df.head(50).iterrows():
            status = '持仓' if row.get('holding', False) else ''
            reason = row.get('exit_reason', '') if has_reason else ''
            print(f"{idx+1:<5} {row['code']:<8} {row['entry_date']:<12} {row['exit_date']:<12} "
                  f"{row['duration_days']:<6} {row['entry_price']:>8.2f} {row['peak_price']:>8.2f} "
                  f"{row['exit_price']:>8.2f} {row['return_pct']:>7.1f}% {row['max_drawdown_pct']:>5.1f}% "
                  f"{reason:<6} {status}")

    if len(results_df) > 50:
        print(f"\n  ... 共 {len(results_df)} 个波段，仅显示最近 50 个")

    # 保存
    output_path = Path(__file__).parent / output_name
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
