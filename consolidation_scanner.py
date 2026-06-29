#!/usr/bin/env python3
"""
横盘震荡区域扫描器

扫描全市场个股，识别 ~2 年（400-600 交易日）的横盘震荡区间，
分析震荡区域内部的量价特征，并按突破方向（上/下/继续震荡）分组对比。

用法:
    python3 consolidation_scanner.py
    python3 consolidation_scanner.py --range-pct 50 --window-min 400 --window-max 600
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from indicator import sma, ema, atr, zhixing_fast, zhixing_slow, macd, bollinger

DATA_DIR = Path.home() / "Documents" / "mainland_data_2014"


def load_data(filepath: Path, min_days: int = 600) -> pd.DataFrame | None:
    """加载个股日线数据"""
    try:
        df = pd_csv(filepath, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < min_days:
            return None
        return df
    except Exception:
        return None


def pd_csv(filepath, **kwargs):
    return pd.read_csv(filepath, **kwargs)


# ============================================================
# 核心：扫描横盘震荡区域
# ============================================================

def find_consolidation_zones(df: pd.DataFrame, code: str,
                              window_min: int = 400,
                              window_max: int = 600,
                              window_step: int = 20,
                              range_pct_max: float = 50.0,
                              trend_limit_pct: float = 30.0,
                              vol_converge: bool = True) -> list[dict]:
    """
    扫描单只股票的横盘震荡区域。

    条件：
      1. 持续 window_min ~ window_max 个交易日
      2. (最高价 - 最低价) / 中位收盘价 ≤ range_pct_max%
      3. 起终点价格差异 ≤ trend_limit_pct%
      4. 可选：后半段波动率不高于前半段（收敛）

    返回：每个震荡区域的起止索引和基本属性
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    dates = df['date'].values
    n = len(close)

    if n < window_min:
        return []

    zones = []
    used_ranges = []  # 已占用的索引范围，避免重复

    for win_size in range(window_min, window_max + 1, window_step):
        for start in range(0, n - win_size + 1, window_step):
            end = start + win_size

            # 检查是否与已有区域重叠
            overlaps = False
            for (us, ue) in used_ranges:
                if start < ue and end > us:
                    overlaps = True
                    break
            if overlaps:
                continue

            seg_close = close[start:end]
            seg_high = high[start:end]
            seg_low = low[start:end]

            max_price = float(np.max(seg_high))
            min_price = float(np.min(seg_low))
            median_close = float(np.median(seg_close))

            if median_close <= 0:
                continue

            # 条件1：振幅约束
            range_pct = (max_price - min_price) / median_close * 100
            if range_pct > range_pct_max:
                continue

            # 条件2：无强趋势
            start_price = float(close[start])
            end_price = float(close[end - 1])
            if start_price <= 0:
                continue
            trend_pct = abs(end_price - start_price) / start_price * 100
            if trend_pct > trend_limit_pct:
                continue

            # 条件3（可选）：波动率收敛
            if vol_converge:
                half = win_size // 2
                seg_first_half = close[start:start + half]
                seg_second_half = close[start + half:end]
                vol_first = float(np.std(np.diff(seg_first_half) / seg_first_half[:-1]))
                vol_second = float(np.std(np.diff(seg_second_half) / seg_second_half[:-1]))
                if vol_second > vol_first * 1.2:  # 后半段波动率比前半段高20%以上，跳过
                    continue

            zones.append({
                'code': code,
                'start_idx': start,
                'end_idx': end,
                'start_date': pd.Timestamp(dates[start]).strftime('%Y-%m-%d'),
                'end_date': pd.Timestamp(dates[end - 1]).strftime('%Y-%m-%d'),
                'duration_days': win_size,
                'range_pct': round(range_pct, 1),
                'trend_pct': round(trend_pct, 1),
                'median_close': round(median_close, 2),
                'max_price': round(max_price, 2),
                'min_price': round(min_price, 2),
            })

            used_ranges.append((start, end))

    return zones


# ============================================================
# 提取震荡区域内部特征
# ============================================================

def extract_zone_features(df: pd.DataFrame, zone: dict) -> dict:
    """
    对单个震荡区域，提取内部量价特征。
    """
    start = zone['start_idx']
    end = zone['end_idx']
    n = end - start

    close = df['close'].values[start:end]
    high = df['high'].values[start:end]
    low = df['low'].values[start:end]
    volume = df['volume'].values[start:end]
    dates = df['date'].values[start:end]

    features = {}

    # --- 量能特征 ---
    third = n // 3
    vol_first = float(np.mean(volume[:third]))
    vol_last = float(np.mean(volume[-third:]))
    features['vol_shrink_ratio'] = round(vol_last / (vol_first + 1), 3)  # <1 表示缩量

    # 成交量趋势（线性回归斜率 / 均值）
    vol_arr = volume.astype(float)
    x = np.arange(len(vol_arr))
    if np.std(vol_arr) > 0:
        vol_slope = np.polyfit(x, vol_arr, 1)[0]
        features['vol_trend'] = round(vol_slope / np.mean(vol_arr) * 100, 3)
    else:
        features['vol_trend'] = 0.0

    # 累计换手率（如果有 turnover_rate 列）
    if 'turnover_rate' in df.columns:
        tr = df['turnover_rate'].values[start:end]
        features['cum_turnover'] = round(float(np.nansum(tr)), 1)
    else:
        features['cum_turnover'] = 0.0

    # --- 波动率特征 ---
    # ATR
    atr_series = atr(pd.Series(high), pd.Series(low), pd.Series(close), 14)
    atr_vals = atr_series.values
    valid_atr = atr_vals[~np.isnan(atr_vals)]
    if len(valid_atr) > 0:
        features['atr_median'] = round(float(np.median(valid_atr)), 3)
        features['atr_pct'] = round(float(np.median(valid_atr)) / float(np.median(close)) * 100, 2)
    else:
        features['atr_median'] = 0.0
        features['atr_pct'] = 0.0

    # ATR 收敛度：前1/3 vs 后1/3
    if len(valid_atr) > third * 2:
        atr_first = float(np.median(valid_atr[:third]))
        atr_last = float(np.median(valid_atr[-third:]))
        features['atr_converge'] = round(atr_last / (atr_first + 0.001), 3)
    else:
        features['atr_converge'] = 1.0

    # 布林带宽度
    bb = bollinger(pd.Series(close), 20, 2.0)
    bb_width = (bb['upper'].values - bb['lower'].values) / bb['mid'].values * 100
    valid_bb = bb_width[~np.isnan(bb_width)]
    if len(valid_bb) > 0:
        features['bb_width_median'] = round(float(np.median(valid_bb)), 2)
    else:
        features['bb_width_median'] = 0.0

    # 日收益率标准差（年化波动率）
    daily_ret = np.diff(close) / close[:-1]
    features['annual_vol'] = round(float(np.std(daily_ret)) * np.sqrt(250) * 100, 1)

    # --- 均线特征 ---
    ma20 = sma(pd.Series(close), 20).values
    ma60 = sma(pd.Series(close), 60).values
    ma120 = sma(pd.Series(close), 120).values

    # 均线缠绕程度：MA20 相对 MA60 的偏离度标准差
    valid_ma = ~(np.isnan(ma20) | np.isnan(ma60))
    if valid_ma.sum() > 30:
        ma_spread = (ma20[valid_ma] - ma60[valid_ma]) / ma60[valid_ma] * 100
        features['ma_tangle'] = round(float(np.std(ma_spread)), 2)
    else:
        features['ma_tangle'] = 0.0

    # MA 多头排列比例（MA20 > MA60 > MA120 的天数占比）
    valid_all = ~(np.isnan(ma20) | np.isnan(ma60) | np.isnan(ma120))
    if valid_all.sum() > 0:
        bull_days = np.sum((ma20 > ma60) & (ma60 > ma120)) / valid_all.sum()
        features['ma_bull_ratio'] = round(float(bull_days), 3)
    else:
        features['ma_bull_ratio'] = 0.5

    # --- 知行指标特征 ---
    zf = zhixing_fast(pd.Series(close)).values
    zs = zhixing_slow(pd.Series(close)).values
    valid_zx = ~(np.isnan(zf) | np.isnan(zs))
    if valid_zx.sum() > 0:
        # 知行金叉/死叉次数
        cross_up = 0
        cross_down = 0
        for i in range(1, len(zf)):
            if np.isnan(zf[i]) or np.isnan(zs[i]) or np.isnan(zf[i-1]) or np.isnan(zs[i-1]):
                continue
            if zf[i-1] <= zs[i-1] and zf[i] > zs[i]:
                cross_up += 1
            if zf[i-1] >= zs[i-1] and zf[i] < zs[i]:
                cross_down += 1
        features['zx_cross_count'] = cross_up + cross_down

        # 知行缠绕程度：zf 相对 zs 的偏离度标准差
        zx_spread = (zf[valid_zx] - zs[valid_zx]) / zs[valid_zx] * 100
        features['zx_tangle'] = round(float(np.std(zx_spread)), 2)
    else:
        features['zx_cross_count'] = 0
        features['zx_tangle'] = 0.0

    # --- MACD 特征 ---
    macd_df = macd(pd.Series(close))
    dif = macd_df['dif'].values
    valid_dif = dif[~np.isnan(dif)]
    if len(valid_dif) > 0:
        # DIF 在零轴上方的比例
        features['dif_above_zero'] = round(float(np.sum(valid_dif > 0) / len(valid_dif)), 3)
        # DIF 绝对值中位（相对于价格）
        features['dif_abs_pct'] = round(float(np.median(np.abs(valid_dif))) / float(np.median(close)) * 100, 3)
    else:
        features['dif_above_zero'] = 0.5
        features['dif_abs_pct'] = 0.0

    # --- 估值特征 ---
    if 'pe_ttm' in df.columns:
        pe = df['pe_ttm'].values[start:end]
        valid_pe = pe[(~np.isnan(pe)) & (pe > 0) & (pe < 1000)]
        if len(valid_pe) > 0:
            features['pe_median'] = round(float(np.median(valid_pe)), 1)
            features['pe_range'] = round(float(np.percentile(valid_pe, 90) - np.percentile(valid_pe, 10)), 1)
        else:
            features['pe_median'] = 0.0
            features['pe_range'] = 0.0
    else:
        features['pe_median'] = 0.0
        features['pe_range'] = 0.0

    if 'pb' in df.columns:
        pb = df['pb'].values[start:end]
        valid_pb = pb[(~np.isnan(pb)) & (pb > 0) & (pb < 100)]
        if len(valid_pb) > 0:
            features['pb_median'] = round(float(np.median(valid_pb)), 2)
        else:
            features['pb_median'] = 0.0
    else:
        features['pb_median'] = 0.0

    # --- 市值特征 ---
    if 'total_mv' in df.columns:
        mv = df['total_mv'].values[start:end]
        valid_mv = mv[~np.isnan(mv)]
        if len(valid_mv) > 0:
            features['mv_median'] = round(float(np.median(valid_mv)), 0)
            # 市值变化：最后 vs 最初
            features['mv_change'] = round(float(valid_mv[-1] / (valid_mv[0] + 1) - 1) * 100, 1)
        else:
            features['mv_median'] = 0.0
            features['mv_change'] = 0.0
    else:
        features['mv_median'] = 0.0
        features['mv_change'] = 0.0

    return features


# ============================================================
# 判定突破方向
# ============================================================

def determine_breakout(df: pd.DataFrame, zone: dict,
                       breakout_days: int = 120,
                       breakout_pct: float = 30.0) -> dict:
    """
    震荡窗口结束后，观察后续 breakout_days 天的走势。

    返回：
      direction: 'up' / 'down' / 'range'
      max_return: 最大涨幅%
      max_drawdown: 最大回撤%
      final_return: 最终涨幅%
    """
    end_idx = zone['end_idx']
    close = df['close'].values
    n = len(close)

    if end_idx >= n:
        return {'direction': 'unknown', 'max_return': 0, 'max_drawdown': 0, 'final_return': 0}

    look_end = min(end_idx + breakout_days, n)
    future_close = close[end_idx:look_end]

    if len(future_close) < 20:
        return {'direction': 'unknown', 'max_return': 0, 'max_drawdown': 0, 'final_return': 0}

    entry_price = float(close[end_idx - 1])

    # 最大涨幅
    max_price_after = float(np.max(future_close))
    max_return = (max_price_after / entry_price - 1) * 100

    # 最大回撤
    min_price_after = float(np.min(future_close))
    max_drawdown = (1 - min_price_after / entry_price) * 100

    # 最终涨幅
    final_price = float(future_close[-1])
    final_return = (final_price / entry_price - 1) * 100

    # 判定方向
    if max_return >= breakout_pct:
        direction = 'up'
    elif max_drawdown >= breakout_pct:
        direction = 'down'
    else:
        direction = 'range'

    return {
        'direction': direction,
        'max_return': round(max_return, 1),
        'max_drawdown': round(max_drawdown, 1),
        'final_return': round(final_return, 1),
    }


# ============================================================
# 主流程
# ============================================================

def main():
    # 参数
    window_min = 400
    window_max = 600
    window_step = 20
    range_pct = 50.0
    trend_limit = 30.0
    breakout_days = 120
    breakout_pct = 30.0
    vol_converge = True

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--window-min' and i + 1 < len(argv):
            window_min = int(argv[i + 1]); i += 2
        elif argv[i] == '--window-max' and i + 1 < len(argv):
            window_max = int(argv[i + 1]); i += 2
        elif argv[i] == '--range-pct' and i + 1 < len(argv):
            range_pct = float(argv[i + 1]); i += 2
        elif argv[i] == '--trend-limit' and i + 1 < len(argv):
            trend_limit = float(argv[i + 1]); i += 2
        elif argv[i] == '--breakout-days' and i + 1 < len(argv):
            breakout_days = int(argv[i + 1]); i += 2
        elif argv[i] == '--breakout-pct' and i + 1 < len(argv):
            breakout_pct = float(argv[i + 1]); i += 2
        elif argv[i] == '--no-vol-converge':
            vol_converge = False; i += 1
        else:
            i += 1

    files = sorted(DATA_DIR.glob("*.csv"))
    total = len(files)

    print("=" * 72)
    print(f"  横盘震荡区域扫描器")
    print(f"  窗口: {window_min}-{window_max}天 | 振幅 ≤ {range_pct}%")
    print(f"  趋势限制: ≤ {trend_limit}% | 波动率收敛: {'是' if vol_converge else '否'}")
    print(f"  突破窗口: {breakout_days}天 | 突破阈值: {breakout_pct}%")
    print("=" * 72)
    print(f"\n  共 {total} 只个股待扫描\n")

    all_zones = []
    scanned = 0
    skipped = 0

    for f in files:
        code = f.stem
        scanned += 1

        df = load_data(f, window_max + breakout_days + 60)
        if df is None:
            skipped += 1
            continue

        # 扫描震荡区域
        zones = find_consolidation_zones(
            df, code, window_min, window_max, window_step,
            range_pct, trend_limit, vol_converge
        )

        for zone in zones:
            # 提取特征
            features = extract_zone_features(df, zone)
            # 判定突破方向
            breakout = determine_breakout(df, zone, breakout_days, breakout_pct)

            record = {**zone, **features, **breakout}
            # 删除索引字段（不需要输出）
            del record['start_idx']
            del record['end_idx']
            all_zones.append(record)

        if scanned % 500 == 0:
            print(f"  进度: {scanned}/{total} | 已发现 {len(all_zones)} 个震荡区域")

    print(f"  完成: {scanned}/{total} 扫描 | {skipped} 跳过 | "
          f"共发现 {len(all_zones)} 个震荡区域")

    if not all_zones:
        print("\n  未发现任何符合条件的震荡区域。")
        return

    results_df = pd.DataFrame(all_zones)

    # ============================================================
    # 分组统计
    # ============================================================
    print("\n" + "=" * 72)
    print("  突破方向分布")
    print("=" * 72)

    dir_counts = results_df['direction'].value_counts()
    for d in ['up', 'down', 'range']:
        cnt = dir_counts.get(d, 0)
        pct = cnt / len(results_df) * 100
        label = {'up': '向上突破', 'down': '向下突破', 'range': '继续震荡'}[d]
        print(f"  {label}: {cnt} ({pct:.1f}%)")

    # ============================================================
    # 特征对比
    # ============================================================
    feature_cols = [
        'vol_shrink_ratio', 'vol_trend', 'cum_turnover',
        'atr_pct', 'atr_converge', 'bb_width_median', 'annual_vol',
        'ma_tangle', 'ma_bull_ratio',
        'zx_cross_count', 'zx_tangle',
        'dif_above_zero', 'dif_abs_pct',
        'pe_median', 'pe_range', 'pb_median',
        'mv_median', 'mv_change',
        'range_pct', 'trend_pct', 'duration_days',
    ]

    print("\n" + "=" * 72)
    print("  震荡区域特征对比（中位数）")
    print("=" * 72)

    up_df = results_df[results_df['direction'] == 'up']
    down_df = results_df[results_df['direction'] == 'down']
    range_df = results_df[results_df['direction'] == 'range']

    header = f"{'特征':<22} {'向上突破':>10} {'向下突破':>10} {'继续震荡':>10} {'区分度':>8}"
    print(header)
    print("-" * len(header))

    feature_diffs = []
    for col in feature_cols:
        if col not in results_df.columns:
            continue
        up_med = up_df[col].median() if len(up_df) > 0 else 0
        down_med = down_df[col].median() if len(down_df) > 0 else 0
        range_med = range_df[col].median() if len(range_df) > 0 else 0

        # 区分度：向上 vs 向下的差异 / 整体标准差
        overall_std = results_df[col].std()
        if overall_std > 0:
            diff = abs(up_med - down_med) / overall_std
        else:
            diff = 0

        feature_diffs.append((col, up_med, down_med, range_med, diff))

    # 按区分度排序
    feature_diffs.sort(key=lambda x: x[4], reverse=True)

    for col, up_med, down_med, range_med, diff in feature_diffs:
        stars = "★" * min(5, int(diff * 2)) if diff > 0.1 else ""
        print(f"  {col:<20} {up_med:>10.2f} {down_med:>10.2f} {range_med:>10.2f} {stars:>8}")

    # ============================================================
    # 向上突破的特征分布
    # ============================================================
    if len(up_df) > 10:
        print("\n" + "=" * 72)
        print(f"  向上突破组（{len(up_df)} 个）的特征分布")
        print("=" * 72)

        for col, _, _, _, diff in feature_diffs[:8]:
            if col not in up_df.columns:
                continue
            vals = up_df[col]
            print(f"\n  {col}:")
            print(f"    P25={vals.quantile(0.25):.2f}  P50={vals.median():.2f}  P75={vals.quantile(0.75):.2f}  P90={vals.quantile(0.90):.2f}")

    # ============================================================
    # 向下突破的特征分布
    # ============================================================
    if len(down_df) > 10:
        print("\n" + "=" * 72)
        print(f"  向下突破组（{len(down_df)} 个）的特征分布")
        print("=" * 72)

        for col, _, _, _, diff in feature_diffs[:8]:
            if col not in down_df.columns:
                continue
            vals = down_df[col]
            print(f"\n  {col}:")
            print(f"    P25={vals.quantile(0.25):.2f}  P50={vals.median():.2f}  P75={vals.quantile(0.75):.2f}  P90={vals.quantile(0.90):.2f}")

    # ============================================================
    # 保存
    # ============================================================
    output_path = Path(__file__).parent / "consolidation_zones.csv"
    results_df = results_df.sort_values(['direction', 'start_date'], ascending=[True, False])
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")
    print(f"共 {len(results_df)} 个震荡区域")

    # 案例展示
    print("\n" + "=" * 72)
    print("  向上突破典型案例（涨幅最大的 10 个）")
    print("=" * 72)
    if len(up_df) > 0:
        top_up = up_df.nlargest(10, 'max_return')
        for _, row in top_up.iterrows():
            print(f"  {row['code']}  {row['start_date']}~{row['end_date']}  "
                  f"振幅{row['range_pct']:.0f}%  持续{row['duration_days']:.0f}天  "
                  f"→ 涨{row['max_return']:.0f}%  最终{row['final_return']:+.0f}%")

    print("\n" + "=" * 72)
    print("  向下突破典型案例（跌幅最大的 10 个）")
    print("=" * 72)
    if len(down_df) > 0:
        top_down = down_df.nlargest(10, 'max_drawdown')
        for _, row in top_down.iterrows():
            print(f"  {row['code']}  {row['start_date']}~{row['end_date']}  "
                  f"振幅{row['range_pct']:.0f}%  持续{row['duration_days']:.0f}天  "
                  f"→ 跌{row['max_drawdown']:.0f}%  最终{row['final_return']:+.0f}%")


if __name__ == "__main__":
    main()
