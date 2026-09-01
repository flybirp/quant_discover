#!/usr/bin/env python3
"""
周线筑底跌破（"漏了"）形态扫描器

捕获的三段式形态（在周线结构上识别筑底，在日线上定位跌破与走势）：

  头部段落 —— 周线结构筑底：
    - 位置：长期下跌后的首次筑底（first_base_after_drop），
            或宽幅震荡下沿的筑底（range_low）；
    - 形态：至少 min_base_weeks（默认 2）根周K，周低点紧贴同一支撑位
            （各周低点相对筑底低点的离散 ≤ support_tolerance_pct）。

  中部段落 —— "漏了"（跌破），按跌破点价格相对筑底低点的比例分为互斥四组：
    - 筑底完成后，首次跌破筑底低点（跌破日最低价 < 筑底低点）；
    - A 组（深度 0~5%）：筑底低点×0.95 < 跌破点价格 < 筑底低点；
    - B 组（深度 5~10%）：筑底低点×0.90 < 跌破点价格 ≤ 筑底低点×0.95；
    - C 组（深度 10~15%）：筑底低点×0.85 < 跌破点价格 ≤ 筑底低点×0.90；
    - D 组（深度 15~30%）：筑底低点×0.70 < 跌破点价格 ≤ 筑底低点×0.85；
    - 跌破深度 > 30% 不纳入；
    - 按跌破日量能相对前 N 日均量，标注三种条件：
        放量：量比 > vol_ratio_high（默认 1.2）
        缩量：量比 < vol_ratio_low（默认 0.8）
        平量：介于两者之间

  尾部段落 —— 之后 4 周的走势：
    - 记录跌破后第 1~4 周的涨跌幅、期间最大涨幅/最大回撤、是否收回筑底低点。

  数据说明：价格为前复权，除权日在序列中不产生假跳空，跨期价格直接可比，
  无需除权过滤。因此"单日跌破 15%~30%"（B 组）只能发生在创业板/科创板/
  北交所等 20%/30% 涨跌幅品种上，均视为真实暴跌。

用法:
    python3 wave_catcher_base_break.py
    python3 wave_catcher_base_break.py --min-base-weeks 2 --ratios 0.95,0.90,0.85,0.70
    python3 wave_catcher_base_break.py --decline 20 --range-amp 40 --vol-high 1.2 --vol-low 0.8
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_DIR = Path.home() / "Documents" / "mainland_data_2014"


def load_data(filepath: Path, min_days: int = 250) -> pd.DataFrame | None:
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


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日线 -> 周线（按自然周聚合 OHLCV），保留每周首末日在日线中的索引"""
    d = df.reset_index(drop=True).copy()
    d['_idx'] = np.arange(len(d))
    d['week'] = d['date'].dt.to_period('W')
    g = d.groupby('week', sort=True)
    weekly = pd.DataFrame({
        'open': g['open'].first(),
        'high': g['high'].max(),
        'low': g['low'].min(),
        'close': g['close'].last(),
        'volume': g['volume'].sum(),
        'week_start_idx': g['_idx'].min(),
        'week_end_idx': g['_idx'].max(),
    }).reset_index(drop=True)
    return weekly


def detect_base_breaks(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """
    识别"筑底 -> 轻微跌破 -> 观察4周"形态。
    返回每条命中形态的记录。
    """
    weekly = to_weekly(df)
    nw = len(weekly)

    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    dates = df['date'].values
    n = len(df)

    results = []

    # 位置判断需要筑底段之前至少有 lookback_weeks 周
    i = p['lookback_weeks']
    while i < nw:
        # ============ 头部：识别筑底段（连续周，周低点紧贴同一支撑） ============
        # 从第 i 周起向后扩展，只要新一周的周低点不使段内低点离散度
        # 超过 support_tolerance_pct，就纳入筑底段；否则筑底段在此前结束。
        seg_min = float(weekly['low'].iloc[i])
        seg_max = seg_min
        seg_end = i
        j = i
        while j < nw:
            low_j = float(weekly['low'].iloc[j])
            new_min = min(seg_min, low_j)
            new_max = max(seg_max, low_j)
            spread = (new_max - new_min) / new_min * 100
            if spread > p['support_tolerance_pct']:
                break
            seg_min = new_min
            seg_max = new_max
            seg_end = j
            j += 1

        seg_len = seg_end - i + 1
        if seg_len < p['min_base_weeks']:
            i = seg_end + 1
            continue

        base_low = seg_min

        # ============ 位置判断：筑底段之前的 lookback_weeks 周 ============
        lookback_start = i - p['lookback_weeks']
        prior = weekly.iloc[lookback_start:i]
        prior_high = float(prior['high'].max())
        prior_low = float(prior['low'].min())
        if prior_high <= 0 or prior_low <= 0:
            i = seg_end + 1
            continue

        prior_decline = (prior_high - base_low) / prior_high * 100
        prior_range = (prior_high - prior_low) / prior_low * 100
        near_low = (base_low - prior_low) / prior_low * 100

        # 优先判定"宽幅震荡下沿"，否则判定"下跌后首次筑底"
        if prior_range >= p['range_amplitude_pct'] and near_low <= p['range_low_margin_pct']:
            position_type = 'range_low'
        elif prior_decline >= p['decline_pct']:
            position_type = 'first_base_after_drop'
        else:
            i = seg_end + 1
            continue

        # ============ 筑底段在日线上的边界 ============
        base_start_idx = int(weekly['week_start_idx'].iloc[i])
        base_end_idx = int(weekly['week_end_idx'].iloc[seg_end])
        base_start_date = pd.Timestamp(dates[base_start_idx]).strftime('%Y-%m-%d')
        base_end_date = pd.Timestamp(dates[base_end_idx]).strftime('%Y-%m-%d')

        seg_close = close[base_start_idx:base_end_idx + 1]
        seg_high = high[base_start_idx:base_end_idx + 1]
        seg_low = low[base_start_idx:base_end_idx + 1]
        base_range_pct = (seg_high.max() - seg_low.min()) / seg_close.mean() * 100

        # ============ 中部：筑底结束后，首次跌破筑底低点的日子 ============
        break_idx = None
        for k in range(base_end_idx + 1, n):
            if low[k] < base_low:
                break_idx = k
                break
        if break_idx is None:
            i = seg_end + 1
            continue

        # 跌破点价格按相对筑底低点的比例分组（互斥，由浅到深 A/B/C/D...）：
        #   第 g 组：筑底低点×ratios[g] < 价格 ≤ 筑底低点×ratios[g-1]
        #   （首组上界为筑底低点本身；跌破深度超过最后一档不纳入）
        break_low = float(low[break_idx])
        group = None
        for gi, ratio in enumerate(p['group_ratios']):
            if break_low > base_low * ratio:
                group = p['group_names'][gi]
                break
        if group is None:
            i = seg_end + 1
            continue
        penetration_pct = (base_low - break_low) / base_low * 100

        # 量能类别：跌破日量比 = 当日量 / 前 vol_base_days 日均量
        vb_start = max(0, break_idx - p['vol_base_days'])
        vol_base = np.mean(volume[vb_start:break_idx])
        vol_ratio = volume[break_idx] / vol_base if vol_base > 0 else np.nan
        if pd.isna(vol_ratio):
            vol_type = '平量'
        elif vol_ratio > p['vol_ratio_high']:
            vol_type = '放量'
        elif vol_ratio < p['vol_ratio_low']:
            vol_type = '缩量'
        else:
            vol_type = '平量'

        # ============ 尾部：跌破后 4 周（tail_weeks * 5 交易日）走势 ============
        base_close = close[break_idx]
        record = {
            'code': code,
            'group': group,
            'position_type': position_type,
            'base_start_date': base_start_date,
            'base_end_date': base_end_date,
            'base_weeks': seg_len,
            'base_low': round(base_low, 2),
            'base_range_pct': round(base_range_pct, 1),
            'prior_decline_pct': round(prior_decline, 1),
            'break_date': pd.Timestamp(dates[break_idx]).strftime('%Y-%m-%d'),
            'break_low': round(break_low, 2),
            'penetration_pct': round(penetration_pct, 2),
            'volume_ratio': round(float(vol_ratio), 2) if not pd.isna(vol_ratio) else np.nan,
            'volume_type': vol_type,
        }

        tail_days = p['tail_weeks'] * 5
        for w in range(1, p['tail_weeks'] + 1):
            tgt = break_idx + w * 5
            if tgt < n:
                record[f'ret_{w}w'] = round((close[tgt] / base_close - 1) * 100, 2)
            else:
                record[f'ret_{w}w'] = np.nan

        tail_end = min(break_idx + tail_days, n)
        if tail_end > break_idx + 1:
            tail_close = close[break_idx:tail_end]
            tail_high = high[break_idx:tail_end]
            tail_low = low[break_idx:tail_end]
            max_up = (tail_high.max() - base_close) / base_close * 100
            max_dd = (base_close - tail_low.min()) / base_close * 100
            # 收复筑底低点：以"收盘价重新站回"为准（区分度更高）
            recover_base = bool(tail_close.max() > base_low)
        else:
            max_up = np.nan
            max_dd = np.nan
            recover_base = False

        record['max_up_4w'] = round(float(max_up), 2) if not pd.isna(max_up) else np.nan
        record['max_dd_4w'] = round(float(max_dd), 2) if not pd.isna(max_dd) else np.nan
        record['recover_base'] = recover_base

        results.append(record)
        # 跳过整个筑底段，避免滑动窗口重复计数
        i = seg_end + 1

    return results


def main():
    # 默认参数
    p = {
        'min_base_weeks': 2,           # 筑底至少周数
        'support_tolerance_pct': 5.0,  # 周低点支撑容差（%）
        'lookback_weeks': 12,          # 位置判断回看周数
        'decline_pct': 20.0,           # 长期下跌跌幅阈值（%）
        'range_amplitude_pct': 40.0,   # 宽幅震荡振幅阈值（%）
        'range_low_margin_pct': 10.0,  # 筑底低点距震荡下沿最大距离（%）
        'group_ratios': [0.95, 0.90, 0.85, 0.70],  # 各组价格下限=筑底低点×ratio(由浅到深)
        'group_names': ['A', 'B', 'C', 'D'],       # 组名: A(0~5%) B(5~10%) C(10~15%) D(15~30%)
        'vol_base_days': 20,           # 量能基准天数
        'vol_ratio_high': 1.2,         # 放量阈值
        'vol_ratio_low': 0.8,          # 缩量阈值
        'tail_weeks': 4,               # 观察周数
    }

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--min-base-weeks' and i + 1 < len(argv):
            p['min_base_weeks'] = int(argv[i + 1]); i += 2
        elif argv[i] == '--support-tolerance' and i + 1 < len(argv):
            p['support_tolerance_pct'] = float(argv[i + 1]); i += 2
        elif argv[i] == '--lookback-weeks' and i + 1 < len(argv):
            p['lookback_weeks'] = int(argv[i + 1]); i += 2
        elif argv[i] == '--decline' and i + 1 < len(argv):
            p['decline_pct'] = float(argv[i + 1]); i += 2
        elif argv[i] == '--range-amp' and i + 1 < len(argv):
            p['range_amplitude_pct'] = float(argv[i + 1]); i += 2
        elif argv[i] == '--range-low-margin' and i + 1 < len(argv):
            p['range_low_margin_pct'] = float(argv[i + 1]); i += 2
        elif argv[i] == '--ratios' and i + 1 < len(argv):
            p['group_ratios'] = [float(x) for x in argv[i + 1].split(',')]
            p['group_names'] = [chr(ord('A') + k) for k in range(len(p['group_ratios']))]
            i += 2
        elif argv[i] == '--vol-base-days' and i + 1 < len(argv):
            p['vol_base_days'] = int(argv[i + 1]); i += 2
        elif argv[i] == '--vol-high' and i + 1 < len(argv):
            p['vol_ratio_high'] = float(argv[i + 1]); i += 2
        elif argv[i] == '--vol-low' and i + 1 < len(argv):
            p['vol_ratio_low'] = float(argv[i + 1]); i += 2
        elif argv[i] == '--tail-weeks' and i + 1 < len(argv):
            p['tail_weeks'] = int(argv[i + 1]); i += 2
        else:
            i += 1

    files = sorted(DATA_DIR.glob("*.csv"))
    total = len(files)

    print("=" * 72)
    print("  周线筑底跌破（'漏了'）形态扫描器")
    print(f"  筑底: ≥{p['min_base_weeks']}周 | 支撑容差 ≤{p['support_tolerance_pct']}%")
    print(f"  位置: 前{p['lookback_weeks']}周跌≥{p['decline_pct']}% 或 宽幅≥{p['range_amplitude_pct']}%下沿")
    print("  跌破分组（深度由浅到深）: " + " | ".join(
        f"{p['group_names'][k]} {round((1-r)*100)}%~{round((1-(p['group_ratios'][k-1] if k>0 else 1))*100)}%"
        for k, r in enumerate(p['group_ratios'])))
    print(f"  量能: 放>{p['vol_ratio_high']} 缩<{p['vol_ratio_low']} | 观察: 跌破后 {p['tail_weeks']} 周")
    print("=" * 72)
    print(f"\n  共 {total} 只个股待扫描\n")

    all_results = []
    scanned = 0
    skipped = 0

    for f in files:
        code = f.stem
        scanned += 1
        df = load_data(f, p['lookback_weeks'] * 5 + 120)
        if df is None:
            skipped += 1
            continue

        res = detect_base_breaks(df, code, p)
        all_results.extend(res)

        if scanned % 500 == 0:
            print(f"  进度: {scanned}/{total} | 已发现 {len(all_results)} 个形态")

    print(f"  完成: {scanned}/{total} 扫描 | {skipped} 跳过 | "
          f"共发现 {len(all_results)} 个形态")

    if not all_results:
        print("\n  未发现任何符合条件的形态。")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('break_date', ascending=False).reset_index(drop=True)

    # 概览
    print(f"\n  --- 形态概览 ---")
    print(f"  分组分布: {dict(results_df['group'].value_counts())}")
    print(f"  位置分布: {dict(results_df['position_type'].value_counts())}")
    print(f"  量能分布: {dict(results_df['volume_type'].value_counts())}")
    print(f"  跌破深度: P50={results_df['penetration_pct'].median():.2f}%  "
          f"P90={results_df['penetration_pct'].quantile(0.9):.2f}%")

    # 输出表（最近 50 个）
    print(f"\n{'=' * 120}")
    print(f"{'#':<4} {'组':<3} {'代码':<8} {'位置':<18} {'筑底止':<12} {'底价':>8} "
          f"{'跌破日':<12} {'破深%':>7} {'量比':>6} {'量能':<5} "
          f"{'1周':>7} {'2周':>7} {'3周':>7} {'4周':>7} {'涨max':>7} {'跌max':>7} {'收回':<4}")
    print("-" * 124)
    for idx, row in results_df.head(50).iterrows():
        pos = '下跌筑底' if row['position_type'] == 'first_base_after_drop' else '震荡下沿'
        rec = '是' if row['recover_base'] else '否'
        print(f"{idx+1:<4} {row['group']:<3} {row['code']:<8} {pos:<18} {row['base_end_date']:<12} {row['base_low']:>8.2f} "
              f"{row['break_date']:<12} {row['penetration_pct']:>6.2f}% {row['volume_ratio']:>6.2f} {row['volume_type']:<5} "
              f"{row['ret_1w']:>7} {row['ret_2w']:>7} {row['ret_3w']:>7} {row['ret_4w']:>7} "
              f"{row['max_up_4w']:>6.1f}% {row['max_dd_4w']:>6.1f}% {rec:<4}")

    if len(results_df) > 50:
        print(f"\n  ... 共 {len(results_df)} 个形态，仅显示最近 50 个")

    output_path = Path(__file__).parent / "wave_catcher_base_break.csv"
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
