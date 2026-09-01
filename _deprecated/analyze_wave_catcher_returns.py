#!/usr/bin/env python3
"""
形态后续收益分析（通用）

读取 wave_catcher 系列捕捉器的输出 CSV，以形态结束日收盘价为基准，
统计之后 60 / 90 / 120 个交易日的涨跌分布。

支持两种形态（--mode）：
  - consolidation : 横盘形态（wave_catcher_consolidation.csv）
        基准日 = 横盘结束日 end_date，基准价 = end_price（结束日收盘价）
  - spike         : 尖顶形态（wave_catcher_spike_top.csv）
        基准日 = 腰斩日 half_date，基准价 = half_price（腰斩日收盘价）

统计指标（每个持有期 H ∈ {60, 90, 120}）：
  - 终点收益：样本数、均值、中位数、标准差、胜率、P25/P75、最大/最小、
    平均正收益、平均负收益、盈亏比、大涨率(≥20%)、大跌率(≤-20%)
  - 期间最大涨幅 / 最大跌幅（持有期内最高价 / 最低价相对基准价）：均值、中位数

用法:
    python3 analyze_wave_catcher_returns.py --mode consolidation
    python3 analyze_wave_catcher_returns.py --mode spike --horizons 60,90,120
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_DIR = Path.home() / "Documents" / "mainland_data_2014"

MODES = {
    'consolidation': {
        'input': 'wave_catcher_consolidation.csv',
        'date_col': 'end_date',
        'base_col': 'end_price',
        'label': '横盘形态',
        'base_desc': '横盘结束日收盘价',
    },
    'spike': {
        'input': 'wave_catcher_spike_top.csv',
        'date_col': 'half_date',
        'base_col': 'half_price',
        'label': '尖顶形态',
        'base_desc': '腰斩日收盘价',
    },
}


def load_stock(code: str) -> pd.DataFrame | None:
    """加载单只股票日线数据"""
    fp = DATA_DIR / f"{code}.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception:
        return None


def compute_stats(vals: pd.Series) -> dict | None:
    """对一列收益做常见涨跌统计。"""
    vals = vals.dropna()
    if len(vals) == 0:
        return None
    pos = vals[vals > 0]
    neg = vals[vals < 0]
    avg_pos = float(pos.mean()) if len(pos) else np.nan
    avg_neg = float(neg.mean()) if len(neg) else np.nan
    plr = avg_pos / abs(avg_neg) if (len(neg) and not np.isnan(avg_neg) and avg_neg != 0) else np.nan
    return {
        'n': int(len(vals)),
        'mean': float(vals.mean()),
        'median': float(vals.median()),
        'std': float(vals.std()),
        'win_rate': float((vals > 0).sum() / len(vals) * 100),
        'p25': float(vals.quantile(0.25)),
        'p75': float(vals.quantile(0.75)),
        'max': float(vals.max()),
        'min': float(vals.min()),
        'avg_pos': avg_pos,
        'avg_neg': avg_neg,
        'plr': plr,
        'up20': float((vals >= 20).sum() / len(vals) * 100),
        'dn20': float((vals <= -20).sum() / len(vals) * 100),
    }


def main():
    mode = 'consolidation'
    horizons = [60, 90, 120]
    input_name = None

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--mode' and i + 1 < len(argv):
            mode = argv[i + 1]; i += 2
        elif argv[i] == '--horizons' and i + 1 < len(argv):
            horizons = [int(x) for x in argv[i + 1].split(',') if x.strip()]; i += 2
        elif argv[i] == '--input' and i + 1 < len(argv):
            input_name = argv[i + 1]; i += 2
        else:
            i += 1

    if mode not in MODES:
        print(f"未知 mode: {mode}，可选: {list(MODES)}")
        sys.exit(1)

    cfg = MODES[mode]
    date_col = cfg['date_col']
    base_col = cfg['base_col']
    input_name = input_name or cfg['input']

    input_path = Path(__file__).parent / input_name
    if not input_path.exists():
        print(f"未找到输入文件: {input_path}")
        print(f"请先运行 {mode} 对应的捕捉脚本生成形态清单。")
        sys.exit(1)

    signals = pd.read_csv(input_path, encoding='utf-8-sig')
    signals['code'] = signals['code'].astype(str)
    total = len(signals)

    print("=" * 72)
    print(f"  形态后续收益分析 —— {cfg['label']}")
    print(f"  输入: {input_path.name} | 共 {total} 个形态")
    print(f"  基准: {cfg['base_desc']} | 持有期: {', '.join(str(h) for h in horizons)} 交易日")
    print("=" * 72)

    records = []
    # 按 code 分组，减少重复加载
    for code, group in signals.groupby('code'):
        df = load_stock(code)
        if df is None or df.empty:
            continue
        date_to_idx = {d: k for k, d in enumerate(df['date'].values)}
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        n = len(close)

        for _, row in group.iterrows():
            base = float(row[base_col])
            if base <= 0:
                continue
            end_idx = date_to_idx.get(pd.Timestamp(row[date_col]).to_datetime64())
            if end_idx is None:
                continue

            rec = {'code': code, 'base_date': str(row[date_col]), 'base_price': round(base, 2)}
            for h in horizons:
                tgt = end_idx + h
                if tgt < n:
                    rec[f'ret_{h}d'] = round((close[tgt] / base - 1) * 100, 2)
                    seg_hi = high[end_idx + 1:end_idx + h + 1]
                    seg_lo = low[end_idx + 1:end_idx + h + 1]
                    rec[f'maxup_{h}d'] = round((seg_hi.max() / base - 1) * 100, 2)
                    rec[f'maxdn_{h}d'] = round((seg_lo.min() / base - 1) * 100, 2)
                else:
                    rec[f'ret_{h}d'] = np.nan
                    rec[f'maxup_{h}d'] = np.nan
                    rec[f'maxdn_{h}d'] = np.nan
            records.append(rec)

    if not records:
        print("\n  无有效样本（可能是日期无法匹配或数据不足）。")
        return

    res = pd.DataFrame(records)

    # ============================================================
    # 主表：终点收益统计
    # ============================================================
    print(f"\n  有效样本: {len(res)} 个\n")
    print("  【终点收益分布】（第 N 个交易日收盘价相对基准价）")
    print(f"{'持有期':<8} {'样本':>6} {'均值':>9} {'中位':>9} {'标准差':>8} {'胜率':>7} "
          f"{'P25':>8} {'P75':>8} {'最大':>9} {'最小':>9}")
    print("-" * 88)
    for h in horizons:
        col = f'ret_{h}d'
        s = compute_stats(res[col])
        if s is None:
            print(f"{str(h)+'日':<8} {'0':>6}  (无足够未来数据)")
            continue
        print(f"{str(h)+'日':<8} {s['n']:>6} {s['mean']:>8.2f}% {s['median']:>8.2f}% "
              f"{s['std']:>7.2f}% {s['win_rate']:>6.1f}% {s['p25']:>7.2f}% {s['p75']:>7.2f}% "
              f"{s['max']:>8.1f}% {s['min']:>8.1f}%")

    print(f"\n  【盈亏结构】")
    print(f"{'持有期':<8} {'平均正收益':>10} {'平均负收益':>10} {'盈亏比':>8} "
          f"{'大涨≥20%':>10} {'大跌≤-20%':>10}")
    print("-" * 62)
    for h in horizons:
        col = f'ret_{h}d'
        s = compute_stats(res[col])
        if s is None:
            continue
        avg_pos = f"{s['avg_pos']:.2f}%" if not np.isnan(s['avg_pos']) else "—"
        avg_neg = f"{s['avg_neg']:.2f}%" if not np.isnan(s['avg_neg']) else "—"
        plr = f"{s['plr']:.2f}" if not np.isnan(s['plr']) else "—"
        print(f"{str(h)+'日':<8} {avg_pos:>10} {avg_neg:>10} {plr:>8} "
              f"{s['up20']:>9.1f}% {s['dn20']:>9.1f}%")

    # ============================================================
    # 附加表：期间最大涨幅 / 最大跌幅
    # ============================================================
    print(f"\n  【持有期内最大涨幅 / 最大跌幅】（区间最高价 / 最低价相对基准价）")
    print(f"{'持有期':<8} {'最大涨幅均值':>12} {'最大涨幅中位':>12} "
          f"{'最大跌幅均值':>12} {'最大跌幅中位':>12}")
    print("-" * 56)
    for h in horizons:
        up = res[f'maxup_{h}d'].dropna()
        dn = res[f'maxdn_{h}d'].dropna()
        if len(up) == 0:
            continue
        print(f"{str(h)+'日':<8} {up.mean():>11.2f}% {up.median():>11.2f}% "
              f"{dn.mean():>11.2f}% {dn.median():>11.2f}%")

    output_path = Path(__file__).parent / f"wave_catcher_{mode}_returns.csv"
    res = res.sort_values('base_date', ascending=False).reset_index(drop=True)
    res.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n明细已保存至: {output_path}")


if __name__ == "__main__":
    main()
