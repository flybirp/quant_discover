#!/usr/bin/env python3
"""
横盘平台买点收益分析

读取 find_flat_zone_after_a_main_dropping.py 的输出（flat_zone_after_drop.csv），
以每个横盘区间的平均价格作为买点，从横盘结束日起算，统计之后
30 / 90 / 200 个交易日的收益分布（均值、中位数、胜率等）。

买点价格：横盘区间收盘价均价（flat_avg_price），近似"在横盘期间以均价建仓"。
收益基准日：横盘结束日（flat_end_date）之后第 N 个交易日的收盘价。

用法:
    python3 analyze_flat_zone_returns.py
    python3 analyze_flat_zone_returns.py --input flat_zone_after_drop.csv --horizons 30,90,200
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_DIR = Path.home() / "Documents" / "mainland_data_2014"


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


def main():
    input_name = "flat_zone_after_drop.csv"
    horizons = [30, 90, 200]

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--input' and i + 1 < len(argv):
            input_name = argv[i + 1]; i += 2
        elif argv[i] == '--horizons' and i + 1 < len(argv):
            horizons = [int(x) for x in argv[i + 1].split(',') if x.strip()]; i += 2
        else:
            i += 1

    input_path = Path(__file__).parent / input_name
    if not input_path.exists():
        print(f"未找到输入文件: {input_path}")
        print("请先运行 find_flat_zone_after_a_main_dropping.py 生成横盘平台清单。")
        sys.exit(1)

    signals = pd.read_csv(input_path, encoding='utf-8-sig')
    signals['code'] = signals['code'].astype(str)
    total = len(signals)
    print("=" * 72)
    print(f"  横盘平台买点收益分析")
    print(f"  输入: {input_path.name} | 共 {total} 个横盘平台")
    print(f"  买点: 横盘区间均价 | 持有期: {', '.join(str(h) for h in horizons)} 交易日")
    print("=" * 72)

    records = []
    # 按 code 分组，减少重复加载
    for code, group in signals.groupby('code'):
        df = load_stock(code)
        if df is None or df.empty:
            continue
        # date -> 索引 映射
        date_to_idx = {d: k for k, d in enumerate(df['date'].values)}
        close = df['close'].values.astype(float)
        n = len(close)

        for _, row in group.iterrows():
            flat_end = pd.Timestamp(row['flat_end_date']).to_datetime64()
            end_idx = date_to_idx.get(flat_end)
            if end_idx is None:
                continue

            buy_price = float(row['flat_avg_price'])
            if buy_price <= 0:
                continue

            rec = {
                'code': code,
                'flat_start_date': row['flat_start_date'],
                'flat_end_date': row['flat_end_date'],
                'flat_days': row['flat_days'],
                'buy_price': round(buy_price, 2),
                'decline_pct': row.get('decline_pct', np.nan),
            }
            for h in horizons:
                tgt = end_idx + h
                if tgt < n:
                    rec[f'ret_{h}d'] = round((close[tgt] / buy_price - 1) * 100, 2)
                else:
                    rec[f'ret_{h}d'] = np.nan
            records.append(rec)

    if not records:
        print("\n  无有效样本（可能是日期无法匹配或数据不足）。")
        return

    res = pd.DataFrame(records)

    # ============================================================
    # 汇总统计
    # ============================================================
    print(f"\n  有效样本: {len(res)} 个\n")
    print(f"{'持有期':<10} {'样本':>6} {'平均收益':>10} {'中位收益':>10} "
          f"{'胜率':>8} {'P25':>8} {'P75':>8} {'最大':>9} {'最小':>9}")
    print("-" * 82)
    for h in horizons:
        col = f'ret_{h}d'
        vals = res[col].dropna()
        if len(vals) == 0:
            print(f"{str(h)+'日':<10} {'0':>6}  (无足够未来数据)")
            continue
        win_rate = float((vals > 0).sum() / len(vals) * 100)
        print(f"{str(h)+'日':<10} {len(vals):>6} "
              f"{vals.mean():>9.2f}% {vals.median():>9.2f}% "
              f"{win_rate:>7.1f}% {vals.quantile(0.25):>7.2f}% "
              f"{vals.quantile(0.75):>7.2f}% {vals.max():>8.1f}% {vals.min():>8.1f}%")

    output_path = Path(__file__).parent / "flat_zone_returns.csv"
    res = res.sort_values('flat_end_date', ascending=False).reset_index(drop=True)
    res.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n明细已保存至: {output_path}")


if __name__ == "__main__":
    main()
