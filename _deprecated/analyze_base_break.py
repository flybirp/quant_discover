#!/usr/bin/env python3
"""
周线筑底跌破（"漏了"）深度分组 × 量能条件 甜点策略分析

读取 wave_catcher_base_break.py 的输出（wave_catcher_base_break.csv），
按跌破深度把样本分成互斥四组（跌破点价格相对筑底低点的比例）：

  - A 组（深度 0~5%）：筑底低点×0.95 < 跌破点价格 < 筑底低点
  - B 组（深度 5~10%）：筑底低点×0.90 < 跌破点价格 ≤ 筑底低点×0.95
  - C 组（深度 10~15%）：筑底低点×0.85 < 跌破点价格 ≤ 筑底低点×0.90
  - D 组（深度 15~30%）：筑底低点×0.70 < 跌破点价格 ≤ 筑底低点×0.85

在每组内按跌破日量能条件（放量 / 平量 / 缩量）分组，输出：

  ① 每组内量能三分类 vs 跌破后 1~4 周走势（收益/胜率/分位数）
  ② 甜点策略矩阵：组 × 量能 的 4 周收益 / 胜率 / 样本 完整矩阵
  ③ 分年度稳健性：各组的 4 周收益按年份拆分（识别单一年份行情偏差，
     如 2025 年单边牛市对深跌幅组的影响），并输出"剔除2025"对照
  ④ 各组整体对比（不分量能）

用法:
    python3 analyze_base_break.py
    python3 analyze_base_break.py --input wave_catcher_base_break.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 量能条件与分组展示顺序
VOL_ORDER = ['放量', '平量', '缩量']
GROUP_ORDER = ['A', 'B', 'C', 'D']
GROUP_LABELS = {
    'A': '浅跌破 0~5%',
    'B': '中浅跌破 5~10%',
    'C': '中深跌破 10~15%',
    'D': '深跌破 15~30%',
}


def stat_of(vals: pd.Series) -> dict:
    """计算一组收益序列的统计量"""
    v = vals.dropna()
    n = len(v)
    if n == 0:
        return {'n': 0, 'mean': np.nan, 'median': np.nan,
                'win_rate': np.nan, 'p25': np.nan, 'p75': np.nan}
    return {
        'n': n,
        'mean': float(v.mean()),
        'median': float(v.median()),
        'win_rate': float((v > 0).sum() / n * 100),
        'p25': float(v.quantile(0.25)),
        'p75': float(v.quantile(0.75)),
    }


def print_vol_report(df_group: pd.DataFrame, group_label: str, ret_cols: list[str]) -> list[dict]:
    """打印某组内 量能三分类 vs 四周走势，返回该组汇总行（供导出）"""
    print(f"\n{'=' * 96}")
    print(f"  {group_label} 组（{GROUP_LABELS[group_label]}）内 量能条件 vs 后续四周走势")
    print(f"{'=' * 96}")

    vc = df_group['volume_type'].value_counts()
    print("  量能分布: " + " | ".join(f"{vt} {int(vc.get(vt, 0))}" for vt in VOL_ORDER))

    rows = []
    # 1) 各持有周收益
    print("\n  --- 各持有周收益（相对跌破日收盘价）---\n")
    for col in ret_cols:
        label = col.replace('ret_', '').replace('w', ' 周')
        print(f"  【{label}】")
        print(f"  {'量能':<6} {'样本':>5} {'平均收益':>10} {'中位收益':>10} "
              f"{'胜率':>8} {'P25':>8} {'P75':>8}")
        print("  " + "-" * 62)
        for vt in VOL_ORDER:
            sub = df_group[df_group['volume_type'] == vt][col]
            s = stat_of(sub)
            if s['n'] == 0:
                print(f"  {vt:<6} {'0':>5}  (无样本)")
                continue
            print(f"  {vt:<6} {s['n']:>5} {s['mean']:>9.2f}% {s['median']:>9.2f}% "
                  f"{s['win_rate']:>7.1f}% {s['p25']:>7.2f}% {s['p75']:>7.2f}%")
            rows.append({'group': group_label, 'metric': f'ret_{col}',
                         'volume_type': vt, **s})
        print()

    # 2) 最大涨幅 / 最大回撤
    print("\n  --- 4 周内最大涨幅 / 最大回撤 ---\n")
    print(f"  {'量能':<6} {'样本':>5} {'平均max涨':>10} {'中位max涨':>10} "
          f"{'平均max回撤':>11} {'中位max回撤':>11}")
    print("  " + "-" * 62)
    for vt in VOL_ORDER:
        sub = df_group[df_group['volume_type'] == vt]
        up = stat_of(sub['max_up_4w'])
        dd = stat_of(sub['max_dd_4w'])
        if up['n'] == 0:
            print(f"  {vt:<6} {'0':>5}  (无样本)")
            continue
        print(f"  {vt:<6} {up['n']:>5} {up['mean']:>9.2f}% {up['median']:>9.2f}% "
              f"{dd['mean']:>10.2f}% {dd['median']:>10.2f}%")
        rows.append({'group': group_label, 'metric': 'max_up_4w',
                     'volume_type': vt, **up})
        rows.append({'group': group_label, 'metric': 'max_dd_4w',
                     'volume_type': vt, **dd})

    # 3) 收回筑底低点占比
    print(f"\n  {'量能':<6} {'样本':>5} {'收回筑底低点占比':>16}")
    print("  " + "-" * 32)
    for vt in VOL_ORDER:
        sub = df_group[df_group['volume_type'] == vt]
        n = len(sub)
        if n == 0:
            print(f"  {vt:<6} {'0':>5}  (无样本)")
            continue
        rate = sub['recover_base'].fillna(False).astype(bool).sum() / n * 100
        print(f"  {vt:<6} {n:>5} {rate:>14.1f}%")
        rows.append({'group': group_label, 'metric': 'recover_base_rate',
                     'volume_type': vt, 'n': n, 'mean': rate,
                     'median': np.nan, 'win_rate': np.nan,
                     'p25': np.nan, 'p75': np.nan})

    return rows


def print_sweet_spot(df: pd.DataFrame, rows: list[dict]):
    """甜点策略矩阵：组×量能 的 4 周收益/胜率/样本"""
    print(f"\n{'=' * 96}")
    print("  甜点策略矩阵：组 × 量能（4 周收益 | 胜率 | 样本）")
    print(f"{'=' * 96}\n")
    print(f"  {'组':<16} {'放量':>22} {'平量':>22} {'缩量':>22}")
    print("  " + "-" * 86)
    best = None
    for g in GROUP_ORDER:
        cells = []
        for vt in VOL_ORDER:
            s = stat_of(df[(df['group'] == g) & (df['volume_type'] == vt)]['ret_4w'])
            if s['n'] == 0:
                cells.append(f"{'—':>22}")
            else:
                cells.append(f"{s['mean']:>+7.2f}% | {s['win_rate']:>5.1f}% | {s['n']:>6}")
                if best is None or s['mean'] > best[2]:
                    best = (g, vt, s['mean'])
                rows.append({'group': g, 'metric': 'sweet_ret_4w',
                             'volume_type': vt, **s})
        print(f"  {g + ' ' + GROUP_LABELS[g]:<16} {cells[0]:>22} {cells[1]:>22} {cells[2]:>22}")
    if best:
        print(f"\n  ★ 全矩阵最高 4 周均值: {best[0]} 组 + {best[1]}（+{best[2]:.2f}%）"
              f"（需结合分年度稳健性与样本量判断，见下）")


def print_yearly_stability(df: pd.DataFrame, rows: list[dict]):
    """分年度稳健性：各组×量能 4 周收益按年份拆分 + 剔除2025对照"""
    d = df.copy()
    d['year'] = d['break_date'].str[:4]
    years = sorted(d['year'].dropna().unique())

    print(f"\n{'=' * 96}")
    print("  分年度稳健性：组 × 量能 4 周收益均值（样本数）")
    print(f"{'=' * 96}\n")

    header = f"  {'组合':<18}" + "".join(f"{y:>12}" for y in years) + f"{'剔除2025':>12}"
    print(header)
    print("  " + "-" * (18 + 12 * (len(years) + 1)))
    for g in GROUP_ORDER:
        for vt in VOL_ORDER:
            sub = d[(d['group'] == g) & (d['volume_type'] == vt)]
            if len(sub) == 0:
                continue
            line = f"  {g + '·' + vt:<18}"
            for y in years:
                v = sub[sub['year'] == y]['ret_4w']
                s = stat_of(v)
                line += f"{s['mean']:>+7.2f}({s['n']:>3})" if s['n'] > 0 else f"{'—':>12}"
            v25 = sub[sub['year'] != '2025']['ret_4w']
            s25 = stat_of(v25)
            line += f"{s25['mean']:>+7.2f}%" if s25['n'] > 0 else f"{'—':>12}"
            print(line)
            if s25['n'] > 0:
                rows.append({'group': g, 'metric': 'ret_4w_ex2025',
                             'volume_type': vt, **s25})


def print_group_compare(df: pd.DataFrame, ret_cols: list[str]) -> list[dict]:
    """各组整体对比（不分量能），返回汇总行"""
    print(f"\n{'=' * 96}")
    print("  A / B / C / D 各组整体对比（不分量能）")
    print(f"{'=' * 96}")

    rows = []
    print(f"\n  {'分组':<18} {'样本':>7} ", end='')
    for col in ret_cols:
        print(f"{col.replace('ret_', '').replace('w', '周'):>9}", end='')
    print()
    print("  " + "-" * (25 + 9 * len(ret_cols)))
    for g in GROUP_ORDER:
        sub = df[df['group'] == g]
        if len(sub) == 0:
            continue
        print(f"  {g + ' ' + GROUP_LABELS[g]:<18} {len(sub):>7} ", end='')
        for col in ret_cols:
            s = stat_of(sub[col])
            print(f"{s['mean']:>8.2f}%", end='')
        print()

    print(f"\n  {'分组':<18} {'平均max涨':>10} {'平均max回撤':>11} {'收回低点占比':>14}")
    print("  " + "-" * 58)
    for g in GROUP_ORDER:
        sub = df[df['group'] == g]
        n = len(sub)
        if n == 0:
            continue
        up = stat_of(sub['max_up_4w'])
        dd = stat_of(sub['max_dd_4w'])
        rate = sub['recover_base'].fillna(False).astype(bool).sum() / n * 100
        print(f"  {g + ' ' + GROUP_LABELS[g]:<18} {up['mean']:>9.2f}% {dd['mean']:>10.2f}% {rate:>13.1f}%")
        rows.append({'group': g, 'metric': 'ALL_max_up_4w', 'volume_type': '全部',
                     'n': n, 'mean': up['mean'], 'median': up['median'],
                     'win_rate': np.nan, 'p25': np.nan, 'p75': np.nan})
        rows.append({'group': g, 'metric': 'ALL_max_dd_4w', 'volume_type': '全部',
                     'n': n, 'mean': dd['mean'], 'median': dd['median'],
                     'win_rate': np.nan, 'p25': np.nan, 'p75': np.nan})
        rows.append({'group': g, 'metric': 'ALL_recover_base_rate', 'volume_type': '全部',
                     'n': n, 'mean': rate, 'median': np.nan,
                     'win_rate': np.nan, 'p25': np.nan, 'p75': np.nan})

    return rows


def main():
    input_name = "wave_catcher_base_break.csv"

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--input' and i + 1 < len(argv):
            input_name = argv[i + 1]; i += 2
        else:
            i += 1

    input_path = Path(__file__).parent / input_name
    if not input_path.exists():
        print(f"未找到输入文件: {input_path}")
        print("请先运行 wave_catcher_base_break.py 生成形态清单。")
        sys.exit(1)

    df = pd.read_csv(input_path, encoding='utf-8-sig')
    df['code'] = df['code'].astype(str)
    total = len(df)
    print("=" * 96)
    print("  周线筑底跌破（'漏了'）深度分组 × 量能条件 甜点策略分析")
    print(f"  输入: {input_path.name} | 共 {total} 个形态")
    print("=" * 96)

    # 概览
    print("\n  --- 分组概览 ---")
    for g in GROUP_ORDER:
        sub = df[df['group'] == g]
        n = len(sub)
        if n == 0:
            print(f"    {g} 组（{GROUP_LABELS[g]}）: 0 个")
            continue
        vc_v = sub['volume_type'].value_counts()
        detail = " | ".join(f"{vt} {int(vc_v.get(vt, 0))}" for vt in VOL_ORDER)
        print(f"    {g} 组（{GROUP_LABELS[g]}）: {n} 个  [{detail}]")

    if total == 0:
        print("\n  无样本可分析。")
        return

    # 动态识别 ret 列（1w~4w）
    ret_cols = [c for c in df.columns if c.startswith('ret_') and c.endswith('w')]
    ret_cols = sorted(ret_cols, key=lambda c: int(c.split('_')[1][:-1]))

    summary_rows = []
    # ① 各组内量能对比
    for g in GROUP_ORDER:
        sub = df[df['group'] == g]
        if len(sub) == 0:
            continue
        summary_rows.extend(print_vol_report(sub, g, ret_cols))

    # ② 甜点策略矩阵
    print_sweet_spot(df, summary_rows)

    # ③ 分年度稳健性
    print_yearly_stability(df, summary_rows)

    # ④ 各组整体对比
    summary_rows.extend(print_group_compare(df, ret_cols))

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(__file__).parent / "base_break_analysis.csv"
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n  汇总统计已保存至: {summary_path}")


if __name__ == "__main__":
    main()
