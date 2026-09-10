#!/usr/bin/env python3
"""买入信号参数甜点扫描（通用，四信号复用；纪律 5 同一能力只实现一次）。

对每个候选参数（连续列按分位分组、离散列按取值分组）输出：
    各组 n / E / 中位 / 胜率 / 平均MaxDD
    Spearman IC（参数 vs 收益）
    **方向稳定性 p（二项检验）与聚类贡献指数**（year_robustness 组件）
采用标准（30/31 篇确立）：组间差距大 + 方向 p<0.05 + 聚类指数 ≤0.5。

用法:
    python3 tools/buyparam_scan.py output/an_wyckoff_spring.csv \
        --metric ret_60d --params spring_depth_pct --query "spring_depth_pct>=5"
    python3 tools/buyparam_scan.py output/an_base_break_days.csv \
        --metric ret_20d --params penetration_pct --merge \
        output/catch_base_break.csv:penetration_pct,base_weeks,prior_decline_pct
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.year_robustness import year_table, stability, cluster_index  # noqa


def load(fp: str, query: str | None, merge_spec: str | None,
         metric: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / fp, encoding="utf-8-sig", dtype={"code": str},
                    low_memory=False)
    if merge_spec:
        mfp, cols = merge_spec.split(":")
        m = pd.read_csv(ROOT / mfp, encoding="utf-8-sig", dtype={"code": str},
                        low_memory=False, usecols=["code", "base_date"] + cols.split(","))
        # 同一 (code, base_date) 可能存在多个形态实例：按出现序编号后
        # 一一对应，避免笛卡尔积膨胀
        d["_i"] = d.groupby(["code", "base_date"]).cumcount()
        m["_i"] = m.groupby(["code", "base_date"]).cumcount()
        n0 = len(d)
        d = d.merge(m, on=["code", "base_date", "_i"], how="left").drop(columns="_i")
        if len(d) != n0:
            print(f"  ⚠ 合并后行数变化 {n0} -> {len(d)}（检查实例对应）")
    if query:
        d = d.query(query)
    return d.dropna(subset=[metric])


def scan(d: pd.DataFrame, param: str, metric: str, bins: int) -> None:
    s = d[[param, metric]].replace([np.inf, -np.inf], np.nan).dropna()
    if not len(s):
        print(f"  （{param} 无有效样本）")
        return
    ic = s[param].corr(s[metric], method="spearman")
    uniq = s[param].nunique()
    if uniq <= 6:                       # 离散列：按取值分组
        groups = [(str(v), g[metric]) for v, g in s.groupby(param)]
        labels = [g[0] for g in groups]
    else:                               # 连续列：分位分组（箱数可能因去重减少）
        q = pd.qcut(s[param], bins, duplicates="drop")
        cats = list(q.cat.categories)
        groups = [(str(g), s[metric][q == g]) for g in cats]
        labels = [f"Q{i+1} ({g.left:.2f}~{g.right:.2f}]" if uniq > 6 else str(g)
                  for i, g in enumerate(cats)]

    print(f"\n-- 参数 {param}  (n={len(s)}, IC={ic:+.3f}, 唯一值 {uniq})")
    print(f"{'组':<26} {'n':>6} {'E':>8} {'中位':>8} {'胜率':>6} "
          f"{'MaxDD':>8} {'方向p':>7} {'聚类':>6} {'判定':>10}")
    print("-" * 92)
    for (_, vals), lab in zip(groups, labels):
        sub = d.loc[vals.index]
        md = sub[f"maxdn_{metric.replace('ret_', '')}"] \
            if f"maxdn_{metric.replace('ret_', '')}" in sub.columns else None
        st = stability(year_table(sub, metric), "E")
        ci = cluster_index(sub, None, metric)
        print(f"{lab:<26} {len(vals):>6} {vals.mean():>+7.2f} "
              f"{vals.median():>+7.2f} {(vals > 0).mean() * 100:>5.1f}% "
              f"{md.mean() if md is not None else float('nan'):>+7.2f}% "
              f"{st['p_value']:>7.3f} {ci['cluster_index']:>6} "
              f"{ci['verdict']:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--metric", default="ret_60d")
    ap.add_argument("--params", required=True, help="逗号分隔的列名")
    ap.add_argument("--query", default=None)
    ap.add_argument("--merge", default=None,
                    help="额外明细:列1,列2（按 code+base_date 合并）")
    ap.add_argument("--bins", type=int, default=3)
    a = ap.parse_args()

    d = load(a.input, a.query, a.merge, a.metric)
    print(f"== {a.input} | metric={a.metric} | n={len(d)}"
          f"{' | query: ' + a.query if a.query else ''}")
    if a.merge:
        print(f"   合并列: {a.merge.split(':')[1]}")
    for p in a.params.split(","):
        if p not in d.columns:
            print(f"  ⏭ 跳过 {p}（列不存在）")
            continue
        scan(d, p, a.metric, a.bins)
    print("\n采用标准: 组间 E 单调且差距大 + 方向 p<0.05 + 聚类指数 ≤0.5")


if __name__ == "__main__":
    main()
