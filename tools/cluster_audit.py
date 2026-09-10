#!/usr/bin/env python3
"""全项目核心结论的去聚类复核（S1b）：批量跑统一组件，产出标注表。

对 docs 各篇的核心分组结论（有可复现明细者）计算:
    方向稳定性（符号一致年份占比 + 二项检验 p）
    聚类贡献指数 = 1 - |剔除最大样本年后的效应| / |全样本效应|（>0.5 聚类驱动）

产出: output/cluster_audit.csv + 控制台表（供 docs/cluster_robustness_audit.md 引用）

用法: python3 tools/cluster_audit.py
"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.year_robustness import year_table, stability, cluster_index  # noqa

OUT = ROOT / "output"

# (文档, 结论, 明细文件, 指标列, 分组列 or None, 过滤条件 or None)
CASES = [
    ("06 腰斩", "90日 E+19.9%", "an_spike.csv", "ret_90d", None, None),
    ("04 C组", "C组x非缩量 20日 +14.8%", "an_base_break_days.csv", "ret_20d",
     None, "group=='C' and volume_type!='缩量'"),
    ("18 底背离", "60日 E+7.78%", "an_pv_divergence_bottom.csv", "ret_60d",
     None, None),
    ("21 深spring", "spring>=5% 60日 +4.45%", "an_wyckoff_spring.csv",
     "ret_60d", None, "spring_depth_pct>=5"),
    ("23 绝望承压", "承压组 60日 +4.75%", "an_despair_entry.csv", "ret_60d",
     None, "seg>=-15 and seg<-5"),
    ("15 consolidation", "RPS>=80 优于 <20（组间差）", "an_consolidation.csv",
     "ret_60d", "g", None),
    ("01 金叉", "金叉入场 60日", "an_zfzs.csv", "ret_60d", None, None),
    ("16 双底", "20日", "an_double_bottom.csv", "ret_20d", None, None),
    ("19 杯柄", "20日", "an_cup_handle.csv", "ret_20d", None, None),
    ("17 缺口", "20日", "an_gap_down.csv", "ret_20d", None, None),
]


def main():
    rows = []
    for doc, claim, fp, metric, gcol, q in CASES:
        p = OUT / fp
        if not p.exists():
            print(f"  ⏭ 跳过（缺 {fp}）")
            continue
        d = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        if q:
            d = d.query(q)
        if gcol:
            if gcol == "g":  # consolidation 需现算 RPS 分位（外部工具产出）
                continue
        if metric not in d.columns or not len(d):
            print(f"  ⏭ 跳过（无列/无样本 {fp}）")
            continue
        t = year_table(d, metric)
        st = stability(t, "E")
        ci = cluster_index(d, None, metric)
        rows.append({
            "文档": doc, "结论": claim, "指标": metric,
            "n": int(t["n"].sum()),
            "全样本E": round(float(pd.to_numeric(d[metric],
                                                errors="coerce").mean()), 2),
            "同向年": f"{st['consistent']}/{st['n_years']}",
            "方向p": st["p_value"],
            "方向稳定": "✅" if st["p_value"] < 0.05 else "⚠️",
            "聚类指数": ci["cluster_index"],
            "剔除年": ci["top_year"],
            "剔除后E": ci["effect_ex_top_year"],
            "判定": ci["verdict"],
        })

    df = pd.DataFrame(rows)
    out = OUT / "cluster_audit.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n== 全项目去聚类复核")
    print(df.to_string(index=False))
    print(f"\n已保存: {out}")
    print("\n判读: 方向p<0.05=各年同向非随机；聚类指数>0.5=效应主要由"
          "单一年份制造（需标注）")


if __name__ == "__main__":
    main()
