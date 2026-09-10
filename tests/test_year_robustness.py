#!/usr/bin/env python3
"""year_robustness 单测（合成数据）。运行: python3 tests/test_year_robustness.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.year_robustness import year_table, stability, cluster_index


def mk(years, vals):
    return pd.DataFrame({"base_date": [f"{y}-01-05" for y in years],
                         "m": vals})


def test_year_table():
    df = mk([2020, 2020, 2021], [1.0, 3.0, -2.0])
    t = year_table(df, "m")
    assert list(t["year"]) == ["2020", "2021"]
    assert t["n"].tolist() == [2, 1]
    assert t["E"].tolist() == [2.0, -2.0]
    assert t["win"].tolist() == [100.0, 0.0]


def test_stability():
    # 全正向 -> 3/3 同向，二项 p = 2*(0.5^3) = 0.25（n小，未达显著）
    t = year_table(mk([2020, 2021, 2022], [1.0, 2.0, 3.0]), "m")
    st = stability(t, "E")
    assert st["n_years"] == 3 and st["consistent"] == 3
    assert st["direction"] == "positive"
    assert abs(st["p_value"] - 0.25) < 1e-6
    # 多年同向 -> 显著
    t2 = year_table(
        mk([str(2000 + i) for i in range(10)], [1.0] * 10), "m")
    assert stability(t2, "E")["p_value"] < 0.05
    # 方向摇摆
    t3 = year_table(
        mk([str(2000 + i) for i in range(10)], [1, -1] * 5), "m")
    assert stability(t3, "E")["ratio"] == 0.5


def test_cluster_index():
    # 构造：2020 占 80% 样本且效应全部来自 2020
    y = ["2020"] * 80 + ["2021"] * 10 + ["2022"] * 10
    v = [10.0] * 80 + [0.0] * 20
    df = pd.DataFrame({"base_date": [f"{a}-01-05" for a in y], "m": v})
    ci = cluster_index(df, None, "m")
    assert ci["top_year"] == "2020"
    assert ci["cluster_index"] > 0.5 and ci["verdict"] == "聚类驱动"
    # 效应均匀 -> 非聚类驱动
    y2 = ["2020"] * 40 + ["2021"] * 30 + ["2022"] * 30
    v2 = [5.0] * 100
    df2 = pd.DataFrame({"base_date": [f"{a}-01-05" for a in y2], "m": v2})
    ci2 = cluster_index(df2, None, "m")
    assert ci2["cluster_index"] == 0.0 and ci2["verdict"] == "非聚类驱动"
    # 分组版：A 组 vs B 组之差
    g = ["A"] * 50 + ["B"] * 50
    df3 = pd.DataFrame({"base_date": [f"{a}-01-05" for a in y2],
                        "g": g, "m": [10.0] * 50 + [0.0] * 50})
    ci3 = cluster_index(df3, "g", "m")
    assert ci3["effect_all"] == 10.0


if __name__ == "__main__":
    for fn in (test_year_table, test_stability, test_cluster_index):
        fn()
        print(f"  PASS {fn.__name__}")
    print("all green")
