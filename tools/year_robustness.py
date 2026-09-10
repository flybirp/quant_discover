#!/usr/bin/env python3
"""分年稳健性统一组件（全项目复用，纪律 5：同一能力只实现一次）。

三项标准量:
    1. 分年效应表      每年 n / E / 胜率（口径附录一）
    2. 方向稳定性      效应符号一致年份占比 + 二项检验 p 值
                       （H0: 各年方向随机，p<0.05 = 方向稳定）
    3. 聚类贡献指数    = 1 - |剔除最大样本年后的效应| / |全样本效应|
                       >0.5 标注"聚类驱动"（效应主要由单一年份制造）

被 holding_analysis / top_profile / exit_report / rps_check 调用。

用法（库）:
    from tools.year_robustness import year_table, stability, cluster_index
    print(year_table(df, "ret_60d"))          # 需 df 含 base_date 或 year 列
    print(stability(year_table(df, "ret_60d"), "E"))
    print(cluster_index(df, "group", "ret_60d"))
独立运行（示例）:
    python3 tools/year_robustness.py output/an_spike.csv --metric ret_90d
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats


def _year_of(df: pd.DataFrame, year_col: str | None) -> pd.Series:
    if year_col and year_col in df.columns:
        return df[year_col].astype(str).str[:4]
    for c in ("year", "base_date", "date"):
        if c in df.columns:
            return df[c].astype(str).str[:4]
    raise KeyError("未找到年份列（year / base_date / date）")


def year_table(df: pd.DataFrame, metric: str,
               year_col: str | None = None) -> pd.DataFrame:
    """分年效应表：每年 n / E / 中位 / 胜率（胜率口径 ret>0，附录一）。"""
    d = pd.DataFrame({"y": _year_of(df, year_col),
                      "m": pd.to_numeric(df[metric], errors="coerce")})
    d = d.dropna()
    rows = []
    for y, s in d.groupby("y"):
        if not len(s):
            continue
        rows.append({"year": y, "n": len(s),
                     "E": round(s["m"].mean(), 2),
                     "median": round(s["m"].median(), 2),
                     "win": round((s["m"] > 0).mean() * 100, 1)})
    return pd.DataFrame(rows)


def stability(tbl: pd.DataFrame, col: str = "E",
              ref: float = 0.0) -> dict:
    """方向稳定性：效应相对 ref 的符号一致占比 + 二项检验。

    返回 {n_years, consistent, ratio, p_value, direction}
    p<0.05 视为方向稳定（各年同号非随机）。
    """
    v = tbl[col].astype(float)
    sign = np.sign(v - ref)
    sign = sign[sign != 0]
    n = len(sign)
    if n == 0:
        return {"n_years": 0, "consistent": 0, "ratio": np.nan,
                "p_value": np.nan, "direction": "—"}
    pos = int((sign > 0).sum())
    k = max(pos, n - pos)
    # 双侧二项检验：H0 = 方向随机（p=0.5）
    p = stats.binomtest(k, n, 0.5, alternative="greater").pvalue * 2
    p = min(1.0, float(p))
    return {"n_years": n, "consistent": k, "ratio": round(k / n, 3),
            "p_value": round(p, 4),
            "direction": "positive" if pos >= n - pos else "negative"}


def cluster_index(df: pd.DataFrame, group_col: str | None, metric: str,
                  year_col: str | None = None) -> dict:
    """聚类贡献指数：剔除最大样本年后效应的衰减比例。

    用法场景：分组对比（如大顶 vs 假顶）的效应量在剔除聚集年后
    是否归零。输入含分组列时，效应 = 组间 E 之差（第一组 − 最后一组）。
    """
    y = _year_of(df, year_col)
    d = df.assign(_y=y)
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric])

    if group_col and group_col in d.columns:
        g = d.groupby(group_col)[metric].mean()
        eff_all = float(g.iloc[0] - g.iloc[-1])
        top_year = d["_y"].value_counts().idxmax()
        gx = d[d["_y"] != top_year].groupby(group_col)[metric].mean()
        eff_ex = float(gx.iloc[0] - gx.iloc[-1]) if len(gx) > 1 else np.nan
    else:
        eff_all = float(d[metric].mean())
        top_year = d["_y"].value_counts().idxmax()
        eff_ex = float(d[d["_y"] != top_year][metric].mean())

    idx = (1 - abs(eff_ex) / abs(eff_all)) if eff_all else np.nan
    return {"effect_all": round(eff_all, 3),
            "top_year": str(top_year),
            "effect_ex_top_year": round(eff_ex, 3)
            if not pd.isna(eff_ex) else np.nan,
            "cluster_index": round(idx, 3) if not pd.isna(idx) else np.nan,
            "verdict": ("聚类驱动" if (not pd.isna(idx) and idx > 0.5)
                        else "非聚类驱动")}


def report(df: pd.DataFrame, metric: str, group_col: str | None = None,
           year_col: str | None = None) -> None:
    """打印三段标准报告（分年表 + 方向稳定性 + 聚类贡献指数）。"""
    t = year_table(df, metric, year_col)
    print(f"-- 分年效应（{metric}）")
    print(t.to_string(index=False))
    st = stability(t, "E")
    print(f"\n-- 方向稳定性: {st['consistent']}/{st['n_years']} 年同向"
          f"（{st['direction']}），一致率 {st['ratio']}，"
          f"二项检验 p={st['p_value']}"
          f"{'  ✅方向稳定' if st['p_value'] < 0.05 else '  ⚠方向不稳定'}")
    ci = cluster_index(df, group_col, metric, year_col)
    print(f"\n-- 聚类贡献: 全样本效应 {ci['effect_all']} | "
          f"剔除最大样本年({ci['top_year']})后 {ci['effect_ex_top_year']} | "
          f"指数 {ci['cluster_index']} → {ci['verdict']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--metric", required=True)
    ap.add_argument("--group", default=None)
    ap.add_argument("--year-col", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.input, encoding="utf-8-sig", low_memory=False)
    report(df, a.metric, a.group, a.year_col)


if __name__ == "__main__":
    main()
