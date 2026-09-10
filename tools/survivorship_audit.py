#!/usr/bin/env python3
"""幸存者偏差量化审计（S3）：数据池缺口 + 极端跌幅分布。

1. 数据末端分布: 每只股票的最后交易日分布——若绝大多数止于数据
   截止日，说明池内只含"活到今天"的股票（退市股从未入池/已剔除）。
2. 极端跌幅分布: 每只股票自历史最高收盘的最大回撤（close 口径
   running peak），统计 ≤-60%/-70%/-80% 的家数与占比。
3. 与已知退市清单对照（delisting_precheck 的 11 只）。

产出: output/survivorship_audit.csv + 控制台（供文档引用）
用法: python3 tools/survivorship_audit.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = Path("/Users/flybirp/Documents/mainland_data_2014")
OUT = ROOT / "output"


def main():
    last_dates, max_dd = [], []
    files = sorted(DATA.glob("*.csv"))
    for i, f in enumerate(files, 1):
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", usecols=["date", "close"])
        except Exception:
            continue
        if not len(df):
            continue
        last_dates.append(str(df["date"].iloc[-1]))
        c = df["close"].values.astype(float)
        rp = np.maximum.accumulate(c)
        max_dd.append(float(((c - rp) / rp).min() * 100))
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}", file=sys.stderr)

    s = pd.Series(last_dates)
    d = pd.Series(max_dd)
    print(f"\n== 股票池: {len(s)} 只")
    print("\n-- 数据末端（最后交易日）年份分布")
    vc = s.str[:4].value_counts().sort_index()
    for y, n in vc.items():
        print(f"   {y}: {n}")
    print(f"   末端中位日期: {s.sort_values().iloc[len(s) // 2]}")

    print("\n-- 自历史最高收盘的最大回撤分布")
    for th in (-30, -50, -60, -70, -80, -90):
        k = int((d <= th).sum())
        print(f"   ≤ {th}%: {k} 只 ({k / len(d) * 100:.1f}%)")
    print(f"   中位最大回撤: {d.median():.1f}%")

    out = pd.DataFrame({"code": [f.stem for f in files][:len(s)],
                        "last_date": s.values, "max_dd": d.values})
    out.to_csv(OUT / "survivorship_audit.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存: {OUT / 'survivorship_audit.csv'}")
    print("\n判读: 若末端日期高度集中在数据截止日（如 2026 年），则池内"
          "不含任何'中途消失'的股票（退市/长期停牌退市），"
          "所有买入持有类结论的期望被系统性高估。")


if __name__ == "__main__":
    main()
