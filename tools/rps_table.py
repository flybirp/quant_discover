#!/usr/bin/env python3
"""全市场每日 RPS 分位表（一次性预计算，供持有期状态研究查表）。

RPS 定义（附录三/六）：rel_ret_120d = 个股 120 日收益 − 中证1000
同期收益；RPS = 该交易日全市场对 rel_ret_120d 的截面百分位
（0~99，升序，高=强）。与 15 篇 rps_check 的口径完全一致。

实现（内存优化）:
    阶段1 逐股计算 rel_ret_120d，按交易日分桶（整数索引，不存
    字符串）；阶段2 逐日 rank(pct) 得整数分位；阶段3 写 parquet
    长表 (date, code, rps: uint8)。

用法:
    python3 tools/rps_table.py                 # 默认 zz1000 / 120 日
    python3 tools/rps_table.py --period 60     # 60 日相对收益
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.engine import load_data, DEFAULT_DATA_DIR  # noqa: E402
from core.market import load_index, relative_return  # noqa: E402


def main(period: int, index_name: str, min_days: int):
    data_dir = DEFAULT_DATA_DIR
    files = sorted(data_dir.glob("*.csv"))
    idx = load_index(index_name)
    all_dates = [d.strftime("%Y-%m-%d") for d in idx.index]
    dpos = {d: i for i, d in enumerate(all_dates)}
    n_days = len(all_dates)

    # 阶段1: 按交易日分桶（code_idx, rel）
    buckets = [None] * n_days          # 延迟建列表，省内存
    codes = []
    for ci, f in enumerate(files, 1):
        code = f.stem
        df = load_data(f, min_days)
        if df is None or df.empty:
            continue
        rel = relative_return(df, period=period, index_name=index_name)
        ds = df["date"].dt.strftime("%Y-%m-%d").values
        vals = rel.values
        codes.append(code)
        ci_idx = len(codes) - 1
        for d, v in zip(ds, vals):
            if np.isnan(v):
                continue
            i = dpos.get(d)
            if i is None:
                continue
            if buckets[i] is None:
                buckets[i] = []
            buckets[i].append((ci_idx, float(v)))
        if ci % 1000 == 0:
            print(f"  阶段1 {ci}/{len(files)}", file=sys.stderr)

    # 阶段2: 逐日排名 -> 分位
    rows_date, rows_code, rows_rps = [], [], []
    for i, b in enumerate(buckets):
        if not b or len(b) < 50:      # 截面过小不排名（同 rps_check 口径）
            continue
        arr = np.array([x[1] for x in b], dtype=float)
        cs = [x[0] for x in b]
        s = pd.Series(arr)
        pct = ((s.rank() - 1) / len(s) * 100).to_numpy()
        d = all_dates[i]
        for c, p in zip(cs, pct):
            rows_date.append(d)
            rows_code.append(codes[c])
            rows_rps.append(np.uint8(min(99, int(p))))

    out = pd.DataFrame({"date": rows_date, "code": rows_code,
                        "rps": pd.array(rows_rps, dtype="uint8")})
    fp = ROOT / f"output/rps_daily_{index_name}_{period}d.parquet"
    out.to_parquet(fp, index=False)
    print(f"\n已保存: {fp}")
    print(f"行数 {len(out):,} | 交易日 {out['date'].nunique()} | "
          f"股票 {out['code'].nunique()}")
    print(f"日均截面 {len(out) / out['date'].nunique():.0f} 只")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=int, default=120)
    ap.add_argument("--index", default="zz1000")
    ap.add_argument("--min-days", type=int, default=180)
    a = ap.parse_args()
    main(a.period, a.index, a.min_days)
