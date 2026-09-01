#!/usr/bin/env python3
"""
快照对比工具（语义等价验证）

对比"旧版重跑基线"(_baseline/catch_*.csv)与"新引擎输出"(output/catch_*.csv)
的关键列：形态集合应一致（按 keys 匹配），数值列允许容差。

用法:
    python3 tools/verify.py --preset catch_zfzs          # 按 preset 对比
    python3 tools/verify.py --all                        # 对比全部 preset
    python3 tools/verify.py --old A.csv --new B.csv \
        --keys code,entry_date --values return_pct,exit_date   # 通用模式

preset 中的豁免说明:
    - catch_base_break:     豁免 ret_1w~4w / max_up_4w / max_dd_4w / recover_base
                            （forward 数据剥离到 analyze；recover_base 由
                             cross_above(base_low) 复现）；新表多 base_price
    - catch_consol_scanner: 豁免 direction / max_return / max_drawdown / final_return
                            （突破三分类废除，改 analyze 连续指标）
    - catch_spike_top:      周线单顶从 5 日块改为真自然周（有意行为升级），
                            差异单列报告，默认 --lenient 不阻塞
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "_baseline"
NEW_DIR = ROOT / "output"

# 每个 preset: keys=形态唯一键, values=必须一致的数值/类别列,
# ignore=新旧列集合允许不同的列, lenient=预期有差异(只报告不阻塞)
PRESETS = {
    "catch_zfzs": dict(
        keys=["code", "entry_date"],
        values=["exit_date", "duration_days", "entry_price", "peak_price",
                "exit_price", "return_pct", "max_drawdown_pct"],
    ),
    "catch_volume": dict(
        keys=["code", "entry_date"],
        values=["exit_date", "duration_days", "entry_price", "peak_price",
                "exit_price", "return_pct", "max_drawdown_pct", "exit_reason"],
    ),
    "catch_oversold": dict(
        keys=["code", "bottom_date"],
        values=["peak_before_date", "peak_before_price", "bottom_price",
                "decline_pct", "bounce_date", "bounce_price", "bounce_pct",
                "decay_days", "recover_days"],
    ),
    "catch_base_break": dict(
        keys=["code", "break_date"],
        values=["group", "position_type", "base_end_date", "base_weeks",
                "base_low", "base_range_pct", "prior_decline_pct",
                "break_low", "penetration_pct", "volume_ratio", "volume_type"],
        ignore=["ret_1w", "ret_2w", "ret_3w", "ret_4w",
                "max_up_4w", "max_dd_4w", "recover_base"],
        note="剥离 forward 列；新表应新增 base_price",
    ),
    "catch_consolidation": dict(
        keys=["code", "start_date", "end_date"],
        values=["duration_days", "avg_price", "high_price", "low_price",
                "up_pct", "down_pct", "range_pct", "start_price",
                "end_price", "trend_pct"],
    ),
    "catch_spike_top": dict(
        keys=["code", "peak_date", "half_date"],
        values=["low_date", "low_price", "peak_price", "half_price",
                "up_pct", "down_pct", "up_days", "down_days", "total_days"],
        lenient=True,
        note="周线单顶 5日块->真自然周，预期有差异，仅报告",
    ),
    "catch_main_dd50": dict(
        keys=["code", "start_date"],
        values=["end_date", "duration_days", "wave_start_close",
                "peak_close", "wave_end_close", "return_pct",
                "max_drawdown_pct"],
    ),
    "catch_main_trendline_zhixing_slow": dict(
        keys=["code", "start_date"],
        values=["end_date", "duration_days", "start_close", "end_close",
                "return_pct"],
    ),
    "catch_main_trendline_ma60": dict(
        keys=["code", "start_date"],
        values=["end_date", "duration_days", "start_close", "end_close",
                "return_pct"],
    ),
    "catch_flat_zone": dict(
        keys=["code", "flat_start_date", "flat_end_date"],
        values=["accel_start_date", "drop_high_price", "decline_pct",
                "avg_daily_ret_before", "avg_daily_ret_accel", "accel_ratio",
                "flat_days", "flat_avg_price", "flat_range_pct"],
    ),
    "catch_consol_scanner": dict(
        keys=["code", "start_date", "end_date"],
        values=["duration_days", "range_pct", "trend_pct", "median_close",
                "max_price", "min_price"],
        ignore=["direction", "max_return", "max_drawdown", "final_return"],
        note="突破三分类废除；21 个特征列是否透传在任务10确认后再加入 values",
    ),
}

NUM_TOL = 0.02  # 价格/百分比绝对容差


def _load(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, encoding="utf-8-sig", dtype={"code": str})
    if "code" in df.columns:
        # 兼容 pandas 将 code 读成 "600000.0" 的情况
        df["code"] = df["code"].str.replace(r"\.0$", "", regex=True)
    return df


def compare(old_fp: Path, new_fp: Path, keys: list[str], values: list[str],
            tol: float = NUM_TOL, show: int = 5) -> bool:
    """返回 True=通过（或 lenient 场景下仅报告）。"""
    old = _load(old_fp)
    new = _load(new_fp)

    for col in keys + values:
        if col not in old.columns:
            print(f"  [FAIL] 旧表缺少列: {col}")
            return False
        if col not in new.columns:
            print(f"  [FAIL] 新表缺少列: {col}")
            return False

    dup_o = old.duplicated(subset=keys).sum()
    dup_n = new.duplicated(subset=keys).sum()
    if dup_o or dup_n:
        print(f"  [WARN] keys 重复: 旧 {dup_o} 行 / 新 {dup_n} 行（对比按去重后首行）")
        old = old.drop_duplicates(subset=keys, keep="first")
        new = new.drop_duplicates(subset=keys, keep="first")

    m = old[keys + values].merge(
        new[keys + values], on=keys, how="outer",
        suffixes=("_old", "_new"), indicator=True)

    old_only = m[m["_merge"] == "left_only"]
    new_only = m[m["_merge"] == "right_only"]
    both = m[m["_merge"] == "both"]

    mismatches = {}
    for col in values:
        if col in keys:
            continue
        if pd.api.types.is_numeric_dtype(both[f"{col}_old"]):
            diff = (both[f"{col}_old"] - both[f"{col}_new"]).abs()
            bad = both[diff > tol]
            # NaN vs NaN 视为一致
            bad = bad[~(both[f"{col}_old"].isna() & both[f"{col}_new"].isna())
                      .reindex(bad.index, fill_value=False)]
        else:
            eq = (both[f"{col}_old"].astype(str) == both[f"{col}_new"].astype(str)) | \
                 (both[f"{col}_old"].isna() & both[f"{col}_new"].isna())
            bad = both[~eq]
        if len(bad):
            mismatches[col] = bad

    ok = True
    if len(old_only):
        ok = False
        print(f"  [MISS] 仅旧表有: {len(old_only)} 行（应在新引擎出现而未出现）")
        print(old_only[keys].head(show).to_string(index=False))
    if len(new_only):
        ok = False
        print(f"  [EXTRA] 仅新表有: {len(new_only)} 行（新引擎多出的形态）")
        print(new_only[keys].head(show).to_string(index=False))
    for col, bad in mismatches.items():
        ok = False
        print(f"  [DIFF] {col}: {len(bad)} 行不一致")
        print(bad[keys + [f"{col}_old", f"{col}_new"]].head(show).to_string(index=False))

    if ok:
        print(f"  [PASS] {len(both)} 个形态, {len(values)} 个值列全部一致 (tol={tol})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--old")
    ap.add_argument("--new")
    ap.add_argument("--keys", help="逗号分隔")
    ap.add_argument("--values", help="逗号分隔")
    ap.add_argument("--tol", type=float, default=NUM_TOL)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    if args.all:
        all_ok = True
        for name, spec in PRESETS.items():
            old_fp = BASELINE_DIR / f"{name}.csv"
            new_fp = NEW_DIR / f"{name}.csv"
            print(f"\n=== {name} ===")
            if not old_fp.exists():
                print(f"  [SKIP] 基线不存在: {old_fp}")
                continue
            if not new_fp.exists():
                print(f"  [SKIP] 新输出不存在: {new_fp}")
                all_ok = False
                continue
            if spec.get("note"):
                print(f"  (note) {spec['note']}")
            ok = compare(old_fp, new_fp, spec["keys"], spec["values"],
                         args.tol, args.show)
            if not ok:
                if spec.get("lenient"):
                    print("  [LENIENT] 该形态为有意行为升级，差异不阻塞")
                else:
                    all_ok = False
        print("\n==== 总结:", "全部通过" if all_ok else "存在阻塞差异", "====")
        sys.exit(0 if all_ok else 1)
        return

    if args.preset:
        spec = PRESETS[args.preset]
        old_fp = BASELINE_DIR / f"{args.preset}.csv"
        new_fp = NEW_DIR / f"{args.preset}.csv"
        if args.old:
            old_fp = Path(args.old)
        if args.new:
            new_fp = Path(args.new)
        if spec.get("note"):
            print(f"  (note) {spec['note']}")
        ok = compare(old_fp, new_fp, spec["keys"], spec["values"], args.tol, args.show)
        if not ok and spec.get("lenient"):
            print("  [LENIENT] 有意行为升级，不阻塞")
            ok = True
        sys.exit(0 if ok else 1)

    if args.old and args.new and args.keys:
        values = args.values.split(",") if args.values else []
        ok = compare(Path(args.old), Path(args.new),
                     args.keys.split(","), values, args.tol, args.show)
        sys.exit(0 if ok else 1)

    ap.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
