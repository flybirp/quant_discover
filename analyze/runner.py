"""统一 analyze 引擎。

契约（零语义）:
  输入 = 某个 catcher 的输出 CSV，必须含 code / base_date / base_price
  （由 pattern YAML 的 contract 自动附加），分组标签与阈值列任意透传。

能力:
  - forward 收益: day 单位（第 N 个交易日收盘）与 week 单位（真自然周:
    基准日所在 ISO 周为第 0 周(残周)，第 N 周 = 之后第 N 个完整自然周，
    取该周最后一个交易日收盘；整周无交易日记 NaN，不顺延）
  - 期间极值: maxup / maxdn（窗口含基准日）
  - 事件: cross_above(ref_col)（窗口内最高收盘 >= 阈值列）
  - 任意 group_cols 分组统计 + 分年度稳健性（可剔除指定年份）
"""

from pathlib import Path

import numpy as np
import pandas as pd

from core.engine import load_data, DEFAULT_DATA_DIR, OUTPUT_DIR
from analyze.stats import compute_stats, rank_ic

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# horizon 解析
# ------------------------------------------------------------

def parse_horizons(config: dict) -> list[dict]:
    """返回 [{'kind': 'day'|'week', 'n': int, 'col': 'ret_30d'|'ret_1w'}, ...]"""
    hs = config.get("horizons")
    out = []
    if hs is None:
        for n in config.get("days", []):
            out.append({"kind": "day", "n": int(n), "col": f"ret_{n}d"})
        for n in config.get("weeks", []):
            out.append({"kind": "week", "n": int(n), "col": f"ret_{n}w"})
        return out
    for h in hs:
        if "days" in h:
            out.append({"kind": "day", "n": int(h["days"]), "col": f"ret_{h['days']}d"})
        elif "weeks" in h:
            out.append({"kind": "week", "n": int(h["weeks"]), "col": f"ret_{h['weeks']}w"})
    return out


def _horizon_end(h: dict, base_idx: int, periods: np.ndarray, n_rows: int):
    """返回 (ret_end_idx, window_end_idx)。ret_end_idx 为 None 表示 ret=NaN。"""
    if h["kind"] == "day":
        tgt = base_idx + h["n"]
        if tgt >= n_rows:
            return None, n_rows - 1
        return tgt, tgt
    # week: 真自然周
    target_week = periods[base_idx] + h["n"]
    in_week = np.flatnonzero(periods == target_week)
    # 窗口末 = 目标周（含）之前最后一个交易日
    upto = np.flatnonzero(periods <= target_week)
    window_end = int(upto[-1]) if len(upto) else base_idx
    if len(in_week) == 0:
        return None, window_end  # 空周: ret=NaN, 窗口仍有效
    return int(in_week[-1]), window_end


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------

def run_analyzer(config: dict, data_dir: Path | None = None,
                 overrides: dict | None = None, out_path: Path | None = None) -> Path:
    name = config["name"]
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    input_fp = Path(config.get("input", ""))
    if not input_fp.is_absolute():
        input_fp = ROOT / input_fp
    if not input_fp.exists():
        print(f"输入不存在: {input_fp}")
        raise SystemExit(1)

    signals = pd.read_csv(input_fp, encoding="utf-8-sig", dtype={"code": str})
    if "code" in signals.columns:
        signals["code"] = signals["code"].str.replace(r"\.0$", "", regex=True)
    total = len(signals)

    horizons = parse_horizons(config)
    events = config.get("events", [])
    group_cols = [c for c in config.get("group_cols", []) if c in signals.columns]
    metrics = config.get("metrics")
    stability = config.get("stability")

    print("=" * 88)
    print(f"  [analyze] {name}")
    print(f"  输入: {input_fp.name} | {total} 个形态")
    print(f"  horizons: {[h['col'] for h in horizons]} | events: {[e.get('name') for e in events]}")
    print(f"  分组: {group_cols or '无'}")
    print("=" * 88)

    records = []
    for code, group in signals.groupby("code"):
        fp = data_dir / f"{code}.csv"
        if not fp.exists():
            continue
        df = load_data(fp, 10)
        if df is None or df.empty:
            continue
        dates = df["date"].values
        periods = df["date"].dt.to_period("W").values
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        n_rows = len(close)
        date_to_idx = {d: k for k, d in enumerate(dates)}

        for _, row in group.iterrows():
            base = float(row["base_price"])
            if base <= 0 or pd.isna(row["base_date"]):
                continue
            base_idx = date_to_idx.get(pd.Timestamp(row["base_date"]).to_datetime64())
            if base_idx is None:
                continue

            rec = {"code": code, "base_date": str(row["base_date"]),
                   "base_price": round(base, 2)}
            # 透传分组/阈值列
            ic_signal = config.get("ic", {}).get("signal") if config.get("ic") else None
            for c in group_cols + [e["ref_col"] for e in events] + ([ic_signal] if ic_signal else []):
                if c in row.index and c not in rec:
                    rec[c] = row[c]

            for h in horizons:
                ret_idx, win_end = _horizon_end(h, base_idx, periods, n_rows)
                rec[h["col"]] = (round((close[ret_idx] / base - 1) * 100, 2)
                                 if ret_idx is not None else np.nan)
                seg_hi = high[base_idx:win_end + 1]
                seg_lo = low[base_idx:win_end + 1]
                base_tag = h["col"].replace("ret_", "")
                rec[f"maxup_{base_tag}"] = round((float(seg_hi.max()) / base - 1) * 100, 2)
                rec[f"maxdn_{base_tag}"] = round((float(seg_lo.min()) / base - 1) * 100, 2)

            for e in events:
                h = e["horizon"]  # {'days': N} 或 {'weeks': N}
                hkey = {"kind": "day", "n": int(h.get("days", 0))}
                if "weeks" in h:
                    hkey = {"kind": "week", "n": int(h["weeks"])}
                _, win_end = _horizon_end(hkey, base_idx, periods, n_rows)
                threshold = float(row[e["ref_col"]])
                seg_close = close[base_idx:win_end + 1]
                if e.get("type", "above") == "below":
                    rec[e["name"]] = bool(seg_close.min() <= threshold)
                else:
                    rec[e["name"]] = bool(seg_close.max() >= threshold)

            records.append(rec)

    if not records:
        print("  无有效样本。")
        return Path()

    res = pd.DataFrame(records)
    print(f"\n  有效样本: {len(res)} 个\n")

    summary_rows = []
    _print_main_table(res, horizons, metrics, summary_rows)
    _print_rank_ic(res, config, summary_rows)
    _print_profit_structure(res, horizons, summary_rows)
    for e in events:
        _print_event_table(res, e, group_cols, summary_rows)
    if group_cols:
        _print_group_table(res, group_cols, horizons, summary_rows)
    if stability:
        _print_stability(res, group_cols, horizons, stability, summary_rows)

    # 落盘
    out_dir = out_path.parent if out_path else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_path or (out_dir / config.get("output", f"an_{name}.csv"))
    res = res.sort_values("base_date", ascending=False).reset_index(drop=True)
    res.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n  明细已保存: {detail_path}")

    if summary_rows:
        sp = detail_path.with_name(detail_path.stem + "_summary.csv")
        pd.DataFrame(summary_rows).to_csv(sp, index=False, encoding="utf-8-sig")
        print(f"  汇总已保存: {sp}")
    return detail_path


# ------------------------------------------------------------
# 报表
# ------------------------------------------------------------

def _print_main_table(res, horizons, metrics, summary_rows):
    print("  【终点收益分布】")
    print(f"{'持有期':<10} {'样本':>6} {'E(期望)':>9} {'中位':>9} {'胜率':>7}")
    print("-" * 55)
    for h in horizons:
        s = compute_stats(res[h["col"]], metrics)
        if s is None:
            print(f"{h['col']:<10} {'0':>6}  (无足够未来数据)")
            continue
        from analyze.stats import format_row
        print(format_row(h["col"], s))
        summary_rows.append({"section": "ret", "col": h["col"], **s})


def _print_rank_ic(res, config, summary_rows):
    """按 analyzer YAML 的 ic 配置打印 RankIC（信号列 × 前瞻收益的 Spearman）。"""
    ic_cfg = config.get("ic")
    if not ic_cfg:
        return
    sig_col = ic_cfg["signal"]
    ret_col = ic_cfg["horizon"]
    if sig_col not in res.columns or ret_col not in res.columns:
        print(f"\n  [RankIC] 跳过: 列不存在 ({sig_col} / {ret_col})")
        return
    ic, n = rank_ic(res[sig_col], res[ret_col])
    print(f"\n  【RankIC】spearman({sig_col}, {ret_col}) = {ic:+.4f}  (n={n})")
    summary_rows.append({"section": "rank_ic", "col": f"{sig_col}×{ret_col}",
                         "n": n, "mean": ic})


def _print_profit_structure(res, horizons, summary_rows):
    print("\n  【盈亏结构】")
    print(f"{'持有期':<10} {'平均正收益':>10} {'平均负收益':>10} {'盈亏比':>8} "
          f"{'大涨≥20%':>10} {'大跌≤-20%':>10}")
    print("-" * 66)
    for h in horizons:
        s = compute_stats(res[h["col"]])
        if s is None:
            continue

        def fmt(v, suf=""):
            return "—" if (v is None or np.isnan(v)) else f"{v:.2f}{suf}"
        print(f"{h['col']:<10} {fmt(s['avg_pos'], '%'):>10} {fmt(s['avg_neg'], '%'):>10} "
              f"{fmt(s['plr']):>8} {s['up20']:>9.1f}% {s['dn20']:>9.1f}%")
        summary_rows.append({"section": "pl_structure", "col": h["col"],
                             "avg_pos": s["avg_pos"], "avg_neg": s["avg_neg"],
                             "plr": s["plr"], "up20": s["up20"], "dn20": s["dn20"],
                             "n": s["n"]})


def _print_event_table(res, event, group_cols, summary_rows):
    name = event["name"]
    etype = event.get("type", "above")
    desc = ("窗口内最高收盘 ≥ " if etype == "above"
            else "窗口内最低收盘 ≤ ") + event["ref_col"]
    print(f"\n  【事件: {name}】（{desc} 的占比）")
    if group_cols:
        print(f"{'分组':<30} {'样本':>6} {'占比':>9}")
        print("-" * 48)
        for key, sub in res.groupby(group_cols if len(group_cols) > 1 else group_cols[0]):
            if not isinstance(key, tuple):
                key = (key,)
            label = "·".join(str(v) for v in key)
            n = len(sub)
            rate = sub[name].fillna(False).astype(bool).sum() / n * 100
            print(f"{label:<30} {n:>6} {rate:>8.1f}%")
            summary_rows.append({"section": "event", "col": name,
                                 "group": label, "n": n, "mean": rate})
    else:
        n = len(res)
        rate = res[name].fillna(False).astype(bool).sum() / n * 100
        print(f"  全部: {n} 样本, 占比 {rate:.1f}%")
        summary_rows.append({"section": "event", "col": name, "n": n, "mean": rate})


def _print_group_table(res, group_cols, horizons, summary_rows):
    print(f"\n  【分组统计】（分组 × 持有期 均值 | 胜率 | 样本）")
    key = group_cols if len(group_cols) > 1 else group_cols[0]
    groups = list(res.groupby(key))
    header = f"{'分组':<24}" + "".join(f"{h['col']:>16}" for h in horizons)
    print(header)
    print("-" * len(header))
    for gkey, sub in groups:
        if not isinstance(gkey, tuple):
            gkey = (gkey,)
        label = "·".join(str(v) for v in gkey)
        line = f"{label:<24}"
        for h in horizons:
            s = compute_stats(sub[h["col"]])
            if s is None:
                line += f"{'—':>16}"
                continue
            line += f"{s['mean']:>+7.2f}%|{s['win_rate']:>4.0f}|{s['n']:>4}"
            summary_rows.append({"section": "group", "col": h["col"], "group": label,
                                 "n": s["n"], "mean": s["mean"],
                                 "median": s["median"], "win_rate": s["win_rate"]})
        print(line)


def _print_stability(res, group_cols, horizons, stability, summary_rows):
    year_col = stability.get("year_from", "base_date")
    exclude = [str(y) for y in stability.get("exclude_years", [])]
    d = res.copy()
    d["year"] = d[year_col].astype(str).str[:4]
    years = sorted(d["year"].dropna().unique())
    print(f"\n  【分年度稳健性】（均值(样本)，基准列: {year_col}，剔除对照: {exclude or '无'}）")
    key = group_cols if group_cols and len(group_cols) > 1 else (group_cols[0] if group_cols else None)
    groups = list(d.groupby(key)) if key else [("全部", d)]
    header = f"{'分组':<20}" + "".join(f"{y:>14}" for y in years) + f"{'剔除后':>14}"
    print(header)
    print("-" * len(header))
    for gkey, sub in groups:
        if not isinstance(gkey, tuple):
            gkey = (gkey,)
        label = "·".join(str(v) for v in gkey)[:18]
        line = f"{label:<20}"
        for y in years:
            s = compute_stats(sub[sub["year"] == y][horizons[0]["col"]])
            line += f"{s['mean']:>+7.2f}({s['n']:>3})" if s else f"{'—':>14}"
        ex = sub[~sub["year"].isin(exclude)][horizons[0]["col"]]
        s_ex = compute_stats(ex)
        line += f"{s_ex['mean']:>+12.2f}%" if s_ex else f"{'—':>14}"
        print(line)
        if s_ex:
            summary_rows.append({"section": "stability", "col": horizons[0]["col"],
                                 "group": label, "n": s_ex["n"], "mean": s_ex["mean"]})
