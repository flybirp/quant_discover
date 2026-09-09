"""退出评估：净成本口径 + 样本内外分段（附录六）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from exit.replay import Path
from exit.rules import apply_ruleset


def evaluate(paths: list[Path], rules: list[Rule], max_horizon: int,
             cost_roundtrip: float = 0.30,
             post_days: int = 60) -> pd.DataFrame:
    """对每路径应用规则集，返回逐样本退出明细 DataFrame。

    列: code/base_date/days/exit_idx/ret_net/win/reason/segment
        + post_max{N}（退出后 N 日内 max(close)/退出价-1，%）
        + post_below{N}（退出后 N 日 close 是否持续 < 退出价）
    ret_net = (exit_close/entry - 1)*100 - cost_roundtrip（附录六）。
    未触发持有至路径末端收盘平仓。post 列为 R1 卖飞统计的基础
    （附录六：卖飞率/卖对率/错失幅度）。
    """
    from exit.rules import Rule  # noqa: F401  (类型标注)
    rows = []
    for p in paths:
        i, reason = apply_ruleset(p, rules)
        if i is None:
            i, reason = len(p.close) - 1, "hold_end"
        exit_px = p.close[i]
        ret = (exit_px / p.entry - 1) * 100 - cost_roundtrip
        post = p.close[i + 1:i + 1 + post_days]
        if len(post):
            post_max = round((post.max() / exit_px - 1) * 100, 2)
            post_below = bool((post < exit_px).all())
        else:
            post_max, post_below = np.nan, np.nan
        rows.append({
            "code": p.code, "base_date": p.base_date,
            "days": i, "ret_net": round(ret, 2),
            "win": ret > 0, "reason": reason,
            "segment": "fit" if p.base_date[:4] <= "2021" else "validate",
            f"post_max{post_days}": post_max,
            f"post_below{post_days}": post_below})
    return pd.DataFrame(rows)


def flying_stats(df: pd.DataFrame, post_days: int = 60) -> dict:
    """卖飞统计（附录六 R1 节口径）：卖飞率/错失幅度/卖对率。

    输入 evaluate 的输出（含 post_max{N} / post_below{N} 列）。
    """
    pm, pb = df[f"post_max{post_days}"], df[f"post_below{post_days}"]
    ok = pm.notna()
    pm, pb = pm[ok], pb[ok]
    return {
        "n": int(ok.sum()),
        "卖飞率(+5%)": round((pm >= 5).mean() * 100, 1),
        "卖飞率(+10%)": round((pm >= 10).mean() * 100, 1),
        "错失中位": round(pm.median(), 2),
        "错失P75": round(pm.quantile(0.75), 2),
        "卖对率": round((pb.fillna(False)).mean() * 100, 1),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """规则级汇总：全样本 + fit/validate 分段（n/E/中位/胜率/平均天数/单位时间收益）。"""
    def _s(sub: pd.DataFrame) -> dict:
        if not len(sub):
            return {"n": 0}
        days = sub["days"].replace(0, np.nan).mean()
        return {
            "n": len(sub),
            "E净": round(sub["ret_net"].mean(), 2),
            "中位": round(sub["ret_net"].median(), 2),
            "胜率": round(sub["win"].mean() * 100, 1),
            "平均天数": round(sub["days"].mean(), 1),
            "单位时间收益": round(sub["ret_net"].mean() / days, 3)
            if days and not np.isnan(days) else np.nan,
        }

    rows = [ {"segment": "all", **_s(df)},
             {"segment": "fit(14-21)", **_s(df[df["segment"] == "fit"])},
             {"segment": "validate(22-26)", **_s(df[df["segment"] == "validate"])} ]
    return pd.DataFrame(rows)


def reason_table(df: pd.DataFrame) -> pd.DataFrame:
    """退出原因分布（多规则规则集的归因）。"""
    g = df.groupby("reason").agg(
        n=("ret_net", "size"),
        E净=("ret_net", lambda s: round(s.mean(), 2)),
        平均天数=("days", lambda s: round(s.mean(), 1)))
    g["占比%"] = (g["n"] / len(df) * 100).round(1)
    return g
