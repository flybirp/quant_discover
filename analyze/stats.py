"""统计指标库：三个旧 analyze 脚本所用指标的并集 + P15/P85。

analyze 不理解任何形态语义，只对一列收益数字做统计。
"""

import numpy as np
import pandas as pd

ALL_METRICS = [
    "n", "mean", "median", "std", "win_rate",
    "p15", "p25", "p75", "p85", "max", "min",
    "avg_pos", "avg_neg", "plr", "up20", "dn20",
]


def compute_stats(vals: pd.Series, metrics: list[str] | None = None) -> dict | None:
    """对一列收益做常见涨跌统计。metrics 为 None 时输出全集。"""
    vals = pd.Series(vals).dropna()
    n = len(vals)
    if n == 0:
        return None

    pos = vals[vals > 0]
    neg = vals[vals < 0]
    avg_pos = float(pos.mean()) if len(pos) else np.nan
    avg_neg = float(neg.mean()) if len(neg) else np.nan
    if len(neg) and not np.isnan(avg_neg) and avg_neg != 0:
        plr = avg_pos / abs(avg_neg)
    else:
        plr = np.nan

    full = {
        "n": int(n),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "std": float(vals.std()),
        "win_rate": float((vals > 0).sum() / n * 100),
        "p15": float(vals.quantile(0.15)),
        "p25": float(vals.quantile(0.25)),
        "p75": float(vals.quantile(0.75)),
        "p85": float(vals.quantile(0.85)),
        "max": float(vals.max()),
        "min": float(vals.min()),
        "avg_pos": avg_pos,
        "avg_neg": avg_neg,
        "plr": plr,
        "up20": float((vals >= 20).sum() / n * 100),
        "dn20": float((vals <= -20).sum() / n * 100),
    }
    if metrics is None:
        return full
    return {k: v for k, v in full.items() if k in metrics or k == "n"}


def rank_ic(signal: pd.Series, ret: pd.Series) -> tuple[float, int]:
    """RankIC：信号列与前瞻收益的 Spearman 秩相关。

    返回 (rank_ic, n)。量化惯例解读（截面单期）：
    |IC| < 0.02 无效 / 0.02~0.05 弱 / 0.05~0.10 中等 / > 0.10 强。
    """
    s = pd.DataFrame({"sig": signal, "ret": ret}).dropna()
    n = len(s)
    if n < 30:
        return float("nan"), n
    return float(s["sig"].corr(s["ret"], method="spearman")), n


def format_row(prefix: str, s: dict) -> str:
    """一行统计的标准格式: 样本/E(期望)/中位/胜率。"""
    def fmt(v, suffix=""):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}{suffix}"
    return (f"{prefix:<10} {s['n']:>6} {fmt(s.get('mean'), '%'):>9} "
            f"{fmt(s.get('median'), '%'):>9} {fmt(s.get('win_rate'), '%'):>7}")
