"""退出规则组件。口径见附录六"重放"系列条目。

每规则一个类，统一接口：
    trigger(path) -> int | None   # 返回退出下标（相对路径起点），None=未触发
规则集 = 多规则 OR 组合（任一触发即退出，取最早触发）。
"""
from __future__ import annotations

import numpy as np

from exit.replay import Path


class Rule:
    name = "base"

    def trigger(self, p: Path) -> int | None:
        raise NotImplementedError


class TrailingStop(Rule):
    """移动止盈：peak=至今最高收盘；激活 peak/entry-1>=X%；触发 (peak-close)/peak>=Y%。"""

    def __init__(self, activate: float, drawdown: float):
        self.activate = activate
        self.drawdown = drawdown
        self.name = f"trail({activate}/{drawdown})"

    def trigger(self, p: Path) -> int | None:
        cl = p.close
        peak = np.maximum.accumulate(cl)
        activated = np.cumsum((peak / p.entry - 1) * 100 >= self.activate) > 0
        trig = activated & (((peak - cl) / peak * 100) >= self.drawdown)
        idx = np.flatnonzero(trig)
        return int(idx[0]) if len(idx) else None


class TimeStop(Rule):
    """时间止损：第 N 交易日收盘浮亏（close < base）即出。"""

    def __init__(self, days: int):
        self.days = days
        self.name = f"time({days})"

    def trigger(self, p: Path) -> int | None:
        i = min(self.days, p.horizon)
        if i < len(p.close) and p.close[i] < p.entry:
            return i
        return None


class MaBreakExit(Rule):
    """均线 N 日确认退出：连续 N 日 close < MA(n)，第 N 日收盘出（R1）。

    向量化：滑动窗内 below 全 True 的首个窗口右端即触发点。
    """

    def __init__(self, ma: int = 20, confirm: int = 2):
        self.ma = ma
        self.confirm = confirm
        self.name = f"ma{ma}_x{self.confirm}"

    def trigger(self, p: Path) -> int | None:
        m = p.ma.get(self.ma)
        if m is None:
            raise KeyError(f"路径未预计算 MA{self.ma}（load_paths ma_periods）")
        k = self.confirm
        # 自基准次日起判定（下标 0 = 原路径的 i=1）
        below = (p.close < m)[1:].astype(np.int8)
        if len(below) < k:
            return None
        w = np.convolve(below, np.ones(k, dtype=np.int8), "valid")
        cand = np.flatnonzero(w == k)
        return int(cand[0] + k) if len(cand) else None  # +k: 窗口右端(+1偏移)


class StateExit(Rule):
    """状态退出：逐日 close < 行透传阈值列 ref_col 即出（首个下标，自基准次日起）。"""

    def __init__(self, ref_col: str):
        self.ref_col = ref_col
        self.name = f"state({ref_col})"

    def trigger(self, p: Path) -> int | None:
        ref = p.meta.get(self.ref_col)
        if ref is None or (isinstance(ref, float) and np.isnan(ref)):
            return None
        below = p.close[1:] < float(ref)
        idx = np.flatnonzero(below)
        return int(idx[0] + 1) if len(idx) else None


RULE_TYPES = {
    "trailing": lambda cfg: TrailingStop(cfg["activate"], cfg["drawdown"]),
    "time": lambda cfg: TimeStop(cfg["days"]),
    "ma_break": lambda cfg: MaBreakExit(cfg.get("ma", 20), cfg.get("confirm", 2)),
    "state": lambda cfg: StateExit(cfg["ref_col"]),
}


def build_rules(cfgs: list[dict]) -> list[Rule]:
    """从 exit_rules YAML 的 rules 列表构建规则集（OR 组合）。"""
    rules = []
    for c in cfgs:
        try:
            rules.append(RULE_TYPES[c["type"]](c))
        except KeyError as e:
            raise ValueError(f"未知规则类型或缺少参数: {c} ({e})") from e
    return rules


def apply_ruleset(p: Path, rules: list[Rule]) -> tuple[int, str]:
    """规则集 OR：任一触发即退出，取最早触发者。返回 (退出下标, 规则名)。"""
    best_i, best_name = None, None
    for r in rules:
        i = r.trigger(p)
        if i is not None and (best_i is None or i < best_i):
            best_i, best_name = i, r.name
    return best_i, best_name
