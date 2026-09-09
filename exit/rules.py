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
        peak = -np.inf
        activated = False
        for i in range(len(cl)):
            if cl[i] > peak:
                peak = cl[i]
            if not activated and (peak / p.entry - 1) * 100 >= self.activate:
                activated = True
            if activated and (peak - cl[i]) / peak * 100 >= self.drawdown:
                return i
        return None


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
    """均线 N 日确认退出：连续 N 日 close < MA(n)，第 N 日收盘出（R1）。"""

    def __init__(self, ma: int = 20, confirm: int = 2):
        self.ma = ma
        self.confirm = confirm
        self.name = f"ma{ma}_x{self.confirm}"

    def trigger(self, p: Path) -> int | None:
        cl = p.close
        m = p.ma.get(self.ma)
        if m is None:
            raise KeyError(f"路径未预计算 MA{self.ma}（load_paths ma_periods）")
        run = 0
        for i in range(1, len(cl)):          # 自基准次日起判定（基准日为入场日）
            if not np.isnan(m[i]) and cl[i] < m[i]:
                run += 1
                if run >= self.confirm:
                    return i
            else:
                run = 0
        return None


class StateExit(Rule):
    """状态退出：逐日 close < 行透传阈值列 ref_col 即出。"""

    def __init__(self, ref_col: str):
        self.ref_col = ref_col
        self.name = f"state({ref_col})"

    def trigger(self, p: Path) -> int | None:
        ref = p.meta.get(self.ref_col)
        if ref is None or (isinstance(ref, float) and np.isnan(ref)):
            return None
        ref = float(ref)
        for i in range(1, len(p.close)):
            if p.close[i] < ref:
                return i
        return None


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
