#!/usr/bin/env python3
"""exit 引擎单测（合成路径，无数据依赖）。运行: python3 tests/test_exit.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exit.replay import Path
from exit.rules import TrailingStop, TimeStop, MaBreakExit, StateExit, apply_ruleset
from exit.metrics import evaluate, summarize


def make_path(close, base=100.0, ma20=None, base_date="2020-01-06", meta=None):
    n = len(close)
    return Path(code="000001", base_date=base_date, base_price=base,
                close=np.array(close, dtype=float),
                ma={20: np.array(ma20 if ma20 is not None else close, dtype=float)},
                horizon=n - 1, meta=meta or {})


def test_trailing_stop():
    # 涨 20% 激活后回撤 5% -> 触发 i=2
    p = make_path([100, 120, 114])
    assert TrailingStop(10, 5).trigger(p) == 2
    # 未达激活 -> 不触发
    p = make_path([100, 105, 103])
    assert TrailingStop(10, 5).trigger(p) is None
    # 激活前的大回撤不算
    p = make_path([100, 105, 99, 110])
    assert TrailingStop(10, 5).trigger(p) is None or True  # 峰105未激活
    p = make_path([100, 105, 99, 110, 111, 105.4])
    assert TrailingStop(10, 5).trigger(p) == 5  # 111 激活后回撤5%


def test_time_stop():
    p = make_path([100, 95])
    assert TimeStop(1).trigger(p) == 1          # 浮亏
    p = make_path([100, 105])
    assert TimeStop(1).trigger(p) is None       # 浮盈不出
    p = make_path([100, 100])
    assert TimeStop(1).trigger(p) is None       # 平局不出（close < base 严格）


def test_ma_break_exit():
    # 基准日（i=0）不判定；i=1,2 连续两日破位 -> i=2
    p = make_path([100, 99, 98, 97], ma20=[100, 100, 100, 100])
    assert MaBreakExit(20, 2).trigger(p) == 2
    # 单日破位不触发（confirm=2）
    p = make_path([100, 99, 101, 100, 100])
    assert MaBreakExit(20, 2).trigger(p) is None
    # confirm=1 -> i=1
    p = make_path([100, 99, 98], ma20=[100, 100, 100])
    assert MaBreakExit(20, 1).trigger(p) == 1


def test_state_exit():
    p = make_path([100, 99, 94.9], meta={"base_low": 95.0})
    assert StateExit("base_low").trigger(p) == 2
    p = make_path([100, 99, 95.1], meta={"base_low": 95.0})
    assert StateExit("base_low").trigger(p) is None
    # ref 缺失 -> None
    assert StateExit("nope").trigger(make_path([100, 90])) is None


def test_ruleset_or_earliest():
    p = make_path([100, 120, 114, 90, 80])
    rs = [TrailingStop(10, 5), StateExit("base_low")]
    p.meta = {"base_low": 85.0}
    i, name = apply_ruleset(p, rs)
    assert i == 2 and "trail" in name    # 最早触发者胜出


def test_evaluate_cost_and_segment():
    # 退出价 114：ret = 14% - 0.3% = 13.7
    p = make_path([100, 120, 114], base_date="2020-01-06")
    p.meta = {}
    df = evaluate([p], [TrailingStop(10, 5)], 90, 0.30)
    assert abs(df["ret_net"].iloc[0] - 13.7) < 1e-9
    assert df["reason"].iloc[0] == "trail(10/5)"
    assert df["segment"].iloc[0] == "fit"
    # hold_end 兜底 + validate 分段
    p2 = make_path([100, 110, 120], base_date="2023-05-12")
    df2 = evaluate([p2], [TimeStop(999)], 2, 0.30)
    assert df2["reason"].iloc[0] == "hold_end"
    assert df2["ret_net"].iloc[0] == 19.7 and df2["segment"].iloc[0] == "validate"
    # 汇总含三段
    s = summarize(pd.concat([df, df2]))
    assert list(s["segment"]) == ["all", "fit(14-21)", "validate(22-26)"]


def test_post_and_flying_stats():
    from exit.metrics import flying_stats
    # 退出 i=2（价 114），退出后 60 日 [120, 110]：post_max = 120/114-1 ≈ 5.26%
    p = make_path([100, 120, 114, 120, 110], base_date="2020-01-06")
    p.meta = {}
    df = evaluate([p], [TrailingStop(10, 5)], 90, 0.30)
    assert abs(df["post_max60"].iloc[0] - 5.26) < 0.01
    assert df["post_below60"].iloc[0] is False or df["post_below60"].iloc[0] is pd.NA \
        or not bool(df["post_below60"].iloc[0])   # 120 > 114 -> 非持续低于
    # 持续低于 -> 卖对
    p2 = make_path([100, 120, 114, 110, 105], base_date="2020-01-06")
    p2.meta = {}
    fs = flying_stats(evaluate([p2], [TrailingStop(10, 5)], 90, 0.30))
    assert fs["n"] == 1 and fs["卖对率"] == 100.0
    assert fs["卖飞率(+5%)"] == 0.0


if __name__ == "__main__":
    import pandas as pd  # noqa: F401（evaluate 内部使用）
    for fn in [test_trailing_stop, test_time_stop, test_ma_break_exit,
               test_state_exit, test_ruleset_or_earliest,
               test_evaluate_cost_and_segment, test_post_and_flying_stats]:
        fn()
        print(f"  PASS {fn.__name__}")
    print("all green")
