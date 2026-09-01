"""时间框架聚合层：目前只提供真自然周（calendar week）聚合。"""

import numpy as np
import pandas as pd


def to_calendar_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日线 -> 自然周（ISO 周，周一为一周之始）聚合 OHLCV。

    保留每周首/末日在原日线 DataFrame 中的行索引，
    供检测器在周线结构与日线数据之间来回定位。
    """
    d = df.reset_index(drop=True).copy()
    d["_idx"] = np.arange(len(d))
    d["week"] = d["date"].dt.to_period("W")
    g = d.groupby("week", sort=True)
    weekly = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "week_start_idx": g["_idx"].min(),
        "week_end_idx": g["_idx"].max(),
    }).reset_index(drop=True)
    return weekly


def weekly_high_series(dates: np.ndarray, high: np.ndarray,
                       start: int, end: int) -> list[float]:
    """将日线区间 [start, end]（含）按自然周聚合，返回每周最高价序列。

    用于"周线级别单顶"等形态校验。周标识取 date 的 ISO period，
    首尾残周各自成桶。
    """
    seg_dates = pd.Series(pd.DatetimeIndex(dates[start:end + 1]))
    seg_high = high[start:end + 1]
    weeks = seg_dates.dt.to_period("W")
    out = []
    cur_week = None
    cur_max = -np.inf
    for w, h in zip(weeks.values, seg_high):
        if cur_week is None or w != cur_week:
            if cur_week is not None:
                out.append(float(cur_max))
            cur_week = w
            cur_max = float(h)
        else:
            cur_max = max(cur_max, float(h))
    if cur_week is not None:
        out.append(float(cur_max))
    return out
