"""市场环境层：指数加载与个股相对强度（RPS 基础）。

相对收益口径（逐股可算，无需全市场截面缓存）：
  rel_ret_{N}d = 个股 N 日收益 - 指数同期 N 日收益
指数默认中证1000（zz1000，旧文档结论：其对个股环境的区分度最高）。
指数缺日以前值填充（ffill），保证与个股交易日对齐。

截面 RPS 分位（欧奈尔口径 0~99）在分析阶段按 base_date 对全市场样本
排名得到，见 docs/discover_15 的复算方法。
"""

from pathlib import Path

import pandas as pd

INDEX_DATA_DIR = Path.home() / "Documents" / "mainland_index_data_2014"
DEFAULT_INDEX = "zz1000"

_index_cache: dict[str, pd.Series] = {}


def load_index(name: str = DEFAULT_INDEX) -> pd.Series:
    """加载指数收盘序列（date 索引，升序），带进程内缓存。"""
    if name in _index_cache:
        return _index_cache[name]
    fp = INDEX_DATA_DIR / f"{name}.csv"
    df = pd.read_csv(fp, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["close"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    _index_cache[name] = s
    return s


def relative_return(df: pd.DataFrame, period: int = 120,
                    index_name: str = DEFAULT_INDEX) -> pd.Series:
    """给个股日线 df 计算相对指数的 N 日超额收益（百分比）。

    返回 Series（与 df 等长，前 period 行为 NaN）：
      (close/close.shift(period) - 1)*100 - (index 同期涨跌幅)*100
    """
    idx = load_index(index_name)
    dates = pd.DatetimeIndex(df["date"].values)
    # 对齐指数：个股日期 reindex 指数并 ffill（覆盖指数停发/缺失日）
    idx_aligned = idx.reindex(dates.union(idx.index)).ffill().reindex(dates)
    idx_ret = (idx_aligned / idx_aligned.shift(period) - 1) * 100
    stock_ret = (df["close"] / df["close"].shift(period) - 1) * 100
    return (stock_ret - idx_ret.values).rename(f"rel_ret_{period}d")
