"""唯一重放引擎：catch 契约明细 + 日线 → 持有期路径。

契约（附录六）：输入含 code / base_date / base_price，信号日收盘买入，
路径自 base_idx 起（含基准日）。任意额外列（group、ref 阈值列）透传。
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.engine import load_data, DEFAULT_DATA_DIR
from indicator import sma


@dataclass
class Path:
    """单个持有期路径（自 base_idx 起的视图，含基准日）。"""
    code: str
    base_date: str
    base_price: float
    close: np.ndarray          # 自基准日起的收盘（含基准日）
    ma: dict[int, np.ndarray]  # 自基准日起的均线（与 close 对齐）
    horizon: int               # 路径长度（交易日）
    meta: dict = field(default_factory=dict)  # 透传行（分组/ref 列）

    @property
    def entry(self) -> float:
        return self.base_price


def load_paths(input_fp: Path, data_dir: Path | None = None,
               ma_periods: tuple[int, ...] = (20, 60),
               max_horizon: int = 250,
               passthrough: tuple[str, ...] = ()) -> list[Path]:
    """读契约明细，逐股重放生成路径列表。

    ma_periods: 需要预计算的均线周期（规则用到哪些就传哪些）；
    max_horizon: 路径截断长度（超出部分不进入重放）。
    基准日在个股日历中不存在（停牌）或前 ma 不足的行被跳过并计数。
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    sig = pd.read_csv(input_fp, encoding="utf-8-sig", dtype={"code": str})
    if "code" in sig.columns:
        sig["code"] = sig["code"].str.replace(r"\.0$", "", regex=True)

    paths: list[Path] = []
    skipped = 0
    for code, group in sig.groupby("code"):
        fp = data_dir / f"{code}.csv"
        if not fp.exists():
            skipped += len(group)
            continue
        df = load_data(fp, 10)
        if df is None or df.empty:
            skipped += len(group)
            continue
        dates = df["date"].dt.strftime("%Y-%m-%d").values
        close = df["close"].values.astype(float)
        idx = {d: k for k, d in enumerate(dates)}
        ma_arrays = {n: sma(df["close"], n).values for n in ma_periods}

        for _, row in group.iterrows():
            k = idx.get(str(row["base_date"]))
            if k is None or pd.isna(row["base_price"]) or row["base_price"] <= 0:
                skipped += 1
                continue
            end = min(k + 1 + max_horizon, len(close))
            meta = {c: row[c] for c in passthrough if c in row.index}
            paths.append(Path(
                code=code, base_date=str(row["base_date"]),
                base_price=float(row["base_price"]),
                close=close[k:end],
                ma={n: ma_arrays[n][k:end] for n in ma_periods},
                horizon=end - k - 1,
                meta=meta))
    if skipped:
        print(f"  [replay] 跳过 {skipped} 行（停牌/缺数据）", flush=True)
    return paths
