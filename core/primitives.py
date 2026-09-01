"""段检测器注册表（粗粒度原语）。

每个检测器签名统一: (df: DataFrame, code: str, p: dict) -> list[dict]
形态语义全部封装在检测器内部，engine 只负责编排。

四类结构:
  信号状态机      zfzs / volume / main_dd / main_trendline
  贪婪区间扩展    consolidation / long_consolidation / flat_zone(横盘段)
  两窗口速度比较  oversold / flat_zone(下跌段)
  关键点+多段确认 base_break / spike_top

相对旧代码的三处既定修改:
  1. base_break          剥离全部 forward 数据，新增 base_price（跌破日收盘价）
  2. spike_top           周线单顶从"5日块"改为真自然周（行为升级）
  3. long_consolidation  废除突破三分类(determine_breakout)，
                         保留 21 个区间特征透传列
"""

import numpy as np
import pandas as pd

from indicator import sma, atr, bollinger, macd, zhixing_fast, zhixing_slow
from core.resample import to_calendar_weekly, weekly_high_series

REGISTRY: dict = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def get_primitive(name: str):
    if name not in REGISTRY:
        raise KeyError(f"未注册的检测器: {name}，可用: {sorted(REGISTRY)}")
    return REGISTRY[name]


# ============================================================
# 信号状态机
# ============================================================

@register("zfzs")
def detect_zfzs(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """zhixing_fast 金叉 zhixing_slow 入场，MA20 死叉 MA60 离场。"""
    min_days = int(p.get("min_days", 20))
    close = df["close"].values
    dates = df["date"].values
    n = len(close)

    z_fast = zhixing_fast(df["close"]).values
    z_slow = zhixing_slow(df["close"]).values
    ma20 = sma(df["close"], 20).values
    ma60 = sma(df["close"], 60).values

    waves = []
    in_position = False
    entry_idx = None
    entry_price = None
    peak_price = None
    peak_idx = None

    for i in range(1, n):
        if np.isnan(z_fast[i]) or np.isnan(z_slow[i]) or np.isnan(ma20[i]) or np.isnan(ma60[i]):
            continue

        if not in_position:
            if z_fast[i - 1] <= z_slow[i - 1] and z_fast[i] > z_slow[i]:
                in_position = True
                entry_idx = i
                entry_price = close[i]
                peak_price = close[i]
                peak_idx = i
        else:
            if close[i] > peak_price:
                peak_price = close[i]
                peak_idx = i

            if ma20[i - 1] >= ma60[i - 1] and ma20[i] < ma60[i]:
                rec = _zfzs_record(code, dates, close, entry_idx, i,
                                   entry_price, peak_price, peak_idx,
                                   min_days, holding=False)
                if rec:
                    waves.append(rec)
                in_position = False

    if in_position:
        rec = _zfzs_record(code, dates, close, entry_idx, n - 1,
                           entry_price, peak_price, peak_idx,
                           min_days, holding=True)
        if rec:
            waves.append(rec)

    return waves


def _zfzs_record(code, dates, close, entry_idx, exit_idx, entry_price,
                 peak_price, peak_idx, min_days, holding):
    duration = exit_idx - entry_idx
    exit_price = close[exit_idx]
    ret_pct = (exit_price / entry_price - 1) * 100
    seg = close[peak_idx:exit_idx + 1]
    if len(seg) > 0:
        max_dd = (peak_price - float(np.min(seg))) / peak_price * 100
    else:
        max_dd = 0.0
    if duration < min_days:
        return None
    rec = {
        "code": code,
        "entry_date": pd.Timestamp(dates[entry_idx]).strftime("%Y-%m-%d"),
        "exit_date": pd.Timestamp(dates[exit_idx]).strftime("%Y-%m-%d"),
        "duration_days": duration,
        "entry_price": round(float(entry_price), 2),
        "peak_price": round(float(peak_price), 2),
        "exit_price": round(float(exit_price), 2),
        "return_pct": round(ret_pct, 2),
        "max_drawdown_pct": round(max_dd, 1),
    }
    if holding:
        rec["holding"] = True
    return rec


@register("volume")
def detect_volume(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """长期下跌后放量突破 zhixing_fast；止损(前低)/止盈(涨100%)离场。"""
    lookback = int(p.get("lookback", 60))
    decline_pct = float(p.get("decline_pct", 20.0))
    vol_multi = float(p.get("vol_multi", 2.0))
    min_days = int(p.get("min_days", 1))

    close = df["close"].values
    dates = df["date"].values
    low = df["low"].values
    volume = df["volume"].values
    n = len(close)

    z_fast = zhixing_fast(df["close"]).values
    z_slow = zhixing_slow(df["close"]).values
    vol_ma20 = sma(df["volume"], 20).values

    waves = []
    in_position = False
    entry_idx = None
    entry_price = None
    peak_price = None
    stop_price = None
    take_profit = None

    for i in range(lookback + 20, n):
        if np.isnan(z_fast[i]) or np.isnan(z_slow[i]) or np.isnan(vol_ma20[i]):
            continue

        if not in_position:
            window_close = close[i - lookback:i]
            window_zs = z_slow[i - lookback:i]
            if np.any(np.isnan(window_zs)):
                continue
            below_ratio = np.sum(window_close < window_zs) / len(window_close)
            if below_ratio < 0.7:
                continue
            window_high = np.max(window_close)
            decline = (window_high - close[i]) / window_high * 100
            if decline < decline_pct:
                continue
            if volume[i] <= vol_ma20[i] * vol_multi:
                continue
            if close[i] <= z_fast[i]:
                continue

            in_position = True
            entry_idx = i
            entry_price = close[i]
            peak_price = close[i]
            stop_price = float(np.min(low[i - lookback:i]))
            take_profit = entry_price * 2
        else:
            if close[i] > peak_price:
                peak_price = close[i]

            exit_reason = None
            exit_price = None
            if close[i] < stop_price:
                exit_reason = "stop"
                exit_price = close[i]
            elif close[i] >= take_profit:
                exit_reason = "target"
                exit_price = close[i]

            if exit_reason:
                if i - entry_idx >= min_days:
                    waves.append(_volume_record(
                        code, dates, close, entry_idx, i, entry_price,
                        peak_price, exit_price, exit_reason))
                in_position = False

    if in_position:
        i = n - 1
        if i - entry_idx >= min_days:
            waves.append(_volume_record(
                code, dates, close, entry_idx, i, entry_price,
                peak_price, close[i], "holding", holding=True))

    return waves


def _volume_record(code, dates, close, entry_idx, i, entry_price,
                   peak_price, exit_price, exit_reason, holding=False):
    duration = i - entry_idx
    ret_pct = (exit_price / entry_price - 1) * 100
    seg_close = close[entry_idx:i + 1]
    peak_idx_val = entry_idx + int(np.argmax(seg_close))
    after_peak = close[peak_idx_val:i + 1]
    min_after_peak = float(np.min(after_peak)) if len(after_peak) > 0 else exit_price
    max_dd = (peak_price - min_after_peak) / peak_price * 100
    rec = {
        "code": code,
        "entry_date": pd.Timestamp(dates[entry_idx]).strftime("%Y-%m-%d"),
        "exit_date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
        "duration_days": duration,
        "entry_price": round(float(entry_price), 2),
        "peak_price": round(float(peak_price), 2),
        "exit_price": round(float(exit_price), 2),
        "return_pct": round(ret_pct, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "exit_reason": exit_reason,
    }
    if holding:
        rec["holding"] = True
    return rec


# ============================================================
# 两窗口速度比较
# ============================================================

@register("oversold")
def detect_oversold(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """加速下跌后触底反弹 ≥ bounce_target_pct%。（只记录数据点）"""
    lookback_decline = int(p.get("lookback_decline", 40))
    lookback_accel = int(p.get("lookback_accel", 20))
    accel_threshold = float(p.get("accel_threshold", 1.5))
    bounce_target_pct = float(p.get("bounce_target_pct", 30.0))

    close = df["close"].values
    dates = df["date"].values
    n = len(close)
    if n < lookback_decline + lookback_accel + 20:
        return []

    patterns = []
    last_processed_idx = 0

    for i in range(lookback_decline + lookback_accel, n):
        idx_prev_start = i - lookback_decline - lookback_accel
        idx_prev_end = i - lookback_accel
        idx_recent_start = i - lookback_accel
        idx_recent_end = i

        prev_returns = []
        for j in range(idx_prev_start + 1, idx_prev_end):
            prev_returns.append((close[j] - close[j - 1]) / close[j - 1])
        avg_prev_ret = np.mean(prev_returns) if prev_returns else 0.0

        recent_returns = []
        for j in range(idx_recent_start + 1, idx_recent_end):
            recent_returns.append((close[j] - close[j - 1]) / close[j - 1])
        avg_recent_ret = np.mean(recent_returns) if recent_returns else 0.0

        if (avg_prev_ret < -0.001 and avg_recent_ret < avg_prev_ret
                and avg_prev_ret != 0
                and avg_recent_ret / avg_prev_ret > accel_threshold):
            search_end = min(idx_recent_end + 30, n)
            search_start = idx_recent_start
            if search_start >= search_end:
                continue
            bottom_idx = search_start + int(np.argmin(close[search_start:search_end]))

            if bottom_idx > last_processed_idx:
                bottom_price = float(close[bottom_idx])
                for j in range(bottom_idx + 1, min(bottom_idx + 120, n)):
                    bounce_pct = (close[j] - bottom_price) / bottom_price * 100
                    if bounce_pct >= bounce_target_pct:
                        peak_idx = idx_prev_start + int(
                            np.argmax(close[idx_prev_start:idx_prev_end]))
                        peak_price = float(close[peak_idx])
                        decline_pct = (peak_price - bottom_price) / peak_price * 100
                        patterns.append({
                            "code": code,
                            "peak_before_date": pd.Timestamp(dates[peak_idx]).strftime("%Y-%m-%d"),
                            "peak_before_price": round(peak_price, 2),
                            "bottom_date": pd.Timestamp(dates[bottom_idx]).strftime("%Y-%m-%d"),
                            "bottom_price": round(bottom_price, 2),
                            "decline_pct": round(decline_pct, 2),
                            "bounce_date": pd.Timestamp(dates[j]).strftime("%Y-%m-%d"),
                            "bounce_price": round(float(close[j]), 2),
                            "bounce_pct": round(bounce_pct, 2),
                            "decay_days": bottom_idx - peak_idx,
                            "recover_days": j - bottom_idx,
                            "avg_daily_decline_before": round(avg_prev_ret * 100, 3),
                            "avg_daily_decline_accel": round(avg_recent_ret * 100, 3),
                        })
                        break
                last_processed_idx = bottom_idx

    return patterns


# ============================================================
# 贪婪区间扩展
# ============================================================

@register("consolidation")
def detect_consolidation(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """≥min_days 天、窗口均价 ±band_pct% 的窄幅横盘（贪婪扩展）。"""
    min_days = int(p.get("min_days", 60))
    band_pct = float(p.get("band_pct", 10.0))

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    dates = df["date"].values
    n = len(close)

    zones = []
    i = 0
    while i < n:
        hi = high[i]
        lo = low[i]
        s = 0.0
        cnt = 0
        end = i
        for k in range(i, n):
            cnt += 1
            s += close[k]
            if high[k] > hi:
                hi = high[k]
            if low[k] < lo:
                lo = low[k]
            avg = s / cnt
            if avg <= 0:
                break
            up = (hi - avg) / avg * 100
            down = (avg - lo) / avg * 100
            if up > band_pct or down > band_pct:
                break
            end = k

        days = end - i + 1
        if days >= min_days:
            seg_close = close[i:end + 1]
            seg_high = high[i:end + 1]
            seg_low = low[i:end + 1]
            avg = float(np.mean(seg_close))
            hi = float(np.max(seg_high))
            lo = float(np.min(seg_low))
            zones.append({
                "code": code,
                "start_date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(dates[end]).strftime("%Y-%m-%d"),
                "duration_days": days,
                "avg_price": round(avg, 2),
                "high_price": round(hi, 2),
                "low_price": round(lo, 2),
                "up_pct": round((hi - avg) / avg * 100, 2),
                "down_pct": round((avg - lo) / avg * 100, 2),
                "range_pct": round((hi - lo) / avg * 100, 2),
                "start_price": round(float(close[i]), 2),
                "end_price": round(float(close[end]), 2),
                "trend_pct": round((close[end] / close[i] - 1) * 100, 2) if close[i] > 0 else 0.0,
            })
            i = end + 1
        else:
            i += 1

    return zones


@register("flat_zone")
def detect_flat_zone(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """加速下跌段之后紧跟的窄幅横盘平台。"""
    lookback_decline = int(p.get("lookback_decline", 40))
    lookback_accel = int(p.get("lookback_accel", 20))
    accel_threshold = float(p.get("accel_threshold", 1.5))
    min_flat_days = int(p.get("min_flat_days", 15))
    flat_range_pct_max = float(p.get("flat_range_pct_max", 5.0))

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    dates = df["date"].values
    n = len(close)

    total_lookback = lookback_decline + lookback_accel
    results = []

    i = total_lookback
    while i < n:
        idx_prev_start = i - total_lookback
        idx_prev_end = i - lookback_accel

        prev_seg = close[idx_prev_start:idx_prev_end]
        recent_seg = close[i - lookback_accel:i]
        if len(prev_seg) < 2 or len(recent_seg) < 2:
            i += 1
            continue

        avg_prev_ret = float(np.mean(np.diff(prev_seg) / prev_seg[:-1]))
        avg_recent_ret = float(np.mean(np.diff(recent_seg) / recent_seg[:-1]))

        if not (avg_prev_ret < -0.001 and avg_recent_ret < avg_prev_ret
                and avg_prev_ret != 0
                and avg_recent_ret / avg_prev_ret > accel_threshold):
            i += 1
            continue

        window_close = close[idx_prev_start:i]
        window_high = float(np.max(window_close)) if len(window_close) > 0 else close[i]
        decline_pct = (window_high - close[i]) / window_high * 100 if window_high > 0 else 0.0

        end = i
        k = i + 1
        while k < n:
            seg_high = high[i:k + 1]
            seg_low = low[i:k + 1]
            seg_close = close[i:k + 1]
            mean_close = float(np.mean(seg_close))
            if mean_close <= 0:
                break
            rng = (float(np.max(seg_high)) - float(np.min(seg_low))) / mean_close * 100
            if rng > flat_range_pct_max:
                break
            end = k
            k += 1

        flat_days = end - i + 1
        if flat_days >= min_flat_days:
            seg_close = close[i:end + 1]
            seg_high = high[i:end + 1]
            seg_low = low[i:end + 1]
            avg_price = float(np.mean(seg_close))
            flat_range = (float(np.max(seg_high)) - float(np.min(seg_low))) / avg_price * 100

            results.append({
                "code": code,
                "accel_start_date": pd.Timestamp(dates[idx_prev_start]).strftime("%Y-%m-%d"),
                "drop_high_price": round(window_high, 2),
                "decline_pct": round(decline_pct, 1),
                "avg_daily_ret_before": round(avg_prev_ret * 100, 3),
                "avg_daily_ret_accel": round(avg_recent_ret * 100, 3),
                "accel_ratio": round(avg_recent_ret / avg_prev_ret, 2),
                "flat_start_date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                "flat_end_date": pd.Timestamp(dates[end]).strftime("%Y-%m-%d"),
                "flat_days": flat_days,
                "flat_avg_price": round(avg_price, 2),
                "flat_range_pct": round(flat_range, 2),
            })
            i = end + 1
        else:
            i += 1

    return results


# ============================================================
# 回撤状态机（主升浪）
# ============================================================

@register("main_dd")
def detect_main_dd(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """回撤状态机：持续上涨且任意回撤 ≤ max_drawdown_pct% 的波段。"""
    max_drawdown_pct = float(p.get("max_drawdown_pct", 50.0))
    min_days = int(p.get("min_days", 50))
    min_return_pct = float(p.get("min_return_pct", 50.0))

    close = df["close"].values
    dates = df["date"].values
    n = len(close)
    if n < min_days:
        return []

    waves = []
    wave_start_idx = 0
    running_peak = close[0]
    running_peak_idx = 0

    for i in range(1, n):
        price = close[i]
        if price > running_peak:
            running_peak = price
            running_peak_idx = i
        else:
            drawdown_pct = (running_peak - price) / running_peak * 100
            if drawdown_pct > max_drawdown_pct:
                rec = _main_dd_record(code, dates, close,
                                      wave_start_idx, i - 1, min_days, min_return_pct)
                if rec:
                    waves.append(rec)
                wave_start_idx = i
                running_peak = price
                running_peak_idx = i

    if wave_start_idx < n - 1:
        rec = _main_dd_record(code, dates, close,
                              wave_start_idx, n - 1, min_days, min_return_pct)
        if rec:
            waves.append(rec)

    return waves


def _main_dd_record(code, dates, close, wave_start_idx, wave_end_idx,
                    min_days, min_return_pct):
    seg_close = close[wave_start_idx:wave_end_idx + 1]
    abs_low_idx_local = int(np.argmin(seg_close))
    abs_low_idx = wave_start_idx + abs_low_idx_local
    abs_low_price = close[abs_low_idx]
    duration = wave_end_idx - abs_low_idx + 1
    if duration < min_days:
        return None
    ret = (close[wave_end_idx] / abs_low_price - 1) * 100
    if ret < min_return_pct:
        return None
    seg_peak = float(np.max(close[abs_low_idx:wave_end_idx + 1]))
    seg_peak_idx = abs_low_idx + int(np.argmax(close[abs_low_idx:wave_end_idx + 1]))
    after_seg_peak = close[seg_peak_idx:wave_end_idx + 1]
    seg_min_after_peak = float(np.min(after_seg_peak)) if len(after_seg_peak) > 0 else seg_peak
    actual_dd = (seg_peak - seg_min_after_peak) / seg_peak * 100
    return {
        "code": code,
        "start_date": pd.Timestamp(dates[abs_low_idx]).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(dates[wave_end_idx]).strftime("%Y-%m-%d"),
        "duration_days": duration,
        "wave_start_close": round(float(abs_low_price), 2),
        "peak_close": round(seg_peak, 2),
        "wave_end_close": round(float(close[wave_end_idx]), 2),
        "return_pct": round(ret, 2),
        "max_drawdown_pct": round(actual_dd, 1),
    }


@register("main_trendline")
def detect_main_trendline(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """沿趋势线运行（不连续2日跌破），涨幅>100%，持续>200天。"""
    indicator_name = p.get("indicator", "zhixing_slow")
    close = df["close"].values
    dates = df["date"].values
    line = _make_trendline(df["close"], indicator_name).values

    first_valid = 0
    while first_valid < len(line) and pd.isna(line[first_valid]):
        first_valid += 1
    if first_valid >= len(line):
        return []

    clean_segments = []
    in_segment = False
    seg_start = None
    consecutive_below = 0

    for i in range(first_valid, len(close)):
        below = close[i] < line[i]
        if below:
            consecutive_below += 1
        else:
            consecutive_below = 0
        if not in_segment:
            if not below:
                in_segment = True
                seg_start = i
        else:
            if consecutive_below >= 2:
                in_segment = False
                clean_segments.append((seg_start, i - 1))
    if in_segment:
        clean_segments.append((seg_start, len(close) - 1))

    if not clean_segments:
        return []

    all_waves = []
    for seg_start, seg_end in clean_segments:
        seg_duration = seg_end - seg_start + 1
        if seg_duration <= 200:
            continue

        local_close = close[seg_start:seg_end + 1]
        local_dates = dates[seg_start:seg_end + 1]
        local_line = line[seg_start:seg_end + 1]
        seg_len = len(local_close)

        running_max = np.zeros(seg_len)
        running_max_idx = np.zeros(seg_len, dtype=int)
        running_max[-1] = local_close[-1]
        running_max_idx[-1] = seg_len - 1
        for j in range(seg_len - 2, -1, -1):
            if local_close[j] > running_max[j + 1]:
                running_max[j] = local_close[j]
                running_max_idx[j] = j
            else:
                running_max[j] = running_max[j + 1]
                running_max_idx[j] = running_max_idx[j + 1]

        best = None
        best_return = 0
        for i in range(seg_len - 200):
            if local_close[i] < local_line[i]:
                continue
            start_val = local_close[i]
            min_end = i + 200
            if min_end >= seg_len:
                break
            max_val_after = running_max[min_end]
            ret_pct = (max_val_after / start_val - 1) * 100
            if ret_pct > 100 and ret_pct > best_return:
                end_offset = int(running_max_idx[min_end])
                duration = end_offset - i + 1
                best_return = ret_pct
                best = {
                    "code": code,
                    "start_date": pd.Timestamp(local_dates[i]).strftime("%Y-%m-%d"),
                    "end_date": pd.Timestamp(local_dates[end_offset]).strftime("%Y-%m-%d"),
                    "duration_days": duration,
                    "start_close": round(float(start_val), 2),
                    "end_close": round(float(max_val_after), 2),
                    "return_pct": round(ret_pct, 2),
                }
        if best:
            all_waves.append(best)

    return all_waves


def _make_trendline(close: pd.Series, name: str) -> pd.Series:
    if name == "ma60":
        return sma(close, 60)
    elif name == "zhixing_slow":
        return zhixing_slow(close)
    raise ValueError(f"未知趋势线指标: {name}")


# ============================================================
# 关键点 + 多段确认
# ============================================================

@register("base_break")
def detect_base_break(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """周线筑底 -> 首次跌破筑底低点。

    相对旧版: 剥离 ret_1w~4w / max_up_4w / max_dd_4w / recover_base，
    新增 base_price = 跌破日收盘价（统一契约的买点基准）。
    """
    weekly = to_calendar_weekly(df)
    nw = len(weekly)

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    dates = df["date"].values
    n = len(df)

    results = []
    i = p["lookback_weeks"]
    while i < nw:
        # 头部: 周低点紧贴同一支撑的筑底段（贪婪扩展）
        seg_min = float(weekly["low"].iloc[i])
        seg_max = seg_min
        seg_end = i
        j = i
        while j < nw:
            low_j = float(weekly["low"].iloc[j])
            new_min = min(seg_min, low_j)
            new_max = max(seg_max, low_j)
            spread = (new_max - new_min) / new_min * 100
            if spread > p["support_tolerance_pct"]:
                break
            seg_min = new_min
            seg_max = new_max
            seg_end = j
            j += 1

        seg_len = seg_end - i + 1
        if seg_len < p["min_base_weeks"]:
            i = seg_end + 1
            continue

        base_low = seg_min

        # 位置判断: 宽幅震荡下沿 or 长期下跌后首次筑底
        lookback_start = i - p["lookback_weeks"]
        prior = weekly.iloc[lookback_start:i]
        prior_high = float(prior["high"].max())
        prior_low = float(prior["low"].min())
        if prior_high <= 0 or prior_low <= 0:
            i = seg_end + 1
            continue

        prior_decline = (prior_high - base_low) / prior_high * 100
        prior_range = (prior_high - prior_low) / prior_low * 100
        near_low = (base_low - prior_low) / prior_low * 100

        if prior_range >= p["range_amplitude_pct"] and near_low <= p["range_low_margin_pct"]:
            position_type = "range_low"
        elif prior_decline >= p["decline_pct"]:
            position_type = "first_base_after_drop"
        else:
            i = seg_end + 1
            continue

        base_start_idx = int(weekly["week_start_idx"].iloc[i])
        base_end_idx = int(weekly["week_end_idx"].iloc[seg_end])
        base_start_date = pd.Timestamp(dates[base_start_idx]).strftime("%Y-%m-%d")
        base_end_date = pd.Timestamp(dates[base_end_idx]).strftime("%Y-%m-%d")

        seg_close = close[base_start_idx:base_end_idx + 1]
        seg_high = high[base_start_idx:base_end_idx + 1]
        seg_low = low[base_start_idx:base_end_idx + 1]
        base_range_pct = (seg_high.max() - seg_low.min()) / seg_close.mean() * 100

        # 中部: 首次跌破筑底低点
        break_idx = None
        for k in range(base_end_idx + 1, n):
            if low[k] < base_low:
                break_idx = k
                break
        if break_idx is None:
            i = seg_end + 1
            continue

        break_low = float(low[break_idx])
        group = None
        for gi, ratio in enumerate(p["group_ratios"]):
            if break_low > base_low * ratio:
                group = p["group_names"][gi]
                break
        if group is None:
            i = seg_end + 1
            continue
        penetration_pct = (base_low - break_low) / base_low * 100

        vb_start = max(0, break_idx - p["vol_base_days"])
        vol_base = np.mean(volume[vb_start:break_idx])
        vol_ratio = volume[break_idx] / vol_base if vol_base > 0 else np.nan
        if pd.isna(vol_ratio):
            vol_type = "平量"
        elif vol_ratio > p["vol_ratio_high"]:
            vol_type = "放量"
        elif vol_ratio < p["vol_ratio_low"]:
            vol_type = "缩量"
        else:
            vol_type = "平量"

        results.append({
            "code": code,
            "group": group,
            "position_type": position_type,
            "base_start_date": base_start_date,
            "base_end_date": base_end_date,
            "base_weeks": seg_len,
            "base_low": round(base_low, 2),
            "base_range_pct": round(base_range_pct, 1),
            "prior_decline_pct": round(prior_decline, 1),
            "break_date": pd.Timestamp(dates[break_idx]).strftime("%Y-%m-%d"),
            "break_low": round(break_low, 2),
            "base_price": round(float(close[break_idx]), 2),
            "penetration_pct": round(penetration_pct, 2),
            "volume_ratio": round(float(vol_ratio), 2) if not pd.isna(vol_ratio) else np.nan,
            "volume_type": vol_type,
        })
        i = seg_end + 1

    return results


@register("spike_top")
def detect_spike_top(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """翻倍 -> 单顶 -> 腰斩 尖顶形态。

    相对旧版: 周线单顶校验从"每5个交易日切片"改为真自然周聚合（行为升级）。
    """
    local_n = int(p.get("local_n", 30))
    mult = float(p.get("mult", 2.0))
    peak_confirm_pct = float(p.get("peak_confirm_pct", 20.0))
    rebound_pct = float(p.get("rebound_pct", 20.0))
    peak_ratio = float(p.get("peak_ratio", 0.9))
    max_up_days = int(p.get("max_up_days", 500))

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    dates = df["date"].values
    n = len(close)

    patterns = []
    low_indices = _find_local_lows(close, local_n)

    for i in low_indices:
        x = close[i]
        if x <= 0:
            continue

        max_x = x
        m = i
        topped = False
        upper = min(n, i + max_up_days + 1)
        for k in range(i + 1, upper):
            if close[k] > max_x:
                max_x = close[k]
                m = k
            if max_x >= mult * x and (max_x - close[k]) / max_x * 100 >= peak_confirm_pct:
                topped = True
                break
        if not topped or max_x < mult * x:
            continue

        j = -1
        for k in range(m + 1, n):
            if close[k] <= max_x / 2:
                j = k
                break
        if j == -1:
            continue

        if not _strictly_declining(close, m, j, rebound_pct):
            continue

        if not _single_weekly_peak(dates, high, i, j, peak_ratio):
            continue

        patterns.append({
            "code": code,
            "low_date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
            "low_price": round(float(x), 2),
            "peak_date": pd.Timestamp(dates[m]).strftime("%Y-%m-%d"),
            "peak_price": round(float(max_x), 2),
            "half_date": pd.Timestamp(dates[j]).strftime("%Y-%m-%d"),
            "half_price": round(float(close[j]), 2),
            "up_pct": round((max_x / x - 1) * 100, 2),
            "down_pct": round((close[j] / max_x - 1) * 100, 2),
            "up_days": m - i,
            "down_days": j - m,
            "total_days": j - i,
        })

    return patterns


def _find_local_lows(close: np.ndarray, local_n: int) -> list[int]:
    n = len(close)
    s = pd.Series(close)
    left_min = s.rolling(window=local_n + 1, min_periods=1).min()
    right_min = s.iloc[::-1].rolling(window=local_n + 1, min_periods=1).min().iloc[::-1]
    is_low = (s == left_min) & (s == right_min)
    lows = []
    prev = False
    for idx in range(n):
        cur = bool(is_low.iloc[idx])
        if cur and not prev:
            lows.append(idx)
        prev = cur
    return lows


def _strictly_declining(close: np.ndarray, m: int, j: int, rebound_pct: float) -> bool:
    running_low = close[m]
    for k in range(m + 1, j + 1):
        if close[k] < running_low:
            running_low = close[k]
        else:
            rebound = (close[k] - running_low) / running_low * 100
            if rebound > rebound_pct:
                return False
    return True


def _single_weekly_peak(dates: np.ndarray, high: np.ndarray,
                        i: int, j: int, peak_ratio: float) -> bool:
    """真自然周聚合下的单顶判定：除主峰外其余周局部峰 ≤ 主峰×peak_ratio。"""
    weekly = weekly_high_series(dates, high, i, j)
    L = len(weekly)
    if L < 3:
        return True
    main_peak = max(weekly)
    if main_peak <= 0:
        return False
    for t in range(1, L - 1):
        if weekly[t] > weekly[t - 1] and weekly[t] > weekly[t + 1]:
            if weekly[t] != main_peak and weekly[t] > main_peak * peak_ratio:
                return False
    return True


# ============================================================
# 长期宽幅震荡（consolidation_scanner 迁移版）
# ============================================================

@register("long_consolidation")
def detect_long_consolidation(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """400~600 天宽幅震荡区 + 区间内 21 个量价/估值特征透传列。

    相对旧版: 废除 determine_breakout 三分类（direction/max_return/
    max_drawdown/final_return 列不再输出，改由 analyze 连续指标承担）。
    """
    window_min = int(p.get("window_min", 400))
    window_max = int(p.get("window_max", 600))
    window_step = int(p.get("window_step", 20))
    range_pct_max = float(p.get("range_pct_max", 50.0))
    trend_limit_pct = float(p.get("trend_limit_pct", 30.0))
    vol_converge = bool(p.get("vol_converge", True))

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    dates = df["date"].values
    n = len(close)
    if n < window_min:
        return []

    zones = []
    used_ranges = []

    for win_size in range(window_min, window_max + 1, window_step):
        for start in range(0, n - win_size + 1, window_step):
            end = start + win_size
            overlaps = False
            for (us, ue) in used_ranges:
                if start < ue and end > us:
                    overlaps = True
                    break
            if overlaps:
                continue

            seg_close = close[start:end]
            seg_high = high[start:end]
            seg_low = low[start:end]

            max_price = float(np.max(seg_high))
            min_price = float(np.min(seg_low))
            median_close = float(np.median(seg_close))
            if median_close <= 0:
                continue

            range_pct = (max_price - min_price) / median_close * 100
            if range_pct > range_pct_max:
                continue

            start_price = float(close[start])
            end_price = float(close[end - 1])
            if start_price <= 0:
                continue
            trend_pct = abs(end_price - start_price) / start_price * 100
            if trend_pct > trend_limit_pct:
                continue

            if vol_converge:
                half = win_size // 2
                seg_first_half = close[start:start + half]
                seg_second_half = close[start + half:end]
                vol_first = float(np.std(np.diff(seg_first_half) / seg_first_half[:-1]))
                vol_second = float(np.std(np.diff(seg_second_half) / seg_second_half[:-1]))
                if vol_second > vol_first * 1.2:
                    continue

            zone = {
                "code": code,
                "start_idx": start,
                "end_idx": end,
                "start_date": pd.Timestamp(dates[start]).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(dates[end - 1]).strftime("%Y-%m-%d"),
                "duration_days": win_size,
                "range_pct": round(range_pct, 1),
                "trend_pct": round(trend_pct, 1),
                "median_close": round(median_close, 2),
                "max_price": round(max_price, 2),
                "min_price": round(min_price, 2),
                # 契约基准价：区间最后一天收盘价（对应旧 determine_breakout 的 entry）
                "end_price": round(float(close[end - 1]), 2),
            }
            zone.update(_extract_zone_features(df, zone))
            del zone["start_idx"]
            del zone["end_idx"]
            zones.append(zone)
            used_ranges.append((start, end))

    return zones


def _extract_zone_features(df: pd.DataFrame, zone: dict) -> dict:
    """区间内部量价/波动率/均线/知行/MACD/估值/市值特征（不含未来数据）。"""
    start = zone["start_idx"]
    end = zone["end_idx"]
    n = end - start

    close = df["close"].values[start:end]
    high = df["high"].values[start:end]
    low = df["low"].values[start:end]
    volume = df["volume"].values[start:end]

    features = {}

    third = n // 3
    vol_first = float(np.mean(volume[:third]))
    vol_last = float(np.mean(volume[-third:]))
    features["vol_shrink_ratio"] = round(vol_last / (vol_first + 1), 3)

    vol_arr = volume.astype(float)
    x = np.arange(len(vol_arr))
    if np.std(vol_arr) > 0:
        vol_slope = np.polyfit(x, vol_arr, 1)[0]
        features["vol_trend"] = round(vol_slope / np.mean(vol_arr) * 100, 3)
    else:
        features["vol_trend"] = 0.0

    if "turnover_rate" in df.columns:
        tr = df["turnover_rate"].values[start:end]
        features["cum_turnover"] = round(float(np.nansum(tr)), 1)
    else:
        features["cum_turnover"] = 0.0

    atr_series = atr(pd.Series(high), pd.Series(low), pd.Series(close), 14)
    atr_vals = atr_series.values
    valid_atr = atr_vals[~np.isnan(atr_vals)]
    if len(valid_atr) > 0:
        features["atr_median"] = round(float(np.median(valid_atr)), 3)
        features["atr_pct"] = round(float(np.median(valid_atr)) / float(np.median(close)) * 100, 2)
    else:
        features["atr_median"] = 0.0
        features["atr_pct"] = 0.0

    if len(valid_atr) > third * 2:
        atr_first = float(np.median(valid_atr[:third]))
        atr_last = float(np.median(valid_atr[-third:]))
        features["atr_converge"] = round(atr_last / (atr_first + 0.001), 3)
    else:
        features["atr_converge"] = 1.0

    bb = bollinger(pd.Series(close), 20, 2.0)
    bb_width = (bb["upper"].values - bb["lower"].values) / bb["mid"].values * 100
    valid_bb = bb_width[~np.isnan(bb_width)]
    if len(valid_bb) > 0:
        features["bb_width_median"] = round(float(np.median(valid_bb)), 2)
    else:
        features["bb_width_median"] = 0.0

    daily_ret = np.diff(close) / close[:-1]
    features["annual_vol"] = round(float(np.std(daily_ret)) * np.sqrt(250) * 100, 1)

    ma20 = sma(pd.Series(close), 20).values
    ma60 = sma(pd.Series(close), 60).values
    ma120 = sma(pd.Series(close), 120).values

    valid_ma = ~(np.isnan(ma20) | np.isnan(ma60))
    if valid_ma.sum() > 30:
        ma_spread = (ma20[valid_ma] - ma60[valid_ma]) / ma60[valid_ma] * 100
        features["ma_tangle"] = round(float(np.std(ma_spread)), 2)
    else:
        features["ma_tangle"] = 0.0

    valid_all = ~(np.isnan(ma20) | np.isnan(ma60) | np.isnan(ma120))
    if valid_all.sum() > 0:
        bull_days = np.sum((ma20 > ma60) & (ma60 > ma120)) / valid_all.sum()
        features["ma_bull_ratio"] = round(float(bull_days), 3)
    else:
        features["ma_bull_ratio"] = 0.5

    zf = zhixing_fast(pd.Series(close)).values
    zs = zhixing_slow(pd.Series(close)).values
    valid_zx = ~(np.isnan(zf) | np.isnan(zs))
    if valid_zx.sum() > 0:
        cross_up = 0
        cross_down = 0
        for i in range(1, len(zf)):
            if np.isnan(zf[i]) or np.isnan(zs[i]) or np.isnan(zf[i - 1]) or np.isnan(zs[i - 1]):
                continue
            if zf[i - 1] <= zs[i - 1] and zf[i] > zs[i]:
                cross_up += 1
            if zf[i - 1] >= zs[i - 1] and zf[i] < zs[i]:
                cross_down += 1
        features["zx_cross_count"] = cross_up + cross_down
        zx_spread = (zf[valid_zx] - zs[valid_zx]) / zs[valid_zx] * 100
        features["zx_tangle"] = round(float(np.std(zx_spread)), 2)
    else:
        features["zx_cross_count"] = 0
        features["zx_tangle"] = 0.0

    macd_df = macd(pd.Series(close))
    dif = macd_df["dif"].values
    valid_dif = dif[~np.isnan(dif)]
    if len(valid_dif) > 0:
        features["dif_above_zero"] = round(float(np.sum(valid_dif > 0) / len(valid_dif)), 3)
        features["dif_abs_pct"] = round(float(np.median(np.abs(valid_dif))) / float(np.median(close)) * 100, 3)
    else:
        features["dif_above_zero"] = 0.5
        features["dif_abs_pct"] = 0.0

    if "pe_ttm" in df.columns:
        pe = df["pe_ttm"].values[start:end]
        valid_pe = pe[(~np.isnan(pe)) & (pe > 0) & (pe < 1000)]
        if len(valid_pe) > 0:
            features["pe_median"] = round(float(np.median(valid_pe)), 1)
            features["pe_range"] = round(float(np.percentile(valid_pe, 90) - np.percentile(valid_pe, 10)), 1)
        else:
            features["pe_median"] = 0.0
            features["pe_range"] = 0.0
    else:
        features["pe_median"] = 0.0
        features["pe_range"] = 0.0

    if "pb" in df.columns:
        pb = df["pb"].values[start:end]
        valid_pb = pb[(~np.isnan(pb)) & (pb > 0) & (pb < 100)]
        if len(valid_pb) > 0:
            features["pb_median"] = round(float(np.median(valid_pb)), 2)
        else:
            features["pb_median"] = 0.0
    else:
        features["pb_median"] = 0.0

    if "total_mv" in df.columns:
        mv = df["total_mv"].values[start:end]
        valid_mv = mv[~np.isnan(mv)]
        if len(valid_mv) > 0:
            features["mv_median"] = round(float(np.median(valid_mv)), 0)
            features["mv_change"] = round(float(valid_mv[-1] / (valid_mv[0] + 1) - 1) * 100, 1)
        else:
            features["mv_median"] = 0.0
            features["mv_change"] = 0.0
    else:
        features["mv_median"] = 0.0
        features["mv_change"] = 0.0

    return features


# ============================================================
# 回踩买点体系（金叉波段内的支撑回踩事件）
# ============================================================

@register("pullback_zfzs")
def detect_pullback_zfzs(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """zfzs 金叉波段内回踩支撑线事件。

    对每个 zhixing_fast 金叉 → MA20/60 死叉波段，记录前 max_pullbacks 次
    回踩（low 自上方进入支撑线 ±band_pct 带，离开带后重新计数）。

    输出每次回踩的：距线距离、金叉后天数、波段内截至当日最大回撤、
    前5天涨幅、波段前高（供 analyze 的 cross_above 创新高判定）。
    base = 回踩日收盘价。
    """
    line_name = p.get("line", "zf")            # zf | zs
    band_pct = float(p.get("band_pct", 3.0))
    max_pullbacks = int(p.get("max_pullbacks", 7))
    min_days = int(p.get("min_days", 20))

    close = df["close"].values
    low = df["low"].values
    dates = df["date"].values
    n = len(close)

    z_fast = zhixing_fast(df["close"]).values
    z_slow = zhixing_slow(df["close"]).values
    ma20 = sma(df["close"], 20).values
    ma60 = sma(df["close"], 60).values
    line = z_fast if line_name == "zf" else z_slow
    line_label = "zhixing_fast" if line_name == "zf" else "zhixing_slow"

    records = []
    in_position = False
    entry_idx = None

    def _band_dist(i):
        if np.isnan(line[i]):
            return None
        return (low[i] - line[i]) / line[i] * 100

    for i in range(1, n):
        if np.isnan(z_fast[i]) or np.isnan(z_slow[i]) or np.isnan(ma20[i]) or np.isnan(ma60[i]):
            continue

        if not in_position:
            if z_fast[i - 1] <= z_slow[i - 1] and z_fast[i] > z_slow[i]:
                in_position = True
                entry_idx = i
                pullback_no = 0
                in_band = False
                wave_peak = close[i]
        else:
            if close[i] > wave_peak:
                wave_peak = close[i]

            d = _band_dist(i)
            if d is not None:
                if not in_band and d <= band_pct:
                    # 进入回踩带 = 一次回踩
                    in_band = True
                    pullback_no += 1
                    if pullback_no <= max_pullbacks:
                        # 波段内截至当日最大回撤
                        seg = close[entry_idx:i + 1]
                        peak_so_far = float(np.max(seg))
                        dd_so_far = (peak_so_far - close[i]) / peak_so_far * 100
                        # 前5天涨幅
                        ret5 = (close[i] / close[max(entry_idx, i - 5)] - 1) * 100 \
                            if i - 5 >= entry_idx and close[i - 5] > 0 else np.nan
                        # 波段前高（回踩日前，不含当日）
                        prior_high = float(np.max(close[entry_idx:i])) if i > entry_idx else close[entry_idx]
                        dist_bin = _dist_bin(d)
                        records.append({
                            "code": code,
                            "line": line_label,
                            "entry_date": pd.Timestamp(dates[entry_idx]).strftime("%Y-%m-%d"),
                            "pullback_no": pullback_no,
                            "no_bin": f"第{min(pullback_no, 5)}次" if pullback_no <= 5 else "6+次",
                            "pullback_date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                            "days_since_cross": i - entry_idx,
                            "days_bin": _days_bin(i - entry_idx),
                            "dist_pct": round(d, 2),
                            "dist_bin": dist_bin,
                            "broke_line": bool(d < 0),
                            "wave_dd_so_far": round(dd_so_far, 1),
                            "dd_bin": _dd_bin(dd_so_far),
                            "ret5_pct": round(ret5, 2) if not np.isnan(ret5) else np.nan,
                            "prior_high": round(prior_high, 2),
                            "base_price": round(float(close[i]), 2),
                        })
                elif in_band and d > band_pct:
                    in_band = False

            if ma20[i - 1] >= ma60[i - 1] and ma20[i] < ma60[i]:
                in_position = False

    return records


def _dist_bin(d: float) -> str:
    if d >= 2.0:
        return "+2~+3%"
    if d >= 1.0:
        return "+1~+2%"
    if d >= 0.0:
        return "0~+1%"
    if d >= -1.0:
        return "-1~0%"
    if d >= -2.0:
        return "-2~-1%"
    return "-3~-2%"


def _days_bin(days: int) -> str:
    if days < 10:
        return "5-10天"
    if days < 15:
        return "10-15天"
    if days < 20:
        return "15-20天"
    if days < 30:
        return "20-30天"
    if days < 50:
        return "30-50天"
    return "50天+"


def _dd_bin(dd: float) -> str:
    if dd < 5:
        return "0-5%"
    if dd < 10:
        return "5-10%"
    if dd < 15:
        return "10-15%"
    if dd < 20:
        return "15-20%"
    if dd < 30:
        return "20-30%"
    return "30-50%"


# ============================================================
# 底部反转信号族
# ============================================================

@register("fractal_bottom")
def detect_fractal_bottom(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """底分型（三K：中间K的 high/low 均低于两侧）+ 深跌过滤 + 力度特征。

    口径：只记录确认日（第3根K）前 20 天跌幅 >= min_drop_20d（默认 10%）
    的底分型。base = 确认日收盘价；fractal_low = 中间K最低价
    （cross_below 阈值，判定是否破位）。

    力度特征（沿用旧文档《分型力度分析》口径）：
      depth（中间K低点低于两侧低点的较小幅度%）、high_suppress、
      amp_ratio、vol_vs_neighbors、lower_shadow、strength=depth×suppress。
    """
    min_drop_20d = float(p.get("min_drop_20d", 10.0))

    close = df["close"].values.astype(float)
    open_ = df["open"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    dates = df["date"].values
    n = len(close)

    vol_ma20 = sma(df["volume"], 20).values
    ma60 = sma(df["close"], 60).values
    zs = zhixing_slow(df["close"]).values

    records = []
    for i in range(1, n - 1):
        if not (high[i] < high[i - 1] and high[i] < high[i + 1]
                and low[i] < low[i - 1] and low[i] < low[i + 1]):
            continue
        ci = i + 1  # 确认日（第3根K）
        if ci >= n:
            continue
        # 前 20 天跌幅（确认日收盘 vs 20 天前收盘）
        ref = ci - 20
        if ref < 0 or close[ref] <= 0:
            continue
        drop_20d = (close[ci] / close[ref] - 1) * 100
        if drop_20d > -min_drop_20d:
            continue

        # 连跌天数（截至中间K）
        consec = 0
        j = i
        while j - 1 >= 0 and close[j] < close[j - 1]:
            consec += 1
            j -= 1

        vol_ratio = volume[i] / vol_ma20[i] if vol_ma20[i] and not np.isnan(vol_ma20[i]) else np.nan

        # 力度特征
        depth = min((low[i - 1] - low[i]) / low[i] * 100 if low[i] > 0 else 0,
                    (low[i + 1] - low[i]) / low[i] * 100 if low[i] > 0 else 0)
        high_suppress = min((high[i - 1] - high[i]) / high[i] * 100 if high[i] > 0 else 0,
                            (high[i + 1] - high[i]) / high[i] * 100 if high[i] > 0 else 0)
        amp_mid = (high[i] - low[i]) / close[i] * 100 if close[i] > 0 else 0
        amp_side = ((high[i - 1] - low[i - 1]) / close[i - 1] * 100 if close[i - 1] > 0 else 0
                    + (high[i + 1] - low[i + 1]) / close[i + 1] * 100 if close[i + 1] > 0 else 0) / 2
        amp_ratio = amp_mid / (amp_side + 0.01) if amp_side > 0 else np.nan
        vol_vs_nb = volume[i] / ((volume[i - 1] + volume[i + 1]) / 2 + 1)
        bar_range = high[i] - low[i]
        lower_shadow = (close[i] - low[i]) / (bar_range + 1e-10) if bar_range > 0 else 0.5
        strength = depth * high_suppress

        records.append({
            "code": code,
            "fractal_date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
            "confirm_date": pd.Timestamp(dates[ci]).strftime("%Y-%m-%d"),
            "fractal_low": round(float(low[i]), 2),
            "drop_20d": round(drop_20d, 1),
            "drop_bin": ("< -40%" if drop_20d <= -40 else
                         "-40~-30%" if drop_20d <= -30 else
                         "-30~-20%" if drop_20d <= -20 else
                         "-20~-10%" if drop_20d <= -10 else "> -10%"),
            "consec_down": consec,
            "consec_bin": ("≥5天" if consec >= 5 else "3-4天" if consec >= 3 else "0-2天"),
            "vol_ratio": round(float(vol_ratio), 2) if not np.isnan(vol_ratio) else np.nan,
            "vol_bin": ("≥2" if vol_ratio >= 2 else "1.5-2" if vol_ratio >= 1.5
                        else "1-1.5" if vol_ratio >= 1 else "<1"),
            "below_ma60": bool(not np.isnan(ma60[ci]) and close[ci] < ma60[ci]),
            "below_zs": bool(not np.isnan(zs[ci]) and close[ci] < zs[ci]),
            "depth": round(depth, 2),
            "depth_bin": ("<1%" if depth < 1 else "1-3%" if depth < 3
                          else "3-5%" if depth < 5 else "≥5%"),
            "high_suppress": round(high_suppress, 2),
            "amp_ratio": round(float(amp_ratio), 2) if not np.isnan(amp_ratio) else np.nan,
            "vol_vs_neighbors": round(float(vol_vs_nb), 2),
            "lower_shadow": round(float(lower_shadow), 2),
            "strength": round(float(strength), 2),
            "base_price": round(float(close[ci]), 2),
        })

    return records


@register("fractal_top")
def detect_fractal_top(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """顶分型（三K：中间K的 high/low 均高于两侧）+ 前期涨幅过滤 + 力度特征。

    口径：只记录确认日（第3根K）前 20 天涨幅 >= min_rise_20d（默认 5%）
    的顶分型。base = 确认日收盘价；fractal_high = 中间K最高价
    （cross_above 阈值，判定是否创新高——顶分型的"失败"信号）。
    力度特征与底分型对称：depth（中间K高点超出两侧高点的较小幅度%）、
    low_suppress、upper_shadow、strength=depth×low_suppress。
    """
    min_rise_20d = float(p.get("min_rise_20d", 5.0))

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    dates = df["date"].values
    n = len(close)

    vol_ma20 = sma(df["volume"], 20).values
    ma60 = sma(df["close"], 60).values
    zs = zhixing_slow(df["close"]).values

    records = []
    for i in range(1, n - 1):
        if not (high[i] > high[i - 1] and high[i] > high[i + 1]
                and low[i] > low[i - 1] and low[i] > low[i + 1]):
            continue
        ci = i + 1
        if ci >= n:
            continue
        ref = ci - 20
        if ref < 0 or close[ref] <= 0:
            continue
        rise_20d = (close[ci] / close[ref] - 1) * 100
        if rise_20d < min_rise_20d:
            continue

        # 连涨天数（截至中间K）
        consec = 0
        j = i
        while j - 1 >= 0 and close[j] > close[j - 1]:
            consec += 1
            j -= 1

        vol_ratio = volume[i] / vol_ma20[i] if vol_ma20[i] and not np.isnan(vol_ma20[i]) else np.nan

        # 力度（与底分型对称）
        depth = min((high[i] - high[i - 1]) / high[i - 1] * 100 if high[i - 1] > 0 else 0,
                    (high[i] - high[i + 1]) / high[i + 1] * 100 if high[i + 1] > 0 else 0)
        low_suppress = min((low[i] - low[i - 1]) / low[i - 1] * 100 if low[i - 1] > 0 else 0,
                           (low[i] - low[i + 1]) / low[i + 1] * 100 if low[i + 1] > 0 else 0)
        bar_range = high[i] - low[i]
        upper_shadow = (high[i] - close[i]) / (bar_range + 1e-10) if bar_range > 0 else 0.5
        vol_vs_nb = volume[i] / ((volume[i - 1] + volume[i + 1]) / 2 + 1)
        strength = depth * low_suppress

        records.append({
            "code": code,
            "fractal_date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
            "confirm_date": pd.Timestamp(dates[ci]).strftime("%Y-%m-%d"),
            "fractal_high": round(float(high[i]), 2),
            "rise_20d": round(rise_20d, 1),
            "rise_bin": ("≥40%" if rise_20d >= 40 else
                         "30-40%" if rise_20d >= 30 else
                         "20-30%" if rise_20d >= 20 else
                         "10-20%" if rise_20d >= 10 else "5-10%"),
            "consec_up": consec,
            "consec_bin": ("≥5天" if consec >= 5 else "3-4天" if consec >= 3 else "0-2天"),
            "vol_ratio": round(float(vol_ratio), 2) if not np.isnan(vol_ratio) else np.nan,
            "vol_bin": ("≥2" if vol_ratio >= 2 else "1.5-2" if vol_ratio >= 1.5
                        else "1-1.5" if vol_ratio >= 1 else "<1"),
            "above_ma60": bool(not np.isnan(ma60[ci]) and close[ci] > ma60[ci]),
            "above_zs": bool(not np.isnan(zs[ci]) and close[ci] > zs[ci]),
            "depth": round(depth, 2),
            "depth_bin": ("<1%" if depth < 1 else "1-3%" if depth < 3
                          else "3-5%" if depth < 5 else "≥5%"),
            "low_suppress": round(low_suppress, 2),
            "vol_vs_neighbors": round(float(vol_vs_nb), 2),
            "upper_shadow": round(float(upper_shadow), 2),
            "strength": round(float(strength), 2),
            "base_price": round(float(close[ci]), 2),
        })

    return records


@register("engulf")
def detect_engulf(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """看跌吞没形态：阳线实体被紧邻阴线实体完全吞没。

    定义（红K 1.1→1.2，绿K 1.2→1.1）：
      K1 = 阳线（close > open）；K2 = 紧邻阴线（close < open）；
      完全吞没 = K2.open >= K1.close 且 K2.close <= K1.open
      （require_full=False 时放宽为 engulf_pct >= min_engulf_pct）。

    engulf_pct = (K1.open - K2.close) / K1.open × 100（>0 = 吞没过阳线开盘价）。
    过滤：K1 实体涨幅 >= min_red_body（默认 1%）、
          前 20 天涨幅 >= min_rise_20d（默认 5%）。
    base = 绿K收盘价。
    """
    require_full = bool(p.get("require_full", True))
    min_engulf_pct = float(p.get("min_engulf_pct", 0.0))
    min_red_body = float(p.get("min_red_body", 1.0))
    min_rise_20d = float(p.get("min_rise_20d", 5.0))

    close = df["close"].values.astype(float)
    open_ = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)
    dates = df["date"].values
    n = len(close)

    vol_ma20 = sma(df["volume"], 20).values
    ma60 = sma(df["close"], 60).values
    zs = zhixing_slow(df["close"]).values

    records = []
    for i2 in range(1, n):
        i1 = i2 - 1
        # K1 阳线、K2 阴线
        if not (close[i1] > open_[i1] and close[i2] < open_[i2]):
            continue
        red_body = (close[i1] / open_[i1] - 1) * 100 if open_[i1] > 0 else 0
        if red_body < min_red_body:
            continue
        green_body = (1 - close[i2] / open_[i2]) * 100 if open_[i2] > 0 else 0
        # 完全吞没 / 部分吞没
        if open_[i1] > 0:
            engulf_pct = (open_[i1] - close[i2]) / open_[i1] * 100
        else:
            engulf_pct = 0.0
        if require_full:
            if not (open_[i2] >= close[i1] and close[i2] <= open_[i1]):
                continue
        else:
            if engulf_pct < min_engulf_pct:
                continue

        # 前 20 天涨幅（以绿K日收盘计）
        ref = i2 - 20
        if ref < 0 or close[ref] <= 0:
            continue
        rise_20d = (close[i2] / close[ref] - 1) * 100
        if rise_20d < min_rise_20d:
            continue

        vr1 = volume[i1] / vol_ma20[i1] if vol_ma20[i1] and not np.isnan(vol_ma20[i1]) else np.nan
        vr2 = volume[i2] / vol_ma20[i2] if vol_ma20[i2] and not np.isnan(vol_ma20[i2]) else np.nan

        records.append({
            "code": code,
            "red_date": pd.Timestamp(dates[i1]).strftime("%Y-%m-%d"),
            "green_date": pd.Timestamp(dates[i2]).strftime("%Y-%m-%d"),
            "engulf_date": pd.Timestamp(dates[i2]).strftime("%Y-%m-%d"),
            "red_body_pct": round(red_body, 2),
            "red_body_bin": ("1-2%" if red_body < 2 else "2-4%" if red_body < 4
                             else "4-7%" if red_body < 7 else "≥7%"),
            "green_body_pct": round(green_body, 2),
            "engulf_pct": round(engulf_pct, 2),
            "engulf_bin": ("<0%(部分)" if engulf_pct < 0 else
                           "0-1%" if engulf_pct < 1 else
                           "1-3%" if engulf_pct < 3 else "≥3%"),
            "rise_20d": round(rise_20d, 1),
            "rise_bin": ("≥40%" if rise_20d >= 40 else
                         "20-40%" if rise_20d >= 20 else
                         "10-20%" if rise_20d >= 10 else "5-10%"),
            "vol_ratio_red": round(float(vr1), 2) if not np.isnan(vr1) else np.nan,
            "vol_ratio_green": round(float(vr2), 2) if not np.isnan(vr2) else np.nan,
            "vol_bin": ("≥2" if (not np.isnan(vr2) and vr2 >= 2) else
                        "1-2" if (not np.isnan(vr2) and vr2 >= 1) else "<1"),
            "above_ma60": bool(not np.isnan(ma60[i2]) and close[i2] > ma60[i2]),
            "above_zs": bool(not np.isnan(zs[i2]) and close[i2] > zs[i2]),
            "base_price": round(float(close[i2]), 2),
        })

    return records


@register("breakout_pullback")
def detect_breakout_pullback(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """放量突破后 10 根K内缩量回调至突破位的形态。

    定义：
      突破日 t0: volume >= vol_ma20 × vol_multi 且收盘创 lookback 日收盘新高;
      回调确认 t1 ∈ (t0, t0+max_pullback_days]: 首个收盘 <= 突破前平台
        （lookback 日最高收盘 × (1 + tol_pct%)）;
      缩量: 回调段均量 <= vol_ma20 × pullback_vol_max。
      未在窗口内完成回调或量能超标则不记录。

    base = 回调确认日收盘（缩量回调买点）；breakout_close = 突破日收盘
    （cross_above 阈值，判定收复突破日高点）。
    """
    lookback = int(p.get("lookback", 20))
    vol_multi = float(p.get("vol_multi", 2.0))
    max_pullback_days = int(p.get("max_pullback_days", 10))
    tol_pct = float(p.get("tol_pct", 2.0))
    pullback_vol_max = float(p.get("pullback_vol_max", 1.0))

    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float)
    dates = df["date"].values
    n = len(close)

    vol_ma20 = sma(df["volume"], 20).values

    records = []
    i = max(lookback, 25)
    while i < n:
        t0 = i
        if np.isnan(vol_ma20[t0]) or vol_ma20[t0] <= 0:
            i += 1
            continue
        # 突破日：放量 + 收盘创 lookback 日新高
        prior_max = float(np.max(close[t0 - lookback:t0]))
        if not (volume[t0] >= vol_ma20[t0] * vol_multi and close[t0] > prior_max):
            i += 1
            continue

        # 回调确认：窗口内首个收盘回到突破平台附近
        t1 = None
        for k in range(t0 + 1, min(t0 + max_pullback_days + 1, n)):
            if close[k] <= prior_max * (1 + tol_pct / 100):
                t1 = k
                break
        if t1 is None:
            i += 1
            continue

        # 缩量约束：回调段均量
        seg_vol = float(np.mean(volume[t0 + 1:t1 + 1]))
        if seg_vol > vol_ma20[t1] * pullback_vol_max:
            i += 1
            continue

        breakout_gain = (close[t0] / prior_max - 1) * 100 if prior_max > 0 else 0
        pullback_depth = (close[t0] - close[t1]) / close[t0] * 100 if close[t0] > 0 else 0
        records.append({
            "code": code,
            "breakout_date": pd.Timestamp(dates[t0]).strftime("%Y-%m-%d"),
            "confirm_date": pd.Timestamp(dates[t1]).strftime("%Y-%m-%d"),
            "pullback_days": t1 - t0,
            "days_bin": ("1-3天" if t1 - t0 <= 3 else "4-6天" if t1 - t0 <= 6 else "7-10天"),
            "breakout_gain_pct": round(breakout_gain, 2),
            "gain_bin": ("<2%" if breakout_gain < 2 else "2-5%" if breakout_gain < 5 else "≥5%"),
            "breakout_vr": round(volume[t0] / vol_ma20[t0], 2),
            "pullback_vr": round(seg_vol / vol_ma20[t1], 2) if vol_ma20[t1] > 0 else np.nan,
            "shrink_bin": ("<0.7" if (vol_ma20[t1] > 0 and seg_vol / vol_ma20[t1] < 0.7) else
                           "0.7-1.0" if vol_ma20[t1] > 0 else "≥1.0"),
            "pullback_depth_pct": round(pullback_depth, 2),
            "breakout_close": round(float(close[t0]), 2),
            "base_price": round(float(close[t1]), 2),
        })
        i = t1 + 1

    return records


@register("effort_defeat")
def detect_effort_defeat(df: pd.DataFrame, code: str, p: dict) -> list[dict]:
    """看涨击败（威科夫努力 vs 结果）：放量阴线/十字星被随后阳线收破高点。

    - 关键K：量比 >= vol_ratio_min（默认 1.5），阴线或十字星（实体占比 < 0.3）；
    - 击败K：关键K后 gap_days_max（默认 5）天内，阳线收盘 > 关键K最高价；
    - 努力比 = 击败K量比 / 关键K量比（越小越可靠）；
    - 每个关键K只记首次击败（与旧文档代码的全组合口径不同，已去重）。
    base = 击败日收盘价；key_low = 关键K最低价（cross_below 破位判定）。
    """
    vol_ratio_min = float(p.get("vol_ratio_min", 1.5))
    gap_days_max = int(p.get("gap_days_max", 5))
    min_drop_10d = float(p.get("min_drop_10d", 0.0))

    close = df["close"].values.astype(float)
    open_ = df["open"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    dates = df["date"].values
    n = len(close)

    vol_ma20 = sma(df["volume"], 20).values

    records = []
    used_key = set()  # 已被击败的关键K索引

    for i in range(60, n):
        if np.isnan(vol_ma20[i]):
            continue
        vol_ratio_i = volume[i] / (vol_ma20[i] + 1e-10)
        bar_range_i = high[i] - low[i]
        if bar_range_i <= 0:
            continue
        body_i = close[i] - open_[i]
        if body_i <= 0:  # 击败K必须是阳线
            continue

        # 往前 1~gap_days_max 天找关键K
        for lookback in range(1, gap_days_max + 1):
            ki = i - lookback
            if ki < 60 or ki in used_key:
                continue
            if np.isnan(vol_ma20[ki]):
                continue
            vol_ratio_k = volume[ki] / (vol_ma20[ki] + 1e-10)
            if vol_ratio_k < vol_ratio_min:
                continue
            bar_range_k = high[ki] - low[ki]
            if bar_range_k <= 0:
                continue
            body_k = close[ki] - open_[ki]
            body_pct_k = abs(body_k) / bar_range_k
            # 关键K为阴线或十字星
            if not (body_k <= 0 or body_pct_k < 0.3):
                continue
            # 阳线收破关键K高点
            if close[i] <= high[ki]:
                continue

            used_key.add(ki)
            effort_ratio = vol_ratio_i / vol_ratio_k
            ref = ki - 10
            drop_10d = (close[ki] / close[ref] - 1) * 100 if ref >= 0 and close[ref] > 0 else np.nan
            if not np.isnan(drop_10d) and drop_10d > -min_drop_10d:
                continue

            records.append({
                "code": code,
                "key_date": pd.Timestamp(dates[ki]).strftime("%Y-%m-%d"),
                "defeat_date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                "gap_days": lookback,
                "gap_bin": ("1天" if lookback == 1 else "2-3天" if lookback <= 3 else "4-5天"),
                "vol_ratio_key": round(vol_ratio_k, 2),
                "effort_ratio": round(effort_ratio, 3),
                "effort_bin": ("<0.3" if effort_ratio < 0.3 else
                               "0.3-0.6" if effort_ratio < 0.6 else
                               "0.6-1.2" if effort_ratio < 1.2 else "≥1.2"),
                "drop_10d": round(float(drop_10d), 1) if not np.isnan(drop_10d) else np.nan,
                "drop_bin": ("< -20%" if (not np.isnan(drop_10d) and drop_10d <= -20) else
                             "-20~-15%" if (not np.isnan(drop_10d) and drop_10d <= -15) else
                             "-15~-5%" if (not np.isnan(drop_10d) and drop_10d <= -5) else "> -5%"),
                "key_low": round(float(low[ki]), 2),
                "base_price": round(float(close[i]), 2),
            })
            break  # 每个击败K只配首个关键K

    return records
