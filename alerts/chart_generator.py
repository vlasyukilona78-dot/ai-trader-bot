from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

from core.liquidation_map import LiquidationMap, build_liquidation_map
from core.volume_profile import VolumeProfileLevels


def _compute_macd(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if {"macd", "signal", "hist"}.issubset(out.columns):
        return out
    close = pd.to_numeric(out.get("close"), errors="coerce")
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    out["macd"] = macd
    out["signal"] = signal
    out["hist"] = hist
    return out


def _fmt_level(value: float) -> str:
    if value >= 1000:
        return f"{value:,.2f}".replace(",", " ")
    if value >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def _price_tick_formatter(value, _pos):
    return _fmt_level(value)


def _macd_tick_formatter(value, _pos):
    return f"{value:.4f}" if abs(value) < 1 else f"{value:.3f}"


def _price_scale(value: float) -> float:
    return max(abs(float(value)), 1e-6)


def _draw_candles(ax, frame: pd.DataFrame, x_values):
    if len(x_values) >= 2:
        width = max((x_values[1] - x_values[0]) * 0.68, 1e-6)
    else:
        width = 0.00035
    for x, (_, row) in zip(x_values, frame.iterrows()):
        open_px = float(row["open"])
        high_px = float(row["high"])
        low_px = float(row["low"])
        close_px = float(row["close"])
        color = "#5ad692" if close_px >= open_px else "#ff8b7a"
        ax.vlines(x, low_px, high_px, color=color, linewidth=1.12, alpha=0.97, zorder=3)
        body_low = min(open_px, close_px)
        body_height = max(abs(close_px - open_px), max(close_px * 0.00017, 1e-8))
        ax.add_patch(
            Rectangle(
                (x - width / 2.0, body_low),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.98,
                zorder=4,
            )
        )


def _style_axis(ax, *, is_macd: bool = False):
    ax.set_facecolor("#101828" if not is_macd else "#0f1726")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.55, alpha=0.12, color="#d9e2f1")
    ax.grid(True, axis="x", linestyle="-", linewidth=0.45, alpha=0.06, color="#d9e2f1")
    for spine in ax.spines.values():
        spine.set_color("#25324b")
        spine.set_linewidth(0.9)
    ax.tick_params(colors="#b8c4da", labelsize=7, length=0, pad=4)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


def _segment_bounds(
    x_values,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    start_frac: float = 0.70,
    length_frac: float = 0.22,
):
    x_left = float(x_values[0])
    x_right = float(x_values[-1])
    span = max(x_right - x_left, 1e-8)

    if start_index is not None:
        idx = max(0, min(int(start_index), len(x_values) - 1))
        seg_start = float(x_values[idx])
    else:
        seg_start = x_left + span * start_frac

    if end_index is not None:
        idx = max(0, min(int(end_index), len(x_values) - 1))
        seg_end = float(x_values[idx])
    else:
        seg_end = min(x_right, seg_start + span * length_frac)

    if seg_end <= seg_start:
        seg_end = min(x_right, seg_start + span * max(length_frac, 0.08))
    return seg_start, seg_end


def _find_recent_level_touch_index(frame: pd.DataFrame, value: float, *, lookback: int = 88) -> int | None:
    if value <= 0 or frame.empty:
        return None
    start = max(0, len(frame) - lookback)
    price_span = max(float(frame["high"].max()) - float(frame["low"].min()), 1e-8)
    tolerance = max(abs(value) * 0.0011, price_span * 0.004)

    for idx in range(len(frame) - 1, start - 1, -1):
        row = frame.iloc[idx]
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        open_px = float(row["open"])
        if low - tolerance <= value <= high + tolerance:
            return idx
        if min(abs(close - value), abs(open_px - value)) <= tolerance:
            return idx
    return None


def _touch_segment(
    frame: pd.DataFrame,
    value: float,
    *,
    lookback: int = 88,
    lead_bars: int = 2,
    tail_bars: int = 18,
) -> tuple[int | None, int | None]:
    touch_idx = _find_recent_level_touch_index(frame, value, lookback=lookback)
    if touch_idx is None:
        return None, None
    start_idx = max(0, touch_idx - lead_bars)
    end_idx = min(len(frame) - 1, touch_idx + tail_bars)
    return start_idx, end_idx


def _annotate_level(
    ax,
    x_values,
    value: float,
    color: str,
    label: str,
    *,
    linestyle: str = "--",
    linewidth: float = 1.0,
    start_index: int | None = None,
    end_index: int | None = None,
    start_frac: float = 0.70,
    length_frac: float = 0.22,
    show_value: bool = False,
    font_size: float = 6.9,
):
    if value <= 0:
        return
    xmin, xmax = ax.get_xlim()
    pad = (xmax - xmin) * 0.014
    seg_start, seg_end = _segment_bounds(
        x_values,
        start_index=start_index,
        end_index=end_index,
        start_frac=start_frac,
        length_frac=length_frac,
    )
    ax.hlines(value, seg_start, seg_end, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.86, zorder=6)
    label_text = f"{label}  {_fmt_level(value)}" if show_value else label
    ax.text(
        min(xmax - pad, seg_end + (xmax - xmin) * 0.010),
        value,
        label_text,
        ha="left",
        va="center",
        fontsize=font_size,
        color=color,
        bbox={"facecolor": "#0d1726", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        zorder=7,
    )


def _draw_trade_zone(ax, x_values, *, side: str, entry: float, tp: float, sl: float):
    if entry <= 0 or tp <= 0 or sl <= 0:
        return
    seg_start, seg_end = _segment_bounds(x_values, start_frac=0.80, length_frac=0.11)
    width = max(seg_end - seg_start, 1e-8)
    if str(side).upper() == "SHORT":
        reward_low, reward_high = min(tp, entry), max(tp, entry)
        risk_low, risk_high = min(entry, sl), max(entry, sl)
    else:
        reward_low, reward_high = min(entry, tp), max(entry, tp)
        risk_low, risk_high = min(sl, entry), max(sl, entry)

    ax.add_patch(
        Rectangle(
            (seg_start, reward_low),
            width,
            max(reward_high - reward_low, 1e-8),
            facecolor="#1e7f57",
            edgecolor="none",
            alpha=0.10,
            zorder=1.4,
        )
    )
    ax.add_patch(
        Rectangle(
            (seg_start, risk_low),
            width,
            max(risk_high - risk_low, 1e-8),
            facecolor="#9d3850",
            edgecolor="none",
            alpha=0.08,
            zorder=1.35,
        )
        )


def _liquidation_label_x(ax, seg_start: float, seg_end: float) -> float:
    xmin, xmax = ax.get_xlim()
    span = max(xmax - xmin, 1e-8)
    width = max(seg_end - seg_start, 1e-8)
    ideal = seg_start + width * 0.72
    return min(ideal, xmax - span * 0.16)


def _draw_recent_focus(ax, x_values, *, bars: int = 18):
    if len(x_values) < 6:
        return
    start_idx = max(0, len(x_values) - bars)
    seg_start = float(x_values[start_idx])
    seg_end = float(x_values[-1])
    ax.axvspan(seg_start, seg_end, color="#173149", alpha=0.10, zorder=0.4)


def _draw_entry_pump_focus(ax, frame: pd.DataFrame, x_values):
    if frame.empty or len(frame) < 18:
        return
    last_close = float(pd.to_numeric(frame["close"], errors="coerce").dropna().iloc[-1])
    pump_window = _find_recent_entry_pump_window(frame, last_close=last_close)
    if pump_window is None:
        return

    pump_start_rel, peak_rel, pump_pct = pump_window
    if pump_pct < (0.02 if last_close < 0.02 else 0.015):
        return

    seg_start = float(x_values[max(0, pump_start_rel)])
    seg_end = float(x_values[min(int(peak_rel), len(x_values) - 1)])
    if seg_end <= seg_start:
        return

    ax.axvspan(seg_start, seg_end, color="#f2b868", alpha=0.045, zorder=0.55)
    ax.vlines(seg_start, float(frame.iloc[pump_start_rel]["low"]), float(frame.iloc[pump_start_rel]["high"]), color="#f3bf74", linewidth=0.9, alpha=0.22, zorder=1.9)
    ax.vlines(seg_end, float(frame.iloc[peak_rel]["low"]), float(frame.iloc[peak_rel]["high"]), color="#f3bf74", linewidth=0.9, alpha=0.24, zorder=1.9)


def _draw_last_price_marker(ax, x_values, price: float, *, color: str = "#83d6ff"):
    if price <= 0:
        return
    xmin, xmax = ax.get_xlim()
    ax.hlines(
        price,
        float(x_values[0]),
        xmax,
        color=color,
        linestyle=(0, (2.2, 2.6)),
        linewidth=0.9,
        alpha=0.42,
        zorder=1.8,
    )
    ax.text(
        xmax - (xmax - xmin) * 0.002,
        price,
        _fmt_level(price),
        ha="left",
        va="center",
        fontsize=7,
        color="#ebf4ff",
        bbox={"facecolor": "#1b5d82", "edgecolor": "none", "alpha": 0.92, "pad": 1.0},
        zorder=8,
    )


def _resolve_chart_timezone():
    configured = str(os.getenv("BOT_CHART_TIMEZONE", "")).strip()
    if configured and ZoneInfo is not None:
        try:
            return ZoneInfo(configured)
        except Exception:
            pass
    try:
        return datetime.now().astimezone().tzinfo or timezone.utc
    except Exception:
        return timezone.utc


def _clip_isolated_recent_wicks(
    lows: pd.Series,
    highs: pd.Series,
    *,
    last_close: float,
) -> tuple[float, float]:
    raw_low = float(lows.min())
    raw_high = float(highs.max())
    if len(lows) < 8 or len(highs) < 8:
        return raw_low, raw_high

    robust_low = float(lows.quantile(0.16))
    robust_high = float(highs.quantile(0.84))
    robust_span = max(robust_high - robust_low, _price_scale(last_close) * 0.01)
    upper_threshold = robust_high + robust_span * 0.52
    lower_threshold = robust_low - robust_span * 0.52

    upper_extreme_count = int((highs >= upper_threshold).sum())
    lower_extreme_count = int((lows <= lower_threshold).sum())

    clipped_high = raw_high
    clipped_low = raw_low
    if raw_high > upper_threshold and upper_extreme_count <= 2:
        clipped_high = upper_threshold
    if raw_low < lower_threshold and lower_extreme_count <= 2:
        clipped_low = lower_threshold

    return clipped_low, clipped_high


def _append_nearby_level(
    focus_levels: list[float],
    value: float | None,
    *,
    last_close: float,
    max_distance_pct: float,
):
    if value is None or value <= 0:
        return
    distance_pct = abs(float(value) - last_close) / max(abs(last_close), 1e-8)
    if distance_pct <= max_distance_pct:
        focus_levels.append(float(value))


def _entry_level_visible(
    value: float | None,
    *,
    last_close: float,
    y_min: float,
    y_max: float,
    max_distance_pct: float = 0.12,
) -> bool:
    if value is None or value <= 0:
        return False
    value_f = float(value)
    distance_pct = abs(value_f - last_close) / max(abs(last_close), 1e-8)
    if distance_pct <= max_distance_pct:
        return True
    span = max(y_max - y_min, 1e-8)
    return (y_min - span * 0.04) <= value_f <= (y_max + span * 0.04)


def _select_intraday_profile_levels(
    volume_profile: VolumeProfileLevels | None,
    *,
    last_close: float,
    y_min: float,
    y_max: float,
) -> list[tuple[str, float, str, str, float, int, float, float]]:
    if volume_profile is None:
        return []

    specs = {
        "POC": (float(volume_profile.poc), "#f6d06a", "-", 1.05, 24, 0.69, 0.13, 0.088),
        "VAH": (float(volume_profile.vah), "#f0b35f", "--", 0.95, 20, 0.74, 0.12, 0.064),
        "VAL": (float(volume_profile.val), "#8fd36a", "--", 0.95, 20, 0.64, 0.12, 0.064),
    }

    visible: list[tuple[float, tuple[str, float, str, str, float, int, float, float]]] = []
    for label, (value, color, linestyle, linewidth, tail_bars, start_frac, length_frac, max_distance_pct) in specs.items():
        if not _entry_level_visible(
            value,
            last_close=last_close,
            y_min=y_min,
            y_max=y_max,
            max_distance_pct=max_distance_pct,
        ):
            continue
        distance = abs(value - last_close) / max(abs(last_close), 1e-8)
        if label == "POC":
            distance *= 0.82
        visible.append(
            (
                distance,
                (label, value, color, linestyle, linewidth, tail_bars, start_frac, length_frac),
            )
        )

    if not visible:
        return []

    visible.sort(key=lambda item: item[0])
    selected: list[tuple[str, float, str, str, float, int, float, float]] = []
    seen: set[str] = set()

    for _, spec in visible:
        label = spec[0]
        if label == "POC":
            selected.append(spec)
            seen.add(label)
            break

    for _, spec in visible:
        label = spec[0]
        if label in seen:
            continue
        selected.append(spec)
        seen.add(label)
        if len(selected) >= 2:
            break

    return selected


def _build_local_trade_focus_levels(
    frame: pd.DataFrame,
    *,
    last_close: float,
    entry: float,
    tp: float,
    sl: float,
    volume_profile: VolumeProfileLevels | None,
) -> list[float]:
    local = frame.tail(min(len(frame), 34))
    local_lows = pd.to_numeric(local["low"], errors="coerce").dropna()
    local_highs = pd.to_numeric(local["high"], errors="coerce").dropna()
    if local_lows.empty or local_highs.empty:
        return [last_close, entry, tp, sl]

    tail_size = min(len(local_lows), 16)
    tail_lows = local_lows.tail(tail_size)
    tail_highs = local_highs.tail(tail_size)
    local_swing_low, local_swing_high = _clip_isolated_recent_wicks(
        tail_lows,
        tail_highs,
        last_close=last_close,
    )
    levels = [
        local_swing_low,
        local_swing_high,
        float(local_lows.quantile(0.12)),
        float(local_highs.quantile(0.88)),
        last_close,
    ]

    for value in (entry, tp, sl):
        if value and value > 0:
            levels.append(float(value))

    if volume_profile is not None:
        for value in (volume_profile.val, volume_profile.vah, volume_profile.poc):
            _append_nearby_level(levels, value, last_close=last_close, max_distance_pct=0.085)

    for col in ("ema20", "ema50", "vwap"):
        if col in local.columns:
            series = pd.to_numeric(local[col], errors="coerce").dropna()
            if not series.empty:
                _append_nearby_level(levels, float(series.iloc[-1]), last_close=last_close, max_distance_pct=0.13)

    return levels


def _build_local_recent_zoom_levels(
    frame: pd.DataFrame,
    *,
    last_close: float,
    entry: float,
    tp: float,
    sl: float,
    volume_profile: VolumeProfileLevels | None,
) -> list[float]:
    zoom_window = 34 if last_close < 0.02 else 40
    local = frame.tail(min(len(frame), zoom_window))
    local_lows = pd.to_numeric(local["low"], errors="coerce").dropna()
    local_highs = pd.to_numeric(local["high"], errors="coerce").dropna()
    if local_lows.empty or local_highs.empty:
        return _build_local_trade_focus_levels(
            frame,
            last_close=last_close,
            entry=entry,
            tp=tp,
            sl=sl,
            volume_profile=volume_profile,
        )

    tail_size = min(len(local_lows), 16)
    tail_lows = local_lows.tail(tail_size)
    tail_highs = local_highs.tail(tail_size)
    local_swing_low, local_swing_high = _clip_isolated_recent_wicks(
        tail_lows,
        tail_highs,
        last_close=last_close,
    )

    if {"open", "close"}.issubset(local.columns):
        body_lows = pd.concat(
            [
                pd.to_numeric(local["open"], errors="coerce"),
                pd.to_numeric(local["close"], errors="coerce"),
            ],
            axis=1,
        ).min(axis=1)
        body_highs = pd.concat(
            [
                pd.to_numeric(local["open"], errors="coerce"),
                pd.to_numeric(local["close"], errors="coerce"),
            ],
            axis=1,
        ).max(axis=1)
        local_body_low = float(body_lows.tail(tail_size).min())
        local_body_high = float(body_highs.tail(tail_size).max())
    else:
        local_body_low = local_swing_low
        local_body_high = local_swing_high

    levels = [
        local_swing_low,
        local_swing_high,
        local_body_low,
        local_body_high,
        float(local_lows.quantile(0.14)),
        float(local_highs.quantile(0.86)),
        last_close,
    ]

    for value in (entry, tp, sl):
        if value and value > 0:
            levels.append(float(value))

    if volume_profile is not None:
        for value in (volume_profile.val, volume_profile.vah, volume_profile.poc):
            _append_nearby_level(levels, value, last_close=last_close, max_distance_pct=0.082)

    return levels


def _build_detected_entry_pump_levels(
    frame: pd.DataFrame,
    *,
    last_close: float,
) -> list[float]:
    pump_window = _find_recent_entry_pump_window(frame, last_close=last_close)
    if pump_window is None:
        return []

    pump_start_rel, peak_rel, pump_pct = pump_window
    min_visible_pump = 0.022 if last_close < 0.02 else 0.016
    if pump_pct < min_visible_pump:
        return []

    pump_slice = frame.iloc[pump_start_rel : peak_rel + 1].copy()
    if pump_slice.empty:
        return []

    pump_lows = pd.to_numeric(pump_slice["low"], errors="coerce").dropna()
    pump_highs = pd.to_numeric(pump_slice["high"], errors="coerce").dropna()
    if pump_lows.empty or pump_highs.empty:
        return []

    return [
        float(pump_lows.min()),
        float(pump_highs.max()),
        float(pump_lows.quantile(0.12)),
        float(pump_highs.quantile(0.88)),
    ]


def _find_recent_entry_pump_window(frame: pd.DataFrame, *, last_close: float) -> tuple[int, int, float] | None:
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    if len(highs) < 18 or len(lows) < 18:
        return None

    scan_window = min(len(frame), 520 if last_close < 0.02 else 380)
    pump_lookback = 248 if last_close < 0.02 else 184
    scan_start = max(0, len(frame) - scan_window)
    best_candidate: tuple[float, int, int, float, int] | None = None
    recent_candidate: tuple[float, int, int, float, int] | None = None
    ultra_recent_candidate: tuple[float, int, int, float, int] | None = None

    for peak_rel in range(scan_start + 6, len(frame)):
        low_start = max(0, peak_rel - pump_lookback + 1)
        pump_start_rel = low_start + int(lows[low_start : peak_rel + 1].argmin())
        pump_low = float(lows[pump_start_rel])
        pump_high = float(highs[peak_rel])
        if pump_low <= 0 or pump_high <= pump_low:
            continue

        pump_pct = (pump_high - pump_low) / max(pump_low, 1e-8)
        pump_span = max(peak_rel - pump_start_rel + 1, 1)
        peak_distance = max(len(frame) - 1 - peak_rel, 0)
        capped_strength = min(pump_pct, 0.18 if last_close < 0.02 else 0.12)
        recency = 0.66 + 1.10 * ((peak_rel - scan_start) / max(scan_window - 1, 1))
        freshness = (
            1.94
            if peak_distance <= 8
            else 1.68
            if peak_distance <= 14
            else 1.34
            if peak_distance <= 24
            else 0.92
            if peak_distance <= 36
            else 0.68
        )
        retrace_pct = max(0.0, (pump_high - last_close) / max(pump_high, 1e-8))
        retrace_penalty = (
            1.20
            if retrace_pct <= 0.028
            else 1.05
            if retrace_pct <= 0.065
            else 0.86
            if retrace_pct <= 0.12
            else 0.60
        )
        active_bonus = (
            1.62
            if peak_distance <= 10 and retrace_pct <= 0.075
            else 1.34
            if peak_distance <= 18 and retrace_pct <= 0.11
            else 1.0
        )
        stale_penalty = 1.0
        if peak_distance > max(22, pump_span + 8) and retrace_pct > 0.020:
            stale_penalty = 0.74
        if peak_distance > max(34, pump_span * 2) and retrace_pct > 0.032:
            stale_penalty = 0.50

        score = (0.45 + capped_strength) * recency * freshness * retrace_penalty * active_bonus * stale_penalty

        candidate = (score, pump_start_rel, peak_rel, pump_pct, peak_distance)
        if best_candidate is None or score > best_candidate[0]:
            best_candidate = candidate
        else:
            best_peak_distance = int(best_candidate[4])
            if peak_distance < best_peak_distance and score >= best_candidate[0] * 0.80:
                best_candidate = candidate

        if peak_distance <= (32 if last_close < 0.02 else 28):
            if recent_candidate is None or score > recent_candidate[0]:
                recent_candidate = candidate
        if peak_distance <= (18 if last_close < 0.02 else 14):
            if ultra_recent_candidate is None or score > ultra_recent_candidate[0]:
                ultra_recent_candidate = candidate

    if best_candidate is None:
        return None
    if ultra_recent_candidate is not None and best_candidate is not None:
        ultra_recent_min_ratio = 0.32 if last_close < 0.02 else 0.38
        best_span = max(int(best_candidate[2]) - int(best_candidate[1]) + 1, 1)
        ultra_span = max(int(ultra_recent_candidate[2]) - int(ultra_recent_candidate[1]) + 1, 1)
        ultra_strength_ratio = float(ultra_recent_candidate[3]) / max(float(best_candidate[3]), 1e-8)
        ultra_span_ratio = float(ultra_span) / max(float(best_span), 1.0)
        if (
            ultra_recent_candidate[0] >= best_candidate[0] * ultra_recent_min_ratio
            and (ultra_strength_ratio >= 0.88 or ultra_span_ratio >= 0.74)
        ):
            best_candidate = ultra_recent_candidate
    if recent_candidate is not None and best_candidate is not None:
        recent_min_ratio = 0.38 if last_close < 0.02 else 0.44
        best_span = max(int(best_candidate[2]) - int(best_candidate[1]) + 1, 1)
        recent_span = max(int(recent_candidate[2]) - int(recent_candidate[1]) + 1, 1)
        recent_strength_ratio = float(recent_candidate[3]) / max(float(best_candidate[3]), 1e-8)
        recent_span_ratio = float(recent_span) / max(float(best_span), 1.0)
        if (
            recent_candidate[0] >= best_candidate[0] * recent_min_ratio
            and (recent_strength_ratio >= 0.82 or recent_span_ratio >= 0.66)
        ):
            best_candidate = recent_candidate

    if best_candidate is None:
        return None
    pump_start_rel, peak_rel, pump_pct = best_candidate[1], best_candidate[2], best_candidate[3]
    origin_window = 28 if last_close < 0.02 else 22
    if pump_start_rel > 0:
        origin_start = max(0, pump_start_rel - origin_window)
        origin_slice = lows[origin_start : pump_start_rel + 1]
        if len(origin_slice) > 0:
            origin_idx = origin_start + int(origin_slice.argmin())
            origin_low = float(lows[origin_idx])
            pump_low = float(lows[pump_start_rel])
            if origin_low <= pump_low * 1.018:
                pump_start_rel = origin_idx
    return pump_start_rel, peak_rel, pump_pct


def _extend_entry_pump_origin(frame: pd.DataFrame, *, pump_start_rel: int, peak_rel: int, last_close: float) -> int:
    if frame.empty or pump_start_rel <= 0 or peak_rel <= pump_start_rel:
        return pump_start_rel

    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    if len(lows) == 0 or len(highs) == 0 or len(closes) == 0:
        return pump_start_rel

    peak_price = float(highs[peak_rel])
    current_low = float(lows[pump_start_rel])
    if current_low <= 0 or peak_price <= current_low:
        return pump_start_rel

    pump_pct = (peak_price - current_low) / max(current_low, 1e-8)
    if pump_pct <= 0:
        return pump_start_rel

    max_back = 54 if last_close < 0.02 else 40
    near_low_tolerance = 0.026 if last_close < 0.02 else 0.018
    minimum_rebound = max(0.010, min(0.024, pump_pct * 0.42))
    origin = pump_start_rel

    for idx in range(pump_start_rel - 1, max(-1, pump_start_rel - max_back - 1), -1):
        candidate_low = float(lows[idx])
        if not pd.notna(candidate_low) or candidate_low <= 0:
            continue

        rebound = (peak_price - candidate_low) / max(candidate_low, 1e-8)
        if candidate_low <= current_low * (1.0 + near_low_tolerance) and rebound >= minimum_rebound:
            origin = idx
            current_low = min(current_low, candidate_low)
            continue

        local_high = float(highs[idx : pump_start_rel + 1].max())
        local_close = float(closes[idx])
        if rebound < minimum_rebound and idx < pump_start_rel - 3:
            break
        if local_high > candidate_low * (1.0 + near_low_tolerance * 1.3) and local_close > current_low * (1.0 + near_low_tolerance):
            break

    return origin


def _expand_entry_peak_cluster_origin(frame: pd.DataFrame, *, pump_start_rel: int, peak_rel: int, last_close: float) -> int:
    if frame.empty or peak_rel <= 0 or peak_rel >= len(frame):
        return pump_start_rel

    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    if len(highs) == 0 or len(lows) == 0:
        return pump_start_rel

    peak_price = float(highs[peak_rel])
    pump_low = float(lows[pump_start_rel])
    if peak_price <= pump_low or pump_low <= 0:
        return pump_start_rel

    cluster_tol_pct = 0.020 if last_close < 0.02 else 0.014
    retrace_limit = pump_low + (peak_price - pump_low) * (0.58 if last_close < 0.02 else 0.54)
    cluster_back = 96 if last_close < 0.02 else 72
    cluster_left = peak_rel

    for idx in range(peak_rel - 1, max(-1, peak_rel - cluster_back - 1), -1):
        candidate_high = float(highs[idx])
        if candidate_high < peak_price * (1.0 - cluster_tol_pct):
            continue
        valley = float(np.nanmin(lows[idx : cluster_left + 1]))
        if valley < retrace_limit:
            break
        cluster_left = idx

    if cluster_left >= peak_rel:
        return pump_start_rel

    origin_search_start = max(0, cluster_left - (68 if last_close < 0.02 else 52))
    origin_slice = lows[origin_search_start : cluster_left + 1]
    if len(origin_slice) == 0:
        return pump_start_rel
    cluster_origin = origin_search_start + int(origin_slice.argmin())
    return min(pump_start_rel, cluster_origin)


def _find_recent_focus_leg_origin(
    frame: pd.DataFrame,
    *,
    pump_start_rel: int,
    peak_rel: int,
    last_close: float,
) -> int:
    if frame.empty or peak_rel <= pump_start_rel + 4:
        return pump_start_rel

    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    if len(highs) == 0 or len(lows) == 0:
        return pump_start_rel

    peak_price = float(highs[peak_rel])
    pump_low = float(lows[pump_start_rel])
    if pump_low <= 0 or peak_price <= pump_low:
        return pump_start_rel

    pump_span = max(peak_rel - pump_start_rel + 1, 1)
    overall_pump_pct = (peak_price - pump_low) / max(pump_low, 1e-8)
    focus_scan = min(max(18 if last_close < 0.02 else 14, pump_span // 2), 54 if last_close < 0.02 else 42)
    recent_start = max(pump_start_rel, peak_rel - focus_scan)
    recent_slice = lows[recent_start : peak_rel + 1]
    if len(recent_slice) == 0:
        return pump_start_rel

    focus_origin = recent_start + int(recent_slice.argmin())
    focus_low = float(lows[focus_origin])
    if focus_low <= 0 or peak_price <= focus_low:
        return pump_start_rel

    focus_pump_pct = (peak_price - focus_low) / max(focus_low, 1e-8)
    terminal_scan = min(max(12 if last_close < 0.02 else 10, pump_span // 3), 24 if last_close < 0.02 else 18)
    terminal_start = max(pump_start_rel, peak_rel - terminal_scan)
    terminal_slice = lows[terminal_start : peak_rel + 1]
    if len(terminal_slice) > 0:
        terminal_origin = terminal_start + int(terminal_slice.argmin())
        terminal_low = float(lows[terminal_origin])
        terminal_pump_pct = (peak_price - terminal_low) / max(terminal_low, 1e-8)
        terminal_offset = terminal_origin - pump_start_rel
        if (
            terminal_low > 0
            and terminal_pump_pct >= (0.022 if last_close < 0.02 else 0.016)
            and terminal_pump_pct >= overall_pump_pct * (0.34 if last_close < 0.02 else 0.30)
            and terminal_offset >= max(8 if last_close < 0.02 else 6, pump_span // 6)
        ):
            return terminal_origin

    min_focus_pct = 0.032 if last_close < 0.02 else 0.022
    min_ratio = 0.54 if last_close < 0.02 else 0.50
    if focus_pump_pct >= min_focus_pct and focus_pump_pct >= overall_pump_pct * min_ratio:
        return focus_origin
    return pump_start_rel


def _find_terminal_blastoff_origin(
    frame: pd.DataFrame,
    *,
    pump_start_rel: int,
    peak_rel: int,
    last_close: float,
) -> int:
    if frame.empty or peak_rel <= pump_start_rel + 4:
        return pump_start_rel

    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    if len(highs) == 0 or len(lows) == 0:
        return pump_start_rel

    peak_price = float(highs[peak_rel])
    pump_low = float(lows[pump_start_rel])
    if pump_low <= 0 or peak_price <= pump_low:
        return pump_start_rel

    peak_distance = max(len(frame) - 1 - peak_rel, 0)
    if peak_distance > (16 if last_close < 0.02 else 12):
        return pump_start_rel

    overall_pump_pct = (peak_price - pump_low) / max(pump_low, 1e-8)
    micro_scan = min(
        max(8 if last_close < 0.02 else 6, (peak_rel - pump_start_rel) // 5),
        16 if last_close < 0.02 else 12,
    )
    micro_start = max(pump_start_rel, peak_rel - micro_scan)
    micro_slice = lows[micro_start : peak_rel + 1]
    if len(micro_slice) > 0:
        micro_origin = micro_start + int(micro_slice.argmin())
        micro_low = float(lows[micro_origin])
        if micro_low > 0 and peak_price > micro_low:
            micro_pump_pct = (peak_price - micro_low) / max(micro_low, 1e-8)
            if (
                micro_pump_pct >= (0.020 if last_close < 0.02 else 0.015)
                and micro_pump_pct >= overall_pump_pct * (0.62 if last_close < 0.02 else 0.56)
            ):
                return micro_origin

    terminal_scan = min(
        max(12 if last_close < 0.02 else 10, (peak_rel - pump_start_rel) // 3),
        28 if last_close < 0.02 else 22,
    )
    terminal_start = max(pump_start_rel, peak_rel - terminal_scan)
    terminal_slice = lows[terminal_start : peak_rel + 1]
    if len(terminal_slice) == 0:
        return pump_start_rel

    terminal_origin = terminal_start + int(terminal_slice.argmin())
    terminal_low = float(lows[terminal_origin])
    if terminal_low <= 0 or peak_price <= terminal_low:
        return pump_start_rel

    terminal_pump_pct = (peak_price - terminal_low) / max(terminal_low, 1e-8)
    min_terminal_pct = 0.024 if last_close < 0.02 else 0.018
    min_terminal_ratio = 0.54 if last_close < 0.02 else 0.48
    if terminal_pump_pct >= min_terminal_pct and terminal_pump_pct >= overall_pump_pct * min_terminal_ratio:
        return terminal_origin
    return pump_start_rel


def _select_entry_slice_start(
    *,
    source_len: int,
    pump_start_rel: int,
    desired_visible_bars: int,
    lead_bars: int,
) -> int:
    latest_start = max(0, source_len - desired_visible_bars)
    target_origin_frac = 0.38 if desired_visible_bars >= 360 else 0.34 if desired_visible_bars >= 300 else 0.30
    preferred_lead = max(lead_bars, int(desired_visible_bars * target_origin_frac))
    earliest_start = max(0, pump_start_rel - preferred_lead)
    if earliest_start >= latest_start:
        return latest_start
    extra_back = latest_start - earliest_start
    max_extra_back = max(220, min(lead_bars + 160, int(desired_visible_bars * 1.02)))
    return max(0, latest_start - min(extra_back, max_extra_back))


def _shift_slice_left_for_peak_visibility(
    *,
    source_len: int,
    current_start: int,
    peak_rel: int,
    last_close: float,
    max_visible_bars: int,
) -> int:
    visible_bars = max(source_len - current_start, 1)
    peak_frac = (peak_rel - current_start) / max(visible_bars, 1)
    target_peak_frac = 0.42 if last_close < 0.02 else 0.36
    if peak_frac >= target_peak_frac:
        return current_start

    extra_capacity = max(max_visible_bars - visible_bars, 0)
    shift_needed = int((target_peak_frac - peak_frac) * visible_bars) + 8
    shift_left = min(extra_capacity, current_start, max(0, shift_needed))
    if shift_left <= 0:
        return current_start
    return current_start - shift_left


def _shift_slice_right_for_recent_pump_focus(
    *,
    source_len: int,
    current_start: int,
    pump_start_rel: int,
    peak_rel: int,
    last_close: float,
    min_visible_bars: int,
) -> int:
    visible_bars = max(source_len - current_start, 1)
    if visible_bars <= min_visible_bars + 8:
        return current_start

    pump_span = max(peak_rel - pump_start_rel + 1, 1)
    peak_distance = max(source_len - 1 - peak_rel, 0)
    ultra_compact_recent = peak_distance <= (12 if last_close < 0.02 else 10) and pump_span <= (
        24 if last_close < 0.02 else 18
    )
    compact_recent = peak_distance <= (18 if last_close < 0.02 else 14) and pump_span <= (
        58 if last_close < 0.02 else 46
    )
    if not compact_recent:
        return current_start

    pump_origin_frac = (pump_start_rel - current_start) / max(visible_bars, 1)
    if ultra_compact_recent:
        if last_close < 0.02:
            target_origin_frac = 0.38
            max_origin_frac = 0.45
        else:
            target_origin_frac = 0.35
            max_origin_frac = 0.42
    else:
        if last_close < 0.02:
            target_origin_frac = 0.50
            max_origin_frac = 0.58
        else:
            target_origin_frac = 0.47
            max_origin_frac = 0.54
    if pump_origin_frac <= max_origin_frac:
        return current_start

    ideal_start = int((pump_start_rel - source_len * target_origin_frac) / max(1.0 - target_origin_frac, 1e-8))
    ideal_start = max(0, ideal_start)
    max_right_shift = max(0, source_len - min_visible_bars)
    if ultra_compact_recent:
        minimum_lead = max(10 if last_close < 0.02 else 8, int(pump_span * 0.24))
    else:
        minimum_lead = max(16 if last_close < 0.02 else 12, int(pump_span * 0.38))
    ideal_start = min(ideal_start, pump_start_rel - minimum_lead, max_right_shift)
    return max(current_start, ideal_start)


def _trim_stale_left_tail_for_compact_recent_pump(
    *,
    source_len: int,
    current_start: int,
    slice_anchor_rel: int,
    original_pump_start_rel: int,
    peak_rel: int,
    last_close: float,
    min_visible_bars: int,
    max_visible_bars: int,
) -> int:
    visible_bars = max(source_len - current_start, 1)
    pump_span = max(peak_rel - slice_anchor_rel + 1, 1)
    peak_distance = max(source_len - 1 - peak_rel, 0)
    ultra_compact_recent = peak_distance <= (12 if last_close < 0.02 else 10) and pump_span <= (
        24 if last_close < 0.02 else 18
    )
    compact_recent = peak_distance <= (18 if last_close < 0.02 else 14) and pump_span <= (
        54 if last_close < 0.02 else 42
    )
    if not compact_recent:
        return current_start

    anchor_frac = (slice_anchor_rel - current_start) / max(visible_bars, 1)
    if ultra_compact_recent:
        if last_close < 0.02:
            hard_max_anchor_frac = 0.52
            target_anchor_frac = 0.38
            compact_buffer = 16
            minimum_lead = max(8, int(pump_span * 0.18))
        else:
            hard_max_anchor_frac = 0.48
            target_anchor_frac = 0.34
            compact_buffer = 14
            minimum_lead = max(7, int(pump_span * 0.18))
    else:
        if last_close < 0.02:
            hard_max_anchor_frac = 0.66
            target_anchor_frac = 0.52
            compact_buffer = 30
            minimum_lead = max(12, int(pump_span * 0.30))
        else:
            hard_max_anchor_frac = 0.62
            target_anchor_frac = 0.48
            compact_buffer = 24
            minimum_lead = max(10, int(pump_span * 0.28))
    if anchor_frac <= hard_max_anchor_frac:
        return current_start

    compact_context_back = max(
        8 if last_close < 0.02 else 6,
        int(max(peak_rel - original_pump_start_rel + 1, 1) * (0.10 if ultra_compact_recent else 0.16)),
    )
    desired_anchor_rel = max(slice_anchor_rel, original_pump_start_rel - compact_context_back)
    min_visible_target = max(
        min_visible_bars,
        min(
            max_visible_bars,
            pump_span + compact_buffer,
        ),
    )
    max_right_shift = max(0, source_len - min_visible_target)
    desired_start = int((desired_anchor_rel - source_len * target_anchor_frac) / max(1.0 - target_anchor_frac, 1e-8))
    desired_start = max(0, min(desired_start, max_right_shift))

    desired_start = min(desired_start, max(0, desired_anchor_rel - minimum_lead))
    return max(current_start, desired_start)


def _bound_post_peak_context(*, pump_span: int, post_peak_bars: int, last_close: float) -> int:
    base_cap = 48 if last_close < 0.02 else 42
    adaptive_cap = min(base_cap, max(26, pump_span + 8))
    return min(post_peak_bars, adaptive_cap)


def _slice_entry_chart_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or len(frame) <= 64:
        return frame.copy()

    close_series = pd.to_numeric(frame["close"], errors="coerce").dropna()
    high_series = pd.to_numeric(frame["high"], errors="coerce")
    low_series = pd.to_numeric(frame["low"], errors="coerce")
    if close_series.empty or high_series.isna().any() or low_series.isna().any():
        return frame.tail(min(len(frame), 124)).copy()

    last_close = float(close_series.iloc[-1])
    source_window = min(len(frame), 3680 if last_close < 0.02 else 2480)
    source = frame.tail(source_window).copy()
    highs = pd.to_numeric(source["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(source["low"], errors="coerce").to_numpy(dtype=float)
    if len(highs) == 0 or len(lows) == 0:
        return frame.tail(min(len(frame), 168)).copy()

    pump_window = _find_recent_entry_pump_window(source, last_close=last_close)
    if pump_window is None:
        return frame.tail(min(len(frame), 388 if last_close < 0.02 else 308)).copy()
    pump_start_rel, peak_rel, pump_pct = pump_window
    original_pump_start_rel = pump_start_rel
    pump_start_rel = _expand_entry_peak_cluster_origin(
        source,
        pump_start_rel=pump_start_rel,
        peak_rel=peak_rel,
        last_close=last_close,
    )
    pump_start_rel = _extend_entry_pump_origin(
        source,
        pump_start_rel=pump_start_rel,
        peak_rel=peak_rel,
        last_close=last_close,
    )
    original_span = max(peak_rel - original_pump_start_rel + 1, 1)
    max_extra_back = max(34 if last_close < 0.02 else 26, int(original_span * 0.90))
    earliest_reasonable_origin = max(0, original_pump_start_rel - max_extra_back)
    pump_start_rel = max(pump_start_rel, earliest_reasonable_origin)

    minimum_visible_pump = 0.025 if last_close < 0.02 else 0.018
    if pump_pct < minimum_visible_pump:
        fallback_bars = 204 if last_close < 0.02 else 184
        return source.tail(min(len(source), fallback_bars)).copy()

    pump_span = max(peak_rel - pump_start_rel + 1, 1)
    post_peak_bars = max(len(source) - 1 - peak_rel, 0)
    focus_origin_rel = _find_recent_focus_leg_origin(
        source,
        pump_start_rel=pump_start_rel,
        peak_rel=peak_rel,
        last_close=last_close,
    )
    blastoff_origin_rel = _find_terminal_blastoff_origin(
        source,
        pump_start_rel=pump_start_rel,
        peak_rel=peak_rel,
        last_close=last_close,
    )
    focus_pump_span = max(peak_rel - focus_origin_rel + 1, 1)
    compact_recent_pump = post_peak_bars <= (18 if last_close < 0.02 else 14) and min(
        original_span, focus_pump_span
    ) <= (
        58 if last_close < 0.02 else 46
    )
    ultra_compact_recent_pump = compact_recent_pump and post_peak_bars <= (
        12 if last_close < 0.02 else 10
    ) and min(original_span, max(peak_rel - blastoff_origin_rel + 1, 1)) <= (
        24 if last_close < 0.02 else 18
    )
    if compact_recent_pump:
        compact_context_back = max(12 if last_close < 0.02 else 10, int(original_span * 0.16))
        compact_floor = max(0, original_pump_start_rel - compact_context_back)
        slice_anchor_rel = max(focus_origin_rel, blastoff_origin_rel, compact_floor)
    else:
        slice_anchor_rel = pump_start_rel
    display_pump_span = focus_pump_span if compact_recent_pump else pump_span
    terminal_blastoff_mode = ultra_compact_recent_pump and max(peak_rel - blastoff_origin_rel + 1, 1) <= (
        16 if last_close < 0.02 else 12
    )
    if ultra_compact_recent_pump:
        if terminal_blastoff_mode:
            lead_bars = (10 if last_close < 0.02 else 8) + min(display_pump_span, 6)
            min_visible_bars = 34 if last_close < 0.02 else 30
            max_visible_bars = 80 if last_close < 0.02 else 66
            trailing_context = 5 if last_close < 0.02 else 4
        else:
            lead_bars = (16 if last_close < 0.02 else 12) + min(display_pump_span, 10)
            min_visible_bars = 50 if last_close < 0.02 else 44
            max_visible_bars = 120 if last_close < 0.02 else 96
            trailing_context = 12 if last_close < 0.02 else 10
    elif compact_recent_pump:
        lead_bars = (24 if last_close < 0.02 else 20) + min(display_pump_span, 18)
        min_visible_bars = 72 if last_close < 0.02 else 64
        max_visible_bars = 176 if last_close < 0.02 else 152
        trailing_context = 22 if last_close < 0.02 else 18
    else:
        lead_bars = (112 if last_close < 0.02 else 84) + min(display_pump_span // 2, 96)
        min_visible_bars = 188 if last_close < 0.02 else 148
        max_visible_bars = 720 if last_close < 0.02 else 560
        trailing_context = 168
    bounded_post_peak_bars = _bound_post_peak_context(
        pump_span=display_pump_span,
        post_peak_bars=post_peak_bars,
        last_close=last_close,
    )
    desired_visible_bars = max(
        min_visible_bars,
        min(max_visible_bars, display_pump_span + bounded_post_peak_bars + lead_bars + trailing_context),
    )

    visible_start_rel = _select_entry_slice_start(
        source_len=len(source),
        pump_start_rel=slice_anchor_rel,
        desired_visible_bars=desired_visible_bars,
        lead_bars=lead_bars,
    )

    visible_bars = len(source) - visible_start_rel
    if visible_bars < min_visible_bars:
        visible_start_rel = max(0, len(source) - min_visible_bars)
    elif visible_bars > max_visible_bars:
        visible_start_rel = max(0, len(source) - max_visible_bars)

    visible_bars = len(source) - visible_start_rel
    if visible_bars > 0:
        pump_origin_frac = (slice_anchor_rel - visible_start_rel) / max(visible_bars, 1)
        if terminal_blastoff_mode:
            target_origin_frac = 0.42 if last_close < 0.02 else 0.38
        elif compact_recent_pump:
            target_origin_frac = 0.34 if last_close < 0.02 else 0.30
        else:
            target_origin_frac = 0.31 if last_close < 0.02 else 0.27
        if pump_origin_frac < target_origin_frac:
            extra_capacity = max(max_visible_bars - visible_bars, 0)
            shift_needed = int((target_origin_frac - pump_origin_frac) * visible_bars) + 8
            shift_left = min(extra_capacity, visible_start_rel, max(0, shift_needed))
            if shift_left > 0:
                visible_start_rel -= shift_left

    visible_start_rel = _shift_slice_left_for_peak_visibility(
        source_len=len(source),
        current_start=visible_start_rel,
        peak_rel=peak_rel,
        last_close=last_close,
        max_visible_bars=max_visible_bars,
    )
    visible_start_rel = _shift_slice_right_for_recent_pump_focus(
        source_len=len(source),
        current_start=visible_start_rel,
        pump_start_rel=slice_anchor_rel,
        peak_rel=peak_rel,
        last_close=last_close,
        min_visible_bars=min_visible_bars,
    )
    visible_start_rel = _trim_stale_left_tail_for_compact_recent_pump(
        source_len=len(source),
        current_start=visible_start_rel,
        slice_anchor_rel=slice_anchor_rel,
        original_pump_start_rel=original_pump_start_rel,
        peak_rel=peak_rel,
        last_close=last_close,
        min_visible_bars=min_visible_bars,
        max_visible_bars=max_visible_bars,
    )

    if terminal_blastoff_mode:
        minimum_lead = 12 if last_close < 0.02 else 10
    else:
        minimum_lead = 18 if last_close < 0.02 else 14
    lead_anchor_rel = slice_anchor_rel if compact_recent_pump else pump_start_rel
    if lead_anchor_rel - visible_start_rel < minimum_lead:
        visible_start_rel = max(0, lead_anchor_rel - minimum_lead)
        if len(source) - visible_start_rel > max_visible_bars:
            visible_start_rel = max(0, len(source) - max_visible_bars)

    return source.iloc[visible_start_rel:].copy()


def _expand_focus_to_visible_slice(
    frame: pd.DataFrame,
    *,
    last_close: float,
    focus_min: float,
    focus_max: float,
) -> tuple[float, float]:
    lows = pd.to_numeric(frame["low"], errors="coerce").dropna()
    highs = pd.to_numeric(frame["high"], errors="coerce").dropna()
    if lows.empty or highs.empty:
        return focus_min, focus_max

    visible_low = float(lows.quantile(0.035 if last_close < 0.02 else 0.045))
    visible_high = float(highs.quantile(0.985 if last_close < 0.02 else 0.975))

    if {"open", "close"}.issubset(frame.columns):
        body_lows = pd.concat(
            [
                pd.to_numeric(frame["open"], errors="coerce"),
                pd.to_numeric(frame["close"], errors="coerce"),
            ],
            axis=1,
        ).min(axis=1)
        body_highs = pd.concat(
            [
                pd.to_numeric(frame["open"], errors="coerce"),
                pd.to_numeric(frame["close"], errors="coerce"),
            ],
            axis=1,
        ).max(axis=1)
        visible_low = min(visible_low, float(body_lows.quantile(0.03 if last_close < 0.02 else 0.04)))
        visible_high = max(visible_high, float(body_highs.quantile(0.97 if last_close < 0.02 else 0.96)))

    current_span = max(focus_max - focus_min, _price_scale(last_close) * 0.008)
    visible_span = max(visible_high - visible_low, _price_scale(last_close) * 0.008)
    upper_allowance = max(current_span * 1.10, visible_span * 0.78)
    lower_allowance = max(current_span * 1.10, visible_span * 0.78)

    if visible_high > focus_max and (visible_high - focus_max) <= upper_allowance:
        focus_max = visible_high
    if visible_low < focus_min and (focus_min - visible_low) <= lower_allowance:
        focus_min = visible_low

    return focus_min, focus_max


def _compute_price_view_bounds(
    frame: pd.DataFrame,
    *,
    volume_profile: VolumeProfileLevels | None,
    entry: float,
    tp: float,
    sl: float,
    show_trade_levels: bool,
    show_entry_levels: bool,
    liquidation_map: LiquidationMap | None,
    show_liquidation_map: bool,
) -> tuple[float, float]:
    if frame.empty:
        return 0.0, 1.0

    last_close = float(pd.to_numeric(frame["close"], errors="coerce").dropna().iloc[-1])
    recent_bars = min(len(frame), 64 if show_liquidation_map else (56 if last_close < 0.02 else 66))
    recent = frame.tail(max(24, recent_bars))
    lows = pd.to_numeric(recent["low"], errors="coerce").dropna()
    highs = pd.to_numeric(recent["high"], errors="coerce").dropna()

    if lows.empty or highs.empty:
        low = float(pd.to_numeric(frame["low"], errors="coerce").dropna().min())
        high = float(pd.to_numeric(frame["high"], errors="coerce").dropna().max())
        pad = max((high - low) * 0.10, _price_scale(last_close) * 0.0045)
        return low - pad, high + pad

    tail_size = min(len(lows), 18)
    tail_lows = lows.tail(tail_size)
    tail_highs = highs.tail(tail_size)
    recent_swing_low, recent_swing_high = _clip_isolated_recent_wicks(
        tail_lows,
        tail_highs,
        last_close=last_close,
    )
    robust_low = float(lows.quantile(0.10))
    robust_high = float(highs.quantile(0.90))

    if {"open", "close"}.issubset(recent.columns):
        body_lows = pd.concat(
            [
                pd.to_numeric(recent["open"], errors="coerce"),
                pd.to_numeric(recent["close"], errors="coerce"),
            ],
            axis=1,
        ).min(axis=1)
        body_highs = pd.concat(
            [
                pd.to_numeric(recent["open"], errors="coerce"),
                pd.to_numeric(recent["close"], errors="coerce"),
            ],
            axis=1,
        ).max(axis=1)
        recent_body_low = float(body_lows.tail(tail_size).min())
        recent_body_high = float(body_highs.tail(tail_size).max())
    else:
        recent_body_low = recent_swing_low
        recent_body_high = recent_swing_high

    focus_levels: list[float] = [
        recent_swing_low,
        recent_swing_high,
        recent_body_low,
        recent_body_high,
        robust_low,
        robust_high,
        last_close,
    ]

    for col in ("ema20", "ema50", "vwap"):
        if col in recent.columns:
            series = pd.to_numeric(recent[col], errors="coerce").dropna()
            if not series.empty:
                focus_levels.append(float(series.iloc[-1]))

    vp_distance_limit_pct = 0.28 if show_liquidation_map else 0.086
    if volume_profile is not None and show_trade_levels:
        for value in (volume_profile.val, volume_profile.vah, volume_profile.poc):
            _append_nearby_level(
                focus_levels,
                value,
                last_close=last_close,
                max_distance_pct=vp_distance_limit_pct,
            )

    if show_entry_levels:
        for value in (entry, tp, sl):
            if value and value > 0:
                focus_levels.append(float(value))

    if show_liquidation_map and liquidation_map is not None and liquidation_map.bands:
        for band in liquidation_map.bands:
            level = float(band.level)
            if level <= 0:
                continue
            distance_pct = abs(level - last_close) / max(abs(last_close), 1e-8)
            if band.closed_index is None and distance_pct <= 0.16:
                focus_levels.append(level)
            elif band.closed_index is not None and distance_pct <= 0.10:
                focus_levels.append(level)

    focus_min = min(focus_levels)
    focus_max = max(focus_levels)
    focus_span = max(focus_max - focus_min, _price_scale(last_close) * 0.01)

    if show_entry_levels and not show_liquidation_map:
        pump_focus_min = None
        pump_focus_max = None
        local_levels = _build_local_trade_focus_levels(
            frame,
            last_close=last_close,
            entry=entry,
            tp=tp,
            sl=sl,
            volume_profile=volume_profile if show_trade_levels else None,
        )
        local_focus_min = min(local_levels)
        local_focus_max = max(local_levels)
        local_focus_span = max(local_focus_max - local_focus_min, _price_scale(last_close) * 0.008)
        if focus_span > local_focus_span * 1.55:
            focus_min = local_focus_min
            focus_max = local_focus_max
            focus_span = local_focus_span

        pump_levels = _build_detected_entry_pump_levels(frame, last_close=last_close)
        if pump_levels:
            pump_focus_min = min(pump_levels)
            pump_focus_max = max(pump_levels)
            focus_min = min(focus_min, min(pump_levels))
            focus_max = max(focus_max, max(pump_levels))
            focus_span = max(focus_max - focus_min, _price_scale(last_close) * 0.008)

        zoom_levels = _build_local_recent_zoom_levels(
            frame,
            last_close=last_close,
            entry=entry,
            tp=tp,
            sl=sl,
            volume_profile=volume_profile if show_trade_levels else None,
        )
        zoom_focus_min = min(zoom_levels)
        zoom_focus_max = max(zoom_levels)
        zoom_focus_span = max(zoom_focus_max - zoom_focus_min, _price_scale(last_close) * 0.006)
        zoom_ratio = 1.10 if last_close < 0.01 else 1.18 if last_close < 0.05 else 1.28
        if focus_span > zoom_focus_span * zoom_ratio:
            if pump_focus_min is not None and pump_focus_max is not None:
                pump_span = max(pump_focus_max - pump_focus_min, _price_scale(last_close) * 0.006)
                preserve_pump = pump_span >= zoom_focus_span * (0.18 if last_close < 0.02 else 0.15)
                if preserve_pump:
                    zoom_focus_min = min(zoom_focus_min, pump_focus_min)
                    zoom_focus_max = max(zoom_focus_max, pump_focus_max)
                    zoom_focus_span = max(zoom_focus_max - zoom_focus_min, _price_scale(last_close) * 0.006)
            focus_min = zoom_focus_min
            focus_max = zoom_focus_max
            focus_span = zoom_focus_span

        focus_min, focus_max = _expand_focus_to_visible_slice(
            frame,
            last_close=last_close,
            focus_min=focus_min,
            focus_max=focus_max,
        )
        focus_span = max(focus_max - focus_min, _price_scale(last_close) * 0.006)
    visible_tail = frame.tail(min(len(frame), 96 if last_close < 0.02 else 108))
    visible_tail_highs = pd.to_numeric(visible_tail["high"], errors="coerce").dropna()
    visible_tail_lows = pd.to_numeric(visible_tail["low"], errors="coerce").dropna()
    if not visible_tail_highs.empty and not visible_tail_lows.empty:
        _, clipped_visible_high = _clip_isolated_recent_wicks(
            visible_tail_lows.tail(min(len(visible_tail_lows), 22)),
            visible_tail_highs.tail(min(len(visible_tail_highs), 22)),
            last_close=last_close,
        )
        visible_high = max(
            clipped_visible_high,
            float(visible_tail_highs.quantile(0.94 if last_close < 0.02 else 0.92)),
        )
        headroom_limit = max(focus_span * 0.78, _price_scale(last_close) * 0.011)
        if visible_high > focus_max and (visible_high - focus_max) <= headroom_limit:
            focus_max = visible_high
            focus_span = max(focus_max - focus_min, _price_scale(last_close) * 0.006)

    atr_last = 0.0
    if "atr" in frame.columns:
        atr_series = pd.to_numeric(frame["atr"], errors="coerce").dropna()
        if not atr_series.empty:
            atr_last = float(atr_series.iloc[-1])
    if atr_last <= 0:
        atr_last = focus_span * 0.12

    price_pad = max(focus_span * 0.10, atr_last * 1.04, _price_scale(last_close) * 0.0034)
    if show_entry_levels and not show_liquidation_map:
        lower_pad = max(price_pad * 0.94, focus_span * 0.072, _price_scale(last_close) * 0.0033)
        compact_entry_frame = len(frame) <= (86 if last_close < 0.02 else 74)
        pump_is_top_of_frame = (
            not visible_tail_highs.empty
            and visible_high >= focus_max - max(focus_span * 0.10, _price_scale(last_close) * 0.0032)
        )
        if pump_is_top_of_frame:
            if compact_entry_frame:
                upper_pad = max(price_pad * 1.55, focus_span * 0.24, _price_scale(last_close) * 0.0052)
            else:
                upper_pad = max(price_pad * 2.05, focus_span * 0.40, _price_scale(last_close) * 0.0066)
        else:
            if compact_entry_frame:
                upper_pad = max(price_pad * 1.18, focus_span * 0.085, _price_scale(last_close) * 0.0035)
            else:
                upper_pad = max(price_pad * 1.48, focus_span * 0.14, _price_scale(last_close) * 0.0044)
    else:
        lower_pad = price_pad
        upper_pad = price_pad
    y_min = max(0.0, focus_min - lower_pad)
    y_max = focus_max + upper_pad

    if y_max <= y_min:
        y_max = y_min + max(_price_scale(last_close) * 0.02, 1e-6)

    return y_min, y_max


def _configure_time_axis(ax, *, timeframe_label: str, frame: pd.DataFrame):
    if frame.empty or len(frame.index) < 2:
        return

    span_minutes = max((frame.index[-1] - frame.index[0]).total_seconds() / 60.0, 1.0)
    tf = str(timeframe_label or "").lower()
    tz = _resolve_chart_timezone()

    if "h" in tf:
        if span_minutes <= 24 * 60 * 3:
            locator = mdates.DayLocator(interval=1, tz=tz)
        elif span_minutes <= 24 * 60 * 8:
            locator = mdates.DayLocator(interval=2, tz=tz)
        else:
            locator = mdates.DayLocator(interval=3, tz=tz)
        formatter = mdates.DateFormatter("%m-%d", tz=tz)
    else:
        if span_minutes <= 105:
            locator = mdates.MinuteLocator(interval=15, tz=tz)
        elif span_minutes <= 240:
            locator = mdates.MinuteLocator(interval=30, tz=tz)
        elif span_minutes <= 540:
            locator = mdates.HourLocator(interval=1, tz=tz)
        else:
            locator = mdates.HourLocator(interval=2, tz=tz)
        formatter = mdates.DateFormatter("%H:%M", tz=tz)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _draw_liquidation_heatmap(ax, x_values, liq_map: LiquidationMap | None, frame: pd.DataFrame):
    if liq_map is None or not liq_map.bands:
        return

    price_span = max(float(frame["high"].max()) - float(frame["low"].min()), 1e-8)
    last_close = max(float(frame["close"].iloc[-1]), 1e-8)
    atr_series = pd.to_numeric(frame.get("atr"), errors="coerce") if "atr" in frame.columns else None
    atr_last = float(atr_series.dropna().iloc[-1]) if atr_series is not None and not atr_series.dropna().empty else 0.0
    band_half_height = max(price_span * 0.0054, atr_last * 0.24, last_close * 0.0013)
    candle_width = max((x_values[1] - x_values[0]) * 0.60, 1e-6) if len(x_values) >= 2 else 0.00045
    half_step = candle_width / 1.20

    def _band_render_score(item):
        duration = max(int(item.end_index) - int(item.start_index) + 1, 1)
        active_bonus = 1.30 if item.closed_index is None else 0.88
        source_bonus = 1.85 if getattr(item, "source", "synthetic") == "feed" else 1.0
        recency_bonus = 1.0
        if item.closed_index is not None:
            recency_bonus = 1.15 if int(item.closed_index) >= max(len(frame) - 20, 0) else 0.80
        return item.weight * active_bonus * source_bonus * recency_bonus * min(1.0 + duration / 22.0, 2.4)

    def _is_active_on_price_side(band, side: str) -> bool:
        if band.closed_index is not None:
            return False
        if side == "above":
            return band.level > last_close
        return band.level < last_close

    def _pick_visible_for_side(side: str):
        def _distance_ok(band, *, max_dist: float) -> bool:
            return abs((band.level - last_close) / max(last_close, 1e-8)) <= max_dist

        open_feed = sorted(
            (
                band
                for band in liq_map.bands
                if band.side == side
                and _is_active_on_price_side(band, side)
                and getattr(band, "source", "synthetic") == "feed"
                and _distance_ok(band, max_dist=0.22)
            ),
            key=_band_render_score,
            reverse=True,
        )[:1]
        open_synthetic = sorted(
            (
                band
                for band in liq_map.bands
                if band.side == side
                and _is_active_on_price_side(band, side)
                and getattr(band, "source", "synthetic") != "feed"
                and _distance_ok(band, max_dist=0.15)
            ),
            key=_band_render_score,
            reverse=True,
        )[:1]
        recent_closed_feed = sorted(
            (
                band
                for band in liq_map.bands
                if band.side == side
                and band.closed_index is not None
                and getattr(band, "source", "synthetic") == "feed"
                and int(band.closed_index) >= max(len(frame) - 18, 0)
                and max(int(band.end_index) - int(band.start_index) + 1, 1) >= 8
                and abs((band.level - last_close) / max(last_close, 1e-8)) <= 0.18
            ),
            key=_band_render_score,
            reverse=True,
        )[:1]
        selected_open = open_feed or open_synthetic
        if selected_open and recent_closed_feed:
            nearest_open = selected_open[0]
            if abs((recent_closed_feed[0].level - nearest_open.level) / max(last_close, 1e-8)) <= 0.03:
                recent_closed_feed = []
        return selected_open + recent_closed_feed

    visible_bands = sorted(
        _pick_visible_for_side("above") + _pick_visible_for_side("below"),
        key=_band_render_score,
    )

    for band in visible_bands:
        start_idx = max(0, min(int(band.start_index), len(x_values) - 1))
        end_idx = max(0, min(int(band.end_index), len(x_values) - 1))
        seg_start = float(x_values[start_idx]) - half_step
        seg_end = float(x_values[end_idx]) + half_step
        band_width = max(seg_end - seg_start, 1e-8)
        is_closed = band.closed_index is not None and band.closed_index < len(frame) - 1
        duration = max(end_idx - start_idx + 1, 1)
        duration_boost = min(0.22 + duration / max(len(frame), 1), 0.95)
        base_alpha = 0.06 + 0.08 * band.intensity + 0.07 * duration_boost
        hot_alpha = 0.12 + 0.18 * band.intensity + 0.06 * duration_boost
        if getattr(band, "source", "synthetic") == "feed":
            base_alpha *= 1.16
            hot_alpha *= 1.18
        else:
            base_alpha *= 0.72
            hot_alpha *= 0.70
        if is_closed:
            base_alpha *= 0.78
            hot_alpha *= 0.84
        hot_width = max(band_width * (0.28 + 0.24 * band.intensity), candle_width * 2.8)
        hot_start = seg_end - hot_width
        hot_color = "#ffe3a1" if band.side == "above" else "#ebb0ff"

        ax.add_patch(
            Rectangle(
                (seg_start, band.level - band_half_height),
                band_width,
                band_half_height * 2.0,
                facecolor="#4b1f76",
                edgecolor="none",
                alpha=base_alpha,
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (hot_start, band.level - band_half_height * 0.74),
                hot_width,
                band_half_height * 1.48,
                facecolor=hot_color,
                edgecolor="none",
                alpha=hot_alpha,
                zorder=2,
            )
        )

        if is_closed:
            ax.vlines(
                seg_end,
                band.level - band_half_height * 1.05,
                band.level + band_half_height * 1.05,
                color=hot_color,
                linewidth=1.6,
                alpha=0.58,
                zorder=3,
            )
            ax.add_patch(
                Rectangle(
                    (seg_end, band.level - band_half_height * 1.12),
                    candle_width * 0.72,
                    band_half_height * 2.24,
                    facecolor="#0b1320",
                    edgecolor="none",
                    alpha=0.82,
                    zorder=2.8,
                )
            )

    def _label_score(item):
        duration = max(int(item.end_index) - int(item.start_index) + 1, 1)
        active_bonus = 1.25 if item.closed_index is None else 0.85
        return item.weight * active_bonus * min(1.0 + duration / 22.0, 2.25)

    strongest_above = max(
        (band for band in visible_bands if band.side == "above" and _is_active_on_price_side(band, "above")),
        key=_label_score,
        default=None,
    )
    strongest_below = max(
        (band for band in visible_bands if band.side == "below" and _is_active_on_price_side(band, "below")),
        key=_label_score,
        default=None,
    )
    for band, label, color in (
        (strongest_above, "LQ UP", "#ffe0ac"),
        (strongest_below, "LQ DN", "#efc0ff"),
    ):
        if band is None:
            continue
        duration = max(int(band.end_index) - int(band.start_index) + 1, 1)
        if duration < 4 and band.weight < 1.9:
            continue
        seg_start, seg_end = _segment_bounds(x_values, start_index=band.start_index, end_index=band.end_index)
        ax.text(
            _liquidation_label_x(ax, seg_start, seg_end),
            band.level,
            label,
            ha="right",
            va="center",
            fontsize=6.6,
            color=color,
            alpha=0.74,
            bbox={"facecolor": "#120f1c", "edgecolor": "none", "alpha": 0.36, "pad": 0.7},
            zorder=5,
        )


def build_signal_chart(
    symbol: str,
    df: pd.DataFrame,
    side: str,
    entry: float,
    tp: float,
    sl: float,
    volume_profile: VolumeProfileLevels | None = None,
    timeframe_label: str = "1m",
    show_trade_levels: bool = True,
    show_entry_levels: bool = True,
    liquidation_map: LiquidationMap | None = None,
    show_liquidation_map: bool = True,
) -> bytes | None:
    if df.empty or len(df) < 20:
        return None

    raw_last_close = float(pd.to_numeric(df["close"], errors="coerce").dropna().iloc[-1])
    source_frame = df.tail(180 if show_liquidation_map else (2520 if raw_last_close < 0.02 else 1840)).copy()
    source_frame = _compute_macd(source_frame)
    source_frame = source_frame.dropna(subset=["open", "high", "low", "close"])
    if source_frame.empty:
        return None
    frame = source_frame if show_liquidation_map else _slice_entry_chart_frame(source_frame)
    if frame.empty:
        return None

    x_values = mdates.date2num(frame.index.to_pydatetime())
    fig, (ax_price, ax_macd) = plt.subplots(
        2,
        1,
        figsize=(16.2, 11.8),
        sharex=True,
        facecolor="#0b1320",
        gridspec_kw={"height_ratios": [7.1, 1.25]},
    )
    _style_axis(ax_price)
    _style_axis(ax_macd, is_macd=True)

    _draw_candles(ax_price, frame, x_values)
    _draw_recent_focus(ax_price, x_values, bars=24 if show_liquidation_map else 20)
    if not show_liquidation_map:
        _draw_entry_pump_focus(ax_price, frame, x_values)

    if show_liquidation_map:
        liquidation_map = liquidation_map or build_liquidation_map(frame)

    if "ema20" in frame.columns:
        ax_price.plot(x_values, frame["ema20"], color="#5ea8ff", linewidth=1.08, alpha=0.66, zorder=2.2)
    if "ema50" in frame.columns:
        ax_price.plot(x_values, frame["ema50"], color="#d7a35a", linewidth=1.08, alpha=0.58, zorder=2.1)
    if "vwap" in frame.columns:
        ax_price.plot(x_values, frame["vwap"], color="#77cfa2", linewidth=1.10, alpha=0.62, zorder=2.15)

    y_min, y_max = _compute_price_view_bounds(
        frame,
        volume_profile=volume_profile,
        entry=entry,
        tp=tp,
        sl=sl,
        show_trade_levels=show_trade_levels,
        show_entry_levels=show_entry_levels,
        liquidation_map=liquidation_map,
        show_liquidation_map=show_liquidation_map,
    )
    ax_price.set_ylim(y_min, y_max)

    x_left = float(x_values[0])
    x_right = float(x_values[-1])
    if show_liquidation_map:
        x_pad_frac = 0.075
    elif len(frame) <= (92 if raw_last_close < 0.02 else 80):
        x_pad_frac = 0.052
    else:
        x_pad_frac = 0.072
    x_pad = max((x_right - x_left) * x_pad_frac, 0.00032 if not show_liquidation_map else 0.00045)
    ax_price.set_xlim(x_left, x_right + x_pad)

    if volume_profile is not None and show_trade_levels:
        ax_price.axhspan(volume_profile.val, volume_profile.vah, color="#1d3554", alpha=0.16, zorder=0)

    if show_liquidation_map:
        _draw_liquidation_heatmap(ax_price, x_values, liquidation_map, frame)

    if show_trade_levels:
        if volume_profile is not None:
            if show_liquidation_map:
                level_specs = (
                    ("VAH", volume_profile.vah, "#f0b35f", "--", 0.95, 20, 0.74, 0.12),
                    ("VAL", volume_profile.val, "#8fd36a", "--", 0.95, 20, 0.64, 0.12),
                    ("POC", volume_profile.poc, "#f6d06a", "-", 1.05, 24, 0.69, 0.13),
                )
            else:
                level_specs = _select_intraday_profile_levels(
                    volume_profile,
                    last_close=float(frame["close"].iloc[-1]),
                    y_min=y_min,
                    y_max=y_max,
                )
            for label, value, color, linestyle, linewidth, tail_bars, start_frac, length_frac in level_specs:
                if not show_liquidation_map and not _entry_level_visible(
                    value,
                    last_close=float(frame["close"].iloc[-1]),
                    y_min=y_min,
                    y_max=y_max,
                    max_distance_pct=0.072 if label != "POC" else 0.095,
                ):
                    continue
                level_start, level_end = _touch_segment(frame, value, tail_bars=tail_bars)
                _annotate_level(
                    ax_price,
                    x_values,
                    value,
                    color,
                    label,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    start_index=level_start,
                    end_index=level_end,
                    start_frac=start_frac,
                    length_frac=length_frac,
                )
    if show_entry_levels and not show_liquidation_map:
        _draw_trade_zone(ax_price, x_values, side=side, entry=entry, tp=tp, sl=sl)
        _annotate_level(
            ax_price,
            x_values,
            entry,
            "#74a8ff",
            "ENTRY",
            linestyle="-",
            linewidth=1.0,
            start_frac=0.82,
            length_frac=0.09,
        )
        _annotate_level(
            ax_price,
            x_values,
            tp,
            "#2dd07f",
            "TP",
            linestyle="-",
            linewidth=1.0,
            start_frac=0.84,
            length_frac=0.08,
        )
        _annotate_level(
            ax_price,
            x_values,
            sl,
            "#ff6f8f",
            "SL",
            linestyle="-",
            linewidth=1.0,
            start_frac=0.80,
            length_frac=0.10,
        )

    _draw_last_price_marker(ax_price, x_values, float(frame["close"].iloc[-1]))

    hist = pd.to_numeric(frame["hist"], errors="coerce").fillna(0.0)
    hist_colors = ["#65d4ac" if value >= 0 else "#f08f7e" for value in hist]
    if len(x_values) >= 2:
        bar_width = max((x_values[1] - x_values[0]) * 0.66, 1e-6)
    else:
        bar_width = 0.00045
    ax_macd.bar(x_values, hist, width=bar_width, color=hist_colors, alpha=0.56, zorder=2)
    ax_macd.plot(x_values, frame["macd"], color="#436dff", linewidth=1.28, alpha=0.98, zorder=3)
    ax_macd.plot(x_values, frame["signal"], color="#f5a046", linewidth=1.16, alpha=0.98, zorder=3)
    ax_macd.axhline(0.0, color="#7c8db1", linewidth=0.78, alpha=0.42, zorder=1)

    title = f"{symbol} | {timeframe_label}"
    subtitle = "Входной график: локальные уровни и TP/SL" if not show_liquidation_map else "Контекст: HTF уровни и карта ликвидаций"
    ax_price.set_title(title, color="#e9eef9", loc="left", fontsize=11.9, pad=10, fontweight="bold")
    subtitle = "Входной график: локальные уровни и TP/SL" if not show_liquidation_map else "Контекст: HTF уровни и карта ликвидаций"
    subtitle = "Входной график: локальные уровни и TP/SL" if not show_liquidation_map else "Контекст: HTF уровни и карта ликвидаций"
    subtitle = (
        "Входной график: локальные уровни и TP/SL"
        if not show_liquidation_map
        else "Контекст: HTF уровни и карта ликвидаций"
    )
    subtitle = (
        "Входной график: локальные уровни и TP/SL"
        if not show_liquidation_map
        else "Контекст: HTF уровни и карта ликвидаций"
    )
    subtitle = (
        "Входной график: локальные уровни и TP/SL"
        if not show_liquidation_map
        else "Контекст: HTF уровни и карта ликвидаций"
    )
    subtitle = (
        "Входной график: локальные уровни и TP/SL"
        if not show_liquidation_map
        else "Контекст: HTF уровни и карта ликвидаций"
    )
    subtitle = (
        "Входной график: локальные уровни и TP/SL"
        if not show_liquidation_map
        else "Контекст: HTF уровни и карта ликвидаций"
    )
    subtitle = (
        "Входной график: локальные уровни и TP/SL"
        if not show_liquidation_map
        else "Контекст: HTF уровни и карта ликвидаций"
    )
    subtitle = (
        "Входной график: локальные уровни и TP/SL"
        if not show_liquidation_map
        else "Контекст: HTF уровни и карта ликвидаций"
    )
    ax_price.text(
        0.015,
        1.012,
        subtitle,
        transform=ax_price.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.9,
        color="#8ea3c7",
    )
    ax_price.text(
        0.995,
        0.985,
        side.upper(),
        transform=ax_price.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color="#9fb0cf",
        bbox={"facecolor": "#111c2c", "edgecolor": "#23324b", "alpha": 0.55, "pad": 1.2},
    )

    _configure_time_axis(ax_macd, timeframe_label=timeframe_label, frame=frame)
    ax_price.yaxis.set_major_formatter(FuncFormatter(_price_tick_formatter))
    ax_macd.yaxis.set_major_formatter(FuncFormatter(_macd_tick_formatter))
    ax_price.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_macd.yaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.subplots_adjust(left=0.036, right=0.970, top=0.954, bottom=0.080, hspace=0.050)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=(176 if show_liquidation_map else 182), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()
