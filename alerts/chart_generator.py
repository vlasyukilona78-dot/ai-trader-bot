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


def _fmt_compact_notional(value: float) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(amount) or amount <= 0:
        return ""
    if amount >= 1_000_000_000:
        return f"{amount / 1_000_000_000:.1f}B".replace(".0B", "B")
    if amount >= 1_000_000:
        suffix_value = amount / 1_000_000
        return f"{suffix_value:.1f}M".replace(".0M", "M") if suffix_value < 10 else f"{suffix_value:.0f}M"
    if amount >= 1_000:
        suffix_value = amount / 1_000
        return f"{suffix_value:.0f}K" if suffix_value >= 10 else f"{suffix_value:.1f}K".replace(".0K", "K")
    return f"{amount:.0f}"


def _fmt_liquidation_margin_label(
    *,
    margin_usdt: float = 0.0,
    notional_usdt: float = 0.0,
    estimated: bool = False,
) -> str:
    margin_label = _fmt_compact_notional(margin_usdt)
    if margin_label:
        prefix = "~" if estimated else ""
        return f"{prefix}{margin_label} margin"
    notional_label = _fmt_compact_notional(notional_usdt)
    if not notional_label:
        return ""
    prefix = "~" if estimated else ""
    return f"{prefix}{notional_label} margin"


def _quote_volume_usdt_series(frame: pd.DataFrame) -> pd.Series:
    for column in ("turnover", "turnover_usdt", "quote_volume", "quoteVolume", "volume_usdt", "volumeUsd"):
        if column not in frame.columns:
            continue
        series = (
            pd.to_numeric(frame[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0)
        )
        if float(series.sum()) > 0.0:
            return series

    if "volume" not in frame.columns or "close" not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)

    volume = (
        pd.to_numeric(frame["volume"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    close = (
        pd.to_numeric(frame["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
        .clip(lower=0.0)
    )
    return (volume * close).fillna(0.0).clip(lower=0.0)


def _estimate_liquidation_margin_usdt(band, frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    quote_volume = _quote_volume_usdt_series(frame)
    if quote_volume.empty or float(quote_volume.sum()) <= 0.0:
        return 0.0

    start_idx = max(0, min(int(getattr(band, "start_index", 0)), len(frame) - 1))
    end_idx = max(0, min(int(getattr(band, "end_index", start_idx)), len(frame) - 1))
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx

    # Use only the neighborhood that created/sustained the level. This keeps labels
    # stable and avoids turning a full-window volume spike into a fake huge number.
    start_idx = max(0, start_idx - 2)
    end_idx = min(len(frame) - 1, end_idx + 2)
    segment = quote_volume.iloc[start_idx : end_idx + 1]
    if segment.empty:
        return 0.0

    segment_total = float(segment.tail(min(len(segment), 72)).sum())
    if not np.isfinite(segment_total) or segment_total <= 0.0:
        return 0.0

    intensity = min(max(float(getattr(band, "intensity", 0.35) or 0.35), 0.12), 1.0)
    weight = min(max(float(getattr(band, "weight", 1.0) or 1.0), 0.25), 10.0)
    factor = min(max(0.010 + intensity * 0.040 + weight * 0.0025, 0.010), 0.090)
    estimate = segment_total * factor
    if not np.isfinite(estimate) or estimate <= 0.0:
        return 0.0
    return min(max(estimate, 1_000.0), segment_total * 0.22)


def _format_chart_symbol(symbol: str) -> str:
    text = str(symbol or "").strip().upper().replace("/", "").replace("-", "")
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if text.endswith(quote) and len(text) > len(quote):
            return f"{text[:-len(quote)]}_{quote}"
    return text or "UNKNOWN"


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
        color = "#57c993" if close_px >= open_px else "#d45d5b"
        ax.vlines(x, low_px, high_px, color=color, linewidth=0.88, alpha=0.92, zorder=3)
        body_low = min(open_px, close_px)
        body_height = max(abs(close_px - open_px), max(close_px * 0.00017, 1e-8))
        ax.add_patch(
            Rectangle(
                (x - width / 2.0, body_low),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.52,
                alpha=0.93,
                zorder=4,
            )
        )


def _style_axis(ax, *, is_macd: bool = False):
    ax.set_facecolor("#20293a" if not is_macd else "#1e2738")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.55, alpha=0.105, color="#cfd8e8")
    ax.grid(True, axis="x", linestyle="-", linewidth=0.45, alpha=0.050, color="#cfd8e8")
    for spine in ax.spines.values():
        spine.set_color("#475268")
        spine.set_linewidth(1.05)
    ax.tick_params(
        colors="#9aa6b8",
        labelsize=7,
        length=0,
        pad=5,
        labelleft=True,
        labelright=True,
    )
    ax.yaxis.set_ticks_position("both")


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
        zorder=7,
    )


def _liquidation_label_x(ax, seg_start: float, seg_end: float) -> float:
    xmin, xmax = ax.get_xlim()
    span = max(xmax - xmin, 1e-8)
    width = max(seg_end - seg_start, 1e-8)
    ideal = seg_start + width * 0.72
    return min(ideal, xmax - span * 0.16)


def _is_external_liquidation_source(item) -> bool:
    return str(getattr(item, "source", "synthetic") or "").strip().lower() in {"feed", "coinglass", "external"}


def _liquidation_band_render_score(item, *, frame_len: int) -> float:
    duration = max(int(item.end_index) - int(item.start_index) + 1, 1)
    active_bonus = 1.30 if item.closed_index is None else 0.88
    source_bonus = 1.85 if _is_external_liquidation_source(item) else 1.0
    notional_bonus = min(max(float(getattr(item, "notional_usdt", 0.0) or 0.0) / 250_000.0, 0.0), 1.65)
    recency_bonus = 1.0
    if item.closed_index is not None:
        recency_bonus = 1.15 if int(item.closed_index) >= max(frame_len - 20, 0) else 0.80
    return item.weight * active_bonus * source_bonus * recency_bonus * (1.0 + notional_bonus) * min(1.0 + duration / 22.0, 2.4)


def _liquidation_band_active_on_side(band, *, side: str, last_close: float) -> bool:
    if band.closed_index is not None:
        return False
    if side == "above":
        return band.level > last_close
    return band.level < last_close


def _select_visible_liquidation_bands(
    liq_map: LiquidationMap | None,
    *,
    frame_len: int,
    last_close: float,
    max_bands: int = 8,
) -> list:
    if liq_map is None or not liq_map.bands:
        return []

    def _distance_ok(band, *, max_dist: float) -> bool:
        return abs((float(band.level) - last_close) / max(last_close, 1e-8)) <= max_dist

    def _notional_value(band) -> float:
        try:
            value = float(getattr(band, "notional_usdt", 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        return value if np.isfinite(value) else 0.0

    major_external = sorted(
        (
            band
            for band in liq_map.bands
            if _is_external_liquidation_source(band)
            and _notional_value(band) > 0
            and _distance_ok(band, max_dist=0.82)
            and max(int(band.end_index) - int(band.start_index) + 1, 1) >= 2
        ),
        key=lambda band: (
            _notional_value(band),
            _liquidation_band_render_score(band, frame_len=frame_len),
        ),
        reverse=True,
    )[:6]

    def _pick_visible_for_side(side: str) -> list:
        open_feed = sorted(
            (
                band
                for band in liq_map.bands
                if band.side == side
                and _liquidation_band_active_on_side(band, side=side, last_close=last_close)
                and _is_external_liquidation_source(band)
                and _distance_ok(band, max_dist=0.34)
            ),
            key=lambda band: _liquidation_band_render_score(band, frame_len=frame_len),
            reverse=True,
        )[:3]
        open_synthetic = sorted(
            (
                band
                for band in liq_map.bands
                if band.side == side
                and _liquidation_band_active_on_side(band, side=side, last_close=last_close)
                and not _is_external_liquidation_source(band)
                and _distance_ok(band, max_dist=0.18)
            ),
            key=lambda band: _liquidation_band_render_score(band, frame_len=frame_len),
            reverse=True,
        )[:2]
        recent_closed_feed = sorted(
            (
                band
                for band in liq_map.bands
                if band.side == side
                and band.closed_index is not None
                and _is_external_liquidation_source(band)
                and int(band.closed_index) >= max(frame_len - 34, 0)
                and max(int(band.end_index) - int(band.start_index) + 1, 1) >= 8
                and _distance_ok(band, max_dist=0.26)
            ),
            key=lambda band: _liquidation_band_render_score(band, frame_len=frame_len),
            reverse=True,
        )[:2]
        selected_open = open_feed + open_synthetic
        if selected_open and recent_closed_feed:
            open_levels = [float(band.level) for band in selected_open]
            recent_closed_feed = [
                band
                for band in recent_closed_feed
                if min(abs((float(band.level) - level) / max(last_close, 1e-8)) for level in open_levels) > 0.018
            ]
        return selected_open + recent_closed_feed

    combined = major_external + _pick_visible_for_side("above") + _pick_visible_for_side("below")
    deduped: list = []
    for band in sorted(
        combined,
        key=lambda band: _liquidation_band_render_score(band, frame_len=frame_len),
        reverse=True,
    ):
        if any(
            abs((float(band.level) - float(existing.level)) / max(last_close, 1e-8)) < 0.0045
            for existing in deduped
        ):
            continue
        deduped.append(band)
        if len(deduped) >= max_bands:
            break
    return deduped


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
    if show_liquidation_map:
        lows_all = pd.to_numeric(frame["low"], errors="coerce").dropna()
        highs_all = pd.to_numeric(frame["high"], errors="coerce").dropna()
        if lows_all.empty or highs_all.empty:
            return 0.0, max(last_close * 1.12, 1.0)

        focus_levels: list[float] = [
            float(lows_all.min()),
            float(highs_all.max()),
            last_close,
        ]
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
            focus_levels.append(float(body_lows.quantile(0.006)))
            focus_levels.append(float(body_highs.quantile(0.994)))

        for col in ("ema20", "ema50", "vwap"):
            if col in frame.columns:
                series = pd.to_numeric(frame[col], errors="coerce").dropna()
                if not series.empty:
                    focus_levels.append(float(series.min()))
                    focus_levels.append(float(series.max()))

        if volume_profile is not None and show_trade_levels:
            for value in (volume_profile.val, volume_profile.vah, volume_profile.poc):
                if value and value > 0:
                    focus_levels.append(float(value))

        if show_entry_levels:
            for value in (entry, tp, sl):
                if value and value > 0:
                    focus_levels.append(float(value))

        if liquidation_map is not None and liquidation_map.bands:
            for band in _select_visible_liquidation_bands(
                liquidation_map,
                frame_len=len(frame),
                last_close=last_close,
                max_bands=12,
            ):
                level = float(band.level)
                if level > 0:
                    focus_levels.append(level)

        raw_low = min(focus_levels)
        raw_high = max(focus_levels)
        robust_low = float(lows_all.quantile(0.006))
        robust_high = float(highs_all.quantile(0.994))
        robust_span = max(robust_high - robust_low, _price_scale(last_close) * 0.012)
        raw_span = max(raw_high - raw_low, robust_span)

        # Keep the wide Coinglass-like context, but avoid one broken wick flattening the entire chart.
        if raw_span > robust_span * 3.8 and len(frame) >= 80:
            low = min(robust_low, last_close, *(value for value in focus_levels if value >= robust_low - robust_span * 0.25))
            high = max(robust_high, last_close, *(value for value in focus_levels if value <= robust_high + robust_span * 0.25))
        else:
            low = raw_low
            high = raw_high

        span = max(high - low, _price_scale(last_close) * 0.018)
        pad = max(span * 0.075, _price_scale(last_close) * 0.0045)
        return max(0.0, low - pad), high + pad

    recent_bars = min(len(frame), 156 if show_liquidation_map else (56 if last_close < 0.02 else 66))
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

    if show_liquidation_map:
        context = frame.tail(min(len(frame), 220))
        context_lows = pd.to_numeric(context["low"], errors="coerce").dropna()
        context_highs = pd.to_numeric(context["high"], errors="coerce").dropna()
        if not context_lows.empty and not context_highs.empty:
            focus_levels.append(float(context_lows.quantile(0.04)))
            focus_levels.append(float(context_highs.quantile(0.96)))
        if {"open", "close"}.issubset(context.columns):
            context_body_lows = pd.concat(
                [
                    pd.to_numeric(context["open"], errors="coerce"),
                    pd.to_numeric(context["close"], errors="coerce"),
                ],
                axis=1,
            ).min(axis=1)
            context_body_highs = pd.concat(
                [
                    pd.to_numeric(context["open"], errors="coerce"),
                    pd.to_numeric(context["close"], errors="coerce"),
                ],
                axis=1,
            ).max(axis=1)
            focus_levels.append(float(context_body_lows.quantile(0.03)))
            focus_levels.append(float(context_body_highs.quantile(0.97)))

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
        for band in _select_visible_liquidation_bands(
            liquidation_map,
            frame_len=len(frame),
            last_close=last_close,
            max_bands=8,
        ):
            level = float(band.level)
            if level <= 0:
                continue
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
        locator = mdates.AutoDateLocator(minticks=5, maxticks=8, tz=tz)
        formatter = mdates.DateFormatter("%d.%m %H:%M", tz=tz)
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

    axis_low, axis_high = ax.get_ylim()
    price_span = max(float(axis_high) - float(axis_low), 1e-8)
    last_close = max(float(frame["close"].iloc[-1]), 1e-8)
    candle_width = max((x_values[1] - x_values[0]) * 0.60, 1e-6) if len(x_values) >= 2 else 0.00045
    half_step = candle_width / 1.20
    x_left = float(x_values[0])
    x_right = float(x_values[-1])
    x_span = max(x_right - x_left, 1e-8)
    visible_bands = _select_visible_liquidation_bands(
        liq_map,
        frame_len=len(frame),
        last_close=last_close,
        max_bands=12,
    )
    if not visible_bands:
        return

    for rank, band in enumerate(visible_bands):
        start_idx = max(0, min(int(band.start_index), len(x_values) - 1))
        end_idx = max(0, min(int(band.end_index), len(x_values) - 1))
        seg_start = float(x_values[start_idx]) - half_step
        seg_end = float(x_values[end_idx]) + half_step
        is_closed = band.closed_index is not None
        if not is_closed:
            seg_end = max(seg_end, x_right)
        if seg_end <= seg_start:
            seg_end = min(x_right, seg_start + x_span * 0.08)

        external = _is_external_liquidation_source(band)
        explicit_margin_usdt = float(getattr(band, "margin_usdt", 0.0) or 0.0)
        explicit_notional_usdt = float(getattr(band, "notional_usdt", 0.0) or 0.0)
        estimated_margin_usdt = 0.0
        estimated_label = False
        if explicit_margin_usdt <= 0.0 and explicit_notional_usdt <= 0.0:
            estimated_margin_usdt = _estimate_liquidation_margin_usdt(band, frame)
            estimated_label = estimated_margin_usdt > 0.0
        margin_label = _fmt_liquidation_margin_label(
            margin_usdt=explicit_margin_usdt if explicit_margin_usdt > 0.0 else estimated_margin_usdt,
            notional_usdt=explicit_notional_usdt,
            estimated=estimated_label,
        )
        important = bool(margin_label)
        if not important and rank >= 6:
            continue

        if band.side == "above":
            line_color = "#b86483" if (important and external) else "#9a667f"
            label_color = "#d7dce8" if important else "#a9b1c2"
            y_offset = price_span * (0.010 + 0.0025 * (rank % 3))
            va = "bottom"
        else:
            line_color = "#4c9a72" if (important and external) else "#5f9876"
            label_color = "#d7dce8" if important else "#a9b1c2"
            y_offset = -price_span * (0.010 + 0.0025 * (rank % 3))
            va = "top"

        line_alpha = 0.76 if important else 0.34
        if is_closed:
            line_alpha *= 0.62
            line_color = "#8992a2"

        line_width = 0.66 + 0.32 * min(max(float(band.intensity), 0.0), 1.0)
        if important:
            line_width += 0.18
        ax.hlines(
            band.level,
            seg_start,
            seg_end,
            color=line_color,
            linewidth=line_width,
            linestyles=(0, (4.0, 2.4)) if is_closed else "-",
            alpha=line_alpha,
            zorder=4.8,
        )

        if margin_label:
            label_x = min(max(seg_start + x_span * 0.010, x_left + x_span * 0.010), x_right - x_span * 0.070)
            ax.text(
                label_x,
                band.level + y_offset,
                margin_label,
                ha="left",
                va=va,
                fontsize=7.4,
                color=label_color,
                fontweight="bold",
                alpha=0.94 if important else 0.56,
                zorder=5.2,
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
    if show_liquidation_map:
        tf = str(timeframe_label or "").strip().lower()
        htf_tail = 432 if tf in {"1h", "60", "60m"} else 220
        source_frame = df.tail(htf_tail).copy()
    else:
        source_frame = df.tail(2520 if raw_last_close < 0.02 else 1840).copy()
    source_frame = _compute_macd(source_frame)
    source_frame = source_frame.dropna(subset=["open", "high", "low", "close"])
    if source_frame.empty:
        return None
    frame = source_frame if show_liquidation_map else _slice_entry_chart_frame(source_frame)
    if frame.empty:
        return None

    x_values = mdates.date2num(frame.index.to_pydatetime())
    fig_size = (18.0, 9.8) if show_liquidation_map else (16.0, 9.2)
    fig, (ax_price, ax_macd) = plt.subplots(
        2,
        1,
        figsize=fig_size,
        sharex=True,
        facecolor="#20293a",
        gridspec_kw={"height_ratios": [6.7, 1.45]},
    )
    _style_axis(ax_price)
    _style_axis(ax_macd, is_macd=True)

    _draw_candles(ax_price, frame, x_values)

    if show_liquidation_map:
        liquidation_map = liquidation_map or build_liquidation_map(frame)

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

    if not show_liquidation_map:
        _draw_last_price_marker(ax_price, x_values, float(frame["close"].iloc[-1]))

    hist = pd.to_numeric(frame["hist"], errors="coerce").fillna(0.0)
    hist_delta = hist.diff().fillna(0.0)
    hist_colors = []
    for value, delta in zip(hist, hist_delta):
        if value >= 0:
            hist_colors.append("#f1d36f" if delta >= 0 else "#69b79a")
        else:
            hist_colors.append("#d66772" if delta <= 0 else "#f0c36d")
    if len(x_values) >= 2:
        bar_width = max((x_values[1] - x_values[0]) * 0.66, 1e-6)
    else:
        bar_width = 0.00045
    ax_macd.bar(x_values, hist, width=bar_width, color=hist_colors, alpha=0.68, zorder=2)
    ax_macd.plot(x_values, frame["macd"], color="#244cff", linewidth=4.2, alpha=0.18, zorder=2.8)
    ax_macd.plot(x_values, frame["signal"], color="#ff7b20", linewidth=4.0, alpha=0.16, zorder=2.8)
    ax_macd.plot(x_values, frame["macd"], color="#2f68ff", linewidth=1.42, alpha=0.98, zorder=3)
    ax_macd.plot(x_values, frame["signal"], color="#ff8b25", linewidth=1.30, alpha=0.98, zorder=3)
    ax_macd.axhline(0.0, color="#9fa8bb", linewidth=0.72, alpha=0.34, zorder=1)

    title = f"{_format_chart_symbol(symbol)}  {timeframe_label}"
    ax_price.set_title(title, color="#f0f4fb", loc="left", fontsize=12.2, pad=10, fontweight="bold")

    _configure_time_axis(ax_macd, timeframe_label=timeframe_label, frame=frame)
    ax_price.yaxis.set_major_formatter(FuncFormatter(_price_tick_formatter))
    ax_macd.yaxis.set_major_formatter(FuncFormatter(_macd_tick_formatter))
    ax_price.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_macd.yaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.subplots_adjust(left=0.058, right=0.958, top=0.922, bottom=0.086, hspace=0.090)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=(166 if show_liquidation_map else 176), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()
