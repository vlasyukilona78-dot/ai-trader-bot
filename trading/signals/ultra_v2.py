from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import numpy as np
import pandas as pd

from core.liquidation_map import build_liquidation_map


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if frame is None or getattr(frame, "empty", True) or column not in frame.columns:
        return pd.Series(dtype=float)
    return (
        pd.to_numeric(frame[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def _last(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    values = _series(frame, column, default)
    if values.empty:
        return default
    return _safe_float(values.iloc[-1], default)


def _layer_details(metadata: Mapping[str, object] | None, layer_name: str) -> Mapping[str, object]:
    trace = metadata.get("layer_trace", {}) if isinstance(metadata, Mapping) else {}
    layers = trace.get("layers", {}) if isinstance(trace, Mapping) else {}
    layer = layers.get(layer_name, {}) if isinstance(layers, Mapping) else {}
    details = layer.get("details", {}) if isinstance(layer, Mapping) else {}
    return details if isinstance(details, Mapping) else {}


def _layer_flag(metadata: Mapping[str, object] | None, key: str) -> bool:
    for layer_name in ("layer2_weakness_confirmation", "layer3_entry_location"):
        value = _safe_float(_layer_details(metadata, layer_name).get(key), 0.0)
        if value > 0.0:
            return True
    return False


def _latest_peak_age(values: pd.Series, *, reference: float, tolerance_pct: float = 0.0015) -> int:
    if values.empty or reference <= 0.0:
        return 999
    tolerance = max(abs(reference) * tolerance_pct, 1e-12)
    arr = values.to_numpy(dtype=float)
    for offset, value in enumerate(reversed(arr)):
        if value >= reference - tolerance:
            return offset
    return len(arr)


@dataclass(frozen=True)
class UltraV2Config:
    min_pump_pct: float = 0.028
    min_rsi: float = 54.0
    min_volume_spike: float = 1.20
    max_peak_age_bars: int = 5
    min_pump_range_position: float = 0.65
    continuation_reject_score: float = 1.85
    radar_score: float = 0.68
    confirm_score: float = 0.78
    direct_confirm_score: float = 0.84


@dataclass(frozen=True)
class UltraV2Result:
    accepted: bool
    phase: str = ""
    scenario: str = ""
    score: float = 0.0
    reason: str = ""
    triggers: tuple[str, ...] = ()
    level_label: str = ""
    level_value: float = 0.0
    fib50: float = 0.0
    sl: float = 0.0
    features: Mapping[str, float | bool | str] = field(default_factory=dict)


def _liquidation_context(frame: pd.DataFrame, *, close: float, high: float, atr: float) -> tuple[bool, bool, str, float]:
    try:
        liquidation_map = build_liquidation_map(frame)
    except Exception:
        return False, False, "", 0.0

    tolerance = max(atr / max(close, 1e-8) * 1.7, 0.0065 if close < 0.02 else 0.0040)
    level_label = ""
    level_value = 0.0
    swept_above = bool(getattr(liquidation_map, "swept_above", False))
    downside_magnet = bool(getattr(liquidation_map, "downside_magnet", False))
    for band in getattr(liquidation_map, "bands", ()):
        if str(getattr(band, "side", "")) != "above":
            continue
        level = _safe_float(getattr(band, "level", 0.0), 0.0)
        if level <= 0.0:
            continue
        touched = high >= level * (1.0 - tolerance) and close <= level * (1.0 + tolerance)
        closed = getattr(band, "closed_index", None) is not None
        if not (touched or closed):
            continue
        amount = _safe_float(getattr(band, "margin_usdt", 0.0), 0.0) or _safe_float(
            getattr(band, "notional_usdt", 0.0),
            0.0,
        )
        if amount >= 1_000_000:
            amount_label = f"{amount / 1_000_000:.1f}M"
        elif amount >= 1_000:
            amount_label = f"{amount / 1_000:.0f}K"
        else:
            amount_label = ""
        level_label = f"LQ UP {amount_label}".strip()
        level_value = level
        break
    return swept_above, downside_magnet, level_label, level_value


def evaluate_ultra_v2(
    frame: pd.DataFrame,
    *,
    metadata: Mapping[str, object] | None = None,
    config: UltraV2Config | None = None,
) -> UltraV2Result:
    cfg = config or UltraV2Config()
    if frame is None or getattr(frame, "empty", True) or len(frame) < 36:
        return UltraV2Result(False, reason="insufficient_history")

    close = _last(frame, "close", 0.0)
    if close <= 0.0:
        return UltraV2Result(False, reason="missing_close")
    open_px = _last(frame, "open", close)
    high = _last(frame, "high", close)
    low = _last(frame, "low", close)
    prev_close = _safe_float(_series(frame, "close", close).iloc[-2], close) if len(frame) >= 2 else close
    prev_high = _safe_float(_series(frame, "high", high).iloc[-2], high) if len(frame) >= 2 else high
    atr = max(_last(frame, "atr", close * 0.01), close * 0.0012, 1e-8)

    highs = _series(frame.tail(64), "high", high)
    lows = _series(frame.tail(64), "low", low)
    closes = _series(frame.tail(8), "close", close)
    if highs.empty or lows.empty:
        return UltraV2Result(False, reason="missing_price_window")

    recent_high = float(highs.max())
    pump_low = float(lows.min())
    pump_pct = max(recent_high / max(pump_low, 1e-8) - 1.0, 0.0)
    peak_age = _latest_peak_age(highs, reference=recent_high)
    pump_range_position = (close - pump_low) / max(recent_high - pump_low, 1e-8)
    pump_range_position = min(max(pump_range_position, 0.0), 1.0)
    if pump_pct < cfg.min_pump_pct:
        return UltraV2Result(False, reason="pump_too_small")
    if peak_age > cfg.max_peak_age_bars:
        return UltraV2Result(False, reason="peak_too_old")
    if pump_range_position < cfg.min_pump_range_position:
        return UltraV2Result(False, reason="not_near_upper_pump_range")

    rsi = _last(frame, "rsi", 50.0)
    prev_rsi = _safe_float(_series(frame, "rsi", rsi).iloc[-2], rsi) if "rsi" in frame.columns and len(frame) >= 2 else rsi
    volume_spikes = _series(frame.tail(10), "volume_spike", 1.0)
    volume_spike = _last(frame, "volume_spike", 1.0)
    recent_volume_spike = max(float(volume_spikes.max()) if not volume_spikes.empty else volume_spike, volume_spike)
    volume_peak_age = _latest_peak_age(volume_spikes, reference=recent_volume_spike, tolerance_pct=0.0001)
    volume_decay_ratio = volume_spike / max(recent_volume_spike, 1e-8)
    volume_climax_recent = recent_volume_spike >= cfg.min_volume_spike and volume_peak_age <= 4
    if rsi < cfg.min_rsi:
        return UltraV2Result(False, reason="rsi_too_low")
    if volume_spike < cfg.min_volume_spike and not volume_climax_recent:
        return UltraV2Result(False, reason="no_volume_climax")

    hist = _last(frame, "hist", 0.0)
    prev_hist = _safe_float(_series(frame, "hist", hist).iloc[-2], hist) if "hist" in frame.columns and len(frame) >= 2 else hist
    vwap_dist = _last(frame, "vwap_dist", 0.0)
    bb_position = _last(frame, "bb_position", 0.5)
    mtf_trend_5m = _last(frame, "mtf_trend_5m", 0.0)
    mtf_rsi_5m = _last(frame, "mtf_rsi_5m", 50.0)
    mtf_trend_15m = _last(frame, "mtf_trend_15m", 0.0)
    mtf_rsi_15m = _last(frame, "mtf_rsi_15m", 50.0)

    candle_range = max(high - low, atr * 0.35, close * 0.0008, 1e-8)
    upper_wick_ratio = max(high - max(open_px, close), 0.0) / candle_range
    close_position = (close - low) / candle_range
    body_pct = (close - open_px) / max(close, 1e-8)
    pullback_from_high = max((recent_high - close) / max(recent_high, 1e-8), 0.0)
    last3_up = int((closes.diff().tail(3) > 0).sum()) if not closes.empty else 0

    failed_reclaim = _layer_flag(metadata, "failed_reclaim")
    retest_failed_breakout = _layer_flag(metadata, "retest_failed_breakout")
    acceptance_above_high = _layer_flag(metadata, "acceptance_above_swing_high")
    peak_followthrough = _layer_flag(metadata, "peak_followthrough_confirmed")
    downside_displacement = _layer_flag(metadata, "downside_displacement_confirmed")
    swept_above, downside_magnet, lq_label, lq_level = _liquidation_context(
        frame,
        close=close,
        high=high,
        atr=atr,
    )

    rsi_rollover = rsi < prev_rsi
    hist_rollover = hist < prev_hist
    volume_fade = volume_decay_ratio <= 0.82 and volume_peak_age <= 5
    rejection_bar = bool(close < open_px and close < prev_close and close_position <= 0.55)
    sweep_fail = bool((swept_above or high > prev_high) and close <= prev_high * 1.002 and close_position <= 0.58)
    bullish_continuation_bar = bool(close > open_px and close_position >= 0.70 and upper_wick_ratio < 0.16)
    live_peak_extension = bool(close >= recent_high * 0.992 and close_position >= 0.60 and body_pct >= -0.001)
    reacceleration_after_pullback = bool(last3_up >= 2 and close >= prev_close and not volume_fade)

    continuation_score = 0.0
    if bullish_continuation_bar:
        continuation_score += 0.70
    if live_peak_extension:
        continuation_score += 0.55
    if reacceleration_after_pullback:
        continuation_score += 0.50
    if mtf_trend_5m >= 0.007 and mtf_rsi_5m >= 64.0:
        continuation_score += 0.42
    if mtf_trend_15m >= 0.004 and mtf_rsi_15m >= 62.0:
        continuation_score += 0.34
    if acceptance_above_high:
        continuation_score += 0.70
    if continuation_score >= cfg.continuation_reject_score:
        return UltraV2Result(False, reason="continuation_blocker")

    scenarios: dict[str, float] = {}
    scenarios["sweep_and_fail"] = (
        0.26 * float(swept_above or sweep_fail)
        + 0.24 * float(failed_reclaim or retest_failed_breakout or rejection_bar)
        + 0.18 * float(not acceptance_above_high)
        + 0.14 * float(upper_wick_ratio >= 0.18)
        + 0.10 * float(volume_fade)
        + 0.08 * float(rsi_rollover or hist_rollover)
    )
    scenarios["blow_off_wick"] = (
        0.30 * float(upper_wick_ratio >= 0.24)
        + 0.20 * float(close_position <= 0.52)
        + 0.15 * float(pump_range_position >= 0.78)
        + 0.15 * float(recent_volume_spike >= max(cfg.min_volume_spike, 1.55))
        + 0.10 * float(rsi >= 58.0)
        + 0.10 * float(rsi_rollover or hist_rollover)
    )
    scenarios["climax_then_fade"] = (
        0.24 * float(pump_pct >= 0.035)
        + 0.22 * float(1 <= volume_peak_age <= 5)
        + 0.20 * float(volume_fade)
        + 0.16 * float(rsi_rollover or hist_rollover)
        + 0.10 * float(vwap_dist >= 0.018 or bb_position >= 0.72)
        + 0.08 * float(peak_followthrough or downside_displacement)
    )
    scenarios["failed_continuation_after_pullback"] = (
        0.24 * float(peak_age <= 5)
        + 0.24 * float(failed_reclaim or retest_failed_breakout or rejection_bar)
        + 0.16 * float(not acceptance_above_high)
        + 0.14 * float(volume_fade)
        + 0.12 * float(rsi_rollover or hist_rollover)
        + 0.10 * float(downside_magnet or downside_displacement)
    )

    scenario = max(scenarios, key=scenarios.get)
    if (swept_above or sweep_fail) and scenarios["sweep_and_fail"] >= scenarios[scenario] - 0.05:
        scenario = "sweep_and_fail"
    scenario_core = scenarios[scenario]
    peak_context = min(1.0, 0.45 * float(peak_age <= 2) + 0.35 * pump_range_position + 0.20 * float(close >= recent_high * 0.985))
    slowdown = min(1.0, 0.36 * float(volume_fade) + 0.32 * float(rsi_rollover) + 0.32 * float(hist_rollover))
    rejection = min(
        1.0,
        0.34 * float(rejection_bar)
        + 0.28 * float(upper_wick_ratio >= 0.20)
        + 0.38 * float(failed_reclaim or retest_failed_breakout or sweep_fail),
    )
    liquidity_context = min(1.0, 0.55 * float(swept_above) + 0.45 * float(downside_magnet or lq_level > 0.0))
    fib50 = pump_low + (recent_high - pump_low) * 0.50
    if fib50 >= close:
        fib50 = close - max(atr * 2.0, close * 0.006)
    downside_room = max((close - fib50) / max(close, 1e-8), 0.0)
    downside_context = min(1.0, downside_room / 0.035)

    score = (
        0.26 * scenario_core
        + 0.18 * peak_context
        + 0.18 * slowdown
        + 0.16 * rejection
        + 0.12 * liquidity_context
        + 0.10 * downside_context
        - 0.18 * continuation_score
    )
    score = min(max(score, 0.0), 1.0)
    if score < cfg.radar_score:
        return UltraV2Result(False, reason="score_below_radar", scenario=scenario, score=score)

    micro_confirmed = bool(
        close < prev_close
        or rsi_rollover
        or hist_rollover
        or failed_reclaim
        or retest_failed_breakout
        or rejection_bar
        or volume_fade
        or swept_above
    )
    phase = "ULTRA_CONFIRM" if score >= cfg.confirm_score and micro_confirmed else "ULTRA_RADAR"
    if score >= cfg.direct_confirm_score and not acceptance_above_high:
        phase = "ULTRA_CONFIRM"

    level_label = lq_label or "VAH/\u0432\u0435\u0440\u0445\u043d\u044f\u044f \u0437\u043e\u043d\u0430"
    level_value = lq_level or recent_high
    sl = max(level_value, recent_high) + max(atr * 0.42, close * (0.0042 if close < 1.0 else 0.0024))
    triggers = []
    if swept_above or sweep_fail:
        triggers.append("sweep_and_fail")
    if upper_wick_ratio >= 0.20:
        triggers.append("upper_wick")
    if volume_fade:
        triggers.append("volume_decay")
    if rsi_rollover:
        triggers.append("rsi_rollover")
    if hist_rollover:
        triggers.append("macd_rollover")
    if failed_reclaim:
        triggers.append("failed_reclaim")
    if retest_failed_breakout:
        triggers.append("retest_failed_breakout")
    if downside_magnet:
        triggers.append("downside_liq_magnet")

    return UltraV2Result(
        True,
        phase=phase,
        scenario=scenario,
        score=score,
        reason="ultra_v2",
        triggers=tuple(triggers),
        level_label=level_label,
        level_value=level_value,
        fib50=fib50,
        sl=sl,
        features={
            "pump_pct": pump_pct,
            "peak_age_bars": float(peak_age),
            "pump_range_position": pump_range_position,
            "volume_decay_ratio": volume_decay_ratio,
            "volume_peak_age": float(volume_peak_age),
            "upper_wick_ratio": upper_wick_ratio,
            "close_position": close_position,
            "momentum_decay_score": slowdown,
            "peak_hold_quality": max(0.0, min(1.0, close_position - upper_wick_ratio * 0.35)),
            "continuation_score": continuation_score,
            "downside_room": downside_room,
            "swept_above": swept_above,
            "downside_magnet": downside_magnet,
        },
    )
