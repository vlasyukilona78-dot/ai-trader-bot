from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.strategy_interface import StrategyContext
from trading.signals.versioning import STRATEGY_RUNTIME_VERSION, runtime_versions


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _mapping_get(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _details_from_trace(trace_meta: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(trace_meta, Mapping):
        return {}
    trace = trace_meta.get("layer_trace")
    if not isinstance(trace, Mapping):
        trace = trace_meta
    layers = trace.get("layers") if isinstance(trace, Mapping) else None
    if not isinstance(layers, Mapping):
        return dict(trace_meta)

    out: dict[str, Any] = {}
    for layer in layers.values():
        if not isinstance(layer, Mapping):
            continue
        details = layer.get("details")
        if isinstance(details, Mapping):
            out.update(details)
    return out


@dataclass(frozen=True)
class UltraShortConfig:
    min_score_a: float = 0.80
    min_score_a_plus: float = 0.86
    max_continuation_risk_a: float = 1.35
    max_continuation_risk_a_plus: float = 1.15
    min_rr: float = 1.60
    min_rr_a_plus: float = 1.80
    max_stop_atr: float = 1.35
    max_stop_atr_a_plus: float = 1.15
    min_clean_pump_pct: float = 0.030
    min_volume_spike: float = 1.25
    min_pump_range_position: float = 0.70
    max_entry_distance_from_high_atr: float = 1.20
    stop_buffer_atr: float = 0.20
    tp_min_move_pct: float = 0.030
    tp_max_move_pct: float = 0.070


@dataclass(frozen=True)
class UltraShortFeatures:
    symbol: str
    timeframe: str
    close: float
    high: float
    low: float
    open: float
    prev_close: float
    prev_high: float
    atr: float
    atr_pct: float
    rsi: float
    prev_rsi: float
    hist: float
    prev_hist: float
    volume_spike: float
    recent_volume_spike: float
    volume_decay_ratio: float
    volume_peak_age: int
    clean_pump_pct: float
    recent_high: float
    recent_low: float
    peak_age_bars: int
    pump_range_position: float
    entry_distance_from_high_atr: float
    upper_wick_ratio: float
    close_position_in_candle: float
    vwap: float
    vah: float
    poc: float
    val: float
    mtf_trend_5m: float
    mtf_trend_15m: float
    mtf_rsi_5m: float
    mtf_rsi_15m: float
    obv_delta: float
    cvd_delta: float
    swept_above: bool
    downside_magnet: bool
    upside_risk: bool
    failed_reclaim: bool
    retest_failed_breakout: bool
    acceptance_above_high: bool
    rejection_bar: bool
    near_sweep_level: bool


@dataclass(frozen=True)
class UltraShortDecision:
    approved: bool
    scenario: str
    grade: str
    score: float
    entry: float
    stop_loss: float
    take_profit: float
    invalidation_level: float
    rr: float
    continuation_risk: float
    reason: str
    setup_signature: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    symbol: str = ""
    timeframe: str = "1m"

    def to_intent(self) -> StrategyIntent:
        metadata = {
            "signal_profile": "ultra",
            "ultra_scenario": self.scenario,
            "ultra_grade": self.grade,
            "ultra_score": float(self.score),
            "continuation_risk": float(self.continuation_risk),
            "rr": float(self.rr),
            "invalidation_level": float(self.invalidation_level),
            "entry_price": float(self.entry),
            "setup_signature": self.setup_signature,
            "timeframe": self.timeframe,
            "strategy_version": STRATEGY_RUNTIME_VERSION,
            "runtime_versions": runtime_versions(),
            "ultra_diagnostics": dict(self.diagnostics),
        }
        return StrategyIntent(
            symbol=self.symbol,
            action=IntentAction.SHORT_ENTRY,
            reason="ultra_short_entry",
            stop_loss=float(self.stop_loss),
            take_profit=float(self.take_profit),
            confidence=float(self.score),
            metadata=metadata,
        )


@dataclass(frozen=True)
class _TradePlan:
    entry: float
    stop_loss: float
    take_profit: float
    invalidation_level: float
    rr: float
    stop_distance_atr: float
    target_move_pct: float
    target_source: str
    diagnostics: dict[str, Any]


class UltraShortEntryDetector:
    """Precision short detector for pump culmination at a reversal level."""

    def __init__(self, config: UltraShortConfig | None = None):
        self.config = config or UltraShortConfig()

    def evaluate(
        self,
        context: StrategyContext,
        *,
        trace_meta: Mapping[str, Any] | None = None,
        liquidation_map: Any | Mapping[str, Any] | None = None,
        volume_profile: Any | Mapping[str, Any] | None = None,
    ) -> UltraShortDecision:
        try:
            features = self._extract_features(
                context,
                trace_meta=trace_meta,
                liquidation_map=liquidation_map,
                volume_profile=volume_profile,
            )
        except ValueError as exc:
            return self._reject(
                symbol=context.symbol,
                timeframe=context.timeframe or "1m",
                reason=str(exc),
                diagnostics={"reject_stage": "feature_extraction"},
            )

        diagnostics: dict[str, Any] = {"features": self._feature_diagnostics(features)}

        if features.clean_pump_pct < self.config.min_clean_pump_pct:
            return self._reject_from_features(
                features,
                "clean_pump_below_min",
                diagnostics | {"clean_pump_pct": features.clean_pump_pct},
            )
        if features.pump_range_position < self.config.min_pump_range_position:
            return self._reject_from_features(
                features,
                "pump_range_position_below_min",
                diagnostics | {"pump_range_position": features.pump_range_position},
            )
        if features.entry_distance_from_high_atr > self.config.max_entry_distance_from_high_atr:
            return self._reject_from_features(
                features,
                "entry_too_far_from_high",
                diagnostics | {"entry_distance_from_high_atr": features.entry_distance_from_high_atr},
            )
        if features.acceptance_above_high:
            return self._reject_from_features(features, "acceptance_above_high", diagnostics)

        reversal_level_score = self._reversal_level_score(features)
        pump_context_score = self._pump_context_score(features)
        exhaustion_score = self._exhaustion_score(features)
        failed_acceptance_score = self._failed_acceptance_score(features)
        liquidity_sweep_score = self._liquidity_sweep_score(features)
        downside_target_score = self._downside_target_score(features)
        continuation_risk = self._continuation_risk(features)
        scenario, scenario_score = self._scenario_score(features)

        diagnostics.update(
            {
                "reversal_level_score": reversal_level_score,
                "pump_context_score": pump_context_score,
                "exhaustion_score": exhaustion_score,
                "failed_acceptance_score": failed_acceptance_score,
                "liquidity_sweep_score": liquidity_sweep_score,
                "downside_target_score": downside_target_score,
                "continuation_risk": continuation_risk,
                "scenario": scenario,
                "scenario_score": scenario_score,
            }
        )

        if scenario_score <= 0.0:
            return self._reject_from_features(features, "no_ultra_scenario", diagnostics)
        if continuation_risk > self.config.max_continuation_risk_a:
            return self._reject_from_features(features, "continuation_risk_too_high", diagnostics)
        if failed_acceptance_score < 0.35:
            return self._reject_from_features(features, "failed_acceptance_too_weak", diagnostics)

        plan = self._build_trade_plan(features)
        diagnostics["trade_plan"] = plan.diagnostics
        if plan.rr < self.config.min_rr:
            return self._reject_from_features(features, "rr_below_min", diagnostics | {"rr": plan.rr})
        if plan.stop_distance_atr > self.config.max_stop_atr:
            return self._reject_from_features(
                features,
                "stop_distance_atr_too_wide",
                diagnostics | {"stop_distance_atr": plan.stop_distance_atr},
            )

        score = _clamp(
            0.24 * reversal_level_score
            + 0.18 * pump_context_score
            + 0.18 * exhaustion_score
            + 0.18 * failed_acceptance_score
            + 0.10 * liquidity_sweep_score
            + 0.07 * downside_target_score
            + 0.05 * scenario_score
            - max(0.0, continuation_risk - 1.0) * 0.10
        )
        diagnostics["score"] = score

        if score < self.config.min_score_a:
            return self._reject_from_features(features, "score_below_min", diagnostics)

        grade = "A+"
        if not (
            score >= self.config.min_score_a_plus
            and continuation_risk <= self.config.max_continuation_risk_a_plus
            and plan.rr >= self.config.min_rr_a_plus
            and plan.stop_distance_atr <= self.config.max_stop_atr_a_plus
        ):
            grade = "A"

        signature = self._setup_signature(features, scenario, plan)
        return UltraShortDecision(
            approved=True,
            scenario=scenario,
            grade=grade,
            score=score,
            entry=plan.entry,
            stop_loss=plan.stop_loss,
            take_profit=plan.take_profit,
            invalidation_level=plan.invalidation_level,
            rr=plan.rr,
            continuation_risk=continuation_risk,
            reason="approved",
            setup_signature=signature,
            diagnostics=diagnostics,
            symbol=features.symbol,
            timeframe=features.timeframe,
        )

    def _extract_features(
        self,
        context: StrategyContext,
        *,
        trace_meta: Mapping[str, Any] | None,
        liquidation_map: Any | Mapping[str, Any] | None,
        volume_profile: Any | Mapping[str, Any] | None,
    ) -> UltraShortFeatures:
        df = context.market_ohlcv
        if df is None or getattr(df, "empty", True) or len(df) < 20:
            raise ValueError("insufficient_history")

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        tail = df.tail(min(48, len(df)))
        recent = df.tail(min(16, len(df)))
        vol_tail = df.tail(min(8, len(df)))

        close = _safe_float(latest.get("close"))
        high = _safe_float(latest.get("high"), close)
        low = _safe_float(latest.get("low"), close)
        open_px = _safe_float(latest.get("open"), close)
        prev_close = _safe_float(prev.get("close"), close)
        prev_high = _safe_float(prev.get("high"), high)
        atr = _safe_float(latest.get("atr"))
        if atr <= 0.0:
            atr = max(high - low, close * 0.003, 1e-8)
        atr_pct = atr / close if close > 0 else 0.0

        rsi = _safe_float(latest.get("rsi"), 50.0)
        prev_rsi = _safe_float(prev.get("rsi"), rsi)
        hist_key = "hist" if "hist" in df.columns else "macd_hist"
        hist = _safe_float(latest.get(hist_key), 0.0)
        prev_hist = _safe_float(prev.get(hist_key), hist)
        volume_spike = _safe_float(latest.get("volume_spike"), 1.0)
        recent_volume_spike = max(0.0, *[_safe_float(v, 0.0) for v in vol_tail.get("volume_spike", pd.Series([volume_spike])).tolist()])
        recent_volume_spike = recent_volume_spike if recent_volume_spike > 0 else volume_spike
        volume_decay_ratio = volume_spike / recent_volume_spike if recent_volume_spike > 0 else 1.0
        volume_values = [_safe_float(v, 0.0) for v in vol_tail.get("volume_spike", pd.Series([volume_spike])).tolist()]
        volume_peak_age = 0
        if volume_values:
            volume_peak_age = len(volume_values) - 1 - int(max(range(len(volume_values)), key=lambda i: volume_values[i]))

        explicit_clean = _safe_float(latest.get("clean_pump_pct"), -1.0)
        recent_low = _safe_float(tail["low"].min(), low) if "low" in tail else low
        recent_high = _safe_float(recent["high"].max(), high) if "high" in recent else high
        if explicit_clean >= 0.0:
            clean_pump_pct = explicit_clean
        else:
            clean_pump_pct = (close - recent_low) / recent_low if recent_low > 0 else 0.0

        highs = [_safe_float(v, high) for v in recent.get("high", pd.Series([high])).tolist()]
        peak_age_bars = 0
        if highs:
            peak_age_bars = len(highs) - 1 - int(max(range(len(highs)), key=lambda i: highs[i]))

        range_width = max(recent_high - recent_low, 1e-8)
        pump_range_position = _clamp((close - recent_low) / range_width)
        entry_distance_from_high_atr = max(0.0, (recent_high - close) / max(atr, 1e-8))
        candle_range = max(high - low, 1e-8)
        upper_wick_ratio = max(0.0, high - max(open_px, close)) / candle_range
        close_position_in_candle = _clamp((close - low) / candle_range)

        vp_vah = _safe_float(_mapping_get(volume_profile, "vah"))
        vp_poc = _safe_float(_mapping_get(volume_profile, "poc"))
        vp_val = _safe_float(_mapping_get(volume_profile, "val"))
        vwap = _safe_float(latest.get("vwap"), 0.0)
        vah = _safe_float(latest.get("vah"), vp_vah)
        poc = _safe_float(latest.get("poc"), vp_poc)
        val = _safe_float(latest.get("val"), vp_val)

        obv_delta = _safe_float(latest.get("obv"), 0.0) - _safe_float(prev.get("obv"), 0.0)
        cvd_delta = _safe_float(latest.get("cvd"), 0.0) - _safe_float(prev.get("cvd"), 0.0)

        details = _details_from_trace(trace_meta)
        failed_reclaim = _safe_bool(details.get("failed_reclaim") or latest.get("failed_reclaim"))
        retest_failed_breakout = _safe_bool(details.get("retest_failed_breakout") or latest.get("retest_failed_breakout"))
        acceptance_above_high = _safe_bool(
            details.get("acceptance_above_high")
            or details.get("acceptance_above_swing_high")
            or latest.get("acceptance_above_high")
            or latest.get("acceptance_above_swing_high")
        )
        rejection_bar = _safe_bool(details.get("rejection_bar") or latest.get("rejection_bar"))
        near_sweep_level = _safe_bool(details.get("near_sweep_level") or latest.get("near_sweep_level"))

        swept_above = _safe_bool(_mapping_get(liquidation_map, "swept_above", latest.get("swept_above", False)))
        downside_magnet = _safe_bool(_mapping_get(liquidation_map, "downside_magnet", latest.get("downside_magnet", False)))
        upside_risk = _safe_bool(_mapping_get(liquidation_map, "upside_risk", latest.get("upside_risk", False)))

        return UltraShortFeatures(
            symbol=str(context.symbol).replace("/", "").upper(),
            timeframe=str(context.timeframe or "1m"),
            close=close,
            high=high,
            low=low,
            open=open_px,
            prev_close=prev_close,
            prev_high=prev_high,
            atr=atr,
            atr_pct=atr_pct,
            rsi=rsi,
            prev_rsi=prev_rsi,
            hist=hist,
            prev_hist=prev_hist,
            volume_spike=volume_spike,
            recent_volume_spike=recent_volume_spike,
            volume_decay_ratio=volume_decay_ratio,
            volume_peak_age=volume_peak_age,
            clean_pump_pct=clean_pump_pct,
            recent_high=max(recent_high, high),
            recent_low=recent_low,
            peak_age_bars=peak_age_bars,
            pump_range_position=pump_range_position,
            entry_distance_from_high_atr=entry_distance_from_high_atr,
            upper_wick_ratio=upper_wick_ratio,
            close_position_in_candle=close_position_in_candle,
            vwap=vwap,
            vah=vah,
            poc=poc,
            val=val,
            mtf_trend_5m=_safe_float(latest.get("mtf_trend_5m"), 0.0),
            mtf_trend_15m=_safe_float(latest.get("mtf_trend_15m"), 0.0),
            mtf_rsi_5m=_safe_float(latest.get("mtf_rsi_5m"), 50.0),
            mtf_rsi_15m=_safe_float(latest.get("mtf_rsi_15m"), 50.0),
            obv_delta=obv_delta,
            cvd_delta=cvd_delta,
            swept_above=swept_above,
            downside_magnet=downside_magnet,
            upside_risk=upside_risk,
            failed_reclaim=failed_reclaim,
            retest_failed_breakout=retest_failed_breakout,
            acceptance_above_high=acceptance_above_high,
            rejection_bar=rejection_bar,
            near_sweep_level=near_sweep_level,
        )

    def _reversal_level_score(self, features: UltraShortFeatures) -> float:
        score = 0.0
        if features.recent_high > 0:
            high_distance_atr = abs(features.recent_high - features.close) / max(features.atr, 1e-8)
            if high_distance_atr <= 0.35:
                score += 0.28
            elif high_distance_atr <= 0.80:
                score += 0.22
            elif high_distance_atr <= self.config.max_entry_distance_from_high_atr:
                score += 0.14
        if features.vah > 0 and abs(features.close - features.vah) / features.close <= max(0.007, features.atr_pct * 1.2):
            score += 0.22
        if features.vwap > 0 and features.close > features.vwap * (1.0 + max(0.008, features.atr_pct * 0.9)):
            score += 0.20
        if features.swept_above:
            score += 0.18
        if features.near_sweep_level:
            score += 0.12
        return _clamp(score)

    def _pump_context_score(self, features: UltraShortFeatures) -> float:
        clean = _clamp((features.clean_pump_pct - self.config.min_clean_pump_pct) / 0.07)
        volume = _clamp(max(features.volume_spike, features.recent_volume_spike) / 2.4)
        peak = 1.0 if features.peak_age_bars <= 2 else (0.65 if features.peak_age_bars <= 5 else 0.25)
        range_pos = _clamp((features.pump_range_position - 0.60) / 0.40)
        return _clamp(0.36 * clean + 0.24 * volume + 0.22 * range_pos + 0.18 * peak)

    def _exhaustion_score(self, features: UltraShortFeatures) -> float:
        score = 0.0
        if features.volume_decay_ratio <= 0.82:
            score += 0.20
        elif features.volume_decay_ratio <= 0.90:
            score += 0.10
        if features.rsi < features.prev_rsi:
            score += 0.16
        if features.hist < features.prev_hist:
            score += 0.16
        if features.upper_wick_ratio >= 0.22:
            score += 0.17
        if features.close_position_in_candle <= 0.55:
            score += 0.15
        if features.obv_delta < 0:
            score += 0.08
        if features.cvd_delta < 0:
            score += 0.08
        return _clamp(score)

    def _failed_acceptance_score(self, features: UltraShortFeatures) -> float:
        score = 0.0
        if features.failed_reclaim:
            score += 0.28
        if features.retest_failed_breakout:
            score += 0.24
        if features.rejection_bar:
            score += 0.20
        if features.upper_wick_ratio >= 0.28 and features.close_position_in_candle <= 0.48:
            score += 0.18
        if features.close < features.recent_high and features.entry_distance_from_high_atr <= 0.75:
            score += 0.16
        elif features.close < features.recent_high and features.entry_distance_from_high_atr <= self.config.max_entry_distance_from_high_atr:
            score += 0.12
        if features.vah > 0 and features.close < features.vah:
            score += 0.12
        return _clamp(score)

    def _liquidity_sweep_score(self, features: UltraShortFeatures) -> float:
        score = 0.0
        if features.swept_above:
            score += 0.45
        if features.near_sweep_level:
            score += 0.25
        if features.downside_magnet:
            score += 0.30
        return _clamp(score)

    def _downside_target_score(self, features: UltraShortFeatures) -> float:
        targets = self._target_candidates(features)
        best_move = max([move for _, _, move in targets], default=0.0)
        score = 0.0
        if best_move >= self.config.tp_min_move_pct:
            score += 0.45
        if best_move >= 0.050:
            score += 0.25
        if best_move >= self.config.tp_max_move_pct:
            score += 0.12
        if features.downside_magnet:
            score += 0.18
        return _clamp(score)

    def _continuation_risk(self, features: UltraShortFeatures) -> float:
        risk = 0.75
        if features.acceptance_above_high:
            risk += 0.80
        bullish_continuation = features.close > features.prev_close and features.close_position_in_candle >= 0.72
        if bullish_continuation:
            risk += 0.25
        live_peak_extension = features.high >= features.recent_high and features.close_position_in_candle >= 0.68
        if live_peak_extension:
            risk += 0.25
        reacceleration = (
            features.volume_spike >= max(self.config.min_volume_spike, features.recent_volume_spike * 0.92)
            and features.rsi >= features.prev_rsi
            and features.hist >= features.prev_hist
        )
        if reacceleration:
            risk += 0.25
        if features.mtf_trend_5m > 0.0035 and features.mtf_rsi_5m >= 62.0:
            risk += 0.18
        if features.mtf_trend_15m > 0.0025 and features.mtf_rsi_15m >= 60.0:
            risk += 0.18
        if features.upside_risk:
            risk += 0.22
        if features.swept_above:
            risk -= 0.12
        if features.failed_reclaim or features.retest_failed_breakout:
            risk -= 0.18
        if features.rejection_bar:
            risk -= 0.08
        return max(0.0, min(2.5, risk))

    def _scenario_score(self, features: UltraShortFeatures) -> tuple[str, float]:
        scenarios = {
            "sweep_failure_short": self._scenario_sweep_failure(features),
            "blowoff_rejection_short": self._scenario_blowoff_rejection(features),
            "failed_reclaim_short": self._scenario_failed_reclaim(features),
        }
        scenario, score = max(scenarios.items(), key=lambda item: item[1])
        return (scenario, score) if score > 0.0 else ("", 0.0)

    def _scenario_sweep_failure(self, features: UltraShortFeatures) -> float:
        if features.clean_pump_pct < 0.035:
            return 0.0
        if features.peak_age_bars > 2:
            return 0.0
        if not features.swept_above:
            return 0.0
        if features.acceptance_above_high:
            return 0.0
        if not (features.failed_reclaim or features.retest_failed_breakout or features.rejection_bar):
            return 0.0
        score = 0.60
        if features.failed_reclaim:
            score += 0.10
        if features.retest_failed_breakout:
            score += 0.08
        if features.volume_decay_ratio <= 0.82:
            score += 0.08
        if features.rsi < features.prev_rsi or features.hist < features.prev_hist:
            score += 0.08
        if features.downside_magnet:
            score += 0.06
        return _clamp(score)

    def _scenario_blowoff_rejection(self, features: UltraShortFeatures) -> float:
        if features.clean_pump_pct < 0.035:
            return 0.0
        if max(features.volume_spike, features.recent_volume_spike) < 1.80:
            return 0.0
        if features.upper_wick_ratio < 0.28:
            return 0.0
        if features.close_position_in_candle > 0.48:
            return 0.0
        if features.pump_range_position < 0.78:
            return 0.0
        if features.rsi < 60.0:
            return 0.0
        score = 0.54
        if features.volume_decay_ratio <= 0.82:
            score += 0.10
        if features.hist < features.prev_hist:
            score += 0.08
        if features.rsi < features.prev_rsi:
            score += 0.08
        if features.swept_above or features.near_sweep_level:
            score += 0.08
        if features.downside_magnet:
            score += 0.06
        return _clamp(score)

    def _scenario_failed_reclaim(self, features: UltraShortFeatures) -> float:
        if features.clean_pump_pct < 0.030:
            return 0.0
        if features.peak_age_bars > 5:
            return 0.0
        if features.acceptance_above_high:
            return 0.0
        if not (features.failed_reclaim or features.retest_failed_breakout):
            return 0.0
        if not (
            features.volume_decay_ratio <= 0.88
            or features.hist < features.prev_hist
            or features.rsi < features.prev_rsi
        ):
            return 0.0
        score = 0.56
        if features.failed_reclaim:
            score += 0.10
        if features.retest_failed_breakout:
            score += 0.10
        if features.close_position_in_candle <= 0.55:
            score += 0.07
        if features.volume_decay_ratio <= 0.82:
            score += 0.07
        if features.downside_magnet:
            score += 0.06
        return _clamp(score)

    def _build_trade_plan(self, features: UltraShortFeatures) -> _TradePlan:
        entry = features.close
        invalidation = max(features.recent_high, features.high)
        stop_loss = invalidation + features.atr * self.config.stop_buffer_atr
        risk = max(stop_loss - entry, 0.0)
        stop_distance_atr = risk / max(features.atr, 1e-8)
        diagnostics: dict[str, Any] = {
            "entry": entry,
            "invalidation_level": invalidation,
            "stop_loss": stop_loss,
            "risk": risk,
            "stop_distance_atr": stop_distance_atr,
        }
        if entry <= 0.0 or risk <= 0.0:
            return _TradePlan(entry, stop_loss, 0.0, invalidation, 0.0, stop_distance_atr, 0.0, "invalid", diagnostics)

        best_target = 0.0
        best_source = ""
        best_rr = 0.0
        best_move = 0.0
        for source, target, move_pct in self._target_candidates(features):
            reward = entry - target
            rr = reward / risk if risk > 0 else 0.0
            if rr >= self.config.min_rr and (best_target <= 0.0 or target > best_target):
                best_target = target
                best_source = source
                best_rr = rr
                best_move = move_pct

        if best_target <= 0.0:
            fallback = entry * (1.0 - self.config.tp_max_move_pct)
            reward = entry - fallback
            best_target = fallback
            best_source = "fixed_7pct_fallback"
            best_rr = reward / risk if risk > 0 else 0.0
            best_move = self.config.tp_max_move_pct

        diagnostics.update(
            {
                "take_profit": best_target,
                "target_source": best_source,
                "rr": best_rr,
                "target_move_pct": best_move,
                "targets": [
                    {"source": source, "target": target, "move_pct": move_pct}
                    for source, target, move_pct in self._target_candidates(features)
                ],
            }
        )
        return _TradePlan(
            entry=entry,
            stop_loss=stop_loss,
            take_profit=best_target,
            invalidation_level=invalidation,
            rr=best_rr,
            stop_distance_atr=stop_distance_atr,
            target_move_pct=best_move,
            target_source=best_source,
            diagnostics=diagnostics,
        )

    def _reject(
        self,
        *,
        symbol: str,
        timeframe: str,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> UltraShortDecision:
        return UltraShortDecision(
            approved=False,
            scenario="",
            grade="",
            score=0.0,
            entry=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            invalidation_level=0.0,
            rr=0.0,
            continuation_risk=0.0,
            reason=reason,
            setup_signature=self._reject_signature(symbol, timeframe, reason),
            diagnostics=diagnostics or {},
            symbol=symbol,
            timeframe=timeframe,
        )

    def _reject_from_features(
        self,
        features: UltraShortFeatures,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> UltraShortDecision:
        return self._reject(symbol=features.symbol, timeframe=features.timeframe, reason=reason, diagnostics=diagnostics)

    def _target_candidates(self, features: UltraShortFeatures) -> list[tuple[str, float, float]]:
        targets: list[tuple[str, float, float]] = []
        for source, value in (
            ("vwap", features.vwap),
            ("poc", features.poc),
            ("val", features.val),
            ("fixed_3pct", features.close * (1.0 - self.config.tp_min_move_pct)),
            ("fixed_5pct", features.close * 0.95),
            ("fixed_7pct", features.close * (1.0 - self.config.tp_max_move_pct)),
        ):
            if value <= 0.0 or value >= features.close:
                continue
            move_pct = (features.close - value) / features.close
            if move_pct <= 0.0:
                continue
            targets.append((source, value, move_pct))
        targets.sort(key=lambda item: item[1], reverse=True)
        return targets

    @staticmethod
    def _setup_signature(features: UltraShortFeatures, scenario: str, plan: _TradePlan) -> str:
        payload = (
            f"{features.symbol}|{features.timeframe}|{scenario}|"
            f"{plan.entry:.12g}|{plan.invalidation_level:.12g}|{plan.take_profit:.12g}|"
            f"{features.peak_age_bars}|{features.clean_pump_pct:.6f}|{STRATEGY_RUNTIME_VERSION}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _reject_signature(symbol: str, timeframe: str, reason: str) -> str:
        payload = f"{symbol}|{timeframe}|reject|{reason}|{STRATEGY_RUNTIME_VERSION}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _feature_diagnostics(features: UltraShortFeatures) -> dict[str, Any]:
        return {
            "clean_pump_pct": features.clean_pump_pct,
            "volume_spike": features.volume_spike,
            "recent_volume_spike": features.recent_volume_spike,
            "volume_decay_ratio": features.volume_decay_ratio,
            "pump_range_position": features.pump_range_position,
            "entry_distance_from_high_atr": features.entry_distance_from_high_atr,
            "upper_wick_ratio": features.upper_wick_ratio,
            "close_position_in_candle": features.close_position_in_candle,
            "peak_age_bars": features.peak_age_bars,
            "failed_reclaim": features.failed_reclaim,
            "retest_failed_breakout": features.retest_failed_breakout,
            "acceptance_above_high": features.acceptance_above_high,
            "rejection_bar": features.rejection_bar,
            "swept_above": features.swept_above,
            "downside_magnet": features.downside_magnet,
            "upside_risk": features.upside_risk,
        }
