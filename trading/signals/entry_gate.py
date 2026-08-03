from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from trading.signals.models import SignalCandidate
from trading.signals.scoring import boolish, clamp, layer_details, mapping_get, safe_float
from trading.signals.versioning import ENTRY_GATE_VERSION


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return safe_float(raw, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class EntryGateConfig:
    enabled: bool = True
    min_score: float = 0.76
    # Generator confidence is capped at 0.70 for degraded inputs and this gate
    # also applies a 0.10 degraded penalty. At 0.86 the weighted score was
    # mathematically unreachable even with perfect structural evidence.
    min_score_degraded: float = 0.78
    min_rr: float = 1.35
    min_context_quality: float = 0.55
    min_stop_atr: float = 0.35
    max_stop_atr: float = 2.80
    late_entry_bars: int = 3
    max_chase_distance_atr: float = 0.75
    hard_reject_chase_distance_atr: float = 1.35
    reentry_cooldown_bars: int = 6
    mtf_guard_enabled: bool = True
    require_mtf_context: bool = False
    mtf_trend_1h_max_short: float = 0.0036
    mtf_trend_15m_max_short: float = 0.0044
    mtf_trend_5m_max_short: float = 0.0060
    mtf_rsi_1h_max_short: float = 64.0
    mtf_rsi_15m_max_short: float = 68.0
    mtf_rsi_5m_max_short: float = 75.0
    live_continuation_guard_enabled: bool = True
    live_continuation_close_position_min: float = 0.58
    live_continuation_mtf5_trend_min: float = 0.0070
    live_continuation_mtf5_rsi_min: float = 70.0
    live_continuation_mtf15_trend_min: float = 0.0042
    live_continuation_mtf15_rsi_min: float = 66.0
    live_continuation_volume_min: float = 0.62
    live_continuation_bb_position_min: float = 0.76
    live_pullback_volume_max: float = 0.86
    live_pullback_mtf5_trend_min: float = 0.0065
    live_pullback_vwap_dist_min: float = 0.0020
    microstructure_guard_enabled: bool = True
    max_microstructure_spread_bps: float = 35.0
    soft_microstructure_spread_bps: float = 18.0
    max_microstructure_slippage_bps: float = 45.0
    soft_microstructure_slippage_bps: float = 28.0
    min_microstructure_depth_ratio: float = 0.65
    soft_microstructure_depth_ratio: float = 1.05
    max_microstructure_bid_imbalance_short: float = 0.68
    min_aggressor_exhaustion: float = 0.18

    @classmethod
    def from_env(cls) -> "EntryGateConfig":
        return cls(
            enabled=_env_bool("ENTRY_GATE_ENABLED", True),
            min_score=_env_float("ENTRY_GATE_MIN_SCORE", cls.min_score),
            min_score_degraded=_env_float("ENTRY_GATE_MIN_SCORE_DEGRADED", cls.min_score_degraded),
            min_rr=_env_float("ENTRY_GATE_MIN_RR", cls.min_rr),
            min_context_quality=_env_float("ENTRY_GATE_MIN_CONTEXT_QUALITY", cls.min_context_quality),
            min_stop_atr=_env_float("ENTRY_GATE_MIN_STOP_ATR", cls.min_stop_atr),
            max_stop_atr=_env_float("ENTRY_GATE_MAX_STOP_ATR", cls.max_stop_atr),
            late_entry_bars=_env_int("ENTRY_GATE_LATE_ENTRY_BARS", cls.late_entry_bars),
            max_chase_distance_atr=_env_float("ENTRY_GATE_MAX_CHASE_DISTANCE_ATR", cls.max_chase_distance_atr),
            hard_reject_chase_distance_atr=_env_float(
                "ENTRY_GATE_HARD_REJECT_CHASE_DISTANCE_ATR",
                cls.hard_reject_chase_distance_atr,
            ),
            reentry_cooldown_bars=_env_int("ENTRY_GATE_REENTRY_COOLDOWN_BARS", cls.reentry_cooldown_bars),
            mtf_guard_enabled=_env_bool("ENTRY_GATE_MTF_GUARD_ENABLED", cls.mtf_guard_enabled),
            # Production defaults to a complete MTF context. The dataclass keeps
            # False for explicit replay/test configurations created in code.
            require_mtf_context=_env_bool("ENTRY_GATE_REQUIRE_MTF_CONTEXT", True),
            mtf_trend_1h_max_short=_env_float("ENTRY_GATE_MTF_TREND_1H_MAX_SHORT", cls.mtf_trend_1h_max_short),
            mtf_trend_15m_max_short=_env_float("ENTRY_GATE_MTF_TREND_15M_MAX_SHORT", cls.mtf_trend_15m_max_short),
            mtf_trend_5m_max_short=_env_float("ENTRY_GATE_MTF_TREND_5M_MAX_SHORT", cls.mtf_trend_5m_max_short),
            mtf_rsi_1h_max_short=_env_float("ENTRY_GATE_MTF_RSI_1H_MAX_SHORT", cls.mtf_rsi_1h_max_short),
            mtf_rsi_15m_max_short=_env_float("ENTRY_GATE_MTF_RSI_15M_MAX_SHORT", cls.mtf_rsi_15m_max_short),
            mtf_rsi_5m_max_short=_env_float("ENTRY_GATE_MTF_RSI_5M_MAX_SHORT", cls.mtf_rsi_5m_max_short),
            live_continuation_guard_enabled=_env_bool(
                "ENTRY_GATE_LIVE_CONTINUATION_GUARD_ENABLED",
                cls.live_continuation_guard_enabled,
            ),
            live_continuation_close_position_min=_env_float(
                "ENTRY_GATE_LIVE_CONTINUATION_CLOSE_POSITION_MIN",
                cls.live_continuation_close_position_min,
            ),
            live_continuation_mtf5_trend_min=_env_float(
                "ENTRY_GATE_LIVE_CONTINUATION_MTF5_TREND_MIN",
                cls.live_continuation_mtf5_trend_min,
            ),
            live_continuation_mtf5_rsi_min=_env_float(
                "ENTRY_GATE_LIVE_CONTINUATION_MTF5_RSI_MIN",
                cls.live_continuation_mtf5_rsi_min,
            ),
            live_continuation_mtf15_trend_min=_env_float(
                "ENTRY_GATE_LIVE_CONTINUATION_MTF15_TREND_MIN",
                cls.live_continuation_mtf15_trend_min,
            ),
            live_continuation_mtf15_rsi_min=_env_float(
                "ENTRY_GATE_LIVE_CONTINUATION_MTF15_RSI_MIN",
                cls.live_continuation_mtf15_rsi_min,
            ),
            live_continuation_volume_min=_env_float(
                "ENTRY_GATE_LIVE_CONTINUATION_VOLUME_MIN",
                cls.live_continuation_volume_min,
            ),
            live_continuation_bb_position_min=_env_float(
                "ENTRY_GATE_LIVE_CONTINUATION_BB_POSITION_MIN",
                cls.live_continuation_bb_position_min,
            ),
            live_pullback_volume_max=_env_float("ENTRY_GATE_LIVE_PULLBACK_VOLUME_MAX", cls.live_pullback_volume_max),
            live_pullback_mtf5_trend_min=_env_float(
                "ENTRY_GATE_LIVE_PULLBACK_MTF5_TREND_MIN",
                cls.live_pullback_mtf5_trend_min,
            ),
            live_pullback_vwap_dist_min=_env_float(
                "ENTRY_GATE_LIVE_PULLBACK_VWAP_DIST_MIN",
                cls.live_pullback_vwap_dist_min,
            ),
            microstructure_guard_enabled=_env_bool(
                "ENTRY_GATE_MICROSTRUCTURE_GUARD_ENABLED",
                cls.microstructure_guard_enabled,
            ),
            max_microstructure_spread_bps=_env_float(
                "ENTRY_GATE_MICROSTRUCTURE_MAX_SPREAD_BPS",
                cls.max_microstructure_spread_bps,
            ),
            soft_microstructure_spread_bps=_env_float(
                "ENTRY_GATE_MICROSTRUCTURE_SOFT_SPREAD_BPS",
                cls.soft_microstructure_spread_bps,
            ),
            max_microstructure_slippage_bps=_env_float(
                "ENTRY_GATE_MICROSTRUCTURE_MAX_SLIPPAGE_BPS",
                cls.max_microstructure_slippage_bps,
            ),
            soft_microstructure_slippage_bps=_env_float(
                "ENTRY_GATE_MICROSTRUCTURE_SOFT_SLIPPAGE_BPS",
                cls.soft_microstructure_slippage_bps,
            ),
            min_microstructure_depth_ratio=_env_float(
                "ENTRY_GATE_MICROSTRUCTURE_MIN_DEPTH_RATIO",
                cls.min_microstructure_depth_ratio,
            ),
            soft_microstructure_depth_ratio=_env_float(
                "ENTRY_GATE_MICROSTRUCTURE_SOFT_DEPTH_RATIO",
                cls.soft_microstructure_depth_ratio,
            ),
            max_microstructure_bid_imbalance_short=_env_float(
                "ENTRY_GATE_MICROSTRUCTURE_MAX_BID_IMBALANCE_SHORT",
                cls.max_microstructure_bid_imbalance_short,
            ),
            min_aggressor_exhaustion=_env_float(
                "ENTRY_GATE_MICROSTRUCTURE_MIN_AGGRESSOR_EXHAUSTION",
                cls.min_aggressor_exhaustion,
            ),
        )


@dataclass(frozen=True)
class EntryGateDecision:
    approved: bool
    reason: str
    score: float
    penalties: dict[str, float] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    version: str = ENTRY_GATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": bool(self.approved),
            "reason": str(self.reason),
            "score": float(self.score),
            "penalties": {str(k): float(v) for k, v in self.penalties.items()},
            "flags": {str(k): bool(v) for k, v in self.flags.items()},
            "diagnostics": self.diagnostics,
            "version": self.version,
        }


class EntryGate:
    """Admission layer that turns raw short-on-pump candidates into tradeable intents."""

    def __init__(self, config: EntryGateConfig | None = None):
        self.config = config or EntryGateConfig.from_env()

    def evaluate(self, candidate: SignalCandidate) -> EntryGateDecision:
        if not self.config.enabled:
            return EntryGateDecision(
                approved=True,
                reason="disabled",
                score=clamp(candidate.confidence),
                diagnostics={"raw_confidence": float(candidate.confidence)},
            )

        if candidate.side != "SHORT":
            return EntryGateDecision(
                approved=True,
                reason="non_short_passthrough",
                score=clamp(candidate.confidence),
                diagnostics={"side": candidate.side, "raw_confidence": float(candidate.confidence)},
            )

        layer1 = layer_details(candidate.details, "layer1")
        layer2 = layer_details(candidate.details, "layer2")
        layer3 = layer_details(candidate.details, "layer3")
        layer4 = layer_details(candidate.details, "layer4")
        layer5 = layer_details(candidate.details, "layer5")

        entry = safe_float(candidate.entry)
        stop = safe_float(candidate.stop_loss)
        target = safe_float(candidate.take_profit)
        stop_distance = abs(stop - entry)
        target_distance = abs(entry - target)
        rr = target_distance / stop_distance if stop_distance > 0 else 0.0
        atr = max(0.0, safe_float(candidate.latest_atr))
        stop_atr = stop_distance / atr if atr > 0 else 0.0
        chase_atr = self._short_chase_distance_atr(candidate, atr)
        pump_bar_offset = safe_float(mapping_get(layer1, "pump_bar_offset"), 0.0)
        context_quality = self._context_quality(candidate.details)
        degraded_context = self._is_degraded_context(candidate.details)
        mtf_context = self._mtf_context(candidate)

        invalid_geometry = entry <= 0 or stop <= entry or target >= entry or target <= 0
        if invalid_geometry:
            return self._reject(
                "invalid_price_geometry",
                candidate,
                diagnostics={
                    "entry": entry,
                    "stop_loss": stop,
                    "take_profit": target,
                    "risk_reward_ratio": rr,
                },
            )

        if self.config.mtf_guard_enabled and self.config.require_mtf_context and mtf_context["missing"]:
            return self._reject(
                "mtf_context_missing",
                candidate,
                diagnostics=mtf_context,
            )

        if rr + 1e-9 < self.config.min_rr:
            return self._reject(
                "risk_reward_below_min",
                candidate,
                diagnostics={
                    "risk_reward_ratio": rr,
                    "min_rr": self.config.min_rr,
                    "entry": entry,
                    "stop_loss": stop,
                    "take_profit": target,
                },
            )

        if atr > 0 and (stop_atr < self.config.min_stop_atr or stop_atr > self.config.max_stop_atr):
            return self._reject(
                "stop_distance_outside_atr_band",
                candidate,
                diagnostics={
                    "stop_atr": stop_atr,
                    "min_stop_atr": self.config.min_stop_atr,
                    "max_stop_atr": self.config.max_stop_atr,
                    "latest_atr": atr,
                },
            )

        if atr > 0 and chase_atr > self.config.hard_reject_chase_distance_atr:
            return self._reject(
                "entry_chasing_after_peak",
                candidate,
                diagnostics={
                    "chase_distance_atr": chase_atr,
                    "hard_reject_chase_distance_atr": self.config.hard_reject_chase_distance_atr,
                    "recent_high": candidate.recent_high,
                    "entry": entry,
                    "latest_atr": atr,
                },
            )

        if self.config.mtf_guard_enabled and mtf_context["hard_continuation"]:
            return self._reject(
                "mtf_continuation_block",
                candidate,
                diagnostics=mtf_context,
            )

        live_block = self._short_live_continuation_block(candidate, layer1, layer2, layer3)
        if live_block["blocked"]:
            return self._reject(
                str(live_block["reason"]),
                candidate,
                diagnostics=live_block,
            )

        microstructure_context = self._microstructure_context(candidate)
        if microstructure_context["hard_risk"]:
            return self._reject(
                "microstructure_execution_risk",
                candidate,
                diagnostics=microstructure_context,
            )

        penalties: dict[str, float] = {}
        flags: dict[str, bool] = {
            "degraded_context": degraded_context,
            "fallback_rr_used": boolish(mapping_get(layer5, "fallback_rr_used")),
            "late_entry": pump_bar_offset > self.config.late_entry_bars,
            "chasing_after_peak": atr > 0 and chase_atr > self.config.max_chase_distance_atr,
            "context_quality_low": context_quality < self.config.min_context_quality,
            "continuation_risk": self._continuation_risk(candidate, layer1),
            "mtf_continuation_risk": bool(self.config.mtf_guard_enabled and mtf_context["caution_continuation"]),
            "microstructure_soft_risk": bool(microstructure_context["soft_risk"]),
        }

        if flags["degraded_context"]:
            penalties["degraded_context"] = 0.10
        if flags["fallback_rr_used"]:
            penalties["fallback_rr_used"] = 0.07
        if flags["late_entry"]:
            penalties["late_entry"] = min(0.14, 0.035 * max(1.0, pump_bar_offset - self.config.late_entry_bars))
        if flags["chasing_after_peak"]:
            over = chase_atr - self.config.max_chase_distance_atr
            penalties["chasing_after_peak"] = min(0.16, 0.10 + max(0.0, over) * 0.04)
        if flags["context_quality_low"]:
            penalties["context_quality_low"] = 0.08
        if flags["continuation_risk"]:
            penalties["continuation_risk"] = 0.06
        if flags["mtf_continuation_risk"]:
            penalties["mtf_continuation_risk"] = 0.08
        if flags["microstructure_soft_risk"]:
            penalties["microstructure_soft_risk"] = safe_float(microstructure_context.get("penalty"), 0.0)

        score = self._score(
            candidate=candidate,
            layer2=layer2,
            layer3=layer3,
            layer5=layer5,
            rr=rr,
            context_quality=context_quality,
            penalties=penalties,
        )
        min_score = self.config.min_score_degraded if degraded_context else self.config.min_score
        reason = "approved"
        if score + 1e-9 < min_score:
            reason = "degraded_score_below_min" if degraded_context else "score_below_min"
            return EntryGateDecision(
                approved=False,
                reason=reason,
                score=score,
                penalties=penalties,
                flags=flags,
                diagnostics=self._diagnostics(
                    candidate=candidate,
                    rr=rr,
                    stop_atr=stop_atr,
                    chase_atr=chase_atr,
                    pump_bar_offset=pump_bar_offset,
                    context_quality=context_quality,
                    min_score=min_score,
                    layer2=layer2,
                    layer3=layer3,
                    layer5=layer5,
                    mtf_context=mtf_context,
                    microstructure_context=microstructure_context,
                ),
            )

        return EntryGateDecision(
            approved=True,
            reason=reason,
            score=score,
            penalties=penalties,
            flags=flags,
            diagnostics=self._diagnostics(
                candidate=candidate,
                rr=rr,
                stop_atr=stop_atr,
                chase_atr=chase_atr,
                pump_bar_offset=pump_bar_offset,
                context_quality=context_quality,
                min_score=min_score,
                layer2=layer2,
                layer3=layer3,
                layer5=layer5,
                mtf_context=mtf_context,
                microstructure_context=microstructure_context,
            ),
        )

    def _reject(self, reason: str, candidate: SignalCandidate, diagnostics: dict[str, Any]) -> EntryGateDecision:
        diagnostics = {
            "raw_confidence": float(candidate.confidence),
            "symbol": candidate.symbol,
            "side": candidate.side,
            **diagnostics,
        }
        return EntryGateDecision(
            approved=False,
            reason=reason,
            score=0.0,
            diagnostics=diagnostics,
        )

    def _score(
        self,
        *,
        candidate: SignalCandidate,
        layer2: Mapping[str, Any],
        layer3: Mapping[str, Any],
        layer5: Mapping[str, Any],
        rr: float,
        context_quality: float,
        penalties: Mapping[str, float],
    ) -> float:
        raw_confidence = clamp(candidate.confidence)
        weakness = clamp(
            max(
                safe_float(mapping_get(layer2, "weakness_strength"), 0.0),
                safe_float(mapping_get(layer2, "confirmation_strength"), 0.0),
                safe_float(mapping_get(layer2, "cvd_bearish_divergence"), 0.0),
                safe_float(mapping_get(layer2, "obv_bearish_divergence"), 0.0),
            ),
            default=raw_confidence,
        )
        entry_location = clamp(
            max(
                safe_float(mapping_get(layer3, "entry_location_strength"), 0.0),
                safe_float(mapping_get(layer3, "fresh_reaction_strength"), 0.0),
                safe_float(mapping_get(layer3, "reclaim_failure_strength"), 0.0),
            ),
            default=raw_confidence,
        )
        rejection = clamp(
            max(
                safe_float(mapping_get(layer2, "price_rejection_near_high"), 0.0),
                safe_float(mapping_get(layer2, "lower_close_after_peak"), 0.0),
                safe_float(mapping_get(layer2, "lower_high_after_peak"), 0.0),
                safe_float(mapping_get(layer3, "fresh_reaction_from_high"), 0.0),
                safe_float(mapping_get(layer3, "failed_reclaim"), 0.0),
            ),
            default=0.0,
        )
        tp_sl_strength = clamp(mapping_get(layer5, "tp_sl_strength"), default=0.72)
        rr_quality = clamp(rr / max(0.1, self.config.min_rr), default=0.0)
        score = (
            0.25 * raw_confidence
            + 0.24 * weakness
            + 0.20 * entry_location
            + 0.12 * rr_quality
            + 0.09 * clamp(context_quality)
            + 0.06 * rejection
            + 0.04 * tp_sl_strength
        )
        score -= sum(max(0.0, float(v)) for v in penalties.values())
        return clamp(score)

    @staticmethod
    def _is_degraded_context(details: Mapping[str, Any] | None) -> bool:
        layer4 = layer_details(details, "layer4")
        if boolish(mapping_get(layer4, "degraded_mode")):
            return True
        source_flags = mapping_get(layer4, "source_flags")
        if isinstance(source_flags, Mapping):
            return any(str(value).lower() == "unavailable" for value in source_flags.values())
        return False

    @staticmethod
    def _context_quality(details: Mapping[str, Any] | None) -> float:
        layer4 = layer_details(details, "layer4")
        source_flags = mapping_get(layer4, "source_flags")
        if not isinstance(source_flags, Mapping) or not source_flags:
            return 0.62
        quality_values = [
            str(value).lower()
            for key, value in source_flags.items()
            if str(key).lower().endswith("_quality")
        ]
        values = quality_values or [
            str(value).lower()
            for value in source_flags.values()
            if str(value).lower() in {"live", "fallback", "unavailable"}
        ]
        live = values.count("live")
        fallback = values.count("fallback")
        unavailable = values.count("unavailable")
        total = max(1, len(values))
        quality = 0.62 + 0.22 * (live / total) - 0.10 * (fallback / total) - 0.26 * (unavailable / total)
        if boolish(mapping_get(layer4, "degraded_mode")):
            quality -= 0.12
        return clamp(quality)

    @staticmethod
    def _short_chase_distance_atr(candidate: SignalCandidate, atr: float) -> float:
        if atr <= 0:
            return 0.0
        recent_high = safe_float(candidate.recent_high)
        entry = safe_float(candidate.entry)
        if recent_high <= 0 or entry <= 0 or recent_high <= entry:
            return 0.0
        return max(0.0, (recent_high - entry) / atr)

    def _short_live_continuation_block(
        self,
        candidate: SignalCandidate,
        layer1: Mapping[str, Any],
        layer2: Mapping[str, Any],
        layer3: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.config.live_continuation_guard_enabled:
            return {"blocked": False, "reason": ""}

        extras = candidate.market_extras if isinstance(candidate.market_extras, Mapping) else {}
        close = safe_float(candidate.latest_close, safe_float(candidate.entry))
        open_ = safe_float(candidate.latest_open, close)
        high = safe_float(candidate.latest_high, close)
        low = safe_float(candidate.latest_low, close)
        atr = max(safe_float(candidate.latest_atr), close * 0.0008, 1e-8)
        candle_range = max(high - low, atr * 0.35, close * 0.0006, 1e-8)
        close_position = clamp((close - low) / candle_range, default=0.5)
        upper_wick_ratio = clamp(max(high - max(open_, close), 0.0) / candle_range)
        body_pct = (close - open_) / max(close, 1e-8)
        atr_pct = atr / max(close, 1e-8)
        recent_high = max(safe_float(candidate.recent_high), high, close)
        near_recent_high_pct = max((recent_high - close) / max(recent_high, 1e-8), 0.0)
        price_near_peak = near_recent_high_pct <= max(0.0075, min(0.015, atr_pct * 1.15))

        rsi = max(
            safe_float(mapping_get(layer1, "rsi"), 0.0),
            safe_float(extras.get("rsi"), 0.0),
        )
        volume_spike = safe_float(extras.get("volume_spike"), safe_float(mapping_get(layer1, "volume_spike"), 1.0))
        if volume_spike <= 0.0:
            volume_spike = safe_float(mapping_get(layer1, "volume_spike"), 1.0)
        vwap_dist = safe_float(extras.get("vwap_dist"), 0.0)
        bb_position = safe_float(extras.get("bb_position"), 0.5)
        adx = safe_float(extras.get("adx"), 0.0)
        ema20 = safe_float(extras.get("ema20"), 0.0)
        mtf_trend_5m = safe_float(extras.get("mtf_trend_5m"), 0.0)
        mtf_rsi_5m = safe_float(extras.get("mtf_rsi_5m"), 50.0)
        mtf_trend_15m = safe_float(extras.get("mtf_trend_15m"), 0.0)
        mtf_rsi_15m = safe_float(extras.get("mtf_rsi_15m"), 50.0)

        reaction_markers = sum(
            int(boolish(value))
            for value in (
                mapping_get(layer2, "price_rejection_near_high"),
                mapping_get(layer2, "lower_close_after_peak"),
                mapping_get(layer2, "lower_high_after_peak"),
                mapping_get(layer3, "fresh_reaction_from_high"),
            )
        )
        failed_breakout = any(
            boolish(value)
            for value in (
                mapping_get(layer2, "failed_reclaim"),
                mapping_get(layer2, "retest_failed_breakout"),
                mapping_get(layer3, "failed_reclaim"),
                mapping_get(layer3, "retest_failed_breakout"),
            )
        )
        downside_displacement = any(
            boolish(value)
            for value in (
                mapping_get(layer3, "downside_displacement_confirmed"),
                mapping_get(layer3, "structural_reversal_ready"),
                mapping_get(layer3, "peak_followthrough_confirmed"),
            )
        )
        candle_rejection = bool(
            close < open_
            and close_position <= 0.48
            and upper_wick_ratio >= 0.18
            and near_recent_high_pct <= max(0.012, atr_pct * 1.50)
        )
        hard_failure_evidence = bool(
            failed_breakout
            or downside_displacement
            or reaction_markers >= 2
            or candle_rejection
        )

        above_fast_value = bool(ema20 > 0.0 and close >= ema20 * (1.0 + max(atr_pct * 0.18, 0.0006)))
        strong_mtf_drive = bool(
            (
                mtf_trend_5m >= self.config.live_continuation_mtf5_trend_min
                and mtf_rsi_5m >= self.config.live_continuation_mtf5_rsi_min
            )
            or (
                mtf_trend_15m >= self.config.live_continuation_mtf15_trend_min
                and mtf_rsi_15m >= self.config.live_continuation_mtf15_rsi_min
            )
            or (
                bb_position >= self.config.live_continuation_bb_position_min
                and vwap_dist >= max(self.config.live_pullback_vwap_dist_min, atr_pct * 0.35)
            )
        )
        soft_mtf_drive = bool(
            mtf_trend_5m >= self.config.live_pullback_mtf5_trend_min
            and (
                mtf_rsi_5m >= 62.0
                or mtf_trend_15m >= self.config.live_continuation_mtf15_trend_min * 0.82
                or vwap_dist >= self.config.live_pullback_vwap_dist_min
                or adx >= 26.0
            )
        )
        live_drive = bool(strong_mtf_drive or soft_mtf_drive or (above_fast_value and adx >= 28.0))

        diagnostics = {
            "blocked": False,
            "reason": "",
            "close_position": float(close_position),
            "upper_wick_ratio": float(upper_wick_ratio),
            "body_pct": float(body_pct),
            "near_recent_high_pct": float(near_recent_high_pct),
            "price_near_peak": bool(price_near_peak),
            "volume_spike": float(volume_spike),
            "vwap_dist": float(vwap_dist),
            "bb_position": float(bb_position),
            "adx": float(adx),
            "mtf_trend_5m": float(mtf_trend_5m),
            "mtf_rsi_5m": float(mtf_rsi_5m),
            "mtf_trend_15m": float(mtf_trend_15m),
            "mtf_rsi_15m": float(mtf_rsi_15m),
            "reaction_markers": int(reaction_markers),
            "failed_breakout": bool(failed_breakout),
            "downside_displacement": bool(downside_displacement),
            "hard_failure_evidence": bool(hard_failure_evidence),
            "strong_mtf_drive": bool(strong_mtf_drive),
            "soft_mtf_drive": bool(soft_mtf_drive),
            "above_fast_value": bool(above_fast_value),
        }

        if (
            live_drive
            and price_near_peak
            and close_position >= self.config.live_continuation_close_position_min
            and body_pct >= -0.0015
            and upper_wick_ratio <= 0.26
            and volume_spike >= self.config.live_continuation_volume_min
            and not hard_failure_evidence
        ):
            return {**diagnostics, "blocked": True, "reason": "live_continuation_without_rejection"}

        if (
            soft_mtf_drive
            and vwap_dist >= self.config.live_pullback_vwap_dist_min
            and volume_spike <= self.config.live_pullback_volume_max
            and close_position <= 0.54
            and near_recent_high_pct <= max(0.020, atr_pct * 2.40)
            and not hard_failure_evidence
        ):
            return {**diagnostics, "blocked": True, "reason": "low_volume_pullback_without_displacement"}

        if (
            live_drive
            and rsi <= 58.0
            and volume_spike <= 0.96
            and close >= open_ * 0.994
            and not hard_failure_evidence
        ):
            return {**diagnostics, "blocked": True, "reason": "weak_top_without_real_failure"}

        return diagnostics

    @staticmethod
    def _continuation_risk(candidate: SignalCandidate, layer1: Mapping[str, Any]) -> bool:
        close = safe_float(candidate.latest_close)
        open_ = safe_float(candidate.latest_open)
        high = safe_float(candidate.latest_high)
        candle_range = max(0.0, high - safe_float(candidate.latest_low))
        near_high = candle_range > 0 and (high - close) / candle_range <= 0.22
        rsi = safe_float(mapping_get(layer1, "rsi"), 0.0)
        extras = candidate.market_extras if isinstance(candidate.market_extras, Mapping) else {}
        mtf_trend_5m = safe_float(extras.get("mtf_trend_5m"), 0.0)
        mtf_rsi_5m = safe_float(extras.get("mtf_rsi_5m"), 50.0)
        bb_position = safe_float(extras.get("bb_position"), 0.5)
        vwap_dist = safe_float(extras.get("vwap_dist"), 0.0)
        live_drive = (
            mtf_trend_5m >= 0.0065
            and mtf_rsi_5m >= 68.0
        ) or (bb_position >= 0.78 and vwap_dist >= 0.002)
        return close > open_ and near_high and (rsi >= 78.0 or live_drive)

    def _microstructure_context(self, candidate: SignalCandidate) -> dict[str, Any]:
        if not self.config.microstructure_guard_enabled:
            return {
                "enabled": False,
                "missing": True,
                "hard_risk": False,
                "soft_risk": False,
                "penalty": 0.0,
                "hard_reasons": [],
                "soft_reasons": [],
            }

        extras = candidate.market_extras if isinstance(candidate.market_extras, Mapping) else {}

        def first_float(*keys: str) -> tuple[float | None, str]:
            for key in keys:
                if key not in extras:
                    continue
                raw = extras.get(key)
                if raw is None:
                    continue
                return safe_float(raw, 0.0), key
            return None, ""

        spread_bps, spread_key = first_float("spread_bps")
        slippage_bps, slippage_key = first_float("expected_slippage_bps", "orderbook_expected_slippage_bps")
        depth_ratio, depth_key = first_float("depth_ratio", "orderbook_depth_ratio")
        bid_ask_imbalance, imbalance_key = first_float("bid_ask_imbalance")
        aggressor_exhaustion, aggressor_key = first_float("aggressor_exhaustion")

        source_keys = [key for key in (spread_key, slippage_key, depth_key, imbalance_key, aggressor_key) if key]
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []
        penalty_parts: dict[str, float] = {}

        def soft_ratio(value: float, soft: float, hard: float) -> float:
            width = max(abs(hard - soft), 1e-9)
            return clamp((value - soft) / width)

        if spread_bps is not None:
            if self.config.max_microstructure_spread_bps > 0.0 and spread_bps > self.config.max_microstructure_spread_bps:
                hard_reasons.append("spread_too_wide")
            elif self.config.soft_microstructure_spread_bps > 0.0 and spread_bps > self.config.soft_microstructure_spread_bps:
                soft_reasons.append("spread_wide")
                penalty_parts["spread_wide"] = min(
                    0.055,
                    0.020
                    + 0.035
                    * soft_ratio(
                        spread_bps,
                        self.config.soft_microstructure_spread_bps,
                        max(self.config.max_microstructure_spread_bps, self.config.soft_microstructure_spread_bps),
                    ),
                )

        if slippage_bps is not None:
            if (
                self.config.max_microstructure_slippage_bps > 0.0
                and slippage_bps > self.config.max_microstructure_slippage_bps
            ):
                hard_reasons.append("slippage_too_high")
            elif (
                self.config.soft_microstructure_slippage_bps > 0.0
                and slippage_bps > self.config.soft_microstructure_slippage_bps
            ):
                soft_reasons.append("slippage_elevated")
                penalty_parts["slippage_elevated"] = min(
                    0.065,
                    0.025
                    + 0.040
                    * soft_ratio(
                        slippage_bps,
                        self.config.soft_microstructure_slippage_bps,
                        max(
                            self.config.max_microstructure_slippage_bps,
                            self.config.soft_microstructure_slippage_bps,
                        ),
                    ),
                )

        if depth_ratio is not None:
            if self.config.min_microstructure_depth_ratio > 0.0 and depth_ratio < self.config.min_microstructure_depth_ratio:
                hard_reasons.append("depth_too_thin")
            elif self.config.soft_microstructure_depth_ratio > 0.0 and depth_ratio < self.config.soft_microstructure_depth_ratio:
                soft_reasons.append("depth_thin")
                width = max(
                    self.config.soft_microstructure_depth_ratio - self.config.min_microstructure_depth_ratio,
                    1e-9,
                )
                penalty_parts["depth_thin"] = min(
                    0.060,
                    0.020 + 0.040 * clamp((self.config.soft_microstructure_depth_ratio - depth_ratio) / width),
                )

        if bid_ask_imbalance is not None and self.config.max_microstructure_bid_imbalance_short > 0.0:
            hard_imbalance = min(0.92, self.config.max_microstructure_bid_imbalance_short + 0.18)
            if bid_ask_imbalance > hard_imbalance:
                hard_reasons.append("bid_imbalance_against_short")
            elif bid_ask_imbalance > self.config.max_microstructure_bid_imbalance_short:
                soft_reasons.append("bid_imbalance_against_short")
                penalty_parts["bid_imbalance_against_short"] = min(
                    0.045,
                    0.020
                    + 0.025
                    * clamp((bid_ask_imbalance - self.config.max_microstructure_bid_imbalance_short) / 0.18),
                )

        if aggressor_exhaustion is not None and 0.0 <= aggressor_exhaustion <= 1.0:
            if aggressor_exhaustion < self.config.min_aggressor_exhaustion:
                soft_reasons.append("aggressor_not_exhausted")
                penalty_parts["aggressor_not_exhausted"] = min(
                    0.035,
                    0.015
                    + 0.020
                    * clamp(
                        (self.config.min_aggressor_exhaustion - aggressor_exhaustion)
                        / max(self.config.min_aggressor_exhaustion, 1e-9)
                    ),
                )

        penalty = min(0.14, sum(max(0.0, float(value)) for value in penalty_parts.values()))
        return {
            "enabled": True,
            "missing": not bool(source_keys),
            "source_keys": source_keys,
            "hard_risk": bool(hard_reasons),
            "soft_risk": bool(soft_reasons) and not bool(hard_reasons),
            "penalty": float(0.0 if hard_reasons else penalty),
            "penalty_parts": penalty_parts,
            "hard_reasons": hard_reasons,
            "soft_reasons": soft_reasons,
            "spread_bps": spread_bps,
            "expected_slippage_bps": slippage_bps,
            "depth_ratio": depth_ratio,
            "bid_ask_imbalance": bid_ask_imbalance,
            "aggressor_exhaustion": aggressor_exhaustion,
            "max_spread_bps": float(self.config.max_microstructure_spread_bps),
            "max_slippage_bps": float(self.config.max_microstructure_slippage_bps),
            "min_depth_ratio": float(self.config.min_microstructure_depth_ratio),
        }

    @staticmethod
    def _diagnostics(
        *,
        candidate: SignalCandidate,
        rr: float,
        stop_atr: float,
        chase_atr: float,
        pump_bar_offset: float,
        context_quality: float,
        min_score: float,
        layer2: Mapping[str, Any],
        layer3: Mapping[str, Any],
        layer5: Mapping[str, Any],
        mtf_context: Mapping[str, Any],
        microstructure_context: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "symbol": candidate.symbol,
            "side": candidate.side,
            "raw_confidence": float(candidate.confidence),
            "risk_reward_ratio": float(rr),
            "stop_atr": float(stop_atr),
            "chase_distance_atr": float(chase_atr),
            "pump_bar_offset": float(pump_bar_offset),
            "context_quality": float(context_quality),
            "min_score_used": float(min_score),
            "weakness_strength": safe_float(mapping_get(layer2, "weakness_strength"), 0.0),
            "entry_location_strength": safe_float(mapping_get(layer3, "entry_location_strength"), 0.0),
            "tp_sl_strength": safe_float(mapping_get(layer5, "tp_sl_strength"), 0.0),
            "entry": float(candidate.entry),
            "stop_loss": float(candidate.stop_loss),
            "take_profit": float(candidate.take_profit),
            "mtf_context": dict(mtf_context),
            "microstructure_context": dict(microstructure_context),
        }

    def _mtf_context(self, candidate: SignalCandidate) -> dict[str, Any]:
        extras = candidate.market_extras if isinstance(candidate.market_extras, Mapping) else {}
        trend_1h = safe_float(extras.get("mtf_trend_1h"), 0.0)
        trend_15m = safe_float(extras.get("mtf_trend_15m"), 0.0)
        trend_5m = safe_float(extras.get("mtf_trend_5m"), 0.0)
        rsi_1h = safe_float(extras.get("mtf_rsi_1h"), 50.0)
        rsi_15m = safe_float(extras.get("mtf_rsi_15m"), 50.0)
        rsi_5m = safe_float(extras.get("mtf_rsi_5m"), 50.0)
        observed_keys = {
            "mtf_trend_1h",
            "mtf_trend_15m",
            "mtf_trend_5m",
            "mtf_rsi_1h",
            "mtf_rsi_15m",
            "mtf_rsi_5m",
        }
        ready_keys = ("mtf_ready_5m", "mtf_ready_15m", "mtf_ready_1h")
        if any(key in extras for key in ready_keys):
            ready = {key: boolish(extras.get(key)) for key in ready_keys}
            missing = not all(ready.values())
        else:
            # Compatibility for replay fixtures recorded before readiness flags existed.
            ready = {key: key.replace("ready_", "rsi_") in extras for key in ready_keys}
            missing = not any(key in extras for key in observed_keys)
        hard_1h = trend_1h >= self.config.mtf_trend_1h_max_short and rsi_1h >= self.config.mtf_rsi_1h_max_short
        hard_15m = trend_15m >= self.config.mtf_trend_15m_max_short and rsi_15m >= self.config.mtf_rsi_15m_max_short
        hard_5m = trend_5m >= self.config.mtf_trend_5m_max_short and rsi_5m >= self.config.mtf_rsi_5m_max_short
        caution_1h = trend_1h >= self.config.mtf_trend_1h_max_short * 0.78 and rsi_1h >= self.config.mtf_rsi_1h_max_short - 2.0
        caution_15m = trend_15m >= self.config.mtf_trend_15m_max_short * 0.82 and rsi_15m >= self.config.mtf_rsi_15m_max_short - 2.0
        caution_5m = trend_5m >= self.config.mtf_trend_5m_max_short * 0.88 and rsi_5m >= self.config.mtf_rsi_5m_max_short - 2.0
        return {
            "missing": bool(missing),
            "hard_continuation": bool(not missing and (hard_1h or hard_15m or hard_5m)),
            "caution_continuation": bool(not missing and (caution_1h or caution_15m or caution_5m)),
            "hard_1h": bool(hard_1h),
            "hard_15m": bool(hard_15m),
            "hard_5m": bool(hard_5m),
            "mtf_trend_1h": float(trend_1h),
            "mtf_trend_15m": float(trend_15m),
            "mtf_trend_5m": float(trend_5m),
            "mtf_rsi_1h": float(rsi_1h),
            "mtf_rsi_15m": float(rsi_15m),
            "mtf_rsi_5m": float(rsi_5m),
            "ready_5m": bool(ready["mtf_ready_5m"]),
            "ready_15m": bool(ready["mtf_ready_15m"]),
            "ready_1h": bool(ready["mtf_ready_1h"]),
            "mtf_guard_enabled": bool(self.config.mtf_guard_enabled),
            "require_mtf_context": bool(self.config.require_mtf_context),
        }
