from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from trading.signals.models import SignalCandidate
from trading.signals.scoring import boolish, clamp, layer_details, mapping_get, safe_float


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
    min_score_degraded: float = 0.86
    min_rr: float = 1.35
    min_context_quality: float = 0.55
    min_stop_atr: float = 0.35
    max_stop_atr: float = 2.80
    late_entry_bars: int = 3
    max_chase_distance_atr: float = 0.75
    hard_reject_chase_distance_atr: float = 1.35
    reentry_cooldown_bars: int = 6

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
        )


@dataclass(frozen=True)
class EntryGateDecision:
    approved: bool
    reason: str
    score: float
    penalties: dict[str, float] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    version: str = "entry_gate_v1"

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

        penalties: dict[str, float] = {}
        flags: dict[str, bool] = {
            "degraded_context": degraded_context,
            "fallback_rr_used": boolish(mapping_get(layer5, "fallback_rr_used")),
            "late_entry": pump_bar_offset > self.config.late_entry_bars,
            "chasing_after_peak": atr > 0 and chase_atr > self.config.max_chase_distance_atr,
            "context_quality_low": context_quality < self.config.min_context_quality,
            "continuation_risk": self._continuation_risk(candidate, layer1),
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
        values = [str(value).lower() for value in source_flags.values()]
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

    @staticmethod
    def _continuation_risk(candidate: SignalCandidate, layer1: Mapping[str, Any]) -> bool:
        close = safe_float(candidate.latest_close)
        open_ = safe_float(candidate.latest_open)
        high = safe_float(candidate.latest_high)
        candle_range = max(0.0, high - safe_float(candidate.latest_low))
        near_high = candle_range > 0 and (high - close) / candle_range <= 0.22
        rsi = safe_float(mapping_get(layer1, "rsi"), 0.0)
        return close > open_ and near_high and rsi >= 78.0

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
        }
