from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_LAYER_ORDER: tuple[str, ...] = (
    "regime_filter",
    "layer1_pump_detection",
    "layer2_weakness_confirmation",
    "layer3_entry_location",
    "layer4_fake_filter",
    "layer5_tp_sl",
)

_REGIME_FILTER_BLOCKERS: tuple[str, ...] = (
    "htf_trend_ok",
    "stretched_from_vwap",
    "volatility_regime_ok",
    "news_veto",
)

_LAYER1_BLOCKERS: tuple[str, ...] = (
    "rsi_high",
    "volume_spike",
    "above_bollinger_upper",
    "above_keltner_upper",
    "clean_pump_pct",
)

_LAYER4_BLOCKERS: tuple[str, ...] = (
    "price_above_vwap",
    "sentiment_euphoric",
    "funding_supports_short",
    "long_short_ratio_extreme",
    "oi_overheated",
)

_QUALITY_BUCKETS: tuple[str, ...] = ("live", "fallback", "unavailable")


class StrategyAuditCollector:
    """Lightweight runtime audit for layered strategy pass/fail behavior."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.evaluated_count = 0
        self.no_signal_count = 0
        self.short_signal_count = 0
        self.long_signal_count = 0

        self.insufficient_history_count = 0

        self.reached_layer_counts = {name: 0 for name in _LAYER_ORDER}
        self.passed_layer_counts = {name: 0 for name in _LAYER_ORDER}
        self.failed_layer_counts: dict[str, int] = {}

        self.regime_source_quality_counts: dict[str, dict[str, int]] = {}
        self.layer4_source_quality_counts: dict[str, dict[str, int]] = {}
        self.regime_source_quality_summary = {name: 0 for name in _QUALITY_BUCKETS}
        self.layer4_source_quality_summary = {name: 0 for name in _QUALITY_BUCKETS}

        self.regime_filter_blocker_counts = {name: 0 for name in _REGIME_FILTER_BLOCKERS}
        self.regime_filter_degraded_only_count = 0
        self.regime_filter_soft_pass_candidate_count = 0
        self.regime_filter_soft_pass_used_count = 0
        self.regime_filter_degraded_mode_count = 0

        self.layer1_blocker_counts = {name: 0 for name in _LAYER1_BLOCKERS}
        self.layer1_soft_pass_candidate_count = 0
        self.layer1_soft_pass_used_count = 0

        self.layer5_fallback_rr_used_count = 0
        self.layer5_vp_based_count = 0
        self.layer5_fail_missing_atr_count = 0
        self.layer5_fail_missing_volume_profile_count = 0

        self.layer4_fake_filter_degraded_mode_count = 0
        self.layer4_blocker_counts = {name: 0 for name in _LAYER4_BLOCKERS}
        self.layer4_fail_due_to_price_structure_count = 0
        self.layer4_fail_due_to_sentiment_count = 0
        self.layer4_fail_due_to_derivatives_context_count = 0
        self.layer4_fail_due_to_degraded_mode_only_count = 0
        self.layer4_hard_fail_count = 0
        self.layer4_soft_fail_count = 0
        self.layer4_degraded_data_fail_count = 0
        self.layer4_soft_pass_candidate_count = 0

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) != 0.0
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("1", "true", "yes", "y", "on"):
                return True
            if text in ("0", "false", "no", "n", "off", ""):
                return False
        return False

    @staticmethod
    def _inc(dct: dict[str, int], key: str) -> None:
        if not key:
            return
        dct[key] = int(dct.get(key, 0)) + 1

    @classmethod
    def _inc_quality(cls, store: dict[str, dict[str, int]], metric: str, quality: Any) -> None:
        if not metric:
            return
        quality_text = str(quality or "unknown").strip().lower() or "unknown"
        bucket = store.setdefault(metric, {})
        cls._inc(bucket, quality_text)

    @staticmethod
    def _missing_condition_set(details: Mapping[str, Any]) -> set[str]:
        raw = details.get("missing_conditions", "")
        if raw is None:
            return set()
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    @staticmethod
    def _quality_bucket(quality: Any) -> str:
        text = str(quality or "").strip().lower()
        if text == "live":
            return "live"
        if text == "fallback" or text.startswith("fallback") or text.startswith("derived") or text.startswith("synthetic"):
            return "fallback"
        return "unavailable"

    def _record_layer_source_quality(
        self,
        details: Mapping[str, Any],
        store: dict[str, dict[str, int]],
        summary_store: dict[str, int],
    ) -> None:
        source_flags = details.get("source_flags", {}) if isinstance(details, Mapping) else {}
        if not isinstance(source_flags, Mapping):
            return

        for metric, quality in source_flags.items():
            metric_text = str(metric)
            if not metric_text.endswith("_quality"):
                continue
            self._inc_quality(store, metric_text, quality)
            bucket = self._quality_bucket(quality)
            summary_store[bucket] = int(summary_store.get(bucket, 0)) + 1

    def _condition_blocker_value(self, details: Mapping[str, Any], passed: bool, condition_name: str, missing_set: set[str]) -> bool:
        explicit_key = f"blocker_{condition_name}"
        if explicit_key in details:
            return self._as_bool(details.get(explicit_key))
        if passed:
            return False
        return condition_name in missing_set

    def _detail_bool_or_default(self, details: Mapping[str, Any], key: str, default: bool) -> bool:
        if key in details:
            return self._as_bool(details.get(key))
        return default

    def record(self, layer_trace: Mapping[str, Any] | None, signal_side: str | None) -> None:
        self.evaluated_count += 1

        trace = layer_trace if isinstance(layer_trace, Mapping) else {}
        layers = trace.get("layers", {}) if isinstance(trace, Mapping) else {}
        if not isinstance(layers, Mapping):
            layers = {}

        for layer_name in _LAYER_ORDER:
            layer_data = layers.get(layer_name)
            if not isinstance(layer_data, Mapping):
                continue
            self.reached_layer_counts[layer_name] += 1
            if self._as_bool(layer_data.get("passed")):
                self.passed_layer_counts[layer_name] += 1

        failed_layer = str(trace.get("failed_layer") or "").strip()
        if failed_layer:
            self._inc(self.failed_layer_counts, failed_layer)

        insufficient_history = failed_layer == "layer0_input"
        if failed_layer and failed_layer in layers and isinstance(layers.get(failed_layer), Mapping):
            failed_data = layers.get(failed_layer, {})
            failed_details = failed_data.get("details", {}) if isinstance(failed_data, Mapping) else {}
            if isinstance(failed_details, Mapping):
                if str(failed_details.get("failed_reason") or "").strip().lower() == "insufficient_history":
                    insufficient_history = True
        if insufficient_history:
            self.insufficient_history_count += 1

        side = str(signal_side or "").upper()
        if side == "SHORT":
            self.short_signal_count += 1
        elif side == "LONG":
            self.long_signal_count += 1
        else:
            self.no_signal_count += 1

        regime_data = layers.get("regime_filter", {}) if isinstance(layers.get("regime_filter"), Mapping) else {}
        regime_details = regime_data.get("details", {}) if isinstance(regime_data, Mapping) else {}
        if isinstance(regime_details, Mapping):
            self._record_layer_source_quality(
                regime_details,
                self.regime_source_quality_counts,
                self.regime_source_quality_summary,
            )

            regime_passed = self._as_bool(regime_details.get("passed", regime_data.get("passed")))
            regime_missing_set = self._missing_condition_set(regime_details)
            regime_degraded = self._as_bool(regime_details.get("degraded_mode"))
            if regime_degraded:
                self.regime_filter_degraded_mode_count += 1

            regime_blockers = {
                blocker: self._condition_blocker_value(regime_details, regime_passed, blocker, regime_missing_set)
                for blocker in _REGIME_FILTER_BLOCKERS
            }
            for blocker, blocked in regime_blockers.items():
                if blocked:
                    self.regime_filter_blocker_counts[blocker] += 1

            degraded_only_default = (not regime_passed) and regime_degraded and not any(regime_blockers.values())
            degraded_only = self._detail_bool_or_default(
                regime_details,
                "fail_due_to_degraded_mode_only",
                degraded_only_default,
            )
            if degraded_only:
                self.regime_filter_degraded_only_count += 1

            soft_candidate_default = False
            if not regime_passed and self._as_bool(regime_details.get("htf_trend_ok")):
                support_count = (
                    int(self._as_bool(regime_details.get("stretched_from_vwap")))
                    + int(self._as_bool(regime_details.get("volatility_regime_ok")))
                    + int(self._as_bool(regime_details.get("news_veto")))
                )
                soft_candidate_default = support_count >= 2
            soft_candidate = self._detail_bool_or_default(
                regime_details,
                "soft_pass_candidate",
                soft_candidate_default,
            )
            if soft_candidate:
                self.regime_filter_soft_pass_candidate_count += 1

            soft_pass_used = self._detail_bool_or_default(
                regime_details,
                "soft_pass_used",
                False,
            )
            if soft_pass_used:
                self.regime_filter_soft_pass_used_count += 1

        layer1_entry = layers.get("layer1_pump_detection")
        if isinstance(layer1_entry, Mapping):
            layer1_data = layer1_entry
            layer1_details_raw = layer1_data.get("details", {})
            layer1_details = layer1_details_raw if isinstance(layer1_details_raw, Mapping) else {}
            layer1_passed = self._as_bool(layer1_details.get("passed", layer1_data.get("passed")))
            layer1_missing_set = self._missing_condition_set(layer1_details)

            layer1_rsi_high = self._as_bool(layer1_details.get("rsi_high"))
            layer1_volume_spike = self._as_bool(layer1_details.get("volume_spike_high"))
            if "volume_spike_high" not in layer1_details:
                layer1_volume_spike = "volume_spike" not in layer1_missing_set

            layer1_above_bb = self._as_bool(layer1_details.get("above_bollinger_upper"))
            layer1_above_kc = self._as_bool(layer1_details.get("above_keltner_upper"))
            layer1_upper_band_breakout = self._as_bool(layer1_details.get("upper_band_breakout"))
            if "upper_band_breakout" not in layer1_details:
                layer1_upper_band_breakout = "upper_band_breakout" not in layer1_missing_set

            layer1_blockers = {
                "rsi_high": (not layer1_passed) and (not layer1_rsi_high),
                "volume_spike": (not layer1_passed) and (not layer1_volume_spike),
                "above_bollinger_upper": (not layer1_passed) and (not layer1_above_bb) and (not layer1_upper_band_breakout),
                "above_keltner_upper": (not layer1_passed) and (not layer1_above_kc) and (not layer1_upper_band_breakout),
                "clean_pump_pct": (not layer1_passed) and (not self._as_bool(layer1_details.get("clean_pump_ok"))),
            }
            for blocker, blocked in layer1_blockers.items():
                if blocked:
                    self.layer1_blocker_counts[blocker] += 1

            layer1_soft_candidate = self._detail_bool_or_default(
                layer1_details,
                "soft_pass_candidate",
                False,
            )
            if layer1_soft_candidate:
                self.layer1_soft_pass_candidate_count += 1

            layer1_soft_pass_used = self._detail_bool_or_default(
                layer1_details,
                "soft_pass_used",
                False,
            )
            if layer1_soft_pass_used:
                self.layer1_soft_pass_used_count += 1

        layer4_entry = layers.get("layer4_fake_filter")
        layer4_data = layer4_entry if isinstance(layer4_entry, Mapping) else {}
        layer4_details = layer4_data.get("details", {}) if isinstance(layer4_data, Mapping) else {}
        if isinstance(layer4_entry, Mapping) and isinstance(layer4_details, Mapping):
            self._record_layer_source_quality(
                layer4_details,
                self.layer4_source_quality_counts,
                self.layer4_source_quality_summary,
            )

            layer4_passed = self._as_bool(layer4_details.get("passed", layer4_data.get("passed")))
            missing_set = self._missing_condition_set(layer4_details)
            degraded_mode = self._as_bool(layer4_details.get("degraded_mode"))
            if degraded_mode:
                self.layer4_fake_filter_degraded_mode_count += 1

            blocker_values = {
                blocker: self._condition_blocker_value(layer4_details, layer4_passed, blocker, missing_set)
                for blocker in _LAYER4_BLOCKERS
            }
            for blocker, blocked in blocker_values.items():
                if blocked:
                    self.layer4_blocker_counts[blocker] += 1

            fail_due_to_price_structure = self._detail_bool_or_default(
                layer4_details,
                "fail_due_to_price_structure",
                (not layer4_passed) and blocker_values["price_above_vwap"],
            )
            fail_due_to_sentiment = self._detail_bool_or_default(
                layer4_details,
                "fail_due_to_sentiment",
                (not layer4_passed) and blocker_values["sentiment_euphoric"],
            )
            fail_due_to_derivatives_context = self._detail_bool_or_default(
                layer4_details,
                "fail_due_to_derivatives_context",
                (not layer4_passed)
                and (
                    blocker_values["funding_supports_short"]
                    or blocker_values["long_short_ratio_extreme"]
                    or blocker_values["oi_overheated"]
                ),
            )
            fail_due_to_degraded_mode_only = self._detail_bool_or_default(
                layer4_details,
                "fail_due_to_degraded_mode_only",
                (not layer4_passed)
                and degraded_mode
                and not (
                    fail_due_to_price_structure
                    or fail_due_to_sentiment
                    or fail_due_to_derivatives_context
                ),
            )

            hard_fail = self._detail_bool_or_default(
                layer4_details,
                "hard_fail",
                (not layer4_passed) and not degraded_mode,
            )
            degraded_data_fail = self._detail_bool_or_default(
                layer4_details,
                "degraded_data_fail",
                (not layer4_passed) and fail_due_to_degraded_mode_only,
            )
            soft_fail = self._detail_bool_or_default(
                layer4_details,
                "soft_fail",
                (not layer4_passed) and not hard_fail and not degraded_data_fail,
            )

            if fail_due_to_price_structure:
                self.layer4_fail_due_to_price_structure_count += 1
            if fail_due_to_sentiment:
                self.layer4_fail_due_to_sentiment_count += 1
            if fail_due_to_derivatives_context:
                self.layer4_fail_due_to_derivatives_context_count += 1
            if fail_due_to_degraded_mode_only:
                self.layer4_fail_due_to_degraded_mode_only_count += 1
            if hard_fail:
                self.layer4_hard_fail_count += 1
            if soft_fail:
                self.layer4_soft_fail_count += 1
            if degraded_data_fail:
                self.layer4_degraded_data_fail_count += 1

            default_soft_pass_candidate = (not layer4_passed) and self._as_bool(layer4_details.get("price_above_vwap")) and any(
                self._as_bool(layer4_details.get(key))
                for key in (
                    "sentiment_euphoric",
                    "funding_supports_short",
                    "long_short_ratio_extreme",
                    "oi_overheated",
                )
            )
            soft_pass_candidate = self._detail_bool_or_default(
                layer4_details,
                "soft_pass_candidate",
                default_soft_pass_candidate,
            )
            if soft_pass_candidate:
                self.layer4_soft_pass_candidate_count += 1

        layer5_data = layers.get("layer5_tp_sl", {}) if isinstance(layers.get("layer5_tp_sl"), Mapping) else {}
        layer5_details = layer5_data.get("details", {}) if isinstance(layer5_data, Mapping) else {}
        if isinstance(layer5_details, Mapping):
            layer5_passed = self._as_bool(layer5_details.get("passed", layer5_data.get("passed")))
            fallback_rr_used = self._as_bool(layer5_details.get("fallback_rr_used"))
            if fallback_rr_used:
                self.layer5_fallback_rr_used_count += 1
            if layer5_passed and not fallback_rr_used:
                self.layer5_vp_based_count += 1

            if not layer5_passed:
                missing_set = self._missing_condition_set(layer5_details)
                atr_missing = (
                    "atr" in missing_set
                    or ("atr_available" in layer5_details and not self._as_bool(layer5_details.get("atr_available")))
                )
                volume_profile_missing = (
                    "volume_profile" in missing_set
                    or (
                        "volume_profile_available" in layer5_details
                        and not self._as_bool(layer5_details.get("volume_profile_available"))
                    )
                )
                if atr_missing:
                    self.layer5_fail_missing_atr_count += 1
                if volume_profile_missing:
                    self.layer5_fail_missing_volume_profile_count += 1

    def snapshot(self) -> dict[str, Any]:
        regime_filter_blockers = dict(sorted(self.regime_filter_blocker_counts.items()))
        layer1_blockers = dict(sorted(self.layer1_blocker_counts.items()))
        layer4_blockers = dict(sorted(self.layer4_blocker_counts.items()))
        failed_layer_counts = dict(sorted(self.failed_layer_counts.items()))

        source_quality_counts = {
            "regime_filter": {
                key: dict(sorted(values.items()))
                for key, values in sorted(self.regime_source_quality_counts.items())
            },
            "layer4_fake_filter": {
                key: dict(sorted(values.items()))
                for key, values in sorted(self.layer4_source_quality_counts.items())
            },
        }

        reached_regime_filter_count = int(self.reached_layer_counts["regime_filter"])
        passed_regime_filter_count = int(self.passed_layer_counts["regime_filter"])
        regime_filter_fail_count = max(reached_regime_filter_count - passed_regime_filter_count, 0)

        return {
            "evaluations_total": int(self.evaluated_count),
            "evaluated_count": int(self.evaluated_count),
            "insufficient_history_count": int(self.insufficient_history_count),
            "reached_regime_filter_count": reached_regime_filter_count,
            "reached_layer1_count": int(self.reached_layer_counts["layer1_pump_detection"]),
            "reached_layer2_count": int(self.reached_layer_counts["layer2_weakness_confirmation"]),
            "reached_layer3_count": int(self.reached_layer_counts["layer3_entry_location"]),
            "reached_layer4_count": int(self.reached_layer_counts["layer4_fake_filter"]),
            "reached_layer5_count": int(self.reached_layer_counts["layer5_tp_sl"]),
            "passed_regime_filter_count": passed_regime_filter_count,
            "passed_layer1_count": int(self.passed_layer_counts["layer1_pump_detection"]),
            "passed_layer2_count": int(self.passed_layer_counts["layer2_weakness_confirmation"]),
            "passed_layer3_count": int(self.passed_layer_counts["layer3_entry_location"]),
            "passed_layer4_count": int(self.passed_layer_counts["layer4_fake_filter"]),
            "passed_layer5_count": int(self.passed_layer_counts["layer5_tp_sl"]),
            "regime_filter_pass_count": passed_regime_filter_count,
            "regime_filter_fail_count": regime_filter_fail_count,
            "layer1_pass_count": int(self.passed_layer_counts["layer1_pump_detection"]),
            "layer1_fail_count": int(failed_layer_counts.get("layer1_pump_detection", 0)),
            "layer2_fail_count": int(failed_layer_counts.get("layer2_weakness_confirmation", 0)),
            "layer3_fail_count": int(failed_layer_counts.get("layer3_entry_location", 0)),
            "layer4_fail_count": int(failed_layer_counts.get("layer4_fake_filter", 0)),
            "layer5_fail_count": int(failed_layer_counts.get("layer5_tp_sl", 0)),
            "no_signal_count": int(self.no_signal_count),
            "short_signal_count": int(self.short_signal_count),
            "long_signal_count": int(self.long_signal_count),
            "failed_layer_counts": failed_layer_counts,
            "regime_filter_blocker_counts": regime_filter_blockers,
            "regime_filter_htf_trend_blocker_count": int(regime_filter_blockers.get("htf_trend_ok", 0)),
            "regime_filter_vwap_stretch_blocker_count": int(regime_filter_blockers.get("stretched_from_vwap", 0)),
            "regime_filter_volatility_blocker_count": int(regime_filter_blockers.get("volatility_regime_ok", 0)),
            "regime_filter_news_blocker_count": int(regime_filter_blockers.get("news_veto", 0)),
            "regime_filter_degraded_only_count": int(self.regime_filter_degraded_only_count),
            "regime_filter_soft_pass_candidate_count": int(self.regime_filter_soft_pass_candidate_count),
            "regime_filter_soft_pass_used_count": int(self.regime_filter_soft_pass_used_count),
            "layer1_blocker_counts": layer1_blockers,
            "layer1_rsi_high_blocker_count": int(layer1_blockers.get("rsi_high", 0)),
            "layer1_volume_spike_blocker_count": int(layer1_blockers.get("volume_spike", 0)),
            "layer1_above_bollinger_upper_blocker_count": int(layer1_blockers.get("above_bollinger_upper", 0)),
            "layer1_above_keltner_upper_blocker_count": int(layer1_blockers.get("above_keltner_upper", 0)),
            "layer1_clean_pump_pct_blocker_count": int(layer1_blockers.get("clean_pump_pct", 0)),
            "layer1_soft_pass_candidate_count": int(self.layer1_soft_pass_candidate_count),
            "layer1_soft_pass_used_count": int(self.layer1_soft_pass_used_count),
            "layer4_sentiment_blocker_count": int(layer4_blockers.get("sentiment_euphoric", 0)),
            "layer4_funding_blocker_count": int(layer4_blockers.get("funding_supports_short", 0)),
            "layer4_lsr_blocker_count": int(layer4_blockers.get("long_short_ratio_extreme", 0)),
            "layer4_oi_blocker_count": int(layer4_blockers.get("oi_overheated", 0)),
            "layer4_price_blocker_count": int(layer4_blockers.get("price_above_vwap", 0)),
            "layer4_degraded_mode_count": int(self.layer4_fake_filter_degraded_mode_count),
            "source_quality_counts": source_quality_counts,
            "source_quality_summary": {
                "regime_filter": dict(self.regime_source_quality_summary),
                "layer4_fake_filter": dict(self.layer4_source_quality_summary),
            },
            "layer4_blocker_counts": layer4_blockers,
            "layer4_fail_type_counts": {
                "fail_due_to_price_structure": int(self.layer4_fail_due_to_price_structure_count),
                "fail_due_to_sentiment": int(self.layer4_fail_due_to_sentiment_count),
                "fail_due_to_derivatives_context": int(self.layer4_fail_due_to_derivatives_context_count),
                "fail_due_to_degraded_mode_only": int(self.layer4_fail_due_to_degraded_mode_only_count),
                "hard_fail": int(self.layer4_hard_fail_count),
                "soft_fail": int(self.layer4_soft_fail_count),
                "degraded_data_fail": int(self.layer4_degraded_data_fail_count),
            },
            "layer4_soft_pass_candidate_count": int(self.layer4_soft_pass_candidate_count),
            "layer5_fallback_rr_used_count": int(self.layer5_fallback_rr_used_count),
            "layer5_vp_based_count": int(self.layer5_vp_based_count),
            "layer5_fail_missing_atr_count": int(self.layer5_fail_missing_atr_count),
            "layer5_fail_missing_volume_profile_count": int(self.layer5_fail_missing_volume_profile_count),
            "regime_filter_degraded_mode_count": int(self.regime_filter_degraded_mode_count),
            "layer4_fake_filter_degraded_mode_count": int(self.layer4_fake_filter_degraded_mode_count),
        }

    def compact_snapshot(self) -> dict[str, Any]:
        snapshot = self.snapshot()

        def _top_nonzero_blocker(source: object) -> tuple[str, int]:
            if not isinstance(source, Mapping) or not source:
                return "", 0
            top_name, top_count = max(
                ((str(k), int(v)) for k, v in source.items()),
                key=lambda item: item[1],
            )
            if top_count <= 0:
                return "", 0
            return top_name, top_count

        failed_layer_counts = snapshot.get("failed_layer_counts", {})
        top_failed_layer = ""
        top_failed_count = 0
        if isinstance(failed_layer_counts, Mapping) and failed_layer_counts:
            top_failed_layer, top_failed_count = max(
                ((str(k), int(v)) for k, v in failed_layer_counts.items()),
                key=lambda item: item[1],
            )

        regime_blocker_counts = snapshot.get("regime_filter_blocker_counts", {})
        top_regime_filter_blocker, top_regime_filter_blocker_count = _top_nonzero_blocker(regime_blocker_counts)

        layer1_blocker_counts = snapshot.get("layer1_blocker_counts", {})
        top_layer1_blocker, top_layer1_blocker_count = _top_nonzero_blocker(layer1_blocker_counts)

        layer4_blocker_counts = snapshot.get("layer4_blocker_counts", {})
        top_layer4_blocker, top_layer4_blocker_count = _top_nonzero_blocker(layer4_blocker_counts)

        evaluations_total = int(snapshot.get("evaluations_total", 0))
        no_signal_count = int(snapshot.get("no_signal_count", 0))
        short_signal_count = int(snapshot.get("short_signal_count", 0))
        no_signal_ratio = (float(no_signal_count) / float(evaluations_total)) if evaluations_total > 0 else 0.0
        short_signal_ratio = (float(short_signal_count) / float(evaluations_total)) if evaluations_total > 0 else 0.0

        return {
            "evaluations_total": evaluations_total,
            "evaluated_count": evaluations_total,
            "insufficient_history_count": snapshot["insufficient_history_count"],
            "regime_filter_pass_count": snapshot["regime_filter_pass_count"],
            "regime_filter_fail_count": snapshot["regime_filter_fail_count"],
            "regime_filter_htf_trend_blocker_count": snapshot["regime_filter_htf_trend_blocker_count"],
            "regime_filter_vwap_stretch_blocker_count": snapshot["regime_filter_vwap_stretch_blocker_count"],
            "regime_filter_volatility_blocker_count": snapshot["regime_filter_volatility_blocker_count"],
            "regime_filter_news_blocker_count": snapshot["regime_filter_news_blocker_count"],
            "regime_filter_degraded_only_count": snapshot["regime_filter_degraded_only_count"],
            "regime_filter_soft_pass_candidate_count": snapshot["regime_filter_soft_pass_candidate_count"],
            "regime_filter_soft_pass_used_count": snapshot["regime_filter_soft_pass_used_count"],
            "layer1_pass_count": snapshot["layer1_pass_count"],
            "layer1_fail_count": snapshot["layer1_fail_count"],
            "layer1_rsi_high_blocker_count": snapshot["layer1_rsi_high_blocker_count"],
            "layer1_volume_spike_blocker_count": snapshot["layer1_volume_spike_blocker_count"],
            "layer1_above_bollinger_upper_blocker_count": snapshot["layer1_above_bollinger_upper_blocker_count"],
            "layer1_above_keltner_upper_blocker_count": snapshot["layer1_above_keltner_upper_blocker_count"],
            "layer1_clean_pump_pct_blocker_count": snapshot["layer1_clean_pump_pct_blocker_count"],
            "layer1_soft_pass_candidate_count": snapshot["layer1_soft_pass_candidate_count"],
            "layer1_soft_pass_used_count": snapshot["layer1_soft_pass_used_count"],
            "layer2_fail_count": snapshot["layer2_fail_count"],
            "layer3_fail_count": snapshot["layer3_fail_count"],
            "layer4_fail_count": snapshot["layer4_fail_count"],
            "layer5_fail_count": snapshot["layer5_fail_count"],
            "no_signal_count": no_signal_count,
            "short_signal_count": short_signal_count,
            "no_signal_ratio": float(no_signal_ratio),
            "short_signal_ratio": float(short_signal_ratio),
            "layer4_sentiment_blocker_count": snapshot["layer4_sentiment_blocker_count"],
            "layer4_funding_blocker_count": snapshot["layer4_funding_blocker_count"],
            "layer4_lsr_blocker_count": snapshot["layer4_lsr_blocker_count"],
            "layer4_oi_blocker_count": snapshot["layer4_oi_blocker_count"],
            "layer4_price_blocker_count": snapshot["layer4_price_blocker_count"],
            "layer4_degraded_mode_count": snapshot["layer4_degraded_mode_count"],
            "layer4_soft_pass_candidate_count": snapshot["layer4_soft_pass_candidate_count"],
            "layer5_fallback_rr_used_count": snapshot["layer5_fallback_rr_used_count"],
            "layer5_vp_based_count": snapshot["layer5_vp_based_count"],
            "layer5_fail_missing_atr_count": snapshot["layer5_fail_missing_atr_count"],
            "layer5_fail_missing_volume_profile_count": snapshot["layer5_fail_missing_volume_profile_count"],
            "regime_source_quality": snapshot["source_quality_summary"]["regime_filter"],
            "layer4_source_quality": snapshot["source_quality_summary"]["layer4_fake_filter"],
            "regime_filter_degraded_mode_count": snapshot["regime_filter_degraded_mode_count"],
            "layer4_fake_filter_degraded_mode_count": snapshot["layer4_fake_filter_degraded_mode_count"],
            "top_failed_layer": top_failed_layer,
            "top_failed_count": int(top_failed_count),
            "top_regime_filter_blocker": top_regime_filter_blocker,
            "top_regime_filter_blocker_count": int(top_regime_filter_blocker_count),
            "top_layer1_blocker": top_layer1_blocker,
            "top_layer1_blocker_count": int(top_layer1_blocker_count),
            "top_layer4_blocker": top_layer4_blocker,
            "top_layer4_blocker_count": int(top_layer4_blocker_count),
        }


