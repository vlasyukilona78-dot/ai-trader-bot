from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import replace

from core.market_regime import detect_market_regime
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
from core.feature_engineering import sanitize_feature_frame
from core.volume_profile import compute_volume_profile
from trading.exchange.schemas import PositionSide
from trading.portfolio.positions import first_effective_position_for_symbol
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.strategy_audit import StrategyAuditCollector
from trading.signals.strategy_interface import StrategyContext, StrategyInterface
from trading.state.models import TradeState


class LayeredPumpStrategy(StrategyInterface):
    """Adapter around migrated layered strategy that returns intents only."""

    def __init__(self, config: SignalConfig | None = None, audit_collector: StrategyAuditCollector | None = None):
        self._allow_long_entries = str(os.getenv("ENABLE_LONG_SIGNALS", "0")).strip().lower() in {"1", "true", "yes", "on"}
        env_volume_threshold = max(1.0, float(os.getenv("VOLUME_THRESHOLD", "2.0")))
        runtime_config = config or SignalConfig(
            regime_volatility_threshold_override=0.0006,
            regime_volatility_dynamic_floor_mult=0.60,
            regime_volatility_baseline_lookback=96,
            regime_soft_pass_enabled=True,
            layer1_pump_lookback_bars=12,
            layer1_clean_pump_lookback_bars=48,
            layer1_clean_pump_min_pct=0.0400,
            early_watch_clean_pump_min_pct=0.0280,
            early_watch_volume_spike_min=max(0.06, env_volume_threshold * 0.06),
            early_watch_rsi_min=45.5,
            early_watch_quality_min=2.20,
            layer1_soft_pass_enabled=True,
            rsi_high=62.0,
            volume_spike_threshold=max(1.35, env_volume_threshold * 0.70),
            layer1_rsi_soft=47.5,
            layer1_volume_spike_soft=max(0.75, env_volume_threshold * 0.40),
            layer1_upper_band_proximity_pct=0.0018,
            weakness_lookback=3,
            entry_tolerance_pct=0.0085,
            msb_lookback=14,
            msb_recent_bars=11,
            msb_break_buffer_pct=0.0002,
            allow_long_entries=self._allow_long_entries,
        )
        self._generator = SignalGenerator(runtime_config)
        self._audit = audit_collector or StrategyAuditCollector()

    def _trace_meta(self) -> dict:
        trace = self._generator.last_diagnostics if isinstance(self._generator.last_diagnostics, dict) else {}
        failed_layer = str(trace.get("failed_layer") or "") if trace else ""
        layers = trace.get("layers", {}) if isinstance(trace, dict) else {}
        regime_source = (
            layers.get("regime_filter", {})
            .get("details", {})
            .get("source_flags", {})
            if isinstance(layers, dict)
            else {}
        )
        layer4_source = (
            layers.get("layer4_fake_filter", {})
            .get("details", {})
            .get("source_flags", {})
            if isinstance(layers, dict)
            else {}
        )
        return {
            "layer_trace": trace,
            "layer_failed": failed_layer,
            "regime_diagnostics": self._latest_regime_diagnostics(),
            "source_quality": {
                "regime_filter": regime_source,
                "layer4_fake_filter": layer4_source,
            },
        }

    @staticmethod
    def _coalesce(*values):
        for value in values:
            if value is not None:
                return value
        return None

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _regime_condition_state(value: object) -> bool | None:
        if value is None:
            return None
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
        return None

    def audit_snapshot(self) -> dict:
        return self._audit.snapshot()

    def audit_compact_snapshot(self) -> dict:
        return self._audit.compact_snapshot()

    def _latest_regime_details(self) -> dict[str, object]:
        trace = self._generator.last_diagnostics if isinstance(self._generator.last_diagnostics, dict) else {}
        layers = trace.get("layers", {}) if isinstance(trace, dict) else {}
        regime_details = (
            layers.get("regime_filter", {})
            .get("details", {})
            if isinstance(layers, dict)
            else {}
        )
        return regime_details if isinstance(regime_details, dict) else {}

    def _latest_layer1_details(self) -> dict[str, object]:
        trace = self._generator.last_diagnostics if isinstance(self._generator.last_diagnostics, dict) else {}
        layers = trace.get("layers", {}) if isinstance(trace, dict) else {}
        layer1_details = (
            layers.get("layer1_pump_detection", {})
            .get("details", {})
            if isinstance(layers, dict)
            else {}
        )
        return layer1_details if isinstance(layer1_details, dict) else {}

    def _latest_layer2_details(self) -> dict[str, object]:
        trace = self._generator.last_diagnostics if isinstance(self._generator.last_diagnostics, dict) else {}
        layers = trace.get("layers", {}) if isinstance(trace, dict) else {}
        layer2_details = (
            layers.get("layer2_weakness_confirmation", {})
            .get("details", {})
            if isinstance(layers, dict)
            else {}
        )
        return layer2_details if isinstance(layer2_details, dict) else {}

    def _latest_regime_diagnostics(self) -> dict[str, object]:
        regime_details = self._latest_regime_details()

        out: dict[str, object] = {
            "htf_trend_metric_used": None,
            "htf_trend_threshold_used": None,
            "htf_trend_direction_context": "",
            "mtf_trend_5m_used": None,
            "mtf_trend_15m_used": None,
            "mtf_rsi_5m_used": None,
            "mtf_rsi_15m_used": None,
            "mtf_hard_filter_enabled": 0.0,
            "vwap_distance_metric_used": None,
            "vwap_stretch_threshold_used": None,
            "atr_norm": None,
            "volatility_threshold_used": None,
            "failed_reason": "",
            "missing_conditions": "",
            "degraded_mode": 0.0,
            "fail_due_to_degraded_mode_only": 0.0,
            "soft_pass_candidate": 0.0,
            "soft_pass_used": 0.0,
            "soft_pass_reason": "",
            "source_flags": {},
            "regime_filter_subconditions_state": {},
        }
        for key in (
            "htf_trend_metric_used",
            "htf_trend_threshold_used",
            "mtf_trend_5m_used",
            "mtf_trend_15m_used",
            "mtf_rsi_5m_used",
            "mtf_rsi_15m_used",
            "mtf_hard_filter_enabled",
            "vwap_distance_metric_used",
            "vwap_stretch_threshold_used",
            "atr_norm",
            "volatility_threshold_used",
            "fail_due_to_degraded_mode_only",
            "soft_pass_candidate",
            "soft_pass_used",
        ):
            if key not in regime_details:
                continue
            value = regime_details.get(key)
            try:
                out[key] = float(value) if value is not None else None
            except (TypeError, ValueError):
                out[key] = None

        direction = regime_details.get("htf_trend_direction_context")
        out["htf_trend_direction_context"] = str(direction) if direction is not None else ""
        out["failed_reason"] = str(regime_details.get("failed_reason") or "")
        out["missing_conditions"] = str(regime_details.get("missing_conditions") or "")
        out["soft_pass_reason"] = str(regime_details.get("soft_pass_reason") or "")
        try:
            out["degraded_mode"] = float(regime_details.get("degraded_mode", 0.0) or 0.0)
        except (TypeError, ValueError):
            out["degraded_mode"] = 0.0

        source_flags = regime_details.get("source_flags", {})
        out["source_flags"] = dict(source_flags) if isinstance(source_flags, dict) else {}

        subconditions: dict[str, bool] = {}
        for key in ("htf_trend_ok", "mtf_short_context_ok", "stretched_from_vwap", "volatility_regime_ok", "news_veto"):
            state = self._regime_condition_state(regime_details.get(key))
            if state is not None:
                subconditions[key] = state
        out["regime_filter_subconditions_state"] = subconditions
        return out

    def _latest_layer1_diagnostics(self) -> dict[str, object]:
        layer1_details = self._latest_layer1_details()

        out: dict[str, object] = {
            "rsi": None,
            "rsi_high_threshold_used": None,
            "volume_spike": None,
            "volume_spike_high": 0.0,
            "volume_spike_threshold_used": None,
            "close_metric_used": None,
            "bollinger_upper_metric_used": None,
            "keltner_upper_metric_used": None,
            "above_bollinger_upper": 0.0,
            "above_keltner_upper": 0.0,
            "upper_band_breakout": 0.0,
            "pump_context_strength": 0.0,
            "clean_pump_pct": 0.0,
            "clean_pump_min_pct_used": 0.0,
            "clean_pump_ok": 0.0,
            "failed_reason": "",
            "missing_conditions": "",
            "soft_pass_candidate": 0.0,
            "soft_pass_used": 0.0,
            "soft_pass_reason": "",
            "pump_bar_offset": None,
            "layer1_subconditions_state": {},
        }
        for key in (
            "rsi",
            "rsi_high_threshold_used",
            "volume_spike",
            "volume_spike_high",
            "volume_spike_threshold_used",
            "close_metric_used",
            "bollinger_upper_metric_used",
            "keltner_upper_metric_used",
            "above_bollinger_upper",
            "above_keltner_upper",
            "upper_band_breakout",
            "pump_context_strength",
            "clean_pump_pct",
            "clean_pump_min_pct_used",
            "clean_pump_ok",
            "soft_pass_candidate",
            "soft_pass_used",
            "pump_bar_offset",
        ):
            if key not in layer1_details:
                continue
            value = layer1_details.get(key)
            try:
                out[key] = float(value) if value is not None else None
            except (TypeError, ValueError):
                out[key] = None

        out["failed_reason"] = str(layer1_details.get("failed_reason") or "")
        out["missing_conditions"] = str(layer1_details.get("missing_conditions") or "")
        out["soft_pass_reason"] = str(layer1_details.get("soft_pass_reason") or "")

        subconditions: dict[str, bool] = {}
        raw_subconditions = layer1_details.get("layer1_subconditions_state")
        if isinstance(raw_subconditions, dict):
            for key in (
                "rsi_high",
                "volume_spike_high",
                "upper_band_breakout",
                "above_bollinger_upper",
                "above_keltner_upper",
                "clean_pump_ok",
            ):
                state = self._regime_condition_state(raw_subconditions.get(key))
                if state is not None:
                    subconditions[key] = state
        else:
            for key in (
                "rsi_high",
                "volume_spike_high",
                "upper_band_breakout",
                "above_bollinger_upper",
                "above_keltner_upper",
                "clean_pump_ok",
            ):
                state = self._regime_condition_state(layer1_details.get(key))
                if state is not None:
                    subconditions[key] = state
        out["layer1_subconditions_state"] = subconditions
        return out

    def _latest_layer2_diagnostics(self) -> dict[str, object]:
        layer2_details = self._latest_layer2_details()

        out: dict[str, object] = {
            "price_up_or_near_high": 0.0,
            "price_up": 0.0,
            "near_high_context": 0.0,
            "obv_bearish_divergence": 0.0,
            "cvd_bearish_divergence": 0.0,
            "close_last_used": None,
            "close_ref_used": None,
            "obv_last_used": None,
            "obv_ref_used": None,
            "cvd_last_used": None,
            "cvd_ref_used": None,
            "weakness_lookback_used": None,
            "weakness_strength": 0.0,
            "failed_reason": "",
            "missing_conditions": "",
            "layer2_subconditions_state": {},
        }
        for key in (
            "price_up_or_near_high",
            "price_up",
            "near_high_context",
            "obv_bearish_divergence",
            "cvd_bearish_divergence",
            "close_last_used",
            "close_ref_used",
            "obv_last_used",
            "obv_ref_used",
            "cvd_last_used",
            "cvd_ref_used",
            "weakness_lookback_used",
            "weakness_strength",
        ):
            if key not in layer2_details:
                continue
            value = layer2_details.get(key)
            try:
                out[key] = float(value) if value is not None else None
            except (TypeError, ValueError):
                out[key] = None

        out["failed_reason"] = str(layer2_details.get("failed_reason") or "")
        out["missing_conditions"] = str(layer2_details.get("missing_conditions") or "")

        subconditions: dict[str, bool] = {}
        raw_subconditions = layer2_details.get("layer2_subconditions_state")
        if isinstance(raw_subconditions, dict):
            for key in (
                "price_up_or_near_high",
                "price_up",
                "near_high_context",
                "obv_bearish_divergence",
                "cvd_bearish_divergence",
            ):
                state = self._regime_condition_state(raw_subconditions.get(key))
                if state is not None:
                    subconditions[key] = state
        out["layer2_subconditions_state"] = subconditions
        return out

    def audit_observation_snapshot(self) -> dict:
        snapshot = self._audit.snapshot()
        compact = self._audit.compact_snapshot()
        regime_diag = self._latest_regime_diagnostics()
        layer1_diag = self._latest_layer1_diagnostics()
        layer2_diag = self._latest_layer2_diagnostics()
        return {
            "strategy_audit_compact": compact,
            "strategy_audit_regime_filter": {
                "regime_filter_pass_count": int(snapshot.get("regime_filter_pass_count", 0)),
                "regime_filter_fail_count": int(snapshot.get("regime_filter_fail_count", 0)),
                "regime_filter_htf_trend_blocker_count": int(snapshot.get("regime_filter_htf_trend_blocker_count", 0)),
                "regime_filter_vwap_stretch_blocker_count": int(snapshot.get("regime_filter_vwap_stretch_blocker_count", 0)),
                "regime_filter_volatility_blocker_count": int(snapshot.get("regime_filter_volatility_blocker_count", 0)),
                "regime_filter_news_blocker_count": int(snapshot.get("regime_filter_news_blocker_count", 0)),
                "regime_filter_degraded_mode_count": int(snapshot.get("regime_filter_degraded_mode_count", 0)),
                "regime_filter_degraded_only_count": int(snapshot.get("regime_filter_degraded_only_count", 0)),
                "regime_filter_soft_pass_candidate_count": int(snapshot.get("regime_filter_soft_pass_candidate_count", 0)),
                "regime_filter_soft_pass_used_count": int(snapshot.get("regime_filter_soft_pass_used_count", 0)),
                "top_regime_filter_blocker": str(compact.get("top_regime_filter_blocker", "")),
                "top_regime_filter_blocker_count": int(compact.get("top_regime_filter_blocker_count", 0)),
                **regime_diag,
            },
            "strategy_audit_regime_diagnostics": regime_diag,
            "strategy_audit_layer1": {
                "layer1_pass_count": int(snapshot.get("layer1_pass_count", 0)),
                "layer1_fail_count": int(snapshot.get("layer1_fail_count", 0)),
                "layer1_rsi_high_blocker_count": int(snapshot.get("layer1_rsi_high_blocker_count", 0)),
                "layer1_volume_spike_blocker_count": int(snapshot.get("layer1_volume_spike_blocker_count", 0)),
                "layer1_above_bollinger_upper_blocker_count": int(snapshot.get("layer1_above_bollinger_upper_blocker_count", 0)),
                "layer1_above_keltner_upper_blocker_count": int(snapshot.get("layer1_above_keltner_upper_blocker_count", 0)),
                "layer1_clean_pump_pct_blocker_count": int(snapshot.get("layer1_clean_pump_pct_blocker_count", 0)),
                "layer1_soft_pass_candidate_count": int(snapshot.get("layer1_soft_pass_candidate_count", 0)),
                "layer1_soft_pass_used_count": int(snapshot.get("layer1_soft_pass_used_count", 0)),
                "top_layer1_blocker": str(compact.get("top_layer1_blocker", "")),
                "top_layer1_blocker_count": int(compact.get("top_layer1_blocker_count", 0)),
                **layer1_diag,
            },
            "strategy_audit_layer1_diagnostics": layer1_diag,
            "strategy_audit_layer2": {
                "reached_layer2_count": int(snapshot.get("reached_layer2_count", 0)),
                "passed_layer2_count": int(snapshot.get("passed_layer2_count", 0)),
                "layer2_fail_count": int(snapshot.get("layer2_fail_count", 0)),
                **layer2_diag,
            },
            "strategy_audit_layer2_diagnostics": layer2_diag,
            "strategy_audit_layer4": {
                "layer4_fail_count": int(snapshot.get("layer4_fail_count", 0)),
                "layer4_sentiment_blocker_count": int(snapshot.get("layer4_sentiment_blocker_count", 0)),
                "layer4_funding_blocker_count": int(snapshot.get("layer4_funding_blocker_count", 0)),
                "layer4_lsr_blocker_count": int(snapshot.get("layer4_lsr_blocker_count", 0)),
                "layer4_oi_blocker_count": int(snapshot.get("layer4_oi_blocker_count", 0)),
                "layer4_price_blocker_count": int(snapshot.get("layer4_price_blocker_count", 0)),
                "layer4_degraded_mode_count": int(snapshot.get("layer4_degraded_mode_count", 0)),
                "layer4_soft_pass_candidate_count": int(snapshot.get("layer4_soft_pass_candidate_count", 0)),
            },
            "strategy_audit_source_quality": snapshot.get("source_quality_summary", {}),
        }

    def reset_audit(self) -> None:
        self._audit.reset()

    def clone_for_parallel(self) -> "LayeredPumpStrategy":
        return LayeredPumpStrategy(config=replace(self._generator.config))

    def record_external_intent(self, intent: StrategyIntent) -> None:
        metadata = intent.metadata if isinstance(intent.metadata, Mapping) else {}
        trace = metadata.get("layer_trace", {})
        if not isinstance(trace, Mapping):
            trace = {}
        signal_side: str | None = None
        if intent.action == IntentAction.SHORT_ENTRY:
            signal_side = "SHORT"
        elif intent.action == IntentAction.LONG_ENTRY:
            signal_side = "LONG"
        self._audit.record(trace, signal_side=signal_side)

    def _managed_short_exit_intent(
        self,
        *,
        context: StrategyContext,
        enriched,
        volume_profile,
        trace_meta: dict,
    ) -> StrategyIntent | None:
        if context.synced_state != TradeState.SHORT:
            return None

        position = first_effective_position_for_symbol(context.exchange.positions, context.symbol)
        if position is None or position.side != PositionSide.SHORT:
            return None
        if enriched is None or getattr(enriched, "empty", True) or len(enriched) < 3:
            return None

        last = enriched.iloc[-1]
        prev = enriched.iloc[-2]
        close = self._safe_float(last.get("close"), context.mark_price)
        prev_close = self._safe_float(prev.get("close"), close)
        entry_price = self._safe_float(getattr(position, "entry_price", 0.0), 0.0)
        if close <= 0.0 or entry_price <= 0.0:
            return None

        atr = max(self._safe_float(last.get("atr"), close * 0.01), close * 0.001, 1e-8)
        ema20 = self._safe_float(last.get("ema20"), close)
        vwap = self._safe_float(last.get("vwap"), close)
        rsi = self._safe_float(last.get("rsi"), 50.0)
        prev_rsi = self._safe_float(prev.get("rsi"), rsi)
        hist = self._safe_float(last.get("hist"), 0.0)
        prev_hist = self._safe_float(prev.get("hist"), hist)
        atr_pct = atr / max(close, 1e-9)

        recent_support = 0.0
        if "low" in enriched.columns:
            try:
                recent_support = self._safe_float(enriched.tail(min(len(enriched), 24))["low"].min(), 0.0)
            except Exception:
                recent_support = 0.0

        current_stop = self._safe_float(getattr(position, "stop_loss", None), entry_price + atr)
        risk_distance = max(current_stop - entry_price, atr * 0.85, entry_price * 0.0012)
        reward_distance = max(entry_price - close, 0.0)
        reward_r = reward_distance / max(risk_distance, 1e-9)

        holding_minutes = 0.0
        if context.synced_state_updated_at:
            holding_minutes = max(0.0, (time.time() - float(context.synced_state_updated_at)) / 60.0)

        timeframe_minutes = 1.0
        timeframe_text = str(context.timeframe or "1").strip().lower()
        try:
            if timeframe_text.endswith("h"):
                timeframe_minutes = max(1.0, float(timeframe_text[:-1]) * 60.0)
            elif timeframe_text.endswith("m"):
                timeframe_minutes = max(1.0, float(timeframe_text[:-1]))
            else:
                timeframe_minutes = max(1.0, float(timeframe_text))
        except (TypeError, ValueError):
            timeframe_minutes = 1.0

        bars_held = int(max(4.0, min(float(len(enriched)), holding_minutes / max(timeframe_minutes, 1.0) + 2.0)))
        trade_window = enriched.tail(bars_held)
        best_close_low = self._safe_float(trade_window["close"].min(), close)
        if "low" in trade_window.columns:
            try:
                best_wick_low = self._safe_float(trade_window["low"].min(), best_close_low)
            except Exception:
                best_wick_low = best_close_low
            best_low = max(best_wick_low, best_close_low - atr * 0.20)
        else:
            best_low = best_close_low

        vp_poc = self._safe_float(getattr(volume_profile, "poc", 0.0), 0.0) if volume_profile is not None else 0.0
        vp_val = self._safe_float(getattr(volume_profile, "val", 0.0), 0.0) if volume_profile is not None else 0.0
        near_poc = bool(vp_poc > 0.0 and close <= vp_poc * (1.0 + max(atr_pct * 0.85, 0.0032)))
        near_val = bool(vp_val > 0.0 and close <= vp_val * (1.0 + max(atr_pct * 1.05, 0.0048)))
        near_recent_support = bool(
            recent_support > 0.0 and close <= recent_support * (1.0 + max(atr_pct * 0.95, 0.0042))
        )
        target_zone_touched = bool(near_poc or near_val or near_recent_support)

        best_reward_distance = max(entry_price - best_low, 0.0)
        best_reward_r = best_reward_distance / max(risk_distance, 1e-9)
        reward_retention = reward_r / max(best_reward_r, 1e-9) if best_reward_r > 1e-9 else 1.0

        close_above_ema20 = close > ema20 * (1.0 + max(atr_pct * 0.12, 0.0008))
        close_above_vwap = close > vwap * (1.0 + max(atr_pct * 0.12, 0.0008))
        bounce_strength = 0
        bounce_strength += int(close > prev_close)
        bounce_strength += int(rsi >= prev_rsi + 0.8)
        bounce_strength += int(hist > prev_hist)
        bounce_strength += int(close_above_ema20)
        bounce_strength += int(close_above_vwap)

        rsi_turning_up = rsi >= prev_rsi + 0.6
        hist_turning_up = hist > prev_hist and hist > -0.02
        rebound_confirmed = bounce_strength >= 2 and (rsi_turning_up or hist_turning_up)
        reclaiming_entry = close >= entry_price * (1.0 - max(atr_pct * 0.18, 0.0007))
        bullish_reclaim = close_above_ema20 and close_above_vwap and bounce_strength >= 3
        stagnation_exit = holding_minutes >= 14.0 and reward_r < 0.15 and rebound_confirmed and close_above_ema20
        fast_fail_exit = holding_minutes >= 5.0 and reward_r < 0.10 and close_above_ema20 and close_above_vwap and rebound_confirmed
        no_progress_exit = (
            holding_minutes >= 10.0
            and best_reward_r < 0.18
            and reward_r < 0.06
            and reclaiming_entry
            and bounce_strength >= 2
            and not close_above_vwap
        )
        profit_giveback_exit = (
            holding_minutes >= 6.0
            and best_reward_r >= 0.55
            and reward_r >= 0.12
            and reward_r <= max(0.12, best_reward_r * 0.42)
            and reward_retention <= 0.45
            and rebound_confirmed
            and (close_above_ema20 or close_above_vwap or reclaiming_entry)
        )
        time_decay_exit = holding_minutes >= 18.0 and reward_r < 0.12 and rebound_confirmed and (close_above_ema20 or close_above_vwap)
        target_bounce_exit = target_zone_touched and reward_r >= 0.32 and (bullish_reclaim or rebound_confirmed)
        profit_protect_exit = reward_r >= 0.85 and rebound_confirmed and (close_above_ema20 or close_above_vwap)
        hard_reclaim_exit = reward_r >= 0.15 and reclaiming_entry and close_above_ema20 and close_above_vwap and bounce_strength >= 4

        exit_type = ""
        reason = ""
        if hard_reclaim_exit:
            exit_type = "reclaim_invalidation"
            reason = "managed_exit_reclaim_invalidation"
        elif target_bounce_exit:
            exit_type = "target_zone_bounce"
            reason = "managed_exit_target_zone_bounce"
        elif profit_protect_exit:
            exit_type = "profit_reclaim"
            reason = "managed_exit_profit_reclaim"
        elif profit_giveback_exit:
            exit_type = "profit_giveback"
            reason = "managed_exit_profit_giveback"
        elif fast_fail_exit:
            exit_type = "failed_followthrough"
            reason = "managed_exit_failed_followthrough"
        elif no_progress_exit:
            exit_type = "no_progress_rebound"
            reason = "managed_exit_no_progress_rebound"
        elif time_decay_exit:
            exit_type = "time_decay"
            reason = "managed_exit_time_decay"
        elif stagnation_exit:
            exit_type = "stagnation"
            reason = "managed_exit_stagnation"

        if not reason:
            return None

        exit_confidence = min(
            0.96,
            0.52
            + min(0.18, reward_r * 0.12)
            + (0.08 if target_zone_touched else 0.0)
            + (0.06 if bullish_reclaim else 0.0),
        )
        exit_meta = {
            **(trace_meta if isinstance(trace_meta, dict) else {}),
            "managed_exit": True,
            "exit_type": exit_type,
            "managed_exit_reason": reason,
            "managed_exit_details": {
                "reward_r": float(reward_r),
                "best_reward_r": float(best_reward_r),
                "reward_retention": float(reward_retention),
                "holding_minutes": float(holding_minutes),
                "bars_held": float(bars_held),
                "timeframe_minutes": float(timeframe_minutes),
                "target_zone_touched": 1.0 if target_zone_touched else 0.0,
                "near_poc": 1.0 if near_poc else 0.0,
                "near_val": 1.0 if near_val else 0.0,
                "near_recent_support": 1.0 if near_recent_support else 0.0,
                "bullish_reclaim": 1.0 if bullish_reclaim else 0.0,
                "rebound_confirmed": 1.0 if rebound_confirmed else 0.0,
                "rsi_turning_up": 1.0 if rsi_turning_up else 0.0,
                "hist_turning_up": 1.0 if hist_turning_up else 0.0,
                "reclaiming_entry": 1.0 if reclaiming_entry else 0.0,
                "bounce_strength": float(bounce_strength),
            },
        }
        return StrategyIntent(
            symbol=context.symbol,
            action=IntentAction.EXIT_SHORT,
            reason=reason,
            confidence=float(exit_confidence),
            metadata=exit_meta,
        )

    def generate(self, context: StrategyContext) -> StrategyIntent:
        df = context.market_ohlcv
        if df.empty or len(df) < 80:
            self._audit.record({"failed_layer": "layer0_input", "layers": {}}, signal_side=None)
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.HOLD,
                reason="insufficient_history",
                metadata={"layer_failed": "layer0_input", "layer_trace": {}},
            )

        enriched = df
        if "rsi" not in enriched.columns:
            from core.indicators import compute_indicators

            enriched = compute_indicators(df)
        enriched = sanitize_feature_frame(enriched)

        regime = detect_market_regime(enriched)
        vp = compute_volume_profile(enriched)

        sentiment_value = self._coalesce(context.sentiment_value, context.sentiment_index)
        sentiment_index = self._coalesce(context.sentiment_index, context.sentiment_value)
        open_interest_ratio = self._coalesce(context.open_interest_ratio, context.open_interest)
        open_interest = self._coalesce(context.open_interest, context.open_interest_ratio, context.oi_signal)
        oi_source = self._coalesce(context.oi_source, context.open_interest_source)

        signal = self._generator.generate(
            SignalContext(
                symbol=context.symbol,
                df=enriched,
                volume_profile=vp,
                regime=regime,
                sentiment_index=sentiment_index,
                sentiment_value=sentiment_value,
                sentiment_source=context.sentiment_source,
                sentiment_degraded=context.sentiment_degraded,
                funding_rate=context.funding_rate,
                funding_source=context.funding_source,
                funding_degraded=context.funding_degraded,
                long_short_ratio=context.long_short_ratio,
                long_short_ratio_source=context.long_short_ratio_source,
                long_short_ratio_degraded=context.long_short_ratio_degraded,
                open_interest=open_interest,
                open_interest_ratio=open_interest_ratio,
                oi_signal=context.oi_signal,
                oi_source=oi_source,
                oi_degraded=context.oi_degraded,
                open_interest_source=context.open_interest_source,
                news_veto=context.news_veto,
                news_source=context.news_source,
                news_degraded=context.news_degraded,
            )
        )
        trace_meta = self._trace_meta()
        trace = trace_meta.get("layer_trace", {}) if isinstance(trace_meta, dict) else {}
        self._audit.record(trace, signal_side=getattr(signal, "side", None))
        managed_short_exit = self._managed_short_exit_intent(
            context=context,
            enriched=enriched,
            volume_profile=vp,
            trace_meta=trace_meta if isinstance(trace_meta, dict) else {},
        )

        if signal is None:
            if managed_short_exit is not None:
                return managed_short_exit
            failed_layer = trace_meta.get("layer_failed") or "unknown"
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.HOLD,
                reason=f"no_signal_{failed_layer}",
                metadata=trace_meta,
            )

        if context.synced_state in (TradeState.LONG, TradeState.PENDING_EXIT_LONG) and signal.side == "SHORT":
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.EXIT_LONG,
                reason="opposite_signal_close_long",
                confidence=float(signal.confidence),
                metadata={"legacy_signal_id": signal.signal_id, **trace_meta},
            )

        if context.synced_state in (TradeState.SHORT, TradeState.PENDING_EXIT_SHORT) and signal.side == "LONG":
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.EXIT_SHORT,
                reason="opposite_signal_close_short",
                confidence=float(signal.confidence),
                metadata={"legacy_signal_id": signal.signal_id, **trace_meta},
            )

        if managed_short_exit is not None:
            return managed_short_exit

        if context.synced_state != TradeState.FLAT:
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.HOLD,
                reason="state_not_flat",
                metadata=trace_meta,
            )

        entry_meta = {"legacy_signal_id": signal.signal_id, **trace_meta}
        if signal.partial_tps:
            entry_meta["partial_tps"] = [float(x) for x in signal.partial_tps]
        signal_details = signal.details if isinstance(signal.details, Mapping) else {}
        layer5_details = signal_details.get("layer5", {}) if isinstance(signal_details, Mapping) else {}
        if isinstance(layer5_details, Mapping):
            for key in ("tp1", "tp2", "tp3", "tp1_reference", "tp2_reference", "tp3_reference", "tp_reference"):
                if key in layer5_details:
                    entry_meta[key] = layer5_details.get(key)

        if signal.side == "LONG" and not self._allow_long_entries:
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.HOLD,
                reason="long_disabled_for_short_on_pump",
                metadata=trace_meta,
            )

        if signal.side == "LONG":
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.LONG_ENTRY,
                reason="layered_long_entry",
                stop_loss=float(signal.sl),
                take_profit=float(signal.tp),
                confidence=float(signal.confidence),
                metadata=entry_meta,
            )

        return StrategyIntent(
            symbol=context.symbol,
            action=IntentAction.SHORT_ENTRY,
            reason="layered_short_entry",
            stop_loss=float(signal.sl),
            take_profit=float(signal.tp),
            confidence=float(signal.confidence),
            metadata=entry_meta,
        )

