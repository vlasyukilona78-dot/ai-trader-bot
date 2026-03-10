from __future__ import annotations

from core.market_regime import detect_market_regime
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
from core.volume_profile import compute_volume_profile
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.strategy_audit import StrategyAuditCollector
from trading.signals.strategy_interface import StrategyContext, StrategyInterface
from trading.state.models import TradeState


class LayeredPumpStrategy(StrategyInterface):
    """Adapter around migrated layered strategy that returns intents only."""

    def __init__(self, config: SignalConfig | None = None, audit_collector: StrategyAuditCollector | None = None):
        self._generator = SignalGenerator(config or SignalConfig())
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

    def audit_snapshot(self) -> dict:
        return self._audit.snapshot()

    def audit_compact_snapshot(self) -> dict:
        return self._audit.compact_snapshot()

    def audit_observation_snapshot(self) -> dict:
        snapshot = self._audit.snapshot()
        compact = self._audit.compact_snapshot()
        return {
            "strategy_audit_compact": compact,
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

        if signal is None:
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

        if context.synced_state != TradeState.FLAT:
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.HOLD,
                reason="state_not_flat",
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
                metadata={"legacy_signal_id": signal.signal_id, **trace_meta},
            )

        return StrategyIntent(
            symbol=context.symbol,
            action=IntentAction.SHORT_ENTRY,
            reason="layered_short_entry",
            stop_loss=float(signal.sl),
            take_profit=float(signal.tp),
            confidence=float(signal.confidence),
            metadata={"legacy_signal_id": signal.signal_id, **trace_meta},
        )

