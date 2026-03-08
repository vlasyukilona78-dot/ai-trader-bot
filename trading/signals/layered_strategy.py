from __future__ import annotations

from core.market_regime import detect_market_regime
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
from core.volume_profile import compute_volume_profile
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.strategy_interface import StrategyContext, StrategyInterface
from trading.state.models import TradeState


class LayeredPumpStrategy(StrategyInterface):
    """Adapter around migrated layered strategy that returns intents only."""

    def __init__(self, config: SignalConfig | None = None):
        self._generator = SignalGenerator(config or SignalConfig())

    def generate(self, context: StrategyContext) -> StrategyIntent:
        df = context.market_ohlcv
        if df.empty or len(df) < 80:
            return StrategyIntent(symbol=context.symbol, action=IntentAction.HOLD, reason="insufficient_history")

        enriched = df
        if "rsi" not in enriched.columns:
            from core.indicators import compute_indicators

            enriched = compute_indicators(df)

        regime = detect_market_regime(enriched)
        vp = compute_volume_profile(enriched)
        signal = self._generator.generate(
            SignalContext(
                symbol=context.symbol,
                df=enriched,
                volume_profile=vp,
                regime=regime,
                sentiment_index=context.sentiment_index,
                funding_rate=None,
                long_short_ratio=None,
            )
        )
        if signal is None:
            return StrategyIntent(symbol=context.symbol, action=IntentAction.HOLD, reason="no_signal")

        if context.synced_state in (TradeState.LONG, TradeState.PENDING_EXIT_LONG) and signal.side == "SHORT":
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.EXIT_LONG,
                reason="opposite_signal_close_long",
                confidence=float(signal.confidence),
                metadata={"legacy_signal_id": signal.signal_id},
            )

        if context.synced_state in (TradeState.SHORT, TradeState.PENDING_EXIT_SHORT) and signal.side == "LONG":
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.EXIT_SHORT,
                reason="opposite_signal_close_short",
                confidence=float(signal.confidence),
                metadata={"legacy_signal_id": signal.signal_id},
            )

        if context.synced_state != TradeState.FLAT:
            return StrategyIntent(symbol=context.symbol, action=IntentAction.HOLD, reason="state_not_flat")

        if signal.side == "LONG":
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.LONG_ENTRY,
                reason="layered_long_entry",
                stop_loss=float(signal.sl),
                take_profit=float(signal.tp),
                confidence=float(signal.confidence),
                metadata={"legacy_signal_id": signal.signal_id},
            )

        return StrategyIntent(
            symbol=context.symbol,
            action=IntentAction.SHORT_ENTRY,
            reason="layered_short_entry",
            stop_loss=float(signal.sl),
            take_profit=float(signal.tp),
            confidence=float(signal.confidence),
            metadata={"legacy_signal_id": signal.signal_id},
        )
