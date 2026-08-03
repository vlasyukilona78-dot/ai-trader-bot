from __future__ import annotations

import threading
from copy import deepcopy

from core.market_regime import detect_market_regime
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
from core.volume_profile import compute_volume_profile
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.strategy_interface import StrategyContext, StrategyInterface
from trading.signals.volatility_context import VolatilityContext, VolatilityContextConfig
from trading.state.models import TradeState


class LayeredPumpStrategy(StrategyInterface):
    """Adapter around migrated layered strategy that returns intents only."""

    def __init__(
        self,
        config: SignalConfig | None = None,
        volatility_context: VolatilityContext | None = None,
    ):
        self._generator = SignalGenerator(config or SignalConfig())
        # Every scanned symbol contributes its volatility, so the gate compares a
        # candidate against the rest of the board rather than a fixed number.
        self._volatility = volatility_context or VolatilityContext(
            VolatilityContextConfig(fallback_floor=self._generator.config.min_atr_pct)
        )
        self._benchmark = None
        self._htf_cache = None
        # SignalGenerator diagnostics/pending confirmation state and the
        # cross-sectional volatility context are mutable. Scanner workers share
        # one strategy instance, so those operations must form one atomic unit.
        self._state_lock = threading.RLock()

    def set_benchmark(self, frame):
        """Market reference (BTC OHLCV), refreshed once per scan cycle."""
        with self._state_lock:
            self._benchmark = frame

    def begin_sweep(self):
        """Freeze the cross-sectional volatility floor for one scan pass."""
        with self._state_lock:
            self._volatility.start_sweep()

    def set_htf_cache(self, cache):
        """Source of higher-timeframe bars per symbol, so indicators can be read
        on the timeframe where they carry signal rather than all on the entry one."""
        with self._state_lock:
            self._htf_cache = cache

    def _trace_meta(self) -> dict:
        trace = deepcopy(self._generator.last_diagnostics) if isinstance(self._generator.last_diagnostics, dict) else {}
        failed_layer = str(trace.get("failed_layer") or "") if trace else ""
        return {
            "layer_trace": trace,
            "layer_failed": failed_layer,
        }

    def generate(self, context: StrategyContext) -> StrategyIntent:
        df = context.market_ohlcv
        if df.empty or len(df) < 80:
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

        # Cache/network access stays outside the mutable-state lock so symbols
        # can still fetch higher-timeframe context in parallel.
        with self._state_lock:
            benchmark = self._benchmark
            htf_cache = self._htf_cache
        htf_frame = None
        if htf_cache is not None:
            if context.candle_cutoff_ts is None:
                htf_frame = htf_cache.get(context.symbol)
            else:
                htf_frame = htf_cache.get(context.symbol, as_of=context.candle_cutoff_ts)

        last = enriched.iloc[-1]
        close = float(last.get("close") or 0.0)
        atr = float(last.get("atr") or 0.0)
        with self._state_lock:
            if close > 0 and atr > 0:
                self._volatility.observe(context.symbol, atr / close)
            signal = self._generator.generate(
                SignalContext(
                    symbol=context.symbol,
                    df=enriched,
                    volume_profile=vp,
                    regime=regime,
                    sentiment_index=context.sentiment_index,
                    sentiment_source=context.sentiment_source,
                    funding_rate=context.funding_rate,
                    long_short_ratio=context.long_short_ratio,
                    atr_floor=self._volatility.floor(),
                    benchmark=benchmark,
                    htf_frame=htf_frame,
                )
            )
            trace_meta = self._trace_meta()
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
