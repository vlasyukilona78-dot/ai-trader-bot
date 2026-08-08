from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import asdict
from inspect import signature

from core.indicators import compute_indicators
from core.market_regime import detect_market_regime
from core.mexc_strategy_spec import MexcStrategySpec
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
from core.volume_profile import compute_volume_profile
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.strategy_interface import StrategyContext, StrategyInterface
from trading.signals.volatility_context import VolatilityContext, VolatilityContextConfig
from trading.state.models import TradeState


def _runtime_defaults(function) -> dict[str, object]:
    return {
        name: parameter.default
        for name, parameter in signature(function).parameters.items()
        if name != "df" and parameter.default is not parameter.empty
    }


_DEFAULT_INDICATOR_KWARGS = _runtime_defaults(compute_indicators)
_DEFAULT_VOLUME_PROFILE_KWARGS = _runtime_defaults(compute_volume_profile)


class LayeredPumpStrategy(StrategyInterface):
    """Adapter around migrated layered strategy that returns intents only."""

    def __init__(
        self,
        config: SignalConfig | None = None,
        volatility_context: VolatilityContext | None = None,
        *,
        strategy_spec: MexcStrategySpec | None = None,
    ):
        if strategy_spec is not None and config is not None:
            raise ValueError("strategy_spec_and_signal_config_are_mutually_exclusive")
        if strategy_spec is not None and volatility_context is not None:
            raise ValueError("strategy_spec_and_volatility_context_are_mutually_exclusive")
        self._strategy_spec = strategy_spec
        resolved_config = (
            strategy_spec.to_signal_config() if strategy_spec is not None else config or SignalConfig()
        )
        self._indicator_kwargs = (
            strategy_spec.compute_indicators_kwargs()
            if strategy_spec is not None
            else dict(_DEFAULT_INDICATOR_KWARGS)
        )
        generator_min_history = (
            strategy_spec.runtime_semantics.signal_generator_min_history_bars
            if strategy_spec is not None
            else 40
        )
        self._generator = SignalGenerator(
            resolved_config,
            min_history_bars=generator_min_history,
            indicator_kwargs=self._indicator_kwargs,
        )
        # Every scanned symbol contributes its volatility, so the gate compares a
        # candidate against the rest of the board rather than a fixed number.
        self._volatility = volatility_context or VolatilityContext(
            strategy_spec.to_volatility_context_config()
            if strategy_spec is not None
            else VolatilityContextConfig(fallback_floor=self._generator.config.min_atr_pct)
        )
        self._volume_profile_kwargs = (
            strategy_spec.volume_profile_kwargs()
            if strategy_spec is not None
            else dict(_DEFAULT_VOLUME_PROFILE_KWARGS)
        )
        self._minimum_history_bars = (
            strategy_spec.runtime_semantics.layered_min_history_bars
            if strategy_spec is not None
            else 80
        )
        self._benchmark = None
        self._htf_cache = None
        # SignalGenerator diagnostics/pending confirmation state and the
        # cross-sectional volatility context are mutable. Scanner workers share
        # one strategy instance, so those operations must form one atomic unit.
        self._state_lock = threading.RLock()

    @property
    def strategy_spec(self) -> MexcStrategySpec | None:
        return self._strategy_spec

    def assert_strategy_spec_consistency(
        self, expected_spec: MexcStrategySpec | None = None
    ) -> None:
        """Fail if mutable runtime adapters drift from their recorded spec."""

        if (
            self._strategy_spec is not None
            and expected_spec is not None
            and self._strategy_spec.instance_hash != expected_spec.instance_hash
        ):
            raise RuntimeError("strategy_runtime_bound_to_a_different_strategy_spec")
        resolved_spec = self._strategy_spec or expected_spec
        if resolved_spec is None:
            return
        with self._state_lock:
            expected = {
                "signal": asdict(resolved_spec.to_signal_config()),
                "volatility": asdict(
                    resolved_spec.to_volatility_context_config()
                ),
            }
            actual = {
                "signal": asdict(self._generator.config),
                "volatility": asdict(self._volatility.config),
            }
            if actual != expected:
                raise RuntimeError("strategy_runtime_drifted_from_strategy_spec")
            if self._indicator_kwargs != resolved_spec.compute_indicators_kwargs():
                raise RuntimeError("indicator_runtime_drifted_from_strategy_spec")
            if self._generator.indicator_kwargs != resolved_spec.compute_indicators_kwargs():
                raise RuntimeError("htf_indicator_runtime_drifted_from_strategy_spec")
            if self._volume_profile_kwargs != resolved_spec.volume_profile_kwargs():
                raise RuntimeError("volume_profile_runtime_drifted_from_strategy_spec")
            if (
                self._minimum_history_bars
                != resolved_spec.runtime_semantics.layered_min_history_bars
            ):
                raise RuntimeError("history_runtime_drifted_from_strategy_spec")
            if (
                self._generator.min_history_bars
                != resolved_spec.runtime_semantics.signal_generator_min_history_bars
            ):
                raise RuntimeError("signal_generator_history_drifted_from_strategy_spec")

    def set_benchmark(self, frame):
        """Market reference (BTC OHLCV), refreshed once per scan cycle."""
        with self._state_lock:
            self._benchmark = frame

    def configuration_snapshot(self) -> dict:
        """Return only immutable strategy semantics, never mutable scan state."""

        with self._state_lock:
            return {
                "signal": asdict(self._generator.config),
                "volatility": asdict(self._volatility.config),
            }

    def begin_sweep(self):
        """Freeze the cross-sectional volatility floor for one scan pass."""
        with self._state_lock:
            self._volatility.start_sweep()

    def set_htf_cache(self, cache):
        """Source of higher-timeframe bars per symbol, so indicators can be read
        on the timeframe where they carry signal rather than all on the entry one."""
        if self._strategy_spec is not None and (
            getattr(cache, "config", None)
            != self._strategy_spec.to_timeframe_cache_config()
        ):
            raise ValueError("htf_cache_does_not_match_strategy_spec")
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
        if df.empty or len(df) < self._minimum_history_bars:
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.HOLD,
                reason="insufficient_history",
                metadata={"layer_failed": "layer0_input", "layer_trace": {}},
            )

        enriched = df
        if "rsi" not in enriched.columns:
            enriched = compute_indicators(df, **self._indicator_kwargs)

        regime = detect_market_regime(enriched)
        vp = compute_volume_profile(enriched, **self._volume_profile_kwargs)

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
