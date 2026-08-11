from __future__ import annotations

import math
import threading
from copy import deepcopy
from dataclasses import asdict
from inspect import signature

from core.indicators import compute_indicators
from core.market_regime import detect_market_regime
import pandas as pd

from core.mexc_strategy_spec import MexcStrategySpec, strategy_spec_identity
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
from core.volume_profile import compute_volume_profile
from trading.market_data.bar_contract import closed_boundary_ts
from trading.market_data.frame_provenance import (
    FrameProvenanceError,
    FrameRead,
    raw_frame_bundle_hash,
)
from trading.signals.lifecycle_contract import (
    CandidateLifecycleEventV1,
    LifecycleContractError,
)
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
_RAW_FRAME_COLUMNS = ("open", "high", "low", "close", "volume")


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
        # A population sweep spans many per-symbol calls and mutates pending
        # confirmation state. Serialise whole sweeps too: append-time duplicate
        # detection is deliberately too late to undo an in-memory transition.
        self._scan_session_lock = threading.RLock()

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

    def scan_session(self):
        """Return the process-local guard for one complete population sweep."""

        return self._scan_session_lock

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

    @staticmethod
    def _compact_symbol(value: object) -> str:
        return str(value).strip().upper().replace("/", "").replace("_", "").replace("-", "")

    @staticmethod
    def _owned_frame_read(value: object, *, field_name: str) -> FrameRead:
        if not isinstance(value, FrameRead):
            raise LifecycleContractError(f"{field_name}_must_be_frame_read")
        owned_frame = value.frame.copy(deep=True) if value.frame is not None else None
        try:
            return FrameRead(frame=owned_frame, evidence=value.evidence)
        except FrameProvenanceError as exc:
            raise LifecycleContractError(f"{field_name}_is_not_self_consistent") from exc

    @classmethod
    def _validate_read_identity(
        cls,
        read: FrameRead,
        *,
        field_name: str,
        expected_source: str,
        expected_symbol: str,
        expected_timeframe: str,
        requested_cutoff_ts: float,
    ) -> None:
        evidence = read.evidence
        if evidence.source != expected_source:
            raise LifecycleContractError(f"{field_name}_source_mismatch")
        if evidence.venue != "mexc_contract":
            raise LifecycleContractError(f"{field_name}_venue_mismatch")
        compact_symbol = cls._compact_symbol(expected_symbol)
        if cls._compact_symbol(evidence.symbol) != compact_symbol:
            raise LifecycleContractError(f"{field_name}_symbol_mismatch")
        if cls._compact_symbol(evidence.venue_symbol) != compact_symbol:
            raise LifecycleContractError(f"{field_name}_venue_symbol_mismatch")
        if evidence.timeframe != expected_timeframe:
            raise LifecycleContractError(f"{field_name}_timeframe_mismatch")
        if not math.isclose(
            evidence.requested_as_of_ts,
            requested_cutoff_ts,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise LifecycleContractError(f"{field_name}_requested_cutoff_mismatch")
        expected_boundary = float(
            closed_boundary_ts(requested_cutoff_ts, expected_timeframe)
        )
        if not math.isclose(
            evidence.expected_closed_boundary_ts,
            expected_boundary,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise LifecycleContractError(f"{field_name}_closed_boundary_mismatch")

    @staticmethod
    def _current_frame(read: FrameRead) -> pd.DataFrame | None:
        evidence = read.evidence
        if (
            read.frame is None
            or evidence.data_through_ts is None
            or not math.isclose(
                evidence.data_through_ts,
                evidence.expected_closed_boundary_ts,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            return None
        columns = list(_RAW_FRAME_COLUMNS)
        if "turnover" in read.frame.columns:
            columns.append("turnover")
        return read.frame.loc[:, columns].copy(deep=True)

    @staticmethod
    def _intent_from_signal(
        context: StrategyContext,
        signal,
        trace_meta: dict,
    ) -> StrategyIntent:
        """Map the legacy numeric result without adding lifecycle evidence."""

        if signal is None:
            failed_layer = trace_meta.get("layer_failed") or "unknown"
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.HOLD,
                reason=f"no_signal_{failed_layer}",
                metadata=trace_meta,
            )

        if context.synced_state in (
            TradeState.LONG,
            TradeState.PENDING_EXIT_LONG,
        ) and signal.side == "SHORT":
            return StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.EXIT_LONG,
                reason="opposite_signal_close_long",
                confidence=float(signal.confidence),
                metadata={"legacy_signal_id": signal.signal_id, **trace_meta},
            )

        if context.synced_state in (
            TradeState.SHORT,
            TradeState.PENDING_EXIT_SHORT,
        ) and signal.side == "LONG":
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

    def evaluate_with_lifecycle(
        self,
        context: StrategyContext,
        *,
        base_read: FrameRead,
        benchmark_read: FrameRead,
        higher_timeframe_read: FrameRead,
    ) -> tuple[StrategyIntent, CandidateLifecycleEventV1 | None]:
        """Atomically evaluate a spec-bound input and return typed evidence.

        All three reads are explicit and revalidated after taking owned frame
        copies. The typed path never consults mutable benchmark/cache state and
        derives its raw-input identity from the exact evidence it consumes.
        """

        if self._strategy_spec is None:
            raise LifecycleContractError("typed_evaluation_requires_strategy_spec")
        cutoff = context.candle_cutoff_ts
        if type(cutoff) not in (int, float) or not math.isfinite(float(cutoff)):
            raise LifecycleContractError("typed_evaluation_requires_finite_candle_cutoff")
        cutoff = float(cutoff)
        identity = strategy_spec_identity(self._strategy_spec)
        timeframe_seconds = self._strategy_spec.base_interval_seconds

        owned_base = self._owned_frame_read(base_read, field_name="base_read")
        owned_benchmark = self._owned_frame_read(
            benchmark_read, field_name="benchmark_read"
        )
        owned_htf = self._owned_frame_read(
            higher_timeframe_read, field_name="higher_timeframe_read"
        )
        self._validate_read_identity(
            owned_base,
            field_name="base_read",
            expected_source="base_ohlcv",
            expected_symbol=context.symbol,
            expected_timeframe=self._strategy_spec.market_data.base_interval,
            requested_cutoff_ts=cutoff,
        )
        self._validate_read_identity(
            owned_benchmark,
            field_name="benchmark_read",
            expected_source="benchmark_ohlcv",
            expected_symbol="BTCUSDT",
            expected_timeframe=self._strategy_spec.resolved_benchmark_interval,
            requested_cutoff_ts=cutoff,
        )
        self._validate_read_identity(
            owned_htf,
            field_name="higher_timeframe_read",
            expected_source="higher_timeframe_ohlcv",
            expected_symbol=context.symbol,
            expected_timeframe=self._strategy_spec.market_data.higher_timeframe.interval,
            requested_cutoff_ts=cutoff,
        )
        df = self._current_frame(owned_base)
        if df is None:
            raise LifecycleContractError("base_read_must_cover_current_closed_boundary")
        benchmark = self._current_frame(owned_benchmark)
        htf_frame = self._current_frame(owned_htf)
        try:
            input_bundle_hash = raw_frame_bundle_hash(
                [
                    owned_base.evidence,
                    owned_benchmark.evidence,
                    owned_htf.evidence,
                ]
            )
        except FrameProvenanceError as exc:
            raise LifecycleContractError("typed_raw_frame_bundle_is_invalid") from exc

        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            raise LifecycleContractError("typed_evaluation_requires_utc_bar_index")
        bar_open_ts = float(df.index[-1].timestamp())
        if not math.isclose(
            cutoff - bar_open_ts,
            float(timeframe_seconds),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise LifecycleContractError(
                "typed_evaluation_cutoff_does_not_close_last_base_bar"
            )

        if df.empty or len(df) < self._minimum_history_bars:
            intent = StrategyIntent(
                symbol=context.symbol,
                action=IntentAction.HOLD,
                reason="insufficient_history",
                metadata={"layer_failed": "layer0_input", "layer_trace": {}},
            )
            return intent, None

        enriched = compute_indicators(df, **self._indicator_kwargs)
        regime = detect_market_regime(enriched)
        vp = compute_volume_profile(enriched, **self._volume_profile_kwargs)
        last = enriched.iloc[-1]
        close = float(last.get("close") or 0.0)
        atr = float(last.get("atr") or 0.0)

        with self._state_lock:
            self.assert_strategy_spec_consistency(self._strategy_spec)
            if close > 0 and atr > 0:
                self._volatility.observe(context.symbol, atr / close)
            signal, lifecycle_event = self._generator.generate_with_lifecycle(
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
                ),
                strategy_spec_version=identity.spec_version,
                strategy_spec_contract_hash=identity.contract_hash,
                strategy_spec_instance_hash=identity.instance_hash,
                raw_input_bundle_hash=input_bundle_hash,
                timeframe_seconds=timeframe_seconds,
                candle_cutoff_ts=cutoff,
            )
            trace_meta = self._trace_meta()
            intent = self._intent_from_signal(context, signal, trace_meta)
            return intent, lifecycle_event

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
        return self._intent_from_signal(context, signal, trace_meta)
