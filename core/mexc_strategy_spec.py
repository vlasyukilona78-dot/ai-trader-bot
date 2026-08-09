"""Strict, versioned configuration contract for the MEXC signal strategy.

This module deliberately does not wire itself into ``app.scan``.  It captures the
configuration that the current MEXC scanner actually executes so that a later
integration can replace scattered defaults without also changing the strategy.

Window fields remain counts of their source bars.  In particular, a 45-bar pump
window is not silently reinterpreted as a fixed number of seconds when the base
interval changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from core.signal_generator import SignalConfig
from trading.market_data.bar_contract import BarContractError, interval_seconds
from trading.market_data.timeframe_cache import TimeframeCacheConfig
from trading.signals.volatility_context import VolatilityContextConfig

try:
    import yaml
except Exception:  # pragma: no cover - the loader reports this explicitly
    yaml = None


MEXC_STRATEGY_SPEC_V2_VERSION = "mexc_strategy_v2"
# The production-facing current version remains an explicit alias.  Historical
# readers and hashes must use the version carried by their own payload instead.
MEXC_STRATEGY_SPEC_VERSION = MEXC_STRATEGY_SPEC_V2_VERSION
DEFAULT_MEXC_STRATEGY_V2_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "mexc_strategy_v2.yaml"
)
DEFAULT_MEXC_STRATEGY_SPEC_PATH = DEFAULT_MEXC_STRATEGY_V2_SPEC_PATH

_PINNED_CONTRACT_HASHES = {
    # Filled with the digest of the declarative schema below.  A field/layout or
    # adapter-semantics change requires a new MEXC_STRATEGY_SPEC_VERSION.
    MEXC_STRATEGY_SPEC_V2_VERSION: "9c62b88b7804e9663bae6f0eb429c58c541680b61d307c4f16032cb0b62fe3dd",
}

_CANONICAL_INTERVAL_BY_SECONDS = {
    60: "Min1",
    5 * 60: "Min5",
    15 * 60: "Min15",
    30 * 60: "Min30",
    60 * 60: "Min60",
    4 * 60 * 60: "Hour4",
    8 * 60 * 60: "Hour8",
    24 * 60 * 60: "Day1",
    7 * 24 * 60 * 60: "Week1",
}


class MexcStrategySpecError(ValueError):
    """Raised when a strategy specification is incomplete or ambiguous."""


def _as_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MexcStrategySpecError(f"{path}_must_be_a_mapping")
    if any(not isinstance(key, str) for key in value):
        raise MexcStrategySpecError(f"{path}_keys_must_be_strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, expected: tuple[str, ...], path: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    missing = sorted(expected_set - actual_set)
    unknown = sorted(actual_set - expected_set)
    if missing:
        raise MexcStrategySpecError(f"{path}_missing_keys:{','.join(missing)}")
    if unknown:
        raise MexcStrategySpecError(f"{path}_unknown_keys:{','.join(unknown)}")


def _strict_bool(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        raise MexcStrategySpecError(f"{path}_must_be_boolean")
    return value


def _strict_int(
    value: Any,
    *,
    path: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise MexcStrategySpecError(f"{path}_must_be_integer")
    if minimum is not None and value < minimum:
        raise MexcStrategySpecError(f"{path}_below_minimum")
    if maximum is not None and value > maximum:
        raise MexcStrategySpecError(f"{path}_above_maximum")
    return value


def _strict_number(
    value: Any,
    *,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_exclusive: bool = False,
) -> float:
    if type(value) not in (int, float):
        raise MexcStrategySpecError(f"{path}_must_be_number")
    number = float(value)
    if not math.isfinite(number):
        raise MexcStrategySpecError(f"{path}_must_be_finite")
    if minimum is not None and number < minimum:
        raise MexcStrategySpecError(f"{path}_below_minimum")
    if maximum is not None:
        if maximum_exclusive and number >= maximum:
            raise MexcStrategySpecError(f"{path}_above_maximum")
        if not maximum_exclusive and number > maximum:
            raise MexcStrategySpecError(f"{path}_above_maximum")
    return number


def _strict_literal(value: Any, *, path: str, expected: str) -> str:
    if type(value) is not str or value != expected:
        raise MexcStrategySpecError(f"{path}_must_equal:{expected}")
    return value


def canonical_mexc_interval(value: Any, *, path: str = "interval") -> str:
    """Normalize supported numeric and MEXC interval aliases."""

    if type(value) is not str or not value.strip():
        raise MexcStrategySpecError(f"{path}_must_be_non_empty_string")
    try:
        seconds = interval_seconds(value)
    except BarContractError as exc:
        raise MexcStrategySpecError(f"{path}_unsupported") from exc
    canonical = _CANONICAL_INTERVAL_BY_SECONDS.get(seconds)
    if canonical is None:
        raise MexcStrategySpecError(f"{path}_unsupported")
    return canonical


@dataclass(frozen=True, slots=True)
class SignalParameters:
    rsi_high: float
    rsi_low: float
    volume_spike_threshold: float
    weakness_lookback: int
    sentiment_bullish_threshold: float
    sentiment_bearish_threshold: float
    risk_reward: float
    atr_sl_mult: float
    entry_tolerance_pct: float
    vwap_tolerance_pct: float
    funding_tolerance: float
    long_short_ratio_tolerance: float
    msb_lookback: int
    msb_recent_bars: int
    msb_break_buffer_pct: float
    confirmation_enabled: bool
    confirmation_max_wait_bars: int
    confirmation_invalidate_pct: float
    pump_window_enabled: bool
    pump_window_bars: int
    pump_min_move_pct: float
    pump_min_bars_since_peak: int
    pump_max_retrace_pct: float
    pump_entry_max_dist_from_peak_pct: float
    pump_stop_buffer_pct: float
    stop_buffer_atr_mult: float
    structural_anchor_htf: bool
    structural_anchor_htf_bars: int
    max_stop_distance_pct: float
    min_risk_reward: float
    report_leverage: float
    min_atr_pct: float
    min_hourly_usd_volume: float
    liquidity_lookback_bars: int
    min_relative_strength: float
    relative_strength_lookback: int
    require_benchmark: bool
    require_level_overhead: bool
    min_level_dist_pct: float
    weakness_layer_enabled: bool
    min_rsi_4h: float
    require_htf: bool
    min_rsi_1h: float
    require_confluence: int
    max_chase_atr: float
    enable_long_side: bool

    @classmethod
    def from_mapping(cls, value: Any) -> "SignalParameters":
        payload = _as_mapping(value, path="signal")
        names = tuple(field.name for field in fields(cls))
        _require_exact_keys(payload, expected=names, path="signal")

        bool_fields = {
            "confirmation_enabled",
            "pump_window_enabled",
            "structural_anchor_htf",
            "require_benchmark",
            "require_level_overhead",
            "weakness_layer_enabled",
            "require_htf",
            "enable_long_side",
        }
        positive_int_fields = {
            "weakness_lookback",
            "msb_lookback",
            "msb_recent_bars",
            "pump_window_bars",
            "structural_anchor_htf_bars",
            "liquidity_lookback_bars",
            "relative_strength_lookback",
        }
        non_negative_int_fields = {
            "confirmation_max_wait_bars",
            "pump_min_bars_since_peak",
            "require_confluence",
        }
        bounded_100_fields = {
            "rsi_high",
            "rsi_low",
            "sentiment_bullish_threshold",
            "sentiment_bearish_threshold",
            "min_rsi_4h",
            "min_rsi_1h",
        }
        non_negative_fields = {
            "atr_sl_mult",
            "entry_tolerance_pct",
            "vwap_tolerance_pct",
            "funding_tolerance",
            "long_short_ratio_tolerance",
            "msb_break_buffer_pct",
            "confirmation_invalidate_pct",
            "pump_min_move_pct",
            "pump_entry_max_dist_from_peak_pct",
            "pump_stop_buffer_pct",
            "stop_buffer_atr_mult",
            "max_stop_distance_pct",
            "min_atr_pct",
            "min_hourly_usd_volume",
            "min_level_dist_pct",
            "max_chase_atr",
        }
        strictly_positive_fields = {
            "volume_spike_threshold",
            "risk_reward",
            "min_risk_reward",
            "report_leverage",
        }

        parsed: dict[str, Any] = {}
        for name in names:
            raw = payload[name]
            path = f"signal.{name}"
            if name in bool_fields:
                parsed[name] = _strict_bool(raw, path=path)
            elif name in positive_int_fields:
                parsed[name] = _strict_int(raw, path=path, minimum=1)
            elif name in non_negative_int_fields:
                parsed[name] = _strict_int(raw, path=path, minimum=0)
            elif name in bounded_100_fields:
                parsed[name] = _strict_number(raw, path=path, minimum=0.0, maximum=100.0)
            elif name == "pump_max_retrace_pct":
                parsed[name] = _strict_number(raw, path=path, minimum=0.0, maximum=1.0)
            elif name in non_negative_fields:
                parsed[name] = _strict_number(raw, path=path, minimum=0.0)
            elif name in strictly_positive_fields:
                parsed[name] = _strict_number(raw, path=path, minimum=0.0)
                if parsed[name] == 0.0:
                    raise MexcStrategySpecError(f"{path}_must_be_positive")
            else:
                # min_relative_strength is intentionally allowed to be negative;
                # the current implementation treats non-positive values as off.
                parsed[name] = _strict_number(raw, path=path)

        if parsed["rsi_low"] >= parsed["rsi_high"]:
            raise MexcStrategySpecError("signal_rsi_thresholds_not_ordered")
        if parsed["sentiment_bearish_threshold"] > parsed["sentiment_bullish_threshold"]:
            raise MexcStrategySpecError("signal_sentiment_thresholds_not_ordered")
        # These knobs exist in the legacy dataclass but do not yet have a causal
        # runtime source/decision path.  Recording a non-zero value as executable
        # strategy evidence would be worse than rejecting it explicitly.
        if parsed["min_rsi_1h"] != 0.0:
            raise MexcStrategySpecError("signal.min_rsi_1h_not_implemented")
        if parsed["require_confluence"] != 0:
            raise MexcStrategySpecError("signal.require_confluence_not_implemented")
        return cls(**parsed)


@dataclass(frozen=True, slots=True)
class VolatilityParameters:
    percentile: float
    max_age_sec: float
    min_observations: int
    fallback_floor: float

    @classmethod
    def from_mapping(cls, value: Any) -> "VolatilityParameters":
        payload = _as_mapping(value, path="volatility")
        names = tuple(field.name for field in fields(cls))
        _require_exact_keys(payload, expected=names, path="volatility")
        return cls(
            percentile=_strict_number(
                payload["percentile"],
                path="volatility.percentile",
                minimum=0.0,
                maximum=1.0,
                maximum_exclusive=True,
            ),
            max_age_sec=_strict_number(
                payload["max_age_sec"], path="volatility.max_age_sec", minimum=0.0
            ),
            min_observations=_strict_int(
                payload["min_observations"],
                path="volatility.min_observations",
                minimum=1,
            ),
            fallback_floor=_strict_number(
                payload["fallback_floor"],
                path="volatility.fallback_floor",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class HigherTimeframeParameters:
    interval: str
    candles: int
    ttl_sec: float
    max_symbols: int

    @classmethod
    def from_mapping(cls, value: Any) -> "HigherTimeframeParameters":
        payload = _as_mapping(value, path="market_data.higher_timeframe")
        names = tuple(field.name for field in fields(cls))
        _require_exact_keys(
            payload, expected=names, path="market_data.higher_timeframe"
        )
        return cls(
            interval=canonical_mexc_interval(
                payload["interval"], path="market_data.higher_timeframe.interval"
            ),
            candles=_strict_int(
                payload["candles"],
                path="market_data.higher_timeframe.candles",
                minimum=1,
            ),
            ttl_sec=_strict_number(
                payload["ttl_sec"],
                path="market_data.higher_timeframe.ttl_sec",
                minimum=0.0,
            ),
            max_symbols=_strict_int(
                payload["max_symbols"],
                path="market_data.higher_timeframe.max_symbols",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class MarketDataParameters:
    base_interval: str
    base_candles: int
    benchmark_interval: str
    higher_timeframe: HigherTimeframeParameters

    @classmethod
    def from_mapping(cls, value: Any) -> "MarketDataParameters":
        payload = _as_mapping(value, path="market_data")
        names = tuple(field.name for field in fields(cls))
        _require_exact_keys(payload, expected=names, path="market_data")
        return cls(
            base_interval=canonical_mexc_interval(
                payload["base_interval"], path="market_data.base_interval"
            ),
            base_candles=_strict_int(
                payload["base_candles"], path="market_data.base_candles", minimum=1
            ),
            benchmark_interval=_strict_literal(
                payload["benchmark_interval"],
                path="market_data.benchmark_interval",
                expected="same_as_base",
            ),
            higher_timeframe=HigherTimeframeParameters.from_mapping(
                payload["higher_timeframe"]
            ),
        )

    @property
    def resolved_benchmark_interval(self) -> str:
        return self.base_interval


@dataclass(frozen=True, slots=True)
class RuntimeSemantics:
    logic_revision: str
    window_semantics: str
    layered_min_history_bars: int
    signal_generator_min_history_bars: int

    @classmethod
    def from_mapping(cls, value: Any) -> "RuntimeSemantics":
        payload = _as_mapping(value, path="runtime_semantics")
        names = tuple(field.name for field in fields(cls))
        _require_exact_keys(payload, expected=names, path="runtime_semantics")
        return cls(
            logic_revision=_strict_literal(
                payload["logic_revision"],
                path="runtime_semantics.logic_revision",
                expected="layered_pump_signal_v1",
            ),
            window_semantics=_strict_literal(
                payload["window_semantics"],
                path="runtime_semantics.window_semantics",
                expected="fixed_bar_counts",
            ),
            layered_min_history_bars=_strict_int(
                payload["layered_min_history_bars"],
                path="runtime_semantics.layered_min_history_bars",
                minimum=1,
            ),
            signal_generator_min_history_bars=_strict_int(
                payload["signal_generator_min_history_bars"],
                path="runtime_semantics.signal_generator_min_history_bars",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class IndicatorParameters:
    revision: str
    rsi_period_bars: int
    ema_fast_span_bars: int
    ema_slow_span_bars: int
    atr_period_bars: int
    bollinger_period_bars: int
    bollinger_stddev_multiplier: float
    keltner_period_bars: int
    keltner_atr_multiplier: float
    volume_ma_period_bars: int
    adx_period_bars: int
    vwap_mode: str
    obv_mode: str
    cvd_mode: str

    @classmethod
    def from_mapping(cls, value: Any) -> "IndicatorParameters":
        payload = _as_mapping(value, path="indicators")
        names = tuple(field.name for field in fields(cls))
        _require_exact_keys(payload, expected=names, path="indicators")
        int_names = {
            "rsi_period_bars",
            "ema_fast_span_bars",
            "ema_slow_span_bars",
            "atr_period_bars",
            "bollinger_period_bars",
            "keltner_period_bars",
            "volume_ma_period_bars",
            "adx_period_bars",
        }
        parsed: dict[str, Any] = {
            name: _strict_int(payload[name], path=f"indicators.{name}", minimum=1)
            for name in int_names
        }
        parsed.update(
            revision=_strict_literal(
                payload["revision"],
                path="indicators.revision",
                expected="core_indicators_v1",
            ),
            bollinger_stddev_multiplier=_strict_number(
                payload["bollinger_stddev_multiplier"],
                path="indicators.bollinger_stddev_multiplier",
                minimum=0.0,
            ),
            keltner_atr_multiplier=_strict_number(
                payload["keltner_atr_multiplier"],
                path="indicators.keltner_atr_multiplier",
                minimum=0.0,
            ),
            vwap_mode=_strict_literal(
                payload["vwap_mode"],
                path="indicators.vwap_mode",
                expected="cumulative_input_frame",
            ),
            obv_mode=_strict_literal(
                payload["obv_mode"],
                path="indicators.obv_mode",
                expected="cumulative_input_frame",
            ),
            cvd_mode=_strict_literal(
                payload["cvd_mode"],
                path="indicators.cvd_mode",
                expected="candle_direction_cumulative_input_frame",
            ),
        )
        if parsed["ema_fast_span_bars"] >= parsed["ema_slow_span_bars"]:
            raise MexcStrategySpecError("indicator_ema_spans_not_ordered")
        return cls(**parsed)


@dataclass(frozen=True, slots=True)
class VolumeProfileParameters:
    revision: str
    window_bars: int
    bins: int
    value_area: float
    minimum_history_bars: int
    minimum_sample_bars: int

    @classmethod
    def from_mapping(cls, value: Any) -> "VolumeProfileParameters":
        payload = _as_mapping(value, path="volume_profile")
        names = tuple(field.name for field in fields(cls))
        _require_exact_keys(payload, expected=names, path="volume_profile")
        return cls(
            revision=_strict_literal(
                payload["revision"],
                path="volume_profile.revision",
                expected="core_volume_profile_v1",
            ),
            window_bars=_strict_int(
                payload["window_bars"], path="volume_profile.window_bars", minimum=1
            ),
            bins=_strict_int(payload["bins"], path="volume_profile.bins", minimum=8),
            value_area=_strict_number(
                payload["value_area"],
                path="volume_profile.value_area",
                minimum=0.50,
                maximum=0.95,
            ),
            minimum_history_bars=_strict_int(
                payload["minimum_history_bars"],
                path="volume_profile.minimum_history_bars",
                minimum=1,
            ),
            minimum_sample_bars=_strict_int(
                payload["minimum_sample_bars"],
                path="volume_profile.minimum_sample_bars",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class MexcStrategySpec:
    spec_version: str
    market_data: MarketDataParameters
    runtime_semantics: RuntimeSemantics
    signal: SignalParameters
    volatility: VolatilityParameters
    indicators: IndicatorParameters
    volume_profile: VolumeProfileParameters

    @classmethod
    def from_mapping(cls, value: Any) -> "MexcStrategySpec":
        payload = _as_mapping(value, path="strategy_spec")
        names = tuple(field.name for field in fields(cls))
        _require_exact_keys(payload, expected=names, path="strategy_spec")
        spec = cls(
            spec_version=_strict_literal(
                payload["spec_version"],
                path="spec_version",
                expected=MEXC_STRATEGY_SPEC_V2_VERSION,
            ),
            market_data=MarketDataParameters.from_mapping(payload["market_data"]),
            runtime_semantics=RuntimeSemantics.from_mapping(
                payload["runtime_semantics"]
            ),
            signal=SignalParameters.from_mapping(payload["signal"]),
            volatility=VolatilityParameters.from_mapping(payload["volatility"]),
            indicators=IndicatorParameters.from_mapping(payload["indicators"]),
            volume_profile=VolumeProfileParameters.from_mapping(
                payload["volume_profile"]
            ),
        )
        spec._validate_cross_section()
        # Construction is also the fail-closed boundary for an unversioned schema
        # edit, rather than deferring that failure until a caller asks for a hash.
        strategy_spec_contract_hash(MEXC_STRATEGY_SPEC_V2_VERSION)
        return spec

    @classmethod
    def from_yaml(
        cls, path: str | Path = DEFAULT_MEXC_STRATEGY_V2_SPEC_PATH
    ) -> "MexcStrategySpec":
        return cls.from_mapping(_load_unique_yaml(path))

    def _validate_cross_section(self) -> None:
        if self.market_data.base_candles < self.runtime_semantics.layered_min_history_bars:
            raise MexcStrategySpecError("base_candles_below_layered_minimum_history")
        if (
            self.runtime_semantics.layered_min_history_bars
            < self.runtime_semantics.signal_generator_min_history_bars
        ):
            raise MexcStrategySpecError("layered_history_below_signal_generator_history")
        if self.volume_profile.window_bars > self.market_data.base_candles:
            raise MexcStrategySpecError("volume_profile_window_exceeds_base_candles")
        if (
            self.signal.structural_anchor_htf_bars
            > self.market_data.higher_timeframe.candles
        ):
            raise MexcStrategySpecError("structural_anchor_exceeds_htf_candles")

    @property
    def base_interval_seconds(self) -> int:
        return interval_seconds(self.market_data.base_interval)

    @property
    def higher_timeframe_interval_seconds(self) -> int:
        return interval_seconds(self.market_data.higher_timeframe.interval)

    @property
    def resolved_benchmark_interval(self) -> str:
        return self.market_data.resolved_benchmark_interval

    def base_window_seconds(self, bars: int) -> int:
        if type(bars) is not int or bars < 0:
            raise MexcStrategySpecError("base_window_bars_must_be_non_negative_integer")
        return bars * self.base_interval_seconds

    def higher_timeframe_window_seconds(self, bars: int) -> int:
        if type(bars) is not int or bars < 0:
            raise MexcStrategySpecError("htf_window_bars_must_be_non_negative_integer")
        return bars * self.higher_timeframe_interval_seconds

    def to_signal_config(self) -> SignalConfig:
        return SignalConfig(**asdict(self.signal))

    def to_volatility_context_config(self) -> VolatilityContextConfig:
        return VolatilityContextConfig(**asdict(self.volatility))

    def to_timeframe_cache_config(self) -> TimeframeCacheConfig:
        return TimeframeCacheConfig(**asdict(self.market_data.higher_timeframe))

    def indicator_contract(self) -> dict[str, Any]:
        return asdict(self.indicators)

    def compute_indicators_kwargs(self) -> dict[str, float]:
        return {
            "rsi_period": self.indicators.rsi_period_bars,
            "ema_fast_span": self.indicators.ema_fast_span_bars,
            "ema_slow_span": self.indicators.ema_slow_span_bars,
            "atr_period": self.indicators.atr_period_bars,
            "bollinger_period": self.indicators.bollinger_period_bars,
            "bollinger_mult": self.indicators.bollinger_stddev_multiplier,
            "keltner_period": self.indicators.keltner_period_bars,
            "keltner_mult": self.indicators.keltner_atr_multiplier,
            "volume_ma_period": self.indicators.volume_ma_period_bars,
            "adx_period": self.indicators.adx_period_bars,
        }

    def volume_profile_kwargs(self) -> dict[str, Any]:
        return {
            "window": self.volume_profile.window_bars,
            "bins": self.volume_profile.bins,
            "value_area": self.volume_profile.value_area,
            "minimum_history_bars": self.volume_profile.minimum_history_bars,
            "minimum_sample_bars": self.volume_profile.minimum_sample_bars,
        }

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def instance_hash(self) -> str:
        return strategy_spec_hash(self)


if yaml is not None:

    class _UniqueKeyLoader(yaml.SafeLoader):
        pass


    def _construct_unique_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise MexcStrategySpecError("yaml_keys_must_be_strings")
            if key in mapping:
                raise MexcStrategySpecError(f"yaml_duplicate_key:{key}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping


    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_unique_mapping,
    )


def _load_unique_yaml(path: str | Path) -> Mapping[str, Any]:
    if yaml is None:
        raise MexcStrategySpecError("pyyaml_dependency_unavailable")
    resolved = Path(path)
    if not resolved.is_file():
        raise MexcStrategySpecError(f"strategy_spec_file_not_found:{resolved}")
    try:
        parsed = yaml.load(resolved.read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    except MexcStrategySpecError:
        raise
    except yaml.YAMLError as exc:
        raise MexcStrategySpecError("strategy_spec_yaml_invalid") from exc
    return _as_mapping(parsed, path="strategy_spec")


def load_mexc_strategy_spec(
    path: str | Path = DEFAULT_MEXC_STRATEGY_SPEC_PATH,
) -> MexcStrategySpec:
    # Production remains an explicit current-version entrypoint.  Persisted
    # evidence uses the version-dispatched decoder below instead.
    return MexcStrategySpec.from_yaml(path)


def _field_layout(cls) -> list[dict[str, str]]:
    return [
        {"name": field.name, "type": str(field.type)}
        for field in fields(cls)
    ]


def _v2_contract_payload() -> dict[str, Any]:
    return {
        "spec_version": MEXC_STRATEGY_SPEC_V2_VERSION,
        "validation_revision": "strict_mapping_types_ranges_v1",
        "interval_canonicalization_revision": "fixed_mexc_aliases_v1",
        "layouts": {
            cls.__name__: _field_layout(cls)
            for cls in (
                MexcStrategySpec,
                MarketDataParameters,
                HigherTimeframeParameters,
                RuntimeSemantics,
                SignalParameters,
                VolatilityParameters,
                IndicatorParameters,
                VolumeProfileParameters,
            )
        },
        "adapters": {
            "signal": "core.signal_generator.SignalConfig",
            "volatility": "trading.signals.volatility_context.VolatilityContextConfig",
            "higher_timeframe": "trading.market_data.timeframe_cache.TimeframeCacheConfig",
            "benchmark_interval": "same_as_base",
            "indicator_runtime_kwargs": (
                "rsi_period",
                "ema_fast_span",
                "ema_slow_span",
                "atr_period",
                "bollinger_period",
                "bollinger_mult",
                "keltner_period",
                "keltner_mult",
                "volume_ma_period",
                "adx_period",
            ),
            "volume_profile_runtime_kwargs": (
                "window",
                "bins",
                "value_area",
                "minimum_history_bars",
                "minimum_sample_bars",
            ),
            "signal_generator_min_history": "runtime_semantics.signal_generator_min_history_bars",
            "unsupported_nonzero_signal_fields": (
                "min_rsi_1h",
                "require_confluence",
            ),
        },
        "units": {
            "all_named_bar_windows": "source_bar_counts",
            "ttl_and_max_age": "wall_clock_seconds",
            "intervals": "canonical_mexc_fixed_intervals",
        },
    }


@dataclass(frozen=True, slots=True)
class _StrategySpecVersionRegistration:
    spec_version: str
    spec_type: type[Any]
    parse_mapping: Callable[[Any], Any]
    contract_payload: Callable[[], dict[str, Any]]
    hash_instance: Callable[[Any], str]


@dataclass(frozen=True, slots=True)
class StrategySpecIdentity:
    """Version-bound identity of one validated strategy specification."""

    spec_version: str
    contract_hash: str
    instance_hash: str


def _parse_mexc_strategy_spec_v2(value: Any) -> MexcStrategySpec:
    return MexcStrategySpec.from_mapping(value)


def _strategy_spec_hash_v2(spec: Any) -> str:
    """Frozen v2 instance-identity serialization; do not generalize in place."""

    if not isinstance(spec, MexcStrategySpec):
        raise TypeError("spec must be a MexcStrategySpec")
    encoded = json.dumps(
        {
            "contract_hash": strategy_spec_contract_hash(
                MEXC_STRATEGY_SPEC_V2_VERSION
            ),
            "spec": spec.to_mapping(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_STRATEGY_SPEC_VERSION_REGISTRY: Mapping[
    str, _StrategySpecVersionRegistration
] = MappingProxyType(
    {
        MEXC_STRATEGY_SPEC_V2_VERSION: _StrategySpecVersionRegistration(
            spec_version=MEXC_STRATEGY_SPEC_V2_VERSION,
            spec_type=MexcStrategySpec,
            parse_mapping=_parse_mexc_strategy_spec_v2,
            contract_payload=_v2_contract_payload,
            hash_instance=_strategy_spec_hash_v2,
        ),
    }
)

SUPPORTED_MEXC_STRATEGY_SPEC_VERSIONS = tuple(_STRATEGY_SPEC_VERSION_REGISTRY)


def _strategy_spec_version_registration(
    spec_version: Any,
) -> _StrategySpecVersionRegistration:
    if type(spec_version) is not str or not spec_version:
        raise MexcStrategySpecError("spec_version_must_be_non_empty_string")
    registration = _STRATEGY_SPEC_VERSION_REGISTRY.get(spec_version)
    if registration is None:
        raise MexcStrategySpecError(
            f"strategy_spec_version_unsupported:{spec_version}"
        )
    return registration


def parse_mexc_strategy_spec(value: Any) -> MexcStrategySpec:
    """Parse persisted evidence with the immutable parser for its own version."""

    payload = _as_mapping(value, path="strategy_spec")
    if "spec_version" not in payload:
        raise MexcStrategySpecError("strategy_spec_missing_keys:spec_version")
    payload_version = payload["spec_version"]
    registration = _strategy_spec_version_registration(payload_version)
    parsed = registration.parse_mapping(payload)
    if not isinstance(parsed, registration.spec_type):
        raise RuntimeError("strategy_spec_registry_parser_type_mismatch")
    if getattr(parsed, "spec_version", None) != registration.spec_version:
        raise RuntimeError("strategy_spec_registry_parser_version_mismatch")
    return parsed


def decode_mexc_strategy_spec_evidence(
    value: Any,
    *,
    expected_version: str,
) -> MexcStrategySpec:
    """Decode evidence only when its outer and embedded versions agree."""

    _strategy_spec_version_registration(expected_version)
    payload = _as_mapping(value, path="strategy_spec")
    if "spec_version" not in payload:
        raise MexcStrategySpecError("strategy_spec_missing_keys:spec_version")
    payload_version = payload["spec_version"]
    if type(payload_version) is not str or not payload_version:
        raise MexcStrategySpecError("spec_version_must_be_non_empty_string")
    if payload_version != expected_version:
        raise MexcStrategySpecError("strategy_spec_evidence_version_mismatch")
    return parse_mexc_strategy_spec(payload)


def _contract_payload(
    spec_version: str = MEXC_STRATEGY_SPEC_VERSION,
) -> dict[str, Any]:
    registration = _strategy_spec_version_registration(spec_version)
    return registration.contract_payload()


def _contract_digest(
    spec_version: str = MEXC_STRATEGY_SPEC_VERSION,
) -> str:
    encoded = json.dumps(
        _contract_payload(spec_version),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def strategy_spec_contract_hash(
    spec_version: str = MEXC_STRATEGY_SPEC_VERSION,
) -> str:
    digest = _contract_digest(spec_version)
    if _PINNED_CONTRACT_HASHES.get(spec_version) != digest:
        raise RuntimeError("mexc_strategy_spec_changed_without_version_bump")
    return digest


def _strategy_spec_registration_for_instance(
    spec: Any,
) -> _StrategySpecVersionRegistration:
    if not any(
        isinstance(spec, registration.spec_type)
        for registration in _STRATEGY_SPEC_VERSION_REGISTRY.values()
    ):
        raise TypeError("spec must be a MexcStrategySpec")
    registration = _strategy_spec_version_registration(spec.spec_version)
    if not isinstance(spec, registration.spec_type):
        if spec.spec_version == MEXC_STRATEGY_SPEC_V2_VERSION:
            raise TypeError("spec must be a MexcStrategySpec")
        raise TypeError("spec type does not match its registered version")
    return registration


def strategy_spec_hash(spec: MexcStrategySpec) -> str:
    registration = _strategy_spec_registration_for_instance(spec)
    return registration.hash_instance(spec)


def strategy_spec_identity(spec: MexcStrategySpec) -> StrategySpecIdentity:
    registration = _strategy_spec_registration_for_instance(spec)
    return StrategySpecIdentity(
        spec_version=registration.spec_version,
        contract_hash=strategy_spec_contract_hash(registration.spec_version),
        instance_hash=registration.hash_instance(spec),
    )


__all__ = [
    "DEFAULT_MEXC_STRATEGY_SPEC_PATH",
    "DEFAULT_MEXC_STRATEGY_V2_SPEC_PATH",
    "MEXC_STRATEGY_SPEC_V2_VERSION",
    "MEXC_STRATEGY_SPEC_VERSION",
    "SUPPORTED_MEXC_STRATEGY_SPEC_VERSIONS",
    "HigherTimeframeParameters",
    "IndicatorParameters",
    "MarketDataParameters",
    "MexcStrategySpec",
    "MexcStrategySpecError",
    "RuntimeSemantics",
    "SignalParameters",
    "StrategySpecIdentity",
    "VolatilityParameters",
    "VolumeProfileParameters",
    "canonical_mexc_interval",
    "decode_mexc_strategy_spec_evidence",
    "load_mexc_strategy_spec",
    "parse_mexc_strategy_spec",
    "strategy_spec_contract_hash",
    "strategy_spec_hash",
    "strategy_spec_identity",
]
