from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
import inspect

import pandas as pd
import pytest

import core.mexc_strategy_spec as strategy_spec_module
from core.indicators import compute_indicators
from core.mexc_strategy_spec import (
    DEFAULT_MEXC_STRATEGY_SPEC_PATH,
    MEXC_STRATEGY_SPEC_VERSION,
    MexcStrategySpec,
    MexcStrategySpecError,
    load_mexc_strategy_spec,
    strategy_spec_contract_hash,
    strategy_spec_hash,
)
from core.signal_generator import SignalConfig
from core.volume_profile import compute_volume_profile
from trading.market_data.timeframe_cache import TimeframeCacheConfig
from trading.signals.volatility_context import VolatilityContextConfig


PINNED_CONTRACT_HASH = "9c62b88b7804e9663bae6f0eb429c58c541680b61d307c4f16032cb0b62fe3dd"
PINNED_DEFAULT_INSTANCE_HASH = "9f0b2d7035c2a82ab1b6d8595245b8c3a7a8b9faad17bea8c57f6fcacb189466"


def _mapping() -> dict:
    return load_mexc_strategy_spec().to_mapping()


def test_default_yaml_is_the_actual_mexc_runtime_configuration() -> None:
    spec = load_mexc_strategy_spec()
    signal = SignalConfig()

    assert DEFAULT_MEXC_STRATEGY_SPEC_PATH.is_file()
    assert spec.spec_version == MEXC_STRATEGY_SPEC_VERSION
    assert spec.to_signal_config() == signal
    assert spec.to_volatility_context_config() == VolatilityContextConfig(
        fallback_floor=signal.min_atr_pct
    )
    assert spec.to_timeframe_cache_config() == TimeframeCacheConfig()
    assert spec.market_data.base_interval == "Min60"
    assert spec.market_data.base_candles == 320
    assert spec.resolved_benchmark_interval == "Min60"
    assert spec.runtime_semantics.layered_min_history_bars == 80
    assert spec.runtime_semantics.signal_generator_min_history_bars == 40


def test_all_signal_config_fields_are_captured_exactly_once() -> None:
    spec = load_mexc_strategy_spec()
    runtime_names = tuple(field.name for field in fields(SignalConfig))
    contract_names = tuple(field.name for field in fields(type(spec.signal)))

    assert contract_names == runtime_names
    assert spec.to_mapping()["signal"] == vars(SignalConfig())


def test_implicit_indicator_and_volume_profile_defaults_are_explicit() -> None:
    spec = load_mexc_strategy_spec()
    indicators = spec.indicator_contract()

    assert indicators == {
        "revision": "core_indicators_v1",
        "rsi_period_bars": 14,
        "ema_fast_span_bars": 20,
        "ema_slow_span_bars": 50,
        "atr_period_bars": 14,
        "bollinger_period_bars": 20,
        "bollinger_stddev_multiplier": 2.0,
        "keltner_period_bars": 20,
        "keltner_atr_multiplier": 1.5,
        "volume_ma_period_bars": 20,
        "adx_period_bars": 14,
        "vwap_mode": "cumulative_input_frame",
        "obv_mode": "cumulative_input_frame",
        "cvd_mode": "candle_direction_cumulative_input_frame",
    }
    assert spec.compute_indicators_kwargs() == {
        "rsi_period": 14,
        "ema_fast_span": 20,
        "ema_slow_span": 50,
        "atr_period": 14,
        "bollinger_period": 20,
        "bollinger_mult": 2.0,
        "keltner_period": 20,
        "keltner_mult": 1.5,
        "volume_ma_period": 20,
        "adx_period": 14,
    }
    indicator_signature = inspect.signature(compute_indicators).parameters
    for name, value in spec.compute_indicators_kwargs().items():
        assert indicator_signature[name].default == value
    assert spec.volume_profile_kwargs() == {
        "window": 120,
        "bins": 48,
        "value_area": 0.7,
        "minimum_history_bars": 20,
        "minimum_sample_bars": 24,
    }
    vp_signature = inspect.signature(compute_volume_profile).parameters
    assert vp_signature["window"].default == 120
    assert vp_signature["bins"].default == 48
    assert vp_signature["value_area"].default == 0.70
    assert vp_signature["minimum_history_bars"].default == 20
    assert vp_signature["minimum_sample_bars"].default == 24
    assert spec.volume_profile.minimum_history_bars == 20
    assert spec.volume_profile.minimum_sample_bars == 24


def test_contract_and_default_instance_hashes_are_pinned() -> None:
    spec = load_mexc_strategy_spec()

    assert strategy_spec_contract_hash() == PINNED_CONTRACT_HASH
    assert strategy_spec_hash(spec) == PINNED_DEFAULT_INSTANCE_HASH
    assert spec.instance_hash == PINNED_DEFAULT_INSTANCE_HASH


def test_mapping_order_and_interval_aliases_do_not_change_identity() -> None:
    baseline = load_mexc_strategy_spec()
    payload = _mapping()
    payload = dict(reversed(tuple(payload.items())))
    payload["market_data"] = dict(reversed(tuple(payload["market_data"].items())))
    payload["market_data"]["base_interval"] = "60"
    payload["market_data"]["higher_timeframe"]["interval"] = "240"

    rebuilt = MexcStrategySpec.from_mapping(payload)

    assert rebuilt.market_data.base_interval == "Min60"
    assert rebuilt.market_data.higher_timeframe.interval == "Hour4"
    assert rebuilt.instance_hash == baseline.instance_hash


def test_windows_remain_source_bar_counts_when_interval_changes() -> None:
    hourly = load_mexc_strategy_spec()
    payload = _mapping()
    payload["market_data"]["base_interval"] = "Min15"
    quarter_hour = MexcStrategySpec.from_mapping(payload)

    assert hourly.signal.pump_window_bars == 45
    assert quarter_hour.signal.pump_window_bars == 45
    assert hourly.base_window_seconds(hourly.signal.pump_window_bars) == 45 * 3600
    assert (
        quarter_hour.base_window_seconds(quarter_hour.signal.pump_window_bars)
        == 45 * 900
    )
    assert quarter_hour.instance_hash != hourly.instance_hash
    assert hourly.higher_timeframe_window_seconds(12) == 12 * 4 * 3600


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("market_data", "base_candles", 321),
        ("runtime_semantics", "layered_min_history_bars", 81),
        ("volatility", "percentile", 0.81),
        ("indicators", "rsi_period_bars", 15),
        ("volume_profile", "bins", 49),
    ],
)
def test_each_executable_section_changes_the_instance_hash(
    section: str, field: str, replacement: object
) -> None:
    baseline = load_mexc_strategy_spec()
    payload = _mapping()
    payload[section][field] = replacement

    assert MexcStrategySpec.from_mapping(payload).instance_hash != baseline.instance_hash


def test_custom_indicator_and_volume_profile_values_change_execution() -> None:
    payload = _mapping()
    payload["indicators"]["rsi_period_bars"] = 2
    payload["volume_profile"]["minimum_history_bars"] = 31
    spec = MexcStrategySpec.from_mapping(payload)
    close = [100.0 + index for index in range(30)]
    frame = pd.DataFrame(
        {
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [100.0] * len(close),
        }
    )

    default_indicators = compute_indicators(frame)
    custom_indicators = compute_indicators(
        frame, **spec.compute_indicators_kwargs()
    )
    assert default_indicators["rsi"].first_valid_index() == 14
    assert custom_indicators["rsi"].first_valid_index() == 2
    assert compute_volume_profile(frame) is not None
    assert compute_volume_profile(frame, **spec.volume_profile_kwargs()) is None


def test_every_signal_parameter_is_part_of_instance_identity() -> None:
    baseline = load_mexc_strategy_spec()

    for field in fields(type(baseline.signal)):
        if field.name in {"min_rsi_1h", "require_confluence"}:
            continue
        payload = _mapping()
        original = payload["signal"][field.name]
        if type(original) is bool:
            replacement = not original
        elif type(original) is int:
            replacement = original + 1
        else:
            replacement = float(original) + 0.001
        payload["signal"][field.name] = replacement
        changed = MexcStrategySpec.from_mapping(payload)
        assert changed.instance_hash != baseline.instance_hash, field.name


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("min_rsi_1h", 55.0, "signal.min_rsi_1h_not_implemented"),
        ("require_confluence", 1, "signal.require_confluence_not_implemented"),
    ),
)
def test_unimplemented_signal_knobs_fail_closed(
    field: str, value: object, message: str
) -> None:
    payload = _mapping()
    payload["signal"][field] = value

    with pytest.raises(MexcStrategySpecError, match=message):
        MexcStrategySpec.from_mapping(payload)


def test_spec_is_frozen() -> None:
    spec = load_mexc_strategy_spec()
    with pytest.raises(FrozenInstanceError):
        spec.signal.rsi_high = 80.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda payload: payload["signal"].__setitem__("unknown_threshold", 1.0),
            "signal_unknown_keys",
        ),
        (
            lambda payload: payload["signal"].pop("rsi_high"),
            "signal_missing_keys",
        ),
        (
            lambda payload: payload["signal"].__setitem__("rsi_high", "75.0"),
            "signal.rsi_high_must_be_number",
        ),
        (
            lambda payload: payload["market_data"].__setitem__("base_candles", True),
            "market_data.base_candles_must_be_integer",
        ),
        (
            lambda payload: payload["volatility"].__setitem__("max_age_sec", float("nan")),
            "volatility.max_age_sec_must_be_finite",
        ),
        (
            lambda payload: payload["market_data"].__setitem__(
                "benchmark_interval", "Min60"
            ),
            "market_data.benchmark_interval_must_equal:same_as_base",
        ),
    ],
)
def test_mapping_fails_closed_on_unknown_missing_wrong_type_and_nonfinite(
    mutator, match: str
) -> None:
    payload = _mapping()
    mutator(payload)
    with pytest.raises(MexcStrategySpecError, match=match):
        MexcStrategySpec.from_mapping(payload)


def test_yaml_loader_rejects_duplicate_keys(tmp_path) -> None:
    text = DEFAULT_MEXC_STRATEGY_SPEC_PATH.read_text(encoding="utf-8")
    text = text.replace(
        "  base_candles: 320\n",
        "  base_candles: 320\n  base_candles: 321\n",
        1,
    )
    path = tmp_path / "duplicate.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MexcStrategySpecError, match="yaml_duplicate_key:base_candles"):
        MexcStrategySpec.from_yaml(path)


def test_contract_hash_fails_closed_if_the_pin_does_not_match(monkeypatch) -> None:
    with monkeypatch.context() as scoped:
        scoped.setitem(
            strategy_spec_module._PINNED_CONTRACT_HASHES,
            MEXC_STRATEGY_SPEC_VERSION,
            "0" * 64,
        )
        strategy_spec_contract_hash.cache_clear()
        with pytest.raises(
            RuntimeError, match="mexc_strategy_spec_changed_without_version_bump"
        ):
            strategy_spec_contract_hash()

    strategy_spec_contract_hash.cache_clear()
    assert strategy_spec_contract_hash() == PINNED_CONTRACT_HASH
