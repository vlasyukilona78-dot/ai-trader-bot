from __future__ import annotations

from copy import deepcopy
import sys
from unittest.mock import patch

import pytest

from app.scan import _resolve_scan_spec, parse_args, scan_once
from core.indicators import compute_indicators
from core.mexc_strategy_spec import (
    MEXC_STRATEGY_SPEC_VERSION,
    MexcStrategySpec,
    load_mexc_strategy_spec,
    strategy_spec_contract_hash,
)
from core.signal_generator import SignalConfig
from trading.market_data.source_timing import SourceTiming
from trading.metrics.cycle_envelope import (
    CYCLE_ENVELOPE_SCHEMA_VERSION,
    CycleEnvelope,
    CycleEnvelopeError,
)
from trading.metrics.population_journal import SCHEMA_VERSION, make_cycle_id
from trading.signals.layered_strategy import LayeredPumpStrategy
from trading.signals.signal_types import IntentAction
from v2.test_scan_v2 import (
    _CaptureJournal,
    _FakeFeed,
    _FakeStrategy,
    _FakeUniverse,
    _Logger,
    _ohlcv,
)


def _envelope(
    spec: MexcStrategySpec, *, timeframe: str | None = None
) -> CycleEnvelope:
    received_at = 1_700_000_001.0
    cutoff = 1_700_002_800.0
    symbols = ("AAAUSDT",)
    universe_timing = SourceTiming(
        source="universe_ticker",
        request_started_at=1_700_000_000.0,
        received_at=received_at,
        source_as_of=received_at,
        source_ts=received_at,
        status="ok",
    )
    resolved_timeframe = timeframe or spec.market_data.base_interval
    return CycleEnvelope.build(
        cycle_id=make_cycle_id(
            timeframe=resolved_timeframe,
            candle_cutoff_ts=cutoff,
            universe_received_at=received_at,
            universe_symbols=symbols,
            schema_version=SCHEMA_VERSION,
        ),
        timeframe=resolved_timeframe,
        cycle_started_at=cutoff + 1.0,
        candle_cutoff_ts=cutoff,
        universe_symbols=symbols,
        universe_timing=universe_timing,
        source_timings=(universe_timing,),
        strategy_spec_version=spec.spec_version,
        strategy_spec_contract_hash=strategy_spec_contract_hash(),
        strategy_spec_instance_hash=spec.instance_hash,
        strategy_spec_payload=spec.to_mapping(),
        universe_policy_hash="a" * 64,
        ranking_ready_ts=cutoff + 2.0,
        cycle_completed_ts=cutoff + 3.0,
    )


def test_cycle_envelope_v3_round_trip_rebuilds_the_resolved_strategy_spec() -> None:
    spec = load_mexc_strategy_spec()
    envelope = _envelope(spec)

    assert CYCLE_ENVELOPE_SCHEMA_VERSION == 3
    assert envelope.strategy_spec_version == MEXC_STRATEGY_SPEC_VERSION
    assert envelope.strategy_spec_contract_hash == strategy_spec_contract_hash()
    assert envelope.strategy_spec_instance_hash == spec.instance_hash
    assert envelope.strategy_config_hash == spec.instance_hash
    assert envelope.as_dict()["strategy_spec_payload"] == spec.to_mapping()
    assert CycleEnvelope.from_dict(envelope.as_dict()).as_dict() == envelope.as_dict()


def test_cycle_envelope_rejects_a_timeframe_not_executed_by_its_strategy_spec() -> None:
    with pytest.raises(CycleEnvelopeError, match="timeframe_disagrees_with_strategy_spec"):
        _envelope(load_mexc_strategy_spec(), timeframe="Min15")


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload["strategy_spec_payload"]["signal"].__setitem__(
                "rsi_high", 76.0
            ),
            "strategy_spec_instance_hash_mismatch",
        ),
        (
            lambda payload: payload.__setitem__("strategy_spec_contract_hash", "f" * 64),
            "strategy_spec_contract_hash_mismatch",
        ),
        (
            lambda payload: payload["strategy_spec_payload"]["market_data"].__setitem__(
                "base_interval", "60"
            ),
            "strategy_spec_payload_must_be_canonical",
        ),
    ),
)
def test_cycle_envelope_rejects_strategy_spec_tampering(mutate, message) -> None:
    payload = deepcopy(_envelope(load_mexc_strategy_spec()).as_dict())
    mutate(payload)

    with pytest.raises(CycleEnvelopeError, match="invalid_cycle_envelope") as caught:
        CycleEnvelope.from_dict(payload)
    assert isinstance(caught.value.__cause__, CycleEnvelopeError)
    assert message in str(caught.value.__cause__)


def test_scanner_uses_one_bound_spec_for_market_data_strategy_and_evidence() -> None:
    spec = load_mexc_strategy_spec()
    feed = _FakeFeed({"BTCUSDT": _ohlcv(), "AAAUSDT": _ohlcv()})
    journal = _CaptureJournal()
    strategy = _FakeStrategy(IntentAction.HOLD)

    scan_once(
        universe=_FakeUniverse(["AAAUSDT"]),
        feed=feed,
        strategy=strategy,
        strategy_spec=spec,
        logger=_Logger(),
        workers=1,
        population_journal=journal,
    )

    assert [(timeframe, candles) for _, timeframe, candles, _ in feed.closed_requests] == [
        (spec.resolved_benchmark_interval, spec.market_data.base_candles),
        (spec.market_data.base_interval, spec.market_data.base_candles),
    ]
    envelope = journal.envelopes[0]
    assert envelope.strategy_spec_instance_hash == spec.instance_hash
    assert envelope.strategy_spec_payload == spec.to_mapping()
    assert journal.cycles[0][0].metadata["provenance"]["strategy_config_hash"] == (
        spec.instance_hash
    )


def test_runtime_adapters_preserve_the_current_layered_strategy_defaults() -> None:
    spec = load_mexc_strategy_spec()
    legacy = LayeredPumpStrategy()
    resolved = LayeredPumpStrategy(strategy_spec=spec)

    assert resolved.strategy_spec is spec
    assert resolved.configuration_snapshot() == legacy.configuration_snapshot()
    assert resolved._indicator_kwargs == spec.compute_indicators_kwargs()
    assert resolved._generator.indicator_kwargs == spec.compute_indicators_kwargs()
    assert resolved._volume_profile_kwargs == spec.volume_profile_kwargs()
    assert resolved._minimum_history_bars == spec.runtime_semantics.layered_min_history_bars
    assert (
        resolved._generator.min_history_bars
        == spec.runtime_semantics.signal_generator_min_history_bars
    )


def test_custom_executable_fields_reach_every_runtime_adapter() -> None:
    payload = load_mexc_strategy_spec().to_mapping()
    payload["indicators"]["rsi_period_bars"] = 9
    payload["volume_profile"]["minimum_history_bars"] = 31
    payload["volume_profile"]["minimum_sample_bars"] = 32
    payload["runtime_semantics"]["signal_generator_min_history_bars"] = 79
    spec = MexcStrategySpec.from_mapping(payload)
    strategy = LayeredPumpStrategy(strategy_spec=spec)

    assert strategy._indicator_kwargs["rsi_period"] == 9
    assert strategy._generator.indicator_kwargs["rsi_period"] == 9
    assert strategy._volume_profile_kwargs["minimum_history_bars"] == 31
    assert strategy._volume_profile_kwargs["minimum_sample_bars"] == 32
    assert strategy._generator.min_history_bars == 79
    strategy.assert_strategy_spec_consistency(spec)
    with patch("core.indicators.compute_indicators", wraps=compute_indicators) as mocked:
        strategy._generator._layer1c_market_context(
            _ohlcv(), _ohlcv(), htf_frame=_ohlcv()
        )
    assert mocked.call_args.kwargs == spec.compute_indicators_kwargs()


def test_legacy_cli_market_data_flags_are_fail_closed_assertions() -> None:
    spec = load_mexc_strategy_spec()
    strategy = LayeredPumpStrategy(strategy_spec=spec)

    assert (
        _resolve_scan_spec(
            strategy=strategy,
            strategy_spec=spec,
            timeframe="60",
            candles=320,
        )
        is spec
    )
    with pytest.raises(ValueError, match="cli_timeframe_does_not_match_strategy_spec"):
        _resolve_scan_spec(
            strategy=strategy,
            strategy_spec=spec,
            timeframe="15",
            candles=None,
        )
    with pytest.raises(ValueError, match="cli_candles_do_not_match_strategy_spec"):
        _resolve_scan_spec(
            strategy=strategy,
            strategy_spec=spec,
            timeframe=None,
            candles=319,
        )


def test_explicit_spec_rejects_an_unbound_strategy_with_different_runtime_config() -> None:
    spec = load_mexc_strategy_spec()
    strategy = LayeredPumpStrategy(SignalConfig(rsi_high=88.0))

    with pytest.raises(RuntimeError, match="strategy_runtime_drifted_from_strategy_spec"):
        _resolve_scan_spec(
            strategy=strategy,
            strategy_spec=spec,
            timeframe=None,
            candles=None,
        )


def test_legacy_offline_scan_call_gets_a_canonical_generated_spec() -> None:
    resolved = _resolve_scan_spec(
        strategy=_FakeStrategy(),
        strategy_spec=None,
        timeframe="60",
        candles=120,
    )

    assert resolved.market_data.base_interval == "Min60"
    assert resolved.market_data.base_candles == 120
    assert MexcStrategySpec.from_mapping(resolved.to_mapping()).instance_hash == (
        resolved.instance_hash
    )
    assert resolved.instance_hash != load_mexc_strategy_spec().instance_hash


def test_cli_defaults_to_dedicated_spec_and_journal_v5() -> None:
    with patch.object(sys, "argv", ["scan"]):
        args = parse_args()

    assert args.strategy_spec.endswith("config\\mexc_strategy_v2.yaml")
    assert args.timeframe is None
    assert args.candles is None
    assert args.population_journal == "data/runtime/mexc_population_decisions_v5.jsonl"
