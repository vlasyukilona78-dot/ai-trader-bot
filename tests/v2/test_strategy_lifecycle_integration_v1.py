from __future__ import annotations

from types import MappingProxyType

import pandas as pd
import pytest

from core.market_regime import MarketRegime
from core.mexc_strategy_spec import MexcStrategySpec, strategy_spec_identity
from core.signal_generator import (
    SignalConfig,
    SignalContext,
    SignalGenerator,
    SignalResult,
)
from trading.exchange.schemas import AccountSnapshot
from trading.market_data.frame_provenance import (
    FrameRead,
    SourceReadEvidenceV1,
    raw_frame_bundle_hash,
)
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.signals.layered_strategy import LayeredPumpStrategy
from trading.signals.lifecycle_contract import (
    CandidateLifecycleState,
    LifecycleContractError,
    ProposalObservationBasis,
    ProposalObservationStatus,
)
from trading.signals.signal_types import IntentAction
from trading.signals.strategy_interface import StrategyContext
from trading.state.models import TradeState


_ARM_OPEN_TS = 1_700_002_800.0
_TIMEFRAME_SECONDS = 3_600
_SPEC_CONTRACT_HASH = "1" * 64
_SPEC_INSTANCE_HASH = "2" * 64


def _frame(
    elapsed_bars: int,
    *,
    close: float = 100.0,
    high: float | None = None,
    low: float | None = None,
    count: int = 40,
) -> pd.DataFrame:
    end = pd.Timestamp(
        _ARM_OPEN_TS + elapsed_bars * _TIMEFRAME_SECONDS,
        unit="s",
        tz="UTC",
    )
    index = pd.date_range(end=end, periods=count, freq="h")
    closes = [100.0] * count
    closes[-1] = close
    highs = [100.5] * count
    lows = [99.5] * count
    highs[-1] = max(close, high if high is not None else close + 0.25)
    lows[-1] = min(close, low if low is not None else close - 0.25)
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * count,
            "turnover": [10_000.0] * count,
            "atr": [1.0] * count,
            "rsi": [60.0] * count,
            "volume_spike": [1.0] * count,
            "adx": [20.0] * count,
        },
        index=index,
    )


def _signal_context(frame: pd.DataFrame, symbol: str = "ALTUSDT") -> SignalContext:
    return SignalContext(
        symbol=symbol,
        df=frame,
        volume_profile=None,
        regime=MarketRegime.RANGE,
        sentiment_index=75.0,
        sentiment_source="provided",
        funding_rate=0.001,
        long_short_ratio=1.2,
    )


def _typed_evaluate(
    generator: SignalGenerator,
    frame: pd.DataFrame,
    raw_input_bundle_hash: str,
):
    return generator.generate_with_lifecycle(
        _signal_context(frame),
        strategy_spec_version="mexc_strategy_v2",
        strategy_spec_contract_hash=_SPEC_CONTRACT_HASH,
        strategy_spec_instance_hash=_SPEC_INSTANCE_HASH,
        raw_input_bundle_hash=raw_input_bundle_hash,
        timeframe_seconds=_TIMEFRAME_SECONDS,
        candle_cutoff_ts=float(frame.index[-1].timestamp()) + _TIMEFRAME_SECONDS,
    )


def _install_short_gates(monkeypatch, generator: SignalGenerator) -> None:
    layer1 = {"volume_spike": 3.0, "move_pct": 0.08}
    layer2 = {"skipped": 1.0}
    layer3 = {"entry_ok": 1.0}
    layer4 = {"sentiment": 75.0, "crowd_extreme": 1.0}

    def evaluate_gates(df, context, trace):
        trace["layers"].update(
            {
                "layer1_pump_detection": {
                    "passed": True,
                    "side": "SHORT",
                    "details": layer1,
                },
                "layer2_weakness_confirmation": {
                    "passed": True,
                    "details": layer2,
                },
                "layer3_entry_location": {
                    "passed": True,
                    "details": layer3,
                },
                "layer4_fake_filter": {
                    "passed": True,
                    "details": layer4,
                },
            }
        )
        return "SHORT", layer1, layer2, layer3, layer4

    monkeypatch.setattr(generator, "_evaluate_gates", evaluate_gates)


def _install_finalize(
    monkeypatch,
    generator: SignalGenerator,
    *,
    outcome: str = "created",
    signal_id: str = "legacy-clock-dependent-id",
) -> None:
    def finalize(
        df,
        context,
        side,
        layer1,
        layer2,
        layer3,
        layer4,
        trace,
        entry=None,
    ):
        if outcome == "raises":
            raise RuntimeError("synthetic_layer5_failure")
        resolved_entry = float(entry if entry is not None else df.iloc[-1]["close"])
        if outcome == "rejected":
            trace["failed_layer"] = "layer5_stop_too_wide"
            trace["layers"]["layer5_tp_sl"] = {
                "passed": False,
                "details": {
                    "entry": resolved_entry,
                    "sl": resolved_entry + 4.0,
                    "stop_distance_pct": 0.04,
                    "max_stop_distance_pct": 0.03,
                },
            }
            generator.last_diagnostics = trace
            return None
        trace["failed_layer"] = None
        trace["layers"]["layer5_tp_sl"] = {
            "passed": True,
            "details": {
                "entry": resolved_entry,
                "sl": resolved_entry + 2.0,
                "tp": resolved_entry - 4.0,
                "realized_risk_reward": 2.0,
            },
        }
        generator.last_diagnostics = trace
        return SignalResult(
            signal_id=signal_id,
            symbol=context.symbol,
            side=side,
            entry=resolved_entry,
            sl=resolved_entry + 2.0,
            tp=resolved_entry - 4.0,
            confidence=0.8,
        )

    monkeypatch.setattr(generator, "_finalize_signal", finalize)


def _generator(*, max_wait: int = 3, confirmation_enabled: bool = True) -> SignalGenerator:
    return SignalGenerator(
        SignalConfig(
            confirmation_enabled=confirmation_enabled,
            confirmation_max_wait_bars=max_wait,
            confirmation_invalidate_pct=0.05,
            pump_window_enabled=False,
        )
    )


def test_arm_wait_same_bar_confirm_preserves_arm_and_separate_counts(monkeypatch) -> None:
    generator = _generator()
    _install_short_gates(monkeypatch, generator)
    _install_finalize(monkeypatch, generator)

    signal, armed = _typed_evaluate(generator, _frame(0), "a" * 64)
    assert signal is None
    assert armed is not None
    assert armed.state is CandidateLifecycleState.ARMED
    assert isinstance(armed.arm.arm_trace, MappingProxyType)
    assert set(armed.arm.arm_trace["layers"]) == {
        "layer1_pump_detection",
        "layer2_weakness_confirmation",
        "layer3_entry_location",
        "layer4_fake_filter",
    }

    signal, waiting = _typed_evaluate(generator, _frame(1), "b" * 64)
    assert signal is None
    assert waiting is not None
    assert waiting.state is CandidateLifecycleState.WAITING
    assert waiting.confirmation is not None
    assert waiting.confirmation.distinct_observation_count == 1
    assert waiting.confirmation.elapsed_bars == 1

    signal, same_bar = _typed_evaluate(generator, _frame(1), "b" * 64)
    assert signal is None
    assert same_bar is not None
    assert same_bar.state is CandidateLifecycleState.SAME_BAR
    assert same_bar.confirmation is not None
    assert same_bar.confirmation.distinct_observation_count == 1
    assert same_bar.confirmation.elapsed_bars == 1
    assert same_bar.confirmation.observation_id != waiting.confirmation.observation_id

    signal, confirmed = _typed_evaluate(
        generator,
        _frame(2, close=99.0, high=100.0, low=98.5),
        "c" * 64,
    )
    assert signal is not None
    assert confirmed is not None
    assert confirmed.state is CandidateLifecycleState.CONFIRMED
    assert confirmed.state_epoch == 3
    assert confirmed.confirmation is not None
    assert confirmed.confirmation.distinct_observation_count == 2
    assert confirmed.confirmation.elapsed_bars == 2
    assert confirmed.proposal.status is ProposalObservationStatus.CREATED
    assert confirmed.proposal.basis is ProposalObservationBasis.CONFIRMATION
    assert armed.arm is waiting.arm is same_bar.arm is confirmed.arm
    assert "ALTUSDT" not in generator._pending


def test_invalidation_has_priority_over_same_bar_close_confirmation(monkeypatch) -> None:
    generator = _generator()
    _install_short_gates(monkeypatch, generator)
    _install_finalize(monkeypatch, generator)
    _typed_evaluate(generator, _frame(0), "a" * 64)

    signal, invalidated = _typed_evaluate(
        generator,
        _frame(1, close=99.0, high=106.0, low=98.5),
        "b" * 64,
    )

    assert signal is None
    assert invalidated is not None
    assert invalidated.state is CandidateLifecycleState.INVALIDATED
    assert invalidated.proposal.status is ProposalObservationStatus.NOT_EVALUATED
    assert "ALTUSDT" not in generator._pending


def test_distinct_observations_do_not_hide_physical_bar_gaps_or_change_expiry(
    monkeypatch,
) -> None:
    generator = _generator(max_wait=2)
    _install_short_gates(monkeypatch, generator)
    _install_finalize(monkeypatch, generator)
    _typed_evaluate(generator, _frame(0), "a" * 64)

    _, waiting = _typed_evaluate(generator, _frame(3), "b" * 64)
    assert waiting is not None and waiting.confirmation is not None
    assert waiting.state is CandidateLifecycleState.WAITING
    assert waiting.confirmation.distinct_observation_count == 1
    assert waiting.confirmation.elapsed_bars == 3
    assert generator.last_diagnostics["layers"]["layer_confirmation"]["details"][
        "bars_waited"
    ] == 1.0

    _, expired = _typed_evaluate(generator, _frame(5), "c" * 64)
    assert expired is not None and expired.confirmation is not None
    assert expired.state is CandidateLifecycleState.EXPIRED
    assert expired.confirmation.distinct_observation_count == 2
    assert expired.confirmation.elapsed_bars == 5
    assert generator.last_diagnostics["layers"]["layer_confirmation"]["details"][
        "bars_waited"
    ] == 2.0
    assert "ALTUSDT" not in generator._pending


def test_backward_bar_fails_closed_without_rewriting_pending_event(monkeypatch) -> None:
    generator = _generator()
    _install_short_gates(monkeypatch, generator)
    _install_finalize(monkeypatch, generator)
    _typed_evaluate(generator, _frame(0), "a" * 64)
    _, waiting = _typed_evaluate(generator, _frame(2), "b" * 64)
    assert waiting is not None
    pending_before = generator._pending["ALTUSDT"]
    event_before = pending_before.lifecycle_event

    with pytest.raises(LifecycleContractError, match="bar_moved_backward"):
        _typed_evaluate(generator, _frame(1), "c" * 64)

    assert generator._pending["ALTUSDT"] is pending_before
    assert generator._pending["ALTUSDT"].lifecycle_event is event_before


def test_typed_evaluation_refuses_to_fabricate_link_for_legacy_pending(monkeypatch) -> None:
    generator = _generator()
    _install_short_gates(monkeypatch, generator)
    generator.generate(_signal_context(_frame(0)))
    assert generator._pending["ALTUSDT"].lifecycle_event is None

    with pytest.raises(
        LifecycleContractError,
        match="typed_evaluation_cannot_continue_legacy_pending_candidate",
    ):
        _typed_evaluate(generator, _frame(1), "b" * 64)

    assert generator._pending["ALTUSDT"].lifecycle_event is None


def test_typed_pending_is_preserved_when_layer5_raises_before_evidence_commit(
    monkeypatch,
) -> None:
    generator = _generator()
    _install_short_gates(monkeypatch, generator)
    _install_finalize(monkeypatch, generator, outcome="raises")
    _typed_evaluate(generator, _frame(0), "a" * 64)

    with pytest.raises(RuntimeError, match="synthetic_layer5_failure"):
        _typed_evaluate(
            generator,
            _frame(1, close=99.0, high=100.0, low=98.5),
            "b" * 64,
        )

    pending = generator._pending["ALTUSDT"]
    assert pending.lifecycle_event is not None
    assert pending.lifecycle_event.state is CandidateLifecycleState.ARMED
    assert pending.distinct_observation_count == 0
    assert pending.bars_waited == 0


def test_observation_contract_failure_does_not_partially_advance_pending(
    monkeypatch,
) -> None:
    generator = _generator()
    _install_short_gates(monkeypatch, generator)
    _install_finalize(monkeypatch, generator)
    _typed_evaluate(generator, _frame(0), "a" * 64)
    pending_before = generator._pending["ALTUSDT"]
    event_before = pending_before.lifecycle_event
    bad = _frame(1)
    bad.iloc[-1, bad.columns.get_loc("low")] = 101.0

    with pytest.raises(LifecycleContractError, match="observed_close_must_lie"):
        _typed_evaluate(generator, bad, "b" * 64)

    pending_after = generator._pending["ALTUSDT"]
    assert pending_after is pending_before
    assert pending_after.lifecycle_event is event_before
    assert pending_after.distinct_observation_count == 0
    assert pending_after.bars_waited == 0
    _, waiting = _typed_evaluate(generator, _frame(1), "b" * 64)
    assert waiting is not None
    assert waiting.state is CandidateLifecycleState.WAITING


@pytest.mark.parametrize(
    ("outcome", "expected_status", "has_signal"),
    [
        ("created", ProposalObservationStatus.CREATED, True),
        ("rejected", ProposalObservationStatus.REJECTED, False),
    ],
)
def test_confirmation_disabled_emits_bypassed_proposal_outcome(
    monkeypatch,
    outcome,
    expected_status,
    has_signal,
) -> None:
    generator = _generator(confirmation_enabled=False)
    _install_short_gates(monkeypatch, generator)
    _install_finalize(monkeypatch, generator, outcome=outcome)

    signal, event = _typed_evaluate(generator, _frame(0), "a" * 64)

    assert (signal is not None) is has_signal
    assert event is not None
    assert event.state is CandidateLifecycleState.BYPASSED
    assert event.state_epoch == 0
    assert event.confirmation is None
    assert event.proposal.status is expected_status
    assert event.proposal.basis is ProposalObservationBasis.ARM_BYPASS
    assert event.proposal.execution_bound is False
    assert event.proposal.reference_bar_open_ts == event.arm.arm_bar_open_ts
    assert event.proposal.reference_candle_cutoff_ts == event.arm.arm_candle_cutoff_ts


def test_semantic_events_ignore_legacy_signal_clock_identity(monkeypatch) -> None:
    events = []
    for signal_id in ("worker-a-at-clock-1", "worker-b-at-clock-999"):
        generator = _generator()
        _install_short_gates(monkeypatch, generator)
        _install_finalize(monkeypatch, generator, signal_id=signal_id)
        _typed_evaluate(generator, _frame(0), "a" * 64)
        _, event = _typed_evaluate(
            generator,
            _frame(1, close=99.0, high=100.0, low=98.5),
            "b" * 64,
        )
        assert event is not None
        events.append(event)

    assert events[0].arm.candidate_id == events[1].arm.candidate_id
    assert events[0].proposal.proposal_observation_id == events[1].proposal.proposal_observation_id
    assert events[0].event_id == events[1].event_id


def _strategy_context(frame: pd.DataFrame) -> StrategyContext:
    exchange = ExchangeSnapshot(
        symbol="ALTUSDT",
        account=AccountSnapshot(
            equity_usdt=1_000.0,
            available_balance_usdt=1_000.0,
        ),
        positions=[],
        open_orders=[],
    )
    return StrategyContext(
        symbol="ALTUSDT",
        market_ohlcv=frame,
        mark_price=float(frame.iloc[-1]["close"]),
        exchange=exchange,
        synced_state=TradeState.FLAT,
        sentiment_index=75.0,
        sentiment_source="provided",
        funding_rate=0.001,
        long_short_ratio=1.2,
        candle_cutoff_ts=float(frame.index[-1].timestamp()) + _TIMEFRAME_SECONDS,
    )


def _available_read(
    frame: pd.DataFrame,
    *,
    source: str,
    symbol: str,
    timeframe: str,
    requested_as_of_ts: float,
) -> FrameRead:
    evidence = SourceReadEvidenceV1.from_frame(
        frame,
        source=source,
        venue="mexc_contract",
        symbol=symbol,
        venue_symbol=(symbol[:-4] + "_USDT" if symbol.endswith("USDT") else symbol),
        timeframe=timeframe,
        requested_as_of_ts=requested_as_of_ts,
        request_started_at=requested_as_of_ts + 1.0,
        received_at=requested_as_of_ts + 2.0,
        source_ts=requested_as_of_ts + 2.0,
        cache_hit=False,
        cache_age_sec=0.0,
    )
    return FrameRead(frame=frame.copy(deep=True), evidence=evidence)


def _missing_read(
    *,
    source: str,
    symbol: str,
    timeframe: str,
    requested_as_of_ts: float,
) -> FrameRead:
    evidence = SourceReadEvidenceV1.not_requested(
        source=source,
        venue="mexc_contract",
        symbol=symbol,
        venue_symbol=(symbol[:-4] + "_USDT" if symbol.endswith("USDT") else symbol),
        timeframe=timeframe,
        requested_as_of_ts=requested_as_of_ts,
        reason="not_available",
    )
    return FrameRead(frame=None, evidence=evidence)


def _typed_reads(frame: pd.DataFrame, spec: MexcStrategySpec):
    cutoff = float(frame.index[-1].timestamp()) + _TIMEFRAME_SECONDS
    base = _available_read(
        frame,
        source="base_ohlcv",
        symbol="ALTUSDT",
        timeframe=spec.market_data.base_interval,
        requested_as_of_ts=cutoff,
    )
    benchmark = _available_read(
        frame,
        source="benchmark_ohlcv",
        symbol="BTCUSDT",
        timeframe=spec.resolved_benchmark_interval,
        requested_as_of_ts=cutoff,
    )
    htf = _missing_read(
        source="higher_timeframe_ohlcv",
        symbol="ALTUSDT",
        timeframe=spec.market_data.higher_timeframe.interval,
        requested_as_of_ts=cutoff,
    )
    return base, benchmark, htf


def test_public_typed_api_owns_explicit_reads_and_never_reads_mutable_sources(monkeypatch) -> None:
    spec = MexcStrategySpec.from_yaml()
    strategy = LayeredPumpStrategy(strategy_spec=spec)
    calls = {"cache": 0, "contexts": [], "kwargs": []}

    class ForbiddenCache:
        config = spec.to_timeframe_cache_config()

        def get(self, *args, **kwargs):
            calls["cache"] += 1
            raise AssertionError("typed API must not consult the HTF cache")

    strategy.set_htf_cache(ForbiddenCache())
    strategy.set_benchmark(_frame(0, count=20).assign(close=999.0))

    def typed_stub(signal_context, **kwargs):
        calls["contexts"].append(signal_context)
        calls["kwargs"].append(kwargs)
        strategy._generator.last_diagnostics = {
            "failed_layer": "layer1_pump_detection",
            "layers": {},
        }
        return None, None

    monkeypatch.setattr(strategy._generator, "generate_with_lifecycle", typed_stub)
    frame = _frame(0, count=80)
    context = _strategy_context(frame)
    context.market_ohlcv = frame.assign(rsi=999.0, atr=999.0)
    base_read, benchmark_read, htf_read = _typed_reads(frame, spec)

    intent, event = strategy.evaluate_with_lifecycle(
        context,
        base_read=base_read,
        benchmark_read=benchmark_read,
        higher_timeframe_read=htf_read,
    )

    assert intent.action is IntentAction.HOLD
    assert event is None
    assert calls["cache"] == 0
    assert calls["contexts"][0].htf_frame is None
    assert float(calls["contexts"][0].benchmark.iloc[-1]["close"]) == float(
        benchmark_read.frame.iloc[-1]["close"]
    )
    assert calls["contexts"][0].benchmark is not benchmark_read.frame
    assert calls["contexts"][0].df is not base_read.frame
    assert float(calls["contexts"][0].df.iloc[-1]["rsi"]) != 999.0
    assert float(calls["contexts"][0].df.iloc[-1]["atr"]) != 999.0
    assert calls["kwargs"][0]["raw_input_bundle_hash"] == raw_frame_bundle_hash(
        [base_read.evidence, benchmark_read.evidence, htf_read.evidence]
    )
    identity = strategy_spec_identity(spec)
    assert calls["kwargs"][0]["strategy_spec_version"] == identity.spec_version
    assert calls["kwargs"][0]["strategy_spec_contract_hash"] == identity.contract_hash
    assert calls["kwargs"][0]["strategy_spec_instance_hash"] == identity.instance_hash
    with pytest.raises(TypeError, match="higher_timeframe_read"):
        strategy.evaluate_with_lifecycle(
            context,
            base_read=base_read,
            benchmark_read=benchmark_read,
        )


def test_public_typed_api_rejects_mismatched_read_identity() -> None:
    spec = MexcStrategySpec.from_yaml()
    strategy = LayeredPumpStrategy(strategy_spec=spec)
    frame = _frame(0, count=80)
    context = _strategy_context(frame)
    _, benchmark, htf = _typed_reads(frame, spec)
    wrong_base = _available_read(
        frame,
        source="benchmark_ohlcv",
        symbol="ALTUSDT",
        timeframe=spec.market_data.base_interval,
        requested_as_of_ts=context.candle_cutoff_ts,
    )

    with pytest.raises(LifecycleContractError, match="base_read_source_mismatch"):
        strategy.evaluate_with_lifecycle(
            context,
            base_read=wrong_base,
            benchmark_read=benchmark,
            higher_timeframe_read=htf,
        )


def test_public_typed_api_returns_hold_with_confirmed_rejected_proposal(
    monkeypatch,
) -> None:
    spec = MexcStrategySpec.from_yaml()
    strategy = LayeredPumpStrategy(strategy_spec=spec)
    _install_short_gates(monkeypatch, strategy._generator)
    _install_finalize(monkeypatch, strategy._generator, outcome="rejected")
    arm_frame = _frame(0, count=80)
    arm_reads = _typed_reads(arm_frame, spec)

    arm_intent, armed = strategy.evaluate_with_lifecycle(
        _strategy_context(arm_frame),
        base_read=arm_reads[0],
        benchmark_read=arm_reads[1],
        higher_timeframe_read=arm_reads[2],
    )
    confirmation_frame = _frame(1, close=99.0, high=100.0, low=98.5, count=81)
    confirmation_reads = _typed_reads(confirmation_frame, spec)
    rejected_intent, confirmed = strategy.evaluate_with_lifecycle(
        _strategy_context(confirmation_frame),
        base_read=confirmation_reads[0],
        benchmark_read=confirmation_reads[1],
        higher_timeframe_read=confirmation_reads[2],
    )

    assert arm_intent.action is IntentAction.HOLD
    assert armed is not None and armed.state is CandidateLifecycleState.ARMED
    assert rejected_intent.action is IntentAction.HOLD
    assert rejected_intent.reason == "no_signal_layer5_stop_too_wide"
    assert confirmed is not None
    assert confirmed.state is CandidateLifecycleState.CONFIRMED
    assert confirmed.proposal.status is ProposalObservationStatus.REJECTED
    assert confirmed.proposal.execution_bound is False
    assert "lifecycle_event" not in rejected_intent.metadata


@pytest.mark.parametrize(
    ("context_change", "error"),
    [
        ({"candle_cutoff_ts": None}, "requires_finite_candle_cutoff"),
        ({"candle_cutoff_ts": True}, "requires_finite_candle_cutoff"),
        (
            {"candle_cutoff_ts": _ARM_OPEN_TS + 2 * _TIMEFRAME_SECONDS},
            "requested_cutoff_mismatch",
        ),
    ],
)
def test_public_typed_api_rejects_ambiguous_bar_cutoff(context_change, error) -> None:
    spec = MexcStrategySpec.from_yaml()
    strategy = LayeredPumpStrategy(strategy_spec=spec)
    frame = _frame(0, count=80)
    context = _strategy_context(frame)
    reads = _typed_reads(frame, spec)
    for name, value in context_change.items():
        setattr(context, name, value)

    with pytest.raises(LifecycleContractError, match=error):
        strategy.evaluate_with_lifecycle(
            context,
            base_read=reads[0],
            benchmark_read=reads[1],
            higher_timeframe_read=reads[2],
        )
