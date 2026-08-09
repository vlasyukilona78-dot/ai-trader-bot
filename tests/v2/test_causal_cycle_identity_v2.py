"""Phase 1 slice 1: causal time and cycle/cohort identity.

Two defects motivated this contract. Cohorts were reconstructed from equality of
a float `decision_ts`, so worker latency decided which candidates competed; and
the universe snapshot was dated with a timestamp read before its own request,
which claimed knowledge the process did not have. Both are proved closed here.
"""

from __future__ import annotations

import json
import time

import pytest

from ai.reversal.population_dataset import PopulationDatasetError, population_feature_records
from core.mexc_strategy_spec import (
    MEXC_STRATEGY_SPEC_VERSION,
    load_mexc_strategy_spec,
    strategy_spec_contract_hash,
)
from backtesting.single_position import (
    ScoredCandidate,
    SinglePositionContractError,
    build_replay_evidence,
    select_single_position,
)
from trading.market_data.bar_contract import interval_seconds, is_bar_aligned, next_bar_open_ts
from trading.market_data.source_timing import SourceTiming, SourceTimingError
from trading.market_data.universe import SymbolUniverse, UniverseConfig
from trading.metrics.cycle_envelope import CycleEnvelope, CycleEnvelopeError
from trading.metrics.population_journal import (
    CYCLE_IDENTITY_VERSION,
    SCHEMA_VERSION,
    PopulationJournalError,
    make_cycle_id,
)

from v2.test_scan_v2 import _CaptureJournal, _FakeFeed, _FakeStrategy, _FakeUniverse, _Logger, _ohlcv
from v2.test_single_position_contract_v2 import _ENTRY_BAR_OPEN_TS, _bars, _contract, _plan

from app.scan import scan_once


_STRATEGY_SPEC = load_mexc_strategy_spec()


# --------------------------------------------------------------------------
# 1. Worker order and latency must not change cycle or cohort identity.
# --------------------------------------------------------------------------


class _SlowFeed(_FakeFeed):
    """Finishes symbols in a different order than they were submitted."""

    def __init__(self, frames: dict, delays: dict):
        super().__init__(frames)
        self.delays = delays
        self.completion_order: list[str] = []

    def fetch_closed_frame(self, symbol, timeframe, candles, *, as_of):
        time.sleep(self.delays.get(symbol, 0.0))
        frame = super().fetch_closed_frame(symbol, timeframe, candles, as_of=as_of)
        self.completion_order.append(symbol)
        return frame


def _run_scan(workers: int):
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    # BTC is fetched first and must not be delayed into the middle of the pass.
    delays = {"AAAUSDT": 0.06, "BBBUSDT": 0.02, "CCCUSDT": 0.0}
    feed = _SlowFeed({symbol: _ohlcv() for symbol in symbols + ["BTCUSDT"]}, delays)
    journal = _CaptureJournal()
    journal.enabled = True
    scan_once(
        universe=_FakeUniverse(symbols),
        feed=feed,
        strategy=_FakeStrategy(),
        logger=_Logger(),
        timeframe="60",
        candles=120,
        workers=workers,
        population_journal=journal,
    )
    return feed, journal.cycles[0]


def test_worker_order_does_not_change_cycle_or_cohort_identity() -> None:
    feed, records = _run_scan(workers=4)

    # The pass really did finish out of submission order, so the guarantee below
    # is not vacuous.
    assert feed.completion_order[-3:] != ["AAAUSDT", "BBBUSDT", "CCCUSDT"]

    # Ordinals follow the universe, not whoever answered first.
    assert [record.symbol for record in records] == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    assert len({record.cycle_id for record in records}) == 1
    # Every cycle-level instant is one fact shared by the whole cohort.
    for field in (
        "ranking_ready_ts",
        "cycle_completed_ts",
        "actionable_ts",
        "entry_eligible_ts",
        "entry_bar_open_ts",
        "universe_received_at",
    ):
        assert len({getattr(record, field) for record in records}) == 1, field


def test_single_and_multi_worker_passes_agree_on_cohort_shape() -> None:
    _, single = _run_scan(workers=1)
    _, parallel = _run_scan(workers=4)

    assert [r.symbol for r in single] == [r.symbol for r in parallel]
    assert [r.cycle_ordinal for r in single] == [r.cycle_ordinal for r in parallel]


def test_cycle_id_ignores_everything_a_worker_could_influence() -> None:
    base = dict(
        timeframe="60",
        candle_cutoff_ts=1_700_002_800.0,
        universe_received_at=1_700_000_000.0,
        universe_symbols=["AAAUSDT", "BBBUSDT"],
    )
    assert make_cycle_id(**base) == make_cycle_id(**base)
    reordered = dict(base, universe_symbols=["BBBUSDT", "AAAUSDT"])
    assert make_cycle_id(**reordered) != make_cycle_id(**base)


def test_cycle_identity_is_pinned_independently_from_journal_serialization(
    monkeypatch,
) -> None:
    assert CYCLE_IDENTITY_VERSION == 5
    assert SCHEMA_VERSION == 5
    values = {
        "timeframe": "60",
        "candle_cutoff_ts": 1_700_002_800.0,
        "universe_received_at": 1_700_000_001.0,
        "universe_symbols": ["AAAUSDT", "BBBUSDT"],
    }
    expected = make_cycle_id(**values)
    assert expected == make_cycle_id(
        **values,
        schema_version=CYCLE_IDENTITY_VERSION,
    )
    monkeypatch.setattr(
        "trading.metrics.population_journal.SCHEMA_VERSION",
        SCHEMA_VERSION + 1,
    )
    assert make_cycle_id(**values) == expected
    assert make_cycle_id(
        **values,
        schema_version=CYCLE_IDENTITY_VERSION + 1,
    ) != expected


# --------------------------------------------------------------------------
# 2-4. Cohorts come from cohort_id, never from a wall-clock timestamp.
# --------------------------------------------------------------------------


def _candidate(score: float, symbol: str, *, cohort_id: str, entry_bar_open_ts: float,
               decision_ts: float,
               actionable_ts: float | None = None) -> ScoredCandidate:
    from backtesting.single_position import replay_single_short

    plan = _plan(
        symbol=symbol,
        cohort_id=cohort_id,
        decision_ts=decision_ts,
        actionable_ts=entry_bar_open_ts - 250.0 if actionable_ts is None else actionable_ts,
        entry_eligible_ts=entry_bar_open_ts - 240.0,
        entry_bar_open_ts=entry_bar_open_ts,
    )
    contract = _contract()
    bars = _bars(
        [(100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0)],
        start_ts=entry_bar_open_ts,
    )
    result = replay_single_short(
        bars,
        plan=plan,
        contract=contract,
    )
    evidence = build_replay_evidence(bars, plan=plan, contract=contract)
    return ScoredCandidate(score, plan, contract, evidence, result)


def test_one_cycle_with_different_per_symbol_timestamps_is_one_cohort() -> None:
    """Symbols decided milliseconds apart inside a cycle still compete once."""
    winner = _candidate(0.9, "AUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                        decision_ts=900.0)
    loser = _candidate(0.6, "BUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                       decision_ts=947.3)

    selection = select_single_position([loser, winner], minimum_score=0.5)

    assert [item.result.symbol for item in selection.selected] == ["AUSDT"]
    assert selection.skipped_busy == 1


def test_two_cohorts_sharing_a_timestamp_do_not_merge() -> None:
    """Identical decision clocks are not evidence of a shared cohort."""
    first = _candidate(0.9, "AUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                       decision_ts=900.0)
    second = _candidate(0.6, "BUSDT", cohort_id="cycle-2", entry_bar_open_ts=1800.0,
                        decision_ts=900.0)

    selection = select_single_position([first, second], minimum_score=0.5)

    # Two cohorts, and the book is free again by the second one.
    assert [item.result.symbol for item in selection.selected] == ["AUSDT", "BUSDT"]
    assert selection.skipped_busy == 0


def test_a_cohort_may_not_carry_two_different_entry_times() -> None:
    left = _candidate(0.9, "AUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                      decision_ts=900.0)
    right = _candidate(0.6, "BUSDT", cohort_id="cycle-1", entry_bar_open_ts=1500.0,
                       decision_ts=900.0)

    with pytest.raises(SinglePositionContractError, match="cohort_timing_conflict"):
        select_single_position([left, right], minimum_score=0.5)


def test_candidate_requires_the_plan_it_scored() -> None:
    candidate = _candidate(0.9, "AUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                           decision_ts=900.0)
    other = _plan(symbol="OTHERUSDT", cohort_id="cycle-9")

    with pytest.raises(SinglePositionContractError):
        ScoredCandidate(
            0.9,
            other,
            candidate.contract,
            candidate.evidence,
            candidate.result,
        )


# --------------------------------------------------------------------------
# 5-6. Temporal invariants fail closed; the entry bar always follows the decision.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_started_at": 100.0, "received_at": 99.0},
        {"request_started_at": 100.0, "received_at": 101.0, "source_as_of": 200.0},
        {"request_started_at": float("nan"), "received_at": 100.0},
    ],
)
def test_source_timing_rejects_impossible_orderings(kwargs) -> None:
    with pytest.raises(SourceTimingError):
        SourceTiming(source="universe", **kwargs)


def test_failed_source_still_records_the_time_the_cycle_waited() -> None:
    timing = SourceTiming(
        source="benchmark",
        request_started_at=100.0,
        received_at=105.0,
        status="error",
        error_code="TimeoutError",
    )
    assert not timing.ok
    assert timing.received_at == 105.0


@pytest.mark.parametrize("interval", ["60", "Min5", "Hour4"])
@pytest.mark.parametrize("offset", [0.0, 0.001, 1.0, 59.0, 3599.0])
def test_entry_bar_always_opens_strictly_after_the_decision(interval, offset) -> None:
    seconds = interval_seconds(interval)
    actionable = 1_700_000_000.0 + offset
    entry = next_bar_open_ts(actionable, interval)

    assert entry > actionable
    assert is_bar_aligned(entry, interval)
    assert entry - actionable <= seconds


def test_a_decision_made_exactly_on_a_boundary_waits_for_the_next_bar() -> None:
    boundary = 1_700_002_800.0
    assert is_bar_aligned(boundary, "60")
    assert next_bar_open_ts(boundary, "60") == boundary + 3600.0


def _envelope(**overrides):
    universe = SourceTiming(source="universe", request_started_at=90.0, received_at=100.0)
    base = dict(
        cycle_id="a" * 64,
        timeframe="60",
        cycle_started_at=1_700_002_900.0,
        candle_cutoff_ts=1_700_002_800.0,
        universe_symbols=("AAAUSDT",),
        universe_timing=universe,
        source_timings=(universe,),
        strategy_spec_version=MEXC_STRATEGY_SPEC_VERSION,
        strategy_spec_contract_hash=strategy_spec_contract_hash(),
        strategy_spec_instance_hash=_STRATEGY_SPEC.instance_hash,
        strategy_spec_payload=_STRATEGY_SPEC.to_mapping(),
        universe_policy_hash="c" * 64,
        ranking_ready_ts=1_700_002_950.0,
        cycle_completed_ts=1_700_002_960.0,
    )
    base.update(overrides)
    return CycleEnvelope.build(**base)


def test_envelope_derives_actionable_and_entry_instead_of_trusting_arithmetic() -> None:
    envelope = _envelope()

    assert envelope.actionable_ts == 1_700_002_950.0
    assert envelope.entry_eligible_ts == 1_700_002_960.0
    assert envelope.entry_bar_open_ts == 1_700_006_400.0
    assert envelope.entry_bar_open_ts > envelope.actionable_ts


def test_envelope_rejects_a_cycle_sealed_before_it_was_ranked() -> None:
    with pytest.raises(CycleEnvelopeError, match="cycle_completed_ts_precedes"):
        _envelope(cycle_completed_ts=1_700_002_940.0)


def test_envelope_rejects_ranking_that_precedes_a_source_response() -> None:
    late = SourceTiming(source="universe", request_started_at=90.0, received_at=1_700_003_000.0)
    with pytest.raises(CycleEnvelopeError, match="ranking_ready_ts_precedes"):
        _envelope(universe_timing=late, source_timings=(late,))


def test_envelope_records_an_empty_universe_rather_than_vanishing() -> None:
    envelope = _envelope(universe_symbols=(), status="empty_universe")
    assert envelope.status == "empty_universe"
    assert envelope.universe_symbols == ()


# --------------------------------------------------------------------------
# 7. The universe response time is measured, not assumed.
# --------------------------------------------------------------------------


class _TimedClient:
    def __init__(self):
        self.called_at = 0.0

    def fetch_all_tickers(self, force: bool = False):
        time.sleep(0.05)
        self.called_at = time.time()
        return [
            {
                "symbol": "AAA_USDT",
                "amount24": 1_000_000.0,
                "riseFallRate": 0.2,
                "lastPrice": 1.0,
                "fundingRate": 0.0001,
                "holdVol": 10.0,
            }
        ]


def test_universe_received_at_is_taken_after_the_response() -> None:
    client = _TimedClient()
    universe = SymbolUniverse(client, UniverseConfig(min_turnover_24h_usdt=1.0))

    snapshot = universe.refresh()

    assert snapshot.symbols == ["AAAUSDT"]
    assert snapshot.received_at >= client.called_at
    # The TTL anchor is the completed successful refresh, never the instant read
    # before sending the request.
    assert snapshot.refreshed_at == snapshot.received_at
    assert snapshot.request_started_at <= snapshot.received_at


# --------------------------------------------------------------------------
# 8. Schema drift is detected instead of silently reinterpreted.
# --------------------------------------------------------------------------


def test_reader_fails_closed_on_a_previous_schema_version(tmp_path) -> None:
    from v2.test_population_feature_dataset_v2 import _records

    row = _records()[0].as_dict()
    row["schema_version"] = 1
    path = tmp_path / "population.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(PopulationDatasetError, match="unsupported_population_schema_version"):
        population_feature_records(path)


def test_journal_rejects_an_entry_bar_that_does_not_follow_the_decision() -> None:
    from v2.test_population_journal_v2 import _record

    with pytest.raises(PopulationJournalError, match="entry bar does not open after"):
        _record(entry_bar_open_ts=1_700_000_100.0)


def test_journal_rejects_an_unaligned_entry_bar() -> None:
    from v2.test_population_journal_v2 import _record

    with pytest.raises(PopulationJournalError, match="not aligned"):
        _record(entry_bar_open_ts=1_700_000_401.0)


def test_journal_rejects_a_universe_response_before_its_own_request() -> None:
    from v2.test_population_journal_v2 import _record

    with pytest.raises(PopulationJournalError, match="precedes its own request"):
        _record(universe_request_started_at=1_700_000_050.0)


# --------------------------------------------------------------------------
# Slice A: defects an adversarial review found in the first Phase 1 attempt.
# --------------------------------------------------------------------------


def test_a_refresh_that_crosses_a_bar_boundary_does_not_move_the_cutoff() -> None:
    """The cutoff is frozen before the universe request.

    Deriving it afterwards let a slow refresh produce a cutoff later than the
    cycle's own start, which both claims data the cycle could not have had and
    trips the envelope invariant.
    """
    symbols = ["AAAUSDT"]
    feed = _FakeFeed({symbol: _ohlcv() for symbol in symbols + ["BTCUSDT"]})
    journal = _CaptureJournal()
    journal.enabled = True

    boundary = 1_700_006_400.0
    clock = iter([boundary - 0.1, boundary + 0.1, boundary + 0.2, boundary + 0.3])
    real_time = time.time

    def fake_time():
        try:
            return next(clock)
        except StopIteration:
            return boundary + 0.4

    import app.scan as scan_module

    original = scan_module.time.time
    scan_module.time.time = fake_time
    try:
        scan_once(
            universe=_FakeUniverse(symbols),
            feed=feed,
            strategy=_FakeStrategy(),
            logger=_Logger(),
            timeframe="60",
            candles=120,
            workers=1,
            population_journal=journal,
        )
    finally:
        scan_module.time.time = original
        assert scan_module.time.time is real_time

    record = journal.cycles[0][0]
    assert record.candle_cutoff_ts <= boundary - 0.1
    assert record.candle_cutoff_ts == boundary - 3600.0


def test_entry_bar_must_be_the_first_reachable_one() -> None:
    """A later aligned bar is a delay, and a delayed entry measures a different
    trade than the one the signal proposed."""
    from backtesting.single_position import replay_single_short

    delayed = _plan(entry_bar_open_ts=_ENTRY_BAR_OPEN_TS + 300.0)
    with pytest.raises(SinglePositionContractError, match="first_reachable_bar"):
        replay_single_short(
            _bars(
                [(100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0)],
                start_ts=_ENTRY_BAR_OPEN_TS + 300.0,
            ),
            plan=delayed,
            contract=_contract(),
        )


def test_plan_requires_eligibility_not_only_actionability() -> None:
    with pytest.raises(SinglePositionContractError, match="entry_eligible_ts_precedes"):
        _plan(actionable_ts=960.0, entry_eligible_ts=950.0)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("decision_ts", 901.0, "result_hash_mismatch"),
        ("entry_bar_open_ts", 1500.0, "entry_ts_differs_from_entry_bar"),
    ],
)
def test_candidate_rejects_a_result_from_a_different_plan(field, value, error) -> None:
    candidate = _candidate(0.9, "AUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                           decision_ts=900.0)

    # Result-level chronology now rejects an internally inconsistent entry bar
    # before ScoredCandidate needs to compare it with the plan.
    with pytest.raises(SinglePositionContractError, match=error):
        mismatched = candidate.result.__class__(**{**candidate.result.__dict__, field: value})
        ScoredCandidate(
            0.9,
            candidate.plan,
            candidate.contract,
            candidate.evidence,
            mismatched,
        )


def test_candidate_rejects_a_fill_that_did_not_happen_on_the_planned_bar() -> None:
    candidate = _candidate(0.9, "AUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                           decision_ts=900.0)
    assert candidate.result.filled
    with pytest.raises(SinglePositionContractError, match="entry_ts_differs_from_entry_bar"):
        moved = candidate.result.__class__(**{**candidate.result.__dict__, "entry_ts": 1500.0})
        ScoredCandidate(
            0.9,
            candidate.plan,
            candidate.contract,
            candidate.evidence,
            moved,
        )


def test_same_entry_bar_is_reserved_by_the_earliest_actionable_cohort() -> None:
    """Two cycles can target one bar. The earlier decision keeps it; ordering by
    cohort_id would decide it by SHA order, which means nothing causally."""
    early = _candidate(0.5, "EARLYUSDT", cohort_id="zzz-late-hash", entry_bar_open_ts=1200.0,
                       decision_ts=900.0, actionable_ts=940.0)
    late = _candidate(0.9, "LATEUSDT", cohort_id="aaa-early-hash", entry_bar_open_ts=1200.0,
                      decision_ts=900.0, actionable_ts=950.0)

    selection = select_single_position([late, early], minimum_score=0.1)

    # The higher score does not win across cohorts: the slot was already taken.
    assert [item.result.symbol for item in selection.selected] == ["EARLYUSDT"]


def test_an_unfilled_earlier_cohort_does_not_hand_the_bar_to_a_later_one() -> None:
    from backtesting.single_position import replay_single_short

    invalid_plan = _plan(symbol="EARLYUSDT", cohort_id="cycle-early")
    invalid_bars = _bars(
        [(106.0, 107.0, 105.5, 106.0), (106.0, 107.0, 105.0, 106.0)]
    )
    invalid = replay_single_short(
        invalid_bars,
        plan=invalid_plan,
        contract=_contract(),
    )
    invalid_evidence = build_replay_evidence(
        invalid_bars, plan=invalid_plan, contract=_contract()
    )
    later = _candidate(0.9, "LATEUSDT", cohort_id="cycle-late", entry_bar_open_ts=1200.0,
                       decision_ts=900.0, actionable_ts=955.0)

    selection = select_single_position(
        [
            ScoredCandidate(
                0.95, invalid_plan, _contract(), invalid_evidence, invalid
            ),
            later,
        ],
        minimum_score=0.1,
    )

    # The earlier cohort's entry simply did not happen; that is not permission for
    # a later cohort to take the same bar using the knowledge that it failed.
    assert selection.selected == ()
    assert selection.skipped_unfilled == 1


def test_cold_start_volatility_floor_is_invariant_to_worker_order() -> None:
    """The first sweep of a fresh process must hold the fallback floor for every
    symbol. Testing the frozen list for emptiness let it fall through to the live
    observations being written by the sweep itself."""
    from core.signal_generator import SignalConfig
    from trading.signals.layered_strategy import LayeredPumpStrategy

    symbols = [f"S{index:03d}USDT" for index in range(28)]
    floors: list[list[float]] = []

    for order in (symbols, list(reversed(symbols))):
        strategy = LayeredPumpStrategy(SignalConfig())
        strategy.begin_sweep()
        seen = []
        for index, symbol in enumerate(order):
            # A spread of volatilities, so a leaking floor would visibly move.
            strategy._volatility.observe(symbol, 0.001 + index * 0.004)
            seen.append(strategy._volatility.floor())
        floors.append(seen)

    fallback = SignalConfig().min_atr_pct
    assert all(value == fallback for run in floors for value in run)
    assert floors[0] == floors[1]


def test_a_completed_sweep_freezes_the_next_one_at_its_own_distribution() -> None:
    from trading.signals.volatility_context import VolatilityContext, VolatilityContextConfig

    context = VolatilityContext(VolatilityContextConfig(min_observations=5, fallback_floor=0.5))
    context.begin = None  # guard against accidental API drift
    context.start_sweep()
    for index in range(10):
        context.observe(f"S{index}", 0.01 + index * 0.01)
    # Still the first sweep: the floor may not react to its own observations.
    assert context.floor() == 0.5

    context.start_sweep()
    frozen = context.floor()
    assert frozen != 0.5
    context.observe("LATE", 99.0)
    assert context.floor() == frozen
