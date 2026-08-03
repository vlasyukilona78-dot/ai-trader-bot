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
from backtesting.single_position import (
    ScoredCandidate,
    SinglePositionContractError,
    select_single_position,
)
from trading.market_data.bar_contract import interval_seconds, is_bar_aligned, next_bar_open_ts
from trading.market_data.source_timing import SourceTiming, SourceTimingError
from trading.market_data.universe import SymbolUniverse, UniverseConfig
from trading.metrics.cycle_envelope import CycleEnvelope, CycleEnvelopeError
from trading.metrics.population_journal import PopulationJournalError, make_cycle_id

from v2.test_scan_v2 import _CaptureJournal, _FakeFeed, _FakeStrategy, _FakeUniverse, _Logger, _ohlcv
from v2.test_single_position_contract_v2 import _bars, _contract, _plan

from app.scan import scan_once


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


# --------------------------------------------------------------------------
# 2-4. Cohorts come from cohort_id, never from a wall-clock timestamp.
# --------------------------------------------------------------------------


def _candidate(score: float, symbol: str, *, cohort_id: str, entry_bar_open_ts: float,
               decision_ts: float, exit_ts: float) -> ScoredCandidate:
    from backtesting.single_position import replay_single_short

    plan = _plan(
        symbol=symbol,
        cohort_id=cohort_id,
        decision_ts=decision_ts,
        actionable_ts=entry_bar_open_ts - 250.0,
        entry_bar_open_ts=entry_bar_open_ts,
    )
    result = replay_single_short(
        _bars([(100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0)],
              start_ts=entry_bar_open_ts),
        plan=plan,
        contract=_contract(),
    )
    return ScoredCandidate(score, plan, result.__class__(**{**result.__dict__, "exit_ts": exit_ts}))


def test_one_cycle_with_different_per_symbol_timestamps_is_one_cohort() -> None:
    """Symbols decided milliseconds apart inside a cycle still compete once."""
    winner = _candidate(0.9, "AUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                        decision_ts=900.0, exit_ts=2000.0)
    loser = _candidate(0.6, "BUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                       decision_ts=947.3, exit_ts=2000.0)

    selection = select_single_position([loser, winner], minimum_score=0.5)

    assert [item.result.symbol for item in selection.selected] == ["AUSDT"]
    assert selection.skipped_busy == 1


def test_two_cohorts_sharing_a_timestamp_do_not_merge() -> None:
    """Identical decision clocks are not evidence of a shared cohort."""
    first = _candidate(0.9, "AUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                       decision_ts=900.0, exit_ts=1300.0)
    second = _candidate(0.6, "BUSDT", cohort_id="cycle-2", entry_bar_open_ts=1500.0,
                        decision_ts=900.0, exit_ts=1800.0)

    selection = select_single_position([first, second], minimum_score=0.5)

    # Two cohorts, and the book is free again by the second one.
    assert [item.result.symbol for item in selection.selected] == ["AUSDT", "BUSDT"]
    assert selection.skipped_busy == 0


def test_a_cohort_may_not_carry_two_different_entry_times() -> None:
    left = _candidate(0.9, "AUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                      decision_ts=900.0, exit_ts=2000.0)
    right = _candidate(0.6, "BUSDT", cohort_id="cycle-1", entry_bar_open_ts=1500.0,
                       decision_ts=900.0, exit_ts=2000.0)

    with pytest.raises(SinglePositionContractError, match="cohort_timing_conflict"):
        select_single_position([left, right], minimum_score=0.5)


def test_candidate_requires_the_plan_it_scored() -> None:
    candidate = _candidate(0.9, "AUSDT", cohort_id="cycle-1", entry_bar_open_ts=1200.0,
                           decision_ts=900.0, exit_ts=2000.0)
    other = _plan(symbol="OTHERUSDT", cohort_id="cycle-9")

    with pytest.raises(SinglePositionContractError):
        ScoredCandidate(0.9, other, candidate.result)


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
        strategy_config_hash="b" * 64,
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
    # `refreshed_at` is read before the request and is therefore always earlier;
    # it must never stand in for the moment the data was known.
    assert snapshot.refreshed_at < snapshot.received_at
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
