"""The scheduler's job is to make the collected population analysable.

Speed and throughput are not what these check. They check the two properties
that decide whether the resulting dataset can answer anything at all: a triggered
episode never lands without its control, and nothing is ever refused silently.
Both failures would be invisible in the collected files and fatal in the analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from trading.market_data.episode_scheduler import (
    EpisodeKind,
    EpisodeScheduler,
    RefusalReason,
)


def _scheduler(capacity: int = 4, episode_seconds: int = 1800) -> EpisodeScheduler:
    return EpisodeScheduler(
        capacity=capacity,
        episode_seconds=episode_seconds,
        control_delay_range=(300, 600),
        rng=np.random.default_rng(1),
    )


def test_a_triggered_episode_never_lands_without_its_control() -> None:
    scheduler = _scheduler()
    result = scheduler.offer("AAA", 1_000, control_symbol="ZZZ")

    assert result.admitted
    assert result.triggered.kind is EpisodeKind.TRIGGERED
    assert result.control.kind is EpisodeKind.CONTROL
    kinds = [e.kind for e in scheduler.admitted]
    assert kinds.count(EpisodeKind.TRIGGERED) == kinds.count(EpisodeKind.CONTROL)


def test_the_pair_is_refused_as_a_unit_when_only_one_slot_is_free() -> None:
    """Half a pair is worse than nothing: it is a sample with no comparison."""

    scheduler = _scheduler(capacity=3)
    assert scheduler.offer("AAA", 1_000, control_symbol="ZZZ").admitted

    result = scheduler.offer("BBB", 1_100, control_symbol="YYY")
    assert not result.admitted
    assert result.triggered is None and result.control is None
    assert {r.reason for r in result.refusals} == {RefusalReason.NO_CAPACITY}


def test_every_refusal_is_recorded_with_a_reason() -> None:
    """A refusal that leaves no trace is indistinguishable from a quiet market."""

    scheduler = _scheduler(capacity=2)
    scheduler.offer("AAA", 1_000, control_symbol="ZZZ")
    for index in range(5):
        scheduler.offer("SYM%d" % index, 1_000 + index, control_symbol="CTRL")

    assert len(scheduler.refusals) == 10          # five pairs, both halves logged
    assert all(r.reason is RefusalReason.NO_CAPACITY for r in scheduler.refusals)
    assert scheduler.refusal_rate() == pytest.approx(10 / 11)


def test_capacity_returns_once_episodes_have_finished() -> None:
    scheduler = _scheduler(capacity=2, episode_seconds=600)
    assert scheduler.offer("AAA", 1_000, control_symbol="ZZZ").admitted
    assert not scheduler.offer("BBB", 1_200, control_symbol="YYY").admitted

    # Both the trigger and its later control have ended by now.
    assert scheduler.offer("BBB", 5_000, control_symbol="YYY").admitted
    assert scheduler.in_flight == 2


def test_one_symbol_is_never_recorded_twice_at_once() -> None:
    scheduler = _scheduler()
    scheduler.offer("AAA", 1_000, control_symbol="ZZZ")

    result = scheduler.offer("AAA", 1_100, control_symbol="YYY")
    assert not result.admitted
    assert result.refusals[0].reason is RefusalReason.SYMBOL_ALREADY_RECORDING


def test_a_busy_control_symbol_refuses_the_pair_rather_than_dropping_it() -> None:
    scheduler = _scheduler()
    scheduler.offer("AAA", 1_000, control_symbol="ZZZ")

    result = scheduler.offer("BBB", 1_050, control_symbol="ZZZ")
    assert not result.admitted
    assert result.refusals[0].reason is RefusalReason.CONTROL_UNPLACEABLE


def test_the_control_is_placed_later_than_the_trigger_and_inside_the_range() -> None:
    scheduler = _scheduler()
    result = scheduler.offer("AAA", 10_000, control_symbol="ZZZ")

    offset = result.control.start_ts - result.triggered.start_ts
    assert 300 <= offset < 600


def test_the_same_seed_produces_the_same_schedule() -> None:
    """Collection has to be reproducible from its recorded seed."""

    def run() -> list[int]:
        scheduler = _scheduler(capacity=20)
        return [
            scheduler.offer("S%d" % i, 1_000 + i * 10, control_symbol="C%d" % i).control.start_ts
            for i in range(5)
        ]

    assert run() == run()


def test_a_scheduler_too_small_to_hold_a_pair_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        EpisodeScheduler(capacity=1, episode_seconds=600)
    with pytest.raises(ValueError):
        EpisodeScheduler(capacity=4, episode_seconds=0)
    with pytest.raises(ValueError):
        EpisodeScheduler(capacity=4, episode_seconds=600, control_delay_range=(600, 300))


def test_an_episode_cannot_end_before_it_starts() -> None:
    from trading.market_data.episode_scheduler import Episode

    with pytest.raises(ValueError):
        Episode("AAA", EpisodeKind.TRIGGERED, 1_000, 1_000)
