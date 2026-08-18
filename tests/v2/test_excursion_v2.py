"""The excursion stand is only worth its numbers if it cannot see the future.

These pin the properties that make the 2026-08-18 measurement trustworthy rather
than the arithmetic, which is trivial: entry comes strictly after the decision,
observation windows do not overlap, and an ambiguous bar is charged to the stop.
A stand that quietly broke any of these would still produce plausible-looking
distributions, which is exactly why they are asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.excursion import (
    EventSpec,
    event_indices,
    excursion_at,
    first_touch_return,
    matched_random_indices,
)


def _frame(closes: list[float], *, highs=None, lows=None, opens=None) -> pd.DataFrame:
    closes_array = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes_array if opens is None else np.asarray(opens, dtype=float),
            "high": closes_array if highs is None else np.asarray(highs, dtype=float),
            "low": closes_array if lows is None else np.asarray(lows, dtype=float),
            "close": closes_array,
        }
    )


def test_entry_uses_the_next_bar_open_not_the_decision_close() -> None:
    """The decision bar's own close must never become the entry price."""

    frame = _frame(
        closes=[100.0, 100.0, 100.0, 100.0],
        opens=[100.0, 100.0, 50.0, 100.0],
        highs=[100.0, 100.0, 60.0, 100.0],
        lows=[100.0, 100.0, 40.0, 100.0],
    )
    # Deciding at bar 1 enters at the open of bar 2, which is 50.
    result = excursion_at(frame, index=1, horizon=1)
    assert result.adverse == pytest.approx(60.0 / 50.0 - 1.0)
    assert result.favourable == pytest.approx(1.0 - 40.0 / 50.0)


def test_horizon_window_excludes_the_decision_bar() -> None:
    """A spike on the decision bar cannot be counted as an excursion."""

    frame = _frame(
        closes=[100.0, 100.0, 100.0, 100.0],
        opens=[100.0, 100.0, 100.0, 100.0],
        highs=[100.0, 999.0, 100.0, 100.0],
        lows=[100.0, 1.0, 100.0, 100.0],
    )
    result = excursion_at(frame, index=1, horizon=2)
    assert result.adverse == pytest.approx(0.0)
    assert result.favourable == pytest.approx(0.0)


def test_events_never_overlap() -> None:
    """Overlapping windows would measure one move repeatedly and shrink every CI."""

    rng = np.random.default_rng(7)
    # A long noisy ramp produces many candidate bars; the spacing rule must hold.
    base = np.cumsum(rng.normal(0.0, 1.0, 4000)) + 500.0
    frame = _frame(list(base), highs=list(base * 1.05), lows=list(base * 0.95))
    spec = EventSpec(warmup_bars=50)
    horizon = 30

    indices = event_indices(frame, spec, horizon=horizon)
    assert np.all(np.diff(indices) >= horizon)


def test_events_leave_room_for_the_entry_and_the_whole_horizon() -> None:
    rng = np.random.default_rng(11)
    base = np.cumsum(rng.normal(0.0, 1.0, 1500)) + 500.0
    frame = _frame(list(base), highs=list(base * 1.05), lows=list(base * 0.95))
    spec = EventSpec(warmup_bars=50)
    horizon = 40

    for index in event_indices(frame, spec, horizon=horizon):
        assert index + 1 + horizon <= len(frame)
        excursion_at(frame, int(index), horizon=horizon)  # must not raise


def test_a_bar_holding_both_levels_is_charged_the_stop() -> None:
    """The pessimistic tie rule is the only one that cannot flatter a result."""

    frame = _frame(
        closes=[100.0, 100.0, 100.0],
        opens=[100.0, 100.0, 100.0],
        highs=[100.0, 100.0, 110.0],   # stop at +5% is inside this bar
        lows=[100.0, 100.0, 90.0],     # so is the target at -5%
    )
    result = first_touch_return(
        frame, index=1, stop_pct=5.0, target_pct=5.0, horizon=1, cost=0.0
    )
    assert result == pytest.approx(-0.05)


def test_an_earlier_target_beats_a_later_stop() -> None:
    frame = _frame(
        closes=[100.0, 100.0, 100.0, 100.0],
        opens=[100.0, 100.0, 100.0, 100.0],
        highs=[100.0, 100.0, 100.0, 110.0],
        lows=[100.0, 100.0, 94.0, 100.0],
    )
    result = first_touch_return(
        frame, index=1, stop_pct=5.0, target_pct=5.0, horizon=2, cost=0.0
    )
    assert result == pytest.approx(0.05)


def test_an_unresolved_trade_exits_at_the_horizon_close() -> None:
    frame = _frame(
        closes=[100.0, 100.0, 100.0, 98.0],
        opens=[100.0, 100.0, 100.0, 100.0],
        highs=[100.0, 100.0, 101.0, 100.0],
        lows=[100.0, 100.0, 99.0, 97.0],
    )
    result = first_touch_return(
        frame, index=1, stop_pct=20.0, target_pct=20.0, horizon=2, cost=0.0
    )
    assert result == pytest.approx(0.02)   # short from 100, closed at 98


def test_cost_is_charged_on_every_outcome() -> None:
    frame = _frame(
        closes=[100.0] * 4,
        opens=[100.0] * 4,
        highs=[100.0, 100.0, 100.0, 100.0],
        lows=[100.0, 100.0, 94.0, 100.0],
    )
    gross = first_touch_return(
        frame, index=1, stop_pct=5.0, target_pct=5.0, horizon=2, cost=0.0
    )
    net = first_touch_return(
        frame, index=1, stop_pct=5.0, target_pct=5.0, horizon=2, cost=0.00217
    )
    assert gross - net == pytest.approx(0.00217)


def test_control_entries_stay_inside_the_same_usable_range() -> None:
    frame = _frame(list(np.full(1000, 100.0)))
    spec = EventSpec(warmup_bars=100)
    horizon = 25
    picks = matched_random_indices(
        frame, 50, spec=spec, horizon=horizon, rng=np.random.default_rng(3)
    )

    assert picks.size == 50
    assert picks.min() >= 100
    assert picks.max() + 1 + horizon <= len(frame)
    assert np.all(np.diff(picks) > 0)   # sorted, no repeats


def test_a_malformed_spec_is_rejected_rather_than_silently_used() -> None:
    with pytest.raises(ValueError):
        EventSpec(peak_within_bars=99, lookback_bars=24)
    with pytest.raises(ValueError):
        EventSpec(min_fade_fraction=1.5)


def test_a_frame_without_ohlc_is_rejected() -> None:
    with pytest.raises(ValueError):
        excursion_at(pd.DataFrame({"close": [1.0, 2.0]}), index=0, horizon=1)
