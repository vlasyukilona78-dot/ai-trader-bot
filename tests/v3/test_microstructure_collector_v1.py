"""The collector's job is to know when the book stopped being trustworthy.

These check the accounting around breaks, which is where a collector quietly
fails: a reconnect that loses nothing must not be recorded as a loss, a reconnect
that loses an update must be, and an episode that ends mid-break must not be
handed out as if it were whole.

One test asserts the module contains no address. That is not style — the depth
and tape stream endpoints are unverified, and a plausible-looking URL committed
here would be believed later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trading.market_data.microstructure_collector import (
    BreakFrame,
    DepthFrame,
    TradeFrame,
    collect_episode,
)
from trading.market_data.microstructure_store import (
    GapReason,
    MicrostructureContractError,
    Side,
)

BIDS = ((100.0, 5.0), (99.9, 7.0))
ASKS = ((100.1, 4.0), (100.2, 6.0))


def _depth(ts_us: int, sequence: int) -> DepthFrame:
    return DepthFrame(ts_us=ts_us, sequence=sequence, bids=BIDS, asks=ASKS)


def _collect(frames):
    return collect_episode(
        frames, symbol="AAAUSDT", kind="triggered", start_ts_us=1_000, end_ts_us=9_000
    )


def test_an_unbroken_stream_records_no_gaps() -> None:
    episode = _collect([_depth(1_100, 1), _depth(1_200, 2), _depth(1_300, 3)])
    assert episode.continuous
    assert len(episode.snapshots) == 3


def test_a_sequence_jump_without_any_warning_is_still_recorded() -> None:
    episode = _collect([_depth(1_100, 1), _depth(1_200, 5)])
    assert len(episode.gaps) == 1
    assert episode.gaps[0].reason is GapReason.SEQUENCE_JUMP
    assert (episode.gaps[0].last_sequence, episode.gaps[0].next_sequence) == (1, 5)


def test_a_reconnect_is_closed_with_real_sequence_numbers() -> None:
    """The missed range is unknown until the stream resumes, so it waits."""

    episode = _collect([
        _depth(1_100, 1),
        BreakFrame(ts_us=1_150, reason=GapReason.RECONNECT),
        _depth(1_200, 9),
    ])
    gap = episode.gaps[0]
    assert gap.reason is GapReason.RECONNECT
    assert gap.ts_us == 1_150                    # when the break began
    assert (gap.last_sequence, gap.next_sequence) == (1, 9)


def test_a_reconnect_that_lost_nothing_is_not_recorded_as_a_loss() -> None:
    """Overstating breaks would discard usable episodes for no reason."""

    episode = _collect([
        _depth(1_100, 1),
        BreakFrame(ts_us=1_150, reason=GapReason.RECONNECT),
        _depth(1_200, 2),
    ])
    assert episode.continuous


def test_a_break_before_any_data_loses_nothing() -> None:
    episode = _collect([BreakFrame(ts_us=1_050, reason=GapReason.SUBSCRIBE_LOST), _depth(1_100, 7)])
    assert episode.continuous
    assert len(episode.snapshots) == 1


def test_an_episode_ending_inside_a_break_is_refused() -> None:
    """Its final stretch is unaccounted for, so it is not a whole episode."""

    with pytest.raises(MicrostructureContractError) as caught:
        _collect([_depth(1_100, 1), BreakFrame(ts_us=1_500, reason=GapReason.RECONNECT)])
    assert "ends_inside_a_break" in str(caught.value)


def test_a_frame_outside_the_declared_window_is_refused_not_skipped() -> None:
    with pytest.raises(MicrostructureContractError) as caught:
        _collect([_depth(1_100, 1), _depth(50_000, 2)])
    assert "outside_window" in str(caught.value)


def test_a_sequence_going_backwards_is_refused() -> None:
    with pytest.raises(MicrostructureContractError):
        _collect([_depth(1_100, 5), _depth(1_200, 3)])


def test_a_repeated_sequence_is_refused_rather_than_deduplicated() -> None:
    with pytest.raises(MicrostructureContractError):
        _collect([_depth(1_100, 5), _depth(1_200, 5)])


def test_trades_are_carried_through_alongside_the_book() -> None:
    episode = _collect([
        _depth(1_100, 1),
        TradeFrame(ts_us=1_150, trade_id="t-1", price=100.05, size=3.0, aggressor=Side.ASK),
        _depth(1_200, 2),
    ])
    assert len(episode.trades) == 1
    assert episode.trades[0].aggressor is Side.ASK
    assert episode.continuous


def test_an_unrecognised_frame_is_refused_rather_than_ignored() -> None:
    class Rogue:
        ts_us = 1_100

    with pytest.raises(MicrostructureContractError) as caught:
        _collect([Rogue()])
    assert "unknown_frame_type" in str(caught.value)


def test_the_module_carries_no_endpoint_and_no_transport() -> None:
    """Depth and tape addresses are unverified; a plausible URL here would be believed."""

    source = Path("trading/market_data/microstructure_collector.py").read_text(encoding="utf-8")
    body = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    for forbidden in ("wss://", "https://", "http://", "import socket", "import requests",
                      "websockets", "urllib"):
        assert forbidden not in body, "collector must stay endpoint-free: %s" % forbidden
