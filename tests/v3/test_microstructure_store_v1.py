"""A recorded book is only evidence if its discontinuities are visible.

The failure these guard against is silent: a stream that missed an update still
produces a well-formed file, and features computed from it are wrong rather than
noisy. Nothing downstream can detect that, so it has to be caught here.
"""

from __future__ import annotations

import pytest

from trading.market_data.microstructure_store import (
    MICROSTRUCTURE_EPISODE_VERSION,
    DepthSnapshot,
    EpisodeRecord,
    GapReason,
    MicrostructureContractError,
    Side,
    StreamGap,
    TradeRecord,
    parse_episode,
)


def _snapshot(ts_us: int, sequence: int, best_bid: float = 100.0) -> DepthSnapshot:
    return DepthSnapshot(
        ts_us=ts_us,
        sequence=sequence,
        bids=((best_bid, 5.0), (best_bid - 0.1, 9.0)),
        asks=((best_bid + 0.1, 4.0), (best_bid + 0.2, 8.0)),
    )


def _episode(snapshots, gaps=(), trades=()) -> EpisodeRecord:
    return EpisodeRecord(
        contract_version=MICROSTRUCTURE_EPISODE_VERSION,
        symbol="AAAUSDT",
        kind="triggered",
        start_ts_us=1_000,
        end_ts_us=9_000,
        snapshots=tuple(snapshots),
        trades=tuple(trades),
        gaps=tuple(gaps),
    )


def test_an_unexplained_sequence_jump_is_refused() -> None:
    """The whole point: a missed update must not pass as a continuous book."""

    with pytest.raises(MicrostructureContractError) as caught:
        _episode([_snapshot(1_100, 10), _snapshot(1_200, 14)])
    assert "unexplained_sequence_gap" in str(caught.value)


def test_the_same_jump_is_accepted_once_it_is_recorded_as_a_gap() -> None:
    episode = _episode(
        [_snapshot(1_100, 10), _snapshot(1_200, 14)],
        gaps=[StreamGap(1_150, GapReason.RECONNECT, last_sequence=10, next_sequence=14)],
    )
    assert not episode.continuous
    assert len(episode.snapshots) == 2


def test_an_uninterrupted_episode_reports_itself_continuous() -> None:
    episode = _episode([_snapshot(1_100, 10), _snapshot(1_200, 11)])
    assert episode.continuous


def test_a_gap_that_does_not_advance_is_rejected() -> None:
    with pytest.raises(MicrostructureContractError):
        StreamGap(1_150, GapReason.SEQUENCE_JUMP, last_sequence=14, next_sequence=14)


def test_a_sequence_going_backwards_is_never_explainable() -> None:
    with pytest.raises(MicrostructureContractError) as caught:
        _episode([_snapshot(1_100, 14), _snapshot(1_200, 10)])
    assert "not_advancing" in str(caught.value)


def test_a_crossed_book_is_refused() -> None:
    with pytest.raises(MicrostructureContractError) as caught:
        DepthSnapshot(ts_us=1, sequence=1, bids=((101.0, 1.0),), asks=((100.0, 1.0),))
    assert "crossed" in str(caught.value)


def test_levels_must_be_ordered_outward_from_the_touch() -> None:
    """Otherwise "best bid" is whichever level happened to be written first."""

    with pytest.raises(MicrostructureContractError):
        DepthSnapshot(ts_us=1, sequence=1, bids=((99.0, 1.0), (100.0, 1.0)), asks=((101.0, 1.0),))
    with pytest.raises(MicrostructureContractError):
        DepthSnapshot(ts_us=1, sequence=1, bids=((100.0, 1.0),), asks=((102.0, 1.0), (101.0, 1.0)))


def test_records_outside_the_declared_window_are_refused() -> None:
    with pytest.raises(MicrostructureContractError) as caught:
        _episode([_snapshot(1_100, 10), _snapshot(99_000, 11)])
    assert "outside_window" in str(caught.value)


def test_records_out_of_time_order_are_refused() -> None:
    with pytest.raises(MicrostructureContractError) as caught:
        _episode([_snapshot(2_000, 10), _snapshot(1_500, 11)])
    assert "out_of_order" in str(caught.value)


def test_an_episode_without_a_book_is_not_an_episode() -> None:
    with pytest.raises(MicrostructureContractError):
        _episode([])


def test_only_the_two_known_kinds_are_accepted() -> None:
    """A file mixing triggered and control rows destroys the comparison."""

    with pytest.raises(MicrostructureContractError):
        EpisodeRecord(
            contract_version=MICROSTRUCTURE_EPISODE_VERSION,
            symbol="AAAUSDT",
            kind="whatever",
            start_ts_us=1_000,
            end_ts_us=9_000,
            snapshots=(_snapshot(1_100, 1),),
            trades=(),
            gaps=(),
        )


def test_an_episode_survives_a_round_trip_unchanged() -> None:
    original = _episode(
        [_snapshot(1_100, 10), _snapshot(1_200, 14)],
        gaps=[StreamGap(1_150, GapReason.SUBSCRIBE_LOST, last_sequence=10, next_sequence=14)],
        trades=[TradeRecord(1_180, "t-1", 100.05, 2.5, Side.ASK)],
    )
    restored = parse_episode(original.to_bytes())

    assert restored == original
    assert restored.episode_hash == original.episode_hash


def test_a_forged_file_that_drops_the_gap_record_fails_to_parse() -> None:
    """Deleting the explanation must not turn a broken stream into a clean one."""

    original = _episode(
        [_snapshot(1_100, 10), _snapshot(1_200, 14)],
        gaps=[StreamGap(1_150, GapReason.RECONNECT, last_sequence=10, next_sequence=14)],
    )
    forged = original.to_bytes().replace(
        b'"gaps":[{"last_sequence":10,"next_sequence":14,'
        b'"reason":"reconnect","ts_us":1150,"type":"gap"}]',
        b'"gaps":[]',
    )
    assert forged != original.to_bytes()          # the edit actually landed
    with pytest.raises(MicrostructureContractError):
        parse_episode(forged)


def test_unreadable_bytes_are_refused_rather_than_guessed_at() -> None:
    with pytest.raises(MicrostructureContractError):
        parse_episode(b"not json at all")
    with pytest.raises(MicrostructureContractError):
        parse_episode(b'["a", "list"]')


def test_the_spread_comes_from_the_touch() -> None:
    assert _snapshot(1_100, 1, best_bid=100.0).spread == pytest.approx(0.1)
