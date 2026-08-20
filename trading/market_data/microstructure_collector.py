"""Turn a stream of decoded frames into an episode, accounting for every break.

This module deliberately contains no URL, no socket and no default transport.
The depth and tape stream addresses are not known to this project — only
``contract/kline``, ``contract/ticker``, ``contract/funding_rate`` and
``contract/detail`` appear anywhere in it — and inventing one would be the exact
class of guess that `mexc_endpoint_official_evidence` exists to prevent. The
caller supplies frames; wire decoding belongs to an adapter written once an
endpoint has been verified.

What lives here is the part that is endpoint-independent and easy to get wrong:
deciding when the book stopped being trustworthy. A reconnect does not know which
update it missed until the next one arrives, so the break is held open and closed
with real sequence numbers rather than guessed at. A frame that cannot be placed
is refused rather than skipped, because a silently dropped frame is exactly the
corruption the storage contract is designed to make visible.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from trading.market_data.microstructure_store import (
    MICROSTRUCTURE_EPISODE_VERSION,
    DepthSnapshot,
    EpisodeRecord,
    GapReason,
    MicrostructureContractError,
    Side,
    StreamGap,
    TradeRecord,
)


@dataclass(frozen=True)
class DepthFrame:
    ts_us: int
    sequence: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class TradeFrame:
    ts_us: int
    trade_id: str
    price: float
    size: float
    aggressor: Side


@dataclass(frozen=True)
class BreakFrame:
    """The transport reporting that continuity was lost, not that nothing happened."""

    ts_us: int
    reason: GapReason


Frame = DepthFrame | TradeFrame | BreakFrame


@dataclass
class _PendingBreak:
    ts_us: int
    reason: GapReason
    last_sequence: int


def collect_episode(
    frames: Iterable[Frame],
    *,
    symbol: str,
    kind: str,
    start_ts_us: int,
    end_ts_us: int,
) -> EpisodeRecord:
    """Build one episode, recording every discontinuity it passed through.

    Raises rather than returning a partial episode: a caller that receives a
    record can rely on it describing a book whose breaks are all accounted for.
    """

    snapshots: list[DepthSnapshot] = []
    trades: list[TradeRecord] = []
    gaps: list[StreamGap] = []
    pending: _PendingBreak | None = None
    last_sequence: int | None = None

    for frame in frames:
        if not start_ts_us <= frame.ts_us <= end_ts_us:
            raise MicrostructureContractError("microstructure_frame_outside_window")

        if isinstance(frame, BreakFrame):
            if last_sequence is None:
                # Nothing has been received yet, so nothing was lost.
                continue
            # Held open: which update was missed is unknown until the next one.
            pending = _PendingBreak(frame.ts_us, frame.reason, last_sequence)
            continue

        if isinstance(frame, TradeFrame):
            trades.append(
                TradeRecord(
                    ts_us=frame.ts_us,
                    trade_id=frame.trade_id,
                    price=frame.price,
                    size=frame.size,
                    aggressor=frame.aggressor,
                )
            )
            continue

        if not isinstance(frame, DepthFrame):
            raise MicrostructureContractError("microstructure_unknown_frame_type")

        if last_sequence is not None:
            step = frame.sequence - last_sequence
            if step <= 0:
                raise MicrostructureContractError("microstructure_sequence_not_advancing")
            if step > 1:
                reason = pending.reason if pending else GapReason.SEQUENCE_JUMP
                at = pending.ts_us if pending else frame.ts_us
                gaps.append(
                    StreamGap(
                        ts_us=at,
                        reason=reason,
                        last_sequence=last_sequence,
                        next_sequence=frame.sequence,
                    )
                )
            elif pending is not None:
                # A break was reported but the stream resumed without losing an
                # update. Nothing was missed, so nothing is recorded as missing.
                pass
        pending = None

        snapshots.append(
            DepthSnapshot(
                ts_us=frame.ts_us,
                sequence=frame.sequence,
                bids=frame.bids,
                asks=frame.asks,
            )
        )
        last_sequence = frame.sequence

    if pending is not None:
        raise MicrostructureContractError("microstructure_episode_ends_inside_a_break")

    return EpisodeRecord(
        contract_version=MICROSTRUCTURE_EPISODE_VERSION,
        symbol=symbol,
        kind=kind,
        start_ts_us=start_ts_us,
        end_ts_us=end_ts_us,
        snapshots=tuple(snapshots),
        trades=tuple(trades),
        gaps=tuple(gaps),
    )
