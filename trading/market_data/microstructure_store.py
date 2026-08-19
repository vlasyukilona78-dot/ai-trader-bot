"""Storage contract for recorded order book and trade tape episodes.

An order book reconstructed from a stream is only as good as the continuity of
that stream. Exchanges number their depth updates, and a single missed update
leaves a book that still looks well formed while no longer describing the market.
Features computed from it are not noisy, they are wrong, and nothing downstream
can tell.

So a gap is a first-class record here rather than an absence. The reader refuses
a file whose sequence numbers jump without a `StreamGap` explaining the jump,
which turns an invisible corruption into a loud one. This is the same principle
the journal already applies to stale and failed sources: an outcome that is not
data must still be recorded as an outcome, never silently promoted to data.

Episodes are homogeneous. One episode per file, one contract version, one symbol,
one kind. Mixing is rejected rather than merged, because the analysis compares
triggered episodes against controls and a file that quietly contains both
destroys the comparison it exists to support.

This module is a contract and a codec. It opens no socket and starts no
recording; the transport that fills it is supplied from outside.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

MICROSTRUCTURE_EPISODE_VERSION = "mexc_microstructure_episode_v1"

_MAX_LEVELS = 200
_MAX_PRICE = 1e12


class MicrostructureContractError(RuntimeError):
    """Raised whenever a record cannot be trusted to describe the market."""


class Side(str, Enum):
    BID = "bid"
    ASK = "ask"


class GapReason(str, Enum):
    SEQUENCE_JUMP = "sequence_jump"
    RECONNECT = "reconnect"
    SUBSCRIBE_LOST = "subscribe_lost"
    LOCAL_OVERLOAD = "local_overload"


def _canonical_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MicrostructureContractError("microstructure_payload_not_canonical") from exc


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _positive(value: float, *, field: str) -> float:
    number = float(value)
    if not number > 0.0 or number >= _MAX_PRICE:
        raise MicrostructureContractError("microstructure_%s_out_of_range" % field)
    return number


@dataclass(frozen=True)
class DepthSnapshot:
    """One observation of the book, carrying the sequence it belongs to."""

    ts_us: int
    sequence: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.ts_us <= 0 or self.sequence < 0:
            raise MicrostructureContractError("microstructure_snapshot_identity_invalid")
        if not self.bids or not self.asks:
            raise MicrostructureContractError("microstructure_snapshot_one_sided")
        if len(self.bids) > _MAX_LEVELS or len(self.asks) > _MAX_LEVELS:
            raise MicrostructureContractError("microstructure_snapshot_too_deep")
        for price, size in (*self.bids, *self.asks):
            _positive(price, field="price")
            _positive(size, field="size")
        # Levels must be ordered outward from the touch, or "best bid" is a guess.
        if any(a[0] <= b[0] for a, b in zip(self.bids, self.bids[1:])):
            raise MicrostructureContractError("microstructure_bids_not_descending")
        if any(a[0] >= b[0] for a, b in zip(self.asks, self.asks[1:])):
            raise MicrostructureContractError("microstructure_asks_not_ascending")
        if self.bids[0][0] >= self.asks[0][0]:
            raise MicrostructureContractError("microstructure_book_crossed")

    @property
    def spread(self) -> float:
        return self.asks[0][0] - self.bids[0][0]

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "depth",
            "ts_us": self.ts_us,
            "sequence": self.sequence,
            "bids": [list(level) for level in self.bids],
            "asks": [list(level) for level in self.asks],
        }


@dataclass(frozen=True)
class TradeRecord:
    ts_us: int
    trade_id: str
    price: float
    size: float
    aggressor: Side

    def __post_init__(self) -> None:
        if self.ts_us <= 0 or not self.trade_id:
            raise MicrostructureContractError("microstructure_trade_identity_invalid")
        _positive(self.price, field="price")
        _positive(self.size, field="size")

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "trade",
            "ts_us": self.ts_us,
            "trade_id": self.trade_id,
            "price": self.price,
            "size": self.size,
            "aggressor": self.aggressor.value,
        }


@dataclass(frozen=True)
class StreamGap:
    """A discontinuity, recorded so that it cannot be mistaken for calm."""

    ts_us: int
    reason: GapReason
    last_sequence: int
    next_sequence: int

    def __post_init__(self) -> None:
        if self.ts_us <= 0:
            raise MicrostructureContractError("microstructure_gap_identity_invalid")
        if self.next_sequence <= self.last_sequence:
            raise MicrostructureContractError("microstructure_gap_does_not_advance")

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "gap",
            "ts_us": self.ts_us,
            "reason": self.reason.value,
            "last_sequence": self.last_sequence,
            "next_sequence": self.next_sequence,
        }


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode: one symbol, one kind, one continuous stretch of evidence."""

    contract_version: str
    symbol: str
    kind: str
    start_ts_us: int
    end_ts_us: int
    snapshots: tuple[DepthSnapshot, ...]
    trades: tuple[TradeRecord, ...]
    gaps: tuple[StreamGap, ...]

    def __post_init__(self) -> None:
        if self.contract_version != MICROSTRUCTURE_EPISODE_VERSION:
            raise MicrostructureContractError("microstructure_unknown_contract_version")
        if not self.symbol or self.kind not in {"triggered", "control"}:
            raise MicrostructureContractError("microstructure_episode_identity_invalid")
        if self.end_ts_us <= self.start_ts_us:
            raise MicrostructureContractError("microstructure_episode_window_invalid")
        if not self.snapshots:
            raise MicrostructureContractError("microstructure_episode_has_no_book")
        self._check_window()
        self._check_sequence()

    def _check_window(self) -> None:
        for item in (*self.snapshots, *self.trades, *self.gaps):
            if not self.start_ts_us <= item.ts_us <= self.end_ts_us:
                raise MicrostructureContractError("microstructure_record_outside_window")
        for series in (self.snapshots, self.trades):
            stamps = [item.ts_us for item in series]
            if stamps != sorted(stamps):
                raise MicrostructureContractError("microstructure_records_out_of_order")

    def _check_sequence(self) -> None:
        """Every jump in sequence must be explained by a recorded gap."""

        explained = {(gap.last_sequence, gap.next_sequence) for gap in self.gaps}
        for previous, current in zip(self.snapshots, self.snapshots[1:]):
            step = current.sequence - previous.sequence
            if step == 1:
                continue
            if step <= 0:
                raise MicrostructureContractError("microstructure_sequence_not_advancing")
            if (previous.sequence, current.sequence) not in explained:
                raise MicrostructureContractError("microstructure_unexplained_sequence_gap")

    @property
    def continuous(self) -> bool:
        """True only when the book was never interrupted during the episode."""

        return not self.gaps

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "symbol": self.symbol,
            "kind": self.kind,
            "start_ts_us": self.start_ts_us,
            "end_ts_us": self.end_ts_us,
            "snapshots": [item.as_dict() for item in self.snapshots],
            "trades": [item.as_dict() for item in self.trades],
            "gaps": [item.as_dict() for item in self.gaps],
        }

    @property
    def episode_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def to_bytes(self) -> bytes:
        return _canonical_bytes(self.as_dict())


def parse_episode(raw: bytes) -> EpisodeRecord:
    """Read an episode back, refusing anything the writer could not have produced."""

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MicrostructureContractError("microstructure_episode_not_readable") from exc
    if not isinstance(payload, dict):
        raise MicrostructureContractError("microstructure_episode_not_an_object")

    try:
        snapshots = tuple(
            DepthSnapshot(
                ts_us=int(item["ts_us"]),
                sequence=int(item["sequence"]),
                bids=tuple((float(p), float(s)) for p, s in item["bids"]),
                asks=tuple((float(p), float(s)) for p, s in item["asks"]),
            )
            for item in payload["snapshots"]
        )
        trades = tuple(
            TradeRecord(
                ts_us=int(item["ts_us"]),
                trade_id=str(item["trade_id"]),
                price=float(item["price"]),
                size=float(item["size"]),
                aggressor=Side(item["aggressor"]),
            )
            for item in payload["trades"]
        )
        gaps = tuple(
            StreamGap(
                ts_us=int(item["ts_us"]),
                reason=GapReason(item["reason"]),
                last_sequence=int(item["last_sequence"]),
                next_sequence=int(item["next_sequence"]),
            )
            for item in payload["gaps"]
        )
        return EpisodeRecord(
            contract_version=str(payload["contract_version"]),
            symbol=str(payload["symbol"]),
            kind=str(payload["kind"]),
            start_ts_us=int(payload["start_ts_us"]),
            end_ts_us=int(payload["end_ts_us"]),
            snapshots=snapshots,
            trades=trades,
            gaps=gaps,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MicrostructureContractError("microstructure_episode_schema_mismatch") from exc
