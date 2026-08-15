"""Deterministic UTC aggregation of canonical MEXC Min1 rows.

This is a new v3 evidence contract.  It intentionally does not extend the
frozen v2 frame/journal codecs.  Callers must supply one upstream receipt for
every normalized Min1 row so a derived bar cannot become available before the
last input row was actually received.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from trading.market_data.bar_contract import interval_seconds, is_bar_aligned


MIN1_AGGREGATION_CONTRACT_VERSION = "mexc_min1_aggregation_v1"
MIN1_SOURCE_ROW_HASH_VERSION = "mexc_min1_source_row_hash_v1"
MIN1_SOURCE_BUNDLE_HASH_VERSION = "mexc_min1_source_bundle_hash_v1"
DERIVED_BAR_CONTENT_HASH_VERSION = "mexc_derived_bar_content_hash_v1"
AGGREGATED_BAR_EVIDENCE_HASH_VERSION = "mexc_aggregated_bar_evidence_hash_v1"

_PINNED_CONTRACT_HASH = (
    "0d851b253cde913d95a693e0db7296b59ff78a6048bacb20838386f2e8e20a21"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9_]{0,63}$")
_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume", "turnover")
_RECEIPT_KEYS = frozenset(
    {
        "bar_open_ts",
        "request_started_at",
        "received_at",
        "source_content_hash",
        "source_lineage_hash",
        "normalized_row_hash",
    }
)
_DERIVED_BAR_KEYS = frozenset({"bar_open_ts", *_REQUIRED_COLUMNS})
_EVIDENCE_KEYS = frozenset(
    {
        "contract_version",
        "contract_hash",
        "venue",
        "symbol",
        "venue_symbol",
        "source_timeframe",
        "target_timeframe",
        "target_bar_open_ts",
        "target_bar_close_ts",
        "source_bar_count",
        "source_receipts",
        "input_row_hashes",
        "source_bundle_hash",
        "derived_content_hash",
        "available_at",
        "evidence_hash",
    }
)
_WRAPPER_KEYS = frozenset(
    {"contract_version", "contract_hash", "target_timeframe", "bars", "evidence"}
)
_SOURCE_TIMEFRAME = "Min1"
_SOURCE_SECONDS = 60
_TARGET_TIMEFRAME_BY_SECONDS = {
    5 * 60: "Min5",
    15 * 60: "Min15",
    60 * 60: "Min60",
    4 * 60 * 60: "Hour4",
}


class Min1AggregationError(ValueError):
    """Base error for data that cannot satisfy the v3 aggregation contract."""


class InvalidMin1FrameError(Min1AggregationError):
    """The input is not a canonical, finite Min1 OHLCV+turnover frame."""


class DuplicateMin1BarError(Min1AggregationError):
    """A Min1 bar open or its receipt occurs more than once."""


class UnalignedMin1BarError(Min1AggregationError):
    """A source minute or requested group is not aligned to UTC boundaries."""


class Min1GapError(Min1AggregationError):
    """Consecutive input rows do not cover every expected UTC minute."""


class IncompleteAggregationGroupError(Min1AggregationError):
    """The requested input range contains a partial target-timeframe group."""


class Min1ReceiptError(Min1AggregationError):
    """Upstream source receipts cannot prove causal availability of every row."""


class UnsupportedAggregationTargetError(Min1AggregationError):
    """The requested derived timeframe is outside the v3 aggregation contract."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Min1AggregationError("aggregation_payload_is_not_canonical_json") from exc


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _finite(value: object, *, field: str, error_type=InvalidMin1FrameError) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise error_type(f"{field}_must_be_a_finite_number")
    number = float(value)
    if not math.isfinite(number):
        raise error_type(f"{field}_must_be_a_finite_number")
    return 0.0 if number == 0.0 else number


def _safe_string(
    value: object,
    *,
    field: str,
    pattern: re.Pattern[str],
) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise Min1AggregationError(f"{field}_is_invalid")
    return value


def _canonical_target_timeframe(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UnsupportedAggregationTargetError("target_timeframe_is_unsupported")
    try:
        seconds = interval_seconds(value)
    except (TypeError, ValueError) as exc:
        raise UnsupportedAggregationTargetError(
            "target_timeframe_is_unsupported"
        ) from exc
    try:
        return _TARGET_TIMEFRAME_BY_SECONDS[seconds]
    except KeyError as exc:
        raise UnsupportedAggregationTargetError(
            "target_timeframe_is_unsupported"
        ) from exc


_CONTRACT_SCHEMA = {
    "contract_version": MIN1_AGGREGATION_CONTRACT_VERSION,
    "source_row_hash_version": MIN1_SOURCE_ROW_HASH_VERSION,
    "source_bundle_hash_version": MIN1_SOURCE_BUNDLE_HASH_VERSION,
    "derived_bar_content_hash_version": DERIVED_BAR_CONTENT_HASH_VERSION,
    "aggregated_bar_evidence_hash_version": AGGREGATED_BAR_EVIDENCE_HASH_VERSION,
    "source": {
        "timeframe": _SOURCE_TIMEFRAME,
        "timeframe_seconds": _SOURCE_SECONDS,
        "required_columns": list(_REQUIRED_COLUMNS),
        "index": "unique_monotonic_timezone_aware_utc_aligned_bar_open",
        "quality": [
            "finite_ohlcvt",
            "strictly_positive_prices",
            "coherent_ohlc_geometry",
            "nonnegative_volume_and_turnover",
            "strictly_contiguous_minutes",
        ],
    },
    "targets": [
        {"timeframe": name, "seconds": seconds}
        for seconds, name in sorted(_TARGET_TIMEFRAME_BY_SECONDS.items())
    ],
    "grouping": {
        "anchor": "unix_epoch_utc",
        "leading_partial": "error",
        "trailing_partial": "error",
        "internal_gap": "error",
        "duplicate": "error",
        "silent_drop": False,
    },
    "aggregation": {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "ordered_math_fsum",
        "turnover": "ordered_math_fsum",
    },
    "timing": {
        "one_receipt_per_source_row": True,
        "receipt_normalized_row_hash_must_match_consumed_row": True,
        "source_request_not_before_source_bar_close": True,
        "source_receipt_not_before_source_request": True,
        "derived_available_at": "max_all_group_source_received_at",
    },
    "serialization": {
        "derived_rows": "immutable_typed_values_with_copy_on_frame_projection",
        "parsers": "exact_keys_and_full_hash_recomputation",
        "wrapper": "derived_content_hash_verified_against_each_typed_row",
        "receipt_keys": sorted(_RECEIPT_KEYS),
        "derived_bar_keys": sorted(_DERIVED_BAR_KEYS),
        "evidence_keys": sorted(_EVIDENCE_KEYS),
        "wrapper_keys": sorted(_WRAPPER_KEYS),
    },
    "hash_domains": {
        "source_row": "normalized_market_row_without_operational_timing",
        "source_bundle": (
            "ordered_source_rows_plus_opaque_upstream_content_and_lineage_hashes_"
            "without_operational_timing"
        ),
        "derived_content": "derived_market_bar_without_operational_timing",
        "evidence": (
            "contract_and_content_and_full_source_receipts_and_availability"
        ),
    },
    "explicit_errors": [
        "invalid_frame",
        "duplicate",
        "unaligned",
        "gap",
        "incomplete_group",
        "invalid_receipt",
        "unsupported_target",
    ],
}


def _computed_contract_hash() -> str:
    return _sha256(_CONTRACT_SCHEMA)


def min1_aggregation_contract_hash() -> str:
    """Return the pinned v1 schema hash, rejecting an in-place semantic edit."""

    digest = _computed_contract_hash()
    if digest != _PINNED_CONTRACT_HASH:
        raise RuntimeError("min1_aggregation_contract_changed_without_version_bump")
    return digest


@dataclass(frozen=True)
class Min1BarReceiptV1:
    """Upstream timing and immutable commitments for one normalized Min1 row.

    ``source_content_hash`` normally identifies the exact raw response/page
    bytes. ``source_lineage_hash`` identifies the upstream normalized
    shard/manifest lineage.  They remain opaque here: this contract binds them
    to the row it consumed and independently hashes that normalized row.
    """

    bar_open_ts: float
    request_started_at: float
    received_at: float
    source_content_hash: str
    source_lineage_hash: str
    normalized_row_hash: str

    def __post_init__(self) -> None:
        bar_open = _finite(
            self.bar_open_ts,
            field="receipt_bar_open_ts",
            error_type=Min1ReceiptError,
        )
        started = _finite(
            self.request_started_at,
            field="receipt_request_started_at",
            error_type=Min1ReceiptError,
        )
        received = _finite(
            self.received_at,
            field="receipt_received_at",
            error_type=Min1ReceiptError,
        )
        if not is_bar_aligned(bar_open, _SOURCE_TIMEFRAME):
            raise UnalignedMin1BarError("receipt_bar_open_is_not_utc_minute_aligned")
        bar_close = bar_open + _SOURCE_SECONDS
        if started + 1e-6 < bar_close:
            raise Min1ReceiptError("request_started_at_precedes_source_bar_close")
        if received < started:
            raise Min1ReceiptError("receipt_received_at_precedes_request_started_at")
        if not isinstance(self.source_content_hash, str) or not _SHA256_RE.fullmatch(
            self.source_content_hash
        ):
            raise Min1ReceiptError("source_content_hash_is_not_sha256")
        if not isinstance(self.source_lineage_hash, str) or not _SHA256_RE.fullmatch(
            self.source_lineage_hash
        ):
            raise Min1ReceiptError("source_lineage_hash_is_not_sha256")
        if not isinstance(self.normalized_row_hash, str) or not _SHA256_RE.fullmatch(
            self.normalized_row_hash
        ):
            raise Min1ReceiptError("normalized_row_hash_is_not_sha256")
        object.__setattr__(self, "bar_open_ts", bar_open)
        object.__setattr__(self, "request_started_at", started)
        object.__setattr__(self, "received_at", received)

    def as_dict(self) -> dict[str, object]:
        return {
            "bar_open_ts": self.bar_open_ts,
            "request_started_at": self.request_started_at,
            "received_at": self.received_at,
            "source_content_hash": self.source_content_hash,
            "source_lineage_hash": self.source_lineage_hash,
            "normalized_row_hash": self.normalized_row_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Min1BarReceiptV1:
        if not isinstance(payload, Mapping) or set(payload) != _RECEIPT_KEYS:
            raise Min1ReceiptError("min1_receipt_keys_are_not_exact")
        return cls(
            bar_open_ts=payload["bar_open_ts"],
            request_started_at=payload["request_started_at"],
            received_at=payload["received_at"],
            source_content_hash=payload["source_content_hash"],
            source_lineage_hash=payload["source_lineage_hash"],
            normalized_row_hash=payload["normalized_row_hash"],
        )


@dataclass(frozen=True)
class AggregatedBarV1:
    """Immutable derived OHLCV+turnover values for one UTC target bar."""

    bar_open_ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float

    def __post_init__(self) -> None:
        open_ts = _finite(self.bar_open_ts, field="derived_bar_open_ts")
        values = {
            field: _finite(getattr(self, field), field=f"derived_{field}")
            for field in _REQUIRED_COLUMNS
        }
        if (
            values["open"] <= 0.0
            or values["high"] <= 0.0
            or values["low"] <= 0.0
            or values["close"] <= 0.0
        ):
            raise InvalidMin1FrameError("derived_prices_must_be_positive")
        if (
            values["high"] < max(values["open"], values["close"])
            or values["low"] > min(values["open"], values["close"])
            or values["low"] > values["high"]
        ):
            raise InvalidMin1FrameError("derived_ohlc_geometry_is_incoherent")
        if values["volume"] < 0.0:
            raise InvalidMin1FrameError("derived_volume_must_not_be_negative")
        if values["turnover"] < 0.0:
            raise InvalidMin1FrameError("derived_turnover_must_not_be_negative")
        object.__setattr__(self, "bar_open_ts", open_ts)
        for field, value in values.items():
            object.__setattr__(self, field, value)

    def values_dict(self) -> dict[str, float]:
        return {field: getattr(self, field) for field in _REQUIRED_COLUMNS}

    def as_dict(self) -> dict[str, object]:
        return {"bar_open_ts": self.bar_open_ts, **self.values_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AggregatedBarV1:
        if not isinstance(payload, Mapping) or set(payload) != _DERIVED_BAR_KEYS:
            raise InvalidMin1FrameError("derived_bar_keys_are_not_exact")
        return cls(
            bar_open_ts=payload["bar_open_ts"],
            open=payload["open"],
            high=payload["high"],
            low=payload["low"],
            close=payload["close"],
            volume=payload["volume"],
            turnover=payload["turnover"],
        )


def _source_rows(
    receipts: Sequence[Min1BarReceiptV1],
    input_row_hashes: Sequence[str],
    *,
    include_timing: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ordinal, (receipt, row_hash) in enumerate(zip(receipts, input_row_hashes)):
        row: dict[str, object] = {
            "ordinal": ordinal,
            "bar_open_ts": receipt.bar_open_ts,
            "normalized_row_hash": row_hash,
            "source_content_hash": receipt.source_content_hash,
            "source_lineage_hash": receipt.source_lineage_hash,
        }
        if include_timing:
            row.update(
                {
                    "request_started_at": receipt.request_started_at,
                    "received_at": receipt.received_at,
                }
            )
        rows.append(row)
    return rows


def _source_bundle_hash(
    *,
    venue: str,
    symbol: str,
    venue_symbol: str,
    target_timeframe: str,
    target_bar_open_ts: float,
    target_bar_close_ts: float,
    receipts: Sequence[Min1BarReceiptV1],
    input_row_hashes: Sequence[str],
) -> str:
    return _sha256(
        {
            "hash_version": MIN1_SOURCE_BUNDLE_HASH_VERSION,
            "venue": venue,
            "symbol": symbol,
            "venue_symbol": venue_symbol,
            "source_timeframe": _SOURCE_TIMEFRAME,
            "target_timeframe": target_timeframe,
            "target_bar_open_ts": target_bar_open_ts,
            "target_bar_close_ts": target_bar_close_ts,
            "sources": _source_rows(
                receipts, input_row_hashes, include_timing=False
            ),
        }
    )


def _derived_content_hash(
    bar: AggregatedBarV1,
    *,
    venue: str,
    symbol: str,
    venue_symbol: str,
    target_timeframe: str,
) -> str:
    target_seconds = interval_seconds(target_timeframe)
    return _sha256(
        {
            "hash_version": DERIVED_BAR_CONTENT_HASH_VERSION,
            "venue": venue,
            "symbol": symbol,
            "venue_symbol": venue_symbol,
            "timeframe": target_timeframe,
            "timeframe_seconds": target_seconds,
            "bar_open_ts": bar.bar_open_ts,
            "bar_close_ts": bar.bar_open_ts + target_seconds,
            "values": bar.values_dict(),
        }
    )


def _aggregated_evidence_hash(
    *,
    contract_hash: str,
    venue: str,
    symbol: str,
    venue_symbol: str,
    target_timeframe: str,
    target_bar_open_ts: float,
    target_bar_close_ts: float,
    source_bar_count: int,
    receipts: Sequence[Min1BarReceiptV1],
    input_row_hashes: Sequence[str],
    source_bundle_hash: str,
    derived_content_hash: str,
    available_at: float,
) -> str:
    return _sha256(
        {
            "hash_version": AGGREGATED_BAR_EVIDENCE_HASH_VERSION,
            "contract_version": MIN1_AGGREGATION_CONTRACT_VERSION,
            "contract_hash": contract_hash,
            "venue": venue,
            "symbol": symbol,
            "venue_symbol": venue_symbol,
            "source_timeframe": _SOURCE_TIMEFRAME,
            "target_timeframe": target_timeframe,
            "target_bar_open_ts": target_bar_open_ts,
            "target_bar_close_ts": target_bar_close_ts,
            "source_bar_count": source_bar_count,
            "source_rows": _source_rows(
                receipts, input_row_hashes, include_timing=True
            ),
            "source_bundle_hash": source_bundle_hash,
            "derived_content_hash": derived_content_hash,
            "available_at": available_at,
        }
    )


@dataclass(frozen=True)
class AggregatedBarEvidenceV1:
    """Content identity, upstream lineage and causal availability for one bar."""

    contract_version: str
    contract_hash: str
    venue: str
    symbol: str
    venue_symbol: str
    source_timeframe: str
    target_timeframe: str
    target_bar_open_ts: float
    target_bar_close_ts: float
    source_bar_count: int
    source_receipts: tuple[Min1BarReceiptV1, ...]
    input_row_hashes: tuple[str, ...]
    source_bundle_hash: str
    derived_content_hash: str
    available_at: float
    evidence_hash: str

    def __post_init__(self) -> None:
        expected_contract_hash = min1_aggregation_contract_hash()
        if self.contract_version != MIN1_AGGREGATION_CONTRACT_VERSION:
            raise Min1AggregationError("aggregation_contract_version_is_invalid")
        if self.contract_hash != expected_contract_hash:
            raise Min1AggregationError("aggregation_contract_hash_is_invalid")
        venue = _safe_string(self.venue, field="venue", pattern=_IDENTIFIER_RE)
        symbol = _safe_string(self.symbol, field="symbol", pattern=_SYMBOL_RE)
        venue_symbol = _safe_string(
            self.venue_symbol, field="venue_symbol", pattern=_SYMBOL_RE
        )
        if self.source_timeframe != _SOURCE_TIMEFRAME:
            raise Min1AggregationError("source_timeframe_must_be_canonical_min1")
        target = _canonical_target_timeframe(self.target_timeframe)
        if target != self.target_timeframe:
            raise Min1AggregationError("target_timeframe_must_be_canonical")
        target_seconds = interval_seconds(target)
        open_ts = _finite(self.target_bar_open_ts, field="target_bar_open_ts")
        close_ts = _finite(self.target_bar_close_ts, field="target_bar_close_ts")
        if not is_bar_aligned(open_ts, target):
            raise UnalignedMin1BarError("target_bar_open_is_not_utc_aligned")
        if not math.isclose(
            close_ts, open_ts + target_seconds, rel_tol=0.0, abs_tol=1e-6
        ):
            raise Min1AggregationError("target_bar_close_does_not_match_timeframe")
        expected_count = target_seconds // _SOURCE_SECONDS
        if (
            isinstance(self.source_bar_count, bool)
            or not isinstance(self.source_bar_count, int)
            or self.source_bar_count != expected_count
        ):
            raise Min1AggregationError("source_bar_count_does_not_match_target")
        if not isinstance(self.source_receipts, tuple) or not all(
            isinstance(item, Min1BarReceiptV1) for item in self.source_receipts
        ):
            raise Min1ReceiptError("source_receipts_must_be_an_immutable_tuple")
        if not isinstance(self.input_row_hashes, tuple):
            raise Min1AggregationError("input_row_hashes_must_be_an_immutable_tuple")
        if (
            len(self.source_receipts) != expected_count
            or len(self.input_row_hashes) != expected_count
        ):
            raise Min1AggregationError("source_evidence_count_does_not_match_target")
        for ordinal, receipt in enumerate(self.source_receipts):
            expected_open = open_ts + ordinal * _SOURCE_SECONDS
            if not math.isclose(
                receipt.bar_open_ts, expected_open, rel_tol=0.0, abs_tol=1e-6
            ):
                raise Min1ReceiptError("source_receipts_are_not_exactly_contiguous")
            if receipt.normalized_row_hash != self.input_row_hashes[ordinal]:
                raise Min1ReceiptError(
                    "receipt_normalized_row_hash_does_not_match_input_row_hash"
                )
        if any(
            not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
            for value in self.input_row_hashes
        ):
            raise Min1AggregationError("input_row_hash_is_not_sha256")
        available = _finite(self.available_at, field="available_at")
        expected_available = max(
            receipt.received_at for receipt in self.source_receipts
        )
        if available != expected_available:
            raise Min1AggregationError("available_at_is_not_last_source_receipt")
        expected_bundle = _source_bundle_hash(
            venue=venue,
            symbol=symbol,
            venue_symbol=venue_symbol,
            target_timeframe=target,
            target_bar_open_ts=open_ts,
            target_bar_close_ts=close_ts,
            receipts=self.source_receipts,
            input_row_hashes=self.input_row_hashes,
        )
        if self.source_bundle_hash != expected_bundle:
            raise Min1AggregationError("source_bundle_hash_mismatch")
        if not isinstance(self.derived_content_hash, str) or not _SHA256_RE.fullmatch(
            self.derived_content_hash
        ):
            raise Min1AggregationError("derived_content_hash_is_not_sha256")
        expected_evidence = _aggregated_evidence_hash(
            contract_hash=expected_contract_hash,
            venue=venue,
            symbol=symbol,
            venue_symbol=venue_symbol,
            target_timeframe=target,
            target_bar_open_ts=open_ts,
            target_bar_close_ts=close_ts,
            source_bar_count=expected_count,
            receipts=self.source_receipts,
            input_row_hashes=self.input_row_hashes,
            source_bundle_hash=expected_bundle,
            derived_content_hash=self.derived_content_hash,
            available_at=available,
        )
        if self.evidence_hash != expected_evidence:
            raise Min1AggregationError("aggregated_evidence_hash_mismatch")
        object.__setattr__(self, "venue", venue)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "venue_symbol", venue_symbol)
        object.__setattr__(self, "target_bar_open_ts", open_ts)
        object.__setattr__(self, "target_bar_close_ts", close_ts)
        object.__setattr__(self, "available_at", available)

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "contract_hash": self.contract_hash,
            "venue": self.venue,
            "symbol": self.symbol,
            "venue_symbol": self.venue_symbol,
            "source_timeframe": self.source_timeframe,
            "target_timeframe": self.target_timeframe,
            "target_bar_open_ts": self.target_bar_open_ts,
            "target_bar_close_ts": self.target_bar_close_ts,
            "source_bar_count": self.source_bar_count,
            "source_receipts": [receipt.as_dict() for receipt in self.source_receipts],
            "input_row_hashes": list(self.input_row_hashes),
            "source_bundle_hash": self.source_bundle_hash,
            "derived_content_hash": self.derived_content_hash,
            "available_at": self.available_at,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AggregatedBarEvidenceV1:
        if not isinstance(payload, Mapping) or set(payload) != _EVIDENCE_KEYS:
            raise Min1AggregationError("aggregated_evidence_keys_are_not_exact")
        receipt_payloads = payload["source_receipts"]
        row_hashes = payload["input_row_hashes"]
        if not isinstance(receipt_payloads, list) or not isinstance(row_hashes, list):
            raise Min1AggregationError("aggregated_evidence_lists_are_invalid")
        return cls(
            contract_version=payload["contract_version"],
            contract_hash=payload["contract_hash"],
            venue=payload["venue"],
            symbol=payload["symbol"],
            venue_symbol=payload["venue_symbol"],
            source_timeframe=payload["source_timeframe"],
            target_timeframe=payload["target_timeframe"],
            target_bar_open_ts=payload["target_bar_open_ts"],
            target_bar_close_ts=payload["target_bar_close_ts"],
            source_bar_count=payload["source_bar_count"],
            source_receipts=tuple(
                Min1BarReceiptV1.from_dict(item) for item in receipt_payloads
            ),
            input_row_hashes=tuple(row_hashes),
            source_bundle_hash=payload["source_bundle_hash"],
            derived_content_hash=payload["derived_content_hash"],
            available_at=payload["available_at"],
            evidence_hash=payload["evidence_hash"],
        )


@dataclass(frozen=True)
class AggregatedMin1FrameV1:
    """Immutable bars paired one-for-one with fully revalidated evidence."""

    bars: tuple[AggregatedBarV1, ...]
    evidence: tuple[AggregatedBarEvidenceV1, ...]
    target_timeframe: str

    def __post_init__(self) -> None:
        target = _canonical_target_timeframe(self.target_timeframe)
        if target != self.target_timeframe:
            raise Min1AggregationError("wrapper_target_timeframe_must_be_canonical")
        if not isinstance(self.bars, tuple) or not isinstance(self.evidence, tuple):
            raise Min1AggregationError("aggregated_wrapper_values_must_be_immutable")
        if not self.bars or len(self.bars) != len(self.evidence):
            raise Min1AggregationError("aggregated_bars_and_evidence_must_match")
        target_seconds = interval_seconds(target)
        first_open = self.bars[0].bar_open_ts
        first_identity: tuple[str, str, str] | None = None
        for ordinal, (bar, evidence) in enumerate(zip(self.bars, self.evidence)):
            if not isinstance(bar, AggregatedBarV1) or not isinstance(
                evidence, AggregatedBarEvidenceV1
            ):
                raise Min1AggregationError("aggregated_wrapper_contains_invalid_item")
            expected_open = first_open + ordinal * target_seconds
            if not math.isclose(
                bar.bar_open_ts, expected_open, rel_tol=0.0, abs_tol=1e-6
            ):
                raise Min1AggregationError("derived_bars_are_not_contiguous")
            if evidence.target_timeframe != target or not math.isclose(
                evidence.target_bar_open_ts,
                bar.bar_open_ts,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise Min1AggregationError("derived_bar_and_evidence_identity_mismatch")
            identity = (evidence.venue, evidence.symbol, evidence.venue_symbol)
            if first_identity is None:
                first_identity = identity
            elif identity != first_identity:
                raise Min1AggregationError("aggregated_wrapper_mixes_instruments")
            expected_content_hash = _derived_content_hash(
                bar,
                venue=evidence.venue,
                symbol=evidence.symbol,
                venue_symbol=evidence.venue_symbol,
                target_timeframe=target,
            )
            if evidence.derived_content_hash != expected_content_hash:
                raise Min1AggregationError("derived_bar_content_hash_mismatch")

    @property
    def frame(self) -> pd.DataFrame:
        index = pd.to_datetime(
            [bar.bar_open_ts for bar in self.bars], unit="s", utc=True
        ).as_unit("ns")
        output = pd.DataFrame(
            [bar.values_dict() for bar in self.bars],
            index=index,
            columns=list(_REQUIRED_COLUMNS),
        )
        output.attrs.update(
            {
                "aggregation_contract_version": MIN1_AGGREGATION_CONTRACT_VERSION,
                "aggregation_contract_hash": min1_aggregation_contract_hash(),
                "source_timeframe": _SOURCE_TIMEFRAME,
                "target_timeframe": self.target_timeframe,
            }
        )
        return output

    @property
    def available_at(self) -> float:
        return max(item.available_at for item in self.evidence)

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": MIN1_AGGREGATION_CONTRACT_VERSION,
            "contract_hash": min1_aggregation_contract_hash(),
            "target_timeframe": self.target_timeframe,
            "bars": [bar.as_dict() for bar in self.bars],
            "evidence": [item.as_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AggregatedMin1FrameV1:
        if not isinstance(payload, Mapping) or set(payload) != _WRAPPER_KEYS:
            raise Min1AggregationError("aggregated_wrapper_keys_are_not_exact")
        if payload["contract_version"] != MIN1_AGGREGATION_CONTRACT_VERSION:
            raise Min1AggregationError("aggregation_contract_version_is_invalid")
        if payload["contract_hash"] != min1_aggregation_contract_hash():
            raise Min1AggregationError("aggregation_contract_hash_is_invalid")
        bars = payload["bars"]
        evidence = payload["evidence"]
        if not isinstance(bars, list) or not isinstance(evidence, list):
            raise Min1AggregationError("aggregated_wrapper_lists_are_invalid")
        return cls(
            bars=tuple(AggregatedBarV1.from_dict(item) for item in bars),
            evidence=tuple(
                AggregatedBarEvidenceV1.from_dict(item) for item in evidence
            ),
            target_timeframe=payload["target_timeframe"],
        )


def _canonical_min1_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise InvalidMin1FrameError("min1_frame_must_be_a_pandas_dataframe")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise InvalidMin1FrameError("min1_index_must_be_a_datetime_index")
    if frame.index.tz is None:
        raise InvalidMin1FrameError("min1_index_must_be_timezone_aware")
    if frame.index.hasnans:
        raise InvalidMin1FrameError("min1_index_must_not_contain_nat")
    if frame.index.has_duplicates:
        raise DuplicateMin1BarError("duplicate_min1_bar_open")
    if not frame.index.is_monotonic_increasing:
        raise InvalidMin1FrameError("min1_index_must_be_monotonic_increasing")
    if frame.empty:
        raise IncompleteAggregationGroupError("aggregation_input_is_empty")
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise InvalidMin1FrameError("min1_frame_is_missing_required_columns")

    # Pandas 2.x may preserve datetime64[s/us/ms].  ``asi8`` uses the index's
    # own unit, so normalize explicitly before any nanosecond arithmetic.
    utc_index = frame.index.tz_convert("UTC").as_unit("ns")
    minute_ns = _SOURCE_SECONDS * 1_000_000_000
    open_ns_values = utc_index.asi8
    if any(int(open_ns) % minute_ns != 0 for open_ns in open_ns_values):
        raise UnalignedMin1BarError("min1_bar_open_is_not_utc_minute_aligned")
    if len(open_ns_values) > 1 and any(
        int(current) - int(previous) != minute_ns
        for previous, current in zip(open_ns_values[:-1], open_ns_values[1:])
    ):
        raise Min1GapError("min1_input_contains_missing_or_irregular_minutes")

    values: dict[str, list[float]] = {column: [] for column in _REQUIRED_COLUMNS}
    for position in range(len(frame)):
        row: dict[str, float] = {}
        for column in _REQUIRED_COLUMNS:
            parsed = _finite(frame[column].iloc[position], field=f"min1_{column}")
            row[column] = parsed
            values[column].append(parsed)
        if (
            row["open"] <= 0.0
            or row["high"] <= 0.0
            or row["low"] <= 0.0
            or row["close"] <= 0.0
        ):
            raise InvalidMin1FrameError("min1_prices_must_be_positive")
        if (
            row["high"] < max(row["open"], row["close"])
            or row["low"] > min(row["open"], row["close"])
            or row["low"] > row["high"]
        ):
            raise InvalidMin1FrameError("min1_ohlc_geometry_is_incoherent")
        if row["volume"] < 0.0:
            raise InvalidMin1FrameError("min1_volume_must_not_be_negative")
        if row["turnover"] < 0.0:
            raise InvalidMin1FrameError("min1_turnover_must_not_be_negative")

    return pd.DataFrame(values, index=utc_index.copy())


def _receipts_by_open(
    receipts: Sequence[Min1BarReceiptV1],
    *,
    expected_index: pd.DatetimeIndex,
) -> dict[int, Min1BarReceiptV1]:
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise Min1ReceiptError("min1_receipts_must_be_a_sequence")
    by_open: dict[int, Min1BarReceiptV1] = {}
    for receipt in receipts:
        if not isinstance(receipt, Min1BarReceiptV1):
            raise Min1ReceiptError("min1_receipts_contain_invalid_item")
        open_ns = pd.Timestamp(receipt.bar_open_ts, unit="s", tz="UTC").value
        if open_ns in by_open:
            raise DuplicateMin1BarError("duplicate_min1_receipt_for_bar_open")
        by_open[open_ns] = receipt

    expected = {int(value) for value in expected_index.asi8}
    if set(by_open) != expected:
        raise Min1ReceiptError("min1_receipts_do_not_match_input_rows")
    return by_open


def normalized_min1_source_row_hash(
    *,
    venue: str,
    symbol: str,
    venue_symbol: str,
    bar_open_ts: float,
    values: Mapping[str, object],
) -> str:
    """Hash one exact float64 Min1 row in the executable S3 domain."""

    venue = _safe_string(venue, field="venue", pattern=_IDENTIFIER_RE)
    symbol = _safe_string(symbol, field="symbol", pattern=_SYMBOL_RE)
    venue_symbol = _safe_string(
        venue_symbol, field="venue_symbol", pattern=_SYMBOL_RE
    )
    bar_open_ts = _finite(bar_open_ts, field="source_row_bar_open_ts")
    if not is_bar_aligned(bar_open_ts, _SOURCE_TIMEFRAME):
        raise UnalignedMin1BarError("source_row_bar_open_is_not_utc_aligned")
    if not isinstance(values, Mapping) or set(values) != set(_REQUIRED_COLUMNS):
        raise InvalidMin1FrameError("source_row_values_keys_are_not_exact")
    parsed = {
        column: _finite(values[column], field=f"source_row_{column}")
        for column in _REQUIRED_COLUMNS
    }
    if any(parsed[column] <= 0.0 for column in ("open", "high", "low", "close")):
        raise InvalidMin1FrameError("source_row_prices_must_be_positive")
    if (
        parsed["high"] < max(parsed["open"], parsed["close"])
        or parsed["low"] > min(parsed["open"], parsed["close"])
        or parsed["low"] > parsed["high"]
    ):
        raise InvalidMin1FrameError("source_row_ohlc_geometry_is_incoherent")
    if parsed["volume"] < 0.0 or parsed["turnover"] < 0.0:
        raise InvalidMin1FrameError(
            "source_row_volume_or_turnover_must_not_be_negative"
        )
    return _sha256(
        {
            "hash_version": MIN1_SOURCE_ROW_HASH_VERSION,
            "venue": venue,
            "symbol": symbol,
            "venue_symbol": venue_symbol,
            "timeframe": _SOURCE_TIMEFRAME,
            "timeframe_seconds": _SOURCE_SECONDS,
            "bar_open_ts": bar_open_ts,
            "bar_close_ts": bar_open_ts + _SOURCE_SECONDS,
            "values": parsed,
        }
    )


def _normalized_source_row_hash(
    row: pd.Series,
    *,
    venue: str,
    symbol: str,
    venue_symbol: str,
    bar_open_ts: float,
) -> str:
    return normalized_min1_source_row_hash(
        venue=venue,
        symbol=symbol,
        venue_symbol=venue_symbol,
        bar_open_ts=bar_open_ts,
        values={column: row[column] for column in _REQUIRED_COLUMNS},
    )


def _derived_values(group: pd.DataFrame) -> dict[str, float]:
    volume = math.fsum(group["volume"].tolist())
    turnover = math.fsum(group["turnover"].tolist())
    return {
        "open": float(group["open"].iloc[0]),
        "high": float(group["high"].max()),
        "low": float(group["low"].min()),
        "close": float(group["close"].iloc[-1]),
        "volume": 0.0 if volume == 0.0 else volume,
        "turnover": 0.0 if turnover == 0.0 else turnover,
    }


def aggregate_canonical_min1(
    frame: pd.DataFrame,
    *,
    target_timeframe: str,
    receipts: Sequence[Min1BarReceiptV1],
    venue: str,
    symbol: str,
    venue_symbol: str,
) -> AggregatedMin1FrameV1:
    """Aggregate a closed, contiguous Min1 range without trimming or gap fill.

    The range must begin at a UTC target boundary and end after an integer
    number of target groups.  Any leading/trailing partial group is an error;
    callers must request an exact range instead of relying on resample defaults.
    """

    contract_hash = min1_aggregation_contract_hash()
    target = _canonical_target_timeframe(target_timeframe)
    target_seconds = interval_seconds(target)
    group_size = target_seconds // _SOURCE_SECONDS
    venue = _safe_string(venue, field="venue", pattern=_IDENTIFIER_RE)
    symbol = _safe_string(symbol, field="symbol", pattern=_SYMBOL_RE)
    venue_symbol = _safe_string(
        venue_symbol, field="venue_symbol", pattern=_SYMBOL_RE
    )
    canonical = _canonical_min1_frame(frame)
    by_open = _receipts_by_open(receipts, expected_index=canonical.index)

    first_open_ns = int(canonical.index.asi8[0])
    target_ns = target_seconds * 1_000_000_000
    if first_open_ns % target_ns != 0:
        raise UnalignedMin1BarError("min1_range_start_is_not_target_utc_aligned")
    if len(canonical) % group_size != 0:
        raise IncompleteAggregationGroupError(
            "min1_range_contains_incomplete_target_group"
        )

    derived_bars: list[AggregatedBarV1] = []
    evidences: list[AggregatedBarEvidenceV1] = []

    for start in range(0, len(canonical), group_size):
        group = canonical.iloc[start : start + group_size]
        group_open = group.index[0]
        group_open_ts = float(group_open.timestamp())
        group_close_ts = group_open_ts + float(target_seconds)
        values = _derived_values(group)
        derived_bar = AggregatedBarV1(bar_open_ts=group_open_ts, **values)

        source_receipts: list[Min1BarReceiptV1] = []
        input_row_hashes: list[str] = []
        for bar_open, row in group.iterrows():
            bar_open_ts = float(bar_open.timestamp())
            receipt = by_open[int(bar_open.value)]
            row_hash = _normalized_source_row_hash(
                row,
                venue=venue,
                symbol=symbol,
                venue_symbol=venue_symbol,
                bar_open_ts=bar_open_ts,
            )
            if receipt.normalized_row_hash != row_hash:
                raise Min1ReceiptError(
                    "receipt_normalized_row_hash_does_not_match_consumed_row"
                )
            source_receipts.append(receipt)
            input_row_hashes.append(row_hash)

        receipt_tuple = tuple(source_receipts)
        row_hash_tuple = tuple(input_row_hashes)
        source_bundle_hash = _source_bundle_hash(
            venue=venue,
            symbol=symbol,
            venue_symbol=venue_symbol,
            target_timeframe=target,
            target_bar_open_ts=group_open_ts,
            target_bar_close_ts=group_close_ts,
            receipts=receipt_tuple,
            input_row_hashes=row_hash_tuple,
        )
        derived_content_hash = _derived_content_hash(
            derived_bar,
            venue=venue,
            symbol=symbol,
            venue_symbol=venue_symbol,
            target_timeframe=target,
        )
        available_at = max(receipt.received_at for receipt in source_receipts)
        evidence_hash = _aggregated_evidence_hash(
            contract_hash=contract_hash,
            venue=venue,
            symbol=symbol,
            venue_symbol=venue_symbol,
            target_timeframe=target,
            target_bar_open_ts=group_open_ts,
            target_bar_close_ts=group_close_ts,
            source_bar_count=group_size,
            receipts=receipt_tuple,
            input_row_hashes=row_hash_tuple,
            source_bundle_hash=source_bundle_hash,
            derived_content_hash=derived_content_hash,
            available_at=available_at,
        )

        derived_bars.append(derived_bar)
        evidences.append(
            AggregatedBarEvidenceV1(
                contract_version=MIN1_AGGREGATION_CONTRACT_VERSION,
                contract_hash=contract_hash,
                venue=venue,
                symbol=symbol,
                venue_symbol=venue_symbol,
                source_timeframe=_SOURCE_TIMEFRAME,
                target_timeframe=target,
                target_bar_open_ts=group_open_ts,
                target_bar_close_ts=group_close_ts,
                source_bar_count=group_size,
                source_receipts=receipt_tuple,
                input_row_hashes=row_hash_tuple,
                source_bundle_hash=source_bundle_hash,
                derived_content_hash=derived_content_hash,
                available_at=available_at,
                evidence_hash=evidence_hash,
            )
        )

    return AggregatedMin1FrameV1(
        bars=tuple(derived_bars),
        evidence=tuple(evidences),
        target_timeframe=target,
    )


__all__ = [
    "AGGREGATED_BAR_EVIDENCE_HASH_VERSION",
    "DERIVED_BAR_CONTENT_HASH_VERSION",
    "MIN1_AGGREGATION_CONTRACT_VERSION",
    "MIN1_SOURCE_BUNDLE_HASH_VERSION",
    "MIN1_SOURCE_ROW_HASH_VERSION",
    "AggregatedBarV1",
    "AggregatedBarEvidenceV1",
    "AggregatedMin1FrameV1",
    "DuplicateMin1BarError",
    "IncompleteAggregationGroupError",
    "InvalidMin1FrameError",
    "Min1AggregationError",
    "Min1BarReceiptV1",
    "Min1GapError",
    "Min1ReceiptError",
    "UnalignedMin1BarError",
    "UnsupportedAggregationTargetError",
    "aggregate_canonical_min1",
    "normalized_min1_source_row_hash",
    "min1_aggregation_contract_hash",
]
