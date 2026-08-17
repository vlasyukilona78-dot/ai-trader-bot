"""Strict, offline-testable MEXC history acquisition contract.

This module is deliberately separate from :mod:`trading.market_data.history`.
The legacy collector is discovery tooling: it can turn request failures into
empty frames, samples cache coverage, and writes mutable CSV files.  None of
those artifacts are admissible v3 evidence.

The contract below has no default network transport and no default artifact
directory.  A caller must provide both explicitly.  Every HTTP response body is
stored content-addressed before JSON parsing, and a complete normalized shard is
published only when every expected closed UTC bar is present exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence
import uuid

import pandas as pd

from trading.market_data.bar_contract import closed_boundary_ts, interval_seconds
from trading.market_data.frame_provenance import canonical_frame_timeframe


STRICT_HISTORY_CONTRACT_VERSION = "mexc_strict_history_v1"
STRICT_HISTORY_RAW_ATTEMPT_VERSION = "mexc_raw_kline_attempt_v1"
STRICT_HISTORY_TRANSPORT_FAILURE_VERSION = "mexc_kline_transport_failure_v1"
STRICT_HISTORY_PAGE_RECEIPT_VERSION = "mexc_history_page_receipt_v1"
STRICT_HISTORY_NORMALIZED_ROW_VERSION = "mexc_normalized_history_row_v1"
STRICT_HISTORY_MANIFEST_VERSION = "mexc_complete_history_manifest_v1"

_PINNED_CONTRACT_HASH = (
    "6c17bd9de3e25210139da4491a1f35fbd0cec557707fb5d376a60ce23e04c6c1"
)
_MAX_MEXC_PAGE_SIZE = 2_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9_]{0,63}$")
_SAFE_HEADER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/;=+* -]{0,255}$")
_PUBLIC_SAFE_HEADER_NAMES = frozenset(
    {
        "content-type",
        "content-length",
        "date",
        "etag",
        "last-modified",
        "retry-after",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
    }
)
_REQUIRED_ARRAYS = ("time", "open", "high", "low", "close", "vol", "amount")
_HTTP_ATTEMPT_KEYS = frozenset(
    {
        "contract_version",
        "page_request",
        "page_id",
        "attempt_ordinal",
        "request_started_at",
        "received_at",
        "http_status",
        "safe_headers",
        "raw_body_length",
        "raw_body_sha256",
    }
)
_TRANSPORT_FAILURE_KEYS = frozenset(
    {
        "contract_version",
        "page_request",
        "page_id",
        "attempt_ordinal",
        "request_started_at",
        "failed_at",
        "outcome",
        "safe_error_code",
        "http_status",
        "raw_body_length",
        "raw_body_sha256",
    }
)


class StrictHistoryError(RuntimeError):
    """Base class for stable, non-body-bearing history failures."""

    code = "strict_history_error"

    def __init__(self, code: str | None = None):
        self.code = code or self.code
        super().__init__(self.code)


class HistoryRangeContractError(StrictHistoryError):
    code = "history_range_contract_error"


class HistoryTransportError(StrictHistoryError):
    code = "history_transport_error"

    def __init__(self, failure_receipt=None, code: str | None = None):
        self.failure_receipt = failure_receipt
        super().__init__(code)


class HistoryNetworkError(HistoryTransportError):
    code = "history_network_error"


class HistoryTimeoutError(HistoryTransportError):
    code = "history_timeout_error"


class HistoryHttpStatusError(HistoryTransportError):
    code = "history_http_status_error"


class HistoryJsonDecodeError(StrictHistoryError):
    code = "history_json_decode_error"


class HistoryApiRejectedError(StrictHistoryError):
    code = "history_api_rejected"


class HistoryPayloadError(StrictHistoryError):
    code = "history_payload_error"


class HistoryPayloadSchemaError(HistoryPayloadError):
    code = "history_payload_schema_error"


class HistoryPayloadValueError(HistoryPayloadError):
    code = "history_payload_value_error"


class HistoryPayloadRangeError(HistoryPayloadError):
    code = "history_payload_range_error"


class HistoryDuplicateTimestampError(HistoryPayloadError):
    code = "history_duplicate_timestamp"


class HistoryIncompleteRangeError(StrictHistoryError):
    """The requested non-empty grid could not be proven complete."""

    code = "history_incomplete_range"
    VALID_REASONS = frozenset(
        {
            "page_budget_exceeded",
            "empty_success",
            "missing_timestamps",
            "unexpected_timestamps",
        }
    )

    def __init__(
        self,
        reason: str,
        *,
        missing_timestamps: Sequence[int] = (),
        unexpected_timestamps: Sequence[int] = (),
    ):
        if reason not in self.VALID_REASONS:
            raise ValueError("invalid_history_incomplete_reason")
        self.reason = reason
        self.missing_timestamps = tuple(int(value) for value in missing_timestamps)
        self.unexpected_timestamps = tuple(
            int(value) for value in unexpected_timestamps
        )
        self.code = type(self).code
        RuntimeError.__init__(self, f"{self.code}:{reason}")


class HistoryStorageError(StrictHistoryError):
    code = "history_storage_error"


class HistoryArtifactConflictError(HistoryStorageError):
    code = "history_artifact_conflict"


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
        raise StrictHistoryError("history_payload_is_not_canonical_json") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_payload(payload: object) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


class _CanonicalContract(Protocol):
    """A frozen contract whose canonical dict is fixed by its field values."""

    def as_dict(self) -> dict[str, object]: ...


@lru_cache(maxsize=1024)
def _frozen_contract_hash(contract: _CanonicalContract) -> str:
    """Hash a frozen contract, keyed by value rather than by instance.

    These identity hashes are read far more often than the contracts are built:
    a single validation pass asks for them hundreds of thousands of times over a
    handful of distinct values, and every miss costs a full canonical JSON
    encode of a nested contract tree. Freezing makes this safe — equal contracts
    always canonicalize identically — and keying by value rather than by
    instance also collapses rebuilt-but-equal contracts onto one entry.
    """

    return _sha256_payload(contract.as_dict())


def _safe_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise HistoryRangeContractError(f"{field}_is_invalid")
    return value


def _safe_symbol(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SYMBOL_RE.fullmatch(value):
        raise HistoryRangeContractError(f"{field}_is_invalid")
    return value


def _strict_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise HistoryRangeContractError(f"{field}_must_be_an_integer")
    return value


def _finite_time(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HistoryRangeContractError(f"{field}_must_be_finite")
    result = float(value)
    if not math.isfinite(result):
        raise HistoryRangeContractError(f"{field}_must_be_finite")
    return result


def _canonical_decimal(value: object, *, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise HistoryPayloadValueError(f"{field}_must_be_finite_numeric")
    try:
        if isinstance(value, Decimal):
            number = value
        elif isinstance(value, (int, float, str)):
            number = Decimal(str(value).strip())
        else:
            raise HistoryPayloadValueError(f"{field}_must_be_finite_numeric")
    except (InvalidOperation, ValueError) as exc:
        raise HistoryPayloadValueError(f"{field}_must_be_finite_numeric") from exc
    if not number.is_finite():
        raise HistoryPayloadValueError(f"{field}_must_be_finite_numeric")
    if number == 0:
        return "0"
    sign, digits_tuple, exponent = number.as_tuple()
    if len(digits_tuple) > 80 or exponent < -100 or exponent > 100:
        raise HistoryPayloadValueError(f"{field}_precision_is_out_of_range")
    digits = "".join(str(digit) for digit in digits_tuple)
    if exponent >= 0:
        rendered = digits + ("0" * exponent)
    else:
        point = len(digits) + exponent
        if point <= 0:
            rendered = "0." + ("0" * (-point)) + digits
        else:
            rendered = digits[:point] + "." + digits[point:]
        rendered = rendered.rstrip("0").rstrip(".")
    return ("-" if sign else "") + rendered


def _decimal_float(value: str) -> float:
    result = float(Decimal(value))
    if not math.isfinite(result):
        raise HistoryPayloadValueError("normalized_value_overflows_float64")
    return result


def _parse_epoch_second(value: object) -> int:
    if isinstance(value, bool):
        raise HistoryPayloadValueError("history_time_must_be_integer_seconds")
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value.strip()):
        return int(value.strip())
    raise HistoryPayloadValueError("history_time_must_be_integer_seconds")


def _reject_json_constant(_value: str) -> None:
    raise HistoryJsonDecodeError("history_json_contains_nonfinite_constant")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HistoryJsonDecodeError("history_json_contains_duplicate_key")
        result[key] = value
    return result


@dataclass(frozen=True)
class HistoryRangeRequestV1:
    """Exact half-open grid of closed UTC candle opens."""

    venue: str
    symbol: str
    venue_symbol: str
    interval: str
    start_open_ts: int
    end_open_ts_exclusive: int
    collection_as_of_ts: float
    endpoint_identity: str
    page_size: int = _MAX_MEXC_PAGE_SIZE
    max_pages: int = 200
    max_attempts_per_page: int = 1
    contract_version: str = STRICT_HISTORY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_CONTRACT_VERSION:
            raise HistoryRangeContractError("history_contract_version_mismatch")
        object.__setattr__(self, "venue", _safe_identifier(self.venue, field="venue"))
        object.__setattr__(self, "symbol", _safe_symbol(self.symbol, field="symbol"))
        object.__setattr__(
            self,
            "venue_symbol",
            _safe_symbol(self.venue_symbol, field="venue_symbol"),
        )
        object.__setattr__(
            self,
            "endpoint_identity",
            _safe_identifier(self.endpoint_identity, field="endpoint_identity"),
        )
        try:
            canonical_interval = canonical_frame_timeframe(self.interval)
        except (TypeError, ValueError) as exc:
            raise HistoryRangeContractError("history_interval_is_invalid") from exc
        object.__setattr__(self, "interval", canonical_interval)

        start = _strict_int(self.start_open_ts, field="start_open_ts")
        end = _strict_int(
            self.end_open_ts_exclusive, field="end_open_ts_exclusive"
        )
        as_of = _finite_time(self.collection_as_of_ts, field="collection_as_of_ts")
        object.__setattr__(self, "collection_as_of_ts", as_of)
        page_size = _strict_int(self.page_size, field="page_size")
        max_pages = _strict_int(self.max_pages, field="max_pages")
        max_attempts = _strict_int(
            self.max_attempts_per_page, field="max_attempts_per_page"
        )
        if page_size < 1 or page_size > _MAX_MEXC_PAGE_SIZE:
            raise HistoryRangeContractError("history_page_size_is_out_of_range")
        if max_pages < 1:
            raise HistoryRangeContractError("history_max_pages_is_out_of_range")
        if max_attempts < 1 or max_attempts > 10:
            raise HistoryRangeContractError(
                "history_max_attempts_per_page_is_out_of_range"
            )
        step = interval_seconds(canonical_interval)
        if start >= end:
            raise HistoryRangeContractError("history_range_must_be_nonempty")
        if start % step != 0 or end % step != 0:
            raise HistoryRangeContractError("history_range_is_not_utc_aligned")
        boundary = int(closed_boundary_ts(as_of, canonical_interval))
        if end > boundary:
            raise HistoryRangeContractError("history_range_contains_unclosed_bar")

    @property
    def interval_seconds(self) -> int:
        return interval_seconds(self.interval)

    @property
    def expected_row_count(self) -> int:
        return (self.end_open_ts_exclusive - self.start_open_ts) // self.interval_seconds

    @property
    def required_pages(self) -> int:
        return (self.expected_row_count + self.page_size - 1) // self.page_size

    @property
    def request_id(self) -> str:
        return _sha256_payload(self.as_dict())

    def expected_timestamps(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.start_open_ts,
                self.end_open_ts_exclusive,
                self.interval_seconds,
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "venue": self.venue,
            "symbol": self.symbol,
            "venue_symbol": self.venue_symbol,
            "interval": self.interval,
            "interval_seconds": self.interval_seconds,
            "start_open_ts": self.start_open_ts,
            "end_open_ts_exclusive": self.end_open_ts_exclusive,
            "collection_as_of_ts": self.collection_as_of_ts,
            "endpoint_identity": self.endpoint_identity,
            "page_size": self.page_size,
            "max_pages": self.max_pages,
            "max_attempts_per_page": self.max_attempts_per_page,
        }


@dataclass(frozen=True)
class KlinePageRequestV1:
    range_request_id: str
    endpoint_identity: str
    venue_symbol: str
    interval: str
    page_ordinal: int
    start_open_ts: int
    end_open_ts_inclusive: int
    expected_row_count: int

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.range_request_id):
            raise HistoryRangeContractError("page_range_request_id_is_invalid")
        _safe_identifier(self.endpoint_identity, field="endpoint_identity")
        _safe_symbol(self.venue_symbol, field="venue_symbol")
        try:
            canonical = canonical_frame_timeframe(self.interval)
        except (TypeError, ValueError) as exc:
            raise HistoryRangeContractError("page_interval_is_invalid") from exc
        object.__setattr__(self, "interval", canonical)
        ordinal = _strict_int(self.page_ordinal, field="page_ordinal")
        start = _strict_int(self.start_open_ts, field="page_start_open_ts")
        end = _strict_int(self.end_open_ts_inclusive, field="page_end_open_ts_inclusive")
        count = _strict_int(self.expected_row_count, field="page_expected_row_count")
        step = interval_seconds(canonical)
        if ordinal < 0 or count < 1 or count > _MAX_MEXC_PAGE_SIZE:
            raise HistoryRangeContractError("page_shape_is_invalid")
        if start % step or end % step or end < start:
            raise HistoryRangeContractError("page_range_is_invalid")
        if ((end - start) // step) + 1 != count:
            raise HistoryRangeContractError("page_expected_count_disagrees_with_range")

    @property
    def page_id(self) -> str:
        return _sha256_payload(self.as_dict())

    @property
    def canonical_path(self) -> str:
        return f"/api/v1/contract/kline/{self.venue_symbol}"

    def expected_timestamps(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.start_open_ts,
                self.end_open_ts_inclusive + interval_seconds(self.interval),
                interval_seconds(self.interval),
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "range_request_id": self.range_request_id,
            "endpoint_identity": self.endpoint_identity,
            "venue_symbol": self.venue_symbol,
            "interval": self.interval,
            "page_ordinal": self.page_ordinal,
            "start_open_ts": self.start_open_ts,
            "end_open_ts_inclusive": self.end_open_ts_inclusive,
            "expected_row_count": self.expected_row_count,
            "canonical_query": {
                "interval": self.interval,
                "start": self.start_open_ts,
                "end": self.end_open_ts_inclusive,
            },
            "canonical_path": self.canonical_path,
        }


@dataclass(frozen=True)
class RawHttpResponseV1:
    """Exact application body and safe operational receipt from one attempt."""

    page_request: KlinePageRequestV1
    request_started_at: float
    received_at: float
    http_status: int
    body: bytes
    safe_headers: tuple[tuple[str, str], ...] = ()
    attempt_ordinal: int = 0
    contract_version: str = STRICT_HISTORY_RAW_ATTEMPT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_RAW_ATTEMPT_VERSION:
            raise HistoryRangeContractError("raw_attempt_contract_version_mismatch")
        if not isinstance(self.page_request, KlinePageRequestV1):
            raise HistoryRangeContractError("raw_attempt_page_request_is_invalid")
        started = _finite_time(self.request_started_at, field="request_started_at")
        received = _finite_time(self.received_at, field="received_at")
        object.__setattr__(self, "request_started_at", started)
        object.__setattr__(self, "received_at", received)
        if received < started:
            raise HistoryRangeContractError("raw_attempt_timing_is_invalid")
        if type(self.http_status) is not int or not (100 <= self.http_status <= 599):
            raise HistoryRangeContractError("raw_attempt_http_status_is_invalid")
        if type(self.attempt_ordinal) is not int or self.attempt_ordinal < 0:
            raise HistoryRangeContractError("raw_attempt_ordinal_is_invalid")
        if not isinstance(self.body, bytes):
            raise HistoryRangeContractError("raw_attempt_body_must_be_bytes")
        normalized_headers: list[tuple[str, str]] = []
        seen: set[str] = set()
        for pair in self.safe_headers:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise HistoryRangeContractError("raw_attempt_headers_are_invalid")
            name, value = pair
            lower = str(name).strip().lower()
            rendered = str(value).strip()
            if lower in seen or lower not in _PUBLIC_SAFE_HEADER_NAMES:
                raise HistoryRangeContractError("raw_attempt_headers_are_invalid")
            if not _IDENTIFIER_RE.fullmatch(lower) or not _SAFE_HEADER_RE.fullmatch(rendered):
                raise HistoryRangeContractError("raw_attempt_headers_are_invalid")
            seen.add(lower)
            normalized_headers.append((lower, rendered))
        object.__setattr__(self, "safe_headers", tuple(sorted(normalized_headers)))

    @property
    def raw_body_sha256(self) -> str:
        return _sha256_bytes(self.body)

    @property
    def attempt_receipt_hash(self) -> str:
        return _sha256_payload(self.receipt_dict())

    def receipt_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "page_request": self.page_request.as_dict(),
            "page_id": self.page_request.page_id,
            "attempt_ordinal": self.attempt_ordinal,
            "request_started_at": self.request_started_at,
            "received_at": self.received_at,
            "http_status": self.http_status,
            "safe_headers": [list(pair) for pair in self.safe_headers],
            "raw_body_length": len(self.body),
            "raw_body_sha256": self.raw_body_sha256,
        }


@dataclass(frozen=True)
class TransportFailureReceiptV1:
    """A network/timeout attempt with no HTTP application response body."""

    page_request: KlinePageRequestV1
    attempt_ordinal: int
    request_started_at: float
    failed_at: float
    outcome: str
    safe_error_code: str
    contract_version: str = STRICT_HISTORY_TRANSPORT_FAILURE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_TRANSPORT_FAILURE_VERSION:
            raise HistoryRangeContractError(
                "transport_failure_contract_version_mismatch"
            )
        if not isinstance(self.page_request, KlinePageRequestV1):
            raise HistoryRangeContractError(
                "transport_failure_page_request_is_invalid"
            )
        if type(self.attempt_ordinal) is not int or self.attempt_ordinal < 0:
            raise HistoryRangeContractError("transport_failure_ordinal_is_invalid")
        started = _finite_time(
            self.request_started_at, field="transport_failure_request_started_at"
        )
        failed = _finite_time(self.failed_at, field="transport_failure_failed_at")
        object.__setattr__(self, "request_started_at", started)
        object.__setattr__(self, "failed_at", failed)
        if failed < started:
            raise HistoryRangeContractError("transport_failure_timing_is_invalid")
        if self.outcome not in {"network_error", "timeout"}:
            raise HistoryRangeContractError("transport_failure_outcome_is_invalid")
        _safe_identifier(self.safe_error_code, field="safe_error_code")

    @property
    def attempt_receipt_hash(self) -> str:
        return _sha256_payload(self.receipt_dict())

    def receipt_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "page_request": self.page_request.as_dict(),
            "page_id": self.page_request.page_id,
            "attempt_ordinal": self.attempt_ordinal,
            "request_started_at": self.request_started_at,
            "failed_at": self.failed_at,
            "outcome": self.outcome,
            "safe_error_code": self.safe_error_code,
            "http_status": None,
            "raw_body_length": None,
            "raw_body_sha256": None,
        }


class RawKlinePageTransport(Protocol):
    """Injected transport; implementations must never hide attempts/errors."""

    def fetch_page(
        self,
        request: KlinePageRequestV1,
        *,
        attempt_ordinal: int,
    ) -> RawHttpResponseV1: ...


@dataclass(frozen=True)
class NormalizedHistoryRowV1:
    venue: str
    symbol: str
    venue_symbol: str
    interval: str
    bar_open_ts: int
    bar_close_ts: int
    open: str
    high: str
    low: str
    close: str
    volume_contracts: str
    turnover_quote: str
    source_page_receipt_hash: str
    source_raw_body_sha256: str
    source_row_ordinal: int
    contract_version: str = STRICT_HISTORY_NORMALIZED_ROW_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_NORMALIZED_ROW_VERSION:
            raise HistoryPayloadSchemaError("normalized_row_contract_version_mismatch")
        _safe_identifier(self.venue, field="venue")
        _safe_symbol(self.symbol, field="symbol")
        _safe_symbol(self.venue_symbol, field="venue_symbol")
        canonical = canonical_frame_timeframe(self.interval)
        object.__setattr__(self, "interval", canonical)
        step = interval_seconds(canonical)
        if type(self.bar_open_ts) is not int or self.bar_open_ts % step:
            raise HistoryPayloadValueError("normalized_bar_open_is_invalid")
        if type(self.bar_close_ts) is not int or self.bar_close_ts != self.bar_open_ts + step:
            raise HistoryPayloadValueError("normalized_bar_close_is_invalid")
        decimal_fields = (
            "open",
            "high",
            "low",
            "close",
            "volume_contracts",
            "turnover_quote",
        )
        try:
            values = {name: Decimal(getattr(self, name)) for name in decimal_fields}
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise HistoryPayloadValueError(
                "normalized_row_contains_invalid_decimal"
            ) from exc
        if any(not value.is_finite() for value in values.values()):
            raise HistoryPayloadValueError("normalized_row_contains_nonfinite_value")
        for name in decimal_fields:
            if getattr(self, name) != _canonical_decimal(
                getattr(self, name), field=f"normalized_{name}"
            ):
                raise HistoryPayloadValueError(
                    "normalized_row_decimal_is_not_canonical"
                )
        if values["open"] <= 0 or values["high"] <= 0 or values["low"] <= 0 or values["close"] <= 0:
            raise HistoryPayloadValueError("normalized_prices_must_be_positive")
        if values["high"] < max(values["open"], values["close"]) or values["low"] > min(values["open"], values["close"]) or values["low"] > values["high"]:
            raise HistoryPayloadValueError("normalized_ohlc_geometry_is_invalid")
        if values["volume_contracts"] < 0 or values["turnover_quote"] < 0:
            raise HistoryPayloadValueError("normalized_volume_or_turnover_is_negative")
        if not _SHA256_RE.fullmatch(self.source_page_receipt_hash) or not _SHA256_RE.fullmatch(self.source_raw_body_sha256):
            raise HistoryPayloadValueError("normalized_source_hash_is_invalid")
        if type(self.source_row_ordinal) is not int or self.source_row_ordinal < 0:
            raise HistoryPayloadValueError("normalized_source_row_ordinal_is_invalid")

    @property
    def logical_row_hash(self) -> str:
        return _sha256_payload(self.market_dict())

    def market_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "venue": self.venue,
            "symbol": self.symbol,
            "venue_symbol": self.venue_symbol,
            "interval": self.interval,
            "bar_open_ts": self.bar_open_ts,
            "bar_close_ts": self.bar_close_ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume_contracts": self.volume_contracts,
            "turnover_quote": self.turnover_quote,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self.market_dict(),
            "logical_row_hash": self.logical_row_hash,
            "source_page_receipt_hash": self.source_page_receipt_hash,
            "source_raw_body_sha256": self.source_raw_body_sha256,
            "source_row_ordinal": self.source_row_ordinal,
        }


@dataclass(frozen=True)
class HistoryPageReceiptV1:
    page_request: KlinePageRequestV1
    attempt_receipt_hashes: tuple[str, ...]
    request_started_at: float
    received_at: float
    http_status: int
    api_code: int
    raw_body_sha256: str
    raw_body_length: int
    row_count: int
    first_bar_open_ts: int
    last_bar_open_ts: int
    normalized_page_hash: str
    contract_version: str = STRICT_HISTORY_PAGE_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_PAGE_RECEIPT_VERSION:
            raise HistoryPayloadSchemaError("page_receipt_contract_version_mismatch")
        if not isinstance(self.attempt_receipt_hashes, tuple) or not self.attempt_receipt_hashes:
            raise HistoryPayloadValueError("page_receipt_attempts_are_missing")
        for digest in (
            *self.attempt_receipt_hashes,
            self.raw_body_sha256,
            self.normalized_page_hash,
        ):
            if not _SHA256_RE.fullmatch(digest):
                raise HistoryPayloadValueError("page_receipt_hash_is_invalid")
        started = _finite_time(self.request_started_at, field="request_started_at")
        received = _finite_time(self.received_at, field="received_at")
        object.__setattr__(self, "request_started_at", started)
        object.__setattr__(self, "received_at", received)
        if received < started:
            raise HistoryPayloadValueError("page_receipt_timing_is_invalid")
        if type(self.http_status) is not int or self.http_status != 200:
            raise HistoryPayloadValueError("page_receipt_http_status_is_invalid")
        if type(self.api_code) is not int or self.api_code != 0:
            raise HistoryPayloadValueError("page_receipt_api_code_is_invalid")
        if type(self.raw_body_length) is not int or self.raw_body_length < 0:
            raise HistoryPayloadValueError("page_receipt_body_length_is_invalid")
        if type(self.row_count) is not int or self.row_count < 1:
            raise HistoryPayloadValueError("page_receipt_row_count_is_invalid")
        if self.row_count != self.page_request.expected_row_count:
            raise HistoryPayloadValueError("page_receipt_row_count_mismatch")
        if self.first_bar_open_ts != self.page_request.start_open_ts or self.last_bar_open_ts != self.page_request.end_open_ts_inclusive:
            raise HistoryPayloadValueError("page_receipt_range_mismatch")

    @property
    def page_receipt_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "page_request": self.page_request.as_dict(),
            "page_id": self.page_request.page_id,
            "attempt_receipt_hashes": list(self.attempt_receipt_hashes),
            "request_started_at": self.request_started_at,
            "received_at": self.received_at,
            "http_status": self.http_status,
            "api_code": self.api_code,
            "raw_body_sha256": self.raw_body_sha256,
            "raw_body_length": self.raw_body_length,
            "row_count": self.row_count,
            "first_bar_open_ts": self.first_bar_open_ts,
            "last_bar_open_ts": self.last_bar_open_ts,
            "normalized_page_hash": self.normalized_page_hash,
        }


@dataclass(frozen=True)
class HistoryCollectionManifestV1:
    request: HistoryRangeRequestV1
    page_receipts: tuple[HistoryPageReceiptV1, ...]
    normalized_logical_hash: str
    normalized_shard_sha256: str
    expected_row_count: int
    actual_row_count: int
    first_bar_open_ts: int
    last_bar_open_ts: int
    completed_at: float
    contract_version: str = STRICT_HISTORY_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_MANIFEST_VERSION:
            raise HistoryPayloadSchemaError("history_manifest_version_mismatch")
        if not isinstance(self.request, HistoryRangeRequestV1):
            raise HistoryPayloadValueError("history_manifest_request_is_invalid")
        if not isinstance(self.page_receipts, tuple) or not all(
            isinstance(receipt, HistoryPageReceiptV1)
            for receipt in self.page_receipts
        ):
            raise HistoryPayloadValueError(
                "history_manifest_page_receipts_are_not_immutable"
            )
        if len(self.page_receipts) != self.request.required_pages:
            raise HistoryPayloadValueError("history_manifest_page_count_mismatch")
        if tuple(receipt.page_request.page_ordinal for receipt in self.page_receipts) != tuple(range(len(self.page_receipts))):
            raise HistoryPayloadValueError("history_manifest_page_order_mismatch")
        flattened: list[int] = []
        for receipt in self.page_receipts:
            page = receipt.page_request
            if len(receipt.attempt_receipt_hashes) > self.request.max_attempts_per_page:
                raise HistoryPayloadValueError(
                    "history_manifest_attempt_count_exceeds_request"
                )
            if (
                page.range_request_id != self.request.request_id
                or page.endpoint_identity != self.request.endpoint_identity
                or page.venue_symbol != self.request.venue_symbol
                or page.interval != self.request.interval
            ):
                raise HistoryPayloadValueError(
                    "history_manifest_page_identity_mismatch"
                )
            flattened.extend(page.expected_timestamps())
        if tuple(flattened) != self.request.expected_timestamps():
            raise HistoryPayloadValueError("history_manifest_page_grid_mismatch")
        for digest in (self.normalized_logical_hash, self.normalized_shard_sha256):
            if not _SHA256_RE.fullmatch(digest):
                raise HistoryPayloadValueError("history_manifest_hash_is_invalid")
        if self.expected_row_count != self.request.expected_row_count or self.actual_row_count != self.expected_row_count:
            raise HistoryPayloadValueError("history_manifest_row_count_mismatch")
        expected = self.request.expected_timestamps()
        if self.first_bar_open_ts != expected[0] or self.last_bar_open_ts != expected[-1]:
            raise HistoryPayloadValueError("history_manifest_range_mismatch")
        completed = _finite_time(self.completed_at, field="completed_at")
        object.__setattr__(self, "completed_at", completed)
        if completed != max(receipt.received_at for receipt in self.page_receipts):
            raise HistoryPayloadValueError("history_manifest_completion_time_mismatch")

    @property
    def manifest_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "history_contract_version": STRICT_HISTORY_CONTRACT_VERSION,
            "history_contract_hash": strict_history_contract_hash(),
            "request": self.request.as_dict(),
            "request_id": self.request.request_id,
            "page_receipts": [receipt.as_dict() for receipt in self.page_receipts],
            "page_receipt_hashes": [
                receipt.page_receipt_hash for receipt in self.page_receipts
            ],
            "normalized_units": {
                "volume_contracts": "exchange_reported_contract_count",
                "turnover_quote": "exchange_reported_amount",
                "base_volume": "unavailable_without_point_in_time_contract_size",
            },
            "normalized_logical_hash": self.normalized_logical_hash,
            "normalized_shard_sha256": self.normalized_shard_sha256,
            "expected_row_count": self.expected_row_count,
            "actual_row_count": self.actual_row_count,
            "first_bar_open_ts": self.first_bar_open_ts,
            "last_bar_open_ts": self.last_bar_open_ts,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True)
class CompleteHistoryShardV1:
    rows: tuple[NormalizedHistoryRowV1, ...]
    manifest: HistoryCollectionManifestV1

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple) or not all(
            isinstance(row, NormalizedHistoryRowV1) for row in self.rows
        ):
            raise HistoryPayloadValueError(
                "complete_history_rows_are_not_immutable"
            )
        if not isinstance(self.manifest, HistoryCollectionManifestV1):
            raise HistoryPayloadValueError("complete_history_manifest_is_invalid")
        request = self.manifest.request
        expected = request.expected_timestamps()
        if tuple(row.bar_open_ts for row in self.rows) != expected:
            raise HistoryPayloadValueError("complete_history_rows_do_not_match_grid")
        receipt_by_hash = {
            receipt.page_receipt_hash: receipt
            for receipt in self.manifest.page_receipts
        }
        rows_by_receipt: dict[str, list[NormalizedHistoryRowV1]] = {
            digest: [] for digest in receipt_by_hash
        }
        for row in self.rows:
            if (
                row.venue != request.venue
                or row.symbol != request.symbol
                or row.venue_symbol != request.venue_symbol
                or row.interval != request.interval
            ):
                raise HistoryPayloadValueError(
                    "complete_history_row_identity_mismatch"
                )
            receipt = receipt_by_hash.get(row.source_page_receipt_hash)
            if receipt is None or row.source_raw_body_sha256 != receipt.raw_body_sha256:
                raise HistoryPayloadValueError(
                    "complete_history_row_source_mismatch"
                )
            if row.bar_open_ts not in receipt.page_request.expected_timestamps():
                raise HistoryPayloadValueError(
                    "complete_history_row_page_range_mismatch"
                )
            rows_by_receipt[row.source_page_receipt_hash].append(row)
        for digest, page_rows in rows_by_receipt.items():
            receipt = receipt_by_hash[digest]
            if len(page_rows) != receipt.row_count:
                raise HistoryPayloadValueError(
                    "complete_history_page_row_count_mismatch"
                )
            if {row.source_row_ordinal for row in page_rows} != set(
                range(receipt.row_count)
            ):
                raise HistoryPayloadValueError(
                    "complete_history_source_ordinals_mismatch"
                )
            page_hash = _sha256_payload(
                [row.market_dict() for row in sorted(page_rows, key=lambda item: item.bar_open_ts)]
            )
            if page_hash != receipt.normalized_page_hash:
                raise HistoryPayloadValueError(
                    "complete_history_normalized_page_hash_mismatch"
                )
        logical_hash = _sha256_payload([row.market_dict() for row in self.rows])
        if logical_hash != self.manifest.normalized_logical_hash:
            raise HistoryPayloadValueError("complete_history_logical_hash_mismatch")
        if _sha256_bytes(self.normalized_jsonl_bytes()) != self.manifest.normalized_shard_sha256:
            raise HistoryPayloadValueError("complete_history_shard_hash_mismatch")

    def normalized_jsonl_bytes(self) -> bytes:
        return b"".join(_canonical_bytes(row.as_dict()) + b"\n" for row in self.rows)

    def to_frame(self) -> pd.DataFrame:
        """Return an explicit float64 adapter; hashes remain Decimal-string based."""

        index = pd.to_datetime(
            [row.bar_open_ts for row in self.rows], unit="s", utc=True
        )
        frame = pd.DataFrame(
            {
                "open": [_decimal_float(row.open) for row in self.rows],
                "high": [_decimal_float(row.high) for row in self.rows],
                "low": [_decimal_float(row.low) for row in self.rows],
                "close": [_decimal_float(row.close) for row in self.rows],
                "volume": [_decimal_float(row.volume_contracts) for row in self.rows],
                "turnover": [_decimal_float(row.turnover_quote) for row in self.rows],
            },
            index=index,
        )
        frame.index.name = "datetime"
        frame.attrs.update(
            {
                "history_contract_version": STRICT_HISTORY_CONTRACT_VERSION,
                "history_contract_hash": strict_history_contract_hash(),
                "history_request_id": self.manifest.request.request_id,
                "history_manifest_hash": self.manifest.manifest_hash,
                "normalized_logical_hash": self.manifest.normalized_logical_hash,
            }
        )
        return frame

    def to_min1_aggregation_inputs(self):
        """Return the explicit S2→S3 adapter with full-manifest lineage.

        The DataFrame is a projection, not the semantic identity.  Every S3
        receipt therefore binds both the exact raw page and this completed S2
        manifest; a Decimal-level source change remains visible even if float64
        projection rounds to the same numeric value.
        """

        if self.manifest.request.interval != "Min1":
            raise HistoryRangeContractError(
                "history_shard_is_not_min1_aggregation_input"
            )
        from trading.market_data.min1_aggregation import (
            Min1BarReceiptV1,
            normalized_min1_source_row_hash,
        )

        page_by_hash = {
            receipt.page_receipt_hash: receipt
            for receipt in self.manifest.page_receipts
        }
        receipts = tuple(
            Min1BarReceiptV1(
                bar_open_ts=float(row.bar_open_ts),
                request_started_at=page_by_hash[
                    row.source_page_receipt_hash
                ].request_started_at,
                received_at=page_by_hash[
                    row.source_page_receipt_hash
                ].received_at,
                source_content_hash=row.source_raw_body_sha256,
                source_lineage_hash=self.manifest.manifest_hash,
                normalized_row_hash=normalized_min1_source_row_hash(
                    venue=row.venue,
                    symbol=row.symbol,
                    venue_symbol=row.venue_symbol,
                    bar_open_ts=float(row.bar_open_ts),
                    values={
                        "open": _decimal_float(row.open),
                        "high": _decimal_float(row.high),
                        "low": _decimal_float(row.low),
                        "close": _decimal_float(row.close),
                        "volume": _decimal_float(row.volume_contracts),
                        "turnover": _decimal_float(row.turnover_quote),
                    },
                ),
            )
            for row in self.rows
        )
        return self.to_frame(), receipts


class StrictHistoryArtifactStoreV1:
    """Immutable, content-addressed artifacts under an explicit non-legacy root."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()
        lowered = [part.lower() for part in self.root.parts]
        if any(
            lowered[index : index + 2] == ["data", "history"]
            for index in range(max(0, len(lowered) - 1))
        ):
            raise HistoryStorageError("legacy_history_root_is_forbidden")
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fsync_parent(path: Path) -> None:
        try:
            descriptor = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def _publish_immutable(self, relative: Path, payload: bytes) -> Path:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                existing = target.read_bytes()
            except OSError as exc:
                raise HistoryStorageError("history_artifact_read_failed") from exc
            if existing != payload:
                raise HistoryArtifactConflictError()
            return target

        # Do not repeat the 64-hex content-addressed target name in the temp
        # filename.  Deep pytest/pilot roots on Windows otherwise exceed the
        # legacy MAX_PATH limit before the final immutable path itself does.
        temporary = target.with_name(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.read_bytes() != payload:
                    raise HistoryArtifactConflictError()
            except OSError as exc:
                raise HistoryStorageError("history_artifact_atomic_publish_failed") from exc
            self._fsync_parent(target)
        except HistoryStorageError:
            raise
        except OSError as exc:
            raise HistoryStorageError("history_artifact_write_failed") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return target

    def persist_raw_attempt(self, response: RawHttpResponseV1) -> None:
        raw_hash = response.raw_body_sha256
        self._publish_immutable(
            Path("raw") / "sha256" / raw_hash[:2] / f"{raw_hash}.bin",
            response.body,
        )
        self._publish_immutable(
            Path("attempts") / f"{response.attempt_receipt_hash}.json",
            _canonical_bytes(response.receipt_dict()) + b"\n",
        )

    def persist_transport_failure(
        self, receipt: TransportFailureReceiptV1
    ) -> None:
        self._publish_immutable(
            Path("attempts") / f"{receipt.attempt_receipt_hash}.json",
            _canonical_bytes(receipt.receipt_dict()) + b"\n",
        )

    def _verify_source_artifacts(self, shard: CompleteHistoryShardV1) -> None:
        prior_collection_terminal_at: float | None = None
        for receipt in shard.manifest.page_receipts:
            prior_terminal_at = prior_collection_terminal_at
            for attempt_index, attempt_hash in enumerate(
                receipt.attempt_receipt_hashes
            ):
                attempt_path = self.root / "attempts" / f"{attempt_hash}.json"
                try:
                    attempt_bytes = attempt_path.read_bytes()
                    attempt_payload = json.loads(
                        attempt_bytes.decode("utf-8"),
                        parse_constant=_reject_json_constant,
                        object_pairs_hook=_unique_object,
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise HistoryStorageError(
                        "history_attempt_artifact_is_missing_or_invalid"
                    ) from exc
                if (
                    not isinstance(attempt_payload, dict)
                    or attempt_bytes != _canonical_bytes(attempt_payload) + b"\n"
                    or _sha256_payload(attempt_payload) != attempt_hash
                    or attempt_payload.get("page_id")
                    != receipt.page_request.page_id
                    or attempt_payload.get("attempt_ordinal") != attempt_index
                ):
                    raise HistoryStorageError(
                        "history_attempt_artifact_hash_mismatch"
                    )
                is_http_attempt = set(attempt_payload) == _HTTP_ATTEMPT_KEYS
                is_transport_failure = (
                    set(attempt_payload) == _TRANSPORT_FAILURE_KEYS
                )
                if not is_http_attempt and not is_transport_failure:
                    raise HistoryStorageError(
                        "history_attempt_artifact_schema_mismatch"
                    )
                if (
                    is_http_attempt
                    and attempt_payload.get("contract_version")
                    != STRICT_HISTORY_RAW_ATTEMPT_VERSION
                ) or (
                    is_transport_failure
                    and attempt_payload.get("contract_version")
                    != STRICT_HISTORY_TRANSPORT_FAILURE_VERSION
                ):
                    raise HistoryStorageError(
                        "history_attempt_artifact_version_mismatch"
                    )
                if attempt_payload.get("page_request") != receipt.page_request.as_dict():
                    raise HistoryStorageError(
                        "history_attempt_page_request_mismatch"
                    )
                started = attempt_payload.get("request_started_at")
                if isinstance(started, bool) or not isinstance(started, (int, float)):
                    raise HistoryStorageError(
                        "history_attempt_artifact_timing_mismatch"
                    )
                started = float(started)
                terminal = attempt_payload.get(
                    "received_at", attempt_payload.get("failed_at")
                )
                if isinstance(terminal, bool) or not isinstance(
                    terminal, (int, float)
                ):
                    raise HistoryStorageError(
                        "history_attempt_artifact_timing_mismatch"
                    )
                terminal = float(terminal)
                if (
                    started < shard.manifest.request.collection_as_of_ts
                    or terminal < started
                    or (
                        prior_terminal_at is not None
                        and started < prior_terminal_at
                    )
                ):
                    raise HistoryStorageError(
                        "history_attempt_artifact_timing_mismatch"
                    )
                prior_terminal_at = terminal
                if is_transport_failure:
                    if (
                        attempt_payload.get("outcome")
                        not in {"network_error", "timeout"}
                        or attempt_payload.get("http_status") is not None
                        or attempt_payload.get("raw_body_sha256") is not None
                    ):
                        raise HistoryStorageError(
                            "history_transport_failure_artifact_mismatch"
                        )
                else:
                    attempt_raw_hash = attempt_payload.get("raw_body_sha256")
                    if not isinstance(attempt_raw_hash, str) or not _SHA256_RE.fullmatch(
                        attempt_raw_hash
                    ):
                        raise HistoryStorageError(
                            "history_http_attempt_raw_hash_mismatch"
                        )
                    attempt_raw_path = (
                        self.root
                        / "raw"
                        / "sha256"
                        / attempt_raw_hash[:2]
                        / f"{attempt_raw_hash}.bin"
                    )
                    try:
                        attempt_raw = attempt_raw_path.read_bytes()
                    except OSError as exc:
                        raise HistoryStorageError(
                            "history_raw_artifact_is_missing"
                        ) from exc
                    if _sha256_bytes(attempt_raw) != attempt_raw_hash:
                        raise HistoryStorageError(
                            "history_raw_artifact_hash_mismatch"
                        )
                if attempt_index == len(receipt.attempt_receipt_hashes) - 1:
                    if (
                        attempt_payload.get("http_status") != 200
                        or attempt_payload.get("raw_body_sha256")
                        != receipt.raw_body_sha256
                        or started != receipt.request_started_at
                        or terminal != receipt.received_at
                    ):
                        raise HistoryStorageError(
                            "history_final_attempt_artifact_mismatch"
                        )
                elif is_http_attempt:
                    status = attempt_payload.get("http_status")
                    if type(status) is not int or not (
                        status in {408, 425, 429} or status >= 500
                    ):
                        raise HistoryStorageError(
                            "history_nonfinal_http_attempt_is_not_retryable"
                        )
            raw_path = (
                self.root
                / "raw"
                / "sha256"
                / receipt.raw_body_sha256[:2]
                / f"{receipt.raw_body_sha256}.bin"
            )
            try:
                raw = raw_path.read_bytes()
            except OSError as exc:
                raise HistoryStorageError(
                    "history_raw_artifact_is_missing"
                ) from exc
            if _sha256_bytes(raw) != receipt.raw_body_sha256:
                raise HistoryStorageError("history_raw_artifact_hash_mismatch")
            prior_collection_terminal_at = prior_terminal_at

    def publish_complete(self, shard: CompleteHistoryShardV1) -> None:
        # Revalidate the immutable source graph immediately before publishing
        # either normalized bytes or the completion marker.
        self._verify_source_artifacts(shard)
        request_id = shard.manifest.request.request_id
        normalized = shard.normalized_jsonl_bytes()
        self._publish_immutable(
            Path("normalized")
            / request_id
            / f"{shard.manifest.normalized_shard_sha256}.jsonl",
            normalized,
        )
        # The manifest is the completion marker and is deliberately written last.
        self._publish_immutable(
            Path("collections") / request_id / "manifest.json",
            _canonical_bytes(
                {
                    **shard.manifest.as_dict(),
                    "manifest_hash": shard.manifest.manifest_hash,
                }
            )
            + b"\n",
        )

    def has_complete_manifest(self, request_id: str) -> bool:
        if not _SHA256_RE.fullmatch(request_id):
            raise HistoryStorageError("history_request_id_is_invalid")
        return (self.root / "collections" / request_id / "manifest.json").is_file()

    def verify_complete_artifacts(self, shard: CompleteHistoryShardV1) -> None:
        request_id = shard.manifest.request.request_id
        manifest_path = self.root / "collections" / request_id / "manifest.json"
        normalized_path = (
            self.root
            / "normalized"
            / request_id
            / f"{shard.manifest.normalized_shard_sha256}.jsonl"
        )
        try:
            manifest_payload = manifest_path.read_bytes()
            normalized_payload = normalized_path.read_bytes()
        except OSError as exc:
            raise HistoryStorageError("history_complete_artifact_is_missing") from exc
        expected_manifest = _canonical_bytes(
            {
                **shard.manifest.as_dict(),
                "manifest_hash": shard.manifest.manifest_hash,
            }
        ) + b"\n"
        if manifest_payload != expected_manifest or _sha256_bytes(normalized_payload) != shard.manifest.normalized_shard_sha256:
            raise HistoryStorageError("history_complete_artifact_hash_mismatch")
        self._verify_source_artifacts(shard)


@dataclass(frozen=True)
class _NormalizedPage:
    rows: tuple[NormalizedHistoryRowV1, ...]
    receipt: HistoryPageReceiptV1


def _parse_mexc_page(
    response: RawHttpResponseV1,
    *,
    range_request: HistoryRangeRequestV1,
    attempt_receipt_hashes: tuple[str, ...],
) -> _NormalizedPage:
    if response.page_request.range_request_id != range_request.request_id:
        raise HistoryPayloadRangeError("page_response_belongs_to_another_range")
    if response.http_status != 200:
        raise HistoryHttpStatusError()
    if response.request_started_at < range_request.collection_as_of_ts:
        raise HistoryPayloadRangeError(
            "page_request_started_before_collection_as_of"
        )
    try:
        payload = json.loads(
            response.body.decode("utf-8"),
            parse_float=Decimal,
            parse_int=int,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except HistoryJsonDecodeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HistoryJsonDecodeError() from exc
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise HistoryApiRejectedError()
    api_code = payload.get("code")
    if api_code == "0":
        api_code = 0
    if type(api_code) is not int or api_code != 0:
        raise HistoryApiRejectedError()
    data = payload.get("data")
    if not isinstance(data, dict) or any(
        key not in data or not isinstance(data[key], list) for key in _REQUIRED_ARRAYS
    ):
        raise HistoryPayloadSchemaError("history_payload_required_arrays_missing")
    lengths = {len(data[key]) for key in _REQUIRED_ARRAYS}
    if len(lengths) != 1:
        raise HistoryPayloadSchemaError("history_payload_array_lengths_differ")
    row_count = next(iter(lengths))
    expected = response.page_request.expected_timestamps()
    if row_count == 0:
        raise HistoryIncompleteRangeError(
            "empty_success", missing_timestamps=expected
        )

    raw_rows: list[tuple[int, int, dict[str, str]]] = []
    seen: set[int] = set()
    for ordinal in range(row_count):
        timestamp = _parse_epoch_second(data["time"][ordinal])
        if timestamp in seen:
            raise HistoryDuplicateTimestampError()
        seen.add(timestamp)
        values = {
            "open": _canonical_decimal(data["open"][ordinal], field="open"),
            "high": _canonical_decimal(data["high"][ordinal], field="high"),
            "low": _canonical_decimal(data["low"][ordinal], field="low"),
            "close": _canonical_decimal(data["close"][ordinal], field="close"),
            "volume_contracts": _canonical_decimal(data["vol"][ordinal], field="vol"),
            "turnover_quote": _canonical_decimal(data["amount"][ordinal], field="amount"),
        }
        raw_rows.append((timestamp, ordinal, values))

    actual = set(seen)
    expected_set = set(expected)
    unexpected = tuple(sorted(actual - expected_set))
    missing = tuple(sorted(expected_set - actual))
    if unexpected:
        raise HistoryIncompleteRangeError(
            "unexpected_timestamps",
            missing_timestamps=missing,
            unexpected_timestamps=unexpected,
        )
    if missing:
        raise HistoryIncompleteRangeError(
            "missing_timestamps", missing_timestamps=missing
        )

    raw_hash = response.raw_body_sha256
    provisional_rows: list[NormalizedHistoryRowV1] = []
    # The page receipt hash includes the normalized page hash, while rows bind to
    # that receipt.  Compute the market-only page hash first to avoid a cycle.
    market_rows = []
    for timestamp, _ordinal, values in sorted(raw_rows, key=lambda item: item[0]):
        market_rows.append(
            {
                "contract_version": STRICT_HISTORY_NORMALIZED_ROW_VERSION,
                "venue": range_request.venue,
                "symbol": range_request.symbol,
                "venue_symbol": range_request.venue_symbol,
                "interval": range_request.interval,
                "bar_open_ts": timestamp,
                "bar_close_ts": timestamp + range_request.interval_seconds,
                **values,
            }
        )
    normalized_page_hash = _sha256_payload(market_rows)
    receipt_without_hash = {
        "contract_version": STRICT_HISTORY_PAGE_RECEIPT_VERSION,
        "page_request": response.page_request.as_dict(),
        "page_id": response.page_request.page_id,
        "attempt_receipt_hashes": list(attempt_receipt_hashes),
        "request_started_at": response.request_started_at,
        "received_at": response.received_at,
        "http_status": response.http_status,
        "api_code": api_code,
        "raw_body_sha256": raw_hash,
        "raw_body_length": len(response.body),
        "row_count": row_count,
        "first_bar_open_ts": expected[0],
        "last_bar_open_ts": expected[-1],
        "normalized_page_hash": normalized_page_hash,
    }
    page_receipt_hash = _sha256_payload(receipt_without_hash)
    for timestamp, ordinal, values in sorted(raw_rows, key=lambda item: item[0]):
        provisional_rows.append(
            NormalizedHistoryRowV1(
                venue=range_request.venue,
                symbol=range_request.symbol,
                venue_symbol=range_request.venue_symbol,
                interval=range_request.interval,
                bar_open_ts=timestamp,
                bar_close_ts=timestamp + range_request.interval_seconds,
                source_page_receipt_hash=page_receipt_hash,
                source_raw_body_sha256=raw_hash,
                source_row_ordinal=ordinal,
                **values,
            )
        )
    receipt = HistoryPageReceiptV1(
        page_request=response.page_request,
        attempt_receipt_hashes=attempt_receipt_hashes,
        request_started_at=response.request_started_at,
        received_at=response.received_at,
        http_status=response.http_status,
        api_code=api_code,
        raw_body_sha256=raw_hash,
        raw_body_length=len(response.body),
        row_count=row_count,
        first_bar_open_ts=expected[0],
        last_bar_open_ts=expected[-1],
        normalized_page_hash=normalized_page_hash,
    )
    if receipt.page_receipt_hash != page_receipt_hash:
        raise HistoryPayloadValueError("page_receipt_hash_construction_drift")
    return _NormalizedPage(rows=tuple(provisional_rows), receipt=receipt)


class StrictMexcHistoryCollectorV1:
    """Collect one exact range or fail without publishing a success shard."""

    def __init__(
        self,
        *,
        transport: RawKlinePageTransport,
        store: StrictHistoryArtifactStoreV1,
    ):
        if transport is None or not callable(getattr(transport, "fetch_page", None)):
            raise HistoryRangeContractError("history_transport_is_required")
        if not isinstance(store, StrictHistoryArtifactStoreV1):
            raise HistoryRangeContractError("history_store_is_required")
        self.transport = transport
        self.store = store

    @staticmethod
    def plan_pages(request: HistoryRangeRequestV1) -> tuple[KlinePageRequestV1, ...]:
        if request.required_pages > request.max_pages:
            raise HistoryIncompleteRangeError("page_budget_exceeded")
        pages: list[KlinePageRequestV1] = []
        step = request.interval_seconds
        cursor = request.start_open_ts
        for ordinal in range(request.required_pages):
            remaining = (request.end_open_ts_exclusive - cursor) // step
            count = min(request.page_size, remaining)
            end_inclusive = cursor + (count - 1) * step
            pages.append(
                KlinePageRequestV1(
                    range_request_id=request.request_id,
                    endpoint_identity=request.endpoint_identity,
                    venue_symbol=request.venue_symbol,
                    interval=request.interval,
                    page_ordinal=ordinal,
                    start_open_ts=cursor,
                    end_open_ts_inclusive=end_inclusive,
                    expected_row_count=count,
                )
            )
            cursor = end_inclusive + step
        if cursor != request.end_open_ts_exclusive:
            raise HistoryRangeContractError("history_page_plan_does_not_cover_range")
        return tuple(pages)

    def collect_range(self, request: HistoryRangeRequestV1) -> CompleteHistoryShardV1:
        if not isinstance(request, HistoryRangeRequestV1):
            raise HistoryRangeContractError("history_request_is_invalid")
        pages = self.plan_pages(request)
        rows: list[NormalizedHistoryRowV1] = []
        receipts: list[HistoryPageReceiptV1] = []
        prior_page_terminal_at: float | None = None
        for page in pages:
            attempt_hashes: list[str] = []
            response: RawHttpResponseV1 | None = None
            prior_attempt_terminal_at = prior_page_terminal_at
            for attempt_ordinal in range(request.max_attempts_per_page):
                try:
                    candidate = self.transport.fetch_page(
                        page, attempt_ordinal=attempt_ordinal
                    )
                except (HistoryNetworkError, HistoryTimeoutError) as exc:
                    failure = exc.failure_receipt
                    if not isinstance(failure, TransportFailureReceiptV1):
                        raise HistoryTransportError(
                            None, "history_transport_failure_receipt_is_required"
                        ) from exc
                    if (
                        failure.page_request != page
                        or failure.attempt_ordinal != attempt_ordinal
                    ):
                        raise HistoryTransportError(
                            None, "history_transport_failure_receipt_mismatch"
                        ) from exc
                    if isinstance(exc, HistoryNetworkError) and failure.outcome != "network_error":
                        raise HistoryTransportError(
                            None, "history_transport_failure_outcome_mismatch"
                        ) from exc
                    if isinstance(exc, HistoryTimeoutError) and failure.outcome != "timeout":
                        raise HistoryTransportError(
                            None, "history_transport_failure_outcome_mismatch"
                        ) from exc
                    self.store.persist_transport_failure(failure)
                    if failure.request_started_at < request.collection_as_of_ts:
                        raise HistoryTransportError(
                            None, "history_attempt_started_before_collection_as_of"
                        ) from exc
                    if (
                        prior_attempt_terminal_at is not None
                        and failure.request_started_at < prior_attempt_terminal_at
                    ):
                        raise HistoryTransportError(
                            None, "history_attempt_timing_regressed"
                        ) from exc
                    attempt_hashes.append(failure.attempt_receipt_hash)
                    prior_attempt_terminal_at = failure.failed_at
                    if attempt_ordinal + 1 >= request.max_attempts_per_page:
                        raise
                    continue
                if not isinstance(candidate, RawHttpResponseV1):
                    raise HistoryTransportError(
                        None, "history_transport_returned_invalid_receipt"
                    )
                if (
                    candidate.page_request != page
                    or candidate.attempt_ordinal != attempt_ordinal
                ):
                    raise HistoryTransportError(
                        None, "history_transport_returned_wrong_page"
                    )
                # This precedes HTTP, JSON, API and payload validation by contract.
                self.store.persist_raw_attempt(candidate)
                if candidate.request_started_at < request.collection_as_of_ts:
                    raise HistoryTransportError(
                        None, "history_attempt_started_before_collection_as_of"
                    )
                if (
                    prior_attempt_terminal_at is not None
                    and candidate.request_started_at < prior_attempt_terminal_at
                ):
                    raise HistoryTransportError(
                        None, "history_attempt_timing_regressed"
                    )
                attempt_hashes.append(candidate.attempt_receipt_hash)
                prior_attempt_terminal_at = candidate.received_at
                retryable_http = candidate.http_status in {408, 425, 429} or candidate.http_status >= 500
                if retryable_http and attempt_ordinal + 1 < request.max_attempts_per_page:
                    continue
                response = candidate
                break
            if response is None:
                raise HistoryTransportError(
                    None, "history_transport_attempts_exhausted_without_response"
                )
            normalized = _parse_mexc_page(
                response,
                range_request=request,
                attempt_receipt_hashes=tuple(attempt_hashes),
            )
            rows.extend(normalized.rows)
            receipts.append(normalized.receipt)
            prior_page_terminal_at = response.received_at

        expected = request.expected_timestamps()
        actual = tuple(row.bar_open_ts for row in rows)
        if len(set(actual)) != len(actual):
            raise HistoryDuplicateTimestampError()
        unexpected = tuple(sorted(set(actual) - set(expected)))
        missing = tuple(sorted(set(expected) - set(actual)))
        if unexpected:
            raise HistoryIncompleteRangeError(
                "unexpected_timestamps",
                missing_timestamps=missing,
                unexpected_timestamps=unexpected,
            )
        if missing or actual != expected:
            raise HistoryIncompleteRangeError(
                "missing_timestamps", missing_timestamps=missing
            )

        logical_hash = _sha256_payload([row.market_dict() for row in rows])
        normalized_bytes = b"".join(
            _canonical_bytes(row.as_dict()) + b"\n" for row in rows
        )
        manifest = HistoryCollectionManifestV1(
            request=request,
            page_receipts=tuple(receipts),
            normalized_logical_hash=logical_hash,
            normalized_shard_sha256=_sha256_bytes(normalized_bytes),
            expected_row_count=len(expected),
            actual_row_count=len(rows),
            first_bar_open_ts=expected[0],
            last_bar_open_ts=expected[-1],
            completed_at=max(receipt.received_at for receipt in receipts),
        )
        shard = CompleteHistoryShardV1(rows=tuple(rows), manifest=manifest)
        self.store.publish_complete(shard)
        return shard


_CONTRACT_SCHEMA = {
    "contract_version": STRICT_HISTORY_CONTRACT_VERSION,
    "raw_attempt_version": STRICT_HISTORY_RAW_ATTEMPT_VERSION,
    "transport_failure_version": STRICT_HISTORY_TRANSPORT_FAILURE_VERSION,
    "page_receipt_version": STRICT_HISTORY_PAGE_RECEIPT_VERSION,
    "normalized_row_version": STRICT_HISTORY_NORMALIZED_ROW_VERSION,
    "manifest_version": STRICT_HISTORY_MANIFEST_VERSION,
    "range": {
        "bounds": "half_open_bar_opens_start_inclusive_end_exclusive",
        "alignment": "fixed_interval_utc_epoch",
        "closed": "end_exclusive_lte_closed_boundary_at_collection_as_of",
        "pagination": "deterministic_expected_grid_pages_max_2000",
        "page_end_query": "inclusive_last_expected_bar_open",
        "budget_exhaustion": "typed_incomplete_before_transport",
        "attempt_limit": "one_to_ten_explicit_attempts_per_page",
    },
    "raw_evidence": {
        "body": "exact_application_bytes_content_addressed_before_parse",
        "attempt": "canonical_page_request_timing_http_safe_headers_body_hash",
        "safe_response_header_allowlist": sorted(_PUBLIC_SAFE_HEADER_NAMES),
        "endpoint": "explicit_versioned_identity_no_default_network_adapter",
        "canonical_path": "/api/v1/contract/kline/{venue_symbol}",
        "transport_failure": (
            "separate_network_or_timeout_receipt_without_fabricated_http_body"
        ),
        "retryable_http": [408, 425, 429, "5xx"],
        "attempt_chain": (
            "all_attempt_receipt_hashes_in_order_bound_to_page_with_"
            "monotonic_timing_across_attempts_and_pages"
        ),
        "request_timing": "every_successful_attempt_starts_at_or_after_collection_as_of",
    },
    "payload": {
        "envelope": "json_object_success_literal_true_and_code_integer_or_string_zero",
        "parallel_arrays": list(_REQUIRED_ARRAYS),
        "time": "integer_epoch_seconds_aligned_unique_in_page_grid",
        "numbers": (
            "finite_decimal_strings_or_json_numbers_bool_rejected_"
            "max_80_digits_exponent_minus100_to_plus100_context_independent"
        ),
        "prices": "positive_coherent_ohlc",
        "volume": "nonnegative_exchange_reported_contract_count",
        "turnover": "nonnegative_exchange_reported_amount_never_reconstructed",
        "raw_order": "may_be_permuted_but_source_ordinal_is_retained",
        "silent_drop_sort_dedup_gap_fill": False,
    },
    "completion": {
        "expected_grid": "every_timestamp_exactly_once",
        "empty_success": "typed_incomplete_for_nonempty_request",
        "success_marker": "manifest_published_last_only_for_complete_shard",
    },
    "serialization": {
        "containers": "frozen_dataclasses_with_exact_immutable_tuples",
        "http_attempt_keys": sorted(_HTTP_ATTEMPT_KEYS),
        "transport_failure_keys": sorted(_TRANSPORT_FAILURE_KEYS),
        "attempt_artifacts": (
            "every_referenced_attempt_exists_hashes_canonically_and_has_"
            "monotonic_timing_final_attempt_equals_page_receipt"
        ),
        "page_and_manifest_identity": "full_grid_source_and_hash_revalidated",
    },
    "hashes": {
        "raw": "sha256_exact_body_bytes",
        "normalized_page": "sha256_canonical_market_rows",
        "normalized_logical": "sha256_all_ordered_canonical_market_rows",
        "normalized_shard": "sha256_canonical_jsonl_with_lineage",
        "manifest": "sha256_contract_request_page_receipts_units_and_shard",
        "paths_and_local_clock_zone": "excluded",
    },
    "storage": {
        "root": "explicit_and_not_legacy_data_history",
        "publication": "same_volume_temp_fsync_atomic_hardlink_no_overwrite",
        "raw_and_attempts": "retained_on_later_failure",
        "normalized_and_manifest": "only_after_complete_validation",
        "manifest": "completion_marker_written_last",
    },
    "s2_to_s3": {
        "frame": "explicit_float64_projection_not_semantic_identity",
        "receipt_content": "exact_raw_body_sha256",
        "receipt_lineage": "complete_history_manifest_hash",
        "receipt_row_binding": "s3_normalized_min1_source_row_hash",
    },
    "errors": [
        "range_contract",
        "network",
        "timeout",
        "http_status",
        "json_decode",
        "api_rejected",
        "payload_schema",
        "payload_value",
        "payload_range",
        "duplicate_timestamp",
        "incomplete_range",
        "storage",
        "artifact_conflict",
    ],
}


def _computed_contract_hash() -> str:
    return _sha256_payload(_CONTRACT_SCHEMA)


def strict_history_contract_hash() -> str:
    """Return the pinned schema hash; in-place semantic edits fail closed."""

    digest = _computed_contract_hash()
    if digest != _PINNED_CONTRACT_HASH:
        raise RuntimeError("strict_history_contract_changed_without_version_bump")
    return digest
