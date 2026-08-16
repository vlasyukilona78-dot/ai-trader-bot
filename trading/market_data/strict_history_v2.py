"""Strict pre-pilot MEXC history contract with bounded restart evidence.

This is a new writer/reader line.  It deliberately does not modify or silently
reinterpret :mod:`trading.market_data.strict_history` v1.  A caller supplies a
versioned endpoint contract, resource limits, retry policy, raw transport,
clock and explicit artifact root.  There is no default network path.

The manifest publishes an evidence graph but is deliberately not a success
marker.  A separate positive admission marker is installed only after the
complete graph has been reconstructed from disk.  On Windows this proves
process-crash/restart-verifiable, atomic no-overwrite visibility, not parent
directory or sudden-power-loss durability.  Restart reconciliation is
read-only and never promotes, deletes, truncates or repairs artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from typing import Any, Iterator, Mapping, Protocol, Sequence
import uuid

import pandas as pd

from trading.market_data.bar_contract import interval_seconds
from trading.market_data.frame_provenance import canonical_frame_timeframe
from trading.market_data.mexc_futures_transport import (
    CompleteHttpAttemptEvidenceV1,
    EvidenceClock,
    HistoryResourceLimitsV1,
    HistoryRetryPolicyV1,
    HttpAttemptEvidenceV1,
    IncompleteHttpAttemptEvidenceV1,
    MexcFuturesEndpointContractV1,
    mexc_futures_transport_contract_hash,
    parse_http_attempt_evidence_v1,
    retry_after_delay_us,
)
from trading.market_data.strict_history import (
    HistoryApiRejectedError,
    HistoryArtifactConflictError,
    HistoryDuplicateTimestampError,
    HistoryHttpStatusError,
    HistoryIncompleteRangeError,
    HistoryJsonDecodeError,
    HistoryPayloadRangeError,
    HistoryPayloadSchemaError,
    HistoryPayloadValueError,
    HistoryRangeContractError,
    HistoryStorageError,
    HistoryTransportError,
    KlinePageRequestV1,
    NormalizedHistoryRowV1,
    StrictHistoryError,
    _canonical_bytes,
    _canonical_decimal,
    _decimal_float,
    _parse_epoch_second,
    _sha256_bytes,
    _sha256_payload,
)


STRICT_HISTORY_V2_CONTRACT_VERSION = "mexc_strict_history_v2"
STRICT_HISTORY_V2_PAGE_RECEIPT_VERSION = "mexc_history_page_receipt_v2"
STRICT_HISTORY_V2_MANIFEST_VERSION = "mexc_complete_history_manifest_v2"
STRICT_HISTORY_V2_RESTART_VERSION = "mexc_strict_history_restart_v1"
STRICT_HISTORY_V2_STORAGE_VERSION = "mexc_strict_history_storage_v1"
STRICT_HISTORY_V2_ADMISSION_VERSION = "mexc_strict_history_admission_v1"
STRICT_HISTORY_V2_SCOPE_VERSION = "mexc_strict_history_shard_scope_v1"

WINDOWS_NTFS_DURABILITY_PROFILE_V1 = "windows_ntfs_hardlink_best_effort_v1"
POSIX_DURABILITY_PROFILE_V1 = "posix_hardlink_directory_fsync_v1"

_PINNED_CONTRACT_HASH = (
    "cce9922317ec5f0008f3b293103f9f5a17504b7143f81af1845d9d4765c44086"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9_]{0,63}$")
_REQUIRED_ARRAYS = ("time", "open", "high", "low", "close", "vol", "amount")
_MAX_MEXC_PAGE_SIZE = 2_000
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_ATTEMPT_RECEIPT_BYTES = 1024 * 1024
_MAX_SCOPE_MARKER_BYTES = 256 * 1024
_MAX_TEMP_SCAN_ENTRIES = 20_000
_MAX_RESTART_SCAN_ENTRIES = 25_000
_ACTIVE_WRITER_LOCKS_GUARD = threading.Lock()
_ACTIVE_WRITER_LOCK_PATHS: set[str] = set()

_HARD_LIMITS = {
    "max_pages": 200,
    "max_rows": 400_000,
    "max_attempts_per_page": 10,
    "max_total_attempts": 2_000,
    "max_raw_body_bytes_per_attempt": 8_388_608,
    "max_total_raw_body_bytes": 268_435_456,
    "max_logical_storage_bytes": 536_870_912,
    "max_collection_runtime_us": 3_600_000_000,
    # The transport contract is the tighter authority: 60 seconds.
    "max_attempt_runtime_us": 60_000_000,
}


class HistoryBudgetExceededError(StrictHistoryError):
    code = "history_budget_exceeded"

    def __init__(self, resource: str, limit: int, observed: int):
        if not _IDENTIFIER_RE.fullmatch(resource):
            raise ValueError("history_budget_resource_is_invalid")
        if type(limit) is not int or type(observed) is not int:
            raise ValueError("history_budget_values_must_be_integers")
        self.resource = resource
        self.limit = limit
        self.observed = observed
        super().__init__(f"{self.code}.{resource}.{limit}.{observed}")


class HistoryArtifactCorruptionError(HistoryStorageError):
    code = "history_artifact_corruption"


class HistoryArtifactForkError(HistoryStorageError):
    code = "history_artifact_fork"


class HistoryRestartIncompleteError(HistoryStorageError):
    code = "history_restart_incomplete"


class RawHistoryTransportV2(Protocol):
    @property
    def endpoint_contract_hash(self) -> str: ...

    @property
    def resource_limits_hash(self) -> str: ...

    @property
    def retry_policy_hash(self) -> str: ...

    @property
    def transport_contract_hash(self) -> str: ...

    def fetch_page(
        self,
        request: KlinePageRequestV1,
        *,
        attempt_ordinal: int,
        prior_attempt: HttpAttemptEvidenceV1 | None = None,
    ) -> HttpAttemptEvidenceV1: ...


def _strict_int(value: object, *, field: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise HistoryRangeContractError(f"{field}_must_be_an_integer")
    if minimum is not None and value < minimum:
        raise HistoryRangeContractError(f"{field}_is_out_of_range")
    return value


def _safe_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise HistoryRangeContractError(f"{field}_is_invalid")
    return value


def _safe_symbol(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SYMBOL_RE.fullmatch(value):
        raise HistoryRangeContractError(f"{field}_is_invalid")
    return value


def _clock_us(clock: EvidenceClock, name: str) -> int:
    candidate = getattr(clock, name, None)
    value = candidate() if callable(candidate) else candidate
    if type(value) is not int or value < 0:
        raise HistoryRangeContractError(f"history_clock_{name}_is_invalid")
    return value


def _exact_keys(payload: object, expected: frozenset[str], *, code: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise HistoryArtifactCorruptionError(code)
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HistoryArtifactCorruptionError("history_artifact_duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise HistoryArtifactCorruptionError("history_artifact_nonfinite_json")


def _parse_canonical_json(payload: bytes, *, code: str) -> dict[str, Any]:
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except HistoryArtifactCorruptionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HistoryArtifactCorruptionError(code) from exc
    if not isinstance(decoded, dict):
        raise HistoryArtifactCorruptionError(code)
    try:
        canonical = _canonical_bytes(decoded)
    except StrictHistoryError as exc:
        raise HistoryArtifactCorruptionError(code) from exc
    if payload != canonical + b"\n":
        raise HistoryArtifactCorruptionError("history_artifact_is_not_canonical_json")
    return decoded


def _durability_profile_for_host() -> str:
    return (
        WINDOWS_NTFS_DURABILITY_PROFILE_V1
        if os.name == "nt"
        else POSIX_DURABILITY_PROFILE_V1
    )


def _storage_contract_payload(profile: str) -> dict[str, object]:
    if profile not in {
        WINDOWS_NTFS_DURABILITY_PROFILE_V1,
        POSIX_DURABILITY_PROFILE_V1,
    }:
        raise HistoryRangeContractError("history_storage_profile_is_invalid")
    windows = profile == WINDOWS_NTFS_DURABILITY_PROFILE_V1
    return {
        "contract_version": STRICT_HISTORY_V2_STORAGE_VERSION,
        "profile": profile,
        "same_directory_temp": True,
        "temp_file_fsync_before_publish": True,
        "final_name_install": "hardlink_create_new_no_overwrite",
        "atomic_visibility_not_graph_transaction": True,
        "directory_fsync": "best_effort_unsupported" if windows else "required",
        # The store is a multi-file graph and newly-created parent directories
        # are not one transaction.  Even on POSIX, a successful return is not a
        # proof that the complete graph survived sudden power loss.
        "power_loss_durable_at_return": False,
        "manifest_role": "reloadable_graph_evidence_not_success",
        "admission_role": "positive_success_after_full_graph_reload",
        "fresh_process_full_graph_verification_required": True,
        "automatic_repair_or_temp_promotion": False,
        "writer_concurrency": (
            "nonblocking_process_local_plus_os_file_lock_for_full_collection"
        ),
        "writer_lock_artifact": "persistent_sibling_outside_evidence_graph",
        "cooperating_writers_only": True,
    }


def storage_profile_hash(profile: str) -> str:
    return _sha256_payload(_storage_contract_payload(profile))


@dataclass(frozen=True)
class HistoryRangeRequestV2:
    venue: str
    symbol: str
    venue_symbol: str
    interval: str
    start_open_ts: int
    end_open_ts_exclusive: int
    collection_as_of_us: int
    endpoint_contract: MexcFuturesEndpointContractV1
    resource_limits: HistoryResourceLimitsV1
    retry_policy: HistoryRetryPolicyV1
    page_size: int = _MAX_MEXC_PAGE_SIZE
    storage_profile: str = ""
    contract_version: str = STRICT_HISTORY_V2_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_V2_CONTRACT_VERSION:
            raise HistoryRangeContractError("history_v2_contract_version_mismatch")
        object.__setattr__(self, "venue", _safe_identifier(self.venue, field="venue"))
        object.__setattr__(self, "symbol", _safe_symbol(self.symbol, field="symbol"))
        object.__setattr__(
            self, "venue_symbol", _safe_symbol(self.venue_symbol, field="venue_symbol")
        )
        try:
            canonical = canonical_frame_timeframe(self.interval)
        except (TypeError, ValueError) as exc:
            raise HistoryRangeContractError("history_interval_is_invalid") from exc
        object.__setattr__(self, "interval", canonical)
        start = _strict_int(self.start_open_ts, field="start_open_ts")
        end = _strict_int(self.end_open_ts_exclusive, field="end_open_ts_exclusive")
        as_of_us = _strict_int(
            self.collection_as_of_us, field="collection_as_of_us", minimum=0
        )
        page_size = _strict_int(self.page_size, field="page_size", minimum=1)
        if page_size > _MAX_MEXC_PAGE_SIZE:
            raise HistoryRangeContractError("history_page_size_is_out_of_range")
        if not isinstance(self.endpoint_contract, MexcFuturesEndpointContractV1):
            raise HistoryRangeContractError("history_endpoint_contract_is_invalid")
        if not isinstance(self.resource_limits, HistoryResourceLimitsV1):
            raise HistoryRangeContractError("history_resource_limits_are_invalid")
        if not isinstance(self.retry_policy, HistoryRetryPolicyV1):
            raise HistoryRangeContractError("history_retry_policy_is_invalid")
        profile = self.storage_profile or _durability_profile_for_host()
        _storage_contract_payload(profile)
        object.__setattr__(self, "storage_profile", profile)
        limits = self.resource_limits.as_dict()
        for field, hard_limit in _HARD_LIMITS.items():
            value = limits.get(field)
            if type(value) is not int or value < 1 or value > hard_limit:
                raise HistoryRangeContractError(f"history_{field}_is_out_of_range")
        step = interval_seconds(canonical)
        if start >= end:
            raise HistoryRangeContractError("history_range_must_be_nonempty")
        if start % step or end % step:
            raise HistoryRangeContractError("history_range_is_not_utc_aligned")
        closed_boundary = (
            as_of_us // (step * 1_000_000)
        ) * step
        if end > closed_boundary:
            raise HistoryRangeContractError("history_range_contains_unclosed_bar")
        if self.expected_row_count > self.resource_limits.max_rows:
            raise HistoryBudgetExceededError(
                "rows", self.resource_limits.max_rows, self.expected_row_count
            )
        if self.required_pages > self.resource_limits.max_pages:
            raise HistoryBudgetExceededError(
                "pages", self.resource_limits.max_pages, self.required_pages
            )
        possible_attempts = (
            self.required_pages * self.resource_limits.max_attempts_per_page
        )
        if possible_attempts > self.resource_limits.max_total_attempts:
            raise HistoryBudgetExceededError(
                "attempts",
                self.resource_limits.max_total_attempts,
                possible_attempts,
            )

    @property
    def interval_seconds(self) -> int:
        return interval_seconds(self.interval)

    @property
    def expected_row_count(self) -> int:
        return (
            self.end_open_ts_exclusive - self.start_open_ts
        ) // self.interval_seconds

    @property
    def required_pages(self) -> int:
        return (self.expected_row_count + self.page_size - 1) // self.page_size

    @property
    def endpoint_identity(self) -> str:
        return self.endpoint_contract.endpoint_identity

    @property
    def request_id(self) -> str:
        return _sha256_payload(self.as_dict())

    @property
    def attempt_contract_hash(self) -> str:
        return mexc_futures_transport_contract_hash()

    @property
    def storage_profile_hash(self) -> str:
        return storage_profile_hash(self.storage_profile)

    def expected_timestamps(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.start_open_ts,
                self.end_open_ts_exclusive,
                self.interval_seconds,
            )
        )

    def contract_identities(self) -> dict[str, object]:
        return {
            "endpoint_identity": self.endpoint_contract.endpoint_identity,
            "endpoint_contract_version": self.endpoint_contract.contract_version,
            "endpoint_contract_hash": self.endpoint_contract.contract_hash,
            "resource_limits_version": self.resource_limits.contract_version,
            "resource_limits_hash": self.resource_limits.contract_hash,
            "retry_policy_version": self.retry_policy.contract_version,
            "retry_policy_hash": self.retry_policy.contract_hash,
            "attempt_contract_hash": self.attempt_contract_hash,
            "storage_contract_version": STRICT_HISTORY_V2_STORAGE_VERSION,
            "storage_profile": self.storage_profile,
            "storage_profile_hash": self.storage_profile_hash,
        }

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
            "collection_as_of_us": self.collection_as_of_us,
            "page_size": self.page_size,
            "endpoint_contract": self.endpoint_contract.as_dict(),
            "resource_limits": self.resource_limits.as_dict(),
            "retry_policy": self.retry_policy.as_dict(),
            "contract_identities": self.contract_identities(),
        }


def _plan_pages(request: HistoryRangeRequestV2) -> tuple[KlinePageRequestV1, ...]:
    pages: list[KlinePageRequestV1] = []
    cursor = request.start_open_ts
    for ordinal in range(request.required_pages):
        remaining = (
            request.end_open_ts_exclusive - cursor
        ) // request.interval_seconds
        count = min(request.page_size, remaining)
        end = cursor + (count - 1) * request.interval_seconds
        pages.append(
            KlinePageRequestV1(
                range_request_id=request.request_id,
                endpoint_identity=request.endpoint_identity,
                venue_symbol=request.venue_symbol,
                interval=request.interval,
                page_ordinal=ordinal,
                start_open_ts=cursor,
                end_open_ts_inclusive=end,
                expected_row_count=count,
            )
        )
        cursor = end + request.interval_seconds
    if cursor != request.end_open_ts_exclusive:
        raise HistoryRangeContractError("history_v2_page_plan_does_not_cover_range")
    return tuple(pages)


@dataclass(frozen=True)
class HistoryPageReceiptV2:
    page_request: KlinePageRequestV1
    attempt_receipt_hashes: tuple[str, ...]
    request_started_at_us: int
    received_at_us: int
    raw_body_sha256: str
    raw_body_length: int
    row_count: int
    first_bar_open_ts: int
    last_bar_open_ts: int
    normalized_page_hash: str
    http_status: int = 200
    api_code: int = 0
    contract_version: str = STRICT_HISTORY_V2_PAGE_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_V2_PAGE_RECEIPT_VERSION:
            raise HistoryPayloadSchemaError("history_v2_page_receipt_version_mismatch")
        if not isinstance(self.page_request, KlinePageRequestV1):
            raise HistoryPayloadValueError("history_v2_page_request_is_invalid")
        if not isinstance(self.attempt_receipt_hashes, tuple) or not self.attempt_receipt_hashes:
            raise HistoryPayloadValueError("history_v2_page_attempts_are_missing")
        for digest in (
            *self.attempt_receipt_hashes,
            self.raw_body_sha256,
            self.normalized_page_hash,
        ):
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise HistoryPayloadValueError("history_v2_page_hash_is_invalid")
        start = _strict_int(
            self.request_started_at_us,
            field="page_request_started_at_us",
            minimum=0,
        )
        end = _strict_int(self.received_at_us, field="page_received_at_us", minimum=0)
        if end < start:
            raise HistoryPayloadValueError("history_v2_page_timing_is_invalid")
        if type(self.first_bar_open_ts) is not int or type(self.last_bar_open_ts) is not int:
            raise HistoryPayloadValueError("history_v2_page_range_type_is_invalid")
        source_close_us = (
            self.last_bar_open_ts + interval_seconds(self.page_request.interval)
        ) * 1_000_000
        if start < source_close_us:
            raise HistoryPayloadValueError(
                "history_v2_page_request_started_before_source_close"
            )
        if type(self.http_status) is not int or self.http_status != 200:
            raise HistoryPayloadValueError("history_v2_page_http_status_is_invalid")
        if type(self.api_code) is not int or self.api_code != 0:
            raise HistoryPayloadValueError("history_v2_page_api_code_is_invalid")
        if type(self.raw_body_length) is not int or self.raw_body_length < 0:
            raise HistoryPayloadValueError("history_v2_page_body_length_is_invalid")
        if type(self.row_count) is not int or self.row_count != self.page_request.expected_row_count:
            raise HistoryPayloadValueError("history_v2_page_row_count_mismatch")
        if (
            self.first_bar_open_ts != self.page_request.start_open_ts
            or self.last_bar_open_ts != self.page_request.end_open_ts_inclusive
        ):
            raise HistoryPayloadValueError("history_v2_page_range_mismatch")

    @property
    def page_receipt_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "page_request": self.page_request.as_dict(),
            "page_id": self.page_request.page_id,
            "attempt_receipt_hashes": list(self.attempt_receipt_hashes),
            "request_started_at_us": self.request_started_at_us,
            "received_at_us": self.received_at_us,
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
class HistoryCollectionManifestV2:
    request: HistoryRangeRequestV2
    page_receipts: tuple[HistoryPageReceiptV2, ...]
    normalized_logical_hash: str
    normalized_shard_sha256: str
    expected_row_count: int
    actual_row_count: int
    first_bar_open_ts: int
    last_bar_open_ts: int
    completed_at_us: int
    actual_attempt_count: int
    actual_total_raw_body_bytes: int
    logical_storage_bytes: int
    collection_runtime_us: int
    contract_version: str = STRICT_HISTORY_V2_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_V2_MANIFEST_VERSION:
            raise HistoryPayloadSchemaError("history_v2_manifest_version_mismatch")
        if not isinstance(self.request, HistoryRangeRequestV2):
            raise HistoryPayloadValueError("history_v2_manifest_request_is_invalid")
        if not isinstance(self.page_receipts, tuple) or not all(
            isinstance(item, HistoryPageReceiptV2) for item in self.page_receipts
        ):
            raise HistoryPayloadValueError("history_v2_manifest_pages_are_not_immutable")
        if len(self.page_receipts) != self.request.required_pages:
            raise HistoryPayloadValueError("history_v2_manifest_page_count_mismatch")
        if tuple(item.page_request for item in self.page_receipts) != _plan_pages(self.request):
            raise HistoryPayloadValueError("history_v2_manifest_page_plan_mismatch")
        for item in self.page_receipts:
            if item.request_started_at_us < self.request.collection_as_of_us:
                raise HistoryPayloadValueError(
                    "history_v2_manifest_page_started_before_collection_as_of"
                )
            if (
                len(item.attempt_receipt_hashes)
                > self.request.resource_limits.max_attempts_per_page
            ):
                raise HistoryBudgetExceededError(
                    "attempts_per_page",
                    self.request.resource_limits.max_attempts_per_page,
                    len(item.attempt_receipt_hashes),
                )
        for digest in (self.normalized_logical_hash, self.normalized_shard_sha256):
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise HistoryPayloadValueError("history_v2_manifest_hash_is_invalid")
        if type(self.expected_row_count) is not int or type(self.actual_row_count) is not int:
            raise HistoryPayloadValueError("history_v2_manifest_row_count_type_is_invalid")
        if (
            self.expected_row_count != self.request.expected_row_count
            or self.actual_row_count != self.expected_row_count
        ):
            raise HistoryPayloadValueError("history_v2_manifest_row_count_mismatch")
        expected = self.request.expected_timestamps()
        if type(self.first_bar_open_ts) is not int or type(self.last_bar_open_ts) is not int:
            raise HistoryPayloadValueError("history_v2_manifest_range_type_is_invalid")
        if self.first_bar_open_ts != expected[0] or self.last_bar_open_ts != expected[-1]:
            raise HistoryPayloadValueError("history_v2_manifest_range_mismatch")
        completed = _strict_int(self.completed_at_us, field="completed_at_us", minimum=0)
        if completed != max(item.received_at_us for item in self.page_receipts):
            raise HistoryPayloadValueError("history_v2_manifest_completion_time_mismatch")
        if type(self.actual_attempt_count) is not int:
            raise HistoryPayloadValueError(
                "history_v2_manifest_attempt_count_type_is_invalid"
            )
        attempts = sum(len(item.attempt_receipt_hashes) for item in self.page_receipts)
        if self.actual_attempt_count != attempts:
            raise HistoryPayloadValueError("history_v2_manifest_attempt_count_mismatch")
        for field in (
            "actual_total_raw_body_bytes",
            "logical_storage_bytes",
            "collection_runtime_us",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise HistoryPayloadValueError(f"history_v2_manifest_{field}_is_invalid")
        limits = self.request.resource_limits
        if self.actual_attempt_count > limits.max_total_attempts:
            raise HistoryBudgetExceededError(
                "attempts", limits.max_total_attempts, self.actual_attempt_count
            )
        if self.actual_total_raw_body_bytes > limits.max_total_raw_body_bytes:
            raise HistoryBudgetExceededError(
                "total_raw_body_bytes",
                limits.max_total_raw_body_bytes,
                self.actual_total_raw_body_bytes,
            )
        if self.logical_storage_bytes > limits.max_logical_storage_bytes:
            raise HistoryBudgetExceededError(
                "logical_storage_bytes",
                limits.max_logical_storage_bytes,
                self.logical_storage_bytes,
            )
        if self.collection_runtime_us > limits.max_collection_runtime_us:
            raise HistoryBudgetExceededError(
                "collection_runtime_us",
                limits.max_collection_runtime_us,
                self.collection_runtime_us,
            )

    @property
    def manifest_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "history_contract_version": STRICT_HISTORY_V2_CONTRACT_VERSION,
            "history_contract_hash": strict_history_v2_contract_hash(),
            "request": self.request.as_dict(),
            "request_id": self.request.request_id,
            "contract_identities": self.request.contract_identities(),
            "page_receipts": [item.as_dict() for item in self.page_receipts],
            "page_receipt_hashes": [item.page_receipt_hash for item in self.page_receipts],
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
            "completed_at_us": self.completed_at_us,
            "actual_attempt_count": self.actual_attempt_count,
            "actual_total_raw_body_bytes": self.actual_total_raw_body_bytes,
            "logical_storage_bytes": self.logical_storage_bytes,
            "collection_runtime_us": self.collection_runtime_us,
            "storage_semantics": _storage_contract_payload(self.request.storage_profile),
        }


@dataclass(frozen=True)
class CompleteHistoryShardV2:
    rows: tuple[NormalizedHistoryRowV1, ...]
    manifest: HistoryCollectionManifestV2

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple) or not all(
            isinstance(item, NormalizedHistoryRowV1) for item in self.rows
        ):
            raise HistoryPayloadValueError("history_v2_rows_are_not_immutable")
        if not isinstance(self.manifest, HistoryCollectionManifestV2):
            raise HistoryPayloadValueError("history_v2_manifest_is_invalid")
        request = self.manifest.request
        if tuple(row.bar_open_ts for row in self.rows) != request.expected_timestamps():
            raise HistoryPayloadValueError("history_v2_rows_do_not_match_grid")
        receipts = {item.page_receipt_hash: item for item in self.manifest.page_receipts}
        rows_by_receipt: dict[str, list[NormalizedHistoryRowV1]] = {
            key: [] for key in receipts
        }
        for row in self.rows:
            if (
                row.venue != request.venue
                or row.symbol != request.symbol
                or row.venue_symbol != request.venue_symbol
                or row.interval != request.interval
            ):
                raise HistoryPayloadValueError("history_v2_row_identity_mismatch")
            receipt = receipts.get(row.source_page_receipt_hash)
            if receipt is None or row.source_raw_body_sha256 != receipt.raw_body_sha256:
                raise HistoryPayloadValueError("history_v2_row_source_mismatch")
            if row.bar_open_ts not in receipt.page_request.expected_timestamps():
                raise HistoryPayloadValueError("history_v2_row_page_range_mismatch")
            rows_by_receipt[row.source_page_receipt_hash].append(row)
        for digest, page_rows in rows_by_receipt.items():
            receipt = receipts[digest]
            if len(page_rows) != receipt.row_count:
                raise HistoryPayloadValueError("history_v2_page_row_count_mismatch")
            ordered_page_rows = sorted(page_rows, key=lambda row: row.bar_open_ts)
            if tuple(item.source_row_ordinal for item in ordered_page_rows) != tuple(
                range(receipt.row_count)
            ):
                raise HistoryPayloadValueError("history_v2_source_ordinals_mismatch")
            page_hash = _sha256_payload(
                [item.market_dict() for item in ordered_page_rows]
            )
            if page_hash != receipt.normalized_page_hash:
                raise HistoryPayloadValueError("history_v2_normalized_page_hash_mismatch")
        logical = _sha256_payload([item.market_dict() for item in self.rows])
        if logical != self.manifest.normalized_logical_hash:
            raise HistoryPayloadValueError("history_v2_logical_hash_mismatch")
        if _sha256_bytes(self.normalized_jsonl_bytes()) != self.manifest.normalized_shard_sha256:
            raise HistoryPayloadValueError("history_v2_shard_hash_mismatch")

    def normalized_jsonl_bytes(self) -> bytes:
        return b"".join(_canonical_bytes(row.as_dict()) + b"\n" for row in self.rows)

    def to_frame(self) -> pd.DataFrame:
        index = pd.to_datetime([row.bar_open_ts for row in self.rows], unit="s", utc=True)
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
                "history_contract_version": STRICT_HISTORY_V2_CONTRACT_VERSION,
                "history_contract_hash": strict_history_v2_contract_hash(),
                "history_request_id": self.manifest.request.request_id,
                "history_manifest_hash": self.manifest.manifest_hash,
            }
        )
        return frame

    def to_min1_aggregation_inputs(self):
        if self.manifest.request.interval != "Min1":
            raise HistoryRangeContractError("history_v2_shard_is_not_min1")
        from trading.market_data.min1_aggregation import (
            Min1BarReceiptV1,
            normalized_min1_source_row_hash,
        )

        page_by_hash = {
            item.page_receipt_hash: item for item in self.manifest.page_receipts
        }
        if any(
            item.request_started_at_us
            < (
                item.last_bar_open_ts
                + interval_seconds(item.page_request.interval)
            )
            * 1_000_000
            for item in page_by_hash.values()
        ):
            raise HistoryPayloadValueError(
                "history_v2_min1_source_was_requested_before_bar_close"
            )
        frame = self.to_frame()
        receipts = tuple(
            Min1BarReceiptV1(
                bar_open_ts=float(row.bar_open_ts),
                request_started_at=page_by_hash[row.source_page_receipt_hash].request_started_at_us / 1_000_000,
                received_at=page_by_hash[row.source_page_receipt_hash].received_at_us / 1_000_000,
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
        return frame, receipts


@dataclass(frozen=True)
class _NormalizedPageV2:
    rows: tuple[NormalizedHistoryRowV1, ...]
    receipt: HistoryPageReceiptV2


def _parse_mexc_page_v2(
    attempt: CompleteHttpAttemptEvidenceV1,
    *,
    request: HistoryRangeRequestV2,
    attempt_hashes: tuple[str, ...],
) -> _NormalizedPageV2:
    if not attempt.body_complete or attempt.http_status != 200 or attempt.outcome != "complete":
        raise HistoryTransportError(None, "history_v2_parser_requires_complete_200_body")
    if attempt.page_request.range_request_id != request.request_id:
        raise HistoryPayloadRangeError("history_v2_page_belongs_to_another_range")
    try:
        payload = json.loads(
            attempt.body_bytes.decode("utf-8"),
            parse_float=Decimal,
            parse_int=int,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                HistoryJsonDecodeError("history_json_contains_nonfinite_constant")
            ),
            object_pairs_hook=lambda pairs: _payload_unique_object(pairs),
        )
    except HistoryJsonDecodeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HistoryJsonDecodeError() from exc
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise HistoryApiRejectedError()
    code = payload.get("code")
    if code == "0":
        code = 0
    if type(code) is not int or code != 0:
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
    expected = attempt.page_request.expected_timestamps()
    if row_count == 0:
        raise HistoryIncompleteRangeError("empty_success", missing_timestamps=expected)
    if row_count > request.resource_limits.max_rows:
        raise HistoryBudgetExceededError("rows", request.resource_limits.max_rows, row_count)
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
    missing = tuple(sorted(set(expected) - seen))
    unexpected = tuple(sorted(seen - set(expected)))
    if unexpected:
        raise HistoryIncompleteRangeError(
            "unexpected_timestamps",
            missing_timestamps=missing,
            unexpected_timestamps=unexpected,
        )
    if missing:
        raise HistoryIncompleteRangeError("missing_timestamps", missing_timestamps=missing)
    actual_order = tuple(item[0] for item in raw_rows)
    if actual_order != expected:
        raise HistoryPayloadRangeError(
            "history_v2_timestamps_are_not_in_expected_order"
        )
    market_rows = [
        {
            "contract_version": row_contract_version(),
            "venue": request.venue,
            "symbol": request.symbol,
            "venue_symbol": request.venue_symbol,
            "interval": request.interval,
            "bar_open_ts": timestamp,
            "bar_close_ts": timestamp + request.interval_seconds,
            **values,
        }
        for timestamp, _ordinal, values in raw_rows
    ]
    normalized_page_hash = _sha256_payload(market_rows)
    receipt = HistoryPageReceiptV2(
        page_request=attempt.page_request,
        attempt_receipt_hashes=attempt_hashes,
        request_started_at_us=attempt.request_started_at_us,
        received_at_us=attempt.terminal_at_us,
        raw_body_sha256=attempt.captured_body_sha256,
        raw_body_length=attempt.captured_body_length,
        row_count=row_count,
        first_bar_open_ts=expected[0],
        last_bar_open_ts=expected[-1],
        normalized_page_hash=normalized_page_hash,
    )
    rows = tuple(
        NormalizedHistoryRowV1(
            venue=request.venue,
            symbol=request.symbol,
            venue_symbol=request.venue_symbol,
            interval=request.interval,
            bar_open_ts=timestamp,
            bar_close_ts=timestamp + request.interval_seconds,
            source_page_receipt_hash=receipt.page_receipt_hash,
            source_raw_body_sha256=attempt.captured_body_sha256,
            source_row_ordinal=ordinal,
            **values,
        )
        for timestamp, ordinal, values in raw_rows
    )
    return _NormalizedPageV2(rows=rows, receipt=receipt)


def _payload_unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HistoryJsonDecodeError("history_json_contains_duplicate_key")
        result[key] = value
    return result


def row_contract_version() -> str:
    # Avoid duplicating the frozen literal in the v2 hash schema.
    return NormalizedHistoryRowV1.__dataclass_fields__["contract_version"].default


@dataclass(frozen=True)
class _LoadedGraph:
    shard: CompleteHistoryShardV2
    attempt_hashes: tuple[str, ...]
    raw_hashes: tuple[str, ...]


@dataclass(frozen=True)
class HistoryRestartRequestStateV1:
    request_id: str
    state: str
    manifest_hash: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.request_id):
            raise HistoryRangeContractError("restart_request_id_is_invalid")
        if self.state not in {
            "absent",
            "incomplete",
            "complete_verified",
            "corrupt",
            "ambiguous_fork",
        }:
            raise HistoryRangeContractError("restart_state_is_invalid")


@dataclass(frozen=True)
class HistoryRestartReportV1:
    request_states: tuple[HistoryRestartRequestStateV1, ...]
    temp_paths: tuple[str, ...]
    unreferenced_attempt_paths: tuple[str, ...]
    unreferenced_raw_paths: tuple[str, ...]
    alternate_normalized_paths: tuple[str, ...]
    ready: bool
    contract_version: str = STRICT_HISTORY_V2_RESTART_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_V2_RESTART_VERSION:
            raise HistoryRangeContractError("restart_report_version_mismatch")
        for field in (
            "request_states",
            "temp_paths",
            "unreferenced_attempt_paths",
            "unreferenced_raw_paths",
            "alternate_normalized_paths",
        ):
            if not isinstance(getattr(self, field), tuple):
                raise HistoryRangeContractError("restart_report_is_not_immutable")
        expected_ready = bool(self.request_states) and all(
            item.state == "complete_verified" for item in self.request_states
        ) and not any(
            (
                self.unreferenced_attempt_paths,
                self.unreferenced_raw_paths,
                self.alternate_normalized_paths,
            )
        )
        if self.ready is not expected_ready:
            raise HistoryRangeContractError("restart_report_ready_mismatch")


class StrictHistoryArtifactStoreV2:
    """Immutable one-range/shard store with writable and read-only modes."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        writable: bool = False,
        storage_profile: str | None = None,
    ):
        supplied_root = Path(os.path.abspath(os.fspath(root)))
        # ``Path.resolve`` follows links.  Inspect the caller-supplied chain
        # first so a junction/symlink alias cannot be laundered into an
        # apparently ordinary resolved directory.
        for candidate in (*reversed(supplied_root.parents), supplied_root):
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise HistoryStorageError(
                    "history_v2_supplied_root_chain_stat_failed"
                ) from exc
            attrs = getattr(info, "st_file_attributes", 0)
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if stat.S_ISLNK(info.st_mode) or attrs & reparse:
                raise HistoryArtifactCorruptionError(
                    "history_v2_supplied_root_reparse_point_is_forbidden"
                )
        self.root = supplied_root.resolve()
        lowered = [part.lower() for part in self.root.parts]
        if any(
            lowered[index : index + 2] == ["data", "history"]
            for index in range(max(0, len(lowered) - 1))
        ):
            raise HistoryStorageError("legacy_history_root_is_forbidden")
        self.writable = bool(writable)
        self.storage_profile = storage_profile or _durability_profile_for_host()
        self._writer_request_id: str | None = None
        self._writer_session_request_id: str | None = None
        self._writer_session_owner_thread_id: int | None = None
        _storage_contract_payload(self.storage_profile)
        if self.writable:
            self.root.mkdir(parents=True, exist_ok=True)
            self._validate_platform_profile()
        elif not self.root.is_dir():
            raise HistoryStorageError("history_v2_store_does_not_exist")
        self._reject_reparse(self.root, allow_directory=True)

    def _validate_platform_profile(self) -> None:
        if os.name == "nt":
            if self.storage_profile != WINDOWS_NTFS_DURABILITY_PROFILE_V1:
                raise HistoryStorageError("history_v2_windows_storage_profile_mismatch")
            anchor = self.root.anchor
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(str(anchor))
            if drive_type != 3:  # DRIVE_FIXED
                raise HistoryStorageError("history_v2_storage_must_be_local_fixed_disk")
            fs_name = ctypes.create_unicode_buffer(64)
            ok = ctypes.windll.kernel32.GetVolumeInformationW(
                str(anchor), None, 0, None, None, None, fs_name, len(fs_name)
            )
            if not ok or fs_name.value.upper() != "NTFS":
                raise HistoryStorageError("history_v2_windows_storage_must_be_ntfs")
        elif self.storage_profile != POSIX_DURABILITY_PROFILE_V1:
            raise HistoryStorageError("history_v2_posix_storage_profile_mismatch")

    @staticmethod
    def _reject_reparse(path: Path, *, allow_directory: bool = False) -> os.stat_result:
        try:
            info = path.lstat()
        except OSError as exc:
            raise HistoryStorageError("history_v2_artifact_stat_failed") from exc
        if stat.S_ISLNK(info.st_mode):
            raise HistoryArtifactCorruptionError("history_v2_symlink_is_forbidden")
        attrs = getattr(info, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if attrs & reparse:
            raise HistoryArtifactCorruptionError("history_v2_reparse_point_is_forbidden")
        if allow_directory:
            if not stat.S_ISDIR(info.st_mode):
                raise HistoryStorageError("history_v2_store_is_not_directory")
        elif not stat.S_ISREG(info.st_mode):
            raise HistoryArtifactCorruptionError("history_v2_artifact_is_not_regular_file")
        return info

    def _require_writable(self) -> None:
        if not self.writable:
            raise HistoryStorageError("history_v2_store_is_read_only")

    def _require_active_writer(self, request_id: str) -> None:
        self._require_writable()
        if self._writer_session_request_id != request_id:
            raise HistoryStorageError("history_v2_writer_session_is_required")
        if self._writer_session_owner_thread_id != threading.get_ident():
            raise HistoryStorageError(
                "history_v2_writer_session_owner_mismatch"
            )

    @staticmethod
    def _scope_marker_bytes(request: HistoryRangeRequestV2) -> bytes:
        return _canonical_bytes(
            {
                "contract_version": STRICT_HISTORY_V2_SCOPE_VERSION,
                "history_contract_version": STRICT_HISTORY_V2_CONTRACT_VERSION,
                "history_contract_hash": strict_history_v2_contract_hash(),
                "request_id": request.request_id,
                "request": request.as_dict(),
            }
        ) + b"\n"

    def _scope_binding_exists(
        self,
        request: HistoryRangeRequestV2,
        *,
        allow_absent: bool,
    ) -> bool:
        expected = self._scope_marker_bytes(request)
        try:
            actual = self._read_limited(
                self.root / "scope.json",
                limit=_MAX_SCOPE_MARKER_BYTES,
                missing_code="history_v2_store_scope_marker_is_missing",
            )
        except HistoryRestartIncompleteError:
            if allow_absent:
                return False
            raise
        if actual != expected:
            raise HistoryArtifactForkError(
                "history_v2_store_scope_request_mismatch"
            )
        return True

    @property
    def _writer_lock_path(self) -> Path:
        return self.root.parent / f".{self.root.name}.strict-history-v2.writer.lock"

    @contextmanager
    def writer_session(
        self,
        request: HistoryRangeRequestV2,
        *,
        clock: EvidenceClock | None = None,
    ) -> Iterator[None]:
        """Hold the one-shard writer lock through pristine check and admission."""

        self._require_writable()
        if not isinstance(request, HistoryRangeRequestV2):
            raise HistoryRangeContractError("history_v2_writer_request_is_invalid")
        scope_payload = self._scope_marker_bytes(request)
        scope_limit = min(
            _MAX_SCOPE_MARKER_BYTES,
            request.resource_limits.max_logical_storage_bytes,
        )
        if len(scope_payload) > scope_limit:
            raise HistoryBudgetExceededError(
                "scope_marker_bytes", scope_limit, len(scope_payload)
            )
        if self._writer_session_request_id is not None:
            raise HistoryArtifactForkError(
                "history_v2_writer_session_is_already_active"
            )
        lock_path = self._writer_lock_path
        lock_key = os.path.normcase(os.path.abspath(os.fspath(lock_path)))
        with _ACTIVE_WRITER_LOCKS_GUARD:
            if lock_key in _ACTIVE_WRITER_LOCK_PATHS:
                raise HistoryArtifactForkError(
                    "history_v2_writer_session_is_already_active"
                )
            _ACTIVE_WRITER_LOCK_PATHS.add(lock_key)

        handle = None
        acquired = False
        try:
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    lock_info = lock_path.lstat()
                except FileNotFoundError:
                    lock_info = None
                if lock_info is not None:
                    self._reject_reparse(lock_path)
                handle = lock_path.open("a+b")
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise HistoryArtifactCorruptionError(
                        "history_v2_writer_lock_is_not_regular_file"
                    )
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                    os.fsync(handle.fileno())
                handle.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(
                            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                except OSError as exc:
                    if (
                        exc.errno in {errno.EACCES, errno.EAGAIN}
                        or getattr(exc, "winerror", None) in {33, 36}
                    ):
                        raise HistoryArtifactForkError(
                            "history_v2_writer_session_is_already_active"
                        ) from exc
                    raise HistoryStorageError(
                        "history_v2_writer_lock_acquisition_failed"
                    ) from exc
            except HistoryStorageError:
                raise
            except OSError as exc:
                raise HistoryStorageError(
                    "history_v2_writer_lock_acquisition_failed"
                ) from exc
            acquired = True
            scope_exists = self._scope_binding_exists(request, allow_absent=True)
            self._bind_writer_request_scope(request)
            initial = self.reconcile_restart((request,), clock=clock)
            if (
                len(initial.request_states) != 1
                or initial.request_states[0].state != "absent"
                or any(
                    (
                        initial.temp_paths,
                        initial.unreferenced_attempt_paths,
                        initial.unreferenced_raw_paths,
                        initial.alternate_normalized_paths,
                    )
                )
            ):
                raise HistoryArtifactForkError(
                    "history_v2_collection_requires_pristine_request_namespace"
                )
            if not scope_exists:
                self._publish_immutable(
                    Path("scope.json"), scope_payload
                )
                self._scope_binding_exists(request, allow_absent=False)
            self._writer_session_request_id = request.request_id
            self._writer_session_owner_thread_id = threading.get_ident()
            yield
        finally:
            self._writer_session_owner_thread_id = None
            self._writer_session_request_id = None
            if acquired and handle is not None:
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    # Admission is intentionally non-revocable.  Closing the
                    # descriptor still asks the OS to release a surviving lock.
                    pass
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            with _ACTIVE_WRITER_LOCKS_GUARD:
                _ACTIVE_WRITER_LOCK_PATHS.discard(lock_key)

    def _bind_writer_request_scope(self, request: HistoryRangeRequestV2) -> None:
        """Latch this writable store instance to exactly one range/shard."""

        self._require_writable()
        if self._writer_request_id is None:
            self._writer_request_id = request.request_id
        elif self._writer_request_id != request.request_id:
            raise HistoryRangeContractError(
                "history_v2_store_is_single_request_shard_scope"
            )

    def _validate_parent_chain(
        self, directory: Path, *, allow_missing_suffix: bool = False
    ) -> None:
        try:
            relative = directory.relative_to(self.root)
        except ValueError as exc:
            raise HistoryStorageError("history_v2_artifact_escaped_store") from exc
        current = self.root
        self._reject_reparse(current, allow_directory=True)
        for part in relative.parts:
            current = current / part
            try:
                self._reject_reparse(current, allow_directory=True)
            except HistoryStorageError:
                if allow_missing_suffix:
                    try:
                        current.lstat()
                    except FileNotFoundError:
                        return
                raise

    def _sync_parent(self, path: Path) -> None:
        try:
            descriptor = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            if self.storage_profile == WINDOWS_NTFS_DURABILITY_PROFILE_V1:
                return
            raise HistoryStorageError("history_v2_directory_fsync_open_failed")
        failure: BaseException | None = None
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if self.storage_profile != WINDOWS_NTFS_DURABILITY_PROFILE_V1:
                failure = HistoryStorageError("history_v2_directory_fsync_failed")
                failure.__cause__ = exc
        except BaseException as exc:
            failure = exc
        try:
            os.close(descriptor)
        except OSError as exc:
            if failure is None:
                failure = HistoryStorageError("history_v2_directory_fsync_close_failed")
                failure.__cause__ = exc
        if failure is not None:
            raise failure

    def _publish_immutable(self, relative: Path, payload: bytes) -> Path:
        self._require_writable()
        if (
            not isinstance(relative, Path)
            or relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise HistoryStorageError("history_v2_artifact_path_is_invalid")
        target = self.root / relative
        self._validate_parent_chain(target.parent, allow_missing_suffix=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._validate_parent_chain(target.parent)
        try:
            target.lstat()
            target_exists = True
        except FileNotFoundError:
            target_exists = False
        except OSError as exc:
            raise HistoryStorageError("history_v2_artifact_stat_failed") from exc
        if target_exists:
            self._reject_reparse(target)
            try:
                existing = self._read_limited(
                    target,
                    limit=len(payload),
                    missing_code="history_v2_existing_artifact_disappeared",
                )
            except HistoryBudgetExceededError:
                raise HistoryArtifactConflictError()
            if existing != payload:
                raise HistoryArtifactConflictError()
            return target
        temporary = target.with_name(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                self._reject_reparse(target)
                try:
                    existing = self._read_limited(
                        target,
                        limit=len(payload),
                        missing_code="history_v2_existing_artifact_disappeared",
                    )
                except HistoryBudgetExceededError:
                    raise HistoryArtifactConflictError()
                if existing != payload:
                    raise HistoryArtifactConflictError()
            except OSError as exc:
                raise HistoryStorageError("history_v2_atomic_hardlink_failed") from exc
            # The inode was flushed through the temporary writable handle before
            # linking.  Windows FlushFileBuffers rejects a read-only descriptor;
            # opening the immutable final name for write merely to flush the same
            # inode would add no directory-entry guarantee, so do not pretend it
            # does.  POSIX accepts the extra read-handle barrier.
            if os.name != "nt":
                with target.open("rb") as final_handle:
                    os.fsync(final_handle.fileno())
            self._sync_parent(target)
        except HistoryStorageError:
            raise
        except OSError as exc:
            raise HistoryStorageError("history_v2_artifact_write_failed") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return target

    def persist_attempt(self, attempt: HttpAttemptEvidenceV1) -> int:
        if not isinstance(
            attempt,
            (CompleteHttpAttemptEvidenceV1, IncompleteHttpAttemptEvidenceV1),
        ):
            raise HistoryTransportError(None, "history_v2_attempt_type_is_invalid")
        self._require_active_writer(attempt.page_request.range_request_id)
        body = attempt.body_bytes
        if not isinstance(body, bytes):
            raise HistoryTransportError(None, "history_v2_attempt_body_is_not_bytes")
        if len(body) != attempt.captured_body_length or _sha256_bytes(body) != attempt.captured_body_sha256:
            raise HistoryTransportError(None, "history_v2_attempt_body_receipt_mismatch")
        receipt_bytes = _canonical_bytes(attempt.receipt_dict()) + b"\n"
        self._publish_immutable(
            Path("raw")
            / "sha256"
            / attempt.captured_body_sha256[:2]
            / f"{attempt.captured_body_sha256}.bin",
            body,
        )
        self._publish_immutable(
            Path("attempts") / f"{attempt.attempt_receipt_hash}.json",
            receipt_bytes,
        )
        return len(body) + len(receipt_bytes)

    @staticmethod
    def _manifest_bytes(manifest: HistoryCollectionManifestV2) -> bytes:
        return _canonical_bytes(
            {**manifest.as_dict(), "manifest_hash": manifest.manifest_hash}
        ) + b"\n"

    def publish_graph_candidate(self, shard: CompleteHistoryShardV2) -> None:
        """Publish a reloadable graph, but deliberately no success marker."""

        if not isinstance(shard, CompleteHistoryShardV2):
            raise HistoryPayloadValueError("history_v2_shard_is_invalid")
        self._require_active_writer(shard.manifest.request.request_id)
        self._verify_source_artifacts(shard.manifest.request, shard.manifest)
        request_id = shard.manifest.request.request_id
        self._publish_immutable(
            Path("normalized")
            / request_id
            / f"{shard.manifest.normalized_shard_sha256}.jsonl",
            shard.normalized_jsonl_bytes(),
        )
        self._publish_immutable(
            Path("collections") / request_id / "manifest.json",
            self._manifest_bytes(shard.manifest),
        )

    @staticmethod
    def _admission_bytes(
        *,
        request: HistoryRangeRequestV2,
        manifest_hash: str,
        admission_decision_runtime_us: int,
        graph_logical_storage_bytes: int,
    ) -> bytes:
        if not _SHA256_RE.fullmatch(manifest_hash):
            raise HistoryRangeContractError(
                "history_v2_admission_manifest_hash_is_invalid"
            )
        runtime = _strict_int(
            admission_decision_runtime_us,
            field="admission_decision_runtime_us",
            minimum=0,
        )
        graph_bytes = _strict_int(
            graph_logical_storage_bytes,
            field="graph_logical_storage_bytes",
            minimum=0,
        )
        if runtime > request.resource_limits.max_collection_runtime_us:
            raise HistoryBudgetExceededError(
                "collection_runtime_us",
                request.resource_limits.max_collection_runtime_us,
                runtime,
            )
        admitted_total = graph_bytes
        for _ in range(8):
            body = {
                "contract_version": STRICT_HISTORY_V2_ADMISSION_VERSION,
                "request_id": request.request_id,
                "manifest_hash": manifest_hash,
                "history_contract_hash": strict_history_v2_contract_hash(),
                "storage_profile_hash": request.storage_profile_hash,
                "admission_decision_runtime_us": runtime,
                "runtime_boundary": (
                    "after_full_disk_reload_before_atomic_admission_install"
                ),
                "graph_logical_storage_bytes": graph_bytes,
                "admitted_total_logical_storage_bytes": admitted_total,
            }
            payload = {**body, "admission_hash": _sha256_payload(body)}
            rendered = _canonical_bytes(payload) + b"\n"
            candidate_total = graph_bytes + len(rendered)
            if candidate_total == admitted_total:
                return rendered
            admitted_total = candidate_total
        raise HistoryStorageError("history_v2_admission_size_did_not_stabilize")

    def _publish_admission_marker(
        self,
        request: HistoryRangeRequestV2,
        *,
        manifest_hash: str,
        admission_decision_runtime_us: int,
        minimum_admission_runtime_us: int,
        graph_logical_storage_bytes: int,
    ) -> None:
        """Install the only positive success marker after graph reload."""

        self._require_active_writer(request.request_id)

        minimum_runtime = _strict_int(
            minimum_admission_runtime_us,
            field="minimum_admission_runtime_us",
            minimum=0,
        )
        decision_runtime = _strict_int(
            admission_decision_runtime_us,
            field="admission_decision_runtime_us",
            minimum=0,
        )
        if decision_runtime < minimum_runtime:
            raise HistoryRangeContractError(
                "history_v2_admission_runtime_precedes_manifest_runtime"
            )

        payload = self._admission_bytes(
            request=request,
            manifest_hash=manifest_hash,
            admission_decision_runtime_us=decision_runtime,
            graph_logical_storage_bytes=graph_logical_storage_bytes,
        )
        total = graph_logical_storage_bytes + len(payload)
        if total > request.resource_limits.max_logical_storage_bytes:
            raise HistoryBudgetExceededError(
                "logical_storage_bytes",
                request.resource_limits.max_logical_storage_bytes,
                total,
            )
        self._publish_immutable(
            Path("collections") / request.request_id / "admission.json",
            payload,
        )

    def admit_reloaded_graph(
        self,
        request: HistoryRangeRequestV2,
        *,
        expected_manifest_hash: str,
        collection_started_monotonic_us: int,
        clock: EvidenceClock,
    ) -> CompleteHistoryShardV2:
        """Reload the full graph, decide the deadline, then install admission."""

        self._require_active_writer(request.request_id)
        started = _strict_int(
            collection_started_monotonic_us,
            field="collection_started_monotonic_us",
            minimum=0,
        )
        graph = self._load_complete_graph(
            request,
            expected_manifest_hash=expected_manifest_hash,
            require_admission=False,
        )
        now = _clock_us(clock, "monotonic_us")
        runtime = now - started
        if runtime < 0:
            raise HistoryRangeContractError(
                "history_v2_collection_monotonic_clock_regressed"
            )
        if runtime > request.resource_limits.max_collection_runtime_us:
            raise HistoryBudgetExceededError(
                "collection_runtime_us",
                request.resource_limits.max_collection_runtime_us,
                runtime,
            )
        self._publish_admission_marker(
            request,
            manifest_hash=graph.shard.manifest.manifest_hash,
            admission_decision_runtime_us=runtime,
            minimum_admission_runtime_us=graph.shard.manifest.collection_runtime_us,
            graph_logical_storage_bytes=graph.shard.manifest.logical_storage_bytes,
        )
        return graph.shard

    def _verify_admission(
        self,
        request: HistoryRangeRequestV2,
        *,
        manifest_hash: str,
        minimum_admission_runtime_us: int,
        graph_logical_storage_bytes: int,
    ) -> None:
        path = self.root / "collections" / request.request_id / "admission.json"
        payload = _parse_canonical_json(
            self._read_limited(
                path,
                limit=16 * 1024,
                missing_code="history_v2_admission_marker_is_missing",
            ),
            code="history_v2_admission_marker_is_invalid",
        )
        _exact_keys(
            payload,
            frozenset(
                {
                    "contract_version",
                    "request_id",
                    "manifest_hash",
                    "history_contract_hash",
                    "storage_profile_hash",
                    "admission_decision_runtime_us",
                    "runtime_boundary",
                    "graph_logical_storage_bytes",
                    "admitted_total_logical_storage_bytes",
                    "admission_hash",
                }
            ),
            code="history_v2_admission_marker_schema_mismatch",
        )
        body = dict(payload)
        admission_hash = body.pop("admission_hash")
        if (
            payload["contract_version"] != STRICT_HISTORY_V2_ADMISSION_VERSION
            or payload["request_id"] != request.request_id
            or payload["manifest_hash"] != manifest_hash
            or payload["history_contract_hash"] != strict_history_v2_contract_hash()
            or payload["storage_profile_hash"] != request.storage_profile_hash
            or payload["runtime_boundary"]
            != "after_full_disk_reload_before_atomic_admission_install"
            or payload["graph_logical_storage_bytes"]
            != graph_logical_storage_bytes
            or not isinstance(admission_hash, str)
            or _sha256_payload(body) != admission_hash
        ):
            raise HistoryArtifactCorruptionError(
                "history_v2_admission_marker_identity_mismatch"
            )
        runtime = payload["admission_decision_runtime_us"]
        if type(runtime) is not int or runtime < 0:
            raise HistoryArtifactCorruptionError(
                "history_v2_admission_runtime_is_invalid"
            )
        if runtime > request.resource_limits.max_collection_runtime_us:
            raise HistoryArtifactCorruptionError(
                "history_v2_admission_runtime_exceeds_budget"
            )
        if runtime < minimum_admission_runtime_us:
            raise HistoryArtifactCorruptionError(
                "history_v2_admission_runtime_precedes_manifest_runtime"
            )
        total = graph_logical_storage_bytes + len(
            self._admission_bytes(
                request=request,
                manifest_hash=manifest_hash,
                admission_decision_runtime_us=runtime,
                graph_logical_storage_bytes=graph_logical_storage_bytes,
            )
        )
        if (
            payload["admitted_total_logical_storage_bytes"] != total
            or total > request.resource_limits.max_logical_storage_bytes
        ):
            raise HistoryArtifactCorruptionError(
                "history_v2_admission_storage_total_mismatch"
            )

    def _read_limited(self, path: Path, *, limit: int, missing_code: str) -> bytes:
        if type(limit) is not int or limit < 0:
            raise HistoryRangeContractError("history_v2_read_limit_is_invalid")
        self._validate_parent_chain(path.parent, allow_missing_suffix=True)
        pre_open: os.stat_result | None = None
        try:
            pre_open = self._reject_reparse(path)
        except HistoryStorageError:
            try:
                path.lstat()
            except FileNotFoundError as exc:
                raise HistoryRestartIncompleteError(missing_code) from exc
            raise
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(str(path), flags)
        except FileNotFoundError as exc:
            raise HistoryRestartIncompleteError(missing_code) from exc
        except OSError as exc:
            raise HistoryArtifactCorruptionError("history_v2_artifact_read_failed") from exc
        failure: BaseException | None = None
        payload: bytes | None = None
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise HistoryArtifactCorruptionError(
                    "history_v2_artifact_is_not_regular_file"
                )
            pre_identity = (
                pre_open.st_dev,
                pre_open.st_ino,
                pre_open.st_size,
                getattr(pre_open, "st_mtime_ns", None),
            )
            opened_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                getattr(before, "st_mtime_ns", None),
            )
            if pre_identity != opened_identity:
                raise HistoryArtifactCorruptionError(
                    "history_v2_artifact_changed_before_open"
                )
            if before.st_size > limit:
                raise HistoryBudgetExceededError(
                    "artifact_bytes", limit, before.st_size
                )
            chunks: list[bytes] = []
            remaining = limit + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > limit:
                raise HistoryBudgetExceededError(
                    "artifact_bytes", limit, len(payload)
                )
            after = os.fstat(descriptor)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                getattr(before, "st_mtime_ns", None),
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                getattr(after, "st_mtime_ns", None),
            )
            if identity_before != identity_after or len(payload) != after.st_size:
                raise HistoryArtifactCorruptionError(
                    "history_v2_artifact_changed_while_reading"
                )
        except OSError as exc:
            failure = HistoryArtifactCorruptionError(
                "history_v2_artifact_read_io_failed"
            )
            failure.__cause__ = exc
        except BaseException as exc:
            failure = exc
        try:
            os.close(descriptor)
        except OSError as exc:
            if failure is None:
                failure = HistoryArtifactCorruptionError(
                    "history_v2_artifact_read_close_failed"
                )
                failure.__cause__ = exc
        if failure is not None:
            raise failure
        if payload is None:
            raise HistoryArtifactCorruptionError(
                "history_v2_artifact_read_produced_no_payload"
            )
        return payload

    def _load_attempt(
        self,
        attempt_hash: str,
        *,
        page_request: KlinePageRequestV1,
        limits: HistoryResourceLimitsV1,
    ) -> tuple[HttpAttemptEvidenceV1, int]:
        receipt_path = self.root / "attempts" / f"{attempt_hash}.json"
        receipt_bytes = self._read_limited(
            receipt_path,
            limit=_MAX_ATTEMPT_RECEIPT_BYTES,
            missing_code="history_v2_attempt_is_missing",
        )
        payload = _parse_canonical_json(receipt_bytes, code="history_v2_attempt_is_invalid")
        raw_hash = payload.get("captured_body_sha256")
        raw_length = payload.get("captured_body_length")
        if not isinstance(raw_hash, str) or not _SHA256_RE.fullmatch(raw_hash):
            raise HistoryArtifactCorruptionError("history_v2_attempt_raw_hash_is_invalid")
        if type(raw_length) is not int or raw_length < 0:
            raise HistoryArtifactCorruptionError("history_v2_attempt_raw_length_is_invalid")
        if raw_length > limits.max_raw_body_bytes_per_attempt:
            raise HistoryBudgetExceededError(
                "raw_body_bytes_per_attempt",
                limits.max_raw_body_bytes_per_attempt,
                raw_length,
            )
        raw_path = self.root / "raw" / "sha256" / raw_hash[:2] / f"{raw_hash}.bin"
        raw = self._read_limited(
            raw_path,
            limit=limits.max_raw_body_bytes_per_attempt,
            missing_code="history_v2_raw_body_is_missing",
        )
        if len(raw) != raw_length or _sha256_bytes(raw) != raw_hash:
            raise HistoryArtifactCorruptionError("history_v2_raw_body_length_or_hash_mismatch")
        try:
            attempt = parse_http_attempt_evidence_v1(
                payload, page_request=page_request, body_bytes=raw
            )
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise HistoryArtifactCorruptionError("history_v2_attempt_reconstruction_failed") from exc
        if attempt.attempt_receipt_hash != attempt_hash:
            raise HistoryArtifactCorruptionError("history_v2_attempt_hash_mismatch")
        return attempt, len(receipt_bytes) + len(raw)

    def _verify_source_artifacts(
        self,
        request: HistoryRangeRequestV2,
        manifest: HistoryCollectionManifestV2,
    ) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
        attempt_hashes: list[str] = []
        raw_hashes: list[str] = []
        logical_bytes = 0
        raw_total = 0
        prior_terminal: int | None = None
        prior_terminal_monotonic: int | None = None
        prior_request_started: int | None = None
        prior_request_started_monotonic: int | None = None
        observed_inter_attempt_gap_us = 0
        for page_receipt in manifest.page_receipts:
            prior: HttpAttemptEvidenceV1 | None = None
            for ordinal, attempt_hash in enumerate(page_receipt.attempt_receipt_hashes):
                attempt, stored_bytes = self._load_attempt(
                    attempt_hash,
                    page_request=page_receipt.page_request,
                    limits=request.resource_limits,
                )
                if attempt.attempt_ordinal != ordinal:
                    raise HistoryArtifactCorruptionError("history_v2_attempt_ordinal_mismatch")
                if (
                    attempt.endpoint_contract_hash != request.endpoint_contract.contract_hash
                    or attempt.resource_limits_hash != request.resource_limits.contract_hash
                    or attempt.retry_policy_hash != request.retry_policy.contract_hash
                    or attempt.transport_contract_hash != request.attempt_contract_hash
                ):
                    raise HistoryArtifactCorruptionError("history_v2_attempt_contract_identity_mismatch")
                if attempt.request_started_at_us < request.collection_as_of_us:
                    raise HistoryArtifactCorruptionError("history_v2_attempt_started_before_as_of")
                if (
                    prior_terminal is not None
                    and attempt.request_started_at_us < prior_terminal
                ) or (
                    prior_terminal_monotonic is not None
                    and attempt.request_started_monotonic_us
                    < prior_terminal_monotonic
                ):
                    raise HistoryArtifactCorruptionError(
                        "history_v2_attempt_timing_regressed"
                    )
                if (
                    prior_request_started is not None
                    and (
                        attempt.scheduled_not_before_us
                        < prior_request_started
                        + request.retry_policy.min_request_spacing_us
                        or attempt.scheduled_not_before_monotonic_us
                        < prior_request_started_monotonic
                        + request.retry_policy.min_request_spacing_us
                    )
                ):
                    raise HistoryArtifactCorruptionError(
                        "history_v2_request_spacing_was_not_honoured"
                    )
                if prior_terminal_monotonic is not None:
                    observed_inter_attempt_gap_us += max(
                        0,
                        attempt.request_started_monotonic_us
                        - prior_terminal_monotonic,
                    )
                    if (
                        observed_inter_attempt_gap_us
                        > request.retry_policy.max_total_sleep_us
                    ):
                        raise HistoryBudgetExceededError(
                            "observed_inter_attempt_gap_us",
                            request.retry_policy.max_total_sleep_us,
                            observed_inter_attempt_gap_us,
                        )
                if attempt.elapsed_monotonic_us > request.resource_limits.max_attempt_runtime_us:
                    raise HistoryBudgetExceededError(
                        "attempt_runtime_us",
                        request.resource_limits.max_attempt_runtime_us,
                        attempt.elapsed_monotonic_us,
                    )
                if prior is not None:
                    if isinstance(prior, CompleteHttpAttemptEvidenceV1):
                        retryable = (
                            prior.http_status in {408, 425, 429}
                            or prior.http_status >= 500
                        )
                        retry_after = retry_after_delay_us(
                            prior.safe_headers,
                            received_at_us=prior.terminal_at_us,
                            policy=request.retry_policy,
                        )
                    else:
                        retryable = prior.outcome in {"network_error", "timeout"}
                        retry_after = 0
                    if not retryable:
                        raise HistoryArtifactCorruptionError(
                            "history_v2_nonfinal_attempt_is_not_retryable"
                        )
                    retry_delay = max(
                        request.retry_policy.backoff_before_attempt_us(ordinal),
                        retry_after,
                    )
                    if (
                        attempt.scheduled_not_before_us
                        < prior.terminal_at_us + retry_delay
                        or attempt.scheduled_not_before_monotonic_us
                        < prior.terminal_monotonic_us + retry_delay
                    ):
                        raise HistoryArtifactCorruptionError(
                            "history_v2_retry_schedule_was_not_honoured"
                        )
                prior = attempt
                prior_terminal = attempt.terminal_at_us
                prior_terminal_monotonic = attempt.terminal_monotonic_us
                prior_request_started = attempt.request_started_at_us
                prior_request_started_monotonic = attempt.request_started_monotonic_us
                attempt_hashes.append(attempt_hash)
                raw_hashes.append(attempt.captured_body_sha256)
                logical_bytes += stored_bytes
                raw_total += attempt.captured_body_length
                if raw_total > request.resource_limits.max_total_raw_body_bytes:
                    raise HistoryBudgetExceededError(
                        "total_raw_body_bytes",
                        request.resource_limits.max_total_raw_body_bytes,
                        raw_total,
                    )
                if logical_bytes > request.resource_limits.max_logical_storage_bytes:
                    raise HistoryBudgetExceededError(
                        "logical_storage_bytes",
                        request.resource_limits.max_logical_storage_bytes,
                        logical_bytes,
                    )
            if prior is None or not isinstance(prior, CompleteHttpAttemptEvidenceV1):
                raise HistoryArtifactCorruptionError("history_v2_page_has_no_complete_final_attempt")
            if (
                not prior.body_complete
                or prior.http_status != 200
                or prior.outcome != "complete"
                or prior.attempt_receipt_hash != page_receipt.attempt_receipt_hashes[-1]
                or prior.request_started_at_us != page_receipt.request_started_at_us
                or prior.terminal_at_us != page_receipt.received_at_us
                or prior.captured_body_sha256 != page_receipt.raw_body_sha256
                or prior.captured_body_length != page_receipt.raw_body_length
            ):
                raise HistoryArtifactCorruptionError("history_v2_final_attempt_page_mismatch")
        if raw_total != manifest.actual_total_raw_body_bytes:
            raise HistoryArtifactCorruptionError("history_v2_manifest_raw_total_mismatch")
        if len(attempt_hashes) != manifest.actual_attempt_count:
            raise HistoryArtifactCorruptionError("history_v2_manifest_attempt_total_mismatch")
        return tuple(attempt_hashes), tuple(raw_hashes), logical_bytes, raw_total

    @staticmethod
    def _page_receipt_from_payload(
        payload: object, page_request: KlinePageRequestV1
    ) -> HistoryPageReceiptV2:
        expected_keys = frozenset(
            {
                "contract_version",
                "page_request",
                "page_id",
                "attempt_receipt_hashes",
                "request_started_at_us",
                "received_at_us",
                "http_status",
                "api_code",
                "raw_body_sha256",
                "raw_body_length",
                "row_count",
                "first_bar_open_ts",
                "last_bar_open_ts",
                "normalized_page_hash",
            }
        )
        data = _exact_keys(payload, expected_keys, code="history_v2_page_receipt_schema_mismatch")
        if data["page_request"] != page_request.as_dict() or data["page_id"] != page_request.page_id:
            raise HistoryArtifactCorruptionError("history_v2_page_receipt_request_mismatch")
        attempts = data["attempt_receipt_hashes"]
        if not isinstance(attempts, list):
            raise HistoryArtifactCorruptionError("history_v2_page_attempts_are_invalid")
        try:
            receipt = HistoryPageReceiptV2(
                page_request=page_request,
                attempt_receipt_hashes=tuple(attempts),
                request_started_at_us=data["request_started_at_us"],
                received_at_us=data["received_at_us"],
                http_status=data["http_status"],
                api_code=data["api_code"],
                raw_body_sha256=data["raw_body_sha256"],
                raw_body_length=data["raw_body_length"],
                row_count=data["row_count"],
                first_bar_open_ts=data["first_bar_open_ts"],
                last_bar_open_ts=data["last_bar_open_ts"],
                normalized_page_hash=data["normalized_page_hash"],
                contract_version=data["contract_version"],
            )
        except Exception as exc:
            raise HistoryArtifactCorruptionError("history_v2_page_receipt_is_invalid") from exc
        if receipt.as_dict() != data:
            raise HistoryArtifactCorruptionError("history_v2_page_receipt_roundtrip_mismatch")
        return receipt

    @staticmethod
    def _row_from_payload(payload: object) -> NormalizedHistoryRowV1:
        expected_keys = frozenset(
            {
                "contract_version",
                "venue",
                "symbol",
                "venue_symbol",
                "interval",
                "bar_open_ts",
                "bar_close_ts",
                "open",
                "high",
                "low",
                "close",
                "volume_contracts",
                "turnover_quote",
                "logical_row_hash",
                "source_page_receipt_hash",
                "source_raw_body_sha256",
                "source_row_ordinal",
            }
        )
        data = _exact_keys(payload, expected_keys, code="history_v2_row_schema_mismatch")
        try:
            row = NormalizedHistoryRowV1(
                venue=data["venue"],
                symbol=data["symbol"],
                venue_symbol=data["venue_symbol"],
                interval=data["interval"],
                bar_open_ts=data["bar_open_ts"],
                bar_close_ts=data["bar_close_ts"],
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                volume_contracts=data["volume_contracts"],
                turnover_quote=data["turnover_quote"],
                source_page_receipt_hash=data["source_page_receipt_hash"],
                source_raw_body_sha256=data["source_raw_body_sha256"],
                source_row_ordinal=data["source_row_ordinal"],
                contract_version=data["contract_version"],
            )
        except Exception as exc:
            raise HistoryArtifactCorruptionError("history_v2_row_is_invalid") from exc
        if row.as_dict() != data:
            raise HistoryArtifactCorruptionError("history_v2_row_roundtrip_mismatch")
        return row

    def _load_complete_graph(
        self,
        expected_request: HistoryRangeRequestV2,
        *,
        expected_manifest_hash: str | None = None,
        require_admission: bool = True,
    ) -> _LoadedGraph:
        if expected_request.storage_profile != self.storage_profile:
            raise HistoryStorageError("history_v2_store_profile_request_mismatch")
        if type(require_admission) is not bool:
            raise HistoryRangeContractError(
                "history_v2_require_admission_must_be_boolean"
            )
        self._scope_binding_exists(expected_request, allow_absent=False)
        request_id = expected_request.request_id
        manifest_path = self.root / "collections" / request_id / "manifest.json"
        manifest_bytes = self._read_limited(
            manifest_path,
            limit=min(_MAX_MANIFEST_BYTES, expected_request.resource_limits.max_logical_storage_bytes),
            missing_code="history_v2_manifest_is_missing",
        )
        payload = _parse_canonical_json(manifest_bytes, code="history_v2_manifest_is_invalid")
        manifest_hash = payload.get("manifest_hash")
        if not isinstance(manifest_hash, str) or not _SHA256_RE.fullmatch(manifest_hash):
            raise HistoryArtifactCorruptionError("history_v2_manifest_hash_is_invalid")
        body = dict(payload)
        body.pop("manifest_hash")
        if _sha256_payload(body) != manifest_hash:
            raise HistoryArtifactCorruptionError("history_v2_manifest_hash_mismatch")
        if expected_manifest_hash is not None:
            if not _SHA256_RE.fullmatch(expected_manifest_hash):
                raise HistoryRangeContractError("expected_manifest_hash_is_invalid")
            if manifest_hash != expected_manifest_hash:
                raise HistoryArtifactForkError("history_v2_detached_manifest_hash_mismatch")
        if body.get("request") != expected_request.as_dict() or body.get("request_id") != request_id:
            raise HistoryArtifactForkError("history_v2_manifest_request_mismatch")
        pages_payload = body.get("page_receipts")
        if not isinstance(pages_payload, list):
            raise HistoryArtifactCorruptionError("history_v2_manifest_pages_are_invalid")
        planned = _plan_pages(expected_request)
        if len(pages_payload) != len(planned):
            raise HistoryArtifactCorruptionError("history_v2_manifest_page_count_mismatch")
        pages = tuple(
            self._page_receipt_from_payload(item, page)
            for item, page in zip(pages_payload, planned, strict=True)
        )
        try:
            manifest = HistoryCollectionManifestV2(
                request=expected_request,
                page_receipts=pages,
                normalized_logical_hash=body["normalized_logical_hash"],
                normalized_shard_sha256=body["normalized_shard_sha256"],
                expected_row_count=body["expected_row_count"],
                actual_row_count=body["actual_row_count"],
                first_bar_open_ts=body["first_bar_open_ts"],
                last_bar_open_ts=body["last_bar_open_ts"],
                completed_at_us=body["completed_at_us"],
                actual_attempt_count=body["actual_attempt_count"],
                actual_total_raw_body_bytes=body["actual_total_raw_body_bytes"],
                logical_storage_bytes=body["logical_storage_bytes"],
                collection_runtime_us=body["collection_runtime_us"],
                contract_version=body["contract_version"],
            )
        except (KeyError, TypeError, ValueError, OverflowError, StrictHistoryError) as exc:
            raise HistoryArtifactCorruptionError("history_v2_manifest_reconstruction_failed") from exc
        expected_body = manifest.as_dict()
        if body != expected_body:
            raise HistoryArtifactCorruptionError("history_v2_manifest_schema_or_identity_mismatch")
        if manifest.manifest_hash != manifest_hash:
            raise HistoryArtifactCorruptionError("history_v2_manifest_recomputed_hash_mismatch")
        normalized_path = (
            self.root
            / "normalized"
            / request_id
            / f"{manifest.normalized_shard_sha256}.jsonl"
        )
        normalized_bytes = self._read_limited(
            normalized_path,
            limit=expected_request.resource_limits.max_logical_storage_bytes,
            missing_code="history_v2_normalized_shard_is_missing",
        )
        if _sha256_bytes(normalized_bytes) != manifest.normalized_shard_sha256:
            raise HistoryArtifactCorruptionError("history_v2_normalized_shard_hash_mismatch")
        lines = normalized_bytes.split(b"\n")
        if not lines or lines[-1] != b"" or any(not line for line in lines[:-1]):
            raise HistoryArtifactCorruptionError("history_v2_normalized_shard_has_torn_tail")
        if len(lines) - 1 > expected_request.resource_limits.max_rows:
            raise HistoryBudgetExceededError(
                "rows", expected_request.resource_limits.max_rows, len(lines) - 1
            )
        rows = tuple(
            self._row_from_payload(
                _parse_canonical_json(line + b"\n", code="history_v2_row_json_is_invalid")
            )
            for line in lines[:-1]
        )
        try:
            shard = CompleteHistoryShardV2(rows=rows, manifest=manifest)
        except StrictHistoryError as exc:
            raise HistoryArtifactCorruptionError("history_v2_complete_shard_is_invalid") from exc
        attempt_hashes, raw_hashes, source_bytes, _raw_total = self._verify_source_artifacts(
            expected_request, manifest
        )
        actual_logical = (
            len(self._scope_marker_bytes(expected_request))
            + source_bytes
            + len(normalized_bytes)
            + len(manifest_bytes)
        )
        if actual_logical != manifest.logical_storage_bytes:
            raise HistoryArtifactCorruptionError("history_v2_logical_storage_total_mismatch")
        if require_admission:
            self._verify_admission(
                expected_request,
                manifest_hash=manifest.manifest_hash,
                minimum_admission_runtime_us=manifest.collection_runtime_us,
                graph_logical_storage_bytes=manifest.logical_storage_bytes,
            )
        return _LoadedGraph(shard, attempt_hashes, raw_hashes)

    def load_complete_from_disk(
        self,
        expected_request: HistoryRangeRequestV2,
        *,
        expected_manifest_hash: str | None = None,
    ) -> CompleteHistoryShardV2:
        return self._load_complete_graph(
            expected_request, expected_manifest_hash=expected_manifest_hash
        ).shard

    def reconcile_restart(
        self,
        expected_requests: Sequence[HistoryRangeRequestV2],
        *,
        expected_manifest_hashes: Mapping[str, str] | None = None,
        clock: EvidenceClock | None = None,
    ) -> HistoryRestartReportV1:
        if not expected_requests:
            raise HistoryRangeContractError("restart_expected_requests_are_empty")
        if len(expected_requests) != 1:
            raise HistoryRangeContractError(
                "restart_requires_exactly_one_request_shard"
            )
        if len({item.request_id for item in expected_requests}) != len(expected_requests):
            raise HistoryRangeContractError("restart_expected_requests_are_duplicate")
        expected_request_ids = {item.request_id for item in expected_requests}
        try:
            expected_hashes = dict(expected_manifest_hashes or {})
        except (TypeError, ValueError) as exc:
            raise HistoryRangeContractError(
                "restart_expected_manifest_hashes_are_invalid"
            ) from exc
        if any(
            not isinstance(key, str)
            or _SHA256_RE.fullmatch(key) is None
            or not isinstance(value, str)
            or _SHA256_RE.fullmatch(value) is None
            for key, value in expected_hashes.items()
        ):
            raise HistoryRangeContractError(
                "restart_expected_manifest_hashes_are_invalid"
            )
        if set(expected_hashes) - expected_request_ids:
            raise HistoryRangeContractError(
                "restart_expected_manifest_hash_has_unknown_request"
            )
        scan_limit = sum(
            item.resource_limits.max_total_attempts for item in expected_requests
        )
        scan_entry_limit = min(
            _MAX_RESTART_SCAN_ENTRIES,
            max(
                16,
                (scan_limit * 2)
                + sum(item.resource_limits.max_pages for item in expected_requests)
                + _MAX_TEMP_SCAN_ENTRIES,
            ),
        )
        scan_byte_limit = sum(
            item.resource_limits.max_logical_storage_bytes
            for item in expected_requests
        )
        scan_runtime_limit = sum(
            item.resource_limits.max_collection_runtime_us
            for item in expected_requests
        )
        started_scan = (
            _clock_us(clock, "monotonic_us")
            if clock is not None
            else time.monotonic_ns() // 1_000
        )
        scan_entries = 0
        scan_bytes = 0
        scanned_paths: set[Path] = set()

        def check_scan_runtime() -> None:
            now = (
                _clock_us(clock, "monotonic_us")
                if clock is not None
                else time.monotonic_ns() // 1_000
            )
            elapsed = now - started_scan
            if elapsed < 0:
                raise HistoryRangeContractError(
                    "history_v2_restart_monotonic_clock_regressed"
                )
            if elapsed > scan_runtime_limit:
                raise HistoryBudgetExceededError(
                    "restart_scan_runtime_us", scan_runtime_limit, elapsed
                )

        def bounded_tree_scan(
            namespace_root: Path,
            *,
            namespace_limit: int,
            resource: str,
            recursive: bool,
            include_file: object,
            reject_unmatched_files: bool = False,
            allow_unmatched_file: object | None = None,
            allow_directory: object | None = None,
        ) -> tuple[Path, ...]:
            nonlocal scan_entries, scan_bytes
            collected: list[Path] = []
            if not namespace_root.exists():
                return ()
            # Validate the namespace itself.  When scanning the store root,
            # validating ``root.parent`` would intentionally escape the store
            # and reject every otherwise valid reconciliation.
            self._validate_parent_chain(namespace_root)
            self._reject_reparse(namespace_root, allow_directory=True)
            directories = [namespace_root]
            check_scan_runtime()
            try:
                while directories:
                    directory = directories.pop()
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            path = Path(entry.path)
                            try:
                                info = entry.stat(follow_symlinks=False)
                            except OSError as exc:
                                raise HistoryStorageError(
                                    "history_v2_restart_entry_stat_failed"
                                ) from exc
                            attrs = getattr(info, "st_file_attributes", 0)
                            reparse = getattr(
                                stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
                            )
                            if stat.S_ISLNK(info.st_mode) or attrs & reparse:
                                raise HistoryArtifactCorruptionError(
                                    "history_v2_restart_reparse_point_is_forbidden"
                                )
                            if path not in scanned_paths:
                                if scan_entries >= scan_entry_limit:
                                    raise HistoryBudgetExceededError(
                                        "restart_scan_entries",
                                        scan_entry_limit,
                                        scan_entries + 1,
                                    )
                                scan_entries += 1
                                scanned_paths.add(path)
                                is_temp_alias = (
                                    path.name.startswith(".")
                                    and path.name.endswith(".tmp")
                                )
                                if stat.S_ISREG(info.st_mode) and not is_temp_alias:
                                    scan_bytes += info.st_size
                                    if scan_bytes > scan_byte_limit:
                                        raise HistoryBudgetExceededError(
                                            "restart_scan_bytes",
                                            scan_byte_limit,
                                            scan_bytes,
                                        )
                            if stat.S_ISDIR(info.st_mode):
                                self._reject_reparse(path, allow_directory=True)
                                if callable(allow_directory) and not allow_directory(path):
                                    raise HistoryArtifactCorruptionError(
                                        "history_v2_restart_unexpected_directory"
                                    )
                                if recursive:
                                    directories.append(path)
                                elif reject_unmatched_files and not (
                                    callable(allow_directory) and allow_directory(path)
                                ):
                                    raise HistoryArtifactCorruptionError(
                                        "history_v2_restart_unexpected_directory"
                                    )
                            elif stat.S_ISREG(info.st_mode):
                                included = include_file(path)  # type: ignore[operator]
                                if included:
                                    if len(collected) >= namespace_limit:
                                        raise HistoryBudgetExceededError(
                                            resource,
                                            namespace_limit,
                                            len(collected) + 1,
                                        )
                                    collected.append(path)
                                elif reject_unmatched_files and not (
                                    callable(allow_unmatched_file)
                                    and allow_unmatched_file(path)
                                ):
                                    raise HistoryArtifactCorruptionError(
                                        "history_v2_restart_unexpected_file"
                                    )
                            else:
                                raise HistoryArtifactCorruptionError(
                                    "history_v2_restart_special_file_is_forbidden"
                                )
                            check_scan_runtime()
            except (HistoryStorageError, HistoryBudgetExceededError):
                raise
            except OSError as exc:
                raise HistoryStorageError(
                    "history_v2_restart_namespace_scan_failed"
                ) from exc
            check_scan_runtime()
            return tuple(sorted(collected))

        known_namespaces = {"attempts", "raw", "normalized", "collections"}
        bounded_tree_scan(
            self.root,
            namespace_limit=_MAX_TEMP_SCAN_ENTRIES,
            resource="restart_root_files",
            recursive=False,
            include_file=lambda path: path.name.startswith(".")
            and path.name.endswith(".tmp"),
            reject_unmatched_files=True,
            allow_unmatched_file=lambda path: path.name == "scope.json",
            allow_directory=lambda path: path.name in known_namespaces,
        )

        for namespace_name in ("collections", "normalized"):
            namespace_root = self.root / namespace_name
            bounded_tree_scan(
                namespace_root,
                namespace_limit=_MAX_TEMP_SCAN_ENTRIES,
                resource=f"restart_{namespace_name}_root_files",
                recursive=False,
                include_file=lambda path: path.name.startswith(".")
                and path.name.endswith(".tmp"),
                reject_unmatched_files=True,
                allow_unmatched_file=lambda path: path.name in expected_request_ids,
                allow_directory=lambda path: path.name in expected_request_ids,
            )

        attempts_dir = self.root / "attempts"
        attempt_paths = bounded_tree_scan(
            attempts_dir,
            namespace_limit=scan_limit,
            resource="restart_attempt_files",
            recursive=False,
            include_file=lambda path: path.suffix == ".json",
            reject_unmatched_files=True,
            allow_unmatched_file=lambda path: path.name.startswith(".")
            and path.name.endswith(".tmp"),
        )
        attempt_request_ids: dict[str, str | None] = {}
        for path in attempt_paths:
            self._reject_reparse(path)
            try:
                receipt = _parse_canonical_json(
                    self._read_limited(
                        path,
                        limit=_MAX_ATTEMPT_RECEIPT_BYTES,
                        missing_code="history_v2_attempt_is_missing",
                    ),
                    code="history_v2_attempt_is_invalid",
                )
                page_payload = receipt.get("page_request")
                candidate = (
                    page_payload.get("range_request_id")
                    if isinstance(page_payload, dict)
                    else None
                )
                attempt_request_ids[path.stem] = (
                    candidate
                    if isinstance(candidate, str) and _SHA256_RE.fullmatch(candidate)
                    else None
                )
            except HistoryStorageError:
                attempt_request_ids[path.stem] = None
        states: list[HistoryRestartRequestStateV1] = []
        referenced_attempts: set[str] = set()
        referenced_raw: set[str] = set()
        alternate_normalized: list[str] = []
        for request in expected_requests:
            request_alternates: list[str] = []
            collection_dir = self.root / "collections" / request.request_id
            bounded_tree_scan(
                collection_dir,
                namespace_limit=_MAX_TEMP_SCAN_ENTRIES + 2,
                resource="restart_collection_files",
                recursive=False,
                include_file=lambda path: path.name
                in {"manifest.json", "admission.json"},
                reject_unmatched_files=True,
                allow_unmatched_file=lambda path: path.name.startswith(".")
                and path.name.endswith(".tmp"),
            )
            marker = collection_dir / "manifest.json"
            try:
                graph = self._load_complete_graph(
                    request,
                    expected_manifest_hash=expected_hashes.get(request.request_id),
                )
            except HistoryRestartIncompleteError as exc:
                normalized_dir = self.root / "normalized" / request.request_id
                has_attempt = request.request_id in attempt_request_ids.values()
                state = (
                    "incomplete"
                    if normalized_dir.exists() or marker.parent.exists() or has_attempt
                    else "absent"
                )
                states.append(
                    HistoryRestartRequestStateV1(
                        request_id=request.request_id,
                        state=state,
                        error_code=exc.code,
                    )
                )
                continue
            except HistoryArtifactForkError as exc:
                states.append(
                    HistoryRestartRequestStateV1(
                        request_id=request.request_id,
                        state="ambiguous_fork",
                        error_code=exc.code,
                    )
                )
                continue
            except HistoryStorageError as exc:
                states.append(
                    HistoryRestartRequestStateV1(
                        request_id=request.request_id,
                        state="corrupt",
                        error_code=exc.code,
                    )
                )
                continue
            referenced_attempts.update(graph.attempt_hashes)
            referenced_raw.update(graph.raw_hashes)
            normalized_dir = self.root / "normalized" / request.request_id
            expected_name = f"{graph.shard.manifest.normalized_shard_sha256}.jsonl"
            if normalized_dir.exists():
                normalized_paths = bounded_tree_scan(
                    normalized_dir,
                    namespace_limit=request.resource_limits.max_pages + 1,
                    resource="restart_normalized_files",
                    recursive=False,
                    include_file=lambda _path: True,
                    reject_unmatched_files=True,
                )
                for path in normalized_paths:
                    if path.name != expected_name and not path.name.endswith(".tmp"):
                        rendered = path.relative_to(self.root).as_posix()
                        alternate_normalized.append(rendered)
                        request_alternates.append(rendered)
            has_unreferenced_attempt = any(
                candidate == request.request_id and digest not in referenced_attempts
                for digest, candidate in attempt_request_ids.items()
            )
            state = (
                "ambiguous_fork"
                if request_alternates or has_unreferenced_attempt
                else "complete_verified"
            )
            states.append(
                HistoryRestartRequestStateV1(
                    request_id=request.request_id,
                    state=state,
                    manifest_hash=graph.shard.manifest.manifest_hash,
                )
            )
        temp_path_objects = bounded_tree_scan(
            self.root,
            namespace_limit=_MAX_TEMP_SCAN_ENTRIES,
            resource="restart_temp_files",
            recursive=True,
            include_file=lambda path: path.name.startswith(".")
            and path.name.endswith(".tmp"),
        )
        temp_paths = tuple(
            path.relative_to(self.root).as_posix() for path in temp_path_objects
        )
        raw_root = self.root / "raw"

        def raw_directory_is_canonical(path: Path) -> bool:
            parts = path.relative_to(raw_root).parts
            return parts == ("sha256",) or (
                len(parts) == 2
                and parts[0] == "sha256"
                and re.fullmatch(r"[0-9a-f]{2}", parts[1]) is not None
            )

        def raw_file_is_canonical(path: Path) -> bool:
            parts = path.relative_to(raw_root).parts
            return (
                len(parts) == 3
                and parts[0] == "sha256"
                and re.fullmatch(r"[0-9a-f]{2}", parts[1]) is not None
                and path.suffix == ".bin"
                and _SHA256_RE.fullmatch(path.stem) is not None
                and path.stem.startswith(parts[1])
            )

        raw_paths = bounded_tree_scan(
            raw_root,
            namespace_limit=scan_limit,
            resource="restart_raw_files",
            recursive=True,
            include_file=raw_file_is_canonical,
            reject_unmatched_files=True,
            allow_unmatched_file=lambda path: path.name.startswith(".")
            and path.name.endswith(".tmp"),
            allow_directory=raw_directory_is_canonical,
        )
        unreferenced_attempts = tuple(
            path.relative_to(self.root).as_posix()
            for path in attempt_paths
            if path.stem not in referenced_attempts
        )
        unreferenced_raw = tuple(
            path.relative_to(self.root).as_posix()
            for path in raw_paths
            if path.stem not in referenced_raw
        )
        ready = bool(states) and all(item.state == "complete_verified" for item in states) and not any(
            (unreferenced_attempts, unreferenced_raw, alternate_normalized)
        )
        check_scan_runtime()
        return HistoryRestartReportV1(
            request_states=tuple(states),
            temp_paths=temp_paths,
            unreferenced_attempt_paths=unreferenced_attempts,
            unreferenced_raw_paths=unreferenced_raw,
            alternate_normalized_paths=tuple(sorted(alternate_normalized)),
            ready=ready,
        )


class StrictMexcHistoryCollectorV2:
    def __init__(
        self,
        *,
        transport: RawHistoryTransportV2,
        store: StrictHistoryArtifactStoreV2,
        clock: EvidenceClock,
    ):
        if transport is None or not callable(getattr(transport, "fetch_page", None)):
            raise HistoryRangeContractError("history_v2_transport_is_required")
        if not isinstance(store, StrictHistoryArtifactStoreV2) or not store.writable:
            raise HistoryRangeContractError("history_v2_writable_store_is_required")
        if clock is None:
            raise HistoryRangeContractError("history_v2_clock_is_required")
        self.transport = transport
        self.store = store
        self.clock = clock

    @staticmethod
    def plan_pages(request: HistoryRangeRequestV2) -> tuple[KlinePageRequestV1, ...]:
        if not isinstance(request, HistoryRangeRequestV2):
            raise HistoryRangeContractError("history_v2_request_is_invalid")
        return _plan_pages(request)

    def _check_runtime(
        self, request: HistoryRangeRequestV2, started_monotonic_us: int
    ) -> int:
        if _clock_us(self.clock, "epoch_us") < request.collection_as_of_us:
            raise HistoryRangeContractError(
                "history_v2_clock_precedes_collection_as_of"
            )
        elapsed = _clock_us(self.clock, "monotonic_us") - started_monotonic_us
        if elapsed < 0:
            raise HistoryRangeContractError("history_v2_monotonic_clock_regressed")
        if elapsed > request.resource_limits.max_collection_runtime_us:
            raise HistoryBudgetExceededError(
                "collection_runtime_us",
                request.resource_limits.max_collection_runtime_us,
                elapsed,
            )
        return elapsed

    def _validate_transport_contract_identity(
        self, request: HistoryRangeRequestV2
    ) -> None:
        transport_identities = {
            "endpoint_contract_hash": request.endpoint_contract.contract_hash,
            "resource_limits_hash": request.resource_limits.contract_hash,
            "retry_policy_hash": request.retry_policy.contract_hash,
            "transport_contract_hash": request.attempt_contract_hash,
        }
        if any(
            getattr(self.transport, name, None) != expected
            for name, expected in transport_identities.items()
        ):
            raise HistoryRangeContractError(
                "history_v2_transport_contract_identity_mismatch"
            )

    @staticmethod
    def _validate_attempt_is_persistable(
        attempt: object,
        *,
        request: HistoryRangeRequestV2,
    ) -> HttpAttemptEvidenceV1:
        if not isinstance(
            attempt,
            (CompleteHttpAttemptEvidenceV1, IncompleteHttpAttemptEvidenceV1),
        ):
            raise HistoryTransportError(
                None, "history_v2_transport_returned_invalid_attempt_type"
            )
        if attempt.captured_body_length != len(attempt.body_bytes):
            raise HistoryTransportError(
                None, "history_v2_attempt_body_length_mismatch"
            )
        if (
            attempt.captured_body_length
            > request.resource_limits.max_raw_body_bytes_per_attempt
        ):
            raise HistoryBudgetExceededError(
                "raw_body_bytes_per_attempt",
                request.resource_limits.max_raw_body_bytes_per_attempt,
                attempt.captured_body_length,
            )
        receipt_size = len(_canonical_bytes(attempt.receipt_dict()) + b"\n")
        if receipt_size > _MAX_ATTEMPT_RECEIPT_BYTES:
            raise HistoryBudgetExceededError(
                "attempt_receipt_bytes",
                _MAX_ATTEMPT_RECEIPT_BYTES,
                receipt_size,
            )
        return attempt

    @staticmethod
    def _validate_attempt(
        attempt: HttpAttemptEvidenceV1,
        *,
        request: HistoryRangeRequestV2,
        page: KlinePageRequestV1,
        ordinal: int,
        prior: HttpAttemptEvidenceV1 | None,
    ) -> None:
        if attempt.page_request != page or attempt.attempt_ordinal != ordinal:
            raise HistoryTransportError(None, "history_v2_transport_returned_wrong_attempt")
        if (
            attempt.endpoint_contract_hash != request.endpoint_contract.contract_hash
            or attempt.resource_limits_hash != request.resource_limits.contract_hash
            or attempt.retry_policy_hash != request.retry_policy.contract_hash
            or attempt.transport_contract_hash != request.attempt_contract_hash
        ):
            raise HistoryTransportError(None, "history_v2_attempt_contract_identity_mismatch")
        if attempt.request_started_at_us < request.collection_as_of_us:
            raise HistoryTransportError(None, "history_v2_attempt_started_before_as_of")
        if prior is not None:
            if (
                attempt.request_started_at_us < prior.terminal_at_us
                or attempt.request_started_monotonic_us
                < prior.terminal_monotonic_us
            ):
                raise HistoryTransportError(None, "history_v2_attempt_timing_regressed")
            if (
                attempt.scheduled_not_before_us < prior.terminal_at_us
                or attempt.scheduled_not_before_monotonic_us
                < prior.terminal_monotonic_us
            ):
                raise HistoryTransportError(None, "history_v2_retry_schedule_regressed")
            if isinstance(prior, CompleteHttpAttemptEvidenceV1):
                retryable = prior.http_status in {408, 425, 429} or prior.http_status >= 500
                retry_after = retry_after_delay_us(
                    prior.safe_headers,
                    received_at_us=prior.terminal_at_us,
                    policy=request.retry_policy,
                )
            else:
                retryable = prior.outcome in {"network_error", "timeout"}
                retry_after = 0
            if not retryable:
                raise HistoryTransportError(
                    None, "history_v2_nonretryable_attempt_has_successor"
                )
            retry_delay = max(
                request.retry_policy.backoff_before_attempt_us(ordinal),
                retry_after,
            )
            if (
                attempt.scheduled_not_before_us
                < prior.terminal_at_us + retry_delay
                or attempt.scheduled_not_before_monotonic_us
                < prior.terminal_monotonic_us + retry_delay
            ):
                raise HistoryTransportError(
                    None, "history_v2_retry_delay_was_not_honoured"
                )
        if attempt.elapsed_monotonic_us > request.resource_limits.max_attempt_runtime_us:
            raise HistoryBudgetExceededError(
                "attempt_runtime_us",
                request.resource_limits.max_attempt_runtime_us,
                attempt.elapsed_monotonic_us,
            )

    def collect_range(self, request: HistoryRangeRequestV2) -> CompleteHistoryShardV2:
        if not isinstance(request, HistoryRangeRequestV2):
            raise HistoryRangeContractError("history_v2_request_is_invalid")
        self._validate_transport_contract_identity(request)
        started_monotonic = _clock_us(self.clock, "monotonic_us")
        with self.store.writer_session(request, clock=self.clock):
            return self._collect_range_locked(
                request, started_monotonic_us=started_monotonic
            )

    def _collect_range_locked(
        self,
        request: HistoryRangeRequestV2,
        *,
        started_monotonic_us: int,
    ) -> CompleteHistoryShardV2:
        if not isinstance(request, HistoryRangeRequestV2):
            raise HistoryRangeContractError("history_v2_request_is_invalid")
        if request.storage_profile != self.store.storage_profile:
            raise HistoryRangeContractError("history_v2_store_profile_request_mismatch")
        self._validate_transport_contract_identity(request)
        pages = _plan_pages(request)
        self._check_runtime(request, started_monotonic_us)
        rows: list[NormalizedHistoryRowV1] = []
        receipts: list[HistoryPageReceiptV2] = []
        attempt_count = 0
        raw_total = 0
        attempt_storage_total = len(self.store._scope_marker_bytes(request))
        prior_collection_terminal: int | None = None
        prior_collection_terminal_monotonic: int | None = None
        prior_collection_request_started: int | None = None
        prior_collection_request_started_monotonic: int | None = None
        observed_inter_attempt_gap_us = 0
        for page in pages:
            attempt_hashes: list[str] = []
            prior_attempt: HttpAttemptEvidenceV1 | None = None
            success: CompleteHttpAttemptEvidenceV1 | None = None
            for ordinal in range(request.resource_limits.max_attempts_per_page):
                self._check_runtime(request, started_monotonic_us)
                if attempt_count + 1 > request.resource_limits.max_total_attempts:
                    raise HistoryBudgetExceededError(
                        "attempts", request.resource_limits.max_total_attempts, attempt_count + 1
                    )
                # Do not start an attempt whose worst admissible captured prefix
                # cannot be retained.  Once an attempt starts, its complete or
                # partial evidence is persisted before any outcome failure.
                reserved_raw = (
                    raw_total
                    + request.resource_limits.max_raw_body_bytes_per_attempt
                )
                if reserved_raw > request.resource_limits.max_total_raw_body_bytes:
                    raise HistoryBudgetExceededError(
                        "total_raw_body_bytes",
                        request.resource_limits.max_total_raw_body_bytes,
                        reserved_raw,
                    )
                reserved_storage = (
                    attempt_storage_total
                    + request.resource_limits.max_raw_body_bytes_per_attempt
                    + _MAX_ATTEMPT_RECEIPT_BYTES
                )
                if reserved_storage > request.resource_limits.max_logical_storage_bytes:
                    raise HistoryBudgetExceededError(
                        "logical_storage_bytes",
                        request.resource_limits.max_logical_storage_bytes,
                        reserved_storage,
                    )
                attempt_window_started_epoch = _clock_us(self.clock, "epoch_us")
                attempt_window_started_monotonic = _clock_us(
                    self.clock, "monotonic_us"
                )
                attempt = self.transport.fetch_page(
                    page, attempt_ordinal=ordinal, prior_attempt=prior_attempt
                )
                attempt_window_ended_epoch = _clock_us(self.clock, "epoch_us")
                attempt_window_ended_monotonic = _clock_us(
                    self.clock, "monotonic_us"
                )
                attempt = self._validate_attempt_is_persistable(
                    attempt, request=request
                )
                # A started attempt is evidence even when its duration or the
                # enclosing collection crossed a deadline.  Persist the bounded
                # receipt/raw pair before enforcing those post-fetch failures.
                stored = self.store.persist_attempt(attempt)
                attempt_count += 1
                raw_total += attempt.captured_body_length
                attempt_storage_total += stored
                attempt_hashes.append(attempt.attempt_receipt_hash)
                prior_attempt_for_validation = prior_attempt
                prior_attempt = attempt
                if (
                    attempt.request_started_at_us < attempt_window_started_epoch
                    or attempt.terminal_at_us > attempt_window_ended_epoch
                    or attempt.request_started_monotonic_us
                    < attempt_window_started_monotonic
                    or attempt.terminal_monotonic_us
                    > attempt_window_ended_monotonic
                ):
                    raise HistoryTransportError(
                        attempt.receipt_dict(),
                        "history_v2_attempt_clock_domain_mismatch",
                    )
                if raw_total > request.resource_limits.max_total_raw_body_bytes:
                    raise HistoryBudgetExceededError(
                        "total_raw_body_bytes",
                        request.resource_limits.max_total_raw_body_bytes,
                        raw_total,
                    )
                if attempt_storage_total > request.resource_limits.max_logical_storage_bytes:
                    raise HistoryBudgetExceededError(
                        "logical_storage_bytes",
                        request.resource_limits.max_logical_storage_bytes,
                        attempt_storage_total,
                    )
                self._check_runtime(request, started_monotonic_us)
                self._validate_attempt(
                    attempt,
                    request=request,
                    page=page,
                    ordinal=ordinal,
                    prior=prior_attempt_for_validation,
                )
                if (
                    prior_collection_terminal is not None
                    and (
                        attempt.request_started_at_us < prior_collection_terminal
                        or attempt.request_started_monotonic_us
                        < prior_collection_terminal_monotonic
                    )
                ):
                    raise HistoryTransportError(
                        None, "history_v2_collection_timing_regressed"
                    )
                if prior_collection_request_started is not None and (
                    attempt.scheduled_not_before_us
                    < prior_collection_request_started
                    + request.retry_policy.min_request_spacing_us
                    or attempt.scheduled_not_before_monotonic_us
                    < prior_collection_request_started_monotonic
                    + request.retry_policy.min_request_spacing_us
                ):
                    raise HistoryTransportError(
                        None, "history_v2_collection_spacing_was_not_honoured"
                    )
                if prior_collection_terminal_monotonic is not None:
                    observed_inter_attempt_gap_us += max(
                        0,
                        attempt.request_started_monotonic_us
                        - prior_collection_terminal_monotonic,
                    )
                    if (
                        observed_inter_attempt_gap_us
                        > request.retry_policy.max_total_sleep_us
                    ):
                        raise HistoryBudgetExceededError(
                            "observed_inter_attempt_gap_us",
                            request.retry_policy.max_total_sleep_us,
                            observed_inter_attempt_gap_us,
                        )
                prior_collection_terminal = attempt.terminal_at_us
                prior_collection_terminal_monotonic = attempt.terminal_monotonic_us
                prior_collection_request_started = attempt.request_started_at_us
                prior_collection_request_started_monotonic = (
                    attempt.request_started_monotonic_us
                )
                if isinstance(attempt, CompleteHttpAttemptEvidenceV1) and attempt.body_complete:
                    if attempt.http_status == 200 and attempt.outcome == "complete":
                        success = attempt
                        break
                    retryable = attempt.http_status in {408, 425, 429} or (
                        type(attempt.http_status) is int and attempt.http_status >= 500
                    )
                    if not retryable:
                        raise HistoryHttpStatusError()
                elif attempt.outcome == "body_limit_exceeded":
                    raise HistoryBudgetExceededError(
                        "raw_body_bytes_per_attempt",
                        request.resource_limits.max_raw_body_bytes_per_attempt,
                        attempt.captured_body_length + 1,
                    )
            if success is None:
                if isinstance(prior_attempt, CompleteHttpAttemptEvidenceV1):
                    raise HistoryHttpStatusError()
                if isinstance(prior_attempt, IncompleteHttpAttemptEvidenceV1):
                    raise HistoryTransportError(
                        prior_attempt.receipt_dict(),
                        "history_v2_attempts_exhausted."
                        f"{prior_attempt.outcome}.{prior_attempt.safe_error_code}",
                    )
                raise HistoryTransportError(
                    None, "history_v2_attempts_exhausted_without_evidence"
                )
            normalized = _parse_mexc_page_v2(
                success, request=request, attempt_hashes=tuple(attempt_hashes)
            )
            self._check_runtime(request, started_monotonic_us)
            rows.extend(normalized.rows)
            receipts.append(normalized.receipt)
        expected = request.expected_timestamps()
        actual = tuple(row.bar_open_ts for row in rows)
        if len(set(actual)) != len(actual):
            raise HistoryDuplicateTimestampError()
        if actual != expected:
            raise HistoryIncompleteRangeError(
                "missing_timestamps",
                missing_timestamps=tuple(sorted(set(expected) - set(actual))),
                unexpected_timestamps=tuple(sorted(set(actual) - set(expected))),
            )
        normalized_bytes = b"".join(_canonical_bytes(row.as_dict()) + b"\n" for row in rows)
        logical_hash = _sha256_payload([row.market_dict() for row in rows])
        manifest = HistoryCollectionManifestV2(
            request=request,
            page_receipts=tuple(receipts),
            normalized_logical_hash=logical_hash,
            normalized_shard_sha256=_sha256_bytes(normalized_bytes),
            expected_row_count=len(expected),
            actual_row_count=len(rows),
            first_bar_open_ts=expected[0],
            last_bar_open_ts=expected[-1],
            completed_at_us=max(item.received_at_us for item in receipts),
            actual_attempt_count=attempt_count,
            actual_total_raw_body_bytes=raw_total,
            logical_storage_bytes=0,
            collection_runtime_us=self._check_runtime(
                request, started_monotonic_us
            ),
        )
        # The total includes the manifest carrying the total.  Iterate to the
        # stable decimal width/hash rather than estimating it.
        for _ in range(8):
            manifest_bytes = self.store._manifest_bytes(manifest)
            total = attempt_storage_total + len(normalized_bytes) + len(manifest_bytes)
            if total == manifest.logical_storage_bytes:
                break
            manifest = replace(manifest, logical_storage_bytes=total)
        else:
            raise HistoryStorageError("history_v2_logical_storage_total_did_not_stabilize")
        if manifest.logical_storage_bytes > request.resource_limits.max_logical_storage_bytes:
            raise HistoryBudgetExceededError(
                "logical_storage_bytes",
                request.resource_limits.max_logical_storage_bytes,
                manifest.logical_storage_bytes,
            )
        shard = CompleteHistoryShardV2(rows=tuple(rows), manifest=manifest)
        self._check_runtime(request, started_monotonic_us)
        self.store.publish_graph_candidate(shard)
        self._check_runtime(request, started_monotonic_us)
        # The manifest is evidence, not success.  The store reconstructs every
        # published byte, makes the final bounded runtime decision, and only
        # then installs the positive admission marker.
        admitted_shard = self.store.admit_reloaded_graph(
            request,
            expected_manifest_hash=manifest.manifest_hash,
            collection_started_monotonic_us=started_monotonic_us,
            clock=self.clock,
        )
        # The admitted boundary ends at atomic marker installation.  No later
        # fallible check can retroactively turn a visible admission into failure.
        return admitted_shard


_CONTRACT_SCHEMA = {
    "contract_version": STRICT_HISTORY_V2_CONTRACT_VERSION,
    "page_receipt_version": STRICT_HISTORY_V2_PAGE_RECEIPT_VERSION,
    "manifest_version": STRICT_HISTORY_V2_MANIFEST_VERSION,
    "restart_version": STRICT_HISTORY_V2_RESTART_VERSION,
    "storage_version": STRICT_HISTORY_V2_STORAGE_VERSION,
    "admission_version": STRICT_HISTORY_V2_ADMISSION_VERSION,
    "scope_version": STRICT_HISTORY_V2_SCOPE_VERSION,
    "normalized_row_version": row_contract_version(),
    "page_request_version": "strict_history_v1_byte_semantics",
    "identities": [
        "endpoint_contract",
        "resource_limits",
        "retry_policy",
        "attempt_transport_contract",
        "storage_profile",
    ],
    "dependency_hashes": {
        "attempt_transport_contract": mexc_futures_transport_contract_hash(),
    },
    "clock": "integer_epoch_and_monotonic_microseconds",
    "collector_transport_clock_binding": (
        "persist_then_require_attempt_start_terminal_inside_shared_clock_call_bracket"
    ),
    "collector_transport_identity_preflight": (
        "endpoint_resource_retry_and_attempt_contract_hashes_match_before_"
        "scope_marker_artifact_or_fetch"
    ),
    "collection_start": (
        "scope_payload_size_preflight_before_lock_or_artifact_then_bounded_"
        "restart_reconciliation_requires_pristine_absent_request_"
        "namespace_then_immutable_exact_request_scope_marker_no_automatic_"
        "resume_or_repair"
    ),
    "store_scope": (
        "one_dedicated_artifact_store_root_per_history_range_request_shard_"
        "immutable_scope_marker_bytes_are_included_in_graph_logical_storage_"
        "full_universe_requires_future_run_manifest_and_global_budgets"
    ),
    "writer_lock": (
        "nonblocking_process_local_and_os_file_lock_from_pristine_check_through_"
        "admission_session_binds_durable_store_request_scope_and_owner_thread_"
        "and_requires_pristine_namespace_before_yield_high_level_writes_require_"
        "that_exact_active_owner_release_is_non_revoking_cooperating_writers_"
        "contention_only_maps_to_fork_other_lock_io_is_storage_failure_not_"
        "adversarial_filesystem_mutation"
    ),
    "hard_limits": _HARD_LIMITS,
    "internal_artifact_limits": {
        "manifest_bytes": _MAX_MANIFEST_BYTES,
        "attempt_receipt_bytes": _MAX_ATTEMPT_RECEIPT_BYTES,
        "scope_marker_bytes": _MAX_SCOPE_MARKER_BYTES,
        "temp_scan_entries": _MAX_TEMP_SCAN_ENTRIES,
        "restart_scan_entries": _MAX_RESTART_SCAN_ENTRIES,
    },
    "completion": (
        "complete_200_body_full_grid_graph_manifest_then_fresh_full_disk_reload_"
        "and_positive_admission_marker_last"
    ),
    "admission_runtime": (
        "store_full_reload_then_final_pre_install_runtime_not_less_than_"
        "manifest_collection_runtime_low_level_marker_installer_is_private"
    ),
    "retry_revalidation": (
        "epoch_and_monotonic_global_spacing_retryable_predecessor_backoff_"
        "retry_after_and_observed_gap_cap"
    ),
    "runtime": (
        "integer_monotonic_deadline_through_full_reload_and_final_pre_admission_"
        "decision_no_revocable_post_admission_check"
    ),
    "raw_lengths": "actual_bytes_equal_attempt_and_page_redundant_lengths",
    "payload": {
        "json": "utf8_duplicate_keys_and_nonfinite_constants_rejected",
        "api_acceptance": "success_is_true_and_code_is_integer_zero_or_string_zero",
        "required_parallel_arrays": list(_REQUIRED_ARRAYS),
        "required_array_lengths": "equal_and_nonempty_for_expected_nonempty_page",
        "numbers": (
            "frozen_normalized_history_row_v1_exact_decimal_semantics_"
            "positive_ohlc_nonnegative_volume_and_turnover"
        ),
    },
    "payload_order": "exact_expected_ascending_grid_no_sort_or_dedup",
    "source_row_lineage": "ordinal_equals_position_in_exact_ascending_page_payload",
    "canonical_types": (
        "strict_json_integer_types_for_receipt_manifest_counts_ranges_and_clocks_"
        "python_numeric_equality_does_not_canonicalize_float_as_integer"
    ),
    "loader": "bounded_exact_canonical_full_graph_and_admission_before_return",
    "root_paths": (
        "reject_symlink_junction_or_reparse_in_supplied_chain_before_resolve_"
        "and_validate_existing_internal_parent_prefix_before_and_after_mkdir"
    ),
    "restart": (
        "streaming_cap_plus_one_entry_byte_time_bounded_read_only_no_repair_"
        "delete_promote_truncate_or_resume_exact_root_namespace_and_raw_path_shape_"
        "detached_anchor_keys_must_be_known_expected_request_ids_exactly_one_"
        "expected_request_per_shard_store_reported_temp_aliases_are_never_"
        "read_or_charged_against_evidence_byte_budget_and_do_not_invalidate_"
        "an_otherwise_complete_admitted_graph_while_entry_and_time_caps_apply"
    ),
    "fork": "fixed_manifest_no_overwrite_optional_detached_hash_and_residue_report",
    "storage_profiles": {
        WINDOWS_NTFS_DURABILITY_PROFILE_V1: _storage_contract_payload(
            WINDOWS_NTFS_DURABILITY_PROFILE_V1
        ),
        POSIX_DURABILITY_PROFILE_V1: _storage_contract_payload(
            POSIX_DURABILITY_PROFILE_V1
        ),
    },
    "windows_acceptance": (
        "process_crash_restart_verifiable_atomic_no_overwrite_visibility_only_"
        "not_parent_directory_or_sudden_power_loss_durability"
    ),
}


def _computed_contract_hash() -> str:
    return _sha256_payload(_CONTRACT_SCHEMA)


def strict_history_v2_contract_hash() -> str:
    digest = _computed_contract_hash()
    if digest != _PINNED_CONTRACT_HASH:
        raise RuntimeError("strict_history_v2_contract_changed_without_version_bump")
    return digest


__all__ = [
    "CompleteHistoryShardV2",
    "HistoryArtifactCorruptionError",
    "HistoryArtifactForkError",
    "HistoryBudgetExceededError",
    "HistoryCollectionManifestV2",
    "HistoryPageReceiptV2",
    "HistoryRangeRequestV2",
    "HistoryRestartIncompleteError",
    "HistoryRestartReportV1",
    "HistoryRestartRequestStateV1",
    "POSIX_DURABILITY_PROFILE_V1",
    "RawHistoryTransportV2",
    "STRICT_HISTORY_V2_CONTRACT_VERSION",
    "StrictHistoryArtifactStoreV2",
    "StrictMexcHistoryCollectorV2",
    "WINDOWS_NTFS_DURABILITY_PROFILE_V1",
    "storage_profile_hash",
    "strict_history_v2_contract_hash",
]
