"""Offline-testable candidate MEXC Futures endpoint and transport evidence.

This module deliberately provides no real network executor.  It freezes the
request target, resource limits, retry policy, evidence clocks, and streaming
attempt receipts that a later explicitly-authorized public-data adapter must
obey.  The bundled endpoint is a candidate, not an assertion that U5 or current
official/live verification has occurred.

The legacy :mod:`trading.market_data.mexc_client` and strict-history v1 writer
are intentionally not imported as transports or mutated here.  Only the
immutable :class:`~trading.market_data.strict_history.KlinePageRequestV1` page
shape is reused at this boundary.
"""

from __future__ import annotations

from base64 import urlsafe_b64encode
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any, Iterator, Protocol, Sequence, TypeAlias

from trading.market_data.strict_history import KlinePageRequestV1


MEXC_FUTURES_ENDPOINT_CONTRACT_VERSION = (
    "mexc_futures_kline_endpoint_candidate_v1"
)
MEXC_FUTURES_TRANSPORT_CONTRACT_VERSION = "mexc_futures_raw_transport_v1"
HISTORY_RESOURCE_LIMITS_VERSION = "mexc_history_resource_limits_v1"
HISTORY_RETRY_POLICY_VERSION = "mexc_history_retry_policy_v1"
COMPLETE_HTTP_ATTEMPT_VERSION = "mexc_complete_http_attempt_v1"
INCOMPLETE_HTTP_ATTEMPT_VERSION = "mexc_incomplete_http_attempt_v1"

HARD_MAX_PAGES = 200
HARD_MAX_ROWS = 400_000
HARD_MAX_ATTEMPTS_PER_PAGE = 10
HARD_MAX_TOTAL_ATTEMPTS = 2_000
HARD_MAX_RAW_BODY_BYTES_PER_ATTEMPT = 8 * 1024 * 1024
HARD_MAX_TOTAL_RAW_BODY_BYTES = 256 * 1024 * 1024
HARD_MAX_LOGICAL_STORAGE_BYTES = 512 * 1024 * 1024
HARD_MAX_COLLECTION_RUNTIME_US = 60 * 60 * 1_000_000
HARD_MAX_ATTEMPT_RUNTIME_US = 60 * 1_000_000
HARD_MAX_MIN_REQUEST_SPACING_US = 60 * 1_000_000
HARD_MAX_BACKOFF_US = 30 * 1_000_000
HARD_MAX_RETRY_AFTER_US = 120 * 1_000_000
HARD_MAX_TOTAL_SLEEP_US = 15 * 60 * 1_000_000
HARD_MAX_RESPONSE_HEADER_COUNT = 64
HARD_MAX_RESPONSE_HEADER_NAME_CHARS = 64
HARD_MAX_RESPONSE_HEADER_VALUE_CHARS = 4_096
HARD_MAX_RESPONSE_HEADER_AGGREGATE_CHARS = 16_384

_PINNED_CANDIDATE_ENDPOINT_HASH = (
    "54f57d755cd679eb92444d48b38013621caad37067125e80cf7c5e45fe2ab220"
)
_PINNED_CANDIDATE_RESOURCE_LIMITS_HASH = (
    "937d053e33c513d128389259e308156c8758e5cfe44b5849e3eb27ea49d96bdc"
)
_PINNED_CANDIDATE_RETRY_POLICY_HASH = (
    "78f92d14cc26ead1a372d840a05fe8a60dae97d5d9a3cdacc539a098194a2cc9"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_HEADER_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?$")
_ETAG_RE = re.compile(r'^(W/)?"([^"\\]*)"$')
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
_FORBIDDEN_REQUEST_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "signature",
        "request-time",
    }
)
_ENDPOINT_KEYS = frozenset(
    {
        "authentication",
        "contract_version",
        "host",
        "identity_prefix",
        "method",
        "path_template",
        "port",
        "query_encoding",
        "query_order",
        "redirects",
        "request_headers",
        "scheme",
        "tls_verification",
        "verification",
    }
)
_VERIFICATION_KEYS = frozenset(
    {
        "current_official_docs_verified",
        "live_endpoint_verified",
        "plan_reference_url",
        "status",
    }
)
_ATTEMPT_RECEIPT_KEYS = frozenset(
    {
        "contract_version",
        "page_request",
        "page_id",
        "attempt_ordinal",
        "endpoint_contract_hash",
        "resource_limits_hash",
        "retry_policy_hash",
        "transport_contract_hash",
        "scheduled_not_before_us",
        "scheduled_not_before_monotonic_us",
        "request_started_at_us",
        "request_started_monotonic_us",
        "headers_received_at_us",
        "terminal_at_us",
        "terminal_monotonic_us",
        "elapsed_monotonic_us",
        "http_status",
        "safe_headers",
        "body_complete",
        "outcome",
        "safe_error_code",
        "captured_body_length",
        "captured_body_sha256",
    }
)


class MexcFuturesTransportError(RuntimeError):
    """Stable, non-secret-bearing base error for the frozen foundation."""


class EndpointContractError(MexcFuturesTransportError):
    pass


class ResourceLimitContractError(MexcFuturesTransportError):
    pass


class ResourceBudgetExceededError(MexcFuturesTransportError):
    pass


class RetryPolicyContractError(MexcFuturesTransportError):
    pass


class RetryAfterContractError(MexcFuturesTransportError):
    pass


class TransportContractError(MexcFuturesTransportError):
    pass


class StreamingExecutorNetworkError(MexcFuturesTransportError):
    def __init__(self, safe_error_code: str = "executor_network_error"):
        self.safe_error_code = _safe_code(safe_error_code)
        super().__init__(self.safe_error_code)


class StreamingExecutorTimeoutError(MexcFuturesTransportError):
    def __init__(self, safe_error_code: str = "executor_timeout"):
        self.safe_error_code = _safe_code(safe_error_code)
        super().__init__(self.safe_error_code)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EndpointContractError("candidate_endpoint_duplicate_json_key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise EndpointContractError("candidate_endpoint_nonfinite_json")


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
        raise MexcFuturesTransportError("transport_payload_not_canonical") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_payload(payload: object) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MexcFuturesTransportError(f"{field}_is_invalid")
    return value


def _bounded_int(
    value: object, *, field: str, minimum: int, maximum: int
) -> int:
    if type(value) is not int or value < minimum:
        raise ResourceLimitContractError(f"{field}_is_invalid")
    result = value
    if result > maximum:
        raise ResourceLimitContractError(f"{field}_exceeds_hard_cap")
    return result


def _safe_code(value: object) -> str:
    if not isinstance(value, str) or not _SAFE_CODE_RE.fullmatch(value):
        raise TransportContractError("safe_error_code_is_invalid")
    return value


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TransportContractError(f"{field}_is_invalid")
    return value


def _exact_tuple_pairs(
    value: object, *, field: str
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple):
        raise TransportContractError(f"{field}_must_be_an_immutable_tuple")
    pairs: list[tuple[str, str]] = []
    for pair in value:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not isinstance(pair[1], str)
        ):
            raise TransportContractError(f"{field}_is_invalid")
        pairs.append(pair)
    return tuple(pairs)


@dataclass(frozen=True)
class MexcFuturesEndpointContractV1:
    contract_version: str
    verification_status: str
    current_official_docs_verified: bool
    live_endpoint_verified: bool
    plan_reference_url: str
    identity_prefix: str
    scheme: str
    host: str
    port: int
    method: str
    path_template: str
    query_order: tuple[tuple[str, str], ...]
    query_encoding: str
    authentication: str
    redirects: str
    tls_verification: str
    request_headers: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.contract_version != MEXC_FUTURES_ENDPOINT_CONTRACT_VERSION:
            raise EndpointContractError("candidate_endpoint_version_mismatch")
        if self.verification_status != "candidate_not_u5_verified":
            raise EndpointContractError("candidate_endpoint_status_mismatch")
        if (
            type(self.current_official_docs_verified) is not bool
            or self.current_official_docs_verified
            or type(self.live_endpoint_verified) is not bool
            or self.live_endpoint_verified
        ):
            raise EndpointContractError("candidate_endpoint_must_remain_unverified")
        if not isinstance(self.plan_reference_url, str) or not self.plan_reference_url.startswith(
            "https://www.mexc.com/"
        ):
            raise EndpointContractError("candidate_endpoint_reference_is_invalid")
        if not isinstance(self.identity_prefix, str) or not _IDENTIFIER_RE.fullmatch(
            self.identity_prefix
        ):
            raise EndpointContractError("candidate_endpoint_identity_prefix_invalid")
        if self.scheme != "https" or self.host != "api.mexc.com" or self.port != 443:
            raise EndpointContractError("candidate_endpoint_authority_mismatch")
        if not _HOST_RE.fullmatch(self.host):
            raise EndpointContractError("candidate_endpoint_host_invalid")
        if self.method != "GET" or self.path_template != "/api/v1/contract/kline/{venue_symbol}":
            raise EndpointContractError("candidate_endpoint_target_mismatch")
        if self.query_order != (
            ("interval", "interval"),
            ("start", "start_open_ts"),
            ("end", "end_open_ts_inclusive"),
        ):
            raise EndpointContractError("candidate_endpoint_query_mismatch")
        if self.query_encoding != "ascii_exact_ordered":
            raise EndpointContractError("candidate_endpoint_query_encoding_mismatch")
        if self.authentication != "none" or self.redirects != "reject":
            raise EndpointContractError("candidate_endpoint_public_semantics_mismatch")
        if self.tls_verification != "required":
            raise EndpointContractError("candidate_endpoint_tls_mismatch")
        expected_headers = (
            ("accept", "application/json"),
            ("accept-encoding", "identity"),
            ("user-agent", "koteika-strict-history/1.0"),
        )
        if self.request_headers != expected_headers:
            raise EndpointContractError("candidate_endpoint_request_headers_mismatch")
        if any(name in _FORBIDDEN_REQUEST_HEADER_NAMES for name, _ in self.request_headers):
            raise EndpointContractError("candidate_endpoint_credentials_forbidden")

    def as_dict(self) -> dict[str, object]:
        return {
            "authentication": self.authentication,
            "contract_version": self.contract_version,
            "host": self.host,
            "identity_prefix": self.identity_prefix,
            "method": self.method,
            "path_template": self.path_template,
            "port": self.port,
            "query_encoding": self.query_encoding,
            "query_order": [list(pair) for pair in self.query_order],
            "redirects": self.redirects,
            "request_headers": [list(pair) for pair in self.request_headers],
            "scheme": self.scheme,
            "tls_verification": self.tls_verification,
            "verification": {
                "current_official_docs_verified": self.current_official_docs_verified,
                "live_endpoint_verified": self.live_endpoint_verified,
                "plan_reference_url": self.plan_reference_url,
                "status": self.verification_status,
            },
        }

    @property
    def contract_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    @property
    def endpoint_identity(self) -> str:
        result = f"{self.identity_prefix}.{self.contract_hash}"
        if not _IDENTIFIER_RE.fullmatch(result):
            raise EndpointContractError("candidate_endpoint_identity_invalid")
        return result

    def prepare(self, page_request: KlinePageRequestV1) -> "PreparedPublicRequestV1":
        if not isinstance(page_request, KlinePageRequestV1):
            raise EndpointContractError("candidate_endpoint_page_request_invalid")
        if page_request.endpoint_identity != self.endpoint_identity:
            raise EndpointContractError("candidate_endpoint_identity_mismatch")
        if page_request.canonical_path != self.path_template.format(
            venue_symbol=page_request.venue_symbol
        ):
            raise EndpointContractError("candidate_endpoint_path_mismatch")
        return PreparedPublicRequestV1(
            endpoint_identity=self.endpoint_identity,
            endpoint_contract_hash=self.contract_hash,
            method=self.method,
            scheme=self.scheme,
            host=self.host,
            port=self.port,
            path=page_request.canonical_path,
            query=(
                ("interval", page_request.interval),
                ("start", str(page_request.start_open_ts)),
                ("end", str(page_request.end_open_ts_inclusive)),
            ),
            headers=self.request_headers,
            tls_verify=True,
            allow_redirects=False,
            trust_env=False,
            body=None,
        )


def candidate_endpoint_fixture_path() -> Path:
    return Path(__file__).with_name("fixtures") / (
        "mexc_futures_kline_endpoint_candidate_v1.json"
    )


def load_mexc_futures_endpoint_contract_v1(
    path: str | Path,
) -> MexcFuturesEndpointContractV1:
    fixture_path = Path(path)
    try:
        raw = fixture_path.read_bytes()
    except OSError as exc:
        raise EndpointContractError("candidate_endpoint_fixture_unreadable") from exc
    if len(raw) > 64 * 1024:
        raise EndpointContractError("candidate_endpoint_fixture_oversized")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except EndpointContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EndpointContractError("candidate_endpoint_fixture_invalid_json") from exc
    if not isinstance(payload, dict) or set(payload) != _ENDPOINT_KEYS:
        raise EndpointContractError("candidate_endpoint_fixture_schema_mismatch")
    verification = payload.get("verification")
    if not isinstance(verification, dict) or set(verification) != _VERIFICATION_KEYS:
        raise EndpointContractError("candidate_endpoint_verification_schema_mismatch")
    if raw != _canonical_bytes(payload) + b"\n":
        raise EndpointContractError("candidate_endpoint_fixture_not_canonical")
    query_order = payload.get("query_order")
    request_headers = payload.get("request_headers")
    if not isinstance(query_order, list) or not isinstance(request_headers, list):
        raise EndpointContractError("candidate_endpoint_fixture_schema_mismatch")
    try:
        query_pairs = tuple((pair[0], pair[1]) for pair in query_order)
        header_pairs = tuple((pair[0], pair[1]) for pair in request_headers)
    except (IndexError, TypeError) as exc:
        raise EndpointContractError("candidate_endpoint_fixture_schema_mismatch") from exc
    if any(
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(item, str) for item in pair)
        for pair in (*query_order, *request_headers)
    ):
        raise EndpointContractError("candidate_endpoint_fixture_schema_mismatch")
    result = MexcFuturesEndpointContractV1(
        contract_version=payload["contract_version"],
        verification_status=verification["status"],
        current_official_docs_verified=verification[
            "current_official_docs_verified"
        ],
        live_endpoint_verified=verification["live_endpoint_verified"],
        plan_reference_url=verification["plan_reference_url"],
        identity_prefix=payload["identity_prefix"],
        scheme=payload["scheme"],
        host=payload["host"],
        port=payload["port"],
        method=payload["method"],
        path_template=payload["path_template"],
        query_order=query_pairs,
        query_encoding=payload["query_encoding"],
        authentication=payload["authentication"],
        redirects=payload["redirects"],
        tls_verification=payload["tls_verification"],
        request_headers=header_pairs,
    )
    if result.contract_hash != _PINNED_CANDIDATE_ENDPOINT_HASH:
        raise EndpointContractError("candidate_endpoint_contract_hash_drift")
    return result


@dataclass(frozen=True)
class HistoryResourceLimitsV1:
    max_pages: int
    max_rows: int
    max_attempts_per_page: int
    max_total_attempts: int
    max_raw_body_bytes_per_attempt: int
    max_total_raw_body_bytes: int
    max_logical_storage_bytes: int
    max_collection_runtime_us: int
    max_attempt_runtime_us: int
    contract_version: str = HISTORY_RESOURCE_LIMITS_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != HISTORY_RESOURCE_LIMITS_VERSION:
            raise ResourceLimitContractError("resource_limits_version_mismatch")
        _bounded_int(self.max_pages, field="max_pages", minimum=1, maximum=HARD_MAX_PAGES)
        _bounded_int(self.max_rows, field="max_rows", minimum=1, maximum=HARD_MAX_ROWS)
        _bounded_int(
            self.max_attempts_per_page,
            field="max_attempts_per_page",
            minimum=1,
            maximum=HARD_MAX_ATTEMPTS_PER_PAGE,
        )
        _bounded_int(
            self.max_total_attempts,
            field="max_total_attempts",
            minimum=1,
            maximum=HARD_MAX_TOTAL_ATTEMPTS,
        )
        _bounded_int(
            self.max_raw_body_bytes_per_attempt,
            field="max_raw_body_bytes_per_attempt",
            minimum=1,
            maximum=HARD_MAX_RAW_BODY_BYTES_PER_ATTEMPT,
        )
        _bounded_int(
            self.max_total_raw_body_bytes,
            field="max_total_raw_body_bytes",
            minimum=1,
            maximum=HARD_MAX_TOTAL_RAW_BODY_BYTES,
        )
        _bounded_int(
            self.max_logical_storage_bytes,
            field="max_logical_storage_bytes",
            minimum=1,
            maximum=HARD_MAX_LOGICAL_STORAGE_BYTES,
        )
        _bounded_int(
            self.max_collection_runtime_us,
            field="max_collection_runtime_us",
            minimum=1,
            maximum=HARD_MAX_COLLECTION_RUNTIME_US,
        )
        _bounded_int(
            self.max_attempt_runtime_us,
            field="max_attempt_runtime_us",
            minimum=1,
            maximum=HARD_MAX_ATTEMPT_RUNTIME_US,
        )
        if self.max_total_attempts < self.max_attempts_per_page:
            raise ResourceLimitContractError("total_attempts_below_per_page_limit")
        if self.max_total_raw_body_bytes < self.max_raw_body_bytes_per_attempt:
            raise ResourceLimitContractError("total_raw_body_below_attempt_limit")
        if self.max_attempt_runtime_us > self.max_collection_runtime_us:
            raise ResourceLimitContractError("attempt_runtime_exceeds_collection_runtime")

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "max_pages": self.max_pages,
            "max_rows": self.max_rows,
            "max_attempts_per_page": self.max_attempts_per_page,
            "max_total_attempts": self.max_total_attempts,
            "max_raw_body_bytes_per_attempt": self.max_raw_body_bytes_per_attempt,
            "max_total_raw_body_bytes": self.max_total_raw_body_bytes,
            "max_logical_storage_bytes": self.max_logical_storage_bytes,
            "max_collection_runtime_us": self.max_collection_runtime_us,
            "max_attempt_runtime_us": self.max_attempt_runtime_us,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "HistoryResourceLimitsV1":
        keys = set(cls.__dataclass_fields__)
        if not isinstance(payload, dict) or set(payload) != keys:
            raise ResourceLimitContractError("resource_limits_schema_mismatch")
        return cls(**payload)

    @property
    def contract_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    @property
    def identity(self) -> str:
        return f"{self.contract_version}.{self.contract_hash}"

    def validate_request_shape(
        self, *, required_pages: int, expected_rows: int, max_attempts_per_page: int
    ) -> None:
        pages = _strict_int(required_pages, field="required_pages", minimum=1)
        rows = _strict_int(expected_rows, field="expected_rows", minimum=1)
        attempts = _strict_int(
            max_attempts_per_page, field="request_max_attempts_per_page", minimum=1
        )
        if pages > self.max_pages:
            raise ResourceBudgetExceededError("history_page_budget_exceeded")
        if rows > self.max_rows:
            raise ResourceBudgetExceededError("history_row_budget_exceeded")
        if attempts > self.max_attempts_per_page:
            raise ResourceBudgetExceededError("history_attempts_per_page_exceeded")
        if pages * attempts > self.max_total_attempts:
            raise ResourceBudgetExceededError("history_total_attempt_budget_exceeded")


def candidate_history_resource_limits_v1() -> HistoryResourceLimitsV1:
    result = HistoryResourceLimitsV1(
        max_pages=200,
        max_rows=400_000,
        max_attempts_per_page=3,
        max_total_attempts=600,
        max_raw_body_bytes_per_attempt=8 * 1024 * 1024,
        max_total_raw_body_bytes=256 * 1024 * 1024,
        max_logical_storage_bytes=512 * 1024 * 1024,
        max_collection_runtime_us=60 * 60 * 1_000_000,
        max_attempt_runtime_us=30 * 1_000_000,
    )
    if result.contract_hash != _PINNED_CANDIDATE_RESOURCE_LIMITS_HASH:
        raise ResourceLimitContractError("candidate_resource_limits_hash_drift")
    return result


@dataclass(frozen=True)
class HistoryRetryPolicyV1:
    min_request_spacing_us: int
    base_backoff_us: int
    backoff_multiplier: int
    max_backoff_us: int
    max_retry_after_us: int
    max_total_sleep_us: int
    contract_version: str = HISTORY_RETRY_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != HISTORY_RETRY_POLICY_VERSION:
            raise RetryPolicyContractError("retry_policy_version_mismatch")
        for field, minimum in (
            ("min_request_spacing_us", 0),
            ("base_backoff_us", 1),
            ("max_backoff_us", 1),
            ("max_retry_after_us", 0),
            ("max_total_sleep_us", 0),
        ):
            _strict_int(getattr(self, field), field=field, minimum=minimum)
        _strict_int(self.backoff_multiplier, field="backoff_multiplier", minimum=1)
        if self.backoff_multiplier > 16:
            raise RetryPolicyContractError("backoff_multiplier_is_out_of_range")
        if self.base_backoff_us > self.max_backoff_us:
            raise RetryPolicyContractError("base_backoff_exceeds_cap")
        if self.min_request_spacing_us > HARD_MAX_MIN_REQUEST_SPACING_US:
            raise RetryPolicyContractError("min_request_spacing_exceeds_hard_cap")
        if self.max_backoff_us > HARD_MAX_BACKOFF_US:
            raise RetryPolicyContractError("max_backoff_exceeds_hard_cap")
        if self.max_retry_after_us > HARD_MAX_RETRY_AFTER_US:
            raise RetryPolicyContractError("max_retry_after_exceeds_hard_cap")
        if self.max_total_sleep_us > HARD_MAX_TOTAL_SLEEP_US:
            raise RetryPolicyContractError("max_total_sleep_exceeds_hard_cap")

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "min_request_spacing_us": self.min_request_spacing_us,
            "base_backoff_us": self.base_backoff_us,
            "backoff_multiplier": self.backoff_multiplier,
            "max_backoff_us": self.max_backoff_us,
            "max_retry_after_us": self.max_retry_after_us,
            "max_total_sleep_us": self.max_total_sleep_us,
            "jitter": "none",
        }

    @classmethod
    def from_dict(cls, payload: object) -> "HistoryRetryPolicyV1":
        expected = set(cls.__dataclass_fields__) | {"jitter"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise RetryPolicyContractError("retry_policy_schema_mismatch")
        if payload.get("jitter") != "none":
            raise RetryPolicyContractError("retry_policy_jitter_forbidden")
        values = dict(payload)
        values.pop("jitter")
        return cls(**values)

    @property
    def contract_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    @property
    def identity(self) -> str:
        return f"{self.contract_version}.{self.contract_hash}"

    def backoff_before_attempt_us(self, attempt_ordinal: int) -> int:
        ordinal = _strict_int(
            attempt_ordinal, field="attempt_ordinal", minimum=1
        )
        value = self.base_backoff_us
        for _ in range(ordinal - 1):
            if value >= self.max_backoff_us:
                return self.max_backoff_us
            value = min(self.max_backoff_us, value * self.backoff_multiplier)
        return min(value, self.max_backoff_us)


def candidate_history_retry_policy_v1() -> HistoryRetryPolicyV1:
    result = HistoryRetryPolicyV1(
        min_request_spacing_us=500_000,
        base_backoff_us=1_000_000,
        backoff_multiplier=2,
        max_backoff_us=30_000_000,
        max_retry_after_us=120_000_000,
        max_total_sleep_us=15 * 60 * 1_000_000,
    )
    if result.contract_hash != _PINNED_CANDIDATE_RETRY_POLICY_HASH:
        raise RetryPolicyContractError("candidate_retry_policy_hash_drift")
    return result


class EvidenceClock(Protocol):
    """Exact integer-microsecond evidence and deterministic sleep clock."""

    def epoch_us(self) -> int: ...

    def monotonic_us(self) -> int: ...

    def sleep_us(self, duration_us: int) -> None: ...


@dataclass(frozen=True)
class PreparedPublicRequestV1:
    endpoint_identity: str
    endpoint_contract_hash: str
    method: str
    scheme: str
    host: str
    port: int
    path: str
    query: tuple[tuple[str, str], ...]
    headers: tuple[tuple[str, str], ...]
    tls_verify: bool
    allow_redirects: bool
    trust_env: bool
    body: None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_identity, str) or not _IDENTIFIER_RE.fullmatch(
            self.endpoint_identity
        ):
            raise TransportContractError("prepared_endpoint_identity_invalid")
        _digest(self.endpoint_contract_hash, field="prepared_endpoint_contract_hash")
        if (
            self.method != "GET"
            or self.scheme != "https"
            or self.host != "api.mexc.com"
            or self.port != 443
            or not self.path.startswith("/api/v1/contract/kline/")
        ):
            raise TransportContractError("prepared_request_target_invalid")
        if self.query != tuple(self.query) or tuple(name for name, _ in self.query) != (
            "interval",
            "start",
            "end",
        ):
            raise TransportContractError("prepared_request_query_invalid")
        _exact_tuple_pairs(self.query, field="prepared_query")
        _exact_tuple_pairs(self.headers, field="prepared_headers")
        if any(name in _FORBIDDEN_REQUEST_HEADER_NAMES for name, _ in self.headers):
            raise TransportContractError("prepared_request_credentials_forbidden")
        if (
            self.tls_verify is not True
            or self.allow_redirects is not False
            or self.trust_env is not False
            or self.body is not None
        ):
            raise TransportContractError("prepared_request_safety_semantics_invalid")

    @property
    def url(self) -> str:
        query = "&".join(f"{name}={value}" for name, value in self.query)
        return f"{self.scheme}://{self.host}{self.path}?{query}"


class StreamingHttpResponse(Protocol):
    @property
    def http_status(self) -> int: ...

    @property
    def headers(self) -> Sequence[tuple[str, str]]: ...

    def iter_body(self, chunk_size: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class StreamingHttpExecutor(Protocol):
    """Injected one-attempt executor; there is intentionally no implementation."""

    def open(
        self,
        request: PreparedPublicRequestV1,
        *,
        connect_timeout_us: int,
        read_timeout_us: int,
    ) -> StreamingHttpResponse: ...


def _invalid_header_marker(value: str) -> str:
    return f"invalid.{_sha256_bytes(value.encode('utf-8', errors='surrogatepass'))}"


def _canonical_http_date(value: str) -> str:
    if re.fullmatch(r"unix=0|unix=[1-9][0-9]*", value):
        return value
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            raise ValueError("date_is_naive")
        normalized = parsed.astimezone(timezone.utc)
        if normalized.strftime("%a, %d %b %Y %H:%M:%S GMT") != value:
            raise ValueError("date_is_not_imf_fixdate")
        epoch = int(normalized.timestamp())
        if normalized.microsecond != 0:
            raise ValueError("date_has_subseconds")
        if epoch < 0:
            raise ValueError("date_precedes_unix_epoch")
        return f"unix={epoch}"
    except (TypeError, ValueError, OverflowError):
        return _invalid_header_marker(value)


def canonicalize_public_response_headers(
    headers: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Return only non-credential public headers in canonical safe form."""

    if not isinstance(headers, Sequence) or isinstance(headers, (str, bytes)):
        raise TransportContractError("response_headers_invalid")
    if len(headers) > HARD_MAX_RESPONSE_HEADER_COUNT:
        raise TransportContractError("response_header_count_exceeds_hard_cap")
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    aggregate_chars = 0
    for pair in headers:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not isinstance(pair[1], str)
        ):
            raise TransportContractError("response_headers_invalid")
        aggregate_chars += len(pair[0]) + len(pair[1])
        if aggregate_chars > HARD_MAX_RESPONSE_HEADER_AGGREGATE_CHARS:
            raise TransportContractError(
                "response_header_aggregate_exceeds_hard_cap"
            )
        if len(pair[0]) > HARD_MAX_RESPONSE_HEADER_NAME_CHARS:
            continue
        name = pair[0].strip().lower()
        raw_value = pair[1]
        value_oversized = len(raw_value) > HARD_MAX_RESPONSE_HEADER_VALUE_CHARS
        value = raw_value if value_oversized else raw_value.strip()
        if not _HEADER_NAME_RE.fullmatch(name):
            continue
        if name not in _PUBLIC_SAFE_HEADER_NAMES:
            continue
        if name in seen:
            raise TransportContractError("response_safe_header_duplicate")
        seen.add(name)
        if value_oversized:
            rendered = "invalid.oversized"
        elif "\r" in raw_value or "\n" in raw_value:
            rendered = _invalid_header_marker(raw_value)
        elif re.fullmatch(r"invalid\.(?:[0-9a-f]{64}|oversized)", value):
            rendered = value
        elif name in {"date", "last-modified"}:
            rendered = _canonical_http_date(value)
        elif name == "etag":
            canonical_etag = re.fullmatch(
                r"(?:strong|weak)\.[A-Za-z0-9_-]*", value
            )
            match = _ETAG_RE.fullmatch(value)
            if canonical_etag is not None:
                rendered = value
            elif match is None:
                rendered = _invalid_header_marker(value)
            else:
                prefix = "weak" if match.group(1) else "strong"
                opaque = urlsafe_b64encode(match.group(2).encode("utf-8")).decode(
                    "ascii"
                ).rstrip("=")
                rendered = f"{prefix}.{opaque}"
        elif name == "retry-after":
            if re.fullmatch(r"(?:delay|unix)=(?:0|[1-9][0-9]*)", value):
                rendered = value
            elif re.fullmatch(r"0|[1-9][0-9]*", value):
                rendered = f"delay={int(value)}"
            else:
                rendered = _canonical_http_date(value)
        elif name == "content-length":
            if re.fullmatch(r"0|[1-9][0-9]*", value):
                rendered = str(int(value))
            else:
                rendered = _invalid_header_marker(value)
        elif any(ord(char) < 32 or ord(char) > 126 for char in value):
            rendered = _invalid_header_marker(value)
        else:
            rendered = value
        normalized.append((name, rendered))
    return tuple(sorted(normalized))


def _header_value(headers: tuple[tuple[str, str], ...], name: str) -> str | None:
    for candidate, value in headers:
        if candidate == name:
            return value
    return None


def retry_after_delay_us(
    headers: tuple[tuple[str, str], ...],
    *,
    received_at_us: int,
    policy: HistoryRetryPolicyV1,
) -> int:
    received = _strict_int(received_at_us, field="received_at_us", minimum=0)
    value = _header_value(headers, "retry-after")
    if value is None:
        return 0
    if value.startswith("invalid."):
        raise RetryAfterContractError("retry_after_is_invalid")
    if value.startswith("delay="):
        seconds = int(value.removeprefix("delay="))
        delay = seconds * 1_000_000
    elif value.startswith("unix="):
        target = int(value.removeprefix("unix=")) * 1_000_000
        delay = max(0, target - received)
    else:
        raise RetryAfterContractError("retry_after_is_invalid")
    if delay > policy.max_retry_after_us:
        raise RetryAfterContractError("retry_after_exceeds_policy_cap")
    return delay


def _validate_common_evidence(
    *,
    page_request: KlinePageRequestV1,
    attempt_ordinal: int,
    endpoint_contract_hash: str,
    resource_limits_hash: str,
    retry_policy_hash: str,
    transport_contract_hash: str,
    scheduled_not_before_us: int,
    scheduled_not_before_monotonic_us: int,
    request_started_at_us: int,
    request_started_monotonic_us: int,
    headers_received_at_us: int | None,
    terminal_at_us: int,
    terminal_monotonic_us: int,
    elapsed_monotonic_us: int,
    http_status: int | None,
    safe_headers: tuple[tuple[str, str], ...],
    body_bytes: bytes,
) -> None:
    if not isinstance(page_request, KlinePageRequestV1):
        raise TransportContractError("attempt_page_request_invalid")
    _strict_int(attempt_ordinal, field="attempt_ordinal", minimum=0)
    for field, value in (
        ("endpoint_contract_hash", endpoint_contract_hash),
        ("resource_limits_hash", resource_limits_hash),
        ("retry_policy_hash", retry_policy_hash),
        ("transport_contract_hash", transport_contract_hash),
    ):
        _digest(value, field=field)
    scheduled = _strict_int(
        scheduled_not_before_us, field="scheduled_not_before_us", minimum=0
    )
    started = _strict_int(
        request_started_at_us, field="request_started_at_us", minimum=0
    )
    scheduled_mono = _strict_int(
        scheduled_not_before_monotonic_us,
        field="scheduled_not_before_monotonic_us",
        minimum=0,
    )
    started_mono = _strict_int(
        request_started_monotonic_us,
        field="request_started_monotonic_us",
        minimum=0,
    )
    terminal = _strict_int(terminal_at_us, field="terminal_at_us", minimum=0)
    terminal_mono = _strict_int(
        terminal_monotonic_us, field="terminal_monotonic_us", minimum=0
    )
    elapsed = _strict_int(
        elapsed_monotonic_us, field="elapsed_monotonic_us", minimum=0
    )
    if started < scheduled or terminal < started:
        raise TransportContractError("attempt_timing_invalid")
    if started_mono < scheduled_mono or terminal_mono < started_mono:
        raise TransportContractError("attempt_monotonic_timing_invalid")
    if elapsed != terminal_mono - started_mono:
        raise TransportContractError("attempt_monotonic_elapsed_mismatch")
    if headers_received_at_us is not None:
        headers_at = _strict_int(
            headers_received_at_us, field="headers_received_at_us", minimum=0
        )
        if headers_at < started or headers_at > terminal:
            raise TransportContractError("attempt_header_timing_invalid")
    if http_status is not None and (
        type(http_status) is not int or not (100 <= http_status <= 599)
    ):
        raise TransportContractError("attempt_http_status_invalid")
    canonical = canonicalize_public_response_headers(safe_headers)
    if canonical != safe_headers:
        raise TransportContractError("attempt_headers_not_canonical")
    if not isinstance(body_bytes, bytes):
        raise TransportContractError("attempt_body_must_be_bytes")


@dataclass(frozen=True)
class CompleteHttpAttemptEvidenceV1:
    page_request: KlinePageRequestV1
    attempt_ordinal: int
    endpoint_contract_hash: str
    resource_limits_hash: str
    retry_policy_hash: str
    transport_contract_hash: str
    scheduled_not_before_us: int
    scheduled_not_before_monotonic_us: int
    request_started_at_us: int
    request_started_monotonic_us: int
    headers_received_at_us: int
    terminal_at_us: int
    terminal_monotonic_us: int
    elapsed_monotonic_us: int
    http_status: int
    safe_headers: tuple[tuple[str, str], ...]
    body_bytes: bytes
    body_complete: bool = True
    outcome: str = "complete"
    safe_error_code: None = None
    contract_version: str = COMPLETE_HTTP_ATTEMPT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != COMPLETE_HTTP_ATTEMPT_VERSION:
            raise TransportContractError("complete_attempt_version_mismatch")
        if self.body_complete is not True or self.outcome != "complete" or self.safe_error_code is not None:
            raise TransportContractError("complete_attempt_outcome_invalid")
        _validate_common_evidence(
            page_request=self.page_request,
            attempt_ordinal=self.attempt_ordinal,
            endpoint_contract_hash=self.endpoint_contract_hash,
            resource_limits_hash=self.resource_limits_hash,
            retry_policy_hash=self.retry_policy_hash,
            transport_contract_hash=self.transport_contract_hash,
            scheduled_not_before_us=self.scheduled_not_before_us,
            scheduled_not_before_monotonic_us=self.scheduled_not_before_monotonic_us,
            request_started_at_us=self.request_started_at_us,
            request_started_monotonic_us=self.request_started_monotonic_us,
            headers_received_at_us=self.headers_received_at_us,
            terminal_at_us=self.terminal_at_us,
            terminal_monotonic_us=self.terminal_monotonic_us,
            elapsed_monotonic_us=self.elapsed_monotonic_us,
            http_status=self.http_status,
            safe_headers=self.safe_headers,
            body_bytes=self.body_bytes,
        )

    @property
    def captured_body_length(self) -> int:
        return len(self.body_bytes)

    @property
    def captured_body_sha256(self) -> str:
        return _sha256_bytes(self.body_bytes)

    def receipt_dict(self) -> dict[str, object]:
        return _attempt_receipt_dict(self)

    @property
    def attempt_receipt_hash(self) -> str:
        return _sha256_payload(self.receipt_dict())


@dataclass(frozen=True)
class IncompleteHttpAttemptEvidenceV1:
    page_request: KlinePageRequestV1
    attempt_ordinal: int
    endpoint_contract_hash: str
    resource_limits_hash: str
    retry_policy_hash: str
    transport_contract_hash: str
    scheduled_not_before_us: int
    scheduled_not_before_monotonic_us: int
    request_started_at_us: int
    request_started_monotonic_us: int
    headers_received_at_us: int | None
    terminal_at_us: int
    terminal_monotonic_us: int
    elapsed_monotonic_us: int
    http_status: int | None
    safe_headers: tuple[tuple[str, str], ...]
    body_bytes: bytes
    outcome: str
    safe_error_code: str
    body_complete: bool = False
    contract_version: str = INCOMPLETE_HTTP_ATTEMPT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != INCOMPLETE_HTTP_ATTEMPT_VERSION:
            raise TransportContractError("incomplete_attempt_version_mismatch")
        if self.body_complete is not False or self.outcome not in {
            "body_limit_exceeded",
            "network_error",
            "timeout",
        }:
            raise TransportContractError("incomplete_attempt_outcome_invalid")
        _safe_code(self.safe_error_code)
        _validate_common_evidence(
            page_request=self.page_request,
            attempt_ordinal=self.attempt_ordinal,
            endpoint_contract_hash=self.endpoint_contract_hash,
            resource_limits_hash=self.resource_limits_hash,
            retry_policy_hash=self.retry_policy_hash,
            transport_contract_hash=self.transport_contract_hash,
            scheduled_not_before_us=self.scheduled_not_before_us,
            scheduled_not_before_monotonic_us=self.scheduled_not_before_monotonic_us,
            request_started_at_us=self.request_started_at_us,
            request_started_monotonic_us=self.request_started_monotonic_us,
            headers_received_at_us=self.headers_received_at_us,
            terminal_at_us=self.terminal_at_us,
            terminal_monotonic_us=self.terminal_monotonic_us,
            elapsed_monotonic_us=self.elapsed_monotonic_us,
            http_status=self.http_status,
            safe_headers=self.safe_headers,
            body_bytes=self.body_bytes,
        )
        if (self.headers_received_at_us is None) != (self.http_status is None):
            raise TransportContractError("incomplete_attempt_http_identity_invalid")
        if self.http_status is None and (self.safe_headers or self.body_bytes):
            raise TransportContractError("preheader_failure_contains_http_evidence")
        if self.outcome == "body_limit_exceeded" and self.http_status is None:
            raise TransportContractError("body_limit_failure_missing_http_response")

    @property
    def captured_body_length(self) -> int:
        return len(self.body_bytes)

    @property
    def captured_body_sha256(self) -> str:
        return _sha256_bytes(self.body_bytes)

    def receipt_dict(self) -> dict[str, object]:
        return _attempt_receipt_dict(self)

    @property
    def attempt_receipt_hash(self) -> str:
        return _sha256_payload(self.receipt_dict())


HttpAttemptEvidenceV1: TypeAlias = (
    CompleteHttpAttemptEvidenceV1 | IncompleteHttpAttemptEvidenceV1
)


def _attempt_receipt_dict(evidence: HttpAttemptEvidenceV1) -> dict[str, object]:
    return {
        "contract_version": evidence.contract_version,
        "page_request": evidence.page_request.as_dict(),
        "page_id": evidence.page_request.page_id,
        "attempt_ordinal": evidence.attempt_ordinal,
        "endpoint_contract_hash": evidence.endpoint_contract_hash,
        "resource_limits_hash": evidence.resource_limits_hash,
        "retry_policy_hash": evidence.retry_policy_hash,
        "transport_contract_hash": evidence.transport_contract_hash,
        "scheduled_not_before_us": evidence.scheduled_not_before_us,
        "scheduled_not_before_monotonic_us": evidence.scheduled_not_before_monotonic_us,
        "request_started_at_us": evidence.request_started_at_us,
        "request_started_monotonic_us": evidence.request_started_monotonic_us,
        "headers_received_at_us": evidence.headers_received_at_us,
        "terminal_at_us": evidence.terminal_at_us,
        "terminal_monotonic_us": evidence.terminal_monotonic_us,
        "elapsed_monotonic_us": evidence.elapsed_monotonic_us,
        "http_status": evidence.http_status,
        "safe_headers": [list(pair) for pair in evidence.safe_headers],
        "body_complete": evidence.body_complete,
        "outcome": evidence.outcome,
        "safe_error_code": evidence.safe_error_code,
        "captured_body_length": evidence.captured_body_length,
        "captured_body_sha256": evidence.captured_body_sha256,
    }


def parse_http_attempt_evidence_v1(
    payload: object,
    *,
    page_request: KlinePageRequestV1,
    body_bytes: bytes,
) -> HttpAttemptEvidenceV1:
    if not isinstance(payload, dict) or set(payload) != _ATTEMPT_RECEIPT_KEYS:
        raise TransportContractError("attempt_receipt_schema_mismatch")
    if payload.get("page_request") != page_request.as_dict() or payload.get(
        "page_id"
    ) != page_request.page_id:
        raise TransportContractError("attempt_receipt_page_mismatch")
    if not isinstance(body_bytes, bytes):
        raise TransportContractError("attempt_body_must_be_bytes")
    if payload.get("captured_body_length") != len(body_bytes) or payload.get(
        "captured_body_sha256"
    ) != _sha256_bytes(body_bytes):
        raise TransportContractError("attempt_receipt_body_mismatch")
    headers = payload.get("safe_headers")
    if not isinstance(headers, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(item, str) for item in pair)
        for pair in headers
    ):
        raise TransportContractError("attempt_receipt_headers_invalid")
    common = {
        "page_request": page_request,
        "attempt_ordinal": payload["attempt_ordinal"],
        "endpoint_contract_hash": payload["endpoint_contract_hash"],
        "resource_limits_hash": payload["resource_limits_hash"],
        "retry_policy_hash": payload["retry_policy_hash"],
        "transport_contract_hash": payload["transport_contract_hash"],
        "scheduled_not_before_us": payload["scheduled_not_before_us"],
        "scheduled_not_before_monotonic_us": payload[
            "scheduled_not_before_monotonic_us"
        ],
        "request_started_at_us": payload["request_started_at_us"],
        "request_started_monotonic_us": payload[
            "request_started_monotonic_us"
        ],
        "headers_received_at_us": payload["headers_received_at_us"],
        "terminal_at_us": payload["terminal_at_us"],
        "terminal_monotonic_us": payload["terminal_monotonic_us"],
        "elapsed_monotonic_us": payload["elapsed_monotonic_us"],
        "http_status": payload["http_status"],
        "safe_headers": tuple((pair[0], pair[1]) for pair in headers),
        "body_bytes": body_bytes,
        "body_complete": payload["body_complete"],
        "outcome": payload["outcome"],
        "safe_error_code": payload["safe_error_code"],
        "contract_version": payload["contract_version"],
    }
    if payload["contract_version"] == COMPLETE_HTTP_ATTEMPT_VERSION:
        return CompleteHttpAttemptEvidenceV1(**common)
    if payload["contract_version"] == INCOMPLETE_HTTP_ATTEMPT_VERSION:
        return IncompleteHttpAttemptEvidenceV1(**common)
    raise TransportContractError("attempt_receipt_version_mismatch")


def _is_retryable(evidence: HttpAttemptEvidenceV1) -> bool:
    if isinstance(evidence, CompleteHttpAttemptEvidenceV1):
        return evidence.http_status in {408, 425, 429} or evidence.http_status >= 500
    return evidence.outcome in {"network_error", "timeout"}


class MexcFuturesRawTransportV1:
    """Deterministic adapter over an injected streaming executor."""

    def __init__(
        self,
        *,
        endpoint: MexcFuturesEndpointContractV1,
        resource_limits: HistoryResourceLimitsV1,
        retry_policy: HistoryRetryPolicyV1,
        executor: StreamingHttpExecutor,
        clock: EvidenceClock,
    ):
        if not isinstance(endpoint, MexcFuturesEndpointContractV1):
            raise TransportContractError("transport_endpoint_required")
        if not isinstance(resource_limits, HistoryResourceLimitsV1):
            raise TransportContractError("transport_resource_limits_required")
        if not isinstance(retry_policy, HistoryRetryPolicyV1):
            raise TransportContractError("transport_retry_policy_required")
        if executor is None or not callable(getattr(executor, "open", None)):
            raise TransportContractError("streaming_executor_required")
        for method in ("epoch_us", "monotonic_us", "sleep_us"):
            if clock is None or not callable(getattr(clock, method, None)):
                raise TransportContractError("evidence_clock_required")
        self.endpoint = endpoint
        self.resource_limits = resource_limits
        self.retry_policy = retry_policy
        self.executor = executor
        self.clock = clock
        self._lock = threading.RLock()
        self._collection_started_monotonic_us: int | None = None
        self._last_request_started_epoch_us: int | None = None
        self._last_request_started_monotonic_us: int | None = None
        self._last_terminal_epoch_us: int | None = None
        self._last_terminal_monotonic_us: int | None = None
        self._total_sleep_us = 0
        self._total_attempts = 0
        self._total_raw_body_bytes = 0
        self._observed_epoch_us: int | None = None
        self._observed_monotonic_us: int | None = None
        self._range_request_id: str | None = None
        self._started_attempt_keys: set[tuple[str, int]] = set()

    @property
    def transport_contract_hash(self) -> str:
        return mexc_futures_transport_contract_hash()

    @property
    def endpoint_contract_hash(self) -> str:
        return self.endpoint.contract_hash

    @property
    def resource_limits_hash(self) -> str:
        return self.resource_limits.contract_hash

    @property
    def retry_policy_hash(self) -> str:
        return self.retry_policy.contract_hash

    def _now(self) -> tuple[int, int]:
        epoch = self.clock.epoch_us()
        monotonic = self.clock.monotonic_us()
        if type(epoch) is not int or epoch < 0 or type(monotonic) is not int or monotonic < 0:
            raise TransportContractError("evidence_clock_must_return_integer_microseconds")
        if self._observed_epoch_us is not None and epoch < self._observed_epoch_us:
            raise TransportContractError("evidence_clock_epoch_regressed")
        if (
            self._observed_monotonic_us is not None
            and monotonic < self._observed_monotonic_us
        ):
            raise TransportContractError("evidence_clock_monotonic_regressed")
        self._observed_epoch_us = epoch
        self._observed_monotonic_us = monotonic
        return epoch, monotonic

    def _incomplete(
        self,
        *,
        page_request: KlinePageRequestV1,
        attempt_ordinal: int,
        scheduled_not_before_us: int,
        scheduled_not_before_monotonic_us: int,
        started_epoch_us: int,
        started_monotonic_us: int,
        headers_received_at_us: int | None,
        http_status: int | None,
        safe_headers: tuple[tuple[str, str], ...],
        body: bytes,
        outcome: str,
        safe_error_code: str,
    ) -> IncompleteHttpAttemptEvidenceV1:
        terminal_epoch, terminal_mono = self._now()
        evidence = IncompleteHttpAttemptEvidenceV1(
            page_request=page_request,
            attempt_ordinal=attempt_ordinal,
            endpoint_contract_hash=self.endpoint.contract_hash,
            resource_limits_hash=self.resource_limits.contract_hash,
            retry_policy_hash=self.retry_policy.contract_hash,
            transport_contract_hash=self.transport_contract_hash,
            scheduled_not_before_us=scheduled_not_before_us,
            scheduled_not_before_monotonic_us=scheduled_not_before_monotonic_us,
            request_started_at_us=started_epoch_us,
            request_started_monotonic_us=started_monotonic_us,
            headers_received_at_us=headers_received_at_us,
            terminal_at_us=terminal_epoch,
            terminal_monotonic_us=terminal_mono,
            elapsed_monotonic_us=terminal_mono - started_monotonic_us,
            http_status=http_status,
            safe_headers=safe_headers,
            body_bytes=body,
            outcome=outcome,
            safe_error_code=safe_error_code,
        )
        self._record_terminal(evidence)
        return evidence

    def _record_terminal(self, evidence: HttpAttemptEvidenceV1) -> None:
        self._last_terminal_epoch_us = evidence.terminal_at_us
        self._last_terminal_monotonic_us = evidence.terminal_monotonic_us
        self._total_raw_body_bytes += evidence.captured_body_length

    @staticmethod
    def _close_response(
        response: StreamingHttpResponse,
    ) -> tuple[str, str] | None:
        """Close once and convert every close failure into safe attempt state."""

        try:
            response.close()
        except StreamingExecutorTimeoutError as exc:
            return "timeout", exc.safe_error_code
        except StreamingExecutorNetworkError as exc:
            return "network_error", exc.safe_error_code
        except Exception:
            return "network_error", "executor_close_error"
        return None

    def fetch_page(
        self,
        request: KlinePageRequestV1,
        *,
        attempt_ordinal: int,
        prior_attempt: HttpAttemptEvidenceV1 | None = None,
    ) -> HttpAttemptEvidenceV1:
        with self._lock:
            if not isinstance(request, KlinePageRequestV1):
                raise TransportContractError("transport_page_request_invalid")
            ordinal = _strict_int(
                attempt_ordinal, field="attempt_ordinal", minimum=0
            )
            prepared = self.endpoint.prepare(request)
            if (
                self._range_request_id is not None
                and request.range_request_id != self._range_request_id
            ):
                raise TransportContractError("transport_range_request_mismatch")
            attempt_key = (request.page_id, ordinal)
            if attempt_key in self._started_attempt_keys:
                raise TransportContractError("transport_attempt_coordinate_reused")
            if ordinal >= self.resource_limits.max_attempts_per_page:
                raise ResourceBudgetExceededError("attempt_ordinal_exceeds_page_budget")
            if ordinal == 0 and prior_attempt is not None:
                raise TransportContractError("first_attempt_must_not_have_prior")
            if ordinal > 0:
                if (
                    not isinstance(
                        prior_attempt,
                        (CompleteHttpAttemptEvidenceV1, IncompleteHttpAttemptEvidenceV1),
                    )
                    or prior_attempt.page_request != request
                    or prior_attempt.attempt_ordinal != ordinal - 1
                ):
                    raise TransportContractError("retry_prior_attempt_mismatch")
                if not _is_retryable(prior_attempt):
                    raise TransportContractError("prior_attempt_is_not_retryable")
                if (
                    prior_attempt.endpoint_contract_hash != self.endpoint.contract_hash
                    or prior_attempt.resource_limits_hash
                    != self.resource_limits.contract_hash
                    or prior_attempt.retry_policy_hash != self.retry_policy.contract_hash
                    or prior_attempt.transport_contract_hash
                    != self.transport_contract_hash
                ):
                    raise TransportContractError("prior_attempt_contract_mismatch")
                if (
                    self._last_terminal_epoch_us != prior_attempt.terminal_at_us
                    or self._last_terminal_monotonic_us
                    != prior_attempt.terminal_monotonic_us
                ):
                    raise TransportContractError("prior_attempt_clock_domain_mismatch")
            if self._total_attempts >= self.resource_limits.max_total_attempts:
                raise ResourceBudgetExceededError("total_attempt_budget_exceeded")
            if self._total_raw_body_bytes >= self.resource_limits.max_total_raw_body_bytes:
                raise ResourceBudgetExceededError("total_raw_body_budget_exceeded")
            if self._range_request_id is None:
                self._range_request_id = request.range_request_id
            now_epoch, now_mono = self._now()
            if self._collection_started_monotonic_us is None:
                self._collection_started_monotonic_us = now_mono
            if now_mono - self._collection_started_monotonic_us > self.resource_limits.max_collection_runtime_us:
                raise ResourceBudgetExceededError("collection_runtime_budget_exceeded")
            if self._last_terminal_epoch_us is not None and now_epoch < self._last_terminal_epoch_us:
                raise TransportContractError("evidence_clock_epoch_regressed")
            scheduled = now_epoch
            scheduled_mono = now_mono
            if self._last_request_started_epoch_us is not None:
                scheduled = max(
                    scheduled,
                    self._last_request_started_epoch_us
                    + self.retry_policy.min_request_spacing_us,
                )
            if self._last_request_started_monotonic_us is not None:
                scheduled_mono = max(
                    scheduled_mono,
                    self._last_request_started_monotonic_us
                    + self.retry_policy.min_request_spacing_us,
                )
            if prior_attempt is not None:
                retry_after = (
                    retry_after_delay_us(
                        prior_attempt.safe_headers,
                        received_at_us=prior_attempt.terminal_at_us,
                        policy=self.retry_policy,
                    )
                    if isinstance(prior_attempt, CompleteHttpAttemptEvidenceV1)
                    else 0
                )
                retry_delay = max(
                    self.retry_policy.backoff_before_attempt_us(ordinal),
                    retry_after,
                )
                scheduled = max(scheduled, prior_attempt.terminal_at_us + retry_delay)
                scheduled_mono = max(
                    scheduled_mono,
                    prior_attempt.terminal_monotonic_us + retry_delay,
                )
            sleep_us = scheduled_mono - now_mono
            if self._total_sleep_us + sleep_us > self.retry_policy.max_total_sleep_us:
                raise ResourceBudgetExceededError("total_retry_sleep_budget_exceeded")
            if (
                now_mono
                - self._collection_started_monotonic_us
                + sleep_us
                > self.resource_limits.max_collection_runtime_us
            ):
                raise ResourceBudgetExceededError("sleep_crosses_collection_deadline")
            if sleep_us:
                self.clock.sleep_us(sleep_us)
            started_epoch, started_mono = self._now()
            if started_epoch < scheduled:
                raise TransportContractError("evidence_clock_sleep_undershot")
            if started_mono < scheduled_mono:
                raise TransportContractError("evidence_clock_monotonic_sleep_undershot")
            observed_sleep_us = started_mono - now_mono
            if self._total_sleep_us + observed_sleep_us > self.retry_policy.max_total_sleep_us:
                raise ResourceBudgetExceededError("observed_retry_sleep_budget_exceeded")
            if (
                started_mono - self._collection_started_monotonic_us
                > self.resource_limits.max_collection_runtime_us
            ):
                raise ResourceBudgetExceededError("observed_sleep_crosses_collection_deadline")
            self._total_sleep_us += observed_sleep_us
            self._last_request_started_epoch_us = started_epoch
            self._last_request_started_monotonic_us = started_mono
            self._started_attempt_keys.add(attempt_key)
            self._total_attempts += 1
            try:
                response = self.executor.open(
                    prepared,
                    connect_timeout_us=min(
                        5_000_000, self.resource_limits.max_attempt_runtime_us
                    ),
                    read_timeout_us=min(
                        10_000_000, self.resource_limits.max_attempt_runtime_us
                    ),
                )
            except StreamingExecutorTimeoutError as exc:
                return self._incomplete(
                    page_request=request,
                    attempt_ordinal=ordinal,
                    scheduled_not_before_us=scheduled,
                    scheduled_not_before_monotonic_us=scheduled_mono,
                    started_epoch_us=started_epoch,
                    started_monotonic_us=started_mono,
                    headers_received_at_us=None,
                    http_status=None,
                    safe_headers=(),
                    body=b"",
                    outcome="timeout",
                    safe_error_code=exc.safe_error_code,
                )
            except StreamingExecutorNetworkError as exc:
                return self._incomplete(
                    page_request=request,
                    attempt_ordinal=ordinal,
                    scheduled_not_before_us=scheduled,
                    scheduled_not_before_monotonic_us=scheduled_mono,
                    started_epoch_us=started_epoch,
                    started_monotonic_us=started_mono,
                    headers_received_at_us=None,
                    http_status=None,
                    safe_headers=(),
                    body=b"",
                    outcome="network_error",
                    safe_error_code=exc.safe_error_code,
                )
            except Exception:
                return self._incomplete(
                    page_request=request,
                    attempt_ordinal=ordinal,
                    scheduled_not_before_us=scheduled,
                    scheduled_not_before_monotonic_us=scheduled_mono,
                    started_epoch_us=started_epoch,
                    started_monotonic_us=started_mono,
                    headers_received_at_us=None,
                    http_status=None,
                    safe_headers=(),
                    body=b"",
                    outcome="network_error",
                    safe_error_code="executor_open_error",
                )
            if response is None:
                return self._incomplete(
                    page_request=request,
                    attempt_ordinal=ordinal,
                    scheduled_not_before_us=scheduled,
                    scheduled_not_before_monotonic_us=scheduled_mono,
                    started_epoch_us=started_epoch,
                    started_monotonic_us=started_mono,
                    headers_received_at_us=None,
                    http_status=None,
                    safe_headers=(),
                    body=b"",
                    outcome="network_error",
                    safe_error_code="executor_response_missing",
                )
            response_close = getattr(response, "close", None)
            response_iter = getattr(response, "iter_body", None)
            if not callable(response_close):
                return self._incomplete(
                    page_request=request,
                    attempt_ordinal=ordinal,
                    scheduled_not_before_us=scheduled,
                    scheduled_not_before_monotonic_us=scheduled_mono,
                    started_epoch_us=started_epoch,
                    started_monotonic_us=started_mono,
                    headers_received_at_us=None,
                    http_status=None,
                    safe_headers=(),
                    body=b"",
                    outcome="network_error",
                    safe_error_code="executor_response_close_missing",
                )
            if not callable(response_iter):
                close_failure = self._close_response(response)
                outcome, code = close_failure or (
                    "network_error",
                    "executor_response_body_iterator_missing",
                )
                return self._incomplete(
                    page_request=request,
                    attempt_ordinal=ordinal,
                    scheduled_not_before_us=scheduled,
                    scheduled_not_before_monotonic_us=scheduled_mono,
                    started_epoch_us=started_epoch,
                    started_monotonic_us=started_mono,
                    headers_received_at_us=None,
                    http_status=None,
                    safe_headers=(),
                    body=b"",
                    outcome=outcome,
                    safe_error_code=code,
                )
            status: int | None = None
            safe_headers: tuple[tuple[str, str], ...] = ()
            header_failure: tuple[str, str] | None = None
            try:
                status = response.http_status
                if type(status) is not int or not (100 <= status <= 599):
                    raise TransportContractError("streaming_response_status_invalid")
                safe_headers = canonicalize_public_response_headers(response.headers)
            except StreamingExecutorTimeoutError as exc:
                header_failure = ("timeout", exc.safe_error_code)
            except StreamingExecutorNetworkError as exc:
                header_failure = ("network_error", exc.safe_error_code)
            except TransportContractError:
                header_failure = ("network_error", "response_header_contract_error")
            except Exception:
                header_failure = ("network_error", "response_header_executor_error")
            headers_epoch, headers_mono = self._now()
            if header_failure is not None:
                close_failure = self._close_response(response)
                outcome, code = close_failure or header_failure
                _post_close_epoch, post_close_mono = self._now()
                if (
                    post_close_mono - started_mono
                    > self.resource_limits.max_attempt_runtime_us
                ):
                    outcome, code = (
                        "timeout",
                        f"attempt_runtime_exceeded_after_{outcome}",
                    )
                elif (
                    post_close_mono - self._collection_started_monotonic_us
                    > self.resource_limits.max_collection_runtime_us
                ):
                    outcome, code = (
                        "timeout",
                        f"collection_runtime_exceeded_after_{outcome}",
                    )
                valid_status = status if type(status) is int and 100 <= status <= 599 else None
                return self._incomplete(
                    page_request=request,
                    attempt_ordinal=ordinal,
                    scheduled_not_before_us=scheduled,
                    scheduled_not_before_monotonic_us=scheduled_mono,
                    started_epoch_us=started_epoch,
                    started_monotonic_us=started_mono,
                    headers_received_at_us=(
                        headers_epoch if valid_status is not None else None
                    ),
                    http_status=valid_status,
                    safe_headers=safe_headers if valid_status is not None else (),
                    body=b"",
                    outcome=outcome,
                    safe_error_code=code,
                )
            header_attempt_expired = (
                headers_mono - started_mono
                > self.resource_limits.max_attempt_runtime_us
            )
            header_collection_expired = (
                headers_mono - self._collection_started_monotonic_us
                > self.resource_limits.max_collection_runtime_us
            )
            if header_attempt_expired or header_collection_expired:
                close_failure = self._close_response(response)
                outcome, code = close_failure or (
                    "timeout",
                    (
                        "attempt_runtime_exceeded_before_body"
                        if header_attempt_expired
                        else "collection_runtime_exceeded_before_body"
                    ),
                )
                return self._incomplete(
                    page_request=request,
                    attempt_ordinal=ordinal,
                    scheduled_not_before_us=scheduled,
                    scheduled_not_before_monotonic_us=scheduled_mono,
                    started_epoch_us=started_epoch,
                    started_monotonic_us=started_mono,
                    headers_received_at_us=headers_epoch,
                    http_status=status,
                    safe_headers=safe_headers,
                    body=b"",
                    outcome=outcome,
                    safe_error_code=code,
                )
            body = bytearray()
            per_attempt_cap = self.resource_limits.max_raw_body_bytes_per_attempt
            total_remaining = (
                self.resource_limits.max_total_raw_body_bytes
                - self._total_raw_body_bytes
            )
            capture_cap = min(per_attempt_cap, total_remaining)
            pending_failure: tuple[str, str] | None = None
            try:
                for chunk in response.iter_body(64 * 1024):
                    if not isinstance(chunk, bytes):
                        raise TransportContractError("streaming_response_chunk_invalid")
                    if not chunk:
                        continue
                    remaining_to_marker = capture_cap + 1 - len(body)
                    body.extend(chunk[:remaining_to_marker])
                    _epoch, current_mono = self._now()
                    attempt_expired = (
                        current_mono - started_mono
                        > self.resource_limits.max_attempt_runtime_us
                    )
                    collection_expired = (
                        current_mono - self._collection_started_monotonic_us
                        > self.resource_limits.max_collection_runtime_us
                    )
                    if attempt_expired or collection_expired:
                        pending_failure = (
                            "timeout",
                            (
                                "attempt_runtime_exceeded"
                                if attempt_expired
                                else "collection_runtime_exceeded"
                            ),
                        )
                        break
                    if len(body) > capture_cap:
                        code = (
                            "attempt_body_limit_exceeded"
                            if capture_cap == per_attempt_cap
                            else "total_raw_body_limit_exceeded"
                        )
                        pending_failure = ("body_limit_exceeded", code)
                        break
            except StreamingExecutorTimeoutError as exc:
                pending_failure = ("timeout", exc.safe_error_code)
            except StreamingExecutorNetworkError as exc:
                pending_failure = ("network_error", exc.safe_error_code)
            except Exception:
                pending_failure = ("network_error", "executor_body_error")
            close_failure = self._close_response(response)
            if close_failure is not None:
                if (
                    pending_failure is not None
                    and pending_failure[0] == "body_limit_exceeded"
                ):
                    pending_failure = (
                        "body_limit_exceeded",
                        f"{pending_failure[1]}_and_close_failure",
                    )
                else:
                    pending_failure = close_failure
            post_close_epoch, post_close_mono = self._now()
            post_close_attempt_expired = (
                post_close_mono - started_mono
                > self.resource_limits.max_attempt_runtime_us
            )
            post_close_collection_expired = (
                post_close_mono - self._collection_started_monotonic_us
                > self.resource_limits.max_collection_runtime_us
            )
            if post_close_attempt_expired or post_close_collection_expired:
                prior_outcome = (
                    pending_failure[0] if pending_failure is not None else "complete"
                )
                prior_error = pending_failure[1] if pending_failure is not None else ""
                if prior_outcome == "body_limit_exceeded":
                    suffix = (
                        "attempt_runtime_exceeded"
                        if post_close_attempt_expired
                        else "collection_runtime_exceeded"
                    )
                    if not prior_error.endswith(suffix):
                        pending_failure = (
                            "body_limit_exceeded",
                            f"{prior_error}_and_{suffix}",
                        )
                elif post_close_attempt_expired and not prior_error.startswith(
                    "attempt_runtime_exceeded"
                ):
                    pending_failure = (
                        "timeout",
                        f"attempt_runtime_exceeded_after_{prior_outcome}",
                    )
                elif post_close_collection_expired and not prior_error.startswith(
                    "collection_runtime_exceeded"
                ):
                    pending_failure = (
                        "timeout",
                        f"collection_runtime_exceeded_after_{prior_outcome}",
                    )
            if pending_failure is not None:
                outcome, code = pending_failure
                return self._incomplete(
                    page_request=request,
                    attempt_ordinal=ordinal,
                    scheduled_not_before_us=scheduled,
                    scheduled_not_before_monotonic_us=scheduled_mono,
                    started_epoch_us=started_epoch,
                    started_monotonic_us=started_mono,
                    headers_received_at_us=headers_epoch,
                    http_status=status,
                    safe_headers=safe_headers,
                    body=bytes(body[:capture_cap]),
                    outcome=outcome,
                    safe_error_code=code,
                )
            content_length = _header_value(safe_headers, "content-length")
            if content_length is not None and (
                content_length.startswith("invalid.")
                or int(content_length) != len(body)
            ):
                return self._incomplete(
                    page_request=request,
                    attempt_ordinal=ordinal,
                    scheduled_not_before_us=scheduled,
                    scheduled_not_before_monotonic_us=scheduled_mono,
                    started_epoch_us=started_epoch,
                    started_monotonic_us=started_mono,
                    headers_received_at_us=headers_epoch,
                    http_status=status,
                    safe_headers=safe_headers,
                    body=bytes(body),
                    outcome="network_error",
                    safe_error_code="content_length_mismatch",
                )
            terminal_epoch, terminal_mono = post_close_epoch, post_close_mono
            eof_attempt_expired = (
                terminal_mono - started_mono
                > self.resource_limits.max_attempt_runtime_us
            )
            eof_collection_expired = (
                terminal_mono - self._collection_started_monotonic_us
                > self.resource_limits.max_collection_runtime_us
            )
            if eof_attempt_expired or eof_collection_expired:
                return self._incomplete(
                    page_request=request,
                    attempt_ordinal=ordinal,
                    scheduled_not_before_us=scheduled,
                    scheduled_not_before_monotonic_us=scheduled_mono,
                    started_epoch_us=started_epoch,
                    started_monotonic_us=started_mono,
                    headers_received_at_us=headers_epoch,
                    http_status=status,
                    safe_headers=safe_headers,
                    body=bytes(body),
                    outcome="timeout",
                    safe_error_code=(
                        "attempt_runtime_exceeded_at_eof"
                        if eof_attempt_expired
                        else "collection_runtime_exceeded_at_eof"
                    ),
                )
            evidence = CompleteHttpAttemptEvidenceV1(
                page_request=request,
                attempt_ordinal=ordinal,
                endpoint_contract_hash=self.endpoint.contract_hash,
                resource_limits_hash=self.resource_limits.contract_hash,
                retry_policy_hash=self.retry_policy.contract_hash,
                transport_contract_hash=self.transport_contract_hash,
                scheduled_not_before_us=scheduled,
                scheduled_not_before_monotonic_us=scheduled_mono,
                request_started_at_us=started_epoch,
                request_started_monotonic_us=started_mono,
                headers_received_at_us=headers_epoch,
                terminal_at_us=terminal_epoch,
                terminal_monotonic_us=terminal_mono,
                elapsed_monotonic_us=terminal_mono - started_mono,
                http_status=status,
                safe_headers=safe_headers,
                body_bytes=bytes(body),
            )
            self._record_terminal(evidence)
            return evidence


_TRANSPORT_CONTRACT_SCHEMA = {
    "contract_version": MEXC_FUTURES_TRANSPORT_CONTRACT_VERSION,
    "endpoint_contract_version": MEXC_FUTURES_ENDPOINT_CONTRACT_VERSION,
    "resource_limits_version": HISTORY_RESOURCE_LIMITS_VERSION,
    "retry_policy_version": HISTORY_RETRY_POLICY_VERSION,
    "complete_attempt_version": COMPLETE_HTTP_ATTEMPT_VERSION,
    "incomplete_attempt_version": INCOMPLETE_HTTP_ATTEMPT_VERSION,
    "page_shape": "strict_history.KlinePageRequestV1",
    "clock": (
        "exact_integer_epoch_and_monotonic_microseconds_with_"
        "scheduled_start_terminal_monotonic_receipt_fields"
    ),
    "network_executor": "injected_streaming_only_no_default_or_real_executor",
    "instance_scope": (
        "one_range_acquisition_per_instance_latched_range_request_id_"
        "no_duplicate_page_attempt_coordinate"
    ),
    "started_attempt_executor_failures": (
        "typed_incomplete_evidence_for_open_error_missing_or_invalid_response"
    ),
    "request": {
        "method": "GET",
        "credentials": False,
        "redirects": False,
        "trust_env": False,
        "tls_verify": True,
        "body": None,
    },
    "body": {
        "streaming": True,
        "oversize_probe": "read_at_most_declared_cap_plus_one_byte",
        "retained_partial_prefix": "at_most_declared_cap_bytes",
        "observed_extra_byte": "body_limit_exceeded_outcome_not_persisted",
        "failure_priority": (
            "body_limit_exceeded_remains_nonretryable_when_close_or_runtime_"
            "also_fails_secondary_failure_is_encoded_in_safe_error_code"
        ),
        "complete_only_after_eof": True,
        "partial_bytes_are_content_addressed": True,
        "close": (
            "always_before_terminal_sample_close_latency_in_deadlines_"
            "close_failure_is_typed_incomplete_evidence"
        ),
    },
    "retry": {
        "statuses": [408, 425, 429, "5xx"],
        "delay": (
            "max_global_spacing_exponential_backoff_retry_after_"
            "scheduled_and_enforced_on_monotonic_clock"
        ),
        "sleep_accounting": "observed_monotonic_delta_not_requested_duration",
        "jitter": False,
        "retry_after_over_cap": "fail_closed",
    },
    "headers": {
        "allowlist": sorted(_PUBLIC_SAFE_HEADER_NAMES),
        "unknown_and_credentials": "discarded",
        "date": "nonnegative_unix_epoch_seconds",
        "etag": "strength_plus_base64url_opaque_tag",
        "retry_after": "delay_seconds_or_unix_epoch_seconds",
        "invalid_public_value": "invalid_dot_sha256",
        "hard_caps": {
            "count": HARD_MAX_RESPONSE_HEADER_COUNT,
            "name_chars": HARD_MAX_RESPONSE_HEADER_NAME_CHARS,
            "value_chars": HARD_MAX_RESPONSE_HEADER_VALUE_CHARS,
            "aggregate_chars": HARD_MAX_RESPONSE_HEADER_AGGREGATE_CHARS,
        },
    },
    "hard_caps": {
        "pages": HARD_MAX_PAGES,
        "rows": HARD_MAX_ROWS,
        "attempts_per_page": HARD_MAX_ATTEMPTS_PER_PAGE,
        "total_attempts": HARD_MAX_TOTAL_ATTEMPTS,
        "raw_body_bytes_per_attempt": HARD_MAX_RAW_BODY_BYTES_PER_ATTEMPT,
        "total_raw_body_bytes": HARD_MAX_TOTAL_RAW_BODY_BYTES,
        "logical_storage_bytes": HARD_MAX_LOGICAL_STORAGE_BYTES,
        "collection_runtime_us": HARD_MAX_COLLECTION_RUNTIME_US,
        "attempt_runtime_us": HARD_MAX_ATTEMPT_RUNTIME_US,
    },
    "attempt_receipt_keys": sorted(_ATTEMPT_RECEIPT_KEYS),
    "incomplete_outcomes": [
        "body_limit_exceeded",
        "network_error",
        "timeout",
    ],
}

_PINNED_TRANSPORT_CONTRACT_HASH = (
    "7d3bd40c6753e7bda2f1904ce2ffa2ff55770ecce9ba6d5614d2b30ae0664d22"
)


def mexc_futures_transport_contract_hash() -> str:
    digest = _sha256_payload(_TRANSPORT_CONTRACT_SCHEMA)
    if _PINNED_TRANSPORT_CONTRACT_HASH and digest != _PINNED_TRANSPORT_CONTRACT_HASH:
        raise TransportContractError("mexc_futures_transport_contract_hash_drift")
    return digest


__all__ = [
    "COMPLETE_HTTP_ATTEMPT_VERSION",
    "CompleteHttpAttemptEvidenceV1",
    "EndpointContractError",
    "EvidenceClock",
    "HISTORY_RESOURCE_LIMITS_VERSION",
    "HISTORY_RETRY_POLICY_VERSION",
    "HistoryResourceLimitsV1",
    "HistoryRetryPolicyV1",
    "HttpAttemptEvidenceV1",
    "INCOMPLETE_HTTP_ATTEMPT_VERSION",
    "IncompleteHttpAttemptEvidenceV1",
    "MEXC_FUTURES_ENDPOINT_CONTRACT_VERSION",
    "MEXC_FUTURES_TRANSPORT_CONTRACT_VERSION",
    "MexcFuturesEndpointContractV1",
    "MexcFuturesRawTransportV1",
    "PreparedPublicRequestV1",
    "ResourceBudgetExceededError",
    "ResourceLimitContractError",
    "RetryAfterContractError",
    "RetryPolicyContractError",
    "StreamingExecutorNetworkError",
    "StreamingExecutorTimeoutError",
    "StreamingHttpExecutor",
    "StreamingHttpResponse",
    "TransportContractError",
    "candidate_endpoint_fixture_path",
    "candidate_history_resource_limits_v1",
    "candidate_history_retry_policy_v1",
    "canonicalize_public_response_headers",
    "load_mexc_futures_endpoint_contract_v1",
    "mexc_futures_transport_contract_hash",
    "parse_http_attempt_evidence_v1",
    "retry_after_delay_us",
]
