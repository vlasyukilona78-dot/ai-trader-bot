"""Offline-only evidence contract for the frozen MEXC endpoint announcement.

This module deliberately contains no HTTP client and no default executor.  Its
only current provenance mode is ``reviewed_fake_fixture_only``.  Consequently,
objects produced by this v1 module are useful for deterministic storage and
parser testing, but can never authorize U5, a live probe, acquisition, or a
terminal pilot receipt.

The future network adapter must construct the attempt receipt from transport
observations.  This module only validates and durably stores that receipt and
reparses the exact observed entity bytes; it never accepts a caller supplied
``official_document_supports_candidate`` boolean.

Filesystem checks are defense in depth for a static namespace or cooperating
writers.  The ordinary pathname API used here is not an atomic directory
snapshot and makes no TOCTOU claim against a concurrently hostile process.  A
future real-U5 path therefore needs an accepted stronger, handle-relative store
or an equivalent common-writer-lock boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from typing import Any, NoReturn
from urllib.parse import urlsplit

from trading.market_data import mexc_pilot_output_layout as _pilot_output_layout


OFFICIAL_REFERENCE_REQUEST_VERSION = "mexc_endpoint_official_reference_request_v1"
OFFICIAL_REFERENCE_HTTP_ATTEMPT_VERSION = (
    "mexc_endpoint_official_reference_http_attempt_v1"
)
OFFICIAL_DOCUMENT_SPAN_CLAIM_VERSION = "mexc_endpoint_official_document_span_claim_v1"
OFFICIAL_DOCUMENT_EVIDENCE_VERSION = "mexc_endpoint_official_document_evidence_v1"
OFFICIAL_DOCUMENT_READER_VERSION = "mexc_endpoint_official_document_reader_v1"
OFFICIAL_EVIDENCE_BUNDLE_FILE_VERSION = "mexc_endpoint_official_evidence_bundle_file_v1"
OFFICIAL_EVIDENCE_BUNDLE_VERSION = "mexc_endpoint_official_evidence_bundle_v1"
OFFICIAL_EVIDENCE_STORE_VERSION = "mexc_endpoint_official_evidence_store_v1"
OFFICIAL_EVIDENCE_COMPATIBILITY_VERSION = (
    "mexc_endpoint_official_evidence_compatibility_v1"
)
OFFICIAL_EVIDENCE_STRICT_ADAPTER_VERSION = (
    "mexc_endpoint_official_evidence_strict_adapter_v1"
)

OFFICIAL_REFERENCE_URL = (
    "https://www.mexc.com/announcements/article/"
    "futures-api-access-domain-update-17827791532974"
)
OFFICIAL_REFERENCE_SCHEME = "https"
OFFICIAL_REFERENCE_HOST = "www.mexc.com"
OFFICIAL_REFERENCE_PORT = 443
OFFICIAL_REFERENCE_PATH = (
    "/announcements/article/futures-api-access-domain-update-17827791532974"
)
OFFICIAL_REFERENCE_ID = "17827791532974"

CANDIDATE_CONTRACT_VERSION = "mexc_futures_kline_endpoint_candidate_v1"
CANDIDATE_CONTRACT_HASH = (
    "54f57d755cd679eb92444d48b38013621caad37067125e80cf7c5e45fe2ab220"
)
CANDIDATE_METHOD = "GET"
CANDIDATE_URL_TEMPLATE = (
    "https://api.mexc.com/api/v1/contract/kline/{venue_symbol}"
)
CANDIDATE_PATH_TEMPLATE = "/api/v1/contract/kline/{venue_symbol}"
CANDIDATE_QUERY_ORDER = ("interval", "start", "end")
LEGACY_FUTURES_HOST = "contract.mexc.com"
CANDIDATE_FUTURES_HOST = "api.mexc.com"

REVIEWED_FAKE_FIXTURE_ONLY = "reviewed_fake_fixture_only"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
OFFICIAL_RECEIPT_MAX_BYTES = 64 * 1024
OFFICIAL_EVIDENCE_MAX_BYTES = 256 * 1024
OFFICIAL_IO_CHUNK_BYTES = 64 * 1024
OFFICIAL_IO_DEADLINE_US = 30 * 1_000_000
OFFICIAL_RAW_BODY_HARD_CAP_BYTES = 16 * 1024 * 1024
OFFICIAL_JSON_MAX_DEPTH = 32
OFFICIAL_JSON_MAX_CONTAINER_ITEMS = 4096
OFFICIAL_JSON_MAX_TOTAL_NODES = 20_000
OFFICIAL_JSON_MAX_STRING_BYTES = 8192
OFFICIAL_JSON_MAX_INTEGER_DIGITS = 20
OFFICIAL_STORAGE_CONCURRENCY_BOUNDARY = (
    "static_or_cooperating_writers_only_no_atomic_directory_snapshot_or_toctou_guarantee"
)
PILOT_OUTPUT_LAYOUT_EXPECTED_CONTRACT_VERSION = (
    "mexc_public_qa_pilot_output_layout_v1"
)
PILOT_OUTPUT_LAYOUT_EXPECTED_CONTRACT_HASH = (
    "cb19e6a53d122139ec3a76b4d54c67c04a31da9550db9ca8c186496c6bb8e934"
)
PILOT_OUTPUT_LAYOUT_CONTRACT_VERSION = (
    _pilot_output_layout.PILOT_OUTPUT_LAYOUT_CONTRACT_VERSION
)

_PINNED_CONTRACT_HASH = (
    "421802f03282ea5f61f253607001036e80a1933e1d1ea16449c5ee261889e04d"
)
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9._:-]{0,127}$")
_SAFE_HEADER_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_SAFE_RESPONSE_HEADERS = frozenset(
    {
        "cache-control",
        "content-encoding",
        "content-length",
        "content-type",
        "date",
        "etag",
        "last-modified",
    }
)
_SECRET_RESPONSE_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "proxy-authenticate",
        "proxy-authorization",
        "set-cookie",
        "www-authenticate",
    }
)
_REQUEST_HEADERS = (
    ("accept", "text/html,application/xhtml+xml"),
    ("accept-encoding", "identity"),
    ("user-agent", "koteika-mexc-official-evidence/1.0"),
)
_MIGRATION_STATEMENT = (
    b"Futures API domain migration: contract.mexc.com -> api.mexc.com"
)
_FULL_CANDIDATE_STATEMENT = (
    b"Candidate contract: method=GET; scheme=https; host=api.mexc.com; port=443; "
    b"path=/api/v1/contract/kline/{venue_symbol}; query_encoding=ascii_exact_ordered; "
    b"query_order=interval:{interval},start:{start_open_ts},"
    b"end:{end_open_ts_inclusive}; request_headers=accept:application/json,"
    b"accept-encoding:identity,user-agent:koteika-strict-history/1.0; "
    b"authentication=none; tls_verification=required; redirects=reject"
)
_CLAIM_LITERALS: dict[str, bytes] = {
    "domain_migration_statement_v1": _MIGRATION_STATEMENT,
    "full_candidate_contract_statement_v1": _FULL_CANDIDATE_STATEMENT,
}
_MIGRATION_ROLES = frozenset({"domain_migration_statement_v1"})
_FULL_CANDIDATE_ROLES = frozenset(_CLAIM_LITERALS)
_TERMINAL_PROGRESS_PHASES = (
    "before_tls_validation",
    "tls_validation_failed",
    "tls_validated_before_headers",
    "headers_received_before_body_eof",
    "body_eof",
)
_TERMINAL_PROGRESS_EVENT_PRESENCE = {
    "before_tls_validation": (False, False, False),
    "tls_validation_failed": (False, False, False),
    "tls_validated_before_headers": (True, False, False),
    "headers_received_before_body_eof": (True, True, False),
    "body_eof": (True, True, True),
}
_TERMINAL_BLOCKERS = (
    "official_bundle_namespace_absent_from_frozen_preflight",
    "observed_current_official_body_version_absent",
    "single_migration_announcement_may_not_prove_full_candidate_contract",
    "live_reload_inventory_anchor_overhead_unreserved",
    "official_store_host_clock_not_bound_to_attempt_clock",
    "attempt_parent_receipt_hashes_are_opaque_not_fresh_source_objects",
    "runtime_tls_trust_bindings_are_declarative_not_attested",
    "attempt_and_evidence_clock_samples_are_structural_fake_only",
    "incomplete_or_failure_official_attempt_bundle_unsupported",
    "plan_root_sibling_inventory_delegated_to_future_pinned_output_layout",
    "partial_three_file_publication_is_nonresumable_and_not_transactional",
    "hostile_concurrent_filesystem_toctou_boundary_unaccepted",
    "terminal_endpoint_receipt_publisher_unbound",
)
_SAFE_ERROR_CODES_BY_OUTCOME = {
    "incomplete_transport_error": frozenset(
        {
            "dns_resolution_failed",
            "transport_connect_failed",
            "transport_connection_closed",
            "transport_timeout",
        }
    ),
    "incomplete_tls_error": frozenset(
        {
            "tls_certificate_validation_failed",
            "tls_handshake_failed",
            "tls_policy_rejected",
            "tls_sni_mismatch",
        }
    ),
    "incomplete_http_body_error": frozenset(
        {
            "body_cap_exceeded",
            "content_length_mismatch",
            "http_body_eof_missing",
            "http_body_read_failed",
        }
    ),
    "rejected_protocol": frozenset(
        {
            "content_encoding_not_identity",
            "content_length_invalid",
            "content_type_not_official_html",
            "http_status_not_200",
            "redirect_status_rejected",
        }
    ),
}
_TRANSPORT_PROGRESS_BY_SAFE_ERROR = {
    "dns_resolution_failed": frozenset({"before_tls_validation"}),
    "transport_connect_failed": frozenset({"before_tls_validation"}),
    "transport_connection_closed": frozenset(
        {"before_tls_validation", "tls_validated_before_headers"}
    ),
    "transport_timeout": frozenset(
        {"before_tls_validation", "tls_validated_before_headers"}
    ),
}
_WINDOWS_RESERVED = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)


class MexcOfficialEvidenceError(RuntimeError):
    """Base error for this offline evidence contract."""


class MexcOfficialEvidenceContractError(MexcOfficialEvidenceError):
    pass


class MexcOfficialEvidenceSemanticStop(MexcOfficialEvidenceError):
    pass


class MexcOfficialEvidenceStorageStop(MexcOfficialEvidenceError):
    pass


class MexcOfficialEvidenceBudgetStop(MexcOfficialEvidenceError):
    pass


class MexcOfficialEvidenceTerminalStop(MexcOfficialEvidenceError):
    """Typed STOP returned by the intentionally non-terminal v1 adapter."""

    def __init__(
        self,
        code: str,
        blockers: tuple[str, ...],
        classification: str,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.blockers = blockers
        self.classification = classification


def _validated_pilot_output_layout_dependency() -> tuple[str, str]:
    """Fail closed if the frozen namespace dependency drifts at runtime."""

    version = _pilot_output_layout.PILOT_OUTPUT_LAYOUT_CONTRACT_VERSION
    try:
        contract_hash = _pilot_output_layout.pilot_output_layout_contract_hash()
    except _pilot_output_layout.PilotOutputLayoutContractError as exc:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_pilot_output_layout_dependency_invalid"
        ) from exc
    if (
        type(version) is not str
        or version != PILOT_OUTPUT_LAYOUT_CONTRACT_VERSION
        or version != PILOT_OUTPUT_LAYOUT_EXPECTED_CONTRACT_VERSION
        or type(contract_hash) is not str
        or contract_hash != PILOT_OUTPUT_LAYOUT_EXPECTED_CONTRACT_HASH
    ):
        raise MexcOfficialEvidenceContractError(
            "official_evidence_pilot_output_layout_dependency_drift"
        )
    return version, contract_hash


def _validate_json_shape(payload: object) -> None:
    nodes = 0

    def visit(value: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > OFFICIAL_JSON_MAX_TOTAL_NODES:
            raise MexcOfficialEvidenceContractError(
                "official_evidence_json_node_cap_exceeded"
            )
        if depth > OFFICIAL_JSON_MAX_DEPTH:
            raise MexcOfficialEvidenceContractError(
                "official_evidence_json_depth_cap_exceeded"
            )
        if value is None or type(value) is bool:
            return
        if type(value) is int:
            if len(str(abs(value))) > OFFICIAL_JSON_MAX_INTEGER_DIGITS:
                raise MexcOfficialEvidenceContractError(
                    "official_evidence_json_integer_digit_cap_exceeded"
                )
            return
        if type(value) is str:
            if len(value.encode("utf-8")) > OFFICIAL_JSON_MAX_STRING_BYTES:
                raise MexcOfficialEvidenceContractError(
                    "official_evidence_json_string_cap_exceeded"
                )
            return
        if type(value) is list:
            if len(value) > OFFICIAL_JSON_MAX_CONTAINER_ITEMS:
                raise MexcOfficialEvidenceContractError(
                    "official_evidence_json_container_cap_exceeded"
                )
            for item in value:
                visit(item, depth + 1)
            return
        if type(value) is dict:
            if len(value) > OFFICIAL_JSON_MAX_CONTAINER_ITEMS:
                raise MexcOfficialEvidenceContractError(
                    "official_evidence_json_container_cap_exceeded"
                )
            for key, item in value.items():
                if type(key) is not str:
                    raise MexcOfficialEvidenceContractError(
                        "official_evidence_json_key_type_is_invalid"
                    )
                visit(key, depth + 1)
                visit(item, depth + 1)
            return
        raise MexcOfficialEvidenceContractError(
            "official_evidence_json_scalar_type_is_invalid"
        )

    try:
        visit(payload, 0)
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_json_shape_validation_failed"
        ) from exc


def _canonical_json_bytes(payload: object) -> bytes:
    _validate_json_shape(payload)
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (
        TypeError,
        ValueError,
        RecursionError,
        OverflowError,
        UnicodeError,
    ) as exc:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_payload_is_not_canonical_json"
        ) from exc


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if type(pairs) is not list or len(pairs) > OFFICIAL_JSON_MAX_CONTAINER_ITEMS:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_json_container_cap_exceeded"
        )
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or len(key.encode("utf-8")) > OFFICIAL_JSON_MAX_STRING_BYTES:
            raise MexcOfficialEvidenceContractError(
                "official_evidence_json_key_is_invalid"
            )
        if key in result:
            raise MexcOfficialEvidenceContractError(
                "official_evidence_json_duplicate_key"
            )
        result[key] = value
    return result


def _reject_float(_: str) -> NoReturn:
    raise MexcOfficialEvidenceContractError("official_evidence_json_float_is_forbidden")


def _reject_constant(_: str) -> NoReturn:
    raise MexcOfficialEvidenceContractError(
        "official_evidence_json_constant_is_forbidden"
    )


def _parse_bounded_int(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if not digits or len(digits) > OFFICIAL_JSON_MAX_INTEGER_DIGITS:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_json_integer_digit_cap_exceeded"
        )
    try:
        return int(value, 10)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_json_integer_decode_failed"
        ) from exc


def parse_canonical_json_lf_v1(raw: bytes, *, max_bytes: int) -> object:
    """Parse exact canonical UTF-8 JSON with one LF and no duplicate keys."""

    if type(max_bytes) is not int or max_bytes < 1:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_json_bound_is_invalid"
        )
    if type(raw) is not bytes or len(raw) > max_bytes:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_json_exceeds_bound"
        )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_json_requires_single_lf"
        )
    try:
        payload = json.loads(
            raw[:-1].decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_int=_parse_bounded_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except MexcOfficialEvidenceContractError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        RecursionError,
        UnicodeError,
    ) as exc:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_json_decode_failed"
        ) from exc
    _validate_json_shape(payload)
    if _canonical_json_bytes(payload) != raw:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_json_is_not_exact_canonical_lf"
        )
    return payload


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MexcOfficialEvidenceContractError(f"{field}_is_not_sha256")
    return value


def _identifier(value: object, *, field: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise MexcOfficialEvidenceContractError(f"{field}_is_invalid")
    return value


def _strict_int(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise MexcOfficialEvidenceContractError(f"{field}_is_invalid")
    return value


def _optional_us(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _strict_int(value, field=field, minimum=1)


def _exact_mapping(
    payload: object, expected: frozenset[str], *, code: str
) -> dict[str, Any]:
    if type(payload) is not dict or frozenset(payload) != expected:
        raise MexcOfficialEvidenceContractError(code)
    return dict(payload)


def _validate_reference_url(value: object) -> str:
    if type(value) is not str or value != OFFICIAL_REFERENCE_URL:
        raise MexcOfficialEvidenceContractError(
            "official_reference_url_is_not_frozen"
        )
    parsed = urlsplit(value)
    if (
        parsed.scheme != OFFICIAL_REFERENCE_SCHEME
        or parsed.hostname != OFFICIAL_REFERENCE_HOST
        or parsed.port not in (None, OFFICIAL_REFERENCE_PORT)
        or parsed.path != OFFICIAL_REFERENCE_PATH
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or parsed.netloc != OFFICIAL_REFERENCE_HOST
    ):
        raise MexcOfficialEvidenceContractError("official_reference_url_has_alias")
    return value


def _exact_headers(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or not all(
        type(item) is tuple
        and len(item) == 2
        and all(type(part) is str for part in item)
        for item in value
    ):
        raise MexcOfficialEvidenceContractError(
            "official_reference_headers_are_not_exact_tuple"
        )
    if value != _REQUEST_HEADERS:
        raise MexcOfficialEvidenceContractError(
            "official_reference_headers_are_not_frozen"
        )
    return value


def _safe_headers(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or not all(
        type(item) is tuple
        and len(item) == 2
        and type(item[0]) is str
        and type(item[1]) is str
        for item in value
    ):
        raise MexcOfficialEvidenceContractError(
            "official_attempt_headers_are_not_exact_tuple"
        )
    previous = ""
    seen: set[str] = set()
    for name, header_value in value:
        try:
            header_value_bytes = len(header_value.encode("utf-8"))
        except UnicodeError as exc:
            raise MexcOfficialEvidenceContractError(
                "official_attempt_safe_header_is_not_utf8"
            ) from exc
        if (
            name != name.lower()
            or _SAFE_HEADER_NAME_RE.fullmatch(name) is None
            or name in _SECRET_RESPONSE_HEADERS
            or name not in _SAFE_RESPONSE_HEADERS
            or name in seen
            or name <= previous
            or not header_value
            or header_value_bytes > 2048
            or any(ord(character) < 32 or ord(character) == 127 for character in header_value)
        ):
            raise MexcOfficialEvidenceContractError(
                "official_attempt_safe_header_is_invalid"
            )
        seen.add(name)
        previous = name
    return value


def _canonical_content_length(value: object) -> int | None:
    """Return an exact ASCII decimal entity length, or ``None`` if invalid."""

    if type(value) is not str or re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
        return None
    try:
        return int(value, 10)
    except (TypeError, ValueError):
        return None


def _official_html_content_type(value: object) -> bool:
    return type(value) is str and value.split(";", 1)[0] in {
        "text/html",
        "application/xhtml+xml",
    }


@dataclass(frozen=True, slots=True)
class OfficialReferencePreparedRequestV1:
    verification_plan_hash: str
    endpoint_runner_contract_version: str
    endpoint_runner_contract_hash: str
    parser_contract_version: str
    parser_contract_hash: str
    transport_contract_version: str
    transport_contract_hash: str
    runtime_contract_version: str
    runtime_contract_hash: str
    candidate_contract_version: str = CANDIDATE_CONTRACT_VERSION
    candidate_contract_hash: str = CANDIDATE_CONTRACT_HASH
    method: str = "GET"
    url: str = OFFICIAL_REFERENCE_URL
    scheme: str = OFFICIAL_REFERENCE_SCHEME
    host: str = OFFICIAL_REFERENCE_HOST
    port: int = OFFICIAL_REFERENCE_PORT
    path: str = OFFICIAL_REFERENCE_PATH
    query: str = ""
    fragment: str = ""
    userinfo: str = ""
    reference_id: str = OFFICIAL_REFERENCE_ID
    headers: tuple[tuple[str, str], ...] = _REQUEST_HEADERS
    body_byte_count: int = 0
    body_sha256: str = EMPTY_SHA256
    tls_verify: bool = True
    tls_server_name: str = OFFICIAL_REFERENCE_HOST
    allow_redirects: bool = False
    trust_env: bool = False
    proxy_enabled: bool = False
    cookies_enabled: bool = False
    netrc_enabled: bool = False
    authentication_enabled: bool = False
    contract_version: str = OFFICIAL_REFERENCE_REQUEST_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != OFFICIAL_REFERENCE_REQUEST_VERSION:
            raise MexcOfficialEvidenceContractError(
                "official_reference_request_version_mismatch"
            )
        _digest(self.verification_plan_hash, field="official_verification_plan_hash")
        for version_field, hash_field in (
            ("endpoint_runner_contract_version", "endpoint_runner_contract_hash"),
            ("parser_contract_version", "parser_contract_hash"),
            ("transport_contract_version", "transport_contract_hash"),
            ("runtime_contract_version", "runtime_contract_hash"),
        ):
            _identifier(getattr(self, version_field), field=version_field)
            _digest(getattr(self, hash_field), field=hash_field)
        if (
            self.candidate_contract_version != CANDIDATE_CONTRACT_VERSION
            or self.candidate_contract_hash != CANDIDATE_CONTRACT_HASH
        ):
            raise MexcOfficialEvidenceContractError(
                "official_reference_candidate_binding_mismatch"
            )
        _validate_reference_url(self.url)
        _exact_headers(self.headers)
        _strict_int(
            self.port,
            field="official_reference_port",
            minimum=OFFICIAL_REFERENCE_PORT,
            maximum=OFFICIAL_REFERENCE_PORT,
        )
        _strict_int(
            self.body_byte_count,
            field="official_reference_body_byte_count",
            minimum=0,
            maximum=0,
        )
        exact = {
            "method": "GET",
            "scheme": OFFICIAL_REFERENCE_SCHEME,
            "host": OFFICIAL_REFERENCE_HOST,
            "port": OFFICIAL_REFERENCE_PORT,
            "path": OFFICIAL_REFERENCE_PATH,
            "query": "",
            "fragment": "",
            "userinfo": "",
            "reference_id": OFFICIAL_REFERENCE_ID,
            "body_byte_count": 0,
            "body_sha256": EMPTY_SHA256,
            "tls_server_name": OFFICIAL_REFERENCE_HOST,
        }
        if any(getattr(self, field) != expected for field, expected in exact.items()):
            raise MexcOfficialEvidenceContractError(
                "official_reference_prepared_get_is_not_exact"
            )
        booleans = {
            "tls_verify": True,
            "allow_redirects": False,
            "trust_env": False,
            "proxy_enabled": False,
            "cookies_enabled": False,
            "netrc_enabled": False,
            "authentication_enabled": False,
        }
        if any(getattr(self, field) is not expected for field, expected in booleans.items()):
            raise MexcOfficialEvidenceContractError(
                "official_reference_transport_safety_mismatch"
            )

    @property
    def prepared_request_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result["headers"] = [list(item) for item in self.headers]
        return result

    @classmethod
    def from_dict(cls, payload: object) -> "OfficialReferencePreparedRequestV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="official_reference_request_schema_mismatch",
        )
        raw_headers = values.get("headers")
        if type(raw_headers) is not list or not all(
            type(item) is list and len(item) == 2 for item in raw_headers
        ):
            raise MexcOfficialEvidenceContractError(
                "official_reference_request_headers_wire_type_mismatch"
            )
        values["headers"] = tuple((item[0], item[1]) for item in raw_headers)
        try:
            return cls(**values)
        except TypeError as exc:
            raise MexcOfficialEvidenceContractError(
                "official_reference_request_reconstruction_failed"
            ) from exc


def derive_official_bundle_root_v1(verification_plan_hash: str) -> str:
    _validated_pilot_output_layout_dependency()
    local_root = (
        "endpoint-evidence/"
        f"{_digest(verification_plan_hash, field='official_bundle_plan_hash')}/official"
    )
    try:
        layout_root = _pilot_output_layout.derive_official_bundle_root_v1(
            verification_plan_hash
        )
    except _pilot_output_layout.PilotOutputLayoutError as exc:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_layout_bundle_root_derivation_failed"
        ) from exc
    if type(layout_root) is not str or layout_root != local_root:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_layout_bundle_root_drift"
        )
    return local_root


def official_bundle_relative_paths_v1(
    verification_plan_hash: str,
) -> tuple[str, str, str]:
    root = derive_official_bundle_root_v1(verification_plan_hash)
    local_paths = (
        f"{root}/attempt-000.body.bin",
        f"{root}/attempt-000.receipt.json",
        f"{root}/evidence.json",
    )
    try:
        layout_paths = _pilot_output_layout.derive_official_bundle_locators_v1(
            verification_plan_hash
        )
    except _pilot_output_layout.PilotOutputLayoutError as exc:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_layout_bundle_locator_derivation_failed"
        ) from exc
    if type(layout_paths) is not tuple or layout_paths != local_paths:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_layout_bundle_locator_drift"
        )
    return local_paths


@dataclass(frozen=True, slots=True)
class OfficialReferenceHttpAttemptV1:
    manifest_hash: str
    authorization_receipt_hash: str
    preflight_receipt_hash: str
    verification_plan_hash: str
    network_intent_hash: str
    endpoint_runner_contract_version: str
    endpoint_runner_contract_hash: str
    runtime_authority_binding_hash: str
    clock_domain_id: str
    tls_policy_version: str
    tls_policy_hash: str
    trust_store_version: str
    trust_store_hash: str
    prepared_request_hash: str
    gate_checked_at_us: int
    gate_checked_monotonic_us: int
    request_started_at_us: int
    request_started_monotonic_us: int
    tls_validated_at_us: int | None
    tls_validated_monotonic_us: int | None
    headers_received_at_us: int | None
    headers_received_monotonic_us: int | None
    body_eof_at_us: int | None
    body_eof_monotonic_us: int | None
    connection_closed_at_us: int
    connection_closed_monotonic_us: int
    tls_version: str | None
    peer_leaf_certificate_sha256: str | None
    validated_chain_sha256: str | None
    pkix_validated: bool
    status_code: int | None
    response_headers: tuple[tuple[str, str], ...]
    body_complete: bool
    terminal_progress: str
    outcome: str
    safe_error_code: str | None
    raw_body_byte_count: int
    raw_body_sha256: str
    operation: str = "official_reference_fetch"
    attempt_ordinal: int = 0
    requested_url: str = OFFICIAL_REFERENCE_URL
    final_url: str = OFFICIAL_REFERENCE_URL
    redirects_followed: int = 0
    tls_sni: str = OFFICIAL_REFERENCE_HOST
    credentials_used: bool = False
    proxy_used: bool = False
    cookies_used: bool = False
    netrc_used: bool = False
    trust_env: bool = False
    authority_status: str = REVIEWED_FAKE_FIXTURE_ONLY
    terminal_compatible: bool = False
    contract_version: str = OFFICIAL_REFERENCE_HTTP_ATTEMPT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != OFFICIAL_REFERENCE_HTTP_ATTEMPT_VERSION:
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_version_mismatch"
            )
        for field in (
            "manifest_hash",
            "authorization_receipt_hash",
            "preflight_receipt_hash",
            "verification_plan_hash",
            "network_intent_hash",
            "endpoint_runner_contract_hash",
            "runtime_authority_binding_hash",
            "tls_policy_hash",
            "trust_store_hash",
            "prepared_request_hash",
            "raw_body_sha256",
        ):
            _digest(getattr(self, field), field=field)
        for field in (
            "endpoint_runner_contract_version",
            "clock_domain_id",
            "tls_policy_version",
            "trust_store_version",
        ):
            _identifier(getattr(self, field), field=field)
        if type(self.operation) is not str or self.operation != "official_reference_fetch":
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_slot_mismatch"
            )
        _strict_int(
            self.attempt_ordinal,
            field="official_http_attempt_ordinal",
            minimum=0,
            maximum=0,
        )
        _validate_reference_url(self.requested_url)
        _validate_reference_url(self.final_url)
        _strict_int(
            self.redirects_followed,
            field="official_http_attempt_redirect_count",
            minimum=0,
            maximum=0,
        )
        if type(self.tls_sni) is not str or self.tls_sni != OFFICIAL_REFERENCE_HOST:
            raise MexcOfficialEvidenceContractError("official_http_attempt_sni_mismatch")
        for field in (
            "pkix_validated",
            "body_complete",
            "credentials_used",
            "proxy_used",
            "cookies_used",
            "netrc_used",
            "trust_env",
            "terminal_compatible",
        ):
            if type(getattr(self, field)) is not bool:
                raise MexcOfficialEvidenceContractError(
                    f"official_http_attempt_{field}_must_be_exact_bool"
                )
        if any(
            getattr(self, field) is not False
            for field in (
                "credentials_used",
                "proxy_used",
                "cookies_used",
                "netrc_used",
                "trust_env",
            )
        ):
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_ambient_authority_detected"
            )
        if (
            type(self.authority_status) is not str
            or self.authority_status != REVIEWED_FAKE_FIXTURE_ONLY
            or self.terminal_compatible is not False
        ):
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_must_remain_fake_nonterminal"
            )
        _safe_headers(self.response_headers)
        epoch_fields = (
            "gate_checked_at_us",
            "request_started_at_us",
            "tls_validated_at_us",
            "headers_received_at_us",
            "body_eof_at_us",
            "connection_closed_at_us",
        )
        monotonic_fields = (
            "gate_checked_monotonic_us",
            "request_started_monotonic_us",
            "tls_validated_monotonic_us",
            "headers_received_monotonic_us",
            "body_eof_monotonic_us",
            "connection_closed_monotonic_us",
        )
        epoch = tuple(_optional_us(getattr(self, field), field=field) for field in epoch_fields)
        monotonic = tuple(
            _optional_us(getattr(self, field), field=field) for field in monotonic_fields
        )
        if any((left is None) != (right is None) for left, right in zip(epoch, monotonic)):
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_clock_pairs_mismatch"
            )
        if any(epoch[index] is None for index in (0, 1, 5)) or any(
            monotonic[index] is None for index in (0, 1, 5)
        ):
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_required_timestamps_are_missing"
            )
        present_epoch = tuple(value for value in epoch if value is not None)
        present_mono = tuple(value for value in monotonic if value is not None)
        if list(present_epoch) != sorted(present_epoch) or list(present_mono) != sorted(present_mono):
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_timeline_is_not_monotonic"
            )
        _strict_int(
            self.raw_body_byte_count,
            field="official_raw_body_bytes",
            maximum=OFFICIAL_RAW_BODY_HARD_CAP_BYTES,
        )
        if self.raw_body_byte_count == 0 and self.raw_body_sha256 != EMPTY_SHA256:
            raise MexcOfficialEvidenceContractError(
                "official_empty_raw_body_hash_mismatch"
            )
        if self.status_code is not None:
            _strict_int(self.status_code, field="official_http_status", minimum=100, maximum=599)
        if self.tls_version is not None and (
            type(self.tls_version) is not str
            or self.tls_version not in {"TLSv1.2", "TLSv1.3"}
        ):
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_tls_version_is_invalid"
            )
        for field in ("peer_leaf_certificate_sha256", "validated_chain_sha256"):
            value = getattr(self, field)
            if value is not None:
                _digest(value, field=field)
        if (
            type(self.terminal_progress) is not str
            or self.terminal_progress not in _TERMINAL_PROGRESS_PHASES
        ):
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_terminal_progress_is_invalid"
            )
        if type(self.outcome) is not str or self.outcome not in {
            "complete",
            "incomplete_transport_error",
            "incomplete_tls_error",
            "incomplete_http_body_error",
            "rejected_protocol",
        }:
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_outcome_is_invalid"
            )
        event_presence = tuple(value is not None for value in epoch[2:5])
        if event_presence != _TERMINAL_PROGRESS_EVENT_PRESENCE[
            self.terminal_progress
        ]:
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_terminal_progress_prefix_mismatch"
            )
        tls_validated_facts = (
            self.tls_version is not None
            and self.peer_leaf_certificate_sha256 is not None
            and self.validated_chain_sha256 is not None
            and self.pkix_validated is True
        )
        tls_facts_absent = (
            self.tls_version is None
            and self.peer_leaf_certificate_sha256 is None
            and self.validated_chain_sha256 is None
            and self.pkix_validated is False
        )
        tls_failure_prefix_is_consistent = (
            self.validated_chain_sha256 is None
            and self.pkix_validated is False
            and (
                self.peer_leaf_certificate_sha256 is None
                or self.tls_version is not None
            )
        )
        header_map = dict(self.response_headers)
        content_encoding = header_map.get("content-encoding")
        content_type = header_map.get("content-type")
        content_type_is_official_html = _official_html_content_type(content_type)
        raw_content_length = header_map.get("content-length")
        content_length = _canonical_content_length(raw_content_length)
        if self.outcome == "complete":
            if (
                self.terminal_progress != "body_eof"
                or not tls_validated_facts
                or self.status_code != 200
                or self.body_complete is not True
                or self.safe_error_code is not None
                or self.raw_body_byte_count < 1
                or not content_type_is_official_html
            ):
                raise MexcOfficialEvidenceContractError(
                    "official_complete_attempt_evidence_is_incomplete"
                )
            if content_encoding not in (None, "identity"):
                raise MexcOfficialEvidenceContractError(
                    "official_complete_attempt_is_content_encoded"
                )
            if raw_content_length is not None:
                if content_length is None:
                    raise MexcOfficialEvidenceContractError(
                        "official_content_length_is_invalid"
                    )
                if content_length != self.raw_body_byte_count:
                    raise MexcOfficialEvidenceContractError(
                        "official_content_length_mismatch"
                    )
            return
        allowed_errors = _SAFE_ERROR_CODES_BY_OUTCOME[self.outcome]
        if (
            type(self.safe_error_code) is not str
            or self.safe_error_code not in allowed_errors
            or self.body_complete is not False
        ):
            raise MexcOfficialEvidenceContractError(
                "official_incomplete_attempt_lacks_allowlisted_failure_evidence"
            )
        if self.outcome == "incomplete_transport_error":
            allowed_progress = _TRANSPORT_PROGRESS_BY_SAFE_ERROR[
                self.safe_error_code
            ]
            valid = (
                self.terminal_progress in allowed_progress
                and self.status_code is None
                and self.response_headers == ()
                and self.raw_body_byte_count == 0
                and self.raw_body_sha256 == EMPTY_SHA256
                and (
                    (
                        self.terminal_progress == "before_tls_validation"
                        and tls_facts_absent
                    )
                    or (
                        self.terminal_progress == "tls_validated_before_headers"
                        and tls_validated_facts
                    )
                )
            )
        elif self.outcome == "incomplete_tls_error":
            valid = (
                self.terminal_progress == "tls_validation_failed"
                and tls_failure_prefix_is_consistent
                and self.status_code is None
                and self.response_headers == ()
                and self.raw_body_byte_count == 0
                and self.raw_body_sha256 == EMPTY_SHA256
            )
            if self.safe_error_code in {
                "tls_certificate_validation_failed",
                "tls_sni_mismatch",
            }:
                valid = valid and self.peer_leaf_certificate_sha256 is not None
            elif self.safe_error_code == "tls_policy_rejected":
                valid = valid and self.tls_version is not None
            elif self.safe_error_code == "tls_handshake_failed":
                valid = valid
        elif self.outcome == "incomplete_http_body_error":
            valid = (
                self.terminal_progress == "headers_received_before_body_eof"
                and tls_validated_facts
                and self.status_code == 200
                and content_type_is_official_html
                and content_encoding in (None, "identity")
                and (
                    raw_content_length is None or content_length is not None
                )
            )
            if self.safe_error_code == "content_length_mismatch":
                valid = (
                    valid
                    and raw_content_length is not None
                    and content_length is not None
                    and content_length != self.raw_body_byte_count
                )
            elif self.safe_error_code == "body_cap_exceeded":
                valid = valid and (
                    self.raw_body_byte_count == OFFICIAL_RAW_BODY_HARD_CAP_BYTES
                    or (
                        content_length is not None
                        and content_length > OFFICIAL_RAW_BODY_HARD_CAP_BYTES
                    )
                )
        else:
            valid = (
                self.terminal_progress == "headers_received_before_body_eof"
                and tls_validated_facts
                and self.status_code is not None
                and self.raw_body_byte_count == 0
                and self.raw_body_sha256 == EMPTY_SHA256
            )
            if self.safe_error_code == "redirect_status_rejected":
                valid = valid and 300 <= self.status_code <= 399
            elif self.safe_error_code == "http_status_not_200":
                valid = valid and self.status_code != 200 and not (
                    300 <= self.status_code <= 399
                )
            elif self.safe_error_code == "content_encoding_not_identity":
                valid = valid and self.status_code == 200 and content_encoding not in (
                    None,
                    "identity",
                )
            elif self.safe_error_code == "content_type_not_official_html":
                valid = (
                    valid
                    and self.status_code == 200
                    and content_encoding in (None, "identity")
                    and not content_type_is_official_html
                )
            elif self.safe_error_code == "content_length_invalid":
                valid = (
                    valid
                    and self.status_code == 200
                    and content_encoding in (None, "identity")
                    and content_type_is_official_html
                    and raw_content_length is not None
                    and content_length is None
                )
        if not valid:
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_outcome_state_matrix_mismatch"
            )

    @property
    def raw_body_relative_path(self) -> str:
        return official_bundle_relative_paths_v1(self.verification_plan_hash)[0]

    @property
    def attempt_receipt_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result["response_headers"] = [list(item) for item in self.response_headers]
        result["raw_body_relative_path"] = self.raw_body_relative_path
        return result

    @classmethod
    def from_dict(cls, payload: object) -> "OfficialReferenceHttpAttemptV1":
        expected = frozenset(cls.__dataclass_fields__) | {"raw_body_relative_path"}
        values = _exact_mapping(
            payload,
            expected,
            code="official_http_attempt_schema_mismatch",
        )
        locator = values.pop("raw_body_relative_path")
        raw_headers = values.get("response_headers")
        if type(raw_headers) is not list or not all(
            type(item) is list and len(item) == 2 for item in raw_headers
        ):
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_headers_wire_type_mismatch"
            )
        values["response_headers"] = tuple((item[0], item[1]) for item in raw_headers)
        try:
            result = cls(**values)
        except TypeError as exc:
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_reconstruction_failed"
            ) from exc
        if locator != result.raw_body_relative_path:
            raise MexcOfficialEvidenceContractError(
                "official_http_attempt_raw_locator_mismatch"
            )
        return result


@dataclass(frozen=True, slots=True)
class OfficialDocumentSpanClaimV1:
    role: str
    start_byte: int
    end_byte_exclusive: int
    span_sha256: str
    contract_version: str = OFFICIAL_DOCUMENT_SPAN_CLAIM_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != OFFICIAL_DOCUMENT_SPAN_CLAIM_VERSION:
            raise MexcOfficialEvidenceContractError(
                "official_document_span_claim_version_mismatch"
            )
        if type(self.role) is not str or self.role not in _CLAIM_LITERALS:
            raise MexcOfficialEvidenceContractError(
                "official_document_span_claim_role_is_invalid"
            )
        start = _strict_int(self.start_byte, field="official_claim_start")
        end = _strict_int(self.end_byte_exclusive, field="official_claim_end", minimum=1)
        if end <= start or end - start != len(_CLAIM_LITERALS[self.role]):
            raise MexcOfficialEvidenceContractError(
                "official_document_span_claim_bounds_are_invalid"
            )
        _digest(self.span_sha256, field="official_claim_span_hash")

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "OfficialDocumentSpanClaimV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="official_document_span_claim_schema_mismatch",
        )
        try:
            return cls(**values)
        except TypeError as exc:
            raise MexcOfficialEvidenceContractError(
                "official_document_span_claim_reconstruction_failed"
            ) from exc


def _derive_claims_from_raw_body(
    raw_body: bytes,
) -> tuple[OfficialDocumentSpanClaimV1, ...]:
    if type(raw_body) is not bytes or not raw_body:
        raise MexcOfficialEvidenceSemanticStop("official_document_raw_body_is_empty")
    if len(raw_body) > OFFICIAL_RAW_BODY_HARD_CAP_BYTES:
        raise MexcOfficialEvidenceBudgetStop(
            "official_document_raw_body_hard_cap_exceeded"
        )
    derived: list[OfficialDocumentSpanClaimV1] = []
    for role, literal in _CLAIM_LITERALS.items():
        start = raw_body.find(literal)
        if start < 0:
            continue
        if raw_body.find(literal, start + 1) >= 0:
            raise MexcOfficialEvidenceSemanticStop(
                "official_document_statement_is_ambiguous"
            )
        end = start + len(literal)
        if (
            (start != 0 and raw_body[start - 1 : start] != b"\n")
            or (end != len(raw_body) and raw_body[end : end + 1] != b"\n")
        ):
            raise MexcOfficialEvidenceSemanticStop(
                "official_document_statement_lacks_exact_line_context"
            )
        derived.append(
            OfficialDocumentSpanClaimV1(
                role=role,
                start_byte=start,
                end_byte_exclusive=end,
                span_sha256=hashlib.sha256(literal).hexdigest(),
            )
        )
    return tuple(
        sorted(
            derived,
            key=lambda item: (item.start_byte, item.end_byte_exclusive, item.role),
        )
    )


def _verify_and_derive_support_scope(
    raw_body: bytes, claims: tuple[OfficialDocumentSpanClaimV1, ...]
) -> str:
    derived_claims = _derive_claims_from_raw_body(raw_body)
    if type(claims) is not tuple or not all(
        type(claim) is OfficialDocumentSpanClaimV1 for claim in claims
    ):
        raise MexcOfficialEvidenceContractError(
            "official_document_claims_are_not_exact_immutable_tuple"
        )
    if not claims:
        raise MexcOfficialEvidenceSemanticStop("official_document_has_no_claims")
    if claims != derived_claims:
        raise MexcOfficialEvidenceContractError(
            "official_document_claims_are_not_deterministically_derived"
        )
    if claims != tuple(
        sorted(claims, key=lambda item: (item.start_byte, item.end_byte_exclusive, item.role))
    ):
        raise MexcOfficialEvidenceContractError(
            "official_document_claims_are_not_canonical_order"
        )
    roles: dict[str, OfficialDocumentSpanClaimV1] = {}
    previous_end = 0
    for claim in claims:
        if claim.role in roles or claim.start_byte < previous_end:
            raise MexcOfficialEvidenceContractError(
                "official_document_claims_overlap_or_repeat"
            )
        if claim.end_byte_exclusive > len(raw_body):
            raise MexcOfficialEvidenceContractError(
                "official_document_claim_exceeds_raw_body"
            )
        observed = raw_body[claim.start_byte : claim.end_byte_exclusive]
        if (
            observed != _CLAIM_LITERALS[claim.role]
            or hashlib.sha256(observed).hexdigest() != claim.span_sha256
        ):
            raise MexcOfficialEvidenceContractError(
                "official_document_claim_does_not_match_exact_raw_span"
            )
        roles[claim.role] = claim
        previous_end = claim.end_byte_exclusive
    present = frozenset(roles)
    if not _MIGRATION_ROLES.issubset(present):
        raise MexcOfficialEvidenceSemanticStop(
            "official_document_does_not_prove_domain_migration"
        )
    return (
        "full_candidate_contract"
        if _FULL_CANDIDATE_ROLES.issubset(present)
        else "domain_migration_only"
    )


@dataclass(frozen=True, slots=True)
class OfficialDocumentEvidenceV1:
    verification_plan_hash: str
    prepared_request_hash: str
    attempt_receipt_hash: str
    attempt_receipt_byte_count: int
    raw_body_relative_path: str
    raw_body_byte_count: int
    raw_body_sha256: str
    reference_url: str
    reference_id: str
    observed_body_fetched_at_us: int
    observed_body_fetched_monotonic_us: int
    candidate_contract_version: str
    candidate_contract_hash: str
    parser_contract_version: str
    parser_contract_hash: str
    reader_contract_version: str
    reader_contract_hash: str
    claims: tuple[OfficialDocumentSpanClaimV1, ...]
    support_scope: str
    verdict: str
    parse_started_at_us: int
    parse_completed_at_us: int
    parse_started_monotonic_us: int
    parse_completed_monotonic_us: int
    reload_completed_at_us: int
    reload_completed_monotonic_us: int
    authority_status: str = REVIEWED_FAKE_FIXTURE_ONLY
    terminal_compatible: bool = False
    contract_version: str = OFFICIAL_DOCUMENT_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != OFFICIAL_DOCUMENT_EVIDENCE_VERSION:
            raise MexcOfficialEvidenceContractError(
                "official_document_evidence_version_mismatch"
            )
        for field in (
            "verification_plan_hash",
            "prepared_request_hash",
            "attempt_receipt_hash",
            "raw_body_sha256",
            "candidate_contract_hash",
            "parser_contract_hash",
            "reader_contract_hash",
        ):
            _digest(getattr(self, field), field=field)
        _strict_int(
            self.attempt_receipt_byte_count,
            field="official_evidence_attempt_receipt_bytes",
            minimum=1,
            maximum=OFFICIAL_RECEIPT_MAX_BYTES,
        )
        if self.raw_body_relative_path != official_bundle_relative_paths_v1(
            self.verification_plan_hash
        )[0]:
            raise MexcOfficialEvidenceContractError(
                "official_document_evidence_raw_locator_mismatch"
            )
        _strict_int(
            self.raw_body_byte_count,
            field="official_evidence_raw_bytes",
            minimum=1,
            maximum=OFFICIAL_RAW_BODY_HARD_CAP_BYTES,
        )
        _validate_reference_url(self.reference_url)
        if self.reference_id != OFFICIAL_REFERENCE_ID:
            raise MexcOfficialEvidenceContractError(
                "official_document_reference_id_mismatch"
            )
        if (
            self.candidate_contract_version != CANDIDATE_CONTRACT_VERSION
            or self.candidate_contract_hash != CANDIDATE_CONTRACT_HASH
            or self.reader_contract_version != OFFICIAL_DOCUMENT_READER_VERSION
            or self.reader_contract_hash
            != mexc_endpoint_official_evidence_contract_hash()
        ):
            raise MexcOfficialEvidenceContractError(
                "official_document_semantic_dependency_mismatch"
            )
        for field in ("parser_contract_version", "reader_contract_version"):
            _identifier(getattr(self, field), field=field)
        if type(self.claims) is not tuple or not all(
            type(claim) is OfficialDocumentSpanClaimV1 for claim in self.claims
        ):
            raise MexcOfficialEvidenceContractError(
                "official_document_evidence_claims_are_not_exact_tuple"
            )
        if type(self.support_scope) is not str or self.support_scope not in {
            "domain_migration_only",
            "full_candidate_contract",
        }:
            raise MexcOfficialEvidenceContractError(
                "official_document_support_scope_is_invalid"
            )
        expected_verdict = {
            "domain_migration_only": "additional_current_official_contract_evidence_required",
            "full_candidate_contract": "candidate_contract_semantics_observed",
        }[self.support_scope]
        if type(self.verdict) is not str or self.verdict != expected_verdict:
            raise MexcOfficialEvidenceContractError(
                "official_document_evidence_verdict_mismatch"
            )
        epoch = tuple(
            _strict_int(getattr(self, field), field=field, minimum=1)
            for field in (
                "observed_body_fetched_at_us",
                "parse_started_at_us",
                "parse_completed_at_us",
                "reload_completed_at_us",
            )
        )
        mono = tuple(
            _strict_int(getattr(self, field), field=field, minimum=1)
            for field in (
                "observed_body_fetched_monotonic_us",
                "parse_started_monotonic_us",
                "parse_completed_monotonic_us",
                "reload_completed_monotonic_us",
            )
        )
        if (
            epoch != tuple(sorted(epoch))
            or mono != tuple(sorted(mono))
            or epoch[0] >= epoch[1]
            or mono[0] >= mono[1]
            or epoch[2] >= epoch[3]
            or mono[2] >= mono[3]
        ):
            raise MexcOfficialEvidenceContractError(
                "official_document_evidence_timeline_is_invalid"
            )
        if (
            type(self.terminal_compatible) is not bool
            or type(self.authority_status) is not str
            or self.authority_status != REVIEWED_FAKE_FIXTURE_ONLY
            or self.terminal_compatible is not False
        ):
            raise MexcOfficialEvidenceContractError(
                "official_document_evidence_must_remain_fake_nonterminal"
            )

    @property
    def observed_body_version(self) -> dict[str, object]:
        return {
            "canonical_url": self.reference_url,
            "reference_id": self.reference_id,
            "fetched_at_us": self.observed_body_fetched_at_us,
            "fetched_monotonic_us": self.observed_body_fetched_monotonic_us,
            "body_sha256": self.raw_body_sha256,
        }

    def as_dict(self) -> dict[str, object]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result["claims"] = [claim.as_dict() for claim in self.claims]
        result["observed_body_version"] = self.observed_body_version
        return result

    @property
    def canonical_lf_bytes(self) -> bytes:
        return _canonical_json_bytes(self.as_dict())

    @property
    def evidence_hash(self) -> str:
        return hashlib.sha256(self.canonical_lf_bytes).hexdigest()

    @classmethod
    def from_dict(cls, payload: object) -> "OfficialDocumentEvidenceV1":
        expected = frozenset(cls.__dataclass_fields__) | {"observed_body_version"}
        values = _exact_mapping(
            payload,
            expected,
            code="official_document_evidence_schema_mismatch",
        )
        raw_version = values.pop("observed_body_version")
        raw_claims = values.get("claims")
        if type(raw_claims) is not list:
            raise MexcOfficialEvidenceContractError(
                "official_document_evidence_claims_wire_type_mismatch"
            )
        values["claims"] = tuple(
            OfficialDocumentSpanClaimV1.from_dict(item) for item in raw_claims
        )
        try:
            result = cls(**values)
        except TypeError as exc:
            raise MexcOfficialEvidenceContractError(
                "official_document_evidence_reconstruction_failed"
            ) from exc
        if raw_version != result.observed_body_version:
            raise MexcOfficialEvidenceContractError(
                "official_document_observed_body_version_mismatch"
            )
        return result


def build_exact_span_claims_v1(
    raw_body: bytes,
) -> tuple[OfficialDocumentSpanClaimV1, ...]:
    """Extract every exact standalone statement; caller cannot select roles."""

    if type(raw_body) is not bytes:
        raise MexcOfficialEvidenceContractError(
            "official_span_claim_builder_inputs_are_invalid"
        )
    claims = _derive_claims_from_raw_body(raw_body)
    if not _MIGRATION_ROLES.issubset({claim.role for claim in claims}):
        raise MexcOfficialEvidenceSemanticStop(
            "official_document_does_not_prove_domain_migration"
        )
    return claims


def read_official_document_evidence_v1(
    *,
    raw_body: bytes,
    attempt: OfficialReferenceHttpAttemptV1,
    prepared_request: OfficialReferencePreparedRequestV1,
    claims: tuple[OfficialDocumentSpanClaimV1, ...],
    parser_contract_version: str,
    parser_contract_hash: str,
    reader_contract_hash: str,
    parse_started_at_us: int,
    parse_completed_at_us: int,
    parse_started_monotonic_us: int,
    parse_completed_monotonic_us: int,
    reload_completed_at_us: int,
    reload_completed_monotonic_us: int,
) -> OfficialDocumentEvidenceV1:
    """Reparse exact bytes and derive semantic scope; never trust a support flag."""

    if type(raw_body) is not bytes:
        raise MexcOfficialEvidenceContractError(
            "official_document_reader_raw_body_must_be_exact_bytes"
        )
    if type(attempt) is not OfficialReferenceHttpAttemptV1 or type(
        prepared_request
    ) is not OfficialReferencePreparedRequestV1:
        raise MexcOfficialEvidenceContractError(
            "official_document_reader_inputs_are_not_exact_contracts"
        )
    if attempt.outcome != "complete":
        raise MexcOfficialEvidenceSemanticStop(
            "official_document_attempt_is_not_complete"
        )
    if (
        parser_contract_version != prepared_request.parser_contract_version
        or parser_contract_hash != prepared_request.parser_contract_hash
    ):
        raise MexcOfficialEvidenceContractError(
            "official_document_reader_parser_binding_mismatch"
        )
    if reader_contract_hash != mexc_endpoint_official_evidence_contract_hash():
        raise MexcOfficialEvidenceContractError(
            "official_document_reader_self_binding_mismatch"
        )
    if (
        attempt.prepared_request_hash != prepared_request.prepared_request_hash
        or attempt.verification_plan_hash != prepared_request.verification_plan_hash
        or attempt.raw_body_byte_count != len(raw_body)
        or attempt.raw_body_sha256 != hashlib.sha256(raw_body).hexdigest()
    ):
        raise MexcOfficialEvidenceContractError(
            "official_document_reader_binding_mismatch"
        )
    if attempt.body_eof_at_us is None or attempt.body_eof_monotonic_us is None:
        raise MexcOfficialEvidenceContractError(
            "official_document_reader_lacks_fetch_time"
        )
    scope = _verify_and_derive_support_scope(raw_body, claims)
    attempt_receipt_bytes = _canonical_json_bytes(attempt.as_dict())
    if len(attempt_receipt_bytes) > OFFICIAL_RECEIPT_MAX_BYTES:
        raise MexcOfficialEvidenceBudgetStop(
            "official_document_attempt_receipt_cap_exceeded"
        )
    evidence = OfficialDocumentEvidenceV1(
        verification_plan_hash=attempt.verification_plan_hash,
        prepared_request_hash=prepared_request.prepared_request_hash,
        attempt_receipt_hash=attempt.attempt_receipt_hash,
        attempt_receipt_byte_count=len(attempt_receipt_bytes),
        raw_body_relative_path=attempt.raw_body_relative_path,
        raw_body_byte_count=len(raw_body),
        raw_body_sha256=hashlib.sha256(raw_body).hexdigest(),
        reference_url=prepared_request.url,
        reference_id=prepared_request.reference_id,
        observed_body_fetched_at_us=attempt.body_eof_at_us,
        observed_body_fetched_monotonic_us=attempt.body_eof_monotonic_us,
        candidate_contract_version=CANDIDATE_CONTRACT_VERSION,
        candidate_contract_hash=CANDIDATE_CONTRACT_HASH,
        parser_contract_version=parser_contract_version,
        parser_contract_hash=parser_contract_hash,
        reader_contract_version=OFFICIAL_DOCUMENT_READER_VERSION,
        reader_contract_hash=reader_contract_hash,
        claims=claims,
        support_scope=scope,
        verdict=(
            "candidate_contract_semantics_observed"
            if scope == "full_candidate_contract"
            else "additional_current_official_contract_evidence_required"
        ),
        parse_started_at_us=parse_started_at_us,
        parse_completed_at_us=parse_completed_at_us,
        parse_started_monotonic_us=parse_started_monotonic_us,
        parse_completed_monotonic_us=parse_completed_monotonic_us,
        reload_completed_at_us=reload_completed_at_us,
        reload_completed_monotonic_us=reload_completed_monotonic_us,
    )
    # Revalidate from raw after construction so no passed scope/verdict can be
    # authoritative even if this function is refactored later.
    if _verify_and_derive_support_scope(raw_body, evidence.claims) != evidence.support_scope:
        raise MexcOfficialEvidenceContractError(
            "official_document_rederived_scope_mismatch"
        )
    return evidence


@dataclass(frozen=True, slots=True)
class OfficialEvidenceBundleFileV1:
    role: str
    relative_path: str
    artifact_sha256: str
    byte_count: int
    contract_version: str = OFFICIAL_EVIDENCE_BUNDLE_FILE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != OFFICIAL_EVIDENCE_BUNDLE_FILE_VERSION:
            raise MexcOfficialEvidenceContractError(
                "official_bundle_file_version_mismatch"
            )
        if type(self.role) is not str or self.role not in {
            "raw_body",
            "attempt_receipt",
            "semantic_evidence",
        }:
            raise MexcOfficialEvidenceContractError(
                "official_bundle_file_role_is_invalid"
            )
        _relative_locator(self.relative_path)
        _digest(self.artifact_sha256, field="official_bundle_file_hash")
        _strict_int(self.byte_count, field="official_bundle_file_bytes", minimum=1)

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "OfficialEvidenceBundleFileV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="official_bundle_file_schema_mismatch",
        )
        try:
            return cls(**values)
        except TypeError as exc:
            raise MexcOfficialEvidenceContractError(
                "official_bundle_file_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class OfficialEvidenceBundleV1:
    verification_plan_hash: str
    prepared_request_hash: str
    attempt_receipt_hash: str
    evidence_hash: str
    raw_body_sha256: str
    files: tuple[OfficialEvidenceBundleFileV1, ...]
    authority_status: str = REVIEWED_FAKE_FIXTURE_ONLY
    terminal_compatible: bool = False
    contract_version: str = OFFICIAL_EVIDENCE_BUNDLE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != OFFICIAL_EVIDENCE_BUNDLE_VERSION:
            raise MexcOfficialEvidenceContractError(
                "official_bundle_version_mismatch"
            )
        for field in (
            "verification_plan_hash",
            "prepared_request_hash",
            "attempt_receipt_hash",
            "evidence_hash",
            "raw_body_sha256",
        ):
            _digest(getattr(self, field), field=field)
        if type(self.files) is not tuple or not all(
            type(item) is OfficialEvidenceBundleFileV1 for item in self.files
        ):
            raise MexcOfficialEvidenceContractError(
                "official_bundle_files_are_not_exact_tuple"
            )
        if tuple(item.role for item in self.files) != (
            "raw_body",
            "attempt_receipt",
            "semantic_evidence",
        ):
            raise MexcOfficialEvidenceContractError(
                "official_bundle_file_roles_mismatch"
            )
        if tuple(item.relative_path for item in self.files) != official_bundle_relative_paths_v1(
            self.verification_plan_hash
        ):
            raise MexcOfficialEvidenceContractError(
                "official_bundle_file_locators_mismatch"
            )
        if self.files[0].artifact_sha256 != self.raw_body_sha256:
            raise MexcOfficialEvidenceContractError(
                "official_bundle_raw_body_hash_mismatch"
            )
        if self.files[1].artifact_sha256 != self.attempt_receipt_hash:
            raise MexcOfficialEvidenceContractError(
                "official_bundle_attempt_hash_mismatch"
            )
        if self.files[2].artifact_sha256 != self.evidence_hash:
            raise MexcOfficialEvidenceContractError(
                "official_bundle_evidence_hash_mismatch"
            )
        if (
            type(self.terminal_compatible) is not bool
            or type(self.authority_status) is not str
            or self.authority_status != REVIEWED_FAKE_FIXTURE_ONLY
            or self.terminal_compatible is not False
        ):
            raise MexcOfficialEvidenceContractError(
                "official_bundle_must_remain_fake_nonterminal"
            )

    @property
    def bundle_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    @property
    def total_storage_bytes(self) -> int:
        return sum(item.byte_count for item in self.files)

    def as_dict(self) -> dict[str, object]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result["files"] = [item.as_dict() for item in self.files]
        return result

    @classmethod
    def from_dict(cls, payload: object) -> "OfficialEvidenceBundleV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="official_bundle_schema_mismatch",
        )
        raw_files = values.get("files")
        if type(raw_files) is not list:
            raise MexcOfficialEvidenceContractError(
                "official_bundle_files_wire_type_mismatch"
            )
        values["files"] = tuple(
            OfficialEvidenceBundleFileV1.from_dict(item) for item in raw_files
        )
        try:
            return cls(**values)
        except TypeError as exc:
            raise MexcOfficialEvidenceContractError(
                "official_bundle_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class OfficialEvidenceCompatibilityV1:
    verification_plan: object
    verification_plan_hash: str
    max_network_attempts: int
    max_total_raw_body_bytes: int
    max_total_storage_bytes: int
    max_runtime_us: int
    reserved_live_raw_body_bytes: int
    reserved_live_storage_bytes: int
    reserved_live_runtime_us: int
    residual_official_raw_body_bytes: int
    residual_official_storage_bytes: int
    residual_official_runtime_us: int
    compatibility_status: str = "offline_bundle_only_terminal_stop"
    authority_status: str = REVIEWED_FAKE_FIXTURE_ONLY
    terminal_compatible: bool = False
    contract_version: str = OFFICIAL_EVIDENCE_COMPATIBILITY_VERSION

    def __post_init__(self) -> None:
        from trading.market_data.mexc_pilot_run import EndpointVerificationPlanV1

        if self.contract_version != OFFICIAL_EVIDENCE_COMPATIBILITY_VERSION:
            raise MexcOfficialEvidenceContractError(
                "official_compatibility_version_mismatch"
            )
        if type(self.verification_plan) is not EndpointVerificationPlanV1:
            raise MexcOfficialEvidenceContractError(
                "official_compatibility_requires_exact_frozen_plan"
            )
        plan = self.verification_plan
        _digest(self.verification_plan_hash, field="official_compatibility_plan_hash")
        if self.verification_plan_hash != plan.plan_hash:
            raise MexcOfficialEvidenceContractError(
                "official_compatibility_plan_hash_mismatch"
            )
        if type(self.max_network_attempts) is not int:
            raise MexcOfficialEvidenceContractError(
                "official_compatibility_max_network_attempts_must_be_exact_int"
            )
        if self.max_network_attempts != 2:
            raise MexcOfficialEvidenceBudgetStop(
                "official_compatibility_requires_exactly_two_network_attempts"
            )
        if self.max_network_attempts != plan.max_network_attempts:
            raise MexcOfficialEvidenceBudgetStop(
                "official_compatibility_requires_exactly_two_network_attempts"
            )
        for field in (
            "max_total_raw_body_bytes",
            "max_total_storage_bytes",
            "max_runtime_us",
            "reserved_live_raw_body_bytes",
            "reserved_live_storage_bytes",
            "reserved_live_runtime_us",
            "residual_official_raw_body_bytes",
            "residual_official_storage_bytes",
            "residual_official_runtime_us",
        ):
            _strict_int(
                getattr(self, field),
                field=f"official_compatibility_{field}",
                minimum=0,
            )
        limits = plan.probe_request.resource_limits
        exact_values = {
            "max_total_raw_body_bytes": plan.max_total_raw_body_bytes,
            "max_total_storage_bytes": plan.max_total_storage_bytes,
            "max_runtime_us": plan.max_runtime_us,
            "reserved_live_raw_body_bytes": limits.max_total_raw_body_bytes,
            "reserved_live_storage_bytes": limits.max_logical_storage_bytes,
            "reserved_live_runtime_us": limits.max_collection_runtime_us,
        }
        if any(getattr(self, field) != value for field, value in exact_values.items()):
            raise MexcOfficialEvidenceBudgetStop(
                "official_compatibility_live_reservation_is_not_frozen_probe_limit"
            )
        triples = (
            (
                "raw_body_bytes",
                self.max_total_raw_body_bytes,
                self.reserved_live_raw_body_bytes,
                self.residual_official_raw_body_bytes,
            ),
            (
                "storage_bytes",
                self.max_total_storage_bytes,
                self.reserved_live_storage_bytes,
                self.residual_official_storage_bytes,
            ),
            (
                "runtime_us",
                self.max_runtime_us,
                self.reserved_live_runtime_us,
                self.residual_official_runtime_us,
            ),
        )
        for field, total, reserved, residual in triples:
            _strict_int(total, field=f"official_total_{field}", minimum=1)
            _strict_int(reserved, field=f"official_reserved_live_{field}", minimum=1)
            _strict_int(residual, field=f"official_residual_{field}", minimum=0)
            if residual != total - reserved:
                raise MexcOfficialEvidenceContractError(
                    f"official_residual_{field}_is_not_derived"
                )
            if residual <= 0:
                raise MexcOfficialEvidenceBudgetStop(
                    f"official_residual_{field}_is_nonpositive"
                )
        if (
            type(self.compatibility_status) is not str
            or self.compatibility_status != "offline_bundle_only_terminal_stop"
        ):
            raise MexcOfficialEvidenceContractError(
                "official_compatibility_status_mismatch"
            )
        if (
            type(self.terminal_compatible) is not bool
            or type(self.authority_status) is not str
            or self.authority_status != REVIEWED_FAKE_FIXTURE_ONLY
            or self.terminal_compatible is not False
        ):
            raise MexcOfficialEvidenceContractError(
                "official_compatibility_must_remain_fake_nonterminal"
            )

    @property
    def compatibility_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result["verification_plan"] = self.verification_plan.as_dict()
        return result

    @classmethod
    def from_dict(cls, payload: object) -> "OfficialEvidenceCompatibilityV1":
        from trading.market_data.mexc_pilot_run import EndpointVerificationPlanV1

        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="official_compatibility_schema_mismatch",
        )
        values["verification_plan"] = EndpointVerificationPlanV1.from_dict(
            values["verification_plan"]
        )
        try:
            return cls(**values)
        except TypeError as exc:
            raise MexcOfficialEvidenceContractError(
                "official_compatibility_reconstruction_failed"
            ) from exc


def assess_official_evidence_compatibility_v1(
    *,
    verification_plan: object,
) -> OfficialEvidenceCompatibilityV1:
    """Reserve the frozen one-bar request maxima, then derive official residuals."""

    from trading.market_data.mexc_pilot_run import EndpointVerificationPlanV1

    if type(verification_plan) is not EndpointVerificationPlanV1:
        raise MexcOfficialEvidenceContractError(
            "official_compatibility_requires_exact_frozen_plan"
        )
    limits = verification_plan.probe_request.resource_limits
    return OfficialEvidenceCompatibilityV1(
        verification_plan=verification_plan,
        verification_plan_hash=verification_plan.plan_hash,
        max_network_attempts=verification_plan.max_network_attempts,
        max_total_raw_body_bytes=verification_plan.max_total_raw_body_bytes,
        max_total_storage_bytes=verification_plan.max_total_storage_bytes,
        max_runtime_us=verification_plan.max_runtime_us,
        reserved_live_raw_body_bytes=limits.max_total_raw_body_bytes,
        reserved_live_storage_bytes=limits.max_logical_storage_bytes,
        reserved_live_runtime_us=limits.max_collection_runtime_us,
        residual_official_raw_body_bytes=(
            verification_plan.max_total_raw_body_bytes
            - limits.max_total_raw_body_bytes
        ),
        residual_official_storage_bytes=(
            verification_plan.max_total_storage_bytes - limits.max_logical_storage_bytes
        ),
        residual_official_runtime_us=(
            verification_plan.max_runtime_us - limits.max_collection_runtime_us
        ),
    )


def _relative_locator(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or "\\" in value
        or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise MexcOfficialEvidenceContractError(
            "official_bundle_locator_is_invalid"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise MexcOfficialEvidenceContractError(
            "official_bundle_locator_is_not_canonical"
        )
    for part in path.parts:
        if (
            any(character in '<>:"|?*' for character in part)
            or part.endswith((".", " "))
            or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED
        ):
            raise MexcOfficialEvidenceContractError(
                "official_bundle_locator_is_not_windows_safe"
            )
    return value


def _is_reparse(observed: os.stat_result) -> bool:
    attributes = getattr(observed, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _plain_mode(observed: os.stat_result, *, directory: bool) -> bool:
    predicate = stat.S_ISDIR if directory else stat.S_ISREG
    return predicate(observed.st_mode) and not _is_reparse(observed)


def _deadline_ns(
    deadline_monotonic_ns: int | None, *, operation_budget_us: int
) -> int:
    budget = _strict_int(
        operation_budget_us,
        field="official_bundle_operation_runtime_budget_us",
        minimum=1,
    )
    local = time.monotonic_ns() + min(OFFICIAL_IO_DEADLINE_US, budget) * 1_000
    if deadline_monotonic_ns is None:
        return local
    if type(deadline_monotonic_ns) is not int or deadline_monotonic_ns < 1:
        raise MexcOfficialEvidenceContractError(
            "official_bundle_deadline_is_invalid"
        )
    return min(local, deadline_monotonic_ns)


def _check_deadline(deadline_ns: int) -> None:
    if time.monotonic_ns() > deadline_ns:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_io_deadline_exceeded"
        )


def _absolute_root(value: str | os.PathLike[str]) -> Path:
    try:
        path = Path(os.path.abspath(os.fspath(value)))
    except (TypeError, ValueError, OSError) as exc:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_output_root_is_invalid"
        ) from exc
    if (
        not path.is_absolute()
        or path.parent == path
        or len(os.fspath(path)) > 4096
        or len(path.parts) > 64
    ):
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_output_root_is_too_broad_or_unbounded"
        )
    return path


def _validate_plain_existing_chain(path: Path, *, deadline_ns: int) -> None:
    chain: list[Path] = []
    current = path
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    for item in reversed(chain):
        _check_deadline(deadline_ns)
        try:
            observed = item.lstat()
        except OSError as exc:
            raise MexcOfficialEvidenceStorageStop(
                "official_bundle_directory_chain_is_missing"
            ) from exc
        if not _plain_mode(observed, directory=True):
            raise MexcOfficialEvidenceStorageStop(
                "official_bundle_directory_chain_has_reparse"
            )
        _reject_windows_named_streams(item, deadline_ns=deadline_ns)


def _ensure_plain_directory(path: Path, *, deadline_ns: int) -> None:
    chain: list[Path] = []
    current = path
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    for item in reversed(chain):
        _check_deadline(deadline_ns)
        try:
            observed = item.lstat()
        except FileNotFoundError:
            try:
                item.mkdir()
            except FileExistsError:
                pass
            except OSError as exc:
                raise MexcOfficialEvidenceStorageStop(
                    "official_bundle_directory_create_failed"
                ) from exc
            try:
                observed = item.lstat()
            except OSError as exc:
                raise MexcOfficialEvidenceStorageStop(
                    "official_bundle_created_directory_probe_failed"
                ) from exc
        except OSError as exc:
            raise MexcOfficialEvidenceStorageStop(
                "official_bundle_directory_probe_failed"
            ) from exc
        if not _plain_mode(observed, directory=True):
            raise MexcOfficialEvidenceStorageStop(
                "official_bundle_directory_chain_has_reparse"
            )
        _reject_windows_named_streams(item, deadline_ns=deadline_ns)


def _reject_windows_short_alias(path: Path) -> None:
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    current = path
    while True:
        try:
            current.lstat()
            break
        except FileNotFoundError:
            if current.parent == current:
                raise MexcOfficialEvidenceStorageStop(
                    "official_bundle_long_path_probe_failed"
                )
            current = current.parent
        except OSError as exc:
            raise MexcOfficialEvidenceStorageStop(
                "official_bundle_long_path_probe_failed"
            ) from exc
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_long = kernel32.GetLongPathNameW
    get_long.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_long.restype = wintypes.DWORD
    required = get_long(os.fspath(current), None, 0)
    if required == 0:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_long_path_probe_failed"
        )
    buffer = ctypes.create_unicode_buffer(required + 1)
    written = get_long(os.fspath(current), buffer, len(buffer))
    if written == 0 or written >= len(buffer):
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_long_path_probe_failed"
        )
    requested = os.path.normcase(os.path.abspath(os.fspath(current))).casefold()
    expanded = os.path.normcase(os.path.abspath(buffer.value)).casefold()
    if requested != expanded:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_short_path_alias_is_forbidden"
        )


def _reject_windows_named_streams(path: Path, *, deadline_ns: int) -> None:
    if os.name != "nt":
        return
    _check_deadline(deadline_ns)
    import ctypes
    from ctypes import wintypes

    class _FindStreamData(ctypes.Structure):
        _fields_ = [
            ("stream_size", ctypes.c_longlong),
            ("stream_name", wintypes.WCHAR * 296),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(_FindStreamData),
        wintypes.DWORD,
    ]
    find_first.restype = wintypes.HANDLE
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_FindStreamData)]
    find_next.restype = wintypes.BOOL
    find_close = kernel32.FindClose
    find_close.argtypes = [wintypes.HANDLE]
    find_close.restype = wintypes.BOOL
    data = _FindStreamData()
    handle = find_first(os.fspath(path), 0, ctypes.byref(data), 0)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        if ctypes.get_last_error() == 38:
            return
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_stream_enumeration_failed"
        )
    count = 0
    try:
        while True:
            _check_deadline(deadline_ns)
            count += 1
            if data.stream_name.casefold() != "::$data" or count > 1:
                raise MexcOfficialEvidenceStorageStop(
                    "official_bundle_named_stream_is_forbidden"
                )
            if find_next(handle, ctypes.byref(data)):
                continue
            if ctypes.get_last_error() != 38:
                raise MexcOfficialEvidenceStorageStop(
                    "official_bundle_stream_enumeration_failed"
                )
            break
    finally:
        find_close(handle)


def _stable_signature(observed: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_mode,
        observed.st_nlink,
    )


def _target_for(root: Path, locator: str) -> Path:
    parts = PurePosixPath(_relative_locator(locator)).parts
    target = root.joinpath(*parts)
    try:
        common = os.path.commonpath((os.fspath(root), os.fspath(target)))
    except ValueError as exc:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_locator_escapes_root"
        ) from exc
    if os.path.normcase(common) != os.path.normcase(os.fspath(root)):
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_locator_escapes_root"
        )
    return target


def _read_plain_file(
    path: Path, *, max_bytes: int, deadline_ns: int
) -> bytes:
    _check_deadline(deadline_ns)
    _validate_plain_existing_chain(path.parent, deadline_ns=deadline_ns)
    try:
        before = path.lstat()
    except OSError as exc:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_artifact_is_missing"
        ) from exc
    if not _plain_mode(before, directory=False) or before.st_nlink != 1:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_artifact_is_aliased"
        )
    if before.st_size < 1 or before.st_size > max_bytes:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_artifact_exceeds_bound"
        )
    _reject_windows_named_streams(path, deadline_ns=deadline_ns)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_artifact_open_failed"
        ) from exc
    chunks: list[bytes] = []
    count = 0
    try:
        try:
            opened = os.fstat(descriptor)
            if (
                not _plain_mode(opened, directory=False)
                or opened.st_nlink != 1
                or _stable_signature(opened) != _stable_signature(before)
            ):
                raise MexcOfficialEvidenceStorageStop(
                    "official_bundle_artifact_changed_before_read"
                )
            while True:
                _check_deadline(deadline_ns)
                chunk = os.read(
                    descriptor,
                    min(OFFICIAL_IO_CHUNK_BYTES, max_bytes + 1 - count),
                )
                _check_deadline(deadline_ns)
                if not chunk:
                    break
                chunks.append(chunk)
                count += len(chunk)
                if count > max_bytes:
                    raise MexcOfficialEvidenceStorageStop(
                        "official_bundle_artifact_exceeds_bound"
                    )
            after_open = os.fstat(descriptor)
        except MexcOfficialEvidenceError:
            raise
        except OSError as exc:
            raise MexcOfficialEvidenceStorageStop(
                "official_bundle_artifact_read_failed"
            ) from exc
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            raise MexcOfficialEvidenceStorageStop(
                "official_bundle_artifact_close_failed"
            ) from exc
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_artifact_vanished"
        ) from exc
    if (
        _stable_signature(after_open) != _stable_signature(opened)
        or _stable_signature(after_path) != _stable_signature(opened)
        or count != opened.st_size
    ):
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_artifact_changed_during_read"
        )
    _reject_windows_named_streams(path, deadline_ns=deadline_ns)
    try:
        final_path = path.lstat()
    except OSError as exc:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_artifact_final_probe_failed"
        ) from exc
    if _stable_signature(final_path) != _stable_signature(opened):
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_artifact_changed_after_stream_check"
        )
    _check_deadline(deadline_ns)
    return b"".join(chunks)


def _write_create_new(path: Path, body: bytes, *, deadline_ns: int) -> None:
    if type(body) is not bytes or not body:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_create_new_body_is_empty"
        )
    _ensure_plain_directory(path.parent, deadline_ns=deadline_ns)
    _check_deadline(deadline_ns)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_create_new_slot_preexists"
        ) from exc
    except OSError as exc:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_create_new_open_failed"
        ) from exc
    written = 0
    try:
        try:
            opened = os.fstat(descriptor)
            if not _plain_mode(opened, directory=False) or opened.st_nlink != 1:
                raise MexcOfficialEvidenceStorageStop(
                    "official_bundle_created_artifact_is_aliased"
                )
            while written < len(body):
                _check_deadline(deadline_ns)
                end = min(written + OFFICIAL_IO_CHUNK_BYTES, len(body))
                count = os.write(descriptor, body[written:end])
                if count < 1:
                    raise MexcOfficialEvidenceStorageStop(
                        "official_bundle_create_new_short_write"
                    )
                written += count
            os.fsync(descriptor)
            after_open = os.fstat(descriptor)
        except MexcOfficialEvidenceError:
            raise
        except OSError as exc:
            raise MexcOfficialEvidenceStorageStop(
                "official_bundle_create_new_write_failed"
            ) from exc
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            raise MexcOfficialEvidenceStorageStop(
                "official_bundle_create_new_close_failed"
            ) from exc
    try:
        visible = path.lstat()
    except OSError as exc:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_created_artifact_vanished"
        ) from exc
    if (
        written != len(body)
        or not _plain_mode(visible, directory=False)
        or visible.st_nlink != 1
        or (after_open.st_dev, after_open.st_ino) != (visible.st_dev, visible.st_ino)
        or visible.st_size != len(body)
    ):
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_created_artifact_identity_mismatch"
        )
    _reject_windows_named_streams(path, deadline_ns=deadline_ns)


def _exact_directory_names(path: Path, *, expected: tuple[str, ...], deadline_ns: int) -> None:
    _check_deadline(deadline_ns)
    _validate_plain_existing_chain(path, deadline_ns=deadline_ns)
    try:
        names_list: list[str] = []
        with os.scandir(path) as iterator:
            for item in iterator:
                _check_deadline(deadline_ns)
                names_list.append(item.name)
                if len(names_list) > 3:
                    raise MexcOfficialEvidenceStorageStop(
                        "official_bundle_directory_entry_cap_exceeded"
                    )
        names = tuple(sorted(names_list))
    except MexcOfficialEvidenceError:
        raise
    except OSError as exc:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_directory_scan_failed"
        ) from exc
    _check_deadline(deadline_ns)
    _reject_windows_named_streams(path, deadline_ns=deadline_ns)
    if names != tuple(sorted(expected)):
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_directory_contains_residue"
        )


def _validate_bundle_semantics(
    *,
    raw_body: bytes,
    attempt: OfficialReferenceHttpAttemptV1,
    evidence: OfficialDocumentEvidenceV1,
    prepared_request: OfficialReferencePreparedRequestV1,
) -> None:
    if type(raw_body) is not bytes:
        raise MexcOfficialEvidenceContractError(
            "official_bundle_raw_body_must_be_exact_bytes"
        )
    if attempt.outcome != "complete" or attempt.body_complete is not True:
        raise MexcOfficialEvidenceContractError(
            "official_bundle_incomplete_attempt_is_unsupported"
        )
    if (
        attempt.verification_plan_hash != prepared_request.verification_plan_hash
        or evidence.verification_plan_hash != attempt.verification_plan_hash
        or attempt.prepared_request_hash != prepared_request.prepared_request_hash
        or evidence.prepared_request_hash != prepared_request.prepared_request_hash
        or evidence.attempt_receipt_hash != attempt.attempt_receipt_hash
        or evidence.attempt_receipt_byte_count
        != len(_canonical_json_bytes(attempt.as_dict()))
        or evidence.raw_body_relative_path != attempt.raw_body_relative_path
        or evidence.raw_body_byte_count != len(raw_body)
        or evidence.raw_body_sha256 != hashlib.sha256(raw_body).hexdigest()
        or attempt.raw_body_byte_count != len(raw_body)
        or attempt.raw_body_sha256 != hashlib.sha256(raw_body).hexdigest()
        or evidence.parser_contract_version != prepared_request.parser_contract_version
        or evidence.parser_contract_hash != prepared_request.parser_contract_hash
        or evidence.reader_contract_version != OFFICIAL_DOCUMENT_READER_VERSION
        or evidence.reader_contract_hash
        != mexc_endpoint_official_evidence_contract_hash()
        or evidence.observed_body_fetched_at_us != attempt.body_eof_at_us
        or evidence.observed_body_fetched_monotonic_us
        != attempt.body_eof_monotonic_us
        or attempt.endpoint_runner_contract_version
        != prepared_request.endpoint_runner_contract_version
        or attempt.endpoint_runner_contract_hash
        != prepared_request.endpoint_runner_contract_hash
    ):
        raise MexcOfficialEvidenceContractError(
            "official_bundle_cross_binding_mismatch"
        )
    derived = _verify_and_derive_support_scope(raw_body, evidence.claims)
    if derived != evidence.support_scope:
        raise MexcOfficialEvidenceContractError(
            "official_bundle_support_scope_was_not_rederived"
        )


def _build_bundle(
    *,
    raw_body: bytes,
    receipt_bytes: bytes,
    evidence_bytes: bytes,
    attempt: OfficialReferenceHttpAttemptV1,
    evidence: OfficialDocumentEvidenceV1,
) -> OfficialEvidenceBundleV1:
    locators = official_bundle_relative_paths_v1(attempt.verification_plan_hash)
    bodies = (raw_body, receipt_bytes, evidence_bytes)
    roles = ("raw_body", "attempt_receipt", "semantic_evidence")
    return OfficialEvidenceBundleV1(
        verification_plan_hash=attempt.verification_plan_hash,
        prepared_request_hash=attempt.prepared_request_hash,
        attempt_receipt_hash=attempt.attempt_receipt_hash,
        evidence_hash=evidence.evidence_hash,
        raw_body_sha256=evidence.raw_body_sha256,
        files=tuple(
            OfficialEvidenceBundleFileV1(
                role=role,
                relative_path=locator,
                artifact_sha256=hashlib.sha256(body).hexdigest(),
                byte_count=len(body),
            )
            for role, locator, body in zip(roles, locators, bodies)
        ),
    )


def _require_prewrite_canonical_roundtrip(
    *,
    attempt: OfficialReferenceHttpAttemptV1,
    receipt_bytes: bytes,
    evidence: OfficialDocumentEvidenceV1,
    evidence_bytes: bytes,
    bundle: OfficialEvidenceBundleV1,
) -> None:
    """Reject any wire malleability before the first filesystem side effect."""

    reloaded_attempt = OfficialReferenceHttpAttemptV1.from_dict(
        parse_canonical_json_lf_v1(
            receipt_bytes,
            max_bytes=OFFICIAL_RECEIPT_MAX_BYTES,
        )
    )
    reloaded_evidence = OfficialDocumentEvidenceV1.from_dict(
        parse_canonical_json_lf_v1(
            evidence_bytes,
            max_bytes=OFFICIAL_EVIDENCE_MAX_BYTES,
        )
    )
    bundle_bytes = _canonical_json_bytes(bundle.as_dict())
    reloaded_bundle = OfficialEvidenceBundleV1.from_dict(
        parse_canonical_json_lf_v1(
            bundle_bytes,
            max_bytes=OFFICIAL_EVIDENCE_MAX_BYTES,
        )
    )
    if (
        reloaded_attempt != attempt
        or reloaded_evidence != evidence
        or reloaded_bundle != bundle
        or hashlib.sha256(receipt_bytes).hexdigest()
        != attempt.attempt_receipt_hash
        or hashlib.sha256(evidence_bytes).hexdigest() != evidence.evidence_hash
        or reloaded_bundle.raw_body_sha256 != reloaded_evidence.raw_body_sha256
    ):
        raise MexcOfficialEvidenceContractError(
            "official_bundle_prewrite_canonical_roundtrip_mismatch"
        )


def publish_official_evidence_bundle_v1(
    *,
    output_root: str | os.PathLike[str],
    raw_body: bytes,
    attempt: OfficialReferenceHttpAttemptV1,
    evidence: OfficialDocumentEvidenceV1,
    prepared_request: OfficialReferencePreparedRequestV1,
    compatibility: OfficialEvidenceCompatibilityV1,
    deadline_monotonic_ns: int | None = None,
) -> OfficialEvidenceBundleV1:
    """Create the exact immutable three-file fake bundle and freshly reload it."""

    if type(raw_body) is not bytes:
        raise MexcOfficialEvidenceContractError(
            "official_bundle_publish_raw_body_must_be_exact_bytes"
        )
    if (
        type(attempt) is not OfficialReferenceHttpAttemptV1
        or type(evidence) is not OfficialDocumentEvidenceV1
        or type(prepared_request) is not OfficialReferencePreparedRequestV1
        or type(compatibility) is not OfficialEvidenceCompatibilityV1
    ):
        raise MexcOfficialEvidenceContractError(
            "official_bundle_publish_inputs_are_not_exact_contracts"
        )
    if compatibility.verification_plan_hash != attempt.verification_plan_hash:
        raise MexcOfficialEvidenceContractError(
            "official_bundle_compatibility_binding_mismatch"
        )
    deadline_ns = _deadline_ns(
        deadline_monotonic_ns,
        operation_budget_us=compatibility.residual_official_runtime_us,
    )
    _check_deadline(deadline_ns)
    if len(raw_body) > compatibility.residual_official_raw_body_bytes:
        raise MexcOfficialEvidenceBudgetStop(
            "official_bundle_raw_body_budget_exceeded"
        )
    _validate_bundle_semantics(
        raw_body=raw_body,
        attempt=attempt,
        evidence=evidence,
        prepared_request=prepared_request,
    )
    observed_runtime_us = (
        evidence.reload_completed_monotonic_us
        - attempt.request_started_monotonic_us
    )
    if observed_runtime_us < 0 or observed_runtime_us > compatibility.residual_official_runtime_us:
        raise MexcOfficialEvidenceBudgetStop(
            "official_bundle_observed_runtime_budget_exceeded"
        )
    _check_deadline(deadline_ns)
    receipt_bytes = _canonical_json_bytes(attempt.as_dict())
    evidence_bytes = evidence.canonical_lf_bytes
    if len(receipt_bytes) > OFFICIAL_RECEIPT_MAX_BYTES:
        raise MexcOfficialEvidenceBudgetStop(
            "official_bundle_attempt_receipt_cap_exceeded"
        )
    if len(evidence_bytes) > OFFICIAL_EVIDENCE_MAX_BYTES:
        raise MexcOfficialEvidenceBudgetStop(
            "official_bundle_evidence_cap_exceeded"
        )
    bundle = _build_bundle(
        raw_body=raw_body,
        receipt_bytes=receipt_bytes,
        evidence_bytes=evidence_bytes,
        attempt=attempt,
        evidence=evidence,
    )
    _require_prewrite_canonical_roundtrip(
        attempt=attempt,
        receipt_bytes=receipt_bytes,
        evidence=evidence,
        evidence_bytes=evidence_bytes,
        bundle=bundle,
    )
    _check_deadline(deadline_ns)
    if bundle.total_storage_bytes > compatibility.residual_official_storage_bytes:
        raise MexcOfficialEvidenceBudgetStop(
            "official_bundle_storage_budget_exceeded"
        )
    root = _absolute_root(output_root)
    _reject_windows_short_alias(root)
    _ensure_plain_directory(root, deadline_ns=deadline_ns)
    _reject_windows_short_alias(root)
    locators = official_bundle_relative_paths_v1(attempt.verification_plan_hash)
    targets = tuple(_target_for(root, locator) for locator in locators)
    bundle_dir = targets[0].parent
    _ensure_plain_directory(bundle_dir, deadline_ns=deadline_ns)
    _exact_directory_names(bundle_dir, expected=(), deadline_ns=deadline_ns)
    for target, body in zip(targets, (raw_body, receipt_bytes, evidence_bytes)):
        _write_create_new(target, body, deadline_ns=deadline_ns)
    _exact_directory_names(
        bundle_dir,
        expected=tuple(path.name for path in targets),
        deadline_ns=deadline_ns,
    )
    reloaded = reload_official_evidence_bundle_v1(
        output_root=root,
        verification_plan_hash=attempt.verification_plan_hash,
        prepared_request=prepared_request,
        compatibility=compatibility,
        deadline_monotonic_ns=deadline_ns,
    )
    if reloaded.bundle_hash != bundle.bundle_hash:
        raise MexcOfficialEvidenceStorageStop(
            "official_bundle_fresh_reload_mismatch"
        )
    return reloaded


def reload_official_evidence_bundle_v1(
    *,
    output_root: str | os.PathLike[str],
    verification_plan_hash: str,
    prepared_request: OfficialReferencePreparedRequestV1,
    compatibility: OfficialEvidenceCompatibilityV1,
    deadline_monotonic_ns: int | None = None,
) -> OfficialEvidenceBundleV1:
    """Freshly reload all three exact files and rederive every semantic claim."""

    if type(prepared_request) is not OfficialReferencePreparedRequestV1 or type(
        compatibility
    ) is not OfficialEvidenceCompatibilityV1:
        raise MexcOfficialEvidenceContractError(
            "official_bundle_reload_inputs_are_not_exact_contracts"
        )
    plan_hash = _digest(verification_plan_hash, field="official_bundle_reload_plan_hash")
    if (
        prepared_request.verification_plan_hash != plan_hash
        or compatibility.verification_plan_hash != plan_hash
    ):
        raise MexcOfficialEvidenceContractError(
            "official_bundle_reload_binding_mismatch"
        )
    deadline_ns = _deadline_ns(
        deadline_monotonic_ns,
        operation_budget_us=compatibility.residual_official_runtime_us,
    )
    _check_deadline(deadline_ns)
    root = _absolute_root(output_root)
    _reject_windows_short_alias(root)
    _validate_plain_existing_chain(root, deadline_ns=deadline_ns)
    locators = official_bundle_relative_paths_v1(plan_hash)
    targets = tuple(_target_for(root, locator) for locator in locators)
    _exact_directory_names(
        targets[0].parent,
        expected=tuple(path.name for path in targets),
        deadline_ns=deadline_ns,
    )
    raw_body = _read_plain_file(
        targets[0],
        max_bytes=compatibility.residual_official_raw_body_bytes,
        deadline_ns=deadline_ns,
    )
    receipt_bytes = _read_plain_file(
        targets[1], max_bytes=OFFICIAL_RECEIPT_MAX_BYTES, deadline_ns=deadline_ns
    )
    evidence_bytes = _read_plain_file(
        targets[2], max_bytes=OFFICIAL_EVIDENCE_MAX_BYTES, deadline_ns=deadline_ns
    )
    attempt = OfficialReferenceHttpAttemptV1.from_dict(
        parse_canonical_json_lf_v1(
            receipt_bytes, max_bytes=OFFICIAL_RECEIPT_MAX_BYTES
        )
    )
    evidence = OfficialDocumentEvidenceV1.from_dict(
        parse_canonical_json_lf_v1(
            evidence_bytes, max_bytes=OFFICIAL_EVIDENCE_MAX_BYTES
        )
    )
    _validate_bundle_semantics(
        raw_body=raw_body,
        attempt=attempt,
        evidence=evidence,
        prepared_request=prepared_request,
    )
    observed_runtime_us = (
        evidence.reload_completed_monotonic_us
        - attempt.request_started_monotonic_us
    )
    if observed_runtime_us < 0 or observed_runtime_us > compatibility.residual_official_runtime_us:
        raise MexcOfficialEvidenceBudgetStop(
            "official_bundle_observed_runtime_budget_exceeded"
        )
    bundle = _build_bundle(
        raw_body=raw_body,
        receipt_bytes=receipt_bytes,
        evidence_bytes=evidence_bytes,
        attempt=attempt,
        evidence=evidence,
    )
    if bundle.total_storage_bytes > compatibility.residual_official_storage_bytes:
        raise MexcOfficialEvidenceBudgetStop(
            "official_bundle_reloaded_storage_budget_exceeded"
        )
    _exact_directory_names(
        targets[0].parent,
        expected=tuple(path.name for path in targets),
        deadline_ns=deadline_ns,
    )
    _check_deadline(deadline_ns)
    return bundle


def require_terminal_compatible_official_evidence_v1(
    *,
    bundle: OfficialEvidenceBundleV1,
    evidence: OfficialDocumentEvidenceV1,
    compatibility: OfficialEvidenceCompatibilityV1,
) -> NoReturn:
    """Always STOP: v1 is fake-only and is outside the frozen pilot namespace."""

    if (
        type(bundle) is not OfficialEvidenceBundleV1
        or type(evidence) is not OfficialDocumentEvidenceV1
        or type(compatibility) is not OfficialEvidenceCompatibilityV1
    ):
        raise MexcOfficialEvidenceContractError(
            "official_terminal_adapter_inputs_are_not_exact_contracts"
        )
    if (
        bundle.verification_plan_hash != evidence.verification_plan_hash
        or bundle.verification_plan_hash != compatibility.verification_plan_hash
        or bundle.prepared_request_hash != evidence.prepared_request_hash
        or bundle.attempt_receipt_hash != evidence.attempt_receipt_hash
        or bundle.evidence_hash != evidence.evidence_hash
        or bundle.raw_body_sha256 != evidence.raw_body_sha256
        or tuple(item.byte_count for item in bundle.files)
        != (
            evidence.raw_body_byte_count,
            evidence.attempt_receipt_byte_count,
            len(evidence.canonical_lf_bytes),
        )
    ):
        raise MexcOfficialEvidenceContractError(
            "official_terminal_adapter_binding_mismatch"
        )
    raise MexcOfficialEvidenceTerminalStop(
        "official_evidence_v1_is_not_terminal_compatible",
        _TERMINAL_BLOCKERS,
        "reviewed_fake_structural_nonterminal",
    )


def _pilot_run_dependency() -> tuple[str, str]:
    from trading.market_data.mexc_pilot_run import (
        PILOT_RUN_CONTRACT_VERSION,
        pilot_run_contract_hash,
    )

    return PILOT_RUN_CONTRACT_VERSION, pilot_run_contract_hash()


_PILOT_RUN_DEPENDENCY_VERSION, _PILOT_RUN_DEPENDENCY_HASH = (
    _pilot_run_dependency()
)
_PILOT_OUTPUT_LAYOUT_DEPENDENCY_VERSION, _PILOT_OUTPUT_LAYOUT_DEPENDENCY_HASH = (
    _validated_pilot_output_layout_dependency()
)

# Freeze not only the layout digest but also the exact canonical official
# namespace helper results used by this store.
_LAYOUT_ZERO_PLAN_ROOT = _pilot_output_layout.derive_official_bundle_root_v1(
    "0" * 64
)
_LAYOUT_ZERO_PLAN_LOCATORS = (
    _pilot_output_layout.derive_official_bundle_locators_v1("0" * 64)
)
if (
    type(_LAYOUT_ZERO_PLAN_ROOT) is not str
    or _LAYOUT_ZERO_PLAN_ROOT != "endpoint-evidence/" + "0" * 64 + "/official"
    or type(_LAYOUT_ZERO_PLAN_LOCATORS) is not tuple
    or _LAYOUT_ZERO_PLAN_LOCATORS
    != (
        _LAYOUT_ZERO_PLAN_ROOT + "/attempt-000.body.bin",
        _LAYOUT_ZERO_PLAN_ROOT + "/attempt-000.receipt.json",
        _LAYOUT_ZERO_PLAN_ROOT + "/evidence.json",
    )
):
    raise MexcOfficialEvidenceContractError(
        "official_evidence_pilot_output_layout_helper_vector_drift"
    )


_CONTRACT_SCHEMA = {
    "contract_versions": {
        "prepared_request": OFFICIAL_REFERENCE_REQUEST_VERSION,
        "http_attempt": OFFICIAL_REFERENCE_HTTP_ATTEMPT_VERSION,
        "span_claim": OFFICIAL_DOCUMENT_SPAN_CLAIM_VERSION,
        "semantic_evidence": OFFICIAL_DOCUMENT_EVIDENCE_VERSION,
        "reader": OFFICIAL_DOCUMENT_READER_VERSION,
        "bundle_file": OFFICIAL_EVIDENCE_BUNDLE_FILE_VERSION,
        "bundle": OFFICIAL_EVIDENCE_BUNDLE_VERSION,
        "store": OFFICIAL_EVIDENCE_STORE_VERSION,
        "compatibility": OFFICIAL_EVIDENCE_COMPATIBILITY_VERSION,
        "strict_adapter": OFFICIAL_EVIDENCE_STRICT_ADAPTER_VERSION,
    },
    "field_sets": {
        "prepared_request": list(OfficialReferencePreparedRequestV1.__dataclass_fields__),
        "http_attempt": list(OfficialReferenceHttpAttemptV1.__dataclass_fields__),
        "span_claim": list(OfficialDocumentSpanClaimV1.__dataclass_fields__),
        "semantic_evidence": list(OfficialDocumentEvidenceV1.__dataclass_fields__),
        "bundle_file": list(OfficialEvidenceBundleFileV1.__dataclass_fields__),
        "bundle": list(OfficialEvidenceBundleV1.__dataclass_fields__),
        "compatibility": list(OfficialEvidenceCompatibilityV1.__dataclass_fields__),
    },
    "dependency_bindings": {
        "pilot_run": {
            "version": _PILOT_RUN_DEPENDENCY_VERSION,
            "hash": _PILOT_RUN_DEPENDENCY_HASH,
            "status": "exact_frozen_dependency",
        },
        "candidate_endpoint": {
            "version": CANDIDATE_CONTRACT_VERSION,
            "hash": CANDIDATE_CONTRACT_HASH,
            "status": "exact_frozen_dependency",
        },
        "pilot_output_layout": {
            "version": _PILOT_OUTPUT_LAYOUT_DEPENDENCY_VERSION,
            "hash": _PILOT_OUTPUT_LAYOUT_DEPENDENCY_HASH,
            "status": "exact_frozen_dependency_and_canonical_helpers",
        },
    },
    "public_api_signatures": {
        "derive_official_bundle_root_v1": (
            "(verification_plan_hash:str)->str"
        ),
        "official_bundle_relative_paths_v1": (
            "(verification_plan_hash:str)->tuple[str,str,str]"
        ),
        "parse_canonical_json_lf_v1": (
            "(raw:bytes,*,max_bytes:int)->object"
        ),
        "build_exact_span_claims_v1": "(raw_body:bytes)->tuple[OfficialDocumentSpanClaimV1,...]",
        "read_official_document_evidence_v1": (
            "(*,raw_body:bytes,attempt:OfficialReferenceHttpAttemptV1,"
            "prepared_request:OfficialReferencePreparedRequestV1,"
            "claims:tuple[OfficialDocumentSpanClaimV1,...],"
            "parser_contract_version:str,parser_contract_hash:str,"
            "reader_contract_hash:str,parse_started_at_us:int,"
            "parse_completed_at_us:int,parse_started_monotonic_us:int,"
            "parse_completed_monotonic_us:int,reload_completed_at_us:int,"
            "reload_completed_monotonic_us:int)->OfficialDocumentEvidenceV1"
        ),
        "assess_official_evidence_compatibility_v1": (
            "(*,verification_plan:EndpointVerificationPlanV1)"
            "->OfficialEvidenceCompatibilityV1"
        ),
        "publish_official_evidence_bundle_v1": (
            "(*,output_root:path,raw_body:bytes,attempt:OfficialReferenceHttpAttemptV1,"
            "evidence:OfficialDocumentEvidenceV1,"
            "prepared_request:OfficialReferencePreparedRequestV1,"
            "compatibility:OfficialEvidenceCompatibilityV1,"
            "deadline_monotonic_ns:int|None=None)->OfficialEvidenceBundleV1"
        ),
        "reload_official_evidence_bundle_v1": (
            "(*,output_root:path,verification_plan_hash:str,"
            "prepared_request:OfficialReferencePreparedRequestV1,"
            "compatibility:OfficialEvidenceCompatibilityV1,"
            "deadline_monotonic_ns:int|None=None)->OfficialEvidenceBundleV1"
        ),
        "require_terminal_compatible_official_evidence_v1": (
            "(*,bundle:OfficialEvidenceBundleV1,evidence:OfficialDocumentEvidenceV1,"
            "compatibility:OfficialEvidenceCompatibilityV1)->NoReturn"
        ),
        "official_evidence_contract_descriptor_v1": "()->dict[str,object]",
        "mexc_endpoint_official_evidence_contract_hash": "()->sha256_hex",
    },
    "reference": {
        "url": OFFICIAL_REFERENCE_URL,
        "scheme": OFFICIAL_REFERENCE_SCHEME,
        "host": OFFICIAL_REFERENCE_HOST,
        "port": OFFICIAL_REFERENCE_PORT,
        "path": OFFICIAL_REFERENCE_PATH,
        "reference_id": OFFICIAL_REFERENCE_ID,
        "method": "GET",
        "headers": [list(item) for item in _REQUEST_HEADERS],
        "empty_body_sha256": EMPTY_SHA256,
        "tls_verify": True,
        "tls_sni": OFFICIAL_REFERENCE_HOST,
        "allow_redirects": False,
        "trust_env": False,
        "proxy_cookie_netrc_auth": "disabled",
    },
    "candidate": {
        "version": CANDIDATE_CONTRACT_VERSION,
        "hash": CANDIDATE_CONTRACT_HASH,
        "method": CANDIDATE_METHOD,
        "url_template": CANDIDATE_URL_TEMPLATE,
        "path_template": CANDIDATE_PATH_TEMPLATE,
        "ordered_query": list(CANDIDATE_QUERY_ORDER),
        "migration": [LEGACY_FUTURES_HOST, CANDIDATE_FUTURES_HOST],
    },
    "semantic_reader": {
        "caller_support_boolean": "absent_and_never_trusted",
        "claims": {role: literal.decode("ascii") for role, literal in _CLAIM_LITERALS.items()},
        "migration_roles": sorted(_MIGRATION_ROLES),
        "full_candidate_roles": sorted(_FULL_CANDIDATE_ROLES),
        "support_scopes": ["domain_migration_only", "full_candidate_contract"],
        "raw_bytes_reparsed_on_build_and_reload": True,
        "statement_context": "unique_exact_standalone_lf_delimited_line",
        "caller_role_selection": False,
        "unclaimed_present_statement": "reject",
        "raw_body_pre_scan_hard_cap": OFFICIAL_RAW_BODY_HARD_CAP_BYTES,
        "official_evidence_hash": "sha256_exact_canonical_lf_evidence_json_bytes",
    },
    "bundle": {
        "root": "endpoint-evidence/<verification_plan_hash>/official",
        "files": [
            "attempt-000.body.bin",
            "attempt-000.receipt.json",
            "evidence.json",
        ],
        "publish": "bounded_create_new_no_overwrite_then_fresh_reload",
        "publish_order": [
            "validate_exact_bindings_and_budgets",
            "canonical_serialize_parse_and_roundtrip_attempt_evidence_bundle",
            "validate_static_plain_directory_chain_and_empty_official_directory",
            "create_new_raw_body",
            "create_new_attempt_receipt",
            "create_new_semantic_evidence",
            "validate_exact_directory_inventory",
            "fresh_reload_all_three",
            "reparse_raw_and_rederive_claims",
            "compare_bundle_hash",
        ],
        "reload_order": [
            "validate_exact_plan_and_compatibility_binding",
            "validate_static_plain_chain_and_exact_three_entry_inventory",
            "bounded_read_raw_body",
            "bounded_read_attempt_receipt",
            "bounded_read_semantic_evidence",
            "parse_exact_canonical_lf_json",
            "reparse_raw_and_rederive_claims",
            "validate_raw_storage_and_observed_runtime_caps",
            "final_exact_inventory_and_deadline_check",
        ],
        "receipt_cap_bytes": OFFICIAL_RECEIPT_MAX_BYTES,
        "evidence_cap_bytes": OFFICIAL_EVIDENCE_MAX_BYTES,
        "raw_body_hard_cap_bytes": OFFICIAL_RAW_BODY_HARD_CAP_BYTES,
        "chunk_bytes": OFFICIAL_IO_CHUNK_BYTES,
        "deadline_us": OFFICIAL_IO_DEADLINE_US,
        "hostile_static_paths": {
            "reparse_junction_symlink": "reject",
            "hardlink": "reject_nlink_not_one",
            "named_stream": "reject",
            "short_path_alias": "reject",
            "residue_or_preexisting_slot": "reject",
        },
        "concurrency_boundary": OFFICIAL_STORAGE_CONCURRENCY_BOUNDARY,
        "atomic_directory_snapshot": False,
        "hostile_concurrent_toctou_guarantee": False,
        "hostile_concurrent_toctou_terminal_blocker": (
            "hostile_concurrent_filesystem_toctou_boundary_unaccepted"
        ),
        "plan_root_sibling_inventory": (
            "not_scanned_delegated_to_future_exact_pinned_output_layout"
        ),
        "partial_publication": (
            "nontransactional_create_new_prefix_remains_nonresumable_terminal_stop"
        ),
        "relative_locator_grammar": (
            "canonical_posix_relative_no_empty_dot_dotdot_backslash_control_"
            "windows_illegal_trailing_dot_space_or_reserved_device"
        ),
    },
    "compatibility": {
        "max_network_attempts": 2,
        "live_reservations": (
            "exact_probe_resource_limits_max_total_raw_body_"
            "max_logical_storage_max_collection_runtime"
        ),
        "official_residuals": "exact_total_minus_reserved_live",
        "formulas": {
            "live_raw": "probe.resource_limits.max_total_raw_body_bytes",
            "live_storage": "probe.resource_limits.max_logical_storage_bytes",
            "live_runtime": "probe.resource_limits.max_collection_runtime_us",
            "official_raw": "plan.max_total_raw_body_bytes-live_raw",
            "official_storage": "plan.max_total_storage_bytes-live_storage",
            "official_runtime": "plan.max_runtime_us-live_runtime",
        },
        "publish_runtime_observation": (
            "evidence.reload_completed_monotonic_us-"
            "attempt.request_started_monotonic_us"
        ),
        "nonpositive_residual": "typed_stop",
    },
    "attempt_validation": {
        "operation": "official_reference_fetch",
        "ordinal": 0,
        "required_bindings": [
            "manifest_hash",
            "authorization_receipt_hash",
            "preflight_receipt_hash",
            "verification_plan_hash",
            "network_intent_hash",
            "endpoint_runner_contract_version_hash",
            "runtime_authority_binding_hash",
            "clock_domain_id",
            "tls_policy_version_hash",
            "trust_store_version_hash",
            "prepared_request_hash",
        ],
        "timeline_epoch": [
            "gate_checked_at_us",
            "request_started_at_us",
            "tls_validated_at_us",
            "headers_received_at_us",
            "body_eof_at_us",
            "connection_closed_at_us",
        ],
        "timeline_monotonic": [
            "gate_checked_monotonic_us",
            "request_started_monotonic_us",
            "tls_validated_monotonic_us",
            "headers_received_monotonic_us",
            "body_eof_monotonic_us",
            "connection_closed_monotonic_us",
        ],
        "terminal_progress_phases": list(_TERMINAL_PROGRESS_PHASES),
        "terminal_progress_event_prefixes": {
            phase: {
                "tls_validated": presence[0],
                "headers_received": presence[1],
                "body_eof": presence[2],
            }
            for phase, presence in _TERMINAL_PROGRESS_EVENT_PRESENCE.items()
        },
        "observed_prefix_rule": (
            "every_non_none_tls_timestamp_status_header_and_raw_fact_is_preserved"
        ),
        "outcome_state_matrix": {
            "complete": {
                "terminal_progress": ["body_eof"],
                "events": [True, True, True],
                "tls_facts": "required_pkix_true",
                "status": 200,
                "headers": "safe_content_type_required_identity_encoding",
                "body": "complete_nonempty_exact_length_sha",
                "safe_errors": [],
            },
            "incomplete_transport_error": {
                "terminal_progress": [
                    "before_tls_validation",
                    "tls_validated_before_headers",
                ],
                "events": "exact_prefix_selected_by_terminal_progress",
                "tls_facts": (
                    "absent_before_tls_or_full_validated_prefix_before_headers"
                ),
                "status": None,
                "headers": "empty",
                "body": "incomplete_empty_sha256_empty",
                "safe_errors": sorted(
                    _SAFE_ERROR_CODES_BY_OUTCOME["incomplete_transport_error"]
                ),
            },
            "incomplete_tls_error": {
                "terminal_progress": ["tls_validation_failed"],
                "events": [False, False, False],
                "tls_facts": (
                    "pkix_false_chain_absent_observed_version_and_peer_prefix_preserved"
                ),
                "status": None,
                "headers": "empty",
                "body": "incomplete_empty_sha256_empty",
                "safe_errors": sorted(
                    _SAFE_ERROR_CODES_BY_OUTCOME["incomplete_tls_error"]
                ),
            },
            "incomplete_http_body_error": {
                "terminal_progress": ["headers_received_before_body_eof"],
                "events": [True, True, False],
                "tls_facts": "required_pkix_true",
                "status": 200,
                "headers": "safe_content_type_required_identity_encoding",
                "body": "incomplete_partial_or_empty_exact_length_sha",
                "safe_errors": sorted(
                    _SAFE_ERROR_CODES_BY_OUTCOME["incomplete_http_body_error"]
                ),
            },
            "rejected_protocol": {
                "terminal_progress": ["headers_received_before_body_eof"],
                "events": [True, True, False],
                "tls_facts": "required_pkix_true",
                "status": "required_and_corroborates_safe_error",
                "headers": "safe_subset_only",
                "body": "incomplete_empty_sha256_empty",
                "safe_errors": sorted(
                    _SAFE_ERROR_CODES_BY_OUTCOME["rejected_protocol"]
                ),
            },
        },
        "safe_error_predicates": {
            "dns_resolution_failed": {
                "outcome": "incomplete_transport_error",
                "terminal_progress": ["before_tls_validation"],
                "predicate": "tls_status_headers_raw_absent",
            },
            "transport_connect_failed": {
                "outcome": "incomplete_transport_error",
                "terminal_progress": ["before_tls_validation"],
                "predicate": "tls_status_headers_raw_absent",
            },
            "transport_connection_closed": {
                "outcome": "incomplete_transport_error",
                "terminal_progress": [
                    "before_tls_validation",
                    "tls_validated_before_headers",
                ],
                "predicate": "phase_exact_tls_prefix_status_headers_raw_absent",
            },
            "transport_timeout": {
                "outcome": "incomplete_transport_error",
                "terminal_progress": [
                    "before_tls_validation",
                    "tls_validated_before_headers",
                ],
                "predicate": "phase_exact_tls_prefix_status_headers_raw_absent",
            },
            "tls_certificate_validation_failed": {
                "outcome": "incomplete_tls_error",
                "terminal_progress": ["tls_validation_failed"],
                "predicate": "peer_leaf_required_pkix_false_chain_absent",
            },
            "tls_sni_mismatch": {
                "outcome": "incomplete_tls_error",
                "terminal_progress": ["tls_validation_failed"],
                "predicate": "peer_leaf_required_pkix_false_chain_absent",
            },
            "tls_policy_rejected": {
                "outcome": "incomplete_tls_error",
                "terminal_progress": ["tls_validation_failed"],
                "predicate": "tls_version_required_pkix_false_chain_absent",
            },
            "tls_handshake_failed": {
                "outcome": "incomplete_tls_error",
                "terminal_progress": ["tls_validation_failed"],
                "predicate": (
                    "empty_or_observed_version_then_peer_prefix_pkix_false_chain_absent"
                ),
            },
            "body_cap_exceeded": {
                "outcome": "incomplete_http_body_error",
                "terminal_progress": ["headers_received_before_body_eof"],
                "predicate": "stored_raw_at_cap_or_canonical_declared_length_above_cap",
            },
            "content_length_mismatch": {
                "outcome": "incomplete_http_body_error",
                "terminal_progress": ["headers_received_before_body_eof"],
                "predicate": "canonical_declared_length_not_equal_stored_partial_length",
            },
            "http_body_eof_missing": {
                "outcome": "incomplete_http_body_error",
                "terminal_progress": ["headers_received_before_body_eof"],
                "predicate": "validated_http_200_html_identity_partial_or_empty_raw",
            },
            "http_body_read_failed": {
                "outcome": "incomplete_http_body_error",
                "terminal_progress": ["headers_received_before_body_eof"],
                "predicate": "validated_http_200_html_identity_partial_or_empty_raw",
            },
            "redirect_status_rejected": {
                "outcome": "rejected_protocol",
                "terminal_progress": ["headers_received_before_body_eof"],
                "predicate": "status_300_through_399_raw_empty",
            },
            "http_status_not_200": {
                "outcome": "rejected_protocol",
                "terminal_progress": ["headers_received_before_body_eof"],
                "predicate": "status_not_200_and_not_3xx_raw_empty",
            },
            "content_encoding_not_identity": {
                "outcome": "rejected_protocol",
                "terminal_progress": ["headers_received_before_body_eof"],
                "predicate": "status_200_content_encoding_present_nonidentity_raw_empty",
            },
            "content_type_not_official_html": {
                "outcome": "rejected_protocol",
                "terminal_progress": ["headers_received_before_body_eof"],
                "predicate": "status_200_identity_content_type_not_html_raw_empty",
            },
            "content_length_invalid": {
                "outcome": "rejected_protocol",
                "terminal_progress": ["headers_received_before_body_eof"],
                "predicate": (
                    "status_200_identity_html_content_length_present_noncanonical_raw_empty"
                ),
            },
        },
        "safe_response_header_allowlist": sorted(_SAFE_RESPONSE_HEADERS),
        "secret_header_denylist": sorted(_SECRET_RESPONSE_HEADERS),
        "redirects_followed": 0,
        "requested_final_url": OFFICIAL_REFERENCE_URL,
        "ambient_authority": "credentials_proxy_cookies_netrc_trust_env_all_false",
    },
    "canonical_json": {
        "encoding": "utf8",
        "key_order": "sorted",
        "separators": [",", ":"],
        "trailing_newline": "exactly_one_lf",
        "carriage_return": "forbidden",
        "duplicate_keys": "forbidden",
        "floats_nan_infinity": "forbidden",
        "bom": "forbidden_by_exact_reencode",
        "exact_keys": True,
        "exact_scalar_types": True,
        "max_depth": OFFICIAL_JSON_MAX_DEPTH,
        "max_container_items": OFFICIAL_JSON_MAX_CONTAINER_ITEMS,
        "max_total_nodes": OFFICIAL_JSON_MAX_TOTAL_NODES,
        "max_string_utf8_bytes": OFFICIAL_JSON_MAX_STRING_BYTES,
        "max_integer_digits": OFFICIAL_JSON_MAX_INTEGER_DIGITS,
        "typed_decode_errors": [
            "UnicodeDecodeError",
            "JSONDecodeError",
            "ValueError",
            "TypeError",
            "RecursionError",
            "UnicodeError",
        ],
    },
    "exact_scalar_types": {
        "prepared_request": {
            "port": "exact_int_443",
            "body_byte_count": "exact_int_0",
            "transport_flags": "exact_bool",
        },
        "http_attempt": {
            "attempt_ordinal": "exact_int_0",
            "redirects_followed": "exact_int_0",
            "timestamps_and_counts": "exact_int_not_bool",
            "all_boolean_fields": "exact_bool",
            "terminal_progress_outcome_safe_error": "exact_str",
        },
        "span_claim": {
            "role": "exact_str_allowlisted",
        },
        "semantic_evidence": {
            "attempt_receipt_byte_count": "exact_int_not_bool",
        },
        "bundle_file": {
            "role": "exact_str_allowlisted",
            "byte_count": "exact_int_not_bool",
        },
        "compatibility": {
            "max_network_attempts": "exact_int_2",
            "all_totals_reservations_residuals": "exact_int_not_bool",
        },
    },
    "hash_formulas": {
        "prepared_request_hash": "sha256_exact_canonical_lf_prepared_request_json",
        "attempt_receipt_hash": "sha256_exact_canonical_lf_attempt_receipt_json",
        "evidence_hash": "sha256_exact_canonical_lf_evidence_json_artifact_bytes",
        "bundle_hash": "sha256_exact_canonical_lf_bundle_json_projection",
        "raw_body_sha256": "sha256_exact_entity_bytes",
        "span_sha256": "sha256_exact_half_open_raw_byte_span",
        "reader_contract_hash": "mexc_endpoint_official_evidence_contract_hash",
        "attempt_receipt_byte_count": (
            "len_exact_canonical_lf_attempt_receipt_json_bytes"
        ),
    },
    "provenance_and_clock": {
        "parser_binding": (
            "evidence.parser_version_hash_equals_prepared_request.parser_version_hash"
        ),
        "reader_self_binding": (
            "evidence.reader_version_is_reader_v1_and_hash_equals_this_contract_hash"
        ),
        "observed_body_fetch_binding": {
            "epoch": "evidence.observed_body_fetched_at_us_equals_attempt.body_eof_at_us",
            "monotonic": (
                "evidence.observed_body_fetched_monotonic_us_equals_"
                "attempt.body_eof_monotonic_us"
            ),
        },
        "evidence_order_both_domains": [
            "body_eof",
            "parse_started_strictly_after_eof",
            "parse_completed",
            "reload_completed_strictly_after_parse_completed",
        ],
        "runtime_budget_observation": (
            "structural_fake_only_not_authoritative_due_unbound_clock_source"
        ),
        "opaque_parent_hashes": "terminal_stop_not_source_object_provenance",
        "runtime_tls_trust_identities": "terminal_stop_declarative_not_attested",
    },
    "helper_vectors": {
        "zero_plan_bundle_root": (
            _LAYOUT_ZERO_PLAN_ROOT
        ),
        "zero_plan_files": list(_LAYOUT_ZERO_PLAN_LOCATORS),
        "layout_helper_binding": (
            "exact_derive_official_bundle_root_v1_and_"
            "derive_official_bundle_locators_v1"
        ),
        "empty_sha256": EMPTY_SHA256,
        "migration_statement_sha256": hashlib.sha256(
            _MIGRATION_STATEMENT
        ).hexdigest(),
        "full_candidate_statement_sha256": hashlib.sha256(
            _FULL_CANDIDATE_STATEMENT
        ).hexdigest(),
        "canonical_json_sample": "{\"a\":1,\"b\":false}\\n",
    },
    "authority": {
        "current_output": REVIEWED_FAKE_FIXTURE_ONLY,
        "terminal_compatible": False,
        "network_code_or_default_executor": False,
        "u5_granted": False,
        "detached_official_anchor": False,
        "strict_adapter": "always_typed_stop",
        "strict_adapter_classifications": [
            "reviewed_fake_structural_nonterminal",
        ],
        "strict_adapter_raw_reparse": False,
        "terminal_blockers": list(_TERMINAL_BLOCKERS),
        "opaque_structural_bindings": [
            "manifest_hash",
            "authorization_receipt_hash",
            "preflight_receipt_hash",
            "network_intent_hash",
            "runtime_authority_binding_hash",
            "tls_policy_version_hash",
            "trust_store_version_hash",
        ],
        "clock_samples": "reviewed_fake_structural_not_runtime_attestation",
        "incomplete_failure_bundle": "unsupported_typed_terminal_stop",
    },
}


def _computed_contract_hash() -> str:
    if (
        derive_official_bundle_root_v1("0" * 64) != _LAYOUT_ZERO_PLAN_ROOT
        or official_bundle_relative_paths_v1("0" * 64)
        != _LAYOUT_ZERO_PLAN_LOCATORS
    ):
        raise MexcOfficialEvidenceContractError(
            "official_evidence_pilot_output_layout_helper_vector_drift"
        )
    return _sha256_payload(_CONTRACT_SCHEMA)


def official_evidence_contract_descriptor_v1() -> dict[str, object]:
    """Return a detached exact-JSON copy of the declarative contract descriptor."""

    _computed_contract_hash()
    return json.loads(_canonical_json_bytes(_CONTRACT_SCHEMA).decode("utf-8"))


def mexc_endpoint_official_evidence_contract_hash() -> str:
    digest = _computed_contract_hash()
    if _PINNED_CONTRACT_HASH and digest != _PINNED_CONTRACT_HASH:
        raise MexcOfficialEvidenceContractError(
            "official_evidence_contract_changed_without_version_bump"
        )
    return digest


__all__ = [
    "CANDIDATE_CONTRACT_HASH",
    "CANDIDATE_CONTRACT_VERSION",
    "MexcOfficialEvidenceBudgetStop",
    "MexcOfficialEvidenceContractError",
    "MexcOfficialEvidenceError",
    "MexcOfficialEvidenceSemanticStop",
    "MexcOfficialEvidenceStorageStop",
    "MexcOfficialEvidenceTerminalStop",
    "OFFICIAL_DOCUMENT_EVIDENCE_VERSION",
    "OFFICIAL_DOCUMENT_READER_VERSION",
    "OFFICIAL_EVIDENCE_BUNDLE_VERSION",
    "OFFICIAL_EVIDENCE_COMPATIBILITY_VERSION",
    "OFFICIAL_EVIDENCE_STORE_VERSION",
    "OFFICIAL_REFERENCE_HTTP_ATTEMPT_VERSION",
    "OFFICIAL_REFERENCE_ID",
    "OFFICIAL_REFERENCE_REQUEST_VERSION",
    "OFFICIAL_REFERENCE_URL",
    "OFFICIAL_RAW_BODY_HARD_CAP_BYTES",
    "OFFICIAL_STORAGE_CONCURRENCY_BOUNDARY",
    "PILOT_OUTPUT_LAYOUT_CONTRACT_VERSION",
    "PILOT_OUTPUT_LAYOUT_EXPECTED_CONTRACT_HASH",
    "OfficialDocumentEvidenceV1",
    "OfficialDocumentSpanClaimV1",
    "OfficialEvidenceBundleFileV1",
    "OfficialEvidenceBundleV1",
    "OfficialEvidenceCompatibilityV1",
    "OfficialReferenceHttpAttemptV1",
    "OfficialReferencePreparedRequestV1",
    "REVIEWED_FAKE_FIXTURE_ONLY",
    "assess_official_evidence_compatibility_v1",
    "build_exact_span_claims_v1",
    "derive_official_bundle_root_v1",
    "mexc_endpoint_official_evidence_contract_hash",
    "official_bundle_relative_paths_v1",
    "official_evidence_contract_descriptor_v1",
    "parse_canonical_json_lf_v1",
    "publish_official_evidence_bundle_v1",
    "read_official_document_evidence_v1",
    "reload_official_evidence_bundle_v1",
    "require_terminal_compatible_official_evidence_v1",
]
