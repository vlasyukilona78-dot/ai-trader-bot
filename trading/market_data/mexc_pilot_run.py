"""Immutable, network-free orchestration contract for the MEXC P2 QA pilot.

This module deliberately contains no HTTP executor and cannot grant U5.  It
describes the exact run that a later, detached authorization may permit and a
pure state projection that makes endpoint verification precede acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import uuid
from typing import Any, Mapping

from trading.market_data.mexc_futures_transport import (
    HISTORY_RESOURCE_LIMITS_VERSION,
    HISTORY_RETRY_POLICY_VERSION,
    MEXC_FUTURES_ENDPOINT_CONTRACT_VERSION,
    MEXC_FUTURES_TRANSPORT_CONTRACT_VERSION,
    HistoryResourceLimitsV1,
    HistoryRetryPolicyV1,
    candidate_endpoint_fixture_path,
    candidate_history_resource_limits_v1,
    candidate_history_retry_policy_v1,
    load_mexc_futures_endpoint_contract_v1,
    mexc_futures_transport_contract_hash,
)
from trading.market_data.min1_aggregation import (
    MIN1_AGGREGATION_CONTRACT_VERSION,
    min1_aggregation_contract_hash,
)
from trading.market_data.strict_history import _frozen_contract_hash
from trading.market_data.strict_history_v2 import (
    STRICT_HISTORY_V2_CONTRACT_VERSION,
    HistoryRangeRequestV2,
    StrictMexcHistoryCollectorV2,
    storage_profile_hash,
    strict_history_v2_contract_hash,
)


PILOT_RUN_CONTRACT_VERSION = "mexc_public_qa_pilot_run_v1"
PILOT_GLOBAL_BUDGET_VERSION = "mexc_public_qa_pilot_budget_v1"
PILOT_SHARD_PLAN_VERSION = "mexc_public_qa_pilot_shard_v1"
ENDPOINT_VERIFICATION_PLAN_VERSION = "mexc_endpoint_verification_plan_v1"
U5_AUTHORIZATION_RECEIPT_VERSION = "mexc_u5_public_pilot_authorization_v1"
PILOT_PREFLIGHT_RECEIPT_VERSION = "mexc_public_qa_pilot_preflight_v1"
PILOT_INTENT_DURABILITY_RECEIPT_VERSION = "mexc_public_qa_pilot_intent_durability_v1"
PILOT_NETWORK_INTENT_VERSION = "mexc_public_qa_pilot_network_intent_v1"
ENDPOINT_VERIFICATION_RECEIPT_VERSION = "mexc_endpoint_verification_receipt_v1"
PILOT_SHARD_RESULT_VERSION = "mexc_public_qa_pilot_shard_result_v1"
PILOT_STEP_FAILURE_RECEIPT_VERSION = "mexc_public_qa_pilot_step_failure_v1"
PILOT_RUN_ANCHOR_VERSION = "mexc_public_qa_pilot_anchor_v1"
PILOT_RUN_STATE_VERSION = "mexc_public_qa_pilot_state_v1"

_PINNED_CONTRACT_HASH = (
    "f3d642d436e9d4a44e65f35c6ea8375bd92b4b36b30f1c86af54936a608ce65e"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_FILE_LOCATOR_RE = re.compile(r"^file:///[A-Z]:/[A-Za-z0-9._~!$&'()+,;=@/ -]{1,480}$")

_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_RUN_CONTROL_ENTRY_BYTES = 512 * 1024
_MAX_PILOT_SHARDS = 32
_MAX_PILOT_SYMBOLS = 11
_MAX_TOTAL_PAGES = 6_400
_MAX_TOTAL_ROWS = 12_800_000
_MAX_TOTAL_ATTEMPTS = 64_000
_MAX_TOTAL_RAW_BYTES = 16 * 1024**3
_MAX_TOTAL_STORAGE_BYTES = 32 * 1024**3
_MAX_RUN_CONTROL_BYTES = 128 * 1024**2
_MAX_TOTAL_OUTPUT_BYTES = 40 * 1024**3
_MAX_TOTAL_RUNTIME_US = 48 * 60 * 60 * 1_000_000
_MAX_TOTAL_SLEEP_US = 8 * 60 * 60 * 1_000_000
_MAX_SPACING_US = 60 * 1_000_000
_MAX_INVENTORY_ENTRIES = 100_000
_MAX_PREFLIGHT_AGE_US = 15 * 60 * 1_000_000

_SHARD_ROLES = frozenset(
    {"qa_min1", "deep_min1", "native_min60_control"}
)
_FORBIDDEN_SCOPES = (
    "credentials",
    "dot_env",
    "private_api",
    "telegram",
    "scanner",
    "model_fit",
    "orders",
    "capital_action",
)
_VERIFICATION_ACTIONS = (
    "fetch_current_official_reference",
    "run_exact_live_kline_probe",
    "persist_reload_anchor_combined_receipt",
)
_AUTHORIZED_OPERATIONS = (
    "current_official_reference_verification",
    "exact_live_kline_probe",
    "conditional_manifest_history_acquisition",
)
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CLOCK$",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


class PilotRunError(RuntimeError):
    """Base error for the offline P2 pilot contract."""


class PilotRunContractError(PilotRunError):
    pass


class PilotRunBudgetExceededError(PilotRunError):
    def __init__(self, resource: str, limit: int, observed: int):
        self.resource = resource
        self.limit = limit
        self.observed = observed
        super().__init__(f"pilot_budget_exceeded.{resource}.{limit}.{observed}")


class PilotRunPreflightError(PilotRunError):
    pass


class PilotRunAuthorizationError(PilotRunError):
    pass


class PilotRunTransitionError(PilotRunError):
    pass


class PilotRunArtifactError(PilotRunError):
    pass


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
        raise PilotRunContractError("pilot_payload_is_not_canonical_json") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_payload(payload: object) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


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
        raise PilotRunContractError(f"{field}_is_invalid")
    return value


def _safe_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise PilotRunContractError(f"{field}_is_invalid")
    return value


def _safe_code(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_CODE_RE.fullmatch(value):
        raise PilotRunContractError(f"{field}_is_invalid")
    return value


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PilotRunContractError(f"{field}_is_invalid")
    return value


def _commit(value: object) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise PilotRunContractError("repository_commit_is_invalid")
    return value


def _exact_mapping(
    payload: object, expected: frozenset[str], *, code: str
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise PilotRunContractError(code)
    return dict(payload)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PilotRunContractError("pilot_json_contains_duplicate_key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise PilotRunContractError("pilot_json_contains_nonfinite_constant")


def _parse_canonical_json(raw: bytes, *, max_bytes: int = _MAX_MANIFEST_BYTES) -> dict[str, Any]:
    if not isinstance(raw, bytes) or len(raw) > max_bytes:
        raise PilotRunArtifactError("pilot_artifact_is_oversized")
    if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
        raise PilotRunArtifactError("pilot_artifact_line_ending_is_invalid")
    try:
        payload = json.loads(
            raw[:-1].decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except PilotRunError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotRunArtifactError("pilot_artifact_json_is_invalid") from exc
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload) + b"\n":
        raise PilotRunArtifactError("pilot_artifact_is_not_canonical")
    return payload


def _read_artifact_limited(path: Path, *, max_bytes: int, code: str) -> bytes:
    """Read one regular immutable artifact through one bounded file handle."""

    target = Path(path)
    _validate_existing_directory_chain(target.parent, code=code)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        before_path = target.lstat()
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(before_path.st_mode) or getattr(
            before_path, "st_file_attributes", 0
        ) & reparse:
            raise PilotRunArtifactError(f"{code}_is_reparse_or_symlink")
        descriptor = os.open(target, flags)
        before = os.fstat(descriptor)
        if (
            before_path.st_dev,
            before_path.st_ino,
            before_path.st_size,
            getattr(before_path, "st_mtime_ns", None),
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            getattr(before, "st_mtime_ns", None),
        ):
            raise PilotRunArtifactError(f"{code}_changed_before_open")
        if not stat.S_ISREG(before.st_mode):
            raise PilotRunArtifactError(f"{code}_is_not_regular")
        if before.st_size > max_bytes:
            raise PilotRunArtifactError(f"{code}_is_oversized")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise PilotRunArtifactError(f"{code}_is_oversized")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            getattr(before, "st_mtime_ns", None),
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            getattr(after, "st_mtime_ns", None),
        ) or len(raw) != after.st_size:
            raise PilotRunArtifactError(f"{code}_changed_during_read")
        return raw
    except PilotRunError:
        raise
    except OSError as exc:
        raise PilotRunArtifactError(f"{code}_is_unreadable") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _validate_existing_directory_chain(path: Path, *, code: str) -> None:
    chain = tuple(reversed((path, *path.parents)))
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    try:
        for current in chain:
            item = current.lstat()
            if (
                stat.S_ISLNK(item.st_mode)
                or getattr(item, "st_file_attributes", 0) & reparse
                or not stat.S_ISDIR(item.st_mode)
            ):
                raise PilotRunArtifactError(
                    f"{code}_parent_chain_is_not_a_plain_directory"
                )
    except PilotRunError:
        raise
    except OSError as exc:
        raise PilotRunArtifactError(f"{code}_parent_chain_is_unreadable") from exc


def _relative_root(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or len(value) > 240:
        raise PilotRunContractError(f"{field}_is_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PilotRunContractError(f"{field}_is_invalid")
    lowered = tuple(part.lower() for part in path.parts)
    if any(
        part.endswith((".", " "))
        or part.rstrip(" .").split(".", 1)[0].upper()
        in _WINDOWS_RESERVED_BASENAMES
        for part in path.parts
    ):
        raise PilotRunContractError(f"{field}_uses_reserved_windows_name")
    if any(lowered[index : index + 2] == ("data", "history") for index in range(len(lowered) - 1)):
        raise PilotRunContractError(f"{field}_uses_legacy_history")
    rendered = path.as_posix()
    if rendered != value:
        raise PilotRunContractError(f"{field}_is_not_canonical")
    return rendered


def _output_root_locator(value: object) -> str:
    if not isinstance(value, str) or not _FILE_LOCATOR_RE.fullmatch(value):
        raise PilotRunContractError("output_root_locator_is_invalid")
    if "\\" in value or value.endswith("/"):
        raise PilotRunContractError("output_root_locator_is_invalid")
    components = value[len("file:///") :].split("/")
    if (
        not re.fullmatch(r"[A-Z]:", components[0])
        or any(
            component in {"", ".", ".."}
            or component.endswith((".", " "))
            or component.rstrip(" .").split(".", 1)[0].upper()
            in _WINDOWS_RESERVED_BASENAMES
            for component in components[1:]
        )
    ):
        raise PilotRunContractError("output_root_locator_is_not_canonical")
    lowered = value.lower()
    if "/data/history/" in f"{lowered}/":
        raise PilotRunContractError("output_root_locator_uses_legacy_history")
    return value


def _prepared_request_payload(request: HistoryRangeRequestV2) -> dict[str, object]:
    page = StrictMexcHistoryCollectorV2.plan_pages(request)[0]
    prepared = request.endpoint_contract.prepare(page)
    return {
        "endpoint_identity": prepared.endpoint_identity,
        "endpoint_contract_hash": prepared.endpoint_contract_hash,
        "method": prepared.method,
        "scheme": prepared.scheme,
        "host": prepared.host,
        "port": prepared.port,
        "path": prepared.path,
        "query": [list(item) for item in prepared.query],
        "headers": [list(item) for item in prepared.headers],
        "tls_verify": prepared.tls_verify,
        "allow_redirects": prepared.allow_redirects,
        "trust_env": prepared.trust_env,
        "body": prepared.body,
    }


def _reserved_retry_sleep_us(request: HistoryRangeRequestV2) -> int:
    if request.resource_limits.max_attempts_per_page <= 1:
        return 0
    return request.retry_policy.max_total_sleep_us


def _planned_run_control_entry_count(shard_count: int) -> int:
    _strict_int(
        shard_count,
        field="pilot_planned_run_control_shard_count",
        minimum=1,
        maximum=_MAX_PILOT_SHARDS,
    )
    return 4 * shard_count + 7


def _planned_run_control_bytes(shard_count: int) -> int:
    """Reserve the manifest plus every bounded success-path control artifact."""

    return _MAX_MANIFEST_BYTES + (
        _planned_run_control_entry_count(shard_count) - 1
    ) * _MAX_RUN_CONTROL_ENTRY_BYTES


def _control_inventory_entry(
    *,
    kind: str,
    locator: str,
    semantic_hash: str,
    payload: object,
) -> tuple[str, str, str, str, int]:
    _safe_identifier(kind.replace(":", "."), field="pilot_control_entry_kind")
    normalized_locator = _relative_root(locator, field="pilot_control_entry_locator")
    _digest(semantic_hash, field="pilot_control_entry_semantic_hash")
    artifact = _canonical_bytes(payload) + b"\n"
    maximum = (
        _MAX_MANIFEST_BYTES
        if kind == "manifest"
        else _MAX_RUN_CONTROL_ENTRY_BYTES
    )
    if len(artifact) > maximum:
        raise PilotRunBudgetExceededError(
            "run_control_entry_bytes",
            maximum,
            len(artifact),
        )
    return (
        kind,
        normalized_locator,
        semantic_hash,
        _sha256_bytes(artifact),
        len(artifact),
    )


def _parse_history_request(payload: object) -> HistoryRangeRequestV2:
    expected = frozenset(
        {
            "contract_version",
            "venue",
            "symbol",
            "venue_symbol",
            "interval",
            "interval_seconds",
            "start_open_ts",
            "end_open_ts_exclusive",
            "collection_as_of_us",
            "page_size",
            "endpoint_contract",
            "resource_limits",
            "retry_policy",
            "contract_identities",
        }
    )
    values = _exact_mapping(payload, expected, code="pilot_history_request_schema_mismatch")
    endpoint = load_mexc_futures_endpoint_contract_v1(candidate_endpoint_fixture_path())
    if _canonical_bytes(values["endpoint_contract"]) != _canonical_bytes(endpoint.as_dict()):
        raise PilotRunContractError("pilot_embedded_endpoint_is_not_the_pinned_candidate")
    limits = HistoryResourceLimitsV1.from_dict(values["resource_limits"])
    retry = HistoryRetryPolicyV1.from_dict(values["retry_policy"])
    identities = values["contract_identities"]
    if not isinstance(identities, dict) or not isinstance(identities.get("storage_profile"), str):
        raise PilotRunContractError("pilot_history_request_identities_are_invalid")
    result = HistoryRangeRequestV2(
        venue=values["venue"],
        symbol=values["symbol"],
        venue_symbol=values["venue_symbol"],
        interval=values["interval"],
        start_open_ts=values["start_open_ts"],
        end_open_ts_exclusive=values["end_open_ts_exclusive"],
        collection_as_of_us=values["collection_as_of_us"],
        endpoint_contract=endpoint,
        resource_limits=limits,
        retry_policy=retry,
        page_size=values["page_size"],
        storage_profile=identities["storage_profile"],
        contract_version=values["contract_version"],
    )
    if _canonical_bytes(result.as_dict()) != _canonical_bytes(values):
        raise PilotRunContractError("pilot_history_request_round_trip_mismatch")
    return result


@dataclass(frozen=True)
class PilotGlobalBudgetsV1:
    max_symbols: int
    max_shards: int
    max_total_pages: int
    max_total_rows: int
    max_verification_attempts: int
    max_acquisition_attempts: int
    max_network_attempts: int
    max_total_raw_body_bytes: int
    max_total_logical_storage_bytes: int
    max_run_control_bytes: int
    max_total_output_bytes: int
    max_sum_shard_runtime_us: int
    max_run_elapsed_us: int
    max_observed_sleep_us: int
    min_inter_step_spacing_us: int
    max_active_shards: int
    max_in_flight_http_attempts: int
    min_free_disk_bytes_before_run: int
    required_free_disk_bytes_after_reservation: int
    max_inventory_entries: int
    max_preflight_age_us: int
    contract_version: str = PILOT_GLOBAL_BUDGET_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_GLOBAL_BUDGET_VERSION:
            raise PilotRunContractError("pilot_budget_version_mismatch")
        limits = {
            "max_symbols": _MAX_PILOT_SYMBOLS,
            "max_shards": _MAX_PILOT_SHARDS,
            "max_total_pages": _MAX_TOTAL_PAGES,
            "max_total_rows": _MAX_TOTAL_ROWS,
            "max_verification_attempts": 4,
            "max_acquisition_attempts": _MAX_TOTAL_ATTEMPTS,
            "max_network_attempts": _MAX_TOTAL_ATTEMPTS + 4,
            "max_total_raw_body_bytes": _MAX_TOTAL_RAW_BYTES,
            "max_total_logical_storage_bytes": _MAX_TOTAL_STORAGE_BYTES,
            "max_run_control_bytes": _MAX_RUN_CONTROL_BYTES,
            "max_total_output_bytes": _MAX_TOTAL_OUTPUT_BYTES,
            "max_sum_shard_runtime_us": _MAX_TOTAL_RUNTIME_US,
            "max_run_elapsed_us": _MAX_TOTAL_RUNTIME_US,
            "max_observed_sleep_us": _MAX_TOTAL_SLEEP_US,
            "min_inter_step_spacing_us": _MAX_SPACING_US,
            "min_free_disk_bytes_before_run": _MAX_TOTAL_OUTPUT_BYTES * 2,
            "required_free_disk_bytes_after_reservation": _MAX_TOTAL_OUTPUT_BYTES,
            "max_inventory_entries": _MAX_INVENTORY_ENTRIES,
            "max_preflight_age_us": _MAX_PREFLIGHT_AGE_US,
        }
        for field, maximum in limits.items():
            minimum = 0 if field in {"min_inter_step_spacing_us", "max_observed_sleep_us"} else 1
            _strict_int(getattr(self, field), field=field, minimum=minimum, maximum=maximum)
        if self.max_active_shards != 1 or self.max_in_flight_http_attempts != 1:
            raise PilotRunContractError("pilot_v1_execution_must_be_serial")
        if self.max_network_attempts < self.max_verification_attempts + self.max_acquisition_attempts:
            raise PilotRunContractError("pilot_network_attempt_budget_is_inconsistent")
        if self.max_total_output_bytes < self.max_total_logical_storage_bytes + self.max_run_control_bytes:
            raise PilotRunContractError("pilot_output_budget_is_inconsistent")
        required = self.max_total_output_bytes + self.required_free_disk_bytes_after_reservation
        if self.min_free_disk_bytes_before_run < required:
            raise PilotRunContractError("pilot_disk_budget_is_inconsistent")
        if self.max_run_elapsed_us < self.max_sum_shard_runtime_us:
            raise PilotRunContractError("pilot_wall_runtime_below_shard_runtime")

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "PilotGlobalBudgetsV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_budget_schema_mismatch",
        )
        return cls(**values)

    @property
    def contract_hash(self) -> str:
        return _frozen_contract_hash(self)


@dataclass(frozen=True)
class PilotShardPlanV1:
    ordinal: int
    role: str
    request: HistoryRangeRequestV2
    relative_artifact_root: str
    source_min1_request_id: str | None = None
    contract_version: str = PILOT_SHARD_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_SHARD_PLAN_VERSION:
            raise PilotRunContractError("pilot_shard_version_mismatch")
        _strict_int(self.ordinal, field="pilot_shard_ordinal", minimum=0)
        if self.role not in _SHARD_ROLES:
            raise PilotRunContractError("pilot_shard_role_is_invalid")
        if not isinstance(self.request, HistoryRangeRequestV2):
            raise PilotRunContractError("pilot_shard_request_is_invalid")
        object.__setattr__(
            self,
            "relative_artifact_root",
            _relative_root(self.relative_artifact_root, field="pilot_shard_root"),
        )
        if self.role == "native_min60_control":
            _digest(self.source_min1_request_id, field="source_min1_request_id")
            if self.request.interval != "Min60":
                raise PilotRunContractError("native_control_must_use_min60")
        else:
            if self.source_min1_request_id is not None:
                raise PilotRunContractError("min1_shard_cannot_bind_source_request")
            if self.request.interval != "Min1":
                raise PilotRunContractError("pilot_min1_shard_interval_mismatch")
        rows = self.request.expected_row_count
        if self.role == "qa_min1" and (rows % 1_440 or not 7 <= rows // 1_440 <= 14):
            raise PilotRunContractError("qa_min1_range_must_be_7_to_14_days")
        if self.role == "deep_min1" and rows != 140 * 1_440:
            raise PilotRunContractError("deep_min1_range_must_be_exactly_140_days")

    @property
    def plan_id(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "ordinal": self.ordinal,
            "role": self.role,
            "request": self.request.as_dict(),
            "request_id": self.request.request_id,
            "expected_rows": self.request.expected_row_count,
            "required_pages": self.request.required_pages,
            "relative_artifact_root": self.relative_artifact_root,
            "source_min1_request_id": self.source_min1_request_id,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PilotShardPlanV1":
        expected = frozenset(
            {
                "contract_version",
                "ordinal",
                "role",
                "request",
                "request_id",
                "expected_rows",
                "required_pages",
                "relative_artifact_root",
                "source_min1_request_id",
            }
        )
        values = _exact_mapping(payload, expected, code="pilot_shard_schema_mismatch")
        request = _parse_history_request(values.pop("request"))
        repeated = {
            "request_id": request.request_id,
            "expected_rows": request.expected_row_count,
            "required_pages": request.required_pages,
        }
        for field, expected_value in repeated.items():
            if values.pop(field) != expected_value:
                raise PilotRunContractError(f"pilot_shard_{field}_mismatch")
        return cls(request=request, **values)


@dataclass(frozen=True)
class EndpointVerificationPlanV1:
    probe_request: HistoryRangeRequestV2
    relative_artifact_root: str
    official_reference_url: str
    verifier_contract_version: str
    verifier_contract_hash: str
    max_network_attempts: int
    max_total_raw_body_bytes: int
    max_total_storage_bytes: int
    max_runtime_us: int
    max_total_sleep_us: int
    contract_version: str = ENDPOINT_VERIFICATION_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != ENDPOINT_VERIFICATION_PLAN_VERSION:
            raise PilotRunContractError("endpoint_verification_plan_version_mismatch")
        if not isinstance(self.probe_request, HistoryRangeRequestV2):
            raise PilotRunContractError("endpoint_probe_request_is_invalid")
        if (
            self.probe_request.interval != "Min1"
            or self.probe_request.expected_row_count != 1
            or self.probe_request.required_pages != 1
            or self.probe_request.resource_limits.max_attempts_per_page != 1
            or self.probe_request.resource_limits.max_total_attempts != 1
        ):
            raise PilotRunContractError("endpoint_probe_must_be_one_bar_one_attempt")
        object.__setattr__(
            self,
            "relative_artifact_root",
            _relative_root(self.relative_artifact_root, field="endpoint_probe_root"),
        )
        endpoint = self.probe_request.endpoint_contract
        pinned_endpoint = load_mexc_futures_endpoint_contract_v1(
            candidate_endpoint_fixture_path()
        )
        if _canonical_bytes(endpoint.as_dict()) != _canonical_bytes(
            pinned_endpoint.as_dict()
        ):
            raise PilotRunContractError("endpoint_probe_is_not_the_pinned_candidate")
        if self.official_reference_url != endpoint.plan_reference_url:
            raise PilotRunContractError("endpoint_reference_url_mismatch")
        _safe_identifier(self.verifier_contract_version, field="verifier_contract_version")
        _digest(self.verifier_contract_hash, field="verifier_contract_hash")
        _strict_int(self.max_network_attempts, field="verification_attempts", minimum=2, maximum=4)
        for field, maximum in (
            ("max_total_raw_body_bytes", 16 * 1024**2),
            ("max_total_storage_bytes", 32 * 1024**2),
            ("max_runtime_us", 5 * 60 * 1_000_000),
            ("max_total_sleep_us", 2 * 60 * 1_000_000),
        ):
            _strict_int(getattr(self, field), field=field, minimum=1, maximum=maximum)
        if self.max_total_storage_bytes < self.max_total_raw_body_bytes:
            raise PilotRunContractError("verification_storage_below_raw_budget")

    @property
    def prepared_live_request_hash(self) -> str:
        return _sha256_payload(_prepared_request_payload(self.probe_request))

    @property
    def plan_hash(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        endpoint = self.probe_request.endpoint_contract
        return {
            "contract_version": self.contract_version,
            "ordered_actions": list(_VERIFICATION_ACTIONS),
            "mismatch_action": "stop_before_any_acquisition",
            "candidate_endpoint_version": endpoint.contract_version,
            "candidate_endpoint_hash": endpoint.contract_hash,
            "candidate_endpoint_identity": endpoint.endpoint_identity,
            "candidate_verification_status": endpoint.verification_status,
            "official_reference_url": self.official_reference_url,
            "probe_request": self.probe_request.as_dict(),
            "probe_request_id": self.probe_request.request_id,
            "prepared_live_request_hash": self.prepared_live_request_hash,
            "relative_artifact_root": self.relative_artifact_root,
            "verifier_contract_version": self.verifier_contract_version,
            "verifier_contract_hash": self.verifier_contract_hash,
            "max_network_attempts": self.max_network_attempts,
            "max_total_raw_body_bytes": self.max_total_raw_body_bytes,
            "max_total_storage_bytes": self.max_total_storage_bytes,
            "max_runtime_us": self.max_runtime_us,
            "max_total_sleep_us": self.max_total_sleep_us,
            "success_predicate": {
                "current_official_document_supports_candidate": True,
                "tls_verified": True,
                "redirects_followed": False,
                "credentials_used": False,
                "trust_env": False,
                "http_status": 200,
                "body_complete": True,
                "api_success": True,
                "api_code": 0,
                "exact_grid_and_ohlcv_amount_schema": True,
                "fresh_disk_reload": True,
                "detached_anchor_before_acquisition": True,
            },
        }

    @classmethod
    def from_dict(cls, payload: object) -> "EndpointVerificationPlanV1":
        expected = frozenset(
            {
                "contract_version",
                "ordered_actions",
                "mismatch_action",
                "candidate_endpoint_version",
                "candidate_endpoint_hash",
                "candidate_endpoint_identity",
                "candidate_verification_status",
                "official_reference_url",
                "probe_request",
                "probe_request_id",
                "prepared_live_request_hash",
                "relative_artifact_root",
                "verifier_contract_version",
                "verifier_contract_hash",
                "max_network_attempts",
                "max_total_raw_body_bytes",
                "max_total_storage_bytes",
                "max_runtime_us",
                "max_total_sleep_us",
                "success_predicate",
            }
        )
        values = _exact_mapping(payload, expected, code="endpoint_verification_plan_schema_mismatch")
        if values.pop("ordered_actions") != list(_VERIFICATION_ACTIONS):
            raise PilotRunContractError("endpoint_verification_action_order_mismatch")
        if values.pop("mismatch_action") != "stop_before_any_acquisition":
            raise PilotRunContractError("endpoint_verification_mismatch_action_changed")
        request = _parse_history_request(values.pop("probe_request"))
        endpoint = request.endpoint_contract
        repeated = {
            "candidate_endpoint_version": endpoint.contract_version,
            "candidate_endpoint_hash": endpoint.contract_hash,
            "candidate_endpoint_identity": endpoint.endpoint_identity,
            "candidate_verification_status": endpoint.verification_status,
            "probe_request_id": request.request_id,
            "prepared_live_request_hash": _sha256_payload(_prepared_request_payload(request)),
        }
        for field, expected_value in repeated.items():
            if values.pop(field) != expected_value:
                raise PilotRunContractError(f"endpoint_verification_{field}_mismatch")
        expected_success = cls(
            probe_request=request,
            official_reference_url=values["official_reference_url"],
            relative_artifact_root=values["relative_artifact_root"],
            verifier_contract_version=values["verifier_contract_version"],
            verifier_contract_hash=values["verifier_contract_hash"],
            max_network_attempts=values["max_network_attempts"],
            max_total_raw_body_bytes=values["max_total_raw_body_bytes"],
            max_total_storage_bytes=values["max_total_storage_bytes"],
            max_runtime_us=values["max_runtime_us"],
            max_total_sleep_us=values["max_total_sleep_us"],
            contract_version=values["contract_version"],
        )
        if values.pop("success_predicate") != expected_success.as_dict()["success_predicate"]:
            raise PilotRunContractError("endpoint_verification_success_predicate_mismatch")
        return expected_success


@dataclass(frozen=True)
class MexcPublicQaPilotRunManifestV1:
    repository_commit: str
    repository_tree_receipt_hash: str
    created_at_us: int
    parent_master_plan_path: str
    parent_master_plan_sha256: str
    parent_adr_path: str
    parent_adr_sha256: str
    output_root_locator: str
    shard_executor_contract_version: str
    shard_executor_contract_hash: str
    endpoint_verification: EndpointVerificationPlanV1
    shards: tuple[PilotShardPlanV1, ...]
    budgets: PilotGlobalBudgetsV1
    contract_version: str = PILOT_RUN_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_RUN_CONTRACT_VERSION:
            raise PilotRunContractError("pilot_manifest_version_mismatch")
        _commit(self.repository_commit)
        _digest(
            self.repository_tree_receipt_hash,
            field="repository_tree_receipt_hash",
        )
        _strict_int(self.created_at_us, field="manifest_created_at_us", minimum=1)
        if self.parent_master_plan_path != "docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md":
            raise PilotRunContractError("pilot_parent_master_plan_path_mismatch")
        if self.parent_adr_path != "docs/ADR_MEXC_V3_FINAL_BOT_2026-08-15.md":
            raise PilotRunContractError("pilot_parent_adr_path_mismatch")
        _digest(self.parent_master_plan_sha256, field="parent_master_plan_sha256")
        _digest(self.parent_adr_sha256, field="parent_adr_sha256")
        object.__setattr__(
            self, "output_root_locator", _output_root_locator(self.output_root_locator)
        )
        _safe_identifier(
            self.shard_executor_contract_version,
            field="shard_executor_contract_version",
        )
        _digest(
            self.shard_executor_contract_hash,
            field="shard_executor_contract_hash",
        )
        if not isinstance(self.endpoint_verification, EndpointVerificationPlanV1):
            raise PilotRunContractError("pilot_endpoint_verification_plan_is_invalid")
        if not isinstance(self.budgets, PilotGlobalBudgetsV1):
            raise PilotRunContractError("pilot_global_budget_is_invalid")
        if not isinstance(self.shards, tuple) or not self.shards or not all(
            isinstance(item, PilotShardPlanV1) for item in self.shards
        ):
            raise PilotRunContractError("pilot_shards_are_not_an_immutable_tuple")
        if tuple(item.ordinal for item in self.shards) != tuple(range(len(self.shards))):
            raise PilotRunContractError("pilot_shard_ordinals_are_not_contiguous")
        if len(self.shards) > self.budgets.max_shards:
            raise PilotRunBudgetExceededError(
                "shards", self.budgets.max_shards, len(self.shards)
            )

        probe = self.endpoint_verification.probe_request
        all_requests = (probe, *(item.request for item in self.shards))
        candidate_limits = candidate_history_resource_limits_v1()
        candidate_retry = candidate_history_retry_policy_v1()
        endpoint_hash = probe.endpoint_contract.contract_hash
        endpoint_identity = probe.endpoint_contract.endpoint_identity
        storage_profile = probe.storage_profile
        if any(
            request.endpoint_contract.contract_hash != endpoint_hash
            or request.endpoint_contract.endpoint_identity != endpoint_identity
            or request.storage_profile != storage_profile
            or request.collection_as_of_us != probe.collection_as_of_us
            for request in all_requests
        ):
            raise PilotRunContractError("pilot_requests_do_not_share_run_identity")
        if any(request.venue != "mexc_contract" for request in all_requests):
            raise PilotRunContractError("pilot_request_venue_mismatch")
        limit_fields = tuple(
            field
            for field in HistoryResourceLimitsV1.__dataclass_fields__
            if field != "contract_version"
        )
        if any(
            any(
                getattr(request.resource_limits, field)
                > getattr(candidate_limits, field)
                for field in limit_fields
            )
            for request in all_requests
        ):
            raise PilotRunContractError("pilot_request_exceeds_candidate_resource_ceiling")
        if any(
            _canonical_bytes(request.retry_policy.as_dict())
            != _canonical_bytes(candidate_retry.as_dict())
            for request in all_requests
        ):
            raise PilotRunContractError("pilot_request_retry_policy_mismatch")
        request_ids = tuple(request.request_id for request in all_requests)
        if len(set(request_ids)) != len(request_ids):
            raise PilotRunContractError("pilot_request_ids_are_not_unique")

        roots = (
            self.endpoint_verification.relative_artifact_root,
            *(item.relative_artifact_root for item in self.shards),
        )
        if len(set(roots)) != len(roots):
            raise PilotRunContractError("pilot_artifact_roots_are_not_unique")
        root_parts = tuple(PurePosixPath(root).parts for root in roots)
        for left_index, left in enumerate(root_parts):
            for right_index, right in enumerate(root_parts):
                if left_index != right_index and len(left) < len(right) and right[: len(left)] == left:
                    raise PilotRunContractError("pilot_artifact_roots_are_nested")
        expected_probe_root = f"verification/{probe.request_id}"
        if self.endpoint_verification.relative_artifact_root != expected_probe_root:
            raise PilotRunContractError("pilot_probe_root_is_not_request_derived")
        for item in self.shards:
            expected_root = f"shards/{item.ordinal:04d}.{item.role}.{item.request.request_id}"
            if item.relative_artifact_root != expected_root:
                raise PilotRunContractError("pilot_shard_root_is_not_request_derived")

        qa = tuple(item for item in self.shards if item.role == "qa_min1")
        deep = tuple(item for item in self.shards if item.role == "deep_min1")
        controls = tuple(
            item for item in self.shards if item.role == "native_min60_control"
        )
        qa_symbols = tuple(item.request.symbol for item in qa)
        if len(set(qa_symbols)) != len(qa_symbols) or not 9 <= len(qa_symbols) <= 11:
            raise PilotRunContractError("pilot_requires_btc_plus_8_to_10_unique_symbols")
        if "BTCUSDT" not in qa_symbols:
            raise PilotRunContractError("pilot_requires_btcusdt")
        qa_venue_by_symbol = {
            item.request.symbol: item.request.venue_symbol for item in qa
        }
        if any(
            qa_venue_by_symbol.get(item.request.symbol)
            != item.request.venue_symbol
            for item in deep
        ):
            raise PilotRunContractError("deep_probe_symbol_is_not_in_qa_population")
        if not deep:
            raise PilotRunContractError("pilot_requires_a_140_day_deep_probe")
        qa_by_request = {item.request.request_id: item for item in qa}
        if len(controls) != len(qa):
            raise PilotRunContractError("pilot_requires_one_native_control_per_qa_symbol")
        control_symbols: set[str] = set()
        for item in controls:
            source = qa_by_request.get(item.source_min1_request_id)
            if source is None:
                raise PilotRunContractError("native_control_source_request_is_unknown")
            if (
                item.request.symbol != source.request.symbol
                or item.request.venue_symbol != source.request.venue_symbol
                or item.request.start_open_ts != source.request.start_open_ts
                or item.request.end_open_ts_exclusive != source.request.end_open_ts_exclusive
            ):
                raise PilotRunContractError("native_control_range_does_not_match_min1")
            control_symbols.add(item.request.symbol)
        if control_symbols != set(qa_symbols):
            raise PilotRunContractError("native_control_symbol_coverage_is_incomplete")
        if len(set(request.venue_symbol for request in (item.request for item in qa))) != len(qa):
            raise PilotRunContractError("pilot_venue_symbols_are_not_unique")
        if len(qa_symbols) > self.budgets.max_symbols:
            raise PilotRunBudgetExceededError(
                "symbols", self.budgets.max_symbols, len(qa_symbols)
            )

        reservations = self.planned_reservations
        comparisons = (
            ("pages", self.budgets.max_total_pages, reservations["total_pages"]),
            ("rows", self.budgets.max_total_rows, reservations["total_rows"]),
            (
                "verification_attempts",
                self.budgets.max_verification_attempts,
                reservations["verification_attempts"],
            ),
            (
                "acquisition_attempts",
                self.budgets.max_acquisition_attempts,
                reservations["acquisition_attempts"],
            ),
            (
                "network_attempts",
                self.budgets.max_network_attempts,
                reservations["network_attempts"],
            ),
            (
                "raw_body_bytes",
                self.budgets.max_total_raw_body_bytes,
                reservations["raw_body_bytes"],
            ),
            (
                "logical_storage_bytes",
                self.budgets.max_total_logical_storage_bytes,
                reservations["logical_storage_bytes"],
            ),
            (
                "run_control_bytes",
                self.budgets.max_run_control_bytes,
                reservations["run_control_bytes"],
            ),
            (
                "total_output_bytes",
                self.budgets.max_total_output_bytes,
                reservations["total_output_bytes"],
            ),
            (
                "sum_shard_runtime_us",
                self.budgets.max_sum_shard_runtime_us,
                reservations["sum_shard_runtime_us"],
            ),
            (
                "run_elapsed_us",
                self.budgets.max_run_elapsed_us,
                reservations["run_elapsed_us"],
            ),
            (
                "inventory_entries",
                self.budgets.max_inventory_entries,
                reservations["inventory_entries"],
            ),
        )
        for resource, limit, observed in comparisons:
            if observed > limit:
                raise PilotRunBudgetExceededError(resource, limit, observed)
        reserved_sleep = (
            self.endpoint_verification.max_total_sleep_us
            + sum(_reserved_retry_sleep_us(item.request) for item in self.shards)
            + len(self.shards) * self.budgets.min_inter_step_spacing_us
        )
        if reserved_sleep > self.budgets.max_observed_sleep_us:
            raise PilotRunBudgetExceededError(
                "observed_sleep_us",
                self.budgets.max_observed_sleep_us,
                reserved_sleep,
            )
        base_control_bytes = len(_canonical_bytes(self.as_dict()) + b"\n")
        if base_control_bytes > _MAX_MANIFEST_BYTES:
            raise PilotRunBudgetExceededError(
                "manifest_bytes",
                _MAX_MANIFEST_BYTES,
                base_control_bytes,
            )

    @property
    def planned_reservations(self) -> dict[str, int]:
        verification = self.endpoint_verification
        acquisition_attempts = 0
        raw_bytes = verification.max_total_raw_body_bytes
        storage_bytes = verification.max_total_storage_bytes
        shard_runtime = 0
        inventory_entries = 2 * verification.max_network_attempts + 5
        total_pages = 0
        total_rows = 0
        for item in self.shards:
            request = item.request
            attempts = request.required_pages * request.resource_limits.max_attempts_per_page
            acquisition_attempts += attempts
            total_pages += request.required_pages
            total_rows += request.expected_row_count
            raw_bytes += min(
                request.resource_limits.max_total_raw_body_bytes,
                attempts * request.resource_limits.max_raw_body_bytes_per_attempt,
            )
            storage_bytes += request.resource_limits.max_logical_storage_bytes
            shard_runtime += request.resource_limits.max_collection_runtime_us
            inventory_entries += 2 * attempts + 6
        network_attempts = verification.max_network_attempts + acquisition_attempts
        run_control_bytes = _planned_run_control_bytes(len(self.shards))
        total_output = storage_bytes + run_control_bytes
        wall = (
            verification.max_runtime_us
            + shard_runtime
            + len(self.shards) * self.budgets.min_inter_step_spacing_us
        )
        reserved_sleep = (
            verification.max_total_sleep_us
            + sum(_reserved_retry_sleep_us(item.request) for item in self.shards)
            + len(self.shards) * self.budgets.min_inter_step_spacing_us
        )
        return {
            "symbols": len(
                {item.request.symbol for item in self.shards if item.role == "qa_min1"}
            ),
            "shards": len(self.shards),
            "total_pages": total_pages,
            "total_rows": total_rows,
            "verification_attempts": verification.max_network_attempts,
            "acquisition_attempts": acquisition_attempts,
            "network_attempts": network_attempts,
            "raw_body_bytes": raw_bytes,
            "logical_storage_bytes": storage_bytes,
            "run_control_bytes": run_control_bytes,
            "total_output_bytes": total_output,
            "sum_shard_runtime_us": shard_runtime,
            "run_elapsed_us": wall,
            "observed_sleep_us": reserved_sleep,
            "inventory_entries": (
                inventory_entries
                + _planned_run_control_entry_count(len(self.shards))
            ),
        }

    @property
    def manifest_hash(self) -> str:
        return _frozen_contract_hash(self)

    @property
    def manifest_identity(self) -> str:
        return f"{self.contract_version}.{self.manifest_hash}"

    def as_dict(self) -> dict[str, object]:
        endpoint = self.endpoint_verification.probe_request.endpoint_contract
        candidate_limits = candidate_history_resource_limits_v1()
        candidate_retry = candidate_history_retry_policy_v1()
        profile = self.endpoint_verification.probe_request.storage_profile
        return {
            "contract_version": self.contract_version,
            "purpose": "p2_public_qa_data_mechanics_only",
            "not_edge_evidence": True,
            "canonical_serialization": "canonical_json_utf8_lf_v1",
            "authorization_requirement": "detached_explicit_u5",
            "u5_granted_by_manifest": False,
            "forbidden_scopes": list(_FORBIDDEN_SCOPES),
            "repository_commit": self.repository_commit,
            "repository_worktree_requirement": "clean",
            "repository_tree_receipt_hash": self.repository_tree_receipt_hash,
            "created_at_us": self.created_at_us,
            "parent_master_plan": {
                "path": self.parent_master_plan_path,
                "sha256": self.parent_master_plan_sha256,
            },
            "parent_adr": {
                "path": self.parent_adr_path,
                "sha256": self.parent_adr_sha256,
            },
            "output_root_locator": self.output_root_locator,
            "shard_executor_contract_version": self.shard_executor_contract_version,
            "shard_executor_contract_hash": self.shard_executor_contract_hash,
            "endpoint_verification": self.endpoint_verification.as_dict(),
            "shards": [item.as_dict() for item in self.shards],
            "budgets": self.budgets.as_dict(),
            "planned_reservations": self.planned_reservations,
            "execution_policy": {
                "max_active_shards": 1,
                "max_in_flight_http_attempts": 1,
                "probe_before_acquisition": True,
                "probe_requires_official_and_live_evidence": True,
                "probe_requires_detached_anchor": True,
                "fresh_root_per_request": True,
                "preflight_before_every_network_step": True,
                "reserve_worst_case_before_step": True,
                "resume_incomplete_shard": False,
                "repair_or_promote_partial": False,
                "partial_run_is_success": False,
                "full_universe_or_p3_claim": False,
            },
            "contract_bindings": {
                "candidate_endpoint_version": MEXC_FUTURES_ENDPOINT_CONTRACT_VERSION,
                "candidate_endpoint_hash": endpoint.contract_hash,
                "candidate_endpoint_identity": endpoint.endpoint_identity,
                "candidate_endpoint_status": endpoint.verification_status,
                "candidate_resource_limits_version": HISTORY_RESOURCE_LIMITS_VERSION,
                "candidate_resource_limits_hash": candidate_limits.contract_hash,
                "candidate_retry_policy_version": HISTORY_RETRY_POLICY_VERSION,
                "candidate_retry_policy_hash": candidate_retry.contract_hash,
                "transport_version": MEXC_FUTURES_TRANSPORT_CONTRACT_VERSION,
                "transport_hash": mexc_futures_transport_contract_hash(),
                "strict_history_version": STRICT_HISTORY_V2_CONTRACT_VERSION,
                "strict_history_hash": strict_history_v2_contract_hash(),
                "aggregation_version": MIN1_AGGREGATION_CONTRACT_VERSION,
                "aggregation_hash": min1_aggregation_contract_hash(),
                "storage_profile": profile,
                "storage_profile_hash": storage_profile_hash(profile),
            },
        }

    @classmethod
    def from_dict(cls, payload: object) -> "MexcPublicQaPilotRunManifestV1":
        expected = frozenset(
            {
                "contract_version",
                "purpose",
                "not_edge_evidence",
                "canonical_serialization",
                "authorization_requirement",
                "u5_granted_by_manifest",
                "forbidden_scopes",
                "repository_commit",
                "repository_worktree_requirement",
                "repository_tree_receipt_hash",
                "created_at_us",
                "parent_master_plan",
                "parent_adr",
                "output_root_locator",
                "shard_executor_contract_version",
                "shard_executor_contract_hash",
                "endpoint_verification",
                "shards",
                "budgets",
                "planned_reservations",
                "execution_policy",
                "contract_bindings",
            }
        )
        values = _exact_mapping(payload, expected, code="pilot_manifest_schema_mismatch")
        literals = {
            "purpose": "p2_public_qa_data_mechanics_only",
            "not_edge_evidence": True,
            "canonical_serialization": "canonical_json_utf8_lf_v1",
            "authorization_requirement": "detached_explicit_u5",
            "u5_granted_by_manifest": False,
            "forbidden_scopes": list(_FORBIDDEN_SCOPES),
            "repository_worktree_requirement": "clean",
        }
        for field, expected_value in literals.items():
            if values.pop(field) != expected_value:
                raise PilotRunContractError(f"pilot_manifest_{field}_mismatch")
        parent_master = _exact_mapping(
            values.pop("parent_master_plan"),
            frozenset({"path", "sha256"}),
            code="pilot_parent_master_schema_mismatch",
        )
        parent_adr = _exact_mapping(
            values.pop("parent_adr"),
            frozenset({"path", "sha256"}),
            code="pilot_parent_adr_schema_mismatch",
        )
        endpoint = EndpointVerificationPlanV1.from_dict(
            values.pop("endpoint_verification")
        )
        raw_shards = values.pop("shards")
        if not isinstance(raw_shards, list):
            raise PilotRunContractError("pilot_manifest_shards_must_be_an_array")
        shards = tuple(PilotShardPlanV1.from_dict(item) for item in raw_shards)
        budgets = PilotGlobalBudgetsV1.from_dict(values.pop("budgets"))
        result = cls(
            repository_commit=values.pop("repository_commit"),
            repository_tree_receipt_hash=values.pop("repository_tree_receipt_hash"),
            created_at_us=values.pop("created_at_us"),
            parent_master_plan_path=parent_master["path"],
            parent_master_plan_sha256=parent_master["sha256"],
            parent_adr_path=parent_adr["path"],
            parent_adr_sha256=parent_adr["sha256"],
            output_root_locator=values.pop("output_root_locator"),
            shard_executor_contract_version=values.pop(
                "shard_executor_contract_version"
            ),
            shard_executor_contract_hash=values.pop("shard_executor_contract_hash"),
            endpoint_verification=endpoint,
            shards=shards,
            budgets=budgets,
            contract_version=values.pop("contract_version"),
        )
        repeated = {
            "planned_reservations": result.planned_reservations,
            "execution_policy": result.as_dict()["execution_policy"],
            "contract_bindings": result.as_dict()["contract_bindings"],
        }
        for field, expected_value in repeated.items():
            if _canonical_bytes(values.pop(field)) != _canonical_bytes(expected_value):
                raise PilotRunContractError(f"pilot_manifest_{field}_mismatch")
        if values:
            raise PilotRunContractError("pilot_manifest_parser_left_unknown_fields")
        return result

    def remaining_storage_reservation(self, step_ordinal: int) -> int:
        if type(step_ordinal) is not int or step_ordinal < -1 or step_ordinal >= len(self.shards):
            raise PilotRunContractError("pilot_preflight_step_ordinal_is_invalid")
        if step_ordinal == -1:
            return self.planned_reservations["total_output_bytes"]
        return (
            sum(
                item.request.resource_limits.max_logical_storage_bytes
                for item in self.shards[step_ordinal:]
            )
            + self.planned_reservations["run_control_bytes"]
        )

    def remaining_run_elapsed_reservation(
        self,
        step_ordinal: int,
        *,
        intent_anchor_us: int,
        previous_completed_at_us: int | None = None,
    ) -> int:
        if type(step_ordinal) is not int or step_ordinal < -1 or step_ordinal >= len(self.shards):
            raise PilotRunContractError("pilot_remaining_run_step_is_invalid")
        anchor = _strict_int(
            intent_anchor_us,
            field="pilot_remaining_run_intent_anchor_us",
            minimum=1,
        )
        if step_ordinal == -1:
            if previous_completed_at_us is not None:
                raise PilotRunContractError(
                    "pilot_endpoint_remaining_run_has_previous_completion"
                )
            return self.remaining_run_after_stage_start(-1)
        previous = _strict_int(
            previous_completed_at_us,
            field="pilot_remaining_run_previous_completed_at_us",
            minimum=1,
        )
        spacing = self.budgets.min_inter_step_spacing_us
        wait_before_current = max(0, previous + spacing - anchor)
        return (
            wait_before_current
            + self.remaining_run_after_stage_start(step_ordinal)
        )

    def remaining_run_after_stage_start(self, step_ordinal: int) -> int:
        if type(step_ordinal) is not int or step_ordinal < -1 or step_ordinal >= len(self.shards):
            raise PilotRunContractError("pilot_remaining_stage_step_is_invalid")
        if step_ordinal == -1:
            return self.planned_reservations["run_elapsed_us"]
        remaining_shards = self.shards[step_ordinal:]
        return sum(
            item.request.resource_limits.max_collection_runtime_us
            for item in remaining_shards
        ) + max(0, len(remaining_shards) - 1) * self.budgets.min_inter_step_spacing_us

    def remaining_fresh_roots(self, step_ordinal: int) -> tuple[str, ...]:
        self.remaining_storage_reservation(step_ordinal)
        if step_ordinal == -1:
            return (
                self.endpoint_verification.relative_artifact_root,
                *(item.relative_artifact_root for item in self.shards),
            )
        return tuple(item.relative_artifact_root for item in self.shards[step_ordinal:])


@dataclass(frozen=True)
class U5PublicPilotAuthorizationReceiptV1:
    manifest_hash: str
    manifest_identity: str
    authority_id: str
    orchestrator_session_id: str
    authorized_at_us: int
    expires_at_us: int
    allowed_domains: tuple[str, ...]
    allowed_operations: tuple[str, ...]
    max_network_attempts: int
    max_total_raw_body_bytes: int
    max_total_output_bytes: int
    max_run_elapsed_us: int
    storage_profile: str
    storage_profile_hash: str
    windows_sudden_power_loss_boundary_accepted: bool
    restart_network_policy: str
    external_authority_evidence_hash: str
    forbidden_scopes: tuple[str, ...] = _FORBIDDEN_SCOPES
    contract_version: str = U5_AUTHORIZATION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != U5_AUTHORIZATION_RECEIPT_VERSION:
            raise PilotRunContractError("u5_receipt_version_mismatch")
        _digest(self.manifest_hash, field="u5_manifest_hash")
        expected_identity = f"{PILOT_RUN_CONTRACT_VERSION}.{self.manifest_hash}"
        if self.manifest_identity != expected_identity:
            raise PilotRunContractError("u5_manifest_identity_mismatch")
        _safe_identifier(self.authority_id, field="u5_authority_id")
        _safe_identifier(
            self.orchestrator_session_id,
            field="u5_orchestrator_session_id",
        )
        start = _strict_int(self.authorized_at_us, field="u5_authorized_at_us", minimum=1)
        end = _strict_int(self.expires_at_us, field="u5_expires_at_us", minimum=1)
        if end <= start:
            raise PilotRunContractError("u5_authorization_window_is_invalid")
        if self.allowed_domains != ("www.mexc.com", "api.mexc.com"):
            raise PilotRunContractError("u5_allowed_domains_mismatch")
        if self.allowed_operations != _AUTHORIZED_OPERATIONS:
            raise PilotRunContractError("u5_allowed_operations_mismatch")
        for field in (
            "max_network_attempts",
            "max_total_raw_body_bytes",
            "max_total_output_bytes",
            "max_run_elapsed_us",
        ):
            _strict_int(getattr(self, field), field=f"u5_{field}", minimum=1)
        _safe_identifier(self.storage_profile, field="u5_storage_profile")
        _digest(self.storage_profile_hash, field="u5_storage_profile_hash")
        if type(self.windows_sudden_power_loss_boundary_accepted) is not bool:
            raise PilotRunContractError("u5_storage_risk_decision_is_invalid")
        if self.restart_network_policy != "forbid_network_after_process_restart":
            raise PilotRunContractError("u5_restart_network_policy_mismatch")
        _digest(
            self.external_authority_evidence_hash,
            field="u5_external_authority_evidence_hash",
        )
        if self.forbidden_scopes != _FORBIDDEN_SCOPES:
            raise PilotRunContractError("u5_forbidden_scope_mismatch")

    def validate_for(
        self,
        manifest: MexcPublicQaPilotRunManifestV1,
        *,
        now_us: int,
        require_full_run_window: bool = False,
    ) -> None:
        _strict_int(now_us, field="u5_validation_now_us", minimum=1)
        if type(require_full_run_window) is not bool:
            raise PilotRunContractError("u5_full_run_window_flag_is_invalid")
        if self.manifest_hash != manifest.manifest_hash:
            raise PilotRunAuthorizationError("u5_receipt_is_for_another_manifest")
        if self.authorized_at_us < manifest.created_at_us:
            raise PilotRunAuthorizationError("u5_receipt_precedes_manifest")
        if not self.authorized_at_us <= now_us < self.expires_at_us:
            raise PilotRunAuthorizationError("u5_receipt_is_not_current")
        budgets = manifest.budgets
        caps = (
            (self.max_network_attempts, budgets.max_network_attempts),
            (self.max_total_raw_body_bytes, budgets.max_total_raw_body_bytes),
            (self.max_total_output_bytes, budgets.max_total_output_bytes),
            (self.max_run_elapsed_us, budgets.max_run_elapsed_us),
        )
        if any(authority > manifest_cap for authority, manifest_cap in caps):
            raise PilotRunAuthorizationError("u5_receipt_is_broader_than_manifest")
        planned = manifest.planned_reservations
        required = (
            (self.max_network_attempts, planned["network_attempts"]),
            (self.max_total_raw_body_bytes, planned["raw_body_bytes"]),
            (self.max_total_output_bytes, planned["total_output_bytes"]),
            (self.max_run_elapsed_us, planned["run_elapsed_us"]),
        )
        if any(authority < needed for authority, needed in required):
            raise PilotRunAuthorizationError("u5_receipt_cannot_cover_planned_run")
        if (
            require_full_run_window
            and now_us + planned["run_elapsed_us"] >= self.expires_at_us
        ):
            raise PilotRunAuthorizationError(
                "u5_window_cannot_cover_planned_run"
            )
        request = manifest.endpoint_verification.probe_request
        if (
            self.storage_profile != request.storage_profile
            or self.storage_profile_hash != request.storage_profile_hash
        ):
            raise PilotRunAuthorizationError("u5_storage_profile_mismatch")
        if (
            request.storage_profile == "windows_ntfs_hardlink_best_effort_v1"
            and self.windows_sudden_power_loss_boundary_accepted is not True
        ):
            raise PilotRunAuthorizationError("u5_windows_storage_boundary_not_accepted")

    @property
    def receipt_hash(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "manifest_hash": self.manifest_hash,
            "manifest_identity": self.manifest_identity,
            "authority_id": self.authority_id,
            "orchestrator_session_id": self.orchestrator_session_id,
            "authorized_at_us": self.authorized_at_us,
            "expires_at_us": self.expires_at_us,
            "allowed_domains": list(self.allowed_domains),
            "allowed_operations": list(self.allowed_operations),
            "max_network_attempts": self.max_network_attempts,
            "max_total_raw_body_bytes": self.max_total_raw_body_bytes,
            "max_total_output_bytes": self.max_total_output_bytes,
            "max_run_elapsed_us": self.max_run_elapsed_us,
            "storage_profile": self.storage_profile,
            "storage_profile_hash": self.storage_profile_hash,
            "windows_sudden_power_loss_boundary_accepted": self.windows_sudden_power_loss_boundary_accepted,
            "restart_network_policy": self.restart_network_policy,
            "external_authority_evidence_hash": self.external_authority_evidence_hash,
            "forbidden_scopes": list(self.forbidden_scopes),
            "public_credential_free_scope_only": True,
            "receipt_is_not_self_authorizing": True,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "U5PublicPilotAuthorizationReceiptV1":
        expected = frozenset(
            {
                *cls.__dataclass_fields__,
                "public_credential_free_scope_only",
                "receipt_is_not_self_authorizing",
            }
        )
        values = _exact_mapping(payload, expected, code="u5_receipt_schema_mismatch")
        if values.pop("public_credential_free_scope_only") is not True or values.pop(
            "receipt_is_not_self_authorizing"
        ) is not True:
            raise PilotRunContractError("u5_receipt_scope_literal_mismatch")
        for field in ("allowed_domains", "allowed_operations", "forbidden_scopes"):
            raw = values[field]
            if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
                raise PilotRunContractError(f"u5_{field}_is_invalid")
            values[field] = tuple(raw)
        return cls(**values)


@dataclass(frozen=True)
class PilotDiskPreflightReceiptV1:
    manifest_hash: str
    authorization_receipt_hash: str
    step_ordinal: int
    checked_at_us: int
    valid_until_us: int
    output_root_locator: str
    volume_identity: str
    storage_profile: str
    storage_profile_hash: str
    free_bytes_before: int
    reserved_bytes: int
    free_bytes_after_reservation: int
    fresh_relative_roots: tuple[str, ...]
    path_chain_reparse_free: bool
    local_fixed_volume: bool
    same_volume_publication: bool
    hardlink_create_new_supported: bool
    outcome: str = "passed"
    contract_version: str = PILOT_PREFLIGHT_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_PREFLIGHT_RECEIPT_VERSION:
            raise PilotRunContractError("pilot_preflight_version_mismatch")
        _digest(self.manifest_hash, field="preflight_manifest_hash")
        _digest(
            self.authorization_receipt_hash,
            field="preflight_authorization_receipt_hash",
        )
        if type(self.step_ordinal) is not int or self.step_ordinal < -1:
            raise PilotRunContractError("pilot_preflight_step_is_invalid")
        checked = _strict_int(self.checked_at_us, field="preflight_checked_at_us", minimum=1)
        valid = _strict_int(self.valid_until_us, field="preflight_valid_until_us", minimum=1)
        if valid <= checked:
            raise PilotRunContractError("pilot_preflight_window_is_invalid")
        object.__setattr__(self, "output_root_locator", _output_root_locator(self.output_root_locator))
        _safe_identifier(self.volume_identity, field="preflight_volume_identity")
        _safe_identifier(self.storage_profile, field="preflight_storage_profile")
        _digest(self.storage_profile_hash, field="preflight_storage_profile_hash")
        before = _strict_int(self.free_bytes_before, field="preflight_free_bytes_before")
        reserved = _strict_int(self.reserved_bytes, field="preflight_reserved_bytes")
        after = _strict_int(
            self.free_bytes_after_reservation,
            field="preflight_free_bytes_after_reservation",
        )
        if after != before - reserved or before < reserved:
            raise PilotRunContractError("pilot_preflight_disk_arithmetic_mismatch")
        if not isinstance(self.fresh_relative_roots, tuple) or not self.fresh_relative_roots:
            raise PilotRunContractError("pilot_preflight_roots_are_not_immutable")
        normalized = tuple(
            _relative_root(item, field="preflight_fresh_root")
            for item in self.fresh_relative_roots
        )
        if normalized != self.fresh_relative_roots or len(set(normalized)) != len(normalized):
            raise PilotRunContractError("pilot_preflight_roots_are_invalid")
        for field in (
            "path_chain_reparse_free",
            "local_fixed_volume",
            "same_volume_publication",
            "hardlink_create_new_supported",
        ):
            if getattr(self, field) is not True:
                raise PilotRunPreflightError(f"{field}_is_required")
        if self.outcome != "passed":
            raise PilotRunPreflightError("pilot_preflight_did_not_pass")

    def validate_for(
        self,
        manifest: MexcPublicQaPilotRunManifestV1,
        authorization: U5PublicPilotAuthorizationReceiptV1,
        *,
        now_us: int,
    ) -> None:
        _strict_int(now_us, field="preflight_validation_now_us", minimum=1)
        if self.manifest_hash != manifest.manifest_hash:
            raise PilotRunPreflightError("preflight_is_for_another_manifest")
        if self.authorization_receipt_hash != authorization.receipt_hash:
            raise PilotRunPreflightError("preflight_is_for_another_authorization")
        if self.checked_at_us < authorization.authorized_at_us:
            raise PilotRunPreflightError("preflight_precedes_authorization")
        if not self.checked_at_us <= now_us < self.valid_until_us:
            raise PilotRunPreflightError("preflight_is_not_current")
        if self.valid_until_us - self.checked_at_us > manifest.budgets.max_preflight_age_us:
            raise PilotRunPreflightError("preflight_freshness_exceeds_manifest")
        request = manifest.endpoint_verification.probe_request
        if (
            self.output_root_locator != manifest.output_root_locator
            or self.storage_profile != request.storage_profile
            or self.storage_profile_hash != request.storage_profile_hash
        ):
            raise PilotRunPreflightError("preflight_storage_binding_mismatch")
        expected_reservation = manifest.remaining_storage_reservation(self.step_ordinal)
        if self.reserved_bytes != expected_reservation:
            raise PilotRunPreflightError("preflight_reservation_mismatch")
        if self.fresh_relative_roots != manifest.remaining_fresh_roots(self.step_ordinal):
            raise PilotRunPreflightError("preflight_fresh_root_set_mismatch")
        if (
            self.step_ordinal == -1
            and self.free_bytes_before
            < manifest.budgets.min_free_disk_bytes_before_run
        ):
            raise PilotRunPreflightError("preflight_free_space_below_manifest_minimum")
        if (
            self.free_bytes_after_reservation
            < manifest.budgets.required_free_disk_bytes_after_reservation
        ):
            raise PilotRunPreflightError("preflight_post_reservation_space_is_low")

    @property
    def receipt_hash(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "manifest_hash": self.manifest_hash,
            "authorization_receipt_hash": self.authorization_receipt_hash,
            "step_ordinal": self.step_ordinal,
            "checked_at_us": self.checked_at_us,
            "valid_until_us": self.valid_until_us,
            "output_root_locator": self.output_root_locator,
            "volume_identity": self.volume_identity,
            "storage_profile": self.storage_profile,
            "storage_profile_hash": self.storage_profile_hash,
            "free_bytes_before": self.free_bytes_before,
            "reserved_bytes": self.reserved_bytes,
            "free_bytes_after_reservation": self.free_bytes_after_reservation,
            "fresh_relative_roots": list(self.fresh_relative_roots),
            "path_chain_reparse_free": self.path_chain_reparse_free,
            "local_fixed_volume": self.local_fixed_volume,
            "same_volume_publication": self.same_volume_publication,
            "hardlink_create_new_supported": self.hardlink_create_new_supported,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PilotDiskPreflightReceiptV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_preflight_schema_mismatch",
        )
        roots = values["fresh_relative_roots"]
        if not isinstance(roots, list) or not all(isinstance(item, str) for item in roots):
            raise PilotRunContractError("pilot_preflight_roots_are_invalid")
        values["fresh_relative_roots"] = tuple(roots)
        return cls(**values)


@dataclass(frozen=True)
class PilotIntentDurabilityReceiptV1:
    intent_candidate_hash: str
    intent_slot_id: str
    intent_candidate_locator: str
    intent_candidate_artifact_sha256: str
    publisher_instance_id: str
    durable_publication_receipt_hash: str
    fresh_reload_receipt_hash: str
    detached_reservation_anchor_hash: str
    published_at_us: int
    reloaded_at_us: int
    anchored_at_us: int
    published_monotonic_us: int
    reloaded_monotonic_us: int
    anchored_monotonic_us: int
    publication_outcome: str = "create_new_winner_for_this_process"
    exclusive_create_new_winner: bool = True
    fresh_reload_valid: bool = True
    contract_version: str = PILOT_INTENT_DURABILITY_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_INTENT_DURABILITY_RECEIPT_VERSION:
            raise PilotRunContractError("pilot_intent_durability_version_mismatch")
        for field in (
            "intent_candidate_hash",
            "intent_slot_id",
            "intent_candidate_artifact_sha256",
            "durable_publication_receipt_hash",
            "fresh_reload_receipt_hash",
            "detached_reservation_anchor_hash",
        ):
            _digest(getattr(self, field), field=f"pilot_intent_durability_{field}")
        object.__setattr__(
            self,
            "intent_candidate_locator",
            _relative_root(
                self.intent_candidate_locator,
                field="pilot_intent_candidate_locator",
            ),
        )
        _safe_identifier(
            self.publisher_instance_id,
            field="pilot_intent_publisher_instance_id",
        )
        published = _strict_int(
            self.published_at_us,
            field="pilot_intent_published_at_us",
            minimum=1,
        )
        reloaded = _strict_int(
            self.reloaded_at_us,
            field="pilot_intent_reloaded_at_us",
            minimum=1,
        )
        anchored = _strict_int(
            self.anchored_at_us,
            field="pilot_intent_anchored_at_us",
            minimum=1,
        )
        published_monotonic = _strict_int(
            self.published_monotonic_us,
            field="pilot_intent_published_monotonic_us",
        )
        reloaded_monotonic = _strict_int(
            self.reloaded_monotonic_us,
            field="pilot_intent_reloaded_monotonic_us",
        )
        anchored_monotonic = _strict_int(
            self.anchored_monotonic_us,
            field="pilot_intent_anchored_monotonic_us",
        )
        if not published <= reloaded <= anchored or not (
            published_monotonic <= reloaded_monotonic <= anchored_monotonic
        ):
            raise PilotRunContractError("pilot_intent_durability_order_mismatch")
        if self.publication_outcome != "create_new_winner_for_this_process":
            raise PilotRunContractError("pilot_intent_publication_outcome_mismatch")
        if self.exclusive_create_new_winner is not True:
            raise PilotRunContractError("pilot_intent_create_new_winner_is_required")
        if self.fresh_reload_valid is not True:
            raise PilotRunContractError("pilot_intent_fresh_reload_is_required")

    def validate_for(
        self,
        *,
        intent_candidate_hash: str,
        intent_slot_id: str,
        intent_candidate_locator: str,
        intent_candidate_artifact_sha256: str,
        publisher_instance_id: str,
        issued_at_us: int,
        issued_monotonic_us: int,
    ) -> None:
        if self.intent_candidate_hash != intent_candidate_hash:
            raise PilotRunTransitionError("pilot_intent_durability_subject_mismatch")
        if (
            self.intent_slot_id != intent_slot_id
            or self.intent_candidate_locator != intent_candidate_locator
            or self.intent_candidate_artifact_sha256
            != intent_candidate_artifact_sha256
            or self.publisher_instance_id != publisher_instance_id
        ):
            raise PilotRunTransitionError("pilot_intent_durability_slot_mismatch")
        if (
            self.published_at_us < issued_at_us
            or self.published_monotonic_us < issued_monotonic_us
        ):
            raise PilotRunTransitionError("pilot_intent_published_before_candidate")

    @property
    def receipt_hash(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "PilotIntentDurabilityReceiptV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_intent_durability_schema_mismatch",
        )
        return cls(**values)


@dataclass(frozen=True)
class PilotNetworkIntentV1:
    manifest_hash: str
    authorization_receipt_hash: str
    preflight_receipt_hash: str
    stage: str
    ordinal: int
    step_binding_hash: str
    relative_artifact_root: str
    clock_domain_id: str
    orchestrator_session_id: str
    publisher_instance_id: str
    issued_at_us: int
    issued_monotonic_us: int
    reserved_network_attempts: int
    reserved_raw_body_bytes: int
    reserved_storage_bytes: int
    reserved_runtime_us: int
    durability_receipt: PilotIntentDurabilityReceiptV1
    restart_policy: str = "unresolved_intent_stops_without_network"
    contract_version: str = PILOT_NETWORK_INTENT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_NETWORK_INTENT_VERSION:
            raise PilotRunContractError("pilot_network_intent_version_mismatch")
        for field in (
            "manifest_hash",
            "authorization_receipt_hash",
            "preflight_receipt_hash",
            "step_binding_hash",
        ):
            _digest(getattr(self, field), field=f"pilot_intent_{field}")
        if self.stage not in {"endpoint_verification", "shard_acquisition"}:
            raise PilotRunContractError("pilot_network_intent_stage_is_invalid")
        if type(self.ordinal) is not int or (
            self.stage == "endpoint_verification" and self.ordinal != -1
        ) or (self.stage == "shard_acquisition" and self.ordinal < 0):
            raise PilotRunContractError("pilot_network_intent_ordinal_is_invalid")
        object.__setattr__(
            self,
            "relative_artifact_root",
            _relative_root(
                self.relative_artifact_root,
                field="pilot_network_intent_root",
            ),
        )
        _safe_identifier(self.clock_domain_id, field="pilot_intent_clock_domain_id")
        _safe_identifier(
            self.orchestrator_session_id,
            field="pilot_intent_orchestrator_session_id",
        )
        _safe_identifier(
            self.publisher_instance_id,
            field="pilot_intent_publisher_instance_id",
        )
        _strict_int(self.issued_at_us, field="pilot_intent_issued_at_us", minimum=1)
        _strict_int(
            self.issued_monotonic_us,
            field="pilot_intent_issued_monotonic_us",
        )
        for field in (
            "reserved_network_attempts",
            "reserved_raw_body_bytes",
            "reserved_storage_bytes",
            "reserved_runtime_us",
        ):
            _strict_int(getattr(self, field), field=f"pilot_intent_{field}", minimum=1)
        if self.restart_policy != "unresolved_intent_stops_without_network":
            raise PilotRunContractError("pilot_network_intent_restart_policy_mismatch")
        if not isinstance(self.durability_receipt, PilotIntentDurabilityReceiptV1):
            raise PilotRunContractError("pilot_network_intent_durability_is_invalid")
        self.durability_receipt.validate_for(
            intent_candidate_hash=self.intent_candidate_hash,
            intent_slot_id=self.intent_slot_id,
            intent_candidate_locator=self.intent_candidate_locator,
            intent_candidate_artifact_sha256=self.intent_candidate_artifact_sha256,
            publisher_instance_id=self.publisher_instance_id,
            issued_at_us=self.issued_at_us,
            issued_monotonic_us=self.issued_monotonic_us,
        )

    def validate_for(
        self,
        manifest: MexcPublicQaPilotRunManifestV1,
        authorization: U5PublicPilotAuthorizationReceiptV1,
        preflight: PilotDiskPreflightReceiptV1,
    ) -> None:
        if (
            self.manifest_hash != manifest.manifest_hash
            or self.authorization_receipt_hash != authorization.receipt_hash
            or self.preflight_receipt_hash != preflight.receipt_hash
        ):
            raise PilotRunTransitionError("pilot_network_intent_binding_mismatch")
        if self.orchestrator_session_id != authorization.orchestrator_session_id:
            raise PilotRunTransitionError("pilot_network_intent_session_mismatch")
        if self.stage == "endpoint_verification":
            plan = manifest.endpoint_verification
            expected = (
                plan.plan_hash,
                plan.relative_artifact_root,
                plan.max_network_attempts,
                plan.max_total_raw_body_bytes,
                plan.max_total_storage_bytes,
                plan.max_runtime_us,
            )
        else:
            if self.ordinal >= len(manifest.shards):
                raise PilotRunTransitionError("pilot_network_intent_shard_is_unknown")
            plan = manifest.shards[self.ordinal]
            request = plan.request
            attempts = (
                request.required_pages
                * request.resource_limits.max_attempts_per_page
            )
            expected = (
                plan.plan_id,
                plan.relative_artifact_root,
                attempts,
                min(
                    request.resource_limits.max_total_raw_body_bytes,
                    attempts
                    * request.resource_limits.max_raw_body_bytes_per_attempt,
                ),
                request.resource_limits.max_logical_storage_bytes,
                request.resource_limits.max_collection_runtime_us,
            )
        observed = (
            self.step_binding_hash,
            self.relative_artifact_root,
            self.reserved_network_attempts,
            self.reserved_raw_body_bytes,
            self.reserved_storage_bytes,
            self.reserved_runtime_us,
        )
        if observed != expected:
            raise PilotRunTransitionError("pilot_network_intent_reservation_mismatch")
        authorization.validate_for(
            manifest, now_us=self.durability_receipt.anchored_at_us
        )
        preflight.validate_for(
            manifest,
            authorization,
            now_us=self.durability_receipt.anchored_at_us,
        )
        if (
            self.durability_receipt.anchored_at_us + self.reserved_runtime_us
            >= authorization.expires_at_us
        ):
            raise PilotRunAuthorizationError(
                "u5_window_cannot_cover_reserved_network_step"
            )
        if self.stage == "endpoint_verification" and (
            self.durability_receipt.anchored_at_us
            + manifest.remaining_run_elapsed_reservation(
                -1,
                intent_anchor_us=self.durability_receipt.anchored_at_us,
            )
            >= authorization.expires_at_us
        ):
            raise PilotRunAuthorizationError(
                "u5_window_cannot_cover_remaining_run"
            )
        if (
            preflight.valid_until_us
            + manifest.remaining_run_after_stage_start(self.ordinal)
            >= authorization.expires_at_us
        ):
            raise PilotRunAuthorizationError(
                "u5_preflight_window_cannot_cover_remaining_run"
            )
        if self.issued_at_us < preflight.checked_at_us:
            raise PilotRunTransitionError("pilot_network_intent_precedes_preflight")

    @property
    def intent_candidate_hash(self) -> str:
        return _sha256_payload(self.intent_candidate_payload)

    @property
    def intent_candidate_payload(self) -> dict[str, object]:
        return self.candidate_payload_for(
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "durability_receipt"
            }
        )

    @staticmethod
    def candidate_payload_for(
        *,
        manifest_hash: str,
        authorization_receipt_hash: str,
        preflight_receipt_hash: str,
        stage: str,
        ordinal: int,
        step_binding_hash: str,
        relative_artifact_root: str,
        clock_domain_id: str,
        orchestrator_session_id: str,
        publisher_instance_id: str,
        issued_at_us: int,
        issued_monotonic_us: int,
        reserved_network_attempts: int,
        reserved_raw_body_bytes: int,
        reserved_storage_bytes: int,
        reserved_runtime_us: int,
        restart_policy: str = "unresolved_intent_stops_without_network",
        contract_version: str = PILOT_NETWORK_INTENT_VERSION,
    ) -> dict[str, object]:
        return {
            "domain": "mexc_public_qa_pilot_network_intent_candidate_v1",
            "manifest_hash": manifest_hash,
            "authorization_receipt_hash": authorization_receipt_hash,
            "preflight_receipt_hash": preflight_receipt_hash,
            "stage": stage,
            "ordinal": ordinal,
            "step_binding_hash": step_binding_hash,
            "relative_artifact_root": relative_artifact_root,
            "clock_domain_id": clock_domain_id,
            "orchestrator_session_id": orchestrator_session_id,
            "publisher_instance_id": publisher_instance_id,
            "issued_at_us": issued_at_us,
            "issued_monotonic_us": issued_monotonic_us,
            "reserved_network_attempts": reserved_network_attempts,
            "reserved_raw_body_bytes": reserved_raw_body_bytes,
            "reserved_storage_bytes": reserved_storage_bytes,
            "reserved_runtime_us": reserved_runtime_us,
            "restart_policy": restart_policy,
            "contract_version": contract_version,
        }

    @staticmethod
    def candidate_hash_for(**fields: object) -> str:
        return _sha256_payload(PilotNetworkIntentV1.candidate_payload_for(**fields))

    @staticmethod
    def candidate_artifact_sha256_for(**fields: object) -> str:
        payload = PilotNetworkIntentV1.candidate_payload_for(**fields)
        return _sha256_bytes(_canonical_bytes(payload) + b"\n")

    @property
    def intent_slot_id(self) -> str:
        return self.slot_id_for(
            manifest_hash=self.manifest_hash,
            stage=self.stage,
            ordinal=self.ordinal,
        )

    @staticmethod
    def slot_id_for(*, manifest_hash: str, stage: str, ordinal: int) -> str:
        return _sha256_payload(
            {
                "domain": "mexc_public_qa_pilot_network_intent_slot_v1",
                "manifest_hash": manifest_hash,
                "stage": stage,
                "ordinal": ordinal,
            }
        )

    @property
    def intent_candidate_locator(self) -> str:
        return self.slot_locator_for(
            manifest_hash=self.manifest_hash,
            stage=self.stage,
            ordinal=self.ordinal,
        )

    @property
    def sealed_intent_locator(self) -> str:
        return self.intent_candidate_locator.removesuffix(".candidate.json") + ".sealed.json"

    @staticmethod
    def slot_locator_for(*, manifest_hash: str, stage: str, ordinal: int) -> str:
        slot_id = PilotNetworkIntentV1.slot_id_for(
            manifest_hash=manifest_hash,
            stage=stage,
            ordinal=ordinal,
        )
        step = "endpoint" if ordinal == -1 else f"shard-{ordinal:04d}"
        return (
            "run-control/network-intents/"
            f"{step}.{slot_id}.candidate.json"
        )

    @property
    def intent_candidate_artifact_sha256(self) -> str:
        return _sha256_bytes(_canonical_bytes(self.intent_candidate_payload) + b"\n")

    @property
    def intent_hash(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        return {
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "durability_receipt"
            },
            "intent_candidate_hash": self.intent_candidate_hash,
            "durability_receipt": self.durability_receipt.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PilotNetworkIntentV1":
        expected = frozenset(
            {field for field in cls.__dataclass_fields__ if field != "durability_receipt"}
            | {"intent_candidate_hash", "durability_receipt"}
        )
        values = _exact_mapping(
            payload,
            expected,
            code="pilot_network_intent_schema_mismatch",
        )
        candidate_hash = values.pop("intent_candidate_hash")
        durability = PilotIntentDurabilityReceiptV1.from_dict(
            values.pop("durability_receipt")
        )
        result = cls(durability_receipt=durability, **values)
        if candidate_hash != result.intent_candidate_hash:
            raise PilotRunContractError("pilot_network_intent_candidate_hash_mismatch")
        return result


@dataclass(frozen=True)
class EndpointVerificationReceiptV1:
    manifest_hash: str
    authorization_receipt_hash: str
    verification_plan_hash: str
    network_intent_hash: str
    clock_domain_id: str
    started_at_us: int
    completed_at_us: int
    started_monotonic_us: int
    completed_monotonic_us: int
    actual_network_attempts: int
    actual_raw_body_bytes: int
    actual_storage_bytes: int
    actual_runtime_us: int
    observed_sleep_us: int
    official_document_evidence_hash: str
    official_document_request_started_at_us: int
    official_document_fetched_at_us: int
    official_document_request_started_monotonic_us: int
    official_document_fetched_monotonic_us: int
    live_history_manifest_hash: str
    live_attempt_receipt_hash: str
    live_raw_body_sha256: str
    live_observed_rows: int
    live_probe_started_at_us: int
    live_probe_completed_at_us: int
    live_probe_started_monotonic_us: int
    live_probe_completed_monotonic_us: int
    fresh_disk_reload_completed_at_us: int
    fresh_disk_reload_completed_monotonic_us: int
    output_inventory_hash: str
    output_inventory_entries: int
    detached_anchor_receipt_hash: str
    detached_anchor_at_us: int
    detached_anchor_monotonic_us: int
    tls_verified: bool = True
    redirects_followed: bool = False
    credentials_used: bool = False
    trust_env: bool = False
    http_status: int = 200
    body_complete: bool = True
    api_success: bool = True
    api_code: int = 0
    exact_grid_and_schema_valid: bool = True
    fresh_disk_reload_valid: bool = True
    status: str = "verified_for_exact_run"
    contract_version: str = ENDPOINT_VERIFICATION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != ENDPOINT_VERIFICATION_RECEIPT_VERSION:
            raise PilotRunContractError("endpoint_verification_receipt_version_mismatch")
        for field in (
            "manifest_hash",
            "authorization_receipt_hash",
            "verification_plan_hash",
            "network_intent_hash",
            "official_document_evidence_hash",
            "live_history_manifest_hash",
            "live_attempt_receipt_hash",
            "live_raw_body_sha256",
            "output_inventory_hash",
            "detached_anchor_receipt_hash",
        ):
            _digest(getattr(self, field), field=field)
        _safe_identifier(self.clock_domain_id, field="verification_clock_domain_id")
        start = _strict_int(self.started_at_us, field="verification_started_at_us", minimum=1)
        end = _strict_int(self.completed_at_us, field="verification_completed_at_us", minimum=1)
        mono_start = _strict_int(
            self.started_monotonic_us,
            field="verification_started_monotonic_us",
        )
        mono_end = _strict_int(
            self.completed_monotonic_us,
            field="verification_completed_monotonic_us",
        )
        if end < start or mono_end < mono_start:
            raise PilotRunContractError("endpoint_verification_timing_is_invalid")
        for field in (
            "actual_network_attempts",
            "actual_raw_body_bytes",
            "actual_storage_bytes",
            "actual_runtime_us",
            "observed_sleep_us",
        ):
            _strict_int(getattr(self, field), field=field, minimum=0)
        if self.actual_network_attempts < 2:
            raise PilotRunContractError("endpoint_verification_did_not_cover_both_actions")
        if (
            self.actual_raw_body_bytes < 1
            or self.actual_storage_bytes < self.actual_raw_body_bytes
            or self.actual_runtime_us < 1
        ):
            raise PilotRunContractError("endpoint_verification_evidence_is_empty")
        if self.actual_runtime_us != mono_end - mono_start:
            raise PilotRunContractError("endpoint_verification_runtime_mismatch")
        document_started = _strict_int(
            self.official_document_request_started_at_us,
            field="official_document_request_started_at_us",
            minimum=1,
        )
        fetched = _strict_int(
            self.official_document_fetched_at_us,
            field="official_document_fetched_at_us",
            minimum=1,
        )
        live_started = _strict_int(
            self.live_probe_started_at_us,
            field="live_probe_started_at_us",
            minimum=1,
        )
        live_completed = _strict_int(
            self.live_probe_completed_at_us,
            field="live_probe_completed_at_us",
            minimum=1,
        )
        reload_completed = _strict_int(
            self.fresh_disk_reload_completed_at_us,
            field="fresh_disk_reload_completed_at_us",
            minimum=1,
        )
        anchor_at = _strict_int(
            self.detached_anchor_at_us,
            field="endpoint_detached_anchor_at_us",
            minimum=1,
        )
        if not (
            start
            <= document_started
            <= fetched
            <= live_started
            <= live_completed
            <= reload_completed
            <= anchor_at
            <= end
        ):
            raise PilotRunContractError("official_document_timing_is_invalid")
        document_started_monotonic = _strict_int(
            self.official_document_request_started_monotonic_us,
            field="official_document_request_started_monotonic_us",
        )
        fetched_monotonic = _strict_int(
            self.official_document_fetched_monotonic_us,
            field="official_document_fetched_monotonic_us",
        )
        live_started_monotonic = _strict_int(
            self.live_probe_started_monotonic_us,
            field="live_probe_started_monotonic_us",
        )
        live_completed_monotonic = _strict_int(
            self.live_probe_completed_monotonic_us,
            field="live_probe_completed_monotonic_us",
        )
        reload_completed_monotonic = _strict_int(
            self.fresh_disk_reload_completed_monotonic_us,
            field="fresh_disk_reload_completed_monotonic_us",
        )
        anchor_monotonic = _strict_int(
            self.detached_anchor_monotonic_us,
            field="endpoint_detached_anchor_monotonic_us",
        )
        if not (
            mono_start
            <= document_started_monotonic
            <= fetched_monotonic
            <= live_started_monotonic
            <= live_completed_monotonic
            <= reload_completed_monotonic
            <= anchor_monotonic
            <= mono_end
        ):
            raise PilotRunContractError(
                "official_document_monotonic_timing_is_invalid"
            )
        if self.live_observed_rows != 1:
            raise PilotRunContractError("endpoint_probe_row_count_mismatch")
        _strict_int(
            self.output_inventory_entries,
            field="verification_output_inventory_entries",
            minimum=1,
        )
        if self.actual_storage_bytes < self.actual_raw_body_bytes:
            raise PilotRunContractError("endpoint_verification_storage_is_inconsistent")
        if self.output_inventory_entries < 2 * self.actual_network_attempts + 1:
            raise PilotRunContractError("endpoint_verification_inventory_is_incomplete")
        expected = {
            "tls_verified": True,
            "redirects_followed": False,
            "credentials_used": False,
            "trust_env": False,
            "http_status": 200,
            "body_complete": True,
            "api_success": True,
            "api_code": 0,
            "exact_grid_and_schema_valid": True,
            "fresh_disk_reload_valid": True,
            "status": "verified_for_exact_run",
        }
        for field, value in expected.items():
            if getattr(self, field) != value or type(getattr(self, field)) is not type(value):
                raise PilotRunContractError(f"endpoint_verification_{field}_mismatch")

    def validate_for(
        self,
        manifest: MexcPublicQaPilotRunManifestV1,
        authorization: U5PublicPilotAuthorizationReceiptV1,
    ) -> None:
        plan = manifest.endpoint_verification
        if (
            self.manifest_hash != manifest.manifest_hash
            or self.authorization_receipt_hash != authorization.receipt_hash
            or self.verification_plan_hash != plan.plan_hash
        ):
            raise PilotRunTransitionError("endpoint_verification_binding_mismatch")
        comparisons = (
            ("verification_attempts", plan.max_network_attempts, self.actual_network_attempts),
            ("verification_raw_bytes", plan.max_total_raw_body_bytes, self.actual_raw_body_bytes),
            ("verification_storage_bytes", plan.max_total_storage_bytes, self.actual_storage_bytes),
            ("verification_runtime_us", plan.max_runtime_us, self.actual_runtime_us),
            ("verification_sleep_us", plan.max_total_sleep_us, self.observed_sleep_us),
            (
                "verification_inventory_entries",
                2 * plan.max_network_attempts + 5,
                self.output_inventory_entries,
            ),
        )
        for resource, limit, observed in comparisons:
            if observed > limit:
                raise PilotRunBudgetExceededError(resource, limit, observed)

    @property
    def receipt_hash(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "EndpointVerificationReceiptV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="endpoint_verification_receipt_schema_mismatch",
        )
        return cls(**values)


@dataclass(frozen=True)
class PilotShardResultV1:
    manifest_hash: str
    shard_plan_id: str
    network_intent_hash: str
    ordinal: int
    request_id: str
    relative_artifact_root: str
    clock_domain_id: str
    step_started_at_us: int
    step_completed_at_us: int
    step_started_monotonic_us: int
    step_completed_monotonic_us: int
    observed_inter_step_delay_us: int
    observed_internal_sleep_us: int
    history_manifest_hash: str
    actual_pages: int
    actual_rows: int
    actual_attempts: int
    actual_raw_body_bytes: int
    actual_logical_storage_bytes: int
    actual_collection_runtime_us: int
    output_inventory_hash: str
    output_inventory_entries: int
    output_inventory_bytes: int
    detached_shard_anchor_receipt_hash: str
    fresh_disk_reload_completed_at_us: int
    detached_shard_anchor_at_us: int
    fresh_disk_reload_completed_monotonic_us: int
    detached_shard_anchor_monotonic_us: int
    fresh_disk_reload_valid: bool = True
    restart_reconciliation_ready: bool = True
    unexpected_residue_absent: bool = True
    contract_version: str = PILOT_SHARD_RESULT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_SHARD_RESULT_VERSION:
            raise PilotRunContractError("pilot_shard_result_version_mismatch")
        for field in (
            "manifest_hash",
            "shard_plan_id",
            "network_intent_hash",
            "request_id",
            "history_manifest_hash",
            "output_inventory_hash",
            "detached_shard_anchor_receipt_hash",
        ):
            _digest(getattr(self, field), field=field)
        _strict_int(self.ordinal, field="shard_result_ordinal", minimum=0)
        object.__setattr__(
            self,
            "relative_artifact_root",
            _relative_root(self.relative_artifact_root, field="shard_result_root"),
        )
        _safe_identifier(self.clock_domain_id, field="shard_result_clock_domain_id")
        start = _strict_int(self.step_started_at_us, field="shard_step_started_at_us", minimum=1)
        end = _strict_int(self.step_completed_at_us, field="shard_step_completed_at_us", minimum=1)
        mono_start = _strict_int(
            self.step_started_monotonic_us,
            field="shard_step_started_monotonic_us",
        )
        mono_end = _strict_int(
            self.step_completed_monotonic_us,
            field="shard_step_completed_monotonic_us",
        )
        if end < start or mono_end < mono_start:
            raise PilotRunContractError("pilot_shard_result_timing_is_invalid")
        reload_at = _strict_int(
            self.fresh_disk_reload_completed_at_us,
            field="shard_fresh_reload_completed_at_us",
            minimum=1,
        )
        anchor_at = _strict_int(
            self.detached_shard_anchor_at_us,
            field="shard_detached_anchor_at_us",
            minimum=1,
        )
        reload_monotonic = _strict_int(
            self.fresh_disk_reload_completed_monotonic_us,
            field="shard_fresh_reload_completed_monotonic_us",
        )
        anchor_monotonic = _strict_int(
            self.detached_shard_anchor_monotonic_us,
            field="shard_detached_anchor_monotonic_us",
        )
        if not (
            start <= reload_at <= anchor_at <= end
            and mono_start <= reload_monotonic <= anchor_monotonic <= mono_end
        ):
            raise PilotRunContractError("pilot_shard_reload_anchor_order_mismatch")
        for field in (
            "observed_inter_step_delay_us",
            "observed_internal_sleep_us",
            "actual_pages",
            "actual_rows",
            "actual_attempts",
            "actual_raw_body_bytes",
            "actual_logical_storage_bytes",
            "actual_collection_runtime_us",
            "output_inventory_entries",
            "output_inventory_bytes",
        ):
            _strict_int(getattr(self, field), field=f"shard_result_{field}", minimum=0)
        for field in (
            "fresh_disk_reload_valid",
            "restart_reconciliation_ready",
            "unexpected_residue_absent",
        ):
            if getattr(self, field) is not True:
                raise PilotRunTransitionError(f"pilot_shard_{field}_is_required")
        if self.output_inventory_bytes != self.actual_logical_storage_bytes:
            raise PilotRunContractError("pilot_shard_inventory_byte_total_mismatch")
        if self.actual_collection_runtime_us > mono_end - mono_start:
            raise PilotRunContractError("pilot_shard_collection_runtime_exceeds_step")

    def validate_for(
        self,
        manifest: MexcPublicQaPilotRunManifestV1,
        plan: PilotShardPlanV1,
    ) -> None:
        if (
            self.manifest_hash != manifest.manifest_hash
            or self.shard_plan_id != plan.plan_id
            or self.ordinal != plan.ordinal
            or self.request_id != plan.request.request_id
            or self.relative_artifact_root != plan.relative_artifact_root
        ):
            raise PilotRunTransitionError("pilot_shard_result_binding_mismatch")
        request = plan.request
        expected_attempts = request.required_pages * request.resource_limits.max_attempts_per_page
        reserved_raw_bytes = min(
            request.resource_limits.max_total_raw_body_bytes,
            expected_attempts
            * request.resource_limits.max_raw_body_bytes_per_attempt,
        )
        comparisons = (
            ("shard_pages", request.required_pages, self.actual_pages),
            ("shard_rows", request.expected_row_count, self.actual_rows),
            ("shard_attempts", expected_attempts, self.actual_attempts),
            (
                "shard_raw_body_bytes",
                reserved_raw_bytes,
                self.actual_raw_body_bytes,
            ),
            (
                "shard_logical_storage_bytes",
                request.resource_limits.max_logical_storage_bytes,
                self.actual_logical_storage_bytes,
            ),
            (
                "shard_collection_runtime_us",
                request.resource_limits.max_collection_runtime_us,
                self.actual_collection_runtime_us,
            ),
            (
                "shard_step_runtime_us",
                request.resource_limits.max_collection_runtime_us,
                self.step_completed_monotonic_us
                - self.step_started_monotonic_us,
            ),
            (
                "shard_internal_sleep_us",
                _reserved_retry_sleep_us(request),
                self.observed_internal_sleep_us,
            ),
            (
                "shard_inventory_entries",
                2 * expected_attempts + 6,
                self.output_inventory_entries,
            ),
        )
        for resource, limit, observed in comparisons:
            if observed > limit:
                raise PilotRunBudgetExceededError(resource, limit, observed)
        if self.actual_pages != request.required_pages or self.actual_rows != request.expected_row_count:
            raise PilotRunTransitionError("pilot_shard_result_is_incomplete")
        if self.actual_attempts < request.required_pages:
            raise PilotRunTransitionError("pilot_shard_attempt_count_is_impossible")
        if self.actual_raw_body_bytes < 1 or self.actual_logical_storage_bytes < 1:
            raise PilotRunTransitionError("pilot_shard_storage_evidence_is_empty")
        if self.actual_logical_storage_bytes < self.actual_raw_body_bytes:
            raise PilotRunTransitionError("pilot_shard_storage_evidence_is_inconsistent")
        if self.output_inventory_entries < self.actual_attempts + 5:
            raise PilotRunTransitionError("pilot_shard_inventory_is_incomplete")

    @property
    def receipt_hash(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "PilotShardResultV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_shard_result_schema_mismatch",
        )
        return cls(**values)


@dataclass(frozen=True)
class PilotStepFailureReceiptV1:
    manifest_hash: str
    authorization_receipt_hash: str
    network_intent_hash: str
    stage: str
    ordinal: int
    step_binding_hash: str
    clock_domain_id: str
    step_started_at_us: int
    step_completed_at_us: int
    step_started_monotonic_us: int
    step_completed_monotonic_us: int
    actual_network_attempts: int
    actual_raw_body_bytes: int
    actual_storage_bytes: int
    actual_runtime_us: int
    observed_internal_sleep_us: int
    observed_inter_step_delay_us: int
    output_inventory_hash: str
    output_inventory_entries: int
    output_inventory_bytes: int
    error_code: str
    error_evidence_hash: str
    candidate_publication_receipt_hash: str
    candidate_reload_receipt_hash: str
    candidate_detached_anchor_hash: str
    published_at_us: int
    reloaded_at_us: int
    anchored_at_us: int
    published_monotonic_us: int
    reloaded_monotonic_us: int
    anchored_monotonic_us: int
    durable_publication_valid: bool = True
    fresh_reload_valid: bool = True
    status: str = "terminal_failure"
    contract_version: str = PILOT_STEP_FAILURE_RECEIPT_VERSION
    failure_candidate_hash: str = field(init=False)
    failure_candidate_artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_STEP_FAILURE_RECEIPT_VERSION:
            raise PilotRunContractError("pilot_step_failure_version_mismatch")
        for field in (
            "manifest_hash",
            "authorization_receipt_hash",
            "network_intent_hash",
            "step_binding_hash",
            "output_inventory_hash",
            "error_evidence_hash",
            "candidate_publication_receipt_hash",
            "candidate_reload_receipt_hash",
            "candidate_detached_anchor_hash",
        ):
            _digest(getattr(self, field), field=f"pilot_failure_{field}")
        if self.stage not in {"endpoint_verification", "shard_acquisition"}:
            raise PilotRunContractError("pilot_failure_stage_is_invalid")
        if type(self.ordinal) is not int or (
            self.stage == "endpoint_verification" and self.ordinal != -1
        ) or (self.stage == "shard_acquisition" and self.ordinal < 0):
            raise PilotRunContractError("pilot_failure_ordinal_is_invalid")
        _safe_identifier(self.clock_domain_id, field="pilot_failure_clock_domain_id")
        start = _strict_int(
            self.step_started_at_us,
            field="pilot_failure_started_at_us",
            minimum=1,
        )
        end = _strict_int(
            self.step_completed_at_us,
            field="pilot_failure_completed_at_us",
            minimum=1,
        )
        mono_start = _strict_int(
            self.step_started_monotonic_us,
            field="pilot_failure_started_monotonic_us",
        )
        mono_end = _strict_int(
            self.step_completed_monotonic_us,
            field="pilot_failure_completed_monotonic_us",
        )
        if end < start or mono_end < mono_start:
            raise PilotRunContractError("pilot_failure_timing_is_invalid")
        published_at = _strict_int(
            self.published_at_us,
            field="pilot_failure_published_at_us",
            minimum=1,
        )
        reloaded_at = _strict_int(
            self.reloaded_at_us,
            field="pilot_failure_reloaded_at_us",
            minimum=1,
        )
        anchored_at = _strict_int(
            self.anchored_at_us,
            field="pilot_failure_anchored_at_us",
            minimum=1,
        )
        published_monotonic = _strict_int(
            self.published_monotonic_us,
            field="pilot_failure_published_monotonic_us",
        )
        reloaded_monotonic = _strict_int(
            self.reloaded_monotonic_us,
            field="pilot_failure_reloaded_monotonic_us",
        )
        anchored_monotonic = _strict_int(
            self.anchored_monotonic_us,
            field="pilot_failure_anchored_monotonic_us",
        )
        if not (
            end <= published_at <= reloaded_at <= anchored_at
            and mono_end
            <= published_monotonic
            <= reloaded_monotonic
            <= anchored_monotonic
        ):
            raise PilotRunContractError(
                "pilot_failure_publication_reload_anchor_order_mismatch"
            )
        for field in (
            "actual_network_attempts",
            "actual_raw_body_bytes",
            "actual_storage_bytes",
            "actual_runtime_us",
            "observed_internal_sleep_us",
            "observed_inter_step_delay_us",
            "output_inventory_entries",
            "output_inventory_bytes",
        ):
            _strict_int(getattr(self, field), field=f"pilot_failure_{field}")
        if self.actual_network_attempts < 1:
            raise PilotRunContractError("pilot_failure_has_no_started_attempt")
        if self.actual_runtime_us != mono_end - mono_start:
            raise PilotRunContractError("pilot_failure_runtime_mismatch")
        if (
            self.actual_storage_bytes < self.actual_raw_body_bytes
            or self.actual_storage_bytes < 1
            or self.output_inventory_bytes != self.actual_storage_bytes
            or self.output_inventory_entries < 1
        ):
            raise PilotRunContractError("pilot_failure_storage_is_inconsistent")
        _safe_code(self.error_code, field="pilot_failure_error_code")
        if self.durable_publication_valid is not True:
            raise PilotRunContractError(
                "pilot_failure_durable_publication_is_required"
            )
        if self.fresh_reload_valid is not True:
            raise PilotRunContractError("pilot_failure_fresh_reload_is_required")
        if self.status != "terminal_failure":
            raise PilotRunContractError("pilot_failure_status_mismatch")
        object.__setattr__(
            self,
            "failure_candidate_hash",
            _sha256_payload(self.failure_candidate_payload),
        )
        object.__setattr__(
            self,
            "failure_candidate_artifact_sha256",
            _sha256_bytes(_canonical_bytes(self.failure_candidate_payload) + b"\n"),
        )

    @property
    def failure_candidate_payload(self) -> dict[str, object]:
        return {
            "domain": "mexc_public_qa_pilot_step_failure_candidate_v1",
            **{
                name: getattr(self, name)
                for name in (
                    "manifest_hash",
                    "authorization_receipt_hash",
                    "network_intent_hash",
                    "stage",
                    "ordinal",
                    "step_binding_hash",
                    "clock_domain_id",
                    "step_started_at_us",
                    "step_completed_at_us",
                    "step_started_monotonic_us",
                    "step_completed_monotonic_us",
                    "actual_network_attempts",
                    "actual_raw_body_bytes",
                    "actual_storage_bytes",
                    "actual_runtime_us",
                    "observed_internal_sleep_us",
                    "observed_inter_step_delay_us",
                    "output_inventory_hash",
                    "output_inventory_entries",
                    "output_inventory_bytes",
                    "error_code",
                    "error_evidence_hash",
                    "status",
                    "contract_version",
                )
            },
        }

    def validate_for(
        self,
        manifest: MexcPublicQaPilotRunManifestV1,
        authorization: U5PublicPilotAuthorizationReceiptV1,
    ) -> None:
        if (
            self.manifest_hash != manifest.manifest_hash
            or self.authorization_receipt_hash != authorization.receipt_hash
        ):
            raise PilotRunTransitionError("pilot_failure_binding_mismatch")
        if self.stage == "endpoint_verification":
            plan = manifest.endpoint_verification
            if self.step_binding_hash != plan.plan_hash:
                raise PilotRunTransitionError("pilot_failure_step_binding_mismatch")
            limits = (
                ("failure_attempts", plan.max_network_attempts, self.actual_network_attempts),
                ("failure_raw_bytes", plan.max_total_raw_body_bytes, self.actual_raw_body_bytes),
                ("failure_storage_bytes", plan.max_total_storage_bytes, self.actual_storage_bytes),
                ("failure_runtime_us", plan.max_runtime_us, self.actual_runtime_us),
                ("failure_sleep_us", plan.max_total_sleep_us, self.observed_internal_sleep_us),
            )
            if self.observed_inter_step_delay_us != 0:
                raise PilotRunTransitionError("endpoint_failure_has_inter_step_delay")
        else:
            if self.ordinal >= len(manifest.shards):
                raise PilotRunTransitionError("pilot_failure_shard_is_unknown")
            plan = manifest.shards[self.ordinal]
            request = plan.request
            if self.step_binding_hash != plan.plan_id:
                raise PilotRunTransitionError("pilot_failure_step_binding_mismatch")
            reserved_attempts = (
                request.required_pages
                * request.resource_limits.max_attempts_per_page
            )
            reserved_raw_bytes = min(
                request.resource_limits.max_total_raw_body_bytes,
                reserved_attempts
                * request.resource_limits.max_raw_body_bytes_per_attempt,
            )
            limits = (
                ("failure_attempts", reserved_attempts, self.actual_network_attempts),
                ("failure_raw_bytes", reserved_raw_bytes, self.actual_raw_body_bytes),
                ("failure_storage_bytes", request.resource_limits.max_logical_storage_bytes, self.actual_storage_bytes),
                ("failure_runtime_us", request.resource_limits.max_collection_runtime_us, self.actual_runtime_us),
                ("failure_sleep_us", _reserved_retry_sleep_us(request), self.observed_internal_sleep_us),
            )
        for resource, limit, observed in limits:
            if observed > limit:
                raise PilotRunBudgetExceededError(resource, limit, observed)

    @property
    def receipt_hash(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "PilotStepFailureReceiptV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_step_failure_schema_mismatch",
        )
        candidate_hash = values.pop("failure_candidate_hash")
        candidate_artifact = values.pop("failure_candidate_artifact_sha256")
        result = cls(**values)
        if (
            candidate_hash != result.failure_candidate_hash
            or candidate_artifact != result.failure_candidate_artifact_sha256
        ):
            raise PilotRunContractError("pilot_failure_candidate_binding_mismatch")
        return result


@dataclass(frozen=True)
class PilotRunAnchorReceiptV1:
    manifest_hash: str
    result_candidate_hash: str
    run_control_inventory_hash: str
    run_control_inventory_entries: int
    output_inventory_hash: str
    output_inventory_entries: int
    run_control_bytes: int
    total_output_bytes: int
    clock_domain_id: str
    fresh_inventory_scan_receipt_hash: str
    fresh_inventory_reload_receipt_hash: str
    fresh_inventory_scanned_at_us: int
    fresh_inventory_reloaded_at_us: int
    fresh_inventory_scanned_monotonic_us: int
    fresh_inventory_reloaded_monotonic_us: int
    external_anchor_domain_id: str
    external_anchor_evidence_hash: str
    anchored_at_us: int
    anchored_monotonic_us: int
    final_run_elapsed_us: int
    all_planned_artifacts_present: bool = True
    unexpected_artifacts_absent: bool = True
    fresh_disk_inventory_reload_valid: bool = True
    contract_version: str = PILOT_RUN_ANCHOR_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_RUN_ANCHOR_VERSION:
            raise PilotRunContractError("pilot_anchor_version_mismatch")
        _digest(self.manifest_hash, field="pilot_anchor_manifest_hash")
        _digest(self.result_candidate_hash, field="pilot_anchor_result_hash")
        _digest(
            self.run_control_inventory_hash,
            field="pilot_anchor_run_control_inventory_hash",
        )
        _strict_int(
            self.run_control_inventory_entries,
            field="pilot_anchor_run_control_inventory_entries",
            minimum=1,
        )
        _digest(self.output_inventory_hash, field="pilot_anchor_inventory_hash")
        _strict_int(
            self.output_inventory_entries,
            field="pilot_anchor_inventory_entries",
            minimum=1,
        )
        _strict_int(
            self.run_control_bytes,
            field="pilot_anchor_run_control_bytes",
            minimum=1,
        )
        _strict_int(
            self.total_output_bytes,
            field="pilot_anchor_total_output_bytes",
            minimum=1,
        )
        _safe_identifier(self.clock_domain_id, field="pilot_anchor_clock_domain_id")
        _digest(
            self.fresh_inventory_scan_receipt_hash,
            field="pilot_fresh_inventory_scan_receipt_hash",
        )
        _digest(
            self.fresh_inventory_reload_receipt_hash,
            field="pilot_fresh_inventory_reload_receipt_hash",
        )
        scanned_at = _strict_int(
            self.fresh_inventory_scanned_at_us,
            field="pilot_fresh_inventory_scanned_at_us",
            minimum=1,
        )
        reloaded_at = _strict_int(
            self.fresh_inventory_reloaded_at_us,
            field="pilot_fresh_inventory_reloaded_at_us",
            minimum=1,
        )
        scanned_monotonic = _strict_int(
            self.fresh_inventory_scanned_monotonic_us,
            field="pilot_fresh_inventory_scanned_monotonic_us",
        )
        reloaded_monotonic = _strict_int(
            self.fresh_inventory_reloaded_monotonic_us,
            field="pilot_fresh_inventory_reloaded_monotonic_us",
        )
        _safe_identifier(
            self.external_anchor_domain_id,
            field="pilot_external_anchor_domain_id",
        )
        _digest(
            self.external_anchor_evidence_hash,
            field="pilot_external_anchor_evidence_hash",
        )
        _strict_int(self.anchored_at_us, field="pilot_anchored_at_us", minimum=1)
        _strict_int(
            self.anchored_monotonic_us,
            field="pilot_anchored_monotonic_us",
        )
        if not (
            scanned_at <= reloaded_at <= self.anchored_at_us
            and scanned_monotonic
            <= reloaded_monotonic
            <= self.anchored_monotonic_us
        ):
            raise PilotRunContractError(
                "pilot_final_inventory_scan_reload_anchor_order_mismatch"
            )
        for field in (
            "all_planned_artifacts_present",
            "unexpected_artifacts_absent",
            "fresh_disk_inventory_reload_valid",
        ):
            if getattr(self, field) is not True:
                raise PilotRunContractError(
                    f"pilot_final_inventory_{field}_is_required"
                )
        _strict_int(
            self.final_run_elapsed_us,
            field="pilot_final_run_elapsed_us",
            minimum=1,
        )

    @property
    def receipt_hash(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "PilotRunAnchorReceiptV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_anchor_schema_mismatch",
        )
        return cls(**values)


@dataclass(frozen=True)
class PilotRunStateV1:
    manifest: MexcPublicQaPilotRunManifestV1
    authorization: U5PublicPilotAuthorizationReceiptV1 | None = None
    preflight_receipts: tuple[PilotDiskPreflightReceiptV1, ...] = ()
    network_intents: tuple[PilotNetworkIntentV1, ...] = ()
    endpoint_verification: EndpointVerificationReceiptV1 | None = None
    shard_results: tuple[PilotShardResultV1, ...] = ()
    failure_receipt: PilotStepFailureReceiptV1 | None = None
    final_anchor: PilotRunAnchorReceiptV1 | None = None
    stop_reason: str | None = None
    stop_evidence_hash: str | None = None
    contract_version: str = PILOT_RUN_STATE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_RUN_STATE_VERSION:
            raise PilotRunContractError("pilot_state_version_mismatch")
        if not isinstance(self.manifest, MexcPublicQaPilotRunManifestV1):
            raise PilotRunContractError("pilot_state_manifest_is_invalid")
        if not isinstance(self.preflight_receipts, tuple) or not all(
            isinstance(item, PilotDiskPreflightReceiptV1)
            for item in self.preflight_receipts
        ):
            raise PilotRunContractError("pilot_state_preflights_are_not_immutable")
        if not isinstance(self.shard_results, tuple) or not all(
            isinstance(item, PilotShardResultV1) for item in self.shard_results
        ):
            raise PilotRunContractError("pilot_state_results_are_not_immutable")
        if not isinstance(self.network_intents, tuple) or not all(
            isinstance(item, PilotNetworkIntentV1) for item in self.network_intents
        ):
            raise PilotRunContractError("pilot_state_network_intents_are_not_immutable")
        if self.failure_receipt is not None:
            if not isinstance(self.failure_receipt, PilotStepFailureReceiptV1):
                raise PilotRunContractError("pilot_state_failure_receipt_is_invalid")
            if (
                self.stop_reason != self.failure_receipt.error_code
                or self.stop_evidence_hash != self.failure_receipt.receipt_hash
            ):
                raise PilotRunContractError("pilot_failure_stop_projection_mismatch")
            if self.final_anchor is not None:
                raise PilotRunContractError("pilot_failed_state_cannot_be_complete")
        elif self.stop_reason is None:
            if self.stop_evidence_hash is not None:
                raise PilotRunContractError("pilot_stop_evidence_without_reason")
        else:
            _safe_code(self.stop_reason, field="pilot_stop_reason")
            _digest(self.stop_evidence_hash, field="pilot_stop_evidence_hash")
            if self.final_anchor is not None:
                raise PilotRunContractError("pilot_stopped_state_cannot_be_complete")
        if self.authorization is None:
            if any(
                (
                    self.preflight_receipts,
                    self.network_intents,
                    self.endpoint_verification is not None,
                    self.shard_results,
                    self.failure_receipt is not None,
                    self.final_anchor is not None,
                )
            ):
                raise PilotRunTransitionError("pilot_state_advanced_without_authorization")
            self._validate_run_control_budget()
            return
        if not isinstance(self.authorization, U5PublicPilotAuthorizationReceiptV1):
            raise PilotRunContractError("pilot_state_authorization_is_invalid")
        if self.authorization.manifest_hash != self.manifest.manifest_hash:
            raise PilotRunTransitionError("pilot_state_authorization_binding_mismatch")
        self.authorization.validate_for(
            self.manifest,
            now_us=self.authorization.authorized_at_us,
            require_full_run_window=True,
        )
        expected_preflight_ordinals = (
            ()
            if not self.preflight_receipts
            else (-1, *range(len(self.preflight_receipts) - 1))
        )
        if tuple(item.step_ordinal for item in self.preflight_receipts) != expected_preflight_ordinals:
            raise PilotRunTransitionError("pilot_preflight_sequence_is_invalid")
        if self.preflight_receipts and any(
            item.volume_identity != self.preflight_receipts[0].volume_identity
            for item in self.preflight_receipts[1:]
        ):
            raise PilotRunTransitionError("pilot_preflight_volume_identity_changed")
        for receipt in self.preflight_receipts:
            receipt.validate_for(
                self.manifest,
                self.authorization,
                now_us=receipt.checked_at_us,
            )
        if self.endpoint_verification is None:
            if len(self.network_intents) > 1:
                raise PilotRunTransitionError("pilot_endpoint_has_multiple_intents")
            endpoint_intent = self.network_intents[0] if self.network_intents else None
            if endpoint_intent is not None:
                if (
                    len(self.preflight_receipts) != 1
                    or endpoint_intent.stage != "endpoint_verification"
                ):
                    raise PilotRunTransitionError("pilot_endpoint_intent_is_out_of_order")
                endpoint_intent.validate_for(
                    self.manifest,
                    self.authorization,
                    self.preflight_receipts[0],
                )
            if self.failure_receipt is None:
                if len(self.preflight_receipts) > 1 or self.shard_results or self.final_anchor:
                    raise PilotRunTransitionError("pilot_state_skipped_endpoint_verification")
                self._validate_run_control_budget()
                if endpoint_intent is not None:
                    self._validate_actual_totals()
                return
            if (
                len(self.preflight_receipts) != 1
                or endpoint_intent is None
                or self.shard_results
                or self.final_anchor
                or self.failure_receipt.stage != "endpoint_verification"
                or self.failure_receipt.network_intent_hash
                != endpoint_intent.intent_hash
            ):
                raise PilotRunTransitionError("pilot_state_skipped_endpoint_verification")
            failure = self.failure_receipt
            failure.validate_for(self.manifest, self.authorization)
            self.authorization.validate_for(
                self.manifest, now_us=failure.step_started_at_us
            )
            self.authorization.validate_for(
                self.manifest, now_us=failure.step_completed_at_us
            )
            self.preflight_receipts[0].validate_for(
                self.manifest,
                self.authorization,
                now_us=failure.step_started_at_us,
            )
            if failure.step_started_at_us < self.preflight_receipts[0].checked_at_us:
                raise PilotRunTransitionError("pilot_failure_started_before_preflight")
            if (
                failure.clock_domain_id != endpoint_intent.clock_domain_id
                or failure.step_started_at_us
                < endpoint_intent.durability_receipt.anchored_at_us
                or failure.step_started_monotonic_us
                < endpoint_intent.durability_receipt.anchored_monotonic_us
            ):
                raise PilotRunTransitionError("pilot_failure_precedes_network_intent")
            self._validate_actual_totals()
            return
        if not isinstance(self.endpoint_verification, EndpointVerificationReceiptV1):
            raise PilotRunContractError("pilot_state_endpoint_receipt_is_invalid")
        if not self.preflight_receipts or self.preflight_receipts[0].step_ordinal != -1:
            raise PilotRunTransitionError("pilot_endpoint_verification_lacks_preflight")
        if not self.network_intents:
            raise PilotRunTransitionError("pilot_endpoint_verification_lacks_intent")
        endpoint_intent = self.network_intents[0]
        endpoint_intent.validate_for(
            self.manifest,
            self.authorization,
            self.preflight_receipts[0],
        )
        if (
            endpoint_intent.stage != "endpoint_verification"
            or self.endpoint_verification.network_intent_hash
            != endpoint_intent.intent_hash
            or self.endpoint_verification.clock_domain_id
            != endpoint_intent.clock_domain_id
            or self.endpoint_verification.started_at_us
            < endpoint_intent.durability_receipt.anchored_at_us
            or self.endpoint_verification.started_monotonic_us
            < endpoint_intent.durability_receipt.anchored_monotonic_us
        ):
            raise PilotRunTransitionError("pilot_endpoint_intent_binding_mismatch")
        self.endpoint_verification.validate_for(self.manifest, self.authorization)
        self.authorization.validate_for(
            self.manifest, now_us=self.endpoint_verification.started_at_us
        )
        self.authorization.validate_for(
            self.manifest, now_us=self.endpoint_verification.completed_at_us
        )
        self.preflight_receipts[0].validate_for(
            self.manifest,
            self.authorization,
            now_us=self.endpoint_verification.started_at_us,
        )
        if (
            self.endpoint_verification.started_at_us
            < self.preflight_receipts[0].checked_at_us
        ):
            raise PilotRunTransitionError("pilot_endpoint_started_before_preflight")
        if len(self.shard_results) > len(self.manifest.shards):
            raise PilotRunTransitionError("pilot_state_has_too_many_shard_results")
        if len(self.preflight_receipts) not in {
            len(self.shard_results) + 1,
            len(self.shard_results) + 2,
        }:
            raise PilotRunTransitionError("pilot_state_preflight_result_count_mismatch")
        if len(self.preflight_receipts) == len(self.shard_results) + 2 and len(
            self.shard_results
        ) >= len(self.manifest.shards):
            raise PilotRunTransitionError("pilot_state_has_unused_terminal_preflight")
        completed_intent_count = 1 + len(self.shard_results)
        expected_intent_counts = (
            {completed_intent_count + 1}
            if self.failure_receipt is not None
            else {completed_intent_count, completed_intent_count + 1}
        )
        if len(self.network_intents) not in expected_intent_counts:
            raise PilotRunTransitionError("pilot_network_intent_count_mismatch")
        if (
            len(self.network_intents) == completed_intent_count + 1
            and len(self.preflight_receipts) != len(self.shard_results) + 2
        ):
            raise PilotRunTransitionError("pilot_active_intent_lacks_preflight")
        previous_epoch = self.endpoint_verification.completed_at_us
        previous_monotonic = self.endpoint_verification.completed_monotonic_us
        clock_domain = self.endpoint_verification.clock_domain_id
        for index, result in enumerate(self.shard_results):
            plan = self.manifest.shards[index]
            intent = self.network_intents[index + 1]
            intent.validate_for(
                self.manifest,
                self.authorization,
                self.preflight_receipts[index + 1],
            )
            result.validate_for(self.manifest, plan)
            if result.ordinal != index:
                raise PilotRunTransitionError("pilot_shard_results_are_out_of_order")
            if result.clock_domain_id != clock_domain:
                raise PilotRunTransitionError("pilot_run_clock_domain_changed")
            if (
                intent.stage != "shard_acquisition"
                or intent.ordinal != index
                or intent.clock_domain_id != clock_domain
                or intent.orchestrator_session_id
                != endpoint_intent.orchestrator_session_id
                or result.network_intent_hash != intent.intent_hash
                or result.step_started_at_us
                < intent.durability_receipt.anchored_at_us
                or result.step_started_monotonic_us
                < intent.durability_receipt.anchored_monotonic_us
            ):
                raise PilotRunTransitionError("pilot_shard_intent_binding_mismatch")
            if (
                intent.durability_receipt.anchored_at_us
                + self.manifest.remaining_run_elapsed_reservation(
                    intent.ordinal,
                    intent_anchor_us=intent.durability_receipt.anchored_at_us,
                    previous_completed_at_us=previous_epoch,
                )
                >= self.authorization.expires_at_us
            ):
                raise PilotRunAuthorizationError(
                    "u5_window_cannot_cover_remaining_run"
                )
            step_preflight = self.preflight_receipts[index + 1]
            self.authorization.validate_for(
                self.manifest, now_us=result.step_started_at_us
            )
            self.authorization.validate_for(
                self.manifest, now_us=result.step_completed_at_us
            )
            step_preflight.validate_for(
                self.manifest,
                self.authorization,
                now_us=result.step_started_at_us,
            )
            if result.step_started_at_us < step_preflight.checked_at_us:
                raise PilotRunTransitionError("pilot_shard_started_before_preflight")
            if step_preflight.checked_at_us < previous_epoch:
                raise PilotRunTransitionError("pilot_shard_preflight_is_backdated")
            spacing = self.manifest.budgets.min_inter_step_spacing_us
            observed_delay = result.step_started_monotonic_us - previous_monotonic
            if result.observed_inter_step_delay_us != observed_delay:
                raise PilotRunTransitionError("pilot_inter_step_delay_mismatch")
            if (
                result.step_started_at_us < previous_epoch + spacing
                or result.step_started_monotonic_us < previous_monotonic + spacing
            ):
                raise PilotRunTransitionError("pilot_inter_step_spacing_was_not_honoured")
            previous_epoch = result.step_completed_at_us
            previous_monotonic = result.step_completed_monotonic_us
        if len(self.preflight_receipts) == len(self.shard_results) + 2:
            if self.preflight_receipts[-1].checked_at_us < previous_epoch:
                raise PilotRunTransitionError("pilot_next_preflight_is_backdated")
        active_intent = (
            self.network_intents[-1]
            if len(self.network_intents) == completed_intent_count + 1
            else None
        )
        if active_intent is not None:
            active_intent.validate_for(
                self.manifest,
                self.authorization,
                self.preflight_receipts[-1],
            )
            if (
                active_intent.stage != "shard_acquisition"
                or active_intent.ordinal != len(self.shard_results)
                or active_intent.clock_domain_id != clock_domain
                or active_intent.orchestrator_session_id
                != endpoint_intent.orchestrator_session_id
                or active_intent.durability_receipt.anchored_at_us < previous_epoch
                or active_intent.durability_receipt.anchored_monotonic_us
                < previous_monotonic
            ):
                raise PilotRunTransitionError("pilot_active_intent_is_out_of_order")
            remaining_spacing_us = max(
                0,
                previous_epoch
                + self.manifest.budgets.min_inter_step_spacing_us
                - active_intent.durability_receipt.anchored_at_us,
            )
            if (
                active_intent.durability_receipt.anchored_at_us
                + remaining_spacing_us
                + active_intent.reserved_runtime_us
                >= self.authorization.expires_at_us
            ):
                raise PilotRunAuthorizationError(
                    "u5_window_cannot_cover_reserved_shard_step"
                )
            if (
                active_intent.durability_receipt.anchored_at_us
                + self.manifest.remaining_run_elapsed_reservation(
                    active_intent.ordinal,
                    intent_anchor_us=(
                        active_intent.durability_receipt.anchored_at_us
                    ),
                    previous_completed_at_us=previous_epoch,
                )
                >= self.authorization.expires_at_us
            ):
                raise PilotRunAuthorizationError(
                    "u5_window_cannot_cover_remaining_run"
                )
        if self.failure_receipt is not None:
            failure = self.failure_receipt
            if (
                failure.stage != "shard_acquisition"
                or failure.ordinal != len(self.shard_results)
                or failure.ordinal >= len(self.manifest.shards)
                or len(self.preflight_receipts) != len(self.shard_results) + 2
                or active_intent is None
                or failure.network_intent_hash != active_intent.intent_hash
            ):
                raise PilotRunTransitionError("pilot_failure_is_out_of_order")
            failure.validate_for(self.manifest, self.authorization)
            self.authorization.validate_for(
                self.manifest, now_us=failure.step_started_at_us
            )
            self.authorization.validate_for(
                self.manifest, now_us=failure.step_completed_at_us
            )
            failure_preflight = self.preflight_receipts[-1]
            failure_preflight.validate_for(
                self.manifest,
                self.authorization,
                now_us=failure.step_started_at_us,
            )
            if failure_preflight.checked_at_us < previous_epoch:
                raise PilotRunTransitionError("pilot_failure_preflight_is_backdated")
            if failure.step_started_at_us < failure_preflight.checked_at_us:
                raise PilotRunTransitionError("pilot_failure_started_before_preflight")
            spacing = self.manifest.budgets.min_inter_step_spacing_us
            observed_delay = failure.step_started_monotonic_us - previous_monotonic
            if failure.observed_inter_step_delay_us != observed_delay:
                raise PilotRunTransitionError("pilot_failure_inter_step_delay_mismatch")
            if (
                failure.clock_domain_id != clock_domain
                or failure.step_started_at_us < previous_epoch + spacing
                or failure.step_started_monotonic_us < previous_monotonic + spacing
            ):
                raise PilotRunTransitionError("pilot_failure_inter_step_spacing_mismatch")
        self._validate_actual_totals()
        if self.final_anchor is not None:
            if not isinstance(self.final_anchor, PilotRunAnchorReceiptV1):
                raise PilotRunContractError("pilot_final_anchor_is_invalid")
            if len(self.shard_results) != len(self.manifest.shards):
                raise PilotRunTransitionError("pilot_anchor_precedes_all_shards")
            if (
                self.final_anchor.manifest_hash != self.manifest.manifest_hash
                or self.final_anchor.result_candidate_hash != self.result_candidate_hash
                or self.final_anchor.fresh_inventory_scanned_at_us < previous_epoch
                or self.final_anchor.fresh_inventory_scanned_monotonic_us
                < previous_monotonic
                or self.final_anchor.anchored_at_us < previous_epoch
                or self.final_anchor.anchored_monotonic_us < previous_monotonic
                or self.final_anchor.clock_domain_id != clock_domain
                or self.final_anchor.final_run_elapsed_us
                != self.final_anchor.anchored_monotonic_us
                - self.network_intents[0].durability_receipt.anchored_monotonic_us
            ):
                raise PilotRunTransitionError("pilot_final_anchor_binding_mismatch")
            if (
                self.final_anchor.run_control_inventory_entries
                != self.final_run_control_inventory_entries
                or self.final_anchor.run_control_inventory_hash
                != self.final_run_control_inventory_hash
                or self.final_anchor.output_inventory_entries
                != self.actual_totals["inventory_entries"] + 1
                or self.final_anchor.output_inventory_hash
                != self.expected_output_inventory_hash()
                or self.final_anchor.run_control_bytes
                != self.final_run_control_inventory_bytes
                or self.final_anchor.run_control_bytes
                > self.manifest.budgets.max_run_control_bytes
                or self.final_anchor.total_output_bytes
                != self.actual_totals["logical_storage_bytes"]
                + self.final_anchor.run_control_bytes
                or self.final_anchor.total_output_bytes
                > min(
                    self.manifest.budgets.max_total_output_bytes,
                    self.authorization.max_total_output_bytes,
                )
                or self.final_anchor.final_run_elapsed_us
                > min(
                    self.manifest.budgets.max_run_elapsed_us,
                    self.authorization.max_run_elapsed_us,
                )
            ):
                raise PilotRunTransitionError("pilot_final_inventory_budget_mismatch")

    @property
    def next_action(self) -> str:
        if self.stop_reason is not None:
            return "stopped"
        if self.authorization is None:
            return "await_detached_u5_authorization"
        if not self.preflight_receipts:
            return "run_local_preflight:-1"
        if self.endpoint_verification is None:
            if self.network_intents:
                return "await_endpoint_verification_receipt_no_network_retry"
            return "run_endpoint_verification_stage"
        next_ordinal = len(self.shard_results)
        if next_ordinal < len(self.manifest.shards):
            if len(self.preflight_receipts) == next_ordinal + 1:
                return f"run_local_preflight:{next_ordinal}"
            if len(self.network_intents) == next_ordinal + 2:
                return f"await_shard_receipt_no_network_retry:{next_ordinal}"
            return f"collect_shard:{next_ordinal}"
        if self.final_anchor is None:
            return "publish_detached_result_anchor"
        return "complete"

    @property
    def actual_totals(self) -> dict[str, int]:
        verification = self.endpoint_verification
        failure = self.failure_receipt
        if verification is None and failure is None:
            return {
                "network_attempts": 0,
                "verification_attempts": 0,
                "acquisition_attempts": 0,
                "raw_body_bytes": 0,
                "logical_storage_bytes": 0,
                "sum_shard_runtime_us": 0,
                "run_elapsed_us": 0,
                "observed_sleep_us": 0,
                "inventory_entries": self.run_control_inventory_entries,
                "rows": 0,
                "pages": 0,
            }
        if verification is None:
            assert failure is not None
            verification_attempts = failure.actual_network_attempts
            acquisition_attempts = 0
            raw_body_bytes = failure.actual_raw_body_bytes
            logical_storage_bytes = failure.actual_storage_bytes
            sum_shard_runtime_us = 0
            run_elapsed_us = (
                failure.anchored_monotonic_us
                - self.network_intents[0].durability_receipt.anchored_monotonic_us
            )
            observed_sleep_us = failure.observed_internal_sleep_us
            inventory_entries = (
                failure.output_inventory_entries + self.run_control_inventory_entries
            )
        else:
            failed_acquisition_attempts = (
                failure.actual_network_attempts if failure is not None else 0
            )
            verification_attempts = verification.actual_network_attempts
            acquisition_attempts = sum(
                item.actual_attempts for item in self.shard_results
            ) + failed_acquisition_attempts
            raw_body_bytes = verification.actual_raw_body_bytes + sum(
                item.actual_raw_body_bytes for item in self.shard_results
            ) + (failure.actual_raw_body_bytes if failure is not None else 0)
            logical_storage_bytes = verification.actual_storage_bytes + sum(
                item.actual_logical_storage_bytes for item in self.shard_results
            ) + (failure.actual_storage_bytes if failure is not None else 0)
            sum_shard_runtime_us = sum(
                item.step_completed_monotonic_us
                - item.step_started_monotonic_us
                for item in self.shard_results
            ) + (failure.actual_runtime_us if failure is not None else 0)
            last_monotonic = (
                failure.anchored_monotonic_us
                if failure is not None
                else (
                    self.shard_results[-1].step_completed_monotonic_us
                    if self.shard_results
                    else verification.completed_monotonic_us
                )
            )
            run_elapsed_us = (
                last_monotonic
                - self.network_intents[0].durability_receipt.anchored_monotonic_us
            )
            observed_sleep_us = verification.observed_sleep_us + sum(
                item.observed_inter_step_delay_us + item.observed_internal_sleep_us
                for item in self.shard_results
            ) + (
                failure.observed_internal_sleep_us
                + failure.observed_inter_step_delay_us
                if failure is not None
                else 0
            )
            inventory_entries = verification.output_inventory_entries + sum(
                item.output_inventory_entries for item in self.shard_results
            ) + (
                failure.output_inventory_entries if failure is not None else 0
            ) + self.run_control_inventory_entries
        return {
            "network_attempts": verification_attempts + acquisition_attempts,
            "verification_attempts": verification_attempts,
            "acquisition_attempts": acquisition_attempts,
            "raw_body_bytes": raw_body_bytes,
            "logical_storage_bytes": logical_storage_bytes,
            "sum_shard_runtime_us": sum_shard_runtime_us,
            "run_elapsed_us": run_elapsed_us,
            "observed_sleep_us": observed_sleep_us,
            "inventory_entries": inventory_entries,
            "rows": sum(item.actual_rows for item in self.shard_results),
            "pages": sum(item.actual_pages for item in self.shard_results),
        }

    @property
    def active_network_intent(self) -> PilotNetworkIntentV1 | None:
        completed = (1 if self.endpoint_verification is not None else 0) + len(
            self.shard_results
        )
        if self.failure_receipt is None and len(self.network_intents) == completed + 1:
            return self.network_intents[-1]
        return None

    @property
    def charged_totals(self) -> dict[str, int]:
        totals = dict(self.actual_totals)
        intent = self.active_network_intent
        if intent is None:
            return totals
        totals["network_attempts"] += intent.reserved_network_attempts
        totals["raw_body_bytes"] += intent.reserved_raw_body_bytes
        totals["logical_storage_bytes"] += intent.reserved_storage_bytes
        if intent.stage == "endpoint_verification":
            totals["verification_attempts"] += intent.reserved_network_attempts
            totals["run_elapsed_us"] = max(
                totals["run_elapsed_us"], intent.reserved_runtime_us
            )
            totals["observed_sleep_us"] += (
                self.manifest.endpoint_verification.max_total_sleep_us
            )
            totals["inventory_entries"] += 2 * intent.reserved_network_attempts + 5
        else:
            totals["acquisition_attempts"] += intent.reserved_network_attempts
            totals["sum_shard_runtime_us"] += intent.reserved_runtime_us
            assert self.endpoint_verification is not None
            previous_monotonic = (
                self.shard_results[-1].step_completed_monotonic_us
                if self.shard_results
                else self.endpoint_verification.completed_monotonic_us
            )
            reserved_inter_step_delay = max(
                intent.durability_receipt.anchored_monotonic_us
                - previous_monotonic,
                self.manifest.budgets.min_inter_step_spacing_us,
            )
            reserved_start_monotonic = previous_monotonic + reserved_inter_step_delay
            totals["run_elapsed_us"] = max(
                totals["run_elapsed_us"],
                reserved_start_monotonic
                - self.network_intents[0].durability_receipt.anchored_monotonic_us
                + intent.reserved_runtime_us,
            )
            totals["observed_sleep_us"] += (
                reserved_inter_step_delay
                + _reserved_retry_sleep_us(
                    self.manifest.shards[intent.ordinal].request
                )
            )
            totals["inventory_entries"] += 2 * intent.reserved_network_attempts + 6
        return totals

    @property
    def run_control_inventory(
        self,
    ) -> tuple[tuple[str, str, str, str, int], ...]:
        entries: list[tuple[str, str, str, str, int]] = [
            _control_inventory_entry(
                kind="manifest",
                locator="run-control/manifest.json",
                semantic_hash=self.manifest.manifest_hash,
                payload=self.manifest.as_dict(),
            )
        ]
        if self.authorization is not None:
            entries.append(
                _control_inventory_entry(
                    kind="authorization",
                    locator="run-control/authorization.json",
                    semantic_hash=self.authorization.receipt_hash,
                    payload=self.authorization.as_dict(),
                )
            )
        for item in self.preflight_receipts:
            step = (
                "endpoint"
                if item.step_ordinal == -1
                else f"shard-{item.step_ordinal:04d}"
            )
            entries.append(
                _control_inventory_entry(
                    kind=f"preflight:{item.step_ordinal}",
                    locator=f"run-control/preflights/{step}.json",
                    semantic_hash=item.receipt_hash,
                    payload=item.as_dict(),
                )
            )
        for index, item in enumerate(self.network_intents):
            entries.append(
                _control_inventory_entry(
                    kind=f"network_intent_candidate:{index}",
                    locator=item.intent_candidate_locator,
                    semantic_hash=item.intent_candidate_hash,
                    payload=item.intent_candidate_payload,
                )
            )
            entries.append(
                _control_inventory_entry(
                    kind=f"network_intent_sealed:{index}",
                    locator=item.sealed_intent_locator,
                    semantic_hash=item.intent_hash,
                    payload=item.as_dict(),
                )
            )
        if self.endpoint_verification is not None:
            entries.append(
                _control_inventory_entry(
                    kind="endpoint_verification",
                    locator="run-control/endpoint-verification.json",
                    semantic_hash=self.endpoint_verification.receipt_hash,
                    payload=self.endpoint_verification.as_dict(),
                )
            )
        entries.extend(
            _control_inventory_entry(
                kind=f"shard_result:{item.ordinal}",
                locator=f"run-control/shard-results/{item.ordinal:04d}.json",
                semantic_hash=item.receipt_hash,
                payload=item.as_dict(),
            )
            for item in self.shard_results
        )
        if self.failure_receipt is not None:
            entries.append(
                _control_inventory_entry(
                    kind="terminal_failure_candidate",
                    locator="run-control/terminal-failure-candidate.json",
                    semantic_hash=self.failure_receipt.failure_candidate_hash,
                    payload=self.failure_receipt.failure_candidate_payload,
                )
            )
            entries.append(
                _control_inventory_entry(
                    kind="terminal_failure_sealed",
                    locator="run-control/terminal-failure.json",
                    semantic_hash=self.failure_receipt.receipt_hash,
                    payload=self.failure_receipt.as_dict(),
                )
            )
        return tuple(entries)

    @property
    def run_control_inventory_entries(self) -> int:
        return len(self.run_control_inventory)

    @property
    def run_control_inventory_bytes(self) -> int:
        return sum(
            byte_count
            for _kind, _locator, _semantic, _artifact, byte_count
            in self.run_control_inventory
        )

    @property
    def run_control_inventory_hash(self) -> str:
        return _sha256_payload(
            {
                "domain": "mexc_public_qa_pilot_run_control_inventory_v1",
                "entries": [
                    {
                        "kind": kind,
                        "locator": locator,
                        "semantic_hash": semantic_hash,
                        "artifact_sha256": artifact_sha256,
                        "bytes": byte_count,
                    }
                    for (
                        kind,
                        locator,
                        semantic_hash,
                        artifact_sha256,
                        byte_count,
                    ) in self.run_control_inventory
                ],
            }
        )

    @property
    def final_run_control_inventory(
        self,
    ) -> tuple[tuple[str, str, str, str, int], ...]:
        if self.failure_receipt is not None or self.endpoint_verification is None or len(
            self.shard_results
        ) != len(self.manifest.shards):
            raise PilotRunTransitionError("pilot_final_control_inventory_is_incomplete")
        return (
            *self.run_control_inventory,
            _control_inventory_entry(
                kind="result_candidate",
                locator="run-control/result-candidate.json",
                semantic_hash=self.result_candidate_hash,
                payload=self.result_candidate_payload,
            ),
        )

    @property
    def final_run_control_inventory_entries(self) -> int:
        return len(self.final_run_control_inventory)

    @property
    def final_run_control_inventory_bytes(self) -> int:
        return sum(
            byte_count
            for _kind, _locator, _semantic, _artifact, byte_count
            in self.final_run_control_inventory
        )

    @property
    def final_run_control_inventory_hash(self) -> str:
        return _sha256_payload(
            {
                "domain": "mexc_public_qa_pilot_run_control_inventory_v1",
                "entries": [
                    {
                        "kind": kind,
                        "locator": locator,
                        "semantic_hash": semantic_hash,
                        "artifact_sha256": artifact_sha256,
                        "bytes": byte_count,
                    }
                    for (
                        kind,
                        locator,
                        semantic_hash,
                        artifact_sha256,
                        byte_count,
                    ) in self.final_run_control_inventory
                ],
            }
        )

    def _validate_actual_totals(self) -> None:
        self._validate_run_control_budget()
        totals = self.charged_totals
        budgets = self.manifest.budgets
        comparisons = (
            ("network_attempts", budgets.max_network_attempts),
            ("verification_attempts", budgets.max_verification_attempts),
            ("acquisition_attempts", budgets.max_acquisition_attempts),
            ("raw_body_bytes", budgets.max_total_raw_body_bytes),
            ("logical_storage_bytes", budgets.max_total_logical_storage_bytes),
            ("sum_shard_runtime_us", budgets.max_sum_shard_runtime_us),
            ("run_elapsed_us", budgets.max_run_elapsed_us),
            ("observed_sleep_us", budgets.max_observed_sleep_us),
            ("inventory_entries", budgets.max_inventory_entries),
            ("rows", budgets.max_total_rows),
            ("pages", budgets.max_total_pages),
        )
        for resource, limit in comparisons:
            if totals[resource] > limit:
                raise PilotRunBudgetExceededError(resource, limit, totals[resource])
        assert self.authorization is not None
        authority_caps = (
            ("network_attempts", self.authorization.max_network_attempts),
            ("raw_body_bytes", self.authorization.max_total_raw_body_bytes),
            ("run_elapsed_us", self.authorization.max_run_elapsed_us),
        )
        for resource, limit in authority_caps:
            if totals[resource] > limit:
                raise PilotRunAuthorizationError(
                    f"u5_actual_{resource}_exceeded_authority"
                )

    def _validate_run_control_budget(self) -> None:
        observed = self.run_control_inventory_bytes
        if (
            self.authorization is not None
            and self.failure_receipt is None
            and isinstance(
                self.endpoint_verification,
                EndpointVerificationReceiptV1,
            )
            and len(self.shard_results) == len(self.manifest.shards)
        ):
            observed = self.final_run_control_inventory_bytes
        limit = self.manifest.budgets.max_run_control_bytes
        if observed > limit:
            raise PilotRunBudgetExceededError(
                "run_control_bytes",
                limit,
                observed,
            )

    @property
    def result_candidate_payload(self) -> dict[str, object]:
        if self.endpoint_verification is None or len(self.shard_results) != len(
            self.manifest.shards
        ):
            raise PilotRunTransitionError("pilot_result_candidate_is_incomplete")
        assert self.authorization is not None
        return {
            "domain": "mexc_public_qa_pilot_result_candidate_v1",
            "manifest_hash": self.manifest.manifest_hash,
            "authorization_receipt_hash": self.authorization.receipt_hash,
            "preflight_receipt_hashes": [
                item.receipt_hash for item in self.preflight_receipts
            ],
            "endpoint_verification_receipt_hash": self.endpoint_verification.receipt_hash,
            "shard_result_hashes": [item.receipt_hash for item in self.shard_results],
            "network_intent_hashes": [
                item.intent_hash for item in self.network_intents
            ],
            "actual_totals": self.actual_totals,
            "charged_totals": self.charged_totals,
        }

    @property
    def result_candidate_hash(self) -> str:
        return _sha256_payload(self.result_candidate_payload)

    def expected_output_inventory_hash(self) -> str:
        if self.endpoint_verification is None or len(self.shard_results) != len(
            self.manifest.shards
        ):
            raise PilotRunTransitionError("pilot_output_inventory_is_incomplete")
        run_control_bytes = self.final_run_control_inventory_bytes
        return _sha256_payload(
            {
                "domain": "mexc_public_qa_pilot_output_inventory_v1",
                "manifest_hash": self.manifest.manifest_hash,
                "endpoint_inventory": {
                    "hash": self.endpoint_verification.output_inventory_hash,
                    "entries": self.endpoint_verification.output_inventory_entries,
                    "bytes": self.endpoint_verification.actual_storage_bytes,
                },
                "shard_inventories": [
                    {
                        "ordinal": item.ordinal,
                        "hash": item.output_inventory_hash,
                        "entries": item.output_inventory_entries,
                        "bytes": item.output_inventory_bytes,
                    }
                    for item in self.shard_results
                ],
                "run_control_inventory": {
                    "hash": self.final_run_control_inventory_hash,
                    "entries": self.final_run_control_inventory_entries,
                    "bytes": run_control_bytes,
                },
                "total_entries": self.actual_totals["inventory_entries"] + 1,
                "total_bytes": (
                    self.actual_totals["logical_storage_bytes"] + run_control_bytes
                ),
            }
        )

    @property
    def state_hash(self) -> str:
        return _frozen_contract_hash(self)

    def with_authorization(
        self, receipt: U5PublicPilotAuthorizationReceiptV1, *, now_us: int
    ) -> "PilotRunStateV1":
        if self.stop_reason is not None:
            raise PilotRunTransitionError("pilot_run_is_stopped")
        if self.authorization is not None:
            if self.authorization == receipt:
                return self
            raise PilotRunTransitionError("pilot_authorization_conflict")
        receipt.validate_for(
            self.manifest,
            now_us=now_us,
            require_full_run_window=True,
        )
        return replace(self, authorization=receipt)

    def with_preflight(
        self, receipt: PilotDiskPreflightReceiptV1, *, now_us: int
    ) -> "PilotRunStateV1":
        if self.authorization is None:
            raise PilotRunTransitionError("pilot_preflight_precedes_authorization")
        expected = -1 if not self.preflight_receipts else len(self.preflight_receipts) - 1
        action = f"run_local_preflight:{expected}"
        if self.next_action != action:
            if receipt in self.preflight_receipts:
                return self
            raise PilotRunTransitionError("pilot_preflight_is_out_of_order")
        if receipt.step_ordinal != expected:
            raise PilotRunTransitionError("pilot_preflight_step_mismatch")
        self.authorization.validate_for(self.manifest, now_us=now_us)
        receipt.validate_for(
            self.manifest, self.authorization, now_us=now_us
        )
        if expected >= 0:
            previous_completed_at = (
                self.shard_results[-1].step_completed_at_us
                if self.shard_results
                else self.endpoint_verification.completed_at_us
            )
            if receipt.checked_at_us < previous_completed_at:
                raise PilotRunTransitionError("pilot_preflight_is_backdated")
        return replace(self, preflight_receipts=(*self.preflight_receipts, receipt))

    def with_network_intent(
        self, receipt: PilotNetworkIntentV1
    ) -> "PilotRunStateV1":
        if self.stop_reason is not None:
            raise PilotRunTransitionError("pilot_run_is_stopped")
        if not isinstance(receipt, PilotNetworkIntentV1):
            raise PilotRunContractError("pilot_network_intent_is_invalid")
        expected_action = (
            "run_endpoint_verification_stage"
            if receipt.stage == "endpoint_verification"
            else f"collect_shard:{receipt.ordinal}"
        )
        if self.next_action != expected_action:
            if self.network_intents and self.network_intents[-1] == receipt:
                return self
            raise PilotRunTransitionError("pilot_network_intent_is_out_of_order")
        assert self.authorization is not None
        preflight = self.preflight_receipts[-1]
        receipt.validate_for(self.manifest, self.authorization, preflight)
        if receipt.stage == "shard_acquisition":
            assert self.endpoint_verification is not None
            previous_epoch = (
                self.shard_results[-1].step_completed_at_us
                if self.shard_results
                else self.endpoint_verification.completed_at_us
            )
            previous_monotonic = (
                self.shard_results[-1].step_completed_monotonic_us
                if self.shard_results
                else self.endpoint_verification.completed_monotonic_us
            )
            if (
                receipt.clock_domain_id
                != self.endpoint_verification.clock_domain_id
                or receipt.durability_receipt.anchored_at_us < previous_epoch
                or receipt.durability_receipt.anchored_monotonic_us
                < previous_monotonic
            ):
                raise PilotRunTransitionError("pilot_network_intent_is_backdated")
            remaining_spacing_us = max(
                0,
                previous_epoch
                + self.manifest.budgets.min_inter_step_spacing_us
                - receipt.durability_receipt.anchored_at_us,
            )
            if (
                receipt.durability_receipt.anchored_at_us
                + remaining_spacing_us
                + receipt.reserved_runtime_us
                >= self.authorization.expires_at_us
            ):
                raise PilotRunAuthorizationError(
                    "u5_window_cannot_cover_reserved_shard_step"
                )
            if (
                receipt.durability_receipt.anchored_at_us
                + self.manifest.remaining_run_elapsed_reservation(
                    receipt.ordinal,
                    intent_anchor_us=receipt.durability_receipt.anchored_at_us,
                    previous_completed_at_us=previous_epoch,
                )
                >= self.authorization.expires_at_us
            ):
                raise PilotRunAuthorizationError(
                    "u5_window_cannot_cover_remaining_run"
                )
        return replace(self, network_intents=(*self.network_intents, receipt))

    def with_endpoint_verification(
        self, receipt: EndpointVerificationReceiptV1
    ) -> "PilotRunStateV1":
        if self.stop_reason is not None:
            raise PilotRunTransitionError("pilot_run_is_stopped")
        if self.next_action != "await_endpoint_verification_receipt_no_network_retry":
            if self.endpoint_verification == receipt:
                return self
            raise PilotRunTransitionError("pilot_endpoint_verification_is_out_of_order")
        assert self.authorization is not None
        preflight = self.preflight_receipts[-1]
        intent = self.network_intents[-1]
        if receipt.network_intent_hash != intent.intent_hash:
            raise PilotRunTransitionError("pilot_endpoint_intent_binding_mismatch")
        self.authorization.validate_for(self.manifest, now_us=receipt.started_at_us)
        self.authorization.validate_for(self.manifest, now_us=receipt.completed_at_us)
        preflight.validate_for(
            self.manifest,
            self.authorization,
            now_us=receipt.started_at_us,
        )
        receipt.validate_for(self.manifest, self.authorization)
        if receipt.started_at_us < preflight.checked_at_us:
            raise PilotRunTransitionError("pilot_endpoint_started_before_preflight")
        return replace(self, endpoint_verification=receipt)

    def with_shard_result(self, receipt: PilotShardResultV1) -> "PilotRunStateV1":
        expected = len(self.shard_results)
        if self.next_action != f"await_shard_receipt_no_network_retry:{expected}":
            if expected > 0 and self.shard_results[-1] == receipt:
                return self
            raise PilotRunTransitionError("pilot_shard_result_is_out_of_order")
        assert self.authorization is not None
        preflight = self.preflight_receipts[-1]
        intent = self.network_intents[-1]
        if receipt.network_intent_hash != intent.intent_hash:
            raise PilotRunTransitionError("pilot_shard_intent_binding_mismatch")
        self.authorization.validate_for(
            self.manifest, now_us=receipt.step_started_at_us
        )
        self.authorization.validate_for(
            self.manifest, now_us=receipt.step_completed_at_us
        )
        preflight.validate_for(
            self.manifest,
            self.authorization,
            now_us=receipt.step_started_at_us,
        )
        receipt.validate_for(self.manifest, self.manifest.shards[expected])
        if receipt.step_started_at_us < preflight.checked_at_us:
            raise PilotRunTransitionError("pilot_shard_started_before_preflight")
        return replace(self, shard_results=(*self.shard_results, receipt))

    def with_step_failure(
        self, receipt: PilotStepFailureReceiptV1
    ) -> "PilotRunStateV1":
        if self.failure_receipt is not None:
            if self.failure_receipt == receipt:
                return self
            raise PilotRunTransitionError("pilot_failure_receipt_conflict")
        if self.stop_reason is not None or self.next_action == "complete":
            raise PilotRunTransitionError("pilot_failure_is_out_of_order")
        if not isinstance(receipt, PilotStepFailureReceiptV1):
            raise PilotRunContractError("pilot_failure_receipt_is_invalid")
        expected_action = (
            "await_endpoint_verification_receipt_no_network_retry"
            if receipt.stage == "endpoint_verification"
            else f"await_shard_receipt_no_network_retry:{receipt.ordinal}"
        )
        if self.next_action != expected_action:
            raise PilotRunTransitionError("pilot_failure_is_out_of_order")
        assert self.authorization is not None
        preflight = self.preflight_receipts[-1]
        intent = self.network_intents[-1]
        if receipt.network_intent_hash != intent.intent_hash:
            raise PilotRunTransitionError("pilot_failure_intent_binding_mismatch")
        receipt.validate_for(self.manifest, self.authorization)
        self.authorization.validate_for(
            self.manifest, now_us=receipt.step_started_at_us
        )
        self.authorization.validate_for(
            self.manifest, now_us=receipt.step_completed_at_us
        )
        preflight.validate_for(
            self.manifest,
            self.authorization,
            now_us=receipt.step_started_at_us,
        )
        if receipt.step_started_at_us < preflight.checked_at_us:
            raise PilotRunTransitionError("pilot_failure_started_before_preflight")
        return replace(
            self,
            failure_receipt=receipt,
            stop_reason=receipt.error_code,
            stop_evidence_hash=receipt.receipt_hash,
        )

    def with_final_anchor(
        self, receipt: PilotRunAnchorReceiptV1
    ) -> "PilotRunStateV1":
        if self.next_action != "publish_detached_result_anchor":
            if self.final_anchor == receipt:
                return self
            raise PilotRunTransitionError("pilot_anchor_is_out_of_order")
        if (
            receipt.manifest_hash != self.manifest.manifest_hash
            or receipt.result_candidate_hash != self.result_candidate_hash
            or receipt.fresh_inventory_scanned_at_us
            < self.shard_results[-1].step_completed_at_us
            or receipt.fresh_inventory_scanned_monotonic_us
            < self.shard_results[-1].step_completed_monotonic_us
            or receipt.run_control_inventory_entries
            != self.final_run_control_inventory_entries
            or receipt.run_control_inventory_hash
            != self.final_run_control_inventory_hash
            or receipt.output_inventory_hash
            != self.expected_output_inventory_hash()
            or receipt.clock_domain_id
            != self.endpoint_verification.clock_domain_id
            or receipt.anchored_at_us
            < self.shard_results[-1].step_completed_at_us
            or receipt.anchored_monotonic_us
            < self.shard_results[-1].step_completed_monotonic_us
            or receipt.final_run_elapsed_us
            != receipt.anchored_monotonic_us
            - self.network_intents[0].durability_receipt.anchored_monotonic_us
            or receipt.output_inventory_entries
            != self.actual_totals["inventory_entries"] + 1
            or receipt.run_control_bytes
            != self.final_run_control_inventory_bytes
            or receipt.run_control_bytes > self.manifest.budgets.max_run_control_bytes
            or receipt.total_output_bytes
            != self.actual_totals["logical_storage_bytes"]
            + receipt.run_control_bytes
            or receipt.total_output_bytes
            > min(
                self.manifest.budgets.max_total_output_bytes,
                self.authorization.max_total_output_bytes,
            )
            or receipt.final_run_elapsed_us
            > min(
                self.manifest.budgets.max_run_elapsed_us,
                self.authorization.max_run_elapsed_us,
            )
        ):
            raise PilotRunTransitionError("pilot_anchor_binding_mismatch")
        return replace(self, final_anchor=receipt)

    def stopped(
        self, *, reason: str, evidence_hash: str
    ) -> "PilotRunStateV1":
        _safe_code(reason, field="pilot_stop_reason")
        _digest(evidence_hash, field="pilot_stop_evidence_hash")
        if self.next_action == "complete":
            raise PilotRunTransitionError("completed_pilot_cannot_be_stopped")
        if self.stop_reason is not None:
            if self.stop_reason == reason and self.stop_evidence_hash == evidence_hash:
                return self
            raise PilotRunTransitionError("pilot_stop_receipt_conflict")
        return replace(self, stop_reason=reason, stop_evidence_hash=evidence_hash)

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "manifest": self.manifest.as_dict(),
            "manifest_hash": self.manifest.manifest_hash,
            "authorization": self.authorization.as_dict() if self.authorization else None,
            "preflight_receipts": [item.as_dict() for item in self.preflight_receipts],
            "network_intents": [item.as_dict() for item in self.network_intents],
            "endpoint_verification": (
                self.endpoint_verification.as_dict()
                if self.endpoint_verification
                else None
            ),
            "shard_results": [item.as_dict() for item in self.shard_results],
            "failure_receipt": (
                self.failure_receipt.as_dict() if self.failure_receipt else None
            ),
            "actual_totals": self.actual_totals,
            "charged_totals": self.charged_totals,
            "run_control_inventory": [
                {
                    "kind": kind,
                    "locator": locator,
                    "semantic_hash": semantic_hash,
                    "artifact_sha256": artifact_sha256,
                    "bytes": byte_count,
                }
                for (
                    kind,
                    locator,
                    semantic_hash,
                    artifact_sha256,
                    byte_count,
                ) in self.run_control_inventory
            ],
            "run_control_inventory_hash": self.run_control_inventory_hash,
            "result_candidate_hash": (
                self.result_candidate_hash
                if self.endpoint_verification is not None
                and len(self.shard_results) == len(self.manifest.shards)
                else None
            ),
            "final_anchor": self.final_anchor.as_dict() if self.final_anchor else None,
            "stop_reason": self.stop_reason,
            "stop_evidence_hash": self.stop_evidence_hash,
            "next_action": self.next_action,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PilotRunStateV1":
        expected = frozenset(
            {
                "contract_version",
                "manifest",
                "manifest_hash",
                "authorization",
                "preflight_receipts",
                "network_intents",
                "endpoint_verification",
                "shard_results",
                "failure_receipt",
                "actual_totals",
                "charged_totals",
                "run_control_inventory",
                "run_control_inventory_hash",
                "result_candidate_hash",
                "final_anchor",
                "stop_reason",
                "stop_evidence_hash",
                "next_action",
            }
        )
        values = _exact_mapping(payload, expected, code="pilot_state_schema_mismatch")
        manifest = MexcPublicQaPilotRunManifestV1.from_dict(values.pop("manifest"))
        if values.pop("manifest_hash") != manifest.manifest_hash:
            raise PilotRunContractError("pilot_state_manifest_hash_mismatch")
        authorization_payload = values.pop("authorization")
        authorization = (
            None
            if authorization_payload is None
            else U5PublicPilotAuthorizationReceiptV1.from_dict(authorization_payload)
        )
        raw_preflights = values.pop("preflight_receipts")
        raw_intents = values.pop("network_intents")
        raw_results = values.pop("shard_results")
        if (
            not isinstance(raw_preflights, list)
            or not isinstance(raw_intents, list)
            or not isinstance(raw_results, list)
        ):
            raise PilotRunContractError("pilot_state_array_schema_mismatch")
        endpoint_payload = values.pop("endpoint_verification")
        failure_payload = values.pop("failure_receipt")
        anchor_payload = values.pop("final_anchor")
        result = cls(
            manifest=manifest,
            authorization=authorization,
            preflight_receipts=tuple(
                PilotDiskPreflightReceiptV1.from_dict(item)
                for item in raw_preflights
            ),
            network_intents=tuple(
                PilotNetworkIntentV1.from_dict(item) for item in raw_intents
            ),
            endpoint_verification=(
                None
                if endpoint_payload is None
                else EndpointVerificationReceiptV1.from_dict(endpoint_payload)
            ),
            shard_results=tuple(PilotShardResultV1.from_dict(item) for item in raw_results),
            failure_receipt=(
                None
                if failure_payload is None
                else PilotStepFailureReceiptV1.from_dict(failure_payload)
            ),
            final_anchor=(
                None
                if anchor_payload is None
                else PilotRunAnchorReceiptV1.from_dict(anchor_payload)
            ),
            stop_reason=values.pop("stop_reason"),
            stop_evidence_hash=values.pop("stop_evidence_hash"),
            contract_version=values.pop("contract_version"),
        )
        repeated = {
            "actual_totals": result.actual_totals,
            "charged_totals": result.charged_totals,
            "run_control_inventory": [
                {
                    "kind": kind,
                    "locator": locator,
                    "semantic_hash": semantic_hash,
                    "artifact_sha256": artifact_sha256,
                    "bytes": byte_count,
                }
                for (
                    kind,
                    locator,
                    semantic_hash,
                    artifact_sha256,
                    byte_count,
                ) in result.run_control_inventory
            ],
            "run_control_inventory_hash": result.run_control_inventory_hash,
            "result_candidate_hash": (
                result.result_candidate_hash
                if result.endpoint_verification is not None
                and len(result.shard_results) == len(result.manifest.shards)
                else None
            ),
            "next_action": result.next_action,
        }
        for field, expected_value in repeated.items():
            if _canonical_bytes(values.pop(field)) != _canonical_bytes(expected_value):
                raise PilotRunContractError(f"pilot_state_{field}_mismatch")
        if values:
            raise PilotRunContractError("pilot_state_parser_left_unknown_fields")
        return result


def parse_pilot_run_manifest_v1(
    payload: bytes | Mapping[str, Any],
) -> MexcPublicQaPilotRunManifestV1:
    if isinstance(payload, bytes):
        values = _parse_canonical_json(payload)
    elif isinstance(payload, Mapping):
        values = dict(payload)
    else:
        raise PilotRunContractError("pilot_manifest_payload_type_is_invalid")
    result = MexcPublicQaPilotRunManifestV1.from_dict(values)
    if _canonical_bytes(result.as_dict()) != _canonical_bytes(values):
        raise PilotRunContractError("pilot_manifest_round_trip_mismatch")
    return result


def parse_pilot_run_state_v1(
    payload: bytes | Mapping[str, Any],
) -> PilotRunStateV1:
    if isinstance(payload, bytes):
        values = _parse_canonical_json(payload)
    elif isinstance(payload, Mapping):
        values = dict(payload)
    else:
        raise PilotRunContractError("pilot_state_payload_type_is_invalid")
    result = PilotRunStateV1.from_dict(values)
    if _canonical_bytes(result.as_dict()) != _canonical_bytes(values):
        raise PilotRunContractError("pilot_state_round_trip_mismatch")
    return result


def _publish_immutable_json(path: Path, payload: object) -> Path:
    target = Path(path)
    if not target.is_absolute() or target.name in {"", ".", ".."}:
        raise PilotRunArtifactError("pilot_artifact_path_must_be_absolute")
    if not target.parent.is_dir():
        raise PilotRunArtifactError("pilot_artifact_parent_does_not_exist")
    _validate_existing_directory_chain(target.parent, code="pilot_artifact")
    body = _canonical_bytes(payload) + b"\n"
    if len(body) > _MAX_MANIFEST_BYTES:
        raise PilotRunArtifactError("pilot_artifact_is_oversized")
    if target.exists():
        existing = _read_artifact_limited(
            target,
            max_bytes=_MAX_MANIFEST_BYTES,
            code="pilot_existing_artifact",
        )
        if existing != body:
            raise PilotRunArtifactError("pilot_artifact_conflict")
        return target
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    cleanup_error: OSError | None = None
    try:
        with temporary.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _read_artifact_limited(
                target,
                max_bytes=_MAX_MANIFEST_BYTES,
                code="pilot_concurrent_artifact",
            )
            if existing != body:
                raise PilotRunArtifactError("pilot_artifact_conflict")
        except OSError as exc:
            raise PilotRunArtifactError("pilot_artifact_atomic_link_failed") from exc
    except PilotRunError:
        raise
    except OSError as exc:
        raise PilotRunArtifactError("pilot_artifact_write_failed") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            cleanup_error = exc
    if cleanup_error is not None:
        raise PilotRunArtifactError("pilot_artifact_temp_cleanup_failed") from cleanup_error
    _validate_existing_directory_chain(target.parent, code="pilot_artifact")
    if _read_artifact_limited(
        target,
        max_bytes=_MAX_MANIFEST_BYTES,
        code="pilot_published_artifact",
    ) != body:
        raise PilotRunArtifactError("pilot_published_artifact_verification_failed")
    return target


def publish_pilot_run_manifest_v1(
    path: str | os.PathLike[str], manifest: MexcPublicQaPilotRunManifestV1
) -> Path:
    if not isinstance(manifest, MexcPublicQaPilotRunManifestV1):
        raise PilotRunContractError("pilot_manifest_is_invalid")
    return _publish_immutable_json(Path(path), manifest.as_dict())


def load_pilot_run_manifest_v1(
    path: str | os.PathLike[str],
) -> MexcPublicQaPilotRunManifestV1:
    target = Path(path)
    raw = _read_artifact_limited(
        target,
        max_bytes=_MAX_MANIFEST_BYTES,
        code="pilot_manifest_file",
    )
    return parse_pilot_run_manifest_v1(raw)


_CONTRACT_SCHEMA = {
    "contract_version": PILOT_RUN_CONTRACT_VERSION,
    "component_versions": {
        "global_budget": PILOT_GLOBAL_BUDGET_VERSION,
        "shard_plan": PILOT_SHARD_PLAN_VERSION,
        "endpoint_verification_plan": ENDPOINT_VERIFICATION_PLAN_VERSION,
        "u5_authorization": U5_AUTHORIZATION_RECEIPT_VERSION,
        "disk_preflight": PILOT_PREFLIGHT_RECEIPT_VERSION,
        "intent_durability": PILOT_INTENT_DURABILITY_RECEIPT_VERSION,
        "network_intent": PILOT_NETWORK_INTENT_VERSION,
        "endpoint_verification_receipt": ENDPOINT_VERIFICATION_RECEIPT_VERSION,
        "shard_result": PILOT_SHARD_RESULT_VERSION,
        "step_failure": PILOT_STEP_FAILURE_RECEIPT_VERSION,
        "run_anchor": PILOT_RUN_ANCHOR_VERSION,
        "run_state": PILOT_RUN_STATE_VERSION,
    },
    "field_sets": {
        "global_budget": list(PilotGlobalBudgetsV1.__dataclass_fields__),
        "shard_plan": list(PilotShardPlanV1.__dataclass_fields__),
        "endpoint_verification_plan": list(
            EndpointVerificationPlanV1.__dataclass_fields__
        ),
        "manifest": list(MexcPublicQaPilotRunManifestV1.__dataclass_fields__),
        "u5_authorization": list(
            U5PublicPilotAuthorizationReceiptV1.__dataclass_fields__
        ),
        "disk_preflight": list(PilotDiskPreflightReceiptV1.__dataclass_fields__),
        "intent_durability": list(
            PilotIntentDurabilityReceiptV1.__dataclass_fields__
        ),
        "network_intent": list(PilotNetworkIntentV1.__dataclass_fields__),
        "endpoint_verification_receipt": list(
            EndpointVerificationReceiptV1.__dataclass_fields__
        ),
        "shard_result": list(PilotShardResultV1.__dataclass_fields__),
        "step_failure": list(PilotStepFailureReceiptV1.__dataclass_fields__),
        "run_anchor": list(PilotRunAnchorReceiptV1.__dataclass_fields__),
        "run_state": list(PilotRunStateV1.__dataclass_fields__),
    },
    "dependency_hashes": {
        "candidate_endpoint": load_mexc_futures_endpoint_contract_v1(
            candidate_endpoint_fixture_path()
        ).contract_hash,
        "candidate_resource_limits": candidate_history_resource_limits_v1().contract_hash,
        "candidate_retry_policy": candidate_history_retry_policy_v1().contract_hash,
        "transport": mexc_futures_transport_contract_hash(),
        "strict_history_v2": strict_history_v2_contract_hash(),
        "min1_aggregation_v1": min1_aggregation_contract_hash(),
    },
    "purpose": "p2_public_qa_data_mechanics_only",
    "authority": {
        "manifest_grants_u5": False,
        "detached_explicit_u5_receipt_required": True,
        "receipt_binds_exact_manifest": True,
        "authorization_expiry_checked_before_each_network_step": True,
        "initial_authorization_window_covers_full_planned_run": True,
        "every_network_intent_window_covers_remaining_planned_run": True,
        "latest_preflight_valid_start_covers_remaining_planned_run": True,
        "forbidden_scopes": list(_FORBIDDEN_SCOPES),
        "allowed_domains": ["www.mexc.com", "api.mexc.com"],
        "authorized_operations": list(_AUTHORIZED_OPERATIONS),
    },
    "endpoint_verification": {
        "ordered_actions": list(_VERIFICATION_ACTIONS),
        "official_document_and_live_probe_required": True,
        "one_closed_min1_live_bar": True,
        "fresh_reload_and_detached_anchor_required": True,
        "output_inventory_entry_cap": "2_times_max_network_attempts_plus_5",
        "candidate_fixture_remains_unverified": True,
        "failure_action": "stop_before_any_acquisition",
    },
    "pilot_composition": {
        "shard_roles": sorted(_SHARD_ROLES),
        "qa_symbols": "BTCUSDT_plus_8_to_10_distinct_symbols",
        "qa_min1_days": [7, 14],
        "deep_min1_days": 140,
        "native_min60_control_per_qa_symbol": True,
        "common_collection_as_of_us": True,
        "concrete_symbols_dates_and_caps_are_instance_fields": True,
        "full_universe_or_p3_contract": False,
    },
    "budgets": {
        "strict_integer_no_bool": True,
        "worst_case_reserved_before_step": True,
        "actuals_from_freshly_reloaded_admitted_shard": True,
        "started_failures_and_retries_count": True,
        "max_active_shards": 1,
        "max_in_flight_http_attempts": 1,
        "inter_step_spacing_required": True,
        "hard_caps": {
            "shards": _MAX_PILOT_SHARDS,
            "symbols": _MAX_PILOT_SYMBOLS,
            "pages": _MAX_TOTAL_PAGES,
            "rows": _MAX_TOTAL_ROWS,
            "attempts": _MAX_TOTAL_ATTEMPTS,
            "raw_bytes": _MAX_TOTAL_RAW_BYTES,
            "storage_bytes": _MAX_TOTAL_STORAGE_BYTES,
            "run_control_bytes": _MAX_RUN_CONTROL_BYTES,
            "output_bytes": _MAX_TOTAL_OUTPUT_BYTES,
            "runtime_us": _MAX_TOTAL_RUNTIME_US,
            "sleep_us": _MAX_TOTAL_SLEEP_US,
            "spacing_us": _MAX_SPACING_US,
            "inventory_entries": _MAX_INVENTORY_ENTRIES,
            "preflight_age_us": _MAX_PREFLIGHT_AGE_US,
            "manifest_bytes": _MAX_MANIFEST_BYTES,
            "non_manifest_run_control_entry_bytes": _MAX_RUN_CONTROL_ENTRY_BYTES,
        },
        "planned_run_control_reservation": {
            "success_entry_count": "4_times_shards_plus_7",
            "bytes": "max_manifest_bytes_plus_every_other_entry_times_max_entry_bytes",
        },
    },
    "storage": {
        "output_root_grammar": "file:///UPPERCASE_DRIVE/canonical_segments_no_empty_dot_dotdot_trailing_dot_space_or_windows_device_alias_v1",
        "absolute_output_locator_and_relative_request_roots": True,
        "one_fresh_root_per_request": True,
        "legacy_data_history_forbidden": True,
        "preflight_before_every_network_step": True,
        "free_space_observation_is_not_reservation": True,
        "windows_sudden_power_loss_not_proven": True,
        "windows_reserved_basenames": sorted(_WINDOWS_RESERVED_BASENAMES),
        "detached_result_anchor_required": True,
    },
    "orchestration": {
        "pure_state_projection_no_executor": True,
        "deterministic_order": [
            "authorization",
            "probe_preflight",
            "probe_network_intent",
            "endpoint_verification",
            "per_shard_preflight_intent_and_result",
            "detached_result_anchor",
        ],
        "incomplete_shard_resume_or_repair": False,
        "unresolved_network_intent_allows_retry": False,
        "network_intent_slot": {
            "identity_fields": ["manifest_hash", "stage", "ordinal"],
            "candidate_independent": True,
            "candidate_publication_outcome": "create_new_winner_for_this_process",
            "preexisting_identical_or_conflicting_slot": "stop_without_network",
            "os_create_new_arbitration_is_future_executor_responsibility": True,
            "reload_never_upgrades_loser_to_network_permission": True,
        },
        "terminal_failure_usage_is_charged": True,
        "terminal_failure_requires_publish_reload_and_detached_anchor": True,
        "terminal_failure_candidate_precedes_durability_evidence": True,
        "terminal_failure_durability_subject": "failure_candidate_hash_and_artifact_sha256",
        "terminal_failure_candidate_and_sealed_receipt_are_separate_inventory_entries": True,
        "run_control_inventory": {
            "logical_entries": [
                "manifest",
                "authorization",
                "preflight_receipts",
                "durably_sealed_network_intents",
                "endpoint_receipt",
                "shard_receipts_or_terminal_failure",
                "result_candidate_on_success",
            ],
            "planned_success_entry_count": "4_times_shards_plus_7",
            "result_candidate_is_appended_after_acyclic_base_totals": True,
            "each_entry_binds_kind_locator_semantic_hash_artifact_sha256_and_canonical_lf_byte_length": True,
            "candidate_and_sealed_intent_are_separate_inventory_entries": True,
            "final_run_control_bytes_are_derived_not_caller_selected": True,
            "materialized_state_snapshots_are_non_authoritative_and_not_published_by_this_contract": True,
            "final_positive_anchor_requires_fresh_full_inventory_scan_and_reload": True,
            "detached_final_inventory_and_anchor_evidence_are_outside_subject_inventory": True,
        },
        "partial_run_is_success": False,
        "idempotent_same_receipt_conflicting_receipt_rejected": True,
    },
    "canonicalization": {
        "json": "utf8_sorted_keys_compact_no_nan_lf",
        "exact_keys": True,
        "duplicate_keys_rejected": True,
        "immutable_create_new_publication": True,
        "filesystem_threat_model": "cooperating_writers_plain_non_reparse_parent_chain_point_in_time_validation_v1",
    },
}


def _computed_contract_hash() -> str:
    return _sha256_payload(_CONTRACT_SCHEMA)


def pilot_run_contract_hash() -> str:
    digest = _computed_contract_hash()
    if _PINNED_CONTRACT_HASH and digest != _PINNED_CONTRACT_HASH:
        raise PilotRunContractError("pilot_run_contract_changed_without_version_bump")
    return digest


__all__ = [
    "ENDPOINT_VERIFICATION_PLAN_VERSION",
    "ENDPOINT_VERIFICATION_RECEIPT_VERSION",
    "EndpointVerificationPlanV1",
    "EndpointVerificationReceiptV1",
    "MexcPublicQaPilotRunManifestV1",
    "PILOT_GLOBAL_BUDGET_VERSION",
    "PILOT_INTENT_DURABILITY_RECEIPT_VERSION",
    "PILOT_PREFLIGHT_RECEIPT_VERSION",
    "PILOT_NETWORK_INTENT_VERSION",
    "PILOT_RUN_ANCHOR_VERSION",
    "PILOT_RUN_CONTRACT_VERSION",
    "PILOT_RUN_STATE_VERSION",
    "PILOT_SHARD_PLAN_VERSION",
    "PILOT_SHARD_RESULT_VERSION",
    "PILOT_STEP_FAILURE_RECEIPT_VERSION",
    "PilotDiskPreflightReceiptV1",
    "PilotGlobalBudgetsV1",
    "PilotIntentDurabilityReceiptV1",
    "PilotNetworkIntentV1",
    "PilotRunAnchorReceiptV1",
    "PilotRunArtifactError",
    "PilotRunAuthorizationError",
    "PilotRunBudgetExceededError",
    "PilotRunContractError",
    "PilotRunError",
    "PilotRunPreflightError",
    "PilotRunStateV1",
    "PilotRunTransitionError",
    "PilotShardPlanV1",
    "PilotShardResultV1",
    "PilotStepFailureReceiptV1",
    "U5_AUTHORIZATION_RECEIPT_VERSION",
    "U5PublicPilotAuthorizationReceiptV1",
    "load_pilot_run_manifest_v1",
    "parse_pilot_run_manifest_v1",
    "parse_pilot_run_state_v1",
    "pilot_run_contract_hash",
    "publish_pilot_run_manifest_v1",
]
