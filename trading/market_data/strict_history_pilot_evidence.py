"""Read-only, bounded accounting for one admitted StrictHistory v2 shard.

This sibling contract does not collect, repair, promote, delete, or publish
anything.  It first asks the frozen public StrictHistory v2 reader to reconcile
and reload the exact detached manifest.  It then independently re-reads the
admission and source evidence to derive pilot counters and two inventories:

* logical references charge a raw CAS object once for every attempt reference;
* physical files contain each referenced pathname exactly once.

The persistent sibling writer lock is classified separately.  This reader only
proves the StrictHistory shard namespace free of residue at bounded stable
points; the surrounding pilot-output layout deliberately remains unresolved.

The live entry point uses one continuation deadline beginning before store
construction.  Synchronous calls cannot be preempted in-process.  Successful
filesystem operations have continuation checkpoints, while expected
StrictHistory/OSError failures from frozen public-store calls are also charged
to the deadline before they are classified.  Hard wall-clock preemption belongs
in a future process-isolated pilot coordinator.

``parse`` validates an internally self-consistent detached observation.  It is
not origin attestation, authentication, or proof that the live reader ran.  The
fact therefore remains explicitly non-authoritative and its strict adapter
always STOPs until a future pinned layout/accounting contract binds provenance.
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
from typing import Any

from trading.market_data.mexc_futures_transport import (
    CompleteHttpAttemptEvidenceV1,
    HttpAttemptEvidenceV1,
    parse_http_attempt_evidence_v1,
)
from trading.market_data.strict_history import StrictHistoryError
from trading.market_data.strict_history_v2 import (
    HistoryRangeRequestV2,
    STRICT_HISTORY_V2_ADMISSION_VERSION,
    STRICT_HISTORY_V2_RESTART_VERSION,
    StrictHistoryArtifactStoreV2,
    strict_history_v2_contract_hash,
)


STRICT_HISTORY_PILOT_EVIDENCE_CONTRACT_VERSION = (
    "mexc_strict_history_pilot_evidence_v1"
)
STRICT_HISTORY_PILOT_LOGICAL_REFERENCE_VERSION = (
    "mexc_strict_history_pilot_logical_reference_v1"
)
STRICT_HISTORY_PILOT_PHYSICAL_FILE_VERSION = (
    "mexc_strict_history_pilot_physical_file_v1"
)
STRICT_HISTORY_PILOT_WRITER_LOCK_FACT_VERSION = (
    "mexc_strict_history_pilot_writer_lock_fact_v1"
)
STRICT_HISTORY_PILOT_RESTART_PROOF_VERSION = (
    "mexc_strict_history_pilot_restart_no_residue_proof_v1"
)
STRICT_HISTORY_PILOT_PAGE_ACCOUNTING_VERSION = (
    "mexc_strict_history_pilot_page_accounting_v1"
)
STRICT_HISTORY_PILOT_ATTEMPT_ACCOUNTING_VERSION = (
    "mexc_strict_history_pilot_attempt_accounting_v1"
)
STRICT_HISTORY_PILOT_ADMISSION_ACCOUNTING_VERSION = (
    "mexc_strict_history_pilot_admission_accounting_v1"
)
PILOT_OUTPUT_LAYOUT_STATUS_UNRESOLVED = (
    "unresolved_not_verified_by_shard_reader"
)
PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE = (
    "non_authoritative_observation_requires_bound_live_reader_and_layout_contract"
)

_PINNED_CONTRACT_HASH = (
    "a546b37de9ed2da04eefb8d607b98719a09ab8378c2ab1d459eac02ecb899b8e"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_ADMISSION_BYTES = 16 * 1024
_MAX_ATTEMPT_RECEIPT_BYTES = 1024 * 1024
_MAX_SCOPE_BYTES = 256 * 1024
_MAX_READER_RUNTIME_US = 60_000_000
_READ_CHUNK_BYTES = 1024 * 1024
_MAX_DIRECTORY_SCAN_ENTRIES = 8_192
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class StrictHistoryPilotEvidenceError(RuntimeError):
    """Stable, non-secret-bearing base error for this read-only contract."""


class StrictHistoryPilotEvidenceContractError(StrictHistoryPilotEvidenceError):
    """The caller or a reconstructed result violated the frozen contract."""


class StrictHistoryPilotEvidenceStop(StrictHistoryPilotEvidenceError):
    """Typed STOP: evidence cannot authorize pilot accounting."""


class StrictHistoryPilotEvidenceBoundsStop(StrictHistoryPilotEvidenceStop):
    """Typed STOP: an entry, byte, or monotonic read bound was exceeded."""


class StrictHistoryPilotEvidenceResidueStop(StrictHistoryPilotEvidenceStop):
    """Typed STOP: the StrictHistory namespace is not exactly residue-free."""


class StrictHistoryPilotEvidenceLayoutStop(StrictHistoryPilotEvidenceStop):
    """Typed STOP: no pinned surrounding pilot-output layout is available."""


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
        raise StrictHistoryPilotEvidenceContractError(
            "strict_history_pilot_payload_is_not_canonical_json"
        ) from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_payload(payload: object) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StrictHistoryPilotEvidenceContractError(f"{field}_is_invalid")
    return value


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise StrictHistoryPilotEvidenceContractError(f"{field}_is_invalid")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise StrictHistoryPilotEvidenceContractError(f"{field}_is_invalid")
    return value


def _role(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ROLE_RE.fullmatch(value) is None:
        raise StrictHistoryPilotEvidenceContractError(f"{field}_is_invalid")
    return value


def _relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or len(value) > 512:
        raise StrictHistoryPilotEvidenceContractError(f"{field}_is_invalid")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or candidate.as_posix() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise StrictHistoryPilotEvidenceContractError(f"{field}_is_invalid")
    return value


def _exact_mapping(
    payload: object,
    expected: frozenset[str],
    *,
    code: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise StrictHistoryPilotEvidenceContractError(code)
    return dict(payload)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_duplicate_json_key"
            )
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise StrictHistoryPilotEvidenceStop("strict_history_pilot_nonfinite_json")


def _parse_canonical_lf_object(raw: bytes, *, code: str) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise StrictHistoryPilotEvidenceStop(code)
    try:
        payload = json.loads(
            raw[:-1].decode("utf-8", errors="strict"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except StrictHistoryPilotEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StrictHistoryPilotEvidenceStop(code) from exc
    if not isinstance(payload, dict) or _canonical_bytes(payload) + b"\n" != raw:
        raise StrictHistoryPilotEvidenceStop(code)
    return payload


@dataclass(frozen=True, slots=True)
class PilotLogicalReferenceV1:
    ordinal: int
    role: str
    relative_path: str
    reference_hash: str
    file_sha256: str
    byte_count: int
    page_ordinal: int
    attempt_ordinal: int
    contract_version: str = STRICT_HISTORY_PILOT_LOGICAL_REFERENCE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_PILOT_LOGICAL_REFERENCE_VERSION:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_logical_reference_version_mismatch"
            )
        _strict_int(self.ordinal, field="logical_reference_ordinal")
        _role(self.role, field="logical_reference_role")
        _relative_path(self.relative_path, field="logical_reference_path")
        _digest(self.reference_hash, field="logical_reference_hash")
        _digest(self.file_sha256, field="logical_reference_file_sha256")
        _strict_int(self.byte_count, field="logical_reference_byte_count")
        if type(self.page_ordinal) is not int or self.page_ordinal < -1:
            raise StrictHistoryPilotEvidenceContractError(
                "logical_reference_page_ordinal_is_invalid"
            )
        if type(self.attempt_ordinal) is not int or self.attempt_ordinal < -1:
            raise StrictHistoryPilotEvidenceContractError(
                "logical_reference_attempt_ordinal_is_invalid"
            )
        attempt_role = self.role in {"attempt_receipt", "raw_body"}
        if attempt_role != (self.page_ordinal >= 0 and self.attempt_ordinal >= 0):
            raise StrictHistoryPilotEvidenceContractError(
                "logical_reference_attempt_coordinates_mismatch"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "ordinal": self.ordinal,
            "role": self.role,
            "relative_path": self.relative_path,
            "reference_hash": self.reference_hash,
            "file_sha256": self.file_sha256,
            "byte_count": self.byte_count,
            "page_ordinal": self.page_ordinal,
            "attempt_ordinal": self.attempt_ordinal,
        }

    @classmethod
    def parse(cls, payload: object) -> "PilotLogicalReferenceV1":
        data = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="strict_history_pilot_logical_reference_schema_mismatch",
        )
        try:
            return cls(**data)
        except TypeError as exc:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_logical_reference_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class PilotPhysicalFileV1:
    relative_path: str
    role: str
    file_sha256: str
    byte_count: int
    logical_reference_count: int
    contract_version: str = STRICT_HISTORY_PILOT_PHYSICAL_FILE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_PILOT_PHYSICAL_FILE_VERSION:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_physical_file_version_mismatch"
            )
        _relative_path(self.relative_path, field="physical_file_path")
        _role(self.role, field="physical_file_role")
        _digest(self.file_sha256, field="physical_file_sha256")
        _strict_int(self.byte_count, field="physical_file_byte_count")
        _strict_int(
            self.logical_reference_count,
            field="physical_file_logical_reference_count",
            minimum=1,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "relative_path": self.relative_path,
            "role": self.role,
            "file_sha256": self.file_sha256,
            "byte_count": self.byte_count,
            "logical_reference_count": self.logical_reference_count,
        }

    @classmethod
    def parse(cls, payload: object) -> "PilotPhysicalFileV1":
        data = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="strict_history_pilot_physical_file_schema_mismatch",
        )
        try:
            return cls(**data)
        except TypeError as exc:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_physical_file_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class PilotWriterLockFactV1:
    status: str
    symbolic_locator: str
    file_sha256: str | None
    byte_count: int
    link_count: int
    contract_version: str = STRICT_HISTORY_PILOT_WRITER_LOCK_FACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_PILOT_WRITER_LOCK_FACT_VERSION:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_writer_lock_fact_version_mismatch"
            )
        if self.status not in {"absent", "present_plain_regular"}:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_writer_lock_status_is_invalid"
            )
        if self.symbolic_locator != "persistent_sibling_writer_lock_outside_shard_root":
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_writer_lock_locator_is_invalid"
            )
        _strict_int(self.byte_count, field="writer_lock_byte_count")
        _strict_int(self.link_count, field="writer_lock_link_count")
        if self.status == "absent":
            if self.file_sha256 is not None or self.byte_count != 0 or self.link_count != 0:
                raise StrictHistoryPilotEvidenceContractError(
                    "strict_history_pilot_absent_writer_lock_has_metadata"
                )
        else:
            _digest(self.file_sha256, field="writer_lock_file_sha256")
            if (
                self.file_sha256 != _sha256_bytes(b"0")
                or self.byte_count != 1
                or self.link_count != 1
            ):
                raise StrictHistoryPilotEvidenceContractError(
                    "strict_history_pilot_writer_lock_metadata_is_invalid"
                )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "status": self.status,
            "symbolic_locator": self.symbolic_locator,
            "file_sha256": self.file_sha256,
            "byte_count": self.byte_count,
            "link_count": self.link_count,
        }

    @classmethod
    def parse(cls, payload: object) -> "PilotWriterLockFactV1":
        data = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="strict_history_pilot_writer_lock_fact_schema_mismatch",
        )
        try:
            return cls(**data)
        except TypeError as exc:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_writer_lock_fact_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class PilotRestartNoResidueProofV1:
    request_id: str
    manifest_hash: str
    request_state: str
    ready: bool
    temp_paths: tuple[str, ...]
    unreferenced_attempt_paths: tuple[str, ...]
    unreferenced_raw_paths: tuple[str, ...]
    alternate_normalized_paths: tuple[str, ...]
    observation_count: int
    restart_contract_version: str
    contract_version: str = STRICT_HISTORY_PILOT_RESTART_PROOF_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_PILOT_RESTART_PROOF_VERSION:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_restart_proof_version_mismatch"
            )
        _digest(self.request_id, field="restart_proof_request_id")
        _digest(self.manifest_hash, field="restart_proof_manifest_hash")
        if self.request_state != "complete_verified" or self.ready is not True:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_restart_proof_is_not_ready_complete"
            )
        for field in (
            "temp_paths",
            "unreferenced_attempt_paths",
            "unreferenced_raw_paths",
            "alternate_normalized_paths",
        ):
            value = getattr(self, field)
            if not isinstance(value, tuple) or value:
                raise StrictHistoryPilotEvidenceContractError(
                    "strict_history_pilot_restart_proof_residue_is_not_empty"
                )
        if self.observation_count != 2:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_restart_proof_observation_count_mismatch"
            )
        if self.restart_contract_version != STRICT_HISTORY_V2_RESTART_VERSION:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_restart_proof_dependency_mismatch"
            )

    @property
    def proof_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "request_id": self.request_id,
            "manifest_hash": self.manifest_hash,
            "request_state": self.request_state,
            "ready": self.ready,
            "temp_paths": list(self.temp_paths),
            "unreferenced_attempt_paths": list(self.unreferenced_attempt_paths),
            "unreferenced_raw_paths": list(self.unreferenced_raw_paths),
            "alternate_normalized_paths": list(self.alternate_normalized_paths),
            "observation_count": self.observation_count,
            "restart_contract_version": self.restart_contract_version,
        }

    @classmethod
    def parse(cls, payload: object) -> "PilotRestartNoResidueProofV1":
        data = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="strict_history_pilot_restart_proof_schema_mismatch",
        )
        for field in (
            "temp_paths",
            "unreferenced_attempt_paths",
            "unreferenced_raw_paths",
            "alternate_normalized_paths",
        ):
            value = data.get(field)
            if not isinstance(value, list):
                raise StrictHistoryPilotEvidenceContractError(
                    "strict_history_pilot_restart_proof_wire_type_mismatch"
                )
            data[field] = tuple(value)
        try:
            return cls(**data)
        except TypeError as exc:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_restart_proof_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class PilotPageAccountingV1:
    page_ordinal: int
    page_receipt_hash: str
    row_count: int
    attempt_count: int
    contract_version: str = STRICT_HISTORY_PILOT_PAGE_ACCOUNTING_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_PILOT_PAGE_ACCOUNTING_VERSION:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_page_accounting_version_mismatch"
            )
        _strict_int(self.page_ordinal, field="page_accounting_ordinal")
        _digest(self.page_receipt_hash, field="page_accounting_receipt_hash")
        _strict_int(self.row_count, field="page_accounting_row_count", minimum=1)
        _strict_int(
            self.attempt_count,
            field="page_accounting_attempt_count",
            minimum=1,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "page_ordinal": self.page_ordinal,
            "page_receipt_hash": self.page_receipt_hash,
            "row_count": self.row_count,
            "attempt_count": self.attempt_count,
        }

    @classmethod
    def parse(cls, payload: object) -> "PilotPageAccountingV1":
        data = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="strict_history_pilot_page_accounting_schema_mismatch",
        )
        try:
            return cls(**data)
        except TypeError as exc:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_page_accounting_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class PilotAttemptAccountingV1:
    page_ordinal: int
    attempt_ordinal: int
    attempt_receipt_hash: str
    raw_body_sha256: str
    raw_body_length: int
    request_started_monotonic_us: int
    terminal_monotonic_us: int
    elapsed_monotonic_us: int
    contract_version: str = STRICT_HISTORY_PILOT_ATTEMPT_ACCOUNTING_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_PILOT_ATTEMPT_ACCOUNTING_VERSION:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_attempt_accounting_version_mismatch"
            )
        _strict_int(self.page_ordinal, field="attempt_accounting_page_ordinal")
        _strict_int(self.attempt_ordinal, field="attempt_accounting_ordinal")
        _digest(
            self.attempt_receipt_hash,
            field="attempt_accounting_receipt_hash",
        )
        _digest(self.raw_body_sha256, field="attempt_accounting_raw_hash")
        _strict_int(
            self.raw_body_length,
            field="attempt_accounting_raw_length",
        )
        started = _strict_int(
            self.request_started_monotonic_us,
            field="attempt_accounting_started_monotonic_us",
        )
        terminal = _strict_int(
            self.terminal_monotonic_us,
            field="attempt_accounting_terminal_monotonic_us",
        )
        elapsed = _strict_int(
            self.elapsed_monotonic_us,
            field="attempt_accounting_elapsed_monotonic_us",
        )
        if terminal < started or elapsed != terminal - started:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_attempt_accounting_timing_mismatch"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "page_ordinal": self.page_ordinal,
            "attempt_ordinal": self.attempt_ordinal,
            "attempt_receipt_hash": self.attempt_receipt_hash,
            "raw_body_sha256": self.raw_body_sha256,
            "raw_body_length": self.raw_body_length,
            "request_started_monotonic_us": self.request_started_monotonic_us,
            "terminal_monotonic_us": self.terminal_monotonic_us,
            "elapsed_monotonic_us": self.elapsed_monotonic_us,
        }

    @classmethod
    def parse(cls, payload: object) -> "PilotAttemptAccountingV1":
        data = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="strict_history_pilot_attempt_accounting_schema_mismatch",
        )
        try:
            return cls(**data)
        except TypeError as exc:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_attempt_accounting_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class PilotAdmissionAccountingV1:
    admission_hash: str
    graph_logical_storage_bytes: int
    admission_marker_bytes: int
    admitted_total_logical_storage_bytes: int
    manifest_collection_runtime_us: int
    admission_full_reload_runtime_us: int
    contract_version: str = STRICT_HISTORY_PILOT_ADMISSION_ACCOUNTING_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_PILOT_ADMISSION_ACCOUNTING_VERSION:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_admission_accounting_version_mismatch"
            )
        _digest(self.admission_hash, field="admission_accounting_hash")
        graph_bytes = _strict_int(
            self.graph_logical_storage_bytes,
            field="admission_accounting_graph_bytes",
        )
        marker_bytes = _strict_int(
            self.admission_marker_bytes,
            field="admission_accounting_marker_bytes",
            minimum=1,
        )
        admitted_bytes = _strict_int(
            self.admitted_total_logical_storage_bytes,
            field="admission_accounting_total_bytes",
        )
        manifest_runtime = _strict_int(
            self.manifest_collection_runtime_us,
            field="admission_accounting_manifest_runtime_us",
        )
        admission_runtime = _strict_int(
            self.admission_full_reload_runtime_us,
            field="admission_accounting_full_runtime_us",
        )
        if (
            admitted_bytes != graph_bytes + marker_bytes
            or admission_runtime < manifest_runtime
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_admission_accounting_mismatch"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "admission_hash": self.admission_hash,
            "graph_logical_storage_bytes": self.graph_logical_storage_bytes,
            "admission_marker_bytes": self.admission_marker_bytes,
            "admitted_total_logical_storage_bytes": (
                self.admitted_total_logical_storage_bytes
            ),
            "manifest_collection_runtime_us": self.manifest_collection_runtime_us,
            "admission_full_reload_runtime_us": (
                self.admission_full_reload_runtime_us
            ),
        }

    @classmethod
    def parse(cls, payload: object) -> "PilotAdmissionAccountingV1":
        data = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="strict_history_pilot_admission_accounting_schema_mismatch",
        )
        try:
            return cls(**data)
        except TypeError as exc:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_admission_accounting_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class StrictHistoryPilotEvidenceV1:
    evidence_contract_hash: str
    request_id: str
    manifest_hash: str
    history_contract_hash: str
    normalized_shard_sha256: str
    page_count: int
    row_count: int
    attempt_count: int
    raw_body_reference_count: int
    unique_raw_body_count: int
    actual_total_raw_body_bytes: int
    unique_physical_raw_body_bytes: int
    manifest_collection_runtime_us: int
    admission_full_reload_runtime_us: int
    attempt_elapsed_runtime_us: int
    observed_monotonic_inter_attempt_sleep_us: int
    admitted_total_logical_storage_bytes: int
    unique_physical_referenced_bytes: int
    logical_references: tuple[PilotLogicalReferenceV1, ...]
    physical_files: tuple[PilotPhysicalFileV1, ...]
    page_accounting: tuple[PilotPageAccountingV1, ...]
    attempt_accounting: tuple[PilotAttemptAccountingV1, ...]
    admission_accounting: PilotAdmissionAccountingV1
    restart_contract_version: str
    restart_observation_count: int
    strict_history_namespace_residue_free: bool
    restart_no_residue_proof: PilotRestartNoResidueProofV1
    writer_lock: PilotWriterLockFactV1
    pilot_output_layout_status: str
    authority_status: str
    contract_version: str = STRICT_HISTORY_PILOT_EVIDENCE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != STRICT_HISTORY_PILOT_EVIDENCE_CONTRACT_VERSION:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_evidence_version_mismatch"
            )
        for field in (
            "evidence_contract_hash",
            "request_id",
            "manifest_hash",
            "history_contract_hash",
            "normalized_shard_sha256",
        ):
            _digest(getattr(self, field), field=f"evidence_{field}")
        if self.evidence_contract_hash != strict_history_pilot_evidence_contract_hash():
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_evidence_contract_hash_mismatch"
            )
        for field in (
            "page_count",
            "row_count",
            "attempt_count",
            "raw_body_reference_count",
            "unique_raw_body_count",
            "actual_total_raw_body_bytes",
            "unique_physical_raw_body_bytes",
            "manifest_collection_runtime_us",
            "admission_full_reload_runtime_us",
            "attempt_elapsed_runtime_us",
            "observed_monotonic_inter_attempt_sleep_us",
            "admitted_total_logical_storage_bytes",
            "unique_physical_referenced_bytes",
        ):
            _strict_int(getattr(self, field), field=f"evidence_{field}")
        if not isinstance(self.logical_references, tuple) or not all(
            isinstance(item, PilotLogicalReferenceV1) for item in self.logical_references
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_logical_references_are_not_immutable"
            )
        if not isinstance(self.physical_files, tuple) or not all(
            isinstance(item, PilotPhysicalFileV1) for item in self.physical_files
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_physical_files_are_not_immutable"
            )
        if not isinstance(self.page_accounting, tuple) or not all(
            isinstance(item, PilotPageAccountingV1) for item in self.page_accounting
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_page_accounting_is_not_immutable"
            )
        if not isinstance(self.attempt_accounting, tuple) or not all(
            isinstance(item, PilotAttemptAccountingV1)
            for item in self.attempt_accounting
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_attempt_accounting_is_not_immutable"
            )
        if not isinstance(self.admission_accounting, PilotAdmissionAccountingV1):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_admission_accounting_is_invalid"
            )
        if tuple(item.ordinal for item in self.logical_references) != tuple(
            range(len(self.logical_references))
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_logical_reference_order_is_invalid"
            )
        if tuple(item.relative_path for item in self.physical_files) != tuple(
            sorted(item.relative_path for item in self.physical_files)
        ) or len({item.relative_path for item in self.physical_files}) != len(
            self.physical_files
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_physical_file_order_is_invalid"
            )
        if self.restart_contract_version != STRICT_HISTORY_V2_RESTART_VERSION:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_restart_contract_version_mismatch"
            )
        if self.restart_observation_count != 2:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_restart_observation_count_mismatch"
            )
        _strict_bool(
            self.strict_history_namespace_residue_free,
            field="evidence_namespace_residue_free",
        )
        if self.strict_history_namespace_residue_free is not True:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_namespace_must_be_residue_free"
            )
        if not isinstance(self.writer_lock, PilotWriterLockFactV1):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_writer_lock_fact_is_invalid"
            )
        if not isinstance(
            self.restart_no_residue_proof, PilotRestartNoResidueProofV1
        ) or (
            self.restart_no_residue_proof.request_id != self.request_id
            or self.restart_no_residue_proof.manifest_hash != self.manifest_hash
            or self.restart_no_residue_proof.restart_contract_version
            != self.restart_contract_version
            or self.restart_no_residue_proof.observation_count
            != self.restart_observation_count
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_restart_proof_binding_mismatch"
            )
        if self.pilot_output_layout_status != PILOT_OUTPUT_LAYOUT_STATUS_UNRESOLVED:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_output_layout_status_mismatch"
            )
        if self.authority_status != PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_evidence_authority_status_mismatch"
            )
        logical_total = sum(item.byte_count for item in self.logical_references)
        physical_total = sum(item.byte_count for item in self.physical_files)
        if logical_total != self.admitted_total_logical_storage_bytes:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_logical_inventory_total_mismatch"
            )
        if physical_total != self.unique_physical_referenced_bytes:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_physical_inventory_total_mismatch"
            )
        if self.raw_body_reference_count != self.attempt_count:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_raw_reference_count_mismatch"
            )
        raw_logical = tuple(
            item for item in self.logical_references if item.role == "raw_body"
        )
        raw_physical = tuple(item for item in self.physical_files if item.role == "raw_body")
        if (
            len(raw_logical) != self.raw_body_reference_count
            or len(raw_physical) != self.unique_raw_body_count
            or sum(item.byte_count for item in raw_logical)
            != self.actual_total_raw_body_bytes
            or sum(item.byte_count for item in raw_physical)
            != self.unique_physical_raw_body_bytes
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_raw_accounting_mismatch"
            )
        reference_counts: dict[str, int] = {}
        for item in self.logical_references:
            reference_counts[item.relative_path] = (
                reference_counts.get(item.relative_path, 0) + 1
            )
        if {
            item.relative_path: item.logical_reference_count
            for item in self.physical_files
        } != reference_counts:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_physical_reference_counts_mismatch"
            )
        if self.admission_full_reload_runtime_us < self.manifest_collection_runtime_us:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_admission_runtime_precedes_manifest"
            )
        if self.history_contract_hash != strict_history_v2_contract_hash():
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_history_contract_hash_mismatch"
            )
        if (
            self.page_count < 1
            or self.row_count < self.page_count
            or self.attempt_count < self.page_count
            or self.unique_raw_body_count < 1
            or self.unique_raw_body_count > self.attempt_count
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_positive_success_counts_are_invalid"
            )
        expected_logical_count = (2 * self.attempt_count) + 4
        if len(self.logical_references) != expected_logical_count:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_logical_reference_shape_mismatch"
            )
        if (
            self.logical_references[0].role != "scope_marker"
            or self.logical_references[0].relative_path != "scope.json"
            or self.logical_references[-3].role != "normalized_shard"
            or self.logical_references[-3].relative_path
            != (
                f"normalized/{self.request_id}/"
                f"{self.normalized_shard_sha256}.jsonl"
            )
            or self.logical_references[-3].reference_hash
            != self.normalized_shard_sha256
            or self.logical_references[-2].role != "manifest"
            or self.logical_references[-2].relative_path
            != f"collections/{self.request_id}/manifest.json"
            or self.logical_references[-2].reference_hash != self.manifest_hash
            or self.logical_references[-1].role != "admission_marker"
            or self.logical_references[-1].relative_path
            != f"collections/{self.request_id}/admission.json"
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_fixed_logical_reference_identity_mismatch"
            )
        coordinates: list[tuple[int, int]] = []
        source_references = self.logical_references[1:-3]
        for offset in range(0, len(source_references), 2):
            receipt = source_references[offset]
            raw = source_references[offset + 1]
            if (
                receipt.role != "attempt_receipt"
                or raw.role != "raw_body"
                or receipt.page_ordinal != raw.page_ordinal
                or receipt.attempt_ordinal != raw.attempt_ordinal
                or receipt.relative_path
                != f"attempts/{receipt.reference_hash}.json"
                or raw.relative_path
                != (
                    f"raw/sha256/{raw.reference_hash[:2]}/"
                    f"{raw.reference_hash}.bin"
                )
            ):
                raise StrictHistoryPilotEvidenceContractError(
                    "strict_history_pilot_attempt_reference_pair_mismatch"
                )
            if raw.file_sha256 != raw.reference_hash:
                raise StrictHistoryPilotEvidenceContractError(
                    "strict_history_pilot_raw_reference_content_hash_mismatch"
                )
            coordinates.append((receipt.page_ordinal, receipt.attempt_ordinal))
        expected_coordinates: list[tuple[int, int]] = []
        cursor = 0
        for page_ordinal in range(self.page_count):
            page_attempts = 0
            while cursor < len(coordinates) and coordinates[cursor][0] == page_ordinal:
                if coordinates[cursor][1] != page_attempts:
                    raise StrictHistoryPilotEvidenceContractError(
                        "strict_history_pilot_attempt_coordinate_order_mismatch"
                    )
                expected_coordinates.append((page_ordinal, page_attempts))
                page_attempts += 1
                cursor += 1
            if page_attempts == 0:
                raise StrictHistoryPilotEvidenceContractError(
                    "strict_history_pilot_page_has_no_attempt_reference"
                )
        if cursor != len(coordinates) or coordinates != expected_coordinates:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_attempt_coordinate_domain_mismatch"
            )
        if (
            len(self.page_accounting) != self.page_count
            or tuple(item.page_ordinal for item in self.page_accounting)
            != tuple(range(self.page_count))
            or len({item.page_receipt_hash for item in self.page_accounting})
            != self.page_count
            or sum(item.row_count for item in self.page_accounting) != self.row_count
            or sum(item.attempt_count for item in self.page_accounting)
            != self.attempt_count
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_page_accounting_summary_mismatch"
            )
        coordinate_counts = tuple(
            sum(1 for coordinate in coordinates if coordinate[0] == page_ordinal)
            for page_ordinal in range(self.page_count)
        )
        if tuple(item.attempt_count for item in self.page_accounting) != coordinate_counts:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_page_attempt_accounting_mismatch"
            )
        if (
            len(self.attempt_accounting) != self.attempt_count
            or tuple(
                (item.page_ordinal, item.attempt_ordinal)
                for item in self.attempt_accounting
            )
            != tuple(coordinates)
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_attempt_accounting_coordinate_mismatch"
            )
        prior_terminal: int | None = None
        derived_sleep = 0
        for index, accounting in enumerate(self.attempt_accounting):
            receipt_reference = source_references[index * 2]
            raw_reference = source_references[(index * 2) + 1]
            if (
                accounting.attempt_receipt_hash
                != receipt_reference.reference_hash
                or accounting.raw_body_sha256 != raw_reference.reference_hash
                or accounting.raw_body_length != raw_reference.byte_count
            ):
                raise StrictHistoryPilotEvidenceContractError(
                    "strict_history_pilot_attempt_accounting_source_binding_mismatch"
                )
            if prior_terminal is not None:
                gap = accounting.request_started_monotonic_us - prior_terminal
                if gap < 0:
                    raise StrictHistoryPilotEvidenceContractError(
                        "strict_history_pilot_attempt_accounting_timing_regressed"
                    )
                derived_sleep += gap
            prior_terminal = accounting.terminal_monotonic_us
        if (
            sum(item.raw_body_length for item in self.attempt_accounting)
            != self.actual_total_raw_body_bytes
            or sum(item.elapsed_monotonic_us for item in self.attempt_accounting)
            != self.attempt_elapsed_runtime_us
            or derived_sleep != self.observed_monotonic_inter_attempt_sleep_us
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_attempt_derived_scalar_mismatch"
            )
        if (
            self.attempt_elapsed_runtime_us
            + self.observed_monotonic_inter_attempt_sleep_us
            > self.manifest_collection_runtime_us
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_collection_runtime_accounting_mismatch"
            )
        admission_reference = self.logical_references[-1]
        graph_logical_bytes = sum(
            item.byte_count for item in self.logical_references[:-1]
        )
        if (
            self.admission_accounting.admission_hash
            != admission_reference.reference_hash
            or self.admission_accounting.graph_logical_storage_bytes
            != graph_logical_bytes
            or self.admission_accounting.admission_marker_bytes
            != admission_reference.byte_count
            or self.admission_accounting.admitted_total_logical_storage_bytes
            != self.admitted_total_logical_storage_bytes
            or self.admission_accounting.manifest_collection_runtime_us
            != self.manifest_collection_runtime_us
            or self.admission_accounting.admission_full_reload_runtime_us
            != self.admission_full_reload_runtime_us
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_admission_derived_scalar_mismatch"
            )
        physical_by_path = {item.relative_path: item for item in self.physical_files}
        logical_by_path: dict[str, PilotLogicalReferenceV1] = {}
        for item in self.logical_references:
            prior = logical_by_path.setdefault(item.relative_path, item)
            if (
                prior.role != item.role
                or prior.file_sha256 != item.file_sha256
                or prior.byte_count != item.byte_count
                or (
                    prior.relative_path == item.relative_path
                    and item.role != "raw_body"
                    and prior is not item
                )
            ):
                raise StrictHistoryPilotEvidenceContractError(
                    "strict_history_pilot_duplicate_logical_path_is_invalid"
                )
            physical = physical_by_path.get(item.relative_path)
            if (
                physical is None
                or physical.role != item.role
                or physical.file_sha256 != item.file_sha256
                or physical.byte_count != item.byte_count
            ):
                raise StrictHistoryPilotEvidenceContractError(
                    "strict_history_pilot_logical_physical_file_binding_mismatch"
                )
        if self.unique_raw_body_count != len(
            {item.relative_path for item in raw_logical}
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_unique_raw_count_mismatch"
            )
        if (
            self.logical_references[0].reference_hash
            != self.logical_references[0].file_sha256
            or self.logical_references[-3].reference_hash
            != self.logical_references[-3].file_sha256
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_content_addressed_fixed_file_mismatch"
            )
        logical_role_counts = {
            role: sum(1 for item in self.logical_references if item.role == role)
            for role in {
                "scope_marker",
                "attempt_receipt",
                "raw_body",
                "normalized_shard",
                "manifest",
                "admission_marker",
            }
        }
        if logical_role_counts != {
            "scope_marker": 1,
            "attempt_receipt": self.attempt_count,
            "raw_body": self.attempt_count,
            "normalized_shard": 1,
            "manifest": 1,
            "admission_marker": 1,
        }:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_logical_role_multiplicity_mismatch"
            )
        physical_role_counts = {
            role: sum(1 for item in self.physical_files if item.role == role)
            for role in logical_role_counts
        }
        if physical_role_counts != {
            "scope_marker": 1,
            "attempt_receipt": self.attempt_count,
            "raw_body": self.unique_raw_body_count,
            "normalized_shard": 1,
            "manifest": 1,
            "admission_marker": 1,
        }:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_physical_role_multiplicity_mismatch"
            )

    @property
    def evidence_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "evidence_contract_hash": self.evidence_contract_hash,
            "request_id": self.request_id,
            "manifest_hash": self.manifest_hash,
            "history_contract_hash": self.history_contract_hash,
            "normalized_shard_sha256": self.normalized_shard_sha256,
            "page_count": self.page_count,
            "row_count": self.row_count,
            "attempt_count": self.attempt_count,
            "raw_body_reference_count": self.raw_body_reference_count,
            "unique_raw_body_count": self.unique_raw_body_count,
            "actual_total_raw_body_bytes": self.actual_total_raw_body_bytes,
            "unique_physical_raw_body_bytes": self.unique_physical_raw_body_bytes,
            "manifest_collection_runtime_us": self.manifest_collection_runtime_us,
            "admission_full_reload_runtime_us": self.admission_full_reload_runtime_us,
            "attempt_elapsed_runtime_us": self.attempt_elapsed_runtime_us,
            "observed_monotonic_inter_attempt_sleep_us": (
                self.observed_monotonic_inter_attempt_sleep_us
            ),
            "admitted_total_logical_storage_bytes": (
                self.admitted_total_logical_storage_bytes
            ),
            "unique_physical_referenced_bytes": self.unique_physical_referenced_bytes,
            "logical_references": [item.as_dict() for item in self.logical_references],
            "physical_files": [item.as_dict() for item in self.physical_files],
            "page_accounting": [item.as_dict() for item in self.page_accounting],
            "attempt_accounting": [
                item.as_dict() for item in self.attempt_accounting
            ],
            "admission_accounting": self.admission_accounting.as_dict(),
            "restart_contract_version": self.restart_contract_version,
            "restart_observation_count": self.restart_observation_count,
            "strict_history_namespace_residue_free": (
                self.strict_history_namespace_residue_free
            ),
            "restart_no_residue_proof": self.restart_no_residue_proof.as_dict(),
            "writer_lock": self.writer_lock.as_dict(),
            "pilot_output_layout_status": self.pilot_output_layout_status,
            "authority_status": self.authority_status,
        }

    def require_pilot_compatible(self) -> "StrictHistoryPilotEvidenceV1":
        """Typed STOP until live-reader authority and pilot layout are bound."""

        raise StrictHistoryPilotEvidenceLayoutStop(
            "strict_history_pilot_evidence_is_non_authoritative_and_layout_unresolved"
        )

    @classmethod
    def parse(cls, payload: object) -> "StrictHistoryPilotEvidenceV1":
        """Validate detached self-consistency, never live-reader provenance."""

        data = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="strict_history_pilot_evidence_schema_mismatch",
        )
        logical = data.get("logical_references")
        physical = data.get("physical_files")
        pages = data.get("page_accounting")
        attempts = data.get("attempt_accounting")
        if (
            not isinstance(logical, list)
            or not isinstance(physical, list)
            or not isinstance(pages, list)
            or not isinstance(attempts, list)
        ):
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_inventory_wire_types_are_invalid"
            )
        data["logical_references"] = tuple(
            PilotLogicalReferenceV1.parse(item) for item in logical
        )
        data["physical_files"] = tuple(PilotPhysicalFileV1.parse(item) for item in physical)
        data["page_accounting"] = tuple(
            PilotPageAccountingV1.parse(item) for item in pages
        )
        data["attempt_accounting"] = tuple(
            PilotAttemptAccountingV1.parse(item) for item in attempts
        )
        data["admission_accounting"] = PilotAdmissionAccountingV1.parse(
            data.get("admission_accounting")
        )
        data["restart_no_residue_proof"] = PilotRestartNoResidueProofV1.parse(
            data.get("restart_no_residue_proof")
        )
        data["writer_lock"] = PilotWriterLockFactV1.parse(data.get("writer_lock"))
        try:
            return cls(**data)
        except TypeError as exc:
            raise StrictHistoryPilotEvidenceContractError(
                "strict_history_pilot_evidence_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class _ReadFile:
    raw: bytes
    signature: tuple[int, int, int, int | None, int]


def _deadline_ns() -> int:
    return time.monotonic_ns() + (_MAX_READER_RUNTIME_US * 1_000)


def _check_deadline(deadline_ns: int) -> None:
    if time.monotonic_ns() > deadline_ns:
        raise StrictHistoryPilotEvidenceBoundsStop(
            "strict_history_pilot_read_runtime_exceeded"
        )


def _raise_public_call_failure(
    exc: BaseException,
    *,
    deadline_ns: int,
    code: str,
) -> None:
    try:
        _check_deadline(deadline_ns)
    except StrictHistoryPilotEvidenceBoundsStop as deadline_exc:
        deadline_exc.__cause__ = exc
        raise deadline_exc
    failure = StrictHistoryPilotEvidenceStop(code)
    failure.__cause__ = exc
    raise failure


def _is_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    )


def _signature(info: os.stat_result) -> tuple[int, int, int, int | None, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        getattr(info, "st_mtime_ns", None),
        info.st_nlink,
    )


def _validate_plain_directory_chain(
    anchor: Path,
    directory: Path,
    *,
    deadline_ns: int,
) -> None:
    """Reject a directory/reparse swap between public and independent reads."""

    try:
        relative = directory.relative_to(anchor)
    except ValueError as exc:
        raise StrictHistoryPilotEvidenceContractError(
            "strict_history_pilot_directory_escapes_anchor"
        ) from exc
    candidates = [anchor]
    cursor = anchor
    for part in relative.parts:
        cursor = cursor / part
        candidates.append(cursor)
    anchor_device: int | None = None
    for candidate in candidates:
        _check_deadline(deadline_ns)
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_directory_chain_stat_failed"
            ) from exc
        if _is_reparse(info) or not stat.S_ISDIR(info.st_mode):
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_directory_chain_is_not_plain"
            )
        if anchor_device is None:
            anchor_device = info.st_dev
        elif info.st_dev != anchor_device:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_directory_chain_crosses_volume"
            )
        _reject_windows_named_streams(
            candidate,
            deadline_ns=deadline_ns,
            allow_no_streams=True,
        )
        _check_deadline(deadline_ns)


def _reject_windows_named_streams(
    path: Path,
    *,
    deadline_ns: int,
    allow_no_streams: bool = False,
) -> None:
    """Reject NTFS alternate data streams before claiming physical bytes."""

    _check_deadline(deadline_ns)
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class _Win32FindStreamData(ctypes.Structure):
        _fields_ = [
            ("stream_size", ctypes.c_longlong),
            ("stream_name", ctypes.c_wchar * 296),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(_Win32FindStreamData),
        wintypes.DWORD,
    ]
    find_first.restype = wintypes.HANDLE
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Win32FindStreamData)]
    find_next.restype = wintypes.BOOL
    find_close = kernel32.FindClose
    find_close.argtypes = [wintypes.HANDLE]
    find_close.restype = wintypes.BOOL
    data = _Win32FindStreamData()
    handle = find_first(os.fspath(path), 0, ctypes.byref(data), 0)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        error = ctypes.get_last_error()
        if allow_no_streams and error == 38:  # directory with no data streams
            _check_deadline(deadline_ns)
            return
        raise StrictHistoryPilotEvidenceStop(
            f"strict_history_pilot_stream_enumeration_failed.{error}"
        )
    close_error: int | None = None
    failure: StrictHistoryPilotEvidenceStop | None = None
    try:
        _check_deadline(deadline_ns)
        if allow_no_streams:
            failure = StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_named_data_stream_is_forbidden"
            )
        elif data.stream_name != "::$DATA":
            failure = StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_named_data_stream_is_forbidden"
            )
        elif find_next(handle, ctypes.byref(data)):
            failure = StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_named_data_stream_is_forbidden"
            )
        else:
            error = ctypes.get_last_error()
            if error != 38:  # ERROR_HANDLE_EOF
                failure = StrictHistoryPilotEvidenceStop(
                    f"strict_history_pilot_stream_enumeration_failed.{error}"
                )
    finally:
        if not find_close(handle):
            close_error = ctypes.get_last_error()
    if close_error is not None:
        raise StrictHistoryPilotEvidenceStop(
            f"strict_history_pilot_stream_enumeration_close_failed.{close_error}"
        )
    if failure is not None:
        raise failure
    _check_deadline(deadline_ns)


def _scan_plain_directory_namespace(
    root: Path,
    *,
    deadline_ns: int,
) -> tuple[tuple[str, tuple[int, int, int, int | None, int]], ...]:
    """Bound every present directory, including canonical empty raw prefixes."""

    _check_deadline(deadline_ns)
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_directory_scan_root_stat_failed"
        ) from exc
    if _is_reparse(root_info) or not stat.S_ISDIR(root_info.st_mode):
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_directory_scan_root_is_not_plain"
        )
    root_device = root_info.st_dev
    pending = [root]
    result: dict[str, tuple[int, int, int, int | None, int]] = {}
    observed_entries = 0
    while pending:
        _check_deadline(deadline_ns)
        directory = pending.pop()
        try:
            directory_info = directory.lstat()
        except OSError as exc:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_directory_scan_stat_failed"
            ) from exc
        if (
            _is_reparse(directory_info)
            or not stat.S_ISDIR(directory_info.st_mode)
            or directory_info.st_dev != root_device
        ):
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_directory_scan_found_hostile_directory"
            )
        _reject_windows_named_streams(
            directory,
            deadline_ns=deadline_ns,
            allow_no_streams=True,
        )
        relative = (
            ""
            if directory == root
            else _relative_path(
                directory.relative_to(root).as_posix(),
                field="directory_scan_relative_path",
            )
        )
        if relative in result:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_directory_scan_duplicate_path"
            )
        result[relative] = _signature(directory_info)
        if len(result) > _MAX_DIRECTORY_SCAN_ENTRIES:
            raise StrictHistoryPilotEvidenceBoundsStop(
                "strict_history_pilot_directory_scan_entry_bound_exceeded"
            )
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    _check_deadline(deadline_ns)
                    observed_entries += 1
                    if observed_entries > _MAX_DIRECTORY_SCAN_ENTRIES:
                        raise StrictHistoryPilotEvidenceBoundsStop(
                            "strict_history_pilot_directory_scan_entry_bound_exceeded"
                        )
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise StrictHistoryPilotEvidenceStop(
                            "strict_history_pilot_directory_scan_entry_stat_failed"
                        ) from exc
                    if _is_reparse(info):
                        raise StrictHistoryPilotEvidenceStop(
                            "strict_history_pilot_directory_scan_reparse_is_forbidden"
                        )
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(Path(entry.path))
                    elif not stat.S_ISREG(info.st_mode):
                        raise StrictHistoryPilotEvidenceStop(
                            "strict_history_pilot_directory_scan_special_file"
                        )
        except StrictHistoryPilotEvidenceError:
            raise
        except OSError as exc:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_directory_scan_failed"
            ) from exc
    _check_deadline(deadline_ns)
    return tuple(sorted(result.items()))


def _read_bounded_plain_file(
    path: Path,
    *,
    max_bytes: int,
    deadline_ns: int,
    missing_code: str,
    directory_anchor: Path | None = None,
) -> _ReadFile:
    if type(max_bytes) is not int or max_bytes < 0:
        raise StrictHistoryPilotEvidenceContractError(
            "strict_history_pilot_read_bound_is_invalid"
        )
    _check_deadline(deadline_ns)
    if directory_anchor is not None:
        _validate_plain_directory_chain(
            directory_anchor,
            path.parent,
            deadline_ns=deadline_ns,
        )
    try:
        before_path = path.lstat()
    except FileNotFoundError as exc:
        raise StrictHistoryPilotEvidenceStop(missing_code) from exc
    except OSError as exc:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_artifact_stat_failed"
        ) from exc
    if _is_reparse(before_path) or not stat.S_ISREG(before_path.st_mode):
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_artifact_is_not_plain_regular_file"
        )
    if directory_anchor is not None:
        try:
            anchor_info = directory_anchor.lstat()
        except OSError as exc:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_directory_anchor_stat_failed"
            ) from exc
        if before_path.st_dev != anchor_info.st_dev:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_artifact_crosses_volume"
            )
    if before_path.st_nlink != 1:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_artifact_has_external_hardlink_alias"
        )
    if before_path.st_size > max_bytes:
        raise StrictHistoryPilotEvidenceBoundsStop(
            "strict_history_pilot_artifact_byte_bound_exceeded"
        )
    _reject_windows_named_streams(path, deadline_ns=deadline_ns)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_artifact_open_failed"
        ) from exc
    failure: BaseException | None = None
    raw = b""
    opened: os.stat_result | None = None
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _signature(opened) != _signature(before_path)
        ):
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_artifact_changed_before_open"
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            _check_deadline(deadline_ns)
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES, max_bytes + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > max_bytes:
                raise StrictHistoryPilotEvidenceBoundsStop(
                    "strict_history_pilot_artifact_byte_bound_exceeded"
                )
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if _signature(after) != _signature(opened) or len(raw) != after.st_size:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_artifact_changed_while_reading"
            )
    except OSError as exc:
        failure = StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_artifact_read_io_failed"
        )
        failure.__cause__ = exc
    except BaseException as exc:
        failure = exc
    try:
        os.close(descriptor)
    except OSError as exc:
        if failure is None:
            failure = StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_artifact_close_failed"
            )
            failure.__cause__ = exc
    if failure is not None:
        raise failure
    assert opened is not None
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_artifact_post_read_stat_failed"
        ) from exc
    if _signature(after_path) != _signature(opened) or _is_reparse(after_path):
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_artifact_changed_after_read"
        )
    _reject_windows_named_streams(path, deadline_ns=deadline_ns)
    if directory_anchor is not None:
        _validate_plain_directory_chain(
            directory_anchor,
            path.parent,
            deadline_ns=deadline_ns,
        )
    _check_deadline(deadline_ns)
    return _ReadFile(raw=raw, signature=_signature(opened))


def _relative(root: Path, path: Path) -> str:
    try:
        rendered = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise StrictHistoryPilotEvidenceContractError(
            "strict_history_pilot_artifact_escapes_root"
        ) from exc
    return _relative_path(rendered, field="artifact_relative_path")


def _require_clean_report(report: object, *, request_id: str, manifest_hash: str) -> None:
    try:
        states = report.request_states
        clean = (
            report.contract_version == STRICT_HISTORY_V2_RESTART_VERSION
            and report.ready is True
            and len(states) == 1
            and states[0].request_id == request_id
            and states[0].state == "complete_verified"
            and states[0].manifest_hash == manifest_hash
            and states[0].error_code is None
            and report.temp_paths == ()
            and report.unreferenced_attempt_paths == ()
            and report.unreferenced_raw_paths == ()
            and report.alternate_normalized_paths == ()
        )
    except AttributeError as exc:
        raise StrictHistoryPilotEvidenceResidueStop(
            "strict_history_pilot_restart_report_is_invalid"
        ) from exc
    if not clean:
        raise StrictHistoryPilotEvidenceResidueStop(
            "strict_history_pilot_restart_report_is_not_clean_complete_exact"
        )


def _classify_writer_lock(root: Path, *, deadline_ns: int) -> tuple[PilotWriterLockFactV1, _ReadFile | None, Path]:
    path = root.parent / f".{root.name}.strict-history-v2.writer.lock"
    _validate_plain_directory_chain(
        root.parent,
        root.parent,
        deadline_ns=deadline_ns,
    )
    try:
        info = path.lstat()
    except FileNotFoundError:
        return (
            PilotWriterLockFactV1(
                status="absent",
                symbolic_locator="persistent_sibling_writer_lock_outside_shard_root",
                file_sha256=None,
                byte_count=0,
                link_count=0,
            ),
            None,
            path,
        )
    except OSError as exc:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_writer_lock_stat_failed"
        ) from exc
    if _is_reparse(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_writer_lock_is_not_plain_unaliased_file"
        )
    loaded = _read_bounded_plain_file(
        path,
        max_bytes=4 * 1024,
        deadline_ns=deadline_ns,
        missing_code="strict_history_pilot_writer_lock_disappeared",
        directory_anchor=root.parent,
    )
    if loaded.raw != b"0":
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_writer_lock_content_is_unexpected"
        )
    return (
        PilotWriterLockFactV1(
            status="present_plain_regular",
            symbolic_locator="persistent_sibling_writer_lock_outside_shard_root",
            file_sha256=_sha256_bytes(loaded.raw),
            byte_count=len(loaded.raw),
            link_count=loaded.signature[-1],
        ),
        loaded,
        path,
    )


def _logical_reference(
    references: list[PilotLogicalReferenceV1],
    *,
    role: str,
    relative_path: str,
    reference_hash: str,
    loaded: _ReadFile,
    page_ordinal: int = -1,
    attempt_ordinal: int = -1,
) -> None:
    references.append(
        PilotLogicalReferenceV1(
            ordinal=len(references),
            role=role,
            relative_path=relative_path,
            reference_hash=reference_hash,
            file_sha256=_sha256_bytes(loaded.raw),
            byte_count=len(loaded.raw),
            page_ordinal=page_ordinal,
            attempt_ordinal=attempt_ordinal,
        )
    )


def read_strict_history_pilot_evidence_v1(
    *,
    request: HistoryRangeRequestV2,
    artifact_root: str | os.PathLike[str],
    expected_manifest_hash: str,
) -> StrictHistoryPilotEvidenceV1:
    """Return bounded immutable facts for one exact admitted success shard."""

    if type(request) is not HistoryRangeRequestV2:
        raise StrictHistoryPilotEvidenceContractError(
            "strict_history_pilot_request_must_be_exact_v2"
        )
    _digest(expected_manifest_hash, field="expected_manifest_hash")
    if not isinstance(artifact_root, (str, os.PathLike)):
        raise StrictHistoryPilotEvidenceContractError(
            "strict_history_pilot_artifact_root_is_invalid"
        )
    try:
        rendered_root = os.fspath(artifact_root)
    except TypeError as exc:
        raise StrictHistoryPilotEvidenceContractError(
            "strict_history_pilot_artifact_root_is_invalid"
        ) from exc
    if not isinstance(rendered_root, str) or not rendered_root or "\x00" in rendered_root:
        raise StrictHistoryPilotEvidenceContractError(
            "strict_history_pilot_artifact_root_is_invalid"
        )
    supplied_root = Path(os.path.abspath(rendered_root))
    deadline_ns = _deadline_ns()
    _check_deadline(deadline_ns)
    try:
        _check_deadline(deadline_ns)
        store = StrictHistoryArtifactStoreV2(
            supplied_root,
            writable=False,
            storage_profile=request.storage_profile,
        )
        _check_deadline(deadline_ns)
        first_report = store.reconcile_restart(
            (request,),
            expected_manifest_hashes={request.request_id: expected_manifest_hash},
        )
        _check_deadline(deadline_ns)
        _require_clean_report(
            first_report,
            request_id=request.request_id,
            manifest_hash=expected_manifest_hash,
        )
        _check_deadline(deadline_ns)
        shard = store.load_complete_from_disk(
            request,
            expected_manifest_hash=expected_manifest_hash,
        )
        _check_deadline(deadline_ns)
    except StrictHistoryPilotEvidenceError:
        raise
    except (StrictHistoryError, OSError) as exc:
        _raise_public_call_failure(
            exc,
            deadline_ns=deadline_ns,
            code="strict_history_pilot_public_reload_failed",
        )
    root = store.root
    if os.path.normcase(os.path.abspath(os.fspath(root))) != os.path.normcase(
        os.path.abspath(os.fspath(supplied_root))
    ):
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_root_alias_is_forbidden"
        )
    if shard.manifest.manifest_hash != expected_manifest_hash:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_loaded_manifest_hash_mismatch"
        )
    directory_snapshot = _scan_plain_directory_namespace(
        root,
        deadline_ns=deadline_ns,
    )

    logical: list[PilotLogicalReferenceV1] = []
    loaded_by_path: dict[Path, _ReadFile] = {}
    page_accounting: list[PilotPageAccountingV1] = []
    attempt_accounting: list[PilotAttemptAccountingV1] = []

    def read_known(path: Path, *, limit: int, missing: str) -> _ReadFile:
        existing = loaded_by_path.get(path)
        if existing is not None:
            return existing
        loaded = _read_bounded_plain_file(
            path,
            max_bytes=limit,
            deadline_ns=deadline_ns,
            missing_code=missing,
            directory_anchor=root,
        )
        loaded_by_path[path] = loaded
        return loaded

    scope_path = root / "scope.json"
    scope = read_known(
        scope_path,
        limit=min(_MAX_SCOPE_BYTES, request.resource_limits.max_logical_storage_bytes),
        missing="strict_history_pilot_scope_is_missing",
    )
    expected_scope = {
        "contract_version": "mexc_strict_history_shard_scope_v1",
        "history_contract_version": "mexc_strict_history_v2",
        "history_contract_hash": strict_history_v2_contract_hash(),
        "request_id": request.request_id,
        "request": request.as_dict(),
    }
    if scope.raw != _canonical_bytes(expected_scope) + b"\n":
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_scope_identity_mismatch"
        )
    _logical_reference(
        logical,
        role="scope_marker",
        relative_path=_relative(root, scope_path),
        reference_hash=_sha256_bytes(scope.raw),
        loaded=scope,
    )

    attempts: list[HttpAttemptEvidenceV1] = []
    raw_paths: set[Path] = set()
    for page_index, page in enumerate(shard.manifest.page_receipts):
        page_attempts: list[HttpAttemptEvidenceV1] = []
        for attempt_index, attempt_hash in enumerate(page.attempt_receipt_hashes):
            attempt_path = root / "attempts" / f"{attempt_hash}.json"
            attempt_file = read_known(
                attempt_path,
                limit=_MAX_ATTEMPT_RECEIPT_BYTES,
                missing="strict_history_pilot_attempt_receipt_is_missing",
            )
            attempt_payload = _parse_canonical_lf_object(
                attempt_file.raw,
                code="strict_history_pilot_attempt_receipt_is_invalid",
            )
            raw_hash = attempt_payload.get("captured_body_sha256")
            raw_length = attempt_payload.get("captured_body_length")
            _digest(raw_hash, field="attempt_raw_hash")
            if type(raw_length) is not int or raw_length < 0:
                raise StrictHistoryPilotEvidenceStop(
                    "strict_history_pilot_attempt_raw_length_is_invalid"
                )
            if raw_length > request.resource_limits.max_raw_body_bytes_per_attempt:
                raise StrictHistoryPilotEvidenceBoundsStop(
                    "strict_history_pilot_raw_body_per_attempt_bound_exceeded"
                )
            raw_path = root / "raw" / "sha256" / raw_hash[:2] / f"{raw_hash}.bin"
            raw_file = read_known(
                raw_path,
                limit=request.resource_limits.max_raw_body_bytes_per_attempt,
                missing="strict_history_pilot_raw_body_is_missing",
            )
            raw_paths.add(raw_path)
            try:
                attempt = parse_http_attempt_evidence_v1(
                    attempt_payload,
                    page_request=page.page_request,
                    body_bytes=raw_file.raw,
                )
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise StrictHistoryPilotEvidenceStop(
                    "strict_history_pilot_attempt_reconstruction_failed"
                ) from exc
            if (
                attempt.attempt_receipt_hash != attempt_hash
                or attempt.attempt_ordinal != attempt_index
                or attempt.endpoint_contract_hash
                != request.endpoint_contract.contract_hash
                or attempt.resource_limits_hash
                != request.resource_limits.contract_hash
                or attempt.retry_policy_hash != request.retry_policy.contract_hash
                or attempt.transport_contract_hash != request.attempt_contract_hash
                or attempt.request_started_at_us < request.collection_as_of_us
                or attempt.receipt_dict() != attempt_payload
                or _canonical_bytes(attempt.receipt_dict()) + b"\n" != attempt_file.raw
            ):
                raise StrictHistoryPilotEvidenceStop(
                    "strict_history_pilot_attempt_identity_mismatch"
                )
            attempts.append(attempt)
            page_attempts.append(attempt)
            attempt_accounting.append(
                PilotAttemptAccountingV1(
                    page_ordinal=page_index,
                    attempt_ordinal=attempt_index,
                    attempt_receipt_hash=attempt_hash,
                    raw_body_sha256=attempt.captured_body_sha256,
                    raw_body_length=attempt.captured_body_length,
                    request_started_monotonic_us=(
                        attempt.request_started_monotonic_us
                    ),
                    terminal_monotonic_us=attempt.terminal_monotonic_us,
                    elapsed_monotonic_us=attempt.elapsed_monotonic_us,
                )
            )
            _logical_reference(
                logical,
                role="attempt_receipt",
                relative_path=_relative(root, attempt_path),
                reference_hash=attempt_hash,
                loaded=attempt_file,
                page_ordinal=page_index,
                attempt_ordinal=attempt_index,
            )
            _logical_reference(
                logical,
                role="raw_body",
                relative_path=_relative(root, raw_path),
                reference_hash=raw_hash,
                loaded=raw_file,
                page_ordinal=page_index,
                attempt_ordinal=attempt_index,
            )
        final_attempt = page_attempts[-1] if page_attempts else None
        if (
            not isinstance(final_attempt, CompleteHttpAttemptEvidenceV1)
            or final_attempt.body_complete is not True
            or final_attempt.outcome != "complete"
            or final_attempt.http_status != 200
            or final_attempt.attempt_receipt_hash
            != page.attempt_receipt_hashes[-1]
            or final_attempt.request_started_at_us != page.request_started_at_us
            or final_attempt.terminal_at_us != page.received_at_us
            or final_attempt.captured_body_sha256 != page.raw_body_sha256
            or final_attempt.captured_body_length != page.raw_body_length
        ):
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_success_page_final_attempt_mismatch"
            )
        page_accounting.append(
            PilotPageAccountingV1(
                page_ordinal=page_index,
                page_receipt_hash=page.page_receipt_hash,
                row_count=page.row_count,
                attempt_count=len(page_attempts),
            )
        )

    normalized_path = (
        root
        / "normalized"
        / request.request_id
        / f"{shard.manifest.normalized_shard_sha256}.jsonl"
    )
    normalized = read_known(
        normalized_path,
        limit=request.resource_limits.max_logical_storage_bytes,
        missing="strict_history_pilot_normalized_shard_is_missing",
    )
    if (
        _sha256_bytes(normalized.raw) != shard.manifest.normalized_shard_sha256
        or normalized.raw != shard.normalized_jsonl_bytes()
    ):
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_normalized_shard_identity_mismatch"
        )
    _logical_reference(
        logical,
        role="normalized_shard",
        relative_path=_relative(root, normalized_path),
        reference_hash=shard.manifest.normalized_shard_sha256,
        loaded=normalized,
    )

    manifest_path = root / "collections" / request.request_id / "manifest.json"
    manifest = read_known(
        manifest_path,
        limit=min(8 * 1024 * 1024, request.resource_limits.max_logical_storage_bytes),
        missing="strict_history_pilot_manifest_is_missing",
    )
    expected_manifest = {
        **shard.manifest.as_dict(),
        "manifest_hash": expected_manifest_hash,
    }
    if manifest.raw != _canonical_bytes(expected_manifest) + b"\n":
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_manifest_identity_mismatch"
        )
    _logical_reference(
        logical,
        role="manifest",
        relative_path=_relative(root, manifest_path),
        reference_hash=expected_manifest_hash,
        loaded=manifest,
    )

    admission_path = root / "collections" / request.request_id / "admission.json"
    admission = read_known(
        admission_path,
        limit=_MAX_ADMISSION_BYTES,
        missing="strict_history_pilot_admission_is_missing",
    )
    admission_payload = _parse_canonical_lf_object(
        admission.raw,
        code="strict_history_pilot_admission_is_invalid",
    )
    expected_admission_keys = frozenset(
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
    )
    if set(admission_payload) != expected_admission_keys:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_admission_schema_mismatch"
        )
    admission_body = dict(admission_payload)
    admission_hash = admission_body.pop("admission_hash")
    runtime = admission_payload.get("admission_decision_runtime_us")
    admitted_total = admission_payload.get("admitted_total_logical_storage_bytes")
    if (
        admission_payload.get("contract_version") != STRICT_HISTORY_V2_ADMISSION_VERSION
        or admission_payload.get("request_id") != request.request_id
        or admission_payload.get("manifest_hash") != expected_manifest_hash
        or admission_payload.get("history_contract_hash")
        != strict_history_v2_contract_hash()
        or admission_payload.get("storage_profile_hash") != request.storage_profile_hash
        or admission_payload.get("runtime_boundary")
        != "after_full_disk_reload_before_atomic_admission_install"
        or admission_payload.get("graph_logical_storage_bytes")
        != shard.manifest.logical_storage_bytes
        or not isinstance(admission_hash, str)
        or _SHA256_RE.fullmatch(admission_hash) is None
        or _sha256_payload(admission_body) != admission_hash
        or type(runtime) is not int
        or runtime < shard.manifest.collection_runtime_us
        or runtime > request.resource_limits.max_collection_runtime_us
        or type(admitted_total) is not int
        or admitted_total != shard.manifest.logical_storage_bytes + len(admission.raw)
        or admitted_total > request.resource_limits.max_logical_storage_bytes
    ):
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_admission_identity_or_accounting_mismatch"
        )
    _logical_reference(
        logical,
        role="admission_marker",
        relative_path=_relative(root, admission_path),
        reference_hash=admission_hash,
        loaded=admission,
    )

    expected_directories = {
        "",
        "attempts",
        "collections",
        f"collections/{request.request_id}",
        "normalized",
        f"normalized/{request.request_id}",
        "raw",
        "raw/sha256",
        *(
            f"raw/sha256/{path.parent.name}"
            for path in raw_paths
        ),
    }
    observed_directories = {relative for relative, _signature_value in directory_snapshot}
    if observed_directories != expected_directories:
        raise StrictHistoryPilotEvidenceResidueStop(
            "strict_history_pilot_directory_namespace_is_not_exact"
        )

    logical_total = sum(item.byte_count for item in logical)
    if logical_total != admitted_total:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_logical_reference_total_mismatch"
        )
    if len(attempts) != shard.manifest.actual_attempt_count:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_attempt_count_mismatch"
        )
    raw_logical_bytes = sum(item.captured_body_length for item in attempts)
    if raw_logical_bytes != shard.manifest.actual_total_raw_body_bytes:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_raw_logical_bytes_mismatch"
        )
    observed_sleep = 0
    prior_terminal: int | None = None
    for attempt in attempts:
        if prior_terminal is not None:
            gap = attempt.request_started_monotonic_us - prior_terminal
            if gap < 0:
                raise StrictHistoryPilotEvidenceStop(
                    "strict_history_pilot_attempt_monotonic_timing_regressed"
                )
            observed_sleep += gap
        prior_terminal = attempt.terminal_monotonic_us
    if observed_sleep > request.retry_policy.max_total_sleep_us:
        raise StrictHistoryPilotEvidenceBoundsStop(
            "strict_history_pilot_observed_sleep_bound_exceeded"
        )

    counts: dict[str, int] = {}
    roles: dict[str, str] = {}
    for item in logical:
        counts[item.relative_path] = counts.get(item.relative_path, 0) + 1
        prior_role = roles.setdefault(item.relative_path, item.role)
        if prior_role != item.role:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_physical_file_has_multiple_roles"
            )
    physical = tuple(
        PilotPhysicalFileV1(
            relative_path=relative,
            role=roles[relative],
            file_sha256=_sha256_bytes(loaded_by_path[root / PurePosixPath(relative)].raw),
            byte_count=len(loaded_by_path[root / PurePosixPath(relative)].raw),
            logical_reference_count=counts[relative],
        )
        for relative in sorted(counts)
    )

    writer_lock, lock_loaded, lock_path = _classify_writer_lock(
        root,
        deadline_ns=deadline_ns,
    )
    _check_deadline(deadline_ns)
    try:
        second_report = store.reconcile_restart(
            (request,),
            expected_manifest_hashes={request.request_id: expected_manifest_hash},
        )
    except (StrictHistoryError, OSError) as exc:
        _raise_public_call_failure(
            exc,
            deadline_ns=deadline_ns,
            code="strict_history_pilot_final_public_reconciliation_failed",
        )
    _check_deadline(deadline_ns)
    _require_clean_report(
        second_report,
        request_id=request.request_id,
        manifest_hash=expected_manifest_hash,
    )
    final_directory_snapshot = _scan_plain_directory_namespace(
        root,
        deadline_ns=deadline_ns,
    )
    if final_directory_snapshot != directory_snapshot:
        raise StrictHistoryPilotEvidenceStop(
            "strict_history_pilot_directory_namespace_changed_before_stable_point"
        )
    for path, loaded in loaded_by_path.items():
        _check_deadline(deadline_ns)
        _validate_plain_directory_chain(
            root,
            path.parent,
            deadline_ns=deadline_ns,
        )
        try:
            current = path.lstat()
        except OSError as exc:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_artifact_missing_at_stable_point"
            ) from exc
        if _is_reparse(current) or _signature(current) != loaded.signature:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_artifact_changed_before_stable_point"
            )
        _reject_windows_named_streams(path, deadline_ns=deadline_ns)
    _validate_plain_directory_chain(
        root.parent,
        root.parent,
        deadline_ns=deadline_ns,
    )
    if lock_loaded is not None:
        try:
            lock_current = lock_path.lstat()
        except OSError as exc:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_writer_lock_missing_at_stable_point"
            ) from exc
        if _is_reparse(lock_current) or _signature(lock_current) != lock_loaded.signature:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_writer_lock_changed_before_stable_point"
            )
        _reject_windows_named_streams(lock_path, deadline_ns=deadline_ns)
    else:
        try:
            lock_path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_writer_lock_final_stat_failed"
            ) from exc
        else:
            raise StrictHistoryPilotEvidenceStop(
                "strict_history_pilot_writer_lock_appeared_before_stable_point"
            )
    _check_deadline(deadline_ns)

    unique_raw_bytes = sum(
        len(loaded_by_path[path].raw) for path in raw_paths
    )
    result = StrictHistoryPilotEvidenceV1(
        evidence_contract_hash=strict_history_pilot_evidence_contract_hash(),
        request_id=request.request_id,
        manifest_hash=expected_manifest_hash,
        history_contract_hash=strict_history_v2_contract_hash(),
        normalized_shard_sha256=shard.manifest.normalized_shard_sha256,
        page_count=len(shard.manifest.page_receipts),
        row_count=len(shard.rows),
        attempt_count=len(attempts),
        raw_body_reference_count=len(attempts),
        unique_raw_body_count=len(raw_paths),
        actual_total_raw_body_bytes=raw_logical_bytes,
        unique_physical_raw_body_bytes=unique_raw_bytes,
        manifest_collection_runtime_us=shard.manifest.collection_runtime_us,
        admission_full_reload_runtime_us=runtime,
        attempt_elapsed_runtime_us=sum(item.elapsed_monotonic_us for item in attempts),
        observed_monotonic_inter_attempt_sleep_us=observed_sleep,
        admitted_total_logical_storage_bytes=admitted_total,
        unique_physical_referenced_bytes=sum(item.byte_count for item in physical),
        logical_references=tuple(logical),
        physical_files=physical,
        page_accounting=tuple(page_accounting),
        attempt_accounting=tuple(attempt_accounting),
        admission_accounting=PilotAdmissionAccountingV1(
            admission_hash=admission_hash,
            graph_logical_storage_bytes=shard.manifest.logical_storage_bytes,
            admission_marker_bytes=len(admission.raw),
            admitted_total_logical_storage_bytes=admitted_total,
            manifest_collection_runtime_us=shard.manifest.collection_runtime_us,
            admission_full_reload_runtime_us=runtime,
        ),
        restart_contract_version=STRICT_HISTORY_V2_RESTART_VERSION,
        restart_observation_count=2,
        strict_history_namespace_residue_free=True,
        restart_no_residue_proof=PilotRestartNoResidueProofV1(
            request_id=request.request_id,
            manifest_hash=expected_manifest_hash,
            request_state="complete_verified",
            ready=True,
            temp_paths=(),
            unreferenced_attempt_paths=(),
            unreferenced_raw_paths=(),
            alternate_normalized_paths=(),
            observation_count=2,
            restart_contract_version=STRICT_HISTORY_V2_RESTART_VERSION,
        ),
        writer_lock=writer_lock,
        pilot_output_layout_status=PILOT_OUTPUT_LAYOUT_STATUS_UNRESOLVED,
        authority_status=PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE,
    )
    if StrictHistoryPilotEvidenceV1.parse(result.as_dict()) != result:
        raise StrictHistoryPilotEvidenceContractError(
            "strict_history_pilot_evidence_roundtrip_mismatch"
        )
    _check_deadline(deadline_ns)
    return result


def require_strict_history_pilot_compatible_evidence_v1(
    *,
    request: HistoryRangeRequestV2,
    artifact_root: str | os.PathLike[str],
    expected_manifest_hash: str,
) -> StrictHistoryPilotEvidenceV1:
    """Strict adapter that cannot pass until a future layout contract exists."""

    return read_strict_history_pilot_evidence_v1(
        request=request,
        artifact_root=artifact_root,
        expected_manifest_hash=expected_manifest_hash,
    ).require_pilot_compatible()


_CONTRACT_SCHEMA = {
    "contract_version": STRICT_HISTORY_PILOT_EVIDENCE_CONTRACT_VERSION,
    "component_versions": {
        "logical_reference": STRICT_HISTORY_PILOT_LOGICAL_REFERENCE_VERSION,
        "physical_file": STRICT_HISTORY_PILOT_PHYSICAL_FILE_VERSION,
        "writer_lock_fact": STRICT_HISTORY_PILOT_WRITER_LOCK_FACT_VERSION,
        "restart_no_residue_proof": STRICT_HISTORY_PILOT_RESTART_PROOF_VERSION,
        "page_accounting": STRICT_HISTORY_PILOT_PAGE_ACCOUNTING_VERSION,
        "attempt_accounting": STRICT_HISTORY_PILOT_ATTEMPT_ACCOUNTING_VERSION,
        "admission_accounting": STRICT_HISTORY_PILOT_ADMISSION_ACCOUNTING_VERSION,
    },
    "dependency_hashes": {"strict_history_v2": strict_history_v2_contract_hash()},
    "canonicalization": {
        "encoding": "utf8",
        "json_keys": "sorted",
        "separators": [",", ":"],
        "ensure_ascii": False,
        "nonfinite_numbers": "rejected",
        "wire_objects": "exact_key_sets_then_frozen_constructor_validation",
        "identity_hash": "sha256_exact_canonical_object_without_trailing_lf",
    },
    "entry_point": {
        "parameters": ["request", "artifact_root", "expected_manifest_hash"],
        "keyword_only": True,
        "network": False,
        "writes_repairs_or_promotion": False,
        "public_strict_history_calls_first": [
            "StrictHistoryArtifactStoreV2.reconcile_restart",
            "StrictHistoryArtifactStoreV2.load_complete_from_disk",
        ],
        "final_reconciliation": True,
        "deadline": (
            "one_continuation_deadline_starts_before_store_construction_and_is_"
            "checked_before_and_after_each_public_call_and_after_final_"
            "roundtrip"
        ),
        "synchronous_preemption": (
            "not_provided_in_process_successful_filesystem_operations_have_"
            "continuation_checkpoints_expected_stricthistory_or_oserror_"
            "failures_from_frozen_public_store_calls_are_charged_before_"
            "classification_external_process_watchdog_required_for_hard_"
            "preemption"
        ),
    },
    "bounds": {
        "reader_runtime_us": _MAX_READER_RUNTIME_US,
        "read_chunk_bytes": _READ_CHUNK_BYTES,
        "admission_bytes": _MAX_ADMISSION_BYTES,
        "attempt_receipt_bytes": _MAX_ATTEMPT_RECEIPT_BYTES,
        "scope_bytes": _MAX_SCOPE_BYTES,
        "directory_scan_entries_including_files_and_directories": (
            _MAX_DIRECTORY_SCAN_ENTRIES
        ),
        "raw_manifest_normalized_and_counts": "frozen_request_resource_limits",
    },
    "parsing": {
        "canonical_json": "utf8_lf_exact_sorted_no_duplicate_keys_no_nonfinite",
        "attempts": "every_logical_attempt_via_public_parse_http_attempt_evidence_v1",
        "admission": "exact_frozen_key_set_hash_identity_runtime_and_byte_totals",
        "static_hostile": (
            "lstat_open_fstat_read_fstat_close_lstat_exact_identity_plain_regular_"
            "plain_directory_chain_no_reparse_no_external_hardlink_alias_"
            "single_volume_bounded_recursive_all_present_directory_scan_"
            "windows_named_stream_enumeration_on_every_present_directory_and_"
            "every_referenced_file_exact_default_stream_only_then_final_"
            "stable_point_stat"
        ),
    },
    "accounting": {
        "logical_reference_inventory": (
            "frozen_v2_scope_marker_is_charged_then_each_attempt_receipt_and_"
            "raw_reference_then_normalized_manifest_admission_duplicate_raw_"
            "references_charged_per_attempt"
        ),
        "physical_inventory": "unique_relative_path_sorted_with_reference_count",
        "inventory_constructor_validation": (
            "exact_fixed_roles_paths_role_multiplicities_file_facts_and_totals_"
            "attempt_and_raw_pairs_cover_page_ordinals_zero_through_page_count_"
            "minus_one_with_contiguous_attempt_ordinals_from_zero"
        ),
        "page_summary": (
            "ordered_page_ordinal_page_receipt_hash_row_count_attempt_count_"
            "with_exact_top_level_sums_and_logical_coordinate_counts"
        ),
        "attempt_summary": (
            "ordered_page_attempt_coordinates_receipt_and_raw_hashes_raw_length_"
            "monotonic_start_terminal_elapsed_with_exact_logical_binding_and_"
            "derived_raw_elapsed_sleep_totals"
        ),
        "admission_summary": (
            "admission_hash_graph_bytes_marker_bytes_admitted_bytes_manifest_"
            "runtime_full_reload_runtime_with_exact_final_logical_and_top_level_"
            "binding"
        ),
        "runtime_hierarchy": (
            "sum_attempt_elapsed_plus_observed_inter_attempt_gaps_not_greater_"
            "than_manifest_collection_runtime_not_greater_than_admission_full_"
            "reload_runtime"
        ),
        "physical_byte_semantics": (
            "regular_file_default_stream_content_bytes_not_allocated_disk_blocks_"
            "ntfs_named_streams_are_enumerated_and_rejected"
        ),
        "absolute_host_paths": "excluded",
        "host_read_clocks": "excluded",
        "artifact_monotonic_clocks": (
            "attempt_source_summary_start_and_terminal_are_semantically_required_"
            "to_recompute_elapsed_and_inter_attempt_gaps_no_host_read_clock_or_"
            "epoch_clock_is_serialized"
        ),
        "official_document_evidence": "not_fabricated",
        "failure_evidence": "not_applicable_success_shard_only",
        "success_only": (
            "every_page_final_attempt_is_complete_http_200_and_exactly_matches_"
            "the_frozen_page_receipt"
        ),
    },
    "restart": {
        "observations": 2,
        "required_state": "complete_verified",
        "ready": True,
        "all_temp_unreferenced_and_alternate_tuples": "exactly_empty",
        "scope": "strict_history_shard_namespace_only",
        "directory_namespace": (
            "two_identical_bounded_snapshots_and_exact_directory_set_derived_"
            "from_request_and_referenced_raw_prefixes_extra_empty_canonical_"
            "directories_are_residue"
        ),
        "windows_directory_streams": (
            "all_present_shard_directories_and_writer_lock_parent_checked_"
            "named_data_streams_forbidden"
        ),
    },
    "writer_lock": {
        "scope": "persistent_sibling_outside_evidence_graph",
        "classification": ["absent", "present_plain_regular"],
        "charged_or_listed_as_shard_evidence": False,
        "symbolic_locator_only": True,
    },
    "pilot_output_layout": {
        "raw_fact": PILOT_OUTPUT_LAYOUT_STATUS_UNRESOLVED,
        "global_cleanliness_claimed": False,
        "authority_status": PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE,
        "detached_parse_authority": (
            "self_consistency_only_no_provenance_origin_attestation_or_"
            "authentication_coherent_rewrites_can_remain_parseable"
        ),
        "strict_adapter": (
            "typed_stop_until_future_pinned_live_reader_provenance_layout_and_"
            "accounting_contract"
        ),
    },
    "field_sets": {
        "logical_reference": list(PilotLogicalReferenceV1.__dataclass_fields__),
        "physical_file": list(PilotPhysicalFileV1.__dataclass_fields__),
        "writer_lock_fact": list(PilotWriterLockFactV1.__dataclass_fields__),
        "restart_no_residue_proof": list(
            PilotRestartNoResidueProofV1.__dataclass_fields__
        ),
        "page_accounting": list(PilotPageAccountingV1.__dataclass_fields__),
        "attempt_accounting": list(PilotAttemptAccountingV1.__dataclass_fields__),
        "admission_accounting": list(
            PilotAdmissionAccountingV1.__dataclass_fields__
        ),
        "evidence": list(StrictHistoryPilotEvidenceV1.__dataclass_fields__),
    },
    "immutability": {
        "dataclasses": "frozen_and_slotted",
        "inventories": "tuples_in_memory_lists_on_wire",
        "nested_parsers": "exact_and_roundtrip_checked",
        "evidence_self_binding": "pinned_reader_contract_hash",
    },
}


def strict_history_pilot_evidence_contract_hash() -> str:
    digest = _sha256_payload(_CONTRACT_SCHEMA)
    if digest != _PINNED_CONTRACT_HASH:
        raise StrictHistoryPilotEvidenceContractError(
            "strict_history_pilot_evidence_contract_changed_without_version_bump"
        )
    return digest


__all__ = [
    "PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE",
    "PILOT_OUTPUT_LAYOUT_STATUS_UNRESOLVED",
    "PilotAdmissionAccountingV1",
    "PilotAttemptAccountingV1",
    "PilotLogicalReferenceV1",
    "PilotPageAccountingV1",
    "PilotPhysicalFileV1",
    "PilotRestartNoResidueProofV1",
    "PilotWriterLockFactV1",
    "STRICT_HISTORY_PILOT_EVIDENCE_CONTRACT_VERSION",
    "STRICT_HISTORY_PILOT_ADMISSION_ACCOUNTING_VERSION",
    "STRICT_HISTORY_PILOT_ATTEMPT_ACCOUNTING_VERSION",
    "STRICT_HISTORY_PILOT_LOGICAL_REFERENCE_VERSION",
    "STRICT_HISTORY_PILOT_PAGE_ACCOUNTING_VERSION",
    "STRICT_HISTORY_PILOT_PHYSICAL_FILE_VERSION",
    "STRICT_HISTORY_PILOT_RESTART_PROOF_VERSION",
    "STRICT_HISTORY_PILOT_WRITER_LOCK_FACT_VERSION",
    "StrictHistoryPilotEvidenceBoundsStop",
    "StrictHistoryPilotEvidenceContractError",
    "StrictHistoryPilotEvidenceError",
    "StrictHistoryPilotEvidenceLayoutStop",
    "StrictHistoryPilotEvidenceResidueStop",
    "StrictHistoryPilotEvidenceStop",
    "StrictHistoryPilotEvidenceV1",
    "read_strict_history_pilot_evidence_v1",
    "require_strict_history_pilot_compatible_evidence_v1",
    "strict_history_pilot_evidence_contract_hash",
]
