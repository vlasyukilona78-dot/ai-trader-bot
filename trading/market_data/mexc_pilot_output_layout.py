"""Pure global output-layout and physical-accounting contract for the MEXC pilot.

This module does not read or write the filesystem.  It composes immutable
objects produced by the frozen pilot store and strict-history evidence reader.
The physical inventory is authoritative only under the frozen local-store
scan boundary: that scan rejects reparse points, non-regular files, link counts
other than one, and duplicate file identities.  A caller-constructed lookalike
``PilotInventoryScanV1`` is not provenance or cryptographic attestation.

Per-stage evidence reads and the later global store scan are not an atomic
cross-tree snapshot.  The assessment therefore applies only while writers are
static or cooperating with the pilot locks.  It deliberately makes no claim
against a hostile writer racing between those observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any

from trading.market_data.mexc_pilot_local_coordinator import (
    PILOT_LOCAL_COORDINATOR_CONTRACT_VERSION,
    pilot_local_coordinator_contract_hash,
)
from trading.market_data.mexc_pilot_local_executor import (
    PILOT_LOCAL_EXECUTOR_CONTRACT_VERSION,
    pilot_local_executor_contract_hash,
)
from trading.market_data.mexc_pilot_local_store import (
    PILOT_LOCAL_INVENTORY_VERSION,
    PILOT_LOCAL_STORE_CONTRACT_VERSION,
    PilotInventoryEntryV1,
    PilotInventoryScanV1,
    mexc_pilot_local_store_contract_hash,
)
from trading.market_data.mexc_pilot_run import (
    PILOT_RUN_CONTRACT_VERSION,
    MexcPublicQaPilotRunManifestV1,
    PilotDiskPreflightReceiptV1,
    PilotRunStateV1,
    U5PublicPilotAuthorizationReceiptV1,
    pilot_run_contract_hash,
)
from trading.market_data.strict_history import _frozen_contract_hash
from trading.market_data.strict_history_pilot_evidence import (
    PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE,
    PILOT_OUTPUT_LAYOUT_STATUS_UNRESOLVED,
    STRICT_HISTORY_PILOT_EVIDENCE_CONTRACT_VERSION,
    StrictHistoryPilotEvidenceV1,
    strict_history_pilot_evidence_contract_hash,
)


PILOT_OUTPUT_LAYOUT_CONTRACT_VERSION = "mexc_public_qa_pilot_output_layout_v1"
PILOT_OUTPUT_LOCATOR_PLAN_VERSION = "mexc_public_qa_pilot_output_locator_plan_v1"
PILOT_OUTPUT_PHYSICAL_ENTRY_VERSION = "mexc_public_qa_pilot_output_physical_entry_v1"
PILOT_OUTPUT_STAGE_ACCOUNTING_VERSION = (
    "mexc_public_qa_pilot_output_stage_accounting_v1"
)
PILOT_OFFICIAL_DOCUMENT_PLACEHOLDER_VERSION = (
    "mexc_public_qa_pilot_official_document_placeholder_v1"
)
PILOT_OFFICIAL_DOCUMENT_PLACEHOLDER_FILE_VERSION = (
    "mexc_public_qa_pilot_official_document_placeholder_file_v1"
)
PILOT_OUTPUT_LAYOUT_PLAN_VERSION = "mexc_public_qa_pilot_output_layout_plan_v1"
PILOT_OUTPUT_READINESS_ASSESSMENT_VERSION = (
    "mexc_public_qa_pilot_output_readiness_assessment_v1"
)

_PINNED_CONTRACT_HASH = "cb19e6a53d122139ec3a76b4d54c67c04a31da9550db9ca8c186496c6bb8e934"
_SHA256_ZERO_BYTE = hashlib.sha256(b"0").hexdigest()
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
)
_TERMINAL_BLOCKERS = tuple(
    sorted(
        (
            "fresh_global_inventory_scan_provenance_unbound",
            "global_directory_namespace_provenance_unbound",
            "frozen_final_anchor_excludes_infrastructure_locks",
            "official_document_bundle_root_not_in_frozen_preflight",
            "official_document_evidence_schema_unbound",
            "strict_history_live_reader_provenance_unbound",
            "terminal_publisher_unbound",
            "u5_inventory_and_traversal_caps_absent_from_frozen_authorization",
        )
    )
)
_SNAPSHOT_BOUNDARY = (
    "store_scan_stable_point_only_no_cross_evidence_atomic_snapshot"
)
_WRITER_BOUNDARY = "static_or_cooperating_writer_required"
_HARDLINK_ASSURANCE = (
    "frozen_local_store_scan_rejects_nlink_not_one_and_duplicate_file_identity"
)


class PilotOutputLayoutError(RuntimeError):
    pass


class PilotOutputLayoutContractError(PilotOutputLayoutError):
    pass


class PilotOutputLayoutBudgetStop(PilotOutputLayoutError):
    pass


class PilotOutputLayoutInventoryStop(PilotOutputLayoutError):
    pass


class PilotOutputLayoutTerminalStop(PilotOutputLayoutError):
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
        raise PilotOutputLayoutContractError(
            "pilot_output_layout_payload_is_not_canonical_json"
        ) from exc


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PilotOutputLayoutContractError(f"{field}_is_not_sha256")
    return value


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PilotOutputLayoutContractError(f"{field}_is_invalid")
    return value


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise PilotOutputLayoutContractError(f"{field}_is_invalid")
    return value


def _role(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ROLE_RE.fullmatch(value) is None:
        raise PilotOutputLayoutContractError(f"{field}_is_invalid")
    return value


def _relative_path(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
            for character in value
        )
    ):
        raise PilotOutputLayoutContractError(f"{field}_is_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PilotOutputLayoutContractError(f"{field}_is_invalid")
    if path.as_posix() != value:
        raise PilotOutputLayoutContractError(f"{field}_is_not_canonical")
    for part in path.parts:
        if (
            len(part) > 255
            or any(character in '<>:"|?*' for character in part)
            or part.endswith((".", " "))
            or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED
        ):
            raise PilotOutputLayoutContractError(f"{field}_is_not_windows_safe")
    lowered = tuple(part.casefold() for part in path.parts)
    if any(
        lowered[index : index + 2] == ("data", "history")
        for index in range(len(lowered) - 1)
    ):
        raise PilotOutputLayoutContractError(f"{field}_uses_legacy_history")
    return value


def _exact_mapping(payload: object, expected: frozenset[str], *, code: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or frozenset(payload) != expected:
        raise PilotOutputLayoutContractError(code)
    return dict(payload)


def _inventory_scan_as_dict(scan: PilotInventoryScanV1) -> dict[str, object]:
    return {
        "manifest_hash": scan.manifest_hash,
        "entries": [item.as_dict() for item in scan.entries],
        "total_bytes": scan.total_bytes,
        "scanned_at_us": scan.scanned_at_us,
        "scanned_monotonic_us": scan.scanned_monotonic_us,
        "clock_domain_id": scan.clock_domain_id,
        "contract_version": scan.contract_version,
    }


def _parse_inventory_scan(payload: object) -> PilotInventoryScanV1:
    values = _exact_mapping(
        payload,
        frozenset(PilotInventoryScanV1.__dataclass_fields__),
        code="pilot_output_inventory_scan_schema_mismatch",
    )
    raw_entries = values.get("entries")
    if not isinstance(raw_entries, list):
        raise PilotOutputLayoutContractError(
            "pilot_output_inventory_scan_entries_wire_type_mismatch"
        )
    try:
        values["entries"] = tuple(
            PilotInventoryEntryV1(
                **_exact_mapping(
                    item,
                    frozenset(PilotInventoryEntryV1.__dataclass_fields__),
                    code="pilot_output_inventory_scan_entry_schema_mismatch",
                )
            )
            for item in raw_entries
        )
        return PilotInventoryScanV1(**values)
    except TypeError as exc:
        raise PilotOutputLayoutContractError(
            "pilot_output_inventory_scan_reconstruction_failed"
        ) from exc


@lru_cache(maxsize=4096)
def _path_parts(value: str) -> tuple[str, ...]:
    # The pairwise overlap checks below are quadratic in the number of declared
    # paths, so the same handful of strings gets reparsed thousands of times.
    return tuple(part.casefold() for part in PurePosixPath(value).parts)


def _paths_overlap(left: str, right: str) -> bool:
    left_parts = _path_parts(left)
    right_parts = _path_parts(right)
    shorter = min(len(left_parts), len(right_parts))
    return left_parts[:shorter] == right_parts[:shorter]


def derive_persistent_writer_lock_locator_v1(relative_artifact_root: str) -> str:
    """Derive ``parent/.<leaf>.strict-history-v2.writer.lock`` exactly."""

    root = PurePosixPath(
        _relative_path(relative_artifact_root, field="pilot_output_artifact_root")
    )
    lock_name = f".{root.name}.strict-history-v2.writer.lock"
    parent = root.parent
    locator = lock_name if parent == PurePosixPath(".") else (parent / lock_name).as_posix()
    return _relative_path(locator, field="pilot_output_writer_lock_locator")


def derive_official_bundle_root_v1(endpoint_plan_hash: str) -> str:
    """Return the unresolved, external-to-strict-history bundle root."""

    return _relative_path(
        f"endpoint-evidence/{_digest(endpoint_plan_hash, field='pilot_endpoint_plan_hash')}/official",
        field="pilot_official_document_bundle_root",
    )


def derive_official_bundle_locators_v1(endpoint_plan_hash: str) -> tuple[str, str, str]:
    root = derive_official_bundle_root_v1(endpoint_plan_hash)
    return (
        f"{root}/attempt-000.body.bin",
        f"{root}/attempt-000.receipt.json",
        f"{root}/evidence.json",
    )


@dataclass(frozen=True, slots=True)
class PilotOutputLocatorPlanV1:
    stage: str
    ordinal: int
    request_id: str
    plan_binding_hash: str
    relative_artifact_root: str
    writer_lock_relative_path: str
    frozen_remaining_storage_reservation: int
    extra_remaining_lock_entries: int
    extra_remaining_lock_bytes: int
    contract_version: str = PILOT_OUTPUT_LOCATOR_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_OUTPUT_LOCATOR_PLAN_VERSION:
            raise PilotOutputLayoutContractError("pilot_output_locator_plan_version_mismatch")
        if self.stage not in {"endpoint_verification", "shard_acquisition"}:
            raise PilotOutputLayoutContractError("pilot_output_locator_stage_is_invalid")
        if type(self.ordinal) is not int or (
            (self.stage == "endpoint_verification" and self.ordinal != -1)
            or (self.stage == "shard_acquisition" and self.ordinal < 0)
        ):
            raise PilotOutputLayoutContractError("pilot_output_locator_ordinal_is_invalid")
        _digest(self.request_id, field="pilot_output_locator_request_id")
        _digest(self.plan_binding_hash, field="pilot_output_locator_plan_binding_hash")
        root = _relative_path(self.relative_artifact_root, field="pilot_output_locator_root")
        lock = _relative_path(
            self.writer_lock_relative_path,
            field="pilot_output_locator_writer_lock",
        )
        if lock != derive_persistent_writer_lock_locator_v1(root):
            raise PilotOutputLayoutContractError("pilot_output_writer_lock_derivation_mismatch")
        _strict_int(
            self.frozen_remaining_storage_reservation,
            field="pilot_output_frozen_remaining_storage_reservation",
            minimum=1,
        )
        entries = _strict_int(
            self.extra_remaining_lock_entries,
            field="pilot_output_extra_remaining_lock_entries",
            minimum=1,
        )
        lock_bytes = _strict_int(
            self.extra_remaining_lock_bytes,
            field="pilot_output_extra_remaining_lock_bytes",
            minimum=1,
        )
        if lock_bytes != entries:
            raise PilotOutputLayoutContractError("pilot_output_extra_lock_byte_count_mismatch")

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "PilotOutputLocatorPlanV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_output_locator_plan_schema_mismatch",
        )
        try:
            return cls(**values)
        except TypeError as exc:
            raise PilotOutputLayoutContractError(
                "pilot_output_locator_plan_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class PilotOfficialDocumentPlaceholderFileV1:
    role: str
    relative_path: str
    artifact_sha256: str
    byte_count: int
    contract_version: str = PILOT_OFFICIAL_DOCUMENT_PLACEHOLDER_FILE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_OFFICIAL_DOCUMENT_PLACEHOLDER_FILE_VERSION:
            raise PilotOutputLayoutContractError(
                "pilot_official_document_placeholder_file_version_mismatch"
            )
        if self.role not in {"raw_body", "attempt_receipt", "evidence"}:
            raise PilotOutputLayoutContractError(
                "pilot_official_document_placeholder_file_role_mismatch"
            )
        _relative_path(
            self.relative_path,
            field="pilot_official_document_placeholder_file_path",
        )
        _digest(
            self.artifact_sha256,
            field="pilot_official_document_placeholder_file_hash",
        )
        _strict_int(
            self.byte_count,
            field="pilot_official_document_placeholder_file_bytes",
            minimum=1,
        )

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(
        cls, payload: object
    ) -> "PilotOfficialDocumentPlaceholderFileV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_official_document_placeholder_file_schema_mismatch",
        )
        try:
            return cls(**values)
        except TypeError as exc:
            raise PilotOutputLayoutContractError(
                "pilot_official_document_placeholder_file_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class PilotOfficialDocumentPlaceholderV1:
    endpoint_plan_hash: str
    files: tuple[PilotOfficialDocumentPlaceholderFileV1, ...]
    schema_status: str = "unresolved_official_document_evidence_schema"
    authority_status: str = "non_authoritative_physical_metadata_only"
    contract_version: str = PILOT_OFFICIAL_DOCUMENT_PLACEHOLDER_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_OFFICIAL_DOCUMENT_PLACEHOLDER_VERSION:
            raise PilotOutputLayoutContractError(
                "pilot_official_document_placeholder_version_mismatch"
            )
        _digest(self.endpoint_plan_hash, field="pilot_official_document_endpoint_plan_hash")
        if not isinstance(self.files, tuple) or not all(
            type(item) is PilotOfficialDocumentPlaceholderFileV1 for item in self.files
        ):
            raise PilotOutputLayoutContractError(
                "pilot_official_document_placeholder_files_are_not_exact_immutable"
            )
        if tuple(item.role for item in self.files) != (
            "raw_body",
            "attempt_receipt",
            "evidence",
        ):
            raise PilotOutputLayoutContractError(
                "pilot_official_document_placeholder_bundle_shape_mismatch"
            )
        expected_paths = derive_official_bundle_locators_v1(self.endpoint_plan_hash)
        if tuple(item.relative_path for item in self.files) != expected_paths:
            raise PilotOutputLayoutContractError(
                "pilot_official_document_placeholder_bundle_locator_mismatch"
            )
        if self.schema_status != "unresolved_official_document_evidence_schema":
            raise PilotOutputLayoutContractError(
                "pilot_official_document_placeholder_schema_status_mismatch"
            )
        if self.authority_status != "non_authoritative_physical_metadata_only":
            raise PilotOutputLayoutContractError(
                "pilot_official_document_placeholder_authority_status_mismatch"
            )

    def as_dict(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["files"] = [item.as_dict() for item in self.files]
        return result

    @classmethod
    def from_dict(cls, payload: object) -> "PilotOfficialDocumentPlaceholderV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_official_document_placeholder_schema_mismatch",
        )
        raw_files = values.get("files")
        if not isinstance(raw_files, list):
            raise PilotOutputLayoutContractError(
                "pilot_official_document_placeholder_files_wire_type_mismatch"
            )
        values["files"] = tuple(
            PilotOfficialDocumentPlaceholderFileV1.from_dict(item)
            for item in raw_files
        )
        try:
            return cls(**values)
        except TypeError as exc:
            raise PilotOutputLayoutContractError(
                "pilot_official_document_placeholder_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class PilotOutputPhysicalEntryV1:
    relative_path: str
    artifact_sha256: str
    byte_count: int
    source: str
    role: str
    stage: str
    ordinal: int
    logical_reference_count: int
    contract_version: str = PILOT_OUTPUT_PHYSICAL_ENTRY_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_OUTPUT_PHYSICAL_ENTRY_VERSION:
            raise PilotOutputLayoutContractError("pilot_output_physical_entry_version_mismatch")
        _relative_path(self.relative_path, field="pilot_output_physical_entry_path")
        _digest(self.artifact_sha256, field="pilot_output_physical_entry_hash")
        _strict_int(self.byte_count, field="pilot_output_physical_entry_bytes")
        if self.source not in {
            "run_control",
            "strict_history_physical",
            "infrastructure_writer_lock",
            "official_document_placeholder",
        }:
            raise PilotOutputLayoutContractError("pilot_output_physical_entry_source_is_invalid")
        _role(self.role, field="pilot_output_physical_entry_role")
        if self.stage not in {"run_control", "endpoint_verification", "shard_acquisition"}:
            raise PilotOutputLayoutContractError("pilot_output_physical_entry_stage_is_invalid")
        if type(self.ordinal) is not int or (
            (self.stage == "run_control" and self.ordinal != -2)
            or (self.stage == "endpoint_verification" and self.ordinal != -1)
            or (self.stage == "shard_acquisition" and self.ordinal < 0)
        ):
            raise PilotOutputLayoutContractError("pilot_output_physical_entry_ordinal_is_invalid")
        references = _strict_int(
            self.logical_reference_count,
            field="pilot_output_physical_entry_reference_count",
        )
        if self.source == "strict_history_physical" and references < 1:
            raise PilotOutputLayoutContractError(
                "pilot_output_history_physical_entry_has_no_logical_reference"
            )
        if self.source != "strict_history_physical" and references != 0:
            raise PilotOutputLayoutContractError(
                "pilot_output_nonhistory_entry_has_logical_reference"
            )
        if (
            (self.source == "run_control" and self.stage != "run_control")
            or (self.source != "run_control" and self.stage == "run_control")
            or (
                self.source == "official_document_placeholder"
                and self.stage != "endpoint_verification"
            )
        ):
            raise PilotOutputLayoutContractError(
                "pilot_output_physical_entry_source_stage_mismatch"
            )
        if self.source == "official_document_placeholder" and self.role not in {
            "official_document_raw_body_unresolved",
            "official_document_attempt_receipt_unresolved",
            "official_document_evidence_unresolved",
        }:
            raise PilotOutputLayoutContractError(
                "pilot_output_official_physical_entry_role_mismatch"
            )
        if self.source == "official_document_placeholder" and self.byte_count < 1:
            raise PilotOutputLayoutContractError(
                "pilot_output_official_physical_entry_is_empty"
            )
        if self.source == "infrastructure_writer_lock" and (
            self.role != "persistent_writer_lock"
            or self.artifact_sha256 != _SHA256_ZERO_BYTE
            or self.byte_count != 1
        ):
            raise PilotOutputLayoutContractError(
                "pilot_output_writer_lock_physical_entry_mismatch"
            )

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "PilotOutputPhysicalEntryV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_output_physical_entry_schema_mismatch",
        )
        try:
            return cls(**values)
        except TypeError as exc:
            raise PilotOutputLayoutContractError(
                "pilot_output_physical_entry_reconstruction_failed"
            ) from exc


def _require_canonical_entries(
    entries: tuple[PilotOutputPhysicalEntryV1, ...], *, field: str
) -> None:
    if not isinstance(entries, tuple) or not all(
        type(item) is PilotOutputPhysicalEntryV1 for item in entries
    ):
        raise PilotOutputLayoutContractError(f"{field}_is_not_immutable_exact_entries")
    paths = tuple(item.relative_path for item in entries)
    if paths != tuple(sorted(paths)) or len({path.casefold() for path in paths}) != len(paths):
        raise PilotOutputLayoutContractError(f"{field}_has_duplicate_or_noncanonical_paths")


@dataclass(frozen=True, slots=True)
class PilotStageOutputAccountingV1:
    pilot_manifest_hash: str
    locator_plan: PilotOutputLocatorPlanV1
    strict_history_evidence: StrictHistoryPilotEvidenceV1
    official_document_placeholder: PilotOfficialDocumentPlaceholderV1 | None
    stage: str
    ordinal: int
    request_id: str
    plan_binding_hash: str
    relative_artifact_root: str
    history_manifest_hash: str
    strict_history_evidence_hash: str
    logical_inventory_hash: str
    logical_reference_entries: int
    logical_reference_bytes: int
    unique_history_physical_inventory_hash: str
    unique_history_physical_entries: int
    unique_history_physical_bytes: int
    writer_lock_relative_path: str
    expected_physical_entries: tuple[PilotOutputPhysicalEntryV1, ...]
    expected_physical_inventory_hash: str
    dependent_receipt_inventory_hash: str
    dependent_receipt_inventory_entries: int
    dependent_receipt_inventory_bytes: int
    contract_version: str = PILOT_OUTPUT_STAGE_ACCOUNTING_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_OUTPUT_STAGE_ACCOUNTING_VERSION:
            raise PilotOutputLayoutContractError("pilot_output_stage_accounting_version_mismatch")
        if type(self.locator_plan) is not PilotOutputLocatorPlanV1:
            raise PilotOutputLayoutContractError("pilot_output_stage_locator_source_is_invalid")
        if type(self.strict_history_evidence) is not StrictHistoryPilotEvidenceV1:
            raise PilotOutputLayoutContractError("pilot_output_stage_evidence_source_is_invalid")
        if self.official_document_placeholder is not None and type(
            self.official_document_placeholder
        ) is not PilotOfficialDocumentPlaceholderV1:
            raise PilotOutputLayoutContractError("pilot_output_stage_placeholder_source_is_invalid")
        for field in (
            "pilot_manifest_hash",
            "request_id",
            "plan_binding_hash",
            "history_manifest_hash",
            "strict_history_evidence_hash",
            "logical_inventory_hash",
            "unique_history_physical_inventory_hash",
            "expected_physical_inventory_hash",
            "dependent_receipt_inventory_hash",
        ):
            _digest(getattr(self, field), field=f"pilot_output_stage_{field}")
        if self.stage not in {"endpoint_verification", "shard_acquisition"}:
            raise PilotOutputLayoutContractError("pilot_output_stage_accounting_stage_is_invalid")
        if type(self.ordinal) is not int or (
            (self.stage == "endpoint_verification" and self.ordinal != -1)
            or (self.stage == "shard_acquisition" and self.ordinal < 0)
        ):
            raise PilotOutputLayoutContractError("pilot_output_stage_accounting_ordinal_is_invalid")
        source = self.locator_plan
        evidence = self.strict_history_evidence
        if (
            self.stage != source.stage
            or self.ordinal != source.ordinal
            or self.request_id != source.request_id
            or self.plan_binding_hash != source.plan_binding_hash
            or self.relative_artifact_root != source.relative_artifact_root
            or self.writer_lock_relative_path != source.writer_lock_relative_path
            or self.history_manifest_hash != evidence.manifest_hash
            or self.strict_history_evidence_hash != evidence.evidence_hash
            or evidence.request_id != source.request_id
        ):
            raise PilotOutputLayoutContractError("pilot_output_stage_source_projection_mismatch")
        root = _relative_path(self.relative_artifact_root, field="pilot_output_stage_root")
        lock = _relative_path(self.writer_lock_relative_path, field="pilot_output_stage_lock")
        if lock != derive_persistent_writer_lock_locator_v1(root):
            raise PilotOutputLayoutContractError("pilot_output_stage_lock_derivation_mismatch")
        for field in (
            "logical_reference_entries",
            "logical_reference_bytes",
            "unique_history_physical_entries",
            "unique_history_physical_bytes",
            "dependent_receipt_inventory_entries",
            "dependent_receipt_inventory_bytes",
        ):
            _strict_int(getattr(self, field), field=f"pilot_output_stage_{field}", minimum=1)
        _require_canonical_entries(
            self.expected_physical_entries,
            field="pilot_output_stage_expected_physical_entries",
        )
        history = tuple(
            item for item in self.expected_physical_entries
            if item.source == "strict_history_physical"
        )
        locks = tuple(
            item for item in self.expected_physical_entries
            if item.source == "infrastructure_writer_lock"
        )
        placeholders = tuple(
            item for item in self.expected_physical_entries
            if item.source == "official_document_placeholder"
        )
        if (
            len(history) != self.unique_history_physical_entries
            or sum(item.byte_count for item in history) != self.unique_history_physical_bytes
            or len(locks) != 1
            or locks[0].relative_path != lock
            or locks[0].artifact_sha256 != _SHA256_ZERO_BYTE
            or locks[0].byte_count != 1
            or len(placeholders) != (3 if self.stage == "endpoint_verification" else 0)
        ):
            raise PilotOutputLayoutContractError("pilot_output_stage_physical_shape_mismatch")
        derived = _derive_stage_components(
            pilot_manifest_hash=self.pilot_manifest_hash,
            locator_plan=self.locator_plan,
            evidence=self.strict_history_evidence,
            official_document_placeholder=self.official_document_placeholder,
        )
        (
            derived_logical_hash,
            derived_history_entries,
            derived_history_hash,
            derived_expected_entries,
            derived_expected_hash,
        ) = derived
        if (
            self.logical_inventory_hash != derived_logical_hash
            or self.logical_reference_entries != len(evidence.logical_references)
            or self.logical_reference_bytes != evidence.admitted_total_logical_storage_bytes
            or self.unique_history_physical_inventory_hash != derived_history_hash
            or self.unique_history_physical_entries != len(derived_history_entries)
            or self.unique_history_physical_bytes != evidence.unique_physical_referenced_bytes
            or self.expected_physical_entries != derived_expected_entries
            or self.expected_physical_inventory_hash != derived_expected_hash
        ):
            raise PilotOutputLayoutContractError("pilot_output_stage_derived_accounting_mismatch")
        if self.stage == "endpoint_verification":
            subject = tuple(
                item
                for item in self.expected_physical_entries
                if item.source
                in {"strict_history_physical", "official_document_placeholder"}
            )
            receipt_hash = _sha256_payload(
                {
                    "domain": "mexc_public_qa_pilot_endpoint_subject_inventory_v1",
                    "pilot_manifest_hash": self.pilot_manifest_hash,
                    "plan_binding_hash": self.plan_binding_hash,
                    "entries": [item.as_dict() for item in subject],
                }
            )
            receipt_entries = len(subject)
            receipt_bytes = sum(item.byte_count for item in subject)
        else:
            receipt_hash = self.logical_inventory_hash
            receipt_entries = self.logical_reference_entries
            receipt_bytes = self.logical_reference_bytes
        if (
            self.dependent_receipt_inventory_hash != receipt_hash
            or self.dependent_receipt_inventory_entries != receipt_entries
            or self.dependent_receipt_inventory_bytes != receipt_bytes
        ):
            raise PilotOutputLayoutContractError(
                "pilot_output_stage_dependent_receipt_projection_mismatch"
            )
        physical_payload = [item.as_dict() for item in self.expected_physical_entries]
        if self.expected_physical_inventory_hash != _sha256_payload(
            {
                "domain": PILOT_OUTPUT_STAGE_ACCOUNTING_VERSION,
                "pilot_manifest_hash": self.pilot_manifest_hash,
                "stage": self.stage,
                "ordinal": self.ordinal,
                "entries": physical_payload,
            }
        ):
            raise PilotOutputLayoutContractError("pilot_output_stage_physical_hash_mismatch")

    def as_dict(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["locator_plan"] = self.locator_plan.as_dict()
        result["strict_history_evidence"] = self.strict_history_evidence.as_dict()
        result["official_document_placeholder"] = (
            None
            if self.official_document_placeholder is None
            else self.official_document_placeholder.as_dict()
        )
        result["expected_physical_entries"] = [
            item.as_dict() for item in self.expected_physical_entries
        ]
        return result

    @classmethod
    def from_dict(cls, payload: object) -> "PilotStageOutputAccountingV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_output_stage_accounting_schema_mismatch",
        )
        raw_entries = values.get("expected_physical_entries")
        if not isinstance(raw_entries, list):
            raise PilotOutputLayoutContractError(
                "pilot_output_stage_accounting_entries_wire_type_mismatch"
            )
        values["expected_physical_entries"] = tuple(
            PilotOutputPhysicalEntryV1.from_dict(item) for item in raw_entries
        )
        values["locator_plan"] = PilotOutputLocatorPlanV1.from_dict(
            values.get("locator_plan")
        )
        values["strict_history_evidence"] = StrictHistoryPilotEvidenceV1.parse(
            values.get("strict_history_evidence")
        )
        raw_placeholder = values.get("official_document_placeholder")
        values["official_document_placeholder"] = (
            None
            if raw_placeholder is None
            else PilotOfficialDocumentPlaceholderV1.from_dict(raw_placeholder)
        )
        try:
            return cls(**values)
        except TypeError as exc:
            raise PilotOutputLayoutContractError(
                "pilot_output_stage_accounting_reconstruction_failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class PilotOutputLayoutPlanV1:
    manifest: MexcPublicQaPilotRunManifestV1
    manifest_hash: str
    locator_plans: tuple[PilotOutputLocatorPlanV1, ...]
    frozen_planned_inventory_entries: int
    infrastructure_lock_entries: int
    required_max_inventory_entries: int
    inventory_entry_headroom: int
    maximum_scan_directory_entries: int
    required_scan_traversal_entries: int
    scan_traversal_entry_headroom: int
    frozen_planned_output_bytes: int
    infrastructure_lock_bytes: int
    required_max_total_output_bytes: int
    output_byte_headroom: int
    required_free_disk_bytes_after_reservation: int
    contract_version: str = PILOT_OUTPUT_LAYOUT_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_OUTPUT_LAYOUT_PLAN_VERSION:
            raise PilotOutputLayoutContractError("pilot_output_layout_plan_version_mismatch")
        if type(self.manifest) is not MexcPublicQaPilotRunManifestV1:
            raise PilotOutputLayoutContractError("pilot_output_layout_plan_manifest_type_mismatch")
        _digest(self.manifest_hash, field="pilot_output_layout_plan_manifest_hash")
        if self.manifest_hash != self.manifest.manifest_hash:
            raise PilotOutputLayoutContractError("pilot_output_layout_plan_manifest_hash_mismatch")
        if self.manifest.endpoint_verification.max_network_attempts != 2:
            raise PilotOutputLayoutContractError(
                "pilot_output_layout_v1_requires_exactly_two_endpoint_attempts"
            )
        if not isinstance(self.locator_plans, tuple) or not self.locator_plans or not all(
            type(item) is PilotOutputLocatorPlanV1 for item in self.locator_plans
        ):
            raise PilotOutputLayoutContractError("pilot_output_layout_locator_plans_are_invalid")
        if tuple(item.ordinal for item in self.locator_plans) != (
            -1,
            *range(len(self.locator_plans) - 1),
        ):
            raise PilotOutputLayoutContractError("pilot_output_layout_locator_plan_order_mismatch")
        if self.locator_plans != _derive_locator_plans(self.manifest):
            raise PilotOutputLayoutContractError("pilot_output_layout_locator_plan_derivation_mismatch")
        roots = tuple(item.relative_artifact_root for item in self.locator_plans)
        locks = tuple(item.writer_lock_relative_path for item in self.locator_plans)
        reserved_namespaces = (*roots, *locks, "run-control", derive_official_bundle_root_v1(self.manifest.endpoint_verification.plan_hash))
        for index, left in enumerate(reserved_namespaces):
            for right in reserved_namespaces[index + 1 :]:
                if _paths_overlap(left, right):
                    raise PilotOutputLayoutContractError(
                        "pilot_output_layout_root_lock_or_namespace_collision"
                    )
        for field in (
            "frozen_planned_inventory_entries",
            "infrastructure_lock_entries",
            "required_max_inventory_entries",
            "inventory_entry_headroom",
            "maximum_scan_directory_entries",
            "required_scan_traversal_entries",
            "scan_traversal_entry_headroom",
            "frozen_planned_output_bytes",
            "infrastructure_lock_bytes",
            "required_max_total_output_bytes",
            "output_byte_headroom",
            "required_free_disk_bytes_after_reservation",
        ):
            _strict_int(getattr(self, field), field=f"pilot_output_layout_plan_{field}")
        reservations = self.manifest.planned_reservations
        expected_file_reservation = reservations["inventory_entries"] + 1
        expected_directories = _maximum_scan_directory_entries(self.manifest)
        if (
            self.frozen_planned_inventory_entries != reservations["inventory_entries"]
            or self.infrastructure_lock_entries != len(self.locator_plans)
            # Frozen shard reservations each contain one spare entry; those
            # N credits cover N of N+1 sibling locks, leaving one global entry.
            or self.required_max_inventory_entries != expected_file_reservation
            or self.inventory_entry_headroom
            != self.manifest.budgets.max_inventory_entries - expected_file_reservation
            or self.maximum_scan_directory_entries != expected_directories
            or self.required_scan_traversal_entries
            != expected_file_reservation + expected_directories
            or self.scan_traversal_entry_headroom
            != self.manifest.budgets.max_inventory_entries
            - self.required_scan_traversal_entries
            or self.frozen_planned_output_bytes != reservations["total_output_bytes"]
            or self.infrastructure_lock_bytes != self.infrastructure_lock_entries
            or self.required_max_total_output_bytes
            != self.frozen_planned_output_bytes + self.infrastructure_lock_bytes
            or self.output_byte_headroom
            != self.manifest.budgets.max_total_output_bytes
            - self.required_max_total_output_bytes
            or self.required_free_disk_bytes_after_reservation
            != self.manifest.budgets.required_free_disk_bytes_after_reservation
        ):
            raise PilotOutputLayoutContractError("pilot_output_layout_plan_budget_arithmetic_mismatch")
        if self.inventory_entry_headroom < 0:
            raise PilotOutputLayoutBudgetStop(
                "pilot_output_layout_global_entry_cap_omits_endpoint_writer_lock"
            )
        if self.scan_traversal_entry_headroom < 0:
            raise PilotOutputLayoutBudgetStop(
                "pilot_output_layout_scan_cap_omits_directory_traversal_headroom"
            )
        if self.output_byte_headroom < 0:
            raise PilotOutputLayoutBudgetStop(
                "pilot_output_layout_global_byte_cap_omits_infrastructure_locks"
            )

    @property
    def plan_hash(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["manifest"] = self.manifest.as_dict()
        result["locator_plans"] = [item.as_dict() for item in self.locator_plans]
        return result

    @classmethod
    def from_dict(cls, payload: object) -> "PilotOutputLayoutPlanV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_output_layout_plan_schema_mismatch",
        )
        raw_plans = values.get("locator_plans")
        raw_manifest = values.get("manifest")
        if not isinstance(raw_plans, list) or not isinstance(raw_manifest, dict):
            raise PilotOutputLayoutContractError("pilot_output_layout_plan_wire_type_mismatch")
        values["manifest"] = MexcPublicQaPilotRunManifestV1.from_dict(raw_manifest)
        values["locator_plans"] = tuple(
            PilotOutputLocatorPlanV1.from_dict(item) for item in raw_plans
        )
        try:
            return cls(**values)
        except TypeError as exc:
            raise PilotOutputLayoutContractError(
                "pilot_output_layout_plan_reconstruction_failed"
            ) from exc

    def validate_preflight(self, receipt: PilotDiskPreflightReceiptV1) -> None:
        if type(receipt) is not PilotDiskPreflightReceiptV1:
            raise PilotOutputLayoutContractError("pilot_output_layout_preflight_type_mismatch")
        if receipt.manifest_hash != self.manifest_hash:
            raise PilotOutputLayoutBudgetStop("pilot_output_layout_preflight_manifest_mismatch")
        try:
            locator = self.locator_plans[receipt.step_ordinal + 1]
        except (IndexError, TypeError) as exc:
            raise PilotOutputLayoutBudgetStop("pilot_output_layout_preflight_step_is_invalid") from exc
        if locator.ordinal != receipt.step_ordinal:
            raise PilotOutputLayoutBudgetStop("pilot_output_layout_preflight_step_is_invalid")
        if receipt.reserved_bytes != locator.frozen_remaining_storage_reservation:
            raise PilotOutputLayoutBudgetStop("pilot_output_layout_preflight_frozen_reservation_mismatch")
        required_after = (
            self.required_free_disk_bytes_after_reservation
            + locator.extra_remaining_lock_bytes
        )
        if receipt.free_bytes_after_reservation < required_after:
            raise PilotOutputLayoutBudgetStop(
                "pilot_output_layout_preflight_omits_infrastructure_lock_headroom"
            )

    def validate_authorization(
        self, receipt: U5PublicPilotAuthorizationReceiptV1
    ) -> None:
        if type(receipt) is not U5PublicPilotAuthorizationReceiptV1:
            raise PilotOutputLayoutContractError(
                "pilot_output_layout_authorization_type_mismatch"
            )
        if receipt.manifest_hash != self.manifest_hash:
            raise PilotOutputLayoutBudgetStop(
                "pilot_output_layout_authorization_manifest_mismatch"
            )
        if receipt.max_total_output_bytes < self.required_max_total_output_bytes:
            raise PilotOutputLayoutBudgetStop(
                "pilot_output_layout_u5_output_cap_omits_infrastructure_lock_bytes"
            )


def _derive_locator_plans(
    manifest: MexcPublicQaPilotRunManifestV1,
) -> tuple[PilotOutputLocatorPlanV1, ...]:
    stage_count = 1 + len(manifest.shards)
    locators: list[PilotOutputLocatorPlanV1] = []
    endpoint = manifest.endpoint_verification
    locators.append(
        PilotOutputLocatorPlanV1(
            stage="endpoint_verification",
            ordinal=-1,
            request_id=endpoint.probe_request.request_id,
            plan_binding_hash=endpoint.plan_hash,
            relative_artifact_root=endpoint.relative_artifact_root,
            writer_lock_relative_path=derive_persistent_writer_lock_locator_v1(
                endpoint.relative_artifact_root
            ),
            frozen_remaining_storage_reservation=manifest.remaining_storage_reservation(-1),
            extra_remaining_lock_entries=stage_count,
            extra_remaining_lock_bytes=stage_count,
        )
    )
    for item in manifest.shards:
        remaining = len(manifest.shards) - item.ordinal
        locators.append(
            PilotOutputLocatorPlanV1(
                stage="shard_acquisition",
                ordinal=item.ordinal,
                request_id=item.request.request_id,
                plan_binding_hash=item.plan_id,
                relative_artifact_root=item.relative_artifact_root,
                writer_lock_relative_path=derive_persistent_writer_lock_locator_v1(
                    item.relative_artifact_root
                ),
                frozen_remaining_storage_reservation=manifest.remaining_storage_reservation(
                    item.ordinal
                ),
                extra_remaining_lock_entries=remaining,
                extra_remaining_lock_bytes=remaining,
            )
        )
    return tuple(locators)


def _maximum_scan_directory_entries(
    manifest: MexcPublicQaPilotRunManifestV1,
) -> int:
    """Conservative exact-v1 directory traversal bound, excluding output root."""

    official_root = derive_official_bundle_root_v1(
        manifest.endpoint_verification.plan_hash
    )
    fixed: set[str] = {
        "run-control",
        "run-control/preflights",
        "run-control/network-intents",
        "run-control/shard-results",
        "endpoint-evidence",
        PurePosixPath(official_root).parent.as_posix(),
        official_root,
    }
    raw_prefix_directories = 0
    requests = (
        (
            manifest.endpoint_verification.relative_artifact_root,
            manifest.endpoint_verification.probe_request,
        ),
        *((item.relative_artifact_root, item.request) for item in manifest.shards),
    )
    for root, request in requests:
        root_path = PurePosixPath(root)
        for parent in reversed(root_path.parents):
            if parent != PurePosixPath("."):
                fixed.add(parent.as_posix())
        fixed.add(root)
        fixed.update(
            {
                f"{root}/attempts",
                f"{root}/raw",
                f"{root}/raw/sha256",
                f"{root}/normalized",
                f"{root}/normalized/{request.request_id}",
                f"{root}/collections",
                f"{root}/collections/{request.request_id}",
            }
        )
        maximum_attempts = (
            request.required_pages
            * request.resource_limits.max_attempts_per_page
        )
        raw_prefix_directories += min(256, maximum_attempts)
    return len(fixed) + raw_prefix_directories


def build_pilot_output_layout_plan_v1(
    manifest: MexcPublicQaPilotRunManifestV1,
) -> PilotOutputLayoutPlanV1:
    if type(manifest) is not MexcPublicQaPilotRunManifestV1:
        raise PilotOutputLayoutContractError("pilot_output_layout_manifest_type_mismatch")
    if manifest.endpoint_verification.max_network_attempts != 2:
        raise PilotOutputLayoutContractError(
            "pilot_output_layout_v1_requires_exactly_two_endpoint_attempts"
        )
    locators = _derive_locator_plans(manifest)
    stage_count = len(locators)
    reservations = manifest.planned_reservations
    required_entries = reservations["inventory_entries"] + 1
    maximum_directories = _maximum_scan_directory_entries(manifest)
    required_traversal = required_entries + maximum_directories
    required_bytes = reservations["total_output_bytes"] + stage_count
    entry_headroom = manifest.budgets.max_inventory_entries - required_entries
    traversal_headroom = manifest.budgets.max_inventory_entries - required_traversal
    byte_headroom = manifest.budgets.max_total_output_bytes - required_bytes
    if entry_headroom < 0:
        raise PilotOutputLayoutBudgetStop(
            "pilot_output_layout_global_entry_cap_omits_endpoint_writer_lock"
        )
    if traversal_headroom < 0:
        raise PilotOutputLayoutBudgetStop(
            "pilot_output_layout_scan_cap_omits_directory_traversal_headroom"
        )
    if byte_headroom < 0:
        raise PilotOutputLayoutBudgetStop(
            "pilot_output_layout_global_byte_cap_omits_infrastructure_locks"
        )
    return PilotOutputLayoutPlanV1(
        manifest=manifest,
        manifest_hash=manifest.manifest_hash,
        locator_plans=locators,
        frozen_planned_inventory_entries=reservations["inventory_entries"],
        infrastructure_lock_entries=stage_count,
        required_max_inventory_entries=required_entries,
        inventory_entry_headroom=entry_headroom,
        maximum_scan_directory_entries=maximum_directories,
        required_scan_traversal_entries=required_traversal,
        scan_traversal_entry_headroom=traversal_headroom,
        frozen_planned_output_bytes=reservations["total_output_bytes"],
        infrastructure_lock_bytes=stage_count,
        required_max_total_output_bytes=required_bytes,
        output_byte_headroom=byte_headroom,
        required_free_disk_bytes_after_reservation=(
            manifest.budgets.required_free_disk_bytes_after_reservation
        ),
    )


def _prefixed(root: str, relative_path: str) -> str:
    return _relative_path(
        f"{root}/{relative_path}",
        field="pilot_output_prefixed_history_path",
    )


def _derive_stage_components(
    *,
    pilot_manifest_hash: str,
    locator_plan: PilotOutputLocatorPlanV1,
    evidence: StrictHistoryPilotEvidenceV1,
    official_document_placeholder: PilotOfficialDocumentPlaceholderV1 | None,
) -> tuple[
    str,
    tuple[PilotOutputPhysicalEntryV1, ...],
    str,
    tuple[PilotOutputPhysicalEntryV1, ...],
    str,
]:
    _digest(pilot_manifest_hash, field="pilot_output_stage_manifest_hash")
    if type(locator_plan) is not PilotOutputLocatorPlanV1:
        raise PilotOutputLayoutContractError("pilot_output_stage_locator_type_mismatch")
    if type(evidence) is not StrictHistoryPilotEvidenceV1:
        raise PilotOutputLayoutContractError("pilot_output_stage_evidence_type_mismatch")
    if evidence.request_id != locator_plan.request_id:
        raise PilotOutputLayoutInventoryStop("pilot_output_stage_request_binding_mismatch")
    if (
        evidence.pilot_output_layout_status != PILOT_OUTPUT_LAYOUT_STATUS_UNRESOLVED
        or evidence.authority_status != PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE
    ):
        raise PilotOutputLayoutInventoryStop("pilot_output_stage_evidence_boundary_mismatch")
    lock = evidence.writer_lock
    if (
        lock.status != "present_plain_regular"
        or lock.file_sha256 != _SHA256_ZERO_BYTE
        or lock.byte_count != 1
        or lock.link_count != 1
    ):
        raise PilotOutputLayoutInventoryStop("pilot_output_stage_writer_lock_missing_or_invalid")

    logical_payload = [
        {
            **item.as_dict(),
            "relative_path": _prefixed(
                locator_plan.relative_artifact_root,
                item.relative_path,
            ),
        }
        for item in evidence.logical_references
    ]
    logical_hash = _sha256_payload(
        {
            "domain": "mexc_public_qa_pilot_prefixed_logical_inventory_v1",
            "pilot_manifest_hash": pilot_manifest_hash,
            "stage": locator_plan.stage,
            "ordinal": locator_plan.ordinal,
            "entries": logical_payload,
        }
    )
    history_entries = tuple(
        sorted(
            (
                PilotOutputPhysicalEntryV1(
                    relative_path=_prefixed(
                        locator_plan.relative_artifact_root,
                        item.relative_path,
                    ),
                    artifact_sha256=item.file_sha256,
                    byte_count=item.byte_count,
                    source="strict_history_physical",
                    role=item.role,
                    stage=locator_plan.stage,
                    ordinal=locator_plan.ordinal,
                    logical_reference_count=item.logical_reference_count,
                )
                for item in evidence.physical_files
            ),
            key=lambda item: item.relative_path,
        )
    )
    history_hash = _sha256_payload(
        {
            "domain": "mexc_public_qa_pilot_prefixed_unique_physical_inventory_v1",
            "pilot_manifest_hash": pilot_manifest_hash,
            "stage": locator_plan.stage,
            "ordinal": locator_plan.ordinal,
            "entries": [item.as_dict() for item in history_entries],
        }
    )
    extras: list[PilotOutputPhysicalEntryV1] = [
        PilotOutputPhysicalEntryV1(
            relative_path=locator_plan.writer_lock_relative_path,
            artifact_sha256=_SHA256_ZERO_BYTE,
            byte_count=1,
            source="infrastructure_writer_lock",
            role="persistent_writer_lock",
            stage=locator_plan.stage,
            ordinal=locator_plan.ordinal,
            logical_reference_count=0,
        )
    ]
    if locator_plan.stage == "endpoint_verification":
        if type(official_document_placeholder) is not PilotOfficialDocumentPlaceholderV1:
            raise PilotOutputLayoutInventoryStop(
                "pilot_output_endpoint_official_document_placeholder_is_required"
            )
        if official_document_placeholder.endpoint_plan_hash != locator_plan.plan_binding_hash:
            raise PilotOutputLayoutInventoryStop(
                "pilot_output_official_document_placeholder_plan_mismatch"
            )
        official_root = derive_official_bundle_root_v1(
            official_document_placeholder.endpoint_plan_hash
        )
        if _paths_overlap(locator_plan.relative_artifact_root, official_root) or _paths_overlap(
            locator_plan.writer_lock_relative_path, official_root
        ):
            raise PilotOutputLayoutInventoryStop(
                "pilot_output_stage_and_official_bundle_namespace_overlap"
            )
        extras.extend(
            PilotOutputPhysicalEntryV1(
                relative_path=file.relative_path,
                artifact_sha256=file.artifact_sha256,
                byte_count=file.byte_count,
                source="official_document_placeholder",
                role=f"official_document_{file.role}_unresolved",
                stage="endpoint_verification",
                ordinal=-1,
                logical_reference_count=0,
            )
            for file in official_document_placeholder.files
        )
    elif official_document_placeholder is not None:
        raise PilotOutputLayoutInventoryStop(
            "pilot_output_shard_cannot_have_official_document_placeholder"
        )
    expected_entries = tuple(
        sorted((*history_entries, *extras), key=lambda item: item.relative_path)
    )
    paths = tuple(item.relative_path for item in expected_entries)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if _paths_overlap(left, right):
                raise PilotOutputLayoutInventoryStop(
                    "pilot_output_stage_physical_path_or_prefix_collision"
                )
    expected_hash = _sha256_payload(
        {
            "domain": PILOT_OUTPUT_STAGE_ACCOUNTING_VERSION,
            "pilot_manifest_hash": pilot_manifest_hash,
            "stage": locator_plan.stage,
            "ordinal": locator_plan.ordinal,
            "entries": [item.as_dict() for item in expected_entries],
        }
    )
    return (
        logical_hash,
        history_entries,
        history_hash,
        expected_entries,
        expected_hash,
    )


def build_pilot_stage_output_accounting_v1(
    *,
    pilot_manifest_hash: str,
    locator_plan: PilotOutputLocatorPlanV1,
    evidence: StrictHistoryPilotEvidenceV1,
    official_document_placeholder: PilotOfficialDocumentPlaceholderV1 | None = None,
) -> PilotStageOutputAccountingV1:
    (
        logical_hash,
        history_entries,
        history_hash,
        expected_entries,
        expected_hash,
    ) = _derive_stage_components(
        pilot_manifest_hash=pilot_manifest_hash,
        locator_plan=locator_plan,
        evidence=evidence,
        official_document_placeholder=official_document_placeholder,
    )
    if locator_plan.stage == "endpoint_verification":
        subject = tuple(
            item
            for item in expected_entries
            if item.source
            in {"strict_history_physical", "official_document_placeholder"}
        )
        receipt_inventory_hash = _sha256_payload(
            {
                "domain": "mexc_public_qa_pilot_endpoint_subject_inventory_v1",
                "pilot_manifest_hash": pilot_manifest_hash,
                "plan_binding_hash": locator_plan.plan_binding_hash,
                "entries": [item.as_dict() for item in subject],
            }
        )
        receipt_inventory_entries = len(subject)
        receipt_inventory_bytes = sum(item.byte_count for item in subject)
    else:
        receipt_inventory_hash = logical_hash
        receipt_inventory_entries = len(evidence.logical_references)
        receipt_inventory_bytes = evidence.admitted_total_logical_storage_bytes
    return PilotStageOutputAccountingV1(
        pilot_manifest_hash=pilot_manifest_hash,
        locator_plan=locator_plan,
        strict_history_evidence=evidence,
        official_document_placeholder=official_document_placeholder,
        stage=locator_plan.stage,
        ordinal=locator_plan.ordinal,
        request_id=locator_plan.request_id,
        plan_binding_hash=locator_plan.plan_binding_hash,
        relative_artifact_root=locator_plan.relative_artifact_root,
        history_manifest_hash=evidence.manifest_hash,
        strict_history_evidence_hash=evidence.evidence_hash,
        logical_inventory_hash=logical_hash,
        logical_reference_entries=len(evidence.logical_references),
        logical_reference_bytes=evidence.admitted_total_logical_storage_bytes,
        unique_history_physical_inventory_hash=history_hash,
        unique_history_physical_entries=len(history_entries),
        unique_history_physical_bytes=evidence.unique_physical_referenced_bytes,
        writer_lock_relative_path=locator_plan.writer_lock_relative_path,
        expected_physical_entries=expected_entries,
        expected_physical_inventory_hash=expected_hash,
        dependent_receipt_inventory_hash=receipt_inventory_hash,
        dependent_receipt_inventory_entries=receipt_inventory_entries,
        dependent_receipt_inventory_bytes=receipt_inventory_bytes,
    )


def _run_control_entries(state: PilotRunStateV1) -> tuple[PilotOutputPhysicalEntryV1, ...]:
    if state.next_action != "publish_detached_result_anchor":
        raise PilotOutputLayoutInventoryStop(
            "pilot_output_run_control_requires_post_candidate_pre_anchor_state"
        )
    return tuple(
        sorted(
            (
                PilotOutputPhysicalEntryV1(
                    relative_path=locator,
                    artifact_sha256=artifact_sha256,
                    byte_count=byte_count,
                    source="run_control",
                    role=kind,
                    stage="run_control",
                    ordinal=-2,
                    logical_reference_count=0,
                )
                for kind, locator, _semantic_hash, artifact_sha256, byte_count
                in state.final_run_control_inventory
            ),
            key=lambda item: item.relative_path,
        )
    )


def derive_expected_pilot_output_inventory_v1(
    *,
    state: PilotRunStateV1,
    stage_accounting: tuple[PilotStageOutputAccountingV1, ...],
) -> tuple[PilotOutputPhysicalEntryV1, ...]:
    if type(state) is not PilotRunStateV1:
        raise PilotOutputLayoutContractError("pilot_output_layout_state_type_mismatch")
    if not isinstance(stage_accounting, tuple) or not all(
        type(item) is PilotStageOutputAccountingV1 for item in stage_accounting
    ):
        raise PilotOutputLayoutContractError("pilot_output_stage_accounting_tuple_is_invalid")
    completed_count = (
        (1 if state.endpoint_verification is not None else 0)
        + len(state.shard_results)
    )
    expected_locators = _derive_locator_plans(state.manifest)[:completed_count]
    if (
        len(stage_accounting) != completed_count
        or tuple(item.locator_plan for item in stage_accounting) != expected_locators
        or any(item.pilot_manifest_hash != state.manifest.manifest_hash for item in stage_accounting)
    ):
        raise PilotOutputLayoutInventoryStop(
            "pilot_output_stage_accounting_state_projection_mismatch"
        )
    entries = [*_run_control_entries(state)]
    for item in stage_accounting:
        entries.extend(item.expected_physical_entries)
    result = tuple(sorted(entries, key=lambda item: item.relative_path))
    paths = tuple(item.relative_path for item in result)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if _paths_overlap(left, right):
                raise PilotOutputLayoutInventoryStop(
                    "pilot_output_global_physical_path_collision"
                )
    return result


def _validate_readiness_source_contracts(
    *,
    layout_plan: PilotOutputLayoutPlanV1,
    state: PilotRunStateV1,
    stage_accounting: tuple[PilotStageOutputAccountingV1, ...],
) -> None:
    """Replay every source-side gate required by a readiness assessment."""

    if state.manifest != layout_plan.manifest:
        raise PilotOutputLayoutInventoryStop(
            "pilot_output_readiness_plan_state_manifest_mismatch"
        )
    if state.authorization is None:
        raise PilotOutputLayoutInventoryStop(
            "pilot_output_layout_authorization_is_missing"
        )
    layout_plan.validate_authorization(state.authorization)
    if len(state.preflight_receipts) != len(layout_plan.locator_plans):
        raise PilotOutputLayoutInventoryStop(
            "pilot_output_readiness_preflight_set_mismatch"
        )
    for receipt in state.preflight_receipts:
        layout_plan.validate_preflight(receipt)
    if (
        state.endpoint_verification is None
        or len(stage_accounting) != len(layout_plan.locator_plans)
        or tuple(item.locator_plan for item in stage_accounting)
        != layout_plan.locator_plans
    ):
        raise PilotOutputLayoutInventoryStop(
            "pilot_output_readiness_stage_source_set_mismatch"
        )

    endpoint_item = stage_accounting[0]
    endpoint_evidence = endpoint_item.strict_history_evidence
    official = endpoint_item.official_document_placeholder
    endpoint_receipt = state.endpoint_verification
    if (
        type(official) is not PilotOfficialDocumentPlaceholderV1
        or official.endpoint_plan_hash
        != layout_plan.manifest.endpoint_verification.plan_hash
        or official.files[-1].artifact_sha256
        != endpoint_receipt.official_document_evidence_hash
    ):
        raise PilotOutputLayoutInventoryStop(
            "pilot_output_layout_official_placeholder_binding_mismatch"
        )
    official_raw = official.files[0]
    if (
        endpoint_evidence.manifest_hash
        != endpoint_receipt.live_history_manifest_hash
        or endpoint_evidence.row_count != endpoint_receipt.live_observed_rows
        or endpoint_evidence.attempt_count != 1
        or endpoint_evidence.attempt_accounting[0].attempt_receipt_hash
        != endpoint_receipt.live_attempt_receipt_hash
        or endpoint_evidence.attempt_accounting[0].raw_body_sha256
        != endpoint_receipt.live_raw_body_sha256
        or endpoint_receipt.actual_network_attempts != 2
        or endpoint_receipt.actual_raw_body_bytes
        != official_raw.byte_count + endpoint_evidence.actual_total_raw_body_bytes
        or endpoint_receipt.actual_storage_bytes
        != endpoint_item.dependent_receipt_inventory_bytes
        or endpoint_receipt.output_inventory_hash
        != endpoint_item.dependent_receipt_inventory_hash
        or endpoint_receipt.output_inventory_entries
        != endpoint_item.dependent_receipt_inventory_entries
    ):
        raise PilotOutputLayoutInventoryStop(
            "pilot_output_endpoint_evidence_receipt_mismatch"
        )

    for item, receipt in zip(
        stage_accounting[1:], state.shard_results, strict=True
    ):
        evidence = item.strict_history_evidence
        locator = item.locator_plan
        if (
            receipt.ordinal != locator.ordinal
            or receipt.history_manifest_hash != evidence.manifest_hash
            or receipt.actual_pages != evidence.page_count
            or receipt.actual_rows != evidence.row_count
            or receipt.actual_attempts != evidence.attempt_count
            or receipt.actual_raw_body_bytes != evidence.actual_total_raw_body_bytes
            or receipt.actual_logical_storage_bytes
            != evidence.admitted_total_logical_storage_bytes
            or receipt.actual_collection_runtime_us
            != evidence.manifest_collection_runtime_us
            or receipt.observed_internal_sleep_us
            != evidence.observed_monotonic_inter_attempt_sleep_us
            or receipt.output_inventory_entries != len(evidence.logical_references)
            or receipt.output_inventory_bytes
            != evidence.admitted_total_logical_storage_bytes
            or receipt.output_inventory_hash
            != item.dependent_receipt_inventory_hash
        ):
            raise PilotOutputLayoutInventoryStop(
                "pilot_output_shard_evidence_receipt_mismatch"
            )


@dataclass(frozen=True, slots=True)
class PilotOutputReadinessAssessmentV1:
    layout_plan: PilotOutputLayoutPlanV1
    state: PilotRunStateV1
    inventory_scan: PilotInventoryScanV1
    manifest_hash: str
    state_hash: str
    layout_plan_hash: str
    readiness_phase: str
    stage_accounting: tuple[PilotStageOutputAccountingV1, ...]
    completed_stage_count: int
    planned_stage_count: int
    completed_stage_set_exact: bool
    logical_reference_entries: int
    logical_reference_bytes: int
    unique_stage_physical_entries: int
    unique_stage_physical_bytes: int
    run_control_entries: int
    run_control_bytes: int
    infrastructure_lock_entries: int
    infrastructure_lock_bytes: int
    official_document_placeholder_entries: int
    official_document_placeholder_bytes: int
    expected_physical_entries_detail: tuple[PilotOutputPhysicalEntryV1, ...]
    expected_inventory_hash: str
    expected_inventory_entries: int
    expected_inventory_bytes: int
    observed_physical_entries_detail: tuple[PilotInventoryEntryV1, ...]
    observed_inventory_hash: str
    observed_inventory_entries: int
    observed_inventory_bytes: int
    exact_file_inventory_match: bool
    directory_namespace_exact: bool
    directory_namespace_status: str
    hardlink_alias_assurance: str
    snapshot_boundary: str
    writer_boundary: str
    blockers: tuple[str, ...]
    terminal_compatible: bool = False
    contract_version: str = PILOT_OUTPUT_READINESS_ASSESSMENT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_OUTPUT_READINESS_ASSESSMENT_VERSION:
            raise PilotOutputLayoutContractError("pilot_output_readiness_version_mismatch")
        if type(self.layout_plan) is not PilotOutputLayoutPlanV1:
            raise PilotOutputLayoutContractError("pilot_output_readiness_layout_plan_invalid")
        if type(self.state) is not PilotRunStateV1:
            raise PilotOutputLayoutContractError("pilot_output_readiness_state_invalid")
        if type(self.inventory_scan) is not PilotInventoryScanV1:
            raise PilotOutputLayoutContractError("pilot_output_readiness_inventory_scan_invalid")
        if (
            self.state.next_action != "publish_detached_result_anchor"
            or self.state.endpoint_verification is None
            or not self.state.shard_results
            or len(self.state.shard_results) != len(self.state.manifest.shards)
        ):
            raise PilotOutputLayoutContractError(
                "pilot_output_readiness_state_phase_mismatch"
            )
        for field in (
            "manifest_hash",
            "state_hash",
            "layout_plan_hash",
            "expected_inventory_hash",
            "observed_inventory_hash",
        ):
            _digest(getattr(self, field), field=f"pilot_output_readiness_{field}")
        if not isinstance(self.stage_accounting, tuple) or not all(
            type(item) is PilotStageOutputAccountingV1 for item in self.stage_accounting
        ):
            raise PilotOutputLayoutContractError("pilot_output_readiness_stage_accounting_invalid")
        if self.readiness_phase != "post_result_candidate_pre_anchor":
            raise PilotOutputLayoutContractError("pilot_output_readiness_phase_mismatch")
        _validate_readiness_source_contracts(
            layout_plan=self.layout_plan,
            state=self.state,
            stage_accounting=self.stage_accounting,
        )
        _require_canonical_entries(
            self.expected_physical_entries_detail,
            field="pilot_output_readiness_expected_entries",
        )
        if not isinstance(self.observed_physical_entries_detail, tuple) or not all(
            type(item) is PilotInventoryEntryV1
            for item in self.observed_physical_entries_detail
        ):
            raise PilotOutputLayoutContractError(
                "pilot_output_readiness_observed_entries_are_not_exact_immutable"
            )
        observed_paths = tuple(
            item.relative_path for item in self.observed_physical_entries_detail
        )
        if observed_paths != tuple(sorted(observed_paths)) or len(
            {item.casefold() for item in observed_paths}
        ) != len(observed_paths):
            raise PilotOutputLayoutContractError(
                "pilot_output_readiness_observed_entries_are_not_canonical"
            )
        for field in (
            "completed_stage_count",
            "planned_stage_count",
            "logical_reference_entries",
            "logical_reference_bytes",
            "unique_stage_physical_entries",
            "unique_stage_physical_bytes",
            "run_control_entries",
            "run_control_bytes",
            "infrastructure_lock_entries",
            "infrastructure_lock_bytes",
            "official_document_placeholder_entries",
            "official_document_placeholder_bytes",
            "expected_inventory_entries",
            "expected_inventory_bytes",
            "observed_inventory_entries",
            "observed_inventory_bytes",
        ):
            _strict_int(getattr(self, field), field=f"pilot_output_readiness_{field}")
        if self.completed_stage_set_exact is not True:
            raise PilotOutputLayoutContractError("pilot_output_readiness_completion_flag_invalid")
        expected = self.expected_physical_entries_detail
        observed = self.observed_physical_entries_detail
        history = tuple(item for item in expected if item.source == "strict_history_physical")
        controls = tuple(item for item in expected if item.source == "run_control")
        locks = tuple(item for item in expected if item.source == "infrastructure_writer_lock")
        official = tuple(item for item in expected if item.source == "official_document_placeholder")
        expected_hash = _sha256_payload(
            {
                "domain": "mexc_public_qa_pilot_exact_global_physical_inventory_v1",
                "manifest_hash": self.manifest_hash,
                "entries": [item.as_dict() for item in expected],
                "total_bytes": sum(item.byte_count for item in expected),
            }
        )
        observed_total = sum(item.byte_count for item in observed)
        observed_hash = _sha256_payload(
            {
                "domain": PILOT_LOCAL_INVENTORY_VERSION,
                "manifest_hash": self.manifest_hash,
                "entries": [item.as_dict() for item in observed],
                "total_bytes": observed_total,
            }
        )
        expected_projection = tuple(
            (item.relative_path, item.artifact_sha256, item.byte_count)
            for item in expected
        )
        observed_projection = tuple(
            (item.relative_path, item.artifact_sha256, item.byte_count)
            for item in observed
        )
        derived_expected = derive_expected_pilot_output_inventory_v1(
            state=self.state,
            stage_accounting=self.stage_accounting,
        )
        last = self.state.shard_results[-1]
        if (
            self.manifest_hash != self.layout_plan.manifest_hash
            or self.state.manifest != self.layout_plan.manifest
            or self.state_hash != self.state.state_hash
            or self.inventory_scan.manifest_hash != self.manifest_hash
            or self.inventory_scan.entries != observed
            or self.inventory_scan.inventory_hash != self.observed_inventory_hash
            or self.inventory_scan.total_bytes != self.observed_inventory_bytes
            or self.inventory_scan.clock_domain_id
            != self.state.endpoint_verification.clock_domain_id  # type: ignore[union-attr]
            or self.inventory_scan.scanned_at_us < last.step_completed_at_us
            or self.inventory_scan.scanned_monotonic_us
            < last.step_completed_monotonic_us
            or self.layout_plan_hash != self.layout_plan.plan_hash
            or self.completed_stage_count != len(self.stage_accounting)
            or self.planned_stage_count != len(self.layout_plan.locator_plans)
            or self.completed_stage_count != self.planned_stage_count
            or tuple(item.locator_plan for item in self.stage_accounting)
            != self.layout_plan.locator_plans
            or expected != derived_expected
            or self.logical_reference_entries
            != sum(item.logical_reference_entries for item in self.stage_accounting)
            or self.logical_reference_bytes
            != sum(item.logical_reference_bytes for item in self.stage_accounting)
            or self.unique_stage_physical_entries != len(history)
            or self.unique_stage_physical_bytes
            != sum(item.byte_count for item in history)
            or self.run_control_entries != len(controls)
            or self.run_control_bytes != sum(item.byte_count for item in controls)
            or self.infrastructure_lock_entries != len(locks)
            or self.infrastructure_lock_bytes != sum(item.byte_count for item in locks)
            or self.official_document_placeholder_entries != len(official)
            or self.official_document_placeholder_bytes
            != sum(item.byte_count for item in official)
            or self.expected_inventory_hash != expected_hash
            or self.expected_inventory_entries != len(expected)
            or self.expected_inventory_bytes != sum(item.byte_count for item in expected)
            or self.observed_inventory_hash != observed_hash
            or self.observed_inventory_entries != len(observed)
            or self.observed_inventory_bytes != observed_total
            or expected_projection != observed_projection
        ):
            raise PilotOutputLayoutContractError(
                "pilot_output_readiness_derived_projection_mismatch"
            )
        if self.exact_file_inventory_match is not True:
            raise PilotOutputLayoutContractError("pilot_output_readiness_inventory_must_be_exact")
        if (
            self.directory_namespace_exact is not False
            or self.directory_namespace_status
            != "unobserved_by_frozen_file_inventory_scan"
        ):
            raise PilotOutputLayoutContractError(
                "pilot_output_readiness_directory_namespace_boundary_mismatch"
            )
        if self.hardlink_alias_assurance != _HARDLINK_ASSURANCE:
            raise PilotOutputLayoutContractError("pilot_output_readiness_hardlink_boundary_mismatch")
        if self.snapshot_boundary != _SNAPSHOT_BOUNDARY or self.writer_boundary != _WRITER_BOUNDARY:
            raise PilotOutputLayoutContractError("pilot_output_readiness_snapshot_boundary_mismatch")
        if not isinstance(self.blockers, tuple) or self.blockers != tuple(sorted(set(self.blockers))):
            raise PilotOutputLayoutContractError("pilot_output_readiness_blockers_not_canonical")
        if self.blockers != tuple(sorted(_TERMINAL_BLOCKERS)):
            raise PilotOutputLayoutContractError("pilot_output_readiness_terminal_blockers_mismatch")
        if self.terminal_compatible is not False:
            raise PilotOutputLayoutContractError("pilot_output_readiness_cannot_be_terminal")

    @property
    def assessment_hash(self) -> str:
        return _frozen_contract_hash(self)

    def require_terminal_compatible(self) -> "PilotOutputReadinessAssessmentV1":
        raise PilotOutputLayoutTerminalStop(
            "pilot_output_layout_terminal_stop_official_document_schema_and_publisher_unbound"
        )

    def as_dict(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result["layout_plan"] = self.layout_plan.as_dict()
        result["state"] = self.state.as_dict()
        result["inventory_scan"] = _inventory_scan_as_dict(self.inventory_scan)
        result["stage_accounting"] = [item.as_dict() for item in self.stage_accounting]
        result["expected_physical_entries_detail"] = [
            item.as_dict() for item in self.expected_physical_entries_detail
        ]
        result["observed_physical_entries_detail"] = [
            item.as_dict() for item in self.observed_physical_entries_detail
        ]
        result["blockers"] = list(self.blockers)
        return result

    @classmethod
    def from_dict(cls, payload: object) -> "PilotOutputReadinessAssessmentV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_output_readiness_schema_mismatch",
        )
        raw_stages = values.get("stage_accounting")
        raw_blockers = values.get("blockers")
        raw_expected = values.get("expected_physical_entries_detail")
        raw_observed = values.get("observed_physical_entries_detail")
        if (
            not isinstance(raw_stages, list)
            or not isinstance(raw_blockers, list)
            or not all(isinstance(item, str) for item in raw_blockers)
            or not isinstance(raw_expected, list)
            or not isinstance(raw_observed, list)
        ):
            raise PilotOutputLayoutContractError("pilot_output_readiness_wire_type_mismatch")
        values["layout_plan"] = PilotOutputLayoutPlanV1.from_dict(
            values.get("layout_plan")
        )
        values["state"] = PilotRunStateV1.from_dict(values.get("state"))
        values["inventory_scan"] = _parse_inventory_scan(
            values.get("inventory_scan")
        )
        values["stage_accounting"] = tuple(
            PilotStageOutputAccountingV1.from_dict(item) for item in raw_stages
        )
        values["blockers"] = tuple(raw_blockers)
        values["expected_physical_entries_detail"] = tuple(
            PilotOutputPhysicalEntryV1.from_dict(item) for item in raw_expected
        )
        try:
            values["observed_physical_entries_detail"] = tuple(
                PilotInventoryEntryV1(
                    **_exact_mapping(
                        item,
                        frozenset(PilotInventoryEntryV1.__dataclass_fields__),
                        code="pilot_output_readiness_observed_entry_schema_mismatch",
                    )
                )
                for item in raw_observed
            )
        except TypeError as exc:
            raise PilotOutputLayoutContractError(
                "pilot_output_readiness_observed_entry_reconstruction_failed"
            ) from exc
        try:
            return cls(**values)
        except TypeError as exc:
            raise PilotOutputLayoutContractError(
                "pilot_output_readiness_reconstruction_failed"
            ) from exc


def _completed_locator_plans(
    state: PilotRunStateV1,
    plan: PilotOutputLayoutPlanV1,
) -> tuple[PilotOutputLocatorPlanV1, ...]:
    count = (1 if state.endpoint_verification is not None else 0) + len(state.shard_results)
    if state.endpoint_verification is None and state.shard_results:
        raise PilotOutputLayoutInventoryStop("pilot_output_shards_without_endpoint")
    return plan.locator_plans[:count]


def assess_pilot_output_layout_v1(
    *,
    manifest: MexcPublicQaPilotRunManifestV1,
    state: PilotRunStateV1,
    inventory_scan: PilotInventoryScanV1,
    stage_evidences: tuple[StrictHistoryPilotEvidenceV1, ...],
    official_document_placeholder: PilotOfficialDocumentPlaceholderV1 | None = None,
) -> PilotOutputReadinessAssessmentV1:
    if type(manifest) is not MexcPublicQaPilotRunManifestV1:
        raise PilotOutputLayoutContractError("pilot_output_layout_manifest_type_mismatch")
    if type(state) is not PilotRunStateV1 or state.manifest != manifest:
        raise PilotOutputLayoutContractError("pilot_output_layout_state_manifest_mismatch")
    if type(inventory_scan) is not PilotInventoryScanV1:
        raise PilotOutputLayoutContractError("pilot_output_layout_scan_type_mismatch")
    if inventory_scan.manifest_hash != manifest.manifest_hash:
        raise PilotOutputLayoutInventoryStop("pilot_output_layout_scan_manifest_mismatch")
    if not isinstance(stage_evidences, tuple) or not all(
        type(item) is StrictHistoryPilotEvidenceV1 for item in stage_evidences
    ):
        raise PilotOutputLayoutContractError("pilot_output_layout_stage_evidences_are_invalid")
    if state.failure_receipt is not None or state.stop_reason is not None or state.final_anchor is not None:
        raise PilotOutputLayoutInventoryStop("pilot_output_layout_requires_positive_preterminal_state")
    if state.active_network_intent is not None:
        raise PilotOutputLayoutInventoryStop("pilot_output_layout_active_intent_has_unresolved_residue")
    if (
        state.next_action != "publish_detached_result_anchor"
        or state.endpoint_verification is None
        or len(state.shard_results) != len(manifest.shards)
    ):
        raise PilotOutputLayoutInventoryStop(
            "pilot_output_layout_requires_complete_post_candidate_pre_anchor_phase"
        )
    last_result = state.shard_results[-1]
    if (
        inventory_scan.clock_domain_id
        != state.endpoint_verification.clock_domain_id
        or inventory_scan.scanned_at_us < last_result.step_completed_at_us
        or inventory_scan.scanned_monotonic_us
        < last_result.step_completed_monotonic_us
    ):
        raise PilotOutputLayoutInventoryStop(
            "pilot_output_inventory_scan_clock_or_freshness_mismatch"
        )

    plan = build_pilot_output_layout_plan_v1(manifest)
    completed = _completed_locator_plans(state, plan)
    if len(stage_evidences) != len(completed):
        raise PilotOutputLayoutInventoryStop("pilot_output_layout_stage_evidence_count_mismatch")
    if type(official_document_placeholder) is not PilotOfficialDocumentPlaceholderV1:
        raise PilotOutputLayoutInventoryStop("pilot_output_layout_official_placeholder_required")
    accounting: list[PilotStageOutputAccountingV1] = []
    for locator, evidence in zip(completed, stage_evidences, strict=True):
        placeholder = official_document_placeholder if locator.ordinal == -1 else None
        item = build_pilot_stage_output_accounting_v1(
            pilot_manifest_hash=manifest.manifest_hash,
            locator_plan=locator,
            evidence=evidence,
            official_document_placeholder=placeholder,
        )
        accounting.append(item)

    accounting_tuple = tuple(accounting)
    _validate_readiness_source_contracts(
        layout_plan=plan,
        state=state,
        stage_accounting=accounting_tuple,
    )
    expected = derive_expected_pilot_output_inventory_v1(
        state=state,
        stage_accounting=accounting_tuple,
    )
    observed_by_path = {
        item.relative_path: (item.artifact_sha256, item.byte_count)
        for item in inventory_scan.entries
    }
    expected_by_path = {
        item.relative_path: (item.artifact_sha256, item.byte_count)
        for item in expected
    }
    if observed_by_path != expected_by_path:
        missing = tuple(sorted(set(expected_by_path) - set(observed_by_path)))
        unexpected = tuple(sorted(set(observed_by_path) - set(expected_by_path)))
        changed = tuple(
            sorted(
                path for path in set(expected_by_path) & set(observed_by_path)
                if expected_by_path[path] != observed_by_path[path]
            )
        )
        evidence_hash = _sha256_payload(
            {
                "domain": "mexc_public_qa_pilot_output_inventory_mismatch_v1",
                "manifest_hash": manifest.manifest_hash,
                "missing": list(missing),
                "unexpected": list(unexpected),
                "changed": list(changed),
            }
        )
        raise PilotOutputLayoutInventoryStop(
            f"pilot_output_inventory_union_mismatch:{evidence_hash}"
        )
    expected_bytes = sum(item.byte_count for item in expected)
    if expected_bytes != inventory_scan.total_bytes:
        raise PilotOutputLayoutInventoryStop("pilot_output_inventory_total_byte_mismatch")
    expected_hash = _sha256_payload(
        {
            "domain": "mexc_public_qa_pilot_exact_global_physical_inventory_v1",
            "manifest_hash": manifest.manifest_hash,
            "entries": [item.as_dict() for item in expected],
            "total_bytes": expected_bytes,
        }
    )
    run_control = tuple(item for item in expected if item.source == "run_control")
    history = tuple(item for item in expected if item.source == "strict_history_physical")
    locks = tuple(item for item in expected if item.source == "infrastructure_writer_lock")
    official = tuple(item for item in expected if item.source == "official_document_placeholder")
    return PilotOutputReadinessAssessmentV1(
        layout_plan=plan,
        state=state,
        inventory_scan=inventory_scan,
        manifest_hash=manifest.manifest_hash,
        state_hash=state.state_hash,
        layout_plan_hash=plan.plan_hash,
        readiness_phase="post_result_candidate_pre_anchor",
        stage_accounting=accounting_tuple,
        completed_stage_count=len(completed),
        planned_stage_count=len(plan.locator_plans),
        completed_stage_set_exact=True,
        logical_reference_entries=sum(item.logical_reference_entries for item in accounting_tuple),
        logical_reference_bytes=sum(item.logical_reference_bytes for item in accounting_tuple),
        unique_stage_physical_entries=len(history),
        unique_stage_physical_bytes=sum(item.byte_count for item in history),
        run_control_entries=len(run_control),
        run_control_bytes=sum(item.byte_count for item in run_control),
        infrastructure_lock_entries=len(locks),
        infrastructure_lock_bytes=sum(item.byte_count for item in locks),
        official_document_placeholder_entries=len(official),
        official_document_placeholder_bytes=sum(item.byte_count for item in official),
        expected_physical_entries_detail=expected,
        expected_inventory_hash=expected_hash,
        expected_inventory_entries=len(expected),
        expected_inventory_bytes=expected_bytes,
        observed_physical_entries_detail=inventory_scan.entries,
        observed_inventory_hash=inventory_scan.inventory_hash,
        observed_inventory_entries=len(inventory_scan.entries),
        observed_inventory_bytes=inventory_scan.total_bytes,
        exact_file_inventory_match=True,
        directory_namespace_exact=False,
        directory_namespace_status="unobserved_by_frozen_file_inventory_scan",
        hardlink_alias_assurance=_HARDLINK_ASSURANCE,
        snapshot_boundary=_SNAPSHOT_BOUNDARY,
        writer_boundary=_WRITER_BOUNDARY,
        blockers=tuple(sorted(_TERMINAL_BLOCKERS)),
    )


_CONTRACT_SCHEMA = {
    "contract_version": PILOT_OUTPUT_LAYOUT_CONTRACT_VERSION,
    "dependencies": {
        "pilot_run": {
            "version": PILOT_RUN_CONTRACT_VERSION,
            "hash": pilot_run_contract_hash(),
        },
        "local_store": {
            "version": PILOT_LOCAL_STORE_CONTRACT_VERSION,
            "hash": mexc_pilot_local_store_contract_hash(),
            "inventory_version": PILOT_LOCAL_INVENTORY_VERSION,
        },
        "local_executor": {
            "version": PILOT_LOCAL_EXECUTOR_CONTRACT_VERSION,
            "hash": pilot_local_executor_contract_hash(),
        },
        "readiness_coordinator": {
            "version": PILOT_LOCAL_COORDINATOR_CONTRACT_VERSION,
            "hash": pilot_local_coordinator_contract_hash(),
        },
        "strict_history_pilot_evidence": {
            "version": STRICT_HISTORY_PILOT_EVIDENCE_CONTRACT_VERSION,
            "hash": strict_history_pilot_evidence_contract_hash(),
        },
    },
    "component_versions": {
        "locator_plan": PILOT_OUTPUT_LOCATOR_PLAN_VERSION,
        "physical_entry": PILOT_OUTPUT_PHYSICAL_ENTRY_VERSION,
        "stage_accounting": PILOT_OUTPUT_STAGE_ACCOUNTING_VERSION,
        "official_document_placeholder": PILOT_OFFICIAL_DOCUMENT_PLACEHOLDER_VERSION,
        "official_document_placeholder_file": (
            PILOT_OFFICIAL_DOCUMENT_PLACEHOLDER_FILE_VERSION
        ),
        "layout_plan": PILOT_OUTPUT_LAYOUT_PLAN_VERSION,
        "readiness_assessment": PILOT_OUTPUT_READINESS_ASSESSMENT_VERSION,
    },
    "field_sets": {
        "locator_plan": list(PilotOutputLocatorPlanV1.__dataclass_fields__),
        "physical_entry": list(PilotOutputPhysicalEntryV1.__dataclass_fields__),
        "stage_accounting": list(PilotStageOutputAccountingV1.__dataclass_fields__),
        "official_document_placeholder": list(
            PilotOfficialDocumentPlaceholderV1.__dataclass_fields__
        ),
        "official_document_placeholder_file": list(
            PilotOfficialDocumentPlaceholderFileV1.__dataclass_fields__
        ),
        "layout_plan": list(PilotOutputLayoutPlanV1.__dataclass_fields__),
        "readiness": list(PilotOutputReadinessAssessmentV1.__dataclass_fields__),
    },
    "layout": {
        "path_grammar": {
            "relative_canonical_posix": True,
            "ascii_windows_safe": True,
            "maximum_characters": 512,
            "maximum_characters_per_component": 255,
            "casefold_collision_and_component_prefix_collision": "rejected",
            "legacy_data_history_segments": "rejected",
        },
        "history_paths": "exact_plan_relative_artifact_root_prefix",
        "writer_lock": "parent_dot_leaf_strict_history_v2_writer_lock",
        "writer_lock_required": {
            "status": "present_plain_regular",
            "bytes": 1,
            "sha256": _SHA256_ZERO_BYTE,
            "link_count": 1,
        },
        "logical_and_unique_physical_accounting_are_distinct": True,
        "raw_cas_dedup_is_one_physical_entry_with_many_logical_references": True,
        "derivation_vectors": {
            "persistent_writer_lock": {
                "input": "alpha/beta/gamma",
                "output": derive_persistent_writer_lock_locator_v1(
                    "alpha/beta/gamma"
                ),
            },
            "official_bundle": {
                "input": "a" * 64,
                "root": derive_official_bundle_root_v1("a" * 64),
                "locators": list(derive_official_bundle_locators_v1("a" * 64)),
            },
        },
        "physical_entry_source_stage_role_matrix": {
            "run_control": "run_control_stage_ordinal_minus_two",
            "strict_history_physical": "endpoint_or_shard_stage_positive_references",
            "infrastructure_writer_lock": (
                "endpoint_or_shard_stage_exact_role_zero_byte_hash_one_byte_zero_references"
            ),
            "official_document_placeholder": (
                "endpoint_stage_ordinal_minus_one_exact_three_unresolved_roles"
            ),
        },
    },
    "official_document_placeholder": {
        "official_evidence_contract": "unbound",
        "bundle_root_formula": (
            "endpoint-evidence/<lowercase_sha256_verification_plan_hash>/official"
        ),
        "roles_in_order": ["raw_body", "attempt_receipt", "evidence"],
        "relative_names_in_order": [
            "attempt-000.body.bin",
            "attempt-000.receipt.json",
            "evidence.json",
        ],
        "full_locator_helper": "bundle_root_slash_each_relative_name",
        "each_file": {
            "artifact_sha256": "lowercase_sha256",
            "byte_count": "positive_exact_integer",
            "authority": "non_authoritative_physical_metadata_only",
        },
        "schema_status": "unresolved_official_document_evidence_schema",
        "namespace_in_frozen_preflight_fresh_roots": False,
    },
    "global_inventory": {
        "exact_union": [
            "pilot_run_state_final_run_control_inventory_including_result_candidate",
            "prefixed_strict_history_unique_physical_files",
            "persistent_sibling_writer_locks",
            "three_file_endpoint_official_document_unresolved_placeholder_bundle",
        ],
        "exactness_scope": "files_hashes_and_bytes_only",
        "directory_set_observed": False,
        "empty_unexpected_directories_detectable": False,
        "readiness_phase": "post_result_candidate_pre_anchor",
        "missing_unexpected_changed_casefold_duplicate_paths": "typed_stop",
        "hardlink_alias_assurance": _HARDLINK_ASSURANCE,
        "scan_provenance": "exact_frozen_store_scan_required_but_not_crypto_attested",
        "scan_provenance_bound_to_store_session_capability": False,
        "embedded_scan_fields": list(PilotInventoryScanV1.__dataclass_fields__),
        "scan_clock_domain": "exact_frozen_run_clock_domain",
        "scan_epoch_and_monotonic": (
            "both_not_before_last_shard_step_completion"
        ),
    },
    "budgets": {
        "frozen_planned_reservations_are_not_rewritten": True,
        "stage_count": "one_endpoint_plus_shard_count",
        "infrastructure_lock_entries": "stage_count",
        "frozen_shard_entry_credit": "one_per_shard",
        "required_global_file_entry_reservation": (
            "frozen_planned_inventory_entries_plus_one_endpoint_lock"
        ),
        "required_global_output_bytes": (
            "frozen_planned_output_bytes_plus_stage_count_one_byte_locks"
        ),
        "maximum_scan_directory_entries": (
            "nine_fixed_plus_eight_times_stage_count_plus_sum_min_strict_attempt_ceiling_256"
        ),
        "required_scan_traversal_entries": (
            "required_global_file_entry_reservation_plus_maximum_scan_directory_entries"
        ),
        "preflight_post_reservation_headroom_adds_remaining_lock_bytes": True,
        "u5_max_total_output_bytes": (
            "must_cover_frozen_planned_output_bytes_plus_all_one_byte_locks"
        ),
        "u5_inventory_or_traversal_cap": "absent_in_frozen_authorization_terminal_blocker",
    },
    "dependent_receipt_reconciliation": {
        "endpoint": {
            "network_attempts": 2,
            "subject": "three_official_files_plus_one_attempt_strict_physical_graph_no_lock",
            "raw_bytes": "official_body_bytes_plus_strict_actual_total_raw_body_bytes",
            "storage_bytes": "sum_unique_subject_physical_bytes",
            "inventory_entries": 9,
            "inventory_hash": "exact_endpoint_subject_inventory_hash",
        },
        "shard": {
            "inventory": "prefixed_logical_reference_inventory",
            "entries": "logical_reference_count",
            "bytes": "admitted_total_logical_storage_bytes",
            "hash": "prefixed_logical_inventory_hash",
            "observed_internal_sleep_us": (
                "strict_history_observed_monotonic_inter_attempt_sleep_us"
            ),
            "physical_raw_cas": "separately_unique_in_global_inventory",
        },
    },
    "concurrency_boundary": {
        "snapshot": _SNAPSHOT_BOUNDARY,
        "writer": _WRITER_BOUNDARY,
        "hostile_racing_writer_supported": False,
        "atomic_cross_evidence_snapshot_claimed": False,
    },
    "evidence_authority_boundary": {
        "strict_history_input_status": (
            "layout_unresolved_and_non_authoritative_required_exactly"
        ),
        "strict_history_live_reader_provenance_bound": False,
        "inventory_scan_exact_type_is_provenance": False,
        "both_are_exact_terminal_blockers": True,
    },
    "terminal": {
        "official_evidence_contract": "unbound",
        "official_document_evidence_schema": "unbound_three_file_placeholder_only",
        "terminal_publisher": "unbound",
        "frozen_final_anchor_infrastructure_lock_accounting": "unbound_new_version_required",
        "terminal_blockers_exact": list(_TERMINAL_BLOCKERS),
        "require_terminal_compatible": "typed_stop",
        "network": False,
        "filesystem_writes": False,
        "u5_factory": False,
        "concrete_manifest": False,
    },
    "public_signatures": {
        "derive_persistent_writer_lock_locator_v1": {
            "parameters": ["relative_artifact_root"],
            "returns": "str",
        },
        "derive_official_bundle_root_v1": {
            "parameters": ["endpoint_plan_hash"],
            "returns": "str",
        },
        "derive_official_bundle_locators_v1": {
            "parameters": ["endpoint_plan_hash"],
            "returns": "tuple[str,str,str]",
        },
        "build_pilot_output_layout_plan_v1": {
            "parameters": ["manifest"],
            "returns": "PilotOutputLayoutPlanV1",
        },
        "PilotOutputLayoutPlanV1.validate_authorization": {
            "parameters": ["receipt"],
            "returns": "None_or_typed_stop",
        },
        "PilotOutputLayoutPlanV1.validate_preflight": {
            "parameters": ["receipt"],
            "returns": "None_or_typed_stop",
        },
        "build_pilot_stage_output_accounting_v1": {
            "keyword_only_parameters": [
                "pilot_manifest_hash",
                "locator_plan",
                "evidence",
                "official_document_placeholder=None",
            ],
            "returns": "PilotStageOutputAccountingV1",
        },
        "derive_expected_pilot_output_inventory_v1": {
            "keyword_only_parameters": ["state", "stage_accounting"],
            "returns": "tuple[PilotOutputPhysicalEntryV1,...]",
        },
        "assess_pilot_output_layout_v1": {
            "keyword_only_parameters": [
                "manifest",
                "state",
                "inventory_scan",
                "stage_evidences",
                "official_document_placeholder=None",
            ],
            "returns": "PilotOutputReadinessAssessmentV1",
        },
    },
    "parsers": {
        "exact_key_sets": True,
        "nested_source_objects_included": True,
        "all_repeated_hash_count_byte_and_summary_projections_recomputed": True,
        "readiness_replays_authorization_preflights_and_stage_receipts": True,
        "frozen_objects": True,
    },
}


def pilot_output_layout_contract_hash() -> str:
    digest = _sha256_payload(_CONTRACT_SCHEMA)
    if _PINNED_CONTRACT_HASH and digest != _PINNED_CONTRACT_HASH:
        raise PilotOutputLayoutContractError(
            "pilot_output_layout_contract_changed_without_version_bump"
        )
    return digest


__all__ = [
    "PILOT_OFFICIAL_DOCUMENT_PLACEHOLDER_FILE_VERSION",
    "PILOT_OFFICIAL_DOCUMENT_PLACEHOLDER_VERSION",
    "PILOT_OUTPUT_LAYOUT_CONTRACT_VERSION",
    "PILOT_OUTPUT_LAYOUT_PLAN_VERSION",
    "PILOT_OUTPUT_LOCATOR_PLAN_VERSION",
    "PILOT_OUTPUT_PHYSICAL_ENTRY_VERSION",
    "PILOT_OUTPUT_READINESS_ASSESSMENT_VERSION",
    "PILOT_OUTPUT_STAGE_ACCOUNTING_VERSION",
    "PilotOfficialDocumentPlaceholderFileV1",
    "PilotOfficialDocumentPlaceholderV1",
    "PilotOutputLayoutBudgetStop",
    "PilotOutputLayoutContractError",
    "PilotOutputLayoutError",
    "PilotOutputLayoutInventoryStop",
    "PilotOutputLayoutPlanV1",
    "PilotOutputLayoutTerminalStop",
    "PilotOutputLocatorPlanV1",
    "PilotOutputPhysicalEntryV1",
    "PilotOutputReadinessAssessmentV1",
    "PilotStageOutputAccountingV1",
    "assess_pilot_output_layout_v1",
    "build_pilot_output_layout_plan_v1",
    "build_pilot_stage_output_accounting_v1",
    "derive_expected_pilot_output_inventory_v1",
    "derive_official_bundle_locators_v1",
    "derive_official_bundle_root_v1",
    "derive_persistent_writer_lock_locator_v1",
    "pilot_output_layout_contract_hash",
]
