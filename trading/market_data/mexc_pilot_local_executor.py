"""Pure, offline foundation for an explicitly supplied MEXC pilot executor.

This module defines identities, dependency protocols, non-authoritative stage
drafts, and an explicit concrete-manifest builder.  It deliberately contains
no coordinator, artifact store, authorization factory, network implementation,
environment discovery, repository discovery, or executable default.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Protocol

from trading.market_data.mexc_pilot_run import (
    EndpointVerificationPlanV1,
    MexcPublicQaPilotRunManifestV1,
    PilotDiskPreflightReceiptV1,
    PilotGlobalBudgetsV1,
    PilotNetworkIntentV1,
    PilotShardPlanV1,
    U5PublicPilotAuthorizationReceiptV1,
    pilot_run_contract_hash,
)
from trading.market_data.strict_history import _frozen_contract_hash
from trading.market_data.strict_history_v2 import HistoryRangeRequestV2


PILOT_LOCAL_EXECUTOR_CONTRACT_VERSION = "mexc_public_qa_pilot_local_executor_v1"
PILOT_CLOCK_PROTOCOL_VERSION = "mexc_public_qa_pilot_clock_protocol_v1"
PILOT_DETACHED_ANCHOR_SINK_PROTOCOL_VERSION = (
    "mexc_public_qa_pilot_detached_anchor_sink_protocol_v1"
)
PILOT_ENDPOINT_STAGE_RUNNER_PROTOCOL_VERSION = (
    "mexc_public_qa_pilot_endpoint_stage_runner_protocol_v1"
)
PILOT_SHARD_STAGE_RUNNER_PROTOCOL_VERSION = (
    "mexc_public_qa_pilot_shard_stage_runner_protocol_v1"
)
PILOT_EXECUTOR_BINDINGS_VERSION = "mexc_public_qa_pilot_executor_bindings_v1"
PILOT_ENDPOINT_EXECUTOR_BINDING_VERSION = (
    "mexc_public_qa_pilot_endpoint_executor_binding_v1"
)
PILOT_SHARD_EXECUTOR_BINDING_VERSION = (
    "mexc_public_qa_pilot_shard_executor_binding_v1"
)
PILOT_DETACHED_ANCHOR_SUBJECT_VERSION = (
    "mexc_public_qa_pilot_detached_anchor_subject_v1"
)
PILOT_DETACHED_ANCHOR_EVIDENCE_VERSION = (
    "mexc_public_qa_pilot_detached_anchor_evidence_v1"
)
PILOT_ENDPOINT_STAGE_DRAFT_VERSION = "mexc_public_qa_pilot_endpoint_stage_draft_v1"
PILOT_SHARD_STAGE_DRAFT_VERSION = "mexc_public_qa_pilot_shard_stage_draft_v1"
PILOT_STAGE_FAILURE_DRAFT_VERSION = "mexc_public_qa_pilot_stage_failure_draft_v1"

_PINNED_CONTRACT_HASH = (
    "72c206bc2f22a8101a7d6fdc97458e865a6c4c3e5ed7290c64c1ca8c3594fc31"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_WINDOWS_ILLEGAL_PATH_CHARACTER_RE = re.compile(r'[<>:"|?*]')
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


class PilotLocalExecutorError(RuntimeError):
    """Base error for the offline local-executor foundation."""


class PilotLocalExecutorContractError(PilotLocalExecutorError):
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
        raise PilotLocalExecutorContractError(
            "pilot_local_executor_payload_is_not_canonical_json"
        ) from exc


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PilotLocalExecutorContractError(f"{field}_is_invalid")
    return value


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise PilotLocalExecutorContractError(f"{field}_is_invalid")
    return value


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PilotLocalExecutorContractError(f"{field}_is_invalid")
    return value


def _relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or len(value) > 240:
        raise PilotLocalExecutorContractError(f"{field}_is_invalid")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PilotLocalExecutorContractError(f"{field}_is_invalid")
    lowered = tuple(part.lower() for part in path.parts)
    if any(
        any(ord(character) < 32 for character in part)
        or _WINDOWS_ILLEGAL_PATH_CHARACTER_RE.search(part) is not None
        for part in path.parts
    ):
        raise PilotLocalExecutorContractError(
            f"{field}_contains_illegal_windows_character"
        )
    if any(
        part.endswith((".", " "))
        or part.rstrip(" .").split(".", 1)[0].upper()
        in _WINDOWS_RESERVED_BASENAMES
        for part in path.parts
    ):
        raise PilotLocalExecutorContractError(f"{field}_uses_reserved_windows_name")
    if any(
        lowered[index : index + 2] == ("data", "history")
        for index in range(len(lowered) - 1)
    ):
        raise PilotLocalExecutorContractError(f"{field}_uses_legacy_history")
    rendered = path.as_posix()
    if rendered != value:
        raise PilotLocalExecutorContractError(f"{field}_is_not_canonical")
    return rendered


def _exact_mapping(
    payload: object,
    expected: frozenset[str],
    *,
    code: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise PilotLocalExecutorContractError(code)
    return dict(payload)


class PilotClock(Protocol):
    """One explicit epoch/monotonic clock domain; never a module default."""

    @property
    def contract_version(self) -> str: ...

    @property
    def contract_hash(self) -> str: ...

    @property
    def clock_domain_id(self) -> str: ...

    def epoch_us(self) -> int: ...

    def monotonic_us(self) -> int: ...

    def sleep_us(self, duration_us: int) -> None: ...


@dataclass(frozen=True)
class DetachedAnchorSubjectV1:
    manifest_hash: str
    subject_kind: str
    subject_hash: str
    clock_contract_version: str
    clock_contract_hash: str
    clock_domain_id: str
    requested_at_us: int
    requested_monotonic_us: int
    contract_version: str = PILOT_DETACHED_ANCHOR_SUBJECT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_DETACHED_ANCHOR_SUBJECT_VERSION:
            raise PilotLocalExecutorContractError(
                "pilot_detached_anchor_subject_version_mismatch"
            )
        _digest(self.manifest_hash, field="pilot_anchor_subject_manifest_hash")
        _identifier(self.subject_kind, field="pilot_anchor_subject_kind")
        _digest(self.subject_hash, field="pilot_anchor_subject_hash")
        _identifier(
            self.clock_contract_version,
            field="pilot_anchor_subject_clock_contract_version",
        )
        _digest(
            self.clock_contract_hash,
            field="pilot_anchor_subject_clock_contract_hash",
        )
        _identifier(self.clock_domain_id, field="pilot_anchor_subject_clock_domain_id")
        _strict_int(
            self.requested_at_us,
            field="pilot_anchor_subject_requested_at_us",
            minimum=1,
        )
        _strict_int(
            self.requested_monotonic_us,
            field="pilot_anchor_subject_requested_monotonic_us",
        )

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "DetachedAnchorSubjectV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_detached_anchor_subject_schema_mismatch",
        )
        return cls(**values)

    @property
    def subject_receipt_hash(self) -> str:
        return _frozen_contract_hash(self)


@dataclass(frozen=True)
class DetachedAnchorEvidenceV1:
    subject_receipt_hash: str
    anchor_sink_contract_version: str
    anchor_sink_contract_hash: str
    anchor_domain_id: str
    clock_contract_version: str
    clock_contract_hash: str
    clock_domain_id: str
    evidence_hash: str
    anchored_at_us: int
    anchored_monotonic_us: int
    contract_version: str = PILOT_DETACHED_ANCHOR_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_DETACHED_ANCHOR_EVIDENCE_VERSION:
            raise PilotLocalExecutorContractError(
                "pilot_detached_anchor_evidence_version_mismatch"
            )
        _digest(
            self.subject_receipt_hash,
            field="pilot_anchor_evidence_subject_receipt_hash",
        )
        _identifier(
            self.anchor_sink_contract_version,
            field="pilot_anchor_sink_contract_version",
        )
        _digest(
            self.anchor_sink_contract_hash,
            field="pilot_anchor_sink_contract_hash",
        )
        _identifier(self.anchor_domain_id, field="pilot_anchor_domain_id")
        _identifier(
            self.clock_contract_version,
            field="pilot_anchor_evidence_clock_contract_version",
        )
        _digest(
            self.clock_contract_hash,
            field="pilot_anchor_evidence_clock_contract_hash",
        )
        _identifier(
            self.clock_domain_id,
            field="pilot_anchor_evidence_clock_domain_id",
        )
        _digest(self.evidence_hash, field="pilot_anchor_evidence_hash")
        _strict_int(
            self.anchored_at_us,
            field="pilot_anchor_evidence_anchored_at_us",
            minimum=1,
        )
        _strict_int(
            self.anchored_monotonic_us,
            field="pilot_anchor_evidence_anchored_monotonic_us",
        )

    def validate_for(
        self,
        subject: DetachedAnchorSubjectV1,
        *,
        anchor_sink_contract_version: str,
        anchor_sink_contract_hash: str,
        anchor_domain_id: str,
    ) -> None:
        if (
            self.subject_receipt_hash != subject.subject_receipt_hash
            or self.anchor_sink_contract_version != anchor_sink_contract_version
            or self.anchor_sink_contract_hash != anchor_sink_contract_hash
            or self.anchor_domain_id != anchor_domain_id
            or self.clock_contract_version != subject.clock_contract_version
            or self.clock_contract_hash != subject.clock_contract_hash
            or self.clock_domain_id != subject.clock_domain_id
            or self.anchored_at_us < subject.requested_at_us
            or self.anchored_monotonic_us < subject.requested_monotonic_us
        ):
            raise PilotLocalExecutorContractError(
                "pilot_detached_anchor_evidence_binding_mismatch"
            )

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "DetachedAnchorEvidenceV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_detached_anchor_evidence_schema_mismatch",
        )
        return cls(**values)


class DetachedAnchorSink(Protocol):
    """Explicit detached evidence sink; no implementation is bundled here."""

    @property
    def contract_version(self) -> str: ...

    @property
    def contract_hash(self) -> str: ...

    @property
    def domain_id(self) -> str: ...

    def create_once(
        self,
        subject: DetachedAnchorSubjectV1,
        *,
        clock: PilotClock,
    ) -> DetachedAnchorEvidenceV1: ...

    def reload(self, evidence_hash: str) -> DetachedAnchorEvidenceV1: ...


@dataclass(frozen=True)
class EndpointStageDraftV1:
    manifest_hash: str
    authorization_receipt_hash: str
    network_intent_hash: str
    clock_domain_id: str
    stage_started_at_us: int
    stage_completed_at_us: int
    stage_started_monotonic_us: int
    stage_completed_monotonic_us: int
    official_document_evidence_relative_path: str
    official_document_evidence_hash: str
    official_document_request_started_at_us: int
    official_document_fetched_at_us: int
    official_document_request_started_monotonic_us: int
    official_document_fetched_monotonic_us: int
    live_probe_store_relative_root: str
    live_history_manifest_hash: str
    live_probe_started_at_us: int
    live_probe_completed_at_us: int
    live_probe_started_monotonic_us: int
    live_probe_completed_monotonic_us: int
    contract_version: str = PILOT_ENDPOINT_STAGE_DRAFT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_ENDPOINT_STAGE_DRAFT_VERSION:
            raise PilotLocalExecutorContractError(
                "pilot_endpoint_stage_draft_version_mismatch"
            )
        for field in (
            "manifest_hash",
            "authorization_receipt_hash",
            "network_intent_hash",
            "official_document_evidence_hash",
            "live_history_manifest_hash",
        ):
            _digest(getattr(self, field), field=f"pilot_endpoint_draft_{field}")
        _identifier(self.clock_domain_id, field="pilot_endpoint_draft_clock_domain_id")
        object.__setattr__(
            self,
            "official_document_evidence_relative_path",
            _relative_path(
                self.official_document_evidence_relative_path,
                field="pilot_endpoint_draft_official_evidence_path",
            ),
        )
        object.__setattr__(
            self,
            "live_probe_store_relative_root",
            _relative_path(
                self.live_probe_store_relative_root,
                field="pilot_endpoint_draft_live_probe_root",
            ),
        )
        epoch = tuple(
            _strict_int(getattr(self, field), field=f"pilot_endpoint_draft_{field}", minimum=1)
            for field in (
                "stage_started_at_us",
                "official_document_request_started_at_us",
                "official_document_fetched_at_us",
                "live_probe_started_at_us",
                "live_probe_completed_at_us",
                "stage_completed_at_us",
            )
        )
        monotonic = tuple(
            _strict_int(getattr(self, field), field=f"pilot_endpoint_draft_{field}")
            for field in (
                "stage_started_monotonic_us",
                "official_document_request_started_monotonic_us",
                "official_document_fetched_monotonic_us",
                "live_probe_started_monotonic_us",
                "live_probe_completed_monotonic_us",
                "stage_completed_monotonic_us",
            )
        )
        if epoch != tuple(sorted(epoch)) or monotonic != tuple(sorted(monotonic)):
            raise PilotLocalExecutorContractError(
                "pilot_endpoint_stage_draft_timing_is_invalid"
            )

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "EndpointStageDraftV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_endpoint_stage_draft_schema_mismatch",
        )
        return cls(**values)


@dataclass(frozen=True)
class ShardStageDraftV1:
    manifest_hash: str
    network_intent_hash: str
    ordinal: int
    clock_domain_id: str
    step_started_at_us: int
    step_completed_at_us: int
    step_started_monotonic_us: int
    step_completed_monotonic_us: int
    history_manifest_hash: str
    contract_version: str = PILOT_SHARD_STAGE_DRAFT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_SHARD_STAGE_DRAFT_VERSION:
            raise PilotLocalExecutorContractError(
                "pilot_shard_stage_draft_version_mismatch"
            )
        _digest(self.manifest_hash, field="pilot_shard_draft_manifest_hash")
        _digest(self.network_intent_hash, field="pilot_shard_draft_network_intent_hash")
        _strict_int(self.ordinal, field="pilot_shard_draft_ordinal")
        _identifier(self.clock_domain_id, field="pilot_shard_draft_clock_domain_id")
        _digest(
            self.history_manifest_hash,
            field="pilot_shard_draft_history_manifest_hash",
        )
        start = _strict_int(
            self.step_started_at_us,
            field="pilot_shard_draft_step_started_at_us",
            minimum=1,
        )
        end = _strict_int(
            self.step_completed_at_us,
            field="pilot_shard_draft_step_completed_at_us",
            minimum=1,
        )
        mono_start = _strict_int(
            self.step_started_monotonic_us,
            field="pilot_shard_draft_step_started_monotonic_us",
        )
        mono_end = _strict_int(
            self.step_completed_monotonic_us,
            field="pilot_shard_draft_step_completed_monotonic_us",
        )
        if end < start or mono_end < mono_start:
            raise PilotLocalExecutorContractError(
                "pilot_shard_stage_draft_timing_is_invalid"
            )

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "ShardStageDraftV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_shard_stage_draft_schema_mismatch",
        )
        return cls(**values)


@dataclass(frozen=True)
class StageFailureDraftV1:
    manifest_hash: str
    authorization_receipt_hash: str
    network_intent_hash: str
    stage: str
    ordinal: int
    clock_domain_id: str
    step_started_at_us: int
    step_completed_at_us: int
    step_started_monotonic_us: int
    step_completed_monotonic_us: int
    error_code: str
    error_evidence_hash: str
    contract_version: str = PILOT_STAGE_FAILURE_DRAFT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_STAGE_FAILURE_DRAFT_VERSION:
            raise PilotLocalExecutorContractError(
                "pilot_stage_failure_draft_version_mismatch"
            )
        for field in (
            "manifest_hash",
            "authorization_receipt_hash",
            "network_intent_hash",
            "error_evidence_hash",
        ):
            _digest(getattr(self, field), field=f"pilot_failure_draft_{field}")
        if self.stage not in {"endpoint_verification", "shard_acquisition"}:
            raise PilotLocalExecutorContractError(
                "pilot_failure_draft_stage_is_invalid"
            )
        if type(self.ordinal) is not int or (
            self.stage == "endpoint_verification" and self.ordinal != -1
        ) or (self.stage == "shard_acquisition" and self.ordinal < 0):
            raise PilotLocalExecutorContractError(
                "pilot_failure_draft_ordinal_is_invalid"
            )
        _identifier(self.clock_domain_id, field="pilot_failure_draft_clock_domain_id")
        _identifier(self.error_code, field="pilot_failure_draft_error_code")
        start = _strict_int(
            self.step_started_at_us,
            field="pilot_failure_draft_step_started_at_us",
            minimum=1,
        )
        end = _strict_int(
            self.step_completed_at_us,
            field="pilot_failure_draft_step_completed_at_us",
            minimum=1,
        )
        mono_start = _strict_int(
            self.step_started_monotonic_us,
            field="pilot_failure_draft_step_started_monotonic_us",
        )
        mono_end = _strict_int(
            self.step_completed_monotonic_us,
            field="pilot_failure_draft_step_completed_monotonic_us",
        )
        if end < start or mono_end < mono_start:
            raise PilotLocalExecutorContractError(
                "pilot_failure_draft_timing_is_invalid"
            )

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "StageFailureDraftV1":
        values = _exact_mapping(
            payload,
            frozenset(cls.__dataclass_fields__),
            code="pilot_stage_failure_draft_schema_mismatch",
        )
        return cls(**values)


class EndpointStageRunner(Protocol):
    """Injected endpoint runner; production and fake implementations are external."""

    @property
    def contract_version(self) -> str: ...

    @property
    def contract_hash(self) -> str: ...

    def execute(
        self,
        *,
        manifest: MexcPublicQaPilotRunManifestV1,
        authorization: U5PublicPilotAuthorizationReceiptV1,
        preflight: PilotDiskPreflightReceiptV1,
        network_intent: PilotNetworkIntentV1,
        artifact_root: Path,
        clock: PilotClock,
    ) -> EndpointStageDraftV1 | StageFailureDraftV1: ...


class ShardStageRunner(Protocol):
    """Injected one-shard runner; it cannot select authoritative totals."""

    @property
    def contract_version(self) -> str: ...

    @property
    def contract_hash(self) -> str: ...

    def execute(
        self,
        *,
        manifest: MexcPublicQaPilotRunManifestV1,
        authorization: U5PublicPilotAuthorizationReceiptV1,
        preflight: PilotDiskPreflightReceiptV1,
        network_intent: PilotNetworkIntentV1,
        shard_plan: PilotShardPlanV1,
        artifact_root: Path,
        clock: PilotClock,
    ) -> ShardStageDraftV1 | StageFailureDraftV1: ...


@dataclass(frozen=True)
class PilotExecutorBindingsV1:
    coordinator_contract_version: str
    coordinator_contract_hash: str
    local_store_contract_version: str
    local_store_contract_hash: str
    clock_contract_version: str
    clock_contract_hash: str
    detached_anchor_sink_contract_version: str
    detached_anchor_sink_contract_hash: str
    endpoint_runner_contract_version: str
    endpoint_runner_contract_hash: str
    shard_runner_contract_version: str
    shard_runner_contract_hash: str
    contract_version: str = PILOT_EXECUTOR_BINDINGS_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_EXECUTOR_BINDINGS_VERSION:
            raise PilotLocalExecutorContractError(
                "pilot_executor_bindings_version_mismatch"
            )
        for prefix in (
            "coordinator",
            "local_store",
            "clock",
            "detached_anchor_sink",
            "endpoint_runner",
            "shard_runner",
        ):
            _identifier(
                getattr(self, f"{prefix}_contract_version"),
                field=f"pilot_{prefix}_contract_version",
            )
            _digest(
                getattr(self, f"{prefix}_contract_hash"),
                field=f"pilot_{prefix}_contract_hash",
            )

    @staticmethod
    def _identity(version: str, digest: str) -> dict[str, str]:
        return {"contract_version": version, "contract_hash": digest}

    @property
    def endpoint_verifier_binding_payload(self) -> dict[str, object]:
        return {
            "domain": PILOT_ENDPOINT_EXECUTOR_BINDING_VERSION,
            "bindings_contract_version": self.contract_version,
            "protocol_versions": {
                "clock": PILOT_CLOCK_PROTOCOL_VERSION,
                "detached_anchor_sink": PILOT_DETACHED_ANCHOR_SINK_PROTOCOL_VERSION,
                "stage_runner": PILOT_ENDPOINT_STAGE_RUNNER_PROTOCOL_VERSION,
            },
            "coordinator": self._identity(
                self.coordinator_contract_version,
                self.coordinator_contract_hash,
            ),
            "local_store": self._identity(
                self.local_store_contract_version,
                self.local_store_contract_hash,
            ),
            "clock": self._identity(
                self.clock_contract_version,
                self.clock_contract_hash,
            ),
            "detached_anchor_sink": self._identity(
                self.detached_anchor_sink_contract_version,
                self.detached_anchor_sink_contract_hash,
            ),
            "endpoint_runner": self._identity(
                self.endpoint_runner_contract_version,
                self.endpoint_runner_contract_hash,
            ),
        }

    @property
    def shard_executor_binding_payload(self) -> dict[str, object]:
        return {
            "domain": PILOT_SHARD_EXECUTOR_BINDING_VERSION,
            "bindings_contract_version": self.contract_version,
            "protocol_versions": {
                "clock": PILOT_CLOCK_PROTOCOL_VERSION,
                "detached_anchor_sink": PILOT_DETACHED_ANCHOR_SINK_PROTOCOL_VERSION,
                "stage_runner": PILOT_SHARD_STAGE_RUNNER_PROTOCOL_VERSION,
            },
            "coordinator": self._identity(
                self.coordinator_contract_version,
                self.coordinator_contract_hash,
            ),
            "local_store": self._identity(
                self.local_store_contract_version,
                self.local_store_contract_hash,
            ),
            "clock": self._identity(
                self.clock_contract_version,
                self.clock_contract_hash,
            ),
            "detached_anchor_sink": self._identity(
                self.detached_anchor_sink_contract_version,
                self.detached_anchor_sink_contract_hash,
            ),
            "shard_runner": self._identity(
                self.shard_runner_contract_version,
                self.shard_runner_contract_hash,
            ),
        }

    @property
    def endpoint_verifier_binding_version(self) -> str:
        return PILOT_ENDPOINT_EXECUTOR_BINDING_VERSION

    @property
    def endpoint_verifier_binding_hash(self) -> str:
        return _sha256_payload(self.endpoint_verifier_binding_payload)

    @property
    def shard_executor_binding_version(self) -> str:
        return PILOT_SHARD_EXECUTOR_BINDING_VERSION

    @property
    def shard_executor_binding_hash(self) -> str:
        return _sha256_payload(self.shard_executor_binding_payload)

    @property
    def bindings_hash(self) -> str:
        return _frozen_contract_hash(self)

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "coordinator": self._identity(
                self.coordinator_contract_version,
                self.coordinator_contract_hash,
            ),
            "local_store": self._identity(
                self.local_store_contract_version,
                self.local_store_contract_hash,
            ),
            "clock": self._identity(
                self.clock_contract_version,
                self.clock_contract_hash,
            ),
            "detached_anchor_sink": self._identity(
                self.detached_anchor_sink_contract_version,
                self.detached_anchor_sink_contract_hash,
            ),
            "endpoint_runner": self._identity(
                self.endpoint_runner_contract_version,
                self.endpoint_runner_contract_hash,
            ),
            "shard_runner": self._identity(
                self.shard_runner_contract_version,
                self.shard_runner_contract_hash,
            ),
            "endpoint_verifier_binding": {
                "contract_version": self.endpoint_verifier_binding_version,
                "contract_hash": self.endpoint_verifier_binding_hash,
            },
            "shard_executor_binding": {
                "contract_version": self.shard_executor_binding_version,
                "contract_hash": self.shard_executor_binding_hash,
            },
        }

    @classmethod
    def from_dict(cls, payload: object) -> "PilotExecutorBindingsV1":
        values = _exact_mapping(
            payload,
            frozenset(
                {
                    "contract_version",
                    "coordinator",
                    "local_store",
                    "clock",
                    "detached_anchor_sink",
                    "endpoint_runner",
                    "shard_runner",
                    "endpoint_verifier_binding",
                    "shard_executor_binding",
                }
            ),
            code="pilot_executor_bindings_schema_mismatch",
        )

        def identity(name: str) -> tuple[str, str]:
            item = _exact_mapping(
                values.pop(name),
                frozenset({"contract_version", "contract_hash"}),
                code=f"pilot_executor_{name}_identity_schema_mismatch",
            )
            return item["contract_version"], item["contract_hash"]

        coordinator = identity("coordinator")
        local_store = identity("local_store")
        clock = identity("clock")
        anchor = identity("detached_anchor_sink")
        endpoint = identity("endpoint_runner")
        shard = identity("shard_runner")
        repeated_endpoint = identity("endpoint_verifier_binding")
        repeated_shard = identity("shard_executor_binding")
        result = cls(
            coordinator_contract_version=coordinator[0],
            coordinator_contract_hash=coordinator[1],
            local_store_contract_version=local_store[0],
            local_store_contract_hash=local_store[1],
            clock_contract_version=clock[0],
            clock_contract_hash=clock[1],
            detached_anchor_sink_contract_version=anchor[0],
            detached_anchor_sink_contract_hash=anchor[1],
            endpoint_runner_contract_version=endpoint[0],
            endpoint_runner_contract_hash=endpoint[1],
            shard_runner_contract_version=shard[0],
            shard_runner_contract_hash=shard[1],
            contract_version=values.pop("contract_version"),
        )
        if repeated_endpoint != (
            result.endpoint_verifier_binding_version,
            result.endpoint_verifier_binding_hash,
        ) or repeated_shard != (
            result.shard_executor_binding_version,
            result.shard_executor_binding_hash,
        ):
            raise PilotLocalExecutorContractError(
                "pilot_executor_composite_binding_mismatch"
            )
        if values:
            raise PilotLocalExecutorContractError(
                "pilot_executor_bindings_parser_left_unknown_fields"
            )
        return result


def build_concrete_pilot_manifest_v1(
    *,
    repository_commit: str,
    repository_tree_receipt_hash: str,
    created_at_us: int,
    parent_master_plan_path: str,
    parent_master_plan_sha256: str,
    parent_adr_path: str,
    parent_adr_sha256: str,
    output_root_locator: str,
    endpoint_probe_request: HistoryRangeRequestV2,
    endpoint_relative_artifact_root: str,
    official_reference_url: str,
    endpoint_max_network_attempts: int,
    endpoint_max_total_raw_body_bytes: int,
    endpoint_max_total_storage_bytes: int,
    endpoint_max_runtime_us: int,
    endpoint_max_total_sleep_us: int,
    ordered_shards: tuple[PilotShardPlanV1, ...],
    budgets: PilotGlobalBudgetsV1,
    executor_bindings: PilotExecutorBindingsV1,
) -> MexcPublicQaPilotRunManifestV1:
    """Build one exact manifest without discovery, defaults, or publication."""

    if not isinstance(executor_bindings, PilotExecutorBindingsV1):
        raise PilotLocalExecutorContractError(
            "pilot_manifest_executor_bindings_are_invalid"
        )
    if not isinstance(endpoint_probe_request, HistoryRangeRequestV2):
        raise PilotLocalExecutorContractError(
            "pilot_manifest_endpoint_probe_request_is_invalid"
        )
    if not isinstance(ordered_shards, tuple) or not ordered_shards or not all(
        isinstance(item, PilotShardPlanV1) for item in ordered_shards
    ):
        raise PilotLocalExecutorContractError(
            "pilot_manifest_ordered_shards_are_invalid"
        )
    if not isinstance(budgets, PilotGlobalBudgetsV1):
        raise PilotLocalExecutorContractError("pilot_manifest_budgets_are_invalid")

    endpoint = EndpointVerificationPlanV1(
        probe_request=endpoint_probe_request,
        relative_artifact_root=endpoint_relative_artifact_root,
        official_reference_url=official_reference_url,
        verifier_contract_version=(
            executor_bindings.endpoint_verifier_binding_version
        ),
        verifier_contract_hash=executor_bindings.endpoint_verifier_binding_hash,
        max_network_attempts=endpoint_max_network_attempts,
        max_total_raw_body_bytes=endpoint_max_total_raw_body_bytes,
        max_total_storage_bytes=endpoint_max_total_storage_bytes,
        max_runtime_us=endpoint_max_runtime_us,
        max_total_sleep_us=endpoint_max_total_sleep_us,
    )
    return MexcPublicQaPilotRunManifestV1(
        repository_commit=repository_commit,
        repository_tree_receipt_hash=repository_tree_receipt_hash,
        created_at_us=created_at_us,
        parent_master_plan_path=parent_master_plan_path,
        parent_master_plan_sha256=parent_master_plan_sha256,
        parent_adr_path=parent_adr_path,
        parent_adr_sha256=parent_adr_sha256,
        output_root_locator=output_root_locator,
        shard_executor_contract_version=(
            executor_bindings.shard_executor_binding_version
        ),
        shard_executor_contract_hash=executor_bindings.shard_executor_binding_hash,
        endpoint_verification=endpoint,
        shards=ordered_shards,
        budgets=budgets,
    )


_CONTRACT_SCHEMA = {
    "contract_version": PILOT_LOCAL_EXECUTOR_CONTRACT_VERSION,
    "component_versions": {
        "clock_protocol": PILOT_CLOCK_PROTOCOL_VERSION,
        "detached_anchor_sink_protocol": (
            PILOT_DETACHED_ANCHOR_SINK_PROTOCOL_VERSION
        ),
        "endpoint_stage_runner_protocol": (
            PILOT_ENDPOINT_STAGE_RUNNER_PROTOCOL_VERSION
        ),
        "shard_stage_runner_protocol": PILOT_SHARD_STAGE_RUNNER_PROTOCOL_VERSION,
        "executor_bindings": PILOT_EXECUTOR_BINDINGS_VERSION,
        "endpoint_executor_binding": PILOT_ENDPOINT_EXECUTOR_BINDING_VERSION,
        "shard_executor_binding": PILOT_SHARD_EXECUTOR_BINDING_VERSION,
        "detached_anchor_subject": PILOT_DETACHED_ANCHOR_SUBJECT_VERSION,
        "detached_anchor_evidence": PILOT_DETACHED_ANCHOR_EVIDENCE_VERSION,
        "endpoint_stage_draft": PILOT_ENDPOINT_STAGE_DRAFT_VERSION,
        "shard_stage_draft": PILOT_SHARD_STAGE_DRAFT_VERSION,
        "stage_failure_draft": PILOT_STAGE_FAILURE_DRAFT_VERSION,
    },
    "dependency_hashes": {"pilot_run": pilot_run_contract_hash()},
    "canonicalization": {
        "encoding": "utf8",
        "json_keys": "sorted",
        "separators": [",", ":"],
        "ensure_ascii": False,
        "nonfinite_numbers": "rejected",
        "parsers": "exact_key_set_then_frozen_constructor_validation",
    },
    "scalar_contracts": {
        "digest": "lowercase_sha256_exactly_64_hex",
        "identifier": "lowercase_letter_then_0_to_127_lowercase_alnum_dot_dash_underscore",
        "integer": "python_int_exact_bool_rejected",
        "relative_path": {
            "separator": "/",
            "maximum_characters": 240,
            "absolute_empty_root_dot_dot_and_parent_segments": "rejected",
            "backslash_and_ascii_controls": "rejected",
            "windows_illegal_characters": '<>:"|?*',
            "windows_trailing_dot_or_space": "rejected",
            "windows_reserved_basenames": sorted(_WINDOWS_RESERVED_BASENAMES),
            "legacy_data_history_pair_case_insensitive": "rejected",
            "canonical_posix_rerender_required": True,
        },
    },
    "field_sets": {
        "executor_bindings": list(PilotExecutorBindingsV1.__dataclass_fields__),
        "detached_anchor_subject": list(
            DetachedAnchorSubjectV1.__dataclass_fields__
        ),
        "detached_anchor_evidence": list(
            DetachedAnchorEvidenceV1.__dataclass_fields__
        ),
        "endpoint_stage_draft": list(EndpointStageDraftV1.__dataclass_fields__),
        "shard_stage_draft": list(ShardStageDraftV1.__dataclass_fields__),
        "stage_failure_draft": list(StageFailureDraftV1.__dataclass_fields__),
    },
    "object_contracts": {
        "detached_anchor_subject": {
            "frozen": True,
            "digests": [
                "manifest_hash",
                "subject_hash",
                "clock_contract_hash",
            ],
            "identifiers": [
                "subject_kind",
                "clock_contract_version",
                "clock_domain_id",
            ],
            "exact_integers": {
                "requested_at_us": {"minimum": 1},
                "requested_monotonic_us": {"minimum": 0},
            },
            "identity": "sha256_of_exact_canonical_object",
        },
        "detached_anchor_evidence": {
            "frozen": True,
            "digests": [
                "subject_receipt_hash",
                "anchor_sink_contract_hash",
                "clock_contract_hash",
                "evidence_hash",
            ],
            "identifiers": [
                "anchor_sink_contract_version",
                "anchor_domain_id",
                "clock_contract_version",
                "clock_domain_id",
            ],
            "exact_integers": {
                "anchored_at_us": {"minimum": 1},
                "anchored_monotonic_us": {"minimum": 0},
            },
            "validation": (
                "subject_receipt_hash_sink_version_sink_hash_sink_domain_"
                "subject_clock_version_subject_clock_hash_subject_clock_domain_"
                "and_non_decreasing_epoch_and_monotonic"
            ),
        },
        "endpoint_stage_draft": {
            "frozen": True,
            "digests": [
                "manifest_hash",
                "authorization_receipt_hash",
                "network_intent_hash",
                "official_document_evidence_hash",
                "live_history_manifest_hash",
            ],
            "identifiers": ["clock_domain_id"],
            "relative_paths": [
                "official_document_evidence_relative_path",
                "live_probe_store_relative_root",
            ],
            "epoch_order": [
                "stage_started_at_us",
                "official_document_request_started_at_us",
                "official_document_fetched_at_us",
                "live_probe_started_at_us",
                "live_probe_completed_at_us",
                "stage_completed_at_us",
            ],
            "monotonic_order": [
                "stage_started_monotonic_us",
                "official_document_request_started_monotonic_us",
                "official_document_fetched_monotonic_us",
                "live_probe_started_monotonic_us",
                "live_probe_completed_monotonic_us",
                "stage_completed_monotonic_us",
            ],
            "epoch_minimum": 1,
            "monotonic_minimum": 0,
        },
        "shard_stage_draft": {
            "frozen": True,
            "digests": [
                "manifest_hash",
                "network_intent_hash",
                "history_manifest_hash",
            ],
            "identifiers": ["clock_domain_id"],
            "exact_integer": {"ordinal": {"minimum": 0}},
            "epoch_fields": ["step_started_at_us", "step_completed_at_us"],
            "monotonic_fields": [
                "step_started_monotonic_us",
                "step_completed_monotonic_us",
            ],
            "epoch_minimum": 1,
            "monotonic_minimum": 0,
            "timing_order": "end_not_before_start_in_both_clock_domains",
        },
        "stage_failure_draft": {
            "frozen": True,
            "digests": [
                "manifest_hash",
                "authorization_receipt_hash",
                "network_intent_hash",
                "error_evidence_hash",
            ],
            "identifiers": ["clock_domain_id", "error_code"],
            "stage_ordinal_coupling": {
                "endpoint_verification": -1,
                "shard_acquisition": "integer_at_least_zero",
            },
            "epoch_fields": ["step_started_at_us", "step_completed_at_us"],
            "monotonic_fields": [
                "step_started_monotonic_us",
                "step_completed_monotonic_us",
            ],
            "epoch_minimum": 1,
            "monotonic_minimum": 0,
            "timing_order": "end_not_before_start_in_both_clock_domains",
        },
        "executor_bindings": {
            "frozen": True,
            "six_component_identities": [
                "coordinator",
                "local_store",
                "clock",
                "detached_anchor_sink",
                "endpoint_runner",
                "shard_runner",
            ],
            "component_identity_shape": ["contract_version", "contract_hash"],
            "wire_top_level_keys_in_order": [
                "contract_version",
                "coordinator",
                "local_store",
                "clock",
                "detached_anchor_sink",
                "endpoint_runner",
                "shard_runner",
                "endpoint_verifier_binding",
                "shard_executor_binding",
            ],
            "wire_identity_fields": ["contract_version", "contract_hash"],
            "wire_component_sources": {
                "coordinator": [
                    "coordinator_contract_version",
                    "coordinator_contract_hash",
                ],
                "local_store": [
                    "local_store_contract_version",
                    "local_store_contract_hash",
                ],
                "clock": ["clock_contract_version", "clock_contract_hash"],
                "detached_anchor_sink": [
                    "detached_anchor_sink_contract_version",
                    "detached_anchor_sink_contract_hash",
                ],
                "endpoint_runner": [
                    "endpoint_runner_contract_version",
                    "endpoint_runner_contract_hash",
                ],
                "shard_runner": [
                    "shard_runner_contract_version",
                    "shard_runner_contract_hash",
                ],
                "endpoint_verifier_binding": [
                    "derived_endpoint_verifier_binding_version",
                    "derived_endpoint_verifier_binding_hash",
                ],
                "shard_executor_binding": [
                    "derived_shard_executor_binding_version",
                    "derived_shard_executor_binding_hash",
                ],
            },
            "parser_recomputes_both_composite_bindings": True,
        },
    },
    "protocols": {
        "clock": {
            "properties": {
                "contract_version": "str",
                "contract_hash": "str",
                "clock_domain_id": "str",
            },
            "methods": {
                "epoch_us": {"parameters": [], "returns": "int"},
                "monotonic_us": {"parameters": [], "returns": "int"},
                "sleep_us": {
                    "parameters": [["duration_us", "int", "positional_or_keyword"]],
                    "returns": "None",
                },
            },
        },
        "detached_anchor_sink": {
            "properties": {
                "contract_version": "str",
                "contract_hash": "str",
                "domain_id": "str",
            },
            "methods": {
                "create_once": {
                    "parameters": [
                        ["subject", "DetachedAnchorSubjectV1", "positional_or_keyword"],
                        ["clock", "PilotClock", "keyword_only"],
                    ],
                    "returns": "DetachedAnchorEvidenceV1",
                },
                "reload": {
                    "parameters": [
                        ["evidence_hash", "str", "positional_or_keyword"]
                    ],
                    "returns": "DetachedAnchorEvidenceV1",
                },
            },
        },
        "endpoint_stage_runner": {
            "properties": {
                "contract_version": "str",
                "contract_hash": "str",
            },
            "execute_keyword_only_parameters": {
                "manifest": "MexcPublicQaPilotRunManifestV1",
                "authorization": "U5PublicPilotAuthorizationReceiptV1",
                "preflight": "PilotDiskPreflightReceiptV1",
                "network_intent": "PilotNetworkIntentV1",
                "artifact_root": "Path",
                "clock": "PilotClock",
            },
            "returns": ["EndpointStageDraftV1", "StageFailureDraftV1"],
        },
        "shard_stage_runner": {
            "properties": {
                "contract_version": "str",
                "contract_hash": "str",
            },
            "execute_keyword_only_parameters": {
                "manifest": "MexcPublicQaPilotRunManifestV1",
                "authorization": "U5PublicPilotAuthorizationReceiptV1",
                "preflight": "PilotDiskPreflightReceiptV1",
                "network_intent": "PilotNetworkIntentV1",
                "shard_plan": "PilotShardPlanV1",
                "artifact_root": "Path",
                "clock": "PilotClock",
            },
            "returns": ["ShardStageDraftV1", "StageFailureDraftV1"],
        },
    },
    "bindings": {
        "shared": [
            "coordinator",
            "local_store",
            "clock",
            "detached_anchor_sink",
        ],
        "endpoint_specific": "endpoint_runner",
        "shard_specific": "shard_runner",
        "identity_drift_changes_manifest_hash": True,
        "endpoint_payload_exact_keys": [
            "domain",
            "bindings_contract_version",
            "protocol_versions",
            "coordinator",
            "local_store",
            "clock",
            "detached_anchor_sink",
            "endpoint_runner",
        ],
        "endpoint_protocol_versions_exact_keys": [
            "clock",
            "detached_anchor_sink",
            "stage_runner",
        ],
        "shard_payload_exact_keys": [
            "domain",
            "bindings_contract_version",
            "protocol_versions",
            "coordinator",
            "local_store",
            "clock",
            "detached_anchor_sink",
            "shard_runner",
        ],
        "shard_protocol_versions_exact_keys": [
            "clock",
            "detached_anchor_sink",
            "stage_runner",
        ],
        "composite_hash": "sha256_exact_canonical_payload",
    },
    "draft_boundary": {
        "authoritative_counters_are_caller_selected": False,
        "rows_pages_attempts_raw_storage_inventory_and_runtime_are_reloaded": True,
        "drafts_bind_subjects_paths_timing_and_evidence_hashes_only": True,
    },
    "builder": {
        "all_inputs_keyword_only_and_required": True,
        "parameters_in_order": [
            "repository_commit",
            "repository_tree_receipt_hash",
            "created_at_us",
            "parent_master_plan_path",
            "parent_master_plan_sha256",
            "parent_adr_path",
            "parent_adr_sha256",
            "output_root_locator",
            "endpoint_probe_request",
            "endpoint_relative_artifact_root",
            "official_reference_url",
            "endpoint_max_network_attempts",
            "endpoint_max_total_raw_body_bytes",
            "endpoint_max_total_storage_bytes",
            "endpoint_max_runtime_us",
            "endpoint_max_total_sleep_us",
            "ordered_shards",
            "budgets",
            "executor_bindings",
        ],
        "repository_document_time_path_probe_shards_budgets_bindings_explicit": True,
        "endpoint_plan_mapping": {
            "probe_request": "endpoint_probe_request",
            "relative_artifact_root": "endpoint_relative_artifact_root",
            "official_reference_url": "official_reference_url",
            "verifier_contract_version": "derived_endpoint_composite_version",
            "verifier_contract_hash": "derived_endpoint_composite_hash",
            "max_network_attempts": "endpoint_max_network_attempts",
            "max_total_raw_body_bytes": "endpoint_max_total_raw_body_bytes",
            "max_total_storage_bytes": "endpoint_max_total_storage_bytes",
            "max_runtime_us": "endpoint_max_runtime_us",
            "max_total_sleep_us": "endpoint_max_total_sleep_us",
        },
        "manifest_binding_mapping": {
            "repository_commit": "repository_commit",
            "repository_tree_receipt_hash": "repository_tree_receipt_hash",
            "created_at_us": "created_at_us",
            "parent_master_plan_path": "parent_master_plan_path",
            "parent_master_plan_sha256": "parent_master_plan_sha256",
            "parent_adr_path": "parent_adr_path",
            "parent_adr_sha256": "parent_adr_sha256",
            "output_root_locator": "output_root_locator",
            "endpoint_verifier": "derived_endpoint_composite_identity",
            "shard_executor": "derived_shard_composite_identity",
            "shards": "ordered_shards_exact_tuple",
            "budgets": "budgets_exact_object",
        },
        "defaults": False,
        "publication": False,
        "environment_lookup": False,
        "repository_lookup": False,
        "network": False,
        "authorization_creation": False,
        "concrete_instance_bundled": False,
    },
    "excluded": [
        "coordinator",
        "artifact_store",
        "u5_factory",
        "network_executor",
        "default_runner",
        "concrete_manifest",
    ],
}


def _computed_contract_hash() -> str:
    return _sha256_payload(_CONTRACT_SCHEMA)


def pilot_local_executor_contract_hash() -> str:
    digest = _computed_contract_hash()
    if _PINNED_CONTRACT_HASH and digest != _PINNED_CONTRACT_HASH:
        raise PilotLocalExecutorContractError(
            "pilot_local_executor_contract_changed_without_version_bump"
        )
    return digest


__all__ = [
    "DetachedAnchorEvidenceV1",
    "DetachedAnchorSink",
    "DetachedAnchorSubjectV1",
    "EndpointStageDraftV1",
    "EndpointStageRunner",
    "PILOT_CLOCK_PROTOCOL_VERSION",
    "PILOT_DETACHED_ANCHOR_EVIDENCE_VERSION",
    "PILOT_DETACHED_ANCHOR_SINK_PROTOCOL_VERSION",
    "PILOT_DETACHED_ANCHOR_SUBJECT_VERSION",
    "PILOT_ENDPOINT_EXECUTOR_BINDING_VERSION",
    "PILOT_ENDPOINT_STAGE_DRAFT_VERSION",
    "PILOT_ENDPOINT_STAGE_RUNNER_PROTOCOL_VERSION",
    "PILOT_EXECUTOR_BINDINGS_VERSION",
    "PILOT_LOCAL_EXECUTOR_CONTRACT_VERSION",
    "PILOT_SHARD_EXECUTOR_BINDING_VERSION",
    "PILOT_SHARD_STAGE_DRAFT_VERSION",
    "PILOT_SHARD_STAGE_RUNNER_PROTOCOL_VERSION",
    "PILOT_STAGE_FAILURE_DRAFT_VERSION",
    "PilotClock",
    "PilotExecutorBindingsV1",
    "PilotLocalExecutorContractError",
    "PilotLocalExecutorError",
    "ShardStageDraftV1",
    "ShardStageRunner",
    "StageFailureDraftV1",
    "build_concrete_pilot_manifest_v1",
    "pilot_local_executor_contract_hash",
]
