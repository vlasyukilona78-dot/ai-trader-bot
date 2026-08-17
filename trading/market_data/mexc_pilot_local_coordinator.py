"""Reviewed-fake, readiness-only coordinator for the frozen MEXC pilot.

This module deliberately stops one boundary before terminal publication.  The
frozen local store can durably arbitrate and consume one network intent, but it
does not expose a public terminal receipt writer or an active-intent CAS after
the callback.  The frozen strict-history reader also omits facts required by
the frozen pilot receipts.  Consequently this module can run exactly one
reviewed fake callback, reconstruct evidence from disk, and report blockers;
it can never publish or imply endpoint, shard, failure, or final success.

All dependency identities used here are declarative review bindings, not code
attestation.  No production runner, default runner, U5 factory, network path,
environment discovery, or restart/resume permission is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Protocol

from trading.market_data.mexc_pilot_local_executor import (
    EndpointStageDraftV1,
    PILOT_ENDPOINT_STAGE_DRAFT_VERSION,
    PILOT_SHARD_STAGE_DRAFT_VERSION,
    PILOT_STAGE_FAILURE_DRAFT_VERSION,
    StageFailureDraftV1,
    ShardStageDraftV1,
    pilot_local_executor_contract_hash,
)
from trading.market_data.mexc_pilot_local_store import (
    MexcPilotLocalStoreV1,
    PILOT_LOCAL_STORE_CONTRACT_VERSION,
    PilotClockSampleV1,
    PilotIntentClaimResultV1,
    PilotInventoryScanV1,
    PilotRunSessionCapability,
    mexc_pilot_local_store_contract_hash,
    pilot_runtime_authority_binding_contract_hash,
)
from trading.market_data.mexc_pilot_run import (
    PilotNetworkIntentV1,
    PilotRunStateV1,
    pilot_run_contract_hash,
)
from trading.market_data.strict_history import (
    STRICT_HISTORY_CONTRACT_VERSION,
    strict_history_contract_hash,
)
from trading.market_data.strict_history_v2 import (
    HistoryRestartReportV1,
    STRICT_HISTORY_V2_CONTRACT_VERSION,
    StrictHistoryArtifactStoreV2,
    strict_history_v2_contract_hash,
)


PILOT_LOCAL_COORDINATOR_CONTRACT_VERSION = (
    "mexc_public_qa_pilot_local_coordinator_readiness_v1"
)
PILOT_COORDINATOR_BINDINGS_VERSION = (
    "mexc_public_qa_pilot_coordinator_bindings_v1"
)
PILOT_REVIEWED_FAKE_RUNNER_BINDING_VERSION = (
    "mexc_public_qa_pilot_reviewed_fake_runner_binding_v1"
)
PILOT_FRESH_OUTPUT_SNAPSHOT_VERSION = (
    "mexc_public_qa_pilot_fresh_output_snapshot_v1"
)
PILOT_FRESH_HISTORY_EVIDENCE_VERSION = (
    "mexc_public_qa_pilot_fresh_history_evidence_v1"
)
PILOT_COORDINATOR_READINESS_ASSESSMENT_VERSION = (
    "mexc_public_qa_pilot_coordinator_readiness_assessment_v1"
)
PILOT_FAKE_RUNNER_REVIEW_POLICY_VERSION = (
    "mexc_public_qa_pilot_fake_runner_review_policy_v1"
)

_PINNED_CONTRACT_HASH = (
    "a19d002c04a3ab09d16a18bfdce66adcbc43399200ee092ffd0b179abb9016fc"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_TERMINAL_PUBLISHER_ABSENT_HASH = hashlib.sha256(
    b"mexc_public_qa_pilot_terminal_publisher_absent_v1"
).hexdigest()

_COMMON_BLOCKERS = (
    "authoritative_state_remains_unresolved_intent",
    "detached_terminal_anchor_protocol_not_exposed_by_local_store_v1",
    "local_store_v1_has_no_public_active_intent_terminal_cas",
    "local_store_v1_has_no_public_terminal_writer",
    "logical_reference_inventory_contract_missing",
    "strict_history_admission_decision_runtime_not_public",
    "strict_history_admitted_logical_storage_total_not_public",
    "strict_history_observed_internal_sleep_not_public",
    "strict_history_physical_raw_dedup_not_logical_reference_accounting",
    "strict_history_writer_lock_inventory_layout_unresolved",
)
_ENDPOINT_BLOCKERS = (
    "endpoint_official_document_evidence_parser_missing",
    "endpoint_official_document_network_accounting_missing",
    "endpoint_terminal_receipt_construction_forbidden",
)
_SHARD_BLOCKERS = (
    "shard_terminal_receipt_construction_forbidden",
)
_FAILURE_BLOCKERS = (
    "failure_candidate_publish_reload_anchor_seal_api_missing",
    "partial_stage_authoritative_accounting_missing",
    "terminal_failure_receipt_construction_forbidden",
)


class PilotLocalCoordinatorError(RuntimeError):
    """Base error for the readiness-only coordinator."""


class PilotLocalCoordinatorContractError(PilotLocalCoordinatorError):
    pass


class PilotLocalCoordinatorEvidenceError(PilotLocalCoordinatorError):
    """Fresh evidence could not be reconstructed after permission was consumed."""


class PilotLocalCoordinatorCallbackStopError(PilotLocalCoordinatorError):
    """The one-shot callback failed; its persisted intent remains STOP/no-retry."""


class PilotLocalCoordinatorPreCallbackError(PilotLocalCoordinatorError):
    """The store rejected execution before the injected callback was entered."""


def _canonical_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PilotLocalCoordinatorContractError(
            "pilot_coordinator_payload_is_not_canonical_json"
        ) from exc


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PilotLocalCoordinatorContractError(f"{field}_is_invalid")
    return value


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise PilotLocalCoordinatorContractError(f"{field}_is_invalid")
    return value


def _strict_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PilotLocalCoordinatorContractError(f"{field}_is_invalid")
    return value


def _exact_mapping(
    payload: object,
    expected: frozenset[str],
    *,
    code: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise PilotLocalCoordinatorContractError(code)
    return dict(payload)


def _relative_root(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise PilotLocalCoordinatorContractError(
            "pilot_coordinator_relative_root_is_invalid"
        )
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise PilotLocalCoordinatorContractError(
            "pilot_coordinator_relative_root_is_invalid"
        )
    return value


def _fake_review_policy_hash() -> str:
    return _sha256_payload(
        {
            "contract_version": PILOT_FAKE_RUNNER_REVIEW_POLICY_VERSION,
            "execution_mode": "reviewed_fake_local_fixture_only",
            "network_capable": False,
            "environment_access_permitted": False,
            "production_use_permitted": False,
            "identity_assurance": "declarative_review_evidence_not_code_attestation",
            "one_callback_only": True,
        }
    )


@dataclass(frozen=True)
class PilotCoordinatorBindingsV1:
    coordinator_contract_version: str
    coordinator_contract_hash: str
    pilot_run_contract_version: str
    pilot_run_contract_hash: str
    local_executor_contract_version: str
    local_executor_contract_hash: str
    local_store_contract_version: str
    local_store_contract_hash: str
    strict_history_v1_contract_version: str
    strict_history_v1_contract_hash: str
    strict_history_v2_contract_version: str
    strict_history_v2_contract_hash: str
    runtime_authority_binding_contract_hash: str
    fake_runner_review_policy_version: str
    fake_runner_review_policy_hash: str
    reviewed_fake_runner_binding_hashes: tuple[str, ...]
    terminal_publisher_contract_version: str = "absent"
    terminal_publisher_contract_hash: str = _TERMINAL_PUBLISHER_ABSENT_HASH
    contract_version: str = PILOT_COORDINATOR_BINDINGS_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_COORDINATOR_BINDINGS_VERSION:
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_bindings_version_mismatch"
            )
        for name in (
            "coordinator_contract_version",
            "pilot_run_contract_version",
            "local_executor_contract_version",
            "local_store_contract_version",
            "strict_history_v1_contract_version",
            "strict_history_v2_contract_version",
            "fake_runner_review_policy_version",
            "terminal_publisher_contract_version",
        ):
            _identifier(getattr(self, name), field=f"pilot_coordinator_{name}")
        for name in (
            "coordinator_contract_hash",
            "pilot_run_contract_hash",
            "local_executor_contract_hash",
            "local_store_contract_hash",
            "strict_history_v1_contract_hash",
            "strict_history_v2_contract_hash",
            "runtime_authority_binding_contract_hash",
            "fake_runner_review_policy_hash",
            "terminal_publisher_contract_hash",
        ):
            _digest(getattr(self, name), field=f"pilot_coordinator_{name}")
        if (
            not isinstance(self.reviewed_fake_runner_binding_hashes, tuple)
            or not self.reviewed_fake_runner_binding_hashes
            or self.reviewed_fake_runner_binding_hashes
            != tuple(sorted(set(self.reviewed_fake_runner_binding_hashes)))
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_reviewed_runner_bindings_are_not_canonical"
            )
        for digest in self.reviewed_fake_runner_binding_hashes:
            _digest(digest, field="pilot_coordinator_reviewed_runner_binding_hash")
        expected = {
            "coordinator_contract_version": PILOT_LOCAL_COORDINATOR_CONTRACT_VERSION,
            "coordinator_contract_hash": pilot_local_coordinator_contract_hash(),
            "pilot_run_contract_version": "mexc_public_qa_pilot_run_v1",
            "pilot_run_contract_hash": pilot_run_contract_hash(),
            "local_executor_contract_version": (
                "mexc_public_qa_pilot_local_executor_v1"
            ),
            "local_executor_contract_hash": pilot_local_executor_contract_hash(),
            "local_store_contract_version": PILOT_LOCAL_STORE_CONTRACT_VERSION,
            "local_store_contract_hash": mexc_pilot_local_store_contract_hash(),
            "strict_history_v1_contract_version": STRICT_HISTORY_CONTRACT_VERSION,
            "strict_history_v1_contract_hash": strict_history_contract_hash(),
            "strict_history_v2_contract_version": STRICT_HISTORY_V2_CONTRACT_VERSION,
            "strict_history_v2_contract_hash": strict_history_v2_contract_hash(),
            "runtime_authority_binding_contract_hash": (
                pilot_runtime_authority_binding_contract_hash()
            ),
            "fake_runner_review_policy_version": (
                PILOT_FAKE_RUNNER_REVIEW_POLICY_VERSION
            ),
            "fake_runner_review_policy_hash": _fake_review_policy_hash(),
            "terminal_publisher_contract_version": "absent",
            "terminal_publisher_contract_hash": _TERMINAL_PUBLISHER_ABSENT_HASH,
            "contract_version": PILOT_COORDINATOR_BINDINGS_VERSION,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_bindings_do_not_match_frozen_dependencies"
            )

    @classmethod
    def frozen(
        cls,
        *,
        reviewed_fake_runner_binding_hashes: tuple[str, ...],
    ) -> "PilotCoordinatorBindingsV1":
        return cls(
            coordinator_contract_version=PILOT_LOCAL_COORDINATOR_CONTRACT_VERSION,
            coordinator_contract_hash=pilot_local_coordinator_contract_hash(),
            pilot_run_contract_version="mexc_public_qa_pilot_run_v1",
            pilot_run_contract_hash=pilot_run_contract_hash(),
            local_executor_contract_version="mexc_public_qa_pilot_local_executor_v1",
            local_executor_contract_hash=pilot_local_executor_contract_hash(),
            local_store_contract_version=PILOT_LOCAL_STORE_CONTRACT_VERSION,
            local_store_contract_hash=mexc_pilot_local_store_contract_hash(),
            strict_history_v1_contract_version=STRICT_HISTORY_CONTRACT_VERSION,
            strict_history_v1_contract_hash=strict_history_contract_hash(),
            strict_history_v2_contract_version=STRICT_HISTORY_V2_CONTRACT_VERSION,
            strict_history_v2_contract_hash=strict_history_v2_contract_hash(),
            runtime_authority_binding_contract_hash=(
                pilot_runtime_authority_binding_contract_hash()
            ),
            fake_runner_review_policy_version=PILOT_FAKE_RUNNER_REVIEW_POLICY_VERSION,
            fake_runner_review_policy_hash=_fake_review_policy_hash(),
            reviewed_fake_runner_binding_hashes=tuple(
                sorted(reviewed_fake_runner_binding_hashes)
            ),
        )

    @property
    def binding_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PilotReviewedFakeRunnerBindingV1:
    stage: str
    runner_contract_version: str
    runner_contract_hash: str
    review_evidence_hash: str
    fixture_set_hash: str
    execution_mode: str = "reviewed_fake_local_fixture_only"
    network_capable: bool = False
    environment_access_permitted: bool = False
    production_use_permitted: bool = False
    contract_version: str = PILOT_REVIEWED_FAKE_RUNNER_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_REVIEWED_FAKE_RUNNER_BINDING_VERSION:
            raise PilotLocalCoordinatorContractError(
                "pilot_fake_runner_binding_version_mismatch"
            )
        if self.stage not in {"endpoint_verification", "shard_acquisition"}:
            raise PilotLocalCoordinatorContractError(
                "pilot_fake_runner_binding_stage_is_invalid"
            )
        _identifier(
            self.runner_contract_version,
            field="pilot_fake_runner_contract_version",
        )
        for name in ("runner_contract_hash", "review_evidence_hash", "fixture_set_hash"):
            _digest(getattr(self, name), field=f"pilot_fake_{name}")
        if (
            self.execution_mode != "reviewed_fake_local_fixture_only"
            or self.network_capable is not False
            or self.environment_access_permitted is not False
            or self.production_use_permitted is not False
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_fake_runner_binding_does_not_fail_closed"
            )

    def resolve_validated_execute(self, runner: object) -> Any:
        observed = (
            getattr(runner, "contract_version", None),
            getattr(runner, "contract_hash", None),
            getattr(runner, "review_evidence_hash", None),
            getattr(runner, "fixture_set_hash", None),
            getattr(runner, "execution_mode", None),
            getattr(runner, "network_capable", None),
            getattr(runner, "environment_access_permitted", None),
            getattr(runner, "production_use_permitted", None),
        )
        execute = getattr(runner, "execute", None)
        expected = (
            self.runner_contract_version,
            self.runner_contract_hash,
            self.review_evidence_hash,
            self.fixture_set_hash,
            self.execution_mode,
            False,
            False,
            False,
        )
        if observed != expected or not callable(execute):
            raise PilotLocalCoordinatorContractError(
                "pilot_fake_runner_runtime_binding_mismatch"
            )
        return execute

    def validate_runner(self, runner: object) -> None:
        self.resolve_validated_execute(runner)

    @property
    def binding_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PilotFreshOutputSnapshotV1:
    manifest_hash: str
    inventory_hash: str
    inventory_entries: int
    inventory_bytes: int
    scanned_at_us: int
    scanned_monotonic_us: int
    clock_domain_id: str
    contract_version: str = PILOT_FRESH_OUTPUT_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_FRESH_OUTPUT_SNAPSHOT_VERSION:
            raise PilotLocalCoordinatorContractError(
                "pilot_fresh_output_snapshot_version_mismatch"
            )
        _digest(self.manifest_hash, field="pilot_snapshot_manifest_hash")
        _digest(self.inventory_hash, field="pilot_snapshot_inventory_hash")
        _strict_int(self.inventory_entries, field="pilot_snapshot_entries", minimum=1)
        _strict_int(self.inventory_bytes, field="pilot_snapshot_bytes", minimum=1)
        _strict_int(self.scanned_at_us, field="pilot_snapshot_scanned_at_us", minimum=1)
        _strict_int(
            self.scanned_monotonic_us,
            field="pilot_snapshot_scanned_monotonic_us",
        )
        _identifier(self.clock_domain_id, field="pilot_snapshot_clock_domain_id")

    @classmethod
    def from_scan(cls, scan: PilotInventoryScanV1) -> "PilotFreshOutputSnapshotV1":
        if type(scan) is not PilotInventoryScanV1:
            raise PilotLocalCoordinatorContractError(
                "pilot_snapshot_requires_store_inventory_scan"
            )
        return cls(
            manifest_hash=scan.manifest_hash,
            inventory_hash=scan.inventory_hash,
            inventory_entries=len(scan.entries),
            inventory_bytes=scan.total_bytes,
            scanned_at_us=scan.scanned_at_us,
            scanned_monotonic_us=scan.scanned_monotonic_us,
            clock_domain_id=scan.clock_domain_id,
        )

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PilotFreshHistoryEvidenceV1:
    pilot_manifest_hash: str
    stage: str
    ordinal: int
    request_id: str
    relative_artifact_root: str
    history_manifest_hash: str
    strict_manifest_pages: int
    strict_manifest_rows: int
    strict_manifest_attempts: int
    strict_manifest_raw_body_bytes: int
    strict_manifest_graph_logical_storage_bytes: int
    strict_manifest_collection_runtime_us: int
    physical_inventory_hash: str
    physical_inventory_entries: int
    physical_inventory_bytes: int
    parent_output_inventory_hash: str
    fresh_reload_completed_at_us: int
    fresh_reload_completed_monotonic_us: int
    clock_domain_id: str
    restart_reconciliation_ready: bool = True
    all_restart_residue_absent: bool = True
    terminal_accounting_ready: bool = False
    contract_version: str = PILOT_FRESH_HISTORY_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_FRESH_HISTORY_EVIDENCE_VERSION:
            raise PilotLocalCoordinatorContractError(
                "pilot_fresh_history_evidence_version_mismatch"
            )
        for name in (
            "pilot_manifest_hash",
            "request_id",
            "history_manifest_hash",
            "physical_inventory_hash",
            "parent_output_inventory_hash",
        ):
            _digest(getattr(self, name), field=f"pilot_history_{name}")
        if self.stage not in {"endpoint_verification", "shard_acquisition"}:
            raise PilotLocalCoordinatorContractError(
                "pilot_history_stage_is_invalid"
            )
        if type(self.ordinal) is not int or (
            self.stage == "endpoint_verification" and self.ordinal != -1
        ) or (self.stage == "shard_acquisition" and self.ordinal < 0):
            raise PilotLocalCoordinatorContractError(
                "pilot_history_ordinal_is_invalid"
            )
        _relative_root(self.relative_artifact_root)
        for name in (
            "strict_manifest_pages",
            "strict_manifest_rows",
            "strict_manifest_attempts",
            "strict_manifest_raw_body_bytes",
            "strict_manifest_graph_logical_storage_bytes",
            "physical_inventory_entries",
            "physical_inventory_bytes",
        ):
            _strict_int(getattr(self, name), field=f"pilot_history_{name}", minimum=1)
        _strict_int(
            self.strict_manifest_collection_runtime_us,
            field="pilot_history_strict_manifest_collection_runtime_us",
        )
        if (
            self.strict_manifest_attempts < self.strict_manifest_pages
            or self.strict_manifest_rows < self.strict_manifest_pages
            or self.strict_manifest_graph_logical_storage_bytes
            < self.strict_manifest_raw_body_bytes
            or self.physical_inventory_entries
            < self.strict_manifest_attempts + 5
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_history_strict_manifest_accounting_is_impossible"
            )
        _strict_int(
            self.fresh_reload_completed_at_us,
            field="pilot_history_reload_at_us",
            minimum=1,
        )
        _strict_int(
            self.fresh_reload_completed_monotonic_us,
            field="pilot_history_reload_monotonic_us",
        )
        _identifier(self.clock_domain_id, field="pilot_history_clock_domain_id")
        if (
            self.restart_reconciliation_ready is not True
            or self.all_restart_residue_absent is not True
            or self.terminal_accounting_ready is not False
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_history_evidence_cannot_claim_terminal_readiness"
            )

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PilotCoordinatorReadinessAssessmentV1:
    manifest_hash: str
    network_intent_hash: str
    coordinator_bindings_hash: str
    runner_binding_hash: str
    runner_contract_version: str
    runner_contract_hash: str
    runner_review_evidence_hash: str
    runner_fixture_set_hash: str
    stage: str
    ordinal: int
    draft_contract_version: str
    draft_hash: str
    output_snapshot: PilotFreshOutputSnapshotV1
    fresh_history_evidence: PilotFreshHistoryEvidenceV1 | None
    blockers: tuple[str, ...]
    authoritative_recovery_stop_code: str
    callback_consumed_once: bool = True
    network_retry_permitted: bool = False
    terminal_receipt_constructible: bool = False
    authoritative_terminal_published: bool = False
    next_action: str = "stop_unresolved_intent_no_retry"
    contract_version: str = PILOT_COORDINATOR_READINESS_ASSESSMENT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_COORDINATOR_READINESS_ASSESSMENT_VERSION:
            raise PilotLocalCoordinatorContractError(
                "pilot_readiness_assessment_version_mismatch"
            )
        _digest(self.manifest_hash, field="pilot_assessment_manifest_hash")
        _digest(self.network_intent_hash, field="pilot_assessment_intent_hash")
        for name in (
            "coordinator_bindings_hash",
            "runner_binding_hash",
            "runner_contract_hash",
            "runner_review_evidence_hash",
            "runner_fixture_set_hash",
        ):
            _digest(getattr(self, name), field=f"pilot_assessment_{name}")
        _identifier(
            self.runner_contract_version,
            field="pilot_assessment_runner_contract_version",
        )
        reconstructed_runner_binding = PilotReviewedFakeRunnerBindingV1(
            stage=self.stage,
            runner_contract_version=self.runner_contract_version,
            runner_contract_hash=self.runner_contract_hash,
            review_evidence_hash=self.runner_review_evidence_hash,
            fixture_set_hash=self.runner_fixture_set_hash,
        )
        if self.runner_binding_hash != reconstructed_runner_binding.binding_hash:
            raise PilotLocalCoordinatorContractError(
                "pilot_assessment_runner_binding_hash_mismatch"
            )
        _identifier(
            self.draft_contract_version,
            field="pilot_assessment_draft_version",
        )
        _digest(self.draft_hash, field="pilot_assessment_draft_hash")
        if self.stage not in {"endpoint_verification", "shard_acquisition"}:
            raise PilotLocalCoordinatorContractError(
                "pilot_assessment_stage_is_invalid"
            )
        if type(self.ordinal) is not int or (
            self.stage == "endpoint_verification" and self.ordinal != -1
        ) or (self.stage == "shard_acquisition" and self.ordinal < 0):
            raise PilotLocalCoordinatorContractError(
                "pilot_assessment_ordinal_is_invalid"
            )
        if type(self.output_snapshot) is not PilotFreshOutputSnapshotV1:
            raise PilotLocalCoordinatorContractError(
                "pilot_assessment_output_snapshot_is_invalid"
            )
        if self.output_snapshot.manifest_hash != self.manifest_hash:
            raise PilotLocalCoordinatorContractError(
                "pilot_assessment_output_snapshot_manifest_mismatch"
            )
        if self.fresh_history_evidence is not None and type(
            self.fresh_history_evidence
        ) is not PilotFreshHistoryEvidenceV1:
            raise PilotLocalCoordinatorContractError(
                "pilot_assessment_history_evidence_is_invalid"
            )
        if self.draft_contract_version == PILOT_ENDPOINT_STAGE_DRAFT_VERSION:
            expected_blockers = tuple(sorted((*_COMMON_BLOCKERS, *_ENDPOINT_BLOCKERS)))
            if self.stage != "endpoint_verification" or self.fresh_history_evidence is None:
                raise PilotLocalCoordinatorContractError(
                    "pilot_assessment_endpoint_outcome_binding_mismatch"
                )
        elif self.draft_contract_version == PILOT_SHARD_STAGE_DRAFT_VERSION:
            expected_blockers = tuple(sorted((*_COMMON_BLOCKERS, *_SHARD_BLOCKERS)))
            if self.stage != "shard_acquisition" or self.fresh_history_evidence is None:
                raise PilotLocalCoordinatorContractError(
                    "pilot_assessment_shard_outcome_binding_mismatch"
                )
        elif self.draft_contract_version == PILOT_STAGE_FAILURE_DRAFT_VERSION:
            expected_blockers = tuple(sorted((*_COMMON_BLOCKERS, *_FAILURE_BLOCKERS)))
            if self.fresh_history_evidence is not None:
                raise PilotLocalCoordinatorContractError(
                    "pilot_assessment_failure_outcome_binding_mismatch"
                )
        else:
            raise PilotLocalCoordinatorContractError(
                "pilot_assessment_draft_contract_is_unknown"
            )
        if self.blockers != expected_blockers:
            raise PilotLocalCoordinatorContractError(
                "pilot_assessment_required_blockers_mismatch"
            )
        _identifier(
            self.authoritative_recovery_stop_code,
            field="pilot_assessment_recovery_stop_code",
        )
        if (
            self.authoritative_recovery_stop_code
            != "unresolved_network_intent_after_restart"
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_assessment_recovery_must_be_unresolved_intent_stop"
            )
        history = self.fresh_history_evidence
        if history is not None and (
            history.pilot_manifest_hash != self.manifest_hash
            or history.stage != self.stage
            or history.ordinal != self.ordinal
            or history.parent_output_inventory_hash
            != self.output_snapshot.inventory_hash
            or history.clock_domain_id != self.output_snapshot.clock_domain_id
            or history.fresh_reload_completed_at_us
            < self.output_snapshot.scanned_at_us
            or history.fresh_reload_completed_monotonic_us
            < self.output_snapshot.scanned_monotonic_us
            or history.physical_inventory_entries
            > self.output_snapshot.inventory_entries
            or history.physical_inventory_bytes
            > self.output_snapshot.inventory_bytes
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_assessment_nested_evidence_binding_mismatch"
            )
        if (
            self.callback_consumed_once is not True
            or self.network_retry_permitted is not False
            or self.terminal_receipt_constructible is not False
            or self.authoritative_terminal_published is not False
            or self.next_action != "stop_unresolved_intent_no_retry"
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_assessment_must_remain_stop_only"
            )

    @property
    def assessment_hash(self) -> str:
        return _sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"output_snapshot", "fresh_history_evidence"}
            },
            "output_snapshot": self.output_snapshot.as_dict(),
            "fresh_history_evidence": (
                None
                if self.fresh_history_evidence is None
                else self.fresh_history_evidence.as_dict()
            ),
        }


class ReviewedFakeStageRunner(Protocol):
    @property
    def contract_version(self) -> str: ...

    @property
    def contract_hash(self) -> str: ...

    def execute(self, **kwargs: object) -> object: ...


class _ReviewedRunnerAdapter:
    __slots__ = ("_coordinator", "_state", "_runner", "_binding", "_invoked")

    def __init__(
        self,
        coordinator: "MexcPilotLocalCoordinatorReadinessV1",
        state: PilotRunStateV1,
        runner: ReviewedFakeStageRunner,
        binding: PilotReviewedFakeRunnerBindingV1,
    ) -> None:
        self._coordinator = coordinator
        self._state = state
        self._runner = runner
        self._binding = binding
        self._invoked = False

    @property
    def invoked(self) -> bool:
        return self._invoked

    @property
    def contract_version(self) -> str:
        return self._binding.runner_contract_version

    @property
    def contract_hash(self) -> str:
        return self._binding.runner_contract_hash

    def __call__(self, intent: PilotNetworkIntentV1) -> object:
        self._invoked = True
        # Re-resolve the exact reviewed callable after the Store has consumed
        # permission.  Any mutation during preflight/claim becomes STOP before
        # the injected implementation can run; the bound local callable is not
        # looked up from the runner again for this attempt.
        execute = self._binding.resolve_validated_execute(self._runner)
        state = self._state
        manifest = self._coordinator.store.manifest
        if (
            not state.network_intents
            or state.network_intents[-1] != intent
            or state.authorization is None
            or not state.preflight_receipts
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_callback_state_binding_mismatch"
            )
        common = {
            "manifest": manifest,
            "authorization": state.authorization,
            "preflight": state.preflight_receipts[-1],
            "network_intent": intent,
            "artifact_root": self._coordinator.store.output_root
            / Path(*PurePosixPath(intent.relative_artifact_root).parts),
            "clock": self._coordinator.store.clock,
        }
        if intent.stage == "endpoint_verification":
            return execute(**common)
        return execute(
            **common,
            shard_plan=manifest.shards[intent.ordinal],
        )


class MexcPilotLocalCoordinatorReadinessV1:
    """Consume one reviewed fake intent and report why terminalization is STOP."""

    __slots__ = ("_store", "_bindings")

    def __init__(
        self,
        *,
        store: MexcPilotLocalStoreV1,
        bindings: PilotCoordinatorBindingsV1,
    ) -> None:
        if type(store) is not MexcPilotLocalStoreV1:
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_requires_frozen_local_store_v1"
            )
        if type(bindings) is not PilotCoordinatorBindingsV1:
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_bindings_are_required"
            )
        runtime = store.runtime_authority_binding
        if (
            runtime.coordinator_implementation_contract_version
            != bindings.contract_version
            or runtime.coordinator_implementation_contract_hash
            != bindings.binding_hash
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_runtime_transitive_binding_mismatch"
            )
        clock = store.clock
        if any(
            not callable(getattr(clock, name, None))
            for name in ("sample", "epoch_us", "monotonic_us", "sleep_us")
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_requires_reviewed_dual_interface_clock"
            )
        sink = store.detached_evidence_sink
        if any(
            not callable(getattr(sink, name, None))
            for name in ("anchor", "create_once", "reload")
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_requires_reviewed_dual_interface_sink"
            )
        self._store = store
        self._bindings = bindings

    @property
    def store(self) -> MexcPilotLocalStoreV1:
        return self._store

    @property
    def bindings(self) -> PilotCoordinatorBindingsV1:
        return self._bindings

    @staticmethod
    def _require_clean_restart_report(report: HistoryRestartReportV1) -> None:
        if (
            type(report) is not HistoryRestartReportV1
            or report.ready is not True
            or len(report.request_states) != 1
            or report.request_states[0].state != "complete_verified"
            or any(
                (
                    report.temp_paths,
                    report.unreferenced_attempt_paths,
                    report.unreferenced_raw_paths,
                    report.alternate_normalized_paths,
                )
            )
        ):
            raise PilotLocalCoordinatorEvidenceError(
                "pilot_coordinator_strict_history_reconciliation_not_clean_no_retry"
            )

    def _require_ordered_bound_sample(
        self,
        sample: object,
        scan: PilotInventoryScanV1,
    ) -> PilotClockSampleV1:
        if (
            type(sample) is not PilotClockSampleV1
            or sample.clock_domain_id
            != self.store.runtime_authority_binding.clock_domain_id
            or sample.clock_domain_id != scan.clock_domain_id
            or sample.epoch_us < scan.scanned_at_us
            or sample.monotonic_us < scan.scanned_monotonic_us
        ):
            raise PilotLocalCoordinatorEvidenceError(
                "pilot_coordinator_final_clock_sample_binding_or_order_invalid_no_retry"
            )
        return sample

    def _fresh_history_evidence(
        self,
        *,
        draft: EndpointStageDraftV1 | ShardStageDraftV1,
        intent: PilotNetworkIntentV1,
    ) -> tuple[PilotFreshHistoryEvidenceV1, PilotInventoryScanV1]:
        manifest = self.store.manifest
        if intent.stage == "endpoint_verification":
            assert isinstance(draft, EndpointStageDraftV1)
            plan = manifest.endpoint_verification
            request = plan.probe_request
            root_locator = plan.relative_artifact_root
            expected_history_hash = draft.live_history_manifest_hash
            if draft.live_probe_store_relative_root != root_locator:
                raise PilotLocalCoordinatorEvidenceError(
                    "pilot_coordinator_endpoint_history_root_mismatch_no_retry"
                )
        else:
            assert isinstance(draft, ShardStageDraftV1)
            plan = manifest.shards[intent.ordinal]
            request = plan.request
            root_locator = plan.relative_artifact_root
            expected_history_hash = draft.history_manifest_hash

        root = self.store.output_root / Path(*PurePosixPath(root_locator).parts)
        try:
            before_store = StrictHistoryArtifactStoreV2(
                root,
                writable=False,
                storage_profile=request.storage_profile,
            )
            before = before_store.reconcile_restart(
                [request],
                expected_manifest_hashes={request.request_id: expected_history_hash},
            )
            self._require_clean_restart_report(before)
            # The frozen store performs the bounded, double-checked physical
            # inventory.  Reconcile once on either side of that stable-point
            # scan; no draft-provided count or byte total enters the result.
            scan = self.store.scan_inventory()
            if intent.stage == "endpoint_verification":
                assert isinstance(draft, EndpointStageDraftV1)
                official = next(
                    (
                        entry
                        for entry in scan.entries
                        if entry.relative_path
                        == draft.official_document_evidence_relative_path
                    ),
                    None,
                )
                if (
                    official is None
                    or official.artifact_sha256
                    != draft.official_document_evidence_hash
                ):
                    raise PilotLocalCoordinatorEvidenceError(
                        "pilot_coordinator_official_document_artifact_mismatch_no_retry"
                    )
            after_store = StrictHistoryArtifactStoreV2(
                root,
                writable=False,
                storage_profile=request.storage_profile,
            )
            after = after_store.reconcile_restart(
                [request],
                expected_manifest_hashes={request.request_id: expected_history_hash},
            )
            self._require_clean_restart_report(after)
            if before != after:
                raise PilotLocalCoordinatorEvidenceError(
                    "pilot_coordinator_strict_history_reconciliation_changed_no_retry"
                )
            shard = after_store.load_complete_from_disk(
                request,
                expected_manifest_hash=expected_history_hash,
            )
        except PilotLocalCoordinatorError:
            raise
        except Exception as exc:
            raise PilotLocalCoordinatorEvidenceError(
                "pilot_coordinator_strict_history_fresh_reload_failed_no_retry"
            ) from exc
        history = shard.manifest
        if history.manifest_hash != expected_history_hash:
            raise PilotLocalCoordinatorEvidenceError(
                "pilot_coordinator_strict_history_manifest_mismatch_no_retry"
            )
        prefix = root_locator + "/"
        scoped = tuple(
            entry for entry in scan.entries if entry.relative_path.startswith(prefix)
        )
        if not scoped:
            raise PilotLocalCoordinatorEvidenceError(
                "pilot_coordinator_history_inventory_is_empty_no_retry"
            )
        physical_bytes = sum(entry.byte_count for entry in scoped)
        physical_hash = _sha256_payload(
            {
                "domain": PILOT_FRESH_HISTORY_EVIDENCE_VERSION,
                "pilot_manifest_hash": manifest.manifest_hash,
                "stage": intent.stage,
                "ordinal": intent.ordinal,
                "request_id": request.request_id,
                "relative_artifact_root": root_locator,
                "entries": [entry.as_dict() for entry in scoped],
                "physical_inventory_bytes": physical_bytes,
            }
        )
        sample = self._require_ordered_bound_sample(
            self.store.clock.sample(),
            scan,
        )
        evidence = PilotFreshHistoryEvidenceV1(
            pilot_manifest_hash=manifest.manifest_hash,
            stage=intent.stage,
            ordinal=intent.ordinal,
            request_id=request.request_id,
            relative_artifact_root=root_locator,
            history_manifest_hash=history.manifest_hash,
            strict_manifest_pages=len(history.page_receipts),
            strict_manifest_rows=history.actual_row_count,
            strict_manifest_attempts=history.actual_attempt_count,
            strict_manifest_raw_body_bytes=history.actual_total_raw_body_bytes,
            strict_manifest_graph_logical_storage_bytes=history.logical_storage_bytes,
            strict_manifest_collection_runtime_us=history.collection_runtime_us,
            physical_inventory_hash=physical_hash,
            physical_inventory_entries=len(scoped),
            physical_inventory_bytes=physical_bytes,
            parent_output_inventory_hash=scan.inventory_hash,
            fresh_reload_completed_at_us=sample.epoch_us,
            fresh_reload_completed_monotonic_us=sample.monotonic_us,
            clock_domain_id=sample.clock_domain_id,
        )
        return evidence, scan

    def run_one_reviewed_fake_stage(
        self,
        *,
        expected_state: PilotRunStateV1,
        session_capability: PilotRunSessionCapability,
        runner: ReviewedFakeStageRunner,
        runner_binding: PilotReviewedFakeRunnerBindingV1,
    ) -> PilotCoordinatorReadinessAssessmentV1:
        """Run one fake callback; every return remains unresolved and no-retry."""

        if type(runner_binding) is not PilotReviewedFakeRunnerBindingV1:
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_requires_exact_reviewed_runner_binding"
            )
        runner_binding.validate_runner(runner)
        if runner_binding.binding_hash not in (
            self.bindings.reviewed_fake_runner_binding_hashes
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_runner_review_binding_not_prebound"
            )
        action = getattr(expected_state, "next_action", None)
        if action == "run_local_preflight:-1":
            expected_stage = "endpoint_verification"
            expected_identity = (
                self.store.executor_bindings.endpoint_runner_contract_version,
                self.store.executor_bindings.endpoint_runner_contract_hash,
            )
        elif isinstance(action, str) and action.startswith("run_local_preflight:"):
            expected_stage = "shard_acquisition"
            expected_identity = (
                self.store.executor_bindings.shard_runner_contract_version,
                self.store.executor_bindings.shard_runner_contract_hash,
            )
        else:
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_expected_state_is_not_at_preflight"
            )
        if (
            runner_binding.stage != expected_stage
            or expected_identity
            != (
                runner_binding.runner_contract_version,
                runner_binding.runner_contract_hash,
            )
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_runner_not_in_expected_stage_binding"
            )
        preflighted = self.store.measure_and_publish_preflight(expected_state)
        claimed = self.store.claim_and_seal_next_intent(
            preflighted,
            session_capability,
        )
        if type(claimed) is not PilotIntentClaimResultV1:
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_intent_claim_result_is_invalid"
            )
        intent = claimed.intent
        if intent.stage != runner_binding.stage:
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_runner_stage_binding_mismatch"
            )
        claimed_identity = (
            (
                self.store.executor_bindings.endpoint_runner_contract_version,
                self.store.executor_bindings.endpoint_runner_contract_hash,
            )
            if intent.stage == "endpoint_verification"
            else (
                self.store.executor_bindings.shard_runner_contract_version,
                self.store.executor_bindings.shard_runner_contract_hash,
            )
        )
        if claimed_identity != (
            runner_binding.runner_contract_version,
            runner_binding.runner_contract_hash,
        ):
            raise PilotLocalCoordinatorContractError(
                "pilot_coordinator_runner_not_in_executor_bindings"
            )
        adapter = _ReviewedRunnerAdapter(
            self,
            claimed.state,
            runner,
            runner_binding,
        )
        try:
            draft = self.store.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                adapter,
            )
        except Exception as exc:
            if adapter.invoked:
                raise PilotLocalCoordinatorCallbackStopError(
                    "pilot_coordinator_callback_failed_intent_consumed_no_retry"
                ) from exc
            raise PilotLocalCoordinatorPreCallbackError(
                "pilot_coordinator_store_rejected_before_callback_not_consumed"
            ) from exc

        expected_type: type[object]
        if intent.stage == "endpoint_verification":
            expected_type = EndpointStageDraftV1
        else:
            expected_type = ShardStageDraftV1
        if type(draft) not in {expected_type, StageFailureDraftV1}:
            raise PilotLocalCoordinatorEvidenceError(
                "pilot_coordinator_runner_returned_invalid_draft_no_retry"
            )
        if (
            draft.manifest_hash != self.store.manifest.manifest_hash
            or draft.network_intent_hash != intent.intent_hash
            or draft.clock_domain_id
            != self.store.runtime_authority_binding.clock_domain_id
        ):
            raise PilotLocalCoordinatorEvidenceError(
                "pilot_coordinator_draft_binding_mismatch_no_retry"
            )
        if (
            type(draft) is EndpointStageDraftV1
            and (
                claimed.state.authorization is None
                or draft.authorization_receipt_hash
                != claimed.state.authorization.receipt_hash
            )
        ):
            raise PilotLocalCoordinatorEvidenceError(
                "pilot_coordinator_endpoint_draft_authorization_mismatch_no_retry"
            )
        if type(draft) is StageFailureDraftV1:
            if (
                draft.stage != intent.stage
                or draft.ordinal != intent.ordinal
                or claimed.state.authorization is None
                or draft.authorization_receipt_hash
                != claimed.state.authorization.receipt_hash
            ):
                raise PilotLocalCoordinatorEvidenceError(
                    "pilot_coordinator_failure_draft_binding_mismatch_no_retry"
                )
        elif intent.stage == "shard_acquisition" and draft.ordinal != intent.ordinal:
            raise PilotLocalCoordinatorEvidenceError(
                "pilot_coordinator_shard_draft_ordinal_mismatch_no_retry"
            )

        recovery = self.store.reconstruct_authoritative_state()
        if (
            recovery.network_permitted is not False
            or recovery.stop_code != "unresolved_network_intent_after_restart"
        ):
            raise PilotLocalCoordinatorEvidenceError(
                "pilot_coordinator_post_callback_recovery_not_stop_no_retry"
            )
        history_evidence: PilotFreshHistoryEvidenceV1 | None = None
        if type(draft) in {EndpointStageDraftV1, ShardStageDraftV1}:
            history_evidence, scan = self._fresh_history_evidence(
                draft=draft,
                intent=intent,
            )
        else:
            scan = self.store.scan_inventory()
        if (
            scan.manifest_hash != self.store.manifest.manifest_hash
            or scan.clock_domain_id
            != self.store.runtime_authority_binding.clock_domain_id
        ):
            raise PilotLocalCoordinatorEvidenceError(
                "pilot_coordinator_output_scan_binding_mismatch_no_retry"
            )
        snapshot = PilotFreshOutputSnapshotV1.from_scan(scan)
        if type(draft) is StageFailureDraftV1:
            blockers = (*_COMMON_BLOCKERS, *_FAILURE_BLOCKERS)
        elif type(draft) is EndpointStageDraftV1:
            blockers = (*_COMMON_BLOCKERS, *_ENDPOINT_BLOCKERS)
        else:
            blockers = (*_COMMON_BLOCKERS, *_SHARD_BLOCKERS)
        return PilotCoordinatorReadinessAssessmentV1(
            manifest_hash=self.store.manifest.manifest_hash,
            network_intent_hash=intent.intent_hash,
            coordinator_bindings_hash=self.bindings.binding_hash,
            runner_binding_hash=runner_binding.binding_hash,
            runner_contract_version=runner_binding.runner_contract_version,
            runner_contract_hash=runner_binding.runner_contract_hash,
            runner_review_evidence_hash=runner_binding.review_evidence_hash,
            runner_fixture_set_hash=runner_binding.fixture_set_hash,
            stage=intent.stage,
            ordinal=intent.ordinal,
            draft_contract_version=draft.contract_version,
            draft_hash=_sha256_payload(draft.as_dict()),
            output_snapshot=snapshot,
            fresh_history_evidence=history_evidence,
            blockers=tuple(sorted(blockers)),
            authoritative_recovery_stop_code=recovery.stop_code,
        )


_CONTRACT_SCHEMA = {
    "contract_version": PILOT_LOCAL_COORDINATOR_CONTRACT_VERSION,
    "dependencies": {
        "pilot_run": pilot_run_contract_hash(),
        "local_executor": pilot_local_executor_contract_hash(),
        "local_store": mexc_pilot_local_store_contract_hash(),
        "runtime_authority_binding": pilot_runtime_authority_binding_contract_hash(),
        "strict_history_v1": strict_history_contract_hash(),
        "strict_history_v2": strict_history_v2_contract_hash(),
    },
    "field_sets": {
        "bindings": list(PilotCoordinatorBindingsV1.__dataclass_fields__),
        "reviewed_fake_runner": list(
            PilotReviewedFakeRunnerBindingV1.__dataclass_fields__
        ),
        "output_snapshot": list(PilotFreshOutputSnapshotV1.__dataclass_fields__),
        "history_evidence": list(PilotFreshHistoryEvidenceV1.__dataclass_fields__),
        "assessment": list(
            PilotCoordinatorReadinessAssessmentV1.__dataclass_fields__
        ),
    },
    "authority": {
        "mode": "reviewed_fake_local_fixture_only",
        "real_strict_history_terminal_adapter": "hard_stop_unimplemented",
        "terminal_publisher": "absent",
        "u5_factory": "absent",
        "network_runner": "absent",
        "default_runner": "absent",
        "concrete_manifest": "absent",
        "environment_or_repository_discovery": "absent",
        "draft_is_authoritative": False,
        "assessment_is_terminal_receipt": False,
        "assessment_grants_network_or_retry": False,
        "restart_grants_network_or_retry": False,
        "only_network_permission_boundary": "frozen_store_run_owned_intent_once",
        "callback_count": "exactly_one_then_consumed",
        "callback_exception": "entered_means_consumed_typed_stop_no_retry",
        "store_pre_callback_exception": "typed_not_consumed_no_callback_claim",
        "terminal_state_after_callback": "unresolved_intent_stop_no_retry",
    },
    "composition": {
        "runtime_binding": "coordinator_bindings_hash",
        "clock": "same_reviewed_object_implements_store_and_executor_protocols",
        "sink": "same_reviewed_object_implements_store_and_executor_protocols",
        "fake_runner_identity": "executor_binding_plus_review_evidence",
        "identity_assurance": "declarative_review_binding_not_code_attestation",
        "store_calls": [
            "measure_and_publish_preflight",
            "claim_and_seal_next_intent",
            "run_owned_intent_once",
            "reconstruct_authoritative_state",
            "scan_inventory",
        ],
        "private_store_or_history_api_calls": False,
        "dataclass_replace_or_direct_state_construction": False,
    },
    "fresh_evidence": {
        "source": "read_only_strict_history_v2_reconcile_reload_and_store_inventory",
        "draft_role": "locator_and_expected_hash_hint_only",
        "reconcile_before_and_after_inventory": True,
        "all_restart_residue_tuples_empty": True,
        "physical_and_logical_accounting_remain_distinct": True,
        "terminal_accounting_ready": False,
    },
    "blockers": {
        "common": list(_COMMON_BLOCKERS),
        "endpoint": list(_ENDPOINT_BLOCKERS),
        "shard": list(_SHARD_BLOCKERS),
        "failure": list(_FAILURE_BLOCKERS),
    },
    "future_failure_publication_order": [
        "candidate",
        "immutable_create_new_publish",
        "fresh_reload",
        "detached_anchor",
        "sealed_terminal_receipt",
        "authoritative_replay",
    ],
}


def pilot_local_coordinator_contract_hash() -> str:
    digest = _sha256_payload(_CONTRACT_SCHEMA)
    if _PINNED_CONTRACT_HASH and digest != _PINNED_CONTRACT_HASH:
        raise PilotLocalCoordinatorContractError(
            "pilot_local_coordinator_contract_changed_without_version_bump"
        )
    return digest


__all__ = [
    "MexcPilotLocalCoordinatorReadinessV1",
    "PILOT_COORDINATOR_BINDINGS_VERSION",
    "PILOT_COORDINATOR_READINESS_ASSESSMENT_VERSION",
    "PILOT_FAKE_RUNNER_REVIEW_POLICY_VERSION",
    "PILOT_FRESH_HISTORY_EVIDENCE_VERSION",
    "PILOT_FRESH_OUTPUT_SNAPSHOT_VERSION",
    "PILOT_LOCAL_COORDINATOR_CONTRACT_VERSION",
    "PILOT_REVIEWED_FAKE_RUNNER_BINDING_VERSION",
    "PilotCoordinatorBindingsV1",
    "PilotCoordinatorReadinessAssessmentV1",
    "PilotFreshHistoryEvidenceV1",
    "PilotFreshOutputSnapshotV1",
    "PilotLocalCoordinatorCallbackStopError",
    "PilotLocalCoordinatorContractError",
    "PilotLocalCoordinatorError",
    "PilotLocalCoordinatorEvidenceError",
    "PilotLocalCoordinatorPreCallbackError",
    "PilotReviewedFakeRunnerBindingV1",
    "ReviewedFakeStageRunner",
    "pilot_local_coordinator_contract_hash",
]
