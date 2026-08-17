from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from tests.v3.test_mexc_pilot_run_contract_v1 import (
    _authorization,
    _manifest,
    _network_intent,
    _preflight,
    _shard_result,
    _verification,
)
from trading.market_data.mexc_pilot_local_coordinator import (
    pilot_local_coordinator_contract_hash,
)
from trading.market_data.mexc_pilot_local_executor import (
    pilot_local_executor_contract_hash,
)
from trading.market_data.mexc_pilot_local_store import (
    PilotInventoryEntryV1,
    PilotInventoryScanV1,
    mexc_pilot_local_store_contract_hash,
)
from trading.market_data.mexc_pilot_output_layout import (
    PilotOfficialDocumentPlaceholderFileV1,
    PilotOfficialDocumentPlaceholderV1,
    PilotOutputLayoutBudgetStop,
    PilotOutputLayoutContractError,
    PilotOutputLayoutInventoryStop,
    PilotOutputLayoutPlanV1,
    PilotOutputLocatorPlanV1,
    PilotOutputLayoutTerminalStop,
    PilotOutputPhysicalEntryV1,
    PilotOutputReadinessAssessmentV1,
    PilotStageOutputAccountingV1,
    assess_pilot_output_layout_v1,
    build_pilot_output_layout_plan_v1,
    build_pilot_stage_output_accounting_v1,
    derive_expected_pilot_output_inventory_v1,
    derive_official_bundle_locators_v1,
    derive_official_bundle_root_v1,
    derive_persistent_writer_lock_locator_v1,
    pilot_output_layout_contract_hash,
)
from trading.market_data.mexc_pilot_run import (
    PilotGlobalBudgetsV1,
    PilotRunStateV1,
    pilot_run_contract_hash,
)
from trading.market_data.strict_history_pilot_evidence import (
    PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE,
    PILOT_OUTPUT_LAYOUT_STATUS_UNRESOLVED,
    PilotAdmissionAccountingV1,
    PilotAttemptAccountingV1,
    PilotLogicalReferenceV1,
    PilotPageAccountingV1,
    PilotPhysicalFileV1,
    PilotRestartNoResidueProofV1,
    PilotWriterLockFactV1,
    StrictHistoryPilotEvidenceV1,
    strict_history_pilot_evidence_contract_hash,
)
from trading.market_data.strict_history_v2 import (
    STRICT_HISTORY_V2_RESTART_VERSION,
    StrictMexcHistoryCollectorV2,
    strict_history_v2_contract_hash,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _evidence(request, *, label: str, duplicate_raw: bool = False):
    pages = StrictMexcHistoryCollectorV2.plan_pages(request)
    manifest_hash = _hash(f"{label}:manifest")
    normalized_hash = _hash(f"{label}:normalized")
    logical: list[PilotLogicalReferenceV1] = []

    def add(
        role: str,
        relative_path: str,
        reference_hash: str,
        file_sha256: str,
        byte_count: int,
        page: int = -1,
        attempt: int = -1,
    ) -> None:
        logical.append(
            PilotLogicalReferenceV1(
                ordinal=len(logical),
                role=role,
                relative_path=relative_path,
                reference_hash=reference_hash,
                file_sha256=file_sha256,
                byte_count=byte_count,
                page_ordinal=page,
                attempt_ordinal=attempt,
            )
        )

    scope_hash = _hash(f"{label}:scope")
    add("scope_marker", "scope.json", scope_hash, scope_hash, 19)
    attempts: list[PilotAttemptAccountingV1] = []
    page_accounting: list[PilotPageAccountingV1] = []
    raw_logical_bytes = 0
    for page in pages:
        page_ordinal = page.page_ordinal
        receipt_hash = _hash(f"{label}:attempt:{page_ordinal}")
        receipt_file_hash = _hash(f"{label}:attempt-file:{page_ordinal}")
        raw_index = 0 if duplicate_raw and page_ordinal < 2 else page_ordinal
        raw_hash = _hash(f"{label}:raw:{raw_index}")
        raw_bytes = 11 + raw_index
        add(
            "attempt_receipt",
            f"attempts/{receipt_hash}.json",
            receipt_hash,
            receipt_file_hash,
            29 + page_ordinal,
            page_ordinal,
            0,
        )
        add(
            "raw_body",
            f"raw/sha256/{raw_hash[:2]}/{raw_hash}.bin",
            raw_hash,
            raw_hash,
            raw_bytes,
            page_ordinal,
            0,
        )
        raw_logical_bytes += raw_bytes
        attempts.append(
            PilotAttemptAccountingV1(
                page_ordinal=page_ordinal,
                attempt_ordinal=0,
                attempt_receipt_hash=receipt_hash,
                raw_body_sha256=raw_hash,
                raw_body_length=raw_bytes,
                request_started_monotonic_us=0,
                terminal_monotonic_us=0,
                elapsed_monotonic_us=0,
            )
        )
        page_accounting.append(
            PilotPageAccountingV1(
                page_ordinal=page_ordinal,
                page_receipt_hash=_hash(f"{label}:page:{page_ordinal}"),
                row_count=page.expected_row_count,
                attempt_count=1,
            )
        )
    add(
        "normalized_shard",
        f"normalized/{request.request_id}/{normalized_hash}.jsonl",
        normalized_hash,
        normalized_hash,
        37,
    )
    add(
        "manifest",
        f"collections/{request.request_id}/manifest.json",
        manifest_hash,
        _hash(f"{label}:manifest-file"),
        41,
    )
    admission_hash = _hash(f"{label}:admission")
    add(
        "admission_marker",
        f"collections/{request.request_id}/admission.json",
        admission_hash,
        _hash(f"{label}:admission-file"),
        43,
    )
    counts: dict[str, int] = {}
    first: dict[str, PilotLogicalReferenceV1] = {}
    for item in logical:
        counts[item.relative_path] = counts.get(item.relative_path, 0) + 1
        first.setdefault(item.relative_path, item)
    physical = tuple(
        PilotPhysicalFileV1(
            relative_path=path,
            role=first[path].role,
            file_sha256=first[path].file_sha256,
            byte_count=first[path].byte_count,
            logical_reference_count=counts[path],
        )
        for path in sorted(first)
    )
    logical_bytes = sum(item.byte_count for item in logical)
    physical_bytes = sum(item.byte_count for item in physical)
    admission_bytes = logical[-1].byte_count
    return StrictHistoryPilotEvidenceV1(
        evidence_contract_hash=strict_history_pilot_evidence_contract_hash(),
        request_id=request.request_id,
        manifest_hash=manifest_hash,
        history_contract_hash=strict_history_v2_contract_hash(),
        normalized_shard_sha256=normalized_hash,
        page_count=len(pages),
        row_count=request.expected_row_count,
        attempt_count=len(attempts),
        raw_body_reference_count=len(attempts),
        unique_raw_body_count=len(
            {item.raw_body_sha256 for item in attempts}
        ),
        actual_total_raw_body_bytes=raw_logical_bytes,
        unique_physical_raw_body_bytes=sum(
            item.byte_count for item in physical if item.role == "raw_body"
        ),
        manifest_collection_runtime_us=0,
        admission_full_reload_runtime_us=0,
        attempt_elapsed_runtime_us=0,
        observed_monotonic_inter_attempt_sleep_us=0,
        admitted_total_logical_storage_bytes=logical_bytes,
        unique_physical_referenced_bytes=physical_bytes,
        logical_references=tuple(logical),
        physical_files=physical,
        page_accounting=tuple(page_accounting),
        attempt_accounting=tuple(attempts),
        admission_accounting=PilotAdmissionAccountingV1(
            admission_hash=admission_hash,
            graph_logical_storage_bytes=logical_bytes - admission_bytes,
            admission_marker_bytes=admission_bytes,
            admitted_total_logical_storage_bytes=logical_bytes,
            manifest_collection_runtime_us=0,
            admission_full_reload_runtime_us=0,
        ),
        restart_contract_version=STRICT_HISTORY_V2_RESTART_VERSION,
        restart_observation_count=2,
        strict_history_namespace_residue_free=True,
        restart_no_residue_proof=PilotRestartNoResidueProofV1(
            request_id=request.request_id,
            manifest_hash=manifest_hash,
            request_state="complete_verified",
            ready=True,
            temp_paths=(),
            unreferenced_attempt_paths=(),
            unreferenced_raw_paths=(),
            alternate_normalized_paths=(),
            observation_count=2,
            restart_contract_version=STRICT_HISTORY_V2_RESTART_VERSION,
        ),
        writer_lock=PilotWriterLockFactV1(
            status="present_plain_regular",
            symbolic_locator="persistent_sibling_writer_lock_outside_shard_root",
            file_sha256=hashlib.sha256(b"0").hexdigest(),
            byte_count=1,
            link_count=1,
        ),
        pilot_output_layout_status=PILOT_OUTPUT_LAYOUT_STATUS_UNRESOLVED,
        authority_status=PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE,
    )


def _placeholder(plan_hash: str) -> PilotOfficialDocumentPlaceholderV1:
    paths = derive_official_bundle_locators_v1(plan_hash)
    return PilotOfficialDocumentPlaceholderV1(
        endpoint_plan_hash=plan_hash,
        files=tuple(
            PilotOfficialDocumentPlaceholderFileV1(
                role=role,
                relative_path=path,
                artifact_sha256=_hash(f"official:{role}"),
                byte_count=byte_count,
            )
            for role, path, byte_count in zip(
                ("raw_body", "attempt_receipt", "evidence"),
                paths,
                (17, 23, 31),
                strict=True,
            )
        ),
    )


def _headroom_preflight(manifest, auth, plan, step, now):
    receipt = _preflight(manifest, auth, step, now)
    extra = plan.locator_plans[step + 1].extra_remaining_lock_bytes
    return replace(
        receipt,
        free_bytes_before=receipt.free_bytes_before + extra,
        free_bytes_after_reservation=receipt.free_bytes_after_reservation + extra,
    )


def _complete_subject():
    manifest = _manifest()
    plan = build_pilot_output_layout_plan_v1(manifest)
    evidences = tuple(
        _evidence(locator_request, label=f"stage:{index}")
        for index, locator_request in enumerate(
            (
                manifest.endpoint_verification.probe_request,
                *(item.request for item in manifest.shards),
            )
        )
    )
    official = _placeholder(manifest.endpoint_verification.plan_hash)
    stage_accounting = tuple(
        build_pilot_stage_output_accounting_v1(
            pilot_manifest_hash=manifest.manifest_hash,
            locator_plan=locator,
            evidence=evidence,
            official_document_placeholder=official if locator.ordinal == -1 else None,
        )
        for locator, evidence in zip(plan.locator_plans, evidences, strict=True)
    )

    auth = _authorization(manifest)
    now = auth.authorized_at_us + 1
    state = PilotRunStateV1(manifest).with_authorization(auth, now_us=now)
    endpoint_preflight = _headroom_preflight(manifest, auth, plan, -1, now + 1)
    state = state.with_preflight(endpoint_preflight, now_us=now + 1)
    endpoint_intent = _network_intent(
        manifest,
        auth,
        endpoint_preflight,
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=now + 3,
        mono=9_000_000,
    )
    state = state.with_network_intent(endpoint_intent)
    endpoint_evidence = evidences[0]
    endpoint_layout = stage_accounting[0]
    verification = replace(
        _verification(manifest, auth, endpoint_intent, start=now + 10),
        actual_raw_body_bytes=(
            official.files[0].byte_count
            + endpoint_evidence.actual_total_raw_body_bytes
        ),
        actual_storage_bytes=endpoint_layout.dependent_receipt_inventory_bytes,
        official_document_evidence_hash=official.files[-1].artifact_sha256,
        live_history_manifest_hash=endpoint_evidence.manifest_hash,
        live_attempt_receipt_hash=(
            endpoint_evidence.attempt_accounting[0].attempt_receipt_hash
        ),
        live_raw_body_sha256=(
            endpoint_evidence.attempt_accounting[0].raw_body_sha256
        ),
        output_inventory_hash=endpoint_layout.dependent_receipt_inventory_hash,
        output_inventory_entries=(
            endpoint_layout.dependent_receipt_inventory_entries
        ),
    )
    state = state.with_endpoint_verification(verification)
    epoch = verification.completed_at_us
    monotonic = verification.completed_monotonic_us
    for ordinal, (evidence, layout) in enumerate(
        zip(evidences[1:], stage_accounting[1:], strict=True)
    ):
        checked = epoch + 10
        preflight = _headroom_preflight(manifest, auth, plan, ordinal, checked)
        state = state.with_preflight(preflight, now_us=checked)
        intent = _network_intent(
            manifest,
            auth,
            preflight,
            stage="shard_acquisition",
            ordinal=ordinal,
            issued_at=checked + 1,
            mono=monotonic + 1,
        )
        state = state.with_network_intent(intent)
        epoch += manifest.budgets.min_inter_step_spacing_us
        monotonic += manifest.budgets.min_inter_step_spacing_us
        result = replace(
            _shard_result(
                manifest,
                ordinal,
                start=epoch,
                mono=monotonic,
                intent=intent,
            ),
            history_manifest_hash=evidence.manifest_hash,
            actual_pages=evidence.page_count,
            actual_rows=evidence.row_count,
            actual_attempts=evidence.attempt_count,
            actual_raw_body_bytes=evidence.actual_total_raw_body_bytes,
            actual_logical_storage_bytes=(
                evidence.admitted_total_logical_storage_bytes
            ),
            actual_collection_runtime_us=evidence.manifest_collection_runtime_us,
            output_inventory_hash=layout.dependent_receipt_inventory_hash,
            output_inventory_entries=layout.dependent_receipt_inventory_entries,
            output_inventory_bytes=layout.dependent_receipt_inventory_bytes,
        )
        state = state.with_shard_result(result)
        epoch = result.step_completed_at_us
        monotonic = result.step_completed_monotonic_us
    assert state.next_action == "publish_detached_result_anchor"
    expected = derive_expected_pilot_output_inventory_v1(
        state=state,
        stage_accounting=stage_accounting,
    )
    scan_entries = tuple(
        PilotInventoryEntryV1(
            relative_path=item.relative_path,
            artifact_sha256=item.artifact_sha256,
            byte_count=item.byte_count,
        )
        for item in expected
    )
    scan = PilotInventoryScanV1(
        manifest_hash=manifest.manifest_hash,
        entries=scan_entries,
        total_bytes=sum(item.byte_count for item in scan_entries),
        scanned_at_us=epoch + 1,
        scanned_monotonic_us=monotonic + 1,
        clock_domain_id="fixture_clock_domain",
    )
    return manifest, plan, state, evidences, official, stage_accounting, scan


@pytest.fixture(scope="module")
def complete_subject():
    return _complete_subject()


def _rebuild_state_and_scan(
    subject,
    *,
    authorization_output_cap: int | None = None,
    include_lock_headroom: bool = True,
):
    manifest, plan, _old_state, evidences, official, accounting, _old_scan = subject
    auth = _authorization(manifest)
    if authorization_output_cap is not None:
        auth = replace(auth, max_total_output_bytes=authorization_output_cap)

    def preflight(step: int, now: int):
        if include_lock_headroom or step == -1:
            return _headroom_preflight(manifest, auth, plan, step, now)
        receipt = _preflight(manifest, auth, step, now)
        required_after = manifest.budgets.required_free_disk_bytes_after_reservation
        return replace(
            receipt,
            free_bytes_before=receipt.reserved_bytes + required_after,
            free_bytes_after_reservation=required_after,
        )

    now = auth.authorized_at_us + 1
    state = PilotRunStateV1(manifest).with_authorization(auth, now_us=now)
    endpoint_preflight = preflight(-1, now + 1)
    state = state.with_preflight(endpoint_preflight, now_us=now + 1)
    endpoint_intent = _network_intent(
        manifest,
        auth,
        endpoint_preflight,
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=now + 3,
        mono=9_000_000,
    )
    state = state.with_network_intent(endpoint_intent)
    endpoint_evidence = evidences[0]
    endpoint_layout = accounting[0]
    verification = replace(
        _verification(manifest, auth, endpoint_intent, start=now + 10),
        actual_raw_body_bytes=(
            official.files[0].byte_count
            + endpoint_evidence.actual_total_raw_body_bytes
        ),
        actual_storage_bytes=endpoint_layout.dependent_receipt_inventory_bytes,
        official_document_evidence_hash=official.files[-1].artifact_sha256,
        live_history_manifest_hash=endpoint_evidence.manifest_hash,
        live_attempt_receipt_hash=(
            endpoint_evidence.attempt_accounting[0].attempt_receipt_hash
        ),
        live_raw_body_sha256=endpoint_evidence.attempt_accounting[0].raw_body_sha256,
        output_inventory_hash=endpoint_layout.dependent_receipt_inventory_hash,
        output_inventory_entries=endpoint_layout.dependent_receipt_inventory_entries,
    )
    state = state.with_endpoint_verification(verification)
    epoch = verification.completed_at_us
    monotonic = verification.completed_monotonic_us
    for ordinal, (evidence, layout) in enumerate(
        zip(evidences[1:], accounting[1:], strict=True)
    ):
        checked = epoch + 10
        step_preflight = preflight(ordinal, checked)
        state = state.with_preflight(step_preflight, now_us=checked)
        intent = _network_intent(
            manifest,
            auth,
            step_preflight,
            stage="shard_acquisition",
            ordinal=ordinal,
            issued_at=checked + 1,
            mono=monotonic + 1,
        )
        state = state.with_network_intent(intent)
        epoch += manifest.budgets.min_inter_step_spacing_us
        monotonic += manifest.budgets.min_inter_step_spacing_us
        result = replace(
            _shard_result(
                manifest,
                ordinal,
                start=epoch,
                mono=monotonic,
                intent=intent,
            ),
            history_manifest_hash=evidence.manifest_hash,
            actual_pages=evidence.page_count,
            actual_rows=evidence.row_count,
            actual_attempts=evidence.attempt_count,
            actual_raw_body_bytes=evidence.actual_total_raw_body_bytes,
            actual_logical_storage_bytes=evidence.admitted_total_logical_storage_bytes,
            actual_collection_runtime_us=evidence.manifest_collection_runtime_us,
            output_inventory_hash=layout.dependent_receipt_inventory_hash,
            output_inventory_entries=layout.dependent_receipt_inventory_entries,
            output_inventory_bytes=layout.dependent_receipt_inventory_bytes,
        )
        state = state.with_shard_result(result)
        epoch = result.step_completed_at_us
        monotonic = result.step_completed_monotonic_us
    expected = derive_expected_pilot_output_inventory_v1(
        state=state,
        stage_accounting=accounting,
    )
    entries = tuple(
        PilotInventoryEntryV1(
            relative_path=item.relative_path,
            artifact_sha256=item.artifact_sha256,
            byte_count=item.byte_count,
        )
        for item in expected
    )
    scan = PilotInventoryScanV1(
        manifest_hash=manifest.manifest_hash,
        entries=entries,
        total_bytes=sum(item.byte_count for item in entries),
        scanned_at_us=epoch + 1,
        scanned_monotonic_us=monotonic + 1,
        clock_domain_id="fixture_clock_domain",
    )
    return state, scan


def _coherent_readiness_payload(result, *, state, accounting, scan):
    expected = derive_expected_pilot_output_inventory_v1(
        state=state,
        stage_accounting=accounting,
    )
    total = sum(item.byte_count for item in expected)
    groups = {
        source: tuple(item for item in expected if item.source == source)
        for source in (
            "strict_history_physical",
            "run_control",
            "infrastructure_writer_lock",
            "official_document_placeholder",
        )
    }
    payload = copy.deepcopy(result.as_dict())
    payload.update(
        state=state.as_dict(),
        state_hash=state.state_hash,
        stage_accounting=[item.as_dict() for item in accounting],
        logical_reference_entries=sum(
            item.logical_reference_entries for item in accounting
        ),
        logical_reference_bytes=sum(item.logical_reference_bytes for item in accounting),
        unique_stage_physical_entries=len(groups["strict_history_physical"]),
        unique_stage_physical_bytes=sum(
            item.byte_count for item in groups["strict_history_physical"]
        ),
        run_control_entries=len(groups["run_control"]),
        run_control_bytes=sum(item.byte_count for item in groups["run_control"]),
        infrastructure_lock_entries=len(groups["infrastructure_writer_lock"]),
        infrastructure_lock_bytes=sum(
            item.byte_count for item in groups["infrastructure_writer_lock"]
        ),
        official_document_placeholder_entries=len(
            groups["official_document_placeholder"]
        ),
        official_document_placeholder_bytes=sum(
            item.byte_count for item in groups["official_document_placeholder"]
        ),
        expected_physical_entries_detail=[item.as_dict() for item in expected],
        expected_inventory_hash=_payload_hash(
            {
                "domain": "mexc_public_qa_pilot_exact_global_physical_inventory_v1",
                "manifest_hash": state.manifest.manifest_hash,
                "entries": [item.as_dict() for item in expected],
                "total_bytes": total,
            }
        ),
        expected_inventory_entries=len(expected),
        expected_inventory_bytes=total,
        inventory_scan={
            "manifest_hash": scan.manifest_hash,
            "entries": [item.as_dict() for item in scan.entries],
            "total_bytes": scan.total_bytes,
            "scanned_at_us": scan.scanned_at_us,
            "scanned_monotonic_us": scan.scanned_monotonic_us,
            "clock_domain_id": scan.clock_domain_id,
            "contract_version": scan.contract_version,
        },
        observed_physical_entries_detail=[item.as_dict() for item in scan.entries],
        observed_inventory_hash=scan.inventory_hash,
        observed_inventory_entries=len(scan.entries),
        observed_inventory_bytes=scan.total_bytes,
    )
    return payload


def test_contract_binds_all_frozen_dependencies_and_plan_roundtrips() -> None:
    manifest = _manifest()
    plan = build_pilot_output_layout_plan_v1(manifest)
    assert pilot_output_layout_contract_hash() == (
        "cb19e6a53d122139ec3a76b4d54c67c04a31da9550db9ca8c186496c6bb8e934"
    )
    assert pilot_run_contract_hash() == (
        "f3d642d436e9d4a44e65f35c6ea8375bd92b4b36b30f1c86af54936a608ce65e"
    )
    assert mexc_pilot_local_store_contract_hash() == (
        "21f27ec667d588ac254b893c5f25e44634cc2de1f8567efafc85d08fccca94ab"
    )
    assert pilot_local_executor_contract_hash() == (
        "72c206bc2f22a8101a7d6fdc97458e865a6c4c3e5ed7290c64c1ca8c3594fc31"
    )
    assert pilot_local_coordinator_contract_hash() == (
        "a19d002c04a3ab09d16a18bfdce66adcbc43399200ee092ffd0b179abb9016fc"
    )
    assert strict_history_pilot_evidence_contract_hash() == (
        "a546b37de9ed2da04eefb8d607b98719a09ab8378c2ab1d459eac02ecb899b8e"
    )
    assert PilotOutputLayoutPlanV1.from_dict(plan.as_dict()) == plan


def test_nested_writer_lock_and_exact_external_official_locators() -> None:
    assert derive_persistent_writer_lock_locator_v1("alpha/beta/gamma") == (
        "alpha/beta/.gamma.strict-history-v2.writer.lock"
    )
    plan_hash = "a" * 64
    assert derive_official_bundle_root_v1(plan_hash) == (
        f"endpoint-evidence/{plan_hash}/official"
    )
    assert derive_official_bundle_locators_v1(plan_hash) == (
        f"endpoint-evidence/{plan_hash}/official/attempt-000.body.bin",
        f"endpoint-evidence/{plan_hash}/official/attempt-000.receipt.json",
        f"endpoint-evidence/{plan_hash}/official/evidence.json",
    )


def test_windows_component_length_boundary_is_exact() -> None:
    accepted = PilotOutputPhysicalEntryV1(
        relative_path="a" * 255,
        artifact_sha256=_hash("component-boundary"),
        byte_count=1,
        source="run_control",
        role="manifest",
        stage="run_control",
        ordinal=-2,
        logical_reference_count=0,
    )
    assert len(accepted.relative_path) == 255
    payload = accepted.as_dict()
    payload["relative_path"] = "a" * 256
    with pytest.raises(PilotOutputLayoutContractError, match="path_is_not_windows_safe"):
        PilotOutputPhysicalEntryV1.from_dict(payload)


def test_physical_entry_parser_enforces_source_stage_and_exact_lock_role() -> None:
    base = PilotOutputPhysicalEntryV1(
        relative_path="run-control/manifest.json",
        artifact_sha256=_hash("physical-entry"),
        byte_count=1,
        source="run_control",
        role="manifest",
        stage="run_control",
        ordinal=-2,
        logical_reference_count=0,
    ).as_dict()
    wrong_stage = dict(base)
    wrong_stage.update(stage="endpoint_verification", ordinal=-1)
    with pytest.raises(PilotOutputLayoutContractError, match="source_stage"):
        PilotOutputPhysicalEntryV1.from_dict(wrong_stage)

    wrong_official = dict(base)
    wrong_official.update(
        source="official_document_placeholder",
        stage="endpoint_verification",
        ordinal=-1,
        role="arbitrary",
    )
    with pytest.raises(PilotOutputLayoutContractError, match="official_physical_entry_role"):
        PilotOutputPhysicalEntryV1.from_dict(wrong_official)

    empty_official = dict(wrong_official)
    empty_official.update(
        role="official_document_raw_body_unresolved",
        byte_count=0,
    )
    with pytest.raises(PilotOutputLayoutContractError, match="official_physical_entry_is_empty"):
        PilotOutputPhysicalEntryV1.from_dict(empty_official)

    wrong_lock = dict(base)
    wrong_lock.update(
        source="infrastructure_writer_lock",
        stage="shard_acquisition",
        ordinal=0,
        role="not_the_lock_role",
    )
    with pytest.raises(PilotOutputLayoutContractError, match="writer_lock_physical_entry"):
        PilotOutputPhysicalEntryV1.from_dict(wrong_lock)


def test_plan_accounts_lock_bytes_file_credit_and_scan_directories() -> None:
    manifest = _manifest()
    plan = build_pilot_output_layout_plan_v1(manifest)
    assert plan.infrastructure_lock_entries == 1 + len(manifest.shards)
    assert plan.infrastructure_lock_bytes == plan.infrastructure_lock_entries
    assert plan.required_max_inventory_entries == (
        manifest.planned_reservations["inventory_entries"] + 1
    )
    assert plan.required_max_total_output_bytes == (
        manifest.planned_reservations["total_output_bytes"]
        + plan.infrastructure_lock_bytes
    )
    assert plan.required_scan_traversal_entries == (
        plan.required_max_inventory_entries + plan.maximum_scan_directory_entries
    )
    assert plan.maximum_scan_directory_entries == 334


def test_preflight_must_cover_remaining_lock_bytes() -> None:
    manifest = _manifest()
    plan = build_pilot_output_layout_plan_v1(manifest)
    auth = _authorization(manifest)
    receipt = _preflight(manifest, auth, -1, auth.authorized_at_us + 1)
    receipt = replace(
        receipt,
        free_bytes_before=(
            receipt.reserved_bytes
            + manifest.budgets.required_free_disk_bytes_after_reservation
        ),
        free_bytes_after_reservation=(
            manifest.budgets.required_free_disk_bytes_after_reservation
        ),
    )
    with pytest.raises(PilotOutputLayoutBudgetStop, match="lock_headroom"):
        plan.validate_preflight(receipt)
    plan.validate_preflight(
        _headroom_preflight(
            manifest,
            auth,
            plan,
            -1,
            auth.authorized_at_us + 1,
        )
    )


def test_u5_output_cap_must_cover_all_writer_lock_bytes() -> None:
    manifest = _manifest()
    plan = build_pilot_output_layout_plan_v1(manifest)
    narrow = replace(
        _authorization(manifest),
        max_total_output_bytes=manifest.planned_reservations["total_output_bytes"],
    )
    with pytest.raises(PilotOutputLayoutBudgetStop, match="u5_output_cap"):
        plan.validate_authorization(narrow)
    plan.validate_authorization(_authorization(manifest))


def test_duplicate_raw_cas_is_logical_twice_but_physical_once() -> None:
    manifest = _manifest()
    plan = build_pilot_output_layout_plan_v1(manifest)
    request = manifest.shards[0].request
    evidence = _evidence(request, label="dedup", duplicate_raw=True)
    accounting = build_pilot_stage_output_accounting_v1(
        pilot_manifest_hash=manifest.manifest_hash,
        locator_plan=plan.locator_plans[1],
        evidence=evidence,
    )
    duplicate_raw = tuple(
        item
        for item in accounting.expected_physical_entries
        if item.role == "raw_body" and item.logical_reference_count == 2
    )
    assert len(duplicate_raw) == 1
    assert accounting.logical_reference_entries == 2 * evidence.attempt_count + 4
    assert accounting.unique_history_physical_entries == len(evidence.physical_files)
    assert accounting.logical_reference_bytes > accounting.unique_history_physical_bytes
    assert sum(
        item.byte_count
        for item in accounting.expected_physical_entries
        if item.role == "raw_body"
    ) == evidence.unique_physical_raw_body_bytes


def test_writer_lock_omission_and_wrong_official_bundle_stop() -> None:
    manifest = _manifest()
    plan = build_pilot_output_layout_plan_v1(manifest)
    evidence = _evidence(manifest.shards[0].request, label="lock")
    absent = replace(
        evidence,
        writer_lock=PilotWriterLockFactV1(
            status="absent",
            symbolic_locator="persistent_sibling_writer_lock_outside_shard_root",
            file_sha256=None,
            byte_count=0,
            link_count=0,
        ),
    )
    with pytest.raises(PilotOutputLayoutInventoryStop, match="writer_lock"):
        build_pilot_stage_output_accounting_v1(
            pilot_manifest_hash=manifest.manifest_hash,
            locator_plan=plan.locator_plans[1],
            evidence=absent,
        )
    placeholder = _placeholder(manifest.endpoint_verification.plan_hash)
    payload = placeholder.as_dict()
    payload["files"][0]["relative_path"] = (
        manifest.endpoint_verification.relative_artifact_root + "/official.raw"
    )
    with pytest.raises(PilotOutputLayoutContractError, match="bundle_locator"):
        PilotOfficialDocumentPlaceholderV1.from_dict(payload)


def test_standalone_stage_rejects_official_namespace_nested_with_strict_root() -> None:
    manifest = _manifest()
    endpoint = manifest.endpoint_verification
    official = _placeholder(endpoint.plan_hash)
    parent_root = f"endpoint-evidence/{endpoint.plan_hash}"
    locator = PilotOutputLocatorPlanV1(
        stage="endpoint_verification",
        ordinal=-1,
        request_id=endpoint.probe_request.request_id,
        plan_binding_hash=endpoint.plan_hash,
        relative_artifact_root=parent_root,
        writer_lock_relative_path=derive_persistent_writer_lock_locator_v1(
            parent_root
        ),
        frozen_remaining_storage_reservation=1,
        extra_remaining_lock_entries=1,
        extra_remaining_lock_bytes=1,
    )
    with pytest.raises(PilotOutputLayoutInventoryStop, match="namespace_overlap"):
        build_pilot_stage_output_accounting_v1(
            pilot_manifest_hash=manifest.manifest_hash,
            locator_plan=locator,
            evidence=_evidence(endpoint.probe_request, label="overlap"),
            official_document_placeholder=official,
        )


def test_global_exact_union_roundtrip_and_terminal_stop(complete_subject) -> None:
    manifest, _plan, state, evidences, official, _accounting, scan = complete_subject
    result = assess_pilot_output_layout_v1(
        manifest=manifest,
        state=state,
        inventory_scan=scan,
        stage_evidences=evidences,
        official_document_placeholder=official,
    )
    assert result.exact_file_inventory_match is True
    assert result.directory_namespace_exact is False
    assert result.completed_stage_set_exact is True
    assert result.readiness_phase == "post_result_candidate_pre_anchor"
    assert result.infrastructure_lock_entries == 1 + len(manifest.shards)
    assert result.infrastructure_lock_bytes == result.infrastructure_lock_entries
    assert result.official_document_placeholder_entries == 3
    assert result.expected_inventory_entries == len(scan.entries)
    assert result.expected_inventory_bytes == scan.total_bytes
    assert result.observed_inventory_hash == scan.inventory_hash
    assert "no_cross_evidence_atomic_snapshot" in result.snapshot_boundary
    assert result.writer_boundary == "static_or_cooperating_writer_required"
    assert PilotOutputReadinessAssessmentV1.from_dict(result.as_dict()) == result
    with pytest.raises(FrozenInstanceError):
        result.exact_file_inventory_match = False  # type: ignore[misc]
    with pytest.raises(PilotOutputLayoutTerminalStop, match="schema_and_publisher"):
        result.require_terminal_compatible()


@pytest.mark.parametrize("mode", ("stale_epoch", "stale_monotonic", "foreign_clock"))
def test_inventory_scan_must_share_run_clock_and_follow_last_shard(
    complete_subject,
    mode,
) -> None:
    manifest, _plan, state, evidences, official, _accounting, scan = complete_subject
    last = state.shard_results[-1]
    changes = {
        "stale_epoch": {"scanned_at_us": last.step_completed_at_us - 1},
        "stale_monotonic": {
            "scanned_monotonic_us": last.step_completed_monotonic_us - 1
        },
        "foreign_clock": {"clock_domain_id": "foreign_clock_domain"},
    }[mode]
    changed = replace(scan, **changes)
    with pytest.raises(PilotOutputLayoutInventoryStop, match="clock_or_freshness"):
        assess_pilot_output_layout_v1(
            manifest=manifest,
            state=state,
            inventory_scan=changed,
            stage_evidences=evidences,
            official_document_placeholder=official,
        )


def test_shard_receipt_sleep_must_match_reloaded_attempt_timing(
    complete_subject,
) -> None:
    manifest, _plan, state, evidences, official, _accounting, scan = complete_subject
    original = evidences[1]
    attempts = tuple(
        replace(
            item,
            request_started_monotonic_us=index,
            terminal_monotonic_us=index,
            elapsed_monotonic_us=0,
        )
        for index, item in enumerate(original.attempt_accounting)
    )
    sleep = len(attempts) - 1
    changed_evidence = replace(
        original,
        attempt_accounting=attempts,
        observed_monotonic_inter_attempt_sleep_us=sleep,
        manifest_collection_runtime_us=sleep,
        admission_full_reload_runtime_us=sleep,
        admission_accounting=replace(
            original.admission_accounting,
            manifest_collection_runtime_us=sleep,
            admission_full_reload_runtime_us=sleep,
        ),
    )
    first_result = replace(
        state.shard_results[0],
        actual_collection_runtime_us=sleep,
    )
    changed_state = replace(
        state,
        shard_results=(first_result, *state.shard_results[1:]),
    )
    with pytest.raises(PilotOutputLayoutInventoryStop, match="shard_evidence"):
        assess_pilot_output_layout_v1(
            manifest=manifest,
            state=changed_state,
            inventory_scan=scan,
            stage_evidences=(evidences[0], changed_evidence, *evidences[2:]),
            official_document_placeholder=official,
        )


def test_readiness_parser_rejects_tampered_embedded_scan_timing(
    complete_subject,
) -> None:
    manifest, _plan, state, evidences, official, _accounting, scan = complete_subject
    result = assess_pilot_output_layout_v1(
        manifest=manifest,
        state=state,
        inventory_scan=scan,
        stage_evidences=evidences,
        official_document_placeholder=official,
    )
    payload = copy.deepcopy(result.as_dict())
    payload["inventory_scan"]["clock_domain_id"] = "foreign_clock_domain"
    with pytest.raises(PilotOutputLayoutContractError):
        PilotOutputReadinessAssessmentV1.from_dict(payload)


@pytest.mark.parametrize("mode", ("missing", "extra", "changed"))
def test_inventory_missing_extra_and_changed_stop(complete_subject, mode) -> None:
    manifest, _plan, state, evidences, official, _accounting, scan = complete_subject
    entries = list(scan.entries)
    if mode == "missing":
        entries.pop()
    elif mode == "extra":
        entries.append(PilotInventoryEntryV1("unexpected.bin", "f" * 64, 1))
    else:
        entries[0] = replace(entries[0], artifact_sha256="e" * 64)
    entries = sorted(entries, key=lambda item: item.relative_path)
    changed = PilotInventoryScanV1(
        manifest_hash=scan.manifest_hash,
        entries=tuple(entries),
        total_bytes=sum(item.byte_count for item in entries),
        scanned_at_us=scan.scanned_at_us,
        scanned_monotonic_us=scan.scanned_monotonic_us,
        clock_domain_id=scan.clock_domain_id,
    )
    with pytest.raises(PilotOutputLayoutInventoryStop, match="union_mismatch"):
        assess_pilot_output_layout_v1(
            manifest=manifest,
            state=state,
            inventory_scan=changed,
            stage_evidences=evidences,
            official_document_placeholder=official,
        )


def test_adversarial_stage_parser_rejects_wrong_prefix_stage_and_hash(
    complete_subject,
) -> None:
    accounting = complete_subject[5][1]
    for mutate in ("path", "stage", "hash"):
        payload = copy.deepcopy(accounting.as_dict())
        entry = next(
            item
            for item in payload["expected_physical_entries"]
            if item["source"] == "strict_history_physical"
        )
        if mutate == "path":
            entry["relative_path"] = "wrong-prefix/scope.json"
        elif mutate == "stage":
            entry["ordinal"] = 3
        else:
            payload["unique_history_physical_inventory_hash"] = "0" * 64
        with pytest.raises(PilotOutputLayoutContractError):
            PilotStageOutputAccountingV1.from_dict(payload)


def test_adversarial_readiness_parser_rejects_divergent_projections(
    complete_subject,
) -> None:
    manifest, _plan, state, evidences, official, _accounting, scan = complete_subject
    result = assess_pilot_output_layout_v1(
        manifest=manifest,
        state=state,
        inventory_scan=scan,
        stage_evidences=evidences,
        official_document_placeholder=official,
    )
    for field, value in (
        ("expected_inventory_bytes", result.expected_inventory_bytes + 1),
        ("observed_inventory_hash", "0" * 64),
        ("completed_stage_count", result.completed_stage_count - 1),
        ("exact_file_inventory_match", False),
    ):
        payload = copy.deepcopy(result.as_dict())
        payload[field] = value
        with pytest.raises(PilotOutputLayoutContractError):
            PilotOutputReadinessAssessmentV1.from_dict(payload)


def test_readiness_constructor_and_parser_replay_narrow_u5_cap(
    complete_subject,
) -> None:
    manifest, _plan, state, evidences, official, accounting, scan = complete_subject
    result = assess_pilot_output_layout_v1(
        manifest=manifest,
        state=state,
        inventory_scan=scan,
        stage_evidences=evidences,
        official_document_placeholder=official,
    )
    narrow_state, narrow_scan = _rebuild_state_and_scan(
        complete_subject,
        authorization_output_cap=manifest.planned_reservations["total_output_bytes"],
    )
    with pytest.raises(PilotOutputLayoutBudgetStop, match="u5_output_cap"):
        replace(result, state=narrow_state)
    payload = _coherent_readiness_payload(
        result,
        state=narrow_state,
        accounting=accounting,
        scan=narrow_scan,
    )
    with pytest.raises(PilotOutputLayoutBudgetStop, match="u5_output_cap"):
        PilotOutputReadinessAssessmentV1.from_dict(payload)


def test_readiness_constructor_and_parser_replay_lock_preflight_headroom(
    complete_subject,
) -> None:
    manifest, _plan, state, evidences, official, accounting, scan = complete_subject
    result = assess_pilot_output_layout_v1(
        manifest=manifest,
        state=state,
        inventory_scan=scan,
        stage_evidences=evidences,
        official_document_placeholder=official,
    )
    underheaded_state, underheaded_scan = _rebuild_state_and_scan(
        complete_subject,
        include_lock_headroom=False,
    )
    with pytest.raises(PilotOutputLayoutBudgetStop, match="lock_headroom"):
        replace(result, state=underheaded_state)
    payload = _coherent_readiness_payload(
        result,
        state=underheaded_state,
        accounting=accounting,
        scan=underheaded_scan,
    )
    with pytest.raises(PilotOutputLayoutBudgetStop, match="lock_headroom"):
        PilotOutputReadinessAssessmentV1.from_dict(payload)


def test_readiness_constructor_and_parser_replay_shard_receipt_evidence_binding(
    complete_subject,
) -> None:
    manifest, plan, state, evidences, official, accounting, scan = complete_subject
    result = assess_pilot_output_layout_v1(
        manifest=manifest,
        state=state,
        inventory_scan=scan,
        stage_evidences=evidences,
        official_document_placeholder=official,
    )
    swapped_evidence = _evidence(
        manifest.shards[0].request,
        label="adversarial-swapped-shard-zero",
    )
    swapped_item = build_pilot_stage_output_accounting_v1(
        pilot_manifest_hash=manifest.manifest_hash,
        locator_plan=plan.locator_plans[1],
        evidence=swapped_evidence,
    )
    swapped_accounting = (accounting[0], swapped_item, *accounting[2:])
    expected = derive_expected_pilot_output_inventory_v1(
        state=state,
        stage_accounting=swapped_accounting,
    )
    swapped_entries = tuple(
        PilotInventoryEntryV1(
            relative_path=item.relative_path,
            artifact_sha256=item.artifact_sha256,
            byte_count=item.byte_count,
        )
        for item in expected
    )
    swapped_scan = PilotInventoryScanV1(
        manifest_hash=manifest.manifest_hash,
        entries=swapped_entries,
        total_bytes=sum(item.byte_count for item in swapped_entries),
        scanned_at_us=scan.scanned_at_us,
        scanned_monotonic_us=scan.scanned_monotonic_us,
        clock_domain_id=scan.clock_domain_id,
    )
    with pytest.raises(PilotOutputLayoutInventoryStop, match="shard_evidence"):
        replace(result, stage_accounting=swapped_accounting)
    payload = _coherent_readiness_payload(
        result,
        state=state,
        accounting=swapped_accounting,
        scan=swapped_scan,
    )
    with pytest.raises(PilotOutputLayoutInventoryStop, match="shard_evidence"):
        PilotOutputReadinessAssessmentV1.from_dict(payload)


def test_plan_parser_rejects_tampered_per_step_headroom() -> None:
    payload = build_pilot_output_layout_plan_v1(_manifest()).as_dict()
    payload["locator_plans"][1]["extra_remaining_lock_bytes"] -= 1
    with pytest.raises(PilotOutputLayoutContractError):
        PilotOutputLayoutPlanV1.from_dict(payload)

    payload = build_pilot_output_layout_plan_v1(_manifest()).as_dict()
    payload["locator_plans"][-1]["extra_remaining_lock_bytes"] = True
    with pytest.raises(PilotOutputLayoutContractError, match="extra_remaining_lock_bytes"):
        PilotOutputLayoutPlanV1.from_dict(payload)


def test_global_cap_underreservation_stops_before_readiness() -> None:
    base = _manifest()
    reservations = base.planned_reservations
    budgets_payload = base.budgets.as_dict()
    budgets_payload["max_inventory_entries"] = reservations["inventory_entries"]
    budgets = PilotGlobalBudgetsV1.from_dict(budgets_payload)
    manifest = replace(base, budgets=budgets)
    with pytest.raises(PilotOutputLayoutBudgetStop):
        build_pilot_output_layout_plan_v1(manifest)


def test_scan_traversal_cap_underreservation_is_distinct_from_file_cap() -> None:
    base = _manifest()
    budgets_payload = base.budgets.as_dict()
    budgets_payload["max_inventory_entries"] = 600
    manifest = replace(base, budgets=PilotGlobalBudgetsV1.from_dict(budgets_payload))
    with pytest.raises(PilotOutputLayoutBudgetStop, match="directory_traversal"):
        build_pilot_output_layout_plan_v1(manifest)


def test_output_byte_cap_must_cover_every_one_byte_lock() -> None:
    base = _manifest()
    reservations = base.planned_reservations
    budgets_payload = base.budgets.as_dict()
    budgets_payload["max_total_logical_storage_bytes"] = reservations[
        "logical_storage_bytes"
    ]
    budgets_payload["max_run_control_bytes"] = reservations["run_control_bytes"]
    budgets_payload["max_total_output_bytes"] = reservations["total_output_bytes"]
    budgets_payload["min_free_disk_bytes_before_run"] = (
        reservations["total_output_bytes"]
        + budgets_payload["required_free_disk_bytes_after_reservation"]
    )
    manifest = replace(base, budgets=PilotGlobalBudgetsV1.from_dict(budgets_payload))
    with pytest.raises(PilotOutputLayoutBudgetStop, match="byte_cap"):
        build_pilot_output_layout_plan_v1(manifest)


def test_same_physical_path_collision_is_rejected(complete_subject) -> None:
    state = complete_subject[2]
    accounting = list(complete_subject[5])
    second = accounting[1]
    first_entry = second.expected_physical_entries[0]
    forged_entries = tuple(
        sorted(
            (
                replace(
                    second.expected_physical_entries[0],
                    relative_path="run-control/manifest.json",
                ),
                *second.expected_physical_entries[1:],
            ),
            key=lambda item: item.relative_path,
        )
    )
    with pytest.raises(PilotOutputLayoutContractError):
        accounting[1] = replace(
            second,
            expected_physical_entries=forged_entries,
            expected_physical_inventory_hash=first_entry.artifact_sha256,
        )
        derive_expected_pilot_output_inventory_v1(
            state=state,
            stage_accounting=tuple(accounting),
        )


def test_result_candidate_is_mandatory_in_post_candidate_pre_anchor_union(
    complete_subject,
) -> None:
    manifest, _plan, state, evidences, official, _accounting, scan = complete_subject
    entries = tuple(
        item
        for item in scan.entries
        if item.relative_path != "run-control/result-candidate.json"
    )
    changed = replace(
        scan,
        entries=entries,
        total_bytes=sum(item.byte_count for item in entries),
    )
    with pytest.raises(PilotOutputLayoutInventoryStop, match="union_mismatch"):
        assess_pilot_output_layout_v1(
            manifest=manifest,
            state=state,
            inventory_scan=changed,
            stage_evidences=evidences,
            official_document_placeholder=official,
        )
