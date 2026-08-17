from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import trading.market_data.mexc_pilot_local_coordinator as coordinator_module
from trading.market_data.mexc_pilot_local_coordinator import (
    MexcPilotLocalCoordinatorReadinessV1,
    PILOT_LOCAL_COORDINATOR_CONTRACT_VERSION,
    PilotCoordinatorBindingsV1,
    PilotCoordinatorReadinessAssessmentV1,
    PilotFreshHistoryEvidenceV1,
    PilotFreshOutputSnapshotV1,
    PilotLocalCoordinatorCallbackStopError,
    PilotLocalCoordinatorContractError,
    PilotLocalCoordinatorEvidenceError,
    PilotLocalCoordinatorPreCallbackError,
    PilotReviewedFakeRunnerBindingV1,
    pilot_local_coordinator_contract_hash,
)
from trading.market_data.mexc_pilot_local_executor import (
    EndpointStageDraftV1,
    PILOT_STAGE_FAILURE_DRAFT_VERSION,
    StageFailureDraftV1,
)
from trading.market_data.mexc_pilot_local_store import (
    PilotClockSampleV1,
    PilotIntentClaimResultV1,
    PilotInventoryEntryV1,
    PilotInventoryScanV1,
)
from trading.market_data.mexc_pilot_run import pilot_run_contract_hash
from trading.market_data.strict_history import strict_history_contract_hash
from trading.market_data.strict_history_v2 import strict_history_v2_contract_hash
from trading.market_data.strict_history_v2 import (
    HistoryRestartReportV1,
    HistoryRestartRequestStateV1,
)


H = "a" * 64
RUNNER_HASH = "b" * 64
REVIEW_HASH = "c" * 64
FIXTURE_HASH = "d" * 64
INTENT_HASH = "e" * 64


class _DualClock:
    contract_version = "fake_clock_v1"
    contract_hash = "1" * 64
    clock_domain_id = "fake_clock_domain"

    def sample(self) -> PilotClockSampleV1:
        return PilotClockSampleV1(1_000_000, 500_000, self.clock_domain_id)

    def epoch_us(self) -> int:
        return self.sample().epoch_us

    def monotonic_us(self) -> int:
        return self.sample().monotonic_us

    def sleep_us(self, duration_us: int) -> None:
        assert duration_us >= 0


class _DualSink:
    contract_version = "fake_sink_v1"
    contract_hash = "2" * 64
    domain_id = "fake_sink_domain"

    def anchor(self, request):
        raise AssertionError("not used by readiness coordinator")

    def create_once(self, subject, *, clock):
        raise AssertionError("terminal anchoring is forbidden")

    def reload(self, evidence_hash):
        raise AssertionError("terminal anchoring is forbidden")


class _FailureRunner:
    contract_version = "fake_endpoint_runner_v1"
    contract_hash = RUNNER_HASH
    review_evidence_hash = REVIEW_HASH
    fixture_set_hash = FIXTURE_HASH
    execution_mode = "reviewed_fake_local_fixture_only"
    network_capable = False
    environment_access_permitted = False
    production_use_permitted = False

    def __init__(self, *, raises: bool = False) -> None:
        self.calls = 0
        self.raises = raises

    def execute(self, **kwargs):
        self.calls += 1
        if self.raises:
            raise RuntimeError("fixture failure")
        intent = kwargs["network_intent"]
        state_auth = kwargs["authorization"]
        return StageFailureDraftV1(
            manifest_hash=kwargs["manifest"].manifest_hash,
            authorization_receipt_hash=state_auth.receipt_hash,
            network_intent_hash=intent.intent_hash,
            stage="endpoint_verification",
            ordinal=-1,
            clock_domain_id="fake_clock_domain",
            step_started_at_us=100,
            step_completed_at_us=200,
            step_started_monotonic_us=10,
            step_completed_monotonic_us=20,
            error_code="fixture_failure",
            error_evidence_hash="f" * 64,
        )


class _FakeStore:
    def __init__(self, bindings: PilotCoordinatorBindingsV1) -> None:
        self.clock = _DualClock()
        self.detached_evidence_sink = _DualSink()
        self.output_root = Path("C:/fake/pilot-output")
        self.runtime_authority_binding = SimpleNamespace(
            coordinator_implementation_contract_version=bindings.contract_version,
            coordinator_implementation_contract_hash=bindings.binding_hash,
            clock_domain_id="fake_clock_domain",
        )
        self.executor_bindings = SimpleNamespace(
            endpoint_runner_contract_version="fake_endpoint_runner_v1",
            endpoint_runner_contract_hash=RUNNER_HASH,
            shard_runner_contract_version="fake_shard_runner_v1",
            shard_runner_contract_hash="3" * 64,
        )
        self.manifest = SimpleNamespace(manifest_hash=H, shards=())
        self.intent = SimpleNamespace(
            stage="endpoint_verification",
            ordinal=-1,
            intent_hash=INTENT_HASH,
            relative_artifact_root="verification/fake",
        )
        self.authorization = SimpleNamespace(receipt_hash="4" * 64)
        self.state = SimpleNamespace(
            authorization=self.authorization,
            preflight_receipts=(SimpleNamespace(),),
            network_intents=(self.intent,),
            next_action="run_local_preflight:-1",
        )
        self.calls: list[str] = []
        self.callback_calls = 0
        self.reject_before_callback = False
        self.before_adapter = None
        self.inventory_scan_override = None
        self.events: list[str] = []

    def measure_and_publish_preflight(self, expected_state):
        self.calls.append("preflight")
        return self.state

    def claim_and_seal_next_intent(self, state, session_capability):
        self.calls.append("claim")
        return PilotIntentClaimResultV1(self.state, self.intent, object())

    def run_owned_intent_once(self, state, capability, runner):
        self.calls.append("run_once")
        if self.reject_before_callback:
            raise RuntimeError("store gate rejected")
        self.callback_calls += 1
        if self.before_adapter is not None:
            self.before_adapter()
        return runner(self.intent)

    def reconstruct_authoritative_state(self):
        self.calls.append("reconstruct")
        return SimpleNamespace(
            network_permitted=False,
            stop_code="unresolved_network_intent_after_restart",
        )

    def scan_inventory(self):
        self.calls.append("inventory")
        self.events.append("scan")
        if self.inventory_scan_override is not None:
            return self.inventory_scan_override
        return PilotInventoryScanV1(
            manifest_hash=H,
            entries=(
                PilotInventoryEntryV1(
                    relative_path="run-control/manifest.json",
                    artifact_sha256="5" * 64,
                    byte_count=100,
                ),
            ),
            total_bytes=100,
            scanned_at_us=1_000_000,
            scanned_monotonic_us=500_000,
            clock_domain_id="fake_clock_domain",
        )


def _binding() -> PilotReviewedFakeRunnerBindingV1:
    return PilotReviewedFakeRunnerBindingV1(
        stage="endpoint_verification",
        runner_contract_version="fake_endpoint_runner_v1",
        runner_contract_hash=RUNNER_HASH,
        review_evidence_hash=REVIEW_HASH,
        fixture_set_hash=FIXTURE_HASH,
    )


def _coordinator(monkeypatch):
    bindings = PilotCoordinatorBindingsV1.frozen(
        reviewed_fake_runner_binding_hashes=(_binding().binding_hash,),
    )
    store = _FakeStore(bindings)
    monkeypatch.setattr(coordinator_module, "MexcPilotLocalStoreV1", _FakeStore)
    return (
        MexcPilotLocalCoordinatorReadinessV1(store=store, bindings=bindings),
        store,
    )


def test_contract_binds_all_frozen_lines_and_has_no_terminal_publisher() -> None:
    review = _binding()
    bindings = PilotCoordinatorBindingsV1.frozen(
        reviewed_fake_runner_binding_hashes=(review.binding_hash,),
    )
    assert bindings.coordinator_contract_version == (
        PILOT_LOCAL_COORDINATOR_CONTRACT_VERSION
    )
    assert pilot_local_coordinator_contract_hash() == (
        "a19d002c04a3ab09d16a18bfdce66adcbc43399200ee092ffd0b179abb9016fc"
    )
    assert bindings.coordinator_contract_hash == pilot_local_coordinator_contract_hash()
    assert bindings.pilot_run_contract_hash == pilot_run_contract_hash()
    assert bindings.strict_history_v1_contract_hash == strict_history_contract_hash()
    assert bindings.strict_history_v2_contract_hash == strict_history_v2_contract_hash()
    assert bindings.terminal_publisher_contract_version == "absent"
    assert bindings.reviewed_fake_runner_binding_hashes == (review.binding_hash,)
    assert len(bindings.binding_hash) == 64


def test_frozen_bindings_reject_one_dependency_substitution() -> None:
    values = PilotCoordinatorBindingsV1.frozen(
        reviewed_fake_runner_binding_hashes=(_binding().binding_hash,),
    ).as_dict()
    values["strict_history_v2_contract_hash"] = "0" * 64
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="do_not_match_frozen_dependencies",
    ):
        PilotCoordinatorBindingsV1(**values)


def test_coordinator_rejects_store_and_bindings_subclasses(monkeypatch) -> None:
    review = _binding()
    bindings = PilotCoordinatorBindingsV1.frozen(
        reviewed_fake_runner_binding_hashes=(review.binding_hash,),
    )

    class StoreSubclass(_FakeStore):
        pass

    monkeypatch.setattr(coordinator_module, "MexcPilotLocalStoreV1", _FakeStore)
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="requires_frozen_local_store_v1",
    ):
        MexcPilotLocalCoordinatorReadinessV1(
            store=StoreSubclass(bindings),
            bindings=bindings,
        )

    class BindingsSubclass(PilotCoordinatorBindingsV1):
        pass

    subclass_bindings = BindingsSubclass(**bindings.as_dict())
    store = _FakeStore(subclass_bindings)
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="bindings_are_required",
    ):
        MexcPilotLocalCoordinatorReadinessV1(
            store=store,
            bindings=subclass_bindings,
        )


def test_review_binding_is_fail_closed_and_runtime_checked() -> None:
    binding = _binding()
    runner = _FailureRunner()
    binding.validate_runner(runner)
    runner.network_capable = True
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="runtime_binding_mismatch",
    ):
        binding.validate_runner(runner)


def test_history_evidence_distinguishes_graph_logical_and_physical_bytes() -> None:
    evidence = PilotFreshHistoryEvidenceV1(
        pilot_manifest_hash=H,
        stage="shard_acquisition",
        ordinal=0,
        request_id="1" * 64,
        relative_artifact_root="shards/0000.qa_min1.abc",
        history_manifest_hash="2" * 64,
        strict_manifest_pages=1,
        strict_manifest_rows=10,
        strict_manifest_attempts=2,
        strict_manifest_raw_body_bytes=80,
        strict_manifest_graph_logical_storage_bytes=300,
        strict_manifest_collection_runtime_us=50,
        physical_inventory_hash="3" * 64,
        physical_inventory_entries=7,
        physical_inventory_bytes=220,
        parent_output_inventory_hash="4" * 64,
        fresh_reload_completed_at_us=1_000,
        fresh_reload_completed_monotonic_us=500,
        clock_domain_id="fake_clock_domain",
    )
    assert evidence.physical_inventory_bytes != (
        evidence.strict_manifest_graph_logical_storage_bytes
    )
    assert evidence.terminal_accounting_ready is False
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="strict_manifest_accounting_is_impossible",
    ):
        replace(
            evidence,
            strict_manifest_pages=10,
            strict_manifest_rows=1,
            strict_manifest_attempts=1,
            strict_manifest_raw_body_bytes=1_000,
            strict_manifest_graph_logical_storage_bytes=1,
        )


def test_assessment_can_never_claim_terminal_or_retry() -> None:
    reviewed_runner = _binding()
    snapshot = PilotFreshOutputSnapshotV1(
        manifest_hash=H,
        inventory_hash="1" * 64,
        inventory_entries=1,
        inventory_bytes=10,
        scanned_at_us=100,
        scanned_monotonic_us=10,
        clock_domain_id="fake_clock_domain",
    )
    assessment = PilotCoordinatorReadinessAssessmentV1(
        manifest_hash=H,
        network_intent_hash=INTENT_HASH,
        coordinator_bindings_hash="6" * 64,
        runner_binding_hash=reviewed_runner.binding_hash,
        runner_contract_version=reviewed_runner.runner_contract_version,
        runner_contract_hash=reviewed_runner.runner_contract_hash,
        runner_review_evidence_hash=reviewed_runner.review_evidence_hash,
        runner_fixture_set_hash=reviewed_runner.fixture_set_hash,
        stage="endpoint_verification",
        ordinal=-1,
        draft_contract_version=PILOT_STAGE_FAILURE_DRAFT_VERSION,
        draft_hash="2" * 64,
        output_snapshot=snapshot,
        fresh_history_evidence=None,
        blockers=tuple(
            sorted(
                (
                    *coordinator_module._COMMON_BLOCKERS,
                    *coordinator_module._FAILURE_BLOCKERS,
                )
            )
        ),
        authoritative_recovery_stop_code=(
            "unresolved_network_intent_after_restart"
        ),
    )
    assert assessment.callback_consumed_once is True
    assert assessment.network_retry_permitted is False
    assert assessment.terminal_receipt_constructible is False
    assert assessment.authoritative_terminal_published is False
    assert assessment.next_action == "stop_unresolved_intent_no_retry"
    assert len(assessment.assessment_hash) == 64
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="required_blockers_mismatch",
    ):
        replace(assessment, blockers=("cosmetic_only",))
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="recovery_must_be_unresolved_intent_stop",
    ):
        replace(assessment, authoritative_recovery_stop_code="terminal_success")
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="output_snapshot_manifest_mismatch",
    ):
        replace(
            assessment,
            output_snapshot=replace(snapshot, manifest_hash="b" * 64),
        )
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="runner_binding_hash_mismatch",
    ):
        replace(assessment, runner_binding_hash="0" * 64)


def test_failure_draft_consumes_exactly_once_and_returns_stop_assessment(
    monkeypatch,
) -> None:
    coordinator, store = _coordinator(monkeypatch)
    result = coordinator.run_one_reviewed_fake_stage(
        expected_state=store.state,
        session_capability=object(),
        runner=_FailureRunner(),
        runner_binding=_binding(),
    )
    assert store.callback_calls == 1
    assert store.calls == [
        "preflight",
        "claim",
        "run_once",
        "reconstruct",
        "inventory",
    ]
    assert result.fresh_history_evidence is None
    assert "failure_candidate_publish_reload_anchor_seal_api_missing" in (
        result.blockers
    )
    assert result.authoritative_recovery_stop_code == (
        "unresolved_network_intent_after_restart"
    )
    assert result.network_retry_permitted is False


def test_callback_exception_is_typed_stop_and_is_never_retried(monkeypatch) -> None:
    coordinator, store = _coordinator(monkeypatch)
    runner = _FailureRunner(raises=True)
    with pytest.raises(
        PilotLocalCoordinatorCallbackStopError,
        match="intent_consumed_no_retry",
    ):
        coordinator.run_one_reviewed_fake_stage(
            expected_state=store.state,
            session_capability=object(),
            runner=runner,
            runner_binding=_binding(),
        )
    assert runner.calls == 1
    assert store.callback_calls == 1
    assert store.calls == ["preflight", "claim", "run_once"]


def test_store_rejection_before_adapter_entry_does_not_claim_consumption(
    monkeypatch,
) -> None:
    coordinator, store = _coordinator(monkeypatch)
    store.reject_before_callback = True
    runner = _FailureRunner()
    with pytest.raises(
        PilotLocalCoordinatorPreCallbackError,
        match="before_callback_not_consumed",
    ):
        coordinator.run_one_reviewed_fake_stage(
            expected_state=store.state,
            session_capability=object(),
            runner=runner,
            runner_binding=_binding(),
        )
    assert runner.calls == 0
    assert store.callback_calls == 0
    assert store.calls == ["preflight", "claim", "run_once"]


def test_runner_mutation_during_claim_stops_after_consume_before_execute(
    monkeypatch,
) -> None:
    coordinator, store = _coordinator(monkeypatch)
    runner = _FailureRunner()
    store.before_adapter = lambda: setattr(runner, "network_capable", True)
    with pytest.raises(
        PilotLocalCoordinatorCallbackStopError,
        match="intent_consumed_no_retry",
    ):
        coordinator.run_one_reviewed_fake_stage(
            expected_state=store.state,
            session_capability=object(),
            runner=runner,
            runner_binding=_binding(),
        )
    assert store.callback_calls == 1
    assert runner.calls == 0


def test_duck_review_binding_is_rejected_before_any_publication(monkeypatch) -> None:
    coordinator, store = _coordinator(monkeypatch)
    real = _binding()
    duck = SimpleNamespace(
        **real.as_dict(),
        binding_hash=real.binding_hash,
        validate_runner=lambda runner: None,
    )
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="requires_exact_reviewed_runner_binding",
    ):
        coordinator.run_one_reviewed_fake_stage(
            expected_state=store.state,
            session_capability=object(),
            runner=_FailureRunner(),
            runner_binding=duck,
        )
    assert store.calls == []


def test_wrong_stage_is_rejected_before_any_publication(monkeypatch) -> None:
    wrong = PilotReviewedFakeRunnerBindingV1(
        stage="shard_acquisition",
        runner_contract_version="fake_endpoint_runner_v1",
        runner_contract_hash=RUNNER_HASH,
        review_evidence_hash=REVIEW_HASH,
        fixture_set_hash=FIXTURE_HASH,
    )
    bindings = PilotCoordinatorBindingsV1.frozen(
        reviewed_fake_runner_binding_hashes=(wrong.binding_hash,),
    )
    store = _FakeStore(bindings)
    monkeypatch.setattr(coordinator_module, "MexcPilotLocalStoreV1", _FakeStore)
    coordinator = MexcPilotLocalCoordinatorReadinessV1(
        store=store,
        bindings=bindings,
    )
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="expected_stage_binding",
    ):
        coordinator.run_one_reviewed_fake_stage(
            expected_state=store.state,
            session_capability=object(),
            runner=_FailureRunner(),
            runner_binding=wrong,
        )
    assert store.calls == []
    assert store.callback_calls == 0


def test_final_history_clock_sample_must_follow_scan_in_bound_domain(
    monkeypatch,
) -> None:
    coordinator, store = _coordinator(monkeypatch)
    scan = store.scan_inventory()
    coordinator._require_ordered_bound_sample(
        PilotClockSampleV1(
            scan.scanned_at_us,
            scan.scanned_monotonic_us,
            scan.clock_domain_id,
        ),
        scan,
    )
    with pytest.raises(
        PilotLocalCoordinatorEvidenceError,
        match="binding_or_order_invalid_no_retry",
    ):
        coordinator._require_ordered_bound_sample(
            PilotClockSampleV1(
                scan.scanned_at_us - 1,
                scan.scanned_monotonic_us,
                scan.clock_domain_id,
            ),
            scan,
        )
    with pytest.raises(
        PilotLocalCoordinatorEvidenceError,
        match="binding_or_order_invalid_no_retry",
    ):
        coordinator._require_ordered_bound_sample(
            PilotClockSampleV1(
                scan.scanned_at_us,
                scan.scanned_monotonic_us,
                "other_clock_domain",
            ),
            scan,
        )


def test_fresh_endpoint_evidence_brackets_inventory_and_rejects_residue(
    monkeypatch,
) -> None:
    coordinator, store = _coordinator(monkeypatch)
    request_id = "1" * 64
    history_hash = "2" * 64
    root = "verification/fake"
    official_path = "official/reference.json"
    official_hash = "3" * 64
    request = SimpleNamespace(
        request_id=request_id,
        storage_profile="fake_storage_profile",
    )
    store.manifest = SimpleNamespace(
        manifest_hash=H,
        endpoint_verification=SimpleNamespace(
            probe_request=request,
            relative_artifact_root=root,
        ),
        shards=(),
    )
    store.intent.relative_artifact_root = root
    entries = [
        PilotInventoryEntryV1(official_path, official_hash, 10),
        PilotInventoryEntryV1(f"{root}/attempts/a.json", "4" * 64, 10),
        PilotInventoryEntryV1(
            f"{root}/collections/{request_id}/admission.json",
            "5" * 64,
            10,
        ),
        PilotInventoryEntryV1(
            f"{root}/collections/{request_id}/manifest.json",
            "6" * 64,
            10,
        ),
        PilotInventoryEntryV1(
            f"{root}/normalized/{request_id}/a.jsonl",
            "7" * 64,
            10,
        ),
        PilotInventoryEntryV1(f"{root}/raw/sha256/aa/a.bin", "8" * 64, 10),
        PilotInventoryEntryV1(f"{root}/scope.json", "9" * 64, 10),
    ]
    store.inventory_scan_override = PilotInventoryScanV1(
        manifest_hash=H,
        entries=tuple(sorted(entries, key=lambda item: item.relative_path)),
        total_bytes=70,
        scanned_at_us=1_000_000,
        scanned_monotonic_us=500_000,
        clock_domain_id="fake_clock_domain",
    )
    draft = EndpointStageDraftV1(
        manifest_hash=H,
        authorization_receipt_hash="4" * 64,
        network_intent_hash=INTENT_HASH,
        clock_domain_id="fake_clock_domain",
        stage_started_at_us=100,
        stage_completed_at_us=150,
        stage_started_monotonic_us=10,
        stage_completed_monotonic_us=60,
        official_document_evidence_relative_path=official_path,
        official_document_evidence_hash=official_hash,
        official_document_request_started_at_us=110,
        official_document_fetched_at_us=120,
        official_document_request_started_monotonic_us=20,
        official_document_fetched_monotonic_us=30,
        live_probe_store_relative_root=root,
        live_history_manifest_hash=history_hash,
        live_probe_started_at_us=130,
        live_probe_completed_at_us=140,
        live_probe_started_monotonic_us=40,
        live_probe_completed_monotonic_us=50,
    )
    history = SimpleNamespace(
        manifest_hash=history_hash,
        page_receipts=(object(),),
        actual_row_count=1,
        actual_attempt_count=1,
        actual_total_raw_body_bytes=10,
        logical_storage_bytes=50,
        collection_runtime_us=10,
    )
    residue = [False]

    class FakeStrictStore:
        def __init__(self, artifact_root, *, writable, storage_profile):
            assert artifact_root == store.output_root / "verification" / "fake"
            assert writable is False
            assert storage_profile == request.storage_profile
            store.events.append("strict_init")

        def reconcile_restart(self, requests, *, expected_manifest_hashes):
            assert requests == [request]
            assert expected_manifest_hashes == {request_id: history_hash}
            store.events.append("reconcile")
            return HistoryRestartReportV1(
                request_states=(
                    HistoryRestartRequestStateV1(
                        request_id=request_id,
                        state="complete_verified",
                        manifest_hash=history_hash,
                    ),
                ),
                temp_paths=(".residue.tmp",) if residue[0] else (),
                unreferenced_attempt_paths=(),
                unreferenced_raw_paths=(),
                alternate_normalized_paths=(),
                ready=True,
            )

        def load_complete_from_disk(self, expected_request, *, expected_manifest_hash):
            assert expected_request is request
            assert expected_manifest_hash == history_hash
            store.events.append("load")
            return SimpleNamespace(manifest=history)

    monkeypatch.setattr(
        coordinator_module,
        "StrictHistoryArtifactStoreV2",
        FakeStrictStore,
    )
    evidence, scan = coordinator._fresh_history_evidence(
        draft=draft,
        intent=store.intent,
    )
    assert store.events == [
        "strict_init",
        "reconcile",
        "scan",
        "strict_init",
        "reconcile",
        "load",
    ]
    assert scan is store.inventory_scan_override
    assert evidence.history_manifest_hash == history_hash
    assert evidence.parent_output_inventory_hash == scan.inventory_hash
    assert evidence.physical_inventory_entries == 6
    assert evidence.physical_inventory_bytes == 60
    assert evidence.terminal_accounting_ready is False
    reviewed_runner = _binding()
    assessment = PilotCoordinatorReadinessAssessmentV1(
        manifest_hash=H,
        network_intent_hash=INTENT_HASH,
        coordinator_bindings_hash=coordinator.bindings.binding_hash,
        runner_binding_hash=reviewed_runner.binding_hash,
        runner_contract_version=reviewed_runner.runner_contract_version,
        runner_contract_hash=reviewed_runner.runner_contract_hash,
        runner_review_evidence_hash=reviewed_runner.review_evidence_hash,
        runner_fixture_set_hash=reviewed_runner.fixture_set_hash,
        stage="endpoint_verification",
        ordinal=-1,
        draft_contract_version=draft.contract_version,
        draft_hash="a" * 64,
        output_snapshot=PilotFreshOutputSnapshotV1.from_scan(scan),
        fresh_history_evidence=evidence,
        blockers=tuple(
            sorted(
                (
                    *coordinator_module._COMMON_BLOCKERS,
                    *coordinator_module._ENDPOINT_BLOCKERS,
                )
            )
        ),
        authoritative_recovery_stop_code=(
            "unresolved_network_intent_after_restart"
        ),
    )
    with pytest.raises(
        PilotLocalCoordinatorContractError,
        match="nested_evidence_binding_mismatch",
    ):
        replace(
            assessment,
            fresh_history_evidence=replace(
                evidence,
                physical_inventory_entries=len(scan.entries) + 1,
            ),
        )

    residue[0] = True
    store.events.clear()
    with pytest.raises(
        PilotLocalCoordinatorEvidenceError,
        match="reconciliation_not_clean_no_retry",
    ):
        coordinator._fresh_history_evidence(draft=draft, intent=store.intent)
    assert store.events == ["strict_init", "reconcile"]


def test_module_has_no_private_store_calls_or_terminal_receipt_constructors() -> None:
    source = Path(coordinator_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "._publish_json",
        "._reload_json",
        "._anchor_detached",
        "._require_expected_state",
        "EndpointVerificationReceiptV1(",
        "PilotShardResultV1(",
        "PilotStepFailureReceiptV1(",
        "PilotRunAnchorReceiptV1(",
        "dataclasses.replace",
    ):
        assert forbidden not in source
    assert "hard_stop_unimplemented" in source
    assert "writer_lock_inventory_layout_unresolved" in source
    assert "logical_reference_inventory_contract_missing" in source
