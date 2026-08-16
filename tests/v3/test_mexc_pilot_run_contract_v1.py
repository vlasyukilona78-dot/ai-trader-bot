from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from trading.market_data.mexc_futures_transport import (
    HistoryResourceLimitsV1,
    candidate_endpoint_fixture_path,
    candidate_history_retry_policy_v1,
    load_mexc_futures_endpoint_contract_v1,
    mexc_futures_transport_contract_hash,
)
from trading.market_data.min1_aggregation import min1_aggregation_contract_hash
from trading.market_data.mexc_pilot_run import (
    EndpointVerificationPlanV1,
    EndpointVerificationReceiptV1,
    MexcPublicQaPilotRunManifestV1,
    PilotDiskPreflightReceiptV1,
    PilotGlobalBudgetsV1,
    PilotIntentDurabilityReceiptV1,
    PilotNetworkIntentV1,
    PilotRunAnchorReceiptV1,
    PilotRunArtifactError,
    PilotRunAuthorizationError,
    PilotRunBudgetExceededError,
    PilotRunContractError,
    PilotRunPreflightError,
    PilotRunStateV1,
    PilotRunTransitionError,
    PilotShardPlanV1,
    PilotShardResultV1,
    PilotStepFailureReceiptV1,
    U5PublicPilotAuthorizationReceiptV1,
    load_pilot_run_manifest_v1,
    parse_pilot_run_manifest_v1,
    parse_pilot_run_state_v1,
    pilot_run_contract_hash,
    publish_pilot_run_manifest_v1,
)
from trading.market_data.strict_history_v2 import (
    HistoryRangeRequestV2,
    storage_profile_hash,
    strict_history_v2_contract_hash,
)


BASE = 1_767_225_600
WINDOWS_PROFILE = "windows_ntfs_hardlink_best_effort_v1"
SYMBOLS = (
    ("BTCUSDT", "BTC_USDT"),
    ("AAAUSDT", "AAA_USDT"),
    ("BBBUSDT", "BBB_USDT"),
    ("CCCUSDT", "CCC_USDT"),
    ("DDDUSDT", "DDD_USDT"),
    ("EEEUSDT", "EEE_USDT"),
    ("FFFUSDT", "FFF_USDT"),
    ("GGGUSDT", "GGG_USDT"),
    ("HHHUSDT", "HHH_USDT"),
)


def _endpoint():
    return load_mexc_futures_endpoint_contract_v1(candidate_endpoint_fixture_path())


def _limits(*, rows: int, page_size: int = 2_000) -> HistoryResourceLimitsV1:
    pages = (rows + page_size - 1) // page_size
    if rows == 140 * 1_440:
        raw = 128 * 1024**2
        storage = 256 * 1024**2
        runtime = 60 * 60 * 1_000_000
    elif rows >= 7 * 1_440:
        raw = 8 * 1024**2
        storage = 16 * 1024**2
        runtime = 10 * 60 * 1_000_000
    else:
        raw = 2 * 1024**2
        storage = 4 * 1024**2
        runtime = 5 * 60 * 1_000_000
    return HistoryResourceLimitsV1(
        max_pages=pages,
        max_rows=rows,
        max_attempts_per_page=1,
        max_total_attempts=pages,
        max_raw_body_bytes_per_attempt=1024**2,
        max_total_raw_body_bytes=raw,
        max_logical_storage_bytes=storage,
        max_collection_runtime_us=runtime,
        max_attempt_runtime_us=30 * 1_000_000,
    )


def _request(
    symbol: str,
    venue_symbol: str,
    *,
    interval: str,
    days: int | None = None,
    rows: int | None = None,
) -> HistoryRangeRequestV2:
    seconds = 60 if interval == "Min1" else 3_600
    if rows is None:
        assert days is not None
        rows = days * 86_400 // seconds
    start = BASE - rows * seconds
    return HistoryRangeRequestV2(
        venue="mexc_contract",
        symbol=symbol,
        venue_symbol=venue_symbol,
        interval=interval,
        start_open_ts=start,
        end_open_ts_exclusive=BASE,
        collection_as_of_us=BASE * 1_000_000,
        endpoint_contract=_endpoint(),
        resource_limits=_limits(rows=rows),
        retry_policy=candidate_history_retry_policy_v1(),
        page_size=2_000,
        storage_profile=WINDOWS_PROFILE,
    )


def _probe_request() -> HistoryRangeRequestV2:
    limits = HistoryResourceLimitsV1(
        max_pages=1,
        max_rows=1,
        max_attempts_per_page=1,
        max_total_attempts=1,
        max_raw_body_bytes_per_attempt=1024**2,
        max_total_raw_body_bytes=1024**2,
        max_logical_storage_bytes=4 * 1024**2,
        max_collection_runtime_us=60 * 1_000_000,
        max_attempt_runtime_us=30 * 1_000_000,
    )
    return HistoryRangeRequestV2(
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        interval="Min1",
        start_open_ts=BASE - 60,
        end_open_ts_exclusive=BASE,
        collection_as_of_us=BASE * 1_000_000,
        endpoint_contract=_endpoint(),
        resource_limits=limits,
        retry_policy=candidate_history_retry_policy_v1(),
        page_size=1,
        storage_profile=WINDOWS_PROFILE,
    )


def _shards() -> tuple[PilotShardPlanV1, ...]:
    plans: list[PilotShardPlanV1] = []
    qa: dict[str, PilotShardPlanV1] = {}
    for symbol, venue_symbol in SYMBOLS:
        request = _request(symbol, venue_symbol, interval="Min1", days=7)
        item = PilotShardPlanV1(
            ordinal=len(plans),
            role="qa_min1",
            request=request,
            relative_artifact_root=(
                f"shards/{len(plans):04d}.qa_min1.{request.request_id}"
            ),
        )
        plans.append(item)
        qa[symbol] = item
    for symbol, venue_symbol in SYMBOLS:
        request = _request(symbol, venue_symbol, interval="Min60", days=7)
        plans.append(
            PilotShardPlanV1(
                ordinal=len(plans),
                role="native_min60_control",
                request=request,
                relative_artifact_root=(
                    f"shards/{len(plans):04d}.native_min60_control."
                    f"{request.request_id}"
                ),
                source_min1_request_id=qa[symbol].request.request_id,
            )
        )
    deep = _request("BTCUSDT", "BTC_USDT", interval="Min1", days=140)
    plans.append(
        PilotShardPlanV1(
            ordinal=len(plans),
            role="deep_min1",
            request=deep,
            relative_artifact_root=(
                f"shards/{len(plans):04d}.deep_min1.{deep.request_id}"
            ),
        )
    )
    return tuple(plans)


def _budgets() -> PilotGlobalBudgetsV1:
    mib = 1024**2
    return PilotGlobalBudgetsV1(
        max_symbols=9,
        max_shards=20,
        max_total_pages=200,
        max_total_rows=400_000,
        max_verification_attempts=2,
        max_acquisition_attempts=200,
        max_network_attempts=202,
        max_total_raw_body_bytes=256 * mib,
        max_total_logical_storage_bytes=512 * mib,
        max_run_control_bytes=64 * mib,
        max_total_output_bytes=700 * mib,
        max_sum_shard_runtime_us=4 * 60 * 60 * 1_000_000,
        max_run_elapsed_us=5 * 60 * 60 * 1_000_000,
        max_observed_sleep_us=10 * 60 * 1_000_000,
        min_inter_step_spacing_us=500_000,
        max_active_shards=1,
        max_in_flight_http_attempts=1,
        min_free_disk_bytes_before_run=800 * mib,
        required_free_disk_bytes_after_reservation=100 * mib,
        max_inventory_entries=1_000,
        max_preflight_age_us=60 * 1_000_000,
    )


def _manifest(*, budgets: PilotGlobalBudgetsV1 | None = None):
    probe = _probe_request()
    verification = EndpointVerificationPlanV1(
        probe_request=probe,
        relative_artifact_root=f"verification/{probe.request_id}",
        official_reference_url=probe.endpoint_contract.plan_reference_url,
        verifier_contract_version="fixture_endpoint_verifier_v1",
        verifier_contract_hash="a" * 64,
        max_network_attempts=2,
        max_total_raw_body_bytes=2 * 1024**2,
        max_total_storage_bytes=8 * 1024**2,
        max_runtime_us=2 * 60 * 1_000_000,
        max_total_sleep_us=30 * 1_000_000,
    )
    return MexcPublicQaPilotRunManifestV1(
        repository_commit="1" * 40,
        repository_tree_receipt_hash="2" * 64,
        created_at_us=1_900_000_000_000_000,
        parent_master_plan_path="docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md",
        parent_master_plan_sha256="3" * 64,
        parent_adr_path="docs/ADR_MEXC_V3_FINAL_BOT_2026-08-15.md",
        parent_adr_sha256="4" * 64,
        output_root_locator="file:///C:/koteika-pilot/test-run",
        shard_executor_contract_version="fixture_shard_executor_v1",
        shard_executor_contract_hash="5" * 64,
        endpoint_verification=verification,
        shards=_shards(),
        budgets=budgets or _budgets(),
    )


def _authorization(manifest, *, start=1_900_000_000_100_000):
    request = manifest.endpoint_verification.probe_request
    return U5PublicPilotAuthorizationReceiptV1(
        manifest_hash=manifest.manifest_hash,
        manifest_identity=manifest.manifest_identity,
        authority_id="fixture_user_authority",
        orchestrator_session_id="fixture_orchestrator_session",
        authorized_at_us=start,
        expires_at_us=start + 24 * 60 * 60 * 1_000_000,
        allowed_domains=("www.mexc.com", "api.mexc.com"),
        allowed_operations=(
            "current_official_reference_verification",
            "exact_live_kline_probe",
            "conditional_manifest_history_acquisition",
        ),
        max_network_attempts=manifest.budgets.max_network_attempts,
        max_total_raw_body_bytes=manifest.budgets.max_total_raw_body_bytes,
        max_total_output_bytes=manifest.budgets.max_total_output_bytes,
        max_run_elapsed_us=manifest.budgets.max_run_elapsed_us,
        storage_profile=request.storage_profile,
        storage_profile_hash=request.storage_profile_hash,
        windows_sudden_power_loss_boundary_accepted=True,
        restart_network_policy="forbid_network_after_process_restart",
        external_authority_evidence_hash="6" * 64,
    )


def _preflight(manifest, authorization, step, now):
    reserved = manifest.remaining_storage_reservation(step)
    free_before = max(
        manifest.budgets.min_free_disk_bytes_before_run,
        reserved + manifest.budgets.required_free_disk_bytes_after_reservation,
    )
    request = manifest.endpoint_verification.probe_request
    return PilotDiskPreflightReceiptV1(
        manifest_hash=manifest.manifest_hash,
        authorization_receipt_hash=authorization.receipt_hash,
        step_ordinal=step,
        checked_at_us=now,
        valid_until_us=now + 30 * 1_000_000,
        output_root_locator=manifest.output_root_locator,
        volume_identity="fixture_volume_c",
        storage_profile=request.storage_profile,
        storage_profile_hash=request.storage_profile_hash,
        free_bytes_before=free_before,
        reserved_bytes=reserved,
        free_bytes_after_reservation=free_before - reserved,
        fresh_relative_roots=manifest.remaining_fresh_roots(step),
        path_chain_reparse_free=True,
        local_fixed_volume=True,
        same_volume_publication=True,
        hardlink_create_new_supported=True,
    )


def _network_intent(
    manifest,
    authorization,
    preflight,
    *,
    stage,
    ordinal,
    issued_at,
    mono,
    orchestrator_session_id=None,
    publisher_instance_id="fixture_publisher_instance",
):
    if stage == "endpoint_verification":
        plan = manifest.endpoint_verification
        binding = plan.plan_hash
        root = plan.relative_artifact_root
        attempts = plan.max_network_attempts
        raw = plan.max_total_raw_body_bytes
        storage = plan.max_total_storage_bytes
        runtime = plan.max_runtime_us
    else:
        plan = manifest.shards[ordinal]
        request = plan.request
        binding = plan.plan_id
        root = plan.relative_artifact_root
        attempts = request.required_pages * request.resource_limits.max_attempts_per_page
        raw = min(
            request.resource_limits.max_total_raw_body_bytes,
            attempts * request.resource_limits.max_raw_body_bytes_per_attempt,
        )
        storage = request.resource_limits.max_logical_storage_bytes
        runtime = request.resource_limits.max_collection_runtime_us
    base = {
        "manifest_hash": manifest.manifest_hash,
        "authorization_receipt_hash": authorization.receipt_hash,
        "preflight_receipt_hash": preflight.receipt_hash,
        "stage": stage,
        "ordinal": ordinal,
        "step_binding_hash": binding,
        "relative_artifact_root": root,
        "clock_domain_id": "fixture_clock_domain",
        "orchestrator_session_id": (
            authorization.orchestrator_session_id
            if orchestrator_session_id is None
            else orchestrator_session_id
        ),
        "publisher_instance_id": publisher_instance_id,
        "issued_at_us": issued_at,
        "issued_monotonic_us": mono,
        "reserved_network_attempts": attempts,
        "reserved_raw_body_bytes": raw,
        "reserved_storage_bytes": storage,
        "reserved_runtime_us": runtime,
    }
    candidate_hash = PilotNetworkIntentV1.candidate_hash_for(**base)
    slot_id = PilotNetworkIntentV1.slot_id_for(
        manifest_hash=manifest.manifest_hash,
        stage=stage,
        ordinal=ordinal,
    )
    durability = PilotIntentDurabilityReceiptV1(
        intent_candidate_hash=candidate_hash,
        intent_slot_id=slot_id,
        intent_candidate_locator=PilotNetworkIntentV1.slot_locator_for(
            manifest_hash=manifest.manifest_hash,
            stage=stage,
            ordinal=ordinal,
        ),
        intent_candidate_artifact_sha256=(
            PilotNetworkIntentV1.candidate_artifact_sha256_for(**base)
        ),
        publisher_instance_id=publisher_instance_id,
        durable_publication_receipt_hash="1" * 64,
        fresh_reload_receipt_hash="2" * 64,
        detached_reservation_anchor_hash="3" * 64,
        published_at_us=issued_at + 1,
        reloaded_at_us=issued_at + 2,
        anchored_at_us=issued_at + 3,
        published_monotonic_us=mono + 1,
        reloaded_monotonic_us=mono + 2,
        anchored_monotonic_us=mono + 3,
    )
    return PilotNetworkIntentV1(**base, durability_receipt=durability)


def _verification(manifest, authorization, intent, *, start, mono=10_000_000):
    return EndpointVerificationReceiptV1(
        manifest_hash=manifest.manifest_hash,
        authorization_receipt_hash=authorization.receipt_hash,
        verification_plan_hash=manifest.endpoint_verification.plan_hash,
        network_intent_hash=intent.intent_hash,
        clock_domain_id="fixture_clock_domain",
        started_at_us=start,
        completed_at_us=start + 1_000,
        started_monotonic_us=mono,
        completed_monotonic_us=mono + 1_000,
        actual_network_attempts=2,
        actual_raw_body_bytes=1_000,
        actual_storage_bytes=2_000,
        actual_runtime_us=1_000,
        observed_sleep_us=0,
        official_document_evidence_hash="7" * 64,
        official_document_request_started_at_us=start + 100,
        official_document_fetched_at_us=start + 200,
        official_document_request_started_monotonic_us=mono + 100,
        official_document_fetched_monotonic_us=mono + 200,
        live_history_manifest_hash="8" * 64,
        live_attempt_receipt_hash="9" * 64,
        live_raw_body_sha256="a" * 64,
        live_observed_rows=1,
        live_probe_started_at_us=start + 300,
        live_probe_completed_at_us=start + 600,
        live_probe_started_monotonic_us=mono + 300,
        live_probe_completed_monotonic_us=mono + 600,
        fresh_disk_reload_completed_at_us=start + 700,
        fresh_disk_reload_completed_monotonic_us=mono + 700,
        output_inventory_hash="c" * 64,
        output_inventory_entries=5,
        detached_anchor_receipt_hash="b" * 64,
        detached_anchor_at_us=start + 800,
        detached_anchor_monotonic_us=mono + 800,
    )


def _shard_result(manifest, ordinal, *, start, mono, intent=None):
    plan = manifest.shards[ordinal]
    request = plan.request
    attempts = request.required_pages
    return PilotShardResultV1(
        manifest_hash=manifest.manifest_hash,
        shard_plan_id=plan.plan_id,
        network_intent_hash=(intent.intent_hash if intent else "d" * 64),
        ordinal=ordinal,
        request_id=request.request_id,
        relative_artifact_root=plan.relative_artifact_root,
        clock_domain_id="fixture_clock_domain",
        step_started_at_us=start,
        step_completed_at_us=start + 1_000,
        step_started_monotonic_us=mono,
        step_completed_monotonic_us=mono + 1_000,
        observed_inter_step_delay_us=manifest.budgets.min_inter_step_spacing_us,
        observed_internal_sleep_us=0,
        history_manifest_hash=f"{ordinal + 10:064x}",
        actual_pages=request.required_pages,
        actual_rows=request.expected_row_count,
        actual_attempts=attempts,
        actual_raw_body_bytes=attempts * 100,
        actual_logical_storage_bytes=attempts * 500,
        actual_collection_runtime_us=1_000,
        output_inventory_hash=f"{ordinal + 100:064x}",
        output_inventory_entries=2 * attempts + 4,
        output_inventory_bytes=attempts * 500,
        detached_shard_anchor_receipt_hash=f"{ordinal + 200:064x}",
        fresh_disk_reload_completed_at_us=start + 800,
        detached_shard_anchor_at_us=start + 900,
        fresh_disk_reload_completed_monotonic_us=mono + 800,
        detached_shard_anchor_monotonic_us=mono + 900,
    )


def test_contract_and_frozen_dependency_hashes_are_pinned() -> None:
    assert pilot_run_contract_hash() == (
        "f3d642d436e9d4a44e65f35c6ea8375bd92b4b36b30f1c86af54936a608ce65e"
    )
    assert mexc_futures_transport_contract_hash() == (
        "7d3bd40c6753e7bda2f1904ce2ffa2ff55770ecce9ba6d5614d2b30ae0664d22"
    )
    assert strict_history_v2_contract_hash() == (
        "cce9922317ec5f0008f3b293103f9f5a17504b7143f81af1845d9d4765c44086"
    )
    assert min1_aggregation_contract_hash() == (
        "0d851b253cde913d95a693e0db7296b59ff78a6048bacb20838386f2e8e20a21"
    )


def test_manifest_is_canonical_round_trip_and_does_not_grant_u5() -> None:
    manifest = _manifest()
    payload = manifest.as_dict()
    rebuilt = parse_pilot_run_manifest_v1(payload)
    assert rebuilt == manifest
    assert rebuilt.manifest_hash == manifest.manifest_hash
    assert payload["u5_granted_by_manifest"] is False
    assert payload["purpose"] == "p2_public_qa_data_mechanics_only"
    assert payload["execution_policy"]["full_universe_or_p3_claim"] is False
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert parse_pilot_run_manifest_v1(raw) == manifest
    assert _endpoint().verification_status == "candidate_not_u5_verified"


def test_manifest_parser_rejects_duplicate_nonfinite_and_extra_keys() -> None:
    manifest = _manifest()
    raw = json.dumps(manifest.as_dict(), sort_keys=True, separators=(",", ":"))
    duplicate = (raw[:-1] + ',"purpose":"p2_public_qa_data_mechanics_only"}\n').encode()
    with pytest.raises(PilotRunContractError, match="duplicate_key"):
        parse_pilot_run_manifest_v1(duplicate)
    with pytest.raises(PilotRunContractError, match="nonfinite"):
        parse_pilot_run_manifest_v1(b'{"value":NaN}\n')
    extra = manifest.as_dict()
    extra["extra"] = True
    with pytest.raises(PilotRunContractError, match="schema_mismatch"):
        parse_pilot_run_manifest_v1(extra)


def test_manifest_composition_and_root_aliases_fail_closed() -> None:
    manifest = _manifest()
    reduced: list[PilotShardPlanV1] = []
    for item in manifest.shards:
        if item.request.symbol == "HHHUSDT":
            continue
        ordinal = len(reduced)
        reduced.append(
            replace(
                item,
                ordinal=ordinal,
                relative_artifact_root=(
                    f"shards/{ordinal:04d}.{item.role}.{item.request.request_id}"
                ),
            )
        )
    with pytest.raises(PilotRunContractError, match="8_to_10"):
        replace(manifest, shards=tuple(reduced))
    bad = replace(manifest.shards[1], relative_artifact_root=manifest.shards[0].relative_artifact_root)
    shards = list(manifest.shards)
    shards[1] = bad
    with pytest.raises(PilotRunContractError, match="roots_are_not_unique"):
        replace(manifest, shards=tuple(shards))
    with pytest.raises(PilotRunContractError, match="root_is_invalid"):
        replace(manifest.shards[0], relative_artifact_root="../escape")


def test_global_reservation_is_checked_before_any_execution_state() -> None:
    manifest = _manifest()
    reservations = manifest.planned_reservations
    too_small = replace(
        manifest.budgets, max_total_pages=reservations["total_pages"] - 1
    )
    with pytest.raises(PilotRunBudgetExceededError, match="pages"):
        replace(manifest, budgets=too_small)
    with pytest.raises(PilotRunContractError, match="serial"):
        replace(manifest.budgets, max_active_shards=2)
    with pytest.raises(PilotRunBudgetExceededError, match="run_control_bytes"):
        _manifest(budgets=replace(_budgets(), max_run_control_bytes=100_000))
    assert reservations["run_control_bytes"] == 49 * 1024**2
    assert reservations["inventory_entries"] >= 4 * len(manifest.shards) + 7


def test_manifest_immutable_publication_round_trip_and_conflict(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "pilot-manifest.json"
    assert publish_pilot_run_manifest_v1(path, manifest) == path
    assert load_pilot_run_manifest_v1(path) == manifest
    assert publish_pilot_run_manifest_v1(path, manifest) == path
    changed = replace(manifest, repository_tree_receipt_hash="f" * 64)
    with pytest.raises(PilotRunArtifactError, match="conflict"):
        publish_pilot_run_manifest_v1(path, changed)


def test_authorization_and_preflight_are_detached_and_exact() -> None:
    manifest = _manifest()
    auth = _authorization(manifest)
    state = PilotRunStateV1(manifest)
    assert state.next_action == "await_detached_u5_authorization"
    now = auth.authorized_at_us + 1
    state = state.with_authorization(auth, now_us=now)
    assert state.next_action == "run_local_preflight:-1"
    wrong = replace(auth, manifest_hash="f" * 64, manifest_identity=f"mexc_public_qa_pilot_run_v1.{('f' * 64)}")
    with pytest.raises(PilotRunAuthorizationError, match="another_manifest"):
        PilotRunStateV1(manifest).with_authorization(wrong, now_us=now)
    with pytest.raises(PilotRunAuthorizationError, match="not_current"):
        PilotRunStateV1(manifest).with_authorization(
            auth, now_us=auth.expires_at_us
        )
    preflight = _preflight(manifest, auth, -1, now + 1)
    state = state.with_preflight(preflight, now_us=now + 1)
    assert state.next_action == "run_endpoint_verification_stage"
    low = replace(
        preflight,
        free_bytes_before=preflight.reserved_bytes,
        free_bytes_after_reservation=0,
    )
    with pytest.raises(PilotRunPreflightError):
        PilotRunStateV1(manifest).with_authorization(auth, now_us=now).with_preflight(
            low, now_us=now + 1
        )


def test_authorization_must_cover_worst_case_and_budgets_reject_bool() -> None:
    manifest = _manifest()
    auth = _authorization(manifest)
    insufficient = replace(
        auth,
        max_network_attempts=manifest.planned_reservations["network_attempts"] - 1,
    )
    with pytest.raises(PilotRunAuthorizationError, match="cannot_cover_planned_run"):
        PilotRunStateV1(manifest).with_authorization(
            insufficient,
            now_us=insufficient.authorized_at_us + 1,
        )
    with pytest.raises(PilotRunContractError, match="max_total_pages_is_invalid"):
        replace(manifest.budgets, max_total_pages=True)
    with pytest.raises(PilotRunAuthorizationError, match="cannot_cover_planned_run"):
        PilotRunStateV1(manifest, authorization=insufficient)
    backdated = replace(
        auth,
        authorized_at_us=manifest.created_at_us - 1,
        expires_at_us=auth.expires_at_us,
    )
    with pytest.raises(PilotRunAuthorizationError, match="precedes_manifest"):
        PilotRunStateV1(manifest, authorization=backdated)


def test_preflight_and_intent_cannot_be_backdated_before_authorization() -> None:
    manifest = _manifest()
    auth = _authorization(manifest)
    state = PilotRunStateV1(manifest).with_authorization(
        auth,
        now_us=auth.authorized_at_us,
    )
    backdated = _preflight(
        manifest,
        auth,
        -1,
        auth.authorized_at_us - 1,
    )
    with pytest.raises(PilotRunPreflightError, match="precedes_authorization"):
        state.with_preflight(backdated, now_us=auth.authorized_at_us)

    current = _preflight(manifest, auth, -1, auth.authorized_at_us)
    state = state.with_preflight(current, now_us=auth.authorized_at_us)
    intent = _network_intent(
        manifest,
        auth,
        current,
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=current.checked_at_us - 1,
        mono=9_000_000,
    )
    with pytest.raises(PilotRunTransitionError, match="precedes_preflight"):
        state.with_network_intent(intent)


def test_same_stage_candidates_contend_on_one_create_new_slot() -> None:
    manifest = _manifest()
    auth = _authorization(manifest)
    first_preflight = _preflight(
        manifest,
        auth,
        -1,
        auth.authorized_at_us + 1,
    )
    second_preflight = _preflight(
        manifest,
        auth,
        -1,
        auth.authorized_at_us + 2,
    )
    first = _network_intent(
        manifest,
        auth,
        first_preflight,
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=auth.authorized_at_us + 3,
        mono=9_000_000,
        publisher_instance_id="publisher_one",
    )
    second = _network_intent(
        manifest,
        auth,
        second_preflight,
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=auth.authorized_at_us + 4,
        mono=9_000_001,
        publisher_instance_id="publisher_two",
    )
    assert first.intent_candidate_hash != second.intent_candidate_hash
    assert first.intent_slot_id == second.intent_slot_id
    assert first.intent_candidate_locator == second.intent_candidate_locator
    assert (
        first.durability_receipt.publication_outcome
        == "create_new_winner_for_this_process"
    )
    with pytest.raises(PilotRunContractError, match="create_new_winner"):
        replace(first.durability_receipt, exclusive_create_new_winner=False)
    with pytest.raises(PilotRunContractError, match="publication_outcome"):
        replace(
            first.durability_receipt,
            publication_outcome="reloaded_preexisting_identical",
        )


def test_u5_window_must_cover_reserved_endpoint_and_remaining_shard_spacing() -> None:
    manifest = _manifest()
    auth = _authorization(manifest)
    short_auth = replace(
        auth,
        expires_at_us=(
            auth.authorized_at_us
            + manifest.planned_reservations["run_elapsed_us"]
        ),
    )
    with pytest.raises(PilotRunAuthorizationError, match="planned_run"):
        PilotRunStateV1(manifest).with_authorization(
            short_auth,
            now_us=short_auth.authorized_at_us,
        )

    endpoint_runtime = manifest.endpoint_verification.max_runtime_us
    planned_wall = manifest.planned_reservations["run_elapsed_us"]
    preflight_late_state = PilotRunStateV1(manifest).with_authorization(
        auth,
        now_us=auth.authorized_at_us,
    )
    preflight_late_checked = auth.expires_at_us - planned_wall - 20_000_000
    preflight_late = _preflight(
        manifest,
        auth,
        -1,
        preflight_late_checked,
    )
    preflight_late_state = preflight_late_state.with_preflight(
        preflight_late,
        now_us=preflight_late_checked,
    )
    preflight_late_intent = _network_intent(
        manifest,
        auth,
        preflight_late,
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=preflight_late_checked + 1,
        mono=7_000_000,
    )
    assert (
        preflight_late_intent.durability_receipt.anchored_at_us + planned_wall
        < auth.expires_at_us
    )
    assert preflight_late.valid_until_us + planned_wall >= auth.expires_at_us
    with pytest.raises(PilotRunAuthorizationError, match="preflight_window"):
        preflight_late_state.with_network_intent(preflight_late_intent)

    delayed_state = PilotRunStateV1(manifest).with_authorization(
        auth,
        now_us=auth.authorized_at_us,
    )
    delayed_checked = auth.expires_at_us - endpoint_runtime - 10
    delayed_preflight = _preflight(manifest, auth, -1, delayed_checked)
    delayed_state = delayed_state.with_preflight(
        delayed_preflight,
        now_us=delayed_checked,
    )
    delayed_intent = _network_intent(
        manifest,
        auth,
        delayed_preflight,
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=delayed_checked + 1,
        mono=8_000_000,
    )
    assert (
        delayed_intent.durability_receipt.anchored_at_us
        + delayed_intent.reserved_runtime_us
        < auth.expires_at_us
    )
    with pytest.raises(PilotRunAuthorizationError, match="remaining_run"):
        delayed_state.with_network_intent(delayed_intent)

    endpoint_state = PilotRunStateV1(manifest).with_authorization(
        auth,
        now_us=auth.authorized_at_us,
    )
    endpoint_checked = auth.expires_at_us - endpoint_runtime - 4
    endpoint_preflight = _preflight(
        manifest,
        auth,
        -1,
        endpoint_checked,
    )
    endpoint_state = endpoint_state.with_preflight(
        endpoint_preflight,
        now_us=endpoint_preflight.checked_at_us,
    )
    late_endpoint_intent = _network_intent(
        manifest,
        auth,
        endpoint_preflight,
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=endpoint_checked + 1,
        mono=9_000_000,
    )
    with pytest.raises(PilotRunAuthorizationError, match="reserved_network_step"):
        endpoint_state.with_network_intent(late_endpoint_intent)

    shard_runtime = manifest.shards[0].request.resource_limits.max_collection_runtime_us
    spacing = manifest.budgets.min_inter_step_spacing_us
    now = auth.authorized_at_us + 1
    state = PilotRunStateV1(manifest).with_authorization(
        auth,
        now_us=auth.authorized_at_us,
    )
    probe_preflight = _preflight(manifest, auth, -1, now)
    state = state.with_preflight(probe_preflight, now_us=probe_preflight.checked_at_us)
    endpoint_intent = _network_intent(
        manifest,
        auth,
        probe_preflight,
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=now + 1,
        mono=9_000_000,
    )
    state = state.with_network_intent(endpoint_intent)
    verification = _verification(
        manifest,
        auth,
        endpoint_intent,
        start=now + 10,
    )
    state = state.with_endpoint_verification(verification)
    previous_completed = verification.completed_at_us
    assert manifest.remaining_run_elapsed_reservation(
        0,
        intent_anchor_us=previous_completed + 1,
        previous_completed_at_us=previous_completed,
    ) == (
        spacing - 1
        + sum(
            item.request.resource_limits.max_collection_runtime_us
            for item in manifest.shards
        )
        + (len(manifest.shards) - 1) * spacing
    )
    shard_checked = auth.expires_at_us - shard_runtime - 10
    shard_preflight = _preflight(
        manifest,
        auth,
        0,
        shard_checked,
    )
    state = state.with_preflight(
        shard_preflight,
        now_us=shard_preflight.checked_at_us,
    )
    shard_intent = _network_intent(
        manifest,
        auth,
        shard_preflight,
        stage="shard_acquisition",
        ordinal=0,
        issued_at=shard_preflight.checked_at_us + 1,
        mono=verification.completed_monotonic_us + spacing + 1,
    )
    assert (
        shard_intent.durability_receipt.anchored_at_us
        + shard_intent.reserved_runtime_us
        < auth.expires_at_us
    )
    with pytest.raises(PilotRunAuthorizationError, match="remaining_run"):
        state.with_network_intent(shard_intent)


def test_direct_endpoint_and_root_aliases_are_rejected() -> None:
    endpoint = replace(
        _endpoint(),
        plan_reference_url="https://www.mexc.com/announcements/article/other",
    )
    probe = replace(_probe_request(), endpoint_contract=endpoint)
    with pytest.raises(PilotRunContractError, match="pinned_candidate"):
        EndpointVerificationPlanV1(
            probe_request=probe,
            relative_artifact_root=f"verification/{probe.request_id}",
            official_reference_url=endpoint.plan_reference_url,
            verifier_contract_version="fixture_endpoint_verifier_v1",
            verifier_contract_hash="a" * 64,
            max_network_attempts=2,
            max_total_raw_body_bytes=2 * 1024**2,
            max_total_storage_bytes=8 * 1024**2,
            max_runtime_us=2 * 60 * 1_000_000,
            max_total_sleep_us=30 * 1_000_000,
        )
    for alias in (
        "file:///C:/pilot//run",
        "file:///C:/pilot/./run",
        "file:///C:/pilot/run. ",
        "file:///C:/pilot/run.",
        "file:///C:/pilot/CON",
        "file:///C:/pilot/aux.txt",
        "file:///C:/pilot/COM1",
    ):
        with pytest.raises(PilotRunContractError, match="canonical|invalid"):
            replace(_manifest(), output_root_locator=alias)


def test_sealed_intent_holds_reservation_and_failure_is_terminal() -> None:
    manifest = _manifest()
    auth = _authorization(manifest)
    now = auth.authorized_at_us + 1
    state = PilotRunStateV1(manifest).with_authorization(auth, now_us=now)
    state = state.with_preflight(
        _preflight(manifest, auth, -1, now + 1), now_us=now + 1
    )
    intent = _network_intent(
        manifest,
        auth,
        state.preflight_receipts[-1],
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=now + 3,
        mono=9_000_000,
    )
    state = state.with_network_intent(intent)
    assert state.next_action == "await_endpoint_verification_receipt_no_network_retry"
    assert state.actual_totals["network_attempts"] == 0
    assert state.charged_totals["verification_attempts"] == intent.reserved_network_attempts
    assert parse_pilot_run_state_v1(state.as_dict()) == state

    foreign_session_intent = _network_intent(
        manifest,
        auth,
        state.preflight_receipts[-1],
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=now + 4,
        mono=9_000_001,
        orchestrator_session_id="foreign_orchestrator_session",
    )
    base = PilotRunStateV1(manifest).with_authorization(auth, now_us=now)
    base = base.with_preflight(
        state.preflight_receipts[-1],
        now_us=state.preflight_receipts[-1].checked_at_us,
    )
    with pytest.raises(PilotRunTransitionError, match="session_mismatch"):
        base.with_network_intent(foreign_session_intent)

    started_at = intent.durability_receipt.anchored_at_us + 1
    started_mono = intent.durability_receipt.anchored_monotonic_us + 1
    failure = PilotStepFailureReceiptV1(
        manifest_hash=manifest.manifest_hash,
        authorization_receipt_hash=auth.receipt_hash,
        network_intent_hash=intent.intent_hash,
        stage="endpoint_verification",
        ordinal=-1,
        step_binding_hash=manifest.endpoint_verification.plan_hash,
        clock_domain_id="fixture_clock_domain",
        step_started_at_us=started_at,
        step_completed_at_us=started_at + 100,
        step_started_monotonic_us=started_mono,
        step_completed_monotonic_us=started_mono + 100,
        actual_network_attempts=1,
        actual_raw_body_bytes=0,
        actual_storage_bytes=100,
        actual_runtime_us=100,
        observed_internal_sleep_us=0,
        observed_inter_step_delay_us=0,
        output_inventory_hash="e" * 64,
        output_inventory_entries=2,
        output_inventory_bytes=100,
        error_code="http_status_rejected",
        error_evidence_hash="f" * 64,
        candidate_publication_receipt_hash="a" * 64,
        candidate_reload_receipt_hash="b" * 64,
        candidate_detached_anchor_hash="c" * 64,
        published_at_us=started_at + 101,
        reloaded_at_us=started_at + 102,
        anchored_at_us=started_at + 103,
        published_monotonic_us=started_mono + 101,
        reloaded_monotonic_us=started_mono + 102,
        anchored_monotonic_us=started_mono + 103,
    )
    failed = state.with_step_failure(failure)
    assert failed.next_action == "stopped"
    assert failed.actual_totals["network_attempts"] == 1
    assert failed.charged_totals == failed.actual_totals
    assert parse_pilot_run_state_v1(failed.as_dict()) == failed
    assert failure.failure_candidate_hash != failure.receipt_hash
    assert failure.failure_candidate_artifact_sha256 == hashlib.sha256(
        json.dumps(
            failure.failure_candidate_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    ).hexdigest()
    tampered_failure = failure.as_dict()
    tampered_failure["failure_candidate_hash"] = "0" * 64
    with pytest.raises(PilotRunContractError, match="candidate_binding"):
        PilotStepFailureReceiptV1.from_dict(tampered_failure)
    late_anchor_epoch = (
        intent.durability_receipt.anchored_at_us
        + manifest.budgets.max_run_elapsed_us
        + 1
    )
    late_anchor_monotonic = (
        intent.durability_receipt.anchored_monotonic_us
        + manifest.budgets.max_run_elapsed_us
        + 1
    )
    late_failure = replace(
        failure,
        published_at_us=late_anchor_epoch - 2,
        reloaded_at_us=late_anchor_epoch - 1,
        anchored_at_us=late_anchor_epoch,
        published_monotonic_us=late_anchor_monotonic - 2,
        reloaded_monotonic_us=late_anchor_monotonic - 1,
        anchored_monotonic_us=late_anchor_monotonic,
    )
    with pytest.raises(PilotRunBudgetExceededError, match="run_elapsed_us"):
        state.with_step_failure(late_failure)
    with pytest.raises(PilotRunContractError, match="no_started_attempt"):
        replace(failure, actual_network_attempts=0)


def test_probe_is_mandatory_before_any_shard_and_failure_stops() -> None:
    manifest = _manifest()
    auth = _authorization(manifest)
    now = auth.authorized_at_us + 1
    state = PilotRunStateV1(manifest).with_authorization(auth, now_us=now)
    state = state.with_preflight(
        _preflight(manifest, auth, -1, now + 1), now_us=now + 1
    )
    result = _shard_result(manifest, 0, start=now + 2_000_000, mono=20_000_000)
    with pytest.raises(PilotRunTransitionError, match="out_of_order"):
        state.with_shard_result(result)
    stopped = state.stopped(reason="endpoint_schema_mismatch", evidence_hash="c" * 64)
    assert stopped.next_action == "stopped"
    endpoint_intent = _network_intent(
        manifest,
        auth,
        state.preflight_receipts[-1],
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=now + 3,
        mono=9_000_000,
    )
    with pytest.raises(PilotRunTransitionError, match="stopped"):
        stopped.with_endpoint_verification(
            _verification(manifest, auth, endpoint_intent, start=now + 10)
        )


def test_endpoint_inventory_cannot_consume_future_shard_reservation() -> None:
    manifest = _manifest()
    auth = _authorization(manifest)
    now = auth.authorized_at_us + 1
    state = PilotRunStateV1(manifest).with_authorization(
        auth,
        now_us=auth.authorized_at_us,
    )
    preflight = _preflight(manifest, auth, -1, now)
    state = state.with_preflight(preflight, now_us=now)
    intent = _network_intent(
        manifest,
        auth,
        preflight,
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=now + 1,
        mono=9_000_000,
    )
    state = state.with_network_intent(intent)
    result = _verification(manifest, auth, intent, start=now + 10)
    per_stage_limit = 2 * manifest.endpoint_verification.max_network_attempts + 5
    with pytest.raises(
        PilotRunBudgetExceededError,
        match="verification_inventory_entries",
    ):
        state.with_endpoint_verification(
            replace(result, output_inventory_entries=per_stage_limit + 1)
        )


def test_pure_state_machine_completes_only_after_every_shard_and_anchor() -> None:
    manifest = _manifest()
    auth = _authorization(manifest)
    now = auth.authorized_at_us + 1
    state = PilotRunStateV1(manifest).with_authorization(auth, now_us=now)
    state = state.with_preflight(
        _preflight(manifest, auth, -1, now + 1), now_us=now + 1
    )
    verification_start = now + 10
    endpoint_intent = _network_intent(
        manifest,
        auth,
        state.preflight_receipts[-1],
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=now + 3,
        mono=9_000_000,
    )
    state = state.with_network_intent(endpoint_intent)
    verification = _verification(
        manifest, auth, endpoint_intent, start=verification_start
    )
    state = state.with_endpoint_verification(verification)
    epoch = verification.completed_at_us
    monotonic = verification.completed_monotonic_us
    for ordinal in range(len(manifest.shards)):
        checked = epoch + 10
        step_preflight = _preflight(manifest, auth, ordinal, checked)
        if ordinal == 0:
            with pytest.raises(
                PilotRunTransitionError,
                match="volume_identity_changed",
            ):
                state.with_preflight(
                    replace(step_preflight, volume_identity="fixture_volume_d"),
                    now_us=checked,
                )
        state = state.with_preflight(step_preflight, now_us=checked)
        intent = _network_intent(
            manifest,
            auth,
            state.preflight_receipts[-1],
            stage="shard_acquisition",
            ordinal=ordinal,
            issued_at=checked + 1,
            mono=monotonic + 1,
        )
        state = state.with_network_intent(intent)
        epoch += manifest.budgets.min_inter_step_spacing_us
        monotonic += manifest.budgets.min_inter_step_spacing_us
        result = _shard_result(
            manifest, ordinal, start=epoch, mono=monotonic, intent=intent
        )
        state = state.with_shard_result(result)
        epoch = result.step_completed_at_us
        monotonic = result.step_completed_monotonic_us
    assert state.next_action == "publish_detached_result_anchor"
    candidate_hash = state.result_candidate_hash
    run_control_bytes = state.final_run_control_inventory_bytes
    anchor = PilotRunAnchorReceiptV1(
        manifest_hash=manifest.manifest_hash,
        result_candidate_hash=candidate_hash,
        run_control_inventory_hash=state.final_run_control_inventory_hash,
        run_control_inventory_entries=state.final_run_control_inventory_entries,
        output_inventory_hash=state.expected_output_inventory_hash(),
        output_inventory_entries=state.actual_totals["inventory_entries"] + 1,
        run_control_bytes=run_control_bytes,
        total_output_bytes=(
            state.actual_totals["logical_storage_bytes"] + run_control_bytes
        ),
        clock_domain_id="fixture_clock_domain",
        fresh_inventory_scan_receipt_hash="a" * 64,
        fresh_inventory_reload_receipt_hash="b" * 64,
        fresh_inventory_scanned_at_us=epoch + 1,
        fresh_inventory_reloaded_at_us=epoch + 2,
        fresh_inventory_scanned_monotonic_us=monotonic + 1,
        fresh_inventory_reloaded_monotonic_us=monotonic + 2,
        external_anchor_domain_id="fixture_external_anchor",
        external_anchor_evidence_hash="d" * 64,
        anchored_at_us=epoch + 3,
        anchored_monotonic_us=monotonic + 3,
        final_run_elapsed_us=(
            monotonic
            + 3
            - endpoint_intent.durability_receipt.anchored_monotonic_us
        ),
    )
    complete = state.with_final_anchor(anchor)
    assert complete.next_action == "complete"
    assert complete.with_final_anchor(anchor) is complete
    assert parse_pilot_run_state_v1(complete.as_dict()) == complete
    assert anchor.run_control_bytes == state.final_run_control_inventory_bytes
    assert all(
        byte_count > 0
        for _kind, _locator, _semantic, _artifact, byte_count
        in state.final_run_control_inventory
    )
    with pytest.raises(PilotRunTransitionError, match="anchor_binding_mismatch"):
        state.with_final_anchor(
            replace(
                anchor,
                run_control_bytes=anchor.run_control_bytes + 1,
                total_output_bytes=anchor.total_output_bytes + 1,
            )
        )
    with pytest.raises(PilotRunContractError, match="unexpected_artifacts"):
        replace(anchor, unexpected_artifacts_absent=False)
    with pytest.raises(PilotRunTransitionError, match="anchor_binding_mismatch"):
        state.with_final_anchor(
            replace(
                anchor,
                fresh_inventory_scanned_at_us=epoch - 1,
                fresh_inventory_reloaded_at_us=epoch - 1,
            )
        )
    tampered = complete.as_dict()
    tampered["actual_totals"]["rows"] += 1
    with pytest.raises(PilotRunContractError, match="actual_totals_mismatch"):
        parse_pilot_run_state_v1(tampered)


def test_inter_step_spacing_and_actual_budget_fail_closed() -> None:
    manifest = _manifest()
    auth = _authorization(manifest)
    now = auth.authorized_at_us + 1
    state = PilotRunStateV1(manifest).with_authorization(auth, now_us=now)
    state = state.with_preflight(
        _preflight(manifest, auth, -1, now + 1), now_us=now + 1
    )
    endpoint_intent = _network_intent(
        manifest,
        auth,
        state.preflight_receipts[-1],
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=now + 3,
        mono=9_000_000,
    )
    state = state.with_network_intent(endpoint_intent)
    verification = _verification(
        manifest, auth, endpoint_intent, start=now + 10
    )
    state = state.with_endpoint_verification(verification)
    checked = verification.completed_at_us + 10
    state = state.with_preflight(
        _preflight(manifest, auth, 0, checked), now_us=checked
    )
    shard_intent = _network_intent(
        manifest,
        auth,
        state.preflight_receipts[-1],
        stage="shard_acquisition",
        ordinal=0,
        issued_at=checked + 1,
        mono=verification.completed_monotonic_us + 1,
    )
    state = state.with_network_intent(shard_intent)
    early = _shard_result(
        manifest,
        0,
        start=verification.completed_at_us + manifest.budgets.min_inter_step_spacing_us - 1,
        mono=verification.completed_monotonic_us
        + manifest.budgets.min_inter_step_spacing_us
        - 1,
        intent=shard_intent,
    )
    early = replace(
        early,
        observed_inter_step_delay_us=(
            manifest.budgets.min_inter_step_spacing_us - 1
        ),
    )
    with pytest.raises(PilotRunTransitionError, match="spacing"):
        state.with_shard_result(early)
    valid = _shard_result(
        manifest,
        0,
        start=verification.completed_at_us + manifest.budgets.min_inter_step_spacing_us,
        mono=verification.completed_monotonic_us
        + manifest.budgets.min_inter_step_spacing_us,
        intent=shard_intent,
    )
    too_many = replace(
        valid,
        actual_attempts=manifest.shards[0].request.required_pages + 1,
    )
    with pytest.raises(PilotRunBudgetExceededError, match="shard_attempts"):
        state.with_shard_result(too_many)
    with pytest.raises(PilotRunBudgetExceededError, match="shard_internal_sleep"):
        state.with_shard_result(replace(valid, observed_internal_sleep_us=1))


def test_actual_step_requires_current_preflight_and_final_totals_are_exact() -> None:
    manifest = _manifest()
    auth = _authorization(manifest)
    now = auth.authorized_at_us + 1
    state = PilotRunStateV1(manifest).with_authorization(auth, now_us=now)
    state = state.with_preflight(
        _preflight(manifest, auth, -1, now + 1), now_us=now + 1
    )
    endpoint_intent = _network_intent(
        manifest,
        auth,
        state.preflight_receipts[-1],
        stage="endpoint_verification",
        ordinal=-1,
        issued_at=now + 3,
        mono=9_000_000,
    )
    state = state.with_network_intent(endpoint_intent)
    verification = _verification(
        manifest, auth, endpoint_intent, start=now + 10
    )
    state = state.with_endpoint_verification(verification)
    checked = verification.completed_at_us + 10
    valid_preflight = _preflight(manifest, auth, 0, checked)
    state = state.with_preflight(valid_preflight, now_us=checked)
    shard_intent = _network_intent(
        manifest,
        auth,
        state.preflight_receipts[-1],
        stage="shard_acquisition",
        ordinal=0,
        issued_at=checked + 1,
        mono=verification.completed_monotonic_us + 1,
    )
    state = state.with_network_intent(shard_intent)
    start = verification.completed_at_us + manifest.budgets.min_inter_step_spacing_us
    result = _shard_result(
        manifest,
        0,
        start=start,
        mono=(
            verification.completed_monotonic_us
            + manifest.budgets.min_inter_step_spacing_us
        ),
        intent=shard_intent,
    )
    expired = replace(valid_preflight, valid_until_us=start)
    expired_intent = _network_intent(
        manifest,
        auth,
        expired,
        stage="shard_acquisition",
        ordinal=0,
        issued_at=checked + 1,
        mono=verification.completed_monotonic_us + 1,
    )
    expired_state = replace(
        state,
        preflight_receipts=(state.preflight_receipts[0], expired),
        network_intents=(state.network_intents[0], expired_intent),
    )
    expired_result = replace(
        result,
        network_intent_hash=expired_intent.intent_hash,
    )
    with pytest.raises(PilotRunPreflightError, match="not_current"):
        expired_state.with_shard_result(expired_result)


def test_artifact_reads_are_bounded_and_state_remains_a_pure_projection(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    state = PilotRunStateV1(manifest)
    assert parse_pilot_run_state_v1(state.as_dict()) == state

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    with pytest.raises(PilotRunArtifactError, match="oversized"):
        load_pilot_run_manifest_v1(oversized)
    with pytest.raises(PilotRunArtifactError, match="oversized"):
        publish_pilot_run_manifest_v1(oversized, manifest)


def test_no_network_or_executor_is_exposed_by_contract_module() -> None:
    import trading.market_data.mexc_pilot_run as module

    public = set(module.__all__)
    assert not any("Executor" in name or "Transport" in name for name in public)
    assert not hasattr(module, "run_network_pilot")
    assert "publish_pilot_run_state_v1" not in public
