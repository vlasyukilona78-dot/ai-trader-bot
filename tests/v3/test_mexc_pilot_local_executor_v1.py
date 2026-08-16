from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import inspect
from pathlib import Path

import pytest

import trading.market_data.mexc_pilot_local_executor as module
from trading.market_data.mexc_futures_transport import (
    HistoryResourceLimitsV1,
    candidate_endpoint_fixture_path,
    candidate_history_retry_policy_v1,
    load_mexc_futures_endpoint_contract_v1,
)
from trading.market_data.mexc_pilot_local_executor import (
    DetachedAnchorEvidenceV1,
    DetachedAnchorSubjectV1,
    EndpointStageDraftV1,
    PilotExecutorBindingsV1,
    PilotLocalExecutorContractError,
    ShardStageDraftV1,
    StageFailureDraftV1,
    build_concrete_pilot_manifest_v1,
    pilot_local_executor_contract_hash,
)
from trading.market_data.mexc_pilot_run import (
    PilotGlobalBudgetsV1,
    PilotRunContractError,
    PilotShardPlanV1,
    parse_pilot_run_manifest_v1,
    pilot_run_contract_hash,
)
from trading.market_data.strict_history_v2 import HistoryRangeRequestV2


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
    rows: int,
) -> HistoryRangeRequestV2:
    seconds = 60 if interval == "Min1" else 3_600
    return HistoryRangeRequestV2(
        venue="mexc_contract",
        symbol=symbol,
        venue_symbol=venue_symbol,
        interval=interval,
        start_open_ts=BASE - rows * seconds,
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
        request = _request(
            symbol,
            venue_symbol,
            interval="Min1",
            rows=7 * 1_440,
        )
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
        request = _request(
            symbol,
            venue_symbol,
            interval="Min60",
            rows=7 * 24,
        )
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
    deep = _request(
        "BTCUSDT",
        "BTC_USDT",
        interval="Min1",
        rows=140 * 1_440,
    )
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


def _bindings() -> PilotExecutorBindingsV1:
    return PilotExecutorBindingsV1(
        coordinator_contract_version="fixture_coordinator_v1",
        coordinator_contract_hash="1" * 64,
        local_store_contract_version="fixture_local_store_v1",
        local_store_contract_hash="2" * 64,
        clock_contract_version="fixture_clock_v1",
        clock_contract_hash="3" * 64,
        detached_anchor_sink_contract_version="fixture_anchor_sink_v1",
        detached_anchor_sink_contract_hash="4" * 64,
        endpoint_runner_contract_version="fixture_endpoint_runner_v1",
        endpoint_runner_contract_hash="5" * 64,
        shard_runner_contract_version="fixture_shard_runner_v1",
        shard_runner_contract_hash="6" * 64,
    )


def _builder_args() -> dict[str, object]:
    probe = _probe_request()
    return {
        "repository_commit": "1" * 40,
        "repository_tree_receipt_hash": "2" * 64,
        "created_at_us": 1_900_000_000_000_000,
        "parent_master_plan_path": "docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md",
        "parent_master_plan_sha256": "3" * 64,
        "parent_adr_path": "docs/ADR_MEXC_V3_FINAL_BOT_2026-08-15.md",
        "parent_adr_sha256": "4" * 64,
        "output_root_locator": "file:///C:/koteika-pilot/executor-fixture",
        "endpoint_probe_request": probe,
        "endpoint_relative_artifact_root": f"verification/{probe.request_id}",
        "official_reference_url": probe.endpoint_contract.plan_reference_url,
        "endpoint_max_network_attempts": 2,
        "endpoint_max_total_raw_body_bytes": 2 * 1024**2,
        "endpoint_max_total_storage_bytes": 8 * 1024**2,
        "endpoint_max_runtime_us": 2 * 60 * 1_000_000,
        "endpoint_max_total_sleep_us": 30 * 1_000_000,
        "ordered_shards": _shards(),
        "budgets": _budgets(),
        "executor_bindings": _bindings(),
    }


def test_binding_round_trip_and_component_drift_domains_are_exact() -> None:
    bindings = _bindings()
    assert PilotExecutorBindingsV1.from_dict(bindings.as_dict()) == bindings
    assert len(bindings.bindings_hash) == 64

    endpoint_changed = replace(bindings, endpoint_runner_contract_hash="a" * 64)
    assert (
        endpoint_changed.endpoint_verifier_binding_hash
        != bindings.endpoint_verifier_binding_hash
    )
    assert (
        endpoint_changed.shard_executor_binding_hash
        == bindings.shard_executor_binding_hash
    )

    shard_changed = replace(bindings, shard_runner_contract_hash="b" * 64)
    assert (
        shard_changed.endpoint_verifier_binding_hash
        == bindings.endpoint_verifier_binding_hash
    )
    assert shard_changed.shard_executor_binding_hash != bindings.shard_executor_binding_hash

    for field in (
        "coordinator_contract_hash",
        "local_store_contract_hash",
        "clock_contract_hash",
        "detached_anchor_sink_contract_hash",
    ):
        changed = replace(bindings, **{field: "c" * 64})
        assert changed.endpoint_verifier_binding_hash != bindings.endpoint_verifier_binding_hash
        assert changed.shard_executor_binding_hash != bindings.shard_executor_binding_hash

    tampered = bindings.as_dict()
    tampered["endpoint_verifier_binding"]["contract_hash"] = "d" * 64
    with pytest.raises(
        PilotLocalExecutorContractError,
        match="composite_binding_mismatch",
    ):
        PilotExecutorBindingsV1.from_dict(tampered)
    with pytest.raises(FrozenInstanceError):
        bindings.clock_contract_hash = "e" * 64


def test_builder_wires_every_explicit_input_and_round_trips() -> None:
    args = _builder_args()
    bindings = args["executor_bindings"]
    assert isinstance(bindings, PilotExecutorBindingsV1)
    manifest = build_concrete_pilot_manifest_v1(**args)

    assert manifest.repository_commit == args["repository_commit"]
    assert manifest.repository_tree_receipt_hash == args["repository_tree_receipt_hash"]
    assert manifest.created_at_us == args["created_at_us"]
    assert manifest.parent_master_plan_path == args["parent_master_plan_path"]
    assert manifest.parent_master_plan_sha256 == args["parent_master_plan_sha256"]
    assert manifest.parent_adr_path == args["parent_adr_path"]
    assert manifest.parent_adr_sha256 == args["parent_adr_sha256"]
    assert manifest.output_root_locator == args["output_root_locator"]
    assert manifest.shards == args["ordered_shards"]
    assert manifest.budgets == args["budgets"]
    assert (
        manifest.endpoint_verification.verifier_contract_version
        == bindings.endpoint_verifier_binding_version
    )
    assert (
        manifest.endpoint_verification.verifier_contract_hash
        == bindings.endpoint_verifier_binding_hash
    )
    assert (
        manifest.shard_executor_contract_version
        == bindings.shard_executor_binding_version
    )
    assert manifest.shard_executor_contract_hash == bindings.shard_executor_binding_hash
    assert manifest.endpoint_verification.probe_request == args["endpoint_probe_request"]
    assert (
        manifest.endpoint_verification.relative_artifact_root
        == args["endpoint_relative_artifact_root"]
    )
    assert parse_pilot_run_manifest_v1(manifest.as_dict()) == manifest

    changed_args = dict(args)
    changed_args["executor_bindings"] = replace(
        bindings,
        local_store_contract_hash="f" * 64,
    )
    changed = build_concrete_pilot_manifest_v1(**changed_args)
    assert changed.manifest_hash != manifest.manifest_hash
    assert changed.endpoint_verification.verifier_contract_hash != (
        manifest.endpoint_verification.verifier_contract_hash
    )
    assert changed.shard_executor_contract_hash != manifest.shard_executor_contract_hash


def test_builder_has_no_defaults_and_rejects_missing_or_implicit_inputs() -> None:
    signature = inspect.signature(build_concrete_pilot_manifest_v1)
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    args = _builder_args()
    for name in signature.parameters:
        missing = dict(args)
        missing.pop(name)
        with pytest.raises(TypeError):
            build_concrete_pilot_manifest_v1(**missing)

    with pytest.raises(TypeError):
        build_concrete_pilot_manifest_v1(*args.values())

    invalid = dict(args)
    invalid["ordered_shards"] = list(args["ordered_shards"])
    with pytest.raises(
        PilotLocalExecutorContractError,
        match="ordered_shards_are_invalid",
    ):
        build_concrete_pilot_manifest_v1(**invalid)

    for field, value, code in (
        ("executor_bindings", object(), "executor_bindings_are_invalid"),
        ("endpoint_probe_request", object(), "endpoint_probe_request_is_invalid"),
        ("budgets", object(), "budgets_are_invalid"),
    ):
        invalid = dict(args)
        invalid[field] = value
        with pytest.raises(PilotLocalExecutorContractError, match=code):
            build_concrete_pilot_manifest_v1(**invalid)


def test_builder_delegates_exact_manifest_and_endpoint_validation() -> None:
    args = _builder_args()
    invalid = dict(args)
    invalid["repository_commit"] = "not-a-commit"
    with pytest.raises(PilotRunContractError, match="repository_commit_is_invalid"):
        build_concrete_pilot_manifest_v1(**invalid)

    invalid = dict(args)
    invalid["official_reference_url"] = "https://example.invalid/not-mexc"
    with pytest.raises(PilotRunContractError, match="endpoint_reference_url_mismatch"):
        build_concrete_pilot_manifest_v1(**invalid)

    invalid = dict(args)
    invalid["endpoint_max_network_attempts"] = 1
    with pytest.raises(PilotRunContractError, match="verification_attempts_is_invalid"):
        build_concrete_pilot_manifest_v1(**invalid)

    invalid = dict(args)
    invalid["output_root_locator"] = "file:///C:/data/history/pilot"
    with pytest.raises(PilotRunContractError, match="legacy_history"):
        build_concrete_pilot_manifest_v1(**invalid)


def test_stage_drafts_are_immutable_references_without_authoritative_totals() -> None:
    endpoint = EndpointStageDraftV1(
        manifest_hash="1" * 64,
        authorization_receipt_hash="2" * 64,
        network_intent_hash="3" * 64,
        clock_domain_id="fixture_clock_domain",
        stage_started_at_us=100,
        stage_completed_at_us=160,
        stage_started_monotonic_us=1_000,
        stage_completed_monotonic_us=1_060,
        official_document_evidence_relative_path="official/reference.json",
        official_document_evidence_hash="4" * 64,
        official_document_request_started_at_us=110,
        official_document_fetched_at_us=120,
        official_document_request_started_monotonic_us=1_010,
        official_document_fetched_monotonic_us=1_020,
        live_probe_store_relative_root="live-probe",
        live_history_manifest_hash="5" * 64,
        live_probe_started_at_us=130,
        live_probe_completed_at_us=150,
        live_probe_started_monotonic_us=1_030,
        live_probe_completed_monotonic_us=1_050,
    )
    shard = ShardStageDraftV1(
        manifest_hash="1" * 64,
        network_intent_hash="3" * 64,
        ordinal=0,
        clock_domain_id="fixture_clock_domain",
        step_started_at_us=200,
        step_completed_at_us=250,
        step_started_monotonic_us=2_000,
        step_completed_monotonic_us=2_050,
        history_manifest_hash="6" * 64,
    )
    failure = StageFailureDraftV1(
        manifest_hash="1" * 64,
        authorization_receipt_hash="2" * 64,
        network_intent_hash="3" * 64,
        stage="shard_acquisition",
        ordinal=0,
        clock_domain_id="fixture_clock_domain",
        step_started_at_us=200,
        step_completed_at_us=225,
        step_started_monotonic_us=2_000,
        step_completed_monotonic_us=2_025,
        error_code="fixture_transport_failure",
        error_evidence_hash="7" * 64,
    )
    assert EndpointStageDraftV1.from_dict(endpoint.as_dict()) == endpoint
    assert ShardStageDraftV1.from_dict(shard.as_dict()) == shard
    assert StageFailureDraftV1.from_dict(failure.as_dict()) == failure

    forbidden_counters = {
        "actual_network_attempts",
        "actual_pages",
        "actual_rows",
        "actual_attempts",
        "actual_raw_body_bytes",
        "actual_storage_bytes",
        "actual_logical_storage_bytes",
        "output_inventory_entries",
        "output_inventory_bytes",
        "observed_sleep_us",
        "actual_runtime_us",
    }
    for draft_type in (EndpointStageDraftV1, ShardStageDraftV1, StageFailureDraftV1):
        assert forbidden_counters.isdisjoint(draft_type.__dataclass_fields__)

    with pytest.raises(FrozenInstanceError):
        shard.ordinal = 1
    with pytest.raises(
        PilotLocalExecutorContractError,
        match="timing_is_invalid",
    ):
        replace(endpoint, live_probe_started_at_us=105)
    with pytest.raises(
        PilotLocalExecutorContractError,
        match="ordinal_is_invalid",
    ):
        replace(failure, stage="endpoint_verification", ordinal=0)


def test_detached_anchor_types_bind_subject_sink_domain_and_clock() -> None:
    subject = DetachedAnchorSubjectV1(
        manifest_hash="1" * 64,
        subject_kind="network_intent_reservation",
        subject_hash="2" * 64,
        clock_contract_version="fixture_clock_v1",
        clock_contract_hash="8" * 64,
        clock_domain_id="fixture_clock_domain",
        requested_at_us=100,
        requested_monotonic_us=1_000,
    )
    evidence = DetachedAnchorEvidenceV1(
        subject_receipt_hash=subject.subject_receipt_hash,
        anchor_sink_contract_version="fixture_anchor_sink_v1",
        anchor_sink_contract_hash="3" * 64,
        anchor_domain_id="fixture_anchor_domain",
        clock_contract_version="fixture_clock_v1",
        clock_contract_hash="8" * 64,
        clock_domain_id="fixture_clock_domain",
        evidence_hash="4" * 64,
        anchored_at_us=101,
        anchored_monotonic_us=1_001,
    )
    evidence.validate_for(
        subject,
        anchor_sink_contract_version="fixture_anchor_sink_v1",
        anchor_sink_contract_hash="3" * 64,
        anchor_domain_id="fixture_anchor_domain",
    )
    assert DetachedAnchorSubjectV1.from_dict(subject.as_dict()) == subject
    assert DetachedAnchorEvidenceV1.from_dict(evidence.as_dict()) == evidence
    with pytest.raises(
        PilotLocalExecutorContractError,
        match="binding_mismatch",
    ):
        replace(evidence, anchor_domain_id="another_domain").validate_for(
            subject,
            anchor_sink_contract_version="fixture_anchor_sink_v1",
            anchor_sink_contract_hash="3" * 64,
            anchor_domain_id="fixture_anchor_domain",
        )

    for field, value in (
        ("clock_contract_version", "another_clock_v1"),
        ("clock_contract_hash", "9" * 64),
        ("clock_domain_id", "another_clock_domain"),
    ):
        with pytest.raises(
            PilotLocalExecutorContractError,
            match="binding_mismatch",
        ):
            replace(evidence, **{field: value}).validate_for(
                subject,
                anchor_sink_contract_version="fixture_anchor_sink_v1",
                anchor_sink_contract_hash="3" * 64,
                anchor_domain_id="fixture_anchor_domain",
            )


@pytest.mark.parametrize(
    "relative_path",
    (
        "CON",
        "AUX.txt",
        ".",
        "C:foo",
        "foo.",
        "foo ",
        "official/reference.json:stream",
        "a<b",
        "a?b",
        "a\x01b",
        "data/history/new",
        "Data/History/new",
    ),
)
def test_stage_draft_paths_reject_windows_aliases_and_legacy_cache(
    relative_path: str,
) -> None:
    with pytest.raises(PilotLocalExecutorContractError):
        EndpointStageDraftV1(
            manifest_hash="1" * 64,
            authorization_receipt_hash="2" * 64,
            network_intent_hash="3" * 64,
            clock_domain_id="fixture_clock_domain",
            stage_started_at_us=100,
            stage_completed_at_us=160,
            stage_started_monotonic_us=1_000,
            stage_completed_monotonic_us=1_060,
            official_document_evidence_relative_path=relative_path,
            official_document_evidence_hash="4" * 64,
            official_document_request_started_at_us=110,
            official_document_fetched_at_us=120,
            official_document_request_started_monotonic_us=1_010,
            official_document_fetched_monotonic_us=1_020,
            live_probe_store_relative_root="live-probe",
            live_history_manifest_hash="5" * 64,
            live_probe_started_at_us=130,
            live_probe_completed_at_us=150,
            live_probe_started_monotonic_us=1_030,
            live_probe_completed_monotonic_us=1_050,
        )


def test_module_has_no_discovery_network_defaults_or_authorization_factory() -> None:
    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(
        {"os", "subprocess", "socket", "requests", "httpx", "urllib", "git"}
    )

    public = set(module.__all__)
    assert not any("Coordinator" in name or "ArtifactStore" in name for name in public)
    assert not any("Authorization" in name or "Http" in name for name in public)
    assert not hasattr(module, "run_pilot")
    assert not hasattr(module, "create_u5_authorization")
    assert not hasattr(module, "default_endpoint_runner")
    assert not hasattr(module, "default_shard_runner")


def test_schema_pins_public_protocol_and_bindings_wire_shapes() -> None:
    protocols = module._CONTRACT_SCHEMA["protocols"]

    def non_self_parameters(owner: type, method: str) -> list[inspect.Parameter]:
        return list(inspect.signature(getattr(owner, method)).parameters.values())[1:]

    assert [item.name for item in non_self_parameters(module.PilotClock, "sleep_us")] == [
        "duration_us"
    ]
    assert [item.name for item in non_self_parameters(module.DetachedAnchorSink, "create_once")] == [
        "subject",
        "clock",
    ]
    assert non_self_parameters(module.DetachedAnchorSink, "create_once")[1].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    for owner, schema_key in (
        (module.EndpointStageRunner, "endpoint_stage_runner"),
        (module.ShardStageRunner, "shard_stage_runner"),
    ):
        parameters = non_self_parameters(owner, "execute")
        assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters)
        assert [item.name for item in parameters] == list(
            protocols[schema_key]["execute_keyword_only_parameters"]
        )

    bindings = _bindings()
    assert list(bindings.as_dict()) == module._CONTRACT_SCHEMA["object_contracts"][
        "executor_bindings"
    ]["wire_top_level_keys_in_order"]
    assert set(bindings.as_dict()["coordinator"]) == {
        "contract_version",
        "contract_hash",
    }


def test_executor_schema_is_pinned_and_preserves_parent_pin() -> None:
    first = pilot_local_executor_contract_hash()
    assert first == "72c206bc2f22a8101a7d6fdc97458e865a6c4c3e5ed7290c64c1ca8c3594fc31"
    assert pilot_local_executor_contract_hash() == first
    assert pilot_run_contract_hash() == (
        "f3d642d436e9d4a44e65f35c6ea8375bd92b4b36b30f1c86af54936a608ce65e"
    )
    assert module._PINNED_CONTRACT_HASH == first
    assert module._CONTRACT_SCHEMA["scalar_contracts"]["relative_path"] == {
        "separator": "/",
        "maximum_characters": 240,
        "absolute_empty_root_dot_dot_and_parent_segments": "rejected",
        "backslash_and_ascii_controls": "rejected",
        "windows_illegal_characters": '<>:"|?*',
        "windows_trailing_dot_or_space": "rejected",
        "windows_reserved_basenames": sorted(module._WINDOWS_RESERVED_BASENAMES),
        "legacy_data_history_pair_case_insensitive": "rejected",
        "canonical_posix_rerender_required": True,
    }
    assert module._CONTRACT_SCHEMA["builder"]["parameters_in_order"] == list(
        inspect.signature(build_concrete_pilot_manifest_v1).parameters
    )
