from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading

import pytest
import trading.market_data.mexc_pilot_local_store as pilot_store_module

from trading.market_data.mexc_futures_transport import (
    HistoryResourceLimitsV1,
    candidate_endpoint_fixture_path,
    candidate_history_retry_policy_v1,
    load_mexc_futures_endpoint_contract_v1,
)
from trading.market_data.mexc_pilot_local_store import (
    PILOT_LOCAL_STORE_CONTRACT_VERSION,
    PILOT_FILESYSTEM_MUTATION_BOUNDARY,
    PILOT_FUTURE_HTTP_ATTEMPT_GATE_POLICY,
    PILOT_REAL_U5_CONSTRUCTION_POLICY,
    PILOT_RUNTIME_IDENTITY_ASSURANCE,
    MexcPilotLocalStoreV1,
    PilotClockSampleV1,
    PilotDetachedEvidenceReceiptV1,
    PilotDetachedEvidenceRequestV1,
    PilotLocalStoreBoundsError,
    PilotLocalStoreConflictError,
    PilotLocalStoreError,
    PilotLocalStoreLockError,
    PilotLocalStoreRecoveryError,
    PilotRuntimeAuthorityBindingV1,
    PilotU5VerificationEvidenceV1,
    PilotU5VerificationRequestV1,
    canonical_lf_bytes,
    mexc_pilot_local_store_contract_hash,
    pilot_runtime_authority_binding_contract_hash,
    parse_canonical_lf_json,
)
from trading.market_data.mexc_pilot_local_executor import PilotExecutorBindingsV1
from trading.market_data.mexc_pilot_run import (
    EndpointVerificationPlanV1,
    MexcPublicQaPilotRunManifestV1,
    PilotDiskPreflightReceiptV1,
    PilotGlobalBudgetsV1,
    PilotNetworkIntentV1,
    PilotRunAuthorizationError,
    PilotRunPreflightError,
    PilotRunStateV1,
    PilotShardPlanV1,
    U5PublicPilotAuthorizationReceiptV1,
)
from trading.market_data.strict_history_v2 import HistoryRangeRequestV2


BASE = 1_767_225_600
EPOCH = 1_900_000_000_100_000
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
CLOCK_VERSION = "fixture_clock_v1"
CLOCK_HASH = "7" * 64
CLOCK_DOMAIN = "fixture_clock_domain"
SINK_VERSION = "fixture_detached_sink_v1"
SINK_HASH = "8" * 64
SINK_DOMAIN = "fixture_detached_sink_domain"
U5_VERIFIER_VERSION = "fixture_u5_verifier_v1"
U5_VERIFIER_HASH = "9" * 64
U5_VERIFIER_DOMAIN = "fixture_u5_verifier_domain"
U5_TRUST_KEY = "fixture_u5_trust_key"
U5_POLICY_VERSION = "fixture_u5_policy_v1"
U5_POLICY_HASH = "a" * 64
ENDPOINT_RUNNER_VERSION = "fixture_endpoint_runner_v1"
ENDPOINT_RUNNER_HASH = "b" * 64
SHARD_RUNNER_VERSION = "fixture_shard_runner_v1"
SHARD_RUNNER_HASH = "c" * 64


class FakeClock:
    def __init__(
        self,
        *,
        epoch_us: int = EPOCH + 10,
        monotonic_us: int = 10_000_000,
        step_us: int = 10,
    ) -> None:
        self.epoch_us = epoch_us
        self.monotonic_us = monotonic_us
        self.step_us = step_us

    contract_version = CLOCK_VERSION
    contract_hash = CLOCK_HASH
    clock_domain_id = CLOCK_DOMAIN

    def sample(self) -> PilotClockSampleV1:
        result = PilotClockSampleV1(
            epoch_us=self.epoch_us,
            monotonic_us=self.monotonic_us,
            clock_domain_id=self.clock_domain_id,
        )
        self.epoch_us += self.step_us
        self.monotonic_us += self.step_us
        return result


class FakeDetachedSink:
    contract_version = SINK_VERSION
    contract_hash = SINK_HASH
    domain_id = SINK_DOMAIN

    def __init__(self, *, fail_kind: str | None = None, on_anchor=None) -> None:
        self.fail_kind = fail_kind
        self.on_anchor = on_anchor
        self.requests: list[PilotDetachedEvidenceRequestV1] = []

    def anchor(
        self, request: PilotDetachedEvidenceRequestV1
    ) -> PilotDetachedEvidenceReceiptV1:
        self.requests.append(request)
        if self.on_anchor is not None:
            self.on_anchor(request)
        if request.evidence_kind == self.fail_kind:
            raise RuntimeError("fixture_detached_sink_failure")
        evidence = hashlib.sha256(
            canonical_lf_bytes(
                {
                    "domain": "fixture_detached_evidence_v1",
                    "ordinal": len(self.requests),
                    "request": request.as_dict(),
                }
            )
        ).hexdigest()
        return PilotDetachedEvidenceReceiptV1(
            request_hash=request.request_hash,
            evidence_hash=evidence,
            anchored_at_us=request.observed_at_us,
            anchored_monotonic_us=request.observed_monotonic_us,
            clock_domain_id=request.clock_domain_id,
        )


class FakeU5Verifier:
    contract_version = U5_VERIFIER_VERSION
    contract_hash = U5_VERIFIER_HASH
    domain_id = U5_VERIFIER_DOMAIN
    trust_key_id = U5_TRUST_KEY
    policy_version = U5_POLICY_VERSION
    policy_hash = U5_POLICY_HASH

    def __init__(self, *, corrupt_binding: bool = False) -> None:
        self.corrupt_binding = corrupt_binding
        self.requests: list[PilotU5VerificationRequestV1] = []

    def verify(
        self, request: PilotU5VerificationRequestV1
    ) -> PilotU5VerificationEvidenceV1:
        self.requests.append(request)
        evidence_hash = hashlib.sha256(
            canonical_lf_bytes(
                {
                    "domain": "fixture_offline_u5_verification_v1",
                    "request": request.as_dict(),
                }
            )
        ).hexdigest()
        return PilotU5VerificationEvidenceV1(
            request_hash=request.request_hash,
            manifest_hash=("f" * 64 if self.corrupt_binding else request.manifest_hash),
            authorization_receipt_hash=request.authorization_receipt_hash,
            external_authority_evidence_hash=(
                request.external_authority_evidence_hash
            ),
            executor_bindings_hash=request.executor_bindings_hash,
            runtime_authority_binding_hash=(
                request.runtime_authority_binding_hash
            ),
            process_challenge_hash=request.process_challenge_hash,
            verification_evidence_hash=evidence_hash,
            verified_at_us=request.requested_at_us,
            verified_monotonic_us=request.requested_monotonic_us,
            clock_contract_version=request.clock_contract_version,
            clock_contract_hash=request.clock_contract_hash,
            clock_domain_id=request.clock_domain_id,
            u5_verifier_contract_version=request.u5_verifier_contract_version,
            u5_verifier_contract_hash=request.u5_verifier_contract_hash,
            u5_verifier_domain_id=request.u5_verifier_domain_id,
            u5_verifier_trust_key_id=request.u5_verifier_trust_key_id,
            u5_verifier_policy_version=request.u5_verifier_policy_version,
            u5_verifier_policy_hash=request.u5_verifier_policy_hash,
        )


class FakeOwnedIntentRunner:
    def __init__(self, callback, *, shard: bool = False) -> None:
        self.contract_version = (
            SHARD_RUNNER_VERSION if shard else ENDPOINT_RUNNER_VERSION
        )
        self.contract_hash = SHARD_RUNNER_HASH if shard else ENDPOINT_RUNNER_HASH
        self.callback = callback

    def __call__(self, intent: PilotNetworkIntentV1):
        return self.callback(intent)


def _runtime_binding() -> PilotRuntimeAuthorityBindingV1:
    return PilotRuntimeAuthorityBindingV1(
        coordinator_implementation_contract_version="fixture_coordinator_impl_v1",
        coordinator_implementation_contract_hash="d" * 64,
        clock_contract_version=CLOCK_VERSION,
        clock_contract_hash=CLOCK_HASH,
        clock_domain_id=CLOCK_DOMAIN,
        detached_anchor_sink_contract_version=SINK_VERSION,
        detached_anchor_sink_contract_hash=SINK_HASH,
        detached_anchor_sink_domain_id=SINK_DOMAIN,
        u5_verifier_contract_version=U5_VERIFIER_VERSION,
        u5_verifier_contract_hash=U5_VERIFIER_HASH,
        u5_verifier_domain_id=U5_VERIFIER_DOMAIN,
        u5_verifier_trust_key_id=U5_TRUST_KEY,
        u5_verifier_policy_version=U5_POLICY_VERSION,
        u5_verifier_policy_hash=U5_POLICY_HASH,
    )


def _executor_bindings(
    runtime: PilotRuntimeAuthorityBindingV1 | None = None,
) -> PilotExecutorBindingsV1:
    selected = runtime or _runtime_binding()
    return PilotExecutorBindingsV1(
        coordinator_contract_version=selected.contract_version,
        coordinator_contract_hash=selected.binding_hash,
        local_store_contract_version=PILOT_LOCAL_STORE_CONTRACT_VERSION,
        local_store_contract_hash=mexc_pilot_local_store_contract_hash(),
        clock_contract_version=CLOCK_VERSION,
        clock_contract_hash=CLOCK_HASH,
        detached_anchor_sink_contract_version=SINK_VERSION,
        detached_anchor_sink_contract_hash=SINK_HASH,
        endpoint_runner_contract_version=ENDPOINT_RUNNER_VERSION,
        endpoint_runner_contract_hash=ENDPOINT_RUNNER_HASH,
        shard_runner_contract_version=SHARD_RUNNER_VERSION,
        shard_runner_contract_hash=SHARD_RUNNER_HASH,
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
    days: int,
) -> HistoryRangeRequestV2:
    seconds = 60 if interval == "Min1" else 3_600
    rows = days * 86_400 // seconds
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


def _output_locator(path: Path) -> str:
    if os.name == "nt":
        absolute = Path(os.path.abspath(path)).as_posix()
        return f"file:///{absolute[0].upper()}{absolute[1:]}"
    identity = hashlib.sha256(os.fsencode(path)).hexdigest()[:16]
    return f"file:///C:/pilot-local-store-tests/{identity}"


def _manifest(
    output_root: Path,
    executor_bindings: PilotExecutorBindingsV1 | None = None,
) -> MexcPublicQaPilotRunManifestV1:
    bindings = executor_bindings or _executor_bindings()
    probe = _probe_request()
    verification = EndpointVerificationPlanV1(
        probe_request=probe,
        relative_artifact_root=f"verification/{probe.request_id}",
        official_reference_url=probe.endpoint_contract.plan_reference_url,
        verifier_contract_version=bindings.endpoint_verifier_binding_version,
        verifier_contract_hash=bindings.endpoint_verifier_binding_hash,
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
        output_root_locator=_output_locator(output_root),
        shard_executor_contract_version=bindings.shard_executor_binding_version,
        shard_executor_contract_hash=bindings.shard_executor_binding_hash,
        endpoint_verification=verification,
        shards=_shards(),
        budgets=_budgets(),
    )


def _authorization(
    manifest: MexcPublicQaPilotRunManifestV1,
    *,
    expires_at_us: int | None = None,
) -> U5PublicPilotAuthorizationReceiptV1:
    request = manifest.endpoint_verification.probe_request
    return U5PublicPilotAuthorizationReceiptV1(
        manifest_hash=manifest.manifest_hash,
        manifest_identity=manifest.manifest_identity,
        authority_id="fixture_user_authority",
        orchestrator_session_id="fixture_orchestrator_session",
        authorized_at_us=EPOCH,
        expires_at_us=(
            expires_at_us
            if expires_at_us is not None
            else EPOCH + 24 * 60 * 60 * 1_000_000
        ),
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


def _preflight(
    manifest: MexcPublicQaPilotRunManifestV1,
    authorization: U5PublicPilotAuthorizationReceiptV1,
    *,
    step: int = -1,
    checked_at_us: int = EPOCH + 20,
) -> PilotDiskPreflightReceiptV1:
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
        checked_at_us=checked_at_us,
        valid_until_us=checked_at_us + 30 * 1_000_000,
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


def _store(
    tmp_path: Path,
    *,
    manifest: MexcPublicQaPilotRunManifestV1 | None = None,
    sink: FakeDetachedSink | None = None,
    clock: FakeClock | None = None,
    verifier: FakeU5Verifier | None = None,
) -> tuple[MexcPilotLocalStoreV1, MexcPublicQaPilotRunManifestV1, FakeDetachedSink]:
    output_root = tmp_path / "subject"
    runtime = _runtime_binding()
    bindings = _executor_bindings(runtime)
    selected = manifest or _manifest(output_root, bindings)
    selected_sink = sink or FakeDetachedSink()
    return (
        MexcPilotLocalStoreV1(
            manifest=selected,
            executor_bindings=bindings,
            runtime_authority_binding=runtime,
            output_root=output_root,
            detached_evidence_sink=selected_sink,
            clock=clock or FakeClock(),
            publisher_instance_id="fixture_publisher_instance",
            u5_authority_verifier=verifier or FakeU5Verifier(),
        ),
        selected,
        selected_sink,
    )


def _state_ready_for_intent(
    store: MexcPilotLocalStoreV1,
    manifest: MexcPublicQaPilotRunManifestV1,
) -> tuple[PilotRunStateV1, U5PublicPilotAuthorizationReceiptV1]:
    state = store.publish_manifest()
    authorization = _authorization(manifest)
    state = store.publish_authorization(authorization)
    state = store.measure_and_publish_preflight(state)
    assert state.next_action == "run_endpoint_verification_stage"
    return state, authorization


def test_local_store_and_runtime_authority_contracts_are_pinned() -> None:
    digest = mexc_pilot_local_store_contract_hash()
    assert digest == "21f27ec667d588ac254b893c5f25e44634cc2de1f8567efafc85d08fccca94ab"
    assert (
        pilot_runtime_authority_binding_contract_hash()
        == "392aa908b30dac7d244a3efdd67887933ab54e5ec07168b184dbedc5870c1004"
    )
    runtime = _runtime_binding()
    assert PilotRuntimeAuthorityBindingV1.from_dict(runtime.as_dict()) == runtime
    with pytest.raises(PilotLocalStoreError, match="schema_mismatch"):
        PilotRuntimeAuthorityBindingV1.from_dict(
            {**runtime.as_dict(), "unknown": "forbidden"}
        )


def test_runtime_identities_are_explicitly_not_real_u5_attestation() -> None:
    assert PILOT_RUNTIME_IDENTITY_ASSURANCE == (
        "declarative_self_reported_not_code_or_cryptographic_attestation"
    )
    assert PILOT_REAL_U5_CONSTRUCTION_POLICY == (
        "blocked_until_reviewed_coordinator_constructs_no_arbitrary_plugins_or_"
        "independent_pinned_key_crypto_verifier"
    )
    assert PILOT_FUTURE_HTTP_ATTEMPT_GATE_POLICY == (
        "trusted_now_latest_preflight_entire_remaining_worst_case_before_every_"
        "http_attempt_no_network_between_gate_and_attempt"
    )
    assert PILOT_FILESYSTEM_MUTATION_BOUNDARY == (
        "static_hostile_state_rejected_cooperating_writers_share_run_lock_finite_"
        "double_scan_not_atomic_external_mutation_out_of_scope_real_u5_requires_"
        "operator_acceptance_or_handle_relative_snapshot"
    )
    assert "Real U5 remains\nblocked" in (pilot_store_module.__doc__ or "")
    assert "not an atomic NTFS snapshot" in (pilot_store_module.__doc__ or "")
    assert not hasattr(MexcPilotLocalStoreV1, "create_u5_authorization")
    assert not hasattr(MexcPilotLocalStoreV1, "load_runtime_plugin")


def test_exact_canonical_lf_parser_rejects_aliases() -> None:
    assert parse_canonical_lf_json(b'{"a":1}\n') == {"a": 1}
    for bad in (
        b'{"a":1,"a":1}\n',
        b'{"a":1.0}\n',
        b'{"a":NaN}\n',
        b'{ "a":1}\n',
        b'{"a":1}\r\n',
        b'{"a":1}',
    ):
        with pytest.raises(PilotLocalStoreError):
            parse_canonical_lf_json(bad)


def test_run_lock_has_process_os_and_foreign_thread_owner_guards(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    peer, _, _ = _store(tmp_path, manifest=manifest)
    with store.acquire_run_lock():
        store.publish_manifest()
        with pytest.raises(PilotLocalStoreLockError, match="process_lock"):
            peer.acquire_run_lock()

        failures: list[BaseException] = []

        def foreign_thread() -> None:
            try:
                store.scan_inventory()
            except BaseException as exc:  # assertion captures the exact fail-closed edge
                failures.append(exc)

        thread = threading.Thread(target=foreign_thread)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], PilotLocalStoreLockError)
        assert "foreign_thread" in str(failures[0])

        script = r"""
import sys
from pathlib import Path
from trading.market_data.mexc_pilot_run import parse_pilot_run_manifest_v1
from trading.market_data.mexc_pilot_local_store import (
    PILOT_LOCAL_STORE_CONTRACT_VERSION, MexcPilotLocalStoreV1,
    PilotClockSampleV1, PilotDetachedEvidenceReceiptV1,
    PilotLocalStoreLockError, PilotRuntimeAuthorityBindingV1,
    mexc_pilot_local_store_contract_hash,
)
from trading.market_data.mexc_pilot_local_executor import PilotExecutorBindingsV1
class Clock:
    contract_version = "fixture_clock_v1"
    contract_hash = "7" * 64
    clock_domain_id = "fixture_clock_domain"
    def sample(self):
        return PilotClockSampleV1(1900000000100010, 1, self.clock_domain_id)
class Sink:
    contract_version = "fixture_detached_sink_v1"
    contract_hash = "8" * 64
    domain_id = "fixture_detached_sink_domain"
    def anchor(self, request):
        return PilotDetachedEvidenceReceiptV1(
            request.request_hash, "a" * 64, request.observed_at_us,
            request.observed_monotonic_us, request.clock_domain_id,
        )
class Verifier:
    contract_version = "fixture_u5_verifier_v1"
    contract_hash = "9" * 64
    domain_id = "fixture_u5_verifier_domain"
    trust_key_id = "fixture_u5_trust_key"
    policy_version = "fixture_u5_policy_v1"
    policy_hash = "a" * 64
    def verify(self, request):
        raise AssertionError("child_lock_test_must_not_verify_u5")
runtime = PilotRuntimeAuthorityBindingV1(
    coordinator_implementation_contract_version="fixture_coordinator_impl_v1",
    coordinator_implementation_contract_hash="d" * 64,
    clock_contract_version=Clock.contract_version,
    clock_contract_hash=Clock.contract_hash,
    clock_domain_id=Clock.clock_domain_id,
    detached_anchor_sink_contract_version=Sink.contract_version,
    detached_anchor_sink_contract_hash=Sink.contract_hash,
    detached_anchor_sink_domain_id=Sink.domain_id,
    u5_verifier_contract_version=Verifier.contract_version,
    u5_verifier_contract_hash=Verifier.contract_hash,
    u5_verifier_domain_id=Verifier.domain_id,
    u5_verifier_trust_key_id=Verifier.trust_key_id,
    u5_verifier_policy_version=Verifier.policy_version,
    u5_verifier_policy_hash=Verifier.policy_hash,
)
bindings = PilotExecutorBindingsV1(
    coordinator_contract_version=runtime.contract_version,
    coordinator_contract_hash=runtime.binding_hash,
    local_store_contract_version=PILOT_LOCAL_STORE_CONTRACT_VERSION,
    local_store_contract_hash=mexc_pilot_local_store_contract_hash(),
    clock_contract_version=Clock.contract_version,
    clock_contract_hash=Clock.contract_hash,
    detached_anchor_sink_contract_version=Sink.contract_version,
    detached_anchor_sink_contract_hash=Sink.contract_hash,
    endpoint_runner_contract_version="fixture_endpoint_runner_v1",
    endpoint_runner_contract_hash="b" * 64,
    shard_runner_contract_version="fixture_shard_runner_v1",
    shard_runner_contract_hash="c" * 64,
)
manifest = parse_pilot_run_manifest_v1(Path(sys.argv[1]).read_bytes())
store = MexcPilotLocalStoreV1(
    manifest=manifest, executor_bindings=bindings,
    runtime_authority_binding=runtime, output_root=sys.argv[2],
    detached_evidence_sink=Sink(), clock=Clock(),
    publisher_instance_id="child_publisher", u5_authority_verifier=Verifier(),
)
try:
    lease = store.acquire_run_lock()
except PilotLocalStoreLockError:
    raise SystemExit(0)
else:
    lease.close()
    raise SystemExit(7)
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                os.fspath(store.output_root / "run-control" / "manifest.json"),
                os.fspath(store.output_root),
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_precreated_lock_hardlink_or_symlink_is_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    hardlink_store, _, _ = _store(tmp_path / "hardlink")
    hardlink_store._run_lock_path.parent.mkdir(parents=True)
    source = tmp_path / "hardlink" / "outside-lock-source.bin"
    source.write_bytes(b"sentinel")
    os.link(source, hardlink_store._run_lock_path)
    with pytest.raises(PilotLocalStoreLockError, match="aliased"):
        hardlink_store.acquire_run_lock()
    assert source.read_bytes() == b"sentinel"

    symlink_store, _, _ = _store(tmp_path / "symlink")
    symlink_store._run_lock_path.parent.mkdir(parents=True)
    symlink_source = tmp_path / "symlink" / "outside-lock-source.bin"
    symlink_source.write_bytes(b"sentinel")
    try:
        symlink_store._run_lock_path.symlink_to(symlink_source)
    except OSError as exc:
        pytest.skip(f"host does not allow test symlink creation: {exc}")
    with pytest.raises(PilotLocalStoreLockError, match="aliased"):
        symlink_store.acquire_run_lock()
    assert symlink_source.read_bytes() == b"sentinel"


def test_live_lock_leaf_and_output_root_file_id_are_rechecked_each_mutator(
    tmp_path: Path,
) -> None:
    lock_store, _, _ = _store(tmp_path / "lock")
    with lock_store.acquire_run_lock():
        alias = tmp_path / "lock" / "live-lock-alias.bin"
        os.link(lock_store._run_lock_path, alias)
        try:
            with pytest.raises(PilotLocalStoreLockError, match="aliased"):
                lock_store.publish_manifest()
        finally:
            alias.unlink()
        lock_store.publish_manifest()

    root_store, _, _ = _store(tmp_path / "root")
    with root_store.acquire_run_lock():
        root_store.publish_manifest()
        original = root_store.output_root
        moved = original.with_name("subject-moved")
        original.rename(moved)
        original.mkdir()
        try:
            with pytest.raises(PilotLocalStoreLockError, match="identity_changed"):
                root_store.scan_inventory()
        finally:
            original.rmdir()
            moved.rename(original)


@pytest.mark.skipif(os.name != "nt", reason="Windows 8.3 alias regression")
def test_windows_short_name_alias_is_rejected_before_lock_namespace_split(
    tmp_path: Path,
) -> None:
    import ctypes
    from ctypes import wintypes

    store, manifest, _ = _store(tmp_path)
    with store.acquire_run_lock():
        store.publish_manifest()
    buffer = ctypes.create_unicode_buffer(32768)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_short = kernel32.GetShortPathNameW
    get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short.restype = wintypes.DWORD
    length = get_short(os.fspath(store.output_root), buffer, len(buffer))
    if not length or os.path.normcase(buffer.value) == os.path.normcase(
        os.fspath(store.output_root)
    ):
        pytest.skip("8.3 short-name alias unavailable on this volume")
    runtime = _runtime_binding()
    bindings = _executor_bindings(runtime)
    with pytest.raises(PilotLocalStoreError, match="alias_is_forbidden"):
        MexcPilotLocalStoreV1(
            manifest=manifest,
            executor_bindings=bindings,
            runtime_authority_binding=runtime,
            output_root=buffer.value,
            detached_evidence_sink=FakeDetachedSink(),
            clock=FakeClock(),
            publisher_instance_id="fixture_publisher_instance",
            u5_authority_verifier=FakeU5Verifier(),
        )


def test_intent_capability_is_live_nonserializable_and_consumed_before_callback(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    with store.acquire_run_lock():
        state, authorization = _state_ready_for_intent(store, manifest)
        session = store.claim_process_session(state)
        claimed = store.claim_and_seal_next_intent(state, session)
        assert claimed.state.next_action == "await_endpoint_verification_receipt_no_network_retry"
        assert claimed.intent.stage == "endpoint_verification"
        callbacks: list[str] = []
        result = store.run_owned_intent_once(
            claimed.state,
            claimed.owner_capability,
            FakeOwnedIntentRunner(
                lambda intent: callbacks.append(intent.intent_hash)
                or intent.intent_hash
            ),
        )
        assert result == claimed.intent.intent_hash
        assert callbacks == [claimed.intent.intent_hash]
        with pytest.raises(PilotLocalStoreLockError, match="capability_mismatch"):
            store.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                FakeOwnedIntentRunner(
                    lambda intent: callbacks.append(intent.intent_hash)
                ),
            )
        assert callbacks == [claimed.intent.intent_hash]
        claimed.owner_capability._consumed = False
        claimed.owner_capability._terminal = False
        store._active_intent_capability = claimed.owner_capability
        with pytest.raises(PilotLocalStoreLockError, match="capability_mismatch"):
            store.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                FakeOwnedIntentRunner(
                    lambda intent: callbacks.append(intent.intent_hash)
                ),
            )
        assert callbacks == [claimed.intent.intent_hash]
        with pytest.raises(TypeError, match="not_serializable"):
            pickle.dumps(session)
        with pytest.raises(TypeError, match="not_serializable"):
            pickle.dumps(claimed.owner_capability)

    with pytest.raises(PilotLocalStoreLockError):
        store.run_owned_intent_once(
            claimed.state,
            claimed.owner_capability,
            FakeOwnedIntentRunner(
                lambda intent: callbacks.append(intent.intent_hash)
            ),
        )
    assert callbacks == [claimed.intent.intent_hash]


def test_crash_after_candidate_and_restart_are_zero_permit_stop(
    tmp_path: Path,
) -> None:
    sink = FakeDetachedSink(fail_kind="intent_candidate_publication")
    store, manifest, _ = _store(tmp_path, sink=sink)
    authorization = _authorization(manifest)
    with store.acquire_run_lock():
        state, _ = _state_ready_for_intent(store, manifest)
        session = store.claim_process_session(state)
        before_epoch = store.clock.epoch_us
        before_monotonic = store.clock.monotonic_us
        with pytest.raises(RuntimeError, match="fixture_detached_sink_failure"):
            store.claim_and_seal_next_intent(state, session)
        candidates = list(
            (store.output_root / "run-control" / "network-intents").glob(
                "*.candidate.json"
            )
        )
        assert len(candidates) == 1
        assert not list(candidates[0].parent.glob("*.sealed.json"))
        # Reset the deterministic clock to reproduce byte-identical candidate
        # content.  A pre-existing identical slot is still a loser.
        assert isinstance(store.clock, FakeClock)
        store.clock.epoch_us = before_epoch
        store.clock.monotonic_us = before_monotonic
        sink.fail_kind = None
        callbacks: list[str] = []
        with pytest.raises(PilotLocalStoreRecoveryError, match="unresolved_network_intent"):
            store.claim_and_seal_next_intent(state, session)
        assert callbacks == []

    restarted, _, _ = _store(tmp_path, manifest=manifest, clock=FakeClock())
    with restarted.acquire_run_lock():
        report = restarted.reconstruct_authoritative_state()
        assert report.status == "stopped_no_network"
        assert report.stop_code == "unresolved_network_intent_after_restart"
        assert report.network_permitted is False
        assert report.restart_detected is True
        assert report.state.next_action == "stopped"


def test_callback_gate_rechecks_u5_remaining_window_before_consumption(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    store, manifest, _ = _store(tmp_path, clock=clock)
    authorization = _authorization(manifest)
    callbacks: list[str] = []
    with store.acquire_run_lock():
        state = store.publish_manifest()
        state = store.publish_authorization(authorization)
        state = store.measure_and_publish_preflight(state)
        session = store.claim_process_session(state)
        claimed = store.claim_and_seal_next_intent(state, session)
        # Equality is outside the frozen strict '< expiry' boundary.  The
        # authorization check executes before capability consumption/callback.
        clock.epoch_us = authorization.expires_at_us
        with pytest.raises(PilotRunAuthorizationError, match="not_current"):
            store.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                FakeOwnedIntentRunner(
                    lambda intent: callbacks.append(intent.intent_hash)
                ),
            )
        assert callbacks == []
        assert claimed.owner_capability._consumed is False


def test_bare_or_wrong_runner_identity_never_consumes_or_calls(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    callbacks: list[str] = []
    with store.acquire_run_lock():
        state, _ = _state_ready_for_intent(store, manifest)
        session = store.claim_process_session(state)
        claimed = store.claim_and_seal_next_intent(state, session)
        with pytest.raises(PilotLocalStoreError, match="runner_binding_mismatch"):
            store.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                FakeOwnedIntentRunner(
                    lambda intent: callbacks.append(intent.intent_hash),
                    shard=True,
                ),
            )
        with pytest.raises(PilotLocalStoreError, match="runner_binding_mismatch"):
            store.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                lambda intent: callbacks.append(intent.intent_hash),  # type: ignore[arg-type]
            )
        assert callbacks == []
        assert claimed.owner_capability._consumed is False
        store.run_owned_intent_once(
            claimed.state,
            claimed.owner_capability,
            FakeOwnedIntentRunner(
                lambda intent: callbacks.append(intent.intent_hash)
            ),
        )
        assert callbacks == [claimed.intent.intent_hash]


def test_restart_after_sealed_intent_cannot_recreate_owner_capability(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    authorization = _authorization(manifest)
    with store.acquire_run_lock():
        state, _ = _state_ready_for_intent(store, manifest)
        session = store.claim_process_session(state)
        claimed = store.claim_and_seal_next_intent(state, session)
        assert isinstance(claimed.intent, PilotNetworkIntentV1)

    restarted, _, _ = _store(tmp_path, manifest=manifest)
    with restarted.acquire_run_lock():
        report = restarted.reconstruct_authoritative_state()
        assert report.status == "stopped_no_network"
        assert report.stop_code == "unresolved_network_intent_after_restart"
        assert report.network_permitted is False
        assert report.restart_detected is True
        callbacks: list[str] = []
        with pytest.raises(PilotLocalStoreLockError):
            restarted.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                FakeOwnedIntentRunner(
                    lambda intent: callbacks.append(intent.intent_hash)
                ),
            )
        assert callbacks == []


def test_callback_gate_uses_post_probe_time_not_pre_probe_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    store, manifest, _ = _store(tmp_path, clock=clock)
    authorization = _authorization(manifest)
    callbacks: list[str] = []
    with store.acquire_run_lock():
        state, _ = _state_ready_for_intent(store, manifest)
        session = store.claim_process_session(state)
        claimed = store.claim_and_seal_next_intent(state, session)
        original_probe = MexcPilotLocalStoreV1._probe_hardlink_create_new

        def probe_that_crosses_expiry(selected: MexcPilotLocalStoreV1) -> None:
            original_probe(selected)
            clock.epoch_us = authorization.expires_at_us
            clock.monotonic_us += 1

        monkeypatch.setattr(
            MexcPilotLocalStoreV1,
            "_probe_hardlink_create_new",
            probe_that_crosses_expiry,
        )
        with pytest.raises(PilotRunAuthorizationError, match="not_current"):
            store.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                FakeOwnedIntentRunner(
                    lambda intent: callbacks.append(intent.intent_hash)
                ),
            )
        assert callbacks == []
        assert claimed.owner_capability._consumed is False


def test_runner_identity_getters_run_before_final_u5_gate(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    store, manifest, _ = _store(tmp_path, clock=clock)
    authorization = _authorization(manifest)
    callbacks: list[str] = []

    class AdvancingRunner:
        @property
        def contract_version(self) -> str:
            clock.epoch_us = authorization.expires_at_us
            clock.monotonic_us += 1
            return ENDPOINT_RUNNER_VERSION

        @property
        def contract_hash(self) -> str:
            return ENDPOINT_RUNNER_HASH

        def __call__(self, intent: PilotNetworkIntentV1) -> None:
            callbacks.append(intent.intent_hash)

    with store.acquire_run_lock():
        state, _ = _state_ready_for_intent(store, manifest)
        session = store.claim_process_session(state)
        claimed = store.claim_and_seal_next_intent(state, session)
        with pytest.raises(PilotRunAuthorizationError, match="not_current"):
            store.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                AdvancingRunner(),
            )
        assert callbacks == []
        assert claimed.owner_capability._consumed is False


def test_runner_identity_getter_cannot_swap_bound_output_root(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    callbacks: list[str] = []
    original = store.output_root
    moved = original.with_name("subject-moved-by-runner")

    class SwappingRunner:
        swapped = False

        @property
        def contract_version(self) -> str:
            if not self.swapped:
                original.rename(moved)
                original.mkdir()
                self.swapped = True
            return ENDPOINT_RUNNER_VERSION

        @property
        def contract_hash(self) -> str:
            return ENDPOINT_RUNNER_HASH

        def __call__(self, intent: PilotNetworkIntentV1) -> None:
            callbacks.append(intent.intent_hash)

    with store.acquire_run_lock():
        state, _ = _state_ready_for_intent(store, manifest)
        session = store.claim_process_session(state)
        claimed = store.claim_and_seal_next_intent(state, session)
        try:
            with pytest.raises(PilotLocalStoreLockError, match="identity_changed"):
                store.run_owned_intent_once(
                    claimed.state,
                    claimed.owner_capability,
                    SwappingRunner(),
                )
            assert callbacks == []
            assert claimed.owner_capability._consumed is False
        finally:
            original.rmdir()
            moved.rename(original)


def test_final_clock_sample_cannot_swap_bound_output_root(
    tmp_path: Path,
) -> None:
    original = tmp_path / "subject"
    moved = tmp_path / "subject-moved-by-clock"

    class SwappingClock(FakeClock):
        samples_until_swap: int | None = None

        def sample(self) -> PilotClockSampleV1:
            if self.samples_until_swap is not None:
                self.samples_until_swap -= 1
                if self.samples_until_swap == 0:
                    original.rename(moved)
                    original.mkdir()
                    self.samples_until_swap = None
            return super().sample()

    clock = SwappingClock()
    store, manifest, _ = _store(tmp_path, clock=clock)
    callbacks: list[str] = []
    with store.acquire_run_lock():
        state, _ = _state_ready_for_intent(store, manifest)
        session = store.claim_process_session(state)
        claimed = store.claim_and_seal_next_intent(state, session)
        clock.samples_until_swap = 3
        try:
            with pytest.raises(PilotLocalStoreLockError, match="identity_changed"):
                store.run_owned_intent_once(
                    claimed.state,
                    claimed.owner_capability,
                    FakeOwnedIntentRunner(
                        lambda intent: callbacks.append(intent.intent_hash)
                    ),
                )
            assert callbacks == []
            assert claimed.owner_capability._consumed is False
        finally:
            original.rmdir()
            moved.rename(original)


@pytest.mark.parametrize("boundary", ("authorization", "preflight"))
def test_slow_final_replay_cannot_use_stale_pre_replay_gate_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    clock = FakeClock()
    store, manifest, _ = _store(tmp_path, clock=clock)
    authorization = _authorization(manifest)
    callbacks: list[str] = []
    with store.acquire_run_lock():
        state, _ = _state_ready_for_intent(store, manifest)
        session = store.claim_process_session(state)
        claimed = store.claim_and_seal_next_intent(state, session)
        original_require = MexcPilotLocalStoreV1._require_expected_state
        replays = 0

        def replay_that_crosses_expiry(
            selected: MexcPilotLocalStoreV1,
            expected: PilotRunStateV1,
            *,
            allow_active_intent: bool = False,
            recheck_dependencies: bool = True,
        ):
            nonlocal replays
            replays += 1
            result = original_require(
                selected,
                expected,
                allow_active_intent=allow_active_intent,
                recheck_dependencies=recheck_dependencies,
            )
            if replays == 2:
                clock.epoch_us = (
                    authorization.expires_at_us
                    if boundary == "authorization"
                    else claimed.state.preflight_receipts[-1].valid_until_us
                )
                clock.monotonic_us += 1
            return result

        monkeypatch.setattr(
            MexcPilotLocalStoreV1,
            "_require_expected_state",
            replay_that_crosses_expiry,
        )
        expected_error = (
            PilotRunAuthorizationError
            if boundary == "authorization"
            else PilotRunPreflightError
        )
        with pytest.raises(expected_error, match="not_current"):
            store.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                FakeOwnedIntentRunner(
                    lambda intent: callbacks.append(intent.intent_hash)
                ),
            )
        assert replays == 2
        assert callbacks == []
        assert claimed.owner_capability._consumed is False


def test_runner_getter_cannot_delete_sealed_intent_or_live_session_claim(
    tmp_path: Path,
) -> None:
    for delete_kind in ("sealed", "session"):
        store, manifest, _ = _store(tmp_path / delete_kind)
        callbacks: list[str] = []
        with store.acquire_run_lock():
            state, _ = _state_ready_for_intent(store, manifest)
            session = store.claim_process_session(state)
            claimed = store.claim_and_seal_next_intent(state, session)

            class DeletingRunner:
                deleted = False

                @property
                def contract_version(self) -> str:
                    if not self.deleted:
                        if delete_kind == "sealed":
                            target = store.output_root.joinpath(
                                *claimed.intent.sealed_intent_locator.split("/")
                            )
                        else:
                            target = store.external_state_root.joinpath(
                                *session.session_claim_locator.split("/")
                            )
                        target.unlink()
                        self.deleted = True
                    return ENDPOINT_RUNNER_VERSION

                @property
                def contract_hash(self) -> str:
                    return ENDPOINT_RUNNER_HASH

                def __call__(self, intent: PilotNetworkIntentV1) -> None:
                    callbacks.append(intent.intent_hash)

            with pytest.raises(PilotLocalStoreError):
                store.run_owned_intent_once(
                    claimed.state,
                    claimed.owner_capability,
                    DeletingRunner(),
                )
            assert callbacks == []
            assert claimed.owner_capability._consumed is False


@pytest.mark.skipif(os.name != "nt", reason="Windows dangling-junction regression")
def test_dangling_junction_in_fresh_root_blocks_callback(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    callbacks: list[str] = []
    with store.acquire_run_lock():
        state, _ = _state_ready_for_intent(store, manifest)
        session = store.claim_process_session(state)
        claimed = store.claim_and_seal_next_intent(state, session)
        relative_root = manifest.remaining_fresh_roots(-1)[0]
        junction = store.output_root.joinpath(*relative_root.split("/"))
        junction.parent.mkdir(parents=True, exist_ok=True)
        missing_target = tmp_path / "missing-junction-target"
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", os.fspath(junction), os.fspath(missing_target)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip(f"host cannot create junction: {completed.stderr}")
        with pytest.raises(PilotLocalStoreError, match="fresh_root_preexists"):
            store.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                FakeOwnedIntentRunner(
                    lambda intent: callbacks.append(intent.intent_hash)
                ),
            )
        assert callbacks == []
        assert claimed.owner_capability._consumed is False


def test_authoritative_reconstruction_uses_fixed_prefix_and_rejects_state_json(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    with store.acquire_run_lock():
        state = store.publish_manifest()
        authorization = _authorization(manifest)
        state = store.publish_authorization(authorization)
        state = store.measure_and_publish_preflight(state)
        clean = store.reconstruct_authoritative_state()
        assert clean.state == state
        assert clean.status == "reconstructed_no_network"
        assert clean.network_permitted is False

        # A materialized state snapshot is never an authoritative replay source.
        (store.output_root / "run-control" / "state.json").write_bytes(
            canonical_lf_bytes(
                {
                    "fabricated": "complete",
                    "network_permitted": True,
                }
            )
        )
        stopped = store.reconstruct_authoritative_state()
        assert stopped.status == "stopped_no_network"
        assert stopped.stop_code == "unexpected_run_control_residue"
        assert stopped.residue_paths == ("run-control/state.json",)
        assert stopped.network_permitted is False


def test_control_replay_rejects_hardlink_alias_and_unexpected_empty_directory(
    tmp_path: Path,
) -> None:
    hardlink_store, manifest, _ = _store(tmp_path / "hardlink")
    with hardlink_store.acquire_run_lock():
        hardlink_store.publish_manifest()
        outside = tmp_path / "hardlink" / "outside-authorization.json"
        outside.write_bytes(canonical_lf_bytes(_authorization(manifest).as_dict()))
        os.link(
            outside,
            hardlink_store.output_root / "run-control" / "authorization.json",
        )
        with pytest.raises(PilotLocalStoreRecoveryError, match="hardlink_alias"):
            hardlink_store.reconstruct_authoritative_state()

    directory_store, _, _ = _store(tmp_path / "directory")
    with directory_store.acquire_run_lock():
        directory_store.publish_manifest()
        (directory_store.output_root / "run-control" / "state.json").mkdir()
        report = directory_store.reconstruct_authoritative_state()
        assert report.status == "stopped_no_network"
        assert report.stop_code == "unexpected_run_control_residue"
        assert report.residue_paths == ("run-control/state.json/",)


def test_inventory_is_bounded_non_reparse_and_rejects_hardlink_alias(
    tmp_path: Path,
) -> None:
    store, _, _ = _store(tmp_path)
    with store.acquire_run_lock():
        store.publish_manifest()
        scan = store.scan_inventory()
        assert scan.total_bytes > 0
        assert scan.entries[0].relative_path == "run-control/manifest.json"
        assert len(scan.inventory_hash) == 64
        with pytest.raises(PilotLocalStoreBoundsError, match="entry_bound"):
            store.scan_inventory(max_entries=0)
        with pytest.raises(PilotLocalStoreBoundsError, match="byte_budget"):
            store.scan_inventory(max_bytes=1)
        with pytest.raises(PilotLocalStoreBoundsError, match="exceeds_manifest"):
            store.scan_inventory(
                max_entries=store.manifest.budgets.max_inventory_entries + 1
            )
        with pytest.raises(PilotLocalStoreBoundsError, match="exceeds_manifest"):
            store.scan_inventory(
                max_bytes=store.manifest.budgets.max_total_output_bytes + 1
            )

        source = store.output_root / "run-control" / "manifest.json"
        alias = store.output_root / "manifest-hardlink-alias.json"
        try:
            os.link(source, alias)
        except OSError as exc:
            pytest.skip(f"host filesystem does not support hardlinks: {exc}")
        with pytest.raises(PilotLocalStoreError, match="hardlink_alias"):
            store.scan_inventory()


def test_inventory_rejects_symlink_or_windows_reparse_when_host_allows_it(
    tmp_path: Path,
) -> None:
    store, _, _ = _store(tmp_path)
    with store.acquire_run_lock():
        store.publish_manifest()
        target = store.output_root / "run-control" / "manifest.json"
        alias = store.output_root / "manifest-symlink-alias.json"
        try:
            alias.symlink_to(target)
        except OSError as exc:
            pytest.skip(f"host does not allow test symlink creation: {exc}")
        with pytest.raises(PilotLocalStoreError, match="reparse"):
            store.scan_inventory()


@pytest.mark.skipif(os.name != "nt", reason="NTFS ADS regression")
def test_named_ntfs_stream_blocks_permission_and_inventory(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    callbacks: list[str] = []
    with store.acquire_run_lock():
        state, _ = _state_ready_for_intent(store, manifest)
        session = store.claim_process_session(state)
        claimed = store.claim_and_seal_next_intent(state, session)
        manifest_path = store.output_root / "run-control" / "manifest.json"
        try:
            with open(os.fspath(manifest_path) + ":hidden", "wb") as stream:
                stream.write(b"hidden")
        except OSError as exc:
            pytest.skip(f"host filesystem does not support NTFS ADS: {exc}")
        with pytest.raises(PilotLocalStoreError, match="named_stream"):
            store.run_owned_intent_once(
                claimed.state,
                claimed.owner_capability,
                FakeOwnedIntentRunner(
                    lambda intent: callbacks.append(intent.intent_hash)
                ),
            )
        assert callbacks == []
        assert claimed.owner_capability._consumed is False
        with pytest.raises(PilotLocalStoreError, match="named_stream"):
            store.scan_inventory()


@pytest.mark.skipif(os.name != "nt", reason="NTFS late ADS regression")
@pytest.mark.parametrize("operation", ("inventory", "replay", "callback"))
def test_late_ntfs_stream_after_clean_enumeration_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    store, manifest, _ = _store(tmp_path)
    callbacks: list[str] = []
    with store.acquire_run_lock():
        if operation == "callback":
            state, _ = _state_ready_for_intent(store, manifest)
            session = store.claim_process_session(state)
            claimed = store.claim_and_seal_next_intent(state, session)
        else:
            store.publish_manifest()

        manifest_path = store.output_root / "run-control" / "manifest.json"
        probe = store.output_root / "ads-capability-probe.bin"
        probe.write_bytes(b"probe")
        try:
            with open(os.fspath(probe) + ":probe", "wb") as stream:
                stream.write(b"probe")
        except OSError as exc:
            pytest.skip(f"host filesystem does not support NTFS ADS: {exc}")
        finally:
            probe.unlink(missing_ok=True)

        before = manifest_path.stat()
        original_reject = pilot_store_module._reject_windows_named_streams
        injected = False

        def reject_then_inject(
            path: Path,
            *,
            deadline_ns: int | None,
        ) -> int:
            nonlocal injected
            result = original_reject(path, deadline_ns=deadline_ns)
            if not injected and os.path.normcase(os.path.abspath(path)) == (
                os.path.normcase(os.path.abspath(manifest_path))
            ):
                with open(os.fspath(manifest_path) + ":late", "wb") as stream:
                    stream.write(b"late-hidden-stream")
                os.utime(
                    manifest_path,
                    ns=(before.st_atime_ns, before.st_mtime_ns),
                )
                injected = True
            return result

        monkeypatch.setattr(
            pilot_store_module,
            "_reject_windows_named_streams",
            reject_then_inject,
        )
        with pytest.raises(PilotLocalStoreError, match="named_stream"):
            if operation == "inventory":
                store.scan_inventory()
            elif operation == "replay":
                store.reconstruct_authoritative_state()
            else:
                store.run_owned_intent_once(
                    claimed.state,
                    claimed.owner_capability,
                    FakeOwnedIntentRunner(
                        lambda intent: callbacks.append(intent.intent_hash)
                    ),
                )
        assert injected is True
        assert callbacks == []
        if operation == "callback":
            assert claimed.owner_capability._consumed is False


@pytest.mark.parametrize("operation", ("inventory", "callback"))
def test_leaf_replacement_after_post_stream_check_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    store, manifest, _ = _store(tmp_path)
    callbacks: list[str] = []
    with store.acquire_run_lock():
        if operation == "callback":
            state, _ = _state_ready_for_intent(store, manifest)
            session = store.claim_process_session(state)
            claimed = store.claim_and_seal_next_intent(state, session)
        else:
            store.publish_manifest()

        manifest_path = store.output_root / "run-control" / "manifest.json"
        observed = manifest_path.stat()
        replacement = tmp_path / "replacement-manifest.json"
        replacement.write_bytes(manifest_path.read_bytes())
        os.utime(
            replacement,
            ns=(observed.st_atime_ns, observed.st_mtime_ns),
        )
        displaced = tmp_path / "displaced-manifest.json"
        original_reject = pilot_store_module._reject_windows_named_streams
        target_checks = 0
        replaced = False

        def reject_then_replace(
            path: Path,
            *,
            deadline_ns: int | None,
        ) -> int:
            nonlocal replaced, target_checks
            result = original_reject(path, deadline_ns=deadline_ns)
            if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
                os.path.abspath(manifest_path)
            ):
                target_checks += 1
                if target_checks == 2:
                    manifest_path.replace(displaced)
                    replacement.replace(manifest_path)
                    os.utime(
                        manifest_path,
                        ns=(observed.st_atime_ns, observed.st_mtime_ns),
                    )
                    replaced = True
            return result

        monkeypatch.setattr(
            pilot_store_module,
            "_reject_windows_named_streams",
            reject_then_replace,
        )
        with pytest.raises(PilotLocalStoreError, match="changed_after_stream"):
            if operation == "inventory":
                store.scan_inventory()
            else:
                store.run_owned_intent_once(
                    claimed.state,
                    claimed.owner_capability,
                    FakeOwnedIntentRunner(
                        lambda intent: callbacks.append(intent.intent_hash)
                    ),
                )
        assert target_checks == 2
        assert replaced is True
        assert callbacks == []
        if operation == "callback":
            assert claimed.owner_capability._consumed is False


@pytest.mark.parametrize("operation", ("inventory", "callback"))
def test_terminal_tree_pass_detects_late_plain_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    store, manifest, _ = _store(tmp_path)
    callbacks: list[str] = []
    with store.acquire_run_lock():
        if operation == "callback":
            state, _ = _state_ready_for_intent(store, manifest)
            session = store.claim_process_session(state)
            claimed = store.claim_and_seal_next_intent(state, session)
        else:
            store.publish_manifest()

        original_snapshot = MexcPilotLocalStoreV1._scan_tree_identity_snapshot
        late_path = store.output_root / "run-control" / "late-ordinary.json"
        injected = False

        def snapshot_after_late_entry(
            selected: MexcPilotLocalStoreV1,
            **kwargs,
        ):
            nonlocal injected
            if not injected:
                late_path.write_bytes(b"{}\n")
                injected = True
            return original_snapshot(selected, **kwargs)

        monkeypatch.setattr(
            MexcPilotLocalStoreV1,
            "_scan_tree_identity_snapshot",
            snapshot_after_late_entry,
        )
        with pytest.raises(PilotLocalStoreError, match="tree_changed"):
            if operation == "inventory":
                store.scan_inventory()
            else:
                store.run_owned_intent_once(
                    claimed.state,
                    claimed.owner_capability,
                    FakeOwnedIntentRunner(
                        lambda intent: callbacks.append(intent.intent_hash)
                    ),
                )
        assert injected is True
        assert callbacks == []
        if operation == "callback":
            assert claimed.owner_capability._consumed is False


def test_post_stream_and_terminal_passes_use_separate_bounded_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, _ = _store(tmp_path)
    with store.acquire_run_lock():
        store.publish_manifest()
        monkeypatch.setattr(
            pilot_store_module,
            "_reject_windows_named_streams",
            lambda path, *, deadline_ns: 1,
        )
        # The main pass and the terminal verification pass each visit exactly
        # two locators (run-control and manifest).  The legal default stream
        # record and the second-pass counter are rejection work, not extra
        # semantic inventory entries.
        scan = store.scan_inventory(max_entries=2)
        assert [entry.relative_path for entry in scan.entries] == [
            "run-control/manifest.json"
        ]


@pytest.mark.skipif(os.name != "nt", reason="Windows junction-swap regression")
@pytest.mark.parametrize("operation", ("inventory", "callback"))
def test_queued_plain_directory_swapped_once_to_junction_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    store, manifest, _ = _store(tmp_path)
    callbacks: list[str] = []
    with store.acquire_run_lock():
        if operation == "callback":
            state, _ = _state_ready_for_intent(store, manifest)
            session = store.claim_process_session(state)
            claimed = store.claim_and_seal_next_intent(state, session)
            queued = store.output_root / "run-control" / "network-intents"
        else:
            store.publish_manifest()
            queued = store.output_root / "queued-directory"
            queued.mkdir()
            (queued / "original.bin").write_bytes(b"original")

        outside = tmp_path / "outside-junction-target"
        outside.mkdir()
        (outside / "foreign.bin").write_bytes(b"foreign")
        probe_target = tmp_path / "junction-probe-target"
        probe_target.mkdir()
        probe_link = tmp_path / "junction-probe-link"
        probe_result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", os.fspath(probe_link), os.fspath(probe_target)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if probe_result.returncode != 0:
            pytest.skip(f"host cannot create test junction: {probe_result.stderr}")
        os.rmdir(probe_link)

        backup = tmp_path / "queued-directory-backup"
        original_scandir = pilot_store_module.os.scandir
        swapped = False

        def same_path(left: Path, right: Path) -> bool:
            return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
                os.path.abspath(right)
            )

        class RestoringScandir:
            def __init__(self, iterator) -> None:
                self.iterator = iterator
                self.restored = False

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.iterator)

            def __enter__(self):
                return self

            def restore(self) -> None:
                if self.restored:
                    return
                self.iterator.close()
                os.rmdir(queued)
                backup.rename(queued)
                self.restored = True

            def __exit__(self, exc_type, exc, traceback) -> None:
                self.restore()

        def scandir_with_one_junction_swap(path):
            nonlocal swapped
            selected = Path(path)
            if not swapped and same_path(selected, queued):
                queued.rename(backup)
                result = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", os.fspath(queued), os.fspath(outside)],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                if result.returncode != 0:
                    backup.rename(queued)
                    pytest.skip(f"host junction swap failed: {result.stderr}")
                swapped = True
                try:
                    return RestoringScandir(original_scandir(queued))
                except BaseException:
                    os.rmdir(queued)
                    backup.rename(queued)
                    raise
            return original_scandir(path)

        monkeypatch.setattr(
            pilot_store_module.os,
            "scandir",
            scandir_with_one_junction_swap,
        )
        with pytest.raises(PilotLocalStoreError, match="reparse|tree_changed"):
            if operation == "inventory":
                store.scan_inventory()
            else:
                store.run_owned_intent_once(
                    claimed.state,
                    claimed.owner_capability,
                    FakeOwnedIntentRunner(
                        lambda intent: callbacks.append(intent.intent_hash)
                    ),
                )
        assert swapped is True
        assert queued.is_dir()
        assert not backup.exists()
        assert callbacks == []
        if operation == "callback":
            assert claimed.owner_capability._consumed is False


def test_inventory_deadline_is_checked_immediately_after_each_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, _ = _store(tmp_path)
    with store.acquire_run_lock():
        store.publish_manifest()
        original_read = pilot_store_module.os.read
        read_returned = False

        def observed_read(descriptor: int, size: int) -> bytes:
            nonlocal read_returned
            result = original_read(descriptor, size)
            read_returned = True
            return result

        def deadline_clock() -> int:
            return 10**18 if read_returned else 0

        monkeypatch.setattr(pilot_store_module.os, "read", observed_read)
        monkeypatch.setattr(
            pilot_store_module.time,
            "monotonic_ns",
            deadline_clock,
        )
        with pytest.raises(PilotLocalStoreBoundsError, match="scan_runtime"):
            store.scan_inventory()


def test_inventory_deadline_is_checked_after_result_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled = False

    class MarkingClock(FakeClock):
        def sample(self) -> PilotClockSampleV1:
            nonlocal sampled
            sampled = True
            return super().sample()

    store, _, _ = _store(tmp_path, clock=MarkingClock())
    with store.acquire_run_lock():
        store.publish_manifest()

        def deadline_clock() -> int:
            return 10**18 if sampled else 0

        monkeypatch.setattr(
            pilot_store_module.time,
            "monotonic_ns",
            deadline_clock,
        )
        with pytest.raises(PilotLocalStoreBoundsError, match="scan_runtime"):
            store.scan_inventory()


def test_inventory_and_control_scans_bound_directories_and_depth(
    tmp_path: Path,
) -> None:
    inventory_store, _, _ = _store(tmp_path / "inventory")
    with inventory_store.acquire_run_lock():
        inventory_store.publish_manifest()
        flood = inventory_store.output_root / "directory-flood"
        for ordinal in range(4):
            (flood / f"d{ordinal}").mkdir(parents=True)
        with pytest.raises(PilotLocalStoreBoundsError, match="visited_entry"):
            inventory_store.scan_inventory(max_entries=3)

    depth_store, _, _ = _store(tmp_path / "depth")
    with depth_store.acquire_run_lock():
        depth_store.publish_manifest()
        current = depth_store.output_root / "deep"
        for _ in range(66):
            current = current / "d"
        current.mkdir(parents=True)
        with pytest.raises(PilotLocalStoreBoundsError, match="depth_budget"):
            depth_store.scan_inventory()

    control_store, _, _ = _store(tmp_path / "control")
    with control_store.acquire_run_lock():
        control_store.publish_manifest()
        flood = control_store.output_root / "run-control" / "directory-flood"
        for ordinal in range(control_store.manifest.budgets.max_inventory_entries + 1):
            (flood / f"d{ordinal}").mkdir(parents=True)
        with pytest.raises(PilotLocalStoreBoundsError, match="visited_entry"):
            control_store.reconstruct_authoritative_state()


def test_store_slice_has_no_caller_authoritative_terminal_publishers(
    tmp_path: Path,
) -> None:
    store, _, _ = _store(tmp_path)
    for name in (
        "publish_endpoint_verification",
        "publish_shard_result",
        "publish_step_failure",
        "publish_result_candidate",
    ):
        assert not hasattr(store, name)


def test_output_identity_and_external_state_separation_are_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "subject")
    runtime = _runtime_binding()
    bindings = _executor_bindings(runtime)
    if os.name == "nt":
        with pytest.raises(PilotLocalStoreError, match="locator_mismatch"):
            MexcPilotLocalStoreV1(
                manifest=manifest,
                executor_bindings=bindings,
                runtime_authority_binding=runtime,
                output_root=tmp_path / "different-subject",
                detached_evidence_sink=FakeDetachedSink(),
                clock=FakeClock(),
                publisher_instance_id="fixture_publisher_instance",
                u5_authority_verifier=FakeU5Verifier(),
            )
    with pytest.raises(TypeError, match="external_state_root"):
        MexcPilotLocalStoreV1(
            manifest=manifest,
            executor_bindings=bindings,
            runtime_authority_binding=runtime,
            output_root=tmp_path / "subject",
            external_state_root=tmp_path / "caller-selected-root",  # type: ignore[call-arg]
            detached_evidence_sink=FakeDetachedSink(),
            clock=FakeClock(),
            publisher_instance_id="fixture_publisher_instance",
            u5_authority_verifier=FakeU5Verifier(),
        )
    store, _, _ = _store(tmp_path, manifest=manifest)
    competing_manifest = replace(manifest, repository_tree_receipt_hash="f" * 64)
    competitor, _, _ = _store(tmp_path, manifest=competing_manifest)
    assert store.external_state_root == competitor.external_state_root
    callbacks: list[str] = []
    with store.acquire_run_lock():
        with pytest.raises(PilotLocalStoreLockError, match="process_lock"):
            lease = competitor.acquire_run_lock()
            callbacks.append("entered")
            lease.close()
    assert callbacks == []


def test_constructor_bindings_are_read_only_and_rechecked_on_every_mutation(
    tmp_path: Path,
) -> None:
    store, _, _ = _store(tmp_path)
    other_root = tmp_path / "other-subject"
    with store.acquire_run_lock():
        for name, value in (
            ("output_root", other_root),
            ("external_state_root", tmp_path / "other-state"),
            ("clock", FakeClock()),
            ("detached_evidence_sink", FakeDetachedSink()),
            ("u5_authority_verifier", FakeU5Verifier()),
            ("executor_bindings", _executor_bindings()),
            ("runtime_authority_binding", _runtime_binding()),
            ("_output_root", other_root),
            ("_clock", FakeClock()),
            ("_bound_output_root_identity", (999, 999)),
        ):
            with pytest.raises(AttributeError):
                setattr(store, name, value)

        original_root = store.output_root
        object.__setattr__(store, "_output_root", other_root)
        with pytest.raises(PilotLocalStoreLockError, match="binding_changed"):
            store.publish_manifest()
        assert not (other_root / "run-control" / "manifest.json").exists()
        object.__setattr__(store, "_output_root", original_root)


def test_runtime_executor_manifest_and_dependency_identities_are_exact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "subject"
    runtime = _runtime_binding()
    bindings = _executor_bindings(runtime)
    manifest = _manifest(root, bindings)
    changed_runtime = replace(runtime, clock_contract_hash="f" * 64)
    with pytest.raises(PilotLocalStoreError, match="runtime_binding_mismatch"):
        MexcPilotLocalStoreV1(
            manifest=manifest,
            executor_bindings=bindings,
            runtime_authority_binding=changed_runtime,
            output_root=root,
            detached_evidence_sink=FakeDetachedSink(),
            clock=FakeClock(),
            publisher_instance_id="fixture_publisher_instance",
            u5_authority_verifier=FakeU5Verifier(),
        )

    wrong_clock = FakeClock()
    wrong_clock.contract_hash = "f" * 64
    with pytest.raises(PilotLocalStoreError, match="dependency_identity_mismatch"):
        MexcPilotLocalStoreV1(
            manifest=manifest,
            executor_bindings=bindings,
            runtime_authority_binding=runtime,
            output_root=root,
            detached_evidence_sink=FakeDetachedSink(),
            clock=wrong_clock,
            publisher_instance_id="fixture_publisher_instance",
            u5_authority_verifier=FakeU5Verifier(),
        )

    wrong_endpoint = replace(
        manifest.endpoint_verification,
        verifier_contract_hash="f" * 64,
    )
    with pytest.raises(PilotLocalStoreError, match="manifest_composite_binding_mismatch"):
        MexcPilotLocalStoreV1(
            manifest=replace(manifest, endpoint_verification=wrong_endpoint),
            executor_bindings=bindings,
            runtime_authority_binding=runtime,
            output_root=root,
            detached_evidence_sink=FakeDetachedSink(),
            clock=FakeClock(),
            publisher_instance_id="fixture_publisher_instance",
            u5_authority_verifier=FakeU5Verifier(),
        )


def test_ordinary_publication_is_idempotent_but_conflict_is_rejected(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    with store.acquire_run_lock():
        assert store.publish_manifest().manifest == manifest
        assert store.publish_manifest().manifest == manifest
        manifest_path = store.output_root / "run-control" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["repository_tree_receipt_hash"] = "f" * 64
        manifest_path.write_bytes(canonical_lf_bytes(payload))
        with pytest.raises(PilotLocalStoreConflictError, match="conflict"):
            store.publish_manifest()


def test_caller_state_and_declarative_preflight_cannot_mint_permission(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    authorization = _authorization(manifest)
    with store.acquire_run_lock():
        state = store.publish_manifest()
        state = store.publish_authorization(authorization)
        fabricated = state.with_preflight(
            _preflight(manifest, authorization),
            now_us=EPOCH + 20,
        )
        with pytest.raises(PilotLocalStoreConflictError, match="cas_mismatch"):
            store.claim_process_session(fabricated)
        assert not hasattr(store, "publish_preflight")

        measured = store.measure_and_publish_preflight(state)
        session = store.claim_process_session(measured)
        fresh_root = manifest.remaining_fresh_roots(-1)[0]
        target = store.output_root.joinpath(*fresh_root.split("/"))
        target.mkdir(parents=True)
        with pytest.raises(PilotLocalStoreError, match="fresh_root_preexists"):
            store.claim_and_seal_next_intent(measured, session)


def test_durable_authorization_and_verified_u5_challenge_are_both_required(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path / "missing-durable")
    authorization = _authorization(manifest)
    with store.acquire_run_lock():
        durable = store.publish_manifest()
        fabricated = durable.with_authorization(
            authorization,
            now_us=authorization.authorized_at_us,
        )
        with pytest.raises(PilotLocalStoreConflictError, match="cas_mismatch"):
            store.claim_process_session(fabricated)

    missing_root = tmp_path / "missing-verifier" / "subject"
    runtime = _runtime_binding()
    bindings = _executor_bindings(runtime)
    with pytest.raises(PilotLocalStoreError, match="verifier_is_required"):
        MexcPilotLocalStoreV1(
            manifest=_manifest(missing_root, bindings),
            executor_bindings=bindings,
            runtime_authority_binding=runtime,
            output_root=missing_root,
            detached_evidence_sink=FakeDetachedSink(),
            clock=FakeClock(),
            publisher_instance_id="fixture_publisher_instance",
            u5_authority_verifier=None,  # type: ignore[arg-type]
        )

    corrupt, manifest3, _ = _store(
        tmp_path / "corrupt-verifier",
        verifier=FakeU5Verifier(corrupt_binding=True),
    )
    with corrupt.acquire_run_lock():
        state = corrupt.publish_manifest()
        state = corrupt.publish_authorization(_authorization(manifest3))
        with pytest.raises(PilotLocalStoreError, match="binding_mismatch"):
            corrupt.claim_process_session(state)
        assert not (corrupt.external_state_root / "session-claims").exists()


def test_session_claim_revalidates_u5_after_verifier_and_detached_anchor(
    tmp_path: Path,
) -> None:
    verifier_clock = FakeClock()
    verifier_root = tmp_path / "verifier"
    verifier_manifest = _manifest(verifier_root / "subject")
    verifier_authorization = _authorization(verifier_manifest)

    class AdvancingVerifier(FakeU5Verifier):
        def verify(self, request: PilotU5VerificationRequestV1):
            evidence = super().verify(request)
            verifier_clock.epoch_us = verifier_authorization.expires_at_us
            verifier_clock.monotonic_us += 1
            return evidence

    verifier_store, _, _ = _store(
        verifier_root,
        manifest=verifier_manifest,
        clock=verifier_clock,
        verifier=AdvancingVerifier(),
    )
    with verifier_store.acquire_run_lock():
        state = verifier_store.publish_manifest()
        state = verifier_store.publish_authorization(verifier_authorization)
        with pytest.raises(PilotRunAuthorizationError, match="not_current"):
            verifier_store.claim_process_session(state)
        assert not (verifier_store.external_state_root / "session-claims").exists()

    anchor_clock = FakeClock()
    anchor_root = tmp_path / "anchor"
    anchor_manifest = _manifest(anchor_root / "subject")
    anchor_authorization = _authorization(anchor_manifest)

    def expire_after_session_anchor(request: PilotDetachedEvidenceRequestV1) -> None:
        if request.evidence_kind == "session_claim_reload":
            anchor_clock.epoch_us = anchor_authorization.expires_at_us
            anchor_clock.monotonic_us += 1

    anchor_store, _, _ = _store(
        anchor_root,
        manifest=anchor_manifest,
        clock=anchor_clock,
        sink=FakeDetachedSink(on_anchor=expire_after_session_anchor),
    )
    with anchor_store.acquire_run_lock():
        state = anchor_store.publish_manifest()
        state = anchor_store.publish_authorization(anchor_authorization)
        with pytest.raises(PilotRunAuthorizationError, match="not_current"):
            anchor_store.claim_process_session(state)
        claims = list((anchor_store.external_state_root / "session-claims").glob("*"))
        assert len(claims) == 1
        report = anchor_store.reconstruct_authoritative_state()
        assert report.status == "stopped_no_network"
        assert report.stop_code == "preexisting_process_session_claim_after_restart"
        assert report.restart_detected is True


def test_clean_restart_session_claim_is_persistent_zero_permit(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    with store.acquire_run_lock():
        state = store.publish_manifest()
        state = store.publish_authorization(_authorization(manifest))
        session = store.claim_process_session(state)
        assert session.manifest_hash == manifest.manifest_hash
    restarted, _, _ = _store(tmp_path, manifest=manifest)
    with restarted.acquire_run_lock():
        report = restarted.reconstruct_authoritative_state()
        assert report.restart_detected is True
        assert report.network_permitted is False
        assert report.stop_code == "preexisting_process_session_claim_after_restart"
        with pytest.raises(PilotLocalStoreRecoveryError, match="authoritative_prefix_is_stopped"):
            restarted.claim_process_session(state)


@pytest.mark.skipif(os.name != "nt", reason="Windows dangling claim junction")
def test_dangling_session_claim_slot_is_restart_stop_evidence(
    tmp_path: Path,
) -> None:
    store, manifest, _ = _store(tmp_path)
    authorization = _authorization(manifest)
    with store.acquire_run_lock():
        store.publish_manifest()
        store.publish_authorization(authorization)
        locator = store._session_claim_locator(authorization)
        claim = store.external_state_root.joinpath(*locator.split("/"))
        claim.parent.mkdir(parents=True, exist_ok=True)
        missing_target = tmp_path / "missing-session-claim-target"
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", os.fspath(claim), os.fspath(missing_target)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip(f"host cannot create claim junction: {completed.stderr}")
        report = store.reconstruct_authoritative_state()
        assert report.status == "stopped_no_network"
        assert report.stop_code == "preexisting_process_session_claim_after_restart"
        assert report.restart_detected is True
        assert report.network_permitted is False
