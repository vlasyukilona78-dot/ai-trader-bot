from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from trading.market_data.mexc_futures_transport import (
    CompleteHttpAttemptEvidenceV1,
    HistoryResourceLimitsV1,
    IncompleteHttpAttemptEvidenceV1,
    candidate_endpoint_fixture_path,
    candidate_history_resource_limits_v1,
    candidate_history_retry_policy_v1,
    load_mexc_futures_endpoint_contract_v1,
    mexc_futures_transport_contract_hash,
)
from trading.market_data.strict_history import (
    HistoryArtifactConflictError,
    HistoryHttpStatusError,
    HistoryPayloadRangeError,
    HistoryPayloadValueError,
    HistoryStorageError,
    HistoryTransportError,
    strict_history_contract_hash,
)
import trading.market_data.strict_history_v2 as history_v2_module
from trading.market_data.strict_history_v2 import (
    HistoryArtifactCorruptionError,
    HistoryBudgetExceededError,
    HistoryRangeRequestV2,
    HistoryRestartIncompleteError,
    StrictHistoryArtifactStoreV2,
    StrictMexcHistoryCollectorV2,
    strict_history_v2_contract_hash,
)


BASE = 1_767_225_600


class _Clock:
    def __init__(self) -> None:
        self.epoch = 1_900_000_000_000_000
        self.monotonic = 1_000_000

    def epoch_us(self) -> int:
        return self.epoch

    def monotonic_us(self) -> int:
        return self.monotonic

    def sleep_us(self, duration_us: int) -> None:
        self.epoch += duration_us
        self.monotonic += duration_us


def _endpoint():
    return load_mexc_futures_endpoint_contract_v1(candidate_endpoint_fixture_path())


def _request(
    rows: int = 5,
    *,
    page_size: int = 2_000,
    limits: HistoryResourceLimitsV1 | None = None,
) -> HistoryRangeRequestV2:
    end = BASE + rows * 60
    return HistoryRangeRequestV2(
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        interval="Min1",
        start_open_ts=BASE,
        end_open_ts_exclusive=end,
        collection_as_of_us=end * 1_000_000,
        endpoint_contract=_endpoint(),
        resource_limits=limits or candidate_history_resource_limits_v1(),
        retry_policy=candidate_history_retry_policy_v1(),
        page_size=page_size,
    )


def _payload(page, *, malformed: bytes | None = None) -> bytes:
    if malformed is not None:
        return malformed
    timestamps = list(page.expected_timestamps())
    return json.dumps(
        {
            "success": True,
            "code": 0,
            "data": {
                "time": timestamps,
                "open": ["100.00"] * len(timestamps),
                "high": [103] * len(timestamps),
                "low": ["99"] * len(timestamps),
                "close": ["102.0"] * len(timestamps),
                "vol": [10] * len(timestamps),
                "amount": ["1000.50"] * len(timestamps),
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")


class _Transport:
    def __init__(self, request: HistoryRangeRequestV2, script=None, *, clock=None):
        self.request = request
        self.script = list(script or [])
        self.calls = []
        self.clock = clock

    @property
    def endpoint_contract_hash(self):
        return self.request.endpoint_contract.contract_hash

    @property
    def resource_limits_hash(self):
        return self.request.resource_limits.contract_hash

    @property
    def retry_policy_hash(self):
        return self.request.retry_policy.contract_hash

    @property
    def transport_contract_hash(self):
        return mexc_futures_transport_contract_hash()

    def _emit(self, evidence):
        if self.clock is not None:
            delta = evidence.terminal_monotonic_us - self.clock.monotonic_us()
            if delta > 0:
                self.clock.sleep_us(delta)
        return evidence

    def _common(self, page, ordinal):
        start = (
            1_900_000_000_000_000
            + page.page_ordinal * 1_000_000
            + ordinal * 2_000
        )
        monotonic_start = (
            1_000_000
            + page.page_ordinal * 1_000_000
            + ordinal * 2_000
        )
        return {
            "page_request": page,
            "attempt_ordinal": ordinal,
            "endpoint_contract_hash": self.request.endpoint_contract.contract_hash,
            "resource_limits_hash": self.request.resource_limits.contract_hash,
            "retry_policy_hash": self.request.retry_policy.contract_hash,
            "transport_contract_hash": mexc_futures_transport_contract_hash(),
            "scheduled_not_before_us": start,
            "scheduled_not_before_monotonic_us": monotonic_start,
            "request_started_at_us": start,
            "request_started_monotonic_us": monotonic_start,
            "headers_received_at_us": start + 100,
            "terminal_at_us": start + 1_000,
            "terminal_monotonic_us": monotonic_start + 1_000,
            "elapsed_monotonic_us": 1_000,
            "safe_headers": (("content-type", "application/json"),),
        }

    def fetch_page(self, page, *, attempt_ordinal, prior_attempt=None):
        self.calls.append((page, attempt_ordinal, prior_attempt))
        common = self._common(page, attempt_ordinal)
        if prior_attempt is not None:
            epoch_start = max(
                common["request_started_at_us"],
                prior_attempt.request_started_at_us
                + self.request.retry_policy.min_request_spacing_us,
                prior_attempt.terminal_at_us
                + self.request.retry_policy.backoff_before_attempt_us(
                    attempt_ordinal
                ),
            )
            monotonic_start = max(
                common["request_started_monotonic_us"],
                prior_attempt.request_started_monotonic_us
                + self.request.retry_policy.min_request_spacing_us,
                prior_attempt.terminal_monotonic_us
                + self.request.retry_policy.backoff_before_attempt_us(
                    attempt_ordinal
                ),
            )
            common.update(
                scheduled_not_before_us=epoch_start,
                scheduled_not_before_monotonic_us=monotonic_start,
                request_started_at_us=epoch_start,
                request_started_monotonic_us=monotonic_start,
                headers_received_at_us=epoch_start + 100,
                terminal_at_us=epoch_start + 1_000,
                terminal_monotonic_us=monotonic_start + 1_000,
            )
        if self.script:
            action = self.script.pop(0)
            if action == "timeout":
                common.update(
                    headers_received_at_us=None,
                    http_status=None,
                    safe_headers=(),
                )
                return self._emit(IncompleteHttpAttemptEvidenceV1(
                    **common,
                    body_bytes=b"",
                    outcome="timeout",
                    safe_error_code="fixture_timeout",
                ))
            if action == "body_limit":
                return self._emit(IncompleteHttpAttemptEvidenceV1(
                    **common,
                    http_status=200,
                    body_bytes=b"partial",
                    outcome="body_limit_exceeded",
                    safe_error_code="fixture_body_limit",
                ))
            if action == "postheader_timeout":
                return self._emit(IncompleteHttpAttemptEvidenceV1(
                    **common,
                    http_status=200,
                    body_bytes=b"partial",
                    outcome="timeout",
                    safe_error_code="fixture_read_timeout",
                ))
            if isinstance(action, int):
                return self._emit(CompleteHttpAttemptEvidenceV1(
                    **common,
                    http_status=action,
                    body_bytes=(b'{"success":false,"code":503}' if action != 200 else _payload(page)),
                ))
            if isinstance(action, bytes):
                return self._emit(CompleteHttpAttemptEvidenceV1(
                    **common, http_status=200, body_bytes=action
                ))
        return self._emit(CompleteHttpAttemptEvidenceV1(
            **common, http_status=200, body_bytes=_payload(page)
        ))


def _collect(tmp_path: Path, request: HistoryRangeRequestV2 | None = None, script=None):
    request = request or _request()
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "strict-v2", writable=True, storage_profile=request.storage_profile
    )
    clock = _Clock()
    transport = _Transport(request, script, clock=clock)
    collector = StrictMexcHistoryCollectorV2(
        transport=transport, store=store, clock=clock
    )
    return collector.collect_range(request), store, transport


def _inventory(root: Path):
    return tuple(
        sorted(
            (
                path.relative_to(root).as_posix(),
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in root.rglob("*")
            if path.is_file()
        )
    )


def test_v2_hash_is_pinned_without_changing_v1() -> None:
    assert strict_history_v2_contract_hash() == (
        "cce9922317ec5f0008f3b293103f9f5a17504b7143f81af1845d9d4765c44086"
    )
    assert strict_history_contract_hash() == (
        "6c17bd9de3e25210139da4491a1f35fbd0cec557707fb5d376a60ce23e04c6c1"
    )


def test_request_binds_every_pre_pilot_contract_and_exact_integer_clocks() -> None:
    request = _request()
    identities = request.contract_identities()
    assert identities["endpoint_contract_hash"] == request.endpoint_contract.contract_hash
    assert identities["resource_limits_hash"] == request.resource_limits.contract_hash
    assert identities["retry_policy_hash"] == request.retry_policy.contract_hash
    assert identities["attempt_contract_hash"] == mexc_futures_transport_contract_hash()
    assert type(request.collection_as_of_us) is int
    assert request.as_dict()["collection_as_of_us"] == (BASE + 300) * 1_000_000


def test_preflight_rows_pages_and_attempt_product_fail_before_transport() -> None:
    base = candidate_history_resource_limits_v1()
    limits = replace(base, max_pages=1, max_rows=3_000, max_attempts_per_page=3, max_total_attempts=3)
    with pytest.raises(HistoryBudgetExceededError, match="pages"):
        _request(2_001, page_size=2_000, limits=limits)


def test_transport_contract_identity_mismatch_fails_before_fetch_or_artifact(
    tmp_path,
) -> None:
    request = _request()
    other = _request(limits=replace(request.resource_limits, max_pages=199))
    transport = _Transport(other)
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "identity-preflight",
        writable=True,
        storage_profile=request.storage_profile,
    )
    collector = StrictMexcHistoryCollectorV2(
        transport=transport,
        store=store,
        clock=_Clock(),
    )
    with pytest.raises(
        history_v2_module.HistoryRangeContractError,
        match="transport_contract_identity_mismatch",
    ):
        collector.collect_range(request)
    assert transport.calls == []
    assert _inventory(store.root) == ()


def test_golden_collection_returns_fresh_disk_graph_and_restart_is_ready(tmp_path) -> None:
    request = _request()
    shard, store, transport = _collect(tmp_path, request)
    assert len(transport.calls) == 1
    assert tuple(row.bar_open_ts for row in shard.rows) == request.expected_timestamps()
    assert shard.rows[0].turnover_quote == "1000.5"
    assert shard.manifest.actual_attempt_count == 1
    readonly = StrictHistoryArtifactStoreV2(
        store.root, writable=False, storage_profile=request.storage_profile
    )
    loaded = readonly.load_complete_from_disk(
        request, expected_manifest_hash=shard.manifest.manifest_hash
    )
    assert loaded == shard
    report = readonly.reconcile_restart(
        (request,),
        expected_manifest_hashes={request.request_id: shard.manifest.manifest_hash},
    )
    assert report.ready
    assert report.request_states[0].state == "complete_verified"
    assert shard.manifest.logical_storage_bytes == sum(
        path.stat().st_size
        for path in store.root.rglob("*")
        if path.is_file() and path.name != "admission.json"
    )
    admission = json.loads(
        (
            store.root
            / "collections"
            / request.request_id
            / "admission.json"
        ).read_text(encoding="utf-8")
    )
    assert admission["manifest_hash"] == shard.manifest.manifest_hash
    assert admission["admitted_total_logical_storage_bytes"] == sum(
        path.stat().st_size for path in store.root.rglob("*") if path.is_file()
    )
    assert shard.manifest.as_dict()["storage_semantics"][
        "power_loss_durable_at_return"
    ] is False


def test_receipt_manifest_integer_fields_reject_python_numeric_type_aliases(
    tmp_path,
) -> None:
    request = _request()
    shard, store, _transport = _collect(tmp_path, request)
    page = shard.manifest.page_receipts[0]
    with pytest.raises(HistoryPayloadValueError, match="page_range_type"):
        replace(page, first_bar_open_ts=float(page.first_bar_open_ts))
    with pytest.raises(HistoryPayloadValueError, match="row_count_type"):
        replace(
            shard.manifest,
            expected_row_count=float(shard.manifest.expected_row_count),
        )

    malformed_page = page.as_dict()
    malformed_page["last_bar_open_ts"] = str(page.last_bar_open_ts)
    with pytest.raises(HistoryArtifactCorruptionError, match="page_receipt_is_invalid"):
        store._page_receipt_from_payload(malformed_page, page.page_request)


def test_source_row_ordinal_is_bound_to_exact_exchange_payload_position(tmp_path) -> None:
    request = _request()
    shard, _store, _transport = _collect(tmp_path, request)
    rows = list(shard.rows)
    rows[0] = replace(rows[0], source_row_ordinal=1)
    rows[1] = replace(rows[1], source_row_ordinal=0)
    swapped = tuple(rows)
    normalized = b"".join(
        history_v2_module._canonical_bytes(row.as_dict()) + b"\n" for row in swapped
    )
    manifest = replace(
        shard.manifest,
        normalized_shard_sha256=hashlib.sha256(normalized).hexdigest(),
    )
    with pytest.raises(HistoryPayloadValueError, match="source_ordinals"):
        type(shard)(rows=swapped, manifest=manifest)


def test_retry_evidence_is_complete_and_bound_before_success(tmp_path) -> None:
    request = _request()
    shard, store, _transport = _collect(tmp_path, request, script=[503, 200])
    page = shard.manifest.page_receipts[0]
    assert len(page.attempt_receipt_hashes) == 2
    assert shard.manifest.actual_attempt_count == 2
    assert len(list((store.root / "attempts").glob("*.json"))) == 2
    store.load_complete_from_disk(request)


@pytest.mark.parametrize("action", ["timeout", "body_limit"])
def test_incomplete_attempt_is_retained_but_never_parsed_or_completed(tmp_path, action) -> None:
    request = _request(limits=replace(candidate_history_resource_limits_v1(), max_attempts_per_page=1))
    store = StrictHistoryArtifactStoreV2(
        tmp_path / action, writable=True, storage_profile=request.storage_profile
    )
    clock = _Clock()
    collector = StrictMexcHistoryCollectorV2(
        transport=_Transport(request, [action], clock=clock),
        store=store,
        clock=clock,
    )
    expected = HistoryBudgetExceededError if action == "body_limit" else Exception
    with pytest.raises(expected):
        collector.collect_range(request)
    assert list((store.root / "attempts").glob("*.json"))
    assert not (store.root / "collections" / request.request_id / "manifest.json").exists()
    with pytest.raises(HistoryRestartIncompleteError):
        store.load_complete_from_disk(request)


def test_crash_orphan_attempt_blocks_fresh_collection_and_admission(tmp_path) -> None:
    request = _request()
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "orphan-attempt",
        writable=True,
        storage_profile=request.storage_profile,
    )
    page = history_v2_module._plan_pages(request)[0]
    orphan = _Transport(request, ["timeout"]).fetch_page(
        page,
        attempt_ordinal=0,
    )
    with store.writer_session(request):
        store.persist_attempt(orphan)

    clock = _Clock()
    fresh_transport = _Transport(request, clock=clock)
    collector = StrictMexcHistoryCollectorV2(
        transport=fresh_transport,
        store=store,
        clock=clock,
    )
    with pytest.raises(
        history_v2_module.HistoryArtifactForkError,
        match="pristine_request_namespace",
    ):
        collector.collect_range(request)
    assert fresh_transport.calls == []
    collection = store.root / "collections" / request.request_id
    assert not (collection / "manifest.json").exists()
    assert not (collection / "admission.json").exists()


def test_writer_store_is_explicitly_one_range_shard_not_full_universe(tmp_path) -> None:
    first = _request()
    _shard, store, _transport = _collect(tmp_path, first)
    second = replace(first, symbol="ETHUSDT", venue_symbol="ETH_USDT")
    clock = _Clock()
    transport = _Transport(second, clock=clock)
    collector = StrictMexcHistoryCollectorV2(
        transport=transport,
        store=store,
        clock=clock,
    )
    with pytest.raises(
        history_v2_module.HistoryArtifactForkError,
        match="store_scope_request_mismatch",
    ):
        collector.collect_range(second)
    assert transport.calls == []


def test_lifetime_writer_lock_blocks_second_store_before_fetch(tmp_path) -> None:
    request = _request()
    root = tmp_path / "writer-lock"
    first = StrictHistoryArtifactStoreV2(
        root,
        writable=True,
        storage_profile=request.storage_profile,
    )
    second = StrictHistoryArtifactStoreV2(
        root,
        writable=True,
        storage_profile=request.storage_profile,
    )
    clock = _Clock()
    transport = _Transport(request, clock=clock)
    collector = StrictMexcHistoryCollectorV2(
        transport=transport,
        store=second,
        clock=clock,
    )
    with first.writer_session(request):
        with pytest.raises(
            history_v2_module.HistoryArtifactForkError,
            match="writer_session_is_already_active",
        ):
            collector.collect_range(request)
    assert transport.calls == []


def test_writer_lock_is_interprocess_and_released_after_exception(tmp_path) -> None:
    request = _request()
    root = tmp_path / "interprocess-writer-lock"
    first = StrictHistoryArtifactStoreV2(
        root,
        writable=True,
        storage_profile=request.storage_profile,
    )
    child_code = """
import sys
from trading.market_data.mexc_futures_transport import (
    candidate_endpoint_fixture_path,
    candidate_history_resource_limits_v1,
    candidate_history_retry_policy_v1,
    load_mexc_futures_endpoint_contract_v1,
)
from trading.market_data.strict_history_v2 import (
    HistoryArtifactForkError,
    HistoryRangeRequestV2,
    StrictHistoryArtifactStoreV2,
)
base = 1767225600
request = HistoryRangeRequestV2(
    venue="mexc_contract",
    symbol="BTCUSDT",
    venue_symbol="BTC_USDT",
    interval="Min1",
    start_open_ts=base,
    end_open_ts_exclusive=base + 300,
    collection_as_of_us=(base + 300) * 1_000_000,
    endpoint_contract=load_mexc_futures_endpoint_contract_v1(
        candidate_endpoint_fixture_path()
    ),
    resource_limits=candidate_history_resource_limits_v1(),
    retry_policy=candidate_history_retry_policy_v1(),
)
store = StrictHistoryArtifactStoreV2(
    sys.argv[1], writable=True, storage_profile=request.storage_profile
)
try:
    with store.writer_session(request):
        raise SystemExit(3)
except HistoryArtifactForkError:
    print("blocked")
"""
    with pytest.raises(RuntimeError, match="fixture_writer_body_error"):
        with first.writer_session(request):
            child = subprocess.run(
                [sys.executable, "-c", child_code, str(root)],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            assert child.returncode == 0, child.stderr
            assert child.stdout.strip() == "blocked"
            raise RuntimeError("fixture_writer_body_error")

    reopened = StrictHistoryArtifactStoreV2(
        root,
        writable=True,
        storage_profile=request.storage_profile,
    )
    with reopened.writer_session(request):
        pass


@pytest.mark.skipif(
    history_v2_module.os.name != "nt",
    reason="Windows msvcrt lock error classification",
)
def test_writer_lock_io_failure_is_not_misclassified_as_contention(
    tmp_path, monkeypatch
) -> None:
    import msvcrt

    request = _request()
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "lock-io",
        writable=True,
        storage_profile=request.storage_profile,
    )
    real_locking = msvcrt.locking

    def fail_lock(_fd, mode, _count):
        if mode == msvcrt.LK_NBLCK:
            raise OSError(history_v2_module.errno.EIO, "fixture_lock_io_error")
        return real_locking(_fd, mode, _count)

    monkeypatch.setattr(msvcrt, "locking", fail_lock)
    with pytest.raises(HistoryStorageError, match="writer_lock_acquisition_failed"):
        with store.writer_session(request):
            pass

    monkeypatch.setattr(msvcrt, "locking", real_locking)
    with store.writer_session(request):
        pass


def test_attempt_persistence_requires_active_writer_session(tmp_path) -> None:
    request = _request()
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "writer-required",
        writable=True,
        storage_profile=request.storage_profile,
    )
    attempt = _Transport(request, ["timeout"]).fetch_page(
        history_v2_module._plan_pages(request)[0],
        attempt_ordinal=0,
    )
    with pytest.raises(HistoryStorageError, match="writer_session_is_required"):
        store.persist_attempt(attempt)


def test_direct_writer_session_binds_shard_scope_across_reopen(tmp_path) -> None:
    first = _request()
    second = replace(first, symbol="ETHUSDT", venue_symbol="ETH_USDT")
    root = tmp_path / "direct-session-scope"
    store = StrictHistoryArtifactStoreV2(
        root,
        writable=True,
        storage_profile=first.storage_profile,
    )
    attempt = _Transport(first, ["timeout"]).fetch_page(
        history_v2_module._plan_pages(first)[0],
        attempt_ordinal=0,
    )
    with store.writer_session(first):
        store.persist_attempt(attempt)
    with pytest.raises(
        history_v2_module.HistoryArtifactForkError,
        match="store_scope_request_mismatch",
    ):
        with store.writer_session(second):
            pass

    reopened = StrictHistoryArtifactStoreV2(
        root,
        writable=True,
        storage_profile=first.storage_profile,
    )
    with pytest.raises(
        history_v2_module.HistoryArtifactForkError,
        match="store_scope_request_mismatch",
    ):
        with reopened.writer_session(second):
            pass


def test_empty_writer_session_persists_exact_shard_scope_across_reopen(
    tmp_path,
) -> None:
    first = _request()
    second = replace(first, symbol="ETHUSDT", venue_symbol="ETH_USDT")
    root = tmp_path / "empty-session-scope"
    store = StrictHistoryArtifactStoreV2(
        root,
        writable=True,
        storage_profile=first.storage_profile,
    )
    with store.writer_session(first):
        pass
    assert (root / "scope.json").is_file()

    reopened = StrictHistoryArtifactStoreV2(
        root,
        writable=True,
        storage_profile=first.storage_profile,
    )
    with pytest.raises(
        history_v2_module.HistoryArtifactForkError,
        match="store_scope_request_mismatch",
    ):
        with reopened.writer_session(second):
            pass
    with reopened.writer_session(first):
        pass


def test_oversized_scope_marker_is_rejected_before_artifact(tmp_path) -> None:
    request = _request()
    oversized_endpoint = replace(
        request.endpoint_contract,
        plan_reference_url="https://www.mexc.com/" + ("x" * 300_000),
    )
    oversized_request = replace(request, endpoint_contract=oversized_endpoint)
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "oversized-scope",
        writable=True,
        storage_profile=oversized_request.storage_profile,
    )

    with pytest.raises(
        HistoryBudgetExceededError,
        match="scope_marker_bytes",
    ):
        with store.writer_session(oversized_request):
            pass
    assert _inventory(store.root) == ()
    assert not store._writer_lock_path.exists()


def test_writer_session_rejects_foreign_thread_high_level_write(tmp_path) -> None:
    request = _request()
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "writer-thread-owner",
        writable=True,
        storage_profile=request.storage_profile,
    )
    attempt = _Transport(request, ["timeout"]).fetch_page(
        history_v2_module._plan_pages(request)[0],
        attempt_ordinal=0,
    )
    failures: list[BaseException] = []

    def foreign_write() -> None:
        try:
            store.persist_attempt(attempt)
        except BaseException as exc:
            failures.append(exc)

    with store.writer_session(request):
        worker = threading.Thread(target=foreign_write)
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive()

    assert len(failures) == 1
    assert isinstance(failures[0], HistoryStorageError)
    assert "writer_session_owner_mismatch" in str(failures[0])
    assert not (store.root / "attempts").exists()
    assert not (store.root / "raw").exists()


def test_nonretryable_complete_http_body_is_retained_but_not_parsed(tmp_path) -> None:
    request = _request()
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "http", writable=True, storage_profile=request.storage_profile
    )
    clock = _Clock()
    collector = StrictMexcHistoryCollectorV2(
        transport=_Transport(request, [404], clock=clock),
        store=store,
        clock=clock,
    )
    with pytest.raises(HistoryHttpStatusError):
        collector.collect_range(request)
    assert list((store.root / "raw").rglob("*.bin"))
    assert not (store.root / "collections").exists()


def test_loader_rejects_actual_raw_length_disagreement(tmp_path) -> None:
    request = _request()
    shard, store, _transport = _collect(tmp_path, request)
    attempt_hash = shard.manifest.page_receipts[0].attempt_receipt_hashes[0]
    attempt_path = store.root / "attempts" / f"{attempt_hash}.json"
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    payload["captured_body_length"] += 1
    attempt_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(HistoryArtifactCorruptionError, match="raw_body_length"):
        store.load_complete_from_disk(request)


def test_loader_rejects_raw_hardlink_tamper(tmp_path) -> None:
    request = _request()
    shard, store, _transport = _collect(tmp_path, request)
    digest = shard.manifest.page_receipts[0].raw_body_sha256
    raw_path = store.root / "raw" / "sha256" / digest[:2] / f"{digest}.bin"
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    with pytest.raises(HistoryArtifactCorruptionError, match="length_or_hash"):
        store.load_complete_from_disk(request)


def test_torn_or_duplicate_key_manifest_never_becomes_complete(tmp_path) -> None:
    request = _request()
    shard, store, _transport = _collect(tmp_path, request)
    manifest = store.root / "collections" / request.request_id / "manifest.json"
    original = manifest.read_bytes()
    manifest.write_bytes(original[:-1])
    with pytest.raises(HistoryArtifactCorruptionError):
        store.load_complete_from_disk(request)
    manifest.write_bytes(b'{"manifest_hash":"x","manifest_hash":"y"}\n')
    with pytest.raises(HistoryArtifactCorruptionError, match="duplicate"):
        store.load_complete_from_disk(request)
    assert shard.manifest.manifest_hash


def test_restart_reports_temp_without_promoting_or_mutating_it(tmp_path) -> None:
    request = _request()
    shard, store, _transport = _collect(tmp_path, request)
    temp = store.root / "collections" / request.request_id / ".crash.tmp"
    temp.write_bytes(b"partial")
    before = _inventory(store.root)
    report = store.reconcile_restart((request,))
    after = _inventory(store.root)
    assert report.ready
    assert report.request_states[0].state == "complete_verified"
    assert report.temp_paths
    assert before == after
    assert temp.read_bytes() == b"partial"
    assert shard.manifest.manifest_hash


def test_admission_temp_alias_keeps_restart_ready(tmp_path, monkeypatch) -> None:
    request = _request()
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "s",
        writable=True,
        storage_profile=request.storage_profile,
    )
    original_unlink = Path.unlink
    temp_cleanup_count = {"value": 0}

    def guarded_unlink(path: Path, *args, **kwargs):
        if path.name.startswith(".") and path.name.endswith(".tmp"):
            temp_cleanup_count["value"] += 1
            if temp_cleanup_count["value"] == 6:
                raise OSError("fixture_admission_temp_cleanup_failed")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", guarded_unlink)
    clock = _Clock()
    collector = StrictMexcHistoryCollectorV2(
        transport=_Transport(request, clock=clock),
        store=store,
        clock=clock,
    )
    shard = collector.collect_range(request)
    assert temp_cleanup_count["value"] == 6

    report = store.reconcile_restart((request,))
    assert report.ready
    assert report.request_states[0].state == "complete_verified"
    assert len(report.temp_paths) == 1
    assert report.temp_paths[0].startswith(
        f"collections/{request.request_id}/."
    )
    assert store.load_complete_from_disk(request) == shard


def test_restart_rejects_detached_anchor_for_unknown_request_id(tmp_path) -> None:
    request = _request()
    shard, store, _transport = _collect(tmp_path, request)
    with pytest.raises(
        history_v2_module.HistoryRangeContractError,
        match="unknown_request",
    ):
        store.reconcile_restart(
            (request,),
            expected_manifest_hashes={"f" * 64: shard.manifest.manifest_hash},
        )


def test_restart_contract_is_exactly_one_request_per_shard_store(tmp_path) -> None:
    first = _request()
    second = replace(first, symbol="ETHUSDT", venue_symbol="ETH_USDT")
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "single-restart-scope",
        writable=True,
        storage_profile=first.storage_profile,
    )
    with pytest.raises(
        history_v2_module.HistoryRangeContractError,
        match="exactly_one_request_shard",
    ):
        store.reconcile_restart((first, second))


def test_alternate_normalized_shard_is_an_ambiguous_fork(tmp_path) -> None:
    request = _request()
    _shard, store, _transport = _collect(tmp_path, request)
    alternate = store.root / "normalized" / request.request_id / f"{'1' * 64}.jsonl"
    alternate.write_bytes(b"{}\n")
    report = store.reconcile_restart((request,))
    assert not report.ready
    assert report.request_states[0].state == "ambiguous_fork"
    assert report.alternate_normalized_paths


def test_read_only_store_cannot_publish_or_clean_residue(tmp_path) -> None:
    request = _request()
    shard, store, _transport = _collect(tmp_path, request)
    readonly = StrictHistoryArtifactStoreV2(
        store.root, writable=False, storage_profile=request.storage_profile
    )
    before = _inventory(store.root)
    with pytest.raises(HistoryStorageError, match="read_only"):
        readonly.publish_graph_candidate(shard)
    assert _inventory(store.root) == before


def test_fixed_manifest_path_rejects_different_evidence(tmp_path) -> None:
    request = _request()
    shard, store, _transport = _collect(tmp_path, request)
    manifest_path = store.root / "collections" / request.request_id / "manifest.json"
    with pytest.raises(HistoryArtifactConflictError):
        store._publish_immutable(
            manifest_path.relative_to(store.root), manifest_path.read_bytes() + b" "
        )
    assert store.load_complete_from_disk(request) == shard


def test_manifest_without_valid_admission_is_never_complete(tmp_path) -> None:
    request = _request()
    shard, store, _transport = _collect(tmp_path, request)
    admission = store.root / "collections" / request.request_id / "admission.json"
    admission.unlink()
    with pytest.raises(HistoryRestartIncompleteError, match="admission_marker"):
        store.load_complete_from_disk(request)
    report = store.reconcile_restart((request,))
    assert not report.ready
    assert report.request_states[0].state == "incomplete"

    admission.write_bytes(b"{}\n")
    with pytest.raises(HistoryArtifactCorruptionError, match="admission_marker"):
        store.load_complete_from_disk(request)
    assert shard.manifest.manifest_hash


def test_admission_runtime_cannot_predate_manifest_collection_runtime(tmp_path) -> None:
    request = _request()
    shard, store, _transport = _collect(tmp_path, request)
    assert shard.manifest.collection_runtime_us > 0
    admission = store.root / "collections" / request.request_id / "admission.json"
    admission.unlink()
    admission.write_bytes(
        store._admission_bytes(
            request=request,
            manifest_hash=shard.manifest.manifest_hash,
            admission_decision_runtime_us=0,
            graph_logical_storage_bytes=shard.manifest.logical_storage_bytes,
        )
    )
    with pytest.raises(
        HistoryArtifactCorruptionError, match="runtime_precedes_manifest"
    ):
        store.load_complete_from_disk(request)


def test_admission_publication_failure_leaves_only_incomplete_graph(tmp_path) -> None:
    request = _request()

    class AdmissionFailStore(StrictHistoryArtifactStoreV2):
        def _publish_admission_marker(self, *args, **kwargs):
            raise HistoryStorageError("fixture_admission_write_failed")

    store = AdmissionFailStore(
        tmp_path / "admission-fail",
        writable=True,
        storage_profile=request.storage_profile,
    )
    clock = _Clock()
    collector = StrictMexcHistoryCollectorV2(
        transport=_Transport(request, clock=clock), store=store, clock=clock
    )
    with pytest.raises(HistoryStorageError, match="fixture_admission_write_failed"):
        collector.collect_range(request)
    collection = store.root / "collections" / request.request_id
    assert (collection / "manifest.json").is_file()
    assert not (collection / "admission.json").exists()
    with pytest.raises(HistoryRestartIncompleteError, match="admission_marker"):
        store.load_complete_from_disk(request)


def test_runtime_overrun_during_graph_reload_cannot_publish_admission(tmp_path) -> None:
    limits = replace(
        candidate_history_resource_limits_v1(),
        max_collection_runtime_us=2_000,
        max_attempt_runtime_us=2_000,
    )
    request = _request(limits=limits)
    clock = _Clock()

    class SlowReloadStore(StrictHistoryArtifactStoreV2):
        def _load_complete_graph(self, *args, **kwargs):
            graph = super()._load_complete_graph(*args, **kwargs)
            clock.sleep_us(2_001)
            return graph

    store = SlowReloadStore(
        tmp_path / "slow-reload",
        writable=True,
        storage_profile=request.storage_profile,
    )
    collector = StrictMexcHistoryCollectorV2(
        transport=_Transport(request, clock=clock), store=store, clock=clock
    )
    with pytest.raises(HistoryBudgetExceededError, match="collection_runtime_us"):
        collector.collect_range(request)
    collection = store.root / "collections" / request.request_id
    assert (collection / "manifest.json").exists()
    assert not (collection / "admission.json").exists()


def test_started_attempt_is_persisted_before_runtime_rejection(tmp_path) -> None:
    limits = replace(
        candidate_history_resource_limits_v1(),
        max_attempt_runtime_us=999,
    )
    request = _request(limits=limits)
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "persist-first",
        writable=True,
        storage_profile=request.storage_profile,
    )
    clock = _Clock()
    collector = StrictMexcHistoryCollectorV2(
        transport=_Transport(request, clock=clock), store=store, clock=clock
    )
    with pytest.raises(HistoryBudgetExceededError, match="attempt_runtime_us"):
        collector.collect_range(request)
    assert list((store.root / "attempts").glob("*.json"))
    assert list((store.root / "raw").rglob("*.bin"))
    assert not (store.root / "collections").exists()


def test_attempt_clock_domain_mismatch_is_persisted_then_rejected(tmp_path) -> None:
    request = _request()
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "clock-domain",
        writable=True,
        storage_profile=request.storage_profile,
    )
    # This fake returns a terminal timestamp in the future but deliberately
    # does not advance the collector's shared clock.
    collector = StrictMexcHistoryCollectorV2(
        transport=_Transport(request), store=store, clock=_Clock()
    )
    with pytest.raises(HistoryTransportError, match="clock_domain_mismatch"):
        collector.collect_range(request)
    assert list((store.root / "attempts").glob("*.json"))
    assert not (store.root / "collections").exists()


def test_postheader_timeout_exhaustion_remains_transport_typed(tmp_path) -> None:
    limits = replace(
        candidate_history_resource_limits_v1(), max_attempts_per_page=1
    )
    request = _request(limits=limits)
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "postheader-timeout",
        writable=True,
        storage_profile=request.storage_profile,
    )
    clock = _Clock()
    collector = StrictMexcHistoryCollectorV2(
        transport=_Transport(request, ["postheader_timeout"], clock=clock),
        store=store,
        clock=clock,
    )
    with pytest.raises(HistoryTransportError, match="timeout.fixture_read_timeout"):
        collector.collect_range(request)
    assert list((store.root / "attempts").glob("*.json"))


def test_unordered_exchange_payload_is_not_sorted_or_laundered(tmp_path) -> None:
    request = _request()
    payload = json.loads(_payload(StrictMexcHistoryCollectorV2.plan_pages(request)[0]))
    for key in ("time", "open", "high", "low", "close", "vol", "amount"):
        payload["data"][key] = list(reversed(payload["data"][key]))
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "unordered",
        writable=True,
        storage_profile=request.storage_profile,
    )
    clock = _Clock()
    collector = StrictMexcHistoryCollectorV2(
        transport=_Transport(request, [body], clock=clock),
        store=store,
        clock=clock,
    )
    with pytest.raises(HistoryPayloadRangeError, match="not_in_expected_order"):
        collector.collect_range(request)
    assert list((store.root / "attempts").glob("*.json"))
    assert not (store.root / "collections").exists()


def test_retry_delay_is_revalidated_and_underpaced_attempt_is_retained(tmp_path) -> None:
    request = _request()

    class UnderpacedTransport(_Transport):
        def fetch_page(self, page, *, attempt_ordinal, prior_attempt=None):
            if attempt_ordinal == 0:
                return super().fetch_page(
                    page, attempt_ordinal=attempt_ordinal, prior_attempt=prior_attempt
                )
            common = self._common(page, attempt_ordinal)
            epoch_start = prior_attempt.terminal_at_us + 1
            mono_start = prior_attempt.terminal_monotonic_us + 1
            common.update(
                scheduled_not_before_us=epoch_start,
                scheduled_not_before_monotonic_us=mono_start,
                request_started_at_us=epoch_start,
                request_started_monotonic_us=mono_start,
                headers_received_at_us=epoch_start + 1,
                terminal_at_us=epoch_start + 2,
                terminal_monotonic_us=mono_start + 2,
                elapsed_monotonic_us=2,
            )
            return self._emit(CompleteHttpAttemptEvidenceV1(
                **common, http_status=200, body_bytes=_payload(page)
            ))

    store = StrictHistoryArtifactStoreV2(
        tmp_path / "underpaced",
        writable=True,
        storage_profile=request.storage_profile,
    )
    clock = _Clock()
    transport = UnderpacedTransport(request, [503], clock=clock)
    collector = StrictMexcHistoryCollectorV2(
        transport=transport, store=store, clock=clock
    )
    with pytest.raises(HistoryTransportError, match="retry_delay_was_not_honoured"):
        collector.collect_range(request)
    assert len(list((store.root / "attempts").glob("*.json"))) == 2


def test_restart_reports_attempt_and_raw_temps_without_promoting(tmp_path) -> None:
    request = _request()
    _shard, store, _transport = _collect(tmp_path, request)
    attempt_temp = store.root / "attempts" / ".attempt.crash.tmp"
    raw_temp = store.root / "raw" / ".raw.crash.tmp"
    attempt_temp.write_bytes(b"partial")
    raw_temp.write_bytes(b"partial")
    before = _inventory(store.root)
    report = store.reconcile_restart((request,))
    assert report.ready
    assert report.request_states[0].state == "complete_verified"
    assert attempt_temp.relative_to(store.root).as_posix() in report.temp_paths
    assert raw_temp.relative_to(store.root).as_posix() in report.temp_paths
    assert _inventory(store.root) == before


def test_large_temp_alias_is_reported_without_double_charging_byte_budget(
    tmp_path,
) -> None:
    base = _request()
    request = _request(
        limits=replace(
            base.resource_limits,
            max_logical_storage_bytes=10_000_000,
        )
    )
    _shard, store, _transport = _collect(tmp_path, request)
    alias_source = tmp_path / "large-alias-source.bin"
    with alias_source.open("wb") as handle:
        handle.truncate(request.resource_limits.max_logical_storage_bytes)
    temp_alias = (
        store.root
        / "collections"
        / request.request_id
        / ".large-hardlink-alias.tmp"
    )
    history_v2_module.os.link(alias_source, temp_alias)

    report = store.reconcile_restart((request,))
    assert report.ready
    assert temp_alias.relative_to(store.root).as_posix() in report.temp_paths


def test_restart_rejects_unknown_collection_artifact(tmp_path) -> None:
    request = _request()
    _shard, store, _transport = _collect(tmp_path, request)
    unknown = store.root / "collections" / request.request_id / "shadow.json"
    unknown.write_bytes(b"{}\n")
    with pytest.raises(HistoryArtifactCorruptionError, match="unexpected_file"):
        store.reconcile_restart((request,))


@pytest.mark.parametrize(
    "residue",
    ["root_file", "foreign_collection", "noncanonical_raw_path"],
)
def test_restart_rejects_unknown_store_namespace_residue(tmp_path, residue) -> None:
    request = _request()
    _shard, store, _transport = _collect(tmp_path, request)
    if residue == "root_file":
        (store.root / "shadow.bin").write_bytes(b"shadow")
    elif residue == "foreign_collection":
        (store.root / "collections" / ("1" * 64)).mkdir()
    else:
        path = store.root / "raw" / "shadow"
        path.mkdir()
        (path / (("1" * 64) + ".bin")).write_bytes(b"shadow")
    with pytest.raises(
        HistoryArtifactCorruptionError,
        match="unexpected_(?:file|directory)",
    ):
        store.reconcile_restart((request,))


def test_restart_scan_fails_typed_before_materializing_artifact_flood(
    tmp_path, monkeypatch
) -> None:
    request = _request()
    _shard, store, _transport = _collect(tmp_path, request)
    monkeypatch.setattr(history_v2_module, "_MAX_RESTART_SCAN_ENTRIES", 3)
    with pytest.raises(HistoryBudgetExceededError, match="restart_scan_entries"):
        store.reconcile_restart((request,))


def test_store_rejects_lexical_path_escape(tmp_path) -> None:
    request = _request()
    store = StrictHistoryArtifactStoreV2(
        tmp_path / "no-escape",
        writable=True,
        storage_profile=request.storage_profile,
    )
    escaped = store.root.parent / "escaped.bin"
    with pytest.raises(HistoryStorageError, match="artifact_path_is_invalid"):
        store._publish_immutable(Path("..") / escaped.name, b"no")
    assert not escaped.exists()


def test_store_rejects_symlink_or_junction_in_supplied_root_chain(tmp_path) -> None:
    real = tmp_path / "real-root"
    real.mkdir()
    alias = tmp_path / "alias-root"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit creating a directory symlink")
    with pytest.raises(
        HistoryArtifactCorruptionError, match="supplied_root_reparse_point"
    ):
        StrictHistoryArtifactStoreV2(
            alias / "pilot",
            writable=True,
            storage_profile=_request().storage_profile,
        )


def test_min1_adapter_enforces_integer_microsecond_source_close(tmp_path) -> None:
    request = _request()
    shard, _store, _transport = _collect(tmp_path, request)
    frame, receipts = shard.to_min1_aggregation_inputs()
    assert len(frame) == len(receipts) == 5
    assert receipts[-1].request_started_at >= BASE + 300
    page = shard.manifest.page_receipts[0]
    with pytest.raises(
        HistoryPayloadValueError, match="request_started_before_source_close"
    ):
        replace(
            page,
            request_started_at_us=(page.last_bar_open_ts + 60) * 1_000_000 - 1,
        )
