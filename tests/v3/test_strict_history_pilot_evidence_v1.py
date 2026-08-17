from __future__ import annotations

from dataclasses import FrozenInstanceError
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from trading.market_data.mexc_futures_transport import (
    CompleteHttpAttemptEvidenceV1,
    candidate_endpoint_fixture_path,
    candidate_history_resource_limits_v1,
    candidate_history_retry_policy_v1,
    load_mexc_futures_endpoint_contract_v1,
    mexc_futures_transport_contract_hash,
)
from trading.market_data.strict_history_v2 import (
    HistoryRangeRequestV2,
    StrictHistoryArtifactStoreV2,
    StrictMexcHistoryCollectorV2,
)
import trading.market_data.strict_history_pilot_evidence as evidence_module
from trading.market_data.strict_history_pilot_evidence import (
    PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE,
    PILOT_OUTPUT_LAYOUT_STATUS_UNRESOLVED,
    PilotAdmissionAccountingV1,
    PilotAttemptAccountingV1,
    PilotLogicalReferenceV1,
    PilotPageAccountingV1,
    StrictHistoryPilotEvidenceBoundsStop,
    StrictHistoryPilotEvidenceContractError,
    StrictHistoryPilotEvidenceLayoutStop,
    StrictHistoryPilotEvidenceResidueStop,
    StrictHistoryPilotEvidenceStop,
    StrictHistoryPilotEvidenceV1,
    read_strict_history_pilot_evidence_v1,
    require_strict_history_pilot_compatible_evidence_v1,
    strict_history_pilot_evidence_contract_hash,
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


def _request(rows: int = 4, *, page_size: int = 2_000) -> HistoryRangeRequestV2:
    end = BASE + (rows * 60)
    return HistoryRangeRequestV2(
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        interval="Min1",
        start_open_ts=BASE,
        end_open_ts_exclusive=end,
        collection_as_of_us=end * 1_000_000,
        endpoint_contract=load_mexc_futures_endpoint_contract_v1(
            candidate_endpoint_fixture_path()
        ),
        resource_limits=candidate_history_resource_limits_v1(),
        retry_policy=candidate_history_retry_policy_v1(),
        page_size=page_size,
    )


def _success_body(page) -> bytes:
    timestamps = list(page.expected_timestamps())
    return json.dumps(
        {
            "success": True,
            "code": 0,
            "data": {
                "time": timestamps,
                "open": ["100"] * len(timestamps),
                "high": ["103"] * len(timestamps),
                "low": ["99"] * len(timestamps),
                "close": ["102"] * len(timestamps),
                "vol": ["10"] * len(timestamps),
                "amount": ["1000.5"] * len(timestamps),
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")


class _Transport:
    def __init__(self, request: HistoryRangeRequestV2, clock: _Clock, statuses=()):
        self.request = request
        self.clock = clock
        self.statuses = list(statuses)
        self.emitted = []

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

    def fetch_page(self, page, *, attempt_ordinal, prior_attempt=None):
        base_start = (
            1_900_000_000_000_000
            + (page.page_ordinal * 10_000_000)
            + (attempt_ordinal * 2_000)
        )
        base_mono = (
            1_000_000
            + (page.page_ordinal * 10_000_000)
            + (attempt_ordinal * 2_000)
        )
        if prior_attempt is not None:
            retry = self.request.retry_policy.backoff_before_attempt_us(
                attempt_ordinal
            )
            base_start = max(
                base_start,
                prior_attempt.request_started_at_us
                + self.request.retry_policy.min_request_spacing_us,
                prior_attempt.terminal_at_us + retry,
            )
            base_mono = max(
                base_mono,
                prior_attempt.request_started_monotonic_us
                + self.request.retry_policy.min_request_spacing_us,
                prior_attempt.terminal_monotonic_us + retry,
            )
        common = {
            "page_request": page,
            "attempt_ordinal": attempt_ordinal,
            "endpoint_contract_hash": self.request.endpoint_contract.contract_hash,
            "resource_limits_hash": self.request.resource_limits.contract_hash,
            "retry_policy_hash": self.request.retry_policy.contract_hash,
            "transport_contract_hash": mexc_futures_transport_contract_hash(),
            "scheduled_not_before_us": base_start,
            "scheduled_not_before_monotonic_us": base_mono,
            "request_started_at_us": base_start,
            "request_started_monotonic_us": base_mono,
            "headers_received_at_us": base_start + 100,
            "terminal_at_us": base_start + 1_000,
            "terminal_monotonic_us": base_mono + 1_000,
            "elapsed_monotonic_us": 1_000,
            "safe_headers": (("content-type", "application/json"),),
        }
        status = self.statuses.pop(0) if self.statuses else 200
        attempt = CompleteHttpAttemptEvidenceV1(
            **common,
            http_status=status,
            body_bytes=(
                b'{"success":false,"code":503}'
                if status == 503
                else _success_body(page)
            ),
        )
        delta = attempt.terminal_monotonic_us - self.clock.monotonic_us()
        if delta > 0:
            self.clock.sleep_us(delta)
        self.emitted.append(attempt)
        return attempt


def _collect(tmp_path: Path, *, request=None, statuses=()):
    request = request or _request()
    root = tmp_path / "strict-v2"
    store = StrictHistoryArtifactStoreV2(
        root,
        writable=True,
        storage_profile=request.storage_profile,
    )
    clock = _Clock()
    transport = _Transport(request, clock, statuses=statuses)
    shard = StrictMexcHistoryCollectorV2(
        transport=transport,
        store=store,
        clock=clock,
    ).collect_range(request)
    return request, shard, store.root, transport


def _read(tmp_path: Path, *, request=None, statuses=()):
    request, shard, root, transport = _collect(
        tmp_path,
        request=request,
        statuses=statuses,
    )
    result = read_strict_history_pilot_evidence_v1(
        request=request,
        artifact_root=root,
        expected_manifest_hash=shard.manifest.manifest_hash,
    )
    return request, shard, root, transport, result


def _attempt_paths(shard, root: Path) -> tuple[Path, ...]:
    return tuple(
        root / "attempts" / f"{digest}.json"
        for page in shard.manifest.page_receipts
        for digest in page.attempt_receipt_hashes
    )


def _raw_paths(shard, root: Path) -> tuple[Path, ...]:
    result = []
    for attempt_path in _attempt_paths(shard, root):
        payload = json.loads(attempt_path.read_text(encoding="utf-8"))
        digest = payload["captured_body_sha256"]
        result.append(root / "raw" / "sha256" / digest[:2] / f"{digest}.bin")
    return tuple(result)


def test_contract_is_pinned_and_exact_result_roundtrips(tmp_path) -> None:
    _request_value, _shard, root, _transport, result = _read(tmp_path)
    assert strict_history_pilot_evidence_contract_hash() == (
        "a546b37de9ed2da04eefb8d607b98719a09ab8378c2ab1d459eac02ecb899b8e"
    )
    assert StrictHistoryPilotEvidenceV1.parse(result.as_dict()) == result
    assert result.evidence_hash == hashlib.sha256(
        json.dumps(
            result.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    rendered = json.dumps(result.as_dict(), sort_keys=True)
    assert str(root) not in rendered
    assert "official_document" not in rendered
    assert "failure_evidence" not in rendered
    with pytest.raises(FrozenInstanceError):
        result.row_count = 0  # type: ignore[misc]
    with pytest.raises(StrictHistoryPilotEvidenceContractError):
        StrictHistoryPilotEvidenceV1.parse({**result.as_dict(), "extra": True})


def test_golden_success_counts_runtime_inventory_lock_and_layout_stop(tmp_path) -> None:
    request, shard, root, transport, result = _read(tmp_path)
    assert result.request_id == request.request_id
    assert result.manifest_hash == shard.manifest.manifest_hash
    assert result.page_count == request.required_pages == 1
    assert result.row_count == request.expected_row_count == 4
    assert result.attempt_count == len(transport.emitted) == 1
    assert result.actual_total_raw_body_bytes == len(transport.emitted[0].body_bytes)
    assert result.manifest_collection_runtime_us == shard.manifest.collection_runtime_us
    assert result.admission_full_reload_runtime_us >= result.manifest_collection_runtime_us
    assert result.attempt_elapsed_runtime_us == 1_000
    assert result.observed_monotonic_inter_attempt_sleep_us == 0
    assert result.strict_history_namespace_residue_free is True
    assert result.restart_observation_count == 2
    assert result.restart_no_residue_proof.request_state == "complete_verified"
    assert result.restart_no_residue_proof.ready is True
    assert result.restart_no_residue_proof.temp_paths == ()
    assert result.restart_no_residue_proof.unreferenced_attempt_paths == ()
    assert result.restart_no_residue_proof.unreferenced_raw_paths == ()
    assert result.restart_no_residue_proof.alternate_normalized_paths == ()
    assert result.pilot_output_layout_status == PILOT_OUTPUT_LAYOUT_STATUS_UNRESOLVED
    assert result.authority_status == (
        PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE
    )
    assert result.page_accounting == (
        PilotPageAccountingV1(
            page_ordinal=0,
            page_receipt_hash=shard.manifest.page_receipts[0].page_receipt_hash,
            row_count=4,
            attempt_count=1,
        ),
    )
    assert len(result.attempt_accounting) == 1
    assert isinstance(result.attempt_accounting[0], PilotAttemptAccountingV1)
    assert isinstance(result.admission_accounting, PilotAdmissionAccountingV1)
    assert result.writer_lock.status == "present_plain_regular"
    assert all("writer.lock" not in item.relative_path for item in result.physical_files)
    assert sum(item.byte_count for item in result.logical_references) == (
        result.admitted_total_logical_storage_bytes
    )
    assert [item.role for item in result.logical_references].count("scope_marker") == 1
    admission_bytes = next(
        item.byte_count
        for item in result.logical_references
        if item.role == "admission_marker"
    )
    assert sum(
        item.byte_count
        for item in result.logical_references
        if item.role != "admission_marker"
    ) == shard.manifest.logical_storage_bytes
    assert result.admitted_total_logical_storage_bytes == (
        shard.manifest.logical_storage_bytes + admission_bytes
    )
    assert sum(item.byte_count for item in result.physical_files) == (
        result.unique_physical_referenced_bytes
    )
    assert result.admitted_total_logical_storage_bytes == sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    with pytest.raises(StrictHistoryPilotEvidenceLayoutStop, match="non_authoritative"):
        result.require_pilot_compatible()
    with pytest.raises(StrictHistoryPilotEvidenceLayoutStop, match="non_authoritative"):
        require_strict_history_pilot_compatible_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


def test_duplicate_nonempty_raw_cas_is_logically_charged_per_attempt(tmp_path) -> None:
    _request_value, shard, _root, transport, result = _read(
        tmp_path,
        statuses=(503, 503),
    )
    assert len(transport.emitted) == shard.manifest.actual_attempt_count == 3
    assert transport.emitted[0].body_bytes == transport.emitted[1].body_bytes != b""
    assert transport.emitted[0].captured_body_sha256 == (
        transport.emitted[1].captured_body_sha256
    )
    assert result.raw_body_reference_count == result.attempt_count == 3
    assert result.unique_raw_body_count == 2
    assert result.actual_total_raw_body_bytes == sum(
        len(item.body_bytes) for item in transport.emitted
    )
    assert result.unique_physical_raw_body_bytes == sum(
        len(body)
        for body in {
            item.captured_body_sha256: item.body_bytes for item in transport.emitted
        }.values()
    )
    duplicate_raw = [
        item
        for item in result.physical_files
        if item.role == "raw_body" and item.logical_reference_count == 2
    ]
    assert len(duplicate_raw) == 1
    assert result.admitted_total_logical_storage_bytes > (
        result.unique_physical_referenced_bytes
    )


def test_every_logical_attempt_is_reparsed_through_public_parser(
    tmp_path, monkeypatch
) -> None:
    request, shard, root, _transport = _collect(tmp_path, statuses=(503, 503))
    real_parser = evidence_module.parse_http_attempt_evidence_v1
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return real_parser(*args, **kwargs)

    monkeypatch.setattr(evidence_module, "parse_http_attempt_evidence_v1", counted)
    result = read_strict_history_pilot_evidence_v1(
        request=request,
        artifact_root=root,
        expected_manifest_hash=shard.manifest.manifest_hash,
    )
    assert len(calls) == result.attempt_count == 3


@pytest.mark.parametrize("kind", ["admission", "attempt", "raw"])
def test_missing_required_source_is_typed_stop(tmp_path, kind) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    paths = {
        "admission": root / "collections" / request.request_id / "admission.json",
        "attempt": _attempt_paths(shard, root)[0],
        "raw": _raw_paths(shard, root)[0],
    }
    paths[kind].unlink()
    with pytest.raises(StrictHistoryPilotEvidenceStop):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


@pytest.mark.parametrize("kind", ["admission", "attempt", "raw"])
def test_corrupt_required_source_is_typed_stop(tmp_path, kind) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    paths = {
        "admission": root / "collections" / request.request_id / "admission.json",
        "attempt": _attempt_paths(shard, root)[0],
        "raw": _raw_paths(shard, root)[0],
    }
    paths[kind].write_bytes(b"corrupt\n")
    with pytest.raises(StrictHistoryPilotEvidenceStop):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


@pytest.mark.parametrize(
    "residue",
    ["temp", "unreferenced_attempt", "unreferenced_raw", "alternate_normalized"],
)
def test_every_restart_residue_class_is_rejected(tmp_path, residue) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    if residue == "temp":
        (root / ".fixture.tmp").write_bytes(b"temp")
    elif residue == "unreferenced_attempt":
        (root / "attempts" / f"{'a' * 64}.json").write_bytes(b"{}\n")
    elif residue == "unreferenced_raw":
        digest = hashlib.sha256(b"orphan").hexdigest()
        path = root / "raw" / "sha256" / digest[:2] / f"{digest}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"orphan")
    else:
        source = (
            root
            / "normalized"
            / request.request_id
            / f"{shard.manifest.normalized_shard_sha256}.jsonl"
        )
        (source.parent / f"{'b' * 64}.jsonl").write_bytes(source.read_bytes())
    with pytest.raises(StrictHistoryPilotEvidenceResidueStop):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


def test_wrong_detached_manifest_and_stale_request_are_stops(tmp_path) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    with pytest.raises(StrictHistoryPilotEvidenceResidueStop):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash="f" * 64,
        )
    stale = _request(rows=5)
    with pytest.raises(StrictHistoryPilotEvidenceStop):
        read_strict_history_pilot_evidence_v1(
            request=stale,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


def test_runtime_and_monotonic_inter_attempt_sleep_are_derived(tmp_path) -> None:
    _request_value, _shard, _root, transport, result = _read(
        tmp_path,
        statuses=(503, 503),
    )
    expected_sleep = sum(
        current.request_started_monotonic_us - prior.terminal_monotonic_us
        for prior, current in zip(transport.emitted, transport.emitted[1:])
    )
    assert expected_sleep > 0
    assert result.observed_monotonic_inter_attempt_sleep_us == expected_sleep
    assert result.attempt_elapsed_runtime_us == sum(
        item.elapsed_monotonic_us for item in transport.emitted
    )


def test_multiple_pages_preserve_global_attempt_order_and_counts(tmp_path) -> None:
    request = _request(rows=4, page_size=2)
    _request_value, shard, _root, transport, result = _read(
        tmp_path,
        request=request,
    )
    assert result.page_count == request.required_pages == 2
    assert result.row_count == 4
    assert result.attempt_count == len(transport.emitted) == 2
    assert [
        (item.page_ordinal, item.attempt_ordinal)
        for item in result.logical_references
        if item.role == "attempt_receipt"
    ] == [(0, 0), (1, 0)]
    assert result.observed_monotonic_inter_attempt_sleep_us == (
        transport.emitted[1].request_started_monotonic_us
        - transport.emitted[0].terminal_monotonic_us
    )


def test_bounded_reader_rejects_oversize_and_expired_deadline(tmp_path) -> None:
    path = tmp_path / "bounded.bin"
    path.write_bytes(b"1234")
    with pytest.raises(StrictHistoryPilotEvidenceBoundsStop, match="byte_bound"):
        evidence_module._read_bounded_plain_file(
            path,
            max_bytes=3,
            deadline_ns=evidence_module.time.monotonic_ns() + 1_000_000_000,
            missing_code="missing",
        )
    with pytest.raises(StrictHistoryPilotEvidenceBoundsStop, match="runtime"):
        evidence_module._read_bounded_plain_file(
            path,
            max_bytes=4,
            deadline_ns=0,
            missing_code="missing",
        )


def test_bounded_reader_maps_os_read_failure_to_typed_stop(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "read-error.bin"
    path.write_bytes(b"x")

    def fail_read(_descriptor, _amount):
        raise OSError("fixture_read_failure")

    monkeypatch.setattr(evidence_module.os, "read", fail_read)
    with pytest.raises(StrictHistoryPilotEvidenceStop, match="read_io_failed"):
        evidence_module._read_bounded_plain_file(
            path,
            max_bytes=1,
            deadline_ns=evidence_module.time.monotonic_ns() + 1_000_000_000,
            missing_code="missing",
        )


def test_external_hardlink_alias_is_static_hostile_stop(tmp_path) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    admission = root / "collections" / request.request_id / "admission.json"
    alias = tmp_path / "admission-alias.json"
    try:
        os.link(admission, alias)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")
    with pytest.raises(StrictHistoryPilotEvidenceStop, match="hardlink_alias"):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


def test_sibling_writer_lock_is_classified_not_charged_and_hostile_alias_stops(
    tmp_path,
) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    lock = root.parent / f".{root.name}.strict-history-v2.writer.lock"
    alias = tmp_path / "lock-alias"
    try:
        os.link(lock, alias)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")
    with pytest.raises(StrictHistoryPilotEvidenceStop, match="writer_lock"):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


def test_absent_sibling_writer_lock_is_explicit_fact(tmp_path) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    lock = root.parent / f".{root.name}.strict-history-v2.writer.lock"
    lock.unlink()
    result = read_strict_history_pilot_evidence_v1(
        request=request,
        artifact_root=root,
        expected_manifest_hash=shard.manifest.manifest_hash,
    )
    assert result.writer_lock.status == "absent"
    assert result.writer_lock.file_sha256 is None
    assert result.writer_lock.byte_count == 0


def test_reparse_root_alias_is_rejected_when_supported(tmp_path) -> None:
    request, shard, root, _transport = _collect(tmp_path / "real")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(StrictHistoryPilotEvidenceStop) as captured:
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=alias,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )
    assert "reparse" in str(captured.value) or "alias" in str(captured.value)


@pytest.mark.skipif(os.name != "nt", reason="NTFS alternate data streams are Windows-only")
def test_ntfs_named_stream_is_static_hostile_stop(tmp_path) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    admission = root / "collections" / request.request_id / "admission.json"
    try:
        Path(f"{admission}:fixture").write_bytes(b"hidden")
    except OSError as exc:
        pytest.skip(f"named streams unavailable: {exc}")
    with pytest.raises(StrictHistoryPilotEvidenceStop, match="named_data_stream"):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


@pytest.mark.skipif(os.name != "nt", reason="NTFS alternate data streams are Windows-only")
@pytest.mark.parametrize(
    "directory_kind",
    [
        "root_parent",
        "root",
        "attempts",
        "collections",
        "request_collection",
        "raw",
        "raw_sha256",
        "referenced_raw_prefix",
        "extra_raw_prefix",
        "normalized",
        "normalized_request",
    ],
)
def test_ntfs_named_stream_on_required_directory_is_static_hostile_stop(
    tmp_path, directory_kind
) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    raw_prefixes = {path.parent.name for path in _raw_paths(shard, root)}
    unused_prefix = next(
        f"{value:02x}" for value in range(256) if f"{value:02x}" not in raw_prefixes
    )
    directories = {
        "root_parent": root.parent,
        "root": root,
        "attempts": root / "attempts",
        "collections": root / "collections",
        "request_collection": root / "collections" / request.request_id,
        "raw": root / "raw",
        "raw_sha256": root / "raw" / "sha256",
        "referenced_raw_prefix": _raw_paths(shard, root)[0].parent,
        "extra_raw_prefix": root / "raw" / "sha256" / unused_prefix,
        "normalized": root / "normalized",
        "normalized_request": root / "normalized" / request.request_id,
    }
    target = directories[directory_kind]
    target.mkdir(parents=True, exist_ok=True)
    try:
        Path(f"{target}:fixture").write_bytes(b"hidden-directory-stream")
    except OSError as exc:
        pytest.skip(f"directory named streams unavailable: {exc}")
    with pytest.raises(StrictHistoryPilotEvidenceStop, match="named_data_stream"):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


@pytest.mark.skipif(os.name != "nt", reason="NTFS alternate data streams are Windows-only")
def test_late_parent_directory_stream_is_rejected_when_writer_lock_is_absent(
    tmp_path, monkeypatch
) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    lock = root.parent / f".{root.name}.strict-history-v2.writer.lock"
    lock.unlink()
    real_reconcile = StrictHistoryArtifactStoreV2.reconcile_restart
    calls = 0

    def inject_after_final_reconcile(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        report = real_reconcile(self, *args, **kwargs)
        if calls == 2:
            try:
                Path(f"{root.parent}:late-fixture").write_bytes(b"hidden")
            except OSError as exc:
                pytest.skip(f"directory named streams unavailable: {exc}")
        return report

    monkeypatch.setattr(
        StrictHistoryArtifactStoreV2,
        "reconcile_restart",
        inject_after_final_reconcile,
    )
    with pytest.raises(StrictHistoryPilotEvidenceStop, match="named_data_stream"):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


def test_unused_empty_canonical_raw_prefix_is_directory_residue(tmp_path) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    raw_prefixes = {path.parent.name for path in _raw_paths(shard, root)}
    unused_prefix = next(
        f"{value:02x}" for value in range(256) if f"{value:02x}" not in raw_prefixes
    )
    (root / "raw" / "sha256" / unused_prefix).mkdir()
    with pytest.raises(
        StrictHistoryPilotEvidenceResidueStop,
        match="directory_namespace_is_not_exact",
    ):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


@pytest.mark.parametrize(
    "expired_stage", ["first_reconcile", "public_load", "final_reconcile"]
)
def test_public_calls_are_charged_to_one_reader_deadline(
    tmp_path, monkeypatch, expired_stage
) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    expired = False
    real_reconcile = StrictHistoryArtifactStoreV2.reconcile_restart
    real_load = StrictHistoryArtifactStoreV2.load_complete_from_disk
    load_called = False
    reconcile_calls = 0

    def controlled_monotonic_ns():
        return evidence_module._MAX_READER_RUNTIME_US * 1_000 + 1 if expired else 0

    def reconcile(self, *args, **kwargs):
        nonlocal expired, reconcile_calls
        reconcile_calls += 1
        report = real_reconcile(self, *args, **kwargs)
        if expired_stage == "first_reconcile" and reconcile_calls == 1:
            expired = True
        if expired_stage == "final_reconcile" and reconcile_calls == 2:
            expired = True
        return report

    def load(self, *args, **kwargs):
        nonlocal expired, load_called
        load_called = True
        result = real_load(self, *args, **kwargs)
        if expired_stage == "public_load":
            expired = True
        return result

    monkeypatch.setattr(evidence_module.time, "monotonic_ns", controlled_monotonic_ns)
    monkeypatch.setattr(StrictHistoryArtifactStoreV2, "reconcile_restart", reconcile)
    monkeypatch.setattr(StrictHistoryArtifactStoreV2, "load_complete_from_disk", load)
    with pytest.raises(StrictHistoryPilotEvidenceBoundsStop, match="read_runtime"):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )
    assert load_called is (expired_stage != "first_reconcile")
    assert reconcile_calls == (2 if expired_stage == "final_reconcile" else 1)


@pytest.mark.parametrize("failing_reconcile_call", [1, 2])
def test_public_exception_after_deadline_overrun_is_charged_as_bounds_stop(
    tmp_path, monkeypatch, failing_reconcile_call
) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    expired = False
    reconcile_calls = 0
    real_reconcile = StrictHistoryArtifactStoreV2.reconcile_restart

    def controlled_monotonic_ns():
        return evidence_module._MAX_READER_RUNTIME_US * 1_000 + 1 if expired else 0

    def reconcile(self, *args, **kwargs):
        nonlocal expired, reconcile_calls
        reconcile_calls += 1
        report = real_reconcile(self, *args, **kwargs)
        if reconcile_calls == failing_reconcile_call:
            expired = True
            raise OSError("controlled_public_failure_after_overrun")
        return report

    monkeypatch.setattr(evidence_module.time, "monotonic_ns", controlled_monotonic_ns)
    monkeypatch.setattr(StrictHistoryArtifactStoreV2, "reconcile_restart", reconcile)
    with pytest.raises(StrictHistoryPilotEvidenceBoundsStop, match="read_runtime") as caught:
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )
    assert isinstance(caught.value.__cause__, OSError)
    assert reconcile_calls == failing_reconcile_call


def test_roundtrip_work_is_charged_before_live_reader_returns(
    tmp_path, monkeypatch
) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    expired = False
    real_parse = StrictHistoryPilotEvidenceV1.parse.__func__

    def controlled_monotonic_ns():
        return evidence_module._MAX_READER_RUNTIME_US * 1_000 + 1 if expired else 0

    def parse_and_expire(cls, payload):
        nonlocal expired
        result = real_parse(cls, payload)
        expired = True
        return result

    monkeypatch.setattr(evidence_module.time, "monotonic_ns", controlled_monotonic_ns)
    monkeypatch.setattr(
        StrictHistoryPilotEvidenceV1,
        "parse",
        classmethod(parse_and_expire),
    )
    with pytest.raises(StrictHistoryPilotEvidenceBoundsStop, match="read_runtime"):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )


def test_nested_wire_objects_reject_extra_keys(tmp_path) -> None:
    _request_value, _shard, _root, _transport, result = _read(tmp_path)
    payload = result.as_dict()
    payload["logical_references"][0]["extra"] = True  # type: ignore[index]
    with pytest.raises(StrictHistoryPilotEvidenceContractError, match="schema"):
        StrictHistoryPilotEvidenceV1.parse(payload)
    with pytest.raises(StrictHistoryPilotEvidenceContractError):
        PilotLogicalReferenceV1.parse({})


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload: payload.__setitem__("history_contract_hash", "a" * 64),
            "history_contract_hash_mismatch",
        ),
        (
            lambda payload: payload["logical_references"][1].__setitem__(
                "role", "raw_body"
            ),
            "raw_accounting_mismatch",
        ),
        (
            lambda payload: payload["physical_files"][0].__setitem__(
                "file_sha256", "b" * 64
            ),
            "logical_physical_file_binding_mismatch",
        ),
        (
            lambda payload: payload.__setitem__(
                "page_count", payload["page_count"] + 1
            ),
            "positive_success_counts_are_invalid",
        ),
        (
            lambda payload: payload.__setitem__(
                "unique_raw_body_count", payload["unique_raw_body_count"] + 1
            ),
            "raw_accounting_mismatch",
        ),
        (
            lambda payload: payload.__setitem__(
                "row_count", payload["row_count"] + 1
            ),
            "page_accounting_summary_mismatch",
        ),
        (
            lambda payload: payload.__setitem__(
                "admission_full_reload_runtime_us",
                payload["admission_full_reload_runtime_us"] + 1,
            ),
            "admission_derived_scalar_mismatch",
        ),
        (
            lambda payload: payload.__setitem__(
                "attempt_elapsed_runtime_us",
                payload["attempt_elapsed_runtime_us"] + 1,
            ),
            "attempt_derived_scalar_mismatch",
        ),
        (
            lambda payload: payload.__setitem__(
                "observed_monotonic_inter_attempt_sleep_us",
                payload["observed_monotonic_inter_attempt_sleep_us"] + 1,
            ),
            "attempt_derived_scalar_mismatch",
        ),
        (
            lambda payload: payload.__setitem__("authority_status", "authoritative"),
            "authority_status_mismatch",
        ),
        (
            lambda payload: (
                payload.__setitem__("manifest_collection_runtime_us", 0),
                payload.__setitem__("admission_full_reload_runtime_us", 0),
                payload["admission_accounting"].__setitem__(
                    "manifest_collection_runtime_us", 0
                ),
                payload["admission_accounting"].__setitem__(
                    "admission_full_reload_runtime_us", 0
                ),
            ),
            "collection_runtime_accounting_mismatch",
        ),
    ],
)
def test_serialized_fact_cannot_forge_constructor_accounting(
    tmp_path, mutation, error
) -> None:
    _request_value, _shard, _root, _transport, result = _read(tmp_path)
    payload = copy.deepcopy(result.as_dict())
    mutation(payload)
    with pytest.raises(StrictHistoryPilotEvidenceContractError, match=error):
        StrictHistoryPilotEvidenceV1.parse(payload)


def test_coherent_detached_rewrite_remains_explicitly_non_authoritative(
    tmp_path,
) -> None:
    _request_value, _shard, _root, _transport, result = _read(tmp_path)
    payload = copy.deepcopy(result.as_dict())
    payload["row_count"] += 1
    payload["page_accounting"][0]["row_count"] += 1
    rewritten = StrictHistoryPilotEvidenceV1.parse(payload)
    assert rewritten.row_count == result.row_count + 1
    assert rewritten.authority_status == (
        PILOT_EVIDENCE_AUTHORITY_STATUS_NON_AUTHORITATIVE
    )
    assert rewritten.evidence_hash != result.evidence_hash
    with pytest.raises(StrictHistoryPilotEvidenceLayoutStop, match="non_authoritative"):
        rewritten.require_pilot_compatible()


def test_serialized_fact_cannot_forge_raw_content_address_or_lock(tmp_path) -> None:
    _request_value, _shard, _root, _transport, result = _read(tmp_path)
    payload = copy.deepcopy(result.as_dict())
    raw_logical = next(
        item for item in payload["logical_references"] if item["role"] == "raw_body"
    )
    raw_physical = next(
        item for item in payload["physical_files"] if item["role"] == "raw_body"
    )
    raw_logical["file_sha256"] = "c" * 64
    raw_physical["file_sha256"] = "c" * 64
    with pytest.raises(StrictHistoryPilotEvidenceContractError, match="content_hash"):
        StrictHistoryPilotEvidenceV1.parse(payload)

    payload = copy.deepcopy(result.as_dict())
    payload["writer_lock"]["file_sha256"] = "d" * 64
    with pytest.raises(StrictHistoryPilotEvidenceContractError, match="metadata"):
        StrictHistoryPilotEvidenceV1.parse(payload)


def test_final_stable_point_detects_in_process_file_swap(tmp_path, monkeypatch) -> None:
    request, shard, root, _transport = _collect(tmp_path)
    real_reconcile = StrictHistoryArtifactStoreV2.reconcile_restart
    calls = 0

    def swapping(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        report = real_reconcile(self, *args, **kwargs)
        if calls == 2:
            admission = (
                root / "collections" / request.request_id / "admission.json"
            )
            raw = admission.read_bytes()
            admission.write_bytes(raw)
        return report

    monkeypatch.setattr(StrictHistoryArtifactStoreV2, "reconcile_restart", swapping)
    with pytest.raises(StrictHistoryPilotEvidenceStop, match="stable_point"):
        read_strict_history_pilot_evidence_v1(
            request=request,
            artifact_root=root,
            expected_manifest_hash=shard.manifest.manifest_hash,
        )
