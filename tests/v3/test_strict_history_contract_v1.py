from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from trading.market_data.min1_aggregation import (
    Min1ReceiptError,
    aggregate_canonical_min1,
)
from trading.market_data.strict_history import (
    CompleteHistoryShardV1,
    HistoryApiRejectedError,
    HistoryArtifactConflictError,
    HistoryDuplicateTimestampError,
    HistoryHttpStatusError,
    HistoryIncompleteRangeError,
    HistoryJsonDecodeError,
    HistoryPayloadSchemaError,
    HistoryPayloadValueError,
    HistoryRangeContractError,
    HistoryStorageError,
    HistoryTransportError,
    HistoryNetworkError,
    HistoryTimeoutError,
    HistoryRangeRequestV1,
    KlinePageRequestV1,
    RawHttpResponseV1,
    StrictHistoryArtifactStoreV1,
    StrictMexcHistoryCollectorV1,
    TransportFailureReceiptV1,
    strict_history_contract_hash,
)


BASE = 1_767_225_600  # 2026-01-01T00:00:00Z


def _request(
    rows: int = 5,
    *,
    interval: str = "Min1",
    page_size: int = 2_000,
    max_pages: int = 200,
    max_attempts_per_page: int = 1,
) -> HistoryRangeRequestV1:
    step = {"Min1": 60, "Min5": 300}[interval]
    end = BASE + rows * step
    return HistoryRangeRequestV1(
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        interval=interval,
        start_open_ts=BASE,
        end_open_ts_exclusive=end,
        collection_as_of_ts=float(end),
        endpoint_identity="mexc_futures_kline_fixture_v1",
        page_size=page_size,
        max_pages=max_pages,
        max_attempts_per_page=max_attempts_per_page,
    )


def _page_payload(
    request: KlinePageRequestV1,
    *,
    order: list[int] | None = None,
    missing: set[int] | None = None,
    override: dict[str, object] | None = None,
) -> bytes:
    timestamps = list(request.expected_timestamps())
    if missing:
        timestamps = [value for value in timestamps if value not in missing]
    if order is not None:
        timestamps = [timestamps[index] for index in order]
    data: dict[str, object] = {
        "time": timestamps,
        "open": ["100.00" for _ in timestamps],
        "high": [103 for _ in timestamps],
        "low": ["99" for _ in timestamps],
        "close": ["102.0" for _ in timestamps],
        "vol": [10 for _ in timestamps],
        "amount": ["1000.50" for _ in timestamps],
    }
    if override:
        data.update(override)
    return json.dumps(
        {"success": True, "code": 0, "data": data},
        separators=(",", ":"),
    ).encode("utf-8")


class _Transport:
    def __init__(self, factory):
        self.factory = factory
        self.requests: list[KlinePageRequestV1] = []

    def fetch_page(
        self, request: KlinePageRequestV1, *, attempt_ordinal: int
    ) -> RawHttpResponseV1:
        self.requests.append(request)
        result = self.factory(request)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, RawHttpResponseV1):
            return result
        ordinal = request.page_ordinal
        return RawHttpResponseV1(
            page_request=request,
            request_started_at=1_900_000_000.0 + ordinal,
            received_at=1_900_000_001.0 + ordinal,
            http_status=200,
            body=result,
            safe_headers=(("content-type", "application/json"),),
            attempt_ordinal=attempt_ordinal,
        )


def _collector(tmp_path: Path, transport: _Transport):
    store = StrictHistoryArtifactStoreV1(tmp_path / "strict-pilot")
    return StrictMexcHistoryCollectorV1(transport=transport, store=store), store


def _collect(tmp_path: Path, request: HistoryRangeRequestV1 | None = None):
    request = request or _request()
    transport = _Transport(_page_payload)
    collector, store = _collector(tmp_path, transport)
    return collector.collect_range(request), transport, store


def test_contract_hash_is_pinned() -> None:
    assert strict_history_contract_hash() == (
        "6c17bd9de3e25210139da4491a1f35fbd0cec557707fb5d376a60ce23e04c6c1"
    )


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"start_open_ts": BASE + 1}, "history_range_is_not_utc_aligned"),
        ({"end_open_ts_exclusive": BASE}, "history_range_must_be_nonempty"),
        ({"collection_as_of_ts": BASE + 299.0}, "history_range_contains_unclosed_bar"),
        ({"page_size": 2_001}, "history_page_size_is_out_of_range"),
        ({"max_pages": 0}, "history_max_pages_is_out_of_range"),
    ],
)
def test_range_preflight_fails_before_transport(changes, error) -> None:
    values = {
        "venue": "mexc_contract",
        "symbol": "BTCUSDT",
        "venue_symbol": "BTC_USDT",
        "interval": "Min1",
        "start_open_ts": BASE,
        "end_open_ts_exclusive": BASE + 300,
        "collection_as_of_ts": float(BASE + 300),
        "endpoint_identity": "mexc_futures_kline_fixture_v1",
    }
    values.update(changes)
    with pytest.raises(HistoryRangeContractError, match=error):
        HistoryRangeRequestV1(**values)


@pytest.mark.parametrize(
    "rows, expected_pages, last_count",
    [(1_999, 1, 1_999), (2_000, 1, 2_000), (2_001, 2, 1)],
)
def test_page_plan_is_exact_nonoverlapping_grid(rows, expected_pages, last_count) -> None:
    request = _request(rows)
    pages = StrictMexcHistoryCollectorV1.plan_pages(request)
    assert len(pages) == expected_pages
    assert pages[-1].expected_row_count == last_count
    flattened = tuple(value for page in pages for value in page.expected_timestamps())
    assert flattened == request.expected_timestamps()
    assert pages[-1].end_open_ts_inclusive == request.end_open_ts_exclusive - 60


def test_140_day_min1_requires_101_pages_and_budget_failure_is_explicit() -> None:
    request = _request(140 * 24 * 60, max_pages=100)
    assert request.required_pages == 101
    with pytest.raises(HistoryIncompleteRangeError) as exc:
        StrictMexcHistoryCollectorV1.plan_pages(request)
    assert exc.value.reason == "page_budget_exceeded"


def test_golden_page_preserves_exact_amount_and_publishes_manifest_last(tmp_path) -> None:
    shard, transport, store = _collect(tmp_path)
    assert isinstance(shard, CompleteHistoryShardV1)
    assert len(transport.requests) == 1
    assert [row.bar_open_ts for row in shard.rows] == list(_request().expected_timestamps())
    assert {row.volume_contracts for row in shard.rows} == {"10"}
    assert {row.turnover_quote for row in shard.rows} == {"1000.5"}
    assert shard.rows[0].source_raw_body_sha256 == shard.manifest.page_receipts[0].raw_body_sha256
    assert shard.manifest.expected_row_count == shard.manifest.actual_row_count == 5
    assert store.has_complete_manifest(shard.manifest.request.request_id)
    store.verify_complete_artifacts(shard)
    frame = shard.to_frame()
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert str(frame.index.tz) == "UTC"
    assert frame.iloc[0].to_dict() == {
        "open": 100.0,
        "high": 103.0,
        "low": 99.0,
        "close": 102.0,
        "volume": 10.0,
        "turnover": 1000.5,
    }


def test_two_page_collection_is_grid_driven_and_deterministic(tmp_path) -> None:
    request = _request(2_001)
    shard, transport, _store = _collect(tmp_path, request)
    assert [page.expected_row_count for page in transport.requests] == [2_000, 1]
    assert len(shard.rows) == 2_001
    assert tuple(row.bar_open_ts for row in shard.rows) == request.expected_timestamps()


def test_raw_order_and_whitespace_change_raw_hash_not_logical_hash(tmp_path) -> None:
    request = _request()
    forward = _Transport(_page_payload)
    reverse = _Transport(
        lambda page: json.dumps(
            json.loads(_page_payload(page, order=list(reversed(range(5))))),
            indent=2,
        ).encode("utf-8")
    )
    collector_a, _ = _collector(tmp_path / "a", forward)
    collector_b, _ = _collector(tmp_path / "b", reverse)
    shard_a = collector_a.collect_range(request)
    shard_b = collector_b.collect_range(request)
    assert shard_a.manifest.normalized_logical_hash == shard_b.manifest.normalized_logical_hash
    assert shard_a.manifest.page_receipts[0].raw_body_sha256 != shard_b.manifest.page_receipts[0].raw_body_sha256
    assert [row.source_row_ordinal for row in shard_b.rows] == [4, 3, 2, 1, 0]


@pytest.mark.parametrize(
    "factory, error_type",
    [
        (lambda page: replace(RawHttpResponseV1(
            page_request=page,
            request_started_at=float(page.end_open_ts_inclusive + 61),
            received_at=float(page.end_open_ts_inclusive + 62),
            http_status=200,
            body=b"{}",
        ), http_status=503), HistoryHttpStatusError),
        (lambda _page: b"not-json", HistoryJsonDecodeError),
        (lambda _page: b'{"success":false,"code":500}', HistoryApiRejectedError),
    ],
)
def test_http_json_and_api_failures_are_typed_and_raw_is_retained(
    tmp_path, factory, error_type
) -> None:
    transport = _Transport(factory)
    collector, store = _collector(tmp_path, transport)
    with pytest.raises(error_type):
        collector.collect_range(_request())
    assert list((store.root / "raw").rglob("*.bin"))
    assert list((store.root / "attempts").glob("*.json"))
    assert not (store.root / "collections").exists()


@pytest.mark.parametrize(
    "error_type, outcome", [(HistoryNetworkError, "network_error"), (HistoryTimeoutError, "timeout")]
)
def test_transport_failures_never_become_empty_data(tmp_path, error_type, outcome) -> None:
    class FailureTransport:
        def fetch_page(self, page, *, attempt_ordinal):
            receipt = TransportFailureReceiptV1(
                page_request=page,
                attempt_ordinal=attempt_ordinal,
                request_started_at=1_900_000_000.0,
                failed_at=1_900_000_001.0,
                outcome=outcome,
                safe_error_code=f"fixture_{outcome}",
            )
            raise error_type(receipt)

    transport = FailureTransport()
    collector, store = _collector(tmp_path, transport)
    with pytest.raises(error_type):
        collector.collect_range(_request())
    assert list((store.root / "attempts").glob("*.json"))
    assert not (store.root / "collections").exists()


def test_retry_chain_persists_every_attempt_and_is_chronological(tmp_path) -> None:
    class RetryTransport:
        def fetch_page(self, page, *, attempt_ordinal):
            if attempt_ordinal == 0:
                failure = TransportFailureReceiptV1(
                    page_request=page,
                    attempt_ordinal=0,
                    request_started_at=1_900_000_000.0,
                    failed_at=1_900_000_001.0,
                    outcome="timeout",
                    safe_error_code="fixture_timeout",
                )
                raise HistoryTimeoutError(failure)
            return RawHttpResponseV1(
                page_request=page,
                request_started_at=1_900_000_001.0,
                received_at=1_900_000_002.0,
                http_status=200,
                body=_page_payload(page),
                attempt_ordinal=attempt_ordinal,
            )

    request = _request(max_attempts_per_page=2)
    collector, store = _collector(tmp_path, RetryTransport())
    shard = collector.collect_range(request)
    assert len(shard.manifest.page_receipts[0].attempt_receipt_hashes) == 2
    assert len(list((store.root / "attempts").glob("*.json"))) == 2
    store.verify_complete_artifacts(shard)


def test_retryable_http_response_is_retained_before_success(tmp_path) -> None:
    class HttpRetryTransport:
        def fetch_page(self, page, *, attempt_ordinal):
            return RawHttpResponseV1(
                page_request=page,
                request_started_at=1_900_000_000.0 + attempt_ordinal,
                received_at=1_900_000_001.0 + attempt_ordinal,
                http_status=503 if attempt_ordinal == 0 else 200,
                body=(
                    b'{"success":false,"code":503}'
                    if attempt_ordinal == 0
                    else _page_payload(page)
                ),
                attempt_ordinal=attempt_ordinal,
            )

    collector, store = _collector(tmp_path, HttpRetryTransport())
    shard = collector.collect_range(_request(max_attempts_per_page=2))
    receipt = shard.manifest.page_receipts[0]
    assert len(receipt.attempt_receipt_hashes) == 2
    assert len(list((store.root / "raw").rglob("*.bin"))) == 2
    store.verify_complete_artifacts(shard)


def test_retry_timing_regression_fails_closed(tmp_path) -> None:
    class BackwardRetryTransport:
        def fetch_page(self, page, *, attempt_ordinal):
            if attempt_ordinal == 0:
                failure = TransportFailureReceiptV1(
                    page_request=page,
                    attempt_ordinal=0,
                    request_started_at=1_900_000_000.0,
                    failed_at=1_900_000_002.0,
                    outcome="network_error",
                    safe_error_code="fixture_network_error",
                )
                raise HistoryNetworkError(failure)
            return RawHttpResponseV1(
                page_request=page,
                request_started_at=1_900_000_001.0,
                received_at=1_900_000_003.0,
                http_status=200,
                body=_page_payload(page),
                attempt_ordinal=attempt_ordinal,
            )

    collector, store = _collector(tmp_path, BackwardRetryTransport())
    with pytest.raises(HistoryTransportError, match="history_attempt_timing_regressed"):
        collector.collect_range(_request(max_attempts_per_page=2))
    assert not (store.root / "collections").exists()


def test_successful_empty_and_internal_missing_bar_are_explicit_incomplete(tmp_path) -> None:
    empty = _Transport(
        lambda page: _page_payload(page, missing=set(page.expected_timestamps()))
    )
    collector, store = _collector(tmp_path / "empty", empty)
    with pytest.raises(HistoryIncompleteRangeError) as exc:
        collector.collect_range(_request())
    assert exc.value.reason == "empty_success"
    assert len(exc.value.missing_timestamps) == 5
    assert not store.has_complete_manifest(_request().request_id)

    missing_ts = BASE + 120
    missing = _Transport(lambda page: _page_payload(page, missing={missing_ts}))
    collector, store = _collector(tmp_path / "missing", missing)
    with pytest.raises(HistoryIncompleteRangeError) as exc:
        collector.collect_range(_request())
    assert exc.value.reason == "missing_timestamps"
    assert exc.value.missing_timestamps == (missing_ts,)
    assert not store.has_complete_manifest(_request().request_id)


@pytest.mark.parametrize(
    "override, error_type",
    [
        ({"amount": None}, HistoryPayloadSchemaError),
        ({"amount": [1]}, HistoryPayloadSchemaError),
        ({"time": [BASE, BASE, BASE + 120, BASE + 180, BASE + 240]}, HistoryDuplicateTimestampError),
        ({"time": [BASE + 1, BASE + 60, BASE + 120, BASE + 180, BASE + 240]}, HistoryIncompleteRangeError),
        ({"open": [True] * 5}, HistoryPayloadValueError),
        ({"open": ["NaN"] * 5}, HistoryPayloadValueError),
        ({"high": [98] * 5}, HistoryPayloadValueError),
        ({"vol": [-1] * 5}, HistoryPayloadValueError),
        ({"amount": [-1] * 5}, HistoryPayloadValueError),
    ],
)
def test_malformed_payloads_cannot_be_coerced_dropped_or_deduplicated(
    tmp_path, override, error_type
) -> None:
    transport = _Transport(lambda page: _page_payload(page, override=override))
    collector, _store = _collector(tmp_path, transport)
    with pytest.raises(error_type):
        collector.collect_range(_request())


def test_success_true_with_nonzero_api_code_is_rejected(tmp_path) -> None:
    transport = _Transport(
        lambda page: json.dumps(
            {**json.loads(_page_payload(page)), "code": 500},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    collector, _store = _collector(tmp_path, transport)
    with pytest.raises(HistoryApiRejectedError):
        collector.collect_range(_request())


def test_decimal_normalization_is_context_independent_beyond_28_digits(tmp_path) -> None:
    first = "1234567890123456789012345678901234567890.123456789"
    second = "1234567890123456789012345678901234567890.123456788"

    def collect_amount(root, amount):
        transport = _Transport(
            lambda page: _page_payload(page, override={"amount": [amount] * 5})
        )
        collector, _store = _collector(root, transport)
        return collector.collect_range(_request())

    shard_a = collect_amount(tmp_path / "a", first)
    shard_b = collect_amount(tmp_path / "b", second)
    assert shard_a.rows[0].turnover_quote == first
    assert shard_b.rows[0].turnover_quote == second
    assert shard_a.manifest.normalized_logical_hash != shard_b.manifest.normalized_logical_hash


def test_identical_replay_is_idempotent_but_different_evidence_conflicts(tmp_path) -> None:
    request = _request()
    transport = _Transport(_page_payload)
    collector, store = _collector(tmp_path, transport)
    first = collector.collect_range(request)
    second = collector.collect_range(request)
    assert first.manifest.manifest_hash == second.manifest.manifest_hash
    store.verify_complete_artifacts(second)

    different = _Transport(
        lambda page: json.dumps(json.loads(_page_payload(page)), indent=1).encode("utf-8")
    )
    conflict = StrictMexcHistoryCollectorV1(transport=different, store=store)
    with pytest.raises(HistoryArtifactConflictError):
        conflict.collect_range(request)


def test_publish_failure_leaves_no_completed_marker(tmp_path, monkeypatch) -> None:
    transport = _Transport(_page_payload)
    collector, store = _collector(tmp_path, transport)
    original = store._publish_immutable

    def fail_manifest(relative, payload):
        if Path(relative).name == "manifest.json":
            raise HistoryStorageError("injected_manifest_failure")
        return original(relative, payload)

    monkeypatch.setattr(store, "_publish_immutable", fail_manifest)
    request = _request()
    with pytest.raises(HistoryStorageError, match="injected_manifest_failure"):
        collector.collect_range(request)
    assert not store.has_complete_manifest(request.request_id)


def test_source_deletion_before_publish_cannot_create_success_marker(
    tmp_path, monkeypatch
) -> None:
    transport = _Transport(_page_payload)
    collector, store = _collector(tmp_path, transport)
    original = store.publish_complete

    def delete_source_then_publish(shard):
        raw_hash = shard.manifest.page_receipts[0].raw_body_sha256
        raw_path = (
            store.root / "raw" / "sha256" / raw_hash[:2] / f"{raw_hash}.bin"
        )
        raw_path.unlink()
        return original(shard)

    monkeypatch.setattr(store, "publish_complete", delete_source_then_publish)
    request = _request()
    with pytest.raises(HistoryStorageError, match="raw_artifact_is_missing"):
        collector.collect_range(request)
    assert not store.has_complete_manifest(request.request_id)


def test_verifier_rejects_modified_raw_bytes(tmp_path) -> None:
    shard, _transport, store = _collect(tmp_path)
    receipt = shard.manifest.page_receipts[0]
    raw_path = (
        store.root
        / "raw"
        / "sha256"
        / receipt.raw_body_sha256[:2]
        / f"{receipt.raw_body_sha256}.bin"
    )
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    with pytest.raises(HistoryStorageError, match="history_raw_artifact_hash_mismatch"):
        store.verify_complete_artifacts(shard)


def test_verifier_rejects_modified_attempt_receipt(tmp_path) -> None:
    shard, _transport, store = _collect(tmp_path)
    attempt_hash = shard.manifest.page_receipts[0].attempt_receipt_hashes[0]
    attempt_path = store.root / "attempts" / f"{attempt_hash}.json"
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    payload["request_started_at"] += 1.0
    attempt_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(HistoryStorageError, match="attempt_artifact_hash_mismatch"):
        store.verify_complete_artifacts(shard)


def test_completed_contract_rejects_mutable_list_containers(tmp_path) -> None:
    shard, _transport, _store = _collect(tmp_path)
    page = shard.manifest.page_receipts[0]
    with pytest.raises(HistoryPayloadValueError, match="attempts_are_missing"):
        replace(page, attempt_receipt_hashes=list(page.attempt_receipt_hashes))
    with pytest.raises(HistoryPayloadValueError, match="not_immutable"):
        replace(
            shard.manifest,
            page_receipts=list(shard.manifest.page_receipts),
        )
    with pytest.raises(HistoryPayloadValueError, match="not_immutable"):
        CompleteHistoryShardV1(rows=list(shard.rows), manifest=shard.manifest)


def test_store_refuses_legacy_data_history_path(tmp_path) -> None:
    with pytest.raises(HistoryStorageError, match="legacy_history_root_is_forbidden"):
        StrictHistoryArtifactStoreV1(tmp_path / "data" / "history")
    with pytest.raises(HistoryStorageError, match="legacy_history_root_is_forbidden"):
        StrictHistoryArtifactStoreV1(
            tmp_path / "data" / "history" / "strict-pilot"
        )


@pytest.mark.parametrize("header", ["x-api-key", "proxy-authorization", "authorization"])
def test_raw_receipt_rejects_non_allowlisted_or_secret_headers(header) -> None:
    page = StrictMexcHistoryCollectorV1.plan_pages(_request())[0]
    with pytest.raises(HistoryRangeContractError, match="headers_are_invalid"):
        RawHttpResponseV1(
            page_request=page,
            request_started_at=1_900_000_000.0,
            received_at=1_900_000_001.0,
            http_status=200,
            body=_page_payload(page),
            safe_headers=((header, "must-not-persist"),),
        )


def test_complete_min1_shard_maps_explicitly_into_s3_receipts(tmp_path) -> None:
    shard, _transport, _store = _collect(tmp_path)
    frame, receipts = shard.to_min1_aggregation_inputs()
    assert all(
        receipt.source_lineage_hash == shard.manifest.manifest_hash
        for receipt in receipts
    )
    aggregated = aggregate_canonical_min1(
        frame,
        target_timeframe="Min5",
        receipts=receipts,
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
    )
    assert len(aggregated.bars) == 1
    assert aggregated.bars[0].volume == 50.0
    assert aggregated.bars[0].turnover == 5002.5

    mutated = frame.copy()
    mutated.iloc[0, mutated.columns.get_loc("close")] = 101.0
    with pytest.raises(Min1ReceiptError, match="normalized_row_hash"):
        aggregate_canonical_min1(
            mutated,
            target_timeframe="Min5",
            receipts=receipts,
            venue="mexc_contract",
            symbol="BTCUSDT",
            venue_symbol="BTC_USDT",
        )
