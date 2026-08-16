from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from trading.market_data.mexc_futures_transport import (
    EndpointContractError,
    HistoryResourceLimitsV1,
    IncompleteHttpAttemptEvidenceV1,
    MexcFuturesRawTransportV1,
    ResourceBudgetExceededError,
    ResourceLimitContractError,
    RetryAfterContractError,
    StreamingExecutorNetworkError,
    StreamingExecutorTimeoutError,
    TransportContractError,
    candidate_endpoint_fixture_path,
    candidate_history_resource_limits_v1,
    candidate_history_retry_policy_v1,
    canonicalize_public_response_headers,
    load_mexc_futures_endpoint_contract_v1,
    mexc_futures_transport_contract_hash,
    parse_http_attempt_evidence_v1,
    retry_after_delay_us,
)
from trading.market_data.strict_history import KlinePageRequestV1


BASE = 1_767_225_600


class _Clock:
    def __init__(self, epoch_us: int = 1_900_000_000_000_000):
        self.epoch = epoch_us
        self.monotonic = 10_000_000
        self.sleeps: list[int] = []

    def epoch_us(self) -> int:
        return self.epoch

    def monotonic_us(self) -> int:
        return self.monotonic

    def sleep_us(self, duration_us: int) -> None:
        assert type(duration_us) is int and duration_us >= 0
        self.sleeps.append(duration_us)
        self.epoch += duration_us
        self.monotonic += duration_us

    def advance(self, duration_us: int) -> None:
        self.epoch += duration_us
        self.monotonic += duration_us


class _Response:
    def __init__(
        self,
        status: int,
        chunks,
        *,
        headers=(),
        clock: _Clock | None = None,
        advance_per_chunk_us: int = 0,
        close_advance_us: int = 0,
        close_error: Exception | None = None,
    ):
        self._status = status
        self._headers = tuple(headers)
        self.chunks = list(chunks)
        self.clock = clock
        self.advance_per_chunk_us = advance_per_chunk_us
        self.close_advance_us = close_advance_us
        self.close_error = close_error
        self.yielded = 0
        self.closed = False

    @property
    def http_status(self):
        return self._status

    @property
    def headers(self):
        return self._headers

    def iter_body(self, chunk_size: int):
        assert chunk_size == 64 * 1024
        for item in self.chunks:
            if isinstance(item, Exception):
                raise item
            self.yielded += 1
            if self.clock is not None and self.advance_per_chunk_us:
                self.clock.advance(self.advance_per_chunk_us)
            yield item

    def close(self):
        self.closed = True
        if self.clock is not None and self.close_advance_us:
            self.clock.advance(self.close_advance_us)
        if self.close_error is not None:
            raise self.close_error


class _Executor:
    def __init__(self, *results):
        self.results = list(results)
        self.requests = []

    def open(self, request, *, connect_timeout_us, read_timeout_us):
        self.requests.append((request, connect_timeout_us, read_timeout_us))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _endpoint():
    return load_mexc_futures_endpoint_contract_v1(
        candidate_endpoint_fixture_path()
    )


def _page(*, ordinal: int = 0, endpoint_identity: str | None = None):
    endpoint_identity = endpoint_identity or _endpoint().endpoint_identity
    start = BASE + ordinal * 60
    return KlinePageRequestV1(
        range_request_id="a" * 64,
        endpoint_identity=endpoint_identity,
        venue_symbol="BTC_USDT",
        interval="Min1",
        page_ordinal=ordinal,
        start_open_ts=start,
        end_open_ts_inclusive=start,
        expected_row_count=1,
    )


def _transport(executor, *, clock=None, limits=None, retry=None):
    return MexcFuturesRawTransportV1(
        endpoint=_endpoint(),
        resource_limits=limits or candidate_history_resource_limits_v1(),
        retry_policy=retry or candidate_history_retry_policy_v1(),
        executor=executor,
        clock=clock or _Clock(),
    )


def test_candidate_fixture_is_canonical_unverified_and_hash_pinned() -> None:
    path = candidate_endpoint_fixture_path()
    raw = path.read_bytes()
    payload = json.loads(raw)
    assert raw == json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode() + b"\n"
    endpoint = _endpoint()
    assert endpoint.verification_status == "candidate_not_u5_verified"
    assert endpoint.current_official_docs_verified is False
    assert endpoint.live_endpoint_verified is False
    assert endpoint.host == "api.mexc.com"
    assert endpoint.contract_hash == (
        "54f57d755cd679eb92444d48b38013621caad37067125e80cf7c5e45fe2ab220"
    )
    assert endpoint.endpoint_identity == (
        "mexc_futures_kline_candidate_v1."
        "54f57d755cd679eb92444d48b38013621caad37067125e80cf7c5e45fe2ab220"
    )


def test_fixture_mutation_cannot_keep_candidate_identity(tmp_path) -> None:
    payload = json.loads(candidate_endpoint_fixture_path().read_text())
    payload["host"] = "contract.mexc.com"
    mutated = tmp_path / "endpoint.json"
    mutated.write_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    with pytest.raises(EndpointContractError, match="authority_mismatch"):
        load_mexc_futures_endpoint_contract_v1(mutated)


def test_prepared_request_has_exact_public_no_secret_semantics() -> None:
    prepared = _endpoint().prepare(_page())
    assert prepared.url == (
        "https://api.mexc.com/api/v1/contract/kline/BTC_USDT"
        f"?interval=Min1&start={BASE}&end={BASE}"
    )
    assert prepared.headers == (
        ("accept", "application/json"),
        ("accept-encoding", "identity"),
        ("user-agent", "koteika-strict-history/1.0"),
    )
    assert prepared.tls_verify is True
    assert prepared.allow_redirects is False
    assert prepared.trust_env is False
    assert prepared.body is None
    with pytest.raises(EndpointContractError, match="identity_mismatch"):
        _endpoint().prepare(_page(endpoint_identity="wrong.fixture.v1"))


def test_resource_retry_and_transport_contract_hashes_are_pinned() -> None:
    limits = candidate_history_resource_limits_v1()
    retry = candidate_history_retry_policy_v1()
    assert limits.contract_hash == (
        "937d053e33c513d128389259e308156c8758e5cfe44b5849e3eb27ea49d96bdc"
    )
    assert retry.contract_hash == (
        "78f92d14cc26ead1a372d840a05fe8a60dae97d5d9a3cdacc539a098194a2cc9"
    )
    assert mexc_futures_transport_contract_hash() == (
        "7d3bd40c6753e7bda2f1904ce2ffa2ff55770ecce9ba6d5614d2b30ae0664d22"
    )
    assert HistoryResourceLimitsV1.from_dict(limits.as_dict()) == limits
    assert type(limits.as_dict()["max_collection_runtime_us"]) is int
    limits.validate_request_shape(
        required_pages=101,
        expected_rows=201_600,
        max_attempts_per_page=3,
    )


def test_transport_instance_is_latched_to_one_range_and_attempt_coordinate() -> None:
    first_response = _Response(200, [b"ok"])
    executor = _Executor(first_response)
    transport = _transport(executor)
    page = _page()
    transport.fetch_page(page, attempt_ordinal=0)

    with pytest.raises(TransportContractError, match="range_request_mismatch"):
        transport.fetch_page(
            replace(page, range_request_id="2" * 64),
            attempt_ordinal=0,
        )
    with pytest.raises(TransportContractError, match="attempt_coordinate_reused"):
        transport.fetch_page(page, attempt_ordinal=0)
    assert len(executor.requests) == 1


@pytest.mark.parametrize(
    "change, error",
    [
        ({"max_pages": 201}, "max_pages_exceeds_hard_cap"),
        ({"max_rows": 400_001}, "max_rows_exceeds_hard_cap"),
        ({"max_attempts_per_page": 11}, "max_attempts_per_page_exceeds_hard_cap"),
        ({"max_pages": 1.0}, "max_pages_is_invalid"),
    ],
)
def test_resource_hard_caps_reject_before_transport(change, error) -> None:
    values = candidate_history_resource_limits_v1().as_dict()
    values.update(change)
    with pytest.raises((ResourceLimitContractError, TransportContractError), match=error):
        HistoryResourceLimitsV1.from_dict(values)


def test_public_headers_canonicalize_date_etag_and_drop_credentials() -> None:
    date = "Sat, 15 Aug 2026 12:34:56 GMT"
    epoch = int(datetime(2026, 8, 15, 12, 34, 56, tzinfo=timezone.utc).timestamp())
    headers = canonicalize_public_response_headers(
        (
            ("ETag", 'W/"abc-123"'),
            ("Date", date),
            ("Authorization", "must-not-persist"),
            ("Set-Cookie", "must-not-persist"),
            ("X-API-Key", "must-not-persist"),
        )
    )
    assert headers == (
        ("date", f"unix={epoch}"),
        ("etag", "weak.YWJjLTEyMw"),
    )
    assert canonicalize_public_response_headers(headers) == headers


def test_pre_epoch_http_date_is_stably_invalid_and_attempt_is_still_evidence() -> None:
    raw = "Wed, 31 Dec 1969 23:59:59 GMT"
    headers = canonicalize_public_response_headers((("Last-Modified", raw),))
    assert headers[0][1].startswith("invalid.")
    assert canonicalize_public_response_headers(headers) == headers

    response = _Response(200, [b"ok"], headers=(("Last-Modified", raw),))
    evidence = _transport(_Executor(response)).fetch_page(_page(), attempt_ordinal=0)
    assert evidence.body_complete is True
    assert evidence.safe_headers == headers
    assert response.closed is True


def test_complete_attempt_is_exact_reconstructable_evidence() -> None:
    body = b'{"success":true}'
    response = _Response(
        200,
        [body[:5], body[5:]],
        headers=(
            ("Content-Length", str(len(body))),
            ("Content-Type", "application/json"),
            ("ETag", '"abc"'),
            ("Cookie", "must-not-persist"),
        ),
    )
    executor = _Executor(response)
    evidence = _transport(executor).fetch_page(_page(), attempt_ordinal=0)
    assert evidence.body_complete is True
    assert evidence.body_bytes == body
    assert evidence.outcome == "complete"
    assert evidence.safe_headers == (
        ("content-length", str(len(body))),
        ("content-type", "application/json"),
        ("etag", "strong.YWJj"),
    )
    assert response.closed is True
    restored = parse_http_attempt_evidence_v1(
        evidence.receipt_dict(), page_request=_page(), body_bytes=body
    )
    assert restored == evidence
    assert restored.attempt_receipt_hash == evidence.attempt_receipt_hash
    prepared, connect_timeout, read_timeout = executor.requests[0]
    assert prepared.trust_env is False and prepared.allow_redirects is False
    assert (connect_timeout, read_timeout) == (5_000_000, 10_000_000)


def test_streaming_stops_at_declared_cap_plus_one() -> None:
    limits = replace(
        candidate_history_resource_limits_v1(),
        max_raw_body_bytes_per_attempt=4,
    )
    response = _Response(200, [b"abc", b"def", b"must-not-be-read"])
    evidence = _transport(_Executor(response), limits=limits).fetch_page(
        _page(), attempt_ordinal=0
    )
    assert isinstance(evidence, IncompleteHttpAttemptEvidenceV1)
    assert evidence.outcome == "body_limit_exceeded"
    assert evidence.safe_error_code == "attempt_body_limit_exceeded"
    assert evidence.body_bytes == b"abcd"
    assert evidence.captured_body_length == 4
    assert response.yielded == 2
    assert response.closed is True

    exact = _Response(200, [b"ab", b"cd"])
    complete = _transport(_Executor(exact), limits=limits).fetch_page(
        _page(), attempt_ordinal=0
    )
    assert complete.body_complete is True
    assert complete.body_bytes == b"abcd"


@pytest.mark.parametrize(
    "failure,outcome,code",
    [
        (StreamingExecutorNetworkError("fixture_dns_error"), "network_error", "fixture_dns_error"),
        (StreamingExecutorTimeoutError("fixture_connect_timeout"), "timeout", "fixture_connect_timeout"),
    ],
)
def test_preheader_network_and_timeout_have_bodyless_evidence(failure, outcome, code) -> None:
    evidence = _transport(_Executor(failure)).fetch_page(_page(), attempt_ordinal=0)
    assert isinstance(evidence, IncompleteHttpAttemptEvidenceV1)
    assert evidence.outcome == outcome
    assert evidence.safe_error_code == code
    assert evidence.http_status is None
    assert evidence.headers_received_at_us is None
    assert evidence.body_bytes == b""


@pytest.mark.parametrize(
    "result,code",
    [
        (RuntimeError("must-not-escape"), "executor_open_error"),
        (None, "executor_response_missing"),
        (object(), "executor_response_close_missing"),
    ],
)
def test_started_attempt_with_invalid_executor_result_is_typed_evidence(
    result, code
) -> None:
    evidence = _transport(_Executor(result)).fetch_page(_page(), attempt_ordinal=0)
    assert isinstance(evidence, IncompleteHttpAttemptEvidenceV1)
    assert evidence.outcome == "network_error"
    assert evidence.safe_error_code == code
    assert evidence.http_status is None
    assert evidence.body_bytes == b""


def test_response_without_body_iterator_is_closed_and_typed() -> None:
    class NoIterator:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    response = NoIterator()
    evidence = _transport(_Executor(response)).fetch_page(_page(), attempt_ordinal=0)
    assert response.closed is True
    assert evidence.outcome == "network_error"
    assert evidence.safe_error_code == "executor_response_body_iterator_missing"


def test_postheader_timeout_retains_partial_body() -> None:
    response = _Response(
        200,
        [b"partial", StreamingExecutorTimeoutError("fixture_read_timeout")],
        headers=(("Content-Type", "application/json"),),
    )
    evidence = _transport(_Executor(response)).fetch_page(_page(), attempt_ordinal=0)
    assert evidence.outcome == "timeout"
    assert evidence.safe_error_code == "fixture_read_timeout"
    assert evidence.http_status == 200
    assert evidence.body_bytes == b"partial"
    assert response.closed is True


def test_fake_clock_applies_retry_after_and_exponential_backoff() -> None:
    clock = _Clock()
    first = _Response(
        503,
        [b"bad"],
        headers=(("Content-Length", "3"), ("Retry-After", "2")),
    )
    second = _Response(200, [b"ok"], headers=(("Content-Length", "2"),))
    executor = _Executor(first, second)
    transport = _transport(executor, clock=clock)
    page = _page()
    attempt0 = transport.fetch_page(page, attempt_ordinal=0)
    assert attempt0.http_status == 503
    attempt1 = transport.fetch_page(
        page, attempt_ordinal=1, prior_attempt=attempt0
    )
    assert attempt1.http_status == 200
    assert clock.sleeps == [2_000_000]
    assert attempt1.request_started_at_us == attempt0.terminal_at_us + 2_000_000


def test_retry_after_http_date_is_exact_and_cap_is_fail_closed() -> None:
    policy = replace(
        candidate_history_retry_policy_v1(), max_retry_after_us=5_000_000
    )
    received = 1_900_000_000_000_000
    target = received // 1_000_000 + 4
    rendered = datetime.fromtimestamp(target, tz=timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    headers = canonicalize_public_response_headers((("Retry-After", rendered),))
    assert retry_after_delay_us(
        headers, received_at_us=received, policy=policy
    ) == 4_000_000
    over = canonicalize_public_response_headers((("Retry-After", "6"),))
    with pytest.raises(RetryAfterContractError, match="exceeds_policy_cap"):
        retry_after_delay_us(over, received_at_us=received, policy=policy)
    invalid = canonicalize_public_response_headers((("Retry-After", "1.5"),))
    with pytest.raises(RetryAfterContractError, match="is_invalid"):
        retry_after_delay_us(invalid, received_at_us=received, policy=policy)


def test_minimum_spacing_applies_between_pages_without_retry() -> None:
    clock = _Clock()
    executor = _Executor(_Response(200, [b"a"]), _Response(200, [b"b"]))
    transport = _transport(executor, clock=clock)
    first = transport.fetch_page(_page(ordinal=0), attempt_ordinal=0)
    second = transport.fetch_page(_page(ordinal=1), attempt_ordinal=0)
    assert clock.sleeps == [500_000]
    assert second.request_started_at_us == first.request_started_at_us + 500_000


def test_attempt_and_runtime_resource_failures_are_closed() -> None:
    limits = replace(
        candidate_history_resource_limits_v1(),
        max_attempts_per_page=1,
        max_total_attempts=1,
    )
    transport = _transport(_Executor(_Response(200, [b"ok"])), limits=limits)
    transport.fetch_page(_page(), attempt_ordinal=0)
    with pytest.raises(ResourceBudgetExceededError, match="total_attempt_budget"):
        transport.fetch_page(_page(ordinal=1), attempt_ordinal=0)

    clock = _Clock()
    short = replace(
        candidate_history_resource_limits_v1(), max_attempt_runtime_us=3
    )
    response = _Response(
        200,
        [b"a"],
        clock=clock,
        advance_per_chunk_us=4,
    )
    evidence = _transport(_Executor(response), clock=clock, limits=short).fetch_page(
        _page(), attempt_ordinal=0
    )
    assert evidence.outcome == "timeout"
    assert evidence.safe_error_code == "attempt_runtime_exceeded"


def test_executor_runtime_is_checked_even_when_body_is_empty() -> None:
    clock = _Clock()

    class AdvancingExecutor(_Executor):
        def open(self, request, *, connect_timeout_us, read_timeout_us):
            clock.advance(4)
            return super().open(
                request,
                connect_timeout_us=connect_timeout_us,
                read_timeout_us=read_timeout_us,
            )

    limits = replace(
        candidate_history_resource_limits_v1(), max_attempt_runtime_us=3
    )
    response = _Response(200, [])
    evidence = _transport(
        AdvancingExecutor(response), clock=clock, limits=limits
    ).fetch_page(_page(), attempt_ordinal=0)
    assert evidence.outcome == "timeout"
    assert evidence.safe_error_code == "attempt_runtime_exceeded_before_body"
    assert response.closed is True


def test_collection_runtime_is_checked_during_later_attempt() -> None:
    clock = _Clock()
    limits = replace(
        candidate_history_resource_limits_v1(),
        max_collection_runtime_us=5,
        max_attempt_runtime_us=5,
    )
    retry = replace(
        candidate_history_retry_policy_v1(), min_request_spacing_us=0
    )
    first = _Response(200, [b"a"], clock=clock, advance_per_chunk_us=4)
    second = _Response(200, [b"b"], clock=clock, advance_per_chunk_us=2)
    transport = _transport(
        _Executor(first, second), clock=clock, limits=limits, retry=retry
    )
    transport.fetch_page(_page(ordinal=0), attempt_ordinal=0)
    evidence = transport.fetch_page(_page(ordinal=1), attempt_ordinal=0)
    assert evidence.outcome == "timeout"
    assert evidence.safe_error_code == "collection_runtime_exceeded"


def test_sleep_must_advance_both_epoch_and_monotonic_clocks() -> None:
    class BrokenSleepClock(_Clock):
        def sleep_us(self, duration_us: int) -> None:
            self.sleeps.append(duration_us)
            self.epoch += duration_us

    clock = BrokenSleepClock()
    transport = _transport(
        _Executor(_Response(200, [b"a"]), _Response(200, [b"b"])),
        clock=clock,
    )
    transport.fetch_page(_page(ordinal=0), attempt_ordinal=0)
    with pytest.raises(TransportContractError, match="monotonic_sleep_undershot"):
        transport.fetch_page(_page(ordinal=1), attempt_ordinal=0)


def test_retry_rejects_foreign_policy_or_resource_identity() -> None:
    page = _page()
    first_transport = _transport(_Executor(_Response(503, [b"bad"])))
    first = first_transport.fetch_page(page, attempt_ordinal=0)
    different_limits = replace(
        candidate_history_resource_limits_v1(),
        max_logical_storage_bytes=512 * 1024 * 1024 - 1,
    )
    second_transport = _transport(
        _Executor(_Response(200, [b"ok"])), limits=different_limits
    )
    with pytest.raises(TransportContractError, match="prior_attempt_contract_mismatch"):
        second_transport.fetch_page(page, attempt_ordinal=1, prior_attempt=first)


def test_crlf_header_is_not_laundered_and_invalid_status_closes_response() -> None:
    headers = canonicalize_public_response_headers(
        (("Date", "Sat, 15 Aug 2026 12:34:56 GMT\r\n"),)
    )
    assert headers[0][1].startswith("invalid.")

    response = _Response(99, [b"no"])
    evidence = _transport(_Executor(response)).fetch_page(_page(), attempt_ordinal=0)
    assert evidence.outcome == "network_error"
    assert evidence.safe_error_code == "response_header_contract_error"
    assert response.closed is True


def test_close_latency_and_failure_are_inside_typed_attempt_evidence() -> None:
    clock = _Clock()
    limits = replace(
        candidate_history_resource_limits_v1(),
        max_raw_body_bytes_per_attempt=3,
        max_attempt_runtime_us=3,
    )
    slow_close = _Response(
        200,
        [b"four"],
        clock=clock,
        close_advance_us=10,
    )
    evidence = _transport(
        _Executor(slow_close), clock=clock, limits=limits
    ).fetch_page(_page(), attempt_ordinal=0)
    assert evidence.outcome == "body_limit_exceeded"
    assert evidence.safe_error_code == (
        "attempt_body_limit_exceeded_and_attempt_runtime_exceeded"
    )
    assert evidence.elapsed_monotonic_us == 10
    assert evidence.body_bytes == b"fou"

    raising_close = _Response(
        200,
        [b"ok"],
        close_error=RuntimeError("must-not-escape"),
    )
    failed = _transport(_Executor(raising_close)).fetch_page(
        _page(), attempt_ordinal=0
    )
    assert failed.outcome == "network_error"
    assert failed.safe_error_code == "executor_close_error"
    assert failed.body_bytes == b"ok"

    cap_then_raising_close = _Response(
        200,
        [b"four"],
        close_error=RuntimeError("must-not-escape"),
    )
    terminal = _transport(
        _Executor(cap_then_raising_close), limits=limits
    ).fetch_page(_page(), attempt_ordinal=0)
    assert terminal.outcome == "body_limit_exceeded"
    assert terminal.safe_error_code == (
        "attempt_body_limit_exceeded_and_close_failure"
    )


def test_observed_oversleep_is_accounted_and_fails_closed() -> None:
    class OversleepClock(_Clock):
        def sleep_us(self, duration_us: int) -> None:
            actual = duration_us + 200_000
            self.sleeps.append(duration_us)
            self.epoch += actual
            self.monotonic += actual

    clock = OversleepClock()
    retry = replace(
        candidate_history_retry_policy_v1(), max_total_sleep_us=600_000
    )
    transport = _transport(
        _Executor(_Response(200, [b"a"]), _Response(200, [b"b"])),
        clock=clock,
        retry=retry,
    )
    transport.fetch_page(_page(ordinal=0), attempt_ordinal=0)
    with pytest.raises(ResourceBudgetExceededError, match="observed_retry_sleep"):
        transport.fetch_page(_page(ordinal=1), attempt_ordinal=0)
    assert len(transport.executor.requests) == 1


def test_epoch_forward_jump_cannot_bypass_monotonic_spacing() -> None:
    clock = _Clock()
    transport = _transport(
        _Executor(_Response(200, [b"a"]), _Response(200, [b"b"])),
        clock=clock,
    )
    first = transport.fetch_page(_page(ordinal=0), attempt_ordinal=0)
    clock.epoch += 10_000_000
    second = transport.fetch_page(_page(ordinal=1), attempt_ordinal=0)
    assert clock.sleeps == [500_000]
    assert (
        second.request_started_monotonic_us
        - first.request_started_monotonic_us
        == 500_000
    )
    assert second.request_started_monotonic_us >= second.scheduled_not_before_monotonic_us


def test_header_work_is_bounded_and_deadline_is_sampled_after_canonicalization() -> None:
    assert canonicalize_public_response_headers(
        (("ETag", '"' + ("x" * 5_000) + '"'),)
    ) == (("etag", "invalid.oversized"),)

    too_many = tuple((f"x-unknown-{index}", "v") for index in range(65))
    response = _Response(200, [b"ok"], headers=too_many)
    evidence = _transport(_Executor(response)).fetch_page(_page(), attempt_ordinal=0)
    assert evidence.outcome == "network_error"
    assert evidence.safe_error_code == "response_header_contract_error"
    assert response.closed is True

    clock = _Clock()

    class SlowHeaders(_Response):
        @property
        def headers(self):
            clock.advance(4)
            return self._headers

    limits = replace(
        candidate_history_resource_limits_v1(), max_attempt_runtime_us=3
    )
    slow = SlowHeaders(200, [b"must-not-read"])
    timed = _transport(_Executor(slow), clock=clock, limits=limits).fetch_page(
        _page(), attempt_ordinal=0
    )
    assert timed.outcome == "timeout"
    assert timed.safe_error_code == "attempt_runtime_exceeded_before_body"
    assert slow.yielded == 0 and slow.closed is True


def test_attempt_parser_rejects_body_or_page_tampering() -> None:
    evidence = _transport(_Executor(_Response(200, [b"ok"]))).fetch_page(
        _page(), attempt_ordinal=0
    )
    with pytest.raises(TransportContractError, match="body_mismatch"):
        parse_http_attempt_evidence_v1(
            evidence.receipt_dict(), page_request=_page(), body_bytes=b"no"
        )
    payload = evidence.receipt_dict()
    payload["page_id"] = "b" * 64
    with pytest.raises(TransportContractError, match="page_mismatch"):
        parse_http_attempt_evidence_v1(
            payload, page_request=_page(), body_bytes=b"ok"
        )
