from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import pytest

import trading.market_data.mexc_endpoint_official_evidence as official_module

from trading.market_data.mexc_endpoint_official_evidence import (
    CANDIDATE_CONTRACT_HASH,
    CANDIDATE_CONTRACT_VERSION,
    MexcOfficialEvidenceBudgetStop,
    MexcOfficialEvidenceContractError,
    MexcOfficialEvidenceSemanticStop,
    MexcOfficialEvidenceStorageStop,
    MexcOfficialEvidenceTerminalStop,
    OFFICIAL_DOCUMENT_EVIDENCE_VERSION,
    OFFICIAL_DOCUMENT_READER_VERSION,
    OFFICIAL_EVIDENCE_BUNDLE_VERSION,
    OFFICIAL_EVIDENCE_COMPATIBILITY_VERSION,
    OFFICIAL_EVIDENCE_STORE_VERSION,
    OFFICIAL_REFERENCE_HTTP_ATTEMPT_VERSION,
    OFFICIAL_REFERENCE_ID,
    OFFICIAL_REFERENCE_REQUEST_VERSION,
    OFFICIAL_REFERENCE_URL,
    OFFICIAL_RAW_BODY_HARD_CAP_BYTES,
    OFFICIAL_STORAGE_CONCURRENCY_BOUNDARY,
    OfficialDocumentEvidenceV1,
    OfficialDocumentSpanClaimV1,
    OfficialEvidenceBundleFileV1,
    OfficialEvidenceBundleV1,
    OfficialEvidenceCompatibilityV1,
    OfficialReferenceHttpAttemptV1,
    OfficialReferencePreparedRequestV1,
    REVIEWED_FAKE_FIXTURE_ONLY,
    assess_official_evidence_compatibility_v1,
    build_exact_span_claims_v1,
    derive_official_bundle_root_v1,
    mexc_endpoint_official_evidence_contract_hash,
    official_bundle_relative_paths_v1,
    official_evidence_contract_descriptor_v1,
    parse_canonical_json_lf_v1,
    publish_official_evidence_bundle_v1,
    read_official_document_evidence_v1,
    reload_official_evidence_bundle_v1,
    require_terminal_compatible_official_evidence_v1,
)
from trading.market_data.mexc_futures_transport import (
    HistoryResourceLimitsV1,
    candidate_endpoint_fixture_path,
    candidate_history_retry_policy_v1,
    load_mexc_futures_endpoint_contract_v1,
)
from trading.market_data.mexc_pilot_run import EndpointVerificationPlanV1
from trading.market_data.strict_history_v2 import HistoryRangeRequestV2


BASE_EPOCH = 1_900_000_000_000_000
BASE_MONO = 10_000_000
MIGRATION_STATEMENT = (
    b"Futures API domain migration: contract.mexc.com -> api.mexc.com"
)
FULL_CANDIDATE_STATEMENT = (
    b"Candidate contract: method=GET; scheme=https; host=api.mexc.com; port=443; "
    b"path=/api/v1/contract/kline/{venue_symbol}; query_encoding=ascii_exact_ordered; "
    b"query_order=interval:{interval},start:{start_open_ts},"
    b"end:{end_open_ts_inclusive}; request_headers=accept:application/json,"
    b"accept-encoding:identity,user-agent:koteika-strict-history/1.0; "
    b"authentication=none; tls_verification=required; redirects=reject"
)


def _body(*, full: bool = False) -> bytes:
    result = b"Reviewed fake fixture only.\n" + MIGRATION_STATEMENT + b"\n"
    if full:
        result += FULL_CANDIDATE_STATEMENT + b"\n"
    return result


def _prepared(*, plan_hash: str | None = None) -> OfficialReferencePreparedRequestV1:
    if plan_hash is None:
        plan_hash = _verification_plan().plan_hash
    return OfficialReferencePreparedRequestV1(
        verification_plan_hash=plan_hash,
        endpoint_runner_contract_version="fixture_endpoint_runner_v1",
        endpoint_runner_contract_hash="2" * 64,
        parser_contract_version="fixture_official_parser_v1",
        parser_contract_hash="3" * 64,
        transport_contract_version="fixture_official_transport_v1",
        transport_contract_hash="4" * 64,
        runtime_contract_version="fixture_official_runtime_v1",
        runtime_contract_hash="5" * 64,
    )


def _attempt(
    raw_body: bytes,
    prepared: OfficialReferencePreparedRequestV1,
    *,
    outcome: str = "complete",
) -> OfficialReferenceHttpAttemptV1:
    complete = outcome == "complete"
    return OfficialReferenceHttpAttemptV1(
        manifest_hash="6" * 64,
        authorization_receipt_hash="7" * 64,
        preflight_receipt_hash="8" * 64,
        verification_plan_hash=prepared.verification_plan_hash,
        network_intent_hash="9" * 64,
        endpoint_runner_contract_version=prepared.endpoint_runner_contract_version,
        endpoint_runner_contract_hash=prepared.endpoint_runner_contract_hash,
        runtime_authority_binding_hash="a" * 64,
        clock_domain_id="fixture_official_clock",
        tls_policy_version="fixture_tls_policy_v1",
        tls_policy_hash="b" * 64,
        trust_store_version="fixture_trust_store_v1",
        trust_store_hash="c" * 64,
        prepared_request_hash=prepared.prepared_request_hash,
        gate_checked_at_us=BASE_EPOCH,
        gate_checked_monotonic_us=BASE_MONO,
        request_started_at_us=BASE_EPOCH + 100,
        request_started_monotonic_us=BASE_MONO + 100,
        tls_validated_at_us=BASE_EPOCH + 200 if complete else None,
        tls_validated_monotonic_us=BASE_MONO + 200 if complete else None,
        headers_received_at_us=BASE_EPOCH + 300 if complete else None,
        headers_received_monotonic_us=BASE_MONO + 300 if complete else None,
        body_eof_at_us=BASE_EPOCH + 400 if complete else None,
        body_eof_monotonic_us=BASE_MONO + 400 if complete else None,
        connection_closed_at_us=BASE_EPOCH + 450,
        connection_closed_monotonic_us=BASE_MONO + 450,
        tls_version="TLSv1.3" if complete else None,
        peer_leaf_certificate_sha256="d" * 64 if complete else None,
        validated_chain_sha256="e" * 64 if complete else None,
        pkix_validated=complete,
        status_code=200 if complete else None,
        response_headers=(
            (
                ("content-length", str(len(raw_body))),
                ("content-type", "text/html; charset=utf-8"),
            )
            if complete
            else ()
        ),
        body_complete=complete,
        terminal_progress=(
            "body_eof" if complete else "before_tls_validation"
        ),
        outcome=outcome,
        safe_error_code=None if complete else "transport_connection_closed",
        raw_body_byte_count=len(raw_body),
        raw_body_sha256=hashlib.sha256(raw_body).hexdigest(),
    )


def _claims(raw_body: bytes, *, full: bool = False):
    claims = build_exact_span_claims_v1(raw_body)
    assert (len(claims) == 2) is full
    return claims


def _failure_attempt(
    prepared: OfficialReferencePreparedRequestV1,
    outcome: str,
) -> OfficialReferenceHttpAttemptV1:
    complete = _attempt(_body(), prepared)
    empty_hash = hashlib.sha256(b"").hexdigest()
    if outcome in {"incomplete_transport_error", "incomplete_tls_error"}:
        tls_failure = outcome == "incomplete_tls_error"
        return replace(
            complete,
            tls_validated_at_us=None,
            tls_validated_monotonic_us=None,
            headers_received_at_us=None,
            headers_received_monotonic_us=None,
            body_eof_at_us=None,
            body_eof_monotonic_us=None,
            tls_version="TLSv1.3" if tls_failure else None,
            peer_leaf_certificate_sha256="d" * 64 if tls_failure else None,
            validated_chain_sha256=None,
            pkix_validated=False,
            status_code=None,
            response_headers=(),
            body_complete=False,
            terminal_progress=(
                "tls_validation_failed"
                if tls_failure
                else "before_tls_validation"
            ),
            outcome=outcome,
            safe_error_code=(
                "transport_timeout"
                if outcome == "incomplete_transport_error"
                else "tls_certificate_validation_failed"
            ),
            raw_body_byte_count=0,
            raw_body_sha256=empty_hash,
        )
    if outcome == "incomplete_http_body_error":
        partial = b"partial"
        return replace(
            complete,
            body_eof_at_us=None,
            body_eof_monotonic_us=None,
            body_complete=False,
            terminal_progress="headers_received_before_body_eof",
            outcome=outcome,
            safe_error_code="http_body_read_failed",
            raw_body_byte_count=len(partial),
            raw_body_sha256=hashlib.sha256(partial).hexdigest(),
        )
    if outcome == "rejected_protocol":
        return replace(
            complete,
            body_eof_at_us=None,
            body_eof_monotonic_us=None,
            status_code=503,
            body_complete=False,
            terminal_progress="headers_received_before_body_eof",
            outcome=outcome,
            safe_error_code="http_status_not_200",
            raw_body_byte_count=0,
            raw_body_sha256=empty_hash,
        )
    raise AssertionError(outcome)


def _evidence(
    raw_body: bytes,
    prepared: OfficialReferencePreparedRequestV1,
    attempt: OfficialReferenceHttpAttemptV1,
    *,
    full: bool = False,
    **overrides: object,
) -> OfficialDocumentEvidenceV1:
    values: dict[str, object] = {
        "raw_body": raw_body,
        "attempt": attempt,
        "prepared_request": prepared,
        "claims": _claims(raw_body, full=full),
        "parser_contract_version": prepared.parser_contract_version,
        "parser_contract_hash": prepared.parser_contract_hash,
        "reader_contract_hash": mexc_endpoint_official_evidence_contract_hash(),
        "parse_started_at_us": BASE_EPOCH + 500,
        "parse_completed_at_us": BASE_EPOCH + 600,
        "parse_started_monotonic_us": BASE_MONO + 500,
        "parse_completed_monotonic_us": BASE_MONO + 600,
        "reload_completed_at_us": BASE_EPOCH + 700,
        "reload_completed_monotonic_us": BASE_MONO + 700,
    }
    values.update(overrides)
    return read_official_document_evidence_v1(  # type: ignore[arg-type]
        **values,
    )


def _compatibility(
    *,
    verification_plan: EndpointVerificationPlanV1 | None = None,
) -> OfficialEvidenceCompatibilityV1:
    return assess_official_evidence_compatibility_v1(
        verification_plan=verification_plan or _verification_plan(),
    )


def _bundle(tmp_path: Path, *, full: bool = False):
    raw_body = _body(full=full)
    plan = _verification_plan()
    prepared = _prepared(plan_hash=plan.plan_hash)
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt, full=full)
    compatibility = _compatibility(verification_plan=plan)
    output_root = tmp_path / "pilot-output"
    bundle = publish_official_evidence_bundle_v1(
        output_root=output_root,
        raw_body=raw_body,
        attempt=attempt,
        evidence=evidence,
        prepared_request=prepared,
        compatibility=compatibility,
    )
    return output_root, raw_body, prepared, attempt, evidence, compatibility, bundle


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def test_frozen_reference_request_and_candidate_binding_are_exact() -> None:
    prepared = _prepared()
    assert prepared.contract_version == OFFICIAL_REFERENCE_REQUEST_VERSION
    assert prepared.url == OFFICIAL_REFERENCE_URL
    assert prepared.reference_id == OFFICIAL_REFERENCE_ID
    assert (prepared.scheme, prepared.host, prepared.port, prepared.path) == (
        "https",
        "www.mexc.com",
        443,
        "/announcements/article/futures-api-access-domain-update-17827791532974",
    )
    assert (prepared.query, prepared.fragment, prepared.userinfo) == ("", "", "")
    assert prepared.headers == (
        ("accept", "text/html,application/xhtml+xml"),
        ("accept-encoding", "identity"),
        ("user-agent", "koteika-mexc-official-evidence/1.0"),
    )
    assert prepared.body_byte_count == 0
    assert prepared.body_sha256 == hashlib.sha256(b"").hexdigest()
    assert prepared.tls_verify is True
    assert prepared.allow_redirects is False
    assert prepared.trust_env is False
    assert prepared.proxy_enabled is False
    assert prepared.cookies_enabled is False
    assert prepared.netrc_enabled is False
    assert prepared.authentication_enabled is False
    assert prepared.candidate_contract_version == CANDIDATE_CONTRACT_VERSION
    assert prepared.candidate_contract_hash == CANDIDATE_CONTRACT_HASH
    assert OfficialReferencePreparedRequestV1.from_dict(prepared.as_dict()) == prepared


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("url", OFFICIAL_REFERENCE_URL + "?alias=1"),
        ("url", OFFICIAL_REFERENCE_URL + "#fragment"),
        ("url", OFFICIAL_REFERENCE_URL.replace("www.mexc.com", "www.mexc.com:443")),
        ("host", "WWW.MEXC.COM"),
        ("path", "//announcements/article/futures-api-access-domain-update-17827791532974"),
        ("allow_redirects", True),
        ("trust_env", True),
        ("proxy_enabled", True),
        ("cookies_enabled", True),
        ("netrc_enabled", True),
        ("authentication_enabled", True),
        ("tls_verify", False),
    ),
)
def test_prepared_request_rejects_url_aliases_and_ambient_authority(
    field: str, value: object
) -> None:
    with pytest.raises(MexcOfficialEvidenceContractError):
        replace(_prepared(), **{field: value})


def test_attempt_receipt_binds_complete_tls_status_headers_and_raw_body() -> None:
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    assert attempt.contract_version == OFFICIAL_REFERENCE_HTTP_ATTEMPT_VERSION
    assert attempt.outcome == "complete"
    assert attempt.pkix_validated is True
    assert attempt.status_code == 200
    assert attempt.redirects_followed == 0
    assert attempt.raw_body_relative_path.endswith("/official/attempt-000.body.bin")
    assert OfficialReferenceHttpAttemptV1.from_dict(attempt.as_dict()) == attempt


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("port", True),
        ("body_byte_count", False),
    ),
)
def test_prepared_request_rejects_bool_int_aliases(field: str, value: object) -> None:
    with pytest.raises(MexcOfficialEvidenceContractError):
        replace(_prepared(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("attempt_ordinal", False),
        ("redirects_followed", False),
        ("gate_checked_at_us", True),
        ("raw_body_byte_count", True),
        ("status_code", True),
        ("pkix_validated", 1),
        ("body_complete", 1),
        ("credentials_used", 0),
        ("terminal_compatible", 0),
        ("terminal_progress", ["body_eof"]),
    ),
)
def test_attempt_rejects_bool_int_and_int_bool_aliases(
    field: str, value: object
) -> None:
    raw_body = _body()
    with pytest.raises(MexcOfficialEvidenceContractError):
        replace(_attempt(raw_body, _prepared()), **{field: value})


def test_terminal_progress_rejects_string_subclass_alias() -> None:
    class StringAlias(str):
        pass

    with pytest.raises(MexcOfficialEvidenceContractError):
        replace(
            _attempt(_body(), _prepared()),
            terminal_progress=StringAlias("body_eof"),
        )


@pytest.mark.parametrize(
    "outcome",
    (
        "incomplete_transport_error",
        "incomplete_tls_error",
        "incomplete_http_body_error",
        "rejected_protocol",
    ),
)
def test_each_incomplete_outcome_has_one_exact_roundtrippable_state(
    outcome: str,
) -> None:
    attempt = _failure_attempt(_prepared(), outcome)
    assert attempt.outcome == outcome
    assert attempt.body_complete is False
    assert OfficialReferenceHttpAttemptV1.from_dict(attempt.as_dict()) == attempt


@pytest.mark.parametrize(
    ("outcome", "mutations"),
    (
        ("incomplete_transport_error", {"status_code": 200}),
        ("incomplete_tls_error", {"validated_chain_sha256": "e" * 64}),
        (
            "incomplete_http_body_error",
            {"body_eof_at_us": BASE_EPOCH + 400, "body_eof_monotonic_us": BASE_MONO + 400},
        ),
        ("rejected_protocol", {"status_code": 200}),
        (
            "incomplete_transport_error",
            {"safe_error_code": "tls_handshake_failed"},
        ),
    ),
)
def test_outcome_specific_state_matrix_rejects_cross_state_facts(
    outcome: str, mutations: dict[str, object]
) -> None:
    with pytest.raises(MexcOfficialEvidenceContractError):
        replace(_failure_attempt(_prepared(), outcome), **mutations)


@pytest.mark.parametrize(
    "safe_error_code",
    ("transport_timeout", "transport_connection_closed"),
)
def test_transport_failure_preserves_validated_tls_prefix_before_headers(
    safe_error_code: str,
) -> None:
    prepared = _prepared()
    complete = _attempt(_body(), prepared)
    empty_hash = hashlib.sha256(b"").hexdigest()
    attempt = replace(
        complete,
        headers_received_at_us=None,
        headers_received_monotonic_us=None,
        body_eof_at_us=None,
        body_eof_monotonic_us=None,
        status_code=None,
        response_headers=(),
        body_complete=False,
        terminal_progress="tls_validated_before_headers",
        outcome="incomplete_transport_error",
        safe_error_code=safe_error_code,
        raw_body_byte_count=0,
        raw_body_sha256=empty_hash,
    )
    assert attempt.tls_validated_at_us is not None
    assert attempt.peer_leaf_certificate_sha256 is not None
    assert OfficialReferenceHttpAttemptV1.from_dict(attempt.as_dict()) == attempt
    with pytest.raises(MexcOfficialEvidenceContractError, match="state_matrix"):
        replace(attempt, safe_error_code="dns_resolution_failed")


@pytest.mark.parametrize(
    "safe_error_code",
    ("dns_resolution_failed", "transport_connect_failed"),
)
def test_pre_tls_transport_codes_have_exact_empty_prefix(
    safe_error_code: str,
) -> None:
    attempt = replace(
        _failure_attempt(_prepared(), "incomplete_transport_error"),
        safe_error_code=safe_error_code,
    )
    assert attempt.terminal_progress == "before_tls_validation"
    assert attempt.tls_version is None


def test_tls_failure_preserves_observed_version_and_peer_without_claiming_pkix() -> None:
    certificate_failure = _failure_attempt(_prepared(), "incomplete_tls_error")
    assert certificate_failure.terminal_progress == "tls_validation_failed"
    assert certificate_failure.tls_version == "TLSv1.3"
    assert certificate_failure.peer_leaf_certificate_sha256 == "d" * 64
    assert certificate_failure.validated_chain_sha256 is None
    assert certificate_failure.pkix_validated is False
    for code in ("tls_certificate_validation_failed", "tls_sni_mismatch"):
        observed_peer_failure = replace(certificate_failure, safe_error_code=code)
        with pytest.raises(MexcOfficialEvidenceContractError, match="state_matrix"):
            replace(observed_peer_failure, peer_leaf_certificate_sha256=None)

    handshake_failure = replace(
        certificate_failure,
        safe_error_code="tls_handshake_failed",
        tls_version=None,
        peer_leaf_certificate_sha256=None,
    )
    assert handshake_failure.tls_version is None
    policy_failure = replace(
        certificate_failure,
        safe_error_code="tls_policy_rejected",
        peer_leaf_certificate_sha256=None,
    )
    assert policy_failure.tls_version == "TLSv1.3"
    with pytest.raises(MexcOfficialEvidenceContractError, match="state_matrix"):
        replace(policy_failure, tls_version=None)


@pytest.mark.parametrize(
    ("safe_error_code", "headers"),
    (
        (
            "content_type_not_official_html",
            (("content-type", "application/json"),),
        ),
        (
            "content_length_invalid",
            (
                ("content-length", "01"),
                ("content-type", "text/html; charset=utf-8"),
            ),
        ),
    ),
)
def test_protocol_rejection_codes_require_exact_header_predicates(
    safe_error_code: str,
    headers: tuple[tuple[str, str], ...],
) -> None:
    attempt = replace(
        _failure_attempt(_prepared(), "rejected_protocol"),
        status_code=200,
        response_headers=headers,
        safe_error_code=safe_error_code,
    )
    assert attempt.terminal_progress == "headers_received_before_body_eof"
    with pytest.raises(MexcOfficialEvidenceContractError, match="state_matrix"):
        replace(
            attempt,
            response_headers=(("content-type", "text/html; charset=utf-8"),),
        )


def test_protocol_status_redirect_and_encoding_codes_are_nonoverlapping() -> None:
    base = _failure_attempt(_prepared(), "rejected_protocol")
    redirect = replace(
        base,
        status_code=302,
        safe_error_code="redirect_status_rejected",
    )
    assert redirect.status_code == 302
    with pytest.raises(MexcOfficialEvidenceContractError, match="state_matrix"):
        replace(redirect, safe_error_code="http_status_not_200")

    encoding = replace(
        base,
        status_code=200,
        response_headers=(
            ("content-encoding", "gzip"),
            ("content-type", "text/html; charset=utf-8"),
        ),
        safe_error_code="content_encoding_not_identity",
    )
    assert dict(encoding.response_headers)["content-encoding"] == "gzip"
    with pytest.raises(MexcOfficialEvidenceContractError, match="state_matrix"):
        replace(encoding, response_headers=(("content-type", "text/html"),))


def test_content_length_mismatch_body_error_requires_canonical_mismatched_length() -> None:
    attempt = replace(
        _failure_attempt(_prepared(), "incomplete_http_body_error"),
        safe_error_code="content_length_mismatch",
    )
    assert dict(attempt.response_headers)["content-length"] != str(
        attempt.raw_body_byte_count
    )
    with pytest.raises(MexcOfficialEvidenceContractError, match="state_matrix"):
        replace(
            attempt,
            response_headers=(
                ("content-length", str(attempt.raw_body_byte_count)),
                ("content-type", "text/html; charset=utf-8"),
            ),
        )


def test_body_cap_error_requires_exact_cap_or_declared_over_cap() -> None:
    base = _failure_attempt(_prepared(), "incomplete_http_body_error")
    at_cap = replace(
        base,
        safe_error_code="body_cap_exceeded",
        raw_body_byte_count=OFFICIAL_RAW_BODY_HARD_CAP_BYTES,
        raw_body_sha256="f" * 64,
    )
    assert at_cap.raw_body_byte_count == OFFICIAL_RAW_BODY_HARD_CAP_BYTES
    with pytest.raises(MexcOfficialEvidenceContractError, match="state_matrix"):
        replace(at_cap, raw_body_byte_count=OFFICIAL_RAW_BODY_HARD_CAP_BYTES - 1)

    declared_over_cap = replace(
        base,
        safe_error_code="body_cap_exceeded",
        response_headers=(
            ("content-length", str(OFFICIAL_RAW_BODY_HARD_CAP_BYTES + 1)),
            ("content-type", "text/html; charset=utf-8"),
        ),
    )
    assert int(dict(declared_over_cap.response_headers)["content-length"]) > (
        OFFICIAL_RAW_BODY_HARD_CAP_BYTES
    )


@pytest.mark.parametrize(
    "terminal_progress",
    (
        "before_tls_validation",
        "tls_validation_failed",
        "tls_validated_before_headers",
        "headers_received_before_body_eof",
    ),
)
def test_complete_attempt_rejects_every_non_eof_terminal_progress(
    terminal_progress: str,
) -> None:
    with pytest.raises(MexcOfficialEvidenceContractError):
        replace(_attempt(_body(), _prepared()), terminal_progress=terminal_progress)


def test_safe_error_code_is_allowlisted_not_free_text() -> None:
    with pytest.raises(MexcOfficialEvidenceContractError, match="allowlisted"):
        replace(
            _failure_attempt(_prepared(), "incomplete_transport_error"),
            safe_error_code="arbitrary_user_visible_secret_detail",
        )


def test_incomplete_attempt_is_typed_and_cannot_be_semantic_evidence() -> None:
    raw_body = b""
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared, outcome="incomplete_transport_error")
    assert attempt.body_complete is False
    assert attempt.safe_error_code == "transport_connection_closed"
    with pytest.raises(
        MexcOfficialEvidenceSemanticStop,
        match="official_document_attempt_is_not_complete",
    ):
        read_official_document_evidence_v1(
            raw_body=raw_body,
            attempt=attempt,
            prepared_request=prepared,
            claims=(),
            parser_contract_version=prepared.parser_contract_version,
            parser_contract_hash=prepared.parser_contract_hash,
            reader_contract_hash=mexc_endpoint_official_evidence_contract_hash(),
            parse_started_at_us=BASE_EPOCH + 500,
            parse_completed_at_us=BASE_EPOCH + 600,
            parse_started_monotonic_us=BASE_MONO + 500,
            parse_completed_monotonic_us=BASE_MONO + 600,
            reload_completed_at_us=BASE_EPOCH + 700,
            reload_completed_monotonic_us=BASE_MONO + 700,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("redirects_followed", 1),
        ("final_url", OFFICIAL_REFERENCE_URL + "/redirected"),
        ("tls_sni", "api.mexc.com"),
        ("tls_version", "TLSv1.1"),
        ("pkix_validated", False),
        ("status_code", 204),
        ("credentials_used", True),
        ("proxy_used", True),
        ("cookies_used", True),
        ("netrc_used", True),
        ("trust_env", True),
    ),
)
def test_complete_attempt_rejects_redirect_tls_status_and_ambient_authority(
    field: str, value: object
) -> None:
    raw_body = _body()
    prepared = _prepared()
    with pytest.raises(MexcOfficialEvidenceContractError):
        replace(_attempt(raw_body, prepared), **{field: value})


@pytest.mark.parametrize(
    "headers",
    (
        (("set-cookie", "session=secret"),),
        (("authorization", "Bearer secret"),),
        (("content-encoding", "gzip"), ("content-type", "text/html")),
        (("content-length", "0001"), ("content-type", "text/html")),
        (("content-type", "text/html\r\nset-cookie: secret"),),
    ),
)
def test_attempt_rejects_secret_or_unsafe_header_evidence(headers) -> None:
    raw_body = _body()
    prepared = _prepared()
    with pytest.raises(MexcOfficialEvidenceContractError):
        replace(_attempt(raw_body, prepared), response_headers=headers)


def test_reader_derives_migration_only_from_exact_statement_span() -> None:
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt)
    assert evidence.contract_version == OFFICIAL_DOCUMENT_EVIDENCE_VERSION
    assert evidence.reader_contract_version == OFFICIAL_DOCUMENT_READER_VERSION
    assert evidence.support_scope == "domain_migration_only"
    assert evidence.verdict == "additional_current_official_contract_evidence_required"
    assert evidence.observed_body_version == {
        "canonical_url": OFFICIAL_REFERENCE_URL,
        "reference_id": OFFICIAL_REFERENCE_ID,
        "fetched_at_us": BASE_EPOCH + 400,
        "fetched_monotonic_us": BASE_MONO + 400,
        "body_sha256": hashlib.sha256(raw_body).hexdigest(),
    }
    assert hashlib.sha256(evidence.canonical_lf_bytes).hexdigest() == evidence.evidence_hash
    assert OfficialDocumentEvidenceV1.from_dict(evidence.as_dict()) == evidence


def test_reader_requires_exact_prepared_parser_and_self_contract_bindings() -> None:
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    with pytest.raises(MexcOfficialEvidenceContractError, match="parser_binding"):
        _evidence(
            raw_body,
            prepared,
            attempt,
            parser_contract_hash="0" * 64,
        )
    with pytest.raises(MexcOfficialEvidenceContractError, match="self_binding"):
        _evidence(
            raw_body,
            prepared,
            attempt,
            reader_contract_hash="0" * 64,
        )


def test_reader_binds_eof_in_both_clock_domains_and_parse_starts_after_eof(
    tmp_path: Path,
) -> None:
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt)
    assert evidence.observed_body_fetched_at_us == attempt.body_eof_at_us
    assert (
        evidence.observed_body_fetched_monotonic_us
        == attempt.body_eof_monotonic_us
    )
    with pytest.raises(MexcOfficialEvidenceContractError, match="timeline"):
        replace(
            evidence,
            parse_started_at_us=evidence.observed_body_fetched_at_us,
        )
    with pytest.raises(MexcOfficialEvidenceContractError, match="timeline"):
        replace(
            evidence,
            parse_started_monotonic_us=(
                evidence.observed_body_fetched_monotonic_us
            ),
        )
    forged_observation = replace(
        evidence,
        observed_body_fetched_at_us=evidence.observed_body_fetched_at_us + 1,
        observed_body_fetched_monotonic_us=(
            evidence.observed_body_fetched_monotonic_us + 1
        ),
    )
    output_root = tmp_path / "forged-clock"
    with pytest.raises(MexcOfficialEvidenceContractError, match="cross_binding"):
        publish_official_evidence_bundle_v1(
            output_root=output_root,
            raw_body=raw_body,
            attempt=attempt,
            evidence=forged_observation,
            prepared_request=prepared,
            compatibility=_compatibility(),
        )
    assert not output_root.exists()


def test_evidence_object_rejects_non_self_bound_reader_hash() -> None:
    raw_body = _body()
    prepared = _prepared()
    evidence = _evidence(raw_body, prepared, _attempt(raw_body, prepared))
    with pytest.raises(MexcOfficialEvidenceContractError, match="dependency"):
        replace(evidence, reader_contract_hash="0" * 64)


def test_reader_can_only_derive_full_scope_from_one_exact_structural_statement() -> None:
    raw_body = _body(full=True)
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt, full=True)
    assert evidence.support_scope == "full_candidate_contract"
    assert evidence.verdict == "candidate_contract_semantics_observed"
    assert evidence.authority_status == REVIEWED_FAKE_FIXTURE_ONLY
    assert evidence.terminal_compatible is False


def test_scattered_candidate_literals_do_not_prove_full_contract() -> None:
    raw_body = (
        b"Reviewed fake fixture only.\n"
        + MIGRATION_STATEMENT
        + b"\nGET is mentioned elsewhere. The path is "
        + b"/api/v1/contract/kline/{venue_symbol}. interval start end identity.\n"
    )
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    claims = build_exact_span_claims_v1(raw_body)
    assert tuple(claim.role for claim in claims) == (
        "domain_migration_statement_v1",
    )
    evidence = read_official_document_evidence_v1(
        raw_body=raw_body,
        attempt=attempt,
        prepared_request=prepared,
        claims=claims,
        parser_contract_version=prepared.parser_contract_version,
        parser_contract_hash=prepared.parser_contract_hash,
        reader_contract_hash=mexc_endpoint_official_evidence_contract_hash(),
        parse_started_at_us=BASE_EPOCH + 500,
        parse_completed_at_us=BASE_EPOCH + 600,
        parse_started_monotonic_us=BASE_MONO + 500,
        parse_completed_monotonic_us=BASE_MONO + 600,
        reload_completed_at_us=BASE_EPOCH + 700,
        reload_completed_monotonic_us=BASE_MONO + 700,
    )
    assert evidence.support_scope == "domain_migration_only"


def test_domain_cooccurrence_or_reverse_text_does_not_prove_migration() -> None:
    raw_body = (
        b"This unrelated text mentions contract.mexc.com before api.mexc.com "
        b"but contains no reviewed migration statement."
    )
    with pytest.raises(MexcOfficialEvidenceSemanticStop):
        build_exact_span_claims_v1(raw_body)


def test_negated_same_line_statement_does_not_prove_migration() -> None:
    raw_body = b"Not an assertion: " + MIGRATION_STATEMENT + b"\n"
    with pytest.raises(
        MexcOfficialEvidenceSemanticStop,
        match="lacks_exact_line_context",
    ):
        build_exact_span_claims_v1(raw_body)


def test_caller_cannot_omit_full_statement_claim_from_full_body() -> None:
    raw_body = _body(full=True)
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    claims = build_exact_span_claims_v1(raw_body)
    with pytest.raises(
        MexcOfficialEvidenceContractError,
        match="not_deterministically_derived",
    ):
        read_official_document_evidence_v1(
            raw_body=raw_body,
            attempt=attempt,
            prepared_request=prepared,
            claims=(claims[0],),
            parser_contract_version=prepared.parser_contract_version,
            parser_contract_hash=prepared.parser_contract_hash,
            reader_contract_hash=mexc_endpoint_official_evidence_contract_hash(),
            parse_started_at_us=BASE_EPOCH + 500,
            parse_completed_at_us=BASE_EPOCH + 600,
            parse_started_monotonic_us=BASE_MONO + 500,
            parse_completed_monotonic_us=BASE_MONO + 600,
            reload_completed_at_us=BASE_EPOCH + 700,
            reload_completed_monotonic_us=BASE_MONO + 700,
        )


def test_semantic_scans_stop_before_processing_hard_cap_plus_one() -> None:
    raw_body = b"x" * (OFFICIAL_RAW_BODY_HARD_CAP_BYTES + 1)
    with pytest.raises(MexcOfficialEvidenceBudgetStop, match="hard_cap"):
        build_exact_span_claims_v1(raw_body)


def test_forged_span_hash_bounds_and_claimed_scope_are_rejected(tmp_path: Path) -> None:
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    claim = _claims(raw_body)[0]
    forged_hash = replace(claim, span_sha256="0" * 64)
    with pytest.raises(MexcOfficialEvidenceContractError):
        read_official_document_evidence_v1(
            raw_body=raw_body,
            attempt=attempt,
            prepared_request=prepared,
            claims=(forged_hash,),
            parser_contract_version=prepared.parser_contract_version,
            parser_contract_hash=prepared.parser_contract_hash,
            reader_contract_hash=mexc_endpoint_official_evidence_contract_hash(),
            parse_started_at_us=BASE_EPOCH + 500,
            parse_completed_at_us=BASE_EPOCH + 600,
            parse_started_monotonic_us=BASE_MONO + 500,
            parse_completed_monotonic_us=BASE_MONO + 600,
            reload_completed_at_us=BASE_EPOCH + 700,
            reload_completed_monotonic_us=BASE_MONO + 700,
        )
    evidence = _evidence(raw_body, prepared, attempt)
    forged_scope = replace(
        evidence,
        support_scope="full_candidate_contract",
        verdict="candidate_contract_semantics_observed",
    )
    with pytest.raises(
        MexcOfficialEvidenceContractError,
        match="support_scope_was_not_rederived",
    ):
        publish_official_evidence_bundle_v1(
            output_root=tmp_path / "forged-scope",
            raw_body=raw_body,
            attempt=attempt,
            evidence=forged_scope,
            prepared_request=prepared,
            compatibility=_compatibility(),
        )


def test_span_claim_role_requires_exact_string_before_membership() -> None:
    claim = _claims(_body())[0]

    class StringAlias(str):
        pass

    for role in ([], StringAlias(claim.role)):
        with pytest.raises(
            MexcOfficialEvidenceContractError,
            match="span_claim_role_is_invalid",
        ):
            replace(claim, role=role)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "wire",
    (
        b'{"a":1, "b":2}\n',
        b'{"a":1,"a":1}\n',
        b'{"a":1}\r\n',
        b'\xef\xbb\xbf{"a":1}\n',
        b'{"a":1.0}\n',
        b'{"a":NaN}\n',
        b'{"a":1}\n\n',
    ),
)
def test_exact_json_parser_rejects_noncanonical_or_ambiguous_wire(wire: bytes) -> None:
    with pytest.raises(MexcOfficialEvidenceContractError):
        parse_canonical_json_lf_v1(wire, max_bytes=1024)


def test_exact_json_parsers_reject_extra_or_missing_keys() -> None:
    prepared = _prepared()
    extra = prepared.as_dict()
    extra["support"] = True
    with pytest.raises(MexcOfficialEvidenceContractError):
        OfficialReferencePreparedRequestV1.from_dict(extra)
    missing = prepared.as_dict()
    del missing["trust_env"]
    with pytest.raises(MexcOfficialEvidenceContractError):
        OfficialReferencePreparedRequestV1.from_dict(missing)


@pytest.mark.parametrize(
    "wire",
    (
        ("[" * 33 + "0" + "]" * 33 + "\n").encode("ascii"),
        ("[" + ",".join("0" for _ in range(4097)) + "]\n").encode("ascii"),
        json.dumps("x" * 8193, separators=(",", ":")).encode("utf-8") + b"\n",
        b"123456789012345678901\n",
    ),
)
def test_json_parser_enforces_depth_container_string_and_integer_bounds(
    wire: bytes,
) -> None:
    with pytest.raises(MexcOfficialEvidenceContractError):
        parse_canonical_json_lf_v1(wire, max_bytes=1024 * 1024)


@pytest.mark.parametrize("raised", (TypeError("fixture"), RecursionError("fixture")))
def test_json_parser_wraps_platform_parser_failures(
    monkeypatch: pytest.MonkeyPatch, raised: BaseException
) -> None:
    def fail(*_args, **_kwargs):
        raise raised

    monkeypatch.setattr(official_module.json, "loads", fail)
    with pytest.raises(MexcOfficialEvidenceContractError, match="decode_failed"):
        parse_canonical_json_lf_v1(b'{"a":1}\n', max_bytes=1024)


def test_raw_body_requires_exact_bytes_before_len_or_hash(tmp_path: Path) -> None:
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt)
    with pytest.raises(MexcOfficialEvidenceContractError):
        build_exact_span_claims_v1(bytearray(raw_body))  # type: ignore[arg-type]
    with pytest.raises(MexcOfficialEvidenceContractError):
        parse_canonical_json_lf_v1(bytearray(b'{}\n'), max_bytes=16)  # type: ignore[arg-type]
    output_root = tmp_path / "non-bytes"
    with pytest.raises(MexcOfficialEvidenceContractError, match="exact_bytes"):
        publish_official_evidence_bundle_v1(
            output_root=output_root,
            raw_body=bytearray(raw_body),  # type: ignore[arg-type]
            attempt=attempt,
            evidence=evidence,
            prepared_request=prepared,
            compatibility=_compatibility(),
        )
    assert not output_root.exists()


def test_prewrite_roundtrip_failure_cannot_strand_bundle_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt)

    def reject_roundtrip(_cls, _payload):
        raise MexcOfficialEvidenceContractError("fixture_bundle_roundtrip_rejected")

    monkeypatch.setattr(
        OfficialEvidenceBundleV1,
        "from_dict",
        classmethod(reject_roundtrip),
    )
    output_root = tmp_path / "roundtrip-rejected"
    with pytest.raises(
        MexcOfficialEvidenceContractError,
        match="fixture_bundle_roundtrip_rejected",
    ):
        publish_official_evidence_bundle_v1(
            output_root=output_root,
            raw_body=raw_body,
            attempt=attempt,
            evidence=evidence,
            prepared_request=prepared,
            compatibility=_compatibility(),
        )
    assert not output_root.exists()


def test_exact_bundle_namespace_and_fresh_reload(tmp_path: Path) -> None:
    output_root, raw_body, prepared, attempt, evidence, compatibility, bundle = _bundle(
        tmp_path
    )
    plan_hash = prepared.verification_plan_hash
    expected = (
        f"endpoint-evidence/{plan_hash}/official/attempt-000.body.bin",
        f"endpoint-evidence/{plan_hash}/official/attempt-000.receipt.json",
        f"endpoint-evidence/{plan_hash}/official/evidence.json",
    )
    assert derive_official_bundle_root_v1(plan_hash) == (
        f"endpoint-evidence/{plan_hash}/official"
    )
    assert official_bundle_relative_paths_v1(plan_hash) == expected
    assert tuple(item.relative_path for item in bundle.files) == expected
    assert bundle.contract_version == OFFICIAL_EVIDENCE_BUNDLE_VERSION
    assert bundle.total_storage_bytes == sum(item.byte_count for item in bundle.files)
    assert (output_root / Path(expected[0])).read_bytes() == raw_body
    reloaded = reload_official_evidence_bundle_v1(
        output_root=output_root,
        verification_plan_hash=plan_hash,
        prepared_request=prepared,
        compatibility=compatibility,
    )
    assert reloaded == bundle
    assert bundle.attempt_receipt_hash == attempt.attempt_receipt_hash
    assert bundle.evidence_hash == evidence.evidence_hash
    assert bundle.raw_body_sha256 == evidence.raw_body_sha256
    assert bundle.files[0].byte_count == evidence.raw_body_byte_count
    assert bundle.files[1].byte_count == evidence.attempt_receipt_byte_count
    assert bundle.files[2].byte_count == len(evidence.canonical_lf_bytes)


def test_official_namespace_is_exact_frozen_layout_helper_output() -> None:
    from trading.market_data.mexc_pilot_output_layout import (
        derive_official_bundle_locators_v1,
        derive_official_bundle_root_v1 as derive_layout_root,
    )

    plan_hash = "a" * 64
    assert derive_official_bundle_root_v1(plan_hash) == derive_layout_root(plan_hash)
    assert official_bundle_relative_paths_v1(plan_hash) == (
        derive_official_bundle_locators_v1(plan_hash)
    )


@pytest.mark.parametrize("dependency_part", ("version", "hash"))
def test_layout_version_or_hash_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    dependency_part: str,
) -> None:
    if dependency_part == "version":
        monkeypatch.setattr(
            official_module._pilot_output_layout,
            "PILOT_OUTPUT_LAYOUT_CONTRACT_VERSION",
            "mexc_public_qa_pilot_output_layout_v2",
        )
    else:
        monkeypatch.setattr(
            official_module._pilot_output_layout,
            "pilot_output_layout_contract_hash",
            lambda: "0" * 64,
        )
    with pytest.raises(MexcOfficialEvidenceContractError, match="dependency_drift"):
        derive_official_bundle_root_v1("a" * 64)
    with pytest.raises(MexcOfficialEvidenceContractError, match="dependency_drift"):
        official_evidence_contract_descriptor_v1()


def test_layout_canonical_locator_helper_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_hash = "a" * 64
    root = f"endpoint-evidence/{plan_hash}/official"
    monkeypatch.setattr(
        official_module._pilot_output_layout,
        "derive_official_bundle_locators_v1",
        lambda _plan_hash: (
            f"{root}/attempt-000.raw",
            f"{root}/attempt-000.receipt.json",
            f"{root}/evidence.json",
        ),
    )
    with pytest.raises(MexcOfficialEvidenceContractError, match="locator_drift"):
        official_bundle_relative_paths_v1(plan_hash)
    with pytest.raises(MexcOfficialEvidenceContractError, match="locator_drift"):
        mexc_endpoint_official_evidence_contract_hash()


def test_bundle_contract_binds_raw_file_hash_to_evidence_raw_hash(
    tmp_path: Path,
) -> None:
    _, _, _, _, _, _, bundle = _bundle(tmp_path)
    with pytest.raises(MexcOfficialEvidenceContractError, match="raw_body_hash"):
        replace(bundle, raw_body_sha256="0" * 64)


def test_bundle_file_role_requires_exact_string_before_membership(
    tmp_path: Path,
) -> None:
    _, _, _, _, _, _, bundle = _bundle(tmp_path)

    class StringAlias(str):
        pass

    assert type(bundle.files[0]) is OfficialEvidenceBundleFileV1
    for role in ([], StringAlias("raw_body")):
        with pytest.raises(
            MexcOfficialEvidenceContractError,
            match="bundle_file_role_is_invalid",
        ):
            replace(bundle.files[0], role=role)  # type: ignore[arg-type]


def test_incomplete_attempt_bundle_is_intentionally_unsupported(
    tmp_path: Path,
) -> None:
    raw_body = _body()
    prepared = _prepared()
    complete = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, complete)
    incomplete = _failure_attempt(prepared, "incomplete_transport_error")
    output_root = tmp_path / "incomplete-unsupported"
    with pytest.raises(MexcOfficialEvidenceContractError, match="incomplete_attempt"):
        publish_official_evidence_bundle_v1(
            output_root=output_root,
            raw_body=raw_body,
            attempt=incomplete,
            evidence=evidence,
            prepared_request=prepared,
            compatibility=_compatibility(),
        )
    assert not output_root.exists()


def test_plan_root_sibling_inventory_is_explicitly_delegated_and_terminal_stops(
    tmp_path: Path,
) -> None:
    output_root, _, prepared, _, evidence, compatibility, bundle = _bundle(tmp_path)
    plan_root = (
        output_root
        / "endpoint-evidence"
        / prepared.verification_plan_hash
    )
    (plan_root / "unexpected-sibling.txt").write_bytes(b"delegated residue")
    assert reload_official_evidence_bundle_v1(
        output_root=output_root,
        verification_plan_hash=prepared.verification_plan_hash,
        prepared_request=prepared,
        compatibility=compatibility,
    ) == bundle
    with pytest.raises(MexcOfficialEvidenceTerminalStop) as caught:
        require_terminal_compatible_official_evidence_v1(
            bundle=bundle,
            evidence=evidence,
            compatibility=compatibility,
        )
    assert (
        "plan_root_sibling_inventory_delegated_to_future_pinned_output_layout"
        in caught.value.blockers
    )


def test_bundle_publish_is_create_new_and_rejects_residue(tmp_path: Path) -> None:
    output_root, raw_body, prepared, attempt, evidence, compatibility, _ = _bundle(
        tmp_path
    )
    with pytest.raises(MexcOfficialEvidenceStorageStop):
        publish_official_evidence_bundle_v1(
            output_root=output_root,
            raw_body=raw_body,
            attempt=attempt,
            evidence=evidence,
            prepared_request=prepared,
            compatibility=compatibility,
        )
    second_root = tmp_path / "second-output"
    directory = second_root / derive_official_bundle_root_v1(
        prepared.verification_plan_hash
    )
    directory.mkdir(parents=True)
    (directory / ".attempt-000.tmp").write_bytes(b"residue")
    with pytest.raises(
        MexcOfficialEvidenceStorageStop,
        match="official_bundle_directory_contains_residue",
    ):
        publish_official_evidence_bundle_v1(
            output_root=second_root,
            raw_body=raw_body,
            attempt=attempt,
            evidence=evidence,
            prepared_request=prepared,
            compatibility=compatibility,
        )


def test_publish_and_reload_reject_expired_deadline_before_io(tmp_path: Path) -> None:
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt)
    compatibility = _compatibility()
    never_created = tmp_path / "expired-publish"
    with pytest.raises(MexcOfficialEvidenceStorageStop, match="deadline"):
        publish_official_evidence_bundle_v1(
            output_root=never_created,
            raw_body=raw_body,
            attempt=attempt,
            evidence=evidence,
            prepared_request=prepared,
            compatibility=compatibility,
            deadline_monotonic_ns=time.monotonic_ns() - 1,
        )
    assert not never_created.exists()
    output_root, _, prepared, _, _, compatibility, _ = _bundle(
        tmp_path / "published"
    )
    with pytest.raises(MexcOfficialEvidenceStorageStop, match="deadline"):
        reload_official_evidence_bundle_v1(
            output_root=output_root,
            verification_plan_hash=prepared.verification_plan_hash,
            prepared_request=prepared,
            compatibility=compatibility,
            deadline_monotonic_ns=time.monotonic_ns() - 1,
        )


def test_observed_official_runtime_cap_plus_one_stops_before_publish(
    tmp_path: Path,
) -> None:
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt)
    compatibility = _compatibility()
    forged_slow = replace(
        evidence,
        reload_completed_at_us=(
            attempt.request_started_at_us
            + compatibility.residual_official_runtime_us
            + 1
        ),
        reload_completed_monotonic_us=(
            attempt.request_started_monotonic_us
            + compatibility.residual_official_runtime_us
            + 1
        ),
    )
    output_root = tmp_path / "runtime-cap"
    with pytest.raises(MexcOfficialEvidenceBudgetStop, match="runtime_budget"):
        publish_official_evidence_bundle_v1(
            output_root=output_root,
            raw_body=raw_body,
            attempt=attempt,
            evidence=forged_slow,
            prepared_request=prepared,
            compatibility=compatibility,
        )
    assert not output_root.exists()


@pytest.mark.parametrize("failed_operation", ("write", "fsync"))
def test_platform_write_failures_are_typed_and_leave_nonresumable_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_operation: str
) -> None:
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt)
    compatibility = _compatibility()
    output_root = tmp_path / f"typed-{failed_operation}"
    original = getattr(os, failed_operation)

    def fail(*_args, **_kwargs):
        raise OSError(f"fixture {failed_operation} failure")

    monkeypatch.setattr(os, failed_operation, fail)
    with pytest.raises(
        MexcOfficialEvidenceStorageStop,
        match="create_new_write_failed",
    ):
        publish_official_evidence_bundle_v1(
            output_root=output_root,
            raw_body=raw_body,
            attempt=attempt,
            evidence=evidence,
            prepared_request=prepared,
            compatibility=compatibility,
        )
    monkeypatch.setattr(os, failed_operation, original)
    body_path = output_root / attempt.raw_body_relative_path
    assert body_path.exists()
    with pytest.raises(MexcOfficialEvidenceStorageStop, match="residue"):
        publish_official_evidence_bundle_v1(
            output_root=output_root,
            raw_body=raw_body,
            attempt=attempt,
            evidence=evidence,
            prepared_request=prepared,
            compatibility=compatibility,
        )


def test_platform_read_failure_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, _, prepared, _, _, compatibility, _ = _bundle(tmp_path)
    original = os.read

    def fail(*_args, **_kwargs):
        raise OSError("fixture read failure")

    monkeypatch.setattr(os, "read", fail)
    with pytest.raises(MexcOfficialEvidenceStorageStop, match="artifact_read_failed"):
        reload_official_evidence_bundle_v1(
            output_root=output_root,
            verification_plan_hash=prepared.verification_plan_hash,
            prepared_request=prepared,
            compatibility=compatibility,
        )
    monkeypatch.setattr(os, "read", original)


def test_bundle_rejects_body_cap_storage_cap_and_stale_namespace(tmp_path: Path) -> None:
    plan = _verification_plan()
    prepared = _prepared(plan_hash=plan.plan_hash)
    compatibility = _compatibility(verification_plan=plan)
    raw_body = _body() + b"x" * (
        compatibility.residual_official_raw_body_bytes - len(_body()) + 1
    )
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt)
    with pytest.raises(MexcOfficialEvidenceBudgetStop, match="raw_body_budget"):
        publish_official_evidence_bundle_v1(
            output_root=tmp_path / "raw-cap",
            raw_body=raw_body,
            attempt=attempt,
            evidence=evidence,
            prepared_request=prepared,
            compatibility=compatibility,
        )
    tiny_storage_plan = _verification_plan(max_total_storage_bytes=4 * 1024**2 + 1)
    tiny_prepared = _prepared(plan_hash=tiny_storage_plan.plan_hash)
    tiny_compatibility = _compatibility(verification_plan=tiny_storage_plan)
    small_body = _body()
    tiny_attempt = _attempt(small_body, tiny_prepared)
    tiny_evidence = _evidence(small_body, tiny_prepared, tiny_attempt)
    with pytest.raises(MexcOfficialEvidenceBudgetStop, match="storage_budget"):
        publish_official_evidence_bundle_v1(
            output_root=tmp_path / "storage-cap",
            raw_body=small_body,
            attempt=tiny_attempt,
            evidence=tiny_evidence,
            prepared_request=tiny_prepared,
            compatibility=tiny_compatibility,
        )
    output_root, _, prepared, _, _, compatibility, _ = _bundle(tmp_path / "ok")
    with pytest.raises(MexcOfficialEvidenceContractError, match="reload_binding"):
        reload_official_evidence_bundle_v1(
            output_root=output_root,
            verification_plan_hash="0" * 64,
            prepared_request=prepared,
            compatibility=compatibility,
        )


def test_bundle_reload_rejects_hardlink_alias(tmp_path: Path) -> None:
    output_root, _, prepared, _, _, compatibility, _ = _bundle(tmp_path)
    plan_hash = prepared.verification_plan_hash
    body_path = output_root / official_bundle_relative_paths_v1(plan_hash)[0]
    alias = tmp_path / "body-hardlink-alias.bin"
    try:
        os.link(body_path, alias)
    except OSError as exc:
        pytest.skip(f"hardlinks unsupported: {exc}")
    with pytest.raises(MexcOfficialEvidenceStorageStop, match="aliased"):
        reload_official_evidence_bundle_v1(
            output_root=output_root,
            verification_plan_hash=plan_hash,
            prepared_request=prepared,
            compatibility=compatibility,
        )


def test_bundle_reload_rejects_symlink_leaf(tmp_path: Path) -> None:
    output_root, _, prepared, _, _, compatibility, _ = _bundle(tmp_path)
    plan_hash = prepared.verification_plan_hash
    evidence_path = output_root / official_bundle_relative_paths_v1(plan_hash)[2]
    replacement = tmp_path / "replacement-evidence.json"
    replacement.write_bytes(evidence_path.read_bytes())
    evidence_path.unlink()
    try:
        evidence_path.symlink_to(replacement)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(MexcOfficialEvidenceStorageStop, match="aliased"):
        reload_official_evidence_bundle_v1(
            output_root=output_root,
            verification_plan_hash=plan_hash,
            prepared_request=prepared,
            compatibility=compatibility,
        )


def test_bundle_publish_rejects_symlink_or_junction_chain(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt)
    with pytest.raises(MexcOfficialEvidenceStorageStop, match="reparse"):
        publish_official_evidence_bundle_v1(
            output_root=alias,
            raw_body=raw_body,
            attempt=attempt,
            evidence=evidence,
            prepared_request=prepared,
            compatibility=_compatibility(),
        )


@pytest.mark.skipif(os.name != "nt", reason="NTFS alternate data streams are Windows-only")
def test_bundle_reload_rejects_ntfs_alternate_data_stream(tmp_path: Path) -> None:
    output_root, _, prepared, _, _, compatibility, _ = _bundle(tmp_path)
    plan_hash = prepared.verification_plan_hash
    body_path = output_root / official_bundle_relative_paths_v1(plan_hash)[0]
    try:
        with open(f"{body_path}:secret", "wb") as handle:
            handle.write(b"secret")
    except OSError as exc:
        pytest.skip(f"named streams unsupported on this volume: {exc}")
    with pytest.raises(MexcOfficialEvidenceStorageStop, match="named_stream"):
        reload_official_evidence_bundle_v1(
            output_root=output_root,
            verification_plan_hash=plan_hash,
            prepared_request=prepared,
            compatibility=compatibility,
        )


@pytest.mark.skipif(os.name != "nt", reason="NTFS alternate data streams are Windows-only")
@pytest.mark.parametrize("location", ("output_root", "official_directory"))
def test_bundle_rejects_directory_named_streams(
    tmp_path: Path, location: str
) -> None:
    if location == "output_root":
        output_root = tmp_path / "directory-ads-output"
        output_root.mkdir()
        stream_target = output_root
        raw_body = _body()
        prepared = _prepared()
        attempt = _attempt(raw_body, prepared)
        evidence = _evidence(raw_body, prepared, attempt)
        compatibility = _compatibility()
    else:
        output_root, _, prepared, _, _, compatibility, _ = _bundle(tmp_path)
        stream_target = output_root / derive_official_bundle_root_v1(
            prepared.verification_plan_hash
        )
    try:
        with open(f"{stream_target}:secret", "wb") as handle:
            handle.write(b"secret")
    except OSError as exc:
        pytest.skip(f"directory named streams unsupported on this volume: {exc}")
    if location == "output_root":
        with pytest.raises(MexcOfficialEvidenceStorageStop, match="named_stream"):
            publish_official_evidence_bundle_v1(
                output_root=output_root,
                raw_body=raw_body,
                attempt=attempt,
                evidence=evidence,
                prepared_request=prepared,
                compatibility=compatibility,
            )
    else:
        with pytest.raises(MexcOfficialEvidenceStorageStop, match="named_stream"):
            reload_official_evidence_bundle_v1(
                output_root=output_root,
                verification_plan_hash=prepared.verification_plan_hash,
                prepared_request=prepared,
                compatibility=compatibility,
            )


@pytest.mark.skipif(os.name != "nt", reason="8.3 aliases are Windows-only")
def test_bundle_rejects_windows_short_path_alias(tmp_path: Path) -> None:
    import ctypes
    from ctypes import wintypes

    long_root = tmp_path / "official-evidence-long-output-name"
    long_root.mkdir()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_short = kernel32.GetShortPathNameW
    get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    written = get_short(str(long_root), buffer, len(buffer))
    if written == 0 or os.path.normcase(buffer.value) == os.path.normcase(str(long_root)):
        pytest.skip("8.3 aliases are disabled on this volume")
    raw_body = _body()
    prepared = _prepared()
    attempt = _attempt(raw_body, prepared)
    evidence = _evidence(raw_body, prepared, attempt)
    with pytest.raises(MexcOfficialEvidenceStorageStop, match="short_path_alias"):
        publish_official_evidence_bundle_v1(
            output_root=buffer.value,
            raw_body=raw_body,
            attempt=attempt,
            evidence=evidence,
            prepared_request=prepared,
            compatibility=_compatibility(),
        )


def _verification_plan(
    *,
    max_network_attempts: int = 2,
    max_total_raw_body_bytes: int = 2 * 1024**2,
    max_total_storage_bytes: int = 8 * 1024**2,
    max_runtime_us: int = 2 * 60 * 1_000_000,
) -> EndpointVerificationPlanV1:
    endpoint = load_mexc_futures_endpoint_contract_v1(candidate_endpoint_fixture_path())
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
    request = HistoryRangeRequestV2(
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        interval="Min1",
        start_open_ts=1_767_225_540,
        end_open_ts_exclusive=1_767_225_600,
        collection_as_of_us=1_767_225_600_000_000,
        endpoint_contract=endpoint,
        resource_limits=limits,
        retry_policy=candidate_history_retry_policy_v1(),
        page_size=1,
        storage_profile="windows_ntfs_hardlink_best_effort_v1",
    )
    return EndpointVerificationPlanV1(
        probe_request=request,
        relative_artifact_root=f"verification/{request.request_id}",
        official_reference_url=OFFICIAL_REFERENCE_URL,
        verifier_contract_version="fixture_endpoint_verifier_v1",
        verifier_contract_hash="a" * 64,
        max_network_attempts=max_network_attempts,
        max_total_raw_body_bytes=max_total_raw_body_bytes,
        max_total_storage_bytes=max_total_storage_bytes,
        max_runtime_us=max_runtime_us,
        max_total_sleep_us=30 * 1_000_000,
    )


def test_compatibility_requires_two_attempts_and_frozen_probe_residuals() -> None:
    plan = _verification_plan()
    result = assess_official_evidence_compatibility_v1(
        verification_plan=plan,
    )
    assert result.contract_version == OFFICIAL_EVIDENCE_COMPATIBILITY_VERSION
    assert result.max_network_attempts == 2
    assert result.residual_official_raw_body_bytes == 1024**2
    assert result.residual_official_storage_bytes == 4 * 1024**2
    assert result.residual_official_runtime_us == 60 * 1_000_000
    assert OfficialEvidenceCompatibilityV1.from_dict(result.as_dict()) == result
    with pytest.raises(MexcOfficialEvidenceBudgetStop, match="exactly_two"):
        assess_official_evidence_compatibility_v1(
            verification_plan=_verification_plan(max_network_attempts=3),
        )
    with pytest.raises(MexcOfficialEvidenceBudgetStop, match="live_reservation"):
        replace(
            result,
            reserved_live_raw_body_bytes=1,
            reserved_live_storage_bytes=1,
            reserved_live_runtime_us=1,
            residual_official_raw_body_bytes=result.max_total_raw_body_bytes - 1,
            residual_official_storage_bytes=result.max_total_storage_bytes - 1,
            residual_official_runtime_us=result.max_runtime_us - 1,
        )
    with pytest.raises(MexcOfficialEvidenceBudgetStop, match="nonpositive"):
        assess_official_evidence_compatibility_v1(
            verification_plan=_verification_plan(
                max_total_raw_body_bytes=1024**2,
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_network_attempts", True),
        ("max_total_raw_body_bytes", True),
        ("reserved_live_storage_bytes", False),
        ("residual_official_runtime_us", 60_000_000.0),
        ("terminal_compatible", 0),
    ),
)
def test_compatibility_rejects_every_scalar_type_alias(
    field: str, value: object
) -> None:
    with pytest.raises(MexcOfficialEvidenceContractError):
        replace(_compatibility(), **{field: value})


def test_fake_bundle_can_never_authorize_terminal_adapter(tmp_path: Path) -> None:
    _, _, _, _, evidence, compatibility, bundle = _bundle(tmp_path, full=True)
    assert bundle.authority_status == REVIEWED_FAKE_FIXTURE_ONLY
    assert bundle.terminal_compatible is False
    assert evidence.terminal_compatible is False
    assert compatibility.terminal_compatible is False
    with pytest.raises(MexcOfficialEvidenceTerminalStop) as caught:
        require_terminal_compatible_official_evidence_v1(
            bundle=bundle,
            evidence=evidence,
            compatibility=compatibility,
        )
    assert caught.value.code == "official_evidence_v1_is_not_terminal_compatible"
    assert caught.value.classification == "reviewed_fake_structural_nonterminal"
    assert caught.value.blockers == (
        "official_bundle_namespace_absent_from_frozen_preflight",
        "observed_current_official_body_version_absent",
        "single_migration_announcement_may_not_prove_full_candidate_contract",
        "live_reload_inventory_anchor_overhead_unreserved",
        "official_store_host_clock_not_bound_to_attempt_clock",
        "attempt_parent_receipt_hashes_are_opaque_not_fresh_source_objects",
        "runtime_tls_trust_bindings_are_declarative_not_attested",
        "attempt_and_evidence_clock_samples_are_structural_fake_only",
        "incomplete_or_failure_official_attempt_bundle_unsupported",
        "plan_root_sibling_inventory_delegated_to_future_pinned_output_layout",
        "partial_three_file_publication_is_nonresumable_and_not_transactional",
        "hostile_concurrent_filesystem_toctou_boundary_unaccepted",
        "terminal_endpoint_receipt_publisher_unbound",
    )


def test_terminal_adapter_rejects_every_bundle_cross_binding_mismatch(
    tmp_path: Path,
) -> None:
    _, _, _, _, evidence, compatibility, bundle = _bundle(tmp_path)
    with pytest.raises(MexcOfficialEvidenceContractError, match="binding_mismatch"):
        require_terminal_compatible_official_evidence_v1(
            bundle=replace(bundle, prepared_request_hash="f" * 64),
            evidence=evidence,
            compatibility=compatibility,
        )

    forged_attempt_hash = "f" * 64
    forged_receipt_file = replace(
        bundle.files[1], artifact_sha256=forged_attempt_hash
    )
    with pytest.raises(MexcOfficialEvidenceContractError, match="binding_mismatch"):
        require_terminal_compatible_official_evidence_v1(
            bundle=replace(
                bundle,
                attempt_receipt_hash=forged_attempt_hash,
                files=(bundle.files[0], forged_receipt_file, bundle.files[2]),
            ),
            evidence=evidence,
            compatibility=compatibility,
        )


@pytest.mark.parametrize("file_index", (0, 1, 2))
def test_terminal_adapter_rejects_each_bundle_file_byte_count_mismatch(
    tmp_path: Path,
    file_index: int,
) -> None:
    _, _, _, _, evidence, compatibility, bundle = _bundle(tmp_path)
    forged_files = list(bundle.files)
    forged_files[file_index] = replace(
        forged_files[file_index],
        byte_count=forged_files[file_index].byte_count + 1,
    )
    with pytest.raises(MexcOfficialEvidenceContractError, match="binding_mismatch"):
        require_terminal_compatible_official_evidence_v1(
            bundle=replace(bundle, files=tuple(forged_files)),
            evidence=evidence,
            compatibility=compatibility,
        )


def test_terminal_adapter_does_not_trust_structural_support_scope(
    tmp_path: Path,
) -> None:
    _, _, _, _, evidence, compatibility, bundle = _bundle(tmp_path)
    structural_forgery = replace(
        evidence,
        support_scope="full_candidate_contract",
        verdict="candidate_contract_semantics_observed",
    )
    forged_evidence_file = replace(
        bundle.files[2],
        artifact_sha256=structural_forgery.evidence_hash,
        byte_count=len(structural_forgery.canonical_lf_bytes),
    )
    forged_bundle = replace(
        bundle,
        evidence_hash=structural_forgery.evidence_hash,
        files=(bundle.files[0], bundle.files[1], forged_evidence_file),
    )
    with pytest.raises(MexcOfficialEvidenceTerminalStop) as caught:
        require_terminal_compatible_official_evidence_v1(
            bundle=forged_bundle,
            evidence=structural_forgery,
            compatibility=compatibility,
        )
    assert caught.value.classification == "reviewed_fake_structural_nonterminal"


def test_contract_versions_and_schema_hash_are_stable_shape() -> None:
    from trading.market_data.mexc_pilot_output_layout import (
        PILOT_OUTPUT_LAYOUT_CONTRACT_VERSION,
        pilot_output_layout_contract_hash,
    )
    from trading.market_data.mexc_pilot_run import (
        PILOT_RUN_CONTRACT_VERSION,
        pilot_run_contract_hash,
    )

    assert OFFICIAL_REFERENCE_REQUEST_VERSION == "mexc_endpoint_official_reference_request_v1"
    assert OFFICIAL_REFERENCE_HTTP_ATTEMPT_VERSION == (
        "mexc_endpoint_official_reference_http_attempt_v1"
    )
    assert OFFICIAL_DOCUMENT_EVIDENCE_VERSION == "mexc_endpoint_official_document_evidence_v1"
    assert OFFICIAL_DOCUMENT_READER_VERSION == "mexc_endpoint_official_document_reader_v1"
    assert OFFICIAL_EVIDENCE_STORE_VERSION == "mexc_endpoint_official_evidence_store_v1"
    assert OFFICIAL_STORAGE_CONCURRENCY_BOUNDARY == (
        "static_or_cooperating_writers_only_no_atomic_directory_snapshot_or_toctou_guarantee"
    )
    descriptor = official_evidence_contract_descriptor_v1()
    assert descriptor["public_api_signatures"][
        "assess_official_evidence_compatibility_v1"
    ] == (
        "(*,verification_plan:EndpointVerificationPlanV1)"
        "->OfficialEvidenceCompatibilityV1"
    )
    assert descriptor["attempt_validation"]["outcome_state_matrix"]["complete"][
        "status"
    ] == 200
    assert descriptor["attempt_validation"]["terminal_progress_phases"] == [
        "before_tls_validation",
        "tls_validation_failed",
        "tls_validated_before_headers",
        "headers_received_before_body_eof",
        "body_eof",
    ]
    assert descriptor["attempt_validation"]["safe_error_predicates"][
        "content_type_not_official_html"
    ]["predicate"] == "status_200_identity_content_type_not_html_raw_empty"
    assert descriptor["attempt_validation"]["safe_error_predicates"][
        "content_length_invalid"
    ]["predicate"] == (
        "status_200_identity_html_content_length_present_noncanonical_raw_empty"
    )
    assert descriptor["bundle"]["publish_order"][0] == (
        "validate_exact_bindings_and_budgets"
    )
    assert descriptor["canonical_json"]["duplicate_keys"] == "forbidden"
    assert descriptor["canonical_json"]["max_depth"] == 32
    assert descriptor["exact_scalar_types"]["http_attempt"][
        "attempt_ordinal"
    ] == "exact_int_0"
    assert descriptor["hash_formulas"]["reader_contract_hash"] == (
        "mexc_endpoint_official_evidence_contract_hash"
    )
    assert descriptor["dependency_bindings"]["pilot_run"] == {
        "version": PILOT_RUN_CONTRACT_VERSION,
        "hash": pilot_run_contract_hash(),
        "status": "exact_frozen_dependency",
    }
    assert descriptor["dependency_bindings"]["pilot_output_layout"] == {
        "version": PILOT_OUTPUT_LAYOUT_CONTRACT_VERSION,
        "hash": pilot_output_layout_contract_hash(),
        "status": "exact_frozen_dependency_and_canonical_helpers",
    }
    assert descriptor["helper_vectors"]["zero_plan_bundle_root"] == (
        derive_official_bundle_root_v1("0" * 64)
    )
    assert tuple(descriptor["helper_vectors"]["zero_plan_files"]) == (
        official_bundle_relative_paths_v1("0" * 64)
    )
    assert descriptor["bundle"]["partial_publication"] == (
        "nontransactional_create_new_prefix_remains_nonresumable_terminal_stop"
    )
    assert descriptor["bundle"]["hostile_concurrent_toctou_terminal_blocker"] == (
        "hostile_concurrent_filesystem_toctou_boundary_unaccepted"
    )
    assert descriptor["authority"]["strict_adapter_classifications"] == [
        "reviewed_fake_structural_nonterminal"
    ]
    assert official_module._PINNED_CONTRACT_HASH == (
        "421802f03282ea5f61f253607001036e80a1933e1d1ea16449c5ee261889e04d"
    )
    assert mexc_endpoint_official_evidence_contract_hash() == (
        official_module._PINNED_CONTRACT_HASH
    )


def test_bundle_and_evidence_exact_parsers_reject_forged_wire(tmp_path: Path) -> None:
    _, _, _, _, evidence, _, bundle = _bundle(tmp_path)
    forged_bundle = bundle.as_dict()
    forged_bundle["terminal_compatible"] = True
    with pytest.raises(MexcOfficialEvidenceContractError):
        OfficialEvidenceBundleV1.from_dict(forged_bundle)
    forged_evidence = evidence.as_dict()
    forged_evidence["support_scope"] = "full_candidate_contract"
    forged_evidence["verdict"] = "candidate_contract_semantics_observed"
    # Structural parsing alone is deliberately not authority.  The bundle
    # reader below must reparse the raw body and reject the forged scope.
    structural = OfficialDocumentEvidenceV1.from_dict(forged_evidence)
    assert structural.support_scope == "full_candidate_contract"
    assert _canonical(structural.as_dict()) != evidence.canonical_lf_bytes
