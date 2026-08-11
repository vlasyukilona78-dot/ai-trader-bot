from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import threading

import pytest

from ai.reversal.feature_contract import (
    build_runtime_feature_snapshot,
    market_feature_hash,
    model_feature_names,
)
from ai.reversal.population_dataset import (
    PopulationDatasetError,
    iter_population_cycles,
    model_input_records,
    verify_population_journal,
)
from trading.market_data.bar_contract import closed_boundary_ts, interval_seconds
from trading.market_data.frame_provenance import SourceReadEvidenceV1
from trading.market_data.source_timing import SourceTiming
from trading.metrics.cycle_envelope import CycleEnvelope, CycleEnvelopeError
from trading.metrics.population_journal import (
    CURRENT_WRITE_SCHEMA,
    EVIDENCE_CONTRACT_KEYS,
    HEADER_KEYS_V6,
    JournalCheckpointReceipt,
    PopulationDecisionV6,
    PopulationJournal,
    PopulationJournalError,
    compute_cycle_commit,
    rows_checksum,
)
from trading.signals.lifecycle_contract import (
    CandidateArmV1,
    CandidateLifecycleEventV1,
    CandidateLifecycleState,
    CandidateSide,
    ConfirmationObservationV1,
)

from v2.test_population_feature_dataset_v2 import (
    _STRATEGY_SPEC,
    _UNIVERSE_TIMING,
    _benchmark_evidence,
    _envelope,
    _metadata,
    _records,
    _symbol_evidence,
)


_V5_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "mexc_population_journal_v5_minimal.jsonl"
)


def _write_v6(path: Path) -> PopulationJournal:
    journal = PopulationJournal(path)
    assert journal.append_cycle(_records(), envelope=_envelope()) is True
    return journal


def _recreate(record: PopulationDecisionV6, **changes) -> PopulationDecisionV6:
    values = {
        key: value
        for key, value in record.__dict__.items()
        if key not in {
            "schema_version",
            "snapshot_id",
            "input_hash",
            "raw_frame_bundle_hash",
        }
    }
    values.update(changes)
    return PopulationDecisionV6.create(**values)


def _rehash_cycle(lines: list[dict], header_index: int) -> None:
    footer_index = next(
        index
        for index in range(header_index + 1, len(lines))
        if lines[index]["record_type"] == "cycle_footer"
    )
    body = lines[header_index + 1 : footer_index]
    footer = lines[footer_index]
    footer["rows_checksum"] = rows_checksum(body)
    footer_core = {key: value for key, value in footer.items() if key != "cycle_commit"}
    footer["cycle_commit"] = compute_cycle_commit(
        lines[header_index], body, footer_core
    )


def _source_for_cycle(
    *, symbol: str, timeframe: str, cutoff: float, source: str, digest: str
) -> SourceReadEvidenceV1:
    seconds = interval_seconds(timeframe)
    expected = float(closed_boundary_ts(cutoff, timeframe))
    return SourceReadEvidenceV1(
        source=source,
        venue="mexc_contract",
        symbol=symbol,
        venue_symbol=("BTC_USDT" if symbol == "BTCUSDT" else symbol[:-4] + "_USDT"),
        timeframe=timeframe,
        requested_as_of_ts=cutoff,
        expected_closed_boundary_ts=expected,
        request_started_at=cutoff + 1.0,
        received_at=cutoff + 1.5,
        source_ts=cutoff + 1.4,
        cache_hit=False,
        cache_age_sec=0.0,
        outcome="fresh",
        error_code=None,
        missing_reason=None,
        first_bar_open_ts=expected - 40 * seconds,
        last_bar_open_ts=expected - seconds,
        last_bar_close_ts=expected,
        data_through_ts=expected,
        bar_count=40,
        frame_hash=digest * 64,
    )


def _single_cycle(cutoff: float):
    envelope = _envelope(
        candle_cutoff_ts=cutoff,
        cycle_started_at=cutoff + 1.0,
        universe_symbols=("AAAUSDT",),
        ranking_ready_ts=cutoff + 4.0,
        cycle_completed_ts=cutoff + 5.0,
    )
    metadata = {
        "universe": {
            "turnover_24h_usdt": 1_000_000.0,
            "change_24h": 0.1,
            "funding_rate": 0.0,
            "open_interest": None,
            "min_notional_usdt": None,
            "max_leverage": None,
        },
        "base": {"bar_count": 320, "mark_price": 108.0},
        "benchmark_status": "available",
        "provenance": {
            "strategy_config_hash": envelope.strategy_spec_instance_hash,
            "universe_policy_hash": envelope.universe_policy_hash,
        },
    }
    metadata["feature_snapshot"] = build_runtime_feature_snapshot(
        metadata, bar_cutoff_ts=cutoff
    )
    metadata["feature_provenance"] = {
        "universe_received_at": envelope.universe_timing.received_at,
        "universe_source_ts": envelope.universe_timing.source_ts,
        "universe_cache_hit": envelope.universe_timing.cache_hit,
        "envelope_hash": envelope.envelope_hash(),
        "market_feature_hash": market_feature_hash(
            metadata["feature_snapshot"],
            symbol="AAAUSDT",
            timeframe_seconds=interval_seconds(envelope.timeframe),
        ),
    }
    base = _source_for_cycle(
        symbol="AAAUSDT",
        timeframe=envelope.timeframe,
        cutoff=cutoff,
        source="base_ohlcv",
        digest="4",
    )
    htf = SourceReadEvidenceV1.not_requested(
        source="higher_timeframe_ohlcv",
        venue="mexc_contract",
        symbol="AAAUSDT",
        venue_symbol="AAA_USDT",
        timeframe=_STRATEGY_SPEC.market_data.higher_timeframe.interval,
        requested_as_of_ts=cutoff,
        reason="not_used",
    )
    benchmark = _source_for_cycle(
        symbol="BTCUSDT",
        timeframe=_STRATEGY_SPEC.resolved_benchmark_interval,
        cutoff=cutoff,
        source="benchmark_ohlcv",
        digest="5",
    )
    common = dict(
        cycle_id=envelope.cycle_id,
        universe_refreshed_at=1_700_000_000.0,
        universe_request_started_at=envelope.universe_timing.request_started_at,
        universe_received_at=envelope.universe_timing.received_at,
        scan_observed_at=cutoff + 1.0,
        candle_cutoff_ts=cutoff,
        decision_ts=cutoff + 2.0,
        ranking_ready_ts=envelope.ranking_ready_ts,
        cycle_completed_ts=envelope.cycle_completed_ts,
        actionable_ts=envelope.actionable_ts,
        entry_eligible_ts=envelope.entry_eligible_ts,
        entry_bar_open_ts=envelope.entry_bar_open_ts,
        symbol="AAAUSDT",
        timeframe=envelope.timeframe,
        status="evaluated",
        base_bar_open_ts=cutoff - interval_seconds(envelope.timeframe),
        base_bar_close_ts=cutoff,
        action="HOLD",
        reason="test",
        confidence=0.0,
        metadata=metadata,
        base_source_evidence=base,
        higher_timeframe_source_evidence=htf,
        benchmark_source_evidence=benchmark,
    )
    preliminary = PopulationDecisionV6.create(**common)
    return envelope, common, preliminary


def test_v6_writer_reader_receipt_and_model_export_keep_evidence_non_predictive(
    tmp_path,
) -> None:
    path = tmp_path / "population_v6.jsonl"
    journal = _write_v6(path)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    header, first_row, _second_row, footer = lines

    assert {line["schema_version"] for line in lines} == {CURRENT_WRITE_SCHEMA}
    assert set(header) == HEADER_KEYS_V6
    assert set(header["evidence_contracts"]) == EVIDENCE_CONTRACT_KEYS
    assert header["benchmark_source_evidence"]["source"] == "benchmark_ohlcv"
    assert footer["cycle_commit"] == compute_cycle_commit(
        header,
        lines[1:-1],
        {key: value for key, value in footer.items() if key != "cycle_commit"},
    )

    receipt = journal.checkpoint_receipt()
    assert receipt.journal_schema_version == CURRENT_WRITE_SCHEMA
    trust = verify_population_journal(path, trusted_checkpoint=receipt)
    assert trust.journal_schema_version == CURRENT_WRITE_SCHEMA
    records = model_input_records(path, trusted_checkpoint=receipt)
    assert records[0]["typed_evidence_status"] == "typed_v6"
    assert records[0]["raw_frame_bundle_hash"] == first_row["raw_frame_bundle_hash"]
    assert tuple(records[0]["features"]) == model_feature_names()
    assert not {
        "raw_frame_bundle_hash",
        "lifecycle_event",
        "base_source_evidence",
    } & set(records[0]["features"])
    assert not {
        "raw_frame_bundle_hash",
        "lifecycle_event",
        "base_source_evidence",
    } & set(records[0]["observed"])


def test_v5_is_readable_but_read_only_and_model_export_requires_opt_in(tmp_path) -> None:
    copied = tmp_path / "legacy_v5.jsonl"
    shutil.copyfile(_V5_FIXTURE, copied)
    legacy = PopulationJournal(copied)
    assert legacy.checkpoint_receipt().journal_schema_version == 5
    with pytest.raises(PopulationDatasetError, match="legacy_v5_model_export"):
        model_input_records(copied, allow_unanchored=True)
    assert model_input_records(
        copied, allow_unanchored=True, allow_legacy_v5=True
    )[0]["typed_evidence_status"] == "legacy_missing"
    with pytest.raises(PopulationJournalError, match="schema v5 is frozen read-only"):
        legacy.append_cycle(_records(), envelope=_envelope())


def test_evaluated_stale_base_and_quality_error_no_rows_fail_closed(tmp_path) -> None:
    records = _records()
    original = records[0]
    base = original.base_source_evidence
    assert base is not None
    stale = replace(
        base,
        outcome="stale",
        missing_reason="data_lag",
        last_bar_open_ts=base.last_bar_open_ts - 3600.0,
        last_bar_close_ts=base.last_bar_close_ts - 3600.0,
        data_through_ts=base.data_through_ts - 3600.0,
        frame_hash="d" * 64,
    )
    stale_record = PopulationDecisionV6.create(
        **{
            **{
                key: value
                for key, value in original.__dict__.items()
                if key not in {
                    "schema_version",
                    "snapshot_id",
                    "input_hash",
                    "raw_frame_bundle_hash",
                }
            },
            "base_source_evidence": stale,
        }
    )
    with pytest.raises(PopulationJournalError, match="requires fresh base evidence"):
        PopulationJournal(tmp_path / "stale.jsonl").append_cycle(
            [stale_record, records[1]], envelope=_envelope()
        )

    no_rows_payload = base.as_dict()
    no_rows_payload.update(
        outcome="no_rows",
        error_code=None,
        missing_reason="no_rows",
        first_bar_open_ts=None,
        last_bar_open_ts=None,
        last_bar_close_ts=None,
        data_through_ts=None,
        bar_count=0,
        frame_hash=None,
    )
    no_rows = SourceReadEvidenceV1.from_dict(no_rows_payload)
    quality = PopulationDecisionV6.create(
        **{
            **{
                key: value
                for key, value in original.__dict__.items()
                if key not in {
                    "schema_version",
                    "snapshot_id",
                    "input_hash",
                    "raw_frame_bundle_hash",
                    "status",
                    "base_bar_open_ts",
                    "base_bar_close_ts",
                    "error_code",
                }
            },
            "status": "data_quality_error",
            "base_bar_open_ts": None,
            "base_bar_close_ts": None,
            "error_code": "FrameQualityError",
            "base_source_evidence": no_rows,
        }
    )
    with pytest.raises(PopulationJournalError, match="requires stale or failed"):
        PopulationJournal(tmp_path / "quality.jsonl").append_cycle(
            [quality, records[1]], envelope=_envelope()
        )


def test_entry_action_without_lifecycle_event_is_rejected(tmp_path) -> None:
    envelope, common, _ = _single_cycle(1_700_006_400.0)
    htf = _source_for_cycle(
        symbol="AAAUSDT",
        timeframe=_STRATEGY_SPEC.market_data.higher_timeframe.interval,
        cutoff=envelope.candle_cutoff_ts,
        source="higher_timeframe_ohlcv",
        digest="6",
    )
    metadata = dict(common["metadata"])
    metadata["stop_loss"] = 111.0
    metadata["take_profit"] = 100.0
    record = PopulationDecisionV6.create(
        **{
            **common,
            "action": "SHORT_ENTRY",
            "metadata": metadata,
            "higher_timeframe_source_evidence": htf,
        }
    )
    with pytest.raises(PopulationJournalError, match="requires typed lifecycle"):
        PopulationJournal(tmp_path / "entry.jsonl").append_cycle(
            [record], envelope=envelope
        )


def test_cross_cycle_forged_predecessor_fails_after_coherent_rehash(tmp_path) -> None:
    first_envelope, first_common, first_preliminary = _single_cycle(
        1_700_006_400.0
    )
    arm = CandidateArmV1(
        strategy_spec_version=first_envelope.strategy_spec_version,
        strategy_spec_contract_hash=first_envelope.strategy_spec_contract_hash,
        strategy_spec_instance_hash=first_envelope.strategy_spec_instance_hash,
        raw_input_bundle_hash=first_preliminary.raw_frame_bundle_hash,
        symbol="AAAUSDT",
        side=CandidateSide.SHORT,
        timeframe_seconds=interval_seconds(first_envelope.timeframe),
        arm_bar_open_ts=first_preliminary.base_bar_open_ts,
        arm_candle_cutoff_ts=first_preliminary.candle_cutoff_ts,
        armed_high=109.0,
        armed_low=107.0,
        armed_close=108.0,
        invalidate_level=111.0,
        confirmation_enabled=True,
        confirmation_max_wait_observations=3,
        arm_trace={"failed_layer": "layer_confirmation_pending"},
    )
    armed = CandidateLifecycleEventV1.armed(arm)
    first = PopulationDecisionV6.create(**{**first_common, "lifecycle_event": armed})

    second_envelope, second_common, second_preliminary = _single_cycle(
        1_700_010_000.0
    )
    observation = ConfirmationObservationV1(
        candidate_id=arm.candidate_id,
        observation_input_bundle_hash=second_preliminary.raw_frame_bundle_hash,
        state=CandidateLifecycleState.WAITING,
        state_epoch=1,
        timeframe_seconds=arm.timeframe_seconds,
        observation_bar_open_ts=second_preliminary.base_bar_open_ts,
        observation_candle_cutoff_ts=second_preliminary.candle_cutoff_ts,
        observed_high=109.0,
        observed_low=107.0,
        observed_close=108.0,
        distinct_observation_count=1,
        elapsed_bars=1,
    )
    waiting = CandidateLifecycleEventV1.transition(
        armed, confirmation=observation
    )
    second = PopulationDecisionV6.create(
        **{**second_common, "lifecycle_event": waiting}
    )
    path = tmp_path / "chain.jsonl"
    journal = PopulationJournal(path)
    journal.append_cycle([first], envelope=first_envelope)
    journal.append_cycle([second], envelope=second_envelope)
    assert len(list(iter_population_cycles(path))) == 2

    forged = replace(waiting, previous_event_id="f" * 64)
    forged_second = PopulationDecisionV6.create(
        **{**second_common, "lifecycle_event": forged}
    )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[4] = forged_second.as_dict()
    _rehash_cycle(lines, 3)
    path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) for row in lines
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        PopulationDatasetError, match="invalid_cross_cycle_lifecycle_chain"
    ):
        verify_population_journal(path)
    with pytest.raises(
        PopulationJournalError, match="witnessed predecessor"
    ):
        PopulationJournal(path)


def test_contains_cycle_refreshes_after_an_external_append(tmp_path) -> None:
    path = tmp_path / "population_v6.jsonl"
    stale = PopulationJournal(path)
    writer = PopulationJournal(path)
    envelope = _envelope()
    assert stale.contains_cycle(envelope.cycle_id) is False
    writer.append_cycle(_records(), envelope=envelope)
    assert stale.contains_cycle(envelope.cycle_id) is True


def test_runtime_session_rejects_a_second_scanner_owner(tmp_path) -> None:
    path = tmp_path / "runtime.jsonl"
    first = PopulationJournal(path)
    second = PopulationJournal(path)
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def own_runtime() -> None:
        try:
            with first.runtime_session():
                entered.set()
                if not release.wait(timeout=2.0):
                    raise AssertionError("runtime owner was not released")
        except BaseException as exc:  # surfaced in the parent assertion below
            failures.append(exc)

    owner = threading.Thread(target=own_runtime)
    owner.start()
    assert entered.wait(timeout=2.0)
    try:
        with pytest.raises(PopulationJournalError, match="runtime is already active"):
            with second.runtime_session():
                raise AssertionError("second scanner unexpectedly acquired runtime")
    finally:
        release.set()
        owner.join(timeout=2.0)
    assert not owner.is_alive()
    assert failures == []


def test_runtime_session_releases_lock_and_preserves_body_error(tmp_path) -> None:
    journal = PopulationJournal(tmp_path / "runtime-error.jsonl")
    body_error = "timed out waiting inside scanner body"
    with pytest.raises(PopulationJournalError, match=body_error):
        with journal.runtime_session():
            raise PopulationJournalError(body_error)
    with journal.runtime_session():
        pass


def test_v6_constructor_rejects_forged_ids_and_unknown_actions() -> None:
    record = _records()[0]
    with pytest.raises(PopulationJournalError, match="input hash mismatch"):
        replace(record, input_hash="0" * 64)
    with pytest.raises(PopulationJournalError, match="snapshot ID mismatch"):
        replace(record, snapshot_id="0" * 64)
    with pytest.raises(PopulationJournalError, match="action is unsupported"):
        _recreate(record, action="EXECUTE_ANYTHING")


def test_writer_binds_venue_symbol_receipt_time_and_cycle_owned_fields(tmp_path) -> None:
    record = _records()[0]
    base = record.base_source_evidence
    assert base is not None

    wrong_venue = _recreate(
        record,
        base_source_evidence=replace(base, venue_symbol="EVIL_USDT"),
    )
    with pytest.raises(PopulationJournalError, match="venue symbol"):
        PopulationJournal(tmp_path / "venue.jsonl").append_cycle(
            [wrong_venue, _records()[1]], envelope=_envelope()
        )

    late_base = replace(
        base,
        request_started_at=record.decision_ts + 0.25,
        source_ts=record.decision_ts + 0.4,
        received_at=record.decision_ts + 0.5,
    )
    late = _recreate(record, base_source_evidence=late_base)
    with pytest.raises(PopulationJournalError, match="arrives after decision"):
        PopulationJournal(tmp_path / "late.jsonl").append_cycle(
            [late, _records()[1]], envelope=_envelope()
        )

    drifted_cycle_fact = _recreate(
        record,
        universe_received_at=record.universe_received_at + 0.25,
    )
    with pytest.raises(PopulationJournalError, match="universe_received_at"):
        PopulationJournal(tmp_path / "cycle_fact.jsonl").append_cycle(
            [drifted_cycle_fact, _records()[1]], envelope=_envelope()
        )


def test_writer_requires_exact_source_timing_projection_and_terminal_policy(tmp_path) -> None:
    records = _records()
    first = records[0]
    base = first.base_source_evidence
    assert base is not None
    timing_drift = _recreate(
        first,
        base_source_evidence=replace(
            base,
            request_started_at=base.request_started_at + 0.1,
            source_ts=base.source_ts + 0.1,
            received_at=base.received_at + 0.1,
        ),
    )
    with pytest.raises(PopulationJournalError, match="base_ohlcv timing differs"):
        PopulationJournal(tmp_path / "timing.jsonl").append_cycle(
            [timing_drift, records[1]], envelope=_envelope()
        )

    terminal = _envelope(universe_symbols=(), status="empty_universe")
    with pytest.raises(PopulationJournalError, match="must not request its benchmark"):
        PopulationJournal(tmp_path / "terminal.jsonl").append_cycle(
            (),
            envelope=terminal,
            benchmark_source_evidence=_benchmark_evidence(),
        )


def test_writer_rejects_an_untyped_extra_source_timing(tmp_path) -> None:
    envelope = _envelope()
    payload = envelope.as_dict()
    payload["source_timings"].append(
        SourceTiming(
            source="fabricated_feed",
            request_started_at=envelope.candle_cutoff_ts + 1.0,
            received_at=envelope.candle_cutoff_ts + 1.25,
        ).as_dict()
    )
    drifted = CycleEnvelope.from_dict(payload)
    records = []
    for record in _records():
        metadata = record.as_dict()["metadata"]
        metadata["feature_provenance"]["envelope_hash"] = drifted.envelope_hash()
        records.append(_recreate(record, metadata=metadata))
    with pytest.raises(PopulationJournalError, match="unsupported source timing"):
        PopulationJournal(tmp_path / "extra_source.jsonl").append_cycle(
            records,
            envelope=drifted,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("receipt_schema_version", True, "receipt schema"),
        ("receipt_schema_version", 1.0, "receipt schema"),
        ("journal_schema_version", 6.0, "journal schema"),
        ("sequence_no", True, "sequence_no"),
        ("prefix_length_bytes", 1.0, "prefix length"),
    ],
)
def test_checkpoint_receipt_rejects_noncanonical_integer_types(
    tmp_path, field, value, message
) -> None:
    receipt = _write_v6(tmp_path / "receipt.jsonl").checkpoint_receipt().as_dict()
    receipt[field] = value
    with pytest.raises(PopulationJournalError, match=message):
        JournalCheckpointReceipt.from_dict(receipt)


def test_persisted_evidence_and_envelope_reject_numeric_type_drift() -> None:
    evidence = _benchmark_evidence().as_dict()
    evidence["requested_as_of_ts"] = int(evidence["requested_as_of_ts"])
    with pytest.raises(Exception, match="not_canonical|invalid_source_read_evidence"):
        SourceReadEvidenceV1.from_dict(evidence)

    envelope = _envelope().as_dict()
    envelope["cycle_started_at"] = int(envelope["cycle_started_at"])
    with pytest.raises(CycleEnvelopeError, match="source_mismatch"):
        CycleEnvelope.from_dict(envelope)
