from __future__ import annotations

import json
import hashlib

import pytest

from core.mexc_strategy_spec import (
    MEXC_STRATEGY_SPEC_VERSION,
    MexcStrategySpec,
    load_mexc_strategy_spec,
    strategy_spec_contract_hash,
)
from trading.market_data.bar_contract import closed_boundary_ts, interval_seconds
from trading.market_data.frame_provenance import (
    SourceReadEvidenceV1,
    aggregate_source_timing_from_evidence,
    source_timing_from_evidence,
)
from trading.market_data.source_timing import SourceTiming
from trading.metrics.cycle_envelope import CycleEnvelope
from trading.metrics.population_journal import (
    PopulationDecisionV6,
    PopulationJournal,
    PopulationJournalError,
    make_cycle_id,
    safe_error_code,
)


_UNIVERSE_TIMING = SourceTiming(
    source="universe",
    request_started_at=1_699_999_999.5,
    received_at=1_700_000_000.0,
)
_STRATEGY_SPEC_PAYLOAD = load_mexc_strategy_spec().to_mapping()
_STRATEGY_SPEC_PAYLOAD["market_data"]["base_interval"] = "Min5"
_STRATEGY_SPEC = MexcStrategySpec.from_mapping(_STRATEGY_SPEC_PAYLOAD)


def _envelope(**overrides) -> CycleEnvelope:
    values = dict(
        cycle_id=_cycle_id(),
        timeframe="Min5",
        cycle_started_at=1_700_000_101.0,
        candle_cutoff_ts=1_700_000_100.0,
        universe_symbols=("AAA_USDT", "BBB_USDT"),
        universe_timing=_UNIVERSE_TIMING,
        source_timings=(),
        strategy_spec_version=MEXC_STRATEGY_SPEC_VERSION,
        strategy_spec_contract_hash=strategy_spec_contract_hash(),
        strategy_spec_instance_hash=_STRATEGY_SPEC.instance_hash,
        strategy_spec_payload=_STRATEGY_SPEC.to_mapping(),
        universe_policy_hash="b" * 64,
        ranking_ready_ts=1_700_000_103.0,
        cycle_completed_ts=1_700_000_104.0,
    )
    values.update(overrides)
    if "source_timings" not in overrides and values.get("status", "completed") != "completed":
        values["source_timings"] = (values["universe_timing"],)
    elif "source_timings" not in overrides:
        cutoff = float(values["candle_cutoff_ts"])
        timeframe = str(values["timeframe"])
        reads = [
            _source_evidence(
                symbol=symbol,
                timeframe=timeframe,
                cutoff=cutoff,
                status="evaluated",
            )
            for symbol in values["universe_symbols"]
        ]
        benchmark_timing = source_timing_from_evidence(
            reads[0][2],
            source="benchmark",
        )
        base_timing = aggregate_source_timing_from_evidence(
            [base for base, _, _ in reads],
            source="base_ohlcv",
        )
        htf_timing = aggregate_source_timing_from_evidence(
            [htf for _, htf, _ in reads],
            source="higher_timeframe",
        )
        values["source_timings"] = tuple(
            timing
            for timing in (
                values["universe_timing"],
                benchmark_timing,
                base_timing,
                htf_timing,
            )
            if timing is not None
        )
    return CycleEnvelope.build(**values)


def _cycle_id() -> str:
    return make_cycle_id(
        timeframe="Min5",
        candle_cutoff_ts=1_700_000_100.0,
        universe_received_at=1_700_000_000.0,
        universe_symbols=["AAA_USDT", "BBB_USDT"],
    )


def _source_evidence(
    *, symbol: str, timeframe: str, cutoff: float, status: str
) -> tuple[SourceReadEvidenceV1, SourceReadEvidenceV1, SourceReadEvidenceV1]:
    venue_symbol = symbol if "_" in symbol else symbol[:-4] + "_USDT"
    base_seconds = interval_seconds(timeframe)
    common = dict(
        venue="mexc_contract",
        symbol=symbol,
        venue_symbol=venue_symbol,
        timeframe=timeframe,
        requested_as_of_ts=cutoff,
        expected_closed_boundary_ts=float(closed_boundary_ts(cutoff, timeframe)),
        request_started_at=cutoff + 1.0,
        received_at=cutoff + 1.5,
        source_ts=cutoff + 1.4,
        cache_hit=False,
        cache_age_sec=0.0,
    )
    if status == "no_data":
        base = SourceReadEvidenceV1(
            source="base_ohlcv",
            outcome="no_rows",
            error_code=None,
            missing_reason="no_rows",
            first_bar_open_ts=None,
            last_bar_open_ts=None,
            last_bar_close_ts=None,
            data_through_ts=None,
            bar_count=0,
            frame_hash=None,
            **common,
        )
    else:
        base = SourceReadEvidenceV1(
            source="base_ohlcv",
            outcome="fresh",
            error_code=None,
            missing_reason=None,
            first_bar_open_ts=cutoff - 20 * base_seconds,
            last_bar_open_ts=cutoff - base_seconds,
            last_bar_close_ts=cutoff,
            data_through_ts=cutoff,
            bar_count=20,
            frame_hash=hashlib.sha256(symbol.encode()).hexdigest(),
            **common,
        )
    htf_name = _STRATEGY_SPEC.market_data.higher_timeframe.interval
    htf = SourceReadEvidenceV1.not_requested(
        source="higher_timeframe_ohlcv",
        venue="mexc_contract",
        symbol=symbol,
        venue_symbol=venue_symbol,
        timeframe=htf_name,
        requested_as_of_ts=cutoff,
        reason="not_used",
    )
    benchmark_timeframe = _STRATEGY_SPEC.resolved_benchmark_interval
    benchmark_seconds = interval_seconds(benchmark_timeframe)
    benchmark = SourceReadEvidenceV1(
        source="benchmark_ohlcv",
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        timeframe=benchmark_timeframe,
        requested_as_of_ts=cutoff,
        expected_closed_boundary_ts=float(
            closed_boundary_ts(cutoff, benchmark_timeframe)
        ),
        request_started_at=cutoff + 1.0,
        received_at=cutoff + 1.5,
        source_ts=cutoff + 1.4,
        cache_hit=False,
        cache_age_sec=0.0,
        outcome="fresh",
        error_code=None,
        missing_reason=None,
        first_bar_open_ts=cutoff - 20 * benchmark_seconds,
        last_bar_open_ts=cutoff - benchmark_seconds,
        last_bar_close_ts=cutoff,
        data_through_ts=cutoff,
        bar_count=20,
        frame_hash="b" * 64,
    )
    return base, htf, benchmark


def _record(**overrides: object) -> PopulationDecisionV6:
    values: dict[str, object] = {
        "cycle_id": _cycle_id(),
        "universe_refreshed_at": 1_699_999_999.0,
        "universe_request_started_at": 1_699_999_999.5,
        "universe_received_at": 1_700_000_000.0,
        "scan_observed_at": 1_700_000_101.0,
        "candle_cutoff_ts": 1_700_000_100.0,
        "decision_ts": 1_700_000_102.0,
        "ranking_ready_ts": 1_700_000_103.0,
        "cycle_completed_ts": 1_700_000_104.0,
        "actionable_ts": 1_700_000_103.0,
        "entry_eligible_ts": 1_700_000_104.0,
        # first Min5 boundary strictly after entry_eligible_ts
        "entry_bar_open_ts": 1_700_000_400.0,
        "symbol": "AAA_USDT",
        "timeframe": "Min5",
        "status": "evaluated",
        "base_bar_open_ts": 1_699_999_800.0,
        "base_bar_close_ts": 1_700_000_100.0,
        "action": "HOLD",
        "reason": "no_setup",
        "confidence": 0.25,
        "metadata": {"features": {"z": 2.0, "a": 1.0}},
    }
    values.update(overrides)
    metadata = dict(values["metadata"])
    metadata.setdefault(
        "provenance",
        {
            "strategy_config_hash": _STRATEGY_SPEC.instance_hash,
            "universe_policy_hash": "b" * 64,
        },
    )
    values["metadata"] = metadata
    base, htf, benchmark = _source_evidence(
        symbol=str(values["symbol"]),
        timeframe=str(values["timeframe"]),
        cutoff=float(values["candle_cutoff_ts"]),
        status=str(values["status"]),
    )
    values.update(
        base_source_evidence=base,
        higher_timeframe_source_evidence=htf,
        benchmark_source_evidence=benchmark,
    )
    return PopulationDecisionV6.create(**values)  # type: ignore[arg-type]


def test_ids_are_stable_for_different_mapping_key_order() -> None:
    first = _record(metadata={"features": {"z": 2.0, "a": 1.0}, "regime": "risk_on"})
    second = _record(metadata={"regime": "risk_on", "features": {"a": 1.0, "z": 2.0}})

    assert first.input_hash == second.input_hash
    assert first.snapshot_id == second.snapshot_id


def test_causal_change_changes_input_and_snapshot_ids() -> None:
    hold = _record(action="HOLD")
    entry = _record(action="SHORT_ENTRY")

    assert hold.input_hash != entry.input_hash
    assert hold.snapshot_id != entry.snapshot_id


def test_wall_clock_decision_time_does_not_change_causal_ids() -> None:
    """Neither the per-symbol decision clock nor the cycle's own timing may enter
    causal identity: the same market inputs must hash the same however slowly the
    scan happened to run."""
    first = _record(decision_ts=1_700_000_102.0)
    later = _record(
        decision_ts=1_700_000_999.0,
        scan_observed_at=1_700_000_998.0,
        ranking_ready_ts=1_700_001_000.0,
        cycle_completed_ts=1_700_001_001.0,
        actionable_ts=1_700_001_000.0,
        entry_eligible_ts=1_700_001_001.0,
        entry_bar_open_ts=1_700_001_300.0,
    )

    assert first.input_hash == later.input_hash
    assert first.snapshot_id == later.snapshot_id


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), object()])
def test_unsafe_metadata_is_rejected(bad_value: object) -> None:
    with pytest.raises(PopulationJournalError):
        _record(metadata={"bad": bad_value})


def test_non_string_metadata_key_is_rejected_cleanly() -> None:
    with pytest.raises(PopulationJournalError, match="invalid key"):
        _record(metadata={1: "bad"})


def test_exception_message_is_never_serialized(tmp_path) -> None:
    secret_marker = "DO_NOT_PERSIST_THIS_EXCEPTION_TEXT"
    try:
        raise RuntimeError(secret_marker)
    except RuntimeError as exc:
        record = _record(
            status="strategy_error",
            action="HOLD",
            reason="strategy_error",
            error_code=safe_error_code(exc),
            metadata={"stage": "strategy"},
        )

    path = tmp_path / "population.jsonl"
    records = [
        record.__class__(**{**record.__dict__, "cycle_ordinal": 0, "cycle_size": 2}),
        _record(symbol="BBB_USDT", cycle_ordinal=1, cycle_size=2),
    ]
    PopulationJournal(path).append_cycle(records, envelope=_envelope())
    contents = path.read_text(encoding="utf-8")

    assert secret_marker not in contents
    assert "RuntimeError" in contents


def test_exception_text_metadata_keys_are_rejected() -> None:
    with pytest.raises(PopulationJournalError, match="exception text"):
        _record(metadata={"exception_message": "sensitive details"})


def test_append_cycle_writes_one_canonical_json_row_per_record(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    records = [
        _record(symbol="AAA_USDT", cycle_ordinal=0, cycle_size=2),
        _record(symbol="BBB_USDT", cycle_ordinal=1, cycle_size=2),
    ]

    assert PopulationJournal(path).append_cycle(records, envelope=_envelope()) is True

    lines = path.read_text(encoding="utf-8").splitlines()
    decoded = [json.loads(line) for line in lines]
    # One header, the ordered rows, one footer. The envelope appears once.
    assert [row["record_type"] for row in decoded] == [
        "cycle_header", "decision", "decision", "cycle_footer"
    ]
    assert sum("universe_symbols" in json.dumps(row) for row in decoded) == 1
    rows = [row for row in decoded if row["record_type"] == "decision"]
    assert [row["symbol"] for row in rows] == ["AAA_USDT", "BBB_USDT"]
    assert decoded[-1]["row_count"] == 2
    assert lines[1] == json.dumps(rows[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_disabled_journal_does_not_create_a_file(tmp_path) -> None:
    path = tmp_path / "population.jsonl"

    assert PopulationJournal(path, enabled=False).append_cycle([_record()], envelope=_envelope()) is False

    assert not path.exists()


def test_cycle_id_depends_on_ordered_point_in_time_universe() -> None:
    first = make_cycle_id(
        timeframe="Min5",
        candle_cutoff_ts=1_700_000_100.0,
        universe_received_at=1_700_000_000.0,
        universe_symbols=["AAA_USDT", "BBB_USDT"],
    )
    second = make_cycle_id(
        timeframe="Min5",
        candle_cutoff_ts=1_700_000_100.0,
        universe_received_at=1_700_000_000.0,
        universe_symbols=["BBB_USDT", "AAA_USDT"],
    )

    assert first != second


def test_equivalent_timeframe_names_have_the_same_causal_ids() -> None:
    generic_cycle = make_cycle_id(
        timeframe="60",
        candle_cutoff_ts=1_700_002_800.0,
        universe_received_at=1_700_000_000.0,
        universe_symbols=["AAA_USDT"],
    )
    mexc_cycle = make_cycle_id(
        timeframe="Min60",
        candle_cutoff_ts=1_700_002_800.0,
        universe_received_at=1_700_000_000.0,
        universe_symbols=["AAA_USDT"],
    )
    generic = _record(
        cycle_id=generic_cycle,
        timeframe="60",
        candle_cutoff_ts=1_700_002_800.0,
        scan_observed_at=1_700_002_801.0,
        decision_ts=1_700_002_802.0,
        ranking_ready_ts=1_700_002_803.0,
        cycle_completed_ts=1_700_002_804.0,
        actionable_ts=1_700_002_803.0,
        entry_eligible_ts=1_700_002_804.0,
        entry_bar_open_ts=1_700_006_400.0,
        base_bar_open_ts=1_699_999_200.0,
        base_bar_close_ts=1_700_002_800.0,
    )
    mexc = _record(
        cycle_id=mexc_cycle,
        timeframe="Min60",
        candle_cutoff_ts=1_700_002_800.0,
        scan_observed_at=1_700_002_801.0,
        decision_ts=1_700_002_802.0,
        ranking_ready_ts=1_700_002_803.0,
        cycle_completed_ts=1_700_002_804.0,
        actionable_ts=1_700_002_803.0,
        entry_eligible_ts=1_700_002_804.0,
        entry_bar_open_ts=1_700_006_400.0,
        base_bar_open_ts=1_699_999_200.0,
        base_bar_close_ts=1_700_002_800.0,
    )

    assert generic_cycle == mexc_cycle
    assert generic.input_hash == mexc.input_hash
    assert generic.snapshot_id == mexc.snapshot_id


def test_no_data_uses_absent_bar_timestamps_instead_of_a_sentinel() -> None:
    record = _record(
        status="no_data",
        base_bar_open_ts=None,
        base_bar_close_ts=None,
        action="HOLD",
        reason="no_data",
        confidence=0.0,
    )

    assert record.base_bar_open_ts is None
    assert record.base_bar_close_ts is None


def test_feature_bearing_base_bar_must_close_at_the_causal_cutoff() -> None:
    with pytest.raises(PopulationJournalError, match="does not close at the causal cutoff"):
        _record(
            base_bar_open_ts=1_699_999_500.0,
            base_bar_close_ts=1_699_999_800.0,
        )


def test_base_bar_must_be_aligned_to_the_timeframe() -> None:
    with pytest.raises(PopulationJournalError, match="not aligned"):
        _record(
            base_bar_open_ts=1_699_999_800.5,
            base_bar_close_ts=1_700_000_100.0,
        )


def test_append_cycle_deduplicates_the_latest_complete_cycle(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    records = [
        _record(symbol="AAA_USDT", cycle_ordinal=0, cycle_size=2),
        _record(symbol="BBB_USDT", cycle_ordinal=1, cycle_size=2),
    ]
    journal = PopulationJournal(path)

    assert journal.append_cycle(records, envelope=_envelope()) is True
    assert journal.append_cycle(records, envelope=_envelope()) is False
    assert PopulationJournal(path).append_cycle(records, envelope=_envelope()) is False
    # header + two decision rows + footer, written exactly once
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4


def test_append_cycle_requires_complete_ordered_batch(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    incomplete = _record(cycle_ordinal=0, cycle_size=2)

    with pytest.raises(PopulationJournalError, match="cycle size"):
        PopulationJournal(path).append_cycle([incomplete], envelope=_envelope())
