from __future__ import annotations

import json

import pytest

from trading.metrics.population_journal import (
    PopulationDecision,
    PopulationJournal,
    PopulationJournalError,
    make_cycle_id,
    safe_error_code,
)


def _cycle_id() -> str:
    return make_cycle_id(
        timeframe="Min5",
        candle_cutoff_ts=1_700_000_100.0,
        universe_refreshed_at=1_700_000_000.0,
        universe_symbols=["AAA_USDT", "BBB_USDT"],
    )


def _record(**overrides: object) -> PopulationDecision:
    values: dict[str, object] = {
        "cycle_id": _cycle_id(),
        "universe_refreshed_at": 1_700_000_000.0,
        "scan_observed_at": 1_700_000_101.0,
        "candle_cutoff_ts": 1_700_000_100.0,
        "decision_ts": 1_700_000_102.0,
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
    return PopulationDecision.create(**values)  # type: ignore[arg-type]


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
    first = _record(decision_ts=1_700_000_102.0)
    later = _record(decision_ts=1_700_000_999.0, scan_observed_at=1_700_000_998.0)

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
    PopulationJournal(path).append_cycle([record])
    contents = path.read_text(encoding="utf-8")

    assert secret_marker not in contents
    assert "RuntimeError" in contents


def test_exception_text_metadata_keys_are_rejected() -> None:
    with pytest.raises(PopulationJournalError, match="exception text"):
        _record(metadata={"exception_message": "sensitive details"})


def test_append_cycle_writes_one_canonical_json_row_per_record(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    records = [_record(symbol="AAA_USDT"), _record(symbol="BBB_USDT")]

    PopulationJournal(path).append_cycle(records)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    decoded = [json.loads(line) for line in lines]
    assert [row["symbol"] for row in decoded] == ["AAA_USDT", "BBB_USDT"]
    assert lines[0] == json.dumps(decoded[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_disabled_journal_does_not_create_a_file(tmp_path) -> None:
    path = tmp_path / "population.jsonl"

    PopulationJournal(path, enabled=False).append_cycle([_record()])

    assert not path.exists()


def test_cycle_id_depends_on_ordered_point_in_time_universe() -> None:
    first = make_cycle_id(
        timeframe="Min5",
        candle_cutoff_ts=1_700_000_100.0,
        universe_refreshed_at=1_700_000_000.0,
        universe_symbols=["AAA_USDT", "BBB_USDT"],
    )
    second = make_cycle_id(
        timeframe="Min5",
        candle_cutoff_ts=1_700_000_100.0,
        universe_refreshed_at=1_700_000_000.0,
        universe_symbols=["BBB_USDT", "AAA_USDT"],
    )

    assert first != second
