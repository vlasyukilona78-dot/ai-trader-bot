"""Slice B: the cycle envelope is one durable record, not a per-row copy.

Repeating the envelope on every decision row made the journal quadratic in
universe size and pushed the ordered symbol list past the bounds that keep
arbitrary per-row metadata safe. It is now a header/rows/footer cycle, and the
reader rebuilds and cross-checks it instead of trusting what the rows claim.
"""

from __future__ import annotations

import json

import pytest

from ai.reversal.population_dataset import (
    PopulationDatasetError,
    iter_population_cycles,
    iter_population_feature_rows,
)
from trading.metrics.population_journal import (
    SCHEMA_VERSION,
    PopulationJournal,
    PopulationJournalError,
)

from v2.test_population_feature_dataset_v2 import (
    _envelope,
    _lines,
    _records,
    _rewrite,
    _write,
)


def test_a_three_hundred_symbol_cycle_writes_the_universe_once(tmp_path) -> None:
    """The old layout raised past 256 symbols and copied the universe per row."""
    symbols = [f"S{index:04d}USDT" for index in range(300)]
    envelope = _envelope(universe_symbols=tuple(symbols))

    path = tmp_path / "population.jsonl"
    # No decision rows are needed to prove the envelope itself is writable and
    # bounded; the per-row path is covered by the other cycles in this file.
    assert PopulationJournal(path).append_cycle((), envelope=envelope) is True

    lines = _lines(path)
    assert [row["record_type"] for row in lines] == ["cycle_header", "cycle_footer"]
    assert len(lines[0]["envelope"]["universe_symbols"]) == 300
    assert sum("universe_symbols" in json.dumps(row) for row in lines) == 1


def test_an_empty_cycle_survives_a_restart(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    empty = _envelope(universe_symbols=(), status="empty_universe")
    assert PopulationJournal(path).append_cycle((), envelope=empty) is True

    # A fresh process must be able to read the evidence back.
    cycles = list(iter_population_cycles(path))
    assert len(cycles) == 1
    envelope, rows = cycles[0]
    assert envelope.status == "empty_universe"
    assert rows == []


def test_an_error_cycle_survives_a_restart(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    failed = _envelope(status="error", error_code="TimeoutError")
    assert PopulationJournal(path).append_cycle((), envelope=failed) is True

    envelope, rows = next(iter(iter_population_cycles(path)))
    assert envelope.status == "error"
    assert envelope.error_code == "TimeoutError"
    assert rows == []


def test_appending_to_a_previous_schema_is_refused_before_writing(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    rows[0]["schema_version"] = SCHEMA_VERSION - 1
    _rewrite(path, rows)
    size_before = path.stat().st_size

    with pytest.raises(PopulationJournalError, match="written by schema"):
        PopulationJournal(path)

    assert path.stat().st_size == size_before


def test_a_truncated_tail_is_not_concatenated_into_one_line(tmp_path) -> None:
    """A process that died mid-write leaves a line without its newline. Appending
    would glue two JSON objects into one unparseable line."""
    path = tmp_path / "population.jsonl"
    _write(path)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    path.write_text(text.rstrip("\n"), encoding="utf-8")

    with pytest.raises(PopulationJournalError, match="truncated mid-write"):
        PopulationJournal(path)

    assert "}{" not in path.read_text(encoding="utf-8")


def test_a_batch_may_not_mix_schema_versions(tmp_path) -> None:
    records = _records()
    stale = records[0].__class__(
        **{**records[0].__dict__, "schema_version": SCHEMA_VERSION - 1}
    )
    with pytest.raises(PopulationJournalError, match="mixes journal schema versions"):
        PopulationJournal(tmp_path / "p.jsonl").append_cycle(
            [stale, records[1]], envelope=_envelope()
        )


def test_rows_from_another_cycle_are_refused(tmp_path) -> None:
    with pytest.raises(PopulationJournalError, match="does not belong to the envelope cycle"):
        PopulationJournal(tmp_path / "p.jsonl").append_cycle(
            _records(), envelope=_envelope(cycle_id="f" * 64)
        )


def test_reader_requires_the_envelope(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    _rewrite(path, rows[1:])

    with pytest.raises(PopulationDatasetError, match="journal_row_before_its_cycle_header"):
        list(iter_population_feature_rows(path))


def test_reader_rejects_an_envelope_that_does_not_rebuild(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    # Claim the cycle was actionable earlier than its own inputs allow.
    rows[0]["envelope"]["actionable_ts"] -= 60.0
    _rewrite(path, rows)

    with pytest.raises(PopulationDatasetError, match="cycle_envelope"):
        list(iter_population_feature_rows(path))


def test_reader_rejects_a_row_that_disagrees_with_the_envelope(tmp_path) -> None:
    """Wall-clock fields are deliberately outside input_hash, so nothing but the
    envelope cross-check stands between a tampered timestamp and the dataset."""
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    # Still valid on its own terms: after its request, before the scan.
    rows[1]["universe_received_at"] = rows[1]["universe_request_started_at"] + 0.25
    _rewrite(path, rows)

    with pytest.raises(
        PopulationDatasetError,
        match="row_disagrees_with_cycle_envelope:universe_received_at",
    ):
        list(iter_population_feature_rows(path))


def test_reader_rejects_a_truncated_body(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    # Remove one decision row but keep the footer's promise of two.
    _rewrite(path, [rows[0], rows[1], rows[3]])

    with pytest.raises(PopulationDatasetError, match="cycle_row_count_mismatch|incomplete_or_unordered_cycle"):
        list(iter_population_feature_rows(path))


def test_reader_rejects_a_reordered_body(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    _rewrite(path, [rows[0], rows[2], rows[1], rows[3]])

    with pytest.raises(PopulationDatasetError, match="incomplete_or_unordered_cycle"):
        list(iter_population_feature_rows(path))


def test_reader_rejects_a_header_whose_id_does_not_match_its_envelope(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    rows[0]["cycle_id"] = "e" * 64
    _rewrite(path, rows)

    with pytest.raises(PopulationDatasetError, match="cycle_header_id_mismatch"):
        list(iter_population_feature_rows(path))
