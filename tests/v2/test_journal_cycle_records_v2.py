"""Slice B: the cycle envelope is one durable record, not a per-row copy.

Repeating the envelope on every decision row made the journal quadratic in
universe size and pushed the ordered symbol list past the bounds that keep
arbitrary per-row metadata safe. It is now a header/rows/footer cycle, and the
reader rebuilds and cross-checks it instead of trusting what the rows claim.
"""

from __future__ import annotations

import json
import multiprocessing

import pytest

from ai.reversal.feature_contract import market_feature_hash
from ai.reversal.population_dataset import (
    PopulationDatasetError,
    iter_population_cycles,
    iter_population_feature_rows,
)
from trading.metrics.population_journal import (
    SCHEMA_VERSION,
    PopulationJournal,
    PopulationJournalError,
    compute_cycle_commit,
    genesis_cycle_commit,
    rows_checksum,
)

from v2.test_population_feature_dataset_v2 import (
    _envelope,
    _lines,
    _records,
    _rewrite,
    _write,
)


def _append_empty_cycle_after_release(path, ready, release, results) -> None:
    """Spawn-safe worker used to exercise the OS-level journal lock."""

    try:
        envelope = _envelope(universe_symbols=(), status="empty_universe")
        journal = PopulationJournal(path)
        ready.put(True)
        if not release.wait(timeout=20.0):
            raise TimeoutError("test release event was not signalled")
        results.put(("ok", journal.append_cycle((), envelope=envelope)))
    except BaseException as exc:  # pragma: no cover - returned to the parent
        results.put(("error", type(exc).__name__, str(exc)))


def test_a_three_hundred_symbol_error_envelope_writes_the_universe_once(tmp_path) -> None:
    """The old layout raised past 256 symbols and copied the universe per row."""
    symbols = [f"S{index:04d}USDT" for index in range(300)]
    envelope = _envelope(
        universe_symbols=tuple(symbols), status="error", error_code="ScannerUnavailable"
    )

    path = tmp_path / "population.jsonl"
    # A typed pre-evaluation failure has no decision rows, but still proves the
    # large envelope is written once and remains bounded.
    assert PopulationJournal(path).append_cycle((), envelope=envelope) is True

    lines = _lines(path)
    assert [row["record_type"] for row in lines] == ["cycle_header", "cycle_footer"]
    assert len(lines[0]["envelope"]["universe_symbols"]) == 300
    assert sum("universe_symbols" in json.dumps(row) for row in lines) == 1


def test_writer_rejects_a_completed_cycle_with_no_rows(tmp_path) -> None:
    with pytest.raises(PopulationJournalError, match="completed cycle"):
        PopulationJournal(tmp_path / "population.jsonl").append_cycle(
            (), envelope=_envelope()
        )


def test_reader_rejects_a_serialized_completed_cycle_with_no_rows(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    envelope = _envelope()
    envelope_hash = envelope.envelope_hash()
    journal_id = "c" * 64
    prev_cycle_commit = genesis_cycle_commit(journal_id)
    header = {
        "record_type": "cycle_header",
        "schema_version": SCHEMA_VERSION,
        "journal_id": journal_id,
        "sequence_no": 0,
        "prev_cycle_commit": prev_cycle_commit,
        "cycle_id": envelope.cycle_id,
        "row_count": 0,
        "envelope_hash": envelope_hash,
        "envelope": envelope.as_dict(),
    }
    footer_core = {
        "record_type": "cycle_footer",
        "schema_version": SCHEMA_VERSION,
        "journal_id": journal_id,
        "sequence_no": 0,
        "prev_cycle_commit": prev_cycle_commit,
        "cycle_id": envelope.cycle_id,
        "row_count": 0,
        "envelope_hash": envelope_hash,
        "rows_checksum": rows_checksum(()),
    }
    _rewrite(
        path,
        [
            header,
            {
                **footer_core,
                "cycle_commit": compute_cycle_commit(header, (), footer_core),
            },
        ],
    )

    with pytest.raises(PopulationDatasetError, match="completed_cycle_has_no_decision_rows"):
        list(iter_population_cycles(path))


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


def test_restart_rejects_a_mixed_schema_in_the_middle(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    rows[1]["schema_version"] = SCHEMA_VERSION - 1
    _rewrite(path, rows)

    with pytest.raises(PopulationJournalError, match="written by schema"):
        PopulationJournal(path)


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
            _records(), envelope=_envelope(universe_symbols=("BBBUSDT", "AAAUSDT"))
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


def test_reader_rejects_an_altered_ordered_envelope_universe(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    rows[0]["envelope"]["universe_symbols"].reverse()
    rows[0]["envelope_hash"] = _envelope(
        universe_symbols=("BBBUSDT", "AAAUSDT")
    ).envelope_hash()
    _rewrite(path, rows)

    with pytest.raises(PopulationDatasetError, match="population_cycle_id_mismatch"):
        list(iter_population_cycles(path))


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
        match="feature_provenance_universe_received_at_mismatch",
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


def test_header_footer_and_body_counts_must_all_agree(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    rows[-1]["row_count"] = rows[0]["row_count"] - 1
    _rewrite(path, rows)

    with pytest.raises(PopulationDatasetError, match="cycle_row_count_mismatch"):
        list(iter_population_feature_rows(path))


def test_completed_population_must_equal_envelope_universe_in_order(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    records = list(reversed(_records()))
    records = [
        record.__class__(
            **{
                **record.__dict__,
                "cycle_ordinal": ordinal,
            }
        )
        for ordinal, record in enumerate(records)
    ]

    with pytest.raises(PopulationJournalError, match="universe order"):
        PopulationJournal(path).append_cycle(records, envelope=_envelope())


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("envelope_hash", "envelope hash mismatch"),
        ("market_feature_hash", "market feature hash mismatch"),
    ],
)
def test_writer_rejects_unbound_feature_provenance(tmp_path, field, message) -> None:
    records = _records()
    payload = records[0].as_dict()
    payload["metadata"]["feature_provenance"][field] = "0" * 64
    bad = records[0].__class__(
        **{
            key: value
            for key, value in payload.items()
            if key not in {"record_type", "timeframe_seconds"}
        }
    )

    with pytest.raises(PopulationJournalError, match=message):
        PopulationJournal(tmp_path / "population.jsonl").append_cycle(
            [bad, records[1]], envelope=_envelope()
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("universe_received_at", 1_700_000_002.0, "universe response mismatch"),
        ("universe_source_ts", 1_699_999_991.0, "universe source mismatch"),
        ("universe_cache_hit", False, "universe cache mismatch"),
        ("unexpected", "value", "provenance schema mismatch"),
    ],
)
def test_writer_rejects_drifted_feature_provenance(
    tmp_path, field, value, message
) -> None:
    records = _records()
    payload = records[0].as_dict()
    payload["metadata"]["feature_provenance"][field] = value
    bad = records[0].__class__(
        **{
            key: item
            for key, item in payload.items()
            if key not in {"record_type", "timeframe_seconds"}
        }
    )

    with pytest.raises(PopulationJournalError, match=message):
        PopulationJournal(tmp_path / "population.jsonl").append_cycle(
            [bad, records[1]], envelope=_envelope()
        )


def test_writer_rejects_self_consistent_snapshot_substitution(tmp_path) -> None:
    """Changing both the snapshot and its hash must not hide unchanged sources."""

    records = _records()
    payload = records[0].as_dict()
    snapshot = payload["metadata"]["feature_snapshot"]
    snapshot["values"]["funding_rate"] = 999.0
    payload["metadata"]["feature_provenance"]["market_feature_hash"] = (
        market_feature_hash(snapshot, symbol=payload["symbol"], timeframe_seconds=3600)
    )
    bad = records[0].__class__.create(
        **{
            key: value
            for key, value in payload.items()
            if key
            not in {
                "record_type",
                "timeframe_seconds",
                "input_hash",
                "snapshot_id",
            }
        }
    )

    with pytest.raises(PopulationJournalError, match="does not match its source metadata"):
        PopulationJournal(tmp_path / "population.jsonl").append_cycle(
            [bad, records[1]], envelope=_envelope()
        )


def test_reader_validates_the_whole_file_before_first_yield(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    corrupt = dict(rows[0])
    corrupt["schema_version"] = SCHEMA_VERSION - 1
    _rewrite(path, [*rows, corrupt])

    iterator = iter_population_cycles(path)
    with pytest.raises(PopulationDatasetError, match="unsupported_population_schema_version"):
        next(iterator)


def test_a_b_a_duplicate_is_suppressed_and_rejected_if_present(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    records = _records()
    first = _envelope()
    second = _envelope(universe_symbols=(), status="empty_universe")
    journal = PopulationJournal(path)
    assert journal.append_cycle(records, envelope=first) is True
    first_block = path.read_text(encoding="utf-8")
    assert journal.append_cycle((), envelope=second) is True
    assert journal.append_cycle(records, envelope=first) is False

    # A hand-concatenated or concurrently produced A-B-A file must fail closed.
    path.write_text(path.read_text(encoding="utf-8") + first_block, encoding="utf-8")
    with pytest.raises(PopulationDatasetError, match="duplicate_cycle"):
        list(iter_population_cycles(path))
    with pytest.raises(PopulationJournalError, match="duplicate cycle"):
        PopulationJournal(path)


def test_two_preopened_instances_revalidate_before_duplicate_append(tmp_path) -> None:
    """A stale in-memory ID set must not permit a second copy of one cycle."""

    path = tmp_path / "population.jsonl"
    envelope = _envelope(universe_symbols=(), status="empty_universe")
    first = PopulationJournal(path)
    second = PopulationJournal(path)

    assert first.append_cycle((), envelope=envelope) is True
    assert second.append_cycle((), envelope=envelope) is False

    cycles = list(iter_population_cycles(path))
    assert len(cycles) == 1
    assert cycles[0][0].cycle_id == envelope.cycle_id


def test_two_preopened_instances_preserve_distinct_complete_batches(tmp_path) -> None:
    """An external append is adopted before a stale instance appends its batch."""

    path = tmp_path / "population.jsonl"
    first = PopulationJournal(path)
    second = PopulationJournal(path)
    completed = _envelope()
    empty = _envelope(universe_symbols=(), status="empty_universe")

    assert first.append_cycle(_records(), envelope=completed) is True
    assert second.append_cycle((), envelope=empty) is True

    cycles = list(iter_population_cycles(path))
    assert [envelope.cycle_id for envelope, _ in cycles] == [
        completed.cycle_id,
        empty.cycle_id,
    ]


def test_stale_instance_rebuilds_rows_after_an_external_change(tmp_path) -> None:
    """Fingerprint drift triggers semantic validation, not blind ID adoption."""

    path = tmp_path / "population.jsonl"
    stale = PopulationJournal(path)
    _write(path)
    rows = _lines(path)
    rows[1]["reason"] = "tampered"
    _rewrite(path, rows)
    size_before = path.stat().st_size

    with pytest.raises(PopulationJournalError, match="does not rebuild exactly"):
        stale.append_cycle(
            (),
            envelope=_envelope(universe_symbols=(), status="empty_universe"),
        )

    assert path.stat().st_size == size_before


def test_restart_rejects_semantically_substituted_feature_snapshot(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    payload = rows[1]
    snapshot = payload["metadata"]["feature_snapshot"]
    snapshot["values"]["funding_rate"] = 999.0
    payload["metadata"]["feature_provenance"]["market_feature_hash"] = (
        market_feature_hash(snapshot, symbol=payload["symbol"], timeframe_seconds=3600)
    )
    replacement = _records()[0].__class__.create(
        **{
            key: value
            for key, value in payload.items()
            if key
            not in {
                "record_type",
                "timeframe_seconds",
                "input_hash",
                "snapshot_id",
            }
        }
    )
    rows[1] = replacement.as_dict()
    _rewrite(path, rows)

    with pytest.raises(PopulationJournalError, match="does not match its source metadata"):
        PopulationJournal(path)


def test_full_row_checksum_binds_noncausal_decision_timestamps(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    # decision_ts is intentionally outside input_hash, but it is still evidence
    # carried by the journal and therefore must be bound by the cycle footer.
    rows[1]["decision_ts"] += 0.25
    _rewrite(path, rows)

    with pytest.raises(PopulationDatasetError, match="cycle_rows_checksum_mismatch"):
        list(iter_population_cycles(path))
    with pytest.raises(PopulationJournalError, match="rows checksum mismatch"):
        PopulationJournal(path)


def test_spawned_writers_cannot_append_the_same_cycle_twice(tmp_path) -> None:
    """Both child processes construct stale instances before either may write."""

    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    release = context.Event()
    results = context.Queue()
    path = tmp_path / "population.jsonl"
    processes = [
        context.Process(
            target=_append_empty_cycle_after_release,
            args=(str(path), ready, release, results),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    try:
        assert [ready.get(timeout=20.0) for _ in processes] == [True, True]
        release.set()
        outcomes = [results.get(timeout=20.0) for _ in processes]
    finally:
        release.set()
        for process in processes:
            process.join(timeout=20.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(outcomes) == [("ok", False), ("ok", True)]
    assert len(list(iter_population_cycles(path))) == 1
