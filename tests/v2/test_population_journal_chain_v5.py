from __future__ import annotations

from copy import deepcopy
import json
import os

import pytest

import ai.reversal.population_dataset as population_dataset
from ai.reversal.population_dataset import (
    PopulationDatasetError,
    iter_population_feature_rows,
    iter_population_cycles,
    model_input_records,
    population_feature_records,
    verify_population_journal,
)
from core.mexc_strategy_spec import parse_mexc_strategy_spec, strategy_spec_identity
from trading.metrics.cycle_envelope import CycleEnvelope
from trading.metrics.population_journal import (
    JournalCheckpointReceipt,
    PopulationJournal,
    PopulationJournalError,
    compute_cycle_commit,
    genesis_cycle_commit,
    rows_checksum,
)

from v2.test_population_feature_dataset_v2 import _envelope, _records


def _lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite(path, lines: list[dict]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                line,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )


def _rehash_cycle(lines: list[dict], *, header_index: int) -> str:
    footer_index = next(
        index
        for index in range(header_index + 1, len(lines))
        if lines[index].get("record_type") == "cycle_footer"
    )
    rows = lines[header_index + 1 : footer_index]
    footer = lines[footer_index]
    footer["rows_checksum"] = rows_checksum(rows)
    footer_core = {key: value for key, value in footer.items() if key != "cycle_commit"}
    footer["cycle_commit"] = compute_cycle_commit(
        lines[header_index],
        rows,
        footer_core,
    )
    return footer["cycle_commit"]


def _with_alternate_strategy_identity(envelope: CycleEnvelope) -> CycleEnvelope:
    payload = deepcopy(envelope.as_dict())
    strategy_payload = payload["strategy_spec_payload"]
    strategy_payload["signal"]["rsi_high"] = 76.0
    spec = parse_mexc_strategy_spec(strategy_payload)
    identity = strategy_spec_identity(spec)
    payload["strategy_spec_version"] = identity.spec_version
    payload["strategy_spec_contract_hash"] = identity.contract_hash
    payload["strategy_spec_instance_hash"] = identity.instance_hash
    return CycleEnvelope.from_dict(payload)


def test_writer_builds_one_contiguous_domain_separated_chain(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    journal = PopulationJournal(path)
    assert journal.append_cycle(_records(), envelope=_envelope()) is True
    assert journal.append_cycle(
        (),
        envelope=_envelope(universe_symbols=(), status="empty_universe"),
    ) is True

    lines = _lines(path)
    headers = [line for line in lines if line["record_type"] == "cycle_header"]
    footers = [line for line in lines if line["record_type"] == "cycle_footer"]
    assert [header["sequence_no"] for header in headers] == [0, 1]
    assert headers[0]["journal_id"] == headers[1]["journal_id"]
    assert headers[0]["prev_cycle_commit"] == genesis_cycle_commit(
        headers[0]["journal_id"]
    )
    assert headers[1]["prev_cycle_commit"] == footers[0]["cycle_commit"]
    assert footers[1]["prev_cycle_commit"] == footers[0]["cycle_commit"]

    state = verify_population_journal(path)
    assert state.integrity == "internally_consistent_unanchored"
    assert state.last_sequence_no == 1
    assert state.last_cycle_commit == footers[1]["cycle_commit"]


def test_writer_rejects_a_second_strategy_identity_without_touching_the_file(
    tmp_path,
) -> None:
    path = tmp_path / "population.jsonl"
    journal = PopulationJournal(path)
    journal.append_cycle(_records(), envelope=_envelope())
    before = path.read_bytes()
    alternate = _with_alternate_strategy_identity(
        _envelope(universe_symbols=(), status="empty_universe")
    )

    with pytest.raises(
        PopulationJournalError,
        match="population journal mixes strategy identities",
    ):
        journal.append_cycle((), envelope=alternate)

    assert path.read_bytes() == before
    PopulationJournal(path)


def test_restart_reader_and_model_export_reject_a_coherently_mixed_identity_file(
    tmp_path,
) -> None:
    path = tmp_path / "population.jsonl"
    journal = PopulationJournal(path)
    journal.append_cycle(_records(), envelope=_envelope())
    journal.append_cycle(
        (),
        envelope=_envelope(universe_symbols=(), status="empty_universe"),
    )

    lines = _lines(path)
    second_header = next(
        index
        for index, line in enumerate(lines)
        if index > 0 and line.get("record_type") == "cycle_header"
    )
    second_footer = next(
        index
        for index in range(second_header + 1, len(lines))
        if lines[index].get("record_type") == "cycle_footer"
    )
    alternate = _with_alternate_strategy_identity(
        CycleEnvelope.from_dict(lines[second_header]["envelope"])
    )
    alternate_hash = alternate.envelope_hash()
    lines[second_header]["envelope"] = alternate.as_dict()
    lines[second_header]["envelope_hash"] = alternate_hash
    lines[second_footer]["envelope_hash"] = alternate_hash
    _rehash_cycle(lines, header_index=second_header)
    _rewrite(path, lines)

    with pytest.raises(
        PopulationJournalError,
        match="population journal mixes strategy identities",
    ):
        PopulationJournal(path)
    with pytest.raises(PopulationDatasetError, match="mixed_strategy_spec_identities"):
        verify_population_journal(path)
    with pytest.raises(PopulationDatasetError, match="mixed_strategy_spec_identities"):
        model_input_records(path, allow_unanchored=True)


def test_explicit_checkpoint_anchors_only_its_prefix_by_default(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    journal = PopulationJournal(path)
    journal.append_cycle(_records(), envelope=_envelope())
    receipt = journal.checkpoint_receipt()
    journal.append_cycle(
        (),
        envelope=_envelope(universe_symbols=(), status="empty_universe"),
    )

    state = verify_population_journal(path, trusted_checkpoint=receipt)
    assert state.integrity == "anchored_through_sequence=0"
    assert state.anchored_through_sequence_no == 0
    assert state.last_sequence_no == 1
    assert len(list(iter_population_cycles(path, trusted_checkpoint=receipt))) == 1
    assert len(
        list(
            iter_population_cycles(
                path,
                trusted_checkpoint=receipt,
                anchored_only=False,
            )
        )
    ) == 2
    assert len(model_input_records(path, trusted_checkpoint=receipt)) == 2


def test_model_inputs_require_an_anchor_or_an_explicit_unsafe_override(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    PopulationJournal(path).append_cycle(_records(), envelope=_envelope())

    with pytest.raises(
        PopulationDatasetError,
        match="trusted_checkpoint_required_for_model_inputs",
    ):
        model_input_records(path)
    assert len(model_input_records(path, allow_unanchored=True)) == 2


def test_explicit_anchored_reads_require_a_checkpoint(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    PopulationJournal(path).append_cycle(_records(), envelope=_envelope())

    # The neutral default remains the generic integrity-only reader.  Callers
    # that explicitly request anchored evidence must actually provide a receipt.
    assert len(list(iter_population_cycles(path))) == 1
    assert len(list(iter_population_cycles(path, anchored_only=False))) == 1
    with pytest.raises(
        PopulationDatasetError,
        match="trusted_checkpoint_required_for_anchored_read",
    ):
        list(iter_population_cycles(path, anchored_only=True))
    with pytest.raises(
        PopulationDatasetError,
        match="trusted_checkpoint_required_for_anchored_read",
    ):
        list(iter_population_feature_rows(path, anchored_only=True))
    with pytest.raises(
        PopulationDatasetError,
        match="trusted_checkpoint_required_for_anchored_read",
    ):
        population_feature_records(path, anchored_only=True)


def test_coordinated_whole_file_rewrite_is_unanchored_but_receipt_detects_it(
    tmp_path,
) -> None:
    path = tmp_path / "population.jsonl"
    journal = PopulationJournal(path)
    journal.append_cycle(_records(), envelope=_envelope())
    receipt = journal.checkpoint_receipt()

    lines = _lines(path)
    lines[1]["decision_ts"] += 0.25
    _rehash_cycle(lines, header_index=0)
    _rewrite(path, lines)

    # A public hash chain cannot authenticate a coordinated rewrite by itself.
    assert verify_population_journal(path).integrity == "internally_consistent_unanchored"
    PopulationJournal(path)  # restart validation proves internal consistency only
    with pytest.raises(PopulationDatasetError, match="trusted_checkpoint_cycle_commit_mismatch"):
        verify_population_journal(path, trusted_checkpoint=receipt)


def test_rehashed_earlier_cycle_breaks_the_unchanged_successor(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    journal = PopulationJournal(path)
    journal.append_cycle(_records(), envelope=_envelope())
    journal.append_cycle(
        (),
        envelope=_envelope(universe_symbols=(), status="empty_universe"),
    )

    lines = _lines(path)
    lines[1]["decision_ts"] += 0.25
    _rehash_cycle(lines, header_index=0)
    _rewrite(path, lines)

    with pytest.raises(PopulationDatasetError, match="cycle_predecessor_mismatch"):
        verify_population_journal(path)
    with pytest.raises(PopulationJournalError, match="chain predecessor mismatch"):
        PopulationJournal(path)


def test_stale_instance_rejects_a_self_consistent_rewrite_of_observed_history(
    tmp_path,
) -> None:
    path = tmp_path / "population.jsonl"
    writer = PopulationJournal(path)
    writer.append_cycle(_records(), envelope=_envelope())
    stale = PopulationJournal(path)

    lines = _lines(path)
    lines[1]["decision_ts"] += 0.25
    _rehash_cycle(lines, header_index=0)
    _rewrite(path, lines)

    with pytest.raises(PopulationJournalError, match="history was rewritten"):
        stale.append_cycle(
            (),
            envelope=_envelope(universe_symbols=(), status="empty_universe"),
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("journal_id", "f" * 64, "trusted_checkpoint_journal_id_mismatch"),
        ("prefix_sha256", "f" * 64, "trusted_checkpoint_prefix_sha256_mismatch"),
        ("prefix_length_bytes", 1, "trusted_checkpoint_prefix_length_mismatch"),
    ],
)
def test_checkpoint_receipt_is_exact_and_never_auto_discovered(
    tmp_path,
    field,
    replacement,
    message,
) -> None:
    path = tmp_path / "population.jsonl"
    journal = PopulationJournal(path)
    journal.append_cycle(_records(), envelope=_envelope())
    receipt = journal.checkpoint_receipt()
    payload = receipt.as_dict()
    payload[field] = replacement
    bad = JournalCheckpointReceipt.from_dict(payload)

    # Merely possessing a bad sidecar-like object changes no default trust claim.
    assert verify_population_journal(path).integrity == "internally_consistent_unanchored"
    with pytest.raises(PopulationDatasetError, match=message):
        verify_population_journal(path, trusted_checkpoint=bad)


def test_anchored_iterator_never_yields_a_same_size_replacement_between_passes(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "population.jsonl"
    alternate = tmp_path / "alternate.jsonl"
    journal = PopulationJournal(path)
    journal.append_cycle(_records(), envelope=_envelope())
    journal.append_cycle(
        (),
        envelope=_envelope(universe_symbols=(), status="empty_universe"),
    )
    receipt = journal.checkpoint_receipt()
    original_stat = path.stat()

    lines = _lines(path)
    lines[1]["decision_ts"] += 1.0
    first_commit = _rehash_cycle(lines, header_index=0)
    second_header = next(
        index
        for index, line in enumerate(lines)
        if index > 0 and line.get("record_type") == "cycle_header"
    )
    second_footer = next(
        index
        for index in range(second_header + 1, len(lines))
        if lines[index].get("record_type") == "cycle_footer"
    )
    lines[second_header]["prev_cycle_commit"] = first_commit
    lines[second_footer]["prev_cycle_commit"] = first_commit
    _rehash_cycle(lines, header_index=second_header)
    alternate.write_bytes(
        (
            "\n".join(
                json.dumps(
                    line,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for line in lines
            )
            + "\n"
        ).encode("utf-8")
    )
    os.utime(
        alternate,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert alternate.stat().st_size == original_stat.st_size

    real_fingerprint = population_dataset._journal_fingerprint
    calls = 0
    validated_fingerprint = None
    replacement_changed_identity = False

    def replace_after_first_pass(source):
        nonlocal calls, validated_fingerprint, replacement_changed_identity
        calls += 1
        fingerprint = real_fingerprint(source)
        if calls == 2:
            validated_fingerprint = fingerprint
            os.replace(alternate, source)
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            replacement_changed_identity = real_fingerprint(source) != fingerprint
            return fingerprint
        if validated_fingerprint is not None:
            # Even if a filesystem or hostile writer concealed every metadata
            # change, the independently rebuilt cycle commitment must still
            # block the replacement before a row can be yielded.
            return validated_fingerprint
        return fingerprint

    monkeypatch.setattr(
        population_dataset,
        "_journal_fingerprint",
        replace_after_first_pass,
    )
    iterator = iter_population_cycles(
        path,
        trusted_checkpoint=receipt,
        anchored_only=True,
    )
    with pytest.raises(
        PopulationDatasetError,
        match="population_journal_cycle_changed_after_validation",
    ):
        next(iterator)
    assert replacement_changed_identity is True
