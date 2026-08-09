"""Frozen v5 evidence for the future journal-version dispatch boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from ai.reversal.population_dataset import (
    iter_population_cycles,
    model_input_records,
    verify_population_journal,
)
from trading.metrics.population_journal import (
    CHECKPOINT_RECEIPT_SCHEMA_VERSION,
    CYCLE_IDENTITY_VERSION,
    SCHEMA_VERSION,
    JournalCheckpointReceipt,
    PopulationJournal,
    compute_cycle_commit,
    genesis_cycle_commit,
    rows_checksum,
)


_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
JOURNAL_PATH = _FIXTURES / "mexc_population_journal_v5_minimal.jsonl"
RECEIPT_PATH = (
    _FIXTURES / "mexc_population_journal_v5_minimal_receipt.json"
)

JOURNAL_FILE_SHA256 = (
    "2b782e76d668f9d90efde3229cdf3b9287c1b7fd688947d4ab1066c8a07a1529"
)
RECEIPT_FILE_SHA256 = (
    "2ecc55423983f2ab79698f0755f3bd3fb06f743f818c293c6373217c6cb8097b"
)
JOURNAL_LENGTH_BYTES = 10_247
JOURNAL_ID = "a" * 64
CYCLE_ID = "b7f8d578c3b0077aca95817af6f913dbf71ab1801932c1528e5014dcc2beaa7b"
GENESIS_COMMIT = "2dc1afd6082ddb07b9e1c3bc657d03e99ad7073cc933dbfd0bf29dbc22b47020"
ENVELOPE_HASH = "63354037fa5d417215b9f4275d5ca04fce40b94b8400085bf8d6ca11d281c004"
INPUT_HASH = "cccc40f43bced7b35f2715803f60650cfc330dca66d2c9e890d6053e41a1f0c8"
SNAPSHOT_ID = "69c1e5d477e4d149c397265004611764dfffad4f11ac3517fbb83db690141f10"
MARKET_FEATURE_HASH = (
    "1088d2a88b15e6d792ab090436ec8550a54929a7a0def312703b94595511a090"
)
ROWS_CHECKSUM = "841a3c7354b95d3baa84e4c02c084bd6fce1e49d00e77559612758bbdae01e44"
CYCLE_COMMIT = "48df017be8e99d822e60f6e4cc3264ee431d4c3137ec02559686d0487ee1c785"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _journal_records() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
    ]


def _receipt() -> JournalCheckpointReceipt:
    return JournalCheckpointReceipt.from_dict(
        json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    )


def test_frozen_v5_bytes_hashes_and_chain_are_exact() -> None:
    journal_bytes = JOURNAL_PATH.read_bytes()
    receipt_bytes = RECEIPT_PATH.read_bytes()
    assert len(journal_bytes) == JOURNAL_LENGTH_BYTES
    assert journal_bytes.endswith(b"\n")
    assert _sha256(journal_bytes) == JOURNAL_FILE_SHA256
    assert _sha256(receipt_bytes) == RECEIPT_FILE_SHA256

    header, row, footer = _journal_records()
    assert [
        header["record_type"],
        row["record_type"],
        footer["record_type"],
    ] == ["cycle_header", "decision", "cycle_footer"]
    assert {
        header["schema_version"],
        row["schema_version"],
        footer["schema_version"],
    } == {5}
    assert header["journal_id"] == footer["journal_id"] == JOURNAL_ID
    assert header["cycle_id"] == row["cycle_id"] == footer["cycle_id"] == CYCLE_ID
    assert header["envelope_hash"] == footer["envelope_hash"] == ENVELOPE_HASH
    assert row["input_hash"] == INPUT_HASH
    assert row["snapshot_id"] == SNAPSHOT_ID
    assert (
        row["metadata"]["feature_provenance"]["market_feature_hash"]
        == MARKET_FEATURE_HASH
    )

    assert genesis_cycle_commit(JOURNAL_ID) == GENESIS_COMMIT
    assert (
        header["prev_cycle_commit"]
        == footer["prev_cycle_commit"]
        == GENESIS_COMMIT
    )
    assert rows_checksum((row,)) == footer["rows_checksum"] == ROWS_CHECKSUM
    footer_core = {key: value for key, value in footer.items() if key != "cycle_commit"}
    assert compute_cycle_commit(header, (row,), footer_core) == CYCLE_COMMIT
    assert footer["cycle_commit"] == CYCLE_COMMIT


def test_frozen_v5_receipt_anchors_the_exact_current_reader_view(tmp_path) -> None:
    assert SCHEMA_VERSION == 5
    assert CYCLE_IDENTITY_VERSION == 5
    assert CHECKPOINT_RECEIPT_SCHEMA_VERSION == 1

    receipt = _receipt()
    assert receipt.as_dict() == {
        "record_type": "population_journal_checkpoint",
        "receipt_schema_version": 1,
        "journal_schema_version": 5,
        "journal_id": JOURNAL_ID,
        "sequence_no": 0,
        "cycle_id": CYCLE_ID,
        "cycle_commit": CYCLE_COMMIT,
        "prefix_length_bytes": JOURNAL_LENGTH_BYTES,
        "prefix_sha256": JOURNAL_FILE_SHA256,
    }

    trust = verify_population_journal(JOURNAL_PATH, trusted_checkpoint=receipt)
    assert trust.integrity == "anchored_through_sequence=0"
    assert trust.last_sequence_no == 0
    assert trust.last_cycle_id == CYCLE_ID
    assert trust.last_cycle_commit == CYCLE_COMMIT

    cycles = list(iter_population_cycles(JOURNAL_PATH, trusted_checkpoint=receipt))
    assert len(cycles) == 1
    envelope, rows = cycles[0]
    assert envelope.cycle_id == CYCLE_ID
    assert envelope.envelope_hash() == ENVELOPE_HASH
    assert len(rows) == 1
    assert rows[0].snapshot_id == SNAPSHOT_ID
    assert rows[0].market_feature_hash == MARKET_FEATURE_HASH

    model_rows = model_input_records(JOURNAL_PATH, trusted_checkpoint=receipt)
    assert len(model_rows) == 1
    assert model_rows[0]["snapshot_id"] == SNAPSHOT_ID

    # The append-side v5 reader uses an advisory sidecar, so validate it against
    # an exact temporary copy and leave the committed evidence directory inert.
    copied = tmp_path / JOURNAL_PATH.name
    shutil.copyfile(JOURNAL_PATH, copied)
    reopened = PopulationJournal(copied)
    assert reopened.checkpoint_receipt() == receipt
    assert copied.read_bytes() == JOURNAL_PATH.read_bytes()
