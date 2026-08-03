from __future__ import annotations

import json

import pytest

from ai.reversal.feature_contract import build_runtime_feature_snapshot
from ai.reversal.population_dataset import (
    PopulationDatasetError,
    iter_population_feature_rows,
    model_input_records,
    population_feature_records,
)
from trading.market_data.source_timing import SourceTiming
from trading.metrics.cycle_envelope import CycleEnvelope
from trading.metrics.population_journal import PopulationDecision, PopulationJournal, make_cycle_id


_UNIVERSE_TIMING = SourceTiming(
    source="universe",
    request_started_at=1_700_000_000.0,
    received_at=1_700_000_001.0,
)


def _envelope(**overrides) -> CycleEnvelope:
    values = dict(
        cycle_id=_cycle_id(),
        timeframe="60",
        cycle_started_at=1_700_002_801.0,
        candle_cutoff_ts=1_700_002_800.0,
        universe_symbols=("AAAUSDT", "BBBUSDT"),
        universe_timing=_UNIVERSE_TIMING,
        source_timings=(_UNIVERSE_TIMING,),
        strategy_config_hash="a" * 64,
        universe_policy_hash="b" * 64,
        ranking_ready_ts=1_700_002_804.0,
        cycle_completed_ts=1_700_002_805.0,
    )
    values.update(overrides)
    return CycleEnvelope.build(**values)


def _cycle_id() -> str:
    return make_cycle_id(
        timeframe="60",
        candle_cutoff_ts=1_700_002_800.0,
        universe_received_at=1_700_000_001.0,
        universe_symbols=["AAAUSDT", "BBBUSDT"],
    )


def _write(path) -> None:
    PopulationJournal(path).append_cycle(_records(), envelope=_envelope())


def _lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _metadata(*, funding: float | None) -> dict:
    metadata = {
        "universe": {
            "turnover_24h_usdt": 1_000_000.0,
            "change_24h": 0.1,
            "funding_rate": funding,
            "open_interest": None,
            "min_notional_usdt": None,
            "max_leverage": None,
        },
        "base": {"bar_count": 320, "mark_price": 1.0},
        "benchmark_status": "available",
        "provenance": {
            "strategy_config_hash": "a" * 64,
            "universe_policy_hash": "b" * 64,
        },
    }
    metadata["feature_snapshot"] = build_runtime_feature_snapshot(
        metadata,
        bar_cutoff_ts=1_700_002_800.0,
        universe_refreshed_at=1_700_000_000.0,
    )
    return metadata


def _records() -> list[PopulationDecision]:
    cycle_id = _cycle_id()
    out = []
    # Per-symbol decision clocks differ inside one cycle; the cycle-level timing
    # below is identical on every row, which is what makes them one cohort.
    for ordinal, (symbol, funding) in enumerate((("AAAUSDT", 0.0), ("BBBUSDT", None))):
        out.append(
            PopulationDecision.create(
                cycle_id=cycle_id,
                universe_refreshed_at=1_700_000_000.0,
                universe_request_started_at=1_700_000_000.0,
                universe_received_at=1_700_000_001.0,
                scan_observed_at=1_700_002_801.0,
                candle_cutoff_ts=1_700_002_800.0,
                decision_ts=1_700_002_802.0 + ordinal,
                ranking_ready_ts=1_700_002_804.0,
                cycle_completed_ts=1_700_002_805.0,
                actionable_ts=1_700_002_804.0,
                entry_eligible_ts=1_700_002_805.0,
                entry_bar_open_ts=1_700_006_400.0,
                symbol=symbol,
                timeframe="60",
                status="evaluated",
                base_bar_open_ts=1_699_999_200.0,
                base_bar_close_ts=1_700_002_800.0,
                action="HOLD",
                reason="test",
                confidence=0.0,
                metadata=_metadata(funding=funding),
                cycle_ordinal=ordinal,
                cycle_size=2,
            )
        )
    return out


def test_reader_preserves_complete_population_and_real_missingness(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)

    rows = list(iter_population_feature_rows(path))

    assert [row.symbol for row in rows] == ["AAAUSDT", "BBBUSDT"]
    assert [row.action for row in rows] == ["HOLD", "HOLD"]
    assert rows[0].features["funding_rate"] == 0.0
    assert rows[0].observed["funding_rate"] == 1
    assert rows[1].features["funding_rate"] is None
    assert rows[1].observed["funding_rate"] == 0

    flat = population_feature_records(path)
    assert flat[0]["funding_rate"] == 0.0
    assert flat[1]["funding_rate"] is None
    assert flat[1]["funding_rate__observed"] == 0

    model_rows = model_input_records(path)
    assert "action" not in model_rows[0]
    assert "status" not in model_rows[0]
    assert "open_interest" not in model_rows[0]["features"]
    assert "poc" not in model_rows[0]["features"]
    assert "funding_rate" in model_rows[0]["features"]


def test_reader_rejects_contract_hash_drift(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    rows[1]["metadata"]["feature_snapshot"]["contract_hash"] = "0" * 64
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(PopulationDatasetError, match="feature_contract_hash_mismatch"):
        list(iter_population_feature_rows(path))


def test_reader_rejects_feature_tampering_even_with_valid_contract_hash(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    rows[1]["metadata"]["feature_snapshot"]["values"]["funding_rate"] = 999.0
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(PopulationDatasetError, match="feature_snapshot_source_mismatch"):
        list(iter_population_feature_rows(path))


def test_reader_rejects_incomplete_cycle(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    # Drop the footer: a body without its closing record is an unfinished cycle.
    _rewrite(path, rows[:-1])

    with pytest.raises(PopulationDatasetError, match="journal_ends_with_incomplete_cycle"):
        list(iter_population_feature_rows(path))


def test_reader_rejects_a_decision_row_with_no_cycle_header(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    _rewrite(path, rows[1:])

    with pytest.raises(PopulationDatasetError, match="journal_row_before_its_cycle_header"):
        list(iter_population_feature_rows(path))


def test_reader_revalidates_temporal_contract_not_covered_by_input_hash(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    rows[1]["decision_ts"] = rows[1]["candle_cutoff_ts"] - 1.0
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(PopulationDatasetError, match="invalid_population_record"):
        list(iter_population_feature_rows(path))


def test_reader_rejects_duplicate_json_keys_and_nonstandard_numbers(tmp_path) -> None:
    valid_path = tmp_path / "valid.jsonl"
    _write(valid_path)
    first_line = valid_path.read_text(encoding="utf-8").splitlines()[1]

    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text('{"schema_version":1,' + first_line[1:] + "\n", encoding="utf-8")
    with pytest.raises(PopulationDatasetError, match="invalid_json_line"):
        list(iter_population_feature_rows(duplicate_path))

    nonstandard_path = tmp_path / "nonstandard.jsonl"
    row = json.loads(first_line)
    row["decision_ts"] = float("nan")
    nonstandard_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(PopulationDatasetError, match="invalid_json_line"):
        list(iter_population_feature_rows(nonstandard_path))
