from __future__ import annotations

import json

import pytest

from core.mexc_strategy_spec import (
    MEXC_STRATEGY_SPEC_VERSION,
    load_mexc_strategy_spec,
    strategy_spec_contract_hash,
    strategy_spec_identity,
)
from ai.reversal.feature_contract import build_runtime_feature_snapshot, market_feature_hash
from ai.reversal.population_dataset import (
    PopulationDatasetError,
    iter_population_cycles,
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
    source_as_of=1_699_999_990.0,
    cache_hit=True,
    cache_age_sec=11.0,
    source_ts=1_699_999_990.0,
)
_STRATEGY_SPEC = load_mexc_strategy_spec()


def _envelope(**overrides) -> CycleEnvelope:
    values = dict(
        cycle_id=_cycle_id(),
        timeframe="60",
        cycle_started_at=1_700_002_801.0,
        candle_cutoff_ts=1_700_002_800.0,
        universe_symbols=("AAAUSDT", "BBBUSDT"),
        universe_timing=_UNIVERSE_TIMING,
        source_timings=(_UNIVERSE_TIMING,),
        strategy_spec_version=MEXC_STRATEGY_SPEC_VERSION,
        strategy_spec_contract_hash=strategy_spec_contract_hash(),
        strategy_spec_instance_hash=_STRATEGY_SPEC.instance_hash,
        strategy_spec_payload=_STRATEGY_SPEC.to_mapping(),
        universe_policy_hash="b" * 64,
        ranking_ready_ts=1_700_002_804.0,
        cycle_completed_ts=1_700_002_805.0,
    )
    values.update(overrides)
    if "cycle_id" not in overrides:
        values["cycle_id"] = make_cycle_id(
            timeframe=values["timeframe"],
            candle_cutoff_ts=values["candle_cutoff_ts"],
            universe_received_at=values["universe_timing"].received_at,
            universe_symbols=values["universe_symbols"],
        )
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


def _metadata(*, funding: float | None, symbol: str = "AAAUSDT") -> dict:
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
            "strategy_config_hash": _STRATEGY_SPEC.instance_hash,
            "universe_policy_hash": "b" * 64,
        },
    }
    metadata["feature_snapshot"] = build_runtime_feature_snapshot(
        metadata,
        bar_cutoff_ts=1_700_002_800.0,
    )
    metadata["feature_provenance"] = {
        "universe_received_at": _UNIVERSE_TIMING.received_at,
        "universe_source_ts": _UNIVERSE_TIMING.source_ts,
        "universe_cache_hit": _UNIVERSE_TIMING.cache_hit,
        "envelope_hash": _envelope().envelope_hash(),
        "market_feature_hash": market_feature_hash(
            metadata["feature_snapshot"], symbol=symbol, timeframe_seconds=3600
        ),
    }
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
                metadata=_metadata(funding=funding, symbol=symbol),
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
    assert flat[0]["envelope_hash"] == rows[0].envelope_hash
    assert flat[0]["market_feature_hash"] == rows[0].market_feature_hash

    model_rows = model_input_records(path, allow_unanchored=True)
    identity = strategy_spec_identity(_STRATEGY_SPEC)
    assert "action" not in model_rows[0]
    assert "status" not in model_rows[0]
    assert "open_interest" not in model_rows[0]["features"]
    assert "poc" not in model_rows[0]["features"]
    assert "funding_rate" in model_rows[0]["features"]
    assert model_rows[0]["envelope_hash"] == rows[0].envelope_hash
    assert model_rows[0]["market_feature_hash"] == rows[0].market_feature_hash
    assert model_rows[0]["strategy_spec_version"] == identity.spec_version
    assert model_rows[0]["strategy_spec_contract_hash"] == identity.contract_hash
    assert model_rows[0]["strategy_spec_instance_hash"] == identity.instance_hash
    assert set(model_rows[0]["features"]) == set(model_rows[0]["feature_names"])
    assert set(model_rows[0]["observed"]) == set(model_rows[0]["feature_names"])
    for identity_field in (
        "strategy_spec_version",
        "strategy_spec_contract_hash",
        "strategy_spec_instance_hash",
    ):
        assert identity_field not in model_rows[0]["features"]
        assert identity_field not in model_rows[0]["observed"]


def test_reader_round_trips_all_scanner_cache_provenance(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)

    envelope, _ = next(iter(iter_population_cycles(path)))
    assert envelope.universe_timing.as_dict() == _UNIVERSE_TIMING.as_dict()
    assert envelope.universe_timing.cache_hit is True
    assert envelope.universe_timing.cache_age_sec == 11.0
    assert envelope.universe_timing.source_ts == 1_699_999_990.0


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


def test_reader_rejects_row_envelope_hash_tampering(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    rows[1]["metadata"]["feature_provenance"]["envelope_hash"] = "0" * 64
    _rewrite(path, rows)

    with pytest.raises(PopulationDatasetError, match="envelope_hash"):
        list(iter_population_feature_rows(path))


def test_reader_rejects_market_feature_hash_tampering(tmp_path) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    rows[1]["metadata"]["feature_provenance"]["market_feature_hash"] = "0" * 64
    _rewrite(path, rows)

    with pytest.raises(PopulationDatasetError, match="market_feature_hash_mismatch"):
        list(iter_population_feature_rows(path))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("universe_received_at", _UNIVERSE_TIMING.received_at + 1.0, "received_at_mismatch"),
        ("universe_source_ts", _UNIVERSE_TIMING.source_ts + 1.0, "universe_source_ts"),
        ("universe_cache_hit", False, "universe_cache_hit"),
        ("unexpected", "value", "feature_provenance_schema_mismatch"),
    ],
)
def test_reader_rejects_drifted_feature_provenance(
    tmp_path, field, value, message
) -> None:
    path = tmp_path / "population.jsonl"
    _write(path)
    rows = _lines(path)
    rows[1]["metadata"]["feature_provenance"][field] = value
    _rewrite(path, rows)

    with pytest.raises(PopulationDatasetError, match=message):
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
