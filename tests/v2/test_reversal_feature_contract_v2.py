from __future__ import annotations

import re

from ai.reversal.feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_SPECS,
    RuntimeStatus,
    build_runtime_feature_snapshot,
    captured_feature_specs,
    feature_contract_hash,
    feature_registry_hash,
    market_feature_hash,
    model_feature_names,
)


def _metadata() -> dict:
    return {
        "universe": {
            "turnover_24h_usdt": 2_500_000.0,
            "change_24h": 0.42,
            "funding_rate": 0.0,
            "open_interest": None,
            "min_notional_usdt": 5.0,
            "max_leverage": 50.0,
        },
        "base": {"bar_count": 320, "mark_price": 1.25},
        "strategy": {
            "layer_trace": {
                "layers": {
                    "layer1_pump_detection": {
                        "details": {
                            "run_up_pct": 0.08,
                            "rsi": 79.0,
                            "volume_spike": 3.2,
                        }
                    },
                    "layer4_fake_filter": {
                        "details": {
                            "sentiment": 50.0,
                            "sentiment_available": 0.0,
                            "funding_rate": 0.0,
                            "funding_available": 1.0,
                            "long_short_ratio": 0.0,
                            "long_short_ratio_available": 0.0,
                        }
                    },
                    "layer3_entry_location": {
                        "details": {
                            "poc": 0.0,
                            "vah": 0.0,
                            "val": 0.0,
                            "vp_levels_available": 0.0,
                        }
                    },
                }
            }
        },
    }


def test_registry_is_unique_versioned_and_causal() -> None:
    names = [spec.name for spec in FEATURE_SPECS]
    assert len(names) == len(set(names))
    assert FEATURE_CONTRACT_VERSION == "mexc_reversal_features_v2"
    assert feature_contract_hash() == "20f9f61d4e2d787c5ad05f54ee3ccd8b7f8ea3a99fe09bc38bbefe09872c496c"
    assert re.fullmatch(r"[0-9a-f]{64}", feature_registry_hash())
    assert feature_contract_hash() != feature_registry_hash()
    assert {spec.timing for spec in FEATURE_SPECS}.issubset(
        {
            "closed_bar_cutoff",
            "universe_snapshot_before_decision",
            "context_snapshot_before_decision",
        }
    )
    assert next(spec for spec in FEATURE_SPECS if spec.name == "rsi_entry").timing == "closed_bar_cutoff"
    assert next(spec for spec in FEATURE_SPECS if spec.name == "funding_rate").timing == "universe_snapshot_before_decision"
    assert not {"action", "confidence", "target_win", "future_return"}.intersection(names)


def test_only_wired_or_conditional_features_are_captured() -> None:
    captured = captured_feature_specs()
    assert captured
    assert all(spec.path is not None for spec in captured)
    assert all(
        spec.runtime_status
        in {RuntimeStatus.WIRED, RuntimeStatus.CONDITIONAL, RuntimeStatus.SOURCE_MISSING}
        for spec in captured
    )
    assert "fib_618_distance" not in {spec.name for spec in captured}
    assert "news_context_score" not in {spec.name for spec in captured}
    assert "open_interest" not in model_feature_names()
    assert "poc" not in model_feature_names()
    assert "layer4_funding_rate" not in model_feature_names()
    assert "funding_rate" in model_feature_names()


def test_snapshot_keeps_real_zero_distinct_from_missing() -> None:
    snapshot = build_runtime_feature_snapshot(
        _metadata(),
        bar_cutoff_ts=1_700_002_800.0,
    )

    assert snapshot["contract_version"] == FEATURE_CONTRACT_VERSION
    assert snapshot["contract_hash"] == feature_contract_hash()
    assert snapshot["values"]["funding_rate"] == 0.0
    assert snapshot["observed"]["funding_rate"] == 1
    assert snapshot["values"]["open_interest"] is None
    assert snapshot["observed"]["open_interest"] == 0
    assert "open_interest" in snapshot["missing"]
    assert snapshot["availability_reason"]["open_interest"] == "source_unavailable_or_invalid"

    # A value serialized as zero is still not an observation when its explicit
    # provenance bit says the upstream source was absent.
    assert snapshot["values"]["long_short_ratio"] == 0.0
    assert snapshot["observed"]["long_short_ratio"] == 0
    assert "long_short_ratio" in snapshot["missing"]
    assert snapshot["availability_reason"]["long_short_ratio"] == "provenance_unavailable"
    assert snapshot["values"]["long_short_ratio_observed"] == 0.0
    assert snapshot["observed"]["long_short_ratio_observed"] == 1
    assert snapshot["values"]["sentiment_index"] == 50.0
    assert snapshot["observed"]["sentiment_index"] == 0
    assert snapshot["observed"]["poc"] == 0


def test_partial_gate_trace_has_fixed_schema_and_explicit_missingness() -> None:
    snapshot = build_runtime_feature_snapshot(
        _metadata(),
        bar_cutoff_ts=1_700_002_800.0,
    )
    expected_names = [spec.name for spec in captured_feature_specs()]

    assert list(snapshot["values"]) == expected_names
    assert list(snapshot["observed"]) == expected_names
    assert snapshot["values"]["pump_run_up_pct"] == 0.08
    assert snapshot["values"]["rsi_4h"] is None
    assert snapshot["observed"]["rsi_4h"] == 0
    assert snapshot["availability_reason"]["rsi_4h"] == "not_computed"
    assert snapshot["coverage"]["expected"] == len(expected_names)
    assert 0.0 < snapshot["coverage"]["fraction"] < 1.0


def test_market_feature_hash_binds_instrument_timeframe_and_snapshot() -> None:
    snapshot = build_runtime_feature_snapshot(
        _metadata(),
        bar_cutoff_ts=1_700_002_800.0,
    )

    digest = market_feature_hash(snapshot, symbol="AAAUSDT", timeframe_seconds=3600)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert (
        market_feature_hash(dict(snapshot), symbol="AAAUSDT", timeframe_seconds=3600)
        == digest
    )
    assert market_feature_hash(snapshot, symbol="BBBUSDT", timeframe_seconds=3600) != digest
    assert market_feature_hash(snapshot, symbol="AAAUSDT", timeframe_seconds=900) != digest

    changed = dict(snapshot)
    changed["source_times"] = {"bar_cutoff_ts": 1_700_003_100.0}
    assert (
        market_feature_hash(changed, symbol="AAAUSDT", timeframe_seconds=3600)
        != digest
    )
