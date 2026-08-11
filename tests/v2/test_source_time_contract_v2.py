"""Slice C: what each source knew, and when it actually knew it.

The snapshot used to be stamped with `universe_refreshed_at`, read before the
universe request, and a cached ticker response was dated with the current instant
as though the exchange had just answered. Both inflated provenance. Wall-clock
instants are now recorded beside the market snapshot rather than inside its
identity, so the same bars hash the same however slowly they were fetched.
"""

from __future__ import annotations

import time

import pytest

from ai.reversal.feature_contract import (
    FEATURE_CONTRACT_VERSION,
    build_runtime_feature_snapshot,
    feature_contract_hash,
    market_feature_hash,
)
from trading.market_data.source_timing import SourceTiming, SourceTimingError
from trading.market_data.universe import SymbolUniverse, UniverseConfig
from trading.metrics.cycle_envelope import TIMING_BASIS_RESEARCH_RANKING

from v2.test_population_feature_dataset_v2 import _envelope, _metadata, _records


_CARRIED = (
    "cycle_id", "symbol", "timeframe", "status", "action", "reason", "confidence",
    "candle_cutoff_ts", "base_bar_open_ts", "base_bar_close_ts", "cycle_ordinal",
    "cycle_size", "universe_refreshed_at", "universe_request_started_at",
    "universe_received_at", "scan_observed_at",
    "error_code", "base_source_evidence", "higher_timeframe_source_evidence",
    "benchmark_source_evidence", "lifecycle_event",
)


def _rebuild(base, **changes):
    values = {key: getattr(base, key) for key in _CARRIED}
    values.update(
        {
            "decision_ts": base.decision_ts,
            "ranking_ready_ts": base.ranking_ready_ts,
            "cycle_completed_ts": base.cycle_completed_ts,
            "actionable_ts": base.actionable_ts,
            "entry_eligible_ts": base.entry_eligible_ts,
            "entry_bar_open_ts": base.entry_bar_open_ts,
            "metadata": _metadata(funding=0.0),
        }
    )
    values.update(changes)
    return base.__class__.create(**values)


def test_contract_is_v2_and_its_executable_hash_is_pinned() -> None:
    assert FEATURE_CONTRACT_VERSION == "mexc_reversal_features_v2"
    assert feature_contract_hash() == (
        "20f9f61d4e2d787c5ad05f54ee3ccd8b7f8ea3a99fe09bc38bbefe09872c496c"
    )


def test_snapshot_carries_no_wall_clock() -> None:
    snapshot = build_runtime_feature_snapshot({}, bar_cutoff_ts=1_700_002_800.0)
    assert set(snapshot["source_times"]) == {"bar_cutoff_ts"}


def test_runtime_timing_changes_row_identity_but_not_the_market_feature_hash() -> None:
    """A slower cycle is a distinct observation of the same market features."""
    fast = _records()[0]
    slow = _rebuild(
        fast,
        # Real cycle identity contains the universe response instant, so another
        # observation cycle intentionally receives another row identity.
        cycle_id="f" * 64,
        decision_ts=fast.decision_ts + 45.0,
        ranking_ready_ts=fast.ranking_ready_ts + 60.0,
        cycle_completed_ts=fast.cycle_completed_ts + 60.0,
        actionable_ts=fast.actionable_ts + 60.0,
        entry_eligible_ts=fast.entry_eligible_ts + 60.0,
    )

    assert slow.input_hash != fast.input_hash
    assert slow.snapshot_id != fast.snapshot_id
    assert market_feature_hash(
        slow.metadata["feature_snapshot"], symbol=slow.symbol, timeframe_seconds=3600
    ) == market_feature_hash(
        fast.metadata["feature_snapshot"], symbol=fast.symbol, timeframe_seconds=3600
    )


def test_feature_provenance_is_recorded_but_not_hashed() -> None:
    base = _records()[0]
    metadata = dict(_metadata(funding=0.0))
    metadata["feature_provenance"] = {
        "universe_received_at": 1.0,
        "envelope_hash": "f" * 64,
    }
    with_provenance = _rebuild(base, metadata=metadata)

    assert with_provenance.input_hash == base.input_hash
    assert "feature_provenance" in with_provenance.metadata


def test_started_after_received_fails_closed_instead_of_being_clamped() -> None:
    with pytest.raises(SourceTimingError, match="received_at_precedes_request_started_at"):
        SourceTiming(source="universe_ticker", request_started_at=200.0, received_at=100.0)


def test_a_cache_hit_keeps_its_own_source_instant() -> None:
    timing = SourceTiming(
        source="universe_ticker",
        request_started_at=500.0,
        received_at=500.1,
        cache_hit=True,
        cache_age_sec=120.0,
        source_ts=380.0,
    )
    assert timing.source_ts == 380.0
    assert timing.as_dict()["cache_hit"] is True


def test_a_cache_hit_must_say_when_the_data_was_produced() -> None:
    with pytest.raises(SourceTimingError, match="cache_hit_requires_source_ts"):
        SourceTiming(
            source="universe_ticker", request_started_at=1.0, received_at=2.0, cache_hit=True
        )


def test_a_cache_hit_must_report_its_age() -> None:
    with pytest.raises(SourceTimingError, match="cache_hit_requires_cache_age_sec"):
        SourceTiming(
            source="universe_ticker",
            request_started_at=2.0,
            received_at=3.0,
            cache_hit=True,
            source_ts=1.0,
        )


def test_cache_age_must_match_the_source_and_observation_interval() -> None:
    with pytest.raises(
        SourceTimingError, match="cache_age_sec_is_incoherent_with_source_ts"
    ):
        SourceTiming(
            source="universe_ticker",
            request_started_at=100.0,
            received_at=101.0,
            cache_hit=True,
            cache_age_sec=0.0,
            source_ts=10.0,
        )


def test_source_ts_may_not_follow_the_response() -> None:
    with pytest.raises(SourceTimingError, match="source_ts_follows_received_at"):
        SourceTiming(
            source="universe_ticker", request_started_at=1.0, received_at=2.0, source_ts=3.0
        )


@pytest.mark.parametrize("status", ["error", "stale_cache"])
def test_non_ok_source_as_of_may_not_follow_the_response(status: str) -> None:
    with pytest.raises(SourceTimingError, match="source_as_of_follows_received_at"):
        SourceTiming(
            source="base_ohlcv",
            request_started_at=10.0,
            received_at=11.0,
            status=status,
            source_as_of=999.0,
            error_code="SyntheticUnavailable",
        )


class _CachingClient:
    """Answers once, then serves the same rows from cache."""

    def __init__(self):
        self.requests = 0
        self.first_received = 0.0

    def fetch_all_tickers_with_provenance(self, force: bool = False):
        started = time.time()
        rows = [
            {
                "symbol": "AAA_USDT",
                "amount24": 1_000_000.0,
                "riseFallRate": 0.2,
                "lastPrice": 1.0,
                "fundingRate": 0.0001,
                "holdVol": 10.0,
            }
        ]
        if self.requests:
            now = time.time()
            return rows, {
                "request_started_at": started,
                "received_at": now,
                "source_ts": self.first_received,
                "cache_hit": True,
                "cache_age_sec": now - self.first_received,
                "status": "ok",
            }
        self.requests += 1
        self.first_received = time.time()
        return rows, {
            "request_started_at": started,
            "received_at": self.first_received,
            "source_ts": self.first_received,
            "cache_hit": False,
            "cache_age_sec": 0.0,
            "status": "ok",
        }


def test_cached_tickers_are_not_relabelled_as_a_fresh_response() -> None:
    client = _CachingClient()
    universe = SymbolUniverse(client, UniverseConfig(min_turnover_24h_usdt=1.0, refresh_sec=0.0))

    first = universe.refresh(force=True)
    time.sleep(0.02)
    second = universe.refresh(force=True)

    assert first.cache_hit is False
    assert second.cache_hit is True
    # The second snapshot arrived later, but the data itself is the older one.
    assert second.received_at > first.received_at
    assert second.source_ts == first.source_ts
    assert second.cache_age_sec > 0.0


def test_the_envelope_does_not_claim_execution_readiness() -> None:
    """Between cycle completion and a delivered alert there is still record
    construction, an fsync, the return path and the channel itself. None of that
    is measured here, so the timing may only claim research ranking."""
    envelope = _envelope()
    assert envelope.timing_basis == TIMING_BASIS_RESEARCH_RANKING
    assert envelope.as_dict()["timing_basis"] == "research_ranking_ready"


def test_contract_details_get_their_own_timing_when_requested() -> None:
    from app.scan import _details_timing

    class _Snapshot:
        details_request_started_at = 100.0
        details_received_at = 100.5
        details_status = "ok"

    timing = _details_timing(_Snapshot())
    assert timing is not None
    assert timing.source == "contract_details"
    assert timing.received_at == 100.5

    class _NoDetails:
        details_request_started_at = None
        details_received_at = None
        details_status = None

    assert _details_timing(_NoDetails()) is None
