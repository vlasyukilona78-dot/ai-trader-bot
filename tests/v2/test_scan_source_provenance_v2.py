"""Offline end-to-end checks for scanner source provenance.

These tests deliberately cross the client/universe/scanner boundary.  Unit
tests of each layer missed the production failure where a failed ticker request
became a fabricated successful empty universe at the journal boundary.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.scan import _aggregate_htf_timing, _timed_source, scan_once
from trading.market_data.mexc_client import MexcContractClient
from trading.market_data.universe import SymbolUniverse, UniverseConfig

from v2.test_scan_v2 import (
    _CaptureJournal,
    _FakeFeed,
    _FakeStrategy,
    _Logger,
    _ohlcv,
)


def _ticker() -> dict[str, object]:
    return {
        "symbol": "AAA_USDT",
        "amount24": 1_000_000.0,
        "riseFallRate": 0.2,
        "lastPrice": 1.0,
        "fundingRate": 0.0001,
        "holdVol": 10.0,
    }


class _SequencedTickerClient:
    def __init__(self, outcomes: list[str]):
        self.outcomes = list(outcomes)
        self.source_ts: float | None = None

    def fetch_all_tickers_with_provenance(self, force: bool = False):
        del force
        started = time.time()
        outcome = self.outcomes.pop(0)
        received = time.time()
        if outcome == "fresh":
            self.source_ts = received
            return [_ticker()], {
                "request_started_at": started,
                "received_at": received,
                "source_ts": received,
                "cache_hit": False,
                "cache_age_sec": 0.0,
                "status": "ok",
                "error_code": None,
            }
        if outcome == "stale_cache":
            assert self.source_ts is not None
            return [_ticker()], {
                "request_started_at": started,
                "received_at": received,
                "source_ts": self.source_ts,
                "cache_hit": True,
                "cache_age_sec": max(0.0, started - self.source_ts),
                "status": "stale_cache",
                "error_code": "SyntheticTickerUnavailable",
            }
        if outcome == "empty":
            return [], {
                "request_started_at": started,
                "received_at": received,
                "source_ts": received,
                "cache_hit": False,
                "cache_age_sec": 0.0,
                "status": "ok",
                "error_code": None,
            }
        return [], {
            "request_started_at": started,
            "received_at": received,
            "source_ts": None,
            "cache_hit": False,
            "cache_age_sec": None,
            "status": "error",
            "error_code": "SyntheticTickerUnavailable",
        }


class _TickerAndDetailsSequence(_SequencedTickerClient):
    def fetch_contract_details_with_provenance(self, force: bool = False):
        del force
        started = time.time()
        received = time.time()
        return {
            "AAA_USDT": {
                "symbol": "AAA_USDT",
                "contractSize": 1.0,
                "minVol": 1.0,
                "maxLeverage": 50.0,
            }
        }, {
            "request_started_at": started,
            "received_at": received,
            "source_ts": received,
            "cache_hit": False,
            "cache_age_sec": 0.0,
            "status": "ok",
            "error_code": None,
        }


def _scan(universe: SymbolUniverse, journal: _CaptureJournal) -> None:
    scan_once(
        universe=universe,
        feed=_FakeFeed({"BTCUSDT": _ohlcv(), "AAAUSDT": _ohlcv()}),
        strategy=_FakeStrategy(),
        logger=_Logger(),
        timeframe="60",
        candles=320,
        workers=1,
        population_journal=journal,
    )


def test_fresh_ticker_path_reaches_the_envelope_as_fresh() -> None:
    universe = SymbolUniverse(
        _SequencedTickerClient(["fresh"]),
        UniverseConfig(min_turnover_24h_usdt=1.0, refresh_sec=0),
    )
    journal = _CaptureJournal()

    _scan(universe, journal)

    timing = journal.envelopes[0].universe_timing
    assert journal.envelopes[0].status == "completed"
    assert timing.status == "ok"
    assert timing.cache_hit is False
    assert timing.cache_age_sec == 0.0
    assert timing.source_ts == timing.received_at


def test_failed_refresh_with_prior_rows_keeps_stale_cache_provenance() -> None:
    client = _SequencedTickerClient(["fresh", "stale_cache"])
    universe = SymbolUniverse(
        client,
        UniverseConfig(min_turnover_24h_usdt=1.0, refresh_sec=0),
    )
    first = universe.refresh(force=True)
    journal = _CaptureJournal()

    _scan(universe, journal)

    timing = journal.envelopes[0].universe_timing
    assert journal.envelopes[0].status == "completed"
    assert timing.status == "stale_cache"
    assert timing.error_code == "SyntheticTickerUnavailable"
    assert timing.cache_hit is True
    assert timing.source_ts == first.source_ts
    assert timing.cache_age_sec is not None


def test_stale_ticker_fallback_relabels_reused_contract_details_as_cache() -> None:
    client = _TickerAndDetailsSequence(["fresh", "error"])
    universe = SymbolUniverse(
        client,
        UniverseConfig(
            min_turnover_24h_usdt=1.0,
            max_min_notional_usdt=100.0,
            refresh_sec=0,
        ),
    )
    first = universe.refresh(force=True)
    time.sleep(0.002)
    second = universe.refresh(force=True)

    assert first.details_cache_hit is False
    assert first.details_source_ts is not None
    assert second.source_status == "stale_cache"
    assert second.details_cache_hit is True
    assert second.details_source_ts == first.details_source_ts
    assert second.details_cache_age_sec is not None
    assert second.details_cache_age_sec > 0.0


def test_failed_first_ticker_attempt_is_a_durable_error_cycle() -> None:
    universe = SymbolUniverse(
        _SequencedTickerClient(["error"]),
        UniverseConfig(min_turnover_24h_usdt=1.0, refresh_sec=0),
    )
    journal = _CaptureJournal()

    _scan(universe, journal)

    assert journal.cycles == [[]]
    envelope = journal.envelopes[0]
    assert envelope.status == "error"
    assert envelope.error_code == "SyntheticTickerUnavailable"
    assert envelope.universe_timing.status == "error"
    assert envelope.universe_timing.cache_hit is False
    assert envelope.universe_timing.source_ts is None


def test_refresh_exception_is_journalled_as_a_typed_error_cycle() -> None:
    class _ExplodingUniverse:
        config = UniverseConfig()

        def refresh(self):
            raise TimeoutError("secret response text")

    journal = _CaptureJournal()
    logger = _Logger()

    scan_once(
        universe=_ExplodingUniverse(),
        feed=_FakeFeed({}),
        strategy=_FakeStrategy(),
        logger=logger,
        timeframe="60",
        candles=320,
        workers=1,
        population_journal=journal,
    )

    envelope = journal.envelopes[0]
    assert envelope.status == "error"
    assert envelope.error_code == "TimeoutError"
    assert envelope.universe_timing.error_code == "TimeoutError"
    assert "secret response text" not in str(logger.warnings)


class _FailingJournal:
    enabled = True

    def append_cycle(self, records, *, envelope, benchmark_source_evidence):
        del records, envelope, benchmark_source_evidence
        raise OSError("disk path and private details must not be logged")


def test_refresh_error_stops_when_its_terminal_cycle_cannot_be_persisted() -> None:
    class _ExplodingUniverse:
        config = UniverseConfig()

        def refresh(self):
            raise TimeoutError("secret response text")

    with pytest.raises(OSError, match="disk path"):
        scan_once(
            universe=_ExplodingUniverse(),
            feed=_FakeFeed({}),
            strategy=_FakeStrategy(),
            logger=_Logger(),
            timeframe="60",
            candles=320,
            workers=1,
            population_journal=_FailingJournal(),
        )


def test_empty_universe_stops_when_its_terminal_cycle_cannot_be_persisted() -> None:
    universe = SymbolUniverse(
        _SequencedTickerClient(["empty"]),
        UniverseConfig(min_turnover_24h_usdt=1.0, refresh_sec=0),
    )

    with pytest.raises(OSError, match="disk path"):
        _scan(universe, _FailingJournal())


def test_base_ohlcv_failure_marks_the_cycle_source_as_error() -> None:
    universe = SymbolUniverse(
        _SequencedTickerClient(["fresh"]),
        UniverseConfig(min_turnover_24h_usdt=1.0, refresh_sec=0),
    )
    journal = _CaptureJournal()

    scan_once(
        universe=universe,
        feed=_FakeFeed({"BTCUSDT": _ohlcv(), "AAAUSDT": TimeoutError("private")}),
        strategy=_FakeStrategy(),
        logger=_Logger(),
        timeframe="60",
        candles=320,
        workers=1,
        population_journal=journal,
    )

    base = next(
        timing
        for timing in journal.envelopes[0].source_timings
        if timing.source == "base_ohlcv"
    )
    assert base.status == "error"
    assert base.error_code == "BaseOhlcvUnavailable"
    assert base.source_as_of is None


def test_failed_benchmark_does_not_claim_the_requested_cutoff_as_observed() -> None:
    timing = _timed_source(
        "benchmark",
        request_started_at=100.0,
        received_at=101.0,
        source_as_of=90.0,
        error=TimeoutError("private"),
    )

    assert timing.status == "error"
    assert timing.source_as_of is None


def test_all_failed_htf_reads_do_not_claim_a_market_cutoff() -> None:
    timing = _aggregate_htf_timing(
        [
            {
                "request_started_at": 100.0,
                "received_at": 100.5,
                "source_as_of": 90.0,
                "status": "error",
                "cache_hit": False,
                "source_ts": None,
                "cache_age_sec": None,
                "error_code": "TimeoutError",
            }
        ]
    )

    assert timing is not None
    assert timing.status == "error"
    assert timing.error_code == "HigherTimeframeUnavailable"
    assert timing.source_as_of is None


def test_mixed_fresh_and_cached_htf_reads_remain_visibly_cached() -> None:
    timing = _aggregate_htf_timing(
        [
            {
                "request_started_at": 100.0,
                "received_at": 100.5,
                "source_as_of": 90.0,
                "status": "ok",
                "cache_hit": False,
                "source_ts": 100.5,
                "cache_age_sec": 0.0,
                "error_code": None,
            },
            {
                "request_started_at": 100.1,
                "received_at": 100.6,
                "source_as_of": 90.0,
                "status": "ok",
                "cache_hit": True,
                "source_ts": 80.0,
                "cache_age_sec": 20.1,
                "error_code": None,
            },
        ]
    )

    assert timing is not None
    assert timing.status == "ok"
    assert timing.cache_hit is True
    assert timing.source_ts == 80.0
    assert timing.cache_age_sec == 20.0


def test_contract_details_distinguish_fresh_cache_stale_and_first_failure() -> None:
    client = MexcContractClient(max_retries=1)
    payload = {
        "data": [
            {
                "symbol": "AAA_USDT",
                "contractSize": 1,
                "minVol": 1,
                "maxLeverage": 50,
            }
        ]
    }
    with patch.object(client, "_request_public", return_value=payload):
        fresh_rows, fresh = client.fetch_contract_details_with_provenance(force=True)
    cached_rows, cached = client.fetch_contract_details_with_provenance()
    with patch.object(client, "_request_public", return_value=None):
        stale_rows, stale = client.fetch_contract_details_with_provenance(force=True)

    assert fresh_rows == cached_rows == stale_rows
    assert fresh["status"] == "ok" and fresh["cache_hit"] is False
    assert cached["status"] == "ok" and cached["cache_hit"] is True
    assert stale["status"] == "stale_cache" and stale["cache_hit"] is True
    assert stale["error_code"] == "MexcContractDetailsUnavailable"
    assert stale["source_ts"] == fresh["source_ts"]

    empty_client = MexcContractClient(max_retries=1)
    with patch.object(empty_client, "_request_public", return_value=None):
        rows, failed = empty_client.fetch_contract_details_with_provenance(force=True)
    assert rows == {}
    assert failed["status"] == "error"
    assert failed["cache_hit"] is False
    assert failed["source_ts"] is None
