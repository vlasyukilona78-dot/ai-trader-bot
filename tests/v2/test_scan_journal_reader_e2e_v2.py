"""Offline proof that scanner output survives the real schema-v6 journal path.

The unit boundaries around scanning, journalling and dataset parsing are useful,
but they can all be green while their serialized contracts disagree.  These
tests keep the exchange and delivery layers fake while deliberately using the
real ``scan_once`` producer, ``PopulationJournal`` writer and strict population
readers on one physical JSONL file.
"""

from __future__ import annotations

import json
import time

from ai.reversal.population_dataset import (
    iter_population_cycles,
    iter_population_feature_rows,
)
from app.scan import scan_once
from trading.market_data.universe import SymbolUniverse, UniverseConfig
from trading.metrics.population_journal import CURRENT_WRITE_SCHEMA, PopulationJournal

from v2.test_scan_v2 import _FakeFeed, _FakeStrategy, _Logger, _StaleData, _ohlcv


def _ticker() -> dict[str, object]:
    return {
        "symbol": "AAA_USDT",
        "amount24": 1_000_000.0,
        "riseFallRate": 0.2,
        "lastPrice": 1.0,
        "fundingRate": 0.0001,
        "holdVol": 10.0,
    }


class _TickerSequence:
    """Return fresh rows, their stale cached copy, or a failed first request."""

    def __init__(self, outcomes: list[str]):
        self._outcomes = list(outcomes)
        self._cached_rows: list[dict[str, object]] = []
        self._source_ts: float | None = None

    def fetch_all_tickers_with_provenance(self, force: bool = False):
        del force
        started = time.time()
        outcome = self._outcomes.pop(0)
        received = time.time()

        if outcome == "fresh":
            self._cached_rows = [_ticker()]
            self._source_ts = received
            return list(self._cached_rows), {
                "request_started_at": started,
                "received_at": received,
                "source_ts": received,
                "cache_hit": False,
                "cache_age_sec": 0.0,
                "status": "ok",
                "error_code": None,
            }

        if outcome == "stale_cache":
            assert self._source_ts is not None
            return list(self._cached_rows), {
                "request_started_at": started,
                "received_at": received,
                "source_ts": self._source_ts,
                "cache_hit": True,
                "cache_age_sec": max(0.0, started - self._source_ts),
                "status": "stale_cache",
                "error_code": "SyntheticTickerUnavailable",
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


def _universe(client: _TickerSequence) -> SymbolUniverse:
    return SymbolUniverse(
        client,
        UniverseConfig(min_turnover_24h_usdt=1.0, refresh_sec=0),
    )


def _scan(universe: SymbolUniverse, journal: PopulationJournal) -> None:
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


def test_scan_journal_reader_round_trip_preserves_fresh_stale_and_error_cycles(
    tmp_path,
) -> None:
    assert CURRENT_WRITE_SCHEMA == 6
    journal_path = tmp_path / "population-v6.jsonl"
    journal = PopulationJournal(journal_path)

    reused_universe = _universe(_TickerSequence(["fresh", "stale_cache"]))
    _scan(reused_universe, journal)
    _scan(reused_universe, journal)

    # A separate universe proves that a failed *first* refresh is not laundered
    # through the previous universe's cache and is durable despite having no rows.
    _scan(_universe(_TickerSequence(["error"])), journal)

    raw_records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    assert raw_records
    assert {record["schema_version"] for record in raw_records} == {6}

    cycles = list(iter_population_cycles(journal_path))
    assert len(cycles) == 3

    fresh_envelope, fresh_rows = cycles[0]
    assert fresh_envelope.status == "completed"
    assert fresh_envelope.universe_timing.status == "ok"
    assert fresh_envelope.universe_timing.cache_hit is False
    assert fresh_envelope.universe_timing.cache_age_sec == 0.0
    assert fresh_envelope.universe_timing.source_ts == fresh_envelope.universe_timing.received_at
    assert [row.symbol for row in fresh_rows] == ["AAAUSDT"]
    assert fresh_rows[0].envelope_hash == fresh_envelope.envelope_hash()

    stale_envelope, stale_rows = cycles[1]
    assert stale_envelope.status == "completed"
    assert stale_envelope.universe_timing.status == "stale_cache"
    assert stale_envelope.universe_timing.error_code == "SyntheticTickerUnavailable"
    assert stale_envelope.universe_timing.cache_hit is True
    assert stale_envelope.universe_timing.source_ts == fresh_envelope.universe_timing.source_ts
    assert stale_envelope.universe_timing.cache_age_sec is not None
    assert [row.symbol for row in stale_rows] == ["AAAUSDT"]
    assert stale_rows[0].envelope_hash == stale_envelope.envelope_hash()

    error_envelope, error_rows = cycles[2]
    assert error_envelope.status == "error"
    assert error_envelope.error_code == "SyntheticTickerUnavailable"
    assert error_envelope.universe_timing.status == "error"
    assert error_envelope.universe_timing.cache_hit is False
    assert error_envelope.universe_timing.source_ts is None
    assert error_rows == []

    # The flat reader must expose both completed rows and silently expose no
    # fabricated feature row for the valid zero-row error cycle.
    flat_rows = list(iter_population_feature_rows(journal_path))
    assert [row.snapshot_id for row in flat_rows] == [
        fresh_rows[0].snapshot_id,
        stale_rows[0].snapshot_id,
    ]


def test_stale_base_frame_round_trip_preserves_range_without_decision_bar(
    tmp_path,
) -> None:
    journal_path = tmp_path / "stale-base-v6.jsonl"
    journal = PopulationJournal(journal_path)
    scan_once(
        universe=_universe(_TickerSequence(["fresh"])),
        feed=_FakeFeed(
            {"BTCUSDT": _ohlcv(), "AAAUSDT": _StaleData(_ohlcv())}
        ),
        strategy=_FakeStrategy(),
        logger=_Logger(),
        timeframe="60",
        candles=320,
        workers=1,
        population_journal=journal,
    )

    cycles = list(iter_population_cycles(journal_path))
    assert len(cycles) == 1
    _, rows = cycles[0]
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "data_quality_error"
    assert row.base_source_evidence is not None
    assert row.base_source_evidence["outcome"] == "stale"
    assert row.base_source_evidence["last_bar_close_ts"] is not None
    assert row.base_source_evidence["frame_hash"] is not None
    decision = next(
        json.loads(line)
        for line in journal_path.read_text().splitlines()
        if json.loads(line)["record_type"] == "decision"
    )
    assert decision["base_bar_open_ts"] is None
    assert decision["base_bar_close_ts"] is None
