from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.signal_generator import SignalConfig, SignalGenerator
from trading.market_data.bar_contract import interval_seconds
from trading.market_data.mexc_client import MexcOhlcvRequestError
from trading.market_data.timeframe_cache import HigherTimeframeCache, TimeframeCacheConfig


@dataclass
class _Frame:
    ohlcv: pd.DataFrame


class _FakeFeed:
    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail
        self.as_of = []

    def fetch_closed_frame(self, symbol: str, timeframe: str, candles: int, *, as_of):
        self.calls += 1
        self.as_of.append(as_of)
        if self.fail:
            raise MexcOhlcvRequestError("network down")
        boundary = pd.Timestamp(float(as_of), unit="s", tz="UTC")
        delta = pd.Timedelta(seconds=interval_seconds(timeframe))
        index = pd.DatetimeIndex([boundary - (3 * delta), boundary - (2 * delta), boundary - delta])
        return _Frame(
            pd.DataFrame(
                {
                    "open": [1.0, 2.0, 3.0],
                    "high": [1.1, 2.1, 3.1],
                    "low": [0.9, 1.9, 2.9],
                    "close": [1.0, 2.0, 3.0],
                    "volume": [10.0, 20.0, 30.0],
                    "symbol": [symbol] * 3,
                },
                index=index,
            )
        )


class _LegacyFeed:
    def __init__(self):
        self.limits = []

    def fetch_frame(self, symbol: str, timeframe: str, candles: int):
        self.limits.append(candles)
        index = pd.date_range("2026-01-01T10:00:00Z", periods=4, freq="h")
        close = [1.0, 2.0, 3.0, 4.0]
        return _Frame(
            pd.DataFrame(
                {
                    "open": close,
                    "high": [value + 0.1 for value in close],
                    "low": [value - 0.1 for value in close],
                    "close": close,
                    "volume": [10.0] * 4,
                },
                index=index,
            )
        )


class _StaleFeed(_FakeFeed):
    def fetch_closed_frame(self, symbol: str, timeframe: str, candles: int, *, as_of):
        self.calls += 1
        self.as_of.append(as_of)
        boundary = pd.Timestamp(float(as_of), unit="s", tz="UTC")
        delta = pd.Timedelta(seconds=interval_seconds(timeframe))
        index = pd.DatetimeIndex(
            [boundary - (4 * delta), boundary - (3 * delta), boundary - (2 * delta)]
        )
        close = [1.0, 2.0, 3.0]
        return _Frame(
            pd.DataFrame(
                {
                    "open": close,
                    "high": [value + 0.1 for value in close],
                    "low": [value - 0.1 for value in close],
                    "close": close,
                    "volume": [10.0] * 3,
                },
                index=index,
            )
        )


class _EmptyFeed:
    def fetch_closed_frame(self, symbol: str, timeframe: str, candles: int, *, as_of):
        frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame.index = pd.DatetimeIndex([], tz="UTC")
        return _Frame(frame)


class _BuggyFeed:
    def fetch_closed_frame(self, symbol: str, timeframe: str, candles: int, *, as_of):
        raise RuntimeError("programming bug")


class _BlockingFeed(_FakeFeed):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def fetch_closed_frame(self, symbol: str, timeframe: str, candles: int, *, as_of):
        self.entered.set()
        if not self.release.wait(timeout=5.0):
            raise AssertionError("test did not release blocked feed")
        return super().fetch_closed_frame(symbol, timeframe, candles, as_of=as_of)


class HigherTimeframeCacheV2Tests(unittest.TestCase):
    _BAR_1230 = pd.Timestamp("2026-01-01T12:30:00Z")
    _BAR_1255 = pd.Timestamp("2026-01-01T12:55:00Z")
    _BAR_1300 = pd.Timestamp("2026-01-01T13:00:00Z")

    @staticmethod
    def _config(**kwargs):
        return TimeframeCacheConfig(interval="Min60", **kwargs)

    def test_second_read_in_same_boundary_inside_ttl_does_not_refetch(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=600))
        cache.get("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        cache.get("BTCUSDT", as_of=self._BAR_1255, now=1100.0)
        self.assertEqual(feed.calls, 1)

    def test_new_boundary_refetches_even_inside_ttl(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=600))
        cache.get("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        cache.get("BTCUSDT", as_of=self._BAR_1300, now=1001.0)
        self.assertEqual(feed.calls, 2)

    def test_same_boundary_refetches_after_ttl(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=600))
        cache.get("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        cache.get("BTCUSDT", as_of=self._BAR_1255, now=2000.0)
        self.assertEqual(feed.calls, 2)

    def test_symbols_are_cached_independently(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=600))
        a = cache.get("AUSDT", as_of=self._BAR_1230, now=1000.0)
        b = cache.get("BUSDT", as_of=self._BAR_1230, now=1000.0)
        self.assertEqual(a["symbol"].iloc[0], "AUSDT")
        self.assertEqual(b["symbol"].iloc[0], "BUSDT")
        self.assertEqual(cache.cached_symbols, 2)

    def test_failed_refresh_can_serve_same_boundary_frame(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=1))
        cache.get("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        fresh = cache.drain_timings()[0]
        feed.fail = True
        stale = cache.get("BTCUSDT", as_of=self._BAR_1255, now=5000.0)
        self.assertIsNotNone(stale)
        fallback = cache.drain_timings()[0]
        self.assertEqual(fallback["status"], "stale_cache")
        self.assertTrue(fallback["cache_hit"])
        self.assertEqual(fallback["source_ts"], fresh["source_ts"])
        self.assertIsNotNone(fallback["cache_age_sec"])
        self.assertEqual(fallback["error_code"], "MexcOhlcvRequestError")

    def test_failed_fetch_at_new_boundary_does_not_serve_previous_frame(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=600))
        cache.get("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        feed.fail = True
        self.assertIsNone(cache.get("BTCUSDT", as_of=self._BAR_1300, now=1001.0))

    def test_failure_with_no_cache_returns_none(self):
        cache = HigherTimeframeCache(_FakeFeed(fail=True), self._config())
        self.assertIsNone(cache.get("BTCUSDT", as_of=self._BAR_1230, now=1000.0))

    def test_eviction_keeps_the_cache_bounded(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=600, max_symbols=3))
        for i in range(6):
            cache.get(f"S{i}", as_of=self._BAR_1230, now=1000.0 + i)
        self.assertLessEqual(cache.cached_symbols, 3)

    def test_callers_cannot_mutate_the_cached_frame(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=600))
        first = cache.get("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        first.iloc[0, first.columns.get_loc("close")] = 999.0
        second = cache.get("BTCUSDT", as_of=self._BAR_1255, now=1001.0)
        self.assertEqual(second["close"].iloc[0], 1.0)

    def test_same_boundary_cache_hit_keeps_original_source_timestamp_and_age(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=600))
        cache.get("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        fresh = cache.drain_timings()[0]
        cache.get("BTCUSDT", as_of=self._BAR_1255, now=1001.0)
        hit = cache.drain_timings()[0]

        self.assertEqual(fresh["status"], "ok")
        self.assertFalse(fresh["cache_hit"])
        self.assertIsNotNone(fresh["source_ts"])
        self.assertEqual(hit["status"], "ok")
        self.assertTrue(hit["cache_hit"])
        self.assertEqual(hit["source_ts"], fresh["source_ts"])
        self.assertIsNotNone(hit["cache_age_sec"])

    def test_legacy_feed_fetches_extra_bar_and_is_filtered(self):
        feed = _LegacyFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=600, candles=3))
        frame = cache.get("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        self.assertEqual(feed.limits, [4])
        self.assertEqual(list(frame.index.hour), [10, 11])

    def test_get_with_provenance_returns_exact_symbol_timeframe_and_frame_hash(self):
        cache = HigherTimeframeCache(_FakeFeed(), self._config(ttl_sec=600))
        read = cache.get_with_provenance("BTCUSDT", as_of=self._BAR_1230, now=1000.0)

        self.assertIsNotNone(read.frame)
        self.assertEqual(read.evidence.symbol, "BTCUSDT")
        self.assertEqual(read.evidence.timeframe, "Min60")
        self.assertEqual(read.evidence.outcome, "fresh")
        self.assertRegex(read.evidence.frame_hash or "", r"^[0-9a-f]{64}$")

    def test_cache_hit_keeps_content_hash_but_records_new_call_timing(self):
        cache = HigherTimeframeCache(_FakeFeed(), self._config(ttl_sec=600))
        first = cache.get_with_provenance("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        second = cache.get_with_provenance("BTCUSDT", as_of=self._BAR_1255, now=1001.0)

        self.assertEqual(first.evidence.frame_hash, second.evidence.frame_hash)
        self.assertFalse(first.evidence.cache_hit)
        self.assertTrue(second.evidence.cache_hit)
        self.assertEqual(first.evidence.source_ts, second.evidence.source_ts)
        self.assertNotEqual(
            first.evidence.requested_as_of_ts,
            second.evidence.requested_as_of_ts,
        )

    def test_successful_but_lagging_frame_reports_actual_data_through(self):
        cache = HigherTimeframeCache(_StaleFeed(), self._config(ttl_sec=600))
        read = cache.get_with_provenance("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        timing = cache.drain_timings()[0]

        self.assertIsNotNone(read.frame)  # behavior is deliberately preserved
        self.assertEqual(read.evidence.outcome, "stale")
        self.assertLess(
            read.evidence.data_through_ts,
            read.evidence.expected_closed_boundary_ts,
        )
        self.assertEqual(timing["source_as_of"], read.evidence.data_through_ts)
        self.assertEqual(timing["symbol"], "BTCUSDT")
        self.assertEqual(timing["timeframe"], "Min60")

    def test_mutating_config_interval_cannot_reuse_other_timeframe_cache(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=600))
        first = cache.get_with_provenance("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        cache.config.interval = "Hour4"
        second = cache.get_with_provenance("BTCUSDT", as_of=self._BAR_1230, now=1001.0)

        self.assertEqual(feed.calls, 2)
        self.assertEqual(first.evidence.timeframe, "Min60")
        self.assertEqual(second.evidence.timeframe, "Hour4")
        self.assertEqual(cache.cached_symbols, 2)

    def test_mutating_requested_window_cannot_reuse_shorter_cached_frame(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(
            feed, self._config(ttl_sec=600, candles=2)
        )
        cache.get_with_provenance("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        cache.config.candles = 3
        cache.get_with_provenance("BTCUSDT", as_of=self._BAR_1230, now=1001.0)

        self.assertEqual(feed.calls, 2)
        self.assertEqual(cache.cached_symbols, 2)

    def test_request_failure_and_true_empty_are_distinct_evidence(self):
        failed = HigherTimeframeCache(_FakeFeed(fail=True), self._config()).get_with_provenance(
            "BTCUSDT", as_of=self._BAR_1230, now=1000.0
        )
        empty = HigherTimeframeCache(_EmptyFeed(), self._config()).get_with_provenance(
            "BTCUSDT", as_of=self._BAR_1230, now=1000.0
        )

        self.assertIsNone(failed.frame)
        self.assertEqual(failed.evidence.outcome, "request_failed")
        self.assertEqual(failed.evidence.error_code, "MexcOhlcvRequestError")
        self.assertIsNotNone(empty.frame)
        self.assertTrue(empty.frame.empty)
        self.assertEqual(empty.evidence.outcome, "no_rows")
        self.assertIsNone(empty.evidence.error_code)

    def test_programming_exception_is_not_disguised_as_request_failure(self):
        cache = HigherTimeframeCache(_BuggyFeed(), self._config())
        with self.assertRaisesRegex(RuntimeError, "programming bug"):
            cache.get_with_provenance(
                "BTCUSDT", as_of=self._BAR_1230, now=1000.0
            )

    def test_same_key_concurrent_miss_is_single_flight(self):
        feed = _BlockingFeed()
        cache = HigherTimeframeCache(feed, self._config(ttl_sec=600))
        start = threading.Barrier(3)

        def read_once():
            start.wait(timeout=5.0)
            return cache.get_with_provenance(
                "BTCUSDT", as_of=self._BAR_1230, now=1000.0
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(read_once)
            second = pool.submit(read_once)
            start.wait(timeout=5.0)
            self.assertTrue(feed.entered.wait(timeout=5.0))
            feed.release.set()
            reads = [first.result(timeout=5.0), second.result(timeout=5.0)]

        self.assertEqual(feed.calls, 1)
        self.assertEqual(reads[0].evidence.frame_hash, reads[1].evidence.frame_hash)
        self.assertEqual(sum(read.evidence.cache_hit for read in reads), 1)

    def test_later_stale_refresh_cannot_downgrade_same_boundary_fresh_cache(self):
        fresh_feed = _FakeFeed()
        cache = HigherTimeframeCache(fresh_feed, self._config(ttl_sec=1))
        first = cache.get_with_provenance(
            "BTCUSDT", as_of=self._BAR_1230, now=1000.0
        )
        cache.feed = _StaleFeed()
        second = cache.get_with_provenance(
            "BTCUSDT", as_of=self._BAR_1255, now=2000.0
        )

        self.assertEqual(first.evidence.frame_hash, second.evidence.frame_hash)
        self.assertEqual(second.evidence.outcome, "stale")
        self.assertTrue(second.evidence.cache_hit)
        self.assertEqual(second.evidence.error_code, "HigherTimeframeDataLag")
        self.assertEqual(
            second.evidence.data_through_ts,
            second.evidence.expected_closed_boundary_ts,
        )

    def test_legacy_get_returns_stale_rows_but_timing_exposes_staleness(self):
        cache = HigherTimeframeCache(_StaleFeed(), self._config(ttl_sec=600))
        frame = cache.get("BTCUSDT", as_of=self._BAR_1230, now=1000.0)
        timing = cache.drain_timings()[0]

        self.assertIsNotNone(frame)
        self.assertFalse(frame.empty)
        self.assertEqual(timing["outcome"], "stale")
        self.assertLess(timing["data_through_ts"], timing["expected_closed_boundary_ts"])


class HigherTimeframeGateV2Tests(unittest.TestCase):
    """4h RSI separated outcomes where the 1h reading did not, so it is read on
    its own timeframe rather than resampled from the entry frame."""

    def _df(self, closes):
        n = len(closes)
        df = pd.DataFrame({"open": closes, "high": [c * 1.002 for c in closes],
                           "low": [c * 0.998 for c in closes], "close": closes,
                           "volume": [1000.0] * n})
        df["atr"] = 1.0
        return df

    def test_weak_higher_timeframe_blocks_the_signal(self):
        gen = SignalGenerator(SignalConfig(min_relative_strength=0.0, min_rsi_4h=61.6))
        falling_htf = self._df(list(np.linspace(200, 100, 60)))  # RSI far below the floor
        ok, d = gen._layer1c_market_context(self._df([100.0] * 40), None, falling_htf)
        self.assertFalse(ok)
        self.assertEqual(d["rsi_htf_ok"], 0.0)

    def test_strong_higher_timeframe_allows_it(self):
        gen = SignalGenerator(SignalConfig(min_relative_strength=0.0, min_rsi_4h=61.6))
        rising_htf = self._df(list(np.linspace(100, 200, 60)))
        ok, d = gen._layer1c_market_context(self._df([100.0] * 40), None, rising_htf)
        self.assertTrue(ok)

    def test_missing_higher_timeframe_blocks_by_default(self):
        """Fail closed. A misconfigured interval or a failed fetch yields an empty
        frame, and passing on that silently disables a gate adopted precisely
        because it cut the tail - the bot would keep trading while believing it
        was protected."""
        gen = SignalGenerator(SignalConfig(min_relative_strength=0.0, min_rsi_4h=61.6))
        ok, d = gen._layer1c_market_context(self._df([100.0] * 40), None, None)
        self.assertFalse(ok)
        self.assertEqual(d["htf_available"], 0.0)

    def test_missing_higher_timeframe_can_be_allowed_explicitly(self):
        gen = SignalGenerator(SignalConfig(min_relative_strength=0.0, min_rsi_4h=61.6,
                                           require_htf=False))
        ok, _ = gen._layer1c_market_context(self._df([100.0] * 40), None, None)
        self.assertTrue(ok)

    def test_default_is_to_require_the_higher_timeframe(self):
        self.assertTrue(SignalConfig().require_htf)

    def test_default_threshold_matches_the_measured_value(self):
        self.assertEqual(SignalConfig().min_rsi_4h, 61.6)
        self.assertEqual(SignalConfig().require_confluence, 0)


if __name__ == "__main__":
    unittest.main()
