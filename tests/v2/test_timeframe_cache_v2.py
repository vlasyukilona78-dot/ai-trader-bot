from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.signal_generator import SignalConfig, SignalGenerator
from trading.market_data.timeframe_cache import HigherTimeframeCache, TimeframeCacheConfig


@dataclass
class _Frame:
    ohlcv: pd.DataFrame


class _FakeFeed:
    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def fetch_frame(self, symbol: str, timeframe: str, candles: int):
        self.calls += 1
        if self.fail:
            raise RuntimeError("network down")
        return _Frame(pd.DataFrame({"close": [1.0, 2.0, 3.0], "symbol": [symbol] * 3}))


class HigherTimeframeCacheV2Tests(unittest.TestCase):
    def test_second_read_inside_ttl_does_not_refetch(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, TimeframeCacheConfig(ttl_sec=600))
        cache.get("BTCUSDT", now=1000.0)
        cache.get("BTCUSDT", now=1100.0)
        self.assertEqual(feed.calls, 1)

    def test_refetches_after_ttl(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, TimeframeCacheConfig(ttl_sec=600))
        cache.get("BTCUSDT", now=1000.0)
        cache.get("BTCUSDT", now=2000.0)
        self.assertEqual(feed.calls, 2)

    def test_symbols_are_cached_independently(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, TimeframeCacheConfig(ttl_sec=600))
        a = cache.get("AUSDT", now=1000.0)
        b = cache.get("BUSDT", now=1000.0)
        self.assertEqual(a["symbol"].iloc[0], "AUSDT")
        self.assertEqual(b["symbol"].iloc[0], "BUSDT")
        self.assertEqual(cache.cached_symbols, 2)

    def test_failed_fetch_serves_the_previous_frame(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, TimeframeCacheConfig(ttl_sec=1))
        cache.get("BTCUSDT", now=1000.0)
        feed.fail = True
        stale = cache.get("BTCUSDT", now=5000.0)
        self.assertIsNotNone(stale)

    def test_failure_with_no_cache_returns_none(self):
        cache = HigherTimeframeCache(_FakeFeed(fail=True), TimeframeCacheConfig())
        self.assertIsNone(cache.get("BTCUSDT", now=1000.0))

    def test_eviction_keeps_the_cache_bounded(self):
        feed = _FakeFeed()
        cache = HigherTimeframeCache(feed, TimeframeCacheConfig(ttl_sec=600, max_symbols=3))
        for i in range(6):
            cache.get(f"S{i}", now=1000.0 + i)
        self.assertLessEqual(cache.cached_symbols, 3)


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

    def test_missing_higher_timeframe_does_not_block(self):
        gen = SignalGenerator(SignalConfig(min_relative_strength=0.0, min_rsi_4h=61.6))
        ok, _ = gen._layer1c_market_context(self._df([100.0] * 40), None, None)
        self.assertTrue(ok)

    def test_default_threshold_matches_the_measured_value(self):
        self.assertEqual(SignalConfig().min_rsi_4h, 61.6)
        self.assertEqual(SignalConfig().require_confluence, 0)


if __name__ == "__main__":
    unittest.main()
