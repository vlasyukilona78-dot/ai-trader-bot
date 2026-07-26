from __future__ import annotations

import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from app.scan import describe, scan_once
from trading.market_data.mexc_client import _RateLimiter
from trading.signals.signal_types import IntentAction, StrategyIntent


class RateLimiterV2Tests(unittest.TestCase):
    """Concurrency without pacing silently lost symbols: at 8 workers MEXC
    dropped 13 of 60 requests and the client returned empty frames that are
    indistinguishable from 'no data'."""

    def test_requests_are_paced_to_the_configured_rate(self):
        limiter = _RateLimiter(rate_per_sec=20.0)
        start = time.monotonic()
        for _ in range(30):
            limiter.acquire()
        elapsed = time.monotonic() - start
        # 30 tokens at 20/s with a full bucket: at least the overflow must wait
        self.assertGreater(elapsed, 0.3)

    def test_limiter_is_thread_safe(self):
        limiter = _RateLimiter(rate_per_sec=50.0)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: limiter.acquire(), range(40)))
        self.assertTrue(True)  # no deadlock or exception


class _FakeFrame:
    def __init__(self, ohlcv, mark_price=1.0):
        self.ohlcv = ohlcv
        self.mark_price = mark_price


class _FakeFeed:
    def __init__(self, frames: dict):
        self.frames = frames

    def fetch_frame(self, symbol, timeframe, candles):
        got = self.frames.get(symbol)
        if isinstance(got, Exception):
            raise got
        return _FakeFrame(got if got is not None else pd.DataFrame())


class _FakeUniverse:
    def __init__(self, symbols):
        self._symbols = symbols

    def refresh(self):
        class S:
            symbols = self._symbols
        return S()


class _FakeStrategy:
    def __init__(self, action=IntentAction.HOLD):
        self.action = action
        self.benchmark = "unset"

    def set_benchmark(self, frame):
        self.benchmark = frame

    def generate(self, ctx):
        return StrategyIntent(symbol=ctx.symbol, action=self.action, reason="test")


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def warning(self, msg, *a, **k):
        self.warnings.append(msg)


def _ohlcv(n=100):
    return pd.DataFrame({
        "open": [1.0] * n, "high": [1.01] * n, "low": [0.99] * n,
        "close": [1.0] * n, "volume": [100.0] * n,
    })


class ScanOnceV2Tests(unittest.TestCase):
    def _run(self, frames, symbols, action=IntentAction.HOLD):
        logger = _Logger()
        strategy = _FakeStrategy(action)
        signals = scan_once(universe=_FakeUniverse(symbols), feed=_FakeFeed(frames),
                            strategy=strategy, logger=logger, timeframe="60",
                            candles=320, workers=2)
        return signals, logger, strategy

    def test_returns_only_entry_intents(self):
        frames = {s: _ohlcv() for s in ["BTCUSDT", "AUSDT", "BUSDT"]}
        signals, _, _ = self._run(frames, ["AUSDT", "BUSDT"], IntentAction.SHORT_ENTRY)
        self.assertEqual(len(signals), 2)
        holds, _, _ = self._run(frames, ["AUSDT", "BUSDT"], IntentAction.HOLD)
        self.assertEqual(holds, [])

    def test_short_history_is_skipped_not_evaluated(self):
        frames = {"BTCUSDT": _ohlcv(), "AUSDT": _ohlcv(10)}
        signals, logger, _ = self._run(frames, ["AUSDT"], IntentAction.SHORT_ENTRY)
        self.assertEqual(signals, [])
        self.assertTrue(logger.warnings)  # coverage warning fired

    def test_low_coverage_raises_a_warning(self):
        frames = {"BTCUSDT": _ohlcv(), "AUSDT": _ohlcv()}
        frames["BUSDT"] = pd.DataFrame()
        _, logger, _ = self._run(frames, ["AUSDT", "BUSDT"], IntentAction.SHORT_ENTRY)
        self.assertTrue(any("coverage_low" in str(w) for w in logger.warnings))

    def test_benchmark_failure_does_not_stop_the_scan(self):
        frames = {"BTCUSDT": RuntimeError("down"), "AUSDT": _ohlcv()}
        signals, logger, strategy = self._run(frames, ["AUSDT"], IntentAction.SHORT_ENTRY)
        self.assertEqual(len(signals), 1)
        self.assertIsNone(strategy.benchmark)

    def test_empty_universe_is_reported(self):
        _, logger, _ = self._run({}, [])
        self.assertTrue(any("empty_universe" in str(w) for w in logger.warnings))


class DescribeV2Tests(unittest.TestCase):
    def test_includes_margin_terms_and_safe_leverage(self):
        intent = StrategyIntent(
            symbol="XUSDT", action=IntentAction.SHORT_ENTRY, reason="t",
            metadata={"layer_trace": {"layers": {"layer5_tp_sl": {"details": {
                "entry": 1.0, "sl": 1.02, "tp": 0.97,
                "stop_pct_of_margin": 200.0, "target_pct_of_margin": 300.0,
                "max_safe_leverage": 50.0,
            }}}}},
        )
        text = describe("XUSDT", intent)
        self.assertIn("SHORT_ENTRY XUSDT", text)
        self.assertIn("% of margin", text)
        self.assertIn("max safe leverage 50x", text)

    def test_missing_metadata_does_not_raise(self):
        intent = StrategyIntent(symbol="XUSDT", action=IntentAction.SHORT_ENTRY, reason="t")
        self.assertIn("XUSDT", describe("XUSDT", intent))


if __name__ == "__main__":
    unittest.main()
