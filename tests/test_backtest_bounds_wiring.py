"""The backtest runs, and reports profitability as a range with a verdict.

``run_backtest`` had been broken since ``SignalContext`` gained a required
``sentiment_source`` field, so the whole path was dead. Fixing it is part of
this work: three execution bounds are worth nothing if the backtest that
produces them cannot run.

A historical bar carries no sentiment feed. The honest value is "unavailable",
not a fabricated neutral 50 presented as if it had been measured.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtesting.backtest import BacktestConfig, run_backtest
from core.signal_generator import SignalConfig


def _ohlcv(bars: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    steps = rng.normal(0.0, 0.004, size=bars)
    # A few sharp pumps, which is what this strategy is built to fade.
    for start in (150, 320, 470):
        steps[start : start + 4] += 0.05
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.004, size=bars)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.004, size=bars)))
    frame = pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "volume": rng.lognormal(10.0, 0.5, size=bars),
        },
        index=pd.date_range("2026-01-01", periods=bars, freq="15min", tz="UTC"),
    )
    return frame


class RunsAtAllTests(unittest.TestCase):
    def test_the_backtest_completes(self):
        trades, stats = run_backtest(_ohlcv(), BacktestConfig())

        self.assertIsInstance(trades, pd.DataFrame)
        self.assertIsInstance(stats, dict)


def _permissive() -> SignalConfig:
    """Loosened thresholds so the wiring is exercised, not the strategy edge.

    With shipped defaults the generator fires almost never, which would make
    these assertions skip rather than verify anything.
    """

    return SignalConfig(
        rsi_high=52.0,
        volume_spike_threshold=1.0,
        entry_tolerance_pct=0.08,
        vwap_tolerance_pct=0.08,
        msb_break_buffer_pct=0.0,
        weakness_lookback=2,
        msb_recent_bars=10,
    )


class BoundedReportingTests(unittest.TestCase):
    def setUp(self):
        self.trades, self.stats = run_backtest(_ohlcv(), BacktestConfig(), _permissive())
        self.assertFalse(
            self.trades.empty, "the permissive config must produce trades to verify"
        )

    def test_all_three_bounds_are_reported(self):
        for key in ("pnl_optimistic", "pnl_neutral", "pnl_pessimistic"):
            with self.subTest(key=key):
                self.assertIn(key, self.stats)

    def test_the_bounds_are_ordered(self):
        self.assertGreaterEqual(self.stats["pnl_optimistic"], self.stats["pnl_neutral"])
        self.assertGreaterEqual(self.stats["pnl_neutral"], self.stats["pnl_pessimistic"])

    def test_a_verdict_is_reported(self):
        self.assertIn(
            self.stats["profitability_gate"],
            {"pass", "reject_optimistic_only", "reject_non_positive"},
        )

    def test_the_cost_model_is_not_claimed_to_be_measured(self):
        self.assertFalse(self.stats["cost_model_live_ready"])

    def test_per_trade_bounds_are_recorded(self):
        for column in ("pnl_optimistic", "pnl_neutral", "pnl_pessimistic"):
            with self.subTest(column=column):
                self.assertIn(column, self.trades.columns)

    def test_headline_pnl_is_the_neutral_bound(self):
        self.assertAlmostEqual(
            float(self.trades["pnl"].sum()), float(self.trades["pnl_neutral"].sum())
        )


if __name__ == "__main__":
    unittest.main()
