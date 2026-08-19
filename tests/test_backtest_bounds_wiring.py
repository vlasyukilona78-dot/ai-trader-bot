"""The backtest reports cost as three bounds and decides on the middle one.

These tests verify the wiring, not the strategy. The live generator's regime
filter needs multi-timeframe context that synthetic bars do not carry, so
driving it here would test the fixture rather than the bounds. A stub signal
makes the cost path deterministic; the bound arithmetic itself is covered
independently in tests/test_execution_bounds.py.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtesting.backtest import BacktestConfig, run_backtest
from core.signal_generator import SignalGenerator, SignalResult


def _ohlcv(bars: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    steps = rng.normal(0.0, 0.004, size=bars)
    for start in (150, 250, 330):
        steps[start : start + 4] += 0.05
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.004, size=bars)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.004, size=bars)))
    return pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "volume": rng.lognormal(10.0, 0.5, size=bars),
        },
        index=pd.date_range("2026-01-01", periods=bars, freq="15min", tz="UTC"),
    )


def _stub_signal(self, context):
    """Fire a short every 40th bar, with levels derived from the bar itself."""

    if len(context.df) % 40 != 0:
        return None
    entry = float(context.df.iloc[-1]["close"])
    return SignalResult(
        signal_id=f"stub-{len(context.df)}",
        symbol=context.symbol,
        side="SHORT",
        entry=entry,
        sl=entry * 1.02,
        tp=entry * 0.97,
        confidence=0.8,
    )


class RunsAtAllTests(unittest.TestCase):
    def test_the_backtest_completes(self):
        trades, stats = run_backtest(_ohlcv(), BacktestConfig())

        self.assertIsInstance(trades, pd.DataFrame)
        self.assertIsInstance(stats, dict)


class BoundedReportingTests(unittest.TestCase):
    def setUp(self):
        with patch.object(SignalGenerator, "generate", _stub_signal):
            self.trades, self.stats = run_backtest(_ohlcv(), BacktestConfig())
        self.assertFalse(self.trades.empty, "the stub must produce trades to verify")

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

    def test_costs_actually_differ_between_bounds(self):
        # Guard against a vacuous pass: if every bound produced the same number
        # the ordering assertions above would hold while proving nothing.
        self.assertNotAlmostEqual(
            self.stats["pnl_optimistic"], self.stats["pnl_pessimistic"]
        )


if __name__ == "__main__":
    unittest.main()
