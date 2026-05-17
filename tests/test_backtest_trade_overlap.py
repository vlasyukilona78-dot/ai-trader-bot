from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from backtesting.backtest import BacktestConfig, PaperTrader, run_backtest


class _AlwaysShortGenerator:
    def __init__(self, *_args, **_kwargs):
        pass

    def generate(self, context):
        price = float(context.df.iloc[-1]["close"])
        return SimpleNamespace(
            side="SHORT",
            entry=price,
            tp=price * 0.99,
            sl=price * 1.01,
            confidence=0.75,
        )


class BacktestTradeOverlapTests(unittest.TestCase):
    def test_run_backtest_skips_until_exit_bar_to_avoid_overlapping_trades(self):
        idx = pd.date_range("2026-01-01", periods=100, freq="min", tz="UTC")
        close = [100.0] * 100
        high = [100.2] * 100
        low = [99.8] * 100
        # First possible signal is at bar 80. TP is touched only at bar 85.
        low[85] = 98.5
        # If overlapping trades were allowed, bars 81-84 would also create trades.
        low[91] = 98.5
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": [1000.0] * 100,
            },
            index=idx,
        )

        with patch("backtesting.backtest.SignalGenerator", _AlwaysShortGenerator):
            trades, _stats = run_backtest(df, BacktestConfig(max_hold_bars=10))

        self.assertGreaterEqual(len(trades), 2)
        first_exit = pd.Timestamp(trades.iloc[0]["exit_time"])
        second_entry = pd.Timestamp(trades.iloc[1]["entry_time"])
        self.assertGreater(second_entry, first_exit)

    def test_paper_trader_uses_quality_guard_before_signal_generation(self):
        idx = pd.date_range("2026-01-01", periods=90, freq="min", tz="UTC")
        close = [100.0] * 90
        bars = [
            {
                "datetime": ts,
                "open": close[i],
                "high": close[i] + 0.2,
                "low": close[i] - 0.2,
                "close": close[i],
                "volume": 0.0 if i >= 78 else 1000.0,
                "symbol": "BADUSDT",
            }
            for i, ts in enumerate(idx)
        ]

        with patch("backtesting.backtest.SignalGenerator", _AlwaysShortGenerator):
            trader = PaperTrader()
            result = None
            for bar in bars:
                result = trader.on_new_bar(bar)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
