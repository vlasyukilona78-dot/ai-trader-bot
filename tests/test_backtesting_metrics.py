from __future__ import annotations

import unittest

import pandas as pd

from backtesting.metrics import build_equity_curve, infer_trade_periods_per_year, summarize_trades


class BacktestingMetricsTests(unittest.TestCase):
    def test_build_equity_curve_prefers_recorded_equity_after(self):
        trades = pd.DataFrame(
            [
                {"exit_time": "2026-01-01T00:10:00Z", "pnl": 15.0, "equity_after": 1015.0},
                {"exit_time": "2026-01-01T00:30:00Z", "pnl": -5.0, "equity_after": 1010.0},
            ]
        )
        equity = build_equity_curve(trades, initial_equity=1000.0)
        self.assertEqual(list(equity.astype(float)), [1015.0, 1010.0])

    def test_infer_trade_periods_per_year_uses_trade_spacing(self):
        exit_times = pd.Series(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T12:00:00Z",
                "2026-01-02T00:00:00Z",
            ]
        )
        annualization = infer_trade_periods_per_year(exit_times)
        self.assertGreater(annualization, 700.0)
        self.assertLess(annualization, 750.0)

    def test_summarize_trades_uses_compounded_equity_path(self):
        trades = pd.DataFrame(
            [
                {
                    "exit_time": "2026-01-01T00:10:00Z",
                    "pnl": 20.0,
                    "ret": 0.02,
                    "equity_before": 1000.0,
                    "equity_after": 1020.0,
                },
                {
                    "exit_time": "2026-01-01T12:10:00Z",
                    "pnl": -10.2,
                    "ret": -0.01,
                    "equity_before": 1020.0,
                    "equity_after": 1009.8,
                },
            ]
        )
        summary = summarize_trades(trades, initial_equity=1000.0)
        self.assertEqual(summary["trades"], 2)
        self.assertAlmostEqual(summary["final_equity"], 1009.8)
        self.assertAlmostEqual(summary["net_pnl"], 9.8)


if __name__ == "__main__":
    unittest.main()
