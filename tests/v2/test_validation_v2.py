from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtesting.validation import (
    PortfolioConfig,
    clustered_bootstrap_ci,
    simulate_portfolio,
    walk_forward_folds,
)


class WalkForwardV2Tests(unittest.TestCase):
    def test_every_test_block_lies_after_its_training_data(self):
        for train, test in walk_forward_folds(1000, n_folds=4):
            self.assertLess(train.max(), test.min())

    def test_training_window_expands(self):
        folds = walk_forward_folds(1000, n_folds=4)
        sizes = [len(tr) for tr, _ in folds]
        self.assertEqual(sizes, sorted(sizes))

    def test_test_blocks_do_not_overlap_and_cover_the_tail(self):
        folds = walk_forward_folds(1000, n_folds=4)
        blocks = [te for _, te in folds]
        for a, b in zip(blocks, blocks[1:]):
            self.assertEqual(a.max() + 1, b.min())
        self.assertEqual(blocks[-1].max(), 999)

    def test_too_little_data_yields_no_folds(self):
        self.assertEqual(walk_forward_folds(5), [])


class ClusteredBootstrapV2Tests(unittest.TestCase):
    def test_clustering_widens_the_interval_versus_ignoring_it(self):
        """Forty events on one coin are not forty independent observations, and
        pretending otherwise reports an interval that is far too narrow."""
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(10), 40)
        # strong per-symbol effect: within a symbol values barely vary
        values = np.repeat(rng.normal(0, 1.0, 10), 40) + rng.normal(0, 0.01, 400)

        lo_c, hi_c = clustered_bootstrap_ci(values, groups, n_boot=2000)
        lo_n, hi_n = clustered_bootstrap_ci(values, np.arange(len(values)), n_boot=2000)
        self.assertGreater(hi_c - lo_c, (hi_n - lo_n) * 2)

    def test_interval_brackets_the_mean(self):
        rng = np.random.default_rng(1)
        groups = np.repeat(np.arange(20), 5)
        values = rng.normal(0.5, 0.1, 100)
        lo, hi = clustered_bootstrap_ci(values, groups, n_boot=2000)
        self.assertLess(lo, values.mean())
        self.assertGreater(hi, values.mean())

    def test_single_group_cannot_be_bootstrapped(self):
        lo, hi = clustered_bootstrap_ci(np.ones(10), np.zeros(10))
        self.assertTrue(np.isnan(lo) and np.isnan(hi))


def _trades(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["entry_ts", "exit_ts", "pnl_on_initial", "max_deployed"])


class PortfolioV2Tests(unittest.TestCase):
    def test_signals_arriving_with_a_full_book_are_skipped(self):
        overlapping = _trades([(i * 10, 10_000, 0.02, 1.0) for i in range(10)])
        r = simulate_portfolio(overlapping, PortfolioConfig(max_concurrent=3))
        self.assertEqual(r.taken, 3)
        self.assertEqual(r.skipped_capacity, 7)
        self.assertLessEqual(r.peak_concurrent, 3)

    def test_sequential_trades_all_fit(self):
        sequential = _trades([(i * 100, i * 100 + 50, 0.02, 1.0) for i in range(10)])
        r = simulate_portfolio(sequential, PortfolioConfig(max_concurrent=3))
        self.assertEqual(r.taken, 10)
        self.assertEqual(r.skipped_capacity, 0)

    def test_capital_limit_blocks_deep_averaging(self):
        heavy = _trades([(i * 10, 10_000, 0.02, 6.0) for i in range(10)])
        r = simulate_portfolio(heavy, PortfolioConfig(capital=200.0, leg_notional=20.0,
                                                     max_concurrent=99))
        self.assertLess(r.taken, 10)
        self.assertLessEqual(r.peak_notional, 200.0)

    def test_equity_and_drawdown_are_tracked(self):
        mixed = _trades([(0, 10, 0.05, 1.0), (20, 30, -0.50, 1.0), (40, 50, 0.05, 1.0)])
        r = simulate_portfolio(mixed, PortfolioConfig(capital=1000.0, leg_notional=100.0))
        self.assertEqual(r.taken, 3)
        self.assertAlmostEqual(r.final_equity, 1000.0 + (0.05 - 0.50 + 0.05) * 100.0, places=6)
        self.assertGreater(r.max_drawdown, 0.0)

    def test_empty_input_is_handled(self):
        r = simulate_portfolio(pd.DataFrame())
        self.assertEqual(r.taken, 0)
        self.assertEqual(r.total_return, 0.0)


if __name__ == "__main__":
    unittest.main()
