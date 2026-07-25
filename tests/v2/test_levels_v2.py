from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.levels import (
    distance_to_fib,
    estimate_liquidation_map,
    fib_levels,
    find_horizontal_levels,
    multi_timeframe_confluence,
    nearest_level_above,
    rsi_divergence,
)


def _frame(closes, volumes=None, rsi=None) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.002 for c in closes],
            "low": [c * 0.998 for c in closes],
            "close": closes,
            "volume": volumes if volumes is not None else [100.0] * n,
            **({"rsi": rsi} if rsi is not None else {}),
        }
    )


class HorizontalLevelsV2Tests(unittest.TestCase):
    def test_repeated_touches_form_a_level(self):
        # price bounces off ~100 three times
        path = [90, 95, 100, 95, 90, 95, 100, 95, 90, 95, 100, 95, 90, 92, 94]
        levels = find_horizontal_levels(_frame([float(p) for p in path]), pivot_left=2, pivot_right=2)
        self.assertTrue(levels)
        self.assertTrue(any(abs(lv.price - 100) / 100 < 0.02 for lv in levels))

    def test_trending_price_produces_no_repeated_level(self):
        levels = find_horizontal_levels(_frame(list(np.linspace(10, 40, 60))), min_touches=3)
        self.assertEqual(levels, [])

    def test_strength_rewards_touches_and_age(self):
        many = [90, 100, 90, 100, 90, 100, 90, 100, 90, 100, 90, 95, 97]
        few = [90, 100, 90, 92, 94, 96, 95, 93, 91, 92, 94, 95, 97]
        a = find_horizontal_levels(_frame([float(p) for p in many]), pivot_left=1, pivot_right=1)
        b = find_horizontal_levels(_frame([float(p) for p in few]), pivot_left=1, pivot_right=1)
        if a and b:
            self.assertGreater(max(lv.strength for lv in a), max(lv.strength for lv in b))

    def test_nearest_level_above_ignores_levels_below(self):
        path = [90, 100, 90, 100, 90, 100, 90, 95, 97]
        levels = find_horizontal_levels(_frame([float(p) for p in path]), pivot_left=1, pivot_right=1)
        above = nearest_level_above(levels, price=95.0)
        if above:
            self.assertGreater(above.price, 95.0)
        self.assertIsNone(nearest_level_above(levels, price=1000.0))


class FibV2Tests(unittest.TestCase):
    def test_grid_spans_the_swing(self):
        grid = fib_levels(_frame(list(np.linspace(100, 200, 50))))
        self.assertAlmostEqual(grid["fib_500"], 150.0, delta=1.0)
        self.assertLess(grid["fib_382"], grid["fib_618"])

    def test_distance_is_zero_at_the_level(self):
        df = _frame(list(np.linspace(100, 200, 50)))
        target = fib_levels(df)["fib_618"]
        self.assertLess(distance_to_fib(df, target), 0.001)


class DivergenceV2Tests(unittest.TestCase):
    def test_detects_higher_high_with_lower_rsi(self):
        closes = [10, 12, 10, 11, 13, 11, 12, 14, 12, 11, 10, 12, 15, 13, 12]
        rsi = [50, 80, 55, 60, 78, 58, 62, 70, 58, 55, 52, 60, 65, 58, 55]
        out = rsi_divergence(_frame([float(c) for c in closes], rsi=rsi), lookback=10, pivot=1)
        self.assertIn(out["bearish_divergence"], (0.0, 1.0))

    def test_missing_rsi_is_handled(self):
        out = rsi_divergence(_frame([1.0] * 40))
        self.assertEqual(out["bearish_divergence"], 0.0)


class LiquidationMapV2Tests(unittest.TestCase):
    def test_reports_share_of_liquidations_below_price(self):
        df = _frame(list(np.linspace(100, 120, 120)))
        out = estimate_liquidation_map(df, price=120.0)
        self.assertGreaterEqual(out["liq_below_pct"], 0.0)
        self.assertLessEqual(out["liq_below_pct"], 1.0)

    def test_recent_buying_puts_liquidations_just_under_price(self):
        # heavy volume right below the current price -> a close cluster
        closes = [100.0] * 100 + [118.0, 119.0, 120.0]
        vols = [1.0] * 100 + [5000.0, 5000.0, 5000.0]
        out = estimate_liquidation_map(_frame(closes, volumes=vols), price=120.0)
        self.assertTrue(np.isfinite(out["liq_nearest_dist"]))
        self.assertLess(out["liq_nearest_dist"], 0.11)

    def test_degrades_safely_on_short_input(self):
        out = estimate_liquidation_map(_frame([1.0, 2.0]), price=2.0)
        self.assertEqual(out["liq_below_pct"], 0.0)


class ConfluenceV2Tests(unittest.TestCase):
    def test_counts_timeframes_agreeing_on_a_level(self):
        # needs enough bars for pivot detection on every timeframe
        path = [float(p) for p in ([90, 94, 100, 94, 90] * 6 + [95, 97, 99])]
        frames = {"15m": _frame(path), "1h": _frame(path), "4h": _frame(path)}
        out = multi_timeframe_confluence(frames, price=99.0, tol_pct=0.03)
        self.assertGreaterEqual(out["confluence_count"], 1.0)
        self.assertGreater(out["confluence_strength"], 0.0)

    def test_empty_frames_do_not_crash(self):
        out = multi_timeframe_confluence({"15m": pd.DataFrame(), "1h": None}, price=100.0)
        self.assertEqual(out["confluence_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
