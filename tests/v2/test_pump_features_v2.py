from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.pump_features import (
    build_pump_features,
    extension,
    pump_acceleration,
    rejection_wicks,
    relative_strength,
    volume_exhaustion,
)


def _frame(closes, volumes=None, highs=None, opens=None) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "open": opens if opens is not None else closes,
            "high": highs if highs is not None else [c * 1.002 for c in closes],
            "low": [c * 0.998 for c in closes],
            "close": closes,
            "volume": volumes if volumes is not None else [100.0] * n,
        }
    )


class RelativeStrengthV2Tests(unittest.TestCase):
    def test_isolates_a_single_coin_move_from_a_market_rally(self):
        coin = _frame(list(np.linspace(100, 130, 40)))       # +30%
        flat_btc = _frame([100.0] * 40)                      # market unchanged
        out = relative_strength(coin, flat_btc, lookback=24)
        self.assertGreater(out["relative_strength"], 0.1)
        self.assertGreater(out["idiosyncratic"], 0.9)

    def test_market_wide_rally_is_not_idiosyncratic(self):
        coin = _frame(list(np.linspace(100, 130, 40)))
        btc = _frame(list(np.linspace(100, 130, 40)))        # same move
        out = relative_strength(coin, btc, lookback=24)
        self.assertAlmostEqual(out["relative_strength"], 0.0, places=6)
        self.assertLess(out["idiosyncratic"], 0.05)

    def test_missing_benchmark_degrades_without_raising(self):
        out = relative_strength(_frame(list(np.linspace(100, 130, 40))), None)
        self.assertTrue(np.isfinite(out["coin_return"]))
        self.assertTrue(np.isnan(out["relative_strength"]))


class VolumeExhaustionV2Tests(unittest.TestCase):
    def test_fading_volume_gives_a_negative_trend(self):
        out = volume_exhaustion(_frame([100.0] * 24, volumes=list(np.linspace(1000, 100, 24))))
        self.assertLess(out["volume_trend"], 0)
        self.assertLess(out["late_vs_early_volume"], 1.0)

    def test_building_volume_gives_a_positive_trend(self):
        out = volume_exhaustion(_frame([100.0] * 24, volumes=list(np.linspace(100, 1000, 24))))
        self.assertGreater(out["volume_trend"], 0)
        self.assertGreater(out["late_vs_early_volume"], 1.0)


class RejectionWicksV2Tests(unittest.TestCase):
    def test_long_upper_wicks_score_high(self):
        closes = [100.0] * 12
        out = rejection_wicks(_frame(closes, highs=[110.0] * 12, opens=[99.0] * 12))
        self.assertGreater(out["upper_wick_ratio"], 0.5)

    def test_bodies_without_wicks_score_low(self):
        closes = [100.0] * 12
        out = rejection_wicks(_frame(closes, highs=[100.05] * 12, opens=[99.0] * 12))
        self.assertLess(out["upper_wick_ratio"], 0.2)


class AccelerationV2Tests(unittest.TestCase):
    def test_decelerating_run_is_negative(self):
        path = list(np.linspace(100, 130, 12)) + list(np.linspace(130, 132, 12))
        self.assertLess(pump_acceleration(_frame(path))["acceleration"], 0)

    def test_accelerating_run_is_positive(self):
        path = list(np.linspace(100, 102, 12)) + list(np.linspace(102, 130, 12))
        self.assertGreater(pump_acceleration(_frame(path))["acceleration"], 0)


class ExtensionV2Tests(unittest.TestCase):
    def test_extension_is_measured_in_atr_units(self):
        df = _frame([100.0] * 30)
        df["atr"] = 1.0
        df["ema20"] = 95.0
        df["ema50"] = 90.0
        out = extension(df)
        self.assertAlmostEqual(out["ext_ema20_atr"], 5.0)
        self.assertAlmostEqual(out["ext_ema50_atr"], 10.0)

    def test_counts_the_current_up_run(self):
        df = _frame([100, 99, 98, 99, 100, 101, 102])
        df["atr"] = 1.0
        self.assertEqual(extension(df)["consecutive_up"], 4.0)


class BuildPumpFeaturesV2Tests(unittest.TestCase):
    def test_returns_the_full_set_without_a_benchmark(self):
        df = _frame(list(np.linspace(100, 120, 40)))
        df["atr"] = 1.0
        df["ema20"] = 110.0
        feats = build_pump_features(df)
        for key in ("coin_return", "volume_trend", "upper_wick_ratio", "acceleration", "ext_ema20_atr"):
            self.assertIn(key, feats)


if __name__ == "__main__":
    unittest.main()
