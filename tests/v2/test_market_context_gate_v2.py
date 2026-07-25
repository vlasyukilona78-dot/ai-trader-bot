from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.signal_generator import SignalConfig, SignalGenerator


def _frame(closes, atr=1.0) -> pd.DataFrame:
    n = len(closes)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.002 for c in closes],
            "low": [c * 0.998 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        }
    )
    df["atr"] = atr
    return df


class MarketContextGateV2Tests(unittest.TestCase):
    """The gate exists to reject beta: a coin that merely followed the market up
    is not the engineered pump the strategy fades."""

    def _gen(self, **over) -> SignalGenerator:
        cfg = {"min_relative_strength": 0.05, "relative_strength_lookback": 24}
        cfg.update(over)
        return SignalGenerator(SignalConfig(**cfg))

    def test_accepts_a_coin_that_outran_the_market(self):
        coin = _frame(list(np.linspace(100, 130, 40)))
        btc = _frame([100.0] * 40)
        ok, d = self._gen()._layer1c_market_context(coin, btc)
        self.assertTrue(ok)
        self.assertGreater(d["relative_strength"], 0.05)

    def test_rejects_a_market_wide_rally(self):
        coin = _frame(list(np.linspace(100, 130, 40)))
        btc = _frame(list(np.linspace(100, 130, 40)))
        ok, d = self._gen()._layer1c_market_context(coin, btc)
        self.assertFalse(ok)
        self.assertEqual(d["relative_strength_ok"], 0.0)

    def test_missing_benchmark_does_not_block_everything(self):
        coin = _frame(list(np.linspace(100, 130, 40)))
        ok, d = self._gen()._layer1c_market_context(coin, None)
        self.assertTrue(ok)
        self.assertEqual(d["relative_strength_ok"], 1.0)

    def test_gate_can_be_disabled(self):
        coin = _frame(list(np.linspace(100, 130, 40)))
        btc = _frame(list(np.linspace(100, 130, 40)))
        ok, _ = self._gen(min_relative_strength=0.0)._layer1c_market_context(coin, btc)
        self.assertTrue(ok)

    def test_level_requirement_is_optional_and_off_by_default(self):
        self.assertFalse(SignalConfig().require_level_overhead)

    def test_level_requirement_rejects_a_breakout_with_nothing_overhead(self):
        gen = self._gen(min_relative_strength=0.0, require_level_overhead=True, min_level_dist_pct=0.018)
        breakout = _frame(list(np.linspace(100, 160, 60)))  # straight up, no level above
        ok, d = gen._layer1c_market_context(breakout, None)
        self.assertFalse(ok)
        self.assertEqual(d["level_ok"], 0.0)


class WeaknessLayerDefaultV2Tests(unittest.TestCase):
    def test_divergence_layer_is_off_by_default(self):
        """Measured at -0.0057 expectancy with the layer versus +0.0245 without,
        so it stays off until something better replaces it."""
        self.assertFalse(SignalConfig().weakness_layer_enabled)

    def test_relative_strength_default_matches_the_measured_threshold(self):
        self.assertEqual(SignalConfig().min_relative_strength, 0.05)


if __name__ == "__main__":
    unittest.main()
