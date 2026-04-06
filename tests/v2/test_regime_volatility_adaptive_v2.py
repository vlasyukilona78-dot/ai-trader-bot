from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.market_regime import MarketRegime
from core.signal_generator import SignalConfig, SignalGenerator


class RegimeVolatilityAdaptiveV2Tests(unittest.TestCase):
    def test_regime_filter_adapts_volatility_threshold_for_quieter_symbol(self):
        idx = pd.date_range("2026-03-01", periods=120, freq="min", tz="UTC")
        close = np.linspace(100.0, 101.2, 120)
        df = pd.DataFrame(
            {
                "close": close,
                "ema20": close - 0.1,
                "ema50": close,
                "adx": np.full(120, 18.0),
                "vwap": close - 0.2,
                "vwap_dist": np.full(120, 0.0015),
                "atr": np.full(120, 0.056),
            },
            index=idx,
        )

        generator = SignalGenerator(
            SignalConfig(
                regime_volatility_threshold_override=0.0006,
                regime_volatility_dynamic_floor_mult=0.75,
                regime_volatility_baseline_lookback=96,
            )
        )

        passed, details = generator._regime_filter(
            df,
            MarketRegime.RANGE,
            news_veto=None,
            news_source="unavailable",
        )

        self.assertTrue(passed)
        self.assertLess(float(details["volatility_threshold_used"]), 0.0006)
        self.assertGreater(float(details["volatility_threshold_used"]), 0.00045)


if __name__ == "__main__":
    unittest.main()
