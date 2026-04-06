from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alerts.chart_generator import build_signal_chart
from core.liquidation_map import build_liquidation_map
from core.volume_profile import VolumeProfileLevels


class LiquidationMapV2Tests(unittest.TestCase):
    @staticmethod
    def _build_df() -> pd.DataFrame:
        idx = pd.date_range("2026-03-01", periods=96, freq="min", tz="UTC")
        close = np.linspace(100.0, 107.0, 96)
        close[-12:] = [106.2, 106.8, 107.4, 108.9, 108.1, 107.5, 107.9, 107.4, 106.9, 106.2, 105.8, 105.6]
        volume = np.linspace(1000.0, 1900.0, 96)
        volume[-14] = 3400.0
        volume[-4] = 2800.0
        df = pd.DataFrame(
            {
                "open": close - 0.35,
                "high": close + 0.55,
                "low": close - 0.60,
                "close": close,
                "volume": volume,
                "atr": np.full(96, 0.95),
                "ema20": pd.Series(close).ewm(span=20, adjust=False).mean().values,
                "ema50": pd.Series(close).ewm(span=50, adjust=False).mean().values,
                "vwap": close - 0.25,
                "hist": np.linspace(0.22, -0.04, 96),
            },
            index=idx,
        )
        df.iloc[-14, df.columns.get_loc("high")] = float(df.iloc[-14]["close"]) + 2.1
        df.iloc[-4, df.columns.get_loc("low")] = float(df.iloc[-4]["close"]) - 1.9
        return df

    def test_build_liquidation_map_returns_bands(self):
        liq_map = build_liquidation_map(self._build_df())
        self.assertTrue(liq_map.bands)
        self.assertGreaterEqual(liq_map.strongest_above_weight, 0.0)
        self.assertGreaterEqual(liq_map.strongest_below_weight, 0.0)

    def test_chart_renders_with_liquidation_heatmap(self):
        df = self._build_df()
        image = build_signal_chart(
            symbol="SUIUSDT",
            df=df,
            side="SHORT",
            entry=105.6,
            tp=103.9,
            sl=107.2,
            volume_profile=VolumeProfileLevels(poc=104.8, vah=106.9, val=103.7),
            timeframe_label="1m",
            show_trade_levels=True,
            liquidation_map=build_liquidation_map(df),
            show_liquidation_map=True,
        )
        self.assertIsInstance(image, bytes)
        self.assertGreater(len(image), 1024)


if __name__ == "__main__":
    unittest.main()
