from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alerts.chart_generator import build_signal_chart
from alerts.chart_generator import _estimate_liquidation_margin_usdt
from alerts.chart_generator import _fmt_compact_notional
from alerts.chart_generator import _fmt_liquidation_margin_label
from alerts.chart_generator import _select_visible_liquidation_bands
from core.liquidation_map import LiquidationBand
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
        df.attrs["coinglass_liquidation_bands"] = [
            {
                "level": 109.0,
                "weight": 5.0,
                "side": "above",
                "source": "coinglass",
                "start_ts": int(df.index[-30].timestamp()),
                "end_ts": int(df.index[-1].timestamp()),
                "notional_usdt": 503000.0,
            }
        ]
        liq_map = build_liquidation_map(df)
        self.assertTrue(any(band.notional_usdt >= 503000.0 for band in liq_map.bands))
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
            liquidation_map=liq_map,
            show_liquidation_map=True,
        )
        self.assertIsInstance(image, bytes)
        self.assertGreater(len(image), 1024)

    def test_compact_notional_formatter_matches_chart_labels(self):
        self.assertEqual(_fmt_compact_notional(77000.0), "77K")
        self.assertEqual(_fmt_compact_notional(2_000_000.0), "2M")
        self.assertEqual(_fmt_compact_notional(1_250_000.0), "1.2M")
        self.assertEqual(_fmt_liquidation_margin_label(margin_usdt=12_500.0, notional_usdt=77_000.0), "12K margin")
        self.assertEqual(_fmt_liquidation_margin_label(notional_usdt=77_000.0), "77K margin")
        self.assertEqual(_fmt_liquidation_margin_label(margin_usdt=77_000.0, estimated=True), "~77K margin")

    def test_chart_estimates_missing_liquidation_margin_labels_from_volume(self):
        df = self._build_df()
        df["turnover_usdt"] = pd.to_numeric(df["volume"], errors="coerce") * pd.to_numeric(df["close"], errors="coerce")
        band = LiquidationBand(
            level=108.7,
            weight=4.0,
            intensity=0.80,
            side="above",
            start_index=70,
            end_index=95,
            source="synthetic",
        )

        estimate = _estimate_liquidation_margin_usdt(band, df)

        self.assertGreater(estimate, 0.0)
        self.assertTrue(_fmt_liquidation_margin_label(margin_usdt=estimate, estimated=True).startswith("~"))

    def test_visible_liquidation_bands_prioritize_near_external_notional_levels(self):
        df = self._build_df()
        df.attrs["coinglass_liquidation_bands"] = [
            {
                "level": 108.7,
                "weight": 3.2,
                "side": "above",
                "source": "coinglass",
                "start_ts": int(df.index[-34].timestamp()),
                "end_ts": int(df.index[-1].timestamp()),
                "notional_usdt": 77_000.0,
            },
            {
                "level": 140.0,
                "weight": 8.0,
                "side": "above",
                "source": "coinglass",
                "start_ts": int(df.index[-34].timestamp()),
                "end_ts": int(df.index[-1].timestamp()),
                "notional_usdt": 3_000_000.0,
            },
        ]
        liq_map = build_liquidation_map(df)

        visible = _select_visible_liquidation_bands(
            liq_map,
            frame_len=len(df),
            last_close=float(df["close"].iloc[-1]),
            max_bands=4,
        )

        self.assertTrue(any(abs(band.level - 108.7) < 0.8 and band.notional_usdt >= 77_000.0 for band in visible))
        self.assertFalse(any(abs(band.level - 140.0) < 0.8 for band in visible))


if __name__ == "__main__":
    unittest.main()
