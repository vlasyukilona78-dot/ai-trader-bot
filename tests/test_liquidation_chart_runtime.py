from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alerts.chart_generator import build_signal_chart
from app.main import _build_early_watch_candidate
from core.liquidation_map import build_liquidation_map
from core.volume_profile import VolumeProfileLevels
from trading.signals.signal_types import IntentAction, StrategyIntent


class LiquidationChartRuntimeTests(unittest.TestCase):
    @staticmethod
    def _build_df() -> pd.DataFrame:
        idx = pd.date_range("2026-03-01", periods=80, freq="min", tz="UTC")
        close = np.linspace(100.0, 108.0, 80)
        close[-1] = 107.6
        close[-2] = 108.1
        volume = np.linspace(1200.0, 2000.0, 80)
        df = pd.DataFrame(
            {
                "open": close - 0.35,
                "high": close + 0.55,
                "low": close - 0.60,
                "close": close,
                "volume": volume,
                "atr": np.full(80, 0.95),
                "ema20": pd.Series(close).ewm(span=20, adjust=False).mean().values,
                "ema50": pd.Series(close).ewm(span=50, adjust=False).mean().values,
                "vwap": close - 0.25,
                "rsi": np.linspace(54.0, 61.0, 80),
                "volume_spike": np.linspace(1.0, 1.35, 80),
                "bb_upper": close + 0.20,
                "kc_upper": close + 0.12,
                "hist": np.linspace(0.20, -0.03, 80),
                "obv": np.linspace(100.0, 170.0, 80),
                "cvd": np.linspace(120.0, 180.0, 80),
            },
            index=idx,
        )
        df.iloc[-1, df.columns.get_loc("high")] = float(df.iloc[-1]["close"]) + 1.3
        df.iloc[-1, df.columns.get_loc("volume_spike")] = 1.18
        df.iloc[-1, df.columns.get_loc("obv")] = float(df.iloc[-2]["obv"]) - 1.2
        df.iloc[-1, df.columns.get_loc("cvd")] = float(df.iloc[-2]["cvd"]) - 1.1
        return df

    @staticmethod
    def _trace_meta() -> dict:
        return {
            "layer_trace": {
                "layers": {
                    "regime_filter": {"passed": True, "details": {}},
                    "layer1_pump_detection": {
                        "passed": False,
                        "details": {
                            "clean_pump_pct": 0.058,
                            "clean_pump_min_pct_used": 0.05,
                            "clean_pump_ok": 1.0,
                            "rsi": 58.5,
                            "volume_spike": 1.10,
                        },
                    },
                    "layer2_weakness_confirmation": {
                        "passed": False,
                        "details": {
                            "weakness_strength": 0.72,
                            "failed_reclaim": 1.0,
                            "retest_failed_breakout": 1.0,
                        },
                    },
                }
            },
            "layer_failed": "layer1_pump_detection",
        }

    @staticmethod
    def _build_df_with_open_and_closed_bands() -> pd.DataFrame:
        idx = pd.date_range("2026-03-01", periods=72, freq="h", tz="UTC")
        close = np.linspace(1.00, 0.96, 72)
        df = pd.DataFrame(
            {
                "open": close + 0.005,
                "high": close + 0.018,
                "low": close - 0.018,
                "close": close,
                "volume": np.linspace(800.0, 1500.0, 72),
                "atr": np.full(72, 0.02),
                "ema20": pd.Series(close).ewm(span=20, adjust=False).mean().values,
                "ema50": pd.Series(close).ewm(span=50, adjust=False).mean().values,
                "vwap": close - 0.004,
                "hist": np.linspace(0.03, -0.02, 72),
                "volume_spike": np.ones(72),
            },
            index=idx,
        )
        df.iloc[14, df.columns.get_loc("high")] = 1.08
        df.iloc[30, df.columns.get_loc("high")] = 1.082
        df.iloc[50, df.columns.get_loc("low")] = 0.90
        return df

    def test_liquidation_map_builds_context(self):
        liq_map = build_liquidation_map(self._build_df())
        self.assertTrue(liq_map.bands)
        self.assertGreaterEqual(liq_map.upside_risk, 0.0)

    def test_liquidation_map_tracks_open_and_closed_bands(self):
        liq_map = build_liquidation_map(self._build_df_with_open_and_closed_bands())
        self.assertTrue(liq_map.bands)
        self.assertTrue(any(band.closed_index is not None for band in liq_map.bands))
        self.assertTrue(any(band.closed_index is None for band in liq_map.bands))
        self.assertTrue(any(band.end_index < 71 for band in liq_map.bands if band.closed_index is not None))
        self.assertTrue(any(band.end_index == 71 for band in liq_map.bands if band.closed_index is None))

    def test_chart_renders_with_liquidation_map(self):
        df = self._build_df()
        image = build_signal_chart(
            symbol="SUIUSDT",
            df=df,
            side="SHORT",
            entry=105.7,
            tp=104.0,
            sl=107.4,
            volume_profile=VolumeProfileLevels(poc=104.9, vah=106.8, val=103.8),
            timeframe_label="1m",
            show_trade_levels=True,
            liquidation_map=build_liquidation_map(df),
            show_liquidation_map=True,
        )
        self.assertIsInstance(image, bytes)
        self.assertGreater(len(image), 1024)

    def test_early_watch_candidate_uses_liquidation_context(self):
        df = self._build_df()
        intent = StrategyIntent(
            symbol="SUIUSDT",
            action=IntentAction.HOLD,
            reason="no_signal_layer1_pump_detection",
            metadata=self._trace_meta(),
        )
        candidate = _build_early_watch_candidate(
            symbol="SUIUSDT",
            timeframe="1",
            mode="paper",
            enriched=df,
            intent=intent,
        )
        self.assertIsNotNone(candidate)
        self.assertIn("WATCH", str(candidate["phase"]))


if __name__ == "__main__":
    unittest.main()
