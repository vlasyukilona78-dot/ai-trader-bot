from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trading.signals.ultra_v2 import UltraV2Config, evaluate_ultra_v2


class UltraV2Tests(unittest.TestCase):
    @staticmethod
    def _trace_meta(**details: float) -> dict[str, object]:
        return {
            "layer_trace": {
                "layers": {
                    "layer2_weakness_confirmation": {
                        "details": {
                            "failed_reclaim": details.get("failed_reclaim", 0.0),
                            "retest_failed_breakout": details.get("retest_failed_breakout", 0.0),
                        }
                    },
                    "layer3_entry_location": {
                        "details": {
                            "acceptance_above_swing_high": details.get("acceptance_above_swing_high", 0.0),
                            "peak_followthrough_confirmed": details.get("peak_followthrough_confirmed", 0.0),
                            "downside_displacement_confirmed": details.get("downside_displacement_confirmed", 0.0),
                        }
                    },
                }
            }
        }

    @staticmethod
    def _climax_frame() -> pd.DataFrame:
        idx = pd.date_range("2026-01-01", periods=80, freq="min", tz="UTC")
        close = np.linspace(100.0, 110.0, 80)
        close[-5:] = [108.0, 110.0, 109.0, 108.5, 107.8]
        open_px = close - 0.10
        high = close + 0.40
        low = close - 0.40
        open_px[-1] = 108.7
        high[-1] = 111.0
        low[-1] = 107.0
        volume_spike = np.full(80, 1.05)
        volume_spike[-4] = 2.25
        volume_spike[-1] = 1.45
        rsi = np.linspace(55.0, 73.0, 80)
        rsi[-2] = 74.0
        rsi[-1] = 70.0
        hist = np.linspace(0.01, 0.05, 80)
        hist[-2] = 0.060
        hist[-1] = 0.034
        df = pd.DataFrame(
            {
                "open": open_px,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.linspace(1000.0, 2200.0, 80),
                "atr": np.full(80, 0.80),
                "rsi": rsi,
                "volume_spike": volume_spike,
                "hist": hist,
                "vwap_dist": np.full(80, 0.026),
                "bb_position": np.full(80, 0.80),
            },
            index=idx,
        )
        df.attrs["coinglass_liquidation_bands"] = [
            {
                "level": 110.7,
                "weight": 5.0,
                "side": "above",
                "source": "coinglass",
                "start_ts": int(idx[-20].timestamp()),
                "end_ts": int(idx[-1].timestamp()),
                "notional_usdt": 724_000.0,
            },
            {
                "level": 104.8,
                "weight": 2.6,
                "side": "below",
                "source": "coinglass",
                "start_ts": int(idx[-24].timestamp()),
                "end_ts": int(idx[-1].timestamp()),
                "notional_usdt": 360_000.0,
            },
        ]
        return df

    def test_accepts_sweep_and_fail_confirm(self):
        result = evaluate_ultra_v2(
            self._climax_frame(),
            metadata=self._trace_meta(failed_reclaim=1.0, downside_displacement_confirmed=1.0),
        )

        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(result.phase, "ULTRA_CONFIRM")
        self.assertEqual(result.scenario, "sweep_and_fail")
        self.assertGreaterEqual(result.score, 0.68)
        self.assertIn("volume_decay", result.triggers)
        self.assertLess(result.fib50, float(self._climax_frame().iloc[-1]["close"]))
        self.assertGreater(result.sl, float(self._climax_frame().iloc[-1]["close"]))

    def test_rejects_strong_continuation_before_ultra(self):
        df = self._climax_frame()
        df.iloc[-3:, df.columns.get_loc("close")] = [109.2, 110.0, 110.7]
        df.iloc[-3:, df.columns.get_loc("open")] = [108.9, 109.5, 110.0]
        df.iloc[-3:, df.columns.get_loc("high")] = [109.5, 110.4, 110.95]
        df.iloc[-1, df.columns.get_loc("low")] = 109.8
        df["mtf_trend_5m"] = 0.010
        df["mtf_rsi_5m"] = 70.0
        df["mtf_trend_15m"] = 0.006
        df["mtf_rsi_15m"] = 67.0

        result = evaluate_ultra_v2(
            df,
            metadata=self._trace_meta(acceptance_above_swing_high=1.0),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "continuation_blocker")

    def test_radar_can_exist_without_full_confirmation(self):
        df = self._climax_frame()
        df.iloc[-1, df.columns.get_loc("close")] = 109.35
        df.iloc[-1, df.columns.get_loc("open")] = 109.20
        df.iloc[-1, df.columns.get_loc("low")] = 108.90
        df.iloc[-1, df.columns.get_loc("high")] = 110.10
        df.iloc[-1, df.columns.get_loc("rsi")] = 73.5
        df.iloc[-1, df.columns.get_loc("hist")] = 0.061
        df.iloc[-1, df.columns.get_loc("volume_spike")] = 1.95

        result = evaluate_ultra_v2(
            df,
            metadata=self._trace_meta(),
            config=UltraV2Config(radar_score=0.46, confirm_score=0.95, direct_confirm_score=0.98),
        )

        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(result.phase, "ULTRA_RADAR")
        self.assertIn(result.scenario, {"blow_off_wick", "climax_then_fade", "sweep_and_fail"})


if __name__ == "__main__":
    unittest.main()
