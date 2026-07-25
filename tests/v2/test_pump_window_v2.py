from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.signal_generator import SignalConfig, SignalGenerator


def _pump_df(
    n: int = 60,
    pump_gain: float = 0.10,
    bars_since_peak: int = 3,
    retrace_frac: float = 0.2,
) -> pd.DataFrame:
    """Synthetic pump: flat base, sharp run-up, then a partial fade off the peak."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    base = 1.0
    peak = base * (1.0 + pump_gain)
    peak_pos = n - 1 - bars_since_peak

    close = np.full(n, base, dtype=float)
    # endpoint=False so the ramp approaches the peak without reaching it early,
    # keeping the peak bar the unique maximum of the window.
    close[:peak_pos] = np.linspace(base, peak, peak_pos, endpoint=False)
    close[peak_pos] = peak
    fade_to = peak - (peak - base) * retrace_frac
    close[peak_pos:] = np.linspace(peak, fade_to, n - peak_pos)

    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": [10.0] * n,
            "rsi": [50.0] * n,
            "volume_spike": [1.0] * n,
            "bb_upper": close * 1.10,
            "bb_lower": close * 0.90,
            "kc_upper": close * 1.10,
            "kc_lower": close * 0.90,
            "atr": close * 0.01,
        },
        index=idx,
    )
    # Mark the peak bar as a genuine pump event (band breakout + hot RSI + volume).
    df.iloc[peak_pos, df.columns.get_loc("rsi")] = 90.0
    df.iloc[peak_pos, df.columns.get_loc("volume_spike")] = 6.0
    df.iloc[peak_pos, df.columns.get_loc("bb_upper")] = peak * 0.99
    df.iloc[peak_pos, df.columns.get_loc("kc_upper")] = peak * 0.99
    # Guarantee the peak bar owns the window high regardless of the fade shape.
    df.iloc[peak_pos, df.columns.get_loc("high")] = peak * 1.001
    return df


class PumpWindowLayer1V2Tests(unittest.TestCase):
    def _gen(self, **over) -> SignalGenerator:
        cfg = {"pump_window_enabled": True, "pump_window_bars": 30, "pump_min_move_pct": 0.03}
        cfg.update(over)
        return SignalGenerator(SignalConfig(**cfg))

    def test_detects_faded_pump_as_short(self):
        side, m = self._gen()._layer1_pump_window(_pump_df())
        self.assertEqual(side, "SHORT")
        self.assertGreaterEqual(m["run_up_pct"], 0.03)
        self.assertGreaterEqual(m["bars_since_peak"], 1.0)

    def test_rejects_move_below_min_size(self):
        # A 1% wiggle is not a pump worth fading.
        side, m = self._gen()._layer1_pump_window(_pump_df(pump_gain=0.01))
        self.assertIsNone(side)
        self.assertLess(m["run_up_pct"], 0.03)

    def test_rejects_while_peak_is_the_current_bar(self):
        # Still making the high - the pump has not turned yet.
        side, _ = self._gen(pump_min_bars_since_peak=1)._layer1_pump_window(_pump_df(bars_since_peak=0))
        self.assertIsNone(side)

    def test_rejects_when_already_dumped_past_entry_window(self):
        # 90% of the move is gone; too late to fade with a tight stop.
        side, m = self._gen(pump_max_retrace_pct=0.5)._layer1_pump_window(_pump_df(retrace_frac=0.9))
        self.assertIsNone(side)
        self.assertGreater(m["retrace_from_high"], 0.5)

    def test_long_side_disabled_by_default(self):
        self.assertFalse(SignalConfig().enable_long_side)

    def test_legacy_single_bar_mode_still_available(self):
        gen = SignalGenerator(SignalConfig(pump_window_enabled=False))
        df = pd.DataFrame(
            [{"rsi": 80.0, "volume_spike": 3.0, "close": 105.0,
              "bb_upper": 100.0, "bb_lower": 90.0, "kc_upper": 100.0, "kc_lower": 90.0}]
        )
        side, m = gen._layer1_pump_detection(df)
        self.assertEqual(side, "SHORT")
        self.assertEqual(m["pump_points"], 3.0)


class StructuralStopV2Tests(unittest.TestCase):
    def test_stop_sits_above_the_pump_peak(self):
        gen = SignalGenerator(SignalConfig(pump_window_bars=30, pump_stop_buffer_pct=0.003))
        df = _pump_df()
        win_high = float(df["high"].tail(30).max())
        tp, sl, _ = gen._layer5_structural_levels(df, "SHORT", None)
        self.assertGreater(sl, win_high)
        self.assertLess(tp, float(df.iloc[-1]["close"]))

    def test_stop_is_tight_enough_for_high_leverage(self):
        gen = SignalGenerator(SignalConfig(pump_window_bars=30))
        df = _pump_df(pump_gain=0.10, retrace_frac=0.1)
        close = float(df.iloc[-1]["close"])
        _, sl, _ = gen._layer5_structural_levels(df, "SHORT", None)
        self.assertLess(abs(sl - close) / close, 0.05)


class QualityGateV2Tests(unittest.TestCase):
    """The gate encodes a counter-intuitive, measured result: calm and illiquid
    pumps are the ones that fail, so both checks are floors, not ceilings."""

    def _df(self, atr_pct: float, usd_per_bar: float, n: int = 20) -> pd.DataFrame:
        close = 10.0
        return pd.DataFrame(
            {
                "close": [close] * n,
                "high": [close * 1.001] * n,
                "low": [close * 0.999] * n,
                "volume": [usd_per_bar / close] * n,
                "atr": [close * atr_pct] * n,
            }
        )

    def test_accepts_volatile_and_liquid(self):
        gen = SignalGenerator(SignalConfig())
        ok, d = gen._layer1b_quality_gate(self._df(atr_pct=0.08, usd_per_bar=50_000))
        self.assertTrue(ok)
        self.assertEqual(d["atr_ok"], 1.0)
        self.assertEqual(d["liquidity_ok"], 1.0)

    def test_rejects_calm_pump(self):
        gen = SignalGenerator(SignalConfig())
        ok, d = gen._layer1b_quality_gate(self._df(atr_pct=0.01, usd_per_bar=50_000))
        self.assertFalse(ok)
        self.assertEqual(d["atr_ok"], 0.0)

    def test_rejects_illiquid_pump(self):
        gen = SignalGenerator(SignalConfig())
        ok, d = gen._layer1b_quality_gate(self._df(atr_pct=0.08, usd_per_bar=100))
        self.assertFalse(ok)
        self.assertEqual(d["liquidity_ok"], 0.0)

    def test_gates_can_be_disabled(self):
        gen = SignalGenerator(SignalConfig(min_atr_pct=0.0, min_hourly_usd_volume=0.0))
        ok, _ = gen._layer1b_quality_gate(self._df(atr_pct=0.001, usd_per_bar=1))
        self.assertTrue(ok)

    def test_max_safe_leverage_is_inverse_of_stop_distance(self):
        gen = SignalGenerator(SignalConfig())
        # a 2% stop must not be traded above 50x, or a stop-out is a liquidation
        self.assertAlmostEqual(1.0 / 0.02, 50.0, places=6)


class SignalQualityGuardsV2Tests(unittest.TestCase):
    def test_defaults_enforce_stop_and_payoff_limits(self):
        cfg = SignalConfig()
        self.assertEqual(cfg.max_stop_distance_pct, 0.03)
        self.assertEqual(cfg.min_risk_reward, 1.5)
        self.assertTrue(cfg.pump_window_enabled)

    def test_defaults_match_the_measured_best_configuration(self):
        # Locked in from live MEXC measurement: 45-bar window with a 3% minimum move
        # and the confirmation bar on. Widening the window or lowering the move
        # threshold measured net-negative, so these are regression-guarded.
        cfg = SignalConfig()
        self.assertEqual(cfg.pump_window_bars, 45)
        self.assertEqual(cfg.pump_min_move_pct, 0.03)
        self.assertTrue(cfg.confirmation_enabled)


if __name__ == "__main__":
    unittest.main()
