from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.signal_generator import SignalConfig, SignalGenerator


def _frame(closes, atr: float) -> pd.DataFrame:
    n = len(closes)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        }
    )
    df["atr"] = atr
    return df


class StopBufferScalesWithVolatilityV2Tests(unittest.TestCase):
    """A flat percentage buffer cannot serve coins whose ranges differ tenfold.

    Measured on the real pipeline, a 0.3% buffer put the stop inside ordinary
    hourly noise, so the invalidation level was tagged almost immediately and
    the confirmation bar killed nearly every candidate before it could fire.
    """

    def test_a_volatile_coin_gets_a_wider_buffer_than_a_quiet_one(self):
        gen = SignalGenerator(SignalConfig(stop_buffer_atr_mult=0.5))
        quiet = gen._stop_buffer(_frame([100.0] * 30, atr=0.2), reference=100.0)
        volatile = gen._stop_buffer(_frame([100.0] * 30, atr=8.0), reference=100.0)
        self.assertGreater(volatile, quiet)
        self.assertAlmostEqual(volatile, 4.0)  # 0.5 * ATR

    def test_the_percentage_acts_as_a_floor_not_a_cap(self):
        """A near-zero ATR must not collapse the buffer to nothing."""
        gen = SignalGenerator(SignalConfig(stop_buffer_atr_mult=0.5, pump_stop_buffer_pct=0.003))
        buf = gen._stop_buffer(_frame([100.0] * 30, atr=0.0), reference=100.0)
        self.assertAlmostEqual(buf, 0.3)  # falls back to 0.3% of 100

    def test_a_missing_atr_column_does_not_raise(self):
        gen = SignalGenerator(SignalConfig(stop_buffer_atr_mult=0.5))
        df = _frame([100.0] * 30, atr=1.0).drop(columns=["atr"])
        self.assertAlmostEqual(gen._stop_buffer(df, reference=100.0), 0.3)

    def test_zero_multiplier_reproduces_the_old_flat_behaviour(self):
        gen = SignalGenerator(SignalConfig(stop_buffer_atr_mult=0.0, pump_stop_buffer_pct=0.003))
        self.assertAlmostEqual(gen._stop_buffer(_frame([100.0] * 30, atr=8.0), 100.0), 0.3)

    def test_short_stop_sits_above_the_swing_by_the_scaled_buffer(self):
        gen = SignalGenerator(SignalConfig(stop_buffer_atr_mult=0.5, structural_anchor_htf=False))
        df = _frame([100.0] * 30, atr=4.0)
        _, sl, _ = gen._layer5_structural_levels(df, "SHORT", None)
        win_high = float(df["high"].max())
        self.assertAlmostEqual(sl, win_high + 2.0)  # 0.5 * 4.0

    def test_long_stop_sits_below_the_swing_by_the_scaled_buffer(self):
        gen = SignalGenerator(SignalConfig(stop_buffer_atr_mult=0.5, structural_anchor_htf=False))
        df = _frame([100.0] * 30, atr=4.0)
        _, sl, _ = gen._layer5_structural_levels(df, "LONG", None)
        win_low = float(df["low"].min())
        self.assertAlmostEqual(sl, win_low - 2.0)


class StructuralAnchorV2Tests(unittest.TestCase):
    """The swing is measured on the higher timeframe when one is available."""

    def test_the_higher_timeframe_frame_supplies_the_swing(self):
        gen = SignalGenerator(SignalConfig(structural_anchor_htf=True, structural_anchor_htf_bars=12))
        entry = _frame(list(np.linspace(100, 110, 40)), atr=1.0)
        htf = _frame(list(np.linspace(100, 130, 20)), atr=1.0)
        high, _ = gen._window_extremes(entry, htf)
        self.assertAlmostEqual(high, float(htf["high"].tail(12).max()))

    def test_it_falls_back_to_the_entry_frame_when_no_htf_is_supplied(self):
        gen = SignalGenerator(SignalConfig(structural_anchor_htf=True))
        entry = _frame(list(np.linspace(100, 110, 40)), atr=1.0)
        high, low = gen._window_extremes(entry, None)
        self.assertAlmostEqual(high, float(entry["high"].max()))
        self.assertAlmostEqual(low, float(entry["low"].min()))

    def test_a_too_short_htf_frame_falls_back_rather_than_anchoring_on_noise(self):
        gen = SignalGenerator(SignalConfig(structural_anchor_htf=True))
        entry = _frame(list(np.linspace(100, 110, 40)), atr=1.0)
        stub = _frame([200.0, 201.0], atr=1.0)  # two bars is not a swing
        high, _ = gen._window_extremes(entry, stub)
        self.assertAlmostEqual(high, float(entry["high"].max()))

    def test_the_toggle_restores_the_entry_frame_anchor(self):
        gen = SignalGenerator(SignalConfig(structural_anchor_htf=False))
        entry = _frame(list(np.linspace(100, 110, 40)), atr=1.0)
        htf = _frame(list(np.linspace(100, 130, 20)), atr=1.0)
        high, _ = gen._window_extremes(entry, htf)
        self.assertAlmostEqual(high, float(entry["high"].max()))


class BufferDefaultsV2Tests(unittest.TestCase):
    def test_defaults_are_pinned(self):
        cfg = SignalConfig()
        self.assertEqual(cfg.stop_buffer_atr_mult, 0.5)
        self.assertTrue(cfg.structural_anchor_htf)
        self.assertEqual(cfg.structural_anchor_htf_bars, 12)


if __name__ == "__main__":
    unittest.main()
