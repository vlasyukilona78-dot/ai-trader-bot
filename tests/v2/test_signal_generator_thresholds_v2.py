from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from core.signal_generator import SignalConfig, SignalGenerator
from core.volume_profile import VolumeProfileLevels


def _panic_row(rsi: float, volume_spike: float, close: float = 95.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rsi": rsi,
                "volume_spike": volume_spike,
                "close": close,
                "bb_upper": 110.0,
                "bb_lower": 96.0,  # close below bb_lower -> band_down True
                "kc_upper": 110.0,
                "kc_lower": 97.0,
            }
        ]
    )


class SignalConfigDefaultsV2Tests(unittest.TestCase):
    def test_new_tightened_defaults(self):
        cfg = SignalConfig()
        self.assertEqual(cfg.rsi_low, 25.0)
        self.assertEqual(cfg.rsi_high, 75.0)
        self.assertEqual(cfg.volume_spike_threshold, 2.5)
        self.assertEqual(cfg.entry_tolerance_pct, 0.0015)
        self.assertTrue(cfg.confirmation_enabled)
        self.assertEqual(cfg.confirmation_max_wait_bars, 3)
        self.assertEqual(cfg.confirmation_invalidate_pct, 0.0015)


class Layer1ThresholdBoundaryV2Tests(unittest.TestCase):
    """A signal that used to pass under the old (32 / 1.6 / >=2-of-3) gate but
    must be rejected under the new (25 / 2.5 / >=3-of-3) gate."""

    def test_borderline_rsi_and_volume_now_rejected(self):
        gen = SignalGenerator(SignalConfig())
        df = _panic_row(rsi=30.0, volume_spike=1.8)
        side, metrics = gen._layer1_pump_detection(df)
        self.assertIsNone(side)
        self.assertEqual(metrics["panic_points"], 1.0)  # only band_down counts now

    def test_same_row_admitted_under_old_thresholds_alone(self):
        # Confirms the rejection above comes from the tightened thresholds, not just
        # the 3-of-3 gate: with the OLD rsi_low/volume_spike_threshold values (but the
        # current >=3-of-3 gate), this row has all three conditions true and is admitted.
        old_cfg = SignalConfig(rsi_low=32.0, volume_spike_threshold=1.6)
        gen = SignalGenerator(old_cfg)
        df = _panic_row(rsi=30.0, volume_spike=1.8)
        side, metrics = gen._layer1_pump_detection(df)
        self.assertEqual(side, "LONG")
        self.assertEqual(metrics["panic_points"], 3.0)

    def test_two_of_three_no_longer_sufficient(self):
        # band_down + rsi extreme true, volume_spike short of threshold -> only 2 of 3
        gen = SignalGenerator(SignalConfig())
        df = _panic_row(rsi=20.0, volume_spike=2.0)  # rsi<=25 true, volume>=2.5 false, band_down true
        side, metrics = gen._layer1_pump_detection(df)
        self.assertIsNone(side)
        self.assertEqual(metrics["panic_points"], 2.0)

    def test_all_three_conditions_still_admits_signal(self):
        gen = SignalGenerator(SignalConfig())
        df = _panic_row(rsi=20.0, volume_spike=3.0)  # all three conditions true
        side, metrics = gen._layer1_pump_detection(df)
        self.assertEqual(side, "LONG")
        self.assertEqual(metrics["panic_points"], 3.0)


class Layer3EntryToleranceV2Tests(unittest.TestCase):
    def _df(self, prev_close: float, close: float) -> pd.DataFrame:
        return pd.DataFrame({"close": [prev_close, close]})

    def test_tighter_tolerance_rejects_previously_admitted_entry(self):
        vp = VolumeProfileLevels(poc=100.0, vah=100.0, val=90.0)
        df = self._df(prev_close=99.7, close=99.8)

        new_gen = SignalGenerator(SignalConfig(entry_tolerance_pct=0.0015, pump_window_enabled=False))
        with patch.object(new_gen, "_layer3_msb_confirmation", return_value=(True, {})):
            ok_new, _ = new_gen._layer3_entry_location(df, "SHORT", vp)
        self.assertFalse(ok_new)

        old_gen = SignalGenerator(SignalConfig(entry_tolerance_pct=0.004, pump_window_enabled=False))
        with patch.object(old_gen, "_layer3_msb_confirmation", return_value=(True, {})):
            ok_old, _ = old_gen._layer3_entry_location(df, "SHORT", vp)
        self.assertTrue(ok_old)


if __name__ == "__main__":
    unittest.main()
