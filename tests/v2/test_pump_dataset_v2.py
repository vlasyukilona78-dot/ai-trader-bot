from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ai.pump_dataset import EventConfig, LabelConfig, PumpEvent, detect_events, label_event


def _hourly(closes: list[float], start_ts: int = 1_700_000_000) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "time": [start_ts + i * 3600 for i in range(n)],
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
        }
    )


def _forward(path: list[float], start_ts: int) -> pd.DataFrame:
    n = len(path)
    return pd.DataFrame(
        {
            "time": [start_ts + i * 300 for i in range(n)],
            "open": path,
            "high": [p * 1.0005 for p in path],
            "low": [p * 0.9995 for p in path],
            "close": path,
            "volume": [10.0] * n,
        }
    )


class DetectEventsV2Tests(unittest.TestCase):
    def test_flat_series_produces_no_events(self):
        df = _hourly([1.0] * 40)
        self.assertEqual(detect_events(df, EventConfig(min_move_pct=0.05)), [])

    def test_detects_a_run_up_above_threshold(self):
        df = _hourly([1.0] * 20 + list(np.linspace(1.0, 1.20, 10)) + [1.20] * 10)
        events = detect_events(df, EventConfig(min_move_pct=0.05, lookback_hours=6))
        self.assertTrue(events)
        self.assertGreaterEqual(events[0].move_pct, 0.05)

    def test_cooldown_prevents_duplicate_events_on_one_run_up(self):
        df = _hourly([1.0] * 20 + list(np.linspace(1.0, 1.40, 30)))
        few = detect_events(df, EventConfig(min_move_pct=0.05, cooldown_hours=12))
        many = detect_events(df, EventConfig(min_move_pct=0.05, cooldown_hours=1))
        self.assertLess(len(few), len(many))

    def test_move_below_threshold_is_ignored(self):
        df = _hourly([1.0] * 20 + list(np.linspace(1.0, 1.02, 20)))
        self.assertEqual(detect_events(df, EventConfig(min_move_pct=0.05)), [])


class LabelEventV2Tests(unittest.TestCase):
    def _event(self, ts: int = 1_700_100_000, entry: float = 100.0) -> PumpEvent:
        return PumpEvent(symbol="TEST", ts=ts, entry=entry, move_pct=0.10, run_up_bars=3)

    def test_immediate_fade_needs_no_averaging(self):
        ev = self._event()
        fwd = _forward([100.0, 99.0, 97.0, 95.0, 94.0], ev.ts)
        out = label_event(ev, fwd, LabelConfig(dca_target_pct=0.03))
        self.assertEqual(out["n_averages"], 0)
        self.assertEqual(out["dca_resolved"], 1)
        self.assertGreater(out["mfe_pct"], out["mae_pct"])
        self.assertEqual(out["mfe_beats_mae"], 1)

    def test_run_up_forces_averaging_legs(self):
        ev = self._event()
        # +18% against the short, then a collapse well below the blended entry
        fwd = _forward([100.0, 105.0, 110.0, 118.0, 100.0, 90.0, 80.0], ev.ts)
        out = label_event(ev, fwd, LabelConfig(dca_step_pct=0.08, dca_max_adds=6, dca_target_pct=0.03))
        self.assertGreaterEqual(out["n_averages"], 2)
        self.assertEqual(out["dca_resolved"], 1)
        self.assertGreater(out["mae_pct"], 0.15)

    def test_unresolved_when_price_never_comes_back(self):
        ev = self._event()
        fwd = _forward([100.0, 110.0, 125.0, 140.0, 160.0], ev.ts)
        out = label_event(ev, fwd, LabelConfig(dca_max_adds=6, dca_target_pct=0.03))
        self.assertEqual(out["dca_resolved"], 0)
        self.assertGreater(out["dca_peak_drawdown_units"], 0.0)

    def test_good_mae_flags_track_the_threshold(self):
        ev = self._event()
        fwd = _forward([100.0, 101.5, 99.0, 96.0], ev.ts)  # MAE ~2%
        out = label_event(ev, fwd, LabelConfig(good_mae_thresholds=(0.03, 0.05, 0.08)))
        self.assertEqual(out["good_mae_3"], 1)
        self.assertEqual(out["good_mae_5"], 1)

        fwd2 = _forward([100.0, 107.0, 99.0, 96.0], ev.ts)  # MAE ~7%
        out2 = label_event(ev, fwd2, LabelConfig(good_mae_thresholds=(0.03, 0.05, 0.08)))
        self.assertEqual(out2["good_mae_3"], 0)
        self.assertEqual(out2["good_mae_8"], 1)

    def test_labels_use_only_forward_data(self):
        """MAE/MFE must be measured from the event price, not the window minimum."""
        ev = self._event(entry=100.0)
        fwd = _forward([100.0, 104.0, 98.0], ev.ts)
        out = label_event(ev, fwd, LabelConfig())
        self.assertAlmostEqual(out["mae_pct"], (104.0 * 1.0005 - 100.0) / 100.0, places=4)
        self.assertAlmostEqual(out["mfe_pct"], (100.0 - 98.0 * 0.9995) / 100.0, places=4)


if __name__ == "__main__":
    unittest.main()
