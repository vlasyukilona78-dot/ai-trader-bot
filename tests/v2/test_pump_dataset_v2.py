from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ai.pump_dataset import (
    BAR_SECONDS_1H,
    EventConfig,
    LabelConfig,
    PumpEvent,
    _closed_by,
    detect_events,
    label_event,
)


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


class NoLookAheadV2Tests(unittest.TestCase):
    """MEXC stamps a bar with its OPEN time, so filtering on the stamp alone lets
    a bar that is still forming at the decision moment into the feature set. Both
    the forward window and every higher timeframe must respect the close."""

    def test_only_bars_finished_by_the_decision_are_kept(self):
        # 4h bars stamped at 0, 14400, 28800; decision at 18000
        frame = pd.DataFrame({"time": [0, 14400, 28800], "close": [1.0, 2.0, 3.0]})
        kept = _closed_by(frame, 14400, 18000)
        # the 14400 bar closes at 28800, after the decision - it must be excluded
        self.assertEqual(list(kept["time"]), [0])

    def test_a_bar_closing_exactly_on_the_decision_is_included(self):
        frame = pd.DataFrame({"time": [0, 3600], "close": [1.0, 2.0]})
        kept = _closed_by(frame, 3600, 7200)
        self.assertEqual(list(kept["time"]), [0, 3600])

    def test_empty_and_missing_frames_are_handled(self):
        self.assertTrue(_closed_by(pd.DataFrame(), 3600, 100).empty)
        self.assertTrue(_closed_by(None, 3600, 100).empty)

    def test_decision_lags_the_event_stamp_by_one_bar(self):
        self.assertEqual(BAR_SECONDS_1H, 3600)

    def test_time_to_target_is_measured_from_the_decision_not_the_stamp(self):
        ev = PumpEvent(symbol="T", ts=1_700_000_000, entry=100.0, move_pct=0.1, run_up_bars=3)
        # forward frame starts one hour after the stamp, target hit on its 3rd bar
        start = ev.ts + BAR_SECONDS_1H
        fwd = pd.DataFrame({
            "time": [start, start + 300, start + 600],
            "open": [100.0, 99.0, 96.0], "high": [100.0, 99.0, 96.0],
            "low": [100.0, 99.0, 96.0], "close": [100.0, 99.0, 96.0],
            "volume": [1.0, 1.0, 1.0],
        })
        out = label_event(ev, fwd, LabelConfig(dca_target_pct=0.03))
        # 10 minutes after the decision, not 70 minutes after the stamp
        self.assertEqual(out["time_to_target_min"], 10)


if __name__ == "__main__":
    unittest.main()
