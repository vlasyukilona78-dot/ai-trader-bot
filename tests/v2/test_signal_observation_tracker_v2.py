from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading.state.signal_observation_tracker import SignalObservationTracker


class SignalObservationTrackerV2Tests(unittest.TestCase):
    def test_tracks_only_post_signal_bars_and_completes_locally(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.json"
            tracker = SignalObservationTracker(path, horizon_minutes=2)
            signal_bar = pd.Timestamp("2026-07-24T00:00:00Z")
            tracker.record_short(
                signal_id="sig-1",
                symbol="BTCUSDT",
                phase="WATCH",
                entry=100.0,
                take_profit=98.0,
                stop_loss=102.0,
                signal_ts=signal_bar.timestamp(),
                signal_bar_ts=signal_bar,
                delivered=True,
                candidate_source="pre_main",
            )
            frame = pd.DataFrame(
                {
                    "high": [105.0, 100.2, 102.2],
                    "low": [95.0, 99.0, 97.5],
                    "close": [99.0, 99.4, 98.0],
                },
                index=pd.date_range(signal_bar, periods=3, freq="1min"),
            )

            tracker.update_frame(
                "BTCUSDT",
                frame,
                observed_at=pd.Timestamp("2026-07-24T00:03:00Z").timestamp(),
            )

            self.assertEqual(tracker.active_count(), 0)
            completed_path = path.with_name("observations_completed.jsonl")
            completed = json.loads(completed_path.read_text(encoding="utf-8").strip())
            self.assertEqual(completed["bars_observed"], 2)
            self.assertAlmostEqual(completed["favorable_excursion_pct"], 2.5)
            self.assertAlmostEqual(completed["adverse_excursion_pct"], 2.2)
            self.assertTrue(completed["tp_hit"])
            self.assertTrue(completed["sl_hit"])
            self.assertIn("2", completed["horizon_metrics"])

    def test_active_observation_exposes_current_post_signal_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.json"
            tracker = SignalObservationTracker(path, horizon_minutes=90)
            signal_bar = pd.Timestamp("2026-07-24T00:00:00Z")
            tracker.record_short(
                signal_id="sig-active",
                symbol="BANKUSDT",
                phase="WATCH",
                entry=100.0,
                take_profit=94.0,
                stop_loss=103.0,
                signal_ts=signal_bar.timestamp(),
                signal_bar_ts=signal_bar,
                delivered=True,
            )
            frame = pd.DataFrame(
                {
                    "high": [100.0, 101.5, 100.0],
                    "low": [100.0, 99.0, 95.0],
                    "close": [100.0, 99.5, 96.0],
                },
                index=pd.date_range(signal_bar, periods=3, freq="1min"),
            )
            tracker.update_frame(
                "BANKUSDT",
                frame,
                observed_at=pd.Timestamp("2026-07-24T00:03:00Z").timestamp(),
            )

            active = tracker.active_observation(
                signal_id="sig-active",
                symbol="BANKUSDT",
            )

            self.assertIsNotNone(active)
            self.assertAlmostEqual(active["favorable_excursion_pct"], 5.0)
            self.assertAlmostEqual(active["adverse_excursion_pct"], 1.5)
            self.assertAlmostEqual(active["close_move_pct"], 4.0)

    def test_persists_active_observation_across_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.json"
            tracker = SignalObservationTracker(path)
            tracker.record_short(
                signal_id="sig-2",
                symbol="ETHUSDT",
                phase="SETUP",
                entry=200.0,
                take_profit=190.0,
                stop_loss=204.0,
                signal_ts=1000.0,
                signal_bar_ts=960.0,
                delivered=False,
            )

            restored = SignalObservationTracker(path)

            self.assertEqual(restored.active_count(), 1)
            duplicate = restored.record_short(
                signal_id="sig-2",
                symbol="ETHUSDT",
                phase="SETUP",
                entry=201.0,
                take_profit=190.0,
                stop_loss=204.0,
                signal_ts=1010.0,
                signal_bar_ts=960.0,
                delivered=True,
            )
            self.assertEqual(duplicate["entry"], 200.0)
            self.assertTrue(duplicate["delivered"])
            self.assertEqual(restored.active_count(), 1)

    def test_expires_incomplete_observation_when_symbol_leaves_runtime_universe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.json"
            tracker = SignalObservationTracker(path, horizon_minutes=2)
            signal_bar = pd.Timestamp("2026-07-24T00:00:00Z")
            tracker.record_short(
                signal_id="sig-missing-symbol",
                symbol="OLDUSDT",
                phase="WATCH",
                entry=100.0,
                take_profit=95.0,
                stop_loss=102.0,
                signal_ts=signal_bar.timestamp(),
                signal_bar_ts=signal_bar,
                delivered=True,
            )
            partial_frame = pd.DataFrame(
                {"high": [100.0, 100.5], "low": [100.0, 99.0], "close": [100.0, 99.5]},
                index=pd.date_range(signal_bar, periods=2, freq="1min"),
            )
            tracker.update_frame(
                "OLDUSDT",
                partial_frame,
                observed_at=pd.Timestamp("2026-07-24T00:01:30Z").timestamp(),
            )

            expired = tracker.expire_stale(
                observed_at=pd.Timestamp("2026-07-24T00:03:00Z").timestamp(),
            )

            self.assertEqual(expired, 1)
            self.assertEqual(tracker.active_count(), 0)
            completed_path = path.with_name("observations_completed.jsonl")
            completed = json.loads(completed_path.read_text(encoding="utf-8").strip())
            self.assertEqual(completed["status"], "expired_incomplete")
            self.assertFalse(completed["observation_complete"])
            self.assertEqual(
                completed["completion_reason"],
                "wall_clock_horizon_elapsed_without_full_market_data",
            )
            self.assertAlmostEqual(completed["coverage_ratio"], 0.5)
            self.assertNotIn("2", completed["horizon_metrics"])

    def test_completed_signal_id_is_not_reopened(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.json"
            tracker = SignalObservationTracker(path, horizon_minutes=1)
            signal_bar = pd.Timestamp("2026-07-24T00:00:00Z")
            tracker.record_short(
                signal_id="sig-completed",
                symbol="BTCUSDT",
                phase="ENTRY",
                entry=100.0,
                take_profit=98.0,
                stop_loss=102.0,
                signal_ts=signal_bar.timestamp(),
                signal_bar_ts=signal_bar,
                delivered=True,
            )
            frame = pd.DataFrame(
                {"high": [100.0, 100.1], "low": [100.0, 99.0], "close": [100.0, 99.5]},
                index=pd.date_range(signal_bar, periods=2, freq="1min"),
            )
            tracker.update_frame(
                "BTCUSDT",
                frame,
                observed_at=pd.Timestamp("2026-07-24T00:02:00Z").timestamp(),
            )

            restored = SignalObservationTracker(path, horizon_minutes=1)
            duplicate = restored.record_short(
                signal_id="sig-completed",
                symbol="BTCUSDT",
                phase="ENTRY",
                entry=100.0,
                take_profit=98.0,
                stop_loss=102.0,
                signal_ts=signal_bar.timestamp(),
                signal_bar_ts=signal_bar,
                delivered=True,
            )

            self.assertIsNone(duplicate)
            self.assertEqual(restored.active_count(), 0)


if __name__ == "__main__":
    unittest.main()
