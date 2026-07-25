from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trading.state.signal_position_tracker import SignalPositionTracker


class SignalPositionTrackerV2Tests(unittest.TestCase):
    def test_tracks_best_price_and_persists_active_short(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "signal_positions.json"
            tracker = SignalPositionTracker(path)
            tracker.record_short(
                symbol="BTC/USDT",
                entry_price=100.0,
                stop_loss=102.0,
                take_profit=94.0,
                opened_at=1000.0,
                pump_id="pump-1",
            )
            tracker.update_mark("BTCUSDT", 96.0, updated_at=1100.0)
            tracker.update_mark("BTCUSDT", 98.0, updated_at=1200.0)

            restored = SignalPositionTracker(path).active("BTCUSDT")

            self.assertIsNotNone(restored)
            self.assertEqual(restored["best_price"], 96.0)
            self.assertEqual(restored["last_price"], 98.0)
            self.assertEqual(restored["pump_id"], "pump-1")

    def test_close_records_underlying_short_return(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "signal_positions.json"
            tracker = SignalPositionTracker(path)
            tracker.record_short(
                symbol="ETHUSDT",
                entry_price=100.0,
                stop_loss=102.0,
                take_profit=94.0,
                opened_at=1000.0,
            )
            closed = tracker.close(
                "ETHUSDT",
                exit_price=95.0,
                reason="support_rebound",
                closed_at=1300.0,
            )

            self.assertIsNotNone(closed)
            self.assertAlmostEqual(closed["price_return"], 0.05)
            self.assertIsNone(tracker.active("ETHUSDT"))
            event_rows = [
                json.loads(line)
                for line in path.with_name("signal_positions_events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["event"] for row in event_rows], ["OPEN_SHORT", "CLOSE_SHORT"])

    def test_manual_replace_corrects_signal_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "signal_positions.json"
            tracker = SignalPositionTracker(path)
            tracker.record_short(
                symbol="SOLUSDT",
                entry_price=100.0,
                stop_loss=102.0,
                take_profit=94.0,
            )
            tracker.record_short(
                symbol="SOLUSDT",
                entry_price=101.0,
                stop_loss=103.0,
                take_profit=95.0,
                source="manual_cli",
                replace=True,
            )

            corrected = tracker.active("SOLUSDT")

            self.assertEqual(corrected["entry_price"], 101.0)
            self.assertEqual(corrected["source"], "manual_cli")


if __name__ == "__main__":
    unittest.main()
