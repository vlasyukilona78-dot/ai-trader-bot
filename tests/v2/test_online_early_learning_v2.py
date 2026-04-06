from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ai.online_early_learning import EarlySignalOutcomeLearner


class EarlySignalOutcomeLearnerV2Tests(unittest.TestCase):
    def test_resolves_early_signal_into_profile_dataset_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "online_training_dataset_early.csv"
            pending_path = Path(tmp) / "pending_early.json"
            learner = EarlySignalOutcomeLearner(
                dataset_path=str(dataset_path),
                pending_path=str(pending_path),
                timeframe_minutes=1,
            )

            signal_bar_ts = pd.Timestamp("2026-04-05 00:00:00", tz="UTC")
            features = {"rsi": 82.0, "volume_spike": 1.9, "liq_low_dist": -0.014}
            learner.record_signal(
                symbol="HIPPOUSDT",
                phase="WATCH",
                market_regime="PUMP",
                signal_price=1.0,
                signal_ts=signal_bar_ts.timestamp(),
                signal_bar_ts=signal_bar_ts,
                features=features,
                horizon_bars=6,
                success_move_pct=0.02,
                failure_move_pct=0.015,
            )

            index = pd.date_range(signal_bar_ts, periods=8, freq="1min", tz="UTC")
            frame = pd.DataFrame(
                {
                    "open": [1.0, 0.999, 0.995, 0.989, 0.985, 0.980, 0.983, 0.987],
                    "high": [1.0, 1.006, 1.004, 0.996, 0.990, 0.986, 0.990, 0.992],
                    "low": [0.999, 0.992, 0.986, 0.980, 0.976, 0.974, 0.978, 0.982],
                    "close": [1.0, 0.995, 0.989, 0.985, 0.980, 0.982, 0.988, 0.989],
                },
                index=index,
            )

            row = learner.resolve_with_frame(symbol="HIPPOUSDT", enriched=frame)

            self.assertIsNotNone(row)
            self.assertTrue(dataset_path.exists())
            saved = pd.read_csv(dataset_path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.iloc[0]["signal_family"], "early")
            self.assertEqual(saved.iloc[0]["signal_phase"], "WATCH")
            self.assertEqual(int(saved.iloc[0]["target_win"]), 1)
            self.assertGreater(float(saved.iloc[0]["future_return"]), 0.0)


if __name__ == "__main__":
    unittest.main()
