from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ai.online_demo_learning import DEFAULT_FEATURE_NAMES, DemoTradeLearner


class DemoTradeLearnerV2Tests(unittest.TestCase):
    def test_closed_demo_trade_appends_online_dataset_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "online_training_dataset.csv"
            pending_path = Path(tmp) / "pending.json"
            learner = DemoTradeLearner(
                dataset_path=str(dataset_path),
                pending_path=str(pending_path),
                timeframe_minutes=5,
            )

            features = {name: 0.0 for name in DEFAULT_FEATURE_NAMES}
            features["rsi"] = 71.5
            features["volume_spike"] = 2.4
            features["liq_high_dist"] = -0.012

            learner.record_entry(
                symbol="BTCUSDT",
                side="SHORT",
                market_regime="PUMP",
                entry_price=100.0,
                qty=2.0,
                entry_ts=1_700_000_000.0,
                features=features,
            )

            row = learner.record_exit(
                symbol="BTCUSDT",
                exit_ts=1_700_000_600.0,
                realized_pnl=12.0,
                qty=2.0,
            )

            self.assertIsNotNone(row)
            self.assertTrue(dataset_path.exists())

            frame = pd.read_csv(dataset_path)
            self.assertEqual(len(frame), 1)
            self.assertEqual(int(frame.iloc[0]["target_win"]), 1)
            self.assertGreater(float(frame.iloc[0]["future_return"]), 0.0)
            self.assertGreater(float(frame.iloc[0]["target_horizon"]), 0.0)
            self.assertEqual(frame.iloc[0]["market_regime"], "PUMP")
            self.assertAlmostEqual(float(frame.iloc[0]["volume_spike"]), 2.4, places=6)


if __name__ == "__main__":
    unittest.main()
