from __future__ import annotations

import unittest

try:
    import pandas as pd
except Exception:
    pd = None


@unittest.skipIf(pd is None, "pandas unavailable")
class MlDatasetSideLabelsV2Tests(unittest.TestCase):
    def test_directional_outcome_labels_short_opposite_to_long(self):
        from ai.build_dataset import _simulate_directional_outcome

        future = pd.DataFrame(
            [
                {"high": 100.4, "low": 97.9, "close": 98.2},
                {"high": 99.0, "low": 97.5, "close": 98.0},
            ]
        )

        short_win, short_horizon, short_return = _simulate_directional_outcome(
            future=future,
            side="SHORT",
            entry=100.0,
            atr=1.0,
            atr_mult=1.0,
            rr=2.0,
            lookahead=2,
        )
        long_win, long_horizon, long_return = _simulate_directional_outcome(
            future=future,
            side="LONG",
            entry=100.0,
            atr=1.0,
            atr_mult=1.0,
            rr=2.0,
            lookahead=2,
        )

        self.assertEqual(short_win, 1)
        self.assertEqual(short_horizon, 1.0)
        self.assertGreater(short_return, 0.0)
        self.assertEqual(long_win, 0)
        self.assertEqual(long_horizon, 1.0)
        self.assertLess(long_return, 0.0)

    def test_training_validation_blocks_future_feature_columns(self):
        from ai.training.validate import TrainingValidationError, validate_no_feature_leakage

        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
                "target_win": [1, 0, 1],
                "target_horizon": [3.0, 4.0, 5.0],
                "future_return": [0.02, -0.01, 0.03],
                "rsi": [70.0, 75.0, 80.0],
                "future_high_5": [1.0, 1.1, 1.2],
            }
        )

        with self.assertRaisesRegex(TrainingValidationError, "feature_leakage_column:future_high_5"):
            validate_no_feature_leakage(frame)


if __name__ == "__main__":
    unittest.main()
