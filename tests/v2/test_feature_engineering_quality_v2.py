import unittest

import pandas as pd

from core.feature_engineering import assess_feature_frame_quality, sanitize_feature_frame


class FeatureEngineeringQualityV2Tests(unittest.TestCase):
    def test_sanitize_drops_nonpositive_prices_and_repairs_ohlc_bounds(self):
        index = pd.date_range("2026-01-01 00:00:00", periods=3, freq="min")
        frame = pd.DataFrame(
            {
                "open": [1.00, 0.00, 1.05],
                "high": [0.95, 1.10, 1.02],
                "low": [0.99, 0.90, 1.10],
                "close": [1.02, 1.00, 1.03],
                "volume": [10.0, 5.0, -4.0],
                "rsi": [150.0, 50.0, -20.0],
                "volume_spike": [100.0, 1.0, -3.0],
            },
            index=index,
        )

        sanitized = sanitize_feature_frame(frame)

        self.assertEqual(len(sanitized), 2)
        self.assertTrue((sanitized[["open", "high", "low", "close"]] > 0.0).all().all())
        self.assertTrue(
            (sanitized["high"] >= sanitized[["open", "close", "low"]].max(axis=1)).all()
        )
        self.assertTrue(
            (sanitized["low"] <= sanitized[["open", "close", "high"]].min(axis=1)).all()
        )
        self.assertTrue((sanitized["volume"] >= 0.0).all())
        self.assertTrue(sanitized["rsi"].between(0.0, 100.0).all())
        self.assertTrue(sanitized["volume_spike"].between(0.0, 30.0).all())

    def test_assess_feature_frame_quality_blocks_latest_gap(self):
        base = pd.Timestamp("2026-01-01 00:00:00")
        index = [base + pd.Timedelta(minutes=i) for i in range(20)]
        index.append(base + pd.Timedelta(minutes=30))
        frame = pd.DataFrame(
            {
                "open": 1.0,
                "high": 1.01,
                "low": 0.99,
                "close": 1.0,
                "volume": 10.0,
            },
            index=pd.DatetimeIndex(index),
        )

        quality = assess_feature_frame_quality(frame)

        self.assertFalse(bool(quality["usable"]))
        self.assertEqual(quality["reason"], "latest_gap_exceeds_threshold")


if __name__ == "__main__":
    unittest.main()
