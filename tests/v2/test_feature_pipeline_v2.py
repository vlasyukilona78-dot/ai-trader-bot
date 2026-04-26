from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY_PANDAS = True
except Exception:
    np = None
    pd = None
    HAS_NUMPY_PANDAS = False

if HAS_NUMPY_PANDAS:
    from core.feature_engineering import (
        REQUIRED_MODEL_FEATURES,
        assess_feature_frame_quality,
        compute_mtf_feature_snapshot,
        sanitize_feature_frame,
    )
    from trading.features.pipeline import FeaturePipeline
    from trading.features.validators import FeatureValidationError, assert_no_future_rows
else:
    REQUIRED_MODEL_FEATURES = []


@unittest.skipUnless(HAS_NUMPY_PANDAS, "numpy/pandas are not installed")
class FeaturePipelineV2Tests(unittest.TestCase):
    def _build_df(self, n: int = 220) -> pd.DataFrame:
        idx = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
        close = np.linspace(100.0, 120.0, n)
        return pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.4,
                "low": close - 0.6,
                "close": close,
                "volume": np.linspace(10.0, 20.0, n),
            },
            index=idx,
        )

    def test_no_leakage_guard(self):
        df = self._build_df()
        with self.assertRaises(FeatureValidationError):
            assert_no_future_rows(df, df.index[-2])

    def test_train_inference_parity_and_no_nans(self):
        df = self._build_df()
        pipe = FeaturePipeline()
        bundle = pipe.build(symbol="BTCUSDT", ohlcv=df, as_of=df.index[-1], extras={"sentiment_index": 55.0})
        self.assertTrue(set(REQUIRED_MODEL_FEATURES).issubset(set(bundle.row.values.keys())))
        for name in REQUIRED_MODEL_FEATURES:
            self.assertTrue(np.isfinite(float(bundle.row.values[name])))

    def test_compute_mtf_feature_snapshot_populates_one_hour_context(self):
        df = self._build_df(n=2200)

        snapshot = compute_mtf_feature_snapshot(df)

        self.assertIn("mtf_rsi_1h", snapshot)
        self.assertIn("mtf_atr_norm_1h", snapshot)
        self.assertIn("mtf_trend_1h", snapshot)
        self.assertTrue(np.isfinite(float(snapshot["mtf_rsi_1h"])))
        self.assertTrue(np.isfinite(float(snapshot["mtf_atr_norm_1h"])))
        self.assertTrue(np.isfinite(float(snapshot["mtf_trend_1h"])))

    def test_sanitize_feature_frame_repairs_inf_and_duplicate_rows(self):
        df = self._build_df(n=16)
        broken = pd.concat([df.iloc[:8], df.iloc[7:]], axis=0)
        broken["rsi"] = np.nan
        broken["volume_spike"] = np.inf
        broken["hist"] = np.nan
        out = sanitize_feature_frame(broken)

        self.assertTrue(out.index.is_monotonic_increasing)
        self.assertEqual(len(out.index.unique()), len(out))
        self.assertTrue(np.isfinite(float(out["volume_spike"].iloc[-1])))
        self.assertTrue(np.isfinite(float(out["rsi"].iloc[-1])))
        self.assertTrue(np.isfinite(float(out["hist"].iloc[-1])))

    def test_assess_feature_frame_quality_blocks_severe_recent_gap_cluster(self):
        df = self._build_df(n=48)
        broken = df.drop(df.index[[30, 31, 40, 41]])
        out = assess_feature_frame_quality(broken, recent_window=24, severe_gap_multiple=3.0, max_recent_gap_ratio=0.05)

        self.assertFalse(bool(out["usable"]))
        self.assertIn(str(out["reason"]), {"latest_gap_exceeds_threshold", "recent_gap_cluster"})
        self.assertGreater(float(out["recent_severe_gap_count"]), 0.0)

    def test_assess_feature_frame_quality_blocks_recent_zero_volume_cluster(self):
        df = self._build_df(n=48)
        df.loc[df.index[-6:], "volume"] = 0.0

        out = assess_feature_frame_quality(
            df,
            recent_window=24,
            max_recent_zero_volume_ratio=0.10,
            min_recent_zero_volume_count=3,
        )

        self.assertFalse(bool(out["usable"]))
        self.assertEqual(str(out["reason"]), "recent_zero_volume_cluster")
        self.assertGreaterEqual(float(out["recent_zero_volume_count"]), 3.0)
        self.assertGreater(float(out["recent_zero_volume_ratio"]), 0.10)


if __name__ == "__main__":
    unittest.main()
