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
    from core.feature_engineering import REQUIRED_MODEL_FEATURES
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


if __name__ == "__main__":
    unittest.main()
