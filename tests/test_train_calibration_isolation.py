"""The calibrator must be fitted on its own interval, never on the test rows.

Fitting the isotonic calibrator on the same rows that produce the reported AUC
leaves no held-out interval to measure calibration quality on, and makes the
reported reliability optimistic by construction.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ai.train import train_models
from ai.utils import DEFAULT_FEATURE_NAMES


def _synthetic_dataset(path: Path, rows: int = 800) -> None:
    """Write a dataset whose labels are learnable but not perfectly separable."""

    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {name: rng.normal(size=rows) for name in DEFAULT_FEATURE_NAMES}
    )
    signal = frame["rsi"] + 0.5 * frame["atr"]
    noise = rng.normal(scale=1.0, size=rows)
    frame["target_win"] = (signal + noise > 0).astype(int)
    frame["target_horizon"] = rng.integers(1, 6, size=rows).astype(float)
    frame["timestamp"] = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    frame.to_csv(path, index=False)


class CalibrationIsolationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = self.root / "dataset.csv"
        self.model_dir = self.root / "models"
        _synthetic_dataset(self.dataset)

    def tearDown(self):
        self._tmp.cleanup()

    def _train(self):
        # Unequal calibration and test shares, so fitting on the wrong interval
        # produces a different row count and the assertion below catches it.
        return train_models(
            dataset_path=str(self.dataset),
            model_dir=str(self.model_dir),
            model_type="gbdt",
            train_frac=0.60,
            calib_frac=0.25,
            embargo=2,
        )

    def test_calibrator_is_fitted_on_the_calibration_interval(self):
        summary = self._train()

        self.assertGreater(summary["calibration_rows"], 0)
        self.assertNotEqual(summary["calibration_rows"], summary["test_rows"])
        self.assertEqual(summary["calibration_fit_rows"], summary["calibration_rows"])

    def test_metrics_are_reported_on_the_test_interval(self):
        summary = self._train()

        self.assertEqual(summary["metrics_rows"], summary["test_rows"])

    def test_split_composition_is_recorded_in_the_manifest(self):
        self._train()

        manifest = pd.read_json(self.model_dir / "manifest.json", typ="series")
        split = manifest["split"]

        self.assertEqual(split["embargo"], 2)
        self.assertGreater(split["train_rows"], 0)
        self.assertGreater(split["calibration_rows"], 0)
        self.assertGreater(split["test_rows"], 0)

    def test_purged_rows_are_reported(self):
        summary = self._train()

        # Horizons run 1..5 and the embargo is 2, so rows close to each boundary
        # must be dropped rather than silently kept.
        self.assertGreater(summary["purged_train"], 0)
        self.assertGreater(summary["purged_calibration"], 0)


if __name__ == "__main__":
    unittest.main()
