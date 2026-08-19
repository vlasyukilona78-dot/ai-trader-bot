"""Training must not turn absent feature values into zeros.

The dataset builder writes 0.0 for any feature it could not compute, and the
old training path filled remaining gaps with 0.0 as well. For divergence and
spike features zero is a meaningful reading, so the model was being taught that
"not measured" and "measured as neutral" are the same event.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ai.train import _prepare_xy, train_models
from ai.utils import DEFAULT_FEATURE_NAMES


class PrepareXyTests(unittest.TestCase):
    def test_absent_values_are_preserved_not_zeroed(self):
        frame = pd.DataFrame(
            {
                "rsi": [1.0, np.nan, 3.0],
                "target_win": [1, 0, 1],
                "target_horizon": [4.0, 4.0, 4.0],
            }
        )

        X, _, _ = _prepare_xy(frame, ["rsi"])

        self.assertTrue(np.isnan(X.loc[1, "rsi"]))

    def test_a_column_missing_from_the_dataset_is_not_invented_as_zero(self):
        frame = pd.DataFrame(
            {
                "rsi": [1.0, 2.0],
                "target_win": [1, 0],
                "target_horizon": [4.0, 4.0],
            }
        )

        X, _, _ = _prepare_xy(frame, ["rsi", "atr"])

        self.assertTrue(X["atr"].isna().all())

    def test_infinities_are_treated_as_absent(self):
        frame = pd.DataFrame(
            {
                "rsi": [1.0, np.inf],
                "target_win": [1, 0],
                "target_horizon": [4.0, 4.0],
            }
        )

        X, _, _ = _prepare_xy(frame, ["rsi"])

        self.assertTrue(np.isnan(X.loc[1, "rsi"]))


def _dataset_with_gaps(path: Path, rows: int = 800) -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({name: rng.normal(size=rows) for name in DEFAULT_FEATURE_NAMES})
    # Punch holes into one feature so the policy has something to learn.
    holes = rng.random(rows) < 0.20
    frame.loc[holes, "cvd_div_short"] = np.nan
    signal = frame["rsi"] + 0.5 * frame["atr"]
    frame["target_win"] = (signal + rng.normal(size=rows) > 0).astype(int)
    frame["target_horizon"] = rng.integers(1, 6, size=rows).astype(float)
    frame["timestamp"] = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    frame.to_csv(path, index=False)


class TrainingWiringTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dataset = self.root / "dataset.csv"
        self.model_dir = self.root / "models"
        _dataset_with_gaps(self.dataset)
        self.summary = train_models(
            dataset_path=str(self.dataset),
            model_dir=str(self.model_dir),
            model_type="sklearn",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_rates_are_reported(self):
        rates = self.summary["train_missing_rate"]

        self.assertGreater(rates["cvd_div_short"], 0.10)
        self.assertLess(rates["cvd_div_short"], 0.30)

    def test_a_fully_present_feature_reports_no_gaps(self):
        self.assertAlmostEqual(self.summary["train_missing_rate"]["rsi"], 0.0)

    def test_the_missingness_policy_is_saved_for_inference(self):
        self.assertTrue((self.model_dir / "missing_policy.pkl").exists())

    def test_the_runtime_feature_contract_is_unchanged(self):
        # Callers still supply the base features; indicators are internal.
        names = pd.read_json(self.model_dir / "manifest.json", typ="series")["feature_names"]

        self.assertNotIn("cvd_div_short__missing", names)
        self.assertIn("cvd_div_short", names)


class OodEnvelopeArtifactTests(TrainingWiringTests):
    """The support envelope is fitted on training rows and shipped with the model."""

    def test_the_envelope_is_saved(self):
        self.assertTrue((self.model_dir / "ood_envelope.pkl").exists())

    def test_the_envelope_version_is_reported(self):
        self.assertTrue(self.summary["ood_envelope_version"])

    def test_the_envelope_covers_every_trained_feature(self):
        import joblib

        envelope = joblib.load(self.model_dir / "ood_envelope.pkl")
        names = pd.read_json(self.model_dir / "manifest.json", typ="series")["feature_names"]

        self.assertEqual(set(envelope.bounds), set(names))


if __name__ == "__main__":
    unittest.main()
