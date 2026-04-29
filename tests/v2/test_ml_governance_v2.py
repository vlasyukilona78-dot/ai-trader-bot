from __future__ import annotations

import tempfile
import unittest
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import joblib
except Exception:
    joblib = None

try:
    import pandas as pd
except Exception:
    pd = None

from ai.inference.artifact_loader import ArtifactBundle, ArtifactContractError, validate_artifact_bundle
from ai.inference.governance import (
    compute_feature_schema_hash,
    load_registry,
    register_challenger,
    resolve_version_alias,
    rollback_champion,
)
from ai.inference.model_service import ModelService


class _DummyClassifier:
    def predict_proba(self, row):
        return [[0.2, 0.8]]


class _DummyRegressor:
    def predict(self, row):
        return [12.0]


class MlGovernanceV2Tests(unittest.TestCase):
    def test_feature_schema_hash_stable(self):
        features = ["a", "b", "c"]
        self.assertEqual(compute_feature_schema_hash(features), compute_feature_schema_hash(features))

    def test_registry_champion_challenger_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = str(Path(tmpdir))
            reg = load_registry(model_dir)
            self.assertEqual(reg.get("champion"), "default")
            register_challenger(model_dir, "v2")
            self.assertEqual(resolve_version_alias(model_dir, "challenger"), "v2")
            rollback_champion(model_dir, "default")
            self.assertEqual(resolve_version_alias(model_dir, "champion"), "default")

    def test_model_service_strict_feature_parity(self):
        features = ["f1", "f2"]
        bundle = ArtifactBundle(
            classifier=_DummyClassifier(),
            regressor=_DummyRegressor(),
            scaler=None,
            calibrator=None,
            feature_names=features,
            version="default",
            feature_schema_hash=compute_feature_schema_hash(features),
            manifest={},
        )
        service = ModelService(bundle, strict_schema=True)

        miss = service.predict({"f1": 1.0})
        self.assertFalse(miss.model_enabled)
        self.assertEqual(miss.reason, "feature_parity_missing")

        extra = service.predict({"f1": 1.0, "f2": 2.0, "extra": 3.0})
        self.assertFalse(extra.model_enabled)
        self.assertEqual(extra.reason, "feature_parity_extra")

        ok = service.predict({"f1": 1.0, "f2": 2.0})
        self.assertTrue(ok.model_enabled)
        self.assertGreater(ok.probability, 0.0)

    @unittest.skipIf(joblib is None, "joblib unavailable")
    def test_validate_artifact_bundle_requires_manifest_feature_schema_contract(self):
        features = ["f1", "f2"]
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            joblib.dump(_DummyClassifier(), model_dir / "model_win.pkl")
            joblib.dump(_DummyRegressor(), model_dir / "model_horizon.pkl")
            joblib.dump(object(), model_dir / "scaler.pkl")
            joblib.dump(features, model_dir / "features.pkl")
            (model_dir / "manifest.json").write_text(
                '{"feature_names":["f1","f2"],"feature_schema_hash":"bad"}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ArtifactContractError, "feature_schema_hash_mismatch"):
                validate_artifact_bundle(str(model_dir), version="default")

            (model_dir / "manifest.json").write_text(
                '{"feature_names":["f1","f2"],"feature_schema_hash":"%s"}'
                % compute_feature_schema_hash(features),
                encoding="utf-8",
            )

            bundle = validate_artifact_bundle(str(model_dir), version="default")
            self.assertEqual(bundle.feature_names, features)
            self.assertEqual(bundle.version, "default")

    @unittest.skipIf(joblib is None or pd is None, "training dependencies unavailable")
    def test_train_models_writes_quality_metrics_to_manifest_and_registry(self):
        from ai.train import train_models
        from ai.utils import DEFAULT_FEATURE_NAMES

        base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for idx in range(160):
            row = {
                "timestamp": (base_ts + timedelta(minutes=idx)).isoformat(),
                "target_win": int(idx % 2 == 0),
                "target_horizon": float(4 + (idx % 12)),
            }
            for feature_idx, name in enumerate(DEFAULT_FEATURE_NAMES):
                row[name] = float(((idx + feature_idx) % 19) / 10.0)
            rows.append(row)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "training.csv"
            model_dir = tmp / "models"
            pd.DataFrame(rows).to_csv(dataset, index=False)

            train_models(str(dataset), model_dir=str(model_dir), model_type="sklearn")

            manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
            metrics = manifest.get("metrics", {})
            self.assertEqual(metrics.get("train_rows"), 128)
            self.assertEqual(metrics.get("test_rows"), 32)
            self.assertIn("test_auc", metrics)
            self.assertIn("test_horizon_mae", metrics)
            self.assertIn("walk_forward_folds", metrics)

            registry = json.loads((model_dir / "registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["history"][-1]["action"], "register_artifact")
            self.assertIn("metrics", registry["history"][-1])


if __name__ == "__main__":
    unittest.main()

