from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai.inference.artifact_loader import ArtifactBundle
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


if __name__ == "__main__":
    unittest.main()

