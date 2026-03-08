from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

from ai.inference.artifact_loader import ArtifactBundle
from ai.inference.governance import compute_feature_schema_hash


@dataclass
class InferenceResult:
    probability: float
    horizon: float
    model_enabled: bool
    reason: str = "ok"


class ModelService:
    def __init__(self, artifacts: ArtifactBundle | None, strict_schema: bool = True):
        self.artifacts = artifacts
        self.strict_schema = bool(strict_schema)

    def _build_row(self, names: list[str], features: dict[str, float]):
        values = [float(features.get(name, 0.0)) for name in names]
        if np is None:
            return [values]
        return np.array([values], dtype=float)

    def _finite(self, row) -> bool:
        if np is not None:
            return bool(np.isfinite(row).all())
        for item in row[0]:
            if not math.isfinite(float(item)):
                return False
        return True

    def predict(self, features: dict[str, float]) -> InferenceResult:
        if self.artifacts is None or self.artifacts.classifier is None or self.artifacts.regressor is None:
            return InferenceResult(probability=0.5, horizon=8.0, model_enabled=False, reason="model_disabled")

        names = self.artifacts.feature_names or sorted(features.keys())
        if self.strict_schema and self.artifacts.feature_names:
            expected_names = [str(name) for name in self.artifacts.feature_names]
            runtime_names = sorted(str(name) for name in features.keys() if str(name))
            missing = [name for name in expected_names if name not in features]
            unexpected = [name for name in runtime_names if name not in expected_names]
            if missing:
                return InferenceResult(probability=0.5, horizon=8.0, model_enabled=False, reason="feature_parity_missing")
            if unexpected:
                return InferenceResult(probability=0.5, horizon=8.0, model_enabled=False, reason="feature_parity_extra")
            expected_hash = self.artifacts.feature_schema_hash or compute_feature_schema_hash(expected_names)
            actual_hash = compute_feature_schema_hash(expected_names)
            if expected_hash and expected_hash != actual_hash:
                return InferenceResult(probability=0.5, horizon=8.0, model_enabled=False, reason="feature_schema_mismatch")

        row = self._build_row(names, features)
        if not self._finite(row):
            return InferenceResult(probability=0.5, horizon=8.0, model_enabled=False, reason="non_finite_features")

        if self.artifacts.scaler is not None:
            if pd is not None and np is not None:
                frame = pd.DataFrame(row, columns=names)
                row = self.artifacts.scaler.transform(frame)
            else:
                row = self.artifacts.scaler.transform(row)

        prob = float(self.artifacts.classifier.predict_proba(row)[0][-1]) if hasattr(self.artifacts.classifier, "predict_proba") else float(self.artifacts.classifier.predict(row)[0])
        horizon = float(self.artifacts.regressor.predict(row)[0])
        if self.artifacts.calibrator is not None and hasattr(self.artifacts.calibrator, "transform"):
            try:
                prob = float(self.artifacts.calibrator.transform([prob])[0])
            except Exception:
                pass

        prob = float(max(0.0, min(1.0, prob)))
        horizon = float(max(1.0, min(240.0, horizon)))
        return InferenceResult(probability=prob, horizon=horizon, model_enabled=True, reason="ok")

