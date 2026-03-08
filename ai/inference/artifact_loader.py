from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import joblib
except Exception:
    joblib = None

from ai.inference.governance import compute_feature_schema_hash, resolve_version_alias


@dataclass
class ArtifactBundle:
    classifier: Any | None
    regressor: Any | None
    scaler: Any | None
    calibrator: Any | None
    feature_names: list[str]
    version: str
    feature_schema_hash: str
    manifest: dict[str, Any]



def _safe_load(path: Path):
    if joblib is None:
        return None
    try:
        if path.exists():
            return joblib.load(path)
    except Exception:
        return None
    return None


def _load_manifest(root: Path, version: str) -> dict[str, Any]:
    suffix = "" if version == "default" else f"_{version}"
    path = root / f"manifest{suffix}.json"
    if not path.exists() and version == "default":
        path = root / "manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_artifacts(model_dir: str, version: str = "champion") -> ArtifactBundle:
    root = Path(model_dir)
    actual_version = resolve_version_alias(model_dir, version)
    suffix = "" if actual_version == "default" else f"_{actual_version}"

    clf = _safe_load(root / f"model_win{suffix}.pkl")
    reg = _safe_load(root / f"model_horizon{suffix}.pkl")
    scaler = _safe_load(root / f"scaler{suffix}.pkl")
    calibrator = _safe_load(root / f"calibrator{suffix}.pkl")
    feature_names = _safe_load(root / f"features{suffix}.pkl")
    if not isinstance(feature_names, list):
        feature_names = []

    manifest = _load_manifest(root, actual_version)
    schema_hash = str(manifest.get("feature_schema_hash") or "")
    if not schema_hash:
        schema_hash = compute_feature_schema_hash(feature_names)

    return ArtifactBundle(
        classifier=clf,
        regressor=reg,
        scaler=scaler,
        calibrator=calibrator,
        feature_names=feature_names,
        version=actual_version,
        feature_schema_hash=schema_hash,
        manifest=manifest,
    )
