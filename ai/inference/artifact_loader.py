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


ModelArtifacts = ArtifactBundle


class ArtifactContractError(RuntimeError):
    pass


def _artifact_path(root: Path, stem: str, version: str, suffix: str, ext: str = ".pkl") -> Path:
    if stem == "manifest":
        return root / ("manifest.json" if version == "default" else f"manifest_{version}.json")
    return root / f"{stem}{suffix}{ext}"


def _safe_load(path: Path, *, strict: bool = False, artifact_name: str = ""):
    if joblib is None:
        if strict:
            raise ArtifactContractError("joblib_unavailable")
        return None
    try:
        if path.exists():
            return joblib.load(path)
        if strict:
            raise ArtifactContractError(f"missing_artifact:{artifact_name or path.name}")
    except Exception:
        if strict:
            raise ArtifactContractError(f"invalid_artifact:{artifact_name or path.name}")
        return None
    return None


def _load_manifest(root: Path, version: str, *, strict: bool = False) -> dict[str, Any]:
    suffix = "" if version == "default" else f"_{version}"
    path = _artifact_path(root, "manifest", version, suffix, ext=".json")
    if not path.exists():
        if strict:
            raise ArtifactContractError("missing_artifact:manifest")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if strict and not isinstance(data, dict):
            raise ArtifactContractError("invalid_artifact:manifest")
        return data if isinstance(data, dict) else {}
    except ArtifactContractError:
        raise
    except Exception as exc:
        if strict:
            raise ArtifactContractError("invalid_artifact:manifest") from exc
        return {}


def _validate_contract(bundle: ArtifactBundle):
    if bundle.classifier is None:
        raise ArtifactContractError("missing_artifact:model_win")
    if bundle.regressor is None:
        raise ArtifactContractError("missing_artifact:model_horizon")
    if bundle.scaler is None:
        raise ArtifactContractError("missing_artifact:scaler")
    if not bundle.feature_names:
        raise ArtifactContractError("missing_artifact:features")

    manifest_features = bundle.manifest.get("feature_names")
    if not isinstance(manifest_features, list) or not manifest_features:
        raise ArtifactContractError("manifest_missing_feature_names")
    normalized_manifest_features = [str(name) for name in manifest_features]
    normalized_artifact_features = [str(name) for name in bundle.feature_names]
    if normalized_manifest_features != normalized_artifact_features:
        raise ArtifactContractError("feature_schema_names_mismatch")

    expected_hash = str(bundle.manifest.get("feature_schema_hash") or "")
    actual_hash = compute_feature_schema_hash(normalized_artifact_features)
    if not expected_hash:
        raise ArtifactContractError("manifest_missing_feature_schema_hash")
    if expected_hash != actual_hash:
        raise ArtifactContractError("feature_schema_hash_mismatch")


def load_artifacts(model_dir: str, version: str = "champion", *, strict: bool = False) -> ArtifactBundle:
    root = Path(model_dir)
    actual_version = resolve_version_alias(model_dir, version)
    suffix = "" if actual_version == "default" else f"_{actual_version}"

    clf = _safe_load(_artifact_path(root, "model_win", actual_version, suffix), strict=strict, artifact_name="model_win")
    reg = _safe_load(
        _artifact_path(root, "model_horizon", actual_version, suffix),
        strict=strict,
        artifact_name="model_horizon",
    )
    scaler = _safe_load(_artifact_path(root, "scaler", actual_version, suffix), strict=strict, artifact_name="scaler")
    calibrator = _safe_load(_artifact_path(root, "calibrator", actual_version, suffix), artifact_name="calibrator")
    feature_names = _safe_load(
        _artifact_path(root, "features", actual_version, suffix),
        strict=strict,
        artifact_name="features",
    )
    if not isinstance(feature_names, list):
        if strict:
            raise ArtifactContractError("invalid_artifact:features")
        feature_names = []

    manifest = _load_manifest(root, actual_version, strict=strict)
    schema_hash = str(manifest.get("feature_schema_hash") or "")
    if not schema_hash:
        schema_hash = compute_feature_schema_hash(feature_names)

    bundle = ArtifactBundle(
        classifier=clf,
        regressor=reg,
        scaler=scaler,
        calibrator=calibrator,
        feature_names=feature_names,
        version=actual_version,
        feature_schema_hash=schema_hash,
        manifest=manifest,
    )
    if strict:
        _validate_contract(bundle)
    return bundle


def validate_artifact_bundle(model_dir: str, version: str = "champion") -> ArtifactBundle:
    return load_artifacts(model_dir, version=version, strict=True)
