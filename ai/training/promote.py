from __future__ import annotations

import os
from pathlib import Path

from ai.inference.governance import load_registry, rollback_champion, save_registry


class PromotionError(RuntimeError):
    pass


def _artifact_names(version: str) -> list[str]:
    suffix = "" if version == "default" else f"_{version}"
    return [
        f"model_win{suffix}.pkl",
        f"model_horizon{suffix}.pkl",
        f"scaler{suffix}.pkl",
        f"features{suffix}.pkl",
        f"calibrator{suffix}.pkl",
        "manifest.json" if version == "default" else f"manifest_{version}.json",
    ]


def promote_candidate_model(candidate_dir: str, production_dir: str):
    if os.getenv("ML_AUTO_PROMOTION_ENABLED", "false").lower() not in ("1", "true", "yes"):
        raise PromotionError("auto_promotion_disabled")

    src = Path(candidate_dir)
    dst = Path(production_dir)
    if not src.exists():
        raise PromotionError("candidate_dir_not_found")
    dst.mkdir(parents=True, exist_ok=True)

    for name in ("model_win.pkl", "model_horizon.pkl", "scaler.pkl", "features.pkl", "calibrator.pkl", "manifest.json"):
        src_file = src / name
        if src_file.exists():
            (dst / name).write_bytes(src_file.read_bytes())

    registry = load_registry(str(dst))
    registry.setdefault("history", []).append({"action": "promote_candidate_copy", "candidate_dir": str(src)})
    save_registry(str(dst), registry)


def promote_registered_challenger(model_dir: str):
    if os.getenv("ML_AUTO_PROMOTION_ENABLED", "false").lower() not in ("1", "true", "yes"):
        raise PromotionError("auto_promotion_disabled")

    registry = load_registry(model_dir)
    challenger = registry.get("challenger")
    if not challenger:
        raise PromotionError("challenger_not_set")
    current = str(registry.get("champion") or "default")
    registry["champion"] = str(challenger)
    registry.setdefault("history", []).append({"action": "promote_challenger", "from": current, "to": challenger})
    save_registry(model_dir, registry)


def rollback_model(model_dir: str, target_version: str):
    existing = load_registry(model_dir)
    current = str(existing.get("champion") or "default")
    candidate_files = _artifact_names(target_version)
    missing = [name for name in candidate_files if not (Path(model_dir) / name).exists()]
    if missing:
        raise PromotionError(f"rollback_version_missing_artifacts:{target_version}")
    rollback_champion(model_dir, target_version)
    registry = load_registry(model_dir)
    registry.setdefault("history", []).append({"action": "rollback_confirmed", "from": current, "to": target_version})
    save_registry(model_dir, registry)
