from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def compute_feature_schema_hash(feature_names: list[str]) -> str:
    normalized = [str(x).strip() for x in feature_names if str(x).strip()]
    payload = "\n".join(normalized)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _registry_path(model_dir: str) -> Path:
    return Path(model_dir) / "registry.json"


def load_registry(model_dir: str) -> dict:
    path = _registry_path(model_dir)
    if not path.exists():
        return {
            "champion": "default",
            "challenger": None,
            "auto_promotion": False,
            "history": [],
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("invalid_registry")
        data.setdefault("champion", "default")
        data.setdefault("challenger", None)
        data.setdefault("auto_promotion", False)
        data.setdefault("history", [])
        return data
    except Exception:
        return {
            "champion": "default",
            "challenger": None,
            "auto_promotion": False,
            "history": [],
        }


def save_registry(model_dir: str, registry: dict):
    path = _registry_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_version_alias(model_dir: str, requested_version: str) -> str:
    requested = str(requested_version or "default").strip().lower()
    if requested not in ("champion", "challenger"):
        return requested_version or "default"
    registry = load_registry(model_dir)
    value = registry.get(requested)
    if not value:
        return "default"
    return str(value)


def register_challenger(model_dir: str, challenger_version: str):
    registry = load_registry(model_dir)
    registry["challenger"] = challenger_version
    registry.setdefault("history", []).append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": "register_challenger",
            "version": challenger_version,
        }
    )
    save_registry(model_dir, registry)


def promote_challenger(model_dir: str, *, require_explicit: bool = True):
    registry = load_registry(model_dir)
    challenger = registry.get("challenger")
    if not challenger:
        raise RuntimeError("challenger_not_set")
    if require_explicit and not bool(registry.get("auto_promotion", False)):
        raise RuntimeError("auto_promotion_disabled")

    previous = registry.get("champion", "default")
    registry["champion"] = challenger
    registry.setdefault("history", []).append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": "promote_challenger",
            "from": previous,
            "to": challenger,
        }
    )
    save_registry(model_dir, registry)


def rollback_champion(model_dir: str, target_version: str):
    registry = load_registry(model_dir)
    previous = registry.get("champion", "default")
    registry["champion"] = target_version
    registry.setdefault("history", []).append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": "rollback",
            "from": previous,
            "to": target_version,
        }
    )
    save_registry(model_dir, registry)
