from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def clamp(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    result = safe_float(value, default)
    return max(float(low), min(float(high), result))


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def mapping_get(mapping: Mapping[str, Any] | None, key: str) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return None


def layer_details(details: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(details, Mapping):
        return {}
    value = details.get(key)
    return dict(value) if isinstance(value, Mapping) else {}
