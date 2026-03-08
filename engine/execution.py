from __future__ import annotations

import os

if os.getenv("ALLOW_LEGACY_RUNTIME", "false").strip().lower() not in ("1", "true", "yes"):
    raise RuntimeError("Legacy runtime is quarantined. Use V2 entrypoint app/main.py and trading/* modules.")

import time


def build_signal_id(symbol: str) -> str:
    base = (symbol or "").replace("/", "").upper()
    return f"{base}-{int(time.time() * 1000)}"


def order_succeeded(order_result: dict | None) -> bool:
    return isinstance(order_result, dict) and order_result.get("retCode", 0) == 0


def extract_order_id(order_result: dict | None, symbol: str) -> str:
    result = order_result.get("result", {}) if isinstance(order_result, dict) else {}
    return str(result.get("orderId") or result.get("orderLinkId") or build_signal_id(symbol))


def extract_order_avg_price(order_result: dict | None, fallback: float) -> float:
    result = order_result.get("result", {}) if isinstance(order_result, dict) else {}
    for key in ("avgPrice", "filled_avg_price"):
        value = result.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return float(fallback)

