from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SIGNAL_COLUMNS = [
    "time",
    "symbol",
    "direction",
    "entry",
    "tp",
    "sl",
    "rsi",
    "vol_change",
    "ai_prob",
    "ai_horizon",
    "strategy",
    "vwap",
    "poc",
    "vah",
    "val",
    "sentiment",
    "obv",
    "cvd",
    "state",
]

TRADE_COLUMNS = ["time", "symbol", "profit", "duration", "side"]


def _to_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def validate_signal_row(row: dict) -> dict:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    normalized = {
        "time": str(row.get("time") or now_str),
        "symbol": str(row.get("symbol") or ""),
        "direction": str(row.get("direction") or ""),
        "entry": _to_float(row.get("entry")),
        "tp": _to_float(row.get("tp")),
        "sl": _to_float(row.get("sl")),
        "rsi": _to_float(row.get("rsi")),
        "vol_change": _to_float(row.get("vol_change")),
        "ai_prob": _to_float(row.get("ai_prob")),
        "ai_horizon": _to_float(row.get("ai_horizon")),
        "strategy": str(row.get("strategy") or ""),
        "vwap": _to_float(row.get("vwap")),
        "poc": _to_float(row.get("poc")),
        "vah": _to_float(row.get("vah")),
        "val": _to_float(row.get("val")),
        "sentiment": _to_float(row.get("sentiment")),
        "obv": _to_float(row.get("obv")),
        "cvd": _to_float(row.get("cvd")),
        "state": str(row.get("state") or "detected"),
    }

    for required in ("symbol", "direction", "strategy"):
        if not normalized[required]:
            raise ValueError(f"Signal row missing required field: {required}")

    for required_num in ("entry", "tp", "sl"):
        if not pd.notna(normalized[required_num]):
            raise ValueError(f"Signal row missing numeric field: {required_num}")

    return normalized


def validate_trade_row(row: dict) -> dict:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    normalized = {
        "time": str(row.get("time") or now_str),
        "symbol": str(row.get("symbol") or ""),
        "profit": _to_float(row.get("profit"), 0.0),
        "duration": _to_float(row.get("duration"), 0.0),
        "side": str(row.get("side") or ""),
    }
    if not normalized["symbol"]:
        raise ValueError("Trade row missing symbol")
    return normalized


def append_row_csv(path: str | Path, row: dict, columns: list[str]):
    row_df = pd.DataFrame([{col: row.get(col) for col in columns}])
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if path_obj.exists():
        existing = pd.read_csv(path_obj, nrows=0)
        missing = [c for c in columns if c not in existing.columns]
        if missing:
            raise ValueError(f"CSV schema mismatch for {path_obj}: missing columns {missing}")
        row_df.to_csv(path_obj, mode="a", index=False, header=False)
    else:
        row_df.to_csv(path_obj, index=False)
