from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").upper().strip()


class SignalPositionTracker:
    """Persistent shadow positions for users who trade bot signals manually."""

    def __init__(self, path: str | Path, *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = bool(enabled)
        self._positions: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.enabled or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except Exception:
            return
        rows = payload.get("positions", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(rows, Mapping):
            return
        for raw_symbol, raw in rows.items():
            if not isinstance(raw, Mapping):
                continue
            symbol = _normalize_symbol(str(raw_symbol))
            entry = _safe_float(raw.get("entry_price"), 0.0)
            if not symbol or entry <= 0:
                continue
            record = dict(raw)
            record["symbol"] = symbol
            record["entry_price"] = entry
            record["active"] = True
            self._positions[symbol] = record

    def _save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "positions": self._positions,
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.path)

    def _append_event(self, event: str, record: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        event_path = self.path.with_name(f"{self.path.stem}_events.jsonl")
        event_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(),
            "event": str(event),
            **dict(record),
        }
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def active(self, symbol: str) -> dict[str, Any] | None:
        record = self._positions.get(_normalize_symbol(symbol))
        return dict(record) if isinstance(record, Mapping) else None

    def record_short(
        self,
        *,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        opened_at: float | None = None,
        pump_id: str = "",
        signal_id: str = "",
        leverage: float = 0.0,
        source: str = "main_signal",
        replace: bool = False,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        normalized = _normalize_symbol(symbol)
        entry = _safe_float(entry_price, 0.0)
        if not normalized or entry <= 0:
            return None

        existing = self._positions.get(normalized)
        if isinstance(existing, Mapping) and not replace:
            return dict(existing)

        ts = _safe_float(opened_at, time.time())
        record = {
            "active": True,
            "symbol": normalized,
            "side": "SHORT",
            "entry_price": entry,
            "stop_loss": _safe_float(stop_loss, 0.0),
            "take_profit": _safe_float(take_profit, 0.0),
            "opened_at": ts,
            "updated_at": ts,
            "best_price": entry,
            "worst_price": entry,
            "last_price": entry,
            "pump_id": str(pump_id or ""),
            "signal_id": str(signal_id or ""),
            "leverage": max(0.0, _safe_float(leverage, 0.0)),
            "source": str(source or "main_signal"),
        }
        self._positions[normalized] = record
        self._save()
        self._append_event("REPLACE_SHORT" if isinstance(existing, Mapping) else "OPEN_SHORT", record)
        return dict(record)

    def update_mark(self, symbol: str, mark_price: float, *, updated_at: float | None = None) -> dict[str, Any] | None:
        normalized = _normalize_symbol(symbol)
        record = self._positions.get(normalized)
        mark = _safe_float(mark_price, 0.0)
        if not isinstance(record, dict) or mark <= 0:
            return None
        record["last_price"] = mark
        record["best_price"] = min(_safe_float(record.get("best_price"), mark), mark)
        record["worst_price"] = max(_safe_float(record.get("worst_price"), mark), mark)
        record["updated_at"] = _safe_float(updated_at, time.time())
        self._save()
        return dict(record)

    def close(
        self,
        symbol: str,
        *,
        exit_price: float,
        reason: str,
        closed_at: float | None = None,
    ) -> dict[str, Any] | None:
        normalized = _normalize_symbol(symbol)
        record = self._positions.pop(normalized, None)
        if not isinstance(record, dict):
            return None
        exit_value = _safe_float(exit_price, _safe_float(record.get("last_price"), 0.0))
        entry = _safe_float(record.get("entry_price"), 0.0)
        closed = {
            **record,
            "active": False,
            "exit_price": exit_value,
            "closed_at": _safe_float(closed_at, time.time()),
            "close_reason": str(reason or ""),
            "price_return": ((entry - exit_value) / entry) if entry > 0 and exit_value > 0 else 0.0,
        }
        self._save()
        self._append_event("CLOSE_SHORT", closed)
        return closed
