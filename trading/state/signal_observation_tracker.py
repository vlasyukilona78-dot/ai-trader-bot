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


def _timestamp_seconds(value: Any) -> float:
    try:
        timestamp = value.timestamp()
    except Exception:
        timestamp = value
    return _safe_float(timestamp, 0.0)


class SignalObservationTracker:
    """Persist post-signal OHLC observations for offline, no-look-ahead scoring."""

    def __init__(
        self,
        path: str | Path,
        *,
        enabled: bool = True,
        horizon_minutes: int = 90,
        reaction_threshold_pct: float = 0.0035,
    ):
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.horizon_sec = max(60, int(horizon_minutes) * 60)
        self.measurement_minutes = tuple(
            sorted({3, 5, 10, 15, 20, 60, max(1, int(horizon_minutes))})
        )
        self.reaction_threshold_pct = max(0.0001, float(reaction_threshold_pct))
        self._active: dict[str, dict[str, Any]] = {}
        self._completed_ids: set[str] = set()
        self._load()

    @property
    def completed_path(self) -> Path:
        return self.path.with_name(f"{self.path.stem}_completed.jsonl")

    def _load(self) -> None:
        if not self.enabled:
            return
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
            except Exception:
                payload = {}
            rows = payload.get("observations", {}) if isinstance(payload, Mapping) else {}
            if isinstance(rows, Mapping):
                for raw_id, raw in rows.items():
                    if not isinstance(raw, Mapping):
                        continue
                    signal_id = str(raw_id or raw.get("signal_id") or "").strip()
                    symbol = _normalize_symbol(str(raw.get("symbol") or ""))
                    entry = _safe_float(raw.get("entry"), 0.0)
                    if not signal_id or not symbol or entry <= 0:
                        continue
                    record = dict(raw)
                    record["signal_id"] = signal_id
                    record["symbol"] = symbol
                    record["entry"] = entry
                    self._active[signal_id] = record
        if self.completed_path.exists():
            try:
                with self.completed_path.open("r", encoding="utf-8-sig") as handle:
                    for line in handle:
                        try:
                            row = json.loads(line)
                        except (TypeError, ValueError):
                            continue
                        signal_id = str(row.get("signal_id") or "").strip()
                        if signal_id:
                            self._completed_ids.add(signal_id)
            except OSError:
                pass

    def _save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "updated_at": time.time(),
            "horizon_sec": self.horizon_sec,
            "measurement_minutes": self.measurement_minutes,
            "observations": self._active,
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.path)

    def _append_completed(self, record: Mapping[str, Any]) -> None:
        self.completed_path.parent.mkdir(parents=True, exist_ok=True)
        with self.completed_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        signal_id = str(record.get("signal_id") or "").strip()
        if signal_id:
            self._completed_ids.add(signal_id)

    @staticmethod
    def _metrics(record: Mapping[str, Any]) -> dict[str, Any]:
        entry = _safe_float(record.get("entry"), 0.0)
        if entry <= 0:
            return {}
        min_low = _safe_float(record.get("min_low"), entry)
        max_high = _safe_float(record.get("max_high"), entry)
        signal_ts = _safe_float(record.get("signal_ts"), 0.0)
        first_favorable_ts = _safe_float(record.get("first_favorable_ts"), 0.0)
        first_adverse_ts = _safe_float(record.get("first_adverse_ts"), 0.0)
        return {
            "favorable_excursion_pct": max((entry - min_low) / entry * 100.0, 0.0),
            "adverse_excursion_pct": max((max_high - entry) / entry * 100.0, 0.0),
            "close_move_pct": (
                (entry - _safe_float(record.get("last_close"), entry)) / entry * 100.0
            ),
            "minutes_to_first_favorable": (
                max((first_favorable_ts - signal_ts) / 60.0, 0.0)
                if first_favorable_ts > 0 and signal_ts > 0
                else None
            ),
            "minutes_to_first_adverse": (
                max((first_adverse_ts - signal_ts) / 60.0, 0.0)
                if first_adverse_ts > 0 and signal_ts > 0
                else None
            ),
            "tp_hit": _safe_float(record.get("tp_hit_ts"), 0.0) > 0,
            "sl_hit": _safe_float(record.get("sl_hit_ts"), 0.0) > 0,
        }

    def active_count(self) -> int:
        return len(self._active)

    def expire_stale(self, *, observed_at: float | None = None) -> int:
        """Close wall-clock-expired records that never received a full horizon.

        A symbol can leave the runtime universe after its signal (for example
        because its turnover falls below the discovery floor).  In that case
        ``update_frame`` is never called for the symbol again.  Persist the
        partial observation explicitly instead of leaving it active forever.
        Partial rows use a distinct status so calibration cannot mistake them
        for complete horizon observations.
        """
        if not self.enabled or not self._active:
            return 0
        now_ts = _safe_float(observed_at, time.time())
        expired_ids: list[str] = []
        for signal_id, record in self._active.items():
            signal_ts = _safe_float(record.get("signal_ts"), 0.0)
            latest_bar_ts = _safe_float(record.get("last_bar_ts"), 0.0)
            if signal_ts <= 0 or now_ts < signal_ts + self.horizon_sec:
                continue
            if latest_bar_ts >= signal_ts + self.horizon_sec:
                continue
            expired_ids.append(signal_id)

        for signal_id in expired_ids:
            record = self._active.pop(signal_id, None)
            if not isinstance(record, dict):
                continue
            signal_ts = _safe_float(record.get("signal_ts"), 0.0)
            latest_bar_ts = _safe_float(record.get("last_bar_ts"), signal_ts)
            coverage_sec = max(latest_bar_ts - signal_ts, 0.0)
            record.update(
                {
                    "status": "expired_incomplete",
                    "completed_at": now_ts,
                    "completion_reason": "wall_clock_horizon_elapsed_without_full_market_data",
                    "observation_complete": False,
                    "coverage_sec": coverage_sec,
                    "coverage_ratio": min(coverage_sec / self.horizon_sec, 1.0),
                    **self._metrics(record),
                }
            )
            self._append_completed(record)

        if expired_ids:
            self._save()
        return len(expired_ids)

    def active_observation(
        self,
        *,
        signal_id: str = "",
        symbol: str = "",
    ) -> dict[str, Any] | None:
        normalized_id = str(signal_id or "").strip()
        normalized_symbol = _normalize_symbol(symbol)
        record = self._active.get(normalized_id) if normalized_id else None
        if isinstance(record, Mapping):
            if not normalized_symbol or record.get("symbol") == normalized_symbol:
                return {**dict(record), **self._metrics(record)}
        if not normalized_symbol:
            return None
        matches = [
            row
            for row in self._active.values()
            if isinstance(row, Mapping) and row.get("symbol") == normalized_symbol
        ]
        if not matches:
            return None
        latest = max(matches, key=lambda row: _safe_float(row.get("signal_ts"), 0.0))
        return {**dict(latest), **self._metrics(latest)}

    def record_short(
        self,
        *,
        signal_id: str,
        symbol: str,
        phase: str,
        entry: float,
        take_profit: float,
        stop_loss: float,
        signal_ts: float,
        signal_bar_ts: Any,
        delivered: bool,
        candidate_source: str = "",
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        normalized_id = str(signal_id or "").strip()
        normalized_symbol = _normalize_symbol(symbol)
        entry_price = _safe_float(entry, 0.0)
        if not normalized_id or not normalized_symbol or entry_price <= 0:
            return None
        existing = self._active.get(normalized_id)
        if isinstance(existing, Mapping):
            existing_record = dict(existing)
            if delivered and not bool(existing_record.get("delivered")):
                existing_record["delivered"] = True
                existing_record["delivery_succeeded_ts"] = time.time()
                self._active[normalized_id] = existing_record
                self._save()
            return dict(existing_record)
        if normalized_id in self._completed_ids:
            return None

        ts = _safe_float(signal_ts, time.time())
        record = {
            "signal_id": normalized_id,
            "symbol": normalized_symbol,
            "phase": str(phase or "").upper(),
            "side": "SHORT",
            "entry": entry_price,
            "take_profit": _safe_float(take_profit, 0.0),
            "stop_loss": _safe_float(stop_loss, 0.0),
            "signal_ts": ts,
            "signal_bar_ts": _timestamp_seconds(signal_bar_ts),
            "last_bar_ts": _timestamp_seconds(signal_bar_ts),
            "last_observed_ts": ts,
            "delivered": bool(delivered),
            "candidate_source": str(candidate_source or ""),
            "bars_observed": 0,
            "min_low": entry_price,
            "max_high": entry_price,
            "last_close": entry_price,
            "first_favorable_ts": 0.0,
            "first_adverse_ts": 0.0,
            "tp_hit_ts": 0.0,
            "sl_hit_ts": 0.0,
            "horizon_metrics": {},
        }
        self._active[normalized_id] = record
        self._save()
        return dict(record)

    def update_frame(self, symbol: str, frame, *, observed_at: float | None = None) -> int:
        if not self.enabled or frame is None or getattr(frame, "empty", True):
            return 0
        normalized_symbol = _normalize_symbol(symbol)
        if not normalized_symbol:
            return 0
        now_ts = _safe_float(observed_at, time.time())
        completed_ids: list[str] = []
        updated = 0

        for signal_id, record in list(self._active.items()):
            if record.get("symbol") != normalized_symbol:
                continue
            entry = _safe_float(record.get("entry"), 0.0)
            if entry <= 0:
                continue
            last_bar_ts = _safe_float(record.get("last_bar_ts"), 0.0)
            rows_seen = 0
            for index, row in frame.iterrows():
                bar_ts = _timestamp_seconds(index)
                if bar_ts <= last_bar_ts or bar_ts > now_ts:
                    continue
                high = _safe_float(row.get("high"), 0.0)
                low = _safe_float(row.get("low"), 0.0)
                close = _safe_float(row.get("close"), 0.0)
                if min(high, low, close) <= 0:
                    continue
                rows_seen += 1
                record["min_low"] = min(_safe_float(record.get("min_low"), entry), low)
                record["max_high"] = max(_safe_float(record.get("max_high"), entry), high)
                record["last_close"] = close
                record["last_bar_ts"] = bar_ts
                record["last_observed_ts"] = now_ts
                if (
                    _safe_float(record.get("first_favorable_ts"), 0.0) <= 0
                    and low <= entry * (1.0 - self.reaction_threshold_pct)
                ):
                    record["first_favorable_ts"] = bar_ts
                if (
                    _safe_float(record.get("first_adverse_ts"), 0.0) <= 0
                    and high >= entry * (1.0 + self.reaction_threshold_pct)
                ):
                    record["first_adverse_ts"] = bar_ts
                take_profit = _safe_float(record.get("take_profit"), 0.0)
                stop_loss = _safe_float(record.get("stop_loss"), 0.0)
                if _safe_float(record.get("tp_hit_ts"), 0.0) <= 0 and take_profit > 0 and low <= take_profit:
                    record["tp_hit_ts"] = bar_ts
                if _safe_float(record.get("sl_hit_ts"), 0.0) <= 0 and stop_loss > 0 and high >= stop_loss:
                    record["sl_hit_ts"] = bar_ts
                signal_ts = _safe_float(record.get("signal_ts"), 0.0)
                horizon_metrics = record.get("horizon_metrics")
                if not isinstance(horizon_metrics, dict):
                    horizon_metrics = {}
                    record["horizon_metrics"] = horizon_metrics
                for horizon_minutes in self.measurement_minutes:
                    horizon_key = str(horizon_minutes)
                    if (
                        horizon_key not in horizon_metrics
                        and signal_ts > 0
                        and bar_ts >= signal_ts + (horizon_minutes * 60)
                    ):
                        horizon_metrics[horizon_key] = {
                            **self._metrics(record),
                            "observed_bar_ts": bar_ts,
                        }

            if rows_seen:
                record["bars_observed"] = int(record.get("bars_observed", 0)) + rows_seen
                updated += 1
            signal_ts = _safe_float(record.get("signal_ts"), 0.0)
            latest_bar_ts = _safe_float(record.get("last_bar_ts"), 0.0)
            if signal_ts > 0 and latest_bar_ts >= signal_ts + self.horizon_sec:
                completed_ids.append(signal_id)

        for signal_id in completed_ids:
            record = self._active.pop(signal_id, None)
            if not isinstance(record, dict):
                continue
            record.update(
                {
                    "status": "completed",
                    "completed_at": now_ts,
                    **self._metrics(record),
                }
            )
            self._append_completed(record)

        if updated or completed_ids:
            self._save()
        return updated
