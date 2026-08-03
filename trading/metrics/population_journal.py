from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
POPULATION_STATUSES = frozenset(
    {
        "evaluated",
        "no_data",
        "short_history",
        "invalid_bar_contract",
        "strategy_error",
    }
)

_MAX_STRING_LENGTH = 2_048
_MAX_COLLECTION_LENGTH = 256
_MAX_METADATA_BYTES = 65_536
_MAX_DEPTH = 8
_ERROR_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,127}$")
_FORBIDDEN_METADATA_KEYS = frozenset(
    {"exception", "exception_message", "traceback", "stacktrace", "error_message"}
)


class PopulationJournalError(ValueError):
    """Raised when a population record cannot be represented safely."""


def _bounded_string(value: object, *, name: str, max_length: int = _MAX_STRING_LENGTH) -> str:
    if not isinstance(value, str):
        raise PopulationJournalError(f"{name} must be a string")
    if not value:
        raise PopulationJournalError(f"{name} must not be empty")
    if len(value) > max_length:
        raise PopulationJournalError(f"{name} exceeds {max_length} characters")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PopulationJournalError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PopulationJournalError(f"{name} must be a finite number")
    return result


def _freeze_json_value(value: object, *, path: str, depth: int = 0) -> object:
    if depth > _MAX_DEPTH:
        raise PopulationJournalError(f"{path} exceeds maximum nesting depth")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise PopulationJournalError(f"{path} contains a non-finite number")
        return number
    if isinstance(value, str):
        if len(value) > _MAX_STRING_LENGTH:
            raise PopulationJournalError(f"{path} contains an oversized string")
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_COLLECTION_LENGTH:
            raise PopulationJournalError(f"{path} contains too many keys")
        keys = list(value)
        if any(not isinstance(key, str) or not key or len(key) > 128 for key in keys):
            raise PopulationJournalError(f"{path} contains an invalid key")
        frozen: dict[str, object] = {}
        for key in sorted(keys):
            if key.casefold() in _FORBIDDEN_METADATA_KEYS:
                raise PopulationJournalError(f"{path}.{key} may expose exception text")
            frozen[key] = _freeze_json_value(value[key], path=f"{path}.{key}", depth=depth + 1)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_COLLECTION_LENGTH:
            raise PopulationJournalError(f"{path} contains too many items")
        return tuple(
            _freeze_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise PopulationJournalError(f"{path} contains unsupported type {type(value).__name__}")


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PopulationJournalError("payload is not canonical JSON") from exc
    return encoded


def _sha256_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def make_cycle_id(
    *,
    timeframe: str,
    candle_cutoff_ts: float,
    universe_refreshed_at: float,
    universe_symbols: Sequence[str],
    schema_version: int = SCHEMA_VERSION,
) -> str:
    """Return a stable ID for one point-in-time universe evaluation cycle."""

    if isinstance(schema_version, bool) or not isinstance(schema_version, Integral) or schema_version < 1:
        raise PopulationJournalError("schema_version must be a positive integer")
    clean_symbols = [_bounded_string(symbol, name="universe symbol", max_length=64) for symbol in universe_symbols]
    if len(clean_symbols) > 10_000:
        raise PopulationJournalError("universe contains too many symbols")
    if len(set(clean_symbols)) != len(clean_symbols):
        raise PopulationJournalError("universe symbols must be unique")
    return _sha256_id(
        {
            "schema_version": int(schema_version),
            "timeframe": _bounded_string(timeframe, name="timeframe", max_length=32),
            "candle_cutoff_ts": _finite_float(candle_cutoff_ts, name="candle_cutoff_ts"),
            "universe_refreshed_at": _finite_float(universe_refreshed_at, name="universe_refreshed_at"),
            "universe_symbols": clean_symbols,
        }
    )


def safe_error_code(exc: BaseException) -> str:
    """Return an exception class name without serializing its message."""

    code = type(exc).__name__
    if not _ERROR_CODE_RE.fullmatch(code):
        return "UnknownError"
    return code


@dataclass(frozen=True)
class PopulationDecision:
    schema_version: int
    cycle_id: str
    snapshot_id: str
    input_hash: str
    universe_refreshed_at: float
    scan_observed_at: float
    candle_cutoff_ts: float
    decision_ts: float
    symbol: str
    timeframe: str
    status: str
    base_bar_open_ts: float
    base_bar_close_ts: float
    action: str
    reason: str
    confidence: float
    metadata: Mapping[str, object]
    error_code: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, Integral) or self.schema_version < 1:
            raise PopulationJournalError("schema_version must be a positive integer")
        for field_name in ("cycle_id", "snapshot_id", "input_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise PopulationJournalError(f"{field_name} must be a lowercase SHA-256 hex digest")
        for field_name in (
            "universe_refreshed_at",
            "scan_observed_at",
            "candle_cutoff_ts",
            "decision_ts",
            "base_bar_open_ts",
            "base_bar_close_ts",
            "confidence",
        ):
            object.__setattr__(self, field_name, _finite_float(getattr(self, field_name), name=field_name))
        if self.base_bar_close_ts > self.candle_cutoff_ts:
            raise PopulationJournalError("base bar closes after the causal cutoff")
        if self.decision_ts < self.candle_cutoff_ts:
            raise PopulationJournalError("decision_ts precedes the causal cutoff")
        if not 0.0 <= self.confidence <= 1.0:
            raise PopulationJournalError("confidence must be between 0 and 1")
        object.__setattr__(self, "symbol", _bounded_string(self.symbol, name="symbol", max_length=64))
        object.__setattr__(self, "timeframe", _bounded_string(self.timeframe, name="timeframe", max_length=32))
        object.__setattr__(self, "action", _bounded_string(self.action, name="action", max_length=64))
        object.__setattr__(self, "reason", _bounded_string(self.reason, name="reason", max_length=512))
        if self.status not in POPULATION_STATUSES:
            raise PopulationJournalError(f"unsupported population status: {self.status}")
        if self.error_code is not None and not _ERROR_CODE_RE.fullmatch(self.error_code):
            raise PopulationJournalError("error_code must be a safe exception class name")
        if self.status == "strategy_error" and self.error_code is None:
            raise PopulationJournalError("strategy_error requires error_code")
        if not isinstance(self.metadata, Mapping):
            raise PopulationJournalError("metadata must be a mapping")
        frozen_metadata = _freeze_json_value(self.metadata, path="metadata")
        metadata_bytes = _canonical_bytes(_thaw_json_value(frozen_metadata))
        if len(metadata_bytes) > _MAX_METADATA_BYTES:
            raise PopulationJournalError("metadata exceeds maximum encoded size")
        object.__setattr__(self, "metadata", frozen_metadata)

    @classmethod
    def create(
        cls,
        *,
        cycle_id: str,
        universe_refreshed_at: float,
        scan_observed_at: float,
        candle_cutoff_ts: float,
        decision_ts: float,
        symbol: str,
        timeframe: str,
        status: str,
        base_bar_open_ts: float,
        base_bar_close_ts: float,
        action: str,
        reason: str,
        confidence: float,
        metadata: Mapping[str, object],
        error_code: str | None = None,
        schema_version: int = SCHEMA_VERSION,
    ) -> "PopulationDecision":
        if not isinstance(metadata, Mapping):
            raise PopulationJournalError("metadata must be a mapping")
        frozen_metadata = _freeze_json_value(metadata, path="metadata")
        causal_payload: dict[str, object] = {
            "schema_version": int(schema_version),
            "cycle_id": cycle_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "status": status,
            "candle_cutoff_ts": _finite_float(candle_cutoff_ts, name="candle_cutoff_ts"),
            "base_bar_open_ts": _finite_float(base_bar_open_ts, name="base_bar_open_ts"),
            "base_bar_close_ts": _finite_float(base_bar_close_ts, name="base_bar_close_ts"),
            "action": action,
            "reason": reason,
            "confidence": _finite_float(confidence, name="confidence"),
            "metadata": _thaw_json_value(frozen_metadata),
            "error_code": error_code,
        }
        input_hash = _sha256_id(causal_payload)
        snapshot_id = _sha256_id(
            {
                "schema_version": int(schema_version),
                "cycle_id": cycle_id,
                "symbol": symbol,
                "input_hash": input_hash,
            }
        )
        return cls(
            schema_version=int(schema_version),
            cycle_id=cycle_id,
            snapshot_id=snapshot_id,
            input_hash=input_hash,
            universe_refreshed_at=universe_refreshed_at,
            scan_observed_at=scan_observed_at,
            candle_cutoff_ts=candle_cutoff_ts,
            decision_ts=decision_ts,
            symbol=symbol,
            timeframe=timeframe,
            status=status,
            base_bar_open_ts=base_bar_open_ts,
            base_bar_close_ts=base_bar_close_ts,
            action=action,
            reason=reason,
            confidence=confidence,
            metadata=frozen_metadata,
            error_code=error_code,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": int(self.schema_version),
            "cycle_id": self.cycle_id,
            "snapshot_id": self.snapshot_id,
            "input_hash": self.input_hash,
            "universe_refreshed_at": self.universe_refreshed_at,
            "scan_observed_at": self.scan_observed_at,
            "candle_cutoff_ts": self.candle_cutoff_ts,
            "decision_ts": self.decision_ts,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "status": self.status,
            "base_bar_open_ts": self.base_bar_open_ts,
            "base_bar_close_ts": self.base_bar_close_ts,
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "metadata": _thaw_json_value(self.metadata),
            "error_code": self.error_code,
        }


class PopulationJournal:
    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self._path = Path(path)
        self._enabled = bool(enabled)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def append_cycle(self, records: Sequence[PopulationDecision]) -> None:
        if not self._enabled or not records:
            return
        rows: list[bytes] = []
        for record in records:
            if not isinstance(record, PopulationDecision):
                raise PopulationJournalError("records must contain PopulationDecision instances")
            rows.append(_canonical_bytes(record.as_dict()) + b"\n")
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("ab") as handle:
                for row in rows:
                    handle.write(row)
                handle.flush()
                os.fsync(handle.fileno())
