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

from trading.market_data.bar_contract import interval_seconds, is_bar_aligned


# v2 added explicit source-response and cycle-completion timing plus the first
# executable entry bar, because v1 dated its universe data with a timestamp read
# before the request. v3 changes the file layout: the cycle envelope is written
# once as a header record with a closing footer instead of being copied into
# every decision row, which made a 300-symbol scan quadratic and pushed the
# ordered universe past the per-row metadata bounds. Two incompatible layouts
# must not both be called v2, so the version moves again.
SCHEMA_VERSION = 3

RECORD_TYPE_HEADER = "cycle_header"
RECORD_TYPE_DECISION = "decision"
RECORD_TYPE_FOOTER = "cycle_footer"
POPULATION_STATUSES = frozenset(
    {
        "evaluated",
        "no_data",
        "short_history",
        "invalid_bar_contract",
        "data_error",
        "data_quality_error",
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


# Recorded, but never part of market identity: these are wall clocks and cycle
# bookkeeping, and the same bars must hash the same however slowly they were
# fetched.
_NON_CAUSAL_METADATA_KEYS = frozenset({"feature_provenance"})


def _causal_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _NON_CAUSAL_METADATA_KEYS
    }


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
    universe_received_at: float,
    universe_symbols: Sequence[str],
    schema_version: int = SCHEMA_VERSION,
) -> str:
    """Return a stable ID for one point-in-time universe evaluation cycle.

    Built only from the cycle's causal inputs - the bar cutoff, the instant the
    universe response arrived, and the ordered symbol set. Nothing here depends on
    how quickly individual workers finished, so the identity of a cycle cannot
    change with thread scheduling.
    """

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
            "timeframe_seconds": interval_seconds(
                _bounded_string(timeframe, name="timeframe", max_length=32)
            ),
            "candle_cutoff_ts": _finite_float(candle_cutoff_ts, name="candle_cutoff_ts"),
            "universe_received_at": _finite_float(universe_received_at, name="universe_received_at"),
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
    universe_request_started_at: float
    universe_received_at: float
    scan_observed_at: float
    candle_cutoff_ts: float
    decision_ts: float
    ranking_ready_ts: float
    cycle_completed_ts: float
    actionable_ts: float
    entry_eligible_ts: float
    entry_bar_open_ts: float
    symbol: str
    timeframe: str
    status: str
    base_bar_open_ts: float | None
    base_bar_close_ts: float | None
    action: str
    reason: str
    confidence: float
    metadata: Mapping[str, object]
    cycle_ordinal: int = 0
    cycle_size: int = 1
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
            "universe_request_started_at",
            "universe_received_at",
            "scan_observed_at",
            "candle_cutoff_ts",
            "decision_ts",
            "ranking_ready_ts",
            "cycle_completed_ts",
            "actionable_ts",
            "entry_eligible_ts",
            "entry_bar_open_ts",
            "confidence",
        ):
            object.__setattr__(self, field_name, _finite_float(getattr(self, field_name), name=field_name))
        if self.universe_refreshed_at > self.scan_observed_at:
            raise PopulationJournalError("universe refresh follows scan observation time")
        if self.universe_received_at < self.universe_request_started_at:
            raise PopulationJournalError("universe response precedes its own request")
        if self.universe_received_at > self.scan_observed_at:
            raise PopulationJournalError("universe response follows scan observation time")
        if self.scan_observed_at < self.candle_cutoff_ts:
            raise PopulationJournalError("scan observation precedes the causal cutoff")
        if self.decision_ts < self.scan_observed_at:
            raise PopulationJournalError("decision_ts precedes scan observation time")

        # The cycle is comparable only once every one of its symbols has been
        # decided, and reachable only once it is also sealed.
        if self.ranking_ready_ts < self.decision_ts:
            raise PopulationJournalError("ranking_ready_ts precedes this decision")
        if self.cycle_completed_ts < self.ranking_ready_ts:
            raise PopulationJournalError("cycle_completed_ts precedes ranking_ready_ts")
        if self.actionable_ts < self.ranking_ready_ts:
            raise PopulationJournalError("actionable_ts precedes ranking_ready_ts")
        if self.actionable_ts < self.universe_received_at:
            raise PopulationJournalError("actionable_ts precedes the universe response")
        if self.entry_eligible_ts < self.actionable_ts:
            raise PopulationJournalError("entry_eligible_ts precedes actionable_ts")
        if self.entry_eligible_ts < self.cycle_completed_ts:
            raise PopulationJournalError("entry_eligible_ts precedes cycle completion")
        # A decision known at a bar's open cannot be filled at that open.
        if self.entry_bar_open_ts <= self.actionable_ts:
            raise PopulationJournalError("entry bar does not open after actionable_ts")
        if self.entry_bar_open_ts <= self.entry_eligible_ts:
            raise PopulationJournalError("entry bar does not open after entry_eligible_ts")
        if not is_bar_aligned(self.entry_bar_open_ts, self.timeframe):
            raise PopulationJournalError("entry bar open is not aligned to the timeframe")
        if self.entry_bar_open_ts - self.entry_eligible_ts > interval_seconds(self.timeframe):
            raise PopulationJournalError("entry bar skips a reachable bar")
        if (self.base_bar_open_ts is None) != (self.base_bar_close_ts is None):
            raise PopulationJournalError("base bar open and close must both be present or absent")
        if self.base_bar_open_ts is not None:
            base_open = _finite_float(self.base_bar_open_ts, name="base_bar_open_ts")
            base_close = _finite_float(self.base_bar_close_ts, name="base_bar_close_ts")
            if base_open >= base_close:
                raise PopulationJournalError("base bar open must precede its close")
            if base_close > self.candle_cutoff_ts:
                raise PopulationJournalError("base bar closes after the causal cutoff")
            expected_seconds = interval_seconds(self.timeframe)
            if not math.isclose(base_close - base_open, expected_seconds, rel_tol=0.0, abs_tol=1e-6):
                raise PopulationJournalError("base bar duration differs from timeframe")
            object.__setattr__(self, "base_bar_open_ts", base_open)
            object.__setattr__(self, "base_bar_close_ts", base_close)
        elif self.status in {"evaluated", "short_history", "strategy_error"}:
            raise PopulationJournalError(f"{self.status} requires real base bar timestamps")
        if not 0.0 <= self.confidence <= 1.0:
            raise PopulationJournalError("confidence must be between 0 and 1")
        object.__setattr__(self, "symbol", _bounded_string(self.symbol, name="symbol", max_length=64))
        object.__setattr__(self, "timeframe", _bounded_string(self.timeframe, name="timeframe", max_length=32))
        object.__setattr__(self, "action", _bounded_string(self.action, name="action", max_length=64))
        object.__setattr__(self, "reason", _bounded_string(self.reason, name="reason", max_length=512))
        if self.status not in POPULATION_STATUSES:
            raise PopulationJournalError(f"unsupported population status: {self.status}")
        if isinstance(self.cycle_ordinal, bool) or not isinstance(self.cycle_ordinal, Integral):
            raise PopulationJournalError("cycle_ordinal must be an integer")
        if isinstance(self.cycle_size, bool) or not isinstance(self.cycle_size, Integral):
            raise PopulationJournalError("cycle_size must be an integer")
        if self.cycle_size < 1 or not 0 <= self.cycle_ordinal < self.cycle_size:
            raise PopulationJournalError("cycle ordinal is outside the declared cycle size")
        object.__setattr__(self, "cycle_ordinal", int(self.cycle_ordinal))
        object.__setattr__(self, "cycle_size", int(self.cycle_size))
        if self.error_code is not None and not _ERROR_CODE_RE.fullmatch(self.error_code):
            raise PopulationJournalError("error_code must be a safe exception class name")
        error_statuses = {
            "invalid_bar_contract",
            "data_error",
            "data_quality_error",
            "strategy_error",
        }
        if self.status in error_statuses and self.error_code is None:
            raise PopulationJournalError(f"{self.status} requires error_code")
        if self.status not in error_statuses and self.error_code is not None:
            raise PopulationJournalError(f"{self.status} must not carry error_code")
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
        universe_request_started_at: float,
        universe_received_at: float,
        scan_observed_at: float,
        candle_cutoff_ts: float,
        decision_ts: float,
        ranking_ready_ts: float,
        cycle_completed_ts: float,
        actionable_ts: float,
        entry_eligible_ts: float,
        entry_bar_open_ts: float,
        symbol: str,
        timeframe: str,
        status: str,
        base_bar_open_ts: float | None,
        base_bar_close_ts: float | None,
        action: str,
        reason: str,
        confidence: float,
        metadata: Mapping[str, object],
        cycle_ordinal: int = 0,
        cycle_size: int = 1,
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
            "timeframe_seconds": interval_seconds(timeframe),
            "status": status,
            "candle_cutoff_ts": _finite_float(candle_cutoff_ts, name="candle_cutoff_ts"),
            "base_bar_open_ts": (
                _finite_float(base_bar_open_ts, name="base_bar_open_ts")
                if base_bar_open_ts is not None
                else None
            ),
            "base_bar_close_ts": (
                _finite_float(base_bar_close_ts, name="base_bar_close_ts")
                if base_bar_close_ts is not None
                else None
            ),
            "action": action,
            "reason": reason,
            "confidence": _finite_float(confidence, name="confidence"),
            "metadata": _causal_metadata(_thaw_json_value(frozen_metadata)),
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
            universe_request_started_at=universe_request_started_at,
            universe_received_at=universe_received_at,
            scan_observed_at=scan_observed_at,
            candle_cutoff_ts=candle_cutoff_ts,
            decision_ts=decision_ts,
            ranking_ready_ts=ranking_ready_ts,
            cycle_completed_ts=cycle_completed_ts,
            actionable_ts=actionable_ts,
            entry_eligible_ts=entry_eligible_ts,
            entry_bar_open_ts=entry_bar_open_ts,
            symbol=symbol,
            timeframe=timeframe,
            status=status,
            base_bar_open_ts=base_bar_open_ts,
            base_bar_close_ts=base_bar_close_ts,
            action=action,
            reason=reason,
            confidence=confidence,
            metadata=frozen_metadata,
            cycle_ordinal=cycle_ordinal,
            cycle_size=cycle_size,
            error_code=error_code,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "record_type": RECORD_TYPE_DECISION,
            "schema_version": int(self.schema_version),
            "cycle_id": self.cycle_id,
            "snapshot_id": self.snapshot_id,
            "input_hash": self.input_hash,
            "universe_refreshed_at": self.universe_refreshed_at,
            "universe_request_started_at": self.universe_request_started_at,
            "universe_received_at": self.universe_received_at,
            "scan_observed_at": self.scan_observed_at,
            "candle_cutoff_ts": self.candle_cutoff_ts,
            "decision_ts": self.decision_ts,
            "ranking_ready_ts": self.ranking_ready_ts,
            "cycle_completed_ts": self.cycle_completed_ts,
            "actionable_ts": self.actionable_ts,
            "entry_eligible_ts": self.entry_eligible_ts,
            "entry_bar_open_ts": self.entry_bar_open_ts,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timeframe_seconds": interval_seconds(self.timeframe),
            "status": self.status,
            "base_bar_open_ts": self.base_bar_open_ts,
            "base_bar_close_ts": self.base_bar_close_ts,
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "metadata": _thaw_json_value(self.metadata),
            "cycle_ordinal": self.cycle_ordinal,
            "cycle_size": self.cycle_size,
            "error_code": self.error_code,
        }


def rows_checksum(snapshot_ids: Sequence[str]) -> str:
    """Digest over the ordered snapshot IDs of one cycle.

    Lets a reader detect a truncated or reordered body without re-deriving every
    row, and lets the footer state what the header promised.
    """

    return _sha256_id({"snapshot_ids": [str(value) for value in snapshot_ids]})


class PopulationJournal:
    """Append-only cycle log: one header, its decision rows, then a footer.

    The envelope is written once per cycle. Copying it onto every row made the
    file quadratic in universe size and pushed the ordered symbol list past the
    bounds that keep arbitrary per-row metadata safe.
    """

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self._path = Path(path)
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._last_cycle_id = self._inspect_existing_file() if self._enabled else None

    def _inspect_existing_file(self) -> str | None:
        """Refuse to append to a file written by a different schema or left torn."""

        if not self._path.exists() or self._path.stat().st_size == 0:
            return None
        try:
            with self._path.open("rb") as handle:
                first_line = handle.readline()
                size = handle.seek(0, os.SEEK_END)
                handle.seek(max(0, size - 1_048_576), os.SEEK_SET)
                tail = handle.read()
        except OSError as exc:
            raise PopulationJournalError("population journal is unreadable") from exc

        # A previous process that died mid-write leaves a line without its
        # newline. Appending would concatenate two JSON objects into one
        # unparseable line, so stop and say how to recover.
        if not tail.endswith(b"\n"):
            raise PopulationJournalError(
                "population journal ends without a newline; it was truncated mid-write. "
                "Move the file aside and start a new one rather than appending to it."
            )

        try:
            header = json.loads(first_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PopulationJournalError("population journal has an unreadable first record") from exc
        if not isinstance(header, Mapping):
            raise PopulationJournalError("population journal first record is not an object")
        existing_version = header.get("schema_version")
        if existing_version != SCHEMA_VERSION:
            raise PopulationJournalError(
                f"population journal was written by schema {existing_version!r}, "
                f"this build writes {SCHEMA_VERSION}. Use a separate file per schema."
            )
        if header.get("record_type") != RECORD_TYPE_HEADER:
            raise PopulationJournalError("population journal does not start with a cycle header")

        lines = [line for line in tail.splitlines() if line.strip()]
        if not lines:
            return None
        try:
            footer = json.loads(lines[-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PopulationJournalError("population journal has an unreadable tail") from exc
        if not isinstance(footer, Mapping) or footer.get("record_type") != RECORD_TYPE_FOOTER:
            raise PopulationJournalError("population journal ends with an incomplete cycle")
        cycle_id = footer.get("cycle_id")
        if not isinstance(cycle_id, str) or not re.fullmatch(r"[0-9a-f]{64}", cycle_id):
            raise PopulationJournalError("population journal tail has an invalid cycle ID")
        return cycle_id

    @property
    def enabled(self) -> bool:
        return self._enabled

    def append_cycle(
        self,
        records: Sequence[PopulationDecision],
        *,
        envelope: object,
    ) -> bool:
        """Write one complete cycle: header, ordered rows, footer.

        A cycle with no rows is still written. An empty universe or a failure
        before evaluation is evidence, and a hole in the log cannot be told apart
        from a scan that never ran.
        """

        if not self._enabled:
            return False
        if envelope is None:
            raise PopulationJournalError("a cycle requires its envelope")

        envelope_payload = envelope.as_dict()
        cycle_id = envelope_payload.get("cycle_id")
        if not isinstance(cycle_id, str) or not re.fullmatch(r"[0-9a-f]{64}", cycle_id):
            raise PopulationJournalError("cycle envelope has an invalid cycle ID")

        rows: list[bytes] = []
        for record in records:
            if not isinstance(record, PopulationDecision):
                raise PopulationJournalError("records must contain PopulationDecision instances")
            if record.schema_version != SCHEMA_VERSION:
                raise PopulationJournalError("append batch mixes journal schema versions")
            if record.cycle_id != cycle_id:
                raise PopulationJournalError("decision row does not belong to the envelope cycle")
            rows.append(_canonical_bytes(record.as_dict()) + b"\n")

        expected_size = len(records)
        if expected_size:
            if any(record.cycle_size != expected_size for record in records):
                raise PopulationJournalError("declared cycle size differs from append batch")
            if [record.cycle_ordinal for record in records] != list(range(expected_size)):
                raise PopulationJournalError("cycle records must be complete and ordered")
            if len({record.symbol for record in records}) != expected_size:
                raise PopulationJournalError("cycle contains duplicate symbols")
            if len({record.snapshot_id for record in records}) != expected_size:
                raise PopulationJournalError("cycle contains duplicate snapshots")

        snapshot_ids = [record.snapshot_id for record in records]
        header = _canonical_bytes(
            {
                "record_type": RECORD_TYPE_HEADER,
                "schema_version": SCHEMA_VERSION,
                "cycle_id": cycle_id,
                "row_count": expected_size,
                "envelope": envelope_payload,
            }
        ) + b"\n"
        footer = _canonical_bytes(
            {
                "record_type": RECORD_TYPE_FOOTER,
                "schema_version": SCHEMA_VERSION,
                "cycle_id": cycle_id,
                "row_count": expected_size,
                "rows_checksum": rows_checksum(snapshot_ids),
            }
        ) + b"\n"

        batch = header + b"".join(rows) + footer
        with self._lock:
            if self._last_cycle_id == cycle_id:
                return False
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("ab") as handle:
                written = handle.write(batch)
                if written != len(batch):
                    raise OSError("population journal batch write was incomplete")
                handle.flush()
                os.fsync(handle.fileno())
            self._last_cycle_id = cycle_id
        return True
