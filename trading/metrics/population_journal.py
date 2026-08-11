from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence
from weakref import WeakValueDictionary

from core.mexc_strategy_spec import decode_mexc_strategy_spec_evidence
from trading.market_data.bar_contract import closed_boundary_ts, interval_seconds, is_bar_aligned
from trading.market_data.frame_provenance import (
    FRAME_PROVENANCE_CONTRACT_VERSION,
    RAW_BUNDLE_HASH_CONTRACT_VERSION,
    FrameProvenanceError,
    SourceReadEvidenceV1,
    aggregate_source_timing_from_evidence,
    frame_provenance_contract_hash,
    raw_frame_bundle_hash,
    source_timing_from_evidence,
)
from trading.metrics.cycle_envelope import CycleEnvelope, CycleEnvelopeError
from trading.signals.lifecycle_contract import (
    LIFECYCLE_CONTRACT_VERSION,
    CandidateLifecycleEventV1,
    CandidateLifecycleState,
    CandidateSide,
    LifecycleContractError,
    ProposalObservationStatus,
    lifecycle_contract_hash,
)


# v2 added explicit source-response and cycle-completion timing plus the first
# executable entry bar, because v1 dated its universe data with a timestamp read
# before the request. v3 changes the file layout: the cycle envelope is written
# once as a header record with a closing footer instead of being copied into
# every decision row, which made a 300-symbol scan quadratic and pushed the
# ordered universe past the per-row metadata bounds. Two incompatible layouts
# must not both be called v2, so the version moves again. v4 binds that envelope
# (whose own schema and captured feature contract changed after v3 shipped) to
# both the body and footer, and refuses mixed or incomplete population cycles.
# v5 adds an ordered, domain-separated cycle commitment.  The chain does not
# authenticate itself (a trusted checkpoint is still required for that), but it
# makes every later cycle depend on the exact canonical bytes of its prefix.
SCHEMA_VERSION = 5

# ``SCHEMA_VERSION`` remains the public compatibility alias for the immutable
# v5 evidence already on disk.  It must never be repointed at a newer writer:
# callers use it to rebuild the frozen v5 commitment domains and fixtures.
CURRENT_WRITE_SCHEMA = 6
SUPPORTED_JOURNAL_SCHEMAS = frozenset({SCHEMA_VERSION, CURRENT_WRITE_SCHEMA})

# Cycle identity is a causal cohort contract, not a serialization detail of the
# journal that happens to carry it. The original identity algorithm shipped
# while journal schema v5 was current and included that literal in its hash
# payload. Keep the value pinned independently so a future journal-format bump
# cannot silently rename otherwise identical cohorts.
CYCLE_IDENTITY_VERSION = 5

RECORD_TYPE_HEADER = "cycle_header"
RECORD_TYPE_DECISION = "decision"
RECORD_TYPE_FOOTER = "cycle_footer"
RECORD_TYPE_CHECKPOINT = "population_journal_checkpoint"
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
V6_ACTIONS = frozenset({"HOLD", "SHORT_ENTRY", "LONG_ENTRY"})
_MAX_JOURNAL_LINE_BYTES = 2_000_000
FEATURE_PROVENANCE_KEYS = frozenset(
    {
        "universe_received_at",
        "universe_source_ts",
        "universe_cache_hit",
        "envelope_hash",
        "market_feature_hash",
    }
)
_LOCK_ACQUIRE_TIMEOUT_SEC = 60.0
_LOCK_RETRY_INTERVAL_SEC = 0.05
_JOURNAL_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_GENESIS_DOMAIN_V5 = b"KOTEIKA_POPULATION_GENESIS_V5\x00"
_CYCLE_COMMIT_DOMAIN_V5 = b"KOTEIKA_POPULATION_CYCLE_V5\x00"
_GENESIS_DOMAIN_V6 = b"KOTEIKA_POPULATION_GENESIS_V6\x00"
_CYCLE_COMMIT_DOMAIN_V6 = b"KOTEIKA_POPULATION_CYCLE_V6\x00"
CHECKPOINT_RECEIPT_SCHEMA_VERSION = 1

HEADER_KEYS_V5 = frozenset(
    {
        "record_type",
        "schema_version",
        "journal_id",
        "sequence_no",
        "prev_cycle_commit",
        "cycle_id",
        "row_count",
        "envelope_hash",
        "envelope",
    }
)
EVIDENCE_CONTRACT_KEYS = frozenset(
    {
        "frame_provenance_contract_version",
        "frame_provenance_contract_hash",
        "raw_frame_bundle_hash_contract_version",
        "lifecycle_contract_version",
        "lifecycle_contract_hash",
    }
)
HEADER_KEYS_V6 = HEADER_KEYS_V5 | {
    "evidence_contracts",
    "benchmark_source_evidence",
}
# Historical import retained for code/tests that intentionally inspect v5.
HEADER_KEYS = HEADER_KEYS_V5
FOOTER_CORE_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "journal_id",
        "sequence_no",
        "prev_cycle_commit",
        "cycle_id",
        "row_count",
        "envelope_hash",
        "rows_checksum",
    }
)
FOOTER_KEYS = FOOTER_CORE_KEYS | {"cycle_commit"}


def header_keys_for_schema(schema_version: int) -> frozenset[str]:
    if schema_version == SCHEMA_VERSION:
        return HEADER_KEYS_V5
    if schema_version == CURRENT_WRITE_SCHEMA:
        return HEADER_KEYS_V6
    raise PopulationJournalError("unsupported population journal schema")


def evidence_contracts_payload() -> dict[str, str]:
    """Return the exact executable identities required by journal v6."""

    return {
        "frame_provenance_contract_version": FRAME_PROVENANCE_CONTRACT_VERSION,
        "frame_provenance_contract_hash": frame_provenance_contract_hash(),
        "raw_frame_bundle_hash_contract_version": RAW_BUNDLE_HASH_CONTRACT_VERSION,
        "lifecycle_contract_version": LIFECYCLE_CONTRACT_VERSION,
        "lifecycle_contract_hash": lifecycle_contract_hash(),
    }


def _validated_evidence_contracts(payload: object) -> Mapping[str, str]:
    if not isinstance(payload, Mapping) or set(payload) != EVIDENCE_CONTRACT_KEYS:
        raise PopulationJournalError("evidence contracts schema mismatch")
    expected = evidence_contracts_payload()
    if dict(payload) != expected:
        raise PopulationJournalError("evidence contracts identity mismatch")
    return MappingProxyType(dict(expected))


# File locks alone do not provide portable thread exclusion: POSIX and Windows
# differ in whether two descriptors owned by one process contend with each
# other.  Every spelling of the same resolved journal path therefore shares a
# process-local lock as well as the OS-level sidecar lock below.
_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: WeakValueDictionary[str, threading.RLock] = WeakValueDictionary()


def _canonical_path(path: Path) -> Path:
    return Path(os.path.normcase(str(path.expanduser().resolve(strict=False))))


def _process_path_lock(path: Path) -> threading.RLock:
    key = str(_canonical_path(path))
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


class _InterProcessPathLock:
    """Exclusive advisory lock backed only by the Python standard library.

    A stable sidecar is used instead of locking the journal itself: the journal
    may not exist at first use, and opening it merely to lock would turn the
    distinction between an absent and an empty journal into a race.  OS locks
    are released automatically when a process dies; the harmless sidecar may
    remain on disk.
    """

    def __init__(self, path: Path, *, timeout_sec: float = _LOCK_ACQUIRE_TIMEOUT_SEC) -> None:
        self._path = path
        if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, Real):
            raise PopulationJournalError("population journal lock timeout must be numeric")
        self._timeout_sec = max(0.0, float(timeout_sec))
        self._handle: Any | None = None
        self._locked = False

    def __enter__(self) -> "_InterProcessPathLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Unbuffered mode keeps the Windows lock byte and file position in
            # sync with the descriptor used by msvcrt.locking().
            handle = self._path.open("a+b", buffering=0)
        except OSError as exc:
            raise PopulationJournalError("population journal lock is unavailable") from exc
        self._handle = handle
        try:
            self._ensure_windows_lock_byte()
            self._acquire()
        except BaseException:
            handle.close()
            self._handle = None
            raise
        return self

    def _ensure_windows_lock_byte(self) -> None:
        if os.name != "nt":
            return
        assert self._handle is not None
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            # Concurrent creators may both append a byte.  That is harmless:
            # every participant still locks the first byte only.
            self._handle.write(b"\x00")
            self._handle.flush()

    def _acquire(self) -> None:
        assert self._handle is not None
        deadline = time.monotonic() + self._timeout_sec
        while True:
            try:
                self._handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(
                        self._handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                self._locked = True
                return
            except (BlockingIOError, OSError) as exc:
                if not _is_lock_contention(exc):
                    raise PopulationJournalError(
                        "population journal lock could not be acquired"
                    ) from exc
                if time.monotonic() >= deadline:
                    raise PopulationJournalError(
                        "timed out waiting for the population journal lock"
                    ) from exc
                time.sleep(_LOCK_RETRY_INTERVAL_SEC)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        handle = self._handle
        try:
            if handle is not None and self._locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._locked = False
            self._handle = None
            if handle is not None:
                handle.close()


def _is_lock_contention(exc: OSError) -> bool:
    return (
        isinstance(exc, BlockingIOError)
        or exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
        or getattr(exc, "winerror", None) in {33, 36}
    )


def _journal_file_state(path: Path) -> tuple[int, int, int, int, int] | None:
    """Cheap identity used to avoid rescanning an unchanged append-only file."""

    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PopulationJournalError("population journal is unreadable") from exc
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
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


def _framed_payload(payload: Mapping[str, object]) -> bytes:
    encoded = _canonical_bytes(payload)
    return len(encoded).to_bytes(8, byteorder="big", signed=False) + encoded


def _sha256_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _commit_domains(schema_version: int) -> tuple[bytes, bytes]:
    if schema_version == SCHEMA_VERSION:
        return _GENESIS_DOMAIN_V5, _CYCLE_COMMIT_DOMAIN_V5
    if schema_version == CURRENT_WRITE_SCHEMA:
        return _GENESIS_DOMAIN_V6, _CYCLE_COMMIT_DOMAIN_V6
    raise PopulationJournalError("unsupported population journal schema")


def genesis_cycle_commit(
    journal_id: str,
    *,
    schema_version: int = SCHEMA_VERSION,
) -> str:
    """Return the domain-separated predecessor for the first cycle."""

    if not isinstance(journal_id, str) or not _JOURNAL_ID_RE.fullmatch(journal_id):
        raise PopulationJournalError("journal_id must be a lowercase SHA-256 hex digest")
    digest = hashlib.sha256()
    genesis_domain, _ = _commit_domains(schema_version)
    digest.update(genesis_domain)
    digest.update(
        _framed_payload(
            {
                "journal_id": journal_id,
                "schema_version": schema_version,
            }
        )
    )
    return digest.hexdigest()


def compute_cycle_commit(
    header: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    footer_core: Mapping[str, object],
) -> str:
    """Commit to one exact canonical cycle without trusting serialized hashes.

    Length-prefixing every component makes the byte stream unambiguous.  The
    footer passed here deliberately excludes ``cycle_commit`` to avoid a
    self-reference; exact key sets make that omission explicit and fail closed.
    """

    schema_version = header.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, Integral):
        raise PopulationJournalError("cycle commitment schema version is invalid")
    schema_version = int(schema_version)
    if set(header) != header_keys_for_schema(schema_version):
        raise PopulationJournalError("cycle commitment header schema mismatch")
    if set(footer_core) != FOOTER_CORE_KEYS:
        raise PopulationJournalError("cycle commitment footer schema mismatch")
    if footer_core.get("schema_version") != schema_version:
        raise PopulationJournalError("cycle commitment mixes journal schemas")
    _, cycle_domain = _commit_domains(schema_version)
    digest = hashlib.sha256()
    digest.update(cycle_domain)
    digest.update(_framed_payload(header))
    for row in rows:
        if not isinstance(row, Mapping):
            raise PopulationJournalError("cycle commitment requires decision objects")
        digest.update(_framed_payload(row))
    digest.update(_framed_payload(footer_core))
    return digest.hexdigest()


def make_cycle_id(
    *,
    timeframe: str,
    candle_cutoff_ts: float,
    universe_received_at: float,
    universe_symbols: Sequence[str],
    schema_version: int = CYCLE_IDENTITY_VERSION,
) -> str:
    """Return a stable ID for one point-in-time universe evaluation cycle.

    Built only from the cycle's causal inputs - the bar cutoff, the instant the
    universe response arrived, and the ordered symbol set. Nothing here depends on
    how quickly individual workers finished, so the identity of a cycle cannot
    change with thread scheduling.
    """

    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, Integral)
        or schema_version < 1
    ):
        raise PopulationJournalError("schema_version must be a positive integer")
    clean_symbols = [_bounded_string(symbol, name="universe symbol", max_length=64) for symbol in universe_symbols]
    if len(clean_symbols) > 10_000:
        raise PopulationJournalError("universe contains too many symbols")
    if len(set(clean_symbols)) != len(clean_symbols):
        raise PopulationJournalError("universe symbols must be unique")
    return _sha256_id(
        {
            # Preserve the originally published canonical payload key. Renaming
            # the key would rename every existing cycle even though its meaning
            # is now correctly owned by CYCLE_IDENTITY_VERSION.
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
        if not is_bar_aligned(self.candle_cutoff_ts, self.timeframe):
            raise PopulationJournalError("candle cutoff is not aligned to the timeframe")
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
            if not is_bar_aligned(base_open, self.timeframe) or not is_bar_aligned(
                base_close, self.timeframe
            ):
                raise PopulationJournalError("base bar is not aligned to the timeframe")
            if not math.isclose(base_close - base_open, expected_seconds, rel_tol=0.0, abs_tol=1e-6):
                raise PopulationJournalError("base bar duration differs from timeframe")
            if not math.isclose(
                base_close,
                self.candle_cutoff_ts,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise PopulationJournalError("base bar does not close at the causal cutoff")
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


def update_rows_checksum(digest: Any, payload: Mapping[str, object]) -> None:
    """Add one canonical full decision record to an ordered cycle digest."""

    if not isinstance(payload, Mapping):
        raise PopulationJournalError("row checksum requires decision objects")
    digest.update(_framed_payload(payload))


def rows_checksum(rows: Sequence[Mapping[str, object]]) -> str:
    """Digest over every canonical field of the ordered decision rows.

    Snapshot IDs deliberately exclude wall-clock/provenance fields. Hashing only
    those IDs allowed a valid-looking timestamp or provenance substitution to
    leave the footer unchanged. Length-prefixing the canonical full rows binds
    both membership and all serialized evidence without concatenation ambiguity.
    """

    digest = hashlib.sha256()
    for payload in rows:
        update_rows_checksum(digest, payload)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON number: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_journal_record(raw: bytes, *, line_number: int) -> Mapping[str, Any]:
    if len(raw) > _MAX_JOURNAL_LINE_BYTES:
        raise PopulationJournalError(f"population journal line {line_number} is too large")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PopulationJournalError(
            f"population journal has invalid JSON at line {line_number}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise PopulationJournalError(
            f"population journal record at line {line_number} is not an object"
        )
    return decoded


def _validated_envelope(payload: object) -> CycleEnvelope:
    if not isinstance(payload, Mapping):
        raise PopulationJournalError("cycle header envelope is not an object")
    try:
        envelope = CycleEnvelope.from_dict(payload)
    except CycleEnvelopeError as exc:
        raise PopulationJournalError("cycle header contains an invalid envelope") from exc
    expected_cycle_id = make_cycle_id(
        timeframe=envelope.timeframe,
        candle_cutoff_ts=envelope.candle_cutoff_ts,
        universe_received_at=envelope.universe_timing.received_at,
        universe_symbols=envelope.universe_symbols,
        schema_version=CYCLE_IDENTITY_VERSION,
    )
    if envelope.cycle_id != expected_cycle_id:
        raise PopulationJournalError("cycle envelope ID does not match its ordered universe")
    return envelope


def _envelope_strategy_identity(envelope: CycleEnvelope) -> tuple[str, str, str]:
    """Return the exact strategy namespace shared by every cycle in one file."""

    return (
        envelope.strategy_spec_version,
        envelope.strategy_spec_contract_hash,
        envelope.strategy_spec_instance_hash,
    )


def _validate_cycle_body(
    envelope: CycleEnvelope,
    *,
    declared_rows: int,
    row_symbols: Sequence[str],
) -> None:
    """Bind status, ordered universe and declared/actual decision population."""

    if declared_rows < 0:
        raise PopulationJournalError("cycle row count must not be negative")
    symbols = tuple(row_symbols)
    if envelope.status == "completed":
        if not symbols:
            raise PopulationJournalError("completed cycle must contain decision rows")
        if symbols != envelope.universe_symbols:
            raise PopulationJournalError(
                "completed cycle rows do not match the envelope universe order"
            )
    elif symbols:
        raise PopulationJournalError(
            f"{envelope.status} cycle must not contain decision rows"
        )
    if declared_rows != len(symbols):
        raise PopulationJournalError("declared cycle row count differs from its body")


def _validate_record_against_envelope(
    record: PopulationDecisionV6,
    *,
    envelope: CycleEnvelope,
) -> None:
    """Bind every cycle-owned v6 row fact to the authoritative envelope."""

    if not isinstance(record, PopulationDecisionV6):
        raise PopulationJournalError("cycle-envelope validation requires a v6 row")
    for record_field, envelope_field in (
        ("cycle_id", "cycle_id"),
        ("timeframe", "timeframe"),
        ("candle_cutoff_ts", "candle_cutoff_ts"),
        ("ranking_ready_ts", "ranking_ready_ts"),
        ("cycle_completed_ts", "cycle_completed_ts"),
        ("actionable_ts", "actionable_ts"),
        ("entry_eligible_ts", "entry_eligible_ts"),
        ("entry_bar_open_ts", "entry_bar_open_ts"),
    ):
        if getattr(record, record_field) != getattr(envelope, envelope_field):
            raise PopulationJournalError(
                f"decision row disagrees with cycle envelope: {record_field}"
            )
    if (
        record.universe_request_started_at
        != envelope.universe_timing.request_started_at
    ):
        raise PopulationJournalError(
            "decision row disagrees with cycle envelope: universe_request_started_at"
        )
    if record.universe_received_at != envelope.universe_timing.received_at:
        raise PopulationJournalError(
            "decision row disagrees with cycle envelope: universe_received_at"
        )

    metadata = _thaw_json_value(record.metadata)
    provenance = metadata.get("provenance") if isinstance(metadata, Mapping) else None
    expected_keys = {"strategy_config_hash", "universe_policy_hash"}
    if not isinstance(provenance, Mapping) or set(provenance) != expected_keys:
        raise PopulationJournalError("decision row provenance schema mismatch")
    if provenance.get("strategy_config_hash") != envelope.strategy_spec_instance_hash:
        raise PopulationJournalError(
            "decision row strategy identity differs from cycle envelope"
        )
    if provenance.get("universe_policy_hash") != envelope.universe_policy_hash:
        raise PopulationJournalError(
            "decision row universe policy differs from cycle envelope"
        )


def _validate_cycle_source_timings(
    *,
    envelope: CycleEnvelope,
    benchmark_source_evidence: SourceReadEvidenceV1,
    base_source_evidences: Sequence[SourceReadEvidenceV1],
    higher_timeframe_source_evidences: Sequence[SourceReadEvidenceV1],
) -> None:
    """Require envelope aggregates to exactly derive from persisted evidence."""

    timings_by_source: dict[str, object] = {}
    for timing in envelope.source_timings:
        if timing.source in timings_by_source:
            raise PopulationJournalError(
                "cycle envelope contains duplicate source timing identities"
            )
        timings_by_source[timing.source] = timing
    allowed_sources = {
        envelope.universe_timing.source,
        "contract_details",
        "benchmark",
        "base_ohlcv",
        "higher_timeframe",
    }
    unexpected_sources = set(timings_by_source).difference(allowed_sources)
    if unexpected_sources:
        raise PopulationJournalError(
            "cycle envelope contains an unsupported source timing identity"
        )
    try:
        expected = {
            "benchmark": source_timing_from_evidence(
                benchmark_source_evidence,
                source="benchmark",
            ),
            "base_ohlcv": aggregate_source_timing_from_evidence(
                base_source_evidences,
                source="base_ohlcv",
            ),
            "higher_timeframe": aggregate_source_timing_from_evidence(
                higher_timeframe_source_evidences,
                source="higher_timeframe",
            ),
        }
    except FrameProvenanceError as exc:
        raise PopulationJournalError(
            "source evidence cannot rebuild cycle source timings"
        ) from exc

    if envelope.status == "completed":
        if expected["benchmark"] is None or expected["base_ohlcv"] is None:
            raise PopulationJournalError(
                "completed cycle lacks attempted benchmark or base evidence"
            )
    elif any(value is not None for value in expected.values()):
        raise PopulationJournalError(
            f"{envelope.status} cycle must not contain attempted market evidence"
        )

    for source, projected in expected.items():
        actual = timings_by_source.get(source)
        if projected is None:
            if actual is not None:
                raise PopulationJournalError(
                    f"cycle envelope carries unexpected {source} timing"
                )
        elif actual != projected:
            raise PopulationJournalError(
                f"cycle envelope {source} timing differs from typed evidence"
            )


def _validate_feature_provenance(
    record: PopulationDecision,
    *,
    envelope: CycleEnvelope,
) -> None:
    """Fail early when a feature-bearing row is not bound to its market/envelope.

    ``PopulationJournal`` remains usable for generic diagnostic rows that carry
    neither object. Once either the canonical feature snapshot or provenance is
    present, however, schema v4 requires both and validates the same SHA-256
    links that the strict reader later rebuilds independently.
    """

    metadata = _thaw_json_value(record.metadata)
    if not isinstance(metadata, Mapping):  # guarded by PopulationDecision
        raise PopulationJournalError("record metadata is not a mapping")
    snapshot = metadata.get("feature_snapshot")
    provenance = metadata.get("feature_provenance")
    if snapshot is None and provenance is None:
        return
    if not isinstance(snapshot, Mapping) or not isinstance(provenance, Mapping):
        raise PopulationJournalError(
            "feature snapshot and feature provenance must be present together"
        )
    if set(provenance) != FEATURE_PROVENANCE_KEYS:
        raise PopulationJournalError("feature provenance schema mismatch")
    if provenance.get("envelope_hash") != envelope.envelope_hash():
        raise PopulationJournalError("feature provenance envelope hash mismatch")
    if provenance.get("universe_received_at") != envelope.universe_timing.received_at:
        raise PopulationJournalError("feature provenance universe response mismatch")
    if provenance.get("universe_source_ts") != envelope.universe_timing.source_ts:
        raise PopulationJournalError("feature provenance universe source mismatch")
    if type(provenance.get("universe_cache_hit")) is not bool:
        raise PopulationJournalError("feature provenance cache flag must be boolean")
    if provenance.get("universe_cache_hit") is not envelope.universe_timing.cache_hit:
        raise PopulationJournalError("feature provenance universe cache mismatch")
    # Imported lazily so this low-level journal remains importable while
    # ai.reversal.population_dataset imports PopulationDecision. append_cycle()
    # is only called after module initialization, so the dependency is safe at
    # runtime without turning module import into a cycle.
    from ai.reversal.feature_contract import (
        build_runtime_feature_snapshot,
        market_feature_hash,
    )

    try:
        rebuilt_snapshot = build_runtime_feature_snapshot(
            metadata,
            bar_cutoff_ts=record.candle_cutoff_ts,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PopulationJournalError("feature snapshot cannot be rebuilt") from exc
    if _canonical_bytes(snapshot) != _canonical_bytes(rebuilt_snapshot):
        raise PopulationJournalError("feature snapshot does not match its source metadata")
    recorded_market_hash = provenance.get("market_feature_hash")
    if (
        not isinstance(recorded_market_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", recorded_market_hash)
        or recorded_market_hash
        != market_feature_hash(
            rebuilt_snapshot,
            symbol=record.symbol,
            timeframe_seconds=interval_seconds(record.timeframe),
        )
    ):
        raise PopulationJournalError("feature provenance market feature hash mismatch")


def _validated_source_evidence(payload: object, *, field: str) -> SourceReadEvidenceV1:
    if not isinstance(payload, Mapping):
        raise PopulationJournalError(f"{field} is not an object")
    try:
        return SourceReadEvidenceV1.from_dict(payload)
    except (FrameProvenanceError, TypeError, ValueError) as exc:
        raise PopulationJournalError(f"{field} is invalid") from exc


def _source_cutoff_matches_cycle(
    evidence: SourceReadEvidenceV1,
    *,
    envelope: CycleEnvelope,
) -> bool:
    expected = float(closed_boundary_ts(envelope.candle_cutoff_ts, evidence.timeframe))
    return (
        math.isclose(
            evidence.requested_as_of_ts,
            envelope.candle_cutoff_ts,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            evidence.expected_closed_boundary_ts,
            expected,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )


def _validate_benchmark_source_evidence(
    evidence: SourceReadEvidenceV1,
    *,
    envelope: CycleEnvelope,
) -> None:
    try:
        spec = decode_mexc_strategy_spec_evidence(
            envelope.strategy_spec_payload,
            expected_version=envelope.strategy_spec_version,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PopulationJournalError("strategy spec cannot validate source evidence") from exc
    if evidence.source != "benchmark_ohlcv":
        raise PopulationJournalError("benchmark source identity mismatch")
    if evidence.venue != "mexc_contract":
        raise PopulationJournalError("benchmark venue identity mismatch")
    if evidence.symbol.replace("_", "") != "BTCUSDT":
        raise PopulationJournalError("benchmark symbol identity mismatch")
    if evidence.venue_symbol != "BTC_USDT":
        raise PopulationJournalError("benchmark venue symbol identity mismatch")
    if interval_seconds(evidence.timeframe) != interval_seconds(
        spec.resolved_benchmark_interval
    ):
        raise PopulationJournalError("benchmark timeframe identity mismatch")
    if not _source_cutoff_matches_cycle(evidence, envelope=envelope):
        raise PopulationJournalError("benchmark cutoff identity mismatch")
    if envelope.status == "completed":
        if evidence.outcome == "not_requested":
            raise PopulationJournalError("completed cycle did not request its benchmark")
    elif evidence.outcome != "not_requested":
        raise PopulationJournalError(
            f"{envelope.status} cycle must not request its benchmark"
        )


def _validate_lifecycle_projection(
    record: PopulationDecisionV6,
    *,
    envelope: CycleEnvelope,
) -> None:
    event = record.lifecycle_event
    if event is None:
        if record.action in {"SHORT_ENTRY", "LONG_ENTRY"}:
            raise PopulationJournalError(
                "entry action requires typed lifecycle evidence"
            )
        return
    arm = event.arm
    if arm.symbol != record.symbol:
        raise PopulationJournalError("lifecycle symbol differs from decision row")
    if arm.timeframe_seconds != interval_seconds(record.timeframe):
        raise PopulationJournalError("lifecycle timeframe differs from decision row")
    if (
        arm.strategy_spec_version,
        arm.strategy_spec_contract_hash,
        arm.strategy_spec_instance_hash,
    ) != _envelope_strategy_identity(envelope):
        raise PopulationJournalError("lifecycle strategy identity mismatch")

    if event.confirmation is None:
        current_bundle = arm.raw_input_bundle_hash
        current_open = arm.arm_bar_open_ts
        current_cutoff = arm.arm_candle_cutoff_ts
    else:
        current_bundle = event.confirmation.observation_input_bundle_hash
        current_open = event.confirmation.observation_bar_open_ts
        current_cutoff = event.confirmation.observation_candle_cutoff_ts
    if current_bundle != record.raw_frame_bundle_hash:
        raise PopulationJournalError("lifecycle raw bundle differs from decision row")
    if record.base_bar_open_ts is None or record.base_bar_close_ts is None:
        raise PopulationJournalError("lifecycle event requires a real base bar")
    if not math.isclose(current_open, record.base_bar_open_ts, rel_tol=0.0, abs_tol=1e-6):
        raise PopulationJournalError("lifecycle reference bar differs from decision row")
    if not math.isclose(
        current_cutoff,
        record.candle_cutoff_ts,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise PopulationJournalError("lifecycle cutoff differs from decision row")
    if record.status != "evaluated":
        raise PopulationJournalError("lifecycle event requires evaluated status")

    proposal = event.proposal
    expected_action = (
        "SHORT_ENTRY" if proposal.side is CandidateSide.SHORT else "LONG_ENTRY"
    )
    metadata = _thaw_json_value(record.metadata)
    if proposal.status is ProposalObservationStatus.CREATED:
        if record.action != expected_action:
            raise PopulationJournalError("created proposal does not project to row action")
        if metadata.get("stop_loss") != proposal.stop_price:
            raise PopulationJournalError("proposal stop differs from decision row")
        if metadata.get("take_profit") != proposal.take_profit_price:
            raise PopulationJournalError("proposal target differs from decision row")
    elif record.action != "HOLD":
        raise PopulationJournalError("non-created proposal must project to HOLD")


def _validate_v6_decision_evidence(
    record: PopulationDecisionV6,
    *,
    envelope: CycleEnvelope,
    benchmark_source_evidence: SourceReadEvidenceV1,
) -> None:
    assert record.base_source_evidence is not None
    assert record.higher_timeframe_source_evidence is not None
    base = record.base_source_evidence
    htf = record.higher_timeframe_source_evidence
    try:
        spec = decode_mexc_strategy_spec_evidence(
            envelope.strategy_spec_payload,
            expected_version=envelope.strategy_spec_version,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PopulationJournalError("strategy spec cannot validate source evidence") from exc

    if base.source != "base_ohlcv" or htf.source != "higher_timeframe_ohlcv":
        raise PopulationJournalError("per-symbol source identity mismatch")
    if base.venue != "mexc_contract" or htf.venue != "mexc_contract":
        raise PopulationJournalError("per-symbol venue identity mismatch")
    if benchmark_source_evidence.venue != base.venue or htf.venue != base.venue:
        raise PopulationJournalError("source evidence mixes venues")
    if base.symbol != record.symbol or htf.symbol != record.symbol:
        raise PopulationJournalError("source evidence symbol differs from decision row")
    compact_symbol = record.symbol.replace("_", "")
    if base.venue_symbol != htf.venue_symbol or (
        base.venue_symbol.replace("_", "") != compact_symbol
    ):
        raise PopulationJournalError(
            "per-symbol venue symbol differs from decision row"
        )
    if interval_seconds(base.timeframe) != interval_seconds(record.timeframe):
        raise PopulationJournalError("base source timeframe differs from decision row")
    if interval_seconds(htf.timeframe) != spec.higher_timeframe_interval_seconds:
        raise PopulationJournalError("higher-timeframe source identity mismatch")
    if not _source_cutoff_matches_cycle(base, envelope=envelope):
        raise PopulationJournalError("base source cutoff identity mismatch")
    if not _source_cutoff_matches_cycle(htf, envelope=envelope):
        raise PopulationJournalError("higher-timeframe cutoff identity mismatch")

    expected_bundle = raw_frame_bundle_hash(
        (benchmark_source_evidence, base, htf)
    )
    if record.raw_frame_bundle_hash != expected_bundle:
        raise PopulationJournalError("raw frame bundle hash mismatch")

    if record.status in {"evaluated", "short_history", "strategy_error"} and base.outcome != "fresh":
        raise PopulationJournalError("decision status requires fresh base evidence")
    if record.status == "no_data" and base.outcome != "no_rows":
        raise PopulationJournalError("no_data status requires no_rows base evidence")
    if record.status == "data_error" and base.outcome != "request_failed":
        raise PopulationJournalError("data_error status requires failed base evidence")
    if record.status == "invalid_bar_contract" and base.outcome != "request_failed":
        raise PopulationJournalError(
            "invalid_bar_contract requires failed base evidence"
        )
    if record.status == "data_quality_error" and base.outcome not in {
        "request_failed",
        "stale",
    }:
        raise PopulationJournalError(
            "data_quality_error requires stale or failed base evidence"
        )
    if base.outcome == "fresh":
        if record.base_bar_open_ts != base.last_bar_open_ts:
            raise PopulationJournalError("base bar open differs from source evidence")
        if record.base_bar_close_ts != base.last_bar_close_ts:
            raise PopulationJournalError("base bar close differs from source evidence")
    elif record.base_bar_open_ts is not None or record.base_bar_close_ts is not None:
        raise PopulationJournalError("missing base evidence must not project a base bar")
    for evidence in (base, htf, benchmark_source_evidence):
        if (
            evidence.received_at is not None
            and evidence.received_at > record.decision_ts
        ):
            raise PopulationJournalError("source evidence arrives after decision completion")
    if record.action in {"SHORT_ENTRY", "LONG_ENTRY"}:
        for name, evidence in (
            ("benchmark", benchmark_source_evidence),
            ("higher-timeframe", htf),
        ):
            if (
                evidence.data_through_ts is None
                or not math.isclose(
                    evidence.data_through_ts,
                    evidence.expected_closed_boundary_ts,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ):
                raise PopulationJournalError(
                    f"entry action requires current {name} evidence"
                )
    _validate_lifecycle_projection(record, envelope=envelope)


@dataclass
class _LifecycleChainState:
    """Validate witnessed candidate transitions across ordered journal cycles.

    Rows without an event do not invent a state transition. A fresh initial
    event for the same symbol explicitly right-censors an older candidate. A
    follow-up can only name the exact latest event for its candidate, which
    rejects orphans, forks and fabricated predecessor IDs without pretending
    that process-local pending state was rehydrated after a restart.
    """

    active_candidate_by_symbol: dict[str, str] = field(default_factory=dict)
    latest_by_candidate: dict[str, CandidateLifecycleEventV1] = field(default_factory=dict)
    seen_candidate_ids: set[str] = field(default_factory=set)
    seen_event_ids: set[str] = field(default_factory=set)

    def clone(self) -> "_LifecycleChainState":
        return _LifecycleChainState(
            active_candidate_by_symbol=dict(self.active_candidate_by_symbol),
            latest_by_candidate=dict(self.latest_by_candidate),
            seen_candidate_ids=set(self.seen_candidate_ids),
            seen_event_ids=set(self.seen_event_ids),
        )

    def observe(
        self,
        *,
        symbol: str,
        event: CandidateLifecycleEventV1 | None,
    ) -> None:
        if event is None:
            return
        if event.event_id in self.seen_event_ids:
            raise PopulationJournalError("lifecycle event is duplicated")
        if event.state in {
            CandidateLifecycleState.ARMED,
            CandidateLifecycleState.BYPASSED,
        }:
            if event.arm.candidate_id in self.seen_candidate_ids:
                raise PopulationJournalError("lifecycle candidate is duplicated")
            # A new initial event is an explicit right-censor boundary for any
            # older nonterminal candidate on this symbol.
            old_candidate = self.active_candidate_by_symbol.pop(symbol, None)
            if old_candidate is not None:
                self.latest_by_candidate.pop(old_candidate, None)
            self.seen_candidate_ids.add(event.arm.candidate_id)
            if event.state is CandidateLifecycleState.ARMED:
                self.active_candidate_by_symbol[symbol] = event.arm.candidate_id
                self.latest_by_candidate[event.arm.candidate_id] = event
        else:
            previous = self.latest_by_candidate.get(event.arm.candidate_id)
            if previous is None:
                raise PopulationJournalError(
                    "lifecycle follow-up has no witnessed predecessor"
                )
            try:
                previous.validate_successor(event)
            except LifecycleContractError as exc:
                raise PopulationJournalError(
                    "lifecycle follow-up does not match witnessed predecessor"
                ) from exc
            if event.state in {
                CandidateLifecycleState.CONFIRMED,
                CandidateLifecycleState.INVALIDATED,
                CandidateLifecycleState.EXPIRED,
            }:
                if (
                    self.active_candidate_by_symbol.get(symbol)
                    == event.arm.candidate_id
                ):
                    self.active_candidate_by_symbol.pop(symbol, None)
                self.latest_by_candidate[event.arm.candidate_id] = event
            else:
                self.active_candidate_by_symbol[symbol] = event.arm.candidate_id
                self.latest_by_candidate[event.arm.candidate_id] = event
        self.seen_event_ids.add(event.event_id)

    def finish_cycle(self, envelope: CycleEnvelope) -> None:
        # The absence of a row/event is not a witnessed state transition.
        # Right-censoring occurs only when a later initial event explicitly
        # replaces the candidate for the same symbol.
        del envelope


def _validated_decision_record(
    payload: Mapping[str, Any],
    *,
    schema_version: int,
) -> PopulationDecision:
    """Rebuild one serialized row and re-derive both of its causal digests."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PopulationJournalError("cycle decision row metadata is not an object")
    try:
        extra: dict[str, object] = {}
        if schema_version == CURRENT_WRITE_SCHEMA:
            base_evidence = _validated_source_evidence(
                payload.get("base_source_evidence"), field="base source evidence"
            )
            htf_evidence = _validated_source_evidence(
                payload.get("higher_timeframe_source_evidence"),
                field="higher-timeframe source evidence",
            )
            lifecycle_payload = payload.get("lifecycle_event")
            if lifecycle_payload is None:
                lifecycle_event = None
            elif isinstance(lifecycle_payload, Mapping):
                try:
                    lifecycle_event = CandidateLifecycleEventV1.from_dict(
                        lifecycle_payload
                    )
                except (LifecycleContractError, TypeError, ValueError) as exc:
                    raise PopulationJournalError("lifecycle event is invalid") from exc
            else:
                raise PopulationJournalError("lifecycle event is not an object or null")
            extra = {
                "base_source_evidence": base_evidence,
                "higher_timeframe_source_evidence": htf_evidence,
                # The cycle-scoped benchmark is not available in this row-only
                # decoder.  Construct directly below and re-derive the hashes
                # independently from the serialized causal payload.
                "raw_frame_bundle_hash": payload.get("raw_frame_bundle_hash"),
                "lifecycle_event": lifecycle_event,
            }
        common = dict(
            cycle_id=payload.get("cycle_id"),
            universe_refreshed_at=payload.get("universe_refreshed_at"),
            universe_request_started_at=payload.get("universe_request_started_at"),
            universe_received_at=payload.get("universe_received_at"),
            scan_observed_at=payload.get("scan_observed_at"),
            candle_cutoff_ts=payload.get("candle_cutoff_ts"),
            decision_ts=payload.get("decision_ts"),
            ranking_ready_ts=payload.get("ranking_ready_ts"),
            cycle_completed_ts=payload.get("cycle_completed_ts"),
            actionable_ts=payload.get("actionable_ts"),
            entry_eligible_ts=payload.get("entry_eligible_ts"),
            entry_bar_open_ts=payload.get("entry_bar_open_ts"),
            symbol=payload.get("symbol"),
            timeframe=payload.get("timeframe"),
            status=payload.get("status"),
            base_bar_open_ts=payload.get("base_bar_open_ts"),
            base_bar_close_ts=payload.get("base_bar_close_ts"),
            action=payload.get("action"),
            reason=payload.get("reason"),
            confidence=payload.get("confidence"),
            metadata=metadata,
            cycle_ordinal=payload.get("cycle_ordinal"),
            cycle_size=payload.get("cycle_size"),
            error_code=payload.get("error_code"),
        )
        if schema_version == SCHEMA_VERSION:
            record = PopulationDecision.create(
                schema_version=SCHEMA_VERSION,
                **common,
            )
        elif schema_version == CURRENT_WRITE_SCHEMA:
            record = PopulationDecisionV6(
                schema_version=CURRENT_WRITE_SCHEMA,
                snapshot_id=payload.get("snapshot_id"),
                input_hash=payload.get("input_hash"),
                **common,
                **extra,
            )
        else:
            raise PopulationJournalError("unsupported population journal schema")
    except (PopulationJournalError, TypeError, ValueError) as exc:
        raise PopulationJournalError("population journal contains an invalid decision row") from exc
    # Canonical byte equality catches unknown/missing fields and JSON type drift
    # (for example 1 versus 1.0), while create() above re-derived input_hash and
    # snapshot_id rather than trusting either serialized digest.
    if _canonical_bytes(record.as_dict()) != _canonical_bytes(payload):
        raise PopulationJournalError("population journal decision row does not rebuild exactly")
    if schema_version == CURRENT_WRITE_SCHEMA:
        assert isinstance(record, PopulationDecisionV6)
        causal_payload = {
            "schema_version": CURRENT_WRITE_SCHEMA,
            "cycle_id": record.cycle_id,
            "symbol": record.symbol,
            "timeframe_seconds": interval_seconds(record.timeframe),
            "status": record.status,
            "candle_cutoff_ts": record.candle_cutoff_ts,
            "base_bar_open_ts": record.base_bar_open_ts,
            "base_bar_close_ts": record.base_bar_close_ts,
            "action": record.action,
            "reason": record.reason,
            "confidence": record.confidence,
            "metadata": _causal_metadata(_thaw_json_value(record.metadata)),
            "error_code": record.error_code,
            "raw_frame_bundle_hash": record.raw_frame_bundle_hash,
            "lifecycle_event_id": (
                record.lifecycle_event.event_id
                if record.lifecycle_event is not None
                else None
            ),
        }
        if record.input_hash != _sha256_id(causal_payload):
            raise PopulationJournalError("population journal v6 input hash mismatch")
        expected_snapshot = _sha256_id(
            {
                "schema_version": CURRENT_WRITE_SCHEMA,
                "cycle_id": record.cycle_id,
                "symbol": record.symbol,
                "input_hash": record.input_hash,
            }
        )
        if record.snapshot_id != expected_snapshot:
            raise PopulationJournalError("population journal v6 snapshot ID mismatch")
    return record


@dataclass(frozen=True)
class JournalCheckpointReceipt:
    """Detached description of one trusted journal prefix.

    The receipt is intentionally unsigned.  It becomes trusted only when a
    caller obtains it from an independently protected location (or authenticates
    its canonical payload out of band) and passes it explicitly to the reader.
    Keeping a copy beside the writable journal provides no additional trust.
    """

    receipt_schema_version: int
    journal_schema_version: int
    journal_id: str
    sequence_no: int
    cycle_id: str
    cycle_commit: str
    prefix_length_bytes: int
    prefix_sha256: str
    record_type: str = RECORD_TYPE_CHECKPOINT

    def __post_init__(self) -> None:
        if self.record_type != RECORD_TYPE_CHECKPOINT:
            raise PopulationJournalError("unsupported checkpoint receipt record type")
        if type(self.receipt_schema_version) is not int or (
            self.receipt_schema_version != CHECKPOINT_RECEIPT_SCHEMA_VERSION
        ):
            raise PopulationJournalError("unsupported checkpoint receipt schema")
        if type(self.journal_schema_version) is not int or (
            self.journal_schema_version not in SUPPORTED_JOURNAL_SCHEMAS
        ):
            raise PopulationJournalError("checkpoint targets an unsupported journal schema")
        for name in ("journal_id", "cycle_id", "cycle_commit", "prefix_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _JOURNAL_ID_RE.fullmatch(value):
                raise PopulationJournalError(f"checkpoint {name} must be a SHA-256 digest")
        if isinstance(self.sequence_no, bool) or not isinstance(self.sequence_no, Integral):
            raise PopulationJournalError("checkpoint sequence_no must be an integer")
        if self.sequence_no < 0:
            raise PopulationJournalError("checkpoint sequence_no must not be negative")
        if isinstance(self.prefix_length_bytes, bool) or not isinstance(
            self.prefix_length_bytes, Integral
        ):
            raise PopulationJournalError("checkpoint prefix length must be an integer")
        if self.prefix_length_bytes <= 0:
            raise PopulationJournalError("checkpoint prefix length must be positive")
        object.__setattr__(self, "sequence_no", int(self.sequence_no))
        object.__setattr__(self, "prefix_length_bytes", int(self.prefix_length_bytes))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "JournalCheckpointReceipt":
        expected = {
            "record_type",
            "receipt_schema_version",
            "journal_schema_version",
            "journal_id",
            "sequence_no",
            "cycle_id",
            "cycle_commit",
            "prefix_length_bytes",
            "prefix_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise PopulationJournalError("checkpoint receipt schema mismatch")
        return cls(
            record_type=payload.get("record_type"),  # type: ignore[arg-type]
            receipt_schema_version=payload.get("receipt_schema_version"),  # type: ignore[arg-type]
            journal_schema_version=payload.get("journal_schema_version"),  # type: ignore[arg-type]
            journal_id=payload.get("journal_id"),  # type: ignore[arg-type]
            sequence_no=payload.get("sequence_no"),  # type: ignore[arg-type]
            cycle_id=payload.get("cycle_id"),  # type: ignore[arg-type]
            cycle_commit=payload.get("cycle_commit"),  # type: ignore[arg-type]
            prefix_length_bytes=payload.get("prefix_length_bytes"),  # type: ignore[arg-type]
            prefix_sha256=payload.get("prefix_sha256"),  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "receipt_schema_version": self.receipt_schema_version,
            "journal_schema_version": self.journal_schema_version,
            "journal_id": self.journal_id,
            "sequence_no": self.sequence_no,
            "cycle_id": self.cycle_id,
            "cycle_commit": self.cycle_commit,
            "prefix_length_bytes": self.prefix_length_bytes,
            "prefix_sha256": self.prefix_sha256,
        }


def _v6_causal_ids(
    *,
    cycle_id: str,
    symbol: str,
    timeframe: str,
    status: str,
    candle_cutoff_ts: float,
    base_bar_open_ts: float | None,
    base_bar_close_ts: float | None,
    action: str,
    reason: str,
    confidence: float,
    metadata: Mapping[str, object],
    error_code: str | None,
    raw_frame_bundle_hash_value: str,
    lifecycle_event: CandidateLifecycleEventV1 | None,
) -> tuple[str, str]:
    """Re-derive schema-v6 row identity from its complete causal payload."""

    causal_payload: dict[str, object] = {
        "schema_version": CURRENT_WRITE_SCHEMA,
        "cycle_id": cycle_id,
        "symbol": symbol,
        "timeframe_seconds": interval_seconds(timeframe),
        "status": status,
        "candle_cutoff_ts": _finite_float(
            candle_cutoff_ts, name="candle_cutoff_ts"
        ),
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
        "metadata": _causal_metadata(_thaw_json_value(metadata)),
        "error_code": error_code,
        "raw_frame_bundle_hash": raw_frame_bundle_hash_value,
        "lifecycle_event_id": (
            lifecycle_event.event_id if lifecycle_event is not None else None
        ),
    }
    input_hash = _sha256_id(causal_payload)
    snapshot_id = _sha256_id(
        {
            "schema_version": CURRENT_WRITE_SCHEMA,
            "cycle_id": cycle_id,
            "symbol": symbol,
            "input_hash": input_hash,
        }
    )
    return input_hash, snapshot_id


@dataclass(frozen=True)
class PopulationDecisionV6(PopulationDecision):
    """Schema-v6 row with typed, non-predictive market/lifecycle evidence.

    The benchmark read is cycle-scoped and therefore lives on the header.  It
    is accepted by :meth:`create` solely to derive the latency-free raw bundle
    commitment; strict append/read validation recomputes that commitment from
    all three persisted source records.
    """

    base_source_evidence: SourceReadEvidenceV1 | None = None
    higher_timeframe_source_evidence: SourceReadEvidenceV1 | None = None
    raw_frame_bundle_hash: str = ""
    lifecycle_event: CandidateLifecycleEventV1 | None = None
    benchmark_source_evidence: SourceReadEvidenceV1 | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.schema_version != CURRENT_WRITE_SCHEMA:
            raise PopulationJournalError("v6 decision requires journal schema 6")
        if self.action not in V6_ACTIONS:
            raise PopulationJournalError("v6 decision action is unsupported")
        if not isinstance(self.base_source_evidence, SourceReadEvidenceV1):
            raise PopulationJournalError("v6 decision requires base source evidence")
        if not isinstance(
            self.higher_timeframe_source_evidence, SourceReadEvidenceV1
        ):
            raise PopulationJournalError(
                "v6 decision requires higher-timeframe source evidence"
            )
        if not isinstance(self.raw_frame_bundle_hash, str) or not _JOURNAL_ID_RE.fullmatch(
            self.raw_frame_bundle_hash
        ):
            raise PopulationJournalError("raw_frame_bundle_hash must be a SHA-256 digest")
        if self.lifecycle_event is not None and not isinstance(
            self.lifecycle_event, CandidateLifecycleEventV1
        ):
            raise PopulationJournalError(
                "lifecycle_event must be CandidateLifecycleEventV1 or null"
            )
        if self.benchmark_source_evidence is not None and not isinstance(
            self.benchmark_source_evidence, SourceReadEvidenceV1
        ):
            raise PopulationJournalError("benchmark_source_evidence is invalid")
        expected_input_hash, expected_snapshot_id = _v6_causal_ids(
            cycle_id=self.cycle_id,
            symbol=self.symbol,
            timeframe=self.timeframe,
            status=self.status,
            candle_cutoff_ts=self.candle_cutoff_ts,
            base_bar_open_ts=self.base_bar_open_ts,
            base_bar_close_ts=self.base_bar_close_ts,
            action=self.action,
            reason=self.reason,
            confidence=self.confidence,
            metadata=self.metadata,
            error_code=self.error_code,
            raw_frame_bundle_hash_value=self.raw_frame_bundle_hash,
            lifecycle_event=self.lifecycle_event,
        )
        if self.input_hash != expected_input_hash:
            raise PopulationJournalError("population journal v6 input hash mismatch")
        if self.snapshot_id != expected_snapshot_id:
            raise PopulationJournalError("population journal v6 snapshot ID mismatch")

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
        base_source_evidence: SourceReadEvidenceV1,
        higher_timeframe_source_evidence: SourceReadEvidenceV1,
        benchmark_source_evidence: SourceReadEvidenceV1,
        lifecycle_event: CandidateLifecycleEventV1 | None = None,
        cycle_ordinal: int = 0,
        cycle_size: int = 1,
        error_code: str | None = None,
        schema_version: int = CURRENT_WRITE_SCHEMA,
    ) -> "PopulationDecisionV6":
        if schema_version != CURRENT_WRITE_SCHEMA:
            raise PopulationJournalError("v6 decision requires journal schema 6")
        for name, evidence in (
            ("base_source_evidence", base_source_evidence),
            ("higher_timeframe_source_evidence", higher_timeframe_source_evidence),
            ("benchmark_source_evidence", benchmark_source_evidence),
        ):
            if not isinstance(evidence, SourceReadEvidenceV1):
                raise PopulationJournalError(f"{name} is invalid")
        if lifecycle_event is not None and not isinstance(
            lifecycle_event, CandidateLifecycleEventV1
        ):
            raise PopulationJournalError("lifecycle_event is invalid")
        if not isinstance(metadata, Mapping):
            raise PopulationJournalError("metadata must be a mapping")
        frozen_metadata = _freeze_json_value(metadata, path="metadata")
        bundle_hash = raw_frame_bundle_hash(
            (
                benchmark_source_evidence,
                base_source_evidence,
                higher_timeframe_source_evidence,
            )
        )
        input_hash, snapshot_id = _v6_causal_ids(
            cycle_id=cycle_id,
            symbol=symbol,
            timeframe=timeframe,
            status=status,
            candle_cutoff_ts=candle_cutoff_ts,
            base_bar_open_ts=base_bar_open_ts,
            base_bar_close_ts=base_bar_close_ts,
            action=action,
            reason=reason,
            confidence=confidence,
            metadata=frozen_metadata,
            error_code=error_code,
            raw_frame_bundle_hash_value=bundle_hash,
            lifecycle_event=lifecycle_event,
        )
        return cls(
            schema_version=CURRENT_WRITE_SCHEMA,
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
            base_source_evidence=base_source_evidence,
            higher_timeframe_source_evidence=higher_timeframe_source_evidence,
            raw_frame_bundle_hash=bundle_hash,
            lifecycle_event=lifecycle_event,
            benchmark_source_evidence=benchmark_source_evidence,
        )

    def as_dict(self) -> dict[str, object]:
        payload = super().as_dict()
        assert self.base_source_evidence is not None
        assert self.higher_timeframe_source_evidence is not None
        payload.update(
            {
                "base_source_evidence": self.base_source_evidence.as_dict(),
                "higher_timeframe_source_evidence": (
                    self.higher_timeframe_source_evidence.as_dict()
                ),
                "raw_frame_bundle_hash": self.raw_frame_bundle_hash,
                "lifecycle_event": (
                    self.lifecycle_event.as_dict()
                    if self.lifecycle_event is not None
                    else None
                ),
            }
        )
        return payload


@dataclass(frozen=True)
class _JournalInspection:
    journal_schema_version: int | None
    journal_id: str | None
    strategy_spec_identity: tuple[str, str, str] | None
    evidence_contracts: Mapping[str, str] | None
    lifecycle_chain: _LifecycleChainState
    cycle_ids: tuple[str, ...]
    cycle_commits: tuple[str, ...]
    prefix_lengths: tuple[int, ...]
    prefix_sha256s: tuple[str, ...]
    tail_digest: Any

    @property
    def last_cycle_id(self) -> str | None:
        return self.cycle_ids[-1] if self.cycle_ids else None


class PopulationJournal:
    """Append-only cycle log: one header, its decision rows, then a footer.

    The envelope is written once per cycle. Copying it onto every row made the
    file quadratic in universe size and pushed the ordered symbol list past the
    bounds that keep arbitrary per-row metadata safe.
    """

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self._path = _canonical_path(Path(path))
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._path_lock = _process_path_lock(self._path)
        self._lock_path = self._path.with_name(f".{self._path.name}.lock")
        self._runtime_lock_path = self._path.with_name(
            f".{self._path.name}.runtime.lock"
        )
        self._runtime_path_lock = _process_path_lock(self._runtime_lock_path)
        self._file_state: tuple[int, int, int, int, int] | None = None
        if self._enabled:
            # Initialization participates in the same critical section as an
            # append.  Otherwise one instance could inspect a half-written
            # batch while another instance is between header and footer.
            with self._path_lock, _InterProcessPathLock(self._lock_path):
                observed_state = _journal_file_state(self._path)
                inspection = self._inspect_existing_file()
                self._adopt_inspection(inspection)
                self._file_state = _journal_file_state(self._path)
                if self._file_state != observed_state:
                    raise PopulationJournalError(
                        "population journal changed while it was being validated"
                    )
        else:
            self._journal_schema_version = None
            self._journal_id = None
            self._strategy_spec_identity = None
            self._evidence_contracts = None
            self._last_cycle_id = None
            self._cycle_ids: set[str] = set()
            self._cycle_ids_ordered: tuple[str, ...] = ()
            self._cycle_commits: tuple[str, ...] = ()
            self._prefix_lengths: tuple[int, ...] = ()
            self._prefix_sha256s: tuple[str, ...] = ()
            self._tail_digest = hashlib.sha256()
            self._lifecycle_chain = _LifecycleChainState()

    def _adopt_inspection(self, inspection: _JournalInspection) -> None:
        self._journal_schema_version = inspection.journal_schema_version
        self._journal_id = inspection.journal_id
        self._strategy_spec_identity = inspection.strategy_spec_identity
        self._evidence_contracts = inspection.evidence_contracts
        # This is evidence-validation state, not strategy runtime state.  The
        # scanner still does not rehydrate a pending candidate on restart.
        self._lifecycle_chain = inspection.lifecycle_chain.clone()
        self._last_cycle_id = inspection.last_cycle_id
        self._cycle_ids = set(inspection.cycle_ids)
        self._cycle_ids_ordered = inspection.cycle_ids
        self._cycle_commits = inspection.cycle_commits
        self._prefix_lengths = inspection.prefix_lengths
        self._prefix_sha256s = inspection.prefix_sha256s
        self._tail_digest = inspection.tail_digest.copy()

    def _refresh_if_changed(self) -> None:
        """Adopt cycles written by another instance while holding both locks."""

        observed_state = _journal_file_state(self._path)
        if observed_state == self._file_state:
            return
        inspection = self._inspect_existing_file()
        validated_state = _journal_file_state(self._path)
        if validated_state != observed_state:
            # A writer that ignores our sidecar lock changed the evidence while
            # it was being validated.  Never append on top of that ambiguity.
            raise PopulationJournalError(
                "population journal changed while it was being validated"
            )
        old_count = len(self._cycle_commits)
        if old_count:
            if inspection.journal_schema_version != self._journal_schema_version:
                raise PopulationJournalError("population journal schema was replaced")
            if inspection.journal_id != self._journal_id:
                raise PopulationJournalError("population journal identity was replaced")
            if inspection.evidence_contracts != self._evidence_contracts:
                raise PopulationJournalError("population journal evidence identity was replaced")
            if len(inspection.cycle_commits) < old_count:
                raise PopulationJournalError("population journal history was rolled back")
            if inspection.cycle_commits[:old_count] != self._cycle_commits:
                raise PopulationJournalError("population journal history was rewritten")
            if inspection.prefix_lengths[:old_count] != self._prefix_lengths or (
                inspection.prefix_sha256s[:old_count] != self._prefix_sha256s
            ):
                raise PopulationJournalError("population journal prefix bytes were rewritten")
        self._adopt_inspection(inspection)
        self._file_state = validated_state

    def _inspect_existing_file(self) -> _JournalInspection:
        """Validate every existing cycle before permitting another append.

        Looking only at the first header and final footer allowed an incompatible
        or corrupt cycle in the middle of a file to survive a restart.  This
        state machine validates the complete outer journal, including ordered
        population membership and A-B-A duplicate cycles, without importing the
        feature reader (which would create a dependency cycle).
        """

        if not self._path.exists() or self._path.stat().st_size == 0:
            return _JournalInspection(
                journal_schema_version=None,
                journal_id=None,
                strategy_spec_identity=None,
                evidence_contracts=None,
                lifecycle_chain=_LifecycleChainState(),
                cycle_ids=(),
                cycle_commits=(),
                prefix_lengths=(),
                prefix_sha256s=(),
                tail_digest=hashlib.sha256(),
            )
        try:
            with self._path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                ends_with_newline = handle.read(1) == b"\n"
        except OSError as exc:
            raise PopulationJournalError("population journal is unreadable") from exc

        # A previous process that died mid-write leaves a line without its
        # newline. Appending would concatenate two JSON objects into one
        # unparseable line, so stop and say how to recover.
        if not ends_with_newline:
            raise PopulationJournalError(
                "population journal ends without a newline; it was truncated mid-write. "
                "Move the file aside and start a new one rather than appending to it."
            )

        current_envelope: CycleEnvelope | None = None
        current_header: Mapping[str, object] | None = None
        current_cycle_id: str | None = None
        current_envelope_hash: str | None = None
        current_journal_id: str | None = None
        current_sequence_no: int | None = None
        current_prev_commit: str | None = None
        current_benchmark_evidence: SourceReadEvidenceV1 | None = None
        declared_rows = 0
        row_symbols: list[str] = []
        row_payloads: list[Mapping[str, object]] = []
        base_source_evidences: list[SourceReadEvidenceV1] = []
        higher_timeframe_source_evidences: list[SourceReadEvidenceV1] = []
        rows_digest = hashlib.sha256()
        seen_symbols: set[str] = set()
        seen_snapshots: set[str] = set()
        completed_ids: set[str] = set()
        completed_cycle_ids: list[str] = []
        completed_commits: list[str] = []
        prefix_lengths: list[int] = []
        prefix_sha256s: list[str] = []
        journal_id: str | None = None
        file_digest = hashlib.sha256()
        prefix_length = 0
        strategy_spec_identity: tuple[str, str, str] | None = None
        journal_schema_version: int | None = None
        evidence_contracts: Mapping[str, str] | None = None
        lifecycle_chain = _LifecycleChainState()

        try:
            handle = self._path.open("rb")
        except OSError as exc:
            raise PopulationJournalError("population journal is unreadable") from exc
        with handle:
            for line_number, raw in enumerate(handle, start=1):
                file_digest.update(raw)
                prefix_length += len(raw)
                if not raw.strip():
                    raise PopulationJournalError("population journal contains a blank line")
                payload = _decode_journal_record(raw, line_number=line_number)
                existing_version = payload.get("schema_version")
                if (
                    isinstance(existing_version, bool)
                    or not isinstance(existing_version, Integral)
                    or int(existing_version) not in SUPPORTED_JOURNAL_SCHEMAS
                ):
                    raise PopulationJournalError(
                        f"population journal was written by schema {existing_version!r}, "
                        f"supported schemas are {sorted(SUPPORTED_JOURNAL_SCHEMAS)}"
                    )
                existing_version = int(existing_version)
                if journal_schema_version is None:
                    journal_schema_version = existing_version
                elif existing_version != journal_schema_version:
                    raise PopulationJournalError("population journal mixes schema versions")
                record_type = payload.get("record_type")

                if record_type == RECORD_TYPE_HEADER:
                    if set(payload) != header_keys_for_schema(existing_version):
                        raise PopulationJournalError("cycle header schema mismatch")
                    if current_envelope is not None:
                        raise PopulationJournalError(
                            "population journal starts a new cycle before closing the previous one"
                        )
                    current_envelope = _validated_envelope(payload.get("envelope"))
                    current_strategy_identity = _envelope_strategy_identity(
                        current_envelope
                    )
                    if strategy_spec_identity is None:
                        strategy_spec_identity = current_strategy_identity
                    elif current_strategy_identity != strategy_spec_identity:
                        raise PopulationJournalError(
                            "population journal mixes strategy identities"
                        )
                    current_header = payload
                    if existing_version == CURRENT_WRITE_SCHEMA:
                        current_contracts = _validated_evidence_contracts(
                            payload.get("evidence_contracts")
                        )
                        if evidence_contracts is None:
                            evidence_contracts = current_contracts
                        elif dict(current_contracts) != dict(evidence_contracts):
                            raise PopulationJournalError(
                                "population journal mixes evidence contract identities"
                            )
                        current_benchmark_evidence = _validated_source_evidence(
                            payload.get("benchmark_source_evidence"),
                            field="benchmark source evidence",
                        )
                        _validate_benchmark_source_evidence(
                            current_benchmark_evidence,
                            envelope=current_envelope,
                        )
                    else:
                        current_benchmark_evidence = None
                    current_cycle_id = current_envelope.cycle_id
                    if payload.get("cycle_id") != current_cycle_id:
                        raise PopulationJournalError("cycle header ID differs from its envelope")
                    if current_cycle_id in completed_ids:
                        raise PopulationJournalError("population journal contains a duplicate cycle")
                    raw_journal_id = payload.get("journal_id")
                    if not isinstance(raw_journal_id, str) or not _JOURNAL_ID_RE.fullmatch(
                        raw_journal_id
                    ):
                        raise PopulationJournalError("cycle header has an invalid journal ID")
                    if journal_id is None:
                        journal_id = raw_journal_id
                    elif raw_journal_id != journal_id:
                        raise PopulationJournalError("population journal mixes journal IDs")
                    current_journal_id = raw_journal_id
                    raw_sequence = payload.get("sequence_no")
                    if isinstance(raw_sequence, bool) or not isinstance(raw_sequence, Integral):
                        raise PopulationJournalError("cycle sequence number must be an integer")
                    current_sequence_no = int(raw_sequence)
                    if current_sequence_no != len(completed_commits):
                        raise PopulationJournalError("population journal sequence is not contiguous")
                    expected_prev = (
                        genesis_cycle_commit(
                            raw_journal_id,
                            schema_version=existing_version,
                        )
                        if not completed_commits
                        else completed_commits[-1]
                    )
                    current_prev_commit = payload.get("prev_cycle_commit")  # type: ignore[assignment]
                    if current_prev_commit != expected_prev:
                        raise PopulationJournalError("population journal chain predecessor mismatch")
                    raw_count = payload.get("row_count")
                    if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
                        raise PopulationJournalError("cycle header row count must be an integer")
                    declared_rows = int(raw_count)
                    if declared_rows < 0:
                        raise PopulationJournalError("cycle row count must not be negative")
                    current_envelope_hash = current_envelope.envelope_hash()
                    if payload.get("envelope_hash") != current_envelope_hash:
                        raise PopulationJournalError("cycle header envelope hash mismatch")
                    row_symbols = []
                    row_payloads = []
                    base_source_evidences = []
                    higher_timeframe_source_evidences = []
                    rows_digest = hashlib.sha256()
                    seen_symbols = set()
                    seen_snapshots = set()
                    continue

                if current_envelope is None or current_cycle_id is None:
                    raise PopulationJournalError("population journal row precedes its cycle header")

                if record_type == RECORD_TYPE_DECISION:
                    record = _validated_decision_record(
                        payload,
                        schema_version=existing_version,
                    )
                    if record.cycle_id != current_cycle_id:
                        raise PopulationJournalError("decision row belongs to another cycle")
                    if (
                        record.cycle_ordinal != len(row_symbols)
                        or record.cycle_size != declared_rows
                    ):
                        raise PopulationJournalError("cycle decision rows are incomplete or unordered")
                    if current_envelope_hash is None:  # guarded by the header state
                        raise PopulationJournalError("cycle header envelope hash is absent")
                    _validate_feature_provenance(
                        record,
                        envelope=current_envelope,
                    )
                    if existing_version == CURRENT_WRITE_SCHEMA:
                        if not isinstance(record, PopulationDecisionV6):
                            raise PopulationJournalError("v6 row decoder returned wrong type")
                        if current_benchmark_evidence is None:
                            raise PopulationJournalError(
                                "v6 cycle header lacks benchmark evidence"
                            )
                        _validate_record_against_envelope(
                            record,
                            envelope=current_envelope,
                        )
                        _validate_v6_decision_evidence(
                            record,
                            envelope=current_envelope,
                            benchmark_source_evidence=current_benchmark_evidence,
                        )
                        lifecycle_chain.observe(
                            symbol=record.symbol,
                            event=record.lifecycle_event,
                        )
                        assert record.base_source_evidence is not None
                        assert record.higher_timeframe_source_evidence is not None
                        base_source_evidences.append(record.base_source_evidence)
                        higher_timeframe_source_evidences.append(
                            record.higher_timeframe_source_evidence
                        )
                    symbol = record.symbol
                    snapshot_id = record.snapshot_id
                    if symbol in seen_symbols or snapshot_id in seen_snapshots:
                        raise PopulationJournalError("cycle contains duplicate symbols or snapshots")
                    seen_symbols.add(symbol)
                    seen_snapshots.add(snapshot_id)
                    row_symbols.append(symbol)
                    row_payloads.append(payload)
                    update_rows_checksum(rows_digest, payload)
                    continue

                if record_type == RECORD_TYPE_FOOTER:
                    if set(payload) != FOOTER_KEYS:
                        raise PopulationJournalError("cycle footer schema mismatch")
                    if payload.get("cycle_id") != current_cycle_id:
                        raise PopulationJournalError("cycle footer ID mismatch")
                    footer_count = payload.get("row_count")
                    if isinstance(footer_count, bool) or not isinstance(footer_count, Integral):
                        raise PopulationJournalError("cycle footer row count must be an integer")
                    if int(footer_count) != declared_rows or declared_rows != len(row_symbols):
                        raise PopulationJournalError("cycle header/footer/body row counts disagree")
                    if payload.get("envelope_hash") != current_envelope_hash:
                        raise PopulationJournalError("cycle footer envelope hash mismatch")
                    if payload.get("rows_checksum") != rows_digest.hexdigest():
                        raise PopulationJournalError("cycle footer rows checksum mismatch")
                    if payload.get("journal_id") != current_journal_id:
                        raise PopulationJournalError("cycle footer journal ID mismatch")
                    if payload.get("sequence_no") != current_sequence_no:
                        raise PopulationJournalError("cycle footer sequence mismatch")
                    if payload.get("prev_cycle_commit") != current_prev_commit:
                        raise PopulationJournalError("cycle footer predecessor mismatch")
                    if current_header is None:
                        raise PopulationJournalError("cycle header payload is absent")
                    footer_core = {
                        key: value for key, value in payload.items() if key != "cycle_commit"
                    }
                    expected_commit = compute_cycle_commit(
                        current_header,
                        row_payloads,
                        footer_core,
                    )
                    recorded_commit = payload.get("cycle_commit")
                    if (
                        not isinstance(recorded_commit, str)
                        or not _JOURNAL_ID_RE.fullmatch(recorded_commit)
                        or recorded_commit != expected_commit
                    ):
                        raise PopulationJournalError("cycle commitment mismatch")
                    _validate_cycle_body(
                        current_envelope,
                        declared_rows=declared_rows,
                        row_symbols=row_symbols,
                    )
                    if journal_schema_version == CURRENT_WRITE_SCHEMA:
                        if current_benchmark_evidence is None:
                            raise PopulationJournalError(
                                "v6 cycle header lacks benchmark evidence"
                            )
                        _validate_cycle_source_timings(
                            envelope=current_envelope,
                            benchmark_source_evidence=current_benchmark_evidence,
                            base_source_evidences=base_source_evidences,
                            higher_timeframe_source_evidences=(
                                higher_timeframe_source_evidences
                            ),
                        )
                    lifecycle_chain.finish_cycle(current_envelope)
                    completed_ids.add(current_cycle_id)
                    completed_cycle_ids.append(current_cycle_id)
                    completed_commits.append(expected_commit)
                    prefix_lengths.append(prefix_length)
                    prefix_sha256s.append(file_digest.hexdigest())
                    current_envelope = None
                    current_header = None
                    current_cycle_id = None
                    current_envelope_hash = None
                    current_journal_id = None
                    current_sequence_no = None
                    current_prev_commit = None
                    current_benchmark_evidence = None
                    declared_rows = 0
                    row_symbols = []
                    row_payloads = []
                    base_source_evidences = []
                    higher_timeframe_source_evidences = []
                    rows_digest = hashlib.sha256()
                    continue

                raise PopulationJournalError(
                    f"unknown population journal record type: {record_type!r}"
                )

        if current_envelope is not None:
            raise PopulationJournalError("population journal ends with an incomplete cycle")
        if not completed_ids:
            raise PopulationJournalError("population journal contains no complete cycles")
        return _JournalInspection(
            journal_schema_version=journal_schema_version,
            journal_id=journal_id,
            strategy_spec_identity=strategy_spec_identity,
            evidence_contracts=evidence_contracts,
            lifecycle_chain=lifecycle_chain,
            cycle_ids=tuple(completed_cycle_ids),
            cycle_commits=tuple(completed_commits),
            prefix_lengths=tuple(prefix_lengths),
            prefix_sha256s=tuple(prefix_sha256s),
            tail_digest=file_digest,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def runtime_session(self) -> Iterator[None]:
        """Exclusively own this scanner runtime across threads and processes.

        The journal's append lock protects bytes but cannot roll back strategy
        lifecycle state changed before a losing duplicate append. This separate
        non-blocking lifetime lock makes a second scanner fail before its first
        market request. The sidecar may remain, but its OS byte lock is released
        automatically when the owner exits or crashes.
        """

        if not self._runtime_path_lock.acquire(blocking=False):
            raise PopulationJournalError("population scanner runtime is already active")
        os_lock = _InterProcessPathLock(self._runtime_lock_path, timeout_sec=0.0)
        try:
            try:
                os_lock.__enter__()
            except PopulationJournalError as exc:
                if "timed out waiting" in str(exc):
                    raise PopulationJournalError(
                        "population scanner runtime is already active"
                    ) from exc
                raise
            try:
                yield
            finally:
                os_lock.__exit__(None, None, None)
        finally:
            self._runtime_path_lock.release()

    def contains_cycle(self, cycle_id: str) -> bool:
        """Return membership from a fully refreshed, lock-protected view."""

        if not isinstance(cycle_id, str) or not _JOURNAL_ID_RE.fullmatch(cycle_id):
            raise PopulationJournalError("cycle_id must be a lowercase SHA-256 digest")
        if not self._enabled:
            return False
        with self._lock, self._path_lock, _InterProcessPathLock(self._lock_path):
            self._refresh_if_changed()
            return cycle_id in self._cycle_ids

    def checkpoint_receipt(self) -> JournalCheckpointReceipt:
        """Describe the latest fsynced prefix for external anchoring.

        Returning this object does not make it trusted.  The caller must move or
        authenticate it outside the journal writer's trust domain and later pass
        it explicitly to the dataset verification API.
        """

        if not self._enabled:
            raise PopulationJournalError("disabled journal has no checkpoint")
        with self._lock, self._path_lock, _InterProcessPathLock(self._lock_path):
            self._refresh_if_changed()
            if (
                self._journal_id is None
                or self._journal_schema_version is None
                or not self._cycle_commits
                or not self._cycle_ids_ordered
            ):
                raise PopulationJournalError("empty journal has no checkpoint")
            return JournalCheckpointReceipt(
                receipt_schema_version=CHECKPOINT_RECEIPT_SCHEMA_VERSION,
                journal_schema_version=self._journal_schema_version,
                journal_id=self._journal_id,
                sequence_no=len(self._cycle_commits) - 1,
                cycle_id=self._cycle_ids_ordered[-1],
                cycle_commit=self._cycle_commits[-1],
                prefix_length_bytes=self._prefix_lengths[-1],
                prefix_sha256=self._prefix_sha256s[-1],
            )

    def append_cycle(
        self,
        records: Sequence[PopulationDecisionV6],
        *,
        envelope: object,
        benchmark_source_evidence: SourceReadEvidenceV1 | None = None,
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
        records = tuple(records)
        # A schema-v5 journal is a frozen historical evidence file.  Reject an
        # append before interpreting any schema-v6 arguments; callers must not
        # be able to get a misleading v6 validation error from a read-only v5
        # target.  The same guard is repeated after refresh under the OS lock.
        if self._journal_schema_version == SCHEMA_VERSION:
            raise PopulationJournalError(
                "population journal schema v5 is frozen read-only; use a separate v6 file"
            )

        try:
            envelope_payload = envelope.as_dict()
        except (AttributeError, TypeError, ValueError) as exc:
            raise PopulationJournalError("cycle envelope cannot be serialized") from exc
        validated_envelope = _validated_envelope(envelope_payload)
        incoming_strategy_identity = _envelope_strategy_identity(validated_envelope)
        if validated_envelope.status == "completed" and not records:
            raise PopulationJournalError("completed cycle must contain decision rows")
        if benchmark_source_evidence is None and records:
            inferred = {
                record.benchmark_source_evidence
                for record in records
                if isinstance(record, PopulationDecisionV6)
            }
            if len(inferred) == 1:
                benchmark_source_evidence = inferred.pop()
        if benchmark_source_evidence is None and validated_envelope.status in {
            "empty_universe",
            "error",
        }:
            spec = decode_mexc_strategy_spec_evidence(
                validated_envelope.strategy_spec_payload,
                expected_version=validated_envelope.strategy_spec_version,
            )
            benchmark_source_evidence = SourceReadEvidenceV1.not_requested(
                source="benchmark_ohlcv",
                venue="mexc_contract",
                symbol="BTCUSDT",
                venue_symbol="BTC_USDT",
                timeframe=spec.resolved_benchmark_interval,
                requested_as_of_ts=validated_envelope.candle_cutoff_ts,
                reason="cycle_not_completed",
            )
        if not isinstance(benchmark_source_evidence, SourceReadEvidenceV1):
            raise PopulationJournalError("v6 cycle requires benchmark source evidence")
        _validate_benchmark_source_evidence(
            benchmark_source_evidence,
            envelope=validated_envelope,
        )
        incoming_evidence_contracts = _validated_evidence_contracts(
            evidence_contracts_payload()
        )
        cycle_id = envelope_payload.get("cycle_id")
        if not isinstance(cycle_id, str) or not re.fullmatch(r"[0-9a-f]{64}", cycle_id):
            raise PopulationJournalError("cycle envelope has an invalid cycle ID")
        envelope_hash = validated_envelope.envelope_hash()

        rows: list[bytes] = []
        row_payloads: list[Mapping[str, object]] = []
        for record in records:
            if not isinstance(record, PopulationDecisionV6):
                raise PopulationJournalError(
                    "records must contain PopulationDecisionV6 instances"
                )
            if record.schema_version != CURRENT_WRITE_SCHEMA:
                raise PopulationJournalError("append batch mixes journal schema versions")
            if record.cycle_id != cycle_id:
                raise PopulationJournalError("decision row does not belong to the envelope cycle")
            _validate_record_against_envelope(record, envelope=validated_envelope)
            _validate_feature_provenance(record, envelope=validated_envelope)
            _validate_v6_decision_evidence(
                record,
                envelope=validated_envelope,
                benchmark_source_evidence=benchmark_source_evidence,
            )
            row_payload = record.as_dict()
            row_payloads.append(row_payload)
            rows.append(_canonical_bytes(row_payload) + b"\n")

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
        _validate_cycle_body(
            validated_envelope,
            declared_rows=expected_size,
            row_symbols=[record.symbol for record in records],
        )
        _validate_cycle_source_timings(
            envelope=validated_envelope,
            benchmark_source_evidence=benchmark_source_evidence,
            base_source_evidences=[
                record.base_source_evidence for record in records
                if record.base_source_evidence is not None
            ],
            higher_timeframe_source_evidences=[
                record.higher_timeframe_source_evidence for record in records
                if record.higher_timeframe_source_evidence is not None
            ],
        )

        with self._lock, self._path_lock, _InterProcessPathLock(self._lock_path):
            # Another object/process may have appended since this instance was
            # constructed.  A cheap file fingerprint makes unchanged appends
            # O(1), while any external change is fully validated before use.
            self._refresh_if_changed()
            if self._journal_schema_version == SCHEMA_VERSION:
                raise PopulationJournalError(
                    "population journal schema v5 is frozen read-only; use a separate v6 file"
                )
            if self._journal_schema_version not in (None, CURRENT_WRITE_SCHEMA):
                raise PopulationJournalError("population journal uses another schema")
            if (
                self._strategy_spec_identity is not None
                and incoming_strategy_identity != self._strategy_spec_identity
            ):
                raise PopulationJournalError(
                    "population journal mixes strategy identities"
                )
            if self._evidence_contracts is not None and dict(
                incoming_evidence_contracts
            ) != dict(self._evidence_contracts):
                raise PopulationJournalError(
                    "population journal mixes evidence contract identities"
                )
            if cycle_id in self._cycle_ids:
                return False
            next_lifecycle_chain = self._lifecycle_chain.clone()
            for record in records:
                next_lifecycle_chain.observe(
                    symbol=record.symbol,
                    event=record.lifecycle_event,
                )
            next_lifecycle_chain.finish_cycle(validated_envelope)
            journal_id = self._journal_id or secrets.token_hex(32)
            sequence_no = len(self._cycle_commits)
            prev_cycle_commit = (
                self._cycle_commits[-1]
                if self._cycle_commits
                else genesis_cycle_commit(
                    journal_id,
                    schema_version=CURRENT_WRITE_SCHEMA,
                )
            )
            header_payload: dict[str, object] = {
                "record_type": RECORD_TYPE_HEADER,
                "schema_version": CURRENT_WRITE_SCHEMA,
                "journal_id": journal_id,
                "sequence_no": sequence_no,
                "prev_cycle_commit": prev_cycle_commit,
                "cycle_id": cycle_id,
                "row_count": expected_size,
                "envelope_hash": envelope_hash,
                "envelope": envelope_payload,
                "evidence_contracts": dict(incoming_evidence_contracts),
                "benchmark_source_evidence": benchmark_source_evidence.as_dict(),
            }
            footer_core: dict[str, object] = {
                "record_type": RECORD_TYPE_FOOTER,
                "schema_version": CURRENT_WRITE_SCHEMA,
                "journal_id": journal_id,
                "sequence_no": sequence_no,
                "prev_cycle_commit": prev_cycle_commit,
                "cycle_id": cycle_id,
                "row_count": expected_size,
                "envelope_hash": envelope_hash,
                "rows_checksum": rows_checksum(row_payloads),
            }
            cycle_commit = compute_cycle_commit(
                header_payload,
                row_payloads,
                footer_core,
            )
            footer_payload = {**footer_core, "cycle_commit": cycle_commit}
            header = _canonical_bytes(header_payload) + b"\n"
            footer = _canonical_bytes(footer_payload) + b"\n"
            batch = header + b"".join(rows) + footer
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("ab") as handle:
                written = handle.write(batch)
                if written != len(batch):
                    raise OSError("population journal batch write was incomplete")
                handle.flush()
                os.fsync(handle.fileno())
            next_digest = self._tail_digest.copy()
            next_digest.update(batch)
            previous_length = self._prefix_lengths[-1] if self._prefix_lengths else 0
            next_length = previous_length + len(batch)
            self._journal_id = journal_id
            self._journal_schema_version = CURRENT_WRITE_SCHEMA
            self._strategy_spec_identity = incoming_strategy_identity
            self._evidence_contracts = incoming_evidence_contracts
            self._last_cycle_id = cycle_id
            self._cycle_ids.add(cycle_id)
            self._cycle_ids_ordered = (*self._cycle_ids_ordered, cycle_id)
            self._cycle_commits = (*self._cycle_commits, cycle_commit)
            self._prefix_lengths = (*self._prefix_lengths, next_length)
            self._prefix_sha256s = (*self._prefix_sha256s, next_digest.hexdigest())
            self._tail_digest = next_digest
            self._lifecycle_chain = next_lifecycle_chain
            self._file_state = _journal_file_state(self._path)
        return True
