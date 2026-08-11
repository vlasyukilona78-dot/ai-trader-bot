"""Strict evidence for one closed market-data frame read.

The candle boundary a caller requested is not necessarily the last candle an
exchange returned.  This module keeps those facts separate and commits to the
exact post-filter frame consumed by the strategy without letting request
latency, cache age, or worker scheduling change the market identity.

The hash is an evidence commitment, not a raw-data archive.  Replaying the
frame still requires the corresponding OHLCV rows to be retained elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from trading.market_data.bar_contract import closed_boundary_ts, interval_seconds
from trading.market_data.source_timing import SourceTiming, SourceTimingError


FRAME_PROVENANCE_CONTRACT_VERSION = "mexc_closed_frame_provenance_v1"
FRAME_HASH_CONTRACT_VERSION = "mexc_closed_ohlcv_hash_v1"
RAW_BUNDLE_HASH_CONTRACT_VERSION = "mexc_raw_frame_bundle_hash_v1"

_PINNED_CONTRACT_HASH = (
    "f4004ac933cc1725b2560e93ffbe278c826910424e350059ad29420ed3665dbf"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9_]{0,63}$")
_ERROR_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,127}$")
_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
_CANONICAL_TIMEFRAME_BY_SECONDS = {
    60: "Min1",
    5 * 60: "Min5",
    15 * 60: "Min15",
    30 * 60: "Min30",
    60 * 60: "Min60",
    4 * 60 * 60: "Hour4",
    8 * 60 * 60: "Hour8",
    24 * 60 * 60: "Day1",
    7 * 24 * 60 * 60: "Week1",
}
_OUTCOMES = frozenset(
    {"fresh", "stale", "no_rows", "request_failed", "not_requested"}
)
_EXACT_EVIDENCE_KEYS = frozenset(
    {
        "contract_version",
        "source",
        "venue",
        "symbol",
        "venue_symbol",
        "timeframe",
        "timeframe_seconds",
        "requested_as_of_ts",
        "expected_closed_boundary_ts",
        "request_started_at",
        "received_at",
        "source_ts",
        "cache_hit",
        "cache_age_sec",
        "outcome",
        "error_code",
        "missing_reason",
        "first_bar_open_ts",
        "last_bar_open_ts",
        "last_bar_close_ts",
        "data_through_ts",
        "bar_count",
        "frame_hash_contract_version",
        "frame_hash",
    }
)


class FrameProvenanceError(ValueError):
    """Raised when source evidence cannot describe the frame honestly."""


class FrameQualityError(FrameProvenanceError):
    """Raised when fetched rows violate the executable candle contract."""


def _finite(value: object, *, field: str) -> float:
    # Pandas exposes numeric cells as numpy scalar types.  ``numbers.Real``
    # accepts those without also accepting strings or arbitrary float-like
    # objects; bool remains explicitly forbidden despite being an int subclass.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FrameProvenanceError(f"{field}_must_be_a_finite_number")
    number = float(value)
    if not math.isfinite(number):
        raise FrameProvenanceError(f"{field}_must_be_a_finite_number")
    # Canonical JSON should not distinguish negative zero from zero.
    return 0.0 if number == 0.0 else number


def _optional_finite(value: object, *, field: str) -> float | None:
    return None if value is None else _finite(value, field=field)


def _safe_string(value: object, *, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise FrameProvenanceError(f"{field}_is_invalid")
    return value


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FrameProvenanceError("payload_is_not_canonical_json") from exc


def canonical_frame_timeframe(value: object) -> str:
    """Normalize numeric/MEXC aliases before persistence and hashing."""

    if not isinstance(value, str) or not value.strip():
        raise FrameProvenanceError("timeframe_is_invalid")
    try:
        seconds = interval_seconds(value)
    except (TypeError, ValueError) as exc:
        raise FrameProvenanceError("timeframe_is_invalid") from exc
    canonical = _CANONICAL_TIMEFRAME_BY_SECONDS.get(seconds)
    if canonical is None:
        raise FrameProvenanceError("timeframe_is_invalid")
    return canonical


_CONTRACT_SCHEMA = {
    "contract_version": FRAME_PROVENANCE_CONTRACT_VERSION,
    "frame_hash_contract_version": FRAME_HASH_CONTRACT_VERSION,
    "raw_bundle_hash_contract_version": RAW_BUNDLE_HASH_CONTRACT_VERSION,
    "evidence_keys": sorted(_EXACT_EVIDENCE_KEYS),
    "outcomes": sorted(_OUTCOMES),
    "frame_columns": [*_REQUIRED_COLUMNS, "turnover", "turnover_observed"],
    "frame_index": "utc_epoch_nanoseconds_of_bar_open",
    "market_identity_excludes": [
        "request_started_at",
        "received_at",
        "source_ts",
        "cache_hit",
        "cache_age_sec",
    ],
    "outcome_invariants": {
        "fresh": "nonempty_hashed_frame_data_through_expected_boundary_no_failure",
        "stale": "nonempty_hashed_frame_with_data_lag_or_refresh_failure",
        "no_rows": "attempted_successful_read_with_empty_frame",
        "request_failed": "attempted_read_without_frame_and_safe_error_code",
        "not_requested": "no_read_timing_no_frame_and_explicit_reason",
    },
    "time_semantics": {
        "requested_as_of_ts": "caller_decision_cutoff",
        "expected_closed_boundary_ts": "closed_boundary_of_requested_as_of",
        "data_through_ts": "actual_last_bar_close_never_requested_boundary_substitute",
    },
    "timeframe_semantics": "canonical_mexc_name_by_fixed_interval_seconds",
    "frame_quality_invariants": [
        "utc_aligned_contiguous_unique_bar_opens",
        "finite_ohlcv_with_high_low_geometry",
        "nonnegative_volume_and_observed_turnover",
        "only_absent_or_explicit_null_turnover_is_unobserved",
        "last_bar_close_not_after_cutoff",
    ],
    "read_timing_invariants": {
        "direct": "requested_as_of_lte_request_start_lte_source_ts_lte_received",
        "cache": "source_ts_lte_request_start_lte_received_with_coherent_age",
    },
    "market_identity_available_frame": (
        "frame_hash_and_actual_bar_coverage_excluding_cache_refresh_outcome"
    ),
    "source_timing_projection": {
        "not_requested": "omitted_from_cycle_source_timings",
        "request_started_at": "minimum_attempt_start",
        "received_at": "maximum_attempt_receipt",
        "source_as_of": "minimum_available_data_through",
        "source_ts": "minimum_available_source_timestamp",
        "cache_hit": "true_if_any_attempt_used_cache",
        "cache_age_sec": "maximum_coherent_age_at_aggregate_receipt",
        "status": (
            "ok_if_all_fresh_else_stale_cache_if_every_attempt_has_current_data_"
            "else_error"
        ),
        "error_code": "deterministic_safe_code_for_aggregate_outcome",
    },
}


def _computed_contract_hash() -> str:
    return hashlib.sha256(_canonical_bytes(_CONTRACT_SCHEMA)).hexdigest()


def frame_provenance_contract_hash() -> str:
    """Return the pinned executable schema hash, failing on an in-place edit."""

    digest = _computed_contract_hash()
    if digest != _PINNED_CONTRACT_HASH:
        raise RuntimeError("frame_provenance_contract_changed_without_version_bump")
    return digest


def _frame_finite(value: object, *, field: str) -> float:
    try:
        return _finite(value, field=field)
    except FrameProvenanceError as exc:
        raise FrameQualityError(str(exc)) from exc


def _frame_rows(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    cutoff_ts: float | None = None,
) -> list[dict[str, object]]:
    if not isinstance(frame, pd.DataFrame):
        raise FrameQualityError("frame_must_be_a_pandas_dataframe")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise FrameQualityError("frame_index_must_be_a_datetime_index")
    if frame.index.tz is None:
        raise FrameQualityError("frame_index_must_be_timezone_aware")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise FrameQualityError("frame_index_must_be_ordered_and_unique")
    if frame.index.hasnans:
        raise FrameQualityError("frame_index_must_not_contain_nat")
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FrameQualityError("frame_is_missing_required_ohlcv_columns")

    timeframe = canonical_frame_timeframe(timeframe)
    seconds = interval_seconds(timeframe)
    step_ns = seconds * 1_000_000_000
    utc_index = frame.index.tz_convert("UTC")
    open_ns_values = utc_index.asi8
    if any(int(open_ns) % step_ns != 0 for open_ns in open_ns_values):
        raise FrameQualityError("frame_bar_open_is_not_timeframe_aligned")
    if len(open_ns_values) > 1 and any(
        int(current) - int(previous) != step_ns
        for previous, current in zip(open_ns_values[:-1], open_ns_values[1:])
    ):
        raise FrameQualityError("frame_contains_missing_or_irregular_bars")
    if cutoff_ts is not None:
        cutoff = _frame_finite(cutoff_ts, field="cutoff_ts")
        expected_cutoff = float(closed_boundary_ts(cutoff, timeframe))
        if not math.isclose(cutoff, expected_cutoff, rel_tol=0.0, abs_tol=1e-6):
            raise FrameQualityError("frame_cutoff_is_not_timeframe_aligned")
        if len(open_ns_values) > 0:
            last_close_ts = float(utc_index[-1].timestamp()) + float(seconds)
            if last_close_ts > cutoff + 1e-6:
                raise FrameQualityError("frame_contains_bar_after_cutoff")

    normalized = frame.loc[:, list(_REQUIRED_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    if normalized.isna().any().any():
        raise FrameQualityError("required_ohlcv_contains_non_numeric_values")

    turnover_present = "turnover" in frame.columns
    rows: list[dict[str, object]] = []
    for position, open_ns in enumerate(open_ns_values):
        row: dict[str, object] = {"bar_open_ns": int(open_ns)}
        for column in _REQUIRED_COLUMNS:
            row[column] = _frame_finite(
                normalized.iloc[position][column], field=f"frame_{column}"
            )
        open_value = float(row["open"])
        high_value = float(row["high"])
        low_value = float(row["low"])
        close_value = float(row["close"])
        volume_value = float(row["volume"])
        if high_value < max(open_value, close_value) or low_value > min(
            open_value, close_value
        ) or low_value > high_value:
            raise FrameQualityError("frame_ohlc_geometry_is_incoherent")
        if volume_value < 0.0:
            raise FrameQualityError("frame_volume_must_not_be_negative")

        raw_turnover = frame["turnover"].iloc[position] if turnover_present else None
        try:
            turnover_missing = raw_turnover is None or bool(pd.isna(raw_turnover))
        except (TypeError, ValueError) as exc:
            raise FrameQualityError("frame_turnover_must_be_scalar") from exc
        turnover_observed = bool(turnover_present and not turnover_missing)
        turnover_value: float | None = None
        if turnover_observed:
            try:
                parsed_turnover = pd.to_numeric(raw_turnover, errors="raise")
            except (TypeError, ValueError) as exc:
                raise FrameQualityError(
                    "frame_turnover_contains_non_numeric_value"
                ) from exc
            turnover_value = _frame_finite(
                parsed_turnover, field="frame_turnover"
            )
            if turnover_value < 0.0:
                raise FrameQualityError("frame_turnover_must_not_be_negative")
        row["turnover_observed"] = turnover_observed
        row["turnover"] = turnover_value
        rows.append(row)
    return rows


def canonical_closed_frame_hash(
    frame: pd.DataFrame,
    *,
    venue: str,
    symbol: str,
    venue_symbol: str,
    timeframe: str,
    cutoff_ts: float,
) -> str:
    """Hash the exact post-filter frame, independent of read latency/cache use."""

    venue = _safe_string(venue, field="venue", pattern=_IDENTIFIER_RE)
    symbol = _safe_string(symbol, field="symbol", pattern=_SYMBOL_RE)
    venue_symbol = _safe_string(
        venue_symbol, field="venue_symbol", pattern=_SYMBOL_RE
    )
    timeframe = canonical_frame_timeframe(timeframe)
    seconds = interval_seconds(timeframe)
    cutoff = _finite(cutoff_ts, field="cutoff_ts")
    payload = {
        "contract_version": FRAME_HASH_CONTRACT_VERSION,
        "venue": venue,
        "symbol": symbol,
        "venue_symbol": venue_symbol,
        "timeframe": timeframe,
        "timeframe_seconds": seconds,
        "cutoff_ts": cutoff,
        "rows": _frame_rows(frame, timeframe=timeframe, cutoff_ts=cutoff),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _bar_range(
    frame: pd.DataFrame, *, timeframe: str
) -> tuple[float | None, float | None, float | None]:
    if frame.empty:
        return None, None, None
    seconds = interval_seconds(timeframe)
    utc = frame.index.tz_convert("UTC")
    first_open = float(utc[0].timestamp())
    last_open = float(utc[-1].timestamp())
    return first_open, last_open, last_open + float(seconds)


@dataclass(frozen=True)
class SourceReadEvidenceV1:
    """Immutable, exactly serializable evidence for one source read."""

    source: str
    venue: str
    symbol: str
    venue_symbol: str
    timeframe: str
    requested_as_of_ts: float
    expected_closed_boundary_ts: float
    request_started_at: float | None
    received_at: float | None
    source_ts: float | None
    cache_hit: bool
    cache_age_sec: float | None
    outcome: str
    error_code: str | None
    missing_reason: str | None
    first_bar_open_ts: float | None
    last_bar_open_ts: float | None
    last_bar_close_ts: float | None
    data_through_ts: float | None
    bar_count: int
    frame_hash: str | None
    contract_version: str = FRAME_PROVENANCE_CONTRACT_VERSION
    frame_hash_contract_version: str = FRAME_HASH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != FRAME_PROVENANCE_CONTRACT_VERSION:
            raise FrameProvenanceError("unsupported_frame_provenance_contract_version")
        if self.frame_hash_contract_version != FRAME_HASH_CONTRACT_VERSION:
            raise FrameProvenanceError("unsupported_frame_hash_contract_version")
        _safe_string(self.source, field="source", pattern=_IDENTIFIER_RE)
        _safe_string(self.venue, field="venue", pattern=_IDENTIFIER_RE)
        _safe_string(self.symbol, field="symbol", pattern=_SYMBOL_RE)
        _safe_string(self.venue_symbol, field="venue_symbol", pattern=_SYMBOL_RE)
        canonical_timeframe = canonical_frame_timeframe(self.timeframe)
        object.__setattr__(self, "timeframe", canonical_timeframe)
        seconds = interval_seconds(canonical_timeframe)
        requested = _finite(self.requested_as_of_ts, field="requested_as_of_ts")
        expected = _finite(
            self.expected_closed_boundary_ts,
            field="expected_closed_boundary_ts",
        )
        if expected != float(closed_boundary_ts(requested, self.timeframe)):
            raise FrameProvenanceError("expected_boundary_disagrees_with_requested_as_of")
        object.__setattr__(self, "requested_as_of_ts", requested)
        object.__setattr__(self, "expected_closed_boundary_ts", expected)

        if not isinstance(self.cache_hit, bool):
            raise FrameProvenanceError("cache_hit_must_be_boolean")
        if isinstance(self.bar_count, bool) or not isinstance(self.bar_count, int):
            raise FrameProvenanceError("bar_count_must_be_an_integer")
        if self.bar_count < 0:
            raise FrameProvenanceError("bar_count_must_not_be_negative")
        if self.outcome not in _OUTCOMES:
            raise FrameProvenanceError("unsupported_source_read_outcome")

        started = _optional_finite(
            self.request_started_at, field="request_started_at"
        )
        received = _optional_finite(self.received_at, field="received_at")
        source_ts = _optional_finite(self.source_ts, field="source_ts")
        age = _optional_finite(self.cache_age_sec, field="cache_age_sec")
        for name, value in (
            ("request_started_at", started),
            ("received_at", received),
            ("source_ts", source_ts),
            ("cache_age_sec", age),
        ):
            object.__setattr__(self, name, value)

        if self.outcome == "not_requested":
            if any(value is not None for value in (started, received, source_ts, age)):
                raise FrameProvenanceError("not_requested_must_not_carry_read_timing")
            if self.cache_hit:
                raise FrameProvenanceError("not_requested_must_not_be_a_cache_hit")
        else:
            if started is None or received is None:
                raise FrameProvenanceError("attempted_read_requires_request_timing")
            if started < requested:
                raise FrameProvenanceError("request_started_at_precedes_requested_as_of")
            if received < started:
                raise FrameProvenanceError("received_at_precedes_request_started_at")
            if source_ts is not None and source_ts > received:
                raise FrameProvenanceError("source_ts_follows_received_at")

        if self.cache_hit:
            if source_ts is None or age is None:
                raise FrameProvenanceError("cache_hit_requires_source_ts_and_age")
            if age < 0:
                raise FrameProvenanceError("cache_age_sec_must_not_be_negative")
            if source_ts > started:
                raise FrameProvenanceError("cache_source_ts_follows_request_start")
            earliest = max(0.0, (started or 0.0) - source_ts)
            latest = max(0.0, (received or 0.0) - source_ts)
            if not earliest - 1e-6 <= age <= latest + 1e-6:
                raise FrameProvenanceError("cache_age_sec_is_incoherent")
        elif age not in (None, 0.0):
            raise FrameProvenanceError("non_cache_read_must_not_carry_positive_age")
        elif source_ts is not None and started is not None and source_ts < started:
            raise FrameProvenanceError("direct_source_ts_precedes_request_start")

        if self.error_code is not None and (
            not isinstance(self.error_code, str)
            or not _ERROR_CODE_RE.fullmatch(self.error_code)
        ):
            raise FrameProvenanceError("error_code_is_not_safe")
        if self.missing_reason is not None and (
            not isinstance(self.missing_reason, str)
            or not _IDENTIFIER_RE.fullmatch(self.missing_reason)
        ):
            raise FrameProvenanceError("missing_reason_is_invalid")

        first_open = _optional_finite(
            self.first_bar_open_ts, field="first_bar_open_ts"
        )
        last_open = _optional_finite(
            self.last_bar_open_ts, field="last_bar_open_ts"
        )
        last_close = _optional_finite(
            self.last_bar_close_ts, field="last_bar_close_ts"
        )
        data_through = _optional_finite(
            self.data_through_ts, field="data_through_ts"
        )
        for name, value in (
            ("first_bar_open_ts", first_open),
            ("last_bar_open_ts", last_open),
            ("last_bar_close_ts", last_close),
            ("data_through_ts", data_through),
        ):
            object.__setattr__(self, name, value)

        has_frame = self.outcome in {"fresh", "stale"}
        if has_frame:
            if self.bar_count <= 0:
                raise FrameProvenanceError("available_frame_requires_rows")
            if any(
                value is None
                for value in (first_open, last_open, last_close, data_through)
            ):
                raise FrameProvenanceError("available_frame_requires_bar_range")
            if first_open > last_open:
                raise FrameProvenanceError("first_bar_follows_last_bar")
            if last_close != last_open + float(seconds):
                raise FrameProvenanceError("last_bar_duration_disagrees_with_timeframe")
            if data_through != last_close:
                raise FrameProvenanceError("data_through_must_equal_last_bar_close")
            if data_through > expected:
                raise FrameProvenanceError("frame_contains_data_after_expected_boundary")
            if not isinstance(self.frame_hash, str) or not _SHA256_RE.fullmatch(
                self.frame_hash
            ):
                raise FrameProvenanceError("available_frame_requires_sha256_hash")
        else:
            if self.bar_count != 0:
                raise FrameProvenanceError("missing_frame_must_have_zero_rows")
            if any(
                value is not None
                for value in (first_open, last_open, last_close, data_through, self.frame_hash)
            ):
                raise FrameProvenanceError("missing_frame_must_not_carry_bar_identity")

        if self.outcome == "fresh":
            if data_through != expected:
                raise FrameProvenanceError("fresh_frame_must_reach_expected_boundary")
            if self.error_code is not None or self.missing_reason is not None:
                raise FrameProvenanceError("fresh_frame_must_not_carry_failure_metadata")
        elif self.outcome == "stale":
            if self.missing_reason is None:
                raise FrameProvenanceError("stale_frame_requires_reason")
            if data_through == expected and self.error_code is None:
                raise FrameProvenanceError("stale_current_frame_requires_refresh_error")
        elif self.outcome == "no_rows":
            if self.error_code is not None or self.missing_reason != "no_rows":
                raise FrameProvenanceError("no_rows_outcome_has_invalid_failure_metadata")
        elif self.outcome == "request_failed":
            if self.error_code is None or self.missing_reason != "request_failed":
                raise FrameProvenanceError("request_failed_requires_safe_error_code")
            if self.cache_hit:
                raise FrameProvenanceError("request_failed_must_not_be_a_cache_hit")
        elif self.outcome == "not_requested":
            if self.error_code is not None or self.missing_reason is None:
                raise FrameProvenanceError("not_requested_requires_reason_only")

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        source: str,
        venue: str,
        symbol: str,
        venue_symbol: str,
        timeframe: str,
        requested_as_of_ts: float,
        request_started_at: float,
        received_at: float,
        source_ts: float | None = None,
        cache_hit: bool = False,
        cache_age_sec: float | None = None,
        error_code: str | None = None,
        missing_reason: str | None = None,
    ) -> "SourceReadEvidenceV1":
        expected = float(closed_boundary_ts(requested_as_of_ts, timeframe))
        # Validate even a truthful empty frame: an empty response with no UTC
        # index or without the declared OHLCV schema is malformed, not no_rows.
        _frame_rows(frame, timeframe=timeframe, cutoff_ts=expected)
        first_open, last_open, last_close = _bar_range(frame, timeframe=timeframe)
        if frame.empty:
            return cls(
                source=source,
                venue=venue,
                symbol=symbol,
                venue_symbol=venue_symbol,
                timeframe=timeframe,
                requested_as_of_ts=requested_as_of_ts,
                expected_closed_boundary_ts=expected,
                request_started_at=request_started_at,
                received_at=received_at,
                source_ts=received_at if source_ts is None else source_ts,
                cache_hit=cache_hit,
                cache_age_sec=cache_age_sec,
                outcome="no_rows",
                error_code=None,
                missing_reason="no_rows",
                first_bar_open_ts=None,
                last_bar_open_ts=None,
                last_bar_close_ts=None,
                data_through_ts=None,
                bar_count=0,
                frame_hash=None,
            )
        assert last_close is not None
        lagging = not math.isclose(last_close, expected, rel_tol=0.0, abs_tol=1e-6)
        outcome = "stale" if lagging or error_code is not None else "fresh"
        reason = missing_reason or (
            "data_lag" if lagging else "refresh_failed" if error_code else None
        )
        return cls(
            source=source,
            venue=venue,
            symbol=symbol,
            venue_symbol=venue_symbol,
            timeframe=timeframe,
            requested_as_of_ts=requested_as_of_ts,
            expected_closed_boundary_ts=expected,
            request_started_at=request_started_at,
            received_at=received_at,
            source_ts=received_at if source_ts is None else source_ts,
            cache_hit=cache_hit,
            cache_age_sec=cache_age_sec,
            outcome=outcome,
            error_code=error_code,
            missing_reason=reason,
            first_bar_open_ts=first_open,
            last_bar_open_ts=last_open,
            last_bar_close_ts=last_close,
            data_through_ts=last_close,
            bar_count=len(frame),
            frame_hash=canonical_closed_frame_hash(
                frame,
                venue=venue,
                symbol=symbol,
                venue_symbol=venue_symbol,
                timeframe=timeframe,
                cutoff_ts=expected,
            ),
        )

    @classmethod
    def request_failed(
        cls,
        *,
        source: str,
        venue: str,
        symbol: str,
        venue_symbol: str,
        timeframe: str,
        requested_as_of_ts: float,
        request_started_at: float,
        received_at: float,
        error_code: str,
    ) -> "SourceReadEvidenceV1":
        return cls(
            source=source,
            venue=venue,
            symbol=symbol,
            venue_symbol=venue_symbol,
            timeframe=timeframe,
            requested_as_of_ts=requested_as_of_ts,
            expected_closed_boundary_ts=float(
                closed_boundary_ts(requested_as_of_ts, timeframe)
            ),
            request_started_at=request_started_at,
            received_at=received_at,
            source_ts=None,
            cache_hit=False,
            cache_age_sec=None,
            outcome="request_failed",
            error_code=error_code,
            missing_reason="request_failed",
            first_bar_open_ts=None,
            last_bar_open_ts=None,
            last_bar_close_ts=None,
            data_through_ts=None,
            bar_count=0,
            frame_hash=None,
        )

    @classmethod
    def not_requested(
        cls,
        *,
        source: str,
        venue: str,
        symbol: str,
        venue_symbol: str,
        timeframe: str,
        requested_as_of_ts: float,
        reason: str,
    ) -> "SourceReadEvidenceV1":
        return cls(
            source=source,
            venue=venue,
            symbol=symbol,
            venue_symbol=venue_symbol,
            timeframe=timeframe,
            requested_as_of_ts=requested_as_of_ts,
            expected_closed_boundary_ts=float(
                closed_boundary_ts(requested_as_of_ts, timeframe)
            ),
            request_started_at=None,
            received_at=None,
            source_ts=None,
            cache_hit=False,
            cache_age_sec=None,
            outcome="not_requested",
            error_code=None,
            missing_reason=reason,
            first_bar_open_ts=None,
            last_bar_open_ts=None,
            last_bar_close_ts=None,
            data_through_ts=None,
            bar_count=0,
            frame_hash=None,
        )

    def with_cache_read(
        self,
        *,
        requested_as_of_ts: float,
        request_started_at: float,
        received_at: float,
        source_ts: float,
        cache_age_sec: float,
        refresh_error_code: str | None = None,
    ) -> "SourceReadEvidenceV1":
        """Re-date a caller-owned cache result without changing its market identity."""

        outcome = self.outcome
        reason = self.missing_reason
        error_code = self.error_code
        if refresh_error_code is not None and self.bar_count > 0:
            outcome = "stale"
            reason = "refresh_failed"
            error_code = refresh_error_code
        expected = float(closed_boundary_ts(requested_as_of_ts, self.timeframe))
        if expected != self.expected_closed_boundary_ts:
            raise FrameProvenanceError("cache_read_crosses_expected_boundary")
        return replace(
            self,
            requested_as_of_ts=requested_as_of_ts,
            expected_closed_boundary_ts=expected,
            request_started_at=request_started_at,
            received_at=received_at,
            source_ts=source_ts,
            cache_hit=True,
            cache_age_sec=cache_age_sec,
            outcome=outcome,
            error_code=error_code,
            missing_reason=reason,
        )

    def market_identity_dict(self) -> dict[str, object]:
        """Deterministic source identity; operational clocks are deliberately absent."""

        has_frame = self.bar_count > 0
        data_quality = (
            "fresh"
            if has_frame and self.data_through_ts == self.expected_closed_boundary_ts
            else "stale"
            if has_frame
            else self.outcome
        )
        return {
            "contract_version": self.contract_version,
            "source": self.source,
            "venue": self.venue,
            "symbol": self.symbol,
            "venue_symbol": self.venue_symbol,
            "timeframe": self.timeframe,
            "timeframe_seconds": interval_seconds(self.timeframe),
            "requested_as_of_ts": self.requested_as_of_ts,
            "expected_closed_boundary_ts": self.expected_closed_boundary_ts,
            # A failed refresh that serves the exact same cached frame is an
            # operational fact, not a different market input. Missing inputs do
            # retain their outcome/reason because no frame hash exists for them.
            "outcome": data_quality,
            "error_code": None if has_frame else self.error_code,
            "missing_reason": (
                None
                if has_frame and data_quality == "fresh"
                else "data_lag"
                if has_frame
                else self.missing_reason
            ),
            "first_bar_open_ts": self.first_bar_open_ts,
            "last_bar_open_ts": self.last_bar_open_ts,
            "last_bar_close_ts": self.last_bar_close_ts,
            "data_through_ts": self.data_through_ts,
            "bar_count": self.bar_count,
            "frame_hash_contract_version": self.frame_hash_contract_version,
            "frame_hash": self.frame_hash,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "source": self.source,
            "venue": self.venue,
            "symbol": self.symbol,
            "venue_symbol": self.venue_symbol,
            "timeframe": self.timeframe,
            "timeframe_seconds": interval_seconds(self.timeframe),
            "requested_as_of_ts": self.requested_as_of_ts,
            "expected_closed_boundary_ts": self.expected_closed_boundary_ts,
            "request_started_at": self.request_started_at,
            "received_at": self.received_at,
            "source_ts": self.source_ts,
            "cache_hit": self.cache_hit,
            "cache_age_sec": self.cache_age_sec,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "missing_reason": self.missing_reason,
            "first_bar_open_ts": self.first_bar_open_ts,
            "last_bar_open_ts": self.last_bar_open_ts,
            "last_bar_close_ts": self.last_bar_close_ts,
            "data_through_ts": self.data_through_ts,
            "bar_count": self.bar_count,
            "frame_hash_contract_version": self.frame_hash_contract_version,
            "frame_hash": self.frame_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SourceReadEvidenceV1":
        if not isinstance(payload, Mapping):
            raise FrameProvenanceError("source_read_evidence_must_be_a_mapping")
        if set(payload) != _EXACT_EVIDENCE_KEYS:
            raise FrameProvenanceError("source_read_evidence_schema_mismatch")
        try:
            parsed_seconds = interval_seconds(payload.get("timeframe"))
        except (TypeError, ValueError) as exc:
            raise FrameProvenanceError("timeframe_is_invalid") from exc
        if payload.get("timeframe_seconds") != parsed_seconds:
            raise FrameProvenanceError("timeframe_seconds_mismatch")
        try:
            evidence = cls(
                contract_version=payload.get("contract_version"),
                source=payload.get("source"),
                venue=payload.get("venue"),
                symbol=payload.get("symbol"),
                venue_symbol=payload.get("venue_symbol"),
                timeframe=payload.get("timeframe"),
                requested_as_of_ts=payload.get("requested_as_of_ts"),
                expected_closed_boundary_ts=payload.get(
                    "expected_closed_boundary_ts"
                ),
                request_started_at=payload.get("request_started_at"),
                received_at=payload.get("received_at"),
                source_ts=payload.get("source_ts"),
                cache_hit=payload.get("cache_hit"),
                cache_age_sec=payload.get("cache_age_sec"),
                outcome=payload.get("outcome"),
                error_code=payload.get("error_code"),
                missing_reason=payload.get("missing_reason"),
                first_bar_open_ts=payload.get("first_bar_open_ts"),
                last_bar_open_ts=payload.get("last_bar_open_ts"),
                last_bar_close_ts=payload.get("last_bar_close_ts"),
                data_through_ts=payload.get("data_through_ts"),
                bar_count=payload.get("bar_count"),
                frame_hash_contract_version=payload.get(
                    "frame_hash_contract_version"
                ),
                frame_hash=payload.get("frame_hash"),
            )
        except (FrameProvenanceError, TypeError, ValueError) as exc:
            raise FrameProvenanceError("invalid_source_read_evidence") from exc
        if _canonical_bytes(evidence.as_dict()) != _canonical_bytes(dict(payload)):
            raise FrameProvenanceError("source_read_evidence_is_not_canonical")
        return evidence


def _source_error_prefix(source: str) -> str:
    _safe_string(source, field="source_timing_source", pattern=_IDENTIFIER_RE)
    return "".join(piece.capitalize() for piece in source.split("_"))


def source_timing_from_evidence(
    evidence: SourceReadEvidenceV1,
    *,
    source: str,
) -> SourceTiming | None:
    """Project one exact read into the cycle-level timing contract.

    ``not_requested`` is absence of a read, not a zero-duration read, and is
    therefore omitted.  Every attempted outcome keeps its real request/receipt
    boundary.  Error codes are deterministic, safe identifiers; exchange or
    exception text never enters the envelope.
    """

    if not isinstance(evidence, SourceReadEvidenceV1):
        raise FrameProvenanceError("source_timing_requires_source_read_evidence")
    prefix = _source_error_prefix(source)
    if evidence.outcome == "not_requested":
        return None
    if evidence.request_started_at is None or evidence.received_at is None:
        raise FrameProvenanceError("attempted_evidence_lacks_source_timing")

    if evidence.outcome == "fresh":
        status, error_code = "ok", None
    elif (
        evidence.outcome == "stale"
        and evidence.data_through_ts == evidence.expected_closed_boundary_ts
    ):
        status = "stale_cache"
        error_code = evidence.error_code or f"{prefix}RefreshFailed"
    elif evidence.outcome == "stale":
        status = "error"
        error_code = evidence.error_code or f"{prefix}DataLag"
    elif evidence.outcome == "no_rows":
        status, error_code = "error", f"{prefix}NoRows"
    else:
        status = "error"
        error_code = evidence.error_code or f"{prefix}Unavailable"

    try:
        return SourceTiming(
            source=source,
            request_started_at=evidence.request_started_at,
            received_at=evidence.received_at,
            source_as_of=evidence.data_through_ts,
            status=status,
            error_code=error_code,
            cache_hit=evidence.cache_hit,
            cache_age_sec=evidence.cache_age_sec,
            source_ts=evidence.source_ts,
        )
    except SourceTimingError as exc:
        raise FrameProvenanceError("evidence_cannot_project_to_source_timing") from exc


def aggregate_source_timing_from_evidence(
    evidences: Sequence[SourceReadEvidenceV1],
    *,
    source: str,
) -> SourceTiming | None:
    """Conservatively summarize an ordered population of exact reads.

    Scheduling may change which worker starts or finishes first, so the
    aggregate spans the earliest request and latest receipt.  Data/source
    timestamps use the oldest available value: an aggregate must never look
    fresher than any input it represents.  Missing attempts remain visible in
    the aggregate error status rather than being silently dropped.
    """

    if isinstance(evidences, (str, bytes)):
        raise FrameProvenanceError("source_timing_aggregate_requires_sequence")
    attempted: list[SourceReadEvidenceV1] = []
    for evidence in evidences:
        if not isinstance(evidence, SourceReadEvidenceV1):
            raise FrameProvenanceError(
                "source_timing_aggregate_contains_invalid_evidence"
            )
        if evidence.outcome != "not_requested":
            attempted.append(evidence)
    if not attempted:
        return None

    timings = [
        source_timing_from_evidence(evidence, source=source)
        for evidence in attempted
    ]
    if any(timing is None for timing in timings):  # guarded by attempted
        raise FrameProvenanceError("attempted_evidence_was_not_projected")
    projected = [timing for timing in timings if timing is not None]
    started = min(timing.request_started_at for timing in projected)
    received = max(timing.received_at for timing in projected)
    current = [
        evidence.bar_count > 0
        and evidence.data_through_ts == evidence.expected_closed_boundary_ts
        for evidence in attempted
    ]
    prefix = _source_error_prefix(source)
    if all(timing.status == "ok" for timing in projected):
        status, error_code = "ok", None
    elif all(current):
        status, error_code = "stale_cache", f"{prefix}StaleCache"
    else:
        status = "error"
        error_code = (
            f"{prefix}PartialFailure" if any(current) else f"{prefix}Unavailable"
        )

    source_as_of_values = [
        float(evidence.data_through_ts)
        for evidence in attempted
        if evidence.data_through_ts is not None
    ]
    source_ts_values = [
        float(evidence.source_ts)
        for evidence in attempted
        if evidence.source_ts is not None
    ]
    source_as_of = min(source_as_of_values) if source_as_of_values else None
    source_ts = min(source_ts_values) if source_ts_values else None
    cache_hit = any(evidence.cache_hit for evidence in attempted)
    cache_age = (
        max(0.0, received - source_ts)
        if cache_hit and source_ts is not None
        else 0.0
        if source_ts is not None
        else None
    )
    try:
        return SourceTiming(
            source=source,
            request_started_at=started,
            received_at=received,
            source_as_of=source_as_of,
            status=status,
            error_code=error_code,
            cache_hit=cache_hit,
            cache_age_sec=cache_age,
            source_ts=source_ts,
        )
    except SourceTimingError as exc:
        raise FrameProvenanceError(
            "evidence_population_cannot_project_to_source_timing"
        ) from exc


def parse_source_read_evidence(
    payload: Mapping[str, object],
) -> SourceReadEvidenceV1:
    """Strict public parser for persisted source-read evidence v1."""

    return SourceReadEvidenceV1.from_dict(payload)


@dataclass(frozen=True)
class FrameRead:
    """A caller-owned frame paired with the evidence that describes it."""

    frame: pd.DataFrame | None
    evidence: SourceReadEvidenceV1

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, SourceReadEvidenceV1):
            raise FrameProvenanceError("frame_read_requires_source_evidence")
        if self.evidence.outcome in {"fresh", "stale"}:
            if not isinstance(self.frame, pd.DataFrame) or self.frame.empty:
                raise FrameProvenanceError("available_frame_read_requires_rows")
            digest = canonical_closed_frame_hash(
                self.frame,
                venue=self.evidence.venue,
                symbol=self.evidence.symbol,
                venue_symbol=self.evidence.venue_symbol,
                timeframe=self.evidence.timeframe,
                cutoff_ts=self.evidence.expected_closed_boundary_ts,
            )
            if digest != self.evidence.frame_hash:
                raise FrameProvenanceError("frame_read_hash_mismatch")
            first, last, close = _bar_range(
                self.frame, timeframe=self.evidence.timeframe
            )
            if (
                len(self.frame) != self.evidence.bar_count
                or first != self.evidence.first_bar_open_ts
                or last != self.evidence.last_bar_open_ts
                or close != self.evidence.last_bar_close_ts
            ):
                raise FrameProvenanceError("frame_read_bar_range_mismatch")
        elif self.evidence.outcome == "no_rows":
            if not isinstance(self.frame, pd.DataFrame) or not self.frame.empty:
                raise FrameProvenanceError("no_rows_read_requires_empty_frame")
            # ``no_rows`` means the source truthfully returned the canonical
            # empty OHLCV shape.  Revalidate reconstructed/public FrameRead
            # values too; otherwise valid evidence could be paired with a bare
            # RangeIndex DataFrame and launder malformed input into no_data.
            _frame_rows(
                self.frame,
                timeframe=self.evidence.timeframe,
                cutoff_ts=self.evidence.expected_closed_boundary_ts,
            )
        elif self.frame is not None:
            raise FrameProvenanceError("failed_or_skipped_read_must_not_carry_frame")


def raw_frame_bundle_hash(evidences: Sequence[SourceReadEvidenceV1]) -> str:
    """Hash a sorted set of market identities without operational latency."""

    if isinstance(evidences, (str, bytes)):
        raise FrameProvenanceError("raw_frame_bundle_requires_evidence_sequence")
    rows: list[dict[str, object]] = []
    identities: set[tuple[str, str, str, str]] = set()
    for evidence in evidences:
        if not isinstance(evidence, SourceReadEvidenceV1):
            raise FrameProvenanceError("raw_frame_bundle_contains_invalid_evidence")
        identity = (
            evidence.source,
            evidence.symbol,
            evidence.timeframe,
            str(evidence.expected_closed_boundary_ts),
        )
        if identity in identities:
            raise FrameProvenanceError("raw_frame_bundle_contains_duplicate_source")
        identities.add(identity)
        rows.append(evidence.market_identity_dict())
    rows.sort(
        key=lambda row: (
            str(row["source"]),
            str(row["symbol"]),
            str(row["timeframe"]),
            float(row["expected_closed_boundary_ts"]),
        )
    )
    payload = {
        "contract_version": RAW_BUNDLE_HASH_CONTRACT_VERSION,
        "sources": rows,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


__all__ = [
    "FRAME_HASH_CONTRACT_VERSION",
    "FRAME_PROVENANCE_CONTRACT_VERSION",
    "RAW_BUNDLE_HASH_CONTRACT_VERSION",
    "FrameProvenanceError",
    "FrameQualityError",
    "FrameRead",
    "SourceReadEvidenceV1",
    "aggregate_source_timing_from_evidence",
    "canonical_closed_frame_hash",
    "canonical_frame_timeframe",
    "frame_provenance_contract_hash",
    "parse_source_read_evidence",
    "raw_frame_bundle_hash",
    "source_timing_from_evidence",
]
