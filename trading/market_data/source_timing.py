"""When each market-data source was actually asked for and actually answered.

A candle cutoff says which bars are closed. It does not say when the process
learned about them. Those are different instants, and only the second one bounds
what a decision could have acted on: a ticker fetched at 12:00:03 cannot be
treated as knowledge held at 12:00:00 merely because the bar it describes closed
then.

Every source therefore records its own request and response instants alongside
the causal timestamp of the data it returned. `source_as_of` is the data's own
cutoff, `received_at` is when it arrived, and no consumer may substitute one for
the other.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


class SourceTimingError(ValueError):
    """Raised when a source's timing cannot support a causal decision."""


_STATUS_OK = "ok"
_SOURCE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_STATUS_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ERROR_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,127}$")


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceTimingError(f"{field}_must_be_a_finite_number")
    number = float(value)
    if not math.isfinite(number):
        raise SourceTimingError(f"{field}_must_be_a_finite_number")
    return number


@dataclass(frozen=True)
class SourceTiming:
    """Request/response instants for one point-in-time data source."""

    source: str
    request_started_at: float
    received_at: float
    status: str = _STATUS_OK
    source_as_of: float | None = None
    error_code: str | None = None
    # A cache hit is not a fresh answer. `received_at` is when this process got
    # the rows; `source_ts` is when the exchange produced them, which for a hit is
    # the earlier original response.
    cache_hit: bool = False
    cache_age_sec: float | None = None
    source_ts: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not _SOURCE_RE.fullmatch(self.source):
            raise SourceTimingError("source_must_be_a_short_lowercase_identifier")
        if not isinstance(self.status, str) or not _STATUS_RE.fullmatch(self.status):
            raise SourceTimingError("status_must_be_a_short_lowercase_identifier")

        started = _finite(self.request_started_at, field="request_started_at")
        received = _finite(self.received_at, field="received_at")
        if received < started:
            raise SourceTimingError("received_at_precedes_request_started_at")
        object.__setattr__(self, "request_started_at", started)
        object.__setattr__(self, "received_at", received)

        if self.source_as_of is not None:
            as_of = _finite(self.source_as_of, field="source_as_of")
            # A failed request may still return stale fallback data. Neither a
            # success nor a fallback can describe a causal instant the process
            # had not reached when the request completed. A failure with no data
            # should use source_as_of=None rather than a requested future cutoff.
            if as_of > received:
                raise SourceTimingError("source_as_of_follows_received_at")
            object.__setattr__(self, "source_as_of", as_of)

        if self.cache_age_sec is not None:
            age = _finite(self.cache_age_sec, field="cache_age_sec")
            if age < 0:
                raise SourceTimingError("cache_age_sec_must_not_be_negative")
            object.__setattr__(self, "cache_age_sec", age)
        if self.source_ts is not None:
            produced = _finite(self.source_ts, field="source_ts")
            if produced > received:
                raise SourceTimingError("source_ts_follows_received_at")
            object.__setattr__(self, "source_ts", produced)
        if self.cache_hit:
            if self.source_ts is None:
                raise SourceTimingError("cache_hit_requires_source_ts")
            if self.cache_age_sec is None:
                raise SourceTimingError("cache_hit_requires_cache_age_sec")
            # A caller may measure age immediately before sending a request or
            # immediately after receiving the local cache result.  Both are
            # honest; an age outside that interval is not.  This catches a
            # common provenance bug where the cache flag is set but age zero is
            # attached to data produced minutes earlier.
            earliest_age = max(0.0, started - self.source_ts)
            latest_age = max(0.0, received - self.source_ts)
            tolerance = 1e-6
            if not (
                earliest_age - tolerance
                <= self.cache_age_sec
                <= latest_age + tolerance
            ):
                raise SourceTimingError("cache_age_sec_is_incoherent_with_source_ts")
        elif self.cache_age_sec not in (None, 0.0):
            raise SourceTimingError("non_cache_source_must_not_have_positive_cache_age")

        if self.status == _STATUS_OK:
            if self.error_code is not None:
                raise SourceTimingError("ok_status_must_not_carry_error_code")
        elif self.error_code is None:
            raise SourceTimingError("non_ok_status_requires_error_code")
        elif not _ERROR_CODE_RE.fullmatch(self.error_code):
            raise SourceTimingError("error_code_must_be_a_safe_exception_class_name")

    @property
    def ok(self) -> bool:
        return self.status == _STATUS_OK

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "request_started_at": self.request_started_at,
            "received_at": self.received_at,
            "status": self.status,
            "source_as_of": self.source_as_of,
            "error_code": self.error_code,
            "cache_hit": bool(self.cache_hit),
            "cache_age_sec": self.cache_age_sec,
            "source_ts": self.source_ts,
        }


def latest_received_at(timings: object) -> float:
    """The instant every supplied source had answered.

    Failed sources still count: the cycle waited for them, and a decision taken
    after a timeout is not a decision that existed before it.
    """

    values = [timing.received_at for timing in timings]
    if not values:
        raise SourceTimingError("at_least_one_source_timing_is_required")
    return max(values)
