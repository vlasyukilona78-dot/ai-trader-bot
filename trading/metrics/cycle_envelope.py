"""The temporal envelope of one point-in-time scan cycle.

A cycle is the unit that can be ranked. Until every symbol in it has been
evaluated there is no cohort to compare, and until every source has answered
there is nothing to compare it on. This records both boundaries explicitly so a
later dataset can prove that an entry was reachable rather than assume it.

The derived instants are computed here, never supplied by a caller, so a cycle
cannot claim to have been actionable earlier than its own inputs allow.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

from trading.market_data.bar_contract import (
    interval_seconds,
    is_bar_aligned,
    next_bar_open_ts,
)
from trading.market_data.source_timing import SourceTiming, latest_received_at


CYCLE_ENVELOPE_SCHEMA_VERSION = 1

CYCLE_STATUSES = frozenset({"completed", "empty_universe", "error"})

# Matches the cycle-ID bound. The whole MEXC USDT board is ~1000 contracts, so
# this is a sanity limit rather than an operating constraint.
_MAX_UNIVERSE_SYMBOLS = 10_000

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ERROR_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,127}$")


class CycleEnvelopeError(ValueError):
    """Raised when a cycle cannot describe a causally reachable entry."""


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CycleEnvelopeError(f"{field}_must_be_a_finite_number")
    number = float(value)
    if not math.isfinite(number):
        raise CycleEnvelopeError(f"{field}_must_be_a_finite_number")
    return number


@dataclass(frozen=True)
class CycleEnvelope:
    """One scan cycle's identity, provenance and executable timing."""

    schema_version: int
    cycle_id: str
    timeframe: str
    cycle_started_at: float
    candle_cutoff_ts: float
    universe_symbols: tuple[str, ...]
    universe_timing: SourceTiming
    source_timings: tuple[SourceTiming, ...]
    strategy_config_hash: str
    universe_policy_hash: str
    ranking_ready_ts: float
    cycle_completed_ts: float
    actionable_ts: float
    entry_eligible_ts: float
    entry_bar_open_ts: float
    status: str
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != CYCLE_ENVELOPE_SCHEMA_VERSION:
            raise CycleEnvelopeError("unsupported_cycle_envelope_schema_version")
        if not isinstance(self.cycle_id, str) or not _HASH_RE.fullmatch(self.cycle_id):
            raise CycleEnvelopeError("cycle_id_must_be_a_sha256_digest")
        for name in ("strategy_config_hash", "universe_policy_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
                raise CycleEnvelopeError(f"{name}_must_be_a_sha256_digest")
        if self.status not in CYCLE_STATUSES:
            raise CycleEnvelopeError("unsupported_cycle_status")
        if self.status == "error":
            if self.error_code is None or not _ERROR_CODE_RE.fullmatch(self.error_code):
                raise CycleEnvelopeError("error_status_requires_a_safe_error_code")
        elif self.error_code is not None:
            raise CycleEnvelopeError("non_error_status_must_not_carry_error_code")

        seconds = interval_seconds(self.timeframe)

        for name in (
            "cycle_started_at",
            "candle_cutoff_ts",
            "ranking_ready_ts",
            "cycle_completed_ts",
            "actionable_ts",
            "entry_eligible_ts",
            "entry_bar_open_ts",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), field=name))

        if not is_bar_aligned(self.candle_cutoff_ts, self.timeframe):
            raise CycleEnvelopeError("candle_cutoff_ts_must_sit_on_a_bar_boundary")
        if self.candle_cutoff_ts > self.cycle_started_at:
            raise CycleEnvelopeError("candle_cutoff_ts_follows_cycle_start")

        if not self.source_timings:
            raise CycleEnvelopeError("at_least_one_source_timing_is_required")
        if self.universe_timing not in self.source_timings:
            raise CycleEnvelopeError("universe_timing_must_be_among_the_source_timings")

        latest_source = latest_received_at(self.source_timings)
        if self.ranking_ready_ts < latest_source:
            raise CycleEnvelopeError("ranking_ready_ts_precedes_a_source_response")
        if self.cycle_completed_ts < self.ranking_ready_ts:
            raise CycleEnvelopeError("cycle_completed_ts_precedes_ranking_ready_ts")

        expected_actionable = max(latest_source, self.ranking_ready_ts)
        if self.actionable_ts != expected_actionable:
            raise CycleEnvelopeError("actionable_ts_is_not_the_latest_required_instant")
        expected_eligible = max(self.actionable_ts, self.cycle_completed_ts)
        if self.entry_eligible_ts != expected_eligible:
            raise CycleEnvelopeError("entry_eligible_ts_is_not_the_latest_required_instant")

        if self.entry_bar_open_ts <= self.actionable_ts:
            raise CycleEnvelopeError("entry_bar_open_ts_must_follow_actionable_ts")
        if self.entry_bar_open_ts <= self.entry_eligible_ts:
            raise CycleEnvelopeError("entry_bar_open_ts_must_follow_entry_eligible_ts")
        if not is_bar_aligned(self.entry_bar_open_ts, self.timeframe):
            raise CycleEnvelopeError("entry_bar_open_ts_must_sit_on_a_bar_boundary")
        if self.entry_bar_open_ts - self.entry_eligible_ts > float(seconds):
            raise CycleEnvelopeError("entry_bar_open_ts_skips_a_reachable_bar")

        symbols = tuple(self.universe_symbols)
        if self.status == "empty_universe":
            if symbols:
                raise CycleEnvelopeError("empty_universe_must_not_list_symbols")
        elif self.status == "error":
            # A failure before evaluation may legitimately have no universe yet.
            if len(set(symbols)) != len(symbols):
                raise CycleEnvelopeError("universe_symbols_must_be_unique")
        else:
            if not symbols:
                raise CycleEnvelopeError("cycle_requires_at_least_one_symbol")
            if len(set(symbols)) != len(symbols):
                raise CycleEnvelopeError("universe_symbols_must_be_unique")
        # The envelope is written once per cycle, so a large universe is no longer
        # a per-row cost. It is still bounded, just far above any real scan.
        if len(symbols) > _MAX_UNIVERSE_SYMBOLS:
            raise CycleEnvelopeError("universe_contains_too_many_symbols")
        object.__setattr__(self, "universe_symbols", symbols)
        object.__setattr__(self, "source_timings", tuple(self.source_timings))

    @classmethod
    def build(
        cls,
        *,
        cycle_id: str,
        timeframe: str,
        cycle_started_at: float,
        candle_cutoff_ts: float,
        universe_symbols: Sequence[str],
        universe_timing: SourceTiming,
        source_timings: Sequence[SourceTiming],
        strategy_config_hash: str,
        universe_policy_hash: str,
        ranking_ready_ts: float,
        cycle_completed_ts: float,
        status: str = "completed",
        error_code: str | None = None,
    ) -> "CycleEnvelope":
        """Derive the executable instants instead of trusting a caller's arithmetic."""

        timings = tuple(source_timings)
        latest_source = latest_received_at(timings)
        actionable_ts = max(latest_source, _finite(ranking_ready_ts, field="ranking_ready_ts"))
        entry_eligible_ts = max(
            actionable_ts, _finite(cycle_completed_ts, field="cycle_completed_ts")
        )
        return cls(
            schema_version=CYCLE_ENVELOPE_SCHEMA_VERSION,
            cycle_id=cycle_id,
            timeframe=timeframe,
            cycle_started_at=cycle_started_at,
            candle_cutoff_ts=candle_cutoff_ts,
            universe_symbols=tuple(universe_symbols),
            universe_timing=universe_timing,
            source_timings=timings,
            strategy_config_hash=strategy_config_hash,
            universe_policy_hash=universe_policy_hash,
            ranking_ready_ts=ranking_ready_ts,
            cycle_completed_ts=cycle_completed_ts,
            actionable_ts=actionable_ts,
            entry_eligible_ts=entry_eligible_ts,
            entry_bar_open_ts=next_bar_open_ts(entry_eligible_ts, timeframe),
            status=status,
            error_code=error_code,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": int(self.schema_version),
            "cycle_id": self.cycle_id,
            "timeframe": self.timeframe,
            "timeframe_seconds": interval_seconds(self.timeframe),
            "cycle_started_at": self.cycle_started_at,
            "candle_cutoff_ts": self.candle_cutoff_ts,
            "universe_symbols": list(self.universe_symbols),
            "universe_timing": self.universe_timing.as_dict(),
            "source_timings": [timing.as_dict() for timing in self.source_timings],
            "strategy_config_hash": self.strategy_config_hash,
            "universe_policy_hash": self.universe_policy_hash,
            "ranking_ready_ts": self.ranking_ready_ts,
            "cycle_completed_ts": self.cycle_completed_ts,
            "actionable_ts": self.actionable_ts,
            "entry_eligible_ts": self.entry_eligible_ts,
            "entry_bar_open_ts": self.entry_bar_open_ts,
            "status": self.status,
            "error_code": self.error_code,
        }
