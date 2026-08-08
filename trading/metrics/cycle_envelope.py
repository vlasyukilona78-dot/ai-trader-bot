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
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from core.mexc_strategy_spec import (
    MEXC_STRATEGY_SPEC_VERSION,
    MexcStrategySpec,
    MexcStrategySpecError,
    strategy_spec_contract_hash,
)
from trading.market_data.bar_contract import (
    interval_seconds,
    is_bar_aligned,
    next_bar_open_ts,
)
from trading.market_data.source_timing import (
    SourceTiming,
    SourceTimingError,
    latest_received_at,
)


CYCLE_ENVELOPE_SCHEMA_VERSION = 3

# What `entry_bar_open_ts` is allowed to claim.
#
# It proves that a research replay may compare this cohort and enter on that bar:
# every input had answered and every candidate was decided before it opened. It
# does NOT prove a live signals-only pipeline could have reached it. Between
# `cycle_completed_ts` and an actual alert there is still record construction, an
# fsync, the return path and channel delivery, none of which are measured here
# and any of which can cross the boundary. Calling this execution-ready would be
# a claim the code cannot support.
TIMING_BASIS_RESEARCH_RANKING = "research_ranking_ready"

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


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


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
    strategy_spec_version: str
    strategy_spec_contract_hash: str
    strategy_spec_instance_hash: str
    strategy_spec_payload: Mapping[str, Any]
    universe_policy_hash: str
    ranking_ready_ts: float
    cycle_completed_ts: float
    actionable_ts: float
    entry_eligible_ts: float
    entry_bar_open_ts: float
    status: str
    error_code: str | None = None
    timing_basis: str = TIMING_BASIS_RESEARCH_RANKING

    def __post_init__(self) -> None:
        if self.schema_version != CYCLE_ENVELOPE_SCHEMA_VERSION:
            raise CycleEnvelopeError("unsupported_cycle_envelope_schema_version")
        if self.timing_basis != TIMING_BASIS_RESEARCH_RANKING:
            raise CycleEnvelopeError("unsupported_timing_basis")
        if not isinstance(self.cycle_id, str) or not _HASH_RE.fullmatch(self.cycle_id):
            raise CycleEnvelopeError("cycle_id_must_be_a_sha256_digest")
        if self.strategy_spec_version != MEXC_STRATEGY_SPEC_VERSION:
            raise CycleEnvelopeError("unsupported_strategy_spec_version")
        for name in (
            "strategy_spec_contract_hash",
            "strategy_spec_instance_hash",
            "universe_policy_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
                raise CycleEnvelopeError(f"{name}_must_be_a_sha256_digest")
        if not isinstance(self.strategy_spec_payload, Mapping):
            raise CycleEnvelopeError("strategy_spec_payload_must_be_a_mapping")
        try:
            rebuilt_spec = MexcStrategySpec.from_mapping(self.strategy_spec_payload)
        except (MexcStrategySpecError, RuntimeError, TypeError, ValueError) as exc:
            raise CycleEnvelopeError("invalid_strategy_spec_payload") from exc
        canonical_spec_payload = rebuilt_spec.to_mapping()
        if _thaw_json(self.strategy_spec_payload) != canonical_spec_payload:
            raise CycleEnvelopeError("strategy_spec_payload_must_be_canonical")
        try:
            expected_contract_hash = strategy_spec_contract_hash()
            expected_instance_hash = rebuilt_spec.instance_hash
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CycleEnvelopeError("invalid_strategy_spec_identity") from exc
        if self.strategy_spec_contract_hash != expected_contract_hash:
            raise CycleEnvelopeError("strategy_spec_contract_hash_mismatch")
        if self.strategy_spec_instance_hash != expected_instance_hash:
            raise CycleEnvelopeError("strategy_spec_instance_hash_mismatch")
        if interval_seconds(self.timeframe) != rebuilt_spec.base_interval_seconds:
            raise CycleEnvelopeError("timeframe_disagrees_with_strategy_spec")
        object.__setattr__(self, "strategy_spec_payload", _freeze_json(canonical_spec_payload))
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
        strategy_spec_version: str,
        strategy_spec_contract_hash: str,
        strategy_spec_instance_hash: str,
        strategy_spec_payload: Mapping[str, Any],
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
            strategy_spec_version=strategy_spec_version,
            strategy_spec_contract_hash=strategy_spec_contract_hash,
            strategy_spec_instance_hash=strategy_spec_instance_hash,
            strategy_spec_payload=strategy_spec_payload,
            universe_policy_hash=universe_policy_hash,
            ranking_ready_ts=ranking_ready_ts,
            cycle_completed_ts=cycle_completed_ts,
            actionable_ts=actionable_ts,
            entry_eligible_ts=entry_eligible_ts,
            entry_bar_open_ts=next_bar_open_ts(entry_eligible_ts, timeframe),
            status=status,
            error_code=error_code,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CycleEnvelope":
        """Rebuild a serialized envelope without dropping provenance fields.

        This is intentionally stricter than a permissive deserializer.  A
        journal envelope is part of the evidence chain, so missing, additional,
        or silently defaulted fields are schema drift rather than compatibility.
        In particular, all cache provenance on ``SourceTiming`` must survive a
        writer/reader round trip exactly.
        """

        if not isinstance(payload, Mapping):
            raise CycleEnvelopeError("cycle_envelope_must_be_a_mapping")

        def rebuild_timing(value: object, *, field: str) -> SourceTiming:
            if not isinstance(value, Mapping):
                raise CycleEnvelopeError(f"{field}_must_be_a_mapping")
            cache_hit = value.get("cache_hit")
            if not isinstance(cache_hit, bool):
                raise CycleEnvelopeError(f"{field}_cache_hit_must_be_boolean")
            try:
                return SourceTiming(
                    source=value.get("source"),
                    request_started_at=value.get("request_started_at"),
                    received_at=value.get("received_at"),
                    status=value.get("status"),
                    source_as_of=value.get("source_as_of"),
                    error_code=value.get("error_code"),
                    cache_hit=cache_hit,
                    cache_age_sec=value.get("cache_age_sec"),
                    source_ts=value.get("source_ts"),
                )
            except (SourceTimingError, TypeError, ValueError) as exc:
                raise CycleEnvelopeError(f"invalid_{field}") from exc

        raw_timings = payload.get("source_timings")
        if not isinstance(raw_timings, (list, tuple)):
            raise CycleEnvelopeError("source_timings_must_be_a_sequence")
        timings = tuple(
            rebuild_timing(value, field=f"source_timings_{index}")
            for index, value in enumerate(raw_timings)
        )
        universe = rebuild_timing(payload.get("universe_timing"), field="universe_timing")
        if sum(timing == universe for timing in timings) != 1:
            raise CycleEnvelopeError("universe_timing_must_match_exactly_one_source_timing")

        try:
            rebuilt = cls.build(
                cycle_id=payload.get("cycle_id"),
                timeframe=payload.get("timeframe"),
                cycle_started_at=payload.get("cycle_started_at"),
                candle_cutoff_ts=payload.get("candle_cutoff_ts"),
                universe_symbols=payload.get("universe_symbols"),
                universe_timing=universe,
                source_timings=timings,
                strategy_spec_version=payload.get("strategy_spec_version"),
                strategy_spec_contract_hash=payload.get("strategy_spec_contract_hash"),
                strategy_spec_instance_hash=payload.get("strategy_spec_instance_hash"),
                strategy_spec_payload=payload.get("strategy_spec_payload"),
                universe_policy_hash=payload.get("universe_policy_hash"),
                ranking_ready_ts=payload.get("ranking_ready_ts"),
                cycle_completed_ts=payload.get("cycle_completed_ts"),
                status=payload.get("status"),
                error_code=payload.get("error_code"),
            )
        except (CycleEnvelopeError, TypeError, ValueError) as exc:
            raise CycleEnvelopeError("invalid_cycle_envelope") from exc

        # This comparison catches omitted cache fields, unknown fields, a stale
        # schema/timing basis, and callers that attempted to coerce JSON types.
        if rebuilt.as_dict() != dict(payload):
            raise CycleEnvelopeError("cycle_envelope_source_mismatch")
        return rebuilt

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
            "strategy_spec_version": self.strategy_spec_version,
            "strategy_spec_contract_hash": self.strategy_spec_contract_hash,
            "strategy_spec_instance_hash": self.strategy_spec_instance_hash,
            "strategy_spec_payload": _thaw_json(self.strategy_spec_payload),
            "universe_policy_hash": self.universe_policy_hash,
            "ranking_ready_ts": self.ranking_ready_ts,
            "cycle_completed_ts": self.cycle_completed_ts,
            "actionable_ts": self.actionable_ts,
            "entry_eligible_ts": self.entry_eligible_ts,
            "entry_bar_open_ts": self.entry_bar_open_ts,
            "status": self.status,
            "error_code": self.error_code,
            "timing_basis": self.timing_basis,
        }

    @property
    def strategy_config_hash(self) -> str:
        """Compatibility alias for schema-v4 row provenance.

        Journal-v5 headers carry the complete StrategySpec identity and payload.
        Decision rows retain their existing field name until the feature-row
        contract is versioned independently; its value is now the validated
        StrategySpec instance hash, never an independently computed digest.
        """

        return self.strategy_spec_instance_hash

    def envelope_hash(self) -> str:
        """Identity of this cycle's executable timing, kept apart from the market.

        Rows carry it so a snapshot can be tied to the timing it was ranked under
        without that timing entering the market-feature hash.
        """

        encoded = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
