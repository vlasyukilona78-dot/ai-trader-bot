"""Immutable evidence contract for the candidate confirmation lifecycle.

This module defines the vocabulary used by the typed strategy path to preserve
an arm observation, subsequent confirmation observations, and the strategy's
proposal outcome without pretending that any of them is an executable order.

Semantic IDs bind causal market/rule inputs and bar times.  They deliberately
exclude wall-clock processing, delivery and persistence times; those belong to
the enclosing evidence record and may change with worker scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping


LIFECYCLE_CONTRACT_VERSION = "candidate_lifecycle_v1"
_PINNED_CONTRACT_HASH = "cc75c871b7097aa215f9ac88c736b6572e2443318cb0cf9f8bdaf1b0c8cc8551"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_MAX_STRING_LENGTH = 2_048
_MAX_MAPPING_ITEMS = 512
_MAX_SEQUENCE_ITEMS = 512
_MAX_DEPTH = 12
_TIME_EPSILON = 1e-6


class LifecycleContractError(ValueError):
    """Raised when lifecycle evidence is incomplete or ambiguous."""


class CandidateSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class CandidateLifecycleState(str, Enum):
    ARMED = "armed"
    # Confirmation-disabled v2 evaluates and creates/rejects the proposal on
    # the arm bar itself.  It is not a zero-latency CONFIRMED observation: no
    # confirmation observation exists at all.
    BYPASSED = "bypassed"
    SAME_BAR = "same_bar"
    WAITING = "waiting"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


class ProposalObservationStatus(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    CREATED = "created"
    REJECTED = "rejected"


class ProposalObservationBasis(str, Enum):
    NONE = "none"
    CONFIRMATION = "confirmation"
    ARM_BYPASS = "arm_bypass"


_FOLLOW_UP_STATES = frozenset(
    {
        CandidateLifecycleState.SAME_BAR,
        CandidateLifecycleState.WAITING,
        CandidateLifecycleState.CONFIRMED,
        CandidateLifecycleState.INVALIDATED,
        CandidateLifecycleState.EXPIRED,
    }
)
_TERMINAL_STATES = frozenset(
    {
        CandidateLifecycleState.BYPASSED,
        CandidateLifecycleState.CONFIRMED,
        CandidateLifecycleState.INVALIDATED,
        CandidateLifecycleState.EXPIRED,
    }
)
_ALLOWED_TRANSITIONS = {
    CandidateLifecycleState.ARMED: _FOLLOW_UP_STATES,
    CandidateLifecycleState.SAME_BAR: _FOLLOW_UP_STATES,
    CandidateLifecycleState.WAITING: _FOLLOW_UP_STATES,
}


def _require_exact_keys(
    payload: Mapping[str, Any], *, expected: frozenset[str], path: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise LifecycleContractError(f"{path}_missing_keys:{','.join(missing)}")
    if unknown:
        raise LifecycleContractError(f"{path}_unknown_keys:{','.join(unknown)}")


def _strict_string(value: object, *, field_name: str, maximum: int = _MAX_STRING_LENGTH) -> str:
    if type(value) is not str or not value:
        raise LifecycleContractError(f"{field_name}_must_be_a_non_empty_string")
    if len(value) > maximum:
        raise LifecycleContractError(f"{field_name}_is_too_long")
    return value


def _strict_code(value: object, *, field_name: str) -> str:
    code = _strict_string(value, field_name=field_name, maximum=128)
    if _CODE_RE.fullmatch(code) is None:
        raise LifecycleContractError(f"{field_name}_must_be_a_safe_code")
    return code


def _strict_sha256(value: object, *, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise LifecycleContractError(f"{field_name}_must_be_lowercase_sha256")
    return value


def _strict_int(value: object, *, field_name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise LifecycleContractError(f"{field_name}_must_be_an_integer")
    if value < minimum:
        raise LifecycleContractError(f"{field_name}_is_below_minimum")
    return value


def _strict_number(
    value: object,
    *,
    field_name: str,
    positive: bool = False,
) -> float:
    if type(value) not in (int, float):
        raise LifecycleContractError(f"{field_name}_must_be_a_number")
    result = float(value)
    if not math.isfinite(result):
        raise LifecycleContractError(f"{field_name}_must_be_finite")
    if positive and result <= 0.0:
        raise LifecycleContractError(f"{field_name}_must_be_positive")
    return 0.0 if result == 0.0 else result


def _require_bar(
    *,
    open_ts: object,
    cutoff_ts: object,
    timeframe_seconds: object,
    prefix: str,
) -> tuple[float, float, int]:
    seconds = _strict_int(
        timeframe_seconds,
        field_name=f"{prefix}_timeframe_seconds",
        minimum=1,
    )
    opened = _strict_number(open_ts, field_name=f"{prefix}_bar_open_ts")
    cutoff = _strict_number(cutoff_ts, field_name=f"{prefix}_candle_cutoff_ts")
    if not math.isclose(cutoff - opened, float(seconds), rel_tol=0.0, abs_tol=_TIME_EPSILON):
        raise LifecycleContractError(f"{prefix}_bar_duration_mismatch")
    quotient = opened / float(seconds)
    if not math.isclose(quotient, round(quotient), rel_tol=0.0, abs_tol=1e-9):
        raise LifecycleContractError(f"{prefix}_bar_open_ts_is_not_aligned")
    return opened, cutoff, seconds


def _freeze_json(value: object, *, path: str, depth: int = 0) -> object:
    if depth > _MAX_DEPTH:
        raise LifecycleContractError(f"{path}_exceeds_maximum_depth")
    if value is None or type(value) in (bool, int, str):
        if isinstance(value, str) and len(value) > _MAX_STRING_LENGTH:
            raise LifecycleContractError(f"{path}_contains_an_oversized_string")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise LifecycleContractError(f"{path}_contains_a_non_finite_number")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if len(value) > _MAX_MAPPING_ITEMS:
            raise LifecycleContractError(f"{path}_contains_too_many_keys")
        keys = list(value)
        if any(type(key) is not str or not key or len(key) > 128 for key in keys):
            raise LifecycleContractError(f"{path}_contains_an_invalid_key")
        return MappingProxyType(
            {
                key: _freeze_json(value[key], path=f"{path}.{key}", depth=depth + 1)
                for key in sorted(keys)
            }
        )
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_SEQUENCE_ITEMS:
            raise LifecycleContractError(f"{path}_contains_too_many_items")
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise LifecycleContractError(f"{path}_contains_unsupported_type:{type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LifecycleContractError("payload_is_not_canonical_json") from exc


def _raw_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _semantic_id(kind: str, payload: Mapping[str, object]) -> str:
    return _raw_sha256(
        {
            "contract_hash": lifecycle_contract_hash(),
            "contract_version": LIFECYCLE_CONTRACT_VERSION,
            "kind": kind,
            "payload": payload,
        }
    )


def lifecycle_contract_payload() -> dict[str, object]:
    """Return the declarative schema whose digest namespaces all semantic IDs."""

    return {
        "contract_version": LIFECYCLE_CONTRACT_VERSION,
        "semantic_identity": {
            "excludes": [
                "request_started_at",
                "received_at",
                "decision_completed_ts",
                "proposal_available_ts",
                "delivery_ts",
            ],
            "hash_algorithm": "sha256_canonical_json",
        },
        "enums": {
            "candidate_side": [member.value for member in CandidateSide],
            "lifecycle_state": [member.value for member in CandidateLifecycleState],
            "proposal_status": [member.value for member in ProposalObservationStatus],
            "proposal_basis": [member.value for member in ProposalObservationBasis],
        },
        "types": {
            "CandidateArmV1": [
                "strategy_spec_version",
                "strategy_spec_contract_hash",
                "strategy_spec_instance_hash",
                "raw_input_bundle_hash",
                "symbol",
                "side",
                "timeframe_seconds",
                "arm_bar_open_ts",
                "arm_candle_cutoff_ts",
                "armed_high",
                "armed_low",
                "armed_close",
                "invalidate_level",
                "confirmation_enabled",
                "confirmation_max_wait_observations",
                "arm_trace",
                "arm_trace_hash",
                "candidate_id",
            ],
            "ConfirmationObservationV1": [
                "candidate_id",
                "observation_input_bundle_hash",
                "state",
                "state_epoch",
                "timeframe_seconds",
                "observation_bar_open_ts",
                "observation_candle_cutoff_ts",
                "observed_high",
                "observed_low",
                "observed_close",
                "distinct_observation_count",
                "elapsed_bars",
                "observation_id",
            ],
            "ProposalObservationV1": [
                "candidate_id",
                "side",
                "state_epoch",
                "timeframe_seconds",
                "status",
                "basis",
                "execution_bound",
                "confirmation_observation_id",
                "proposal_input_bundle_hash",
                "reference_bar_open_ts",
                "reference_candle_cutoff_ts",
                "decision_reference_price",
                "stop_price",
                "take_profit_price",
                "rejection_reason",
                "details",
                "proposal_observation_id",
            ],
            "CandidateLifecycleEventV1": [
                "arm",
                "state",
                "state_epoch",
                "previous_event_id",
                "previous_state",
                "confirmation",
                "proposal",
                "event_id",
            ],
        },
        "invariants": [
            "candidate_identity_binds_strategy_input_symbol_side_and_arm_bar",
            "arm_snapshot_is_immutable_across_transitions",
            "observation_count_is_distinct_from_elapsed_physical_bars",
            "same_bar_repeats_the_predecessor_observation_identity_and_counts",
            "same_bar_repeats_the_predecessor_market_values",
            "non_same_bar_observations_are_strictly_monotonic",
            "distinct_observation_count_increments_once_per_new_bar",
            "state_matches_invalidation_confirmation_and_expiry_priority",
            "confirmed_requires_created_or_rejected_proposal_observation",
            "bypassed_is_an_initial_same_arm_bar_proposal_without_confirmation",
            "created_proposal_reference_price_matches_its_basis_close",
            "proposal_observation_is_never_execution_bound",
            "terminal_state_has_no_successor",
        ],
        "allowed_transitions": {
            state.value: sorted(next_state.value for next_state in next_states)
            for state, next_states in sorted(
                _ALLOWED_TRANSITIONS.items(), key=lambda item: item[0].value
            )
        },
    }


@lru_cache(maxsize=1)
def lifecycle_contract_hash() -> str:
    digest = _raw_sha256(lifecycle_contract_payload())
    if digest != _PINNED_CONTRACT_HASH:
        raise RuntimeError("lifecycle_contract_changed_without_version_bump")
    return digest


@dataclass(frozen=True, slots=True)
class CandidateArmV1:
    strategy_spec_version: str
    strategy_spec_contract_hash: str
    strategy_spec_instance_hash: str
    raw_input_bundle_hash: str
    symbol: str
    side: CandidateSide
    timeframe_seconds: int
    arm_bar_open_ts: float
    arm_candle_cutoff_ts: float
    armed_high: float
    armed_low: float
    armed_close: float
    invalidate_level: float
    confirmation_enabled: bool
    confirmation_max_wait_observations: int
    arm_trace: Mapping[str, object]
    arm_trace_hash: str = field(init=False)
    candidate_id: str = field(init=False)

    def __post_init__(self) -> None:
        spec_version = _strict_string(
            self.strategy_spec_version,
            field_name="strategy_spec_version",
            maximum=128,
        )
        spec_contract = _strict_sha256(
            self.strategy_spec_contract_hash,
            field_name="strategy_spec_contract_hash",
        )
        spec_instance = _strict_sha256(
            self.strategy_spec_instance_hash,
            field_name="strategy_spec_instance_hash",
        )
        raw_hash = _strict_sha256(
            self.raw_input_bundle_hash,
            field_name="raw_input_bundle_hash",
        )
        symbol = _strict_string(self.symbol, field_name="symbol", maximum=64)
        if not isinstance(self.side, CandidateSide):
            raise LifecycleContractError("side_must_be_candidate_side")
        opened, cutoff, seconds = _require_bar(
            open_ts=self.arm_bar_open_ts,
            cutoff_ts=self.arm_candle_cutoff_ts,
            timeframe_seconds=self.timeframe_seconds,
            prefix="arm",
        )
        armed_high = _strict_number(
            self.armed_high,
            field_name="armed_high",
            positive=True,
        )
        armed_low = _strict_number(
            self.armed_low,
            field_name="armed_low",
            positive=True,
        )
        armed_close = _strict_number(
            self.armed_close,
            field_name="armed_close",
            positive=True,
        )
        if armed_low > armed_close or armed_close > armed_high:
            raise LifecycleContractError("armed_close_must_lie_between_low_and_high")
        invalidate = _strict_number(
            self.invalidate_level,
            field_name="invalidate_level",
            positive=True,
        )
        if self.side is CandidateSide.SHORT and invalidate < armed_close:
            raise LifecycleContractError("short_invalidate_level_must_not_be_below_armed_close")
        if self.side is CandidateSide.LONG and invalidate > armed_close:
            raise LifecycleContractError("long_invalidate_level_must_not_exceed_armed_close")
        if type(self.confirmation_enabled) is not bool:
            raise LifecycleContractError("confirmation_enabled_must_be_boolean")
        max_wait = _strict_int(
            self.confirmation_max_wait_observations,
            field_name="confirmation_max_wait_observations",
        )
        if not isinstance(self.arm_trace, Mapping):
            raise LifecycleContractError("arm_trace_must_be_a_mapping")
        frozen_trace = _freeze_json(self.arm_trace, path="arm_trace")
        assert isinstance(frozen_trace, Mapping)
        if not frozen_trace:
            raise LifecycleContractError("arm_trace_must_not_be_empty")
        trace_hash = _semantic_id(
            "candidate_arm_trace_v1",
            {"arm_trace": _thaw_json(frozen_trace)},
        )
        candidate_payload = {
            "strategy_spec_version": spec_version,
            "strategy_spec_contract_hash": spec_contract,
            "strategy_spec_instance_hash": spec_instance,
            "raw_input_bundle_hash": raw_hash,
            "symbol": symbol,
            "side": self.side.value,
            "timeframe_seconds": seconds,
            "arm_bar_open_ts": opened,
            "arm_candle_cutoff_ts": cutoff,
            "armed_high": armed_high,
            "armed_low": armed_low,
            "armed_close": armed_close,
            "invalidate_level": invalidate,
            "confirmation_enabled": self.confirmation_enabled,
            "confirmation_max_wait_observations": max_wait,
            "arm_trace_hash": trace_hash,
        }
        object.__setattr__(self, "strategy_spec_version", spec_version)
        object.__setattr__(self, "strategy_spec_contract_hash", spec_contract)
        object.__setattr__(self, "strategy_spec_instance_hash", spec_instance)
        object.__setattr__(self, "raw_input_bundle_hash", raw_hash)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe_seconds", seconds)
        object.__setattr__(self, "arm_bar_open_ts", opened)
        object.__setattr__(self, "arm_candle_cutoff_ts", cutoff)
        object.__setattr__(self, "armed_high", armed_high)
        object.__setattr__(self, "armed_low", armed_low)
        object.__setattr__(self, "armed_close", armed_close)
        object.__setattr__(self, "invalidate_level", invalidate)
        object.__setattr__(self, "confirmation_max_wait_observations", max_wait)
        object.__setattr__(self, "arm_trace", frozen_trace)
        object.__setattr__(self, "arm_trace_hash", trace_hash)
        object.__setattr__(
            self,
            "candidate_id",
            _semantic_id("candidate_arm_v1", candidate_payload),
        )

    @property
    def contract_version(self) -> str:
        return LIFECYCLE_CONTRACT_VERSION

    @property
    def contract_hash(self) -> str:
        return lifecycle_contract_hash()

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "contract_hash": self.contract_hash,
            "candidate_id": self.candidate_id,
            "strategy_spec_version": self.strategy_spec_version,
            "strategy_spec_contract_hash": self.strategy_spec_contract_hash,
            "strategy_spec_instance_hash": self.strategy_spec_instance_hash,
            "raw_input_bundle_hash": self.raw_input_bundle_hash,
            "symbol": self.symbol,
            "side": self.side.value,
            "timeframe_seconds": self.timeframe_seconds,
            "arm_bar_open_ts": self.arm_bar_open_ts,
            "arm_candle_cutoff_ts": self.arm_candle_cutoff_ts,
            "armed_high": self.armed_high,
            "armed_low": self.armed_low,
            "armed_close": self.armed_close,
            "invalidate_level": self.invalidate_level,
            "confirmation_enabled": self.confirmation_enabled,
            "confirmation_max_wait_observations": self.confirmation_max_wait_observations,
            "arm_trace_hash": self.arm_trace_hash,
            "arm_trace": _thaw_json(self.arm_trace),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateArmV1":
        if not isinstance(payload, Mapping):
            raise LifecycleContractError("candidate_arm_must_be_a_mapping")
        _require_exact_keys(payload, expected=_CANDIDATE_ARM_KEYS, path="candidate_arm")
        _require_contract_identity(payload, path="candidate_arm")
        if type(payload["side"]) is not str:
            raise LifecycleContractError("candidate_arm_side_is_invalid")
        try:
            side = CandidateSide(payload["side"])
        except (TypeError, ValueError) as exc:
            raise LifecycleContractError("candidate_arm_side_is_invalid") from exc
        rebuilt = cls(
            strategy_spec_version=payload["strategy_spec_version"],
            strategy_spec_contract_hash=payload["strategy_spec_contract_hash"],
            strategy_spec_instance_hash=payload["strategy_spec_instance_hash"],
            raw_input_bundle_hash=payload["raw_input_bundle_hash"],
            symbol=payload["symbol"],
            side=side,
            timeframe_seconds=payload["timeframe_seconds"],
            arm_bar_open_ts=payload["arm_bar_open_ts"],
            arm_candle_cutoff_ts=payload["arm_candle_cutoff_ts"],
            armed_high=payload["armed_high"],
            armed_low=payload["armed_low"],
            armed_close=payload["armed_close"],
            invalidate_level=payload["invalidate_level"],
            confirmation_enabled=payload["confirmation_enabled"],
            confirmation_max_wait_observations=payload[
                "confirmation_max_wait_observations"
            ],
            arm_trace=payload["arm_trace"],
        )
        _require_derived_id(
            payload["arm_trace_hash"],
            rebuilt.arm_trace_hash,
            field_name="arm_trace_hash",
        )
        _require_derived_id(
            payload["candidate_id"],
            rebuilt.candidate_id,
            field_name="candidate_id",
        )
        return rebuilt


@dataclass(frozen=True, slots=True)
class ConfirmationObservationV1:
    candidate_id: str
    observation_input_bundle_hash: str
    state: CandidateLifecycleState
    state_epoch: int
    timeframe_seconds: int
    observation_bar_open_ts: float
    observation_candle_cutoff_ts: float
    observed_high: float
    observed_low: float
    observed_close: float
    distinct_observation_count: int
    elapsed_bars: int
    observation_id: str = field(init=False)

    def __post_init__(self) -> None:
        candidate_id = _strict_sha256(self.candidate_id, field_name="candidate_id")
        input_hash = _strict_sha256(
            self.observation_input_bundle_hash,
            field_name="observation_input_bundle_hash",
        )
        if not isinstance(self.state, CandidateLifecycleState) or self.state not in _FOLLOW_UP_STATES:
            raise LifecycleContractError("confirmation_state_must_be_a_follow_up_state")
        epoch = _strict_int(self.state_epoch, field_name="state_epoch", minimum=1)
        opened, cutoff, seconds = _require_bar(
            open_ts=self.observation_bar_open_ts,
            cutoff_ts=self.observation_candle_cutoff_ts,
            timeframe_seconds=self.timeframe_seconds,
            prefix="observation",
        )
        high = _strict_number(self.observed_high, field_name="observed_high", positive=True)
        low = _strict_number(self.observed_low, field_name="observed_low", positive=True)
        close = _strict_number(self.observed_close, field_name="observed_close", positive=True)
        if low > close or close > high:
            raise LifecycleContractError("observed_close_must_lie_between_low_and_high")
        distinct = _strict_int(
            self.distinct_observation_count,
            field_name="distinct_observation_count",
        )
        elapsed = _strict_int(self.elapsed_bars, field_name="elapsed_bars")
        if self.state is CandidateLifecycleState.SAME_BAR:
            if distinct > elapsed:
                raise LifecycleContractError("distinct_observation_count_exceeds_elapsed_bars")
        else:
            if distinct < 1 or elapsed < 1:
                raise LifecycleContractError("follow_up_bar_requires_positive_counts")
            if distinct > elapsed:
                raise LifecycleContractError("distinct_observation_count_exceeds_elapsed_bars")
        payload = {
            "candidate_id": candidate_id,
            "observation_input_bundle_hash": input_hash,
            "state": self.state.value,
            "state_epoch": epoch,
            "timeframe_seconds": seconds,
            "observation_bar_open_ts": opened,
            "observation_candle_cutoff_ts": cutoff,
            "observed_high": high,
            "observed_low": low,
            "observed_close": close,
            "distinct_observation_count": distinct,
            "elapsed_bars": elapsed,
        }
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "observation_input_bundle_hash", input_hash)
        object.__setattr__(self, "state_epoch", epoch)
        object.__setattr__(self, "timeframe_seconds", seconds)
        object.__setattr__(self, "observation_bar_open_ts", opened)
        object.__setattr__(self, "observation_candle_cutoff_ts", cutoff)
        object.__setattr__(self, "observed_high", high)
        object.__setattr__(self, "observed_low", low)
        object.__setattr__(self, "observed_close", close)
        object.__setattr__(self, "distinct_observation_count", distinct)
        object.__setattr__(self, "elapsed_bars", elapsed)
        object.__setattr__(
            self,
            "observation_id",
            _semantic_id("confirmation_observation_v1", payload),
        )

    @property
    def contract_version(self) -> str:
        return LIFECYCLE_CONTRACT_VERSION

    @property
    def contract_hash(self) -> str:
        return lifecycle_contract_hash()

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "contract_hash": self.contract_hash,
            "observation_id": self.observation_id,
            "candidate_id": self.candidate_id,
            "observation_input_bundle_hash": self.observation_input_bundle_hash,
            "state": self.state.value,
            "state_epoch": self.state_epoch,
            "timeframe_seconds": self.timeframe_seconds,
            "observation_bar_open_ts": self.observation_bar_open_ts,
            "observation_candle_cutoff_ts": self.observation_candle_cutoff_ts,
            "observed_high": self.observed_high,
            "observed_low": self.observed_low,
            "observed_close": self.observed_close,
            "distinct_observation_count": self.distinct_observation_count,
            "elapsed_bars": self.elapsed_bars,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConfirmationObservationV1":
        if not isinstance(payload, Mapping):
            raise LifecycleContractError("confirmation_observation_must_be_a_mapping")
        _require_exact_keys(
            payload,
            expected=_CONFIRMATION_OBSERVATION_KEYS,
            path="confirmation_observation",
        )
        _require_contract_identity(payload, path="confirmation_observation")
        if type(payload["state"]) is not str:
            raise LifecycleContractError("confirmation_observation_state_is_invalid")
        try:
            state = CandidateLifecycleState(payload["state"])
        except (TypeError, ValueError) as exc:
            raise LifecycleContractError("confirmation_observation_state_is_invalid") from exc
        rebuilt = cls(
            candidate_id=payload["candidate_id"],
            observation_input_bundle_hash=payload["observation_input_bundle_hash"],
            state=state,
            state_epoch=payload["state_epoch"],
            timeframe_seconds=payload["timeframe_seconds"],
            observation_bar_open_ts=payload["observation_bar_open_ts"],
            observation_candle_cutoff_ts=payload["observation_candle_cutoff_ts"],
            observed_high=payload["observed_high"],
            observed_low=payload["observed_low"],
            observed_close=payload["observed_close"],
            distinct_observation_count=payload["distinct_observation_count"],
            elapsed_bars=payload["elapsed_bars"],
        )
        _require_derived_id(
            payload["observation_id"],
            rebuilt.observation_id,
            field_name="observation_id",
        )
        return rebuilt


@dataclass(frozen=True, slots=True)
class ProposalObservationV1:
    candidate_id: str
    side: CandidateSide
    state_epoch: int
    timeframe_seconds: int
    status: ProposalObservationStatus
    basis: ProposalObservationBasis
    confirmation_observation_id: str | None = None
    proposal_input_bundle_hash: str | None = None
    reference_bar_open_ts: float | None = None
    reference_candle_cutoff_ts: float | None = None
    decision_reference_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    rejection_reason: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)
    execution_bound: bool = False
    proposal_observation_id: str = field(init=False)

    def __post_init__(self) -> None:
        candidate_id = _strict_sha256(self.candidate_id, field_name="candidate_id")
        if not isinstance(self.side, CandidateSide):
            raise LifecycleContractError("proposal_side_must_be_candidate_side")
        epoch = _strict_int(self.state_epoch, field_name="state_epoch")
        seconds = _strict_int(
            self.timeframe_seconds,
            field_name="proposal_timeframe_seconds",
            minimum=1,
        )
        if not isinstance(self.status, ProposalObservationStatus):
            raise LifecycleContractError("proposal_status_is_invalid")
        if not isinstance(self.basis, ProposalObservationBasis):
            raise LifecycleContractError("proposal_basis_is_invalid")
        if type(self.execution_bound) is not bool:
            raise LifecycleContractError("execution_bound_must_be_boolean")
        if self.execution_bound:
            raise LifecycleContractError("proposal_observation_must_not_be_execution_bound")
        if not isinstance(self.details, Mapping):
            raise LifecycleContractError("proposal_details_must_be_a_mapping")
        frozen_details = _freeze_json(self.details, path="proposal_details")
        assert isinstance(frozen_details, Mapping)

        confirmation_id: str | None = None
        input_hash: str | None = None
        reference_open: float | None = None
        reference_cutoff: float | None = None
        decision_price: float | None = None
        stop_price: float | None = None
        take_profit: float | None = None
        rejection: str | None = None

        if self.status is ProposalObservationStatus.NOT_EVALUATED:
            if self.basis is not ProposalObservationBasis.NONE:
                raise LifecycleContractError("not_evaluated_proposal_requires_none_basis")
            optional_values = (
                self.confirmation_observation_id,
                self.proposal_input_bundle_hash,
                self.reference_bar_open_ts,
                self.reference_candle_cutoff_ts,
                self.decision_reference_price,
                self.stop_price,
                self.take_profit_price,
                self.rejection_reason,
            )
            if any(value is not None for value in optional_values):
                raise LifecycleContractError("not_evaluated_proposal_must_not_carry_outcome")
            if frozen_details:
                raise LifecycleContractError("not_evaluated_proposal_details_must_be_empty")
        else:
            if self.basis is ProposalObservationBasis.NONE:
                raise LifecycleContractError("proposal_outcome_requires_a_causal_basis")
            if self.basis is ProposalObservationBasis.CONFIRMATION:
                if epoch < 1:
                    raise LifecycleContractError("confirmation_proposal_requires_positive_epoch")
                confirmation_id = _strict_sha256(
                    self.confirmation_observation_id,
                    field_name="confirmation_observation_id",
                )
            elif self.basis is ProposalObservationBasis.ARM_BYPASS:
                if epoch != 0:
                    raise LifecycleContractError("arm_bypass_proposal_requires_epoch_zero")
                if self.confirmation_observation_id is not None:
                    raise LifecycleContractError("arm_bypass_must_not_link_confirmation")
            input_hash = _strict_sha256(
                self.proposal_input_bundle_hash,
                field_name="proposal_input_bundle_hash",
            )
            reference_open, reference_cutoff, parsed_seconds = _require_bar(
                open_ts=self.reference_bar_open_ts,
                cutoff_ts=self.reference_candle_cutoff_ts,
                timeframe_seconds=seconds,
                prefix="proposal_reference",
            )
            if parsed_seconds != seconds:  # defensive; `_require_bar` already normalizes it
                raise LifecycleContractError("proposal_timeframe_mismatch")
            if self.status is ProposalObservationStatus.CREATED:
                decision_price = _strict_number(
                    self.decision_reference_price,
                    field_name="decision_reference_price",
                    positive=True,
                )
                stop_price = _strict_number(
                    self.stop_price,
                    field_name="stop_price",
                    positive=True,
                )
                take_profit = _strict_number(
                    self.take_profit_price,
                    field_name="take_profit_price",
                    positive=True,
                )
                if self.rejection_reason is not None:
                    raise LifecycleContractError("created_proposal_must_not_carry_rejection_reason")
                if self.side is CandidateSide.SHORT:
                    if not stop_price > decision_price > take_profit:
                        raise LifecycleContractError("short_proposal_levels_are_invalid")
                elif not stop_price < decision_price < take_profit:
                    raise LifecycleContractError("long_proposal_levels_are_invalid")
            else:
                if any(
                    value is not None
                    for value in (
                        self.decision_reference_price,
                        self.stop_price,
                        self.take_profit_price,
                    )
                ):
                    raise LifecycleContractError("rejected_proposal_must_not_carry_geometry")
                rejection = _strict_code(
                    self.rejection_reason,
                    field_name="rejection_reason",
                )

        payload = {
            "candidate_id": candidate_id,
            "side": self.side.value,
            "state_epoch": epoch,
            "timeframe_seconds": seconds,
            "status": self.status.value,
            "basis": self.basis.value,
            "execution_bound": False,
            "confirmation_observation_id": confirmation_id,
            "proposal_input_bundle_hash": input_hash,
            "reference_bar_open_ts": reference_open,
            "reference_candle_cutoff_ts": reference_cutoff,
            "decision_reference_price": decision_price,
            "stop_price": stop_price,
            "take_profit_price": take_profit,
            "rejection_reason": rejection,
            "details": _thaw_json(frozen_details),
        }
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "state_epoch", epoch)
        object.__setattr__(self, "timeframe_seconds", seconds)
        object.__setattr__(self, "confirmation_observation_id", confirmation_id)
        object.__setattr__(self, "proposal_input_bundle_hash", input_hash)
        object.__setattr__(self, "reference_bar_open_ts", reference_open)
        object.__setattr__(self, "reference_candle_cutoff_ts", reference_cutoff)
        object.__setattr__(self, "decision_reference_price", decision_price)
        object.__setattr__(self, "stop_price", stop_price)
        object.__setattr__(self, "take_profit_price", take_profit)
        object.__setattr__(self, "rejection_reason", rejection)
        object.__setattr__(self, "details", frozen_details)
        object.__setattr__(self, "execution_bound", False)
        object.__setattr__(
            self,
            "proposal_observation_id",
            _semantic_id("proposal_observation_v1", payload),
        )

    @property
    def contract_version(self) -> str:
        return LIFECYCLE_CONTRACT_VERSION

    @property
    def contract_hash(self) -> str:
        return lifecycle_contract_hash()

    @classmethod
    def not_evaluated(
        cls,
        *,
        candidate_id: str,
        side: CandidateSide,
        state_epoch: int,
        timeframe_seconds: int,
    ) -> "ProposalObservationV1":
        return cls(
            candidate_id=candidate_id,
            side=side,
            state_epoch=state_epoch,
            timeframe_seconds=timeframe_seconds,
            status=ProposalObservationStatus.NOT_EVALUATED,
            basis=ProposalObservationBasis.NONE,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "contract_hash": self.contract_hash,
            "proposal_observation_id": self.proposal_observation_id,
            "candidate_id": self.candidate_id,
            "side": self.side.value,
            "state_epoch": self.state_epoch,
            "timeframe_seconds": self.timeframe_seconds,
            "status": self.status.value,
            "basis": self.basis.value,
            "execution_bound": False,
            "confirmation_observation_id": self.confirmation_observation_id,
            "proposal_input_bundle_hash": self.proposal_input_bundle_hash,
            "reference_bar_open_ts": self.reference_bar_open_ts,
            "reference_candle_cutoff_ts": self.reference_candle_cutoff_ts,
            "decision_reference_price": self.decision_reference_price,
            "stop_price": self.stop_price,
            "take_profit_price": self.take_profit_price,
            "rejection_reason": self.rejection_reason,
            "details": _thaw_json(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposalObservationV1":
        if not isinstance(payload, Mapping):
            raise LifecycleContractError("proposal_observation_must_be_a_mapping")
        _require_exact_keys(
            payload,
            expected=_PROPOSAL_OBSERVATION_KEYS,
            path="proposal_observation",
        )
        _require_contract_identity(payload, path="proposal_observation")
        if (
            type(payload["side"]) is not str
            or type(payload["status"]) is not str
            or type(payload["basis"]) is not str
        ):
            raise LifecycleContractError("proposal_observation_enum_is_invalid")
        try:
            side = CandidateSide(payload["side"])
            status = ProposalObservationStatus(payload["status"])
            basis = ProposalObservationBasis(payload["basis"])
        except (TypeError, ValueError) as exc:
            raise LifecycleContractError("proposal_observation_enum_is_invalid") from exc
        rebuilt = cls(
            candidate_id=payload["candidate_id"],
            side=side,
            state_epoch=payload["state_epoch"],
            timeframe_seconds=payload["timeframe_seconds"],
            status=status,
            basis=basis,
            confirmation_observation_id=payload["confirmation_observation_id"],
            proposal_input_bundle_hash=payload["proposal_input_bundle_hash"],
            reference_bar_open_ts=payload["reference_bar_open_ts"],
            reference_candle_cutoff_ts=payload["reference_candle_cutoff_ts"],
            decision_reference_price=payload["decision_reference_price"],
            stop_price=payload["stop_price"],
            take_profit_price=payload["take_profit_price"],
            rejection_reason=payload["rejection_reason"],
            details=payload["details"],
            execution_bound=payload["execution_bound"],
        )
        _require_derived_id(
            payload["proposal_observation_id"],
            rebuilt.proposal_observation_id,
            field_name="proposal_observation_id",
        )
        return rebuilt


@dataclass(frozen=True, slots=True)
class CandidateLifecycleEventV1:
    arm: CandidateArmV1
    state: CandidateLifecycleState
    state_epoch: int
    previous_event_id: str | None
    previous_state: CandidateLifecycleState | None
    confirmation: ConfirmationObservationV1 | None
    proposal: ProposalObservationV1
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.arm, CandidateArmV1):
            raise LifecycleContractError("event_requires_candidate_arm_v1")
        if not isinstance(self.state, CandidateLifecycleState):
            raise LifecycleContractError("event_state_is_invalid")
        epoch = _strict_int(self.state_epoch, field_name="state_epoch")
        if not isinstance(self.proposal, ProposalObservationV1):
            raise LifecycleContractError("event_requires_proposal_observation_v1")
        if self.proposal.candidate_id != self.arm.candidate_id:
            raise LifecycleContractError("proposal_candidate_id_mismatch")
        if self.proposal.side is not self.arm.side:
            raise LifecycleContractError("proposal_side_mismatch")
        if self.proposal.timeframe_seconds != self.arm.timeframe_seconds:
            raise LifecycleContractError("proposal_timeframe_mismatch")
        if self.proposal.state_epoch != epoch:
            raise LifecycleContractError("proposal_state_epoch_mismatch")

        previous_id: str | None = None
        if self.state is CandidateLifecycleState.ARMED:
            if not self.arm.confirmation_enabled:
                raise LifecycleContractError("armed_event_requires_confirmation_enabled")
            if epoch != 0:
                raise LifecycleContractError("armed_event_requires_epoch_zero")
            if self.previous_event_id is not None or self.previous_state is not None:
                raise LifecycleContractError("armed_event_must_not_have_a_predecessor")
            if self.confirmation is not None:
                raise LifecycleContractError("armed_event_must_not_have_confirmation")
            if self.proposal.status is not ProposalObservationStatus.NOT_EVALUATED:
                raise LifecycleContractError("armed_event_proposal_must_not_be_evaluated")
        elif self.state is CandidateLifecycleState.BYPASSED:
            if self.arm.confirmation_enabled:
                raise LifecycleContractError("bypassed_event_requires_confirmation_disabled")
            if epoch != 0:
                raise LifecycleContractError("bypassed_event_requires_epoch_zero")
            if self.previous_event_id is not None or self.previous_state is not None:
                raise LifecycleContractError("bypassed_event_must_not_have_a_predecessor")
            if self.confirmation is not None:
                raise LifecycleContractError("bypassed_event_must_not_have_confirmation")
            if self.proposal.status not in {
                ProposalObservationStatus.CREATED,
                ProposalObservationStatus.REJECTED,
            }:
                raise LifecycleContractError("bypassed_event_requires_proposal_outcome")
            if self.proposal.basis is not ProposalObservationBasis.ARM_BYPASS:
                raise LifecycleContractError("bypassed_event_requires_arm_bypass_basis")
            if self.proposal.confirmation_observation_id is not None:
                raise LifecycleContractError("bypassed_event_must_not_link_confirmation")
            if self.proposal.proposal_input_bundle_hash != self.arm.raw_input_bundle_hash:
                raise LifecycleContractError("bypassed_proposal_input_bundle_mismatch")
            if self.proposal.reference_bar_open_ts != self.arm.arm_bar_open_ts:
                raise LifecycleContractError("bypassed_proposal_reference_bar_mismatch")
            if self.proposal.reference_candle_cutoff_ts != self.arm.arm_candle_cutoff_ts:
                raise LifecycleContractError("bypassed_proposal_reference_cutoff_mismatch")
            if (
                self.proposal.status is ProposalObservationStatus.CREATED
                and self.proposal.decision_reference_price != self.arm.armed_close
            ):
                raise LifecycleContractError("bypassed_proposal_reference_price_mismatch")
        else:
            if not self.arm.confirmation_enabled:
                raise LifecycleContractError("follow_up_event_requires_confirmation_enabled")
            if epoch < 1:
                raise LifecycleContractError("follow_up_event_requires_positive_epoch")
            previous_id = _strict_sha256(
                self.previous_event_id,
                field_name="previous_event_id",
            )
            if not isinstance(self.previous_state, CandidateLifecycleState):
                raise LifecycleContractError("follow_up_event_requires_previous_state")
            allowed = _ALLOWED_TRANSITIONS.get(self.previous_state, frozenset())
            if self.state not in allowed:
                raise LifecycleContractError("illegal_lifecycle_transition")
            if not isinstance(self.confirmation, ConfirmationObservationV1):
                raise LifecycleContractError("follow_up_event_requires_confirmation")
            if self.confirmation.candidate_id != self.arm.candidate_id:
                raise LifecycleContractError("confirmation_candidate_id_mismatch")
            if self.confirmation.state_epoch != epoch:
                raise LifecycleContractError("confirmation_state_epoch_mismatch")
            if self.confirmation.state is not self.state:
                raise LifecycleContractError("confirmation_state_mismatch")
            if self.confirmation.timeframe_seconds != self.arm.timeframe_seconds:
                raise LifecycleContractError("confirmation_timeframe_mismatch")
            self._validate_observation_against_arm(self.confirmation)
            if self.state is CandidateLifecycleState.CONFIRMED:
                if self.proposal.status not in {
                    ProposalObservationStatus.CREATED,
                    ProposalObservationStatus.REJECTED,
                }:
                    raise LifecycleContractError("confirmed_event_requires_proposal_outcome")
                if self.proposal.basis is not ProposalObservationBasis.CONFIRMATION:
                    raise LifecycleContractError("confirmed_event_requires_confirmation_basis")
                if self.proposal.confirmation_observation_id != self.confirmation.observation_id:
                    raise LifecycleContractError("proposal_confirmation_link_mismatch")
                if (
                    self.proposal.proposal_input_bundle_hash
                    != self.confirmation.observation_input_bundle_hash
                ):
                    raise LifecycleContractError("proposal_input_bundle_mismatch")
                if self.proposal.reference_bar_open_ts != self.confirmation.observation_bar_open_ts:
                    raise LifecycleContractError("proposal_reference_bar_mismatch")
                if (
                    self.proposal.reference_candle_cutoff_ts
                    != self.confirmation.observation_candle_cutoff_ts
                ):
                    raise LifecycleContractError("proposal_reference_cutoff_mismatch")
                if (
                    self.proposal.status is ProposalObservationStatus.CREATED
                    and self.proposal.decision_reference_price
                    != self.confirmation.observed_close
                ):
                    raise LifecycleContractError("proposal_reference_price_mismatch")
            elif self.proposal.status is not ProposalObservationStatus.NOT_EVALUATED:
                raise LifecycleContractError("non_confirmed_event_must_not_evaluate_proposal")
            elif self.proposal.confirmation_observation_id is not None:
                raise LifecycleContractError("non_confirmed_proposal_must_not_link_confirmation")

        event_payload = {
            "candidate_id": self.arm.candidate_id,
            "state": self.state.value,
            "state_epoch": epoch,
            "previous_event_id": previous_id,
            "previous_state": self.previous_state.value if self.previous_state is not None else None,
            "confirmation_observation_id": (
                self.confirmation.observation_id if self.confirmation is not None else None
            ),
            "proposal_observation_id": self.proposal.proposal_observation_id,
        }
        object.__setattr__(self, "state_epoch", epoch)
        object.__setattr__(self, "previous_event_id", previous_id)
        object.__setattr__(
            self,
            "event_id",
            _semantic_id("candidate_lifecycle_event_v1", event_payload),
        )

    def _validate_observation_against_arm(
        self, observation: ConfirmationObservationV1
    ) -> None:
        delta = observation.observation_bar_open_ts - self.arm.arm_bar_open_ts
        if delta < -_TIME_EPSILON:
            raise LifecycleContractError("observation_bar_precedes_arm_bar")
        quotient = delta / float(self.arm.timeframe_seconds)
        rounded = round(quotient)
        if not math.isclose(quotient, rounded, rel_tol=0.0, abs_tol=1e-9):
            raise LifecycleContractError("observation_bar_is_not_on_arm_timeframe")
        if observation.elapsed_bars != int(rounded):
            raise LifecycleContractError("elapsed_bars_disagrees_with_arm_bar")
        if observation.state is CandidateLifecycleState.SAME_BAR:
            if self.previous_state is CandidateLifecycleState.ARMED:
                if observation.observation_input_bundle_hash != self.arm.raw_input_bundle_hash:
                    raise LifecycleContractError("same_bar_input_bundle_mismatch")
                if observation.observation_bar_open_ts != self.arm.arm_bar_open_ts:
                    raise LifecycleContractError("same_bar_observation_bar_mismatch")
                if observation.observation_candle_cutoff_ts != self.arm.arm_candle_cutoff_ts:
                    raise LifecycleContractError("same_bar_cutoff_mismatch")
                if observation.distinct_observation_count != 0:
                    raise LifecycleContractError("same_bar_distinct_count_mismatch")
                if observation.elapsed_bars != 0:
                    raise LifecycleContractError("same_bar_elapsed_bars_mismatch")
                if observation.observed_high != self.arm.armed_high:
                    raise LifecycleContractError("same_bar_observed_high_mismatch")
                if observation.observed_low != self.arm.armed_low:
                    raise LifecycleContractError("same_bar_observed_low_mismatch")
                if observation.observed_close != self.arm.armed_close:
                    raise LifecycleContractError("same_bar_observed_close_mismatch")
        elif observation.observation_bar_open_ts <= self.arm.arm_bar_open_ts:
            raise LifecycleContractError("follow_up_observation_must_follow_arm_bar")
        else:
            if self.arm.side is CandidateSide.SHORT:
                invalidated = observation.observed_high >= self.arm.invalidate_level
                confirmed = observation.observed_close < self.arm.armed_close
            else:
                invalidated = observation.observed_low <= self.arm.invalidate_level
                confirmed = observation.observed_close > self.arm.armed_close
            if invalidated:
                expected_state = CandidateLifecycleState.INVALIDATED
            elif confirmed:
                expected_state = CandidateLifecycleState.CONFIRMED
            elif (
                observation.distinct_observation_count
                >= self.arm.confirmation_max_wait_observations
            ):
                expected_state = CandidateLifecycleState.EXPIRED
            else:
                expected_state = CandidateLifecycleState.WAITING
            if observation.state is not expected_state:
                raise LifecycleContractError(
                    "confirmation_state_disagrees_with_market_and_policy"
                )

    @property
    def contract_version(self) -> str:
        return LIFECYCLE_CONTRACT_VERSION

    @property
    def contract_hash(self) -> str:
        return lifecycle_contract_hash()

    @classmethod
    def armed(cls, arm: CandidateArmV1) -> "CandidateLifecycleEventV1":
        if not isinstance(arm, CandidateArmV1):
            raise LifecycleContractError("armed_event_requires_candidate_arm_v1")
        return cls(
            arm=arm,
            state=CandidateLifecycleState.ARMED,
            state_epoch=0,
            previous_event_id=None,
            previous_state=None,
            confirmation=None,
            proposal=ProposalObservationV1.not_evaluated(
                candidate_id=arm.candidate_id,
                side=arm.side,
                state_epoch=0,
                timeframe_seconds=arm.timeframe_seconds,
            ),
        )

    @classmethod
    def bypassed(
        cls,
        arm: CandidateArmV1,
        *,
        proposal: ProposalObservationV1,
    ) -> "CandidateLifecycleEventV1":
        """Record the confirmation-disabled path without inventing an observation."""

        if not isinstance(arm, CandidateArmV1):
            raise LifecycleContractError("bypassed_event_requires_candidate_arm_v1")
        if not isinstance(proposal, ProposalObservationV1):
            raise LifecycleContractError("bypassed_event_requires_proposal_observation_v1")
        return cls(
            arm=arm,
            state=CandidateLifecycleState.BYPASSED,
            state_epoch=0,
            previous_event_id=None,
            previous_state=None,
            confirmation=None,
            proposal=proposal,
        )

    @classmethod
    def transition(
        cls,
        previous: "CandidateLifecycleEventV1",
        *,
        confirmation: ConfirmationObservationV1,
        proposal: ProposalObservationV1 | None = None,
    ) -> "CandidateLifecycleEventV1":
        if not isinstance(previous, CandidateLifecycleEventV1):
            raise LifecycleContractError("transition_requires_previous_event")
        if previous.state in _TERMINAL_STATES:
            raise LifecycleContractError("terminal_event_has_no_successor")
        if not isinstance(confirmation, ConfirmationObservationV1):
            raise LifecycleContractError("transition_requires_confirmation_observation")
        expected_epoch = previous.state_epoch + 1
        if confirmation.candidate_id != previous.arm.candidate_id:
            raise LifecycleContractError("transition_candidate_id_mismatch")
        if confirmation.state_epoch != expected_epoch:
            raise LifecycleContractError("transition_state_epoch_mismatch")
        if confirmation.state is CandidateLifecycleState.SAME_BAR:
            prior = previous.confirmation
            if prior is None:
                expected_input_hash = previous.arm.raw_input_bundle_hash
                expected_bar_open_ts = previous.arm.arm_bar_open_ts
                expected_cutoff_ts = previous.arm.arm_candle_cutoff_ts
                expected_distinct = 0
                expected_elapsed = 0
                expected_high = previous.arm.armed_high
                expected_low = previous.arm.armed_low
                expected_close = previous.arm.armed_close
            else:
                expected_input_hash = prior.observation_input_bundle_hash
                expected_bar_open_ts = prior.observation_bar_open_ts
                expected_cutoff_ts = prior.observation_candle_cutoff_ts
                expected_distinct = prior.distinct_observation_count
                expected_elapsed = prior.elapsed_bars
                expected_high = prior.observed_high
                expected_low = prior.observed_low
                expected_close = prior.observed_close
            if confirmation.observation_input_bundle_hash != expected_input_hash:
                raise LifecycleContractError("same_bar_input_bundle_mismatch")
            if confirmation.observation_bar_open_ts != expected_bar_open_ts:
                raise LifecycleContractError("same_bar_observation_bar_mismatch")
            if confirmation.observation_candle_cutoff_ts != expected_cutoff_ts:
                raise LifecycleContractError("same_bar_cutoff_mismatch")
            if confirmation.distinct_observation_count != expected_distinct:
                raise LifecycleContractError("same_bar_distinct_count_mismatch")
            if confirmation.elapsed_bars != expected_elapsed:
                raise LifecycleContractError("same_bar_elapsed_bars_mismatch")
            if confirmation.observed_high != expected_high:
                raise LifecycleContractError("same_bar_observed_high_mismatch")
            if confirmation.observed_low != expected_low:
                raise LifecycleContractError("same_bar_observed_low_mismatch")
            if confirmation.observed_close != expected_close:
                raise LifecycleContractError("same_bar_observed_close_mismatch")
        else:
            prior = previous.confirmation
            previous_open = (
                prior.observation_bar_open_ts
                if prior is not None
                else previous.arm.arm_bar_open_ts
            )
            previous_cutoff = (
                prior.observation_candle_cutoff_ts
                if prior is not None
                else previous.arm.arm_candle_cutoff_ts
            )
            previous_distinct = prior.distinct_observation_count if prior is not None else 0
            previous_elapsed = prior.elapsed_bars if prior is not None else 0
            if confirmation.observation_bar_open_ts <= previous_open:
                raise LifecycleContractError("successor_bar_must_be_strictly_later")
            if confirmation.observation_candle_cutoff_ts <= previous_cutoff:
                raise LifecycleContractError("successor_cutoff_must_be_strictly_later")
            if confirmation.elapsed_bars <= previous_elapsed:
                raise LifecycleContractError("successor_elapsed_bars_must_increase")
            if confirmation.distinct_observation_count != previous_distinct + 1:
                raise LifecycleContractError("successor_distinct_count_must_increment_once")
        resolved_proposal = proposal or ProposalObservationV1.not_evaluated(
            candidate_id=previous.arm.candidate_id,
            side=previous.arm.side,
            state_epoch=expected_epoch,
            timeframe_seconds=previous.arm.timeframe_seconds,
        )
        return cls(
            arm=previous.arm,
            state=confirmation.state,
            state_epoch=expected_epoch,
            previous_event_id=previous.event_id,
            previous_state=previous.state,
            confirmation=confirmation,
            proposal=resolved_proposal,
        )

    def validate_successor(self, successor: "CandidateLifecycleEventV1") -> None:
        if not isinstance(successor, CandidateLifecycleEventV1):
            raise LifecycleContractError("successor_must_be_a_lifecycle_event")
        if self.state in _TERMINAL_STATES:
            raise LifecycleContractError("terminal_event_has_no_successor")
        if successor.arm.candidate_id != self.arm.candidate_id:
            raise LifecycleContractError("successor_candidate_id_mismatch")
        if successor.arm != self.arm:
            raise LifecycleContractError("successor_arm_snapshot_mismatch")
        if successor.state_epoch != self.state_epoch + 1:
            raise LifecycleContractError("successor_state_epoch_mismatch")
        if successor.previous_event_id != self.event_id:
            raise LifecycleContractError("successor_previous_event_id_mismatch")
        if successor.previous_state is not self.state:
            raise LifecycleContractError("successor_previous_state_mismatch")
        if (
            successor.state is CandidateLifecycleState.SAME_BAR
            and successor.confirmation is not None
        ):
            prior = self.confirmation
            expected_input_hash = (
                prior.observation_input_bundle_hash
                if prior is not None
                else self.arm.raw_input_bundle_hash
            )
            expected_bar_open_ts = (
                prior.observation_bar_open_ts
                if prior is not None
                else self.arm.arm_bar_open_ts
            )
            expected_cutoff_ts = (
                prior.observation_candle_cutoff_ts
                if prior is not None
                else self.arm.arm_candle_cutoff_ts
            )
            expected_distinct = prior.distinct_observation_count if prior is not None else 0
            expected_elapsed = prior.elapsed_bars if prior is not None else 0
            expected_high = prior.observed_high if prior is not None else self.arm.armed_high
            expected_low = prior.observed_low if prior is not None else self.arm.armed_low
            expected_close = prior.observed_close if prior is not None else self.arm.armed_close
            observation = successor.confirmation
            if observation.observation_input_bundle_hash != expected_input_hash:
                raise LifecycleContractError("same_bar_input_bundle_mismatch")
            if observation.observation_bar_open_ts != expected_bar_open_ts:
                raise LifecycleContractError("same_bar_observation_bar_mismatch")
            if observation.observation_candle_cutoff_ts != expected_cutoff_ts:
                raise LifecycleContractError("same_bar_cutoff_mismatch")
            if observation.distinct_observation_count != expected_distinct:
                raise LifecycleContractError("same_bar_distinct_count_mismatch")
            if observation.elapsed_bars != expected_elapsed:
                raise LifecycleContractError("same_bar_elapsed_bars_mismatch")
            if observation.observed_high != expected_high:
                raise LifecycleContractError("same_bar_observed_high_mismatch")
            if observation.observed_low != expected_low:
                raise LifecycleContractError("same_bar_observed_low_mismatch")
            if observation.observed_close != expected_close:
                raise LifecycleContractError("same_bar_observed_close_mismatch")
        elif successor.confirmation is not None:
            prior = self.confirmation
            previous_open = (
                prior.observation_bar_open_ts
                if prior is not None
                else self.arm.arm_bar_open_ts
            )
            previous_cutoff = (
                prior.observation_candle_cutoff_ts
                if prior is not None
                else self.arm.arm_candle_cutoff_ts
            )
            previous_distinct = prior.distinct_observation_count if prior is not None else 0
            previous_elapsed = prior.elapsed_bars if prior is not None else 0
            observation = successor.confirmation
            if observation.observation_bar_open_ts <= previous_open:
                raise LifecycleContractError("successor_bar_must_be_strictly_later")
            if observation.observation_candle_cutoff_ts <= previous_cutoff:
                raise LifecycleContractError("successor_cutoff_must_be_strictly_later")
            if observation.elapsed_bars <= previous_elapsed:
                raise LifecycleContractError("successor_elapsed_bars_must_increase")
            if observation.distinct_observation_count != previous_distinct + 1:
                raise LifecycleContractError("successor_distinct_count_must_increment_once")

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "contract_hash": self.contract_hash,
            "event_id": self.event_id,
            "arm": self.arm.as_dict(),
            "state": self.state.value,
            "state_epoch": self.state_epoch,
            "previous_event_id": self.previous_event_id,
            "previous_state": self.previous_state.value if self.previous_state is not None else None,
            "confirmation": self.confirmation.as_dict() if self.confirmation is not None else None,
            "proposal": self.proposal.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateLifecycleEventV1":
        if not isinstance(payload, Mapping):
            raise LifecycleContractError("lifecycle_event_must_be_a_mapping")
        _require_exact_keys(payload, expected=_LIFECYCLE_EVENT_KEYS, path="lifecycle_event")
        _require_contract_identity(payload, path="lifecycle_event")
        if type(payload["state"]) is not str:
            raise LifecycleContractError("lifecycle_event_state_is_invalid")
        if payload["previous_state"] is not None and type(payload["previous_state"]) is not str:
            raise LifecycleContractError("lifecycle_event_state_is_invalid")
        try:
            state = CandidateLifecycleState(payload["state"])
            previous_state = (
                CandidateLifecycleState(payload["previous_state"])
                if payload["previous_state"] is not None
                else None
            )
        except (TypeError, ValueError) as exc:
            raise LifecycleContractError("lifecycle_event_state_is_invalid") from exc
        confirmation_payload = payload["confirmation"]
        confirmation = (
            ConfirmationObservationV1.from_dict(confirmation_payload)
            if confirmation_payload is not None
            else None
        )
        rebuilt = cls(
            arm=CandidateArmV1.from_dict(payload["arm"]),
            state=state,
            state_epoch=payload["state_epoch"],
            previous_event_id=payload["previous_event_id"],
            previous_state=previous_state,
            confirmation=confirmation,
            proposal=ProposalObservationV1.from_dict(payload["proposal"]),
        )
        _require_derived_id(payload["event_id"], rebuilt.event_id, field_name="event_id")
        return rebuilt


def _require_contract_identity(payload: Mapping[str, Any], *, path: str) -> None:
    if payload.get("contract_version") != LIFECYCLE_CONTRACT_VERSION:
        raise LifecycleContractError(f"{path}_contract_version_mismatch")
    if payload.get("contract_hash") != lifecycle_contract_hash():
        raise LifecycleContractError(f"{path}_contract_hash_mismatch")


def _require_derived_id(recorded: object, expected: str, *, field_name: str) -> None:
    value = _strict_sha256(recorded, field_name=field_name)
    if value != expected:
        raise LifecycleContractError(f"{field_name}_mismatch")


_CONTRACT_IDENTITY_KEYS = frozenset({"contract_version", "contract_hash"})
_CANDIDATE_ARM_KEYS = _CONTRACT_IDENTITY_KEYS | frozenset(
    {
        "candidate_id",
        "strategy_spec_version",
        "strategy_spec_contract_hash",
        "strategy_spec_instance_hash",
        "raw_input_bundle_hash",
        "symbol",
        "side",
        "timeframe_seconds",
        "arm_bar_open_ts",
        "arm_candle_cutoff_ts",
        "armed_high",
        "armed_low",
        "armed_close",
        "invalidate_level",
        "confirmation_enabled",
        "confirmation_max_wait_observations",
        "arm_trace_hash",
        "arm_trace",
    }
)
_CONFIRMATION_OBSERVATION_KEYS = _CONTRACT_IDENTITY_KEYS | frozenset(
    {
        "observation_id",
        "candidate_id",
        "observation_input_bundle_hash",
        "state",
        "state_epoch",
        "timeframe_seconds",
        "observation_bar_open_ts",
        "observation_candle_cutoff_ts",
        "observed_high",
        "observed_low",
        "observed_close",
        "distinct_observation_count",
        "elapsed_bars",
    }
)
_PROPOSAL_OBSERVATION_KEYS = _CONTRACT_IDENTITY_KEYS | frozenset(
    {
        "proposal_observation_id",
        "candidate_id",
        "side",
        "state_epoch",
        "timeframe_seconds",
        "status",
        "basis",
        "execution_bound",
        "confirmation_observation_id",
        "proposal_input_bundle_hash",
        "reference_bar_open_ts",
        "reference_candle_cutoff_ts",
        "decision_reference_price",
        "stop_price",
        "take_profit_price",
        "rejection_reason",
        "details",
    }
)
_LIFECYCLE_EVENT_KEYS = _CONTRACT_IDENTITY_KEYS | frozenset(
    {
        "event_id",
        "arm",
        "state",
        "state_epoch",
        "previous_event_id",
        "previous_state",
        "confirmation",
        "proposal",
    }
)


__all__ = [
    "CandidateArmV1",
    "CandidateLifecycleEventV1",
    "CandidateLifecycleState",
    "CandidateSide",
    "ConfirmationObservationV1",
    "LIFECYCLE_CONTRACT_VERSION",
    "LifecycleContractError",
    "ProposalObservationStatus",
    "ProposalObservationBasis",
    "ProposalObservationV1",
    "lifecycle_contract_hash",
    "lifecycle_contract_payload",
]
