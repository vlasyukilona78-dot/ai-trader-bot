"""Executable research contract for one causal MEXC SHORT position.

This module is intentionally separate from the historical DCA replay.  It is the
contract future labels and model evaluation must use: one market entry, one stop,
one take-profit, explicit sizing and costs, optional timestamped funding, and a
global concurrency limit of exactly one position.

The contract is research-only.  It does not submit orders and must not be used as
evidence that the strategy has an edge or is safe for live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


_EPS = 1e-9
SINGLE_POSITION_SCHEMA_VERSION = 3
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SinglePositionContractError(ValueError):
    """Raised when an input cannot satisfy the causal replay contract."""


def _finite(value: float) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _require_finite_positive(name: str, value: float) -> None:
    if not _finite(value) or float(value) <= 0:
        raise SinglePositionContractError(f"{name}_must_be_finite_positive")


def _require_close(name: str, actual: float, expected: float) -> None:
    """Reject derived values that do not belong to the recorded replay.

    Results are persisted and may later be reconstructed without going through
    :func:`replay_single_short`.  Checking the arithmetic at the data boundary
    prevents a malformed row from changing selection or aggregate PnL while
    still allowing ordinary floating-point round-off.
    """

    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
        raise SinglePositionContractError(f"{name}_inconsistent")


def _require_timestamp_equal(name: str, actual: float, expected: float) -> None:
    """Require bar-clock equality without a magnitude-dependent tolerance."""

    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=_EPS):
        raise SinglePositionContractError(f"{name}_inconsistent")


def _canonical_float(name: str, value: float) -> float:
    """Return the one JSON representation used by every v3 identity hash."""

    if not _finite(value):
        raise SinglePositionContractError(f"{name}_must_be_finite")
    numeric = float(value)
    # JSON distinguishes -0.0 from 0.0 although the replay does not.  Normalise
    # that one representational ambiguity before hashing.
    return 0.0 if numeric == 0.0 else numeric


def _canonical_sha256(kind: str, payload: object) -> str:
    envelope = {
        "kind": kind,
        "schema_version": SINGLE_POSITION_SCHEMA_VERSION,
        "payload": payload,
    }
    try:
        encoded = json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SinglePositionContractError(f"{kind}_is_not_canonical_json") from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SinglePositionContractError(f"{name}_must_be_lowercase_sha256")


@dataclass(frozen=True)
class ExecutionCosts:
    """All rates are fractions of quote notional, not percentages."""

    entry_fee_rate: float
    exit_fee_rate: float
    half_spread: float
    entry_slippage: float
    exit_slippage: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name, value in (
            ("entry_fee_rate", self.entry_fee_rate),
            ("exit_fee_rate", self.exit_fee_rate),
            ("half_spread", self.half_spread),
            ("entry_slippage", self.entry_slippage),
            ("exit_slippage", self.exit_slippage),
        ):
            if not _finite(value) or float(value) < 0:
                raise SinglePositionContractError(f"{name}_must_be_finite_non_negative")
        if self.half_spread + self.entry_slippage >= 1:
            raise SinglePositionContractError("entry_friction_must_be_below_one")


@dataclass(frozen=True)
class SizingRules:
    equity_quote: float
    risk_fraction: float
    max_notional_quote: float
    max_leverage: float
    quantity_step: float
    min_quantity: float
    min_notional_quote: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name, value in (
            ("equity_quote", self.equity_quote),
            ("risk_fraction", self.risk_fraction),
            ("max_notional_quote", self.max_notional_quote),
            ("max_leverage", self.max_leverage),
        ):
            _require_finite_positive(name, value)
        if self.risk_fraction > 1:
            raise SinglePositionContractError("risk_fraction_must_not_exceed_one")
        for name, value in (
            ("quantity_step", self.quantity_step),
            ("min_quantity", self.min_quantity),
            ("min_notional_quote", self.min_notional_quote),
        ):
            if not _finite(value) or float(value) < 0:
                raise SinglePositionContractError(f"{name}_must_be_finite_non_negative")


@dataclass(frozen=True)
class SinglePositionContract:
    costs: ExecutionCosts
    sizing: SizingRules
    bar_interval_seconds: int
    max_holding_seconds: int
    side: str = "SHORT"
    max_concurrent_positions: int = 1
    # v3 cryptographically binds the complete plan, contract and replay input to
    # every result.  v1/v2 rows cannot be interpreted as v3 because they did not
    # carry enough information to revalidate costs, sizing or bar chronology.
    schema_version: int = SINGLE_POSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.costs, ExecutionCosts):
            raise SinglePositionContractError("contract_requires_execution_costs")
        if not isinstance(self.sizing, SizingRules):
            raise SinglePositionContractError("contract_requires_sizing_rules")
        self.costs.validate()
        self.sizing.validate()
        if self.side != "SHORT":
            raise SinglePositionContractError("only_short_is_supported")
        if type(self.max_concurrent_positions) is not int or self.max_concurrent_positions != 1:
            raise SinglePositionContractError("concurrency_must_equal_one")
        if type(self.schema_version) is not int or self.schema_version != SINGLE_POSITION_SCHEMA_VERSION:
            raise SinglePositionContractError("unsupported_schema_version")
        if type(self.bar_interval_seconds) is not int or type(self.max_holding_seconds) is not int:
            raise SinglePositionContractError("bar_interval_and_horizon_must_be_integers")
        if self.bar_interval_seconds <= 0 or self.max_holding_seconds <= 0:
            raise SinglePositionContractError("bar_interval_and_horizon_must_be_positive")
        if self.max_holding_seconds % self.bar_interval_seconds != 0:
            raise SinglePositionContractError("horizon_must_contain_whole_bars")


@dataclass(frozen=True)
class EntryPlan:
    """One proposed SHORT, carrying the cohort it competed in.

    ``cohort_id`` identifies the evaluation cycle the proposal belongs to. It is
    part of the plan rather than the replay outcome on purpose: which candidates
    compete against each other is decided before any of them is replayed, so a
    selector that reconstructs cohorts from timestamps afterwards is reading the
    wrong thing. Two symbols decided milliseconds apart in one cycle are one
    cohort; two cycles that happen to share a timestamp are not.
    """

    symbol: str
    cohort_id: str
    decision_ts: float
    actionable_ts: float
    entry_eligible_ts: float
    entry_bar_open_ts: float
    decision_price: float
    stop_price: float
    take_profit_price: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise SinglePositionContractError("symbol_required")
        if not isinstance(self.cohort_id, str) or not self.cohort_id.strip():
            raise SinglePositionContractError("cohort_id_required")
        if len(self.cohort_id) > 128:
            raise SinglePositionContractError("cohort_id_too_long")
        for name in ("decision_ts", "actionable_ts", "entry_eligible_ts", "entry_bar_open_ts"):
            if not _finite(getattr(self, name)):
                raise SinglePositionContractError(f"{name}_must_be_finite")
        if self.actionable_ts < self.decision_ts:
            raise SinglePositionContractError("actionable_ts_precedes_decision_ts")
        # Eligibility can only be later than actionability: it additionally waits
        # for the cycle to be sealed.
        if self.entry_eligible_ts < self.actionable_ts:
            raise SinglePositionContractError("entry_eligible_ts_precedes_actionable_ts")
        # A decision known at a bar's open cannot be filled at that open.
        if self.entry_bar_open_ts <= self.entry_eligible_ts:
            raise SinglePositionContractError("entry_bar_must_open_after_entry_eligible_ts")
        _require_finite_positive("decision_price", self.decision_price)
        _require_finite_positive("stop_price", self.stop_price)
        _require_finite_positive("take_profit_price", self.take_profit_price)
        if not self.stop_price > self.decision_price > self.take_profit_price:
            raise SinglePositionContractError("short_levels_must_be_stop_above_entry_above_target")


@dataclass(frozen=True)
class FundingPayment:
    timestamp: float
    rate: float
    mark_price: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not _finite(self.timestamp) or not _finite(self.rate):
            raise SinglePositionContractError("funding_timestamp_and_rate_must_be_finite")
        _require_finite_positive("funding_mark_price", self.mark_price)


def _plan_payload(plan: EntryPlan) -> dict[str, object]:
    if not isinstance(plan, EntryPlan):
        raise SinglePositionContractError("plan_hash_requires_entry_plan")
    plan.validate()
    return {
        "symbol": plan.symbol,
        "cohort_id": plan.cohort_id,
        "decision_ts": _canonical_float("decision_ts", plan.decision_ts),
        "actionable_ts": _canonical_float("actionable_ts", plan.actionable_ts),
        "entry_eligible_ts": _canonical_float("entry_eligible_ts", plan.entry_eligible_ts),
        "entry_bar_open_ts": _canonical_float(
            "entry_bar_open_ts", plan.entry_bar_open_ts
        ),
        "decision_price": _canonical_float("decision_price", plan.decision_price),
        "stop_price": _canonical_float("stop_price", plan.stop_price),
        "take_profit_price": _canonical_float(
            "take_profit_price", plan.take_profit_price
        ),
    }


def plan_hash(plan: EntryPlan) -> str:
    """Canonical SHA-256 identity of every executable EntryPlan field."""

    return _canonical_sha256("single_position_entry_plan", _plan_payload(plan))


def _contract_payload(contract: SinglePositionContract) -> dict[str, object]:
    if not isinstance(contract, SinglePositionContract):
        raise SinglePositionContractError(
            "contract_hash_requires_single_position_contract"
        )
    contract.validate()
    costs = contract.costs
    sizing = contract.sizing
    return {
        "costs": {
            "entry_fee_rate": _canonical_float(
                "entry_fee_rate", costs.entry_fee_rate
            ),
            "exit_fee_rate": _canonical_float("exit_fee_rate", costs.exit_fee_rate),
            "half_spread": _canonical_float("half_spread", costs.half_spread),
            "entry_slippage": _canonical_float(
                "entry_slippage", costs.entry_slippage
            ),
            "exit_slippage": _canonical_float("exit_slippage", costs.exit_slippage),
        },
        "sizing": {
            "equity_quote": _canonical_float("equity_quote", sizing.equity_quote),
            "risk_fraction": _canonical_float("risk_fraction", sizing.risk_fraction),
            "max_notional_quote": _canonical_float(
                "max_notional_quote", sizing.max_notional_quote
            ),
            "max_leverage": _canonical_float("max_leverage", sizing.max_leverage),
            "quantity_step": _canonical_float("quantity_step", sizing.quantity_step),
            "min_quantity": _canonical_float("min_quantity", sizing.min_quantity),
            "min_notional_quote": _canonical_float(
                "min_notional_quote", sizing.min_notional_quote
            ),
        },
        "bar_interval_seconds": contract.bar_interval_seconds,
        "max_holding_seconds": contract.max_holding_seconds,
        "side": contract.side,
        "max_concurrent_positions": contract.max_concurrent_positions,
        "contract_schema_version": contract.schema_version,
    }


def contract_hash(contract: SinglePositionContract) -> str:
    """Canonical SHA-256 identity of costs, sizing and replay mechanics."""

    return _canonical_sha256(
        "single_position_contract", _contract_payload(contract)
    )


def _funding_payload(payment: FundingPayment) -> dict[str, float]:
    if not isinstance(payment, FundingPayment):
        raise SinglePositionContractError("funding_payments_must_be_typed")
    payment.validate()
    return {
        "timestamp": _canonical_float("funding_timestamp", payment.timestamp),
        "rate": _canonical_float("funding_rate", payment.rate),
        "mark_price": _canonical_float("funding_mark_price", payment.mark_price),
    }


def _validate_funding_sequence(
    funding_payments: tuple[FundingPayment, ...],
) -> None:
    previous_timestamp = -math.inf
    for payment in funding_payments:
        if not isinstance(payment, FundingPayment):
            raise SinglePositionContractError("funding_payments_must_be_typed")
        payment.validate()
        timestamp = float(payment.timestamp)
        if timestamp <= previous_timestamp:
            raise SinglePositionContractError(
                "funding_timestamps_must_be_strictly_increasing"
            )
        previous_timestamp = timestamp


def _bar_evidence_payload(
    row: tuple[float, float, float, float, float],
) -> dict[str, float]:
    if type(row) is not tuple or len(row) != 5:
        raise SinglePositionContractError("replay_evidence_bar_shape_invalid")
    timestamp, open_price, high, low, close = row
    return {
        "time": _canonical_float("bar_time", timestamp),
        "open": _canonical_float("bar_open", open_price),
        "high": _canonical_float("bar_high", high),
        "low": _canonical_float("bar_low", low),
        "close": _canonical_float("bar_close", close),
    }


def _replay_input_digest_from_components(
    *,
    bound_plan_hash: str,
    bound_contract_hash: str,
    bars: tuple[tuple[float, float, float, float, float], ...],
    funding_payments: tuple[FundingPayment, ...],
) -> str:
    _require_sha256("replay_evidence_plan_hash", bound_plan_hash)
    _require_sha256("replay_evidence_contract_hash", bound_contract_hash)
    return _canonical_sha256(
        "single_position_replay_input",
        {
            "plan_hash": bound_plan_hash,
            "contract_hash": bound_contract_hash,
            "bars": [_bar_evidence_payload(row) for row in bars],
            # This sequence is already proved strictly chronological. Duplicate
            # settlement timestamps are forbidden so one event cannot be
            # credited twice merely by repeating a row.
            "funding_payments": [
                _funding_payload(payment) for payment in funding_payments
            ],
        },
    )


@dataclass(frozen=True)
class ReplayEvidence:
    """Immutable, independently checkable market input for one replay.

    A bare digest supplied by the result is not evidence: it can be replaced and
    the result content hash recomputed.  This object carries the actual
    normalised bars and ordered funding observations whose digest it claims.
    """

    plan_hash: str
    contract_hash: str
    bars: tuple[tuple[float, float, float, float, float], ...]
    funding_payments: tuple[FundingPayment, ...]
    replay_input_hash: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_sha256("replay_evidence_plan_hash", self.plan_hash)
        _require_sha256("replay_evidence_contract_hash", self.contract_hash)
        _require_sha256("replay_evidence_input_hash", self.replay_input_hash)
        if type(self.bars) is not tuple or not self.bars:
            raise SinglePositionContractError("replay_evidence_bars_must_be_non_empty_tuple")
        for row in self.bars:
            payload = _bar_evidence_payload(row)
            if any(
                payload[field_name] <= 0
                for field_name in ("open", "high", "low", "close")
            ):
                raise SinglePositionContractError("replay_evidence_bar_prices_must_be_positive")
            if not (
                payload["high"] >= max(payload["open"], payload["close"])
                and payload["low"] <= min(payload["open"], payload["close"])
                and payload["high"] >= payload["low"]
            ):
                raise SinglePositionContractError("replay_evidence_ohlc_invalid")
        if type(self.funding_payments) is not tuple:
            raise SinglePositionContractError("replay_evidence_funding_must_be_tuple")
        _validate_funding_sequence(self.funding_payments)
        expected = _replay_input_digest_from_components(
            bound_plan_hash=self.plan_hash,
            bound_contract_hash=self.contract_hash,
            bars=self.bars,
            funding_payments=self.funding_payments,
        )
        if self.replay_input_hash != expected:
            raise SinglePositionContractError("replay_evidence_hash_mismatch")

    def to_frame(self) -> pd.DataFrame:
        self.validate()
        return pd.DataFrame(
            self.bars,
            columns=("time", "open", "high", "low", "close"),
            dtype=float,
        )

    def validate_against(
        self,
        *,
        plan: EntryPlan,
        contract: SinglePositionContract,
    ) -> None:
        self.validate()
        if not isinstance(plan, EntryPlan):
            raise SinglePositionContractError("replay_evidence_requires_entry_plan")
        if not isinstance(contract, SinglePositionContract):
            raise SinglePositionContractError("replay_evidence_requires_contract")
        if self.plan_hash != plan_hash(plan):
            raise SinglePositionContractError("replay_evidence_plan_hash_mismatch")
        if self.contract_hash != contract_hash(contract):
            raise SinglePositionContractError("replay_evidence_contract_hash_mismatch")
        normalised = _normalise_bars(
            self.to_frame(),
            entry_bar_open_ts=plan.entry_bar_open_ts,
            contract=contract,
        )
        if _normalised_bar_tuples(normalised) != self.bars:
            raise SinglePositionContractError("replay_evidence_bars_not_canonical")


@dataclass(frozen=True)
class SinglePositionResult:
    """A schema-v3 replay outcome bound to its complete causal inputs.

    The four hashes are required fields, not compatibility conveniences.  A
    v1/v2 result therefore cannot cross the candidate/portfolio boundary by
    leaving them blank.  ``result_hash`` covers every other field in this
    dataclass, including the input identities.
    """

    symbol: str
    cohort_id: str
    decision_ts: float
    entry_bar_open_ts: float
    plan_hash: str
    contract_hash: str
    replay_input_hash: str
    result_hash: str
    filled: bool
    exit_reason: str
    entry_ts: float | None
    exit_ts: float | None
    entry_reference_price: float | None
    entry_fill_price: float | None
    exit_reference_price: float | None
    exit_fill_price: float | None
    stop_price: float
    take_profit_price: float
    quantity: float
    initial_notional_quote: float
    risk_budget_quote: float
    gross_pnl_quote: float
    fees_quote: float
    funding_pnl_quote: float
    net_pnl_quote: float
    return_on_notional: float
    return_on_risk: float
    bars_held: int
    funding_events_applied: int
    bar_interval_seconds: int
    max_holding_seconds: int
    contract_schema_version: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate a replay result independently of its constructor path.

        ``select_single_position`` calls this again deliberately.  Frozen
        dataclasses can still be corrupted by unsafe deserializers or
        ``object.__setattr__``; the portfolio boundary must therefore fail closed
        even when construction-time validation was bypassed.
        """

        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise SinglePositionContractError("result_symbol_required")
        if not isinstance(self.cohort_id, str) or not self.cohort_id.strip():
            raise SinglePositionContractError("result_cohort_id_required")
        if len(self.cohort_id) > 128:
            raise SinglePositionContractError("result_cohort_id_too_long")
        if type(self.filled) is not bool:
            raise SinglePositionContractError("result_filled_must_be_boolean")
        if not isinstance(self.exit_reason, str) or not self.exit_reason:
            raise SinglePositionContractError("result_exit_reason_required")

        _require_sha256("result_plan_hash", self.plan_hash)
        _require_sha256("result_contract_hash", self.contract_hash)
        _require_sha256("result_replay_input_hash", self.replay_input_hash)
        _require_sha256("result_hash", self.result_hash)

        for name in ("decision_ts", "entry_bar_open_ts"):
            if not _finite(getattr(self, name)):
                raise SinglePositionContractError(f"result_{name}_must_be_finite")
        if self.entry_bar_open_ts <= self.decision_ts:
            raise SinglePositionContractError("result_entry_bar_must_follow_decision")

        _require_finite_positive("result_stop_price", self.stop_price)
        _require_finite_positive("result_take_profit_price", self.take_profit_price)
        if self.stop_price <= self.take_profit_price:
            raise SinglePositionContractError("result_short_levels_are_inverted")

        numeric_fields = (
            "quantity",
            "initial_notional_quote",
            "risk_budget_quote",
            "gross_pnl_quote",
            "fees_quote",
            "funding_pnl_quote",
            "net_pnl_quote",
            "return_on_notional",
            "return_on_risk",
        )
        for name in numeric_fields:
            if not _finite(getattr(self, name)):
                raise SinglePositionContractError(f"result_{name}_must_be_finite")
        if self.quantity < 0 or self.initial_notional_quote < 0 or self.fees_quote < 0:
            raise SinglePositionContractError("result_non_negative_amount_is_negative")
        if self.risk_budget_quote <= 0:
            raise SinglePositionContractError("result_risk_budget_must_be_positive")

        for name in (
            "bars_held",
            "funding_events_applied",
            "bar_interval_seconds",
            "max_holding_seconds",
            "contract_schema_version",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise SinglePositionContractError(f"result_{name}_must_be_integer")
        if self.bars_held < 0 or self.funding_events_applied < 0:
            raise SinglePositionContractError("result_counts_must_be_non_negative")
        if self.bar_interval_seconds <= 0 or self.max_holding_seconds <= 0:
            raise SinglePositionContractError(
                "result_bar_interval_and_horizon_must_be_positive"
            )
        if self.max_holding_seconds % self.bar_interval_seconds != 0:
            raise SinglePositionContractError("result_horizon_must_contain_whole_bars")
        if self.contract_schema_version != SINGLE_POSITION_SCHEMA_VERSION:
            raise SinglePositionContractError("result_unsupported_contract_schema_version")
        if abs(math.remainder(self.entry_bar_open_ts, self.bar_interval_seconds)) > _EPS:
            raise SinglePositionContractError("result_entry_bar_must_be_bar_aligned")

        optional_fields = (
            "entry_ts",
            "exit_ts",
            "entry_reference_price",
            "entry_fill_price",
            "exit_reference_price",
            "exit_fill_price",
        )
        for name in optional_fields:
            value = getattr(self, name)
            if value is not None and not _finite(value):
                raise SinglePositionContractError(f"result_{name}_must_be_finite_or_none")

        filled_reasons = {"horizon", "stop", "stop_gap", "take_profit"}
        unfilled_reasons = {
            "entry_invalidated_by_stop_gap",
            "entry_invalidated_by_target_gap",
            "below_instrument_minimum",
        }
        if not self.filled:
            if self.exit_reason not in unfilled_reasons:
                raise SinglePositionContractError("result_invalid_unfilled_exit_reason")
            if any(getattr(self, name) is not None for name in optional_fields):
                raise SinglePositionContractError("result_unfilled_trade_has_fill_data")
            zero_fields = (
                "quantity",
                "initial_notional_quote",
                "gross_pnl_quote",
                "fees_quote",
                "funding_pnl_quote",
                "net_pnl_quote",
                "return_on_notional",
                "return_on_risk",
            )
            if any(float(getattr(self, name)) != 0.0 for name in zero_fields):
                raise SinglePositionContractError("result_unfilled_trade_has_nonzero_amounts")
            if self.bars_held != 0 or self.funding_events_applied != 0:
                raise SinglePositionContractError("result_unfilled_trade_has_nonzero_counts")
            self._validate_result_hash()
            return

        if self.exit_reason not in filled_reasons:
            raise SinglePositionContractError("result_invalid_filled_exit_reason")
        if any(getattr(self, name) is None for name in optional_fields):
            raise SinglePositionContractError("result_filled_trade_missing_fill_data")

        # The None case was rejected immediately above. Local names make the
        # chronology and arithmetic below explicit to both type checkers and
        # future readers of the executable contract.
        entry_ts = float(self.entry_ts)  # type: ignore[arg-type]
        exit_ts = float(self.exit_ts)  # type: ignore[arg-type]
        entry_reference = float(self.entry_reference_price)  # type: ignore[arg-type]
        entry_fill = float(self.entry_fill_price)  # type: ignore[arg-type]
        exit_reference = float(self.exit_reference_price)  # type: ignore[arg-type]
        exit_fill = float(self.exit_fill_price)  # type: ignore[arg-type]

        for name, value in (
            ("entry_reference_price", entry_reference),
            ("entry_fill_price", entry_fill),
            ("exit_reference_price", exit_reference),
            ("exit_fill_price", exit_fill),
        ):
            _require_finite_positive(f"result_{name}", value)
        if entry_ts != self.entry_bar_open_ts:
            raise SinglePositionContractError("result_entry_ts_differs_from_entry_bar")
        if exit_ts <= entry_ts:
            raise SinglePositionContractError("result_exit_ts_must_follow_entry_ts")
        if self.bars_held <= 0:
            raise SinglePositionContractError("result_filled_trade_must_hold_a_bar")
        maximum_bars = self.max_holding_seconds // self.bar_interval_seconds
        if self.bars_held > maximum_bars:
            raise SinglePositionContractError("result_bars_held_exceeds_horizon")
        _require_timestamp_equal(
            "result_exit_ts",
            exit_ts,
            entry_ts + self.bars_held * self.bar_interval_seconds,
        )
        if self.exit_reason == "horizon" and self.bars_held != maximum_bars:
            raise SinglePositionContractError("result_horizon_exit_precedes_horizon")
        if self.quantity <= 0 or self.initial_notional_quote <= 0:
            raise SinglePositionContractError("result_filled_trade_amounts_must_be_positive")
        if not self.stop_price > entry_reference > self.take_profit_price:
            raise SinglePositionContractError("result_entry_reference_outside_short_levels")
        if entry_fill > entry_reference and not math.isclose(
            entry_fill, entry_reference, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise SinglePositionContractError("result_short_entry_fill_has_price_improvement")
        if exit_fill < exit_reference and not math.isclose(
            exit_fill, exit_reference, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise SinglePositionContractError("result_short_exit_fill_has_price_improvement")
        if self.funding_events_applied == 0 and self.funding_pnl_quote != 0.0:
            raise SinglePositionContractError("result_funding_pnl_without_funding_events")

        if self.exit_reason == "stop":
            _require_close("result_stop_exit_reference", exit_reference, self.stop_price)
        elif self.exit_reason == "stop_gap" and exit_reference < self.stop_price:
            raise SinglePositionContractError("result_stop_gap_reference_below_stop")
        elif self.exit_reason == "take_profit":
            _require_close(
                "result_take_profit_exit_reference", exit_reference, self.take_profit_price
            )
        elif self.exit_reason == "horizon" and not (
            self.stop_price > exit_reference > self.take_profit_price
        ):
            raise SinglePositionContractError("result_horizon_reference_outside_short_levels")

        _require_close(
            "result_initial_notional",
            self.initial_notional_quote,
            self.quantity * entry_fill,
        )
        _require_close(
            "result_gross_pnl",
            self.gross_pnl_quote,
            self.quantity * (entry_fill - exit_fill),
        )
        _require_close(
            "result_net_pnl",
            self.net_pnl_quote,
            self.gross_pnl_quote - self.fees_quote + self.funding_pnl_quote,
        )
        _require_close(
            "result_return_on_notional",
            self.return_on_notional,
            self.net_pnl_quote / self.initial_notional_quote,
        )
        _require_close(
            "result_return_on_risk",
            self.return_on_risk,
            self.net_pnl_quote / self.risk_budget_quote,
        )
        self._validate_result_hash()

    def _validate_result_hash(self) -> None:
        if self.result_hash != single_position_result_hash(self):
            raise SinglePositionContractError("result_hash_mismatch")

    def validate_against(
        self,
        *,
        plan: EntryPlan,
        contract: SinglePositionContract,
    ) -> None:
        """Revalidate this persisted result against its actual plan/contract."""

        self.validate()
        if not isinstance(plan, EntryPlan):
            raise SinglePositionContractError("result_requires_bound_entry_plan")
        if not isinstance(contract, SinglePositionContract):
            raise SinglePositionContractError("result_requires_bound_contract")
        plan.validate()
        contract.validate()

        expected_plan_hash = plan_hash(plan)
        expected_contract_hash = contract_hash(contract)
        if self.plan_hash != expected_plan_hash:
            raise SinglePositionContractError("result_plan_hash_mismatch")
        if self.contract_hash != expected_contract_hash:
            raise SinglePositionContractError("result_contract_hash_mismatch")
        if self.bar_interval_seconds != contract.bar_interval_seconds:
            raise SinglePositionContractError("result_bar_interval_differs_from_contract")
        if self.max_holding_seconds != contract.max_holding_seconds:
            raise SinglePositionContractError("result_horizon_differs_from_contract")
        if self.contract_schema_version != contract.schema_version:
            raise SinglePositionContractError("result_schema_differs_from_contract")

        for field_name in (
            "symbol",
            "cohort_id",
            "decision_ts",
            "entry_bar_open_ts",
            "stop_price",
            "take_profit_price",
        ):
            if getattr(self, field_name) != getattr(plan, field_name):
                raise SinglePositionContractError(
                    f"candidate_plan_and_result_{field_name}_differ"
                )

        expected_entry_bar = first_reachable_bar_open(
            plan.entry_eligible_ts, contract.bar_interval_seconds
        )
        if abs(plan.entry_bar_open_ts - expected_entry_bar) > _EPS:
            raise SinglePositionContractError(
                "entry_bar_must_be_the_first_reachable_bar"
            )

        expected_risk_budget = (
            contract.sizing.equity_quote * contract.sizing.risk_fraction
        )
        _require_close(
            "result_risk_budget", self.risk_budget_quote, expected_risk_budget
        )
        if not self.filled:
            return

        # The filled result must be arithmetically reproducible from the bound
        # execution contract.  This closes the v2 hole where a row could retain
        # plausible PnL while silently swapping fees, slippage or sizing rules.
        entry_reference = float(self.entry_reference_price)  # type: ignore[arg-type]
        exit_reference = float(self.exit_reference_price)  # type: ignore[arg-type]
        expected_entry_fill = _short_entry_fill(entry_reference, contract.costs)
        expected_exit_fill = _short_exit_fill(exit_reference, contract.costs)
        _require_close(
            "result_entry_fill_from_contract",
            float(self.entry_fill_price),  # type: ignore[arg-type]
            expected_entry_fill,
        )
        _require_close(
            "result_exit_fill_from_contract",
            float(self.exit_fill_price),  # type: ignore[arg-type]
            expected_exit_fill,
        )

        expected_stop_fill = _short_exit_fill(plan.stop_price, contract.costs)
        expected_quantity, _ = _size_position(
            expected_entry_fill, expected_stop_fill, contract
        )
        _require_close("result_quantity_from_contract", self.quantity, expected_quantity)

        expected_fees = self.quantity * (
            expected_entry_fill * contract.costs.entry_fee_rate
            + expected_exit_fill * contract.costs.exit_fee_rate
        )
        _require_close("result_fees_from_contract", self.fees_quote, expected_fees)

        sizing = contract.sizing
        if self.quantity + _EPS < sizing.min_quantity:
            raise SinglePositionContractError("result_quantity_below_contract_minimum")
        if self.initial_notional_quote + _EPS < sizing.min_notional_quote:
            raise SinglePositionContractError("result_notional_below_contract_minimum")
        if self.initial_notional_quote > sizing.max_notional_quote + _EPS:
            raise SinglePositionContractError("result_notional_exceeds_contract_maximum")
        if (
            self.initial_notional_quote
            > sizing.equity_quote * sizing.max_leverage + _EPS
        ):
            raise SinglePositionContractError("result_notional_exceeds_contract_leverage")

        stop_loss_per_unit = (
            expected_stop_fill
            - expected_entry_fill
            + expected_entry_fill * contract.costs.entry_fee_rate
            + expected_stop_fill * contract.costs.exit_fee_rate
        )
        if self.quantity * stop_loss_per_unit > expected_risk_budget + _EPS:
            raise SinglePositionContractError("result_stop_risk_exceeds_budget")


def _optional_canonical_float(name: str, value: float | None) -> float | None:
    return None if value is None else _canonical_float(name, value)


def _result_payload(result: SinglePositionResult) -> dict[str, object]:
    return {
        "symbol": result.symbol,
        "cohort_id": result.cohort_id,
        "decision_ts": _canonical_float("result_decision_ts", result.decision_ts),
        "entry_bar_open_ts": _canonical_float(
            "result_entry_bar_open_ts", result.entry_bar_open_ts
        ),
        "plan_hash": result.plan_hash,
        "contract_hash": result.contract_hash,
        "replay_input_hash": result.replay_input_hash,
        "filled": result.filled,
        "exit_reason": result.exit_reason,
        "entry_ts": _optional_canonical_float("result_entry_ts", result.entry_ts),
        "exit_ts": _optional_canonical_float("result_exit_ts", result.exit_ts),
        "entry_reference_price": _optional_canonical_float(
            "result_entry_reference_price", result.entry_reference_price
        ),
        "entry_fill_price": _optional_canonical_float(
            "result_entry_fill_price", result.entry_fill_price
        ),
        "exit_reference_price": _optional_canonical_float(
            "result_exit_reference_price", result.exit_reference_price
        ),
        "exit_fill_price": _optional_canonical_float(
            "result_exit_fill_price", result.exit_fill_price
        ),
        "stop_price": _canonical_float("result_stop_price", result.stop_price),
        "take_profit_price": _canonical_float(
            "result_take_profit_price", result.take_profit_price
        ),
        "quantity": _canonical_float("result_quantity", result.quantity),
        "initial_notional_quote": _canonical_float(
            "result_initial_notional_quote", result.initial_notional_quote
        ),
        "risk_budget_quote": _canonical_float(
            "result_risk_budget_quote", result.risk_budget_quote
        ),
        "gross_pnl_quote": _canonical_float(
            "result_gross_pnl_quote", result.gross_pnl_quote
        ),
        "fees_quote": _canonical_float("result_fees_quote", result.fees_quote),
        "funding_pnl_quote": _canonical_float(
            "result_funding_pnl_quote", result.funding_pnl_quote
        ),
        "net_pnl_quote": _canonical_float(
            "result_net_pnl_quote", result.net_pnl_quote
        ),
        "return_on_notional": _canonical_float(
            "result_return_on_notional", result.return_on_notional
        ),
        "return_on_risk": _canonical_float(
            "result_return_on_risk", result.return_on_risk
        ),
        "bars_held": result.bars_held,
        "funding_events_applied": result.funding_events_applied,
        "bar_interval_seconds": result.bar_interval_seconds,
        "max_holding_seconds": result.max_holding_seconds,
        "contract_schema_version": result.contract_schema_version,
    }


def single_position_result_hash(result: SinglePositionResult) -> str:
    """Canonical SHA-256 over a v3 result and its three input identities."""

    if not isinstance(result, SinglePositionResult):
        raise SinglePositionContractError(
            "result_hash_requires_single_position_result"
        )
    return _canonical_sha256("single_position_result", _result_payload(result))


def _build_result(**values: object) -> SinglePositionResult:
    """Construct a result whose mandatory content hash is correct at birth."""

    # Validation is intentionally delegated to SinglePositionResult.  The
    # temporary object bypasses __init__ only to compute the digest over exactly
    # the same field representation that the final constructor will validate.
    temporary = object.__new__(SinglePositionResult)
    for name, value in values.items():
        object.__setattr__(temporary, name, value)
    digest = single_position_result_hash(temporary)
    return SinglePositionResult(**values, result_hash=digest)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ScoredCandidate:
    """A causal score plus plan, contract, market evidence and replay outcome.

    Every binding object is required. Selection can read causal cohort timing,
    revalidate economics and deterministically reproduce the outcome from the
    concrete bars/funding instead of trusting a self-asserted digest.
    """

    score: float
    plan: EntryPlan
    contract: SinglePositionContract
    evidence: ReplayEvidence
    result: SinglePositionResult

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not _finite(self.score):
            raise SinglePositionContractError("candidate_score_must_be_finite")
        if not isinstance(self.plan, EntryPlan):
            raise SinglePositionContractError("candidate_requires_its_entry_plan")
        if not isinstance(self.contract, SinglePositionContract):
            raise SinglePositionContractError("candidate_requires_its_contract")
        if not isinstance(self.evidence, ReplayEvidence):
            raise SinglePositionContractError("candidate_requires_replay_evidence")
        if not isinstance(self.result, SinglePositionResult):
            raise SinglePositionContractError("candidate_requires_single_position_result")
        self.result.validate_against(plan=self.plan, contract=self.contract)
        self.evidence.validate_against(plan=self.plan, contract=self.contract)
        if self.result.replay_input_hash != self.evidence.replay_input_hash:
            raise SinglePositionContractError("candidate_replay_input_hash_mismatch")
        expected_result = replay_single_short(
            self.evidence.to_frame(),
            plan=self.plan,
            contract=self.contract,
            funding_payments=self.evidence.funding_payments,
        )
        if self.result != expected_result:
            raise SinglePositionContractError(
                "candidate_result_differs_from_replay_evidence"
            )
        if self.result.filled and self.result.entry_ts != self.plan.entry_bar_open_ts:
            raise SinglePositionContractError("filled_entry_ts_differs_from_the_plan")


@dataclass(frozen=True)
class SinglePositionSelection:
    selected: tuple[ScoredCandidate, ...]
    skipped_below_threshold: int
    skipped_unfilled: int
    skipped_busy: int


def _empty_result(
    plan: EntryPlan,
    contract: SinglePositionContract,
    reason: str,
    *,
    input_hash: str,
) -> SinglePositionResult:
    return _build_result(
        symbol=plan.symbol,
        cohort_id=plan.cohort_id,
        decision_ts=float(plan.decision_ts),
        entry_bar_open_ts=float(plan.entry_bar_open_ts),
        plan_hash=plan_hash(plan),
        contract_hash=contract_hash(contract),
        replay_input_hash=input_hash,
        filled=False,
        exit_reason=reason,
        entry_ts=None,
        exit_ts=None,
        entry_reference_price=None,
        entry_fill_price=None,
        exit_reference_price=None,
        exit_fill_price=None,
        stop_price=float(plan.stop_price),
        take_profit_price=float(plan.take_profit_price),
        quantity=0.0,
        initial_notional_quote=0.0,
        risk_budget_quote=float(contract.sizing.equity_quote * contract.sizing.risk_fraction),
        gross_pnl_quote=0.0,
        fees_quote=0.0,
        funding_pnl_quote=0.0,
        net_pnl_quote=0.0,
        return_on_notional=0.0,
        return_on_risk=0.0,
        bars_held=0,
        funding_events_applied=0,
        bar_interval_seconds=contract.bar_interval_seconds,
        max_holding_seconds=contract.max_holding_seconds,
        contract_schema_version=contract.schema_version,
    )


def first_reachable_bar_open(entry_eligible_ts: float, bar_interval_seconds: int) -> float:
    """The only bar a market entry may use: the first one opening after eligibility.

    Anything later is a deliberate delay, and a delayed entry quietly changes the
    trade being measured - it skips the move the signal was about and reports the
    result as if the plan had been followed.
    """

    interval = float(bar_interval_seconds)
    return (math.floor(float(entry_eligible_ts) / interval) + 1.0) * interval


def _normalise_bars(
    bars: pd.DataFrame,
    *,
    entry_bar_open_ts: float,
    contract: SinglePositionContract,
) -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame):
        raise SinglePositionContractError("bars_must_be_a_dataframe")
    required = {"time", "open", "high", "low", "close"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise SinglePositionContractError(f"missing_bar_columns:{','.join(missing)}")

    frame = bars.loc[:, ["time", "open", "high", "low", "close"]].copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame.to_numpy(dtype=float)
    if len(frame) == 0 or not np.isfinite(values).all():
        raise SinglePositionContractError("bars_must_be_non_empty_and_finite")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise SinglePositionContractError("bar_prices_must_be_positive")
    if not (
        (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
        & (frame["high"] >= frame["low"])
    ).all():
        raise SinglePositionContractError("invalid_ohlc_relationship")

    interval = float(contract.bar_interval_seconds)
    # The entry bar is the first one that opens after the decision was actionable,
    # so the horizon is measured from it rather than from the decision instant.
    if abs(math.remainder(entry_bar_open_ts, interval)) > _EPS:
        raise SinglePositionContractError("entry_bar_open_ts_must_be_bar_aligned")
    deadline = float(entry_bar_open_ts + contract.max_holding_seconds)
    frame = frame[
        (frame["time"] >= entry_bar_open_ts - _EPS) & (frame["time"] + interval <= deadline + _EPS)
    ]
    frame = frame.reset_index(drop=True)
    expected_bars = contract.max_holding_seconds // contract.bar_interval_seconds
    if len(frame) != expected_bars:
        raise SinglePositionContractError("incomplete_horizon")
    timestamps = frame["time"].to_numpy(dtype=float)
    if abs(timestamps[0] - entry_bar_open_ts) > _EPS:
        raise SinglePositionContractError("first_bar_must_open_at_entry_bar")
    if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), interval, rtol=0.0, atol=_EPS):
        raise SinglePositionContractError("bar_cadence_gap")
    if abs((timestamps[-1] + interval) - deadline) > _EPS:
        raise SinglePositionContractError("last_bar_must_close_at_horizon")
    return frame


def _normalised_bar_tuples(
    frame: pd.DataFrame,
) -> tuple[tuple[float, float, float, float, float], ...]:
    payload: list[tuple[float, float, float, float, float]] = []
    for row in frame.itertuples(index=False):
        payload.append(
            (
                _canonical_float("bar_time", row.time),
                _canonical_float("bar_open", row.open),
                _canonical_float("bar_high", row.high),
                _canonical_float("bar_low", row.low),
                _canonical_float("bar_close", row.close),
            )
        )
    return tuple(payload)


def _typed_funding_payments(
    funding_payments: Iterable[FundingPayment],
) -> tuple[FundingPayment, ...]:
    try:
        payments = tuple(funding_payments)
    except TypeError as exc:
        raise SinglePositionContractError("funding_payments_must_be_iterable") from exc
    _validate_funding_sequence(payments)
    return payments


def _evidence_from_normalised(
    frame: pd.DataFrame,
    *,
    plan: EntryPlan,
    contract: SinglePositionContract,
    funding_payments: tuple[FundingPayment, ...],
) -> ReplayEvidence:
    bound_plan_hash = plan_hash(plan)
    bound_contract_hash = contract_hash(contract)
    evidence_bars = _normalised_bar_tuples(frame)
    digest = _replay_input_digest_from_components(
        bound_plan_hash=bound_plan_hash,
        bound_contract_hash=bound_contract_hash,
        bars=evidence_bars,
        funding_payments=funding_payments,
    )
    return ReplayEvidence(
        plan_hash=bound_plan_hash,
        contract_hash=bound_contract_hash,
        bars=evidence_bars,
        funding_payments=funding_payments,
        replay_input_hash=digest,
    )


def build_replay_evidence(
    bars: pd.DataFrame,
    *,
    plan: EntryPlan,
    contract: SinglePositionContract,
    funding_payments: Iterable[FundingPayment] = (),
) -> ReplayEvidence:
    """Build the mandatory, concrete evidence for a scored replay outcome."""

    if not isinstance(plan, EntryPlan):
        raise SinglePositionContractError("replay_evidence_requires_entry_plan")
    if not isinstance(contract, SinglePositionContract):
        raise SinglePositionContractError("replay_evidence_requires_contract")
    plan.validate()
    contract.validate()
    expected_entry_bar = first_reachable_bar_open(
        plan.entry_eligible_ts, contract.bar_interval_seconds
    )
    if abs(plan.entry_bar_open_ts - expected_entry_bar) > _EPS:
        raise SinglePositionContractError("entry_bar_must_be_the_first_reachable_bar")
    frame = _normalise_bars(
        bars, entry_bar_open_ts=plan.entry_bar_open_ts, contract=contract
    )
    payments = _typed_funding_payments(funding_payments)
    return _evidence_from_normalised(
        frame,
        plan=plan,
        contract=contract,
        funding_payments=payments,
    )


def replay_input_hash(
    bars: pd.DataFrame,
    *,
    plan: EntryPlan,
    contract: SinglePositionContract,
    funding_payments: Iterable[FundingPayment] = (),
) -> str:
    """Recompute the exact canonical input identity carried by a v3 result."""

    if not isinstance(plan, EntryPlan):
        raise SinglePositionContractError("replay_hash_requires_entry_plan")
    if not isinstance(contract, SinglePositionContract):
        raise SinglePositionContractError("replay_hash_requires_contract")
    return build_replay_evidence(
        bars,
        plan=plan,
        contract=contract,
        funding_payments=funding_payments,
    ).replay_input_hash


def _short_entry_fill(reference: float, costs: ExecutionCosts) -> float:
    return reference * (1.0 - costs.half_spread - costs.entry_slippage)


def _short_exit_fill(reference: float, costs: ExecutionCosts) -> float:
    return reference * (1.0 + costs.half_spread + costs.exit_slippage)


def _floor_quantity(quantity: float, step: float) -> float:
    if step <= 0:
        return quantity
    return math.floor((quantity + _EPS) / step) * step


def _size_position(
    entry_fill: float,
    stop_fill: float,
    contract: SinglePositionContract,
) -> tuple[float, float]:
    costs = contract.costs
    sizing = contract.sizing
    risk_budget = sizing.equity_quote * sizing.risk_fraction
    stop_loss_per_unit = (
        stop_fill
        - entry_fill
        + entry_fill * costs.entry_fee_rate
        + stop_fill * costs.exit_fee_rate
    )
    if not _finite(stop_loss_per_unit) or stop_loss_per_unit <= 0:
        raise SinglePositionContractError("non_positive_stop_risk")

    by_risk = risk_budget / stop_loss_per_unit
    by_notional = sizing.max_notional_quote / entry_fill
    by_leverage = (sizing.equity_quote * sizing.max_leverage) / entry_fill
    quantity = _floor_quantity(min(by_risk, by_notional, by_leverage), sizing.quantity_step)
    return quantity, risk_budget


def replay_single_short(
    bars: pd.DataFrame,
    *,
    plan: EntryPlan,
    contract: SinglePositionContract,
    funding_payments: Iterable[FundingPayment] = (),
) -> SinglePositionResult:
    """Replay one executable SHORT using only bars opening after the decision.

    Bar timestamps are open times.  The first bar must open exactly at the plan's
    ``entry_bar_open_ts``, which is itself required to fall strictly after the
    decision became actionable; its open is the market-entry reference.  The
    complete, gap-free horizon is required.  If stop and target occur inside the
    same bar, the stop wins.  A gap through the stop exits at the worse bar open;
    a gap through the target receives no price improvement beyond the target.

    Positive funding rates mean longs pay shorts, so they add to SHORT PnL.
    Only payments with ``entry_ts < timestamp <= exit_ts`` are applied.
    """

    if not isinstance(plan, EntryPlan):
        raise SinglePositionContractError("replay_requires_entry_plan")
    if not isinstance(contract, SinglePositionContract):
        raise SinglePositionContractError("replay_requires_single_position_contract")
    plan.validate()
    contract.validate()

    expected_entry_bar = first_reachable_bar_open(
        plan.entry_eligible_ts, contract.bar_interval_seconds
    )
    if abs(plan.entry_bar_open_ts - expected_entry_bar) > _EPS:
        raise SinglePositionContractError("entry_bar_must_be_the_first_reachable_bar")

    frame = _normalise_bars(
        bars, entry_bar_open_ts=plan.entry_bar_open_ts, contract=contract
    )
    payments = _typed_funding_payments(funding_payments)
    evidence = _evidence_from_normalised(
        frame,
        plan=plan,
        contract=contract,
        funding_payments=payments,
    )
    input_hash = evidence.replay_input_hash
    first_open = float(frame.iloc[0]["open"])
    if first_open >= plan.stop_price * (1.0 - _EPS):
        return _empty_result(
            plan,
            contract,
            "entry_invalidated_by_stop_gap",
            input_hash=input_hash,
        )
    if first_open <= plan.take_profit_price * (1.0 + _EPS):
        return _empty_result(
            plan,
            contract,
            "entry_invalidated_by_target_gap",
            input_hash=input_hash,
        )

    entry_fill = _short_entry_fill(first_open, contract.costs)
    stop_fill = _short_exit_fill(plan.stop_price, contract.costs)
    quantity, risk_budget = _size_position(entry_fill, stop_fill, contract)
    initial_notional = quantity * entry_fill
    sizing = contract.sizing
    if (
        quantity <= 0
        or quantity + _EPS < sizing.min_quantity
        or initial_notional + _EPS < sizing.min_notional_quote
    ):
        return _empty_result(
            plan,
            contract,
            "below_instrument_minimum",
            input_hash=input_hash,
        )

    exit_reason = "horizon"
    exit_reference = float(frame.iloc[-1]["close"])
    exit_bar_index = len(frame) - 1
    for index, row in frame.iterrows():
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        if open_price >= plan.stop_price * (1.0 - _EPS):
            exit_reason = "stop_gap"
            exit_reference = open_price
            exit_bar_index = int(index)
            break
        if open_price <= plan.take_profit_price * (1.0 + _EPS):
            exit_reason = "take_profit"
            exit_reference = float(plan.take_profit_price)
            exit_bar_index = int(index)
            break

        stop_hit = high >= plan.stop_price * (1.0 - _EPS)
        target_hit = low <= plan.take_profit_price * (1.0 + _EPS)
        if stop_hit:
            exit_reason = "stop"
            exit_reference = float(plan.stop_price)
            exit_bar_index = int(index)
            break
        if target_hit:
            exit_reason = "take_profit"
            exit_reference = float(plan.take_profit_price)
            exit_bar_index = int(index)
            break

    # The fill happens on the entry bar's open, not at the instant the decision
    # was made. Dating it earlier would credit funding the position had not yet
    # been open to receive.
    entry_ts = float(plan.entry_bar_open_ts)
    exit_bar_open_ts = float(frame.iloc[exit_bar_index]["time"])
    exit_ts = float(exit_bar_open_ts + contract.bar_interval_seconds)
    exit_fill = _short_exit_fill(exit_reference, contract.costs)
    gross_pnl = quantity * (entry_fill - exit_fill)
    fees = (
        quantity * entry_fill * contract.costs.entry_fee_rate
        + quantity * exit_fill * contract.costs.exit_fee_rate
    )

    # OHLC data cannot reveal whether an intrabar stop/target happened before a
    # funding timestamp later in that bar.  Crediting it would be optimistic, so
    # touch/gap exits only receive payments known by the bar open.  A horizon
    # close is known to remain open through the full bar.
    funding_cutoff_ts = exit_ts if exit_reason == "horizon" else exit_bar_open_ts
    funding_pnl = 0.0
    funding_count = 0
    for payment in payments:
        if entry_ts < payment.timestamp <= funding_cutoff_ts:
            funding_pnl += quantity * payment.mark_price * payment.rate
            funding_count += 1

    net_pnl = gross_pnl - fees + funding_pnl
    return _build_result(
        symbol=plan.symbol,
        cohort_id=plan.cohort_id,
        decision_ts=float(plan.decision_ts),
        entry_bar_open_ts=float(plan.entry_bar_open_ts),
        plan_hash=plan_hash(plan),
        contract_hash=contract_hash(contract),
        replay_input_hash=input_hash,
        filled=True,
        exit_reason=exit_reason,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        entry_reference_price=first_open,
        entry_fill_price=entry_fill,
        exit_reference_price=exit_reference,
        exit_fill_price=exit_fill,
        stop_price=float(plan.stop_price),
        take_profit_price=float(plan.take_profit_price),
        quantity=quantity,
        initial_notional_quote=initial_notional,
        risk_budget_quote=risk_budget,
        gross_pnl_quote=gross_pnl,
        fees_quote=fees,
        funding_pnl_quote=funding_pnl,
        net_pnl_quote=net_pnl,
        return_on_notional=net_pnl / initial_notional,
        return_on_risk=net_pnl / risk_budget,
        bars_held=exit_bar_index + 1,
        funding_events_applied=funding_count,
        bar_interval_seconds=contract.bar_interval_seconds,
        max_holding_seconds=contract.max_holding_seconds,
        contract_schema_version=contract.schema_version,
    )


def select_single_position(
    candidates: Sequence[ScoredCandidate],
    *,
    minimum_score: float,
) -> SinglePositionSelection:
    """Select a deterministic chronological portfolio with concurrency one.

    Candidates are grouped by ``cohort_id``, never by a wall-clock timestamp: two
    symbols evaluated milliseconds apart inside one cycle belong to the same
    cohort, and two cycles that coincidentally share a timestamp do not. Within a
    cohort only the highest score is selected for an entry attempt. Outcomes never
    influence ranking: an unfilled top candidate cannot be replaced retrospectively
    by a lower-ranked symbol from the same cohort. ``exit_ts`` is used only after a
    filled candidate has been selected to determine when the book becomes free
    again.
    """

    if not _finite(minimum_score):
        raise SinglePositionContractError("minimum_score_must_be_finite")

    # A cohort is one cycle, so its entry timing is a single fact. If two rows
    # disagree the grouping is not trustworthy and the run must stop rather than
    # silently split or merge cohorts.
    cohort_timing: dict[str, tuple[float, float, float]] = {}
    for candidate in candidates:
        if not isinstance(candidate, ScoredCandidate):
            raise SinglePositionContractError("selector_requires_scored_candidates")
        # Revalidate at the portfolio boundary. This protects concurrency-one
        # from unsafe deserializers that bypassed the frozen dataclass constructor
        # and injected (for example) a NaN exit timestamp.
        candidate.validate()
        timing = (
            candidate.plan.actionable_ts,
            candidate.plan.entry_eligible_ts,
            candidate.plan.entry_bar_open_ts,
        )
        existing = cohort_timing.setdefault(candidate.plan.cohort_id, timing)
        if existing != timing:
            raise SinglePositionContractError("cohort_timing_conflict")

    # Two cohorts can target the same entry bar. The one that became actionable
    # first reserves the slot: it was decided while the other did not yet exist,
    # so letting the later one take the bar would be a retroactive substitution.
    # Ordering by cohort_id instead would decide it by SHA order, which carries no
    # causal meaning at all. Ties fall back to the identifier only to stay stable.
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.plan.entry_bar_open_ts,
            candidate.plan.actionable_ts,
            candidate.plan.cohort_id,
            -candidate.score,
            candidate.result.symbol,
        ),
    )
    selected: list[ScoredCandidate] = []
    skipped_below = 0
    skipped_unfilled = 0
    skipped_busy = 0
    active_until = -math.inf
    # An entry bar is consumed by the attempt, not by the fill. Whether the
    # leader's entry filled is only known once that bar has printed, and by then a
    # competing cohort's entry on the same bar is equally in the past. Letting the
    # runner-up cohort take it would be the same hindsight substitution the
    # within-cohort rule already forbids.
    attempted_entry_bars: set[float] = set()
    index = 0
    while index < len(ordered):
        cohort_id = ordered[index].plan.cohort_id
        entry_bar_open_ts = ordered[index].plan.entry_bar_open_ts
        group: list[ScoredCandidate] = []
        while index < len(ordered) and ordered[index].plan.cohort_id == cohort_id:
            group.append(ordered[index])
            index += 1

        eligible: list[ScoredCandidate] = []
        for candidate in group:
            if candidate.score < minimum_score:
                skipped_below += 1
            else:
                eligible.append(candidate)
        if not eligible:
            continue
        if entry_bar_open_ts in attempted_entry_bars or entry_bar_open_ts < active_until:
            skipped_busy += len(eligible)
            continue

        chosen = eligible[0]
        attempted_entry_bars.add(entry_bar_open_ts)
        skipped_busy += len(eligible) - 1
        if not chosen.result.filled or chosen.result.exit_ts is None:
            skipped_unfilled += 1
            continue
        if not _finite(chosen.result.exit_ts):
            raise SinglePositionContractError("selected_exit_ts_must_be_finite")
        selected.append(chosen)
        active_until = float(chosen.result.exit_ts)

    return SinglePositionSelection(
        selected=tuple(selected),
        skipped_below_threshold=skipped_below,
        skipped_unfilled=skipped_unfilled,
        skipped_busy=skipped_busy,
    )
