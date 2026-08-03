"""Executable research contract for one causal MEXC SHORT position.

This module is intentionally separate from the historical DCA replay.  It is the
contract future labels and model evaluation must use: one market entry, one stop,
one take-profit, explicit sizing and costs, optional timestamped funding, and a
global concurrency limit of exactly one position.

The contract is research-only.  It does not submit orders and must not be used as
evidence that the strategy has an edge or is safe for live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


_EPS = 1e-9


class SinglePositionContractError(ValueError):
    """Raised when an input cannot satisfy the causal replay contract."""


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _require_finite_positive(name: str, value: float) -> None:
    if not _finite(value) or float(value) <= 0:
        raise SinglePositionContractError(f"{name}_must_be_finite_positive")


@dataclass(frozen=True)
class ExecutionCosts:
    """All rates are fractions of quote notional, not percentages."""

    entry_fee_rate: float
    exit_fee_rate: float
    half_spread: float
    entry_slippage: float
    exit_slippage: float

    def __post_init__(self) -> None:
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
    # v2 moved the entry reference from the decision instant to the first bar
    # opening strictly after it. A v1 result was measured from a bar the decision
    # did not precede, so the versions are not comparable.
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.side != "SHORT":
            raise SinglePositionContractError("only_short_is_supported")
        if self.max_concurrent_positions != 1:
            raise SinglePositionContractError("concurrency_must_equal_one")
        if self.schema_version != 2:
            raise SinglePositionContractError("unsupported_schema_version")
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
    entry_bar_open_ts: float
    decision_price: float
    stop_price: float
    take_profit_price: float

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise SinglePositionContractError("symbol_required")
        if not isinstance(self.cohort_id, str) or not self.cohort_id.strip():
            raise SinglePositionContractError("cohort_id_required")
        if len(self.cohort_id) > 128:
            raise SinglePositionContractError("cohort_id_too_long")
        for name in ("decision_ts", "actionable_ts", "entry_bar_open_ts"):
            if not _finite(getattr(self, name)):
                raise SinglePositionContractError(f"{name}_must_be_finite")
        if self.actionable_ts < self.decision_ts:
            raise SinglePositionContractError("actionable_ts_precedes_decision_ts")
        # A decision known at a bar's open cannot be filled at that open.
        if self.entry_bar_open_ts <= self.actionable_ts:
            raise SinglePositionContractError("entry_bar_must_open_after_actionable_ts")
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
        if not _finite(self.timestamp) or not _finite(self.rate):
            raise SinglePositionContractError("funding_timestamp_and_rate_must_be_finite")
        _require_finite_positive("funding_mark_price", self.mark_price)


@dataclass(frozen=True)
class SinglePositionResult:
    symbol: str
    cohort_id: str
    decision_ts: float
    entry_bar_open_ts: float
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
    contract_schema_version: int


@dataclass(frozen=True)
class ScoredCandidate:
    """A causal score, the plan it scored, and the replay outcome.

    The plan is kept alongside the result so selection can read cohort membership
    and entry timing from what was known before the replay, never from the
    outcome.
    """

    score: float
    plan: EntryPlan
    result: SinglePositionResult

    def __post_init__(self) -> None:
        if not _finite(self.score):
            raise SinglePositionContractError("candidate_score_must_be_finite")
        if not isinstance(self.plan, EntryPlan):
            raise SinglePositionContractError("candidate_requires_its_entry_plan")
        if self.plan.symbol != self.result.symbol:
            raise SinglePositionContractError("candidate_plan_and_result_symbol_differ")
        if self.plan.cohort_id != self.result.cohort_id:
            raise SinglePositionContractError("candidate_plan_and_result_cohort_differ")


@dataclass(frozen=True)
class SinglePositionSelection:
    selected: tuple[ScoredCandidate, ...]
    skipped_below_threshold: int
    skipped_unfilled: int
    skipped_busy: int


def _empty_result(plan: EntryPlan, contract: SinglePositionContract, reason: str) -> SinglePositionResult:
    return SinglePositionResult(
        symbol=plan.symbol,
        cohort_id=plan.cohort_id,
        decision_ts=float(plan.decision_ts),
        entry_bar_open_ts=float(plan.entry_bar_open_ts),
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
        contract_schema_version=contract.schema_version,
    )


def _normalise_bars(
    bars: pd.DataFrame,
    *,
    entry_bar_open_ts: float,
    contract: SinglePositionContract,
) -> pd.DataFrame:
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

    frame = _normalise_bars(
        bars, entry_bar_open_ts=plan.entry_bar_open_ts, contract=contract
    )
    first_open = float(frame.iloc[0]["open"])
    if first_open >= plan.stop_price * (1.0 - _EPS):
        return _empty_result(plan, contract, "entry_invalidated_by_stop_gap")
    if first_open <= plan.take_profit_price * (1.0 + _EPS):
        return _empty_result(plan, contract, "entry_invalidated_by_target_gap")

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
        return _empty_result(plan, contract, "below_instrument_minimum")

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
    for payment in funding_payments:
        if entry_ts < payment.timestamp <= funding_cutoff_ts:
            funding_pnl += quantity * payment.mark_price * payment.rate
            funding_count += 1

    net_pnl = gross_pnl - fees + funding_pnl
    return SinglePositionResult(
        symbol=plan.symbol,
        cohort_id=plan.cohort_id,
        decision_ts=float(plan.decision_ts),
        entry_bar_open_ts=float(plan.entry_bar_open_ts),
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
    cohort_timing: dict[str, tuple[float, float]] = {}
    for candidate in candidates:
        timing = (candidate.plan.actionable_ts, candidate.plan.entry_bar_open_ts)
        existing = cohort_timing.setdefault(candidate.plan.cohort_id, timing)
        if existing != timing:
            raise SinglePositionContractError("cohort_timing_conflict")

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.plan.entry_bar_open_ts,
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
        if entry_bar_open_ts < active_until:
            skipped_busy += len(eligible)
            continue

        chosen = eligible[0]
        skipped_busy += len(eligible) - 1
        if not chosen.result.filled or chosen.result.exit_ts is None:
            skipped_unfilled += 1
            continue
        selected.append(chosen)
        active_until = float(chosen.result.exit_ts)

    return SinglePositionSelection(
        selected=tuple(selected),
        skipped_below_threshold=skipped_below,
        skipped_unfilled=skipped_unfilled,
        skipped_busy=skipped_busy,
    )
