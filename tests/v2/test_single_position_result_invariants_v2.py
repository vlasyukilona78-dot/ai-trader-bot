from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from backtesting.single_position import (
    EntryPlan,
    ExecutionCosts,
    ScoredCandidate,
    SinglePositionContract,
    SinglePositionContractError,
    SizingRules,
    build_replay_evidence,
    first_reachable_bar_open,
    replay_single_short,
    select_single_position,
    single_position_result_hash,
)


def _contract() -> SinglePositionContract:
    return SinglePositionContract(
        costs=ExecutionCosts(
            entry_fee_rate=0.0,
            exit_fee_rate=0.0,
            half_spread=0.0,
            entry_slippage=0.0,
            exit_slippage=0.0,
        ),
        sizing=SizingRules(
            equity_quote=1_000.0,
            risk_fraction=0.01,
            max_notional_quote=1_000.0,
            max_leverage=1.0,
            quantity_step=0.001,
            min_quantity=0.001,
            min_notional_quote=5.0,
        ),
        bar_interval_seconds=300,
        max_holding_seconds=600,
    )


def _plan(symbol: str, cohort: str, entry_bar: float) -> EntryPlan:
    return EntryPlan(
        symbol=symbol,
        cohort_id=cohort,
        decision_ts=entry_bar - 300.0,
        actionable_ts=entry_bar - 250.0,
        entry_eligible_ts=entry_bar - 240.0,
        entry_bar_open_ts=entry_bar,
        decision_price=100.0,
        stop_price=105.0,
        take_profit_price=95.0,
    )


def _bars(entry_bar: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"time": entry_bar, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {
                "time": entry_bar + 300.0,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
            },
        ]
    )


def _candidate(score: float, symbol: str, cohort: str, entry_bar: float) -> ScoredCandidate:
    plan = _plan(symbol, cohort, entry_bar)
    contract = _contract()
    bars = _bars(entry_bar)
    result = replay_single_short(bars, plan=plan, contract=contract)
    evidence = build_replay_evidence(bars, plan=plan, contract=contract)
    return ScoredCandidate(
        score=score,
        plan=plan,
        contract=contract,
        evidence=evidence,
        result=result,
    )


def _unsafe_rehashed_clone(result, **changes):
    """Model an attacker who can rewrite a row and recompute its content hash."""

    values = {**result.__dict__, **changes}
    values.pop("result_hash")
    clone = object.__new__(result.__class__)
    for name, value in values.items():
        object.__setattr__(clone, name, value)
    object.__setattr__(clone, "result_hash", single_position_result_hash(clone))
    clone.validate()
    return clone


@pytest.mark.parametrize(
    "field",
    [
        "exit_ts",
        "entry_reference_price",
        "quantity",
        "initial_notional_quote",
        "gross_pnl_quote",
        "fees_quote",
        "funding_pnl_quote",
        "net_pnl_quote",
        "return_on_notional",
        "return_on_risk",
    ],
)
def test_result_rejects_non_finite_persisted_numbers(field: str) -> None:
    result = _candidate(0.9, "AUSDT", "cycle-a", 1_200.0).result

    with pytest.raises(SinglePositionContractError, match="must_be_finite"):
        replace(result, **{field: float("nan")})


@pytest.mark.parametrize("field", ["stop_price", "take_profit_price"])
def test_candidate_binds_the_replayed_levels_to_the_plan(field: str) -> None:
    candidate = _candidate(0.9, "AUSDT", "cycle-a", 1_200.0)
    changed = _unsafe_rehashed_clone(
        candidate.result, **{field: getattr(candidate.result, field) + 0.25}
    )

    with pytest.raises(SinglePositionContractError, match=f"result_{field}_differ"):
        ScoredCandidate(
            candidate.score,
            candidate.plan,
            candidate.contract,
            candidate.evidence,
            changed,
        )


def test_unfilled_result_cannot_carry_fill_data_or_pnl() -> None:
    plan = _plan("AUSDT", "cycle-a", 1_200.0)
    invalidated = _bars(1_200.0)
    invalidated.loc[0, ["open", "high", "low", "close"]] = [106.0, 107.0, 105.5, 106.0]
    result = replay_single_short(invalidated, plan=plan, contract=_contract())
    assert not result.filled

    with pytest.raises(SinglePositionContractError, match="unfilled_trade_has_nonzero_amounts"):
        replace(result, net_pnl_quote=1.0)
    with pytest.raises(SinglePositionContractError, match="unfilled_trade_has_fill_data"):
        replace(result, exit_ts=1_500.0)


def test_filled_result_rejects_inconsistent_pnl_counts_and_schema() -> None:
    result = _candidate(0.9, "AUSDT", "cycle-a", 1_200.0).result

    with pytest.raises(SinglePositionContractError, match="net_pnl_inconsistent"):
        replace(result, net_pnl_quote=result.net_pnl_quote + 1.0)
    with pytest.raises(SinglePositionContractError, match="must_hold_a_bar"):
        replace(result, bars_held=0)
    with pytest.raises(SinglePositionContractError, match="must_be_integer"):
        replace(result, funding_events_applied=0.5)
    with pytest.raises(SinglePositionContractError, match="unsupported_contract_schema"):
        replace(result, contract_schema_version=2)


def test_selector_revalidates_nan_exit_and_cannot_unlock_the_book() -> None:
    poisoned = _candidate(0.9, "AUSDT", "cycle-a", 1_200.0)
    # Model an unsafe deserializer that bypasses frozen-dataclass __post_init__.
    object.__setattr__(poisoned.result, "exit_ts", float("nan"))
    later = _candidate(0.8, "BUSDT", "cycle-b", 1_500.0)

    with pytest.raises(SinglePositionContractError, match="exit_ts_must_be_finite"):
        select_single_position([poisoned, later], minimum_score=0.5)


def test_selector_never_selects_overlapping_intervals() -> None:
    first = _candidate(0.9, "AUSDT", "cycle-a", 1_200.0)  # exits at 1_800
    overlapping = _candidate(0.95, "BUSDT", "cycle-b", 1_500.0)
    boundary = _candidate(0.8, "CUSDT", "cycle-c", 1_800.0)

    selection = select_single_position(
        [overlapping, boundary, first], minimum_score=0.5
    )

    assert [candidate.result.symbol for candidate in selection.selected] == [
        "AUSDT",
        "CUSDT",
    ]
    assert selection.skipped_busy == 1
    selected = selection.selected
    assert selected[0].result.exit_ts <= selected[1].plan.entry_bar_open_ts


def test_replay_still_requires_the_exact_first_reachable_bar() -> None:
    plan = _plan("AUSDT", "cycle-a", 1_200.0)
    assert first_reachable_bar_open(plan.entry_eligible_ts, 300) == 1_200.0
    delayed = replace(plan, entry_bar_open_ts=1_500.0)

    with pytest.raises(SinglePositionContractError, match="first_reachable_bar"):
        replay_single_short(_bars(1_500.0), plan=delayed, contract=_contract())
