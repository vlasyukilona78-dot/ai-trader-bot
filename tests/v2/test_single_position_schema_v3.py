from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from backtesting.single_position import (
    EntryPlan,
    ExecutionCosts,
    FundingPayment,
    ScoredCandidate,
    SinglePositionContract,
    SinglePositionContractError,
    SinglePositionResult,
    SizingRules,
    build_replay_evidence,
    contract_hash,
    plan_hash,
    replay_input_hash,
    replay_single_short,
    select_single_position,
    single_position_result_hash,
)


def _plan(**changes) -> EntryPlan:
    values = {
        "symbol": "AAAUSDT",
        "cohort_id": "cycle-a",
        "decision_ts": 900.0,
        "actionable_ts": 950.0,
        "entry_eligible_ts": 960.0,
        "entry_bar_open_ts": 1_200.0,
        "decision_price": 100.0,
        "stop_price": 105.0,
        "take_profit_price": 95.0,
    }
    values.update(changes)
    return EntryPlan(**values)


def _contract(
    *,
    costs: ExecutionCosts | None = None,
    sizing: SizingRules | None = None,
) -> SinglePositionContract:
    return SinglePositionContract(
        costs=costs
        or ExecutionCosts(
            entry_fee_rate=0.0,
            exit_fee_rate=0.0,
            half_spread=0.0,
            entry_slippage=0.0,
            exit_slippage=0.0,
        ),
        sizing=sizing
        or SizingRules(
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


def _bars(*, first_close: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time": 1_200.0,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": first_close,
            },
            {
                "time": 1_500.0,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
            },
        ]
    )


def _replay():
    plan = _plan()
    contract = _contract()
    bars = _bars()
    result = replay_single_short(bars, plan=plan, contract=contract)
    evidence = build_replay_evidence(bars, plan=plan, contract=contract)
    return plan, contract, bars, evidence, result


def _rehashed_result(result: SinglePositionResult, **changes) -> SinglePositionResult:
    """Return a fully self-consistent content-hash wrapper for adversarial rows."""

    values = {**result.__dict__, **changes}
    values.pop("result_hash")
    temporary = object.__new__(SinglePositionResult)
    for name, value in values.items():
        object.__setattr__(temporary, name, value)
    values["result_hash"] = single_position_result_hash(temporary)
    return SinglePositionResult(**values)


def test_v3_hashes_are_canonical_complete_and_recomputable() -> None:
    plan, contract, bars, _, result = _replay()

    assert result.contract_schema_version == 3
    assert result.plan_hash == plan_hash(plan)
    assert result.contract_hash == contract_hash(contract)
    assert result.replay_input_hash == replay_input_hash(
        bars, plan=plan, contract=contract
    )
    assert result.result_hash == single_position_result_hash(result)
    assert all(
        len(value) == 64 and value == value.lower()
        for value in (
            result.plan_hash,
            result.contract_hash,
            result.replay_input_hash,
            result.result_hash,
        )
    )

    equivalent_plan = _plan()
    equivalent_contract = _contract()
    assert plan_hash(equivalent_plan) == plan_hash(plan)
    assert contract_hash(equivalent_contract) == contract_hash(contract)


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", "BBBUSDT"),
        ("cohort_id", "cycle-b"),
        ("decision_ts", 901.0),
        ("actionable_ts", 951.0),
        ("entry_eligible_ts", 961.0),
        ("entry_bar_open_ts", 1_500.0),
        ("decision_price", 100.5),
        ("stop_price", 106.0),
        ("take_profit_price", 94.0),
    ],
)
def test_plan_hash_binds_every_entry_plan_field(field: str, value) -> None:
    baseline = _plan()
    changes = {field: value}
    # Keep temporal/price relationships valid while changing the selected field.
    if field == "entry_eligible_ts":
        changes["entry_bar_open_ts"] = 1_200.0
    changed = replace(baseline, **changes)
    assert plan_hash(changed) != plan_hash(baseline)


@pytest.mark.parametrize(
    "cost_field",
    [
        "entry_fee_rate",
        "exit_fee_rate",
        "half_spread",
        "entry_slippage",
        "exit_slippage",
    ],
)
def test_contract_hash_binds_every_execution_cost(cost_field: str) -> None:
    baseline = _contract()
    changed_costs = replace(baseline.costs, **{cost_field: 0.0001})
    assert contract_hash(_contract(costs=changed_costs)) != contract_hash(baseline)


@pytest.mark.parametrize(
    "sizing_field,value",
    [
        ("equity_quote", 1_100.0),
        ("risk_fraction", 0.02),
        ("max_notional_quote", 900.0),
        ("max_leverage", 2.0),
        ("quantity_step", 0.01),
        ("min_quantity", 0.01),
        ("min_notional_quote", 10.0),
    ],
)
def test_contract_hash_binds_every_sizing_rule(sizing_field: str, value: float) -> None:
    baseline = _contract()
    changed_sizing = replace(baseline.sizing, **{sizing_field: value})
    assert contract_hash(_contract(sizing=changed_sizing)) != contract_hash(baseline)


def test_contract_hash_binds_bar_interval_and_horizon() -> None:
    baseline = _contract()
    assert contract_hash(
        replace(baseline, bar_interval_seconds=60)
    ) != contract_hash(baseline)
    assert contract_hash(
        replace(baseline, max_holding_seconds=900)
    ) != contract_hash(baseline)


def test_replay_input_hash_binds_bars_and_requires_chronological_funding() -> None:
    plan = _plan()
    contract = _contract()
    bars = _bars()
    first = FundingPayment(timestamp=1_300.0, rate=0.001, mark_price=100.0)
    second = FundingPayment(timestamp=1_400.0, rate=-0.002, mark_price=101.0)

    baseline = replay_input_hash(
        bars,
        plan=plan,
        contract=contract,
        funding_payments=(first, second),
    )
    changed_bars = bars.copy()
    changed_bars.loc[1, "close"] = 100.5
    assert replay_input_hash(
        changed_bars,
        plan=plan,
        contract=contract,
        funding_payments=(first, second),
    ) != baseline
    with pytest.raises(
        SinglePositionContractError,
        match="funding_timestamps_must_be_strictly_increasing",
    ):
        replay_input_hash(
            bars,
            plan=plan,
            contract=contract,
            funding_payments=(second, first),
        )


def test_duplicate_funding_timestamp_fails_closed_before_replay() -> None:
    plan = _plan()
    contract = _contract()
    first = FundingPayment(timestamp=1_300.0, rate=0.001, mark_price=100.0)
    duplicate = FundingPayment(timestamp=1_300.0, rate=0.002, mark_price=101.0)

    with pytest.raises(
        SinglePositionContractError,
        match="funding_timestamps_must_be_strictly_increasing",
    ):
        replay_single_short(
            _bars(),
            plan=plan,
            contract=contract,
            funding_payments=(first, duplicate),
        )


def test_arbitrary_replay_hash_cannot_cross_the_candidate_boundary() -> None:
    plan, contract, _, evidence, result = _replay()
    forged = _rehashed_result(result, replay_input_hash="f" * 64)
    # The forged result is syntactically valid and its own content hash is valid.
    forged.validate()

    with pytest.raises(
        SinglePositionContractError, match="candidate_replay_input_hash_mismatch"
    ):
        ScoredCandidate(0.8, plan, contract, evidence, forged)


def test_replay_evidence_rejects_an_arbitrary_digest() -> None:
    _, _, _, evidence, _ = _replay()
    with pytest.raises(SinglePositionContractError, match="replay_evidence_hash_mismatch"):
        replace(evidence, replay_input_hash="f" * 64)


def test_candidate_replays_evidence_instead_of_trusting_a_matching_digest() -> None:
    plan, contract, bars, _, result = _replay()
    alternate_bars = bars.copy()
    alternate_bars.loc[1, "close"] = 101.0
    alternate_evidence = build_replay_evidence(
        alternate_bars, plan=plan, contract=contract
    )
    # Bind the old economics to the alternate, valid market-input digest and
    # recompute the result content hash. Hash equality alone would accept this.
    forged = _rehashed_result(
        result, replay_input_hash=alternate_evidence.replay_input_hash
    )
    forged.validate()

    with pytest.raises(
        SinglePositionContractError,
        match="candidate_result_differs_from_replay_evidence",
    ):
        ScoredCandidate(0.8, plan, contract, alternate_evidence, forged)


def test_candidate_rejects_a_different_full_plan_even_with_valid_result_hash() -> None:
    plan, contract, _, evidence, result = _replay()
    changed_plan = replace(plan, actionable_ts=951.0)

    with pytest.raises(SinglePositionContractError, match="plan_hash_mismatch"):
        ScoredCandidate(0.8, changed_plan, contract, evidence, result)


def test_candidate_rejects_a_different_contract_even_when_result_is_self_consistent() -> None:
    plan, contract, _, evidence, result = _replay()
    paid = _contract(
        costs=replace(contract.costs, entry_fee_rate=0.0005)
    )

    with pytest.raises(SinglePositionContractError, match="contract_hash_mismatch"):
        ScoredCandidate(0.8, plan, paid, evidence, result)


@pytest.mark.parametrize(
    "changes,error",
    [
        (
            {
                "fees_quote": 1.0,
                "net_pnl_quote": -1.0,
                "return_on_notional": -0.005,
                "return_on_risk": -0.1,
            },
            "fees_from_contract",
        ),
        (
            {"quantity": 1.0, "initial_notional_quote": 100.0},
            "quantity_from_contract",
        ),
        (
            {
                "entry_fill_price": 99.0,
                "initial_notional_quote": 198.0,
                "gross_pnl_quote": -2.0,
                "net_pnl_quote": -2.0,
                "return_on_notional": -2.0 / 198.0,
                "return_on_risk": -0.2,
            },
            "entry_fill_from_contract",
        ),
        ({"risk_budget_quote": 20.0}, "risk_budget_inconsistent"),
    ],
)
def test_candidate_rejects_rehashed_cost_sizing_and_risk_forgery(
    changes: dict[str, float], error: str
) -> None:
    plan, contract, _, evidence, result = _replay()
    forged = _rehashed_result(result, **changes)
    # It is internally self-consistent and has a valid content hash; only the
    # separately bound contract can expose the substituted economics.
    forged.validate()

    with pytest.raises(SinglePositionContractError, match=error):
        ScoredCandidate(0.8, plan, contract, evidence, forged)


def test_result_binds_bar_cadence_and_horizon_exactly() -> None:
    _, _, _, _, result = _replay()
    assert result.exit_ts == (
        result.entry_ts + result.bars_held * result.bar_interval_seconds
    )
    assert result.bars_held <= (
        result.max_holding_seconds // result.bar_interval_seconds
    )

    with pytest.raises(SinglePositionContractError, match="exit_ts_inconsistent"):
        replace(result, exit_ts=result.exit_ts + 1.0)
    with pytest.raises(SinglePositionContractError, match="bars_held_exceeds_horizon"):
        replace(result, bars_held=3, exit_ts=result.entry_ts + 900.0)


def test_selector_revalidates_the_whole_candidate_not_only_the_result() -> None:
    plan, contract, _, evidence, result = _replay()
    candidate = ScoredCandidate(0.8, plan, contract, evidence, result)
    object.__setattr__(candidate, "contract", _contract(costs=replace(contract.costs, exit_fee_rate=0.1)))

    with pytest.raises(SinglePositionContractError, match="contract_hash_mismatch"):
        select_single_position([candidate], minimum_score=0.5)


def test_schema_v2_and_missing_binding_objects_have_no_compatibility_bypass() -> None:
    with pytest.raises(SinglePositionContractError, match="unsupported_schema_version"):
        replace(_contract(), schema_version=2)

    plan, contract, _, _, result = _replay()
    with pytest.raises(TypeError):
        # There is deliberately no optional/default contract argument.
        ScoredCandidate(score=0.8, plan=plan, result=result)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        # A contract plus a self-asserted digest is still not replay evidence.
        ScoredCandidate(  # type: ignore[call-arg]
            score=0.8,
            plan=plan,
            contract=contract,
            result=result,
        )
