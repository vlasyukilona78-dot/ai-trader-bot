"""A backtest reports three execution costs, and the middle one decides.

One flat slippage constant answers "is there edge?" with an assumption. Three
bounds answer it with a range, and the rule that a strategy profitable only in
the optimistic bound is rejected removes the most common way a backtest lies.

For a fade entered right after a sharp pump this matters more than usual: the
entry happens when the spread is widest and the book thinnest, which is exactly
where a flat constant is most wrong.
"""

from __future__ import annotations

import unittest

from ai.evidence import EvidenceClass
from backtesting.execution_bounds import (
    BarContext,
    BoundedPnl,
    CostModel,
    ExecutionBound,
    ProfitabilityGate,
    evaluate_gate,
)


def _model(**overrides) -> CostModel:
    base = dict(
        fee_bps_per_side=5.5,
        base_slippage_bps=2.0,
        volatility_coefficient=0.5,
        illiquidity_coefficient=1.0,
    )
    base.update(overrides)
    return CostModel(**base)


def _bar(**overrides) -> BarContext:
    base = dict(range_bps=100.0, volume_ratio=1.0)
    base.update(overrides)
    return BarContext(**base)


class OrderingTests(unittest.TestCase):
    def test_costs_increase_from_optimistic_to_pessimistic(self):
        model, bar = _model(), _bar()

        optimistic = model.cost_bps(ExecutionBound.OPTIMISTIC, bar)
        neutral = model.cost_bps(ExecutionBound.NEUTRAL, bar)
        pessimistic = model.cost_bps(ExecutionBound.PESSIMISTIC, bar)

        self.assertLess(optimistic, neutral)
        self.assertLess(neutral, pessimistic)

    def test_fees_are_charged_on_both_sides_in_every_bound(self):
        model = _model(base_slippage_bps=0.0, volatility_coefficient=0.0, illiquidity_coefficient=0.0)

        for bound in ExecutionBound:
            with self.subTest(bound=bound):
                self.assertGreaterEqual(model.cost_bps(bound, _bar()), 11.0)


class ConditionSensitivityTests(unittest.TestCase):
    def test_a_wider_bar_costs_more(self):
        model = _model()

        calm = model.cost_bps(ExecutionBound.NEUTRAL, _bar(range_bps=50.0))
        violent = model.cost_bps(ExecutionBound.NEUTRAL, _bar(range_bps=400.0))

        self.assertGreater(violent, calm)

    def test_thin_volume_costs_more(self):
        model = _model()

        liquid = model.cost_bps(ExecutionBound.NEUTRAL, _bar(volume_ratio=4.0))
        thin = model.cost_bps(ExecutionBound.NEUTRAL, _bar(volume_ratio=0.25))

        self.assertGreater(thin, liquid)

    def test_a_flat_model_ignores_conditions(self):
        model = _model(volatility_coefficient=0.0, illiquidity_coefficient=0.0)

        calm = model.cost_bps(ExecutionBound.NEUTRAL, _bar(range_bps=50.0))
        violent = model.cost_bps(ExecutionBound.NEUTRAL, _bar(range_bps=900.0))

        self.assertAlmostEqual(calm, violent)

    def test_zero_volume_does_not_produce_an_infinite_cost(self):
        cost = _model().cost_bps(ExecutionBound.PESSIMISTIC, _bar(volume_ratio=0.0))

        self.assertTrue(cost == cost)  # not NaN
        self.assertLess(cost, 1e6)


class GateTests(unittest.TestCase):
    def test_a_positive_neutral_result_passes(self):
        gate = evaluate_gate(BoundedPnl(optimistic=10.0, neutral=4.0, pessimistic=-1.0))

        self.assertIs(gate, ProfitabilityGate.PASS)

    def test_profit_only_in_the_optimistic_bound_is_rejected(self):
        gate = evaluate_gate(BoundedPnl(optimistic=5.0, neutral=-1.0, pessimistic=-8.0))

        self.assertIs(gate, ProfitabilityGate.REJECT_OPTIMISTIC_ONLY)

    def test_a_loss_in_every_bound_is_rejected(self):
        gate = evaluate_gate(BoundedPnl(optimistic=-1.0, neutral=-4.0, pessimistic=-9.0))

        self.assertIs(gate, ProfitabilityGate.REJECT_NON_POSITIVE)

    def test_a_neutral_result_of_exactly_zero_does_not_pass(self):
        gate = evaluate_gate(BoundedPnl(optimistic=3.0, neutral=0.0, pessimistic=-2.0))

        self.assertIs(gate, ProfitabilityGate.REJECT_OPTIMISTIC_ONLY)

    def test_inverted_bounds_are_a_programming_error(self):
        with self.assertRaises(ValueError):
            evaluate_gate(BoundedPnl(optimistic=1.0, neutral=5.0, pessimistic=2.0))


class ProvenanceTests(unittest.TestCase):
    def test_the_model_declares_how_its_numbers_were_established(self):
        model = _model()

        for assumption in model.assumptions():
            with self.subTest(name=assumption.name):
                self.assertIs(assumption.evidence, EvidenceClass.UNVALIDATED)

    def test_an_unvalidated_model_cannot_clear_the_live_gate(self):
        self.assertFalse(_model().live_ready())


if __name__ == "__main__":
    unittest.main()
