"""Position size must price the whole loss, not just the stop distance.

Sizing from ``risk_amount / stop_distance`` assumes the stop fills exactly at
its price and costs nothing. A stop actually fills through a spread that is
widest in the conditions that trigger it, can gap past its level, and pays fees
on both legs. A 1% risk budget sized that way loses more than 1%.
"""

from __future__ import annotations

import unittest

from trading.risk.sizing import (
    LossComponents,
    SizingLimit,
    position_size_for_stop,
    size_position,
)


def _components(**overrides) -> LossComponents:
    base = dict(
        entry_price=100.0,
        stop_loss=98.0,
        stop_slippage_bps=0.0,
        gap_buffer_bps=0.0,
        fee_bps_per_side=0.0,
    )
    base.update(overrides)
    return LossComponents(**base)


class LossDecompositionTests(unittest.TestCase):
    def test_with_no_costs_the_loss_is_the_stop_distance(self):
        self.assertAlmostEqual(_components().loss_per_unit(), 2.0)

    def test_stop_slippage_adds_to_the_loss(self):
        # 50 bps of 100.0 is 0.5 on top of the 2.0 stop distance.
        self.assertAlmostEqual(_components(stop_slippage_bps=50.0).loss_per_unit(), 2.5)

    def test_gap_buffer_adds_to_the_loss(self):
        self.assertAlmostEqual(_components(gap_buffer_bps=30.0).loss_per_unit(), 2.3)

    def test_fees_are_charged_on_both_sides(self):
        self.assertAlmostEqual(_components(fee_bps_per_side=10.0).loss_per_unit(), 2.2)

    def test_components_accumulate(self):
        loss = _components(
            stop_slippage_bps=50.0, gap_buffer_bps=30.0, fee_bps_per_side=10.0
        ).loss_per_unit()

        self.assertAlmostEqual(loss, 3.0)

    def test_a_short_stop_above_entry_gives_the_same_distance(self):
        short = _components(entry_price=100.0, stop_loss=102.0)

        self.assertAlmostEqual(short.loss_per_unit(), 2.0)


class SizingTests(unittest.TestCase):
    def test_costs_reduce_the_permitted_size(self):
        cheap = size_position(
            risk_amount=100.0, components=_components(), qty_step=0.0, caps={}
        )
        costly = size_position(
            risk_amount=100.0,
            components=_components(stop_slippage_bps=50.0, fee_bps_per_side=10.0),
            qty_step=0.0,
            caps={},
        )

        self.assertLess(costly.quantity, cheap.quantity)

    def test_size_never_exceeds_the_risk_budget(self):
        components = _components(stop_slippage_bps=50.0, fee_bps_per_side=10.0)

        result = size_position(
            risk_amount=100.0, components=components, qty_step=0.0, caps={}
        )

        self.assertLessEqual(result.quantity * components.loss_per_unit(), 100.0 + 1e-9)

    def test_the_tightest_cap_wins(self):
        result = size_position(
            risk_amount=100.0,
            components=_components(),
            qty_step=0.0,
            caps={SizingLimit.LIQUIDITY: 5.0, SizingLimit.EXPOSURE: 3.0},
        )

        self.assertAlmostEqual(result.quantity, 3.0)
        self.assertIn(SizingLimit.EXPOSURE, result.limiting_factors)

    def test_the_binding_cap_is_reported(self):
        result = size_position(
            risk_amount=100.0,
            components=_components(),
            qty_step=0.0,
            caps={SizingLimit.LIQUIDITY: 999.0},
        )

        self.assertEqual(result.limiting_factors, (SizingLimit.TRADE_LOSS,))

    def test_ties_report_every_binding_cap(self):
        result = size_position(
            risk_amount=100.0,
            components=_components(),
            qty_step=0.0,
            caps={SizingLimit.LIQUIDITY: 50.0, SizingLimit.EXPOSURE: 50.0},
        )

        self.assertEqual(
            set(result.limiting_factors),
            {SizingLimit.TRADE_LOSS, SizingLimit.LIQUIDITY, SizingLimit.EXPOSURE},
        )


class QuantizationTests(unittest.TestCase):
    def test_quantization_rounds_down(self):
        result = size_position(
            risk_amount=100.0, components=_components(), qty_step=0.3, caps={}
        )

        self.assertAlmostEqual(result.quantity, 49.8)

    def test_the_risk_budget_still_holds_after_quantization(self):
        components = _components(stop_slippage_bps=25.0)
        result = size_position(
            risk_amount=100.0, components=components, qty_step=0.7, caps={}
        )

        self.assertLessEqual(result.quantity * components.loss_per_unit(), 100.0 + 1e-9)
        self.assertTrue(result.recheck_passed)

    def test_a_step_larger_than_the_budget_yields_no_position(self):
        result = size_position(
            risk_amount=100.0, components=_components(), qty_step=1000.0, caps={}
        )

        self.assertEqual(result.quantity, 0.0)
        self.assertFalse(result.recheck_passed)


class InvalidInputTests(unittest.TestCase):
    def test_zero_loss_per_unit_yields_no_position(self):
        result = size_position(
            risk_amount=100.0,
            components=_components(entry_price=100.0, stop_loss=100.0),
            qty_step=0.0,
            caps={},
        )

        self.assertEqual(result.quantity, 0.0)

    def test_non_positive_risk_budget_yields_no_position(self):
        result = size_position(
            risk_amount=0.0, components=_components(), qty_step=0.0, caps={}
        )

        self.assertEqual(result.quantity, 0.0)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_the_original_helper_still_works(self):
        qty = position_size_for_stop(
            equity_usdt=10000.0, risk_pct=0.01, entry_price=100.0, stop_loss=98.0
        )

        self.assertAlmostEqual(qty, 50.0)


if __name__ == "__main__":
    unittest.main()
