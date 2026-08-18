"""The risk engine must size against the priced loss, not the stop distance.

A fade entry is stopped out precisely when the spread is widest, so the cost of
exiting is part of the risk being budgeted. If the engine ignores it, the
configured risk-per-trade is spent before the stop is even reached.
"""

from __future__ import annotations

import unittest

from tests.v2.fakes import FakeAdapter
from trading.risk.engine import RiskEngine
from trading.risk.limits import RiskLimits
from trading.signals.signal_types import IntentAction, StrategyIntent


class SizingWithCostsTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAdapter()
        self.intent = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.SHORT_ENTRY,
            reason="fade",
            stop_loss=102.0,
            take_profit=94.0,
        )

    def _decide(self, **limit_overrides):
        # Exposure ceilings are relaxed so the risk budget is the binding cap;
        # otherwise max_symbol_exposure_pct clamps both cases to the same size
        # and the comparison proves nothing.
        base = dict(max_symbol_exposure_pct=1.0, max_total_notional_pct=1.0)
        base.update(limit_overrides)
        limits = RiskLimits(**base)
        engine = RiskEngine(limits)
        return engine.evaluate(
            intent=self.intent,
            account=self.adapter.get_account(),
            existing_positions=[],
            mark_price=100.0,
            rules=self.adapter.rules,
        )

    def test_costs_shrink_the_approved_quantity(self):
        free = self._decide(stop_slippage_bps=0.0, gap_buffer_bps=0.0, fee_bps_per_side=0.0)
        priced = self._decide(stop_slippage_bps=40.0, gap_buffer_bps=20.0, fee_bps_per_side=5.5)

        self.assertTrue(free.approved)
        self.assertTrue(priced.approved)
        self.assertLess(priced.quantity, free.quantity)

    def test_approved_quantity_respects_the_configured_risk_budget(self):
        limits = RiskLimits(
            max_risk_per_trade_pct=0.01,
            max_symbol_exposure_pct=1.0,
            max_total_notional_pct=1.0,
            stop_slippage_bps=40.0,
            gap_buffer_bps=20.0,
            fee_bps_per_side=5.5,
        )
        engine = RiskEngine(limits)
        account = self.adapter.get_account()

        decision = engine.evaluate(
            intent=self.intent,
            account=account,
            existing_positions=[],
            mark_price=100.0,
            rules=self.adapter.rules,
        )

        # loss/unit = |100-102| + 100*(40+20)/1e4 + 100*5.5*2/1e4 = 2 + 0.6 + 0.11
        loss_per_unit = 2.0 + 0.6 + 0.11
        budget = account.equity_usdt * limits.max_risk_per_trade_pct
        self.assertLessEqual(decision.quantity * loss_per_unit, budget + 1e-6)

    def test_defaults_are_conservative_rather_than_free(self):
        limits = RiskLimits()

        self.assertGreater(limits.stop_slippage_bps, 0.0)
        self.assertGreater(limits.fee_bps_per_side, 0.0)


if __name__ == "__main__":
    unittest.main()
