from __future__ import annotations

import unittest

from trading.exchange.schemas import AccountSnapshot, InstrumentRules
from trading.risk.engine import RiskEngine
from trading.risk.limits import RiskLimits
from trading.signals.signal_types import IntentAction, StrategyIntent


class RiskEngineV2Tests(unittest.TestCase):
    def setUp(self):
        self.limits = RiskLimits(
            max_risk_per_trade_pct=0.01,
            max_daily_loss_pct=0.05,
            max_leverage=2.0,
            max_concurrent_positions=2,
            max_symbol_exposure_pct=0.4,
            max_total_notional_pct=0.8,
            min_liquidation_buffer_pct=0.01,
            require_stop_loss=True,
            pyramiding_enabled=False,
        )
        self.engine = RiskEngine(self.limits)
        self.account = AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0)
        self.rules = InstrumentRules(symbol="BTCUSDT", tick_size=0.1, qty_step=0.001, min_qty=0.001, min_notional=5.0)

    def test_size_calculation_positive(self):
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="t", stop_loss=99.0)
        decision = self.engine.evaluate(intent=intent, account=self.account, existing_positions=[], mark_price=100.0, rules=self.rules)
        self.assertTrue(decision.approved)
        self.assertGreater(decision.quantity, 0.0)

    def test_reject_without_stop(self):
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="t", stop_loss=None)
        decision = self.engine.evaluate(intent=intent, account=self.account, existing_positions=[], mark_price=100.0, rules=self.rules)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "stop_loss_required")

    def test_applies_leverage_limit(self):
        tight = RiskLimits(max_risk_per_trade_pct=0.05, max_leverage=0.2, max_total_notional_pct=1.0, max_symbol_exposure_pct=1.0)
        engine = RiskEngine(tight)
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="t", stop_loss=99.5)
        decision = engine.evaluate(intent=intent, account=self.account, existing_positions=[], mark_price=100.0, rules=self.rules)
        self.assertTrue(decision.approved)
        self.assertLessEqual(decision.implied_leverage, 0.2 + 1e-9)

    def test_reject_liquidation_too_close(self):
        intent = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.LONG_ENTRY,
            reason="t",
            stop_loss=99.0,
            metadata={"liq_price_hint": 99.5},
        )
        decision = self.engine.evaluate(intent=intent, account=self.account, existing_positions=[], mark_price=100.0, rules=self.rules)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "liquidation_too_close")


if __name__ == "__main__":
    unittest.main()

