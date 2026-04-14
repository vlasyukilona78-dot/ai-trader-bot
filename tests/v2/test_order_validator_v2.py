from __future__ import annotations

import unittest

from trading.execution.order_validator import OrderValidationError, validate_order_intent
from trading.exchange.schemas import AccountSnapshot, InstrumentRules, OpenOrderSnapshot, OrderIntent, OrderSide


class OrderValidatorV2Tests(unittest.TestCase):
    def setUp(self):
        self.account = AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0)
        self.rules = InstrumentRules(symbol="BTCUSDT", tick_size=0.1, qty_step=0.01, min_qty=0.01, min_notional=5.0)

    def test_reject_qty_step_mismatch(self):
        intent = OrderIntent(symbol="BTCUSDT", side=OrderSide.BUY, qty=0.015)
        with self.assertRaises(OrderValidationError):
            validate_order_intent(intent, rules=self.rules, account=self.account, mark_price=100.0, open_orders=[])

    def test_reject_invalid_metadata(self):
        bad_rules = InstrumentRules(symbol="BTCUSDT", tick_size=0.0, qty_step=0.0, min_qty=0.0, min_notional=0.0)
        intent = OrderIntent(symbol="BTCUSDT", side=OrderSide.BUY, qty=0.02)
        with self.assertRaises(OrderValidationError):
            validate_order_intent(intent, rules=bad_rules, account=self.account, mark_price=100.0, open_orders=[])

    def test_open_order_conflict(self):
        intent = OrderIntent(symbol="BTCUSDT", side=OrderSide.BUY, qty=0.02)
        open_orders = [
            OpenOrderSnapshot(
                symbol="BTCUSDT",
                order_id="1",
                order_link_id="x",
                side=OrderSide.BUY,
                qty=0.02,
                reduce_only=False,
                position_idx=0,
                status="New",
            )
        ]
        with self.assertRaises(OrderValidationError):
            validate_order_intent(intent, rules=self.rules, account=self.account, mark_price=100.0, open_orders=open_orders)

    def test_reject_above_max_qty(self):
        rules = InstrumentRules(
            symbol="BTCUSDT",
            tick_size=0.1,
            qty_step=0.01,
            min_qty=0.01,
            min_notional=5.0,
            max_qty=0.05,
        )
        intent = OrderIntent(symbol="BTCUSDT", side=OrderSide.BUY, qty=0.06)
        with self.assertRaises(OrderValidationError):
            validate_order_intent(intent, rules=rules, account=self.account, mark_price=100.0, open_orders=[])


if __name__ == "__main__":
    unittest.main()
