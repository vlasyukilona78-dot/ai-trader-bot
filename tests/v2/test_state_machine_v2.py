from __future__ import annotations

import unittest

from trading.exchange.schemas import OpenOrderSnapshot, OrderSide, PositionSide, PositionSnapshot
from trading.state.machine import StateMachine
from trading.state.models import TradeState


class StateMachineV2Tests(unittest.TestCase):
    def test_valid_transition(self):
        sm = StateMachine()
        self.assertTrue(sm.transition("BTCUSDT", TradeState.PENDING_ENTRY_LONG, "test"))
        self.assertEqual(sm.get("BTCUSDT").state, TradeState.PENDING_ENTRY_LONG)

    def test_invalid_transition(self):
        sm = StateMachine()
        sm.transition("BTCUSDT", TradeState.PENDING_ENTRY_LONG, "test")
        self.assertFalse(sm.transition("BTCUSDT", TradeState.SHORT, "invalid"))

    def test_reconcile_mismatch_to_exchange_truth(self):
        sm = StateMachine()
        sm.transition("BTCUSDT", TradeState.PENDING_ENTRY_LONG, "local_pending")
        rec = sm.reconcile(
            "BTCUSDT",
            positions=[
                PositionSnapshot(
                    symbol="BTCUSDT",
                    side=PositionSide.SHORT,
                    qty=1.0,
                    entry_price=100.0,
                    liq_price=0.0,
                    leverage=1.0,
                    position_idx=2,
                )
            ],
            open_orders=[],
        )
        self.assertEqual(rec.state, TradeState.SHORT)

    def test_reconcile_pending_order(self):
        sm = StateMachine()
        rec = sm.reconcile(
            "ETHUSDT",
            positions=[],
            open_orders=[
                OpenOrderSnapshot(
                    symbol="ETHUSDT",
                    order_id="1",
                    order_link_id="k1",
                    side=OrderSide.BUY,
                    qty=1.0,
                    reduce_only=False,
                    position_idx=0,
                    status="New",
                )
            ],
        )
        self.assertEqual(rec.state, TradeState.PENDING_ENTRY_LONG)

    def test_reconcile_ignores_zero_size_placeholder_position(self):
        sm = StateMachine()
        sm.transition("BTCUSDT", TradeState.HALTED, "legacy")

        rec = sm.reconcile(
            "BTCUSDT",
            positions=[
                PositionSnapshot(
                    symbol="BTCUSDT",
                    side=PositionSide.SHORT,
                    qty=0.0,
                    entry_price=0.0,
                    liq_price=0.0,
                    leverage=0.0,
                    position_idx=0,
                )
            ],
            open_orders=[],
        )

        self.assertEqual(rec.state, TradeState.FLAT)


if __name__ == "__main__":
    unittest.main()
