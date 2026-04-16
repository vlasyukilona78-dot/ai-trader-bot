from __future__ import annotations

import unittest
from types import SimpleNamespace

from tests.v2.fakes import FakeAdapter
from trading.execution.engine import ExecutionEngine
from trading.exchange.schemas import OpenOrderSnapshot, OrderSide, PositionSide, PositionSnapshot
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.risk.engine import RiskDecision
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.state.machine import StateMachine
from trading.state.models import TradeState


class ExecutionEngineV2Tests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAdapter()
        self.sm = StateMachine()
        self.exec = ExecutionEngine(
            adapter=self.adapter,
            state_machine=self.sm,
            hedge_mode=False,
            stop_loss_required=True,
            require_reconciliation=True,
            idempotency_ttl_sec=3600,
            max_exchange_retries=2,
        )

    def _snapshot(self, symbol: str) -> ExchangeSnapshot:
        return ExchangeSnapshot(
            symbol=symbol,
            account=self.adapter.get_account(),
            positions=self.adapter.get_positions(symbol),
            open_orders=self.adapter.get_open_orders(symbol),
        )

    def test_duplicate_signal_protection(self):
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=102.0)
        risk = RiskDecision(approved=True, reason="approved", quantity=1.0)

        first = self.exec.execute(intent=intent, risk=risk, snapshot=self._snapshot("BTCUSDT"), mark_price=100.0)
        second = self.exec.execute(intent=intent, risk=risk, snapshot=self._snapshot("BTCUSDT"), mark_price=100.0)

        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertEqual(second.reason, "duplicate_intent")

    def test_reduce_only_on_exit(self):
        self.adapter.positions = [
            PositionSnapshot(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                qty=1.0,
                entry_price=100.0,
                liq_price=0.0,
                leverage=1.0,
                position_idx=0,
            )
        ]
        self.sm.transition("BTCUSDT", TradeState.LONG, "has_pos")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.EXIT_LONG, reason="exit")

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok"),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )
        self.assertTrue(out.accepted)
        self.assertTrue(self.adapter.placed_orders[-1].reduce_only)

    def test_partial_fill_handling(self):
        self.adapter.partial_fill_qty = 0.4
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=103.0)

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )
        self.assertTrue(out.accepted)
        self.assertEqual(out.status, "PARTIAL")

    def test_partial_fill_attaches_stop_for_filled_qty(self):
        self.adapter.partial_fill_qty = 0.25
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=102.0)
        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )
        self.assertTrue(out.accepted)
        self.assertEqual(out.status, "PARTIAL")
        self.assertGreaterEqual(len(self.adapter.stop_calls), 1)
        self.assertAlmostEqual(float(self.adapter.stop_calls[-1]["qty"]), 0.25, places=6)

    def test_rejection_handling(self):
        self.adapter.fail_next_order = True
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.SHORT_ENTRY, reason="x", stop_loss=101.0, take_profit=97.0)

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )
        self.assertFalse(out.accepted)
        self.assertEqual(out.status, "FAILED")
        self.assertEqual(self.sm.get("BTCUSDT").state, TradeState.FLAT)

    def test_stop_attach_failure_protective_recovery(self):
        self.adapter.fail_next_stop = True
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=102.0)

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )
        self.assertFalse(out.accepted)
        self.assertEqual(out.reason, "stop_attach_failed_protective_close")
        self.assertEqual(self.sm.get("BTCUSDT").state, TradeState.FLAT)
        self.assertTrue(self.adapter.placed_orders[-1].reduce_only)

    def test_stop_attach_failure_unprotected_halts(self):
        original_place = self.adapter.place_market_order

        def _fail_recovery(intent):
            if intent.reduce_only:
                from trading.exchange.schemas import OrderResult

                return OrderResult(
                    success=False,
                    order_id="",
                    order_link_id=intent.client_order_id or "",
                    avg_price=0.0,
                    filled_qty=0.0,
                    status="Rejected",
                    raw={"retCode": 10001, "retMsg": "fail"},
                    error="fail",
                )
            return original_place(intent)

        self.adapter.place_market_order = _fail_recovery
        self.adapter.fail_next_stop = True

        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=102.0)

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )
        self.assertFalse(out.accepted)
        self.assertEqual(out.reason, "stop_attach_failed_unprotected")
        self.assertEqual(self.sm.get("BTCUSDT").state, TradeState.HALTED)

    def test_retry_on_rate_limit(self):
        self.adapter.fail_order_times = 1
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=102.0)

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )
        self.assertTrue(out.accepted)
        self.assertGreaterEqual(len(self.adapter.placed_orders), 2)

    def test_entry_caps_qty_with_exchange_safety_margin_before_validation(self):
        self.adapter.instrument_rules["BTCUSDT"].max_qty = 1000.0
        self.adapter.instrument_rules["BTCUSDT"].qty_step = 1.0
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=102.0)

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1000.0),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )

        self.assertTrue(out.accepted)
        self.assertAlmostEqual(float(self.adapter.placed_orders[-1].qty), 998.0)

    def test_manual_external_position_detection(self):
        self.adapter.positions = [
            PositionSnapshot(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                qty=1.0,
                entry_price=100.0,
                liq_price=0.0,
                leverage=1.0,
                position_idx=0,
                stop_loss=99.0,
            )
        ]
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        issues = self.exec.detect_external_intervention("BTCUSDT", self._snapshot("BTCUSDT"))
        self.assertIn("external_position_without_intent", issues)
        self.assertEqual(self.sm.get("BTCUSDT").state, TradeState.RECOVERING)

    def test_demo_mode_auto_closes_unprotected_external_position(self):
        self.adapter.config = SimpleNamespace(demo=True, testnet=False, dry_run=False)
        self.adapter.positions = [
            PositionSnapshot(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                qty=1.0,
                entry_price=100.0,
                liq_price=0.0,
                leverage=1.0,
                position_idx=0,
                stop_loss=None,
            )
        ]
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")

        issues = self.exec.detect_external_intervention("BTCUSDT", self._snapshot("BTCUSDT"))

        self.assertEqual(issues, [])
        self.assertEqual(self.sm.get("BTCUSDT").state, TradeState.FLAT)
        self.assertEqual(len(self.adapter.positions), 0)
        self.assertTrue(self.adapter.placed_orders[-1].reduce_only)

    def test_zero_size_placeholder_position_does_not_trigger_intervention(self):
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        snapshot = ExchangeSnapshot(
            symbol="BTCUSDT",
            account=self.adapter.get_account(),
            positions=[
                PositionSnapshot(
                    symbol="BTCUSDT",
                    side=PositionSide.SHORT,
                    qty=0.0,
                    entry_price=0.0,
                    liq_price=0.0,
                    leverage=0.0,
                    position_idx=0,
                    stop_loss=None,
                )
            ],
            open_orders=[],
        )

        issues = self.exec.detect_external_intervention("BTCUSDT", snapshot)

        self.assertNotIn("external_position_without_intent", issues)
        self.assertNotIn("unprotected_position_without_intent", issues)
        self.assertEqual(self.sm.get("BTCUSDT").state, TradeState.FLAT)

    def test_external_non_reduce_order_detection(self):
        self.adapter.open_orders = [
            OpenOrderSnapshot(
                symbol="BTCUSDT",
                order_id="o1",
                order_link_id="ext",
                side=OrderSide.BUY,
                qty=1.0,
                reduce_only=False,
                position_idx=0,
                status="New",
            )
        ]
        self.sm.transition("BTCUSDT", TradeState.LONG, "live")
        self.adapter.positions = [
            PositionSnapshot(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                qty=1.0,
                entry_price=100.0,
                liq_price=0.0,
                leverage=1.0,
                position_idx=0,
                stop_loss=99.0,
            )
        ]
        issues = self.exec.detect_external_intervention("BTCUSDT", self._snapshot("BTCUSDT"))
        self.assertIn("external_non_reduce_open_order", issues)
        self.assertEqual(self.sm.get("BTCUSDT").state, TradeState.RECOVERING)

    def test_demo_mode_auto_cancels_external_orders(self):
        self.adapter.config = SimpleNamespace(demo=True, testnet=False, dry_run=False)
        self.adapter.open_orders = [
            OpenOrderSnapshot(
                symbol="BTCUSDT",
                order_id="o1",
                order_link_id="ext",
                side=OrderSide.BUY,
                qty=1.0,
                reduce_only=False,
                position_idx=0,
                status="New",
            )
        ]
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")

        issues = self.exec.detect_external_intervention("BTCUSDT", self._snapshot("BTCUSDT"))

        self.assertEqual(issues, [])
        self.assertEqual(self.sm.get("BTCUSDT").state, TradeState.FLAT)
        self.assertEqual(len(self.adapter.open_orders), 0)
        self.assertGreaterEqual(len(self.adapter.canceled_orders), 1)

    def test_entry_rejected_when_recovering_state(self):
        self.sm.transition("BTCUSDT", TradeState.RECOVERING, "manual_check")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=102.0)
        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )
        self.assertFalse(out.accepted)
        self.assertTrue(out.reason.startswith("state:"))


if __name__ == "__main__":
    unittest.main()
