from __future__ import annotations

import unittest

from tests.v2.fakes import FakeAdapter
from trading.execution.engine import ExecutionEngine
from trading.market_data.reconciliation import ExchangeReconciler
from trading.risk.engine import RiskEngine
from trading.risk.limits import RiskLimits
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.state.machine import StateMachine
from trading.state.models import TradeState


class EndToEndDryRunV2Tests(unittest.TestCase):
    def test_flat_entry_exit_and_mismatch_recovery(self):
        adapter = FakeAdapter()
        reconciler = ExchangeReconciler(adapter)
        sm = StateMachine()
        risk = RiskEngine(RiskLimits(max_total_notional_pct=2.0, max_symbol_exposure_pct=2.0, max_leverage=5.0))
        execution = ExecutionEngine(
            adapter=adapter,
            state_machine=sm,
            hedge_mode=False,
            stop_loss_required=True,
            require_reconciliation=True,
        )

        snap0 = reconciler.snapshot("BTCUSDT")
        rec0 = sm.reconcile("BTCUSDT", snap0.positions, snap0.open_orders)
        self.assertEqual(rec0.state, TradeState.FLAT)

        entry_intent = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.LONG_ENTRY,
            reason="test_entry",
            stop_loss=99.0,
            take_profit=103.0,
        )
        decision = risk.evaluate(
            intent=entry_intent,
            account=snap0.account,
            existing_positions=snap0.positions,
            mark_price=100.0,
            rules=adapter.get_instrument_rules("BTCUSDT"),
        )
        out_entry = execution.execute(intent=entry_intent, risk=decision, snapshot=snap0, mark_price=100.0)
        self.assertTrue(out_entry.accepted)
        self.assertEqual(sm.get("BTCUSDT").state, TradeState.LONG)

        snap1 = reconciler.snapshot("BTCUSDT")
        rec1 = sm.reconcile("BTCUSDT", snap1.positions, snap1.open_orders)
        self.assertEqual(rec1.state, TradeState.LONG)

        exit_intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.EXIT_LONG, reason="test_exit")
        out_exit = execution.execute(
            intent=exit_intent,
            risk=risk.evaluate(
                intent=exit_intent,
                account=snap1.account,
                existing_positions=snap1.positions,
                mark_price=100.0,
                rules=adapter.get_instrument_rules("BTCUSDT"),
            ),
            snapshot=snap1,
            mark_price=100.0,
        )
        self.assertTrue(out_exit.accepted)
        self.assertEqual(sm.get("BTCUSDT").state, TradeState.FLAT)

        # Force local mismatch and verify reconciliation uses exchange truth.
        sm.transition("BTCUSDT", TradeState.LONG, "forced_mismatch")
        snap2 = reconciler.snapshot("BTCUSDT")
        rec2 = sm.reconcile("BTCUSDT", snap2.positions, snap2.open_orders)
        self.assertEqual(rec2.state, TradeState.FLAT)


if __name__ == "__main__":
    unittest.main()

