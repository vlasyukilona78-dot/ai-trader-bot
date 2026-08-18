"""A lost response must not become a second order.

When ``place_market_order`` times out, the venue may already hold a live order.
The engine has to stop, ask the venue what exists, and act on the answer —
never resend the same command and hope.
"""

from __future__ import annotations

import unittest

from tests.v2.fakes import FakeAdapter
from trading.execution.engine import ExecutionEngine
from trading.risk.engine import RiskDecision
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.state.machine import StateMachine
from trading.state.models import TradeState
from trading.market_data.reconciliation import ExchangeSnapshot


class UnknownOutcomeTests(unittest.TestCase):
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
            max_exchange_retries=3,
        )
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")

    def _snapshot(self) -> ExchangeSnapshot:
        return ExchangeSnapshot(
            symbol="BTCUSDT",
            account=self.adapter.get_account(),
            positions=self.adapter.get_positions("BTCUSDT"),
            open_orders=self.adapter.get_open_orders("BTCUSDT"),
        )

    def _entry(self, snapshot=None):
        intent = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.LONG_ENTRY,
            reason="x",
            stop_loss=99.0,
            take_profit=102.0,
        )
        risk = RiskDecision(approved=True, reason="approved", quantity=1.0)
        return self.exec.execute(
            intent=intent,
            risk=risk,
            snapshot=snapshot if snapshot is not None else self._snapshot(),
            mark_price=100.0,
        )

    def test_timeout_does_not_resend_the_order(self):
        self.adapter.timeout_order_times = 3

        self._entry()

        entries = [o for o in self.adapter.placed_orders if not o.reduce_only]
        self.assertEqual(len(entries), 1, "a lost response must not be resent")

    def test_timeout_queries_the_venue_before_deciding(self):
        self.adapter.timeout_order_times = 3
        snapshot = self._snapshot()
        before = self.adapter.order_query_calls

        self._entry(snapshot)

        self.assertGreater(
            self.adapter.order_query_calls, before, "engine must reconcile after an unknown outcome"
        )

    def test_timeout_is_reported_as_unknown_not_rejected(self):
        self.adapter.timeout_order_times = 3

        outcome = self._entry()

        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.status, "UNKNOWN")
        self.assertIn("reconcil", outcome.reason.lower())

    def test_rate_limit_is_still_retried(self):
        # Regression guard: throttling never reached matching, so resending is
        # still the correct behaviour.
        self.adapter.fail_order_times = 2

        outcome = self._entry()

        entries = [o for o in self.adapter.placed_orders if not o.reduce_only]
        self.assertEqual(len(entries), 3)
        self.assertTrue(outcome.accepted)

    def test_every_resend_reuses_the_same_client_order_id(self):
        self.adapter.fail_order_times = 2

        self._entry()

        entries = [o for o in self.adapter.placed_orders if not o.reduce_only]
        ids = {o.client_order_id for o in entries}
        self.assertEqual(len(ids), 1, "a resend must keep the venue's duplicate guard working")


class ProtectiveOrderPolicyTests(unittest.TestCase):
    """Attaching a stop has the opposite risk profile to placing an entry.

    Re-sending a stop overwrites a position attribute and is harmless. Failing
    to attach one leaves the position naked. So an ambiguous response here is
    retried, unlike an ambiguous entry.
    """

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
            max_exchange_retries=3,
        )

    def test_ambiguous_stop_response_is_retried(self):
        self.adapter.timeout_stop_times = 2

        result = self.exec._set_stop_with_retry(
            symbol="BTCUSDT", stop_loss=99.0, take_profit=102.0, position_idx=0, qty=1.0
        )

        self.assertTrue(result.success)
        self.assertEqual(len(self.adapter.stop_calls), 3)

    def test_terminal_stop_rejection_is_not_retried(self):
        self.adapter.fail_stop_times = 3
        self.adapter.stop_failure_raw = {"retCode": 110043, "retMsg": "set leverage not modified"}

        self.exec._set_stop_with_retry(
            symbol="BTCUSDT", stop_loss=99.0, take_profit=102.0, position_idx=0, qty=1.0
        )

        self.assertEqual(len(self.adapter.stop_calls), 1)


if __name__ == "__main__":
    unittest.main()
