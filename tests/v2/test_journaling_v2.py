from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tests.v2.fakes import FakeAdapter
from trading.execution.engine import ExecutionEngine
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.risk.engine import RiskDecision
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.state.machine import StateMachine
from trading.state.persistence import RuntimeStore


class JournalingV2Tests(unittest.TestCase):
    def _snapshot(self, adapter: FakeAdapter, symbol: str) -> ExchangeSnapshot:
        return ExchangeSnapshot(
            symbol=symbol,
            account=adapter.get_account(),
            positions=adapter.get_positions(symbol),
            open_orders=adapter.get_open_orders(symbol),
        )

    def test_transition_and_order_decision_journal_persisted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)
            adapter = FakeAdapter()
            sm = StateMachine(persistence=store)
            ex = ExecutionEngine(
                adapter=adapter,
                state_machine=sm,
                hedge_mode=False,
                stop_loss_required=True,
                persistence=store,
            )

            intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=102.0)
            out = ex.execute(
                intent=intent,
                risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
                snapshot=self._snapshot(adapter, "BTCUSDT"),
                mark_price=100.0,
            )
            self.assertTrue(out.accepted)
            store.close()

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            transitions = cur.execute("SELECT COUNT(*) FROM state_transitions").fetchone()[0]
            decisions = cur.execute("SELECT COUNT(*) FROM order_decisions").fetchone()[0]
            conn.close()

            self.assertGreaterEqual(transitions, 1)
            self.assertGreaterEqual(decisions, 1)


if __name__ == "__main__":
    unittest.main()
