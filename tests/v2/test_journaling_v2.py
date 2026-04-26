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

    def test_decision_journal_includes_strategy_risk_and_execution_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)
            try:
                adapter = FakeAdapter()
                sm = StateMachine(persistence=store)
                ex = ExecutionEngine(
                    adapter=adapter,
                    state_machine=sm,
                    hedge_mode=False,
                    stop_loss_required=True,
                    persistence=store,
                )

                intent = StrategyIntent(
                    symbol="BTCUSDT",
                    action=IntentAction.LONG_ENTRY,
                    reason="entry_setup",
                    stop_loss=99.0,
                    take_profit=102.0,
                    confidence=0.73,
                    metadata={"entry": 100.0, "note": "test"},
                )
                out = ex.execute(
                    intent=intent,
                    risk=RiskDecision(approved=True, reason="risk_ok", quantity=1.0, notional=100.0),
                    snapshot=self._snapshot(adapter, "BTCUSDT"),
                    mark_price=100.0,
                )
                self.assertTrue(out.accepted)

                decisions = store.load_order_decisions(limit=100)
                self.assertEqual(len(decisions), 1)
                raw = decisions[0].raw
                self.assertEqual(raw["intent_context"]["reason"], "entry_setup")
                self.assertAlmostEqual(float(raw["intent_context"]["confidence"]), 0.73, places=6)
                self.assertEqual(raw["intent_context"]["metadata"]["note"], "test")
                self.assertEqual(raw["risk_context"]["reason"], "risk_ok")
                self.assertAlmostEqual(float(raw["risk_context"]["quantity"]), 1.0, places=6)
                self.assertEqual(raw["execution_context"]["status"], "FILLED")
                self.assertAlmostEqual(float(raw["execution_context"]["avg_price"]), 100.0, places=6)
                self.assertAlmostEqual(float(raw["entry_price"]), 100.0, places=6)
                self.assertAlmostEqual(float(raw["tp_price"]), 102.0, places=6)
                self.assertAlmostEqual(float(raw["sl_price"]), 99.0, places=6)
            finally:
                store.close()

    def test_exit_decision_journal_includes_managed_exit_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)
            try:
                adapter = FakeAdapter()
                sm = StateMachine(persistence=store)
                ex = ExecutionEngine(
                    adapter=adapter,
                    state_machine=sm,
                    hedge_mode=False,
                    stop_loss_required=True,
                    persistence=store,
                )

                entry = StrategyIntent(
                    symbol="BTCUSDT",
                    action=IntentAction.SHORT_ENTRY,
                    reason="entry_setup",
                    stop_loss=101.0,
                    take_profit=96.0,
                )
                entry_outcome = ex.execute(
                    intent=entry,
                    risk=RiskDecision(approved=True, reason="risk_ok", quantity=1.0, notional=100.0),
                    snapshot=self._snapshot(adapter, "BTCUSDT"),
                    mark_price=100.0,
                )
                self.assertTrue(entry_outcome.accepted)

                adapter.mark_price = 98.0
                exit_intent = StrategyIntent(
                    symbol="BTCUSDT",
                    action=IntentAction.EXIT_SHORT,
                    reason="managed_exit_acceptance_reclaim",
                    confidence=0.81,
                    metadata={
                        "managed_exit": True,
                        "exit_type": "acceptance_reclaim",
                        "managed_exit_reason": "managed_exit_acceptance_reclaim",
                        "managed_exit_details": {"reward_r": 0.55},
                    },
                )
                exit_outcome = ex.execute(
                    intent=exit_intent,
                    risk=RiskDecision(approved=True, reason="approved"),
                    snapshot=self._snapshot(adapter, "BTCUSDT"),
                    mark_price=98.0,
                )
                self.assertTrue(exit_outcome.accepted)

                decisions = store.load_order_decisions(limit=100)
                self.assertEqual(len(decisions), 2)
                raw = decisions[-1].raw
                self.assertEqual(raw["intent_context"]["reason"], "managed_exit_acceptance_reclaim")
                self.assertEqual(raw["managed_exit_reason"], "managed_exit_acceptance_reclaim")
                self.assertEqual(raw["exit_type"], "acceptance_reclaim")
                self.assertTrue(bool(raw["managed_exit"]))
                self.assertAlmostEqual(float(raw["exit_price"]), 98.0, places=6)
                self.assertAlmostEqual(float(raw["realized_pnl"]), 2.0, places=6)
                self.assertEqual(raw["execution_context"]["status"], "FILLED")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
