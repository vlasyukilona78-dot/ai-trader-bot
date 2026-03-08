from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from tests.v2.fakes import FakeAdapter
from trading.execution.engine import ExecutionEngine
from trading.exchange.schemas import OpenOrderSnapshot, OrderSide, PositionSide, PositionSnapshot
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.risk.engine import RiskDecision
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.state.machine import StateMachine
from trading.state.models import TradeState
from trading.state.persistence import RuntimeStore


class RestartRecoveryV2Tests(unittest.TestCase):
    def _snapshot(self, adapter: FakeAdapter, symbol: str) -> ExchangeSnapshot:
        return ExchangeSnapshot(
            symbol=symbol,
            account=adapter.get_account(),
            positions=adapter.get_positions(symbol),
            open_orders=adapter.get_open_orders(symbol),
        )

    def test_state_persisted_across_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)
            sm1 = StateMachine(persistence=store)
            sm1.transition("BTCUSDT", TradeState.PENDING_ENTRY_LONG, "pending")
            store.close()

            store2 = RuntimeStore(db_path)
            sm2 = StateMachine(persistence=store2)
            self.assertEqual(sm2.get("BTCUSDT").state, TradeState.PENDING_ENTRY_LONG)
            store2.close()

    def test_idempotency_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)

            adapter = FakeAdapter()
            sm1 = StateMachine(persistence=store)
            ex1 = ExecutionEngine(
                adapter=adapter,
                state_machine=sm1,
                hedge_mode=False,
                stop_loss_required=True,
                persistence=store,
                idempotency_ttl_sec=3600,
            )
            intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=103.0)
            risk = RiskDecision(approved=True, reason="ok", quantity=1.0)
            first = ex1.execute(intent=intent, risk=risk, snapshot=self._snapshot(adapter, "BTCUSDT"), mark_price=100.0)
            self.assertTrue(first.accepted)

            sm2 = StateMachine(persistence=store)
            ex2 = ExecutionEngine(
                adapter=adapter,
                state_machine=sm2,
                hedge_mode=False,
                stop_loss_required=True,
                persistence=store,
                idempotency_ttl_sec=3600,
            )
            second = ex2.execute(intent=intent, risk=risk, snapshot=self._snapshot(adapter, "BTCUSDT"), mark_price=100.0)
            self.assertFalse(second.accepted)
            self.assertEqual(second.reason, "duplicate_intent")
            store.close()

    def test_restart_pending_submission_keeps_state_with_active_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)
            store.upsert_inflight_intent(
                intent_key="k1",
                symbol="BTCUSDT",
                action="LONG_ENTRY",
                payload={"stop_loss": 99.0, "take_profit": 103.0, "position_idx": 0, "client_order_id": "cid1"},
                status="pending_submission",
            )

            adapter = FakeAdapter()
            adapter.open_orders = [
                OpenOrderSnapshot(
                    symbol="BTCUSDT",
                    order_id="o1",
                    order_link_id="cid1",
                    side=OrderSide.BUY,
                    qty=1.0,
                    reduce_only=False,
                    position_idx=0,
                    status="New",
                    created_ts=time.time(),
                    updated_ts=time.time(),
                )
            ]
            sm = StateMachine(persistence=store)
            ex = ExecutionEngine(adapter=adapter, state_machine=sm, hedge_mode=False, stop_loss_required=True, persistence=store)
            ex.recover_from_restart("BTCUSDT", self._snapshot(adapter, "BTCUSDT"))

            self.assertEqual(sm.get("BTCUSDT").state, TradeState.PENDING_ENTRY_LONG)
            self.assertEqual(len(store.load_open_inflight_intents()), 1)
            store.close()

    def test_restart_cancels_stale_open_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)
            store.upsert_inflight_intent(
                intent_key="k2",
                symbol="BTCUSDT",
                action="LONG_ENTRY",
                payload={"stop_loss": 99.0, "take_profit": 103.0, "position_idx": 0, "client_order_id": "cid2"},
                status="pending_submission",
            )

            adapter = FakeAdapter()
            old_ts = time.time() - 1000
            adapter.open_orders = [
                OpenOrderSnapshot(
                    symbol="BTCUSDT",
                    order_id="o2",
                    order_link_id="cid2",
                    side=OrderSide.BUY,
                    qty=1.0,
                    reduce_only=False,
                    position_idx=0,
                    status="New",
                    created_ts=old_ts,
                    updated_ts=old_ts,
                )
            ]
            sm = StateMachine(persistence=store)
            ex = ExecutionEngine(
                adapter=adapter,
                state_machine=sm,
                hedge_mode=False,
                stop_loss_required=True,
                persistence=store,
                stale_open_order_sec=120,
            )
            ex.recover_from_restart("BTCUSDT", self._snapshot(adapter, "BTCUSDT"))

            self.assertEqual(sm.get("BTCUSDT").state, TradeState.FLAT)
            self.assertGreaterEqual(len(adapter.canceled_orders), 1)
            self.assertEqual(len(store.load_open_inflight_intents()), 0)
            store.close()

    def test_restart_recovery_attaches_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)
            store.upsert_state_record("BTCUSDT", "HALTED", "stop_attach_failed_unprotected", 1.0)
            store.upsert_inflight_intent(
                intent_key="k3",
                symbol="BTCUSDT",
                action="LONG_ENTRY",
                payload={
                    "stop_loss": 99.0,
                    "take_profit": 103.0,
                    "position_idx": 0,
                    "grace_deadline_ts": 9999999999.0,
                    "requested_qty": 1.0,
                    "client_order_id": "cid3",
                },
                status="naked_exposure",
            )

            adapter = FakeAdapter()
            adapter.positions = [
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
            sm = StateMachine(persistence=store)
            ex = ExecutionEngine(adapter=adapter, state_machine=sm, hedge_mode=False, stop_loss_required=True, persistence=store)
            ex.recover_from_restart("BTCUSDT", self._snapshot(adapter, "BTCUSDT"))

            self.assertEqual(sm.get("BTCUSDT").state, TradeState.LONG)
            self.assertGreaterEqual(len(adapter.stop_calls), 1)
            self.assertEqual(len(store.load_open_inflight_intents()), 0)
            store.close()

    def test_restart_partial_fill_stop_attach_failure_halts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)
            store.upsert_inflight_intent(
                intent_key="k4",
                symbol="BTCUSDT",
                action="LONG_ENTRY",
                payload={
                    "stop_loss": 99.0,
                    "take_profit": 103.0,
                    "position_idx": 0,
                    "requested_qty": 1.0,
                    "client_order_id": "cid4",
                    "grace_deadline_ts": time.time() + 5,
                },
                status="partial_fill",
            )

            adapter = FakeAdapter()
            adapter.positions = [
                PositionSnapshot(
                    symbol="BTCUSDT",
                    side=PositionSide.LONG,
                    qty=0.4,
                    entry_price=100.0,
                    liq_price=0.0,
                    leverage=1.0,
                    position_idx=0,
                    stop_loss=None,
                )
            ]
            adapter.open_orders = [
                OpenOrderSnapshot(
                    symbol="BTCUSDT",
                    order_id="o4",
                    order_link_id="cid4",
                    side=OrderSide.BUY,
                    qty=0.6,
                    reduce_only=False,
                    position_idx=0,
                    status="PartiallyFilled",
                    created_ts=time.time(),
                    updated_ts=time.time(),
                )
            ]
            adapter.fail_next_stop = True

            original_place = adapter.place_market_order

            def _fail_reduce_only(intent):
                if intent.reduce_only:
                    from trading.exchange.schemas import OrderResult

                    return OrderResult(
                        success=False,
                        order_id="",
                        order_link_id=intent.client_order_id or "",
                        avg_price=0.0,
                        filled_qty=0.0,
                        remaining_qty=float(intent.qty),
                        status="Rejected",
                        raw={"retCode": 10001, "retMsg": "close_fail"},
                        error="close_fail",
                    )
                return original_place(intent)

            adapter.place_market_order = _fail_reduce_only

            sm = StateMachine(persistence=store)
            ex = ExecutionEngine(adapter=adapter, state_machine=sm, hedge_mode=False, stop_loss_required=True, persistence=store)
            ex.recover_from_restart("BTCUSDT", self._snapshot(adapter, "BTCUSDT"))

            self.assertEqual(sm.get("BTCUSDT").state, TradeState.HALTED)
            self.assertGreaterEqual(len(adapter.canceled_orders), 1)
            store.close()

    def test_restart_recovery_from_manual_exchange_side_intervention(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)
            adapter = FakeAdapter()
            adapter.positions = [
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

            sm = StateMachine(persistence=store)
            sm.transition("BTCUSDT", TradeState.FLAT, "local_flat")
            ex = ExecutionEngine(adapter=adapter, state_machine=sm, hedge_mode=False, stop_loss_required=True, persistence=store)

            issues = ex.detect_external_intervention("BTCUSDT", self._snapshot(adapter, "BTCUSDT"))
            self.assertIn("external_position_without_intent", issues)
            self.assertEqual(sm.get("BTCUSDT").state, TradeState.RECOVERING)
            store.close()


if __name__ == "__main__":
    unittest.main()
