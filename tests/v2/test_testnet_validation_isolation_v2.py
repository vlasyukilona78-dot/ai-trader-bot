from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from app.testnet_validation import TestnetValidationHarness
from trading.exchange.schemas import AccountSnapshot, PositionSide, PositionSnapshot
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.state.machine import StateMachine
from trading.state.models import TradeState
from trading.state.persistence import RuntimeStore


class _ExecutionStub:
    def __init__(self, issues: list[str] | None = None):
        self.issues = list(issues or [])
        self.reset_calls = 0

    def detect_external_intervention(self, symbol: str, snapshot: ExchangeSnapshot) -> list[str]:
        return list(self.issues)

    def reset_idempotency_for_validation(self):
        self.reset_calls += 1


class TestnetValidationIsolationV2Tests(unittest.TestCase):
    @staticmethod
    def _harness_stub(
        *,
        store: RuntimeStore,
        state_machine: StateMachine,
        execution: _ExecutionStub,
        snapshot: ExchangeSnapshot,
    ) -> TestnetValidationHarness:
        harness = TestnetValidationHarness.__new__(TestnetValidationHarness)
        harness.symbol = "BTCUSDT"
        harness.runtime_store = store
        harness.state_machine = state_machine
        harness.execution = execution
        harness._captured_events = [{"stale": True}]
        harness._normalized_ws_events = []
        harness._snapshot = lambda: snapshot
        return harness

    def test_validation_bootstrap_reset_halted_to_flat_when_exchange_flat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStore(str(Path(tmpdir) / "runtime.db"))
            sm = StateMachine(persistence=store)
            sm.transition("BTCUSDT", TradeState.HALTED, "persisted_halted")
            store.upsert_inflight_intent(
                intent_key="inflight-1",
                symbol="BTCUSDT",
                action="LONG_ENTRY",
                payload={"x": 1},
                status="pending_submission",
            )
            store.put_idempotency_key("idem-1", time.time() + 3600)

            execution = _ExecutionStub([])
            snapshot = ExchangeSnapshot(
                symbol="BTCUSDT",
                account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
                positions=[],
                open_orders=[],
            )
            harness = self._harness_stub(store=store, state_machine=sm, execution=execution, snapshot=snapshot)

            try:
                before = TestnetValidationHarness._collect_scenario_start_state(harness, "validation_bootstrap")
                details = TestnetValidationHarness._validation_safe_reset_if_flat(harness, "validation_bootstrap", before)

                self.assertTrue(details["applied"])
                self.assertEqual(sm.get("BTCUSDT").state, TradeState.FLAT)
                self.assertEqual(len(store.load_open_inflight_intents()), 0)
                self.assertEqual(len(store.load_live_idempotency_keys()), 0)
                self.assertGreaterEqual(execution.reset_calls, 1)
            finally:
                store.close()

    def test_prepare_executable_scenario_isolates_successive_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStore(str(Path(tmpdir) / "runtime.db"))
            sm = StateMachine(persistence=store)
            execution = _ExecutionStub([])
            snapshot = ExchangeSnapshot(
                symbol="BTCUSDT",
                account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
                positions=[],
                open_orders=[],
            )
            harness = self._harness_stub(store=store, state_machine=sm, execution=execution, snapshot=snapshot)

            store.upsert_inflight_intent(
                intent_key="inflight-a",
                symbol="BTCUSDT",
                action="LONG_ENTRY",
                payload={"x": "a"},
                status="pending_submission",
            )
            store.put_idempotency_key("idem-a", time.time() + 3600)
            ok1, details1 = TestnetValidationHarness._prepare_executable_scenario(harness, "tiny_capped_order_lifecycle")

            self.assertTrue(ok1)
            self.assertEqual(details1["scenario_start"]["local_state"], TradeState.FLAT.value)
            self.assertGreaterEqual(int(details1["scenario_isolation"]["cleared_inflight"]), 1)
            self.assertGreaterEqual(int(details1["scenario_isolation"]["cleared_idempotency"]), 1)
            self.assertEqual(harness._captured_events, [])

            harness._captured_events = [{"stale": True}]
            store.upsert_inflight_intent(
                intent_key="inflight-b",
                symbol="BTCUSDT",
                action="LONG_ENTRY",
                payload={"x": "b"},
                status="pending_submission",
            )
            store.put_idempotency_key("idem-b", time.time() + 3600)
            ok2, details2 = TestnetValidationHarness._prepare_executable_scenario(harness, "private_ws_normalization")

            self.assertTrue(ok2)
            self.assertEqual(details2["scenario_start"]["local_state"], TradeState.FLAT.value)
            self.assertGreaterEqual(int(details2["scenario_isolation"]["cleared_inflight"]), 1)
            self.assertGreaterEqual(int(details2["scenario_isolation"]["cleared_idempotency"]), 1)
            self.assertEqual(execution.reset_calls, 2)
            store.close()

    def test_prepare_no_duplicate_execution_scenario_starts_clean(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStore(str(Path(tmpdir) / "runtime.db"))
            sm = StateMachine(persistence=store)
            execution = _ExecutionStub([])
            snapshot = ExchangeSnapshot(
                symbol="BTCUSDT",
                account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
                positions=[],
                open_orders=[],
            )
            harness = self._harness_stub(store=store, state_machine=sm, execution=execution, snapshot=snapshot)

            ok, details = TestnetValidationHarness._prepare_executable_scenario(
                harness,
                "no_duplicate_execution_reconnect_retry",
            )

            self.assertTrue(ok)
            self.assertEqual(details["scenario_start"]["local_state"], TradeState.FLAT.value)
            self.assertEqual(int(details["scenario_start"]["exchange_positions_count"]), 0)
            self.assertEqual(int(details["scenario_start"]["exchange_open_orders_count"]), 0)
            self.assertEqual(details["starting_state"], TradeState.FLAT.value)
            store.close()

    def test_prepare_executable_scenario_fails_on_non_flat_start_with_details(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStore(str(Path(tmpdir) / "runtime.db"))
            sm = StateMachine(persistence=store)
            execution = _ExecutionStub([])
            snapshot = ExchangeSnapshot(
                symbol="BTCUSDT",
                account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
                positions=[
                    PositionSnapshot(
                        symbol="BTCUSDT",
                        side=PositionSide.LONG,
                        qty=0.01,
                        entry_price=100.0,
                        liq_price=0.0,
                        leverage=1.0,
                        position_idx=0,
                        stop_loss=99.0,
                    )
                ],
                open_orders=[],
            )
            harness = self._harness_stub(store=store, state_machine=sm, execution=execution, snapshot=snapshot)

            ok, details = TestnetValidationHarness._prepare_executable_scenario(
                harness,
                "tiny_capped_order_lifecycle",
            )

            self.assertFalse(ok)
            self.assertEqual(details["reason"], "scenario_start_not_clean")
            self.assertGreaterEqual(int(details["exchange_positions_before"]), 1)
            self.assertIn("persistence_snapshot", details)
            store.close()

    def test_prepare_executable_scenario_treats_zero_size_placeholder_as_flat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStore(str(Path(tmpdir) / "runtime.db"))
            sm = StateMachine(persistence=store)
            execution = _ExecutionStub([])
            snapshot = ExchangeSnapshot(
                symbol="BTCUSDT",
                account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
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
            harness = self._harness_stub(store=store, state_machine=sm, execution=execution, snapshot=snapshot)

            ok, details = TestnetValidationHarness._prepare_executable_scenario(
                harness,
                "tiny_capped_order_lifecycle",
            )

            self.assertTrue(ok)
            self.assertEqual(int(details["exchange_positions_before"]), 0)
            self.assertEqual(int(details["scenario_start"].get("exchange_positions_raw_count", 0)), 1)
            self.assertEqual(int(details["scenario_start"].get("exchange_effective_open_positions_count", 0)), 0)
            store.close()


if __name__ == "__main__":
    unittest.main()


