from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from app.testnet_validation import STATUS_FAIL, STATUS_PASS, TestnetValidationHarness
from trading.exchange.schemas import OpenOrderSnapshot, OrderResult, OrderSide, PositionSide, PositionSnapshot
from trading.state.models import TradeState


class _AdapterExistingOrder:
    def __init__(self, order: OpenOrderSnapshot):
        self._order = order
        self.place_calls = 0

    def get_open_orders(self, symbol: str | None = None):
        return [self._order]

    def place_limit_order(self, **kwargs):
        self.place_calls += 1
        return OrderResult(
            success=False,
            order_id="",
            order_link_id=str(kwargs.get("client_order_id") or ""),
            avg_price=0.0,
            filled_qty=0.0,
            remaining_qty=float(kwargs.get("qty") or 0.0),
            status="Rejected",
            raw={"retCode": 10001, "retMsg": "OrderLinkedID is duplicate"},
            error="OrderLinkedID is duplicate",
        )


class _AdapterDuplicateThenVisible:
    def __init__(self, order: OpenOrderSnapshot):
        self._order = order
        self.place_calls = 0
        self._visible = False

    def get_open_orders(self, symbol: str | None = None):
        if self._visible:
            return [self._order]
        return []

    def place_limit_order(self, **kwargs):
        self.place_calls += 1
        self._visible = True
        return OrderResult(
            success=False,
            order_id="",
            order_link_id=str(kwargs.get("client_order_id") or ""),
            avg_price=0.0,
            filled_qty=0.0,
            remaining_qty=float(kwargs.get("qty") or 0.0),
            status="Rejected",
            raw={"retCode": 10001, "retMsg": "OrderLinkedID is duplicate"},
            error="OrderLinkedID is duplicate",
        )


class _StateMachineStub:
    def __init__(self):
        self.state = TradeState.FLAT

    def transition(self, symbol: str, target: TradeState, reason: str):
        self.state = target
        return True

    def get(self, symbol: str):
        return SimpleNamespace(state=self.state, reason="")


class _ExecutionDetectionStub:
    def __init__(self, state_machine: _StateMachineStub, issues: list[str]):
        self._sm = state_machine
        self._issues = list(issues)

    def detect_external_intervention(self, symbol: str, snapshot):
        if self._issues:
            self._sm.transition(symbol, TradeState.HALTED, "external_unprotected_position")
        return list(self._issues)


class TestnetValidationScenariosV2Tests(unittest.TestCase):
    @staticmethod
    def _harness_stub() -> TestnetValidationHarness:
        harness = TestnetValidationHarness.__new__(TestnetValidationHarness)
        harness.symbol = "BTCUSDT"
        harness.run_id = "20260308T000000Z"
        return harness

    @staticmethod
    def _open_position() -> PositionSnapshot:
        return PositionSnapshot(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            qty=0.01,
            entry_price=100.0,
            liq_price=0.0,
            leverage=1.0,
            position_idx=0,
        )

    @staticmethod
    def _zero_placeholder() -> PositionSnapshot:
        return PositionSnapshot(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            qty=0.0,
            entry_price=0.0,
            liq_price=0.0,
            leverage=0.0,
            position_idx=0,
        )

    def test_tiny_lifecycle_fails_if_final_cleanup_not_clean(self):
        harness = self._harness_stub()
        harness.execute_orders = True
        harness._run_tiny_lifecycle = lambda tag: {
            "exit_qty_matches_entry_fill": True,
            "final_cleanup_invariant_ok": False,
            "final_cleanup_invariant_reason": "open_positions_remaining",
            "positions_after_cleanup": 1,
            "open_orders_after_cleanup": 0,
            "post_exit_reconcile": {"flat_confirmed": True, "reason": "flat_clean"},
        }

        status, details = TestnetValidationHarness.scenario_tiny_capped_order_lifecycle(harness)

        self.assertEqual(status, STATUS_FAIL)
        self.assertIn("final_cleanup_not_clean", str(details.get("reason") or ""))

    def test_tiny_lifecycle_fails_if_exit_qty_mismatch(self):
        harness = self._harness_stub()
        harness.execute_orders = True
        harness._run_tiny_lifecycle = lambda tag: {
            "exit_qty_matches_entry_fill": False,
            "final_cleanup_invariant_ok": True,
            "final_cleanup_invariant_reason": "flat_clean",
            "positions_after_cleanup": 0,
            "open_orders_after_cleanup": 0,
            "post_exit_reconcile": {"flat_confirmed": True, "reason": "flat_clean"},
        }

        status, details = TestnetValidationHarness.scenario_tiny_capped_order_lifecycle(harness)

        self.assertEqual(status, STATUS_FAIL)
        self.assertEqual(details.get("reason"), "exit_requested_qty_mismatch_entry_fill")

    def test_tiny_lifecycle_passes_only_when_post_exit_and_cleanup_are_clean(self):
        harness = self._harness_stub()
        harness.execute_orders = True
        harness._run_tiny_lifecycle = lambda tag: {
            "exit_qty_matches_entry_fill": True,
            "final_cleanup_invariant_ok": True,
            "final_cleanup_invariant_reason": "flat_clean",
            "positions_after_cleanup": 0,
            "open_orders_after_cleanup": 0,
            "post_exit_reconcile": {"flat_confirmed": True, "reason": "flat_clean"},
            "final_cleanup": {
                "open_positions_raw_after": 1,
                "effective_open_positions_after": 0,
                "zero_size_placeholder_positions_after": [{"qty": 0.0, "side": "SHORT"}],
            },
        }

        status, _ = TestnetValidationHarness.scenario_tiny_capped_order_lifecycle(harness)

        self.assertEqual(status, STATUS_PASS)

    def test_private_ws_normalization_uses_tiny_lifecycle_path(self):
        harness = self._harness_stub()
        harness.cfg = SimpleNamespace(adapter=SimpleNamespace(ws_enabled=True, ws_private_enabled=True))
        harness.execute_orders = True
        harness._captured_events = [{"event_type": "MARKET"}]
        harness._normalized_ws_events = []

        called = {"tag": ""}

        def _run_tiny(tag: str):
            called["tag"] = tag
            return {
                "entry_setup": {
                    "requested_qty": 0.01,
                    "requested_notional": 1.0,
                    "min_qty": 0.001,
                    "min_notional": 5.0,
                    "qty_step": 0.001,
                    "available_balance_snapshot": 100.0,
                    "equity_snapshot": 100.0,
                    "mark_price_used": 100.0,
                    "affordability_inputs": {"qty": 0.01, "notional": 1.0},
                }
            }

        harness._run_tiny_lifecycle = _run_tiny

        drained = {"calls": 0}

        def _drain(_duration: float):
            drained["calls"] += 1
            if drained["calls"] == 1:
                now = time.time()
                harness._normalized_ws_events.extend(
                    [
                        {"ts": now, "event_type": "HEARTBEAT", "symbol": "BTCUSDT", "payload": {}},
                        {"ts": now + 0.01, "event_type": "ORDER", "symbol": "BTCUSDT", "payload": {"channel": "private"}},
                    ]
                )
            return []

        harness._drain_ws_events = _drain

        status, details = TestnetValidationHarness.scenario_private_ws_normalization(harness)

        self.assertEqual(status, STATUS_PASS)
        self.assertEqual(called["tag"], "ws")
        self.assertTrue(details["has_private_events"])
        self.assertGreaterEqual(int(details["private_event_count"]), 1)
        self.assertEqual(details["event_source_buffer"], "_normalized_ws_events")

    def test_private_ws_normalization_fails_as_setup_failed_before_private_events(self):
        harness = self._harness_stub()
        harness.cfg = SimpleNamespace(adapter=SimpleNamespace(ws_enabled=True, ws_private_enabled=True))
        harness.execute_orders = True
        harness._captured_events = []
        harness._normalized_ws_events = []

        def _run_tiny(_tag: str):
            raise RuntimeError("entry_failed:order_validation:insufficient_available_balance")

        harness._run_tiny_lifecycle = _run_tiny
        harness._tiny_entry_setup_context = lambda force_poll_snapshot: {
            "diagnostics": {
                "requested_qty": 0.01,
                "requested_notional": 1.0,
                "min_qty": 0.001,
                "min_notional": 5.0,
                "qty_step": 0.001,
                "available_balance_snapshot": 0.0,
                "equity_snapshot": 100.0,
                "mark_price_used": 100.0,
                "affordability_inputs": {"qty": 0.01, "notional": 1.0, "available_balance_usdt": 0.0},
            }
        }
        harness._drain_ws_events = lambda _duration: []

        status, details = TestnetValidationHarness.scenario_private_ws_normalization(harness)

        self.assertEqual(status, STATUS_FAIL)
        self.assertTrue(str(details.get("reason") or "").startswith("private_ws_setup_failed:"))
        self.assertEqual(int(details.get("private_event_count", -1)), 0)

    def test_place_limit_with_reconcile_skips_duplicate_submit_if_existing_order_found(self):
        harness = self._harness_stub()
        existing = OpenOrderSnapshot(
            symbol="BTCUSDT",
            order_id="o-existing",
            order_link_id="link-existing",
            side=OrderSide.BUY,
            qty=0.01,
            reduce_only=False,
            position_idx=0,
            status="New",
            created_ts=time.time(),
            updated_ts=time.time(),
        )
        adapter = _AdapterExistingOrder(existing)
        harness.adapter = adapter

        result = TestnetValidationHarness._place_limit_order_with_reconcile(
            harness,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            qty=0.01,
            price=100.0,
            reduce_only=False,
            position_idx=0,
            client_order_id="link-existing",
        )

        self.assertTrue(result.success)
        self.assertEqual(adapter.place_calls, 0)
        self.assertTrue(bool((result.raw or {}).get("recovered_existing_order")))

    def test_place_limit_with_reconcile_handles_duplicate_response_by_reloading_exchange_state(self):
        harness = self._harness_stub()
        existing = OpenOrderSnapshot(
            symbol="BTCUSDT",
            order_id="o-after",
            order_link_id="link-after",
            side=OrderSide.BUY,
            qty=0.02,
            reduce_only=False,
            position_idx=0,
            status="PartiallyFilled",
            created_ts=time.time(),
            updated_ts=time.time(),
        )
        adapter = _AdapterDuplicateThenVisible(existing)
        harness.adapter = adapter

        result = TestnetValidationHarness._place_limit_order_with_reconcile(
            harness,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            qty=0.02,
            price=99.0,
            reduce_only=False,
            position_idx=0,
            client_order_id="link-after",
        )

        self.assertTrue(result.success)
        self.assertEqual(adapter.place_calls, 1)
        self.assertEqual(result.order_id, "o-after")
        self.assertEqual((result.raw or {}).get("source"), "duplicate_reconcile")

    def test_safety_cleanup_reports_settle_status_explicitly(self):
        harness = self._harness_stub()
        harness._cancel_all_orders = lambda: 1
        harness._flatten_symbol = lambda: 1

        snapshots = [
            SimpleNamespace(open_orders=[object()], positions=[self._open_position()]),
            SimpleNamespace(open_orders=[], positions=[]),
        ]
        idx = {"value": 0}

        def _snapshot():
            current = min(idx["value"], len(snapshots) - 1)
            idx["value"] += 1
            return snapshots[current]

        harness._snapshot = _snapshot

        details = TestnetValidationHarness._safety_cleanup(harness, settle_timeout_sec=0.05, settle_poll_sec=0.0)

        self.assertTrue(details["flat_confirmed"])
        self.assertFalse(details["timing_artifact_possible"])
        self.assertEqual(details["consistency_status"], "flat_confirmed")

    def test_safety_cleanup_marks_unsettled_state_as_residual_exposure(self):
        harness = self._harness_stub()
        harness._cancel_all_orders = lambda: 1
        harness._flatten_symbol = lambda: 1

        def _snapshot():
            return SimpleNamespace(open_orders=[], positions=[self._open_position()])

        harness._snapshot = _snapshot

        details = TestnetValidationHarness._safety_cleanup(harness, settle_timeout_sec=0.0, settle_poll_sec=0.0)

        self.assertFalse(details["flat_confirmed"])
        self.assertFalse(details["timing_artifact_possible"])
        self.assertTrue(details["residual_exposure_detected"])
        self.assertEqual(details["consistency_status"], "residual_exposure")

    def test_safety_cleanup_classifies_exchange_reporting_lag(self):
        harness = self._harness_stub()
        harness._cancel_all_orders = lambda: 1
        harness._flatten_symbol = lambda: 1

        snapshots = [
            SimpleNamespace(open_orders=[object()], positions=[self._open_position()]),
            SimpleNamespace(open_orders=[], positions=[]),
        ]
        idx = {"value": 0}

        def _snapshot():
            current = min(idx["value"], len(snapshots) - 1)
            idx["value"] += 1
            return snapshots[current]

        harness._snapshot = _snapshot

        details = TestnetValidationHarness._safety_cleanup(harness, settle_timeout_sec=0.0, settle_poll_sec=0.0)

        self.assertTrue(details["flat_confirmed"])
        self.assertTrue(details["exchange_lag_detected"])
        self.assertEqual(details["consistency_status"], "exchange_reporting_lag")

    def test_safety_cleanup_ignores_zero_size_placeholder_rows(self):
        harness = self._harness_stub()
        harness._cancel_all_orders = lambda: 0
        harness._flatten_symbol = lambda: 0

        def _snapshot():
            return SimpleNamespace(open_orders=[], positions=[self._zero_placeholder()])

        harness._snapshot = _snapshot

        details = TestnetValidationHarness._safety_cleanup(harness, settle_timeout_sec=0.0, settle_poll_sec=0.0)

        self.assertTrue(details["flat_confirmed"])
        self.assertEqual(int(details["open_positions_after"]), 0)
        self.assertEqual(int(details["open_positions_raw_after"]), 1)
        self.assertEqual(int(details["effective_open_positions_after"]), 0)
        self.assertFalse(bool(details["residual_exposure_detected"]))
        self.assertEqual(details["consistency_status"], "flat_confirmed")

    def test_manual_intervention_fails_when_no_effective_external_position(self):
        harness = self._harness_stub()
        harness.execute_orders = True
        harness._safety_cleanup = lambda: {"flat_confirmed": True}
        harness._open_external_manual_exposure = lambda scenario_tag: {
            "manual_order_submitted": True,
            "manual_order_id": "m1",
            "manual_order_link_id": "ml1",
            "manual_fill_qty": 0.001,
            "effective_position_confirmed": False,
            "effective_open_positions_count": 0,
        }

        status, details = TestnetValidationHarness.scenario_restart_after_manual_exchange_intervention(harness)

        self.assertEqual(status, STATUS_FAIL)
        self.assertEqual(details.get("reason"), "manual_intervention_setup_failed:no_effective_external_position")

    def test_manual_intervention_passes_when_effective_external_position_exists(self):
        harness = self._harness_stub()
        harness.execute_orders = True
        harness._safety_cleanup = lambda: {"flat_confirmed": True}
        harness._open_external_manual_exposure = lambda scenario_tag: {
            "manual_order_submitted": True,
            "manual_order_id": "m2",
            "manual_order_link_id": "ml2",
            "manual_fill_qty": 0.001,
            "effective_position_confirmed": True,
            "effective_open_positions_count": 1,
        }
        harness._snapshot = lambda: SimpleNamespace(positions=[self._open_position()], open_orders=[])
        harness.state_machine = _StateMachineStub()
        harness.execution = _ExecutionDetectionStub(harness.state_machine, ["unprotected_position_without_intent"])

        status, details = TestnetValidationHarness.scenario_restart_after_manual_exchange_intervention(harness)

        self.assertEqual(status, STATUS_PASS)
        self.assertEqual(details.get("state_before_cleanup"), TradeState.HALTED.value)
        self.assertGreaterEqual(int(details.get("exchange_effective_open_positions_before_detection", 0)), 1)

    def test_halted_semantics_fails_when_no_effective_external_position(self):
        harness = self._harness_stub()
        harness.execute_orders = True
        harness._safety_cleanup = lambda: {"flat_confirmed": True}
        harness._open_external_manual_exposure = lambda scenario_tag: {
            "manual_order_submitted": True,
            "manual_order_id": "m3",
            "manual_order_link_id": "ml3",
            "manual_fill_qty": 0.001,
            "effective_position_confirmed": False,
            "effective_open_positions_count": 0,
        }

        status, details = TestnetValidationHarness.scenario_halted_semantics(harness)

        self.assertEqual(status, STATUS_FAIL)
        self.assertEqual(details.get("reason"), "halted_semantics_setup_failed:no_effective_external_position")

    def test_halted_semantics_passes_when_effective_external_position_exists(self):
        harness = self._harness_stub()
        harness.execute_orders = True
        harness._safety_cleanup = lambda: {"flat_confirmed": True}
        harness._open_external_manual_exposure = lambda scenario_tag: {
            "manual_order_submitted": True,
            "manual_order_id": "m4",
            "manual_order_link_id": "ml4",
            "manual_fill_qty": 0.001,
            "effective_position_confirmed": True,
            "effective_open_positions_count": 1,
        }
        harness._snapshot = lambda: SimpleNamespace(positions=[self._open_position()], open_orders=[])
        harness.state_machine = _StateMachineStub()
        harness.execution = _ExecutionDetectionStub(harness.state_machine, ["unprotected_position_without_intent"])

        status, details = TestnetValidationHarness.scenario_halted_semantics(harness)

        self.assertEqual(status, STATUS_PASS)
        self.assertEqual(details.get("state_after_detection"), TradeState.HALTED.value)
        self.assertTrue(bool(details.get("issues_detected")))


if __name__ == "__main__":
    unittest.main()
