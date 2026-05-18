from __future__ import annotations

import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from tests.v2.fakes import FakeAdapter
from trading.execution.engine import ExecutionEngine
from trading.exchange.schemas import OpenOrderSnapshot, OrderBookQuality, OrderResult, OrderSide, PositionSide, PositionSnapshot
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.risk.engine import RiskDecision
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.versioning import STRATEGY_RUNTIME_VERSION
from trading.state.machine import StateMachine
from trading.state.models import TradeState
from trading.state.persistence import RuntimeStore


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
        self.adapter.instrument_rules["BTCUSDT"] = replace(
            self.adapter.instrument_rules["BTCUSDT"],
            max_qty=1000.0,
            qty_step=1.0,
        )
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

    def test_exit_ignores_stale_snapshot_when_exchange_position_is_already_flat(self):
        self.sm.transition("BTCUSDT", TradeState.SHORT, "live")
        self.adapter.positions = []
        stale_snapshot = ExchangeSnapshot(
            symbol="BTCUSDT",
            account=self.adapter.get_account(),
            positions=[
                PositionSnapshot(
                    symbol="BTCUSDT",
                    side=PositionSide.SHORT,
                    qty=1.0,
                    entry_price=100.0,
                    liq_price=0.0,
                    leverage=1.0,
                    position_idx=0,
                    stop_loss=101.0,
                )
            ],
            open_orders=[],
        )
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.EXIT_SHORT, reason="managed_exit")

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok"),
            snapshot=stale_snapshot,
            mark_price=100.0,
        )

        self.assertFalse(out.accepted)
        self.assertEqual(out.status, "IGNORED")
        self.assertEqual(out.reason, "no_live_position")
        self.assertEqual(self.sm.get("BTCUSDT").state, TradeState.FLAT)

    def test_retries_after_leverage_alignment_error(self):
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        self.adapter.config = SimpleNamespace(demo=False, testnet=True, dry_run=False, target_entry_leverage=3.0)
        self.adapter.fail_order_results = [
            OrderResult(
                success=False,
                order_id="",
                order_link_id="cid",
                avg_price=0.0,
                filled_qty=0.0,
                status="Rejected",
                raw={"retCode": 110090, "retMsg": "Please adjust your leverage to 6 or below to increase the max. position limit."},
                error="Order placement failed as your position may exceed the max. limit.",
            )
        ]
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=102.0)

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )

        self.assertTrue(out.accepted)
        self.assertEqual(len(self.adapter.ensure_leverage_calls), 1)
        self.assertEqual(self.adapter.ensure_leverage_calls[0]["symbol"], "BTCUSDT")

    def test_retries_after_qty_invalid_with_fresh_rules(self):
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        self.adapter.config = SimpleNamespace(demo=False, testnet=True, dry_run=False, target_entry_leverage=3.0)
        self.adapter.instrument_rules["BTCUSDT"] = self.adapter.instrument_rules["BTCUSDT"].__class__(
            symbol="BTCUSDT",
            tick_size=0.1,
            qty_step=0.1,
            min_qty=0.1,
            min_notional=5.0,
            max_qty=0.0,
        )
        self.adapter.fail_order_results = [
            OrderResult(
                success=False,
                order_id="",
                order_link_id="cid",
                avg_price=0.0,
                filled_qty=0.0,
                status="Rejected",
                raw={"retCode": 10001, "retMsg": "Qty invalid"},
                error="Qty invalid",
            )
        ]
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=102.0)

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=0.30000000000000004),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )

        self.assertTrue(out.accepted)
        self.assertEqual(self.adapter.force_refresh_calls, ["BTCUSDT"])
        self.assertAlmostEqual(float(self.adapter.placed_orders[-1].qty), 0.3, places=6)

    def test_entry_revalidates_after_account_refresh_when_snapshot_balance_is_stale(self):
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        self.adapter.account = self.adapter.account.__class__(equity_usdt=1000.0, available_balance_usdt=1000.0)
        stale_snapshot = ExchangeSnapshot(
            symbol="BTCUSDT",
            account=self.adapter.account.__class__(equity_usdt=1000.0, available_balance_usdt=0.0),
            positions=[],
            open_orders=[],
        )
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.SHORT_ENTRY, reason="x", stop_loss=101.0, take_profit=97.0)

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
            snapshot=stale_snapshot,
            mark_price=100.0,
        )

        self.assertTrue(out.accepted)
        self.assertEqual(out.status, "FILLED")

    def test_entry_rejected_when_orderbook_slippage_is_too_high(self):
        self.adapter.orderbook_quality = OrderBookQuality(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            requested_qty=1.0,
            requested_notional_usdt=100.0,
            executable_qty=2.0,
            executable_notional_usdt=200.0,
            depth_ratio=2.0,
            best_bid=100.0,
            best_ask=100.1,
            spread_bps=10.0,
            expected_avg_price=99.0,
            expected_slippage_bps=80.0,
            levels_used=4,
            available=True,
        )
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.SHORT_ENTRY, reason="x", stop_loss=101.0, take_profit=97.0)

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )

        self.assertFalse(out.accepted)
        self.assertEqual(out.reason, "orderbook_slippage_too_high")
        self.assertEqual(self.adapter.placed_orders, [])

    def test_entry_rejected_when_orderbook_depth_is_too_thin(self):
        self.adapter.orderbook_quality = OrderBookQuality(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            requested_qty=1.0,
            requested_notional_usdt=100.0,
            executable_qty=0.4,
            executable_notional_usdt=40.0,
            depth_ratio=0.4,
            best_bid=99.9,
            best_ask=100.0,
            spread_bps=10.0,
            expected_avg_price=100.1,
            expected_slippage_bps=5.0,
            levels_used=2,
            available=True,
        )
        self.sm.transition("BTCUSDT", TradeState.FLAT, "init")
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0, take_profit=103.0)

        out = self.exec.execute(
            intent=intent,
            risk=RiskDecision(approved=True, reason="ok", quantity=1.0),
            snapshot=self._snapshot("BTCUSDT"),
            mark_price=100.0,
        )

        self.assertFalse(out.accepted)
        self.assertEqual(out.reason, "orderbook_depth_too_thin")
        self.assertEqual(self.adapter.placed_orders, [])

    def test_client_order_id_is_stable_for_same_intent(self):
        first = ExecutionEngine._stable_client_order_id("v2", "intent-key", "BTCUSDT", 1.0)
        second = ExecutionEngine._stable_client_order_id("v2", "intent-key", "BTCUSDT", 1.0)
        other = ExecutionEngine._stable_client_order_id("v2", "intent-key", "ETHUSDT", 1.0)

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertLessEqual(len(first), 36)

    def test_idempotency_key_includes_runtime_setup_signature(self):
        base = StrategyIntent(
            symbol="BTC/USDT",
            action=IntentAction.SHORT_ENTRY,
            reason="layered_short_entry",
            stop_loss=101.0,
            take_profit=97.0,
            metadata={
                "timeframe": "1m",
                "strategy_version": "strategy-a",
                "setup_signature": "pump-a",
            },
        )
        same = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.SHORT_ENTRY,
            reason="layered_short_entry",
            stop_loss=101.0,
            take_profit=97.0,
            metadata={
                "timeframe": "1m",
                "strategy_version": "strategy-a",
                "setup_signature": "pump-a",
            },
        )
        other_timeframe = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.SHORT_ENTRY,
            reason="layered_short_entry",
            stop_loss=101.0,
            take_profit=97.0,
            metadata={
                "timeframe": "5m",
                "strategy_version": "strategy-a",
                "setup_signature": "pump-a",
            },
        )
        other_strategy = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.SHORT_ENTRY,
            reason="layered_short_entry",
            stop_loss=101.0,
            take_profit=97.0,
            metadata={
                "timeframe": "1m",
                "strategy_version": "strategy-b",
                "setup_signature": "pump-a",
            },
        )
        other_setup = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.SHORT_ENTRY,
            reason="layered_short_entry",
            stop_loss=101.0,
            take_profit=97.0,
            metadata={
                "timeframe": "1m",
                "strategy_version": "strategy-a",
                "setup_signature": "pump-b",
            },
        )

        base_key = ExecutionEngine._idempotency_key(base)

        self.assertEqual(base_key, ExecutionEngine._idempotency_key(same))
        self.assertNotEqual(base_key, ExecutionEngine._idempotency_key(other_timeframe))
        self.assertNotEqual(base_key, ExecutionEngine._idempotency_key(other_strategy))
        self.assertNotEqual(base_key, ExecutionEngine._idempotency_key(other_setup))
        self.assertIn("tf=1m", base_key)
        self.assertIn("sv=strategy-a", base_key)

    def test_entry_admission_is_persisted_with_execution_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStore(str(Path(tmpdir) / "runtime.db"))
            try:
                adapter = FakeAdapter()
                sm = StateMachine()
                engine = ExecutionEngine(
                    adapter=adapter,
                    state_machine=sm,
                    hedge_mode=False,
                    stop_loss_required=True,
                    require_reconciliation=True,
                    persistence=store,
                )
                sm.transition("BTCUSDT", TradeState.FLAT, "init")
                intent = StrategyIntent(
                    symbol="BTCUSDT",
                    action=IntentAction.SHORT_ENTRY,
                    reason="layered_short_entry",
                    stop_loss=101.0,
                    take_profit=97.0,
                    confidence=0.81,
                    metadata={
                        "legacy_signal_id": "sig-admit-1",
                        "signal_side": "SHORT",
                        "timeframe": "1m",
                        "strategy_version": STRATEGY_RUNTIME_VERSION,
                        "setup_signature": "btc-pump-rejection",
                        "entry_gate": {
                            "approved": True,
                            "reason": "approved",
                            "score": 0.81,
                        },
                        "admission_status": "approved",
                        "admission_reason": "approved",
                    },
                )

                outcome = engine.execute(
                    intent=intent,
                    risk=RiskDecision(approved=True, reason="approved", quantity=1.0),
                    snapshot=ExchangeSnapshot(
                        symbol="BTCUSDT",
                        account=adapter.get_account(),
                        positions=adapter.get_positions("BTCUSDT"),
                        open_orders=adapter.get_open_orders("BTCUSDT"),
                    ),
                    mark_price=100.0,
                )

                admissions = store.load_signal_admissions(limit=100)
                decisions = store.load_order_decisions(limit=100)
                idempotency_keys = store.load_live_idempotency_keys()

                self.assertTrue(outcome.accepted)
                self.assertEqual(outcome.status, "FILLED")
                self.assertEqual(len(admissions), 1)
                self.assertEqual(admissions[0].signal_id, "sig-admit-1")
                self.assertEqual(admissions[0].symbol, "BTCUSDT")
                self.assertEqual(admissions[0].action, "SHORT_ENTRY")
                self.assertTrue(admissions[0].approved)
                self.assertEqual(admissions[0].raw["admission_status"], "approved")
                self.assertEqual(admissions[0].raw["execution_status"], "FILLED")
                self.assertEqual(len(decisions), 1)
                self.assertEqual(decisions[0].raw["intent_context"]["metadata"]["admission_status"], "approved")
                self.assertTrue(any("tf=1m" in key and f"sv={STRATEGY_RUNTIME_VERSION}" in key for key in idempotency_keys))
            finally:
                store.close()

    def test_position_protected_allows_small_bps_drift(self):
        position = PositionSnapshot(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            qty=1.0,
            entry_price=100.0,
            liq_price=0.0,
            leverage=1.0,
            position_idx=0,
            stop_loss=101.0004,
        )

        self.assertTrue(ExecutionEngine._is_position_protected(position, 101.0, tolerance_bps=1.0))
        self.assertFalse(ExecutionEngine._is_position_protected(position, 101.0, tolerance_bps=0.001))


if __name__ == "__main__":
    unittest.main()
