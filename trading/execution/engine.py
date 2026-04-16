from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from trading.execution.idempotency import IdempotencyStore
from trading.execution.order_validator import OrderValidationError, validate_order_intent
from trading.exchange.schemas import OpenOrderSnapshot, OrderIntent, OrderResult, OrderSide, PositionSide, PositionSnapshot
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.portfolio.positions import first_effective_position_for_symbol
from trading.risk.engine import RiskDecision
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.state.machine import StateMachine
from trading.state.models import TradeState
from trading.state.persistence import RuntimeStore

if TYPE_CHECKING:
    from trading.exchange.bybit_adapter import BybitAdapter

@dataclass
class ExecutionOutcome:
    accepted: bool
    status: str
    reason: str
    order_id: str = ""
    order_link_id: str = ""
    filled_qty: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    stopped_out: bool = False
    raw: dict | None = None


class ExecutionEngine:
    """Single order placement path with idempotency and lock protection."""

    def __init__(
        self,
        *,
        adapter: "BybitAdapter",
        state_machine: StateMachine,
        hedge_mode: bool,
        stop_loss_required: bool,
        require_reconciliation: bool = True,
        idempotency_ttl_sec: int = 120,
        stop_attach_grace_sec: int = 8,
        stale_open_order_sec: int = 120,
        max_exchange_retries: int = 2,
        external_recovery_grace_sec: int = 45,
        persistence: RuntimeStore | None = None,
    ):
        self.adapter = adapter
        self.state_machine = state_machine
        self.hedge_mode = hedge_mode
        self.stop_loss_required = stop_loss_required
        self.require_reconciliation = require_reconciliation
        self.stop_attach_grace_sec = max(1, int(stop_attach_grace_sec))
        self.stale_open_order_sec = max(10, int(stale_open_order_sec))
        self.max_exchange_retries = max(1, int(max_exchange_retries))
        self.external_recovery_grace_sec = max(10, int(external_recovery_grace_sec))
        self.persistence = persistence
        self._lock = threading.Lock()
        self._idempotency = IdempotencyStore(ttl_sec=idempotency_ttl_sec)
        self._external_recovery_until: dict[str, float] = {}
        if self.persistence is not None:
            self._idempotency.restore(self.persistence.load_live_idempotency_keys())

    @staticmethod
    def _idempotency_key(intent: StrategyIntent) -> str:
        sl = f"{float(intent.stop_loss):.8f}" if intent.stop_loss else "0"
        tp = f"{float(intent.take_profit):.8f}" if intent.take_profit else "0"
        sid = str(intent.metadata.get("legacy_signal_id", "")) if isinstance(intent.metadata, dict) else ""
        return f"{intent.symbol}|{intent.action.value}|{sl}|{tp}|{sid}"

    @staticmethod
    def _norm_symbol(symbol: str) -> str:
        return str(symbol).replace("/", "").upper()

    def _current_position(self, snapshot: ExchangeSnapshot):
        return first_effective_position_for_symbol(snapshot.positions, snapshot.symbol)

    def _fetch_live_position(self, symbol: str) -> PositionSnapshot | None:
        try:
            positions = self.adapter.get_positions(symbol)
        except Exception:
            return None
        return first_effective_position_for_symbol(positions, symbol)

    def _external_recovery_active(self, symbol: str) -> bool:
        deadline = float(self._external_recovery_until.get(self._norm_symbol(symbol), 0.0))
        return deadline > time.time()

    def _remember_external_recovery(self, symbol: str):
        self._external_recovery_until[self._norm_symbol(symbol)] = time.time() + self.external_recovery_grace_sec

    def _clear_external_recovery(self, symbol: str):
        self._external_recovery_until.pop(self._norm_symbol(symbol), None)

    @staticmethod
    def _safe_max_qty(rules) -> float:
        max_qty = float(getattr(rules, "max_qty", 0.0) or 0.0)
        if max_qty <= 0:
            return 0.0
        qty_step = max(float(getattr(rules, "qty_step", 0.0) or 0.0), 0.0)
        step_buffer = qty_step if qty_step > 0 else 0.0
        pct_buffer = max_qty * 0.002
        candidate = max_qty - max(step_buffer, pct_buffer)
        return candidate if candidate > 0 else max_qty

    def _can_auto_remediate_external(self) -> bool:
        config = getattr(self.adapter, "config", None)
        if config is None:
            return False
        if bool(getattr(config, "dry_run", True)):
            return False
        return bool(getattr(config, "demo", False) or getattr(config, "testnet", False))

    def _cancel_unexpected_orders(self, symbol: str, orders: list[OpenOrderSnapshot]) -> bool:
        ok = True
        for order in orders:
            cancelled = self.adapter.cancel_order(
                symbol=symbol,
                order_id=order.order_id,
                order_link_id=order.order_link_id,
            )
            ok = ok and bool(cancelled)
        return ok

    def _auto_close_external_position(self, symbol: str, position: PositionSnapshot) -> OrderResult:
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        return self._place_order_with_retry(
            OrderIntent(
                symbol=symbol,
                side=close_side,
                qty=float(position.qty),
                reduce_only=True,
                position_idx=int(position.position_idx),
                close_on_trigger=True,
            )
        )

    def _collect_external_intervention_issues(
        self,
        *,
        symbol: str,
        snapshot: ExchangeSnapshot,
        rec,
        position: PositionSnapshot | None,
        open_orders: list[OpenOrderSnapshot],
        inflight: list[Any],
    ) -> list[str]:
        suppress_orphan_checks = self._external_recovery_active(symbol)
        issues: list[str] = []

        if position is not None and rec.state == TradeState.FLAT and not inflight and not suppress_orphan_checks:
            issues.append("external_position_without_intent")

        if open_orders and rec.state == TradeState.FLAT and not inflight and not suppress_orphan_checks:
            issues.append("external_open_order_without_intent")

        if open_orders and not inflight and not suppress_orphan_checks:
            unexpected_non_reduce = [o for o in open_orders if not o.reduce_only]
            if unexpected_non_reduce:
                issues.append("external_non_reduce_open_order")

        if rec.state in (TradeState.PENDING_ENTRY_LONG, TradeState.PENDING_ENTRY_SHORT, TradeState.PENDING_EXIT_LONG, TradeState.PENDING_EXIT_SHORT):
            if position is None and not open_orders and not inflight:
                issues.append("stale_pending_without_exchange_truth")

        if position is not None and rec.state in (TradeState.LONG, TradeState.SHORT):
            expected_side = PositionSide.LONG if rec.state == TradeState.LONG else PositionSide.SHORT
            if position.side != expected_side:
                issues.append("state_exchange_side_mismatch")

        if self.stop_loss_required and position is not None and not inflight and not suppress_orphan_checks:
            if position.stop_loss is None or position.stop_loss <= 0:
                issues.append("unprotected_position_without_intent")

        return issues

    def _attempt_auto_remediate_external(
        self,
        *,
        symbol: str,
        rec,
        position: PositionSnapshot | None,
        open_orders: list[OpenOrderSnapshot],
        inflight: list[Any],
        issues: list[str],
    ) -> bool:
        if not issues or not self._can_auto_remediate_external() or inflight:
            return False

        norm_symbol = self._norm_symbol(symbol)
        remediated = False

        if any(
            issue in issues
            for issue in ("external_open_order_without_intent", "external_non_reduce_open_order")
        ) and open_orders:
            if self._cancel_unexpected_orders(norm_symbol, open_orders):
                remediated = True

        if any(
            issue in issues
            for issue in ("external_position_without_intent", "unprotected_position_without_intent")
        ) and position is not None:
            close_result = self._auto_close_external_position(norm_symbol, position)
            if close_result.success:
                remediated = True
                fully_closed = (
                    float(close_result.remaining_qty or 0.0) <= 1e-9
                    and float(close_result.filled_qty or 0.0) + 1e-9 >= float(position.qty)
                )
                if fully_closed:
                    self._clear_external_recovery(norm_symbol)
                    self.state_machine.transition(norm_symbol, TradeState.FLAT, "auto_recovered_external_position")
                else:
                    self._remember_external_recovery(norm_symbol)
                    self.state_machine.transition(norm_symbol, TradeState.RECOVERING, "auto_recovering_external_position")
                return True

        if remediated:
            self._remember_external_recovery(norm_symbol)
            target_state = TradeState.FLAT if position is None else TradeState.RECOVERING
            self.state_machine.transition(norm_symbol, target_state, "auto_recovered_external_orders")
            return True

        return False

    @staticmethod
    def _is_position_protected(position: PositionSnapshot | None, expected_stop: float) -> bool:
        if position is None:
            return False
        if position.stop_loss is None or position.stop_loss <= 0:
            return False
        return abs(float(position.stop_loss) - float(expected_stop)) <= 1e-6 * max(1.0, abs(expected_stop))

    @staticmethod
    def _is_retryable(result: OrderResult) -> bool:
        raw = result.raw if isinstance(result.raw, dict) else {}
        ret_code = raw.get("retCode")
        msg = str(result.error or raw.get("retMsg") or "").lower()
        return ret_code in (10006, 10016, 30084) or "rate" in msg or "timeout" in msg

    def _place_order_with_retry(self, order_intent: OrderIntent) -> OrderResult:
        last = self.adapter.place_market_order(order_intent)
        if last.success:
            return last
        attempts = max(1, self.max_exchange_retries)
        for attempt in range(1, attempts):
            if not self._is_retryable(last):
                return last
            time.sleep(min(0.5 * attempt, 1.5))
            last = self.adapter.place_market_order(order_intent)
            if last.success:
                return last
        return last

    def _set_stop_with_retry(
        self,
        *,
        symbol: str,
        stop_loss: float,
        take_profit: float | None,
        position_idx: int,
        qty: float,
    ):
        last = self.adapter.set_protective_orders(
            symbol=symbol,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_idx=position_idx,
            qty=qty,
        )
        if last.success:
            return last
        attempts = max(1, self.max_exchange_retries)
        for attempt in range(1, attempts):
            raw = last.raw if isinstance(last.raw, dict) else {}
            code = raw.get("retCode")
            msg = str(last.error or raw.get("retMsg") or "").lower()
            retryable = code in (10006, 10016, 30084) or "rate" in msg or "timeout" in msg
            if not retryable:
                return last
            time.sleep(min(0.5 * attempt, 1.5))
            last = self.adapter.set_protective_orders(
                symbol=symbol,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_idx=position_idx,
                qty=qty,
            )
            if last.success:
                return last
        return last

    def _persist_idempotency(self, key: str):
        if self.persistence is None:
            return
        expiry = self._idempotency.get_expiry(key)
        if expiry is None:
            return
        self.persistence.put_idempotency_key(key=key, expires_at=expiry)

    def _persist_intent_status(
        self,
        *,
        intent_key: str,
        symbol: str,
        action: IntentAction,
        payload: dict,
        status: str,
    ):
        if self.persistence is None:
            return
        self.persistence.upsert_inflight_intent(
            intent_key=intent_key,
            symbol=symbol,
            action=action.value,
            payload=payload,
            status=status,
        )

    def _persist_decision(
        self,
        *,
        intent: StrategyIntent,
        state_before: TradeState,
        risk: RiskDecision,
        outcome: ExecutionOutcome,
    ):
        if self.persistence is None:
            return
        self.persistence.append_order_decision(
            symbol=intent.symbol.replace("/", "").upper(),
            action=intent.action.value,
            state_before=state_before.value,
            risk_reason=risk.reason,
            exec_status=outcome.status,
            exec_reason=outcome.reason,
            order_id=outcome.order_id,
            order_link_id=outcome.order_link_id,
            ts=time.time(),
            raw=outcome.raw,
        )

    def _symbol_inflight_entries(self, symbol: str):
        if self.persistence is None:
            return []
        norm = self._norm_symbol(symbol)
        return [e for e in self.persistence.load_open_inflight_intents() if self._norm_symbol(e.symbol) == norm]

    def reset_idempotency_for_validation(self):
        self._idempotency.clear()
        if self.persistence is not None:
            self._idempotency.restore(self.persistence.load_live_idempotency_keys())

    @staticmethod
    def _matching_orders(orders: list[OpenOrderSnapshot], client_order_id: str) -> list[OpenOrderSnapshot]:
        if not client_order_id:
            return []
        return [order for order in orders if order.order_link_id == client_order_id]

    def detect_external_intervention(self, symbol: str, snapshot: ExchangeSnapshot) -> list[str]:
        norm_symbol = self._norm_symbol(symbol)
        rec = self.state_machine.get(norm_symbol)
        position = self._current_position(snapshot)
        open_orders = [o for o in snapshot.open_orders if self._norm_symbol(o.symbol) == norm_symbol]
        inflight = self._symbol_inflight_entries(norm_symbol)

        issues = self._collect_external_intervention_issues(
            symbol=norm_symbol,
            snapshot=snapshot,
            rec=rec,
            position=position,
            open_orders=open_orders,
            inflight=inflight,
        )

        position_issue_names = {
            "external_position_without_intent",
            "unprotected_position_without_intent",
            "state_exchange_side_mismatch",
        }
        if position is not None and any(issue in position_issue_names for issue in issues):
            live_position = self._fetch_live_position(norm_symbol)
            if live_position is None:
                issues = [issue for issue in issues if issue not in position_issue_names]
                position = None
            else:
                position = live_position
                issues = self._collect_external_intervention_issues(
                    symbol=norm_symbol,
                    snapshot=snapshot,
                    rec=rec,
                    position=position,
                    open_orders=open_orders,
                    inflight=inflight,
                )

        if self._attempt_auto_remediate_external(
            symbol=norm_symbol,
            rec=rec,
            position=position,
            open_orders=open_orders,
            inflight=inflight,
            issues=issues,
        ):
            return []

        if issues:
            if "unprotected_position_without_intent" in issues:
                self.state_machine.transition(norm_symbol, TradeState.HALTED, "external_unprotected_position")
            else:
                self.state_machine.transition(norm_symbol, TradeState.RECOVERING, "external_intervention_detected")
        else:
            self._clear_external_recovery(norm_symbol)

        return issues

    def recover_from_restart(self, symbol: str, snapshot: ExchangeSnapshot) -> bool:
        if self.persistence is None:
            return False

        norm_symbol = self._norm_symbol(symbol)
        entries = self._symbol_inflight_entries(norm_symbol)
        if not entries:
            return False

        position = self._current_position(snapshot)
        open_orders = [o for o in snapshot.open_orders if self._norm_symbol(o.symbol) == norm_symbol]
        now_ts = time.time()
        exchange_mutated = False

        for entry in entries:
            payload = dict(entry.payload) if isinstance(entry.payload, dict) else {}
            stop_loss = float(payload.get("stop_loss") or 0.0)
            take_profit = payload.get("take_profit")
            tp_val = float(take_profit) if take_profit not in (None, "") else None
            position_idx = int(payload.get("position_idx", position.position_idx if position else 0))
            client_order_id = str(payload.get("client_order_id") or "")
            matching_orders = self._matching_orders(open_orders, client_order_id)

            if position is None and matching_orders:
                stale_orders: list[OpenOrderSnapshot] = []
                active_orders: list[OpenOrderSnapshot] = []
                for order in matching_orders:
                    order_ts = float(order.updated_ts or order.created_ts or 0.0)
                    if order_ts > 0 and (now_ts - order_ts) > self.stale_open_order_sec:
                        stale_orders.append(order)
                    else:
                        active_orders.append(order)

                cancel_failures = 0
                for order in stale_orders:
                    ok = self.adapter.cancel_order(
                        symbol=norm_symbol,
                        order_id=order.order_id,
                        order_link_id=order.order_link_id,
                    )
                    exchange_mutated = exchange_mutated or bool(ok)
                    if not ok:
                        cancel_failures += 1

                if cancel_failures > 0:
                    payload["cancel_failures"] = int(cancel_failures)
                    self.persistence.update_inflight_status(entry.intent_key, "recover_cancel_failed", payload)
                    self.state_machine.transition(norm_symbol, TradeState.RECOVERING, "restart_cancel_failed")
                    continue

                if active_orders:
                    payload["active_open_orders"] = len(active_orders)
                    payload["stale_orders_cancelled"] = len(stale_orders)
                    target = TradeState.PENDING_ENTRY_LONG if entry.action == IntentAction.LONG_ENTRY.value else TradeState.PENDING_ENTRY_SHORT
                    self.state_machine.transition(norm_symbol, target, "restart_pending_exchange_order")
                    self.persistence.update_inflight_status(entry.intent_key, "pending_submission", payload)
                else:
                    payload["stale_orders_cancelled"] = len(stale_orders)
                    self.persistence.update_inflight_status(entry.intent_key, "stale_order_cancelled", payload)
                    self.state_machine.transition(norm_symbol, TradeState.FLAT, "restart_cancelled_stale_order")
                continue

            if position is None and not matching_orders:
                self.persistence.update_inflight_status(entry.intent_key, "recovered_flat", payload)
                self.state_machine.transition(norm_symbol, TradeState.FLAT, "restart_recovered_flat")
                continue

            if position is None:
                continue

            expected_side = PositionSide.LONG if entry.action == IntentAction.LONG_ENTRY.value else PositionSide.SHORT
            if position.side != expected_side:
                self.persistence.update_inflight_status(entry.intent_key, "side_mismatch", payload)
                self.state_machine.transition(norm_symbol, TradeState.RECOVERING, "restart_side_mismatch")
                continue

            non_reduce_orders = [o for o in matching_orders if not o.reduce_only]
            if non_reduce_orders:
                cancelled = 0
                for order in non_reduce_orders:
                    if self.adapter.cancel_order(
                        symbol=norm_symbol,
                        order_id=order.order_id,
                        order_link_id=order.order_link_id,
                    ):
                        cancelled += 1
                        exchange_mutated = True
                payload["remaining_entry_orders_cancelled"] = int(cancelled)

            target_state = TradeState.LONG if position.side == PositionSide.LONG else TradeState.SHORT
            requested_qty = float(payload.get("requested_qty") or 0.0)
            is_partial_live = requested_qty > 0 and float(position.qty) + 1e-9 < requested_qty

            if self._is_position_protected(position, stop_loss):
                status = "protected_partial" if is_partial_live else "protected"
                reason = "restart_position_protected_partial" if is_partial_live else "restart_position_protected"
                self.persistence.update_inflight_status(entry.intent_key, status, payload)
                self.state_machine.transition(norm_symbol, target_state, reason)
                continue

            grace_deadline_ts = float(payload.get("grace_deadline_ts") or (entry.updated_at + self.stop_attach_grace_sec))
            if now_ts <= grace_deadline_ts and stop_loss > 0:
                stop_res = self._set_stop_with_retry(
                    symbol=norm_symbol,
                    stop_loss=stop_loss,
                    take_profit=tp_val,
                    position_idx=position_idx,
                    qty=float(position.qty),
                )
                exchange_mutated = exchange_mutated or bool(stop_res.success)
                if stop_res.success:
                    status = "protected_partial" if is_partial_live else "protected"
                    reason = "restart_stop_attached_partial" if is_partial_live else "restart_stop_attached"
                    self.persistence.update_inflight_status(entry.intent_key, status, payload)
                    self.state_machine.transition(norm_symbol, target_state, reason)
                    continue

            close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
            close_res = self._place_order_with_retry(
                OrderIntent(
                    symbol=norm_symbol,
                    side=close_side,
                    qty=float(position.qty),
                    reduce_only=True,
                    position_idx=position.position_idx,
                    client_order_id=f"v2-recover-{abs(hash((entry.intent_key, now_ts))) % 10**12}",
                    close_on_trigger=True,
                )
            )
            exchange_mutated = exchange_mutated or bool(close_res.success)
            if close_res.success and close_res.filled_qty >= max(0.0, float(position.qty) * 0.999):
                self.persistence.update_inflight_status(entry.intent_key, "recovered_close", payload)
                self.state_machine.transition(norm_symbol, TradeState.FLAT, "restart_recovered_close")
            else:
                payload["grace_deadline_ts"] = now_ts + self.stop_attach_grace_sec
                self.persistence.update_inflight_status(entry.intent_key, "naked_exposure", payload)
                self.state_machine.transition(norm_symbol, TradeState.HALTED, "restart_unprotected_exposure")

        return exchange_mutated

    def execute(
        self,
        *,
        intent: StrategyIntent,
        risk: RiskDecision,
        snapshot: ExchangeSnapshot,
        mark_price: float,
    ) -> ExecutionOutcome:
        norm_symbol = intent.symbol.replace("/", "").upper()
        state_before = self.state_machine.get(norm_symbol).state

        if intent.action == IntentAction.HOLD:
            outcome = ExecutionOutcome(accepted=False, status="IGNORED", reason="hold")
            self._persist_decision(intent=intent, state_before=state_before, risk=risk, outcome=outcome)
            return outcome

        if self.require_reconciliation and snapshot.symbol.replace("/", "").upper() != norm_symbol:
            outcome = ExecutionOutcome(accepted=False, status="REJECTED", reason="snapshot_symbol_mismatch")
            self._persist_decision(intent=intent, state_before=state_before, risk=risk, outcome=outcome)
            return outcome

        key = self._idempotency_key(intent)
        if not self._idempotency.put_if_absent(key):
            outcome = ExecutionOutcome(accepted=False, status="IGNORED", reason="duplicate_intent")
            self._persist_decision(intent=intent, state_before=state_before, risk=risk, outcome=outcome)
            return outcome
        self._persist_idempotency(key)

        with self._lock:
            outcome = self._execute_locked(
                intent=intent,
                intent_key=key,
                risk=risk,
                snapshot=snapshot,
                mark_price=mark_price,
            )

        self._persist_decision(intent=intent, state_before=state_before, risk=risk, outcome=outcome)
        return outcome

    def _execute_locked(
        self,
        *,
        intent: StrategyIntent,
        intent_key: str,
        risk: RiskDecision,
        snapshot: ExchangeSnapshot,
        mark_price: float,
    ) -> ExecutionOutcome:
        norm_symbol = intent.symbol.replace("/", "").upper()
        state = self.state_machine.get(norm_symbol).state
        position = self._current_position(snapshot)

        if intent.action in (IntentAction.LONG_ENTRY, IntentAction.SHORT_ENTRY):
            if not risk.approved:
                return ExecutionOutcome(accepted=False, status="REJECTED", reason=f"risk:{risk.reason}")
            if state != TradeState.FLAT:
                return ExecutionOutcome(accepted=False, status="REJECTED", reason=f"state:{state.value}")
            if position is not None:
                return ExecutionOutcome(accepted=False, status="REJECTED", reason="position_exists")
            if self.stop_loss_required and (intent.stop_loss is None or intent.stop_loss <= 0):
                return ExecutionOutcome(accepted=False, status="REJECTED", reason="stop_loss_required")

            pos_side = PositionSide.LONG if intent.action == IntentAction.LONG_ENTRY else PositionSide.SHORT
            order_side = OrderSide.BUY if pos_side == PositionSide.LONG else OrderSide.SELL
            position_idx = self.adapter.position_idx_for_side(pos_side, hedge_mode=self.hedge_mode)

            try:
                rules = self.adapter.get_instrument_rules(norm_symbol)
            except Exception as exc:
                return ExecutionOutcome(accepted=False, status="REJECTED", reason=f"instrument_metadata:{exc}")

            qty = self.adapter.round_qty(risk.quantity, rules.qty_step)
            if rules.max_qty > 0:
                max_qty = self.adapter.round_qty(self._safe_max_qty(rules), rules.qty_step)
                qty = min(qty, max_qty if max_qty > 0 else rules.max_qty)
            if qty <= 0:
                return ExecutionOutcome(accepted=False, status="REJECTED", reason="rounded_qty_zero")

            client_order_id = f"v2-{abs(hash((intent_key, qty))) % 10**12}"
            order_intent = OrderIntent(
                symbol=norm_symbol,
                side=order_side,
                qty=qty,
                reduce_only=False,
                position_idx=position_idx,
                client_order_id=client_order_id,
            )

            payload: dict[str, Any] = {
                "stop_loss": float(intent.stop_loss or 0.0),
                "take_profit": float(intent.take_profit) if intent.take_profit is not None else None,
                "position_idx": int(position_idx),
                "requested_qty": float(qty),
                "client_order_id": client_order_id,
            }
            self._persist_intent_status(
                intent_key=intent_key,
                symbol=norm_symbol,
                action=intent.action,
                payload=payload,
                status="pending_submission",
            )

            try:
                validate_order_intent(
                    order_intent,
                    rules=rules,
                    account=snapshot.account,
                    mark_price=mark_price,
                    open_orders=snapshot.open_orders,
                )
            except OrderValidationError as exc:
                self._persist_intent_status(
                    intent_key=intent_key,
                    symbol=norm_symbol,
                    action=intent.action,
                    payload=payload,
                    status="validation_failed",
                )
                return ExecutionOutcome(accepted=False, status="REJECTED", reason=f"order_validation:{exc}")

            pending_state = TradeState.PENDING_ENTRY_LONG if pos_side == PositionSide.LONG else TradeState.PENDING_ENTRY_SHORT
            self.state_machine.transition(norm_symbol, pending_state, "entry_order_submitted")
            result = self._place_order_with_retry(order_intent)
            if not result.success:
                self.state_machine.transition(norm_symbol, TradeState.FLAT, "entry_order_failed")
                self._persist_intent_status(
                    intent_key=intent_key,
                    symbol=norm_symbol,
                    action=intent.action,
                    payload={**payload, "exchange_error": result.error},
                    status="failed_submission",
                )
                return ExecutionOutcome(
                    accepted=False,
                    status="FAILED",
                    reason=f"exchange_order_failed:{result.error}",
                    raw=result.raw,
                )

            filled_qty = max(result.filled_qty, 0.0)
            if filled_qty <= 0:
                self.state_machine.transition(norm_symbol, TradeState.FLAT, "entry_no_fill")
                self._persist_intent_status(
                    intent_key=intent_key,
                    symbol=norm_symbol,
                    action=intent.action,
                    payload={**payload, "exchange_status": result.status},
                    status="no_fill",
                )
                return ExecutionOutcome(accepted=False, status="FAILED", reason="no_fill", raw=result.raw)

            fill_status = "partial_fill" if filled_qty < qty else "pending_fill"
            self._persist_intent_status(
                intent_key=intent_key,
                symbol=norm_symbol,
                action=intent.action,
                payload={**payload, "filled_qty": float(filled_qty)},
                status=fill_status,
            )

            desired_stop = float(intent.stop_loss or 0.0)
            live_position = self._fetch_live_position(norm_symbol)
            already_protected = self._is_position_protected(live_position, desired_stop)

            if self.stop_loss_required and not already_protected:
                stop_res = self._set_stop_with_retry(
                    symbol=norm_symbol,
                    stop_loss=desired_stop,
                    take_profit=float(intent.take_profit) if intent.take_profit is not None else None,
                    position_idx=position_idx,
                    qty=float(filled_qty),
                )
                if not stop_res.success:
                    grace_deadline_ts = time.time() + float(self.stop_attach_grace_sec)
                    guarded_payload = {
                        **payload,
                        "filled_qty": float(filled_qty),
                        "grace_deadline_ts": float(grace_deadline_ts),
                        "stop_error": stop_res.error,
                    }
                    self._persist_intent_status(
                        intent_key=intent_key,
                        symbol=norm_symbol,
                        action=intent.action,
                        payload=guarded_payload,
                        status="naked_exposure",
                    )

                    emergency_side = OrderSide.SELL if pos_side == PositionSide.LONG else OrderSide.BUY
                    close_res = self._place_order_with_retry(
                        OrderIntent(
                            symbol=norm_symbol,
                            side=emergency_side,
                            qty=filled_qty,
                            reduce_only=True,
                            position_idx=position_idx,
                            client_order_id=f"{client_order_id}-slf",
                            close_on_trigger=True,
                        )
                    )
                    if close_res.success and close_res.filled_qty >= filled_qty * 0.999:
                        self.state_machine.transition(norm_symbol, TradeState.FLAT, "stop_attach_failed_emergency_close")
                        self._persist_intent_status(
                            intent_key=intent_key,
                            symbol=norm_symbol,
                            action=intent.action,
                            payload={**guarded_payload, "recovery": "emergency_close"},
                            status="failed_protected_close",
                        )
                        return ExecutionOutcome(
                            accepted=False,
                            status="FAILED",
                            reason="stop_attach_failed_protective_close",
                            order_id=result.order_id,
                            order_link_id=result.order_link_id,
                            filled_qty=filled_qty,
                            avg_price=result.avg_price,
                            raw={"order": result.raw, "stop": stop_res.raw, "recovery": close_res.raw},
                        )

                    self.state_machine.transition(norm_symbol, TradeState.HALTED, "stop_attach_failed_unprotected")
                    return ExecutionOutcome(
                        accepted=False,
                        status="FAILED",
                        reason="stop_attach_failed_unprotected",
                        order_id=result.order_id,
                        order_link_id=result.order_link_id,
                        filled_qty=filled_qty,
                        avg_price=result.avg_price,
                        raw={"order": result.raw, "stop": stop_res.raw, "recovery": close_res.raw},
                    )

            final_state = TradeState.LONG if pos_side == PositionSide.LONG else TradeState.SHORT
            reason = "entry_partial_fill" if filled_qty < qty else "entry_filled"
            self.state_machine.transition(norm_symbol, final_state, reason)
            self._persist_intent_status(
                intent_key=intent_key,
                symbol=norm_symbol,
                action=intent.action,
                payload={**payload, "filled_qty": float(filled_qty)},
                status="completed",
            )
            status = "PARTIAL" if filled_qty < qty else "FILLED"
            return ExecutionOutcome(
                accepted=True,
                status=status,
                reason=reason,
                order_id=result.order_id,
                order_link_id=result.order_link_id,
                filled_qty=filled_qty,
                avg_price=result.avg_price,
                raw=result.raw,
            )

        if intent.action in (IntentAction.EXIT_LONG, IntentAction.EXIT_SHORT):
            if position is None:
                self.state_machine.transition(norm_symbol, TradeState.FLAT, "exit_without_position")
                return ExecutionOutcome(accepted=False, status="IGNORED", reason="no_position")

            if intent.action == IntentAction.EXIT_LONG and position.side != PositionSide.LONG:
                return ExecutionOutcome(accepted=False, status="REJECTED", reason="position_side_mismatch")
            if intent.action == IntentAction.EXIT_SHORT and position.side != PositionSide.SHORT:
                return ExecutionOutcome(accepted=False, status="REJECTED", reason="position_side_mismatch")

            order_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
            pending_state = TradeState.PENDING_EXIT_LONG if position.side == PositionSide.LONG else TradeState.PENDING_EXIT_SHORT
            self.state_machine.transition(norm_symbol, pending_state, "exit_order_submitted")

            exit_key = f"{intent_key}|exit|{position.qty}"
            payload = {
                "position_qty": float(position.qty),
                "position_idx": int(position.position_idx),
                "entry_price": float(position.entry_price),
            }
            self._persist_intent_status(
                intent_key=exit_key,
                symbol=norm_symbol,
                action=intent.action,
                payload=payload,
                status="pending_submission",
            )

            result = self._place_order_with_retry(
                OrderIntent(
                    symbol=norm_symbol,
                    side=order_side,
                    qty=position.qty,
                    reduce_only=True,
                    position_idx=position.position_idx,
                    client_order_id=f"v2-exit-{abs(hash((intent_key, position.qty))) % 10**12}",
                    close_on_trigger=True,
                )
            )
            if not result.success:
                fallback_state = TradeState.LONG if position.side == PositionSide.LONG else TradeState.SHORT
                self.state_machine.transition(norm_symbol, fallback_state, "exit_order_failed")
                self._persist_intent_status(
                    intent_key=exit_key,
                    symbol=norm_symbol,
                    action=intent.action,
                    payload={**payload, "exchange_error": result.error},
                    status="failed_submission",
                )
                return ExecutionOutcome(
                    accepted=False,
                    status="FAILED",
                    reason=f"exchange_order_failed:{result.error}",
                    raw=result.raw,
                )

            filled_qty = max(0.0, float(result.filled_qty))
            if filled_qty <= 0:
                fallback_state = TradeState.LONG if position.side == PositionSide.LONG else TradeState.SHORT
                self.state_machine.transition(norm_symbol, fallback_state, "exit_no_fill")
                self._persist_intent_status(
                    intent_key=exit_key,
                    symbol=norm_symbol,
                    action=intent.action,
                    payload={**payload, "exchange_status": result.status},
                    status="no_fill",
                )
                return ExecutionOutcome(accepted=False, status="FAILED", reason="exit_no_fill", raw=result.raw)

            avg_exit = float(result.avg_price if result.avg_price > 0 else mark_price)
            if position.side == PositionSide.LONG:
                realized_pnl = (avg_exit - float(position.entry_price)) * filled_qty
            else:
                realized_pnl = (float(position.entry_price) - avg_exit) * filled_qty

            stopped_out = False
            if isinstance(intent.metadata, dict):
                stopped_out = str(intent.metadata.get("exit_type", "")).lower() == "stop_loss"
            if not stopped_out and "stop" in intent.reason.lower():
                stopped_out = True

            if filled_qty < float(position.qty):
                fallback_state = TradeState.LONG if position.side == PositionSide.LONG else TradeState.SHORT
                self.state_machine.transition(norm_symbol, fallback_state, "exit_partial_fill")
                self._persist_intent_status(
                    intent_key=exit_key,
                    symbol=norm_symbol,
                    action=intent.action,
                    payload={**payload, "filled_qty": filled_qty},
                    status="partial_fill",
                )
                return ExecutionOutcome(
                    accepted=True,
                    status="PARTIAL",
                    reason="exit_partial_fill",
                    order_id=result.order_id,
                    order_link_id=result.order_link_id,
                    filled_qty=filled_qty,
                    avg_price=avg_exit,
                    realized_pnl=float(realized_pnl),
                    stopped_out=stopped_out,
                    raw=result.raw,
                )

            self.state_machine.transition(norm_symbol, TradeState.FLAT, "exit_filled")
            self._persist_intent_status(
                intent_key=exit_key,
                symbol=norm_symbol,
                action=intent.action,
                payload={**payload, "filled_qty": filled_qty},
                status="completed",
            )
            return ExecutionOutcome(
                accepted=True,
                status="FILLED",
                reason="exit_filled",
                order_id=result.order_id,
                order_link_id=result.order_link_id,
                filled_qty=filled_qty,
                avg_price=avg_exit,
                realized_pnl=float(realized_pnl),
                stopped_out=stopped_out,
                raw=result.raw,
            )

        return ExecutionOutcome(accepted=False, status="REJECTED", reason="unsupported_intent")





















