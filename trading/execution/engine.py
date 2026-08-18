from __future__ import annotations

import json
import logging
import hashlib
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from trading.execution.failure_class import classify_failure
from trading.execution.idempotency import IdempotencyStore
from trading.execution.order_validator import OrderValidationError, validate_order_intent
from trading.exchange.schemas import OpenOrderSnapshot, OrderBookQuality, OrderIntent, OrderResult, OrderSide, PositionSide, PositionSnapshot
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.portfolio.positions import first_effective_position_for_symbol
from trading.risk.engine import RiskDecision
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.versioning import STRATEGY_RUNTIME_VERSION
from trading.state.machine import StateMachine
from trading.state.models import TradeState
from trading.state.persistence import RuntimeStore

if TYPE_CHECKING:
    from trading.exchange.bybit_adapter import BybitAdapter

logger = logging.getLogger(__name__)

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
        entry_orderbook_guard_enabled: bool = True,
        entry_orderbook_guard_require_live: bool = False,
        entry_orderbook_limit: int = 50,
        entry_orderbook_depth_slippage_bps: float = 35.0,
        max_entry_orderbook_slippage_bps: float = 45.0,
        min_entry_orderbook_depth_ratio: float = 1.15,
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
        self.entry_orderbook_guard_enabled = bool(entry_orderbook_guard_enabled)
        self.entry_orderbook_guard_require_live = bool(entry_orderbook_guard_require_live)
        self.entry_orderbook_limit = max(1, int(entry_orderbook_limit))
        self.entry_orderbook_depth_slippage_bps = max(0.0, float(entry_orderbook_depth_slippage_bps))
        self.max_entry_orderbook_slippage_bps = max(0.0, float(max_entry_orderbook_slippage_bps))
        self.min_entry_orderbook_depth_ratio = max(0.0, float(min_entry_orderbook_depth_ratio))
        self.persistence = persistence
        self._lock = threading.Lock()
        self._idempotency = IdempotencyStore(ttl_sec=idempotency_ttl_sec)
        self._external_recovery_until: dict[str, float] = {}
        if self.persistence is not None:
            self._idempotency.restore(self.persistence.load_live_idempotency_keys())

    @staticmethod
    def _format_key_price(value: float | None) -> str:
        try:
            numeric = float(value or 0.0)
        except (TypeError, ValueError):
            numeric = 0.0
        return f"{numeric:.8f}" if numeric else "0"

    @staticmethod
    def _first_metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = metadata.get(key)
            if value not in (None, ""):
                return value
        return ""

    @classmethod
    def _strategy_version_for_idempotency(cls, metadata: dict[str, Any]) -> str:
        explicit = cls._first_metadata_value(metadata, "strategy_version", "strategy_runtime_version", "strategy_runtime")
        if explicit:
            return str(explicit)
        runtime_versions = metadata.get("runtime_versions")
        if isinstance(runtime_versions, dict):
            runtime_version = runtime_versions.get("strategy_runtime")
            if runtime_version not in (None, ""):
                return str(runtime_version)
        return STRATEGY_RUNTIME_VERSION

    @classmethod
    def _setup_signature_for_idempotency(cls, intent: StrategyIntent, metadata: dict[str, Any]) -> str:
        explicit = cls._first_metadata_value(
            metadata,
            "setup_signature",
            "signal_signature",
            "candidate_signature",
            "legacy_signal_id",
            "signal_id",
        )
        if explicit:
            payload = str(explicit)
        else:
            gate = metadata.get("entry_gate") if isinstance(metadata.get("entry_gate"), dict) else {}
            payload_obj = {
                "symbol": str(intent.symbol).replace("/", "").upper(),
                "action": intent.action.value,
                "reason": str(intent.reason),
                "side": str(metadata.get("signal_side") or intent.action.value),
                "entry": cls._first_metadata_value(metadata, "entry_price", "entry", "entry_px", "mark_price", "price"),
                "admission_reason": str(metadata.get("admission_reason") or gate.get("reason") or ""),
                "gate_score": gate.get("score"),
                "confidence": round(float(intent.confidence or 0.0), 6),
            }
            payload = json.dumps(
                cls._json_safe(payload_obj),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _idempotency_key(cls, intent: StrategyIntent) -> str:
        metadata = intent.metadata if isinstance(intent.metadata, dict) else {}
        sl = cls._format_key_price(intent.stop_loss)
        tp = cls._format_key_price(intent.take_profit)
        timeframe = str(
            cls._first_metadata_value(
                metadata,
                "timeframe",
                "tf",
                "signal_timeframe",
                "entry_timeframe",
                "observation_timeframe",
            )
            or "na"
        )
        strategy_version = cls._strategy_version_for_idempotency(metadata)
        setup_signature = cls._setup_signature_for_idempotency(intent, metadata)
        symbol = str(intent.symbol).replace("/", "").upper()
        return (
            f"{symbol}|{intent.action.value}|tf={timeframe}|sv={strategy_version}|"
            f"sl={sl}|tp={tp}|setup={setup_signature}"
        )

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
    def _stable_client_order_id(prefix: str, *parts: Any, digest_len: int = 16) -> str:
        payload = "|".join(str(part) for part in parts)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[: max(8, int(digest_len))]
        return f"{prefix}-{digest}"[:36]

    @staticmethod
    def _is_position_protected(
        position: PositionSnapshot | None,
        expected_stop: float,
        *,
        tolerance_bps: float = 5.0,
        tick_size: float | None = None,
    ) -> bool:
        if position is None:
            return False
        if position.stop_loss is None or position.stop_loss <= 0:
            return False
        expected = float(expected_stop)
        if expected <= 0:
            return False
        bps_tolerance = abs(expected) * max(0.0, float(tolerance_bps)) / 10000.0
        tick_tolerance = abs(float(tick_size or 0.0)) * 2.0
        min_tolerance = 1e-8 * max(1.0, abs(expected))
        tolerance = max(min_tolerance, bps_tolerance, tick_tolerance)
        return abs(float(position.stop_loss) - expected) <= tolerance

    @staticmethod
    def _is_retryable(result: OrderResult) -> bool:
        return classify_failure(result).permits_resend()

    def _venue_holds_order(self, symbol: str, client_order_id: str) -> bool:
        """Ask the venue whether an order with this identity already exists."""

        if not client_order_id:
            return False
        try:
            orders = self.adapter.get_open_orders(symbol)
        except Exception:
            # The venue could not be reached, so its state stays unknown.
            # Report "not confirmed" rather than "absent"; never resend on this.
            return False
        return bool(self._matching_orders(list(orders), client_order_id))

    def _reconcile_unknown_send(self, order_intent: OrderIntent, last: OrderResult) -> OrderResult:
        """Resolve an ambiguous send against venue state instead of resending."""

        client_order_id = order_intent.client_order_id or ""
        found = self._venue_holds_order(order_intent.symbol, client_order_id)
        raw = dict(last.raw) if isinstance(last.raw, dict) else {}
        raw["unknownOutcome"] = True
        raw["venueHoldsOrder"] = found
        detail = "order_found_at_venue" if found else "order_not_confirmed"
        return OrderResult(
            success=False,
            order_id=last.order_id,
            order_link_id=client_order_id,
            avg_price=last.avg_price,
            filled_qty=last.filled_qty,
            status="UNKNOWN",
            raw=raw,
            error=f"unknown_outcome_requires_reconciliation:{detail}",
        )

    @staticmethod
    def _is_leverage_limit_error(result: OrderResult) -> bool:
        raw = result.raw if isinstance(result.raw, dict) else {}
        ret_code = int(raw.get("retCode", 0) or 0)
        msg = str(result.error or raw.get("retMsg") or "").lower()
        return ret_code == 110090 or ("adjust your leverage" in msg and "position limit" in msg)

    @staticmethod
    def _is_qty_invalid_error(result: OrderResult) -> bool:
        raw = result.raw if isinstance(result.raw, dict) else {}
        ret_code = int(raw.get("retCode", 0) or 0)
        msg = str(result.error or raw.get("retMsg") or "").lower()
        return ret_code == 10001 and "qty invalid" in msg

    @staticmethod
    def _is_zero_position_error(result: OrderResult) -> bool:
        raw = result.raw if isinstance(result.raw, dict) else {}
        ret_code = int(raw.get("retCode", 0) or 0)
        msg = str(result.error or raw.get("retMsg") or "").lower()
        return ret_code == 110017 or "current position is zero" in msg

    def _retry_validation_after_account_refresh(
        self,
        *,
        order_intent: OrderIntent,
        rules,
        account,
        mark_price: float,
        open_orders: list[OpenOrderSnapshot],
        last_error: OrderValidationError,
    ) -> tuple[bool, Any, str]:
        if str(last_error) != "insufficient_available_balance":
            return False, account, str(last_error)
        try:
            refreshed_account = self.adapter.get_account()
        except Exception as exc:
            logger.warning("account_refresh_after_validation_failed symbol=%s error=%s", order_intent.symbol, exc)
            return False, account, str(last_error)
        try:
            validate_order_intent(
                order_intent,
                rules=rules,
                account=refreshed_account,
                mark_price=mark_price,
                open_orders=open_orders,
            )
        except OrderValidationError as exc:
            return False, refreshed_account, str(exc)
        logger.info(
            "account_refresh_after_validation_ok symbol=%s old_available=%.6f new_available=%.6f",
            order_intent.symbol,
            float(getattr(account, "available_balance_usdt", 0.0) or 0.0),
            float(getattr(refreshed_account, "available_balance_usdt", 0.0) or 0.0),
        )
        return True, refreshed_account, ""

    def _retry_after_leverage_alignment(self, order_intent: OrderIntent, last: OrderResult) -> OrderResult:
        if not self._is_leverage_limit_error(last):
            return last
        target = float(getattr(getattr(self.adapter, "config", None), "target_entry_leverage", 0.0) or 0.0)
        if target <= 0:
            return last
        align_fn = getattr(self.adapter, "ensure_position_leverage", None)
        if not callable(align_fn):
            return last
        if not bool(align_fn(order_intent.symbol, target)):
            return last
        logger.info("retrying order after leverage alignment symbol=%s leverage=%s", order_intent.symbol, target)
        return self.adapter.place_market_order(order_intent)

    def _retry_after_qty_refresh(
        self,
        *,
        order_intent: OrderIntent,
        last: OrderResult,
        account,
        mark_price: float,
        open_orders: list[OpenOrderSnapshot],
    ) -> tuple[OrderIntent, OrderResult]:
        if not self._is_qty_invalid_error(last):
            return order_intent, last
        try:
            fresh_rules = self.adapter.get_instrument_rules(order_intent.symbol, force_refresh=True)
        except Exception:
            return order_intent, last

        adjusted_qty = self.adapter.round_qty(float(order_intent.qty), float(fresh_rules.qty_step))
        if fresh_rules.max_qty > 0:
            max_qty = self.adapter.round_qty(self._safe_max_qty(fresh_rules), fresh_rules.qty_step)
            adjusted_qty = min(adjusted_qty, max_qty if max_qty > 0 else fresh_rules.max_qty)
        if adjusted_qty <= 0:
            return order_intent, last

        retried_intent = order_intent
        if abs(float(adjusted_qty) - float(order_intent.qty)) > 1e-12:
            retried_intent = OrderIntent(
                symbol=order_intent.symbol,
                side=order_intent.side,
                qty=float(adjusted_qty),
                reduce_only=order_intent.reduce_only,
                position_idx=order_intent.position_idx,
                client_order_id=order_intent.client_order_id,
                close_on_trigger=order_intent.close_on_trigger,
            )
        try:
            validate_order_intent(
                retried_intent,
                rules=fresh_rules,
                account=account,
                mark_price=mark_price,
                open_orders=open_orders,
            )
        except OrderValidationError:
            return order_intent, last

        logger.info(
            "retrying order after qty refresh symbol=%s old_qty=%s new_qty=%s",
            order_intent.symbol,
            order_intent.qty,
            retried_intent.qty,
        )
        return retried_intent, self.adapter.place_market_order(retried_intent)

    def _place_order_with_retry(self, order_intent: OrderIntent) -> OrderResult:
        last = self.adapter.place_market_order(order_intent)
        if last.success:
            return last
        last = self._retry_after_leverage_alignment(order_intent, last)
        if last.success:
            return last
        attempts = max(1, self.max_exchange_retries)
        for attempt in range(1, attempts):
            outcome = classify_failure(last)
            # A lost or ambiguous response may leave a live order behind. Query
            # the venue instead of sending the same command again on a guess.
            if outcome.requires_reconciliation():
                return self._reconcile_unknown_send(order_intent, last)
            if not outcome.permits_resend():
                return last
            time.sleep(min(0.5 * attempt, 1.5))
            last = self.adapter.place_market_order(order_intent)
            if last.success:
                return last

        if classify_failure(last).requires_reconciliation():
            return self._reconcile_unknown_send(order_intent, last)
        return last

    def _entry_orderbook_guard(self, order_intent: OrderIntent) -> tuple[bool, str, dict[str, Any]]:
        if not self.entry_orderbook_guard_enabled:
            return True, "disabled", {}
        quality_fn = getattr(self.adapter, "get_orderbook_quality", None)
        if not callable(quality_fn):
            return True, "adapter_no_orderbook_quality", {}

        try:
            quality = quality_fn(
                order_intent.symbol,
                order_intent.side,
                float(order_intent.qty),
                limit=self.entry_orderbook_limit,
                depth_slippage_bps=self.entry_orderbook_depth_slippage_bps,
            )
        except Exception as exc:
            if self.entry_orderbook_guard_require_live:
                return False, "orderbook_unavailable", {"error": str(exc)}
            logger.warning("entry_orderbook_guard unavailable symbol=%s error=%s", order_intent.symbol, exc)
            return True, "orderbook_unavailable_soft_pass", {"error": str(exc)}

        raw = self._orderbook_quality_raw(quality)
        if not bool(getattr(quality, "available", False)):
            if self.entry_orderbook_guard_require_live:
                return False, "orderbook_unavailable", raw
            return True, "orderbook_unavailable_soft_pass", raw

        max_slippage = self.max_entry_orderbook_slippage_bps
        if max_slippage > 0.0 and float(quality.expected_slippage_bps) > max_slippage:
            return False, "orderbook_slippage_too_high", raw

        min_depth_ratio = self.min_entry_orderbook_depth_ratio
        if min_depth_ratio > 0.0 and float(quality.depth_ratio) < min_depth_ratio:
            return False, "orderbook_depth_too_thin", raw

        return True, "ok", raw

    @staticmethod
    def _orderbook_quality_raw(quality: OrderBookQuality) -> dict[str, Any]:
        return {
            "symbol": str(quality.symbol),
            "side": str(quality.side.value if hasattr(quality.side, "value") else quality.side),
            "requested_qty": float(quality.requested_qty),
            "requested_notional_usdt": float(quality.requested_notional_usdt),
            "executable_qty": float(quality.executable_qty),
            "executable_notional_usdt": float(quality.executable_notional_usdt),
            "depth_ratio": float(quality.depth_ratio),
            "best_bid": float(quality.best_bid),
            "best_ask": float(quality.best_ask),
            "spread_bps": float(quality.spread_bps),
            "expected_avg_price": float(quality.expected_avg_price),
            "expected_slippage_bps": float(quality.expected_slippage_bps),
            "levels_used": int(quality.levels_used),
            "available": bool(quality.available),
        }

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
            outcome = classify_failure(last)
            # Deliberately different policy to _place_order_with_retry: setting a
            # stop overwrites a position attribute rather than creating an order,
            # so a duplicate is harmless while a missing stop leaves the position
            # naked. An ambiguous response is therefore retried here.
            if not (outcome.permits_resend() or outcome.requires_reconciliation()):
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

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): ExecutionEngine._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [ExecutionEngine._json_safe(item) for item in value]
        return str(value)

    @classmethod
    def _build_persisted_decision_raw(
        cls,
        *,
        intent: StrategyIntent,
        risk: RiskDecision,
        outcome: ExecutionOutcome,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if isinstance(outcome.raw, dict):
            payload.update(cls._json_safe(outcome.raw))

        intent_metadata = intent.metadata if isinstance(intent.metadata, dict) else {}
        safe_metadata = cls._json_safe(intent_metadata)
        payload["intent_context"] = {
            "symbol": str(intent.symbol),
            "action": intent.action.value,
            "reason": str(intent.reason),
            "confidence": float(intent.confidence),
            "created_at": float(intent.created_at),
            "stop_loss": float(intent.stop_loss or 0.0),
            "take_profit": float(intent.take_profit or 0.0),
            "metadata": safe_metadata,
        }
        payload["risk_context"] = {
            "approved": bool(risk.approved),
            "reason": str(risk.reason),
            "quantity": float(risk.quantity),
            "notional": float(risk.notional),
            "implied_leverage": float(risk.implied_leverage),
            "risk_amount_usdt": float(risk.risk_amount_usdt),
            "effective_stop_loss": float(risk.effective_stop_loss),
            "execution_cost_buffer_bps_used": float(risk.execution_cost_buffer_bps_used),
            "quality_penalty_bps_used": float(risk.quality_penalty_bps_used),
            "confidence_size_multiplier_used": float(risk.confidence_size_multiplier_used),
        }
        payload["execution_context"] = {
            "accepted": bool(outcome.accepted),
            "status": str(outcome.status),
            "reason": str(outcome.reason),
            "filled_qty": float(outcome.filled_qty),
            "avg_price": float(outcome.avg_price),
            "realized_pnl": float(outcome.realized_pnl),
            "stopped_out": bool(outcome.stopped_out),
        }

        if intent.take_profit:
            payload.setdefault("tp_price", float(intent.take_profit))
        if intent.stop_loss:
            payload.setdefault("sl_price", float(intent.stop_loss))
        if outcome.filled_qty > 0:
            payload["filled_qty"] = float(outcome.filled_qty)
        if outcome.avg_price > 0:
            if intent.action in (IntentAction.LONG_ENTRY, IntentAction.SHORT_ENTRY):
                payload.setdefault("entry_price", float(outcome.avg_price))
            if intent.action in (IntentAction.EXIT_LONG, IntentAction.EXIT_SHORT):
                payload["exit_price"] = float(outcome.avg_price)
        if intent.action in (IntentAction.EXIT_LONG, IntentAction.EXIT_SHORT) or outcome.realized_pnl != 0.0:
            payload["realized_pnl"] = float(outcome.realized_pnl)
        if outcome.stopped_out:
            payload["stopped_out"] = True

        for key in (
            "entry_price",
            "entry",
            "entry_px",
            "take_profit",
            "tp",
            "take_profit_price",
            "stop_loss",
            "sl",
            "stop_loss_price",
            "exit_type",
            "managed_exit",
            "managed_exit_reason",
            "managed_exit_details",
        ):
            if key in intent_metadata:
                payload[key] = cls._json_safe(intent_metadata.get(key))

        return payload

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
            raw=self._build_persisted_decision_raw(intent=intent, risk=risk, outcome=outcome),
        )
        self._persist_signal_admission(intent=intent, risk=risk, outcome=outcome)

    def _persist_signal_admission(
        self,
        *,
        intent: StrategyIntent,
        risk: RiskDecision,
        outcome: ExecutionOutcome,
    ) -> None:
        if self.persistence is None or not isinstance(intent.metadata, dict):
            return
        gate = intent.metadata.get("entry_gate")
        if not isinstance(gate, dict):
            return
        signal_id = str(intent.metadata.get("legacy_signal_id") or "")
        if not signal_id:
            signal_id = self._idempotency_key(intent)
        raw = {
            "entry_gate": self._json_safe(gate),
            "admission_status": self._json_safe(intent.metadata.get("admission_status")),
            "admission_reason": self._json_safe(intent.metadata.get("admission_reason")),
            "intent_reason": str(intent.reason),
            "intent_confidence": float(intent.confidence),
            "risk_approved": bool(risk.approved),
            "risk_reason": str(risk.reason),
            "execution_status": str(outcome.status),
            "execution_reason": str(outcome.reason),
        }
        self.persistence.append_signal_admission(
            signal_id=signal_id,
            symbol=intent.symbol.replace("/", "").upper(),
            side=str(intent.metadata.get("signal_side") or intent.action.value),
            action=intent.action.value,
            approved=bool(gate.get("approved")),
            reason=str(gate.get("reason") or intent.reason),
            score=float(gate.get("score") or 0.0),
            ts=time.time(),
            raw=raw,
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
                    client_order_id=self._stable_client_order_id("v2-recover", entry.intent_key, norm_symbol),
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

            client_order_id = self._stable_client_order_id("v2", intent_key, norm_symbol, qty)
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

            active_account = snapshot.account
            try:
                validate_order_intent(
                    order_intent,
                    rules=rules,
                    account=active_account,
                    mark_price=mark_price,
                    open_orders=snapshot.open_orders,
                )
            except OrderValidationError as exc:
                refreshed_ok, active_account, refreshed_reason = self._retry_validation_after_account_refresh(
                    order_intent=order_intent,
                    rules=rules,
                    account=active_account,
                    mark_price=mark_price,
                    open_orders=snapshot.open_orders,
                    last_error=exc,
                )
                if not refreshed_ok:
                    self._persist_intent_status(
                        intent_key=intent_key,
                        symbol=norm_symbol,
                        action=intent.action,
                        payload=payload,
                        status="validation_failed",
                    )
                    return ExecutionOutcome(accepted=False, status="REJECTED", reason=f"order_validation:{refreshed_reason}")

            guard_ok, guard_reason, guard_raw = self._entry_orderbook_guard(order_intent)
            if guard_raw:
                payload["orderbook_quality"] = guard_raw
            if not guard_ok:
                self._persist_intent_status(
                    intent_key=intent_key,
                    symbol=norm_symbol,
                    action=intent.action,
                    payload=payload,
                    status="orderbook_guard_failed",
                )
                return ExecutionOutcome(
                    accepted=False,
                    status="REJECTED",
                    reason=guard_reason,
                    raw={"orderbook_quality": guard_raw},
                )

            pending_state = TradeState.PENDING_ENTRY_LONG if pos_side == PositionSide.LONG else TradeState.PENDING_ENTRY_SHORT
            self.state_machine.transition(norm_symbol, pending_state, "entry_order_submitted")
            result = self._place_order_with_retry(order_intent)
            order_intent, result = self._retry_after_qty_refresh(
                order_intent=order_intent,
                last=result,
                account=active_account,
                mark_price=mark_price,
                open_orders=snapshot.open_orders,
            )
            if not result.success:
                result = self._retry_after_leverage_alignment(order_intent, result)
            payload["requested_qty"] = float(order_intent.qty)
            if not result.success:
                self.state_machine.transition(norm_symbol, TradeState.FLAT, "entry_order_failed")
                self._persist_intent_status(
                    intent_key=intent_key,
                    symbol=norm_symbol,
                    action=intent.action,
                    payload={**payload, "exchange_error": result.error},
                    status="failed_submission",
                )
                if str(result.status).upper() == "UNKNOWN":
                    return ExecutionOutcome(
                        accepted=False,
                        status="UNKNOWN",
                        reason=f"requires_reconciliation:{result.error}",
                        order_link_id=result.order_link_id,
                        raw=result.raw,
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

            live_position = self._fetch_live_position(norm_symbol)
            if live_position is None or float(live_position.qty) <= 0:
                self.state_machine.transition(norm_symbol, TradeState.FLAT, "exit_without_live_position")
                return ExecutionOutcome(accepted=False, status="IGNORED", reason="no_live_position")
            position = live_position

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
                    client_order_id=self._stable_client_order_id("v2-exit", intent_key, norm_symbol, position.qty),
                    close_on_trigger=True,
                )
            )
            if not result.success:
                if self._is_zero_position_error(result):
                    self.state_machine.transition(norm_symbol, TradeState.FLAT, "exit_zero_position_on_exchange")
                    self._persist_intent_status(
                        intent_key=exit_key,
                        symbol=norm_symbol,
                        action=intent.action,
                        payload={**payload, "exchange_error": result.error},
                        status="exchange_already_flat",
                    )
                    return ExecutionOutcome(accepted=False, status="IGNORED", reason="no_live_position", raw=result.raw)
                fallback_state = TradeState.LONG if position.side == PositionSide.LONG else TradeState.SHORT
                self.state_machine.transition(norm_symbol, fallback_state, "exit_order_failed")
                self._persist_intent_status(
                    intent_key=exit_key,
                    symbol=norm_symbol,
                    action=intent.action,
                    payload={**payload, "exchange_error": result.error},
                    status="failed_submission",
                )
                if str(result.status).upper() == "UNKNOWN":
                    return ExecutionOutcome(
                        accepted=False,
                        status="UNKNOWN",
                        reason=f"requires_reconciliation:{result.error}",
                        order_link_id=result.order_link_id,
                        raw=result.raw,
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





















