from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.bootstrap import ConfigError, RuntimeConfig, load_runtime_config
from trading.execution.engine import ExecutionEngine
from trading.exchange.bybit_adapter import BybitAdapter
from trading.exchange.schemas import OrderIntent, OrderResult, OrderSide, PositionSide
from trading.market_data.reconciliation import ExchangeReconciler
from trading.market_data.ws_reconciliation import ExchangeSyncService
from trading.metrics.logging import setup_logging
from trading.portfolio.positions import POSITION_SIZE_EPSILON, summarize_positions
from trading.risk.engine import RiskDecision, RiskEngine
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.state.machine import StateMachine
from trading.state.models import TradeState
from trading.state.persistence import RuntimeStore


STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"
STATUS_SKIP = "SKIP"

def classify_config_error_status(error_text: str) -> str:
    text = str(error_text or "").lower()
    if "bybit_api_key and bybit_api_secret are required" in text:
        return STATUS_BLOCKED
    if "private call requires api key/secret" in text:
        return STATUS_BLOCKED
    return STATUS_FAIL


@dataclass
class ScenarioResult:
    name: str
    status: str
    duration_sec: float
    details: dict[str, Any]
    error: str = ""


class TestnetValidationHarness:
    def __init__(
        self,
        cfg: RuntimeConfig,
        *,
        symbol: str,
        max_notional_usdt: float,
        execute_orders: bool,
        soak_seconds: int,
        chaos_cycles: int,
        run_full_suite: bool,
        logger,
        artifacts_root: str,
        deployment_constraints_out: str,
    ):
        self.cfg = cfg
        self.symbol = str(symbol).replace("/", "").upper()
        self.max_notional_usdt = float(max_notional_usdt)
        self.execute_orders = bool(execute_orders)
        self.soak_seconds = max(0, int(soak_seconds))
        self.chaos_cycles = max(0, int(chaos_cycles))
        self.run_full_suite = bool(run_full_suite)
        self.logger = logger
        self.artifacts_root = str(artifacts_root)
        self.deployment_constraints_out = str(deployment_constraints_out)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.artifacts_dir = Path(self.artifacts_root).resolve() / self.run_id
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.adapter = BybitAdapter(cfg.adapter)
        self.adapter.set_ws_symbols([self.symbol])
        self.reconciler = ExchangeReconciler(self.adapter)
        self.sync = ExchangeSyncService(
            self.reconciler,
            poll_interval_sec=max(1, min(cfg.ws_poll_interval_sec, 5)),
            max_event_staleness_sec=max(2, min(cfg.ws_event_staleness_sec, 15)),
        )

        runtime_dir = Path(cfg.runtime_db_path).resolve().parent
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_db = runtime_dir / "v2_testnet_validation.db"

        self.runtime_store = RuntimeStore(str(self.runtime_db))
        self.state_machine = StateMachine(persistence=self.runtime_store)
        self.risk = RiskEngine(cfg.risk_limits, persistence=self.runtime_store)
        self.execution = ExecutionEngine(
            adapter=self.adapter,
            state_machine=self.state_machine,
            hedge_mode=cfg.adapter.hedge_mode,
            stop_loss_required=cfg.risk_limits.require_stop_loss,
            require_reconciliation=cfg.flags.reconciliation_required,
            stop_attach_grace_sec=cfg.stop_attach_grace_sec,
            stale_open_order_sec=cfg.stale_open_order_sec,
            max_exchange_retries=cfg.max_exchange_retries,
            persistence=self.runtime_store,
        )

        self._captured_events: list[dict[str, Any]] = []
        self._raw_ws_events: list[dict[str, Any]] = []
        self._normalized_ws_events: list[dict[str, Any]] = []
        self._order_lifecycle_events: list[dict[str, Any]] = []
        self._recovery_events: list[dict[str, Any]] = []

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(v) for v in value]
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, (str, int, float, bool)):
            return enum_value
        if hasattr(value, "__dict__"):
            return {
                str(k): self._json_safe(v)
                for k, v in value.__dict__.items()
                if not str(k).startswith("_")
            }
        return str(value)

    def _record_order_event(self, *, scenario: str, stage: str, payload: dict[str, Any]):
        self._order_lifecycle_events.append(
            {
                "ts": self._now_iso(),
                "scenario": str(scenario),
                "stage": str(stage),
                "payload": self._json_safe(payload),
            }
        )

    def _pull_adapter_events(self) -> list[Any]:
        raw_events = []
        drain_raw = getattr(self.adapter, "drain_ws_raw_events", None)
        if callable(drain_raw):
            try:
                raw_events = [item for item in (drain_raw() or []) if isinstance(item, dict)]
            except Exception:
                raw_events = []
        if raw_events:
            self._raw_ws_events.extend(self._json_safe(raw_events))

        events = self.adapter.drain_ws_events() or []
        if events:
            normalized = []
            for event in events:
                normalized.append(
                    {
                        "ts": float(getattr(event, "ts", time.time())),
                        "event_type": str(getattr(getattr(event, "event_type", None), "value", "")),
                        "symbol": str(getattr(event, "symbol", "") or ""),
                        "payload": self._json_safe(getattr(event, "payload", {})),
                    }
                )
            self._normalized_ws_events.extend(normalized)
            self.sync.process_events(events)
        return events

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                f.write("\n")
        return str(path.resolve())

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path.resolve())

    @staticmethod
    def _sha256_lines(lines: list[str]) -> str:
        joined = "\n".join(lines)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _write_runtime_constraints_artifacts(self) -> dict[str, Any]:
        cmd = [sys.executable, "-m", "pip", "freeze", "--all"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if int(proc.returncode) != 0:
            return {
                "status": "FAIL",
                "command": " ".join(cmd),
                "returncode": int(proc.returncode),
                "error_tail": "\n".join((proc.stderr or "").strip().splitlines()[-10:]),
            }

        lines = []
        for line in (proc.stdout or "").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            lines.append(value)
        lines = sorted(set(lines))

        constraints_text = "\n".join(lines) + ("\n" if lines else "")
        run_constraints = self.artifacts_dir / "runtime_constraints.lock.txt"
        run_constraints.parent.mkdir(parents=True, exist_ok=True)
        run_constraints.write_text(constraints_text, encoding="utf-8")

        deploy_constraints = Path(self.deployment_constraints_out).resolve()
        deploy_constraints.parent.mkdir(parents=True, exist_ok=True)
        deploy_constraints.write_text(constraints_text, encoding="utf-8")

        return {
            "status": "PASS",
            "constraints_file": str(run_constraints.resolve()),
            "deployment_constraints_file": str(deploy_constraints),
            "package_count": len(lines),
            "sha256": self._sha256_lines(lines),
            "command": " ".join(cmd),
        }

    def close(self):
        try:
            self.runtime_store.close()
        finally:
            self.adapter.close()

    def _run_scenario(self, name: str, fn, *, allow_blocked: bool = True) -> ScenarioResult:
        start = time.time()
        try:
            status, details = fn()
            if status == STATUS_BLOCKED and not allow_blocked:
                status = STATUS_FAIL
            return ScenarioResult(
                name=name,
                status=status,
                duration_sec=round(time.time() - start, 3),
                details=details,
            )
        except Exception as exc:
            return ScenarioResult(
                name=name,
                status=STATUS_FAIL,
                duration_sec=round(time.time() - start, 3),
                details={},
                error=f"{type(exc).__name__}: {exc}",
            )

    def _drain_ws_events(self, duration_sec: float) -> list[Any]:
        out = []
        end_ts = time.time() + max(0.0, float(duration_sec))
        while time.time() < end_ts:
            events = self._pull_adapter_events()
            if events:
                out.extend(events)
            time.sleep(0.2)
        return out

    def _snapshot(self):
        self._pull_adapter_events()
        return self.sync.snapshot(self.symbol)

    def _cancel_all_orders(self) -> int:
        canceled = 0
        for order in self.adapter.get_open_orders(self.symbol):
            ok = self.adapter.cancel_order(
                symbol=self.symbol,
                order_id=order.order_id,
                order_link_id=order.order_link_id,
            )
            self._record_order_event(
                scenario="safety_cleanup",
                stage="cancel_order",
                payload={
                    "symbol": self.symbol,
                    "order_id": order.order_id,
                    "order_link_id": order.order_link_id,
                    "ok": bool(ok),
                },
            )
            if ok:
                canceled += 1
        return canceled

    def _flatten_symbol(self) -> int:
        closed = 0
        positions = self.adapter.get_positions(self.symbol)
        for pos in positions:
            if pos.qty <= 0:
                continue
            close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
            res = self.adapter.place_market_order(
                OrderIntent(
                    symbol=self.symbol,
                    side=close_side,
                    qty=float(pos.qty),
                    reduce_only=True,
                    position_idx=int(pos.position_idx),
                    client_order_id=f"v2-val-flat-{abs(hash((self.symbol, time.time(), pos.qty))) % 10**12}",
                    close_on_trigger=True,
                )
            )
            self._record_order_event(
                scenario="safety_cleanup",
                stage="flatten_position",
                payload={
                    "symbol": self.symbol,
                    "side": close_side.value,
                    "qty": float(pos.qty),
                    "position_idx": int(pos.position_idx),
                    "success": bool(res.success),
                    "error": res.error,
                    "order_id": res.order_id,
                    "order_link_id": res.order_link_id,
                },
            )
            if res.success:
                closed += 1
        return closed

    def _safety_cleanup(self, *, settle_timeout_sec: float = 6.0, settle_poll_sec: float = 0.5) -> dict[str, Any]:
        canceled = self._cancel_all_orders()
        closed = self._flatten_symbol()
        settle_started_ts = time.time()
        settle_attempts = 0
        snap = self._snapshot()
        pos_summary = summarize_positions(
            snap.positions,
            symbol=self.symbol,
            size_epsilon=POSITION_SIZE_EPSILON,
        )
        while True:
            settle_attempts += 1
            open_orders_after = len(snap.open_orders)
            open_positions_after = int(pos_summary.get("effective_open_positions_count", 0))
            if open_orders_after == 0 and open_positions_after == 0:
                break
            if (time.time() - settle_started_ts) >= settle_timeout_sec:
                break

            if open_orders_after > 0:
                canceled += self._cancel_all_orders()
            if open_positions_after > 0:
                closed += self._flatten_symbol()

            time.sleep(settle_poll_sec)
            snap = self._snapshot()
            pos_summary = summarize_positions(
                snap.positions,
                symbol=self.symbol,
                size_epsilon=POSITION_SIZE_EPSILON,
            )

        settle_open_orders = len(snap.open_orders)
        settle_open_positions = int(pos_summary.get("effective_open_positions_count", 0))
        settle_open_positions_raw = int(pos_summary.get("raw_positions_count", 0))
        flat_confirmed = settle_open_orders == 0 and settle_open_positions == 0
        settle_elapsed_sec = round(time.time() - settle_started_ts, 3)

        observe_timeout_sec = 2.0
        observe_poll_sec = 0.5
        observe_started_ts = time.time()
        observe_attempts = 0
        lag_recovered = False
        observe_snap = snap
        observe_pos_summary = pos_summary

        if not flat_confirmed:
            while (time.time() - observe_started_ts) < observe_timeout_sec:
                observe_attempts += 1
                time.sleep(observe_poll_sec)
                observe_snap = self._snapshot()
                observe_pos_summary = summarize_positions(
                    observe_snap.positions,
                    symbol=self.symbol,
                    size_epsilon=POSITION_SIZE_EPSILON,
                )
                if len(observe_snap.open_orders) == 0 and int(observe_pos_summary.get("effective_open_positions_count", 0)) == 0:
                    lag_recovered = True
                    break

        final_open_orders = len(observe_snap.open_orders)
        final_open_positions = int(observe_pos_summary.get("effective_open_positions_count", 0))
        final_open_positions_raw = int(observe_pos_summary.get("raw_positions_count", 0))
        final_flat = final_open_orders == 0 and final_open_positions == 0
        total_elapsed_sec = round(time.time() - settle_started_ts, 3)

        if flat_confirmed:
            consistency_status = "flat_confirmed"
        elif lag_recovered and final_flat:
            consistency_status = "exchange_reporting_lag"
        elif final_open_positions > 0:
            consistency_status = "residual_exposure"
        elif final_open_orders > 0:
            consistency_status = "residual_open_orders"
        else:
            consistency_status = "cleanup_report_timing_bug"

        return {
            "canceled_orders": canceled,
            "closed_positions": closed,
            "open_orders_after": final_open_orders,
            "open_positions_after": final_open_positions,
            "open_positions_raw_after": final_open_positions_raw,
            "effective_open_positions_after": final_open_positions,
            "normalized_open_positions_after": [dict(item) for item in (observe_pos_summary.get("effective_positions") or [])],
            "zero_size_placeholder_positions_after": [dict(item) for item in (observe_pos_summary.get("zero_size_placeholder_positions") or [])],
            "position_size_epsilon": float(observe_pos_summary.get("position_size_epsilon", POSITION_SIZE_EPSILON)),
            "flat_confirmed": bool(final_flat),
            "flat_confirmed_settle_phase": bool(flat_confirmed),
            "flat_confirmed_observe_phase": bool(lag_recovered),
            "settle_open_positions_effective": settle_open_positions,
            "settle_open_positions_raw": settle_open_positions_raw,
            "settle_attempts": settle_attempts,
            "observe_attempts": observe_attempts,
            "settle_elapsed_sec": settle_elapsed_sec,
            "total_elapsed_sec": total_elapsed_sec,
            "timing_artifact_possible": bool((closed > 0 or canceled > 0) and not flat_confirmed and lag_recovered),
            "exchange_lag_detected": bool(not flat_confirmed and lag_recovered),
            "residual_exposure_detected": bool(final_open_positions > 0),
            "residual_open_orders_detected": bool(final_open_orders > 0),
            "consistency_status": consistency_status,
        }

    def _find_open_order_by_link_id(self, order_link_id: str):
        lookup = str(order_link_id or "").strip()
        if not lookup:
            return None
        for order in self.adapter.get_open_orders(self.symbol):
            if str(order.order_link_id or "").strip() == lookup:
                return order
        return None

    def _place_limit_order_with_reconcile(
        self,
        *,
        symbol: str,
        side: OrderSide,
        qty: float,
        price: float,
        reduce_only: bool,
        position_idx: int,
        client_order_id: str,
        close_on_trigger: bool | None = None,
    ) -> OrderResult:
        existing = self._find_open_order_by_link_id(client_order_id)
        if existing is not None:
            return OrderResult(
                success=True,
                order_id=str(existing.order_id),
                order_link_id=str(existing.order_link_id or client_order_id),
                avg_price=float(price),
                filled_qty=0.0,
                remaining_qty=float(existing.qty),
                status=str(existing.status or "New"),
                raw={"recovered_existing_order": True, "source": "pre_submit_reconcile"},
                error=None,
            )

        result = self.adapter.place_limit_order(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            reduce_only=reduce_only,
            position_idx=position_idx,
            client_order_id=client_order_id,
            close_on_trigger=close_on_trigger,
        )
        if result.success:
            return result

        raw = result.raw if isinstance(result.raw, dict) else {}
        err_blob = f"{result.error or ''} {raw.get('retMsg', '')}".lower()
        if "duplicate" not in err_blob:
            return result

        existing_after = self._find_open_order_by_link_id(client_order_id)
        if existing_after is None:
            return result

        return OrderResult(
            success=True,
            order_id=str(existing_after.order_id),
            order_link_id=str(existing_after.order_link_id or client_order_id),
            avg_price=float(price),
            filled_qty=0.0,
            remaining_qty=float(existing_after.qty),
            status=str(existing_after.status or "New"),
            raw={
                "recovered_existing_order": True,
                "source": "duplicate_reconcile",
                "duplicate_error": str(result.error or raw.get("retMsg") or ""),
            },
            error=None,
        )

    def _compute_tiny_qty(self, mark_price: float, min_qty: float, qty_step: float, min_notional: float) -> float:
        if mark_price <= 0:
            raise RuntimeError("invalid_mark_price")

        target_notional = min(self.max_notional_usdt, max(6.0, float(min_notional) * 1.2))
        if target_notional < float(min_notional):
            raise RuntimeError(f"max_notional_below_exchange_min:{self.max_notional_usdt}<{min_notional}")

        qty = target_notional / mark_price
        qty = self.adapter.round_qty(qty, qty_step)
        if qty < min_qty:
            qty = min_qty
            qty = self.adapter.round_qty(qty, qty_step)

        notional = qty * mark_price
        if notional > self.max_notional_usdt * 1.001:
            raise RuntimeError(f"qty_exceeds_cap:{notional:.6f}>{self.max_notional_usdt:.6f}")
        if notional < min_notional:
            raise RuntimeError(f"qty_below_min_notional:{notional:.6f}<{min_notional:.6f}")
        return float(qty)

    def _entry_intent(self, mark_price: float) -> StrategyIntent:
        return StrategyIntent(
            symbol=self.symbol,
            action=IntentAction.LONG_ENTRY,
            reason="testnet_validation_entry",
            stop_loss=float(mark_price * 0.99),
            take_profit=float(mark_price * 1.01),
        )

    def _tiny_entry_setup_context(self, *, force_poll_snapshot: bool) -> dict[str, Any]:
        snap = self.reconciler.snapshot(self.symbol) if force_poll_snapshot else self._snapshot()
        rules = self.adapter.get_instrument_rules(self.symbol)
        mark_price = self.adapter.get_mark_price(self.symbol)
        if mark_price <= 0:
            mark_price = 0.0
            if snap.positions:
                mark_price = float(snap.positions[0].entry_price)
            if mark_price <= 0:
                raise RuntimeError("no_valid_mark_price")

        qty_cap = self._compute_tiny_qty(
            mark_price=mark_price,
            min_qty=rules.min_qty,
            qty_step=rules.qty_step,
            min_notional=rules.min_notional,
        )

        intent = self._entry_intent(mark_price)
        decision = self.risk.evaluate(
            intent=intent,
            account=snap.account,
            existing_positions=snap.positions,
            mark_price=mark_price,
            rules=rules,
        )

        requested_qty = float(min(max(0.0, float(decision.quantity or 0.0)), qty_cap))
        requested_notional = float(requested_qty * mark_price)
        diagnostics = {
            "requested_qty": float(requested_qty),
            "requested_notional": float(requested_notional),
            "mark_price_used": float(mark_price),
            "min_qty": float(rules.min_qty),
            "min_notional": float(rules.min_notional),
            "qty_step": float(rules.qty_step),
            "available_balance_snapshot": float(snap.account.available_balance_usdt),
            "equity_snapshot": float(snap.account.equity_usdt),
            "affordability_inputs": {
                "intent_reduce_only": False,
                "available_balance_usdt": float(snap.account.available_balance_usdt),
                "mark_price": float(mark_price),
                "qty": float(requested_qty),
                "notional": float(requested_notional),
                "open_orders_count": int(len(snap.open_orders)),
            },
            "risk_approved": bool(decision.approved),
            "risk_reason": str(decision.reason),
        }

        final_decision = RiskDecision(
            approved=bool(decision.approved),
            reason=str(decision.reason),
            quantity=float(requested_qty),
            notional=float(min(float(decision.notional or 0.0), requested_notional)),
            implied_leverage=float(decision.implied_leverage),
        )
        return {
            "snapshot": snap,
            "rules": rules,
            "mark_price": float(mark_price),
            "qty_cap": float(qty_cap),
            "intent": intent,
            "decision": final_decision,
            "diagnostics": diagnostics,
        }

    def _wait_for_effective_external_position(
        self,
        *,
        timeout_sec: float = 8.0,
        poll_sec: float = 0.5,
    ) -> dict[str, Any]:
        started = time.time()
        attempts = 0
        last_summary: dict[str, Any] = {
            "raw_positions_count": 0,
            "effective_open_positions_count": 0,
            "effective_positions": [],
            "zero_size_placeholder_positions": [],
            "position_size_epsilon": float(POSITION_SIZE_EPSILON),
        }

        while True:
            attempts += 1
            snap = self._snapshot()
            last_summary = summarize_positions(
                snap.positions,
                symbol=self.symbol,
                size_epsilon=POSITION_SIZE_EPSILON,
            )
            if int(last_summary.get("effective_open_positions_count", 0)) > 0:
                break
            if (time.time() - started) >= max(0.0, float(timeout_sec)):
                break
            time.sleep(max(0.0, float(poll_sec)))

        confirmed = int(last_summary.get("effective_open_positions_count", 0)) > 0
        return {
            "confirmed": bool(confirmed),
            "attempts": int(attempts),
            "elapsed_sec": round(time.time() - started, 3),
            "effective_open_positions_count": int(last_summary.get("effective_open_positions_count", 0)),
            "raw_positions_count": int(last_summary.get("raw_positions_count", 0)),
            "effective_positions": [dict(item) for item in (last_summary.get("effective_positions") or [])],
            "zero_size_placeholder_positions": [dict(item) for item in (last_summary.get("zero_size_placeholder_positions") or [])],
        }

    def _open_external_manual_exposure(self, *, scenario_tag: str) -> dict[str, Any]:
        setup = self._tiny_entry_setup_context(force_poll_snapshot=True)
        mark = float(setup["mark_price"])
        rules = setup["rules"]
        qty = float((setup["diagnostics"] or {}).get("requested_qty", 0.0))
        if qty <= 0:
            qty = self._compute_tiny_qty(mark, float(rules.min_qty), float(rules.qty_step), float(rules.min_notional))

        link_id = f"v2-val-manual-{scenario_tag}-{abs(hash((self.symbol, time.time_ns()))) % 10**10}"
        position_idx = self.adapter.position_idx_for_side(PositionSide.LONG, self.cfg.adapter.hedge_mode)
        manual = self.adapter.place_market_order(
            OrderIntent(
                symbol=self.symbol,
                side=OrderSide.BUY,
                qty=float(qty),
                reduce_only=False,
                position_idx=position_idx,
                client_order_id=link_id,
            )
        )

        wait = self._wait_for_effective_external_position(timeout_sec=8.0, poll_sec=0.5)
        return {
            "manual_order_submitted": bool(manual.success),
            "manual_order_id": str(manual.order_id),
            "manual_order_link_id": str(manual.order_link_id or link_id),
            "manual_fill_qty": float(manual.filled_qty),
            "manual_order_error": str(manual.error or ""),
            "requested_qty": float(qty),
            "requested_notional": float(qty * mark),
            "mark_price_used": float(mark),
            "min_qty": float(rules.min_qty),
            "min_notional": float(rules.min_notional),
            "qty_step": float(rules.qty_step),
            "setup_affordability": dict(setup.get("diagnostics") or {}),
            "effective_position_confirmed": bool(wait.get("confirmed")),
            "effective_open_positions_count": int(wait.get("effective_open_positions_count", 0)),
            "raw_positions_count": int(wait.get("raw_positions_count", 0)),
            "effective_positions": [dict(item) for item in (wait.get("effective_positions") or [])],
            "zero_size_placeholder_positions": [dict(item) for item in (wait.get("zero_size_placeholder_positions") or [])],
            "position_wait_attempts": int(wait.get("attempts", 0)),
            "position_wait_elapsed_sec": float(wait.get("elapsed_sec", 0.0)),
        }

    def _run_tiny_lifecycle(self, *, tag: str) -> dict[str, Any]:
        start_cleanup = self._safety_cleanup()
        # Initial sizing context.
        setup_ctx = self._tiny_entry_setup_context(force_poll_snapshot=False)
        rules = setup_ctx["rules"]
        mark_price = float(setup_ctx["mark_price"])
        qty_cap = float(setup_ctx["qty_cap"])
        intent = setup_ctx["intent"]

        baseline_events = self._drain_ws_events(0.5)

        # Fresh exchange-truth snapshot right before order validation/execution.
        setup_ctx = self._tiny_entry_setup_context(force_poll_snapshot=True)
        snap_for_validation = setup_ctx["snapshot"]
        decision = setup_ctx["decision"]
        entry_setup = dict(setup_ctx.get("diagnostics") or {})

        if not bool(decision.approved):
            raise RuntimeError(f"risk_rejected:{decision.reason}")

        out_entry = self.execution.execute(
            intent=intent,
            risk=decision,
            snapshot=snap_for_validation,
            mark_price=mark_price,
        )
        self._record_order_event(
            scenario=f"{tag}_lifecycle",
            stage="entry_execute",
            payload={
                "accepted": bool(out_entry.accepted),
                "status": out_entry.status,
                "reason": out_entry.reason,
                "filled_qty": float(out_entry.filled_qty),
                "order_id": out_entry.order_id,
                "order_link_id": out_entry.order_link_id,
                "entry_setup": entry_setup,
            },
        )
        if not out_entry.accepted:
            raise RuntimeError(f"entry_failed:{out_entry.reason}")

        time.sleep(1.0)
        snap_after_entry = self._snapshot()
        positions_after_entry = self.adapter.get_positions(self.symbol)
        if not positions_after_entry:
            raise RuntimeError("entry_not_visible_on_exchange")

        pos = positions_after_entry[0]
        entry_filled_qty = float(out_entry.filled_qty)
        exit_requested_qty = float(pos.qty)
        qty_match_tol = max(float(rules.qty_step), 1e-9)
        exit_qty_matches_entry_fill = abs(exit_requested_qty - entry_filled_qty) <= qty_match_tol

        exit_action = IntentAction.EXIT_LONG if pos.side == PositionSide.LONG else IntentAction.EXIT_SHORT
        exit_intent = StrategyIntent(symbol=self.symbol, action=exit_action, reason=f"{tag}_exit")
        exit_decision = self.risk.evaluate(
            intent=exit_intent,
            account=snap_after_entry.account,
            existing_positions=snap_after_entry.positions,
            mark_price=mark_price,
            rules=rules,
        )
        out_exit = self.execution.execute(
            intent=exit_intent,
            risk=exit_decision,
            snapshot=snap_after_entry,
            mark_price=mark_price,
        )
        self._record_order_event(
            scenario=f"{tag}_lifecycle",
            stage="exit_execute",
            payload={
                "accepted": bool(out_exit.accepted),
                "status": out_exit.status,
                "reason": out_exit.reason,
                "filled_qty": float(out_exit.filled_qty),
                "order_id": out_exit.order_id,
                "order_link_id": out_exit.order_link_id,
                "requested_qty": float(exit_requested_qty),
                "reduce_only": True,
            },
        )

        post_exit_reconcile = self._reconcile_until_flat(
            scenario_name=f"{tag}_post_exit",
            timeout_sec=6.0,
            poll_sec=0.5,
        )

        positions_before_cleanup = self.adapter.get_positions(self.symbol)
        open_orders_before_cleanup = self.adapter.get_open_orders(self.symbol)
        exchange_position_size_before_cleanup = float(sum(max(0.0, float(p.qty)) for p in positions_before_cleanup))
        exchange_position_side_before_cleanup = str(positions_before_cleanup[0].side.value) if positions_before_cleanup else ""

        final_cleanup = self._safety_cleanup()
        final_cleanup_state = self._collect_scenario_start_state(f"{tag}_final_cleanup")
        final_cleanup_invariant_ok, final_cleanup_invariant_reason = self._state_clean_invariant(final_cleanup_state)

        final_positions = self.adapter.get_positions(self.symbol)
        final_open_orders = self.adapter.get_open_orders(self.symbol)
        exchange_position_size_after_cleanup = float(sum(max(0.0, float(p.qty)) for p in final_positions))
        exchange_position_side_after_cleanup = str(final_positions[0].side.value) if final_positions else ""

        final_snap = self._snapshot()
        post_events = self._drain_ws_events(2.0)

        captured = [
            {
                "event_type": str(e.event_type.value),
                "symbol": str(e.symbol or ""),
                "payload_keys": sorted(list(e.payload.keys())) if isinstance(e.payload, dict) else [],
            }
            for e in (baseline_events + post_events)
        ]
        self._captured_events.extend(captured)

        return {
            "entry": {
                "accepted": out_entry.accepted,
                "status": out_entry.status,
                "reason": out_entry.reason,
                "filled_qty": out_entry.filled_qty,
                "order_id": out_entry.order_id,
                "order_link_id": out_entry.order_link_id,
            },
            "exit": {
                "accepted": out_exit.accepted,
                "status": out_exit.status,
                "reason": out_exit.reason,
                "filled_qty": out_exit.filled_qty,
                "order_id": out_exit.order_id,
                "order_link_id": out_exit.order_link_id,
            },
            "entry_setup": entry_setup,
            "start_cleanup": start_cleanup,
            "final_cleanup": final_cleanup,
            "post_exit_reconcile": post_exit_reconcile,
            "final_cleanup_state": final_cleanup_state,
            "final_cleanup_invariant_ok": bool(final_cleanup_invariant_ok),
            "final_cleanup_invariant_reason": str(final_cleanup_invariant_reason),
            "entry_filled_qty": float(entry_filled_qty),
            "exit_requested_qty": float(exit_requested_qty),
            "exit_filled_qty": float(out_exit.filled_qty),
            "exit_qty_matches_entry_fill": bool(exit_qty_matches_entry_fill),
            "exit_reduce_only": True,
            "exchange_position_size_before_cleanup": exchange_position_size_before_cleanup,
            "exchange_position_side_before_cleanup": exchange_position_side_before_cleanup,
            "exchange_position_size_after_cleanup": exchange_position_size_after_cleanup,
            "exchange_position_side_after_cleanup": exchange_position_side_after_cleanup,
            "positions_after_entry": len(positions_after_entry),
            "positions_after_exit": len(positions_before_cleanup),
            "open_orders_after_exit": len(open_orders_before_cleanup),
            "open_orders_after_cleanup": len(final_open_orders),
            "positions_after_cleanup": len(final_positions),
            "open_orders_after_final_snapshot": len(final_snap.open_orders),
            "mark_price": mark_price,
            "qty_cap": qty_cap,
            "ws_events_captured": len(captured),
        }

    def _persistence_snapshot_summary(self) -> dict[str, Any]:
        open_inflight = self.runtime_store.load_open_inflight_intents()
        status_counts: dict[str, int] = {}
        for row in open_inflight:
            key = str(getattr(row, "status", "") or "")
            status_counts[key] = int(status_counts.get(key, 0) + 1)

        idempotency = self.runtime_store.load_live_idempotency_keys()
        return {
            "open_inflight_count": len(open_inflight),
            "open_inflight_status_counts": status_counts,
            "open_inflight_keys_sample": [str(getattr(row, "intent_key", "")) for row in open_inflight[:5]],
            "idempotency_key_count": len(idempotency),
            "schema_version": self.runtime_store.get_schema_version(),
        }

    def _collect_scenario_start_state(self, scenario_name: str) -> dict[str, Any]:
        snap = self._snapshot()
        positions_summary = summarize_positions(
            snap.positions,
            symbol=self.symbol,
            size_epsilon=POSITION_SIZE_EPSILON,
        )
        local_before = self.state_machine.get(self.symbol).state.value
        self.state_machine.reconcile(self.symbol, snap.positions, snap.open_orders)
        issues = self.execution.detect_external_intervention(self.symbol, snap)
        local_after = self.state_machine.get(self.symbol).state.value

        open_inflight = self.runtime_store.load_open_inflight_intents()
        symbol_inflight = [row for row in open_inflight if str(getattr(row, "symbol", "")).replace("/", "").upper() == self.symbol]
        effective_open = int(positions_summary.get("effective_open_positions_count", 0))
        raw_positions = int(positions_summary.get("raw_positions_count", 0))
        zero_placeholders = [dict(item) for item in (positions_summary.get("zero_size_placeholder_positions") or [])]
        normalized_positions = [dict(item) for item in (positions_summary.get("effective_positions") or [])]

        return {
            "scenario": str(scenario_name),
            "ts": self._now_iso(),
            "local_state_before_reconcile": str(local_before),
            "local_state": str(local_after),
            "exchange_positions_count": effective_open,
            "exchange_positions_raw_count": raw_positions,
            "exchange_effective_open_positions_count": effective_open,
            "exchange_effective_open_positions": normalized_positions,
            "exchange_zero_size_placeholder_positions_count": int(len(zero_placeholders)),
            "exchange_zero_size_placeholder_positions": zero_placeholders,
            "position_size_epsilon": float(positions_summary.get("position_size_epsilon", POSITION_SIZE_EPSILON)),
            "exchange_open_orders_count": int(len(snap.open_orders)),
            "unresolved_intervention_issues": [str(item) for item in (issues or [])],
            "inflight_intents_count": int(len(symbol_inflight)),
            "inflight_intent_keys": [str(getattr(row, "intent_key", "")) for row in symbol_inflight[:10]],
            "persistence": self._persistence_snapshot_summary(),
        }

    def _reset_scenario_buffers(self):
        self._captured_events.clear()
        self._scenario_ws_start_idx = len(self._normalized_ws_events)

    def _clear_runtime_for_scenario_isolation(self, *, clear_inflight: bool = True) -> dict[str, Any]:
        cleared_inflight = 0
        if clear_inflight:
            cleared_inflight = int(self.runtime_store.clear_inflight_intents(symbol=self.symbol))

        cleared_idempotency = int(self.runtime_store.clear_idempotency_keys())

        resetter = getattr(self.execution, "reset_idempotency_for_validation", None)
        if callable(resetter):
            resetter()

        self._reset_scenario_buffers()
        return {
            "cleared_inflight": int(cleared_inflight),
            "cleared_idempotency": int(cleared_idempotency),
            "buffers_reset": True,
            "ws_window_anchor": int(getattr(self, "_scenario_ws_start_idx", len(self._normalized_ws_events))),
        }

    def _can_validation_reset(self, state: dict[str, Any]) -> bool:
        positions_count = int(state.get("exchange_positions_count", 0))
        open_orders_count = int(state.get("exchange_open_orders_count", 0))
        issues = [str(item) for item in (state.get("unresolved_intervention_issues") or [])]
        local_state = str(state.get("local_state") or "")
        local_before = str(state.get("local_state_before_reconcile") or "")
        return (
            positions_count == 0
            and open_orders_count == 0
            and not issues
            and (
                local_state in (TradeState.HALTED.value, TradeState.RECOVERING.value)
                or local_before in (TradeState.HALTED.value, TradeState.RECOVERING.value)
            )
        )

    def _validation_safe_reset_if_flat(self, scenario_name: str, state_before: dict[str, Any]) -> dict[str, Any]:
        if not self._can_validation_reset(state_before):
            return {
                "applied": False,
                "reason": "not_required_or_unsafe",
                "before": state_before,
                "after": state_before,
                "runtime_clear": {
                    "cleared_inflight": 0,
                    "cleared_idempotency": 0,
                    "buffers_reset": False,
                },
            }

        self.state_machine.transition(self.symbol, TradeState.FLAT, f"validation_mode_safe_reset_{scenario_name}")
        runtime_clear = self._clear_runtime_for_scenario_isolation(clear_inflight=True)
        after = self._collect_scenario_start_state(f"{scenario_name}_after_validation_reset")
        return {
            "applied": True,
            "reason": "flat_exchange_truth_reset_to_flat",
            "before": state_before,
            "after": after,
            "runtime_clear": runtime_clear,
        }

    def _prepare_executable_scenario(self, scenario_name: str) -> tuple[bool, dict[str, Any]]:
        before = self._collect_scenario_start_state(f"{scenario_name}_before")
        reset_info = self._validation_safe_reset_if_flat(scenario_name, before)
        pre = reset_info.get("after") if bool(reset_info.get("applied")) else before

        local_state = str(pre.get("local_state") or "")
        positions_count = int(pre.get("exchange_positions_count", 0))
        open_orders_count = int(pre.get("exchange_open_orders_count", 0))
        issues = [str(item) for item in (pre.get("unresolved_intervention_issues") or [])]

        if (
            local_state != TradeState.FLAT.value
            or positions_count > 0
            or open_orders_count > 0
            or bool(issues)
        ):
            return False, {
                "reason": "scenario_start_not_clean",
                "starting_state": local_state,
                "validation_reset_applied": bool(reset_info.get("applied")),
                "exchange_positions_before": int(before.get("exchange_positions_count", 0)),
                "exchange_open_orders_before": int(before.get("exchange_open_orders_count", 0)),
                "unresolved_intervention_issues_before": [str(item) for item in (before.get("unresolved_intervention_issues") or [])],
                "inflight_intents_before": int(before.get("inflight_intents_count", 0)),
                "persistence_snapshot": before.get("persistence", {}),
                "scenario_start": pre,
                "validation_reset": reset_info,
            }

        isolation = self._clear_runtime_for_scenario_isolation(clear_inflight=True)
        start_ready = self._collect_scenario_start_state(f"{scenario_name}_ready")
        ready_state = str(start_ready.get("local_state") or "")
        ready_positions = int(start_ready.get("exchange_positions_count", 0))
        ready_orders = int(start_ready.get("exchange_open_orders_count", 0))
        ready_issues = [str(item) for item in (start_ready.get("unresolved_intervention_issues") or [])]

        if (
            ready_state != TradeState.FLAT.value
            or ready_positions > 0
            or ready_orders > 0
            or bool(ready_issues)
        ):
            return False, {
                "reason": "scenario_isolation_failed",
                "starting_state": ready_state,
                "validation_reset_applied": bool(reset_info.get("applied")),
                "exchange_positions_before": int(before.get("exchange_positions_count", 0)),
                "exchange_open_orders_before": int(before.get("exchange_open_orders_count", 0)),
                "unresolved_intervention_issues_before": [str(item) for item in (before.get("unresolved_intervention_issues") or [])],
                "inflight_intents_before": int(before.get("inflight_intents_count", 0)),
                "persistence_snapshot": start_ready.get("persistence", {}),
                "scenario_start": start_ready,
                "validation_reset": reset_info,
                "scenario_isolation": isolation,
            }

        return True, {
            "starting_state": ready_state,
            "validation_reset_applied": bool(reset_info.get("applied")),
            "exchange_positions_before": int(before.get("exchange_positions_count", 0)),
            "exchange_open_orders_before": int(before.get("exchange_open_orders_count", 0)),
            "unresolved_intervention_issues_before": [str(item) for item in (before.get("unresolved_intervention_issues") or [])],
            "inflight_intents_before": int(before.get("inflight_intents_count", 0)),
            "scenario_start": start_ready,
            "validation_reset": reset_info,
            "scenario_isolation": isolation,
        }

    def _state_clean_invariant(self, state: dict[str, Any]) -> tuple[bool, str]:
        local_state = str(state.get("local_state") or "")
        positions_count = int(state.get("exchange_positions_count", 0))
        open_orders_count = int(state.get("exchange_open_orders_count", 0))
        issues = [str(item) for item in (state.get("unresolved_intervention_issues") or [])]

        if positions_count > 0:
            return False, "open_positions_remaining"
        if open_orders_count > 0:
            return False, "open_orders_remaining"
        if issues:
            return False, "unresolved_intervention_issues"
        if local_state != TradeState.FLAT.value:
            return False, f"local_state_not_flat:{local_state}"
        return True, "flat_clean"

    def _reconcile_until_flat(self, *, scenario_name: str, timeout_sec: float = 8.0, poll_sec: float = 0.5) -> dict[str, Any]:
        started = time.time()
        attempts = 0
        last_state: dict[str, Any] = {}

        while True:
            attempts += 1
            last_state = self._collect_scenario_start_state(f"{scenario_name}_reconcile")
            ok, reason = self._state_clean_invariant(last_state)
            if ok:
                return {
                    "flat_confirmed": True,
                    "reason": reason,
                    "attempts": attempts,
                    "elapsed_sec": round(time.time() - started, 3),
                    "state": last_state,
                }

            if (time.time() - started) >= max(0.0, float(timeout_sec)):
                break
            time.sleep(max(0.0, float(poll_sec)))

        ok, reason = self._state_clean_invariant(last_state)
        return {
            "flat_confirmed": bool(ok),
            "reason": str(reason),
            "attempts": attempts,
            "elapsed_sec": round(time.time() - started, 3),
            "state": last_state,
        }

    def _post_scenario_cleanup_state(self, scenario_name: str) -> dict[str, Any]:
        cleanup = self._safety_cleanup()
        post = self._collect_scenario_start_state(f"{scenario_name}_post_cleanup")
        invariant_ok, invariant_reason = self._state_clean_invariant(post)
        return {
            "cleanup": cleanup,
            "state": post,
            "consistency_classification": str((cleanup or {}).get("consistency_status") or ""),
            "invariant_ok": bool(invariant_ok),
            "invariant_reason": str(invariant_reason),
        }

    def scenario_startup_preflight(self):
        details: dict[str, Any] = {
            "mode": self.cfg.mode,
            "testnet": bool(self.cfg.adapter.testnet),
            "dry_run": bool(self.cfg.adapter.dry_run),
            "symbol": self.symbol,
            "max_notional_usdt": self.max_notional_usdt,
            "live_trading_enabled": bool(self.cfg.flags.live_trading_enabled),
        }

        if not self.cfg.adapter.api_key or not self.cfg.adapter.api_secret:
            details["reason"] = "missing_bybit_credentials"
            return STATUS_BLOCKED, details

        if self.cfg.mode != "testnet":
            details["reason"] = "BOT_RUNTIME_MODE must be testnet"
            return STATUS_FAIL, details
        if self.cfg.adapter.dry_run:
            details["reason"] = "dry_run must be false for testnet validation"
            return STATUS_FAIL, details
        if not self.cfg.adapter.testnet:
            details["reason"] = "BYBIT_TESTNET must be true for testnet validation"
            return STATUS_FAIL, details

        account = self.adapter.get_account()
        info = self.adapter.get_account_mode_details()
        positions_meta = self.adapter.get_positions_metadata(self.symbol)
        open_orders_meta = self.adapter.get_open_orders_metadata(self.symbol)
        positions = self.adapter.get_positions(self.symbol)
        open_orders = self.adapter.get_open_orders(self.symbol)

        try:
            rules = self.adapter.get_instrument_rules(self.symbol)
            mark_price = float(self.adapter.get_mark_price(self.symbol))
            symbol_available = bool(rules.min_qty > 0 and rules.qty_step > 0 and rules.tick_size > 0 and rules.min_notional > 0)
        except Exception as exc:
            details["reason"] = f"symbol_unavailable:{type(exc).__name__}:{exc}"
            return STATUS_FAIL, details

        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        active_position_rows = [
            row
            for row in positions_meta
            if isinstance(row, dict) and _to_float(row.get("size"), 0.0) > 0
        ]
        margin_mode_values = sorted(
            set(
                str(row.get("tradeMode"))
                for row in active_position_rows
                if str(row.get("tradeMode", "")).strip()
            )
        )
        leverage_values = sorted(
            set(
                str(row.get("leverage"))
                for row in positions_meta
                if isinstance(row, dict) and str(row.get("leverage", "")).strip()
            )
        )

        max_pos_leverage = 0.0
        for row in active_position_rows:
            max_pos_leverage = max(max_pos_leverage, _to_float(row.get("leverage"), 0.0))

        now_ms = int(time.time() * 1000)
        stale_order_ids: list[str] = []
        for row in open_orders_meta:
            if not isinstance(row, dict):
                continue
            created_ms = int(_to_float(row.get("createdTime"), 0.0))
            if created_ms <= 0:
                continue
            age_sec = (now_ms - created_ms) / 1000.0
            if age_sec > float(self.cfg.stale_open_order_sec):
                stale_order_ids.append(str(row.get("orderId") or row.get("orderLinkId") or ""))

        restriction_hits: list[str] = []
        tokens = ("maint", "suspend", "restrict", "freeze", "disabled", "readonly", "halt")
        for key, value in info.items():
            blob = f"{key}:{value}".lower()
            if any(tok in blob for tok in tokens):
                restriction_hits.append(str(key))

        details.update(
            {
                "account": {
                    "equity_usdt": float(account.equity_usdt),
                    "available_balance_usdt": float(account.available_balance_usdt),
                },
                "account_mode": info.get("unifiedMarginStatus"),
                "margin_mode_values": margin_mode_values,
                "leverage_values": leverage_values,
                "max_position_leverage": max_pos_leverage,
                "positions_meta_count": len(positions_meta),
                "open_orders_meta_count": len(open_orders_meta),
                "open_positions_count": len(positions),
                "effective_open_positions_count": len(positions),
                "open_orders_count": len(open_orders),
                "stale_open_order_ids": stale_order_ids,
                "symbol_available": symbol_available,
                "mark_price": mark_price,
                "instrument_rules": {
                    "tick_size": float(rules.tick_size),
                    "qty_step": float(rules.qty_step),
                    "min_qty": float(rules.min_qty),
                    "min_notional": float(rules.min_notional),
                },
                "metadata_health": self.adapter.metadata_health(),
                "restriction_hits": sorted(set(restriction_hits)),
            }
        )

        if int(info.get("retCode", 1)) != 0:
            details["reason"] = f"account_info_retcode:{info.get('retCode')}"
            return STATUS_FAIL, details

        if info.get("unifiedMarginStatus") in (None, ""):
            details["reason"] = "account_mode_missing"
            return STATUS_FAIL, details

        if float(account.equity_usdt) <= 0:
            details["reason"] = "non_positive_testnet_equity"
            return STATUS_FAIL, details

        if not symbol_available or mark_price <= 0:
            details["reason"] = "symbol_unavailable_or_bad_mark_price"
            return STATUS_FAIL, details

        if len(positions) > 0:
            details["reason"] = "unexpected_positions_present"
            return STATUS_FAIL, details

        if len(open_orders) > 0:
            details["reason"] = "unexpected_open_orders_present"
            return STATUS_FAIL, details

        if stale_order_ids:
            details["reason"] = "stale_open_orders_present"
            return STATUS_FAIL, details

        if max_pos_leverage > float(self.cfg.risk_limits.max_leverage) + 1e-9:
            details["reason"] = "leverage_above_risk_limit"
            return STATUS_FAIL, details

        if restriction_hits:
            details["reason"] = "account_restricted_or_maintenance"
            return STATUS_FAIL, details

        return STATUS_PASS, details
    def scenario_mode_gate_semantics(self):
        details = {
            "mode": self.cfg.mode,
            "testnet": bool(self.cfg.adapter.testnet),
            "dry_run": bool(self.cfg.adapter.dry_run),
            "live_trading_enabled": bool(self.cfg.flags.live_trading_enabled),
            "execute_orders": bool(self.execute_orders),
            "execute_orders_allowed_without_live": bool(
                self.cfg.mode == "testnet" and not self.cfg.adapter.dry_run and self.cfg.adapter.testnet
            ),
        }

        if self.cfg.mode != "testnet":
            details["reason"] = "mode_not_testnet"
            return STATUS_FAIL, details
        if self.cfg.flags.live_trading_enabled:
            details["reason"] = "live_toggle_must_be_false_in_testnet"
            return STATUS_FAIL, details
        if self.cfg.adapter.dry_run:
            details["reason"] = "testnet_mode_requires_dry_run_false"
            return STATUS_FAIL, details
        if not self.cfg.adapter.testnet:
            details["reason"] = "testnet_mode_requires_bybit_testnet_true"
            return STATUS_FAIL, details
        return STATUS_PASS, details
    def scenario_tiny_capped_order_lifecycle(self):
        if not self.execute_orders:
            return STATUS_SKIP, {"reason": "order_execution_disabled"}

        details = self._run_tiny_lifecycle(tag="tiny")

        post_exit_reconcile = details.get("post_exit_reconcile") if isinstance(details, dict) else {}
        if not bool((post_exit_reconcile or {}).get("flat_confirmed")):
            details["reason"] = f"post_exit_not_flat:{(post_exit_reconcile or {}).get('reason', 'unknown')}"
            return STATUS_FAIL, details

        if not bool(details.get("exit_qty_matches_entry_fill")):
            details["reason"] = "exit_requested_qty_mismatch_entry_fill"
            return STATUS_FAIL, details

        if not bool(details.get("final_cleanup_invariant_ok")):
            details["reason"] = f"final_cleanup_not_clean:{details.get('final_cleanup_invariant_reason', 'unknown')}"
            return STATUS_FAIL, details

        if int(details.get("positions_after_cleanup", 0)) != 0 or int(details.get("open_orders_after_cleanup", 0)) != 0:
            details["reason"] = "residual_after_cleanup"
            return STATUS_FAIL, details

        return STATUS_PASS, details

    def scenario_private_ws_normalization(self):
        if not self.cfg.adapter.ws_enabled or not self.cfg.adapter.ws_private_enabled:
            return STATUS_BLOCKED, {"reason": "private_ws_disabled"}

        scenario_start_idx = len(self._normalized_ws_events)
        scenario_start_ts = time.time()
        lifecycle_error = ""
        lifecycle_ok = False
        setup_diag: dict[str, Any] = {}

        try:
            if self.execute_orders:
                lifecycle = self._run_tiny_lifecycle(tag="ws")
                lifecycle_ok = True
                setup_diag = dict((lifecycle or {}).get("entry_setup") or {})
            else:
                _ = self._drain_ws_events(2.0)
        except Exception as exc:
            lifecycle_error = f"{type(exc).__name__}:{exc}"
            if self.execute_orders:
                try:
                    setup_ctx = self._tiny_entry_setup_context(force_poll_snapshot=True)
                    setup_diag = dict(setup_ctx.get("diagnostics") or {})
                except Exception as diag_exc:
                    setup_diag = {
                        "diagnostics_error": f"{type(diag_exc).__name__}:{diag_exc}",
                    }

        _ = self._drain_ws_events(2.0)

        events = [
            evt for evt in self._normalized_ws_events[scenario_start_idx:]
            if isinstance(evt, dict)
        ]
        private_types = {"ACCOUNT", "POSITION", "ORDER"}
        event_types = sorted(set(str(evt.get("event_type") or "") for evt in events))
        private_event_count = sum(1 for evt in events if str(evt.get("event_type") or "") in private_types)
        has_private = private_event_count > 0

        ts_values: list[float] = []
        for evt in events:
            try:
                ts_values.append(float(evt.get("ts")))
            except (TypeError, ValueError):
                continue
        ts_values = sorted(ts_values)
        first_ts = datetime.fromtimestamp(ts_values[0], timezone.utc).isoformat() if ts_values else ""
        last_ts = datetime.fromtimestamp(ts_values[-1], timezone.utc).isoformat() if ts_values else ""

        details = {
            "captured_count": len(events),
            "event_types": event_types,
            "has_private_events": bool(has_private),
            "total_normalized_events_during_scenario": len(events),
            "private_event_count": int(private_event_count),
            "first_event_ts": first_ts,
            "last_event_ts": last_ts,
            "event_source_buffer": "_normalized_ws_events",
            "scenario_window_sec": round(time.time() - scenario_start_ts, 3),
            "lifecycle_error": lifecycle_error,
            "lifecycle_ok": bool(lifecycle_ok),
            "requested_qty": float(setup_diag.get("requested_qty", 0.0) or 0.0),
            "requested_notional": float(setup_diag.get("requested_notional", 0.0) or 0.0),
            "min_qty": float(setup_diag.get("min_qty", 0.0) or 0.0),
            "min_notional": float(setup_diag.get("min_notional", 0.0) or 0.0),
            "qty_step": float(setup_diag.get("qty_step", 0.0) or 0.0),
            "available_balance_snapshot": float(setup_diag.get("available_balance_snapshot", 0.0) or 0.0),
            "equity_snapshot": float(setup_diag.get("equity_snapshot", 0.0) or 0.0),
            "mark_price_used": float(setup_diag.get("mark_price_used", 0.0) or 0.0),
            "affordability_inputs": dict(setup_diag.get("affordability_inputs") or {}),
        }

        if self.execute_orders and not lifecycle_ok:
            details["reason"] = f"private_ws_setup_failed:{lifecycle_error or 'unknown'}"
            return STATUS_FAIL, details

        if not has_private:
            details["reason"] = "private_events_not_observed_in_scenario_window"
            return STATUS_FAIL, details
        return STATUS_PASS, details

    def scenario_reconnect_and_snapshot_fallback(self):
        if not self.cfg.adapter.ws_enabled:
            return STATUS_BLOCKED, {"reason": "ws_disabled"}

        self.adapter.force_ws_reconnect()
        events = self._drain_ws_events(8.0)
        event_types = [str(evt.event_type.value) for evt in events]

        has_reconnect = "RECONNECTING" in event_types
        has_connected = "CONNECTED" in event_types
        has_snapshot_required = "SNAPSHOT_REQUIRED" in event_types

        health_before = self.sync.health()
        _ = self._snapshot()
        health_after = self.sync.health()

        details = {
            "event_types": sorted(set(event_types)),
            "reconnecting": has_reconnect,
            "connected": has_connected,
            "snapshot_required_event": has_snapshot_required,
            "health_before": {
                "ws_connected": health_before.ws_connected,
                "ws_stale": health_before.ws_stale,
                "fallback_polling": health_before.fallback_polling,
                "snapshot_required": health_before.snapshot_required,
            },
            "health_after": {
                "ws_connected": health_after.ws_connected,
                "ws_stale": health_after.ws_stale,
                "fallback_polling": health_after.fallback_polling,
                "snapshot_required": health_after.snapshot_required,
            },
        }

        if not (has_reconnect and has_connected):
            return STATUS_FAIL, details
        if health_before.snapshot_required and not health_after.snapshot_required:
            return STATUS_PASS, details
        if has_snapshot_required and health_after.snapshot_required:
            return STATUS_FAIL, details
        return STATUS_PASS, details

    def scenario_no_duplicate_execution_under_reconnect_retry(self):
        if not self.execute_orders:
            return STATUS_SKIP, {"reason": "order_execution_disabled"}

        _ = self._safety_cleanup()
        snap = self._snapshot()
        rules = self.adapter.get_instrument_rules(self.symbol)
        mark = self.adapter.get_mark_price(self.symbol)
        if mark <= 0:
            raise RuntimeError("invalid_mark_price")
        qty_cap = self._compute_tiny_qty(mark, rules.min_qty, rules.qty_step, rules.min_notional)

        intent = self._entry_intent(mark)
        decision = self.risk.evaluate(
            intent=intent,
            account=snap.account,
            existing_positions=snap.positions,
            mark_price=mark,
            rules=rules,
        )
        if not decision.approved:
            raise RuntimeError(f"risk_rejected:{decision.reason}")
        decision = RiskDecision(approved=True, reason=decision.reason, quantity=min(decision.quantity, qty_cap))

        self.adapter.force_ws_reconnect()
        out_first = self.execution.execute(intent=intent, risk=decision, snapshot=snap, mark_price=mark)
        out_second = self.execution.execute(intent=intent, risk=decision, snapshot=snap, mark_price=mark)

        cleanup = self._safety_cleanup()
        details = {
            "first": {
                "accepted": out_first.accepted,
                "status": out_first.status,
                "reason": out_first.reason,
                "order_link_id": out_first.order_link_id,
            },
            "second": {
                "accepted": out_second.accepted,
                "status": out_second.status,
                "reason": out_second.reason,
                "order_link_id": out_second.order_link_id,
            },
            "cleanup": cleanup,
        }

        if not out_first.accepted:
            return STATUS_FAIL, details
        if out_second.reason != "duplicate_intent":
            return STATUS_FAIL, details
        return STATUS_PASS, details

    def _stack_from_db(self, db_path: str, *, stale_open_order_sec: int = 120):
        store = RuntimeStore(db_path)
        sm = StateMachine(persistence=store)
        risk = RiskEngine(self.cfg.risk_limits, persistence=store)
        ex = ExecutionEngine(
            adapter=self.adapter,
            state_machine=sm,
            hedge_mode=self.cfg.adapter.hedge_mode,
            stop_loss_required=self.cfg.risk_limits.require_stop_loss,
            require_reconciliation=self.cfg.flags.reconciliation_required,
            stop_attach_grace_sec=self.cfg.stop_attach_grace_sec,
            stale_open_order_sec=stale_open_order_sec,
            max_exchange_retries=self.cfg.max_exchange_retries,
            persistence=store,
        )
        return store, sm, risk, ex

    def scenario_restart_after_entry_submit_before_confirm(self):
        if not self.execute_orders:
            return STATUS_SKIP, {"reason": "order_execution_disabled"}

        _ = self._safety_cleanup()
        mark = self.adapter.get_mark_price(self.symbol)
        rules = self.adapter.get_instrument_rules(self.symbol)
        qty = self._compute_tiny_qty(mark, rules.min_qty, rules.qty_step, rules.min_notional)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            client_order_id = f"v2-val-crash-{abs(hash((self.symbol, time.time()))) % 10**10}"

            direct = self.adapter.place_market_order(
                OrderIntent(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    qty=qty,
                    reduce_only=False,
                    position_idx=self.adapter.position_idx_for_side(PositionSide.LONG, self.cfg.adapter.hedge_mode),
                    client_order_id=client_order_id,
                )
            )
            if not direct.success:
                raise RuntimeError(f"direct_entry_failed:{direct.error}")

            store, sm, _, ex = self._stack_from_db(db_path)
            try:
                store.upsert_inflight_intent(
                    intent_key="crash_entry",
                    symbol=self.symbol,
                    action=IntentAction.LONG_ENTRY.value,
                    payload={
                        "stop_loss": float(mark * 0.99),
                        "take_profit": float(mark * 1.01),
                        "position_idx": self.adapter.position_idx_for_side(PositionSide.LONG, self.cfg.adapter.hedge_mode),
                        "requested_qty": float(qty),
                        "client_order_id": client_order_id,
                    },
                    status="pending_submission",
                )
            finally:
                store.close()

            store2, sm2, _, ex2 = self._stack_from_db(db_path)
            try:
                snap = self.reconciler.snapshot(self.symbol)
                sm2.reconcile(self.symbol, snap.positions, snap.open_orders)
                ex2.recover_from_restart(self.symbol, snap)
                snap2 = self.reconciler.snapshot(self.symbol)
                state = sm2.reconcile(self.symbol, snap2.positions, snap2.open_orders).state.value
                open_inflight = len(store2.load_open_inflight_intents())
            finally:
                store2.close()

        cleanup = self._safety_cleanup()
        details = {
            "state_after_recovery": state,
            "open_inflight": open_inflight,
            "cleanup": cleanup,
        }
        if state == TradeState.HALTED.value:
            return STATUS_FAIL, details
        return STATUS_PASS, details

    def scenario_restart_after_partial_fill(self):
        if not self.execute_orders:
            return STATUS_SKIP, {"reason": "order_execution_disabled"}

        _ = self._safety_cleanup()
        mark = self.adapter.get_mark_price(self.symbol)
        rules = self.adapter.get_instrument_rules(self.symbol)
        total_qty = self._compute_tiny_qty(mark, rules.min_qty, rules.qty_step, rules.min_notional)
        fill_qty = self.adapter.round_qty(max(rules.min_qty, total_qty * 0.5), rules.qty_step)
        remaining = self.adapter.round_qty(max(rules.min_qty, total_qty - fill_qty), rules.qty_step)
        if remaining <= 0:
            remaining = rules.min_qty

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            base_id = abs(hash((self.symbol, self.run_id, time.time_ns()))) % 10**10
            entry_order_link_id = f"v2-val-pf-e-{base_id}"
            partial_order_link_id = f"v2-val-pf-l-{base_id}"
            position_idx = self.adapter.position_idx_for_side(PositionSide.LONG, self.cfg.adapter.hedge_mode)

            first = self.adapter.place_market_order(
                OrderIntent(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    qty=fill_qty,
                    reduce_only=False,
                    position_idx=position_idx,
                    client_order_id=entry_order_link_id,
                )
            )
            if not first.success:
                raise RuntimeError(f"partial_base_entry_failed:{first.error}")

            lim = self._place_limit_order_with_reconcile(
                symbol=self.symbol,
                side=OrderSide.BUY,
                qty=remaining,
                price=mark * 0.9,
                reduce_only=False,
                position_idx=position_idx,
                client_order_id=partial_order_link_id,
            )
            if not lim.success:
                raise RuntimeError(f"partial_limit_order_failed:{lim.error}")

            store, sm, _, ex = self._stack_from_db(db_path)
            try:
                store.upsert_inflight_intent(
                    intent_key="partial_entry",
                    symbol=self.symbol,
                    action=IntentAction.LONG_ENTRY.value,
                    payload={
                        "stop_loss": float(mark * 0.99),
                        "take_profit": float(mark * 1.01),
                        "position_idx": position_idx,
                        "requested_qty": float(fill_qty + remaining),
                        "client_order_id": partial_order_link_id,
                        "entry_client_order_id": entry_order_link_id,
                        "grace_deadline_ts": time.time() + self.cfg.stop_attach_grace_sec,
                    },
                    status="partial_fill",
                )
            finally:
                store.close()

            store2, sm2, _, ex2 = self._stack_from_db(db_path)
            try:
                snap = self.reconciler.snapshot(self.symbol)
                sm2.reconcile(self.symbol, snap.positions, snap.open_orders)
                ex2.recover_from_restart(self.symbol, snap)
                snap2 = self.reconciler.snapshot(self.symbol)
                state = sm2.reconcile(self.symbol, snap2.positions, snap2.open_orders).state.value
                remaining_orders = [o for o in self.adapter.get_open_orders(self.symbol) if o.order_link_id == partial_order_link_id]
            finally:
                store2.close()

        cleanup = self._safety_cleanup()
        details = {
            "state_after_recovery": state,
            "remaining_orders_for_intent": len(remaining_orders),
            "entry_order_link_id": entry_order_link_id,
            "partial_order_link_id": partial_order_link_id,
            "partial_order_reconciled_existing": bool(isinstance(lim.raw, dict) and lim.raw.get("recovered_existing_order")),
            "cleanup": cleanup,
        }
        if state == TradeState.HALTED.value:
            return STATUS_FAIL, details
        if len(remaining_orders) > 0:
            return STATUS_FAIL, details
        return STATUS_PASS, details

    def scenario_restart_after_stop_attach_failure(self):
        if not self.execute_orders:
            return STATUS_SKIP, {"reason": "order_execution_disabled"}

        _ = self._safety_cleanup()
        mark = self.adapter.get_mark_price(self.symbol)
        rules = self.adapter.get_instrument_rules(self.symbol)
        qty = self._compute_tiny_qty(mark, rules.min_qty, rules.qty_step, rules.min_notional)
        position_idx = self.adapter.position_idx_for_side(PositionSide.LONG, self.cfg.adapter.hedge_mode)

        direct = self.adapter.place_market_order(
            OrderIntent(
                symbol=self.symbol,
                side=OrderSide.BUY,
                qty=qty,
                reduce_only=False,
                position_idx=position_idx,
                client_order_id=f"v2-val-stopfail-{abs(hash((self.symbol, time.time()))) % 10**10}",
            )
        )
        if not direct.success:
            raise RuntimeError(f"direct_entry_failed:{direct.error}")

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store, sm, _, ex = self._stack_from_db(db_path)
            try:
                store.upsert_inflight_intent(
                    intent_key="stop_fail",
                    symbol=self.symbol,
                    action=IntentAction.LONG_ENTRY.value,
                    payload={
                        "stop_loss": float(mark * 1.2),
                        "take_profit": float(mark * 1.3),
                        "position_idx": position_idx,
                        "requested_qty": float(qty),
                        "client_order_id": "v2-val-stopfail",
                        "grace_deadline_ts": time.time() + self.cfg.stop_attach_grace_sec,
                    },
                    status="naked_exposure",
                )
            finally:
                store.close()

            store2, sm2, _, ex2 = self._stack_from_db(db_path)
            try:
                snap = self.reconciler.snapshot(self.symbol)
                sm2.reconcile(self.symbol, snap.positions, snap.open_orders)
                ex2.recover_from_restart(self.symbol, snap)
                snap2 = self.reconciler.snapshot(self.symbol)
                state = sm2.reconcile(self.symbol, snap2.positions, snap2.open_orders).state.value
                open_positions = len(self.adapter.get_positions(self.symbol))
            finally:
                store2.close()

        cleanup = self._safety_cleanup()
        details = {
            "state_after_recovery": state,
            "positions_after_recovery": open_positions,
            "cleanup": cleanup,
        }
        if state == TradeState.HALTED.value and open_positions > 0:
            return STATUS_FAIL, details
        return STATUS_PASS, details

    def scenario_restart_with_stale_open_orders(self):
        if not self.execute_orders:
            return STATUS_SKIP, {"reason": "order_execution_disabled"}

        _ = self._safety_cleanup()
        mark = self.adapter.get_mark_price(self.symbol)
        rules = self.adapter.get_instrument_rules(self.symbol)
        qty = self._compute_tiny_qty(mark, rules.min_qty, rules.qty_step, rules.min_notional)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            client_order_id = f"v2-val-stale-{abs(hash((self.symbol, time.time()))) % 10**10}"
            lim = self.adapter.place_limit_order(
                symbol=self.symbol,
                side=OrderSide.BUY,
                qty=qty,
                price=mark * 0.9,
                reduce_only=False,
                position_idx=self.adapter.position_idx_for_side(PositionSide.LONG, self.cfg.adapter.hedge_mode),
                client_order_id=client_order_id,
            )
            if not lim.success:
                raise RuntimeError(f"stale_limit_create_failed:{lim.error}")

            store, sm, _, ex = self._stack_from_db(db_path, stale_open_order_sec=10)
            try:
                store.upsert_inflight_intent(
                    intent_key="stale_order",
                    symbol=self.symbol,
                    action=IntentAction.LONG_ENTRY.value,
                    payload={
                        "stop_loss": float(mark * 0.99),
                        "take_profit": float(mark * 1.01),
                        "position_idx": self.adapter.position_idx_for_side(PositionSide.LONG, self.cfg.adapter.hedge_mode),
                        "requested_qty": float(qty),
                        "client_order_id": client_order_id,
                    },
                    status="pending_submission",
                )
            finally:
                store.close()

            time.sleep(11.5)
            store2, sm2, _, ex2 = self._stack_from_db(db_path, stale_open_order_sec=10)
            try:
                snap = self.reconciler.snapshot(self.symbol)
                sm2.reconcile(self.symbol, snap.positions, snap.open_orders)
                ex2.recover_from_restart(self.symbol, snap)
                remaining = [o for o in self.adapter.get_open_orders(self.symbol) if o.order_link_id == client_order_id]
                state = sm2.get(self.symbol).state.value
            finally:
                store2.close()

        cleanup = self._safety_cleanup()
        details = {
            "state_after_recovery": state,
            "remaining_orders": len(remaining),
            "cleanup": cleanup,
        }
        if len(remaining) > 0:
            return STATUS_FAIL, details
        return STATUS_PASS, details

    def scenario_restart_after_manual_exchange_intervention(self):
        if not self.execute_orders:
            return STATUS_SKIP, {"reason": "order_execution_disabled"}

        _ = self._safety_cleanup()
        manual_setup = self._open_external_manual_exposure(scenario_tag="intervention")

        if not bool(manual_setup.get("effective_position_confirmed")):
            cleanup = self._safety_cleanup()
            details = {
                "reason": "manual_intervention_setup_failed:no_effective_external_position",
                **manual_setup,
                "cleanup": cleanup,
            }
            return STATUS_FAIL, details

        snap_before_detection = self._snapshot()
        before_summary = summarize_positions(
            snap_before_detection.positions,
            symbol=self.symbol,
            size_epsilon=POSITION_SIZE_EPSILON,
        )

        self.state_machine.transition(self.symbol, TradeState.FLAT, "manual_intervention_test")
        issues = self.execution.detect_external_intervention(self.symbol, snap_before_detection)
        state = self.state_machine.get(self.symbol).state.value

        snap_after_detection = self._snapshot()
        after_summary = summarize_positions(
            snap_after_detection.positions,
            symbol=self.symbol,
            size_epsilon=POSITION_SIZE_EPSILON,
        )

        cleanup = self._safety_cleanup()
        details = {
            **manual_setup,
            "exchange_effective_open_positions_before_detection": int(before_summary.get("effective_open_positions_count", 0)),
            "exchange_effective_open_positions_after_detection": int(after_summary.get("effective_open_positions_count", 0)),
            "issues_before_cleanup": [str(item) for item in (issues or [])],
            "state_before_cleanup": str(state),
            "cleanup": cleanup,
        }
        if not issues:
            return STATUS_FAIL, details
        if state != TradeState.HALTED.value:
            return STATUS_FAIL, details
        return STATUS_PASS, details

    def scenario_halted_semantics(self):
        if not self.execute_orders:
            return STATUS_SKIP, {"reason": "order_execution_disabled"}

        _ = self._safety_cleanup()
        manual_setup = self._open_external_manual_exposure(scenario_tag="halt")

        if not bool(manual_setup.get("effective_position_confirmed")):
            cleanup = self._safety_cleanup()
            details = {
                "reason": "halted_semantics_setup_failed:no_effective_external_position",
                **manual_setup,
                "cleanup": cleanup,
            }
            return STATUS_FAIL, details

        snap_before_detection = self._snapshot()
        before_summary = summarize_positions(
            snap_before_detection.positions,
            symbol=self.symbol,
            size_epsilon=POSITION_SIZE_EPSILON,
        )

        self.state_machine.transition(self.symbol, TradeState.FLAT, "halted_semantics_test")
        issues = self.execution.detect_external_intervention(self.symbol, snap_before_detection)
        state = self.state_machine.get(self.symbol).state.value

        cleanup = self._safety_cleanup()
        details = {
            **manual_setup,
            "effective_open_positions_before_detection": int(before_summary.get("effective_open_positions_count", 0)),
            "issues_detected": [str(item) for item in (issues or [])],
            "state_after_detection": str(state),
            "cleanup": cleanup,
        }
        if not issues:
            return STATUS_FAIL, details
        if state != TradeState.HALTED.value:
            return STATUS_FAIL, details
        return STATUS_PASS, details

    def scenario_v2_suite_target_runtime(self):
        if not self.run_full_suite:
            return STATUS_SKIP, {"reason": "full_suite_disabled"}

        cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests/v2", "-p", "test_*.py", "-v"]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        merged = "\n".join([stdout, stderr])
        skip_match = re.search(r"skipped=(\d+)", merged)
        skipped = int(skip_match.group(1)) if skip_match else 0

        details = {
            "returncode": int(proc.returncode),
            "skipped": skipped,
            "command": " ".join(cmd),
        }
        if proc.returncode != 0:
            details["tail"] = "\n".join(merged.strip().splitlines()[-20:])
            return STATUS_FAIL, details
        if skipped > 0:
            details["reason"] = "skipped_tests_in_target_runtime"
            return STATUS_FAIL, details
        return STATUS_PASS, details

    def scenario_soak_and_chaos(self):
        if self.soak_seconds <= 0 and self.chaos_cycles <= 0:
            return STATUS_SKIP, {"reason": "soak_and_chaos_disabled"}

        if not self.cfg.adapter.ws_enabled:
            return STATUS_BLOCKED, {"reason": "ws_disabled"}

        reconnect_ok = 0
        reconnect_fail = 0
        start = time.time()

        for idx in range(self.chaos_cycles):
            self.adapter.force_ws_reconnect()
            events = self._drain_ws_events(4.0)
            types = {str(evt.event_type.value) for evt in events}
            if "RECONNECTING" in types and "CONNECTED" in types:
                reconnect_ok += 1
            else:
                reconnect_fail += 1
            _ = self._snapshot()
            _ = self.runtime_store.maintenance()
            self.logger.info("chaos_cycle=%s reconnect_ok=%s reconnect_fail=%s", idx + 1, reconnect_ok, reconnect_fail)

        while (time.time() - start) < self.soak_seconds:
            _ = self._drain_ws_events(1.0)
            _ = self._snapshot()
            _ = self.runtime_store.maintenance()
            time.sleep(0.5)

        health = self.sync.health()
        details = {
            "chaos_cycles": self.chaos_cycles,
            "reconnect_ok": reconnect_ok,
            "reconnect_fail": reconnect_fail,
            "soak_seconds": self.soak_seconds,
            "ws_connected": health.ws_connected,
            "ws_stale": health.ws_stale,
            "fallback_polling": health.fallback_polling,
            "snapshot_required": health.snapshot_required,
        }

        if reconnect_fail > 0:
            return STATUS_FAIL, details
        return STATUS_PASS, details
    def _build_artifact_bundle(self, results: list[ScenarioResult]) -> dict[str, Any]:
        ws_raw_file = self._write_jsonl(self.artifacts_dir / "ws_raw_events.jsonl", self._raw_ws_events)
        ws_norm_file = self._write_jsonl(self.artifacts_dir / "ws_normalized_events.jsonl", self._normalized_ws_events)

        decisions = self.runtime_store.load_order_decisions(limit=200000)
        transitions = self.runtime_store.load_state_transitions(limit=200000)

        order_rows = list(self._order_lifecycle_events)
        for row in decisions:
            order_rows.append(
                {
                    "ts": float(row.ts),
                    "scenario": "execution_engine",
                    "stage": "order_decision",
                    "payload": {
                        "symbol": row.symbol,
                        "action": row.action,
                        "state_before": row.state_before,
                        "risk_reason": row.risk_reason,
                        "exec_status": row.exec_status,
                        "exec_reason": row.exec_reason,
                        "order_id": row.order_id,
                        "order_link_id": row.order_link_id,
                    },
                }
            )
        order_rows = [self._json_safe(item) for item in order_rows]
        order_file = self._write_jsonl(self.artifacts_dir / "order_lifecycle.jsonl", order_rows)

        transition_rows = [
            {
                "id": int(row.id),
                "symbol": row.symbol,
                "previous_state": row.previous_state,
                "current_state": row.current_state,
                "reason": row.reason,
                "ts": float(row.ts),
            }
            for row in transitions
        ]
        state_file = self._write_jsonl(self.artifacts_dir / "state_transition_journal.jsonl", transition_rows)

        recovery_rows = [
            {
                "name": item.name,
                "status": item.status,
                "duration_sec": item.duration_sec,
                "details": self._json_safe(item.details),
                "error": item.error,
            }
            for item in results
            if item.name.startswith("restart_") or "intervention" in item.name or item.name == "halted_semantics"
        ]
        self._recovery_events = list(recovery_rows)
        recovery_file = self._write_json(self.artifacts_dir / "recovery_intervention_report.json", recovery_rows)

        ws_by_type: dict[str, int] = {}
        for item in self._normalized_ws_events:
            event_type = str(item.get("event_type") or "")
            ws_by_type[event_type] = int(ws_by_type.get(event_type, 0) + 1)

        order_status_summary: dict[str, int] = {}
        for row in decisions:
            key = str(row.exec_status or "")
            order_status_summary[key] = int(order_status_summary.get(key, 0) + 1)

        recovery_status_summary: dict[str, int] = {}
        for row in recovery_rows:
            key = str(row.get("status") or "")
            recovery_status_summary[key] = int(recovery_status_summary.get(key, 0) + 1)

        constraints = self._write_runtime_constraints_artifacts()
        runtime_manifest = collect_runtime_manifest()
        runtime_manifest["constraints"] = constraints
        runtime_manifest_file = self._write_json(self.artifacts_dir / "runtime_manifest.json", runtime_manifest)

        return {
            "runtime_manifest": runtime_manifest,
            "runtime_manifest_file": runtime_manifest_file,
            "artifacts": {
                "root_dir": str(self.artifacts_dir.resolve()),
                "ws_raw_events": {"path": ws_raw_file, "count": len(self._raw_ws_events)},
                "ws_normalized_events": {"path": ws_norm_file, "count": len(self._normalized_ws_events)},
                "order_lifecycle": {"path": order_file, "count": len(order_rows)},
                "state_transitions": {"path": state_file, "count": len(transition_rows)},
                "recovery_intervention": {"path": recovery_file, "count": len(recovery_rows)},
            },
            "websocket_health_summary": {
                "raw_events": len(self._raw_ws_events),
                "normalized_events": len(self._normalized_ws_events),
                "by_event_type": ws_by_type,
                "sync_health": self._json_safe(self.sync.health()),
            },
            "order_lifecycle_summary": {
                "harness_events": len(self._order_lifecycle_events),
                "execution_decisions": len(decisions),
                "execution_status_counts": order_status_summary,
            },
            "recovery_scenario_summary": {
                "count": len(recovery_rows),
                "status_counts": recovery_status_summary,
                "names": [str(row.get("name")) for row in recovery_rows],
            },
        }
    def run(self) -> dict[str, Any]:
        scenarios: list[tuple[str, Any, bool]] = [
            ("startup_preflight", self.scenario_startup_preflight, False),
            ("mode_gate_semantics", self.scenario_mode_gate_semantics, False),
            ("tiny_capped_order_lifecycle", self.scenario_tiny_capped_order_lifecycle, True),
            ("private_ws_normalization", self.scenario_private_ws_normalization, True),
            ("reconnect_snapshot_fallback", self.scenario_reconnect_and_snapshot_fallback, True),
            ("no_duplicate_execution_reconnect_retry", self.scenario_no_duplicate_execution_under_reconnect_retry, True),
            ("restart_after_entry_submit_before_confirm", self.scenario_restart_after_entry_submit_before_confirm, True),
            ("restart_after_partial_fill", self.scenario_restart_after_partial_fill, True),
            ("restart_after_stop_attach_failure", self.scenario_restart_after_stop_attach_failure, True),
            ("restart_with_stale_open_orders", self.scenario_restart_with_stale_open_orders, True),
            ("restart_after_manual_exchange_intervention", self.scenario_restart_after_manual_exchange_intervention, True),
            ("halted_semantics", self.scenario_halted_semantics, True),
            ("v2_suite_target_runtime", self.scenario_v2_suite_target_runtime, False),
            ("soak_and_chaos", self.scenario_soak_and_chaos, True),
        ]

        results: list[ScenarioResult] = []
        startup_status: str | None = None
        bootstrap_reset_info: dict[str, Any] = {
            "applied": False,
            "reason": "not_attempted",
        }
        executable_started = False

        for name, fn, requires_safe_preflight in scenarios:
            if requires_safe_preflight and startup_status != STATUS_PASS:
                reason = "startup_preflight_not_passed"
                details = {
                    "reason": reason,
                    "startup_preflight_status": startup_status,
                    "post_scenario_cleanup": {
                        "applied": False,
                        "reason": "scenario_not_executed_due_to_startup_gate",
                    },
                }
                result = ScenarioResult(name=name, status=STATUS_BLOCKED, duration_sec=0.0, details=details)
                results.append(result)
                self.logger.info(
                    "validation_scenario_end name=%s status=%s duration=%.3f error=%s",
                    result.name,
                    result.status,
                    result.duration_sec,
                    result.error,
                )
                continue

            scenario_start_details: dict[str, Any] = {}
            if requires_safe_preflight:
                prepared_ok, prepared_details = self._prepare_executable_scenario(name)
                scenario_start_details = dict(prepared_details)

                if not executable_started:
                    bootstrap_reset_info = scenario_start_details.get("validation_reset") or {
                        "applied": False,
                        "reason": "not_required_or_unsafe",
                    }
                    executable_started = True

                if not prepared_ok:
                    details = dict(scenario_start_details)
                    details["post_scenario_cleanup"] = {
                        "applied": False,
                        "reason": "scenario_not_executed_due_to_start_state",
                    }
                    result = ScenarioResult(name=name, status=STATUS_FAIL, duration_sec=0.0, details=details)
                    results.append(result)
                    self.logger.info(
                        "validation_scenario_end name=%s status=%s duration=%.3f error=%s",
                        result.name,
                        result.status,
                        result.duration_sec,
                        result.error,
                    )
                    continue

            self.logger.info("validation_scenario_start name=%s", name)
            result = self._run_scenario(name, fn)

            if requires_safe_preflight:
                details = result.details if isinstance(result.details, dict) else {}
                details["starting_state"] = str(scenario_start_details.get("starting_state") or "")
                details["validation_reset_applied"] = bool(scenario_start_details.get("validation_reset_applied"))
                details["exchange_positions_before"] = int(scenario_start_details.get("exchange_positions_before", 0))
                details["exchange_open_orders_before"] = int(scenario_start_details.get("exchange_open_orders_before", 0))
                details["unresolved_intervention_issues_before"] = [
                    str(item) for item in (scenario_start_details.get("unresolved_intervention_issues_before") or [])
                ]
                details["inflight_intents_before"] = int(scenario_start_details.get("inflight_intents_before", 0))
                details["scenario_start"] = scenario_start_details.get("scenario_start", {})
                details["scenario_isolation"] = scenario_start_details.get("scenario_isolation", {})
                details["validation_reset"] = scenario_start_details.get("validation_reset", {})

                post_cleanup = self._post_scenario_cleanup_state(name)
                details["post_scenario_cleanup"] = post_cleanup
                if not bool(post_cleanup.get("invariant_ok")):
                    invariant_reason = str(post_cleanup.get("invariant_reason") or "unknown")
                    details["reason"] = str(details.get("reason") or f"post_cleanup_not_clean:{invariant_reason}")
                    if result.status == STATUS_PASS:
                        result.status = STATUS_FAIL

                result.details = details

            self.logger.info(
                "validation_scenario_end name=%s status=%s duration=%.3f error=%s",
                result.name,
                result.status,
                result.duration_sec,
                result.error,
            )
            results.append(result)

            if name == "startup_preflight":
                startup_status = result.status

        status_counts: dict[str, int] = {}
        for result in results:
            status_counts[result.status] = int(status_counts.get(result.status, 0) + 1)

        blockers = []
        for result in results:
            if result.status in (STATUS_FAIL, STATUS_BLOCKED):
                blockers.append(
                    {
                        "scenario": result.name,
                        "status": result.status,
                        "error": result.error,
                        "details": result.details,
                    }
                )

        if status_counts.get(STATUS_FAIL, 0) > 0:
            decision = {"paper": "GO", "testnet": "NO_GO", "capped_live": "NO_GO"}
        elif status_counts.get(STATUS_BLOCKED, 0) > 0:
            decision = {"paper": "GO", "testnet": "CONDITIONAL", "capped_live": "NO_GO"}
        else:
            decision = {"paper": "GO", "testnet": "GO", "capped_live": "CONDITIONAL"}

        artifact_bundle = self._build_artifact_bundle(results)

        report = {
            "generated_at": self._now_iso(),
            "runtime": artifact_bundle["runtime_manifest"],
            "runtime_manifest_file": artifact_bundle["runtime_manifest_file"],
            "validation_bootstrap_reset": bootstrap_reset_info,
            "config": {
                "mode": self.cfg.mode,
                "testnet": self.cfg.adapter.testnet,
                "dry_run": self.cfg.adapter.dry_run,
                "symbol": self.symbol,
                "max_notional_usdt": self.max_notional_usdt,
                "risk_max_total_notional_pct": self.cfg.risk_limits.max_total_notional_pct,
                "risk_max_leverage": self.cfg.risk_limits.max_leverage,
                "live_cap_usdt": self.cfg.live_startup_max_notional_usdt,
            },
            "status_counts": status_counts,
            "scenarios": [
                {
                    "name": item.name,
                    "status": item.status,
                    "duration_sec": item.duration_sec,
                    "details": item.details,
                    "error": item.error,
                }
                for item in results
            ],
            "remaining_blockers": blockers,
            "artifacts": artifact_bundle["artifacts"],
            "websocket_health_summary": artifact_bundle["websocket_health_summary"],
            "order_lifecycle_summary": artifact_bundle["order_lifecycle_summary"],
            "recovery_scenario_summary": artifact_bundle["recovery_scenario_summary"],
            "go_no_go": decision,
            "operator_rollout_checklist": [
                "Set BOT_RUNTIME_MODE=testnet and LIVE_TRADING_ENABLED=false.",
                "Run app/testnet_validation.py with --execute-orders on valid testnet credentials.",
                "Install exact validated constraints from runtime constraints file before deploy.",
                "Confirm no FAIL/BLOCKED scenarios in websocket, lifecycle, and recovery categories.",
                "Confirm startup_safety report shows expected symbols, inflight count, and schema version.",
                "Enable capped live only after explicit risk cap and live startup cap are configured and testnet report is clean.",
            ],
        }
        return report


def collect_runtime_manifest() -> dict[str, Any]:
    modules = [
        "numpy",
        "pandas",
        "requests",
        "websockets",
        "scikit-learn",
        "joblib",
        "xgboost",
        "lightgbm",
    ]

    pkg_info: dict[str, str] = {}
    try:
        import importlib.metadata as importlib_metadata
    except Exception:
        importlib_metadata = None

    for name in modules:
        version = "missing"
        if importlib_metadata is not None:
            try:
                version = str(importlib_metadata.version(name))
            except Exception:
                version = "missing"
        pkg_info[name] = version

    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": pkg_info,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bybit testnet pre-live validation harness")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--max-notional-usdt", type=float, default=20.0)
    parser.add_argument("--execute-orders", action="store_true", help="Enable tiny order scenarios on testnet")
    parser.add_argument("--soak-seconds", type=int, default=120)
    parser.add_argument("--chaos-cycles", type=int, default=5)
    parser.add_argument("--run-full-suite", action="store_true", help="Run tests/v2 suite and require no skips")
    parser.add_argument("--artifacts-root", default="logs/testnet_validation_artifacts")
    parser.add_argument("--deployment-constraints-out", default="config/runtime_constraints.lock.txt")
    parser.add_argument("--report-out", default="logs/testnet_validation_report.json")
    return parser.parse_args()


def _write_report(path: str, report: dict[str, Any]):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = setup_logging("INFO")

    try:
        cfg = load_runtime_config()
    except ConfigError as exc:
        status = classify_config_error_status(str(exc))
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "runtime": collect_runtime_manifest(),
            "status_counts": {status: 1},
            "scenarios": [
                {
                    "name": "startup_preflight",
                    "status": status,
                    "duration_sec": 0.0,
                    "details": {},
                    "error": f"ConfigError: {exc}",
                }
            ],
            "remaining_blockers": [
                {
                    "scenario": "startup_preflight",
                    "status": status,
                    "error": f"ConfigError: {exc}",
                    "details": {},
                }
            ],
            "go_no_go": {"paper": "GO", "testnet": "NO_GO", "capped_live": "NO_GO"},
        }
        _write_report(args.report_out, report)
        logger.error("validation_config_error=%s report=%s", exc, args.report_out, extra={"event": "validation_error"})
        if status == STATUS_BLOCKED:
            return 4
        return 2

    symbol = args.symbol.strip().replace("/", "").upper() if args.symbol else (cfg.symbols[0] if cfg.symbols else "BTCUSDT")
    harness = TestnetValidationHarness(
        cfg,
        symbol=symbol,
        max_notional_usdt=float(args.max_notional_usdt),
        execute_orders=bool(args.execute_orders),
        soak_seconds=int(args.soak_seconds),
        chaos_cycles=int(args.chaos_cycles),
        run_full_suite=bool(args.run_full_suite),
        logger=logger,
        artifacts_root=str(args.artifacts_root),
        deployment_constraints_out=str(args.deployment_constraints_out),
    )

    try:
        report = harness.run()
    finally:
        harness.close()

    _write_report(args.report_out, report)
    logger.info("validation_report_written=%s status_counts=%s", args.report_out, report.get("status_counts"), extra={"event": "validation_report"})

    fail_count = int((report.get("status_counts") or {}).get(STATUS_FAIL, 0))
    block_count = int((report.get("status_counts") or {}).get(STATUS_BLOCKED, 0))
    if fail_count > 0:
        return 3
    if block_count > 0:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






















