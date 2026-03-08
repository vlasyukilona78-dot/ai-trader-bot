from __future__ import annotations

import os

if os.getenv("ALLOW_LEGACY_RUNTIME", "false").strip().lower() not in ("1", "true", "yes"):
    raise RuntimeError("Legacy runtime is quarantined. Use V2 entrypoint app/main.py and trading/* modules.")

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    success: bool
    signal_id: str
    symbol: str
    side: str
    qty: float
    fill_price: float
    order_id: str
    raw: dict[str, Any]
    error: str | None = None


class ExecutionEngine:
    def __init__(
        self,
        bybit_client=None,
        dry_run: bool = True,
        duplicate_window_sec: int = 45,
        hedge_mode: bool = False,
    ):
        self.bybit_client = bybit_client
        self.dry_run = dry_run
        self.hedge_mode = bool(hedge_mode)
        self.paper_positions: dict[str, dict[str, Any]] = {}
        self._duplicate_window_sec = max(1, int(duplicate_window_sec))
        self._guard_lock = threading.Lock()
        self._active_symbols: set[str] = set()
        self._recent_order_links: dict[str, float] = {}

    @staticmethod
    def _normalize_signal_side(side: str) -> str:
        s = str(side).upper().strip()
        if s in ("LONG", "BUY"):
            return "LONG"
        if s in ("SHORT", "SELL"):
            return "SHORT"
        return s

    @staticmethod
    def _order_side_for_exchange(side: str) -> str:
        return "sell" if side == "SHORT" else "buy"

    @staticmethod
    def _closing_order_side(position_side: str) -> str:
        return "buy" if position_side == "SHORT" else "sell"

    def _position_idx_for_side(self, side: str) -> int:
        if not self.hedge_mode:
            return 0
        return 2 if side == "SHORT" else 1

    @staticmethod
    def _build_order_link_id(signal_id: str, symbol: str, side: str) -> str:
        key = f"{signal_id}|{symbol}|{side}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:26]
        return f"kbot-{digest}"

    def _prune_recent_locks(self, now_ts: float):
        stale = [link for link, until_ts in self._recent_order_links.items() if until_ts <= now_ts]
        for link in stale:
            self._recent_order_links.pop(link, None)

    def _reserve_execution_slot(self, symbol: str, order_link_id: str) -> tuple[bool, str]:
        now_ts = time.time()
        with self._guard_lock:
            self._prune_recent_locks(now_ts)
            if symbol in self._active_symbols:
                return False, "symbol_execution_inflight"
            if order_link_id in self._recent_order_links:
                return False, "duplicate_order_blocked"
            self._active_symbols.add(symbol)
            self._recent_order_links[order_link_id] = now_ts + self._duplicate_window_sec
        return True, "ok"

    def _release_execution_slot(self, symbol: str):
        with self._guard_lock:
            self._active_symbols.discard(symbol)

    @staticmethod
    def _extract_order_error(result: Any) -> str:
        if not isinstance(result, dict):
            return "order_failed"
        code = result.get("retCode")
        msg = str(result.get("retMsg") or "order_failed")
        if code is None:
            return msg
        return f"order_failed:{code}:{msg}"

    @staticmethod
    def _position_side_from_exchange(raw_side: Any) -> str:
        side = str(raw_side).upper().strip()
        if side in ("BUY", "LONG"):
            return "LONG"
        if side in ("SELL", "SHORT"):
            return "SHORT"
        return side

    def _exchange_has_open_position(self, symbol: str) -> bool:
        if self.bybit_client is None or not hasattr(self.bybit_client, "get_open_positions"):
            return False
        positions = self.bybit_client.get_open_positions(symbol=symbol)
        if not isinstance(positions, list):
            return False
        for item in positions:
            try:
                if float(item.get("size", 0.0)) > 0.0:
                    return True
            except Exception:
                continue
        return False

    def _exchange_has_open_orders(self, symbol: str) -> bool:
        if self.bybit_client is None or not hasattr(self.bybit_client, "get_open_orders"):
            return False
        orders = self.bybit_client.get_open_orders(symbol=symbol)
        return isinstance(orders, list) and len(orders) > 0

    def execute(self, signal, qty: float, fill_price: float | None = None) -> ExecutionResult:
        symbol = str(signal.symbol)
        side = self._normalize_signal_side(signal.side)
        signal_id = str(signal.signal_id)

        if qty <= 0:
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                symbol=symbol,
                side=side,
                qty=qty,
                fill_price=float(signal.entry),
                order_id="",
                raw={},
                error="qty<=0",
            )

        if side not in ("LONG", "SHORT"):
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                symbol=symbol,
                side=side,
                qty=qty,
                fill_price=float(signal.entry),
                order_id="",
                raw={},
                error="invalid_side",
            )

        order_link_id = self._build_order_link_id(signal_id=signal_id, symbol=symbol, side=side)
        reserved, reserve_reason = self._reserve_execution_slot(symbol=symbol.upper(), order_link_id=order_link_id)
        if not reserved:
            return ExecutionResult(
                success=False,
                signal_id=signal_id,
                symbol=symbol,
                side=side,
                qty=float(qty),
                fill_price=float(signal.entry),
                order_id="",
                raw={},
                error=reserve_reason,
            )

        try:
            if self.dry_run or self.bybit_client is None:
                order_id = f"paper-{signal_id}"
                fill = float(fill_price if fill_price is not None else signal.entry)
                self.paper_positions[signal_id] = {
                    "symbol": symbol,
                    "side": side,
                    "qty": float(qty),
                    "entry": fill,
                    "tp": float(signal.tp),
                    "sl": float(signal.sl),
                    "opened_ts": time.time(),
                }
                return ExecutionResult(
                    success=True,
                    signal_id=signal_id,
                    symbol=symbol,
                    side=side,
                    qty=float(qty),
                    fill_price=fill,
                    order_id=order_id,
                    raw={"mode": "paper", "orderLinkId": order_link_id},
                )

            if self._exchange_has_open_position(symbol=symbol):
                return ExecutionResult(
                    success=False,
                    signal_id=signal_id,
                    symbol=symbol,
                    side=side,
                    qty=float(qty),
                    fill_price=float(signal.entry),
                    order_id="",
                    raw={},
                    error="exchange_position_exists",
                )

            if self._exchange_has_open_orders(symbol=symbol):
                return ExecutionResult(
                    success=False,
                    signal_id=signal_id,
                    symbol=symbol,
                    side=side,
                    qty=float(qty),
                    fill_price=float(signal.entry),
                    order_id="",
                    raw={},
                    error="exchange_open_order_exists",
                )

            exchange_side = self._order_side_for_exchange(side)
            position_idx = self._position_idx_for_side(side)
            order_resp = self.bybit_client.place_order_market(
                symbol,
                exchange_side,
                qty,
                reduce_only=False,
                position_idx=position_idx,
                order_link_id=order_link_id,
            )
            order_ok = isinstance(order_resp, dict) and order_resp.get("retCode", 1) == 0
            if not order_ok:
                return ExecutionResult(
                    success=False,
                    signal_id=signal_id,
                    symbol=symbol,
                    side=side,
                    qty=float(qty),
                    fill_price=float(signal.entry),
                    order_id="",
                    raw=order_resp if isinstance(order_resp, dict) else {},
                    error=self._extract_order_error(order_resp),
                )

            payload = order_resp.get("result", {}) if isinstance(order_resp, dict) else {}
            order_id = str(payload.get("orderId") or payload.get("orderLinkId") or order_link_id)
            fill = float(fill_price if fill_price is not None else signal.entry)
            try:
                fill = float(payload.get("avgPrice") or fill)
            except (TypeError, ValueError):
                pass

            stop_resp = self.bybit_client.set_trading_stop(
                symbol=symbol,
                stop_loss=float(signal.sl),
                take_profit=float(signal.tp),
                position_idx=position_idx,
            )
            stop_ok = isinstance(stop_resp, dict) and stop_resp.get("retCode", 1) == 0
            if not stop_ok:
                emergency_close_side = self._closing_order_side(side)
                emergency_resp = self.bybit_client.place_order_market(
                    symbol,
                    emergency_close_side,
                    qty,
                    reduce_only=True,
                    position_idx=position_idx,
                    order_link_id=f"{order_link_id}-slf",
                    close_on_trigger=True,
                )
                return ExecutionResult(
                    success=False,
                    signal_id=signal_id,
                    symbol=symbol,
                    side=side,
                    qty=float(qty),
                    fill_price=fill,
                    order_id=order_id,
                    raw={
                        "order": order_resp,
                        "stop": stop_resp,
                        "emergency_close": emergency_resp,
                    },
                    error="stop_loss_set_failed",
                )

            return ExecutionResult(
                success=True,
                signal_id=signal_id,
                symbol=symbol,
                side=side,
                qty=float(qty),
                fill_price=fill,
                order_id=order_id,
                raw={"order": order_resp, "stop": stop_resp},
                error=None,
            )
        finally:
            self._release_execution_slot(symbol=symbol.upper())

    def update_paper_positions(self, symbol: str, last_price: float) -> list[dict[str, Any]]:
        closed: list[dict[str, Any]] = []
        for signal_id, pos in list(self.paper_positions.items()):
            if pos.get("symbol") != symbol:
                continue

            side = pos["side"]
            entry = float(pos["entry"])
            qty = float(pos["qty"])
            tp = float(pos["tp"])
            sl = float(pos["sl"])

            hit_tp = (last_price <= tp) if side == "SHORT" else (last_price >= tp)
            hit_sl = (last_price >= sl) if side == "SHORT" else (last_price <= sl)
            if not (hit_tp or hit_sl):
                continue

            exit_price = tp if hit_tp else sl
            if side == "SHORT":
                pnl = (entry - exit_price) * qty
            else:
                pnl = (exit_price - entry) * qty

            closed.append(
                {
                    "signal_id": signal_id,
                    "symbol": symbol,
                    "side": side,
                    "entry": entry,
                    "exit": exit_price,
                    "qty": qty,
                    "pnl": pnl,
                    "closed_reason": "tp" if hit_tp else "sl",
                    "duration_sec": time.time() - float(pos.get("opened_ts", time.time())),
                }
            )
            self.paper_positions.pop(signal_id, None)

        return closed

