from __future__ import annotations

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
    def __init__(self, bybit_client=None, dry_run: bool = True):
        self.bybit_client = bybit_client
        self.dry_run = dry_run
        self.paper_positions: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _order_side_for_exchange(side: str) -> str:
        return "sell" if side == "SHORT" else "buy"

    def execute(self, signal, qty: float, fill_price: float | None = None) -> ExecutionResult:
        if qty <= 0:
            return ExecutionResult(
                success=False,
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                side=signal.side,
                qty=qty,
                fill_price=signal.entry,
                order_id="",
                raw={},
                error="qty<=0",
            )

        if self.dry_run or self.bybit_client is None:
            order_id = f"paper-{signal.signal_id}"
            fill = float(fill_price if fill_price is not None else signal.entry)
            self.paper_positions[signal.signal_id] = {
                "symbol": signal.symbol,
                "side": signal.side,
                "qty": float(qty),
                "entry": fill,
                "tp": float(signal.tp),
                "sl": float(signal.sl),
                "opened_ts": time.time(),
            }
            return ExecutionResult(
                success=True,
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                side=signal.side,
                qty=float(qty),
                fill_price=fill,
                order_id=order_id,
                raw={"mode": "paper"},
            )

        exchange_side = self._order_side_for_exchange(signal.side)
        result = self.bybit_client.place_order_market(signal.symbol, exchange_side, qty)
        ok = isinstance(result, dict) and result.get("retCode", 1) == 0
        order_id = ""
        fill = float(fill_price if fill_price is not None else signal.entry)
        if ok:
            payload = result.get("result", {})
            order_id = str(payload.get("orderId") or payload.get("orderLinkId") or signal.signal_id)
            try:
                fill = float(payload.get("avgPrice") or fill)
            except (TypeError, ValueError):
                pass

        return ExecutionResult(
            success=bool(ok),
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.side,
            qty=float(qty),
            fill_price=fill,
            order_id=order_id,
            raw=result if isinstance(result, dict) else {},
            error=None if ok else "order_failed",
        )

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
