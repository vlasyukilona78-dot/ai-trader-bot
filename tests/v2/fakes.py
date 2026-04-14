from __future__ import annotations

import time
from dataclasses import dataclass, field

from trading.exchange.events import NormalizedExchangeEvent
from trading.exchange.schemas import (
    AccountSnapshot,
    InstrumentRules,
    OpenOrderSnapshot,
    OrderIntent,
    OrderResult,
    OrderSide,
    PositionSide,
    PositionSnapshot,
    ProtectiveOrderResult,
)


@dataclass
class FakeAdapter:
    account: AccountSnapshot = field(default_factory=lambda: AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0))
    rules: InstrumentRules = field(default_factory=lambda: InstrumentRules(symbol="BTCUSDT", tick_size=0.1, qty_step=0.001, min_qty=0.001, min_notional=5.0))
    mark_price: float = 100.0
    hedge_mode: bool = False
    ws_health_meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.positions: list[PositionSnapshot] = []
        self.open_orders: list[OpenOrderSnapshot] = []
        self.placed_orders: list[OrderIntent] = []
        self.stop_calls: list[dict] = []
        self.canceled_orders: list[dict] = []
        self.fail_next_order: bool = False
        self.fail_next_stop: bool = False
        self.fail_order_times: int = 0
        self.fail_stop_times: int = 0
        self.partial_fill_qty: float | None = None
        self.partial_fill_leaves_open: bool = False
        self.ws_events: list[NormalizedExchangeEvent] = []

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return str(symbol).replace("/", "").upper()

    @staticmethod
    def position_idx_for_side(position_side: PositionSide, hedge_mode: bool) -> int:
        if not hedge_mode:
            return 0
        return 1 if position_side == PositionSide.LONG else 2

    @staticmethod
    def round_qty(qty: float, qty_step: float) -> float:
        if qty_step <= 0:
            return float(max(qty, 0.0))
        return float(int(max(qty, 0.0) / qty_step) * qty_step)

    def metadata_health(self) -> dict:
        payload = {"cached_symbols": 1, "fresh_symbols": 1, "stale_symbols": 0, "ttl_sec": 900}
        if self.ws_health_meta:
            payload["ws"] = dict(self.ws_health_meta)
        return payload

    def get_account(self) -> AccountSnapshot:
        return self.account

    def get_positions(self, symbol: str | None = None) -> list[PositionSnapshot]:
        if symbol is None:
            return list(self.positions)
        norm = self.normalize_symbol(symbol)
        return [p for p in self.positions if self.normalize_symbol(p.symbol) == norm]

    def get_open_orders(self, symbol: str | None = None) -> list[OpenOrderSnapshot]:
        if symbol is None:
            return list(self.open_orders)
        norm = self.normalize_symbol(symbol)
        return [o for o in self.open_orders if self.normalize_symbol(o.symbol) == norm]

    def get_mark_price(self, symbol: str) -> float:
        return float(self.mark_price)

    def get_instrument_rules(self, symbol: str) -> InstrumentRules:
        return InstrumentRules(
            symbol=self.normalize_symbol(symbol),
            tick_size=self.rules.tick_size,
            qty_step=self.rules.qty_step,
            min_qty=self.rules.min_qty,
            min_notional=self.rules.min_notional,
        )

    def drain_ws_events(self) -> list[NormalizedExchangeEvent]:
        out = list(self.ws_events)
        self.ws_events.clear()
        return out

    def place_market_order(self, intent: OrderIntent) -> OrderResult:
        self.placed_orders.append(intent)
        if self.fail_order_times > 0:
            self.fail_order_times -= 1
            return OrderResult(
                success=False,
                order_id="",
                order_link_id=intent.client_order_id or "",
                avg_price=0.0,
                filled_qty=0.0,
                status="Rejected",
                raw={"retCode": 10006, "retMsg": "rate limit"},
                error="rate limit",
            )
        if self.fail_next_order:
            self.fail_next_order = False
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

        fill_qty = float(intent.qty if self.partial_fill_qty is None else min(intent.qty, self.partial_fill_qty))
        remaining = max(0.0, float(intent.qty) - float(fill_qty))

        if not intent.reduce_only:
            side = PositionSide.LONG if intent.side == OrderSide.BUY else PositionSide.SHORT
            self.positions = [
                PositionSnapshot(
                    symbol=self.normalize_symbol(intent.symbol),
                    side=side,
                    qty=fill_qty,
                    entry_price=self.mark_price,
                    liq_price=0.0,
                    leverage=1.0,
                    position_idx=intent.position_idx,
                    stop_loss=None,
                )
            ]
            if remaining > 0 and self.partial_fill_leaves_open:
                self.open_orders = [
                    OpenOrderSnapshot(
                        symbol=self.normalize_symbol(intent.symbol),
                        order_id=f"open-{len(self.open_orders)+1}",
                        order_link_id=intent.client_order_id or "",
                        side=intent.side,
                        qty=remaining,
                        reduce_only=False,
                        position_idx=intent.position_idx,
                        status="PartiallyFilled",
                        created_ts=time.time(),
                        updated_ts=time.time(),
                    )
                ]
            else:
                self.open_orders = []
        else:
            if self.positions:
                remaining_pos = max(0.0, self.positions[0].qty - fill_qty)
                if remaining_pos <= 0:
                    self.positions = []
                else:
                    self.positions[0].qty = remaining_pos
            self.open_orders = []

        return OrderResult(
            success=True,
            order_id=f"order-{len(self.placed_orders)}",
            order_link_id=intent.client_order_id or "",
            avg_price=self.mark_price,
            filled_qty=fill_qty,
            remaining_qty=remaining,
            status="Filled" if remaining <= 0 else "PartiallyFilled",
            raw={"retCode": 0, "retMsg": "OK"},
            error=None,
        )

    def set_protective_orders(
        self,
        symbol: str,
        *,
        stop_loss: float,
        take_profit: float | None,
        position_idx: int,
        qty: float | None = None,
    ) -> ProtectiveOrderResult:
        self.stop_calls.append(
            {
                "symbol": self.normalize_symbol(symbol),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_idx": position_idx,
                "qty": qty,
            }
        )
        if self.fail_stop_times > 0:
            self.fail_stop_times -= 1
            return ProtectiveOrderResult(success=False, raw={"retCode": 10006, "retMsg": "rate limit"}, error="rate limit")
        if self.fail_next_stop:
            self.fail_next_stop = False
            return ProtectiveOrderResult(success=False, raw={"retCode": 10001}, error="stop_fail")

        for idx, position in enumerate(self.positions):
            if self.normalize_symbol(position.symbol) == self.normalize_symbol(symbol) and position.position_idx == position_idx:
                self.positions[idx].stop_loss = float(stop_loss)

        return ProtectiveOrderResult(success=True, raw={"retCode": 0}, error=None)

    def cancel_order(self, *, symbol: str, order_id: str = "", order_link_id: str = "") -> bool:
        self.canceled_orders.append({"symbol": self.normalize_symbol(symbol), "order_id": order_id, "order_link_id": order_link_id})
        kept: list[OpenOrderSnapshot] = []
        for order in self.open_orders:
            if self.normalize_symbol(order.symbol) != self.normalize_symbol(symbol):
                kept.append(order)
                continue
            if order_id and order.order_id == order_id:
                continue
            if order_link_id and order.order_link_id == order_link_id:
                continue
            if not order_id and not order_link_id:
                continue
            kept.append(order)
        self.open_orders = kept
        return True
