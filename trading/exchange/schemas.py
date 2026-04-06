from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class InstrumentRules:
    symbol: str
    tick_size: float = 0.01
    qty_step: float = 0.001
    min_qty: float = 0.001
    min_notional: float = 5.0
    max_qty: float = 0.0


@dataclass
class AccountSnapshot:
    equity_usdt: float
    available_balance_usdt: float
    updated_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


@dataclass
class PositionSnapshot:
    symbol: str
    side: PositionSide
    qty: float
    entry_price: float
    liq_price: float
    leverage: float
    position_idx: int
    stop_loss: float | None = None


@dataclass
class OpenOrderSnapshot:
    symbol: str
    order_id: str
    order_link_id: str
    side: OrderSide
    qty: float
    reduce_only: bool
    position_idx: int
    status: str
    created_ts: float = 0.0
    updated_ts: float = 0.0


@dataclass
class OrderIntent:
    symbol: str
    side: OrderSide
    qty: float
    reduce_only: bool = False
    position_idx: int = 0
    client_order_id: str | None = None
    close_on_trigger: bool | None = None


@dataclass
class OrderResult:
    success: bool
    order_id: str
    order_link_id: str
    avg_price: float
    filled_qty: float
    status: str
    raw: dict[str, Any]
    remaining_qty: float = 0.0
    error: str | None = None


@dataclass
class ProtectiveOrderResult:
    success: bool
    raw: dict[str, Any]
    error: str | None = None



def now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()
