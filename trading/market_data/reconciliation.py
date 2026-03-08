from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from trading.exchange.schemas import AccountSnapshot, OpenOrderSnapshot, PositionSnapshot


class ExchangeAdapter(Protocol):
    def get_account(self) -> AccountSnapshot: ...

    def get_positions(self, symbol: str | None = None) -> list[PositionSnapshot]: ...

    def get_open_orders(self, symbol: str | None = None) -> list[OpenOrderSnapshot]: ...


@dataclass
class ExchangeSnapshot:
    symbol: str
    account: AccountSnapshot
    positions: list[PositionSnapshot]
    open_orders: list[OpenOrderSnapshot]
    reconciled_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


class ExchangeReconciler:
    def __init__(self, adapter: ExchangeAdapter):
        self.adapter = adapter

    def snapshot(self, symbol: str) -> ExchangeSnapshot:
        account = self.adapter.get_account()
        positions = self.adapter.get_positions(symbol=symbol)
        orders = self.adapter.get_open_orders(symbol=symbol)
        return ExchangeSnapshot(symbol=symbol, account=account, positions=positions, open_orders=orders)
