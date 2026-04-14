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

    def snapshot_many(self, symbols: list[str]) -> dict[str, ExchangeSnapshot]:
        account = self.adapter.get_account()
        positions = self.adapter.get_positions(symbol=None)
        orders = self.adapter.get_open_orders(symbol=None)

        positions_by_symbol: dict[str, list[PositionSnapshot]] = {}
        for position in positions:
            positions_by_symbol.setdefault(str(position.symbol).upper().strip(), []).append(position)

        orders_by_symbol: dict[str, list[OpenOrderSnapshot]] = {}
        for order in orders:
            orders_by_symbol.setdefault(str(order.symbol).upper().strip(), []).append(order)

        snapshots: dict[str, ExchangeSnapshot] = {}
        for symbol in symbols:
            norm = str(symbol).replace("/", "").upper().strip()
            snapshots[norm] = ExchangeSnapshot(
                symbol=norm,
                account=account,
                positions=list(positions_by_symbol.get(norm, [])),
                open_orders=list(orders_by_symbol.get(norm, [])),
            )
        return snapshots
