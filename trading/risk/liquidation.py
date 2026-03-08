from __future__ import annotations

from trading.exchange.schemas import PositionSide


def liquidation_distance_pct(*, entry_price: float, liq_price: float) -> float:
    if entry_price <= 0 or liq_price <= 0:
        return 1.0
    return abs(entry_price - liq_price) / entry_price


def liquidation_buffer_ok(*, side: PositionSide, entry_price: float, liq_price: float, min_buffer_pct: float) -> bool:
    if liq_price <= 0:
        return True
    dist = liquidation_distance_pct(entry_price=entry_price, liq_price=liq_price)
    return dist >= max(min_buffer_pct, 0.0)
