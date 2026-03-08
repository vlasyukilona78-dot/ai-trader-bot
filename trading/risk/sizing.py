from __future__ import annotations


def position_size_for_stop(*, equity_usdt: float, risk_pct: float, entry_price: float, stop_loss: float) -> float:
    if equity_usdt <= 0 or entry_price <= 0 or stop_loss <= 0:
        return 0.0
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        return 0.0
    risk_amount = equity_usdt * max(risk_pct, 0.0)
    if risk_amount <= 0:
        return 0.0
    return float(risk_amount / stop_distance)
