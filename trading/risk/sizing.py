from __future__ import annotations

from decimal import Decimal, InvalidOperation


def _to_decimal(value: float) -> Decimal:
    return Decimal(str(value))


def position_size_for_stop(*, equity_usdt: float, risk_pct: float, entry_price: float, stop_loss: float) -> float:
    try:
        equity_d = _to_decimal(equity_usdt)
        risk_pct_d = _to_decimal(risk_pct)
        entry_d = _to_decimal(entry_price)
        stop_d = _to_decimal(stop_loss)
    except (InvalidOperation, ValueError):
        return 0.0

    if equity_d <= 0 or entry_d <= 0 or stop_d <= 0:
        return 0.0

    stop_distance = abs(entry_d - stop_d)
    if stop_distance <= 0:
        return 0.0

    risk_amount = equity_d * max(risk_pct_d, Decimal("0"))
    if risk_amount <= 0:
        return 0.0

    return float(risk_amount / stop_distance)
