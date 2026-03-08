from __future__ import annotations

from trading.exchange.schemas import AccountSnapshot, InstrumentRules, OpenOrderSnapshot, OrderIntent


class OrderValidationError(ValueError):
    pass


def _is_step_aligned(value: float, step: float, tol: float = 1e-9) -> bool:
    if step <= 0:
        return False
    units = round(value / step)
    return abs(value - units * step) <= tol * max(1.0, step)


def validate_order_intent(
    intent: OrderIntent,
    *,
    rules: InstrumentRules,
    account: AccountSnapshot,
    mark_price: float,
    open_orders: list[OpenOrderSnapshot],
):
    if rules.tick_size <= 0 or rules.qty_step <= 0 or rules.min_qty <= 0 or rules.min_notional <= 0:
        raise OrderValidationError("invalid_instrument_metadata")

    if intent.qty <= 0:
        raise OrderValidationError("qty_must_be_positive")
    if mark_price <= 0:
        raise OrderValidationError("invalid_mark_price")

    if not _is_step_aligned(intent.qty, rules.qty_step):
        raise OrderValidationError("qty_step_mismatch")

    if intent.qty < rules.min_qty:
        raise OrderValidationError("below_min_qty")

    notional = intent.qty * mark_price
    if notional < rules.min_notional:
        raise OrderValidationError("below_min_notional")

    if account.available_balance_usdt <= 0 and not intent.reduce_only:
        raise OrderValidationError("insufficient_available_balance")

    conflict = any(
        o.symbol.replace("/", "").upper() == intent.symbol.replace("/", "").upper()
        and not o.reduce_only
        and not intent.reduce_only
        for o in open_orders
    )
    if conflict:
        raise OrderValidationError("open_order_conflict")
