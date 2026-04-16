from __future__ import annotations

from decimal import Decimal, InvalidOperation

from trading.exchange.schemas import AccountSnapshot, InstrumentRules, OpenOrderSnapshot, OrderIntent


class OrderValidationError(ValueError):
    pass


def _to_decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _is_step_aligned(value: float, step: float, tol: float = 1e-9) -> bool:
    if step <= 0:
        return False
    try:
        value_d = _to_decimal(value)
        step_d = _to_decimal(step)
        tol_d = _to_decimal(tol) * max(Decimal("1"), step_d)
    except (InvalidOperation, ValueError):
        return False
    if step_d <= 0:
        return False
    remainder = value_d % step_d
    return remainder <= tol_d or (step_d - remainder) <= tol_d


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
    if rules.max_qty > 0 and intent.qty > rules.max_qty:
        raise OrderValidationError("above_max_qty")

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
