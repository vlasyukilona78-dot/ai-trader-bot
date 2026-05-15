from __future__ import annotations

from decimal import Decimal, InvalidOperation

from trading.exchange.schemas import AccountSnapshot, InstrumentRules, OpenOrderSnapshot, OrderIntent


class OrderValidationError(ValueError):
    pass


def _to_decimal(value: float) -> Decimal:
    out = Decimal(str(value))
    if not out.is_finite():
        raise InvalidOperation
    return out


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
    try:
        qty_d = _to_decimal(intent.qty)
        mark_price_d = _to_decimal(mark_price)
        tick_size_d = _to_decimal(rules.tick_size)
        qty_step_d = _to_decimal(rules.qty_step)
        min_qty_d = _to_decimal(rules.min_qty)
        min_notional_d = _to_decimal(rules.min_notional)
        max_qty_d = _to_decimal(rules.max_qty) if rules.max_qty > 0 else Decimal("0")
    except (InvalidOperation, ValueError):
        raise OrderValidationError("non_finite_order_input")

    if tick_size_d <= 0 or qty_step_d <= 0 or min_qty_d <= 0 or min_notional_d <= 0:
        raise OrderValidationError("invalid_instrument_metadata")

    if qty_d <= 0:
        raise OrderValidationError("qty_must_be_positive")
    if mark_price_d <= 0:
        raise OrderValidationError("invalid_mark_price")

    if not _is_step_aligned(intent.qty, rules.qty_step):
        raise OrderValidationError("qty_step_mismatch")

    if qty_d < min_qty_d:
        raise OrderValidationError("below_min_qty")
    if max_qty_d > 0 and qty_d > max_qty_d:
        raise OrderValidationError("above_max_qty")

    notional = qty_d * mark_price_d
    if notional < min_notional_d:
        raise OrderValidationError("below_min_notional")

    try:
        available_balance_d = _to_decimal(account.available_balance_usdt)
    except (InvalidOperation, ValueError):
        available_balance_d = Decimal("0")

    if available_balance_d <= 0 and not intent.reduce_only:
        raise OrderValidationError("insufficient_available_balance")

    conflict = any(
        o.symbol.replace("/", "").upper() == intent.symbol.replace("/", "").upper()
        and not o.reduce_only
        and not intent.reduce_only
        for o in open_orders
    )
    if conflict:
        raise OrderValidationError("open_order_conflict")
