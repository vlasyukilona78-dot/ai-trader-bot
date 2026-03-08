from typing import Any

from trading.exchange.schemas import PositionSide, PositionSnapshot


POSITION_SIZE_EPSILON = 1e-9


def _norm_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").upper().strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _side_value(side: Any) -> str:
    if isinstance(side, PositionSide):
        return side.value
    return str(side or "").upper().strip()


def position_has_effective_exposure(
    position: PositionSnapshot,
    *,
    size_epsilon: float = POSITION_SIZE_EPSILON,
) -> bool:
    if not isinstance(position, PositionSnapshot):
        return False
    qty = abs(_safe_float(getattr(position, "qty", 0.0), 0.0))
    if qty <= max(0.0, float(size_epsilon)):
        return False
    return _side_value(getattr(position, "side", "")) in (PositionSide.LONG.value, PositionSide.SHORT.value)


def split_effective_positions(
    positions: list[PositionSnapshot],
    *,
    symbol: str | None = None,
    size_epsilon: float = POSITION_SIZE_EPSILON,
) -> tuple[list[PositionSnapshot], list[PositionSnapshot]]:
    target = _norm_symbol(symbol or "")
    effective: list[PositionSnapshot] = []
    placeholders: list[PositionSnapshot] = []

    for position in positions or []:
        if not isinstance(position, PositionSnapshot):
            continue
        if target and _norm_symbol(position.symbol) != target:
            continue
        if position_has_effective_exposure(position, size_epsilon=size_epsilon):
            effective.append(position)
        else:
            placeholders.append(position)

    return effective, placeholders


def first_effective_position_for_symbol(
    positions: list[PositionSnapshot],
    symbol: str,
    *,
    size_epsilon: float = POSITION_SIZE_EPSILON,
) -> PositionSnapshot | None:
    effective, _ = split_effective_positions(positions, symbol=symbol, size_epsilon=size_epsilon)
    if not effective:
        return None
    return effective[0]


def position_to_report_row(position: PositionSnapshot) -> dict[str, Any]:
    return {
        "symbol": _norm_symbol(position.symbol),
        "side": _side_value(position.side),
        "qty": _safe_float(position.qty, 0.0),
        "position_idx": int(_safe_float(position.position_idx, 0)),
        "entry_price": _safe_float(position.entry_price, 0.0),
        "liq_price": _safe_float(position.liq_price, 0.0),
        "leverage": _safe_float(position.leverage, 0.0),
        "stop_loss": _safe_float(position.stop_loss, 0.0) if position.stop_loss is not None else None,
    }


def summarize_positions(
    positions: list[PositionSnapshot],
    *,
    symbol: str | None = None,
    size_epsilon: float = POSITION_SIZE_EPSILON,
) -> dict[str, Any]:
    target = _norm_symbol(symbol or "")
    raw_positions = [
        position
        for position in (positions or [])
        if isinstance(position, PositionSnapshot) and (not target or _norm_symbol(position.symbol) == target)
    ]
    effective, placeholders = split_effective_positions(
        raw_positions,
        symbol=target or None,
        size_epsilon=size_epsilon,
    )
    return {
        "raw_positions_count": int(len(raw_positions)),
        "effective_open_positions_count": int(len(effective)),
        "effective_positions": [position_to_report_row(position) for position in effective],
        "zero_size_placeholder_positions": [position_to_report_row(position) for position in placeholders],
        "position_size_epsilon": float(max(0.0, float(size_epsilon))),
    }


def total_notional(positions: list[PositionSnapshot]) -> float:
    effective, _ = split_effective_positions(positions)
    return float(sum(max(p.qty, 0.0) * max(p.entry_price, 0.0) for p in effective))


def net_side(positions: list[PositionSnapshot]) -> str:
    effective, _ = split_effective_positions(positions)
    if not effective:
        return "FLAT"
    long_qty = sum(p.qty for p in effective if p.side.value == "LONG")
    short_qty = sum(p.qty for p in effective if p.side.value == "SHORT")
    if long_qty > short_qty:
        return "LONG"
    if short_qty > long_qty:
        return "SHORT"
    return "FLAT"
