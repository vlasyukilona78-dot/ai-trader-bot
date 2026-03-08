from __future__ import annotations

from trading.state.models import TradeState


_ALLOWED: dict[TradeState, set[TradeState]] = {
    TradeState.FLAT: {
        TradeState.PENDING_ENTRY_LONG,
        TradeState.PENDING_ENTRY_SHORT,
        TradeState.RECOVERING,
        TradeState.HALTED,
        TradeState.ERROR,
    },
    TradeState.PENDING_ENTRY_LONG: {TradeState.LONG, TradeState.FLAT, TradeState.RECOVERING, TradeState.HALTED, TradeState.ERROR},
    TradeState.LONG: {TradeState.PENDING_EXIT_LONG, TradeState.RECOVERING, TradeState.HALTED, TradeState.ERROR},
    TradeState.PENDING_EXIT_LONG: {TradeState.FLAT, TradeState.LONG, TradeState.RECOVERING, TradeState.HALTED, TradeState.ERROR},
    TradeState.PENDING_ENTRY_SHORT: {TradeState.SHORT, TradeState.FLAT, TradeState.RECOVERING, TradeState.HALTED, TradeState.ERROR},
    TradeState.SHORT: {TradeState.PENDING_EXIT_SHORT, TradeState.RECOVERING, TradeState.HALTED, TradeState.ERROR},
    TradeState.PENDING_EXIT_SHORT: {TradeState.FLAT, TradeState.SHORT, TradeState.RECOVERING, TradeState.HALTED, TradeState.ERROR},
    TradeState.RECOVERING: {
        TradeState.FLAT,
        TradeState.PENDING_ENTRY_LONG,
        TradeState.PENDING_ENTRY_SHORT,
        TradeState.LONG,
        TradeState.SHORT,
        TradeState.HALTED,
        TradeState.ERROR,
    },
    TradeState.HALTED: {TradeState.FLAT, TradeState.RECOVERING, TradeState.LONG, TradeState.SHORT, TradeState.ERROR},
    TradeState.ERROR: {TradeState.FLAT, TradeState.RECOVERING, TradeState.HALTED},
}


def can_transition(current: TradeState, target: TradeState) -> bool:
    return target in _ALLOWED.get(current, set())
