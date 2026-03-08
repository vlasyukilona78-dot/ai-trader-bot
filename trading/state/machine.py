from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from trading.exchange.schemas import OpenOrderSnapshot, OrderSide, PositionSide, PositionSnapshot
from trading.portfolio.positions import first_effective_position_for_symbol
from trading.state.models import StateRecord, TradeState
from trading.state.persistence import RuntimeStore
from trading.state.transitions import can_transition


@dataclass
class StateTransition:
    symbol: str
    previous: TradeState
    current: TradeState
    reason: str
    ts: float


class StateMachine:
    def __init__(self, persistence: RuntimeStore | None = None):
        self._records: dict[str, StateRecord] = {}
        self._history: list[StateTransition] = []
        self.persistence = persistence
        if self.persistence is not None:
            self._load_persisted_state()

    @staticmethod
    def _now_ts() -> float:
        return datetime.now(timezone.utc).timestamp()

    def _load_persisted_state(self):
        for row in self.persistence.load_state_records():
            try:
                state = TradeState(str(row.state))
            except Exception:
                state = TradeState.ERROR
            self._records[row.symbol] = StateRecord(
                symbol=row.symbol,
                state=state,
                reason=row.reason,
                updated_at=float(row.updated_at),
            )

    def get(self, symbol: str) -> StateRecord:
        key = symbol.replace("/", "").upper()
        rec = self._records.get(key)
        if rec is None:
            rec = StateRecord(symbol=key)
            self._records[key] = rec
            if self.persistence is not None:
                self.persistence.upsert_state_record(
                    symbol=rec.symbol,
                    state=rec.state.value,
                    reason=rec.reason,
                    updated_at=rec.updated_at,
                )
        return rec

    def _persist(self, rec: StateRecord, prev: TradeState | None = None):
        if self.persistence is None:
            return
        self.persistence.upsert_state_record(
            symbol=rec.symbol,
            state=rec.state.value,
            reason=rec.reason,
            updated_at=rec.updated_at,
        )
        if prev is not None:
            self.persistence.append_transition(
                symbol=rec.symbol,
                previous_state=prev.value,
                current_state=rec.state.value,
                reason=rec.reason,
                ts=rec.updated_at,
            )

    def _apply_state(self, symbol: str, target: TradeState, reason: str):
        rec = self.get(symbol)
        prev = rec.state
        rec.state = target
        rec.reason = reason
        rec.updated_at = self._now_ts()
        transition = StateTransition(symbol=rec.symbol, previous=prev, current=target, reason=reason, ts=rec.updated_at)
        self._history.append(transition)
        self._persist(rec, prev=prev)

    def transition(self, symbol: str, target: TradeState, reason: str) -> bool:
        rec = self.get(symbol)
        if rec.state == target:
            rec.reason = reason
            rec.updated_at = self._now_ts()
            self._persist(rec, prev=None)
            return True
        if not can_transition(rec.state, target):
            return False
        self._apply_state(symbol, target, reason)
        return True

    def _reconcile_set(self, symbol: str, target: TradeState, reason: str):
        if not self.transition(symbol, target, reason):
            self._apply_state(symbol, target, f"{reason}_forced")

    def reconcile(
        self,
        symbol: str,
        positions: list[PositionSnapshot],
        open_orders: list[OpenOrderSnapshot],
    ) -> StateRecord:
        key = symbol.replace("/", "").upper()
        rec = self.get(key)

        position = first_effective_position_for_symbol(positions, key)
        orders = [o for o in open_orders if o.symbol.replace("/", "").upper() == key]

        if position is not None:
            target = TradeState.LONG if position.side == PositionSide.LONG else TradeState.SHORT
            self._reconcile_set(key, target, "reconcile_position")
            return self.get(key)

        if orders:
            has_reduce = any(o.reduce_only for o in orders)
            buy_open = any(o.side == OrderSide.BUY and not o.reduce_only for o in orders)
            sell_open = any(o.side == OrderSide.SELL and not o.reduce_only for o in orders)

            if has_reduce:
                if rec.state in (TradeState.LONG, TradeState.PENDING_EXIT_LONG):
                    self._reconcile_set(key, TradeState.PENDING_EXIT_LONG, "reconcile_open_reduce_order")
                elif rec.state in (TradeState.SHORT, TradeState.PENDING_EXIT_SHORT):
                    self._reconcile_set(key, TradeState.PENDING_EXIT_SHORT, "reconcile_open_reduce_order")
                else:
                    self._reconcile_set(key, TradeState.FLAT, "reconcile_reduce_without_position")
                return self.get(key)

            if buy_open and not sell_open:
                self._reconcile_set(key, TradeState.PENDING_ENTRY_LONG, "reconcile_open_entry_order")
                return self.get(key)
            if sell_open and not buy_open:
                self._reconcile_set(key, TradeState.PENDING_ENTRY_SHORT, "reconcile_open_entry_order")
                return self.get(key)

        self._reconcile_set(key, TradeState.FLAT, "reconcile_flat")
        return self.get(key)

    def history(self) -> list[StateTransition]:
        return list(self._history)


