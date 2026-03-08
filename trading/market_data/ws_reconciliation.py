import time
from dataclasses import dataclass

from trading.exchange.events import ExchangeEventType, NormalizedExchangeEvent
from trading.exchange.schemas import AccountSnapshot, OpenOrderSnapshot, PositionSnapshot
from trading.market_data.reconciliation import ExchangeReconciler, ExchangeSnapshot
from trading.portfolio.positions import (
    POSITION_SIZE_EPSILON,
    position_has_effective_exposure,
    split_effective_positions,
)


_TERMINAL_ORDER_STATUSES = {"FILLED", "CANCELLED", "REJECTED", "DEACTIVATED"}


@dataclass
class SyncHealth:
    ws_connected: bool
    ws_last_event_ts: float
    ws_stale: bool
    fallback_polling: bool
    snapshot_required: bool


class ExchangeSyncService:
    """Websocket-first state sync with polling fallback."""

    def __init__(
        self,
        reconciler: ExchangeReconciler,
        *,
        poll_interval_sec: int = 10,
        max_event_staleness_sec: int = 20,
    ):
        self.reconciler = reconciler
        self.poll_interval_sec = max(1, int(poll_interval_sec))
        self.max_event_staleness_sec = max(1, int(max_event_staleness_sec))

        self._ws_connected = False
        self._last_event_ts = 0.0
        self._account: AccountSnapshot | None = None
        self._positions_by_symbol: dict[str, list[PositionSnapshot]] = {}
        self._orders_by_symbol: dict[str, list[OpenOrderSnapshot]] = {}
        self._market_by_symbol: dict[str, float] = {}

        self._snapshot_required_global = False
        self._snapshot_required_symbols: set[str] = set()

        self._polled_snapshots: dict[str, ExchangeSnapshot] = {}
        self._last_poll_ts: dict[str, float] = {}

    @staticmethod
    def _norm_symbol(symbol: str) -> str:
        return str(symbol).replace("/", "").upper().strip()

    @staticmethod
    def _position_idx(position: PositionSnapshot) -> int:
        try:
            return int(getattr(position, "position_idx", 0))
        except (TypeError, ValueError):
            return 0

    def _require_snapshot(self, symbol: str | None = None):
        norm = self._norm_symbol(symbol or "")
        if norm:
            self._snapshot_required_symbols.add(norm)
            return
        self._snapshot_required_global = True

    def _clear_snapshot_requirement(self, symbol: str):
        norm = self._norm_symbol(symbol)
        if norm:
            self._snapshot_required_symbols.discard(norm)
        if self._snapshot_required_global:
            if not self._snapshot_required_symbols:
                self._snapshot_required_global = False

    def _snapshot_required(self, symbol: str) -> bool:
        norm = self._norm_symbol(symbol)
        return self._snapshot_required_global or norm in self._snapshot_required_symbols

    def process_events(self, events: list[NormalizedExchangeEvent]):
        for event in events:
            self.handle_event(event)

    def pull_adapter_events(self, adapter):
        drain = getattr(adapter, "drain_ws_events", None)
        if not callable(drain):
            return
        try:
            events = drain() or []
        except Exception:
            events = []
        if events:
            self.process_events(events)

    def handle_event(self, event: NormalizedExchangeEvent):
        ts = float(event.ts or time.time())
        self._last_event_ts = max(self._last_event_ts, ts)

        if event.event_type == ExchangeEventType.CONNECTED:
            self._ws_connected = True
            return

        if event.event_type in (
            ExchangeEventType.RECONNECTING,
            ExchangeEventType.DISCONNECTED,
            ExchangeEventType.ERROR,
        ):
            self._ws_connected = False
            self._require_snapshot(event.symbol)
            return

        if event.event_type == ExchangeEventType.SNAPSHOT_REQUIRED:
            self._require_snapshot(event.symbol)
            return

        if event.event_type == ExchangeEventType.INTERVENTION:
            self._require_snapshot(event.symbol)
            return

        if event.event_type == ExchangeEventType.HEARTBEAT:
            return

        if event.event_type == ExchangeEventType.ACCOUNT:
            account = event.payload.get("account") if isinstance(event.payload, dict) else None
            if isinstance(account, AccountSnapshot):
                self._account = account
            return

        symbol = self._norm_symbol(event.symbol or "")
        if not symbol:
            return

        if event.event_type == ExchangeEventType.MARKET:
            payload = event.payload if isinstance(event.payload, dict) else {}
            mark_price = payload.get("mark_price")
            try:
                value = float(mark_price)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                self._market_by_symbol[symbol] = value
            return

        if event.event_type == ExchangeEventType.POSITION:
            payload = event.payload if isinstance(event.payload, dict) else {}
            positions = payload.get("positions")
            if isinstance(positions, list):
                snapshots = [p for p in positions if isinstance(p, PositionSnapshot)]
                effective, _ = split_effective_positions(snapshots, symbol=symbol)
                self._positions_by_symbol[symbol] = effective
                return

            position = payload.get("position")
            if isinstance(position, PositionSnapshot):
                current = [p for p in self._positions_by_symbol.get(symbol, []) if isinstance(p, PositionSnapshot)]
                pos_idx = self._position_idx(position)

                if position_has_effective_exposure(position):
                    if pos_idx != 0:
                        current = [p for p in current if self._position_idx(p) != pos_idx]
                    else:
                        # One-way mode updates frequently come with positionIdx=0.
                        current = [p for p in current if self._position_idx(p) != 0]
                    current.append(position)
                else:
                    if pos_idx != 0:
                        current = [p for p in current if self._position_idx(p) != pos_idx]
                    else:
                        # Bybit can emit size=0 placeholder rows with empty side; clear one-way residue.
                        current = [p for p in current if self._position_idx(p) != 0]
                        side_raw = str(payload.get("side_raw") or "").upper().strip()
                        if side_raw in ("BUY", "SELL", "LONG", "SHORT"):
                            current = [p for p in current if p.side.value != ("LONG" if side_raw in ("BUY", "LONG") else "SHORT")]

                effective, _ = split_effective_positions(current, symbol=symbol)
                self._positions_by_symbol[symbol] = effective
                return

        if event.event_type == ExchangeEventType.ORDER:
            payload = event.payload if isinstance(event.payload, dict) else {}
            order = payload.get("order")
            if not isinstance(order, OpenOrderSnapshot):
                return
            status = str(order.status or "").upper()
            current = [o for o in self._orders_by_symbol.get(symbol, []) if o.order_id != order.order_id]
            if status not in _TERMINAL_ORDER_STATUSES:
                current.append(order)
            self._orders_by_symbol[symbol] = current
            return

    def _ws_is_fresh(self) -> bool:
        if not self._ws_connected:
            return False
        if self._last_event_ts <= 0:
            return False
        return (time.time() - self._last_event_ts) <= self.max_event_staleness_sec

    def health(self) -> SyncHealth:
        ws_stale = False
        if self._last_event_ts > 0:
            ws_stale = (time.time() - self._last_event_ts) > self.max_event_staleness_sec
        snapshot_required = self._snapshot_required_global or bool(self._snapshot_required_symbols)
        return SyncHealth(
            ws_connected=self._ws_connected,
            ws_last_event_ts=float(self._last_event_ts),
            ws_stale=ws_stale,
            fallback_polling=(not self._ws_is_fresh()) or snapshot_required,
            snapshot_required=snapshot_required,
        )

    def snapshot(self, symbol: str) -> ExchangeSnapshot:
        norm_symbol = self._norm_symbol(symbol)
        now = time.time()
        force_poll = self._snapshot_required(norm_symbol)

        if self._ws_is_fresh() and not force_poll and self._account is not None:
            if norm_symbol in self._positions_by_symbol or norm_symbol in self._orders_by_symbol:
                effective_positions, _ = split_effective_positions(
                    list(self._positions_by_symbol.get(norm_symbol, [])),
                    symbol=norm_symbol,
                    size_epsilon=POSITION_SIZE_EPSILON,
                )
                snapshot = ExchangeSnapshot(
                    symbol=norm_symbol,
                    account=self._account,
                    positions=effective_positions,
                    open_orders=list(self._orders_by_symbol.get(norm_symbol, [])),
                    reconciled_at=now,
                )
                self._positions_by_symbol[norm_symbol] = list(effective_positions)
                self._polled_snapshots[norm_symbol] = snapshot
                self._last_poll_ts[norm_symbol] = now
                return snapshot

        last_poll = self._last_poll_ts.get(norm_symbol, 0.0)
        if not force_poll and norm_symbol in self._polled_snapshots and (now - last_poll) < self.poll_interval_sec:
            return self._polled_snapshots[norm_symbol]

        snapshot = self.reconciler.snapshot(norm_symbol)
        self._polled_snapshots[norm_symbol] = snapshot
        self._last_poll_ts[norm_symbol] = now

        # Polling snapshot is exchange truth. Use it to heal websocket in-memory state.
        self._account = snapshot.account
        effective_positions, _ = split_effective_positions(
            list(snapshot.positions),
            symbol=norm_symbol,
            size_epsilon=POSITION_SIZE_EPSILON,
        )
        self._positions_by_symbol[norm_symbol] = list(effective_positions)
        self._orders_by_symbol[norm_symbol] = list(snapshot.open_orders)
        self._clear_snapshot_requirement(norm_symbol)

        return snapshot
