from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from trading.exchange.bybit_endpoints import resolve_private_ws_url, resolve_public_ws_url
from trading.exchange.events import ExchangeEventType, NormalizedExchangeEvent
from trading.exchange.schemas import AccountSnapshot, OpenOrderSnapshot, OrderSide, PositionSide, PositionSnapshot

try:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect as ws_connect
except Exception:
    ConnectionClosed = Exception
    ws_connect = None


@dataclass
class BybitWebSocketConfig:
    testnet: bool = True
    demo: bool = False
    api_key: str = ""
    api_secret: str = ""
    symbols: list[str] = field(default_factory=list)
    private_stream_enabled: bool = False
    reconnect_delay_sec: float = 1.0
    max_queue_size: int = 2000
    stale_after_sec: int = 25


class BybitWebSocketStream:
    """Real Bybit V5 WS stream (public + optional private) normalized to internal events."""

    def __init__(self, config: BybitWebSocketConfig):
        self.config = config
        self._events: deque[NormalizedExchangeEvent] = deque(maxlen=max(100, int(config.max_queue_size)))
        self._raw_messages: deque[dict[str, Any]] = deque(maxlen=max(100, int(config.max_queue_size)))
        self._lock = threading.Lock()
        self._running = False
        self._threads: list[threading.Thread] = []
        self._stop_evt = threading.Event()
        self._last_msg_ts: dict[str, float] = {"public": 0.0, "private": 0.0}
        self._last_gap_emit_ts = 0.0

    @staticmethod
    def _norm_symbol(symbol: str) -> str:
        return str(symbol).replace("/", "").upper().strip()

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_order_side(value: str) -> OrderSide:
        raw = str(value or "").upper().strip()
        return OrderSide.BUY if raw == "BUY" else OrderSide.SELL

    @staticmethod
    def _parse_position_side(value: str) -> PositionSide:
        raw = str(value or "").upper().strip()
        return PositionSide.LONG if raw in ("BUY", "LONG") else PositionSide.SHORT

    def _push(self, event_type: ExchangeEventType, *, symbol: str | None = None, payload: dict | None = None):
        event = NormalizedExchangeEvent(
            event_type=event_type,
            symbol=self._norm_symbol(symbol) if symbol else None,
            payload=payload or {},
            ts=time.time(),
        )
        with self._lock:
            self._events.append(event)

    def _push_raw(self, *, channel: str, payload: dict | None):
        with self._lock:
            self._raw_messages.append(
                {
                    "ts": time.time(),
                    "channel": str(channel),
                    "payload": payload if isinstance(payload, dict) else {},
                }
            )

    def _push_snapshot_required(self, reason: str):
        now = time.time()
        if (now - self._last_gap_emit_ts) < 1.0:
            return
        self._last_gap_emit_ts = now
        self._push(ExchangeEventType.SNAPSHOT_REQUIRED, payload={"reason": reason})

    def start(self):
        if ws_connect is None:
            self._push(ExchangeEventType.ERROR, payload={"reason": "websocket_dependency_missing"})
            return

        if self._running:
            return
        self._running = True
        self._stop_evt.clear()

        t_public = threading.Thread(target=self._public_loop, name="bybit-ws-public", daemon=True)
        self._threads.append(t_public)
        t_public.start()

        if self.config.private_stream_enabled and self.config.api_key and self.config.api_secret:
            t_private = threading.Thread(target=self._private_loop, name="bybit-ws-private", daemon=True)
            self._threads.append(t_private)
            t_private.start()

    def close(self):
        self._running = False
        self._stop_evt.set()
        for thread in list(self._threads):
            thread.join(timeout=1.0)
        self._threads.clear()

    def health(self) -> dict:
        return {
            "running": self._running,
            "public_last_msg_ts": float(self._last_msg_ts.get("public", 0.0)),
            "private_last_msg_ts": float(self._last_msg_ts.get("private", 0.0)),
        }

    def drain_raw_messages(self) -> list[dict[str, Any]]:
        with self._lock:
            out = list(self._raw_messages)
            self._raw_messages.clear()
        return out

    def drain_events(self) -> list[NormalizedExchangeEvent]:
        with self._lock:
            out = list(self._events)
            self._events.clear()

        now = time.time()
        stale_after = max(5, int(self.config.stale_after_sec))
        latest = max(self._last_msg_ts.get("public", 0.0), self._last_msg_ts.get("private", 0.0))
        if self._running and latest > 0 and (now - latest) > stale_after:
            out.append(
                NormalizedExchangeEvent(
                    event_type=ExchangeEventType.SNAPSHOT_REQUIRED,
                    payload={"reason": "ws_stale"},
                    ts=now,
                )
            )
        return out

    def _public_endpoint(self) -> str:
        return resolve_public_ws_url(testnet=bool(self.config.testnet))

    def _private_endpoint(self) -> str:
        return resolve_private_ws_url(testnet=bool(self.config.testnet), demo=bool(self.config.demo))

    def _auth_payload(self) -> dict:
        expires = int((time.time() + 10) * 1000)
        payload = f"GET/realtime{expires}"
        sign = hmac.new(
            self.config.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {"op": "auth", "args": [self.config.api_key, expires, sign]}

    def _public_subscribe_payloads(self) -> list[dict]:
        topics = [f"tickers.{self._norm_symbol(symbol)}" for symbol in self.config.symbols if symbol]
        topics = sorted(set(topics))
        if not topics:
            topics = ["tickers.BTCUSDT"]
        batch_size = 80
        return [{"op": "subscribe", "args": topics[i : i + batch_size]} for i in range(0, len(topics), batch_size)]

    @staticmethod
    def _private_subscribe_payload() -> dict:
        return {"op": "subscribe", "args": ["order", "position", "wallet"]}

    def _public_loop(self):
        self._connection_loop(channel="public", url=self._public_endpoint(), is_private=False)

    def _private_loop(self):
        self._connection_loop(channel="private", url=self._private_endpoint(), is_private=True)

    def _connection_loop(self, *, channel: str, url: str, is_private: bool):
        base_backoff = max(0.5, float(self.config.reconnect_delay_sec))
        backoff = base_backoff
        while self._running and not self._stop_evt.is_set():
            try:
                self._push(ExchangeEventType.RECONNECTING, payload={"channel": channel})
                with ws_connect(
                    url,
                    open_timeout=8,
                    close_timeout=4,
                    ping_interval=20,
                    ping_timeout=10,
                    proxy=None,
                ) as ws:
                    backoff = base_backoff
                    self._push(ExchangeEventType.CONNECTED, payload={"channel": channel})
                    self._push_snapshot_required(f"{channel}_connected")
                    if is_private:
                        ws.send(json.dumps(self._auth_payload(), separators=(",", ":")))
                        ws.send(json.dumps(self._private_subscribe_payload(), separators=(",", ":")))
                    else:
                        for payload in self._public_subscribe_payloads():
                            ws.send(json.dumps(payload, separators=(",", ":")))

                    while self._running and not self._stop_evt.is_set():
                        try:
                            msg = ws.recv(timeout=1)
                        except TimeoutError:
                            self._push(ExchangeEventType.HEARTBEAT, payload={"channel": channel})
                            continue

                        if msg is None:
                            continue
                        self._last_msg_ts[channel] = time.time()
                        payload = self._parse_json(msg)
                        self._push_raw(channel=channel, payload=payload)
                        for event in self._normalize_message(channel=channel, payload=payload):
                            self._push(event.event_type, symbol=event.symbol, payload=event.payload)
            except ConnectionClosed as exc:
                self._push(ExchangeEventType.DISCONNECTED, payload={"channel": channel, "error": str(exc)})
                self._push_snapshot_required(f"{channel}_disconnect")
                time.sleep(min(backoff, 5.0))
                backoff = min(backoff * 1.5, 10.0)
            except Exception as exc:
                self._push(ExchangeEventType.ERROR, payload={"channel": channel, "error": str(exc)})
                self._push_snapshot_required(f"{channel}_error")
                time.sleep(min(backoff, 5.0))
                backoff = min(backoff * 1.5, 10.0)

    @staticmethod
    def _parse_json(raw: str | bytes) -> dict:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _normalize_message(self, *, channel: str, payload: dict) -> Iterable[NormalizedExchangeEvent]:
        if not payload:
            return []

        if "success" in payload and "op" in payload:
            if payload.get("success") is False:
                err = str(payload.get("ret_msg") or payload.get("retMsg") or "ws_op_failed")
                return [
                    NormalizedExchangeEvent(
                        event_type=ExchangeEventType.ERROR,
                        payload={"channel": channel, "error": err, "op": payload.get("op")},
                    ),
                    NormalizedExchangeEvent(
                        event_type=ExchangeEventType.SNAPSHOT_REQUIRED,
                        payload={"reason": f"{channel}_op_failed", "op": payload.get("op")},
                    ),
                ]
            return []

        if payload.get("op") == "pong":
            return [NormalizedExchangeEvent(event_type=ExchangeEventType.HEARTBEAT, payload={"channel": channel})]

        topic = str(payload.get("topic") or "")
        data = payload.get("data")
        ws_type = str(payload.get("type") or "")

        if topic.startswith("tickers."):
            symbol = topic.split(".", 1)[1] if "." in topic else ""
            tick = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else {})
            mark_price = self._safe_float((tick or {}).get("markPrice"), 0.0)
            last_price = self._safe_float((tick or {}).get("lastPrice"), 0.0)
            return [
                NormalizedExchangeEvent(
                    event_type=ExchangeEventType.MARKET,
                    symbol=self._norm_symbol(symbol),
                    payload={
                        "channel": channel,
                        "mark_price": mark_price,
                        "last_price": last_price,
                        "ws_type": ws_type,
                    },
                )
            ]

        if topic.startswith("wallet"):
            rows = data if isinstance(data, list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                coins = row.get("coin", [])
                if not isinstance(coins, list):
                    continue
                for coin in coins:
                    if str(coin.get("coin", "")).upper() != "USDT":
                        continue
                    account = AccountSnapshot(
                        equity_usdt=self._safe_float(coin.get("equity"), 0.0),
                        available_balance_usdt=self._safe_float(coin.get("availableToWithdraw"), 0.0),
                        updated_at=time.time(),
                    )
                    return [
                        NormalizedExchangeEvent(
                            event_type=ExchangeEventType.ACCOUNT,
                            payload={"channel": channel, "account": account, "ws_type": ws_type},
                        )
                    ]
            return []

        if topic.startswith("position"):
            rows = data if isinstance(data, list) else []
            out: list[NormalizedExchangeEvent] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol = self._norm_symbol(row.get("symbol", ""))
                if not symbol:
                    continue
                position = PositionSnapshot(
                    symbol=symbol,
                    side=self._parse_position_side(row.get("side", "")),
                    qty=self._safe_float(row.get("size"), 0.0),
                    entry_price=self._safe_float(row.get("entryPrice"), 0.0),
                    liq_price=self._safe_float(row.get("liqPrice"), 0.0),
                    leverage=self._safe_float(row.get("leverage"), 0.0),
                    position_idx=self._safe_int(row.get("positionIdx"), 0),
                    stop_loss=self._safe_float(row.get("stopLoss"), 0.0) or None,
                )
                out.append(
                    NormalizedExchangeEvent(
                        event_type=ExchangeEventType.POSITION,
                        symbol=symbol,
                        payload={"channel": channel, "position": position, "source": "ws", "ws_type": ws_type, "side_raw": str(row.get("side", ""))},
                    )
                )
            return out

        if topic.startswith("order"):
            rows = data if isinstance(data, list) else []
            out: list[NormalizedExchangeEvent] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol = self._norm_symbol(row.get("symbol", ""))
                if not symbol:
                    continue
                qty = self._safe_float(row.get("leavesQty", row.get("qty", 0.0)), 0.0)
                order = OpenOrderSnapshot(
                    symbol=symbol,
                    order_id=str(row.get("orderId") or ""),
                    order_link_id=str(row.get("orderLinkId") or ""),
                    side=self._parse_order_side(row.get("side", "SELL")),
                    qty=max(0.0, qty),
                    reduce_only=bool(row.get("reduceOnly", False)),
                    position_idx=self._safe_int(row.get("positionIdx"), 0),
                    status=str(row.get("orderStatus") or ""),
                    created_ts=self._safe_float(row.get("createdTime"), 0.0) / 1000.0,
                    updated_ts=self._safe_float(row.get("updatedTime"), 0.0) / 1000.0,
                )
                out.append(
                    NormalizedExchangeEvent(
                        event_type=ExchangeEventType.ORDER,
                        symbol=symbol,
                        payload={"channel": channel, "order": order, "source": "ws", "ws_type": ws_type},
                    )
                )
            return out

        return []

