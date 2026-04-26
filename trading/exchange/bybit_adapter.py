from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP
from typing import Any, Protocol

from trading.exchange.events import NormalizedExchangeEvent

from .bybit_client import BybitHttpClient
from .bybit_ws import BybitWebSocketConfig, BybitWebSocketStream
from .schemas import (
    AccountSnapshot,
    InstrumentRules,
    OpenOrderSnapshot,
    OrderIntent,
    OrderResult,
    OrderSide,
    PositionSide,
    PositionSnapshot,
    ProtectiveOrderResult,
)

logger = logging.getLogger(__name__)


class ExchangeAdapter(Protocol):
    def get_account(self) -> AccountSnapshot: ...

    def get_positions(self, symbol: str | None = None) -> list[PositionSnapshot]: ...

    def get_open_orders(self, symbol: str | None = None) -> list[OpenOrderSnapshot]: ...

    def get_mark_price(self, symbol: str) -> float: ...

    def get_instrument_rules(self, symbol: str) -> InstrumentRules: ...

    def place_market_order(self, intent: OrderIntent) -> OrderResult: ...

    def set_protective_orders(
        self,
        symbol: str,
        *,
        stop_loss: float,
        take_profit: float | None,
        position_idx: int,
        qty: float | None = None,
    ) -> ProtectiveOrderResult: ...

    def cancel_order(self, *, symbol: str, order_id: str = "", order_link_id: str = "") -> bool: ...


@dataclass
class BybitAdapterConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    demo: bool = False
    dry_run: bool = True
    recv_window: int = 20000
    hedge_mode: bool = False
    instrument_rules_ttl_sec: int = 900
    instrument_rules_max_age_sec: int = 3600
    ws_enabled: bool = True
    ws_private_enabled: bool = True
    ws_stale_after_sec: int = 25
    ws_reconnect_delay_sec: float = 1.0
    ws_open_timeout_sec: float = 12.0
    ws_close_timeout_sec: float = 6.0
    ws_ping_interval_sec: float = 30.0
    ws_ping_timeout_sec: float = 20.0
    ws_symbols: list[str] = field(default_factory=list)
    target_entry_leverage: float = 3.0
    tpsl_mode: str = ""
    sl_trigger_by: str = "MarkPrice"
    tp_trigger_by: str = "MarkPrice"


@dataclass
class _RulesCacheEntry:
    rules: InstrumentRules
    fetched_at: float


class InstrumentMetadataError(RuntimeError):
    pass


class BybitAdapter:
    """Bybit-specific adapter with schema normalization and precision safety."""

    def __init__(self, config: BybitAdapterConfig):
        self.config = config
        self.client = BybitHttpClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            testnet=config.testnet,
            demo=config.demo,
            dry_run=config.dry_run,
            recv_window=config.recv_window,
            tpsl_mode=config.tpsl_mode,
            sl_trigger_by=config.sl_trigger_by,
            tp_trigger_by=config.tp_trigger_by,
        )
        self._rules_cache: dict[str, _RulesCacheEntry] = {}
        self._applied_leverage_cache: dict[str, float] = {}
        self._ws_stream: BybitWebSocketStream | None = None
        self._demo_auto_fund_attempted = False
        self._init_ws()

    def _init_ws(self):
        if not self.config.ws_enabled:
            return
        ws_cfg = BybitWebSocketConfig(
            testnet=bool(self.config.testnet),
            demo=bool(self.config.demo),
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            symbols=[self.normalize_symbol(s) for s in (self.config.ws_symbols or ["BTCUSDT"])],
            private_stream_enabled=bool(self.config.ws_private_enabled and not self.config.dry_run),
            reconnect_delay_sec=float(self.config.ws_reconnect_delay_sec),
            stale_after_sec=max(5, int(self.config.ws_stale_after_sec)),
            open_timeout_sec=float(self.config.ws_open_timeout_sec),
            close_timeout_sec=float(self.config.ws_close_timeout_sec),
            ping_interval_sec=float(self.config.ws_ping_interval_sec),
            ping_timeout_sec=float(self.config.ws_ping_timeout_sec),
        )
        self._ws_stream = BybitWebSocketStream(ws_cfg)
        self._ws_stream.start()

    @property
    def private_auth_invalid(self) -> bool:
        return bool(getattr(self.client, "private_auth_invalid", False))

    @property
    def private_auth_invalid_reason(self) -> str:
        return str(getattr(self.client, "private_auth_invalid_reason", "") or "")

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return str(symbol).replace("/", "").upper().strip()

    @staticmethod
    def position_idx_for_side(position_side: PositionSide, hedge_mode: bool) -> int:
        if not hedge_mode:
            return 0
        return 1 if position_side == PositionSide.LONG else 2

    @staticmethod
    def _parse_side(raw_side: str) -> PositionSide:
        side = str(raw_side).upper().strip()
        if side in ("BUY", "LONG"):
            return PositionSide.LONG
        return PositionSide.SHORT

    @staticmethod
    def _parse_order_side(raw_side: str) -> OrderSide:
        side = str(raw_side).upper().strip()
        return OrderSide.BUY if side in ("BUY", "LONG") else OrderSide.SELL

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _extract_instrument_rules(cls, symbol: str, payload: dict) -> InstrumentRules:
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        items = result.get("list", []) if isinstance(result, dict) else []
        if not items:
            return InstrumentRules(symbol=symbol, tick_size=0.0, qty_step=0.0, min_qty=0.0, min_notional=0.0, max_qty=0.0)

        item = items[0]
        lot = item.get("lotSizeFilter", {}) if isinstance(item, dict) else {}
        price = item.get("priceFilter", {}) if isinstance(item, dict) else {}

        tick_size = cls._safe_float(price.get("tickSize"), 0.0)
        qty_step = cls._safe_float(lot.get("qtyStep"), 0.0)
        min_qty = cls._safe_float(lot.get("minOrderQty"), 0.0)
        min_notional = cls._safe_float(lot.get("minNotionalValue"), 0.0)
        max_order_qty = cls._safe_float(lot.get("maxOrderQty"), 0.0)
        max_market_qty = cls._safe_float(lot.get("maxMktOrderQty"), 0.0)
        positive_caps = [value for value in (max_order_qty, max_market_qty) if value > 0]
        max_qty = min(positive_caps) if positive_caps else 0.0

        return InstrumentRules(
            symbol=symbol,
            tick_size=tick_size,
            qty_step=qty_step,
            min_qty=min_qty,
            min_notional=min_notional,
            max_qty=max_qty,
        )

    @staticmethod
    def _validate_rules(rules: InstrumentRules):
        if rules.tick_size <= 0:
            raise InstrumentMetadataError(f"invalid_tick_size:{rules.symbol}")
        if rules.qty_step <= 0:
            raise InstrumentMetadataError(f"invalid_qty_step:{rules.symbol}")
        if rules.min_qty <= 0:
            raise InstrumentMetadataError(f"invalid_min_qty:{rules.symbol}")
        if rules.min_notional <= 0:
            raise InstrumentMetadataError(f"invalid_min_notional:{rules.symbol}")

    @staticmethod
    def round_qty(qty: float, qty_step: float) -> float:
        if qty_step <= 0:
            return float(max(qty, 0.0))
        try:
            qty_d = Decimal(str(max(qty, 0.0)))
            step_d = Decimal(str(qty_step))
        except (InvalidOperation, ValueError):
            return float(max(qty, 0.0))
        if step_d <= 0:
            return float(max(qty, 0.0))
        units = (qty_d / step_d).to_integral_value(rounding=ROUND_DOWN)
        rounded = (units * step_d).normalize()
        return float(max(rounded, Decimal("0")))

    @staticmethod
    def round_price(price: float, tick_size: float) -> float:
        if tick_size <= 0:
            return float(max(price, 0.0))
        try:
            price_d = Decimal(str(max(price, 0.0)))
            step_d = Decimal(str(tick_size))
        except (InvalidOperation, ValueError):
            return float(max(price, 0.0))
        if step_d <= 0:
            return float(max(price, 0.0))
        units = (price_d / step_d).to_integral_value(rounding=ROUND_HALF_UP)
        rounded = (units * step_d).normalize()
        return float(max(rounded, Decimal("0")))

    def set_ws_symbols(self, symbols: list[str]):
        clean = [self.normalize_symbol(s) for s in symbols if s]
        target = sorted(set(clean)) or ["BTCUSDT"]
        current = sorted(set(self.config.ws_symbols or []))
        if target == current:
            return
        self.config.ws_symbols = target
        if self._ws_stream is not None:
            self._ws_stream.close()
            self._ws_stream = None
            self._init_ws()

    def force_ws_reconnect(self):
        if self._ws_stream is None:
            return
        self._ws_stream.close()
        self._ws_stream = None
        self._init_ws()

    def close(self):
        if self._ws_stream is not None:
            self._ws_stream.close()
        self.client.close()

    def drain_ws_raw_events(self) -> list[dict[str, Any]]:
        if self._ws_stream is None:
            return []
        drain_raw = getattr(self._ws_stream, "drain_raw_messages", None)
        if not callable(drain_raw):
            return []
        return [item for item in (drain_raw() or []) if isinstance(item, dict)]

    def drain_ws_events(self) -> list[NormalizedExchangeEvent]:
        if self._ws_stream is None:
            return []
        return self._ws_stream.drain_events()

    def metadata_health(self) -> dict:
        now = time.time()
        fresh = 0
        stale = 0
        ttl = max(1, int(self.config.instrument_rules_ttl_sec))
        for entry in self._rules_cache.values():
            age = now - entry.fetched_at
            if age <= ttl:
                fresh += 1
            else:
                stale += 1

        health = {
            "cached_symbols": len(self._rules_cache),
            "fresh_symbols": fresh,
            "stale_symbols": stale,
            "ttl_sec": ttl,
        }
        if self._ws_stream is not None:
            health["ws"] = self._ws_stream.health()
        return health

    def get_account(self) -> AccountSnapshot:
        if self.config.dry_run:
            return AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0)

        payload = self.client.request_private(
            "GET",
            "/v5/account/wallet-balance",
            params={"accountType": "UNIFIED", "coin": "USDT"},
        )
        snapshot = self._extract_account_snapshot(payload)
        if (
            self.config.demo
            and not self.config.dry_run
            and snapshot.equity_usdt <= 0
            and not self.private_auth_invalid
            and not self._demo_auto_fund_attempted
        ):
            self._demo_auto_fund_attempted = True
            try:
                fund_payload = self.client.apply_demo_funds(usdt_amount="100000")
                ret_code = int(fund_payload.get("retCode", 1)) if isinstance(fund_payload, dict) else 1
                if ret_code == 0:
                    logger.info("demo_auto_fund ok amount=100000USDT")
                    time.sleep(0.8)
                    payload = self.client.request_private(
                        "GET",
                        "/v5/account/wallet-balance",
                        params={"accountType": "UNIFIED", "coin": "USDT"},
                    )
                    snapshot = self._extract_account_snapshot(payload)
                else:
                    logger.warning(
                        "demo_auto_fund failed retCode=%s retMsg=%s",
                        fund_payload.get("retCode") if isinstance(fund_payload, dict) else None,
                        fund_payload.get("retMsg") if isinstance(fund_payload, dict) else None,
                    )
            except Exception as exc:
                logger.warning("demo_auto_fund exception: %s", exc)
        return snapshot

    def _extract_account_snapshot(self, payload: dict[str, Any] | None) -> AccountSnapshot:
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        items = result.get("list", []) if isinstance(result, dict) else []
        if not items:
            return AccountSnapshot(equity_usdt=0.0, available_balance_usdt=0.0)

        primary = items[0] if isinstance(items[0], dict) else {}
        coins = primary.get("coin", []) if isinstance(primary, dict) else []

        usdt_equity = 0.0
        usdt_available = 0.0
        for coin in coins:
            if str(coin.get("coin", "")).upper() != "USDT":
                continue
            usdt_equity = self._safe_float(coin.get("equity"), 0.0)
            usdt_available = self._safe_float(coin.get("availableToWithdraw"), 0.0)
            if usdt_available <= 0:
                usdt_available = self._safe_float(coin.get("walletBalance"), usdt_equity)
            break

        total_equity = self._safe_float(primary.get("totalEquity"), 0.0)
        total_available = self._safe_float(primary.get("totalAvailableBalance"), 0.0)
        if total_available <= 0:
            total_available = self._safe_float(primary.get("totalWalletBalance"), total_equity)

        equity = total_equity if total_equity > 0 else usdt_equity
        available = total_available if total_available > 0 else usdt_available
        if available <= 0:
            available = equity

        return AccountSnapshot(equity_usdt=max(equity, 0.0), available_balance_usdt=max(available, 0.0))

    def get_positions(self, symbol: str | None = None) -> list[PositionSnapshot]:
        rows = self.client.get_open_positions(symbol=self.normalize_symbol(symbol) if symbol else None)
        out: list[PositionSnapshot] = []
        for row in rows:
            qty = self._safe_float(row.get("size", 0.0))
            if qty <= 0:
                continue
            out.append(
                PositionSnapshot(
                    symbol=self.normalize_symbol(row.get("symbol", "")),
                    side=self._parse_side(row.get("side", "")),
                    qty=qty,
                    entry_price=self._safe_float(row.get("entry_price", row.get("entryPrice", 0.0)), 0.0),
                    liq_price=self._safe_float(row.get("liq_price", row.get("liqPrice", 0.0)), 0.0),
                    leverage=self._safe_float(row.get("leverage", 0.0)),
                    position_idx=int(self._safe_float(row.get("positionIdx", 0))),
                    stop_loss=self._safe_float(row.get("stopLoss", 0.0), 0.0) or None,
                )
            )
        return out

    def get_open_orders(self, symbol: str | None = None) -> list[OpenOrderSnapshot]:
        rows = self.client.get_open_orders(symbol=self.normalize_symbol(symbol) if symbol else None)
        out: list[OpenOrderSnapshot] = []
        for row in rows:
            out.append(
                OpenOrderSnapshot(
                    symbol=self.normalize_symbol(row.get("symbol", "")),
                    order_id=str(row.get("orderId") or ""),
                    order_link_id=str(row.get("orderLinkId") or ""),
                    side=self._parse_order_side(row.get("side", "SELL")),
                    qty=self._safe_float(row.get("qty", 0.0), 0.0),
                    reduce_only=bool(row.get("reduceOnly", False)),
                    position_idx=int(self._safe_float(row.get("positionIdx", 0))),
                    status=str(row.get("orderStatus", "")),
                    created_ts=self._safe_float(row.get("createdTime"), 0.0) / 1000.0,
                    updated_ts=self._safe_float(row.get("updatedTime"), 0.0) / 1000.0,
                )
            )
        return out

    def get_mark_price(self, symbol: str) -> float:
        payload = self.client.get_ticker_meta(self.normalize_symbol(symbol))
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        rows = result.get("list", []) if isinstance(result, dict) else []
        if not rows:
            return 0.0
        row = rows[0] if isinstance(rows[0], dict) else {}
        for key in ("markPrice", "lastPrice", "indexPrice"):
            val = self._safe_float(row.get(key), 0.0)
            if val > 0:
                return val
        return 0.0

    def get_instrument_rules(self, symbol: str, *, force_refresh: bool = False) -> InstrumentRules:
        norm = self.normalize_symbol(symbol)
        now = time.time()
        ttl = max(1, int(self.config.instrument_rules_ttl_sec))
        max_age = max(ttl, int(self.config.instrument_rules_max_age_sec))

        entry = self._rules_cache.get(norm)
        if not force_refresh and entry is not None and (now - entry.fetched_at) <= ttl:
            return entry.rules

        try:
            payload = self.client.request_public(
                "/v5/market/instruments-info",
                params={"category": "linear", "symbol": norm},
            )
            rules = self._extract_instrument_rules(norm, payload)
            self._validate_rules(rules)
            self._rules_cache[norm] = _RulesCacheEntry(rules=rules, fetched_at=now)
            return rules
        except Exception as exc:
            if entry is not None and (now - entry.fetched_at) <= max_age:
                return entry.rules
            raise InstrumentMetadataError(f"instrument_rules_unavailable:{norm}") from exc

    def ensure_position_leverage(self, symbol: str, leverage: float) -> bool:
        target = float(leverage or 0.0)
        if target <= 0:
            return False
        norm = self.normalize_symbol(symbol)
        cached = float(self._applied_leverage_cache.get(norm, 0.0) or 0.0)
        if abs(cached - target) <= 1e-9:
            return True
        resp = self.client.set_position_leverage(
            symbol=norm,
            buy_leverage=target,
            sell_leverage=target,
        )
        ret_code = int(resp.get("retCode", 1)) if isinstance(resp, dict) else 1
        ret_msg = str(resp.get("retMsg", "") if isinstance(resp, dict) else "")
        normalized_msg = ret_msg.lower()
        ok = ret_code == 0 or ret_code == 110043 or "not modified" in normalized_msg or "same leverage" in normalized_msg
        if ok:
            self._applied_leverage_cache[norm] = target
            logger.info("bybit leverage aligned symbol=%s leverage=%s retCode=%s", norm, target, ret_code)
            return True
        logger.warning(
            "bybit leverage alignment failed symbol=%s leverage=%s retCode=%s retMsg=%s",
            norm,
            target,
            ret_code,
            ret_msg,
        )
        return False

    def get_account_mode_details(self) -> dict[str, Any]:
        payload = self.client.get_account_info()
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        details = dict(result) if isinstance(result, dict) else {}
        details["retCode"] = int(payload.get("retCode", 1)) if isinstance(payload, dict) else 1
        details["retMsg"] = str(payload.get("retMsg", "")) if isinstance(payload, dict) else ""
        return details

    def get_positions_metadata(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = self.normalize_symbol(symbol)
        payload = self.client.request_private("GET", "/v5/position/list", params=params)
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        rows = result.get("list", []) if isinstance(result, dict) else []
        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                out.append(dict(row))
        return out

    def get_open_orders_metadata(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if not symbol:
            return []
        params: dict[str, Any] = {"category": "linear"}
        params["symbol"] = self.normalize_symbol(symbol)
        payload = self.client.request_private("GET", "/v5/order/realtime", params=params)
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        rows = result.get("list", []) if isinstance(result, dict) else []
        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                out.append(dict(row))
        return out

    def place_limit_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        qty: float,
        price: float,
        reduce_only: bool,
        position_idx: int,
        client_order_id: str | None = None,
        close_on_trigger: bool | None = None,
    ) -> OrderResult:
        resp = self.client.place_order_limit(
            symbol=self.normalize_symbol(symbol),
            side="buy" if side == OrderSide.BUY else "sell",
            qty=float(qty),
            price=float(price),
            reduce_only=bool(reduce_only),
            position_idx=int(position_idx),
            order_link_id=client_order_id,
            close_on_trigger=close_on_trigger,
        )

        ok = isinstance(resp, dict) and resp.get("retCode", 1) == 0
        result = resp.get("result", {}) if isinstance(resp, dict) else {}
        avg_price = self._safe_float(result.get("avgPrice"), 0.0)
        filled_qty = self._safe_float(result.get("cumExecQty", 0.0), 0.0)
        remaining_qty = max(0.0, float(qty) - float(filled_qty))
        status = str(result.get("orderStatus") or ("New" if ok else "Rejected"))

        return OrderResult(
            success=bool(ok),
            order_id=str(result.get("orderId") or ""),
            order_link_id=str(result.get("orderLinkId") or client_order_id or ""),
            avg_price=avg_price,
            filled_qty=filled_qty,
            remaining_qty=remaining_qty,
            status=status,
            raw=resp if isinstance(resp, dict) else {},
            error=None if ok else str(resp.get("retMsg") if isinstance(resp, dict) else "order_failed"),
        )

    def place_market_order(self, intent: OrderIntent) -> OrderResult:
        side = "buy" if intent.side == OrderSide.BUY else "sell"
        resp = self.client.place_order_market(
            symbol=self.normalize_symbol(intent.symbol),
            side=side,
            qty=float(intent.qty),
            reduce_only=bool(intent.reduce_only),
            position_idx=int(intent.position_idx),
            order_link_id=intent.client_order_id,
            close_on_trigger=intent.close_on_trigger,
        )

        ok = isinstance(resp, dict) and resp.get("retCode", 1) == 0
        result = resp.get("result", {}) if isinstance(resp, dict) else {}
        avg_price = self._safe_float(result.get("avgPrice"), 0.0)
        filled_qty = self._safe_float(result.get("cumExecQty", result.get("qty")), float(intent.qty))
        remaining_qty = max(0.0, float(intent.qty) - float(filled_qty))
        status = str(result.get("orderStatus") or ("Filled" if ok else "Rejected"))

        return OrderResult(
            success=bool(ok),
            order_id=str(result.get("orderId") or ""),
            order_link_id=str(result.get("orderLinkId") or intent.client_order_id or ""),
            avg_price=avg_price,
            filled_qty=filled_qty,
            remaining_qty=remaining_qty,
            status=status,
            raw=resp if isinstance(resp, dict) else {},
            error=None if ok else str(resp.get("retMsg") if isinstance(resp, dict) else "order_failed"),
        )

    def set_protective_orders(
        self,
        symbol: str,
        *,
        stop_loss: float,
        take_profit: float | None,
        position_idx: int,
        qty: float | None = None,
    ) -> ProtectiveOrderResult:
        rounded_stop_loss = float(stop_loss)
        rounded_take_profit = float(take_profit) if take_profit is not None else None
        try:
            rules = self.get_instrument_rules(symbol)
            rounded_candidate = self.round_price(stop_loss, rules.tick_size)
            if rounded_candidate > 0:
                rounded_stop_loss = rounded_candidate
            if take_profit is not None:
                tp_candidate = self.round_price(take_profit, rules.tick_size)
                if tp_candidate > 0:
                    rounded_take_profit = tp_candidate
        except Exception:
            rounded_stop_loss = float(stop_loss)
            rounded_take_profit = float(take_profit) if take_profit is not None else None

        resp = self.client.set_trading_stop(
            symbol=self.normalize_symbol(symbol),
            stop_loss=rounded_stop_loss,
            take_profit=rounded_take_profit,
            position_idx=int(position_idx),
            qty=float(qty) if qty is not None and qty > 0 else None,
        )
        ok = isinstance(resp, dict) and resp.get("retCode", 1) == 0
        return ProtectiveOrderResult(
            success=bool(ok),
            raw=resp if isinstance(resp, dict) else {},
            error=None if ok else str(resp.get("retMsg") if isinstance(resp, dict) else "set_stop_failed"),
        )

    def cancel_order(self, *, symbol: str, order_id: str = "", order_link_id: str = "") -> bool:
        resp = self.client.cancel_order(symbol=self.normalize_symbol(symbol), order_id=order_id, order_link_id=order_link_id)
        return isinstance(resp, dict) and resp.get("retCode", 1) == 0





