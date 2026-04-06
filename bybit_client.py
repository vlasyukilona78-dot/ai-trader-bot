import hashlib
import hmac
import json
import logging
import time
from typing import Any
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)


class BybitClient:
    """Minimal Bybit V5 client used by the bot."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        sandbox: bool = False,
        demo: bool = False,
        dry_run: bool = True,
        timeout: int = 12,
        recv_window: int = 20000,
        category: str = "linear",
    ):
        self.api_key = str(api_key or "").strip()
        self.api_secret = str(api_secret or "").strip()
        self.sandbox = sandbox
        self.demo = demo
        self.dry_run = dry_run
        self.timeout = timeout
        self.recv_window = str(recv_window)
        self.category = category
        self.base_url = "https://api-testnet.bybit.com" if sandbox else ("https://api-demo.bybit.com" if demo else "https://api.bybit.com")
        self.public_base_url = "https://api-testnet.bybit.com" if sandbox else "https://api.bybit.com"
        self._private_auth_invalid = False
        self._private_auth_invalid_reason = ""
        self._private_auth_invalid_logged = False
        self._time_offset_ms = 0
        self._last_time_sync_monotonic = 0.0

        self._sess = requests.Session()
        self._sess.headers.update(
            {
                "User-Agent": "koteika-bot/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        logger.info(
            "BybitClient init -> %s | dry_run=%s | key=%s",
            "SANDBOX" if sandbox else ("DEMO" if demo else "PRODUCTION"),
            dry_run,
            (self.api_key[:4] + "...") if self.api_key else None,
        )

    def _refresh_time_offset(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_time_sync_monotonic) < 30:
            return
        try:
            response = self._sess.get(f"{self.public_base_url}/v5/market/time", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            result = data.get("result", {}) if isinstance(data, dict) else {}
            server_ms = 0
            if isinstance(result, dict):
                time_nano = result.get("timeNano")
                time_second = result.get("timeSecond")
                if time_nano not in (None, "", 0, "0"):
                    server_ms = int(int(time_nano) / 1_000_000)
                elif time_second not in (None, "", 0, "0"):
                    server_ms = int(time_second) * 1000
            if server_ms > 0:
                self._time_offset_ms = server_ms - int(time.time() * 1000)
                self._last_time_sync_monotonic = now
        except Exception as exc:
            logger.debug("Bybit time sync skipped: %s", exc)

    def close(self):
        self._sess.close()

    @property
    def private_auth_invalid(self) -> bool:
        return bool(self._private_auth_invalid)

    @property
    def private_auth_invalid_reason(self) -> str:
        return str(self._private_auth_invalid_reason or "")

    def _runtime_label(self) -> str:
        if self.sandbox:
            return "testnet"
        if self.demo:
            return "demo"
        return "mainnet"

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").upper()

    @staticmethod
    def _canonical_query(params: dict[str, Any] | None) -> str:
        return urlencode(sorted((params or {}).items()))

    @staticmethod
    def _canonical_json_body(json_body: dict[str, Any] | None) -> str:
        payload = json_body or {}
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    def _sign(self, timestamp: str, payload: str) -> str:
        raw = f"{timestamp}{self.api_key}{self.recv_window}{payload}"
        return hmac.new(self.api_secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()

    def _build_auth(
        self,
        method: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        *,
        query_string: str | None = None,
        body_string: str | None = None,
    ):
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Private call requires API key/secret")

        self._refresh_time_offset()
        timestamp = str(int(time.time() * 1000 + self._time_offset_ms))
        query = query_string if query_string is not None else self._canonical_query(params)
        body = body_string if body_string is not None else self._canonical_json_body(json_body)
        sign_payload = query if method.upper() == "GET" else body
        signature = self._sign(timestamp, sign_payload)

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": self.recv_window,
            "X-BAPI-SIGN": signature,
        }
        return headers, timestamp, sign_payload, signature

    def _log_testnet_post_sign_diag(
        self,
        *,
        endpoint: str,
        timestamp: str,
        sign_body: str,
        wire_body: str,
        signature: str,
    ):
        if not self.sandbox:
            return

        sig = str(signature or "")
        sig_head = sig[:8]
        sig_tail = sig[-8:] if len(sig) > 8 else sig
        bodies_identical = sign_body.encode("utf-8") == wire_body.encode("utf-8")

        logger.info(
            "BybitV5 POST sign diag endpoint=%s ts=%s recv_window=%s sign_body=%s wire_body=%s bodies_identical=%s sig_len=%d sig_head=%s sig_tail=%s",
            endpoint,
            timestamp,
            self.recv_window,
            sign_body,
            wire_body,
            bodies_identical,
            len(sig),
            sig_head,
            sig_tail,
        )

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        private: bool = False,
        json_body: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        base_url = self.base_url if private else self.public_base_url
        url = f"{base_url}{path}"
        method = method.upper()

        if private and self._private_auth_invalid:
            return {
                "retCode": 10003,
                "retMsg": self._private_auth_invalid_reason or "private_auth_invalid",
                "result": {},
            }

        for attempt in range(1, max_retries + 1):
            try:
                headers = None
                timestamp = ""
                signature = ""

                query_string = self._canonical_query(params) if method == "GET" else ""
                body_string = self._canonical_json_body(json_body) if method != "GET" else ""

                if private:
                    headers, timestamp, _, signature = self._build_auth(
                        method,
                        params,
                        json_body,
                        query_string=query_string,
                        body_string=body_string,
                    )

                request_kwargs: dict[str, Any] = {
                    "headers": headers,
                    "timeout": self.timeout,
                }
                if params:
                    request_kwargs["params"] = params
                if method != "GET" and json_body is not None:
                    # Use exactly the same canonical string for signature and HTTP body.
                    request_kwargs["data"] = body_string

                if private and method != "GET":
                    wire_body = str(request_kwargs.get("data", ""))
                    self._log_testnet_post_sign_diag(
                        endpoint=path,
                        timestamp=timestamp,
                        sign_body=body_string,
                        wire_body=wire_body,
                        signature=signature,
                    )

                response = self._sess.request(method, url, **request_kwargs)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and data.get("retCode") == 10002 and private:
                    self._refresh_time_offset(force=True)
                    logger.warning(
                        "Bybit timestamp drift detected mode=%s path=%s recv_window=%s offset_ms=%s attempt=%d",
                        self._runtime_label(),
                        path,
                        self.recv_window,
                        self._time_offset_ms,
                        attempt,
                    )
                    continue
                if isinstance(data, dict) and data.get("retCode") == 10003 and private:
                    ret_msg = str(data.get("retMsg") or "API key is invalid.")
                    self._private_auth_invalid = True
                    self._private_auth_invalid_reason = f"{ret_msg} mode={self._runtime_label()} base_url={base_url}"
                    if not self._private_auth_invalid_logged:
                        logger.error(
                            "Bybit private auth disabled mode=%s base_url=%s path=%s retMsg=%s",
                            self._runtime_label(),
                            base_url,
                            path,
                            ret_msg,
                        )
                        self._private_auth_invalid_logged = True
                    return data
                if isinstance(data, dict) and data.get("retCode") not in (None, 0):
                    logger.warning(
                        "Bybit API returned retCode=%s retMsg=%s for %s",
                        data.get("retCode"),
                        data.get("retMsg"),
                        path,
                    )
                return data
            except requests.exceptions.Timeout as exc:
                logger.warning("Bybit timeout attempt %d for %s: %s", attempt, path, exc)
            except requests.exceptions.RequestException as exc:
                logger.warning("Bybit request error attempt %d for %s: %s", attempt, path, exc)
            except ValueError as exc:
                logger.warning("Bybit JSON decode error attempt %d for %s: %s", attempt, path, exc)
            except Exception as exc:
                logger.exception("Bybit unexpected error attempt %d for %s: %s", attempt, path, exc)
            time.sleep(min(2**attempt, 8))

        logger.error("Bybit request failed for %s after %d attempts", path, max_retries)
        return {}

    def get_klines(self, symbol: str, interval: str = "1", limit: int = 200) -> dict[str, Any]:
        params = {
            "category": self.category,
            "symbol": self._normalize_symbol(symbol),
            "interval": str(interval),
            "limit": int(limit),
        }
        return self._request("GET", "/v5/market/kline", params=params)

    def get_ticker_meta(self, symbol: str) -> dict[str, Any]:
        params = {"category": self.category, "symbol": self._normalize_symbol(symbol)}
        return self._request("GET", "/v5/market/tickers", params=params)

    def get_orderbook(self, symbol: str, limit: int = 50) -> dict[str, Any]:
        params = {
            "category": self.category,
            "symbol": self._normalize_symbol(symbol),
            "limit": int(limit),
        }
        return self._request("GET", "/v5/market/orderbook", params=params)

    def place_order_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        *,
        reduce_only: bool = False,
        position_idx: int | None = None,
        order_link_id: str | None = None,
        close_on_trigger: bool | None = None,
    ) -> dict[str, Any]:
        symbol_s = self._normalize_symbol(symbol)
        side_u = str(side).upper().strip()
        if side_u in ("BUY", "LONG"):
            normalized_side = "Buy"
        elif side_u in ("SELL", "SHORT"):
            normalized_side = "Sell"
        else:
            normalized_side = side.capitalize()
        qty_value = str(qty)

        if self.dry_run:
            logger.info(
                "Bybit dry_run place_order_market %s %s qty=%s reduce_only=%s position_idx=%s",
                symbol_s,
                normalized_side,
                qty_value,
                bool(reduce_only),
                position_idx,
            )
            return {
                "retCode": 0,
                "retMsg": "dry_run_simulation",
                "result": {
                    "symbol": symbol_s,
                    "side": normalized_side,
                    "qty": qty_value,
                    "reduceOnly": bool(reduce_only),
                    "positionIdx": int(position_idx) if position_idx is not None else 0,
                    "orderLinkId": order_link_id or "",
                    "orderStatus": "Filled",
                    "avgPrice": "0",
                },
            }

        body: dict[str, Any] = {
            "category": self.category,
            "symbol": symbol_s,
            "side": normalized_side,
            "orderType": "Market",
            "qty": qty_value,
            "reduceOnly": bool(reduce_only),
        }
        if position_idx is not None:
            body["positionIdx"] = int(position_idx)
        if order_link_id:
            body["orderLinkId"] = str(order_link_id)
        if close_on_trigger is not None:
            body["closeOnTrigger"] = bool(close_on_trigger)
        return self._request("POST", "/v5/order/create", private=True, json_body=body)

    def place_order_limit(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        *,
        reduce_only: bool = False,
        position_idx: int | None = None,
        order_link_id: str | None = None,
        time_in_force: str = "GTC",
        close_on_trigger: bool | None = None,
    ) -> dict[str, Any]:
        symbol_s = self._normalize_symbol(symbol)
        side_u = str(side).upper().strip()
        if side_u in ("BUY", "LONG"):
            normalized_side = "Buy"
        elif side_u in ("SELL", "SHORT"):
            normalized_side = "Sell"
        else:
            normalized_side = side.capitalize()

        qty_value = str(qty)
        price_value = str(price)

        if self.dry_run:
            logger.info(
                "Bybit dry_run place_order_limit %s %s qty=%s price=%s reduce_only=%s position_idx=%s",
                symbol_s,
                normalized_side,
                qty_value,
                price_value,
                bool(reduce_only),
                position_idx,
            )
            return {
                "retCode": 0,
                "retMsg": "dry_run_simulation",
                "result": {
                    "symbol": symbol_s,
                    "side": normalized_side,
                    "qty": qty_value,
                    "price": price_value,
                    "reduceOnly": bool(reduce_only),
                    "positionIdx": int(position_idx) if position_idx is not None else 0,
                    "orderLinkId": order_link_id or "",
                    "orderStatus": "New",
                    "avgPrice": "0",
                },
            }

        body: dict[str, Any] = {
            "category": self.category,
            "symbol": symbol_s,
            "side": normalized_side,
            "orderType": "Limit",
            "qty": qty_value,
            "price": price_value,
            "timeInForce": str(time_in_force or "GTC"),
            "reduceOnly": bool(reduce_only),
        }
        if position_idx is not None:
            body["positionIdx"] = int(position_idx)
        if order_link_id:
            body["orderLinkId"] = str(order_link_id)
        if close_on_trigger is not None:
            body["closeOnTrigger"] = bool(close_on_trigger)
        return self._request("POST", "/v5/order/create", private=True, json_body=body)

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if self.dry_run:
            return []

        params: dict[str, Any] = {"category": self.category, "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = self._normalize_symbol(symbol)
        payload = self._request("GET", "/v5/position/list", params=params, private=True)
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        items = result.get("list", []) if isinstance(result, dict) else []
        normalized = [self._normalize_position(item) for item in items if isinstance(item, dict)]
        return [item for item in normalized if float(item.get("size", 0.0)) > 0.0]

    def get_account_info(self) -> dict[str, Any]:
        if self.dry_run:
            return {"retCode": 0, "retMsg": "dry_run_simulation", "result": {"unifiedMarginStatus": 0}}
        return self._request("GET", "/v5/account/info", params={}, private=True)

    def apply_demo_funds(self, *, usdt_amount: str = "100000") -> dict[str, Any]:
        if self.dry_run:
            return {
                "retCode": 0,
                "retMsg": "dry_run_simulation",
                "result": {"utaDemoApplyMoney": [{"coin": "USDT", "amountStr": str(usdt_amount)}]},
            }
        if not self.demo:
            return {"retCode": 10001, "retMsg": "demo_mode_required", "result": {}}

        body = {
            "utaDemoApplyMoney": [
                {
                    "coin": "USDT",
                    "amountStr": str(usdt_amount),
                }
            ]
        }
        response = self._request("POST", "/v5/account/demo-apply-money", private=True, json_body=body)
        if isinstance(response, dict) and response.get("retCode") == 0:
            logger.info("Bybit demo funds applied usdt=%s", usdt_amount)
        else:
            logger.warning(
                "Bybit demo funds request failed retCode=%s retMsg=%s",
                response.get("retCode") if isinstance(response, dict) else None,
                response.get("retMsg") if isinstance(response, dict) else None,
            )
        return response

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if self.dry_run:
            return []

        params: dict[str, Any] = {"category": self.category}
        if symbol:
            params["symbol"] = self._normalize_symbol(symbol)
        payload = self._request("GET", "/v5/order/realtime", params=params, private=True)
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        items = result.get("list", []) if isinstance(result, dict) else []

        orders: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            status = str(item.get("orderStatus", "")).upper()
            if status in ("FILLED", "CANCELLED", "REJECTED", "DEACTIVATED"):
                continue
            side_raw = str(item.get("side", "")).upper()
            if side_raw in ("BUY", "LONG"):
                side = "BUY"
            elif side_raw in ("SELL", "SHORT"):
                side = "SELL"
            else:
                side = side_raw

            orders.append(
                {
                    "symbol": item.get("symbol"),
                    "side": side,
                    "orderId": item.get("orderId"),
                    "orderLinkId": item.get("orderLinkId"),
                    "orderStatus": item.get("orderStatus"),
                    "qty": self._to_float(item.get("leavesQty", item.get("qty", 0))),
                    "reduceOnly": bool(item.get("reduceOnly", False)),
                    "positionIdx": int(self._to_float(item.get("positionIdx", 0))),
                    "createdTime": item.get("createdTime"),
                    "updatedTime": item.get("updatedTime"),
                }
            )
        return orders

    def set_trading_stop(
        self,
        symbol: str,
        *,
        stop_loss: float,
        take_profit: float | None = None,
        position_idx: int | None = None,
        qty: float | None = None,
    ) -> dict[str, Any]:
        if self.dry_run:
            return {"retCode": 0, "retMsg": "dry_run_simulation", "result": {"symbol": self._normalize_symbol(symbol)}}

        normalized_symbol = self._normalize_symbol(symbol)
        body: dict[str, Any] = {
            "category": self.category,
            "symbol": normalized_symbol,
            "stopLoss": str(stop_loss),
        }

        if qty is not None and qty > 0:
            body["tpslMode"] = "Partial"
            body["slSize"] = str(qty)
            if take_profit is not None:
                body["takeProfit"] = str(take_profit)
                body["tpSize"] = str(qty)
        else:
            body["tpslMode"] = "Full"
            if take_profit is not None:
                body["takeProfit"] = str(take_profit)

        if position_idx is not None:
            body["positionIdx"] = int(position_idx)
        return self._request("POST", "/v5/position/trading-stop", private=True, json_body=body)

    def cancel_order(
        self,
        *,
        symbol: str,
        order_id: str | None = None,
        order_link_id: str | None = None,
    ) -> dict[str, Any]:
        if self.dry_run:
            return {"retCode": 0, "retMsg": "dry_run_simulation", "result": {"symbol": self._normalize_symbol(symbol)}}

        body: dict[str, Any] = {
            "category": self.category,
            "symbol": self._normalize_symbol(symbol),
        }
        if order_id:
            body["orderId"] = str(order_id)
        if order_link_id:
            body["orderLinkId"] = str(order_link_id)
        if not body.get("orderId") and not body.get("orderLinkId"):
            return {"retCode": 10001, "retMsg": "missing_order_id", "result": {}}
        return self._request("POST", "/v5/order/cancel", private=True, json_body=body)

    def _normalize_position(self, item: dict[str, Any]) -> dict[str, Any]:
        size = self._to_float(item.get("size", 0))
        entry_price = self._to_float(item.get("avgPrice", item.get("entryPrice", 0)))
        pnl = self._to_float(item.get("unrealisedPnl", item.get("unrealizedPnl", 0)))
        liq_price = self._to_float(item.get("liqPrice", 0))
        leverage = self._to_float(item.get("leverage", 0))
        mark_price = self._to_float(item.get("markPrice", 0))
        position_idx = int(self._to_float(item.get("positionIdx", 0)))

        side = str(item.get("side", "")).upper()
        if side in ("BUY", "LONG"):
            norm_side = "LONG"
        elif side in ("SELL", "SHORT"):
            norm_side = "SHORT"
        else:
            norm_side = side

        normalized = dict(item)
        normalized.update(
            {
                "symbol": item.get("symbol"),
                "side": norm_side,
                "size": size,
                "entryPrice": entry_price,
                "entry_price": entry_price,
                "unrealisedPnl": pnl,
                "unrealised_pnl": pnl,
                "liqPrice": liq_price,
                "liq_price": liq_price,
                "leverage": leverage,
                "markPrice": mark_price,
                "mark_price": mark_price,
                "positionIdx": position_idx,
            }
        )
        return normalized

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
