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
        dry_run: bool = True,
        timeout: int = 12,
        recv_window: int = 5000,
        category: str = "linear",
    ):
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.sandbox = sandbox
        self.dry_run = dry_run
        self.timeout = timeout
        self.recv_window = str(recv_window)
        self.category = category
        self.base_url = "https://api-testnet.bybit.com" if sandbox else "https://api.bybit.com"

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
            "SANDBOX" if sandbox else "PRODUCTION",
            dry_run,
            (self.api_key[:4] + "...") if self.api_key else None,
        )

    def close(self):
        self._sess.close()

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").upper()

    def _sign(self, timestamp: str, payload: str) -> str:
        raw = f"{timestamp}{self.api_key}{self.recv_window}{payload}"
        return hmac.new(self.api_secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()

    def _build_auth(self, method: str, params: dict[str, Any] | None, json_body: dict[str, Any] | None):
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Private call requires API key/secret")

        timestamp = str(int(time.time() * 1000))
        if method.upper() == "GET":
            payload = urlencode(sorted((params or {}).items()))
        else:
            payload = json.dumps(json_body or {}, separators=(",", ":"), ensure_ascii=False)

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": self.recv_window,
            "X-BAPI-SIGN": self._sign(timestamp, payload),
        }
        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        private: bool = False,
        json_body: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        method = method.upper()

        for attempt in range(1, max_retries + 1):
            try:
                headers = None
                if private:
                    headers = self._build_auth(method, params, json_body)

                response = self._sess.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
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

    def place_order_market(self, symbol: str, side: str, qty: float) -> dict[str, Any]:
        symbol_s = self._normalize_symbol(symbol)
        normalized_side = side.capitalize()
        qty_value = str(qty)

        if self.dry_run:
            logger.info("Bybit dry_run place_order_market %s %s qty=%s", symbol_s, normalized_side, qty_value)
            return {
                "retCode": 0,
                "retMsg": "dry_run_simulation",
                "result": {
                    "symbol": symbol_s,
                    "side": normalized_side,
                    "qty": qty_value,
                    "orderStatus": "Filled",
                    "avgPrice": "0",
                },
            }

        body = {
            "category": self.category,
            "symbol": symbol_s,
            "side": normalized_side,
            "orderType": "Market",
            "qty": qty_value,
        }
        return self._request("POST", "/v5/order/create", private=True, json_body=body)

    def get_open_positions(self) -> list[dict[str, Any]]:
        if self.dry_run:
            return []

        params = {"category": self.category, "settleCoin": "USDT"}
        payload = self._request("GET", "/v5/position/list", params=params, private=True)
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        items = result.get("list", []) if isinstance(result, dict) else []
        return [self._normalize_position(item) for item in items if isinstance(item, dict)]

    def _normalize_position(self, item: dict[str, Any]) -> dict[str, Any]:
        size = self._to_float(item.get("size", 0))
        entry_price = self._to_float(item.get("avgPrice", item.get("entryPrice", 0)))
        pnl = self._to_float(item.get("unrealisedPnl", item.get("unrealizedPnl", 0)))

        side = str(item.get("side", "")).upper()
        normalized = dict(item)
        normalized.update(
            {
                "symbol": item.get("symbol"),
                "side": side,
                "size": size,
                "entryPrice": entry_price,
                "entry_price": entry_price,
                "unrealisedPnl": pnl,
                "unrealised_pnl": pnl,
            }
        )
        return normalized

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
