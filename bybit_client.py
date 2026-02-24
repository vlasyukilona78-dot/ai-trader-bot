import time
import hmac
import hashlib
import requests
import json
from urllib.parse import urlencode

from logger import logger


class BybitClient:
    """
    Универсальный клиент Bybit (v5 API)
    Поддерживает mainnet и sandbox.
    """

    def __init__(self, api_key: str, api_secret: str,
                 sandbox: bool = False, dry_run: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.sandbox = sandbox
        self.dry_run = dry_run

        self.host = (
            "https://api-testnet.bybit.com"
            if sandbox
            else "https://api.bybit.com"
        )

        env_name = "TESTNET" if sandbox else "PRODUCTION"
        logger.info(f"BybitClient init -> {env_name} | dry_run={self.dry_run} | key={self.api_key[:4]}...{self.api_key[-4:]}")

        # Проверка авторизации при инициализации
        try:
            res = self._get("/v5/account/wallet-balance",
                            {"accountType": "UNIFIED"})
            if res.get("retCode") == 0:
                logger.info("✅ Authorized on Bybit mainnet")
            else:
                logger.warning(f"Bybit auth check failed: {res}")
        except Exception as e:
            logger.warning(f"Auth check exception: {e}")

    # ------------------------------------------------------------------
    # INTERNAL SIGNING / REQUEST HELPERS
    # ------------------------------------------------------------------

    def _sign(self, params: dict) -> str:
        """Создаёт подпись по правилам Bybit v5 (алфавитная сортировка параметров)."""
        sorted_params = sorted(params.items())
        query = "&".join(f"{k}={v}" for k, v in sorted_params)
        signature = hmac.new(self.api_secret, query.encode(),
                             hashlib.sha256).hexdigest()
        return signature

    def _headers(self):
        return {"Content-Type": "application/json"}

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        if params is None:
            params = {}
        params["api_key"] = self.api_key
        params["timestamp"] = int(time.time() * 1000)

        params["sign"] = self._sign(params)
        url = f"{self.host}{endpoint}?{urlencode(params)}"
        try:
            r = requests.get(url, headers=self._headers(), timeout=15)
            return r.json()
        except Exception as e:
            logger.error(f"GET {endpoint} failed: {e}")
            return {"error": str(e)}

    def _post(self, endpoint: str, body: dict) -> dict:
        body = dict(body)
        body["api_key"] = self.api_key
        body["timestamp"] = int(time.time() * 1000)
        body["sign"] = self._sign(body)
        try:
            url = f"{self.host}{endpoint}"
            r = requests.post(url, headers=self._headers(),
                              data=json.dumps(body), timeout=15)
            return r.json()
        except Exception as e:
            logger.error(f"POST {endpoint} failed: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------

    def get_wallet_balance(self):
        """Возвращает баланс аккаунта (UNIFIED)."""
        return self._get("/v5/account/wallet-balance", {"accountType": "UNIFIED"})

    def get_open_positions(self):
        """Возвращает открытые позиции."""
        try:
            res = self._get("/v5/position/list",
                            {"category": "linear", "settleCoin": "USDT"})
            if res.get("retCode") == 0:
                return res["result"]["list"]
            else:
                logger.warning(f"get_open_positions error: {res}")
                return []
        except Exception as e:
            logger.error(f"get_open_positions failed: {e}")
            return []

    def place_order_market(self, symbol: str, side: str, qty: float):
        """Создаёт рыночный ордер (или dry-run лог)."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would place {side.upper()} order on {symbol} qty={qty}")
            return {"dry_run": True, "symbol": symbol, "side": side, "qty": qty}

        body = {
            "category": "linear",
            "symbol": symbol.replace("/", ""),
            "side": side.upper(),
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GoodTillCancel"
        }
        res = self._post("/v5/order/create", body)
        if res.get("retCode") == 0:
            logger.info(f"✅ Market order placed {symbol} {side} qty={qty}")
        else:
            logger.warning(f"Order placement failed: {res}")
        return res

    def cancel_all_orders(self, symbol: str):
        """Отменяет все открытые ордера по символу."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would cancel orders for {symbol}")
            return {"dry_run": True}
        body = {"category": "linear", "symbol": symbol.replace("/", "")}
        return self._post("/v5/order/cancel-all", body)

    def get_server_time(self):
        """Проверка синхронизации времени."""
        return self._get("/v5/market/time")