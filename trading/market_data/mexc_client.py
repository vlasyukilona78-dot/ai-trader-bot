from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd
import requests

# Bybit-style interval codes -> MEXC contract interval names.
_INTERVAL_MAP = {
    "1": "Min1",
    "5": "Min5",
    "15": "Min15",
    "30": "Min30",
    "60": "Min60",
    "240": "Hour4",
    "480": "Hour8",
    "D": "Day1",
    "W": "Week1",
    "M": "Month1",
}

_QUOTES = ("USDT", "USDC", "USD")

_EMPTY_OHLCV = ["time", "open", "high", "low", "close", "volume"]

# MEXC klines carry BOTH a contract count (`vol`) and the exact quote turnover
# (`amount`). Multiplying price by `vol` and calling it USD is wrong by the
# contract size, which differs per symbol - on BTC (contractSize 0.0001) it
# overstates turnover ~10,000x, on CHILLGUY (contractSize 10) it understates it
# 10x. Anything comparing liquidity across symbols must use `amount`.
_TURNOVER_COLUMN = "turnover"


class MexcContractClient:
    """Public market data client for MEXC USDT-perpetual contracts.

    Mirrors the public surface of core.market_data.MarketDataClient (fetch_ohlcv,
    fetch_ticker_meta, fetch_funding_rate, close) and returns identically shaped
    data, so it is a drop-in replacement for the Bybit client in MarketDataFeed.
    MEXC-specific field names are translated to the Bybit-style names downstream
    code already expects (e.g. fairPrice -> markPrice).

    Only public endpoints are used - no API key is required.
    """

    def __init__(
        self,
        base_url: str = "https://contract.mexc.com",
        timeout: int = 12,
        max_retries: int = 3,
        tickers_cache_ttl_sec: int = 20,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.tickers_cache_ttl_sec = tickers_cache_ttl_sec
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "crypto-ai-bot/2.0", "Accept": "application/json"})
        self._tickers_cache: list[dict[str, Any]] = []
        self._tickers_cache_ts: float = 0.0
        self._details_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Accepts BTCUSDT / BTC/USDT / BTC_USDT and returns MEXC form BTC_USDT."""
        raw = str(symbol).strip().upper().replace("/", "_").replace("-", "_")
        if "_" in raw:
            return raw
        for quote in _QUOTES:
            if raw.endswith(quote) and len(raw) > len(quote):
                return f"{raw[: -len(quote)]}_{quote}"
        return raw

    @staticmethod
    def denormalize_symbol(symbol: str) -> str:
        """MEXC form BTC_USDT -> compact BTCUSDT used by the rest of the bot."""
        return str(symbol).strip().upper().replace("_", "")

    def _request_public(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        url = f"{self.base_url}{path}"
        delay = 0.5
        for attempt in range(max(1, self.max_retries)):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and payload.get("success") is True:
                    return payload
                return None
            except Exception:
                if attempt >= self.max_retries - 1:
                    return None
                time.sleep(delay)
                delay *= 2
        return None

    def fetch_all_tickers(self, force: bool = False) -> list[dict[str, Any]]:
        """All contract tickers in one call (~1000 symbols), short-TTL cached.

        One call serves both universe filtering and per-symbol ticker lookups,
        which avoids issuing a separate request per scanned symbol.
        """
        now = time.time()
        with self._lock:
            fresh = (now - self._tickers_cache_ts) < self.tickers_cache_ttl_sec
            if self._tickers_cache and fresh and not force:
                return list(self._tickers_cache)

        payload = self._request_public("/api/v1/contract/ticker")
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            with self._lock:
                return list(self._tickers_cache)

        with self._lock:
            self._tickers_cache = items
            self._tickers_cache_ts = now
            return list(items)

    def fetch_contract_details(self, force: bool = False) -> dict[str, dict[str, Any]]:
        """Contract specs for every symbol, keyed by MEXC symbol, cached for the session.

        One call covers the whole board, so the minimum tradeable size can be
        checked without a per-symbol request.
        """
        with self._lock:
            if self._details_cache and not force:
                return dict(self._details_cache)

        payload = self._request_public("/api/v1/contract/detail")
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            with self._lock:
                return dict(self._details_cache)

        out = {str(i.get("symbol")): i for i in items if isinstance(i, dict) and i.get("symbol")}
        with self._lock:
            self._details_cache = out
            return dict(out)

    @staticmethod
    def _translate_ticker(item: dict[str, Any]) -> dict[str, Any]:
        """Map MEXC ticker fields onto the Bybit-style names used downstream."""
        out = dict(item)
        out["markPrice"] = item.get("fairPrice")
        out["lastPrice"] = item.get("lastPrice")
        out["indexPrice"] = item.get("indexPrice")
        out["fundingRate"] = item.get("fundingRate")
        out["turnover24h"] = item.get("amount24")
        out["volume24h"] = item.get("volume24")
        out["openInterest"] = item.get("holdVol")
        out["price24hPcnt"] = item.get("riseFallRate")
        return out

    def fetch_ticker_meta(self, symbol: str) -> dict[str, Any]:
        mexc_symbol = self.normalize_symbol(symbol)

        for item in self.fetch_all_tickers():
            if isinstance(item, dict) and item.get("symbol") == mexc_symbol:
                return self._translate_ticker(item)

        payload = self._request_public("/api/v1/contract/ticker", params={"symbol": mexc_symbol})
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict):
            return {}
        return self._translate_ticker(data)

    def fetch_funding_rate(self, symbol: str) -> float | None:
        try:
            return float(self.fetch_ticker_meta(symbol).get("fundingRate"))
        except (TypeError, ValueError):
            return None

    def fetch_open_interest(self, symbol: str) -> float | None:
        try:
            return float(self.fetch_ticker_meta(symbol).get("openInterest"))
        except (TypeError, ValueError):
            return None

    def fetch_ohlcv(self, symbol: str, interval: str = "1", limit: int = 300) -> pd.DataFrame:
        mexc_symbol = self.normalize_symbol(symbol)
        mexc_interval = _INTERVAL_MAP.get(str(interval), str(interval))
        limit = max(1, int(limit))

        params: dict[str, Any] = {"interval": mexc_interval}
        seconds_per_bar = self._interval_seconds(mexc_interval)
        if seconds_per_bar:
            # Ask only for the window we need instead of the 2000-bar maximum.
            params["start"] = int(time.time()) - seconds_per_bar * (limit + 2)

        payload = self._request_public(f"/api/v1/contract/kline/{mexc_symbol}", params=params)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict) or not data.get("time"):
            return pd.DataFrame(columns=_EMPTY_OHLCV)

        try:
            df = pd.DataFrame(
                {
                    # MEXC returns epoch seconds; convert to ms to match the Bybit client.
                    "time": pd.to_numeric(pd.Series(data["time"]), errors="coerce") * 1000,
                    "open": pd.to_numeric(pd.Series(data["open"]), errors="coerce"),
                    "high": pd.to_numeric(pd.Series(data["high"]), errors="coerce"),
                    "low": pd.to_numeric(pd.Series(data["low"]), errors="coerce"),
                    "close": pd.to_numeric(pd.Series(data["close"]), errors="coerce"),
                    "volume": pd.to_numeric(pd.Series(data["vol"]), errors="coerce"),
                    # exact quote turnover; see _TURNOVER_COLUMN note above
                    _TURNOVER_COLUMN: pd.to_numeric(pd.Series(data.get("amount", [])), errors="coerce"),
                }
            )
        except (KeyError, ValueError):
            return pd.DataFrame(columns=_EMPTY_OHLCV)

        df = df.dropna(subset=_EMPTY_OHLCV)
        df = df.sort_values("time").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"]).set_index("datetime")
        return df.tail(limit)

    @staticmethod
    def _interval_seconds(mexc_interval: str) -> int | None:
        table = {
            "Min1": 60,
            "Min5": 300,
            "Min15": 900,
            "Min30": 1800,
            "Min60": 3600,
            "Hour4": 14400,
            "Hour8": 28800,
            "Day1": 86400,
        }
        return table.get(mexc_interval)
