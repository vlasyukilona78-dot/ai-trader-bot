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


def _empty_ohlcv_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=_EMPTY_OHLCV)
    frame.index = pd.DatetimeIndex([], tz="UTC", name="datetime")
    return frame


class _RateLimiter:
    """Token bucket shared by every thread using one client.

    Without it, scanning the universe concurrently silently loses symbols: at 8
    workers MEXC dropped 13 of 60 requests and the client returned empty frames
    that look exactly like "no data". Pacing requests keeps the whole universe
    visible instead of trading a quieter subset by accident.
    """

    def __init__(self, rate_per_sec: float):
        self.rate = max(0.1, rate_per_sec)
        self._allowance = self.rate
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._allowance = min(self.rate, self._allowance + (now - self._last) * self.rate)
                self._last = now
                if self._allowance >= 1.0:
                    self._allowance -= 1.0
                    return
                wait = (1.0 - self._allowance) / self.rate
            time.sleep(min(wait, 0.25))


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
        requests_per_sec: float = 8.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._limiter = _RateLimiter(requests_per_sec)
        self.tickers_cache_ttl_sec = tickers_cache_ttl_sec
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "crypto-ai-bot/2.0", "Accept": "application/json"})
        self._tickers_cache: list[dict[str, Any]] = []
        self._tickers_cache_ts: float = 0.0
        self._tickers_cache_initialized = False
        self._details_cache: dict[str, dict[str, Any]] = {}
        self._details_cache_ts: float = 0.0
        self._details_cache_initialized = False
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
                self._limiter.acquire()
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

    def fetch_all_tickers_with_provenance(
        self, force: bool = False
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Tickers plus how they were obtained.

        A cache hit and a fresh response are not interchangeable. Returning only
        the rows let a caller stamp cached data with the current instant and treat
        it as though the exchange had just answered, which is exactly the kind of
        provenance inflation the timing contract exists to prevent.

        ``source_ts`` is when the returned rows were actually received from the
        exchange, whether that was now or several minutes ago.
        """

        started = time.time()
        with self._lock:
            cached = list(self._tickers_cache)
            cached_ts = self._tickers_cache_ts
            cache_initialized = self._tickers_cache_initialized
        fresh = (started - cached_ts) < self.tickers_cache_ttl_sec
        if cache_initialized and fresh and not force:
            received = time.time()
            return cached, {
                "request_started_at": started,
                "received_at": received,
                "source_ts": cached_ts,
                "cache_hit": True,
                "cache_age_sec": max(0.0, started - cached_ts),
                "status": "ok",
                "error_code": None,
            }

        payload = self._request_public("/api/v1/contract/ticker")
        received = time.time()
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            # The request failed. The previous rows may still be usable, but they
            # must keep their own age rather than inherit this attempt's clock.
            return cached, {
                "request_started_at": started,
                "received_at": received,
                "source_ts": cached_ts if cache_initialized else None,
                "cache_hit": cache_initialized,
                "cache_age_sec": (
                    max(0.0, started - cached_ts) if cache_initialized else None
                ),
                "status": "stale_cache" if cache_initialized else "error",
                "error_code": "MexcTickerUnavailable",
            }

        with self._lock:
            self._tickers_cache = items
            self._tickers_cache_ts = received
            self._tickers_cache_initialized = True
        return list(items), {
            "request_started_at": started,
            "received_at": received,
            "source_ts": received,
            "cache_hit": False,
            "cache_age_sec": 0.0,
            "status": "ok",
            "error_code": None,
        }

    def fetch_all_tickers(self, force: bool = False) -> list[dict[str, Any]]:
        """All contract tickers in one call (~1000 symbols), short-TTL cached.

        One call serves both universe filtering and per-symbol ticker lookups,
        which avoids issuing a separate request per scanned symbol.
        """
        items, _ = self.fetch_all_tickers_with_provenance(force=force)
        return items

    def fetch_contract_details_with_provenance(
        self, force: bool = False
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Contract specs plus truthful fresh/cache/error provenance.

        One call covers the whole board, so the minimum tradeable size can be
        checked without a per-symbol request.  An empty mapping is ambiguous on
        its own: it can be a valid empty response, a cached empty response, or a
        failed first request.  The companion provenance keeps those cases
        distinct.
        """
        started = time.time()
        with self._lock:
            cached = dict(self._details_cache)
            cached_ts = self._details_cache_ts
            cache_initialized = self._details_cache_initialized
        if cache_initialized and not force:
            return cached, {
                "request_started_at": started,
                "received_at": time.time(),
                "source_ts": cached_ts,
                "cache_hit": True,
                "cache_age_sec": max(0.0, started - cached_ts),
                "status": "ok",
                "error_code": None,
            }

        payload = self._request_public("/api/v1/contract/detail")
        received = time.time()
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return cached, {
                "request_started_at": started,
                "received_at": received,
                "source_ts": cached_ts if cache_initialized else None,
                "cache_hit": cache_initialized,
                "cache_age_sec": (
                    max(0.0, started - cached_ts) if cache_initialized else None
                ),
                "status": "stale_cache" if cache_initialized else "error",
                "error_code": "MexcContractDetailsUnavailable",
            }

        out = {str(i.get("symbol")): i for i in items if isinstance(i, dict) and i.get("symbol")}
        with self._lock:
            self._details_cache = out
            self._details_cache_ts = received
            self._details_cache_initialized = True
        return dict(out), {
            "request_started_at": started,
            "received_at": received,
            "source_ts": received,
            "cache_hit": False,
            "cache_age_sec": 0.0,
            "status": "ok",
            "error_code": None,
        }

    def fetch_contract_details(self, force: bool = False) -> dict[str, dict[str, Any]]:
        """Backward-compatible rows-only contract-details surface."""

        details, _ = self.fetch_contract_details_with_provenance(force=force)
        return details

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
            return _empty_ohlcv_frame()

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
            return _empty_ohlcv_frame()

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
