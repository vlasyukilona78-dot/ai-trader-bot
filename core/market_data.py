from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


@dataclass
class MarketSnapshot:
    symbol: str
    ohlcv: pd.DataFrame
    orderbook: dict[str, Any]
    funding_rate: float | None
    open_interest: float | None
    long_short_ratio: float | None
    liquidation_cluster_high: float | None
    liquidation_cluster_low: float | None
    sentiment_index: float | None


class MarketDataClient:
    """Public market data client for Bybit + external sentiment."""

    def __init__(
        self,
        base_url: str = "https://api.bybit.com",
        sentiment_url: str = "https://api.alternative.me/fng/",
        timeout: int = 12,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.sentiment_url = sentiment_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "crypto-ai-bot/2.0", "Accept": "application/json"})
        self._symbol_categories: dict[str, str] = {}

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return symbol.replace("/", "").upper().strip()

    def _cache_symbol_category(self, symbol: str, category: str):
        key_raw = str(symbol).upper().strip()
        key_norm = self.normalize_symbol(symbol)
        self._symbol_categories[key_raw] = category
        self._symbol_categories[key_norm] = category

    def _category_for_symbol(self, symbol: str) -> str:
        key_raw = str(symbol).upper().strip()
        key_norm = self.normalize_symbol(symbol)

        from_cache = self._symbol_categories.get(key_raw) or self._symbol_categories.get(key_norm)
        if from_cache in ("linear", "inverse"):
            return from_cache

        # Fallback heuristic for manual symbol overrides.
        if key_norm.endswith("USD") and not key_norm.endswith("USDT") and not key_norm.endswith("USDC"):
            return "inverse"
        return "linear"

    def _request_public(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                if isinstance(payload, dict) and payload.get("retCode") not in (None, 0):
                    return None
                return payload if isinstance(payload, dict) else None
            except Exception:
                time.sleep(min(1.5**attempt, 5.0))
        return None

    @staticmethod
    def _format_symbol(item: dict[str, Any]) -> str | None:
        raw = item.get("symbol")
        if raw:
            return str(raw).upper().strip()

        base = item.get("baseCoin")
        quote = item.get("quoteCoin")
        if base and quote:
            return f"{str(base).upper()}/{str(quote).upper()}"
        return None

    def fetch_symbols(self, quote: str | None = None, categories: tuple[str, ...] = ("linear", "inverse")) -> list[str]:
        quote_filter = str(quote).upper().strip() if quote else None

        symbols: list[str] = []
        self._symbol_categories = {}

        for category in categories:
            cursor = ""
            while True:
                params: dict[str, Any] = {"category": category, "limit": 1000, "status": "Trading"}
                if cursor:
                    params["cursor"] = cursor

                payload = self._request_public("/v5/market/instruments-info", params=params)
                if not payload:
                    break

                result = payload.get("result", {}) if isinstance(payload, dict) else {}
                items = result.get("list", []) if isinstance(result, dict) else []
                if not isinstance(items, list):
                    break

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("status") != "Trading":
                        continue

                    quote_coin = str(item.get("quoteCoin", "")).upper()
                    if quote_filter and quote_coin != quote_filter:
                        continue

                    symbol = self._format_symbol(item)
                    if not symbol:
                        continue

                    symbols.append(symbol)
                    self._cache_symbol_category(symbol, category)

                cursor = str(result.get("nextPageCursor") or "")
                if not cursor:
                    break

        return sorted(set(symbols))

    def fetch_ohlcv(self, symbol: str, interval: str = "1", limit: int = 300) -> pd.DataFrame:
        category = self._category_for_symbol(symbol)
        params = {
            "category": category,
            "symbol": self.normalize_symbol(symbol),
            "interval": str(interval),
            "limit": int(limit),
        }
        payload = self._request_public("/v5/market/kline", params=params)
        if not payload:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        rows = payload.get("result", {}).get("list", [])
        if not rows:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "turnover"])
        for col in ("time", "open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["time", "open", "high", "low", "close", "volume"])
        df = df[["time", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("time").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"]).set_index("datetime")
        return df

    def fetch_orderbook(self, symbol: str, limit: int = 50) -> dict[str, Any]:
        category = self._category_for_symbol(symbol)
        payload = self._request_public(
            "/v5/market/orderbook",
            params={
                "category": category,
                "symbol": self.normalize_symbol(symbol),
                "limit": int(limit),
            },
        )
        if not payload:
            return {}
        return payload.get("result", {}) if isinstance(payload, dict) else {}

    def fetch_ticker_meta(self, symbol: str) -> dict[str, Any]:
        category = self._category_for_symbol(symbol)
        payload = self._request_public(
            "/v5/market/tickers",
            params={"category": category, "symbol": self.normalize_symbol(symbol)},
        )
        if not payload:
            return {}
        items = payload.get("result", {}).get("list", [])
        if not items:
            return {}
        return items[0] if isinstance(items[0], dict) else {}

    def fetch_funding_rate(self, symbol: str) -> float | None:
        ticker = self.fetch_ticker_meta(symbol)
        for key in ("fundingRate", "funding_rate"):
            try:
                return float(ticker.get(key))
            except (TypeError, ValueError):
                continue
        return None

    def fetch_open_interest(self, symbol: str) -> float | None:
        category = self._category_for_symbol(symbol)
        payload = self._request_public(
            "/v5/market/open-interest",
            params={
                "category": category,
                "symbol": self.normalize_symbol(symbol),
                "intervalTime": "5min",
                "limit": 1,
            },
        )
        if not payload:
            return None
        items = payload.get("result", {}).get("list", [])
        if not items:
            return None
        row = items[0]
        for key in ("openInterest", "open_interest"):
            try:
                return float(row.get(key))
            except (TypeError, ValueError):
                continue
        return None

    def fetch_long_short_ratio(self, symbol: str) -> float | None:
        category = self._category_for_symbol(symbol)
        payload = self._request_public(
            "/v5/market/account-ratio",
            params={
                "category": category,
                "symbol": self.normalize_symbol(symbol),
                "period": "5min",
                "limit": 1,
            },
        )
        if not payload:
            return None
        items = payload.get("result", {}).get("list", [])
        if not items:
            return None

        row = items[0]
        try:
            buy = float(row.get("buyRatio", 0.0))
            sell = float(row.get("sellRatio", 0.0))
            if sell <= 0:
                return None
            return buy / sell
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def fetch_recent_liquidations(self, symbol: str, limit: int = 50) -> list[dict[str, Any]]:
        """Attempts to fetch Bybit liquidation feed; returns [] when unavailable."""
        category = self._category_for_symbol(symbol)
        payload = self._request_public(
            "/v5/market/liquidation",
            params={
                "category": category,
                "symbol": self.normalize_symbol(symbol),
                "limit": int(limit),
            },
        )
        if not payload:
            return []

        result = payload.get("result", {})
        if isinstance(result, dict):
            rows = result.get("list", [])
            return rows if isinstance(rows, list) else []
        return []

    def fetch_sentiment_index(self, url: str | None = None) -> float | None:
        target = url or self.sentiment_url
        try:
            resp = self._session.get(target, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("data", []) if isinstance(payload, dict) else []
            if not items:
                return None
            return float(items[0].get("value"))
        except Exception:
            return None

    @staticmethod
    def estimate_liquidation_clusters(df: pd.DataFrame, window: int = 80) -> tuple[float | None, float | None]:
        if df.empty or len(df) < 20:
            return None, None

        sample = df.tail(window).copy()
        vol_ma = sample["volume"].rolling(20).mean()
        vol_spike = sample["volume"] / vol_ma.replace(0, pd.NA)

        up_wick = sample["high"] - sample[["open", "close"]].max(axis=1)
        low_wick = sample[["open", "close"]].min(axis=1) - sample["low"]

        high_candidates = sample.loc[(vol_spike > 2.5) & (up_wick > up_wick.rolling(20).mean())]
        low_candidates = sample.loc[(vol_spike > 2.5) & (low_wick > low_wick.rolling(20).mean())]

        high_level = float(high_candidates["high"].median()) if not high_candidates.empty else None
        low_level = float(low_candidates["low"].median()) if not low_candidates.empty else None
        return high_level, low_level

    @staticmethod
    def liquidation_clusters_from_feed(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
        if not rows:
            return None, None

        prices: list[float] = []
        sides: list[str] = []
        for row in rows:
            side = str(row.get("side", "")).upper()
            px = row.get("price") or row.get("liqPrice") or row.get("liq_price")
            try:
                price = float(px)
            except (TypeError, ValueError):
                continue
            prices.append(price)
            sides.append(side)

        if not prices:
            return None, None

        highs = [p for p, s in zip(prices, sides) if s in ("BUY", "LONG")]
        lows = [p for p, s in zip(prices, sides) if s in ("SELL", "SHORT")]

        liq_high = float(pd.Series(highs).median()) if highs else None
        liq_low = float(pd.Series(lows).median()) if lows else None
        return liq_high, liq_low

    def fetch_snapshot(
        self,
        symbol: str,
        interval: str = "1",
        limit: int = 300,
        *,
        include_orderbook: bool = True,
        include_funding_rate: bool = True,
        include_open_interest: bool = True,
        include_long_short_ratio: bool = True,
        include_liquidations: bool = True,
        include_sentiment: bool = True,
        sentiment_index: float | None = None,
    ) -> MarketSnapshot:
        ohlcv = self.fetch_ohlcv(symbol=symbol, interval=interval, limit=limit)

        liq_high: float | None = None
        liq_low: float | None = None
        if include_liquidations:
            liq_feed = self.fetch_recent_liquidations(symbol)
            liq_high, liq_low = self.liquidation_clusters_from_feed(liq_feed)
            if liq_high is None and liq_low is None:
                liq_high, liq_low = self.estimate_liquidation_clusters(ohlcv)

        sentiment_value = sentiment_index
        if sentiment_value is None and include_sentiment:
            sentiment_value = self.fetch_sentiment_index()

        return MarketSnapshot(
            symbol=symbol,
            ohlcv=ohlcv,
            orderbook=self.fetch_orderbook(symbol) if include_orderbook else {},
            funding_rate=self.fetch_funding_rate(symbol) if include_funding_rate else None,
            open_interest=self.fetch_open_interest(symbol) if include_open_interest else None,
            long_short_ratio=self.fetch_long_short_ratio(symbol) if include_long_short_ratio else None,
            liquidation_cluster_high=liq_high,
            liquidation_cluster_low=liq_low,
            sentiment_index=sentiment_value,
        )

