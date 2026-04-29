from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

import pandas as pd

from core.market_data import MarketDataClient
from trading.exchange.bybit_endpoints import resolve_public_http_base_url


@dataclass
class MarketFrame:
    symbol: str
    ohlcv: pd.DataFrame
    mark_price: float
    liquidation_cluster_high: float | None = None
    liquidation_cluster_low: float | None = None
    runtime_payload: dict[str, object] | None = None


class MarketDataFeed:
    def __init__(self, base_url: str | None = None, timeout: int = 8, max_retries: int = 2):
        self._client = MarketDataClient(
            base_url=base_url or resolve_public_http_base_url(testnet=False),
            timeout=timeout,
            max_retries=max_retries,
        )

    def close(self):
        self._client.close()

    @staticmethod
    def _env_truthy(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        value = str(raw).strip().lower()
        if not value:
            return default
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    @classmethod
    def _runtime_payload_from_ticker(cls, ticker: dict[str, Any]) -> dict[str, object]:
        payload: dict[str, object] = {}
        if not isinstance(ticker, dict):
            return payload

        funding_raw = ticker.get("fundingRate")
        if funding_raw in (None, ""):
            funding_raw = ticker.get("funding_rate")
        funding_rate = cls._safe_float(funding_raw)
        if funding_rate is not None:
            payload["funding_rate"] = funding_rate
            payload["funding_source"] = "live:bybit:ticker"
            payload["funding_degraded"] = False

        turnover_raw = ticker.get("turnover24h")
        if turnover_raw in (None, ""):
            turnover_raw = ticker.get("turnover24h_usdt")
        turnover_24h = cls._safe_float(turnover_raw)
        volume_24h = cls._safe_float(ticker.get("volume24h"))
        bid_px = cls._safe_float(ticker.get("bid1Price"))
        ask_px = cls._safe_float(ticker.get("ask1Price"))
        if turnover_24h is not None:
            payload["turnover24h_usdt"] = turnover_24h
        if volume_24h is not None:
            payload["volume24h"] = volume_24h
        if bid_px is not None:
            payload["bid1Price"] = bid_px
        if ask_px is not None:
            payload["ask1Price"] = ask_px
        if bid_px is not None and ask_px is not None and bid_px > 0 and ask_px >= bid_px:
            mid = (bid_px + ask_px) / 2.0
            if mid > 0:
                payload["spread_bps"] = ((ask_px - bid_px) / mid) * 10_000.0
        return payload

    def _fetch_optional_derivative_payload(self, symbol: str) -> dict[str, object]:
        if not self._env_truthy("MARKETDATA_FETCH_DERIVATIVE_CONTEXT", False):
            return {}

        payload: dict[str, object] = {}
        long_short_ratio = self._client.fetch_long_short_ratio(symbol)
        if long_short_ratio is not None:
            payload["long_short_ratio"] = long_short_ratio
            payload["long_short_ratio_source"] = "live:bybit:account-ratio"
            payload["long_short_ratio_degraded"] = False

        open_interest = self._client.fetch_open_interest(symbol)
        if open_interest is not None:
            payload["open_interest_abs"] = open_interest
            payload["open_interest_abs_source"] = "live:bybit:open-interest"
        return payload

    @classmethod
    def _overlay_live_price_to_ohlcv(cls, ohlcv: pd.DataFrame, *, mark_price: float, timeframe: str) -> pd.DataFrame:
        if ohlcv.empty or mark_price <= 0:
            return ohlcv

        try:
            interval_minutes = max(1, int(str(timeframe).strip()))
        except (TypeError, ValueError):
            interval_minutes = 1

        updated = ohlcv.copy()
        last_ts = pd.Timestamp(updated.index[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")

        now_utc = pd.Timestamp.now("UTC")
        current_bucket = now_utc.floor(f"{interval_minutes}min")
        if current_bucket < last_ts:
            return updated

        last_row = updated.iloc[-1].copy()
        last_close = float(last_row.get("close", mark_price))

        if current_bucket == last_ts:
            updated.iloc[-1, updated.columns.get_loc("high")] = max(float(last_row.get("high", mark_price)), mark_price)
            updated.iloc[-1, updated.columns.get_loc("low")] = min(float(last_row.get("low", mark_price)), mark_price)
            updated.iloc[-1, updated.columns.get_loc("close")] = mark_price
            return updated

        if not cls._env_truthy("MARKETDATA_LIVE_OVERLAY_APPEND_NEW_BUCKET", False):
            max_lag = pd.Timedelta(minutes=interval_minutes * 2)
            if current_bucket - last_ts <= max_lag:
                updated.iloc[-1, updated.columns.get_loc("high")] = max(float(last_row.get("high", mark_price)), mark_price)
                updated.iloc[-1, updated.columns.get_loc("low")] = min(float(last_row.get("low", mark_price)), mark_price)
                updated.iloc[-1, updated.columns.get_loc("close")] = mark_price
                return updated

        new_row = last_row.copy()
        new_row["open"] = last_close
        new_row["high"] = max(last_close, mark_price)
        new_row["low"] = min(last_close, mark_price)
        new_row["close"] = mark_price
        new_row["volume"] = 0.0
        updated.loc[current_bucket] = new_row
        return updated.sort_index()

    def fetch_frame(
        self,
        symbol: str,
        timeframe: str,
        candles: int,
        *,
        include_liquidations: bool = False,
        overlay_live_price: bool = False,
    ) -> MarketFrame:
        ohlcv = self._client.fetch_ohlcv(symbol=symbol, interval=timeframe, limit=int(candles))
        ticker = self._client.fetch_ticker_meta(symbol=symbol)
        runtime_payload = self._runtime_payload_from_ticker(ticker)
        runtime_payload.update(self._fetch_optional_derivative_payload(symbol))
        mark_price = 0.0
        for key in ("markPrice", "lastPrice", "indexPrice"):
            try:
                mark_price = float(ticker.get(key))
                if mark_price > 0:
                    break
            except (TypeError, ValueError):
                continue
        if overlay_live_price:
            ohlcv = self._overlay_live_price_to_ohlcv(ohlcv, mark_price=mark_price, timeframe=timeframe)
        liq_high = None
        liq_low = None
        if include_liquidations:
            current_price = mark_price
            if current_price <= 0:
                try:
                    current_price = float(ohlcv.iloc[-1]["close"])
                except Exception:
                    current_price = 0.0
            heatmap_bands = self._client.fetch_liquidation_heatmap_bands(symbol, current_price=current_price)
            if heatmap_bands:
                ohlcv.attrs["coinglass_liquidation_bands"] = heatmap_bands
                ohlcv.attrs["liquidation_feed_bands"] = heatmap_bands
                above = [row for row in heatmap_bands if str(row.get("side")) == "above"]
                below = [row for row in heatmap_bands if str(row.get("side")) == "below"]
                if above:
                    liq_high = float(max(above, key=lambda row: float(row.get("weight", 0.0))).get("level", 0.0))
                if below:
                    liq_low = float(max(below, key=lambda row: float(row.get("weight", 0.0))).get("level", 0.0))
            liq_feed = self._client.fetch_recent_liquidations(symbol)
            bybit_high, bybit_low = self._client.liquidation_clusters_from_feed(liq_feed)
            liq_high = liq_high if liq_high is not None else bybit_high
            liq_low = liq_low if liq_low is not None else bybit_low
            if liq_high is None and liq_low is None:
                liq_high, liq_low = self._client.estimate_liquidation_clusters(ohlcv)
        return MarketFrame(
            symbol=symbol,
            ohlcv=ohlcv,
            mark_price=mark_price,
            liquidation_cluster_high=liq_high,
            liquidation_cluster_low=liq_low,
            runtime_payload=runtime_payload,
        )
