from __future__ import annotations

from dataclasses import dataclass

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
    def _overlay_live_price_to_ohlcv(ohlcv: pd.DataFrame, *, mark_price: float, timeframe: str) -> pd.DataFrame:
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
            liq_feed = self._client.fetch_recent_liquidations(symbol)
            liq_high, liq_low = self._client.liquidation_clusters_from_feed(liq_feed)
            if liq_high is None and liq_low is None:
                liq_high, liq_low = self._client.estimate_liquidation_clusters(ohlcv)
        return MarketFrame(
            symbol=symbol,
            ohlcv=ohlcv,
            mark_price=mark_price,
            liquidation_cluster_high=liq_high,
            liquidation_cluster_low=liq_low,
        )
