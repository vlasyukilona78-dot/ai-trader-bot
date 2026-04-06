from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.market_data import MarketDataClient


@dataclass
class MarketFrame:
    symbol: str
    ohlcv: pd.DataFrame
    mark_price: float
    liquidation_cluster_high: float | None = None
    liquidation_cluster_low: float | None = None


class MarketDataFeed:
    def __init__(self, base_url: str = "https://api.bybit.com", timeout: int = 8, max_retries: int = 2):
        self._client = MarketDataClient(base_url=base_url, timeout=timeout, max_retries=max_retries)

    def close(self):
        self._client.close()

    def fetch_frame(
        self,
        symbol: str,
        timeframe: str,
        candles: int,
        *,
        include_liquidations: bool = False,
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
