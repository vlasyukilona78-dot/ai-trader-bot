from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.market_data import MarketDataClient


@dataclass
class MarketFrame:
    symbol: str
    ohlcv: pd.DataFrame
    mark_price: float


class MarketDataFeed:
    def __init__(self, base_url: str = "https://api.bybit.com", timeout: int = 8, max_retries: int = 2):
        self._client = MarketDataClient(base_url=base_url, timeout=timeout, max_retries=max_retries)

    def close(self):
        self._client.close()

    def fetch_frame(self, symbol: str, timeframe: str, candles: int) -> MarketFrame:
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
        return MarketFrame(symbol=symbol, ohlcv=ohlcv, mark_price=mark_price)
