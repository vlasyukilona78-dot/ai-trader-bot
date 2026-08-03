from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.market_data import MarketDataClient
from trading.market_data.bar_contract import retain_closed_bars


@dataclass
class MarketFrame:
    symbol: str
    ohlcv: pd.DataFrame
    mark_price: float
    candle_cutoff_ts: float | None = None
    last_bar_open_ts: float | None = None
    last_bar_close_ts: float | None = None


class MarketDataFeed:
    def __init__(
        self,
        base_url: str = "https://api.bybit.com",
        timeout: int = 8,
        max_retries: int = 2,
        client=None,
    ):
        # An explicit client lets the same feed serve a different venue. The
        # MEXC client exposes the same fetch_ohlcv/fetch_ticker_meta surface and
        # returns identically shaped frames, so nothing downstream changes.
        self._client = client or MarketDataClient(
            base_url=base_url, timeout=timeout, max_retries=max_retries
        )

    def close(self):
        self._client.close()

    def _fetch_mark_price(self, symbol: str) -> float:
        ticker = self._client.fetch_ticker_meta(symbol=symbol)
        mark_price = 0.0
        for key in ("markPrice", "lastPrice", "indexPrice"):
            try:
                mark_price = float(ticker.get(key))
                if mark_price > 0:
                    break
            except (TypeError, ValueError):
                continue
        return mark_price

    def fetch_frame(self, symbol: str, timeframe: str, candles: int) -> MarketFrame:
        ohlcv = self._client.fetch_ohlcv(symbol=symbol, interval=timeframe, limit=int(candles))
        mark_price = self._fetch_mark_price(symbol)
        return MarketFrame(symbol=symbol, ohlcv=ohlcv, mark_price=mark_price)

    def fetch_closed_frame(
        self,
        symbol: str,
        timeframe: str,
        candles: int,
        *,
        as_of,
    ) -> MarketFrame:
        """Fetch a frame containing only bars closed by the explicit decision time."""

        requested = int(candles)
        if requested <= 0:
            raise ValueError("candles must be positive")
        raw = self._client.fetch_ohlcv(
            symbol=symbol,
            interval=timeframe,
            limit=requested + 1,
        )
        closed = retain_closed_bars(raw, interval=timeframe, as_of=as_of)
        metadata = dict(closed.attrs)
        closed = closed.tail(requested).copy()
        closed.attrs.update(metadata)
        return MarketFrame(
            symbol=symbol,
            ohlcv=closed,
            mark_price=self._fetch_mark_price(symbol),
            candle_cutoff_ts=closed.attrs["candle_cutoff_ts"],
            last_bar_open_ts=closed.attrs["last_bar_open_ts"],
            last_bar_close_ts=closed.attrs["last_bar_close_ts"],
        )
