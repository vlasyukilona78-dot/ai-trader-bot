from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeframeCacheConfig:
    interval: str = "Hour4"
    candles: int = 120
    ttl_sec: float = 1800.0
    max_symbols: int = 500


class HigherTimeframeCache:
    """Per-symbol higher-timeframe bars, cached because they move slowly.

    Different indicators carry their signal on different timeframes - 4h RSI
    separated outcomes where the 1h reading did not - but refetching a 4h series
    every scan would multiply request volume for data that only changes once per
    bar. A long TTL keeps it cheap; a failed fetch returns the stale frame rather
    than nothing, so a hiccup cannot silently disable the gate that depends on it.
    """

    def __init__(self, feed, config: TimeframeCacheConfig | None = None):
        self.feed = feed
        self.config = config or TimeframeCacheConfig()
        self._frames: dict[str, pd.DataFrame] = {}
        self._fetched_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, symbol: str, *, now: float | None = None) -> pd.DataFrame | None:
        now = now if now is not None else time.time()
        with self._lock:
            cached = self._frames.get(symbol)
            age = now - self._fetched_at.get(symbol, 0.0)
            if cached is not None and age < self.config.ttl_sec:
                return cached

        try:
            frame = self.feed.fetch_frame(
                symbol=symbol, timeframe=self.config.interval, candles=self.config.candles
            ).ohlcv
        except Exception:
            with self._lock:
                return self._frames.get(symbol)

        with self._lock:
            if len(self._frames) >= self.config.max_symbols and symbol not in self._frames:
                oldest = min(self._fetched_at, key=self._fetched_at.get, default=None)
                if oldest is not None:
                    self._frames.pop(oldest, None)
                    self._fetched_at.pop(oldest, None)
            self._frames[symbol] = frame
            self._fetched_at[symbol] = now
            return frame

    @property
    def cached_symbols(self) -> int:
        return len(self._frames)
