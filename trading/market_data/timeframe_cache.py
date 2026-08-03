from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pandas as pd

from trading.market_data.bar_contract import closed_boundary_ts, retain_closed_bars


@dataclass
class TimeframeCacheConfig:
    interval: str = "Hour4"
    candles: int = 120
    ttl_sec: float = 1800.0
    max_symbols: int = 500


class HigherTimeframeCache:
    """Per-symbol closed higher-timeframe bars keyed by causal boundary.

    TTL controls request pacing inside one bar. Crossing a candle boundary always
    requires a new fetch. If that fetch fails, the previous boundary is never
    served as though it belonged to the current decision time.
    """

    def __init__(self, feed, config: TimeframeCacheConfig | None = None):
        self.feed = feed
        self.config = config or TimeframeCacheConfig()
        self._frames: dict[str, pd.DataFrame] = {}
        self._fetched_at: dict[str, float] = {}
        self._boundary_at: dict[str, float] = {}
        self._lock = threading.Lock()
        # Per-call spans so a cycle can report higher-timeframe timing as its own
        # source instead of hiding it inside the base-OHLCV span.
        self._timings: list[dict[str, object]] = []

    def drain_timings(self) -> list[dict[str, object]]:
        """Take and clear the spans recorded since the last drain."""

        with self._lock:
            drained = list(self._timings)
            self._timings.clear()
        return drained

    def _record_timing(self, *, started: float, received: float, boundary: float,
                       status: str, cache_hit: bool) -> None:
        with self._lock:
            if len(self._timings) < 5_000:
                self._timings.append(
                    {
                        "request_started_at": started,
                        "received_at": received,
                        "source_as_of": boundary,
                        "status": status,
                        "cache_hit": cache_hit,
                    }
                )

    @staticmethod
    def _copy(frame: pd.DataFrame) -> pd.DataFrame:
        copied = frame.copy(deep=True)
        copied.attrs.update(frame.attrs)
        return copied

    def get(
        self,
        symbol: str,
        *,
        as_of=None,
        now: float | None = None,
    ) -> pd.DataFrame | None:
        started_at = time.time()
        cache_now = float(now if now is not None else time.time())
        # ``now`` as the decision clock is retained only for legacy callers;
        # causal callers should always provide their own explicit ``as_of``.
        decision_time = as_of if as_of is not None else cache_now
        boundary = closed_boundary_ts(decision_time, self.config.interval)
        with self._lock:
            cached = self._frames.get(symbol)
            cached_boundary = self._boundary_at.get(symbol)
            age = cache_now - self._fetched_at.get(symbol, 0.0)
            if (
                cached is not None
                and cached_boundary == boundary
                and 0.0 <= age < self.config.ttl_sec
            ):
                hit = self._copy(cached)
                self._timings.append(
                    {
                        "request_started_at": started_at,
                        "received_at": time.time(),
                        "source_as_of": boundary,
                        "status": "ok",
                        "cache_hit": True,
                    }
                )
                return hit

        try:
            fetch_closed = getattr(self.feed, "fetch_closed_frame", None)
            if callable(fetch_closed):
                fetched = fetch_closed(
                    symbol=symbol,
                    timeframe=self.config.interval,
                    candles=self.config.candles,
                    as_of=boundary,
                ).ohlcv
            else:
                fetched = self.feed.fetch_frame(
                    symbol=symbol,
                    timeframe=self.config.interval,
                    candles=self.config.candles + 1,
                ).ohlcv
            frame = retain_closed_bars(
                fetched,
                interval=self.config.interval,
                as_of=boundary,
            ).tail(self.config.candles)
            frame = self._copy(frame)
        except Exception:
            self._record_timing(
                started=started_at, received=time.time(), boundary=boundary,
                status="error", cache_hit=False,
            )
            with self._lock:
                cached = self._frames.get(symbol)
                if cached is not None and self._boundary_at.get(symbol) == boundary:
                    return self._copy(cached)
                return None

        self._record_timing(
            started=started_at, received=time.time(), boundary=boundary,
            status="ok", cache_hit=False,
        )

        with self._lock:
            existing_boundary = self._boundary_at.get(symbol)
            if existing_boundary is not None and existing_boundary > boundary:
                return self._copy(frame)
            if len(self._frames) >= self.config.max_symbols and symbol not in self._frames:
                oldest = min(self._fetched_at, key=self._fetched_at.get, default=None)
                if oldest is not None:
                    self._frames.pop(oldest, None)
                    self._fetched_at.pop(oldest, None)
                    self._boundary_at.pop(oldest, None)
            self._frames[symbol] = self._copy(frame)
            self._fetched_at[symbol] = cache_now
            self._boundary_at[symbol] = boundary
            return self._copy(frame)

    @property
    def cached_symbols(self) -> int:
        return len(self._frames)
