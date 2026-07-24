from __future__ import annotations

import time
from dataclasses import dataclass

import requests


@dataclass
class SentimentFeedConfig:
    url: str = "https://api.alternative.me/fng/"
    refresh_sec: int = 60
    timeout_sec: int = 5
    max_age_sec: int = 300


class SentimentFeed:
    """Synchronous Fear & Greed Index fetcher with caching and graceful fallback.

    Never raises: any failure degrades to a stale cached value (if still within
    max_age_sec) or the neutral 50.0 fallback, matching the historical default
    behavior of the live loop before this feed existed.
    """

    def __init__(self, config: SentimentFeedConfig | None = None):
        self.config = config or SentimentFeedConfig()
        self._session = requests.Session()
        self._cache_value: float | None = None
        self._cache_ts: float = 0.0

    def get_sentiment(self, now: float | None = None) -> tuple[float, str]:
        now = now if now is not None else time.time()

        if self._cache_value is not None and (now - self._cache_ts) < self.config.refresh_sec:
            return self._cache_value, "cache_fresh"

        try:
            response = self._session.get(self.config.url, timeout=self.config.timeout_sec)
            response.raise_for_status()
            payload = response.json()
            items = payload.get("data", []) if isinstance(payload, dict) else []
            value = float(items[0]["value"])
            self._cache_value, self._cache_ts = value, now
            return value, "live_alternative_me"
        except Exception:
            if self._cache_value is not None and (now - self._cache_ts) <= self.config.max_age_sec:
                return self._cache_value, "fallback_stale_cache"
            return 50.0, "fallback_neutral_50"

    def close(self):
        self._session.close()
