from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import aiohttp


@dataclass
class SentimentConfig:
    url: str
    refresh_sec: int = 60
    timeout_sec: int = 5
    max_age_sec: int = 300


class SentimentFeed:
    def __init__(self, config: SentimentConfig):
        self.config = config
        self._value: float | None = None
        self._ts: float = 0.0
        self._lock = asyncio.Lock()

    async def fetch_once(self, session: aiohttp.ClientSession) -> float | None:
        async with session.get(self.config.url, timeout=self.config.timeout_sec) as resp:
            if resp.status != 200:
                return None
            payload = await resp.json()
            items = payload.get("data", []) if isinstance(payload, dict) else []
            if not items:
                return None
            return float(items[0].get("value"))

    async def run(self, session: aiohttp.ClientSession, stop_event: asyncio.Event):
        while not stop_event.is_set():
            try:
                value = await self.fetch_once(session)
                if value is not None:
                    async with self._lock:
                        self._value = value
                        self._ts = time.time()
            except Exception:
                pass

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=max(5, int(self.config.refresh_sec)))
            except asyncio.TimeoutError:
                continue

    async def get_latest(self) -> float | None:
        async with self._lock:
            if self._value is None:
                return None
            if (time.time() - self._ts) > self.config.max_age_sec:
                return None
            return self._value
