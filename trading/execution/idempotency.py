from __future__ import annotations

import threading
import time


class IdempotencyStore:
    def __init__(self, ttl_sec: int = 120):
        self.ttl_sec = max(1, int(ttl_sec))
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def _cleanup(self, now_ts: float):
        stale = [k for k, ts in self._seen.items() if ts <= now_ts]
        for key in stale:
            self._seen.pop(key, None)

    def restore(self, entries: dict[str, float]):
        now_ts = time.time()
        with self._lock:
            self._cleanup(now_ts)
            for key, expiry in entries.items():
                try:
                    exp = float(expiry)
                except (TypeError, ValueError):
                    continue
                if exp > now_ts:
                    self._seen[str(key)] = exp

    def put_if_absent(self, key: str) -> bool:
        now_ts = time.time()
        with self._lock:
            self._cleanup(now_ts)
            if key in self._seen:
                return False
            self._seen[key] = now_ts + self.ttl_sec
            return True

    def get_expiry(self, key: str) -> float | None:
        now_ts = time.time()
        with self._lock:
            self._cleanup(now_ts)
            expiry = self._seen.get(key)
            return float(expiry) if expiry is not None else None

    def clear(self):
        with self._lock:
            self._seen.clear()

