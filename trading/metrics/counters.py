from __future__ import annotations

from collections import Counter


class MetricsCounter:
    def __init__(self):
        self._counter: Counter[str] = Counter()

    def inc(self, name: str, value: int = 1):
        self._counter[name] += int(value)

    def snapshot(self) -> dict[str, int]:
        return dict(self._counter)
