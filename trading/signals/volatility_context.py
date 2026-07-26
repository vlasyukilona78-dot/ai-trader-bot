from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class VolatilityContextConfig:
    percentile: float = 0.80
    max_age_sec: float = 900.0
    min_observations: int = 20
    fallback_floor: float = 0.046


@dataclass
class _Observation:
    value: float
    ts: float


class VolatilityContext:
    """Cross-sectional volatility floor: how volatile a coin is *relative to the
    rest of the board right now*.

    The measured edge is comparative - violent pumps resolve, calm ones drift -
    but a hardcoded ATR number silently changes meaning when the whole market
    calms down or heats up. Against March data a fixed 4.6% floor admitted only
    3 of 19 signals because that period was simply quieter, whereas the same
    percentile keeps admitting the top slice in any regime.

    Observations are kept one-per-symbol so a frequently scanned symbol cannot
    drag the distribution, and expire so the floor tracks the current regime.
    """

    def __init__(self, config: VolatilityContextConfig | None = None):
        self.config = config or VolatilityContextConfig()
        self._observations: dict[str, _Observation] = {}
        # The floor is frozen from the previous completed sweep. Reading the
        # in-progress dictionary would make a candidate's fate depend on where it
        # happened to sit in the scan order: the same coin was blocked when it
        # came first and admitted after a run of calm symbols had dragged the
        # percentile down.
        self._frozen: list[float] = []

    def observe(self, symbol: str, atr_pct: float, *, now: float | None = None):
        if atr_pct is None or atr_pct != atr_pct or atr_pct <= 0:
            return
        self._observations[symbol] = _Observation(float(atr_pct), now if now is not None else time.time())

    def _fresh_values(self, now: float | None = None) -> list[float]:
        now = now if now is not None else time.time()
        cutoff = now - self.config.max_age_sec
        stale = [s for s, o in self._observations.items() if o.ts < cutoff]
        for s in stale:
            del self._observations[s]
        return sorted(o.value for o in self._observations.values())

    def start_sweep(self, *, now: float | None = None) -> None:
        """Freeze the distribution that this sweep will be judged against.

        Called once per scan, before any symbol is evaluated, so every candidate
        in the sweep faces the same floor regardless of ordering.
        """
        self._frozen = self._fresh_values(now)

    def floor(self, *, now: float | None = None) -> float:
        """ATR cutoff for the current sweep. Falls back to the fixed floor until
        enough symbols have been seen, so a cold start cannot admit everything."""
        values = self._frozen if self._frozen else self._fresh_values(now)
        if len(values) < self.config.min_observations:
            return self.config.fallback_floor

        pct = min(max(self.config.percentile, 0.0), 0.999)
        idx = int(pct * (len(values) - 1))
        return float(values[idx])

    @property
    def observed_symbols(self) -> int:
        return len(self._observations)
