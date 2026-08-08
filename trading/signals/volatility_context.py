from __future__ import annotations

import time
import math
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
        # Whether a sweep is active, tracked separately from the frozen list. On a
        # cold start that list is legitimately empty, and testing it for emptiness
        # made the floor fall through to the live observations of the sweep in
        # progress - reintroducing exactly the scan-order dependence freezing
        # exists to remove.
        self._sweep_active = False
        # All observations produced by one concurrent sweep are stamped with its
        # start time. Worker completion time is scheduling noise: around the TTL
        # boundary it used to decide which symbols survived into the next sweep,
        # making the next floor depend on worker order.
        self._sweep_timestamp: float | None = None

    def observe(self, symbol: str, atr_pct: float, *, now: float | None = None):
        if atr_pct is None or not math.isfinite(float(atr_pct)) or atr_pct <= 0:
            return
        observed_at = self._sweep_timestamp if self._sweep_active else now
        if observed_at is None:
            observed_at = time.time()
        if not math.isfinite(float(observed_at)):
            raise ValueError("observation_timestamp_must_be_finite")
        self._observations[symbol] = _Observation(float(atr_pct), float(observed_at))

    def _fresh_values(self, now: float | None = None) -> list[float]:
        now_value = float(now if now is not None else time.time())
        if not math.isfinite(now_value):
            raise ValueError("volatility_clock_must_be_finite")
        cutoff = now_value - self.config.max_age_sec
        stale = [s for s, o in self._observations.items() if o.ts < cutoff]
        for s in stale:
            del self._observations[s]
        return sorted(o.value for o in self._observations.values())

    def start_sweep(self, *, now: float | None = None) -> None:
        """Freeze the distribution that this sweep will be judged against.

        Called once per scan, before any symbol is evaluated, so every candidate
        in the sweep faces the same floor regardless of ordering. An empty freeze
        is still a freeze: the first sweep of a fresh process holds the fallback
        floor for all of its symbols instead of watching the distribution build up
        underneath it.
        """
        sweep_timestamp = time.time() if now is None else float(now)
        if not math.isfinite(sweep_timestamp):
            raise ValueError("sweep_timestamp_must_be_finite")
        self._frozen = self._fresh_values(sweep_timestamp)
        self._sweep_timestamp = sweep_timestamp
        self._sweep_active = True

    def floor(self, *, now: float | None = None) -> float:
        """ATR cutoff for the current sweep. Falls back to the fixed floor until
        enough symbols have been seen, so a cold start cannot admit everything."""
        values = self._frozen if self._sweep_active else self._fresh_values(now)
        if len(values) < self.config.min_observations:
            return self.config.fallback_floor

        pct = min(max(self.config.percentile, 0.0), 0.999)
        idx = int(pct * (len(values) - 1))
        return float(values[idx])

    @property
    def observed_symbols(self) -> int:
        return len(self._observations)
