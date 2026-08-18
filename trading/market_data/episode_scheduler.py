"""Decide which recording episodes are admitted, and record every one refused.

A collector that records only what triggered produces a population with no
comparison group, which is the defect already present in this project's
`level_dist` column: 30.6% missing, and missing in a way that correlates with the
gates. Every screen that reached a decisive answer did so because a matched
control existed.

Two rules follow, and both are enforced here rather than left to the caller.

Admission is **paired**: a triggered episode is admitted only together with its
control. Giving triggers priority under contention would drop controls exactly
during busy markets, which reintroduces the same bias from the other side. If
both cannot be held, neither is, and the pair is refused as a unit.

Refusals are **recorded**, with a reason. Pumps arrive in market-wide waves, so
episodes lost to a full scheduler are not a random thinning of the sample -- they
remove the busiest periods. A refusal that leaves no trace is indistinguishable
afterwards from a period when nothing happened.

Nothing here opens a socket, reads a clock or touches a disk. It is a pure
admission decision over timestamps supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class EpisodeKind(str, Enum):
    TRIGGERED = "triggered"
    CONTROL = "control"


class RefusalReason(str, Enum):
    NO_CAPACITY = "no_capacity"
    SYMBOL_ALREADY_RECORDING = "symbol_already_recording"
    CONTROL_UNPLACEABLE = "control_unplaceable"


@dataclass(frozen=True)
class Episode:
    symbol: str
    kind: EpisodeKind
    start_ts: int
    end_ts: int

    def __post_init__(self) -> None:
        if self.end_ts <= self.start_ts:
            raise ValueError("episode must end after it starts")


@dataclass(frozen=True)
class Refusal:
    symbol: str
    kind: EpisodeKind
    start_ts: int
    reason: RefusalReason


@dataclass(frozen=True)
class Admission:
    """Either a triggered episode with its control, or a refusal of the pair."""

    triggered: Episode | None
    control: Episode | None
    refusals: tuple[Refusal, ...]

    @property
    def admitted(self) -> bool:
        return self.triggered is not None


@dataclass
class EpisodeScheduler:
    """Admits paired episodes up to a fixed concurrency, refusing the rest aloud."""

    capacity: int
    episode_seconds: int
    control_delay_range: tuple[int, int] = (300, 3600)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _active: list[Episode] = field(default_factory=list, init=False)
    _refusals: list[Refusal] = field(default_factory=list, init=False)
    _admitted: list[Episode] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.capacity < 2:
            raise ValueError("capacity below two cannot hold a pair")
        if self.episode_seconds <= 0:
            raise ValueError("episode_seconds must be positive")
        low, high = self.control_delay_range
        if low <= 0 or high <= low:
            raise ValueError("control_delay_range must be a positive widening range")

    # ------------------------------------------------------------------ state
    def _release(self, now_ts: int) -> None:
        self._active = [e for e in self._active if e.end_ts > now_ts]

    def _recording(self, symbol: str) -> bool:
        return any(e.symbol == symbol for e in self._active)

    @property
    def in_flight(self) -> int:
        return len(self._active)

    @property
    def refusals(self) -> tuple[Refusal, ...]:
        return tuple(self._refusals)

    @property
    def admitted(self) -> tuple[Episode, ...]:
        return tuple(self._admitted)

    def refusal_rate(self) -> float:
        offered = len(self._refusals) + sum(
            1 for e in self._admitted if e.kind is EpisodeKind.TRIGGERED
        )
        return len(self._refusals) / offered if offered else 0.0

    # ------------------------------------------------------------- admission
    def offer(self, symbol: str, decision_ts: int, control_symbol: str) -> Admission:
        """Offer a triggered episode; it is admitted only with its control."""

        self._release(decision_ts)

        if self._recording(symbol):
            return self._refuse(symbol, decision_ts, RefusalReason.SYMBOL_ALREADY_RECORDING)

        # The pair needs two slots, held together or not at all.
        if self.capacity - len(self._active) < 2:
            return self._refuse(symbol, decision_ts, RefusalReason.NO_CAPACITY)

        low, high = self.control_delay_range
        control_start = int(decision_ts + int(self.rng.integers(low, high)))
        if self._recording(control_symbol) and control_symbol != symbol:
            return self._refuse(symbol, decision_ts, RefusalReason.CONTROL_UNPLACEABLE)

        triggered = Episode(symbol, EpisodeKind.TRIGGERED, decision_ts,
                            decision_ts + self.episode_seconds)
        control = Episode(control_symbol, EpisodeKind.CONTROL, control_start,
                          control_start + self.episode_seconds)
        self._active.extend((triggered, control))
        self._admitted.extend((triggered, control))
        return Admission(triggered=triggered, control=control, refusals=())

    def _refuse(self, symbol: str, ts: int, reason: RefusalReason) -> Admission:
        pair = (
            Refusal(symbol, EpisodeKind.TRIGGERED, ts, reason),
            Refusal(symbol, EpisodeKind.CONTROL, ts, reason),
        )
        self._refusals.extend(pair)
        return Admission(triggered=None, control=None, refusals=pair)
