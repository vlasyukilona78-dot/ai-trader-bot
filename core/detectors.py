"""Market episodes with their own lifecycle, independent of any strategy.

A layer that returns a boolean each bar cannot express the difference between
"this setup just appeared", "this setup is established" and "this setup has
been fading for two bars". That difference matters: the last case should let an
open position exit while refusing to open a new one.

An :class:`EpisodeDetector` wraps any per-bar condition and adds:

* an ordered lifecycle — READY, CANDIDATE, CONFIRMED, DECAYING, COOLDOWN;
* a stable episode identity that survives while the episode lives and changes
  when a genuinely new one starts;
* quality flags that travel with the snapshot, so a degraded observation cannot
  silently authorise an entry.

The detector holds no opinion about what the condition means. It is reusable by
any strategy that cares about the same episode.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class EpisodeState(Enum):
    """Ordered lifecycle of one detected episode."""

    #: No episode in progress.
    READY = "ready"
    #: Evidence is accumulating but has not met the confirmation bar.
    CANDIDATE = "candidate"
    #: Enough consecutive evidence; the episode is established.
    CONFIRMED = "confirmed"
    #: Evidence weakened. Exits are allowed; new entries are not.
    DECAYING = "decaying"
    #: The episode ended and re-entry is suppressed for a while.
    COOLDOWN = "cooldown"

    def permits_new_entry(self) -> bool:
        """Only an established episode may open new exposure."""

        return self is EpisodeState.CONFIRMED

    def permits_exit(self) -> bool:
        """A weakening episode is still a reason to leave."""

        return self in (EpisodeState.CONFIRMED, EpisodeState.DECAYING)


class QualityFlag(Enum):
    """A fact about the observation that a consumer must weigh."""

    WARMUP = "warmup"
    INSUFFICIENT_HISTORY = "insufficient_history"
    ESTIMATED_COMPONENT = "estimated_component"
    STALE = "stale"


#: Flags that make an observation unusable for opening exposure.
_BLOCKING_FLAGS = frozenset(
    {QualityFlag.WARMUP, QualityFlag.INSUFFICIENT_HISTORY, QualityFlag.STALE}
)


@dataclass(frozen=True)
class DetectorSnapshot:
    """Immutable publication of a detector's state at one observation."""

    kind: str
    version: str
    state: EpisodeState
    side: str | None
    episode_id: str | None
    evidence_count: int
    confirmations: int
    quality_flags: tuple[QualityFlag, ...] = ()

    def entry_eligible(self) -> bool:
        """Whether this snapshot may support opening new exposure."""

        if not self.state.permits_new_entry():
            return False
        return not any(flag in _BLOCKING_FLAGS for flag in self.quality_flags)


class EpisodeDetector:
    """Drives one episode lifecycle from a per-bar condition.

    Args:
        kind: Stable name of the episode this detector recognises.
        version: Detector version; changing the logic must change this.
        confirmations_required: Consecutive firings before CONFIRMED.
        decay_tolerance: Consecutive quiet observations a confirmed episode
            survives before it ends.
        cooldown_bars: Observations to suppress re-entry after an episode ends.
    """

    def __init__(
        self,
        *,
        kind: str,
        version: str,
        confirmations_required: int = 2,
        decay_tolerance: int = 2,
        cooldown_bars: int = 3,
    ) -> None:
        if not kind.strip():
            raise ValueError("detector requires a kind")
        if not version.strip():
            raise ValueError("detector requires a version")
        if confirmations_required <= 0:
            raise ValueError("confirmations_required must be positive")
        if decay_tolerance <= 0:
            raise ValueError("decay_tolerance must be positive")
        if cooldown_bars < 0:
            raise ValueError("cooldown_bars must not be negative")

        self.kind = kind
        self.version = version
        self.confirmations_required = confirmations_required
        self.decay_tolerance = decay_tolerance
        self.cooldown_bars = cooldown_bars

        self._state = EpisodeState.READY
        self._side: str | None = None
        self._episode_id: str | None = None
        self._episode_seq = 0
        self._evidence = 0
        self._confirmations = 0
        self._quiet = 0
        self._cooldown_left = 0
        self._flags: tuple[QualityFlag, ...] = ()

    @property
    def state(self) -> EpisodeState:
        return self._state

    def snapshot(self) -> DetectorSnapshot:
        """Publish the current state without advancing it."""

        return DetectorSnapshot(
            kind=self.kind,
            version=self.version,
            state=self._state,
            side=self._side,
            episode_id=self._episode_id,
            evidence_count=self._evidence,
            confirmations=self._confirmations,
            quality_flags=self._flags,
        )

    def observe(
        self,
        *,
        fired: bool,
        side: str | None,
        quality_flags: tuple[QualityFlag, ...] = (),
    ) -> DetectorSnapshot:
        """Advance the lifecycle by one observation and publish the result."""

        self._flags = tuple(quality_flags)

        if fired and side is not None and self._side is not None and side != self._side:
            # A flip is a different episode, not a continuation of this one, and
            # not a fade either: the old episode ends and a fresh one opens
            # immediately rather than serving out a cooldown.
            self._reset_ready()

        if fired:
            self._on_evidence(side)
        else:
            self._on_quiet()

        return self.snapshot()

    # -- transitions ------------------------------------------------------

    def _on_evidence(self, side: str | None) -> None:
        if self._state in (EpisodeState.READY, EpisodeState.COOLDOWN):
            if self._state is EpisodeState.COOLDOWN and self._cooldown_left > 0:
                # Still suppressed; count the bar down without reopening.
                self._cooldown_left -= 1
                if self._cooldown_left <= 0:
                    self._state = EpisodeState.READY
                return
            self._begin_episode(side)
            return

        self._side = side or self._side
        self._evidence += 1
        self._quiet = 0
        self._confirmations += 1
        if self._confirmations >= self.confirmations_required:
            self._state = EpisodeState.CONFIRMED

    def _on_quiet(self) -> None:
        if self._state is EpisodeState.COOLDOWN:
            self._cooldown_left -= 1
            if self._cooldown_left <= 0:
                self._state = EpisodeState.READY
                self._episode_id = None
                self._side = None
            return

        if self._state is EpisodeState.CANDIDATE:
            # An unconfirmed candidate that vanishes was never an episode.
            self._reset_ready()
            return

        if self._state in (EpisodeState.CONFIRMED, EpisodeState.DECAYING):
            self._quiet += 1
            self._state = EpisodeState.DECAYING
            if self._quiet >= self.decay_tolerance:
                self._end_episode()

    def _begin_episode(self, side: str | None) -> None:
        self._episode_seq += 1
        self._side = side
        self._evidence = 1
        self._confirmations = 1
        self._quiet = 0
        self._episode_id = self._make_episode_id(side)
        self._state = (
            EpisodeState.CONFIRMED
            if self._confirmations >= self.confirmations_required
            else EpisodeState.CANDIDATE
        )

    def _end_episode(self) -> None:
        self._state = EpisodeState.COOLDOWN
        self._cooldown_left = self.cooldown_bars
        self._evidence = 0
        self._confirmations = 0
        self._quiet = 0
        if self.cooldown_bars == 0:
            self._state = EpisodeState.READY
            self._episode_id = None
            self._side = None

    def _reset_ready(self) -> None:
        self._state = EpisodeState.READY
        self._episode_id = None
        self._side = None
        self._evidence = 0
        self._confirmations = 0
        self._quiet = 0

    def _make_episode_id(self, side: str | None) -> str:
        material = f"{self.kind}|{self.version}|{self._episode_seq}|{side or ''}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
