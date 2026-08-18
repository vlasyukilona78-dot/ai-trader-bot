"""The pump/panic layer as a reusable episode detector.

This wraps :meth:`SignalGenerator._layer1_pump_detection` without changing it,
so detection is byte-for-byte the same condition the strategy already uses.
What it adds is the lifecycle: consecutive firings become one identified
episode, and a pump that stops firing enters DECAYING — a state where an open
position may still be exited but a new one may not be opened.

Wrapping rather than rewriting keeps the existing strategy behaviour intact
while making the episode available to any other consumer.
"""

from __future__ import annotations

import pandas as pd

from core.detectors import DetectorSnapshot, EpisodeDetector, QualityFlag
from core.signal_generator import SignalConfig, SignalGenerator

DETECTOR_VERSION = "pump-v1"


class PumpDetector:
    """Episode lifecycle over the existing pump/panic condition."""

    def __init__(
        self,
        config: SignalConfig | None = None,
        *,
        confirmations_required: int = 2,
        decay_tolerance: int = 2,
        cooldown_bars: int = 3,
        minimum_history: int = 1,
    ) -> None:
        self._generator = SignalGenerator(config or SignalConfig())
        self._episode = EpisodeDetector(
            kind="pump",
            version=DETECTOR_VERSION,
            confirmations_required=confirmations_required,
            decay_tolerance=decay_tolerance,
            cooldown_bars=cooldown_bars,
        )
        self.minimum_history = max(1, int(minimum_history))
        self.last_metrics: dict[str, float | str] = {}

    @property
    def state(self):
        return self._episode.state

    def snapshot(self) -> DetectorSnapshot:
        return self._episode.snapshot()

    def observe(self, df: pd.DataFrame) -> DetectorSnapshot:
        """Evaluate one bar of history and advance the episode."""

        flags: list[QualityFlag] = []
        if len(df) < self.minimum_history:
            flags.append(QualityFlag.INSUFFICIENT_HISTORY)

        side, metrics = self._generator._layer1_pump_detection(df)
        self.last_metrics = metrics

        return self._episode.observe(
            fired=side is not None,
            side=side,
            quality_flags=tuple(flags),
        )
