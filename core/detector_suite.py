"""The four gating layers as independent episodes, observed together.

Each detector wraps the corresponding ``SignalGenerator`` method unchanged, so
what gets detected is identical to what the strategy already detects. The suite
runs them in the generator's own order and adds what a chain of booleans cannot
express:

* every layer has a lifecycle, so a fading setup reads as DECAYING rather than
  simply absent — and DECAYING permits an exit while refusing a new entry;
* when an upstream layer goes quiet, downstream layers observe "no evidence"
  instead of being skipped, so their episodes decay on schedule instead of
  freezing at whatever state they last reached.

The suite is an observer. It never mutates the generator and never decides
anything on its own; ``tests/test_detector_suite.py`` pins its verdict to the
generator's so the extraction cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.detectors import DetectorSnapshot, EpisodeDetector, EpisodeState, QualityFlag
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator

#: Gating layers, in the order the generator evaluates them. Layer 5 computes
#: target and stop levels rather than gating an entry, so it is not an episode.
LAYER_KINDS: tuple[str, ...] = ("pump", "weakness", "entry_location", "fake_filter")

SUITE_VERSION = "layers-v1"

#: Below this many bars the generator refuses outright, so anything observed
#: there is warmup rather than evidence.
_MINIMUM_HISTORY = 40


@dataclass(frozen=True)
class SuiteSnapshot:
    """One observation across every layer."""

    detectors: dict[str, DetectorSnapshot]
    all_layers_passed: bool
    blocking_layer: str | None

    def permits_new_entry(self) -> bool:
        """Every layer must be confirmed and clean to open exposure."""

        return all(
            self.detectors[kind].entry_eligible() for kind in self.detectors
        )

    def permits_exit(self) -> bool:
        """A weakening episode is still a reason to leave a position."""

        return any(
            snapshot.state.permits_exit() for snapshot in self.detectors.values()
        )


class DetectorSuite:
    """Runs the four gating layers as episodes, alongside the generator."""

    def __init__(
        self,
        config: SignalConfig | None = None,
        *,
        confirmations_required: int = 2,
        decay_tolerance: int = 2,
        cooldown_bars: int = 3,
    ) -> None:
        self._generator = SignalGenerator(config or SignalConfig())
        self._detectors: dict[str, EpisodeDetector] = {
            kind: EpisodeDetector(
                kind=kind,
                version=SUITE_VERSION,
                confirmations_required=confirmations_required,
                decay_tolerance=decay_tolerance,
                cooldown_bars=cooldown_bars,
            )
            for kind in LAYER_KINDS
        }
        self.last_details: dict[str, dict] = {}

    @property
    def detectors(self) -> dict[str, EpisodeDetector]:
        return dict(self._detectors)

    def snapshot(self) -> SuiteSnapshot:
        """Publish current state without advancing anything."""

        snapshots = {kind: d.snapshot() for kind, d in self._detectors.items()}
        return SuiteSnapshot(
            detectors=snapshots,
            all_layers_passed=False,
            blocking_layer=self._first_unconfirmed(snapshots),
        )

    def observe(self, context: SignalContext) -> SuiteSnapshot:
        """Evaluate every layer for one bar and advance each episode.

        Layers downstream of a failure still observe — with no evidence — so
        their episodes decay rather than freeze.
        """

        df = context.df
        warmup = len(df) < _MINIMUM_HISTORY
        flags: tuple[QualityFlag, ...] = (
            (QualityFlag.INSUFFICIENT_HISTORY,) if warmup else ()
        )

        fired: dict[str, bool] = {kind: False for kind in LAYER_KINDS}
        sides: dict[str, str | None] = {kind: None for kind in LAYER_KINDS}
        details: dict[str, dict] = {}
        blocking: str | None = None

        if warmup or df.empty:
            blocking = "pump"
        else:
            side, layer1 = self._generator._layer1_pump_detection(df)
            details["pump"] = dict(layer1)
            fired["pump"] = side is not None
            sides["pump"] = side

            if side is None:
                blocking = "pump"
            else:
                ok2, layer2 = self._generator._layer2_weakness_confirmation(df, side)
                details["weakness"] = dict(layer2)
                fired["weakness"] = ok2
                sides["weakness"] = side

                if not ok2:
                    blocking = "weakness"
                else:
                    ok3, layer3 = self._generator._layer3_entry_location(
                        df, side, context.volume_profile
                    )
                    details["entry_location"] = dict(layer3)
                    fired["entry_location"] = ok3
                    sides["entry_location"] = side

                    if not ok3:
                        blocking = "entry_location"
                    else:
                        ok4, layer4 = self._generator._layer4_fake_filter(
                            df=df,
                            side=side,
                            sentiment_index=context.sentiment_index,
                            sentiment_source=context.sentiment_source,
                            funding_rate=context.funding_rate,
                            long_short_ratio=context.long_short_ratio,
                        )
                        details["fake_filter"] = dict(layer4)
                        fired["fake_filter"] = ok4
                        sides["fake_filter"] = side
                        if not ok4:
                            blocking = "fake_filter"

        self.last_details = details

        snapshots = {
            kind: self._detectors[kind].observe(
                fired=fired[kind], side=sides[kind], quality_flags=flags
            )
            for kind in LAYER_KINDS
        }

        return SuiteSnapshot(
            detectors=snapshots,
            all_layers_passed=all(fired.values()),
            blocking_layer=blocking or self._first_unconfirmed(snapshots),
        )

    @staticmethod
    def _first_unconfirmed(snapshots: dict[str, DetectorSnapshot]) -> str | None:
        for kind in LAYER_KINDS:
            if snapshots[kind].state is not EpisodeState.CONFIRMED:
                return kind
        return None
