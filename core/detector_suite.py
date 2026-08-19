"""The generator's gating layers as independent episodes with a lifecycle.

Each layer is a boolean recomputed from scratch every bar, so the strategy
cannot distinguish "this setup just appeared" from "this setup has been fading
for two bars". The distinction matters: a weakening setup should still let an
open position exit while refusing to open a new one.

Rather than re-calling each layer, the suite reads the generator's own
per-layer trace. ``SignalGenerator.generate`` already records every gate it
evaluated and which one failed, so driving the episodes from that record is
faithful by construction: the suite cannot disagree with the generator about
what fired, and it does not break when a layer's signature changes. Layers that
were never reached observe "no evidence", so their episodes decay on schedule
instead of freezing at whatever state they last held.

The suite is an observer. It runs the generator and reads the result; it never
changes what the generator decides.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.detectors import DetectorSnapshot, EpisodeDetector, EpisodeState, QualityFlag
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator

#: Gating layers in the order ``generate`` evaluates them. Every one of these
#: can reject a candidate outright, layer 5 included on this branch.
LAYER_KINDS: tuple[str, ...] = (
    "regime_filter",
    "layer1_pump_detection",
    "layer2_weakness_confirmation",
    "layer3_entry_location",
    "layer4_fake_filter",
    "layer4_degraded_guard",
    "layer5_tp_sl",
)

SUITE_VERSION = "layers-v1"


@dataclass(frozen=True)
class SuiteSnapshot:
    """One observation across every gating layer."""

    detectors: dict[str, DetectorSnapshot]
    all_layers_passed: bool
    blocking_layer: str | None
    signal_produced: bool

    def permits_new_entry(self) -> bool:
        """Every layer must be confirmed and clean to open exposure."""

        return all(snapshot.entry_eligible() for snapshot in self.detectors.values())

    def permits_exit(self) -> bool:
        """A weakening episode is still a reason to leave a position."""

        return any(snapshot.state.permits_exit() for snapshot in self.detectors.values())


class DetectorSuite:
    """Drives one episode lifecycle per gating layer, alongside the generator."""

    def __init__(
        self,
        config: SignalConfig | None = None,
        *,
        generator: SignalGenerator | None = None,
        confirmations_required: int = 2,
        decay_tolerance: int = 2,
        cooldown_bars: int = 3,
    ) -> None:
        self._generator = generator or SignalGenerator(config or SignalConfig())
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
        self.last_trace: dict = {}

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
            signal_produced=False,
        )

    def observe(self, context: SignalContext) -> SuiteSnapshot:
        """Run one evaluation and advance every episode by one observation."""

        result = self._generator.generate(context)
        trace = dict(getattr(self._generator, "last_diagnostics", {}) or {})
        self.last_trace = trace

        layers = trace.get("layers", {}) or {}
        blocking = trace.get("failed_layer") or None

        # A layer absent from the trace was never reached, which is an absence
        # of evidence rather than a pass.
        fired = {
            kind: bool(layers.get(kind, {}).get("passed", False)) for kind in LAYER_KINDS
        }
        side = str(layers.get("layer1_pump_detection", {}).get("side") or "") or None

        flags: tuple[QualityFlag, ...] = ()
        if blocking == "layer0_input":
            flags = (QualityFlag.INSUFFICIENT_HISTORY,)

        snapshots = {
            kind: self._detectors[kind].observe(
                fired=fired[kind],
                side=side if fired[kind] else None,
                quality_flags=flags,
            )
            for kind in LAYER_KINDS
        }

        return SuiteSnapshot(
            detectors=snapshots,
            all_layers_passed=all(fired.values()),
            blocking_layer=blocking or self._first_unconfirmed(snapshots),
            signal_produced=result is not None,
        )

    @staticmethod
    def _first_unconfirmed(snapshots: dict[str, DetectorSnapshot]) -> str | None:
        for kind in LAYER_KINDS:
            if snapshots[kind].state is not EpisodeState.CONFIRMED:
                return kind
        return None
