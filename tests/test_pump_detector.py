"""The pump layer, wrapped as an episode with a lifecycle.

The wrapper reuses the existing layer-1 logic unchanged, so what it detects is
identical. What it adds is memory: the same pump seen on three consecutive bars
is one episode rather than three unrelated firings, and a pump that stops
firing enters DECAYING instead of vanishing.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.detectors import EpisodeState
from core.pump_detector import PumpDetector
from core.signal_generator import SignalConfig


def _bar(*, rsi: float, volume_spike: float, close: float, bb_upper: float) -> dict:
    return {
        "rsi": rsi,
        "volume_spike": volume_spike,
        "close": close,
        "bb_upper": bb_upper,
        "bb_lower": -np.inf,
        "kc_upper": np.inf,
        "kc_lower": -np.inf,
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


_PUMPING = _bar(rsi=80.0, volume_spike=3.0, close=110.0, bb_upper=100.0)
_QUIET = _bar(rsi=50.0, volume_spike=1.0, close=100.0, bb_upper=120.0)


class DetectionParityTests(unittest.TestCase):
    def test_a_pumping_bar_fires_short(self):
        detector = PumpDetector(SignalConfig())

        snapshot = detector.observe(_frame([_PUMPING]))

        self.assertEqual(snapshot.side, "SHORT")

    def test_a_quiet_bar_does_not_fire(self):
        detector = PumpDetector(SignalConfig())

        snapshot = detector.observe(_frame([_QUIET]))

        self.assertIs(snapshot.state, EpisodeState.READY)

    def test_the_underlying_metrics_are_exposed(self):
        detector = PumpDetector(SignalConfig())

        detector.observe(_frame([_PUMPING]))

        self.assertAlmostEqual(detector.last_metrics["rsi"], 80.0)


class EpisodeTests(unittest.TestCase):
    def test_consecutive_pumping_bars_are_one_episode(self):
        detector = PumpDetector(SignalConfig(), confirmations_required=2)

        first = detector.observe(_frame([_PUMPING]))
        second = detector.observe(_frame([_PUMPING]))

        self.assertEqual(first.episode_id, second.episode_id)
        self.assertIs(second.state, EpisodeState.CONFIRMED)

    def test_a_pump_that_stops_decays_rather_than_disappearing(self):
        detector = PumpDetector(SignalConfig(), confirmations_required=1)
        detector.observe(_frame([_PUMPING]))

        snapshot = detector.observe(_frame([_QUIET]))

        self.assertIs(snapshot.state, EpisodeState.DECAYING)
        self.assertFalse(snapshot.state.permits_new_entry())
        self.assertTrue(snapshot.state.permits_exit())

    def test_short_history_is_flagged_rather_than_guessed(self):
        detector = PumpDetector(SignalConfig(), minimum_history=5)

        snapshot = detector.observe(_frame([_PUMPING]))

        self.assertFalse(snapshot.entry_eligible())


if __name__ == "__main__":
    unittest.main()
