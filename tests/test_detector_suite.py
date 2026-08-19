"""Every gating layer as an episode, observed alongside the generator.

The parity tests are the important ones: whatever the suite reports as fully
passed must be exactly what the generator turned into a signal. Because the
suite reads the generator's own trace rather than re-calling each layer, that
agreement holds by construction — these tests exist to catch the day someone
changes how the trace is written.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.detector_suite import DetectorSuite, LAYER_KINDS
from core.detectors import EpisodeState
from core.feature_engineering import sanitize_feature_frame
from core.indicators import compute_indicators
from core.market_regime import detect_market_regime
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator, SignalResult
from core.volume_profile import compute_volume_profile


def _series(bars: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    steps = rng.normal(0.0, 0.004, size=bars)
    for start in (120, 200, 260):
        steps[start : start + 4] += 0.05
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.004, size=bars)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.004, size=bars)))
    return pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "volume": rng.lognormal(10.0, 0.5, size=bars),
        },
        index=pd.date_range("2026-01-01", periods=bars, freq="15min", tz="UTC"),
    )


def _context(hist: pd.DataFrame) -> SignalContext:
    return SignalContext(
        symbol="TEST/USDT",
        df=hist,
        volume_profile=compute_volume_profile(hist),
        regime=detect_market_regime(hist),
        sentiment_index=None,
        sentiment_source="unavailable",
        funding_rate=None,
        long_short_ratio=None,
    )


class CoverageTests(unittest.TestCase):
    def test_every_gating_layer_has_a_detector(self):
        suite = DetectorSuite(SignalConfig())
        enriched = sanitize_feature_frame(compute_indicators(_series()))

        snapshot = suite.observe(_context(enriched.iloc[:100]))

        self.assertEqual(set(snapshot.detectors), set(LAYER_KINDS))

    def test_layer5_is_a_gate_on_this_branch(self):
        # Unlike the older line, _layer5_tp_sl_levels here returns a pass flag
        # and can reject a candidate, so it belongs in the suite.
        self.assertIn("layer5_tp_sl", LAYER_KINDS)

    def test_layers_are_declared_in_generator_order(self):
        self.assertEqual(LAYER_KINDS[0], "regime_filter")
        self.assertEqual(LAYER_KINDS[1], "layer1_pump_detection")
        self.assertEqual(LAYER_KINDS[-1], "layer5_tp_sl")


class ParityTests(unittest.TestCase):
    def test_full_pass_matches_the_generator_firing(self):
        enriched = sanitize_feature_frame(compute_indicators(_series()))
        suite = DetectorSuite(SignalConfig(), confirmations_required=1)

        mismatches = []
        for i in range(80, 280):
            snapshot = suite.observe(_context(enriched.iloc[: i + 1]))
            if snapshot.signal_produced != snapshot.all_layers_passed:
                mismatches.append(i)

        self.assertEqual(mismatches, [])

    def test_a_stubbed_signal_confirms_every_layer(self):
        """Guard against the parity test passing only because nothing fires."""

        full_trace = {
            "failed_layer": None,
            "layers": {kind: {"passed": True} for kind in LAYER_KINDS},
        }
        full_trace["layers"]["layer1_pump_detection"]["side"] = "SHORT"

        def _always(self, context):
            self.last_diagnostics = full_trace
            return SignalResult(
                signal_id="stub", symbol="T", side="SHORT", entry=100.0, sl=102.0, tp=97.0
            )

        enriched = sanitize_feature_frame(compute_indicators(_series()))
        suite = DetectorSuite(SignalConfig(), confirmations_required=1)

        with patch.object(SignalGenerator, "generate", _always):
            snapshot = suite.observe(_context(enriched.iloc[:120]))

        self.assertTrue(snapshot.all_layers_passed)
        self.assertTrue(snapshot.signal_produced)
        self.assertTrue(snapshot.permits_new_entry())
        for kind in LAYER_KINDS:
            with self.subTest(kind=kind):
                self.assertIs(snapshot.detectors[kind].state, EpisodeState.CONFIRMED)


class LifecycleTests(unittest.TestCase):
    def test_a_layer_that_stops_passing_decays_then_ends(self):
        states = iter(
            [
                {"failed_layer": None, "layers": {k: {"passed": True} for k in LAYER_KINDS}},
                {"failed_layer": "layer2_weakness_confirmation", "layers": {"regime_filter": {"passed": True}}},
                {"failed_layer": "layer2_weakness_confirmation", "layers": {"regime_filter": {"passed": True}}},
            ]
        )

        def _scripted(self, context):
            self.last_diagnostics = next(states)
            return None

        enriched = sanitize_feature_frame(compute_indicators(_series()))
        suite = DetectorSuite(SignalConfig(), confirmations_required=1, decay_tolerance=2)

        with patch.object(SignalGenerator, "generate", _scripted):
            context = _context(enriched.iloc[:120])
            suite.observe(context)
            after_one = suite.observe(context)
            after_two = suite.observe(context)

        self.assertIs(after_one.detectors["layer5_tp_sl"].state, EpisodeState.DECAYING)
        self.assertTrue(after_one.permits_exit())
        self.assertFalse(after_one.permits_new_entry())
        self.assertIs(after_two.detectors["layer5_tp_sl"].state, EpisodeState.COOLDOWN)

    def test_the_blocking_layer_is_reported(self):
        enriched = sanitize_feature_frame(compute_indicators(_series()))
        suite = DetectorSuite(SignalConfig())

        snapshot = suite.observe(_context(enriched.iloc[:120]))

        if not snapshot.all_layers_passed:
            self.assertIsNotNone(snapshot.blocking_layer)


if __name__ == "__main__":
    unittest.main()
