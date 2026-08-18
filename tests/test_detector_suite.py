"""All five layers as episodes, running alongside the strategy unchanged.

The suite observes the same four gating layers the generator uses, in the same
order, reusing the same methods. Two things it adds that the generator cannot
express:

* each layer has a lifecycle, so a weakening setup is DECAYING rather than
  simply absent;
* a layer that stops firing decays instead of freezing, because downstream
  layers keep observing "no evidence" when the upstream layer goes quiet.

The parity tests are the important ones: whatever the suite reports as fully
confirmed must be exactly what the generator turns into a signal. If those ever
diverge, the extraction changed behaviour and is wrong.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.detectors import EpisodeState
from core.detector_suite import DetectorSuite, LAYER_KINDS
from core.indicators import compute_indicators
from core.market_regime import detect_market_regime
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
from core.volume_profile import compute_volume_profile


def _permissive() -> SignalConfig:
    return SignalConfig(
        rsi_high=52.0,
        volume_spike_threshold=1.0,
        entry_tolerance_pct=0.08,
        vwap_tolerance_pct=0.08,
        msb_break_buffer_pct=0.0,
        weakness_lookback=2,
        msb_recent_bars=10,
    )


def _series(bars: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    steps = rng.normal(0.0, 0.004, size=bars)
    for start in (120, 240, 330):
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
        suite = DetectorSuite(_permissive())

        snapshot = suite.observe(_context(compute_indicators(_series()).iloc[:100]))

        self.assertEqual(set(snapshot.detectors), set(LAYER_KINDS))

    def test_layers_are_observed_in_the_generator_order(self):
        self.assertEqual(
            LAYER_KINDS, ("pump", "weakness", "entry_location", "fake_filter")
        )


class PropagationTests(unittest.TestCase):
    def setUp(self):
        self.enriched = compute_indicators(_series())
        self.suite = DetectorSuite(_permissive())

    def test_downstream_layers_decay_when_the_pump_goes_quiet(self):
        # Feed bars until the pump confirms, then feed a quiet stretch and check
        # that the downstream detectors moved rather than froze.
        confirmed_at = None
        for i in range(60, len(self.enriched)):
            snapshot = self.suite.observe(_context(self.enriched.iloc[: i + 1]))
            if snapshot.detectors["pump"].state is EpisodeState.CONFIRMED:
                confirmed_at = i
                break
        if confirmed_at is None:
            self.skipTest("pump never confirmed on this series")

        for _ in range(6):
            confirmed_at += 1
            snapshot = self.suite.observe(_context(self.enriched.iloc[: confirmed_at + 1]))

        self.assertIsNot(snapshot.detectors["weakness"].state, EpisodeState.CONFIRMED)

    def test_the_blocking_layer_is_named(self):
        snapshot = self.suite.observe(_context(self.enriched.iloc[:80]))

        if not snapshot.permits_new_entry():
            self.assertIn(snapshot.blocking_layer, LAYER_KINDS)


class PermissionTests(unittest.TestCase):
    def test_a_suite_with_a_quiet_pump_refuses_entry(self):
        suite = DetectorSuite(_permissive())
        enriched = compute_indicators(_series())

        snapshot = suite.observe(_context(enriched.iloc[:60]))

        self.assertFalse(snapshot.permits_new_entry())

    def test_exit_stays_permitted_while_any_layer_is_decaying(self):
        suite = DetectorSuite(_permissive())
        enriched = compute_indicators(_series())
        seen_decay = False

        for i in range(60, 260):
            snapshot = suite.observe(_context(enriched.iloc[: i + 1]))
            if any(
                s.state is EpisodeState.DECAYING for s in snapshot.detectors.values()
            ):
                seen_decay = True
                self.assertTrue(snapshot.permits_exit())
                break

        if not seen_decay:
            self.skipTest("no layer decayed on this series")


class ParityTests(unittest.TestCase):
    """The extraction must not change what counts as a signal."""

    def test_full_confirmation_matches_the_generator_firing(self):
        config = _permissive()
        enriched = compute_indicators(_series())
        generator = SignalGenerator(config)
        # confirmations_required=1 makes a single firing bar confirm, which is
        # what the generator does; anything higher would deliberately differ.
        suite = DetectorSuite(config, confirmations_required=1)

        mismatches = []
        fired = 0
        blocked_at: set[str] = set()
        for i in range(60, 300):
            context = _context(enriched.iloc[: i + 1])
            produced = generator.generate(context) is not None
            snapshot = suite.observe(context)
            fired += produced
            if snapshot.blocking_layer:
                blocked_at.add(snapshot.blocking_layer)
            if produced != snapshot.all_layers_passed:
                mismatches.append(i)

        self.assertEqual(mismatches, [])
        # Guard against a vacuous pass: if the generator never fired and no
        # layer ever blocked, the comparison above proved nothing.
        self.assertGreater(fired, 0, "the generator must fire for parity to mean anything")
        self.assertGreaterEqual(len(blocked_at), 2, "several layers must be exercised")

    def test_the_generator_is_not_mutated_by_the_suite(self):
        config = _permissive()
        enriched = compute_indicators(_series())
        generator = SignalGenerator(config)
        context = _context(enriched.iloc[:200])

        before = generator.generate(context)
        DetectorSuite(config).observe(context)
        after = generator.generate(context)

        self.assertEqual(before is None, after is None)


if __name__ == "__main__":
    unittest.main()
