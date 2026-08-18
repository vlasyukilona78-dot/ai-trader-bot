"""Outside the range it was validated on, the model abstains instead of guessing.

A gradient-boosted model extrapolates by returning the nearest leaf, which
looks like a confident answer and is not one. Pump exhaustion lives in the tail
of every distribution, so the entries this strategy cares about are exactly the
ones most likely to sit outside the support the model was fitted on.

Drift raises an alert and starts a challenger. It never widens the envelope:
letting live data expand its own validity is how a model quietly starts trading
conditions nobody checked.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ai.ood import (
    InferenceAction,
    InferenceContext,
    OodEnvelope,
    detect_drift,
    evaluate_ood,
    fit_envelope,
)


def _envelope() -> OodEnvelope:
    frame = pd.DataFrame({"rsi": [10.0, 50.0, 90.0], "atr": [1.0, 2.0, 3.0]})
    return fit_envelope(frame, version="v1", regimes=("PUMP", "PANIC"))


def _context(**overrides) -> InferenceContext:
    base = dict(features={"rsi": 50.0, "atr": 2.0}, regime="PUMP", quality_ok=True)
    base.update(overrides)
    return InferenceContext(**base)


class FittingTests(unittest.TestCase):
    def test_the_envelope_spans_the_observed_range(self):
        envelope = _envelope()

        self.assertAlmostEqual(envelope.bounds["rsi"].minimum, 10.0)
        self.assertAlmostEqual(envelope.bounds["rsi"].maximum, 90.0)

    def test_fitting_records_the_allowed_regimes(self):
        self.assertEqual(set(_envelope().valid_regimes), {"PUMP", "PANIC"})

    def test_fitting_requires_at_least_one_regime(self):
        with self.assertRaises(ValueError):
            fit_envelope(pd.DataFrame({"rsi": [1.0]}), version="v1", regimes=())

    def test_fitting_ignores_absent_values(self):
        frame = pd.DataFrame({"rsi": [10.0, np.nan, 90.0]})

        envelope = fit_envelope(frame, version="v1", regimes=("PUMP",))

        self.assertAlmostEqual(envelope.bounds["rsi"].maximum, 90.0)

    def test_a_feature_with_no_observations_is_refused(self):
        frame = pd.DataFrame({"rsi": [np.nan, np.nan]})

        with self.assertRaises(ValueError):
            fit_envelope(frame, version="v1", regimes=("PUMP",))


class InsideSupportTests(unittest.TestCase):
    def test_a_point_inside_the_envelope_is_allowed(self):
        decision = evaluate_ood(_envelope(), _context())

        self.assertIs(decision.action, InferenceAction.ALLOW)
        self.assertEqual(decision.reasons, ())

    def test_the_boundary_itself_is_inside(self):
        decision = evaluate_ood(_envelope(), _context(features={"rsi": 90.0, "atr": 3.0}))

        self.assertIs(decision.action, InferenceAction.ALLOW)


class AbstentionTests(unittest.TestCase):
    def test_a_feature_above_the_range_abstains(self):
        decision = evaluate_ood(_envelope(), _context(features={"rsi": 200.0, "atr": 2.0}))

        self.assertIs(decision.action, InferenceAction.ABSTAIN)
        self.assertIn("FEATURE_OOD:rsi", decision.reasons)

    def test_a_feature_below_the_range_abstains(self):
        decision = evaluate_ood(_envelope(), _context(features={"rsi": -5.0, "atr": 2.0}))

        self.assertIn("FEATURE_OOD:rsi", decision.reasons)

    def test_an_absent_feature_abstains(self):
        decision = evaluate_ood(_envelope(), _context(features={"rsi": 50.0}))

        self.assertIn("FEATURE_MISSING:atr", decision.reasons)

    def test_an_unvalidated_regime_abstains(self):
        decision = evaluate_ood(_envelope(), _context(regime="TREND"))

        self.assertIn("REGIME_OOD", decision.reasons)

    def test_degraded_input_quality_abstains(self):
        decision = evaluate_ood(_envelope(), _context(quality_ok=False))

        self.assertIn("FEATURE_QUALITY_INVALID", decision.reasons)

    def test_every_reason_is_reported_not_just_the_first(self):
        decision = evaluate_ood(
            _envelope(), _context(features={"rsi": 999.0}, regime="TREND")
        )

        self.assertEqual(
            set(decision.reasons), {"FEATURE_OOD:rsi", "FEATURE_MISSING:atr", "REGIME_OOD"}
        )


class DriftTests(unittest.TestCase):
    def test_drift_above_the_threshold_raises_an_alert(self):
        envelope = _envelope()
        contexts = [_context(features={"rsi": 999.0, "atr": 2.0}) for _ in range(10)]

        report = detect_drift(envelope, contexts, alert_rate=0.5)

        self.assertTrue(report.alert)
        self.assertEqual(report.abstentions, 10)

    def test_no_alert_when_observations_stay_in_support(self):
        report = detect_drift(_envelope(), [_context() for _ in range(10)], alert_rate=0.5)

        self.assertFalse(report.alert)

    def test_drift_never_widens_the_envelope(self):
        envelope = _envelope()
        before = envelope.bounds["rsi"].maximum

        detect_drift(
            envelope,
            [_context(features={"rsi": 999.0, "atr": 2.0}) for _ in range(10)],
            alert_rate=0.1,
        )

        self.assertAlmostEqual(envelope.bounds["rsi"].maximum, before)

    def test_an_alert_asks_for_a_challenger_rather_than_a_widening(self):
        report = detect_drift(
            _envelope(),
            [_context(features={"rsi": 999.0, "atr": 2.0}) for _ in range(10)],
            alert_rate=0.1,
        )

        self.assertEqual(report.action, "ALERT_AND_CREATE_CHALLENGER")

    def test_drift_needs_observations(self):
        with self.assertRaises(ValueError):
            detect_drift(_envelope(), [], alert_rate=0.5)


if __name__ == "__main__":
    unittest.main()
