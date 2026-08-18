"""Expected calibration error over the held-out test interval.

ECE answers "when the model says 0.7, does it win 70% of the time?" It is the
population-weighted mean gap between predicted probability and realised rate,
measured in bins. Without it, a calibrator can be shipped with no evidence that
it improved anything.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from ai.train import _reliability


class ReliabilityTests(unittest.TestCase):
    def test_perfect_calibration_scores_zero(self):
        # Half the rows predicted at 1.0 and won; half at 0.0 and lost.
        probs = np.array([1.0] * 50 + [0.0] * 50)
        y = pd.Series([1] * 50 + [0] * 50)

        self.assertAlmostEqual(_reliability(y, probs), 0.0, places=9)

    def test_confident_but_wrong_scores_the_full_gap(self):
        # Model always says 0.9; the event never happens.
        probs = np.full(100, 0.9)
        y = pd.Series([0] * 100)

        self.assertAlmostEqual(_reliability(y, probs), 0.9, places=9)

    def test_error_is_weighted_by_bin_population(self):
        # 90 rows are perfectly calibrated at 0.0; 10 rows sit at 1.0 and lose.
        # The 1.0 bin contributes its full gap, but only for a tenth of the rows.
        probs = np.array([0.0] * 90 + [1.0] * 10)
        y = pd.Series([0] * 90 + [0] * 10)

        self.assertAlmostEqual(_reliability(y, probs), 0.1, places=9)

    def test_a_calibrated_prediction_beats_an_overconfident_one(self):
        # True win rate is 30%. One model says 0.3, the other says 0.95.
        y = pd.Series([1] * 30 + [0] * 70)
        honest = np.full(100, 0.3)
        overconfident = np.full(100, 0.95)

        self.assertLess(_reliability(y, honest), _reliability(y, overconfident))

    def test_empty_input_is_not_a_number(self):
        self.assertTrue(math.isnan(_reliability(pd.Series([], dtype=int), np.array([]))))


if __name__ == "__main__":
    unittest.main()
