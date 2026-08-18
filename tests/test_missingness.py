"""A missing feature is not a zero feature.

``cvd_div_short = 0`` means "no divergence was measured". ``cvd_div_short =
NaN`` means "we do not know". Filling the second with the first teaches the
model that absent data is a real, neutral observation, and there is no way for
it to tell the two apart afterwards.

Imputation statistics are fitted on the training rows only. A median taken over
the whole dataset carries information from the test interval back into training.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ai.missing import MissingnessPolicy, missing_report


class PreservationTests(unittest.TestCase):
    def test_absent_values_stay_nan_before_imputation(self):
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})

        report = missing_report(frame)

        self.assertAlmostEqual(report["a"], 1 / 3)

    def test_a_fully_present_column_reports_zero(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

        self.assertAlmostEqual(missing_report(frame)["a"], 0.0)

    def test_a_fully_absent_column_reports_one(self):
        frame = pd.DataFrame({"a": [np.nan, np.nan]})

        self.assertAlmostEqual(missing_report(frame)["a"], 1.0)


class IndicatorTests(unittest.TestCase):
    def test_an_indicator_column_marks_the_missing_rows(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})
        policy = MissingnessPolicy(add_indicators=True).fit(train)

        out = policy.transform(train)

        self.assertIn("a__missing", out.columns)
        self.assertEqual(list(out["a__missing"]), [0.0, 0.0, 0.0, 1.0])

    def test_indicators_can_be_switched_off(self):
        train = pd.DataFrame({"a": [1.0, np.nan]})
        policy = MissingnessPolicy(add_indicators=False).fit(train)

        out = policy.transform(train)

        self.assertNotIn("a__missing", out.columns)

    def test_transform_output_has_no_nan_left(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})
        policy = MissingnessPolicy().fit(train)

        out = policy.transform(train)

        self.assertFalse(out.isna().any().any())


class NoLeakageTests(unittest.TestCase):
    def test_imputation_uses_the_training_median_not_the_test_values(self):
        train = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0]})
        test = pd.DataFrame({"a": [100.0, np.nan]})
        policy = MissingnessPolicy().fit(train)

        out = policy.transform(test)

        # The train median is 1.0. Filling from the test column itself would
        # give 100.0 and quietly carry test information into the feature.
        self.assertAlmostEqual(out.loc[1, "a"], 1.0)

    def test_refitting_on_test_data_is_not_required_to_transform(self):
        train = pd.DataFrame({"a": [2.0, 4.0]})
        policy = MissingnessPolicy().fit(train)

        first = policy.transform(pd.DataFrame({"a": [np.nan]}))
        second = policy.transform(pd.DataFrame({"a": [np.nan]}))

        self.assertAlmostEqual(first.loc[0, "a"], second.loc[0, "a"])

    def test_transform_before_fit_is_refused(self):
        with self.assertRaises(RuntimeError):
            MissingnessPolicy().transform(pd.DataFrame({"a": [1.0]}))


class AbsentColumnTests(unittest.TestCase):
    def test_a_column_absent_at_transform_time_is_fully_missing(self):
        train = pd.DataFrame({"a": [1.0, 3.0], "b": [5.0, 7.0]})
        policy = MissingnessPolicy(add_indicators=True).fit(train)

        out = policy.transform(pd.DataFrame({"a": [2.0, 2.0]}))

        self.assertEqual(list(out["b__missing"]), [1.0, 1.0])
        self.assertTrue((out["b"] == 6.0).all())  # median of the training b column

    def test_a_column_that_was_never_observed_in_training_is_refused(self):
        train = pd.DataFrame({"a": [np.nan, np.nan]})

        with self.assertRaises(ValueError) as ctx:
            MissingnessPolicy().fit(train)

        self.assertIn("a", str(ctx.exception))

    def test_column_order_is_stable(self):
        train = pd.DataFrame({"b": [1.0], "a": [2.0]})
        policy = MissingnessPolicy(add_indicators=False).fit(train)

        out = policy.transform(pd.DataFrame({"a": [9.0], "b": [8.0]}))

        self.assertEqual(list(out.columns), ["b", "a"])


if __name__ == "__main__":
    unittest.main()
