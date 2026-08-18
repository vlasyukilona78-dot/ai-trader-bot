"""Chronological train/calibration/test split with purge and embargo.

The dataset built by ``ai/build_dataset.py`` stores one row per bar in
chronological order, and ``target_horizon`` is the number of bars ahead at
which that row's trade resolved. A row at position ``i`` therefore only
becomes known at bar ``i + horizon``. Any row whose label resolves inside a
later interval leaks future information into the earlier one, so it must be
purged before the split is used.
"""

from __future__ import annotations

import unittest

from ai.splitting import SplitError, temporal_split_3


class ThreeWayChronologicalSplitTests(unittest.TestCase):
    def test_intervals_are_ordered_and_disjoint(self):
        split = temporal_split_3(n=100, horizons=[0.0] * 100, embargo=0)

        self.assertGreater(len(split.train_idx), 0)
        self.assertGreater(len(split.calib_idx), 0)
        self.assertGreater(len(split.test_idx), 0)
        self.assertLess(max(split.train_idx), min(split.calib_idx))
        self.assertLess(max(split.calib_idx), min(split.test_idx))

    def test_default_fractions_are_seventy_fifteen_fifteen(self):
        split = temporal_split_3(n=100, horizons=[0.0] * 100, embargo=0)

        self.assertEqual(len(split.train_idx), 70)
        self.assertEqual(len(split.calib_idx), 15)
        self.assertEqual(len(split.test_idx), 15)


class PurgeTests(unittest.TestCase):
    def test_training_row_resolving_inside_calibration_is_purged(self):
        # Row 69 is the last training row; a horizon of 5 resolves it at bar 74,
        # which lies inside the calibration interval starting at 70.
        horizons = [0.0] * 100
        horizons[69] = 5.0

        split = temporal_split_3(n=100, horizons=horizons, embargo=0)

        self.assertNotIn(69, split.train_idx)
        self.assertEqual(split.purged_train, 1)

    def test_training_row_resolving_before_calibration_is_kept(self):
        # Row 60 with horizon 5 resolves at bar 65, still inside training.
        horizons = [0.0] * 100
        horizons[60] = 5.0

        split = temporal_split_3(n=100, horizons=horizons, embargo=0)

        self.assertIn(60, split.train_idx)
        self.assertEqual(split.purged_train, 0)

    def test_calibration_row_resolving_inside_test_is_purged(self):
        # Row 84 is the last calibration row; horizon 5 resolves it at bar 89,
        # inside the test interval starting at 85.
        horizons = [0.0] * 100
        horizons[84] = 5.0

        split = temporal_split_3(n=100, horizons=horizons, embargo=0)

        self.assertNotIn(84, split.calib_idx)
        self.assertEqual(split.purged_calib, 1)

    def test_test_interval_is_never_purged(self):
        # Nothing follows the test interval, so long horizons cannot leak.
        horizons = [0.0] * 100
        horizons[99] = 50.0

        split = temporal_split_3(n=100, horizons=horizons, embargo=0)

        self.assertIn(99, split.test_idx)
        self.assertEqual(len(split.test_idx), 15)


class EmbargoTests(unittest.TestCase):
    def test_embargo_purges_rows_that_resolve_before_the_boundary(self):
        # Row 68 resolves at bar 69, safely inside training with no embargo.
        # An embargo of 3 moves the cutoff back to 67, so it must be purged.
        horizons = [0.0] * 100
        horizons[68] = 1.0

        without = temporal_split_3(n=100, horizons=horizons, embargo=0)
        with_embargo = temporal_split_3(n=100, horizons=horizons, embargo=3)

        self.assertIn(68, without.train_idx)
        self.assertNotIn(68, with_embargo.train_idx)

    def test_embargo_is_reported_on_the_result(self):
        split = temporal_split_3(n=100, horizons=[0.0] * 100, embargo=7)

        self.assertEqual(split.embargo, 7)


class FailClosedTests(unittest.TestCase):
    def test_raises_when_purge_empties_the_training_interval(self):
        # Every training row resolves well inside calibration.
        horizons = [100.0] * 100

        with self.assertRaises(SplitError) as ctx:
            temporal_split_3(n=100, horizons=horizons, embargo=0)

        self.assertIn("train", str(ctx.exception).lower())

    def test_raises_when_dataset_is_too_small_to_split(self):
        with self.assertRaises(SplitError):
            temporal_split_3(n=5, horizons=[0.0] * 5, embargo=0)

    def test_raises_when_horizons_length_does_not_match(self):
        with self.assertRaises(SplitError):
            temporal_split_3(n=100, horizons=[0.0] * 99, embargo=0)

    def test_raises_on_negative_embargo(self):
        with self.assertRaises(SplitError):
            temporal_split_3(n=100, horizons=[0.0] * 100, embargo=-1)

    def test_raises_when_fractions_do_not_leave_a_test_interval(self):
        with self.assertRaises(SplitError):
            temporal_split_3(
                n=100, horizons=[0.0] * 100, train_frac=0.8, calib_frac=0.2, embargo=0
            )


if __name__ == "__main__":
    unittest.main()
