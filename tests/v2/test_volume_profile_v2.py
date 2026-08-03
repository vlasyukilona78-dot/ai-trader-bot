from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.volume_profile import _contiguous_value_area_bounds, compute_volume_profile


class VolumeProfileV2Tests(unittest.TestCase):
    def test_value_area_expands_contiguously_from_poc(self):
        profile = np.array([0.0, 8.0, 10.0, 7.0, 0.0, 9.0], dtype=float)

        val_idx, vah_idx = _contiguous_value_area_bounds(
            profile,
            poc_idx=2,
            target_volume=24.0,
        )

        self.assertEqual((val_idx, vah_idx), (1, 3))

    def test_profile_levels_are_ordered_and_inside_traded_range(self):
        idx = pd.date_range("2026-01-01", periods=80, freq="min", tz="UTC")
        close = np.linspace(100.0, 108.0, 80)
        frame = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 1.0,
                "low": close - 1.2,
                "close": close,
                "volume": np.linspace(100.0, 300.0, 80),
            },
            index=idx,
        )

        levels = compute_volume_profile(frame, window=80, bins=32)

        self.assertIsNotNone(levels)
        self.assertLessEqual(levels.val, levels.poc)
        self.assertLessEqual(levels.poc, levels.vah)
        self.assertGreaterEqual(levels.val, float(frame["low"].min()))
        self.assertLessEqual(levels.vah, float(frame["high"].max()))


if __name__ == "__main__":
    unittest.main()
