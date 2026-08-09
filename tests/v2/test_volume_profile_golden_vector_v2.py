from __future__ import annotations

import pandas as pd

from core.volume_profile import VolumeProfileLevels, compute_volume_profile


def test_core_volume_profile_v1_matches_the_golden_levels() -> None:
    frame = pd.DataFrame(
        {
            "open": [90.0, 100.0, 101.0, 102.0, 103.0],
            "high": [90.0, 100.0, 101.0, 102.0, 103.0],
            "low": [90.0, 100.0, 101.0, 102.0, 103.0],
            "close": [90.0, 100.0, 101.0, 102.0, 103.0],
            "volume": [10_000.0, 10.0, 9.0, 8.0, 7.0],
        }
    )

    levels = compute_volume_profile(
        frame,
        window=4,
        bins=8,
        value_area=0.70,
        minimum_history_bars=1,
        minimum_sample_bars=1,
    )

    assert levels == VolumeProfileLevels(
        poc=100.1875,
        vah=102.0625,
        val=100.1875,
    )
