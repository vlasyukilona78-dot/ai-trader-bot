from __future__ import annotations

import pandas as pd

from core.indicators import compute_indicators, cvd, obv, vwap


def test_cumulative_indicator_modes_match_the_v1_golden_vector() -> None:
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 10.0, 12.0],
            "high": [11.0, 13.0, 12.0, 13.0],
            "low": [9.0, 10.0, 9.0, 10.0],
            "close": [10.0, 12.0, 10.0, 10.0],
            "volume": [1.0, 2.0, 4.0, 8.0],
        },
        index=pd.date_range("2026-08-08T00:00:00Z", periods=4, freq="h"),
    )
    expected_vwap = pd.Series(
        [10.0, 100.0 / 9.0, 224.0 / 21.0, 488.0 / 45.0],
        index=frame.index,
    )
    expected_obv = pd.Series([0.0, 2.0, -2.0, -2.0], index=frame.index)
    expected_cvd = pd.Series([1.0, 3.0, 7.0, -1.0], index=frame.index)

    pd.testing.assert_series_equal(vwap(frame), expected_vwap)
    pd.testing.assert_series_equal(obv(frame), expected_obv)
    pd.testing.assert_series_equal(cvd(frame), expected_cvd)

    enriched = compute_indicators(frame)
    pd.testing.assert_series_equal(enriched["vwap"], expected_vwap, check_names=False)
    pd.testing.assert_series_equal(enriched["obv"], expected_obv, check_names=False)
    pd.testing.assert_series_equal(enriched["cvd"], expected_cvd, check_names=False)
