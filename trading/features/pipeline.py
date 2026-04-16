from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.feature_engineering import FeatureRow, build_feature_row, sanitize_feature_frame
from core.indicators import compute_indicators
from core.market_regime import detect_market_regime
from core.volume_profile import compute_volume_profile
from trading.features.validators import assert_finite_features, assert_monotonic_time, assert_no_future_rows


@dataclass
class FeatureBundle:
    symbol: str
    as_of: pd.Timestamp
    enriched: pd.DataFrame
    row: FeatureRow


class FeaturePipeline:
    def __init__(self, profile_window: int = 120, profile_bins: int = 48):
        self.profile_window = int(profile_window)
        self.profile_bins = int(profile_bins)

    def build(self, symbol: str, ohlcv: pd.DataFrame, *, as_of: pd.Timestamp, extras: dict | None = None) -> FeatureBundle:
        assert_monotonic_time(ohlcv)
        hist = ohlcv.loc[:as_of].copy()
        assert_no_future_rows(hist, as_of)
        if len(hist) < 80:
            raise ValueError("insufficient_history")

        enriched = sanitize_feature_frame(compute_indicators(hist))
        vp = compute_volume_profile(enriched, window=self.profile_window, bins=self.profile_bins)
        regime = detect_market_regime(enriched)

        row = build_feature_row(
            symbol=symbol,
            df=enriched,
            volume_profile=vp,
            regime=regime,
            extras=extras or {},
        )
        if row is None:
            raise ValueError("feature_row_none")

        assert_finite_features(row.values)
        return FeatureBundle(symbol=symbol, as_of=as_of, enriched=enriched, row=row)
