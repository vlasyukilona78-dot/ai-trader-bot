from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.feature_engineering import FeatureRow, assess_feature_frame_quality, build_feature_row, sanitize_feature_frame
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
        try:
            enriched.attrs.update(getattr(hist, "attrs", {}) or {})
        except Exception:
            pass
        quality = assess_feature_frame_quality(enriched)
        if not bool(quality.get("usable", True)):
            reason = str(quality.get("reason") or "blocked")
            raise ValueError(f"feature_frame_quality:{reason}")
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

        self._bind_row_snapshot_to_latest_frame(enriched, row)
        assert_finite_features(row.values)
        return FeatureBundle(symbol=symbol, as_of=as_of, enriched=enriched, row=row)

    @staticmethod
    def _bind_row_snapshot_to_latest_frame(enriched: pd.DataFrame, row: FeatureRow) -> None:
        if enriched.empty:
            return
        latest_idx = enriched.index[-1]
        for key, value in row.values.items():
            if not key.startswith("mtf_"):
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if key not in enriched.columns:
                enriched[key] = 50.0 if key.startswith("mtf_rsi_") else 0.0
            enriched.loc[latest_idx, key] = numeric_value
