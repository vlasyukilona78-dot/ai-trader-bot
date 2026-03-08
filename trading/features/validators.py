from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureValidationError(ValueError):
    pass


def assert_monotonic_time(df: pd.DataFrame):
    if df.empty:
        raise FeatureValidationError("empty_input")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise FeatureValidationError("datetime_index_required")
    if not df.index.is_monotonic_increasing:
        raise FeatureValidationError("non_monotonic_index")


def assert_no_future_rows(df: pd.DataFrame, as_of: pd.Timestamp):
    if (df.index > as_of).any():
        raise FeatureValidationError("future_rows_detected")


def assert_finite_features(values: dict[str, float]):
    bad = [k for k, v in values.items() if not np.isfinite(float(v))]
    if bad:
        raise FeatureValidationError(f"non_finite_features:{','.join(sorted(bad))}")
