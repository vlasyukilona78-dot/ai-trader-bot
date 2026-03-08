from __future__ import annotations

import pandas as pd


class TrainingValidationError(ValueError):
    pass


def validate_no_feature_leakage(df: pd.DataFrame):
    if "timestamp" not in df.columns:
        raise TrainingValidationError("timestamp_column_required")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise TrainingValidationError("invalid_timestamps")
    if not ts.is_monotonic_increasing:
        raise TrainingValidationError("timestamps_not_monotonic")


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or (train_frac + val_frac >= 1):
        raise TrainingValidationError("invalid_split")

    n = len(df)
    if n < 100:
        raise TrainingValidationError("dataset_too_small")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
