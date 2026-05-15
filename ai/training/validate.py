from __future__ import annotations

import pandas as pd


class TrainingValidationError(ValueError):
    pass


ALLOWED_LABEL_COLUMNS = {
    "target_win",
    "target_horizon",
    "future_return",
    "signal_side",
    "signal_phase",
    "signal_family",
}
NON_FEATURE_COLUMNS = {
    "timestamp",
    "datetime",
    "time",
    "symbol",
    "market_regime",
}
FORBIDDEN_FEATURE_TOKENS = (
    "future",
    "target",
    "label",
    "outcome",
    "exit_",
    "closed_",
    "realized_pnl",
)


def validate_no_feature_leakage(df: pd.DataFrame) -> None:
    if "timestamp" not in df.columns:
        raise TrainingValidationError("timestamp_column_required")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise TrainingValidationError("invalid_timestamps")
    if not ts.is_monotonic_increasing:
        raise TrainingValidationError("timestamps_not_monotonic")

    missing_labels = {"target_win", "target_horizon"} - set(df.columns)
    if missing_labels:
        raise TrainingValidationError(f"missing_target_columns:{','.join(sorted(missing_labels))}")

    for col in df.columns:
        lower = str(col).strip().lower()
        if lower in ALLOWED_LABEL_COLUMNS or lower in NON_FEATURE_COLUMNS:
            continue
        if any(token in lower for token in FORBIDDEN_FEATURE_TOKENS):
            raise TrainingValidationError(f"feature_leakage_column:{col}")


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    purge_bars: int = 0,
):
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or (train_frac + val_frac >= 1):
        raise TrainingValidationError("invalid_split")
    if int(purge_bars) < 0:
        raise TrainingValidationError("invalid_purge_bars")

    n = len(df)
    if n < 100:
        raise TrainingValidationError("dataset_too_small")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    purge = int(purge_bars)
    train = df.iloc[: max(0, train_end - purge)]
    val = df.iloc[min(n, train_end + purge) : max(train_end + purge, val_end - purge)]
    test = df.iloc[min(n, val_end + purge) :]
    if train.empty or val.empty or test.empty:
        raise TrainingValidationError("split_empty_after_purge")
    return train, val, test
