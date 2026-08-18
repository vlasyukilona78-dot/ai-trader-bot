"""Chronological dataset splitting with purge and embargo.

Rows produced by ``ai/build_dataset.py`` are chronological, one per bar, and
``target_horizon`` records how many bars ahead that row's trade resolved. A row
at position ``i`` is therefore only known at bar ``i + horizon``. Splitting on
position alone lets a training row whose trade resolves after the boundary leak
its outcome into the next interval, which inflates every metric measured there.

``temporal_split_3`` cuts three disjoint chronological intervals and drops the
rows that would leak across each boundary. The last interval is never purged:
nothing follows it, so it cannot leak forward.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


class SplitError(ValueError):
    """The requested split is invalid or was emptied by purging."""


@dataclass(frozen=True)
class ChronologicalSplit:
    """Positional row indices for three disjoint chronological intervals."""

    train_idx: tuple[int, ...]
    calib_idx: tuple[int, ...]
    test_idx: tuple[int, ...]
    purged_train: int
    purged_calib: int
    embargo: int


def _keep(index: int, horizon: float, boundary: int, embargo: int) -> bool:
    """Whether a row's label resolves strictly before the guarded boundary."""

    if not math.isfinite(horizon):
        return False
    return index + horizon < boundary - embargo


def temporal_split_3(
    *,
    n: int,
    horizons: Sequence[float],
    train_frac: float = 0.70,
    calib_frac: float = 0.15,
    embargo: int = 0,
    min_rows_per_interval: int = 10,
) -> ChronologicalSplit:
    """Split ``n`` chronological rows into train, calibration and test.

    Args:
        n: Number of rows, which must already be in chronological order.
        horizons: Per-row label horizon in bars; ``horizons[i]`` is the number
            of bars after row ``i`` at which its outcome became known.
        train_frac: Share of rows in the training interval.
        calib_frac: Share of rows in the calibration interval. The remainder
            becomes the test interval.
        embargo: Extra bars of separation required beyond the label horizon.
        min_rows_per_interval: Smallest interval considered usable.

    Raises:
        SplitError: The inputs are inconsistent, the fractions leave an
            interval empty, or purging emptied train or calibration.
    """

    if len(horizons) != n:
        raise SplitError(f"horizons has {len(horizons)} entries for {n} rows")
    if embargo < 0:
        raise SplitError("embargo must not be negative")
    if train_frac <= 0 or calib_frac <= 0:
        raise SplitError("train_frac and calib_frac must be positive")
    if train_frac + calib_frac >= 1.0:
        raise SplitError(
            f"train_frac + calib_frac = {train_frac + calib_frac} leaves no test interval"
        )

    calib_start = int(n * train_frac)
    test_start = int(n * (train_frac + calib_frac))

    sizes = {
        "train": calib_start,
        "calibration": test_start - calib_start,
        "test": n - test_start,
    }
    for name, size in sizes.items():
        if size < min_rows_per_interval:
            raise SplitError(
                f"{name} interval has {size} rows, below the minimum of {min_rows_per_interval}"
            )

    train_idx = tuple(
        i for i in range(calib_start) if _keep(i, horizons[i], calib_start, embargo)
    )
    calib_idx = tuple(
        i
        for i in range(calib_start, test_start)
        if _keep(i, horizons[i], test_start, embargo)
    )
    test_idx = tuple(range(test_start, n))

    if not train_idx:
        raise SplitError("purge and embargo emptied the train interval")
    if not calib_idx:
        raise SplitError("purge and embargo emptied the calibration interval")

    return ChronologicalSplit(
        train_idx=train_idx,
        calib_idx=calib_idx,
        test_idx=test_idx,
        purged_train=calib_start - len(train_idx),
        purged_calib=(test_start - calib_start) - len(calib_idx),
        embargo=embargo,
    )
