"""Handling absent feature values without pretending they are observations.

Two rules:

* Missingness is preserved as information. Each feature gets a companion
  indicator column so the model can separate "measured as zero" from "not
  measured", instead of being handed a single number that means both.
* Imputation statistics come from the training rows only. A median computed
  over the whole dataset carries later information back into training, which is
  the same leak that a careless split introduces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INDICATOR_SUFFIX = "__missing"


def missing_report(frame: pd.DataFrame) -> dict[str, float]:
    """Fraction of rows that are absent, per column."""

    if len(frame) == 0:
        return {str(column): 0.0 for column in frame.columns}
    return {
        str(column): float(frame[column].isna().sum()) / len(frame)
        for column in frame.columns
    }


class MissingnessPolicy:
    """Fit imputation on training rows, then apply it everywhere unchanged."""

    def __init__(self, *, add_indicators: bool = True) -> None:
        self.add_indicators = add_indicators
        self._columns: list[str] | None = None
        self._fill: dict[str, float] = {}
        self._train_missing_rate: dict[str, float] = {}

    @property
    def fitted(self) -> bool:
        return self._columns is not None

    @property
    def train_missing_rate(self) -> dict[str, float]:
        """Per-feature missing rate observed while fitting."""

        return dict(self._train_missing_rate)

    def fit(self, frame: pd.DataFrame) -> "MissingnessPolicy":
        """Learn the fill value for each column from training rows only.

        Raises:
            ValueError: A column has no observed value at all, so no honest
                fill exists for it.
        """

        columns = [str(c) for c in frame.columns]
        fill: dict[str, float] = {}
        empty: list[str] = []

        for column in columns:
            series = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if series.notna().sum() == 0:
                empty.append(column)
                continue
            fill[column] = float(series.median())

        if empty:
            raise ValueError(
                "no value was ever observed for: "
                + ", ".join(sorted(empty))
                + "; a fill value cannot be invented for a feature with no data"
            )

        self._columns = columns
        self._fill = fill
        self._train_missing_rate = missing_report(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted fills and attach missingness indicators.

        Raises:
            RuntimeError: The policy has not been fitted.
        """

        if self._columns is None:
            raise RuntimeError("MissingnessPolicy.fit must be called before transform")

        values: dict[str, pd.Series] = {}
        indicators: dict[str, pd.Series] = {}

        for column in self._columns:
            if column in frame.columns:
                series = pd.to_numeric(frame[column], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
            else:
                # The column did not arrive at all. That is total missingness,
                # not a column of zeros.
                series = pd.Series(np.nan, index=frame.index, dtype=float)

            absent = series.isna()
            values[column] = series.fillna(self._fill[column]).astype(float)
            if self.add_indicators:
                indicators[column + INDICATOR_SUFFIX] = absent.astype(float)

        out = pd.DataFrame(values, index=frame.index)
        if self.add_indicators:
            out = pd.concat([out, pd.DataFrame(indicators, index=frame.index)], axis=1)
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)
