"""Measure how far price runs each way after an event, and which way it goes first.

This is the stand behind the 2026-08-18 finding that the pump condition is a
volatility filter rather than a direction filter. It exists as a module rather
than a script because the same measurement has to be repeated against Min1
history if the QA pilot ever collects any, and because the properties that make
its numbers trustworthy are worth pinning in tests.

Three of those properties matter more than the arithmetic:

* The decision is taken at the close of bar ``t`` and the position is entered at
  the open of ``t+1``. Nothing after the decision instant reaches the trigger.
* Observation windows do not overlap. Overlapping windows re-measure one move
  many times, which inflates the sample count and shrinks every interval.
* Where one bar contains both the stop and the target, the bar cannot say which
  came first, so the stop is charged. That is the only convention that cannot
  flatter a result.

Nothing here trades, calibrates a threshold or selects a parameter. It reports
distributions; choosing a stop from them is a separate, explicit decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OHLC = ("open", "high", "low", "close")


@dataclass(frozen=True)
class EventSpec:
    """How an event to fade is recognised, in bars of the frame being scanned."""

    lookback_bars: int = 24
    min_run_up: float = 0.07
    peak_within_bars: int = 6
    min_fade_fraction: float = 0.20
    warmup_bars: int = 200

    def __post_init__(self) -> None:
        if self.lookback_bars < 2:
            raise ValueError("lookback_bars must span at least two bars")
        if not 0.0 <= self.min_fade_fraction <= 1.0:
            raise ValueError("min_fade_fraction is a fraction of the run")
        if self.peak_within_bars < 0 or self.peak_within_bars >= self.lookback_bars:
            raise ValueError("peak_within_bars must fall inside the lookback window")


@dataclass(frozen=True)
class Excursion:
    """Both extremes reached over the horizon, as fractions of the entry price."""

    adverse: float
    favourable: float


def _checked(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in OHLC if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    return frame


def event_indices(frame: pd.DataFrame, spec: EventSpec, horizon: int) -> np.ndarray:
    """Bars where a run-up has fired and price has turned off the peak.

    Returned indices are non-overlapping: the next event starts only after the
    previous horizon has elapsed.
    """

    _checked(frame)
    if horizon < 1:
        raise ValueError("horizon must be at least one bar")

    high, low, close = frame["high"], frame["low"], frame["close"]
    window_high = high.rolling(spec.lookback_bars).max()
    window_low = low.rolling(spec.lookback_bars).min()
    span = (window_high - window_low).clip(lower=1e-12)

    run_up = (window_high - window_low) / window_low.clip(lower=1e-12)
    faded = (window_high - close) / span
    bars_since_peak = high.rolling(spec.lookback_bars).apply(
        lambda w: len(w) - 1 - int(np.argmax(w)), raw=True
    )

    qualifies = (
        (run_up >= spec.min_run_up)
        & (bars_since_peak <= spec.peak_within_bars)
        & (faded >= spec.min_fade_fraction)
    ).to_numpy()

    # An entry needs bar t+1 to open on and horizon bars after it to observe.
    last_usable = len(frame) - horizon - 1
    start = max(spec.warmup_bars, spec.lookback_bars)
    chosen: list[int] = []
    blocked_until = start
    for index in range(start, last_usable):
        if index < blocked_until or not qualifies[index]:
            continue
        chosen.append(index)
        blocked_until = index + horizon
    return np.asarray(chosen, dtype=int)


def excursion_at(frame: pd.DataFrame, index: int, horizon: int) -> Excursion:
    """Extremes for a short decided at ``index`` and entered at ``index + 1``."""

    _checked(frame)
    entry = float(frame["open"].iloc[index + 1])
    if entry <= 0.0:
        raise ValueError("entry price must be positive")
    window = slice(index + 1, index + 1 + horizon)
    highs = frame["high"].iloc[window].to_numpy()
    lows = frame["low"].iloc[window].to_numpy()
    if highs.size == 0:
        raise ValueError("horizon extends past the end of the frame")
    return Excursion(
        adverse=float(highs.max() / entry - 1.0),
        favourable=float(1.0 - lows.min() / entry),
    )


def first_touch_return(
    frame: pd.DataFrame,
    index: int,
    *,
    stop_pct: float,
    target_pct: float,
    horizon: int,
    cost: float = 0.00217,
) -> float:
    """Net return of a short that exits at whichever level is reached first.

    A bar holding both levels is charged the stop. Anything unresolved by the end
    of the horizon is closed at that bar's close, which is what a time-boxed
    manual exit does.
    """

    _checked(frame)
    if stop_pct <= 0.0 or target_pct <= 0.0:
        raise ValueError("stop and target distances must be positive")
    entry = float(frame["open"].iloc[index + 1])
    window = slice(index + 1, index + 1 + horizon)
    highs = frame["high"].iloc[window].to_numpy()
    lows = frame["low"].iloc[window].to_numpy()
    closes = frame["close"].iloc[window].to_numpy()
    if closes.size == 0:
        raise ValueError("horizon extends past the end of the frame")

    stop_hits = np.flatnonzero(highs >= entry * (1.0 + stop_pct / 100.0))
    target_hits = np.flatnonzero(lows <= entry * (1.0 - target_pct / 100.0))
    first_stop = int(stop_hits[0]) if stop_hits.size else len(highs)
    first_target = int(target_hits[0]) if target_hits.size else len(lows)

    if first_stop <= first_target and first_stop < len(highs):
        gross = -stop_pct / 100.0
    elif first_target < len(lows):
        gross = target_pct / 100.0
    else:
        gross = (entry - float(closes[-1])) / entry
    return gross - cost


def matched_random_indices(
    frame: pd.DataFrame,
    count: int,
    *,
    spec: EventSpec,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Control entries drawn from the same symbol over the same usable range.

    A signal that cannot beat this is not carrying information, however good its
    win rate looks.
    """

    start = max(spec.warmup_bars, spec.lookback_bars)
    pool = np.arange(start, len(frame) - horizon - 1)
    if pool.size == 0 or count <= 0:
        return np.asarray([], dtype=int)
    picks = rng.choice(pool, size=min(count, pool.size), replace=False)
    return np.sort(picks)
