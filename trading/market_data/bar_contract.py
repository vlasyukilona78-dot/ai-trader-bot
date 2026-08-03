from __future__ import annotations

from numbers import Real

import pandas as pd


class BarContractError(ValueError):
    """Raised when candle data cannot satisfy the closed-bar contract."""


_FIXED_INTERVAL_SECONDS = {
    "1": 60,
    "5": 5 * 60,
    "15": 15 * 60,
    "30": 30 * 60,
    "60": 60 * 60,
    "240": 4 * 60 * 60,
    "480": 8 * 60 * 60,
    "d": 24 * 60 * 60,
    "w": 7 * 24 * 60 * 60,
    "min1": 60,
    "min5": 5 * 60,
    "min15": 15 * 60,
    "min30": 30 * 60,
    "min60": 60 * 60,
    "hour4": 4 * 60 * 60,
    "hour8": 8 * 60 * 60,
    "day1": 24 * 60 * 60,
    "week1": 7 * 24 * 60 * 60,
}


def interval_seconds(interval: str) -> int:
    """Return a fixed candle duration, rejecting calendar-dependent intervals."""

    key = str(interval).strip().lower()
    try:
        return _FIXED_INTERVAL_SECONDS[key]
    except KeyError as exc:
        raise BarContractError(f"unsupported or non-fixed candle interval: {interval!r}") from exc


def _as_utc_timestamp(as_of) -> pd.Timestamp:
    if isinstance(as_of, bool):
        raise BarContractError("as_of must be an epoch second or timezone-aware timestamp")
    if isinstance(as_of, Real):
        try:
            timestamp = pd.Timestamp(float(as_of), unit="s", tz="UTC")
        except (TypeError, ValueError, OverflowError) as exc:
            raise BarContractError(f"invalid as_of epoch: {as_of!r}") from exc
    else:
        try:
            timestamp = pd.Timestamp(as_of)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BarContractError(f"invalid as_of timestamp: {as_of!r}") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise BarContractError("as_of must be timezone-aware")
        timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise BarContractError("as_of must not be NaT")
    return timestamp


def closed_boundary_ts(as_of, interval: str) -> float:
    """Return the latest UTC boundary at or before the explicit decision time."""

    seconds = interval_seconds(interval)
    timestamp = _as_utc_timestamp(as_of)
    boundary_ns = (timestamp.value // (seconds * 1_000_000_000)) * seconds * 1_000_000_000
    return float(pd.Timestamp(boundary_ns, tz="UTC").timestamp())


def next_bar_open_ts(as_of, interval: str) -> float:
    """Return the first bar open strictly after the explicit decision time.

    A decision known at a bar's open cannot be filled at that open: the price was
    already printing while the decision was still being computed. Flooring to the
    boundary and adding one interval keeps the result strictly greater even when
    ``as_of`` lands exactly on a boundary, which is the case that would otherwise
    buy a bar the decision did not precede.
    """

    seconds = interval_seconds(interval)
    return closed_boundary_ts(as_of, interval) + float(seconds)


def is_bar_aligned(timestamp, interval: str) -> bool:
    """Whether a timestamp sits exactly on a bar boundary for this interval."""

    seconds = interval_seconds(interval)
    value = _as_utc_timestamp(timestamp).value
    return value % (seconds * 1_000_000_000) == 0


def last_bar_times(frame: pd.DataFrame, *, interval: str) -> tuple[float, float]:
    """Return UTC epoch seconds for the final bar's open and contractual close."""

    if not isinstance(frame, pd.DataFrame):
        raise BarContractError("candle frame must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise BarContractError("candle frame must use a DatetimeIndex of bar opens")
    if frame.index.tz is None:
        raise BarContractError("candle DatetimeIndex must be timezone-aware")
    if frame.empty:
        raise BarContractError("candle frame must not be empty")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise BarContractError("candle DatetimeIndex must be ordered and unique")
    last_open = frame.index[-1].tz_convert("UTC")
    last_close = last_open + pd.Timedelta(seconds=interval_seconds(interval))
    return float(last_open.timestamp()), float(last_close.timestamp())


def retain_closed_bars(
    frame: pd.DataFrame,
    *,
    interval: str,
    as_of,
) -> pd.DataFrame:
    """Return only bars whose ``open + interval <= as_of``.

    The input index is the candle-open timestamp. Both the decision time and the
    index must carry an explicit timezone so a local-machine setting cannot move
    the causal cutoff.
    """

    if not isinstance(frame, pd.DataFrame):
        raise BarContractError("candle frame must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise BarContractError("candle frame must use a DatetimeIndex of bar opens")
    if frame.index.tz is None:
        raise BarContractError("candle DatetimeIndex must be timezone-aware")
    if not frame.index.is_monotonic_increasing:
        raise BarContractError("candle DatetimeIndex must be monotonic increasing")
    if frame.index.has_duplicates:
        raise BarContractError("candle DatetimeIndex must not contain duplicate bar opens")

    seconds = interval_seconds(interval)
    cutoff_epoch = closed_boundary_ts(as_of, interval)
    cutoff = pd.Timestamp(cutoff_epoch, unit="s", tz="UTC")
    closes = frame.index.tz_convert("UTC") + pd.Timedelta(seconds=seconds)
    result = frame.loc[closes <= cutoff].copy()
    original_attrs = dict(frame.attrs)
    result.attrs.update(original_attrs)

    last_open_ts: float | None = None
    last_close_ts: float | None = None
    if not result.empty:
        last_open_ts, last_close_ts = last_bar_times(result, interval=interval)

    result.attrs.update(
        {
            "bar_interval": interval,
            "bar_interval_seconds": seconds,
            "candle_cutoff_ts": cutoff_epoch,
            "last_bar_open_ts": last_open_ts,
            "last_bar_close_ts": last_close_ts,
        }
    )
    return result
