"""Chart-structure features: horizontal levels, fib grid, multi-timeframe
confluence, RSI divergence and an estimated liquidation map.

These encode the techniques used manually in the reference channel. Each is
exposed as a plain number so it can be measured against outcomes on the labelled
dataset before any of it is allowed to gate a live signal - crude proxies for
these ideas have already flipped sign twice on small samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Leverage tiers retail actually uses on MEXC alts, with rough weights. Used to
# project where long liquidations sit below the price they entered at.
DEFAULT_LEVERAGE_TIERS: tuple[tuple[float, float], ...] = (
    (10.0, 0.15),
    (20.0, 0.30),
    (50.0, 0.35),
    (100.0, 0.20),
)


@dataclass
class HorizontalLevel:
    price: float
    touches: int
    volume: float
    first_seen_bars_ago: int
    last_touch_bars_ago: int

    @property
    def strength(self) -> float:
        """More touches, older, and heavier volume all make a level matter more."""
        age = np.log1p(max(self.first_seen_bars_ago, 0))
        untouched = np.log1p(max(self.last_touch_bars_ago, 0))
        return float(self.touches * (1.0 + 0.3 * age) * (1.0 + 0.2 * untouched))


def _pivots(series: pd.Series, left: int, right: int, kind: str) -> list[int]:
    values = series.to_numpy(dtype=float)
    out: list[int] = []
    for i in range(left, len(values) - right):
        window = values[i - left : i + right + 1]
        if kind == "high" and values[i] >= window.max():
            out.append(i)
        elif kind == "low" and values[i] <= window.min():
            out.append(i)
    return out


def find_horizontal_levels(
    df: pd.DataFrame,
    *,
    pivot_left: int = 3,
    pivot_right: int = 3,
    cluster_tol_pct: float = 0.006,
    min_touches: int = 2,
) -> list[HorizontalLevel]:
    """Cluster pivot highs/lows into horizontal levels, keeping how many times
    price tested each one and how long it has been left alone."""
    if df.empty or len(df) < (pivot_left + pivot_right + 5):
        return []

    high = pd.to_numeric(df["high"], errors="coerce").reset_index(drop=True)
    low = pd.to_numeric(df["low"], errors="coerce").reset_index(drop=True)
    vol = pd.to_numeric(df.get("volume", pd.Series(0.0, index=df.index)), errors="coerce").reset_index(drop=True)
    n = len(high)

    candidates: list[tuple[float, int]] = []
    for idx in _pivots(high, pivot_left, pivot_right, "high"):
        candidates.append((float(high.iloc[idx]), idx))
    for idx in _pivots(low, pivot_left, pivot_right, "low"):
        candidates.append((float(low.iloc[idx]), idx))
    if not candidates:
        return []

    candidates.sort(key=lambda c: c[0])
    clusters: list[list[tuple[float, int]]] = [[candidates[0]]]
    for price, idx in candidates[1:]:
        ref = clusters[-1][0][0]
        if ref > 0 and abs(price - ref) / ref <= cluster_tol_pct:
            clusters[-1].append((price, idx))
        else:
            clusters.append([(price, idx)])

    levels: list[HorizontalLevel] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        prices = [p for p, _ in cluster]
        idxs = [i for _, i in cluster]
        level_price = float(np.mean(prices))
        tol = level_price * cluster_tol_pct
        near = ((low <= level_price + tol) & (high >= level_price - tol))
        levels.append(
            HorizontalLevel(
                price=level_price,
                touches=len(cluster),
                volume=float(vol[near].sum()),
                first_seen_bars_ago=int(n - 1 - min(idxs)),
                last_touch_bars_ago=int(n - 1 - max(idxs)),
            )
        )
    return levels


def nearest_level_above(levels: list[HorizontalLevel], price: float, *, max_dist_pct: float = 0.15):
    """Closest resistance overhead - the thing a pump is supposed to stall into."""
    above = [lv for lv in levels if lv.price > price and (lv.price - price) / price <= max_dist_pct]
    if not above:
        return None
    return min(above, key=lambda lv: lv.price - price)


def fib_levels(df: pd.DataFrame, *, lookback: int = 240) -> dict[str, float]:
    """Retracement grid of the dominant swing inside the lookback window."""
    if df.empty or len(df) < 10:
        return {}
    window = df.tail(lookback)
    hi = float(pd.to_numeric(window["high"], errors="coerce").max())
    lo = float(pd.to_numeric(window["low"], errors="coerce").min())
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        return {}
    span = hi - lo
    return {f"fib_{int(r*1000)}": lo + r * span for r in (0.236, 0.382, 0.5, 0.618, 0.786)}


def distance_to_fib(df: pd.DataFrame, price: float, *, ratio: str = "fib_618", lookback: int = 240) -> float:
    grid = fib_levels(df, lookback=lookback)
    target = grid.get(ratio)
    if not target or price <= 0:
        return float("nan")
    return abs(price - target) / price


def multi_timeframe_confluence(
    frames: dict[str, pd.DataFrame],
    price: float,
    *,
    tol_pct: float = 0.01,
    max_dist_pct: float = 0.05,
) -> dict[str, float]:
    """How many independent timeframes place a level near the current price.

    A level that only one timeframe sees is noise; the same price showing up on
    15m, 1h and 4h is what the reference channel calls a confluence zone.
    """
    hits = 0
    strength = 0.0
    nearest = float("nan")
    for _, frame in frames.items():
        if frame is None or frame.empty:
            continue
        levels = find_horizontal_levels(frame)
        overhead = [lv for lv in levels if lv.price >= price and (lv.price - price) / price <= max_dist_pct]
        if not overhead:
            continue
        closest = min(overhead, key=lambda lv: lv.price - price)
        dist = (closest.price - price) / price
        if dist <= tol_pct:
            hits += 1
            strength += closest.strength
        if not np.isfinite(nearest) or dist < nearest:
            nearest = dist
    return {
        "confluence_count": float(hits),
        "confluence_strength": float(strength),
        "nearest_overhead_dist": float(nearest),
    }


def rsi_divergence(df: pd.DataFrame, *, lookback: int = 20, pivot: int = 3) -> dict[str, float]:
    """Bearish divergence: price prints a higher high while RSI prints a lower one."""
    if df.empty or "rsi" not in df.columns or len(df) < lookback + pivot * 2 + 2:
        return {"bearish_divergence": 0.0, "divergence_gap": 0.0}

    window = df.tail(lookback + pivot * 2)
    high = pd.to_numeric(window["high"], errors="coerce").reset_index(drop=True)
    rsi = pd.to_numeric(window["rsi"], errors="coerce").reset_index(drop=True)

    peaks = _pivots(high, pivot, pivot, "high")
    if len(peaks) < 2:
        return {"bearish_divergence": 0.0, "divergence_gap": 0.0}

    last, prev = peaks[-1], peaks[-2]
    price_hh = high.iloc[last] > high.iloc[prev]
    rsi_lh = rsi.iloc[last] < rsi.iloc[prev]
    gap = float(rsi.iloc[prev] - rsi.iloc[last])
    return {
        "bearish_divergence": 1.0 if (price_hh and rsi_lh) else 0.0,
        "divergence_gap": gap if (price_hh and rsi_lh) else 0.0,
    }


def liquidation_histogram(
    df: pd.DataFrame,
    *,
    lookback: int = 240,
    bins: int = 48,
    tiers: tuple[tuple[float, float], ...] = DEFAULT_LEVERAGE_TIERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Binned estimated long-liquidation density by price, for plotting.

    Returns (bin_centres, weights); weights are normalised to a 0-1 scale so the
    shape can be drawn without exposing raw volume units.
    """
    empty = (np.array([]), np.array([]))
    if df.empty or len(df) < 20:
        return empty

    window = df.tail(lookback)
    typical = ((pd.to_numeric(window["high"], errors="coerce")
                + pd.to_numeric(window["low"], errors="coerce")
                + pd.to_numeric(window["close"], errors="coerce")) / 3.0).to_numpy(dtype=float)
    volume = pd.to_numeric(window["volume"], errors="coerce").to_numpy(dtype=float)

    prices: list[float] = []
    weights: list[float] = []
    for entry_px, vol in zip(typical, volume):
        if not np.isfinite(entry_px) or not np.isfinite(vol) or vol <= 0 or entry_px <= 0:
            continue
        for lev, weight in tiers:
            prices.append(entry_px * (1.0 - 1.0 / lev))
            weights.append(vol * weight)

    if not prices:
        return empty

    hist, edges = np.histogram(np.asarray(prices), bins=bins, weights=np.asarray(weights))
    if hist.max() <= 0:
        return empty
    centres = (edges[:-1] + edges[1:]) / 2.0
    return centres, hist / hist.max()


def estimate_liquidation_map(
    df: pd.DataFrame,
    price: float,
    *,
    lookback: int = 240,
    bins: int = 60,
    tiers: tuple[tuple[float, float], ...] = DEFAULT_LEVERAGE_TIERS,
    zone_pct: float = 0.10,
) -> dict[str, float]:
    """Estimate where long liquidations sit below price.

    MEXC publishes no liquidation feed, so this infers it: volume that traded at
    a given price is treated as longs opened there, and each leverage tier puts
    their liquidation a predictable distance below that entry. Heavy estimated
    liquidation volume just under the market is the fuel that turns a fading
    pump into a cascade - which is the whole premise of the strategy.
    """
    if df.empty or len(df) < 20 or price <= 0:
        return {"liq_below_pct": 0.0, "liq_nearest_dist": float("nan"), "liq_peak_dist": float("nan")}

    window = df.tail(lookback)
    typical = ((pd.to_numeric(window["high"], errors="coerce")
                + pd.to_numeric(window["low"], errors="coerce")
                + pd.to_numeric(window["close"], errors="coerce")) / 3.0).to_numpy(dtype=float)
    volume = pd.to_numeric(window["volume"], errors="coerce").to_numpy(dtype=float)

    liq_prices: list[float] = []
    liq_weights: list[float] = []
    for entry_px, vol in zip(typical, volume):
        if not np.isfinite(entry_px) or not np.isfinite(vol) or vol <= 0 or entry_px <= 0:
            continue
        for lev, weight in tiers:
            liq_prices.append(entry_px * (1.0 - 1.0 / lev))
            liq_weights.append(vol * weight)

    if not liq_prices:
        return {"liq_below_pct": 0.0, "liq_nearest_dist": float("nan"), "liq_peak_dist": float("nan")}

    liq = np.asarray(liq_prices)
    wts = np.asarray(liq_weights)
    total = float(wts.sum())

    zone_low = price * (1.0 - zone_pct)
    in_zone = (liq < price) & (liq >= zone_low)
    below_share = float(wts[in_zone].sum() / total) if total > 0 else 0.0

    nearest = float("nan")
    peak_dist = float("nan")
    if in_zone.any():
        nearest = float((price - liq[in_zone].max()) / price)
        hist, edges = np.histogram(liq[in_zone], bins=bins, weights=wts[in_zone])
        if hist.sum() > 0:
            centre = (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2.0
            peak_dist = float((price - centre) / price)

    return {
        "liq_below_pct": below_share,
        "liq_nearest_dist": nearest,
        "liq_peak_dist": peak_dist,
    }
