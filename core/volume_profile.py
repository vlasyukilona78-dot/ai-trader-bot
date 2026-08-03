from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolumeProfileLevels:
    poc: float
    vah: float
    val: float


def _contiguous_value_area_bounds(
    profile: np.ndarray,
    *,
    poc_idx: int,
    target_volume: float,
) -> tuple[int, int]:
    """Expand from POC through adjacent buckets until target volume is covered."""
    if profile.size == 0:
        return 0, 0
    low_idx = high_idx = max(0, min(int(poc_idx), int(profile.size) - 1))
    cumulative = float(profile[poc_idx])
    while cumulative < float(target_volume) and (low_idx > 0 or high_idx < profile.size - 1):
        left_volume = float(profile[low_idx - 1]) if low_idx > 0 else -1.0
        right_volume = float(profile[high_idx + 1]) if high_idx < profile.size - 1 else -1.0
        if right_volume > left_volume:
            high_idx += 1
            cumulative += max(0.0, right_volume)
        else:
            low_idx -= 1
            cumulative += max(0.0, left_volume)
    return low_idx, high_idx


def compute_volume_profile(
    df: pd.DataFrame,
    window: int = 120,
    bins: int = 48,
    value_area: float = 0.70,
) -> VolumeProfileLevels | None:
    if df.empty or len(df) < 20:
        return None

    sample = df.tail(max(window, 24)).copy()
    low_values = pd.to_numeric(sample["low"], errors="coerce").to_numpy(dtype=float)
    high_values = pd.to_numeric(sample["high"], errors="coerce").to_numpy(dtype=float)
    close_values = pd.to_numeric(sample["close"], errors="coerce").to_numpy(dtype=float)
    typical = (high_values + low_values + close_values) / 3.0
    volume = sample["volume"].to_numpy(dtype=float)

    p_min = float(np.nanmin(low_values))
    p_max = float(np.nanmax(high_values))
    if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
        return None

    bins = max(8, int(bins))
    edges = np.linspace(p_min, p_max, bins + 1)

    profile = np.zeros(bins, dtype=float)
    for candle_low, candle_high, candle_typical, vol in zip(low_values, high_values, typical, volume):
        if not np.isfinite(vol) or vol <= 0:
            continue
        if not np.isfinite(candle_low) or not np.isfinite(candle_high):
            continue
        candle_low, candle_high = min(candle_low, candle_high), max(candle_low, candle_high)
        if candle_high <= candle_low + 1e-12:
            idx = int(np.clip(np.digitize(candle_typical, edges) - 1, 0, bins - 1))
            profile[idx] += float(vol)
            continue

        first_idx = int(np.clip(np.searchsorted(edges, candle_low, side="right") - 1, 0, bins - 1))
        last_idx = int(np.clip(np.searchsorted(edges, candle_high, side="left"), 0, bins - 1))
        overlaps: list[tuple[int, float]] = []
        for idx in range(first_idx, last_idx + 1):
            overlap = max(0.0, min(candle_high, edges[idx + 1]) - max(candle_low, edges[idx]))
            if overlap > 0:
                overlaps.append((idx, overlap))
        total_overlap = sum(overlap for _, overlap in overlaps)
        if total_overlap <= 0:
            idx = int(np.clip(np.digitize(candle_typical, edges) - 1, 0, bins - 1))
            profile[idx] += float(vol)
            continue
        for idx, overlap in overlaps:
            profile[idx] += float(vol) * (overlap / total_overlap)

    total_vol = float(profile.sum())
    if total_vol <= 0:
        return None

    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_idx = int(np.argmax(profile))
    poc = float(centers[poc_idx])

    target = total_vol * max(0.50, min(value_area, 0.95))
    val_idx, vah_idx = _contiguous_value_area_bounds(
        profile,
        poc_idx=poc_idx,
        target_volume=target,
    )
    vah = float(centers[vah_idx])
    val = float(centers[val_idx])
    return VolumeProfileLevels(poc=poc, vah=vah, val=val)
