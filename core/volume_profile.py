from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolumeProfileLevels:
    poc: float
    vah: float
    val: float


def compute_volume_profile(
    df: pd.DataFrame,
    window: int = 120,
    bins: int = 48,
    value_area: float = 0.70,
) -> VolumeProfileLevels | None:
    if df.empty or len(df) < 20:
        return None

    sample = df.tail(max(window, 24)).copy()
    typical = ((sample["high"] + sample["low"] + sample["close"]) / 3.0).to_numpy(dtype=float)
    volume = sample["volume"].to_numpy(dtype=float)

    p_min = float(np.nanmin(typical))
    p_max = float(np.nanmax(typical))
    if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
        return None

    bins = max(8, int(bins))
    edges = np.linspace(p_min, p_max, bins + 1)
    bucket = np.clip(np.digitize(typical, edges) - 1, 0, bins - 1)

    profile = np.zeros(bins, dtype=float)
    for idx, vol in zip(bucket, volume):
        if np.isfinite(vol) and vol > 0:
            profile[int(idx)] += float(vol)

    total_vol = float(profile.sum())
    if total_vol <= 0:
        return None

    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_idx = int(np.argmax(profile))
    poc = float(centers[poc_idx])

    target = total_vol * max(0.50, min(value_area, 0.95))
    order = np.argsort(profile)[::-1]
    cumulative = 0.0
    selected: list[int] = []
    for idx in order:
        cumulative += float(profile[idx])
        selected.append(int(idx))
        if cumulative >= target:
            break

    selected_centers = centers[selected]
    vah = float(np.max(selected_centers))
    val = float(np.min(selected_centers))
    return VolumeProfileLevels(poc=poc, vah=vah, val=val)
