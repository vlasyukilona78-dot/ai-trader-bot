from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class LiquidationBand:
    level: float
    weight: float
    intensity: float
    side: str
    start_index: int
    end_index: int
    closed_index: int | None = None
    source: str = "synthetic"
    notional_usdt: float = 0.0
    margin_usdt: float = 0.0


@dataclass(frozen=True)
class LiquidationMap:
    bands: tuple[LiquidationBand, ...]
    nearest_above_distance_pct: float | None
    nearest_below_distance_pct: float | None
    strongest_above_weight: float
    strongest_below_weight: float
    swept_above: bool
    downside_magnet: bool
    upside_risk: float


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _volume_spike(frame: pd.DataFrame) -> pd.Series:
    if "volume_spike" in frame.columns:
        return pd.to_numeric(frame["volume_spike"], errors="coerce").fillna(1.0).clip(lower=0.0)
    volume = _series(frame, "volume", 0.0).clip(lower=0.0)
    baseline = volume.rolling(20, min_periods=4).mean().replace(0.0, pd.NA)
    return (volume / baseline).fillna(1.0).clip(lower=0.0)


def _resolve_close_index(
    *,
    side: str,
    level: float,
    start_index: int,
    close_series: pd.Series,
    high_series: pd.Series,
    low_series: pd.Series,
) -> int | None:
    if level <= 0:
        return None

    price_span = max(_safe_float(high_series.max(), level) - _safe_float(low_series.min(), level), 1e-8)
    tolerance = max(abs(level) * 0.00135, price_span * 0.0040)
    arm_margin = max(tolerance * 1.95, abs(level) * 0.0039)
    sweep_margin = max(tolerance * 1.05, abs(level) * 0.0022)
    acceptance_margin = max(tolerance * 0.72, abs(level) * 0.0012)
    arm_start = min(max(start_index + 5, 0), len(high_series) - 1)
    armed_index: int | None = None

    if side == "above":
        for idx in range(arm_start, len(close_series)):
            if _safe_float(close_series.iloc[idx], 0.0) <= level - arm_margin:
                armed_index = idx
                break
        if armed_index is None:
            return None
        for idx in range(min(armed_index + 3, len(high_series) - 1), len(high_series)):
            high_px = _safe_float(high_series.iloc[idx], 0.0)
            close_px = _safe_float(close_series.iloc[idx], 0.0)
            if high_px >= level + sweep_margin and close_px >= level - acceptance_margin:
                return idx
    else:
        for idx in range(arm_start, len(close_series)):
            if _safe_float(close_series.iloc[idx], 0.0) >= level + arm_margin:
                armed_index = idx
                break
        if armed_index is None:
            return None
        for idx in range(min(armed_index + 3, len(low_series) - 1), len(low_series)):
            low_px = _safe_float(low_series.iloc[idx], 0.0)
            close_px = _safe_float(close_series.iloc[idx], 0.0)
            if low_px <= level - sweep_margin and close_px <= level + acceptance_margin:
                return idx
    return None


def _resolve_explicit_start_index(
    *,
    side: str,
    level: float,
    close_series: pd.Series,
    high_series: pd.Series,
    low_series: pd.Series,
) -> int:
    if level <= 0:
        return max(len(close_series) - 24, 0)

    price_span = max(_safe_float(high_series.max(), level) - _safe_float(low_series.min(), level), 1e-8)
    tolerance = max(abs(level) * 0.0048, price_span * 0.0105)
    start_scan = max(len(close_series) - 72, 0)

    for idx in range(start_scan, len(close_series)):
        high_px = _safe_float(high_series.iloc[idx], 0.0)
        low_px = _safe_float(low_series.iloc[idx], 0.0)
        close_px = _safe_float(close_series.iloc[idx], 0.0)
        if side == "above":
            if high_px >= level - tolerance or close_px >= level - tolerance * 0.8:
                return max(idx - 2, 0)
        else:
            if low_px <= level + tolerance or close_px <= level + tolerance * 0.8:
                return max(idx - 2, 0)

    return max(len(close_series) - 30, 0)


def _is_feed_source(source: object) -> bool:
    return str(source or "").strip().lower() in {"feed", "coinglass", "external"}


def _timestamp_to_sample_index(sample: pd.DataFrame, value: object) -> int | None:
    if value is None or sample.empty:
        return None
    try:
        numeric = float(value)
        if math.isfinite(numeric) and numeric > 0:
            if numeric > 10_000_000_000:
                numeric /= 1000.0
            ts = pd.Timestamp.fromtimestamp(numeric, tz="UTC")
        else:
            ts = pd.Timestamp(value)
    except Exception:
        try:
            ts = pd.Timestamp(value)
        except Exception:
            return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    try:
        index = sample.index
        if getattr(index, "tz", None) is None:
            index = index.tz_localize("UTC")
        else:
            index = index.tz_convert("UTC")
        pos = index.get_indexer([ts], method="nearest")[0]
    except Exception:
        return None
    if pos < 0:
        return None
    return int(min(max(pos, 0), len(sample) - 1))


def _feed_bands_from_attrs(frame: pd.DataFrame) -> list[dict[str, object]]:
    attrs = getattr(frame, "attrs", None)
    if not isinstance(attrs, dict):
        return []
    raw = attrs.get("liquidation_feed_bands") or attrs.get("coinglass_liquidation_bands")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _merge_candidates(
    candidates: list[dict[str, float | int | str]],
    *,
    close: float,
    merge_tolerance: float,
    max_per_side: int = 4,
) -> tuple[LiquidationBand, ...]:
    if not candidates:
        return ()

    tolerance = max(merge_tolerance, close * 0.0008)
    merged: list[dict[str, float | int | str]] = []

    for candidate in sorted(candidates, key=lambda item: float(item["weight"]), reverse=True):
        side = str(candidate["side"])
        level = float(candidate["level"])
        hit = None
        for existing in merged:
            if str(existing["side"]) != side:
                continue
            if abs(float(existing["level"]) - level) <= tolerance:
                hit = existing
                break
        if hit is None:
            merged.append(dict(candidate))
            continue

        total_weight = float(hit["weight"]) + float(candidate["weight"])
        if total_weight <= 0:
            continue
        hit["level"] = (
            float(hit["level"]) * float(hit["weight"]) + level * float(candidate["weight"])
        ) / total_weight
        hit["weight"] = total_weight
        hit["notional_usdt"] = _safe_float(hit.get("notional_usdt"), 0.0) + _safe_float(
            candidate.get("notional_usdt"),
            0.0,
        )
        hit["margin_usdt"] = _safe_float(hit.get("margin_usdt"), 0.0) + _safe_float(
            candidate.get("margin_usdt"),
            0.0,
        )
        hit["start_index"] = min(int(hit["start_index"]), int(candidate["start_index"]))
        if not _is_feed_source(hit.get("source", "synthetic")) and _is_feed_source(candidate.get("source", "synthetic")):
            hit["source"] = "feed"
        hit_closed = hit.get("closed_index")
        candidate_closed = candidate.get("closed_index")
        if hit_closed is None or candidate_closed is None:
            hit["closed_index"] = None
            hit["end_index"] = max(int(hit["end_index"]), int(candidate["end_index"]))
        else:
            merged_closed = max(int(hit_closed), int(candidate_closed))
            hit["closed_index"] = merged_closed
            hit["end_index"] = merged_closed

    def _score(item: dict[str, float | int | str]) -> float:
        duration = max(int(item["end_index"]) - int(item["start_index"]) + 1, 1)
        return float(item["weight"]) * min(1.0 + duration / 28.0, 2.6)

    above = sorted((item for item in merged if str(item["side"]) == "above"), key=_score, reverse=True)[:max_per_side]
    below = sorted((item for item in merged if str(item["side"]) == "below"), key=_score, reverse=True)[:max_per_side]
    kept = sorted(above + below, key=lambda item: (str(item["side"]), float(item["level"])))
    if not kept:
        return ()

    max_weight = max(float(item["weight"]) for item in kept) or 1.0
    out: list[LiquidationBand] = []
    for item in kept:
        weight = float(item["weight"])
        intensity = min(max(weight / max_weight, 0.18), 1.0)
        out.append(
            LiquidationBand(
                level=float(item["level"]),
                weight=weight,
                intensity=intensity,
                side=str(item["side"]),
                start_index=int(item["start_index"]),
                end_index=int(item["end_index"]),
                closed_index=(
                    int(item["closed_index"])
                    if item.get("closed_index") is not None
                    else None
                ),
                source=str(item.get("source") or "synthetic"),
                notional_usdt=_safe_float(item.get("notional_usdt"), 0.0),
                margin_usdt=_safe_float(item.get("margin_usdt"), 0.0),
            )
        )
    return tuple(out)


def _prune_bands(bands: tuple[LiquidationBand, ...], *, sample_len: int) -> tuple[LiquidationBand, ...]:
    if not bands:
        return ()

    kept: list[LiquidationBand] = []
    for band in bands:
        if _is_feed_source(band.source):
            kept.append(band)
            continue
        duration = max(int(band.end_index) - int(band.start_index) + 1, 1)
        is_closed = band.closed_index is not None
        is_recent_open = not is_closed and band.start_index >= max(sample_len - 10, 0)
        if is_closed and duration < 5 and band.weight < 2.15:
            continue
        if is_recent_open and duration < 4 and band.weight < 1.65:
            continue
        if duration < 3 and band.weight < 1.35:
            continue
        kept.append(band)
    return tuple(kept)


def build_liquidation_map(
    frame: pd.DataFrame,
    *,
    lookback: int = 120,
    liquidation_cluster_high: float | None = None,
    liquidation_cluster_low: float | None = None,
    liquidation_bands: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
) -> LiquidationMap:
    if frame is None or frame.empty:
        return LiquidationMap((), None, None, 0.0, 0.0, False, False, 0.0)

    explicit_feed_bands = list(liquidation_bands or [])
    attr_feed_bands = _feed_bands_from_attrs(frame)

    sample = frame.tail(max(30, lookback)).copy()
    if sample.empty:
        return LiquidationMap((), None, None, 0.0, 0.0, False, False, 0.0)

    close_series = _series(sample, "close", 0.0)
    high_series = _series(sample, "high", 0.0)
    low_series = _series(sample, "low", 0.0)
    open_series = _series(sample, "open", 0.0)
    atr_series = _series(sample, "atr", 0.0)
    volume_spike = _volume_spike(sample)

    close = _safe_float(close_series.iloc[-1], 0.0)
    if close <= 0:
        return LiquidationMap((), None, None, 0.0, 0.0, False, False, 0.0)

    atr_last = _safe_float(atr_series.iloc[-1], 0.0)
    sample_price_span = max(_safe_float(high_series.max(), close) - _safe_float(low_series.min(), close), 1e-8)
    merge_tolerance = max(close * 0.0075, atr_last * 0.85, sample_price_span * 0.018)

    body_top = pd.concat([open_series, close_series], axis=1).max(axis=1)
    body_bottom = pd.concat([open_series, close_series], axis=1).min(axis=1)
    candle_range = (high_series - low_series).clip(lower=max(close * 0.0005, 1e-8))
    upper_wick = (high_series - body_top).clip(lower=0.0)
    lower_wick = (body_bottom - low_series).clip(lower=0.0)

    local_high = high_series.rolling(5, center=True, min_periods=1).max()
    local_low = low_series.rolling(5, center=True, min_periods=1).min()

    candidates: list[dict[str, float | int | str]] = []
    sample_len = len(sample)
    last_index = max(sample_len - 1, 1)

    for idx in range(sample_len):
        close_px = _safe_float(close_series.iloc[idx], close)
        high_px = _safe_float(high_series.iloc[idx], close_px)
        low_px = _safe_float(low_series.iloc[idx], close_px)
        if close_px <= 0 or high_px <= 0 or low_px <= 0:
            continue

        rng = _safe_float(candle_range.iloc[idx], close_px * 0.001)
        wick_up_ratio = _safe_float(upper_wick.iloc[idx], 0.0) / max(rng, 1e-8)
        wick_down_ratio = _safe_float(lower_wick.iloc[idx], 0.0) / max(rng, 1e-8)
        vol = max(_safe_float(volume_spike.iloc[idx], 1.0), 0.5)
        recency = 0.55 + 0.45 * (idx / last_index)

        pivot_high = high_px >= _safe_float(local_high.iloc[idx], high_px) * 0.9995
        pivot_low = low_px <= _safe_float(local_low.iloc[idx], low_px) * 1.0005

        if (pivot_high or wick_up_ratio >= 0.22) and vol >= 0.95:
            weight = (0.75 + wick_up_ratio * 1.35) * vol * recency
            start_index = idx
            closed_index = _resolve_close_index(
                side="above",
                level=high_px,
                start_index=start_index,
                close_series=close_series,
                high_series=high_series,
                low_series=low_series,
            )
            candidates.append(
                {
                    "level": high_px,
                    "weight": weight,
                    "side": "above",
                    "start_index": start_index,
                    "end_index": closed_index if closed_index is not None else sample_len - 1,
                    "closed_index": closed_index,
                    "source": "synthetic",
                }
            )

        if (pivot_low or wick_down_ratio >= 0.22) and vol >= 0.95:
            weight = (0.75 + wick_down_ratio * 1.35) * vol * recency
            start_index = idx
            closed_index = _resolve_close_index(
                side="below",
                level=low_px,
                start_index=start_index,
                close_series=close_series,
                high_series=high_series,
                low_series=low_series,
            )
            candidates.append(
                {
                    "level": low_px,
                    "weight": weight,
                    "side": "below",
                    "start_index": start_index,
                    "end_index": closed_index if closed_index is not None else sample_len - 1,
                    "closed_index": closed_index,
                    "source": "synthetic",
                }
            )

    feed_rows = explicit_feed_bands + attr_feed_bands
    for row in feed_rows:
        level = _safe_float(row.get("level") or row.get("price"), 0.0)
        if level <= 0:
            continue
        side_raw = str(row.get("side") or "").strip().lower()
        side = side_raw if side_raw in {"above", "below"} else ("above" if level >= close else "below")
        weight = max(_safe_float(row.get("weight"), 2.8), 0.1)
        notional_usdt = max(
            _safe_float(row.get("notional_usdt"), 0.0)
            or _safe_float(row.get("value_sum"), 0.0)
            or _safe_float(row.get("amount"), 0.0)
            or _safe_float(row.get("volume_usdt"), 0.0),
            0.0,
        )
        margin_usdt = max(
            _safe_float(row.get("margin_usdt"), 0.0)
            or _safe_float(row.get("marginUsd"), 0.0)
            or _safe_float(row.get("margin_usd"), 0.0)
            or _safe_float(row.get("margin"), 0.0),
            0.0,
        )
        start_index = None
        end_index = None
        if row.get("start_index") is not None:
            start_index = int(max(0, min(_safe_float(row.get("start_index"), 0.0), sample_len - 1)))
        if row.get("end_index") is not None:
            end_index = int(max(0, min(_safe_float(row.get("end_index"), sample_len - 1), sample_len - 1)))
        if start_index is None:
            start_index = _timestamp_to_sample_index(sample, row.get("start_ts") or row.get("start_time"))
        if end_index is None:
            end_index = _timestamp_to_sample_index(sample, row.get("end_ts") or row.get("end_time"))
        if start_index is None:
            start_index = _resolve_explicit_start_index(
                side=side,
                level=level,
                close_series=close_series,
                high_series=high_series,
                low_series=low_series,
            )
        if end_index is None:
            end_index = sample_len - 1
        start_index = int(max(0, min(start_index, sample_len - 1)))
        end_index = int(max(start_index, min(int(end_index), sample_len - 1)))
        closed_index = _resolve_close_index(
            side=side,
            level=level,
            start_index=start_index,
            close_series=close_series,
            high_series=high_series,
            low_series=low_series,
        )
        candidates.append(
            {
                "level": level,
                "weight": weight,
                "side": side,
                "start_index": start_index,
                "end_index": closed_index if closed_index is not None else end_index,
                "closed_index": closed_index,
                "source": str(row.get("source") or "coinglass"),
                "notional_usdt": notional_usdt,
                "margin_usdt": margin_usdt,
            }
        )

    explicit_high = _safe_float(liquidation_cluster_high, 0.0)
    if explicit_high > 0:
        start_index = _resolve_explicit_start_index(
            side="above",
            level=explicit_high,
            close_series=close_series,
            high_series=high_series,
            low_series=low_series,
        )
        closed_index = _resolve_close_index(
            side="above",
            level=explicit_high,
            start_index=start_index,
            close_series=close_series,
            high_series=high_series,
            low_series=low_series,
        )
        candidates.append(
            {
                "level": explicit_high,
                "weight": 3.4,
                "side": "above",
                "start_index": start_index,
                "end_index": closed_index if closed_index is not None else sample_len - 1,
                "closed_index": closed_index,
                "source": "feed",
            }
        )

    explicit_low = _safe_float(liquidation_cluster_low, 0.0)
    if explicit_low > 0:
        start_index = _resolve_explicit_start_index(
            side="below",
            level=explicit_low,
            close_series=close_series,
            high_series=high_series,
            low_series=low_series,
        )
        closed_index = _resolve_close_index(
            side="below",
            level=explicit_low,
            start_index=start_index,
            close_series=close_series,
            high_series=high_series,
            low_series=low_series,
        )
        candidates.append(
            {
                "level": explicit_low,
                "weight": 3.4,
                "side": "below",
                "start_index": start_index,
                "end_index": closed_index if closed_index is not None else sample_len - 1,
                "closed_index": closed_index,
                "source": "feed",
            }
        )

    bands = _merge_candidates(candidates, close=close, merge_tolerance=merge_tolerance, max_per_side=5)
    bands = _prune_bands(bands, sample_len=sample_len)
    if not bands:
        return LiquidationMap((), None, None, 0.0, 0.0, False, False, 0.0)

    open_above_bands = [band for band in bands if band.level > close and band.closed_index is None]
    open_below_bands = [band for band in bands if band.level < close and band.closed_index is None]
    closed_above_bands = [band for band in bands if band.side == "above" and band.closed_index is not None]

    nearest_above_distance_pct = (
        min((band.level - close) / close for band in open_above_bands) if open_above_bands else None
    )
    nearest_below_distance_pct = (
        min((close - band.level) / close for band in open_below_bands) if open_below_bands else None
    )
    strongest_above_weight = max((band.weight for band in open_above_bands), default=0.0)
    strongest_below_weight = max((band.weight for band in open_below_bands), default=0.0)

    recent_close_cutoff = max(sample_len - 16, 0)
    swept_above = any(
        int(band.closed_index) >= recent_close_cutoff
        and -0.005 <= (band.level - close) / close <= 0.03
        for band in closed_above_bands
    )
    downside_magnet = any(
        0.003 <= (close - band.level) / close <= 0.045 and band.weight >= 1.15 for band in open_below_bands
    )

    upside_risk = 0.0
    for band in open_above_bands:
        dist = (band.level - close) / close
        if dist <= 0 or dist > 0.03:
            continue
        risk = band.weight * max(0.0, 1.0 - dist / 0.03)
        upside_risk = max(upside_risk, risk)

    if atr_series.max() > 0:
        atr_norm = _safe_float(atr_series.iloc[-1], 0.0) / close
        if atr_norm > 0.0:
            upside_risk *= min(max(0.75 + atr_norm * 120.0, 0.75), 1.35)

    return LiquidationMap(
        bands=bands,
        nearest_above_distance_pct=nearest_above_distance_pct,
        nearest_below_distance_pct=nearest_below_distance_pct,
        strongest_above_weight=strongest_above_weight,
        strongest_below_weight=strongest_below_weight,
        swept_above=swept_above,
        downside_magnet=downside_magnet,
        upside_risk=min(max(upside_risk, 0.0), 4.0),
    )
