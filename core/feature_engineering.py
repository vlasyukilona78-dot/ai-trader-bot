from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .indicators import compute_indicators
from .market_regime import MarketRegime
from .volume_profile import VolumeProfileLevels


REGIME_TO_NUM = {
    MarketRegime.TREND.value: 1.0,
    MarketRegime.RANGE.value: 0.0,
    MarketRegime.PUMP.value: 2.0,
    MarketRegime.PANIC.value: -2.0,
}


@dataclass
class FeatureRow:
    symbol: str
    ts: pd.Timestamp
    values: dict[str, float]


REQUIRED_MODEL_FEATURES = [
    "rsi",
    "ema20",
    "ema50",
    "atr",
    "vwap_dist",
    "bb_position",
    "volume_spike",
    "obv_div_short",
    "obv_div_long",
    "cvd_div_short",
    "cvd_div_long",
    "poc_dist",
    "vah_dist",
    "val_dist",
    "regime_num",
    "funding_rate",
    "open_interest",
    "long_short_ratio",
    "sentiment_index",
    "liq_high_dist",
    "liq_low_dist",
    "close_ret_5",
    "close_ret_20",
    "mtf_rsi_5m",
    "mtf_rsi_15m",
    "mtf_rsi_1h",
    "mtf_atr_norm_5m",
    "mtf_atr_norm_15m",
    "mtf_atr_norm_1h",
    "mtf_trend_5m",
    "mtf_trend_15m",
    "mtf_trend_1h",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (float, int)):
            return float(value)
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _divergence_flags(df: pd.DataFrame, lookback: int = 6) -> tuple[float, float, float, float]:
    if len(df) <= lookback + 1:
        return 0.0, 0.0, 0.0, 0.0

    last = df.iloc[-1]
    prev = df.iloc[-1 - lookback]

    price_up = float(last["close"]) > float(prev["close"])
    price_down = float(last["close"]) < float(prev["close"])

    obv_down = float(last.get("obv", 0.0)) < float(prev.get("obv", 0.0))
    obv_up = float(last.get("obv", 0.0)) > float(prev.get("obv", 0.0))

    cvd_down = float(last.get("cvd", 0.0)) < float(prev.get("cvd", 0.0))
    cvd_up = float(last.get("cvd", 0.0)) > float(prev.get("cvd", 0.0))

    obv_div_short = 1.0 if (price_up and obv_down) else 0.0
    obv_div_long = 1.0 if (price_down and obv_up) else 0.0
    cvd_div_short = 1.0 if (price_up and cvd_down) else 0.0
    cvd_div_long = 1.0 if (price_down and cvd_up) else 0.0
    return obv_div_short, obv_div_long, cvd_div_short, cvd_div_long


def _mtf_momentum(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty:
        return 0.0, 0.0
    close = df["close"]
    ret_5 = float(close.pct_change().tail(5).sum())
    ret_20 = float(close.pct_change().tail(20).sum())
    if not np.isfinite(ret_5):
        ret_5 = 0.0
    if not np.isfinite(ret_20):
        ret_20 = 0.0
    return ret_5, ret_20


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close", "volume"])
    return out


def sanitize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.replace([np.inf, -np.inf], np.nan)

    core_price_cols = [col for col in ("open", "high", "low", "close", "volume") if col in out.columns]
    for col in core_price_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if core_price_cols:
        out = out.dropna(subset=[col for col in ("open", "high", "low", "close") if col in out.columns])
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)

    numeric_cols = list(out.select_dtypes(include=[np.number]).columns)
    for col in numeric_cols:
        series = pd.to_numeric(out[col], errors="coerce")
        if col in {"rsi", "mtf_rsi_5m", "mtf_rsi_15m", "mtf_rsi_1h"}:
            out[col] = series.clip(lower=0.0, upper=100.0).fillna(50.0)
        elif col == "volume_spike":
            out[col] = series.clip(lower=0.0, upper=30.0).fillna(1.0)
        elif col in {"atr", "atr_norm", "mtf_atr_norm_5m", "mtf_atr_norm_15m", "mtf_atr_norm_1h"}:
            out[col] = series.clip(lower=0.0).fillna(0.0)
        elif col in {"vwap_dist", "close_ret_5", "close_ret_20", "mtf_trend_5m", "mtf_trend_15m", "mtf_trend_1h"}:
            out[col] = series.clip(lower=-1.0, upper=1.0).fillna(0.0)
        elif col in {"hist", "obv", "cvd"}:
            out[col] = series.ffill().bfill().fillna(0.0)
        else:
            out[col] = series.ffill().bfill()

    return out


def assess_feature_frame_quality(
    df: pd.DataFrame,
    *,
    recent_window: int = 96,
    severe_gap_multiple: float = 3.0,
    max_recent_gap_ratio: float = 0.05,
    max_recent_zero_volume_ratio: float = 0.10,
    min_recent_zero_volume_count: int = 3,
) -> dict[str, float | str | bool]:
    report: dict[str, float | str | bool] = {
        "usable": True,
        "reason": "",
        "expected_bar_seconds": 0.0,
        "severe_gap_threshold_seconds": 0.0,
        "latest_gap_seconds": 0.0,
        "max_recent_gap_seconds": 0.0,
        "recent_gap_ratio": 0.0,
        "recent_severe_gap_count": 0.0,
        "total_severe_gap_count": 0.0,
        "recent_zero_volume_count": 0.0,
        "recent_zero_volume_ratio": 0.0,
    }
    if df.empty or len(df.index) < 3:
        report["usable"] = False
        report["reason"] = "insufficient_frame_rows"
        return report
    if not isinstance(df.index, pd.DatetimeIndex):
        return report

    ordered = df.sort_index()
    if not ordered.index.is_monotonic_increasing:
        report["usable"] = False
        report["reason"] = "non_monotonic_index"
        return report

    diffs = ordered.index.to_series().diff().dt.total_seconds().dropna()
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.empty:
        report["usable"] = False
        report["reason"] = "invalid_time_index"
        return report

    expected_bar_seconds = float(positive_diffs.median())
    severe_gap_threshold = expected_bar_seconds * max(1.5, float(severe_gap_multiple))
    severe_mask = diffs >= severe_gap_threshold
    recent_diffs = diffs.tail(min(len(diffs), max(4, int(recent_window))))
    recent_severe_count = int((recent_diffs >= severe_gap_threshold).sum()) if not recent_diffs.empty else 0
    recent_gap_ratio = float(recent_severe_count / len(recent_diffs)) if len(recent_diffs) > 0 else 0.0
    latest_gap_seconds = float(recent_diffs.iloc[-1]) if not recent_diffs.empty else 0.0
    max_recent_gap_seconds = float(recent_diffs.max()) if not recent_diffs.empty else 0.0

    report.update(
        {
            "expected_bar_seconds": expected_bar_seconds,
            "severe_gap_threshold_seconds": severe_gap_threshold,
            "latest_gap_seconds": latest_gap_seconds,
            "max_recent_gap_seconds": max_recent_gap_seconds,
            "recent_gap_ratio": recent_gap_ratio,
            "recent_severe_gap_count": float(recent_severe_count),
            "total_severe_gap_count": float(int(severe_mask.sum())),
        }
    )

    recent_frame = ordered.tail(min(len(ordered), max(4, int(recent_window))))
    if "volume" in recent_frame.columns:
        recent_volume = pd.to_numeric(recent_frame["volume"], errors="coerce").fillna(0.0)
        recent_zero_volume_count = int((recent_volume <= 0.0).sum())
        recent_zero_volume_ratio = (
            float(recent_zero_volume_count / len(recent_volume)) if len(recent_volume) > 0 else 0.0
        )
        report["recent_zero_volume_count"] = float(recent_zero_volume_count)
        report["recent_zero_volume_ratio"] = recent_zero_volume_ratio

    if latest_gap_seconds >= severe_gap_threshold:
        report["usable"] = False
        report["reason"] = "latest_gap_exceeds_threshold"
    elif recent_severe_count >= 2 or recent_gap_ratio > float(max_recent_gap_ratio):
        report["usable"] = False
        report["reason"] = "recent_gap_cluster"
    elif (
        float(report.get("recent_zero_volume_ratio", 0.0) or 0.0) > float(max_recent_zero_volume_ratio)
        and float(report.get("recent_zero_volume_count", 0.0) or 0.0) >= float(min_recent_zero_volume_count)
    ):
        report["usable"] = False
        report["reason"] = "recent_zero_volume_cluster"

    return report


def compute_mtf_feature_snapshot(df: pd.DataFrame) -> dict[str, float]:
    values = {
        "mtf_rsi_5m": 50.0,
        "mtf_rsi_15m": 50.0,
        "mtf_rsi_1h": 50.0,
        "mtf_atr_norm_5m": 0.0,
        "mtf_atr_norm_15m": 0.0,
        "mtf_atr_norm_1h": 0.0,
        "mtf_trend_5m": 0.0,
        "mtf_trend_15m": 0.0,
        "mtf_trend_1h": 0.0,
    }

    for rule, key_suffix in (("5min", "5m"), ("15min", "15m"), ("1h", "1h")):
        mtf = _resample_ohlcv(df[["open", "high", "low", "close", "volume"]], rule=rule)
        if len(mtf) < 30:
            continue
        mtf_ind = compute_indicators(mtf)
        last = mtf_ind.iloc[-1]
        close = _safe_float(last.get("close"), 0.0)
        if close <= 0:
            continue

        values[f"mtf_rsi_{key_suffix}"] = _safe_float(last.get("rsi"), 50.0)
        values[f"mtf_atr_norm_{key_suffix}"] = _safe_float(last.get("atr"), 0.0) / close
        values[f"mtf_trend_{key_suffix}"] = (
            _safe_float(last.get("ema20"), close) - _safe_float(last.get("ema50"), close)
        ) / close

    return values


def build_feature_row(
    symbol: str,
    df: pd.DataFrame,
    volume_profile: VolumeProfileLevels | None,
    regime: MarketRegime,
    extras: dict[str, Any] | None = None,
) -> FeatureRow | None:
    if df.empty:
        return None

    extras = extras or {}
    last = df.iloc[-1]
    close = _safe_float(last.get("close"), 0.0)
    if close <= 0:
        return None

    obv_div_short, obv_div_long, cvd_div_short, cvd_div_long = _divergence_flags(df)
    ret_5, ret_20 = _mtf_momentum(df)

    poc = volume_profile.poc if volume_profile else close
    vah = volume_profile.vah if volume_profile else close
    val = volume_profile.val if volume_profile else close

    liq_high = _safe_float(extras.get("liquidation_cluster_high"), close)
    liq_low = _safe_float(extras.get("liquidation_cluster_low"), close)

    values = {
        "rsi": _safe_float(last.get("rsi"), 50.0),
        "ema20": _safe_float(last.get("ema20"), close),
        "ema50": _safe_float(last.get("ema50"), close),
        "atr": _safe_float(last.get("atr"), 0.0),
        "vwap_dist": _safe_float(last.get("vwap_dist"), 0.0),
        "bb_position": _safe_float(last.get("bb_position"), 0.5),
        "volume_spike": _safe_float(last.get("volume_spike"), 1.0),
        "obv_div_short": obv_div_short,
        "obv_div_long": obv_div_long,
        "cvd_div_short": cvd_div_short,
        "cvd_div_long": cvd_div_long,
        "poc_dist": (close - poc) / close,
        "vah_dist": (close - vah) / close,
        "val_dist": (close - val) / close,
        "regime_num": REGIME_TO_NUM.get(regime.value, 0.0),
        "funding_rate": _safe_float(extras.get("funding_rate"), 0.0),
        "open_interest": _safe_float(extras.get("open_interest"), 0.0),
        "long_short_ratio": _safe_float(extras.get("long_short_ratio"), 1.0),
        "sentiment_index": _safe_float(extras.get("sentiment_index"), 50.0),
        "liq_high_dist": (close - liq_high) / close if liq_high else 0.0,
        "liq_low_dist": (close - liq_low) / close if liq_low else 0.0,
        "close_ret_5": ret_5,
        "close_ret_20": ret_20,
    }

    values.update(compute_mtf_feature_snapshot(df))

    for key in list(values.keys()):
        if not np.isfinite(values[key]):
            values[key] = 0.0

    return FeatureRow(symbol=symbol, ts=df.index[-1], values=values)


def to_model_frame(rows: list[FeatureRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=REQUIRED_MODEL_FEATURES)
    records: list[dict[str, float]] = []
    for row in rows:
        records.append({k: float(row.values.get(k, 0.0)) for k in REQUIRED_MODEL_FEATURES})
    return pd.DataFrame(records)
