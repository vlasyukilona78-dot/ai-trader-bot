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
    "mtf_atr_norm_5m",
    "mtf_atr_norm_15m",
    "mtf_trend_5m",
    "mtf_trend_15m",
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


def _compute_mtf_features(df: pd.DataFrame) -> dict[str, float]:
    values = {
        "mtf_rsi_5m": 50.0,
        "mtf_rsi_15m": 50.0,
        "mtf_atr_norm_5m": 0.0,
        "mtf_atr_norm_15m": 0.0,
        "mtf_trend_5m": 0.0,
        "mtf_trend_15m": 0.0,
    }

    for rule, key_suffix in (("5min", "5m"), ("15min", "15m")):
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

    values.update(_compute_mtf_features(df))

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
