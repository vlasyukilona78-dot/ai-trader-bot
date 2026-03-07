from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    PUMP = "PUMP"
    PANIC = "PANIC"


def detect_market_regime(df: pd.DataFrame) -> MarketRegime:
    if df.empty or len(df) < 30:
        return MarketRegime.RANGE

    last = df.iloc[-1]
    vol_spike = float(last.get("volume_spike", 1.0) or 1.0)
    rsi = float(last.get("rsi", 50.0) or 50.0)
    adx = float(last.get("adx", 0.0) or 0.0)

    ret_30 = float(df["close"].pct_change().tail(30).sum())

    if rsi >= 80 and vol_spike >= 5.0 and ret_30 > 0.06:
        return MarketRegime.PUMP
    if rsi <= 20 and vol_spike >= 5.0 and ret_30 < -0.06:
        return MarketRegime.PANIC

    if adx > 25:
        return MarketRegime.TREND
    return MarketRegime.RANGE


def regime_score(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    last = df.iloc[-1]
    adx = float(last.get("adx", 0.0) or 0.0)
    vol = float(last.get("volume_spike", 1.0) or 1.0)
    momentum = float(df["close"].pct_change().tail(10).sum())
    score = 0.5 * np.tanh(adx / 25.0) + 0.3 * np.tanh((vol - 1.0) / 3.0) + 0.2 * np.tanh(momentum * 20.0)
    return float(score)
