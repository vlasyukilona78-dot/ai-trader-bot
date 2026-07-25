"""Pump-dynamics features: is this move engineered, and is it running out of steam?

Everything here is derived from OHLCV so it can be validated on history. The
measured predictors so far have been about magnitude and context rather than
oscillators - RSI and volume-spike showed essentially zero rank correlation with
outcome across 5523 events - so these target the mechanism instead: who is
driving the move, whether the buying behind it is fading, and how stretched it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def relative_strength(
    df: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    *,
    lookback: int = 24,
) -> dict[str, float]:
    """Separate an engineered single-coin pump from a market-wide rally.

    The strategy's premise is a market maker running one coin. If the whole board
    is up, the same move is beta - and shorting beta into a rising market is a
    different trade with a different risk profile. Both legs are reported so the
    difference can be tested rather than assumed.
    """
    out = {"coin_return": float("nan"), "btc_return": float("nan"),
           "relative_strength": float("nan"), "idiosyncratic": float("nan")}
    if df.empty or len(df) < lookback + 1:
        return out

    close = pd.to_numeric(df["close"], errors="coerce")
    coin_ret = float(close.iloc[-1] / close.iloc[-1 - lookback] - 1.0)
    out["coin_return"] = coin_ret

    if benchmark is None or benchmark.empty or len(benchmark) < lookback + 1:
        return out

    bclose = pd.to_numeric(benchmark["close"], errors="coerce")
    btc_ret = float(bclose.iloc[-1] / bclose.iloc[-1 - lookback] - 1.0)
    out["btc_return"] = btc_ret
    out["relative_strength"] = coin_ret - btc_ret
    # 1.0 when the coin moved on its own, 0.0 when it simply followed the market
    denom = abs(coin_ret) + abs(btc_ret)
    out["idiosyncratic"] = float(abs(coin_ret - btc_ret) / denom) if denom > 1e-12 else float("nan")
    return out


def volume_exhaustion(df: pd.DataFrame, *, window: int = 24, segments: int = 3) -> dict[str, float]:
    """Is the buying behind the run drying up?

    A pump sustained by fresh volume can keep going; one whose later legs trade
    on progressively less volume is being held up by fewer and fewer buyers.
    """
    out = {"volume_trend": float("nan"), "late_vs_early_volume": float("nan")}
    if df.empty or len(df) < window:
        return out

    vol = pd.to_numeric(df["volume"], errors="coerce").tail(window).fillna(0.0).to_numpy()
    if vol.sum() <= 0:
        return out

    chunks = np.array_split(vol, max(2, segments))
    means = [float(c.mean()) for c in chunks if len(c)]
    if len(means) < 2 or means[0] <= 0:
        return out

    out["late_vs_early_volume"] = float(means[-1] / means[0])
    x = np.arange(len(means), dtype=float)
    slope = float(np.polyfit(x, np.asarray(means) / max(means[0], 1e-12), 1)[0])
    out["volume_trend"] = slope
    return out


def rejection_wicks(df: pd.DataFrame, *, window: int = 12) -> dict[str, float]:
    """Upper-wick dominance - sellers absorbing into the highs."""
    out = {"upper_wick_ratio": float("nan"), "peak_bar_wick": float("nan")}
    if df.empty or len(df) < 3:
        return out

    tail = df.tail(window)
    high = pd.to_numeric(tail["high"], errors="coerce")
    low = pd.to_numeric(tail["low"], errors="coerce")
    close = pd.to_numeric(tail["close"], errors="coerce")
    open_ = pd.to_numeric(tail["open"], errors="coerce")

    rng = (high - low).replace(0, np.nan)
    upper = high - np.maximum(close, open_)
    out["upper_wick_ratio"] = float((upper / rng).mean())

    peak_idx = int(high.to_numpy().argmax())
    peak_rng = float(high.iloc[peak_idx] - low.iloc[peak_idx])
    if peak_rng > 0:
        peak_upper = float(high.iloc[peak_idx] - max(close.iloc[peak_idx], open_.iloc[peak_idx]))
        out["peak_bar_wick"] = peak_upper / peak_rng
    return out


def pump_acceleration(df: pd.DataFrame, *, window: int = 24) -> dict[str, float]:
    """Is the run still accelerating, or already decelerating into a top?"""
    out = {"acceleration": float("nan"), "second_half_share": float("nan")}
    if df.empty or len(df) < window:
        return out

    close = pd.to_numeric(df["close"], errors="coerce").tail(window).to_numpy(dtype=float)
    if len(close) < 4 or close[0] <= 0:
        return out

    mid = len(close) // 2
    first = (close[mid] - close[0]) / close[0]
    second = (close[-1] - close[mid]) / close[mid] if close[mid] > 0 else np.nan
    out["acceleration"] = float(second - first)
    total = abs(first) + abs(second)
    out["second_half_share"] = float(second / total) if total > 1e-12 else float("nan")
    return out


def extension(df: pd.DataFrame) -> dict[str, float]:
    """How stretched price is from its mean, measured in ATR units.

    Percentage distance is not comparable across coins with different volatility;
    ATR units are, which matters when ranking candidates cross-sectionally.
    """
    out = {"ext_ema20_atr": float("nan"), "ext_ema50_atr": float("nan"), "consecutive_up": float("nan")}
    if df.empty or len(df) < 5:
        return out

    last = df.iloc[-1]
    close = float(last.get("close") or 0.0)
    atr = float(last.get("atr") or 0.0)
    if close <= 0 or atr <= 0:
        return out

    for name, col in (("ext_ema20_atr", "ema20"), ("ext_ema50_atr", "ema50")):
        ref = last.get(col)
        if ref is not None and float(ref) > 0:
            out[name] = float((close - float(ref)) / atr)

    closes = pd.to_numeric(df["close"], errors="coerce").tail(30).to_numpy(dtype=float)
    run = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            run += 1
        else:
            break
    out["consecutive_up"] = float(run)
    return out


def build_pump_features(
    df: pd.DataFrame,
    *,
    benchmark: pd.DataFrame | None = None,
    window: int = 24,
) -> dict[str, float]:
    """All pump-dynamics features for one bar of one symbol."""
    feats: dict[str, float] = {}
    feats.update(relative_strength(df, benchmark, lookback=window))
    feats.update(volume_exhaustion(df, window=window))
    feats.update(rejection_wicks(df))
    feats.update(pump_acceleration(df, window=window))
    feats.update(extension(df))
    return feats
