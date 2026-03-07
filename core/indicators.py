from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def bollinger_bands(series: pd.Series, period: int = 20, mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    return lower, mid, upper


def keltner_channels(df: pd.DataFrame, period: int = 20, mult: float = 1.5) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(df["close"], period)
    range_atr = atr(df, period=period)
    upper = mid + mult * range_atr
    lower = mid - mult * range_atr
    return lower, mid, upper


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cumulative_vol = df["volume"].cumsum().replace(0, np.nan)
    return (typical * df["volume"]).cumsum() / cumulative_vol


def obv(df: pd.DataFrame) -> pd.Series:
    step = np.sign(df["close"].diff()).fillna(0.0) * df["volume"]
    return step.cumsum()


def cvd(df: pd.DataFrame) -> pd.Series:
    signed = np.where(df["close"] >= df["open"], df["volume"], -df["volume"])
    return pd.Series(signed, index=df.index).cumsum()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(period).mean().replace(0, np.nan)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr_series
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr_series

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.rolling(period).mean()


def compute_indicators(df: pd.DataFrame, keltner_mult: float = 1.5) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["rsi"] = rsi(out["close"], period=14)
    out["ema20"] = ema(out["close"], span=20)
    out["ema50"] = ema(out["close"], span=50)
    out["atr"] = atr(out, period=14)

    bb_lower, bb_mid, bb_upper = bollinger_bands(out["close"], period=20, mult=2.0)
    out["bb_lower"] = bb_lower
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_upper

    kc_lower, kc_mid, kc_upper = keltner_channels(out, period=20, mult=keltner_mult)
    out["kc_lower"] = kc_lower
    out["kc_mid"] = kc_mid
    out["kc_upper"] = kc_upper

    out["vwap"] = vwap(out)
    out["vwap_dist"] = (out["close"] - out["vwap"]) / out["vwap"].replace(0, np.nan)

    out["obv"] = obv(out)
    out["cvd"] = cvd(out)

    out["volume_ma20"] = out["volume"].rolling(20).mean()
    out["volume_spike"] = out["volume"] / out["volume_ma20"].replace(0, np.nan)

    out["bb_position"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
    out["adx"] = adx(out, period=14)

    return out
