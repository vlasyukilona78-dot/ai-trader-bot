import numpy as np
import pandas as pd

from app.main import build_liquidation_map, _as_float, _early_config_float, _latest_peak_age_bars


n = 72
idx = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
close = np.full(n, 0.00212)
close[28:64] = np.linspace(0.00212, 0.00280, 36)
close[64:70] = np.linspace(0.00280, 0.00282, 6)
close[70] = 0.00281
close[71] = 0.002798

df = pd.DataFrame(
    {
        "open": close * 0.999,
        "high": close * 1.010,
        "low": close * 0.992,
        "close": close,
        "volume": np.linspace(20.0, 28.0, n),
        "volume_spike": np.concatenate([np.full(n - 5, 1.1), np.array([2.6, 2.1, 1.8, 1.45, 1.28])]),
        "rsi": np.concatenate([np.full(n - 4, 56.0), np.array([68.0, 65.0, 61.0, 58.0])]),
        "hist": np.concatenate([np.full(n - 4, 0.00010), np.array([0.00042, 0.00030, 0.00018, 0.00006])]),
        "obv": np.linspace(100.0, 320.0, n),
        "cvd": np.linspace(90.0, 300.0, n),
        "bb_upper": close * 1.003,
        "kc_upper": close * 1.002,
        "ema20": np.linspace(0.00214, 0.00266, n),
        "ema50": np.linspace(0.00212, 0.00252, n),
        "vwap": np.linspace(0.00213, 0.00258, n),
        "atr": np.full(n, 0.00005),
    },
    index=idx,
)
df.iloc[-1, df.columns.get_loc("high")] = close[-2] * 1.001

row = df.iloc[-1]
prev = df.iloc[-2]
liq = build_liquidation_map(df)
closep = float(row.close)
atr = max(float(row.atr), closep * 0.001, 1e-8)
layer1 = {
    "clean_pump_pct": 0.058,
    "clean_pump_min_pct_used": 0.05,
    "volume_spike_threshold_used": 2.0,
    "volume_spike": 1.6,
    "rsi": 62.0,
}
confirmed_pump_min = max(_as_float(layer1.get("clean_pump_min_pct_used"), 0.05), 0.0)
early_pump_min = max(
    0.0,
    _early_config_float("EARLY_WATCH_CLEAN_PUMP_MIN_PCT", max(0.04, confirmed_pump_min - 0.01)),
)
confirmed_volume_gate = max(_as_float(layer1.get("volume_spike_threshold_used"), 2.0), 2.0)
early_volume_gate = max(
    1.25,
    _early_config_float("EARLY_WATCH_VOLUME_SPIKE_MIN", max(1.15, confirmed_volume_gate * 0.575)),
)
early_rsi_min = _early_config_float("EARLY_WATCH_RSI_MIN", 52.0)
print("early gates", early_pump_min, early_volume_gate, early_rsi_min)

high = float(row.high)
low = float(row.low)
open_px = float(row.open)
rsi = max(_as_float(layer1.get("rsi"), 0.0), float(row.rsi))
prev_rsi = float(prev.rsi)
volume_spike = max(_as_float(layer1.get("volume_spike"), 0.0), float(row.volume_spike))
recent_volume_spike = float(pd.to_numeric(df.tail(5)["volume_spike"], errors="coerce").max())
hist = float(row.hist)
prev_hist = float(prev.hist)
recent_peak_window = min(len(df), 28 if closep < 0.02 else 24)
recent_high_numeric = pd.to_numeric(df.tail(recent_peak_window)["high"], errors="coerce").dropna()
recent_close_numeric = pd.to_numeric(df.tail(recent_peak_window)["close"], errors="coerce").dropna()
recent_high = float(recent_high_numeric.max())
recent_close_high = float(recent_close_numeric.max())
signal_peak_reference = recent_close_high if recent_close_high > 0 else recent_high
if recent_high > signal_peak_reference > 0:
    allowed_peak_wick = max(atr * 0.32, signal_peak_reference * (0.0024 if closep < 0.02 else 0.0018))
    signal_peak_reference = min(recent_high, signal_peak_reference + allowed_peak_wick)
print("recent_high", recent_high, "recent_close_high", recent_close_high, "signal_peak_reference", signal_peak_reference)

close_peak_age = _latest_peak_age_bars(
    recent_close_numeric,
    reference_price=closep,
    atr=atr,
    relative_tolerance=0.0018 if closep < 0.02 else 0.0012,
)
high_peak_age = _latest_peak_age_bars(
    recent_high_numeric,
    reference_price=closep,
    atr=atr,
    relative_tolerance=0.0022 if closep < 0.02 else 0.0016,
)
peak_age_bars = min(close_peak_age, high_peak_age)
print("peak ages", close_peak_age, high_peak_age, peak_age_bars)

pullback_from_peak_pct = max(0.0, (signal_peak_reference - closep) / max(signal_peak_reference, 1e-8))
recent_tail = df.tail(recent_peak_window)
tail_highs = pd.to_numeric(recent_tail["high"], errors="coerce").to_numpy(dtype=float)
tail_lows = pd.to_numeric(recent_tail["low"], errors="coerce").to_numpy(dtype=float)
local_peak_rel = int(np.nanargmax(tail_highs))
pump_back = 18 if closep < 0.02 else 14
pump_base_start = max(0, local_peak_rel - pump_back)
active_pump_low = float(np.nanmin(tail_lows[pump_base_start : local_peak_rel + 1]))
pump_amplitude = max(signal_peak_reference - active_pump_low, atr)
pump_drawdown_ratio = max(0.0, (signal_peak_reference - closep) / max(pump_amplitude, 1e-8))
pump_range_position = min(1.0, max(0.0, (closep - active_pump_low) / max(pump_amplitude, 1e-8)))
pump_midpoint = active_pump_low + pump_amplitude * 0.52
peak_pullback_limit = max(0.0058, min(0.0175, (atr / max(closep, 1e-8)) * 2.45))
early_reversal_pullback = max(0.0011, min(0.0055, (atr / max(closep, 1e-8)) * 0.90))
minimum_reversal_pullback = max(0.0009, min(0.0032, early_reversal_pullback * 0.55))
peak_still_fresh = peak_age_bars <= 1
peak_recent_enough = peak_age_bars <= 2
near_peak_limit = peak_pullback_limit * (1.12 if peak_still_fresh else 1.08 if peak_recent_enough else 1.0)
near_peak = pullback_from_peak_pct <= near_peak_limit
candle_range = max(high - low, 1e-8)
upper_wick = max(high - max(open_px, closep), 0.0)
micro_reversal_near_peak = peak_still_fresh and near_peak and (
    closep < float(prev.close) or rsi < prev_rsi or hist < prev_hist or upper_wick / candle_range >= 0.18
)
first_reaction = (
    pullback_from_peak_pct >= early_reversal_pullback
    or closep < float(prev.close)
    or rsi < prev_rsi
    or hist < prev_hist
    or upper_wick / candle_range >= 0.22
)
watch_pullback_cap = min(
    max(
        peak_pullback_limit * 1.02,
        early_reversal_pullback * 1.14,
        0.0082 if closep < 0.02 else 0.0058,
    ),
    0.018 if closep < 0.02 else 0.014,
)
watch_drawdown_cap = min(max(0.078 * 0.72, 0.052), 0.066)
recent_watch_pullback_allowance = min(
    watch_pullback_cap,
    max(
        peak_pullback_limit * 0.92,
        early_reversal_pullback * 1.06,
        0.0074 if closep < 0.02 else 0.0052,
    ),
)
print("pullback", pullback_from_peak_pct, "near_peak_limit", near_peak_limit, "near_peak", near_peak)
print("drawdown", pump_drawdown_ratio, "range_pos", pump_range_position)
print("midpoint_gate", max(pump_midpoint + atr * 0.12, float(row.ema20) * 1.0004), "close", closep)
print("watch caps", watch_pullback_cap, watch_drawdown_cap, recent_watch_pullback_allowance)
print("fresh", peak_still_fresh, peak_recent_enough, "first_reaction", first_reaction, "micro", micro_reversal_near_peak)
print("wick ratio", upper_wick / candle_range, "liq", liq.swept_above, liq.upside_risk, liq.downside_magnet)
print(
    "guards",
    {
        "g1": not peak_still_fresh
        and not liq.swept_above
        and not (
            peak_recent_enough
            and first_reaction
            and pullback_from_peak_pct <= recent_watch_pullback_allowance
            and pump_drawdown_ratio <= watch_drawdown_cap * 0.92
        ),
        "g2": pullback_from_peak_pct < minimum_reversal_pullback and not micro_reversal_near_peak and not liq.swept_above,
        "g3": pump_drawdown_ratio > watch_drawdown_cap,
        "g4": pump_range_position < 0.70 and not liq.swept_above,
        "g5": closep < max(pump_midpoint + atr * 0.12, float(row.ema20) * 1.0004),
        "g6": ((not near_peak and not liq.swept_above) or pullback_from_peak_pct > watch_pullback_cap),
    },
)

continuation_risk = 0.0
if closep >= signal_peak_reference * 0.9993:
    continuation_risk += 1.25
if volume_spike >= max(recent_volume_spike * 0.97, early_volume_gate):
    continuation_risk += 1.10
if rsi >= prev_rsi and rsi >= max(58.0, early_rsi_min + 4.0):
    continuation_risk += 1.0
if hist >= prev_hist and hist > 0:
    continuation_risk += 1.0
if closep >= float(prev.close):
    continuation_risk += 0.5
if upper_wick / candle_range < 0.16:
    continuation_risk += 0.75
if liq.upside_risk > 0:
    continuation_risk += min(1.75, liq.upside_risk * 0.55)
still_accelerating = (
    closep >= signal_peak_reference * 0.999
    and volume_spike >= max(recent_volume_spike * 0.95, early_volume_gate)
    and rsi >= prev_rsi
    and hist >= prev_hist
)
print("risk", continuation_risk, "still_accel", still_accelerating)
