"""Build a labelled dataset of pump events for the short-the-fade strategy.

Each row is one pump: features known at the moment it fires, plus outcome labels
measured afterwards. The primary label is `mae_pct` - how much further price ran
up after the event - because that is what decides how deep the position goes
against you and how many averaging legs a trade needs. `n_averages` and the
`good_*` class flags are derived from the same forward window so a model can be
trained on any of them without rebuilding the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.indicators import compute_indicators
from core.levels import (
    distance_to_fib,
    estimate_liquidation_map,
    find_horizontal_levels,
    multi_timeframe_confluence,
    nearest_level_above,
    rsi_divergence,
)
from core.pump_features import build_pump_features
from trading.market_data.history import HistoryCollector


@dataclass
class EventConfig:
    min_move_pct: float = 0.05
    lookback_hours: int = 6
    cooldown_hours: int = 6


@dataclass
class LabelConfig:
    horizon_hours: int = 48
    dca_step_pct: float = 0.08
    dca_max_adds: int = 6
    dca_target_pct: float = 0.03
    good_mae_thresholds: tuple[float, ...] = (0.03, 0.05, 0.08)


@dataclass
class PumpEvent:
    symbol: str
    ts: int
    entry: float
    move_pct: float
    run_up_bars: int
    features: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)


def detect_events(df_1h: pd.DataFrame, cfg: EventConfig) -> list[PumpEvent]:
    """Emit one event per distinct run-up, measured low-to-close over a window."""
    if df_1h.empty or len(df_1h) < cfg.lookback_hours + 2:
        return []

    low = pd.to_numeric(df_1h["low"], errors="coerce")
    close = pd.to_numeric(df_1h["close"], errors="coerce")
    window_low = low.rolling(cfg.lookback_hours, min_periods=2).min()
    move = (close - window_low) / window_low.replace(0, np.nan)

    events: list[PumpEvent] = []
    last_ts = -10**18
    for i in range(cfg.lookback_hours, len(df_1h)):
        m = move.iloc[i]
        if not np.isfinite(m) or m < cfg.min_move_pct:
            continue
        ts = int(df_1h["time"].iloc[i])
        if ts - last_ts < cfg.cooldown_hours * 3600:
            continue
        last_ts = ts
        # how many consecutive bars the run-up took
        bars = 1
        while bars < cfg.lookback_hours and close.iloc[i - bars] > low.iloc[i - bars - 1 if i - bars - 1 >= 0 else 0]:
            bars += 1
        events.append(
            PumpEvent(symbol="", ts=ts, entry=float(close.iloc[i]), move_pct=float(m), run_up_bars=int(bars))
        )
    return events


def label_event(event: PumpEvent, df_fwd: pd.DataFrame, cfg: LabelConfig) -> dict:
    """Forward-looking outcome labels. df_fwd must start at or after the event."""
    if df_fwd.empty:
        return {}

    high = pd.to_numeric(df_fwd["high"], errors="coerce")
    low = pd.to_numeric(df_fwd["low"], errors="coerce")
    entry = event.entry

    mae = float((high.max() - entry) / entry)
    mfe = float((entry - low.min()) / entry)

    hit = low[low <= entry * (1 - cfg.dca_target_pct)]
    time_to_target_min = (
        int((int(df_fwd["time"].loc[hit.index[0]]) - event.ts) / 60) if len(hit) else -1
    )

    # Replay the averaging plan: add a leg at each step above entry, take profit
    # once price trades back through the blended entry minus the target.
    legs = [entry]
    n_adds = 0
    resolved = False
    peak_dd = 0.0
    for _, row in df_fwd.iterrows():
        h, l = float(row["high"]), float(row["low"])
        while n_adds < cfg.dca_max_adds and h >= entry * (1 + cfg.dca_step_pct * (n_adds + 1)):
            n_adds += 1
            legs.append(entry * (1 + cfg.dca_step_pct * n_adds))
        avg = sum(legs) / len(legs)
        peak_dd = max(peak_dd, (h - avg) / avg * len(legs))
        if l <= avg * (1 - cfg.dca_target_pct):
            resolved = True
            break

    out = {
        "mae_pct": mae,
        "mfe_pct": mfe,
        "time_to_target_min": time_to_target_min,
        "n_averages": n_adds,
        "dca_resolved": int(resolved),
        "dca_peak_drawdown_units": float(peak_dd),
        "mfe_beats_mae": int(mfe > mae),
    }
    for thr in cfg.good_mae_thresholds:
        out[f"good_mae_{int(thr*100)}"] = int(mae <= thr)
    return out


def _rsi_at(df: pd.DataFrame, ts: int) -> float:
    sub = df[df["time"] <= ts]
    if len(sub) < 20:
        return float("nan")
    enriched = compute_indicators(sub.tail(200))
    return float(enriched.iloc[-1].get("rsi", float("nan")))


def build_features(
    event: PumpEvent,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    funding: pd.DataFrame,
    df_4h: pd.DataFrame | None = None,
    benchmark_1h: pd.DataFrame | None = None,
) -> dict:
    """Everything knowable at the moment the pump fires - no forward data."""
    hist = df_1h[df_1h["time"] <= event.ts]
    if len(hist) < 60:
        return {}

    enriched = compute_indicators(hist.tail(400))
    last = enriched.iloc[-1]
    close = float(last["close"])

    def pct_change(hours: int) -> float:
        sub = hist.tail(hours + 1)
        if len(sub) < 2:
            return float("nan")
        base = float(sub["close"].iloc[0])
        return (close - base) / base if base else float("nan")

    def headroom(hours: int) -> float:
        sub = hist.tail(hours + 1)
        if sub.empty:
            return float("nan")
        prior_high = float(sub["high"].iloc[:-1].max()) if len(sub) > 1 else close
        return (prior_high - close) / close

    atr = float(last.get("atr", float("nan")))
    feats = {
        "move_pct": event.move_pct,
        "run_up_bars": event.run_up_bars,
        "rsi_1h": float(last.get("rsi", float("nan"))),
        "rsi_15m": _rsi_at(df_15m, event.ts) if not df_15m.empty else float("nan"),
        "volume_spike_1h": float(last.get("volume_spike", float("nan"))),
        "atr_pct_1h": atr / close if close and np.isfinite(atr) else float("nan"),
        "change_24h": pct_change(24),
        "change_7d": pct_change(24 * 7),
        "change_30d": pct_change(24 * 30),
        "headroom_7d": headroom(24 * 7),
        "headroom_30d": headroom(24 * 30),
        "ema20_dist": (close - float(last.get("ema20", close))) / close,
        "ema50_dist": (close - float(last.get("ema50", close))) / close,
        "obv_slope": float(enriched["obv"].tail(6).diff().mean()) if "obv" in enriched else float("nan"),
        "hour_utc": int(pd.to_datetime(event.ts, unit="s", utc=True).hour),
    }

    # 4h RSI from resampled hourly bars.
    idx = pd.to_datetime(hist["time"], unit="s", utc=True)
    h4 = (
        hist.set_index(idx)
        .resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    if len(h4) >= 20:
        h4 = h4.reset_index(drop=True)
        feats["rsi_4h"] = float(compute_indicators(h4).iloc[-1].get("rsi", float("nan")))
    else:
        feats["rsi_4h"] = float("nan")

    if not funding.empty:
        prior = funding[funding["time"] <= event.ts]
        feats["funding_rate"] = float(prior["funding_rate"].iloc[-1]) if len(prior) else float("nan")
        feats["funding_mean_3d"] = float(prior["funding_rate"].tail(9).mean()) if len(prior) else float("nan")
    else:
        feats["funding_rate"] = float("nan")
        feats["funding_mean_3d"] = float("nan")

    # --- chart structure: the techniques used manually in the reference channel ---
    levels = find_horizontal_levels(enriched)
    overhead = nearest_level_above(levels, close)
    feats["level_dist"] = float((overhead.price - close) / close) if overhead else float("nan")
    feats["level_strength"] = float(overhead.strength) if overhead else 0.0
    feats["level_touches"] = float(overhead.touches) if overhead else 0.0
    feats["level_untouched_bars"] = float(overhead.last_touch_bars_ago) if overhead else float("nan")
    feats["level_count_above"] = float(sum(1 for lv in levels if lv.price > close))

    feats["fib_618_dist"] = distance_to_fib(hist, close, ratio="fib_618")
    feats["fib_500_dist"] = distance_to_fib(hist, close, ratio="fib_500")

    m15 = df_15m[df_15m["time"] <= event.ts] if not df_15m.empty else pd.DataFrame()
    h4 = df_4h[df_4h["time"] <= event.ts] if df_4h is not None and not df_4h.empty else pd.DataFrame()
    feats.update(multi_timeframe_confluence({"15m": m15.tail(400), "1h": hist.tail(400), "4h": h4.tail(400)}, close))

    feats.update(rsi_divergence(enriched))
    feats.update(estimate_liquidation_map(hist, close))

    bench = benchmark_1h[benchmark_1h["time"] <= event.ts] if benchmark_1h is not None and not benchmark_1h.empty else None
    feats.update(build_pump_features(enriched, benchmark=bench))

    return feats


def build_symbol_rows(
    symbol: str,
    collector: HistoryCollector,
    start_ts: int,
    end_ts: int,
    event_cfg: EventConfig,
    label_cfg: LabelConfig,
    benchmark_1h: pd.DataFrame | None = None,
) -> list[dict]:
    df_1h = collector.fetch_range(symbol, "Min60", start_ts, end_ts).reset_index()
    if df_1h.empty or len(df_1h) < 100:
        return []
    df_15m = collector.fetch_range(symbol, "Min15", start_ts, end_ts).reset_index()
    df_5m = collector.fetch_range(symbol, "Min5", start_ts, end_ts).reset_index()
    df_4h = collector.fetch_range(symbol, "Hour4", start_ts, end_ts).reset_index()
    funding = collector.fetch_funding_history(symbol, pages=15)

    rows: list[dict] = []
    horizon = label_cfg.horizon_hours * 3600
    for ev in detect_events(df_1h, event_cfg):
        ev.symbol = symbol
        if ev.ts + horizon > end_ts:
            continue  # not enough forward data to label honestly
        fwd_src = df_5m if not df_5m.empty else df_15m
        fwd = fwd_src[(fwd_src["time"] >= ev.ts) & (fwd_src["time"] <= ev.ts + horizon)]
        if len(fwd) < 10:
            continue
        feats = build_features(ev, df_1h, df_15m, funding, df_4h=df_4h, benchmark_1h=benchmark_1h)
        if not feats:
            continue
        labels = label_event(ev, fwd, label_cfg)
        if not labels:
            continue
        rows.append({"symbol": symbol, "ts": ev.ts, "entry": ev.entry, **feats, **labels})
    return rows
