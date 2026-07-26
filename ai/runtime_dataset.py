"""Build the labelled dataset from the *runtime* signal pipeline.

The existing builder detects an event as any 5% low-to-close move on an hourly
bar. The bot emits nothing of the sort: it requires a band breakout, RSI and
volume conditions, a peak already behind price, a bounded retrace, four quality
gates and a confirmation bar. Measuring one and shipping the other means the
numbers describe a population the bot never trades.

This walks each symbol's history bar by bar and asks the real
LayeredPumpStrategy what it would have emitted, so a row exists only where the
bot would actually have fired. It is slower than the shortcut, and that is the
point.

Two differences from live are documented rather than hidden:

- the cross-sectional volatility floor needs every symbol at the same instant,
  which an offline per-symbol pass cannot provide, so the fixed floor is used;
- sentiment is held at the neutral fallback, as the live loop does when no feed
  is configured.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai.pump_dataset import BAR_SECONDS_1H, LabelConfig, _closed_by, forward_window_quality, label_event
from core.indicators import compute_indicators
from core.signal_generator import SignalConfig
from trading.market_data.history import HistoryCollector
from trading.signals.layered_strategy import LayeredPumpStrategy
from trading.signals.signal_types import IntentAction
from trading.signals.strategy_interface import StrategyContext
from trading.state.models import TradeState


def calibration_config() -> SignalConfig:
    """A permissive config for building the population the gates are fitted on.

    A gate cannot be calibrated on the population that survives it. With shipping
    thresholds the replay emits nothing, so there is no sample to measure against;
    opening the gates yields the candidate population, each row still carrying what
    every gate measured, and a threshold can then be chosen by how it separates
    forward outcomes.

    This is a measurement instrument, never a trading configuration - it deliberately
    removes the risk limits, so nothing that reads it should reach an exchange.
    """
    return SignalConfig(
        # gates under calibration: measured on every row, enforced on none
        min_atr_pct=0.0,
        min_hourly_usd_volume=0.0,
        pump_entry_max_dist_from_peak_pct=0.25,
        min_relative_strength=0.0,
        require_htf=False,
        min_rsi_4h=0.0,
        require_level_overhead=False,
        # risk limits would truncate the sample before it can be measured
        max_stop_distance_pct=0.0,
        min_risk_reward=0.0,
        # confirmation is a separate question and is evaluated on its own once the
        # location gates are settled; here it would silently drop the sample again
        confirmation_enabled=False,
    )


@dataclass
class RuntimeEvent:
    symbol: str
    ts: int
    decision_ts: int
    entry: float
    side: str
    stop: float
    target: float
    diagnostics: dict


class _StaticHtf:
    """Serves point-in-time higher-timeframe bars from an already-fetched frame."""

    def __init__(self, frame: pd.DataFrame, bar_seconds: int = 14400):
        self.frame = frame
        self.bar_seconds = bar_seconds
        self._decision_ts = 0

    def at(self, decision_ts: int):
        self._decision_ts = decision_ts
        return self

    def get(self, symbol: str, **_):
        return _closed_by(self.frame, self.bar_seconds, self._decision_ts)


# What each gate measured, carried onto the row so a threshold can be chosen by
# comparing it against forward outcomes on this population. Fitting a threshold on
# one population and applying it to another is what put min_atr_pct at the 96th
# percentile of the bars it actually sees.
_GATE_FEATURES: dict[str, tuple[str, ...]] = {
    "layer1_pump_detection": ("run_up_pct", "drop_pct", "bars_since_peak", "retrace_from_high",
                              "rsi", "volume_spike", "pump_event_bars"),
    "layer1b_quality_gate": ("atr_pct", "usd_volume_recent"),
    "layer1c_market_context": ("relative_strength", "rsi_htf", "level_dist_pct"),
    "layer3_entry_location": ("dist_from_extreme_pct", "msb_confirmed"),
    "layer4_fake_filter": ("sentiment_index", "vwap_dist_pct"),
}


def _gate_features(layers: dict) -> dict[str, float]:
    """Flatten the gate measurements out of the layer trace."""
    out: dict[str, float] = {}
    for layer, fields in _GATE_FEATURES.items():
        details = layers.get(layer, {}).get("details", {})
        for field in fields:
            value = details.get(field)
            if isinstance(value, (int, float)):
                out[f"{layer.split('_')[0]}_{field}"] = float(value)
    return out


def replay_runtime_signals(
    symbol: str,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    benchmark_1h: pd.DataFrame | None,
    config: SignalConfig | None = None,
    warmup_bars: int = 120,
) -> list[RuntimeEvent]:
    """Walk the history and collect what the live strategy would have emitted."""
    if df_1h.empty or len(df_1h) < warmup_bars + 10:
        return []

    enriched = compute_indicators(df_1h).reset_index(drop=True)
    enriched["time"] = df_1h["time"].to_numpy()

    cfg = config or SignalConfig()
    strategy = LayeredPumpStrategy(cfg)
    htf = _StaticHtf(df_4h)
    strategy.set_htf_cache(htf)

    events: list[RuntimeEvent] = []
    for i in range(warmup_bars, len(enriched)):
        bar_ts = int(enriched["time"].iloc[i])
        decision_ts = bar_ts + BAR_SECONDS_1H

        window = enriched.iloc[: i + 1]
        htf.at(decision_ts)
        if benchmark_1h is not None and not benchmark_1h.empty:
            strategy.set_benchmark(_closed_by(benchmark_1h, BAR_SECONDS_1H, decision_ts))

        intent = strategy.generate(
            StrategyContext(
                symbol=symbol,
                market_ohlcv=window,
                mark_price=float(window.iloc[-1]["close"]),
                exchange=None,
                synced_state=TradeState.FLAT,
                sentiment_index=50.0,
                sentiment_source="fallback_neutral_50",
            )
        )
        if intent.action not in (IntentAction.SHORT_ENTRY, IntentAction.LONG_ENTRY):
            continue

        meta = intent.metadata if isinstance(intent.metadata, dict) else {}
        layers = meta.get("layer_trace", {}).get("layers", {})
        layer5 = layers.get("layer5_tp_sl", {}).get("details", {})

        diagnostics = {
            "stop_distance_pct": float(layer5.get("stop_distance_pct") or 0.0),
            "realized_risk_reward": float(layer5.get("realized_risk_reward") or 0.0),
            "max_safe_leverage": float(layer5.get("max_safe_leverage") or 0.0),
            "confidence": float(intent.confidence or 0.0),
        }
        diagnostics.update(_gate_features(layers))

        events.append(
            RuntimeEvent(
                symbol=symbol,
                ts=bar_ts,
                decision_ts=decision_ts,
                entry=float(layer5.get("entry") or window.iloc[-1]["close"]),
                side="SHORT" if intent.action is IntentAction.SHORT_ENTRY else "LONG",
                stop=float(layer5.get("sl") or 0.0),
                target=float(layer5.get("tp") or 0.0),
                diagnostics=diagnostics,
            )
        )
    return events


def build_runtime_rows(
    symbol: str,
    collector: HistoryCollector,
    start_ts: int,
    end_ts: int,
    label_cfg: LabelConfig | None = None,
    signal_cfg: SignalConfig | None = None,
    benchmark_1h: pd.DataFrame | None = None,
) -> list[dict]:
    label_cfg = label_cfg or LabelConfig()
    df_1h = collector.fetch_range(symbol, "Min60", start_ts, end_ts).reset_index()
    if df_1h.empty or len(df_1h) < 150:
        return []
    df_4h = collector.fetch_range(symbol, "Hour4", start_ts, end_ts).reset_index()
    df_5m = collector.fetch_range(symbol, "Min5", start_ts, end_ts).reset_index()
    if df_5m.empty:
        return []

    horizon = label_cfg.horizon_hours * 3600
    rows: list[dict] = []
    for ev in replay_runtime_signals(symbol, df_1h, df_4h, benchmark_1h, signal_cfg):
        if ev.decision_ts + horizon > end_ts:
            continue
        fwd = df_5m[(df_5m["time"] >= ev.decision_ts) & (df_5m["time"] < ev.decision_ts + horizon)]
        if len(fwd) < 10:
            continue
        quality = forward_window_quality(fwd, ev.decision_ts, horizon, 300)
        if (quality["coverage"] < label_cfg.min_forward_coverage
                or quality["max_gap_bars"] > label_cfg.max_forward_gap_bars):
            continue

        from ai.pump_dataset import PumpEvent

        labels = label_event(
            PumpEvent(symbol=symbol, ts=ev.ts, entry=ev.entry, move_pct=0.0, run_up_bars=0),
            fwd, label_cfg, decision_ts=ev.decision_ts,
        )
        if not labels:
            continue
        rows.append({
            "symbol": symbol, "ts": ev.ts, "decision_ts": ev.decision_ts,
            "entry": ev.entry, "side": ev.side, "stop": ev.stop, "target": ev.target,
            "fwd_coverage": quality["coverage"], "fwd_max_gap_bars": quality["max_gap_bars"],
            **ev.diagnostics, **labels,
        })
    return rows
