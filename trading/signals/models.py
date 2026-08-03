from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from trading.signals.scoring import safe_float


_MARKET_EXTRA_KEYS: tuple[str, ...] = (
    "mtf_rsi_5m",
    "mtf_rsi_15m",
    "mtf_rsi_1h",
    "mtf_atr_norm_5m",
    "mtf_atr_norm_15m",
    "mtf_atr_norm_1h",
    "mtf_trend_5m",
    "mtf_trend_15m",
    "mtf_trend_1h",
    "mtf_ready_5m",
    "mtf_ready_15m",
    "mtf_ready_1h",
    "mtf_native_15m",
    "mtf_native_1h",
    "volume_spike",
    "vwap_dist",
    "atr_norm",
    "rsi",
    "adx",
    "bb_position",
    "bb_upper",
    "kc_upper",
    "ema20",
    "ema50",
    "spread_bps",
    "expected_slippage_bps",
    "orderbook_expected_slippage_bps",
    "depth_ratio",
    "orderbook_depth_ratio",
    "bid_ask_imbalance",
    "aggressor_exhaustion",
)


@dataclass(frozen=True)
class SignalCandidate:
    """Normalized candidate passed from signal generation into admission checks."""

    signal_id: str
    symbol: str
    side: str
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float
    timeframe: str
    mark_price: float
    created_at: float
    details: Mapping[str, Any] = field(default_factory=dict)
    trace: Mapping[str, Any] = field(default_factory=dict)
    latest_atr: float = 0.0
    latest_open: float = 0.0
    latest_high: float = 0.0
    latest_low: float = 0.0
    latest_close: float = 0.0
    recent_high: float = 0.0
    recent_low: float = 0.0
    market_extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_signal(
        cls,
        *,
        signal: Any,
        context: Any,
        enriched: Any,
        trace_meta: Mapping[str, Any] | None = None,
    ) -> "SignalCandidate":
        details = signal.details if isinstance(getattr(signal, "details", None), Mapping) else {}
        trace = {}
        if isinstance(trace_meta, Mapping):
            trace = trace_meta.get("layer_trace", {})
            if not isinstance(trace, Mapping):
                trace = {}

        latest = enriched.iloc[-1] if getattr(enriched, "empty", True) is False else {}
        recent = enriched.tail(8) if getattr(enriched, "empty", True) is False else enriched
        recent_high = safe_float(getattr(recent["high"], "max", lambda: 0.0)()) if "high" in enriched else 0.0
        recent_low = safe_float(getattr(recent["low"], "min", lambda: 0.0)()) if "low" in enriched else 0.0
        mark_price = safe_float(getattr(context, "mark_price", 0.0), 0.0)
        if mark_price <= 0:
            mark_price = safe_float(getattr(signal, "entry", 0.0), 0.0)

        market_extras: dict[str, Any] = {}
        if hasattr(latest, "get"):
            for key in _MARKET_EXTRA_KEYS:
                if key in latest:
                    market_extras[key] = safe_float(latest.get(key), 0.0)
        for key in _MARKET_EXTRA_KEYS:
            if key in market_extras:
                continue
            value = getattr(context, key, None)
            if value is not None:
                market_extras[key] = safe_float(value, 0.0)

        return cls(
            signal_id=str(getattr(signal, "signal_id", "")),
            symbol=str(getattr(context, "symbol", getattr(signal, "symbol", ""))).replace("/", "").upper(),
            side=str(getattr(signal, "side", "")).upper(),
            entry=safe_float(getattr(signal, "entry", 0.0), 0.0),
            stop_loss=safe_float(getattr(signal, "sl", 0.0), 0.0),
            take_profit=safe_float(getattr(signal, "tp", 0.0), 0.0),
            confidence=safe_float(getattr(signal, "confidence", 0.0), 0.0),
            timeframe=str(getattr(context, "timeframe", "1m") or "1m"),
            mark_price=mark_price,
            created_at=safe_float(getattr(signal, "created_at", 0.0), time.time()),
            details=details,
            trace=trace,
            latest_atr=safe_float(latest.get("atr") if hasattr(latest, "get") else 0.0, 0.0),
            latest_open=safe_float(latest.get("open") if hasattr(latest, "get") else 0.0, 0.0),
            latest_high=safe_float(latest.get("high") if hasattr(latest, "get") else 0.0, 0.0),
            latest_low=safe_float(latest.get("low") if hasattr(latest, "get") else 0.0, 0.0),
            latest_close=safe_float(latest.get("close") if hasattr(latest, "get") else 0.0, 0.0),
            recent_high=recent_high,
            recent_low=recent_low,
            market_extras=market_extras,
        )
