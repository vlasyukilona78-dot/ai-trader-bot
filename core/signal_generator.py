from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .market_regime import MarketRegime
from .volume_profile import VolumeProfileLevels


@dataclass
class SignalConfig:
    rsi_high: float = 75.0
    rsi_low: float = 25.0
    volume_spike_threshold: float = 2.5
    weakness_lookback: int = 4
    sentiment_bullish_threshold: float = 68.0
    sentiment_bearish_threshold: float = 32.0
    risk_reward: float = 1.6
    atr_sl_mult: float = 1.0
    entry_tolerance_pct: float = 0.0015
    vwap_tolerance_pct: float = 0.0025
    funding_tolerance: float = 0.0003
    long_short_ratio_tolerance: float = 0.10
    msb_lookback: int = 20
    msb_recent_bars: int = 6
    msb_break_buffer_pct: float = 0.0005
    confirmation_enabled: bool = True
    confirmation_max_wait_bars: int = 3
    confirmation_invalidate_pct: float = 0.0015


@dataclass
class SignalContext:
    symbol: str
    df: pd.DataFrame
    volume_profile: VolumeProfileLevels | None
    regime: MarketRegime
    sentiment_index: float | None
    sentiment_source: str | None
    funding_rate: float | None
    long_short_ratio: float | None


@dataclass
class SignalResult:
    signal_id: str
    symbol: str
    side: str
    entry: float
    sl: float
    tp: float
    partial_tps: list[float] = field(default_factory=list)
    confidence: float = 0.0
    strategy: str = "layered_pump_panic"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    features: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingCandidate:
    symbol: str
    side: str
    armed_bar_ts: Any
    armed_close: float
    last_seen_bar_ts: Any
    bars_waited: int = 0
    layer_snapshot: dict = field(default_factory=dict)


class SignalGenerator:
    def __init__(self, config: SignalConfig | None = None):
        self.config = config or SignalConfig()
        self.last_diagnostics: dict[str, Any] = {}
        self._pending: dict[str, PendingCandidate] = {}

    @staticmethod
    def _safe(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _layer1_pump_detection(self, df: pd.DataFrame) -> tuple[str | None, dict[str, float | str]]:
        """Layer 1: Pump/Panic detection via RSI + Volume + Bollinger/Keltner breakout."""
        last = df.iloc[-1]
        metrics: dict[str, float | str] = {
            "rsi": self._safe(last.get("rsi"), 50.0),
            "volume_spike": self._safe(last.get("volume_spike"), 1.0),
            "close": self._safe(last.get("close"), 0.0),
            "bb_upper": self._safe(last.get("bb_upper"), np.inf),
            "bb_lower": self._safe(last.get("bb_lower"), -np.inf),
            "kc_upper": self._safe(last.get("kc_upper"), np.inf),
            "kc_lower": self._safe(last.get("kc_lower"), -np.inf),
        }

        bb_breakout_up = float(metrics["close"]) > float(metrics["bb_upper"])
        kc_breakout_up = float(metrics["close"]) > float(metrics["kc_upper"])
        band_up = bb_breakout_up or kc_breakout_up

        bb_breakout_down = float(metrics["close"]) < float(metrics["bb_lower"])
        kc_breakout_down = float(metrics["close"]) < float(metrics["kc_lower"])
        band_down = bb_breakout_down or kc_breakout_down

        pump_points = 0
        if float(metrics["rsi"]) >= self.config.rsi_high:
            pump_points += 1
        if float(metrics["volume_spike"]) >= self.config.volume_spike_threshold:
            pump_points += 1
        if band_up:
            pump_points += 1

        panic_points = 0
        if float(metrics["rsi"]) <= self.config.rsi_low:
            panic_points += 1
        if float(metrics["volume_spike"]) >= self.config.volume_spike_threshold:
            panic_points += 1
        if band_down:
            panic_points += 1

        pump = band_up and pump_points >= 3
        panic = band_down and panic_points >= 3

        metrics.update(
            {
                "rsi_threshold": float(self.config.rsi_high),
                "volume_spike_threshold": float(self.config.volume_spike_threshold),
                "bb_breakout_up": 1.0 if bb_breakout_up else 0.0,
                "kc_breakout_up": 1.0 if kc_breakout_up else 0.0,
                "bb_breakout_down": 1.0 if bb_breakout_down else 0.0,
                "kc_breakout_down": 1.0 if kc_breakout_down else 0.0,
                "pump_points": float(pump_points),
                "panic_points": float(panic_points),
                "pump_detected": 1.0 if pump else 0.0,
                "panic_detected": 1.0 if panic else 0.0,
            }
        )

        if pump:
            return "SHORT", metrics
        if panic:
            return "LONG", metrics
        return None, metrics

    def _layer2_weakness_confirmation(self, df: pd.DataFrame, side: str) -> tuple[bool, dict[str, float]]:
        """Layer 2: Weakness confirmation via price-vs-OBV/CVD divergence."""
        lookback = self.config.weakness_lookback
        if len(df) < lookback + 2:
            return False, {"reason": 1.0, "insufficient_history": 1.0}

        last = df.iloc[-1]
        ref = df.iloc[-1 - lookback]

        price_up = self._safe(last.get("close")) > self._safe(ref.get("close"))
        price_down = self._safe(last.get("close")) < self._safe(ref.get("close"))

        obv_down = self._safe(last.get("obv")) < self._safe(ref.get("obv"))
        obv_up = self._safe(last.get("obv")) > self._safe(ref.get("obv"))

        cvd_down = self._safe(last.get("cvd")) < self._safe(ref.get("cvd"))
        cvd_up = self._safe(last.get("cvd")) > self._safe(ref.get("cvd"))

        if side == "SHORT":
            ok = (price_up and (obv_down or cvd_down)) or (obv_down and cvd_down)
        else:
            ok = (price_down and (obv_up or cvd_up)) or (obv_up and cvd_up)

        details = {
            "price_up": 1.0 if price_up else 0.0,
            "price_down": 1.0 if price_down else 0.0,
            "obv_down": 1.0 if obv_down else 0.0,
            "obv_up": 1.0 if obv_up else 0.0,
            "cvd_down": 1.0 if cvd_down else 0.0,
            "cvd_up": 1.0 if cvd_up else 0.0,
            "lookback": float(lookback),
        }
        return ok, details

    def _layer3_msb_confirmation(self, df: pd.DataFrame, side: str) -> tuple[bool, dict[str, float]]:
        lookback = max(5, int(self.config.msb_lookback))
        recent_bars = max(1, int(self.config.msb_recent_bars))
        break_buf = max(0.0, float(self.config.msb_break_buffer_pct))

        need = lookback + recent_bars + 2
        if len(df) < need:
            return False, {
                "msb_missing": 1.0,
                "msb_lookback": float(lookback),
                "msb_recent_bars": float(recent_bars),
                "msb_break_buffer_pct": break_buf,
            }

        work = df.tail(need)
        close = pd.to_numeric(work["close"], errors="coerce")
        high = pd.to_numeric(work["high"], errors="coerce")
        low = pd.to_numeric(work["low"], errors="coerce")

        prior_low = low.rolling(lookback, min_periods=lookback).min().shift(1)
        prior_high = high.rolling(lookback, min_periods=lookback).max().shift(1)

        msb_down_struct = (close < prior_low * (1.0 - break_buf)).fillna(False)
        msb_up_struct = (close > prior_high * (1.0 + break_buf)).fillna(False)

        if "ema20" in work.columns:
            ema20 = pd.to_numeric(work["ema20"], errors="coerce")
            msb_down_cross = ((close < ema20) & (close.shift(1) >= ema20.shift(1))).fillna(False)
            msb_up_cross = ((close > ema20) & (close.shift(1) <= ema20.shift(1))).fillna(False)
        else:
            msb_down_cross = pd.Series(False, index=work.index)
            msb_up_cross = pd.Series(False, index=work.index)

        msb_down_recent = bool(msb_down_struct.tail(recent_bars).any() or msb_down_cross.tail(recent_bars).any())
        msb_up_recent = bool(msb_up_struct.tail(recent_bars).any() or msb_up_cross.tail(recent_bars).any())

        ok = msb_down_recent if side == "SHORT" else msb_up_recent
        details = {
            "msb_ok": 1.0 if ok else 0.0,
            "msb_down_recent": 1.0 if msb_down_recent else 0.0,
            "msb_up_recent": 1.0 if msb_up_recent else 0.0,
            "msb_struct_break_down": 1.0 if msb_down_struct.tail(recent_bars).any() else 0.0,
            "msb_struct_break_up": 1.0 if msb_up_struct.tail(recent_bars).any() else 0.0,
            "msb_ema_cross_down": 1.0 if msb_down_cross.tail(recent_bars).any() else 0.0,
            "msb_ema_cross_up": 1.0 if msb_up_cross.tail(recent_bars).any() else 0.0,
            "msb_lookback": float(lookback),
            "msb_recent_bars": float(recent_bars),
            "msb_break_buffer_pct": break_buf,
        }
        return ok, details

    def _layer3_entry_location(self, df: pd.DataFrame, side: str, vp: VolumeProfileLevels | None) -> tuple[bool, dict[str, float]]:
        """Layer 3: Entry location via Volume Profile levels + MSB."""
        if vp is None or len(df) < 2:
            return False, {"vp_missing": 1.0}

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = self._safe(last.get("close"))
        prev_close = self._safe(prev.get("close"))
        tol = max(0.0, self.config.entry_tolerance_pct)

        if side == "SHORT":
            entry_ok = prev_close >= vp.vah * (1.0 - tol) and close <= vp.vah * (1.0 + tol)
        else:
            entry_ok = prev_close <= vp.val * (1.0 + tol) and close >= vp.val * (1.0 - tol)

        msb_ok, msb = self._layer3_msb_confirmation(df=df, side=side)
        ok = entry_ok and msb_ok

        details = {
            "close": close,
            "prev_close": prev_close,
            "poc": vp.poc,
            "vah": vp.vah,
            "val": vp.val,
            "entry_tolerance_pct": tol,
            "entry_ok": 1.0 if entry_ok else 0.0,
            "vp_levels_available": 1.0,
            "entry_reference": 1.0 if side == "SHORT" else -1.0,
        }
        details.update(msb)
        return ok, details

    def _layer4_fake_filter(
        self,
        df: pd.DataFrame,
        side: str,
        sentiment_index: float | None,
        sentiment_source: str | None,
        funding_rate: float | None,
        long_short_ratio: float | None,
    ) -> tuple[bool, dict[str, float | str]]:
        """Layer 4: Fake-signal filter via Sentiment + VWAP with explicit graceful fallback."""
        last = df.iloc[-1]
        close = self._safe(last.get("close"))
        vwap = self._safe(last.get("vwap"), close)

        sentiment_missing = sentiment_index is None
        sentiment = 50.0 if sentiment_missing else float(sentiment_index)

        source = (sentiment_source or "").strip().lower()
        if not source:
            source = "fallback_neutral_50" if sentiment_missing else "provided"

        source_unavailable = 1.0 if source in ("unavailable", "missing", "none") else 0.0
        fallback_used = 1.0 if (sentiment_missing or source.startswith("fallback")) else 0.0

        vwap_tol = max(0.0, self.config.vwap_tolerance_pct)
        funding_tol = max(0.0, self.config.funding_tolerance)
        ratio_tol = max(0.0, self.config.long_short_ratio_tolerance)

        if side == "SHORT":
            crowd_not_against = sentiment >= self.config.sentiment_bearish_threshold
            crowd_extreme = sentiment >= self.config.sentiment_bullish_threshold
            ok = crowd_not_against and close >= vwap * (1.0 - vwap_tol)
            if funding_rate is not None:
                ok = ok and funding_rate >= -funding_tol
            if long_short_ratio is not None:
                ok = ok and long_short_ratio >= (1.0 - ratio_tol)
        else:
            crowd_not_against = sentiment <= self.config.sentiment_bullish_threshold
            crowd_extreme = sentiment <= self.config.sentiment_bearish_threshold
            ok = crowd_not_against and close <= vwap * (1.0 + vwap_tol)
            if funding_rate is not None:
                ok = ok and funding_rate <= funding_tol
            if long_short_ratio is not None:
                ok = ok and long_short_ratio <= (1.0 + ratio_tol)

        return ok, {
            "close": close,
            "vwap": vwap,
            "sentiment": float(sentiment),
            "crowd_not_against": 1.0 if crowd_not_against else 0.0,
            "crowd_extreme": 1.0 if crowd_extreme else 0.0,
            "funding_rate": float(funding_rate) if funding_rate is not None else 0.0,
            "long_short_ratio": float(long_short_ratio) if long_short_ratio is not None else 0.0,
            "vwap_tolerance_pct": vwap_tol,
            "funding_tolerance": funding_tol,
            "ratio_tolerance": ratio_tol,
            "sentiment_fallback_used": fallback_used,
            "sentiment_source_unavailable": source_unavailable,
            "degraded_mode": 1.0 if (fallback_used or source_unavailable) else 0.0,
            "sentiment_source": source,
        }

    def _layer5_tp_sl_levels(self, df: pd.DataFrame, side: str, vp: VolumeProfileLevels | None) -> tuple[float, float, list[float]]:
        """Layer 5: TP/SL levels via ATR + Volume Profile (with RR fallback)."""
        last = df.iloc[-1]
        close = self._safe(last.get("close"))
        atr = self._safe(last.get("atr"), close * 0.01)
        if atr <= 0:
            atr = close * 0.01

        if vp is None:
            if side == "SHORT":
                sl = close + atr * self.config.atr_sl_mult
                tp = close - atr * self.config.risk_reward
                partial = [close - atr]
            else:
                sl = close - atr * self.config.atr_sl_mult
                tp = close + atr * self.config.risk_reward
                partial = [close + atr]
            return tp, sl, partial

        if side == "SHORT":
            sl = max(vp.vah, close + atr * self.config.atr_sl_mult)
            tp = vp.poc if vp.poc < close else close - (sl - close) * self.config.risk_reward
            partial = [close - atr, (close + tp) / 2.0]
        else:
            sl = min(vp.val, close - atr * self.config.atr_sl_mult)
            tp = vp.poc if vp.poc > close else close + (close - sl) * self.config.risk_reward
            partial = [close + atr, (close + tp) / 2.0]

        return float(tp), float(sl), [float(x) for x in partial]

    @staticmethod
    def _normalize_levels(entry: float, tp: float, sl: float, side: str) -> tuple[float, float]:
        min_step = max(entry * 0.0001, 1e-8)
        if side == "SHORT":
            if tp >= entry - min_step:
                tp = entry - min_step
            if sl <= entry + min_step:
                sl = entry + min_step
        else:
            if tp <= entry + min_step:
                tp = entry + min_step
            if sl >= entry - min_step:
                sl = entry - min_step
        return float(tp), float(sl)

    def _evaluate_gates(
        self, df: pd.DataFrame, context: SignalContext, trace: dict[str, Any]
    ) -> tuple[str, dict, dict, dict, dict] | None:
        """Run layers 1-4. Returns (side, layer1, layer2, layer3, layer4) on full pass, else None."""
        side, layer1 = self._layer1_pump_detection(df)
        trace["layers"]["layer1_pump_detection"] = {"passed": side is not None, "side": side or "", "details": layer1}
        if side is None:
            trace["failed_layer"] = "layer1_pump_detection"
            return None

        layer2_ok, layer2 = self._layer2_weakness_confirmation(df, side)
        trace["layers"]["layer2_weakness_confirmation"] = {"passed": layer2_ok, "details": layer2}
        if not layer2_ok:
            trace["failed_layer"] = "layer2_weakness_confirmation"
            return None

        layer3_ok, layer3 = self._layer3_entry_location(df, side, context.volume_profile)
        trace["layers"]["layer3_entry_location"] = {"passed": layer3_ok, "details": layer3}
        if not layer3_ok:
            trace["failed_layer"] = "layer3_entry_location"
            return None

        layer4_ok, layer4 = self._layer4_fake_filter(
            df=df,
            side=side,
            sentiment_index=context.sentiment_index,
            sentiment_source=context.sentiment_source,
            funding_rate=context.funding_rate,
            long_short_ratio=context.long_short_ratio,
        )
        trace["layers"]["layer4_fake_filter"] = {"passed": layer4_ok, "details": layer4}
        if not layer4_ok:
            trace["failed_layer"] = "layer4_fake_filter"
            return None

        return side, layer1, layer2, layer3, layer4

    def _finalize_signal(
        self,
        df: pd.DataFrame,
        context: SignalContext,
        side: str,
        layer1: dict,
        layer2: dict,
        layer3: dict,
        layer4: dict,
        trace: dict[str, Any],
        entry: float | None = None,
    ) -> SignalResult:
        entry = float(entry if entry is not None else df.iloc[-1]["close"])
        tp, sl, partial_tps = self._layer5_tp_sl_levels(df, side, context.volume_profile)
        tp, sl = self._normalize_levels(entry=entry, tp=tp, sl=sl, side=side)

        layer5 = {
            "entry": float(entry),
            "tp": float(tp),
            "sl": float(sl),
            "partial_tps": [float(x) for x in partial_tps],
            "atr_sl_mult": float(self.config.atr_sl_mult),
            "risk_reward": float(self.config.risk_reward),
            "vp_available": 1.0 if context.volume_profile is not None else 0.0,
        }
        trace["layers"]["layer5_tp_sl"] = {"passed": True, "details": layer5}
        self.last_diagnostics = trace

        confidence = 0.45
        confidence += 0.20 * min(float(layer1.get("volume_spike", 1.0)) / self.config.volume_spike_threshold, 2.0)
        confidence += 0.10 * abs(float(layer4.get("sentiment", 50.0)) - 50.0) / 50.0
        confidence += 0.10 * float(layer4.get("crowd_extreme", 0.0))
        confidence += 0.20 if context.regime in (MarketRegime.PUMP, MarketRegime.PANIC, MarketRegime.TREND) else 0.05
        confidence = float(max(0.0, min(confidence, 0.99)))

        signal_id = f"{context.symbol.replace('/', '')}-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        return SignalResult(
            signal_id=signal_id,
            symbol=context.symbol,
            side=side,
            entry=entry,
            sl=sl,
            tp=tp,
            partial_tps=partial_tps,
            confidence=confidence,
            details={
                "layer1": layer1,
                "layer2": layer2,
                "layer3": layer3,
                "layer4": layer4,
                "layer5": layer5,
                "layer_trace": trace,
                "regime": context.regime.value,
            },
        )

    def generate(self, context: SignalContext) -> SignalResult | None:
        df = context.df
        trace: dict[str, Any] = {
            "strategy_model": "layered_table_5_softened",
            "failed_layer": None,
            "layers": {},
        }

        if df.empty or len(df) < 40:
            trace["failed_layer"] = "layer0_input"
            trace["layers"]["layer0_input"] = {"passed": False, "details": {"insufficient_history": 1.0}}
            self.last_diagnostics = trace
            return None

        if not self.config.confirmation_enabled:
            gates = self._evaluate_gates(df, context, trace)
            if gates is None:
                self.last_diagnostics = trace
                return None
            side, layer1, layer2, layer3, layer4 = gates
            return self._finalize_signal(df, context, side, layer1, layer2, layer3, layer4, trace)

        return self._generate_with_confirmation(df, context, trace)

    def _generate_with_confirmation(
        self, df: pd.DataFrame, context: SignalContext, trace: dict[str, Any]
    ) -> SignalResult | None:
        bar_ts = df.index[-1]
        pending = self._pending.get(context.symbol)

        if pending is not None:
            if bar_ts == pending.last_seen_bar_ts:
                trace["failed_layer"] = "layer_confirmation_pending"
                trace["layers"]["layer_confirmation"] = {
                    "passed": False,
                    "details": {"status": "same_bar", "bars_waited": float(pending.bars_waited)},
                }
                self.last_diagnostics = trace
                return None

            close_now = float(df.iloc[-1]["close"])
            pending.last_seen_bar_ts = bar_ts

            if pending.side == "LONG":
                confirmed = close_now > pending.armed_close
                invalidated = close_now <= pending.armed_close * (1.0 - self.config.confirmation_invalidate_pct)
            else:
                confirmed = close_now < pending.armed_close
                invalidated = close_now >= pending.armed_close * (1.0 + self.config.confirmation_invalidate_pct)

            if invalidated:
                del self._pending[context.symbol]
                trace["failed_layer"] = "layer_confirmation_invalidated"
                trace["layers"]["layer_confirmation"] = {
                    "passed": False,
                    "details": {
                        "status": "invalidated",
                        "armed_close": float(pending.armed_close),
                        "close_now": close_now,
                        "bars_waited": float(pending.bars_waited),
                    },
                }
                self.last_diagnostics = trace
                return None

            if confirmed:
                del self._pending[context.symbol]
                side = pending.side
                layer1 = pending.layer_snapshot["layer1"]
                layer2 = pending.layer_snapshot["layer2"]
                layer3 = pending.layer_snapshot["layer3"]
                layer4 = pending.layer_snapshot["layer4"]
                trace["layers"]["layer1_pump_detection"] = {"passed": True, "side": side, "details": layer1}
                trace["layers"]["layer2_weakness_confirmation"] = {"passed": True, "details": layer2}
                trace["layers"]["layer3_entry_location"] = {"passed": True, "details": layer3}
                trace["layers"]["layer4_fake_filter"] = {"passed": True, "details": layer4}
                trace["layers"]["layer_confirmation"] = {
                    "passed": True,
                    "details": {
                        "status": "confirmed",
                        "armed_bar_ts": str(pending.armed_bar_ts),
                        "armed_close": float(pending.armed_close),
                        "confirmed_bar_ts": str(bar_ts),
                        "confirmed_close": close_now,
                        "bars_waited": float(pending.bars_waited),
                    },
                }
                return self._finalize_signal(df, context, side, layer1, layer2, layer3, layer4, trace, entry=close_now)

            pending.bars_waited += 1
            if pending.bars_waited >= self.config.confirmation_max_wait_bars:
                del self._pending[context.symbol]
                trace["failed_layer"] = "layer_confirmation_expired"
                trace["layers"]["layer_confirmation"] = {
                    "passed": False,
                    "details": {"status": "expired", "bars_waited": float(pending.bars_waited)},
                }
                self.last_diagnostics = trace
                return None

            trace["failed_layer"] = "layer_confirmation_pending"
            trace["layers"]["layer_confirmation"] = {
                "passed": False,
                "details": {"status": "waiting", "bars_waited": float(pending.bars_waited)},
            }
            self.last_diagnostics = trace
            return None

        gates = self._evaluate_gates(df, context, trace)
        if gates is None:
            self.last_diagnostics = trace
            return None

        side, layer1, layer2, layer3, layer4 = gates
        armed_close = float(df.iloc[-1]["close"])
        self._pending[context.symbol] = PendingCandidate(
            symbol=context.symbol,
            side=side,
            armed_bar_ts=bar_ts,
            armed_close=armed_close,
            last_seen_bar_ts=bar_ts,
            bars_waited=0,
            layer_snapshot={"layer1": layer1, "layer2": layer2, "layer3": layer3, "layer4": layer4},
        )
        trace["failed_layer"] = "layer_confirmation_pending"
        trace["layers"]["layer_confirmation"] = {
            "passed": False,
            "details": {"status": "armed", "armed_close": armed_close, "bars_waited": 0.0},
        }
        self.last_diagnostics = trace
        return None
