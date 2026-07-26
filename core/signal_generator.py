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
    # Layer 1 window mode: treat a pump as a recent *event* to fade, not as a
    # single-bar confluence. Set False to restore the legacy same-bar detector.
    pump_window_enabled: bool = True
    # 45 bars measured best on live MEXC alt data: a 3% run-up inside 20 minutes is
    # rare enough to starve the strategy, while widening to 60 bars with a 2% move
    # floods it with low-quality setups that lose money net of fees.
    pump_window_bars: int = 45
    pump_min_move_pct: float = 0.03
    pump_min_bars_since_peak: int = 1
    pump_max_retrace_pct: float = 0.5
    # Window mode entry/stop: anchor on the pump's own peak, not on VAH. The peak is
    # both the tightest stop and the cleanest invalidation (a new high = thesis wrong),
    # which is what makes an entry viable at 50-100x.
    pump_entry_max_dist_from_peak_pct: float = 0.015
    pump_stop_buffer_pct: float = 0.003
    # The stop sits beyond the structural level by whichever is larger: the flat
    # percentage above, or this multiple of ATR. Measured on the real pipeline, a
    # flat 0.3% left the stop inside hourly noise and invalidated nearly every
    # candidate before it could confirm.
    stop_buffer_atr_mult: float = 0.5
    # Anchor the swing on the higher timeframe. An hourly high is set by whatever
    # wick printed; a 4h swing high is a level the market respected.
    structural_anchor_htf: bool = True
    structural_anchor_htf_bars: int = 12
    # Caps loss per trade, not liquidation distance: on cross margin with a small
    # position relative to equity, liquidation sits far beyond any sane stop, so this
    # is a risk-budget knob rather than a survival one.
    max_stop_distance_pct: float = 0.03
    # Payoff ratio drives expected value - reject setups whose target is too near.
    min_risk_reward: float = 1.5
    # Leverage used only to express stop/target as a percentage of committed margin
    # in the signal payload; it does not affect detection.
    report_leverage: float = 100.0
    # Quality gates measured on 5523 labelled pump events over ~3 months of MEXC
    # history. Counter-intuitively, calm low-volatility pumps are the ones that
    # fail to resolve - violent pumps on volatile, liquid coins come back down
    # almost every time, so these are floors rather than ceilings.
    min_atr_pct: float = 0.046
    min_hourly_usd_volume: float = 100_000.0
    liquidity_lookback_bars: int = 12
    # Measured on 5042 labelled events: requiring the coin to have outrun BTC cut
    # the worst drawdown from 11.84 to 1.57 legs with expectancy unchanged, and
    # held out of sample. A move the whole market is making is beta, not an
    # engineered pump, and fading it is a different trade.
    min_relative_strength: float = 0.05
    relative_strength_lookback: int = 24
    # Optional and off by default: an overhead level cuts the tail further still
    # (0.77 legs) but more than halves the number of signals.
    require_level_overhead: bool = False
    min_level_dist_pct: float = 0.018
    # Layer 2 measured NEGATIVE on the same dataset (-0.0057 with divergence
    # versus +0.0245 without), so it is off unless explicitly re-enabled.
    weakness_layer_enabled: bool = False
    # Each indicator carries its signal on a different timeframe. The 4h RSI
    # separated outcomes where the 1h reading did not: requiring it above ~62 cut
    # the worst drawdown from 1.57 legs to 0.43 out of sample while keeping ~11
    # signals a day. 0 disables the check.
    min_rsi_4h: float = 61.6
    # When the higher-timeframe frame is missing the gate cannot be evaluated.
    # Blocking is the safe answer: a silently absent filter is worse than no
    # signal, because the signal still looks fully vetted.
    require_htf: bool = True
    # Same idea, tighter tail but fewer signals; opt-in.
    min_rsi_1h: float = 0.0
    # Multi-timeframe level confluence measured NEGATIVE across the population
    # (95.2% resolved with no overhead zone versus 84.6% with three), so it is
    # available as a filter but off by default.
    require_confluence: int = 0
    # From Codex's EntryGate: cap how far price has run from its mean before
    # entering, measured in ATR so it compares across coins. Validated on the
    # MEXC dataset - 1.35 lifts expectancy +0.0627 -> +0.0674 and 0.75 reaches
    # +0.0721, but each step roughly halves signal flow, so it is opt-in.
    max_chase_atr: float = 0.0
    # The strategy thesis is short-only (low-cap alts trend down); the long/panic
    # side is opt-in rather than on by default.
    enable_long_side: bool = False


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
    # Cross-sectional volatility cutoff for this scan. When supplied it replaces
    # SignalConfig.min_atr_pct, so the gate tracks the current market regime
    # instead of a number fitted to one period.
    atr_floor: float | None = None
    # Market benchmark (BTC) OHLCV, used to tell an engineered single-coin pump
    # apart from the whole board rallying.
    benchmark: pd.DataFrame | None = None
    # Higher-timeframe bars for this symbol, so indicators can be read on the
    # timeframe where they actually carry signal rather than all on one.
    htf_frame: pd.DataFrame | None = None


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
    # Price level that kills the setup while waiting. In window mode this is the
    # structural extreme (a new high means the fade thesis is wrong); a fixed small
    # percentage would just track noise on a low-cap alt.
    invalidate_level: float | None = None


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

    def _layer1_pump_window(self, df: pd.DataFrame) -> tuple[str | None, dict[str, float | str]]:
        """Layer 1 (window mode): a pump is an *event to fade*, not a same-bar state.

        The legacy detector required RSI extreme + volume spike + band breakout all on
        the current bar, which contradicts layer 2 (it demands the move already be
        weakening) - a bar cannot be at peak pump intensity and exhausted at once.
        Here layer 1 instead asks: did a pump fire somewhere in the recent window, was
        it large enough to be worth fading, has price turned off that peak, and are we
        still early enough in the fade to have room.
        """
        cfg = self.config
        window_bars = max(5, int(cfg.pump_window_bars))
        if len(df) < window_bars + 2:
            return None, {"insufficient_history": 1.0, "pump_window_bars": float(window_bars)}

        work = df.tail(window_bars)

        def _col(name: str, default: float) -> pd.Series:
            if name not in work.columns:
                return pd.Series(default, index=work.index, dtype=float)
            return pd.to_numeric(work[name], errors="coerce").fillna(default)

        close_s = _col("close", 0.0)
        high_s = _col("high", 0.0)
        low_s = _col("low", 0.0)
        rsi_s = _col("rsi", 50.0)
        vol_s = _col("volume_spike", 1.0)

        band_up = (close_s > _col("bb_upper", float("inf"))) | (close_s > _col("kc_upper", float("inf")))
        band_down = (close_s < _col("bb_lower", float("-inf"))) | (close_s < _col("kc_lower", float("-inf")))
        vol_hot = vol_s >= cfg.volume_spike_threshold

        pump_events = (band_up & ((rsi_s >= cfg.rsi_high) | vol_hot)).fillna(False)
        panic_events = (band_down & ((rsi_s <= cfg.rsi_low) | vol_hot)).fillna(False)

        close_now = float(close_s.iloc[-1])
        win_high = float(high_s.max())
        win_low = float(low_s.min())
        span = max(win_high - win_low, 1e-12)

        bars_since_peak = int(len(high_s) - 1 - int(high_s.to_numpy().argmax()))
        bars_since_trough = int(len(low_s) - 1 - int(low_s.to_numpy().argmin()))

        run_up_pct = (win_high - win_low) / max(win_low, 1e-12)
        drop_pct = (win_high - win_low) / max(win_high, 1e-12)
        retrace_from_high = (win_high - close_now) / span
        bounce_from_low = (close_now - win_low) / span

        metrics: dict[str, float | str] = {
            "mode": "window",
            "pump_window_bars": float(window_bars),
            "close": close_now,
            "window_high": win_high,
            "window_low": win_low,
            "run_up_pct": float(run_up_pct),
            "drop_pct": float(drop_pct),
            "bars_since_peak": float(bars_since_peak),
            "bars_since_trough": float(bars_since_trough),
            "retrace_from_high": float(retrace_from_high),
            "bounce_from_low": float(bounce_from_low),
            "pump_event_bars": float(pump_events.sum()),
            "panic_event_bars": float(panic_events.sum()),
            "rsi": float(rsi_s.iloc[-1]),
            "volume_spike": float(vol_s.iloc[-1]),
            "rsi_threshold": float(cfg.rsi_high),
            "volume_spike_threshold": float(cfg.volume_spike_threshold),
            "pump_min_move_pct": float(cfg.pump_min_move_pct),
        }

        pump = bool(
            pump_events.any()
            and run_up_pct >= cfg.pump_min_move_pct
            and bars_since_peak >= cfg.pump_min_bars_since_peak
            and close_now < win_high
            and retrace_from_high <= cfg.pump_max_retrace_pct
        )

        panic = bool(
            cfg.enable_long_side
            and panic_events.any()
            and drop_pct >= cfg.pump_min_move_pct
            and bars_since_trough >= cfg.pump_min_bars_since_peak
            and close_now > win_low
            and bounce_from_low <= cfg.pump_max_retrace_pct
        )

        metrics["pump_detected"] = 1.0 if pump else 0.0
        metrics["panic_detected"] = 1.0 if panic else 0.0
        metrics["long_side_enabled"] = 1.0 if cfg.enable_long_side else 0.0

        if pump:
            return "SHORT", metrics
        if panic:
            return "LONG", metrics
        return None, metrics

    def _layer1c_market_context(
        self,
        df: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        htf_frame: pd.DataFrame | None = None,
    ) -> tuple[bool, dict[str, float]]:
        """Reject moves the whole market is making, and optionally ones with no
        overhead level to stall into.

        Both were measured on the labelled dataset as tail-risk filters rather
        than profit boosters: they barely move expectancy but collapse the worst
        drawdown, which is what actually decides whether an account survives.
        """
        from core.levels import find_horizontal_levels, nearest_level_above
        from core.pump_features import relative_strength

        cfg = self.config
        details: dict[str, float] = {}

        rs = relative_strength(df, benchmark, lookback=cfg.relative_strength_lookback)
        details.update({k: float(v) for k, v in rs.items()})

        value = rs.get("relative_strength")
        if cfg.min_relative_strength > 0 and benchmark is not None and value is not None and value == value:
            rs_ok = value >= cfg.min_relative_strength
        else:
            # No benchmark available: do not silently block every signal.
            rs_ok = True
        details["relative_strength_ok"] = 1.0 if rs_ok else 0.0
        details["min_relative_strength"] = float(cfg.min_relative_strength)

        # Higher-timeframe momentum, read on its own timeframe rather than
        # resampled from the entry frame, which is too short to carry it.
        #
        # Fail closed. A misconfigured interval or a failed fetch returns an
        # empty frame, and treating that as a pass silently disables a gate that
        # was adopted precisely because it cut the tail - the bot would keep
        # trading while believing a filter was protecting it.
        htf_ok = True
        if cfg.min_rsi_4h > 0:
            usable = htf_frame is not None and not htf_frame.empty and len(htf_frame) >= 20
            details["min_rsi_htf"] = float(cfg.min_rsi_4h)
            details["htf_available"] = 1.0 if usable else 0.0
            if not usable:
                htf_ok = not cfg.require_htf
            else:
                from core.indicators import compute_indicators

                rsi_htf = float(compute_indicators(htf_frame.tail(200)).iloc[-1].get("rsi", float("nan")))
                details["rsi_htf"] = rsi_htf
                htf_ok = rsi_htf >= cfg.min_rsi_4h if rsi_htf == rsi_htf else not cfg.require_htf
            details["rsi_htf_ok"] = 1.0 if htf_ok else 0.0

        chase_ok = True
        if cfg.max_chase_atr > 0:
            from core.pump_features import extension

            ext = extension(df).get("ext_ema20_atr", float("nan"))
            details["chase_atr"] = float(ext) if ext == ext else 0.0
            details["max_chase_atr"] = float(cfg.max_chase_atr)
            if ext == ext:
                chase_ok = abs(ext) <= cfg.max_chase_atr
            details["chase_ok"] = 1.0 if chase_ok else 0.0

        level_ok = True
        if cfg.require_level_overhead:
            close = self._safe(df.iloc[-1].get("close"))
            overhead = nearest_level_above(find_horizontal_levels(df), close)
            dist = (overhead.price - close) / close if (overhead and close > 0) else float("nan")
            level_ok = bool(overhead is not None and dist >= cfg.min_level_dist_pct)
            details["level_dist"] = float(dist) if dist == dist else 0.0
            details["level_ok"] = 1.0 if level_ok else 0.0

        return bool(rs_ok and htf_ok and chase_ok and level_ok), details

    def _layer1b_quality_gate(self, df: pd.DataFrame, atr_floor: float | None = None) -> tuple[bool, dict[str, float]]:
        """Reject pumps that historically fail to resolve: calm or illiquid ones.

        Low-volatility pumps drift instead of dumping (86% resolve vs 99% for the
        violent ones), and thin books carry a 4x higher share of runaway losers -
        illiquid events were the single best advance predictor of a deep drawdown.
        """
        cfg = self.config
        last = df.iloc[-1]
        close = self._safe(last.get("close"))
        atr = self._safe(last.get("atr"), 0.0)
        atr_pct = atr / close if close else 0.0

        lookback = max(1, int(cfg.liquidity_lookback_bars))
        tail = df.tail(lookback)
        # `volume` on a MEXC kline is a contract count, so close*volume is only
        # USD by accident - it is wrong by the contract size, which differs per
        # symbol (BTC ~10,000x too high, CHILLGUY 10x too low). The exchange
        # ships exact quote turnover alongside it; prefer that whenever present.
        if "turnover" in tail.columns:
            usd_volume = float(pd.to_numeric(tail["turnover"], errors="coerce").fillna(0.0).sum())
        else:
            usd_volume = float(
                (pd.to_numeric(tail["close"], errors="coerce") * pd.to_numeric(tail["volume"], errors="coerce"))
                .fillna(0.0)
                .sum()
            )

        effective_floor = cfg.min_atr_pct if atr_floor is None else atr_floor
        atr_ok = atr_pct >= effective_floor if effective_floor > 0 else True
        liq_ok = usd_volume >= cfg.min_hourly_usd_volume if cfg.min_hourly_usd_volume > 0 else True

        return bool(atr_ok and liq_ok), {
            "atr_pct": float(atr_pct),
            "min_atr_pct": float(effective_floor),
            "atr_floor_adaptive": 1.0 if atr_floor is not None else 0.0,
            "atr_ok": 1.0 if atr_ok else 0.0,
            "usd_volume_recent": usd_volume,
            "min_hourly_usd_volume": float(cfg.min_hourly_usd_volume),
            "liquidity_ok": 1.0 if liq_ok else 0.0,
        }

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

    def _window_extremes(
        self,
        df: pd.DataFrame,
        htf_frame: pd.DataFrame | None = None,
    ) -> tuple[float, float]:
        """The swing the setup is measured against.

        Taken from the higher timeframe when one is available. An hourly high is
        set by whichever wick happened to print, so a stop referenced to it sits
        inside ordinary noise and invalidation fires on almost every candidate;
        a 4h swing high is a level the market actually respected.
        """
        if self.config.structural_anchor_htf and htf_frame is not None and len(htf_frame) >= 3:
            bars = max(3, int(self.config.structural_anchor_htf_bars))
            work = htf_frame.tail(bars)
        else:
            work = df.tail(max(5, int(self.config.pump_window_bars)))
        high = pd.to_numeric(work["high"], errors="coerce").max()
        low = pd.to_numeric(work["low"], errors="coerce").min()
        return float(high), float(low)

    def _stop_buffer(self, df: pd.DataFrame, reference: float) -> float:
        """Distance to place a stop beyond the structural level, in price.

        A fixed percentage cannot serve coins whose hourly ranges differ tenfold:
        0.3% is a wide berth on a quiet pair and pure noise on a volatile one,
        which is exactly where this strategy operates. The buffer therefore
        scales with ATR and keeps the percentage only as a floor.
        """
        cfg = self.config
        pct_buffer = reference * max(0.0, cfg.pump_stop_buffer_pct)
        atr = self._safe(df.iloc[-1].get("atr"), 0.0)
        atr_buffer = atr * max(0.0, cfg.stop_buffer_atr_mult)
        return max(pct_buffer, atr_buffer)

    def _layer3_entry_near_peak(self, df: pd.DataFrame, side: str, vp: VolumeProfileLevels | None,
                                htf_frame: pd.DataFrame | None = None) -> tuple[bool, dict[str, float]]:
        """Layer 3 (window mode): entry must sit close to the pump's own extreme.

        The legacy check waits for price to travel all the way back to VAH, but a pump
        overshoots VAH by far - by the time price returns there the dump is mostly done
        and no tight stop exists. Anchoring on the window extreme keeps the entry at the
        turn, where the stop is small enough to survive high leverage.
        """
        if len(df) < 2:
            return False, {"insufficient_history": 1.0}

        close = self._safe(df.iloc[-1].get("close"))
        win_high, win_low = self._window_extremes(df, htf_frame)
        max_dist = max(0.0, self.config.pump_entry_max_dist_from_peak_pct)

        if side == "SHORT":
            reference = win_high
            dist = (win_high - close) / max(win_high, 1e-12)
        else:
            reference = win_low
            dist = (close - win_low) / max(win_low, 1e-12)

        entry_ok = 0.0 <= dist <= max_dist
        msb_ok, msb = self._layer3_msb_confirmation(df=df, side=side)

        details = {
            "mode_window": 1.0,
            "close": close,
            "reference_extreme": float(reference),
            "dist_from_extreme_pct": float(dist),
            "max_dist_from_extreme_pct": float(max_dist),
            "entry_ok": 1.0 if entry_ok else 0.0,
            "poc": float(vp.poc) if vp else 0.0,
            "vah": float(vp.vah) if vp else 0.0,
            "val": float(vp.val) if vp else 0.0,
            "vp_levels_available": 1.0 if vp else 0.0,
        }
        details.update(msb)
        return bool(entry_ok and msb_ok), details

    def _layer3_entry_location(self, df: pd.DataFrame, side: str, vp: VolumeProfileLevels | None,
                               htf_frame: pd.DataFrame | None = None) -> tuple[bool, dict[str, float]]:
        """Layer 3: Entry location via Volume Profile levels + MSB."""
        if self.config.pump_window_enabled:
            return self._layer3_entry_near_peak(df, side, vp, htf_frame)

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

    def _layer5_structural_levels(self, df: pd.DataFrame, side: str, vp: VolumeProfileLevels | None,
                                  htf_frame: pd.DataFrame | None = None) -> tuple[float, float, list[float]]:
        """Layer 5 (window mode): stop just beyond the pump extreme, not an ATR multiple.

        The thesis is invalidated by a new extreme, so that is where the stop belongs.
        It is also usually far tighter than an ATR stop, which is what makes the setup
        survivable at high leverage; setups whose structural stop is too wide are
        rejected upstream rather than traded with an arbitrarily tightened stop.
        """
        close = self._safe(df.iloc[-1].get("close"))
        atr = self._safe(df.iloc[-1].get("atr"), close * 0.01) or close * 0.01
        win_high, win_low = self._window_extremes(df, htf_frame)

        if side == "SHORT":
            sl = win_high + self._stop_buffer(df, win_high)
            risk = max(sl - close, 1e-12)
            tp = vp.poc if (vp and vp.poc < close) else close - risk * self.config.risk_reward
            partial = [close - risk, (close + float(tp)) / 2.0]
        else:
            sl = win_low - self._stop_buffer(df, win_low)
            risk = max(close - sl, 1e-12)
            tp = vp.poc if (vp and vp.poc > close) else close + risk * self.config.risk_reward
            partial = [close + risk, (close + float(tp)) / 2.0]

        return float(tp), float(sl), [float(x) for x in partial]

    def _layer5_tp_sl_levels(self, df: pd.DataFrame, side: str, vp: VolumeProfileLevels | None,
                             htf_frame: pd.DataFrame | None = None) -> tuple[float, float, list[float]]:
        """Layer 5: TP/SL levels via ATR + Volume Profile (with RR fallback)."""
        if self.config.pump_window_enabled:
            return self._layer5_structural_levels(df, side, vp, htf_frame)

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
        if self.config.pump_window_enabled:
            side, layer1 = self._layer1_pump_window(df)
        else:
            side, layer1 = self._layer1_pump_detection(df)
            if side == "LONG" and not self.config.enable_long_side:
                side = None
                layer1["long_side_disabled"] = 1.0
        trace["layers"]["layer1_pump_detection"] = {"passed": side is not None, "side": side or "", "details": layer1}
        if side is None:
            trace["failed_layer"] = "layer1_pump_detection"
            return None

        if self.config.pump_window_enabled:
            quality_ok, quality = self._layer1b_quality_gate(df, atr_floor=context.atr_floor)
            trace["layers"]["layer1b_quality_gate"] = {"passed": quality_ok, "details": quality}
            if not quality_ok:
                trace["failed_layer"] = "layer1b_quality_gate"
                return None

            market_ok, market = self._layer1c_market_context(df, context.benchmark, context.htf_frame)
            trace["layers"]["layer1c_market_context"] = {"passed": market_ok, "details": market}
            if not market_ok:
                trace["failed_layer"] = "layer1c_market_context"
                return None

        if self.config.weakness_layer_enabled:
            layer2_ok, layer2 = self._layer2_weakness_confirmation(df, side)
            trace["layers"]["layer2_weakness_confirmation"] = {"passed": layer2_ok, "details": layer2}
            if not layer2_ok:
                trace["failed_layer"] = "layer2_weakness_confirmation"
                return None
        else:
            layer2 = {"skipped": 1.0}
            trace["layers"]["layer2_weakness_confirmation"] = {"passed": True, "details": layer2}

        layer3_ok, layer3 = self._layer3_entry_location(df, side, context.volume_profile, context.htf_frame)
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
    ) -> SignalResult | None:
        entry = float(entry if entry is not None else df.iloc[-1]["close"])
        tp, sl, partial_tps = self._layer5_tp_sl_levels(df, side, context.volume_profile, context.htf_frame)
        tp, sl = self._normalize_levels(entry=entry, tp=tp, sl=sl, side=side)

        stop_distance_pct = abs(sl - entry) / max(entry, 1e-12)
        max_stop = max(0.0, self.config.max_stop_distance_pct)
        if max_stop > 0 and stop_distance_pct > max_stop:
            trace["failed_layer"] = "layer5_stop_too_wide"
            trace["layers"]["layer5_tp_sl"] = {
                "passed": False,
                "details": {
                    "entry": float(entry),
                    "sl": float(sl),
                    "stop_distance_pct": float(stop_distance_pct),
                    "max_stop_distance_pct": float(max_stop),
                },
            }
            self.last_diagnostics = trace
            return None

        risk = abs(sl - entry)
        reward = abs(tp - entry)
        realized_rr = reward / max(risk, 1e-12)
        if self.config.min_risk_reward > 0 and realized_rr < self.config.min_risk_reward:
            trace["failed_layer"] = "layer5_risk_reward_too_low"
            trace["layers"]["layer5_tp_sl"] = {
                "passed": False,
                "details": {
                    "entry": float(entry),
                    "tp": float(tp),
                    "sl": float(sl),
                    "realized_risk_reward": float(realized_rr),
                    "min_risk_reward": float(self.config.min_risk_reward),
                },
            }
            self.last_diagnostics = trace
            return None

        layer5 = {
            "entry": float(entry),
            "tp": float(tp),
            "sl": float(sl),
            "partial_tps": [float(x) for x in partial_tps],
            "atr_sl_mult": float(self.config.atr_sl_mult),
            "risk_reward": float(self.config.risk_reward),
            "vp_available": 1.0 if context.volume_profile is not None else 0.0,
            "stop_distance_pct": float(stop_distance_pct),
            "realized_risk_reward": float(realized_rr),
            # Same levels expressed against committed margin, which is how the trade
            # is actually sized: a 2% move at 100x is 200% of the margin put up.
            "report_leverage": float(self.config.report_leverage),
            "stop_pct_of_margin": float(stop_distance_pct * self.config.report_leverage * 100.0),
            "target_pct_of_margin": float(reward / max(entry, 1e-12) * self.config.report_leverage * 100.0),
            # Largest leverage at which the stop still costs less than the whole
            # margin, so a stop-out is not automatically a liquidation.
            "max_safe_leverage": float(1.0 / max(stop_distance_pct, 1e-6)),
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

            last = df.iloc[-1]
            close_now = float(last["close"])
            # Invalidation must read the extreme, not the close. A bar that takes
            # out the structural level and then closes back inside it has already
            # broken the setup - the stop would have been hit in real time - yet
            # judging on the close alone would confirm the entry instead.
            high_now = float(last.get("high", close_now))
            low_now = float(last.get("low", close_now))
            pending.last_seen_bar_ts = bar_ts

            if pending.side == "LONG":
                confirmed = close_now > pending.armed_close
                floor = (
                    pending.invalidate_level
                    if pending.invalidate_level is not None
                    else pending.armed_close * (1.0 - self.config.confirmation_invalidate_pct)
                )
                invalidated = low_now <= floor
            else:
                confirmed = close_now < pending.armed_close
                ceiling = (
                    pending.invalidate_level
                    if pending.invalidate_level is not None
                    else pending.armed_close * (1.0 + self.config.confirmation_invalidate_pct)
                )
                invalidated = high_now >= ceiling

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

        invalidate_level: float | None = None
        if self.config.pump_window_enabled:
            # Same level the stop will use, so a candidate is only abandoned where
            # the trade itself would have been.
            win_high, win_low = self._window_extremes(df, context.htf_frame)
            invalidate_level = (win_high + self._stop_buffer(df, win_high) if side == "SHORT"
                                else win_low - self._stop_buffer(df, win_low))

        self._pending[context.symbol] = PendingCandidate(
            symbol=context.symbol,
            side=side,
            armed_bar_ts=bar_ts,
            armed_close=armed_close,
            last_seen_bar_ts=bar_ts,
            bars_waited=0,
            layer_snapshot={"layer1": layer1, "layer2": layer2, "layer3": layer3, "layer4": layer4},
            invalidate_level=invalidate_level,
        )
        trace["failed_layer"] = "layer_confirmation_pending"
        trace["layers"]["layer_confirmation"] = {
            "passed": False,
            "details": {
                "status": "armed",
                "armed_close": armed_close,
                "invalidate_level": float(invalidate_level) if invalidate_level is not None else 0.0,
                "bars_waited": 0.0,
            },
        }
        self.last_diagnostics = trace
        return None
