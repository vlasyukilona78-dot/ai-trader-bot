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
    rsi_high: float = 68.0
    rsi_low: float = 32.0
    volume_spike_threshold: float = 1.6
    weakness_lookback: int = 4
    sentiment_bullish_threshold: float = 68.0
    fake_filter_sentiment_euphoric_soft: float = 66.0
    sentiment_bearish_threshold: float = 32.0
    risk_reward: float = 1.6
    atr_sl_mult: float = 1.0
    entry_tolerance_pct: float = 0.004
    vwap_tolerance_pct: float = 0.0025
    funding_tolerance: float = 0.0003
    fake_filter_funding_supports_short_soft: float = 0.0005
    long_short_ratio_tolerance: float = 0.10
    msb_lookback: int = 20
    msb_recent_bars: int = 6
    msb_break_buffer_pct: float = 0.0005
    regime_vwap_stretch_soft: float = 0.0015
    regime_min_atr_norm: float = 0.0010
    regime_strong_trend_adx: float = 28.0
    fake_filter_lsr_extreme_soft: float = 1.01
    fake_filter_oi_volume_spike_soft: float = 1.2


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
    sentiment_value: float | None = None
    sentiment_degraded: bool | None = None
    funding_source: str | None = None
    funding_degraded: bool | None = None
    long_short_ratio_source: str | None = None
    long_short_ratio_degraded: bool | None = None
    news_veto: bool | None = None
    news_source: str | None = None
    news_degraded: bool | None = None
    open_interest: float | None = None
    open_interest_ratio: float | None = None
    oi_signal: float | None = None
    oi_source: str | None = None
    oi_degraded: bool | None = None
    open_interest_source: str | None = None


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


class SignalGenerator:
    def __init__(self, config: SignalConfig | None = None):
        self.config = config or SignalConfig()
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _safe(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _source_tag(source: str | None, fallback: str = "unavailable") -> str:
        src = (source or "").strip().lower()
        return src or fallback

    @staticmethod
    def _source_quality(source: str, *, value_present: bool, degraded: bool | None) -> str:
        src = (source or "").strip().lower()
        if src in ("", "unavailable", "missing", "none", "unknown"):
            return "unavailable"
        if degraded is True:
            return "fallback"
        if src.startswith("fallback") or src.startswith("synthetic") or src.startswith("derived"):
            return "fallback"
        if not value_present:
            return "unavailable"
        return "live"

    @staticmethod
    def _quality_flags(prefix: str, quality: str) -> dict[str, Any]:
        return {
            f"{prefix}_quality": quality,
            f"{prefix}_live_used": 1.0 if quality == "live" else 0.0,
            f"{prefix}_fallback_used": 1.0 if quality == "fallback" else 0.0,
            f"{prefix}_unavailable": 1.0 if quality == "unavailable" else 0.0,
        }

    def _regime_filter(
        self,
        df: pd.DataFrame,
        regime: MarketRegime,
        *,
        news_veto: bool | None,
        news_source: str | None,
        news_degraded: bool | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        if df.empty:
            src = {"news_source": "unavailable", "news_available": 0.0, "vwap_available": 0.0, **self._quality_flags("news", "unavailable"), **self._quality_flags("vwap", "unavailable")}
            return False, {"passed": 0.0, "htf_trend_ok": 0.0, "stretched_from_vwap": 0.0, "volatility_regime_ok": 0.0, "news_veto": 1.0, "missing_conditions": "history", "failed_reason": "insufficient_history", "degraded_mode": 1.0, "source_flags": src}

        last = df.iloc[-1]
        close = self._safe(last.get("close"), 0.0)
        vwap_raw = self._safe(last.get("vwap"), 0.0)
        vwap_available = bool(np.isfinite(vwap_raw) and vwap_raw > 0)
        vwap = vwap_raw if vwap_available else max(close, 1e-9)
        atr_norm = self._safe(last.get("atr"), close * 0.01) / max(close, 1e-9)
        adx = self._safe(last.get("adx"), 0.0)
        ema20 = self._safe(last.get("ema20"), close)
        ema50 = self._safe(last.get("ema50"), close)
        vwap_dist = self._safe(last.get("vwap_dist"), (close - vwap) / max(vwap, 1e-9))

        htf_trend_ok = not (ema20 >= ema50 and adx >= float(self.config.regime_strong_trend_adx) and close >= ema20) and regime != MarketRegime.PANIC
        stretched = vwap_dist >= max(float(self.config.regime_vwap_stretch_soft), float(self.config.vwap_tolerance_pct) * 0.5)
        vol_ok = atr_norm >= max(float(self.config.regime_min_atr_norm), float(self.config.entry_tolerance_pct) * 0.5)

        news_source_tag = self._source_tag(news_source, fallback="provided" if news_veto is not None else "unavailable")
        news_quality = self._source_quality(news_source_tag, value_present=news_veto is not None, degraded=news_degraded)
        news_ok = True if news_veto is None else (not bool(news_veto))

        vwap_quality = "live" if vwap_available else "fallback"
        degraded = news_quality != "live" or vwap_quality != "live"

        missing = []
        if not htf_trend_ok: missing.append("htf_trend_ok")
        if not stretched: missing.append("stretched_from_vwap")
        if not vol_ok: missing.append("volatility_regime_ok")
        if not news_ok: missing.append("news_veto")
        passed = htf_trend_ok and stretched and vol_ok and news_ok

        src = {"news_source": news_source_tag, "news_available": 1.0 if news_quality == "live" else 0.0, "vwap_available": 1.0 if vwap_available else 0.0, **self._quality_flags("news", news_quality), **self._quality_flags("vwap", vwap_quality)}
        return passed, {"passed": 1.0 if passed else 0.0, "htf_trend_ok": 1.0 if htf_trend_ok else 0.0, "stretched_from_vwap": 1.0 if stretched else 0.0, "volatility_regime_ok": 1.0 if vol_ok else 0.0, "news_veto": 1.0 if news_ok else 0.0, "degraded_mode": 1.0 if degraded else 0.0, "missing_conditions": ",".join(missing), "failed_reason": "none" if passed else f"missing:{','.join(missing)}", "source_flags": src}

    def _layer1_pump_detection(self, df: pd.DataFrame) -> tuple[str | None, dict[str, Any]]:
        if df.empty:
            return None, {"passed": 0.0, "failed_reason": "insufficient_history", "missing_conditions": "history", "pump_context_strength": 0.0}
        last = df.iloc[-1]
        rsi = self._safe(last.get("rsi"), 50.0)
        vol = self._safe(last.get("volume_spike"), 1.0)
        close = self._safe(last.get("close"), 0.0)
        bb_u = self._safe(last.get("bb_upper"), np.inf)
        kc_u = self._safe(last.get("kc_upper"), np.inf)

        rsi_high = rsi >= float(self.config.rsi_high)
        vol_high = vol >= float(self.config.volume_spike_threshold)
        above_bb = bool(np.isfinite(bb_u) and close > bb_u)
        above_kc = bool(np.isfinite(kc_u) and close > kc_u)
        band_break = above_bb or above_kc

        pts = int(rsi_high) + int(vol_high) + int(band_break)
        passed = band_break and pts >= 2
        missing = []
        if not rsi_high: missing.append("rsi_high")
        if not vol_high: missing.append("volume_spike")
        if not band_break: missing.append("upper_band_breakout")
        details = {
            "passed": 1.0 if passed else 0.0,
            "failed_reason": "none" if passed else f"missing:{','.join(missing)}",
            "missing_conditions": ",".join(missing),
            "rsi": float(rsi),
            "rsi_high": 1.0 if rsi_high else 0.0,
            "volume_spike": float(vol),
            "above_bollinger_upper": 1.0 if above_bb else 0.0,
            "above_keltner_upper": 1.0 if above_kc else 0.0,
            "upper_band_breakout": 1.0 if band_break else 0.0,
            "pump_context_strength": float(pts / 3.0),
        }
        return ("SHORT", details) if passed else (None, details)

    def _layer2_weakness_confirmation(self, df: pd.DataFrame, side: str) -> tuple[bool, dict[str, Any]]:
        if side != "SHORT":
            return False, {"passed": 0.0, "price_up_or_near_high": 0.0, "obv_bearish_divergence": 0.0, "cvd_bearish_divergence": 0.0, "weakness_strength": 0.0, "missing_conditions": "short_context_required", "failed_reason": "unsupported_side_not_short"}
        lb = int(self.config.weakness_lookback)
        if len(df) < lb + 2:
            return False, {"passed": 0.0, "price_up_or_near_high": 0.0, "obv_bearish_divergence": 0.0, "cvd_bearish_divergence": 0.0, "weakness_strength": 0.0, "missing_conditions": "history", "failed_reason": "insufficient_history"}

        last = df.iloc[-1]
        ref = df.iloc[-1 - lb]
        close_last = self._safe(last.get("close"), 0.0)
        close_ref = self._safe(ref.get("close"), close_last)
        price_up = close_last > close_ref
        near = close_last >= close_ref * 0.95
        price_ctx = price_up or near

        obv_div = self._safe(last.get("obv"), 0.0) < self._safe(ref.get("obv"), 0.0)
        cvd_div = self._safe(last.get("cvd"), 0.0) < self._safe(ref.get("cvd"), 0.0)
        passed = price_ctx and (obv_div or cvd_div)
        strength = (int(price_ctx) + int(obv_div) + int(cvd_div)) / 3.0
        missing = []
        if not price_ctx: missing.append("price_up_or_near_high")
        if not obv_div: missing.append("obv_bearish_divergence")
        if not cvd_div: missing.append("cvd_bearish_divergence")

        return passed, {"passed": 1.0 if passed else 0.0, "price_up_or_near_high": 1.0 if price_ctx else 0.0, "obv_bearish_divergence": 1.0 if obv_div else 0.0, "cvd_bearish_divergence": 1.0 if cvd_div else 0.0, "weakness_strength": float(strength), "missing_conditions": ",".join(missing), "failed_reason": "none" if passed else f"missing:{','.join(missing)}"}

    def _layer3_msb_confirmation(self, df: pd.DataFrame, side: str) -> tuple[bool, dict[str, float]]:
        lb = max(5, int(self.config.msb_lookback))
        rb = max(1, int(self.config.msb_recent_bars))
        bb = max(0.0, float(self.config.msb_break_buffer_pct))
        need = lb + rb + 2
        if len(df) < need:
            return False, {"msb_missing": 1.0, "msb_lookback": float(lb), "msb_recent_bars": float(rb), "msb_break_buffer_pct": bb}

        w = df.tail(need)
        close = pd.to_numeric(w["close"], errors="coerce")
        high = pd.to_numeric(w["high"], errors="coerce")
        low = pd.to_numeric(w["low"], errors="coerce")
        prior_low = low.rolling(lb, min_periods=lb).min().shift(1)
        prior_high = high.rolling(lb, min_periods=lb).max().shift(1)

        down_struct = (close < prior_low * (1.0 - bb)).fillna(False)
        up_struct = (close > prior_high * (1.0 + bb)).fillna(False)
        if "ema20" in w.columns:
            ema20 = pd.to_numeric(w["ema20"], errors="coerce")
            down_cross = ((close < ema20) & (close.shift(1) >= ema20.shift(1))).fillna(False)
            up_cross = ((close > ema20) & (close.shift(1) <= ema20.shift(1))).fillna(False)
        else:
            down_cross = pd.Series(False, index=w.index)
            up_cross = pd.Series(False, index=w.index)

        down_recent = bool(down_struct.tail(rb).any() or down_cross.tail(rb).any())
        up_recent = bool(up_struct.tail(rb).any() or up_cross.tail(rb).any())
        ok = down_recent if side == "SHORT" else up_recent
        return ok, {"msb_ok": 1.0 if ok else 0.0, "msb_down_recent": 1.0 if down_recent else 0.0, "msb_up_recent": 1.0 if up_recent else 0.0, "msb_lookback": float(lb), "msb_recent_bars": float(rb), "msb_break_buffer_pct": bb}

    def _layer3_entry_location(self, df: pd.DataFrame, side: str, vp: VolumeProfileLevels | None) -> tuple[bool, dict[str, Any]]:
        if side != "SHORT":
            return False, {"entry_location_passed": 0.0, "failed_reason": "unsupported_side_not_short", "missing_conditions": "short_context_required", "entry_location_strength": 0.0}
        if vp is None or len(df) < 2:
            return False, {"entry_location_passed": 0.0, "failed_reason": "vp_missing", "missing_conditions": "volume_profile", "entry_location_strength": 0.0, "vp_levels_available": 0.0}

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = self._safe(last.get("close"))
        prev_close = self._safe(prev.get("close"))
        tol = max(0.0, float(self.config.entry_tolerance_pct))

        below_vah = close <= vp.vah * (1.0 + tol)
        rejected = prev_close >= vp.vah * (1.0 - tol) and close <= vp.vah * (1.0 + tol)
        below_or_rej = rejected
        near_poc = abs(close - vp.poc) <= max(vp.poc * max(0.006, tol * 1.5), 1e-8)
        inside_va = close >= vp.val * (1.0 - tol) and close <= vp.vah * (1.0 + tol)
        poc_ctx = near_poc or inside_va
        msb_ok, msb = self._layer3_msb_confirmation(df=df, side="SHORT")

        passed = below_or_rej and poc_ctx and bool(msb_ok)
        strength = (int(below_or_rej) + int(poc_ctx) + int(bool(msb_ok))) / 3.0
        missing = []
        if not below_or_rej: missing.append("below_vah_or_rejected_from_vah")
        if not poc_ctx: missing.append("near_poc_or_value_area_context")
        if not bool(msb_ok): missing.append("msb_bearish_confirmed")

        d = {"entry_location_passed": 1.0 if passed else 0.0, "failed_reason": "none" if passed else f"missing:{','.join(missing)}", "missing_conditions": ",".join(missing), "entry_location_strength": float(strength), "below_vah_or_rejected_from_vah": 1.0 if below_or_rej else 0.0, "near_poc_or_value_area_context": 1.0 if poc_ctx else 0.0, "msb_bearish_confirmed": 1.0 if bool(msb_ok) else 0.0, "vp_levels_available": 1.0, "below_vah": 1.0 if below_vah else 0.0}
        d.update(msb)
        return passed, d

    def _layer4_fake_filter(
        self,
        df: pd.DataFrame,
        side: str,
        sentiment_index: float | None,
        sentiment_source: str | None,
        funding_rate: float | None,
        long_short_ratio: float | None,
        open_interest: float | None = None,
        open_interest_source: str | None = None,
        sentiment_value: float | None = None,
        sentiment_degraded: bool | None = None,
        funding_source: str | None = None,
        funding_degraded: bool | None = None,
        long_short_ratio_source: str | None = None,
        long_short_ratio_degraded: bool | None = None,
        open_interest_ratio: float | None = None,
        oi_signal: float | None = None,
        oi_source: str | None = None,
        oi_degraded: bool | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        if df.empty or side != "SHORT":
            return False, {"passed": 0.0, "degraded_mode": 1.0, "failed_reason": "insufficient_history" if df.empty else "unsupported_side_not_short", "missing_conditions": "history" if df.empty else "short_context_required", "source_flags": {}}

        last = df.iloc[-1]
        close = self._safe(last.get("close"), 0.0)
        vwap_raw = self._safe(last.get("vwap"), 0.0)
        vwap_available = bool(np.isfinite(vwap_raw) and vwap_raw > 0)
        vwap = vwap_raw if vwap_available else max(close, 1e-9)

        sentiment_eff = sentiment_value if sentiment_value is not None else sentiment_index
        sentiment_source_tag = self._source_tag(sentiment_source, fallback="provided" if sentiment_eff is not None else "unavailable")
        sentiment_quality = self._source_quality(sentiment_source_tag, value_present=sentiment_eff is not None, degraded=sentiment_degraded)
        sentiment_num = 50.0 if sentiment_eff is None else float(sentiment_eff)

        funding_source_tag = self._source_tag(funding_source, fallback="provided" if funding_rate is not None else "unavailable")
        funding_quality = self._source_quality(funding_source_tag, value_present=funding_rate is not None, degraded=funding_degraded)

        lsr_source_tag = self._source_tag(long_short_ratio_source, fallback="provided" if long_short_ratio is not None else "unavailable")
        lsr_quality = self._source_quality(lsr_source_tag, value_present=long_short_ratio is not None, degraded=long_short_ratio_degraded)

        oi_metric = open_interest_ratio if open_interest_ratio is not None else oi_signal
        oi_from_abs = oi_metric is None and open_interest is not None
        if oi_metric is None and open_interest is not None:
            oi_metric = float(open_interest)
        oi_source_tag = self._source_tag(oi_source if oi_source is not None else open_interest_source, fallback="provided" if oi_metric is not None else "unavailable")
        oi_quality = self._source_quality(oi_source_tag, value_present=oi_metric is not None, degraded=oi_degraded)

        vwap_quality = "live" if vwap_available else "fallback"
        degraded = any(q != "live" for q in (sentiment_quality, funding_quality, lsr_quality, oi_quality, vwap_quality))

        vwap_tol = max(0.0, float(self.config.vwap_tolerance_pct))
        funding_tol = max(0.0, float(self.config.funding_tolerance))
        funding_supports_short_threshold = max(0.0, float(self.config.fake_filter_funding_supports_short_soft))
        lsr_extreme_threshold = max(1.0, float(self.config.fake_filter_lsr_extreme_soft))
        oi_soft = max(float(self.config.fake_filter_oi_volume_spike_soft), 1.0)

        price_ok = close >= vwap * (1.0 - vwap_tol)
        sentiment_euphoric_threshold = float(self.config.fake_filter_sentiment_euphoric_soft)
        sent_ok = sentiment_num >= sentiment_euphoric_threshold
        sent_or_unavail = sent_ok or sentiment_quality == "unavailable"
        funding_ok = funding_rate is not None and float(funding_rate) >= -funding_supports_short_threshold
        funding_or_unavail = funding_ok or funding_quality == "unavailable"
        lsr_ok = long_short_ratio is not None and float(long_short_ratio) >= lsr_extreme_threshold
        lsr_or_unavail = lsr_ok or lsr_quality == "unavailable"
        if oi_metric is None:
            oi_ok = False
        elif oi_from_abs:
            oi_ok = True
        else:
            oi_ok = float(oi_metric) >= oi_soft
        oi_or_unavail = oi_ok or oi_quality == "unavailable"
        passed = price_ok and sent_or_unavail and funding_or_unavail and lsr_or_unavail and oi_or_unavail
        missing = []
        if not price_ok: missing.append("price_above_vwap")
        if not sent_or_unavail: missing.append("sentiment_euphoric")
        if not funding_or_unavail: missing.append("funding_supports_short")
        if not lsr_or_unavail: missing.append("long_short_ratio_extreme")
        if not oi_or_unavail: missing.append("oi_overheated")

        blocker_price = not price_ok
        blocker_sent = not sent_or_unavail
        blocker_funding = not funding_or_unavail
        blocker_lsr = not lsr_or_unavail
        blocker_oi = not oi_or_unavail

        fail_price = (not passed) and blocker_price
        fail_sent = (not passed) and blocker_sent
        fail_deriv = (not passed) and (blocker_funding or blocker_lsr or blocker_oi)
        fail_degraded_only = (not passed) and degraded and not (fail_price or fail_sent or fail_deriv)
        hard_fail = (not passed) and not degraded
        degraded_data_fail = (not passed) and fail_degraded_only
        soft_fail = (not passed) and not hard_fail and not degraded_data_fail

        soft_pass_candidate = (not passed) and price_ok and (sent_ok or funding_ok or lsr_ok or oi_ok)
        strength = (int(price_ok) + int(sent_or_unavail) + int(funding_or_unavail) + int(lsr_or_unavail) + int(oi_or_unavail)) / 5.0

        src = {
            "sentiment_source": sentiment_source_tag,
            "funding_source": funding_source_tag,
            "long_short_ratio_source": lsr_source_tag,
            "open_interest_source": oi_source_tag,
            "vwap_available": 1.0 if vwap_available else 0.0,
            **self._quality_flags("sentiment", sentiment_quality),
            **self._quality_flags("funding", funding_quality),
            **self._quality_flags("long_short_ratio", lsr_quality),
            **self._quality_flags("open_interest", oi_quality),
            **self._quality_flags("vwap", vwap_quality),
        }
        return passed, {
            "passed": 1.0 if passed else 0.0,
            "price_above_vwap": 1.0 if price_ok else 0.0,
            "sentiment_euphoric": 1.0 if sent_ok else 0.0,
            "sentiment_euphoric_threshold": float(sentiment_euphoric_threshold),
            "funding_supports_short": 1.0 if funding_ok else 0.0,
            "funding_supports_short_threshold": float(funding_supports_short_threshold),
            "long_short_ratio_extreme": 1.0 if lsr_ok else 0.0,
            "long_short_ratio_extreme_threshold": float(lsr_extreme_threshold),
            "oi_overheated": 1.0 if oi_ok else 0.0,
            "sentiment_euphoric_or_unavailable": 1.0 if sent_or_unavail else 0.0,
            "funding_supports_short_or_unavailable": 1.0 if funding_or_unavail else 0.0,
            "long_short_ratio_extreme_or_unavailable": 1.0 if lsr_or_unavail else 0.0,
            "oi_overheated_or_unavailable": 1.0 if oi_or_unavail else 0.0,
            "blocker_price_above_vwap": 1.0 if blocker_price else 0.0,
            "blocker_sentiment_euphoric": 1.0 if blocker_sent else 0.0,
            "blocker_funding_supports_short": 1.0 if blocker_funding else 0.0,
            "blocker_long_short_ratio_extreme": 1.0 if blocker_lsr else 0.0,
            "blocker_oi_overheated": 1.0 if blocker_oi else 0.0,
            "fail_due_to_price_structure": 1.0 if fail_price else 0.0,
            "fail_due_to_sentiment": 1.0 if fail_sent else 0.0,
            "fail_due_to_derivatives_context": 1.0 if fail_deriv else 0.0,
            "fail_due_to_degraded_mode_only": 1.0 if fail_degraded_only else 0.0,
            "hard_fail": 1.0 if hard_fail else 0.0,
            "soft_fail": 1.0 if soft_fail else 0.0,
            "degraded_data_fail": 1.0 if degraded_data_fail else 0.0,
            "soft_pass_candidate": 1.0 if soft_pass_candidate else 0.0,
            "degraded_mode": 1.0 if degraded else 0.0,
            "source_flags": src,
            "missing_conditions": ",".join(missing),
            "failed_reason": "none" if passed else f"missing:{','.join(missing)}",
            "fake_filter_strength": float(strength),
            "sentiment": float(sentiment_num),
            "funding_rate": float(funding_rate) if funding_rate is not None else 0.0,
            "long_short_ratio": float(long_short_ratio) if long_short_ratio is not None else 0.0,
            "crowd_not_against": 1.0 if sent_or_unavail else 0.0,
            "crowd_extreme": 1.0 if sent_ok else 0.0,
            "sentiment_fallback_used": 1.0 if sentiment_quality in ("fallback", "unavailable") else 0.0,
            "sentiment_source_unavailable": 1.0 if sentiment_quality == "unavailable" else 0.0,
            "sentiment_source": sentiment_source_tag,
            "vwap": float(vwap),
            "close": float(close),
            "vwap_tolerance_pct": float(vwap_tol),
            "funding_tolerance": float(funding_tol),
            "ratio_tolerance": float(self.config.long_short_ratio_tolerance),
        }

    def _layer5_tp_sl_levels(self, df: pd.DataFrame, side: str, vp: VolumeProfileLevels | None) -> tuple[bool, dict[str, Any]]:
        if side != "SHORT":
            return False, {"passed": 0.0, "stop_above_invalidation": 0.0, "tp_at_poc_or_better": 0.0, "atr_available": 0.0, "volume_profile_available": 1.0 if vp is not None else 0.0, "fallback_rr_used": 1.0, "invalidation_reference": 0.0, "tp_reference": "unsupported_side", "stop_distance_pct": 0.0, "take_profit_distance_pct": 0.0, "risk_reward_ratio": 0.0, "missing_conditions": "short_context_required", "failed_reason": "unsupported_side_not_short", "tp_sl_strength": 0.0}
        if df.empty:
            return False, {"passed": 0.0, "stop_above_invalidation": 0.0, "tp_at_poc_or_better": 0.0, "atr_available": 0.0, "volume_profile_available": 1.0 if vp is not None else 0.0, "fallback_rr_used": 1.0, "invalidation_reference": 0.0, "tp_reference": "history_missing", "stop_distance_pct": 0.0, "take_profit_distance_pct": 0.0, "risk_reward_ratio": 0.0, "missing_conditions": "history", "failed_reason": "insufficient_history", "tp_sl_strength": 0.0}

        last = df.iloc[-1]
        entry = self._safe(last.get("close"), 0.0)
        atr = self._safe(last.get("atr"), entry * 0.01)
        atr_available = bool(np.isfinite(atr) and atr > 0)
        if not atr_available: atr = max(entry * 0.01, 1e-8)
        if not np.isfinite(entry) or entry <= 0:
            return False, {"passed": 0.0, "stop_above_invalidation": 0.0, "tp_at_poc_or_better": 0.0, "atr_available": 1.0 if atr_available else 0.0, "volume_profile_available": 1.0 if vp is not None else 0.0, "fallback_rr_used": 1.0, "invalidation_reference": 0.0, "tp_reference": "invalid_entry", "stop_distance_pct": 0.0, "take_profit_distance_pct": 0.0, "risk_reward_ratio": 0.0, "missing_conditions": "entry_price", "failed_reason": "invalid_entry_price", "tp_sl_strength": 0.0}

        atr_sl = max(float(self.config.atr_sl_mult), 0.0)
        rr = max(float(self.config.risk_reward), 0.1)
        rr_floor = 0.02
        tol = max(float(self.config.entry_tolerance_pct), 0.0)
        base_inv = entry + atr * atr_sl
        inv_ref = base_inv
        tp_ref = "rr_fallback"
        fallback_rr = vp is None
        tp = entry - max(atr, entry * 0.0001) * rr
        if vp is not None:
            inv_ref = max(float(vp.vah), base_inv)
            vp_poc = float(vp.poc)
            if np.isfinite(vp_poc) and vp_poc > 0 and vp_poc < entry:
                tp = vp_poc
                tp_ref = "vp_poc"
                fallback_rr = False

        sl = max(inv_ref, base_inv)
        tp, sl = self._normalize_levels(entry=entry, tp=tp, sl=sl, side="SHORT")
        stop_pct = (sl - entry) / max(entry, 1e-9)
        tp_pct = (entry - tp) / max(entry, 1e-9)
        rr_val = (tp_pct / max(stop_pct, 1e-9)) if stop_pct > 0 else 0.0
        stop_ok = sl >= inv_ref and sl > entry
        tp_ok = tp <= float(vp.poc) * (1.0 + tol) if (tp_ref == "vp_poc" and vp is not None) else tp < entry
        rr_ok = rr_val >= rr_floor
        passed = stop_ok and tp_ok and rr_ok

        missing = []
        if not stop_ok: missing.append("stop_above_invalidation")
        if not tp_ok: missing.append("tp_at_poc_or_better")
        if not rr_ok: missing.append("risk_reward_ratio_soft_min")
        strength = (int(stop_ok) + int(tp_ok) + int(rr_ok)) / 3.0
        return passed, {"passed": 1.0 if passed else 0.0, "entry": float(entry), "tp": float(tp), "sl": float(sl), "partial_tps": [float(entry - atr), float((entry + tp) / 2.0)], "stop_above_invalidation": 1.0 if stop_ok else 0.0, "tp_at_poc_or_better": 1.0 if tp_ok else 0.0, "atr_available": 1.0 if atr_available else 0.0, "volume_profile_available": 1.0 if vp is not None else 0.0, "fallback_rr_used": 1.0 if fallback_rr else 0.0, "invalidation_reference": float(inv_ref), "tp_reference": str(tp_ref), "stop_distance_pct": float(stop_pct), "take_profit_distance_pct": float(tp_pct), "risk_reward_ratio": float(rr_val), "missing_conditions": ",".join(missing), "failed_reason": "none" if passed else f"missing:{','.join(missing)}", "tp_sl_strength": float(strength)}

    @staticmethod
    def _normalize_levels(entry: float, tp: float, sl: float, side: str) -> tuple[float, float]:
        min_step = max(entry * 0.0001, 1e-8)
        if side == "SHORT":
            if tp >= entry - min_step: tp = entry - min_step
            if sl <= entry + min_step: sl = entry + min_step
        else:
            if tp <= entry + min_step: tp = entry + min_step
            if sl >= entry - min_step: sl = entry - min_step
        return float(tp), float(sl)

    def generate(self, context: SignalContext) -> SignalResult | None:
        df = context.df
        trace: dict[str, Any] = {"strategy_model": "layered_table_5_softened", "failed_layer": None, "layers": {}}
        if df.empty or len(df) < 40:
            trace["failed_layer"] = "layer0_input"
            trace["layers"]["layer0_input"] = {"passed": False, "details": {"insufficient_history": 1.0}}
            self.last_diagnostics = trace
            return None

        regime_ok, regime_filter = self._regime_filter(df, context.regime, news_veto=context.news_veto, news_source=context.news_source, news_degraded=context.news_degraded)
        trace["layers"]["regime_filter"] = {"passed": regime_ok, "details": regime_filter}
        if not regime_ok:
            trace["failed_layer"] = "regime_filter"
            self.last_diagnostics = trace
            return None

        side, layer1 = self._layer1_pump_detection(df)
        trace["layers"]["layer1_pump_detection"] = {"passed": side is not None, "side": side or "", "details": layer1}
        if side is None:
            trace["failed_layer"] = "layer1_pump_detection"
            self.last_diagnostics = trace
            return None

        layer2_ok, layer2 = self._layer2_weakness_confirmation(df, side)
        trace["layers"]["layer2_weakness_confirmation"] = {"passed": layer2_ok, "details": layer2}
        if not layer2_ok:
            trace["failed_layer"] = "layer2_weakness_confirmation"
            self.last_diagnostics = trace
            return None

        layer3_ok, layer3 = self._layer3_entry_location(df, side, context.volume_profile)
        trace["layers"]["layer3_entry_location"] = {"passed": layer3_ok, "details": layer3}
        if not layer3_ok:
            trace["failed_layer"] = "layer3_entry_location"
            self.last_diagnostics = trace
            return None

        layer4_ok, layer4 = self._layer4_fake_filter(df=df, side=side, sentiment_index=context.sentiment_index, sentiment_source=context.sentiment_source, funding_rate=context.funding_rate, long_short_ratio=context.long_short_ratio, open_interest=context.open_interest, open_interest_source=context.open_interest_source, sentiment_value=context.sentiment_value, sentiment_degraded=context.sentiment_degraded, funding_source=context.funding_source, funding_degraded=context.funding_degraded, long_short_ratio_source=context.long_short_ratio_source, long_short_ratio_degraded=context.long_short_ratio_degraded, open_interest_ratio=context.open_interest_ratio, oi_signal=context.oi_signal, oi_source=context.oi_source, oi_degraded=context.oi_degraded)
        trace["layers"]["layer4_fake_filter"] = {"passed": layer4_ok, "details": layer4}
        if not layer4_ok:
            trace["failed_layer"] = "layer4_fake_filter"
            self.last_diagnostics = trace
            return None

        layer5_ok, layer5 = self._layer5_tp_sl_levels(df, side, context.volume_profile)
        trace["layers"]["layer5_tp_sl"] = {"passed": layer5_ok, "details": layer5}
        if not layer5_ok:
            trace["failed_layer"] = "layer5_tp_sl"
            self.last_diagnostics = trace
            return None

        entry = float(layer5.get("entry", self._safe(df.iloc[-1].get("close"), 0.0)))
        tp = float(layer5.get("tp", entry))
        sl = float(layer5.get("sl", entry))
        partial_tps = [float(x) for x in layer5.get("partial_tps", [])]
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
            details={"regime_filter": regime_filter, "layer1": layer1, "layer2": layer2, "layer3": layer3, "layer4": layer4, "layer5": layer5, "layer_trace": trace, "regime": context.regime.value},
        )



