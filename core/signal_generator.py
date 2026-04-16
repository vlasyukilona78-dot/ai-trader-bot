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
    volume_spike_threshold: float = 2.0
    layer1_pump_lookback_bars: int = 1
    layer1_clean_pump_lookback_bars: int = 48
    layer1_clean_pump_min_pct: float = 0.05
    early_watch_clean_pump_min_pct: float = 0.04
    early_watch_volume_spike_min: float = 1.15
    early_watch_rsi_min: float = 52.0
    early_watch_quality_min: float = 4.5
    layer1_soft_pass_enabled: bool = False
    layer1_rsi_soft: float = 50.0
    layer1_volume_spike_soft: float = 1.35
    layer1_upper_band_proximity_pct: float = 0.001
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
    msb_recent_bars: int = 8
    msb_break_buffer_pct: float = 0.0005
    regime_vwap_stretch_soft: float = 0.0012
    regime_min_atr_norm: float = 0.0008
    regime_volatility_entry_tolerance_mult: float = 0.25
    regime_volatility_threshold_override: float | None = None
    regime_volatility_dynamic_floor_mult: float = 1.0
    regime_volatility_baseline_lookback: int = 96
    regime_soft_pass_enabled: bool = False
    regime_strong_trend_adx: float = 30.0
    regime_mtf_hard_filter_enabled: bool = True
    regime_mtf_trend_15m_max_short: float = 0.0020
    regime_mtf_trend_5m_max_short: float = 0.0014
    regime_mtf_rsi_15m_max_short: float = 64.0
    regime_mtf_rsi_5m_max_short: float = 70.0
    fake_filter_lsr_extreme_soft: float = 1.01
    fake_filter_oi_volume_spike_soft: float = 1.2
    allow_long_entries: bool = False


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
    strategy: str = "pump_short_profile"
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
            src = {
                "news_source": "unavailable",
                "news_available": 0.0,
                "vwap_available": 0.0,
                **self._quality_flags("news", "unavailable"),
                **self._quality_flags("vwap", "unavailable"),
            }
            return False, {
                "passed": 0.0,
                "htf_trend_ok": 0.0,
                "stretched_from_vwap": 0.0,
                "volatility_regime_ok": 0.0,
                "news_veto": 1.0,
                "htf_trend_metric_used": 0.0,
                "htf_trend_threshold_used": float(self.config.regime_strong_trend_adx),
                "htf_trend_direction_context": "unavailable",
                "volatility_threshold_used": 0.0,
                "atr_norm": 0.0,
                "missing_conditions": "history",
                "failed_reason": "insufficient_history",
                "degraded_mode": 1.0,
                "source_flags": src,
            }

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

        htf_trend_threshold = float(self.config.regime_strong_trend_adx)
        htf_uptrend_structure = ema20 >= ema50 and close >= ema20
        htf_strong_uptrend = htf_uptrend_structure and adx >= htf_trend_threshold
        mtf_hard_enabled = bool(self.config.regime_mtf_hard_filter_enabled)
        mtf_trend_5m = self._safe(last.get("mtf_trend_5m"), 0.0)
        mtf_trend_15m = self._safe(last.get("mtf_trend_15m"), 0.0)
        mtf_rsi_5m = self._safe(last.get("mtf_rsi_5m"), 50.0)
        mtf_rsi_15m = self._safe(last.get("mtf_rsi_15m"), 50.0)
        mtf_15m_strong_up = (
            mtf_trend_15m >= float(self.config.regime_mtf_trend_15m_max_short)
            and mtf_rsi_15m >= float(self.config.regime_mtf_rsi_15m_max_short)
        )
        mtf_5m_continuation_up = (
            mtf_trend_5m >= float(self.config.regime_mtf_trend_5m_max_short)
            and mtf_rsi_5m >= float(self.config.regime_mtf_rsi_5m_max_short)
        )
        mtf_short_context_ok = not (mtf_hard_enabled and (mtf_15m_strong_up or mtf_5m_continuation_up))
        if regime == MarketRegime.PANIC:
            htf_direction_context = "panic_regime"
            htf_trend_ok = False
        elif htf_strong_uptrend:
            htf_direction_context = "strong_uptrend"
            htf_trend_ok = False
        elif not mtf_short_context_ok:
            htf_direction_context = "mtf_uptrend_block"
            htf_trend_ok = False
        else:
            htf_direction_context = "not_strong_uptrend"
            htf_trend_ok = True
        vwap_stretch_threshold = max(
            float(self.config.regime_vwap_stretch_soft),
            float(self.config.vwap_tolerance_pct) * 0.5,
        )
        stretched = vwap_dist >= vwap_stretch_threshold
        volatility_threshold_override = self.config.regime_volatility_threshold_override
        if volatility_threshold_override is None:
            base_volatility_threshold = max(
                float(self.config.regime_min_atr_norm),
                float(self.config.entry_tolerance_pct) * float(self.config.regime_volatility_entry_tolerance_mult),
            )
        else:
            base_volatility_threshold = max(0.0, float(volatility_threshold_override))
        baseline_lookback = max(24, int(self.config.regime_volatility_baseline_lookback))
        dynamic_floor_mult = min(max(float(self.config.regime_volatility_dynamic_floor_mult), 0.5), 1.0)
        atr_series = pd.to_numeric(df.get("atr"), errors="coerce")
        close_series = pd.to_numeric(df.get("close"), errors="coerce").replace(0.0, np.nan)
        atr_norm_series = (atr_series / close_series).replace([np.inf, -np.inf], np.nan).dropna()
        atr_norm_baseline = (
            float(atr_norm_series.tail(baseline_lookback).median())
            if not atr_norm_series.empty
            else float(atr_norm)
        )
        dynamic_floor = float(base_volatility_threshold) * dynamic_floor_mult
        if np.isfinite(atr_norm_baseline) and atr_norm_baseline > 0:
            volatility_threshold = max(dynamic_floor, min(float(base_volatility_threshold), atr_norm_baseline * 0.95))
        else:
            volatility_threshold = float(base_volatility_threshold)
        vol_ok = atr_norm >= volatility_threshold

        news_participates = news_veto is not None
        news_source_tag = self._source_tag(news_source, fallback="provided" if news_participates else "unavailable")
        news_quality = self._source_quality(news_source_tag, value_present=news_participates, degraded=news_degraded)
        news_ok = True if news_veto is None else (not bool(news_veto))

        vwap_quality = "live" if vwap_available else "fallback"
        # Missing optional news should not mark the whole regime gate as degraded.
        degraded = vwap_quality != "live" or (news_participates and news_quality != "live")

        missing: list[str] = []
        if not htf_trend_ok:
            missing.append("htf_trend_ok")
        if not mtf_short_context_ok:
            missing.append("mtf_short_context_ok")
        if not stretched:
            missing.append("stretched_from_vwap")
        if not vol_ok:
            missing.append("volatility_regime_ok")
        if not news_ok:
            missing.append("news_veto")
        strict_passed = htf_trend_ok and stretched and vol_ok and news_ok
        sweep_ctx = self._short_sweep_reclaim_context(df)
        sweep_trigger_ready = bool(
            (sweep_ctx.get("failed_reclaim") or sweep_ctx.get("retest_failed_breakout"))
            or (
                sweep_ctx.get("near_sweep_level")
                and sweep_ctx.get("rejection_bar")
                and sweep_ctx.get("post_peak_followthrough")
            )
        )
        sweep_soft_pass_candidate = bool(
            self.config.regime_soft_pass_enabled
            and (not strict_passed)
            and (not degraded)
            and htf_trend_ok
            and mtf_short_context_ok
            and news_ok
            and sweep_trigger_ready
            and (not bool(sweep_ctx.get("acceptance_above_high")))
            and len(missing) >= 1
            and all(item in ("stretched_from_vwap", "volatility_regime_ok") for item in missing)
        )
        soft_pass_candidate = bool(
            (not strict_passed)
            and (not degraded)
            and (
                (htf_trend_ok and mtf_short_context_ok and (int(stretched) + int(vol_ok) + int(news_ok) >= 2))
                or ((not htf_trend_ok) and mtf_short_context_ok and stretched and vol_ok and news_ok)
            )
        )
        soft_pass_used = bool(
            self.config.regime_soft_pass_enabled
            and (not strict_passed)
            and (not degraded)
            and mtf_short_context_ok
            and (
                (len(missing) == 1 and missing[0] in ("stretched_from_vwap", "htf_trend_ok"))
                or sweep_soft_pass_candidate
            )
        )
        passed = strict_passed or soft_pass_used
        fail_due_to_degraded_only = (not strict_passed) and degraded and not missing

        src = {
            "news_source": news_source_tag,
            "news_available": 1.0 if news_quality == "live" else 0.0,
            "vwap_available": 1.0 if vwap_available else 0.0,
            **self._quality_flags("news", news_quality),
            **self._quality_flags("vwap", vwap_quality),
        }
        return passed, {
            "passed": 1.0 if passed else 0.0,
            "htf_trend_ok": 1.0 if htf_trend_ok else 0.0,
            "mtf_short_context_ok": 1.0 if mtf_short_context_ok else 0.0,
            "stretched_from_vwap": 1.0 if stretched else 0.0,
            "volatility_regime_ok": 1.0 if vol_ok else 0.0,
            "news_veto": 1.0 if news_ok else 0.0,
            "htf_trend_metric_used": float(adx),
            "htf_trend_threshold_used": float(htf_trend_threshold),
            "htf_trend_direction_context": str(htf_direction_context),
            "mtf_trend_5m_used": float(mtf_trend_5m),
            "mtf_trend_15m_used": float(mtf_trend_15m),
            "mtf_rsi_5m_used": float(mtf_rsi_5m),
            "mtf_rsi_15m_used": float(mtf_rsi_15m),
            "mtf_hard_filter_enabled": 1.0 if mtf_hard_enabled else 0.0,
            "vwap_distance_metric_used": float(vwap_dist),
            "vwap_stretch_threshold_used": float(vwap_stretch_threshold),
            "volatility_threshold_used": float(volatility_threshold),
            "volatility_threshold_base_used": float(base_volatility_threshold),
            "volatility_baseline_atr_norm_used": float(atr_norm_baseline),
            "volatility_dynamic_floor_mult_used": float(dynamic_floor_mult),
            "atr_norm": float(atr_norm),
            "degraded_mode": 1.0 if degraded else 0.0,
            "fail_due_to_degraded_mode_only": 1.0 if fail_due_to_degraded_only else 0.0,
            "soft_pass_candidate": 1.0 if (soft_pass_candidate or sweep_soft_pass_candidate) else 0.0,
            "soft_pass_used": 1.0 if soft_pass_used else 0.0,
            "soft_pass_reason": (
                "sweep_volatility_context"
                if soft_pass_used and sweep_soft_pass_candidate
                else (missing[0] if soft_pass_used and len(missing) == 1 else "")
            ),
            "missing_conditions": "" if passed else ",".join(missing),
            "failed_reason": "none" if passed else f"missing:{','.join(missing)}",
            "failed_reclaim": 1.0 if bool(sweep_ctx.get("failed_reclaim")) else 0.0,
            "retest_failed_breakout": 1.0 if bool(sweep_ctx.get("retest_failed_breakout")) else 0.0,
            "rejection_bar": 1.0 if bool(sweep_ctx.get("rejection_bar")) else 0.0,
            "near_sweep_level": 1.0 if bool(sweep_ctx.get("near_sweep_level")) else 0.0,
            "acceptance_above_swing_high": 1.0 if bool(sweep_ctx.get("acceptance_above_high")) else 0.0,
            "source_flags": src,
        }

    def _layer1_pump_detection(self, df: pd.DataFrame) -> tuple[str | None, dict[str, Any]]:
        if df.empty:
            return None, {
                "passed": 0.0,
                "failed_reason": "insufficient_history",
                "missing_conditions": "history",
                "rsi": None,
                "rsi_high": 0.0,
                "rsi_high_threshold_used": float(self.config.rsi_high),
                "volume_spike": None,
                "volume_spike_high": 0.0,
                "volume_spike_threshold_used": float(self.config.volume_spike_threshold),
                "close_metric_used": None,
                "bollinger_upper_metric_used": None,
                "keltner_upper_metric_used": None,
                "above_bollinger_upper": 0.0,
                "above_keltner_upper": 0.0,
                "upper_band_breakout": 0.0,
                "pump_context_strength": 0.0,
                "clean_pump_pct": 0.0,
                "clean_pump_min_pct_used": float(self.config.layer1_clean_pump_min_pct),
                "clean_pump_ok": 0.0,
                "soft_pass_candidate": 0.0,
                "soft_pass_used": 0.0,
                "soft_pass_reason": "",
                "pump_bar_offset": None,
                "layer1_subconditions_state": {},
            }
        rsi_high_threshold = float(self.config.rsi_high)
        volume_spike_threshold = float(self.config.volume_spike_threshold)
        rsi_soft_threshold = float(self.config.layer1_rsi_soft)
        volume_spike_soft_threshold = float(self.config.layer1_volume_spike_soft)
        upper_band_proximity_pct = max(0.0, float(self.config.layer1_upper_band_proximity_pct))
        lookback = max(1, int(self.config.layer1_pump_lookback_bars))
        clean_lookback = max(lookback, int(self.config.layer1_clean_pump_lookback_bars))
        clean_pump_threshold = max(0.0, float(self.config.layer1_clean_pump_min_pct))
        window = df.tail(lookback)
        support_window = df.tail(max(lookback, 6))
        clean_window = df.tail(clean_lookback)
        clean_close = pd.to_numeric(clean_window.get("close"), errors="coerce").dropna()
        clean_pump_pct = 0.0
        if len(clean_close) >= 2:
            rolling_min = clean_close.cummin().replace(0.0, np.nan)
            pump_track = ((clean_close - rolling_min) / rolling_min).replace([np.inf, -np.inf], np.nan).dropna()
            if not pump_track.empty:
                clean_pump_pct = float(max(0.0, pump_track.max()))
        clean_pump_ok = clean_pump_pct >= clean_pump_threshold
        near_clean_pump_ok = clean_pump_pct >= max(clean_pump_threshold * 0.84, clean_pump_threshold - 0.010)
        sweep_ctx = self._short_sweep_reclaim_context(df)
        failed_reclaim_ctx = bool(sweep_ctx.get("failed_reclaim"))
        retest_failed_breakout_ctx = bool(sweep_ctx.get("retest_failed_breakout"))
        acceptance_above_high_ctx = bool(sweep_ctx.get("acceptance_above_high"))
        rejection_bar_ctx = bool(sweep_ctx.get("rejection_bar"))
        near_sweep_level_ctx = bool(sweep_ctx.get("near_sweep_level"))
        post_peak_followthrough_ctx = bool(sweep_ctx.get("post_peak_followthrough"))

        def _details_from_row(row: pd.Series, *, pump_bar_offset: int) -> tuple[bool, dict[str, Any]]:
            rsi = self._safe(row.get("rsi"), 50.0)
            vol = self._safe(row.get("volume_spike"), 1.0)
            close = self._safe(row.get("close"), 0.0)
            bb_u = self._safe(row.get("bb_upper"), np.inf)
            kc_u = self._safe(row.get("kc_upper"), np.inf)

            rsi_high = rsi >= rsi_high_threshold
            vol_high = vol >= volume_spike_threshold
            above_bb = bool(np.isfinite(bb_u) and close > bb_u)
            above_kc = bool(np.isfinite(kc_u) and close > kc_u)
            band_break = above_bb or above_kc
            near_bb = bool(np.isfinite(bb_u) and close >= bb_u * (1.0 - upper_band_proximity_pct))
            near_kc = bool(np.isfinite(kc_u) and close >= kc_u * (1.0 - upper_band_proximity_pct))
            near_upper_band = near_bb or near_kc
            rsi_soft = rsi >= rsi_soft_threshold
            vol_soft = vol >= volume_spike_soft_threshold

            pts = int(rsi_high) + int(vol_high) + int(band_break)
            recent_sweep_bar = pump_bar_offset <= 2
            sweep_trigger_ready = bool(
                recent_sweep_bar
                and (not acceptance_above_high_ctx)
                and (
                    failed_reclaim_ctx
                    or retest_failed_breakout_ctx
                    or (near_sweep_level_ctx and rejection_bar_ctx and post_peak_followthrough_ctx)
                )
            )
            sweep_fast_pass = bool(
                sweep_trigger_ready
                and (clean_pump_ok or near_clean_pump_ok)
                and (rsi_high or rsi_soft or near_upper_band)
                and (vol_high or vol_soft)
            )
            passed = (clean_pump_ok and band_break and pts >= 2) or sweep_fast_pass
            soft_pass_candidate = clean_pump_ok and (not passed) and pts >= 2
            fade_soft_candidate = clean_pump_ok and (not passed) and near_upper_band and rsi_soft and vol_soft
            near_threshold_soft_candidate = (
                near_clean_pump_ok
                and not clean_pump_ok
                and (not passed)
                and near_upper_band
                and rsi_soft
                and vol_soft
            )
            sweep_soft_candidate = bool(
                sweep_trigger_ready
                and not sweep_fast_pass
                and (clean_pump_ok or near_clean_pump_ok)
                and (rsi_soft or near_upper_band)
                and vol_soft
            )
            missing = []
            if not rsi_high:
                missing.append("rsi_high")
            if not vol_high:
                missing.append("volume_spike")
            if not band_break:
                missing.append("upper_band_breakout")
            if not clean_pump_ok:
                missing.append("clean_pump_pct")
            subconditions = {
                "rsi_high": bool(rsi_high),
                "volume_spike_high": bool(vol_high),
                "upper_band_breakout": bool(band_break),
                "above_bollinger_upper": bool(above_bb),
                "above_keltner_upper": bool(above_kc),
                "clean_pump_ok": bool(clean_pump_ok),
            }
            return passed, {
                "passed": 1.0 if passed else 0.0,
                "failed_reason": "none" if passed else f"missing:{','.join(missing)}",
                "missing_conditions": "" if passed else ",".join(missing),
                "rsi": float(rsi),
                "rsi_high": 1.0 if rsi_high else 0.0,
                "rsi_high_threshold_used": float(rsi_high_threshold),
                "volume_spike": float(vol),
                "volume_spike_high": 1.0 if vol_high else 0.0,
                "volume_spike_threshold_used": float(volume_spike_threshold),
                "close_metric_used": float(close),
                "bollinger_upper_metric_used": float(bb_u) if np.isfinite(bb_u) else None,
                "keltner_upper_metric_used": float(kc_u) if np.isfinite(kc_u) else None,
                "above_bollinger_upper": 1.0 if above_bb else 0.0,
                "above_keltner_upper": 1.0 if above_kc else 0.0,
                "upper_band_breakout": 1.0 if band_break else 0.0,
                "near_upper_band": 1.0 if near_upper_band else 0.0,
                "pump_context_strength": float(pts / 3.0),
                "clean_pump_pct": float(clean_pump_pct),
                "clean_pump_min_pct_used": float(clean_pump_threshold),
                "clean_pump_ok": 1.0 if clean_pump_ok else 0.0,
                "near_clean_pump_ok": 1.0 if near_clean_pump_ok else 0.0,
                "failed_reclaim": 1.0 if failed_reclaim_ctx else 0.0,
                "retest_failed_breakout": 1.0 if retest_failed_breakout_ctx else 0.0,
                "rejection_bar": 1.0 if rejection_bar_ctx else 0.0,
                "near_sweep_level": 1.0 if near_sweep_level_ctx else 0.0,
                "acceptance_above_swing_high": 1.0 if acceptance_above_high_ctx else 0.0,
                "soft_pass_candidate": 1.0 if (soft_pass_candidate or sweep_soft_candidate) else 0.0,
                "soft_pass_used": 1.0 if sweep_fast_pass else 0.0,
                "soft_pass_reason": "",
                "_soft_pass_reason_candidate": (
                    "sweep_reclaim_context"
                    if sweep_soft_candidate or sweep_fast_pass
                    else (
                        "upper_band_breakout"
                        if soft_pass_candidate
                        else (
                            "window_near_threshold_pump_context"
                            if near_threshold_soft_candidate
                            else ("near_upper_band_context" if fade_soft_candidate else "")
                        )
                    )
                ),
                "pump_bar_offset": int(pump_bar_offset),
                "layer1_subconditions_state": subconditions,
            }

        fallback_details: dict[str, Any] | None = None
        soft_pass_details: dict[str, Any] | None = None
        window_details: list[dict[str, Any]] = []
        for idx in range(len(window) - 1, -1, -1):
            pump_bar_offset = len(window) - 1 - idx
            passed, details = _details_from_row(window.iloc[idx], pump_bar_offset=pump_bar_offset)
            window_details.append(details)
            if pump_bar_offset == 0:
                fallback_details = details
            if passed:
                return "SHORT", details
            if soft_pass_details is None and str(details.get("_soft_pass_reason_candidate") or ""):
                soft_pass_details = details

        if len(support_window) > len(window):
            support_window_details = list(window_details)
            for idx in range(len(support_window) - len(window) - 1, -1, -1):
                pump_bar_offset = len(support_window) - 1 - idx
                _, details = _details_from_row(support_window.iloc[idx], pump_bar_offset=pump_bar_offset)
                support_window_details.append(details)
                if soft_pass_details is None and str(details.get("_soft_pass_reason_candidate") or ""):
                    soft_pass_details = details
        else:
            support_window_details = window_details

        recent_rsi_support = any(
            float(details.get("rsi", 0.0) or 0.0) >= rsi_soft_threshold for details in support_window_details
        )
        recent_volume_support = any(
            float(details.get("volume_spike", 0.0) or 0.0) >= volume_spike_soft_threshold for details in support_window_details
        )
        recent_upper_band_support = any(
            bool(float(details.get("upper_band_breakout", 0.0) or 0.0))
            or bool(float(details.get("near_upper_band", 0.0) or 0.0))
            for details in support_window_details
        )
        sweep_expansion_context = bool(
            recent_upper_band_support
            and recent_rsi_support
            and (
                recent_volume_support
                or any(
                    float(details.get("volume_spike", 0.0) or 0.0) >= max(volume_spike_soft_threshold * 0.82, 0.6)
                    for details in support_window_details
                )
            )
        )
        sweep_window_pass = bool(
            not acceptance_above_high_ctx
            and (clean_pump_ok or (near_clean_pump_ok and sweep_expansion_context))
            and (
                failed_reclaim_ctx
                or retest_failed_breakout_ctx
                or (near_sweep_level_ctx and rejection_bar_ctx and post_peak_followthrough_ctx)
            )
            and recent_rsi_support
            and recent_upper_band_support
            and (recent_volume_support or sweep_expansion_context)
        )
        if self.config.layer1_soft_pass_enabled and sweep_window_pass:
            reference_details = next(
                (
                    details
                    for details in support_window_details
                    if bool(float(details.get("upper_band_breakout", 0.0) or 0.0))
                    or (
                        bool(float(details.get("near_upper_band", 0.0) or 0.0))
                        and float(details.get("volume_spike", 0.0) or 0.0) >= max(volume_spike_soft_threshold * 0.82, 0.6)
                        and float(details.get("rsi", 0.0) or 0.0) >= rsi_soft_threshold
                    )
                ),
                max(
                    support_window_details,
                    key=lambda details: (
                        float(details.get("pump_context_strength", 0.0) or 0.0),
                        -int(details.get("pump_bar_offset", 0) or 0),
                    ),
                ),
            )
            used_details = dict(reference_details)
            used_details["passed"] = 1.0
            used_details["failed_reason"] = "none"
            used_details["missing_conditions"] = ""
            used_details["soft_pass_candidate"] = 1.0
            used_details["soft_pass_used"] = 1.0
            used_details["soft_pass_reason"] = "sweep_reclaim_window_context"
            used_details["failed_reclaim"] = 1.0 if failed_reclaim_ctx else 0.0
            used_details["retest_failed_breakout"] = 1.0 if retest_failed_breakout_ctx else 0.0
            used_details["rejection_bar"] = 1.0 if rejection_bar_ctx else 0.0
            used_details["near_sweep_level"] = 1.0 if near_sweep_level_ctx else 0.0
            used_details["acceptance_above_swing_high"] = 1.0 if acceptance_above_high_ctx else 0.0
            used_details["layer1_subconditions_state"] = {
                "rsi_high": bool(recent_rsi_support),
                "volume_spike_high": bool(recent_volume_support),
                "upper_band_breakout": bool(recent_upper_band_support),
                "above_bollinger_upper": any(
                    bool(float(details.get("above_bollinger_upper", 0.0) or 0.0)) for details in support_window_details
                ),
                "above_keltner_upper": any(
                    bool(float(details.get("above_keltner_upper", 0.0) or 0.0)) for details in support_window_details
                ),
                "clean_pump_ok": True,
            }
            return "SHORT", used_details

        current_soft_reason = str((fallback_details or {}).get("_soft_pass_reason_candidate") or "")
        if self.config.layer1_soft_pass_enabled and current_soft_reason == "near_upper_band_context":
            used_details = dict(fallback_details or {})
            used_details["passed"] = 1.0
            used_details["failed_reason"] = "none"
            used_details["missing_conditions"] = ""
            used_details["soft_pass_candidate"] = 1.0
            used_details["soft_pass_used"] = 1.0
            used_details["soft_pass_reason"] = "near_upper_band_context"
            return "SHORT", used_details

        if self.config.layer1_soft_pass_enabled and (clean_pump_ok or near_clean_pump_ok) and support_window_details:
            recent_rsi_high = any(bool(float(details.get("rsi_high", 0.0) or 0.0)) for details in support_window_details)
            recent_volume_spike = any(bool(float(details.get("volume_spike_high", 0.0) or 0.0)) for details in support_window_details)
            recent_upper_band_break = any(bool(float(details.get("upper_band_breakout", 0.0) or 0.0)) for details in support_window_details)
            recent_near_upper_band = any(bool(float(details.get("near_upper_band", 0.0) or 0.0)) for details in support_window_details)
            recent_above_bb = any(bool(float(details.get("above_bollinger_upper", 0.0) or 0.0)) for details in support_window_details)
            recent_above_kc = any(bool(float(details.get("above_keltner_upper", 0.0) or 0.0)) for details in support_window_details)
            if (recent_upper_band_break and (recent_rsi_high or recent_volume_spike)) or (
                near_clean_pump_ok and recent_near_upper_band and recent_rsi_high and recent_volume_spike
            ):
                reference_details = next(
                    (
                        details
                        for details in support_window_details
                        if bool(float(details.get("upper_band_breakout", 0.0) or 0.0))
                        or (
                            near_clean_pump_ok
                            and bool(float(details.get("near_upper_band", 0.0) or 0.0))
                            and bool(float(details.get("rsi_high", 0.0) or 0.0))
                            and bool(float(details.get("volume_spike_high", 0.0) or 0.0))
                        )
                    ),
                    max(
                        support_window_details,
                        key=lambda details: (
                            float(details.get("pump_context_strength", 0.0) or 0.0),
                            -int(details.get("pump_bar_offset", 0) or 0),
                        ),
                    ),
                )
                used_details = dict(reference_details)
                used_details["passed"] = 1.0
                used_details["failed_reason"] = "none"
                used_details["missing_conditions"] = ""
                used_details["soft_pass_candidate"] = 1.0
                used_details["soft_pass_used"] = 1.0
                used_details["soft_pass_reason"] = (
                    "window_near_threshold_pump_context"
                    if near_clean_pump_ok and not clean_pump_ok
                    else "window_pump_context"
                )
                used_details["pump_context_strength"] = float(
                    (
                        int(recent_rsi_high)
                        + int(recent_volume_spike)
                        + int(recent_upper_band_break or recent_near_upper_band)
                    ) / 3.0
                )
                used_details["layer1_subconditions_state"] = {
                    "rsi_high": bool(recent_rsi_high),
                    "volume_spike_high": bool(recent_volume_spike),
                    "upper_band_breakout": bool(recent_upper_band_break or recent_near_upper_band),
                    "above_bollinger_upper": bool(recent_above_bb),
                    "above_keltner_upper": bool(recent_above_kc),
                }
                return "SHORT", used_details

        if self.config.layer1_soft_pass_enabled and soft_pass_details is not None:
            used_details = dict(soft_pass_details)
            used_details["passed"] = 1.0
            used_details["failed_reason"] = "none"
            used_details["missing_conditions"] = ""
            used_details["soft_pass_used"] = 1.0
            used_details["soft_pass_reason"] = str(used_details.get("_soft_pass_reason_candidate") or "upper_band_breakout")
            return "SHORT", used_details

        return None, (fallback_details or _details_from_row(window.iloc[-1], pump_bar_offset=0)[1])

    def _short_sweep_reclaim_context(self, df: pd.DataFrame) -> dict[str, Any]:
        default = {
            "swept_high": False,
            "failed_reclaim": False,
            "retest_failed_breakout": False,
            "acceptance_above_high": False,
            "rejection_bar": False,
            "lower_close": False,
            "lower_high": False,
            "post_peak_followthrough": False,
            "near_sweep_level": False,
            "swing_high": 0.0,
            "sweep_high": 0.0,
            "sweep_buffer": 0.0,
            "distance_from_swing_high_pct": 0.0,
        }
        if df.empty or len(df) < 4:
            return default

        recent = df.tail(min(len(df), 14)).copy()
        if len(recent) < 4:
            return default

        last = recent.iloc[-1]
        prev = recent.iloc[-2]
        prior = recent.iloc[:-1]
        prior_for_sweep = recent.iloc[:-2] if len(recent) > 4 else recent.iloc[:-1]

        high_series = pd.to_numeric(prior_for_sweep.get("high"), errors="coerce").dropna()
        if high_series.empty:
            return default
        local_prior = recent.tail(min(len(recent), 6)).iloc[:-2]
        local_high_series = pd.to_numeric(local_prior.get("high"), errors="coerce").dropna()
        if not local_high_series.empty:
            swing_high = float(local_high_series.max())
        else:
            swing_high = float(high_series.tail(min(len(high_series), 6)).max())
        last_close = self._safe(last.get("close"), 0.0)
        prev_close = self._safe(prev.get("close"), last_close)
        last_high = self._safe(last.get("high"), last_close)
        prev_high = self._safe(prev.get("high"), prev_close)
        last_open = self._safe(last.get("open"), last_close)
        last_low = self._safe(last.get("low"), last_close)
        atr = max(self._safe(last.get("atr"), last_close * 0.01), last_close * 0.001, 1e-8)
        tol = max(float(self.config.entry_tolerance_pct), 0.0)

        sweep_buffer = max(atr * 0.25, swing_high * max(tol * 1.5, 0.0012), 1e-8)
        sweep_high = max(last_high, prev_high)
        swept_high = bool(sweep_high >= swing_high + sweep_buffer * 0.35)
        tested_high = bool(sweep_high >= swing_high - sweep_buffer * 0.08)
        rejection_range = max(last_high - last_low, 1e-8)
        upper_wick = max(last_high - max(last_open, last_close), 0.0)
        rejection_bar = bool(upper_wick / rejection_range >= 0.28 and last_close <= last_open)
        lower_close = bool(last_close < prev_close)
        lower_high = bool(last_high < prev_high)
        near_sweep_level = bool(last_close >= swing_high * (1.0 - max(tol * 4.0, 0.006)))
        failed_reclaim = bool(
            (swept_high or tested_high)
            and last_close <= swing_high + sweep_buffer * 0.18
            and (lower_close or lower_high or rejection_bar)
        )
        retest_failed_breakout = bool(
            prev_high >= swing_high - sweep_buffer * 0.08
            and prev_close <= swing_high + sweep_buffer * 0.22
            and last_close <= prev_close * (1.0 + tol * 0.35)
            and (lower_close or lower_high or rejection_bar)
        )
        acceptance_above_high = bool(
            last_close >= swing_high + sweep_buffer * 0.65
            and prev_close >= swing_high + sweep_buffer * 0.30
        )
        post_peak_followthrough = bool(lower_close or lower_high or rejection_bar or failed_reclaim)
        distance_from_swing_high_pct = max(0.0, abs(last_close - swing_high) / max(swing_high, 1e-8))

        return {
            "swept_high": swept_high,
            "failed_reclaim": failed_reclaim,
            "retest_failed_breakout": retest_failed_breakout,
            "acceptance_above_high": acceptance_above_high,
            "rejection_bar": rejection_bar,
            "lower_close": lower_close,
            "lower_high": lower_high,
            "post_peak_followthrough": post_peak_followthrough,
            "near_sweep_level": near_sweep_level,
            "swing_high": float(swing_high),
            "sweep_high": float(sweep_high),
            "sweep_buffer": float(sweep_buffer),
            "distance_from_swing_high_pct": float(distance_from_swing_high_pct),
        }

    def _layer2_weakness_confirmation(self, df: pd.DataFrame, side: str) -> tuple[bool, dict[str, Any]]:
        if side != "SHORT":
            return False, {
                "passed": 0.0,
                "price_up_or_near_high": 0.0,
                "price_up": 0.0,
                "near_high_context": 0.0,
                "obv_bearish_divergence": 0.0,
                "cvd_bearish_divergence": 0.0,
                "close_last_used": None,
                "close_ref_used": None,
                "obv_last_used": None,
                "obv_ref_used": None,
                "cvd_last_used": None,
                "cvd_ref_used": None,
                "weakness_lookback_used": float(self.config.weakness_lookback),
                "weakness_strength": 0.0,
                "missing_conditions": "short_context_required",
                "failed_reason": "unsupported_side_not_short",
                "layer2_subconditions_state": {},
            }
        lb = int(self.config.weakness_lookback)
        if len(df) < lb + 2:
            return False, {
                "passed": 0.0,
                "price_up_or_near_high": 0.0,
                "price_up": 0.0,
                "near_high_context": 0.0,
                "obv_bearish_divergence": 0.0,
                "cvd_bearish_divergence": 0.0,
                "close_last_used": None,
                "close_ref_used": None,
                "obv_last_used": None,
                "obv_ref_used": None,
                "cvd_last_used": None,
                "cvd_ref_used": None,
                "weakness_lookback_used": float(lb),
                "weakness_strength": 0.0,
                "missing_conditions": "history",
                "failed_reason": "insufficient_history",
                "layer2_subconditions_state": {},
            }

        last = df.iloc[-1]
        prev = df.iloc[-2]
        ref = df.iloc[-1 - lb]
        close_last = self._safe(last.get("close"), 0.0)
        close_ref = self._safe(ref.get("close"), close_last)
        recent_high = float(pd.to_numeric(df.tail(lb + 3).get("high"), errors="coerce").max())
        price_up = close_last > close_ref
        near = close_last >= max(close_ref, recent_high) * (1.0 - max(float(self.config.entry_tolerance_pct) * 4.0, 0.05))
        sweep_ctx = self._short_sweep_reclaim_context(df)
        price_ctx = price_up or near

        obv_div = self._safe(last.get("obv"), 0.0) < self._safe(ref.get("obv"), 0.0)
        cvd_div = self._safe(last.get("cvd"), 0.0) < self._safe(ref.get("cvd"), 0.0)
        rsi_last = self._safe(last.get("rsi"), 50.0)
        rsi_prev = self._safe(prev.get("rsi"), rsi_last)
        hist_last = self._safe(last.get("hist"), 0.0)
        hist_prev = self._safe(prev.get("hist"), hist_last)
        rsi_rollover = bool(rsi_last < rsi_prev and rsi_last >= max(self.config.rsi_high * 0.72, 52.0))
        hist_rollover = bool(hist_last < hist_prev)
        lower_close = bool(sweep_ctx.get("lower_close"))
        lower_high = bool(sweep_ctx.get("lower_high"))
        rejection_bar = bool(sweep_ctx.get("rejection_bar"))
        failed_reclaim = bool(sweep_ctx.get("failed_reclaim"))
        retest_failed_breakout = bool(sweep_ctx.get("retest_failed_breakout"))
        acceptance_above_high = bool(sweep_ctx.get("acceptance_above_high"))

        flow_points = int(obv_div) + int(cvd_div)
        rollover_points = int(rsi_rollover) + int(hist_rollover) + int(lower_close) + int(lower_high) + int(rejection_bar)
        rollover_confirm = bool(failed_reclaim or retest_failed_breakout or rollover_points >= 2)
        flow_confirm = bool(obv_div or cvd_div)
        prev_close = self._safe(prev.get("close"), close_last)
        clean_breakout_continuation = bool(
            price_ctx
            and not failed_reclaim
            and not retest_failed_breakout
            and not rejection_bar
            and not lower_close
            and not lower_high
            and not acceptance_above_high
            and close_last >= prev_close * (1.0 - max(float(self.config.entry_tolerance_pct) * 0.25, 0.0003))
            and rsi_last >= rsi_prev
            and hist_last >= hist_prev
        )
        passed = bool(
            price_ctx
            and not acceptance_above_high
            and not clean_breakout_continuation
            and (
                failed_reclaim
                or retest_failed_breakout
                or (flow_confirm and rollover_points >= 1)
                or rollover_points >= 3
                or (flow_confirm and bool(sweep_ctx.get("near_sweep_level")))
            )
        )
        core_strength_points = int(price_ctx) + int(obv_div) + int(cvd_div)
        confirmation_points = int(rollover_confirm) if core_strength_points > 0 else 0
        strength = (core_strength_points + confirmation_points) / 4.0
        missing = []
        if not price_ctx:
            missing.append("price_up_or_near_high")
        if not obv_div:
            missing.append("obv_bearish_divergence")
        if not cvd_div:
            missing.append("cvd_bearish_divergence")
        if not rollover_confirm and not flow_confirm:
            missing.append("weakness_confirmation")
        if clean_breakout_continuation:
            missing.append("no_trade_breakout_continuation")
        if acceptance_above_high:
            missing.append("acceptance_above_swing_high")

        return passed, {
            "passed": 1.0 if passed else 0.0,
            "price_up_or_near_high": 1.0 if price_ctx else 0.0,
            "price_up": 1.0 if price_up else 0.0,
            "near_high_context": 1.0 if near else 0.0,
            "obv_bearish_divergence": 1.0 if obv_div else 0.0,
            "cvd_bearish_divergence": 1.0 if cvd_div else 0.0,
            "rsi_rollover": 1.0 if rsi_rollover else 0.0,
            "hist_rollover": 1.0 if hist_rollover else 0.0,
            "lower_close_after_peak": 1.0 if lower_close else 0.0,
            "lower_high_after_peak": 1.0 if lower_high else 0.0,
            "rejection_bar": 1.0 if rejection_bar else 0.0,
            "failed_reclaim": 1.0 if failed_reclaim else 0.0,
            "retest_failed_breakout": 1.0 if retest_failed_breakout else 0.0,
            "no_trade_breakout_continuation": 1.0 if clean_breakout_continuation else 0.0,
            "acceptance_above_swing_high": 1.0 if acceptance_above_high else 0.0,
            "close_last_used": float(close_last),
            "close_ref_used": float(close_ref),
            "recent_high_used": float(recent_high),
            "obv_last_used": float(self._safe(last.get("obv"), 0.0)),
            "obv_ref_used": float(self._safe(ref.get("obv"), 0.0)),
            "cvd_last_used": float(self._safe(last.get("cvd"), 0.0)),
            "cvd_ref_used": float(self._safe(ref.get("cvd"), 0.0)),
            "weakness_lookback_used": float(lb),
            "weakness_strength": float(strength),
            "missing_conditions": ",".join(missing),
            "failed_reason": "none" if passed else f"missing:{','.join(missing)}",
            "layer2_extended_state": {
                "rsi_rollover": bool(rsi_rollover),
                "hist_rollover": bool(hist_rollover),
                "lower_close_after_peak": bool(lower_close),
                "lower_high_after_peak": bool(lower_high),
                "rejection_bar": bool(rejection_bar),
                "failed_reclaim": bool(failed_reclaim),
                "retest_failed_breakout": bool(retest_failed_breakout),
                "acceptance_above_swing_high": bool(acceptance_above_high),
                "near_sweep_level": bool(sweep_ctx.get("near_sweep_level")),
            },
            "layer2_subconditions_state": {
                "price_up_or_near_high": bool(price_ctx),
                "price_up": bool(price_up),
                "near_high_context": bool(near),
                "obv_bearish_divergence": bool(obv_div),
                "cvd_bearish_divergence": bool(cvd_div),
            },
        }

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
        prev_high = self._safe(prev.get("high"), prev_close)
        tol = max(0.0, float(self.config.entry_tolerance_pct))

        below_vah = close <= vp.vah * (1.0 - max(tol * 0.12, 0.00045))
        rejected = bool(
            (prev_close >= vp.vah * (1.0 - tol * 0.45) or prev_high >= vp.vah * (1.0 - tol * 0.18))
            and close <= min(prev_close, vp.vah * (1.0 + tol * 0.35))
        )
        below_or_rej = below_vah or rejected
        near_poc = abs(close - vp.poc) <= max(vp.poc * max(0.010, tol * 2.0), 1e-8)
        inside_va = close >= vp.val * (1.0 - tol * 1.5) and close <= vp.vah * (1.0 + tol * 1.1)
        poc_ctx = near_poc or inside_va
        msb_ok, msb = self._layer3_msb_confirmation(df=df, side="SHORT")
        sweep_ctx = self._short_sweep_reclaim_context(df)
        failed_reclaim = bool(sweep_ctx.get("failed_reclaim"))
        retest_failed_breakout = bool(sweep_ctx.get("retest_failed_breakout"))
        acceptance_above_high = bool(sweep_ctx.get("acceptance_above_high"))
        rejection_bar = bool(sweep_ctx.get("rejection_bar"))
        lower_close = bool(sweep_ctx.get("lower_close"))
        lower_high = bool(sweep_ctx.get("lower_high"))
        post_peak_followthrough = bool(sweep_ctx.get("post_peak_followthrough"))
        ema20_last = self._safe(last.get("ema20"), 0.0)
        ema20_prev = self._safe(prev.get("ema20"), ema20_last)

        trigger_ctx = below_or_rej or failed_reclaim or retest_failed_breakout
        value_ctx = poc_ctx or failed_reclaim or retest_failed_breakout
        micro_break_confirmed = (
            lower_close
            or lower_high
            or rejection_bar
            or post_peak_followthrough
            or bool(msb_ok and (rejected or failed_reclaim or retest_failed_breakout))
        )
        fresh_reaction = bool(rejection_bar or lower_close or lower_high or post_peak_followthrough)
        fresh_reaction_or_reject = bool(
            fresh_reaction
            or rejected
            or failed_reclaim
            or retest_failed_breakout
        )
        continuation_above_fast_value = bool(
            ema20_last > 0
            and close >= ema20_last * (1.0 - max(tol * 0.10, 0.00035))
            and ema20_last >= ema20_prev
            and close >= prev_close * (1.0 - max(tol * 0.12, 0.00035))
            and not failed_reclaim
            and not retest_failed_breakout
            and not fresh_reaction
        )
        sweep_entry_ok = bool((failed_reclaim or retest_failed_breakout) and micro_break_confirmed and not acceptance_above_high)
        value_entry_ok = bool(
            trigger_ctx
            and value_ctx
            and bool(msb_ok)
            and micro_break_confirmed
            and fresh_reaction_or_reject
            and not acceptance_above_high
            and not continuation_above_fast_value
        )
        passed = sweep_entry_ok or value_entry_ok
        strength = (
            int(trigger_ctx or sweep_entry_ok)
            + int(value_ctx or sweep_entry_ok)
            + int(bool(msb_ok) or sweep_entry_ok)
            + int(not acceptance_above_high)
        ) / 4.0
        missing = []
        if not (trigger_ctx or sweep_entry_ok):
            missing.append("below_vah_or_rejected_from_vah")
        if not (value_ctx or sweep_entry_ok):
            missing.append("near_poc_or_value_area_context")
        if not (bool(msb_ok) or sweep_entry_ok):
            missing.append("msb_bearish_confirmed")
        if not (fresh_reaction_or_reject or sweep_entry_ok):
            missing.append("fresh_reaction_from_high")
        if acceptance_above_high:
            missing.append("acceptance_above_swing_high")
        if continuation_above_fast_value:
            missing.append("no_trade_continuation_above_fast_value")

        d = {
            "entry_location_passed": 1.0 if passed else 0.0,
            "failed_reason": "none" if passed else f"missing:{','.join(missing)}",
            "missing_conditions": ",".join(missing),
            "entry_location_strength": float(strength),
            "below_vah_or_rejected_from_vah": 1.0 if below_or_rej else 0.0,
            "near_poc_or_value_area_context": 1.0 if poc_ctx else 0.0,
            "failed_reclaim": 1.0 if failed_reclaim else 0.0,
            "retest_failed_breakout": 1.0 if retest_failed_breakout else 0.0,
            "sweep_reclaim_entry_ok": 1.0 if sweep_entry_ok else 0.0,
            "value_area_entry_ok": 1.0 if value_entry_ok else 0.0,
            "acceptance_above_swing_high": 1.0 if acceptance_above_high else 0.0,
            "msb_bearish_confirmed": 1.0 if bool(msb_ok) else 0.0,
            "fresh_reaction_from_high": 1.0 if fresh_reaction else 0.0,
            "continuation_above_fast_value": 1.0 if continuation_above_fast_value else 0.0,
            "vp_levels_available": 1.0,
            "below_vah": 1.0 if below_vah else 0.0,
        }
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
        base_inv = entry + atr * max(atr_sl * 0.92, 0.98)
        inv_ref = base_inv
        tp_ref = "rr_projection"
        fallback_rr = vp is None

        recent_highs = pd.to_numeric(df.tail(min(len(df), 24))["high"], errors="coerce").dropna()
        recent_lows = pd.to_numeric(df.tail(min(len(df), 30))["low"], errors="coerce").dropna()
        recent_high_ref = (
            max(
                float(recent_highs.quantile(0.86)),
                float(recent_highs.tail(min(len(recent_highs), 12)).quantile(0.90)),
            )
            if not recent_highs.empty
            else base_inv
        )
        recent_support = float(recent_lows.quantile(0.16)) if not recent_lows.empty else 0.0
        deep_support = float(recent_lows.quantile(0.08)) if not recent_lows.empty else recent_support

        local_recent_high_ref = (
            max(
                float(recent_highs.tail(min(len(recent_highs), 8)).max()),
                float(recent_highs.tail(min(len(recent_highs), 12)).quantile(0.84)),
            )
            if not recent_highs.empty
            else recent_high_ref
        )
        structural_inv_ref = max(local_recent_high_ref, recent_high_ref, base_inv)
        sweep_ctx = self._short_sweep_reclaim_context(df)
        sweep_high = float(sweep_ctx.get("sweep_high") or 0.0)
        if bool(sweep_ctx.get("swept_high")) and sweep_high > 0:
            structural_inv_ref = max(structural_inv_ref, sweep_high)
        vp_cap = max(
            structural_inv_ref + atr * 0.72,
            entry + atr * max(atr_sl * 0.74, 0.92),
        )
        if vp is not None:
            inv_ref = max(structural_inv_ref, min(float(vp.vah), vp_cap))
        else:
            inv_ref = structural_inv_ref

        failed_reclaim = bool(sweep_ctx.get("failed_reclaim"))
        retest_failed_breakout = bool(sweep_ctx.get("retest_failed_breakout"))
        stop_buffer = max(
            atr * (
                0.32
                if bool(sweep_ctx.get("swept_high")) or failed_reclaim or retest_failed_breakout
                else 0.18
            ),
            entry * max(tol * 0.40, 0.00055),
        )
        sl = max(inv_ref + stop_buffer, base_inv)
        provisional_tp = entry - max(atr, entry * 0.0001) * rr
        provisional_tp, sl = self._normalize_levels(entry=entry, tp=provisional_tp, sl=sl, side="SHORT")

        risk_distance = max(sl - entry, max(atr, entry * 0.0001))
        structure_depth_pct = max(0.0, (entry - deep_support) / max(entry, 1e-8)) if np.isfinite(deep_support) and deep_support > 0 else 0.0
        recent_depth_pct = max(0.0, (entry - recent_support) / max(entry, 1e-8)) if np.isfinite(recent_support) and recent_support > 0 else 0.0
        depth_bonus = min(0.92, max(structure_depth_pct, recent_depth_pct) * 6.2)
        rr_target = max(rr + 0.95, 3.05 + depth_bonus)
        rr_cap = max(rr_target * 2.15, 6.4)
        rr_projection = entry - risk_distance * rr_target
        rr_cap_target = entry - risk_distance * rr_cap

        structural_candidates: list[tuple[float, str]] = []
        if vp is not None:
            vp_poc = float(vp.poc)
            vp_val = float(vp.val)
            if np.isfinite(vp_poc) and 0 < vp_poc < entry:
                structural_candidates.append((vp_poc, "vp_poc"))
            if np.isfinite(vp_val) and 0 < vp_val < entry:
                if np.isfinite(vp_poc) and 0 < vp_poc < entry and vp_val < vp_poc:
                    structural_candidates.append((vp_poc - (vp_poc - vp_val) * 0.55, "vp_balance"))
                structural_candidates.append((vp_val, "vp_val"))
                structural_candidates.append((vp_val - max((vp_poc - vp_val) * 0.28, atr * 0.35), "vp_extension"))
        if np.isfinite(recent_support) and 0 < recent_support < entry:
            structural_candidates.append((recent_support, "recent_support"))
        if np.isfinite(deep_support) and 0 < deep_support < entry:
            structural_candidates.append((deep_support, "deep_support"))

        if structural_candidates:
            deeper_refs = {"vp_extension", "vp_val", "recent_support", "deep_support"}
            has_deeper_candidate = any(ref in deeper_refs for _, ref in structural_candidates)
            ranked_targets: list[tuple[float, float, str]] = []
            for structural_target, structural_ref in structural_candidates:
                if structural_ref == "vp_poc" and has_deeper_candidate:
                    continue
                bounded_target = max(float(structural_target), rr_cap_target)
                rr_candidate = (entry - bounded_target) / max(risk_distance, 1e-9)
                if rr_candidate < 1.45:
                    continue
                ref_bonus = {
                    "vp_extension": 1.02,
                    "deep_support": 0.90,
                    "recent_support": 0.66,
                    "vp_val": 0.56,
                    "vp_balance": 0.24,
                    "vp_poc": -0.15,
                }.get(structural_ref, 0.12)
                score = rr_candidate + ref_bonus
                ranked_targets.append((score, bounded_target, structural_ref))

            if ranked_targets:
                _, best_target, best_ref = max(ranked_targets, key=lambda item: item[0])
                tp = min(best_target, rr_projection) if best_ref == "vp_poc" else best_target
                tp_ref = best_ref if tp != rr_projection else "rr_projection"
                fallback_rr = False
            else:
                tp = rr_projection
                tp_ref = "rr_projection"
                fallback_rr = True
        else:
            tp = rr_projection
            tp_ref = "rr_projection"
            fallback_rr = True

        tp, sl = self._normalize_levels(entry=entry, tp=tp, sl=sl, side="SHORT")
        fair_value = (
            float((float(vp.vah) + float(vp.val)) / 2.0)
            if vp is not None and np.isfinite(float(vp.vah)) and np.isfinite(float(vp.val))
            else 0.0
        )
        vwap_target = self._safe(last.get("vwap"), 0.0)

        def _pick_highest_below(candidates: list[tuple[float, str]], ceiling: float) -> tuple[float, str] | None:
            valid = [(float(value), str(ref)) for value, ref in candidates if np.isfinite(value) and 0 < float(value) < ceiling - max(entry * 0.0001, 1e-8)]
            return max(valid, key=lambda item: item[0]) if valid else None

        def _pick_lowest_below(candidates: list[tuple[float, str]], ceiling: float) -> tuple[float, str] | None:
            valid = [(float(value), str(ref)) for value, ref in candidates if np.isfinite(value) and 0 < float(value) < ceiling - max(entry * 0.0001, 1e-8)]
            return min(valid, key=lambda item: item[0]) if valid else None

        tp1_pick = _pick_highest_below(
            [
                (vwap_target, "session_vwap"),
                (fair_value, "fair_value"),
                (float(vp.vah), "vp_vah") if vp is not None else (0.0, ""),
                (entry - risk_distance * 0.85, "rr_tp1"),
            ],
            entry,
        )
        tp1 = tp1_pick[0] if tp1_pick else 0.0
        tp1_ref = tp1_pick[1] if tp1_pick else ""

        tp2_pick = _pick_highest_below(
            [
                (float(vp.poc), "vp_poc") if vp is not None else (0.0, ""),
                ((float(vp.vah) + float(vp.val)) / 2.0, "vp_mid") if vp is not None else (0.0, ""),
                (fair_value, "fair_value"),
                (recent_support, "recent_support"),
                (rr_projection, "rr_projection"),
            ],
            tp1 if tp1 > 0 else entry,
        )
        tp2 = tp2_pick[0] if tp2_pick else 0.0
        tp2_ref = tp2_pick[1] if tp2_pick else ""

        tp3_pick = _pick_lowest_below(
            [
                (float(vp.val), "vp_val") if vp is not None else (0.0, ""),
                (deep_support, "deep_support"),
                (entry - risk_distance * max(rr + 0.55, 2.4), "rr_tp3"),
                (tp, str(tp_ref)),
            ],
            tp2 if tp2 > 0 else (tp1 if tp1 > 0 else entry),
        )
        tp3 = tp3_pick[0] if tp3_pick else 0.0
        tp3_ref = tp3_pick[1] if tp3_pick else ""

        tp1_rr = ((entry - tp1) / max(risk_distance, 1e-9)) if tp1 > 0 else 0.0
        tp2_rr = ((entry - tp2) / max(risk_distance, 1e-9)) if tp2 > 0 else 0.0
        tp3_rr = ((entry - tp3) / max(risk_distance, 1e-9)) if tp3 > 0 else 0.0
        base_tp_rr = ((entry - tp) / max(risk_distance, 1e-9)) if tp > 0 else 0.0
        strong_unwind_context = bool(
            failed_reclaim
            or retest_failed_breakout
            or bool(sweep_ctx.get("swept_high"))
            or structure_depth_pct >= 0.035
            or recent_depth_pct >= 0.028
        )

        final_tp = tp
        final_tp_ref = str(tp_ref)
        if strong_unwind_context:
            extended_candidates: list[tuple[float, str]] = []
            if tp3 > 0 and tp3_rr >= max(rr + 0.55, 2.20):
                extended_candidates.append((tp3, str(tp3_ref or "tp3")))
            if tp > 0 and base_tp_rr >= max(rr + 0.35, 2.05):
                extended_candidates.append((tp, str(tp_ref)))
            if extended_candidates:
                final_tp, final_tp_ref = min(extended_candidates, key=lambda item: item[0])
            elif tp2 > 0:
                final_tp = tp2
                final_tp_ref = str(tp2_ref or tp_ref)
            elif tp1 > 0:
                final_tp = tp1
                final_tp_ref = str(tp1_ref or tp_ref)
        else:
            if tp2 > 0:
                final_tp = tp2
                final_tp_ref = str(tp2_ref or tp_ref)
            elif tp1 > 0:
                final_tp = tp1
                final_tp_ref = str(tp1_ref or tp_ref)
        final_tp, sl = self._normalize_levels(entry=entry, tp=final_tp, sl=sl, side="SHORT")
        stop_pct = (sl - entry) / max(entry, 1e-9)
        tp_pct = (entry - final_tp) / max(entry, 1e-9)
        rr_val = (tp_pct / max(stop_pct, 1e-9)) if stop_pct > 0 else 0.0
        stop_ok = sl >= inv_ref and sl > entry
        tp_ok = final_tp <= float(vp.poc) * (1.0 + tol) if (final_tp_ref in {"vp_poc", "vp_balance"} and vp is not None) else final_tp < entry
        rr_ok = rr_val >= rr_floor
        passed = stop_ok and tp_ok and rr_ok

        missing = []
        if not stop_ok: missing.append("stop_above_invalidation")
        if not tp_ok: missing.append("tp_at_poc_or_better")
        if not rr_ok: missing.append("risk_reward_ratio_soft_min")
        strength = (int(stop_ok) + int(tp_ok) + int(rr_ok)) / 3.0
        partial_candidates: list[float] = []
        if vp is not None and np.isfinite(float(vp.poc)) and 0 < float(vp.poc) < entry:
            partial_candidates.append(float(vp.poc))
        if np.isfinite(recent_support) and 0 < recent_support < entry:
            partial_candidates.append(float(recent_support))
        partial_candidates.extend([float(x) for x in (tp1, tp2, tp3) if x > 0.0])
        partial_candidates.append(float((entry + final_tp) / 2.0))
        partial_candidates.append(float(final_tp))
        partial_tps = sorted({round(float(x), 10) for x in partial_candidates if 0 < float(x) < entry}, reverse=True)
        details = {
            "passed": 1.0 if passed else 0.0,
            "entry": float(entry),
            "tp": float(final_tp),
            "sl": float(sl),
            "tp1": float(tp1) if tp1 > 0 else 0.0,
            "tp2": float(tp2) if tp2 > 0 else 0.0,
            "tp3": float(tp3) if tp3 > 0 else 0.0,
            "tp1_reference": str(tp1_ref),
            "tp2_reference": str(tp2_ref),
            "tp3_reference": str(tp3_ref),
            "partial_tps": partial_tps,
            "tp_selection_mode": "extended" if strong_unwind_context and final_tp_ref in {str(tp3_ref), str(tp_ref)} else "protective",
            "stop_above_invalidation": 1.0 if stop_ok else 0.0,
            "tp_at_poc_or_better": 1.0 if tp_ok else 0.0,
            "atr_available": 1.0 if atr_available else 0.0,
            "volume_profile_available": 1.0 if vp is not None else 0.0,
            "fallback_rr_used": 1.0 if fallback_rr else 0.0,
            "invalidation_reference": float(inv_ref),
            "tp_reference": str(final_tp_ref),
            "sweep_high_invalidation": 1.0 if bool(sweep_ctx.get("swept_high")) else 0.0,
            "acceptance_above_high": 1.0 if bool(sweep_ctx.get("acceptance_above_high")) else 0.0,
            "stop_distance_pct": float(stop_pct),
            "take_profit_distance_pct": float(tp_pct),
            "risk_reward_ratio": float(rr_val),
            "missing_conditions": ",".join(missing),
            "failed_reason": "none" if passed else f"missing:{','.join(missing)}",
            "tp_sl_strength": float(strength),
        }
        return passed, details

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









