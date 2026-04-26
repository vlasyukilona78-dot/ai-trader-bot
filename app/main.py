from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_RUNTIME_SITE_PACKAGES = PROJECT_ROOT / ".runtime_env" / "Lib" / "site-packages"
if PROJECT_RUNTIME_SITE_PACKAGES.exists():
    runtime_site_packages = str(PROJECT_RUNTIME_SITE_PACKAGES)
    if runtime_site_packages not in sys.path:
        sys.path.insert(0, runtime_site_packages)

import numpy as np
import pandas as pd

from app.bootstrap import ConfigError, RuntimeConfig, load_runtime_config
from alerts.chart_generator import build_signal_chart
from core.market_regime import detect_market_regime
from core.liquidation_map import build_liquidation_map
from core.volume_profile import compute_volume_profile
from trading.alerts.discord import DiscordAlerter
from trading.alerts.signal_card_clean import (
    _normalize_human_text,
    build_early_invalidation_text,
    build_early_signal_caption,
    build_signal_caption,
    build_symbol_copy_reply_markup,
)
from trading.alerts.telegram import TelegramAlerter
from trading.execution.engine import ExecutionEngine, ExecutionOutcome
from trading.exchange.bybit_adapter import BybitAdapter
from trading.exchange.bybit_endpoints import resolve_public_http_base_url
from trading.metrics.counters import MetricsCounter
from trading.metrics.logging import setup_logging
from trading.risk.engine import RiskEngine
from trading.signals.base import HoldStrategy
from trading.signals.layered_strategy import LayeredPumpStrategy
from trading.signals.runtime_source_adapter import build_runtime_signal_inputs
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.strategy_interface import StrategyContext
from trading.state.machine import StateMachine
from trading.state.models import StateRecord, TradeState
from trading.state.persistence import RuntimeStore

if TYPE_CHECKING:
    from trading.features.pipeline import FeaturePipeline
    from trading.market_data.feed import MarketDataFeed
    from trading.market_data.reconciliation import ExchangeReconciler
    from trading.market_data.ws_reconciliation import ExchangeSyncService


def _load_dotenv_file(path: str = ".env") -> None:
    candidates = [Path(path), Path(__file__).resolve().parents[1] / path]
    seen: set[Path] = set()
    for env_path in candidates:
        try:
            resolved = env_path.resolve()
        except Exception:
            resolved = env_path
        if resolved in seen or not env_path.exists():
            continue
        seen.add(resolved)
        try:
            for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                os.environ.setdefault(key, value.strip().strip('"').strip("'"))
        except Exception:
            continue


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _latest_peak_age_bars(
    series: pd.Series,
    *,
    reference_price: float,
    atr: float,
    relative_tolerance: float,
) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0

    values = numeric.to_numpy(dtype=float)
    peak_value = float(np.nanmax(values))
    tolerance = max(abs(reference_price) * relative_tolerance, atr * 0.34, 1e-8)
    peak_indices = np.where(values >= peak_value - tolerance)[0]
    if len(peak_indices) == 0:
        return 0
    return max(len(values) - 1 - int(peak_indices[-1]), 0)


def _phase_rank(phase: str) -> int:
    normalized = str(phase or "").strip().upper()
    return {"WATCH": 1, "SETUP": 2}.get(normalized, 0)


def _is_observation_only_profile(signal_profile: str) -> bool:
    return str(signal_profile or "").strip().lower() == "early"


def _observation_state_record(symbol: str) -> StateRecord:
    clean_symbol = str(symbol or "").replace("/", "").upper().strip()
    return StateRecord(
        symbol=clean_symbol,
        state=TradeState.FLAT,
        reason="observation_profile",
        updated_at=time.time(),
    )


def _decision_priority(intent, early_candidate: Mapping[str, object] | None) -> tuple[int, int]:
    action = getattr(intent, "action", None)
    if action in (
        IntentAction.LONG_ENTRY,
        IntentAction.SHORT_ENTRY,
        IntentAction.EXIT_LONG,
        IntentAction.EXIT_SHORT,
    ):
        return (0, 0)

    phase = str((early_candidate or {}).get("phase") or "").upper()
    if phase == "SETUP":
        return (1, 0)
    if phase == "WATCH":
        return (1, 1)
    return (2, 0)


def _early_config_float(name: str, default: float) -> float:
    return _as_float(os.getenv(name), default)


def _early_quality_grade(score: float) -> str:
    if score >= 8.5:
        return "A+"
    if score >= 7.5:
        return "A"
    if score >= 6.5:
        return "B"
    if score >= 5.5:
        return "C"
    return "D"


def _extract_layer_details(meta: Mapping[str, object] | None, layer_name: str) -> dict[str, object]:
    trace = meta.get("layer_trace", {}) if isinstance(meta, Mapping) else {}
    layers = trace.get("layers", {}) if isinstance(trace, Mapping) else {}
    layer = layers.get(layer_name, {}) if isinstance(layers, Mapping) else {}
    details = layer.get("details", {}) if isinstance(layer, Mapping) else {}
    return details if isinstance(details, Mapping) else {}


def _build_strategy(name: str):
    if name == "hold":
        return HoldStrategy()
    return LayeredPumpStrategy()



def _build_alerters(cfg: RuntimeConfig):
    out = []
    if cfg.alerts.telegram_token and cfg.alerts.telegram_chat_id:
        out.append(TelegramAlerter(token=cfg.alerts.telegram_token, chat_id=cfg.alerts.telegram_chat_id))
    if cfg.alerts.discord_webhook_url:
        out.append(DiscordAlerter(webhook_url=cfg.alerts.discord_webhook_url))
    return out


def _send_alerts(alerters, text: str, reply_markup: dict | None = None):
    attempted = 0
    sent = 0
    for alerter in alerters:
        attempted += 1
        try:
            if alerter.send(text, reply_markup=reply_markup):
                sent += 1
        except TypeError:
            try:
                if alerter.send(text):
                    sent += 1
            except Exception:
                continue
        except Exception:
            continue
    return attempted, sent


def _send_photo_alerts(
    alerters,
    caption: str,
    image_bytes: bytes,
    filename: str = "signal.png",
    reply_markup: dict | None = None,
):
    attempted = 0
    sent = 0
    for alerter in alerters:
        attempted += 1
        send_photo = getattr(alerter, "send_photo", None)
        send_text = getattr(alerter, "send", None)
        delivered = False
        try:
            if callable(send_photo):
                delivered = bool(send_photo(
                    caption=caption,
                    image_bytes=image_bytes,
                    filename=filename,
                    reply_markup=reply_markup,
                ))
                if delivered:
                    sent += 1
            elif callable(send_text) and send_text(caption, reply_markup=reply_markup):
                sent += 1
                delivered = True
        except TypeError:
            try:
                if callable(send_photo):
                    delivered = bool(send_photo(caption=caption, image_bytes=image_bytes, filename=filename))
                    if delivered:
                        sent += 1
                elif callable(send_text) and send_text(caption):
                    sent += 1
                    delivered = True
            except Exception:
                delivered = False
        except Exception:
            delivered = False
        if delivered or not callable(send_text):
            continue
        try:
            if send_text(caption, reply_markup=reply_markup):
                sent += 1
        except TypeError:
            try:
                if send_text(caption):
                    sent += 1
            except Exception:
                continue
        except Exception:
            continue
    return attempted, sent


def _log_alert_delivery(logger, *, event: str, attempted: int, sent: int, skip_reason: str = ""):
    logger.info("%s attempted=%d sent=%d", event, attempted, sent, extra={"event": event})
    if skip_reason:
        logger.warning("%s reason=%s", f"{event}_skipped", skip_reason, extra={"event": f"{event}_skipped"})



def _should_emit_cached_alert(
    cache: dict[str, object],
    *,
    symbol: str,
    key: str,
    now_ts: float,
    cooldown_sec: int,
) -> bool:
    cached = cache.get(symbol)
    if isinstance(cached, Mapping):
        cached_key = str(cached.get("key") or "")
        next_allowed_ts = _as_float(cached.get("next_allowed_ts"), 0.0)
    else:
        cached_key = str(cached or "")
        next_allowed_ts = 0.0
    return cached_key != key or now_ts >= next_allowed_ts


def _remember_cached_alert(
    cache: dict[str, object],
    *,
    symbol: str,
    key: str,
    now_ts: float,
    cooldown_sec: int,
) -> None:
    cached = cache.get(symbol)
    payload = dict(cached) if isinstance(cached, Mapping) else {}
    payload.update(
        {
            "key": key,
            "next_allowed_ts": now_ts + max(60, cooldown_sec),
        }
    )
    cache[symbol] = payload


def _should_emit_symbol_rate_limited_alert(
    cache: dict[str, object],
    *,
    symbol: str,
    now_ts: float,
) -> bool:
    cached = cache.get(symbol)
    if isinstance(cached, Mapping):
        next_allowed_ts = _as_float(cached.get("symbol_next_allowed_ts"), 0.0)
    else:
        next_allowed_ts = 0.0
    return now_ts >= next_allowed_ts


def _remember_symbol_rate_limited_alert(
    cache: dict[str, object],
    *,
    symbol: str,
    now_ts: float,
    cooldown_sec: int,
) -> None:
    cached = cache.get(symbol)
    payload = dict(cached) if isinstance(cached, Mapping) else {}
    payload["symbol_next_allowed_ts"] = now_ts + max(60, cooldown_sec)
    cache[symbol] = payload


def _format_alert_price_bucket(value: float) -> str:
    price = _as_float(value, 0.0)
    if price <= 0:
        return "0"
    if price < 0.01:
        return f"{price:.8f}"
    if price < 1.0:
        return f"{price:.6f}"
    return f"{price:.4f}"


def _build_main_signal_signature(
    *,
    symbol: str,
    intent,
    enriched,
    mark_price: float,
) -> str:
    metadata = intent.metadata if isinstance(getattr(intent, "metadata", None), Mapping) else {}
    legacy_signal_id = str(metadata.get("legacy_signal_id") or "").strip()
    if legacy_signal_id:
        return f"legacy|{legacy_signal_id}"

    bar_ts = ""
    if enriched is not None and not getattr(enriched, "empty", True):
        try:
            bar_ts = str(enriched.index[-1])
        except Exception:
            bar_ts = ""

    return "|".join(
        [
            str(symbol).replace("/", "").upper(),
            str(getattr(intent, "action", "") or ""),
            str(getattr(intent, "reason", "") or ""),
            bar_ts,
            _format_alert_price_bucket(mark_price),
            _format_alert_price_bucket(getattr(intent, "take_profit", None)),
            _format_alert_price_bucket(getattr(intent, "stop_loss", None)),
        ]
    )


def _format_signature_price(value: object) -> str:
    numeric = _as_float(value, 0.0)
    if numeric <= 0:
        return "0"
    if numeric < 0.01:
        precision = 6
    elif numeric < 1:
        precision = 5
    elif numeric < 100:
        precision = 4
    else:
        precision = 2
    return f"{numeric:.{precision}f}"


def _build_early_candidate_signature(candidate: Mapping[str, object], enriched) -> str:
    phase = str(candidate.get("phase") or "").strip().upper()
    last_bar_ts = ""
    try:
        if enriched is not None and len(enriched.index) > 0:
            last_bar = pd.Timestamp(enriched.index[-1])
            if last_bar.tzinfo is not None:
                last_bar = last_bar.tz_convert("UTC").tz_localize(None)
            last_bar_ts = last_bar.isoformat()
    except Exception:
        last_bar_ts = ""
    return "|".join(
        [
            phase,
            _format_signature_price(candidate.get("entry")),
            _format_signature_price(candidate.get("tp")),
            _format_signature_price(candidate.get("sl")),
            last_bar_ts,
        ]
    )


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _summarize_startup_state(
    startup_state: Mapping[str, object],
    *,
    sample_size: int = 12,
) -> dict[str, object]:
    state_counts: Counter[str] = Counter()
    issue_samples: list[str] = []
    total_issues = 0
    for symbol, raw_state in startup_state.items():
        state = str(raw_state or "")
        base_state = state
        if "|issues=" in state:
            base_state, issues = state.split("|issues=", 1)
            total_issues += 1
            if len(issue_samples) < sample_size:
                issue_samples.append(f"{symbol}:{issues}")
        state_counts[base_state or "UNKNOWN"] += 1
    return {
        "state_counts": dict(state_counts),
        "issue_count": total_issues,
        "issue_samples": issue_samples,
    }


def _collect_runtime_payload(frame) -> dict[str, object]:
    payload: dict[str, object] = {}
    frame_payload = getattr(frame, "runtime_payload", None)
    if isinstance(frame_payload, Mapping):
        payload.update(frame_payload)

    ohlcv_attrs = getattr(frame.ohlcv, "attrs", None)
    if isinstance(ohlcv_attrs, Mapping):
        for key in ("runtime_payload", "signal_sources", "source_payload"):
            value = ohlcv_attrs.get(key)
            if isinstance(value, Mapping):
                payload.update(value)

    return payload


def _build_alert_chart(
    symbol: str,
    timeframe: str,
    enriched,
    *,
    side: str,
    entry: float,
    tp: float,
    sl: float,
    show_trade_levels: bool = True,
    show_entry_levels: bool = True,
    show_liquidation_map: bool = True,
    timeframe_label: str | None = None,
    liquidation_cluster_high: float | None = None,
    liquidation_cluster_low: float | None = None,
) -> bytes | None:
    try:
        volume_profile = compute_volume_profile(enriched)
        liquidation_map = build_liquidation_map(
            enriched,
            liquidation_cluster_high=liquidation_cluster_high,
            liquidation_cluster_low=liquidation_cluster_low,
        )
        return build_signal_chart(
            symbol=symbol,
            df=enriched,
            side=side,
            entry=entry,
            tp=tp,
            sl=sl,
            volume_profile=volume_profile,
            timeframe_label=timeframe_label or f"{timeframe}m",
            show_trade_levels=show_trade_levels,
            show_entry_levels=show_entry_levels,
            liquidation_map=liquidation_map,
            show_liquidation_map=show_liquidation_map,
        )
    except Exception:
        return None


def _format_chart_timeframe_label(timeframe: str) -> str:
    try:
        minutes = int(str(timeframe).strip())
    except (TypeError, ValueError):
        return str(timeframe)
    if minutes % 60 == 0 and minutes >= 60:
        hours = minutes // 60
        return f"{hours}h"
    return f"{minutes}m"


def _timeframe_to_minutes(timeframe: str) -> int:
    try:
        return max(1, int(str(timeframe).strip()))
    except (TypeError, ValueError):
        return 1


def _needs_pre_alert_refresh(
    *,
    prepared_ts: float,
    timeframe: str,
    max_age_sec: int | None = None,
) -> bool:
    if prepared_ts <= 0:
        return True
    timeframe_minutes = _timeframe_to_minutes(timeframe)
    if max_age_sec is None:
        if timeframe_minutes <= 1:
            max_age_sec = 5
        elif timeframe_minutes <= 5:
            max_age_sec = 10
        else:
            max_age_sec = min(30, timeframe_minutes * 4)
    return (time.time() - prepared_ts) >= max(2, int(max_age_sec))


def _analysis_worker_count(candidate_count: int) -> int:
    configured = _as_int(os.getenv("CONCURRENT_TASKS", "8"), 8)
    return max(1, min(max(1, configured), max(1, candidate_count)))


def _clone_pipeline(pipeline: "FeaturePipeline") -> "FeaturePipeline":
    try:
        return pipeline.__class__(
            profile_window=int(getattr(pipeline, "profile_window", 120)),
            profile_bins=int(getattr(pipeline, "profile_bins", 48)),
        )
    except Exception:
        return pipeline


def _prepare_symbol_analysis(
    *,
    symbol: str,
    snapshot,
    rec_state,
    feed: "MarketDataFeed",
    pipeline: "FeaturePipeline",
    timeframe: str,
    candles_limit: int,
    overlay_live_price: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {
        "symbol": symbol,
        "snapshot": snapshot,
        "rec_state": rec_state,
    }
    try:
        frame = feed.fetch_frame(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles_limit,
            overlay_live_price=overlay_live_price,
        )
        if frame.ohlcv.empty:
            result["status"] = "empty_ohlcv"
            return result

        as_of = frame.ohlcv.index[-1]
        runtime_payload = _collect_runtime_payload(frame)
        runtime_inputs = build_runtime_signal_inputs(frame.ohlcv, runtime_payload=runtime_payload)
        extras = {
            "sentiment_index": runtime_inputs.get("sentiment_index"),
            "funding_rate": runtime_inputs.get("funding_rate"),
            "long_short_ratio": runtime_inputs.get("long_short_ratio"),
            "open_interest": runtime_inputs.get("open_interest"),
            "news_veto": runtime_inputs.get("news_veto"),
            "news_source": runtime_inputs.get("news_source"),
            "liquidation_cluster_high": frame.liquidation_cluster_high,
            "liquidation_cluster_low": frame.liquidation_cluster_low,
        }
        features = _clone_pipeline(pipeline).build(symbol=symbol, ohlcv=frame.ohlcv, as_of=as_of, extras=extras)
        mark_price = frame.mark_price if frame.mark_price > 0 else float(features.enriched.iloc[-1]["close"])

        result.update(
            {
                "status": "ok",
                "prepared_at": time.time(),
                "frame": frame,
                "runtime_inputs": runtime_inputs,
                "extras": extras,
                "features": features,
                "mark_price": mark_price,
            }
        )
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _clone_strategy_for_parallel(strategy):
    clone = getattr(strategy, "clone_for_parallel", None)
    if callable(clone):
        return clone()
    return None


def _record_parallel_strategy_intent(strategy, intent) -> None:
    recorder = getattr(strategy, "record_external_intent", None)
    if callable(recorder):
        try:
            recorder(intent)
        except Exception:
            return


def _prepare_symbol_decision(
    *,
    symbol: str,
    snapshot,
    rec_state,
    prepared: dict[str, object],
    timeframe: str,
    strategy,
) -> dict[str, object]:
    result: dict[str, object] = {
        "symbol": symbol,
        "snapshot": snapshot,
        "rec_state": rec_state,
    }
    try:
        frame = prepared["frame"]
        runtime_inputs = prepared["runtime_inputs"]
        extras = prepared["extras"]
        features = prepared["features"]
        mark_price = _as_float(prepared.get("mark_price"), float(features.enriched.iloc[-1]["close"]))

        local_strategy = _clone_strategy_for_parallel(strategy)
        if local_strategy is None:
            result["status"] = "error"
            result["error"] = "strategy_not_parallel_safe"
            return result

        intent = local_strategy.generate(
            StrategyContext(
                symbol=symbol,
                market_ohlcv=features.enriched,
                mark_price=mark_price,
                exchange=snapshot,
                synced_state=rec_state.state,
                timeframe=timeframe,
                synced_state_updated_at=_as_float(getattr(rec_state, "updated_at", 0.0), 0.0),
                sentiment_index=runtime_inputs.get("sentiment_index"),
                sentiment_value=runtime_inputs.get("sentiment_value"),
                sentiment_source=runtime_inputs.get("sentiment_source"),
                sentiment_degraded=runtime_inputs.get("sentiment_degraded"),
                funding_rate=runtime_inputs.get("funding_rate"),
                funding_source=runtime_inputs.get("funding_source"),
                funding_degraded=runtime_inputs.get("funding_degraded"),
                long_short_ratio=runtime_inputs.get("long_short_ratio"),
                long_short_ratio_source=runtime_inputs.get("long_short_ratio_source"),
                long_short_ratio_degraded=runtime_inputs.get("long_short_ratio_degraded"),
                open_interest=runtime_inputs.get("open_interest"),
                open_interest_ratio=runtime_inputs.get("open_interest_ratio"),
                oi_signal=runtime_inputs.get("oi_signal"),
                oi_source=runtime_inputs.get("oi_source"),
                oi_degraded=runtime_inputs.get("oi_degraded"),
                open_interest_source=runtime_inputs.get("open_interest_source"),
                news_veto=runtime_inputs.get("news_veto"),
                news_source=runtime_inputs.get("news_source"),
                news_degraded=runtime_inputs.get("news_degraded"),
            )
        )

        result.update(
            {
                "status": "ok",
                "prepared_at": _as_float(prepared.get("prepared_at"), time.time()),
                "frame": frame,
                "runtime_inputs": runtime_inputs,
                "extras": extras,
                "features": features,
                "mark_price": mark_price,
                "intent": intent,
            }
        )
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _should_refresh_live_candidate(intent, early_candidate: Mapping[str, object] | None) -> bool:
    action = getattr(intent, "action", None)
    if action in (IntentAction.LONG_ENTRY, IntentAction.SHORT_ENTRY):
        return True
    return bool(early_candidate)


def _should_block_clean_pump(intent) -> bool:
    if getattr(intent, "action", None) not in (IntentAction.LONG_ENTRY, IntentAction.SHORT_ENTRY):
        return False
    meta = intent.metadata if isinstance(intent.metadata, Mapping) else {}
    layer1 = _extract_layer_details(meta, "layer1_pump_detection")
    reported_clean_pump_pct = _as_float(layer1.get("clean_pump_pct"), 0.0)
    clean_pump_pct = reported_clean_pump_pct
    clean_pump_min = _as_float(layer1.get("clean_pump_min_pct_used"), 0.0)
    if clean_pump_min <= 0 or clean_pump_pct <= 0:
        return False
    return clean_pump_pct + 1e-8 < clean_pump_min


def _current_recent_peak_price(enriched: pd.DataFrame, *, fallback_price: float) -> float:
    if enriched is None or getattr(enriched, "empty", True):
        return max(fallback_price, 0.0)
    try:
        highs = pd.to_numeric(enriched.tail(28)["high"], errors="coerce").dropna()
    except Exception:
        highs = pd.Series(dtype=float)
    recent_high = _as_float(highs.max() if not highs.empty else fallback_price, fallback_price)
    return max(recent_high, fallback_price, 0.0)


def _should_block_recent_main_short_reentry(
    cache: Mapping[str, object] | None,
    *,
    symbol: str,
    now_ts: float,
    entry_price: float,
    recent_peak_price: float,
    atr: float,
    cooldown_sec: int,
) -> bool:
    if cooldown_sec <= 0:
        return False
    cached = cache.get(symbol) if isinstance(cache, Mapping) else None
    if not isinstance(cached, Mapping):
        return False

    last_entry_ts = _as_float(cached.get("main_short_entry_ts"), 0.0)
    last_entry_price = _as_float(cached.get("main_short_entry_price"), 0.0)
    last_peak_price = _as_float(cached.get("main_short_peak_price"), 0.0)
    last_atr = _as_float(cached.get("main_short_atr"), 0.0)
    if last_entry_ts <= 0 or now_ts >= last_entry_ts + max(300, cooldown_sec):
        return False
    if last_entry_price <= 0 or last_peak_price <= 0:
        return False

    atr_ref = max(atr, last_atr, last_entry_price * 0.0022, 1e-8)
    same_entry_zone_abs = max(atr_ref * 0.95, last_entry_price * (0.0068 if last_entry_price < 1.0 else 0.0042))
    meaningful_new_peak_abs = max(atr_ref * 1.45, last_peak_price * (0.0105 if last_peak_price < 1.0 else 0.0068))

    entry_still_same_zone = abs(entry_price - last_entry_price) <= same_entry_zone_abs
    no_meaningful_new_peak = recent_peak_price <= last_peak_price + meaningful_new_peak_abs
    return bool(entry_still_same_zone and no_meaningful_new_peak)


def _remember_recent_main_short_reentry(
    cache: dict[str, object],
    *,
    symbol: str,
    now_ts: float,
    entry_price: float,
    recent_peak_price: float,
    atr: float,
) -> None:
    cached = cache.get(symbol)
    payload = dict(cached) if isinstance(cached, Mapping) else {}
    payload.update(
        {
            "main_short_entry_ts": now_ts,
            "main_short_entry_price": entry_price,
            "main_short_peak_price": recent_peak_price,
            "main_short_atr": atr,
        }
    )
    cache[symbol] = payload


def _should_block_live_main_short_continuation(intent, enriched: pd.DataFrame | None) -> bool:
    if getattr(intent, "action", None) != IntentAction.SHORT_ENTRY:
        return False
    if enriched is None or getattr(enriched, "empty", True) or len(enriched) < 4:
        return False

    meta = intent.metadata if isinstance(getattr(intent, "metadata", None), Mapping) else {}
    layer2 = _extract_layer_details(meta, "layer2_weakness_confirmation")
    layer3 = _extract_layer_details(meta, "layer3_entry_location")
    failed_reclaim = bool(
        _as_float(layer2.get("failed_reclaim"), 0.0) or _as_float(layer3.get("failed_reclaim"), 0.0)
    )
    retest_failed_breakout = bool(
        _as_float(layer2.get("retest_failed_breakout"), 0.0)
        or _as_float(layer3.get("retest_failed_breakout"), 0.0)
    )
    recent_upper_anchor_touch = bool(_as_float(layer3.get("recent_upper_anchor_touch"), 0.0))
    recent_upper_anchor_reject = bool(_as_float(layer3.get("recent_upper_anchor_reject"), 0.0))
    reject_from_upper_anchor = bool(_as_float(layer3.get("reject_from_upper_anchor"), 0.0))
    acceptance_above_high = bool(
        _as_float(layer2.get("acceptance_above_swing_high"), 0.0)
        or _as_float(layer3.get("acceptance_above_swing_high"), 0.0)
    )
    anchored_failure = bool(
        (failed_reclaim or retest_failed_breakout)
        and (recent_upper_anchor_touch or recent_upper_anchor_reject or reject_from_upper_anchor)
    )
    if anchored_failure or acceptance_above_high:
        return False

    row = enriched.iloc[-1]
    prev = enriched.iloc[-2]
    close = _as_float(row.get("close"), 0.0)
    prev_close = _as_float(prev.get("close"), close)
    if close <= 0:
        return False
    high = _as_float(row.get("high"), close)
    low = _as_float(row.get("low"), close)
    prev_high = _as_float(prev.get("high"), high)
    open_px = _as_float(row.get("open"), close)
    candle_range = max(high - low, 1e-8)
    close_position_in_candle = max(0.0, min(1.0, (close - low) / candle_range))
    recent_peak = _current_recent_peak_price(enriched, fallback_price=high)
    recent_high_24_series = pd.to_numeric(
        enriched.tail(min(len(enriched), 24))["high"],
        errors="coerce",
    ).dropna()
    recent_high_24 = _as_float(recent_high_24_series.max() if not recent_high_24_series.empty else high, high)
    recent_close_tail = pd.to_numeric(enriched.tail(4)["close"], errors="coerce").dropna()
    recent_low_tail = pd.to_numeric(enriched.tail(5)["low"], errors="coerce").dropna()
    recent_support_reference = (
        float(recent_low_tail.tail(5).iloc[:-1].min())
        if len(recent_low_tail) >= 2
        else float(recent_low_tail.min()) if not recent_low_tail.empty else low
    )
    up_closes_last3 = 0
    if len(recent_close_tail) >= 2:
        recent_close_values = recent_close_tail.to_numpy(dtype=float)
        up_closes_last3 = int(
            sum(curr > prev_close_value for prev_close_value, curr in zip(recent_close_values[:-1], recent_close_values[1:]))
        )

    rsi = _as_float(row.get("rsi"), 50.0)
    prev_rsi = _as_float(prev.get("rsi"), rsi)
    hist = _as_float(row.get("hist"), 0.0)
    prev_hist = _as_float(prev.get("hist"), hist)
    volume_spike = _as_float(row.get("volume_spike"), 0.0)
    prev_volume_spike = _as_float(prev.get("volume_spike"), volume_spike)
    ema20 = _as_float(row.get("ema20"), close)
    ema20_prev = _as_float(prev.get("ema20"), ema20)
    vah = _as_float(row.get("vah"), 0.0)
    mtf_trend_5m = _as_float(row.get("mtf_trend_5m"), 0.0)
    mtf_trend_15m = _as_float(row.get("mtf_trend_15m"), 0.0)
    mtf_rsi_5m = _as_float(row.get("mtf_rsi_5m"), 50.0)
    mtf_rsi_15m = _as_float(row.get("mtf_rsi_15m"), 50.0)
    atr = max(_as_float(row.get("atr"), close * 0.01), close * 0.0015, 1e-8)
    atr_pct = atr / max(close, 1e-8)
    bb_upper = _as_float(row.get("bb_upper"), 0.0)
    kc_upper = _as_float(row.get("kc_upper"), 0.0)
    upper_band_ref = max(bb_upper, kc_upper, 0.0)
    above_fast_upper_band = bool(
        upper_band_ref > 0
        and close >= upper_band_ref * (1.0 - max(atr_pct * 0.18, 0.00045))
    )
    near_peak = close >= recent_peak * (0.994 if close < 0.05 else 0.996)
    near_vah = vah > 0 and abs(close - vah) / max(close, 1e-8) <= max(atr_pct * 0.40, 0.0022)
    meaningfully_below_vah = bool(
        vah > 0
        and close <= vah * (1.0 - max(atr_pct * 0.32, 0.0011 if close < 0.05 else 0.00085))
    )
    far_from_recent_peak = bool(
        recent_peak > 0
        and (recent_peak - close) / max(recent_peak, 1e-8) >= max(atr_pct * 0.78, 0.0085 if close < 0.05 else 0.0052)
    )
    recent_high_tail = pd.to_numeric(enriched.tail(4)["high"], errors="coerce").dropna()
    upper_anchor_touch_tolerance = min(
        max(
            atr_pct * 0.72,
            0.0052 if close < 0.02 else 0.0032,
        ),
        0.0105 if close < 0.02 else 0.0062,
    )
    upper_anchor_levels: list[float] = []
    if vah > 0:
        upper_anchor_levels.append(vah)
    if recent_high_24 > 0 and (vah <= 0 or recent_high_24 >= vah * (1.0 - max(atr_pct * 0.45, 0.0014))):
        upper_anchor_levels.append(recent_high_24)
    live_recent_upper_anchor_touch = bool(
        upper_anchor_levels
        and len(recent_high_tail) > 0
        and any(
            abs(high_px - anchor_level) / max(anchor_level, 1e-8) <= upper_anchor_touch_tolerance
            for high_px in recent_high_tail.tail(3).to_numpy(dtype=float)
            for anchor_level in upper_anchor_levels
        )
    )
    strong_upper_reversal_bar = bool(
        (high - max(open_px, close)) / candle_range >= 0.24
        and (close <= open_px or close < prev_close)
    )
    momentum_rollover = bool(rsi < prev_rsi or hist < prev_hist)
    volume_fade = bool(
        volume_spike <= max(prev_volume_spike * 0.93, prev_volume_spike - 0.18)
        or volume_spike <= max(0.95, prev_volume_spike * 0.86)
    )
    recent_upper_anchor_touch = bool(recent_upper_anchor_touch or live_recent_upper_anchor_touch)
    recent_upper_anchor_reject = bool(
        recent_upper_anchor_reject
        or (
            recent_upper_anchor_touch
            and (
                strong_upper_reversal_bar
                or close < prev_close
                or high < prev_high
                or momentum_rollover
                or volume_fade
            )
        )
    )
    live_continuation = bool(
        up_closes_last3 >= 2
        and close >= prev_close * (1.0 - max(atr_pct * 0.18, 0.0004))
        and close_position_in_candle >= 0.56
        and volume_spike >= max(0.95, prev_volume_spike * 0.88, 0.85 if close < 1.0 else 0.65)
        and ema20 >= ema20_prev
        and not volume_fade
        and not momentum_rollover
        and not strong_upper_reversal_bar
    )
    mtf_reacceleration = bool(
        (
            (mtf_trend_5m >= 0.0060 and mtf_rsi_5m >= 74.0)
            or (mtf_trend_15m >= 0.0044 and mtf_rsi_15m >= 68.0)
        )
        and close >= prev_close * (1.0 - max(atr_pct * 0.16, 0.00035))
        and close_position_in_candle >= 0.48
        and ema20 >= ema20_prev
        and not volume_fade
        and not momentum_rollover
        and not strong_upper_reversal_bar
    )
    late_rebound_without_trigger = bool(
        far_from_recent_peak
        and meaningfully_below_vah
        and close >= ema20 * (1.0 - max(atr_pct * 0.18, 0.00045))
        and close_position_in_candle >= 0.46
        and (up_closes_last3 >= 2 or close >= prev_close * (1.0 - max(atr_pct * 0.14, 0.00035)))
        and volume_spike >= max(0.22, prev_volume_spike * 0.70)
        and not volume_fade
        and not momentum_rollover
        and not strong_upper_reversal_bar
    )
    downside_displacement_tolerance = max(atr_pct * 0.18, 0.00045 if close < 0.02 else 0.00030)
    downside_displacement_confirmed = bool(
        recent_support_reference > 0
        and close <= recent_support_reference * (1.0 - downside_displacement_tolerance)
    ) or bool(
        close <= prev_close * (1.0 - downside_displacement_tolerance * 0.75)
        and close_position_in_candle <= 0.34
    )
    late_pullback_without_downside_displacement = bool(
        far_from_recent_peak
        and meaningfully_below_vah
        and close <= ema20 * (1.0 + max(atr_pct * 0.18, 0.00055))
        and close_position_in_candle <= 0.48
        and (close - open_px) / max(close, 1e-8) <= max(atr_pct * 0.06, 0.00045)
        and volume_spike <= max(1.35, prev_volume_spike * 1.06)
        and not downside_displacement_confirmed
    )
    green_hold_without_downside_displacement = bool(
        (recent_upper_anchor_touch or above_fast_upper_band)
        and not downside_displacement_confirmed
        and max(0.0, (recent_high_24 - close) / max(recent_high_24, 1e-8)) <= max(
            atr_pct * 1.20,
            0.0068 if close < 0.05 else 0.0046,
        )
        and close >= ema20 * (1.0 - max(atr_pct * 0.10, 0.00035))
        and close_position_in_candle >= 0.70
        and (close - open_px) / max(close, 1e-8) >= max(atr_pct * 0.025, 0.00018)
        and (high - max(open_px, close)) / candle_range <= 0.30
        and (
            mtf_reacceleration
            or above_fast_upper_band
            or volume_spike >= max(0.62, prev_volume_spike * 0.78)
        )
        and not failed_reclaim
        and not retest_failed_breakout
        and not strong_upper_reversal_bar
    )
    bullish_rebound_without_reject = bool(
        far_from_recent_peak
        and meaningfully_below_vah
        and close >= ema20 * (1.0 - max(atr_pct * 0.18, 0.00045))
        and close_position_in_candle >= 0.60
        and (close - open_px) / max(close, 1e-8) >= max(atr_pct * 0.06, 0.00045)
        and up_closes_last3 >= 2
        and volume_spike >= max(0.36, prev_volume_spike * 0.74)
        and not volume_fade
        and not momentum_rollover
        and not strong_upper_reversal_bar
    )
    shallow_plateau_reclaim = bool(
        recent_upper_anchor_touch
        and recent_upper_anchor_reject
        and close_position_in_candle >= 0.52
        and close >= prev_close * (1.0 - max(atr_pct * 0.08, 0.00025))
        and high >= prev_high * (1.0 - max(atr_pct * 0.08, 0.00025))
        and not strong_upper_reversal_bar
    )
    plateau_continuation_near_high = bool(
        recent_upper_anchor_touch
        and (not recent_upper_anchor_reject or shallow_plateau_reclaim)
        and max(0.0, (recent_high_24 - close) / max(recent_high_24, 1e-8)) <= max(
            atr_pct * 0.48,
            0.0032 if close < 0.05 else 0.0021,
        )
        and close >= ema20 * (1.0 - max(atr_pct * 0.14, 0.00035))
        and close_position_in_candle >= 0.56
        and (close - open_px) / max(close, 1e-8) >= -max(atr_pct * 0.08, 0.00040)
        and (high - max(open_px, close)) / candle_range <= 0.34
        and (up_closes_last3 >= 2 or close >= prev_close * (1.0 - max(atr_pct * 0.08, 0.00025)))
        and volume_spike >= max(0.32, prev_volume_spike * 0.72)
        and not strong_upper_reversal_bar
    )
    return bool(
        (live_continuation and (near_peak or near_vah))
        or (mtf_reacceleration and (near_peak or near_vah or far_from_recent_peak))
        or late_rebound_without_trigger
        or late_pullback_without_downside_displacement
        or green_hold_without_downside_displacement
        or bullish_rebound_without_reject
        or plateau_continuation_near_high
    )


def _refresh_symbol_decision_live(
    *,
    symbol: str,
    sync: "ExchangeSyncService",
    execution: ExecutionEngine,
    feed: "MarketDataFeed",
    pipeline: "FeaturePipeline",
    timeframe: str,
    candles_limit: int,
    strategy,
) -> dict[str, object] | None:
    snapshot = sync.snapshot(symbol)
    rec_state = execution.state_machine.reconcile(symbol, snapshot.positions, snapshot.open_orders)
    prepared = _prepare_symbol_analysis(
        symbol=symbol,
        snapshot=snapshot,
        rec_state=rec_state,
        feed=feed,
        pipeline=pipeline,
        timeframe=timeframe,
        candles_limit=candles_limit,
        overlay_live_price=True,
    )
    if str(prepared.get("status") or "") != "ok":
        return None
    refreshed = _prepare_symbol_decision(
        symbol=symbol,
        snapshot=snapshot,
        rec_state=rec_state,
        prepared=prepared,
        timeframe=timeframe,
        strategy=strategy,
    )
    if str(refreshed.get("status") or "") != "ok":
        return None
    return refreshed


def _resolve_signal_profile(raw: str | None) -> str:
    profile = str(raw or "both").strip().lower()
    return profile if profile in {"both", "main", "early"} else "both"


def _profiled_env_first_nonempty(profile: str, *names: str) -> str:
    normalized = _resolve_signal_profile(profile)
    if normalized not in {"main", "early"}:
        expanded: list[str] = []
        for name in names:
            expanded.extend(
                [
                    name,
                    f"{name}_MAIN",
                    f"MAIN_{name}",
                    f"{name}_EARLY",
                    f"EARLY_{name}",
                ]
            )
        for name in expanded:
            value = str(os.getenv(name, "")).strip()
            if value:
                return value
        return ""

    suffix = normalized.upper()
    expanded: list[str] = []
    for name in names:
        expanded.extend(
            [
                f"{name}_{suffix}",
                f"{suffix}_{name}",
                name,
            ]
        )
    for name in expanded:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


def _profiled_dataset_path(base_path: str, profile: str) -> str:
    explicit = str(
        os.getenv(f"ML_ONLINE_DATASET_PATH_{profile.upper()}", "")
        or os.getenv(f"{profile.upper()}_ML_ONLINE_DATASET_PATH", "")
        or os.getenv("ML_ONLINE_DATASET_PATH", "")
    ).strip()
    if explicit or profile == "both":
        return explicit or base_path
    path = Path(base_path)
    suffix = path.suffix or ".csv"
    return str(path.with_name(f"{path.stem}_{profile}{suffix}"))


def _profiled_model_dir(base_dir: str, profile: str) -> str:
    explicit = str(
        os.getenv(f"ML_MODEL_DIR_{profile.upper()}", "")
        or os.getenv(f"{profile.upper()}_ML_MODEL_DIR", "")
        or os.getenv("ML_MODEL_DIR", "")
    ).strip()
    if explicit or profile == "both":
        return explicit or base_dir
    return str(Path(base_dir) / profile)


def _profiled_pending_path(base_path: str, profile: str) -> str:
    explicit = str(
        os.getenv(f"ML_ONLINE_PENDING_PATH_{profile.upper()}", "")
        or os.getenv(f"{profile.upper()}_ML_ONLINE_PENDING_PATH", "")
        or os.getenv("ML_ONLINE_PENDING_PATH", "")
    ).strip()
    if explicit or profile == "both":
        return explicit or base_path
    path = Path(base_path)
    suffix = path.suffix or ".json"
    return str(path.with_name(f"{path.stem}_{profile}{suffix}"))


def _build_context_chart_caption(symbol: str, *, stage_label: str, timeframe_label: str) -> str:
    clean = str(symbol or "").replace("/", "").upper().strip()
    clean_stage_label = _normalize_human_text(stage_label)
    return (
        f"<b>{clean_stage_label}</b>\n"
        f"<b>{clean}</b> | <code>{clean}</code>\n"
        f"\u0421\u0442\u0430\u0440\u0448\u0438\u0439 \u0422\u0424: {timeframe_label}\n"
        f"HTF \u0443\u0440\u043e\u0432\u043d\u0438 + \u043a\u0430\u0440\u0442\u0430 \u043b\u0438\u043a\u0432\u0438\u0434\u0430\u0446\u0438\u0439"
    )


def _build_clean_context_chart_caption(symbol: str, *, stage_label: str, timeframe_label: str) -> str:
    clean = str(symbol or "").replace("/", "").upper().strip()
    raw_stage_label = str(stage_label or "").strip()
    clean_stage_label = _normalize_human_text(raw_stage_label)
    clean_stage_upper = clean_stage_label.upper()
    raw_stage_upper = raw_stage_label.upper()
    if "HTF" in raw_stage_upper:
        if "\u0420\u0410\u041d\u041d\u0418\u0419" in clean_stage_upper or "EARLY" in clean_stage_upper:
            clean_stage_label = "\u0420\u0410\u041d\u041d\u0418\u0419 \u0428\u041e\u0420\u0422: HTF \u041a\u041e\u041d\u0422\u0415\u041a\u0421\u0422"
        elif "\u041b\u041e\u041d\u0413" in clean_stage_upper or "LONG" in clean_stage_upper:
            clean_stage_label = "\u041b\u041e\u041d\u0413 \u0421\u0418\u0413\u041d\u0410\u041b: HTF \u041a\u041e\u041d\u0422\u0415\u041a\u0421\u0422"
        elif "\u0428\u041e\u0420\u0422" in clean_stage_upper or "SHORT" in clean_stage_upper:
            clean_stage_label = "\u0428\u041e\u0420\u0422 \u0421\u0418\u0413\u041d\u0410\u041b: HTF \u041a\u041e\u041d\u0422\u0415\u041a\u0421\u0422"
    return (
        f"<b>{clean_stage_label}</b>\n"
        f"<b>{clean}</b> | <code>{clean}</code>\n"
        f"\u0421\u0442\u0430\u0440\u0448\u0438\u0439 \u0422\u0424: {timeframe_label}\n"
        f"HTF \u0443\u0440\u043e\u0432\u043d\u0438 + \u043a\u0430\u0440\u0442\u0430 \u043b\u0438\u043a\u0432\u0438\u0434\u0430\u0446\u0438\u0439"
    )


def _build_higher_timeframe_chart(
    *,
    symbol: str,
    side: str,
    entry: float,
    tp: float,
    sl: float,
    feed: "MarketDataFeed",
    pipeline: "FeaturePipeline",
    runtime_extras: dict[str, object] | None = None,
) -> bytes | None:
    context_timeframe = str(os.getenv("BOT_ALERT_CONTEXT_TIMEFRAME", "240"))
    context_candles = max(96, _as_int(os.getenv("BOT_ALERT_CONTEXT_CANDLES", "160"), 160))
    try:
        context_frame = feed.fetch_frame(
            symbol=symbol,
            timeframe=context_timeframe,
            candles=context_candles,
            include_liquidations=True,
        )
        if context_frame.ohlcv.empty:
            return None
        as_of = context_frame.ohlcv.index[-1]
        try:
            context_extras = dict(runtime_extras or {})
            if context_frame.liquidation_cluster_high is not None:
                context_extras["liquidation_cluster_high"] = context_frame.liquidation_cluster_high
            if context_frame.liquidation_cluster_low is not None:
                context_extras["liquidation_cluster_low"] = context_frame.liquidation_cluster_low
            context_bundle = pipeline.build(
                symbol=symbol,
                ohlcv=context_frame.ohlcv,
                as_of=as_of,
                extras=context_extras,
            )
            context_enriched = context_bundle.enriched
            try:
                context_enriched.attrs.update(getattr(context_frame.ohlcv, "attrs", {}) or {})
            except Exception:
                pass
        except Exception:
            context_enriched = context_frame.ohlcv

        return _build_alert_chart(
            symbol=symbol,
            timeframe=context_timeframe,
            enriched=context_enriched,
            side=side,
            entry=entry,
            tp=tp,
            sl=sl,
            show_trade_levels=True,
            show_entry_levels=False,
            show_liquidation_map=True,
            timeframe_label=_format_chart_timeframe_label(context_timeframe),
            liquidation_cluster_high=context_frame.liquidation_cluster_high,
            liquidation_cluster_low=context_frame.liquidation_cluster_low,
        )
    except Exception:
        return None


def _build_early_watch_candidate_legacy(*, symbol: str, timeframe: str, mode: str, enriched, intent) -> dict[str, object] | None:
    meta = intent.metadata if isinstance(intent.metadata, Mapping) else {}
    failed_layer = str(meta.get("layer_failed") or "")
    trace = meta.get("layer_trace", {}) if isinstance(meta, Mapping) else {}
    layers = trace.get("layers", {}) if isinstance(trace, Mapping) else {}
    if not isinstance(layers, Mapping):
        return None

    regime_entry = layers.get("regime_filter", {})
    layer1_entry = layers.get("layer1_pump_detection", {})
    layer2_entry = layers.get("layer2_weakness_confirmation", {})
    regime_passed = bool(regime_entry.get("passed")) if isinstance(regime_entry, Mapping) else False
    layer1_passed = bool(layer1_entry.get("passed")) if isinstance(layer1_entry, Mapping) else False

    if not regime_passed:
        return None

    layer1 = _extract_layer_details(meta, "layer1_pump_detection")
    layer2 = _extract_layer_details(meta, "layer2_weakness_confirmation")
    layer3 = _extract_layer_details(meta, "layer3_entry_location")
    liquidation_map = build_liquidation_map(enriched)
    close = _as_float(enriched.iloc[-1].get("close"), 0.0)
    atr = max(_as_float(enriched.iloc[-1].get("atr"), close * 0.01), close * 0.001, 1e-8)
    synthetic_tp = close - atr * 1.6
    synthetic_sl = close + atr

    if failed_layer == "layer2_weakness_confirmation" and layer1_passed:
        layer2_score = _as_float(layer2.get("weakness_strength"), 0.0)
        layer2_triggers: list[str] = []
        if bool(_as_float(layer2.get("near_high_context"), 0.0)):
            layer2_triggers.append("С†РµРЅР° РµС‰С‘ Сѓ Р»РѕРєР°Р»СЊРЅРѕРіРѕ С…Р°СЏ")
        if obv_divergence:
            layer2_triggers.append("OBV СѓР¶Рµ РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚")
        if cvd_divergence:
            layer2_triggers.append("CVD СѓР¶Рµ РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚")
        if liquidation_map.swept_above:
            layer2_triggers.append("\\u0421\\u043d\\u044f\\u043b\\u0438 \\u0432\\u0435\\u0440\\u0445\\u043d\\u044e\\u044e \\u043b\\u0438\\u043a\\u0432\\u0438\\u0434\\u043d\\u043e\\u0441\\u0442\\u044c")
        if liquidation_map.downside_magnet:
            layer2_triggers.append("\\u043d\\u0438\\u0436\\u0435 \\u0435\\u0441\\u0442\\u044c \\u043b\\u0438\\u043a\\u0432\\u0438\\u0434\\u0430\\u0446\\u0438\\u043e\\u043d\\u043d\\u044b\\u0439 \\u043c\\u0430\\u0433\\u043d\\u0438\\u0442")
        if layer2_score < 0.67 and len(layer2_triggers) < 2:
            return None
        continuation_risk = max(0.0, 2.0 - layer2_score)
        if liquidation_map.upside_risk > 0:
            continuation_risk += min(1.2, liquidation_map.upside_risk * 0.35)
        return {
            "phase": "SETUP",
            "caption": build_early_signal_caption(
                symbol=symbol,
                timeframe=timeframe,
                mode=mode,
                phase_label="\\u0420\\u0410\\u041d\\u041d\\u0418\\u0419 \\u0428\\u041e\\u0420\\u0422: \\u0421\\u0415\\u0422\\u0410\\u041f",
                price=close,
                trace_meta=meta,
                watch_score=max(4.5, layer2_score * 6.0),
                watch_max_score=8.0,
                continuation_risk=continuation_risk,
                continuation_max_score=4.0,
                triggers=layer2_triggers or ["РїР°РјРї СѓР¶Рµ РµСЃС‚СЊ", "СЃР»Р°Р±РѕСЃС‚СЊ СЂСЏРґРѕРј", "Р¶РґС‘Рј РІС…РѕРґ РїРѕ СЃС‚СЂР°С‚РµРіРёРё"],
                wait_for="подтверждение полноценного входа",
                enriched=enriched,
            ),
            "entry": close,
            "tp": synthetic_tp,
            "sl": synthetic_sl,
        }

    if failed_layer != "layer1_pump_detection":
        return None

    if not bool(_as_float(layer1.get("clean_pump_ok"), 0.0)):
        return None

    row = enriched.iloc[-1]
    prev = enriched.iloc[-2] if len(enriched) > 1 else row

    close = _as_float(row.get("close"), 0.0)
    high = _as_float(row.get("high"), close)
    low = _as_float(row.get("low"), close)
    open_px = _as_float(row.get("open"), close)
    bb_upper = _as_float(row.get("bb_upper"), close)
    kc_upper = _as_float(row.get("kc_upper"), close)
    rsi = max(_as_float(layer1.get("rsi"), 0.0), _as_float(row.get("rsi"), 50.0))
    prev_rsi = _as_float(prev.get("rsi"), rsi)
    volume_spike = max(_as_float(layer1.get("volume_spike"), 0.0), _as_float(row.get("volume_spike"), 1.0))
    volume_gate = max(
        _as_float(layer1.get("volume_spike_threshold_used"), _as_float(os.getenv("VOLUME_THRESHOLD"), 2.0)),
        2.0,
    )
    recent_volume_spike_series = (
        enriched.tail(5)["volume_spike"] if "volume_spike" in enriched.columns else pd.Series([volume_spike])
    )
    recent_volume_spike_numeric = pd.to_numeric(recent_volume_spike_series, errors="coerce").dropna()
    recent_volume_spike = _as_float(recent_volume_spike_numeric.max(), volume_spike)
    volume_peak_age = 0
    if not recent_volume_spike_numeric.empty:
        volume_spike_values = recent_volume_spike_numeric.to_numpy(dtype=float)
        volume_peak_age = max(len(volume_spike_values) - 1 - int(np.nanargmax(volume_spike_values)), 0)
    hist = _as_float(row.get("hist"), 0.0)
    prev_hist = _as_float(prev.get("hist"), hist)
    recent_high_series = enriched.tail(8)["high"] if "high" in enriched.columns else pd.Series([high])
    recent_high = _as_float(pd.to_numeric(recent_high_series, errors="coerce").max(), high)
    recent_close_high_series = enriched.tail(8)["close"] if "close" in enriched.columns else pd.Series([close])
    recent_close_high = _as_float(pd.to_numeric(recent_close_high_series, errors="coerce").max(), close)
    candle_range = max(high - low, 1e-8)
    upper_wick = max(high - max(open_px, close), 0.0)
    prev_close = _as_float(prev.get("close"), close)
    prev_high = _as_float(prev.get("high"), high)
    prev_low = _as_float(prev.get("low"), low)
    obv = _as_float(row.get("obv"), 0.0)
    prev_obv = _as_float(prev.get("obv"), obv)
    recent_obv_series = enriched.tail(6).head(5)["obv"] if "obv" in enriched.columns else pd.Series([obv])
    recent_obv_ref = _as_float(pd.to_numeric(recent_obv_series, errors="coerce").max(), obv)
    cvd = _as_float(row.get("cvd"), 0.0)
    prev_cvd = _as_float(prev.get("cvd"), cvd)
    recent_cvd_series = enriched.tail(6).head(5)["cvd"] if "cvd" in enriched.columns else pd.Series([cvd])
    recent_cvd_ref = _as_float(pd.to_numeric(recent_cvd_series, errors="coerce").max(), cvd)

    if max(volume_spike, recent_volume_spike) < volume_gate:
        return None

    continuation_risk = 0.0
    if close >= signal_peak_reference * 0.999:
        continuation_risk += 1.25
    if volume_spike >= max(recent_volume_spike * 0.97, volume_gate):
        continuation_risk += 1.25
    if rsi >= prev_rsi and rsi >= 58.0:
        continuation_risk += 1.0
    if hist >= prev_hist and hist > 0:
        continuation_risk += 1.0
    if close >= prev_close:
        continuation_risk += 0.5
    if upper_wick / candle_range < 0.16:
        continuation_risk += 0.75
    if liquidation_map.upside_risk > 0:
        continuation_risk += min(1.75, liquidation_map.upside_risk * 0.55)
    if (
        liquidation_map.nearest_above_distance_pct is not None
        and liquidation_map.nearest_above_distance_pct < 0.0045
        and liquidation_map.upside_risk >= 1.8
        and not liquidation_map.swept_above
    ):
        continuation_risk += 0.75

    still_accelerating = (
        close >= recent_high * 0.999
        and volume_spike >= max(recent_volume_spike * 0.95, volume_gate)
        and rsi >= prev_rsi
        and hist >= prev_hist
    )
    fresh_watch_window = peak_still_fresh or (
        peak_recent_enough
        and first_reaction
        and pullback_from_peak_pct <= max(peak_pullback_limit * 0.82, early_reversal_pullback * 1.18, 0.0046)
        and pump_drawdown_ratio <= min(early_watch_max_drawdown_ratio, 0.072)
    )
    if not fresh_watch_window and not liquidation_map.swept_above:
        return None
    if (still_accelerating and not first_reaction) or continuation_risk >= 4.15:
        return None

    soft_regime_watch = failed_layer == "regime_filter"

    weighted_triggers: list[tuple[str, float]] = []
    obv_divergence = obv <= max(recent_obv_ref * 0.998, prev_obv)
    cvd_divergence = cvd <= max(recent_cvd_ref * 0.998, prev_cvd)
    if peak_still_fresh:
        weighted_triggers.append(("РїРёРє РїР°РјРїР° СЃРѕРІСЃРµРј СЃРІРµР¶РёР№", 1.25))
    if peak_still_fresh:
        weighted_triggers.append(("\\u043f\\u0438\\u043a \\u043f\\u0430\\u043c\\u043f\\u0430 \\u0441\\u043e\\u0432\\u0441\\u0435\\u043c \\u0441\\u0432\\u0435\\u0436\\u0438\\u0439", 1.25))
    if near_peak:
        weighted_triggers.append(("С†РµРЅР° РµС‰С‘ Сѓ РІРµСЂС€РёРЅС‹ РїР°РјРїР°", 1.15))
    if first_reaction:
        weighted_triggers.append(("РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·", 1.20))
    if near_peak:
        weighted_triggers.append(("С†РµРЅР° РµС‰С‘ Сѓ РІРµСЂС€РёРЅС‹ РїР°РјРїР°", 1.15))
    if first_reaction:
        weighted_triggers.append(("РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·", 1.20))
    if near_peak:
        weighted_triggers.append(("С†РµРЅР° РµС‰С‘ Сѓ РІРµСЂС€РёРЅС‹ РїР°РјРїР°", 1.15))
    if first_reaction:
        weighted_triggers.append(("РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·", 1.20))
    if close >= max(bb_upper, kc_upper) * 0.998:
        weighted_triggers.append(("С†РµРЅР° Сѓ РІРµСЂС…РЅРµР№ Р·РѕРЅС‹", 1.0))
    if close >= signal_peak_reference * 0.995:
        weighted_triggers.append(("С†РµРЅР° Сѓ Р»РѕРєР°Р»СЊРЅРѕРіРѕ С…Р°СЏ", 1.0))
    if rsi >= 55.0:
        weighted_triggers.append(("RSI РІС‹С€Рµ РЅРµР№С‚СЂР°Р»Рё", 0.5))
    if rsi < prev_rsi and rsi >= 52.0:
        weighted_triggers.append(("RSI СЂР°Р·РІРѕСЂР°С‡РёРІР°РµС‚СЃСЏ РІРЅРёР·", 1.25))
    if volume_spike >= volume_gate:
        weighted_triggers.append(("РѕР±СЉС‘Рј РµС‰С‘ РїРѕРІС‹С€РµРЅ", 0.5))
    if recent_volume_spike > 0 and volume_spike < recent_volume_spike:
        weighted_triggers.append(("РѕР±СЉС‘Рј Р·Р°С‚СѓС…Р°РµС‚", 1.25))
    if upper_wick / candle_range >= 0.35:
        weighted_triggers.append(("РµСЃС‚СЊ РІРµСЂС…РЅСЏСЏ С‚РµРЅСЊ", 1.0))
    if hist < prev_hist:
        weighted_triggers.append(("MACD РѕСЃР»Р°Р±РµРІР°РµС‚", 1.25))
    if close <= recent_close_high * 0.9995:
        weighted_triggers.append(("С†РµРЅР° РїРµСЂРµСЃС‚Р°Р»Р° СѓСЃРєРѕСЂСЏС‚СЊСЃСЏ", 0.75))
    if obv <= max(recent_obv_ref * 0.998, prev_obv):
        weighted_triggers.append(("OBV РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚", 1.25))
    if cvd <= max(recent_cvd_ref * 0.998, prev_cvd):
        weighted_triggers.append(("CVD РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚", 1.25))

    if liquidation_map.swept_above:
        weighted_triggers.append(("\\u0421\\u043d\\u044f\\u043b\\u0438 \\u0432\\u0435\\u0440\\u0445\\u043d\\u044e\\u044e \\u043b\\u0438\\u043a\\u0432\\u0438\\u0434\\u043d\\u043e\\u0441\\u0442\\u044c", 1.35))
    if liquidation_map.downside_magnet:
        weighted_triggers.append(("\\u043d\\u0438\\u0436\\u0435 \\u0435\\u0441\\u0442\\u044c \\u043b\\u0438\\u043a\\u0432\\u0438\\u0434\\u0430\\u0446\\u0438\\u043e\\u043d\\u043d\\u044b\\u0439 \\u043c\\u0430\\u0433\\u043d\\u0438\\u0442", 1.15))

    obv_divergence = obv <= max(recent_obv_ref * 0.998, prev_obv)
    cvd_divergence = cvd <= max(recent_cvd_ref * 0.998, prev_cvd)
    unique_triggers: list[str] = []
    score = 0.0
    for label, weight in weighted_triggers:
        if label in unique_triggers:
            continue
        unique_triggers.append(label)
        score += float(weight)

    weakness_markers = {
        "RSI СЂР°Р·РІРѕСЂР°С‡РёРІР°РµС‚СЃСЏ РІРЅРёР·",
        "РѕР±СЉС‘Рј Р·Р°С‚СѓС…Р°РµС‚",
        "MACD РѕСЃР»Р°Р±РµРІР°РµС‚",
        "OBV РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚",
        "CVD РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚",
        "С†РµРЅР° РїРµСЂРµСЃС‚Р°Р»Р° СѓСЃРєРѕСЂСЏС‚СЊСЃСЏ",
        "\\u0421\\u043d\\u044f\\u043b\\u0438 \\u0432\\u0435\\u0440\\u0445\\u043d\\u044e\\u044e \\u043b\\u0438\\u043a\\u0432\\u0438\\u0434\\u043d\\u043e\\u0441\\u0442\\u044c",
        "\\u043d\\u0438\\u0436\\u0435 \\u0435\\u0441\\u0442\\u044c \\u043b\\u0438\\u043a\\u0432\\u0438\\u0434\\u0430\\u0446\\u0438\\u043e\\u043d\\u043d\\u044b\\u0439 \\u043c\\u0430\\u0433\\u043d\\u0438\\u0442",
    }
    if score < 4.5 or len(unique_triggers) < 4:
        return None
    if not any(trigger in weakness_markers for trigger in unique_triggers):
        return None

    return {
        "phase": "WATCH",
        "caption": build_early_signal_caption(
            symbol=symbol,
            timeframe=timeframe,
            mode=mode,
            phase_label="\\u0420\\u0410\\u041d\\u041d\\u0418\\u0419 \\u0428\\u041e\\u0420\\u0422: \\u041d\\u0410\\u0411\\u041b\\u042e\\u0414\\u0415\\u041d\\u0418\\u0415",
            price=close,
            trace_meta=meta,
            watch_score=score,
            watch_max_score=10.0,
            continuation_risk=continuation_risk,
            continuation_max_score=5.75,
            triggers=unique_triggers,
            wait_for="подтверждение слабости и входа",
            enriched=enriched,
        ),
        "entry": close,
        "tp": synthetic_tp,
        "sl": synthetic_sl,
    }


def _build_early_watch_candidate(*, symbol: str, timeframe: str, mode: str, enriched, intent) -> dict[str, object] | None:
    meta = intent.metadata if isinstance(intent.metadata, Mapping) else {}
    failed_layer = str(meta.get("layer_failed") or "")
    trace = meta.get("layer_trace", {}) if isinstance(meta, Mapping) else {}
    layers = trace.get("layers", {}) if isinstance(trace, Mapping) else {}
    if not isinstance(layers, Mapping):
        return None

    regime_entry = layers.get("regime_filter", {})
    layer1_entry = layers.get("layer1_pump_detection", {})
    regime = _extract_layer_details(meta, "regime_filter")
    layer1 = _extract_layer_details(meta, "layer1_pump_detection")
    layer2 = _extract_layer_details(meta, "layer2_weakness_confirmation")
    layer3 = _extract_layer_details(meta, "layer3_entry_location")
    regime_passed = bool(regime_entry.get("passed")) if isinstance(regime_entry, Mapping) else False
    layer1_passed = bool(layer1_entry.get("passed")) if isinstance(layer1_entry, Mapping) else False
    layer2_failed_reclaim = bool(_as_float(layer2.get("failed_reclaim"), 0.0))
    layer2_retest_failed_breakout = bool(_as_float(layer2.get("retest_failed_breakout"), 0.0))
    layer2_acceptance_above_high = bool(_as_float(layer2.get("acceptance_above_swing_high"), 0.0))
    layer3_failed_reclaim = bool(_as_float(layer3.get("failed_reclaim"), 0.0))
    layer3_retest_failed_breakout = bool(_as_float(layer3.get("retest_failed_breakout"), 0.0))
    layer3_acceptance_above_high = bool(_as_float(layer3.get("acceptance_above_swing_high"), 0.0))
    regime_failed_reason = str(regime.get("failed_reason") or "")
    regime_missing_conditions = str(regime.get("missing_conditions") or "")
    soft_regime_fail = (
        failed_layer == "regime_filter"
        and not regime_passed
        and "news_veto" not in regime_failed_reason
        and "news_veto" not in regime_missing_conditions
    )

    liquidation_map = build_liquidation_map(enriched)
    row = enriched.iloc[-1]
    prev = enriched.iloc[-2] if len(enriched) > 1 else row

    close = _as_float(row.get("close"), 0.0)
    atr = max(_as_float(row.get("atr"), close * 0.01), close * 0.001, 1e-8)
    ema20_last = _as_float(row.get("ema20"), close)
    local_vah = _as_float(row.get("vah"), 0.0)
    local_poc = _as_float(row.get("poc"), 0.0)
    local_val = _as_float(row.get("val"), 0.0)

    confirmed_pump_min = max(_as_float(layer1.get("clean_pump_min_pct_used"), 0.05), 0.0)
    early_pump_min = max(
        confirmed_pump_min,
        _early_config_float("EARLY_WATCH_CLEAN_PUMP_MIN_PCT", max(0.0195, confirmed_pump_min - 0.0220)),
    )
    early_structural_pump_min = max(
        early_pump_min,
        _early_config_float(
            "EARLY_WATCH_STRUCTURAL_PUMP_MIN_PCT",
            max(early_pump_min, min(confirmed_pump_min, early_pump_min + 0.0085)),
        ),
    )
    reported_clean_pump_pct = _as_float(layer1.get("clean_pump_pct"), 0.0)
    clean_pump_pct = reported_clean_pump_pct
    confirmed_volume_gate = max(
        _as_float(layer1.get("volume_spike_threshold_used"), _as_float(os.getenv("VOLUME_THRESHOLD"), 2.0)),
        2.0,
    )
    early_volume_gate = max(
        0.22,
        _early_config_float("EARLY_WATCH_VOLUME_SPIKE_MIN", max(0.22, confirmed_volume_gate * 0.16)),
    )
    early_rsi_min = _early_config_float("EARLY_WATCH_RSI_MIN", 42.0)
    early_watch_score_min = _early_config_float("EARLY_WATCH_SCORE_MIN", 1.55)
    early_quality_min = _early_config_float("EARLY_WATCH_QUALITY_MIN", 1.95)

    high = _as_float(row.get("high"), close)
    low = _as_float(row.get("low"), close)
    open_px = _as_float(row.get("open"), close)
    bb_upper = _as_float(row.get("bb_upper"), close)
    kc_upper = _as_float(row.get("kc_upper"), close)
    rsi = max(_as_float(layer1.get("rsi"), 0.0), _as_float(row.get("rsi"), 50.0))
    prev_rsi = _as_float(prev.get("rsi"), rsi)
    volume_spike = max(_as_float(layer1.get("volume_spike"), 0.0), _as_float(row.get("volume_spike"), 1.0))
    recent_volume_spike_series = (
        enriched.tail(8)["volume_spike"] if "volume_spike" in enriched.columns else pd.Series([volume_spike])
    )
    recent_volume_spike_numeric = pd.to_numeric(recent_volume_spike_series, errors="coerce").dropna()
    recent_volume_spike = _as_float(recent_volume_spike_numeric.max(), volume_spike)
    volume_peak_age = 0
    if not recent_volume_spike_numeric.empty:
        volume_spike_values = recent_volume_spike_numeric.to_numpy(dtype=float)
        volume_peak_age = max(len(volume_spike_values) - 1 - int(np.nanargmax(volume_spike_values)), 0)
    hist = _as_float(row.get("hist"), 0.0)
    prev_hist = _as_float(prev.get("hist"), hist)
    mtf_trend_5m = _as_float(row.get("mtf_trend_5m"), 0.0)
    mtf_rsi_5m = _as_float(row.get("mtf_rsi_5m"), 50.0)
    mtf_trend_15m = _as_float(row.get("mtf_trend_15m"), 0.0)
    mtf_rsi_15m = _as_float(row.get("mtf_rsi_15m"), 50.0)
    mtf_trend_1h = _as_float(row.get("mtf_trend_1h"), 0.0)
    mtf_rsi_1h = _as_float(row.get("mtf_rsi_1h"), 50.0)
    recent_peak_window = min(len(enriched), 32 if close < 0.02 else 26)
    pump_context_window = min(len(enriched), 56 if close < 0.02 else 42)
    recent_high_series = enriched.tail(recent_peak_window)["high"] if "high" in enriched.columns else pd.Series([high])
    recent_high_numeric = pd.to_numeric(recent_high_series, errors="coerce").dropna()
    recent_high = _as_float(recent_high_numeric.max(), high)
    recent_close_high_series = enriched.tail(recent_peak_window)["close"] if "close" in enriched.columns else pd.Series([close])
    recent_close_numeric = pd.to_numeric(recent_close_high_series, errors="coerce").dropna()
    recent_close_high = _as_float(recent_close_numeric.max(), close)
    recent_close_tail = (
        pd.to_numeric(enriched.tail(4)["close"], errors="coerce").dropna()
        if "close" in enriched.columns
        else pd.Series([close])
    )
    recent_high_tail = (
        pd.to_numeric(enriched.tail(5)["high"], errors="coerce").dropna()
        if "high" in enriched.columns
        else pd.Series([high])
    )
    recent_low_tail = (
        pd.to_numeric(enriched.tail(4)["low"], errors="coerce").dropna()
        if "low" in enriched.columns
        else pd.Series([low])
    )
    signal_peak_reference = recent_close_high if recent_close_high > 0 else recent_high
    if recent_high > signal_peak_reference > 0:
        allowed_peak_wick = max(
            atr * 0.32,
            signal_peak_reference * (0.0024 if close < 0.02 else 0.0018),
        )
        signal_peak_reference = min(recent_high, signal_peak_reference + allowed_peak_wick)
    candle_range = max(high - low, 1e-8)
    upper_wick = max(high - max(open_px, close), 0.0)
    prev_close = _as_float(prev.get("close"), close)
    prev_high = _as_float(prev.get("high"), high)
    prev_low = _as_float(prev.get("low"), low)
    close_position_in_candle = max(0.0, min(1.0, (close - low) / candle_range))
    obv = _as_float(row.get("obv"), 0.0)
    prev_obv = _as_float(prev.get("obv"), obv)
    recent_obv_series = enriched.tail(6).head(5)["obv"] if "obv" in enriched.columns else pd.Series([obv])
    recent_obv_ref = _as_float(pd.to_numeric(recent_obv_series, errors="coerce").max(), obv)
    cvd = _as_float(row.get("cvd"), 0.0)
    prev_cvd = _as_float(prev.get("cvd"), cvd)
    recent_cvd_series = enriched.tail(6).head(5)["cvd"] if "cvd" in enriched.columns else pd.Series([cvd])
    recent_cvd_ref = _as_float(pd.to_numeric(recent_cvd_series, errors="coerce").max(), cvd)
    support_window = min(len(enriched), 36 if close < 0.02 else 28)
    support_lows_series = enriched.tail(support_window)["low"] if "low" in enriched.columns else pd.Series([low])
    support_lows_numeric = pd.to_numeric(support_lows_series, errors="coerce").dropna()
    local_support = _as_float(support_lows_numeric.quantile(0.18), low if low > 0 else close)
    deep_support = _as_float(support_lows_numeric.quantile(0.08), local_support)
    up_closes_last3 = 0
    prev2_close = prev_close
    if len(recent_close_tail) >= 2:
        recent_close_values = recent_close_tail.to_numpy(dtype=float)
        up_closes_last3 = int(sum(curr > prev_close_value for prev_close_value, curr in zip(recent_close_values[:-1], recent_close_values[1:])))
        if len(recent_close_values) >= 3:
            prev2_close = _as_float(recent_close_values[-3], prev_close)
    prev2_high = prev_high
    if len(recent_high_tail) >= 3:
        recent_high_values = recent_high_tail.to_numpy(dtype=float)
        prev2_high = _as_float(recent_high_values[-3], prev_high)
    ema20_prev = _as_float(prev.get("ema20"), ema20_last)
    ema20_slope = ema20_last - ema20_prev
    close_peak_age = _latest_peak_age_bars(
        recent_close_numeric,
        reference_price=close,
        atr=atr,
        relative_tolerance=0.0018 if close < 0.02 else 0.0012,
    )
    high_peak_age = _latest_peak_age_bars(
        recent_high_numeric,
        reference_price=close,
        atr=atr,
        relative_tolerance=0.0022 if close < 0.02 else 0.0016,
    )
    peak_age_bars = min(close_peak_age, high_peak_age) if not recent_close_numeric.empty and not recent_high_numeric.empty else (
        close_peak_age if not recent_close_numeric.empty else high_peak_age
    )
    pullback_from_peak_pct = max(0.0, (signal_peak_reference - close) / max(signal_peak_reference, 1e-8))
    pump_context_tail = enriched.tail(pump_context_window)
    tail_highs = pd.to_numeric(pump_context_tail["high"], errors="coerce").to_numpy(dtype=float) if "high" in pump_context_tail.columns else []
    tail_lows = pd.to_numeric(pump_context_tail["low"], errors="coerce").to_numpy(dtype=float) if "low" in pump_context_tail.columns else []
    active_pump_low = low if low > 0 else close
    if len(tail_highs) and len(tail_lows):
        local_peak_rel = int(np.nanargmax(tail_highs))
        pump_back = min(local_peak_rel, 42 if close < 0.02 else 30)
        pump_base_start = max(0, local_peak_rel - pump_back)
        active_pump_low = _as_float(np.nanmin(tail_lows[pump_base_start : local_peak_rel + 1]), active_pump_low)
    clean_pump_pct = max(
        clean_pump_pct,
        max(0.0, (signal_peak_reference - active_pump_low) / max(active_pump_low, 1e-8)),
    )
    reported_pump_below_min = (
        reported_clean_pump_pct > 0.0
        and reported_clean_pump_pct + 1e-8 < early_pump_min
    )
    marginal_watch_pump = clean_pump_pct < max(
        early_structural_pump_min + 0.0075,
        early_structural_pump_min * 1.16,
    )
    pump_amplitude = max(signal_peak_reference - active_pump_low, atr)
    pump_drawdown_ratio = max(0.0, (signal_peak_reference - close) / max(pump_amplitude, 1e-8))
    pump_range_position = min(1.0, max(0.0, (close - active_pump_low) / max(pump_amplitude, 1e-8)))
    pump_midpoint = active_pump_low + pump_amplitude * 0.52
    peak_pullback_limit = max(0.0058, min(0.0175, (atr / max(close, 1e-8)) * 2.45))
    early_reversal_pullback = max(0.0011, min(0.0055, (atr / max(close, 1e-8)) * 0.90))
    minimum_reversal_pullback = max(0.0009, min(0.0032, early_reversal_pullback * 0.55))
    early_watch_max_drawdown_ratio = _early_config_float("EARLY_WATCH_MAX_PUMP_DRAWDOWN_RATIO", 0.078)
    early_setup_max_drawdown_ratio = _early_config_float("EARLY_SETUP_MAX_PUMP_DRAWDOWN_RATIO", 0.112)
    early_watch_min_peak_position = _early_config_float("EARLY_WATCH_MIN_PUMP_RANGE_POSITION", 0.64)
    early_setup_min_peak_position = _early_config_float("EARLY_SETUP_MIN_PUMP_RANGE_POSITION", 0.60)
    early_watch_peak_age_cap = max(2, _as_int(os.getenv("EARLY_WATCH_MAX_PEAK_AGE_BARS"), 4 if close >= 0.02 else 5))
    early_setup_peak_age_cap = max(2, _as_int(os.getenv("EARLY_SETUP_MAX_PEAK_AGE_BARS"), early_watch_peak_age_cap + 1))
    volume_climax_recent = recent_volume_spike >= early_volume_gate and volume_peak_age <= (4 if close >= 0.02 else 5)
    volume_fade_confirmed = recent_volume_spike > 0 and volume_peak_age <= 5 and volume_spike <= recent_volume_spike * 0.88
    current_volume_supportive = volume_spike >= max(0.46, early_volume_gate * 0.38)
    peak_still_fresh = peak_age_bars <= 1
    peak_recent_enough = peak_age_bars <= 2
    near_peak_limit = peak_pullback_limit * (
        1.12
        if peak_still_fresh
        else 1.08
        if peak_recent_enough
        else 1.0
    )
    near_peak = pullback_from_peak_pct <= near_peak_limit
    recent_reaction_floor = _as_float(pd.to_numeric(recent_low_tail.tail(3), errors="coerce").min(), low)
    price_rejection_score = 0.0
    if pullback_from_peak_pct >= minimum_reversal_pullback:
        price_rejection_score += 1.0
    if close < prev_close:
        price_rejection_score += 0.8
    if close_position_in_candle <= 0.58:
        price_rejection_score += 0.7
    if upper_wick / candle_range >= 0.22:
        price_rejection_score += 0.65
    if recent_reaction_floor > 0 and low <= recent_reaction_floor * 0.9994:
        price_rejection_score += 0.45
    bearish_reversal_bar = (
        (close < prev_close or close < open_px)
        and close_position_in_candle <= 0.70
    )
    stall_rejection_bar = (
        pullback_from_peak_pct >= minimum_reversal_pullback * 0.82
        and close_position_in_candle <= 0.58
        and upper_wick / candle_range >= 0.24
    )
    terminal_rejection_bar = bearish_reversal_bar or stall_rejection_bar
    momentum_rollover_score = 0.0
    if rsi < prev_rsi:
        momentum_rollover_score += 1.0
    if hist < prev_hist:
        momentum_rollover_score += 1.0
    if obv <= max(recent_obv_ref * 0.998, prev_obv):
        momentum_rollover_score += 0.5
    if cvd <= max(recent_cvd_ref * 0.998, prev_cvd):
        momentum_rollover_score += 0.5
    real_rollover = (
        terminal_rejection_bar
        and price_rejection_score >= 1.80
        and (
            momentum_rollover_score >= 1.25
            or liquidation_map.swept_above
            or upper_wick / candle_range >= 0.30
        )
    )
    micro_reversal_near_peak = (
        peak_still_fresh
        and near_peak
        and terminal_rejection_bar
        and price_rejection_score >= 1.25
        and (momentum_rollover_score >= 1.2 or upper_wick / candle_range >= 0.28)
        and (close < prev_close or close_position_in_candle <= 0.62)
    )
    peak_turn_pullback = max(
        minimum_reversal_pullback * 1.12,
        early_reversal_pullback * 0.92,
        0.0016 if close < 0.02 else 0.0010,
    )
    peak_turn_confirmed = (
        peak_recent_enough
        and near_peak
        and terminal_rejection_bar
        and pullback_from_peak_pct >= peak_turn_pullback
        and (close < prev_close or close_position_in_candle <= 0.64 or upper_wick / candle_range >= 0.26)
        and (
            high <= prev_high * 1.0004
            or close <= signal_peak_reference * (1.0 - max(peak_turn_pullback * 0.42, 0.0009 if close < 0.02 else 0.0006))
            or momentum_rollover_score >= 1.0
        )
    )
    first_reaction = real_rollover or peak_turn_confirmed or (
        pullback_from_peak_pct >= early_reversal_pullback * 1.18
        and close < prev_close
        and close_position_in_candle <= 0.68
        and high <= signal_peak_reference * 1.0003
    )
    bullish_continuation_bar = (
        close > open_px
        and close >= prev_close
        and close_position_in_candle >= 0.70
        and upper_wick / candle_range <= 0.16
    )
    live_peak_extension = (
        near_peak
        and close >= prev_close
        and close_position_in_candle >= 0.74
        and upper_wick / candle_range <= 0.18
        and high >= signal_peak_reference * 0.9994
        and not terminal_rejection_bar
    )
    reclaimed_peak = (
        peak_recent_enough
        and close >= signal_peak_reference * (1.0 - max(early_reversal_pullback * 0.34, 0.0016 if close < 0.02 else 0.0010))
        and (high >= signal_peak_reference * 0.9992 or close >= prev_close)
    )
    hard_reclaim_continuation = (
        reclaimed_peak
        and close_position_in_candle >= 0.68
        and close >= prev_close
        and high >= signal_peak_reference * 0.9992
        and momentum_rollover_score < 1.5
        and not liquidation_map.swept_above
    )
    trend_follow_through = (
        up_closes_last3 >= 2
        and ema20_slope >= 0
        and close >= ema20_last * (1.0014 if close < 0.02 else 1.0010)
        and bullish_continuation_bar
    )
    fresh_breakout_bar = (
        near_peak
        and high >= max(prev_high * 0.9996, signal_peak_reference * 0.9992)
        and close >= prev_close
        and close_position_in_candle >= 0.66
        and upper_wick / candle_range <= 0.18
    )
    stair_step_continuation = (
        up_closes_last3 >= 2
        and ema20_slope > 0
        and close >= ema20_last * (1.0017 if close < 0.02 else 1.0012)
        and close >= prev_close
        and high >= prev_high * 0.9995
    )
    peak_reclaim_without_reject = (
        near_peak
        and reclaimed_peak
        and price_rejection_score < 1.85
        and upper_wick / candle_range < 0.24
    )
    continuation_after_pause = (
        near_peak
        and close >= prev_close
        and close_position_in_candle >= 0.60
        and upper_wick / candle_range < 0.22
        and high >= signal_peak_reference * 0.9994
        and momentum_rollover_score < 1.5
        and not liquidation_map.swept_above
    )
    reacceleration_after_pullback = (
        peak_recent_enough
        and first_reaction
        and close > prev_close
        and high >= prev_high * 0.9993
        and close >= signal_peak_reference * 0.9988
        and close_position_in_candle >= 0.57
        and upper_wick / candle_range <= 0.22
        and hist >= prev_hist
        and rsi >= prev_rsi * 0.998
        and not liquidation_map.swept_above
    )
    mtf_5m_continuation_drive = bool(
        mtf_trend_5m >= _early_config_float("EARLY_WATCH_MTF_TREND_RISK_MIN", 0.0060)
        and mtf_rsi_5m >= _early_config_float("EARLY_WATCH_MTF_RSI_RISK_MIN", 75.0)
    )
    mtf_15m_continuation_drive = bool(
        mtf_trend_15m >= _early_config_float("EARLY_WATCH_MTF_TREND_15M_RISK_MIN", 0.0046)
        and mtf_rsi_15m >= _early_config_float("EARLY_WATCH_MTF_RSI_15M_RISK_MIN", 69.0)
    )
    mtf_1h_continuation_drive = bool(
        mtf_trend_1h >= _early_config_float("EARLY_WATCH_MTF_TREND_1H_RISK_MIN", 0.0038)
        and mtf_rsi_1h >= _early_config_float("EARLY_WATCH_MTF_RSI_1H_RISK_MIN", 64.0)
    )
    mtf_continuation_drive = bool(
        (mtf_5m_continuation_drive or mtf_15m_continuation_drive or mtf_1h_continuation_drive)
        and close >= prev_close * (1.0 - (0.0012 if close < 0.02 else 0.0008))
        and close_position_in_candle >= 0.54
        and not real_rollover
        and not peak_turn_confirmed
        and not micro_reversal_near_peak
        and not liquidation_map.swept_above
    )
    continuation_structure_score = 0.0
    if fresh_breakout_bar:
        continuation_structure_score += 1.45
    if stair_step_continuation:
        continuation_structure_score += 1.20
    if peak_reclaim_without_reject:
        continuation_structure_score += 1.05
    if live_peak_extension:
        continuation_structure_score += 1.15
    if reacceleration_after_pullback:
        continuation_structure_score += 1.35
    if close > prev_close and close_position_in_candle >= 0.62:
        continuation_structure_score += 0.45
    if close >= ema20_last * (1.0015 if close < 0.02 else 1.0010) and ema20_slope > 0:
        continuation_structure_score += 0.35
    if mtf_5m_continuation_drive:
        continuation_structure_score += 0.55
    if mtf_15m_continuation_drive:
        continuation_structure_score += 0.60
    if mtf_1h_continuation_drive:
        continuation_structure_score += 0.50
    lower_close_followthrough = (
        close < prev_close
        and prev_close <= prev2_close * 1.0006
    )
    lower_high_followthrough = (
        peak_recent_enough
        and high <= prev_high * 0.9998
        and prev_high >= prev2_high * 0.9992
    )
    break_prev_low_followthrough = (
        peak_recent_enough
        and close <= prev_low * (0.9996 if close < 0.02 else 0.9998)
    )
    post_peak_followthrough = (
        peak_recent_enough
        and pullback_from_peak_pct >= max(minimum_reversal_pullback * 1.18, early_reversal_pullback * 0.76)
        and (
            lower_close_followthrough
            or lower_high_followthrough
            or break_prev_low_followthrough
        )
    )
    peak_followthrough_confirmed = (
        post_peak_followthrough
        or (
            micro_reversal_near_peak
            and close < prev_close
            and high <= signal_peak_reference * 0.9994
        )
    )
    structural_reversal_ready = bool(
        peak_followthrough_confirmed
        or micro_reversal_near_peak
        or layer2_failed_reclaim
        or layer2_retest_failed_breakout
        or layer3_failed_reclaim
        or layer3_retest_failed_breakout
        or liquidation_map.swept_above
    )
    structure_trigger_confirmed = bool(
        layer2_failed_reclaim
        or layer2_retest_failed_breakout
        or layer3_failed_reclaim
        or layer3_retest_failed_breakout
        or liquidation_map.swept_above
    )
    weak_watch_structure = not (
        structure_trigger_confirmed
        or liquidation_map.swept_above
        or peak_followthrough_confirmed
    )
    acceptance_above_high = bool(layer2_acceptance_above_high or layer3_acceptance_above_high)
    stop_ref = max(
        signal_peak_reference,
        _as_float(layer1.get("bollinger_upper_metric_used"), 0.0),
        _as_float(layer1.get("keltner_upper_metric_used"), 0.0),
        close + atr * 0.82,
    )
    structural_stop_context = structural_reversal_ready or first_reaction or liquidation_map.swept_above
    synthetic_sl = stop_ref + max(
        atr * (0.26 if structural_stop_context else 0.18),
        close * (0.00110 if close < 0.02 else 0.00085),
    )
    stop_distance = max(
        synthetic_sl - close,
        atr * (0.88 if structural_stop_context else 0.76),
        close * (0.00125 if close < 0.02 else 0.00100),
    )
    synthetic_rr = 2.35 if failed_layer == "layer1_pump_detection" else 2.65
    if peak_recent_enough and near_peak and first_reaction:
        synthetic_rr += 0.20
    if structural_reversal_ready:
        synthetic_rr += 0.18
    synthetic_tp = close - stop_distance * synthetic_rr
    structural_tp_candidates = [
        value
        for value in (local_poc, local_val, local_support, deep_support)
        if value and value > 0 and value < close
    ]
    if structural_tp_candidates:
        synthetic_tp = min(synthetic_tp, min(structural_tp_candidates))
    if pullback_from_peak_pct > max(peak_pullback_limit * 1.12, 0.0088):
        return None
    if not peak_recent_enough and pullback_from_peak_pct >= max(early_reversal_pullback * 0.80, 0.0028):
        return None
    if peak_age_bars > max(early_setup_peak_age_cap + 1, 5):
        return None

    atr_pct = atr / max(close, 1e-8)
    vah_reject_tolerance = max(
        atr_pct * 1.15,
        0.0048 if close < 0.02 else 0.0030,
    )
    peak_zone_tolerance = max(
        vah_reject_tolerance * 1.10,
        0.0056 if close < 0.02 else 0.0036,
    )
    value_room_floor = max(
        atr_pct * 0.72,
        0.0048 if close < 0.02 else 0.0028,
    )
    near_vah_resistance = bool(
        local_vah > 0
        and abs(close - local_vah) / max(close, 1e-8) <= vah_reject_tolerance
    )
    near_peak_resistance = bool(
        signal_peak_reference > 0
        and (signal_peak_reference - close) / max(signal_peak_reference, 1e-8) <= peak_zone_tolerance
    )
    value_room_to_poc = bool(
        local_poc > 0
        and (close - local_poc) / max(close, 1e-8) >= value_room_floor * 0.78
    )
    value_room_to_val = bool(
        local_val > 0
        and (close - local_val) / max(close, 1e-8) >= value_room_floor
    )
    value_room_to_support = bool(
        local_support > 0
        and (close - local_support) / max(close, 1e-8) >= value_room_floor * 1.05
    )
    downside_target_ready = bool(
        liquidation_map.downside_magnet
        or value_room_to_poc
        or value_room_to_val
        or value_room_to_support
    )
    upper_rejection_zone_ready = bool(
        near_vah_resistance
        or liquidation_map.swept_above
        or (
            (layer2_failed_reclaim or layer3_failed_reclaim or layer2_retest_failed_breakout or layer3_retest_failed_breakout)
            and near_peak_resistance
        )
        or (near_peak_resistance and terminal_rejection_bar)
    )
    open_upside_liq_magnet = bool(
        liquidation_map.nearest_above_distance_pct is not None
        and liquidation_map.nearest_above_distance_pct <= max(
            atr_pct * 1.65,
            0.0060 if close < 0.02 else 0.0040,
        )
        and not liquidation_map.swept_above
    )
    resistance_confluence_score = 0.0
    if near_vah_resistance:
        resistance_confluence_score += 1.00
    if near_peak_resistance:
        resistance_confluence_score += 0.75
    if liquidation_map.swept_above:
        resistance_confluence_score += 1.15
    if layer2_failed_reclaim or layer3_failed_reclaim:
        resistance_confluence_score += 1.05
    if layer2_retest_failed_breakout or layer3_retest_failed_breakout:
        resistance_confluence_score += 0.90
    if terminal_rejection_bar:
        resistance_confluence_score += 0.35
    downside_confluence_score = 0.0
    if liquidation_map.downside_magnet:
        downside_confluence_score += 1.15
    if value_room_to_poc:
        downside_confluence_score += 0.70
    if value_room_to_val:
        downside_confluence_score += 0.85
    if value_room_to_support:
        downside_confluence_score += 0.55
    recent_upper_liq_reaction = any(
        band.side == "above"
        and band.closed_index is not None
        and band.closed_index >= max(len(enriched) - 8, 0)
        and (
            abs(band.level - signal_peak_reference) / max(signal_peak_reference, 1e-8) <= peak_zone_tolerance * 1.35
            or abs(band.level - close) / max(close, 1e-8) <= peak_zone_tolerance * 1.10
        )
        for band in liquidation_map.bands
    )
    lower_liq_target_ready = any(
        band.side == "below"
        and band.closed_index is None
        and band.weight >= 1.05
        and 0.003 <= (close - band.level) / max(close, 1e-8) <= max(value_room_floor * 3.20, 0.035)
        for band in liquidation_map.bands
    )
    if recent_upper_liq_reaction:
        resistance_confluence_score += 0.90
    if lower_liq_target_ready:
        downside_confluence_score += 0.80
    anchor_touch_tolerance = max(
        peak_zone_tolerance * 0.92,
        atr_pct * 1.05,
        0.0048 if close < 0.02 else 0.0031,
    )
    recent_upper_anchor_levels = [
        level
        for level in (local_vah, signal_peak_reference)
        if level > 0
    ]
    recent_upper_anchor_levels.extend(
        band.level
        for band in liquidation_map.bands
        if band.side == "above"
        and (
            band.closed_index is None
            or band.closed_index >= max(len(enriched) - 8, 0)
        )
    )
    recent_high_samples = recent_high_tail.tail(3).to_numpy(dtype=float) if len(recent_high_tail) else np.array([], dtype=float)
    recent_upper_anchor_touch = bool(
        recent_upper_anchor_levels
        and recent_high_samples.size
        and any(
            abs(high_px - anchor_level) / max(anchor_level, 1e-8) <= anchor_touch_tolerance
            for high_px in recent_high_samples
            for anchor_level in recent_upper_anchor_levels
        )
    )
    anchor_reject_score = 0.0
    if terminal_rejection_bar:
        anchor_reject_score += 1.05
    if peak_followthrough_confirmed:
        anchor_reject_score += 1.00
    if structure_trigger_confirmed:
        anchor_reject_score += 1.05
    if recent_upper_liq_reaction:
        anchor_reject_score += 0.95
    if liquidation_map.swept_above:
        anchor_reject_score += 0.90
    if near_vah_resistance:
        anchor_reject_score += 0.50
    if first_reaction and (rsi < prev_rsi or hist < prev_hist or volume_fade_confirmed):
        anchor_reject_score += 0.45
    anchor_reject_floor = (
        1.10
        if recent_upper_liq_reaction or liquidation_map.swept_above
        else 1.35
        if near_vah_resistance and downside_confluence_score >= 0.85 and first_reaction
        else 1.55
    )
    recent_upper_anchor_reject = bool(
        recent_upper_anchor_touch
        and anchor_reject_score >= anchor_reject_floor
    )
    if recent_upper_anchor_touch:
        resistance_confluence_score += 0.55
    if recent_upper_anchor_reject:
        resistance_confluence_score += 0.35
    strong_rejection_cluster = resistance_confluence_score >= 2.00
    hard_upper_anchor_context = bool(
        near_vah_resistance
        or recent_upper_liq_reaction
        or liquidation_map.swept_above
        or layer2_failed_reclaim
        or layer3_failed_reclaim
        or layer2_retest_failed_breakout
        or layer3_retest_failed_breakout
    )
    watch_downside_target_ready = bool(
        downside_target_ready
        or lower_liq_target_ready
        or (
            near_peak_resistance
            and first_reaction
            and resistance_confluence_score >= 1.80
            and pullback_from_peak_pct <= max(early_reversal_pullback * 1.08, 0.0048)
            and (value_room_to_poc or value_room_to_val or value_room_to_support or liquidation_map.downside_magnet)
        )
        or (
            strong_rejection_cluster
            and (
                liquidation_map.swept_above
                or near_vah_resistance
                or (near_peak_resistance and terminal_rejection_bar)
            )
            and (value_room_to_poc or value_room_to_val or value_room_to_support or liquidation_map.downside_magnet)
        )
    )
    fresh_live_continuation_without_reject = bool(
        (
            live_peak_extension
            or trend_follow_through
            or reacceleration_after_pullback
            or mtf_continuation_drive
            or continuation_structure_score >= 2.25
            or (
                near_peak
                and close >= prev_close
                and close_position_in_candle >= 0.68
                and upper_wick / candle_range <= 0.20
                and volume_spike >= max(recent_volume_spike * 0.94, early_volume_gate * 0.80, 0.75)
            )
        )
        and not (
            liquidation_map.swept_above
            or recent_upper_liq_reaction
            or structure_trigger_confirmed
            or peak_followthrough_confirmed
            or micro_reversal_near_peak
            or real_rollover
        )
    )
    shallow_reject_pullback_floor = max(
        minimum_reversal_pullback * 1.08,
        atr_pct * 0.84,
        0.0015 if close < 0.02 else 0.0010,
    )
    shallow_reject_without_displacement = bool(
        recent_upper_anchor_touch
        and (
            recent_upper_anchor_reject
            or micro_reversal_near_peak
            or terminal_rejection_bar
        )
        and pullback_from_peak_pct < shallow_reject_pullback_floor
        and close >= signal_peak_reference * (1.0 - max(shallow_reject_pullback_floor * 0.72, 0.0010 if close < 0.02 else 0.0007))
        and not liquidation_map.swept_above
        and not recent_upper_liq_reaction
    )
    if fresh_live_continuation_without_reject:
        return None
    if shallow_reject_without_displacement:
        return None
    if not upper_rejection_zone_ready:
        return None
    if not recent_upper_anchor_touch and not liquidation_map.swept_above and not recent_upper_liq_reaction:
        return None
    if (
        not recent_upper_anchor_reject
        and not micro_reversal_near_peak
        and not peak_turn_confirmed
        and not structure_trigger_confirmed
        and not soft_regime_fail
    ):
        return None
    if (
        not hard_upper_anchor_context
        and not soft_regime_fail
        and not (
            peak_followthrough_confirmed
            and terminal_rejection_bar
            and micro_reversal_near_peak
            and downside_confluence_score >= 1.10
            and resistance_confluence_score >= 2.25
        )
    ):
        return None
    if not watch_downside_target_ready:
        return None
    if (
        open_upside_liq_magnet
        and not (
            liquidation_map.swept_above
            or layer2_failed_reclaim
            or layer3_failed_reclaim
            or layer2_retest_failed_breakout
            or layer3_retest_failed_breakout
            or peak_followthrough_confirmed
            or resistance_confluence_score >= 2.35
        )
    ):
        return None
    upper_zone_distance_candidates = []
    if local_vah > 0:
        upper_zone_distance_candidates.append(abs(close - local_vah) / max(close, 1e-8))
    if signal_peak_reference > 0:
        upper_zone_distance_candidates.append(abs(signal_peak_reference - close) / max(close, 1e-8))
    for band in liquidation_map.bands:
        if band.side != "above":
            continue
        if band.closed_index is None and band.level > close:
            upper_zone_distance_candidates.append(abs(band.level - close) / max(close, 1e-8))
        elif band.closed_index is not None and band.closed_index >= max(len(enriched) - 8, 0):
            upper_zone_distance_candidates.append(abs(band.level - close) / max(close, 1e-8))
    entry_zone_distance_pct = min(upper_zone_distance_candidates) if upper_zone_distance_candidates else 1.0
    max_watch_entry_zone_distance = max(
        atr_pct * 1.30,
        0.0068 if close < 0.02 else 0.0044,
    )
    if entry_zone_distance_pct > max_watch_entry_zone_distance and not liquidation_map.swept_above:
        return None
    if resistance_confluence_score < 1.55 and not liquidation_map.swept_above and not soft_regime_fail:
        return None
    if downside_confluence_score < 0.55 and not liquidation_map.downside_magnet and not soft_regime_fail:
        return None

    if soft_regime_fail:
        regime_watch_score = 0.0
        if peak_still_fresh:
            regime_watch_score += 1.00
        if near_peak:
            regime_watch_score += 1.10
        if first_reaction:
            regime_watch_score += 1.15
        if micro_reversal_near_peak:
            regime_watch_score += 1.10
        if peak_followthrough_confirmed:
            regime_watch_score += 1.05
        if volume_fade_confirmed:
            regime_watch_score += 0.55
        if hist < prev_hist:
            regime_watch_score += 0.55
        if rsi < prev_rsi:
            regime_watch_score += 0.45
        if close < prev_close:
            regime_watch_score += 0.50
        if liquidation_map.swept_above:
            regime_watch_score += 0.90
        if liquidation_map.downside_magnet:
            regime_watch_score += 0.75

        regime_failure_score = 0.0
        if first_reaction:
            regime_failure_score += 0.80
        if micro_reversal_near_peak:
            regime_failure_score += 1.00
        if peak_followthrough_confirmed:
            regime_failure_score += 1.00
        if volume_fade_confirmed:
            regime_failure_score += 0.50
        if hist < prev_hist:
            regime_failure_score += 0.40
        if rsi < prev_rsi:
            regime_failure_score += 0.30
        if close < prev_close:
            regime_failure_score += 0.45
        if liquidation_map.swept_above:
            regime_failure_score += 0.70

        regime_continuation_risk = continuation_structure_score
        if live_peak_extension:
            regime_continuation_risk += 1.00
        if trend_follow_through:
            regime_continuation_risk += 0.90
        if reclaimed_peak:
            regime_continuation_risk += 0.80
        if bullish_continuation_bar:
            regime_continuation_risk += 0.55
        if reacceleration_after_pullback:
            regime_continuation_risk += 0.95
        if liquidation_map.upside_risk > 0:
            regime_continuation_risk += min(1.0, liquidation_map.upside_risk * 0.28)
        if liquidation_map.swept_above:
            regime_continuation_risk = max(0.0, regime_continuation_risk - 0.55)
        if micro_reversal_near_peak:
            regime_continuation_risk = max(0.0, regime_continuation_risk - 0.35)
        if strong_rejection_cluster:
            regime_continuation_risk = max(0.0, regime_continuation_risk - 0.18)
        if downside_confluence_score >= 1.00:
            regime_continuation_risk = max(0.0, regime_continuation_risk - 0.10)

        regime_quality_score = min(
            10.0,
            max(
                0.0,
                4.7
                + regime_watch_score * 0.70
                - regime_continuation_risk * 0.58
                + (0.40 if peak_still_fresh else 0.0)
                + (0.35 if near_peak else 0.0)
                + (0.35 if first_reaction else 0.0),
            ),
        )
        if strong_rejection_cluster:
            regime_quality_score += min(0.45, resistance_confluence_score * 0.14)
        if downside_confluence_score > 0.0:
            regime_quality_score += min(0.25, downside_confluence_score * 0.08)
        regime_quality_score = min(10.0, regime_quality_score)

        regime_hard_reversal = (
            micro_reversal_near_peak
            or liquidation_map.swept_above
            or peak_followthrough_confirmed
            or (
                first_reaction
                and terminal_rejection_bar
                and (
                    rsi < prev_rsi
                    or hist < prev_hist
                    or volume_fade_confirmed
                )
            )
        )

        if (
            clean_pump_pct >= early_pump_min
            and peak_recent_enough
            and (near_peak or pump_range_position >= early_watch_min_peak_position * 0.91)
            and regime_hard_reversal
            and regime_failure_score >= 1.15
            and regime_watch_score >= max(2.05, early_watch_score_min + 0.15)
            and regime_quality_score >= max(3.60, early_quality_min + 0.85)
            and regime_continuation_risk < 2.85
            and not hard_reclaim_continuation
            and not live_peak_extension
            and not reacceleration_after_pullback
        ):
            regime_triggers = [
                "РїР°РјРї СѓР¶Рµ РµСЃС‚СЊ",
                "С†РµРЅР° РµС‰С‘ Сѓ РІРµСЂС€РёРЅС‹ РїР°РјРїР°",
            ]
            if first_reaction:
                regime_triggers.append("РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·")
            if peak_followthrough_confirmed:
                regime_triggers.append("РїРѕСЃР»Рµ РїРёРєР° РїРѕСЏРІРёР»СЃСЏ lower-high / lower-close")
            if hist < prev_hist:
                regime_triggers.append("MACD РѕСЃР»Р°Р±РµРІР°РµС‚")
            if rsi < prev_rsi:
                regime_triggers.append("RSI СЂР°Р·РІРѕСЂР°С‡РёРІР°РµС‚СЃСЏ РІРЅРёР·")
            if liquidation_map.downside_magnet:
                regime_triggers.append("РЅРёР¶Рµ РµСЃС‚СЊ Р»РёРєРІРёРґР°С†РёРѕРЅРЅС‹Р№ РјР°РіРЅРёС‚")
            if liquidation_map.swept_above:
                regime_triggers.append("РІРµСЂС…РЅСЋСЋ Р»РёРєРІРёРґРЅРѕСЃС‚СЊ СѓР¶Рµ СЃРЅСЏР»Рё")

            return {
                "phase": "WATCH",
                "caption": build_early_signal_caption(
                    symbol=symbol,
                    timeframe=timeframe,
                    mode=mode,
                    phase_label="\\u0420\\u0410\\u041d\\u041d\\u0418\\u0419 \\u0428\\u041e\\u0420\\u0422: \\u041d\\u0410\\u0411\\u041b\\u042e\\u0414\\u0415\\u041d\\u0418\\u0415",
                    price=close,
                    trace_meta=meta,
                    watch_score=regime_watch_score,
                    watch_max_score=8.0,
                    quality_score=regime_quality_score,
                    quality_max_score=10.0,
                    quality_grade=_early_quality_grade(regime_quality_score),
                    continuation_risk=regime_continuation_risk,
                    continuation_max_score=4.0,
                    triggers=regime_triggers,
                    wait_for="подтверждение слабости и входа",
                    enriched=enriched,
                ),
                "entry": close,
                "tp": synthetic_tp,
                "sl": synthetic_sl,
            }

    if failed_layer == "layer2_weakness_confirmation" and layer1_passed:
        layer2_score = _as_float(layer2.get("weakness_strength"), 0.0)
        obv_divergence = bool(_as_float(layer2.get("obv_bearish_divergence"), 0.0))
        cvd_divergence = bool(_as_float(layer2.get("cvd_bearish_divergence"), 0.0))
        near_high_context = bool(_as_float(layer2.get("near_high_context"), 0.0))
        moderate_continuation_risk = continuation_structure_score
        if live_peak_extension:
            moderate_continuation_risk += 1.25
        if trend_follow_through:
            moderate_continuation_risk += 1.0
        if reclaimed_peak:
            moderate_continuation_risk += 0.8
        if bullish_continuation_bar:
            moderate_continuation_risk += 0.55
        if reacceleration_after_pullback:
            moderate_continuation_risk += 1.0
        if liquidation_map.upside_risk > 0:
            moderate_continuation_risk += min(1.0, liquidation_map.upside_risk * 0.28)
        if strong_rejection_cluster:
            moderate_continuation_risk = max(0.0, moderate_continuation_risk - 0.18)
        if downside_confluence_score >= 1.00:
            moderate_continuation_risk = max(0.0, moderate_continuation_risk - 0.10)

        moderate_triggers: list[str] = []
        if peak_still_fresh:
            moderate_triggers.append("РїРёРє РїР°РјРїР° СЃРѕРІСЃРµРј СЃРІРµР¶РёР№")
        if near_peak:
            moderate_triggers.append("С†РµРЅР° РµС‰С‘ Сѓ РІРµСЂС€РёРЅС‹ РїР°РјРїР°")
        if first_reaction:
            moderate_triggers.append("РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·")
        if micro_reversal_near_peak:
            moderate_triggers.append("Р»РѕРєР°Р»СЊРЅС‹Р№ РїРёРє СѓР¶Рµ РЅР°С‡Р°Р» СЂР°Р·РІРѕСЂР°С‡РёРІР°С‚СЊСЃСЏ")
        if obv_divergence:
            moderate_triggers.append("OBV СѓР¶Рµ РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚")
        if cvd_divergence:
            moderate_triggers.append("CVD СѓР¶Рµ РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚")
        if liquidation_map.swept_above:
            moderate_triggers.append("РІРµСЂС…РЅСЋСЋ Р»РёРєРІРёРґРЅРѕСЃС‚СЊ СѓР¶Рµ СЃРЅСЏР»Рё")
        if liquidation_map.downside_magnet:
            moderate_triggers.append("РЅРёР¶Рµ РµСЃС‚СЊ Р»РёРєРІРёРґР°С†РёРѕРЅРЅС‹Р№ РјР°РіРЅРёС‚")

        moderate_watch_score = 0.0
        if near_peak:
            moderate_watch_score += 1.0
        if first_reaction:
            moderate_watch_score += 1.2
        if micro_reversal_near_peak:
            moderate_watch_score += 1.1
        if peak_followthrough_confirmed:
            moderate_watch_score += 1.15
        if obv_divergence:
            moderate_watch_score += 0.9
        if cvd_divergence:
            moderate_watch_score += 0.9
        if liquidation_map.swept_above:
            moderate_watch_score += 1.0
        if liquidation_map.downside_magnet:
            moderate_watch_score += 0.8
        moderate_watch_score += min(max(layer2_score, 0.0), 1.0) * 2.2

        moderate_quality_score = min(
            10.0,
            max(
                0.0,
                4.9
                + moderate_watch_score * 0.62
                - moderate_continuation_risk * 0.62
                + (0.50 if peak_still_fresh else 0.0)
                + (0.35 if near_peak else 0.0)
                + (0.40 if first_reaction else 0.0)
                + (0.45 if micro_reversal_near_peak else 0.0),
            ),
        )
        if strong_rejection_cluster:
            moderate_quality_score += min(0.45, resistance_confluence_score * 0.14)
        if downside_confluence_score > 0.0:
            moderate_quality_score += min(0.25, downside_confluence_score * 0.08)
        moderate_quality_score = min(10.0, moderate_quality_score)
        moderate_failure_score = 0.0
        if first_reaction:
            moderate_failure_score += 0.85
        if micro_reversal_near_peak:
            moderate_failure_score += 1.00
        if real_rollover:
            moderate_failure_score += 1.05
        if peak_followthrough_confirmed:
            moderate_failure_score += 1.10
        if volume_fade_confirmed:
            moderate_failure_score += 0.55
        if hist < prev_hist:
            moderate_failure_score += 0.40
        if rsi < prev_rsi:
            moderate_failure_score += 0.30
        if close < prev_close:
            moderate_failure_score += 0.45
        if liquidation_map.swept_above:
            moderate_failure_score += 0.75
        moderate_trigger_ready = bool(
            first_reaction
            or micro_reversal_near_peak
            or liquidation_map.swept_above
        )
        moderate_hard_reversal = (
            micro_reversal_near_peak
            or liquidation_map.swept_above
            or (first_reaction and peak_followthrough_confirmed)
            or (first_reaction and real_rollover)
        )
        moderate_hard_weakness_count = 0
        if micro_reversal_near_peak:
            moderate_hard_weakness_count += 1
        if peak_followthrough_confirmed:
            moderate_hard_weakness_count += 1
        if real_rollover:
            moderate_hard_weakness_count += 1
        if liquidation_map.swept_above:
            moderate_hard_weakness_count += 1
        if obv_divergence:
            moderate_hard_weakness_count += 1
        if cvd_divergence:
            moderate_hard_weakness_count += 1
        if volume_fade_confirmed:
            moderate_hard_weakness_count += 1
        if rsi < prev_rsi and hist < prev_hist:
            moderate_hard_weakness_count += 1

        moderate_reversal_ready = (
            moderate_hard_reversal
            or (
                first_reaction
                and terminal_rejection_bar
                and (
                    obv_divergence
                    or cvd_divergence
                    or volume_fade_confirmed
                    or (rsi < prev_rsi and hist < prev_hist)
                )
            )
        )

        if (
            clean_pump_pct >= early_pump_min
            and layer2_score >= 0.42
            and peak_recent_enough
            and (near_peak or near_high_context)
            and moderate_trigger_ready
            and moderate_reversal_ready
            and moderate_hard_weakness_count >= (1 if liquidation_map.swept_above else 2)
            and not acceptance_above_high
            and not reclaimed_peak
            and not peak_reclaim_without_reject
            and not hard_reclaim_continuation
            and not live_peak_extension
            and not reacceleration_after_pullback
            and moderate_continuation_risk < 2.65
            and moderate_failure_score >= (1.25 if liquidation_map.swept_above else 1.55)
            and moderate_watch_score >= max(2.35, early_watch_score_min + 0.40)
            and moderate_quality_score >= max(4.10, early_quality_min + 1.65)
        ):
            return {
                "phase": "WATCH",
                "caption": build_early_signal_caption(
                    symbol=symbol,
                    timeframe=timeframe,
                    mode=mode,
                    phase_label="\\u0420\\u0410\\u041d\\u041d\\u0418\\u0419 \\u0428\\u041e\\u0420\\u0422: \\u041d\\u0410\\u0411\\u041b\\u042e\\u0414\\u0415\\u041d\\u0418\\u0415",
                    price=close,
                    trace_meta=meta,
                    watch_score=moderate_watch_score,
                    watch_max_score=8.0,
                    quality_score=moderate_quality_score,
                    quality_max_score=10.0,
                    quality_grade=_early_quality_grade(moderate_quality_score),
                    continuation_risk=moderate_continuation_risk,
                    continuation_max_score=4.0,
                    triggers=moderate_triggers or ["РїР°РјРї СѓР¶Рµ РµСЃС‚СЊ", "РїРѕСЏРІРёР»Р°СЃСЊ РїРµСЂРІР°СЏ СЃР»Р°Р±РѕСЃС‚СЊ", "Р¶РґС‘Рј РїРѕРґС‚РІРµСЂР¶РґРµРЅРёРµ"],
                    wait_for="подтверждение слабости и точки входа",
                    enriched=enriched,
                ),
                "entry": close,
                "tp": synthetic_tp,
                "sl": synthetic_sl,
            }

        if not moderate_reversal_ready or moderate_hard_weakness_count < (1 if liquidation_map.swept_above else 2):
            return None

        triggers: list[str] = []
        if peak_still_fresh:
            triggers.append("РїРёРє РїР°РјРїР° СЃРѕРІСЃРµРј СЃРІРµР¶РёР№")
        if near_peak:
            triggers.append("С†РµРЅР° РµС‰С‘ Сѓ РІРµСЂС€РёРЅС‹ РїР°РјРїР°")
        if first_reaction:
            triggers.append("РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·")
        if bool(_as_float(layer2.get("near_high_context"), 0.0)):
            triggers.append("С†РµРЅР° РІСЃС‘ РµС‰С‘ Сѓ Р»РѕРєР°Р»СЊРЅРѕРіРѕ С…Р°СЏ")
        if bool(_as_float(layer2.get("obv_bearish_divergence"), 0.0)):
            triggers.append("OBV СѓР¶Рµ РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚")
        if bool(_as_float(layer2.get("cvd_bearish_divergence"), 0.0)):
            triggers.append("CVD СѓР¶Рµ РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚")
        if liquidation_map.swept_above:
            triggers.append("РІРµСЂС…РЅСЋСЋ Р»РёРєРІРёРґРЅРѕСЃС‚СЊ СѓР¶Рµ СЃРЅСЏР»Рё")
        if liquidation_map.downside_magnet:
            triggers.append("РЅРёР¶Рµ РµСЃС‚СЊ Р»РёРєРІРёРґР°С†РёРѕРЅРЅС‹Р№ РјР°РіРЅРёС‚")
        hard_weakness_count = 0
        if micro_reversal_near_peak:
            hard_weakness_count += 1
        if rsi < prev_rsi and hist < prev_hist:
            hard_weakness_count += 1
        if upper_wick / candle_range >= 0.28:
            hard_weakness_count += 1
        if obv_divergence:
            hard_weakness_count += 1
        if cvd_divergence:
            hard_weakness_count += 1
        if liquidation_map.swept_above:
            hard_weakness_count += 1
        if hard_reclaim_continuation:
            return None
        if continuation_after_pause:
            return None
        if live_peak_extension:
            return None
        if reacceleration_after_pullback:
            return None
        if continuation_structure_score >= 1.55 and hard_weakness_count < 3 and not liquidation_map.swept_above:
            return None
        setup_pullback_cap = max(early_reversal_pullback * 0.94, 0.0026 if close < 0.02 else 0.0032)
        setup_drawdown_cap = min(max(early_setup_max_drawdown_ratio * 0.76, 0.064), 0.096)
        if not first_reaction and not micro_reversal_near_peak and not liquidation_map.swept_above:
            return None
        if not peak_turn_confirmed and not liquidation_map.swept_above:
            return None
        if not terminal_rejection_bar and not liquidation_map.swept_above:
            return None
        if not real_rollover and not liquidation_map.swept_above:
            return None
        if not peak_followthrough_confirmed and not liquidation_map.swept_above:
            return None
        minimum_live_volume = 0.24 if close < 0.02 else 0.18
        if (
            not current_volume_supportive
            and not (volume_fade_confirmed and (real_rollover or peak_followthrough_confirmed))
            and not liquidation_map.swept_above
        ):
            return None
        if volume_spike < minimum_live_volume and not liquidation_map.swept_above:
            return None
        if volume_peak_age > 5 and not liquidation_map.swept_above:
            return None
        if not volume_climax_recent and hard_weakness_count < 4 and not liquidation_map.swept_above:
            return None
        if (
            pullback_from_peak_pct < minimum_reversal_pullback
            and not micro_reversal_near_peak
            and not liquidation_map.swept_above
        ):
            return None
        if peak_age_bars > early_setup_peak_age_cap and not liquidation_map.swept_above:
            return None
        if pump_drawdown_ratio > setup_drawdown_cap:
            return None
        if pump_range_position < early_setup_min_peak_position and not liquidation_map.swept_above:
            return None
        if close < max(pump_midpoint, ema20_last * 0.9998):
            return None
        if layer2_score < 0.68:
            return None
        if (trend_follow_through or bullish_continuation_bar) and not liquidation_map.swept_above:
            return None
        if reclaimed_peak and hard_weakness_count < 2 and not liquidation_map.swept_above:
            return None
        if peak_reclaim_without_reject and hard_weakness_count < 3 and not liquidation_map.swept_above:
            return None
        if (
            (not peak_still_fresh and not liquidation_map.swept_above)
            or (not near_peak and not liquidation_map.swept_above)
            or (
                pullback_from_peak_pct > setup_pullback_cap
            )
            or (layer2_score < 0.78 and hard_weakness_count < 2)
        ):
            return None
        if peak_age_bars > (2 if liquidation_map.swept_above else 2):
            return None
        if not structure_trigger_confirmed:
            return None
        if hard_weakness_count < (2 if liquidation_map.swept_above else 3):
            return None
        if (
            not volume_fade_confirmed
            and not obv_divergence
            and not cvd_divergence
            and not liquidation_map.swept_above
            and not (rsi < prev_rsi and hist < prev_hist)
        ):
            return None

        continuation_risk = max(0.0, 1.9 - layer2_score)
        continuation_risk += continuation_structure_score
        if liquidation_map.upside_risk > 0:
            continuation_risk += min(1.2, liquidation_map.upside_risk * 0.35)
        if trend_follow_through:
            continuation_risk += 1.15
        if reclaimed_peak:
            continuation_risk += 0.95
        if bullish_continuation_bar:
            continuation_risk += 0.55
        if reacceleration_after_pullback:
            continuation_risk += 1.10
        if not near_peak:
            continuation_risk += 0.4
        if not peak_still_fresh:
            continuation_risk += 0.45
        if pullback_from_peak_pct > max(early_reversal_pullback * 0.95, 0.0029):
            continuation_risk += 0.35
        if micro_reversal_near_peak:
            continuation_risk = max(0.0, continuation_risk - 0.55)
        if hard_weakness_count >= 3:
            continuation_risk = max(0.0, continuation_risk - 0.25)
        if strong_rejection_cluster:
            continuation_risk = max(0.0, continuation_risk - 0.18)
        if downside_confluence_score >= 1.00:
            continuation_risk = max(0.0, continuation_risk - 0.10)
        if continuation_risk >= (2.20 if hard_weakness_count >= 3 else 1.95):
            return None
        quality_score = min(
            10.0,
            max(
                0.0,
                5.2
                + layer2_score * 2.4
                - continuation_risk * 0.75
                + (0.60 if peak_still_fresh else 0.0)
                + (0.25 if peak_recent_enough else 0.0)
                + (0.45 if near_peak else 0.0)
                + (0.40 if first_reaction else 0.0)
                + (0.45 if micro_reversal_near_peak else 0.0)
                + (0.6 if liquidation_map.swept_above else 0.0)
                + (0.4 if liquidation_map.downside_magnet else 0.0),
            ),
        )
        if strong_rejection_cluster:
            quality_score += min(0.50, resistance_confluence_score * 0.16)
        if downside_confluence_score > 0.0:
            quality_score += min(0.25, downside_confluence_score * 0.08)
        quality_score = min(10.0, quality_score)
        if clean_pump_pct < early_structural_pump_min:
            quality_score -= 0.45
        if volume_spike < max(0.85, confirmed_volume_gate * 0.62):
            quality_score -= 0.55
        if hard_weakness_count < 3:
            quality_score -= 0.40
        setup_failure_score = 0.0
        if first_reaction:
            setup_failure_score += 0.80
        if micro_reversal_near_peak:
            setup_failure_score += 1.00
        if real_rollover:
            setup_failure_score += 1.05
        if peak_followthrough_confirmed:
            setup_failure_score += 1.15
        if volume_fade_confirmed:
            setup_failure_score += 0.60
        if obv_divergence:
            setup_failure_score += 0.60
        if cvd_divergence:
            setup_failure_score += 0.60
        if liquidation_map.swept_above:
            setup_failure_score += 0.75
        if clean_pump_pct < early_structural_pump_min:
            quality_score = min(quality_score, 7.1)
        if not volume_climax_recent:
            quality_score = min(quality_score, 7.0)
        if hard_weakness_count < 3:
            quality_score = min(quality_score, 6.9)
        if setup_failure_score < (2.35 if liquidation_map.swept_above else 2.80):
            return None
        if quality_score < max(6.35, early_quality_min + 1.15):
            return None
        if not downside_target_ready:
            return None

        return {
            "phase": "SETUP",
            "caption": build_early_signal_caption(
                symbol=symbol,
                timeframe=timeframe,
                mode=mode,
                phase_label="\\u0420\\u0410\\u041d\\u041d\\u0418\\u0419 \\u0428\\u041e\\u0420\\u0422: \\u0421\\u0415\\u0422\\u0410\\u041f",
                price=close,
                trace_meta=meta,
                watch_score=max(4.5, layer2_score * 6.0),
                watch_max_score=8.0,
                quality_score=quality_score,
                quality_max_score=10.0,
                quality_grade=_early_quality_grade(quality_score),
                continuation_risk=continuation_risk,
                continuation_max_score=4.0,
                triggers=triggers or ["РїР°РјРї СѓР¶Рµ РµСЃС‚СЊ", "СЃР»Р°Р±РѕСЃС‚СЊ СЂСЏРґРѕРј", "Р¶РґС‘Рј РІС…РѕРґ РїРѕ СЃС‚СЂР°С‚РµРіРёРё"],
                wait_for="подтверждение полноценного входа",
                enriched=enriched,
            ),
            "entry": close,
            "tp": synthetic_tp,
            "sl": synthetic_sl,
        }

    if failed_layer == "layer3_entry_location" and layer1_passed:
        layer2_score = _as_float(layer2.get("weakness_strength"), 0.0)
        location_reason = str(layer3.get("failed_reason") or "")
        setup_triggers: list[str] = []
        if peak_recent_enough:
            setup_triggers.append("РїРёРє РїР°РјРїР° РµС‰С‘ СЃРІРµР¶РёР№")
        if near_peak:
            setup_triggers.append("С†РµРЅР° РІСЃС‘ РµС‰С‘ СЂСЏРґРѕРј СЃ РІРµСЂС€РёРЅРѕР№")
        if first_reaction:
            setup_triggers.append("РїРѕСЃР»Рµ РїРёРєР° РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·")
        if peak_followthrough_confirmed:
            setup_triggers.append("РїРѕСЏРІРёР»СЃСЏ lower-high / lower-close РїРѕСЃР»Рµ РїРёРєР°")
        if liquidation_map.swept_above:
            setup_triggers.append("РІРµСЂС…РЅСЋСЋ Р»РёРєРІРёРґРЅРѕСЃС‚СЊ СѓР¶Рµ СЃРЅСЏР»Рё")
        if liquidation_map.downside_magnet:
            setup_triggers.append("РЅРёР¶Рµ РµСЃС‚СЊ Р»РёРєРІРёРґР°С†РёРѕРЅРЅС‹Р№ РјР°РіРЅРёС‚")
        if location_reason:
            setup_triggers.append("СЃРµС‚РєР° РїРѕС‡С‚Рё РіРѕС‚РѕРІР°, Р¶РґС‘Рј Р»СѓС‡С€СѓСЋ С‚РѕС‡РєСѓ РІС…РѕРґР°")

        setup_failure_score = 0.0
        if first_reaction:
            setup_failure_score += 0.80
        if micro_reversal_near_peak:
            setup_failure_score += 1.00
        if real_rollover:
            setup_failure_score += 1.05
        if peak_followthrough_confirmed:
            setup_failure_score += 1.15
        if volume_fade_confirmed:
            setup_failure_score += 0.55
        if liquidation_map.swept_above:
            setup_failure_score += 0.70

        setup_continuation_risk = continuation_structure_score
        if live_peak_extension:
            setup_continuation_risk += 1.15
        if trend_follow_through:
            setup_continuation_risk += 0.95
        if reclaimed_peak:
            setup_continuation_risk += 0.8
        if bullish_continuation_bar:
            setup_continuation_risk += 0.5
        if liquidation_map.upside_risk > 0:
            setup_continuation_risk += min(1.0, liquidation_map.upside_risk * 0.30)
        if peak_followthrough_confirmed:
            setup_continuation_risk = max(0.0, setup_continuation_risk - 0.55)
        if strong_rejection_cluster:
            setup_continuation_risk = max(0.0, setup_continuation_risk - 0.18)
        if downside_confluence_score >= 1.00:
            setup_continuation_risk = max(0.0, setup_continuation_risk - 0.10)

        setup_quality_score = min(
            10.0,
            max(
                0.0,
                6.1
                + min(max(layer2_score, 0.0), 1.0) * 2.3
                - setup_continuation_risk * 0.72
                + (0.45 if first_reaction else 0.0)
                + (0.55 if peak_followthrough_confirmed else 0.0)
                + (0.35 if near_peak else 0.0)
                + (0.55 if liquidation_map.swept_above else 0.0),
            ),
        )
        if strong_rejection_cluster:
            setup_quality_score += min(0.50, resistance_confluence_score * 0.16)
        if downside_confluence_score > 0.0:
            setup_quality_score += min(0.25, downside_confluence_score * 0.08)
        setup_quality_score = min(10.0, setup_quality_score)

        setup_structural_score = (
            float(layer2_failed_reclaim or layer3_failed_reclaim)
            + float(layer2_retest_failed_breakout or layer3_retest_failed_breakout)
            + float(peak_followthrough_confirmed)
            + float(liquidation_map.swept_above)
            + float(real_rollover)
        )

        if (
            clean_pump_pct >= early_pump_min
            and peak_recent_enough
            and near_peak
            and peak_age_bars <= 2
            and setup_failure_score >= (1.95 if liquidation_map.swept_above else 2.35)
            and (first_reaction or micro_reversal_near_peak or liquidation_map.swept_above)
            and (peak_followthrough_confirmed or real_rollover or liquidation_map.swept_above)
            and structural_reversal_ready
            and structure_trigger_confirmed
            and not acceptance_above_high
            and not hard_reclaim_continuation
            and not live_peak_extension
            and not reacceleration_after_pullback
            and setup_continuation_risk < 2.55
            and setup_structural_score >= (2.35 if liquidation_map.swept_above else 2.75)
            and setup_quality_score >= max(5.9, early_quality_min + 0.65)
        ):
            if not (layer2_failed_reclaim or layer3_failed_reclaim or liquidation_map.swept_above):
                setup_quality_score = min(setup_quality_score, 7.3)
            if not peak_followthrough_confirmed:
                setup_quality_score = min(setup_quality_score, 6.9)
            if not downside_target_ready:
                return None
            return {
                "phase": "SETUP",
                "caption": build_early_signal_caption(
                    symbol=symbol,
                    timeframe=timeframe,
                    mode=mode,
                    phase_label="\\u0420\\u0410\\u041d\\u041d\\u0418\\u0419 \\u0428\\u041e\\u0420\\u0422: \\u0421\\u0415\\u0422\\u0410\\u041f",
                    price=close,
                    trace_meta=meta,
                    watch_score=max(4.8, layer2_score * 6.2),
                    watch_max_score=8.0,
                    quality_score=setup_quality_score,
                    quality_max_score=10.0,
                    quality_grade=_early_quality_grade(setup_quality_score),
                    continuation_risk=setup_continuation_risk,
                    continuation_max_score=4.0,
                    triggers=setup_triggers or ["слабость уже есть", "сетка почти готова", "ждём вход по стратегии"],
                    wait_for="лучшую точку входа и подтверждение шорта",
                    enriched=enriched,
                ),
                "entry": close,
                "tp": synthetic_tp,
                "sl": synthetic_sl,
            }

    if failed_layer != "layer1_pump_detection":
        return None

    if clean_pump_pct < early_pump_min:
        return None

    if max(volume_spike, recent_volume_spike) < early_volume_gate:
        return None
    minimum_live_volume = 0.08 if close < 0.02 else 0.05
    if volume_peak_age > 4 and not liquidation_map.swept_above:
        return None
    if (
        max(volume_spike, recent_volume_spike) < max(0.18, early_volume_gate * 0.35)
        and not micro_reversal_near_peak
        and not liquidation_map.swept_above
    ):
        return None
    if (
        not volume_climax_recent
        and not volume_fade_confirmed
        and not liquidation_map.swept_above
        and not (clean_pump_pct >= early_structural_pump_min and peak_recent_enough and near_peak)
    ):
        return None
    if volume_spike < minimum_live_volume and not liquidation_map.swept_above:
        return None
    if max(rsi, prev_rsi) < early_rsi_min:
        return None
    prefire_structural_slowdown = (
        terminal_rejection_bar
        or micro_reversal_near_peak
        or volume_fade_confirmed
        or liquidation_map.swept_above
        or (first_reaction and (close < prev_close or hist < prev_hist or rsi < prev_rsi))
    )
    strong_peak_prefire = (
        peak_still_fresh
        and near_peak
        and clean_pump_pct >= early_structural_pump_min
        and rsi >= max(early_rsi_min, 56.0)
        and pump_range_position >= max(early_watch_min_peak_position, 0.72)
        and prefire_structural_slowdown
        and not reclaimed_peak
        and not bullish_continuation_bar
        and not hard_reclaim_continuation
    )
    if hard_reclaim_continuation:
        return None
    if continuation_after_pause and not strong_peak_prefire:
        return None
    if live_peak_extension and not strong_peak_prefire:
        return None
    if reacceleration_after_pullback and not strong_peak_prefire:
        return None
    if continuation_structure_score >= 1.85 and not liquidation_map.swept_above and not strong_peak_prefire:
        return None

    # Allow a true "prefire" watch near the peak before the setup path becomes active.
    # This is intentionally narrower than a generic soft-pass: we only use it when the
    # pump is fresh, close is still near the top, and we have at least an initial sign
    # of slowdown instead of a pure continuation bar.
    prefire_triggers: list[tuple[str, float]] = []
    if strong_peak_prefire:
        if peak_recent_enough:
            prefire_triggers.append(("РїРёРє РїР°РјРїР° СЃРѕРІСЃРµРј СЃРІРµР¶РёР№", 1.25))
        if near_peak:
            prefire_triggers.append(("С†РµРЅР° РµС‰Рµ Сѓ РІРµСЂС€РёРЅС‹ РїР°РјРїР°", 1.10))
        if clean_pump_pct >= early_pump_min:
            prefire_triggers.append(("С‡РёСЃС‚С‹Р№ РїР°РјРї СѓР¶Рµ РІС‹С€Рµ СЂР°РЅРЅРµРіРѕ РјРёРЅРёРјСѓРјР°", 0.80))
        if first_reaction:
            prefire_triggers.append(("РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·", 1.25))
        if micro_reversal_near_peak:
            prefire_triggers.append(("Сѓ РІРµСЂС€РёРЅС‹ РїРѕСЏРІРёР»Р°СЃСЊ РјРёРєСЂРѕСЂРµР°РєС†РёСЏ РІРЅРёР·", 1.15))
        if upper_wick / candle_range >= 0.18:
            prefire_triggers.append(("РµСЃС‚СЊ РІРµСЂС…РЅСЏСЏ С‚РµРЅСЊ", 0.95))
        if hist < prev_hist:
            prefire_triggers.append(("MACD РѕСЃР»Р°Р±РµРІР°РµС‚", 1.10))
        if recent_volume_spike > 0 and volume_spike < recent_volume_spike:
            prefire_triggers.append(("РѕР±СЉРµРј Р·Р°С‚СѓС…Р°РµС‚", 1.10))
        if liquidation_map.swept_above:
            prefire_triggers.append(("РІРµСЂС…РЅСЋСЋ Р»РёРєРІРёРґРЅРѕСЃС‚СЊ СѓР¶Рµ СЃРЅСЏР»Рё", 1.20))
        if liquidation_map.downside_magnet:
            prefire_triggers.append(("РЅРёР¶Рµ РµСЃС‚СЊ Р»РёРєРІРёРґР°С†РёРѕРЅРЅС‹Р№ РјР°РіРЅРёС‚", 1.00))

        prefire_unique_triggers: list[str] = []
        prefire_watch_score = 0.0
        for label, weight in prefire_triggers:
            if label in prefire_unique_triggers:
                continue
            prefire_unique_triggers.append(label)
            prefire_watch_score += float(weight)
        generic_prefire_penalty = 0.0
        if peak_recent_enough:
            generic_prefire_penalty += 0.70
        if near_peak:
            generic_prefire_penalty += 0.65
        if clean_pump_pct >= early_pump_min:
            generic_prefire_penalty += 0.45
        if liquidation_map.downside_magnet:
            generic_prefire_penalty += 0.35
        prefire_watch_score = max(0.0, prefire_watch_score - generic_prefire_penalty)

        prefire_reclaim_without_rollover = (
            (close >= recent_close_high * 0.9995 or close >= recent_high * 0.9975)
            and rsi >= prev_rsi
            and hist >= prev_hist
            and not liquidation_map.swept_above
        )
        prefire_slowdown = (
            terminal_rejection_bar
            or hist < prev_hist
            or rsi < prev_rsi
            or volume_fade_confirmed
            or liquidation_map.swept_above
            or ((first_reaction or micro_reversal_near_peak) and close < prev_close)
        )
        prefire_continuation_risk = 0.0
        if close >= signal_peak_reference * 0.9992:
            prefire_continuation_risk += 1.05
        if volume_spike >= max(recent_volume_spike * 0.97, early_volume_gate):
            prefire_continuation_risk += 0.95
        if rsi >= prev_rsi and rsi >= max(58.0, early_rsi_min + 5.0):
            prefire_continuation_risk += 0.80
        if bullish_continuation_bar:
            prefire_continuation_risk += 0.75
        if trend_follow_through:
            prefire_continuation_risk += 0.95
        if liquidation_map.upside_risk > 0:
            prefire_continuation_risk += min(1.20, liquidation_map.upside_risk * 0.35)
        if liquidation_map.swept_above:
            prefire_continuation_risk = max(0.0, prefire_continuation_risk - 0.65)
        if micro_reversal_near_peak:
            prefire_continuation_risk = max(0.0, prefire_continuation_risk - 0.50)
        if strong_rejection_cluster:
            prefire_continuation_risk = max(0.0, prefire_continuation_risk - 0.20)
        if downside_confluence_score >= 1.00:
            prefire_continuation_risk = max(0.0, prefire_continuation_risk - 0.10)

        prefire_failure_score = 0.0
        if first_reaction:
            prefire_failure_score += 0.85
        if micro_reversal_near_peak:
            prefire_failure_score += 1.05
        if real_rollover:
            prefire_failure_score += 1.05
        if peak_followthrough_confirmed:
            prefire_failure_score += 1.15
        if terminal_rejection_bar:
            prefire_failure_score += 0.70
        if volume_fade_confirmed:
            prefire_failure_score += 0.55
        if hist < prev_hist:
            prefire_failure_score += 0.40
        if rsi < prev_rsi:
            prefire_failure_score += 0.30
        if liquidation_map.swept_above:
            prefire_failure_score += 0.75

        prefire_quality_score = min(
            10.0,
            max(
                0.0,
                prefire_watch_score
                - prefire_continuation_risk * 0.55
                + (0.55 if peak_recent_enough else 0.0)
                + (0.60 if near_peak else 0.0)
                + (0.75 if first_reaction else 0.0)
                + (0.55 if micro_reversal_near_peak else 0.0)
                + (0.45 if clean_pump_pct >= early_structural_pump_min else 0.0),
            ),
        )
        if strong_rejection_cluster:
            prefire_quality_score += min(0.45, resistance_confluence_score * 0.14)
        if downside_confluence_score > 0.0:
            prefire_quality_score += min(0.20, downside_confluence_score * 0.07)
        prefire_quality_score = min(10.0, prefire_quality_score)
        if not (peak_followthrough_confirmed or liquidation_map.swept_above or real_rollover):
            prefire_quality_score = min(prefire_quality_score, 7.0)
        if not terminal_rejection_bar and not volume_fade_confirmed:
            prefire_quality_score = min(prefire_quality_score, 6.6)

        prefire_hard_reversal = (
            micro_reversal_near_peak
            or liquidation_map.swept_above
            or peak_followthrough_confirmed
            or real_rollover
            or (
                first_reaction
                and terminal_rejection_bar
                and (
                    hist < prev_hist
                    or rsi < prev_rsi
                    or volume_fade_confirmed
                )
            )
        )

        if (
            prefire_slowdown
            and prefire_hard_reversal
            and not prefire_reclaim_without_rollover
            and prefire_failure_score >= (1.35 if liquidation_map.swept_above else 1.55)
            and prefire_watch_score >= max(2.30, early_watch_score_min + 0.40)
            and len(prefire_unique_triggers) >= 4
            and prefire_continuation_risk < 2.05
            and prefire_quality_score >= max(4.25, early_quality_min + 1.55)
        ):
            return {
                "phase": "WATCH",
                "caption": build_early_signal_caption(
                    symbol=symbol,
                    timeframe=timeframe,
                    mode=mode,
                    phase_label="\\u0420\\u0410\\u041d\\u041d\\u0418\\u0419 \\u0428\\u041e\\u0420\\u0422: \\u041d\\u0410\\u0411\\u041b\\u042e\\u0414\\u0415\\u041d\\u0418\\u0415",
                    price=close,
                    trace_meta=meta,
                    watch_score=prefire_watch_score,
                    watch_max_score=10.0,
                    quality_score=prefire_quality_score,
                    quality_max_score=10.0,
                    quality_grade=_early_quality_grade(prefire_quality_score),
                    continuation_risk=prefire_continuation_risk,
                    continuation_max_score=5.75,
                    triggers=prefire_unique_triggers,
                    wait_for="первую нормальную реакцию вниз и подтверждение слабости",
                    enriched=enriched,
                ),
                "entry": close,
                "tp": synthetic_tp,
                "sl": synthetic_sl,
            }

    watch_pullback_cap = min(
        max(
            peak_pullback_limit * 1.02,
            early_reversal_pullback * 1.14,
            0.0082 if close < 0.02 else 0.0058,
        ),
        0.018 if close < 0.02 else 0.014,
    )
    watch_drawdown_cap = min(max(early_watch_max_drawdown_ratio * 0.72, 0.052), 0.066)
    recent_watch_pullback_allowance = min(
        watch_pullback_cap,
        max(
            peak_pullback_limit * 0.92,
            early_reversal_pullback * 1.06,
            0.0074 if close < 0.02 else 0.0052,
        ),
    )
    if (
        not peak_still_fresh
        and not liquidation_map.swept_above
        and not (
            peak_recent_enough
            and first_reaction
            and pullback_from_peak_pct <= recent_watch_pullback_allowance
            and pump_drawdown_ratio <= watch_drawdown_cap * 0.92
        )
    ):
        return None
    if peak_age_bars > early_watch_peak_age_cap:
        return None
    if (
        not peak_turn_confirmed
        and not liquidation_map.swept_above
        and not micro_reversal_near_peak
        and not first_reaction
        and not strong_peak_prefire
    ):
        return None
    if (
        not terminal_rejection_bar
        and not liquidation_map.swept_above
        and not micro_reversal_near_peak
        and not first_reaction
        and not strong_peak_prefire
    ):
        return None
    if (
        not real_rollover
        and not liquidation_map.swept_above
        and not micro_reversal_near_peak
        and not (
            first_reaction
            and (
                rsi < prev_rsi
                or hist < prev_hist
                or upper_wick / candle_range >= 0.20
            )
        )
        and not strong_peak_prefire
    ):
        return None
    if (
        not peak_followthrough_confirmed
        and not liquidation_map.swept_above
        and not first_reaction
        and not strong_peak_prefire
    ):
        return None
    if (
        not current_volume_supportive
        and not (volume_fade_confirmed and (real_rollover or peak_followthrough_confirmed))
        and not liquidation_map.swept_above
        and not (clean_pump_pct >= early_structural_pump_min and near_peak and first_reaction)
        and not strong_peak_prefire
    ):
        return None
    if (
        pullback_from_peak_pct < minimum_reversal_pullback
        and not micro_reversal_near_peak
        and not liquidation_map.swept_above
        and not strong_peak_prefire
    ):
        return None
    if pump_drawdown_ratio > watch_drawdown_cap:
        return None
    if pump_range_position < early_watch_min_peak_position and not liquidation_map.swept_above:
        return None
    if close < max(pump_midpoint + atr * 0.12, ema20_last * 1.0004):
        return None
    if (not near_peak and not liquidation_map.swept_above) or pullback_from_peak_pct > watch_pullback_cap:
        return None

    continuation_risk = 0.0
    continuation_risk += continuation_structure_score
    if close >= signal_peak_reference * 0.9993:
        continuation_risk += 1.45
    if volume_spike >= max(recent_volume_spike * 0.97, early_volume_gate):
        continuation_risk += 1.10
    if rsi >= prev_rsi and rsi >= max(58.0, early_rsi_min + 4.0):
        continuation_risk += 1.0
    if hist >= prev_hist and hist > 0:
        continuation_risk += 1.0
    if close >= prev_close:
        continuation_risk += 0.5
    if upper_wick / candle_range < 0.16:
        continuation_risk += 0.75
    if liquidation_map.upside_risk > 0:
        continuation_risk += min(1.75, liquidation_map.upside_risk * 0.55)
    if open_upside_liq_magnet:
        continuation_risk += 0.75
    if (
        liquidation_map.nearest_above_distance_pct is not None
        and liquidation_map.nearest_above_distance_pct < 0.0045
        and liquidation_map.upside_risk >= 1.8
        and not liquidation_map.swept_above
    ):
        continuation_risk += 0.75
    if trend_follow_through:
        continuation_risk += 1.15
    if reclaimed_peak:
        continuation_risk += 0.95
    if bullish_continuation_bar:
        continuation_risk += 0.65
    if reacceleration_after_pullback:
        continuation_risk += 1.15
    if strong_peak_prefire:
        continuation_risk = max(0.0, continuation_risk - 0.55)
    if upper_rejection_zone_ready:
        continuation_risk = max(0.0, continuation_risk - 0.30)
    if liquidation_map.swept_above:
        continuation_risk = max(0.0, continuation_risk - 0.30)
    if downside_target_ready:
        continuation_risk = max(0.0, continuation_risk - 0.18)

    active_failure_score = 0.0
    if first_reaction:
        active_failure_score += 0.85
    if micro_reversal_near_peak:
        active_failure_score += 1.05
    if real_rollover:
        active_failure_score += 1.10
    if peak_followthrough_confirmed:
        active_failure_score += 1.15
    if terminal_rejection_bar:
        active_failure_score += 0.70
    if volume_fade_confirmed:
        active_failure_score += 0.55
    if hist < prev_hist:
        active_failure_score += 0.40
    if rsi < prev_rsi:
        active_failure_score += 0.30
    if liquidation_map.swept_above:
        active_failure_score += 0.80

    still_accelerating = (
        close >= signal_peak_reference * 0.999
        and (
            volume_spike >= max(recent_volume_spike * 0.95, early_volume_gate)
            or trend_follow_through
            or bullish_continuation_bar
        )
        and rsi >= prev_rsi
        and hist >= prev_hist
    )
    if not first_reaction and not micro_reversal_near_peak and not liquidation_map.swept_above and not strong_peak_prefire and not soft_regime_watch:
        return None
    if pullback_from_peak_pct > watch_pullback_cap:
        return None
    if acceptance_above_high and not liquidation_map.swept_above:
        return None
    if micro_reversal_near_peak:
        continuation_risk = max(0.0, continuation_risk - 1.15)
    if reclaimed_peak and not liquidation_map.swept_above and not micro_reversal_near_peak:
        return None
    if continuation_after_pause and not strong_peak_prefire:
        return None
    if live_peak_extension and not strong_peak_prefire:
        return None
    if continuation_structure_score >= 1.95 and not liquidation_map.swept_above and not micro_reversal_near_peak and not strong_peak_prefire:
        return None
    if (still_accelerating and not micro_reversal_near_peak and not first_reaction and not strong_peak_prefire) or continuation_risk >= (
        4.10 if strong_peak_prefire else 3.45 if micro_reversal_near_peak else 3.10
    ):
        return None
    if active_failure_score < (
        1.20 if soft_regime_watch else 1.35 if strong_peak_prefire else 1.65 if liquidation_map.swept_above else 2.00
    ):
        return None
    if not peak_followthrough_confirmed and not liquidation_map.swept_above and not first_reaction and not strong_peak_prefire and not soft_regime_watch:
        return None

    weighted_triggers: list[tuple[str, float]] = []
    if near_peak:
        weighted_triggers.append(("цена ещё у вершины пампа", 1.15))
    if first_reaction:
        weighted_triggers.append(("пошла первая реакция вниз", 1.20))
    if clean_pump_pct >= early_pump_min:
        weighted_triggers.append(("чистый памп уже выше раннего минимума", 0.75))
    if clean_pump_pct >= early_structural_pump_min:
        weighted_triggers.append(("чистый памп уже дотянул до confirmed-порога", 1.0))
    if close >= max(bb_upper, kc_upper) * 0.998:
        weighted_triggers.append(("цена у верхней зоны", 1.0))
    if close >= signal_peak_reference * 0.995:
        weighted_triggers.append(("цена у локального хая", 1.0))
    if near_vah_resistance:
        weighted_triggers.append(("цена упёрлась в VAH", 1.35))
    if rsi >= early_rsi_min:
        weighted_triggers.append(("RSI уже повышен", 0.5))
    if rsi < prev_rsi and rsi >= early_rsi_min:
        weighted_triggers.append(("RSI разворачивается вниз", 1.25))
    if volume_spike >= early_volume_gate:
        weighted_triggers.append(("объём ещё повышен", 0.5))
    if recent_volume_spike > 0 and volume_spike < recent_volume_spike:
        weighted_triggers.append(("объём затухает", 1.25))
    if upper_wick / candle_range >= 0.35:
        weighted_triggers.append(("есть верхняя тень", 1.0))
    if hist < prev_hist:
        weighted_triggers.append(("MACD ослабевает", 1.25))
    if close <= recent_close_high * 0.9995:
        weighted_triggers.append(("цена перестала ускоряться", 0.75))
    if obv <= max(recent_obv_ref * 0.998, prev_obv):
        weighted_triggers.append(("OBV не подтверждает рост", 1.25))
    if cvd <= max(recent_cvd_ref * 0.998, prev_cvd):
        weighted_triggers.append(("CVD не подтверждает рост", 1.25))
    if liquidation_map.swept_above:
        weighted_triggers.append(("верхнюю ликвидность уже сняли", 1.35))
    if liquidation_map.downside_magnet:
        weighted_triggers.append(("ниже есть ликвидационный магнит", 1.15))
    if value_room_to_poc:
        weighted_triggers.append(("ниже есть POC как цель отката", 0.95))
    if value_room_to_val:
        weighted_triggers.append(("ниже есть VAL как цель отката", 1.00))

    if peak_followthrough_confirmed:
        weighted_triggers.append(("после пика уже появился lower-high / lower-close", 1.45))

    normalized_weighted_triggers: list[tuple[str, float]] = []
    for label, weight in weighted_triggers:
        clean_label = _normalize_human_text(label)
        if clean_label:
            normalized_weighted_triggers.append((clean_label, float(weight)))

    unique_triggers: list[str] = []
    obv_divergence = obv <= max(recent_obv_ref * 0.998, prev_obv)
    cvd_divergence = cvd <= max(recent_cvd_ref * 0.998, prev_cvd)
    watch_score = 0.0
    for label, weight in normalized_weighted_triggers:
        if label in unique_triggers:
            continue
        unique_triggers.append(label)
        watch_score += float(weight)
    generic_watch_penalty = 0.0
    if near_peak:
        generic_watch_penalty += 0.60
    if clean_pump_pct >= early_pump_min:
        generic_watch_penalty += 0.40
    if clean_pump_pct >= early_structural_pump_min:
        generic_watch_penalty += 0.45
    if close >= max(bb_upper, kc_upper) * 0.998:
        generic_watch_penalty += 0.35
    if close >= signal_peak_reference * 0.995:
        generic_watch_penalty += 0.35
    if rsi >= early_rsi_min:
        generic_watch_penalty += 0.20
    if volume_spike >= early_volume_gate:
        generic_watch_penalty += 0.20
    watch_score = max(0.0, watch_score - generic_watch_penalty)

    active_weakness_score = 0.0
    if terminal_rejection_bar:
        active_weakness_score += 1.35
    if real_rollover:
        active_weakness_score += 1.70
    if first_reaction:
        active_weakness_score += 1.00
    if rsi < prev_rsi:
        active_weakness_score += 0.75
    if hist < prev_hist:
        active_weakness_score += 0.80
    if volume_fade_confirmed:
        active_weakness_score += 0.80
    if obv_divergence:
        active_weakness_score += 0.70
    if cvd_divergence:
        active_weakness_score += 0.70
    if liquidation_map.swept_above:
        active_weakness_score += 0.95
    if liquidation_map.downside_magnet:
        active_weakness_score += 0.65
    if continuation_after_pause:
        active_weakness_score = max(0.0, active_weakness_score - 1.25)
    weak_anchor_without_level = bool(
        recent_upper_anchor_touch
        and not (
            near_vah_resistance
            or recent_upper_liq_reaction
            or liquidation_map.swept_above
        )
    )
    watch_live_volume_floor = max(
        0.16 if close < 0.02 else 0.10,
        early_volume_gate * 0.52,
    )
    if near_vah_resistance or recent_upper_liq_reaction or liquidation_map.swept_above:
        watch_live_volume_floor = min(
            watch_live_volume_floor,
            max(
                0.12 if close < 0.02 else 0.08,
                early_volume_gate * 0.42,
            ),
        )
    if (
        volume_spike < watch_live_volume_floor
        and not (
            liquidation_map.swept_above
            or peak_followthrough_confirmed
            or structure_trigger_confirmed
            or recent_upper_liq_reaction
        )
    ):
        return None
    if weak_anchor_without_level:
        if not (peak_followthrough_confirmed or structure_trigger_confirmed or micro_reversal_near_peak):
            return None
        if active_weakness_score < 2.95 and not strong_peak_prefire and not soft_regime_fail:
            return None
    weak_rejection_floor = max(
        minimum_reversal_pullback * 1.12,
        0.0017 if close < 0.02 else 0.0011,
    )
    if weak_watch_structure:
        if (
            not recent_upper_anchor_reject
            and not strong_upper_reversal_bar
            and not peak_followthrough_confirmed
            and not liquidation_map.swept_above
            and not strong_peak_prefire
            and not soft_regime_fail
        ):
            return None
        if (
            pullback_from_peak_pct < weak_rejection_floor
            and not micro_reversal_near_peak
            and not strong_peak_prefire
            and not soft_regime_fail
        ):
            return None
        if active_weakness_score < 2.40 and not strong_peak_prefire and not soft_regime_fail:
            return None
        if (
            continuation_structure_score > 1.35
            and not micro_reversal_near_peak
            and not strong_peak_prefire
            and not soft_regime_fail
        ):
            return None
        if (
            not volume_climax_recent
            and recent_volume_spike < max(0.22, early_volume_gate * 2.10)
            and not liquidation_map.swept_above
            and not soft_regime_fail
        ):
            return None
        if (
            volume_spike >= max(0.95, recent_volume_spike * 0.96, early_volume_gate * 3.0)
            and not volume_fade_confirmed
            and not peak_followthrough_confirmed
            and not liquidation_map.swept_above
            and not strong_peak_prefire
            and not soft_regime_fail
        ):
            return None
    if weak_watch_structure and (reported_pump_below_min or marginal_watch_pump):
        if not (
            recent_upper_liq_reaction
            or micro_reversal_near_peak
            or strong_peak_prefire
            or soft_regime_fail
        ):
            return None
        if downside_confluence_score < 1.05 and not liquidation_map.downside_magnet:
            return None
        if active_weakness_score < 2.65:
            return None

    weakness_markers = {
        "\u0052\u0053\u0049 \u0440\u0430\u0437\u0432\u043e\u0440\u0430\u0447\u0438\u0432\u0430\u0435\u0442\u0441\u044f \u0432\u043d\u0438\u0437",
        "\u043e\u0431\u044a\u0451\u043c \u0437\u0430\u0442\u0443\u0445\u0430\u0435\u0442",
        "\u004d\u0041\u0043\u0044 \u043e\u0441\u043b\u0430\u0431\u0435\u0432\u0430\u0435\u0442",
        "\u004f\u0042\u0056 \u043d\u0435 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0430\u0435\u0442 \u0440\u043e\u0441\u0442",
        "\u0043\u0056\u0044 \u043d\u0435 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0430\u0435\u0442 \u0440\u043e\u0441\u0442",
        "\u0446\u0435\u043d\u0430 \u043f\u0435\u0440\u0435\u0441\u0442\u0430\u043b\u0430 \u0443\u0441\u043a\u043e\u0440\u044f\u0442\u044c\u0441\u044f",
        "\u0432\u0435\u0440\u0445\u043d\u044e\u044e \u043b\u0438\u043a\u0432\u0438\u0434\u043d\u043e\u0441\u0442\u044c \u0443\u0436\u0435 \u0441\u043d\u044f\u043b\u0438",
        "\u043d\u0438\u0436\u0435 \u0435\u0441\u0442\u044c \u043b\u0438\u043a\u0432\u0438\u0434\u0430\u0446\u0438\u043e\u043d\u043d\u044b\u0439 \u043c\u0430\u0433\u043d\u0438\u0442",
        "\u043f\u043e\u0441\u043b\u0435 \u043f\u0438\u043a\u0430 \u0443\u0436\u0435 \u043f\u043e\u044f\u0432\u0438\u043b\u0441\u044f \u006c\u006f\u0077\u0065\u0072\u002d\u0068\u0069\u0067\u0068 \u002f \u006c\u006f\u0077\u0065\u0072\u002d\u0063\u006c\u006f\u0073\u0065",
        "\u043f\u043e\u0448\u043b\u0430 \u043f\u0435\u0440\u0432\u0430\u044f \u0440\u0435\u0430\u043a\u0446\u0438\u044f \u0432\u043d\u0438\u0437",
    }
    normalized_triggers = {_normalize_human_text(trigger) for trigger in unique_triggers}
    weakness_count = sum(1 for trigger in normalized_triggers if trigger in weakness_markers)
    peak_rollover_fast_track = (
        peak_recent_enough
        and near_peak
        and first_reaction
        and terminal_rejection_bar
        and pullback_from_peak_pct <= max(early_reversal_pullback * 1.18, 0.0046)
        and pump_drawdown_ratio <= min(early_watch_max_drawdown_ratio, 0.072)
        and pump_range_position >= early_watch_min_peak_position
        and (close < prev_close or rsi < prev_rsi or hist < prev_hist)
    )
    if peak_followthrough_confirmed:
        peak_rollover_fast_track = True
    if strong_peak_prefire:
        continuation_risk = max(0.0, continuation_risk - 0.40)
    if peak_rollover_fast_track:
        continuation_risk = max(0.0, continuation_risk - 0.70)
    effective_watch_score_min = early_watch_score_min - (0.55 if peak_rollover_fast_track else 0.0)
    effective_quality_min = early_quality_min - (0.28 if peak_rollover_fast_track else 0.0)
    if strong_peak_prefire:
        effective_watch_score_min = min(effective_watch_score_min, early_watch_score_min - 0.35)
        effective_quality_min = min(effective_quality_min, early_quality_min - 0.20)
    required_triggers = (
        3
        if peak_rollover_fast_track
        or strong_peak_prefire
        or (near_peak and (micro_reversal_near_peak or first_reaction))
        else 3
        if near_peak
        else 4
    )
    if soft_regime_watch:
        required_triggers = max(3, required_triggers - 1)
    if watch_score < effective_watch_score_min or len(unique_triggers) < required_triggers:
        return None
    if weakness_count < 2 and not liquidation_map.swept_above and not micro_reversal_near_peak and not strong_peak_prefire:
        return None
    active_weakness_min = 0.75 if soft_regime_watch else 1.10 if strong_peak_prefire else 1.55 if peak_rollover_fast_track else 1.65 if near_peak else 2.10
    if active_weakness_score < active_weakness_min and not liquidation_map.swept_above:
        return None
    if (
        not structural_reversal_ready
        and not liquidation_map.swept_above
        and not peak_followthrough_confirmed
        and not (
            first_reaction
            and terminal_rejection_bar
            and (
                rsi < prev_rsi
                or hist < prev_hist
                or volume_fade_confirmed
            )
        )
    ):
        return None
    if (
        not real_rollover
        and not liquidation_map.swept_above
        and not micro_reversal_near_peak
        and not strong_peak_prefire
        and not soft_regime_watch
        and not (
            first_reaction
            and (
                peak_turn_confirmed
                or rsi < prev_rsi
                or hist < prev_hist
                or upper_wick / candle_range >= 0.16
                or clean_pump_pct >= early_structural_pump_min * 1.05
            )
        )
    ):
        return None
    if ((weakness_count == 0 and not micro_reversal_near_peak and not strong_peak_prefire and not soft_regime_watch)
        or (not first_reaction and not micro_reversal_near_peak and not strong_peak_prefire and not soft_regime_watch)):
        return None

    quality_score = min(
        10.0,
        max(
            0.0,
            watch_score
            - continuation_risk * 0.65
            + active_weakness_score * 0.52
            + (0.72 if peak_still_fresh else 0.0)
            + (0.28 if peak_recent_enough else 0.0)
            + (0.55 if near_peak else 0.0)
            + (0.65 if first_reaction else 0.0)
            + (0.55 if micro_reversal_near_peak else 0.0)
            + (0.75 if clean_pump_pct >= early_structural_pump_min else 0.0)
            + (0.40 if volume_spike >= confirmed_volume_gate else 0.0),
        ),
    )
    if clean_pump_pct < early_structural_pump_min:
        quality_score -= 0.55
    if volume_spike < max(0.95, confirmed_volume_gate * 0.72):
        quality_score -= 0.65
    if weakness_count < 3:
        quality_score -= 0.55
    if clean_pump_pct < early_structural_pump_min:
        quality_score = min(quality_score, 7.8)
    if not volume_climax_recent:
        quality_score = min(quality_score, 7.5)
    if weakness_count < 3:
        quality_score = min(quality_score, 7.5)
    if not (layer2_failed_reclaim or layer3_failed_reclaim or liquidation_map.swept_above or peak_followthrough_confirmed):
        quality_score = min(quality_score, 7.0)
    if peak_rollover_fast_track:
        quality_score = max(quality_score, early_quality_min + 0.15)
    if active_weakness_score < 2.4:
        quality_score = min(quality_score, 6.9)
    if not structural_reversal_ready:
        quality_score = min(quality_score, 6.8 if strong_peak_prefire else 6.4)
    if not (layer2_failed_reclaim or layer3_failed_reclaim or liquidation_map.swept_above) and active_weakness_score < 2.8:
        quality_score = min(quality_score, 6.6)
    if weak_anchor_without_level and not (peak_followthrough_confirmed or liquidation_map.swept_above):
        quality_score = min(quality_score, 7.15)
    if weak_watch_structure and not (recent_upper_anchor_reject or liquidation_map.swept_above):
        quality_score = min(quality_score, 6.75 if strong_upper_reversal_bar else 6.45)
    if (
        phase == "WATCH"
        and not (
            recent_upper_liq_reaction
            or liquidation_map.swept_above
            or near_vah_resistance
            or peak_followthrough_confirmed
        )
    ):
        quality_score = min(quality_score, 7.25)
    if quality_score < effective_quality_min:
        return None

    return {
        "phase": "WATCH",
        "caption": build_early_signal_caption(
            symbol=symbol,
            timeframe=timeframe,
            mode=mode,
            phase_label="\\u0420\\u0410\\u041d\\u041d\\u0418\\u0419 \\u0428\\u041e\\u0420\\u0422: \\u041d\\u0410\\u0411\\u041b\\u042e\\u0414\\u0415\\u041d\\u0418\\u0415",
            price=close,
            trace_meta=meta,
            watch_score=watch_score,
            watch_max_score=10.0,
            quality_score=quality_score,
            quality_max_score=10.0,
            quality_grade=_early_quality_grade(quality_score),
            continuation_risk=continuation_risk,
            continuation_max_score=5.75,
            triggers=unique_triggers,
            wait_for="подтверждение слабости и входа",
            enriched=enriched,
        ),
        "entry": close,
        "tp": synthetic_tp,
        "sl": synthetic_sl,
    }


def _build_approved_early_candidate(*, symbol: str, timeframe: str, mode: str, enriched, intent) -> dict[str, object] | None:
    if getattr(intent, "action", None) != IntentAction.SHORT_ENTRY:
        return None

    meta = intent.metadata if isinstance(intent.metadata, Mapping) else {}
    trace = meta.get("layer_trace", {}) if isinstance(meta, Mapping) else {}
    layers = trace.get("layers", {}) if isinstance(trace, Mapping) else {}
    if not isinstance(layers, Mapping):
        return None

    layer1 = _extract_layer_details(meta, "layer1_pump_detection")
    layer2 = _extract_layer_details(meta, "layer2_weakness_confirmation")
    layer3 = _extract_layer_details(meta, "layer3_entry_location")
    if enriched is None or getattr(enriched, "empty", True):
        return None

    row = enriched.iloc[-1]
    prev = enriched.iloc[-2] if len(enriched) > 1 else row
    close = _as_float(row.get("close"), 0.0)
    if close <= 0.0:
        return None

    liquidation_map = build_liquidation_map(enriched)
    atr = max(_as_float(row.get("atr"), close * 0.01), close * 0.001, 1e-8)
    local_vah = _as_float(row.get("vah"), 0.0)
    local_poc = _as_float(row.get("poc"), 0.0)
    local_val = _as_float(row.get("val"), 0.0)
    recent_high_series = enriched.tail(min(len(enriched), 28 if close >= 0.02 else 36))["high"] if "high" in enriched.columns else pd.Series([close])
    recent_low_series = enriched.tail(min(len(enriched), 28 if close >= 0.02 else 36))["low"] if "low" in enriched.columns else pd.Series([close])
    recent_high = _as_float(pd.to_numeric(recent_high_series, errors="coerce").max(), close)
    recent_low = _as_float(pd.to_numeric(recent_low_series, errors="coerce").min(), close)
    if recent_high <= 0.0 or recent_low <= 0.0 or recent_high <= recent_low:
        return None

    pump_min = max(_as_float(layer1.get("clean_pump_min_pct_used"), 0.05), 0.0)
    derived_pump_pct = max(0.0, (recent_high - recent_low) / max(recent_low, 1e-8))
    clean_pump_pct = max(_as_float(layer1.get("clean_pump_pct"), 0.0), derived_pump_pct)
    volume_spike = max(_as_float(layer1.get("volume_spike"), 0.0), _as_float(row.get("volume_spike"), 0.0))
    recent_volume_spike_series = enriched.tail(8)["volume_spike"] if "volume_spike" in enriched.columns else pd.Series([volume_spike])
    recent_volume_spike = _as_float(pd.to_numeric(recent_volume_spike_series, errors="coerce").max(), volume_spike)
    rsi = max(_as_float(layer1.get("rsi"), 0.0), _as_float(row.get("rsi"), 50.0))
    prev_rsi = _as_float(prev.get("rsi"), rsi)
    hist = _as_float(row.get("hist"), 0.0)
    prev_hist = _as_float(prev.get("hist"), hist)
    mtf_trend_5m = _as_float(row.get("mtf_trend_5m"), 0.0)
    mtf_rsi_5m = _as_float(row.get("mtf_rsi_5m"), 50.0)
    mtf_trend_15m = _as_float(row.get("mtf_trend_15m"), 0.0)
    mtf_rsi_15m = _as_float(row.get("mtf_rsi_15m"), 50.0)
    mtf_trend_1h = _as_float(row.get("mtf_trend_1h"), 0.0)
    mtf_rsi_1h = _as_float(row.get("mtf_rsi_1h"), 50.0)
    prev_close = _as_float(prev.get("close"), close)
    open_px = _as_float(row.get("open"), close)
    high_px = max(_as_float(row.get("high"), close), close, open_px)
    low_px = min(_as_float(row.get("low"), close), close, open_px)
    candle_range = max(high_px - low_px, close * 0.0004, 1e-8)
    upper_wick = max(high_px - max(open_px, close), 0.0)
    upper_wick_share = upper_wick / candle_range
    close_position_in_candle = max(0.0, min(1.0, (close - low_px) / candle_range))
    recent_close_tail = (
        pd.to_numeric(enriched.tail(4)["close"], errors="coerce").dropna()
        if "close" in enriched.columns
        else pd.Series([close])
    )
    recent_close_high = _as_float(recent_close_tail.max() if not recent_close_tail.empty else close, close)
    up_closes_last3 = 0
    if len(recent_close_tail) >= 2:
        recent_close_values = recent_close_tail.to_numpy(dtype=float)
        up_closes_last3 = int(
            sum(curr > prev_close_value for prev_close_value, curr in zip(recent_close_values[:-1], recent_close_values[1:]))
        )

    failed_reclaim = bool(_as_float(layer2.get("failed_reclaim"), 0.0) or _as_float(layer3.get("failed_reclaim"), 0.0))
    retest_failed_breakout = bool(
        _as_float(layer2.get("retest_failed_breakout"), 0.0)
        or _as_float(layer3.get("retest_failed_breakout"), 0.0)
    )
    acceptance_above_high = bool(
        _as_float(layer2.get("acceptance_above_swing_high"), 0.0)
        or _as_float(layer3.get("acceptance_above_swing_high"), 0.0)
    )
    if acceptance_above_high:
        return None

    weakness_strength = max(_as_float(layer2.get("weakness_strength"), 0.0), 0.0)
    distance_from_high_pct = max(0.0, (recent_high - close) / max(recent_high, 1e-8))
    range_position = min(1.0, max(0.0, (close - recent_low) / max(recent_high - recent_low, 1e-8)))
    atr_pct = atr / max(close, 1e-8)
    minimum_reversal_pullback = max(
        atr_pct * 0.68,
        0.0014 if close < 0.02 else 0.0009,
    )
    price_rejection_near_high = bool(
        (
            close < prev_close
            and close_position_in_candle <= 0.72
        )
        or (
            close < open_px
            and close_position_in_candle <= 0.70
        )
        or (
            upper_wick_share >= 0.24
            and close_position_in_candle <= 0.72
            and close <= prev_close * (1.0 + (0.0004 if close < 0.02 else 0.00025))
        )
        or (
            distance_from_high_pct >= minimum_reversal_pullback
            and close_position_in_candle <= 0.64
        )
    )
    momentum_rollover_count = int(hist < prev_hist) + int(rsi < prev_rsi)
    momentum_rollover = momentum_rollover_count >= 1
    volume_fade = recent_volume_spike > 0.0 and volume_spike <= recent_volume_spike * 0.92
    if volume_fade:
        momentum_rollover_count += 1
    first_reaction = bool(
        failed_reclaim
        or retest_failed_breakout
        or (
            price_rejection_near_high
            and momentum_rollover_count >= 2
            and distance_from_high_pct >= minimum_reversal_pullback * 0.70
        )
    )

    if clean_pump_pct < max(pump_min, 0.04):
        return None
    if distance_from_high_pct > (0.0175 if close < 0.02 else 0.0125):
        return None
    if distance_from_high_pct < (0.0014 if close < 0.02 else 0.0010) and not (failed_reclaim or retest_failed_breakout or volume_fade):
        return None
    if range_position < 0.62:
        return None
    if rsi < 56.5:
        return None
    if max(volume_spike, recent_volume_spike) < (0.12 if close < 0.02 else 0.095):
        return None
    has_structural_failure = failed_reclaim or retest_failed_breakout
    vah_reject_tolerance = max(
        atr_pct * 1.15,
        0.0048 if close < 0.02 else 0.0030,
    )
    peak_zone_tolerance = max(
        vah_reject_tolerance * 1.18,
        0.0062 if close < 0.02 else 0.0039,
    )
    value_room_floor = max(
        atr_pct * 0.72,
        0.0048 if close < 0.02 else 0.0028,
    )
    near_vah_resistance = bool(
        local_vah > 0
        and abs(close - local_vah) / max(close, 1e-8) <= vah_reject_tolerance
    )
    near_peak_resistance = bool(
        distance_from_high_pct <= max(vah_reject_tolerance * 1.20, 0.0058 if close < 0.02 else 0.0038)
    )
    value_room_to_poc = bool(
        local_poc > 0
        and (close - local_poc) / max(close, 1e-8) >= value_room_floor * 0.78
    )
    value_room_to_val = bool(
        local_val > 0
        and (close - local_val) / max(close, 1e-8) >= value_room_floor
    )
    value_room_to_recent_low = bool(
        recent_low > 0
        and (close - recent_low) / max(close, 1e-8) >= value_room_floor * 1.05
    )
    recent_upper_liq_reaction = any(
        band.side == "above"
        and band.closed_index is not None
        and band.closed_index >= max(len(enriched) - 8, 0)
        and (
            abs(band.level - recent_high) / max(recent_high, 1e-8) <= peak_zone_tolerance * 1.35
            or abs(band.level - close) / max(close, 1e-8) <= peak_zone_tolerance * 1.05
        )
        for band in liquidation_map.bands
    )
    lower_liq_target_ready = any(
        band.side == "below"
        and band.closed_index is None
        and band.weight >= 1.05
        and 0.003 <= (close - band.level) / max(close, 1e-8) <= max(value_room_floor * 3.20, 0.035)
        for band in liquidation_map.bands
    )
    resistance_confluence_score = 0.0
    if near_vah_resistance:
        resistance_confluence_score += 1.00
    if near_peak_resistance:
        resistance_confluence_score += 0.95
    if recent_upper_liq_reaction:
        resistance_confluence_score += 0.90
    if liquidation_map.swept_above:
        resistance_confluence_score += 1.15
    if has_structural_failure:
        resistance_confluence_score += 1.05
    downside_confluence_score = 0.0
    if liquidation_map.downside_magnet:
        downside_confluence_score += 1.15
    if lower_liq_target_ready:
        downside_confluence_score += 0.80
    if value_room_to_poc:
        downside_confluence_score += 0.75
    if value_room_to_val:
        downside_confluence_score += 0.85
    if value_room_to_recent_low:
        downside_confluence_score += 0.55
    recent_high_tail = (
        pd.to_numeric(enriched.tail(4)["high"], errors="coerce").dropna()
        if "high" in enriched.columns
        else pd.Series([recent_high])
    )
    anchor_touch_tolerance = max(
        peak_zone_tolerance * 0.90,
        atr_pct * 1.00,
        0.0046 if close < 0.02 else 0.0030,
    )
    recent_upper_anchor_levels = [
        level
        for level in (local_vah, recent_high)
        if level > 0
    ]
    recent_upper_anchor_levels.extend(
        band.level
        for band in liquidation_map.bands
        if band.side == "above"
        and (
            band.closed_index is None
            or band.closed_index >= max(len(enriched) - 8, 0)
        )
    )
    recent_high_samples = recent_high_tail.tail(3).to_numpy(dtype=float) if len(recent_high_tail) else np.array([], dtype=float)
    recent_upper_anchor_touch = bool(
        recent_upper_anchor_levels
        and recent_high_samples.size
        and any(
            abs(high_px - anchor_level) / max(anchor_level, 1e-8) <= anchor_touch_tolerance
            for high_px in recent_high_samples
            for anchor_level in recent_upper_anchor_levels
        )
    )
    anchor_reject_score = 0.0
    if first_reaction:
        anchor_reject_score += 0.55
    if momentum_rollover:
        anchor_reject_score += 0.45
    if volume_fade:
        anchor_reject_score += 0.45
    if has_structural_failure:
        anchor_reject_score += 1.05
    if recent_upper_liq_reaction:
        anchor_reject_score += 0.95
    if liquidation_map.swept_above:
        anchor_reject_score += 0.95
    if near_vah_resistance:
        anchor_reject_score += 0.55
    anchor_reject_floor = (
        1.10
        if recent_upper_liq_reaction or liquidation_map.swept_above
        else 1.45
        if near_vah_resistance and downside_confluence_score >= 1.10 and first_reaction and volume_fade
        else 1.85
    )
    recent_upper_anchor_reject = bool(
        recent_upper_anchor_touch
        and anchor_reject_score >= anchor_reject_floor
    )
    strong_upper_reversal_bar = bool(
        upper_wick / candle_range >= 0.24
        and (close < prev_close or close <= open_px)
    )
    hard_upper_anchor_context = bool(
        near_vah_resistance
        or recent_upper_liq_reaction
        or liquidation_map.swept_above
    )
    weak_watch_structure = not (
        has_structural_failure
        or near_vah_resistance
        or recent_upper_liq_reaction
        or liquidation_map.swept_above
    )
    if recent_upper_anchor_touch:
        resistance_confluence_score += 0.55
    if recent_upper_anchor_reject:
        resistance_confluence_score += 0.40
    downside_target_ready = bool(
        liquidation_map.downside_magnet
        or lower_liq_target_ready
        or value_room_to_poc
        or value_room_to_val
        or value_room_to_recent_low
    )
    upper_rejection_zone_ready = bool(
        near_vah_resistance
        or recent_upper_liq_reaction
        or liquidation_map.swept_above
        or has_structural_failure
        or near_peak_resistance
    )
    open_upside_liq_magnet = bool(
        liquidation_map.nearest_above_distance_pct is not None
        and liquidation_map.nearest_above_distance_pct <= max(
            atr_pct * 1.65,
            0.0060 if close < 0.02 else 0.0040,
        )
        and not liquidation_map.swept_above
    )
    if not upper_rejection_zone_ready:
        return None
    if not downside_target_ready:
        return None
    if not recent_upper_anchor_touch and not liquidation_map.swept_above and not recent_upper_liq_reaction:
        return None
    live_pump_weakening = bool(
        first_reaction
        and recent_upper_anchor_touch
        and price_rejection_near_high
        and (momentum_rollover_count >= 2 or (momentum_rollover and strong_upper_reversal_bar))
        and (
            volume_fade
            or weakness_strength >= 0.66
            or strong_upper_reversal_bar
        )
        and (
            near_vah_resistance
            or recent_upper_liq_reaction
            or liquidation_map.swept_above
            or liquidation_map.downside_magnet
            or value_room_to_poc
            or value_room_to_val
        )
        and (
            recent_upper_anchor_reject
            or strong_upper_reversal_bar
            or has_structural_failure
        )
    )
    shallow_reject_pullback_floor = max(
        minimum_reversal_pullback * 1.08,
        atr_pct * 0.82,
        0.0015 if close < 0.02 else 0.0010,
    )
    shallow_reject_without_displacement = bool(
        recent_upper_anchor_touch
        and recent_upper_anchor_reject
        and first_reaction
        and distance_from_high_pct < shallow_reject_pullback_floor
        and close >= recent_high * (1.0 - max(shallow_reject_pullback_floor * 0.72, 0.0010 if close < 0.02 else 0.0007))
        and not has_structural_failure
        and not liquidation_map.swept_above
        and not recent_upper_liq_reaction
        and not strong_upper_reversal_bar
    )
    mtf_5m_continuation_watch = bool(
        mtf_trend_5m >= _early_config_float("EARLY_APPROVED_MTF_TREND_RISK_MIN", 0.0060)
        and mtf_rsi_5m >= _early_config_float("EARLY_APPROVED_MTF_RSI_RISK_MIN", 75.0)
    )
    mtf_15m_continuation_watch = bool(
        mtf_trend_15m >= _early_config_float("EARLY_APPROVED_MTF_TREND_15M_RISK_MIN", 0.0046)
        and mtf_rsi_15m >= _early_config_float("EARLY_APPROVED_MTF_RSI_15M_RISK_MIN", 69.0)
    )
    mtf_1h_continuation_watch = bool(
        mtf_trend_1h >= _early_config_float("EARLY_APPROVED_MTF_TREND_1H_RISK_MIN", 0.0038)
        and mtf_rsi_1h >= _early_config_float("EARLY_APPROVED_MTF_RSI_1H_RISK_MIN", 64.0)
    )
    high_mtf_continuation_watch = bool(
        (mtf_5m_continuation_watch or mtf_15m_continuation_watch or mtf_1h_continuation_watch)
        and close >= prev_close * (1.0 - (0.0012 if close < 0.02 else 0.0008))
        and close_position_in_candle >= 0.52
        and not has_structural_failure
        and not liquidation_map.swept_above
        and not recent_upper_liq_reaction
        and not volume_fade
        and not price_rejection_near_high
    )
    if high_mtf_continuation_watch and not live_pump_weakening:
        return None
    if shallow_reject_without_displacement:
        return None
    if (
        not has_structural_failure
        and not price_rejection_near_high
        and not liquidation_map.swept_above
        and not recent_upper_liq_reaction
        and not live_pump_weakening
    ):
        return None
    if not recent_upper_anchor_reject and not (has_structural_failure and first_reaction) and not live_pump_weakening:
        return None
    if (
        not has_structural_failure
        and not liquidation_map.swept_above
        and not recent_upper_liq_reaction
        and not strong_upper_reversal_bar
        and not live_pump_weakening
    ):
        return None
    if open_upside_liq_magnet and not has_structural_failure:
        return None
    upper_zone_distance_candidates = []
    if local_vah > 0:
        upper_zone_distance_candidates.append(abs(close - local_vah) / max(close, 1e-8))
    if recent_high > 0:
        upper_zone_distance_candidates.append(abs(recent_high - close) / max(close, 1e-8))
    for band in liquidation_map.bands:
        if band.side != "above":
            continue
        if band.closed_index is None and band.level > close:
            upper_zone_distance_candidates.append(abs(band.level - close) / max(close, 1e-8))
        elif band.closed_index is not None and band.closed_index >= max(len(enriched) - 8, 0):
            upper_zone_distance_candidates.append(abs(band.level - close) / max(close, 1e-8))
    entry_zone_distance_pct = min(upper_zone_distance_candidates) if upper_zone_distance_candidates else 1.0
    max_setup_entry_zone_distance = max(
        atr_pct * 1.15,
        0.0054 if close < 0.02 else 0.0035,
    )
    if entry_zone_distance_pct > max_setup_entry_zone_distance and not liquidation_map.swept_above:
        return None
    min_resistance_confluence = 2.15 if has_structural_failure else (2.18 if live_pump_weakening else 2.45)
    if resistance_confluence_score < min_resistance_confluence:
        return None
    if downside_confluence_score < 1.10:
        return None
    if weakness_strength < (0.50 if live_pump_weakening else 0.54) and not (has_structural_failure and first_reaction):
        return None
    if not first_reaction:
        return None
    if not (momentum_rollover or volume_fade or has_structural_failure):
        return None
    if not has_structural_failure and weakness_strength < (0.56 if live_pump_weakening else 0.60) and not volume_fade:
        return None
    if not has_structural_failure and not (
        (momentum_rollover and volume_fade)
        or live_pump_weakening
    ):
        return None
    marginal_pump_without_hard_context = bool(
        clean_pump_pct < max(pump_min + 0.0045, 0.0475)
        and not has_structural_failure
        and not liquidation_map.swept_above
        and not recent_upper_liq_reaction
    )
    if marginal_pump_without_hard_context:
        return None

    triggers: list[str] = []
    if close >= recent_high * 0.992:
        triggers.append("цена всё ещё у локального хая")
    if first_reaction:
        triggers.append("пошла первая реакция вниз")
    if hist < prev_hist:
        triggers.append("MACD ослабевает")
    if rsi < prev_rsi:
        triggers.append("RSI разворачивается вниз")
    if volume_fade:
        triggers.append("объём затухает")
    if failed_reclaim:
        triggers.append("ретестом не вернули уровень")
    if retest_failed_breakout:
        triggers.append("сломанный пробой не удержали")
    if weakness_strength >= 0.62:
        triggers.append("слабость уже подтверждается структурно")
    if near_vah_resistance:
        triggers.append("цена у VAH")
    if liquidation_map.swept_above:
        triggers.append("верхнюю ликвидность уже сняли")
    if liquidation_map.downside_magnet:
        triggers.append("ниже есть ликвидационный магнит")
    if value_room_to_poc:
        triggers.append("ниже есть POC как цель отката")
    if value_room_to_val:
        triggers.append("ниже есть VAL как цель отката")

    unique_triggers: list[str] = []
    for trigger in triggers:
        if trigger not in unique_triggers:
            unique_triggers.append(trigger)

    continuation_risk = 3.15
    continuation_risk -= min(1.20, weakness_strength * 1.35)
    if first_reaction:
        continuation_risk -= 0.40
    if volume_fade:
        continuation_risk -= 0.25
    if failed_reclaim:
        continuation_risk -= 0.35
    if retest_failed_breakout:
        continuation_risk -= 0.30
    if upper_rejection_zone_ready:
        continuation_risk -= 0.20
    if liquidation_map.swept_above:
        continuation_risk -= 0.22
    if downside_target_ready:
        continuation_risk -= 0.15
    if open_upside_liq_magnet:
        continuation_risk += 0.60
    continuation_risk = max(0.8, min(4.0, continuation_risk))

    watch_score = 4.8 + min(2.0, weakness_strength * 2.1)
    if first_reaction:
        watch_score += 0.70
    if failed_reclaim:
        watch_score += 0.60
    if retest_failed_breakout:
        watch_score += 0.55
    if volume_fade:
        watch_score += 0.35
    watch_score = min(8.0, max(0.0, watch_score))

    quality_score = 6.0 + min(2.2, weakness_strength * 2.0)
    if first_reaction:
        quality_score += 0.50
    if failed_reclaim:
        quality_score += 0.55
    if retest_failed_breakout:
        quality_score += 0.45
    if near_vah_resistance:
        quality_score += 0.35
    if liquidation_map.swept_above:
        quality_score += 0.45
    if downside_target_ready:
        quality_score += 0.20
    if resistance_confluence_score > 0.0:
        quality_score += min(0.55, resistance_confluence_score * 0.16)
    if downside_confluence_score > 0.0:
        quality_score += min(0.35, downside_confluence_score * 0.10)
    quality_score -= continuation_risk * 0.28
    if entry_zone_distance_pct > max_setup_entry_zone_distance * 0.78:
        quality_score -= 0.30
    quality_score = min(10.0, max(0.0, quality_score))

    phase = "SETUP" if has_structural_failure else "WATCH"
    watch_live_volume_floor = 0.18 if close < 0.02 else 0.12
    if hard_upper_anchor_context:
        watch_live_volume_floor = min(
            watch_live_volume_floor,
            0.12 if close < 0.02 else 0.08,
        )
    live_continuation_watch = bool(
        phase == "WATCH"
        and up_closes_last3 >= 2
        and close >= recent_close_high * 0.9975
        and close_position_in_candle >= 0.56
        and volume_spike >= max(0.95, recent_volume_spike * 0.86, 0.90 if close < 1.0 else 0.70)
        and not volume_fade
        and not strong_upper_reversal_bar
        and not has_structural_failure
        and not liquidation_map.swept_above
        and not recent_upper_liq_reaction
    )
    if (
        phase == "WATCH"
        and not (recent_upper_liq_reaction or liquidation_map.swept_above)
        and not (
            near_vah_resistance
            and recent_upper_anchor_reject
            and downside_confluence_score >= 1.35
            and volume_fade
            and momentum_rollover
        )
    ):
        return None
    if (
        phase == "WATCH"
        and volume_spike >= (2.40 if close < 0.02 else 1.95)
        and not has_structural_failure
        and not liquidation_map.swept_above
        and not recent_upper_liq_reaction
    ):
        return None
    if (
        phase == "WATCH"
        and weak_watch_structure
        and not (
            recent_upper_anchor_reject
            or strong_upper_reversal_bar
            or peak_followthrough_confirmed
            or liquidation_map.swept_above
        )
    ):
        return None
    if (
        phase == "WATCH"
        and weak_watch_structure
        and volume_spike >= max(0.95, recent_volume_spike * 0.97, 0.95 if close < 0.02 else 0.72)
        and not volume_fade
        and not peak_followthrough_confirmed
        and not liquidation_map.swept_above
    ):
        return None
    if live_continuation_watch and not live_pump_weakening:
        return None
    fresh_live_extension_watch = bool(
        phase == "WATCH"
        and near_peak_resistance
        and close >= prev_close
        and close_position_in_candle >= 0.68
        and upper_wick / candle_range <= 0.20
        and volume_spike >= max(recent_volume_spike * 0.94, 0.75 if close < 1.0 else 0.58)
        and not live_pump_weakening
        and not has_structural_failure
        and not liquidation_map.swept_above
        and not recent_upper_liq_reaction
    )
    if fresh_live_extension_watch:
        return None
    required_trigger_count = 3 if phase == "WATCH" and live_pump_weakening else 4
    if len(unique_triggers) < required_trigger_count:
        return None
    if phase == "WATCH" and not has_structural_failure and not hard_upper_anchor_context:
        return None
    if (
        phase == "WATCH"
        and volume_spike < watch_live_volume_floor
        and not (
            has_structural_failure
            or recent_upper_liq_reaction
            or liquidation_map.swept_above
        )
    ):
        return None
    if phase == "SETUP" and continuation_risk > 2.45:
        return None
    if phase == "WATCH" and continuation_risk > (3.18 if live_pump_weakening else 3.02):
        return None
    if phase == "WATCH" and weak_watch_structure and continuation_risk > 2.65:
        return None
    if phase == "WATCH" and live_continuation_watch and continuation_risk > 2.05:
        return None
    if (
        phase == "WATCH"
        and not has_structural_failure
        and not liquidation_map.swept_above
        and not recent_upper_liq_reaction
        and continuation_risk > 2.35
    ):
        return None
    if phase == "SETUP" and quality_score < 7.05:
        return None
    if phase == "WATCH" and quality_score < (6.40 if live_pump_weakening else 6.70):
        return None
    if phase == "WATCH" and weak_watch_structure and quality_score < (6.90 if live_pump_weakening else 7.15):
        return None
    if phase == "WATCH" and not has_structural_failure:
        quality_score = min(
            quality_score,
            7.60 if live_pump_weakening else (7.45 if strong_upper_reversal_bar else 7.10),
        )
    if phase == "WATCH" and weak_watch_structure:
        quality_score = min(quality_score, 6.95 if recent_upper_anchor_reject else 6.55)
    if phase == "WATCH" and not (recent_upper_liq_reaction or liquidation_map.swept_above):
        quality_score = min(quality_score, 7.15 if near_vah_resistance else 6.95)
    if phase == "WATCH" and volume_spike < (0.20 if close < 0.02 else 0.13):
        quality_score = min(quality_score, 6.85)
    soft_anchor_reject_without_followthrough = bool(
        phase == "WATCH"
        and not has_structural_failure
        and recent_upper_anchor_touch
        and recent_upper_anchor_reject
        and up_closes_last3 >= 2
        and close >= prev_close * (1.0 - (0.0010 if close < 0.02 else 0.0007))
        and close_position_in_candle >= 0.52
        and not volume_fade
        and not strong_upper_reversal_bar
        and not liquidation_map.swept_above
    )
    if soft_anchor_reject_without_followthrough:
        return None
    wait_for = (
        "подтверждение полноценного входа"
        if phase == "SETUP"
        else "подтверждение слабости и входа"
    )
    phase_label = "\\u0420\\u0410\\u041d\\u041d\\u0418\\u0419 \\u0428\\u041e\\u0420\\u0422: \\u0421\\u0415\\u0422\\u0410\\u041f" if phase == "SETUP" else "\\u0420\\u0410\\u041d\\u041d\\u0418\\u0419 \\u0428\\u041e\\u0420\\u0422: \\u041d\\u0410\\u0411\\u041b\\u042e\\u0414\\u0415\\u041d\\u0418\\u0415"

    tp = _as_float(getattr(intent, "take_profit", None), close - atr * 1.55)
    sl = _as_float(getattr(intent, "stop_loss", None), close + atr * 1.05)
    if tp <= 0.0:
        tp = close - atr * 1.55
    if sl <= close:
        sl = close + atr * 1.05

    return {
        "phase": phase,
        "caption": build_early_signal_caption(
            symbol=symbol,
            timeframe=timeframe,
            mode=mode,
            phase_label=phase_label,
            price=close,
            trace_meta=meta,
            watch_score=watch_score,
            watch_max_score=8.0,
            quality_score=quality_score,
            quality_max_score=10.0,
            quality_grade=_early_quality_grade(quality_score),
            continuation_risk=continuation_risk,
            continuation_max_score=4.0,
            triggers=unique_triggers,
            wait_for=wait_for,
            enriched=enriched,
        ),
        "entry": close,
        "tp": tp,
        "sl": sl,
    }


def _build_early_signal_candidate(*, symbol: str, timeframe: str, mode: str, enriched, intent) -> dict[str, object] | None:
    candidate = _build_approved_early_candidate(
        symbol=symbol,
        timeframe=timeframe,
        mode=mode,
        enriched=enriched,
        intent=intent,
    )
    if candidate is not None:
        return candidate
    return _build_early_watch_candidate(
        symbol=symbol,
        timeframe=timeframe,
        mode=mode,
        enriched=enriched,
        intent=intent,
    )


def _strategy_audit_log_payload(strategy) -> dict[str, object]:
    numeric_regime_keys = (
        "htf_trend_metric_used",
        "htf_trend_threshold_used",
        "vwap_distance_metric_used",
        "vwap_stretch_threshold_used",
        "atr_norm",
        "volatility_threshold_used",
        "mtf_trend_5m_used",
        "mtf_trend_15m_used",
        "mtf_rsi_5m_used",
        "mtf_rsi_15m_used",
        "mtf_hard_filter_enabled",
        "degraded_mode",
        "fail_due_to_degraded_mode_only",
        "soft_pass_candidate",
        "soft_pass_used",
    )
    numeric_layer1_keys = (
        "rsi",
        "rsi_high_threshold_used",
        "volume_spike",
        "volume_spike_high",
        "volume_spike_threshold_used",
        "close_metric_used",
        "bollinger_upper_metric_used",
        "keltner_upper_metric_used",
        "above_bollinger_upper",
        "above_keltner_upper",
        "upper_band_breakout",
        "pump_context_strength",
        "clean_pump_pct",
        "clean_pump_min_pct_used",
        "clean_pump_ok",
        "soft_pass_candidate",
        "soft_pass_used",
        "pump_bar_offset",
    )
    numeric_layer2_keys = (
        "price_up_or_near_high",
        "price_up",
        "near_high_context",
        "obv_bearish_divergence",
        "cvd_bearish_divergence",
        "close_last_used",
        "close_ref_used",
        "obv_last_used",
        "obv_ref_used",
        "cvd_last_used",
        "cvd_ref_used",
        "weakness_lookback_used",
        "weakness_strength",
    )
    regime_diag_defaults: dict[str, object] = {
        "htf_trend_metric_used": None,
        "htf_trend_threshold_used": None,
        "htf_trend_direction_context": "",
        "vwap_distance_metric_used": None,
        "vwap_stretch_threshold_used": None,
        "atr_norm": None,
        "volatility_threshold_used": None,
        "mtf_trend_5m_used": None,
        "mtf_trend_15m_used": None,
        "mtf_rsi_5m_used": None,
        "mtf_rsi_15m_used": None,
        "mtf_hard_filter_enabled": 0.0,
        "failed_reason": "",
        "missing_conditions": "",
        "degraded_mode": 0.0,
        "fail_due_to_degraded_mode_only": 0.0,
        "soft_pass_candidate": 0.0,
        "soft_pass_used": 0.0,
        "soft_pass_reason": "",
        "source_flags": {},
        "regime_filter_subconditions_state": {},
    }
    layer1_diag_defaults: dict[str, object] = {
        "rsi": None,
        "rsi_high_threshold_used": None,
        "volume_spike": None,
        "volume_spike_high": 0.0,
        "volume_spike_threshold_used": None,
        "close_metric_used": None,
        "bollinger_upper_metric_used": None,
        "keltner_upper_metric_used": None,
        "above_bollinger_upper": 0.0,
        "above_keltner_upper": 0.0,
        "upper_band_breakout": 0.0,
        "pump_context_strength": 0.0,
        "clean_pump_pct": 0.0,
        "clean_pump_min_pct_used": 0.0,
        "clean_pump_ok": 0.0,
        "failed_reason": "",
        "missing_conditions": "",
        "soft_pass_candidate": 0.0,
        "soft_pass_used": 0.0,
        "soft_pass_reason": "",
        "pump_bar_offset": None,
        "layer1_subconditions_state": {},
    }
    layer2_diag_defaults: dict[str, object] = {
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
        "weakness_lookback_used": None,
        "weakness_strength": 0.0,
        "failed_reason": "",
        "missing_conditions": "",
        "layer2_subconditions_state": {},
    }

    def _normalize_bool_state_map(source: object, keys: tuple[str, ...]) -> dict[str, bool]:
        if not isinstance(source, Mapping):
            return {}
        out: dict[str, bool] = {}
        for key in keys:
            if key not in source:
                continue
            value = source.get(key)
            if isinstance(value, bool):
                out[key] = value
            elif isinstance(value, (int, float)):
                out[key] = float(value) != 0.0
            elif isinstance(value, str):
                text = value.strip().lower()
                if text in ("1", "true", "yes", "y", "on"):
                    out[key] = True
                elif text in ("0", "false", "no", "n", "off", ""):
                    out[key] = False
        return out

    def _extract_regime_diag(source: Mapping[str, object] | object) -> dict[str, object]:
        out = dict(regime_diag_defaults)
        if not isinstance(source, Mapping):
            return out
        for key in numeric_regime_keys:
            if key not in source:
                continue
            value = source.get(key)
            try:
                out[key] = float(value) if value is not None else None
            except (TypeError, ValueError):
                out[key] = None
        direction = source.get("htf_trend_direction_context")
        out["htf_trend_direction_context"] = str(direction) if direction is not None else ""
        out["failed_reason"] = str(source.get("failed_reason") or "")
        out["missing_conditions"] = str(source.get("missing_conditions") or "")
        out["soft_pass_reason"] = str(source.get("soft_pass_reason") or "")
        source_flags = source.get("source_flags", {})
        out["source_flags"] = dict(source_flags) if isinstance(source_flags, Mapping) else {}
        subconditions = source.get("regime_filter_subconditions_state", {})
        out["regime_filter_subconditions_state"] = _normalize_bool_state_map(
            subconditions,
            (
                "htf_trend_ok",
                "stretched_from_vwap",
                "volatility_regime_ok",
                "news_veto",
                "mtf_short_context_ok",
            ),
        )
        return out

    def _extract_layer1_diag(source: Mapping[str, object] | object) -> dict[str, object]:
        out = dict(layer1_diag_defaults)
        if not isinstance(source, Mapping):
            return out
        for key in numeric_layer1_keys:
            if key not in source:
                continue
            value = source.get(key)
            try:
                out[key] = float(value) if value is not None else None
            except (TypeError, ValueError):
                out[key] = None
        out["failed_reason"] = str(source.get("failed_reason") or "")
        out["missing_conditions"] = str(source.get("missing_conditions") or "")
        out["soft_pass_reason"] = str(source.get("soft_pass_reason") or "")
        subconditions = source.get("layer1_subconditions_state", {})
        out["layer1_subconditions_state"] = _normalize_bool_state_map(
            subconditions,
            (
                "rsi_high",
                "volume_spike_high",
                "upper_band_breakout",
                "above_bollinger_upper",
                "above_keltner_upper",
                "clean_pump_ok",
            ),
        )
        return out

    def _extract_layer2_diag(source: Mapping[str, object] | object) -> dict[str, object]:
        out = dict(layer2_diag_defaults)
        if not isinstance(source, Mapping):
            return out
        for key in numeric_layer2_keys:
            if key not in source:
                continue
            value = source.get(key)
            try:
                out[key] = float(value) if value is not None else None
            except (TypeError, ValueError):
                out[key] = None
        out["failed_reason"] = str(source.get("failed_reason") or "")
        out["missing_conditions"] = str(source.get("missing_conditions") or "")
        subconditions = source.get("layer2_subconditions_state", {})
        out["layer2_subconditions_state"] = _normalize_bool_state_map(
            subconditions,
            (
                "price_up_or_near_high",
                "price_up",
                "near_high_context",
                "obv_bearish_divergence",
                "cvd_bearish_divergence",
            ),
        )
        return out

    full_snapshot = {}
    compact_snapshot = {}
    regime_filter_snapshot = {}
    regime_diagnostics_snapshot = {}
    layer1_snapshot = {}
    layer1_diagnostics_snapshot = {}
    layer2_snapshot = {}
    layer2_diagnostics_snapshot = {}
    layer4_snapshot = {}
    source_quality_snapshot = {}

    if hasattr(strategy, "audit_observation_snapshot"):
        try:
            observation = strategy.audit_observation_snapshot()
        except Exception:
            observation = {}
        if isinstance(observation, Mapping):
            if isinstance(observation.get("strategy_audit_compact"), Mapping):
                compact_snapshot = dict(observation.get("strategy_audit_compact", {}))
            if isinstance(observation.get("strategy_audit_regime_filter"), Mapping):
                regime_filter_snapshot = dict(observation.get("strategy_audit_regime_filter", {}))
            if isinstance(observation.get("strategy_audit_regime_diagnostics"), Mapping):
                regime_diagnostics_snapshot = _extract_regime_diag(observation.get("strategy_audit_regime_diagnostics", {}))
            if isinstance(observation.get("strategy_audit_layer1"), Mapping):
                layer1_snapshot = dict(observation.get("strategy_audit_layer1", {}))
            if isinstance(observation.get("strategy_audit_layer1_diagnostics"), Mapping):
                layer1_diagnostics_snapshot = _extract_layer1_diag(observation.get("strategy_audit_layer1_diagnostics", {}))
            if isinstance(observation.get("strategy_audit_layer2"), Mapping):
                layer2_snapshot = dict(observation.get("strategy_audit_layer2", {}))
            if isinstance(observation.get("strategy_audit_layer2_diagnostics"), Mapping):
                layer2_diagnostics_snapshot = _extract_layer2_diag(observation.get("strategy_audit_layer2_diagnostics", {}))
            if isinstance(observation.get("strategy_audit_layer4"), Mapping):
                layer4_snapshot = dict(observation.get("strategy_audit_layer4", {}))
            if isinstance(observation.get("strategy_audit_source_quality"), Mapping):
                source_quality_snapshot = dict(observation.get("strategy_audit_source_quality", {}))

    if hasattr(strategy, "audit_snapshot"):
        try:
            snapshot_candidate = strategy.audit_snapshot()
        except Exception:
            snapshot_candidate = {}
        if isinstance(snapshot_candidate, Mapping):
            full_snapshot = dict(snapshot_candidate)

    if not compact_snapshot:
        if hasattr(strategy, "audit_compact_snapshot"):
            try:
                compact_candidate = strategy.audit_compact_snapshot()
            except Exception:
                compact_candidate = {}
            if isinstance(compact_candidate, Mapping):
                compact_snapshot = dict(compact_candidate)
        if not compact_snapshot:
            compact_snapshot = dict(full_snapshot)

    if not regime_filter_snapshot:
        regime_filter_snapshot = {
            "regime_filter_pass_count": int(full_snapshot.get("regime_filter_pass_count", 0)),
            "regime_filter_fail_count": int(full_snapshot.get("regime_filter_fail_count", 0)),
            "regime_filter_htf_trend_blocker_count": int(full_snapshot.get("regime_filter_htf_trend_blocker_count", 0)),
            "regime_filter_vwap_stretch_blocker_count": int(full_snapshot.get("regime_filter_vwap_stretch_blocker_count", 0)),
            "regime_filter_volatility_blocker_count": int(full_snapshot.get("regime_filter_volatility_blocker_count", 0)),
            "regime_filter_news_blocker_count": int(full_snapshot.get("regime_filter_news_blocker_count", 0)),
            "regime_filter_degraded_mode_count": int(full_snapshot.get("regime_filter_degraded_mode_count", 0)),
            "regime_filter_degraded_only_count": int(full_snapshot.get("regime_filter_degraded_only_count", 0)),
            "regime_filter_soft_pass_candidate_count": int(full_snapshot.get("regime_filter_soft_pass_candidate_count", 0)),
            "regime_filter_soft_pass_used_count": int(full_snapshot.get("regime_filter_soft_pass_used_count", 0)),
            "top_regime_filter_blocker": str(compact_snapshot.get("top_regime_filter_blocker", "")),
            "top_regime_filter_blocker_count": int(compact_snapshot.get("top_regime_filter_blocker_count", 0)),
        }

    if not layer1_snapshot:
        layer1_snapshot = {
            "layer1_pass_count": int(full_snapshot.get("layer1_pass_count", 0)),
            "layer1_fail_count": int(full_snapshot.get("layer1_fail_count", 0)),
            "layer1_rsi_high_blocker_count": int(full_snapshot.get("layer1_rsi_high_blocker_count", 0)),
            "layer1_volume_spike_blocker_count": int(full_snapshot.get("layer1_volume_spike_blocker_count", 0)),
            "layer1_above_bollinger_upper_blocker_count": int(full_snapshot.get("layer1_above_bollinger_upper_blocker_count", 0)),
            "layer1_above_keltner_upper_blocker_count": int(full_snapshot.get("layer1_above_keltner_upper_blocker_count", 0)),
            "layer1_clean_pump_pct_blocker_count": int(full_snapshot.get("layer1_clean_pump_pct_blocker_count", 0)),
            "layer1_soft_pass_candidate_count": int(full_snapshot.get("layer1_soft_pass_candidate_count", 0)),
            "top_layer1_blocker": str(compact_snapshot.get("top_layer1_blocker", "")),
            "top_layer1_blocker_count": int(compact_snapshot.get("top_layer1_blocker_count", 0)),
        }
    if not layer4_snapshot:
        layer4_snapshot = {
            "layer4_fail_count": int(full_snapshot.get("layer4_fail_count", 0)),
            "layer4_sentiment_blocker_count": int(full_snapshot.get("layer4_sentiment_blocker_count", 0)),
            "layer4_funding_blocker_count": int(full_snapshot.get("layer4_funding_blocker_count", 0)),
            "layer4_lsr_blocker_count": int(full_snapshot.get("layer4_lsr_blocker_count", 0)),
            "layer4_oi_blocker_count": int(full_snapshot.get("layer4_oi_blocker_count", 0)),
            "layer4_price_blocker_count": int(full_snapshot.get("layer4_price_blocker_count", 0)),
            "layer4_degraded_mode_count": int(full_snapshot.get("layer4_degraded_mode_count", 0)),
            "layer4_soft_pass_candidate_count": int(full_snapshot.get("layer4_soft_pass_candidate_count", 0)),
        }
    if not layer2_snapshot:
        layer2_snapshot = {
            "reached_layer2_count": int(full_snapshot.get("reached_layer2_count", 0)),
            "passed_layer2_count": int(full_snapshot.get("passed_layer2_count", 0)),
            "layer2_fail_count": int(full_snapshot.get("layer2_fail_count", 0)),
        }
    else:
        layer2_snapshot.setdefault("reached_layer2_count", int(full_snapshot.get("reached_layer2_count", 0)))
        layer2_snapshot.setdefault("passed_layer2_count", int(full_snapshot.get("passed_layer2_count", 0)))
        layer2_snapshot.setdefault("layer2_fail_count", int(full_snapshot.get("layer2_fail_count", 0)))

    if not source_quality_snapshot and isinstance(full_snapshot.get("source_quality_summary"), Mapping):
        source_quality_snapshot = dict(full_snapshot.get("source_quality_summary", {}))

    if not regime_diagnostics_snapshot:
        regime_diagnostics_snapshot = _extract_regime_diag(regime_filter_snapshot)
    for key, value in regime_diagnostics_snapshot.items():
        regime_filter_snapshot.setdefault(key, value)

    if not layer1_diagnostics_snapshot:
        layer1_diagnostics_snapshot = _extract_layer1_diag(layer1_snapshot)
    for key, value in layer1_diagnostics_snapshot.items():
        layer1_snapshot.setdefault(key, value)

    if not layer2_diagnostics_snapshot:
        layer2_diagnostics_snapshot = _extract_layer2_diag(layer2_snapshot)
    for key, value in layer2_diagnostics_snapshot.items():
        layer2_snapshot.setdefault(key, value)

    return {
        "strategy_audit_compact": compact_snapshot,
        "strategy_audit_regime_filter": regime_filter_snapshot,
        "strategy_audit_regime_diagnostics": regime_diagnostics_snapshot,
        "strategy_audit_layer1": layer1_snapshot,
        "strategy_audit_layer1_diagnostics": layer1_diagnostics_snapshot,
        "strategy_audit_layer2": layer2_snapshot,
        "strategy_audit_layer2_diagnostics": layer2_diagnostics_snapshot,
        "strategy_audit_layer4": layer4_snapshot,
        "strategy_audit_source_quality": source_quality_snapshot,
        "strategy_audit": full_snapshot,
    }

def _startup_reconcile(
    *,
    symbols: list[str],
    sync: ExchangeSyncService,
    state_machine: StateMachine,
    execution: ExecutionEngine,
    signal_profile: str = "main",
    logger=None,
):
    normalized_symbols = [str(symbol).replace("/", "").upper().strip() for symbol in symbols if symbol]
    summary: dict[str, str] = {symbol: TradeState.FLAT.value for symbol in normalized_symbols}
    if _is_observation_only_profile(signal_profile):
        return summary

    active_symbols: set[str] = set()
    persisted_records = getattr(state_machine, "_records", {})
    if isinstance(persisted_records, Mapping):
        for symbol, rec in persisted_records.items():
            state_value = getattr(rec, "state", TradeState.FLAT)
            if state_value != TradeState.FLAT:
                active_symbols.add(str(symbol).replace("/", "").upper().strip())

    if execution.persistence is not None:
        try:
            inflight_rows = execution.persistence.load_open_inflight_intents()
        except Exception:
            inflight_rows = []
        for row in inflight_rows:
            active_symbols.add(str(getattr(row, "symbol", "")).replace("/", "").upper().strip())

    bulk_snapshots = sync.snapshot_many(normalized_symbols)
    for symbol, snapshot in bulk_snapshots.items():
        if list(snapshot.positions) or list(snapshot.open_orders):
            active_symbols.add(str(symbol).replace("/", "").upper().strip())

    if logger is not None:
        try:
            logger.info(
                "startup_reconcile_scope total=%d active_candidates=%d exchange_active=%d",
                len(normalized_symbols),
                len(active_symbols),
                sum(
                    1
                    for snapshot in bulk_snapshots.values()
                    if list(snapshot.positions) or list(snapshot.open_orders)
                ),
                extra={"event": "startup_reconcile_scope"},
            )
        except Exception:
            pass

    for symbol in normalized_symbols:
        if symbol not in active_symbols:
            continue
        snapshot = bulk_snapshots.get(symbol) or sync.snapshot(symbol)
        rec_state = state_machine.reconcile(symbol, snapshot.positions, snapshot.open_orders)
        execution.recover_from_restart(symbol, snapshot)

        # Recovery can change exchange state (cancel/close/attach stop), refresh from exchange truth.
        snapshot = sync.reconciler.snapshot(symbol)
        rec_state = state_machine.reconcile(symbol, snapshot.positions, snapshot.open_orders)
        issues = execution.detect_external_intervention(symbol, snapshot)
        if issues:
            summary[symbol] = f"{state_machine.get(symbol).state.value}|issues={','.join(issues)}"
        else:
            summary[symbol] = rec_state.state.value
    return summary



def run_cycle(
    *,
    symbols: list[str],
    adapter: BybitAdapter,
    feed: MarketDataFeed,
    sync: ExchangeSyncService,
    pipeline: FeaturePipeline,
    strategy,
    risk: RiskEngine,
    execution: ExecutionEngine,
    logger,
    counters: MetricsCounter,
    timeframe: str,
    candles_limit: int,
    alerters,
    state_alert_cache: dict[str, object],
    intervention_alert_cache: dict[str, object],
    decision_log_cache: dict[str, object],
    signal_alert_cache: dict[str, object],
    early_signal_state: dict[str, dict[str, object]],
    early_signal_stats: dict[str, int],
    mode: str,
    trade_learner=None,
    online_retrainer=None,
    early_signal_learner=None,
    early_online_retrainer=None,
    signal_profile: str = "both",
):
    sync.pull_adapter_events(adapter)
    ws_recovery_reason = sync.maybe_recover_ws(adapter)
    log_hold_decisions = _env_flag("LOG_HOLD_DECISIONS", False)
    log_dataset_appends = _env_flag("LOG_ONLINE_DATASET_APPENDS", False)
    if ws_recovery_reason:
        logger.warning(
            "ws_reconnect_triggered reason=%s",
            ws_recovery_reason,
            extra={"event": "ws_reconnect_triggered"},
        )
    prepared_contexts: dict[str, dict[str, object]] = {}
    decision_contexts: dict[str, dict[str, object]] = {}
    prepared_decisions: dict[str, dict[str, object]] = {}
    analysis_workers = _analysis_worker_count(len(symbols))
    observation_only_profile = _is_observation_only_profile(signal_profile)
    intervention_alert_cooldown_sec = max(
        300,
        _as_int(os.getenv("INTERVENTION_ALERT_COOLDOWN_SEC", "10800"), 10800),
    )
    state_blocked_alert_cooldown_sec = max(
        300,
        _as_int(os.getenv("STATE_BLOCKED_ALERT_COOLDOWN_SEC", "10800"), 10800),
    )
    signal_alert_cooldown_sec = max(
        300,
        _as_int(os.getenv("SIGNAL_ALERT_COOLDOWN_SEC", "5400"), 5400),
    )
    signal_symbol_cooldown_sec = max(
        900,
        _as_int(os.getenv("SIGNAL_ALERT_SYMBOL_COOLDOWN_SEC", "2700"), 2700),
    )
    main_reentry_cooldown_sec = max(
        300,
        _as_int(os.getenv("MAIN_SIGNAL_REENTRY_COOLDOWN_SEC", "1800"), 1800),
    )
    bulk_snapshots: dict[str, object] = {}

    if symbols:
        if analysis_workers > 1 and len(symbols) > 1:
            with ThreadPoolExecutor(max_workers=analysis_workers) as executor:
                futures = [
                    (
                        symbol,
                        executor.submit(
                            _prepare_symbol_analysis,
                            symbol=symbol,
                            snapshot=None,
                            rec_state=None,
                            feed=feed,
                            pipeline=pipeline,
                            timeframe=timeframe,
                            candles_limit=candles_limit,
                        ),
                    )
                    for symbol in symbols
                ]
                for symbol, future in futures:
                    try:
                        prepared_contexts[symbol] = future.result()
                    except Exception as exc:
                        prepared_contexts[symbol] = {
                            "symbol": symbol,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
        else:
            for symbol in symbols:
                prepared_contexts[symbol] = _prepare_symbol_analysis(
                    symbol=symbol,
                    snapshot=None,
                    rec_state=None,
                    feed=feed,
                    pipeline=pipeline,
                    timeframe=timeframe,
                    candles_limit=candles_limit,
                )

    try:
        bulk_snapshots = sync.snapshot_many(symbols)
    except Exception as exc:
        logger.warning("bulk_snapshot_unavailable err=%s", exc, extra={"event": "bulk_snapshot_unavailable"})
        bulk_snapshots = {}

    for symbol in symbols:
        try:
            snapshot = bulk_snapshots.get(symbol) or sync.snapshot(symbol)
            if observation_only_profile:
                rec_state = _observation_state_record(symbol)
                cycle_ts = time.time()
                intervention = []
            else:
                rec_state = execution.state_machine.reconcile(symbol, snapshot.positions, snapshot.open_orders)
                recovery_touched_exchange = execution.recover_from_restart(symbol, snapshot)

                if recovery_touched_exchange:
                    snapshot = sync.reconciler.snapshot(symbol)
                    rec_state = execution.state_machine.reconcile(symbol, snapshot.positions, snapshot.open_orders)
                else:
                    rec_state = execution.state_machine.get(symbol)

                cycle_ts = time.time()
                intervention = execution.detect_external_intervention(symbol, snapshot)
            if intervention:
                counters.inc("interventions")
                issues = ",".join(intervention)
                current_state_value = execution.state_machine.get(symbol).state.value
                intervention_key = f"{current_state_value}|{issues}"
                if _should_emit_cached_alert(
                    intervention_alert_cache,
                    symbol=symbol,
                    key=intervention_key,
                    now_ts=cycle_ts,
                    cooldown_sec=intervention_alert_cooldown_sec,
                ):
                    logger.error(
                        "symbol=%s state=%s intervention=%s",
                        symbol,
                        current_state_value,
                        issues,
                        extra={"event": "intervention"},
                    )
                    _send_alerts(
                        alerters,
                        f"[\\u041a\\u0420\\u0418\\u0422\\u0418\\u0427\\u041d\\u041e] intervention symbol={symbol} issues={issues} state={current_state_value}",
                    )
                    _remember_cached_alert(
                        intervention_alert_cache,
                        symbol=symbol,
                        key=intervention_key,
                        now_ts=cycle_ts,
                        cooldown_sec=intervention_alert_cooldown_sec,
                    )
            else:
                intervention_alert_cache.pop(symbol, None)

            current_state = rec_state.state if observation_only_profile else execution.state_machine.get(symbol).state
            if current_state in (TradeState.HALTED, TradeState.RECOVERING, TradeState.ERROR):
                counters.inc("state_blocked")
                state_key = current_state.value
                if _should_emit_cached_alert(
                    state_alert_cache,
                    symbol=symbol,
                    key=state_key,
                    now_ts=cycle_ts,
                    cooldown_sec=state_blocked_alert_cooldown_sec,
                ):
                    reason = execution.state_machine.get(symbol).reason
                    logger.error(
                        "symbol=%s blocked_state=%s reason=%s",
                        symbol,
                        state_key,
                        reason,
                        extra={"event": "state_blocked"},
                    )
                    _send_alerts(
                        alerters,
                        f"[\\u041a\\u0420\\u0418\\u0422\\u0418\\u0427\\u041d\\u041e] state_blocked symbol={symbol} state={state_key} reason={reason}",
                    )
                    _remember_cached_alert(
                        state_alert_cache,
                        symbol=symbol,
                        key=state_key,
                        now_ts=cycle_ts,
                        cooldown_sec=state_blocked_alert_cooldown_sec,
                    )
                continue

            state_alert_cache.pop(symbol, None)
            if not observation_only_profile:
                rec_state = execution.state_machine.get(symbol)

            prepared = prepared_contexts.get(symbol)
            if not prepared:
                counters.inc("cycle_errors")
                logger.error("cycle_error symbol=%s err=%s", symbol, "analysis_missing", extra={"event": "cycle_error"})
                continue

            prepared_status = str(prepared.get("status") or "")
            if prepared_status == "empty_ohlcv":
                counters.inc("empty_ohlcv")
                continue
            if prepared_status != "ok":
                counters.inc("cycle_errors")
                logger.error(
                    "cycle_error symbol=%s err=%s",
                    symbol,
                    prepared.get("error") or "analysis_unavailable",
                    extra={"event": "cycle_error"},
                )
                continue

            frame = prepared["frame"]
            runtime_inputs = prepared["runtime_inputs"]
            extras = prepared["extras"]
            features = prepared["features"]

            if early_signal_learner is not None:
                early_row = early_signal_learner.resolve_with_frame(symbol=symbol, enriched=features.enriched)
                if early_row is not None:
                    if log_dataset_appends:
                        logger.info(
                            "online_early_dataset_appended symbol=%s phase=%s target_win=%s future_return=%s target_horizon=%s",
                            symbol,
                            early_row.get("signal_phase"),
                            early_row.get("target_win"),
                            early_row.get("future_return"),
                            early_row.get("target_horizon"),
                            extra={"event": "online_early_dataset_appended"},
                        )
                    if early_online_retrainer is not None and early_online_retrainer.maybe_retrain():
                        logger.info(
                            "online_early_retrain_completed dataset=%s",
                            early_online_retrainer.config.dataset_path,
                        extra={"event": "online_early_retrain_completed"},
                    )

            decision_contexts[symbol] = {
                "symbol": symbol,
                "snapshot": snapshot,
                "rec_state": rec_state,
                "prepared": prepared,
                "frame": frame,
                "runtime_inputs": runtime_inputs,
                "extras": extras,
                "features": features,
            }
        except Exception as exc:
            counters.inc("cycle_errors")
            logger.exception("cycle_error symbol=%s err=%s", symbol, exc, extra={"event": "cycle_error"})

    precomputed_early_candidates: dict[str, dict[str, object] | None] = {}

    if decision_contexts:
        if analysis_workers > 1 and len(decision_contexts) > 1:
            with ThreadPoolExecutor(max_workers=analysis_workers) as executor:
                futures = [
                    (
                        symbol,
                        executor.submit(
                            _prepare_symbol_decision,
                            symbol=symbol,
                            snapshot=context["snapshot"],
                            rec_state=context["rec_state"],
                            prepared=context["prepared"],
                            timeframe=timeframe,
                            strategy=strategy,
                        ),
                    )
                    for symbol, context in decision_contexts.items()
                ]
                for symbol, future in futures:
                    try:
                        prepared_decisions[symbol] = future.result()
                    except Exception as exc:
                        prepared_decisions[symbol] = {
                            "symbol": symbol,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
        else:
            for symbol, context in decision_contexts.items():
                prepared_decisions[symbol] = _prepare_symbol_decision(
                    symbol=symbol,
                    snapshot=context["snapshot"],
                    rec_state=context["rec_state"],
                    prepared=context["prepared"],
                    timeframe=timeframe,
                    strategy=strategy,
                )

    if signal_profile != "main":
        for symbol, decision_prepared in prepared_decisions.items():
            if str(decision_prepared.get("status") or "") != "ok":
                continue
            try:
                precomputed_early_candidates[symbol] = _build_early_signal_candidate(
                    symbol=symbol,
                    timeframe=timeframe,
                    mode=mode,
                    enriched=decision_prepared["features"].enriched,
                    intent=decision_prepared["intent"],
                )
            except Exception:
                precomputed_early_candidates[symbol] = None

    processing_symbols = sorted(
        symbols,
        key=lambda current_symbol: (
            _decision_priority(
                (prepared_decisions.get(current_symbol) or {}).get("intent"),
                precomputed_early_candidates.get(current_symbol),
            ),
            current_symbol,
        ),
    )

    for symbol in processing_symbols:
        try:
            decision_prepared = prepared_decisions.get(symbol)
            if not decision_prepared:
                continue

            prepared_status = str(decision_prepared.get("status") or "")
            if prepared_status != "ok":
                counters.inc("cycle_errors")
                logger.error(
                    "cycle_error symbol=%s err=%s",
                    symbol,
                    decision_prepared.get("error") or "decision_unavailable",
                    extra={"event": "cycle_error"},
                )
                continue

            snapshot = decision_prepared["snapshot"]
            rec_state = decision_prepared["rec_state"]
            frame = decision_prepared["frame"]
            runtime_inputs = decision_prepared["runtime_inputs"]
            extras = decision_prepared["extras"]
            features = decision_prepared["features"]
            mark_price = _as_float(decision_prepared.get("mark_price"), float(features.enriched.iloc[-1]["close"]))
            decision_prepared_ts = _as_float(decision_prepared.get("prepared_at"), 0.0)
            intent = decision_prepared["intent"]
            early_candidate = None
            if signal_profile != "main":
                early_candidate = precomputed_early_candidates.get(symbol)

            if _should_refresh_live_candidate(intent, early_candidate):
                refreshed_decision = _refresh_symbol_decision_live(
                    symbol=symbol,
                    sync=sync,
                    execution=execution,
                    feed=feed,
                    pipeline=pipeline,
                    timeframe=timeframe,
                    candles_limit=candles_limit,
                    strategy=strategy,
                )
                if refreshed_decision is not None:
                    snapshot = refreshed_decision["snapshot"]
                    rec_state = refreshed_decision["rec_state"]
                    frame = refreshed_decision["frame"]
                    runtime_inputs = refreshed_decision["runtime_inputs"]
                    extras = refreshed_decision["extras"]
                    features = refreshed_decision["features"]
                    mark_price = _as_float(
                        refreshed_decision.get("mark_price"),
                        float(features.enriched.iloc[-1]["close"]),
                    )
                    decision_prepared_ts = _as_float(refreshed_decision.get("prepared_at"), time.time())
                    intent = refreshed_decision["intent"]
                    if signal_profile != "main":
                        try:
                            early_candidate = _build_early_signal_candidate(
                                symbol=symbol,
                                timeframe=timeframe,
                                mode=mode,
                                enriched=features.enriched,
                                intent=intent,
                            )
                        except Exception:
                            early_candidate = None
            _record_parallel_strategy_intent(strategy, intent)

            try:
                rules = adapter.get_instrument_rules(symbol)
            except Exception as exc:
                counters.inc("metadata_errors")
                logger.error(
                    "symbol=%s metadata_error=%s", symbol, exc, extra={"event": "metadata_error"}
                )
                continue

            if (
                os.getenv("ENFORCE_CLEAN_PUMP_MIN", "1").strip().lower() not in {"0", "false", "off"}
                and _should_block_clean_pump(intent)
            ):
                intent = StrategyIntent(
                    symbol=symbol,
                    action=IntentAction.HOLD,
                    reason="clean_pump_below_min",
                    metadata=intent.metadata if isinstance(intent.metadata, dict) else {},
                )
            elif signal_profile == "main" and _should_block_live_main_short_continuation(intent, features.enriched):
                intent = StrategyIntent(
                    symbol=symbol,
                    action=IntentAction.HOLD,
                    reason="main_short_live_continuation",
                    metadata=intent.metadata if isinstance(intent.metadata, dict) else {},
                )
            elif signal_profile == "main" and getattr(intent, "action", None) == IntentAction.SHORT_ENTRY:
                recent_peak_price = _current_recent_peak_price(features.enriched, fallback_price=mark_price)
                current_atr = max(
                    _as_float(features.enriched.iloc[-1].get("atr"), mark_price * 0.01),
                    mark_price * 0.0015,
                    1e-8,
                )
                if _should_block_recent_main_short_reentry(
                    signal_alert_cache,
                    symbol=symbol,
                    now_ts=time.time(),
                    entry_price=mark_price,
                    recent_peak_price=recent_peak_price,
                    atr=current_atr,
                    cooldown_sec=main_reentry_cooldown_sec,
                ):
                    intent = StrategyIntent(
                        symbol=symbol,
                        action=IntentAction.HOLD,
                        reason="main_short_same_pump_reentry_blocked",
                        metadata=intent.metadata if isinstance(intent.metadata, dict) else {},
                    )

            decision = risk.evaluate(
                intent=intent,
                account=snapshot.account,
                existing_positions=[] if observation_only_profile else snapshot.positions,
                mark_price=mark_price,
                rules=rules,
            )

            if signal_profile == "early" and intent.action != IntentAction.HOLD:
                outcome = ExecutionOutcome(
                    accepted=False,
                    status="IGNORED",
                    reason="early_profile_monitor_only",
                )
            else:
                outcome = execution.execute(intent=intent, risk=decision, snapshot=snapshot, mark_price=mark_price)

            if (
                signal_profile == "main"
                and getattr(intent, "action", None) == IntentAction.SHORT_ENTRY
                and outcome.accepted
                and outcome.filled_qty > 0
            ):
                recent_peak_price = _current_recent_peak_price(features.enriched, fallback_price=mark_price)
                current_atr = max(
                    _as_float(features.enriched.iloc[-1].get("atr"), mark_price * 0.01),
                    mark_price * 0.0015,
                    1e-8,
                )
                _remember_recent_main_short_reentry(
                    signal_alert_cache,
                    symbol=symbol,
                    now_ts=time.time(),
                    entry_price=float(outcome.avg_price if outcome.avg_price > 0 else mark_price),
                    recent_peak_price=recent_peak_price,
                    atr=current_atr,
                )

            if (
                trade_learner is not None
                and mode in ("demo", "testnet", "live")
                and intent.action in (IntentAction.LONG_ENTRY, IntentAction.SHORT_ENTRY)
                and outcome.accepted
                and outcome.filled_qty > 0
            ):
                regime_name = detect_market_regime(features.enriched).value
                trade_learner.record_entry(
                    symbol=symbol,
                    side="LONG" if intent.action == IntentAction.LONG_ENTRY else "SHORT",
                    market_regime=regime_name,
                    entry_price=float(outcome.avg_price if outcome.avg_price > 0 else mark_price),
                    qty=float(outcome.filled_qty),
                    entry_ts=time.time(),
                    features=dict(features.row.values),
                )

            dataset_row = None
            if (
                trade_learner is not None
                and mode in ("demo", "testnet", "live")
                and intent.action in (IntentAction.EXIT_LONG, IntentAction.EXIT_SHORT)
                and outcome.accepted
                and outcome.filled_qty > 0
            ):
                dataset_row = trade_learner.record_exit(
                    symbol=symbol,
                    exit_ts=time.time(),
                    realized_pnl=float(outcome.realized_pnl),
                    qty=float(outcome.filled_qty),
                )
                if dataset_row is not None:
                    if log_dataset_appends:
                        logger.info(
                            "online_dataset_appended symbol=%s target_win=%s future_return=%s target_horizon=%s",
                            symbol,
                            dataset_row.get("target_win"),
                            dataset_row.get("future_return"),
                            dataset_row.get("target_horizon"),
                            extra={"event": "online_dataset_appended"},
                        )
                    if online_retrainer is not None and online_retrainer.maybe_retrain():
                        logger.info(
                            "online_retrain_completed dataset=%s",
                            online_retrainer.config.dataset_path,
                            extra={"event": "online_retrain_completed"},
                        )

            if intent.action in (IntentAction.EXIT_LONG, IntentAction.EXIT_SHORT) and outcome.filled_qty > 0:
                risk.record_trade_result(outcome.realized_pnl, stopped_out=outcome.stopped_out)

            counters.inc("signals_total")
            counters.inc(f"intent_{intent.action.value.lower()}")
            counters.inc(f"exec_{outcome.status.lower()}")
            layer_trace = intent.metadata.get("layer_trace", {}) if isinstance(intent.metadata, dict) else {}
            layer_failed = intent.metadata.get("layer_failed", "") if isinstance(intent.metadata, dict) else ""
            layer4 = {}
            regime_diag = {}
            if isinstance(layer_trace, dict):
                layer4 = (
                    layer_trace.get("layers", {})
                    .get("layer4_fake_filter", {})
                    .get("details", {})
                )
                regime_diag = (
                    layer_trace.get("layers", {})
                    .get("regime_filter", {})
                    .get("details", {})
                )
            sentiment_mode = layer4.get("sentiment_source", "n/a") if isinstance(layer4, dict) else "n/a"
            sentiment_degraded = False
            sentiment_quality = "n/a"
            regime_news_quality = "n/a"
            if isinstance(layer4, dict):
                sentiment_degraded = bool(float(layer4.get("degraded_mode", 0.0) or 0.0))
                sentiment_quality = (
                    layer4.get("source_flags", {}).get("sentiment_quality")
                    or layer4.get("source_quality", {}).get("sentiment")
                    or "n/a"
                )
            if isinstance(regime_diag, dict):
                regime_news_quality = regime_diag.get("source_flags", {}).get("news_quality") or "n/a"

            log_decision = True
            if intent.action == IntentAction.HOLD and str(outcome.status).upper() == "IGNORED":
                if not log_hold_decisions:
                    log_decision = False
                else:
                    decision_cooldown_sec = max(
                        60,
                        _as_int(os.getenv("DECISION_LOG_COOLDOWN_SEC", "1200"), 1200),
                    )
                    decision_key = "|".join(
                        [
                            str(intent.action.value),
                            str(decision.reason),
                            str(outcome.reason),
                            str(layer_failed),
                            str(regime_news_quality),
                        ]
                    )
                    now_ts = time.time()
                    log_decision = _should_emit_cached_alert(
                        decision_log_cache,
                        symbol=symbol,
                        key=decision_key,
                        now_ts=now_ts,
                        cooldown_sec=decision_cooldown_sec,
                    )
                    if log_decision:
                        _remember_cached_alert(
                            decision_log_cache,
                            symbol=symbol,
                            key=decision_key,
                            now_ts=now_ts,
                            cooldown_sec=decision_cooldown_sec,
                        )

            if log_decision:
                logger.info(
                    "symbol=%s state=%s intent=%s risk=%s exec=%s reason=%s layer_failed=%s sentiment_mode=%s sentiment_quality=%s regime_news_quality=%s sentiment_degraded=%s",
                    symbol,
                    rec_state.state.value,
                    intent.action.value,
                    decision.reason,
                    outcome.status,
                    outcome.reason,
                    layer_failed,
                    sentiment_mode,
                    sentiment_quality,
                    regime_news_quality,
                    sentiment_degraded,
                    extra={"event": "decision"},
                )

            early_state = early_signal_state.get(symbol, {})
            active_phase = str(early_state.get("active_phase") or "")
            last_emitted_phase = str(early_state.get("last_emitted_phase") or "")
            cooldown_until_ts = _as_float(early_state.get("cooldown_until_ts"), 0.0)
            last_emitted_ts = _as_float(early_state.get("last_emitted_ts"), 0.0)
            inactive_cycles = _as_int(early_state.get("inactive_cycles"), 0)
            early_cooldown_sec = max(60, _as_int(os.getenv("EARLY_SIGNAL_COOLDOWN_SEC", "1800"), 1800))
            signature_cooldown_until_ts = _as_float(early_state.get("signature_cooldown_until_ts"), 0.0)
            last_signature = str(early_state.get("last_signature") or "")
            early_signature_cooldown_sec = max(
                early_cooldown_sec,
                _as_int(os.getenv("EARLY_SIGNAL_SIGNATURE_COOLDOWN_SEC", "2700"), 2700),
            )
            cycle_ts = time.time()
            emitted_early_phase = ""

            if intent.action in (IntentAction.LONG_ENTRY, IntentAction.SHORT_ENTRY):
                phase = str(early_candidate.get("phase") or "") if isinstance(early_candidate, Mapping) else ""
                if phase:
                    in_cooldown = cycle_ts < cooldown_until_ts and phase == last_emitted_phase
                    phase_upgraded = _phase_rank(phase) > _phase_rank(active_phase)
                    if in_cooldown and not phase_upgraded:
                        early_signal_stats["suppressed_by_cooldown"] = int(
                            early_signal_stats.get("suppressed_by_cooldown", 0)
                        ) + 1
                    elif phase != active_phase:
                        refreshed_decision = _refresh_symbol_decision_live(
                            symbol=symbol,
                            sync=sync,
                            execution=execution,
                            feed=feed,
                            pipeline=pipeline,
                            timeframe=timeframe,
                            candles_limit=candles_limit,
                            strategy=strategy,
                        )
                        if refreshed_decision is not None:
                            snapshot = refreshed_decision["snapshot"]
                            rec_state = refreshed_decision["rec_state"]
                            frame = refreshed_decision["frame"]
                            runtime_inputs = refreshed_decision["runtime_inputs"]
                            extras = refreshed_decision["extras"]
                            features = refreshed_decision["features"]
                            mark_price = _as_float(
                                refreshed_decision.get("mark_price"),
                                float(features.enriched.iloc[-1]["close"]),
                            )
                            decision_prepared_ts = _as_float(refreshed_decision.get("prepared_at"), time.time())
                            intent = refreshed_decision["intent"]
                            try:
                                early_candidate = _build_early_signal_candidate(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    mode=mode,
                                    enriched=features.enriched,
                                    intent=intent,
                                )
                            except Exception:
                                early_candidate = None
                        if isinstance(early_candidate, Mapping) and str(early_candidate.get("phase") or "") == phase:
                            if _needs_pre_alert_refresh(
                                prepared_ts=decision_prepared_ts,
                                timeframe=timeframe,
                            ):
                                refreshed_decision = _refresh_symbol_decision_live(
                                    symbol=symbol,
                                    sync=sync,
                                    execution=execution,
                                    feed=feed,
                                    pipeline=pipeline,
                                    timeframe=timeframe,
                                    candles_limit=candles_limit,
                                    strategy=strategy,
                                )
                                if refreshed_decision is not None:
                                    snapshot = refreshed_decision["snapshot"]
                                    rec_state = refreshed_decision["rec_state"]
                                    frame = refreshed_decision["frame"]
                                    runtime_inputs = refreshed_decision["runtime_inputs"]
                                    extras = refreshed_decision["extras"]
                                    features = refreshed_decision["features"]
                                    mark_price = _as_float(
                                        refreshed_decision.get("mark_price"),
                                        float(features.enriched.iloc[-1]["close"]),
                                    )
                                    decision_prepared_ts = _as_float(
                                        refreshed_decision.get("prepared_at"), time.time()
                                    )
                                    intent = refreshed_decision["intent"]
                                    try:
                                        early_candidate = _build_early_signal_candidate(
                                            symbol=symbol,
                                            timeframe=timeframe,
                                            mode=mode,
                                            enriched=features.enriched,
                                            intent=intent,
                                        )
                                    except Exception:
                                        early_candidate = None
                            if not isinstance(early_candidate, Mapping) or str(early_candidate.get("phase") or "") != phase:
                                continue
                            early_signature = _build_early_candidate_signature(early_candidate, features.enriched)
                            if (
                                early_signature
                                and early_signature == last_signature
                                and cycle_ts < signature_cooldown_until_ts
                            ):
                                early_signal_stats["suppressed_by_cooldown"] = int(
                                    early_signal_stats.get("suppressed_by_cooldown", 0)
                                ) + 1
                                continue
                            reply_markup = build_symbol_copy_reply_markup(symbol)
                            chart_bytes = _build_alert_chart(
                                symbol=symbol,
                                timeframe=timeframe,
                                enriched=features.enriched,
                                side="SHORT",
                                entry=_as_float(early_candidate.get("entry"), mark_price),
                                tp=_as_float(early_candidate.get("tp"), mark_price * 0.99),
                                sl=_as_float(early_candidate.get("sl"), mark_price * 1.01),
                                show_trade_levels=True,
                                show_liquidation_map=False,
                                timeframe_label=_format_chart_timeframe_label(timeframe),
                            )
                            attempted = 0
                            sent = 0
                            if chart_bytes:
                                a1, s1 = _send_photo_alerts(
                                    alerters,
                                    str(early_candidate.get("caption") or ""),
                                    chart_bytes,
                                    filename=f"{symbol.lower()}_early_1m.png",
                                    reply_markup=reply_markup,
                                )
                                attempted += a1
                                sent += s1
                            else:
                                a1, s1 = _send_alerts(
                                    alerters,
                                    str(early_candidate.get("caption") or ""),
                                    reply_markup=reply_markup,
                                )
                                attempted += a1
                                sent += s1

                            context_chart_bytes = _build_higher_timeframe_chart(
                                symbol=symbol,
                                side="SHORT",
                                entry=_as_float(early_candidate.get("entry"), mark_price),
                                tp=_as_float(early_candidate.get("tp"), mark_price * 0.99),
                                sl=_as_float(early_candidate.get("sl"), mark_price * 1.01),
                                feed=feed,
                                pipeline=pipeline,
                                runtime_extras=extras,
                            )
                            if context_chart_bytes:
                                a2, s2 = _send_photo_alerts(
                                    alerters,
                                    _build_clean_context_chart_caption(
                                        symbol,
                                        stage_label="\u0420\u0410\u041d\u041d\u0418\u0419 \u0428\u041e\u0420\u0422: HTF \u041a\u041e\u041d\u0422\u0415\u041a\u0421\u0422",
                                        timeframe_label=_format_chart_timeframe_label(
                                            os.getenv("BOT_ALERT_CONTEXT_TIMEFRAME", "240")
                                        ),
                                    ),
                                    context_chart_bytes,
                                    filename=f"{symbol.lower()}_early_4h.png",
                                    reply_markup=reply_markup,
                                )
                                attempted += a2
                                sent += s2
                            _log_alert_delivery(
                                logger,
                                event="early_signal_alert_delivery",
                                attempted=attempted,
                                sent=sent,
                                skip_reason="no_alerters_configured" if attempted == 0 else "",
                            )
                            early_signal_stats["watch_sent" if phase == "WATCH" else "setup_sent"] = int(
                                early_signal_stats.get("watch_sent" if phase == "WATCH" else "setup_sent", 0)
                            ) + 1
                            if phase == "SETUP" and active_phase == "WATCH":
                                early_signal_stats["watch_to_setup_promoted"] = int(
                                    early_signal_stats.get("watch_to_setup_promoted", 0)
                                ) + 1
                            if early_signal_learner is not None:
                                signal_price = _as_float(early_candidate.get("entry"), mark_price)
                                tp_price = _as_float(early_candidate.get("tp"), signal_price * 0.99)
                                sl_price = _as_float(early_candidate.get("sl"), signal_price * 1.01)
                                downside_target = max((signal_price - tp_price) / max(signal_price, 1e-8), 0.0)
                                upside_risk = max((sl_price - signal_price) / max(signal_price, 1e-8), 0.0)
                                early_signal_learner.record_signal(
                                    symbol=symbol,
                                    phase=phase,
                                    market_regime=detect_market_regime(features.enriched).value,
                                    signal_price=signal_price,
                                    signal_ts=cycle_ts,
                                    signal_bar_ts=features.enriched.index[-1],
                                    features=dict(features.row.values),
                                    horizon_bars=42 if phase == "WATCH" else 30,
                                    success_move_pct=min(max(downside_target * 0.55, 0.004), 0.03),
                                    failure_move_pct=min(max(upside_risk * 0.60, 0.0035), 0.025),
                                )
                            early_signal_state[symbol] = {
                                "active_phase": phase,
                                "last_emitted_phase": phase,
                                "cooldown_until_ts": cycle_ts + early_cooldown_sec,
                                "last_emitted_ts": cycle_ts,
                                "inactive_cycles": 0,
                                "last_signature": early_signature,
                                "signature_cooldown_until_ts": cycle_ts + early_signature_cooldown_sec,
                            }
                            emitted_early_phase = phase

            if intent.action in (IntentAction.LONG_ENTRY, IntentAction.SHORT_ENTRY):
                if active_phase or emitted_early_phase:
                    early_signal_stats["entry_confirmed"] = int(early_signal_stats.get("entry_confirmed", 0)) + 1
                early_signal_state[symbol] = {
                    "active_phase": "",
                    "last_emitted_phase": emitted_early_phase or last_emitted_phase or active_phase,
                    "cooldown_until_ts": max(cooldown_until_ts, cycle_ts + early_cooldown_sec),
                    "last_emitted_ts": cycle_ts,
                    "inactive_cycles": 0,
                    "last_signature": last_signature,
                    "signature_cooldown_until_ts": max(
                        signature_cooldown_until_ts,
                        cycle_ts + early_signature_cooldown_sec,
                    ),
                }
            elif early_candidate is None:
                if active_phase:
                    invalidation_grace_sec = _as_int(
                        os.getenv("EARLY_INVALIDATION_GRACE_SEC", "180"),
                        180,
                    )
                    invalidation_miss_limit = _as_int(
                        os.getenv("EARLY_INVALIDATION_MISS_LIMIT", "3"),
                        3,
                    )
                    inactive_cycles += 1
                    early_signal_state[symbol] = {
                        "active_phase": active_phase,
                        "last_emitted_phase": last_emitted_phase,
                        "cooldown_until_ts": cooldown_until_ts,
                        "last_emitted_ts": last_emitted_ts,
                        "inactive_cycles": inactive_cycles,
                        "last_signature": last_signature,
                        "signature_cooldown_until_ts": signature_cooldown_until_ts,
                    }
                    if cycle_ts - last_emitted_ts < invalidation_grace_sec:
                        continue
                    if inactive_cycles < invalidation_miss_limit:
                        continue
                    invalidation_text = build_early_invalidation_text(
                        symbol=symbol,
                        timeframe=timeframe,
                        mode=mode,
                        reason="\u0441\u0446\u0435\u043d\u0430\u0440\u0438\u0439 \u0441\u043b\u043e\u043c\u0430\u043d \u0438\u043b\u0438 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u0438\u0435 \u043d\u0435 \u043f\u0440\u0438\u0448\u043b\u043e",
                    )
                    attempted, sent = _send_alerts(
                        alerters,
                        invalidation_text,
                        reply_markup=build_symbol_copy_reply_markup(symbol),
                    )
                    _log_alert_delivery(
                        logger,
                        event="early_signal_invalidated_delivery",
                        attempted=attempted,
                        sent=sent,
                        skip_reason="no_alerters_configured" if attempted == 0 else "",
                    )
                    early_signal_stats["invalidated"] = int(early_signal_stats.get("invalidated", 0)) + 1
                    early_signal_state[symbol] = {
                        "active_phase": "",
                        "last_emitted_phase": last_emitted_phase or active_phase,
                        "cooldown_until_ts": cycle_ts + early_cooldown_sec,
                        "last_emitted_ts": cycle_ts,
                        "inactive_cycles": 0,
                        "last_signature": last_signature,
                        "signature_cooldown_until_ts": max(
                            signature_cooldown_until_ts,
                            cycle_ts + early_signature_cooldown_sec,
                        ),
                    }
            else:
                phase = str(early_candidate.get("phase") or "")
                if phase:
                    in_cooldown = cycle_ts < cooldown_until_ts and phase == last_emitted_phase
                    phase_upgraded = _phase_rank(phase) > _phase_rank(active_phase)
                    if in_cooldown and not phase_upgraded:
                        early_signal_stats["suppressed_by_cooldown"] = int(
                            early_signal_stats.get("suppressed_by_cooldown", 0)
                        ) + 1
                    elif phase != active_phase:
                        refreshed_decision = _refresh_symbol_decision_live(
                            symbol=symbol,
                            sync=sync,
                            execution=execution,
                            feed=feed,
                            pipeline=pipeline,
                            timeframe=timeframe,
                            candles_limit=candles_limit,
                            strategy=strategy,
                        )
                        if refreshed_decision is not None:
                            snapshot = refreshed_decision["snapshot"]
                            rec_state = refreshed_decision["rec_state"]
                            frame = refreshed_decision["frame"]
                            runtime_inputs = refreshed_decision["runtime_inputs"]
                            extras = refreshed_decision["extras"]
                            features = refreshed_decision["features"]
                            mark_price = _as_float(
                                refreshed_decision.get("mark_price"),
                                float(features.enriched.iloc[-1]["close"]),
                            )
                            decision_prepared_ts = _as_float(refreshed_decision.get("prepared_at"), time.time())
                            intent = refreshed_decision["intent"]
                            try:
                                early_candidate = _build_early_signal_candidate(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    mode=mode,
                                    enriched=features.enriched,
                                    intent=intent,
                                )
                            except Exception:
                                early_candidate = None
                        if not isinstance(early_candidate, Mapping):
                            continue
                        if str(early_candidate.get("phase") or "") != phase:
                            continue
                        early_signature = _build_early_candidate_signature(early_candidate, features.enriched)
                        if (
                            early_signature
                            and early_signature == last_signature
                            and cycle_ts < signature_cooldown_until_ts
                        ):
                            early_signal_stats["suppressed_by_cooldown"] = int(
                                early_signal_stats.get("suppressed_by_cooldown", 0)
                            ) + 1
                            continue
                        reply_markup = build_symbol_copy_reply_markup(symbol)
                        chart_bytes = _build_alert_chart(
                            symbol=symbol,
                            timeframe=timeframe,
                            enriched=features.enriched,
                            side="SHORT",
                            entry=_as_float(early_candidate.get("entry"), mark_price),
                            tp=_as_float(early_candidate.get("tp"), mark_price * 0.99),
                            sl=_as_float(early_candidate.get("sl"), mark_price * 1.01),
                            show_trade_levels=True,
                            show_liquidation_map=False,
                            timeframe_label=_format_chart_timeframe_label(timeframe),
                        )
                        attempted = 0
                        sent = 0
                        if chart_bytes:
                            a1, s1 = _send_photo_alerts(
                                alerters,
                                str(early_candidate.get("caption") or ""),
                                chart_bytes,
                                filename=f"{symbol.lower()}_early_1m.png",
                                reply_markup=reply_markup,
                            )
                            attempted += a1
                            sent += s1
                        else:
                            a1, s1 = _send_alerts(
                                alerters,
                                str(early_candidate.get("caption") or ""),
                                reply_markup=reply_markup,
                            )
                            attempted += a1
                            sent += s1

                        context_chart_bytes = _build_higher_timeframe_chart(
                            symbol=symbol,
                            side="SHORT",
                            entry=_as_float(early_candidate.get("entry"), mark_price),
                            tp=_as_float(early_candidate.get("tp"), mark_price * 0.99),
                            sl=_as_float(early_candidate.get("sl"), mark_price * 1.01),
                            feed=feed,
                            pipeline=pipeline,
                            runtime_extras=extras,
                        )
                        if context_chart_bytes:
                            a2, s2 = _send_photo_alerts(
                                alerters,
                                _build_clean_context_chart_caption(
                                    symbol,
                                    stage_label="\u0420\u0410\u041d\u041d\u0418\u0419 \u0428\u041e\u0420\u0422: HTF \u041a\u041e\u041d\u0422\u0415\u041a\u0421\u0422",
                                    timeframe_label=_format_chart_timeframe_label(
                                        os.getenv("BOT_ALERT_CONTEXT_TIMEFRAME", "240")
                                    ),
                                ),
                                context_chart_bytes,
                                filename=f"{symbol.lower()}_early_4h.png",
                                reply_markup=reply_markup,
                            )
                            attempted += a2
                            sent += s2
                        _log_alert_delivery(
                            logger,
                            event="early_signal_alert_delivery",
                            attempted=attempted,
                            sent=sent,
                            skip_reason="no_alerters_configured" if attempted == 0 else "",
                        )
                        early_signal_stats["watch_sent" if phase == "WATCH" else "setup_sent"] = int(
                            early_signal_stats.get("watch_sent" if phase == "WATCH" else "setup_sent", 0)
                        ) + 1
                        if phase == "SETUP" and active_phase == "WATCH":
                            early_signal_stats["watch_to_setup_promoted"] = int(
                                early_signal_stats.get("watch_to_setup_promoted", 0)
                            ) + 1
                        if early_signal_learner is not None:
                            signal_price = _as_float(early_candidate.get("entry"), mark_price)
                            tp_price = _as_float(early_candidate.get("tp"), signal_price * 0.99)
                            sl_price = _as_float(early_candidate.get("sl"), signal_price * 1.01)
                            downside_target = max((signal_price - tp_price) / max(signal_price, 1e-8), 0.0)
                            upside_risk = max((sl_price - signal_price) / max(signal_price, 1e-8), 0.0)
                            early_signal_learner.record_signal(
                                symbol=symbol,
                                phase=phase,
                                market_regime=detect_market_regime(features.enriched).value,
                                signal_price=signal_price,
                                signal_ts=cycle_ts,
                                signal_bar_ts=features.enriched.index[-1],
                                features=dict(features.row.values),
                                horizon_bars=42 if phase == "WATCH" else 30,
                                success_move_pct=min(max(downside_target * 0.55, 0.004), 0.03),
                                failure_move_pct=min(max(upside_risk * 0.60, 0.0035), 0.025),
                            )
                        early_signal_state[symbol] = {
                            "active_phase": phase,
                            "last_emitted_phase": phase,
                            "cooldown_until_ts": cycle_ts + early_cooldown_sec,
                            "last_emitted_ts": cycle_ts,
                            "inactive_cycles": 0,
                            "last_signature": early_signature,
                            "signature_cooldown_until_ts": cycle_ts + early_signature_cooldown_sec,
                        }

            if intent.action in (IntentAction.LONG_ENTRY, IntentAction.SHORT_ENTRY) and outcome.accepted:
                refreshed_decision = _refresh_symbol_decision_live(
                    symbol=symbol,
                    sync=sync,
                    execution=execution,
                    feed=feed,
                    pipeline=pipeline,
                    timeframe=timeframe,
                    candles_limit=candles_limit,
                    strategy=strategy,
                )
                if refreshed_decision is not None:
                    snapshot = refreshed_decision["snapshot"]
                    rec_state = refreshed_decision["rec_state"]
                    frame = refreshed_decision["frame"]
                    runtime_inputs = refreshed_decision["runtime_inputs"]
                    extras = refreshed_decision["extras"]
                    features = refreshed_decision["features"]
                    mark_price = _as_float(
                        refreshed_decision.get("mark_price"),
                        float(features.enriched.iloc[-1]["close"]),
                    )
                    decision_prepared_ts = _as_float(refreshed_decision.get("prepared_at"), time.time())
                    intent = refreshed_decision["intent"]
                    decision = risk.evaluate(
                        intent=intent,
                        account=snapshot.account,
                        existing_positions=snapshot.positions,
                        mark_price=mark_price,
                        rules=rules,
                    )
                    if intent.action not in (IntentAction.LONG_ENTRY, IntentAction.SHORT_ENTRY) or not decision.approved:
                        continue
                signal_signature = _build_main_signal_signature(
                    symbol=symbol,
                    intent=intent,
                    enriched=features.enriched,
                    mark_price=mark_price,
                )
                signal_now_ts = time.time()
                if not _should_emit_cached_alert(
                    signal_alert_cache,
                    symbol=symbol,
                    key=signal_signature,
                    now_ts=signal_now_ts,
                    cooldown_sec=signal_alert_cooldown_sec,
                ):
                    continue
                if not _should_emit_symbol_rate_limited_alert(
                    signal_alert_cache,
                    symbol=symbol,
                    now_ts=signal_now_ts,
                ):
                    continue
                action_label = ("\u0428\u041e\u0420\u0422 \u0421\u0418\u0413\u041d\u0410\u041b" if intent.action == IntentAction.SHORT_ENTRY else "\u041b\u041e\u041d\u0413 \u0421\u0418\u0413\u041d\u0410\u041b")
                caption = build_signal_caption(
                    symbol=symbol,
                    timeframe=timeframe,
                    mode=mode,
                    action_label=action_label,
                    entry=mark_price,
                    tp=float(intent.take_profit or mark_price),
                    sl=float(intent.stop_loss or mark_price),
                    confidence=float(intent.confidence),
                    reason=intent.reason,
                    trace_meta=intent.metadata if isinstance(intent.metadata, Mapping) else {},
                    enriched=features.enriched,
                )
                reply_markup = build_symbol_copy_reply_markup(symbol)
                chart_bytes = _build_alert_chart(
                    symbol=symbol,
                    timeframe=timeframe,
                    enriched=features.enriched,
                    side="SHORT" if intent.action == IntentAction.SHORT_ENTRY else "LONG",
                    entry=mark_price,
                    tp=float(intent.take_profit or mark_price),
                    sl=float(intent.stop_loss or mark_price),
                    show_liquidation_map=False,
                    timeframe_label=_format_chart_timeframe_label(timeframe),
                )
                attempted = 0
                sent = 0
                if chart_bytes:
                    a1, s1 = _send_photo_alerts(
                        alerters,
                        caption,
                        chart_bytes,
                        filename=f"{symbol.lower()}_signal_1m.png",
                        reply_markup=reply_markup,
                    )
                    attempted += a1
                    sent += s1
                else:
                    a1, s1 = _send_alerts(alerters, caption, reply_markup=reply_markup)
                    attempted += a1
                    sent += s1

                context_chart_bytes = _build_higher_timeframe_chart(
                    symbol=symbol,
                    side="SHORT" if intent.action == IntentAction.SHORT_ENTRY else "LONG",
                    entry=mark_price,
                    tp=float(intent.take_profit or mark_price),
                    sl=float(intent.stop_loss or mark_price),
                    feed=feed,
                    pipeline=pipeline,
                    runtime_extras=extras,
                )
                if context_chart_bytes:
                    a2, s2 = _send_photo_alerts(
                        alerters,
                        _build_clean_context_chart_caption(
                            symbol,
                            stage_label=(
                                "\u0428\u041e\u0420\u0422 \u0421\u0418\u0413\u041d\u0410\u041b: HTF \u041a\u041e\u041d\u0422\u0415\u041a\u0421\u0422"
                                if intent.action == IntentAction.SHORT_ENTRY
                                else "\u041b\u041e\u041d\u0413 \u0421\u0418\u0413\u041d\u0410\u041b: HTF \u041a\u041e\u041d\u0422\u0415\u041a\u0421\u0422"
                            ),
                            timeframe_label=_format_chart_timeframe_label(
                                os.getenv("BOT_ALERT_CONTEXT_TIMEFRAME", "240")
                            ),
                        ),
                        context_chart_bytes,
                        filename=f"{symbol.lower()}_signal_4h.png",
                        reply_markup=reply_markup,
                    )
                    attempted += a2
                    sent += s2
                _log_alert_delivery(
                    logger,
                    event="signal_alert_delivery",
                    attempted=attempted,
                    sent=sent,
                    skip_reason="no_alerters_configured" if attempted == 0 else "",
                )
                _remember_cached_alert(
                    signal_alert_cache,
                    symbol=symbol,
                    key=signal_signature,
                    now_ts=signal_now_ts,
                    cooldown_sec=signal_alert_cooldown_sec,
                )
                _remember_symbol_rate_limited_alert(
                    signal_alert_cache,
                    symbol=symbol,
                    now_ts=signal_now_ts,
                    cooldown_sec=signal_symbol_cooldown_sec,
                )

        except Exception as exc:
            counters.inc("cycle_errors")
            logger.exception("cycle_error symbol=%s err=%s", symbol, exc, extra={"event": "cycle_error"})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bybit Futures Bot V2")
    parser.add_argument("--strategy", choices=["layered", "hold"], default="layered")
    parser.add_argument("--loop", action="store_true", help="Run continuous loop")
    parser.add_argument("--signal-profile", choices=["both", "main", "early"], default=None)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    logger = setup_logging("INFO")
    _load_dotenv_file()
    requested_signal_profile = args.signal_profile or os.getenv("BOT_SIGNAL_PROFILE")
    if not requested_signal_profile and str(os.getenv("BOT_RUNTIME_MODE", "")).strip().lower() == "demo":
        requested_signal_profile = "main"
    signal_profile = _resolve_signal_profile(requested_signal_profile)
    os.environ["BOT_SIGNAL_PROFILE"] = signal_profile

    try:
        cfg = load_runtime_config()
    except ConfigError as exc:
        logger.error("startup_config_error=%s", exc, extra={"event": "config_error"})
        return 2

    try:
        from trading.features.pipeline import FeaturePipeline
        from trading.market_data.feed import MarketDataFeed
        from trading.market_data.reconciliation import ExchangeReconciler
        from trading.market_data.ws_reconciliation import ExchangeSyncService
        from ai.online_early_learning import EarlySignalOutcomeLearner
        from ai.online_demo_learning import TradeExecutionLearner
        from ai.retrain_online import OnlineRetrainConfig, OnlineRetrainer
        from core.settings import load_settings
    except Exception as exc:
        logger.error("startup_dependency_error=%s", exc, extra={"event": "dependency_error"})
        return 2

    adapter = BybitAdapter(cfg.adapter)
    adapter.set_ws_symbols(cfg.symbols)

    feed = MarketDataFeed(base_url=resolve_public_http_base_url(testnet=bool(cfg.adapter.testnet)))
    reconciler = ExchangeReconciler(adapter)
    sync = ExchangeSyncService(
        reconciler,
        poll_interval_sec=cfg.ws_poll_interval_sec,
        max_event_staleness_sec=cfg.ws_event_staleness_sec,
        forced_reconnect_cooldown_sec=max(cfg.ws_poll_interval_sec, 15),
    )
    runtime_store = RuntimeStore(cfg.runtime_db_path)

    state_machine = StateMachine(persistence=runtime_store)
    risk = RiskEngine(cfg.risk_limits, persistence=runtime_store)
    execution = ExecutionEngine(
        adapter=adapter,
        state_machine=state_machine,
        hedge_mode=cfg.adapter.hedge_mode,
        stop_loss_required=cfg.risk_limits.require_stop_loss,
        require_reconciliation=cfg.flags.reconciliation_required,
        stop_attach_grace_sec=cfg.stop_attach_grace_sec,
        stale_open_order_sec=cfg.stale_open_order_sec,
        max_exchange_retries=cfg.max_exchange_retries,
        persistence=runtime_store,
    )
    pipeline = FeaturePipeline()
    strategy = _build_strategy(args.strategy)
    counters = MetricsCounter()
    alerters = _build_alerters(cfg)
    app_settings = load_settings()
    trade_learner = None
    online_retrainer = None
    early_signal_learner = None
    early_online_retrainer = None
    learning_supported_mode = cfg.mode in ("demo", "testnet", "live")
    if learning_supported_mode and app_settings.ml.online_retrain_enabled:
        dataset_path = _profiled_dataset_path(app_settings.ml.online_dataset_path, signal_profile)
        model_dir = _profiled_model_dir(app_settings.ml.model_dir, signal_profile)
        pending_path = _profiled_pending_path(f"data/runtime/{cfg.mode}_online_pending.json", signal_profile)
        timeframe_minutes = _timeframe_to_minutes(cfg.timeframe)

        if signal_profile in ("both", "main"):
            trade_learner = TradeExecutionLearner(
                dataset_path=dataset_path,
                pending_path=pending_path,
                timeframe_minutes=timeframe_minutes,
            )
            online_retrainer = OnlineRetrainer(
                OnlineRetrainConfig(
                    dataset_path=dataset_path,
                    model_dir=model_dir,
                    retrain_interval_sec=app_settings.ml.online_retrain_interval_sec,
                    min_new_rows=app_settings.ml.online_retrain_min_rows,
                )
            )

        if signal_profile == "early":
            early_signal_learner = EarlySignalOutcomeLearner(
                dataset_path=dataset_path,
                pending_path=pending_path,
                timeframe_minutes=timeframe_minutes,
            )
            early_online_retrainer = OnlineRetrainer(
                OnlineRetrainConfig(
                    dataset_path=dataset_path,
                    model_dir=model_dir,
                    retrain_interval_sec=app_settings.ml.online_retrain_interval_sec,
                    min_new_rows=app_settings.ml.online_retrain_min_rows,
                )
            )

    startup_state = _startup_reconcile(
        symbols=cfg.symbols,
        sync=sync,
        state_machine=state_machine,
        execution=execution,
        signal_profile=signal_profile,
        logger=logger,
    )
    startup_state_summary = _summarize_startup_state(startup_state)
    maintenance = runtime_store.maintenance()
    inflight = runtime_store.load_open_inflight_intents()
    logger.info(
        "startup_safety mode=%s testnet=%s demo=%s dry_run=%s symbols=%d db=%s inflight=%d state_counts=%s issue_count=%d issue_samples=%s maintenance=%s live_cap=%s",
        cfg.mode,
        cfg.adapter.testnet,
        cfg.adapter.demo,
        cfg.adapter.dry_run,
        len(cfg.symbols),
        cfg.runtime_db_path,
        len(inflight),
        startup_state_summary.get("state_counts", {}),
        startup_state_summary.get("issue_count", 0),
        startup_state_summary.get("issue_samples", []),
        maintenance,
        cfg.live_startup_max_notional_usdt,
        extra={"event": "startup_safety"},
    )
    auth_key_source = "missing"
    if _profiled_env_first_nonempty(
        signal_profile,
        "BYBIT_DEMO_API_KEY",
        "BYBIT_DEMO_KEY",
        "DEMO_BYBIT_API_KEY",
        "DEMO_API_KEY",
        "DEMO_KEY",
        "BYBIT_API_KEY_DEMO",
        "API_KEY_DEMO",
    ):
        auth_key_source = "demo_specific"
    elif _profiled_env_first_nonempty(
        signal_profile,
        "BYBIT_TESTNET_API_KEY",
        "BYBIT_TESTNET_KEY",
        "TESTNET_BYBIT_API_KEY",
    ):
        auth_key_source = "testnet_specific"
    elif _profiled_env_first_nonempty(
        signal_profile,
        "BYBIT_MAINNET_API_KEY",
        "BYBIT_MAINNET_KEY",
        "MAINNET_BYBIT_API_KEY",
    ):
        auth_key_source = "mainnet_specific"
    elif _profiled_env_first_nonempty(signal_profile, "BYBIT_API_KEY", "BYBIT_KEY", "API_KEY"):
        auth_key_source = "generic_for_demo" if cfg.mode == "demo" else "generic"
    logger.info(
        "trade_learning mode=%s profile=%s enabled=%s learner_active=%s early_learner_active=%s api_keys_present=%s key_source=%s dataset=%s model_dir=%s",
        cfg.mode,
        signal_profile,
        bool(app_settings.ml.online_retrain_enabled),
        bool(trade_learner is not None),
        bool(early_signal_learner is not None),
        bool(cfg.adapter.api_key and cfg.adapter.api_secret),
        auth_key_source,
        dataset_path if learning_supported_mode and app_settings.ml.online_retrain_enabled else app_settings.ml.online_dataset_path,
        model_dir if learning_supported_mode and app_settings.ml.online_retrain_enabled else app_settings.ml.model_dir,
        extra={"event": "trade_learning"},
    )
    if learning_supported_mode and trade_learner is not None and getattr(adapter, "private_auth_invalid", False):
        logger.error(
            "trade_learning_disabled reason=private_auth_invalid mode=%s detail=%s",
            cfg.mode,
            getattr(adapter, "private_auth_invalid_reason", ""),
            extra={"event": "trade_learning_disabled"},
        )
        trade_learner = None
        online_retrainer = None
    if app_settings.ml.online_retrain_enabled and cfg.mode == "paper":
        logger.warning(
            "trade_learning_disabled reason=paper_mode requires=demo_or_testnet_or_live",
            extra={"event": "trade_learning_disabled"},
        )

    startup_text = (
        f"<b>\u0421\u0422\u0410\u0420\u0422 \u0411\u041e\u0422\u0410</b>\n"
        f"\u0420\u0435\u0436\u0438\u043c: {cfg.mode}\n"
        f"Demo: {'\u0434\u0430' if cfg.adapter.demo else '\u043d\u0435\u0442'}\n"
        f"Testnet: {'\u0434\u0430' if cfg.adapter.testnet else '\u043d\u0435\u0442'}\n"
        f"Dry run: {'\u0434\u0430' if cfg.adapter.dry_run else '\u043d\u0435\u0442'}\n"
        f"\u0421\u0438\u043c\u0432\u043e\u043b\u043e\u0432: {len(cfg.symbols)}"
    )
    startup_text = "\n".join(_normalize_human_text(line) for line in startup_text.splitlines())
    attempted, sent = _send_alerts(alerters, startup_text)
    _log_alert_delivery(
        logger,
        event="startup_alert_delivery",
        attempted=attempted,
        sent=sent,
        skip_reason="no_alerters_configured" if attempted == 0 else "",
    )

    last_maintenance_ts = time.time()
    last_health_log_ts = 0.0
    health_log_interval_sec = max(60, _as_int(os.getenv("BOT_HEALTH_LOG_INTERVAL_SEC", "180"), 180))
    verbose_health_logging = _env_flag("BOT_HEALTH_LOG_VERBOSE", False)
    state_alert_cache: dict[str, str] = {}
    intervention_alert_cache: dict[str, str] = {}
    decision_log_cache: dict[str, str] = {}
    signal_alert_cache: dict[str, str] = {}
    early_signal_state: dict[str, dict[str, object]] = {}
    early_signal_stats: dict[str, int] = {
        "watch_sent": 0,
        "setup_sent": 0,
        "watch_to_setup_promoted": 0,
        "entry_confirmed": 0,
        "invalidated": 0,
        "suppressed_by_cooldown": 0,
    }
    try:
        while True:
            run_cycle(
                symbols=cfg.symbols,
                adapter=adapter,
                feed=feed,
                sync=sync,
                pipeline=pipeline,
                strategy=strategy,
                risk=risk,
                execution=execution,
                logger=logger,
                counters=counters,
                timeframe=cfg.timeframe,
                candles_limit=cfg.candles_limit,
                alerters=alerters,
                state_alert_cache=state_alert_cache,
                intervention_alert_cache=intervention_alert_cache,
                decision_log_cache=decision_log_cache,
                signal_alert_cache=signal_alert_cache,
                early_signal_state=early_signal_state,
                early_signal_stats=early_signal_stats,
                mode=cfg.mode,
                trade_learner=trade_learner,
                online_retrainer=online_retrainer,
                early_signal_learner=early_signal_learner,
                early_online_retrainer=early_online_retrainer,
                signal_profile=signal_profile,
            )

            now = time.time()
            if now - last_maintenance_ts >= cfg.maintenance_interval_sec:
                runtime_store.maintenance()
                last_maintenance_ts = now

            health = sync.health()
            strategy_audit_payload = _strategy_audit_log_payload(strategy)
            now_ts = time.time()
            if now_ts - last_health_log_ts >= health_log_interval_sec:
                if verbose_health_logging:
                    logger.info(
                        "metrics=%s risk=%s sync=%s metadata=%s early_signal_stats=%s strategy_audit_compact=%s strategy_audit_regime_filter=%s strategy_audit_regime_diagnostics=%s strategy_audit_layer1=%s strategy_audit_layer1_diagnostics=%s strategy_audit_layer2=%s strategy_audit_layer2_diagnostics=%s strategy_audit_layer4=%s strategy_audit_source_quality=%s strategy_audit=%s",
                        counters.snapshot(),
                        risk.health_snapshot(),
                        {
                            "ws_connected": health.ws_connected,
                            "ws_stale": health.ws_stale,
                            "fallback_polling": health.fallback_polling,
                            "snapshot_required": health.snapshot_required,
                        },
                        adapter.metadata_health(),
                        dict(early_signal_stats),
                        strategy_audit_payload.get("strategy_audit_compact", {}),
                        strategy_audit_payload.get("strategy_audit_regime_filter", {}),
                        strategy_audit_payload.get("strategy_audit_regime_diagnostics", {}),
                        strategy_audit_payload.get("strategy_audit_layer1", {}),
                        strategy_audit_payload.get("strategy_audit_layer1_diagnostics", {}),
                        strategy_audit_payload.get("strategy_audit_layer2", {}),
                        strategy_audit_payload.get("strategy_audit_layer2_diagnostics", {}),
                        strategy_audit_payload.get("strategy_audit_layer4", {}),
                        strategy_audit_payload.get("strategy_audit_source_quality", {}),
                        strategy_audit_payload.get("strategy_audit", {}),
                        extra={"event": "health"},
                    )
                else:
                    logger.info(
                        "metrics=%s risk=%s sync=%s metadata=%s early_signal_stats=%s strategy_audit_compact=%s",
                        counters.snapshot(),
                        risk.health_snapshot(),
                        {
                            "ws_connected": health.ws_connected,
                            "ws_stale": health.ws_stale,
                            "fallback_polling": health.fallback_polling,
                            "snapshot_required": health.snapshot_required,
                        },
                        adapter.metadata_health(),
                        dict(early_signal_stats),
                        strategy_audit_payload.get("strategy_audit_compact", {}),
                        extra={"event": "health"},
                    )
                last_health_log_ts = now_ts

            if not args.loop:
                break
            time.sleep(max(1, cfg.scan_interval_sec))
    finally:
        feed.close()
        adapter.close()
        runtime_store.close()

    return 0


if __name__ == "__main__":
    explicit_profile = str(os.getenv("BOT_SIGNAL_PROFILE", "")).strip().lower()
    if len(sys.argv) == 1 and explicit_profile not in {"main", "early"}:
        project_root = Path(__file__).resolve().parents[1]
        root_entry = project_root / "main.py"
        sys.path.insert(0, str(project_root))
        runpy.run_path(str(root_entry), run_name="__main__")
        raise SystemExit(0)
    raise SystemExit(main())



