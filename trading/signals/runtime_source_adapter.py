from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

_UNAVAILABLE_SOURCES = {"", "unavailable", "missing", "none", "unknown"}
_FALLBACK_PREFIXES = ("fallback", "synthetic", "derived")


def _safe_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return None


def _source_tag(source: Any, *, fallback: str = "unavailable") -> str:
    src = str(source).strip().lower() if source is not None else ""
    return src or fallback


def _first_payload_value(payload: Mapping[str, Any], keys: list[str], caster) -> tuple[Any, str | None]:
    for key in keys:
        if key not in payload:
            continue
        value = caster(payload.get(key))
        if value is None:
            continue
        return value, key
    return None, None


def _first_payload_bool(payload: Mapping[str, Any], keys: list[str]) -> tuple[bool | None, str | None]:
    return _first_payload_value(payload, keys, _safe_bool_or_none)


def _first_payload_float(payload: Mapping[str, Any], keys: list[str]) -> tuple[float | None, str | None]:
    return _first_payload_value(payload, keys, _safe_float_or_none)


def _first_payload_text(payload: Mapping[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _first_payload_flag(payload: Mapping[str, Any], keys: list[str]) -> bool | None:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        parsed = _safe_bool_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _latest_numeric_from_frame(frame: pd.DataFrame | None, value_cols: list[str]) -> tuple[float | None, str | None]:
    if frame is None or frame.empty:
        return None, None
    row = frame.iloc[-1]
    for col in value_cols:
        if col not in frame.columns:
            continue
        value = _safe_float_or_none(row.get(col))
        if value is None:
            continue
        return value, col
    return None, None


def _latest_bool_from_frame(frame: pd.DataFrame | None, value_cols: list[str]) -> tuple[bool | None, str | None]:
    if frame is None or frame.empty:
        return None, None
    row = frame.iloc[-1]
    for col in value_cols:
        if col not in frame.columns:
            continue
        value = _safe_bool_or_none(row.get(col))
        if value is None:
            continue
        return value, col
    return None, None


def _latest_text_from_frame(frame: pd.DataFrame | None, source_cols: list[str]) -> str | None:
    if frame is None or frame.empty:
        return None
    row = frame.iloc[-1]
    for col in source_cols:
        if col not in frame.columns:
            continue
        value = row.get(col)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _source_quality(source: str, *, value_present: bool, degraded: bool | None) -> str:
    src = _source_tag(source)
    if src in _UNAVAILABLE_SOURCES:
        return "unavailable"
    if degraded is True:
        return "fallback"
    if src.startswith(_FALLBACK_PREFIXES):
        return "fallback"
    if not value_present:
        return "unavailable"
    return "live"


def _resolve_numeric(
    *,
    frame: pd.DataFrame,
    payload: Mapping[str, Any],
    value_keys: list[str],
    frame_value_cols: list[str],
    source_keys: list[str],
    frame_source_cols: list[str],
    degraded_keys: list[str],
) -> tuple[float | None, str, str, bool]:
    value, payload_value_key = _first_payload_float(payload, value_keys)
    source = _first_payload_text(payload, source_keys)
    degraded = _first_payload_flag(payload, degraded_keys)

    if value is not None and source is None and payload_value_key is not None:
        source = f"fallback:runtime:{payload_value_key}"
        if degraded is None:
            degraded = True

    if value is None:
        value, frame_value_col = _latest_numeric_from_frame(frame, frame_value_cols)
        if value is not None and source is None:
            source = _latest_text_from_frame(frame, frame_source_cols)
            if source is None:
                source = f"fallback:ohlcv:{frame_value_col}"
                if degraded is None:
                    degraded = True

    source_tag = _source_tag(source, fallback="unavailable")
    quality = _source_quality(source_tag, value_present=value is not None, degraded=degraded)
    return value, source_tag, quality, quality == "fallback"


def _resolve_bool(
    *,
    frame: pd.DataFrame,
    payload: Mapping[str, Any],
    value_keys: list[str],
    frame_value_cols: list[str],
    source_keys: list[str],
    frame_source_cols: list[str],
    degraded_keys: list[str],
) -> tuple[bool | None, str, str, bool]:
    value, payload_value_key = _first_payload_bool(payload, value_keys)
    source = _first_payload_text(payload, source_keys)
    degraded = _first_payload_flag(payload, degraded_keys)

    if value is not None and source is None and payload_value_key is not None:
        source = f"fallback:runtime:{payload_value_key}"
        if degraded is None:
            degraded = True

    if value is None:
        value, frame_value_col = _latest_bool_from_frame(frame, frame_value_cols)
        if value is not None and source is None:
            source = _latest_text_from_frame(frame, frame_source_cols)
            if source is None:
                source = f"fallback:ohlcv:{frame_value_col}"
                if degraded is None:
                    degraded = True

    source_tag = _source_tag(source, fallback="unavailable")
    quality = _source_quality(source_tag, value_present=value is not None, degraded=degraded)
    return value, source_tag, quality, quality == "fallback"


def build_runtime_signal_inputs(
    frame: pd.DataFrame,
    runtime_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect runtime signal-source fields with explicit live/fallback/unavailable quality."""
    payload = runtime_payload or {}

    sentiment_value, sentiment_source, sentiment_quality, sentiment_degraded = _resolve_numeric(
        frame=frame,
        payload=payload,
        value_keys=["sentiment_value", "sentiment_index", "sentiment"],
        frame_value_cols=["sentiment_value", "sentiment_index", "sentiment"],
        source_keys=["sentiment_source"],
        frame_source_cols=["sentiment_source"],
        degraded_keys=["sentiment_degraded"],
    )
    funding_rate, funding_source, funding_quality, funding_degraded = _resolve_numeric(
        frame=frame,
        payload=payload,
        value_keys=["funding_rate", "funding"],
        frame_value_cols=["funding_rate", "funding"],
        source_keys=["funding_source"],
        frame_source_cols=["funding_source"],
        degraded_keys=["funding_degraded"],
    )
    long_short_ratio, lsr_source, lsr_quality, lsr_degraded = _resolve_numeric(
        frame=frame,
        payload=payload,
        value_keys=["long_short_ratio", "long_short_account_ratio", "ls_ratio"],
        frame_value_cols=["long_short_ratio", "long_short_account_ratio", "ls_ratio"],
        source_keys=["long_short_ratio_source"],
        frame_source_cols=["long_short_ratio_source"],
        degraded_keys=["long_short_ratio_degraded"],
    )
    open_interest_ratio, oi_ratio_source, oi_ratio_quality, oi_ratio_degraded = _resolve_numeric(
        frame=frame,
        payload=payload,
        value_keys=["open_interest_ratio", "oi_ratio"],
        frame_value_cols=["open_interest_ratio", "oi_ratio"],
        source_keys=["oi_source", "open_interest_source"],
        frame_source_cols=["oi_source", "open_interest_source"],
        degraded_keys=["oi_degraded"],
    )
    oi_signal, oi_signal_source, oi_signal_quality, oi_signal_degraded = _resolve_numeric(
        frame=frame,
        payload=payload,
        value_keys=["oi_signal", "open_interest_signal"],
        frame_value_cols=["oi_signal", "open_interest_signal"],
        source_keys=["oi_source", "open_interest_source"],
        frame_source_cols=["oi_source", "open_interest_source"],
        degraded_keys=["oi_degraded"],
    )
    open_interest_raw, open_interest_raw_source, open_interest_raw_quality, open_interest_raw_degraded = _resolve_numeric(
        frame=frame,
        payload=payload,
        value_keys=["open_interest"],
        frame_value_cols=["open_interest"],
        source_keys=["open_interest_source", "oi_source"],
        frame_source_cols=["open_interest_source", "oi_source"],
        degraded_keys=["oi_degraded"],
    )
    news_veto, news_source, news_quality, news_degraded = _resolve_bool(
        frame=frame,
        payload=payload,
        value_keys=["news_veto", "news_blocked", "catalyst_veto"],
        frame_value_cols=["news_veto", "news_blocked", "catalyst_veto"],
        source_keys=["news_source"],
        frame_source_cols=["news_source"],
        degraded_keys=["news_degraded"],
    )

    open_interest = open_interest_raw
    oi_source = open_interest_raw_source
    oi_quality = open_interest_raw_quality
    oi_degraded_final = open_interest_raw_degraded
    if open_interest is None:
        open_interest = open_interest_ratio
        oi_source = oi_ratio_source
        oi_quality = oi_ratio_quality
        oi_degraded_final = oi_ratio_degraded
    if open_interest is None:
        open_interest = oi_signal
        oi_source = oi_signal_source
        oi_quality = oi_signal_quality
        oi_degraded_final = oi_signal_degraded

    return {
        "sentiment_value": sentiment_value,
        "sentiment_source": sentiment_source,
        "sentiment_degraded": sentiment_degraded,
        "sentiment_quality": sentiment_quality,
        "funding_rate": funding_rate,
        "funding_source": funding_source,
        "funding_degraded": funding_degraded,
        "funding_quality": funding_quality,
        "long_short_ratio": long_short_ratio,
        "long_short_ratio_source": lsr_source,
        "long_short_ratio_degraded": lsr_degraded,
        "long_short_ratio_quality": lsr_quality,
        "open_interest_ratio": open_interest_ratio,
        "oi_signal": oi_signal,
        "oi_source": oi_source,
        "oi_degraded": oi_degraded_final,
        "oi_quality": oi_quality,
        "news_veto": news_veto,
        "news_source": news_source,
        "news_degraded": news_degraded,
        "news_quality": news_quality,
        # Backward-compatible aliases.
        "sentiment_index": sentiment_value,
        "open_interest": open_interest,
        "open_interest_source": oi_source,
    }

