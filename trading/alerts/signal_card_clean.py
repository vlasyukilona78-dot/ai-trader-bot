from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

import pandas as pd

from trading.alerts.signal_card import (
    EARLY_INVALIDATED_LABEL,
    _as_float,
    _base_asset,
    _derive_pump_context,
    _fmt_price,
    _infer_rsi_from_frame,
    _infer_volume_spike_from_frame,
    _last_enriched_float,
    _layer_details,
    _normalize_early_phase_label,
    _normalize_human_text as _legacy_normalize_human_text,
    _window_summary,
    build_symbol_copy_reply_markup,
)


_ICON_PUMP = "\U0001F7E2"
_ICON_TIME = "\u23F1"
_ICON_ENTRY = "\U0001F3AF"
_ICON_PRICE = "\U0001F4CD"
_ICON_RSI = "\U0001F4CA"
_ICON_VOLUME = "\U0001F4E6"
_ICON_RISK = "\u26A0\uFE0F"
_ICON_WINDOW = "\U0001F56F"
_BULLET = "\u2022"

_MOJIBAKE_REPLACEMENTS = {
    "Р РЋРЎвЂљР В°РЎР‚РЎв‚¬Р С‘Р в„– Р СћР В¤": "\u0421\u0442\u0430\u0440\u0448\u0438\u0439 \u0422\u0424",
    "HTF РЎС“РЎР‚Р С•Р Р†Р Р…Р С‘ + Р С”Р В°РЎР‚РЎвЂљР В° Р В»Р С‘Р С”Р Р†Р С‘Р Т‘Р В°РЎвЂ Р С‘Р в„–": "HTF \u0443\u0440\u043e\u0432\u043d\u0438 + \u043a\u0430\u0440\u0442\u0430 \u043b\u0438\u043a\u0432\u0438\u0434\u0430\u0446\u0438\u0439",
    "РїР°РјРї СѓР¶Рµ РµСЃС‚СЊ": "\u043f\u0430\u043c\u043f \u0443\u0436\u0435 \u0435\u0441\u0442\u044c",
    "РїРёРє РїР°РјРїР° СЃРѕРІСЃРµРј СЃРІРµР¶РёР№": "\u043f\u0438\u043a \u043f\u0430\u043c\u043f\u0430 \u0441\u043e\u0432\u0441\u0435\u043c \u0441\u0432\u0435\u0436\u0438\u0439",
    "С†РµРЅР° РµС‰С‘ Сѓ РІРµСЂС€РёРЅС‹ РїР°РјРїР°": "\u0446\u0435\u043d\u0430 \u0435\u0449\u0451 \u0443 \u0432\u0435\u0440\u0448\u0438\u043d\u044b \u043f\u0430\u043c\u043f\u0430",
    "РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·": "\u043f\u043e\u0448\u043b\u0430 \u043f\u0435\u0440\u0432\u0430\u044f \u0440\u0435\u0430\u043a\u0446\u0438\u044f \u0432\u043d\u0438\u0437",
    "РЅРёР¶Рµ РµСЃС‚СЊ Р»РёРєРІРёРґР°С†РёРѕРЅРЅС‹Р№ РјР°РіРЅРёС‚": "\u043d\u0438\u0436\u0435 \u0435\u0441\u0442\u044c \u043b\u0438\u043a\u0432\u0438\u0434\u0430\u0446\u0438\u043e\u043d\u043d\u044b\u0439 \u043c\u0430\u0433\u043d\u0438\u0442",
    "РІРµСЂС…РЅСЋСЋ Р»РёРєРІРёРґРЅРѕСЃС‚СЊ СѓР¶Рµ СЃРЅСЏР»Рё": "\u0432\u0435\u0440\u0445\u043d\u044e\u044e \u043b\u0438\u043a\u0432\u0438\u0434\u043d\u043e\u0441\u0442\u044c \u0443\u0436\u0435 \u0441\u043d\u044f\u043b\u0438",
    "РїРѕРґС‚РІРµСЂР¶РґРµРЅРёРµ СЃР»Р°Р±РѕСЃС‚Рё Рё РІС…РѕРґР°": "\u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u0438\u0435 \u0441\u043b\u0430\u0431\u043e\u0441\u0442\u0438 \u0438 \u0432\u0445\u043e\u0434\u0430",
    "РїРѕРґС‚РІРµСЂР¶РґРµРЅРёРµ СЃР»Р°Р±РѕСЃС‚Рё Рё РІС…Рѕда": "\u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u0438\u0435 \u0441\u043b\u0430\u0431\u043e\u0441\u0442\u0438 \u0438 \u0432\u0445\u043e\u0434\u0430",
}
_ESCAPED_UNICODE_RE = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f]")


def _repair_mojibake(text: str) -> str:
    try:
        return text.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _decode_escaped_unicode(text: str) -> str:
    if "\\u" not in text and "\\U" not in text:
        return text

    def _replace(match: re.Match[str]) -> str:
        short_hex = match.group(1)
        long_hex = match.group(2)
        hex_value = short_hex or long_hex
        try:
            return chr(int(hex_value, 16))
        except (TypeError, ValueError):
            return match.group(0)

    return _ESCAPED_UNICODE_RE.sub(_replace, text)


def _normalize_human_text(value: Any) -> str:
    text = _legacy_normalize_human_text(value)
    normalized = _CONTROL_CHARS_RE.sub("", str(text or "")).replace("\xa0", " ").strip()
    if not normalized:
        return ""
    for _ in range(2):
        decoded = _decode_escaped_unicode(normalized)
        repaired = _repair_mojibake(decoded)
        if decoded == normalized and repaired == normalized:
            break
        normalized = repaired.strip()
    for source, target in _MOJIBAKE_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _resolve_pump_caption_metrics(
    *,
    fallback_price: float,
    strategy_pump_size: float,
    enriched: pd.DataFrame | None,
) -> tuple[float, float, float, float, str]:
    pump_start, pump_end, visible_pump_size = _derive_pump_context(
        fallback_price=fallback_price,
        pump_size=strategy_pump_size,
        enriched=enriched,
    )
    clean_pump_size = max(_as_float(strategy_pump_size, 0.0), 0.0)
    if clean_pump_size <= 0.0:
        clean_pump_size = max(_as_float(visible_pump_size, 0.0), 0.0)
    display_pump_size = max(_as_float(visible_pump_size, 0.0), 0.0)
    if display_pump_size <= 0.0:
        display_pump_size = clean_pump_size
    clean_pump_label = "\u0427\u0438\u0441\u0442\u044b\u0439 \u043f\u0430\u043c\u043f \u0441\u0442\u0440\u0430\u0442\u0435\u0433\u0438\u0438"
    if display_pump_size > clean_pump_size + 0.0025:
        clean_pump_size = display_pump_size
        clean_pump_label = "\u041f\u0430\u043c\u043f \u043f\u043e \u0440\u0430\u043d\u043d\u0435\u043c\u0443 \u043e\u043a\u043d\u0443"
    return pump_start, pump_end, display_pump_size, clean_pump_size, clean_pump_label


def build_signal_caption(
    *,
    symbol: str,
    timeframe: str,
    mode: str,
    action_label: str,
    entry: float,
    tp: float,
    sl: float,
    confidence: float,
    reason: str,
    trace_meta: Mapping[str, Any] | None = None,
    enriched: pd.DataFrame | None = None,
) -> str:
    layer1 = _layer_details(trace_meta, "layer1_pump_detection")
    layer2 = _layer_details(trace_meta, "layer2_weakness_confirmation")
    layer4 = _layer_details(trace_meta, "layer4_fake_filter")

    strategy_pump_size = _as_float(layer1.get("clean_pump_pct"), 0.0)
    pump_min = _as_float(layer1.get("clean_pump_min_pct_used"), 0.05)
    rsi = _as_float(layer1.get("rsi"), _last_enriched_float(enriched, "rsi", 0.0))
    if rsi <= 0.0:
        rsi = _infer_rsi_from_frame(enriched, rsi)
    volume_spike = _as_float(layer1.get("volume_spike"), _last_enriched_float(enriched, "volume_spike", 0.0))
    if volume_spike <= 0.0:
        volume_spike = _infer_volume_spike_from_frame(enriched, volume_spike)
    weakness = _as_float(layer2.get("weakness_strength"), 0.0)
    sentiment_degraded = _as_float(layer4.get("degraded_mode"), 0.0) > 0.0
    pump_start, pump_end, pump_size, clean_pump_size, clean_pump_label = _resolve_pump_caption_metrics(
        fallback_price=entry,
        strategy_pump_size=strategy_pump_size,
        enriched=enriched,
    )
    window_label, window_open, window_close = _window_summary(enriched)
    asset = _base_asset(symbol)
    clean_action_label = _normalize_human_text(action_label)
    clean_reason = _normalize_human_text(reason)

    lines = [
        f"<b>{clean_action_label}</b>",
        f"<b>{asset}</b> | <code>{symbol}</code>",
        f"{_ICON_PUMP} \u041b\u043e\u043a\u0430\u043b\u044c\u043d\u044b\u0439 \u043f\u0430\u043c\u043f {pump_size * 100.0:.2f}% ({_fmt_price(pump_start)} -> {_fmt_price(pump_end)})",
        f"{_ICON_TIME} \u0422\u0424: {timeframe}\u043c | \u0411\u0438\u0440\u0436\u0430: Bybit | \u0420\u0435\u0436\u0438\u043c: {mode}",
        f"{_ICON_ENTRY} \u0412\u0445\u043e\u0434: {_fmt_price(entry)} | TP: {_fmt_price(tp)} | SL: {_fmt_price(sl)}",
        f"{_ICON_RSI} RSI ({timeframe}\u043c): {rsi:.2f}",
        f"{_ICON_VOLUME} \u041e\u0431\u044a\u0451\u043c: {volume_spike:.2f}x | \u0421\u043b\u0430\u0431\u043e\u0441\u0442\u044c: {weakness:.2f} | \u0423\u0432\u0435\u0440\u0435\u043d\u043d\u043e\u0441\u0442\u044c: {confidence * 100.0:.1f}%",
    ]
    if window_open is not None and window_close is not None:
        lines.append(
            f"{_ICON_WINDOW} {window_label}: open {_fmt_price(window_open)} / close {_fmt_price(window_close)}"
        )

    lines.extend(
        [
            "",
            f"{_ICON_RSI} <b>\u041a\u043e\u043d\u0442\u0435\u043a\u0441\u0442 \u043f\u043e \u043c\u043e\u043d\u0435\u0442\u0435 #{asset}</b>",
            f"{_BULLET} {clean_pump_label}: {clean_pump_size * 100.0:.2f}% \u043f\u0440\u0438 \u043c\u0438\u043d\u0438\u043c\u0443\u043c\u0435 {pump_min * 100.0:.2f}%",
        ]
    )
    if sentiment_degraded:
        lines.append(f"{_BULLET} \u041a\u043e\u043d\u0442\u0435\u043a\u0441\u0442 sentiment/derivatives: \u0434\u0435\u0433\u0440\u0430\u0434\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0439 \u0440\u0435\u0436\u0438\u043c")
    if clean_reason:
        lines.append(f"{_BULLET} \u041f\u0440\u0438\u0447\u0438\u043d\u0430: {clean_reason}")
    return "\n".join(_normalize_human_text(line) for line in lines)


def build_early_signal_caption(
    *,
    symbol: str,
    timeframe: str,
    mode: str,
    phase_label: str,
    price: float,
    trace_meta: Mapping[str, Any] | None = None,
    watch_score: float = 0.0,
    watch_max_score: float = 0.0,
    quality_score: float = 0.0,
    quality_max_score: float = 0.0,
    quality_grade: str = "",
    continuation_risk: float = 0.0,
    continuation_max_score: float = 0.0,
    triggers: list[str] | None = None,
    wait_for: str = "",
    enriched: pd.DataFrame | None = None,
) -> str:
    layer1 = _layer_details(trace_meta, "layer1_pump_detection")
    layer2 = _layer_details(trace_meta, "layer2_weakness_confirmation")

    strategy_pump_size = _as_float(layer1.get("clean_pump_pct"), 0.0)
    pump_min = _as_float(layer1.get("clean_pump_min_pct_used"), 0.05)
    rsi = _as_float(layer1.get("rsi"), _last_enriched_float(enriched, "rsi", 0.0))
    if rsi <= 0.0:
        rsi = _infer_rsi_from_frame(enriched, rsi)
    volume_spike = _as_float(layer1.get("volume_spike"), _last_enriched_float(enriched, "volume_spike", 0.0))
    if volume_spike <= 0.0:
        volume_spike = _infer_volume_spike_from_frame(enriched, volume_spike)
    weakness = _as_float(layer2.get("weakness_strength"), 0.0)
    pump_start, pump_end, pump_size, clean_pump_size, clean_pump_label = _resolve_pump_caption_metrics(
        fallback_price=price,
        strategy_pump_size=strategy_pump_size,
        enriched=enriched,
    )
    window_label, window_open, window_close = _window_summary(enriched)
    asset = _base_asset(symbol)
    clean_phase_label = _normalize_human_text(_normalize_early_phase_label(phase_label))
    clean_quality_grade = _normalize_human_text(quality_grade)
    clean_triggers = _dedupe_preserve_order(
        [item for item in (_normalize_human_text(t) for t in (triggers or [])) if item]
    )
    clean_wait_for = _normalize_human_text(wait_for)

    lines = [
        f"<b>{clean_phase_label}</b>",
        f"<b>{asset}</b> | <code>{symbol}</code>",
        f"{_ICON_PUMP} \u041b\u043e\u043a\u0430\u043b\u044c\u043d\u044b\u0439 \u043f\u0430\u043c\u043f {pump_size * 100.0:.2f}% ({_fmt_price(pump_start)} -> {_fmt_price(pump_end)})",
        f"{_ICON_TIME} \u0422\u0424: {timeframe}\u043c | \u0411\u0438\u0440\u0436\u0430: Bybit | \u0420\u0435\u0436\u0438\u043c: {mode}",
        f"{_ICON_PRICE} \u0426\u0435\u043d\u0430: {_fmt_price(price)}",
        f"{_ICON_RSI} RSI ({timeframe}\u043c): {rsi:.2f}",
        f"{_ICON_VOLUME} \u041e\u0431\u044a\u0451\u043c: {volume_spike:.2f}x | \u0421\u043b\u0430\u0431\u043e\u0441\u0442\u044c: {weakness:.2f}",
    ]
    if continuation_max_score > 0:
        lines.append(f"{_ICON_RISK} \u0420\u0438\u0441\u043a \u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0435\u043d\u0438\u044f: {continuation_risk:.1f}/{continuation_max_score:.1f}")
    if window_open is not None and window_close is not None:
        lines.append(
            f"{_ICON_WINDOW} {window_label}: open {_fmt_price(window_open)} / close {_fmt_price(window_close)}"
        )

    if clean_quality_grade and quality_max_score > 0:
        quality_line = (
            f"{_BULLET} \u041a\u043b\u0430\u0441\u0441 \u0441\u0435\u0442\u0430\u043f\u0430: {clean_quality_grade} "
            f"({quality_score:.1f}/{quality_max_score:.1f})"
        )
    elif quality_max_score > 0:
        quality_line = f"{_BULLET} Setup score: {quality_score:.1f}/{quality_max_score:.1f}"
    else:
        quality_line = f"{_BULLET} Setup score: {quality_score:.1f}"

    lines.extend(
        [
            "",
            f"{_ICON_RSI} <b>\u041a\u043e\u043d\u0442\u0435\u043a\u0441\u0442 \u043f\u043e \u043c\u043e\u043d\u0435\u0442\u0435 #{asset}</b>",
            (
                f"{_BULLET} Watch score: {watch_score:.1f}/{watch_max_score:.1f}"
                if watch_max_score > 0
                else f"{_BULLET} Watch score: {watch_score:.1f}"
            ),
            quality_line,
            f"{_BULLET} {clean_pump_label}: {clean_pump_size * 100.0:.2f}% \u043f\u0440\u0438 \u043c\u0438\u043d\u0438\u043c\u0443\u043c\u0435 {pump_min * 100.0:.2f}%",
        ]
    )
    if clean_triggers:
        lines.append(f"{_BULLET} \u0422\u0440\u0438\u0433\u0433\u0435\u0440\u044b: {', '.join(clean_triggers)}")
    if clean_wait_for:
        lines.append(f"{_BULLET} \u0416\u0434\u0451\u043c: {clean_wait_for}")
    return "\n".join(_normalize_human_text(line) for line in lines)


def build_early_invalidation_text(*, symbol: str, timeframe: str, mode: str, reason: str) -> str:
    asset = _base_asset(symbol)
    clean_reason = _normalize_human_text(reason)
    lines = [
        f"<b>{EARLY_INVALIDATED_LABEL}</b>",
        f"<b>{asset}</b> | <code>{symbol}</code>",
        f"{_ICON_TIME} \u0422\u0424: {timeframe}\u043c | \u0420\u0435\u0436\u0438\u043c: {mode}",
    ]
    if clean_reason:
        lines.append(f"\u041f\u0440\u0438\u0447\u0438\u043d\u0430: {clean_reason}")
    return "\n".join(_normalize_human_text(line) for line in lines)

