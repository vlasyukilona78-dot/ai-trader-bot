# -*- coding: utf-8 -*-
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


EARLY_WATCH_LABEL = "РАННИЙ ШОРТ: НАБЛЮДЕНИЕ"
EARLY_SETUP_LABEL = "РАННИЙ ШОРТ: СЕТАП"
EARLY_INVALIDATED_LABEL = "РАННИЙ ШОРТ ОТМЕНЁН"
WINDOW_LABEL = "Окно"

_MOJIBAKE_MARKERS = (
    "Р ",
    "РЃ",
    "СЏ",
    "рџ",
    "вЂў",
    "вЏ±",
)


# Rebind user-facing labels via unicode escapes so caption rendering stays stable
# even when the source file is viewed or saved under a lossy Windows code page.
EARLY_WATCH_LABEL = "\u0420\u0410\u041d\u041d\u0418\u0419 \u0428\u041e\u0420\u0422: \u041d\u0410\u0411\u041b\u042e\u0414\u0415\u041d\u0418\u0415"
EARLY_SETUP_LABEL = "\u0420\u0410\u041d\u041d\u0418\u0419 \u0428\u041e\u0420\u0422: \u0421\u0415\u0422\u0410\u041f"
EARLY_INVALIDATED_LABEL = "\u0420\u0410\u041d\u041d\u0418\u0419 \u0428\u041e\u0420\u0422 \u041e\u0422\u041c\u0415\u041d\u0401\u041d"
WINDOW_LABEL = "\u041e\u043a\u043d\u043e"

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_price(value: Any) -> str:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if price >= 1000:
        return f"{price:,.2f}".replace(",", " ")
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.6f}"


def _looks_mojibake(text: str) -> bool:
    return sum(text.count(marker) for marker in _MOJIBAKE_MARKERS) >= 2


def _text_readability_score(text: str) -> int:
    cyrillic = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
    latin = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    mojibake = sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)
    return cyrillic + latin - mojibake * 3


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(token in haystack for token in needles)


def _canonicalize_signal_phrase(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""

    upper = value.upper().replace(" ", "")
    has_short = _contains_any(
        upper,
        (
            "ШОРТ",
            "SHORT",
            "РЁР",
            "Р РЃР С›Р В Р Сћ",
            "Р В Р РѓР В ",
        ),
    )
    has_early = _contains_any(
        upper,
        (
            "РАННИ",
            "EARLY",
            "Р РђРќРќ",
            "Р В Р С’Р СњР Сњ",
            "РАНН",
        ),
    )
    setup_like = _contains_any(upper, ("СЕТАП", "SETUP", "РЎР•РўРђРџ"))
    watch_like = _contains_any(upper, ("НАБЛ", "WATCH", "РќРђР‘Р›"))
    invalid_like = _contains_any(upper, ("ОТМЕН", "INVALID", "РћРўРњР•Рќ"))

    if setup_like and (has_short or has_early):
        return EARLY_SETUP_LABEL
    if watch_like and (has_short or has_early):
        return EARLY_WATCH_LABEL
    if invalid_like and (has_short or has_early):
        return EARLY_INVALIDATED_LABEL
    return value


def _repair_mojibake(text: str) -> str:
    best_text = text
    best_score = _text_readability_score(text)
    candidates = {text}
    frontier = [text]

    for _ in range(3):
        next_frontier: list[str] = []
        for value in frontier:
            for codec in ("cp1251", "latin1"):
                try:
                    fixed = value.encode(codec, errors="ignore").decode("utf-8", errors="ignore").strip()
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
                if not fixed or fixed in candidates:
                    continue
                candidates.add(fixed)
                next_frontier.append(fixed)
        if not next_frontier:
            break
        frontier = next_frontier

    for candidate in candidates:
        fixed = _canonicalize_signal_phrase(candidate)
        fixed_score = _text_readability_score(fixed)
        if fixed_score > best_score:
            best_text = fixed
            best_score = fixed_score

    return best_text


def _normalize_human_text(value: Any) -> str:
    text = str(value or "").strip().replace("\xa0", " ")
    if not text:
        return ""

    canonical = _canonicalize_signal_phrase(text)
    if canonical in (EARLY_WATCH_LABEL, EARLY_SETUP_LABEL, EARLY_INVALIDATED_LABEL):
        return canonical

    repaired = _repair_mojibake(text) if _looks_mojibake(text) else canonical
    cleaned = _canonicalize_signal_phrase(repaired)
    upper = cleaned.upper()

    if "LOWER-HIGH / LOWER-CLOSE" in upper:
        return "после пика уже появился lower-high / lower-close"
    if "OBV" in upper and _contains_any(upper, ("Р РћРЎРў", "Р РЋР С›Р РЋР Сћ", "Р В Р С›Р РЋР Сћ")):
        return "OBV не подтверждает рост"
    if "CVD" in upper and _contains_any(upper, ("Р РћРЎРў", "Р РЋР С›Р РЋР Сћ", "Р В Р С›Р РЋР Сћ")):
        return "CVD не подтверждает рост"
    if "MACD" in upper and _contains_any(upper, ("РћРЎР›РђР‘", "Р С›Р РЋР вЂєР С’Р вЂ")):
        return "MACD ослабевает"
    if "RSI" in upper:
        if _contains_any(upper, ("РќР•Р™РўР РђР›", "Р СњР вЂўР в„ўР СћР В Р С’Р вЂє")):
            return "RSI выше нейтрали"
        if _contains_any(upper, ("Р РђР—Р’РћР РђР§", "Р В Р С’Р вЂ”Р вЂ™Р С›Р В Р С’Р В§")):
            return "RSI разворачивается вниз"
    if _contains_any(upper, ("Р›РРљР’РР”", "Р вЂєР ВР С™Р вЂ™Р ВР вЂќ")):
        if _contains_any(upper, ("РЎРќРЇ", "Р РЋР СњР Р‡", "Р РЋР РЉР Р‡")):
            return "верхнюю ликвидность уже сняли"
        if _contains_any(upper, ("РќРР–Р•", "РњРђР“РќРРў", "Р СњР ВР вЂ“Р вЂў", "Р СљР С’Р вЂњР СњР ВР Сћ")):
            return "ниже есть ликвидационный магнит"
    if _contains_any(upper, ("РџРРљ", "Р СџР ВР С™")):
        if _contains_any(upper, ("РЎР’Р•Р–", "Р РЋР вЂ™Р вЂўР вЂ“")):
            return "пик пампа совсем свежий"
        if _contains_any(upper, ("РџР•Р Р’РђРЇ", "Р СџР вЂўР В Р вЂ™Р С’Р Р‡")):
            return "пошла первая реакция вниз"
    if _contains_any(upper, ("Р¦Р•РќРђ", "Р В¦Р вЂўР СњР С’")):
        if _contains_any(upper, ("Р’Р•Р РЁРРќ", "Р вЂ™Р вЂўР В Р РЃР ВР Сњ")):
            return "цена ещё у вершины пампа"
        if _contains_any(upper, ("Р›РћРљРђР›Р¬Рќ", "Р вЂєР С›Р С™Р С’Р вЂєР В¬Р Сњ")):
            return "цена у локального хая"
        if _contains_any(upper, ("РџР•Р Р•РЎРўРђР›", "Р СџР вЂўР В Р вЂўР РЋР СћР С’Р вЂє")):
            return "цена перестала ускоряться"
        if _contains_any(upper, ("Р’Р•Р РҐРќ", "Р вЂ™Р вЂўР В Р ТђР Сњ")):
            return "цена у верхней зоны"
    if _contains_any(upper, ("РћР‘РЄ", "РћР‘Р¬", "Р С›Р вЂР Р„", "Р С›Р вЂР В¬")):
        if _contains_any(upper, ("Р—РђРўРЈРҐ", "Р вЂ”Р С’Р СћР Р€Р Тђ")):
            return "объём затухает"
        if _contains_any(upper, ("РџРћР’Р«РЁ", "Р СџР С›Р вЂ™Р В«Р РЃ")):
            return "объём ещё повышен"
    if _contains_any(upper, ("РўР•Рќ", "Р СћР вЂўР Сњ")):
        return "есть верхняя тень"
    return cleaned


def _normalize_early_phase_label(value: Any) -> str:
    text = str(value or "").strip().replace("\xa0", " ")
    if not text:
        return EARLY_WATCH_LABEL

    compact = text.upper().replace(" ", "")
    if _contains_any(compact, ("СЕТАП", "SETUP", "РЎР•РЎ", "РЎР•РўРђРџ")):
        return EARLY_SETUP_LABEL
    if _contains_any(compact, ("НАБЛЮД", "НАБЛ", "WATCH", "РќРђР‘Р›", "РќРђР‘Р›Р®")):
        return EARLY_WATCH_LABEL
    if _contains_any(compact, ("ОТМЕН", "INVALID", "РћРўРњР•Рќ")):
        return EARLY_INVALIDATED_LABEL

    cleaned = _normalize_human_text(text)
    if cleaned in (EARLY_WATCH_LABEL, EARLY_SETUP_LABEL, EARLY_INVALIDATED_LABEL):
        return cleaned
    return EARLY_WATCH_LABEL


_ORIGINAL_NORMALIZE_HUMAN_TEXT = _normalize_human_text
_NORMALIZED_TEXT_REPLACEMENTS = {
    "РЎС‚Р°СЂС€РёР№ РўР¤": "Старший ТФ",
    "HTF СѓСЂРѕРІРЅРё + РєР°СЂС‚Р° Р»РёРєРІРёРґР°С†РёР№": "HTF уровни + карта ликвидаций",
    "РЎРўРђР Рў Р‘РћРўРђ": "СТАРТ БОТА",
    "Р РµР¶РёРј": "Режим",
    "РЎРёРјРІРѕР»РѕРІ": "Символов",
    "РґР°": "да",
    "РЅРµС‚": "нет",
    "РљР РРўРР§РќРћ": "КРИТИЧНО",
    "РЁРћР Рў РЎРР“РќРђР›": "ШОРТ СИГНАЛ",
    "Р›РћРќР“ РЎРР“РќРђР›": "ЛОНГ СИГНАЛ",
    "Р РђРќРќРР™ РЁРћР Рў: HTF РљРћРќРўР•РљРЎРў": "РАННИЙ ШОРТ: HTF КОНТЕКСТ",
    "РЁРћР Рў РЎРР“РќРђР›: HTF РљРћРќРўР•РљРЎРў": "ШОРТ СИГНАЛ: HTF КОНТЕКСТ",
    "Р›РћРќР“ РЎРР“РќРђР›: HTF РљРћРќРўР•РљРЎРў": "ЛОНГ СИГНАЛ: HTF КОНТЕКСТ",
    "РїРѕСЃР»Рµ РїРёРєР° СѓР¶Рµ РїРѕСЏРІРёР»СЃСЏ lower-high / lower-close": "после пика уже появился lower-high / lower-close",
    "OBV РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚": "OBV не подтверждает рост",
    "CVD РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚": "CVD не подтверждает рост",
    "MACD РѕСЃР»Р°Р±РµРІР°РµС‚": "MACD ослабевает",
    "RSI РІС‹С€Рµ РЅРµР№С‚СЂР°Р»Рё": "RSI выше нейтрали",
    "RSI СЂР°Р·РІРѕСЂР°С‡РёРІР°РµС‚СЃСЏ РІРЅРёР·": "RSI разворачивается вниз",
    "РІРµСЂС…РЅСЋСЋ Р»РёРєРІРёРґРЅРѕСЃС‚СЊ СѓР¶Рµ СЃРЅСЏР»Рё": "верхнюю ликвидность уже сняли",
    "РЅРёР¶Рµ РµСЃС‚СЊ Р»РёРєРІРёРґР°С†РёРѕРЅРЅС‹Р№ РјР°РіРЅРёС‚": "ниже есть ликвидационный магнит",
    "РїРёРє РїР°РјРїР° СЃРѕРІСЃРµРј СЃРІРµР¶РёР№": "пик пампа совсем свежий",
    "РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·": "пошла первая реакция вниз",
    "С†РµРЅР° РµС‰С‘ Сѓ РІРµСЂС€РёРЅС‹ РїР°РјРїР°": "цена ещё у вершины пампа",
    "С†РµРЅР° Сѓ Р»РѕРєР°Р»СЊРЅРѕРіРѕ С…Р°СЏ": "цена у локального хая",
    "С†РµРЅР° РїРµСЂРµСЃС‚Р°Р»Р° СѓСЃРєРѕСЂСЏС‚СЊСЃСЏ": "цена перестала ускоряться",
    "С†РµРЅР° Сѓ РІРµСЂС…РЅРµР№ Р·РѕРЅС‹": "цена у верхней зоны",
    "РѕР±СЉС‘Рј Р·Р°С‚СѓС…Р°РµС‚": "объём затухает",
    "РѕР±СЉС‘Рј РµС‰С‘ РїРѕРІС‹С€РµРЅ": "объём ещё повышен",
    "РµСЃС‚СЊ РІРµСЂС…РЅСЏСЏ С‚РµРЅСЊ": "есть верхняя тень",
}


def _normalize_human_text(value: Any) -> str:
    cleaned = _ORIGINAL_NORMALIZE_HUMAN_TEXT(value)
    for bad, good in _NORMALIZED_TEXT_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, good)
    return cleaned


_SECOND_PASS_NORMALIZE_HUMAN_TEXT = _normalize_human_text
_FIXED_RUSSIAN_REPLACEMENTS = {
    "РЎС‚Р°СЂС€РёР№ РўР¤": "Старший ТФ",
    "HTF СѓСЂРѕРІРЅРё + РєР°СЂС‚Р° Р»РёРєРІРёРґР°С†РёР№": "HTF уровни + карта ликвидаций",
    "РЎРўРђР Рў Р‘РћРўРђ": "СТАРТ БОТА",
    "Р РµР¶РёРј": "Режим",
    "РЎРёРјРІРѕР»РѕРІ": "Символов",
    "РґР°": "да",
    "РЅРµС‚": "нет",
    "РљР РРўРР§РќРћ": "КРИТИЧНО",
    "РЁРћР Рў РЎРР“РќРђР›": "ШОРТ СИГНАЛ",
    "Р›РћРќР“ РЎРР“РќРђР›": "ЛОНГ СИГНАЛ",
    "Р РђРќРќРР™ РЁРћР Рў: HTF РљРћРќРўР•РљРЎРў": "РАННИЙ ШОРТ: HTF КОНТЕКСТ",
    "РЁРћР Рў РЎРР“РќРђР›: HTF РљРћРќРўР•РљРЎРў": "ШОРТ СИГНАЛ: HTF КОНТЕКСТ",
    "Р›РћРќР“ РЎРР“РќРђР›: HTF РљРћРќРўР•РљРЎРў": "ЛОНГ СИГНАЛ: HTF КОНТЕКСТ",
    "РїРѕСЃР»Рµ РїРёРєР° СѓР¶Рµ РїРѕСЏРІРёР»СЃСЏ lower-high / lower-close": "после пика уже появился lower-high / lower-close",
    "OBV РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚": "OBV не подтверждает рост",
    "CVD РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚": "CVD не подтверждает рост",
    "MACD РѕСЃР»Р°Р±РµРІР°РµС‚": "MACD ослабевает",
    "RSI РІС‹С€Рµ РЅРµР№С‚СЂР°Р»Рё": "RSI выше нейтрали",
    "RSI СЂР°Р·РІРѕСЂР°С‡РёРІР°РµС‚СЃСЏ РІРЅРёР·": "RSI разворачивается вниз",
    "РІРµСЂС…РЅСЋСЋ Р»РёРєРІРёРґРЅРѕСЃС‚СЊ СѓР¶Рµ СЃРЅСЏР»Рё": "верхнюю ликвидность уже сняли",
    "РЅРёР¶Рµ РµСЃС‚СЊ Р»РёРєРІРёРґР°С†РёРѕРЅРЅС‹Р№ РјР°РіРЅРёС‚": "ниже есть ликвидационный магнит",
    "РїРёРє РїР°РјРїР° СЃРѕРІСЃРµРј СЃРІРµР¶РёР№": "пик пампа совсем свежий",
    "РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·": "пошла первая реакция вниз",
    "С†РµРЅР° РµС‰С‘ Сѓ РІРµСЂС€РёРЅС‹ РїР°РјРїР°": "цена ещё у вершины пампа",
    "С†РµРЅР° Сѓ Р»РѕРєР°Р»СЊРЅРѕРіРѕ С…Р°СЏ": "цена у локального хая",
    "С†РµРЅР° РїРµСЂРµСЃС‚Р°Р»Р° СѓСЃРєРѕСЂСЏС‚СЊСЃСЏ": "цена перестала ускоряться",
    "С†РµРЅР° Сѓ РІРµСЂС…РЅРµР№ Р·РѕРЅС‹": "цена у верхней зоны",
    "РѕР±СЉС‘Рј Р·Р°С‚СѓС…Р°РµС‚": "объём затухает",
    "РѕР±СЉС‘Рј РµС‰С‘ РїРѕРІС‹С€РµРЅ": "объём ещё повышен",
    "РµСЃС‚СЊ РІРµСЂС…РЅСЏСЏ С‚РµРЅСЊ": "есть верхняя тень",
}


def _normalize_human_text(value: Any) -> str:
    cleaned = _SECOND_PASS_NORMALIZE_HUMAN_TEXT(value)
    for bad, good in _FIXED_RUSSIAN_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, good)
    return cleaned


_THIRD_PASS_NORMALIZE_HUMAN_TEXT = _normalize_human_text
_FINAL_RUSSIAN_REPLACEMENTS = {
    "РЎС‚Р°СЂС€РёР№ РўР¤": "Старший ТФ",
    "HTF СѓСЂРѕРІРЅРё + РєР°СЂС‚Р° Р»РёРєРІРёРґР°С†РёР№": "HTF уровни + карта ликвидаций",
    "РЎРўРђР Рў Р‘РћРўРђ": "СТАРТ БОТА",
    "Р РµР¶РёРј": "Режим",
    "РЎРёРјРІРѕР»РѕРІ": "Символов",
    "РґР°": "да",
    "РЅРµС‚": "нет",
    "РїРѕСЃР»Рµ РїРёРєР° СѓР¶Рµ РїРѕСЏРІРёР»СЃСЏ lower-high / lower-close": "после пика уже появился lower-high / lower-close",
    "OBV РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚": "OBV не подтверждает рост",
    "CVD РЅРµ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ СЂРѕСЃС‚": "CVD не подтверждает рост",
    "MACD РѕСЃР»Р°Р±РµРІР°РµС‚": "MACD ослабевает",
    "RSI РІС‹С€Рµ РЅРµР№С‚СЂР°Р»Рё": "RSI выше нейтрали",
    "RSI СЂР°Р·РІРѕСЂР°С‡РёРІР°РµС‚СЃСЏ РІРЅРёР·": "RSI разворачивается вниз",
    "РІРµСЂС…РЅСЋСЋ Р»РёРєРІРёРґРЅРѕСЃС‚СЊ СѓР¶Рµ СЃРЅСЏР»Рё": "верхнюю ликвидность уже сняли",
    "РЅРёР¶Рµ РµСЃС‚СЊ Р»РёРєРІРёРґР°С†РёРѕРЅРЅС‹Р№ РјР°РіРЅРёС‚": "ниже есть ликвидационный магнит",
    "РїРёРє РїР°РјРїР° СЃРѕРІСЃРµРј СЃРІРµР¶РёР№": "пик пампа совсем свежий",
    "РїРѕС€Р»Р° РїРµСЂРІР°СЏ СЂРµР°РєС†РёСЏ РІРЅРёР·": "пошла первая реакция вниз",
    "С†РµРЅР° РµС‰С‘ Сѓ РІРµСЂС€РёРЅС‹ РїР°РјРїР°": "цена ещё у вершины пампа",
    "С†РµРЅР° Сѓ Р»РѕРєР°Р»СЊРЅРѕРіРѕ С…Р°СЏ": "цена у локального хая",
    "С†РµРЅР° РїРµСЂРµСЃС‚Р°Р»Р° СѓСЃРєРѕСЂСЏС‚СЊСЃСЏ": "цена перестала ускоряться",
    "С†РµРЅР° Сѓ РІРµСЂС…РЅРµР№ Р·РѕРЅС‹": "цена у верхней зоны",
    "РѕР±СЉС‘Рј Р·Р°С‚СѓС…Р°РµС‚": "объём затухает",
    "РѕР±СЉС‘Рј РµС‰С‘ РїРѕРІС‹С€РµРЅ": "объём ещё повышен",
    "РµСЃС‚СЊ РІРµСЂС…РЅСЏСЏ С‚РµРЅСЊ": "есть верхняя тень",
}


def _normalize_human_text(value: Any) -> str:
    cleaned = _THIRD_PASS_NORMALIZE_HUMAN_TEXT(value)
    for bad, good in _FINAL_RUSSIAN_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, good)
    return cleaned


def _layer_details(trace_meta: Mapping[str, Any] | None, layer_name: str) -> dict[str, Any]:
    trace = trace_meta.get("layer_trace", {}) if isinstance(trace_meta, Mapping) else {}
    layers = trace.get("layers", {}) if isinstance(trace, Mapping) else {}
    layer = layers.get(layer_name, {}) if isinstance(layers, Mapping) else {}
    details = layer.get("details", {}) if isinstance(layer, Mapping) else {}
    return details if isinstance(details, Mapping) else {}


def _base_asset(symbol: str) -> str:
    clean = str(symbol or "").replace("/", "").upper().strip()
    for suffix in ("USDT", "USDC", "USD"):
        if clean.endswith(suffix) and len(clean) > len(suffix):
            return clean[: -len(suffix)]
    return clean


def build_symbol_copy_reply_markup(symbol: str) -> dict[str, object]:
    clean = str(symbol or "").replace("/", "").upper().strip()
    return {"inline_keyboard": [[{"text": clean, "copy_text": {"text": clean}}]]}


def _pump_prices(entry: float, pump_size: float) -> tuple[float, float]:
    end_price = max(float(entry or 0.0), 1e-8)
    if pump_size <= 0:
        return end_price, end_price
    start_price = end_price / (1.0 + pump_size)
    return start_price, end_price


def _last_enriched_float(enriched: pd.DataFrame | None, column: str, default: float = 0.0) -> float:
    if enriched is None or enriched.empty or column not in enriched.columns:
        return default
    try:
        return _as_float(enriched.iloc[-1].get(column), default)
    except Exception:
        return default


def _infer_rsi_from_frame(enriched: pd.DataFrame | None, default: float = 0.0) -> float:
    if enriched is None or enriched.empty or "close" not in enriched.columns:
        return default
    try:
        close = pd.to_numeric(enriched["close"], errors="coerce").dropna()
        if len(close) < 15:
            return default
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0.0, pd.NA)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        value = _as_float(rsi.iloc[-1], default)
        if 0.0 < value <= 100.0:
            return value
    except Exception:
        pass
    return default


def _infer_volume_spike_from_frame(enriched: pd.DataFrame | None, default: float = 0.0) -> float:
    if enriched is None or enriched.empty or "volume" not in enriched.columns:
        return default
    try:
        volume = pd.to_numeric(enriched["volume"], errors="coerce").dropna()
        if len(volume) < 6:
            return default
        baseline = float(volume.tail(min(24, len(volume) - 1)).iloc[:-1].mean())
        if baseline <= 0:
            return default
        spike = float(volume.iloc[-1]) / baseline
        if spike > 0:
            return spike
    except Exception:
        pass
    return default


def _derive_pump_context(
    *,
    fallback_price: float,
    pump_size: float,
    enriched: pd.DataFrame | None,
) -> tuple[float, float, float]:
    if enriched is not None and not enriched.empty:
        recent = enriched.tail(min(96, len(enriched)))
        try:
            recent_close = pd.to_numeric(recent["close"], errors="coerce").dropna()
            if not recent_close.empty:
                end_price = float(recent_close.iloc[-1])
                # Caption pump size should reflect the visible close-to-close move
                # in the highlighted setup window, not a transient wick low.
                start_price = float(recent_close.min())
                if end_price > start_price > 0:
                    derived_size = max((end_price / start_price) - 1.0, 0.0)
                    if derived_size > 0:
                        return start_price, end_price, max(pump_size, derived_size)
        except Exception:
            pass

    end_price = max(float(fallback_price or 0.0), 1e-8)
    start_price, end_price = _pump_prices(end_price, pump_size)
    return start_price, end_price, max(pump_size, 0.0)


def _window_summary(enriched: pd.DataFrame | None) -> tuple[str, float | None, float | None]:
    if enriched is None or enriched.empty:
        return WINDOW_LABEL, None, None
    try:
        frame = enriched.sort_index()
        return (
            WINDOW_LABEL,
            _as_float(frame.iloc[0].get("open"), 0.0),
            _as_float(frame.iloc[-1].get("close"), 0.0),
        )
    except Exception:
        return WINDOW_LABEL, None, None


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

    pump_size = _as_float(layer1.get("clean_pump_pct"), 0.0)
    pump_min = _as_float(layer1.get("clean_pump_min_pct_used"), 0.05)
    rsi = _as_float(layer1.get("rsi"), _last_enriched_float(enriched, "rsi", 0.0))
    if rsi <= 0.0:
        rsi = _infer_rsi_from_frame(enriched, rsi)
    volume_spike = _as_float(layer1.get("volume_spike"), _last_enriched_float(enriched, "volume_spike", 0.0))
    if volume_spike <= 0.0:
        volume_spike = _infer_volume_spike_from_frame(enriched, volume_spike)
    weakness = _as_float(layer2.get("weakness_strength"), 0.0)
    sentiment_degraded = _as_float(layer4.get("degraded_mode"), 0.0) > 0.0
    pump_start, pump_end, pump_size = _derive_pump_context(
        fallback_price=entry,
        pump_size=pump_size,
        enriched=enriched,
    )
    window_label, window_open, window_close = _window_summary(enriched)
    asset = _base_asset(symbol)
    clean_action_label = _normalize_human_text(action_label)
    clean_reason = _normalize_human_text(reason)

    lines = [
        f"<b>{clean_action_label}</b>",
        f"<b>{asset}</b> | <code>{symbol}</code>",
        f"🟢 Pump {pump_size * 100.0:.2f}% ({_fmt_price(pump_start)} -> {_fmt_price(pump_end)})",
        f"⏱ ТФ: {timeframe}м | Биржа: Bybit | Режим: {mode}",
        f"🎯 Вход: {_fmt_price(entry)} | TP: {_fmt_price(tp)} | SL: {_fmt_price(sl)}",
        f"📊 RSI ({timeframe}м): {rsi:.2f}",
        f"📦 Объём: {volume_spike:.2f}x | Слабость: {weakness:.2f} | Уверенность: {confidence * 100.0:.1f}%",
    ]
    if window_open is not None and window_close is not None:
        lines.append(f"🕯 {window_label}: open {_fmt_price(window_open)} / close {_fmt_price(window_close)}")

    lines.extend(
        [
            "",
            f"📊 <b>Контекст по монете #{asset}</b>",
            f"• Чистый памп: {pump_size * 100.0:.2f}% при минимуме {pump_min * 100.0:.2f}%",
        ]
    )
    if sentiment_degraded:
        lines.append("• Контекст sentiment/derivatives: деградированный режим")
    if clean_reason:
        lines.append(f"• Причина: {clean_reason}")
    return "\n".join(lines)


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

    pump_size = _as_float(layer1.get("clean_pump_pct"), 0.0)
    pump_min = _as_float(layer1.get("clean_pump_min_pct_used"), 0.05)
    rsi = _as_float(layer1.get("rsi"), _last_enriched_float(enriched, "rsi", 0.0))
    if rsi <= 0.0:
        rsi = _infer_rsi_from_frame(enriched, rsi)
    volume_spike = _as_float(layer1.get("volume_spike"), _last_enriched_float(enriched, "volume_spike", 0.0))
    if volume_spike <= 0.0:
        volume_spike = _infer_volume_spike_from_frame(enriched, volume_spike)
    weakness = _as_float(layer2.get("weakness_strength"), 0.0)
    pump_start, pump_end, pump_size = _derive_pump_context(
        fallback_price=price,
        pump_size=pump_size,
        enriched=enriched,
    )
    window_label, window_open, window_close = _window_summary(enriched)
    asset = _base_asset(symbol)
    clean_phase_label = _normalize_early_phase_label(phase_label)
    clean_quality_grade = _normalize_human_text(quality_grade)
    clean_triggers = [item for item in (_normalize_human_text(t) for t in (triggers or [])) if item]
    clean_wait_for = _normalize_human_text(wait_for)

    lines = [
        f"<b>{clean_phase_label}</b>",
        f"<b>{asset}</b> | <code>{symbol}</code>",
        f"🟢 Pump {pump_size * 100.0:.2f}% ({_fmt_price(pump_start)} -> {_fmt_price(pump_end)})",
        f"⏱ ТФ: {timeframe}м | Биржа: Bybit | Режим: {mode}",
        f"📍 Цена: {_fmt_price(price)}",
        f"📊 RSI ({timeframe}м): {rsi:.2f}",
        f"📦 Объём: {volume_spike:.2f}x | Слабость: {weakness:.2f}",
    ]
    if continuation_max_score > 0:
        lines.append(f"⚠️ Риск продолжения: {continuation_risk:.1f}/{continuation_max_score:.1f}")
    if window_open is not None and window_close is not None:
        lines.append(f"🕯 {window_label}: open {_fmt_price(window_open)} / close {_fmt_price(window_close)}")

    if clean_quality_grade and quality_max_score > 0:
        quality_line = f"• Класс сетапа: {clean_quality_grade} ({quality_score:.1f}/{quality_max_score:.1f})"
    elif quality_max_score > 0:
        quality_line = f"• Setup score: {quality_score:.1f}/{quality_max_score:.1f}"
    else:
        quality_line = f"• Setup score: {quality_score:.1f}"

    lines.extend(
        [
            "",
            f"📊 <b>Контекст по монете #{asset}</b>",
            (
                f"• Watch score: {watch_score:.1f}/{watch_max_score:.1f}"
                if watch_max_score > 0
                else f"• Watch score: {watch_score:.1f}"
            ),
            quality_line,
            f"• Чистый памп: {pump_size * 100.0:.2f}% при минимуме {pump_min * 100.0:.2f}%",
        ]
    )
    if clean_triggers:
        lines.append(f"• Триггеры: {', '.join(clean_triggers)}")
    if clean_wait_for:
        lines.append(f"• Ждём: {clean_wait_for}")
    return "\n".join(lines)


def build_early_invalidation_text(*, symbol: str, timeframe: str, mode: str, reason: str) -> str:
    asset = _base_asset(symbol)
    clean_reason = _normalize_human_text(reason)
    lines = [
        f"<b>{EARLY_INVALIDATED_LABEL}</b>",
        f"<b>{asset}</b> | <code>{symbol}</code>",
        f"⏱ ТФ: {timeframe}м | Режим: {mode}",
    ]
    if clean_reason:
        lines.append(f"Причина: {clean_reason}")
    return "\n".join(lines)
