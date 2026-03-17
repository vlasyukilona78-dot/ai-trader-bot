from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

PIPELINE_ORDER: tuple[str, ...] = (
    "regime_filter",
    "layer1_pump_detection",
    "layer2_weakness_confirmation",
    "layer3_entry_location",
    "layer4_fake_filter",
    "layer5_tp_sl",
)

REGIME_BLOCKERS: tuple[str, ...] = (
    "htf_trend_ok",
    "stretched_from_vwap",
    "volatility_regime_ok",
    "news_veto",
)

REGIME_BLOCKER_COMPACT_FIELDS: dict[str, str] = {
    "htf_trend_ok": "regime_filter_htf_trend_blocker_count",
    "stretched_from_vwap": "regime_filter_vwap_stretch_blocker_count",
    "volatility_regime_ok": "regime_filter_volatility_blocker_count",
    "news_veto": "regime_filter_news_blocker_count",
}

LAYER1_BLOCKERS: tuple[str, ...] = (
    "rsi_high",
    "volume_spike",
    "above_bollinger_upper",
    "above_keltner_upper",
)

LAYER4_BLOCKERS: tuple[str, ...] = (
    "price_above_vwap",
    "sentiment_euphoric",
    "funding_supports_short",
    "long_short_ratio_extreme",
    "oi_overheated",
)


@dataclass(frozen=True)
class CalibrationGuardrails:
    min_samples: int = 30
    comparable_size_ratio_min: float = 0.6
    comparable_size_ratio_max: float = 1.67
    min_dominance_share: float = 0.55
    max_dominance_share_delta: float = 0.25
    blocker_shift_score_warn: float = 0.80
    co_dominant_relative_min: float = 0.85


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _rate(num: float | int, den: float | int) -> float:
    den_value = _safe_float(den, 0.0)
    if den_value <= 0.0:
        return 0.0
    return _safe_float(num, 0.0) / den_value


def _extract_literal(msg: str, key: str) -> str | None:
    token = key + "="
    idx = msg.find(token)
    if idx < 0:
        return None
    start = idx + len(token)
    if start >= len(msg):
        return None

    opening = msg[start]
    if opening not in "{[":
        return None
    closing = "}" if opening == "{" else "]"

    depth = 0
    quote: str | None = None
    escaped = False

    for i in range(start, len(msg)):
        ch = msg[i]
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return msg[start : i + 1]

    return None


def _parse_mapping_literal(msg: str, key: str) -> dict[str, Any]:
    raw = _extract_literal(msg, key)
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def parse_observation_extract(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: list[dict[str, Any]] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue

        msg = str(payload.get("msg", ""))
        compact = _parse_mapping_literal(msg, "strategy_audit_compact")
        if not compact:
            continue

        record = {
            "ts": str(payload.get("ts", "")),
            "compact": compact,
            "regime": _parse_mapping_literal(msg, "strategy_audit_regime_filter"),
            "layer1": _parse_mapping_literal(msg, "strategy_audit_layer1"),
            "layer4": _parse_mapping_literal(msg, "strategy_audit_layer4"),
            "source_quality": _parse_mapping_literal(msg, "strategy_audit_source_quality"),
            "audit": _parse_mapping_literal(msg, "strategy_audit"),
        }
        records.append(record)

    return records


def _sum_source_quality(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    out = {"live": 0, "fallback": 0, "unavailable": 0}
    for item in records:
        compact = item.get("compact", {})
        src = compact.get(key, {}) if isinstance(compact, Mapping) else {}
        if not isinstance(src, Mapping):
            continue
        out["live"] += _safe_int(src.get("live", 0))
        out["fallback"] += _safe_int(src.get("fallback", 0))
        out["unavailable"] += _safe_int(src.get("unavailable", 0))
    return out


def _aggregate_failed_layer_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in records:
        audit = item.get("audit", {})
        raw_counts = audit.get("failed_layer_counts", {}) if isinstance(audit, Mapping) else {}
        if not isinstance(raw_counts, Mapping):
            continue
        for name, value in raw_counts.items():
            key = str(name)
            out[key] = int(out.get(key, 0)) + _safe_int(value)
    return dict(sorted(out.items()))


def _top_key_count(counter: Mapping[str, Any]) -> tuple[str, int]:
    if not isinstance(counter, Mapping) or not counter:
        return "", 0
    top_key = ""
    top_count = -1
    for key, value in counter.items():
        count = _safe_int(value)
        if count > top_count:
            top_key = str(key)
            top_count = count
    return top_key, max(top_count, 0)


def _dominance_share(counter: Mapping[str, Any]) -> float:
    if not isinstance(counter, Mapping) or not counter:
        return 0.0
    total = float(sum(_safe_int(v) for v in counter.values()))
    if total <= 0.0:
        return 0.0
    _, top_count = _top_key_count(counter)
    return float(top_count) / total


def _top_share(counter: Mapping[str, Any], total: int) -> float:
    if total <= 0:
        return 0.0
    _, top_count = _top_key_count(counter)
    return _rate(top_count, total)


def _dominant_keys(counter: Mapping[str, Any], *, relative_min: float) -> list[str]:
    if not isinstance(counter, Mapping) or not counter:
        return []
    ranked = sorted(((str(k), _safe_int(v)) for k, v in counter.items()), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return []
    top_count = float(ranked[0][1])
    out: list[str] = []
    for key, count in ranked:
        if count <= 0:
            continue
        if (float(count) / top_count) >= float(relative_min):
            out.append(key)
    return out


def _sorted_counts(counter: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(counter, Mapping):
        return {}
    pairs = ((str(k), _safe_int(v)) for k, v in counter.items())
    return dict(sorted(pairs, key=lambda item: (-item[1], item[0])))


def _combination_label(blockers: list[str]) -> str:
    clean = [str(item) for item in blockers if str(item)]
    if not clean:
        return "none"
    return " + ".join(clean)


def _active_regime_blockers_from_compact(compact: Mapping[str, Any]) -> list[str]:
    active: list[str] = []
    for blocker in REGIME_BLOCKERS:
        field = REGIME_BLOCKER_COMPACT_FIELDS.get(blocker, "")
        if field and _safe_int(compact.get(field, 0)) > 0:
            active.append(blocker)
    return active


def _regime_dominance_semantics(
    raw_counts: Mapping[str, Any],
    label_counts: Mapping[str, Any],
    *,
    fail_count: int,
    sample_count: int,
    guardrails: CalibrationGuardrails | None = None,
) -> dict[str, Any]:
    cfg = guardrails or CalibrationGuardrails()
    top_raw, top_raw_count = _top_key_count(raw_counts)
    top_label, top_label_count = _top_key_count(label_counts)
    top_raw_share = _rate(top_raw_count, fail_count)
    top_label_share = _rate(top_label_count, sample_count)
    co_dominant = _dominant_keys(raw_counts, relative_min=float(cfg.co_dominant_relative_min))
    disagreement = bool(top_raw and top_label and top_raw != top_label)

    dominance_mode = "single_dominant_raw"
    dominance_explanation = (
        f"raw_top={top_raw or 'none'} share={top_raw_share:.3f}; "
        f"label_top={top_label or 'none'} share={top_label_share:.3f}."
    )
    if disagreement:
        dominance_mode = "raw_vs_label_disagreement"
        dominance_explanation = (
            f"raw coverage favors {top_raw or 'none'} ({top_raw_share:.3f}) "
            f"while snapshot labels favor {top_label or 'none'} ({top_label_share:.3f})."
        )
    elif len(co_dominant) >= 2:
        dominance_mode = "co_dominant_raw"
        dominance_explanation = (
            f"raw coverage is co-dominant across {', '.join(co_dominant)} "
            f"with top share {top_raw_share:.3f}."
        )
    elif not top_raw:
        dominance_mode = "no_dominant_blocker"
        dominance_explanation = "No meaningful raw blocker coverage was detected."

    return {
        "top_blocker_by_raw_coverage": top_raw,
        "top_blocker_by_raw_coverage_count": int(top_raw_count),
        "top_blocker_by_raw_coverage_share": float(top_raw_share),
        "top_blocker_by_label": top_label,
        "top_blocker_by_label_count": int(top_label_count),
        "top_blocker_by_label_share": float(top_label_share),
        "co_dominant_blockers": co_dominant,
        "dominance_mode": dominance_mode,
        "dominance_explanation": dominance_explanation,
    }


def _blocker_rates_per_sample(counter: Mapping[str, Any], samples: int) -> dict[str, float]:
    return {str(k): _rate(_safe_int(v), samples) for k, v in counter.items()}


def _blocker_rates_per_fail(counter: Mapping[str, Any], fail_count: int) -> dict[str, float]:
    return {str(k): _rate(_safe_int(v), fail_count) for k, v in counter.items()}


def aggregate_observation(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples": len(records),
        "evaluations_total": 0,
        "insufficient_history_count": 0,
        "regime_filter_pass_count": 0,
        "regime_filter_fail_count": 0,
        "reached_layer1_count": 0,
        "layer1_pass_count": 0,
        "layer1_fail_count": 0,
        "layer2_fail_count": 0,
        "layer3_fail_count": 0,
        "layer4_fail_count": 0,
        "layer5_fail_count": 0,
        "short_signal_count": 0,
        "no_signal_count": 0,
        "regime_filter_htf_trend_blocker_count": 0,
        "regime_filter_vwap_stretch_blocker_count": 0,
        "regime_filter_volatility_blocker_count": 0,
        "regime_filter_news_blocker_count": 0,
        "regime_filter_degraded_only_count": 0,
        "regime_filter_soft_pass_candidate_count": 0,
        "layer1_rsi_high_blocker_count": 0,
        "layer1_volume_spike_blocker_count": 0,
        "layer1_above_bollinger_upper_blocker_count": 0,
        "layer1_above_keltner_upper_blocker_count": 0,
        "layer4_sentiment_blocker_count": 0,
        "layer4_funding_blocker_count": 0,
        "layer4_lsr_blocker_count": 0,
        "layer4_oi_blocker_count": 0,
        "layer4_price_blocker_count": 0,
        "layer4_degraded_mode_count": 0,
        "layer4_soft_pass_candidate_count": 0,
        "layer5_fallback_rr_used_count": 0,
        "layer5_vp_based_count": 0,
        "layer5_fail_missing_atr_count": 0,
        "layer5_fail_missing_volume_profile_count": 0,
        "first_ts": records[0]["ts"] if records else "",
        "last_ts": records[-1]["ts"] if records else "",
    }
    regime_blocker_label_counts = {name: 0 for name in REGIME_BLOCKERS}
    regime_blocker_combination_counts: dict[str, int] = {}
    regime_filter_failed_sample_count = 0

    for item in records:
        compact = item.get("compact", {})
        audit = item.get("audit", {})
        summary["evaluations_total"] += _safe_int(compact.get("evaluations_total", 0), 1)
        summary["insufficient_history_count"] += _safe_int(compact.get("insufficient_history_count", 0))
        summary["regime_filter_pass_count"] += _safe_int(compact.get("regime_filter_pass_count", 0))
        summary["regime_filter_fail_count"] += _safe_int(compact.get("regime_filter_fail_count", 0))
        summary["layer1_pass_count"] += _safe_int(compact.get("layer1_pass_count", 0))
        summary["layer1_fail_count"] += _safe_int(compact.get("layer1_fail_count", 0))
        summary["layer2_fail_count"] += _safe_int(compact.get("layer2_fail_count", 0))
        summary["layer3_fail_count"] += _safe_int(compact.get("layer3_fail_count", 0))
        summary["layer4_fail_count"] += _safe_int(compact.get("layer4_fail_count", 0))
        summary["layer5_fail_count"] += _safe_int(compact.get("layer5_fail_count", 0))
        summary["short_signal_count"] += _safe_int(compact.get("short_signal_count", 0))
        summary["no_signal_count"] += _safe_int(compact.get("no_signal_count", 0))
        summary["regime_filter_htf_trend_blocker_count"] += _safe_int(compact.get("regime_filter_htf_trend_blocker_count", 0))
        summary["regime_filter_vwap_stretch_blocker_count"] += _safe_int(compact.get("regime_filter_vwap_stretch_blocker_count", 0))
        summary["regime_filter_volatility_blocker_count"] += _safe_int(compact.get("regime_filter_volatility_blocker_count", 0))
        summary["regime_filter_news_blocker_count"] += _safe_int(compact.get("regime_filter_news_blocker_count", 0))
        summary["regime_filter_degraded_only_count"] += _safe_int(compact.get("regime_filter_degraded_only_count", 0))
        summary["regime_filter_soft_pass_candidate_count"] += _safe_int(compact.get("regime_filter_soft_pass_candidate_count", 0))
        summary["layer1_rsi_high_blocker_count"] += _safe_int(compact.get("layer1_rsi_high_blocker_count", 0))
        summary["layer1_volume_spike_blocker_count"] += _safe_int(compact.get("layer1_volume_spike_blocker_count", 0))
        summary["layer1_above_bollinger_upper_blocker_count"] += _safe_int(compact.get("layer1_above_bollinger_upper_blocker_count", 0))
        summary["layer1_above_keltner_upper_blocker_count"] += _safe_int(compact.get("layer1_above_keltner_upper_blocker_count", 0))
        summary["layer4_sentiment_blocker_count"] += _safe_int(compact.get("layer4_sentiment_blocker_count", 0))
        summary["layer4_funding_blocker_count"] += _safe_int(compact.get("layer4_funding_blocker_count", 0))
        summary["layer4_lsr_blocker_count"] += _safe_int(compact.get("layer4_lsr_blocker_count", 0))
        summary["layer4_oi_blocker_count"] += _safe_int(compact.get("layer4_oi_blocker_count", 0))
        summary["layer4_price_blocker_count"] += _safe_int(compact.get("layer4_price_blocker_count", 0))
        summary["layer4_degraded_mode_count"] += _safe_int(compact.get("layer4_degraded_mode_count", 0))
        summary["layer4_soft_pass_candidate_count"] += _safe_int(compact.get("layer4_soft_pass_candidate_count", 0))
        summary["layer5_fallback_rr_used_count"] += _safe_int(compact.get("layer5_fallback_rr_used_count", 0))
        summary["layer5_vp_based_count"] += _safe_int(compact.get("layer5_vp_based_count", 0))
        summary["layer5_fail_missing_atr_count"] += _safe_int(compact.get("layer5_fail_missing_atr_count", 0))
        summary["layer5_fail_missing_volume_profile_count"] += _safe_int(compact.get("layer5_fail_missing_volume_profile_count", 0))
        summary["reached_layer1_count"] += _safe_int(audit.get("reached_layer1_count", 0))
        blocker_label = str(compact.get("top_regime_filter_blocker", "")).strip()
        if blocker_label in regime_blocker_label_counts:
            regime_blocker_label_counts[blocker_label] += 1
        if _safe_int(compact.get("regime_filter_fail_count", 0)) > 0:
            regime_filter_failed_sample_count += 1
            combo = _combination_label(_active_regime_blockers_from_compact(compact))
            regime_blocker_combination_counts[combo] = int(regime_blocker_combination_counts.get(combo, 0)) + 1

    if summary["evaluations_total"] <= 0:
        summary["evaluations_total"] = summary["samples"]

    summary["no_signal_ratio"] = _rate(summary["no_signal_count"], summary["evaluations_total"])
    summary["short_signal_ratio"] = _rate(summary["short_signal_count"], summary["evaluations_total"])
    summary["regime_filter_pass_rate"] = _rate(summary["regime_filter_pass_count"], summary["evaluations_total"])
    summary["regime_filter_fail_rate"] = _rate(summary["regime_filter_fail_count"], summary["evaluations_total"])
    summary["layer1_reach_rate"] = _rate(summary["reached_layer1_count"], summary["evaluations_total"])
    summary["layer1_fail_rate_given_reach"] = _rate(summary["layer1_fail_count"], summary["reached_layer1_count"])

    regime_blocker_counts = {
        "htf_trend_ok": summary["regime_filter_htf_trend_blocker_count"],
        "stretched_from_vwap": summary["regime_filter_vwap_stretch_blocker_count"],
        "volatility_regime_ok": summary["regime_filter_volatility_blocker_count"],
        "news_veto": summary["regime_filter_news_blocker_count"],
    }
    layer1_blocker_counts = {
        "rsi_high": summary["layer1_rsi_high_blocker_count"],
        "volume_spike": summary["layer1_volume_spike_blocker_count"],
        "above_bollinger_upper": summary["layer1_above_bollinger_upper_blocker_count"],
        "above_keltner_upper": summary["layer1_above_keltner_upper_blocker_count"],
    }
    layer4_blocker_counts = {
        "price_above_vwap": summary["layer4_price_blocker_count"],
        "sentiment_euphoric": summary["layer4_sentiment_blocker_count"],
        "funding_supports_short": summary["layer4_funding_blocker_count"],
        "long_short_ratio_extreme": summary["layer4_lsr_blocker_count"],
        "oi_overheated": summary["layer4_oi_blocker_count"],
    }

    failed_layer_counts = _aggregate_failed_layer_counts(records)
    if not failed_layer_counts:
        failed_layer_counts = {
            "regime_filter": summary["regime_filter_fail_count"],
            "layer1_pump_detection": summary["layer1_fail_count"],
            "layer2_weakness_confirmation": summary["layer2_fail_count"],
            "layer3_entry_location": summary["layer3_fail_count"],
            "layer4_fake_filter": summary["layer4_fail_count"],
            "layer5_tp_sl": summary["layer5_fail_count"],
        }
        failed_layer_counts = {k: v for k, v in failed_layer_counts.items() if int(v) > 0}

    top_failed_layer, top_failed_count = _top_key_count(failed_layer_counts)
    top_regime_blocker, top_regime_blocker_count = _top_key_count(regime_blocker_counts)
    top_regime_combination, top_regime_combination_count = _top_key_count(regime_blocker_combination_counts)
    top_layer1_blocker, top_layer1_blocker_count = _top_key_count(layer1_blocker_counts)
    top_layer4_blocker, top_layer4_blocker_count = _top_key_count(layer4_blocker_counts)

    summary["failed_layer_counts"] = failed_layer_counts
    summary["top_failed_layer"] = top_failed_layer
    summary["top_failed_count"] = int(top_failed_count)
    summary["top_regime_filter_blocker"] = top_regime_blocker
    summary["top_regime_filter_blocker_count"] = int(top_regime_blocker_count)
    summary["top_layer1_blocker"] = top_layer1_blocker
    summary["top_layer1_blocker_count"] = int(top_layer1_blocker_count)
    summary["top_layer4_blocker"] = top_layer4_blocker
    summary["top_layer4_blocker_count"] = int(top_layer4_blocker_count)
    summary["regime_filter_blocker_label_counts"] = dict(sorted(regime_blocker_label_counts.items()))
    summary["regime_filter_failed_sample_count"] = int(regime_filter_failed_sample_count)
    summary["regime_filter_blocker_combination_counts"] = _sorted_counts(regime_blocker_combination_counts)
    summary["top_regime_filter_blocker_combination"] = top_regime_combination
    summary["top_regime_filter_blocker_combination_count"] = int(top_regime_combination_count)
    summary["top_regime_filter_blocker_combination_share"] = _rate(
        top_regime_combination_count,
        regime_filter_failed_sample_count,
    )

    summary["blocker_rate_per_sample"] = {
        "regime_filter": _blocker_rates_per_sample(regime_blocker_counts, summary["evaluations_total"]),
        "layer1_pump_detection": _blocker_rates_per_sample(layer1_blocker_counts, summary["evaluations_total"]),
        "layer4_fake_filter": _blocker_rates_per_sample(layer4_blocker_counts, summary["evaluations_total"]),
    }
    summary["blocker_rate_per_regime_fail"] = _blocker_rates_per_fail(regime_blocker_counts, summary["regime_filter_fail_count"])
    summary["regime_filter_blocker_combination_share_by_failed_samples"] = _blocker_rates_per_fail(
        regime_blocker_combination_counts,
        regime_filter_failed_sample_count,
    )
    summary["blocker_dominance_share"] = {
        "regime_filter": _dominance_share(regime_blocker_counts),
        "layer1_pump_detection": _dominance_share(layer1_blocker_counts),
        "layer4_fake_filter": _dominance_share(layer4_blocker_counts),
    }
    summary["top_regime_filter_blocker_share"] = _rate(top_regime_blocker_count, summary["regime_filter_fail_count"])

    summary["source_quality_summary"] = {
        "regime_filter": _sum_source_quality(records, "regime_source_quality"),
        "layer4_fake_filter": _sum_source_quality(records, "layer4_source_quality"),
    }

    dominance = _regime_dominance_semantics(
        regime_blocker_counts,
        regime_blocker_label_counts,
        fail_count=int(summary["regime_filter_fail_count"]),
        sample_count=int(summary["samples"]),
    )
    summary.update(dominance)
    return summary


def _window_size_ratio(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    before_eval = _safe_float(before.get("evaluations_total", before.get("samples", 0)))
    after_eval = _safe_float(after.get("evaluations_total", after.get("samples", 0)))
    if before_eval <= 0.0:
        return 0.0
    return after_eval / before_eval


def _regime_blocker_shift_score(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    before_rates = before.get("blocker_rate_per_regime_fail", {})
    after_rates = after.get("blocker_rate_per_regime_fail", {})
    score = 0.0
    for blocker in REGIME_BLOCKERS:
        b = _safe_float(before_rates.get(blocker, 0.0))
        a = _safe_float(after_rates.get(blocker, 0.0))
        score += abs(a - b)
    return score


def assess_window_quality(
    after_summary: Mapping[str, Any],
    before_summary: Mapping[str, Any] | None = None,
    guardrails: CalibrationGuardrails | None = None,
) -> dict[str, Any]:
    cfg = guardrails or CalibrationGuardrails()
    evals = _safe_int(after_summary.get("evaluations_total", after_summary.get("samples", 0)))
    enough_data = evals >= int(cfg.min_samples)

    quality: dict[str, Any] = {
        "sample_count": evals,
        "enough_data": bool(enough_data),
        "comparable_window_size": False,
        "window_size_ratio": 0.0,
        "market_regime_shift_warning": False,
        "blocker_shift_score": 0.0,
        "dominant_blocker_stable": False,
        "dominant_blocker_changed": False,
        "raw_vs_label_disagreement": False,
        "co_dominant_blockers": [],
        "decision_confidence_score": 0.0,
        "decision_confidence": "low",
    }

    if before_summary is None:
        score = 0.35 if enough_data else 0.1
        quality["decision_confidence_score"] = score
        quality["decision_confidence"] = "low" if score < 0.45 else "medium"
        return quality

    size_ratio = _window_size_ratio(before_summary, after_summary)
    comparable = (
        size_ratio >= float(cfg.comparable_size_ratio_min)
        and size_ratio <= float(cfg.comparable_size_ratio_max)
    )
    shift_score = _regime_blocker_shift_score(before_summary, after_summary)
    market_shift = shift_score >= float(cfg.blocker_shift_score_warn)

    before_top = str(before_summary.get("top_regime_filter_blocker", ""))
    after_top = str(after_summary.get("top_regime_filter_blocker", ""))
    before_dom = _safe_float(before_summary.get("blocker_dominance_share", {}).get("regime_filter", 0.0))
    after_dom = _safe_float(after_summary.get("blocker_dominance_share", {}).get("regime_filter", 0.0))
    top_changed = bool(before_top) and bool(after_top) and before_top != after_top
    dom_delta = abs(after_dom - before_dom)
    dominant_stable = (not top_changed) and (dom_delta <= float(cfg.max_dominance_share_delta))

    score = 0.0
    if enough_data:
        score += 0.35
    if comparable:
        score += 0.25
    if not market_shift:
        score += 0.20
    if dominant_stable:
        score += 0.20

    label = "low"
    if score >= 0.75:
        label = "high"
    elif score >= 0.45:
        label = "medium"

    quality.update(
        {
            "comparable_window_size": bool(comparable),
            "window_size_ratio": float(size_ratio),
            "market_regime_shift_warning": bool(market_shift),
            "blocker_shift_score": float(shift_score),
            "dominant_blocker_stable": bool(dominant_stable),
            "dominant_blocker_changed": bool(top_changed),
            "raw_vs_label_disagreement": bool(
                str(after_summary.get("top_blocker_by_raw_coverage", ""))
                and str(after_summary.get("top_blocker_by_label", ""))
                and str(after_summary.get("top_blocker_by_raw_coverage", ""))
                != str(after_summary.get("top_blocker_by_label", ""))
            ),
            "co_dominant_blockers": list(after_summary.get("co_dominant_blockers", []))
            if isinstance(after_summary.get("co_dominant_blockers", []), list)
            else [],
            "decision_confidence_score": float(score),
            "decision_confidence": label,
        }
    )
    return quality


def _earliest_failed_layer(summary: Mapping[str, Any]) -> str:
    fail_counts = {
        "regime_filter": _safe_int(summary.get("regime_filter_fail_count", 0)),
        "layer1_pump_detection": _safe_int(summary.get("layer1_fail_count", 0)),
        "layer2_weakness_confirmation": _safe_int(summary.get("layer2_fail_count", 0)),
        "layer3_entry_location": _safe_int(summary.get("layer3_fail_count", 0)),
        "layer4_fake_filter": _safe_int(summary.get("layer4_fail_count", 0)),
        "layer5_tp_sl": _safe_int(summary.get("layer5_fail_count", 0)),
    }
    for layer in PIPELINE_ORDER:
        if fail_counts.get(layer, 0) > 0:
            return layer
    return ""


def _layer_blocker_counts(summary: Mapping[str, Any], layer: str) -> dict[str, int]:
    if layer == "regime_filter":
        return {
            "htf_trend_ok": _safe_int(summary.get("regime_filter_htf_trend_blocker_count", 0)),
            "stretched_from_vwap": _safe_int(summary.get("regime_filter_vwap_stretch_blocker_count", 0)),
            "volatility_regime_ok": _safe_int(summary.get("regime_filter_volatility_blocker_count", 0)),
            "news_veto": _safe_int(summary.get("regime_filter_news_blocker_count", 0)),
        }
    if layer == "layer1_pump_detection":
        return {
            "rsi_high": _safe_int(summary.get("layer1_rsi_high_blocker_count", 0)),
            "volume_spike": _safe_int(summary.get("layer1_volume_spike_blocker_count", 0)),
            "above_bollinger_upper": _safe_int(summary.get("layer1_above_bollinger_upper_blocker_count", 0)),
            "above_keltner_upper": _safe_int(summary.get("layer1_above_keltner_upper_blocker_count", 0)),
        }
    if layer == "layer4_fake_filter":
        return {
            "price_above_vwap": _safe_int(summary.get("layer4_price_blocker_count", 0)),
            "sentiment_euphoric": _safe_int(summary.get("layer4_sentiment_blocker_count", 0)),
            "funding_supports_short": _safe_int(summary.get("layer4_funding_blocker_count", 0)),
            "long_short_ratio_extreme": _safe_int(summary.get("layer4_lsr_blocker_count", 0)),
            "oi_overheated": _safe_int(summary.get("layer4_oi_blocker_count", 0)),
        }
    return {}


def _top_combination_context(summary: Mapping[str, Any]) -> str:
    combo = str(summary.get("top_regime_filter_blocker_combination", "")).strip()
    if not combo or combo == "none":
        return ""
    share = _safe_float(summary.get("top_regime_filter_blocker_combination_share", 0.0))
    return f" Most common blocker combination is {combo} ({share:.3f} of failed samples)."


def _action_verdict(stop_reason: str, safe_to_continue: bool) -> str:
    if bool(safe_to_continue):
        return "single_blocker_ready"
    if stop_reason in {"co_dominant_regime_blockers", "blocker_semantics_disagreement"}:
        return "co_dominant_overlap"
    return "pause_calibration"


def _build_runbook_actions(
    recommendation: Mapping[str, Any],
    after_summary: Mapping[str, Any],
    before_summary: Mapping[str, Any] | None = None,
) -> list[str]:
    stop_reason = str(recommendation.get("STOP_REASON", ""))
    action_verdict = str(recommendation.get("ACTION_VERDICT", ""))
    target_layer = str(recommendation.get("NEXT_TARGET_LAYER", ""))
    target_subcondition = str(recommendation.get("NEXT_TARGET_SUBCONDITION", ""))
    combo = str(after_summary.get("top_regime_filter_blocker_combination", "")).strip()

    if action_verdict == "single_blocker_ready":
        return [
            f"Prepare one isolated tweak for {target_layer}.{target_subcondition}.",
            "Keep all other thresholds unchanged during the next calibration step.",
            "Collect a post-change validation window with comparable size before any further tweak.",
        ]

    if stop_reason == "previous_validated_window_missing":
        return [
            "Locate the most recent validated BEFORE window for this calibration branch.",
            "Rerun the BEFORE/AFTER comparison before making another threshold change.",
            "Keep thresholds unchanged until the comparison baseline exists.",
        ]

    if stop_reason in {"insufficient_after_window_data", "insufficient_previous_window_data"}:
        return [
            "Collect a longer observation window before making another threshold change.",
            "Use only windows that satisfy the minimum sample guardrail.",
            "Keep thresholds unchanged while sample coverage is insufficient.",
        ]

    if stop_reason == "window_size_not_comparable":
        return [
            "Collect another window with a comparable sample count to the reference window.",
            "Compare like-for-like windows before deciding on another tweak.",
            "Keep thresholds unchanged while the windows are not comparable.",
        ]

    if stop_reason == "market_context_shift_detected":
        return [
            "Pause threshold changes for this branch.",
            "Review the regime blocker shift before selecting any next target.",
            "Collect a fresh baseline window if the market context has materially changed.",
        ]

    if stop_reason == "upstream_bottleneck_closed_regime_filter":
        return [
            "Do not tune downstream layers while regime_filter pass rate is zero.",
            "Focus only on regime_filter blockers until some samples reach lower layers.",
            "Keep lower-layer thresholds unchanged.",
        ]

    if stop_reason == "review_semantics_no_subcondition_signal":
        return [
            "Review audit semantics for the earliest failed layer.",
            "Confirm the blocker counters are being emitted consistently before tuning.",
            "Keep thresholds unchanged until a dominant subcondition is visible.",
        ]

    if stop_reason == "co_dominant_regime_blockers":
        actions = ["Pause threshold changes for this branch."]
        if combo and combo != "none":
            actions.append(f"Review blocker combination '{combo}' as the current overlap driver.")
        actions.extend(
            [
                "Collect another comparable window before attempting a single-threshold tweak.",
                "Do not tune one regime threshold while blocker overlap remains co-dominant.",
            ]
        )
        return actions

    if stop_reason == "blocker_semantics_disagreement":
        actions = ["Pause threshold changes for this branch."]
        if combo and combo != "none":
            actions.append(f"Review blocker combination '{combo}' alongside the raw-vs-label split.")
        actions.extend(
            [
                "Inspect raw coverage and snapshot label semantics before choosing a target.",
                "Do not tune one regime threshold while blocker semantics disagree.",
            ]
        )
        return actions

    if stop_reason in {"dominant_blocker_unstable", "observe_more_dominance_too_low"}:
        return [
            "Collect another comparable observation window before tuning.",
            "Wait for one blocker to show stable dominance across windows.",
            "Keep thresholds unchanged while blocker leadership is unstable.",
        ]

    if stop_reason == "no_failed_layer_detected":
        return [
            "Review the latest observation output for missing or inconsistent failed-layer data.",
            "Confirm the audit pipeline is still emitting blocker counters correctly.",
            "Keep thresholds unchanged until a failed layer is detectable.",
        ]

    return [
        "Pause threshold changes for now.",
        "Review the latest calibration recommendation context before the next tweak.",
        "Collect another observation window if the next action is still unclear.",
    ]


def _finalize_recommendation(
    recommendation: dict[str, Any],
    after_summary: Mapping[str, Any],
    before_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    recommendation["ACTION_VERDICT"] = _action_verdict(
        str(recommendation.get("STOP_REASON", "")),
        bool(recommendation.get("SAFE_TO_CONTINUE", False)),
    )
    recommendation["RUNBOOK_ACTIONS"] = _build_runbook_actions(recommendation, after_summary, before_summary)
    return recommendation


def recommend_calibration_step(
    after_summary: Mapping[str, Any],
    before_summary: Mapping[str, Any] | None = None,
    guardrails: CalibrationGuardrails | None = None,
) -> dict[str, Any]:
    cfg = guardrails or CalibrationGuardrails()
    quality = assess_window_quality(after_summary, before_summary, cfg)
    recommendation: dict[str, Any] = {
        "SAFE_TO_CONTINUE": False,
        "ACTION_VERDICT": "pause_calibration",
        "RUNBOOK_ACTIONS": [],
        "NEXT_TARGET_LAYER": "none",
        "NEXT_TARGET_SUBCONDITION": "none",
        "WHY_THIS_TARGET": "",
        "WHY_NOT_OTHERS": [],
        "REQUIRED_WINDOW_QUALITY": quality,
        "STOP_REASON": "",
    }

    if before_summary is None:
        recommendation["STOP_REASON"] = "previous_validated_window_missing"
        recommendation["WHY_NOT_OTHERS"] = ["Need BEFORE/AFTER comparison to avoid overtuning."]
        return _finalize_recommendation(recommendation, after_summary, before_summary)

    if not bool(quality.get("enough_data", False)):
        recommendation["STOP_REASON"] = "insufficient_after_window_data"
        recommendation["WHY_NOT_OTHERS"] = ["After-window sample size is too small for safe calibration."]
        return _finalize_recommendation(recommendation, after_summary, before_summary)

    before_evals = _safe_int(before_summary.get("evaluations_total", before_summary.get("samples", 0)))
    if before_evals < int(cfg.min_samples):
        recommendation["STOP_REASON"] = "insufficient_previous_window_data"
        recommendation["WHY_NOT_OTHERS"] = ["Previous validated window has insufficient sample size."]
        return _finalize_recommendation(recommendation, after_summary, before_summary)

    if not bool(quality.get("comparable_window_size", False)):
        recommendation["STOP_REASON"] = "window_size_not_comparable"
        recommendation["WHY_NOT_OTHERS"] = ["Window sizes are not comparable; observe more before next tweak."]
        return _finalize_recommendation(recommendation, after_summary, before_summary)

    if bool(quality.get("market_regime_shift_warning", False)):
        recommendation["STOP_REASON"] = "market_context_shift_detected"
        recommendation["WHY_NOT_OTHERS"] = ["Blocker structure changed abruptly across windows; review semantics first."]
        return _finalize_recommendation(recommendation, after_summary, before_summary)

    earliest_layer = _earliest_failed_layer(after_summary)
    if not earliest_layer:
        recommendation["STOP_REASON"] = "no_failed_layer_detected"
        recommendation["WHY_NOT_OTHERS"] = ["No dominant failed layer detected in the current window."]
        return _finalize_recommendation(recommendation, after_summary, before_summary)

    if _safe_float(after_summary.get("regime_filter_pass_rate", 0.0)) <= 0.0 and earliest_layer != "regime_filter":
        recommendation["STOP_REASON"] = "upstream_bottleneck_closed_regime_filter"
        recommendation["WHY_NOT_OTHERS"] = ["Regime filter pass rate is zero; lower-layer calibration is forbidden."]
        return _finalize_recommendation(recommendation, after_summary, before_summary)

    blocker_counts = _layer_blocker_counts(after_summary, earliest_layer)
    target_subcondition, target_count = _top_key_count(blocker_counts)
    if not target_subcondition or target_count <= 0:
        recommendation["STOP_REASON"] = "review_semantics_no_subcondition_signal"
        recommendation["WHY_NOT_OTHERS"] = ["Layer has failures but no dominant subcondition counter."]
        return _finalize_recommendation(recommendation, after_summary, before_summary)

    if earliest_layer == "regime_filter":
        co_dominant = after_summary.get("co_dominant_blockers", [])
        if isinstance(co_dominant, list) and len(co_dominant) >= 2:
            recommendation["STOP_REASON"] = "co_dominant_regime_blockers"
            recommendation["WHY_NOT_OTHERS"] = [
                "Regime raw blocker coverage is co-dominant across "
                + ", ".join(str(item) for item in co_dominant)
                + "; isolate semantics before another tweak."
                + _top_combination_context(after_summary)
            ]
            return _finalize_recommendation(recommendation, after_summary, before_summary)

        raw_blocker = str(after_summary.get("top_blocker_by_raw_coverage", after_summary.get("top_regime_filter_blocker", "")))
        label_blocker = str(after_summary.get("top_blocker_by_label", ""))
        if raw_blocker and label_blocker and raw_blocker != label_blocker:
            recommendation["STOP_REASON"] = "blocker_semantics_disagreement"
            recommendation["WHY_NOT_OTHERS"] = [
                f"Raw coverage favors {raw_blocker} while snapshot labels favor {label_blocker}; do not tune on ambiguous blocker semantics."
                + _top_combination_context(after_summary)
            ]
            return _finalize_recommendation(recommendation, after_summary, before_summary)

        before_blocker = str(before_summary.get("top_regime_filter_blocker", ""))
        after_blocker = str(after_summary.get("top_regime_filter_blocker", ""))
        before_dom = _safe_float(before_summary.get("blocker_dominance_share", {}).get("regime_filter", 0.0))
        after_dom = _safe_float(after_summary.get("blocker_dominance_share", {}).get("regime_filter", 0.0))
        if (
            before_blocker
            and after_blocker
            and before_blocker != after_blocker
            and abs(after_dom - before_dom) > float(cfg.max_dominance_share_delta)
        ):
            recommendation["STOP_REASON"] = "dominant_blocker_unstable"
            recommendation["WHY_NOT_OTHERS"] = ["Dominant regime blocker changed with unstable dominance share."]
            return _finalize_recommendation(recommendation, after_summary, before_summary)

    dominance = _safe_float(after_summary.get("blocker_dominance_share", {}).get(earliest_layer, 0.0))
    if dominance < float(cfg.min_dominance_share):
        recommendation["STOP_REASON"] = "observe_more_dominance_too_low"
        recommendation["WHY_NOT_OTHERS"] = ["Dominant blocker share is weak; continue observation before tuning."]
        return _finalize_recommendation(recommendation, after_summary, before_summary)

    recommendation["SAFE_TO_CONTINUE"] = True
    recommendation["NEXT_TARGET_LAYER"] = earliest_layer
    recommendation["NEXT_TARGET_SUBCONDITION"] = target_subcondition
    recommendation["WHY_THIS_TARGET"] = (
        f"Earliest failed layer is '{earliest_layer}' and dominant blocker "
        f"'{target_subcondition}' has stable dominance share {dominance:.3f}."
    )

    sorted_blockers = sorted(blocker_counts.items(), key=lambda kv: int(kv[1]), reverse=True)
    why_not: list[str] = []
    for name, value in sorted_blockers:
        if name == target_subcondition:
            continue
        why_not.append(f"{name} blocker count={int(value)} is lower than {target_subcondition} count={int(target_count)}.")
    if not why_not:
        why_not = ["Only one meaningful blocker detected in the current layer."]
    recommendation["WHY_NOT_OTHERS"] = why_not
    return _finalize_recommendation(recommendation, after_summary, before_summary)


def compare_observation_windows(before_summary: Mapping[str, Any], after_summary: Mapping[str, Any]) -> dict[str, Any]:
    numeric_fields = [
        "samples",
        "evaluations_total",
        "short_signal_count",
        "no_signal_count",
        "short_signal_ratio",
        "no_signal_ratio",
        "regime_filter_pass_count",
        "regime_filter_fail_count",
        "regime_filter_pass_rate",
        "regime_filter_fail_rate",
        "reached_layer1_count",
        "layer1_reach_rate",
        "layer1_fail_count",
        "layer1_fail_rate_given_reach",
        "regime_filter_htf_trend_blocker_count",
        "regime_filter_vwap_stretch_blocker_count",
        "regime_filter_volatility_blocker_count",
        "regime_filter_news_blocker_count",
        "layer1_rsi_high_blocker_count",
        "layer1_volume_spike_blocker_count",
        "layer1_above_bollinger_upper_blocker_count",
        "layer1_above_keltner_upper_blocker_count",
        "layer4_fail_count",
        "layer5_fail_count",
    ]
    delta: dict[str, float] = {}
    for field in numeric_fields:
        delta[field] = _safe_float(after_summary.get(field, 0.0)) - _safe_float(before_summary.get(field, 0.0))

    delta["top_failed_layer_before"] = str(before_summary.get("top_failed_layer", ""))
    delta["top_failed_layer_after"] = str(after_summary.get("top_failed_layer", ""))
    delta["top_regime_filter_blocker_before"] = str(before_summary.get("top_regime_filter_blocker", ""))
    delta["top_regime_filter_blocker_after"] = str(after_summary.get("top_regime_filter_blocker", ""))
    delta["top_blocker_by_raw_coverage_before"] = str(before_summary.get("top_blocker_by_raw_coverage", ""))
    delta["top_blocker_by_raw_coverage_after"] = str(after_summary.get("top_blocker_by_raw_coverage", ""))
    delta["top_blocker_by_label_before"] = str(before_summary.get("top_blocker_by_label", ""))
    delta["top_blocker_by_label_after"] = str(after_summary.get("top_blocker_by_label", ""))
    delta["dominance_mode_before"] = str(before_summary.get("dominance_mode", ""))
    delta["dominance_mode_after"] = str(after_summary.get("dominance_mode", ""))
    delta["co_dominant_blockers_before"] = list(before_summary.get("co_dominant_blockers", [])) if isinstance(before_summary.get("co_dominant_blockers", []), list) else []
    delta["co_dominant_blockers_after"] = list(after_summary.get("co_dominant_blockers", [])) if isinstance(after_summary.get("co_dominant_blockers", []), list) else []
    delta["top_regime_filter_blocker_combination_before"] = str(before_summary.get("top_regime_filter_blocker_combination", ""))
    delta["top_regime_filter_blocker_combination_after"] = str(after_summary.get("top_regime_filter_blocker_combination", ""))
    delta["top_regime_filter_blocker_combination_share_before"] = _safe_float(before_summary.get("top_regime_filter_blocker_combination_share", 0.0))
    delta["top_regime_filter_blocker_combination_share_after"] = _safe_float(after_summary.get("top_regime_filter_blocker_combination_share", 0.0))
    delta["top_layer1_blocker_before"] = str(before_summary.get("top_layer1_blocker", ""))
    delta["top_layer1_blocker_after"] = str(after_summary.get("top_layer1_blocker", ""))
    delta["top_layer4_blocker_before"] = str(before_summary.get("top_layer4_blocker", ""))
    delta["top_layer4_blocker_after"] = str(after_summary.get("top_layer4_blocker", ""))
    return delta


def build_observation_report(
    after_file: str | Path,
    before_file: str | Path | None = None,
    guardrails: CalibrationGuardrails | None = None,
) -> dict[str, Any]:
    after_path = Path(after_file)
    after_records = parse_observation_extract(after_path)
    after_summary = aggregate_observation(after_records)

    report: dict[str, Any] = {
        "after": {
            "file": str(after_path),
            "summary": after_summary,
        }
    }

    before_summary: dict[str, Any] | None = None
    if before_file is not None and str(before_file).strip():
        before_path = Path(before_file)
        before_records = parse_observation_extract(before_path)
        before_summary = aggregate_observation(before_records)
        report["comparison"] = {
            "before": {
                "file": str(before_path),
                "summary": before_summary,
            },
            "after": {
                "file": str(after_path),
                "summary": after_summary,
            },
            "delta": compare_observation_windows(before_summary, after_summary),
        }

    report["calibration_recommendation"] = recommend_calibration_step(after_summary, before_summary, guardrails)
    return report


def report_to_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False)
