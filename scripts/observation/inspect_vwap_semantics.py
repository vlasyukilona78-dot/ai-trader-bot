from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from review_regime_diagnostics import (
    REGIME_BLOCKER_COMPACT_FIELDS,
    REGIME_BLOCKER_KEYS,
    _active_regime_blockers_from_compact,
    _extract_diagnostics,
    _rate,
    _safe_int,
    _top_key_count,
    parse_mapping_literal,
    stats,
)

BUCKET_LABELS: dict[str, str] = {
    "htf_trend_ok": "htf",
    "stretched_from_vwap": "vwap",
    "volatility_regime_ok": "volatility",
    "news_veto": "news",
}

BUCKET_ORDER: tuple[str, ...] = (
    "htf_trend_ok",
    "stretched_from_vwap",
    "volatility_regime_ok",
    "news_veto",
)

CONTEXT_LIMITATIONS: tuple[str, ...] = (
    "Per-sample failed_reason and missing_conditions are only available in enriched regime observation payloads.",
    "Market-structure context beyond HTF direction and source quality is not logged in regime observation payloads.",
)


def _bucket_label(active_blockers: list[str]) -> str:
    clean = [blocker for blocker in BUCKET_ORDER if blocker in set(active_blockers)]
    if not clean:
        return "none"
    parts = [BUCKET_LABELS.get(blocker, blocker) for blocker in clean]
    if len(parts) == 1:
        return f"{parts[0]}_only"
    return " + ".join(parts)


def _sorted_counts(counter: Mapping[str, Any]) -> dict[str, int]:
    pairs = ((str(key), _safe_int(value)) for key, value in counter.items())
    return dict(sorted(pairs, key=lambda item: (-item[1], item[0])))


def _empty_bucket_state() -> dict[str, Any]:
    return {
        "count": 0,
        "reason_fields_available_count": 0,
        "subcondition_state_available_count": 0,
        "label_counts": Counter(),
        "raw_counts": Counter({key: 0 for key in REGIME_BLOCKER_KEYS}),
        "failed_reason_counts": Counter(),
        "missing_conditions_counts": Counter(),
        "missing_condition_item_counts": Counter(),
        "subcondition_false_pattern_counts": Counter(),
        "semantic_path_counts": Counter(),
        "htf_margin_values": [],
        "vwap_margin_values": [],
        "volatility_margin_values": [],
        "vwap_metric_values": [],
        "vwap_threshold_values": [],
        "volatility_metric_values": [],
        "volatility_threshold_values": [],
        "htf_context_counts": Counter(),
        "vwap_position_context_counts": Counter(),
        "vwap_quality_counts": Counter(),
        "news_quality_counts": Counter(),
        "degraded_mode_counts": Counter(),
        "soft_pass_candidate_counts": Counter(),
    }


def _is_vwap_bucket(bucket_name: str) -> bool:
    return "vwap" in str(bucket_name)


def _single_quality_label(node: Any) -> str:
    if not isinstance(node, Mapping):
        return ""
    label, count = _top_key_count(node)
    return str(label) if int(count) > 0 else ""


def _normalize_source_flags(source: Any) -> dict[str, str]:
    if not isinstance(source, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, value in source.items():
        out[str(key)] = str(value) if value is not None else ""
    return out


def _extract_quality_labels(audit: Mapping[str, Any], regime: Mapping[str, Any]) -> tuple[str, str]:
    source_flags = _normalize_source_flags(regime.get("source_flags", {}))
    vwap_quality = source_flags.get("vwap_quality", "").strip().lower()
    news_quality = source_flags.get("news_quality", "").strip().lower()
    if vwap_quality or news_quality:
        return vwap_quality, news_quality

    source_quality_counts = audit.get("source_quality_counts", {})
    if not isinstance(source_quality_counts, Mapping):
        return "", ""
    regime_filter = source_quality_counts.get("regime_filter", {})
    if not isinstance(regime_filter, Mapping):
        return "", ""
    return (
        _single_quality_label(regime_filter.get("vwap_quality", {})),
        _single_quality_label(regime_filter.get("news_quality", {})),
    )


def _missing_condition_items(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off", ""):
            return False
    return None


def _normalize_subconditions_state(source: Any, compact: Mapping[str, Any]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if isinstance(source, Mapping):
        for key in REGIME_BLOCKER_KEYS:
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
    if out:
        return out

    for key in REGIME_BLOCKER_KEYS:
        if key in compact:
            out[key] = _safe_int(compact.get(key, 0)) > 0
            continue
        field = REGIME_BLOCKER_COMPACT_FIELDS.get(key, "")
        if field and field in compact:
            out[key] = _safe_int(compact.get(field, 0)) <= 0
    return out


def _has_material_non_live_sources(
    source_flags: Mapping[str, Any],
    subconditions_state: Mapping[str, bool],
    missing_conditions: str,
) -> bool:
    normalized_flags = _normalize_source_flags(source_flags)
    missing_items = set(_missing_condition_items(missing_conditions))

    vwap_quality = normalized_flags.get("vwap_quality", "").strip().lower()
    if vwap_quality and vwap_quality != "live":
        return True

    news_quality = normalized_flags.get("news_quality", "").strip().lower()
    news_failed = "news_veto" in missing_items
    if not news_failed and isinstance(subconditions_state, Mapping) and "news_veto" in subconditions_state:
        news_failed = not bool(subconditions_state.get("news_veto"))
    if news_failed and news_quality and news_quality != "live":
        return True

    return False


def _false_subcondition_pattern(subconditions_state: Mapping[str, bool]) -> str:
    if not isinstance(subconditions_state, Mapping) or not subconditions_state:
        return "unavailable"
    failed = [key for key in BUCKET_ORDER if key in subconditions_state and not bool(subconditions_state.get(key))]
    if not failed:
        return "none"
    return " + ".join(failed)


def _semantic_path_label(
    failed_reason: str,
    missing_conditions: str,
    degraded_mode: bool,
    source_flags: Mapping[str, Any],
    subconditions_state: Mapping[str, bool],
) -> str:
    missing_items = set(_missing_condition_items(missing_conditions))
    if str(failed_reason).strip().lower() == "insufficient_history" or "history" in missing_items:
        return "missing_data"

    has_non_live_sources = _has_material_non_live_sources(
        source_flags=source_flags,
        subconditions_state=subconditions_state,
        missing_conditions=missing_conditions,
    )
    reason_available = bool(str(failed_reason).strip()) or bool(missing_items)

    if reason_available and (degraded_mode or has_non_live_sources):
        return "explicit_rule_fail_with_degraded_context"
    if reason_available:
        return "explicit_rule_fail"
    if degraded_mode or has_non_live_sources:
        return "degraded_context_only"
    return "reason_unavailable"


def _vwap_position_context(vwap_metric: float | None, vwap_threshold: float | None) -> str:
    if vwap_metric is None:
        return "metric_unavailable"
    if vwap_metric < 0.0:
        return "below_vwap"
    if vwap_threshold is None:
        return "threshold_unavailable"
    if float(vwap_metric) < float(vwap_threshold):
        return "above_vwap_but_under_threshold"
    return "at_or_above_threshold"


def _derive_degraded_mode(
    regime: Mapping[str, Any],
    subconditions_state: Mapping[str, bool],
    missing_conditions: str,
) -> bool:
    explicit = _optional_bool(regime.get("degraded_mode"))
    if explicit is not None:
        return explicit
    return _has_material_non_live_sources(
        source_flags=regime.get("source_flags", {}),
        subconditions_state=subconditions_state,
        missing_conditions=missing_conditions,
    )


def _derive_soft_pass_candidate(
    regime: Mapping[str, Any],
    subconditions_state: Mapping[str, bool],
    failed_reason: str,
    missing_conditions: str,
) -> bool:
    explicit = _optional_bool(regime.get("soft_pass_candidate"))
    if explicit is not None:
        return explicit
    if not isinstance(subconditions_state, Mapping) or not subconditions_state:
        return False
    missing_items = set(_missing_condition_items(missing_conditions))
    passed = str(failed_reason).strip().lower() == "none" and not missing_items
    if passed:
        return False
    if not bool(subconditions_state.get("htf_trend_ok")):
        return False
    support_count = sum(
        1
        for key in ("stretched_from_vwap", "volatility_regime_ok", "news_veto")
        if bool(subconditions_state.get(key))
    )
    return support_count >= 2


def _bucket_report(
    bucket_name: str,
    bucket_state: Mapping[str, Any],
    failed_sample_count: int,
    vwap_related_failed_sample_count: int,
) -> dict[str, Any]:
    count = _safe_int(bucket_state.get("count", 0))
    reason_fields_available_count = _safe_int(bucket_state.get("reason_fields_available_count", 0))
    subcondition_state_available_count = _safe_int(bucket_state.get("subcondition_state_available_count", 0))
    label_counts = _sorted_counts(bucket_state.get("label_counts", {}))
    raw_counts = _sorted_counts(bucket_state.get("raw_counts", {}))
    failed_reason_counts = _sorted_counts(bucket_state.get("failed_reason_counts", {}))
    missing_conditions_counts = _sorted_counts(bucket_state.get("missing_conditions_counts", {}))
    missing_condition_item_counts = _sorted_counts(bucket_state.get("missing_condition_item_counts", {}))
    subcondition_false_pattern_counts = _sorted_counts(bucket_state.get("subcondition_false_pattern_counts", {}))
    semantic_path_counts = _sorted_counts(bucket_state.get("semantic_path_counts", {}))
    top_label_blocker, top_label_count = _top_key_count(label_counts)
    top_raw_blocker, top_raw_count = _top_key_count(raw_counts)
    top_failed_reason, top_failed_reason_count = _top_key_count(failed_reason_counts)
    top_missing_conditions, top_missing_conditions_count = _top_key_count(missing_conditions_counts)
    top_false_pattern, top_false_pattern_count = _top_key_count(subcondition_false_pattern_counts)
    top_semantic_path, top_semantic_path_count = _top_key_count(semantic_path_counts)

    return {
        "bucket": bucket_name,
        "count": count,
        "reason_fields_available_count": reason_fields_available_count,
        "reason_fields_available_share": _rate(reason_fields_available_count, count),
        "subcondition_state_available_count": subcondition_state_available_count,
        "subcondition_state_available_share": _rate(subcondition_state_available_count, count),
        "share_of_failed_samples": _rate(count, failed_sample_count),
        "share_of_vwap_related_failed_samples": _rate(count, vwap_related_failed_sample_count),
        "top_label_blocker": top_label_blocker,
        "top_label_blocker_count": int(top_label_count),
        "top_label_blocker_share": _rate(top_label_count, count),
        "top_raw_blocker": top_raw_blocker,
        "top_raw_blocker_count": int(top_raw_count),
        "top_raw_blocker_share": _rate(top_raw_count, count),
        "label_blocker_counts": label_counts,
        "raw_blocker_counts": raw_counts,
        "failed_reason_distribution": failed_reason_counts,
        "top_failed_reason": top_failed_reason,
        "top_failed_reason_count": int(top_failed_reason_count),
        "top_failed_reason_share": _rate(top_failed_reason_count, count),
        "missing_conditions_pattern_distribution": missing_conditions_counts,
        "top_missing_conditions_pattern": top_missing_conditions,
        "top_missing_conditions_pattern_count": int(top_missing_conditions_count),
        "top_missing_conditions_pattern_share": _rate(top_missing_conditions_count, count),
        "missing_condition_item_counts": missing_condition_item_counts,
        "subcondition_false_pattern_distribution": subcondition_false_pattern_counts,
        "top_subcondition_false_pattern": top_false_pattern,
        "top_subcondition_false_pattern_count": int(top_false_pattern_count),
        "top_subcondition_false_pattern_share": _rate(top_false_pattern_count, count),
        "semantic_path_distribution": semantic_path_counts,
        "top_semantic_path": top_semantic_path,
        "top_semantic_path_count": int(top_semantic_path_count),
        "top_semantic_path_share": _rate(top_semantic_path_count, count),
        "htf_direction_context_distribution": _sorted_counts(bucket_state.get("htf_context_counts", {})),
        "vwap_position_context_distribution": _sorted_counts(bucket_state.get("vwap_position_context_counts", {})),
        "vwap_quality_distribution": _sorted_counts(bucket_state.get("vwap_quality_counts", {})),
        "news_quality_distribution": _sorted_counts(bucket_state.get("news_quality_counts", {})),
        "degraded_mode_distribution": _sorted_counts(bucket_state.get("degraded_mode_counts", {})),
        "soft_pass_candidate_distribution": _sorted_counts(bucket_state.get("soft_pass_candidate_counts", {})),
        "margins": {
            "htf_metric_minus_threshold": stats(list(bucket_state.get("htf_margin_values", []))),
            "vwap_metric_minus_threshold": stats(list(bucket_state.get("vwap_margin_values", []))),
            "volatility_metric_minus_threshold": stats(list(bucket_state.get("volatility_margin_values", []))),
        },
        "metrics": {
            "vwap_distance_metric_used": stats(list(bucket_state.get("vwap_metric_values", []))),
            "vwap_stretch_threshold_used": stats(list(bucket_state.get("vwap_threshold_values", []))),
            "atr_norm": stats(list(bucket_state.get("volatility_metric_values", []))),
            "volatility_threshold_used": stats(list(bucket_state.get("volatility_threshold_values", []))),
        },
    }


def analyze_file(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    bucket_states: dict[str, dict[str, Any]] = {}
    all_bucket_counts: Counter[str] = Counter()
    failed_sample_count = 0
    reason_fields_available_count = 0
    subcondition_state_available_count = 0
    parsed_json_line_count = 0
    records_with_strategy_audit_compact = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue

        parsed_json_line_count += 1
        msg = str(payload.get("msg", ""))

        compact = parse_mapping_literal(msg, "strategy_audit_compact")
        if not compact:
            continue

        records_with_strategy_audit_compact += 1

        active_blockers = _active_regime_blockers_from_compact(compact)
        if _safe_int(compact.get("regime_filter_fail_count", 0)) <= 0 and not active_blockers:
            continue

        failed_sample_count += 1
        bucket_name = _bucket_label(active_blockers)
        all_bucket_counts[bucket_name] += 1
        if not _is_vwap_bucket(bucket_name):
            continue

        bucket_state = bucket_states.setdefault(bucket_name, _empty_bucket_state())
        bucket_state["count"] = _safe_int(bucket_state.get("count", 0)) + 1

        top_label = str(compact.get("top_regime_filter_blocker", "")).strip()
        if top_label:
            bucket_state["label_counts"][top_label] += 1

        for blocker in REGIME_BLOCKER_KEYS:
            field = REGIME_BLOCKER_COMPACT_FIELDS.get(blocker, "")
            if field and _safe_int(compact.get(field, 0)) > 0:
                bucket_state["raw_counts"][blocker] += 1

        regime = parse_mapping_literal(msg, "strategy_audit_regime_filter")
        audit = parse_mapping_literal(msg, "strategy_audit")
        layer_trace = parse_mapping_literal(msg, "layer_trace")
        extra_maps: list[Mapping[str, Any]] = [
            compact,
            regime,
            audit,
            layer_trace,
            parse_mapping_literal(msg, "regime_filter_details"),
            parse_mapping_literal(msg, "diagnostics"),
            parse_mapping_literal(msg, "details"),
        ]
        extra_maps = [mp for mp in extra_maps if isinstance(mp, Mapping) and mp]
        diag = _extract_diagnostics(msg, extra_maps)
        failed_reason = str(regime.get("failed_reason") or "").strip()
        missing_conditions = str(regime.get("missing_conditions") or "").strip()
        reason_available = bool(failed_reason) or bool(missing_conditions)
        if reason_available:
            reason_fields_available_count += 1
            bucket_state["reason_fields_available_count"] = _safe_int(bucket_state.get("reason_fields_available_count", 0)) + 1
        bucket_state["failed_reason_counts"][failed_reason or "unavailable"] += 1
        bucket_state["missing_conditions_counts"][missing_conditions or "unavailable"] += 1
        for item in _missing_condition_items(missing_conditions):
            bucket_state["missing_condition_item_counts"][item] += 1

        subconditions_state = _normalize_subconditions_state(
            regime.get("regime_filter_subconditions_state", regime),
            compact,
        )
        if subconditions_state:
            subcondition_state_available_count += 1
            bucket_state["subcondition_state_available_count"] = _safe_int(
                bucket_state.get("subcondition_state_available_count", 0)
            ) + 1
        bucket_state["subcondition_false_pattern_counts"][_false_subcondition_pattern(subconditions_state)] += 1

        htf_context = str(diag.get("htf_trend_direction_context", "") or "unknown")
        bucket_state["htf_context_counts"][htf_context] += 1

        htf_metric = diag.get("htf_trend_metric_used")
        htf_threshold = diag.get("htf_trend_threshold_used")
        if htf_metric is not None and htf_threshold is not None:
            bucket_state["htf_margin_values"].append(float(htf_metric) - float(htf_threshold))

        vwap_metric = diag.get("vwap_distance_metric_used")
        vwap_threshold = diag.get("vwap_stretch_threshold_used")
        if vwap_metric is not None:
            bucket_state["vwap_metric_values"].append(float(vwap_metric))
        if vwap_threshold is not None:
            bucket_state["vwap_threshold_values"].append(float(vwap_threshold))
        if vwap_metric is not None and vwap_threshold is not None:
            bucket_state["vwap_margin_values"].append(float(vwap_metric) - float(vwap_threshold))
        bucket_state["vwap_position_context_counts"][_vwap_position_context(vwap_metric, vwap_threshold)] += 1

        volatility_metric = diag.get("atr_norm")
        volatility_threshold = diag.get("volatility_threshold_used")
        if volatility_metric is not None:
            bucket_state["volatility_metric_values"].append(float(volatility_metric))
        if volatility_threshold is not None:
            bucket_state["volatility_threshold_values"].append(float(volatility_threshold))
        if volatility_metric is not None and volatility_threshold is not None:
            bucket_state["volatility_margin_values"].append(float(volatility_metric) - float(volatility_threshold))

        vwap_quality, news_quality = _extract_quality_labels(audit, regime)
        bucket_state["vwap_quality_counts"][vwap_quality or "unknown"] += 1
        bucket_state["news_quality_counts"][news_quality or "unknown"] += 1

        degraded_mode = _derive_degraded_mode(
            regime=regime,
            subconditions_state=subconditions_state,
            missing_conditions=missing_conditions,
        )
        soft_pass_candidate = _derive_soft_pass_candidate(
            regime=regime,
            subconditions_state=subconditions_state,
            failed_reason=failed_reason,
            missing_conditions=missing_conditions,
        )
        bucket_state["degraded_mode_counts"]["degraded" if degraded_mode else "not_degraded"] += 1
        bucket_state["soft_pass_candidate_counts"]["soft_candidate" if soft_pass_candidate else "hard_fail"] += 1
        bucket_state["semantic_path_counts"][
            _semantic_path_label(
                failed_reason=failed_reason,
                missing_conditions=missing_conditions,
                degraded_mode=degraded_mode,
                source_flags=_normalize_source_flags(regime.get("source_flags", {})),
                subconditions_state=subconditions_state,
            )
        ] += 1

    bucket_counts = {name: _safe_int(state.get("count", 0)) for name, state in bucket_states.items()}
    bucket_counts = dict(sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0])))
    vwap_related_failed_sample_count = sum(bucket_counts.values())
    bucket_share_by_failed_samples = {
        name: _rate(count, failed_sample_count)
        for name, count in bucket_counts.items()
    }
    bucket_share_by_vwap_related_failed_samples = {
        name: _rate(count, vwap_related_failed_sample_count)
        for name, count in bucket_counts.items()
    }
    top_focus_bucket, top_focus_bucket_count = _top_key_count(bucket_counts)

    volatility_without_vwap_count = sum(
        count for name, count in all_bucket_counts.items() if "volatility" in name and "vwap" not in name
    )
    vwap_without_volatility_count = sum(
        count for name, count in all_bucket_counts.items() if "vwap" in name and "volatility" not in name
    )

    return {
        "file": str(path),
        "line_count": len(lines),
        "parsed_json_line_count": parsed_json_line_count,
        "records_with_strategy_audit_compact": records_with_strategy_audit_compact,
        "failed_sample_count": int(failed_sample_count),
        "vwap_related_failed_sample_count": int(vwap_related_failed_sample_count),
        "vwap_related_share_of_failed_samples": _rate(vwap_related_failed_sample_count, failed_sample_count),
        "reason_fields_available_count": int(reason_fields_available_count),
        "reason_fields_available_share_of_vwap_related_failed_samples": _rate(
            reason_fields_available_count,
            vwap_related_failed_sample_count,
        ),
        "subcondition_state_available_count": int(subcondition_state_available_count),
        "subcondition_state_available_share_of_vwap_related_failed_samples": _rate(
            subcondition_state_available_count,
            vwap_related_failed_sample_count,
        ),
        "focus_bucket_counts": bucket_counts,
        "focus_bucket_share_by_failed_samples": bucket_share_by_failed_samples,
        "focus_bucket_share_by_vwap_related_failed_samples": bucket_share_by_vwap_related_failed_samples,
        "top_focus_bucket": top_focus_bucket,
        "top_focus_bucket_count": int(top_focus_bucket_count),
        "top_focus_bucket_share_of_failed_samples": _rate(top_focus_bucket_count, failed_sample_count),
        "top_focus_bucket_share_of_vwap_related_failed_samples": _rate(
            top_focus_bucket_count,
            vwap_related_failed_sample_count,
        ),
        "all_bucket_counts": dict(sorted(all_bucket_counts.items(), key=lambda item: (-item[1], item[0]))),
        "independent_blocker_evidence": {
            "vwap_only_count": int(all_bucket_counts.get("vwap_only", 0)),
            "vwap_without_volatility_count": int(vwap_without_volatility_count),
            "volatility_only_count": int(all_bucket_counts.get("volatility_only", 0)),
            "volatility_without_vwap_count": int(volatility_without_vwap_count),
        },
        "context_limitations": list(CONTEXT_LIMITATIONS),
        "buckets": {
            bucket_name: _bucket_report(
                bucket_name,
                bucket_states[bucket_name],
                failed_sample_count,
                vwap_related_failed_sample_count,
            )
            for bucket_name in bucket_counts
        },
    }


def compare_reports(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_counts = {
        str(key): _safe_int(value)
        for key, value in (before.get("focus_bucket_counts", {}) if isinstance(before.get("focus_bucket_counts", {}), Mapping) else {}).items()
    }
    after_counts = {
        str(key): _safe_int(value)
        for key, value in (after.get("focus_bucket_counts", {}) if isinstance(after.get("focus_bucket_counts", {}), Mapping) else {}).items()
    }
    before_failed = _safe_int(before.get("failed_sample_count", 0))
    after_failed = _safe_int(after.get("failed_sample_count", 0))
    before_vwap_related = _safe_int(before.get("vwap_related_failed_sample_count", 0))
    after_vwap_related = _safe_int(after.get("vwap_related_failed_sample_count", 0))

    all_buckets = sorted(set(before_counts) | set(after_counts))

    return {
        "top_focus_bucket_before": str(before.get("top_focus_bucket", "")),
        "top_focus_bucket_after": str(after.get("top_focus_bucket", "")),
        "top_focus_bucket_share_of_failed_samples_before": float(
            before.get("top_focus_bucket_share_of_failed_samples", 0.0) or 0.0
        ),
        "top_focus_bucket_share_of_failed_samples_after": float(
            after.get("top_focus_bucket_share_of_failed_samples", 0.0) or 0.0
        ),
        "focus_bucket_count_delta": {
            bucket: int(after_counts.get(bucket, 0) - before_counts.get(bucket, 0))
            for bucket in all_buckets
        },
        "focus_bucket_share_of_failed_samples_delta": {
            bucket: _rate(after_counts.get(bucket, 0), after_failed) - _rate(before_counts.get(bucket, 0), before_failed)
            for bucket in all_buckets
        },
        "focus_bucket_share_of_vwap_related_failed_samples_delta": {
            bucket: _rate(after_counts.get(bucket, 0), after_vwap_related)
            - _rate(before_counts.get(bucket, 0), before_vwap_related)
            for bucket in all_buckets
        },
        "vwap_only_count_before": _safe_int(before.get("independent_blocker_evidence", {}).get("vwap_only_count", 0)),
        "vwap_only_count_after": _safe_int(after.get("independent_blocker_evidence", {}).get("vwap_only_count", 0)),
        "volatility_without_vwap_count_before": _safe_int(
            before.get("independent_blocker_evidence", {}).get("volatility_without_vwap_count", 0)
        ),
        "volatility_without_vwap_count_after": _safe_int(
            after.get("independent_blocker_evidence", {}).get("volatility_without_vwap_count", 0)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect VWAP-first regime blocker semantics from observation logs.")
    parser.add_argument("--after", required=True, help="Path to AFTER combined or audit-extract observation log")
    parser.add_argument("--before", default="", help="Optional path to BEFORE combined or audit-extract observation log")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    args = parser.parse_args()

    after_path = Path(args.after)
    if not after_path.exists():
        raise FileNotFoundError(f"after log not found: {after_path}")

    report: dict[str, Any] = {
        "after": analyze_file(after_path),
    }

    if args.before:
        before_path = Path(args.before)
        if not before_path.exists():
            raise FileNotFoundError(f"before log not found: {before_path}")
        before_report = analyze_file(before_path)
        report["before"] = before_report
        report["comparison"] = compare_reports(before_report, report["after"])

    text = json.dumps(report, ensure_ascii=False)
    print(text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
