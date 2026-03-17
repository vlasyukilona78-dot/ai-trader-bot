from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

NUMERIC_FIELDS: tuple[str, ...] = (
    "htf_trend_metric_used",
    "htf_trend_threshold_used",
    "vwap_distance_metric_used",
    "vwap_stretch_threshold_used",
    "atr_norm",
    "volatility_threshold_used",
)

TEXT_FIELDS: tuple[str, ...] = (
    "htf_trend_direction_context",
    "top_regime_filter_blocker",
)

REGIME_BLOCKER_KEYS: tuple[str, ...] = (
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

CO_DOMINANT_RELATIVE_MIN = 0.85


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def extract_literal(msg: str, key: str) -> str | None:
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


def parse_mapping_literal(msg: str, key: str) -> dict[str, Any]:
    raw = extract_literal(msg, key)
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _deep_find_value(node: Any, key: str) -> Any | None:
    if isinstance(node, Mapping):
        if key in node:
            return node.get(key)
        for value in node.values():
            found = _deep_find_value(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _deep_find_value(item, key)
            if found is not None:
                return found
    return None


def extract_number_from_msg(msg: str, key: str) -> float | None:
    patterns = (
        rf"{re.escape(key)}\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        rf"['\"]{re.escape(key)}['\"]\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    )
    for pattern in patterns:
        m = re.search(pattern, msg)
        if not m:
            continue
        value = _safe_float(m.group(1))
        if value is not None:
            return value
    return None


def extract_text_from_msg(msg: str, key: str) -> str | None:
    patterns = (
        rf"{re.escape(key)}\s*=\s*'([^']*)'",
        rf"{re.escape(key)}\s*=\s*\"([^\"]*)\"",
        rf"{re.escape(key)}\s*=\s*([^\s,}}]+)",
        rf"['\"]{re.escape(key)}['\"]\s*:\s*'([^']*)'",
        rf"['\"]{re.escape(key)}['\"]\s*:\s*\"([^\"]*)\"",
    )
    for pattern in patterns:
        m = re.search(pattern, msg)
        if m:
            return str(m.group(1)).strip()
    return None


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(arr) - 1)
    frac = pos - lo
    return arr[lo] + (arr[hi] - arr[lo]) * frac


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    vals = [float(v) for v in values]
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "p50": quantile(vals, 0.5),
        "p90": quantile(vals, 0.9),
    }


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
    for blocker in REGIME_BLOCKER_KEYS:
        field = REGIME_BLOCKER_COMPACT_FIELDS.get(blocker, "")
        if field and _safe_int(compact.get(field, 0)) > 0:
            active.append(blocker)
    return active


def _action_verdict(
    *,
    dominance_mode: str,
    top_blocker_raw: str,
    top_blocker_label: str,
    top_blocker_raw_share: float,
    top_blocker_label_share: float,
    co_dominant_blockers: list[str],
) -> tuple[str, str]:
    if dominance_mode in {"raw_vs_label_disagreement", "co_dominant_raw"} or len(co_dominant_blockers) >= 2:
        return (
            "co_dominant_overlap",
            "Raw blocker coverage and snapshot blocker labels are not isolated to one stable blocker.",
        )
    if (
        top_blocker_raw
        and top_blocker_label
        and top_blocker_raw == top_blocker_label
        and top_blocker_raw_share >= 0.55
        and top_blocker_label_share >= 0.55
    ):
        return (
            "single_blocker_ready",
            f"Both raw coverage and snapshot labels agree on {top_blocker_raw}.",
        )
    return (
        "pause_calibration",
        "Blocker dominance is not yet strong enough to justify another isolated tweak.",
    )


def _rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _extract_diagnostics(msg: str, extra_maps: list[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key in NUMERIC_FIELDS:
        value = extract_number_from_msg(msg, key)
        if value is not None:
            out[key] = value
            continue
        for mp in extra_maps:
            found = _deep_find_value(mp, key)
            value2 = _safe_float(found)
            if value2 is not None:
                out[key] = value2
                break

    for key in TEXT_FIELDS:
        value = extract_text_from_msg(msg, key)
        if value is not None:
            out[key] = value
            continue
        for mp in extra_maps:
            found = _deep_find_value(mp, key)
            if found is not None:
                out[key] = str(found)
                break

    return out


def analyze_file(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    presence = Counter()
    numeric_samples: dict[str, list[float]] = {k: [] for k in NUMERIC_FIELDS}
    text_samples: dict[str, list[str]] = {k: [] for k in TEXT_FIELDS}

    htf_margin_values: list[float] = []
    vwap_margin_values: list[float] = []
    volatility_margin_values: list[float] = []

    regime_blocker_counts = Counter({k: 0 for k in REGIME_BLOCKER_KEYS})
    top_blocker_label_counts = Counter()
    blocker_combination_counts = Counter()
    failed_sample_count = 0

    parsed_line_count = 0
    audit_record_count = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue

        parsed_line_count += 1
        msg = str(payload.get("msg", ""))

        compact = parse_mapping_literal(msg, "strategy_audit_compact")
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

        if compact:
            audit_record_count += 1
            top_name = str(compact.get("top_regime_filter_blocker", "")).strip()
            if top_name:
                top_blocker_label_counts[top_name] += 1

            regime_blocker_counts["htf_trend_ok"] += _safe_int(compact.get("regime_filter_htf_trend_blocker_count", 0))
            regime_blocker_counts["stretched_from_vwap"] += _safe_int(compact.get("regime_filter_vwap_stretch_blocker_count", 0))
            regime_blocker_counts["volatility_regime_ok"] += _safe_int(compact.get("regime_filter_volatility_blocker_count", 0))
            regime_blocker_counts["news_veto"] += _safe_int(compact.get("regime_filter_news_blocker_count", 0))
            if _safe_int(compact.get("regime_filter_fail_count", 0)) > 0 or any(
                _safe_int(compact.get(field, 0)) > 0 for field in REGIME_BLOCKER_COMPACT_FIELDS.values()
            ):
                failed_sample_count += 1
                blocker_combination_counts[_combination_label(_active_regime_blockers_from_compact(compact))] += 1

        diag = _extract_diagnostics(msg, extra_maps)
        for key in NUMERIC_FIELDS:
            if key in diag:
                presence[key] += 1
                numeric_samples[key].append(float(diag[key]))
        for key in TEXT_FIELDS:
            if key in diag:
                presence[key] += 1
                text_samples[key].append(str(diag[key]))

        htf_metric = diag.get("htf_trend_metric_used")
        htf_threshold = diag.get("htf_trend_threshold_used")
        if htf_metric is not None and htf_threshold is not None:
            htf_margin_values.append(float(htf_metric) - float(htf_threshold))

        vwap_metric = diag.get("vwap_distance_metric_used")
        vwap_threshold = diag.get("vwap_stretch_threshold_used")
        if vwap_metric is not None and vwap_threshold is not None:
            vwap_margin_values.append(float(vwap_metric) - float(vwap_threshold))

        vol_metric = diag.get("atr_norm")
        vol_threshold = diag.get("volatility_threshold_used")
        if vol_metric is not None and vol_threshold is not None:
            volatility_margin_values.append(float(vol_metric) - float(vol_threshold))

    samples = max(audit_record_count, parsed_line_count)
    top_blocker_label = ""
    top_blocker_label_count = 0
    if top_blocker_label_counts:
        top_blocker_label, top_blocker_label_count = top_blocker_label_counts.most_common(1)[0]
    top_blocker_raw, top_blocker_raw_count = _top_key_count(regime_blocker_counts)
    top_blocker_raw_share = _rate(top_blocker_raw_count, audit_record_count)
    top_blocker_label_share = _rate(top_blocker_label_count, audit_record_count)
    top_combination, top_combination_count = _top_key_count(blocker_combination_counts)
    top_combination_share = _rate(top_combination_count, failed_sample_count)
    co_dominant_blockers = _dominant_keys(regime_blocker_counts, relative_min=CO_DOMINANT_RELATIVE_MIN)
    dominance_mode = "single_dominant_raw"
    dominance_explanation = (
        f"raw coverage favors {top_blocker_raw or 'none'} ({top_blocker_raw_share:.3f}); "
        f"snapshot labels favor {top_blocker_label or 'none'} ({top_blocker_label_share:.3f})."
    )
    if top_blocker_raw and top_blocker_label and top_blocker_raw != top_blocker_label:
        dominance_mode = "raw_vs_label_disagreement"
        dominance_explanation = (
            f"raw coverage favors {top_blocker_raw} ({top_blocker_raw_share:.3f}) "
            f"while snapshot labels favor {top_blocker_label} ({top_blocker_label_share:.3f})."
        )
    elif len(co_dominant_blockers) >= 2:
        dominance_mode = "co_dominant_raw"
        dominance_explanation = (
            "raw blocker coverage is co-dominant across " + ", ".join(co_dominant_blockers) + "."
        )
    action_verdict, action_verdict_reason = _action_verdict(
        dominance_mode=dominance_mode,
        top_blocker_raw=top_blocker_raw,
        top_blocker_label=top_blocker_label,
        top_blocker_raw_share=float(top_blocker_raw_share),
        top_blocker_label_share=float(top_blocker_label_share),
        co_dominant_blockers=co_dominant_blockers,
    )

    report = {
        "file": str(path),
        "line_count": len(lines),
        "parsed_json_line_count": parsed_line_count,
        "records_with_strategy_audit_compact": audit_record_count,
        "presence": {
            key: int(presence.get(key, 0))
            for key in (*NUMERIC_FIELDS, *TEXT_FIELDS)
        },
        "diagnostics": {
            "numeric_stats": {
                key: stats(values)
                for key, values in numeric_samples.items()
            },
            "text_distributions": {
                key: dict(Counter(values))
                for key, values in text_samples.items()
                if values
            },
        },
        "margins": {
            "htf_metric_minus_threshold": stats(htf_margin_values),
            "vwap_metric_minus_threshold": stats(vwap_margin_values),
            "volatility_metric_minus_threshold": stats(volatility_margin_values),
        },
        "blocker_mix": {
            "top_regime_filter_blocker_counts": dict(top_blocker_label_counts),
            "top_regime_filter_blocker": top_blocker_label,
            "top_regime_filter_blocker_count": int(top_blocker_label_count),
            "regime_blocker_counts": {k: int(regime_blocker_counts.get(k, 0)) for k in REGIME_BLOCKER_KEYS},
            "regime_blocker_share_by_samples": {
                k: (float(regime_blocker_counts.get(k, 0)) / float(samples) if samples > 0 else 0.0)
                for k in REGIME_BLOCKER_KEYS
            },
            "top_blocker_label_counts": dict(top_blocker_label_counts),
            "top_blocker_by_label": top_blocker_label,
            "top_blocker_by_label_count": int(top_blocker_label_count),
            "top_blocker_by_label_share": float(top_blocker_label_share),
            "top_blocker_by_raw_coverage": top_blocker_raw,
            "top_blocker_by_raw_coverage_count": int(top_blocker_raw_count),
            "top_blocker_by_raw_coverage_share": float(top_blocker_raw_share),
            "co_dominant_blockers": co_dominant_blockers,
            "dominance_mode": dominance_mode,
            "dominance_explanation": dominance_explanation,
            "action_verdict": action_verdict,
            "action_verdict_reason": action_verdict_reason,
            "regime_filter_failed_sample_count": int(failed_sample_count),
            "regime_blocker_combination_counts": _sorted_counts(blocker_combination_counts),
            "regime_blocker_combination_share_by_failed_samples": {
                key: _rate(_safe_int(value), failed_sample_count)
                for key, value in _sorted_counts(blocker_combination_counts).items()
            },
            "top_regime_filter_blocker_combination": top_combination,
            "top_regime_filter_blocker_combination_count": int(top_combination_count),
            "top_regime_filter_blocker_combination_share": float(top_combination_share),
        },
    }
    return report


def compare_reports(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    def _get(d: Mapping[str, Any], *keys: str) -> float:
        node: Any = d
        for key in keys:
            if not isinstance(node, Mapping):
                return 0.0
            node = node.get(key)
        return float(node) if isinstance(node, (int, float)) else 0.0

    return {
        "margins_delta_mean": {
            "htf_metric_minus_threshold": _get(after, "margins", "htf_metric_minus_threshold", "mean")
            - _get(before, "margins", "htf_metric_minus_threshold", "mean"),
            "vwap_metric_minus_threshold": _get(after, "margins", "vwap_metric_minus_threshold", "mean")
            - _get(before, "margins", "vwap_metric_minus_threshold", "mean"),
            "volatility_metric_minus_threshold": _get(after, "margins", "volatility_metric_minus_threshold", "mean")
            - _get(before, "margins", "volatility_metric_minus_threshold", "mean"),
        },
        "top_regime_filter_blocker_before": str(before.get("blocker_mix", {}).get("top_regime_filter_blocker", "")),
        "top_regime_filter_blocker_after": str(after.get("blocker_mix", {}).get("top_regime_filter_blocker", "")),
        "top_blocker_by_raw_coverage_before": str(before.get("blocker_mix", {}).get("top_blocker_by_raw_coverage", "")),
        "top_blocker_by_raw_coverage_after": str(after.get("blocker_mix", {}).get("top_blocker_by_raw_coverage", "")),
        "top_blocker_by_label_before": str(before.get("blocker_mix", {}).get("top_blocker_by_label", "")),
        "top_blocker_by_label_after": str(after.get("blocker_mix", {}).get("top_blocker_by_label", "")),
        "dominance_mode_before": str(before.get("blocker_mix", {}).get("dominance_mode", "")),
        "dominance_mode_after": str(after.get("blocker_mix", {}).get("dominance_mode", "")),
        "action_verdict_before": str(before.get("blocker_mix", {}).get("action_verdict", "")),
        "action_verdict_after": str(after.get("blocker_mix", {}).get("action_verdict", "")),
        "top_regime_filter_blocker_combination_before": str(
            before.get("blocker_mix", {}).get("top_regime_filter_blocker_combination", "")
        ),
        "top_regime_filter_blocker_combination_after": str(
            after.get("blocker_mix", {}).get("top_regime_filter_blocker_combination", "")
        ),
        "parsed_json_line_count_delta": int(after.get("parsed_json_line_count", 0)) - int(before.get("parsed_json_line_count", 0)),
        "records_with_strategy_audit_compact_delta": int(after.get("records_with_strategy_audit_compact", 0))
        - int(before.get("records_with_strategy_audit_compact", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Review semantic regime diagnostics from combined observation logs.")
    parser.add_argument("--after", required=True, help="Path to AFTER combined observation log")
    parser.add_argument("--before", default="", help="Path to BEFORE combined observation log")
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
