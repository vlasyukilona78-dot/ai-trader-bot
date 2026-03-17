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


def _bucket_label(active_blockers: list[str]) -> str:
    clean = [blocker for blocker in BUCKET_ORDER if blocker in set(active_blockers)]
    if not clean:
        return "none"
    parts = [BUCKET_LABELS.get(blocker, blocker) for blocker in clean]
    if len(parts) == 1:
        return f"{parts[0]}_only"
    return " + ".join(parts)


def _empty_bucket_state() -> dict[str, Any]:
    return {
        "count": 0,
        "label_counts": Counter(),
        "raw_counts": Counter({key: 0 for key in REGIME_BLOCKER_KEYS}),
        "htf_margin_values": [],
        "vwap_margin_values": [],
        "volatility_margin_values": [],
    }


def _sorted_counts(counter: Mapping[str, Any]) -> dict[str, int]:
    pairs = ((str(key), _safe_int(value)) for key, value in counter.items())
    return dict(sorted(pairs, key=lambda item: (-item[1], item[0])))


def _bucket_report(bucket_name: str, bucket_state: Mapping[str, Any], failed_sample_count: int) -> dict[str, Any]:
    count = _safe_int(bucket_state.get("count", 0))
    label_counts = _sorted_counts(bucket_state.get("label_counts", {}))
    raw_counts = _sorted_counts(bucket_state.get("raw_counts", {}))
    top_label_blocker, top_label_count = _top_key_count(label_counts)
    top_raw_blocker, top_raw_count = _top_key_count(raw_counts)

    return {
        "bucket": bucket_name,
        "count": count,
        "share_of_failed_samples": _rate(count, failed_sample_count),
        "top_label_blocker": top_label_blocker,
        "top_label_blocker_count": int(top_label_count),
        "top_label_blocker_share": _rate(top_label_count, count),
        "top_raw_blocker": top_raw_blocker,
        "top_raw_blocker_count": int(top_raw_count),
        "top_raw_blocker_share": _rate(top_raw_count, count),
        "label_blocker_counts": label_counts,
        "raw_blocker_counts": raw_counts,
        "margins": {
            "htf_metric_minus_threshold": stats(list(bucket_state.get("htf_margin_values", []))),
            "vwap_metric_minus_threshold": stats(list(bucket_state.get("vwap_margin_values", []))),
            "volatility_metric_minus_threshold": stats(list(bucket_state.get("volatility_margin_values", []))),
        },
    }


def analyze_file(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    bucket_states: dict[str, dict[str, Any]] = {}
    failed_sample_count = 0
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

        bucket_name = _bucket_label(active_blockers)
        bucket_state = bucket_states.setdefault(bucket_name, _empty_bucket_state())
        bucket_state["count"] = _safe_int(bucket_state.get("count", 0)) + 1
        failed_sample_count += 1

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

        htf_metric = diag.get("htf_trend_metric_used")
        htf_threshold = diag.get("htf_trend_threshold_used")
        if htf_metric is not None and htf_threshold is not None:
            bucket_state["htf_margin_values"].append(float(htf_metric) - float(htf_threshold))

        vwap_metric = diag.get("vwap_distance_metric_used")
        vwap_threshold = diag.get("vwap_stretch_threshold_used")
        if vwap_metric is not None and vwap_threshold is not None:
            bucket_state["vwap_margin_values"].append(float(vwap_metric) - float(vwap_threshold))

        volatility_metric = diag.get("atr_norm")
        volatility_threshold = diag.get("volatility_threshold_used")
        if volatility_metric is not None and volatility_threshold is not None:
            bucket_state["volatility_margin_values"].append(float(volatility_metric) - float(volatility_threshold))

    bucket_counts = {name: _safe_int(state.get("count", 0)) for name, state in bucket_states.items()}
    bucket_counts = dict(sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0])))
    bucket_share_by_failed_samples = {
        name: _rate(count, failed_sample_count)
        for name, count in bucket_counts.items()
    }
    top_bucket, top_bucket_count = _top_key_count(bucket_counts)

    return {
        "file": str(path),
        "line_count": len(lines),
        "parsed_json_line_count": parsed_json_line_count,
        "records_with_strategy_audit_compact": records_with_strategy_audit_compact,
        "failed_sample_count": int(failed_sample_count),
        "bucket_counts": bucket_counts,
        "bucket_share_by_failed_samples": bucket_share_by_failed_samples,
        "top_bucket": top_bucket,
        "top_bucket_count": int(top_bucket_count),
        "top_bucket_share": _rate(top_bucket_count, failed_sample_count),
        "buckets": {
            bucket_name: _bucket_report(bucket_name, bucket_states[bucket_name], failed_sample_count)
            for bucket_name in bucket_counts
        },
    }


def compare_reports(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_counts = {
        str(key): _safe_int(value)
        for key, value in (before.get("bucket_counts", {}) if isinstance(before.get("bucket_counts", {}), Mapping) else {}).items()
    }
    after_counts = {
        str(key): _safe_int(value)
        for key, value in (after.get("bucket_counts", {}) if isinstance(after.get("bucket_counts", {}), Mapping) else {}).items()
    }
    before_failed = _safe_int(before.get("failed_sample_count", 0))
    after_failed = _safe_int(after.get("failed_sample_count", 0))

    all_buckets = sorted(set(before_counts) | set(after_counts))
    count_delta = {
        bucket: int(after_counts.get(bucket, 0) - before_counts.get(bucket, 0))
        for bucket in all_buckets
    }
    share_delta = {
        bucket: _rate(after_counts.get(bucket, 0), after_failed) - _rate(before_counts.get(bucket, 0), before_failed)
        for bucket in all_buckets
    }

    return {
        "top_bucket_before": str(before.get("top_bucket", "")),
        "top_bucket_after": str(after.get("top_bucket", "")),
        "top_bucket_share_before": float(before.get("top_bucket_share", 0.0) or 0.0),
        "top_bucket_share_after": float(after.get("top_bucket_share", 0.0) or 0.0),
        "bucket_count_delta": count_delta,
        "bucket_share_delta": share_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect sample-level regime overlap buckets from combined observation logs.")
    parser.add_argument("--after", required=True, help="Path to AFTER combined observation log")
    parser.add_argument("--before", default="", help="Optional path to BEFORE combined observation log")
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
