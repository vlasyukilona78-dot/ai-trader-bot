from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def _bucket_score(score: float) -> str:
    value = max(0.0, min(1.0, float(score)))
    low = int(value * 10) / 10
    high = min(1.0, low + 0.1)
    return f"{low:.1f}-{high:.1f}"


def _raw_version(raw: Mapping[str, Any]) -> str:
    gate = raw.get("entry_gate")
    if isinstance(gate, Mapping):
        version = gate.get("version")
        if version:
            return str(version)
    version = raw.get("strategy_version")
    return str(version) if version else "unknown"


def summarize_signal_admissions(rows: Iterable[Any]) -> dict[str, Any]:
    rows_list = list(rows)
    total = len(rows_list)
    approved = [row for row in rows_list if bool(getattr(row, "approved", False))]
    rejected = [row for row in rows_list if not bool(getattr(row, "approved", False))]
    reason_counts = Counter(str(getattr(row, "reason", "")) for row in rows_list)
    rejected_reason_counts = Counter(str(getattr(row, "reason", "")) for row in rejected)
    approved_reason_counts = Counter(str(getattr(row, "reason", "")) for row in approved)
    symbol_counts = Counter(str(getattr(row, "symbol", "")) for row in rows_list)
    score_buckets = Counter(_bucket_score(float(getattr(row, "score", 0.0))) for row in rows_list)
    versions = Counter(
        _raw_version(getattr(row, "raw", {}) if isinstance(getattr(row, "raw", {}), Mapping) else {})
        for row in rows_list
    )
    scores = [float(getattr(row, "score", 0.0)) for row in rows_list]
    return {
        "total": total,
        "approved": len(approved),
        "rejected": len(rejected),
        "approval_rate": (len(approved) / total) if total else 0.0,
        "avg_score": (sum(scores) / total) if total else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "first_ts": min((float(getattr(row, "ts", 0.0)) for row in rows_list), default=0.0),
        "last_ts": max((float(getattr(row, "ts", 0.0)) for row in rows_list), default=0.0),
        "reason_counts": dict(reason_counts.most_common()),
        "rejected_reason_counts": dict(rejected_reason_counts.most_common()),
        "approved_reason_counts": dict(approved_reason_counts.most_common()),
        "score_buckets": dict(sorted(score_buckets.items())),
        "top_symbols": dict(symbol_counts.most_common(20)),
        "versions": dict(versions.most_common()),
    }
