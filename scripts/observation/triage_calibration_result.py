from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXIT_CODES: dict[str, int] = {
    "co_dominant_regime_blockers": 10,
    "blocker_semantics_disagreement": 11,
    "market_context_shift_detected": 12,
    "window_size_not_comparable": 13,
}

DEFAULT_PAUSE_EXIT_CODE = 14


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _load_payload(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Calibration result JSON must be an object.")
    return data


def _recommendation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    recommendation = _as_mapping(payload.get("calibration_recommendation"))
    if recommendation:
        return recommendation
    raise ValueError("calibration_recommendation is missing from the JSON payload.")


def _after_summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    after = _as_mapping(payload.get("after"))
    return _as_mapping(after.get("summary"))


def exit_code_for_recommendation(recommendation: Mapping[str, Any]) -> int:
    if bool(recommendation.get("SAFE_TO_CONTINUE", False)):
        return 0

    action_verdict = str(recommendation.get("ACTION_VERDICT", "")).strip()
    if action_verdict == "single_blocker_ready":
        return 0

    stop_reason = str(recommendation.get("STOP_REASON", "")).strip()
    if stop_reason in EXIT_CODES:
        return EXIT_CODES[stop_reason]
    return DEFAULT_PAUSE_EXIT_CODE


def format_triage_lines(payload: Mapping[str, Any], *, max_actions: int) -> list[str]:
    recommendation = _recommendation(payload)
    after_summary = _after_summary(payload)

    verdict = str(recommendation.get("ACTION_VERDICT", "")).strip() or "unknown"
    stop_reason = str(recommendation.get("STOP_REASON", "")).strip() or "none"
    top_combination = str(after_summary.get("top_regime_filter_blocker_combination", "")).strip() or "none"
    raw_actions = recommendation.get("RUNBOOK_ACTIONS", [])
    actions = [str(item).strip() for item in raw_actions if str(item).strip()] if isinstance(raw_actions, list) else []

    lines = [
        f"VERDICT: {verdict}",
        f"STOP_REASON: {stop_reason}",
        f"TOP_COMBINATION: {top_combination}",
    ]
    for idx, action in enumerate(actions[: max(max_actions, 0)], start=1):
        lines.append(f"ACTION {idx}: {action}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact operator triage view from a calibration result JSON.")
    parser.add_argument("json_path", help="Path to summarize_observation comparison JSON")
    parser.add_argument("--max-actions", type=int, default=4, help="Maximum number of RUNBOOK_ACTIONS lines to print")
    args = parser.parse_args()

    payload = _load_payload(Path(args.json_path))
    lines = format_triage_lines(payload, max_actions=int(args.max_actions))
    print("\n".join(lines))
    return exit_code_for_recommendation(_recommendation(payload))


if __name__ == "__main__":
    raise SystemExit(main())
