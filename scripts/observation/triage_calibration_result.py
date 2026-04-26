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


def _quality_overview(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(payload.get("quality_overview"))


def _quality_guidance(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(payload.get("quality_guidance"))


def _format_quality_line(*, label: str, total_count: int, main_verdict: str, early_verdict: str) -> str | None:
    if total_count <= 0 and not main_verdict and not early_verdict:
        return None
    return (
        f"{label}: total={total_count} "
        f"main={main_verdict or 'n/a'} "
        f"early={early_verdict or 'n/a'}"
    )


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
    quality_overview = _quality_overview(payload)
    quality_guidance = _quality_guidance(payload)

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

    signal_line = _format_quality_line(
        label="SIGNAL_OVERVIEW",
        total_count=int(quality_overview.get("signal_total_count", 0) or 0),
        main_verdict=str(quality_overview.get("signal_main_top_verdict", "") or "").strip(),
        early_verdict=str(quality_overview.get("signal_early_top_verdict", "") or "").strip(),
    )
    if signal_line:
        lines.append(signal_line)

    exit_line = _format_quality_line(
        label="EXIT_OVERVIEW",
        total_count=int(quality_overview.get("exit_total_count", 0) or 0),
        main_verdict=str(quality_overview.get("exit_main_top_verdict", "") or "").strip(),
        early_verdict=str(quality_overview.get("exit_early_top_verdict", "") or "").strip(),
    )
    if exit_line:
        lines.append(exit_line)

    entry_focus = str(quality_guidance.get("entry_focus", "") or "").strip()
    entry_priority = str(quality_guidance.get("entry_priority", "") or "").strip()
    if entry_focus:
        lines.append(
            f"QUALITY_ENTRY: {entry_focus} priority={entry_priority or 'n/a'}"
        )

    exit_focus = str(quality_guidance.get("exit_focus", "") or "").strip()
    exit_priority = str(quality_guidance.get("exit_priority", "") or "").strip()
    if exit_focus:
        lines.append(
            f"QUALITY_EXIT: {exit_focus} priority={exit_priority or 'n/a'}"
        )

    guidance_actions = quality_guidance.get("runbook_actions", [])
    if isinstance(guidance_actions, list):
        for idx, action in enumerate(
            [str(item).strip() for item in guidance_actions if str(item).strip()][:2],
            start=1,
        ):
            lines.append(f"QUALITY_ACTION {idx}: {action}")

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
