from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.signals.replay_audit import summarize_signal_admissions
from trading.state.persistence import RuntimeStore


def _iso(ts: float) -> str:
    if ts <= 0:
        return ""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize persisted EntryGate admission decisions.")
    parser.add_argument("--db", default=str(ROOT / "data" / "runtime" / "v2_runtime.db"))
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    store = RuntimeStore(str(args.db))
    try:
        rows = store.load_signal_admissions(limit=max(1, int(args.limit)))
    finally:
        store.close()

    summary = summarize_signal_admissions(rows)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print(f"DB: {args.db}")
    print(f"Rows: {summary['total']}")
    print(f"Approved: {summary['approved']} ({summary['approval_rate']:.2%})")
    print(f"Rejected: {summary['rejected']}")
    print(f"Score avg/min/max: {summary['avg_score']:.3f}/{summary['min_score']:.3f}/{summary['max_score']:.3f}")
    print(f"Window: {_iso(summary['first_ts'])} -> {_iso(summary['last_ts'])}")
    print(f"Reasons: {json.dumps(summary['reason_counts'], ensure_ascii=False)}")
    print(f"Rejected reasons: {json.dumps(summary['rejected_reason_counts'], ensure_ascii=False)}")
    print(f"Versions: {json.dumps(summary['versions'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
