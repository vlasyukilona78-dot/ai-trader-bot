from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading.state.signal_position_tracker import SignalPositionTracker


def _tracker(path: str) -> SignalPositionTracker:
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    return SignalPositionTracker(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage a manually traded short linked to bot exit monitoring.")
    parser.add_argument(
        "--path",
        default="data/runtime/signal_positions.json",
        help="Tracker JSON path, relative to the project root by default.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    open_cmd = sub.add_parser("open", help="Register or correct a manual short.")
    open_cmd.add_argument("symbol")
    open_cmd.add_argument("--entry", type=float, required=True)
    open_cmd.add_argument("--stop", type=float, required=True)
    open_cmd.add_argument("--take-profit", type=float, default=0.0)
    open_cmd.add_argument("--leverage", type=float, default=0.0)
    open_cmd.add_argument("--pump-id", default="")
    open_cmd.add_argument("--replace", action="store_true")

    close_cmd = sub.add_parser("close", help="Stop monitoring a manual short.")
    close_cmd.add_argument("symbol")
    close_cmd.add_argument("--price", type=float, required=True)
    close_cmd.add_argument("--reason", default="manual_close")

    status_cmd = sub.add_parser("status", help="Show one tracked position.")
    status_cmd.add_argument("symbol")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tracker = _tracker(args.path)
    if args.command == "open":
        result = tracker.record_short(
            symbol=args.symbol,
            entry_price=args.entry,
            stop_loss=args.stop,
            take_profit=args.take_profit,
            pump_id=args.pump_id,
            leverage=args.leverage,
            source="manual_cli",
            replace=bool(args.replace),
        )
    elif args.command == "close":
        result = tracker.close(
            args.symbol,
            exit_price=args.price,
            reason=args.reason,
        )
    else:
        result = tracker.active(args.symbol)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
