from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from collections.abc import Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.signals.calibration_control import build_quality_guidance


def _load_optional_json(path_value: str) -> Mapping[str, Any]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, Mapping) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact operator guidance from signal/exit quality JSON payloads.")
    parser.add_argument("--signal-json", default="")
    parser.add_argument("--exit-json", default="")
    parser.add_argument("--timeframe", default="1")
    args = parser.parse_args()

    payload = build_quality_guidance(
        _load_optional_json(args.signal_json),
        _load_optional_json(args.exit_json),
        observation_timeframe=str(args.timeframe),
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
