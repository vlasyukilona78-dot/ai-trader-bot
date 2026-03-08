from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class IntentAction(str, Enum):
    HOLD = "HOLD"
    LONG_ENTRY = "LONG_ENTRY"
    SHORT_ENTRY = "SHORT_ENTRY"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"


@dataclass
class StrategyIntent:
    symbol: str
    action: IntentAction
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
