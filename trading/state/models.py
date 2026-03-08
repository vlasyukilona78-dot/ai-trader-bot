from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class TradeState(str, Enum):
    FLAT = "FLAT"
    PENDING_ENTRY_LONG = "PENDING_ENTRY_LONG"
    LONG = "LONG"
    PENDING_EXIT_LONG = "PENDING_EXIT_LONG"
    PENDING_ENTRY_SHORT = "PENDING_ENTRY_SHORT"
    SHORT = "SHORT"
    PENDING_EXIT_SHORT = "PENDING_EXIT_SHORT"
    RECOVERING = "RECOVERING"
    HALTED = "HALTED"
    ERROR = "ERROR"


@dataclass
class StateRecord:
    symbol: str
    state: TradeState = TradeState.FLAT
    reason: str = "init"
    updated_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
