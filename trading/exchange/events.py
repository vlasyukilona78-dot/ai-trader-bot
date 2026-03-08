from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExchangeEventType(str, Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    HEARTBEAT = "HEARTBEAT"
    ACCOUNT = "ACCOUNT"
    POSITION = "POSITION"
    ORDER = "ORDER"
    MARKET = "MARKET"
    SNAPSHOT_REQUIRED = "SNAPSHOT_REQUIRED"
    INTERVENTION = "INTERVENTION"
    ERROR = "ERROR"


@dataclass
class NormalizedExchangeEvent:
    event_type: ExchangeEventType
    symbol: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
