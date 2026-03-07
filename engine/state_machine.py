from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SignalState(str, Enum):
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    ORDERED = "ordered"
    MANAGED = "managed"
    CLOSED = "closed"
    EXPIRED = "expired"
    REJECTED = "rejected"


_ALLOWED_TRANSITIONS = {
    SignalState.DETECTED: {SignalState.CONFIRMED, SignalState.EXPIRED, SignalState.REJECTED},
    SignalState.CONFIRMED: {SignalState.ORDERED, SignalState.EXPIRED, SignalState.REJECTED},
    SignalState.ORDERED: {SignalState.MANAGED, SignalState.CLOSED, SignalState.REJECTED},
    SignalState.MANAGED: {SignalState.CLOSED, SignalState.REJECTED},
    SignalState.CLOSED: set(),
    SignalState.EXPIRED: set(),
    SignalState.REJECTED: set(),
}


@dataclass
class SignalRecord:
    signal_id: str
    symbol: str
    direction: str
    entry: float
    tp: float
    sl: float
    strategy: str
    ai_prob: float | None
    state: SignalState = SignalState.DETECTED
    created_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


def can_transition(current: SignalState, target: SignalState) -> bool:
    return target in _ALLOWED_TRANSITIONS.get(current, set())


def transition_state(record: SignalRecord, target: SignalState) -> bool:
    if not can_transition(record.state, target):
        return False
    record.state = target
    record.updated_at = datetime.now(timezone.utc).timestamp()
    return True
