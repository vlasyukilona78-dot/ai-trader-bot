"""Hard-breach lifecycle as an explicit, non-skippable state machine.

When a hard limit breaks, the safe response is a fixed sequence: stop new
entries, cancel exposure-increasing orders, reconcile the cancel/fill races
that cancelling creates, reduce what remains, halt, and tell the operator.
Each step needs its own durable evidence before the next one is allowed.

Returning to trading is deliberately hard: it needs an operator confirmation,
reconciled venue state, and exposure back inside the profile — all three, at
once. Waiting is not recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HardBreachReason(Enum):
    """Why trading was stopped."""

    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    GROSS_EXPOSURE = "gross_exposure"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    MANUAL_EMERGENCY_STOP = "manual_emergency_stop"
    RISK_ENGINE_FAILURE = "risk_engine_failure"


class RiskSessionState(Enum):
    """Ordered stages of the breach response."""

    ACTIVE = "active"
    NO_NEW_ENTRIES = "no_new_entries"
    CANCELLING_ORDERS = "cancelling_orders"
    RECONCILING = "reconciling"
    REDUCING_POSITIONS = "reducing_positions"
    HALTED = "halted"
    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"

    def permits_new_entries(self) -> bool:
        """Only a fully active session may open anything."""

        return self is RiskSessionState.ACTIVE


class RiskSessionError(RuntimeError):
    """Evidence arrived that would skip or reorder a required safety step."""


@dataclass(frozen=True)
class RiskSessionEvent:
    """Durable evidence that one step of the response completed."""

    kind: str
    reason: HardBreachReason | None = None
    operator_confirmed: bool = False
    reconciled: bool = False
    within_limits: bool = False

    @classmethod
    def hard_breach(cls, reason: HardBreachReason) -> "RiskSessionEvent":
        return cls(kind="hard_breach", reason=reason)

    @classmethod
    def recovery_proof(
        cls, *, operator_confirmed: bool, reconciled: bool, within_limits: bool
    ) -> "RiskSessionEvent":
        return cls(
            kind="recovery_proof",
            operator_confirmed=operator_confirmed,
            reconciled=reconciled,
            within_limits=within_limits,
        )


# Simple evidence markers, declared after the class so they are instances of it.
RiskSessionEvent.ORDERS_CANCELLED = RiskSessionEvent(kind="orders_cancelled")  # type: ignore[attr-defined]
RiskSessionEvent.RECONCILED = RiskSessionEvent(kind="reconciled")  # type: ignore[attr-defined]
RiskSessionEvent.REDUCTION_STARTED = RiskSessionEvent(kind="reduction_started")  # type: ignore[attr-defined]
RiskSessionEvent.POSITIONS_REDUCED = RiskSessionEvent(kind="positions_reduced")  # type: ignore[attr-defined]
RiskSessionEvent.OPERATOR_NOTIFIED = RiskSessionEvent(kind="operator_notified")  # type: ignore[attr-defined]


_SEQUENCE: dict[tuple[RiskSessionState, str], RiskSessionState] = {
    (RiskSessionState.ACTIVE, "hard_breach"): RiskSessionState.NO_NEW_ENTRIES,
    (RiskSessionState.NO_NEW_ENTRIES, "orders_cancelled"): RiskSessionState.CANCELLING_ORDERS,
    (RiskSessionState.CANCELLING_ORDERS, "reconciled"): RiskSessionState.RECONCILING,
    (RiskSessionState.RECONCILING, "reduction_started"): RiskSessionState.REDUCING_POSITIONS,
    (RiskSessionState.REDUCING_POSITIONS, "positions_reduced"): RiskSessionState.HALTED,
    (RiskSessionState.HALTED, "operator_notified"): RiskSessionState.MANUAL_RECOVERY_REQUIRED,
}


class RiskSessionStateMachine:
    """Applies one evidenced transition at a time, refusing anything else."""

    def __init__(self) -> None:
        self._state = RiskSessionState.ACTIVE
        self._breach_reason: HardBreachReason | None = None

    @property
    def state(self) -> RiskSessionState:
        return self._state

    @property
    def breach_reason(self) -> HardBreachReason | None:
        """The original cause, retained until a successful recovery."""

        return self._breach_reason

    def permits_new_entries(self) -> bool:
        return self._state.permits_new_entries()

    def apply(self, event: RiskSessionEvent) -> RiskSessionState:
        """Advance exactly one step.

        Raises:
            RiskSessionError: The event would skip, reorder or repeat a step.
                The current state is left untouched.
        """

        if event.kind == "recovery_proof":
            return self._apply_recovery(event)

        nxt = _SEQUENCE.get((self._state, event.kind))
        if nxt is None:
            raise RiskSessionError(
                f"event {event.kind!r} is not valid in state {self._state.value!r}"
            )

        if event.kind == "hard_breach":
            self._breach_reason = event.reason
        self._state = nxt
        return self._state

    def _apply_recovery(self, event: RiskSessionEvent) -> RiskSessionState:
        if self._state is not RiskSessionState.MANUAL_RECOVERY_REQUIRED:
            raise RiskSessionError(
                f"recovery proof is not valid in state {self._state.value!r}"
            )
        if not (event.operator_confirmed and event.reconciled and event.within_limits):
            missing = [
                name
                for name, ok in (
                    ("operator_confirmed", event.operator_confirmed),
                    ("reconciled", event.reconciled),
                    ("within_limits", event.within_limits),
                )
                if not ok
            ]
            raise RiskSessionError(f"recovery proof incomplete: {', '.join(missing)}")

        self._state = RiskSessionState.ACTIVE
        self._breach_reason = None
        return self._state
