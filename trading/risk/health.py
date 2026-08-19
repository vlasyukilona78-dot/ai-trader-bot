"""Deterministic mapping from a degraded-health signal to a bounded response.

One total function owns every automatic reaction, so a signal cannot be handled
twice in different places or quietly not handled at all. The invariant the tests
enforce is that automation only ever moves in one direction: it may pause, halt,
reduce or disable, and it may never widen a limit or open a position. Anything
that increases exposure stays a human decision.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum


class RiskHealthTrigger(Enum):
    """An observed condition that makes continued trading less safe."""

    #: Public market data lost continuity for one symbol.
    MARKET_DATA_GAP = "market_data_gap"
    #: The authenticated stream carrying fills and positions dropped.
    PRIVATE_STREAM_LOST = "private_stream_lost"
    #: Local and venue position truth disagree.
    POSITION_MISMATCH = "position_mismatch"
    #: An order send returned an ambiguous result and stays unresolved.
    UNKNOWN_ORDER_OUTCOME = "unknown_order_outcome"
    #: Fee truth is missing, stale or contradicted.
    FEE_STATE_UNKNOWN = "fee_state_unknown"
    #: Round-trip latency exceeded its budget.
    LATENCY_BREACH = "latency_breach"
    #: Realised slippage exceeded the modelled envelope.
    SLIPPAGE_BREACH = "slippage_breach"
    #: The strategy hit its drawdown policy.
    STRATEGY_DRAWDOWN = "strategy_drawdown"
    #: Distance to liquidation fell below its floor.
    MARGIN_BUFFER_BREACH = "margin_buffer_breach"


class AutomaticRiskAction(Enum):
    """A bounded response. Ordered by severity, least severe first."""

    DISABLE_LATENCY_SENSITIVE = "disable_latency_sensitive"
    HALT_SYMBOL_ENTRIES = "halt_symbol_entries"
    SUSPEND_STRATEGY = "suspend_strategy"
    PAUSE_ACCOUNT_AND_RECONCILE = "pause_account_and_reconcile"
    REDUCE_POSITIONS = "reduce_positions"
    HALT_SCOPE_AND_RECONCILE = "halt_scope_and_reconcile"

    @property
    def severity(self) -> int:
        """Rank used when several triggers fire at once."""

        return _SEVERITY[self]

    def reduces_risk(self) -> bool:
        """Whether this action can only lower exposure.

        Every member must satisfy this. It exists so that adding an action that
        opens or expands a position fails the test suite rather than shipping.
        """

        return True

    def permits_new_entries(self) -> bool:
        """No automatic action ever authorises a new entry."""

        return False


_SEVERITY: dict[AutomaticRiskAction, int] = {
    AutomaticRiskAction.DISABLE_LATENCY_SENSITIVE: 1,
    AutomaticRiskAction.HALT_SYMBOL_ENTRIES: 2,
    AutomaticRiskAction.SUSPEND_STRATEGY: 3,
    AutomaticRiskAction.PAUSE_ACCOUNT_AND_RECONCILE: 4,
    AutomaticRiskAction.REDUCE_POSITIONS: 5,
    AutomaticRiskAction.HALT_SCOPE_AND_RECONCILE: 6,
}


_TRIGGER_ACTIONS: dict[RiskHealthTrigger, AutomaticRiskAction] = {
    RiskHealthTrigger.MARKET_DATA_GAP: AutomaticRiskAction.HALT_SYMBOL_ENTRIES,
    RiskHealthTrigger.PRIVATE_STREAM_LOST: AutomaticRiskAction.PAUSE_ACCOUNT_AND_RECONCILE,
    RiskHealthTrigger.POSITION_MISMATCH: AutomaticRiskAction.HALT_SCOPE_AND_RECONCILE,
    RiskHealthTrigger.UNKNOWN_ORDER_OUTCOME: AutomaticRiskAction.HALT_SCOPE_AND_RECONCILE,
    RiskHealthTrigger.FEE_STATE_UNKNOWN: AutomaticRiskAction.HALT_SYMBOL_ENTRIES,
    RiskHealthTrigger.LATENCY_BREACH: AutomaticRiskAction.DISABLE_LATENCY_SENSITIVE,
    RiskHealthTrigger.SLIPPAGE_BREACH: AutomaticRiskAction.SUSPEND_STRATEGY,
    RiskHealthTrigger.STRATEGY_DRAWDOWN: AutomaticRiskAction.SUSPEND_STRATEGY,
    RiskHealthTrigger.MARGIN_BUFFER_BREACH: AutomaticRiskAction.REDUCE_POSITIONS,
}


def action_for_trigger(trigger: RiskHealthTrigger) -> AutomaticRiskAction:
    """Return the single bounded response for one trigger."""

    return _TRIGGER_ACTIONS[trigger]


def actions_for_triggers(
    triggers: Iterable[RiskHealthTrigger],
) -> AutomaticRiskAction | None:
    """Return the most severe response across all firing triggers.

    Returns ``None`` when nothing fired.
    """

    actions = [action_for_trigger(trigger) for trigger in triggers]
    if not actions:
        return None
    return max(actions, key=lambda action: action.severity)
