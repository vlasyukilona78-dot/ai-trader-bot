"""Every degraded-health signal maps to one bounded, risk-reducing action.

Scattering these reactions across the codebase means some signals get handled
twice and others not at all. Keeping the mapping in one total function makes
the gaps visible, and makes the invariant checkable: automation may pause,
halt or reduce, but it may never widen a limit or open a position.
"""

from __future__ import annotations

import unittest

from trading.risk.health import (
    AutomaticRiskAction,
    RiskHealthTrigger,
    action_for_trigger,
    actions_for_triggers,
)


class MappingTests(unittest.TestCase):
    def test_every_trigger_has_an_action(self):
        for trigger in RiskHealthTrigger:
            with self.subTest(trigger=trigger):
                self.assertIsInstance(action_for_trigger(trigger), AutomaticRiskAction)

    def test_market_data_gap_halts_only_the_affected_symbol(self):
        self.assertIs(
            action_for_trigger(RiskHealthTrigger.MARKET_DATA_GAP),
            AutomaticRiskAction.HALT_SYMBOL_ENTRIES,
        )

    def test_lost_private_stream_pauses_the_whole_account(self):
        # Without the private stream we cannot see fills, so no symbol is safe.
        self.assertIs(
            action_for_trigger(RiskHealthTrigger.PRIVATE_STREAM_LOST),
            AutomaticRiskAction.PAUSE_ACCOUNT_AND_RECONCILE,
        )

    def test_position_mismatch_halts_and_reconciles(self):
        self.assertIs(
            action_for_trigger(RiskHealthTrigger.POSITION_MISMATCH),
            AutomaticRiskAction.HALT_SCOPE_AND_RECONCILE,
        )

    def test_unknown_order_outcome_halts_and_reconciles(self):
        # A send whose result was lost may have created a position we cannot see.
        self.assertIs(
            action_for_trigger(RiskHealthTrigger.UNKNOWN_ORDER_OUTCOME),
            AutomaticRiskAction.HALT_SCOPE_AND_RECONCILE,
        )

    def test_margin_buffer_breach_reduces_positions(self):
        self.assertIs(
            action_for_trigger(RiskHealthTrigger.MARGIN_BUFFER_BREACH),
            AutomaticRiskAction.REDUCE_POSITIONS,
        )


class InvariantTests(unittest.TestCase):
    def test_no_action_can_open_or_expand_exposure(self):
        for action in AutomaticRiskAction:
            with self.subTest(action=action):
                self.assertTrue(
                    action.reduces_risk(),
                    "automation must never widen a limit or open a position",
                )

    def test_halting_actions_block_new_entries(self):
        for action in AutomaticRiskAction:
            with self.subTest(action=action):
                self.assertFalse(action.permits_new_entries())


class SeverityTests(unittest.TestCase):
    def test_the_most_severe_action_wins_when_several_triggers_fire(self):
        chosen = actions_for_triggers(
            [RiskHealthTrigger.SLIPPAGE_BREACH, RiskHealthTrigger.POSITION_MISMATCH]
        )

        self.assertIs(chosen, AutomaticRiskAction.HALT_SCOPE_AND_RECONCILE)

    def test_no_triggers_means_no_action(self):
        self.assertIsNone(actions_for_triggers([]))

    def test_reducing_beats_pausing(self):
        chosen = actions_for_triggers(
            [RiskHealthTrigger.LATENCY_BREACH, RiskHealthTrigger.MARGIN_BUFFER_BREACH]
        )

        self.assertIs(chosen, AutomaticRiskAction.REDUCE_POSITIONS)


if __name__ == "__main__":
    unittest.main()
