"""Breaching a hard limit follows a sequence that cannot be skipped.

A boolean "halted" flag lets recovery happen by simply waiting, and lets the
reduction steps run out of order or not at all. Making the breach a state
machine means each step needs its own durable evidence, and returning to
trading needs three conditions at once rather than a timeout.
"""

from __future__ import annotations

import unittest

from trading.risk.session import (
    HardBreachReason,
    RiskSessionError,
    RiskSessionEvent,
    RiskSessionState,
    RiskSessionStateMachine,
)


class EntryPermissionTests(unittest.TestCase):
    def test_only_active_permits_new_entries(self):
        for state in RiskSessionState:
            with self.subTest(state=state):
                self.assertEqual(
                    state.permits_new_entries(), state is RiskSessionState.ACTIVE
                )

    def test_a_fresh_session_starts_active(self):
        self.assertIs(RiskSessionStateMachine().state, RiskSessionState.ACTIVE)


class BreachSequenceTests(unittest.TestCase):
    def setUp(self):
        self.machine = RiskSessionStateMachine()

    def test_hard_breach_stops_new_entries_immediately(self):
        state = self.machine.apply(
            RiskSessionEvent.hard_breach(HardBreachReason.DAILY_LOSS)
        )

        self.assertIs(state, RiskSessionState.NO_NEW_ENTRIES)
        self.assertFalse(state.permits_new_entries())

    def test_the_breach_reason_is_retained(self):
        self.machine.apply(RiskSessionEvent.hard_breach(HardBreachReason.DRAWDOWN))

        self.assertIs(self.machine.breach_reason, HardBreachReason.DRAWDOWN)

    def test_the_full_sequence_reaches_manual_recovery(self):
        self.machine.apply(RiskSessionEvent.hard_breach(HardBreachReason.DAILY_LOSS))
        self.machine.apply(RiskSessionEvent.ORDERS_CANCELLED)
        self.machine.apply(RiskSessionEvent.RECONCILED)
        self.machine.apply(RiskSessionEvent.REDUCTION_STARTED)
        self.machine.apply(RiskSessionEvent.POSITIONS_REDUCED)
        state = self.machine.apply(RiskSessionEvent.OPERATOR_NOTIFIED)

        self.assertIs(state, RiskSessionState.MANUAL_RECOVERY_REQUIRED)

    def test_a_step_cannot_be_skipped(self):
        self.machine.apply(RiskSessionEvent.hard_breach(HardBreachReason.DAILY_LOSS))

        with self.assertRaises(RiskSessionError):
            self.machine.apply(RiskSessionEvent.POSITIONS_REDUCED)

    def test_state_is_unchanged_after_a_rejected_event(self):
        self.machine.apply(RiskSessionEvent.hard_breach(HardBreachReason.DAILY_LOSS))

        with self.assertRaises(RiskSessionError):
            self.machine.apply(RiskSessionEvent.POSITIONS_REDUCED)

        self.assertIs(self.machine.state, RiskSessionState.NO_NEW_ENTRIES)

    def test_a_second_breach_does_not_restart_the_sequence(self):
        self.machine.apply(RiskSessionEvent.hard_breach(HardBreachReason.DAILY_LOSS))
        self.machine.apply(RiskSessionEvent.ORDERS_CANCELLED)

        with self.assertRaises(RiskSessionError):
            self.machine.apply(RiskSessionEvent.hard_breach(HardBreachReason.DRAWDOWN))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.machine = RiskSessionStateMachine()
        self.machine.apply(RiskSessionEvent.hard_breach(HardBreachReason.DAILY_LOSS))
        self.machine.apply(RiskSessionEvent.ORDERS_CANCELLED)
        self.machine.apply(RiskSessionEvent.RECONCILED)
        self.machine.apply(RiskSessionEvent.REDUCTION_STARTED)
        self.machine.apply(RiskSessionEvent.POSITIONS_REDUCED)
        self.machine.apply(RiskSessionEvent.OPERATOR_NOTIFIED)

    def test_full_proof_returns_to_active(self):
        state = self.machine.apply(
            RiskSessionEvent.recovery_proof(
                operator_confirmed=True, reconciled=True, within_limits=True
            )
        )

        self.assertIs(state, RiskSessionState.ACTIVE)

    def test_recovery_clears_the_breach_reason(self):
        self.machine.apply(
            RiskSessionEvent.recovery_proof(
                operator_confirmed=True, reconciled=True, within_limits=True
            )
        )

        self.assertIsNone(self.machine.breach_reason)

    def test_partial_proof_is_refused(self):
        cases = [
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ]
        for confirmed, reconciled, within in cases:
            with self.subTest(confirmed=confirmed, reconciled=reconciled, within=within):
                with self.assertRaises(RiskSessionError):
                    self.machine.apply(
                        RiskSessionEvent.recovery_proof(
                            operator_confirmed=confirmed,
                            reconciled=reconciled,
                            within_limits=within,
                        )
                    )
                self.assertIs(
                    self.machine.state, RiskSessionState.MANUAL_RECOVERY_REQUIRED
                )


if __name__ == "__main__":
    unittest.main()
