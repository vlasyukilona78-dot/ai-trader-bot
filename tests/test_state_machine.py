import unittest

from engine.state_machine import SignalRecord, SignalState, can_transition, transition_state


class StateMachineTests(unittest.TestCase):
    def test_allowed_transitions(self):
        self.assertTrue(can_transition(SignalState.DETECTED, SignalState.CONFIRMED))
        self.assertTrue(can_transition(SignalState.CONFIRMED, SignalState.ORDERED))
        self.assertFalse(can_transition(SignalState.DETECTED, SignalState.CLOSED))

    def test_transition_updates_record(self):
        record = SignalRecord(
            signal_id="s1",
            symbol="BTC/USDT",
            direction="SHORT",
            entry=100.0,
            tp=90.0,
            sl=110.0,
            strategy="pump_short_profile",
            ai_prob=0.7,
        )
        old_ts = record.updated_at
        ok = transition_state(record, SignalState.CONFIRMED)
        self.assertTrue(ok)
        self.assertEqual(record.state, SignalState.CONFIRMED)
        self.assertGreaterEqual(record.updated_at, old_ts)


if __name__ == "__main__":
    unittest.main()
