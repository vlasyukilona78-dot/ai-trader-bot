from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from app.main import (
    _close_early_lifecycle_after_managed_exit,
    _early_observation_has_worked,
    _record_short_signal_position,
    _short_barrier_exit_from_observation,
)
from trading.alerts.signal_card_clean import build_early_resolution_text
from trading.state.signal_position_tracker import SignalPositionTracker


class EarlyLifecycleV2Tests(unittest.TestCase):
    def test_profitable_reaction_is_resolved_not_invalidated(self):
        observation = {
            "favorable_excursion_pct": 4.54,
            "adverse_excursion_pct": 1.72,
            "close_move_pct": 4.10,
            "tp_hit": False,
        }

        self.assertTrue(_early_observation_has_worked(observation))

    def test_small_or_reversed_reaction_remains_invalidated(self):
        self.assertFalse(
            _early_observation_has_worked(
                {
                    "favorable_excursion_pct": 0.6,
                    "adverse_excursion_pct": 1.4,
                    "close_move_pct": -0.2,
                    "tp_hit": False,
                }
            )
        )

    def test_resolution_card_contains_observed_moves(self):
        text = build_early_resolution_text(
            symbol="BANKUSDT",
            timeframe="1",
            mode="paper",
            favorable_excursion_pct=4.54,
            adverse_excursion_pct=1.72,
            current_move_pct=4.10,
            tp_hit=False,
        )

        self.assertIn("РЕАКЦИЯ ОТРАБОТАЛА", text)
        self.assertIn("4.54%", text)
        self.assertIn("1.72%", text)
        self.assertIn("+4.10%", text)

    def test_delivered_early_signal_starts_shadow_exit_management(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = SignalPositionTracker(Path(tmpdir) / "positions.json")

            missing = _record_short_signal_position(
                tracker,
                signal_id="not-delivered",
                symbol="BANKUSDT",
                entry=0.25756,
                tp=0.23958,
                sl=0.26349,
                signal_ts=1000.0,
                delivered=False,
                source="early_watch",
            )
            opened = _record_short_signal_position(
                tracker,
                signal_id="delivered",
                symbol="BANKUSDT",
                entry=0.25756,
                tp=0.23958,
                sl=0.26349,
                signal_ts=1001.0,
                delivered=True,
                pump_id="pump-1",
                source="early_watch",
            )

            self.assertIsNone(missing)
            self.assertIsNotNone(opened)
            self.assertEqual(opened["signal_id"], "delivered")
            self.assertEqual(opened["source"], "early_watch")
            self.assertEqual(tracker.active("BANKUSDT")["entry_price"], 0.25756)

    def test_managed_exit_closes_early_lifecycle_without_second_resolution(self):
        state = {
            "AKEUSDT": {
                "active_phase": "WATCH",
                "last_emitted_phase": "WATCH",
                "last_emitted_ts": 1000.0,
                "cooldown_until_ts": 1100.0,
                "signal_id": "signal-1",
                "last_signature": "signature-1",
                "signature_cooldown_until_ts": 1200.0,
                "inactive_cycles": 2,
            }
        }

        closed = _close_early_lifecycle_after_managed_exit(
            state,
            symbol="AKEUSDT",
            now_ts=1300.0,
            cooldown_sec=300,
        )

        self.assertTrue(closed)
        self.assertEqual(state["AKEUSDT"]["active_phase"], "")
        self.assertEqual(state["AKEUSDT"]["signal_id"], "")
        self.assertEqual(state["AKEUSDT"]["inactive_cycles"], 0)
        self.assertEqual(state["AKEUSDT"]["cooldown_until_ts"], 1600.0)
        self.assertEqual(state["AKEUSDT"]["signature_cooldown_until_ts"], 1600.0)

    def test_intrabar_stop_is_resolved_even_when_close_returns_below_stop(self):
        decision = _short_barrier_exit_from_observation(
            {
                "stop_loss": 0.26077,
                "take_profit": 0.24661,
                "sl_hit_ts": 1060.0,
                "tp_hit_ts": 0.0,
                "last_close": 0.26021,
            }
        )

        self.assertEqual(decision["reason"], "signal_shadow_stop_loss_hit")
        self.assertEqual(decision["exit_price"], 0.26077)

    def test_same_bar_tp_and_sl_uses_conservative_stop(self):
        decision = _short_barrier_exit_from_observation(
            {
                "stop_loss": 102.0,
                "take_profit": 94.0,
                "sl_hit_ts": 1060.0,
                "tp_hit_ts": 1060.0,
            }
        )

        self.assertEqual(decision["exit_type"], "stop_loss_hit")


if __name__ == "__main__":
    unittest.main()
