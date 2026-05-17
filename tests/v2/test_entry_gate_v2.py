from __future__ import annotations

import unittest

from trading.signals.entry_gate import EntryGate, EntryGateConfig
from trading.signals.models import SignalCandidate


class EntryGateV2Tests(unittest.TestCase):
    def _candidate(self, **overrides) -> SignalCandidate:
        base = {
            "signal_id": "sig-1",
            "symbol": "BTCUSDT",
            "side": "SHORT",
            "entry": 100.0,
            "stop_loss": 101.0,
            "take_profit": 98.2,
            "confidence": 0.92,
            "timeframe": "1m",
            "mark_price": 100.0,
            "created_at": 1.0,
            "latest_atr": 1.0,
            "latest_open": 100.8,
            "latest_high": 101.0,
            "latest_low": 99.6,
            "latest_close": 100.0,
            "recent_high": 100.9,
            "recent_low": 99.0,
            "details": {
                "layer1": {"rsi": 74.0, "pump_bar_offset": 1},
                "layer2": {
                    "weakness_strength": 0.95,
                    "price_rejection_near_high": 1.0,
                    "lower_close_after_peak": 1.0,
                    "lower_high_after_peak": 1.0,
                },
                "layer3": {"entry_location_strength": 0.92, "fresh_reaction_from_high": 1.0},
                "layer4": {"source_flags": {"vwap_quality": "live"}},
                "layer5": {"tp_sl_strength": 0.90, "fallback_rr_used": 0.0},
            },
            "trace": {},
            "market_extras": {},
        }
        base.update(overrides)
        return SignalCandidate(**base)

    def test_approves_strong_short_candidate(self):
        gate = EntryGate(EntryGateConfig(min_score=0.72, min_rr=1.35))
        decision = gate.evaluate(self._candidate())
        self.assertTrue(decision.approved)
        self.assertEqual(decision.reason, "approved")
        self.assertGreaterEqual(decision.score, 0.72)
        self.assertGreaterEqual(decision.diagnostics["risk_reward_ratio"], 1.35)

    def test_rejects_bad_reward_to_risk(self):
        gate = EntryGate(EntryGateConfig(min_rr=1.35))
        decision = gate.evaluate(self._candidate(take_profit=99.0))
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "risk_reward_below_min")

    def test_rejects_late_chase_after_peak(self):
        gate = EntryGate(EntryGateConfig(hard_reject_chase_distance_atr=1.0))
        decision = gate.evaluate(self._candidate(entry=98.0, stop_loss=99.0, take_profit=96.0, recent_high=100.0))
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "entry_chasing_after_peak")


if __name__ == "__main__":
    unittest.main()
