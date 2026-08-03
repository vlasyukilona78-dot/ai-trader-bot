from __future__ import annotations

import unittest

import pandas as pd

from trading.signals.entry_gate import EntryGate, EntryGateConfig
from trading.signals.models import SignalCandidate
from trading.signals.replay_audit import summarize_signal_admissions


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

    def test_rejects_strong_mtf_continuation(self):
        gate = EntryGate(EntryGateConfig(min_score=0.70))
        decision = gate.evaluate(
            self._candidate(
                market_extras={
                    "mtf_trend_1h": 0.0040,
                    "mtf_rsi_1h": 66.0,
                    "mtf_trend_15m": 0.0010,
                    "mtf_rsi_15m": 55.0,
                    "mtf_trend_5m": 0.0010,
                    "mtf_rsi_5m": 55.0,
                }
            )
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "mtf_continuation_block")
        self.assertTrue(decision.diagnostics["hard_1h"])

    def test_rejects_live_continuation_without_rejection(self):
        gate = EntryGate(EntryGateConfig(min_score=0.70))
        decision = gate.evaluate(
            self._candidate(
                latest_open=99.8,
                latest_high=100.8,
                latest_low=99.4,
                latest_close=100.55,
                recent_high=100.8,
                details={
                    "layer1": {"rsi": 73.0, "pump_bar_offset": 1, "volume_spike": 1.05},
                    "layer2": {"weakness_strength": 0.82},
                    "layer3": {"entry_location_strength": 0.82},
                    "layer4": {"source_flags": {"vwap_quality": "live"}},
                    "layer5": {"tp_sl_strength": 0.88},
                },
                market_extras={
                    "mtf_trend_5m": 0.0082,
                    "mtf_rsi_5m": 70.5,
                    "volume_spike": 1.05,
                    "vwap_dist": 0.005,
                    "bb_position": 0.82,
                    "ema20": 99.2,
                    "adx": 31.0,
                },
            )
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "live_continuation_without_rejection")

    def test_rejects_low_volume_pullback_in_live_drive(self):
        gate = EntryGate(EntryGateConfig(min_score=0.70))
        decision = gate.evaluate(
            self._candidate(
                latest_open=100.45,
                latest_high=100.6,
                latest_low=99.3,
                latest_close=100.0,
                recent_high=100.8,
                details={
                    "layer1": {"rsi": 57.0, "pump_bar_offset": 1, "volume_spike": 0.55},
                    "layer2": {"weakness_strength": 0.78},
                    "layer3": {"entry_location_strength": 0.80},
                    "layer4": {"source_flags": {"vwap_quality": "live"}},
                    "layer5": {"tp_sl_strength": 0.84},
                },
                market_extras={
                    "mtf_trend_5m": 0.0090,
                    "mtf_rsi_5m": 64.0,
                    "volume_spike": 0.55,
                    "vwap_dist": 0.004,
                    "adx": 28.0,
                },
            )
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "low_volume_pullback_without_displacement")

    def test_allows_live_drive_after_real_failure(self):
        gate = EntryGate(EntryGateConfig(min_score=0.70))
        decision = gate.evaluate(
            self._candidate(
                latest_open=100.45,
                latest_high=101.0,
                latest_low=99.2,
                latest_close=100.0,
                recent_high=101.0,
                market_extras={
                    "mtf_trend_5m": 0.0090,
                    "mtf_rsi_5m": 64.0,
                    "volume_spike": 0.55,
                    "vwap_dist": 0.004,
                    "adx": 28.0,
                },
            )
        )
        self.assertTrue(decision.approved)
        self.assertEqual(decision.reason, "approved")

    def test_rejects_bad_microstructure_execution_risk(self):
        gate = EntryGate(EntryGateConfig(min_score=0.70))
        decision = gate.evaluate(
            self._candidate(
                market_extras={
                    "spread_bps": 42.0,
                    "expected_slippage_bps": 8.0,
                    "depth_ratio": 1.4,
                },
            )
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "microstructure_execution_risk")
        self.assertIn("spread_too_wide", decision.diagnostics["hard_reasons"])

    def test_penalizes_soft_microstructure_without_hard_reject(self):
        gate = EntryGate(EntryGateConfig(min_score=0.60))
        decision = gate.evaluate(
            self._candidate(
                market_extras={
                    "spread_bps": 23.0,
                    "expected_slippage_bps": 31.0,
                    "depth_ratio": 0.92,
                    "bid_ask_imbalance": 0.72,
                    "aggressor_exhaustion": 0.12,
                },
            )
        )
        self.assertTrue(decision.approved)
        self.assertTrue(decision.flags["microstructure_soft_risk"])
        self.assertGreater(decision.penalties["microstructure_soft_risk"], 0.0)
        self.assertIn("microstructure_context", decision.diagnostics)

    def test_strong_degraded_candidate_can_reach_default_degraded_threshold(self):
        details = {
            "layer1": {"rsi": 74.0, "pump_bar_offset": 1},
            "layer2": {
                "weakness_strength": 1.0,
                "price_rejection_near_high": 1.0,
                "lower_close_after_peak": 1.0,
                "lower_high_after_peak": 1.0,
                "failed_reclaim": 1.0,
            },
            "layer3": {
                "entry_location_strength": 1.0,
                "fresh_reaction_from_high": 1.0,
                "failed_reclaim": 1.0,
            },
            "layer4": {
                "degraded_mode": 1.0,
                "source_flags": {
                    "sentiment_source": "unavailable",
                    "sentiment_quality": "unavailable",
                    "sentiment_unavailable": 1.0,
                    "funding_source": "live:bybit:ticker",
                    "funding_quality": "live",
                    "funding_live_used": 1.0,
                    "long_short_ratio_quality": "live",
                    "open_interest_quality": "live",
                    "vwap_quality": "live",
                },
            },
            "layer5": {"tp_sl_strength": 1.0, "fallback_rr_used": 0.0},
        }
        gate = EntryGate(EntryGateConfig())

        decision = gate.evaluate(
            self._candidate(
                confidence=0.70,
                recent_high=100.6,
                details=details,
            )
        )

        self.assertTrue(decision.approved, msg=str(decision))
        self.assertEqual(decision.reason, "approved")
        self.assertEqual(decision.diagnostics["min_score_used"], 0.78)
        self.assertGreaterEqual(decision.score, 0.78)
        self.assertTrue(decision.flags["degraded_context"])

    def test_context_quality_counts_quality_dimensions_not_service_flags(self):
        details = {
            "layer4": {
                "degraded_mode": 1.0,
                "source_flags": {
                    "sentiment_source": "unavailable",
                    "sentiment_quality": "unavailable",
                    "sentiment_unavailable": 1.0,
                    "funding_source": "live:bybit:ticker",
                    "funding_quality": "live",
                    "funding_live_used": 1.0,
                    "long_short_ratio_quality": "live",
                    "open_interest_quality": "live",
                    "vwap_quality": "live",
                },
            }
        }

        quality = EntryGate._context_quality(details)

        self.assertAlmostEqual(quality, 0.624, places=3)

    def test_signal_candidate_collects_microstructure_from_context(self):
        class Signal:
            signal_id = "sig-context"
            side = "SHORT"
            entry = 100.0
            sl = 101.0
            tp = 98.0
            confidence = 0.9
            created_at = 1.0
            details = {}

        class Context:
            symbol = "BTCUSDT"
            timeframe = "1m"
            mark_price = 100.0
            spread_bps = 12.0
            expected_slippage_bps = 4.0
            depth_ratio = 1.8
            bid_ask_imbalance = 0.44
            aggressor_exhaustion = 0.72

        candidate = SignalCandidate.from_signal(
            signal=Signal(),
            context=Context(),
            enriched=pd.DataFrame(
                [{"open": 99.8, "high": 100.5, "low": 99.5, "close": 100.0, "atr": 1.0}]
            ),
        )
        self.assertEqual(candidate.market_extras["spread_bps"], 12.0)
        self.assertEqual(candidate.market_extras["expected_slippage_bps"], 4.0)
        self.assertEqual(candidate.market_extras["depth_ratio"], 1.8)
        self.assertEqual(candidate.market_extras["bid_ask_imbalance"], 0.44)
        self.assertEqual(candidate.market_extras["aggressor_exhaustion"], 0.72)

    def test_can_require_mtf_context(self):
        gate = EntryGate(EntryGateConfig(require_mtf_context=True))
        decision = gate.evaluate(self._candidate())
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "mtf_context_missing")

    def test_readiness_flags_prevent_neutral_mtf_from_passing_as_live(self):
        gate = EntryGate(EntryGateConfig(require_mtf_context=True))
        decision = gate.evaluate(
            self._candidate(
                market_extras={
                    "mtf_trend_1h": 0.0,
                    "mtf_trend_15m": 0.0,
                    "mtf_trend_5m": 0.0,
                    "mtf_rsi_1h": 50.0,
                    "mtf_rsi_15m": 50.0,
                    "mtf_rsi_5m": 50.0,
                    "mtf_ready_1h": 0.0,
                    "mtf_ready_15m": 0.0,
                    "mtf_ready_5m": 1.0,
                }
            )
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "mtf_context_missing")
        self.assertFalse(decision.diagnostics["ready_1h"])

    def test_replay_admission_summary(self):
        class Row:
            def __init__(self, approved: bool, reason: str, score: float, symbol: str):
                self.approved = approved
                self.reason = reason
                self.score = score
                self.symbol = symbol
                self.ts = 1.0
                self.raw = {"entry_gate": {"version": "test_gate"}}

        summary = summarize_signal_admissions(
            [
                Row(True, "approved", 0.84, "BTCUSDT"),
                Row(False, "mtf_continuation_block", 0.0, "BTCUSDT"),
            ]
        )
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["approved"], 1)
        self.assertEqual(summary["rejected_reason_counts"]["mtf_continuation_block"], 1)
        self.assertEqual(summary["versions"]["test_gate"], 2)


if __name__ == "__main__":
    unittest.main()
