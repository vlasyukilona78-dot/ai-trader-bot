from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "observation" / "analyze_recent_signal_quality.py"
SPEC = importlib.util.spec_from_file_location("analyze_recent_signal_quality", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ObservationSignalQualityV2Tests(unittest.TestCase):
    def test_enrich_early_alerts_with_db_execution_attaches_real_entry_levels(self):
        alert = MODULE.SignalEvent(
            profile="early",
            kind="early_alert",
            symbol="TESTUSDT",
            action="SHORT_ENTRY",
            exec_status="IGNORED",
            ts=1_000.0,
            delivery_ts=1_010.0,
            entry_price=None,
            tp=None,
            sl=None,
            risk_reason="approved",
            exec_reason="early_profile_monitor_only",
            order_link_id="",
            raw={},
        )
        decision = MODULE.SignalEvent(
            profile="early",
            kind="db_decision",
            symbol="TESTUSDT",
            action="SHORT_ENTRY",
            exec_status="FILLED",
            ts=1_120.0,
            delivery_ts=None,
            entry_price=105.0,
            tp=101.0,
            sl=107.0,
            risk_reason="approved",
            exec_reason="entry_filled",
            order_link_id="oid-1",
            raw={},
        )

        enriched = MODULE._enrich_early_alerts_with_db_execution([alert], [decision])

        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].entry_price, 105.0)
        self.assertEqual(enriched[0].tp, 101.0)
        self.assertEqual(enriched[0].sl, 107.0)
        self.assertEqual(enriched[0].exec_status, "FILLED")
        self.assertEqual(enriched[0].order_link_id, "oid-1")
        self.assertIn("matched_early_decision_ts", enriched[0].raw)

    def test_bars_until_move_pct_detects_first_down_and_up_bar(self):
        index = pd.date_range("2026-04-01T00:00:00Z", periods=4, freq="1min")
        window = pd.DataFrame(
            {
                "high": [100.10, 100.55, 100.20, 100.80],
                "low": [99.92, 99.85, 99.40, 99.70],
            },
            index=index,
        )

        down_bars = MODULE._bars_until_move_pct(window, entry=100.0, direction="down", threshold_pct=0.35)
        up_bars = MODULE._bars_until_move_pct(window, entry=100.0, direction="up", threshold_pct=0.35)

        self.assertEqual(down_bars, 3)
        self.assertEqual(up_bars, 2)

    def test_classify_short_signal_outcome_flags_continuation_trap(self):
        verdict = MODULE._classify_short_signal_outcome(
            kind="early_alert",
            first_reaction="up",
            up_15=1.2,
            down_15=0.2,
            up_60=1.8,
            down_60=0.4,
            tp_hit_60=False,
            sl_hit_60=False,
            bars_to_first_down_move=None,
            bars_to_first_up_move=1,
        )

        self.assertEqual(verdict, "continuation_trap")

    def test_classify_short_signal_outcome_flags_late_when_down_move_arrives_too_late(self):
        verdict = MODULE._classify_short_signal_outcome(
            kind="db_decision",
            first_reaction="flat",
            up_15=0.6,
            down_15=0.5,
            up_60=0.8,
            down_60=0.6,
            tp_hit_60=False,
            sl_hit_60=False,
            bars_to_first_down_move=9,
            bars_to_first_up_move=2,
        )

        self.assertEqual(verdict, "late_or_weak")

    def test_summarize_results_reports_verdict_mix_and_latency(self):
        summary = MODULE._summarize_results(
            [
                {
                    "verdict": "worked",
                    "favorable_excursion_15m_pct": 1.2,
                    "adverse_excursion_15m_pct": 0.4,
                    "favorable_excursion_60m_pct": 2.3,
                    "adverse_excursion_60m_pct": 0.7,
                    "minutes_to_first_down_035pct": 2.0,
                    "minutes_to_first_up_035pct": 1.0,
                    "tp_hit_60m": True,
                    "sl_hit_60m": False,
                    "bar_horizons": {
                        "3": {"favorable_excursion_pct": 0.7, "adverse_excursion_pct": 0.2, "close_move_pct": 0.4},
                        "5": {"favorable_excursion_pct": 1.0, "adverse_excursion_pct": 0.3, "close_move_pct": 0.6},
                    },
                },
                {
                    "verdict": "late_or_weak",
                    "favorable_excursion_15m_pct": 0.3,
                    "adverse_excursion_15m_pct": 1.1,
                    "favorable_excursion_60m_pct": 0.8,
                    "adverse_excursion_60m_pct": 1.5,
                    "minutes_to_first_down_035pct": 9.0,
                    "minutes_to_first_up_035pct": 2.0,
                    "tp_hit_60m": False,
                    "sl_hit_60m": False,
                    "bar_horizons": {
                        "3": {"favorable_excursion_pct": 0.1, "adverse_excursion_pct": 0.8, "close_move_pct": -0.2},
                        "5": {"favorable_excursion_pct": 0.2, "adverse_excursion_pct": 1.0, "close_move_pct": -0.4},
                    },
                },
            ]
        )

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["verdict_counts"]["worked"], 1)
        self.assertEqual(summary["verdict_counts"]["late_or_weak"], 1)
        self.assertEqual(summary["tp_hit_rate_60m"], 0.5)
        self.assertEqual(summary["sl_hit_rate_60m"], 0.0)
        self.assertEqual(summary["avg_favorable_excursion_15m_pct"], 0.75)
        self.assertEqual(summary["avg_adverse_excursion_15m_pct"], 0.75)
        self.assertEqual(summary["avg_minutes_to_first_down_035pct"], 5.5)
        self.assertEqual(summary["avg_minutes_to_first_up_035pct"], 1.5)
        self.assertEqual(summary["bar_horizons"]["3"]["avg_favorable_excursion_pct"], 0.4)
        self.assertEqual(summary["bar_horizons"]["3"]["avg_adverse_excursion_pct"], 0.5)
        self.assertEqual(summary["bar_horizons"]["3"]["avg_close_move_pct"], 0.1)
        self.assertEqual(summary["bar_horizons"]["5"]["avg_favorable_excursion_pct"], 0.6)
        self.assertEqual(summary["bar_horizons"]["5"]["avg_adverse_excursion_pct"], 0.65)
        self.assertEqual(summary["bar_horizons"]["5"]["avg_close_move_pct"], 0.1)


if __name__ == "__main__":
    unittest.main()
