from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "observation" / "analyze_recent_exit_quality.py"
SPEC = importlib.util.spec_from_file_location("analyze_recent_exit_quality", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ObservationExitQualityV2Tests(unittest.TestCase):
    def test_window_exit_metrics_short_distinguishes_further_drop_vs_rebound(self):
        index = pd.date_range("2026-04-01T00:00:00Z", periods=4, freq="1min")
        window = pd.DataFrame(
            {
                "high": [100.10, 100.55, 100.20, 100.30],
                "low": [99.92, 99.85, 99.40, 99.70],
                "close": [99.98, 100.20, 99.55, 99.72],
            },
            index=index,
        )

        metrics = MODULE._window_exit_metrics_short(window, exit_price=100.0)

        self.assertEqual(metrics["bars_observed"], 4.0)
        self.assertEqual(metrics["further_favorable_pct"], 0.6)
        self.assertEqual(metrics["rebound_pct"], 0.55)
        self.assertEqual(metrics["close_move_pct"], 0.28)

    def test_classify_short_exit_quality_flags_too_early(self):
        verdict = MODULE._classify_short_exit_quality(
            realized_pnl=2.4,
            stopped_out=False,
            further_15=0.6,
            rebound_15=0.2,
            further_60=1.3,
            rebound_60=0.4,
            bars_to_further_down=2,
            bars_to_rebound=9,
        )

        self.assertEqual(verdict, "too_early")

    def test_classify_short_exit_quality_flags_timely(self):
        verdict = MODULE._classify_short_exit_quality(
            realized_pnl=1.1,
            stopped_out=False,
            further_15=0.2,
            rebound_15=0.5,
            further_60=0.45,
            rebound_60=0.8,
            bars_to_further_down=6,
            bars_to_rebound=2,
        )

        self.assertEqual(verdict, "timely")

    def test_summarize_results_reports_verdicts_and_exit_reasons(self):
        summary = MODULE._summarize_results(
            [
                {
                    "verdict": "timely",
                    "exit_type": "acceptance_reclaim",
                    "managed_exit_reason": "managed_exit_acceptance_reclaim",
                    "realized_pnl": 1.8,
                    "stopped_out": False,
                    "further_favorable_15m_pct": 0.2,
                    "rebound_15m_pct": 0.6,
                    "further_favorable_60m_pct": 0.4,
                    "rebound_60m_pct": 1.0,
                    "minutes_to_further_down_035pct": 7.0,
                    "minutes_to_rebound_035pct": 2.0,
                    "bar_horizons": {
                        "3": {"further_favorable_pct": 0.1, "rebound_pct": 0.4, "close_move_pct": -0.1},
                        "5": {"further_favorable_pct": 0.2, "rebound_pct": 0.6, "close_move_pct": -0.3},
                    },
                },
                {
                    "verdict": "too_early",
                    "exit_type": "target_zone_bounce",
                    "managed_exit_reason": "managed_exit_target_zone_bounce",
                    "realized_pnl": 0.9,
                    "stopped_out": False,
                    "further_favorable_15m_pct": 0.6,
                    "rebound_15m_pct": 0.1,
                    "further_favorable_60m_pct": 1.2,
                    "rebound_60m_pct": 0.3,
                    "minutes_to_further_down_035pct": 2.0,
                    "minutes_to_rebound_035pct": 10.0,
                    "bar_horizons": {
                        "3": {"further_favorable_pct": 0.4, "rebound_pct": 0.1, "close_move_pct": 0.3},
                        "5": {"further_favorable_pct": 0.7, "rebound_pct": 0.2, "close_move_pct": 0.5},
                    },
                },
            ]
        )

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["verdict_counts"]["timely"], 1)
        self.assertEqual(summary["verdict_counts"]["too_early"], 1)
        self.assertEqual(summary["exit_type_counts"]["acceptance_reclaim"], 1)
        self.assertEqual(summary["managed_exit_reason_counts"]["managed_exit_acceptance_reclaim"], 1)
        self.assertEqual(summary["avg_realized_pnl"], 1.35)
        self.assertEqual(summary["stopped_out_rate"], 0.0)
        self.assertEqual(summary["avg_further_favorable_60m_pct"], 0.8)
        self.assertEqual(summary["avg_rebound_60m_pct"], 0.65)
        self.assertEqual(summary["avg_minutes_to_further_down_035pct"], 4.5)
        self.assertEqual(summary["avg_minutes_to_rebound_035pct"], 6.0)
        self.assertEqual(summary["bar_horizons"]["3"]["avg_further_favorable_pct"], 0.25)
        self.assertEqual(summary["bar_horizons"]["3"]["avg_rebound_pct"], 0.25)
        self.assertEqual(summary["bar_horizons"]["3"]["avg_close_move_pct"], 0.1)

    def test_load_db_rows_extracts_exit_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "v2_demo_runtime_main.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                """
                CREATE TABLE order_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    state_before TEXT NOT NULL,
                    risk_reason TEXT NOT NULL,
                    exec_status TEXT NOT NULL,
                    exec_reason TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    order_link_id TEXT NOT NULL,
                    ts REAL NOT NULL,
                    raw_json TEXT NOT NULL
                )
                """
            )
            payload = {
                "exit_price": 98.5,
                "entry_price": 100.0,
                "tp_price": 96.0,
                "sl_price": 101.0,
                "realized_pnl": 1.5,
                "exit_type": "acceptance_reclaim",
                "managed_exit_reason": "managed_exit_acceptance_reclaim",
                "execution_context": {"realized_pnl": 1.5},
            }
            conn.execute(
                """
                INSERT INTO order_decisions(
                    symbol, action, state_before, risk_reason, exec_status, exec_reason,
                    order_id, order_link_id, ts, raw_json
                )
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    "BTCUSDT",
                    "EXIT_SHORT",
                    "SHORT",
                    "approved",
                    "FILLED",
                    "exit_filled",
                    "oid-1",
                    "clid-1",
                    1_000.0,
                    json.dumps(payload),
                ),
            )
            conn.commit()
            conn.close()

            rows = MODULE._load_db_rows(db_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].profile, "main")
            self.assertEqual(rows[0].symbol, "BTCUSDT")
            self.assertEqual(rows[0].exit_type, "acceptance_reclaim")
            self.assertEqual(rows[0].managed_exit_reason, "managed_exit_acceptance_reclaim")
            self.assertAlmostEqual(float(rows[0].exit_price or 0.0), 98.5, places=6)
            self.assertAlmostEqual(float(rows[0].realized_pnl), 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
