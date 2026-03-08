from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from trading.state.persistence import RuntimeStore


class RuntimePersistenceV2Tests(unittest.TestCase):
    def test_schema_version_and_maintenance_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)
            try:
                self.assertEqual(store.get_schema_version(), 2)

                now = time.time()
                store.put_idempotency_key("expired", expires_at=now - 1)
                store.put_idempotency_key("live", expires_at=now + 100)
                old_ts = now - (8 * 24 * 3600)
                store.upsert_inflight_intent(
                    intent_key="closed",
                    symbol="BTCUSDT",
                    action="LONG_ENTRY",
                    payload={},
                    status="completed",
                    created_at=old_ts,
                    updated_at=old_ts,
                )
                store.upsert_inflight_intent(
                    intent_key="open",
                    symbol="BTCUSDT",
                    action="LONG_ENTRY",
                    payload={},
                    status="pending_submission",
                    created_at=now,
                    updated_at=now,
                )

                summary = store.maintenance()
                self.assertGreaterEqual(summary.get("deleted_idempotency", 0), 1)
                self.assertGreaterEqual(summary.get("deleted_inflight", 0), 1)
                self.assertIn("live", store.load_live_idempotency_keys())
                self.assertEqual(len(store.load_open_inflight_intents()), 1)
            finally:
                store.close()
    def test_load_transition_and_decision_journals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "runtime.db")
            store = RuntimeStore(db_path)
            try:
                now = time.time()
                store.upsert_state_record("BTCUSDT", "FLAT", "init", now)
                store.append_transition("BTCUSDT", "FLAT", "PENDING_ENTRY_LONG", "intent", now + 1)
                store.append_order_decision(
                    symbol="BTCUSDT",
                    action="LONG_ENTRY",
                    state_before="FLAT",
                    risk_reason="approved",
                    exec_status="ACCEPTED",
                    exec_reason="ok",
                    order_id="oid1",
                    order_link_id="clid1",
                    ts=now + 2,
                    raw={"k": "v"},
                )

                transitions = store.load_state_transitions(limit=100)
                decisions = store.load_order_decisions(limit=100)

                self.assertEqual(len(transitions), 1)
                self.assertEqual(transitions[0].symbol, "BTCUSDT")
                self.assertEqual(transitions[0].current_state, "PENDING_ENTRY_LONG")

                self.assertEqual(len(decisions), 1)
                self.assertEqual(decisions[0].symbol, "BTCUSDT")
                self.assertEqual(decisions[0].exec_status, "ACCEPTED")
                self.assertEqual(decisions[0].raw.get("k"), "v")
            finally:
                store.close()
    def test_corrupted_db_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "runtime.db"
            db_file.write_bytes(b"not-a-sqlite-db")

            store = RuntimeStore(str(db_file))
            try:
                self.assertEqual(store.get_schema_version(), 2)
            finally:
                store.close()

            recovered = list(Path(tmpdir).glob("runtime.db.corrupt.*"))
            self.assertTrue(recovered)


if __name__ == "__main__":
    unittest.main()
