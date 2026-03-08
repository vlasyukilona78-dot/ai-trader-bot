from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PersistedStateRow:
    symbol: str
    state: str
    reason: str
    updated_at: float


@dataclass(frozen=True)
class PersistedIntentRow:
    intent_key: str
    symbol: str
    action: str
    payload: dict
    status: str
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class PersistedRiskRow:
    session_day: str
    realized_pnl: float
    consecutive_losses: int
    cooldown_until_ts: float
    updated_at: float

@dataclass(frozen=True)
class PersistedTransitionRow:
    id: int
    symbol: str
    previous_state: str
    current_state: str
    reason: str
    ts: float


@dataclass(frozen=True)
class PersistedDecisionRow:
    id: int
    symbol: str
    action: str
    state_before: str
    risk_reason: str
    exec_status: str
    exec_reason: str
    order_id: str
    order_link_id: str
    ts: float
    raw: dict

class RuntimeStore:
    """SQLite-backed runtime persistence for restart-safe V2 execution."""

    def __init__(self, db_path: str = "data/runtime/v2_runtime.db"):
        self.db_path = str(db_path)
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = self._open_connection_with_recovery(db_file)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    @staticmethod
    def now_ts() -> float:
        return datetime.now(timezone.utc).timestamp()

    @staticmethod
    def utc_day(ts: float | None = None) -> str:
        value = datetime.fromtimestamp(ts or RuntimeStore.now_ts(), tz=timezone.utc).date()
        return value.isoformat()

    @staticmethod
    def _connection_integrity_ok(conn: sqlite3.Connection) -> bool:
        try:
            row = conn.execute("PRAGMA quick_check").fetchone()
        except sqlite3.DatabaseError:
            return False
        if row is None:
            return False
        value = str(row[0]).strip().lower()
        return value == "ok"

    def _open_connection_with_recovery(self, db_file: Path) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(str(db_file), check_same_thread=False)
        except sqlite3.DatabaseError:
            conn = None

        if conn is not None and self._connection_integrity_ok(conn):
            return conn

        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

        ts = int(time.time())
        corrupt_path = db_file.with_suffix(db_file.suffix + f".corrupt.{ts}")
        try:
            os.replace(str(db_file), str(corrupt_path))
        except OSError:
            pass
        return sqlite3.connect(str(db_file), check_same_thread=False)

    def close(self):
        with self._lock:
            self._conn.close()

    def _table_exists(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def _init_schema(self):
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS runtime_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS state_records (
                    symbol TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS state_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    previous_state TEXT NOT NULL,
                    current_state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    ts REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inflight_intents (
                    intent_key TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    idempotency_key TEXT PRIMARY KEY,
                    expires_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS risk_session (
                    session_day TEXT PRIMARY KEY,
                    realized_pnl REAL NOT NULL,
                    consecutive_losses INTEGER NOT NULL,
                    cooldown_until_ts REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS order_decisions (
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
                );
                """
            )
            self._conn.commit()

            self._migrate_locked()
            self._conn.commit()

    def _read_schema_version_locked(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM runtime_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(str(row["value"]))
        except (TypeError, ValueError):
            return 0

    def _write_schema_version_locked(self, version: int):
        self._conn.execute(
            """
            INSERT INTO runtime_meta(key, value)
            VALUES('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (str(int(version)),),
        )

    def _migrate_locked(self):
        version = self._read_schema_version_locked()
        if version <= 0:
            self._write_schema_version_locked(1)
            version = 1

        if version == 1:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_state_transitions_ts ON state_transitions(ts)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_order_decisions_ts ON order_decisions(ts)"
            )
            self._write_schema_version_locked(2)
            version = 2

        if version < SCHEMA_VERSION:
            self._write_schema_version_locked(SCHEMA_VERSION)

    def get_schema_version(self) -> int:
        with self._lock:
            return self._read_schema_version_locked()

    def upsert_state_record(self, symbol: str, state: str, reason: str, updated_at: float):
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO state_records(symbol, state, reason, updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(symbol) DO UPDATE SET
                    state=excluded.state,
                    reason=excluded.reason,
                    updated_at=excluded.updated_at
                """,
                (symbol, state, reason, float(updated_at)),
            )
            self._conn.commit()

    def append_transition(
        self,
        symbol: str,
        previous_state: str,
        current_state: str,
        reason: str,
        ts: float,
    ):
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO state_transitions(symbol, previous_state, current_state, reason, ts)
                VALUES(?,?,?,?,?)
                """,
                (symbol, previous_state, current_state, reason, float(ts)),
            )
            self._conn.commit()

    def load_state_records(self) -> list[PersistedStateRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT symbol, state, reason, updated_at FROM state_records"
            ).fetchall()
        return [
            PersistedStateRow(
                symbol=str(row["symbol"]),
                state=str(row["state"]),
                reason=str(row["reason"]),
                updated_at=float(row["updated_at"]),
            )
            for row in rows
        ]

    def upsert_inflight_intent(
        self,
        *,
        intent_key: str,
        symbol: str,
        action: str,
        payload: dict,
        status: str,
        created_at: float | None = None,
        updated_at: float | None = None,
    ):
        now = self.now_ts()
        created = float(created_at if created_at is not None else now)
        updated = float(updated_at if updated_at is not None else now)
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO inflight_intents(intent_key, symbol, action, payload_json, status, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(intent_key) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (intent_key, symbol, action, payload_json, status, created, updated),
            )
            self._conn.commit()

    def update_inflight_status(self, intent_key: str, status: str, payload: dict | None = None):
        now = self.now_ts()
        with self._lock:
            if payload is None:
                self._conn.execute(
                    """
                    UPDATE inflight_intents
                    SET status=?, updated_at=?
                    WHERE intent_key=?
                    """,
                    (status, now, intent_key),
                )
            else:
                payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                self._conn.execute(
                    """
                    UPDATE inflight_intents
                    SET payload_json=?, status=?, updated_at=?
                    WHERE intent_key=?
                    """,
                    (payload_json, status, now, intent_key),
                )
            self._conn.commit()

    def load_open_inflight_intents(self) -> list[PersistedIntentRow]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT intent_key, symbol, action, payload_json, status, created_at, updated_at
                FROM inflight_intents
                WHERE status IN ('pending_submission', 'pending_fill', 'partial_fill', 'naked_exposure')
                ORDER BY created_at ASC
                """
            ).fetchall()

        out: list[PersistedIntentRow] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
                if not isinstance(payload, dict):
                    payload = {}
            except Exception:
                payload = {}
            out.append(
                PersistedIntentRow(
                    intent_key=str(row["intent_key"]),
                    symbol=str(row["symbol"]),
                    action=str(row["action"]),
                    payload=payload,
                    status=str(row["status"]),
                    created_at=float(row["created_at"]),
                    updated_at=float(row["updated_at"]),
                )
            )
        return out

    def put_idempotency_key(self, key: str, expires_at: float):
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO idempotency_keys(idempotency_key, expires_at)
                VALUES(?,?)
                ON CONFLICT(idempotency_key) DO UPDATE SET
                    expires_at=excluded.expires_at
                """,
                (key, float(expires_at)),
            )
            self._conn.commit()

    def cleanup_idempotency_keys(self, now_ts: float | None = None) -> int:
        now = float(now_ts if now_ts is not None else self.now_ts())
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM idempotency_keys WHERE expires_at <= ?",
                (now,),
            )
            deleted = int(cur.rowcount if cur.rowcount is not None else 0)
            self._conn.commit()
        return max(0, deleted)

    def clear_idempotency_keys(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM idempotency_keys")
            deleted = int(cur.rowcount if cur.rowcount is not None else 0)
            self._conn.commit()
        return max(0, deleted)

    def clear_inflight_intents(self, symbol: str | None = None) -> int:
        with self._lock:
            if symbol is None:
                cur = self._conn.execute("DELETE FROM inflight_intents")
            else:
                cur = self._conn.execute(
                    "DELETE FROM inflight_intents WHERE symbol=?",
                    (str(symbol).replace("/", "").upper(),),
                )
            deleted = int(cur.rowcount if cur.rowcount is not None else 0)
            self._conn.commit()
        return max(0, deleted)

    def load_live_idempotency_keys(self, now_ts: float | None = None) -> dict[str, float]:
        now = float(now_ts if now_ts is not None else self.now_ts())
        self.cleanup_idempotency_keys(now)
        with self._lock:
            rows = self._conn.execute(
                "SELECT idempotency_key, expires_at FROM idempotency_keys WHERE expires_at > ?",
                (now,),
            ).fetchall()
        return {str(row["idempotency_key"]): float(row["expires_at"]) for row in rows}

    def save_risk_row(self, row: PersistedRiskRow):
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO risk_session(session_day, realized_pnl, consecutive_losses, cooldown_until_ts, updated_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(session_day) DO UPDATE SET
                    realized_pnl=excluded.realized_pnl,
                    consecutive_losses=excluded.consecutive_losses,
                    cooldown_until_ts=excluded.cooldown_until_ts,
                    updated_at=excluded.updated_at
                """,
                (
                    row.session_day,
                    float(row.realized_pnl),
                    int(row.consecutive_losses),
                    float(row.cooldown_until_ts),
                    float(row.updated_at),
                ),
            )
            self._conn.commit()

    def load_risk_row(self, session_day: str) -> PersistedRiskRow | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT session_day, realized_pnl, consecutive_losses, cooldown_until_ts, updated_at
                FROM risk_session WHERE session_day=?
                """,
                (session_day,),
            ).fetchone()
        if row is None:
            return None
        return PersistedRiskRow(
            session_day=str(row["session_day"]),
            realized_pnl=float(row["realized_pnl"]),
            consecutive_losses=int(row["consecutive_losses"]),
            cooldown_until_ts=float(row["cooldown_until_ts"]),
            updated_at=float(row["updated_at"]),
        )

    def append_order_decision(
        self,
        *,
        symbol: str,
        action: str,
        state_before: str,
        risk_reason: str,
        exec_status: str,
        exec_reason: str,
        order_id: str,
        order_link_id: str,
        ts: float,
        raw: dict | None,
    ):
        payload = raw if isinstance(raw, dict) else {}
        raw_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO order_decisions(
                    symbol, action, state_before, risk_reason, exec_status, exec_reason,
                    order_id, order_link_id, ts, raw_json
                )
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    symbol,
                    action,
                    state_before,
                    risk_reason,
                    exec_status,
                    exec_reason,
                    order_id,
                    order_link_id,
                    float(ts),
                    raw_json,
                ),
            )
            self._conn.commit()
    def load_state_transitions(self, limit: int = 50000) -> list[PersistedTransitionRow]:
        safe_limit = max(1, int(limit))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, symbol, previous_state, current_state, reason, ts
                FROM state_transitions
                ORDER BY id ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [
            PersistedTransitionRow(
                id=int(row["id"]),
                symbol=str(row["symbol"]),
                previous_state=str(row["previous_state"]),
                current_state=str(row["current_state"]),
                reason=str(row["reason"]),
                ts=float(row["ts"]),
            )
            for row in rows
        ]

    def load_order_decisions(self, limit: int = 50000) -> list[PersistedDecisionRow]:
        safe_limit = max(1, int(limit))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    id, symbol, action, state_before, risk_reason, exec_status, exec_reason,
                    order_id, order_link_id, ts, raw_json
                FROM order_decisions
                ORDER BY id ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        out: list[PersistedDecisionRow] = []
        for row in rows:
            try:
                raw = json.loads(str(row["raw_json"]))
                if not isinstance(raw, dict):
                    raw = {}
            except Exception:
                raw = {}
            out.append(
                PersistedDecisionRow(
                    id=int(row["id"]),
                    symbol=str(row["symbol"]),
                    action=str(row["action"]),
                    state_before=str(row["state_before"]),
                    risk_reason=str(row["risk_reason"]),
                    exec_status=str(row["exec_status"]),
                    exec_reason=str(row["exec_reason"]),
                    order_id=str(row["order_id"]),
                    order_link_id=str(row["order_link_id"]),
                    ts=float(row["ts"]),
                    raw=raw,
                )
            )
        return out

    def compact_journals(
        self,
        *,
        max_transition_rows: int = 100000,
        max_decision_rows: int = 100000,
        keep_recent: int = 20000,
    ) -> dict[str, int]:
        keep_recent = max(1000, int(keep_recent))
        max_transition_rows = max(keep_recent, int(max_transition_rows))
        max_decision_rows = max(keep_recent, int(max_decision_rows))

        deleted_transitions = 0
        deleted_decisions = 0

        with self._lock:
            transition_count = int(self._conn.execute("SELECT COUNT(*) FROM state_transitions").fetchone()[0])
            if transition_count > max_transition_rows:
                cutoff = self._conn.execute(
                    "SELECT id FROM state_transitions ORDER BY id DESC LIMIT 1 OFFSET ?",
                    (keep_recent - 1,),
                ).fetchone()
                if cutoff is not None:
                    cur = self._conn.execute("DELETE FROM state_transitions WHERE id < ?", (int(cutoff[0]),))
                    deleted_transitions = int(cur.rowcount if cur.rowcount is not None else 0)

            decision_count = int(self._conn.execute("SELECT COUNT(*) FROM order_decisions").fetchone()[0])
            if decision_count > max_decision_rows:
                cutoff = self._conn.execute(
                    "SELECT id FROM order_decisions ORDER BY id DESC LIMIT 1 OFFSET ?",
                    (keep_recent - 1,),
                ).fetchone()
                if cutoff is not None:
                    cur = self._conn.execute("DELETE FROM order_decisions WHERE id < ?", (int(cutoff[0]),))
                    deleted_decisions = int(cur.rowcount if cur.rowcount is not None else 0)

            self._conn.commit()

        return {
            "deleted_transitions": max(0, deleted_transitions),
            "deleted_decisions": max(0, deleted_decisions),
        }

    def cleanup_closed_inflight(self, max_age_sec: int = 7 * 24 * 3600) -> int:
        cutoff = self.now_ts() - max(3600, int(max_age_sec))
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM inflight_intents
                WHERE updated_at < ?
                  AND status NOT IN ('pending_submission', 'pending_fill', 'partial_fill', 'naked_exposure')
                """,
                (cutoff,),
            )
            deleted = int(cur.rowcount if cur.rowcount is not None else 0)
            self._conn.commit()
        return max(0, deleted)

    def maintenance(self) -> dict:
        deleted_idem = self.cleanup_idempotency_keys()
        compacted = self.compact_journals()
        deleted_inflight = self.cleanup_closed_inflight()
        return {
            "deleted_idempotency": deleted_idem,
            "deleted_inflight": deleted_inflight,
            **compacted,
            "schema_version": self.get_schema_version(),
        }


