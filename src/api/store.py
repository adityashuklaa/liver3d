"""SQLite job and audit store.

Holds only anonymous job identifiers, model/config versions, runtime, result
disposition and errors. No patient identifiers are written here.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    requester TEXT,
    model_version TEXT,
    preprocessing_version TEXT,
    input_checksum TEXT,
    runtime_seconds REAL,
    warnings TEXT,
    error_category TEXT,
    error_message TEXT,
    trace_id TEXT,
    result TEXT,
    review_status TEXT DEFAULT 'pending',
    reviewer_role TEXT,
    review_note TEXT,
    reviewed_at REAL
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    job_id TEXT,
    actor TEXT,
    action TEXT NOT NULL,
    detail TEXT
);
"""


class JobStore:
    def __init__(self, path: str = "outputs/api/jobs.db"):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def create_job(self, job_id: str, requester: str, input_checksum: Optional[str]) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, status, created_at, updated_at, requester, input_checksum)"
                " VALUES (?, 'queued', ?, ?, ?, ?)",
                (job_id, now, now, requester, input_checksum),
            )
        self.audit(job_id, requester, "job_created", {"input_checksum": input_checksum})

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        for key in ("warnings", "result"):
            if key in fields and not isinstance(fields[key], (str, type(None))):
                fields[key] = json.dumps(fields[key])
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ?", (*fields.values(), job_id))

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        for key in ("warnings", "result"):
            if record.get(key):
                try:
                    record[key] = json.loads(record[key])
                except (TypeError, ValueError):
                    pass
        return record

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id, status, created_at, model_version, review_status FROM jobs"
                " ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_review(self, job_id: str, status: str, reviewer_role: str, note: str = "") -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET review_status = ?, reviewer_role = ?, review_note = ?,"
                " reviewed_at = ?, updated_at = ? WHERE job_id = ?",
                (status, reviewer_role, note, time.time(), time.time(), job_id),
            )
        self.audit(job_id, reviewer_role, "review_recorded", {"review_status": status})

    def audit(self, job_id: Optional[str], actor: str, action: str, detail: Any = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO audit (ts, job_id, actor, action, detail) VALUES (?, ?, ?, ?, ?)",
                (time.time(), job_id, actor, action, json.dumps(detail, default=str) if detail else None),
            )

    def audit_trail(self, job_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, actor, action, detail FROM audit WHERE job_id = ? ORDER BY id", (job_id,)
            ).fetchall()
        return [dict(r) for r in rows]
