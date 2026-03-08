"""Audit trail: SQLite-based pipeline run history."""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path.home() / ".llmart" / "audit.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    duration_s    REAL,
    mode          TEXT,
    phase         INTEGER,
    top_k         INTEGER,
    input_count   INTEGER,
    output_count  INTEGER,
    signal_green  INTEGER DEFAULT 0,
    signal_yellow INTEGER DEFAULT 0,
    signal_red    INTEGER DEFAULT 0,
    config_json   TEXT,
    result_gz     BLOB,
    errors_json   TEXT
)
"""


@dataclass
class RunRecord:
    """Captures a single pipeline execution for audit."""

    run_id: str
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    duration_s: float | None = None
    mode: str = "mock"
    phase: int = 0
    top_k: int = 30
    input_count: int = 0
    output_count: int = 0
    signal_green: int = 0
    signal_yellow: int = 0
    signal_red: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    result: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def finish(self, candidates: list[dict[str, Any]], errors: list[str] | None = None) -> None:
        """Mark the run as finished and compute signal distribution."""
        self.finished_at = time.time()
        self.duration_s = round(self.finished_at - self.started_at, 3)
        self.result = candidates
        self.output_count = len(candidates)
        if errors:
            self.errors = errors
        for c in candidates:
            sig = c.get("signal", "")
            if sig == "GREEN":
                self.signal_green += 1
            elif sig == "YELLOW":
                self.signal_yellow += 1
            elif sig == "RED":
                self.signal_red += 1


class AuditStore:
    """SQLite-backed audit store for pipeline runs."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def save(self, record: RunRecord) -> None:
        """Persist a RunRecord to SQLite."""
        result_bytes = gzip.compress(json.dumps(record.result, default=str).encode())
        self._conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, started_at, finished_at, duration_s, mode, phase, top_k,
                input_count, output_count, signal_green, signal_yellow, signal_red,
                config_json, result_gz, errors_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.run_id,
                record.started_at,
                record.finished_at,
                record.duration_s,
                record.mode,
                record.phase,
                record.top_k,
                record.input_count,
                record.output_count,
                record.signal_green,
                record.signal_yellow,
                record.signal_red,
                json.dumps(record.config, default=str),
                result_bytes,
                json.dumps(record.errors),
            ),
        )
        self._conn.commit()

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retrieve recent runs (without full result data)."""
        cursor = self._conn.execute(
            """SELECT run_id, started_at, finished_at, duration_s, mode, phase,
                      top_k, input_count, output_count,
                      signal_green, signal_yellow, signal_red, errors_json
               FROM runs ORDER BY started_at DESC LIMIT ?""",
            (limit,),
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row, strict=True)) for row in cursor.fetchall()]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve a full run record including decompressed results."""
        cursor = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cursor.description]
        record = dict(zip(cols, row, strict=True))
        # Decompress result
        result_gz = record.pop("result_gz", None)
        if isinstance(result_gz, bytes):
            record["result"] = json.loads(gzip.decompress(result_gz))
        else:
            record["result"] = []
        return record

    def close(self) -> None:
        self._conn.close()
