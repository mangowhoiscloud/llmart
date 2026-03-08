"""Tests for llmart.audit."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from llmart.audit import AuditStore, RunRecord


@pytest.fixture
def tmp_audit(tmp_path: Path) -> AuditStore:
    db_path = tmp_path / "test_audit.db"
    store = AuditStore(db_path=db_path)
    yield store
    store.close()


class TestRunRecord:
    def test_create(self) -> None:
        r = RunRecord(run_id="test123", mode="mock", phase=0, top_k=30, input_count=100)
        assert r.run_id == "test123"
        assert r.finished_at is None

    def test_finish(self) -> None:
        r = RunRecord(run_id="test123")
        candidates = [
            {"signal": "GREEN"},
            {"signal": "GREEN"},
            {"signal": "YELLOW"},
            {"signal": "RED"},
        ]
        r.finish(candidates, ["warning1"])
        assert r.finished_at is not None
        assert r.duration_s is not None
        assert r.output_count == 4
        assert r.signal_green == 2
        assert r.signal_yellow == 1
        assert r.signal_red == 1
        assert r.errors == ["warning1"]


class TestAuditStore:
    def test_save_and_list(self, tmp_audit: AuditStore) -> None:
        r = RunRecord(run_id="run001", mode="mock", input_count=50)
        r.finish([{"signal": "GREEN"}])
        tmp_audit.save(r)

        runs = tmp_audit.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run001"
        assert runs[0]["signal_green"] == 1

    def test_get_run_with_result(self, tmp_audit: AuditStore) -> None:
        r = RunRecord(run_id="run002")
        r.finish([{"signal": "RED", "title": "Test Game"}])
        tmp_audit.save(r)

        full = tmp_audit.get_run("run002")
        assert full is not None
        assert len(full["result"]) == 1
        assert full["result"][0]["title"] == "Test Game"

    def test_get_nonexistent_run(self, tmp_audit: AuditStore) -> None:
        assert tmp_audit.get_run("nonexistent") is None

    def test_multiple_runs_ordered(self, tmp_audit: AuditStore) -> None:
        for i in range(5):
            r = RunRecord(run_id=f"run{i:03d}", started_at=time.time() + i)
            r.finish([])
            tmp_audit.save(r)

        runs = tmp_audit.list_runs(limit=3)
        assert len(runs) == 3
        # Most recent first
        assert runs[0]["run_id"] == "run004"

    def test_upsert(self, tmp_audit: AuditStore) -> None:
        r1 = RunRecord(run_id="dup001", mode="mock")
        r1.finish([{"signal": "GREEN"}])
        tmp_audit.save(r1)

        r2 = RunRecord(run_id="dup001", mode="real")
        r2.finish([{"signal": "RED"}])
        tmp_audit.save(r2)

        runs = tmp_audit.list_runs()
        assert len(runs) == 1
        assert runs[0]["mode"] == "real"
