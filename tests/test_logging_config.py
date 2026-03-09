"""Tests for llmart.logging_config."""

from __future__ import annotations

import json
import logging

from llmart.logging_config import JSONFormatter, generate_run_id, get_pipeline_logger


class TestJSONFormatter:
    def test_basic_format(self) -> None:
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="llmart",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=None,
            exc_info=None,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["message"] == "test message"
        assert data["logger"] == "llmart"

    def test_pipeline_context_fields(self) -> None:
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="llmart",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="stage done",
            args=None,
            exc_info=None,
        )
        record.run_id = "abc123"  # type: ignore[attr-defined]
        record.stage = "prefilter"  # type: ignore[attr-defined]
        record.game_count = 42  # type: ignore[attr-defined]
        output = fmt.format(record)
        data = json.loads(output)
        assert data["run_id"] == "abc123"
        assert data["stage"] == "prefilter"
        assert data["game_count"] == 42

    def test_exception_in_log(self) -> None:
        fmt = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="llmart",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="error occurred",
                args=None,
                exc_info=sys.exc_info(),
            )
        output = fmt.format(record)
        data = json.loads(output)
        assert "boom" in data["exception"]


class TestGetPipelineLogger:
    def test_returns_logger(self) -> None:
        logger = get_pipeline_logger("test_llmart_logger_unique")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_llmart_logger_unique"

    def test_idempotent_handler(self) -> None:
        name = "test_llmart_idempotent"
        logger1 = get_pipeline_logger(name)
        handler_count = len(logger1.handlers)
        logger2 = get_pipeline_logger(name)
        assert len(logger2.handlers) == handler_count

    def test_plain_text_mode(self) -> None:
        logger = get_pipeline_logger("test_llmart_plain", json_output=False)
        assert logger.handlers
        assert not isinstance(logger.handlers[0].formatter, JSONFormatter)


class TestGenerateRunId:
    def test_length(self) -> None:
        rid = generate_run_id()
        assert len(rid) == 12

    def test_unique(self) -> None:
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100
