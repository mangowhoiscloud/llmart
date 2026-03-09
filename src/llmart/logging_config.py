"""Structured logging for LLMART pipeline."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON structured log formatter with pipeline context."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Pipeline context fields
        for key in ("run_id", "stage", "game_count"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, default=str)


def get_pipeline_logger(
    name: str = "llmart",
    *,
    level: int = logging.INFO,
    json_output: bool = True,
) -> logging.Logger:
    """Create or retrieve a pipeline logger with structured formatting.

    Args:
        name: Logger name (default: "llmart").
        level: Logging level.
        json_output: If True, use JSON formatter; else plain text.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def generate_run_id() -> str:
    """Generate a unique run ID for pipeline execution tracking."""
    return uuid.uuid4().hex[:12]
