"""Result export helpers — CSV and JSON."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

# Fields included in CSV export
_CSV_COLUMNS = [
    "title",
    "genre",
    "selection_score",
    "signal",
    "decision",
    "npv_3y",
    "value_total",
    "jury_score",
    "delta_cal",
    "escalated",
    "phase",
    "w_ml",
    "w_llm",
]


def candidates_to_csv(candidates: list[dict[str, Any]]) -> str:
    """Convert pipeline results to CSV string."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for c in candidates:
        row = {k: c.get(k, "") for k in _CSV_COLUMNS}
        writer.writerow(row)
    return buf.getvalue()


def candidates_to_json(candidates: list[dict[str, Any]]) -> str:
    """Convert pipeline results to JSON string."""
    export = []
    for c in candidates:
        export.append({k: c.get(k) for k in _CSV_COLUMNS})
    return json.dumps(export, indent=2, default=str)
