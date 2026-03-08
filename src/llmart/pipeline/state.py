"""Pipeline state schema."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    """Pydantic validation model for pipeline entry/exit boundaries.

    Used for input validation; the actual LangGraph state is GraphState (TypedDict).
    """

    candidates: list[dict[str, Any]] = Field(default_factory=list)
    stage: str = "init"
    top_k: int = 30
    errors: list[str] = Field(default_factory=list)


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer that merges two dicts (right overwrites left on key conflict)."""
    merged = dict(left)
    merged.update(right)
    return merged


class GraphState(TypedDict, total=False):
    """LangGraph-compatible state for the LLMART pipeline.

    Each node reads/writes ``candidates`` — a list of dicts whose fields
    grow progressively as they flow through the pipeline stages.

    ``candidates`` uses full-replace semantics (no reducer) — safe for
    the current linear-only pipeline. ``errors`` uses the ``add`` reducer
    to accumulate warnings/errors across nodes. ``monitoring`` uses
    ``_merge_dicts`` so prefilter's recall_at_99 survives value node's
    ndcg_at_30 write.
    """

    candidates: list[dict[str, Any]]  # full-replace (linear pipeline)
    stage: str
    top_k: int
    errors: Annotated[list[str], add]  # accumulates across nodes
    mode: str
    genre_params: dict[str, dict[str, Any]]
    total_input: int
    # Phase & monitoring (added in production upgrade)
    phase: int
    n_historical: int
    monitoring: Annotated[dict[str, Any], _merge_dicts]
    run_metadata: dict[str, Any]
    known_hits: list[str]
    training_data: list[dict[str, Any]]
    # Stage timing/counts — production instrumentation
    stage_timings: dict[str, float]
    stage_counts: dict[str, int]
