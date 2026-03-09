"""Tests for pipeline stage timing instrumentation."""

from __future__ import annotations

import time
from typing import Any

from llmart.pipeline.graph import _timed_node
from llmart.pipeline.state import GraphState


def _dummy_node(state: GraphState) -> dict[str, Any]:
    """A trivial node that passes candidates through."""
    time.sleep(0.01)  # Small delay to ensure measurable timing
    return {
        "candidates": state.get("candidates", []),
        "stage": "dummy",
    }


class TestTimedNode:
    def test_records_timing(self) -> None:
        wrapped = _timed_node("dummy", _dummy_node)
        state = GraphState(candidates=[{"game_id": "G001"}], stage="init")
        result = wrapped(state)

        assert "stage_timings" in result
        assert "dummy" in result["stage_timings"]
        assert result["stage_timings"]["dummy"] >= 0.005  # at least some time

    def test_records_count(self) -> None:
        wrapped = _timed_node("dummy", _dummy_node)
        state = GraphState(candidates=[{"game_id": "G001"}, {"game_id": "G002"}], stage="init")
        result = wrapped(state)

        assert "stage_counts" in result
        assert result["stage_counts"]["dummy"] == 2

    def test_accumulates_across_stages(self) -> None:
        """Simulate two stages in sequence — timings and counts should accumulate."""
        wrapped1 = _timed_node("stage_a", _dummy_node)
        wrapped2 = _timed_node("stage_b", _dummy_node)

        state1 = GraphState(candidates=[{"game_id": "G001"}], stage="init")
        result1 = wrapped1(state1)

        # Build state2 with accumulated timing/counts from result1
        state2 = GraphState(
            candidates=result1["candidates"],
            stage="stage_a",
            stage_timings=result1["stage_timings"],
            stage_counts=result1["stage_counts"],
        )
        result2 = wrapped2(state2)

        assert "stage_a" in result2["stage_timings"]
        assert "stage_b" in result2["stage_timings"]
        assert result2["stage_counts"]["stage_a"] == 1
        assert result2["stage_counts"]["stage_b"] == 1

    def test_empty_candidates(self) -> None:
        wrapped = _timed_node("empty_test", _dummy_node)
        state = GraphState(candidates=[], stage="init")
        result = wrapped(state)

        assert result["stage_counts"]["empty_test"] == 0
        assert result["stage_timings"]["empty_test"] >= 0
