"""Tests for PipelineState schema."""

from __future__ import annotations

from llmart.pipeline.state import PipelineState


def test_default_state() -> None:
    state = PipelineState()
    assert state.candidates == []
    assert state.stage == "init"
    assert state.top_k == 30
    assert state.errors == []


def test_state_with_candidates() -> None:
    state = PipelineState(
        candidates=[{"game_id": "G001", "score": 0.9}],
        stage="ml_scoring",
        top_k=50,
    )
    assert len(state.candidates) == 1
    assert state.stage == "ml_scoring"
    assert state.top_k == 50
