"""Tests for monitoring pipeline node."""

from __future__ import annotations

from typing import Any

from llmart.pipeline.nodes.monitoring_node import monitoring_node
from llmart.pipeline.state import GraphState

_BASE_GAME: dict[str, Any] = {
    "genre": "Roguelike",
    "selection_score": 0.5,
    "escalated": False,
    "w_ml": 0.6,
}


def _make_state(candidates: list[dict[str, Any]]) -> GraphState:
    return GraphState(candidates=candidates, stage="value", errors=[])


def test_monitoring_no_candidates() -> None:
    state = _make_state([])
    result = monitoring_node(state)
    assert result["monitoring"]["status"] == "no_candidates"


def test_monitoring_basic_output() -> None:
    candidates = [
        {**_BASE_GAME, "game_id": "G001"},
        {**_BASE_GAME, "game_id": "G002", "genre": "RPG", "escalated": True},
    ]
    result = monitoring_node(_make_state(candidates))
    mon = result["monitoring"]
    assert "psi" in mon
    assert "psi_status" in mon
    assert "genre_distribution" in mon
    assert "escalation_rate" in mon
    assert mon["candidate_count"] == 2


def test_monitoring_escalation_rate() -> None:
    candidates = [{**_BASE_GAME, "game_id": f"G{i}", "escalated": i < 3} for i in range(10)]
    result = monitoring_node(_make_state(candidates))
    assert result["monitoring"]["escalation_rate"] == 0.3


def test_monitoring_does_not_modify_candidates() -> None:
    candidates = [{**_BASE_GAME, "game_id": "G001"}]
    result = monitoring_node(_make_state(candidates))
    assert "candidates" not in result


def test_compute_batch_ece_empty() -> None:
    """Empty jury_scores should return 0.0."""
    from llmart.pipeline.nodes.monitoring_node import _compute_batch_ece

    assert _compute_batch_ece([], []) == 0.0


def test_monitoring_ece_computation() -> None:
    """Monitoring should compute ECE from jury scores."""
    candidates = [
        {**_BASE_GAME, "game_id": "G001", "jury_score": 3.5, "signal": "GREEN"},
        {**_BASE_GAME, "game_id": "G002", "jury_score": 1.5, "signal": "RED"},
    ]
    result = monitoring_node(_make_state(candidates))
    assert "ece" in result["monitoring"]
    assert result["monitoring"]["ece"] >= 0.0


def test_monitoring_genre_distribution() -> None:
    candidates = [
        {**_BASE_GAME, "game_id": "G001"},
        {**_BASE_GAME, "game_id": "G002", "selection_score": 0.6},
        {**_BASE_GAME, "game_id": "G003", "genre": "RPG"},
    ]
    result = monitoring_node(_make_state(candidates))
    genre_dist = result["monitoring"]["genre_distribution"]
    assert abs(genre_dist["Roguelike"] - 2 / 3) < 0.01
    assert abs(genre_dist["RPG"] - 1 / 3) < 0.01
