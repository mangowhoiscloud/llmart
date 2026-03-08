"""Tests for T3 Human Review node."""

from __future__ import annotations

from typing import Any

import pytest

from llmart.models.escalation import EscalateReason
from llmart.pipeline.nodes.human_review import (
    _check_escalation,
    _decision_matrix,
    _dm_score,
    human_review_node,
)
from llmart.pipeline.state import GraphState


def test_decision_matrix_approve() -> None:
    # E(3.2+) + top40% -> APPROVE
    assert _decision_matrix(jury_score=3.2, ml_percentile=0.60) == "APPROVE"
    # >= 2.5 + top20% -> APPROVE
    assert _decision_matrix(jury_score=2.8, ml_percentile=0.85) == "APPROVE"


def test_decision_matrix_review() -> None:
    # E(3.2+) + top70% but not top40% -> REVIEW
    assert _decision_matrix(jury_score=3.2, ml_percentile=0.30) == "REVIEW"
    # >= 2.5 + not top20% but top70% -> REVIEW
    assert _decision_matrix(jury_score=2.5, ml_percentile=0.70) == "REVIEW"
    # jury 3.0 (< 3.2 threshold) + top40% -> REVIEW
    assert _decision_matrix(jury_score=3.0, ml_percentile=0.60) == "REVIEW"
    # >= 2.5 + top70% -> REVIEW
    assert _decision_matrix(jury_score=2.5, ml_percentile=0.50) == "REVIEW"


def test_decision_matrix_reject() -> None:
    assert _decision_matrix(jury_score=1.5, ml_percentile=0.30) == "REJECT"
    assert _decision_matrix(jury_score=2.5, ml_percentile=0.20) == "REJECT"
    # Below 2.5 jury -> REJECT regardless of ml
    assert _decision_matrix(jury_score=2.0, ml_percentile=0.90) == "REJECT"


@pytest.mark.parametrize(
    ("ml", "jury", "pass3", "p75", "p25", "expected"),
    [
        (0.95, 1.0, True, 100, 50, True),  # T1/T2 gap > 1.8 -> DISAGREE
        (0.5, 2.5, False, 100, 50, True),  # pass^3 disagree -> CONTRADICTION
        (0.50, 2.0, True, 400, 100, True),  # boundary (sel=0.5*0.5 + 0.5*0.5 = 0.50)
        (0.7, 3.5, True, 200, 100, False),  # no trigger (sel=0.79)
    ],
)
def test_escalation_triggers(
    ml: float, jury: float, pass3: bool, p75: float, p25: float, *, expected: bool
) -> None:
    game: dict[str, Any] = {
        "ml_percentile": ml,
        "jury_score": jury,
        "pass3_agree": pass3,
        "q1_p75": p75,
        "q1_p25": p25,
    }
    result = _check_escalation(game)
    assert result.should_escalate is expected


def test_escalation_boundary_trigger() -> None:
    """Approx selection score near 0.5 boundary should trigger escalation."""
    # sel = 0.5*0.5 + 0.5*(2.0/4.0) + 0 = 0.25 + 0.25 = 0.50
    game: dict[str, Any] = {
        "ml_percentile": 0.5,
        "jury_score": 2.0,
        "pass3_agree": True,
        "q1_p75": 100,
        "q1_p25": 50,
        "delta_cal": 0.0,
    }
    result = _check_escalation(game)
    assert result.should_escalate is True
    assert EscalateReason.BOUNDARY in result.reasons


def test_escalation_disagree_trigger() -> None:
    """T1/T2 gap > 1.8 triggers DISAGREE."""
    game: dict[str, Any] = {
        "ml_percentile": 0.95,
        "jury_score": 1.0,
        "pass3_agree": True,
        "q1_p75": 100,
        "q1_p25": 50,
    }
    result = _check_escalation(game)
    assert result.should_escalate is True
    assert EscalateReason.DISAGREE in result.reasons


def test_escalation_contradiction_trigger() -> None:
    """pass^3 disagreement triggers CONTRADICTION."""
    game: dict[str, Any] = {
        "ml_percentile": 0.7,
        "jury_score": 3.5,
        "pass3_agree": False,
    }
    result = _check_escalation(game)
    assert result.should_escalate is True
    assert EscalateReason.CONTRADICTION in result.reasons


def test_escalation_no_trigger() -> None:
    """No triggers fired returns EscalationResult with should_escalate=False."""
    game: dict[str, Any] = {
        "ml_percentile": 0.7,
        "jury_score": 3.5,
        "pass3_agree": True,
        "q1_p75": 200,
        "q1_p25": 100,
    }
    result = _check_escalation(game)
    assert result.should_escalate is False
    assert result.reasons == []


def test_escalation_result_guidance() -> None:
    """Guidance strings are populated for each reason."""
    game: dict[str, Any] = {
        "ml_percentile": 0.5,
        "jury_score": 2.0,
        "pass3_agree": True,
        "delta_cal": 0.0,
    }
    result = _check_escalation(game)
    assert len(result.guidance) == len(result.reasons)
    assert all(isinstance(g, str) for g in result.guidance)


def test_dm_score_range() -> None:
    """dm_score should be in [0, 1] for normalised inputs."""
    game: dict[str, Any] = {
        "jury_score": 3.0,
        "ml_percentile": 0.8,
        "release_year": 2024,
    }
    score = _dm_score(game)
    assert 0.0 <= score <= 1.0


def test_dm_score_uses_team_factor() -> None:
    """dm_score should incorporate team_factor from enrichment."""
    game_no_team: dict[str, Any] = {
        "jury_score": 3.0,
        "ml_percentile": 0.8,
        "release_year": 2024,
    }
    game_with_team: dict[str, Any] = {
        **game_no_team,
        "team_factor": 0.9,
    }
    score_default = _dm_score(game_no_team)
    score_team = _dm_score(game_with_team)
    # Higher team factor -> higher dm_score
    assert score_team > score_default


def test_top_k_limit() -> None:
    candidates: list[dict[str, Any]] = [
        {
            "game_id": f"G{i:03d}",
            "title": f"Game {i}",
            "jury_score": 3.5,  # H on 4-cat scale
            "ml_percentile": 0.8,
            "pass3_agree": True,
            "release_year": 2025,
            "q1_p75": 150,
            "q1_p25": 60,
        }
        for i in range(10)
    ]
    state = GraphState(candidates=candidates, top_k=3, stage="enrichment", errors=[])
    result = human_review_node(state)
    assert len(result["candidates"]) <= 3


def test_node_output_has_escalation_fields() -> None:
    """human_review_node output should include escalation_reasons and escalation_guidance."""
    candidates: list[dict[str, Any]] = [
        {
            "game_id": "G001",
            "title": "Test",
            "jury_score": 3.5,
            "ml_percentile": 0.8,
            "pass3_agree": True,
            "release_year": 2025,
        }
    ]
    state = GraphState(candidates=candidates, top_k=10, stage="enrichment", errors=[])
    result = human_review_node(state)
    for g in result["candidates"]:
        assert "escalated" in g
        assert "escalation_reasons" in g
        assert "escalation_guidance" in g


def test_decision_matrix_review_medium_jury_high_ml() -> None:
    """jury >= 2.5 and ml >= 0.60 should return REVIEW (not APPROVE until ml >= 0.80)."""
    assert _decision_matrix(jury_score=2.5, ml_percentile=0.60) == "REVIEW"
    assert _decision_matrix(jury_score=2.8, ml_percentile=0.70) == "REVIEW"


def test_escalation_multi_dim_trigger() -> None:
    """Two dimensions diverging >= 2.0 from mean should trigger MULTI_DIM."""
    # mean = (4+0+4+0+2)/5 = 2.0; deltas: 2.0, 2.0, 2.0, 2.0, 0.0
    game: dict[str, Any] = {
        "ml_percentile": 0.7,
        "jury_score": 3.5,
        "pass3_agree": True,
        "dim_scores": {
            "gameplay": 4.0,
            "innovation": 0.0,
            "monetization": 4.0,
            "polish": 0.0,
            "narrative": 2.0,
        },
    }
    result = _check_escalation(game)
    assert EscalateReason.MULTI_DIM in result.reasons


def test_escalation_extreme_dim_trigger() -> None:
    """Single dimension delta >= 2.5 from mean should trigger EXTREME_DIM."""
    # Mean = (4+4+4+4+0)/5 = 3.2, delta for narrative = |0.0 - 3.2| = 3.2 >= 2.5
    game: dict[str, Any] = {
        "ml_percentile": 0.7,
        "jury_score": 3.5,
        "pass3_agree": True,
        "dim_scores": {
            "gameplay": 4.0,
            "innovation": 4.0,
            "monetization": 4.0,
            "polish": 4.0,
            "narrative": 0.0,  # extreme delta
        },
    }
    result = _check_escalation(game)
    assert EscalateReason.EXTREME_DIM in result.reasons


def test_escalation_low_coverage_trigger() -> None:
    """Feature fill rate < 0.4 should trigger LOW_COVERAGE."""
    game: dict[str, Any] = {
        "ml_percentile": 0.7,
        "jury_score": 3.5,
        "pass3_agree": True,
        "ml_features": {
            "wilson": 0.0,
            "log_reviews": 0.0,
            "recency": 0.0,
            "tag_quality": 0.0,
            "price_norm": 0.5,
        },
    }
    result = _check_escalation(game)
    assert EscalateReason.LOW_COVERAGE in result.reasons


def test_human_review_error_handling() -> None:
    """Malformed candidate should produce error without crashing."""
    bad_game: dict[str, Any] = {
        "game_id": "BAD",
        "jury_score": "not_a_number",
        "ml_percentile": 0.5,
        "pass3_agree": True,
        "release_year": 2024,
    }
    state = GraphState(candidates=[bad_game], top_k=10, stage="enrichment", errors=[])
    result = human_review_node(state)
    # Should either handle gracefully or accumulate error
    assert "errors" in result


def test_check_escalation_with_explicit_config() -> None:
    """Passing explicit config should use it instead of DEFAULT_CONFIG."""
    from llmart.config import DEFAULT_CONFIG

    game: dict[str, Any] = {
        "ml_percentile": 0.7,
        "jury_score": 3.5,
        "pass3_agree": True,
    }
    result = _check_escalation(game, config=DEFAULT_CONFIG)
    assert isinstance(result.should_escalate, bool)


def test_escalation_single_dim_skips_multi() -> None:
    """Single dim_score should skip MULTI_DIM and EXTREME_DIM checks."""
    game: dict[str, Any] = {
        "ml_percentile": 0.7,
        "jury_score": 3.5,
        "pass3_agree": True,
        "dim_scores": {"gameplay": 3.0},
    }
    result = _check_escalation(game)
    assert EscalateReason.MULTI_DIM not in result.reasons
    assert EscalateReason.EXTREME_DIM not in result.reasons


def test_error_accumulation() -> None:
    state = GraphState(candidates=[], stage="enrichment", errors=[])
    result = human_review_node(state)
    assert "errors" in result
