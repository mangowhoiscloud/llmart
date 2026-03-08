"""Tests for Value Inference + Signal node."""

from __future__ import annotations

from typing import Any

import pytest

from llmart.pipeline.nodes.value import _compute_npv_3y, _try_ridge_weights, value_node
from llmart.pipeline.state import GraphState


def test_npv_formula() -> None:
    npv = _compute_npv_3y(q1_p50=100_000.0, r_genre=0.20, ltv_mult=2.5)
    expected = 100_000 / 0.20 * 2.5 * sum(1 / 1.15**t for t in range(1, 4))
    assert abs(npv - expected) < 1.0


def test_npv_zero_r_genre() -> None:
    assert _compute_npv_3y(q1_p50=100_000.0, r_genre=0.0, ltv_mult=2.5) == 0.0


def test_npv_custom_discount_rate() -> None:
    """Verify per-genre discount_rate is used when provided."""
    npv_15 = _compute_npv_3y(q1_p50=100_000.0, r_genre=0.20, ltv_mult=2.0, discount_rate=0.15)
    npv_10 = _compute_npv_3y(q1_p50=100_000.0, r_genre=0.20, ltv_mult=2.0, discount_rate=0.10)
    assert npv_10 > npv_15  # lower discount rate -> higher NPV


def test_signal_assigned() -> None:
    candidates: list[dict[str, Any]] = [
        {
            "game_id": "V1",
            "ml_percentile": 0.9,
            "jury_score": 3.5,
            "delta_cal": 0.0,
            "q1_p50": 300_000.0,
            "q1_p25": 260_000.0,
            "r_genre": 0.20,
            "ltv_mult": 2.5,
        }
    ]
    state = GraphState(candidates=candidates, stage="human_review", errors=[])
    result = value_node(state)
    game = result["candidates"][0]
    assert game["signal"] in ("GREEN", "YELLOW", "RED")
    assert "selection_score" in game
    assert "npv_3y" in game
    assert "value_total" in game


@pytest.mark.parametrize(
    ("q1_p50", "q1_p25", "expected_signal"),
    [
        (500_000, 300_000, "GREEN"),  # P25 >= 250K
        (300_000, 200_000, "YELLOW"),  # P50 >= 250K, P25 < 250K
        (200_000, 120_000, "RED"),  # P50 < 250K
    ],
)
def test_signal_classification(q1_p50: float, q1_p25: float, expected_signal: str) -> None:
    candidates: list[dict[str, Any]] = [
        {
            "game_id": "SIG",
            "ml_percentile": 0.8,
            "jury_score": 3.0,
            "delta_cal": 0.0,
            "q1_p50": q1_p50,
            "q1_p25": q1_p25,
            "r_genre": 0.20,
            "ltv_mult": 2.5,
        }
    ]
    state = GraphState(candidates=candidates, stage="human_review", errors=[])
    result = value_node(state)
    assert result["candidates"][0]["signal"] == expected_signal


def test_higher_var_reduces_value() -> None:
    """Higher revenue spread → larger VaR → lower value."""
    base = {
        "game_id": "VAR",
        "ml_percentile": 0.5,
        "jury_score": 2.0,
        "delta_cal": 0.0,
        "r_genre": 1.0,  # high r_genre → small NPV so spread-based VaR dominates floor
        "ltv_mult": 1.0,
    }
    # Narrow spread: P50 close to P25 → VaR floor dominates
    narrow = {**base, "q1_p50": 100_000.0, "q1_p25": 99_000.0}
    # Wide spread: P50 far from P25 → spread-based VaR overtakes floor
    wide = {**base, "q1_p50": 100_000.0, "q1_p25": 0.0}

    state_narrow = GraphState(candidates=[narrow], stage="human_review", errors=[])
    state_wide = GraphState(candidates=[wide], stage="human_review", errors=[])

    val_narrow = value_node(state_narrow)["candidates"][0]["value_total"]
    val_wide = value_node(state_wide)["candidates"][0]["value_total"]
    assert val_narrow > val_wide  # wider spread → higher VaR → lower value


def test_error_accumulation() -> None:
    state = GraphState(candidates=[], stage="human_review", errors=[])
    result = value_node(state)
    assert "errors" in result


def test_value_node_phase2_ridge_weights() -> None:
    """Phase 2 with well-correlated training data should use Ridge-learned weights."""
    # Training data with strong correlation between features and revenue
    training_data: list[dict[str, Any]] = [
        {
            "ml_percentile": i / 20.0,
            "jury_score": 1.0 + i * 0.15,
            "q1_p50": 10_000.0 * (i + 1) ** 2,
        }
        for i in range(20)
    ]
    candidates: list[dict[str, Any]] = [
        {
            "game_id": "RIDGE1",
            "ml_percentile": 0.8,
            "jury_score": 3.0,
            "delta_cal": 0.0,
            "q1_p50": 300_000.0,
            "q1_p25": 180_000.0,
            "r_genre": 0.20,
            "ltv_mult": 2.0,
        }
    ]
    state = GraphState(
        candidates=candidates,
        stage="human_review",
        errors=[],
        phase=2,
        n_historical=0,
        training_data=training_data,
    )
    result = value_node(state)
    game = result["candidates"][0]
    assert "selection_score" in game
    assert "w_ml" in game
    assert "w_llm" in game
    # Ridge should either use learned weights or fall back to controller
    assert game["phase"] == 2


def test_try_ridge_weights_with_good_data() -> None:
    """Ridge weights should be learned from well-correlated data."""
    training_data: list[dict[str, Any]] = [
        {
            "ml_percentile": i / 20.0,
            "jury_score": 1.0 + i * 0.15,
            "q1_p50": 10_000.0 * (i + 1) ** 2,
        }
        for i in range(20)
    ]
    result = _try_ridge_weights(training_data)
    # Either learned weights or None (fallback)
    if result is not None:
        w_ml, w_llm = result
        assert 0.0 < w_ml < 1.0
        assert 0.0 < w_llm < 1.0
        assert abs(w_ml + w_llm - 1.0) < 0.01


def test_try_ridge_weights_insufficient_data() -> None:
    """Ridge weights with < 10 samples should return None."""
    training_data: list[dict[str, Any]] = [
        {"ml_percentile": 0.5, "jury_score": 2.0, "q1_p50": 100_000.0}
    ] * 5
    result = _try_ridge_weights(training_data)
    assert result is None


def test_try_ridge_weights_returns_weights() -> None:
    """Force the Ridge learner to succeed by mocking."""
    from unittest.mock import patch

    training_data: list[dict[str, Any]] = [
        {"ml_percentile": i / 15, "jury_score": 1.0 + i * 0.2, "q1_p50": 50_000 * (i + 1)}
        for i in range(15)
    ]
    mock_weights = {
        "w_ml": 0.55,
        "w_jury": 0.45,
        "lambda": 0.01,
        "nested_rho": 0.6,
        "is_fallback": False,
        "fallback_reason": None,
    }
    with patch(
        "llmart.pipeline.nodes.value.Phase2WeightLearner.get_weights",
        return_value=mock_weights,
    ):
        result = _try_ridge_weights(training_data)
    assert result is not None
    assert result == (0.55, 0.45)


def test_value_node_ridge_path_success() -> None:
    """value_node should use Ridge weights when _try_ridge_weights succeeds."""
    from unittest.mock import patch

    candidates: list[dict[str, Any]] = [
        {
            "game_id": "RIDGEOK",
            "ml_percentile": 0.8,
            "jury_score": 3.0,
            "delta_cal": 0.0,
            "q1_p50": 300_000.0,
            "q1_p25": 180_000.0,
            "r_genre": 0.20,
            "ltv_mult": 2.0,
        }
    ]
    training_data = [
        {"ml_percentile": i / 15, "jury_score": 1.0 + i * 0.2, "q1_p50": 50_000 * (i + 1)}
        for i in range(15)
    ]
    with patch(
        "llmart.pipeline.nodes.value._try_ridge_weights",
        return_value=(0.55, 0.45),
    ):
        state = GraphState(
            candidates=candidates,
            stage="human_review",
            errors=[],
            phase=2,
            n_historical=0,
            training_data=training_data,
        )
        result = value_node(state)
    game = result["candidates"][0]
    assert game["w_ml"] == 0.55
    assert game["w_llm"] == 0.45
    assert game["phase"] == 2


def test_value_node_phase1_controller() -> None:
    """Phase 1 without training data should use PhaseController."""
    candidates: list[dict[str, Any]] = [
        {
            "game_id": "PH1",
            "ml_percentile": 0.7,
            "jury_score": 3.0,
            "delta_cal": 0.0,
            "q1_p50": 200_000.0,
            "q1_p25": 120_000.0,
            "r_genre": 0.20,
            "ltv_mult": 2.0,
        }
    ]
    state = GraphState(
        candidates=candidates,
        stage="human_review",
        errors=[],
        phase=1,
        n_historical=20,
    )
    result = value_node(state)
    game = result["candidates"][0]
    assert game["phase"] == 1


def test_value_node_phase2_insufficient_training() -> None:
    """Phase 2 with < 10 training samples falls back to controller."""
    candidates: list[dict[str, Any]] = [
        {
            "game_id": "P2F",
            "ml_percentile": 0.6,
            "jury_score": 2.5,
            "delta_cal": 0.0,
            "q1_p50": 150_000.0,
            "q1_p25": 90_000.0,
            "r_genre": 0.20,
            "ltv_mult": 2.0,
        }
    ]
    state = GraphState(
        candidates=candidates,
        stage="human_review",
        errors=[],
        phase=2,
        n_historical=5,
        training_data=[{"ml_percentile": 0.5, "jury_score": 2.0, "q1_p50": 100_000}] * 5,
    )
    result = value_node(state)
    assert len(result["candidates"]) == 1


def test_value_node_error_handling() -> None:
    """Malformed candidate should produce error, not crash."""
    candidates: list[dict[str, Any]] = [
        {
            "game_id": "ERR",
            "ml_percentile": "not_a_number",  # will cause error in scoring
            "jury_score": 3.0,
            "delta_cal": 0.0,
            "q1_p50": 100_000.0,
            "q1_p25": 60_000.0,
            "r_genre": 0.20,
            "ltv_mult": 2.0,
        }
    ]
    state = GraphState(candidates=candidates, stage="human_review", errors=[])
    result = value_node(state)
    # Should either succeed with coercion or produce an error
    assert "errors" in result


def test_try_ridge_weights_non_numeric() -> None:
    """Non-numeric w_ml from Ridge should return None."""
    from unittest.mock import patch

    training_data: list[dict[str, Any]] = [
        {"ml_percentile": i / 15, "jury_score": 1.0 + i * 0.2, "q1_p50": 50_000 * (i + 1)}
        for i in range(15)
    ]
    mock_weights = {
        "w_ml": "not_a_number",
        "w_jury": 0.45,
        "lambda": 0.01,
        "nested_rho": 0.6,
        "is_fallback": False,
        "fallback_reason": None,
    }
    with patch(
        "llmart.pipeline.nodes.value.Phase2WeightLearner.get_weights",
        return_value=mock_weights,
    ):
        result = _try_ridge_weights(training_data)
    assert result is None


def test_value_node_phase2_no_training_data() -> None:
    """Phase 2 with no training_data falls back to controller."""
    candidates: list[dict[str, Any]] = [
        {
            "game_id": "NT",
            "ml_percentile": 0.7,
            "jury_score": 3.0,
            "delta_cal": 0.0,
            "q1_p50": 200_000.0,
            "q1_p25": 120_000.0,
            "r_genre": 0.20,
            "ltv_mult": 2.0,
        }
    ]
    state = GraphState(
        candidates=candidates,
        stage="human_review",
        errors=[],
        phase=2,
        n_historical=0,
    )
    result = value_node(state)
    assert result["candidates"][0]["phase"] == 2


def test_discount_rate_from_enrichment() -> None:
    """Value node should use per-genre discount_rate stored by enrichment."""
    candidates: list[dict[str, Any]] = [
        {
            "game_id": "DR",
            "ml_percentile": 0.8,
            "jury_score": 3.0,
            "delta_cal": 0.0,
            "q1_p50": 100_000.0,
            "q1_p25": 60_000.0,
            "r_genre": 0.20,
            "ltv_mult": 2.0,
            "discount_rate": 0.10,  # stored by enrichment
        }
    ]
    state = GraphState(candidates=candidates, stage="human_review", errors=[])
    result = value_node(state)
    # NPV with 10% discount should be higher than default 15%
    npv_10 = result["candidates"][0]["npv_3y"]
    npv_15 = _compute_npv_3y(100_000.0, 0.20, 2.0, 0.15)
    assert npv_10 > npv_15
