"""Tests for T2 LLM-as-Judge node."""

from __future__ import annotations

from typing import Any

import pytest

from llmart.pipeline.nodes.llm_judge import (
    DIM_WEIGHTS,
    DIMENSIONS,
    SCORE_MAX,
    SCORE_MIN,
    _aggregate_passes,
    _deterministic_seed,
    _jury_score,
    _mock_calibrator,
    _mock_single_pass,
    _pass3_agreement,
    llm_judge_node,
)
from llmart.pipeline.state import GraphState
from llmart.prompts.rubric_5dim import build_dim_schema

SAMPLE_GAME: dict[str, Any] = {
    "game_id": "G001",
    "title": "Test Game",
    "genre": "Roguelike",
    "steam_rating": 0.85,
    "review_count": 5000,
    "price_usd": 19.99,
    "release_year": 2025,
    "tags": ["Roguelike"],
}


def test_mock_scores_in_4cat_range() -> None:
    scores = _mock_single_pass(SAMPLE_GAME, 0)
    assert set(scores.keys()) == set(DIMENSIONS)
    for v in scores.values():
        assert SCORE_MIN <= v <= SCORE_MAX


def test_five_dimension_scores(sample_games_list: list[dict[str, Any]]) -> None:
    state = GraphState(candidates=sample_games_list, stage="ml_scoring", errors=[], mode="mock")
    result = llm_judge_node(state)
    for game in result["candidates"]:
        assert set(game["dim_scores"].keys()) == set(DIMENSIONS)
        for v in game["dim_scores"].values():
            assert float(SCORE_MIN) <= v <= float(SCORE_MAX)


def test_delta_cal_in_range(sample_games_list: list[dict[str, Any]]) -> None:
    state = GraphState(candidates=sample_games_list, stage="ml_scoring", errors=[], mode="mock")
    result = llm_judge_node(state)
    for game in result["candidates"]:
        assert -0.15 <= game["delta_cal"] <= 0.15


def test_pass3_agreement_logic() -> None:
    base = dict.fromkeys(DIMENSIONS, 3)
    passes_agree = [base.copy(), base.copy(), base.copy()]
    assert _pass3_agreement(passes_agree, threshold=1.0) is True

    passes_disagree = [
        {**base, "gameplay": 1},
        {**base, "gameplay": 3},
        {**base, "gameplay": 4},
    ]
    assert _pass3_agreement(passes_disagree, threshold=1.0) is False


def test_jury_score_computation() -> None:
    dim_scores = dict.fromkeys(DIMENSIONS, 3.0)
    expected = 3.0  # all dims same -> weighted sum = 3.0
    assert abs(_jury_score(dim_scores) - expected) < 1e-4


def test_jury_score_weighted() -> None:
    """Verify weights are applied correctly with varied dimension scores."""
    dim_scores = {
        "gameplay": 4.0,
        "innovation": 1.0,
        "monetization": 1.0,
        "polish": 1.0,
        "narrative": 1.0,
    }
    expected = 4.0 * 0.30 + 1.0 * 0.20 + 1.0 * 0.20 + 1.0 * 0.15 + 1.0 * 0.15
    assert abs(_jury_score(dim_scores) - expected) < 1e-4


def test_deterministic_reproducibility() -> None:
    """Same game_id + pass_idx should produce identical scores."""
    scores1 = _mock_single_pass(SAMPLE_GAME, 0)
    scores2 = _mock_single_pass(SAMPLE_GAME, 0)
    assert scores1 == scores2


def test_different_passes_may_differ() -> None:
    """Different pass indices should produce (potentially) different scores."""
    scores0 = _mock_single_pass(SAMPLE_GAME, 0)
    scores1 = _mock_single_pass(SAMPLE_GAME, 1)
    # They may be identical by chance, but the seeds are different
    assert isinstance(scores0, dict)
    assert isinstance(scores1, dict)


@pytest.mark.parametrize(
    ("spread_vals", "agree", "expected_min", "expected_max"),
    [
        ([15, 15, 15], True, -0.001, 0.001),  # agree -> 0.0
        ([10, 15, 20], False, -0.15, 0.15),  # disagree -> signed offset
        ([15, 15, 16], False, -0.15, 0.15),  # disagree, small per-dim diff but sum-spread=5
    ],
)
def test_mock_calibrator_edge_cases(
    spread_vals: list[int], agree: bool, expected_min: float, expected_max: float
) -> None:
    passes = [dict.fromkeys(DIMENSIONS, v) for v in spread_vals]
    result = _mock_calibrator(passes, agree)
    assert expected_min <= result <= expected_max


def test_mock_calibrator_can_be_negative() -> None:
    """Calibrator should return negative when later passes trend lower."""
    passes = [
        dict.fromkeys(DIMENSIONS, 4),  # high first pass
        dict.fromkeys(DIMENSIONS, 3),  # medium
        dict.fromkeys(DIMENSIONS, 1),  # low last pass -> negative trend
    ]
    result = _mock_calibrator(passes, agree=False)
    assert result < 0


def test_aggregate_passes() -> None:
    passes = [
        dict.fromkeys(DIMENSIONS, 2),
        dict.fromkeys(DIMENSIONS, 4),
    ]
    agg = _aggregate_passes(passes)
    for dim in DIMENSIONS:
        assert agg[dim] == 3.0


def test_empty_candidates() -> None:
    state = GraphState(candidates=[], stage="ml_scoring", errors=[], mode="mock")
    result = llm_judge_node(state)
    assert result["candidates"] == []


def test_error_accumulation_with_malformed_game() -> None:
    """Malformed game (missing game_id) should produce an error, not crash."""
    malformed: dict[str, Any] = {"title": "No ID"}  # missing game_id -> KeyError in seed
    state = GraphState(candidates=[malformed], stage="ml_scoring", errors=[], mode="mock")
    result = llm_judge_node(state)
    assert len(result["errors"]) > 0
    assert result["candidates"] == []


def test_dim_weights_sum_to_one() -> None:
    assert abs(sum(DIM_WEIGHTS) - 1.0) < 1e-9


def test_build_dim_schema_order() -> None:
    """Dimension schema should reflect the given order."""
    schema = build_dim_schema(["narrative", "gameplay", "polish", "innovation", "monetization"])
    lines = schema.strip().split("\n")
    assert '"narrative"' in lines[0]
    assert '"gameplay"' in lines[1]
    assert len(lines) == 5


def test_deterministic_seed_stability() -> None:
    """Same inputs should produce same seed; different inputs should differ."""
    s1 = _deterministic_seed("GAME_A", 0)
    s2 = _deterministic_seed("GAME_A", 0)
    s3 = _deterministic_seed("GAME_A", 1)
    s4 = _deterministic_seed("GAME_B", 0)
    assert s1 == s2  # deterministic
    assert s1 != s3  # different pass_idx
    assert s1 != s4  # different game_id


def test_mock_calibrator_positive_trend() -> None:
    """Calibrator should return positive when later passes trend higher."""
    passes = [
        dict.fromkeys(DIMENSIONS, 1),  # low first pass
        dict.fromkeys(DIMENSIONS, 2),  # medium
        dict.fromkeys(DIMENSIONS, 4),  # high last pass -> positive trend
    ]
    result = _mock_calibrator(passes, agree=False)
    assert result > 0


def test_resolve_mode_mock() -> None:
    """Non-real mode should always return mock."""
    from llmart.pipeline.nodes.llm_judge import _resolve_mode

    mode, msgs = _resolve_mode("mock")
    assert mode == "mock"
    assert msgs == []


def test_resolve_mode_real_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real mode without keys should fall back with degradation messages."""
    from llmart.pipeline.nodes.llm_judge import _resolve_mode

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mode, msgs = _resolve_mode("real")
    assert mode == "mock"
    assert len(msgs) > 0


def test_llm_judge_degradation_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real mode with no keys should produce degradation errors."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = GraphState(candidates=[SAMPLE_GAME], stage="ml_scoring", errors=[], mode="real")
    result = llm_judge_node(state)
    assert len(result["errors"]) > 0  # degradation messages


def test_llm_judge_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLMART_REAL_API=true env override should set use_real=True."""
    from unittest.mock import patch

    monkeypatch.setenv("LLMART_REAL_API", "true")
    # Mock _real_llm_judge to avoid actual API calls
    mock_passes = [
        {"gameplay": 3, "innovation": 3, "monetization": 2, "polish": 3, "narrative": 3},
        {"gameplay": 3, "innovation": 2, "monetization": 2, "polish": 3, "narrative": 2},
        {"gameplay": 3, "innovation": 3, "monetization": 2, "polish": 3, "narrative": 3},
    ]
    mock_result = (
        {"gameplay": 3.0, "innovation": 2.5, "monetization": 2.0, "polish": 3.0, "narrative": 2.5},
        2.7,
        0.01,
        True,
        False,
        mock_passes,
    )
    with patch("llmart.pipeline.nodes.llm_judge._real_llm_judge", return_value=mock_result):
        state = GraphState(candidates=[SAMPLE_GAME], stage="ml_scoring", errors=[], mode="mock")
        result = llm_judge_node(state)
        assert len(result["candidates"]) == 1


def test_llm_judge_partial_real_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Partial real mode (OpenAI key only) should use _real_primary_only."""
    from unittest.mock import patch

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mock_passes = [
        {"gameplay": 3, "innovation": 3, "monetization": 2, "polish": 3, "narrative": 3},
        {"gameplay": 3, "innovation": 2, "monetization": 2, "polish": 3, "narrative": 2},
        {"gameplay": 3, "innovation": 3, "monetization": 2, "polish": 3, "narrative": 3},
    ]
    mock_result = (
        {"gameplay": 3.0, "innovation": 2.5, "monetization": 2.0, "polish": 3.0, "narrative": 2.5},
        2.7,
        0.0,
        True,
        False,
        mock_passes,
    )
    with patch("llmart.pipeline.nodes.llm_judge._real_primary_only", return_value=mock_result):
        state = GraphState(candidates=[SAMPLE_GAME], stage="ml_scoring", errors=[], mode="real")
        result = llm_judge_node(state)
        assert len(result["candidates"]) == 1


def test_jury_star_fields_in_output() -> None:
    """Output should include jury_star, jury_delta, jury_conf fields."""
    state = GraphState(candidates=[SAMPLE_GAME], stage="ml_scoring", errors=[], mode="mock")
    result = llm_judge_node(state)
    game = result["candidates"][0]
    assert "jury_star" in game
    assert "jury_delta" in game
    assert "jury_conf" in game


def test_output_structure_fields() -> None:
    """Each output game should have all LLM judge output fields."""
    state = GraphState(candidates=[SAMPLE_GAME], stage="ml_scoring", errors=[], mode="mock")
    result = llm_judge_node(state)
    game = result["candidates"][0]
    assert "dim_scores" in game
    assert "jury_score" in game
    assert "delta_cal" in game
    assert "pass3_agree" in game
    assert "flagged" in game
    assert isinstance(game["jury_score"], float)
    assert isinstance(game["pass3_agree"], bool)
    assert isinstance(game["flagged"], bool)


def test_genre_bias_fidelity() -> None:
    """Roguelike should score higher on gameplay than Visual Novel (genre bias)."""
    roguelike = {**SAMPLE_GAME, "game_id": "RL1", "genre": "Roguelike", "steam_rating": 0.80}
    vn = {**SAMPLE_GAME, "game_id": "VN1", "genre": "Visual Novel", "steam_rating": 0.80}
    rl_scores = [_mock_single_pass(roguelike, i) for i in range(10)]
    vn_scores = [_mock_single_pass(vn, i) for i in range(10)]
    rl_gameplay_avg = sum(s["gameplay"] for s in rl_scores) / len(rl_scores)
    vn_gameplay_avg = sum(s["gameplay"] for s in vn_scores) / len(vn_scores)
    assert rl_gameplay_avg > vn_gameplay_avg, (
        f"Roguelike gameplay ({rl_gameplay_avg:.2f}) should > VN gameplay ({vn_gameplay_avg:.2f})"
    )
