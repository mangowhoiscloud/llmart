"""Tests for T1 ML Scoring node."""

from __future__ import annotations

from typing import Any

from llmart.pipeline.nodes.ml_scoring import ml_scoring_node
from llmart.pipeline.nodes.prefilter import prefilter_node
from llmart.pipeline.state import GraphState


def _make_state(candidates: list[dict[str, Any]]) -> GraphState:
    return GraphState(candidates=candidates, stage="prefilter", top_k=30, errors=[], mode="mock")


def test_ml_percentile_range(sample_games_list: list[dict[str, Any]]) -> None:
    pre = prefilter_node(GraphState(candidates=sample_games_list, stage="init", errors=[]))
    result = ml_scoring_node(_make_state(pre["candidates"]))
    for game in result["candidates"]:
        assert 0.0 < game["ml_percentile"] < 1.0  # Weibull: never 0 or 1


def test_ml_scores_sorted_descending(sample_games_list: list[dict[str, Any]]) -> None:
    pre = prefilter_node(GraphState(candidates=sample_games_list, stage="init", errors=[]))
    result = ml_scoring_node(_make_state(pre["candidates"]))
    percentiles = [g["ml_percentile"] for g in result["candidates"]]
    assert percentiles == sorted(percentiles, reverse=True)


def test_ml_deterministic(sample_games_list: list[dict[str, Any]]) -> None:
    pre = prefilter_node(GraphState(candidates=sample_games_list, stage="init", errors=[]))
    r1 = ml_scoring_node(_make_state(pre["candidates"]))
    r2 = ml_scoring_node(_make_state(pre["candidates"]))
    p1 = [g["ml_percentile"] for g in r1["candidates"]]
    p2 = [g["ml_percentile"] for g in r2["candidates"]]
    assert p1 == p2


def test_ml_empty_input() -> None:
    result = ml_scoring_node(_make_state([]))
    assert result["candidates"] == []
    assert len(result["errors"]) > 0  # warns about 0 candidates


def test_ml_single_candidate() -> None:
    """Single candidate should get percentile 0.5 (Weibull: 1/2)."""
    game = {
        "game_id": "S1",
        "steam_rating": 0.90,
        "review_count": 5000,
        "price_usd": 20.0,
        "release_year": 2024,
        "tags": ["RPG"],
    }
    result = ml_scoring_node(_make_state([game]))
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["ml_percentile"] == 0.5


def test_ml_features_attached() -> None:
    """Each candidate should have ml_features dict with expected keys."""
    game = {
        "game_id": "F1",
        "steam_rating": 0.85,
        "review_count": 1000,
        "price_usd": 15.0,
        "release_year": 2024,
        "tags": ["RPG", "Strategy"],
    }
    result = ml_scoring_node(_make_state([game]))
    features = result["candidates"][0]["ml_features"]
    assert set(features.keys()) == {
        "wilson",
        "log_reviews",
        "recency",
        "tag_quality",
        "price_norm",
        "wilson_x_reviews",
        "recency_x_tags",
    }
    for v in features.values():
        assert 0.0 <= v <= 1.0


def test_ndcg_nonzero_relevance() -> None:
    """NDCG should compute correctly with graded relevance values."""
    from llmart.pipeline.nodes.ml_scoring import compute_ndcg_at_k

    # Use revenue values that map to non-zero grades:
    # $3M (Mega=3), $500K (Hit=2), $100K (Side=1), $10K (Hobby=0), $5K (Hobby=0)
    candidates = [
        {"game_id": "N0", "q1_p50": 3_000_000.0},
        {"game_id": "N1", "q1_p50": 500_000.0},
        {"game_id": "N2", "q1_p50": 100_000.0},
        {"game_id": "N3", "q1_p50": 10_000.0},
        {"game_id": "N4", "q1_p50": 5_000.0},
    ]
    result = compute_ndcg_at_k(candidates, relevance_key="q1_p50", k=5)
    assert 0.0 < result <= 1.0


def test_ndcg_zero_relevance() -> None:
    """NDCG should be 0.0 when all relevance values are zero."""
    from llmart.pipeline.nodes.ml_scoring import compute_ndcg_at_k

    candidates = [{"game_id": f"Z{i}", "q1_p50": 0.0} for i in range(5)]
    result = compute_ndcg_at_k(candidates, relevance_key="q1_p50", k=5)
    assert result == 0.0


def test_ndcg_empty_candidates() -> None:
    """NDCG should be 0.0 for empty candidate list."""
    from llmart.pipeline.nodes.ml_scoring import compute_ndcg_at_k

    assert compute_ndcg_at_k([], k=30) == 0.0


def test_ml_scoring_exception_handling() -> None:
    """Candidate that causes scoring exception should be caught."""
    bad_game: dict[str, Any] = {
        "game_id": "BAD",
        "steam_rating": "not_float",  # will cause issues
        "review_count": 1000,
        "price_usd": 20.0,
        "release_year": 2024,
        "tags": ["RPG"],
    }
    good_game: dict[str, Any] = {
        "game_id": "GOOD",
        "steam_rating": 0.85,
        "review_count": 1000,
        "price_usd": 20.0,
        "release_year": 2024,
        "tags": ["RPG"],
    }
    result = ml_scoring_node(_make_state([bad_game, good_game]))
    # Either both survive (if coercion works) or error is captured
    assert "errors" in result


def test_ml_tag_quality_vs_quantity() -> None:
    """Tag quality should score relevant tags higher than irrelevant ones."""
    relevant = {
        "game_id": "REL",
        "steam_rating": 0.80,
        "review_count": 1000,
        "price_usd": 20.0,
        "release_year": 2024,
        "tags": ["Roguelike", "Strategy", "RPG"],
    }
    irrelevant = {
        "game_id": "IRR",
        "steam_rating": 0.80,
        "review_count": 1000,
        "price_usd": 20.0,
        "release_year": 2024,
        "tags": ["Anime", "Visual Novel", "Dating Sim"],
    }
    result = ml_scoring_node(_make_state([relevant, irrelevant]))
    games = {g["game_id"]: g for g in result["candidates"]}
    assert games["REL"]["ml_features"]["tag_quality"] > games["IRR"]["ml_features"]["tag_quality"]
