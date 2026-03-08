"""Tests for Data Enrichment node."""

from __future__ import annotations

from typing import Any

from llmart.pipeline.nodes.enrichment import (
    DEFAULT_GENRE_PARAMS,
    Q1_SHARE_DEFAULT,
    SALES_MULT,
    STEAM_NET,
    enrichment_node,
)
from llmart.pipeline.state import GraphState


def test_genre_params_mapping(genre_params: dict[str, dict[str, Any]]) -> None:
    candidates = [{"game_id": "G001", "genre": "Roguelike", "review_count": 1000, "price_usd": 20}]
    state = GraphState(
        candidates=candidates, genre_params=genre_params, stage="llm_judge", errors=[]
    )
    result = enrichment_node(state)
    game = result["candidates"][0]
    assert game["r_genre"] == genre_params["Roguelike"]["r_genre"]
    assert game["ltv_mult"] == genre_params["Roguelike"]["ltv_mult"]


def test_unknown_genre_fallback() -> None:
    candidates = [{"game_id": "X1", "genre": "Unknown", "review_count": 100, "price_usd": 10}]
    state = GraphState(candidates=candidates, genre_params={}, stage="llm_judge", errors=[])
    result = enrichment_node(state)
    game = result["candidates"][0]
    assert game["r_genre"] == DEFAULT_GENRE_PARAMS["r_genre"]
    assert game["ltv_mult"] == DEFAULT_GENRE_PARAMS["ltv_mult"]


def test_q1_revenue_boxleiter_method() -> None:
    """Verify Boxleiter-derived Q1 revenue estimates with genre-differentiated Q1 share."""
    candidates = [{"game_id": "Q1", "genre": "RPG", "review_count": 10000, "price_usd": 30.0}]
    state = GraphState(candidates=candidates, genre_params={}, stage="llm_judge", errors=[])
    result = enrichment_node(state)
    game = result["candidates"][0]
    # Default q1_type is "balanced" (Q1_SHARE_DEFAULT = 0.45)
    y1_revenue = 10000 * SALES_MULT * 30.0 * STEAM_NET
    expected_p50 = y1_revenue * Q1_SHARE_DEFAULT
    expected_p25 = expected_p50 * 0.6
    expected_p75 = expected_p50 * 1.5
    assert abs(game["q1_p50"] - expected_p50) < 0.01
    assert abs(game["q1_p25"] - expected_p25) < 0.01
    assert abs(game["q1_p75"] - expected_p75) < 0.01


def test_q1_share_genre_differentiated() -> None:
    """Verify different q1_ratio_types produce different Q1 estimates."""
    results = {}
    for q1_type in ("front-loaded", "balanced", "live-service"):
        candidates = [
            {"game_id": f"T_{q1_type}", "genre": "RPG", "review_count": 5000, "price_usd": 20.0}
        ]
        gp = {
            "RPG": {
                "r_genre": 0.20,
                "ltv_mult": 2.0,
                "discount_rate": 0.15,
                "q1_ratio_type": q1_type,
            }
        }
        state = GraphState(candidates=candidates, genre_params=gp, stage="llm_judge", errors=[])
        result = enrichment_node(state)
        results[q1_type] = result["candidates"][0]["q1_p50"]

    # front-loaded should have highest Q1, live-service lowest
    assert results["front-loaded"] > results["balanced"]
    assert results["balanced"] > results["live-service"]


def test_q1_revenue_realistic_range() -> None:
    """Verify estimates produce realistic Q1 revenue for known games."""
    candidates = [
        {"game_id": "BAL", "genre": "Roguelike", "review_count": 148000, "price_usd": 14.99}
    ]
    state = GraphState(candidates=candidates, genre_params={}, stage="llm_judge", errors=[])
    result = enrichment_node(state)
    game = result["candidates"][0]
    assert game["q1_p50"] > 100_000  # should be a significant amount
    assert game["q1_p25"] > 0


def test_zero_reviews_produces_zero_revenue() -> None:
    candidates = [{"game_id": "Z", "genre": "RPG", "review_count": 0, "price_usd": 20.0}]
    state = GraphState(candidates=candidates, genre_params={}, stage="llm_judge", errors=[])
    result = enrichment_node(state)
    game = result["candidates"][0]
    assert game["q1_p50"] == 0.0
    assert game["q1_p25"] == 0.0
    assert len(result["errors"]) > 0  # warns about zero Q1


def test_enrichment_exception_handling() -> None:
    """Candidate that causes enrichment exception should be caught."""
    bad_game: dict[str, Any] = {
        "game_id": "EXC",
        "genre": "RPG",
        "review_count": None,  # will cause TypeError in multiplication
        "price_usd": 20.0,
    }
    state = GraphState(candidates=[bad_game], genre_params={}, stage="llm_judge", errors=[])
    result = enrichment_node(state)
    assert len(result["errors"]) > 0


def test_discount_rate_stored() -> None:
    """Enrichment should store discount_rate for downstream use."""
    candidates = [{"game_id": "D1", "genre": "Roguelike", "review_count": 100, "price_usd": 10}]
    state = GraphState(
        candidates=candidates,
        genre_params={"Roguelike": {"r_genre": 0.22, "ltv_mult": 2.2, "discount_rate": 0.12}},
        stage="llm_judge",
        errors=[],
    )
    result = enrichment_node(state)
    assert result["candidates"][0]["discount_rate"] == 0.12
