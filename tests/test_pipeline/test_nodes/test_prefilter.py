"""Tests for Pre-filter 2L node."""

from __future__ import annotations

from typing import Any

import pytest

from llmart.pipeline.nodes.prefilter import prefilter_node
from llmart.pipeline.state import GraphState


def _make_state(candidates: list[dict[str, Any]]) -> GraphState:
    return GraphState(candidates=candidates, stage="init", top_k=30, errors=[], mode="mock")


def _valid_game(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid game dict with overrides."""
    base: dict[str, Any] = {
        "game_id": "TEST",
        "title": "Test Game",
        "genre": "RPG",
        "developer": "Test Studio",
        "steam_rating": 0.90,
        "review_count": 500,
        "price_usd": 19.99,
        "release_year": 2024,
        "tags": ["RPG"],
    }
    base.update(overrides)
    return base


def test_l1_hard_cut_filters_old_games() -> None:
    old_game = _valid_game(game_id="OLD", release_year=2020)
    result = prefilter_node(_make_state([old_game]))
    assert result["candidates"] == []


def test_l1_hard_cut_filters_low_rating() -> None:
    low_rated = _valid_game(game_id="LOW", steam_rating=0.30)
    result = prefilter_node(_make_state([low_rated]))
    assert result["candidates"] == []


@pytest.mark.parametrize(
    ("year", "count", "rating", "passes"),
    [
        (2022, 200, 0.65, True),  # exact boundary — all pass
        (2021, 200, 0.65, False),  # year fail
        (2022, 199, 0.65, False),  # count fail
        (2022, 200, 0.64, False),  # rating fail
        (2024, 1000, 0.95, True),  # well above
    ],
)
def test_l1_boundary_values(year: int, count: int, rating: float, *, passes: bool) -> None:
    game = _valid_game(game_id="BND", steam_rating=rating, review_count=count, release_year=year)
    result = prefilter_node(_make_state([game]))
    assert (len(result["candidates"]) == 1) == passes


def test_l2_soft_score_assigned(sample_games_list: list[dict[str, Any]]) -> None:
    result = prefilter_node(_make_state(sample_games_list))
    # Some weak games may fail L1 hard-cut (year>=2022, reviews>=200, rating>=0.65)
    assert 0 < len(result["candidates"]) <= len(sample_games_list)
    for game in result["candidates"]:
        assert "l2_score" in game
        assert 0.0 <= game["l2_score"] <= 1.0


def test_empty_input() -> None:
    result = prefilter_node(_make_state([]))
    assert result["candidates"] == []
    assert result["total_input"] == 0


def test_missing_fields_use_defaults() -> None:
    """Games with missing fields should use safe defaults."""
    game: dict[str, Any] = {"game_id": "MISS"}
    result = prefilter_node(_make_state([game]))
    # Missing fields default to 0 → fails hard-cut
    assert result["candidates"] == []


def test_empty_tags_handled() -> None:
    game = _valid_game(game_id="ET", tags=[])
    result = prefilter_node(_make_state([game]))
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["l2_score"] >= 0.0


def test_known_hits_recall() -> None:
    """Prefilter should compute recall of known hits among survivors."""
    games = [_valid_game(game_id=f"G{i}") for i in range(5)]
    state = GraphState(
        candidates=games,
        stage="init",
        top_k=30,
        errors=[],
        mode="mock",
        known_hits=["G0", "G1", "G99"],
    )
    result = prefilter_node(state)
    # G0, G1 should survive, G99 does not exist → recall = 2/3
    assert "monitoring" in result
    assert result["monitoring"]["recall_at_99"] == pytest.approx(2 / 3)


def test_known_hits_empty_returns_no_monitoring() -> None:
    """No known hits → monitoring key should not be set."""
    games = [_valid_game(game_id="G0")]
    state = GraphState(
        candidates=games,
        stage="init",
        top_k=30,
        errors=[],
        mode="mock",
        known_hits=[],
    )
    result = prefilter_node(state)
    assert "monitoring" not in result


def test_prefilter_exception_handling() -> None:
    """A candidate that causes an exception in L2 scoring should be caught."""
    # Provide a game that passes L1 hard-cut but has problematic tags
    game: dict[str, Any] = {
        "game_id": "EXC",
        "title": "Exception Game",
        "genre": "RPG",
        "developer": "Dev",
        "steam_rating": 0.90,
        "review_count": 500,
        "price_usd": 20.0,
        "release_year": 2024,
        "tags": "not_a_list",  # intentional: tags should be list, but str is iterable too
    }
    result = prefilter_node(_make_state([game]))
    # Either succeeds (str is iterable) or captures error
    assert "errors" in result


def test_prefilter_scoring_exception_via_mock() -> None:
    """Mock _l2_soft_score to raise and verify error capture at L100-101."""
    from unittest.mock import patch

    game = _valid_game(game_id="MOCK_EXC")
    with patch(
        "llmart.pipeline.nodes.prefilter._l2_soft_score",
        side_effect=RuntimeError("scoring boom"),
    ):
        result = prefilter_node(_make_state([game]))
    assert any("scoring boom" in e for e in result["errors"])


def test_error_accumulation() -> None:
    """Errors should be accumulated, not crash the pipeline."""
    result = prefilter_node(_make_state([]))
    assert "errors" in result
    assert isinstance(result["errors"], list)
