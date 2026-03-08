"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmart.models.game import Game
from llmart.pipeline.state import GraphState

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "llmart" / "data"


@pytest.fixture
def sample_game() -> Game:
    """A valid Game instance for testing input validation."""
    return Game(
        game_id="G001",
        title="Test Game",
        genre="Roguelike",
        developer="Test Studio",
        steam_rating=0.85,
        review_count=5000,
        price_usd=19.99,
        release_year=2025,
        tags=["Roguelike", "Indie"],
    )


@pytest.fixture
def sample_game_dict(sample_game: Game) -> dict[str, Any]:
    """sample_game as a dict — mirrors pipeline candidate format."""
    return sample_game.model_dump()


@pytest.fixture
def sample_games_list() -> list[dict[str, Any]]:
    with open(DATA_DIR / "sample_games.json") as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


@pytest.fixture
def genre_params() -> dict[str, dict[str, Any]]:
    with open(DATA_DIR / "genre_params.json") as f:
        data: dict[str, dict[str, Any]] = json.load(f)
    return data


@pytest.fixture
def initial_graph_state(
    sample_games_list: list[dict[str, Any]],
    genre_params: dict[str, dict[str, Any]],
) -> GraphState:
    return GraphState(
        candidates=sample_games_list,
        stage="init",
        top_k=30,
        errors=[],
        mode="mock",
        genre_params=genre_params,
        total_input=len(sample_games_list),
    )
