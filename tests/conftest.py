"""Shared test fixtures."""

from __future__ import annotations

import pytest

from llmart.models.game import Game


@pytest.fixture
def sample_game() -> Game:
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
