"""Game domain model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Game(BaseModel):
    """Represents a game candidate in the pipeline."""

    game_id: str
    title: str
    genre: str
    developer: str
    steam_rating: float = Field(ge=0.0, le=1.0)
    review_count: int = Field(ge=0)
    price_usd: float = Field(ge=0.0)
    release_year: int
    tags: list[str] = Field(default_factory=list)
