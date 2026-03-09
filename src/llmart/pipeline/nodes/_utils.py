"""Shared utilities and constants for pipeline nodes."""

from __future__ import annotations

import math
from datetime import date

# LLM Judge 4-cat scale (L=1, M=2, H=3, E=4)
JURY_SCALE_MAX = 4.0

# Current year for recency calculations
CURRENT_YEAR = date.today().year


def wilson_lower_bound(positive: int, total: int, z: float = 1.96) -> float:
    """Wilson score lower bound for a Bernoulli parameter.

    Used by both prefilter (L2 soft-score) and ml_scoring (feature).
    """
    if total == 0:
        return 0.0
    p_hat = positive / total
    denominator = 1 + z * z / total
    centre = p_hat + z * z / (2 * total)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total)
    return (centre - spread) / denominator


def developer_wilson_score(successes: int, total_games: int, z: float = 1.96) -> float:
    """Wilson lower bound for developer track record."""
    return wilson_lower_bound(successes, total_games, z)


# Shared set of relevant Steam tags for quality scoring.
# Used by both prefilter (L2 soft-score) and ml_scoring (feature).
RELEVANT_TAGS: frozenset[str] = frozenset(
    {
        "Roguelike",
        "Deckbuilder",
        "Deckbuilding",
        "City Builder",
        "RPG",
        "Strategy",
        "Survival",
        "Co-op",
        "Online Co-Op",
        "Card Game",
        "Action Roguelike",
        "Roguelite",
        "Bullet Hell",
        "Tower Defense",
        "Metroidvania",
        "Simulation",
        "Horror",
        "Multiplayer",
        "Open World",
        "Crafting",
        "Base Building",
        "Singleplayer",
        "Management",
        "Resource Management",
        "Exploration",
    }
)
