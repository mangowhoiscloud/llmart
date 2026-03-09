"""Ground Truth 4-Tier Q1 revenue labels."""

from __future__ import annotations

from enum import StrEnum


class RevenueTier(StrEnum):
    """SOT 4-Tier Q1 revenue classification."""

    HOBBY = "Hobby"  # < $250K (66%)
    SIDE = "Side"  # $250K ~ $2M (23%)
    HIT = "Hit"  # $2M ~ $20M (8%)
    MEGA = "Mega"  # >= $20M (2-3%)


# Revenue tier boundaries (USD)
_TIER_BOUNDARIES: list[tuple[float, RevenueTier]] = [
    (20_000_000.0, RevenueTier.MEGA),
    (2_000_000.0, RevenueTier.HIT),
    (250_000.0, RevenueTier.SIDE),
]

# Genre-specific Q1-to-Y1 ratio types
GENRE_Q1_RATIOS: dict[str, float] = {
    "front-loaded": 0.55,
    "balanced": 0.45,
    "live-service": 0.35,
}


def classify_tier(q1_revenue: float) -> RevenueTier:
    """Classify Q1 revenue into SOT 4-Tier label."""
    for threshold, tier in _TIER_BOUNDARIES:
        if q1_revenue >= threshold:
            return tier
    return RevenueTier.HOBBY


def q1_to_y1(q1: float, genre_type: str = "balanced") -> float:
    """Convert Q1 revenue to estimated Year 1 revenue.

    Y1 = Q1 / q1_ratio where q1_ratio is genre-dependent.
    """
    ratio = GENRE_Q1_RATIOS.get(genre_type, GENRE_Q1_RATIOS["balanced"])
    if ratio <= 0:
        return 0.0
    return q1 / ratio
