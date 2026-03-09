"""Synthetic data generator for training, validation, and phase-transition simulation.

Generates realistic game data across 27 genres with:
- Revenue distributions following power-law (Pareto) per genre
- Quarterly labels for PSI drift detection
- Feedback data for phase-transition evaluation
"""

from __future__ import annotations

import hashlib
import json
import math
from importlib import resources
from typing import Any

# --- Deterministic PRNG (no numpy dependency) ---

_DEVELOPER_PREFIXES = [
    "Pixel",
    "Neon",
    "Cyber",
    "Crystal",
    "Storm",
    "Shadow",
    "Dream",
    "Star",
    "Moon",
    "Iron",
    "Void",
    "Ember",
    "Frost",
    "Solar",
    "Quantum",
    "Arcane",
    "Prism",
    "Terra",
    "Nova",
    "Echo",
    "Apex",
    "Flux",
    "Zenith",
]
_DEVELOPER_SUFFIXES = [
    "Works",
    "Games",
    "Labs",
    "Studio",
    "Interactive",
    "Arts",
    "Forge",
    "Soft",
    "Craft",
    "Play",
    "Byte",
    "Ventures",
]
_TAG_POOL = [
    "Indie",
    "Action",
    "Strategy",
    "Singleplayer",
    "Multiplayer",
    "Roguelike",
    "RPG",
    "Survival",
    "Horror",
    "Puzzle",
    "Simulation",
    "Pixel Graphics",
    "Retro",
    "Atmospheric",
    "Difficult",
    "Early Access",
    "Co-op",
    "PvP",
    "Exploration",
    "Base Building",
    "Tower Defense",
    "Card Game",
    "Deckbuilding",
    "Bullet Hell",
    "Arena Shooter",
    "Open World",
    "Sandbox",
    "Story Rich",
    "Dark Fantasy",
    "Sci-fi",
]


class _LCG:
    """Linear Congruential Generator for deterministic pseudo-random numbers."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self._state = (self._state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self._state

    def random(self) -> float:
        """Uniform [0, 1)."""
        return self._next() / 0x100000000

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive [lo, hi]."""
        return lo + self._next() % (hi - lo + 1)

    def choice(self, seq: list[Any]) -> Any:
        """Pick one element."""
        return seq[self._next() % len(seq)]

    def sample(self, seq: list[Any], k: int) -> list[Any]:
        """Sample k elements without replacement (Fisher-Yates partial)."""
        pool = list(seq)
        n = len(pool)
        k = min(k, n)
        for i in range(k):
            j = i + self._next() % (n - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]

    def gauss(self, mu: float, sigma: float) -> float:
        """Box-Muller normal distribution."""
        u1 = max(self.random(), 1e-12)
        u2 = self.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z


def _stable_seed(text: str) -> int:
    """Deterministic seed from string."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _load_genre_params() -> dict[str, dict[str, Any]]:
    """Load bundled genre_params.json."""
    path = resources.files("llmart") / "data" / "genre_params.json"
    with open(str(path), encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


# --- Revenue generation ---

# Lognormal parameters by genre q1_type
# Target tier distribution: Hobby ~60%, Side ~25%, Hit ~10%, Mega ~3-5%
# Q1 tier boundaries: Hobby<$250K, Side $250K-$2M, Hit $2M-$20M, Mega>=$20M
_LOGNORMAL_MU: dict[str, float] = {
    "front-loaded": 11.8,  # higher base (front-loaded monetize faster)
    "balanced": 12.0,
    "live-service": 12.2,  # highest base (long-tail revenue)
}
_LOGNORMAL_SIGMA: dict[str, float] = {
    "front-loaded": 2.6,
    "balanced": 2.8,
    "live-service": 3.0,
}


def _generate_y1_revenue(rng: _LCG, q1_type: str) -> float:
    """Generate Y1 revenue from genre-appropriate lognormal distribution.

    Lognormal produces realistic power-law-like revenue distribution:
    many small games, few mega-hits.
    """
    mu = _LOGNORMAL_MU.get(q1_type, 12.0)
    sigma = _LOGNORMAL_SIGMA.get(q1_type, 2.8)
    log_revenue = rng.gauss(mu, sigma)
    revenue = math.exp(max(log_revenue, math.log(5000.0)))
    # Cap at $200M (realistic Steam indie ceiling)
    return min(revenue, 200_000_000.0)


# --- Q1 share ratios ---
_Q1_SHARES: dict[str, float] = {
    "front-loaded": 0.55,
    "balanced": 0.45,
    "live-service": 0.35,
}


def generate_training_dataset(
    n: int = 500,
    seed: int = 42,
    quarters: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate n synthetic training games with realistic distributions.

    Args:
        n: Number of games to generate.
        seed: Random seed for reproducibility.
        quarters: List of quarter labels to distribute games across.
            Defaults to ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1"].

    Returns:
        List of game dicts with all training fields + quarter label.
    """
    if quarters is None:
        quarters = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1"]

    rng = _LCG(seed)
    genre_params = _load_genre_params()
    genres = list(genre_params.keys())

    games: list[dict[str, Any]] = []

    for i in range(n):
        game_id = f"SYN{i + 1:04d}"
        genre = rng.choice(genres)
        gp = genre_params[genre]
        q1_type: str = gp.get("q1_ratio_type", "balanced")

        # Revenue
        y1_revenue = _generate_y1_revenue(rng, q1_type)
        q1_share = _Q1_SHARES.get(q1_type, 0.45)
        q1_revenue = y1_revenue * q1_share

        # Hit tier from Q1 revenue
        if q1_revenue >= 20_000_000:
            hit_tier = "Mega"
        elif q1_revenue >= 2_000_000:
            hit_tier = "Hit"
        elif q1_revenue >= 250_000:
            hit_tier = "Side"
        else:
            hit_tier = "Hobby"

        # Steam rating: correlated with revenue (weak positive)
        base_rating = 0.65 + 0.25 * min(math.log1p(y1_revenue) / 20.0, 1.0)
        steam_rating = round(max(0.40, min(0.99, base_rating + rng.gauss(0, 0.08))), 2)

        # Review count: correlated with revenue
        revenue_factor = math.log1p(y1_revenue)
        base_reviews = math.exp(revenue_factor * 0.6 + rng.gauss(0, 1.5))
        review_count = max(50, int(base_reviews))

        # Price
        price = round(max(0.99, rng.gauss(15.0, 8.0)), 2)
        price = min(price, 59.99)

        # Release year
        release_year = rng.randint(2020, 2025)

        # Developer
        dev_name = rng.choice(_DEVELOPER_PREFIXES) + " " + rng.choice(_DEVELOPER_SUFFIXES)
        dev_successes = rng.randint(0, 5)
        dev_total = max(dev_successes, rng.randint(1, 10))

        # Tags
        genre_tag = genre.split()[0] if " " in genre else genre
        n_tags = rng.randint(3, 6)
        extra_tags = rng.sample(_TAG_POOL, min(n_tags, len(_TAG_POOL)))
        tags = [genre_tag, *[t for t in extra_tags if t != genre_tag]][:n_tags]

        # Quarter assignment
        quarter = quarters[i % len(quarters)]

        # Title: genre-inspired procedural name
        title_seed = _stable_seed(game_id)
        title_rng = _LCG(title_seed)
        title_adjectives = [
            "Dark",
            "Last",
            "Eternal",
            "Forgotten",
            "Lost",
            "Ancient",
            "Crimson",
            "Silent",
            "Cursed",
            "Wild",
            "Shattered",
            "Hollow",
            "Iron",
            "Mystic",
            "Neon",
            "Crystal",
            "Phantom",
            "Rogue",
        ]
        title_nouns = [
            "Depths",
            "Kingdom",
            "Legacy",
            "Realms",
            "Protocol",
            "Siege",
            "Drift",
            "Echo",
            "Frontier",
            "Haven",
            "Nexus",
            "Rift",
            "Storm",
            "Vigil",
            "Wanderer",
            "Zenith",
            "Quest",
            "Dawn",
        ]
        title = f"{title_rng.choice(title_adjectives)} {title_rng.choice(title_nouns)}"

        games.append(
            {
                "game_id": game_id,
                "title": title,
                "genre": genre,
                "developer": dev_name,
                "steam_rating": steam_rating,
                "review_count": review_count,
                "price_usd": price,
                "release_year": release_year,
                "tags": tags,
                "developer_successes": dev_successes,
                "developer_games_released": dev_total,
                "genre_q1_type": q1_type,
                "estimated_y1_revenue": round(y1_revenue, 2),
                "hit_tier": hit_tier,
                "quarter": quarter,
            }
        )

    return games


def generate_quarterly_feedback(
    training_games: list[dict[str, Any]],
    seed: int = 99,
) -> list[dict[str, Any]]:
    """Generate quarterly feedback: predicted vs actual revenue for phase transitions.

    Simulates a scenario where predictions improve over time (quarters).
    Early quarters have higher prediction error; later quarters converge.

    Args:
        training_games: Games with 'quarter' and 'estimated_y1_revenue' fields.
        seed: Random seed.

    Returns:
        List of feedback dicts with game_id, quarter, predicted_score,
        actual_revenue, and prediction_error.
    """
    rng = _LCG(seed)
    feedback: list[dict[str, Any]] = []

    # Sort quarters for progressive improvement simulation
    quarters_seen: list[str] = []
    for g in training_games:
        q = g.get("quarter", "2024Q1")
        if q not in quarters_seen:
            quarters_seen.append(q)
    quarters_seen.sort()

    # Error decreases with each quarter (simulates model improvement)
    quarter_noise: dict[str, float] = {}
    for i, q in enumerate(quarters_seen):
        quarter_noise[q] = max(0.15, 0.50 - i * 0.08)

    for g in training_games:
        actual_y1 = float(g.get("estimated_y1_revenue", 0.0))
        quarter = g.get("quarter", "2024Q1")
        noise_scale = quarter_noise.get(quarter, 0.30)

        # Predicted score: log-space prediction with quarter-dependent noise
        log_actual = math.log1p(actual_y1)
        log_predicted = log_actual + rng.gauss(0, noise_scale)
        predicted_y1 = math.expm1(max(0.0, log_predicted))

        # Selection score simulation: normalized [0, 1]
        max_rev = 200_000_000.0
        predicted_score = min(1.0, math.log1p(predicted_y1) / math.log1p(max_rev))

        feedback.append(
            {
                "game_id": g["game_id"],
                "quarter": quarter,
                "predicted_score": round(predicted_score, 4),
                "actual_revenue": round(actual_y1, 2),
                "predicted_revenue": round(predicted_y1, 2),
                "prediction_error": round(abs(actual_y1 - predicted_y1) / max(actual_y1, 1.0), 4),
            }
        )

    return feedback


def generate_quarter_scores(
    training_games: list[dict[str, Any]],
    seed: int = 77,
) -> dict[str, dict[str, list[float]]]:
    """Generate per-quarter ML and Jury score distributions for PSI computation.

    Returns:
        Dict mapping quarter -> {"ml_scores": [...], "jury_scores": [...]}.
    """
    rng = _LCG(seed)
    quarter_data: dict[str, dict[str, list[float]]] = {}

    for g in training_games:
        quarter = g.get("quarter", "2024Q1")
        if quarter not in quarter_data:
            quarter_data[quarter] = {"ml_scores": [], "jury_scores": []}

        y1 = float(g.get("estimated_y1_revenue", 0.0))

        # ML score: quantitative signal via review_count (revenue-correlated feature)
        # review_count generation: exp(0.6*log(Y1) + N(0,1.5)) — noisy revenue proxy
        review_count = float(g.get("review_count", 100))
        ml_raw = math.log1p(review_count) / 12.5 + rng.gauss(0, 0.10)
        ml_score = max(0.0, min(1.0, ml_raw))

        # Jury score: qualitative expert assessment — independent revenue view
        # In the real system, LLM judges evaluate game quality/potential,
        # producing scores correlated with revenue through a different pathway.
        steam_rating = float(g.get("steam_rating", 0.65))
        expert_base = math.log1p(y1) / 22.0
        qual_composite = 0.5 * expert_base + 0.5 * steam_rating + rng.gauss(0, 0.10)
        jury_raw = 1.0 + qual_composite * 3.5
        jury_score = max(1.0, min(4.0, jury_raw))

        quarter_data[quarter]["ml_scores"].append(round(ml_score, 4))
        quarter_data[quarter]["jury_scores"].append(round(jury_score, 4))

    return quarter_data
