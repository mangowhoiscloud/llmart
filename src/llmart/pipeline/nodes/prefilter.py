"""Pre-filter 2L node: 600K -> 5K candidates."""

from __future__ import annotations

import math
from typing import Any

from pydantic import ValidationError

from llmart.models.game import Game
from llmart.pipeline.nodes._utils import RELEVANT_TAGS, wilson_lower_bound
from llmart.pipeline.state import GraphState

# L1 hard-cut thresholds
MIN_RELEASE_YEAR = 2022
MIN_REVIEW_COUNT = 200  # PDF slide 1: Reviews ≥ 200
MIN_STEAM_RATING = 0.65  # PDF slide 1: Rating ≥ 0.65

# L2 soft-score weights
W_WILSON = 0.50
W_LOG_REVIEWS = 0.30
W_TAG_RELEVANCE = 0.20
MAX_REVIEWS = 100_000  # log1p normalisation ceiling


def _passes_hard_cut(game: dict[str, Any]) -> bool:
    """L1: reject games that fail any hard threshold."""
    return bool(
        game.get("release_year", 0) >= MIN_RELEASE_YEAR
        and game.get("review_count", 0) >= MIN_REVIEW_COUNT
        and game.get("steam_rating", 0.0) >= MIN_STEAM_RATING
    )


def _l2_soft_score(game: dict[str, Any]) -> float:
    """L2: composite soft-score for ranking survivors."""
    rating = game.get("steam_rating", 0.0)
    reviews = game.get("review_count", 0)
    tags = game.get("tags", [])

    positive = int(rating * reviews)
    wilson = wilson_lower_bound(positive, reviews)
    log_reviews = min(math.log1p(reviews) / math.log1p(MAX_REVIEWS), 1.0)
    tag_overlap = sum(1 for t in tags if t in RELEVANT_TAGS)
    tag_relevance = min(tag_overlap / max(len(tags), 1), 1.0)

    return W_WILSON * wilson + W_LOG_REVIEWS * log_reviews + W_TAG_RELEVANCE * tag_relevance


def prefilter_node(state: GraphState) -> dict[str, Any]:
    """Apply Pre-filter 2L: L1 hard-cut then L2 soft-score."""
    candidates: list[dict[str, Any]] = state.get("candidates", [])
    total_input = len(candidates)
    errors: list[str] = []

    # Validate input schema via Game model (M3: use the Pydantic model)
    validated: list[dict[str, Any]] = []
    for game in candidates:
        try:
            Game(**game)
            validated.append(game)
        except ValidationError as exc:
            errors.append(f"prefilter: invalid input {game.get('game_id', '?')}: {exc}")

    survivors: list[dict[str, Any]] = []
    for game in validated:
        try:
            if _passes_hard_cut(game):
                score = _l2_soft_score(game)
                survivors.append({**game, "l2_score": round(score, 4)})
        except Exception as exc:
            errors.append(f"prefilter: {game.get('game_id', '?')}: {exc}")

    survivors.sort(key=lambda g: g["l2_score"], reverse=True)

    # Recall@99 hook: measure how many known hits survived prefilter
    known_hits: list[str] = state.get("known_hits", [])
    recall_at_99 = _compute_recall(survivors, known_hits)

    stage_counts: dict[str, int] = {"prefilter": len(survivors)}
    result: dict[str, Any] = {
        "candidates": survivors,
        "stage": "prefilter",
        "total_input": total_input,
        "errors": errors,
        "stage_counts": stage_counts,
    }
    if recall_at_99 is not None:
        result["monitoring"] = {"recall_at_99": recall_at_99}
    return result


def _compute_recall(survivors: list[dict[str, Any]], known_hits: list[str]) -> float | None:
    """Compute recall of known hit games among survivors."""
    if not known_hits:
        return None
    ids = {g["game_id"] for g in survivors}
    return sum(1 for h in known_hits if h in ids) / len(known_hits)
