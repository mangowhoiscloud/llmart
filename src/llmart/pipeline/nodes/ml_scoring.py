"""T1 ML Scoring node: LambdaMART NDCG@30, 5K -> 500.

Note: This is a linear approximation of LambdaMART. A production
implementation would use a gradient-boosted tree model optimising
pairwise/listwise ranking loss (NDCG).
"""

from __future__ import annotations

import math
from typing import Any

from llmart.pipeline.nodes._utils import CURRENT_YEAR, RELEVANT_TAGS, wilson_lower_bound
from llmart.pipeline.state import GraphState

# Feature weights (linear approximation of LambdaMART leaf output)
W_WILSON = 0.25
W_REVIEWS = 0.22
W_RECENCY = 0.18
W_TAGS = 0.12
W_PRICE = 0.08
W_WILSON_X_REVIEWS = 0.08
W_RECENCY_X_TAGS = 0.07

MAX_YEAR_SPAN = 5  # years back from CURRENT_YEAR
MAX_PRICE_USD = 60.0  # price normalisation ceiling
MAX_REVIEWS = 100_000  # log1p normalisation ceiling


def _compute_ml_features(game: dict[str, Any]) -> dict[str, float]:
    """Extract and normalise features for the ML scoring model."""
    rating = game.get("steam_rating", 0.0)
    reviews = game.get("review_count", 0)
    tags = game.get("tags", [])
    price = game.get("price_usd", 0.0)
    year = game.get("release_year", CURRENT_YEAR)

    positive = int(rating * reviews)
    wilson = wilson_lower_bound(positive, reviews)
    log_reviews = math.log1p(reviews) / math.log1p(MAX_REVIEWS)
    recency = max(0.0, min(1.0, 1.0 - (CURRENT_YEAR - year) / MAX_YEAR_SPAN))
    # Tag quality: fraction of tags that are relevant (not just count)
    tag_overlap = sum(1 for t in tags if t in RELEVANT_TAGS)
    tag_quality = min(tag_overlap / max(len(tags), 1), 1.0)
    price_norm = min(price / MAX_PRICE_USD, 1.0)

    # Guard against NaN/Inf from edge-case inputs
    wilson = wilson if math.isfinite(wilson) else 0.0
    log_reviews = log_reviews if math.isfinite(log_reviews) else 0.0
    recency = recency if math.isfinite(recency) else 0.0
    tag_quality = tag_quality if math.isfinite(tag_quality) else 0.0
    price_norm = price_norm if math.isfinite(price_norm) else 0.0

    wilson_x_reviews = wilson * log_reviews
    recency_x_tags = recency * tag_quality

    return {
        "wilson": round(wilson, 4),
        "log_reviews": round(log_reviews, 4),
        "recency": round(recency, 4),
        "tag_quality": round(tag_quality, 4),
        "price_norm": round(price_norm, 4),
        "wilson_x_reviews": round(wilson_x_reviews, 4),
        "recency_x_tags": round(recency_x_tags, 4),
    }


def _ml_raw_score(features: dict[str, float]) -> float:
    """Weighted sum of ML features (linear approximation of LambdaMART).

    Weight derivation: base weights from ablation study on Steam 2022-2025 data.
    Interaction features (wilson_x_reviews, recency_x_tags) capture non-linear
    effects — high-Wilson + high-review games are disproportionately successful.
    Weights rebalanced to sum=1.0 after adding interaction terms.
    """
    return (
        W_WILSON * features["wilson"]
        + W_REVIEWS * features["log_reviews"]
        + W_RECENCY * features["recency"]
        + W_TAGS * features["tag_quality"]
        + W_PRICE * features["price_norm"]
        + W_WILSON_X_REVIEWS * features["wilson_x_reviews"]
        + W_RECENCY_X_TAGS * features["recency_x_tags"]
    )


def ml_scoring_node(state: GraphState) -> dict[str, Any]:
    """Compute ML scores and batch-relative percentile for each candidate."""
    candidates: list[dict[str, Any]] = state.get("candidates", [])
    errors: list[str] = []

    if not candidates:
        errors.append("ml_scoring: 0 candidates received from prefilter")
        return {"candidates": [], "stage": "ml_scoring", "errors": errors}

    scored: list[tuple[dict[str, Any], float, dict[str, float]]] = []
    for game in candidates:
        try:
            features = _compute_ml_features(game)
            raw = _ml_raw_score(features)
            scored.append((game, raw, features))
        except Exception as exc:
            errors.append(f"ml_scoring: {game.get('game_id', '?')}: {exc}")

    scored.sort(key=lambda t: t[1])
    n = len(scored)

    result: list[dict[str, Any]] = []
    for rank, (game, _raw, features) in enumerate(scored):
        # Weibull plotting position: avoids 0.0 and 1.0 in small batches
        percentile = (rank + 1) / (n + 1)
        result.append(
            {
                **game,
                "ml_percentile": round(percentile, 4),
                "ml_features": features,
            }
        )

    result.sort(key=lambda g: g["ml_percentile"], reverse=True)

    stage_counts = dict(state.get("stage_counts", {}))
    stage_counts["ml_scoring"] = len(result)
    stage_timings = dict(state.get("stage_timings", {}))

    return {
        "candidates": result,
        "stage": "ml_scoring",
        "errors": errors,
        "stage_counts": stage_counts,
        "stage_timings": stage_timings,
    }


# Revenue-to-grade mapping for NDCG (bounded graded relevance)
# Tier boundaries aligned with ground_truth.RevenueTier: Hobby < $250K, Side < $2M, Hit < $20M
_REVENUE_GRADES = [
    (20_000_000, 3),  # Mega: Q1 >= $20M → grade 3
    (2_000_000, 2),  # Hit: Q1 >= $2M → grade 2
    (250_000, 1),  # Side: Q1 >= $250K → grade 1
    (0, 0),  # Hobby: Q1 < $250K → grade 0
]


def _revenue_to_grade(revenue: float) -> int:
    """Map raw Q1 revenue to bounded relevance grade (0-3)."""
    for threshold, grade in _REVENUE_GRADES:
        if revenue >= threshold:
            return grade
    return 0


def compute_ndcg_at_k(
    ranked: list[dict[str, Any]], relevance_key: str = "q1_p50", k: int = 30
) -> float:
    """Compute NDCG@k using graded relevance (tier-based, 0-3).

    Uses bounded relevance grades instead of raw revenue to prevent
    mega-hits from dominating the metric. Grade mapping:
      Mega (>=$2M) = 3, Hit (>=$250K) = 2, Side (>=$50K) = 1, Hobby = 0

    DCG = sum(rel[i] / log2(i+2)); NDCG = DCG / IDCG.
    """
    top = ranked[:k]
    if not top:
        return 0.0

    rels = [_revenue_to_grade(float(g.get(relevance_key, 0.0))) for g in top]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))

    # IDCG: all candidates sorted by graded relevance
    all_rels = sorted(
        (_revenue_to_grade(float(g.get(relevance_key, 0.0))) for g in ranked),
        reverse=True,
    )[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(all_rels))

    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)
