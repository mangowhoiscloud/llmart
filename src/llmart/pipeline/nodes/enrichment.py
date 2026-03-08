"""Data Enrichment node."""

from __future__ import annotations

from typing import Any

from llmart.pipeline.nodes._utils import developer_wilson_score
from llmart.pipeline.state import GraphState

DEFAULT_GENRE_PARAMS: dict[str, Any] = {
    "r_genre": 0.20,
    "ltv_mult": 2.0,
    "discount_rate": 0.15,
}

# Boxleiter Revenue Estimation Constants (VG Insights, R^2=0.78)
SALES_MULT = 30  # review -> sales multiplier (post-2020 median)
STEAM_NET = 0.70  # developer net after Steam 30% commission
Q1_P25_RATIO = 0.6  # P25 = P50 * 0.6 (downside scenario)
Q1_P75_RATIO = 1.5  # P75 = P50 * 1.5 (upside scenario)

# Genre-differentiated Q1 share of Y1 revenue (aligned with synthetic.py)
# front-loaded: 55% of Y1 in Q1 (big launch spike, fast decay)
# balanced: 45% of Y1 in Q1 (moderate launch, sustained sales)
# live-service: 35% of Y1 in Q1 (slow build, long-tail revenue)
Q1_SHARE_BY_TYPE: dict[str, float] = {
    "front-loaded": 0.55,
    "balanced": 0.45,
    "live-service": 0.35,
}
Q1_SHARE_DEFAULT = 0.45  # fallback if q1_ratio_type unknown

# Y1 share of lifetime revenue (conservative multiplier)
Y1_LIFETIME_RATIO = 0.15  # Y1 ≈ 15% of lifetime revenue

F2P_ARPU = 1.20  # Conservative F2P ARPU (in-app + season pass)


def _estimate_q1(
    review_count: int, price_usd: float, q1_type: str = "balanced"
) -> tuple[float, float, float]:
    """Estimate Q1 revenue using Boxleiter method (VG Insights validated).

    Revenue chain: reviews * SALES_MULT * price * STEAM_NET = estimated Y1 revenue
    Q1 revenue = Y1 * Q1_SHARE_BY_TYPE[q1_type]

    For F2P games: uses F2P_ARPU instead of price * STEAM_NET.
    P25 = P50 * Q1_P25_RATIO (downside scenario)
    P75 = P50 * Q1_P75_RATIO (upside scenario)
    """
    q1_share = Q1_SHARE_BY_TYPE.get(q1_type, Q1_SHARE_DEFAULT)

    if price_usd <= 0:
        # F2P: estimate from review count alone (average F2P ARPU model)
        estimated_players = review_count * SALES_MULT
        y1_revenue = estimated_players * F2P_ARPU
    else:
        y1_revenue = review_count * SALES_MULT * price_usd * STEAM_NET

    p50 = y1_revenue * q1_share
    p25 = p50 * Q1_P25_RATIO
    p75 = p50 * Q1_P75_RATIO
    return round(p25, 2), round(p50, 2), round(p75, 2)


def enrichment_node(state: GraphState) -> dict[str, Any]:
    """Enrich candidates with genre params and Q1 revenue estimates."""
    candidates: list[dict[str, Any]] = state.get("candidates", [])
    if not candidates:
        return {
            "candidates": [],
            "stage": "enrichment",
            "errors": ["enrichment: 0 candidates received"],
            "stage_counts": dict(state.get("stage_counts", {})),
            "stage_timings": dict(state.get("stage_timings", {})),
        }
    genre_params: dict[str, dict[str, Any]] = state.get("genre_params", {})
    errors: list[str] = []

    result: list[dict[str, Any]] = []
    for game in candidates:
        try:
            genre = game.get("genre", "")
            params = genre_params.get(genre, DEFAULT_GENRE_PARAMS)

            q1_type = params.get("q1_ratio_type", "balanced")
            q1_p25, q1_p50, q1_p75 = _estimate_q1(
                game.get("review_count", 0),
                game.get("price_usd", 0.0),
                q1_type=q1_type,
            )

            if q1_p50 <= 0:
                errors.append(f"enrichment: zero Q1 estimate for {game.get('game_id', '?')}")

            # Developer Wilson Score for team factor
            dev_successes = game.get("developer_successes", 0)
            dev_total = game.get("developer_games_released", 0)
            team_score = developer_wilson_score(dev_successes, dev_total) if dev_total > 0 else 0.5

            result.append(
                {
                    **game,
                    "r_genre": params.get("r_genre", DEFAULT_GENRE_PARAMS["r_genre"]),
                    "ltv_mult": params.get("ltv_mult", DEFAULT_GENRE_PARAMS["ltv_mult"]),
                    "discount_rate": params.get(
                        "discount_rate", DEFAULT_GENRE_PARAMS["discount_rate"]
                    ),
                    "q1_p25": q1_p25,
                    "q1_p50": q1_p50,
                    "q1_p75": q1_p75,
                    "team_factor": team_score,
                }
            )
        except Exception as exc:
            errors.append(f"enrichment: {game.get('game_id', '?')}: {exc}")

    stage_counts = dict(state.get("stage_counts", {}))
    stage_counts["enrichment"] = len(result)
    stage_timings = dict(state.get("stage_timings", {}))

    return {
        "candidates": result,
        "stage": "enrichment",
        "errors": errors,
        "stage_counts": stage_counts,
        "stage_timings": stage_timings,
    }
