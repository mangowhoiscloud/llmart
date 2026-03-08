"""T3 Human Review node: Decision Matrix, 100 -> ~30."""

from __future__ import annotations

from typing import Any

from llmart.config import DEFAULT_CONFIG, LLMARTConfig
from llmart.models.escalation import EscalateReason, EscalationResult
from llmart.pipeline.nodes._utils import CURRENT_YEAR, JURY_SCALE_MAX
from llmart.pipeline.state import GraphState


def _decision_matrix(jury_score: float, ml_percentile: float) -> str:
    """Apply the Decision Matrix (PDF slide 2 heatmap).

    ml_percentile is in [0, 1] where higher = better (e.g., 0.80 = top 20%).
    Rules (evaluated top-down):
      jury >= 3.0 (H)  & ml >= 0.50 (top half)  -> APPROVE
      jury >= 3.0 (H)  & ml >= 0.25 (top 75%)   -> REVIEW
      jury >= 2.5       & ml >= 0.80 (top 20%)   -> APPROVE
      jury >= 2.5       & ml >= 0.70 (top 30%)   -> REVIEW
      jury >= 2.0 (M)  & ml >= 0.80 (top 20%)   -> REVIEW
      else -> REJECT
    """
    if jury_score >= 3.2:
        if ml_percentile >= 0.60:
            return "APPROVE"
        if ml_percentile >= 0.30:
            return "REVIEW"
    if jury_score >= 2.5:
        if ml_percentile >= 0.80:
            return "APPROVE"
        if ml_percentile >= 0.60:
            return "REVIEW"
        if ml_percentile >= 0.30:
            return "REVIEW"
    return "REJECT"


def _check_escalation(
    game: dict[str, Any],
    config: LLMARTConfig | None = None,
) -> EscalationResult:
    """Check all 6 escalation triggers and return structured result.

    Triggers:
      1. DISAGREE: T1/T2 gap > 1.5
      2. CONTRADICTION: pass^3 disagreement
      3. BOUNDARY: selection_score near threshold (+/-5%)
      4. MULTI_DIM: 2+ dimensions disagree by >= 2.0
      5. EXTREME_DIM: single dimension delta >= 3.0
      6. LOW_COVERAGE: feature_fill_rate < 0.4
    """
    if config is None:
        config = DEFAULT_CONFIG

    reasons: list[EscalateReason] = []

    ml = game.get("ml_percentile", 0.0)
    jury = game.get("jury_score", 0.0)

    # 1. DISAGREE: T1/T2 gap
    ml_scaled = 1 + ml * (JURY_SCALE_MAX - 1)
    if abs(ml_scaled - jury) > config.escalation_gap_threshold:
        reasons.append(EscalateReason.DISAGREE)

    # 2. CONTRADICTION: pass^3 disagreement
    if not game.get("pass3_agree", True):
        reasons.append(EscalateReason.CONTRADICTION)

    # 3. BOUNDARY: selection score near threshold
    phi_ml = float(game.get("ml_percentile", 0.0))
    phi_llm = float(game.get("jury_score", 0.0)) / JURY_SCALE_MAX
    delta = float(game.get("delta_cal", 0.0))
    sel = config.phase_0_w_ml * phi_ml + config.phase_0_w_llm * phi_llm + delta
    if config.escalation_boundary_low <= sel <= config.escalation_boundary_high:
        reasons.append(EscalateReason.BOUNDARY)

    # 4. MULTI_DIM: 2+ dimensions disagree by >= 2.0
    dim_scores = game.get("dim_scores", {})
    if isinstance(dim_scores, dict) and len(dim_scores) > 0:
        dim_vals = list(dim_scores.values())
        if len(dim_vals) >= 2:
            mean_dim = sum(dim_vals) / len(dim_vals)
            big_deltas = sum(
                1 for v in dim_vals if abs(v - mean_dim) >= config.escalation_multi_dim_delta
            )
            if big_deltas >= config.escalation_multi_dim_count:
                reasons.append(EscalateReason.MULTI_DIM)

            # 5. EXTREME_DIM: single dimension delta >= 3.0
            max_delta = max(abs(v - mean_dim) for v in dim_vals)
            if max_delta >= config.escalation_extreme_dim_delta:
                reasons.append(EscalateReason.EXTREME_DIM)

    # 6. LOW_COVERAGE: feature fill rate < 0.4
    ml_features = game.get("ml_features", {})
    if isinstance(ml_features, dict) and ml_features:
        filled = sum(1 for v in ml_features.values() if v is not None and v > 0)
        fill_rate = filled / len(ml_features)
        if fill_rate < config.escalation_low_coverage_rate:
            reasons.append(EscalateReason.LOW_COVERAGE)

    return EscalationResult.from_reasons(reasons)


# DM Score weights for deterministic top_k ranking (not in SOT — supplementary)
DM_W_JURY = 0.4
DM_W_ML = 0.3
DM_W_RECENCY = 0.2
DM_W_TEAM = 0.1
DM_RECENCY_SPAN = 5.0  # years back from CURRENT_YEAR


def _dm_score(game: dict[str, Any]) -> float:
    """Composite decision-matrix score for final ranking.

    dm_score = jury*DM_W_JURY + ml*DM_W_ML + recency*DM_W_RECENCY + team*DM_W_TEAM

    This is a supplementary ranking formula not in the SOT; it provides
    a deterministic ordering for the top_k selection.
    """
    jury = game.get("jury_score", 0.0) / JURY_SCALE_MAX  # normalise to 0-1
    ml = game.get("ml_percentile", 0.0)
    year = game.get("release_year", 2024)
    recency = max(0.0, min(1.0, 1.0 - (CURRENT_YEAR - year) / DM_RECENCY_SPAN))
    team = game.get("team_factor", 0.5)
    return float(
        round(
            jury * DM_W_JURY + ml * DM_W_ML + recency * DM_W_RECENCY + team * DM_W_TEAM,
            4,
        )
    )


def human_review_node(state: GraphState) -> dict[str, Any]:
    """Apply Decision Matrix, flag escalations, keep APPROVE+REVIEW up to top_k."""
    candidates: list[dict[str, Any]] = state.get("candidates", [])
    if not candidates:
        return {
            "candidates": [],
            "stage": "human_review",
            "errors": ["human_review: 0 candidates received"],
            "stage_counts": dict(state.get("stage_counts", {})),
            "stage_timings": dict(state.get("stage_timings", {})),
        }
    top_k: int = state.get("top_k", 30)
    errors: list[str] = []

    reviewed: list[dict[str, Any]] = []
    for game in candidates:
        try:
            jury = game.get("jury_score", 0.0)
            ml = game.get("ml_percentile", 0.0)
            decision = _decision_matrix(jury, ml)
            esc_result = _check_escalation(game)
            score = _dm_score(game)

            reviewed.append(
                {
                    **game,
                    "decision": decision,
                    "dm_score": score,
                    "escalated": esc_result.should_escalate,
                    "escalation_reasons": [r.value for r in esc_result.reasons],
                    "escalation_guidance": esc_result.guidance,
                }
            )
        except Exception as exc:
            errors.append(f"human_review: {game.get('game_id', '?')}: {exc}")

    # Sort all reviewed games by dm_score, keep top_k (including REJECT for demo visibility)
    reviewed.sort(key=lambda g: g["dm_score"], reverse=True)
    kept = reviewed[:top_k]

    stage_counts = dict(state.get("stage_counts", {}))
    stage_counts["human_review"] = len(kept)
    stage_timings = dict(state.get("stage_timings", {}))

    return {
        "candidates": kept,
        "stage": "human_review",
        "errors": errors,
        "stage_counts": stage_counts,
        "stage_timings": stage_timings,
    }
