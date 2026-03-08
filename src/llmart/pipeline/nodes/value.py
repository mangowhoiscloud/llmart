"""Value Inference + Signal node: NPV_3Y -> GREEN/YELLOW/RED."""

from __future__ import annotations

import logging
from typing import Any

from llmart.models.scoring import SelectionScore
from llmart.models.value import ValueInference
from llmart.phase.controller import PhaseController
from llmart.phase.ridge_learner import Phase2WeightLearner
from llmart.pipeline.nodes._utils import JURY_SCALE_MAX
from llmart.pipeline.nodes.ml_scoring import compute_ndcg_at_k
from llmart.pipeline.state import GraphState

logger = logging.getLogger(__name__)

# Genre-differentiated cost ratios (SOT: front-loaded / balanced / live-service)
COST_RATIOS: dict[str, dict[str, float]] = {
    "front-loaded": {"cac": 0.08, "ua": 0.05, "liveops": 0.01, "var_base": 0.30},
    "balanced": {"cac": 0.05, "ua": 0.03, "liveops": 0.03, "var_base": 0.35},
    "live-service": {"cac": 0.03, "ua": 0.02, "liveops": 0.08, "var_base": 0.40},
}
DEFAULT_COST_KEY = "balanced"

# Genre risk premium for VaR computation
GENRE_RISK_PREMIUM: dict[str, float] = {
    "front-loaded": 0.15,
    "balanced": 0.10,
    "live-service": 0.20,
}


def _compute_npv_3y(
    q1_p50: float,
    r_genre: float,
    ltv_mult: float,
    discount_rate: float = 0.15,
) -> float:
    """E[NPV_3Y] = Q1_P50 / r(genre) * LTV_mult * sum(1/(1+r)^t, t=1..3)."""
    if r_genre <= 0:
        return 0.0
    annuity = sum(1 / (1 + discount_rate) ** t for t in range(1, 4))
    return round(q1_p50 / r_genre * ltv_mult * annuity, 2)


def _try_ridge_weights(
    training_data: list[dict[str, Any]],
) -> tuple[float, float] | None:
    """Attempt Ridge-learned weights from training data (Phase 2+).

    Returns (w_ml, w_llm) if learning succeeds, None on fallback.
    """
    if len(training_data) < 10:
        return None

    ml_norms = [float(g.get("ml_percentile", 0.5)) for g in training_data]
    jury_norms = [float(g.get("jury_score", 2.0)) / JURY_SCALE_MAX for g in training_data]
    # Use log(Q1 revenue) as target for weight learning
    import math

    y1_log = [math.log1p(float(g.get("q1_p50", 1.0))) for g in training_data]

    learner = Phase2WeightLearner()
    learner.fit(ml_norms, jury_norms, y1_log)
    result = learner.get_weights()

    if result["is_fallback"]:
        logger.info("Ridge learner fallback: %s", result["fallback_reason"])
        return None

    w_ml = result["w_ml"]
    w_jury = result["w_jury"]
    if not isinstance(w_ml, (int, float)) or not isinstance(w_jury, (int, float)):
        return None
    return (float(w_ml), float(w_jury))


def value_node(state: GraphState) -> dict[str, Any]:
    """Compute Selection Score, NPV, Value, and Signal for each candidate."""
    candidates: list[dict[str, Any]] = state.get("candidates", [])
    errors: list[str] = []

    # Phase-aware weights: use controller if Phase 1+
    phase = state.get("phase", 0)
    n_historical = state.get("n_historical", 0)
    controller: PhaseController | None = None
    ridge_weights: tuple[float, float] | None = None

    if phase >= 2:
        # Phase 2+: attempt Ridge-learned weights from training data
        training_data: list[dict[str, Any]] = state.get("training_data", [])
        if training_data:
            ridge_weights = _try_ridge_weights(training_data)
            if ridge_weights:
                logger.info("Phase 2 Ridge weights: w_ml=%.4f, w_llm=%.4f", *ridge_weights)

    if phase > 0 and ridge_weights is None:
        controller = PhaseController(initial_phase=phase)

    result: list[dict[str, Any]] = []
    for game in candidates:
        try:
            # Selection Score: S = w_ml * Phi_ml + w_llm * Phi_llm + delta_cal
            phi_ml = game.get("ml_percentile", 0.0)
            phi_llm = game.get("jury_score", 0.0) / JURY_SCALE_MAX
            delta_cal = game.get("delta_cal", 0.0)

            if ridge_weights is not None:
                # Phase 2+: use Ridge-learned weights directly
                ss = SelectionScore(
                    phi_ml=phi_ml,
                    phi_llm=phi_llm,
                    delta_cal=delta_cal,
                    w_ml=ridge_weights[0],
                    w_llm=ridge_weights[1],
                    phase=phase,
                )
            elif controller is not None:
                agreement_delta = abs(
                    (1 + phi_ml * (JURY_SCALE_MAX - 1)) - game.get("jury_score", 0.0)
                )
                ss = SelectionScore.from_phase(
                    phi_ml=phi_ml,
                    phi_llm=phi_llm,
                    delta_cal=delta_cal,
                    controller=controller,
                    n_samples=n_historical,
                    agreement_delta=agreement_delta,
                )
            else:
                ss = SelectionScore(phi_ml=phi_ml, phi_llm=phi_llm, delta_cal=delta_cal)

            # NPV — use per-genre discount_rate if available.
            # Fallback: derive P50/P25 from estimated_q1_revenue when not provided.
            est_q1 = float(game.get("estimated_q1_revenue", 0.0) or 0.0)
            q1_p50 = float(game.get("q1_p50", 0.0) or 0.0) or est_q1
            q1_p25 = float(game.get("q1_p25", 0.0) or 0.0) or est_q1 * 0.6
            r_genre = game.get("r_genre", 0.20)
            ltv_mult = game.get("ltv_mult", 2.0)
            discount_rate = game.get("discount_rate", 0.15)
            npv = _compute_npv_3y(q1_p50, r_genre, ltv_mult, discount_rate)

            # Genre-differentiated cost structure
            genre_q1_type = game.get("genre_q1_type", DEFAULT_COST_KEY)
            cost = COST_RATIOS.get(genre_q1_type, COST_RATIOS[DEFAULT_COST_KEY])

            cac = npv * cost["cac"]
            ua_cost = npv * cost["ua"]
            liveops_cost = npv * cost["liveops"]

            # Revenue-uncertainty-based VaR
            revenue_spread = max(0.0, q1_p50 - q1_p25)
            uncertainty_ratio = revenue_spread / max(q1_p50, 1.0)
            risk_premium = GENRE_RISK_PREMIUM.get(genre_q1_type, 0.15)
            var_5pct = npv * (uncertainty_ratio + risk_premium)

            vi = ValueInference(
                npv_3y=npv,
                cac=cac,
                ua=ua_cost,
                live_ops=liveops_cost,
                var_5pct=var_5pct,
                q1_p25=q1_p25,
                q1_p50=q1_p50,
            )

            result.append(
                {
                    **game,
                    "selection_score": round(ss.score, 4),
                    "w_ml": ss.w_ml,
                    "w_llm": ss.w_llm,
                    "phase": ss.phase,
                    "npv_3y": npv,
                    "value_total": round(vi.value, 2),
                    "signal": vi.signal.value,
                }
            )
        except Exception as exc:
            errors.append(f"value: {game.get('game_id', '?')}: {exc}")

    # NDCG@30 — computed here where q1_p50 is available (not in ml_scoring)
    ndcg_at_30 = compute_ndcg_at_k(result, relevance_key="q1_p50", k=30)

    # Phase promotion evaluation: auto-promote based on accumulated metrics
    promotion_result: str | None = None
    if controller is not None and result:
        from llmart.monitoring.metrics import compute_spearman_rho

        ml_vals = [g.get("ml_percentile", 0.0) for g in result]
        q1_vals = [g.get("q1_p50", 0.0) for g in result]
        rho = compute_spearman_rho(ml_vals, q1_vals) if len(ml_vals) >= 5 else 0.0
        w_std = controller.weight_std
        decision = controller.evaluate_promotion(n=n_historical, rho=rho, w_std=w_std)
        if decision != "hold":
            new_phase = controller.apply_promotion(decision)
            promotion_result = f"{decision} -> Phase {new_phase}"
            logger.info(
                "Phase %s: %s (rho=%.3f, w_std=%.4f, n=%d)",
                decision,
                promotion_result,
                rho,
                w_std,
                n_historical,
            )

    stage_counts = dict(state.get("stage_counts", {}))
    stage_counts["value"] = len(result)
    stage_timings = dict(state.get("stage_timings", {}))

    monitoring_data: dict[str, Any] = {"ndcg_at_30": ndcg_at_30}
    if promotion_result:
        monitoring_data["phase_promotion"] = promotion_result

    return {
        "candidates": result,
        "stage": "value",
        "errors": errors,
        "monitoring": monitoring_data,
        "stage_counts": stage_counts,
        "stage_timings": stage_timings,
    }
