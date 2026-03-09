"""Page 4: Retrain — quarterly retrain controls and visualization."""

from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from llmart.ui.theme import inject_css

inject_css()
st.header("Quarterly Retrain")


def _run_retrain(n_games: int, seed: int) -> dict[str, Any]:
    """Execute retrain cycle and collect results."""
    from collections import Counter

    from llmart.data.synthetic import (
        generate_quarter_scores,
        generate_quarterly_feedback,
        generate_training_dataset,
    )
    from llmart.models.percentile import PercentileRankManager
    from llmart.models.quantile import QuantileRegressor, extract_features
    from llmart.monitoring.metrics import (
        check_pass_condition,
        classify_metric,
        compute_validation_suite,
    )
    from llmart.phase.controller import PhaseController
    from llmart.phase.ridge_learner import Phase2WeightLearner, _spearman_rho

    games = generate_training_dataset(n=n_games, seed=seed)
    feedback = generate_quarterly_feedback(games, seed=seed + 1)
    quarter_scores = generate_quarter_scores(games, seed=seed + 2)
    quarters = sorted(quarter_scores.keys())

    # Per-game scores mapping
    quarter_idx: dict[str, int] = {}
    per_game: list[dict[str, float]] = []
    for g in games:
        q = g.get("quarter", "2024Q1")
        idx = quarter_idx.get(q, 0)
        qd = quarter_scores.get(q, {"ml_scores": [0.5], "jury_scores": [2.0]})
        ml = qd["ml_scores"][idx] if idx < len(qd["ml_scores"]) else 0.5
        jury = qd["jury_scores"][idx] if idx < len(qd["jury_scores"]) else 2.0
        per_game.append({"ml": ml, "jury": jury})
        quarter_idx[q] = idx + 1

    # Ridge LOOCV
    ml_norms = [pg["ml"] for pg in per_game]
    jury_norms = [pg["jury"] / 4.0 for pg in per_game]
    y1_log = [math.log1p(float(g.get("estimated_y1_revenue", 1.0))) for g in games]

    learner = Phase2WeightLearner()
    learner.fit(ml_norms, jury_norms, y1_log)
    ridge_result = learner.get_weights()

    # Phase transitions
    controller = PhaseController(initial_phase=0)
    phase_history: list[dict[str, Any]] = []

    for q in quarters:
        q_feedback = [f for f in feedback if f["quarter"] == q]
        if not q_feedback:
            continue
        pred_scores = [f["predicted_score"] for f in q_feedback]
        actual_revs = [f["actual_revenue"] for f in q_feedback]
        rho = _spearman_rho(pred_scores, actual_revs)
        cumulative_n = sum(1 for g in games if g.get("quarter", "") <= q)

        for f in q_feedback:
            controller.get_weights(n_samples=cumulative_n, agreement_delta=f["prediction_error"])

        decision = controller.evaluate_promotion(
            n=cumulative_n,
            rho=rho,
            w_std=controller.weight_std,
        )
        old_phase = controller.current_phase
        controller.apply_promotion(decision)
        phase_history.append(
            {
                "quarter": q,
                "n_samples": cumulative_n,
                "rho": round(rho, 4),
                "w_std": round(controller.weight_std, 4),
                "decision": decision,
                "phase_from": old_phase,
                "phase_to": controller.current_phase,
            }
        )

    # PSI drift
    prm = PercentileRankManager()
    for q in quarters:
        qd = quarter_scores[q]
        prm.update_quarter(q, qd["ml_scores"], qd["jury_scores"])

    psi_data: list[dict[str, Any]] = []
    available_q = [q for q in quarters if q in prm.quarters]
    for i in range(len(available_q) - 1):
        q1, q2 = available_q[i], available_q[i + 1]
        psi_val = prm.compute_psi(q1, q2, "ml")
        psi_data.append({"pair": f"{q1} vs {q2}", "psi": round(psi_val, 4)})

    # Validation suite
    q1_shares: dict[str, float] = {"front-loaded": 0.55, "balanced": 0.45, "live-service": 0.35}
    features = [extract_features(g) for g in games]
    q1_revenues = [
        float(g.get("estimated_y1_revenue", 0.0))
        * q1_shares.get(str(g.get("genre_q1_type", "balanced")), 0.50)
        for g in games
    ]

    qr = QuantileRegressor()
    qr.fit(features, q1_revenues)

    validation: dict[str, Any] = {}
    if qr.is_fitted:
        sel_scores = [
            math.log1p(qr.predict(f).q1_p50) / math.log1p(200_000_000.0) for f in features
        ]
        metrics = compute_validation_suite(sel_scores, q1_revenues, k=30)
        validation = {
            name: {"value": round(val, 4), "grade": classify_metric(name, val)}
            for name, val in metrics.items()
        }
        validation["passed"] = check_pass_condition(metrics)

    # Tier distribution
    tier_dist = dict(Counter(g["hit_tier"] for g in games))

    return {
        "ridge": ridge_result,
        "phase_history": phase_history,
        "psi_data": psi_data,
        "validation": validation,
        "tier_distribution": tier_dist,
        "n_games": len(games),
        "n_quarters": len(quarters),
        "final_phase": controller.current_phase,
    }


# Controls
with st.sidebar:
    st.subheader("Retrain Settings")
    n_games = st.number_input("Synthetic Games", 100, 2000, 500, step=100)
    retrain_seed = st.number_input("Random Seed", 1, 9999, 42)
    retrain_btn = st.button("Run Retrain", type="primary")

if retrain_btn:
    with st.spinner("Running quarterly retrain cycle..."):
        retrain_result = _run_retrain(n_games, retrain_seed)
    st.session_state["retrain_result"] = retrain_result

if "retrain_result" not in st.session_state:
    st.info("Configure retrain settings in the sidebar and click 'Run Retrain'.")
    st.stop()

rr: dict[str, Any] = st.session_state["retrain_result"]

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", rr["n_games"])
c2.metric("Quarters", rr["n_quarters"])
c3.metric("Final Phase", rr["final_phase"])
ridge = rr["ridge"]
c4.metric("w_ml (learned)", f"{ridge['w_ml']}")

# Phase transition visualization
st.subheader("Phase Transition History")
phase_hist = rr["phase_history"]
if phase_hist:
    fig_phase = go.Figure()
    fig_phase.add_trace(
        go.Scatter(
            x=[p["quarter"] for p in phase_hist],
            y=[p["phase_to"] for p in phase_hist],
            mode="lines+markers",
            name="Phase",
            line={"color": "#6366f1", "width": 3},
            marker={"size": 10},
        )
    )
    fig_phase.add_trace(
        go.Scatter(
            x=[p["quarter"] for p in phase_hist],
            y=[p["rho"] * 3 for p in phase_hist],  # scale rho to phase range
            mode="lines+markers",
            name="Spearman rho (scaled)",
            line={"color": "#22c55e", "dash": "dash"},
            yaxis="y2",
        )
    )
    fig_phase.update_layout(
        title="Phase Transition & Correlation",
        yaxis={"title": "Phase", "dtick": 1},
        yaxis2={"title": "Spearman rho", "overlaying": "y", "side": "right", "range": [0, 3]},
        height=400,
    )
    st.plotly_chart(fig_phase, use_container_width=True)

# Weight learning
st.subheader("Weight Learning")
ridge_data = rr["ridge"]
rcol1, rcol2, rcol3 = st.columns(3)
rcol1.metric("w_ml", str(ridge_data["w_ml"]))
rcol2.metric("w_jury", str(ridge_data["w_jury"]))
rcol3.metric("Lambda", str(ridge_data["lambda"]))
if ridge_data.get("is_fallback"):
    st.warning(f"Fallback active: {ridge_data.get('fallback_reason', '?')}")

# Validation Suite
st.subheader("Validation Suite")
val = rr.get("validation", {})
if val:
    passed = val.pop("passed", False)
    if passed:
        st.success("Validation PASSED (4+ metrics at Conditional+)")
    else:
        st.error("Validation FAILED")

    val_cols = st.columns(3)
    for i, (name, info) in enumerate(val.items()):
        col = val_cols[i % 3]
        grade = info["grade"]
        color_map = {"strong": "green", "conditional": "orange", "fail": "red"}
        col.metric(name, f"{info['value']:.4f}", delta=grade)

# PSI drift heatmap
st.subheader("PSI Drift")
psi_data = rr["psi_data"]
if psi_data:
    fig_psi = go.Figure(
        go.Bar(
            x=[p["pair"] for p in psi_data],
            y=[p["psi"] for p in psi_data],
            marker_color=[
                "#22c55e" if p["psi"] < 0.10 else ("#eab308" if p["psi"] < 0.25 else "#ef4444")
                for p in psi_data
            ],
        )
    )
    fig_psi.add_hline(y=0.10, line_dash="dash", line_color="#eab308", annotation_text="Warning")
    fig_psi.add_hline(y=0.25, line_dash="dash", line_color="#ef4444", annotation_text="Drift")
    fig_psi.update_layout(title="PSI Between Quarters", height=350)
    st.plotly_chart(fig_psi, use_container_width=True)

# Tier distribution stacked bar
st.subheader("Tier Distribution")
tier_dist = rr.get("tier_distribution", {})
if tier_dist:
    tier_order = ["Mega", "Hit", "Side", "Hobby"]
    tier_colors = {"Mega": "#22c55e", "Hit": "#6366f1", "Side": "#eab308", "Hobby": "#94a3b8"}
    fig_tier = go.Figure()
    for tier in tier_order:
        count = tier_dist.get(tier, 0)
        fig_tier.add_trace(
            go.Bar(
                x=[tier],
                y=[count],
                name=tier,
                marker_color=tier_colors.get(tier, "#64748b"),
            )
        )
    fig_tier.update_layout(title="Training Data Tier Distribution", barmode="stack", height=350)
    st.plotly_chart(fig_tier, use_container_width=True)
