"""Page 2: Game Detail — interactive table, radar charts, value breakdown."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from llmart.ui.theme import fmt_money, inject_css, signal_badge

inject_css()
st.header("Game Detail")

DIMENSIONS = ["gameplay", "innovation", "monetization", "polish", "narrative"]


def _radar_chart(dim_scores: dict[str, Any], title: str) -> go.Figure:
    """5-Dim Radar chart (Plotly Scatterpolar)."""
    dims = DIMENSIONS
    values = [float(dim_scores.get(d, 0)) for d in dims]
    values.append(values[0])  # close the polygon

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=[*dims, dims[0]],
            fill="toself",
            marker_color="#6366f1",
        )
    )
    fig.update_layout(
        title=title,
        polar={"radialaxis": {"range": [0, 4], "dtick": 1}},
        height=350,
        margin={"t": 40, "b": 20, "l": 60, "r": 60},
    )
    return fig


def _value_breakdown_chart(candidate: dict[str, Any]) -> go.Figure:
    """Value decomposition waterfall: NPV - CAC - UA - LiveOps - VaR."""
    npv = float(candidate.get("npv_3y", 0))
    cac = float(candidate.get("cac", 0))
    ua = float(candidate.get("ua_cost", 0))
    liveops = float(candidate.get("liveops_cost", 0))
    var_cost = float(candidate.get("var_5pct", 0)) * 0.2
    total = float(candidate.get("value_total", 0))

    fig = go.Figure(
        go.Waterfall(
            name="Value",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "relative", "total"],
            x=["NPV_3Y", "CAC", "UA", "LiveOps", "0.2*VaR", "Value"],
            y=[npv, -cac, -ua, -liveops, -var_cost, total],
            text=[
                fmt_money(npv),
                fmt_money(-cac),
                fmt_money(-ua),
                fmt_money(-liveops),
                fmt_money(-var_cost),
                fmt_money(total),
            ],
            textposition="outside",
            connector={"line": {"color": "#64748b"}},
        )
    )
    fig.update_layout(
        title="Value Decomposition",
        height=400,
        margin={"t": 40, "b": 20, "l": 40, "r": 20},
    )
    return fig


def _ml_features_chart(candidate: dict[str, Any]) -> go.Figure:
    """Bar chart of ML feature scores."""
    features = candidate.get("ml_features", {})
    if not isinstance(features, dict) or not features:
        # Fallback: use common ML-related fields
        features = {
            "steam_rating": float(candidate.get("steam_rating", 0)),
            "review_count_log": float(candidate.get("review_count", 0)),
            "price_usd": float(candidate.get("price_usd", 0)),
            "ml_percentile": float(candidate.get("ml_percentile", 0)),
        }

    fig = go.Figure(
        go.Bar(
            x=list(features.values()),
            y=list(features.keys()),
            orientation="h",
            marker_color="#8b5cf6",
        )
    )
    fig.update_layout(
        title="ML Features",
        height=300,
        margin={"t": 40, "b": 20, "l": 120, "r": 20},
    )
    return fig


# Main content
if "result" not in st.session_state:
    st.info("Run the pipeline from the sidebar to see results.")
    st.stop()

result: dict[str, Any] = st.session_state["result"]
candidates: list[dict[str, Any]] = result.get("candidates", [])

if not candidates:
    st.warning("No candidates in pipeline results.")
    st.stop()

# Filter bar
st.subheader("Filters")
fcol1, fcol2, fcol3, fcol4 = st.columns(4)

with fcol1:
    all_signals = ["GREEN", "YELLOW", "RED"]
    signals = st.multiselect("Signal", all_signals, default=all_signals)
with fcol2:
    all_genres = sorted({c.get("genre", "?") for c in candidates})
    genres = st.multiselect("Genre", all_genres, default=all_genres)
with fcol3:
    all_decisions = sorted({c.get("decision", "?") for c in candidates})
    decisions = st.multiselect("Decision", all_decisions, default=all_decisions)
with fcol4:
    score_range = st.slider(
        "Score Range",
        0.0,
        1.0,
        (0.0, 1.0),
        step=0.01,
    )

# Apply filters
filtered = [
    c
    for c in candidates
    if c.get("signal", "?") in signals
    and c.get("genre", "?") in genres
    and c.get("decision", "?") in decisions
    and score_range[0] <= float(c.get("selection_score", 0)) <= score_range[1]
]

# Interactive data table
rows = []
for c in filtered:
    rows.append(
        {
            "Title": c.get("title", "?"),
            "Genre": c.get("genre", "?"),
            "Score": round(float(c.get("selection_score", 0)), 4),
            "Signal": signal_badge(c.get("signal", "?")),
            "Decision": c.get("decision", "?"),
            "NPV_3Y": fmt_money(float(c.get("npv_3y", 0))),
            "Value": fmt_money(float(c.get("value_total", 0))),
            "Escalated": c.get("escalated", False),
        }
    )

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, height=400)

# Game detail expanders
st.subheader("Game Details")
for c in filtered:
    title = c.get("title", "?")
    signal = c.get("signal", "?")
    with st.expander(f"{signal_badge(signal)} {title} — Score: {c.get('selection_score', 0):.4f}"):
        col_radar, col_value = st.columns(2)

        with col_radar:
            dim_scores = c.get("dim_scores", {})
            if isinstance(dim_scores, dict) and dim_scores:
                st.plotly_chart(
                    _radar_chart(dim_scores, f"{title} — 5-Dim Scores"),
                    use_container_width=True,
                )

        with col_value:
            st.plotly_chart(
                _value_breakdown_chart(c),
                use_container_width=True,
            )

        # Escalation panel
        reasons = c.get("escalation_reasons", [])
        guidance = c.get("escalation_guidance", [])
        if reasons:
            st.warning("**Escalation Triggers**")
            for r, g in zip(reasons, guidance, strict=False):
                st.markdown(f"- **{r}**: {g}")
        else:
            st.success("No escalation triggers")

        # ML Features
        st.plotly_chart(_ml_features_chart(c), use_container_width=True)
