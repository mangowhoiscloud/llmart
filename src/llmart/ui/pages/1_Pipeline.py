"""Page 1: Pipeline Overview — Funnel, timing, signal distribution, KPI cards."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from llmart.ui.theme import SIGNAL_COLORS, STAGE_COLORS, inject_css

inject_css()
st.header("Pipeline Overview")


def _funnel_chart(stage_counts: dict[str, int], total_input: int) -> go.Figure:
    """Create a Plotly Funnel chart for pipeline stages."""
    labels = ["Input", *list(stage_counts.keys())]
    values = [total_input, *list(stage_counts.values())]
    colours = STAGE_COLORS[: len(labels)]

    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=values,
            marker={"color": colours[: len(labels)]},
            textinfo="value+percent initial",
        )
    )
    fig.update_layout(
        title="Pipeline Funnel",
        height=400,
        margin={"t": 40, "b": 20, "l": 20, "r": 20},
    )
    return fig


def _timing_chart(stage_timings: dict[str, float]) -> go.Figure:
    """Horizontal bar chart showing execution time per stage."""
    stages = list(stage_timings.keys())
    times = list(stage_timings.values())

    fig = go.Figure(
        go.Bar(
            x=times,
            y=stages,
            orientation="h",
            marker_color="#6366f1",
            text=[f"{t:.3f}s" for t in times],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Stage Execution Time",
        xaxis_title="Seconds",
        height=300,
        margin={"t": 40, "b": 20, "l": 120, "r": 20},
    )
    return fig


def _signal_donut(candidates: list[dict[str, Any]]) -> go.Figure:
    """Donut chart showing GREEN/YELLOW/RED distribution."""
    counts: dict[str, int] = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    for c in candidates:
        sig = c.get("signal", "")
        if sig in counts:
            counts[sig] += 1

    fig = go.Figure(
        go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            hole=0.5,
            marker={"colors": [SIGNAL_COLORS[s] for s in counts]},
        )
    )
    fig.update_layout(
        title="Signal Distribution",
        height=350,
        margin={"t": 40, "b": 20, "l": 20, "r": 20},
    )
    return fig


def _score_histogram(candidates: list[dict[str, Any]]) -> go.Figure:
    """Selection Score histogram with GREEN/YELLOW/RED zones."""
    scores = [float(c.get("selection_score", 0)) for c in candidates]

    fig = go.Figure(
        go.Histogram(
            x=scores,
            marker_color="#6366f1",
            nbinsx=20,
        )
    )
    fig.update_layout(
        title="Selection Score Distribution",
        xaxis_title="Selection Score",
        yaxis_title="Count",
        height=350,
        margin={"t": 40, "b": 40, "l": 40, "r": 20},
    )
    return fig


# Main content
if "result" not in st.session_state:
    st.info("Run the pipeline from the sidebar to see results.")
    st.stop()

result: dict[str, Any] = st.session_state["result"]
candidates: list[dict[str, Any]] = result.get("candidates", [])
input_count: int = st.session_state.get("input_count", 0)
stage_counts: dict[str, int] = result.get("stage_counts", {})
stage_timings: dict[str, float] = result.get("stage_timings", {})

# KPI metric cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Input", f"{input_count:,}")
col2.metric("Final Selected", len(candidates))

greens = sum(1 for c in candidates if c.get("signal") == "GREEN")
col3.metric("GREEN Games", greens)

avg_score = sum(float(c.get("selection_score", 0)) for c in candidates) / max(len(candidates), 1)
col4.metric("Avg Score", f"{avg_score:.4f}")

# Charts
col_left, col_right = st.columns(2)

with col_left:
    if stage_counts:
        st.plotly_chart(_funnel_chart(stage_counts, input_count), use_container_width=True)
    else:
        st.info("Stage counts not available (timing not enabled).")

    st.plotly_chart(_score_histogram(candidates), use_container_width=True)

with col_right:
    if stage_timings:
        st.plotly_chart(_timing_chart(stage_timings), use_container_width=True)
    else:
        st.info("Stage timings not available.")

    st.plotly_chart(_signal_donut(candidates), use_container_width=True)
