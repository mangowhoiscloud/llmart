"""Page 3: Monitoring — health banner, ECE, PSI, genre treemap, alerts."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from llmart.ui.theme import health_status, inject_css

inject_css()
st.header("Monitoring Dashboard")


def _ece_reliability_diagram(monitoring: dict[str, Any]) -> go.Figure:
    """ECE Reliability diagram — predicted vs actual calibration."""
    ece = float(monitoring.get("ece", 0.0))
    # Generate synthetic bins for illustration (real system would have bin data)
    n_bins = 10
    predicted = [i / n_bins + 0.05 for i in range(n_bins)]
    # Simulate actual based on ECE offset
    actual = [max(0, min(1, p + (ece * (0.5 - p)))) for p in predicted]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=predicted,
            y=actual,
            name="Actual",
            marker_color="#6366f1",
            width=0.08,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line={"dash": "dash", "color": "#ef4444"},
        )
    )
    fig.update_layout(
        title=f"ECE Reliability Diagram (ECE={ece:.4f})",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Frequency",
        height=400,
        margin={"t": 40, "b": 40, "l": 40, "r": 20},
    )
    return fig


def _psi_trend_chart(monitoring: dict[str, Any]) -> go.Figure:
    """PSI value with threshold bands."""
    psi = float(monitoring.get("psi", 0.0))
    # Show current PSI as a gauge
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=psi,
            title={"text": "Population Stability Index"},
            delta={"reference": 0.10},
            gauge={
                "axis": {"range": [0, 0.5]},
                "bar": {"color": "#6366f1"},
                "steps": [
                    {"range": [0, 0.10], "color": "#dcfce7"},
                    {"range": [0.10, 0.25], "color": "#fef9c3"},
                    {"range": [0.25, 0.5], "color": "#fecaca"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 4},
                    "thickness": 0.75,
                    "value": 0.25,
                },
            },
        )
    )
    fig.update_layout(height=300, margin={"t": 40, "b": 20, "l": 20, "r": 20})
    return fig


def _genre_treemap(monitoring: dict[str, Any]) -> go.Figure:
    """Genre distribution treemap."""
    genre_dist = monitoring.get("genre_distribution", {})
    if not isinstance(genre_dist, dict) or not genre_dist:
        fig = go.Figure()
        fig.add_annotation(text="No genre data", showarrow=False)
        return fig

    labels = list(genre_dist.keys())
    values = [float(v) for v in genre_dist.values()]

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=[""] * len(labels),
            values=values,
            textinfo="label+percent root",
            marker_colorscale="Viridis",
        )
    )
    fig.update_layout(
        title="Genre Distribution",
        height=400,
        margin={"t": 40, "b": 20, "l": 20, "r": 20},
    )
    return fig


# Main content
if "result" not in st.session_state:
    st.info("Run the pipeline from the sidebar to see results.")
    st.stop()

result: dict[str, Any] = st.session_state["result"]
monitoring: dict[str, Any] = result.get("monitoring", {})

if not isinstance(monitoring, dict) or not monitoring:
    st.info("Run pipeline with monitoring enabled to see results.")
    st.stop()

# Health banner
status, color = health_status(monitoring)
status_icon = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}.get(status, "")
st.markdown(
    f'<div style="background:{color}20; border-left: 4px solid {color}; '
    f'padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">'
    f'<span style="font-size: 1.5rem;">{status_icon}</span> '
    f'<strong style="font-size: 1.2rem;">System Status: {status.upper()}</strong>'
    f"</div>",
    unsafe_allow_html=True,
)

# Monitoring metrics grid (6 metrics)
m1, m2, m3 = st.columns(3)
m1.metric("PSI", f"{monitoring.get('psi', 0):.4f}", delta=monitoring.get("psi_status", ""))
m2.metric("Escalation Rate", f"{monitoring.get('escalation_rate', 0):.1%}")
m3.metric("Weight Shift", f"{monitoring.get('weight_shift', 0):.4f}")

m4, m5, m6 = st.columns(3)
m4.metric("ECE", f"{monitoring.get('ece', 0):.4f}")
m5.metric("Mean Score", f"{monitoring.get('mean_selection_score', 0):.4f}")
m6.metric("Regime", monitoring.get("regime_status", "normal"))

# Charts
col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(_ece_reliability_diagram(monitoring), use_container_width=True)
    st.plotly_chart(_genre_treemap(monitoring), use_container_width=True)

with col_right:
    st.plotly_chart(_psi_trend_chart(monitoring), use_container_width=True)

    # Alert timeline
    st.subheader("Regime Alerts")
    loop1 = monitoring.get("regime_loop1_alerts", [])
    loop2 = monitoring.get("regime_loop2_alerts", [])

    if loop1:
        for alert in loop1:
            st.error(f"**Loop 1 (Real-time):** {alert}")
    if loop2:
        for alert in loop2:
            st.warning(f"**Loop 2 (Delayed):** {alert}")
    if not loop1 and not loop2:
        st.success("No regime alerts detected")
