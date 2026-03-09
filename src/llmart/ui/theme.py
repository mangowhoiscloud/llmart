"""LLMART dashboard design system — colours, CSS, metric helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st

# Signal palette
SIGNAL_COLORS: dict[str, str] = {
    "GREEN": "#22c55e",
    "YELLOW": "#eab308",
    "RED": "#ef4444",
}

SIGNAL_ICONS: dict[str, str] = {
    "GREEN": "🟢",
    "YELLOW": "🟡",
    "RED": "🔴",
}

# Stage colours for funnel
STAGE_COLORS: list[str] = [
    "#6366f1",  # indigo
    "#8b5cf6",  # violet
    "#a855f7",  # purple
    "#d946ef",  # fuchsia
    "#ec4899",  # pink
    "#f43f5e",  # rose
    "#ef4444",  # red
]

# Health status
HEALTH_COLORS: dict[str, str] = {
    "healthy": "#22c55e",
    "warning": "#eab308",
    "critical": "#ef4444",
}

CUSTOM_CSS = """
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 { color: #94a3b8; font-size: 0.85rem; margin: 0; }
    .metric-card .value { font-size: 2rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-card .delta { font-size: 0.8rem; color: #64748b; }
</style>
"""


def inject_css() -> None:
    """Inject custom CSS into the Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", color: str = "#e2e8f0") -> str:
    """Return HTML for a styled metric card."""
    return (
        f'<div class="metric-card">'
        f"<h3>{label}</h3>"
        f'<div class="value" style="color:{color}">{value}</div>'
        f'<div class="delta">{delta}</div>'
        f"</div>"
    )


def signal_badge(signal: str) -> str:
    """Return a coloured signal badge for display."""
    icon = SIGNAL_ICONS.get(signal, "⚪")
    return f"{icon} {signal}"


def fmt_money(n: float) -> str:
    """Format dollar amount: $3.7M, $293K, $500."""
    if abs(n) >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"${n / 1_000:.0f}K"
    return f"${n:,.0f}"


def health_status(monitoring: dict[str, Any]) -> tuple[str, str]:
    """Determine overall health from monitoring data.

    Returns: (status, color) where status is 'healthy'|'warning'|'critical'.
    """
    psi = float(monitoring.get("psi", 0.0))
    loop1 = monitoring.get("regime_loop1_alerts", [])
    esc_rate = float(monitoring.get("escalation_rate", 0.0))

    if psi > 0.25 or loop1:
        return "critical", HEALTH_COLORS["critical"]
    if psi > 0.10 or esc_rate > 0.3:
        return "warning", HEALTH_COLORS["warning"]
    return "healthy", HEALTH_COLORS["healthy"]
