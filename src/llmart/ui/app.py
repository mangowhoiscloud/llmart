"""LLMART Streamlit Dashboard — Multi-page entry point.

Pages:
1. Pipeline: Funnel, timing, signal distribution, KPI cards
2. Games: Interactive table, radar charts, value breakdown
3. Monitoring: Health banner, ECE, PSI, genre treemap, alerts
4. Retrain: Quarterly retrain controls and visualization
"""

from __future__ import annotations

import json
import sys
import uuid
import warnings
from importlib import resources
from pathlib import Path
from typing import Any

import streamlit as st

from llmart.audit import AuditStore, RunRecord
from llmart.logging_config import generate_run_id
from llmart.ui.export import candidates_to_csv, candidates_to_json
from llmart.ui.theme import inject_css

warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

# Page config
st.set_page_config(
    page_title="LLMART Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("LLMART — Game Selection & Value Inference")
inject_css()


def _bundled_path(filename: str) -> Path:
    return Path(str(resources.files("llmart") / "data" / filename))


def _load_json(path: Path) -> list[dict[str, object]] | dict[str, object]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def _run_pipeline(
    candidates: list[dict[str, object]],
    genre_params: dict[str, object],
    mode: str,
    top_k: int,
    phase: int,
    enable_monitoring: bool,
) -> dict[str, object]:
    from llmart.pipeline.graph import create_llmart_graph

    graph = create_llmart_graph(enable_monitoring=enable_monitoring)
    thread_id = str(uuid.uuid4())
    initial_state = {
        "candidates": candidates,
        "stage": "init",
        "top_k": top_k,
        "errors": [],
        "mode": mode,
        "genre_params": genre_params,
        "total_input": len(candidates),
        "phase": phase,
        "n_historical": 0,
        "run_metadata": {"thread_id": thread_id},
    }
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    # Stream through graph for progress
    final: dict[str, Any] = {}
    for event in graph.stream(initial_state, config=config):  # type: ignore[arg-type]
        for _node_name, node_output in event.items():
            final = {**final, **node_output}

    return final


# Sidebar controls
with st.sidebar:
    st.header("Pipeline Settings")
    mode = st.selectbox("LLM Mode", ["mock", "real"], index=0)
    top_k = st.slider("Top-K", 5, 50, 10)
    phase = st.slider("Phase", 0, 3, 0)
    enable_monitoring = st.checkbox("Enable Monitoring", value=True)

    games_file = st.file_uploader("Custom Games JSON", type=["json"])

    run_button = st.button("Run Pipeline", type="primary", use_container_width=True)

    # Run history
    st.divider()
    st.subheader("Run History")
    try:
        audit = AuditStore()
        recent_runs = audit.list_runs(limit=5)
        audit.close()
        if recent_runs:
            for r in recent_runs:
                dur = f"{r.get('duration_s', 0):.1f}s" if r.get("duration_s") else "?"
                sig_summary = (
                    f"G:{r.get('signal_green', 0)} "
                    f"Y:{r.get('signal_yellow', 0)} "
                    f"R:{r.get('signal_red', 0)}"
                )
                st.caption(f"`{r['run_id']}` {r.get('mode', '?')} {dur} | {sig_summary}")
        else:
            st.caption("No previous runs")
    except Exception:
        st.caption("Audit store unavailable")

# Load data
if games_file is not None:
    candidates = json.loads(games_file.read())
else:
    candidates = _load_json(_bundled_path("sample_games.json"))
genre_params = _load_json(_bundled_path("genre_params.json"))

# Run pipeline
if run_button:
    if not isinstance(candidates, list):
        st.error("Games data must be a list")
        st.stop()

    run_id = generate_run_id()
    record = RunRecord(
        run_id=run_id,
        mode=mode,
        phase=phase,
        top_k=top_k,
        input_count=len(candidates),
    )

    progress = st.progress(0, text="Running LLMART pipeline...")
    result = _run_pipeline(
        candidates,
        genre_params,  # type: ignore[arg-type]
        mode,
        top_k,
        phase,
        enable_monitoring,
    )
    progress.progress(100, text="Pipeline complete!")

    # Audit recording
    result_candidates = result.get("candidates", [])
    if isinstance(result_candidates, list):
        record.finish(result_candidates, result.get("errors", []))  # type: ignore[arg-type]
    try:
        audit_store = AuditStore()
        audit_store.save(record)
        audit_store.close()
    except Exception:  # noqa: S110
        pass  # Audit is non-critical; pipeline result is already stored

    st.session_state["result"] = result
    st.session_state["input_count"] = len(candidates)
    st.session_state["run_id"] = run_id

# Export buttons
if "result" in st.session_state:
    result_data = st.session_state["result"]
    final_candidates: list[dict[str, Any]] = result_data.get("candidates", [])

    if final_candidates:
        with st.sidebar:
            st.divider()
            st.subheader("Export Results")
            col_csv, col_json = st.columns(2)
            with col_csv:
                st.download_button(
                    "CSV",
                    candidates_to_csv(final_candidates),
                    file_name="llmart_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_json:
                st.download_button(
                    "JSON",
                    candidates_to_json(final_candidates),
                    file_name="llmart_results.json",
                    mime="application/json",
                    use_container_width=True,
                )

# Landing page when no results
if "result" not in st.session_state:
    st.info("Configure settings in the sidebar and click **Run Pipeline** to start.")
    st.markdown("""
    **Pages:**
    - **Pipeline** — Funnel chart, stage timing, signal distribution
    - **Games** — Interactive data table, 5-Dim radar, value breakdown
    - **Monitoring** — Health status, ECE, PSI, regime alerts
    - **Retrain** — Quarterly retrain cycle with validation
    """)


def main() -> None:
    """Entry point for ``llmart-ui`` console script."""
    from streamlit.web.cli import main as st_main

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    st_main()
