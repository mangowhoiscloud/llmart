"""Tests for the full LangGraph pipeline."""

from __future__ import annotations

from llmart.pipeline.graph import create_llmart_graph
from llmart.pipeline.state import GraphState

_CONFIG = {"configurable": {"thread_id": "test"}}


def test_e2e_pipeline(initial_graph_state: GraphState) -> None:
    graph = create_llmart_graph()
    final = graph.invoke(initial_graph_state, config=_CONFIG)
    assert final["stage"] == "value"
    assert len(final["candidates"]) > 0


def test_all_candidates_have_signal(initial_graph_state: GraphState) -> None:
    graph = create_llmart_graph()
    final = graph.invoke(initial_graph_state, config=_CONFIG)
    for game in final["candidates"]:
        assert game["signal"] in ("GREEN", "YELLOW", "RED")


def test_no_errors(initial_graph_state: GraphState) -> None:
    graph = create_llmart_graph()
    final = graph.invoke(initial_graph_state, config=_CONFIG)
    assert final.get("errors", []) == []


def test_e2e_with_monitoring(initial_graph_state: GraphState) -> None:
    graph = create_llmart_graph(enable_monitoring=True)
    final = graph.invoke(initial_graph_state, config=_CONFIG)
    assert final["stage"] == "monitoring"
    assert "monitoring" in final
    monitoring = final["monitoring"]
    assert "psi" in monitoring
    assert "escalation_rate" in monitoring


def test_e2e_with_explicit_checkpointer(initial_graph_state: GraphState) -> None:
    """Graph with explicitly provided checkpointer (not None)."""
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    graph = create_llmart_graph(checkpointer=checkpointer)
    final = graph.invoke(initial_graph_state, config=_CONFIG)
    assert len(final["candidates"]) > 0


def test_e2e_with_phase(initial_graph_state: GraphState) -> None:
    state = {**initial_graph_state, "phase": 1, "n_historical": 20}
    graph = create_llmart_graph()
    final = graph.invoke(state, config=_CONFIG)
    assert len(final["candidates"]) > 0
    for game in final["candidates"]:
        assert "w_ml" in game
        assert "w_llm" in game


def test_monitoring_dict_merge(initial_graph_state: GraphState) -> None:
    """A4: monitoring dict merge reducer — both recall_at_99 and ndcg_at_30 survive."""
    state = {**initial_graph_state, "known_hits": ["G001"]}
    graph = create_llmart_graph()
    final = graph.invoke(state, config=_CONFIG)
    monitoring = final.get("monitoring", {})
    # prefilter writes recall_at_99, value writes ndcg_at_30
    assert "recall_at_99" in monitoring, f"Missing recall_at_99; got keys: {list(monitoring)}"
    assert "ndcg_at_30" in monitoring, f"Missing ndcg_at_30; got keys: {list(monitoring)}"


def test_timed_node_exception_handling() -> None:
    """A5: _timed_node wraps unhandled exceptions gracefully."""
    from llmart.pipeline.graph import _timed_node

    def crashing_node(state: GraphState) -> dict:
        raise RuntimeError("intentional crash")

    wrapped = _timed_node("test_crash", crashing_node)
    state = GraphState(candidates=[{"game_id": "X"}], stage="init", errors=[])
    result = wrapped(state)
    assert len(result["errors"]) == 1
    assert "intentional crash" in result["errors"][0]
    assert result["candidates"] == [{"game_id": "X"}]  # candidates preserved
    assert "test_crash" in result["stage_timings"]


def test_prefilter_empty_shortcircuit() -> None:
    """D4: empty candidates after prefilter should short-circuit to END."""
    # Use candidates that will ALL fail prefilter hard-cut
    bad_games = [
        {
            "game_id": "OLD",
            "title": "Old",
            "genre": "RPG",
            "developer": "Dev",
            "steam_rating": 0.90,
            "review_count": 5000,
            "price_usd": 20.0,
            "release_year": 2010,
            "tags": ["RPG"],
        }
    ]
    state = GraphState(candidates=bad_games, stage="init", top_k=30, errors=[], mode="mock")
    graph = create_llmart_graph()
    final = graph.invoke(state, config=_CONFIG)
    # Should end at prefilter with 0 candidates (no crash from downstream nodes)
    assert final["candidates"] == []
    assert final["stage"] == "prefilter"
