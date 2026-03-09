"""LangGraph StateGraph definition."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from llmart.pipeline.nodes import (
    enrichment_node,
    human_review_node,
    llm_judge_node,
    ml_scoring_node,
    prefilter_node,
    value_node,
)
from llmart.pipeline.nodes.monitoring_node import monitoring_node
from llmart.pipeline.state import GraphState


def _timed_node(
    name: str,
    fn: Callable[[GraphState], dict[str, Any]],
) -> Callable[[GraphState], dict[str, Any]]:
    """Wrap a pipeline node to record execution time and candidate count."""

    def wrapper(state: GraphState) -> dict[str, Any]:
        start = time.monotonic()
        try:
            result = fn(state)
        except Exception as exc:
            elapsed = round(time.monotonic() - start, 4)
            timings = dict(state.get("stage_timings") or {})
            timings[name] = elapsed
            counts = dict(state.get("stage_counts") or {})
            counts[name] = len(state.get("candidates", []))
            return {
                "candidates": list(state.get("candidates", [])),
                "stage": name,
                "errors": [f"{name}: unhandled exception: {exc}"],
                "stage_timings": timings,
                "stage_counts": counts,
            }
        elapsed = round(time.monotonic() - start, 4)
        # Merge timing into node's result (preserve node-written stage_counts)
        timings = dict(state.get("stage_timings") or {})
        timings.update(result.get("stage_timings") or {})
        timings[name] = elapsed
        counts = dict(state.get("stage_counts") or {})
        counts.update(result.get("stage_counts") or {})
        # Auto-add candidate count if node didn't provide one for this stage
        if name not in counts:
            counts[name] = len(result.get("candidates", []))
        result["stage_timings"] = timings
        result["stage_counts"] = counts
        return result

    return wrapper


def create_llmart_graph(
    checkpointer: Any = None,
    enable_monitoring: bool = False,
) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Build and compile the linear LLMART pipeline graph.

    START -> prefilter -> ml_scoring -> llm_judge -> enrichment
          -> human_review -> value [-> monitoring] -> END

    Args:
        checkpointer: LangGraph checkpointer for persistence.
            Defaults to MemorySaver() if None.
        enable_monitoring: If True, adds a monitoring node after value.
    """
    graph = StateGraph(GraphState)

    graph.add_node("prefilter", _timed_node("prefilter", prefilter_node))  # type: ignore[call-overload]
    graph.add_node("ml_scoring", _timed_node("ml_scoring", ml_scoring_node))  # type: ignore[call-overload]
    graph.add_node("llm_judge", _timed_node("llm_judge", llm_judge_node))  # type: ignore[call-overload]
    graph.add_node("enrichment", _timed_node("enrichment", enrichment_node))  # type: ignore[call-overload]
    graph.add_node("human_review", _timed_node("human_review", human_review_node))  # type: ignore[call-overload]
    graph.add_node("value", _timed_node("value", value_node))  # type: ignore[call-overload]

    graph.add_edge(START, "prefilter")

    # Short-circuit to END on empty candidates after prefilter
    def _prefilter_router(state: GraphState) -> str:
        if not state.get("candidates"):
            return END
        return "ml_scoring"

    graph.add_conditional_edges(
        "prefilter", _prefilter_router, {"ml_scoring": "ml_scoring", END: END}
    )
    graph.add_edge("ml_scoring", "llm_judge")
    graph.add_edge("llm_judge", "enrichment")
    graph.add_edge("enrichment", "human_review")
    graph.add_edge("human_review", "value")

    if enable_monitoring:
        graph.add_node("monitoring", _timed_node("monitoring", monitoring_node))  # type: ignore[call-overload]
        graph.add_edge("value", "monitoring")
        graph.add_edge("monitoring", END)
    else:
        graph.add_edge("value", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)
