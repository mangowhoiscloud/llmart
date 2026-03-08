"""Pipeline node implementations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from llmart.pipeline.nodes.enrichment import enrichment_node
from llmart.pipeline.nodes.human_review import human_review_node
from llmart.pipeline.nodes.llm_judge import llm_judge_node
from llmart.pipeline.nodes.ml_scoring import ml_scoring_node
from llmart.pipeline.nodes.monitoring_node import monitoring_node
from llmart.pipeline.nodes.prefilter import prefilter_node
from llmart.pipeline.nodes.value import value_node

if __name__ != "__main__":
    from llmart.pipeline.state import GraphState


@runtime_checkable
class NodeProtocol(Protocol):
    """Formal interface contract for pipeline nodes.

    Every pipeline node must accept a GraphState dict and return a dict
    containing at minimum: candidates, stage, errors.
    """

    def __call__(self, state: GraphState) -> dict[str, Any]: ...


__all__ = [
    "NodeProtocol",
    "enrichment_node",
    "human_review_node",
    "llm_judge_node",
    "ml_scoring_node",
    "monitoring_node",
    "prefilter_node",
    "value_node",
]
