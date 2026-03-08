"""Structured escalation model: 6-reason enum with guidance."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EscalateReason(StrEnum):
    """Six escalation trigger types from SOT."""

    DISAGREE = "judge_disagreement"
    CONTRADICTION = "signal_contradiction"
    BOUNDARY = "boundary_case"
    MULTI_DIM = "multi_dim_disagree"
    EXTREME_DIM = "extreme_single_dim"
    LOW_COVERAGE = "low_ml_coverage"


# Human-readable guidance per reason
_GUIDANCE: dict[EscalateReason, str] = {
    EscalateReason.DISAGREE: "T1/T2 gap > 1.8 — review ML features vs LLM rubric alignment",
    EscalateReason.CONTRADICTION: "Pass^3 disagreement — check rubric consistency across passes",
    EscalateReason.BOUNDARY: "Selection score near threshold (+-2%) — manual tie-breaking needed",
    EscalateReason.MULTI_DIM: "2+ dimensions disagree by >= 1.8 — investigate conflicting signals",
    EscalateReason.EXTREME_DIM: "Single dimension delta >= 2.5 — potential outlier or data issue",
    EscalateReason.LOW_COVERAGE: "Feature fill rate < 40% — insufficient ML data coverage",
}


class EscalationResult(BaseModel):
    """Result of escalation check with structured reasons and guidance."""

    should_escalate: bool = Field(default=False)
    reasons: list[EscalateReason] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)

    @classmethod
    def from_reasons(cls, reasons: list[EscalateReason]) -> EscalationResult:
        """Build result from a list of triggered reasons."""
        return cls(
            should_escalate=len(reasons) > 0,
            reasons=reasons,
            guidance=[_GUIDANCE[r] for r in reasons],
        )

    @classmethod
    def none(cls) -> EscalationResult:
        """No escalation needed."""
        return cls(should_escalate=False, reasons=[], guidance=[])
