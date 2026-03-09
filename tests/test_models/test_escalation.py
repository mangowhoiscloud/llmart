"""Tests for escalation model."""

from __future__ import annotations

from llmart.models.escalation import EscalateReason, EscalationResult


def test_escalation_result_from_reasons_empty() -> None:
    result = EscalationResult.from_reasons([])
    assert result.should_escalate is False
    assert result.reasons == []
    assert result.guidance == []


def test_escalation_result_from_reasons_single() -> None:
    result = EscalationResult.from_reasons([EscalateReason.DISAGREE])
    assert result.should_escalate is True
    assert len(result.reasons) == 1
    assert len(result.guidance) == 1


def test_escalation_result_from_reasons_multiple() -> None:
    reasons = [EscalateReason.DISAGREE, EscalateReason.CONTRADICTION]
    result = EscalationResult.from_reasons(reasons)
    assert result.should_escalate is True
    assert len(result.reasons) == 2
    assert len(result.guidance) == 2


def test_escalation_result_none() -> None:
    result = EscalationResult.none()
    assert result.should_escalate is False
    assert result.reasons == []


def test_escalate_reason_values() -> None:
    assert EscalateReason.DISAGREE == "judge_disagreement"
    assert EscalateReason.CONTRADICTION == "signal_contradiction"
    assert EscalateReason.BOUNDARY == "boundary_case"
    assert EscalateReason.MULTI_DIM == "multi_dim_disagree"
    assert EscalateReason.EXTREME_DIM == "extreme_single_dim"
    assert EscalateReason.LOW_COVERAGE == "low_ml_coverage"


def test_all_reasons_have_guidance() -> None:
    """Every EscalateReason should produce guidance text."""
    for reason in EscalateReason:
        result = EscalationResult.from_reasons([reason])
        assert len(result.guidance) == 1
        assert isinstance(result.guidance[0], str)
        assert len(result.guidance[0]) > 0


def test_escalation_result_model_serialisation() -> None:
    result = EscalationResult.from_reasons([EscalateReason.BOUNDARY])
    data = result.model_dump()
    assert data["should_escalate"] is True
    assert data["reasons"] == ["boundary_case"]


def test_escalation_reason_is_strenum() -> None:
    assert isinstance(EscalateReason.DISAGREE, str)


def test_escalation_multi_dim() -> None:
    result = EscalationResult.from_reasons([EscalateReason.MULTI_DIM])
    assert "conflicting signals" in result.guidance[0].lower()


def test_escalation_extreme_dim() -> None:
    result = EscalationResult.from_reasons([EscalateReason.EXTREME_DIM])
    assert "outlier" in result.guidance[0].lower()


def test_escalation_low_coverage() -> None:
    result = EscalationResult.from_reasons([EscalateReason.LOW_COVERAGE])
    assert "coverage" in result.guidance[0].lower()
