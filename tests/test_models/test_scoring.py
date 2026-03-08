"""Tests for Selection Score computation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmart.models.scoring import DEFAULT_W_LLM, DEFAULT_W_ML, SelectionScore


def test_selection_score_default_weights() -> None:
    s = SelectionScore(phi_ml=0.8, phi_llm=0.7)
    # S = 0.6 * 0.8 + 0.4 * 0.7 + 0.0 = 0.48 + 0.28 = 0.76
    assert abs(s.score - 0.76) < 1e-9


def test_selection_score_with_calibration() -> None:
    s = SelectionScore(phi_ml=0.8, phi_llm=0.7, delta_cal=0.05)
    # S = 0.6 * 0.8 + 0.4 * 0.7 + 0.05 = 0.81
    assert abs(s.score - 0.81) < 1e-9


def test_selection_score_custom_weights() -> None:
    s = SelectionScore(phi_ml=1.0, phi_llm=0.0, w_ml=1.0, w_llm=0.0)
    assert abs(s.score - 1.0) < 1e-9


def test_phase0_defaults() -> None:
    """Phase 0 defaults: PDF spec w_ml=0.6, w_llm=0.4."""
    assert DEFAULT_W_ML == 0.6
    assert DEFAULT_W_LLM == 0.4
    s = SelectionScore(phi_ml=0.5, phi_llm=0.5)
    assert s.w_ml == 0.6
    assert s.w_llm == 0.4


def test_zero_inputs() -> None:
    """phi_ml=0, phi_llm=0 -> score equals delta_cal."""
    s = SelectionScore(phi_ml=0.0, phi_llm=0.0, delta_cal=0.05)
    assert abs(s.score - 0.05) < 1e-9


def test_negative_delta_cal() -> None:
    s = SelectionScore(phi_ml=0.5, phi_llm=0.5, delta_cal=-0.10)
    expected = 0.6 * 0.5 + 0.4 * 0.5 - 0.10  # 0.40
    assert abs(s.score - expected) < 1e-9


def test_boundary_phi_values() -> None:
    s_low = SelectionScore(phi_ml=0.0, phi_llm=0.0)
    s_high = SelectionScore(phi_ml=1.0, phi_llm=1.0)
    assert s_low.score == 0.0
    assert abs(s_high.score - 1.0) < 1e-9


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match=r"w_ml.*w_llm.*must equal 1\.0"):
        SelectionScore(phi_ml=0.5, phi_llm=0.5, w_ml=0.7, w_llm=0.5)


def test_phi_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        SelectionScore(phi_ml=1.5, phi_llm=0.5)
    with pytest.raises(ValidationError):
        SelectionScore(phi_ml=-0.1, phi_llm=0.5)


def test_delta_cal_bounds() -> None:
    with pytest.raises(ValidationError):
        SelectionScore(phi_ml=0.5, phi_llm=0.5, delta_cal=0.6)
    with pytest.raises(ValidationError):
        SelectionScore(phi_ml=0.5, phi_llm=0.5, delta_cal=-0.6)
