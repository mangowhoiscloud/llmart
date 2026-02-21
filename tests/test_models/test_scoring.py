"""Tests for Selection Score computation."""

from __future__ import annotations

from llmart.models.scoring import SelectionScore


def test_selection_score_default_weights() -> None:
    s = SelectionScore(phi_ml=0.8, phi_llm=0.7)
    # S = 0.6 * 0.8 + 0.4 * 0.7 + 0.0 = 0.48 + 0.28 = 0.76
    assert abs(s.score - 0.76) < 1e-9


def test_selection_score_with_calibration() -> None:
    s = SelectionScore(phi_ml=0.8, phi_llm=0.7, delta_cal=0.05)
    assert abs(s.score - 0.81) < 1e-9


def test_selection_score_custom_weights() -> None:
    s = SelectionScore(phi_ml=1.0, phi_llm=0.0, w_ml=1.0, w_llm=0.0)
    assert abs(s.score - 1.0) < 1e-9
