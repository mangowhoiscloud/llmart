"""Tests for ECE computation."""

from __future__ import annotations

from llmart.monitoring.ece import CALIBRATION_TABLE, compute_ece, needs_retrain


def test_ece_well_calibrated() -> None:
    """Roughly calibrated predictions => ECE < 0.3."""
    preds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    outcomes = [False, False, False, False, True, True, True, True, True]
    ece = compute_ece(preds, outcomes)
    assert ece < 0.30  # Reasonable for small sample


def test_ece_worst_case() -> None:
    """All predictions wrong -> high ECE."""
    preds = [0.9, 0.9, 0.9]
    outcomes = [False, False, False]
    ece = compute_ece(preds, outcomes)
    assert ece > 0.5


def test_ece_empty_input() -> None:
    assert compute_ece([], []) == 0.0


def test_ece_mismatched_lengths() -> None:
    assert compute_ece([0.5], [True, False]) == 0.0


def test_ece_range() -> None:
    """ECE should be in [0, 1]."""
    preds = [0.3, 0.7, 0.5, 0.9]
    outcomes = [False, True, True, True]
    ece = compute_ece(preds, outcomes)
    assert 0.0 <= ece <= 1.0


def test_needs_retrain_below_target() -> None:
    assert needs_retrain(0.05) is False


def test_needs_retrain_above_target() -> None:
    assert needs_retrain(0.15) is True


def test_needs_retrain_custom_target() -> None:
    assert needs_retrain(0.08, target=0.05) is True
    assert needs_retrain(0.03, target=0.05) is False


def test_calibration_table_values() -> None:
    assert CALIBRATION_TABLE[(1, 2)] == 0.05
    assert CALIBRATION_TABLE[(4, 5)] == 0.60
