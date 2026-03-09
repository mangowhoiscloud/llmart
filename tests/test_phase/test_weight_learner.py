"""Tests for adaptive weight learner."""

from __future__ import annotations

import pytest

from llmart.phase.weight_learner import (
    compute_adaptive_weights,
    ema_smooth,
    jury_reliability,
)


def test_jury_reliability_reference_values() -> None:
    """SOT reference: Delta=0->0.88, 0.5->0.73, 1.0->0.50, 1.5->0.27, 2.0->0.12."""
    assert abs(jury_reliability(0.0) - 0.88) < 0.01
    assert abs(jury_reliability(0.5) - 0.73) < 0.01
    assert abs(jury_reliability(1.0) - 0.50) < 0.01
    assert abs(jury_reliability(1.5) - 0.27) < 0.01
    assert abs(jury_reliability(2.0) - 0.12) < 0.01


def test_jury_reliability_range() -> None:
    """Output should be in (0, 1) for any delta."""
    for delta in [0.0, 0.5, 1.0, 2.0, 5.0]:
        val = jury_reliability(delta)
        assert 0.0 < val < 1.0


def test_jury_reliability_monotone_decreasing() -> None:
    """Higher delta -> lower reliability."""
    prev = 1.0
    for delta in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        val = jury_reliability(delta)
        assert val < prev
        prev = val


def test_compute_adaptive_weights_sum_to_one() -> None:
    w_ml, w_llm = compute_adaptive_weights(0.5)
    assert abs(w_ml + w_llm - 1.0) < 1e-6


def test_compute_adaptive_weights_low_agreement() -> None:
    """High agreement delta -> low w_llm -> high w_ml."""
    w_ml, w_llm = compute_adaptive_weights(3.0)
    assert w_ml > w_llm


def test_compute_adaptive_weights_high_agreement() -> None:
    """Low agreement delta -> high w_llm."""
    _w_ml, w_llm = compute_adaptive_weights(0.0)
    assert w_llm > 0.5


def test_compute_adaptive_weights_clamped() -> None:
    """Weights should stay in [0.1, 0.9]."""
    w_ml, w_llm = compute_adaptive_weights(0.0, conf_scalar=2.0, coverage=1.0)
    assert w_ml >= 0.1
    assert w_llm <= 0.9


def test_ema_smooth_basic() -> None:
    new = (0.7, 0.3)
    old = (0.5, 0.5)
    result = ema_smooth(new, old, alpha=0.5)
    assert abs(result[0] + result[1] - 1.0) < 1e-4


def test_ema_smooth_alpha_one() -> None:
    """alpha=1.0 should fully adopt new weights."""
    new = (0.7, 0.3)
    old = (0.5, 0.5)
    result = ema_smooth(new, old, alpha=1.0)
    assert abs(result[0] - 0.7) < 1e-4
    assert abs(result[1] - 0.3) < 1e-4


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_ema_smooth_preserves_sum(alpha: float) -> None:
    result = ema_smooth((0.6, 0.4), (0.5, 0.5), alpha=alpha)
    assert abs(result[0] + result[1] - 1.0) < 1e-4


def test_ema_smooth_default_alpha() -> None:
    """EMA smooth with default alpha (None) should use config default."""
    result = ema_smooth((0.7, 0.3), (0.5, 0.5))
    assert abs(result[0] + result[1] - 1.0) < 1e-4
    # Default alpha=0.3 → result should be between old and new
    assert 0.5 <= result[0] <= 0.7


def test_ema_smooth_renormalization() -> None:
    """EMA smooth should renormalize to sum=1 after blending."""
    result = ema_smooth((0.8, 0.2), (0.6, 0.4), alpha=0.5)
    assert abs(result[0] + result[1] - 1.0) < 1e-4


def test_ema_smooth_negative_weights() -> None:
    """Negative weights yield total <= 0, skipping renormalization."""
    w_ml, w_llm = ema_smooth((-1.0, -1.0), (-1.0, -1.0), alpha=0.5)
    assert w_ml < 0
    assert w_llm < 0
