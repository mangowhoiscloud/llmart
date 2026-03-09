"""Tests for PSI computation."""

from __future__ import annotations

from llmart.monitoring.psi import classify_psi, compute_psi


def test_psi_identical_distributions() -> None:
    data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    psi = compute_psi(data, data)
    assert psi == 0.0


def test_psi_similar_distributions() -> None:
    """Slightly shifted distributions should have positive but finite PSI."""
    ref = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cur = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
    psi = compute_psi(ref, cur)
    assert psi > 0.0
    assert psi < 10.0  # Finite


def test_psi_different_distributions() -> None:
    ref = [0.1, 0.2, 0.3, 0.4, 0.5]
    cur = [0.6, 0.7, 0.8, 0.9, 1.0]
    psi = compute_psi(ref, cur)
    assert psi > 0.0


def test_psi_empty_input() -> None:
    assert compute_psi([], [0.5]) == 0.0
    assert compute_psi([0.5], []) == 0.0


def test_psi_constant_values() -> None:
    """Same value everywhere -> no spread -> PSI = 0."""
    assert compute_psi([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == 0.0


def test_psi_non_negative() -> None:
    ref = [0.1, 0.3, 0.7, 0.9]
    cur = [0.2, 0.4, 0.6, 0.8]
    psi = compute_psi(ref, cur)
    assert psi >= 0.0


def test_classify_psi_stable() -> None:
    assert classify_psi(0.05) == "stable"


def test_classify_psi_caution() -> None:
    assert classify_psi(0.15) == "caution"


def test_classify_psi_warning() -> None:
    assert classify_psi(0.30) == "warning"


def test_classify_psi_with_config() -> None:
    """Classify PSI with explicit config parameter."""
    from llmart.config import LLMARTConfig

    cfg = LLMARTConfig(psi_stable=0.05, psi_warning=0.15)
    assert classify_psi(0.03, config=cfg) == "stable"
    assert classify_psi(0.10, config=cfg) == "caution"
    assert classify_psi(0.20, config=cfg) == "warning"
