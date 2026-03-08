"""Tests for LLMARTConfig."""

from __future__ import annotations

from llmart.config import DEFAULT_CONFIG, LLMARTConfig


def test_default_config_immutable() -> None:
    """Config is frozen dataclass — mutation should raise."""
    import pytest

    with pytest.raises(AttributeError):
        DEFAULT_CONFIG.phase_0_w_ml = 0.9  # type: ignore[misc]


def test_default_weights_sum_to_one() -> None:
    assert DEFAULT_CONFIG.phase_0_w_ml + DEFAULT_CONFIG.phase_0_w_llm == 1.0


def test_phase_thresholds_ascending() -> None:
    assert DEFAULT_CONFIG.phase_1_min_n < DEFAULT_CONFIG.phase_2_min_n
    assert DEFAULT_CONFIG.phase_2_min_n < DEFAULT_CONFIG.phase_3_min_n


def test_custom_config() -> None:
    cfg = LLMARTConfig(ece_target=0.05, psi_warning=0.30)
    assert cfg.ece_target == 0.05
    assert cfg.psi_warning == 0.30
    # Defaults unchanged
    assert cfg.phase_0_w_ml == 0.6


def test_sigmoid_defaults() -> None:
    assert DEFAULT_CONFIG.sigmoid_k == 2.0
    assert DEFAULT_CONFIG.sigmoid_d0 == 1.0
