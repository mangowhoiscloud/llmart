"""Tests for phase controller."""

from __future__ import annotations

import pytest

from llmart.config import LLMARTConfig
from llmart.phase.controller import PhaseController


def test_phase_0_fixed_weights() -> None:
    ctrl = PhaseController()
    w_ml, w_llm = ctrl.get_weights(n_samples=5)
    assert w_ml == 0.6
    assert w_llm == 0.4
    assert ctrl.current_phase == 0


def test_auto_promote_to_phase_1() -> None:
    ctrl = PhaseController()
    w_ml, w_llm = ctrl.get_weights(n_samples=15)
    assert ctrl.current_phase == 1
    # Should be adaptive now
    assert abs(w_ml + w_llm - 1.0) < 1e-4


def test_phase_1_adaptive_weights() -> None:
    ctrl = PhaseController(initial_phase=1)
    w_ml, w_llm = ctrl.get_weights(n_samples=20, agreement_delta=0.0)
    assert abs(w_ml + w_llm - 1.0) < 1e-4
    # Low delta -> high jury reliability -> higher w_llm
    assert w_llm > 0.5


def test_phase_1_high_disagreement() -> None:
    ctrl = PhaseController(initial_phase=1)
    w_ml, w_llm = ctrl.get_weights(n_samples=20, agreement_delta=3.0)
    # High delta -> low reliability -> w_ml dominates
    assert w_ml > w_llm


def test_evaluate_promotion_0_to_1() -> None:
    ctrl = PhaseController()
    assert ctrl.evaluate_promotion(n=14, rho=0.0) == "hold"
    assert ctrl.evaluate_promotion(n=15, rho=0.0) == "promote"


def test_evaluate_promotion_1_to_2() -> None:
    ctrl = PhaseController(initial_phase=1)
    assert ctrl.evaluate_promotion(n=50, rho=0.40, w_std=0.10) == "promote"
    assert ctrl.evaluate_promotion(n=50, rho=0.40, w_std=0.20) == "hold"  # w_std too high
    assert ctrl.evaluate_promotion(n=50, rho=0.30, w_std=0.10) == "hold"  # rho too low


def test_evaluate_promotion_2_to_3() -> None:
    ctrl = PhaseController(initial_phase=2)
    assert ctrl.evaluate_promotion(n=200, rho=0.55) == "promote"
    assert ctrl.evaluate_promotion(n=200, rho=0.45) == "hold"


def test_evaluate_demotion_phase_1() -> None:
    ctrl = PhaseController(initial_phase=1)
    # rho < 0.35 * 0.5 = 0.175
    assert ctrl.evaluate_promotion(n=30, rho=0.10) == "demote"


def test_evaluate_demotion_phase_3() -> None:
    ctrl = PhaseController(initial_phase=3)
    assert ctrl.evaluate_promotion(n=300, rho=0.40) == "demote"
    assert ctrl.evaluate_promotion(n=300, rho=0.55) == "hold"


def test_apply_promotion() -> None:
    ctrl = PhaseController(initial_phase=1)
    new_phase = ctrl.apply_promotion("promote")
    assert new_phase == 2
    assert ctrl.current_phase == 2


def test_apply_demotion() -> None:
    ctrl = PhaseController(initial_phase=2)
    new_phase = ctrl.apply_promotion("demote")
    assert new_phase == 1


def test_apply_hold() -> None:
    ctrl = PhaseController(initial_phase=1)
    new_phase = ctrl.apply_promotion("hold")
    assert new_phase == 1


def test_apply_promotion_cap_at_3() -> None:
    ctrl = PhaseController(initial_phase=3)
    new_phase = ctrl.apply_promotion("promote")
    assert new_phase == 3


def test_apply_demotion_floor_at_0() -> None:
    ctrl = PhaseController(initial_phase=0)
    new_phase = ctrl.apply_promotion("demote")
    assert new_phase == 0


def test_weight_std() -> None:
    ctrl = PhaseController(initial_phase=1)
    for _ in range(5):
        ctrl.get_weights(n_samples=20, agreement_delta=0.5)
    std = ctrl.weight_std
    assert isinstance(std, float)
    assert std >= 0.0


def test_weight_std_single_sample() -> None:
    ctrl = PhaseController()
    ctrl.get_weights(n_samples=5)
    assert ctrl.weight_std == 0.0


@pytest.mark.parametrize("init_phase", [0, 1, 2, 3])
def test_initial_phase(init_phase: int) -> None:
    ctrl = PhaseController(initial_phase=init_phase)
    assert ctrl.current_phase == init_phase


def test_weight_history_accumulates() -> None:
    ctrl = PhaseController(initial_phase=1)
    for i in range(10):
        ctrl.get_weights(n_samples=20 + i, agreement_delta=0.5)
    assert len(ctrl.weight_history) == 10


def test_evaluate_demotion_phase_2() -> None:
    """Phase 2 with low rho should demote."""
    ctrl = PhaseController(initial_phase=2)
    # rho < promote_1to2_min_rho (0.35) → demote
    assert ctrl.evaluate_promotion(n=100, rho=0.30) == "demote"


def test_custom_config() -> None:
    cfg = LLMARTConfig(phase_0_w_ml=0.7, phase_0_w_llm=0.3)
    ctrl = PhaseController(config=cfg)
    w_ml, w_llm = ctrl.get_weights(n_samples=5)
    assert w_ml == 0.7
    assert w_llm == 0.3
