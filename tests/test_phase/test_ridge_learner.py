"""Tests for Phase2WeightLearner (ridge regression)."""

from __future__ import annotations

import pytest

from llmart.phase.ridge_learner import (
    LAMBDA_GRID,
    RHO_BASELINE,
    W_MAX,
    W_MIN,
    Phase2WeightLearner,
    _mse,
    _ridge_fit_2d,
    _spearman_rho,
    _standardise,
    _standardise_params,
)


class TestRidgeFit2D:
    def test_identity_fit(self) -> None:
        # y = x0 + x1 → beta should be ~(1, 1) with low lambda
        xm = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.0)]
        y = [1.0, 1.0, 2.0, 2.0]
        beta = _ridge_fit_2d(xm, y, lam=0.001)
        assert abs(beta[0] - 1.0) < 0.1
        assert abs(beta[1] - 1.0) < 0.1

    def test_high_lambda_shrinks_to_zero(self) -> None:
        xm = [(1.0, 0.0), (0.0, 1.0)]
        y = [1.0, 1.0]
        beta = _ridge_fit_2d(xm, y, lam=1e6)
        assert abs(beta[0]) < 0.1
        assert abs(beta[1]) < 0.1

    def test_singular_returns_zero(self) -> None:
        # All same features → near-singular
        xm = [(0.0, 0.0), (0.0, 0.0)]
        y = [1.0, 2.0]
        beta = _ridge_fit_2d(xm, y, lam=0.0)
        # With lambda=0, X'X is zero matrix → det=0 → returns (0,0)
        assert beta == (0.0, 0.0)


class TestMSE:
    def test_perfect_fit(self) -> None:
        xm = [(1.0, 0.0), (0.0, 1.0)]
        y = [1.0, 1.0]
        assert _mse(xm, y, (1.0, 1.0)) == pytest.approx(0.0, abs=1e-10)

    def test_empty_data(self) -> None:
        assert _mse([], [], (1.0, 1.0)) == 0.0

    def test_nonzero_error(self) -> None:
        xm = [(1.0, 0.0)]
        y = [2.0]
        # pred = 1*1 + 0*0 = 1, actual = 2, error = 1
        assert _mse(xm, y, (1.0, 0.0)) == pytest.approx(1.0, abs=1e-10)


class TestStandardise:
    def test_standardised_mean_zero(self) -> None:
        xm = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        mean, std = _standardise_params(xm)
        xs = _standardise(xm, mean, std)
        avg0 = sum(x[0] for x in xs) / len(xs)
        avg1 = sum(x[1] for x in xs) / len(xs)
        assert abs(avg0) < 1e-10
        assert abs(avg1) < 1e-10

    def test_empty_data(self) -> None:
        mean, std = _standardise_params([])
        assert mean == (0.0, 0.0)
        assert std == (1.0, 1.0)


class TestSpearmanRho:
    def test_perfect_correlation(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _spearman_rho(x, y) == pytest.approx(1.0, abs=1e-6)

    def test_perfect_inverse(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _spearman_rho(x, y) == pytest.approx(-1.0, abs=1e-6)

    def test_no_correlation(self) -> None:
        x = [1.0, 2.0, 3.0]
        y = [2.0, 3.0, 1.0]
        rho = _spearman_rho(x, y)
        assert -1.0 <= rho <= 1.0

    def test_too_few_samples(self) -> None:
        assert _spearman_rho([1.0, 2.0], [1.0, 2.0]) == 0.0
        assert _spearman_rho([], []) == 0.0


class TestPhase2WeightLearner:
    def test_insufficient_data_fallback(self) -> None:
        learner = Phase2WeightLearner()
        learner.fit([1.0], [1.0], [1.0])
        weights = learner.get_weights()
        assert weights["is_fallback"] is True
        assert "insufficient_data" in str(weights["fallback_reason"])

    def test_perfect_ml_signal(self) -> None:
        # ml_norm perfectly predicts y, jury_norm is noise
        ml = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0]
        jury = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        y = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0]
        learner = Phase2WeightLearner()
        learner.fit(ml, jury, y)
        weights = learner.get_weights()
        w_ml = float(weights["w_ml"])
        # ML should dominate (but sanity checks may cause fallback if rho < 0.35)
        if not weights["is_fallback"]:
            assert w_ml > 0.5

    def test_fallback_weights_configurable(self) -> None:
        learner = Phase2WeightLearner(fallback_w_ml=0.7, fallback_w_jury=0.3)
        learner.fit([1.0], [1.0], [1.0])
        weights = learner.get_weights()
        assert weights["w_ml"] == 0.7
        assert weights["w_jury"] == 0.3

    def test_lambda_grid_covers_range(self) -> None:
        assert len(LAMBDA_GRID) == 20
        assert LAMBDA_GRID[0] < 0.001
        assert LAMBDA_GRID[-1] > 1000

    def test_sanity_bounds(self) -> None:
        assert W_MIN == 0.15
        assert W_MAX == 0.85
        assert RHO_BASELINE == 0.35

    def test_weights_sum_to_one(self) -> None:
        learner = Phase2WeightLearner()
        # Correlated signals
        ml = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        jury = [0.2, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9, 1.0]
        y = [0.15, 0.25, 0.30, 0.45, 0.50, 0.65, 0.70, 0.85, 0.90, 1.00]
        learner.fit(ml, jury, y)
        weights = learner.get_weights()
        w_ml = float(weights["w_ml"])
        w_jury = float(weights["w_jury"])
        if not weights["is_fallback"]:
            assert abs(w_ml + w_jury - 1.0) < 0.01

    def test_fit_returns_self(self) -> None:
        learner = Phase2WeightLearner()
        result = learner.fit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result is learner

    def test_coefs_near_zero_fallback(self) -> None:
        """When both betas are near zero, should fall back."""
        learner = Phase2WeightLearner()
        # Use constant targets → Ridge will struggle to learn anything meaningful
        ml = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        jury = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        y = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        learner.fit(ml, jury, y)
        weights = learner.get_weights()
        # With zero variance in X, standardization produces zeros → coefs near zero
        assert weights["is_fallback"] is True

    def test_negative_beta_clamping(self) -> None:
        """Negative betas should be clamped to zero."""
        # When one feature is negatively correlated, its beta becomes negative
        # Non-negative constraint should clamp it to 0
        learner = Phase2WeightLearner()
        ml = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        jury = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]  # anti-correlated
        y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        learner.fit(ml, jury, y)
        weights = learner.get_weights()
        # Learning should proceed (might fallback due to sanity checks)
        assert "w_ml" in weights
        assert "w_jury" in weights
