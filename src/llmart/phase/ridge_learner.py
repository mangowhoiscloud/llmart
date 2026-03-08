"""Phase 2 Ridge Weight Learner per selection-score-spec §A.1.2.

Pure-Python Non-negative Ridge for 2 features (ml_norm, jury_norm).
Closed-form solution for (X'X + λI)^(-1) X'y with β ≥ 0 constraint.
"""

from __future__ import annotations

import math

# Sanity check bounds
W_MIN: float = 0.15
W_MAX: float = 0.85
RHO_BASELINE: float = 0.35

# Lambda grid: logspace(-4, 4, 20)
LAMBDA_GRID: list[float] = [10 ** (i * 8.0 / 19 - 4.0) for i in range(20)]


class Phase2WeightLearner:
    """Ridge Regression weight learner for Selection Score.

    Non-negative Ridge: β ≥ 0 constraint (sane weights only).
    Nested LOOCV for lambda selection.
    Target: log(Y1 Revenue).

    Attributes:
        learned_w: Learned (w_ml, w_jury) or fallback.
        best_lambda: Selected regularization parameter.
        nested_rho: Out-of-sample Spearman rho from Nested LOOCV.
        fallback_reason: Why fallback was used (None if learning succeeded).
    """

    def __init__(
        self,
        fallback_w_ml: float = 0.5,
        fallback_w_jury: float = 0.5,
    ) -> None:
        self.fallback_w = (fallback_w_ml, fallback_w_jury)
        self.learned_w: tuple[float, float] = self.fallback_w
        self.best_lambda: float | None = None
        self.nested_rho: float | None = None
        self.fallback_reason: str | None = None

    def fit(
        self,
        ml_norms: list[float],
        jury_norms: list[float],
        y1_revenues_log: list[float],
    ) -> Phase2WeightLearner:
        """Nested LOOCV → lambda selection → Ridge fit → weight extraction.

        All inputs must have the same length (n ≥ 3).
        """
        n = len(ml_norms)
        if n < 3 or n != len(jury_norms) or n != len(y1_revenues_log):
            self.fallback_reason = f"insufficient_data: n={n}"
            self.learned_w = self.fallback_w
            return self

        xm = list(zip(ml_norms, jury_norms, strict=True))
        y = list(y1_revenues_log)

        # --- Nested LOOCV ---
        oos_preds: list[float] = [0.0] * n
        for i in range(n):
            xm_train = [xm[j] for j in range(n) if j != i]
            y_train = [y[j] for j in range(n) if j != i]
            xm_test = xm[i]

            # Standardise train
            mean_x, std_x = _standardise_params(xm_train)
            xm_train_s = _standardise(xm_train, mean_x, std_x)
            xm_test_s = (
                (xm_test[0] - mean_x[0]) / max(std_x[0], 1e-8),
                (xm_test[1] - mean_x[1]) / max(std_x[1], 1e-8),
            )

            # Inner: grid search for best lambda (MSE on train)
            best_lam, best_mse = LAMBDA_GRID[0], float("inf")
            for lam in LAMBDA_GRID:
                beta = _ridge_fit_2d(xm_train_s, y_train, lam)
                if beta[0] < 0 or beta[1] < 0:
                    beta = (max(0.0, beta[0]), max(0.0, beta[1]))
                mse = _mse(xm_train_s, y_train, beta)
                if mse < best_mse:
                    best_mse = mse
                    best_lam = lam

            beta = _ridge_fit_2d(xm_train_s, y_train, best_lam)
            beta = (max(0.0, beta[0]), max(0.0, beta[1]))
            oos_preds[i] = beta[0] * xm_test_s[0] + beta[1] * xm_test_s[1]

        # --- Performance: Spearman rho ---
        self.nested_rho = _spearman_rho(oos_preds, y)

        # --- Final model on all data ---
        mean_x, std_x = _standardise_params(xm)
        xm_s = _standardise(xm, mean_x, std_x)

        best_lam_final, best_mse_final = LAMBDA_GRID[0], float("inf")
        for lam in LAMBDA_GRID:
            beta = _ridge_fit_2d(xm_s, y, lam)
            beta = (max(0.0, beta[0]), max(0.0, beta[1]))
            mse = _mse(xm_s, y, beta)
            if mse < best_mse_final:
                best_mse_final = mse
                best_lam_final = lam

        final_beta = _ridge_fit_2d(xm_s, y, best_lam_final)
        final_beta = (max(0.0, final_beta[0]), max(0.0, final_beta[1]))
        self.best_lambda = best_lam_final

        # --- Convert β → weights ---
        coef_sum = final_beta[0] + final_beta[1]
        if coef_sum < 1e-8:
            self.fallback_reason = "coefs_near_zero"
            self.learned_w = self.fallback_w
            return self

        w_ml = final_beta[0] / coef_sum
        w_jury = final_beta[1] / coef_sum

        # --- Sanity check ---
        if self._sanity_check(w_ml, w_jury):
            self.learned_w = (round(w_ml, 4), round(w_jury, 4))
            self.fallback_reason = None
        else:
            self.learned_w = self.fallback_w

        return self

    def _sanity_check(self, w_ml: float, w_jury: float) -> bool:
        """Validate learned weights against safety bounds."""
        checks = {
            "range_ml": W_MIN <= w_ml <= W_MAX,
            "range_jury": W_MIN <= w_jury <= W_MAX,
            "sum_to_one": abs(w_ml + w_jury - 1.0) < 1e-4,
            "rho_above_baseline": (self.nested_rho or 0.0) > RHO_BASELINE,
        }
        failed = [k for k, v in checks.items() if not v]
        if failed:
            self.fallback_reason = f"sanity_check_failed: {failed}"
            return False
        return True

    def get_weights(self) -> dict[str, object]:
        """Return learned weights and metadata."""
        return {
            "w_ml": self.learned_w[0],
            "w_jury": self.learned_w[1],
            "lambda": self.best_lambda,
            "nested_rho": self.nested_rho,
            "is_fallback": self.learned_w == self.fallback_w,
            "fallback_reason": self.fallback_reason,
        }


# --- Pure-Python linear algebra for 2D Ridge ---


def _ridge_fit_2d(
    xm: list[tuple[float, float]],
    y: list[float],
    lam: float,
) -> tuple[float, float]:
    """Closed-form Ridge for 2 features: β = (X'X + λI)^(-1) X'y."""
    n = len(xm)
    # X'X (2x2)
    xx00 = sum(x[0] * x[0] for x in xm) + lam
    xx01 = sum(x[0] * x[1] for x in xm)
    xx11 = sum(x[1] * x[1] for x in xm) + lam

    # X'y (2x1)
    xy0 = sum(xm[i][0] * y[i] for i in range(n))
    xy1 = sum(xm[i][1] * y[i] for i in range(n))

    # Invert 2x2: [a b; b d]^(-1) = 1/det * [d -b; -b a]
    det = xx00 * xx11 - xx01 * xx01
    if abs(det) < 1e-12:
        return (0.0, 0.0)

    b0 = (xx11 * xy0 - xx01 * xy1) / det
    b1 = (xx00 * xy1 - xx01 * xy0) / det
    return (b0, b1)


def _mse(
    xm: list[tuple[float, float]],
    y: list[float],
    beta: tuple[float, float],
) -> float:
    """Mean squared error."""
    n = len(y)
    if n == 0:
        return 0.0
    return sum((y[i] - beta[0] * xm[i][0] - beta[1] * xm[i][1]) ** 2 for i in range(n)) / n


def _standardise_params(
    xm: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute mean and std for 2 features."""
    n = len(xm)
    if n == 0:
        return (0.0, 0.0), (1.0, 1.0)
    m0 = sum(x[0] for x in xm) / n
    m1 = sum(x[1] for x in xm) / n
    s0 = math.sqrt(sum((x[0] - m0) ** 2 for x in xm) / max(n - 1, 1))
    s1 = math.sqrt(sum((x[1] - m1) ** 2 for x in xm) / max(n - 1, 1))
    return (m0, m1), (max(s0, 1e-8), max(s1, 1e-8))


def _standardise(
    xm: list[tuple[float, float]],
    mean: tuple[float, float],
    std: tuple[float, float],
) -> list[tuple[float, float]]:
    """Standardise features to zero mean, unit variance."""
    return [((x[0] - mean[0]) / std[0], (x[1] - mean[1]) / std[1]) for x in xm]


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (pure Python)."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        for rank_pos, (orig_idx, _) in enumerate(indexed):
            ranks[orig_idx] = float(rank_pos + 1)
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry, strict=True))
    return round(1.0 - (6 * d_sq) / (n * (n * n - 1)), 6)
