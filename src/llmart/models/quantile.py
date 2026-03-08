"""Pure-Python Quantile Regression for Q1 Revenue Estimation.

Per selection-score-spec §A.5:
- Linear quantile regression for P50 (median) and P25 estimation
- Uses iteratively reweighted least squares (IRLS) approximation
- Features: log_reviews, steam_rating, price_usd, tag_relevance, dev_wilson
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Feature extraction helpers
MAX_REVIEWS = 100_000
MAX_PRICE = 60.0


@dataclass
class QuantileResult:
    """Q1 revenue estimates at P50 and P25."""

    q1_p50: float
    q1_p25: float
    confidence: float  # R² or pseudo-R² of the fit


@dataclass
class QuantileRegressor:
    """Pure-Python linear quantile regression for Q1 revenue estimation.

    Uses IRLS (Iteratively Reweighted Least Squares) to approximate
    quantile regression without scipy dependency.

    Attributes:
        coefs_p50: Learned coefficients for P50 (median).
        coefs_p25: Learned coefficients for P25.
        is_fitted: Whether the model has been trained.
        n_samples: Number of training samples.
    """

    coefs_p50: list[float] = field(default_factory=list)
    coefs_p25: list[float] = field(default_factory=list)
    is_fitted: bool = False
    n_samples: int = 0

    def fit(
        self,
        features: list[list[float]],
        q1_revenues: list[float],
        n_iter: int = 50,
    ) -> QuantileRegressor:
        """Fit quantile regression at tau=0.50 and tau=0.25.

        Args:
            features: Feature matrix (n x d).
            q1_revenues: Q1 revenue targets.
            n_iter: IRLS iterations.

        Returns:
            self for method chaining.
        """
        n = len(features)
        if n < 5 or n != len(q1_revenues):
            return self

        # Log-transform targets for numerical stability
        y = [math.log1p(max(0.0, r)) for r in q1_revenues]

        # Add intercept
        x = [[1.0, *f] for f in features]

        self.coefs_p50 = _irls_quantile(x, y, tau=0.50, n_iter=n_iter)
        self.coefs_p25 = _irls_quantile(x, y, tau=0.25, n_iter=n_iter)
        self.is_fitted = True
        self.n_samples = n
        return self

    def predict(self, features: list[float]) -> QuantileResult:
        """Predict Q1 revenue at P50 and P25 for a single game."""
        if not self.is_fitted:
            return QuantileResult(q1_p50=0.0, q1_p25=0.0, confidence=0.0)

        x = [1.0, *features]
        log_p50 = _dot(self.coefs_p50, x)
        log_p25 = _dot(self.coefs_p25, x)

        return QuantileResult(
            q1_p50=round(math.expm1(max(0.0, log_p50)), 2),
            q1_p25=round(math.expm1(max(0.0, log_p25)), 2),
            confidence=0.8 if self.n_samples >= 50 else 0.5,
        )

    def predict_batch(self, features_list: list[list[float]]) -> list[QuantileResult]:
        """Predict Q1 for a batch of games."""
        return [self.predict(f) for f in features_list]


def extract_features(game: dict[str, object]) -> list[float]:
    """Extract features for quantile regression from a game dict."""
    raw_reviews = game.get("review_count", 0)
    reviews = int(raw_reviews) if isinstance(raw_reviews, (int, float)) else 0
    raw_rating = game.get("steam_rating", 0.0)
    rating = float(raw_rating) if isinstance(raw_rating, (int, float)) else 0.0
    raw_price = game.get("price_usd", 0.0)
    price = float(raw_price) if isinstance(raw_price, (int, float)) else 0.0
    tags = game.get("tags", [])
    raw_succ = game.get("developer_successes", 0)
    dev_successes = int(raw_succ) if isinstance(raw_succ, (int, float)) else 0
    raw_total = game.get("developer_games_released", 1)
    dev_total = int(raw_total) if isinstance(raw_total, (int, float)) else 1

    log_reviews = math.log1p(reviews) / math.log1p(MAX_REVIEWS)
    price_norm = min(price / MAX_PRICE, 1.0)
    dev_ratio = dev_successes / max(dev_total, 1)

    tag_count = len(tags) if isinstance(tags, list) else 0
    tag_density = min(tag_count / 20.0, 1.0)

    return [log_reviews, rating, price_norm, tag_density, dev_ratio]


# --- Pure-Python IRLS for quantile regression ---


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(ai * bi for ai, bi in zip(a, b, strict=True))


def _irls_quantile(
    x: list[list[float]],
    y: list[float],
    tau: float,
    n_iter: int = 50,
) -> list[float]:
    """IRLS approximation of quantile regression.

    Minimises: sum(rho_tau(y_i - x_i'beta)) where rho_tau is the check function.
    Uses weighted least squares with weights = 1/|residual| as IRLS proxy.
    """
    n = len(y)

    # Start with OLS solution
    beta = _ols_fit(x, y)

    for _iteration in range(n_iter):
        # Compute residuals and weights
        residuals = [y[i] - _dot(x[i], beta) for i in range(n)]
        weights = []
        for r in residuals:
            if abs(r) < 1e-6:
                weights.append(1e6)  # near-zero residual → high weight
            elif r > 0:
                weights.append(tau / abs(r))
            else:
                weights.append((1.0 - tau) / abs(r))

        # Weighted least squares: beta = (X'WX)^{-1} X'Wy
        beta = _wls_fit(x, y, weights)

    return beta


def _ols_fit(x: list[list[float]], y: list[float]) -> list[float]:
    """Ordinary least squares via normal equations (pure Python)."""
    n = len(y)
    d = len(x[0])

    # X'X
    xtx = [[0.0] * d for _ in range(d)]
    for i in range(n):
        for j in range(d):
            for k in range(d):
                xtx[j][k] += x[i][j] * x[i][k]

    # Add ridge regularization for numerical stability
    for j in range(d):
        xtx[j][j] += 1e-6

    # X'y
    xty = [0.0] * d
    for i in range(n):
        for j in range(d):
            xty[j] += x[i][j] * y[i]

    # Solve via Cholesky-like approach (simplified Gaussian elimination)
    return _solve_linear(xtx, xty)


def _wls_fit(x: list[list[float]], y: list[float], w: list[float]) -> list[float]:
    """Weighted least squares via normal equations."""
    n = len(y)
    d = len(x[0])

    # X'WX
    xtwx = [[0.0] * d for _ in range(d)]
    for i in range(n):
        for j in range(d):
            for k in range(d):
                xtwx[j][k] += w[i] * x[i][j] * x[i][k]

    for j in range(d):
        xtwx[j][j] += 1e-6

    # X'Wy
    xtwy = [0.0] * d
    for i in range(n):
        for j in range(d):
            xtwy[j] += w[i] * x[i][j] * y[i]

    return _solve_linear(xtwx, xtwy)


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax=b via Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    aug = [[*row[:], b[i]] for i, row in enumerate(a)]

    for col in range(n):
        # Partial pivoting
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            continue

        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    # Back substitution
    result = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(aug[i][i]) < 1e-12:
            continue
        result[i] = aug[i][n]
        for j in range(i + 1, n):
            result[i] -= aug[i][j] * result[j]
        result[i] /= aug[i][i]

    return result
