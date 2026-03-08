"""Tests for QuantileRegressor (pure-Python quantile regression)."""

from __future__ import annotations

from llmart.models.quantile import (
    QuantileRegressor,
    extract_features,
)


class TestQuantileRegressor:
    def test_unfitted_returns_zero(self) -> None:
        qr = QuantileRegressor()
        result = qr.predict([0.5, 0.7, 0.5, 0.3, 0.2])
        assert result.q1_p50 == 0.0
        assert result.q1_p25 == 0.0
        assert result.confidence == 0.0

    def test_insufficient_data(self) -> None:
        qr = QuantileRegressor()
        qr.fit([[0.5] * 5] * 3, [100_000.0] * 3)
        assert qr.is_fitted is False

    def test_fit_with_enough_data(self) -> None:
        features = [[0.1 * i, 0.5 + 0.05 * i, 0.5, 0.3, 0.1] for i in range(20)]
        revenues = [50_000.0 + i * 100_000 for i in range(20)]
        qr = QuantileRegressor()
        qr.fit(features, revenues)
        assert qr.is_fitted is True
        assert qr.n_samples == 20

    def test_p50_greater_than_p25(self) -> None:
        features = [[0.1 * i, 0.5 + 0.05 * i, 0.5, 0.3, 0.1] for i in range(20)]
        revenues = [50_000.0 + i * 100_000 for i in range(20)]
        qr = QuantileRegressor()
        qr.fit(features, revenues)

        result = qr.predict([0.5, 0.7, 0.5, 0.3, 0.2])
        # P50 should generally be >= P25
        assert result.q1_p50 >= result.q1_p25 - 1.0  # allow small numeric noise

    def test_higher_features_higher_revenue(self) -> None:
        features = [[0.05 * i, 0.3 + 0.07 * i, 0.5, 0.3, 0.1] for i in range(20)]
        revenues = [10_000.0 * (i + 1) for i in range(20)]
        qr = QuantileRegressor()
        qr.fit(features, revenues)

        low = qr.predict([0.1, 0.4, 0.5, 0.3, 0.1])
        high = qr.predict([0.9, 0.9, 0.5, 0.3, 0.1])
        assert high.q1_p50 > low.q1_p50

    def test_predict_batch(self) -> None:
        features = [[0.1 * i, 0.5, 0.5, 0.3, 0.1] for i in range(10)]
        revenues = [100_000.0 * (i + 1) for i in range(10)]
        qr = QuantileRegressor()
        qr.fit(features, revenues)

        batch = qr.predict_batch([[0.3, 0.5, 0.5, 0.3, 0.1], [0.8, 0.5, 0.5, 0.3, 0.1]])
        assert len(batch) == 2

    def test_fit_returns_self(self) -> None:
        qr = QuantileRegressor()
        result = qr.fit([[0.5] * 5] * 10, [100_000.0] * 10)
        assert result is qr


class TestExtractFeatures:
    def test_basic_extraction(self) -> None:
        game = {
            "review_count": 1000,
            "steam_rating": 0.85,
            "price_usd": 15.0,
            "tags": ["RPG", "Roguelike", "Strategy"],
            "developer_successes": 1,
            "developer_games_released": 3,
        }
        feats = extract_features(game)
        assert len(feats) == 5
        assert all(0.0 <= f <= 1.0 for f in feats)

    def test_empty_game(self) -> None:
        feats = extract_features({})
        assert len(feats) == 5
        assert feats[0] == 0.0  # log_reviews of 0

    def test_zero_pivot_in_solve(self) -> None:
        """Test Gaussian elimination with near-singular matrix."""
        from llmart.models.quantile import _solve_linear

        # Near-singular matrix: row with near-zero pivot
        a = [[0.0, 1.0], [1.0, 0.0]]
        b = [2.0, 3.0]
        result = _solve_linear(a, b)
        assert len(result) == 2
        assert abs(result[0] - 3.0) < 0.1
        assert abs(result[1] - 2.0) < 0.1

    def test_singular_diagonal(self) -> None:
        """Test Gaussian elimination with zero diagonal after pivoting."""
        from llmart.models.quantile import _solve_linear

        # Matrix where a pivot is exactly zero
        a = [[0.0, 0.0], [0.0, 1.0]]
        b = [0.0, 1.0]
        result = _solve_linear(a, b)
        # Should handle gracefully (result[0] = 0 since row is all zeros)
        assert len(result) == 2

    def test_high_reviews_normalized(self) -> None:
        game = {"review_count": 500_000, "steam_rating": 0.9, "price_usd": 20.0}
        feats = extract_features(game)
        # log_reviews should be > 1.0 for very high values (unnormalized)
        # but normalized by log1p(MAX_REVIEWS)
        assert feats[0] > 0.9  # 500K reviews → near ceiling
