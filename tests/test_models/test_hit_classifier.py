"""Tests for HitTierClassifier."""

from __future__ import annotations

import pytest

from llmart.models.ground_truth import RevenueTier
from llmart.models.hit_classifier import (
    TIER_ORDER,
    TIER_WEIGHTS,
    HitTierClassifier,
)


class TestHitTierClassifier:
    def test_unfitted_predict_returns_default(self) -> None:
        clf = HitTierClassifier()
        result = clf.predict([0.5, 0.5, 3.0, 0.8, 0.5])
        assert result.confidence == 0.0
        assert result.predicted_tier == RevenueTier.SIDE

    def test_predict_with_q1_revenue_uses_threshold(self) -> None:
        clf = HitTierClassifier()
        # Mega: >= $20M
        result = clf.predict([0.5] * 5, q1_revenue=25_000_000.0)
        assert result.predicted_tier == RevenueTier.MEGA
        assert result.confidence == 1.0

        # Hit: $2M - $20M
        result = clf.predict([0.5] * 5, q1_revenue=5_000_000.0)
        assert result.predicted_tier == RevenueTier.HIT

        # Side: $250K - $2M
        result = clf.predict([0.5] * 5, q1_revenue=500_000.0)
        assert result.predicted_tier == RevenueTier.SIDE

        # Hobby: < $250K
        result = clf.predict([0.5] * 5, q1_revenue=100_000.0)
        assert result.predicted_tier == RevenueTier.HOBBY

    def test_fit_insufficient_data_fallback(self) -> None:
        clf = HitTierClassifier()
        clf.fit([[0.5] * 5] * 3, [100_000.0] * 3)
        assert clf.is_fitted is False
        assert "insufficient_data" in str(clf.fallback_reason)

    def test_fit_with_enough_data(self) -> None:
        features = []
        q1_revenues = []
        # Generate synthetic data for each tier
        for i in range(15):
            features.append([0.1 * i, 0.1 * i, 2.0, 0.5, 0.3])
            q1_revenues.append(50_000.0 + i * 10_000)  # Hobby range

        for i in range(10):
            features.append([0.5 + 0.03 * i, 0.5 + 0.03 * i, 4.0, 0.7, 0.5])
            q1_revenues.append(500_000.0 + i * 100_000)  # Side range

        clf = HitTierClassifier()
        clf.fit(features, q1_revenues)
        assert clf.is_fitted is True
        assert clf.n_samples == 25
        assert clf.fallback_reason is None

    def test_predict_after_fit(self) -> None:
        features = []
        q1_revenues = []
        for _i in range(15):
            features.append([0.1, 0.1, 2.0, 0.5, 0.3])
            q1_revenues.append(100_000.0)
        for _i in range(10):
            features.append([0.9, 0.9, 5.0, 0.9, 0.8])
            q1_revenues.append(5_000_000.0)

        clf = HitTierClassifier()
        clf.fit(features, q1_revenues)

        # Predict for a feature vector similar to hobby
        result = clf.predict([0.1, 0.1, 2.0, 0.5, 0.3])
        assert result.predicted_tier in TIER_ORDER
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_batch(self) -> None:
        clf = HitTierClassifier()
        features_list = [[0.5] * 5, [0.8] * 5]
        q1_revenues = [100_000.0, 5_000_000.0]
        results = clf.predict_batch(features_list, q1_revenues)
        assert len(results) == 2
        assert results[0].predicted_tier == RevenueTier.HOBBY
        assert results[1].predicted_tier == RevenueTier.HIT

    def test_ordinal_score_range(self) -> None:
        clf = HitTierClassifier()
        for tier, expected_ord in [
            (RevenueTier.HOBBY, 0.0),
            (RevenueTier.SIDE, 1 / 3),
            (RevenueTier.HIT, 2 / 3),
            (RevenueTier.MEGA, 1.0),
        ]:
            result = clf.predict(
                [0.5] * 5,
                q1_revenue={
                    RevenueTier.HOBBY: 100_000.0,
                    RevenueTier.SIDE: 500_000.0,
                    RevenueTier.HIT: 5_000_000.0,
                    RevenueTier.MEGA: 25_000_000.0,
                }[tier],
            )
            assert result.ordinal_score == pytest.approx(expected_ord, abs=0.01)

    def test_get_summary(self) -> None:
        clf = HitTierClassifier()
        summary = clf.get_summary()
        assert summary["is_fitted"] is False
        assert summary["n_samples"] == 0

    def test_tier_weights_defined(self) -> None:
        for tier in TIER_ORDER:
            assert tier.value in TIER_WEIGHTS

    def test_fit_returns_self(self) -> None:
        clf = HitTierClassifier()
        result = clf.fit([[0.5] * 5] * 3, [100_000.0] * 3)
        assert result is clf

    def test_fit_empty_features_skips_importance(self) -> None:
        """Empty feature vectors → n_features=0 → skip feature importance."""
        clf = HitTierClassifier()
        # 25 samples with empty features, enough for the n>=10 threshold
        features = [[]] * 25
        q1_revenues = [100_000.0] * 15 + [5_000_000.0] * 10
        clf.fit(features, q1_revenues)
        assert clf.is_fitted is True
        assert clf.feature_importances == {}
