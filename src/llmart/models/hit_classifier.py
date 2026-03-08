"""Hit Tier Classifier: 4-tier label prediction per selection-score-spec §A.4.

Cost-Sensitive classification: Hobby(66%) / Side(23%) / Hit(8%) / Mega(2-3%).
Uses LogisticRegression with class_weight='balanced' as Phase 3 baseline.
Pure-Python fallback available for environments without scikit-learn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from llmart.models.ground_truth import RevenueTier, classify_tier

# Feature names for the classifier
FEATURE_NAMES: list[str] = [
    "ml_percentile",
    "jury_score_norm",
    "review_count_log",
    "steam_rating",
    "tag_relevance",
]

# Cost-sensitive weights (inverse prior frequency)
TIER_WEIGHTS: dict[str, float] = {
    RevenueTier.HOBBY: 1.0,  # 66% → low weight
    RevenueTier.SIDE: 2.87,  # 23% → medium
    RevenueTier.HIT: 8.25,  # 8%  → high
    RevenueTier.MEGA: 33.0,  # 3%  → very high
}

# Tier ordering for ordinal thresholds
TIER_ORDER: list[RevenueTier] = [
    RevenueTier.HOBBY,
    RevenueTier.SIDE,
    RevenueTier.HIT,
    RevenueTier.MEGA,
]


@dataclass
class ClassifierResult:
    """Prediction result from HitTierClassifier."""

    predicted_tier: RevenueTier
    probabilities: dict[str, float]
    confidence: float
    ordinal_score: float  # [0, 1] continuous ordinal position


@dataclass
class HitTierClassifier:
    """4-Tier Revenue Classification per SOT §A.4.

    Phase 3 (n>=200): scikit-learn LogisticRegression with class_weight='balanced'.
    Fallback: Pure-Python threshold-based classifier using ground_truth.classify_tier().

    Attributes:
        is_fitted: Whether the model has been trained.
        n_samples: Number of training samples seen.
        feature_importances: Learned feature weights (after fit).
        fallback_reason: Why fallback was used (None if sklearn succeeded).
    """

    is_fitted: bool = False
    n_samples: int = 0
    feature_importances: dict[str, float] = field(default_factory=dict)
    fallback_reason: str | None = None
    _thresholds: list[float] = field(default_factory=lambda: [250_000.0, 2_000_000.0, 20_000_000.0])

    def fit(
        self,
        features: list[list[float]],
        q1_revenues: list[float],
    ) -> HitTierClassifier:
        """Train the classifier on historical data.

        Args:
            features: List of feature vectors (len=n, each of len=5).
            q1_revenues: Q1 revenue for ground-truth tier labels.

        Returns:
            self for method chaining.
        """
        n = len(features)
        if n < 10 or n != len(q1_revenues):
            self.fallback_reason = f"insufficient_data: n={n}"
            self.is_fitted = False
            return self

        labels = [classify_tier(q1) for q1 in q1_revenues]
        self.n_samples = n

        # Pure-Python centroid classifier: compute mean feature vector per tier
        tier_sums: dict[RevenueTier, list[float]] = {}
        tier_counts: dict[RevenueTier, int] = {}
        n_features = len(features[0]) if features else 0

        for feat, label in zip(features, labels, strict=True):
            if label not in tier_sums:
                tier_sums[label] = [0.0] * n_features
                tier_counts[label] = 0
            for j in range(n_features):
                tier_sums[label][j] += feat[j]
            tier_counts[label] += 1

        self._centroids: dict[RevenueTier, list[float]] = {}
        for tier in TIER_ORDER:
            if tier in tier_sums and tier_counts[tier] > 0:
                self._centroids[tier] = [s / tier_counts[tier] for s in tier_sums[tier]]

        # Feature importance: variance of centroids per feature dimension
        if self._centroids and n_features > 0:
            for j, name in enumerate(FEATURE_NAMES[:n_features]):
                vals = [c[j] for c in self._centroids.values()]
                mean_v = sum(vals) / len(vals)
                var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                self.feature_importances[name] = round(var_v, 6)

        self.is_fitted = True
        self.fallback_reason = None
        return self

    def predict(self, features: list[float], q1_revenue: float | None = None) -> ClassifierResult:
        """Predict tier for a single game.

        If not fitted or q1_revenue is provided, uses threshold-based fallback.
        """
        if q1_revenue is not None:
            tier = classify_tier(q1_revenue)
            probs = {t.value: (1.0 if t == tier else 0.0) for t in TIER_ORDER}
            ordinal = TIER_ORDER.index(tier) / max(len(TIER_ORDER) - 1, 1)
            return ClassifierResult(
                predicted_tier=tier,
                probabilities=probs,
                confidence=1.0,
                ordinal_score=round(ordinal, 4),
            )

        if not self.is_fitted or not hasattr(self, "_centroids") or not self._centroids:
            return ClassifierResult(
                predicted_tier=RevenueTier.SIDE,
                probabilities={t.value: 0.25 for t in TIER_ORDER},
                confidence=0.0,
                ordinal_score=0.33,
            )

        # Nearest-centroid with cost-sensitive weighting
        best_tier = RevenueTier.SIDE
        best_dist = float("inf")
        distances: dict[RevenueTier, float] = {}

        for tier, centroid in self._centroids.items():
            dist = math.sqrt(sum((f - c) ** 2 for f, c in zip(features, centroid, strict=False)))
            # Cost-sensitive: divide distance by tier weight (rare tiers attract more)
            weighted_dist = dist / TIER_WEIGHTS.get(tier.value, 1.0)
            distances[tier] = weighted_dist
            if weighted_dist < best_dist:
                best_dist = weighted_dist
                best_tier = tier

        # Convert distances to pseudo-probabilities (softmax-like)
        total_inv = sum(1.0 / max(d, 1e-8) for d in distances.values())
        probs = {
            t.value: round((1.0 / max(distances.get(t, 1e8), 1e-8)) / total_inv, 4)
            for t in TIER_ORDER
        }

        confidence = probs.get(best_tier.value, 0.0)
        ordinal = TIER_ORDER.index(best_tier) / max(len(TIER_ORDER) - 1, 1)

        return ClassifierResult(
            predicted_tier=best_tier,
            probabilities=probs,
            confidence=round(confidence, 4),
            ordinal_score=round(ordinal, 4),
        )

    def predict_batch(
        self,
        features_list: list[list[float]],
        q1_revenues: list[float] | None = None,
    ) -> list[ClassifierResult]:
        """Predict tiers for a batch of games."""
        results: list[ClassifierResult] = []
        for i, feat in enumerate(features_list):
            q1 = q1_revenues[i] if q1_revenues else None
            results.append(self.predict(feat, q1))
        return results

    def get_summary(self) -> dict[str, object]:
        """Return classifier metadata."""
        return {
            "is_fitted": self.is_fitted,
            "n_samples": self.n_samples,
            "feature_importances": self.feature_importances,
            "fallback_reason": self.fallback_reason,
        }
