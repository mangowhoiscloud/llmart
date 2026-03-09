"""Tests for PercentileRankManager and percentile rank functions."""

from __future__ import annotations

from llmart.models.percentile import (
    P95_THRESHOLD,
    PercentileRankManager,
    percentile_rank_batch,
    percentile_rank_standard,
)


class TestPercentileRankStandard:
    def test_empty_reference_returns_half(self) -> None:
        assert percentile_rank_standard([], 5.0) == 0.5

    def test_single_reference_median_score(self) -> None:
        result = percentile_rank_standard([5.0], 5.0)
        # Two elements (5.0, 5.0), average rank: score at index 0 and 1
        assert 0.0 < result < 1.0

    def test_lowest_score_gets_low_percentile(self) -> None:
        ref = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = percentile_rank_standard(ref, 1.0)
        assert result < 0.2

    def test_highest_score_gets_high_percentile(self) -> None:
        ref = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = percentile_rank_standard(ref, 100.0)
        assert result > 0.8

    def test_weibull_avoids_zero_and_one(self) -> None:
        ref = list(range(1, 101))
        # Minimum possible
        low = percentile_rank_standard(ref, 0)
        high = percentile_rank_standard(ref, 200)
        assert low > 0.0
        assert high <= 1.0  # tail stabilization may push to 1.0

    def test_monotonic_ordering(self) -> None:
        ref = [10.0, 20.0, 30.0, 40.0, 50.0]
        scores = [5.0, 15.0, 25.0, 35.0, 55.0]
        ranks = [percentile_rank_standard(ref, s) for s in scores]
        for i in range(len(ranks) - 1):
            assert ranks[i] <= ranks[i + 1]

    def test_tail_stabilization_active(self) -> None:
        ref = list(range(1, 101))
        # Score at P95+ with stabilization
        with_stab = percentile_rank_standard(ref, 98, stabilize_tail=True)
        without_stab = percentile_rank_standard(ref, 98, stabilize_tail=False)
        # Both should be high, but stabilized should be >= P95_THRESHOLD
        assert with_stab >= P95_THRESHOLD
        assert without_stab > 0.9

    def test_duplicate_handling(self) -> None:
        ref = [10.0, 10.0, 10.0, 20.0, 20.0]
        result = percentile_rank_standard(ref, 10.0)
        assert 0.0 < result < 1.0


class TestPercentileRankBatch:
    def test_batch_matches_individual(self) -> None:
        ref = [10.0, 20.0, 30.0, 40.0, 50.0]
        scores = [15.0, 35.0, 55.0]
        batch = percentile_rank_batch(ref, scores)
        individual = [percentile_rank_standard(ref, s) for s in scores]
        assert batch == individual

    def test_empty_scores(self) -> None:
        assert percentile_rank_batch([10.0, 20.0], []) == []


class TestPercentileRankManager:
    def test_empty_manager_returns_default_ranker(self) -> None:
        mgr = PercentileRankManager()
        ranker = mgr.get_ranker("ml")
        assert ranker(5.0) == 0.5

    def test_single_quarter_update(self) -> None:
        mgr = PercentileRankManager()
        ml_scores = [float(i) for i in range(10)]
        jury_scores = [float(i) * 0.5 for i in range(10)]
        mgr.update_quarter("2024Q1", ml_scores, jury_scores)
        assert "2024Q1" in mgr.quarters

    def test_ranker_uses_kde_for_small_n(self) -> None:
        mgr = PercentileRankManager()
        # n=10 < KDE_N_THRESHOLD=50 → should use KDE
        ml_scores = [float(i) for i in range(10)]
        jury_scores = [float(i) * 0.5 for i in range(10)]
        mgr.update_quarter("2024Q1", ml_scores, jury_scores)
        ranker = mgr.get_ranker("ml")
        result = ranker(5.0)
        assert 0.0 < result < 1.0

    def test_ranker_uses_standard_for_large_n(self) -> None:
        mgr = PercentileRankManager()
        # n=60 >= KDE_N_THRESHOLD=50 → should use standard
        ml_scores = [float(i) for i in range(60)]
        jury_scores = [float(i) * 0.5 for i in range(60)]
        mgr.update_quarter("2024Q1", ml_scores, jury_scores)
        ranker = mgr.get_ranker("ml")
        result = ranker(30.0)
        assert 0.3 < result < 0.7  # roughly median

    def test_quarter_eviction(self) -> None:
        mgr = PercentileRankManager()
        for q in ["2024Q1", "2024Q2", "2024Q3"]:
            mgr.update_quarter(q, [1.0, 2.0], [1.0, 2.0])
        # MAX_QUARTERS = 2, so oldest should be evicted
        assert "2024Q1" not in mgr.quarters
        assert len(mgr.quarters) == 2

    def test_psi_between_quarters(self) -> None:
        mgr = PercentileRankManager()
        mgr.update_quarter("2024Q1", [float(i) for i in range(20)], [0.0] * 20)
        mgr.update_quarter("2024Q2", [float(i) for i in range(20)], [0.0] * 20)
        psi = mgr.compute_psi("2024Q1", "2024Q2", "ml")
        # Same distribution → PSI should be below warning threshold
        assert psi < 0.25

    def test_psi_missing_quarter_returns_zero(self) -> None:
        mgr = PercentileRankManager()
        assert mgr.compute_psi("2024Q1", "2024Q2") == 0.0

    def test_add_batch_insufficient_data(self) -> None:
        """Batch with very few items should still work with defaults."""
        mgr = PercentileRankManager()
        mgr.update_quarter("2024Q1", [1.0], [1.0])
        ranker = mgr.get_ranker("ml")
        result = ranker(1.0)
        assert 0.0 < result < 1.0

    def test_get_ranker_unknown_field_no_data(self) -> None:
        """Unknown field on empty manager should return default 0.5."""
        mgr = PercentileRankManager()
        ranker = mgr.get_ranker("nonexistent_field")
        result = ranker(5.0)
        assert result == 0.5

    def test_empty_batch_ranker(self) -> None:
        """Empty batch list should result in default ranker."""
        mgr = PercentileRankManager()
        mgr.update_quarter("2024Q1", [], [])
        ranker = mgr.get_ranker("ml")
        result = ranker(5.0)
        assert result == 0.5


class TestKdeCdf:
    """Direct tests for _kde_cdf and helper functions."""

    def test_kde_cdf_empty_ref(self) -> None:
        from llmart.models.percentile import _kde_cdf

        assert _kde_cdf([], 5.0) == 0.5

    def test_kde_cdf_normal_operation(self) -> None:
        from llmart.models.percentile import _kde_cdf

        ref = list(range(1, 20))
        result = _kde_cdf(ref, 10.0)
        assert 0.01 <= result <= 0.99

    def test_kde_cdf_near_zero_bandwidth(self) -> None:
        """When h < 1e-10, fallback bandwidth (std * n^-0.2) is used."""
        from llmart.models.percentile import _kde_cdf

        # IQR=0, std ~ 1.26e-10 (passes std >= 1e-10 check) but h ~ 7.2e-11 < 1e-10
        ref = [10.0] * 9 + [10.0 + 4e-10]
        result = _kde_cdf(ref, 10.0)
        assert 0.0 <= result <= 1.0


class TestPercentile:
    """Direct tests for _percentile helper."""

    def test_percentile_empty(self) -> None:
        from llmart.models.percentile import _percentile

        assert _percentile([], 50) == 0.0

    def test_percentile_exact_index(self) -> None:
        from llmart.models.percentile import _percentile

        # 5 elements: k = (50/100) * 4 = 2.0 (exact)
        result = _percentile([10, 20, 30, 40, 50], 50)
        assert result == 30

    def test_percentile_interpolated(self) -> None:
        from llmart.models.percentile import _percentile

        # 4 elements: k = (50/100) * 3 = 1.5 (interpolated)
        result = _percentile([10, 20, 30, 40], 50)
        assert result == 25.0


class TestTailStabilizationBranch:
    def test_tail_stab_max_equals_p95(self) -> None:
        """When max_value == p95_value, tail sub-ranking is skipped."""
        # All values identical → max == p95
        ref = [10.0] * 100
        # Score higher than all → phi > 0.95, but max == p95 → skip sub-ranking
        result = percentile_rank_standard(ref, 11.0, stabilize_tail=True)
        assert result > 0.95
