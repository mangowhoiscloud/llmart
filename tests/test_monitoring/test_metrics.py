"""Tests for validation suite metrics."""

from __future__ import annotations

import pytest

from llmart.monitoring.metrics import (
    check_pass_condition,
    classify_metric,
    compute_ndcg,
    compute_validation_suite,
)


def test_ndcg_perfect_ranking() -> None:
    """Ideal ranking -> NDCG = 1.0."""
    scores = [4.0, 3.0, 2.0, 1.0]
    relevance = [4.0, 3.0, 2.0, 1.0]
    assert compute_ndcg(scores, relevance) == 1.0


def test_ndcg_reversed_ranking() -> None:
    """Worst ranking -> NDCG < 1."""
    scores = [1.0, 2.0, 3.0, 4.0]
    relevance = [4.0, 3.0, 2.0, 1.0]
    ndcg = compute_ndcg(scores, relevance)
    assert 0.0 < ndcg < 1.0


def test_ndcg_empty_input() -> None:
    assert compute_ndcg([], []) == 0.0


def test_ndcg_mismatched_lengths() -> None:
    assert compute_ndcg([1.0], [1.0, 2.0]) == 0.0


def test_ndcg_at_k() -> None:
    scores = list(range(100, 0, -1))
    relevance = list(range(100, 0, -1))
    ndcg = compute_ndcg(scores, relevance, k=30)
    assert ndcg == 1.0


def test_ndcg_zero_relevance() -> None:
    scores = [1.0, 2.0, 3.0]
    relevance = [0.0, 0.0, 0.0]
    assert compute_ndcg(scores, relevance) == 0.0


def test_validation_suite_returns_all_metrics() -> None:
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    revenues = [1_000_000, 800_000, 600_000, 400_000, 200_000]
    result = compute_validation_suite(scores, revenues, k=5)
    expected_keys = {
        "spearman_rho",
        "pearson_log",
        "precision_at_k",
        "ndcg_log_at_k",
        "ndcg_hit_at_k",
        "hit_rate_at_k",
    }
    assert set(result.keys()) == expected_keys


def test_validation_suite_empty() -> None:
    result = compute_validation_suite([], [])
    assert all(v == 0.0 for v in result.values())


def test_validation_suite_perfect_correlation() -> None:
    scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    revenues = [5_000_000, 4_000_000, 3_000_000, 2_000_000, 1_000_000]
    result = compute_validation_suite(scores, revenues, k=5)
    assert result["spearman_rho"] == 1.0


def test_check_pass_condition_passing() -> None:
    metrics = {
        "spearman_rho": 0.50,
        "pearson_log": 0.45,
        "precision_at_k": 0.50,
        "ndcg_log_at_k": 0.70,
        "ndcg_hit_at_k": 0.60,
        "hit_rate_at_k": 0.20,
    }
    assert check_pass_condition(metrics) is True


def test_check_pass_condition_failing() -> None:
    metrics = {
        "spearman_rho": 0.10,
        "pearson_log": 0.10,
        "precision_at_k": 0.10,
        "ndcg_log_at_k": 0.10,
        "ndcg_hit_at_k": 0.10,
        "hit_rate_at_k": 0.05,
    }
    assert check_pass_condition(metrics) is False


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("spearman_rho", 0.60, "strong"),
        ("spearman_rho", 0.40, "conditional"),
        ("spearman_rho", 0.20, "fail"),
        ("ndcg_log_at_k", 0.80, "strong"),
        ("ndcg_log_at_k", 0.60, "conditional"),
    ],
)
def test_classify_metric(name: str, value: float, expected: str) -> None:
    assert classify_metric(name, value) == expected


def test_classify_metric_unknown() -> None:
    assert classify_metric("nonexistent", 0.5) == "unknown"


def test_ndcg_dcg_zero_but_idcg_nonzero() -> None:
    """When top-k has zero DCG but IDCG is non-zero → NDCG = 0."""
    # Score ranking puts zero-relevance items first
    # scores sorted desc: [4,3,2,1], relevance: [0,0,0,10]
    # Top-3 by score: (4,0), (3,0), (2,0) → DCG=0 → returns 0
    ndcg = compute_ndcg([4.0, 3.0, 2.0, 1.0], [0.0, 0.0, 0.0, 10.0], k=3)
    assert ndcg == 0.0


def test_ndcg_dcg_zero_nonzero_input() -> None:
    """DCG = 0 when all top-k relevance is 0 but some relevance exists elsewhere."""
    # Scores sorted desc: [4, 3, 2, 1] → top-k relevance: [0, 0, 0, 1]
    scores = [4.0, 3.0, 2.0, 1.0]
    relevance = [0.0, 0.0, 0.0, 1.0]
    ndcg = compute_ndcg(scores, relevance, k=3)
    # DCG of top-3 = 0 (all zero relevance) → NDCG = 0
    assert ndcg == 0.0


def test_spearman_single_element() -> None:
    """Spearman with < 2 elements should return 0.0."""
    from llmart.monitoring.metrics import _spearman_rho

    assert _spearman_rho([1.0], [1.0]) == 0.0
    assert _spearman_rho([], []) == 0.0


def test_pearson_single_element() -> None:
    """Pearson with < 2 elements should return 0.0."""
    from llmart.monitoring.metrics import _pearson

    assert _pearson([1.0], [1.0]) == 0.0
    assert _pearson([], []) == 0.0


def test_pearson_zero_variance() -> None:
    """Pearson should return 0.0 when one variable has zero variance."""
    from llmart.monitoring.metrics import _pearson

    assert _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert _pearson([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == 0.0
