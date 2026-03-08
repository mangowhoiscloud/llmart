"""Tests for shared pipeline utilities."""

from __future__ import annotations

import pytest

from llmart.pipeline.nodes._utils import CURRENT_YEAR, JURY_SCALE_MAX, wilson_lower_bound


def test_wilson_zero_total() -> None:
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_perfect_score() -> None:
    result = wilson_lower_bound(100, 100)
    assert 0.95 < result < 1.0


def test_wilson_known_value() -> None:
    # 95/100 positive, z=1.96 → known Wilson lower bound ~0.889
    result = wilson_lower_bound(95, 100)
    assert 0.88 < result < 0.92


@pytest.mark.parametrize(
    ("positive", "total", "expected_min", "expected_max"),
    [
        (50, 100, 0.39, 0.51),  # 50% positive
        (1, 10, 0.01, 0.20),  # low count, uncertain
        (990, 1000, 0.97, 1.0),  # high positive
    ],
)
def test_wilson_parametrize(
    positive: int, total: int, expected_min: float, expected_max: float
) -> None:
    result = wilson_lower_bound(positive, total)
    assert expected_min < result < expected_max


def test_jury_scale_max() -> None:
    assert JURY_SCALE_MAX == 4.0


def test_current_year_reasonable() -> None:
    assert 2024 <= CURRENT_YEAR <= 2030
