"""Tests for ground truth revenue tier model."""

from __future__ import annotations

import pytest

from llmart.models.ground_truth import (
    GENRE_Q1_RATIOS,
    RevenueTier,
    classify_tier,
    q1_to_y1,
)


def test_classify_tier_hobby() -> None:
    assert classify_tier(100_000) == RevenueTier.HOBBY
    assert classify_tier(0) == RevenueTier.HOBBY
    assert classify_tier(249_999) == RevenueTier.HOBBY


def test_classify_tier_side() -> None:
    assert classify_tier(250_000) == RevenueTier.SIDE
    assert classify_tier(1_999_999) == RevenueTier.SIDE


def test_classify_tier_hit() -> None:
    assert classify_tier(2_000_000) == RevenueTier.HIT
    assert classify_tier(19_999_999) == RevenueTier.HIT


def test_classify_tier_mega() -> None:
    assert classify_tier(20_000_000) == RevenueTier.MEGA
    assert classify_tier(100_000_000) == RevenueTier.MEGA


def test_q1_to_y1_balanced() -> None:
    y1 = q1_to_y1(450_000, "balanced")
    # Y1 = 450K / 0.45 = 1M
    assert abs(y1 - 1_000_000) < 1


def test_q1_to_y1_front_loaded() -> None:
    y1 = q1_to_y1(550_000, "front-loaded")
    assert abs(y1 - 1_000_000) < 1


def test_q1_to_y1_live_service() -> None:
    y1 = q1_to_y1(350_000, "live-service")
    assert abs(y1 - 1_000_000) < 1


def test_q1_to_y1_unknown_genre_defaults_balanced() -> None:
    y1 = q1_to_y1(450_000, "unknown_genre_type")
    expected = 450_000 / GENRE_Q1_RATIOS["balanced"]
    assert abs(y1 - expected) < 1


@pytest.mark.parametrize("genre_type", ["front-loaded", "balanced", "live-service"])
def test_genre_q1_ratios_range(genre_type: str) -> None:
    ratio = GENRE_Q1_RATIOS[genre_type]
    assert 0 < ratio < 1


def test_q1_to_y1_zero_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero or negative ratio should return 0.0."""
    import llmart.models.ground_truth as gt_mod

    monkeypatch.setitem(gt_mod.GENRE_Q1_RATIOS, "broken", 0.0)
    assert q1_to_y1(100_000.0, "broken") == 0.0


def test_revenue_tier_values() -> None:
    assert RevenueTier.HOBBY == "Hobby"
    assert RevenueTier.SIDE == "Side"
    assert RevenueTier.HIT == "Hit"
    assert RevenueTier.MEGA == "Mega"
