"""Tests for jury_star() confidence-weighted consensus."""

from __future__ import annotations

import pytest

from llmart.models.jury import (
    DIMS,
    JudgeInput,
    jury_star,
    jury_star_from_passes,
)


def _make_judge(scores: dict[str, float], confs: dict[str, float]) -> JudgeInput:
    """Helper to create JudgeInput with defaults for missing dims."""
    full_scores = {d: scores.get(d, 2.5) for d in DIMS}
    full_confs = {d: confs.get(d, 0.8) for d in DIMS}
    return JudgeInput(dim_scores=full_scores, dim_conf=full_confs)


class TestJuryStar:
    def test_identical_judges_returns_same_score(self) -> None:
        scores = dict.fromkeys(DIMS, 3.0)
        confs = dict.fromkeys(DIMS, 0.9)
        judge = _make_judge(scores, confs)
        result = jury_star(judge, judge)
        assert result.delta == 0.0
        assert result.jury_final == pytest.approx(3.0, abs=0.01)

    def test_high_conf_judge_dominates(self) -> None:
        judge_a = _make_judge(
            dict.fromkeys(DIMS, 4.0),
            dict.fromkeys(DIMS, 0.95),
        )
        judge_b = _make_judge(
            dict.fromkeys(DIMS, 1.0),
            dict.fromkeys(DIMS, 0.1),
        )
        result = jury_star(judge_a, judge_b)
        # High-confidence judge_a (score=4.0) should dominate
        assert result.jury_final > 3.0

    def test_delta_measures_disagreement(self) -> None:
        judge_a = _make_judge(dict.fromkeys(DIMS, 4.0), dict.fromkeys(DIMS, 0.8))
        judge_b = _make_judge(dict.fromkeys(DIMS, 1.0), dict.fromkeys(DIMS, 0.8))
        result = jury_star(judge_a, judge_b)
        assert result.delta > 2.0

    def test_unknown_dims_get_low_weight(self) -> None:
        scores = dict.fromkeys(DIMS, 3.0)
        confs = dict.fromkeys(DIMS, 0.9)
        judge = _make_judge(scores, confs)
        # Mark all dims as unknown
        result = jury_star(judge, judge, unknown_dims=set(DIMS))
        assert result.unknown_ratio == 1.0

    def test_partial_unknown_dims(self) -> None:
        judge = _make_judge(
            dict.fromkeys(DIMS, 3.0),
            dict.fromkeys(DIMS, 0.8),
        )
        result = jury_star(judge, judge, unknown_dims={"gameplay", "innovation"})
        assert result.unknown_ratio == pytest.approx(2 / 5, abs=0.01)

    def test_result_has_dim_deltas(self) -> None:
        judge_a = _make_judge(dict.fromkeys(DIMS, 3.0), dict.fromkeys(DIMS, 0.8))
        judge_b = _make_judge(dict.fromkeys(DIMS, 2.0), dict.fromkeys(DIMS, 0.8))
        result = jury_star(judge_a, judge_b)
        assert len(result.dim_deltas) == len(DIMS)
        for dim in DIMS:
            assert result.dim_deltas[dim] == pytest.approx(1.0, abs=0.01)

    def test_zero_confidence_fallback_to_average(self) -> None:
        judge_a = _make_judge(dict.fromkeys(DIMS, 4.0), dict.fromkeys(DIMS, 0.0))
        judge_b = _make_judge(dict.fromkeys(DIMS, 2.0), dict.fromkeys(DIMS, 0.0))
        result = jury_star(judge_a, judge_b)
        # With zero confidence, should fall back to simple average
        assert result.conf_scalar == 0.0


class TestJuryStarFromPasses:
    def test_single_pass_returns_average(self) -> None:
        passes = [{"gameplay": 3, "innovation": 4, "monetization": 2, "polish": 3, "narrative": 3}]
        result = jury_star_from_passes(passes)
        assert result.delta == 0.0

    def test_three_passes_k3(self) -> None:
        passes = [
            {"gameplay": 3, "innovation": 3, "monetization": 3, "polish": 3, "narrative": 3},
            {"gameplay": 3, "innovation": 3, "monetization": 3, "polish": 3, "narrative": 3},
            {"gameplay": 4, "innovation": 4, "monetization": 4, "polish": 4, "narrative": 4},
        ]
        result = jury_star_from_passes(passes)
        # First 2 passes → judge_a (avg=3.0), last pass → judge_b (4.0)
        assert result.jury_final > 3.0
        assert result.delta > 0.0

    def test_consistent_passes_high_confidence(self) -> None:
        passes = [
            {"gameplay": 3, "innovation": 3, "monetization": 3, "polish": 3, "narrative": 3},
            {"gameplay": 3, "innovation": 3, "monetization": 3, "polish": 3, "narrative": 3},
            {"gameplay": 3, "innovation": 3, "monetization": 3, "polish": 3, "narrative": 3},
        ]
        result = jury_star_from_passes(passes)
        # Perfect consistency → high confidence
        assert result.conf_scalar > 0.8

    def test_inconsistent_passes_low_confidence(self) -> None:
        passes = [
            {"gameplay": 1, "innovation": 1, "monetization": 1, "polish": 1, "narrative": 1},
            {"gameplay": 4, "innovation": 4, "monetization": 4, "polish": 4, "narrative": 4},
            {"gameplay": 2, "innovation": 2, "monetization": 2, "polish": 2, "narrative": 2},
        ]
        result = jury_star_from_passes(passes)
        assert result.conf_scalar < 0.5

    def test_unknown_dims_forwarded(self) -> None:
        passes = [
            {"gameplay": 3, "innovation": 3, "monetization": 3, "polish": 3, "narrative": 3},
            {"gameplay": 3, "innovation": 3, "monetization": 3, "polish": 3, "narrative": 3},
        ]
        result = jury_star_from_passes(passes, unknown_dims={"narrative"})
        assert result.unknown_ratio == pytest.approx(0.2, abs=0.01)

    def test_empty_passes(self) -> None:
        result = jury_star_from_passes([])
        assert result.delta == 0.0
        assert result.jury_final == 0.0
