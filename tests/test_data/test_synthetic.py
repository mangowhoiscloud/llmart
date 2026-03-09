"""Tests for synthetic data generator."""

from __future__ import annotations

from collections import Counter

from llmart.data.synthetic import (
    _LCG,
    _generate_y1_revenue,
    _stable_seed,
    generate_quarter_scores,
    generate_quarterly_feedback,
    generate_training_dataset,
)


class TestLCG:
    def test_deterministic(self) -> None:
        rng1 = _LCG(42)
        rng2 = _LCG(42)
        assert [rng1.random() for _ in range(10)] == [rng2.random() for _ in range(10)]

    def test_randint_range(self) -> None:
        rng = _LCG(42)
        for _ in range(100):
            v = rng.randint(5, 10)
            assert 5 <= v <= 10

    def test_choice(self) -> None:
        rng = _LCG(42)
        items = ["a", "b", "c"]
        result = rng.choice(items)
        assert result in items

    def test_sample_no_replacement(self) -> None:
        rng = _LCG(42)
        items = list(range(20))
        sampled = rng.sample(items, 5)
        assert len(sampled) == 5
        assert len(set(sampled)) == 5

    def test_sample_k_exceeds_n(self) -> None:
        rng = _LCG(42)
        items = [1, 2, 3]
        sampled = rng.sample(items, 10)
        assert len(sampled) == 3

    def test_gauss_distribution(self) -> None:
        rng = _LCG(42)
        values = [rng.gauss(0, 1) for _ in range(1000)]
        mean = sum(values) / len(values)
        assert abs(mean) < 0.2  # rough check

    def test_random_in_unit_interval(self) -> None:
        rng = _LCG(42)
        for _ in range(100):
            v = rng.random()
            assert 0.0 <= v < 1.0


class TestStableSeed:
    def test_deterministic(self) -> None:
        assert _stable_seed("hello") == _stable_seed("hello")

    def test_different_inputs(self) -> None:
        assert _stable_seed("a") != _stable_seed("b")


class TestGenerateY1Revenue:
    def test_minimum_floor(self) -> None:
        rng = _LCG(42)
        for _ in range(100):
            rev = _generate_y1_revenue(rng, "balanced")
            assert rev >= 5000.0

    def test_cap_at_200m(self) -> None:
        rng = _LCG(42)
        for _ in range(1000):
            rev = _generate_y1_revenue(rng, "balanced")
            assert rev <= 200_000_000.0

    def test_different_q1_types(self) -> None:
        for q1_type in ["front-loaded", "balanced", "live-service"]:
            rng = _LCG(42)
            rev = _generate_y1_revenue(rng, q1_type)
            assert rev >= 5000.0

    def test_unknown_q1_type_uses_default(self) -> None:
        rng = _LCG(42)
        rev = _generate_y1_revenue(rng, "unknown_type")
        assert rev >= 5000.0


class TestGenerateTrainingDataset:
    def test_correct_count(self) -> None:
        games = generate_training_dataset(n=50, seed=42)
        assert len(games) == 50

    def test_required_fields(self) -> None:
        games = generate_training_dataset(n=10, seed=42)
        required = {
            "game_id",
            "title",
            "genre",
            "developer",
            "steam_rating",
            "review_count",
            "price_usd",
            "release_year",
            "tags",
            "developer_successes",
            "developer_games_released",
            "genre_q1_type",
            "estimated_y1_revenue",
            "hit_tier",
            "quarter",
        }
        for g in games:
            assert required.issubset(set(g.keys()))

    def test_tier_distribution(self) -> None:
        games = generate_training_dataset(n=500, seed=42)
        tiers = Counter(g["hit_tier"] for g in games)
        # Should have all 4 tiers represented
        assert tiers["Hobby"] > 200  # ~60%+
        assert tiers["Side"] > 50  # ~20%+
        assert tiers["Hit"] > 10  # ~5%+
        # Mega may be 0 at small n, so just check it exists at 500
        assert "Mega" in tiers

    def test_all_genres_represented(self) -> None:
        games = generate_training_dataset(n=500, seed=42)
        genres = {g["genre"] for g in games}
        assert len(genres) >= 20  # 27 total, most should appear

    def test_quarter_distribution(self) -> None:
        games = generate_training_dataset(n=100, seed=42)
        quarters = Counter(g["quarter"] for g in games)
        # Default 5 quarters, 100 games → 20 each
        assert len(quarters) == 5
        for _q, count in quarters.items():
            assert count == 20

    def test_custom_quarters(self) -> None:
        games = generate_training_dataset(n=10, seed=42, quarters=["Q1", "Q2"])
        quarters = {g["quarter"] for g in games}
        assert quarters == {"Q1", "Q2"}

    def test_deterministic(self) -> None:
        games1 = generate_training_dataset(n=20, seed=42)
        games2 = generate_training_dataset(n=20, seed=42)
        assert games1 == games2

    def test_different_seeds_differ(self) -> None:
        games1 = generate_training_dataset(n=10, seed=42)
        games2 = generate_training_dataset(n=10, seed=99)
        assert games1 != games2

    def test_field_value_ranges(self) -> None:
        games = generate_training_dataset(n=50, seed=42)
        for g in games:
            assert 0.40 <= g["steam_rating"] <= 0.99
            assert g["review_count"] >= 50
            assert 0.99 <= g["price_usd"] <= 59.99
            assert 2020 <= g["release_year"] <= 2025
            assert len(g["tags"]) >= 1
            assert g["developer_successes"] >= 0
            assert g["developer_games_released"] >= 1


class TestGenerateQuarterlyFeedback:
    def test_correct_count(self) -> None:
        games = generate_training_dataset(n=50, seed=42)
        feedback = generate_quarterly_feedback(games, seed=99)
        assert len(feedback) == 50

    def test_required_fields(self) -> None:
        games = generate_training_dataset(n=10, seed=42)
        feedback = generate_quarterly_feedback(games, seed=99)
        required = {
            "game_id",
            "quarter",
            "predicted_score",
            "actual_revenue",
            "predicted_revenue",
            "prediction_error",
        }
        for f in feedback:
            assert required.issubset(set(f.keys()))

    def test_prediction_error_nonneg(self) -> None:
        games = generate_training_dataset(n=50, seed=42)
        feedback = generate_quarterly_feedback(games, seed=99)
        for f in feedback:
            assert f["prediction_error"] >= 0.0

    def test_predicted_score_range(self) -> None:
        games = generate_training_dataset(n=50, seed=42)
        feedback = generate_quarterly_feedback(games, seed=99)
        for f in feedback:
            assert 0.0 <= f["predicted_score"] <= 1.0


class TestGenerateQuarterScores:
    def test_correct_quarters(self) -> None:
        games = generate_training_dataset(n=50, seed=42)
        qs = generate_quarter_scores(games, seed=77)
        assert len(qs) == 5  # default 5 quarters

    def test_score_ranges(self) -> None:
        games = generate_training_dataset(n=50, seed=42)
        qs = generate_quarter_scores(games, seed=77)
        for _q, data in qs.items():
            for ml in data["ml_scores"]:
                assert 0.0 <= ml <= 1.0
            for jury in data["jury_scores"]:
                assert 1.0 <= jury <= 4.0

    def test_correct_game_count_per_quarter(self) -> None:
        games = generate_training_dataset(n=50, seed=42)
        qs = generate_quarter_scores(games, seed=77)
        total = sum(len(d["ml_scores"]) for d in qs.values())
        assert total == 50
