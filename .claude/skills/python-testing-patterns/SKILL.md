---
name: python-testing-patterns
description: LLMART 테스트 전략. pytest fixtures, mocking LLM APIs, parametrized scoring tests, async pipeline testing. "test", "pytest", "fixture", "mock", "coverage" 키워드로 트리거.
---

# Python Testing Patterns for LLMART

## LLMART Test Architecture

```
tests/
├── conftest.py              # Shared fixtures (games, mock LLM)
├── test_models/
│   ├── test_game.py         # Domain model tests
│   ├── test_scoring.py      # Selection Score formula tests
│   └── test_value.py        # NPV, VaR, Signal tests
├── test_pipeline/
│   ├── test_prefilter.py    # Pre-filter 2L tests
│   ├── test_ml_scoring.py   # T1 ML scoring tests
│   ├── test_llm_judge.py    # T2 LLM-as-Judge tests (mocked)
│   ├── test_human_review.py # T3 Decision Matrix tests
│   └── test_value_node.py   # Value Inference node tests
└── test_integration/
    └── test_full_pipeline.py # End-to-end pipeline test
```

## Core Fixtures (conftest.py)

```python
import pytest

@pytest.fixture
def sample_game() -> dict:
    """Single game fixture for unit tests."""
    return {
        "game_id": "test-001",
        "title": "Test RPG",
        "genre": "balanced",
        "followers": 5000,
        "wishlists": 12000,
        "features": {
            "follower_growth_rate": 0.75,
            "wishlist_velocity": 0.80,
            "dev_wilson_score": 0.65,
            "sentiment_score": 0.70,
            "youtube_views_norm": 0.50,
            "reddit_mentions_norm": 0.40,
            "twitch_peak_norm": 0.30,
        },
    }

@pytest.fixture
def sample_games(sample_game) -> list[dict]:
    """Pool of 100 games for pipeline tests."""
    import copy
    games = []
    for i in range(100):
        g = copy.deepcopy(sample_game)
        g["game_id"] = f"game-{i:03d}"
        g["features"]["follower_growth_rate"] = 0.1 + (i / 100) * 0.9
        games.append(g)
    return games

@pytest.fixture
def mock_llm_evaluator():
    """Mock LLM evaluator — deterministic, no API calls."""
    from unittest.mock import AsyncMock
    evaluator = AsyncMock()
    evaluator.evaluate.return_value = {
        "gameplay": "H", "innovation": "M",
        "monetize": "M", "polish": "H", "narrative": "M",
        "jury_score": 2.65,
        "cot": "Test evaluation",
    }
    return evaluator
```

## Domain Model Tests (Parametrized)

```python
# test_models/test_scoring.py
import pytest

@pytest.mark.parametrize("phi_ml,phi_llm,delta,expected", [
    (0.90, 0.85, 0.0, 0.6*0.90 + 0.4*0.85),   # Standard
    (1.00, 1.00, 0.0, 1.0),                       # Perfect
    (0.00, 0.00, 0.0, 0.0),                       # Zero
    (0.50, 0.50, 0.05, 0.55),                     # With calibration
])
def test_selection_score(phi_ml, phi_llm, delta, expected):
    from llmart.models.scoring import selection_score
    result = selection_score(phi_ml, phi_llm, delta)
    assert abs(result - expected) < 1e-10

@pytest.mark.parametrize("q1_p25,q1_p50,expected_signal", [
    (300_000, 500_000, "GREEN"),
    (200_000, 300_000, "YELLOW"),
    (100_000, 200_000, "RED"),
    (250_000, 250_000, "GREEN"),   # Boundary: P25 == threshold
])
def test_investment_signal(q1_p25, q1_p50, expected_signal):
    from llmart.models.value import investment_signal
    assert investment_signal(q1_p25, q1_p50) == expected_signal
```

## Mocking LLM API Calls

```python
# test_pipeline/test_llm_judge.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_pass3_direct_path(mock_llm_evaluator):
    """All 3 evaluations agree → direct path, no calibrator."""
    from llmart.pipeline.nodes.llm_judge import evaluate_with_pass3

    # All 3 calls return same result
    result = await evaluate_with_pass3(
        game_data={"title": "Test"},
        primary=mock_llm_evaluator,
        calibrator=None,
        k=3,
    )

    assert result["path"] == "direct"
    assert mock_llm_evaluator.evaluate.call_count == 3

@pytest.mark.asyncio
async def test_pass3_calibration_path():
    """Disagreement → routes to calibrator."""
    primary = AsyncMock()
    primary.evaluate.side_effect = [
        {"gameplay": "H", "innovation": "M", ...},
        {"gameplay": "M", "innovation": "M", ...},  # Disagrees
        {"gameplay": "H", "innovation": "M", ...},
    ]

    calibrator = AsyncMock()
    calibrator.evaluate.return_value = {
        "action": "ADJUST",
        "final_scores": {"gameplay": "H", ...},
    }

    result = await evaluate_with_pass3(
        game_data={"title": "Test"},
        primary=primary,
        calibrator=calibrator,
        k=3,
    )

    assert result["path"] == "calibrated"
    calibrator.evaluate.assert_called_once()
```

## Pipeline Integration Test

```python
# test_integration/test_full_pipeline.py
@pytest.mark.asyncio
async def test_full_pipeline_smoke(sample_games, mock_llm_evaluator):
    """End-to-end pipeline with mock LLM."""
    from llmart.pipeline.graph import create_llmart_graph

    graph = create_llmart_graph(
        primary_evaluator=mock_llm_evaluator,
        calibrator=mock_llm_evaluator,
    )

    result = await graph.ainvoke({"games": sample_games})

    # Verify funnel reduction
    assert len(result["prefiltered"]) <= len(sample_games)
    assert len(result["ml_ranked"]) <= len(result["prefiltered"])
    assert len(result["llm_passed"]) <= len(result["ml_ranked"])
    assert all(r["signal"] in ("GREEN", "YELLOW", "RED") for r in result["final_output"])
```

## pytest Configuration

```toml
# pyproject.toml [tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "slow: marks tests requiring real API calls",
    "integration: marks integration tests",
]
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
    "--cov=llmart",
    "--cov-report=term-missing",
]
```

## Best Practices for LLMART

1. **Never call real APIs in CI** — always mock LLM evaluators
2. **Parametrize formula tests** — cover boundary values (250K threshold)
3. **Test each node independently** — nodes are pure functions on state
4. **Test full pipeline with mocks** — smoke test the LangGraph flow
5. **Use `pytest.approx`** — floating point comparisons for NPV/VaR
6. **Fixture composition** — build complex fixtures from simple ones

## Reference

- Based on [wshobson/agents python-testing-patterns](https://github.com/wshobson/agents)
- LLMART CI: `.github/workflows/ci.yml` (test matrix 3.12/3.13)
