---
name: python-type-safety
description: LLMART 타입 안전성 가이드. mypy strict, Protocol, Generic, TypedDict for pipeline state. "type", "mypy", "protocol", "generic", "TypedDict" 키워드로 트리거.
---

# Python Type Safety for LLMART

## LLMART Type Strategy

- **mypy strict** mode in CI (`--strict`)
- **Protocol** for LLM evaluator ports (structural typing)
- **TypedDict** for pipeline state (LangGraph compatible)
- **Literal** for constrained values (L/M/H/E, GREEN/YELLOW/RED)
- **frozen dataclass** for value objects

## Pipeline State (TypedDict)

```python
# pipeline/state.py
from typing import TypedDict, Literal

Signal = Literal["GREEN", "YELLOW", "RED"]
Category = Literal["L", "M", "H", "E"]
Decision = Literal["APPROVE", "REVIEW", "REJECT"]

class LLMEvalResult(TypedDict):
    gameplay: Category
    innovation: Category
    monetize: Category
    polish: Category
    narrative: Category
    jury_score: float
    cot: str

class ValueResult(TypedDict):
    npv_3y: float
    value: float
    signal: Signal
    q1_p25: float
    q1_p50: float

class LLMARTState(TypedDict, total=False):
    # Input
    games: list[dict[str, object]]

    # Pre-filter
    prefiltered: list[dict[str, object]]

    # T1 ML
    ml_scores: dict[str, float]
    ml_ranked: list[dict[str, object]]

    # T2 LLM
    llm_evals: dict[str, LLMEvalResult]
    llm_passed: list[dict[str, object]]

    # T3 Human
    decisions: dict[str, Decision]
    selected: list[dict[str, object]]

    # Value
    values: dict[str, ValueResult]
    final_output: list[dict[str, object]]
```

## Protocol for LLM Ports

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMEvaluatorPort(Protocol):
    """Structural interface — no inheritance required."""

    async def evaluate(
        self,
        game_data: dict[str, object],
        rubric: str,
    ) -> dict[str, object]: ...

# Any class with matching signature satisfies this:
class GPTPrimaryEvaluator:
    async def evaluate(self, game_data: dict[str, object], rubric: str) -> dict[str, object]:
        ...  # OpenAI API call

class MockEvaluator:
    async def evaluate(self, game_data: dict[str, object], rubric: str) -> dict[str, object]:
        return {"gameplay": "H", ...}

# Both pass isinstance check at runtime:
assert isinstance(GPTPrimaryEvaluator(), LLMEvaluatorPort)
assert isinstance(MockEvaluator(), LLMEvaluatorPort)
```

## Value Objects (frozen dataclass)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Genre:
    name: str
    q1_y1_ratio: float

    def __post_init__(self) -> None:
        if not 0 < self.q1_y1_ratio < 1:
            raise ValueError(f"Invalid ratio: {self.q1_y1_ratio}")

@dataclass(frozen=True)
class SelectionScore:
    phi_ml: float
    phi_llm: float
    delta_cal: float
    w_ml: float = 0.6
    w_llm: float = 0.4

    @property
    def value(self) -> float:
        return self.w_ml * self.phi_ml + self.w_llm * self.phi_llm + self.delta_cal
```

## Generic Patterns

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class RankedList(Generic[T]):
    """Type-safe ranked list with cutoff."""

    def __init__(self, items: list[T], scores: list[float]) -> None:
        paired = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
        self._items = [item for item, _ in paired]
        self._scores = [score for _, score in paired]

    def top_k(self, k: int) -> list[T]:
        return self._items[:k]

    def percentile_rank(self, index: int) -> float:
        return 1.0 - (index / len(self._items))
```

## mypy Configuration

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
no_implicit_optional = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = ["openai.*", "anthropic.*", "langgraph.*"]
ignore_missing_imports = true
```

## Key Rules for LLMART

1. **All public functions annotated** — parameters + return types
2. **`Literal` for finite sets** — Signal, Category, Decision
3. **`TypedDict` for state** — LangGraph requires dict-like state
4. **`Protocol` over ABC** — structural typing for ports
5. **`frozen=True` for value objects** — Genre, SelectionScore
6. **Minimize `Any`** — use `object` or specific types

## Reference

- Based on [wshobson/agents python-type-safety](https://github.com/wshobson/agents)
- LLMART CI: mypy strict in `.github/workflows/ci.yml`
