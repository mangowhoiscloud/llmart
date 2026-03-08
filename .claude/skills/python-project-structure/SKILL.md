---
name: python-project-structure
description: LLMART 프로젝트 구조 가이드. src layout, module boundaries, __all__ 정의, import conventions. "project structure", "module", "import", "__all__", "package" 키워드로 트리거.
---

# Python Project Structure for LLMART

## LLMART Project Layout

```
llmart/
├── .claude/
│   └── skills/              # Claude Code skills
├── .github/
│   └── workflows/
│       ├── ci.yml           # 5-job CI pipeline
│       └── release.yml      # Auto-tag + release
├── src/
│   └── llmart/
│       ├── __init__.py      # Package root, __all__
│       ├── cli.py           # Typer CLI entry point
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── graph.py     # LangGraph StateGraph
│       │   ├── state.py     # LLMARTState TypedDict
│       │   └── nodes/
│       │       ├── __init__.py
│       │       ├── prefilter.py
│       │       ├── ml_scoring.py
│       │       ├── llm_judge.py
│       │       ├── enrichment.py
│       │       ├── human_review.py
│       │       └── value.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── game.py      # Game entity + value objects
│       │   ├── scoring.py   # Selection Score formula
│       │   └── value.py     # NPV, VaR, Signal
│       ├── prompts/
│       │   ├── __init__.py
│       │   ├── rubric_5dim.py
│       │   └── calibrator.py
│       └── data/
│           ├── sample_games.json
│           └── genre_params.json
├── tests/
│   ├── conftest.py
│   ├── test_models/
│   ├── test_pipeline/
│   └── test_integration/
├── docs/
│   └── implementation-plan.md
├── pyproject.toml
├── uv.lock
└── README.md
```

## Module Boundaries

| Module | Responsibility | Dependencies |
|--------|---------------|-------------|
| `models/` | Domain logic (pure Python) | None |
| `pipeline/state.py` | State schema | `models/` |
| `pipeline/nodes/` | Business logic nodes | `models/` |
| `pipeline/graph.py` | Orchestration | `pipeline/nodes/`, `pipeline/state` |
| `prompts/` | LLM prompt templates | None (string templates) |
| `data/` | Static data files | None |
| `cli.py` | Presentation layer | `pipeline/graph` |

## Public API Design (__all__)

```python
# src/llmart/__init__.py
"""LLMART: Game Selection & Value Inference System."""

from llmart.models.game import Game, Genre
from llmart.models.scoring import selection_score
from llmart.models.value import npv_3y, value_function, investment_signal
from llmart.pipeline.graph import create_llmart_graph
from llmart.pipeline.state import LLMARTState

__all__ = [
    "Game",
    "Genre",
    "selection_score",
    "npv_3y",
    "value_function",
    "investment_signal",
    "create_llmart_graph",
    "LLMARTState",
]

# src/llmart/models/__init__.py
from .game import Game, Genre
from .scoring import selection_score
from .value import npv_3y, value_function, investment_signal

__all__ = [
    "Game", "Genre",
    "selection_score",
    "npv_3y", "value_function", "investment_signal",
]
```

## Import Conventions

```python
# GOOD: Absolute imports
from llmart.models.scoring import selection_score
from llmart.pipeline.state import LLMARTState
from llmart.pipeline.nodes.prefilter import prefilter_node

# AVOID: Relative imports (fragile when moving modules)
from ..models import Game
from . import state

# GOOD: Package-level import for consumers
from llmart import Game, selection_score, create_llmart_graph
```

## One Concept Per File

```python
# models/scoring.py — Selection Score만
def selection_score(...) -> float: ...

# models/value.py — Value Inference만
def npv_3y(...) -> float: ...
def value_function(...) -> float: ...
def investment_signal(...) -> str: ...

# AVOID: 모든 수식을 한 파일에
# models/formulas.py ← 이렇게 하지 않음
```

## Key Rules

1. **src layout** — `src/llmart/` 아래에 패키지 배치 (build isolation)
2. **Flat hierarchy** — `models/`, `pipeline/`, `prompts/` 수준까지만 (깊은 중첩 금지)
3. **`__all__` 필수** — 모든 `__init__.py`에 명시적 public API
4. **Absolute imports** — 항상 `from llmart.xxx` 형태
5. **Domain first** — `models/`는 외부 의존성 없이 순수 Python

## Reference

- Based on [wshobson/agents python-project-structure](https://github.com/wshobson/agents)
- LLMART pyproject.toml: `[project] packages = [{include = "llmart", from = "src"}]`
