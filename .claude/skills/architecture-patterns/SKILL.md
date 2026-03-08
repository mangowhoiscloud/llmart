---
name: architecture-patterns
description: Backend architecture patterns for LLMART pipeline. Clean Architecture, Hexagonal (Ports/Adapters), Domain-Driven Design. Use when designing pipeline nodes, structuring domain models, or implementing dependency injection. "architecture", "clean", "hexagonal", "ports adapters", "DDD", "dependency injection" 키워드로 트리거.
---

# Architecture Patterns for LLMART

## LLMART Architecture Context

LLMART는 다단계 퍼널(Pre-filter → T1 ML → T2 LLM → T3 Human → Value Inference) 파이프라인.
각 스테이지는 독립적 도메인 모듈로 설계하되, LangGraph StateGraph가 오케스트레이션.

```
┌─────────────────────────────────────────────────────────────┐
│                    Dependency Rule                           │
│        Dependencies ALWAYS point inward (to Domain)         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CLI (Typer)  ──▶  Pipeline (LangGraph)  ──▶  Domain        │
│  Streamlit        Nodes (Use Cases)        (Models/Scoring) │
│                                                ▲             │
│  Infrastructure ───────────────────────────────┘             │
│  (OpenAI, Anthropic SDK, Data I/O)                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## LLMART Directory Structure (Clean Architecture)

```
src/llmart/
├── models/              # Domain Layer (Entities + Value Objects)
│   ├── game.py          # Game entity, GameId, Genre value objects
│   ├── scoring.py       # SelectionScore, PhiML, PhiLLM
│   └── value.py         # NPV, VaR, InvestmentSignal
│
├── pipeline/            # Application Layer (Use Cases / Orchestration)
│   ├── graph.py         # LangGraph StateGraph definition
│   ├── state.py         # LLMARTState TypedDict
│   └── nodes/           # Each node = one Use Case
│       ├── prefilter.py
│       ├── ml_scoring.py
│       ├── llm_judge.py
│       ├── enrichment.py
│       ├── human_review.py
│       └── value.py
│
├── prompts/             # Infrastructure Layer (LLM Adapters)
│   ├── rubric_5dim.py   # GPT-5.2 Primary prompt
│   └── calibrator.py    # Opus-4.5 Calibrator prompt
│
├── data/                # Infrastructure Layer (Data I/O)
│   ├── sample_games.json
│   └── genre_params.json
│
└── cli.py               # Presentation Layer (Typer CLI)
```

## Core Patterns

### 1. Domain Model (Pure Python, No Dependencies)

```python
# models/game.py
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Genre:
    """Value Object: genre with Q1/Y1 ratio."""
    name: str
    q1_y1_ratio: float  # r(genre)

    @classmethod
    def front(cls) -> "Genre":
        return cls("front", 0.55)

    @classmethod
    def balanced(cls) -> "Genre":
        return cls("balanced", 0.45)

    @classmethod
    def live(cls) -> "Genre":
        return cls("live", 0.35)

@dataclass
class Game:
    """Entity: game with identity."""
    game_id: str
    title: str
    genre: Genre
    features: dict[str, float]

    def has_minimum_signals(self) -> bool:
        """Business rule: enough data for evaluation."""
        return bool(self.features.get("followers", 0) > 0)
```

### 2. Port/Adapter for LLM Clients

```python
# LLMART에서는 Protocol 기반 Port 사용
from typing import Protocol

class LLMEvaluatorPort(Protocol):
    """Port: LLM 평가 인터페이스."""
    async def evaluate(
        self,
        game_data: dict,
        rubric: str,
    ) -> dict:
        """5-Dim rubric evaluation 수행."""
        ...

class GPTPrimaryEvaluator:
    """Adapter: OpenAI GPT implementation."""
    def __init__(self, client, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    async def evaluate(self, game_data: dict, rubric: str) -> dict:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": rubric}, ...],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

class MockEvaluator:
    """Test Adapter: deterministic mock."""
    async def evaluate(self, game_data: dict, rubric: str) -> dict:
        return {"gameplay": "H", "innovation": "M", ...}
```

### 3. Pipeline Node as Use Case

```python
# pipeline/nodes/ml_scoring.py
async def ml_scoring_node(state: LLMARTState) -> dict:
    """Use Case: T1 ML Scoring.

    Input: state["prefiltered"] (5K games)
    Output: state["ml_ranked"] (Top 500)

    Pure business logic, no infrastructure dependencies.
    """
    scored = [
        {"game": g, "ml_score": ml_score(g)}
        for g in state["prefiltered"]
    ]
    scored.sort(key=lambda x: x["ml_score"], reverse=True)
    return {
        "ml_scores": {s["game"]["game_id"]: s["ml_score"] for s in scored},
        "ml_ranked": scored[:500],
    }
```

### 4. Dependency Injection via Graph Factory

```python
# pipeline/graph.py
def create_llmart_graph(
    primary_evaluator: LLMEvaluatorPort | None = None,
    calibrator: LLMEvaluatorPort | None = None,
) -> StateGraph:
    """Factory: Port 기반 DI로 테스트 용이성 확보."""
    graph = StateGraph(LLMARTState)

    graph.add_node("prefilter", prefilter_node)
    graph.add_node("ml_scoring", ml_scoring_node)

    if primary_evaluator:
        graph.add_node("llm_judge", create_llm_judge_node(
            primary=primary_evaluator,
            calibrator=calibrator,
        ))

    # ... edges
    return graph.compile()
```

## Key Principles for LLMART

1. **Domain Layer는 순수 Python**: `models/` 디렉토리는 외부 의존성 없음
2. **Node = Use Case**: 각 파이프라인 노드는 하나의 비즈니스 로직 단위
3. **Protocol 기반 Port**: ABC 대신 `Protocol` (structural typing)
4. **Factory Pattern**: Graph 생성 시 DI로 테스트/프로덕션 전환
5. **Value Object 활용**: Genre, InvestmentSignal 등은 frozen dataclass

## Anti-Patterns to Avoid

- Domain model에 OpenAI/Anthropic SDK 직접 임포트
- Pipeline node에서 직접 API 호출 (Port 통해야 함)
- God State: LLMARTState에 모든 것을 flat하게 넣기 (논리적 그룹화 필요)
- Test에서 실제 API 호출 (항상 Mock Adapter 사용)

## T3 Decision Matrix (2D Heatmap)

Axes: ML Score Percentile x Jury Score

| | Top 30% (ML) | Top 20% (ML) |
|------|-------------|-------------|
| >= H (3.0) | APPROVE | APPROVE |
| >= M (2.5) | REVIEW | APPROVE |
| >= L (1.5) | REJECT | REVIEW |

## Escalation Triggers (6 Types)

| Trigger | Condition | Action |
|---------|-----------|--------|
| DISAGREE | ML vs LLM gap > 1.5 | Human review |
| CONTRADICTION | P75/P25 > 3x | Risk flag |
| BOUNDARY | Score near threshold (0.48-0.52) | Dual review |
| MULTI_DIM | 3+ dims flagged | Expert panel |
| EXTREME_DIM | Any dim E(4) or L(1) outlier | Verify |
| LOW_COVERAGE | Missing data > 40% | Data collection |

## Reference

- Based on [wshobson/agents architecture-patterns](https://github.com/wshobson/agents)
- LLMART SOT: `ppt-workspace/task1/pipeline-v8-spec.md`
- Concepts: D3-PipelineArchitecture, D4-KeyDesignChoices
