---
name: prompt-engineering-patterns
description: LLMART 프롬프트 엔지니어링 패턴. 5-Dim Rubric, Calibrator prompt, Structured JSON output, pass^3 bias mitigation. "prompt", "rubric", "structured output", "json mode", "few-shot" 키워드로 트리거.
---

# Prompt Engineering Patterns for LLMART

## LLMART Prompt Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    T2 LLM-as-Judge Prompts                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GPT-5.2 Primary Evaluator                                  │
│  ├─ System: Role + 5-Dim Rubric + JSON schema               │
│  ├─ User: Game data (BLIND — no T1 score)                   │
│  └─ Output: {dims, jury_score, cot}                         │
│                                                              │
│  Opus-4.5 Calibrator (on disagreement)                      │
│  ├─ System: Calibration role + bias checklist                │
│  ├─ User: Primary results (k=3) + game data                 │
│  └─ Output: {action, final_scores, justification}           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Pattern 1: 5-Dim Value Lens Rubric (Structured Output)

```python
RUBRIC_5DIM_SYSTEM = """You are a game investment evaluator for a major publisher.
Evaluate the game on 5 dimensions using ONLY these categories: L (Low), M (Medium), H (High), E (Excellent).

## Rubric
- Gameplay (30%): Core loop quality, fun factor, replayability
- Innovation (20%): Novelty, differentiation from existing titles (MV-UP catch)
- Monetize (20%): Revenue model viability, player lifetime value potential
- Polish (15%): Technical quality, UX, visual/audio production
- Narrative (15%): Story/world appeal, target audience fit

## Scoring Guide
- L (Low): Below market standard, significant concerns
- M (Medium): Market average, no standout qualities
- H (High): Above average, clear strengths
- E (Excellent): Top tier, exceptional quality

## Output Format (JSON)
{
  "gameplay": "L|M|H|E",
  "innovation": "L|M|H|E",
  "monetize": "L|M|H|E",
  "polish": "L|M|H|E",
  "narrative": "L|M|H|E",
  "jury_score": <float>,
  "cot": "<1-2 sentence justification per dimension>"
}

## Rules
- Be objective. Do NOT anchor to any prior scores.
- Evaluate based solely on provided data.
- jury_score = 0.30×gameplay + 0.20×innovation + 0.20×monetize + 0.15×polish + 0.15×narrative
  (where L=1, M=2, H=3, E=4)"""
```

## Pattern 2: Dimension Shuffle (Bias Mitigation)

```python
import random

DIMENSIONS = ["gameplay", "innovation", "monetize", "polish", "narrative"]

def build_shuffled_prompt(game_data: dict, seed: int | None = None) -> str:
    """pass^3에서 각 호출마다 차원 순서를 셔플.

    Ordering bias 방지: 첫 번째 차원에 앵커링되는 현상 완화.
    """
    if seed is not None:
        random.seed(seed)

    shuffled = random.sample(DIMENSIONS, len(DIMENSIONS))

    dim_descriptions = {
        "gameplay": "Gameplay (30%): Core loop quality, fun factor, replayability",
        "innovation": "Innovation (20%): Novelty, differentiation (MV-UP)",
        "monetize": "Monetize (20%): Revenue model, LTV potential",
        "polish": "Polish (15%): Technical quality, UX",
        "narrative": "Narrative (15%): Story appeal, target fit",
    }

    rubric_text = "\n".join(f"- {dim_descriptions[d]}" for d in shuffled)

    return f"""Evaluate this game:

{json.dumps(game_data, indent=2)}

## Evaluation Dimensions (evaluate in this order):
{rubric_text}

Rate each: L (Low=1), M (Medium=2), H (High=3), E (Excellent=4)
Return JSON with all dimensions and jury_score."""
```

## Pattern 3: Calibrator Prompt (Meta-Evaluation)

```python
CALIBRATOR_SYSTEM = """You are a calibration reviewer for a game evaluation pipeline.
The primary evaluator produced inconsistent results across {k} evaluations.

## Your Task
1. Review the inconsistencies across evaluations
2. Check for: hallucination, anchoring bias, category confusion, ordering effects
3. Determine action:
   - CONFIRM: Primary evaluations are reasonable despite variation
   - ADJUST: Modify scores with justification
   - FLAG: Needs human review (high uncertainty)

## Output Format (JSON)
{
  "action": "CONFIRM|ADJUST|FLAG",
  "final_scores": {
    "gameplay": "L|M|H|E",
    "innovation": "L|M|H|E",
    "monetize": "L|M|H|E",
    "polish": "L|M|H|E",
    "narrative": "L|M|H|E"
  },
  "jury_score": <float>,
  "justification": "<explain each dimension decision>",
  "risk_flags": ["flag1", "flag2"]
}"""

def build_calibrator_prompt(
    game_data: dict,
    primary_results: list[dict],
    k: int = 3,
) -> list[dict]:
    """Opus-4.5 Calibrator에게 보낼 메시지 구성."""
    return [
        {"role": "system", "content": CALIBRATOR_SYSTEM.format(k=k)},
        {"role": "user", "content": f"""## Game Data
{json.dumps(game_data, indent=2)}

## Primary Evaluator Results ({k} runs)
{json.dumps(primary_results, indent=2)}

Analyze the inconsistencies and provide your calibrated assessment."""},
    ]
```

## Pattern 4: Structured Output Enforcement

```python
from pydantic import BaseModel, Field
from typing import Literal

class DimEvaluation(BaseModel):
    """Pydantic schema for type-safe LLM output parsing."""
    gameplay: Literal["L", "M", "H", "E"]
    innovation: Literal["L", "M", "H", "E"]
    monetize: Literal["L", "M", "H", "E"]
    polish: Literal["L", "M", "H", "E"]
    narrative: Literal["L", "M", "H", "E"]
    jury_score: float = Field(ge=1.0, le=4.0)
    cot: str

class CalibratorOutput(BaseModel):
    action: Literal["CONFIRM", "ADJUST", "FLAG"]
    final_scores: DimEvaluation
    justification: str
    risk_flags: list[str] = []

# OpenAI JSON mode
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={"type": "json_object"},
)
result = DimEvaluation.model_validate_json(response.choices[0].message.content)
```

## Pattern 5: Blind Evaluation (No Score Leakage)

```python
def prepare_game_context(game: dict, include_ml_score: bool = False) -> dict:
    """T2 LLM에게 전달할 게임 데이터.

    CRITICAL: include_ml_score=False (default)
    T1 ML score를 T2에 전달하면 anchoring bias 발생.
    """
    context = {
        "title": game["title"],
        "genre": game["genre"],
        "description": game.get("description", ""),
        "developer": game.get("developer", ""),
        "features": {
            "followers": game.get("followers"),
            "wishlists": game.get("wishlists"),
            "media_coverage": game.get("media_mentions"),
        },
    }

    if include_ml_score:
        # Only for T3 Human Review, never for T2
        context["ml_percentile"] = game.get("ml_percentile")

    return context
```

## Pattern 6: Error Recovery

```python
import json
from pydantic import ValidationError

async def safe_evaluate(
    client,
    messages: list[dict],
    schema: type[BaseModel],
    max_retries: int = 2,
) -> BaseModel | None:
    """LLM 호출 + JSON 파싱 + 검증 with retry."""
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return schema.model_validate_json(content)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries:
                return None
            # Add error context for retry
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"JSON parsing failed: {e}. Please fix and return valid JSON."})
```

## LLMART Prompt Checklist

- [ ] 5-Dim weights sum to 1.0 (0.30+0.20+0.20+0.15+0.15)
- [ ] 4-cat only (L/M/H/E) — never 1-10 scale
- [ ] Dimension shuffle per pass^3 iteration
- [ ] T1 score NOT included in T2 context (blind)
- [ ] JSON mode enabled for structured output
- [ ] Pydantic validation on every LLM response
- [ ] Calibrator only triggered on disagreement

## Reference

- Based on [wshobson/agents prompt-engineering-patterns](https://github.com/wshobson/agents)
- LLMART SOT: B1-LLM-as-Judge-Pipeline, B2-5Dim-ValueLens-Rubric
