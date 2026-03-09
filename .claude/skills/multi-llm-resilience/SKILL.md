---
name: multi-llm-resilience
description: LLMART Multi-LLM 장애 복원력 가이드. GPT Primary + Opus Calibrator 이중 구조에서 Auth rotation, model fallback, rate limiting, 동시성 제어. OpenClaw Pi Agent failover 패턴 증류. "failover", "fallback", "rate limit", "retry", "resilience", "multi-llm" 키워드로 트리거.
---

# Multi-LLM Resilience for LLMART

> OpenClaw Pi Agent의 Failover 전략을 LLMART T2 LLM-as-Judge에 적용.

## LLMART Multi-LLM Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    T2 LLM-as-Judge Resilience                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Game Data → GPT Primary (k=3, pass^3)                      │
│               │                                              │
│               ├─ Success (all agree) → Direct Score          │
│               │                                              │
│               ├─ Disagree → Opus Calibrator                  │
│               │              │                               │
│               │              ├─ CONFIRM / ADJUST → Score     │
│               │              └─ FLAG → Human Review          │
│               │                                              │
│               └─ API Failure → Failover Chain                │
│                    ├─ L1: Retry with backoff                 │
│                    ├─ L2: Auth profile rotation               │
│                    ├─ L3: Model fallback                     │
│                    └─ L4: Graceful degradation               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Pattern 1: Auth Profile Rotation

OpenClaw는 rate limit 시 다음 API 프로필로 자동 전환.
LLMART에서는 OpenAI/Anthropic 키를 복수 보유 시 적용.

```python
from dataclasses import dataclass, field
import time

@dataclass
class AuthProfile:
    name: str
    api_key: str
    provider: str  # "openai" | "anthropic"
    rate_limited_until: float = 0.0
    error_count: int = 0

class AuthProfileRotator:
    """OpenClaw Auth Profile Rotation 패턴.

    Rate limit 또는 인증 실패 시 다음 프로필로 자동 전환.
    """

    def __init__(self, profiles: list[AuthProfile]) -> None:
        self._profiles = profiles
        self._current_idx = 0

    def get_current(self) -> AuthProfile | None:
        """현재 사용 가능한 프로필 반환."""
        now = time.time()
        for _ in range(len(self._profiles)):
            profile = self._profiles[self._current_idx]
            if profile.rate_limited_until <= now:
                return profile
            self._rotate()
        return None  # 모든 프로필 rate limited

    def mark_rate_limited(self, duration_seconds: float = 60.0) -> None:
        """현재 프로필을 rate limited로 마킹하고 다음으로 전환."""
        profile = self._profiles[self._current_idx]
        profile.rate_limited_until = time.time() + duration_seconds
        profile.error_count += 1
        self._rotate()

    def _rotate(self) -> None:
        self._current_idx = (self._current_idx + 1) % len(self._profiles)
```

## Pattern 2: Model Fallback Chain

OpenClaw의 Thinking Level Fallback을 모델 레벨로 적용.

```python
from typing import Any

# LLMART Model Fallback Chain
MODEL_FALLBACK = {
    "primary": [
        {"provider": "openai", "model": "gpt-4o", "tier": "primary"},
        {"provider": "openai", "model": "gpt-4o-mini", "tier": "fallback"},
    ],
    "calibrator": [
        {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "tier": "primary"},
        {"provider": "openai", "model": "gpt-4o", "tier": "fallback"},
    ],
}

async def call_with_fallback(
    role: str,  # "primary" | "calibrator"
    messages: list[dict],
    clients: dict[str, Any],
    max_retries: int = 2,
) -> dict:
    """Model fallback chain with retry.

    L1: Retry same model (with exponential backoff)
    L2: Fallback to next model in chain
    """
    chain = MODEL_FALLBACK[role]

    for model_spec in chain:
        provider = model_spec["provider"]
        model = model_spec["model"]
        client = clients[provider]

        for attempt in range(max_retries + 1):
            try:
                if provider == "openai":
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                    return {"content": response.choices[0].message.content, "model": model}
                else:  # anthropic
                    response = await client.messages.create(
                        model=model,
                        messages=messages,
                        max_tokens=1024,
                    )
                    return {"content": response.content[0].text, "model": model}

            except Exception as e:
                if _is_rate_limit(e):
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                if _is_auth_error(e):
                    break  # Skip to next model
                if attempt == max_retries:
                    break  # Try next model
                await asyncio.sleep(1)

    raise RuntimeError(f"All models in {role} chain exhausted")

def _is_rate_limit(e: Exception) -> bool:
    return "rate_limit" in str(e).lower() or "429" in str(e)

def _is_auth_error(e: Exception) -> bool:
    return "auth" in str(e).lower() or "401" in str(e)
```

## Pattern 3: Concurrency Control (Lane Queue)

OpenClaw Lane Queue를 asyncio Semaphore로 변환.

```python
import asyncio
from contextlib import asynccontextmanager

class LLMRateLimiter:
    """OpenClaw Lane Queue 패턴 — LLM API 동시성 제어.

    - session_semaphore: 같은 게임의 pass^3 호출은 순서 보장
    - global_semaphore: 전체 동시 API 호출 수 제한
    """

    def __init__(
        self,
        max_concurrent_per_game: int = 3,   # pass^3 k=3
        max_concurrent_global: int = 10,     # 전체 LLM 호출
    ) -> None:
        self._game_semaphores: dict[str, asyncio.Semaphore] = {}
        self._global = asyncio.Semaphore(max_concurrent_global)
        self._max_per_game = max_concurrent_per_game

    @asynccontextmanager
    async def acquire(self, game_id: str):
        """2-level semaphore: game-level + global-level."""
        if game_id not in self._game_semaphores:
            self._game_semaphores[game_id] = asyncio.Semaphore(self._max_per_game)

        async with self._game_semaphores[game_id]:
            async with self._global:
                yield

# Usage in T2 pipeline
rate_limiter = LLMRateLimiter(max_concurrent_per_game=3, max_concurrent_global=10)

async def evaluate_game_batch(games: list[dict]) -> list[dict]:
    """500 games batch evaluation with rate limiting."""
    tasks = []
    for game in games:
        task = evaluate_single_game(game, rate_limiter)
        tasks.append(task)
    return await asyncio.gather(*tasks)

async def evaluate_single_game(game: dict, limiter: LLMRateLimiter) -> dict:
    async with limiter.acquire(game["game_id"]):
        return await call_with_fallback("primary", ...)
```

## Pattern 4: Context Overflow → Auto-Compaction

OpenClaw의 Context Overflow Handling을 LLMART 배치 처리에 적용.

```python
def compact_game_context(game_data: dict, max_tokens: int = 2000) -> dict:
    """게임 데이터가 컨텍스트 윈도우를 초과하면 자동 압축.

    OpenClaw: Overflow 감지 → Auto-compaction → 압축 후 재시도
    LLMART: 게임 description이 길면 truncate + key features만 유지
    """
    import json
    serialized = json.dumps(game_data)

    if len(serialized) < max_tokens * 4:  # ~4 chars/token 추정
        return game_data

    # Compact: 필수 필드만 유지
    return {
        "game_id": game_data["game_id"],
        "title": game_data["title"],
        "genre": game_data["genre"],
        "description": game_data.get("description", "")[:500],  # Truncate
        "features": {
            k: v for k, v in game_data.get("features", {}).items()
            if k in FEATURE_WEIGHTS  # ML scoring에 사용되는 feature만
        },
    }
```

## Pattern 5: Graceful Degradation

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class EvaluationResult:
    game_id: str
    path: Literal["direct", "calibrated", "degraded", "failed"]
    scores: dict | None
    model_used: str
    retries: int
    degradation_reason: str | None = None

async def evaluate_with_resilience(
    game: dict,
    primary: LLMEvaluatorPort,
    calibrator: LLMEvaluatorPort | None,
) -> EvaluationResult:
    """Full resilience chain:
    1. pass^3 with primary → direct
    2. Calibrator on disagreement → calibrated
    3. Single evaluation (k=1) on partial failure → degraded
    4. Skip game on total failure → failed (logged for retry)
    """
    try:
        result = await evaluate_with_pass3(game, primary, calibrator, k=3)
        return EvaluationResult(
            game_id=game["game_id"],
            path=result["path"],
            scores=result["score"],
            model_used=result.get("model", "unknown"),
            retries=0,
        )
    except Exception:
        # Degraded: single evaluation
        try:
            single = await primary.evaluate(game, rubric="...")
            return EvaluationResult(
                game_id=game["game_id"],
                path="degraded",
                scores=single,
                model_used="primary-single",
                retries=1,
                degradation_reason="pass3_failed_single_eval_used",
            )
        except Exception as e:
            return EvaluationResult(
                game_id=game["game_id"],
                path="failed",
                scores=None,
                model_used="none",
                retries=2,
                degradation_reason=str(e),
            )
```

## LLMART Resilience Checklist

- [ ] Primary (GPT) + Calibrator (Opus) 각각 fallback model 지정
- [ ] API key rotation (복수 키 보유 시)
- [ ] Exponential backoff on rate limit (2^n seconds)
- [ ] Global concurrency limit (Semaphore)
- [ ] Context overflow auto-compaction
- [ ] Graceful degradation: pass^3 → single → skip
- [ ] Failed games 로깅 + retry queue
- [ ] **Pre-flight readiness check** (see `graceful-degradation` skill)
- [ ] **3-tier mode resolution**: real → real_no_calibrator → mock

## Related Skills

- **`graceful-degradation`**: API 키 부재 시 자동 감지/폴백/안내 패턴. ReadinessReport, force_dry_run, 3-tier mode resolution. `src/llmart/readiness.py` 참조.

## Reference

- **OpenClaw 출처**: Pi Agent failover (Auth Profile Rotation, Thinking Level Fallback, Context Overflow, Model Failover)
- **분석 리포트**: `ppt-workspace/task2/research/openclaw-analysis-report.md` Part 3.2
- **라우팅 비교**: `ppt-workspace/task2/research/openclaw-routing-analysis.md` §4
- **Geode 출처**: ReadinessReport, force_dry_run, 4-stage LLM failover
