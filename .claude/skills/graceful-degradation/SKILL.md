---
name: graceful-degradation
description: API 키 부재 시 자동 감지/폴백/안내 패턴. ReadinessReport로 사전 탐지, 단계적 축소(full → partial → mock), 사용자 가이드 생성. OpenClaw Auth Profile + Geode force_dry_run 패턴 증류. "readiness", "api key", "fallback", "degradation", "mock mode", "dry run" 키워드로 트리거.
---

# Graceful Degradation for LLM Pipelines

> OpenClaw Auth Profile Detection + Geode ReadinessReport/force_dry_run 패턴을 LLM 파이프라인에 적용.

## Core Concept

LLM 파이프라인은 외부 API 키에 의존한다. 키가 없거나 일부만 있을 때
**크래시 대신 단계적 축소 + 명확한 사용자 안내**를 제공해야 한다.

```
사용자 요청(mode=real) → ReadinessReport 생성 → 키 탐지
  ├─ 모든 키 있음 → real mode (full capability)
  ├─ 일부 키 있음 → partial real mode (degraded)
  └─ 키 없음     → mock mode (force_dry_run)
```

## Pattern 1: Pre-flight Readiness Check (Geode)

파이프라인 실행 **전**에 환경을 탐지하고 보고서를 생성한다.

```python
from dataclasses import dataclass, field
from enum import StrEnum
import os

class Capability(StrEnum):
    """파이프라인이 수행할 수 있는 개별 능력."""
    OPENAI_PRIMARY = "openai_primary"
    ANTHROPIC_CALIBRATOR = "anthropic_calibrator"
    MOCK_MODE = "mock_mode"

@dataclass
class ReadinessReport:
    """Geode ReadinessReport 패턴.

    환경 탐지 → 가능한 모드 결정 → 사용자 안내 생성.
    """
    has_openai_key: bool = False
    has_anthropic_key: bool = False
    capabilities: list[Capability] = field(default_factory=list)
    force_mock: bool = False
    guidance: list[str] = field(default_factory=list)

    @property
    def can_real_mode(self) -> bool:
        return self.has_openai_key and self.has_anthropic_key

    @property
    def can_partial_real(self) -> bool:
        return self.has_openai_key and not self.has_anthropic_key

    @property
    def available_mode(self) -> str:
        if self.can_real_mode: return "real"
        if self.can_partial_real: return "real_no_calibrator"
        return "mock"
```

### Key Design Decisions

1. **Placeholder 감지**: `sk-placeholder-xxx` 같은 더미 키를 real key로 취급하지 않음
2. **Capability 열거**: mock_mode는 항상 가능 → 최소 1개 capability 보장
3. **guidance 문자열**: 사용자에게 정확히 어떤 환경변수를 설정해야 하는지 알려줌

## Pattern 2: Auto-Fallback in Pipeline Nodes (OpenClaw)

파이프라인 노드 내부에서 요청된 모드를 실제 가능한 모드로 변환한다.

```python
def _resolve_mode(requested_mode: str) -> tuple[str, list[str]]:
    """OpenClaw Auth Profile Detection → LLMART mode resolution.

    Returns:
        (effective_mode, degradation_messages)
    """
    report = check_readiness(requested_mode)
    messages = list(report.guidance)

    if requested_mode != "real":
        return "mock", []

    return report.available_mode, messages
```

### 3-Tier Execution

| 모드 | 조건 | 동작 |
|---|---|---|
| `real` | OpenAI + Anthropic 키 모두 있음 | GPT Primary (pass^3) + Opus Calibrator |
| `real_no_calibrator` | OpenAI 키만 있음 | GPT Primary (pass^3), delta_cal=0.0 |
| `mock` | 키 없음 또는 mode=mock | 결정론적 시뮬레이션 |

## Pattern 3: User Guidance Display (Geode + OpenClaw)

CLI에서 파이프라인 실행 전에 readiness 상태를 표시한다.

```python
# CLI에서 readiness 출력
if mode == "real":
    readiness = check_readiness(mode)
    if readiness.guidance:
        console.print(Panel.fit(
            "\n".join(readiness.guidance),
            title="[bold yellow]Readiness Check[/]",
        ))
    if readiness.force_mock:
        mode = "mock"
        console.print("[yellow]Falling back to mock mode.[/]")
```

### Guidance 메시지 예시

**키 없음:**
```
┌─── Readiness Check ───┐
│ No API keys detected. Falling back to mock mode.
│ To enable real mode, set environment variables:
│   export OPENAI_API_KEY='sk-...'       # GPT-4o Primary (T2 pass^3)
│   export ANTHROPIC_API_KEY='sk-ant-...' # Opus Calibrator (disagreement)
└────────────────────────┘
```

**Anthropic 키만 없음:**
```
┌─── Readiness Check ───┐
│ ANTHROPIC_API_KEY missing. Opus Calibrator will be skipped.
│   export ANTHROPIC_API_KEY='sk-ant-...'
│ Running in partial real mode: GPT-4o Primary only, no calibration.
└────────────────────────┘
```

## Pattern 4: Error Channel Integration

readiness 체크 결과를 파이프라인 에러 채널에도 전파한다.
사용자가 결과를 볼 때 왜 mock 모드로 실행되었는지 알 수 있다.

```python
def llm_judge_node(state):
    effective_mode, degradation_msgs = _resolve_mode(mode)
    if degradation_msgs:
        errors.extend(degradation_msgs)  # 에러 채널로 전파
```

## Checklist

- [ ] `check_readiness()` 함수: 환경 탐지 + 보고서 생성
- [ ] Placeholder 키 필터링 (`sk-placeholder-*`, `sk-ant-placeholder-*`)
- [ ] 3-tier 모드 resolution (real → real_no_calibrator → mock)
- [ ] CLI에서 guidance Panel 출력
- [ ] 파이프라인 에러 채널에 degradation 사유 전파
- [ ] 테스트: 키 있음/없음/일부 시나리오 커버

## Reference

- **Geode**: ReadinessReport, force_dry_run, 4-stage LLM failover
- **OpenClaw**: Auth Profile Rotation, Gateway Startup capability enumeration
- **LLMART 구현**: `src/llmart/readiness.py`, `pipeline/nodes/llm_judge.py`
