---
name: pipeline-automation
description: LLMART 파이프라인 자동화 가이드. Regime monitoring, quarterly feedback loop, batch scheduling, drift detection trigger. OpenClaw Cron/Heartbeat/Hooks 패턴 증류. "automation", "cron", "scheduling", "regime", "feedback loop", "drift", "monitoring" 키워드로 트리거.
---

# Pipeline Automation for LLMART

> OpenClaw 4-Layer Automation (Heartbeat/Cron/Hooks)을 LLMART Regime Monitor + Feedback Loop에 적용.

## LLMART Automation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                LLMART Automation (3-Layer)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  L1: Regime Monitor (Heartbeat 패턴)                        │
│  ├─ PSI drift check: daily                                  │
│  ├─ ECE calibration check: weekly                           │
│  ├─ rho correlation check: weekly                           │
│  └─ Active hours: 업무 외 시간 실행                         │
│                                                              │
│  L2: Quarterly Feedback Loop (Cron 패턴)                    │
│  ├─ Q1 label collection: T+90d (at 스케줄)                  │
│  ├─ Model retrain trigger: quarterly (cron 스케줄)          │
│  ├─ Weight recalibration: on Phase transition               │
│  └─ Atomic store + Run log                                  │
│                                                              │
│  L3: Event Hooks (Internal Hooks 패턴)                      │
│  ├─ on_psi_warning → trigger model review                   │
│  ├─ on_ece_exceeded → trigger recalibration                 │
│  ├─ on_phase_transition → update weights                    │
│  └─ on_quarter_complete → collect ground truth              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Pattern 1: Regime Monitor (← OpenClaw Heartbeat)

OpenClaw Heartbeat Runner의 주기적 폴링 패턴을 적용.

```python
import asyncio
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

@dataclass
class MonitorConfig:
    """Regime Monitor 설정 — OpenClaw heartbeat config 대응."""
    check_interval_seconds: int = 86400  # daily
    active_hours_start: time = time(2, 0)   # 02:00 (새벽 실행)
    active_hours_end: time = time(6, 0)     # 06:00
    timezone: str = "Asia/Seoul"
    enabled: bool = True

# Threshold 정의 — LLMART SOT D1-RegimeMonitor
REGIME_THRESHOLDS = {
    "psi": {"warning": 0.25, "action": "WARNING → model review"},
    "migration_rate": {"warning": 0.30, "action": "WARNING → tier recalibration"},
    "rho": {"warning": 0.50, "direction": "below", "action": "WARNING → feature review"},
    "ece": {"warning": 0.10, "direction": "above", "action": "TRIGGER → recalibration"},
}

class RegimeMonitor:
    """OpenClaw Heartbeat Runner 패턴 — LLMART Regime Monitor.

    주기적으로 모델 성능 지표를 체크하고 이상 시 이벤트 발행.
    """

    def __init__(
        self,
        config: MonitorConfig,
        event_handlers: dict[str, list] | None = None,
    ) -> None:
        self._config = config
        self._handlers = event_handlers or {}
        self._running = False

    def is_within_active_hours(self) -> bool:
        """OpenClaw isWithinActiveHours() 대응.

        자정 넘김(wrap-around) 지원.
        """
        tz = ZoneInfo(self._config.timezone)
        now = datetime.now(tz).time()
        start = self._config.active_hours_start
        end = self._config.active_hours_end

        if end > start:
            return start <= now < end
        # Wrap-around (e.g., 22:00 - 06:00)
        return now >= start or now < end

    async def run_once(self, current_metrics: dict[str, float]) -> list[dict]:
        """단일 체크 실행. 위반 시 이벤트 발행."""
        alerts = []

        for metric, threshold_config in REGIME_THRESHOLDS.items():
            value = current_metrics.get(metric)
            if value is None:
                continue

            direction = threshold_config.get("direction", "above")
            threshold = threshold_config["warning"]

            violated = (
                (direction == "above" and value > threshold) or
                (direction == "below" and value < threshold)
            )

            if violated:
                alert = {
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "action": threshold_config["action"],
                    "timestamp": datetime.now().isoformat(),
                }
                alerts.append(alert)
                await self._emit(f"regime_{metric}_violated", alert)

        return alerts

    async def _emit(self, event_name: str, payload: dict) -> None:
        """OpenClaw onAgentEvent 패턴 — 이벤트 핸들러 호출."""
        for handler in self._handlers.get(event_name, []):
            await handler(payload)
```

## Pattern 2: Quarterly Feedback Loop (← OpenClaw Cron)

OpenClaw의 3종 스케줄 (at/every/cron)을 LLMART 피드백 루프에 적용.

```python
from enum import Enum
from typing import Literal

class ScheduleKind(str, Enum):
    AT = "at"       # 1회성 (Q1 label 수집 T+90d)
    EVERY = "every"  # 고정 간격 (signal refresh)
    CRON = "cron"    # CRON 표현식 (quarterly retrain)

@dataclass
class FeedbackJob:
    """OpenClaw CronJob 패턴 — LLMART Feedback Loop Job.

    OpenClaw 원본: id, name, schedule, sessionTarget, payload, state
    """
    job_id: str
    name: str
    schedule_kind: ScheduleKind
    schedule_value: str  # "2026-04-01T00:00:00" | "86400" | "0 0 1 1,4,7,10 *"
    enabled: bool = True
    delete_after_run: bool = False  # at 스케줄 완료 후 삭제

    # State tracking — OpenClaw job.state 대응
    last_run_at: str | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_duration_ms: int | None = None
    next_run_at: str | None = None

# LLMART Feedback Loop Jobs
FEEDBACK_JOBS = [
    FeedbackJob(
        job_id="q1_label_collect",
        name="Q1 Ground Truth Label Collection",
        schedule_kind=ScheduleKind.AT,
        schedule_value="T+90d from selection date",
        delete_after_run=False,  # 매 분기 재생성
    ),
    FeedbackJob(
        job_id="quarterly_retrain",
        name="Quarterly Model Retrain Trigger",
        schedule_kind=ScheduleKind.CRON,
        schedule_value="0 0 1 1,4,7,10 *",  # 매 분기 1일
    ),
    FeedbackJob(
        job_id="daily_signal_refresh",
        name="Daily Signal Refresh (Steam API)",
        schedule_kind=ScheduleKind.EVERY,
        schedule_value="86400",  # 24h in seconds
    ),
    FeedbackJob(
        job_id="weekly_calibration",
        name="Weekly ECE Calibration Check",
        schedule_kind=ScheduleKind.CRON,
        schedule_value="0 8 * * MON",  # 매주 월요일 08:00
    ),
]
```

## Pattern 3: Event Hooks (← OpenClaw Internal Hooks)

OpenClaw의 이벤트 기반 Hook 시스템을 LLMART에 적용.

```python
from collections import defaultdict
from typing import Callable, Awaitable

EventHandler = Callable[[dict], Awaitable[None]]

class LLMARTEventBus:
    """OpenClaw Internal Hooks 패턴 — 이벤트 기반 자동화.

    OpenClaw 이벤트: command:new, agent:bootstrap, gateway:startup
    LLMART 이벤트: regime:psi_warning, phase:transition, quarter:complete
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def on(self, event: str, handler: EventHandler) -> None:
        """이벤트 핸들러 등록.

        OpenClaw 매칭 로직: 일반("regime") + 구체("regime:psi_warning")
        양쪽 모두 실행.
        """
        self._handlers[event].append(handler)

    async def emit(self, event: str, payload: dict) -> None:
        """이벤트 발행 — 일반 + 구체 핸들러 모두 호출."""
        # Specific handlers
        for handler in self._handlers.get(event, []):
            await handler(payload)

        # General handlers (e.g., "regime" catches "regime:psi_warning")
        general = event.split(":")[0]
        if general != event:
            for handler in self._handlers.get(general, []):
                await handler(payload)

# Usage
bus = LLMARTEventBus()

# Register hooks
async def on_psi_warning(payload: dict) -> None:
    """PSI > 0.25 → 모델 리뷰 트리거."""
    print(f"PSI WARNING: {payload['value']:.3f} > {payload['threshold']}")

async def on_ece_exceeded(payload: dict) -> None:
    """ECE > 0.10 → 재캘리브레이션."""
    print(f"ECE EXCEEDED: {payload['value']:.3f}")

async def on_phase_transition(payload: dict) -> None:
    """Phase 0 → Phase 1+ 전환 시 가중치 동적 조정."""
    print(f"Phase transition: n={payload['sample_count']}")

bus.on("regime:psi_warning", on_psi_warning)
bus.on("regime:ece_exceeded", on_ece_exceeded)
bus.on("phase:transition", on_phase_transition)
```

## Pattern 4: Atomic Store + Run Log (← OpenClaw Cron Store)

```python
import json
import os
import tempfile
from pathlib import Path

class AtomicStore:
    """OpenClaw saveCronStore 패턴 — tmp → rename → backup.

    서버 크래시 시에도 데이터 무결성 보장.
    """

    def __init__(self, store_path: Path) -> None:
        self._path = store_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict) -> None:
        """Atomic write: tmp file → rename → backup."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent,
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._path)  # Atomic rename

            # Best-effort backup
            backup = self._path.with_suffix(".json.bak")
            try:
                import shutil
                shutil.copy2(self._path, backup)
            except OSError:
                pass
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def load(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {}

class RunLog:
    """OpenClaw CronRunLog 패턴 — JSONL + auto-pruning.

    max_bytes=2MB, keep_lines=2000
    """

    def __init__(self, log_path: Path, max_bytes: int = 2_000_000) -> None:
        self._path = log_path
        self._max_bytes = max_bytes
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(self._path, "a") as f:
            f.write(line)
        self._prune_if_needed()

    def _prune_if_needed(self) -> None:
        if not self._path.exists():
            return
        if self._path.stat().st_size <= self._max_bytes:
            return
        # Keep last 2000 lines
        lines = self._path.read_text().splitlines()
        kept = lines[-2000:]
        self._path.write_text("\n".join(kept) + "\n")
```

## Pattern 5: Phase Transition Monitor (← OpenClaw Config Reload)

```python
class PhaseMonitor:
    """Phase 0 → Phase 1+ 자동 전환 감지.

    OpenClaw Config Hot Reload → LLMART weight transition.
    Phase 0: n < 50, w_ml=0.6, w_llm=0.4 (고정)
    Phase 1+: n >= 50, w_llm = σ(agree) × cov (동적)
    """

    def __init__(self, event_bus: LLMARTEventBus) -> None:
        self._bus = event_bus
        self._current_phase = 0
        self._sample_count = 0

    async def update(self, new_sample_count: int) -> dict:
        """샘플 수 업데이트 → Phase 전환 체크."""
        self._sample_count = new_sample_count

        if self._current_phase == 0 and new_sample_count >= 50:
            self._current_phase = 1
            payload = {
                "from_phase": 0,
                "to_phase": 1,
                "sample_count": new_sample_count,
                "weight_mode": "dynamic",
            }
            await self._bus.emit("phase:transition", payload)
            return payload

        return {"phase": self._current_phase, "sample_count": new_sample_count}

    def get_weights(self, agreement_rate: float = 0.0, coverage: float = 0.0) -> dict:
        """현재 Phase에 맞는 가중치 반환."""
        if self._current_phase == 0:
            return {"w_ml": 0.6, "w_llm": 0.4, "phase": 0}

        # Phase 1+: 동적 가중치
        import math
        sigma = 1 / (1 + math.exp(-10 * (agreement_rate - 0.5)))
        w_llm = sigma * coverage
        w_ml = 1.0 - w_llm
        return {"w_ml": w_ml, "w_llm": w_llm, "phase": self._current_phase}
```

## Pattern 6: Stuck Job Detection (← OpenClaw)

```python
import time

STUCK_THRESHOLD_SECONDS = 7200  # 2 hours (OpenClaw: STUCK_RUN_MS = 2h)

def detect_stuck_jobs(jobs: list[FeedbackJob]) -> list[FeedbackJob]:
    """OpenClaw stuck job 탐지 — 2시간 이상 실행 중인 job 해제."""
    stuck = []
    now = time.time()
    for job in jobs:
        if (
            job.last_status == "running"
            and job.last_run_at
            and (now - float(job.last_run_at)) > STUCK_THRESHOLD_SECONDS
        ):
            job.last_status = "skipped"  # 강제 해제
            stuck.append(job)
    return stuck
```

## LLMART Automation Schedule Summary

| Job | Schedule | OpenClaw 패턴 | Trigger |
|-----|----------|--------------|---------|
| PSI drift check | daily 02:00 | Heartbeat (every) | Regime Monitor |
| ECE calibration | weekly MON 08:00 | Cron (cron expr) | Regime Monitor |
| Signal refresh | daily | Cron (every) | Automated |
| Q1 label collect | T+90d | Cron (at) | Quarter end |
| Model retrain | quarterly | Cron (cron) | Feedback loop |
| Phase transition | on n>=50 | Config reload | Event hook |

## Phase Transition Criteria

| Transition | Condition | Weight Mode |
|-----------|-----------|-------------|
| 0 → 1 | n >= 15 samples | Sigmoid + EMA |
| 1 → 2 | n >= 50, Spearman rho >= 0.30 | Ridge LOOCV |
| 2 → 3 | n >= 200, rho >= 0.50, ECE < 0.10 | Production |

## Quarterly Feedback Loop Steps

1. **T3 선정** — 30개/분기 선별 완료
2. **실제 매출 수집** — Q1 Revenue (T+90d, Steam/SteamSpy/VGInsights 교차검증)
3. **FACT_EVALS** — AI 기록 (ground truth label, tier assignment)
4. **모델 재학습** — Ridge weight update, Phase transition eval, PSI drift check

## CUSUM Drift Detection

Cumulative Sum of standardized residuals. Threshold h=4.0.
Alert when CUSUM exceeds h, indicating sustained drift in model predictions.

```python
def cusum(residuals: list[float], h: float = 4.0) -> bool:
    s_pos = s_neg = 0.0
    for r in residuals:
        s_pos = max(0, s_pos + r)
        s_neg = min(0, s_neg + r)
        if s_pos > h or abs(s_neg) > h:
            return True  # drift detected
    return False
```

## Reference

- **OpenClaw 출처**: Heartbeat Runner, Cron Service (3종 스케줄), Internal Hooks, Config Hot Reload
- **분석 리포트**: `ppt-workspace/task2/research/openclaw-automation-analysis.md`
- **LLMART SOT**: D1-RegimeMonitor, D2-QuarterlyFeedbackLoop, C1-SelectionScore (Phase transition)
