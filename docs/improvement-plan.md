# LLMART Pipeline Improvement Plan

> Created: 2026-02-22
> Status: In Progress

## 문제 진단

### 실행 결과 (개선 전)

```
╭──────────── LLMART Pipeline ─────────────╮
│ Games: sample_games.json  (3 candidates) │
│ Mode:  mock                              │
│ Top-K: 10                                │
╰──────────────────────────────────────────╯

Pipeline complete — stage: value
                     Final Results (2 games)
┏━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
│ Rank │ Game         │ Genre    │  Score │ Signal │ Decision │     NPV_3Y │
┡━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 1    │ Stellar F.   │ Surv.    │ 0.9330 │  RED   │ APPROVE  │ $1,650,872 │
│ 2    │ Chrono Drift │ Roguelike│ 0.6852 │  RED   │ APPROVE  │   $446,643 │
└──────┴──────────────┴──────────┴────────┴────────┴──────────┴────────────┘
```

### 핵심 문제

| # | 문제 | 원인 | 영향 |
|---|---|---|---|
| 1 | Signal 전부 RED | Q1 Revenue 추정 공식이 비현실적 (`reviews × price × 0.15`) | 데모 임팩트 없음 |
| 2 | 1/3 게임 탈락 | 3개 배치에서 percentile=0.0 → 무조건 REJECT | 데모 다양성 부족 |
| 3 | 가상 데이터 | 합성 게임 → 면접관 신뢰도 저하 | 설명력 약함 |
| 4 | CLI 출력 빈약 | stage 진행, 상세정보, Value 컬럼 없음 | 파이프라인 동작 불투명 |

---

## 개선 사항

### P0 — 데모 크리티컬

#### P0-1: Q1 Revenue 추정 현실화 (Boxleiter Method)

- **파일**: `enrichment.py`
- **변경**: `reviews × price × 0.5 × 0.3` → `reviews × SALES_MULT × price × STEAM_CUT × Q1_RATIO`
- **근거**: VG Insights 연구 (11,445개 게임, R²=0.78)
  - Post-2020 게임 리뷰→판매 multiplier 중앙값: **30x**
  - Steam 수수료 공제: **0.70** (30% 수수료)
  - Q1 비율 (첫 분기 매출/총 매출): **0.50** (출시 초기 집중)
- **공식**: `P50 = reviews × 30 × price × 0.70 × 0.50`
- **출처**:
  - VG Insights: "How to Estimate Steam Video Game Sales"
  - Game Developer (Gamasutra): Boxleiter method (2021 update)
  - Game Oracle: revenue estimation methodology

#### P0-2: 실제 Steam 게임 데이터 수집

- **파일**: `sample_games.json`
- **변경**: 합성 3개 → 실제 Steam 게임 7개
- **데이터 출처**: Steam Store, SteamDB, VG Insights 교차 검증
- **게임 구성** (3-tier 시그널 다양성):
  - GREEN (2): Balatro (148K reviews), Lethal Company (272K reviews)
  - YELLOW (2-3): Manor Lords (37K), Brotato (110K), Halls of Torment (20K)
  - RED (1-2): Dome Keeper (9.8K), Patch Quest (1K)

#### P0-3: 소규모 배치 Percentile 보정 (Hazen Method)

- **파일**: `ml_scoring.py`
- **변경**: `rank / max(n-1, 1)` → `(rank + 1) / (n + 1)` (Hazen plotting position)
- **효과**: n=3일 때 {0.0, 0.5, 1.0} → {0.25, 0.50, 0.75} → 최하위도 REVIEW 가능

### P1 — CLI 사용성

#### P1-1: 노드별 진행 표시

- **파일**: `cli.py`
- **변경**: 각 노드 실행 전후로 stage 로그 출력 (Rich Status/Spinner)

#### P1-2: `--verbose` 상세 모드

- **파일**: `cli.py`
- **변경**: dim_scores, features, escalation 상세 테이블 출력

#### P1-3: Value 컬럼 추가

- **파일**: `cli.py`
- **변경**: NPV_3Y 옆에 Value(= NPV - costs) 컬럼 추가

### P2 — 코드 품질

#### P2-1: 노드별 에러 핸들링

- **파일**: 각 노드
- **변경**: try/except + errors 리스트 누적, 한 게임 실패해도 파이프라인 계속

#### P2-2: Pydantic V1 경고 억제

- **파일**: `cli.py`
- **변경**: warnings.filterwarnings로 LangChain V1 호환 경고 억제

---

## 검증 계획

1. `uv run pytest` — 전체 테스트 통과
2. `uv run ruff check src/ tests/` — lint clean
3. `uv run mypy src/` — typecheck clean
4. `uv run llmart run --mode mock --top-k 10` — 7개 게임, 3색 시그널 확인
5. Skills 스코어링 — 전 스킬 96점 이상 목표
