# LLMART — Game Selection & Value Inference Pipeline

LangGraph 기반 게임 퍼블리싱 의사결정 파이프라인.
600K+ Steam 타이틀에서 통계 ML 랭킹(T1) → LLM-as-Judge 평가(T2) → 휴먼 리뷰(T3)를 거쳐 최종 퍼블리싱 후보를 선별하고, NPV 기반 가치를 추론합니다.

## Features

- **3-Tier Funnel** — Prefilter(600K→5K) → ML Scoring(T1) → LLM Judge(T2) → Human Review(T3)
- **5-Dim Rubric** — Gameplay(30%) · Innovation(20%) · Monetization(20%) · Polish(15%) · Narrative(15%)
- **Dual-Judge Consensus** — GPT-5.2 + Claude Opus 4.6 트리플 패스, confidence-weighted jury
- **4-Phase Controller** — Cold Start(Phase 0) → Adaptive(1) → Validated(2) → Production(3)
- **Ridge Weight Learning** — Nested LOOCV, EMA-smoothed 적응형 가중치
- **Value Inference** — NPV 3-Year, Quantile Regression(P25/P50), Signal(GREEN/YELLOW/RED)
- **Regime Monitoring** — PSI 드리프트, ECE 캘리브레이션, Loop1/Loop2 알림
- **Streamlit Dashboard** — Pipeline · Games · Monitoring · Retrain 4-page UI
- **Graceful Degradation** — API 키 없이도 mock 모드로 전체 파이프라인 실행 가능
- **462 Tests** — pytest + ruff + mypy strict + bandit 전체 통과

## Installation

```bash
uv sync
```

## Quick Start

```bash
# 파이프라인 실행 (mock 모드, API 키 불필요)
uv run llmart run

# Top-10 선별, 상세 출력
uv run llmart run --top-k 10 --verbose

# Streamlit 대시보드
uv run llmart-ui

# 검증
uv run llmart validate

# 분기별 재학습
uv run llmart retrain
```

## Setup

```bash
# 1. API 키 설정 (선택)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Real 모드 실행 (Dual-Judge)
uv run llmart run --mode real --verbose

# API 키 없으면 자동으로 mock 모드로 안내됩니다
```

## Usage

### `llmart run` — 파이프라인 실행

```bash
llmart run [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--games` | `-g` | bundled | Games JSON 파일 |
| `--mode` | `-m` | mock | LLM 모드: mock / real |
| `--top-k` | `-k` | 30 | 최종 선별 수 |
| `--verbose` | `-v` | False | 5-dim 점수 + KPI 카드 출력 |
| `--checkpoint` | | False | SQLite 체크포인트 |
| `--resume` | | None | 이전 Thread ID에서 재개 |
| `--monitor` | | False | 모니터링 노드 활성화 |
| `--phase` | `-p` | 0 | Phase 오버라이드 (0-3) |
| `--training-data` | `-t` | None | Ridge 가중치용 학습 데이터 |

### `llmart validate` — 부트스트랩 검증

```bash
llmart validate --data training.json --top-k 30
```

### `llmart retrain` — 분기별 재학습

```bash
llmart retrain --n-games 500 --seed 42
```

합성 데이터 생성 → Ridge LOOCV → Phase 전이 → PSI 드리프트 → 6-metric 검증

### `llmart-ui` — Streamlit 대시보드

```bash
llmart-ui
```

4-page 대시보드: Pipeline(퍼널/타이밍) · Games(인터랙티브 테이블) · Monitoring(헬스 게이지) · Retrain(재학습 제어)

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  CLI / UI               (cli.py, ui/, readiness, audit)      │
├──────────────────────────────────────────────────────────────┤
│  Pipeline               (graph.py, nodes/, state.py)         │
├──────────────────────────────────────────────────────────────┤
│  Models                 (game, scoring, value, jury, escal.) │
├──────────────────────────────────────────────────────────────┤
│  Phase Controller       (controller, weight_learner, ridge)  │
├──────────────────────────────────────────────────────────────┤
│  Monitoring             (metrics, PSI, ECE, regime)          │
├──────────────────────────────────────────────────────────────┤
│  Data / Prompts         (synthetic, rubric_5dim, calibrator) │
└──────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
START → Prefilter (L1: hard threshold, L2: soft scoring)
      → ML Scoring (T1: LambdaMART-like, 7 features, NDCG@k)
      → LLM Judge (T2: 5-dim rubric, triple-pass, jury consensus)
      → Enrichment (feature enrichment)
      → Human Review (T3: manual review queue)
      → Value Inference (NPV 3Y, quantile, signal)
      → Monitoring (PSI, ECE, regime alerts)
      → END
```

### Phase System

| Phase | Condition | Weights | Description |
|-------|-----------|---------|-------------|
| 0 | Cold start | w_ml=0.6, w_llm=0.4 | 고정 가중치 (PDF spec) |
| 1 | n ≥ 15 | Adaptive (EMA) | 적응형 학습 시작 |
| 2 | n ≥ 50, ρ ≥ 0.35 | Ridge LOOCV | 검증된 적응형 |
| 3 | n ≥ 200, ρ ≥ 0.50 | Production | 프로덕션 |

## Project Structure

```
src/llmart/
├── cli.py                  # Typer CLI (run, validate, retrain)
├── config.py               # LLMARTConfig (immutable dataclass)
├── readiness.py            # API 키 감지 + 모드 결정
├── resilience.py           # Retry 데코레이터
├── audit.py                # RunRecord 추적
├── models/
│   ├── game.py             # Game 도메인 모델
│   ├── scoring.py          # SelectionScore (S = w_ml·Φ_ml + w_llm·Φ_llm + δ)
│   ├── value.py            # ValueInference + Signal (GREEN/YELLOW/RED)
│   ├── jury.py             # Confidence-weighted 합의
│   ├── escalation.py       # 6-reason 에스컬레이션
│   ├── hit_classifier.py   # Cost-sensitive 티어 예측
│   ├── percentile.py       # Percentile 랭킹
│   └── quantile.py         # Q1 수익 예측 (P25/P50)
├── pipeline/
│   ├── graph.py            # LangGraph StateGraph
│   ├── state.py            # GraphState (TypedDict + reducers)
│   └── nodes/
│       ├── prefilter.py    # L1/L2 필터 (600K→5K)
│       ├── ml_scoring.py   # T1: LambdaMART-like 랭킹
│       ├── llm_judge.py    # T2: 5-dim 루브릭 평가
│       ├── enrichment.py   # Feature enrichment
│       ├── human_review.py # T3: 휴먼 리뷰 큐
│       ├── value.py        # NPV 3Y → Signal
│       └── monitoring_node.py  # PSI, ECE, regime
├── phase/
│   ├── controller.py       # 4-phase 상태 머신
│   ├── weight_learner.py   # EMA 적응형 가중치
│   └── ridge_learner.py    # Nested LOOCV Ridge
├── monitoring/
│   ├── metrics.py          # 6-metric 검증 스위트
│   ├── psi.py              # Population Stability Index
│   ├── ece.py              # Expected Calibration Error
│   └── regime.py           # Loop1/Loop2 알림
├── prompts/
│   ├── rubric_5dim.py      # 5-dim 가치 렌즈 루브릭
│   └── calibrator.py       # 캘리브레이션 프롬프트
├── data/
│   └── synthetic.py        # 27-genre 합성 데이터 생성기
└── ui/
    ├── app.py              # Streamlit 엔트리
    ├── theme.py            # CSS 테마
    ├── export.py           # JSON/CSV 내보내기
    └── pages/              # 4-page 대시보드
```

## Monitoring

### Validation Metrics

| Metric | Strong | Conditional | Purpose |
|--------|--------|-------------|---------|
| Spearman ρ | ≥ 0.50 | ≥ 0.35 | 랭킹 품질 |
| Pearson r | ≥ 0.45 | ≥ 0.30 | 선형 상관 |
| NDCG@k (log) | ≥ 0.70 | ≥ 0.55 | 랭킹 정확도 |
| NDCG@k (hit) | ≥ 0.60 | ≥ 0.40 | 티어 랭킹 |
| Precision@k | ≥ 0.50 | ≥ 0.35 | 선별 정밀도 |
| Hit Rate@k | ≥ 0.20 | ≥ 0.12 | Mega/Hit 포착률 |

### Regime Detection

- **Loop 1** (분기별): 장르 편향, 에스컬레이션 비율, 가중치 시프트
- **Loop 2** (프로모션 게이트): Spearman ρ, Hit AUC, 마이그레이션 비율

## Testing

```bash
# 전체 테스트
uv run pytest

# 커버리지
uv run pytest --cov=llmart --cov-report=term-missing

# 품질 검사
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/llmart/
uv run bandit -r src/llmart/ -c pyproject.toml
```

## Configuration

`LLMARTConfig` dataclass로 관리. 환경 변수 오버라이드 지원:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | | GPT API 키 (primary judge) |
| `ANTHROPIC_API_KEY` | | Claude API 키 (calibrator) |

## License

Internal use only.
