---
name: llm-evaluation
description: LLMART LLM 평가 전략. ECE calibration, Cohen's Kappa, NDCG@30, A/B testing, regression detection. T2 LLM-as-Judge 품질 측정 시 참조. "evaluation", "ECE", "kappa", "ndcg", "regression", "benchmark" 키워드로 트리거.
---

# LLM Evaluation for LLMART

## LLMART Evaluation Map

```
┌─────────────────────────────────────────────────────────────┐
│                    LLMART Evaluation Metrics                  │
├──────────────────┬──────────────────────────────────────────┤
│ T1 ML Scoring    │ NDCG@30, Precision@K, Recall@99          │
│ T2 LLM Judge     │ ECE < 0.10, Cohen's Kappa, pass^3 rate  │
│ Selection Score   │ PSI < 0.25, rho > 0.50                  │
│ Value Inference   │ NPV accuracy, Signal precision           │
│ Full Pipeline     │ End-to-end regression detection           │
└──────────────────┴──────────────────────────────────────────┘
```

## ECE (Expected Calibration Error) — Target < 0.10

```python
import numpy as np

def expected_calibration_error(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE: 예측 확률과 실제 비율의 가중 절대 차이.

    LLMART에서 T2 LLM judge의 jury_score가 실제 Q1 성과와
    얼마나 잘 calibrated되어 있는지 측정.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
        if mask.sum() > 0:
            avg_pred = predictions[mask].mean()
            avg_actual = actuals[mask].mean()
            ece += mask.sum() / len(predictions) * abs(avg_pred - avg_actual)
    return ece

# Usage in LLMART
def validate_calibration(jury_scores: list[float], actual_outcomes: list[float]) -> bool:
    ece = expected_calibration_error(
        np.array(jury_scores),
        np.array(actual_outcomes),
    )
    return ece < 0.10  # LLMART threshold
```

## NDCG@30 — T1 ML Ranking Quality

```python
def ndcg_at_k(
    predicted_ranking: list[str],
    true_relevance: dict[str, float],
    k: int = 30,
) -> float:
    """NDCG@30: 상위 30개 순위 품질.

    T1 ML scoring이 실제 히트 게임을 상위에 배치하는지 측정.
    """
    dcg = sum(
        true_relevance.get(predicted_ranking[i], 0) / np.log2(i + 2)
        for i in range(min(k, len(predicted_ranking)))
    )
    ideal = sorted(true_relevance.values(), reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
```

## Cohen's Kappa — Inter-Rater Agreement

```python
from sklearn.metrics import cohen_kappa_score

def evaluate_judge_agreement(
    evaluations_run1: list[str],  # ["H", "M", "L", ...]
    evaluations_run2: list[str],
) -> dict[str, float]:
    """pass^3에서 k번 평가 간 일치도 측정.

    4-cat (L/M/H/E) 사용 시 Cohen's Kappa +0.15~0.20 향상 기대.
    """
    kappa = cohen_kappa_score(evaluations_run1, evaluations_run2)
    return {
        "kappa": kappa,
        "interpretation": _interpret_kappa(kappa),
    }

def _interpret_kappa(kappa: float) -> str:
    if kappa < 0.20: return "Poor"
    if kappa < 0.40: return "Fair"
    if kappa < 0.60: return "Moderate"
    if kappa < 0.80: return "Substantial"
    return "Almost Perfect"
```

## PSI (Population Stability Index) — Drift Detection

```python
def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """PSI: 분포 변화 감지.

    > 0.25 → WARNING (Regime shift 의심)
    > 0.10 → MONITOR
    """
    bins = np.linspace(0, 1, n_bins + 1)
    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bins)[0] / len(actual)

    # Avoid log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
```

## Regression Detection

```python
from dataclasses import dataclass

@dataclass
class RegressionResult:
    metric: str
    baseline: float
    current: float
    change_pct: float
    is_regression: bool

def check_regression(
    baseline: dict[str, float],
    current: dict[str, float],
    threshold: float = 0.05,
) -> list[RegressionResult]:
    """분기별 모델 업데이트 시 성능 하락 감지."""
    regressions = []
    for metric, base_val in baseline.items():
        curr_val = current.get(metric, 0.0)
        change = (curr_val - base_val) / max(base_val, 1e-6)
        regressions.append(RegressionResult(
            metric=metric,
            baseline=base_val,
            current=curr_val,
            change_pct=change,
            is_regression=change < -threshold,
        ))
    return regressions
```

## LLM-as-Judge Evaluation

```python
async def evaluate_judge_quality(
    game_data: dict,
    judge_result: dict,
    reference_label: str,  # Ground truth Q1 tier
) -> dict:
    """T2 judge의 평가 품질 측정.

    Dimensions:
    - Accuracy: jury_score vs actual Q1 outcome
    - Consistency: pass^3 agreement rate
    - Calibration: ECE across score bins
    """
    cat_to_tier = {"L": "Hobby", "M": "Side", "H": "Hit", "E": "Mega"}

    # Weighted score → predicted tier
    predicted_tier = _score_to_tier(judge_result["jury_score"])

    return {
        "tier_match": predicted_tier == reference_label,
        "jury_score": judge_result["jury_score"],
        "actual_tier": reference_label,
        "predicted_tier": predicted_tier,
    }
```

## LLMART Evaluation Thresholds

| Metric | Target | Action if Violated |
|--------|--------|-------------------|
| ECE | < 0.10 | 재학습 트리거 |
| PSI | < 0.25 | WARNING → 모델 검토 |
| rho (Spearman) | > 0.50 | WARNING → feature 재검토 |
| NDCG@30 | > 0.70 | T1 ML 재학습 |
| Cohen's Kappa | > 0.60 | Rubric 조정 |
| pass^3 rate | > 0.70 | Prompt 최적화 |

## Cost Model & Budget Guardrails

| Item | Tokens | Cost (Batch API) |
|------|--------|-----------------|
| Input per eval | ~1,500 | — |
| Output per eval | ~350 | — |
| Per game | ~1,850 | $0.015 |
| Per quarter (500 games) | ~3.25M | $15.63 |

Budget alert at 80% quarterly spend. Circuit breaker trips after 5 consecutive LLM failures.

## delta_cal Calibration Offset

`delta_cal = clip(|ECE_observed - ECE_target|, 0, 0.05)`

Applied as score adjustment in Selection Score: `S = w_ml * Phi_ml + w_llm * Phi_llm + delta_cal`.
Positive delta_cal rewards underrated quality. Negative penalizes inflation. Range: [-0.15, +0.15].

## E(4) Ceiling Guard

Track % of Exceptional (E=4) ratings per dimension per batch. Alert if >10% in any single dimension.
Action: Review rubric anchors, check leniency bias in prompt, consider tightening E(4) criteria.

## Reference

- Based on [wshobson/agents llm-evaluation](https://github.com/wshobson/agents)
- LLMART SOT: B3-ECE-Calibration, D1-RegimeMonitor
