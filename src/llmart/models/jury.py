"""Confidence-Weighted Jury Score per selection-score-spec §2.2.

jury_star() computes a confidence-weighted consensus from dual-judge inputs.
jury_star_from_passes() adapts pass^3 results to the dual-judge interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 5-Dim Value Lens (PDF slide 2)
DIMS: list[str] = ["gameplay", "innovation", "monetization", "polish", "narrative"]
UNKNOWN_DIM_WEIGHT: float = 0.05  # minimal weight for unknown dimensions


@dataclass(frozen=True)
class JudgeInput:
    """Single judge's per-dimension scores and confidence."""

    dim_scores: dict[str, float]  # {dim: score ∈ [1,4]}
    dim_conf: dict[str, float]  # {dim: conf ∈ [0,1]}


@dataclass(frozen=True)
class JuryResult:
    """Output of jury_star(). Feeds into Selection Score and Escalation.

    Attributes:
        jury_final: Confidence-weighted consensus score.
        delta: |score_A - score_B| — agreement measure.
        conf_scalar: Mean confidence across known dims.
        unknown_ratio: #unknown / #DIMS — data coverage measure.
        dim_deltas: Per-dimension |A-B| for escalation checks.
    """

    jury_final: float
    delta: float
    conf_scalar: float
    unknown_ratio: float
    dim_deltas: dict[str, float] = field(default_factory=dict)


def jury_star(
    judge_a: JudgeInput,
    judge_b: JudgeInput,
    unknown_dims: set[str] | None = None,
) -> JuryResult:
    """Confidence-Weighted Jury Score per selection-score-spec §2.2.

    conf² weighting: 0.9→0.81, 0.5→0.25 (nonlinear confidence dampening).
    Unknown dims get UNKNOWN_DIM_WEIGHT to prevent zero-weight collapse.
    """
    if unknown_dims is None:
        unknown_dims = set()

    unknown_ratio = len(unknown_dims) / len(DIMS) if DIMS else 0.0

    judge_scores: list[float] = []
    judge_confs: list[float] = []
    judge_dim_scores: dict[str, list[float]] = {}

    for judge in [judge_a, judge_b]:
        weighted_sum = 0.0
        weight_total = 0.0
        known_confs: list[float] = []

        for dim in DIMS:
            score_d = judge.dim_scores.get(dim, 3.0)  # neutral default
            conf_d = judge.dim_conf.get(dim, 0.5)

            judge_dim_scores.setdefault(dim, []).append(score_d)

            if dim in unknown_dims:
                w_d = UNKNOWN_DIM_WEIGHT
            else:
                w_d = conf_d**2  # 0.9→0.81, 0.5→0.25
                known_confs.append(conf_d)

            weighted_sum += score_d * w_d
            weight_total += w_d

        judge_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        conf_scalar = sum(known_confs) / len(known_confs) if known_confs else 0.0

        judge_scores.append(judge_score)
        judge_confs.append(conf_scalar)

    conf_a, conf_b = judge_confs
    score_a, score_b = judge_scores

    # Confidence-proportional consensus (zero-division guard)
    conf_sum = conf_a + conf_b
    if conf_sum < 1e-8:
        jury_final = (score_a + score_b) / 2
    else:
        jury_final = (score_a * conf_a + score_b * conf_b) / conf_sum

    delta = abs(score_a - score_b)

    dim_deltas = {
        dim: abs(scores[0] - scores[1])
        for dim, scores in judge_dim_scores.items()
        if len(scores) == 2
    }

    return JuryResult(
        jury_final=round(jury_final, 4),
        delta=round(delta, 4),
        conf_scalar=round((conf_a + conf_b) / 2, 4),
        unknown_ratio=round(unknown_ratio, 4),
        dim_deltas=dim_deltas,
    )


def jury_star_from_passes(
    passes: list[dict[str, int | float]],
    unknown_dims: set[str] | None = None,
) -> JuryResult:
    """Adapter: convert pass^3 results to jury_star() dual-judge format.

    Splits k passes into two groups: passes[:-1] → judge_a, passes[-1] → judge_b.
    Confidence is derived from score consistency across passes.
    """
    if len(passes) < 2:
        scores = passes[0] if passes else {}
        total = sum(float(v) for v in scores.values())
        n = max(len(scores), 1)
        return JuryResult(
            jury_final=round(total / n, 4),
            delta=0.0,
            conf_scalar=1.0,
            unknown_ratio=0.0,
            dim_deltas=dict.fromkeys(DIMS, 0.0),
        )

    # Average first k-1 passes as judge_a, last pass as judge_b
    judge_a_scores: dict[str, float] = {}
    for dim in DIMS:
        vals = [float(p.get(dim, 2.0)) for p in passes[:-1]]
        judge_a_scores[dim] = sum(vals) / len(vals) if vals else 2.0

    judge_b_scores = {dim: float(passes[-1].get(dim, 2.0)) for dim in DIMS}

    # Derive confidence from pass-to-pass consistency
    a_confs: dict[str, float] = {}
    b_confs: dict[str, float] = {}
    for dim in DIMS:
        all_vals = [float(p.get(dim, 2.0)) for p in passes]
        spread = max(all_vals) - min(all_vals)
        conf = max(0.1, 1.0 - spread / 3.0)  # max spread=3 → conf=0.1
        a_confs[dim] = conf
        b_confs[dim] = conf

    return jury_star(
        JudgeInput(dim_scores=judge_a_scores, dim_conf=a_confs),
        JudgeInput(dim_scores=judge_b_scores, dim_conf=b_confs),
        unknown_dims=unknown_dims,
    )
