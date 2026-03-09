"""Validation Suite: NDCG@k, Spearman, Pearson, Precision, Hit Rate."""

from __future__ import annotations

import math

# SOT Strong/Conditional thresholds
_THRESHOLDS: dict[str, tuple[float, float]] = {
    # metric_name: (strong, conditional)
    "spearman_rho": (0.50, 0.35),
    "pearson_log": (0.45, 0.30),
    "precision_at_k": (0.50, 0.35),
    "ndcg_log_at_k": (0.70, 0.55),
    "ndcg_hit_at_k": (0.60, 0.40),
    "hit_rate_at_k": (0.20, 0.12),
}


def compute_ndcg(
    scores: list[float],
    relevance: list[float],
    k: int = 30,
) -> float:
    """Compute NDCG@k.

    DCG = sum(rel[i] / log2(i+2)) for i in 0..k-1.
    NDCG = DCG / IDCG.
    """
    if not scores or not relevance or len(scores) != len(relevance):
        return 0.0

    # Rank by scores (descending), take top-k
    paired = sorted(zip(scores, relevance, strict=True), key=lambda x: x[0], reverse=True)
    paired = paired[:k]

    # DCG
    dcg = 0.0
    for i, (_, rel) in enumerate(paired):
        dcg += rel / math.log2(i + 2)

    if dcg == 0.0:
        return 0.0

    # IDCG: sort by relevance descending
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))

    if idcg == 0.0:  # pragma: no cover — DCG>0 implies IDCG>0
        return 0.0

    return round(dcg / idcg, 6)


def compute_spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation with average-rank tie handling."""
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # average rank for tied values
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry, strict=True))
    return round(1.0 - (6 * d_sq) / (n * (n * n - 1)), 6)


# Keep backward-compatible alias
_spearman_rho = compute_spearman_rho


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return round(cov / (sx * sy), 6)


def compute_validation_suite(
    sel_scores: list[float],
    y1_revenues: list[float],
    k: int = 30,
) -> dict[str, float]:
    """Compute the 6-metric Validation Suite.

    Args:
        sel_scores: Selection scores for each candidate.
        y1_revenues: Actual Y1 revenues (ground truth).
        k: Cutoff for @k metrics.

    Returns:
        Dict with all 6 metric values.
    """
    n = min(len(sel_scores), len(y1_revenues))
    if n == 0:
        return dict.fromkeys(_THRESHOLDS, 0.0)

    scores = sel_scores[:n]
    revenues = y1_revenues[:n]
    log_rev = [math.log1p(r) for r in revenues]

    # Sort by score descending, take top-k
    paired = sorted(zip(scores, revenues, log_rev, strict=True), key=lambda x: x[0], reverse=True)
    top_k = paired[:k]

    # Hit threshold: top 10% of revenue
    rev_sorted = sorted(revenues, reverse=True)
    hit_threshold = (
        rev_sorted[0] if len(rev_sorted) < 10 else rev_sorted[max(0, len(rev_sorted) // 10 - 1)]
    )

    # Binary relevance for hit-based metrics
    hit_relevance = [1.0 if r >= hit_threshold else 0.0 for _, r, _ in paired]

    # Spearman rho (full list)
    spearman = _spearman_rho(scores, revenues)

    # Pearson on log revenues
    pearson_log = _pearson(scores, log_rev)

    # Precision@k: fraction of top-k that are hits
    top_k_hits = sum(1.0 for _, r, _ in top_k if r >= hit_threshold)
    precision = round(top_k_hits / min(k, n), 6) if n > 0 else 0.0

    # NDCG@k with log relevance
    ndcg_log = compute_ndcg(scores, log_rev, k=k)

    # NDCG@k with hit binary relevance
    ndcg_hit = compute_ndcg(scores, hit_relevance, k=k)

    # Hit Rate@k: >= 1 hit in top-k
    hit_rate = round(top_k_hits / min(k, n), 6)

    return {
        "spearman_rho": spearman,
        "pearson_log": pearson_log,
        "precision_at_k": precision,
        "ndcg_log_at_k": ndcg_log,
        "ndcg_hit_at_k": ndcg_hit,
        "hit_rate_at_k": hit_rate,
    }


def check_pass_condition(metrics: dict[str, float]) -> bool:
    """Check if 4+ metrics meet Conditional+ threshold -> PASS."""
    passing = 0
    for name, (_, conditional) in _THRESHOLDS.items():
        if metrics.get(name, 0.0) >= conditional:
            passing += 1
    return passing >= 4


def classify_metric(name: str, value: float) -> str:
    """Classify a single metric as 'strong', 'conditional', or 'fail'."""
    if name not in _THRESHOLDS:
        return "unknown"
    strong, conditional = _THRESHOLDS[name]
    if value >= strong:
        return "strong"
    if value >= conditional:
        return "conditional"
    return "fail"
