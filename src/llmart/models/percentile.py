"""Percentile Rank with rolling reference, tail stabilization, and KDE fallback.

Per selection-score-spec §A.3:
- Standard percentile rank (Weibull plotting position) for n >= 50
- KDE-CDF approximation for n < 50 (histogram-smoothed)
- Tail stabilization for top 5% (P95+)
- Rolling 1-Quarter reference distribution management
"""

from __future__ import annotations

import math
from collections.abc import Callable

P95_THRESHOLD: float = 0.95
KDE_N_THRESHOLD: int = 50

# Type alias for ranker callables
_Ranker = Callable[[float], float]


def percentile_rank_standard(
    ref_scores: list[float],
    new_score: float,
    stabilize_tail: bool = True,
) -> float:
    """Compute percentile rank of new_score against reference distribution.

    Uses Weibull plotting position: rank / (n+1) to avoid 0 and 1 extremes.
    Tail stabilization: top 5% uses raw-score proportional sub-ranking.
    """
    if not ref_scores:
        return 0.5

    all_scores = sorted([*ref_scores, new_score])
    n = len(all_scores)
    # Weibull: rank / (n+1)
    rank: float = all_scores.index(new_score) + 1
    # Handle duplicates: use average rank
    count = all_scores.count(new_score)
    if count > 1:
        first_idx = all_scores.index(new_score)
        rank = first_idx + 1 + (count - 1) / 2
    phi = rank / (n + 1)

    if stabilize_tail and phi > P95_THRESHOLD:
        p95_value = _percentile(ref_scores, 95)
        max_value = max(ref_scores)
        if max_value > p95_value:
            phi = P95_THRESHOLD + 0.05 * (new_score - p95_value) / (max_value - p95_value)
            phi = max(P95_THRESHOLD, min(1.0, phi))

    return round(phi, 6)


def percentile_rank_batch(
    ref_scores: list[float],
    scores: list[float],
    stabilize_tail: bool = True,
) -> list[float]:
    """Compute percentile ranks for a batch of scores."""
    return [percentile_rank_standard(ref_scores, s, stabilize_tail) for s in scores]


def _kde_cdf(ref_scores: list[float], new_score: float) -> float:
    """KDE-CDF estimate for small samples (n < 50).

    Uses Gaussian kernel with Silverman bandwidth.
    CDF(x) = mean(Φ((x - xi) / h)) where Φ is standard normal CDF.
    """
    n = len(ref_scores)
    if n == 0:
        return 0.5

    mean_s = sum(ref_scores) / n
    std_s = math.sqrt(sum((x - mean_s) ** 2 for x in ref_scores) / max(n - 1, 1))
    if std_s < 1e-10:
        return 0.5

    # Silverman bandwidth
    iqr = _percentile(ref_scores, 75) - _percentile(ref_scores, 25)
    h = 0.9 * min(std_s, iqr / 1.34 if iqr > 0 else std_s) * n ** (-0.2)
    if h < 1e-10:
        h = std_s * n ** (-0.2)

    # CDF estimate: average of normal CDFs centred at each data point
    total = 0.0
    for xi in ref_scores:
        z = (new_score - xi) / h
        total += _normal_cdf(z)

    cdf = total / n
    return max(0.01, min(0.99, round(cdf, 6)))


def _normal_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _percentile(data: list[float], pct: float) -> float:
    """Simple percentile calculation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (pct / 100.0) * (len(sorted_data) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


class PercentileRankManager:
    """Rolling 1-Quarter Percentile Rank distribution management.

    Maintains reference distributions for ML and Jury scores.
    Uses standard percentile rank for n >= 50, KDE-CDF for n < 50.
    """

    MAX_QUARTERS: int = 2  # Keep 2 quarters for QoQ comparison + PSI

    def __init__(self) -> None:
        self._ml_raw: dict[str, list[float]] = {}
        self._jury_raw: dict[str, list[float]] = {}
        self._ml_phi: dict[str, list[float]] = {}
        self._jury_phi: dict[str, list[float]] = {}

    def update_quarter(
        self,
        quarter: str,
        ml_scores: list[float],
        jury_scores: list[float],
    ) -> None:
        """Register a quarter's scores and compute percentile ranks."""
        self._ml_raw[quarter] = list(ml_scores)
        self._jury_raw[quarter] = list(jury_scores)

        ref_ml = self._get_rolling_raw("ml")
        ref_jury = self._get_rolling_raw("jury")

        ranker_ml = self.get_ranker("ml")
        ranker_jury = self.get_ranker("jury")

        self._ml_phi[quarter] = [ranker_ml(s) for s in ml_scores] if ref_ml else []
        self._jury_phi[quarter] = [ranker_jury(s) for s in jury_scores] if ref_jury else []

        # Evict oldest quarters beyond retention limit
        for store in [self._ml_raw, self._jury_raw, self._ml_phi, self._jury_phi]:
            while len(store) > self.MAX_QUARTERS:
                oldest = min(store.keys())
                del store[oldest]

    def _get_rolling_raw(self, score_type: str) -> list[float]:
        """Get concatenated raw scores from all retained quarters."""
        store = self._ml_raw if score_type == "ml" else self._jury_raw
        result: list[float] = []
        for vals in store.values():
            result.extend(vals)
        return result

    def get_ranker(self, score_type: str = "ml") -> _Ranker:
        """Return a ranking function using appropriate method for sample size."""
        ref = self._get_rolling_raw(score_type)
        if len(ref) >= KDE_N_THRESHOLD:
            return lambda s: percentile_rank_standard(ref, s, stabilize_tail=True)
        if ref:
            return lambda s: _kde_cdf(ref, s)
        return lambda _: 0.5

    def compute_psi(self, q_prev: str, q_curr: str, score_type: str = "ml") -> float:
        """Compute PSI between two quarters' percentile distributions."""
        from llmart.monitoring.psi import compute_psi as _psi

        phi_store = self._ml_phi if score_type == "ml" else self._jury_phi
        prev = phi_store.get(q_prev, [])
        curr = phi_store.get(q_curr, [])
        if not prev or not curr:
            return 0.0
        return _psi(prev, curr, n_bins=10)

    @property
    def quarters(self) -> list[str]:
        """Return sorted list of stored quarters."""
        return sorted(self._ml_raw.keys())
