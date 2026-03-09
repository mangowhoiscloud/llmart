"""Population Stability Index (PSI) computation.

PSI = sum((P_actual% - P_ref%) * ln(P_actual% / P_ref%)) for each bin.
"""

from __future__ import annotations

import math

from llmart.config import DEFAULT_CONFIG, LLMARTConfig


def compute_psi(
    reference: list[float],
    current: list[float],
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between reference and current distributions.

    Args:
        reference: Reference distribution values.
        current: Current distribution values.
        n_bins: Number of equal-width bins.

    Returns:
        PSI value (0 = identical distributions).
    """
    if not reference or not current:
        return 0.0

    all_vals = reference + current
    lo = min(all_vals)
    hi = max(all_vals)
    if hi <= lo:
        return 0.0

    bin_width = (hi - lo) / n_bins
    laplace_count = 0.001  # Laplace smoothing numerator (SOT §A.3)
    laplace_total = n_bins * laplace_count  # Laplace smoothing denominator

    ref_counts: list[int] = [0] * n_bins
    cur_counts: list[int] = [0] * n_bins

    for i in range(n_bins):
        bin_lo = lo + i * bin_width
        bin_hi = bin_lo + bin_width
        for v in reference:
            if bin_lo <= v < bin_hi or (i == n_bins - 1 and v == bin_hi):
                ref_counts[i] += 1
        for v in current:
            if bin_lo <= v < bin_hi or (i == n_bins - 1 and v == bin_hi):
                cur_counts[i] += 1

    psi = 0.0
    ref_total = len(reference)
    cur_total = len(current)
    for i in range(n_bins):
        ref_pct = (ref_counts[i] + laplace_count) / (ref_total + laplace_total)
        cur_pct = (cur_counts[i] + laplace_count) / (cur_total + laplace_total)
        psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)

    return round(abs(psi), 6)


def classify_psi(psi: float, config: LLMARTConfig | None = None) -> str:
    """Classify PSI into stability categories.

    Returns:
        "stable" (< 0.10), "caution" (0.10-0.25), or "warning" (> 0.25).
    """
    if config is None:
        config = DEFAULT_CONFIG
    if psi > config.psi_warning:
        return "warning"
    if psi > config.psi_stable:
        return "caution"
    return "stable"
