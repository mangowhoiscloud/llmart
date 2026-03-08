"""Expected Calibration Error (ECE) computation.

ECE = sum(|B_m|/n * |acc(B_m) - conf(B_m)|) for m in bins.
"""

from __future__ import annotations

from llmart.config import DEFAULT_CONFIG

# SOT calibration table: (min_score, max_score) -> expected pass rate
CALIBRATION_TABLE: dict[tuple[int, int], float] = {
    (1, 2): 0.05,
    (2, 3): 0.15,
    (3, 4): 0.35,
    (4, 5): 0.60,
}


def compute_ece(
    predictions: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        predictions: Predicted probabilities [0, 1].
        outcomes: Binary outcomes (True = positive).
        n_bins: Number of equal-width bins.

    Returns:
        ECE value in [0, 1].
    """
    if not predictions or len(predictions) != len(outcomes):
        return 0.0

    n = len(predictions)
    bin_width = 1.0 / n_bins
    ece = 0.0

    for i in range(n_bins):
        lo = i * bin_width
        hi = lo + bin_width
        # Collect samples in this bin
        indices = [
            j for j, p in enumerate(predictions) if lo <= p < hi or (i == n_bins - 1 and p == hi)
        ]
        if not indices:
            continue
        bin_size = len(indices)
        avg_conf = sum(predictions[j] for j in indices) / bin_size
        avg_acc = sum(1.0 for j in indices if outcomes[j]) / bin_size
        ece += (bin_size / n) * abs(avg_acc - avg_conf)

    return round(ece, 6)


def needs_retrain(ece: float, target: float | None = None) -> bool:
    """Check if ECE exceeds the target threshold."""
    if target is None:
        target = DEFAULT_CONFIG.ece_target
    return ece > target
