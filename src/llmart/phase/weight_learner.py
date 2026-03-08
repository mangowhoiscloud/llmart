"""Adaptive weight learning via jury reliability sigmoid.

sigma(Delta) = 1/(1+exp(k*(Delta-d0)))
SOT reference: Delta=0->0.88, 0.5->0.73, 1.0->0.50, 1.5->0.27, 2.0->0.12
"""

from __future__ import annotations

import math

from llmart.config import DEFAULT_CONFIG, LLMARTConfig


def jury_reliability(
    delta: float,
    k: float | None = None,
    d0: float | None = None,
) -> float:
    """Compute jury reliability via sigmoid.

    sigma(Delta) = 1/(1+exp(k*(Delta-d0)))

    Higher agreement (lower delta) -> higher reliability.
    """
    if k is None:
        k = DEFAULT_CONFIG.sigmoid_k
    if d0 is None:
        d0 = DEFAULT_CONFIG.sigmoid_d0
    return 1.0 / (1.0 + math.exp(k * (delta - d0)))


def compute_adaptive_weights(
    agreement_delta: float,
    conf_scalar: float = 1.0,
    coverage: float = 1.0,
    config: LLMARTConfig | None = None,
) -> tuple[float, float]:
    """Compute adaptive w_ml, w_llm based on agreement and coverage.

    w_llm = sigma(agreement) * conf_scalar * coverage
    w_ml = 1 - w_llm

    Both weights are clamped to [0.1, 0.9] to prevent degenerate extremes.
    """
    if config is None:
        config = DEFAULT_CONFIG
    sigma = jury_reliability(agreement_delta, config.sigmoid_k, config.sigmoid_d0)
    w_llm_raw = sigma * conf_scalar * coverage
    w_llm = max(0.1, min(0.9, w_llm_raw))
    w_ml = 1.0 - w_llm
    return round(w_ml, 4), round(w_llm, 4)


def ema_smooth(
    new_w: tuple[float, float],
    old_w: tuple[float, float],
    alpha: float | None = None,
) -> tuple[float, float]:
    """EMA smoothing: smoothed = alpha * new + (1-alpha) * old."""
    if alpha is None:
        alpha = DEFAULT_CONFIG.ema_alpha
    w_ml = round(alpha * new_w[0] + (1 - alpha) * old_w[0], 4)
    w_llm = round(alpha * new_w[1] + (1 - alpha) * old_w[1], 4)
    # Renormalise to sum=1
    total = w_ml + w_llm
    if total > 0:
        w_ml = round(w_ml / total, 4)
        w_llm = round(w_llm / total, 4)
    return w_ml, w_llm
