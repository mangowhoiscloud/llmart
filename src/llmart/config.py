"""Central SOT constants for LLMART pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class LLMARTConfig:
    """Immutable configuration — SOT constants from pipeline-v8-spec & selection-score-spec."""

    # Phase 0 fixed weights — PDF spec: w = {0.6, 0.4} ML→LLM
    phase_0_w_ml: float = 0.6
    phase_0_w_llm: float = 0.4

    # Phase promotion sample thresholds
    phase_1_min_n: int = 15
    phase_2_min_n: int = 50
    phase_3_min_n: int = 200

    # Sigmoid: sigma(Delta) = 1/(1+exp(k*(Delta-d0)))
    sigmoid_k: float = 2.0
    sigmoid_d0: float = 1.0

    # EMA smoothing
    ema_alpha: float = 0.3

    # Monitoring thresholds
    ece_target: float = 0.10
    psi_stable: float = 0.10
    psi_warning: float = 0.25

    # Phase promotion conditions
    promote_1to2_min_rho: float = 0.35
    promote_1to2_max_w_std: float = 0.15
    promote_2to3_min_rho: float = 0.50
    promote_2to3_min_n: int = 200

    # Escalation thresholds
    escalation_gap_threshold: float = 1.8  # T1/T2 gap — SOT target 10-15%
    escalation_multi_dim_count: int = 2
    escalation_multi_dim_delta: float = 1.8  # multi-dim disagree threshold
    escalation_extreme_dim_delta: float = 2.5  # single-dim extreme threshold
    escalation_low_coverage_rate: float = 0.4
    escalation_boundary_low: float = 0.48  # narrow boundary zone
    escalation_boundary_high: float = 0.52

    # Regime monitoring thresholds (Loop 1 + Loop 2)
    regime_genre_bias_max: float = 0.40
    regime_weight_shift_max: float = 0.10
    regime_escalation_rate_max: float = 0.30
    regime_migration_rate_max: float = 0.30
    regime_rho_min: float = 0.50
    regime_hit_auc_min: float = 0.65

    @classmethod
    def from_env(cls) -> LLMARTConfig:
        """Create config from LLMART_* environment variables (12-factor).

        Maps ``LLMART_PHASE_0_W_ML=0.7`` → ``phase_0_w_ml=0.7``, etc.
        Unset variables keep their default values.
        """
        kwargs: dict[str, float | int] = {}
        for f in fields(cls):
            env_key = f"LLMART_{f.name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                if f.type == "float":
                    kwargs[f.name] = float(env_val)
                elif f.type == "int":
                    kwargs[f.name] = int(env_val)
        return cls(**kwargs)  # type: ignore[arg-type]


# Singleton config instance
DEFAULT_CONFIG = LLMARTConfig()
