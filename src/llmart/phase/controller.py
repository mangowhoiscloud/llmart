"""Phase Controller: manages phase transitions and weight history."""

from __future__ import annotations

from llmart.config import DEFAULT_CONFIG, LLMARTConfig
from llmart.phase.weight_learner import compute_adaptive_weights, ema_smooth


class PhaseController:
    """Manages phase 0->1->2->3 transitions with adaptive weights.

    Phase 0: Fixed weights (0.6/0.4) — cold start
    Phase 1: Adaptive weights begin (n >= 15)
    Phase 2: Validated adaptive (n >= 50, rho >= 0.35, w_std <= 0.15)
    Phase 3: Production (n >= 200, rho >= 0.50)
    """

    def __init__(
        self,
        config: LLMARTConfig | None = None,
        initial_phase: int = 0,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.current_phase: int = initial_phase
        self.weight_history: list[tuple[float, float]] = []

    def get_weights(
        self,
        n_samples: int,
        agreement_delta: float = 0.0,
        conf_scalar: float = 1.0,
        coverage: float = 1.0,
    ) -> tuple[float, float]:
        """Get current weights based on phase and sample count.

        Phase 0 always returns fixed weights. Phase 1+ uses adaptive computation
        with EMA smoothing.
        """
        # Auto-promote to Phase 1 when enough samples
        if self.current_phase == 0 and n_samples >= self.config.phase_1_min_n:
            self.current_phase = 1

        if self.current_phase == 0:
            weights = (self.config.phase_0_w_ml, self.config.phase_0_w_llm)
        else:
            new_w = compute_adaptive_weights(agreement_delta, conf_scalar, coverage, self.config)
            if self.weight_history:
                weights = ema_smooth(new_w, self.weight_history[-1], self.config.ema_alpha)
            else:
                weights = new_w

        self.weight_history.append(weights)
        return weights

    def evaluate_promotion(
        self,
        n: int,
        rho: float,
        w_std: float = 0.0,
    ) -> str:
        """Evaluate whether to promote, demote, or hold current phase.

        Returns:
            "promote", "demote", or "hold".
        """
        if self.current_phase == 0:
            if n >= self.config.phase_1_min_n:
                return "promote"
            return "hold"

        if self.current_phase == 1:
            if (
                n >= self.config.phase_2_min_n
                and rho >= self.config.promote_1to2_min_rho
                and w_std <= self.config.promote_1to2_max_w_std
            ):
                return "promote"
            if rho < self.config.promote_1to2_min_rho * 0.5:
                return "demote"
            return "hold"

        if self.current_phase == 2:
            if n >= self.config.promote_2to3_min_n and rho >= self.config.promote_2to3_min_rho:
                return "promote"
            if rho < self.config.promote_1to2_min_rho:
                return "demote"
            return "hold"

        # Phase 3: can only demote
        if self.current_phase == 3 and rho < self.config.promote_2to3_min_rho:
            return "demote"

        return "hold"

    def apply_promotion(self, decision: str) -> int:
        """Apply a promotion/demotion decision. Returns new phase."""
        if decision == "promote" and self.current_phase < 3:
            self.current_phase += 1
        elif decision == "demote" and self.current_phase > 0:
            self.current_phase -= 1
        return self.current_phase

    @property
    def weight_std(self) -> float:
        """Standard deviation of recent w_ml values (last 20)."""
        if len(self.weight_history) < 2:
            return 0.0
        recent = [w[0] for w in self.weight_history[-20:]]
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return float(round(variance**0.5, 4))
