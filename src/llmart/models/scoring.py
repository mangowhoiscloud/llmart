"""Selection Score formula: S = w_ml * Phi_ml + w_llm * Phi_llm + delta_cal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from llmart.phase.controller import PhaseController

# Phase 0 defaults — PDF spec: w = {0.6, 0.4} ML→LLM
DEFAULT_W_ML: float = 0.6
DEFAULT_W_LLM: float = 0.4


class SelectionScore(BaseModel):
    """Selection Score computation with input validation."""

    phi_ml: float = Field(ge=0.0, le=1.0, description="ML percentile rank [0,1]")
    phi_llm: float = Field(ge=0.0, le=1.0, description="LLM jury normalised score [0,1]")
    delta_cal: float = Field(default=0.0, ge=-0.5, le=0.5, description="Calibration offset")
    w_ml: float = Field(default=DEFAULT_W_ML)
    w_llm: float = Field(default=DEFAULT_W_LLM)
    phase: int = Field(default=0, ge=0, le=3, description="Current pipeline phase")

    @model_validator(mode="after")
    def _check_weights_sum(self) -> SelectionScore:
        if abs(self.w_ml + self.w_llm - 1.0) > 1e-6:
            msg = f"w_ml + w_llm must equal 1.0, got {self.w_ml + self.w_llm}"
            raise ValueError(msg)
        return self

    @property
    def score(self) -> float:
        """S = clamp(w_ml * Phi_ml + w_llm * Phi_llm + delta_cal, 0, 1)."""
        raw = self.w_ml * self.phi_ml + self.w_llm * self.phi_llm + self.delta_cal
        return max(0.0, min(1.0, raw))

    @classmethod
    def from_phase(
        cls,
        phi_ml: float,
        phi_llm: float,
        delta_cal: float,
        controller: PhaseController,
        n_samples: int = 0,
        agreement_delta: float = 0.0,
        coverage: float = 1.0,
    ) -> SelectionScore:
        """Factory that uses PhaseController for adaptive weights."""
        w_ml, w_llm = controller.get_weights(
            n_samples=n_samples,
            agreement_delta=agreement_delta,
            coverage=coverage,
        )
        return cls(
            phi_ml=phi_ml,
            phi_llm=phi_llm,
            delta_cal=delta_cal,
            w_ml=w_ml,
            w_llm=w_llm,
            phase=controller.current_phase,
        )
