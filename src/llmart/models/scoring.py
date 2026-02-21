"""Selection Score formula: S = w_ml * Phi_ml + w_llm * Phi_llm + delta_cal."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Phase 0 defaults
DEFAULT_W_ML: float = 0.6
DEFAULT_W_LLM: float = 0.4


class SelectionScore(BaseModel):
    """Selection Score computation."""

    phi_ml: float = Field(description="ML score (LambdaMART)")
    phi_llm: float = Field(description="LLM judge score")
    delta_cal: float = Field(default=0.0, description="Calibration offset")
    w_ml: float = Field(default=DEFAULT_W_ML)
    w_llm: float = Field(default=DEFAULT_W_LLM)

    @property
    def score(self) -> float:
        """S = w_ml * Phi_ml + w_llm * Phi_llm + delta_cal."""
        return self.w_ml * self.phi_ml + self.w_llm * self.phi_llm + self.delta_cal
