"""Value Inference: NPV_3Y, VaR, and signal classification."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

GREEN_THRESHOLD: float = 250_000.0


class Signal(StrEnum):
    """Traffic-light signal for game value."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class ValueInference(BaseModel):
    """Value(g) = E[NPV_3Y] - CAC - UA - LiveOps - 0.2 * VaR_5%."""

    npv_3y: float = Field(description="Expected 3-year NPV")
    cac: float = Field(default=0.0, description="Customer acquisition cost")
    ua: float = Field(default=0.0, description="User acquisition cost")
    live_ops: float = Field(default=0.0, description="LiveOps cost")
    var_5pct: float = Field(default=0.0, description="5th percentile VaR")
    q1_p25: float = Field(default=0.0, description="Q1 revenue P25")
    q1_p50: float = Field(default=0.0, description="Q1 revenue P50")

    @property
    def value(self) -> float:
        """Compute total value."""
        return self.npv_3y - self.cac - self.ua - self.live_ops - 0.2 * self.var_5pct

    @property
    def signal(self) -> Signal:
        """Classify as GREEN/YELLOW/RED based on Q1 thresholds."""
        if self.q1_p25 >= GREEN_THRESHOLD:
            return Signal.GREEN
        if self.q1_p50 >= GREEN_THRESHOLD:
            return Signal.YELLOW
        return Signal.RED
