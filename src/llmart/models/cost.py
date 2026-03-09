"""Cost tracking for LLM API calls — budget guardrail."""

from __future__ import annotations

from dataclasses import dataclass, field

# Pricing per 1M tokens (USD) — estimated for current models
# These are configurable but rarely change mid-pipeline.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.2": {"input": 2.50, "output": 10.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
}


@dataclass
class APICallCost:
    """Cost record for a single API call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Accumulates LLM API costs across a pipeline run.

    Raises ``BudgetExceededError`` when cumulative cost exceeds
    ``max_batch_cost_usd``.  Spec: ~$0.015/game, $15.63/quarter (500 games).
    """

    max_batch_cost_usd: float = 100.0  # conservative default
    calls: list[APICallCost] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def budget_remaining_usd(self) -> float:
        return max(0.0, self.max_batch_cost_usd - self.total_cost_usd)

    @property
    def budget_utilization_pct(self) -> float:
        if self.max_batch_cost_usd <= 0:
            return 100.0
        return min(100.0, self.total_cost_usd / self.max_batch_cost_usd * 100)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> APICallCost:
        """Record an API call and return its cost entry.

        Raises:
            BudgetExceededError: if cumulative cost exceeds ``max_batch_cost_usd``.
        """
        pricing = MODEL_PRICING.get(model, {"input": 10.0, "output": 50.0})
        input_cost = prompt_tokens * pricing["input"]
        output_cost = completion_tokens * pricing["output"]
        cost = (input_cost + output_cost) / 1_000_000
        entry = APICallCost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost, 6),
        )
        self.calls.append(entry)
        if self.total_cost_usd > self.max_batch_cost_usd:
            raise BudgetExceededError(self.total_cost_usd, self.max_batch_cost_usd)
        return entry

    def summary(self) -> dict[str, float | int]:
        """Return a summary dict suitable for monitoring state."""
        return {
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "call_count": self.call_count,
            "budget_remaining_usd": round(self.budget_remaining_usd, 4),
            "budget_utilization_pct": round(self.budget_utilization_pct, 2),
        }


class BudgetExceededError(RuntimeError):
    """Raised when cumulative LLM cost exceeds the batch budget."""

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exceeded: ${spent:.4f} > ${limit:.2f} limit")
