"""Tests for CostTracker budget guardrail."""

from __future__ import annotations

import pytest

from llmart.models.cost import MODEL_PRICING, APICallCost, BudgetExceededError, CostTracker


def test_empty_tracker() -> None:
    tracker = CostTracker()
    assert tracker.total_cost_usd == 0.0
    assert tracker.call_count == 0
    assert tracker.budget_remaining_usd == 100.0
    assert tracker.budget_utilization_pct == 0.0


def test_record_gpt_call() -> None:
    tracker = CostTracker()
    entry = tracker.record("gpt-5.2", prompt_tokens=1500, completion_tokens=350)
    assert isinstance(entry, APICallCost)
    assert entry.model == "gpt-5.2"
    assert entry.prompt_tokens == 1500
    assert entry.completion_tokens == 350
    # Cost: (1500 * 2.50 + 350 * 10.00) / 1_000_000 = 0.007250
    assert abs(entry.cost_usd - 0.00725) < 1e-5
    assert tracker.call_count == 1
    assert tracker.total_cost_usd > 0


def test_record_opus_call() -> None:
    tracker = CostTracker()
    entry = tracker.record("claude-opus-4-6", prompt_tokens=1000, completion_tokens=200)
    # Cost: (1000 * 15.00 + 200 * 75.00) / 1_000_000 = 0.030000
    assert abs(entry.cost_usd - 0.03) < 1e-5


def test_accumulates_multiple_calls() -> None:
    tracker = CostTracker()
    tracker.record("gpt-5.2", 1500, 350)
    tracker.record("gpt-5.2", 1500, 350)
    tracker.record("gpt-5.2", 1500, 350)
    assert tracker.call_count == 3
    assert tracker.total_prompt_tokens == 4500
    assert tracker.total_completion_tokens == 1050


def test_budget_exceeded_raises() -> None:
    tracker = CostTracker(max_batch_cost_usd=0.001)  # very low budget
    with pytest.raises(BudgetExceededError, match="Budget exceeded"):
        tracker.record("gpt-5.2", 10000, 5000)


def test_budget_exceeded_error_fields() -> None:
    tracker = CostTracker(max_batch_cost_usd=0.001)
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record("gpt-5.2", 10000, 5000)
    assert exc_info.value.spent > 0.001
    assert exc_info.value.limit == 0.001


def test_budget_remaining_decreases() -> None:
    tracker = CostTracker(max_batch_cost_usd=1.0)
    tracker.record("gpt-5.2", 1500, 350)
    assert tracker.budget_remaining_usd < 1.0
    assert tracker.budget_utilization_pct > 0


def test_summary_dict() -> None:
    tracker = CostTracker(max_batch_cost_usd=50.0)
    tracker.record("gpt-5.2", 1500, 350)
    summary = tracker.summary()
    assert "total_cost_usd" in summary
    assert "total_prompt_tokens" in summary
    assert "total_completion_tokens" in summary
    assert "call_count" in summary
    assert "budget_remaining_usd" in summary
    assert "budget_utilization_pct" in summary
    assert summary["call_count"] == 1
    assert summary["total_prompt_tokens"] == 1500


def test_unknown_model_uses_default_pricing() -> None:
    tracker = CostTracker()
    entry = tracker.record("unknown-model-v1", 1000, 500)
    # Default: input=$10/1M, output=$50/1M
    expected = (1000 * 10.0 + 500 * 50.0) / 1_000_000
    assert abs(entry.cost_usd - expected) < 1e-6


def test_per_game_cost_spec() -> None:
    """SOT: ~$0.015/game (3x GPT-5.2 passes, ~1500 input + ~350 output each)."""
    tracker = CostTracker()
    for _ in range(3):  # K_PASSES = 3
        tracker.record("gpt-5.2", prompt_tokens=1500, completion_tokens=350)
    per_game = tracker.total_cost_usd
    # Spec says ~$0.015/game; verify within reasonable range
    assert 0.01 < per_game < 0.05, f"Per-game cost ${per_game:.4f} outside expected range"


def test_model_pricing_keys() -> None:
    assert "gpt-5.2" in MODEL_PRICING
    assert "claude-opus-4-6" in MODEL_PRICING
    for model, prices in MODEL_PRICING.items():
        assert "input" in prices, f"{model} missing input pricing"
        assert "output" in prices, f"{model} missing output pricing"
