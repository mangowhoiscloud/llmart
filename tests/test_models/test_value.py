"""Tests for ValueInference model."""

from __future__ import annotations

import pytest

from llmart.models.value import GREEN_THRESHOLD, Signal, ValueInference


def test_value_computation() -> None:
    vi = ValueInference(npv_3y=500_000, cac=50_000, ua=30_000, live_ops=20_000, var_5pct=100_000)
    # value = 500K - 50K - 30K - 20K - 0.2*100K = 380K
    assert abs(vi.value - 380_000) < 1e-9


def test_signal_green() -> None:
    vi = ValueInference(npv_3y=1_000_000, q1_p25=GREEN_THRESHOLD, q1_p50=400_000)
    assert vi.signal == Signal.GREEN


def test_signal_yellow() -> None:
    vi = ValueInference(npv_3y=500_000, q1_p25=200_000, q1_p50=GREEN_THRESHOLD)
    assert vi.signal == Signal.YELLOW


def test_signal_red() -> None:
    vi = ValueInference(npv_3y=100_000, q1_p25=50_000, q1_p50=100_000)
    assert vi.signal == Signal.RED


@pytest.mark.parametrize(
    ("p25", "p50", "expected"),
    [
        (250_000, 300_000, Signal.GREEN),
        (249_999, 250_000, Signal.YELLOW),
        (100_000, 249_999, Signal.RED),
    ],
)
def test_signal_boundary_values(p25: float, p50: float, expected: Signal) -> None:
    vi = ValueInference(npv_3y=500_000, q1_p25=p25, q1_p50=p50)
    assert vi.signal == expected
