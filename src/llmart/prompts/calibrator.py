"""Opus-4.5 Calibrator prompt for score alignment."""

from __future__ import annotations

CALIBRATOR_SYSTEM = """\
You are a calibration reviewer. Given an LLM judge's scoring output,
verify internal consistency, check for anchoring bias, and produce
a calibration offset (delta_cal) in [-0.15, +0.15].
"""

CALIBRATOR_USER = """\
Original scores: {scores_json}
Game context: {game_context}

Assess whether the scores exhibit any systematic bias.
Return JSON: {{"delta_cal": float, "reasoning": str}}
"""
