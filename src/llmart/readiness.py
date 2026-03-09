"""Pipeline readiness check: API key detection and capability reporting.

Patterns:
- Geode ReadinessReport: detect keys → force_dry_run if absent
- OpenClaw Gateway Startup: capability enumeration before run
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum


class Capability(StrEnum):
    """Individual pipeline capabilities that require external resources."""

    OPENAI_PRIMARY = "openai_primary"
    ANTHROPIC_CALIBRATOR = "anthropic_calibrator"
    MOCK_MODE = "mock_mode"


@dataclass
class ReadinessReport:
    """Result of pre-flight readiness check.

    Geode pattern: enumerate capabilities → determine available mode → guide user.
    """

    has_openai_key: bool = False
    has_anthropic_key: bool = False
    capabilities: list[Capability] = field(default_factory=list)
    force_mock: bool = False
    guidance: list[str] = field(default_factory=list)

    @property
    def can_real_mode(self) -> bool:
        """Both API keys present → full real mode available."""
        return self.has_openai_key and self.has_anthropic_key

    @property
    def can_partial_real(self) -> bool:
        """At least one API key → partial real mode (primary only, no calibrator)."""
        return self.has_openai_key and not self.has_anthropic_key

    @property
    def available_mode(self) -> str:
        """Best available mode given current capabilities."""
        if self.can_real_mode:
            return "real"
        if self.can_partial_real:
            return "real_no_calibrator"
        return "mock"


def check_readiness(requested_mode: str = "mock") -> ReadinessReport:
    """Run pre-flight readiness check.

    Detects API keys from environment variables and determines
    what the pipeline can actually do. Returns guidance for the user
    when requested mode cannot be fulfilled.

    OpenClaw pattern: Auth Profile detection before pipeline start.
    Geode pattern: force_dry_run when keys absent.
    """
    report = ReadinessReport()

    # Detect API keys
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    report.has_openai_key = bool(openai_key and not openai_key.startswith("sk-placeholder"))
    report.has_anthropic_key = bool(
        anthropic_key and not anthropic_key.startswith("sk-ant-placeholder")
    )

    # Enumerate capabilities
    report.capabilities.append(Capability.MOCK_MODE)  # always available
    if report.has_openai_key:
        report.capabilities.append(Capability.OPENAI_PRIMARY)
    if report.has_anthropic_key:
        report.capabilities.append(Capability.ANTHROPIC_CALIBRATOR)

    # Determine if mock is forced
    if requested_mode == "real" and not report.can_real_mode:
        report.force_mock = True

        if not report.has_openai_key and not report.has_anthropic_key:
            report.guidance.append("No API keys detected. Falling back to mock mode.")
            report.guidance.append("To enable real mode, set environment variables:")
            report.guidance.append(
                "  export OPENAI_API_KEY='sk-...'       # GPT-4o Primary (T2 pass^3)"
            )
            report.guidance.append(
                "  export ANTHROPIC_API_KEY='sk-ant-...' # Opus Calibrator (disagreement)"
            )
        elif not report.has_openai_key:
            report.guidance.append("OPENAI_API_KEY missing. GPT-4o Primary requires this key.")
            report.guidance.append("  export OPENAI_API_KEY='sk-...'")
            report.guidance.append(
                "Falling back to mock mode (Anthropic key alone is insufficient for Primary)."
            )
        else:  # not report.has_anthropic_key — exhaustive: openai+anthropic combos covered above
            report.guidance.append("ANTHROPIC_API_KEY missing. Opus Calibrator will be skipped.")
            report.guidance.append("  export ANTHROPIC_API_KEY='sk-ant-...'")
            report.guidance.append(
                "Running in partial real mode: GPT-4o Primary only, no calibration."
            )
            report.force_mock = False  # can still run primary

    return report
