"""Tests for API key readiness check and graceful degradation."""

from __future__ import annotations

import os
from unittest.mock import patch

from llmart.readiness import Capability, ReadinessReport, check_readiness


class TestReadinessReport:
    """ReadinessReport dataclass properties."""

    def test_no_keys(self) -> None:
        report = ReadinessReport()
        assert report.can_real_mode is False
        assert report.can_partial_real is False
        assert report.available_mode == "mock"

    def test_openai_only(self) -> None:
        report = ReadinessReport(has_openai_key=True)
        assert report.can_real_mode is False
        assert report.can_partial_real is True
        assert report.available_mode == "real_no_calibrator"

    def test_anthropic_only(self) -> None:
        report = ReadinessReport(has_anthropic_key=True)
        assert report.can_real_mode is False
        assert report.can_partial_real is False
        assert report.available_mode == "mock"

    def test_both_keys(self) -> None:
        report = ReadinessReport(has_openai_key=True, has_anthropic_key=True)
        assert report.can_real_mode is True
        assert report.available_mode == "real"


class TestCheckReadiness:
    """check_readiness() with env var mocking."""

    def test_mock_mode_no_guidance(self) -> None:
        """Mock mode requested → no guidance needed regardless of keys."""
        with patch.dict(os.environ, {}, clear=True):
            report = check_readiness("mock")
        assert report.force_mock is False
        assert report.guidance == []
        assert Capability.MOCK_MODE in report.capabilities

    def test_real_mode_no_keys(self) -> None:
        """Real mode requested but no keys → force_mock + guidance."""
        env = {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}
        with patch.dict(os.environ, env, clear=True):
            report = check_readiness("real")
        assert report.force_mock is True
        assert report.available_mode == "mock"
        assert len(report.guidance) >= 2
        assert any("No API keys" in g for g in report.guidance)

    def test_real_mode_openai_only(self) -> None:
        """Real mode with only OpenAI → partial real, guidance about Anthropic."""
        env = {"OPENAI_API_KEY": "sk-real-key", "ANTHROPIC_API_KEY": ""}
        with patch.dict(os.environ, env, clear=True):
            report = check_readiness("real")
        assert report.force_mock is False
        assert report.available_mode == "real_no_calibrator"
        assert Capability.OPENAI_PRIMARY in report.capabilities
        assert Capability.ANTHROPIC_CALIBRATOR not in report.capabilities
        assert any("ANTHROPIC_API_KEY" in g for g in report.guidance)

    def test_real_mode_both_keys(self) -> None:
        """Real mode with both keys → full capability, no guidance."""
        env = {"OPENAI_API_KEY": "sk-real-key", "ANTHROPIC_API_KEY": "sk-ant-real-key"}
        with patch.dict(os.environ, env, clear=True):
            report = check_readiness("real")
        assert report.force_mock is False
        assert report.available_mode == "real"
        assert report.guidance == []
        assert Capability.OPENAI_PRIMARY in report.capabilities
        assert Capability.ANTHROPIC_CALIBRATOR in report.capabilities

    def test_placeholder_keys_ignored(self) -> None:
        """Placeholder keys should be treated as absent."""
        env = {
            "OPENAI_API_KEY": "sk-placeholder-test",
            "ANTHROPIC_API_KEY": "sk-ant-placeholder-test",
        }
        with patch.dict(os.environ, env, clear=True):
            report = check_readiness("real")
        assert report.has_openai_key is False
        assert report.has_anthropic_key is False
        assert report.force_mock is True

    def test_anthropic_only_force_mock(self) -> None:
        """Only Anthropic key → still force_mock (Primary requires OpenAI)."""
        env = {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "sk-ant-real-key"}
        with patch.dict(os.environ, env, clear=True):
            report = check_readiness("real")
        assert report.force_mock is True
        assert any("OPENAI_API_KEY" in g for g in report.guidance)

    def test_capabilities_always_include_mock(self) -> None:
        """MOCK_MODE capability is always present."""
        with patch.dict(os.environ, {}, clear=True):
            report = check_readiness("mock")
        assert Capability.MOCK_MODE in report.capabilities


class TestLLMJudgeResolveMode:
    """Test _resolve_mode in llm_judge.py."""

    def test_mock_requested_returns_mock(self) -> None:
        from llmart.pipeline.nodes.llm_judge import _resolve_mode

        mode, msgs = _resolve_mode("mock")
        assert mode == "mock"
        assert msgs == []

    def test_real_no_keys_returns_mock(self) -> None:
        from llmart.pipeline.nodes.llm_judge import _resolve_mode

        env = {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}
        with patch.dict(os.environ, env, clear=True):
            mode, msgs = _resolve_mode("real")
        assert mode == "mock"
        assert len(msgs) > 0

    def test_real_openai_only_returns_partial(self) -> None:
        from llmart.pipeline.nodes.llm_judge import _resolve_mode

        env = {"OPENAI_API_KEY": "sk-real", "ANTHROPIC_API_KEY": ""}
        with patch.dict(os.environ, env, clear=True):
            mode, _msgs = _resolve_mode("real")
        assert mode == "real_no_calibrator"

    def test_real_both_keys_returns_real(self) -> None:
        from llmart.pipeline.nodes.llm_judge import _resolve_mode

        env = {"OPENAI_API_KEY": "sk-real", "ANTHROPIC_API_KEY": "sk-ant-real"}
        with patch.dict(os.environ, env, clear=True):
            mode, msgs = _resolve_mode("real")
        assert mode == "real"
        assert msgs == []
