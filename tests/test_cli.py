"""Tests for CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from llmart.cli import (
    _fmt_money,
    _print_monitoring,
    _print_verbose,
    app,
    quarter_scores_to_per_game,
)

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "llmart" in result.output


def test_run_mock() -> None:
    result = runner.invoke(app, ["run", "--mode", "mock", "--top-k", "10"])
    assert result.exit_code == 0
    assert "LLMART Pipeline" in result.output


def test_run_verbose() -> None:
    result = runner.invoke(app, ["run", "--mode", "mock", "--verbose", "--top-k", "10"])
    assert result.exit_code == 0
    assert "Game Cards" in result.output


def test_run_with_monitor() -> None:
    result = runner.invoke(app, ["run", "--mode", "mock", "--monitor", "--top-k", "10"])
    assert result.exit_code == 0
    assert "Monitoring" in result.output


def test_run_with_checkpoint() -> None:
    result = runner.invoke(app, ["run", "--mode", "mock", "--checkpoint", "--top-k", "10"])
    assert result.exit_code == 0
    assert "Thread ID" in result.output


def test_run_with_resume() -> None:
    result = runner.invoke(
        app, ["run", "--mode", "mock", "--checkpoint", "--resume", "test-thread", "--top-k", "10"]
    )
    assert result.exit_code == 0
    assert "Resume" in result.output


def test_run_with_phase() -> None:
    result = runner.invoke(app, ["run", "--mode", "mock", "--phase", "1", "--top-k", "10"])
    assert result.exit_code == 0


def test_run_real_mode_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real mode without API keys should fall back to mock."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["run", "--mode", "real", "--top-k", "10"])
    assert result.exit_code == 0
    assert "Readiness" in result.output or "mock" in result.output.lower()


class TestFmtMoney:
    def test_millions(self) -> None:
        assert _fmt_money(3_727_114.0) == "$3.7M"

    def test_thousands(self) -> None:
        assert _fmt_money(293_412.0) == "$293K"

    def test_small(self) -> None:
        assert _fmt_money(500.0) == "$500"

    def test_zero(self) -> None:
        assert _fmt_money(0.0) == "$0"

    def test_negative_millions(self) -> None:
        assert _fmt_money(-5_000_000.0) == "$-5.0M"


def test_validate_command() -> None:
    """Test the validate CLI command."""
    result = runner.invoke(app, ["validate", "--top-k", "10"])
    # Will succeed if training_games.json exists, fail otherwise
    if result.exit_code == 0:
        assert "Validation" in result.output or "Tier" in result.output
    else:
        assert "not found" in result.output


def test_print_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    """Test _print_verbose with sample data."""
    candidates: list[dict[str, object]] = [
        {
            "title": "Test Game",
            "dim_scores": {"gameplay": 3.0, "innovation": 2.5},
            "jury_score": 2.8,
            "delta_cal": 0.01,
            "pass3_agree": True,
            "ml_percentile": 0.75,
            "escalated": True,
            "escalation_reasons": ["DISAGREE"],
            "q1_p50": 300_000,
            "q1_p25": 200_000,
            "phase": 0,
            "w_ml": 0.6,
            "w_llm": 0.4,
        }
    ]
    _print_verbose(candidates)
    # Just verify it doesn't crash — output goes through Rich console


def test_print_verbose_with_escalation(capsys: pytest.CaptureFixture[str]) -> None:
    """Test _print_verbose with escalation reasons."""
    candidates: list[dict[str, object]] = [
        {
            "title": "Escalated Game",
            "dim_scores": {"gameplay": 4.0, "innovation": 1.0, "monetization": 3.0},
            "jury_score": 3.0,
            "delta_cal": 0.05,
            "pass3_agree": False,
            "ml_percentile": "0.85",  # string to test str conversion
            "escalated": True,
            "escalation_reasons": ["DISAGREE", "CONTRADICTION"],
            "q1_p50": 500_000,
            "q1_p25": 300_000,
            "phase": 1,
            "w_ml": 0.55,
            "w_llm": 0.45,
        }
    ]
    _print_verbose(candidates)


def test_print_verbose_no_dim_scores(capsys: pytest.CaptureFixture[str]) -> None:
    """Test _print_verbose when dim_scores is not a dict."""
    candidates: list[dict[str, object]] = [
        {
            "title": "No Dims",
            "jury_score": 2.0,
            "delta_cal": 0.0,
            "pass3_agree": True,
            "ml_percentile": 0.5,
            "escalated": False,
            "q1_p50": 100_000,
            "q1_p25": 60_000,
            "phase": 0,
            "w_ml": 0.6,
            "w_llm": 0.4,
        }
    ]
    _print_verbose(candidates)


def test_print_monitoring_no_alerts(capsys: pytest.CaptureFixture[str]) -> None:
    """Test _print_monitoring with no loop alerts."""
    monitoring: dict[str, object] = {
        "psi": 0.02,
        "psi_status": "stable",
        "escalation_rate": 0.05,
        "weight_shift": 0.01,
        "regime_loop1_alerts": [],
        "regime_loop2_alerts": [],
        "genre_distribution": {},
    }
    _print_monitoring(monitoring)


def test_print_monitoring(capsys: pytest.CaptureFixture[str]) -> None:
    """Test _print_monitoring with sample data."""
    monitoring: dict[str, object] = {
        "psi": 0.05,
        "psi_status": "stable",
        "escalation_rate": 0.10,
        "weight_shift": 0.02,
        "regime_loop1_alerts": ["high_psi"],
        "regime_loop2_alerts": ["genre_concentration"],
        "genre_distribution": {"Roguelike": 0.5, "RPG": 0.5},
    }
    _print_monitoring(monitoring)
    # Just verify it doesn't crash — output goes through Rich console


def test_print_monitoring_nondict_genre(capsys: pytest.CaptureFixture[str]) -> None:
    """genre_distribution that is not a dict should skip genre display."""
    monitoring: dict[str, object] = {
        "psi": 0.02,
        "psi_status": "stable",
        "escalation_rate": 0.05,
        "weight_shift": 0.01,
        "regime_loop1_alerts": [],
        "regime_loop2_alerts": [],
        "genre_distribution": "not_a_dict",
    }
    _print_monitoring(monitoring)


def test_run_nonlist_json(tmp_path: Path) -> None:
    """Non-list JSON should error out."""
    f = tmp_path / "bad.json"
    f.write_text('{"not": "a list"}')
    result = runner.invoke(app, ["run", "--games", str(f)])
    assert result.exit_code == 1
    assert "must be a list" in result.output


def test_run_invalid_candidates(tmp_path: Path) -> None:
    """List of non-dicts should fail PipelineState validation."""
    f = tmp_path / "int_list.json"
    f.write_text("[1, 2, 3]")
    result = runner.invoke(app, ["run", "--games", str(f)])
    assert result.exit_code == 1
    assert "validation" in result.output.lower() or "Input" in result.output


def test_run_with_checkpoint_path(tmp_path: Path) -> None:
    """SqliteSaver checkpoint with file path."""
    db_path = tmp_path / "test.db"
    with (
        patch("langgraph.checkpoint.sqlite.SqliteSaver") as mock_cls,
        patch("llmart.pipeline.graph.create_llmart_graph") as mock_create,
    ):
        mock_cls.from_conn_string.return_value = MagicMock()
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter(
            [
                {
                    "value": {
                        "candidates": [
                            {
                                "title": "X",
                                "signal": "GREEN",
                                "genre": "RPG",
                                "selection_score": 0.5,
                                "decision": "APPROVE",
                                "npv_3y": 100_000,
                                "value_total": 80_000,
                            }
                        ],
                        "errors": [],
                    }
                },
            ]
        )
        mock_create.return_value = mock_graph
        result = runner.invoke(
            app,
            [
                "run",
                "--mode",
                "mock",
                "--checkpoint",
                "--checkpoint-path",
                str(db_path),
                "--top-k",
                "10",
            ],
        )
    assert result.exit_code == 0


def test_run_pipeline_exception() -> None:
    """Pipeline exception should be caught and reported."""
    with patch("llmart.pipeline.graph.create_llmart_graph") as mock_create:
        mock_graph = MagicMock()
        mock_graph.stream.side_effect = RuntimeError("boom")
        mock_create.return_value = mock_graph
        result = runner.invoke(app, ["run", "--mode", "mock", "--top-k", "10"])
    assert result.exit_code == 1
    assert "Pipeline failed" in result.output


def test_run_no_candidates_survived() -> None:
    """Empty candidates after pipeline should show warning."""
    with patch("llmart.pipeline.graph.create_llmart_graph") as mock_create:
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter(
            [
                {"value": {"candidates": [], "errors": []}},
            ]
        )
        mock_create.return_value = mock_graph
        result = runner.invoke(app, ["run", "--mode", "mock", "--top-k", "10"])
    assert result.exit_code == 0
    assert "No candidates" in result.output


def test_run_with_pipeline_errors() -> None:
    """Pipeline errors should be printed."""
    with patch("llmart.pipeline.graph.create_llmart_graph") as mock_create:
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter(
            [
                {
                    "value": {
                        "candidates": [
                            {
                                "title": "X",
                                "signal": "GREEN",
                                "genre": "RPG",
                                "selection_score": 0.5,
                                "decision": "APPROVE",
                                "npv_3y": 100_000,
                                "value_total": 80_000,
                            }
                        ],
                        "errors": ["test error alpha", "test error beta"],
                    }
                },
            ]
        )
        mock_create.return_value = mock_graph
        result = runner.invoke(app, ["run", "--mode", "mock", "--top-k", "10"])
    assert result.exit_code == 0
    assert "test error alpha" in result.output


def test_run_real_partial_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real mode with only OpenAI key → no_calibrator message."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-real-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("llmart.pipeline.graph.create_llmart_graph") as mock_create:
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter(
            [
                {
                    "value": {
                        "candidates": [
                            {
                                "title": "X",
                                "signal": "GREEN",
                                "genre": "RPG",
                                "selection_score": 0.5,
                                "decision": "APPROVE",
                                "npv_3y": 100_000,
                                "value_total": 80_000,
                            }
                        ],
                        "errors": [],
                    }
                },
            ]
        )
        mock_create.return_value = mock_graph
        result = runner.invoke(app, ["run", "--mode", "real", "--top-k", "10"])
    assert "calibrator" in result.output.lower()


def test_run_real_both_keys_no_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real mode with both keys → no readiness guidance panel."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-real-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-real-key")
    with patch("llmart.pipeline.graph.create_llmart_graph") as mock_create:
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter(
            [
                {
                    "value": {
                        "candidates": [
                            {
                                "title": "X",
                                "signal": "GREEN",
                                "genre": "RPG",
                                "selection_score": 0.5,
                                "decision": "APPROVE",
                                "npv_3y": 100_000,
                                "value_total": 80_000,
                            }
                        ],
                        "errors": [],
                    }
                },
            ]
        )
        mock_create.return_value = mock_graph
        result = runner.invoke(app, ["run", "--mode", "real", "--top-k", "10"])
    assert result.exit_code == 0
    assert "Readiness" not in result.output


def test_validate_not_found(tmp_path: Path) -> None:
    """Validate with non-existent data file."""
    result = runner.invoke(app, ["validate", "--data", str(tmp_path / "nonexistent.json")])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_validate_nonlist(tmp_path: Path) -> None:
    """Validate with non-list JSON."""
    f = tmp_path / "dict.json"
    f.write_text('{"key": "value"}')
    result = runner.invoke(app, ["validate", "--data", str(f)])
    assert result.exit_code == 1
    assert "must be a list" in result.output


def test_validate_qr_unfittable(tmp_path: Path) -> None:
    """Validate where QuantileRegressor fails to fit (n < 5)."""
    data = [
        {
            "game_id": f"G{i}",
            "steam_rating": 0.5,
            "review_count": 100,
            "price_usd": 10.0,
            "release_year": 2024,
            "tags": ["RPG"],
            "estimated_y1_revenue": 0.0,
            "genre_q1_type": "balanced",
            "hit_tier": "Hobby",
        }
        for i in range(3)
    ]
    f = tmp_path / "few_training.json"
    f.write_text(json.dumps(data))
    result = runner.invoke(app, ["validate", "--data", str(f)])
    assert result.exit_code == 0
    assert "failed to fit" in result.output.lower() or "Tier" in result.output


def test_retrain_command() -> None:
    """Retrain command with small synthetic data should complete."""
    result = runner.invoke(app, ["retrain", "--n-games", "30", "--seed", "42"])
    assert result.exit_code == 0
    assert "Quarterly Retrain Cycle" in result.output
    assert "Ridge Weight Learning" in result.output
    assert "Phase Transition History" in result.output
    assert "PSI" in result.output


def test_retrain_small_n() -> None:
    """Retrain with very few games should still run."""
    result = runner.invoke(app, ["retrain", "--n-games", "10", "--seed", "99"])
    assert result.exit_code == 0


def test_quarter_scores_to_per_game() -> None:
    """Helper should map quarter scores back to per-game."""
    games = [
        {"quarter": "Q1"},
        {"quarter": "Q1"},
        {"quarter": "Q2"},
    ]
    quarter_scores = {
        "Q1": {"ml_scores": [0.5, 0.6], "jury_scores": [2.0, 3.0]},
        "Q2": {"ml_scores": [0.7], "jury_scores": [3.5]},
    }
    result = quarter_scores_to_per_game(games, quarter_scores)
    assert len(result) == 3
    assert result[0] == {"ml": 0.5, "jury": 2.0}
    assert result[1] == {"ml": 0.6, "jury": 3.0}
    assert result[2] == {"ml": 0.7, "jury": 3.5}


def test_quarter_scores_to_per_game_missing_quarter() -> None:
    """Missing quarter should use defaults."""
    games = [{"quarter": "Q99"}]
    quarter_scores: dict[str, dict[str, list[float]]] = {}
    result = quarter_scores_to_per_game(games, quarter_scores)
    assert result[0] == {"ml": 0.5, "jury": 2.0}


def test_quarter_scores_to_per_game_overflow() -> None:
    """Index exceeding available scores should use default."""
    games = [{"quarter": "Q1"}, {"quarter": "Q1"}]
    quarter_scores = {
        "Q1": {"ml_scores": [0.8], "jury_scores": [3.0]},
    }
    result = quarter_scores_to_per_game(games, quarter_scores)
    assert result[0] == {"ml": 0.8, "jury": 3.0}
    assert result[1] == {"ml": 0.5, "jury": 2.0}


def test_run_with_training_data(tmp_path: Path) -> None:
    """Pipeline with --training-data should pass data to value node."""
    training = [
        {
            "game_id": f"T{i}",
            "ml_percentile": 0.5,
            "jury_score": 2.5,
            "q1_p50": 100_000.0,
            "genre_q1_type": "balanced",
            "estimated_y1_revenue": 200_000.0,
        }
        for i in range(20)
    ]
    f = tmp_path / "training.json"
    f.write_text(json.dumps(training))
    result = runner.invoke(
        app,
        ["run", "--mode", "mock", "--phase", "2", "--training-data", str(f), "--top-k", "5"],
    )
    assert result.exit_code == 0
    assert "Training data: 20 games" in result.output


def test_run_with_nonexistent_training_data() -> None:
    """Non-existent training data path should be silently ignored."""
    result = runner.invoke(
        app,
        [
            "run",
            "--mode",
            "mock",
            "--training-data",
            "/tmp/nonexistent_td_llmart.json",  # noqa: S108
            "--top-k",
            "5",
        ],
    )
    assert result.exit_code == 0


def test_run_with_dict_training_data(tmp_path: Path) -> None:
    """Dict (non-list) training data JSON should be silently skipped."""
    f = tmp_path / "td_dict.json"
    f.write_text('{"not": "a list"}')
    result = runner.invoke(
        app,
        ["run", "--mode", "mock", "--training-data", str(f), "--top-k", "5"],
    )
    assert result.exit_code == 0
    assert "Training data:" not in result.output


def test_retrain_very_small_n_qr_fails() -> None:
    """Retrain with 3 games: QR fails to fit, fallback branch hit."""
    result = runner.invoke(app, ["retrain", "--n-games", "3", "--seed", "42"])
    assert result.exit_code == 0
    assert "failed to fit" in result.output.lower() or "Quarterly" in result.output


def test_retrain_validation_failed() -> None:
    """Retrain where validation FAILS (mock check_pass_condition -> False)."""
    with patch("llmart.monitoring.metrics.check_pass_condition", return_value=False):
        result = runner.invoke(app, ["retrain", "--n-games", "30", "--seed", "42"])
    assert result.exit_code == 0
    assert "FAILED" in result.output
