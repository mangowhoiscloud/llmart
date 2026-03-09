"""Tests for regime monitor."""

from __future__ import annotations

from llmart.monitoring.regime import RegimeMonitor, RegimeStatus


def test_regime_status_no_alerts() -> None:
    status = RegimeStatus()
    assert not status.has_alerts


def test_regime_status_with_alerts() -> None:
    status = RegimeStatus(loop1_alerts=["PSI high"])
    assert status.has_alerts


def test_loop1_psi_alert() -> None:
    monitor = RegimeMonitor()
    alerts = monitor.check_loop1(psi=0.30)
    assert len(alerts) == 1
    assert "PSI" in alerts[0]


def test_loop1_genre_bias_alert() -> None:
    monitor = RegimeMonitor()
    alerts = monitor.check_loop1(genre_dist={"Roguelike": 0.50, "RPG": 0.50})
    assert len(alerts) == 1
    assert "Genre bias" in alerts[0]


def test_loop1_weight_shift_alert() -> None:
    monitor = RegimeMonitor()
    alerts = monitor.check_loop1(weight_shift=0.15)
    assert len(alerts) == 1
    assert "Weight shift" in alerts[0]


def test_loop1_escalation_rate_alert() -> None:
    monitor = RegimeMonitor()
    alerts = monitor.check_loop1(escalation_rate=0.40)
    assert len(alerts) == 1
    assert "Escalation rate" in alerts[0]


def test_loop1_no_alerts() -> None:
    monitor = RegimeMonitor()
    alerts = monitor.check_loop1(psi=0.05, genre_dist={"Roguelike": 0.30, "RPG": 0.30, "FPS": 0.40})
    assert alerts == []


def test_loop2_migration_alert() -> None:
    monitor = RegimeMonitor()
    alerts = monitor.check_loop2(migration_rate=0.35)
    assert len(alerts) == 1
    assert "migration" in alerts[0].lower()


def test_loop2_rho_alert() -> None:
    monitor = RegimeMonitor()
    alerts = monitor.check_loop2(rho=0.40)
    assert len(alerts) == 1
    assert "rho" in alerts[0].lower()


def test_loop2_hit_auc_alert() -> None:
    monitor = RegimeMonitor()
    alerts = monitor.check_loop2(hit_auc=0.55)
    assert len(alerts) == 1
    assert "AUC" in alerts[0]


def test_full_check_combined() -> None:
    monitor = RegimeMonitor()
    status = monitor.full_check(psi=0.30, rho=0.40)
    assert len(status.loop1_alerts) >= 1
    assert len(status.loop2_alerts) >= 1
