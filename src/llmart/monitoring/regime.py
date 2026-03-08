"""Regime Monitor: Loop 1 (real-time) + Loop 2 (delayed, 90d+) alerts."""

from __future__ import annotations

from dataclasses import dataclass, field

from llmart.config import DEFAULT_CONFIG, LLMARTConfig


@dataclass
class RegimeStatus:
    """Result of regime monitoring checks."""

    loop1_alerts: list[str] = field(default_factory=list)
    loop2_alerts: list[str] = field(default_factory=list)

    @property
    def has_alerts(self) -> bool:
        return bool(self.loop1_alerts or self.loop2_alerts)


class RegimeMonitor:
    """Monitor for distribution drift and model degradation.

    Loop 1 (real-time, no GT required):
        - PSI > 0.25 (score distribution drift)
        - Genre bias > 40% (single genre dominance)
        - Weight shift > 0.10 (w_ml/w_llm instability)
        - Escalation rate > 30%

    Loop 2 (delayed, requires 90d+ ground truth):
        - Tier migration > 30% (label instability)
        - Spearman rho < 0.50 (rank correlation decay)
        - Hit AUC < 0.65 (discrimination loss)
    """

    def __init__(self, config: LLMARTConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG

    def check_loop1(
        self,
        psi: float = 0.0,
        genre_dist: dict[str, float] | None = None,
        weight_shift: float = 0.0,
        escalation_rate: float = 0.0,
    ) -> list[str]:
        """Real-time alerts (no ground truth needed)."""
        alerts: list[str] = []

        if psi > self.config.psi_warning:
            alerts.append(f"PSI={psi:.3f} > {self.config.psi_warning} (score drift)")

        if genre_dist:
            max_pct = max(genre_dist.values()) if genre_dist else 0.0
            if max_pct > self.config.regime_genre_bias_max:
                top_genre = max(genre_dist, key=lambda g: genre_dist[g])
                bias_max = self.config.regime_genre_bias_max
                alerts.append(f"Genre bias: {top_genre}={max_pct:.1%} > {bias_max:.0%}")

        if weight_shift > self.config.regime_weight_shift_max:
            alerts.append(
                f"Weight shift={weight_shift:.3f} > {self.config.regime_weight_shift_max}"
            )

        if escalation_rate > self.config.regime_escalation_rate_max:
            esc_max = self.config.regime_escalation_rate_max
            alerts.append(f"Escalation rate={escalation_rate:.1%} > {esc_max:.0%}")

        return alerts

    def check_loop2(
        self,
        migration_rate: float = 0.0,
        rho: float = 1.0,
        hit_auc: float = 1.0,
    ) -> list[str]:
        """Delayed alerts (requires 90d+ ground truth)."""
        alerts: list[str] = []

        if migration_rate > self.config.regime_migration_rate_max:
            alerts.append(
                f"Tier migration={migration_rate:.1%} > {self.config.regime_migration_rate_max:.0%}"
            )

        if rho < self.config.regime_rho_min:
            alerts.append(f"Spearman rho={rho:.3f} < {self.config.regime_rho_min}")

        if hit_auc < self.config.regime_hit_auc_min:
            alerts.append(f"Hit AUC={hit_auc:.3f} < {self.config.regime_hit_auc_min}")

        return alerts

    def full_check(
        self,
        psi: float = 0.0,
        genre_dist: dict[str, float] | None = None,
        weight_shift: float = 0.0,
        escalation_rate: float = 0.0,
        migration_rate: float = 0.0,
        rho: float = 1.0,
        hit_auc: float = 1.0,
    ) -> RegimeStatus:
        """Run both loops and return combined status."""
        return RegimeStatus(
            loop1_alerts=self.check_loop1(psi, genre_dist, weight_shift, escalation_rate),
            loop2_alerts=self.check_loop2(migration_rate, rho, hit_auc),
        )
