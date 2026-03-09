"""LLMART CLI entry point."""

from __future__ import annotations

import json
import uuid
import warnings
from importlib import resources
from pathlib import Path
from typing import Any

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmart.audit import AuditStore, RunRecord
from llmart.logging_config import generate_run_id, get_pipeline_logger

# Suppress LangChain Pydantic V1 compatibility warning on Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

app = typer.Typer(
    name="llmart",
    help="LLMART — Game Selection & Value Inference Pipeline",
    no_args_is_help=True,
)
console = Console()
logger = get_pipeline_logger("llmart.cli", json_output=False)

SIGNAL_COLORS = {"GREEN": "green", "YELLOW": "yellow", "RED": "red"}

# Pipeline stages for progress display
STAGES = [
    ("prefilter", "Pre-filter 2L"),
    ("ml_scoring", "T1 ML Scoring"),
    ("llm_judge", "T2 LLM-as-Judge"),
    ("enrichment", "Data Enrichment"),
    ("human_review", "T3 Human Review"),
    ("value", "Value Inference"),
]


def _fmt_money(n: float) -> str:
    """Format dollar amount: $3.7M, $293K, $500."""
    if abs(n) >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"${n / 1_000:.0f}K"
    return f"${n:,.0f}"


def _load_json(path: Path) -> list[dict[str, object]] | dict[str, object]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def _bundled_path(filename: str) -> Path:
    """Resolve a path inside the bundled data package."""
    return Path(str(resources.files("llmart") / "data" / filename))


@app.command()
def run(
    games: Path | None = typer.Option(  # noqa: B008
        None, "--games", "-g", help="Path to games JSON (default: bundled sample)"
    ),
    mode: str = typer.Option("mock", "--mode", "-m", help="LLM mode: mock | real"),
    top_k: int = typer.Option(30, "--top-k", "-k", help="Number of final selections"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed scores"),
    checkpoint: bool = typer.Option(False, "--checkpoint", help="Enable checkpointing"),
    checkpoint_path: Path | None = typer.Option(  # noqa: B008
        None, "--checkpoint-path", help="SQLite checkpoint path"
    ),
    resume: str | None = typer.Option(None, "--resume", help="Resume from thread ID"),
    monitor: bool = typer.Option(False, "--monitor", help="Enable monitoring node"),
    phase: int = typer.Option(0, "--phase", "-p", help="Phase override (0-3)", min=0, max=3),
    training_data: Path | None = typer.Option(  # noqa: B008
        None, "--training-data", "-t", help="Training data JSON for Phase 2+ Ridge weights"
    ),
) -> None:
    """Run the full LLMART pipeline."""
    from llmart.pipeline.graph import create_llmart_graph

    # Load data
    games_path = games or _bundled_path("sample_games.json")
    genre_path = _bundled_path("genre_params.json")

    candidates = _load_json(games_path)
    genre_params = _load_json(genre_path)

    if not isinstance(candidates, list):
        console.print("[red]Error:[/] games JSON must be a list")
        raise typer.Exit(1)

    # Settings panel
    settings_lines = [
        f"[bold]Games:[/] {games_path.name}  ({len(candidates)} candidates)",
        f"[bold]Mode:[/]  {mode}",
        f"[bold]Top-K:[/] {top_k}",
        f"[bold]Phase:[/] {phase}",
    ]
    if checkpoint:
        settings_lines.append(f"[bold]Checkpoint:[/] {checkpoint_path or 'memory'}")
    if resume:
        settings_lines.append(f"[bold]Resume:[/] {resume}")
    if monitor:
        settings_lines.append("[bold]Monitoring:[/] enabled")
    console.print(
        Panel.fit(
            "\n".join(settings_lines),
            title="[bold green]LLMART Pipeline[/]",
        )
    )

    # Readiness check — Geode/OpenClaw pattern: detect keys → guide user
    if mode == "real":
        from llmart.readiness import check_readiness

        readiness = check_readiness(mode)
        if readiness.guidance:
            console.print(
                Panel.fit(
                    "\n".join(readiness.guidance),
                    title="[bold yellow]Readiness Check[/]",
                )
            )
        if readiness.force_mock:
            mode = "mock"
            console.print("[yellow]Falling back to mock mode.[/]")
        elif readiness.available_mode == "real_no_calibrator":
            console.print("[yellow]Running without calibrator (Anthropic key missing).[/]")

    # Validate input via PipelineState (entry boundary validation)
    from llmart.pipeline.state import PipelineState

    try:
        PipelineState(candidates=candidates, stage="init", top_k=top_k, errors=[])
    except Exception as exc:
        console.print(f"[red]Input validation failed:[/] {exc}")
        raise typer.Exit(1) from exc

    # Build checkpointer
    checkpointer = None
    if checkpoint and checkpoint_path:
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpointer = SqliteSaver.from_conn_string(str(checkpoint_path))
        # else: MemorySaver is default in create_llmart_graph

    # Build and run graph
    graph = create_llmart_graph(
        checkpointer=checkpointer,
        enable_monitoring=monitor,
    )
    thread_id = resume or str(uuid.uuid4())
    run_id = generate_run_id()

    # Audit: start recording
    audit = AuditStore()
    record = RunRecord(
        run_id=run_id,
        mode=mode,
        phase=phase,
        top_k=top_k,
        input_count=len(candidates),
        config={"games": str(games_path), "checkpoint": checkpoint, "monitor": monitor},
    )
    logger.info("Pipeline run started run_id=%s mode=%s", run_id, mode)

    # Load training data for Phase 2+ Ridge weight learning
    td_list: list[dict[str, object]] = []
    if training_data and training_data.exists():
        raw_td = _load_json(training_data)
        if isinstance(raw_td, list):
            td_list = raw_td
            console.print(f"[dim]Training data: {len(td_list)} games from {training_data.name}[/]")

    initial_state = {
        "candidates": candidates,
        "stage": "init",
        "top_k": top_k,
        "errors": [],
        "mode": mode,
        "genre_params": genre_params,
        "total_input": len(candidates),
        "phase": phase,
        "n_historical": len(td_list),
        "run_metadata": {"thread_id": thread_id},
        "training_data": td_list,
    }

    run_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    try:
        stage_map = dict(STAGES)
        final: dict[str, Any] = {}
        with console.status("[bold green]Starting pipeline...", spinner="dots") as status:
            for event in graph.stream(initial_state, config=run_config):  # type: ignore[arg-type]
                for node_name, node_output in event.items():
                    label = stage_map.get(node_name, node_name)
                    n = len(node_output.get("candidates", []))
                    status.update(f"[bold green]{label}[/] → {n} candidates")
                    final = {**final, **node_output}
    except Exception as exc:
        console.print(f"[red]Pipeline failed:[/] {exc}")
        raise typer.Exit(1) from exc

    # Audit: finish recording
    result_candidates = final.get("candidates", [])
    record.finish(result_candidates, final.get("errors", []))
    audit.save(record)
    audit.close()
    logger.info("Pipeline run completed run_id=%s duration=%.3fs", run_id, record.duration_s or 0)

    # Stage summary
    total_in = final.get("total_input", len(candidates))
    final_count = len(final.get("candidates", []))
    console.print(
        f"\n[dim]Pipeline complete: {total_in} candidates -> {final_count} selected"
        f" (run_id={run_id})[/]"
    )
    if checkpoint:
        console.print(f"[dim]Thread ID: {thread_id}[/]")

    # Results table — sort by Signal priority (GREEN>YELLOW>RED) then selection_score
    signal_priority = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    result_candidates = final.get("candidates", [])
    result_candidates.sort(
        key=lambda g: (
            signal_priority.get(g.get("signal", "RED"), 3),
            -g.get("selection_score", 0),
        )
    )
    if not result_candidates:
        console.print("[yellow]No candidates survived the pipeline.[/]")
        return

    table = Table(title=f"Final Results ({len(result_candidates)} games)")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Game", style="bold")
    table.add_column("Genre", min_width=12)
    table.add_column("Score", justify="right")
    table.add_column("Signal", justify="center")
    table.add_column("Decision")
    table.add_column("NPV_3Y", justify="right")
    table.add_column("Value", justify="right")

    for i, g in enumerate(result_candidates, 1):
        signal = g.get("signal", "?")
        color = SIGNAL_COLORS.get(signal, "white")
        table.add_row(
            str(i),
            g.get("title", "?"),
            g.get("genre", "?"),
            f"{g.get('selection_score', 0):.4f}",
            f"[{color}]{signal}[/{color}]",
            g.get("decision", "?"),
            _fmt_money(g.get("npv_3y", 0)),
            _fmt_money(g.get("value_total", 0)),
        )

    console.print(table)

    # Verbose: dimension scores and features
    if verbose:
        _print_verbose(result_candidates)

    # Monitoring results
    monitoring_data = final.get("monitoring")
    if monitor and isinstance(monitoring_data, dict):
        _print_monitoring(monitoring_data)

    errors = final.get("errors", [])
    if errors:
        console.print(f"\n[red]Errors ({len(errors)}):[/]")
        for err in errors:
            console.print(f"  - {err}")


_DIM_LABELS: dict[str, tuple[str, int]] = {
    "gameplay": ("Game", 30),
    "innovation": ("Innov", 20),
    "monetization": ("Money", 20),
    "polish": ("Polish", 15),
    "narrative": ("Story", 15),
}

_BAR_WIDTH = 12  # max bar width (score 4 = full bar)
_BLOCK_FULL = "\u2588"
_BLOCK_HALF = "\u2584"


def _dim_bar(score: float, max_score: float = 4.0) -> Text:
    """Render a single dimension as a coloured horizontal bar."""
    ratio = min(score / max_score, 1.0)
    filled = int(ratio * _BAR_WIDTH)
    half = ratio * _BAR_WIDTH - filled >= 0.5

    # Colour based on score tier: E(4)=green, H(3)=cyan, M(2)=yellow, L(1)=red
    if score >= 3.5:
        color = "green"
    elif score >= 2.5:
        color = "cyan"
    elif score >= 1.5:
        color = "yellow"
    else:
        color = "red"

    bar = Text()
    bar.append(_BLOCK_FULL * filled, style=color)
    if half:
        bar.append(_BLOCK_HALF, style=color)
        filled += 1
    bar.append(" " * (_BAR_WIDTH - filled), style="dim")
    bar.append(f" {score:.1f}", style="bold")
    return bar


def _game_card(g: dict[str, object], rank: int) -> Panel:
    """Build a Rich Panel game card with 5-dim bar chart + KPI sidebar."""
    title = str(g.get("title", "?"))
    genre = str(g.get("genre", "?"))
    signal = str(g.get("signal", "?"))
    decision = str(g.get("decision", "?"))

    sig_color = SIGNAL_COLORS.get(signal, "white")
    sig_icon = {"GREEN": "\u2705", "YELLOW": "\u26a0\ufe0f", "RED": "\u274c"}.get(signal, "")

    # Left side: 5-dim bar chart
    dim_scores = g.get("dim_scores", {})
    dim_table = Table.grid(padding=(0, 1))
    dim_table.add_column("dim", width=6, justify="right")
    dim_table.add_column("pct", width=3, justify="right", style="dim")
    dim_table.add_column("bar", width=_BAR_WIDTH + 5)

    if isinstance(dim_scores, dict):
        for dim_key, (label, weight) in _DIM_LABELS.items():
            score = float(dim_scores.get(dim_key, 0))
            dim_table.add_row(label, f"{weight}%", _dim_bar(score))

    # Right side: KPI metrics
    jury = g.get("jury_score", 0)
    delta = g.get("delta_cal", 0)
    delta_f = float(delta) if isinstance(delta, (int, float)) else 0.0
    agree = g.get("pass3_agree", False)
    agree_icon = "\u2713" if agree else "\u2717"
    agree_color = "green" if agree else "red"

    npv = g.get("npv_3y", 0)
    value = g.get("value_total", 0)
    npv_str = _fmt_money(float(npv)) if isinstance(npv, (int, float)) else str(npv)
    val_str = _fmt_money(float(value)) if isinstance(value, (int, float)) else str(value)

    phase = g.get("phase", 0)
    w_ml = g.get("w_ml", 0.5)
    w_llm = g.get("w_llm", 0.5)
    sel_score = g.get("selection_score", 0)

    jury_f = float(jury) if isinstance(jury, (int, float)) else 0.0
    score_f = float(sel_score) if isinstance(sel_score, (int, float)) else 0.0

    kpi_table = Table.grid(padding=(0, 1))
    kpi_table.add_column("key", width=5, justify="right", style="dim")
    kpi_table.add_column("val", width=15, justify="left")

    kpi_table.add_row("Jury", f"[bold]{jury_f:.2f}[/]/4")
    kpi_table.add_row("Score", f"[bold]{score_f:.2f}[/]")
    kpi_table.add_row("NPV3Y", f"[bold]{npv_str}[/]")
    kpi_table.add_row("Net", f"[bold]{val_str}[/]")
    kpi_table.add_row("Agree", f"[{agree_color}]{agree_icon}[/{agree_color}]")
    if abs(delta_f) > 0.001:
        cal_sign = "+" if delta_f > 0 else ""
        kpi_table.add_row("Cal", f"{cal_sign}{delta_f:.2f}")
    w_ml_f = float(w_ml) if isinstance(w_ml, (int, float)) else 0.5
    w_llm_f = float(w_llm) if isinstance(w_llm, (int, float)) else 0.5
    kpi_table.add_row("Phase", f"P{phase}")
    kpi_table.add_row("w_ml", f"{w_ml_f:.3f}")
    kpi_table.add_row("w_llm", f"{w_llm_f:.3f}")

    # Combine left + right
    layout = Columns([dim_table, kpi_table], padding=(0, 1))

    # Escalation footer
    esc_reasons = g.get("escalation_reasons")
    raw_guidance = g.get("escalation_guidance", [])
    esc_lines: list[str] = []
    if isinstance(esc_reasons, list) and esc_reasons:
        esc_guidance: list[object] = list(raw_guidance) if isinstance(raw_guidance, list) else []
        for i, reason in enumerate(esc_reasons):
            guide = str(esc_guidance[i]) if i < len(esc_guidance) else ""
            suffix = f" \u2014 {guide}" if guide else ""
            esc_lines.append(f"[yellow]\u26a0 {reason}{suffix}[/]")

    # Assemble panel content
    parts: list[Columns | Text] = [layout]
    if esc_lines:
        parts.append(Text())  # blank line
        for line in esc_lines:
            parts.append(Text.from_markup(line))

    content = Table.grid()
    content.add_column()
    for part in parts:
        content.add_row(part)

    # Panel title: rank + title + genre | signal + decision
    panel_title = (
        f"[bold]#{rank} {title}[/] [dim]({genre})[/]"
        f"  [{sig_color}]{sig_icon} {signal}[/{sig_color}]"
        f"  [dim]{decision}[/]"
    )

    return Panel(
        content,
        title=panel_title,
        border_style=sig_color,
        expand=False,
        width=65,
    )


def _print_verbose(candidates: list[dict[str, object]]) -> None:
    """Print detailed game cards with 5-dim bar charts and KPI panels."""
    console.print("\n[bold]Game Cards[/]")
    for i, g in enumerate(candidates, 1):
        console.print(_game_card(g, i))


_GAUGE_WIDTH = 16  # bar width for monitoring gauges


def _metric_bar(
    label: str,
    value: float,
    max_val: float,
    thresholds: tuple[float, float],
) -> Text:
    """Render a health-metric gauge bar with threshold colouring.

    *thresholds* = (yellow_cutoff, red_cutoff).
    """
    ratio = min(max(value / max_val, 0.0), 1.0)
    filled = int(ratio * _GAUGE_WIDTH)

    if value < thresholds[0]:
        color, icon, status = "green", "\u2713", "ok"
    elif value < thresholds[1]:
        color, icon, status = "yellow", "~", "watch"
    else:
        color, icon, status = "red", "!", "WARNING"

    bar = Text()
    bar.append(f"  {label:<8} ", style="bold")
    bar.append(_BLOCK_FULL * filled, style=color)
    bar.append("\u2591" * (_GAUGE_WIDTH - filled), style="dim")

    # Format value: percentages vs decimals
    if max_val == 100.0:
        bar.append(f"  {value:5.1f}%", style="bold")
    else:
        bar.append(f"  {value:.4f}" if value < 1.0 else f"  {value:.2f}", style="bold")
    bar.append(f"  {icon} {status}", style=color)
    return bar


def _genre_bars(genre_dist: dict[str, float], top_n: int = 10) -> Text:
    """Render a top-N horizontal bar chart for genre distribution."""
    text = Text()
    if not genre_dist:
        text.append("  [dim]\u2014 No genre data[/]")
        return text

    sorted_genres = sorted(genre_dist.items(), key=lambda x: x[1], reverse=True)
    top = sorted_genres[:top_n]
    rest = sorted_genres[top_n:]

    max_pct = top[0][1] if top else 0.01
    bar_max = 10  # max bar chars for genre chart

    for name, pct in top:
        bar_len = max(1, int((pct / max_pct) * bar_max))
        text.append(f"  {name[:20]:<20} ", style="bold")
        text.append(_BLOCK_FULL * bar_len, style="cyan")
        text.append(f" {pct:.0%}\n")

    if rest:
        avg_pct = sum(v for _, v in rest) / len(rest) if rest else 0
        text.append(f"  {'':20} +{len(rest)} others (avg {avg_pct:.0%})\n", style="dim")

    return text


def _print_monitoring(monitoring: dict[str, object]) -> None:
    """Print structured monitoring dashboard with gauges, alerts, and genre chart."""
    console.print("\n")
    candidate_count = monitoring.get("candidate_count", "?")

    parts: list[Text | str] = []

    # --- Section 1: Health Metrics ---
    def _f(key: str) -> float:
        v = monitoring.get(key, 0)
        return float(v) if isinstance(v, (int, float)) else 0.0

    psi = _f("psi")
    ece = _f("ece")
    escalation = _f("escalation_rate") * 100.0  # ratio → %
    weight_shift = _f("weight_shift")

    parts.append(_metric_bar("PSI", psi, 1.0, (0.10, 0.25)))
    parts.append(Text("\n"))
    parts.append(_metric_bar("ECE", ece, 0.50, (0.10, 0.20)))
    parts.append(Text("\n"))
    parts.append(_metric_bar("Escal", escalation, 100.0, (15.0, 30.0)))
    parts.append(Text("\n"))
    parts.append(_metric_bar("\u0394-wt", weight_shift, 0.30, (0.05, 0.10)))

    # --- Section 2: Regime Alerts ---
    parts.append(Text("\n\n"))
    parts.append(Text("  Regime Alerts ", style="bold underline"))
    parts.append(Text("\n"))

    loop1 = monitoring.get("regime_loop1_alerts", [])
    loop2 = monitoring.get("regime_loop2_alerts", [])

    parts.append(Text("  Loop 1: ", style="bold"))
    if loop1 and isinstance(loop1, list):
        parts.append(Text("\n"))
        for alert in loop1:
            parts.append(Text(f"   \u26a0 {alert}\n", style="yellow"))
    else:
        parts.append(Text("\u2014 No alerts\n", style="dim"))

    parts.append(Text("  Loop 2: ", style="bold"))
    if loop2 and isinstance(loop2, list):
        parts.append(Text("\n"))
        for alert in loop2:
            parts.append(Text(f"   \u26a0 {alert}\n", style="yellow"))
    else:
        parts.append(Text("\u2014 No alerts\n", style="dim"))

    # --- Section 3: Genre Distribution ---
    genre_dist = monitoring.get("genre_distribution", {})
    if isinstance(genre_dist, dict) and genre_dist:
        parts.append(Text("\n"))
        parts.append(Text("  Genre Distribution (top 10) ", style="bold underline"))
        parts.append(Text("\n"))
        parts.append(_genre_bars(genre_dist))

    # Assemble
    content = Text()
    for p in parts:
        if isinstance(p, str):
            content.append(p)
        else:
            content.append_text(p)

    console.print(
        Panel(
            content,
            title=f"[bold blue]Monitoring Dashboard ({candidate_count} candidates)[/]",
            border_style="blue",
            expand=False,
            width=62,
        )
    )


@app.command()
def validate(
    data: Path | None = typer.Option(  # noqa: B008
        None, "--data", "-d", help="Path to training JSON (default: bundled)"
    ),
    top_k: int = typer.Option(30, "--top-k", "-k", help="Top-K for NDCG"),
) -> None:
    """Bootstrap validation on training data — NDCG, Spearman, tier accuracy."""
    from llmart.models.ground_truth import classify_tier
    from llmart.models.quantile import QuantileRegressor, extract_features

    data_path = data or _bundled_path("training_games.json")
    if not data_path.exists():
        console.print(f"[red]Training data not found:[/] {data_path}")
        raise typer.Exit(1)

    training = _load_json(data_path)
    if not isinstance(training, list):
        console.print("[red]Training data must be a list[/]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold]Data:[/] {data_path.name}  ({len(training)} games)\n[bold]Top-K:[/] {top_k}",
            title="[bold green]Bootstrap Validation[/]",
        )
    )

    # Genre Q1 share ratios
    q1_shares: dict[str, float] = {
        "front-loaded": 0.65,
        "balanced": 0.50,
        "live-service": 0.40,
    }

    # Compute ground-truth Q1 and features
    features: list[list[float]] = []
    q1_revenues: list[float] = []
    gt_tiers: list[str] = []

    for g in training:
        feat = extract_features(g)
        features.append(feat)
        raw_y1 = g.get("estimated_y1_revenue", 0.0)
        y1 = float(raw_y1) if isinstance(raw_y1, (int, float)) else 0.0
        q1_type = str(g.get("genre_q1_type", "balanced"))
        q1 = y1 * q1_shares.get(q1_type, 0.50)
        q1_revenues.append(q1)
        raw_tier = g.get("hit_tier", "Hobby")
        gt_tiers.append(str(raw_tier) if raw_tier is not None else "Hobby")

    # Fit quantile regressor (LOOCV)
    qr = QuantileRegressor()
    qr.fit(features, q1_revenues)

    if not qr.is_fitted:
        console.print("[yellow]Quantile regressor failed to fit[/]")
    else:
        # Evaluate predictions
        correct_tiers = 0
        pred_q1_values: list[float] = []
        for i, feat in enumerate(features):
            pred = qr.predict(feat)
            pred_q1_values.append(pred.q1_p50)
            pred_tier = classify_tier(pred.q1_p50).value
            if pred_tier == gt_tiers[i]:
                correct_tiers += 1

        tier_accuracy = correct_tiers / len(training) if training else 0.0

        # Spearman rank correlation
        from llmart.phase.ridge_learner import _spearman_rho

        rho = _spearman_rho(pred_q1_values, q1_revenues)

        # Results table
        table = Table(title="Validation Metrics")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Target", justify="right", style="dim")

        table.add_row("Tier Accuracy", f"{tier_accuracy:.1%}", ">60%")
        table.add_row("Spearman rho", f"{rho:.4f}", ">0.50")
        table.add_row("Training samples", str(len(training)), "50+")

        console.print(table)

    # Tier distribution
    tier_counts: dict[str, int] = {}
    for t in gt_tiers:
        tier_counts[t] = tier_counts.get(t, 0) + 1

    dist_table = Table(title="Tier Distribution")
    dist_table.add_column("Tier", style="bold")
    dist_table.add_column("Count", justify="right")
    dist_table.add_column("Pct", justify="right")

    for tier in ["Mega", "Hit", "Side", "Hobby"]:
        count = tier_counts.get(tier, 0)
        pct = count / len(training) * 100 if training else 0
        dist_table.add_row(tier, str(count), f"{pct:.1f}%")

    console.print(dist_table)


@app.command()
def retrain(
    n_games: int = typer.Option(500, "--n-games", "-n", help="Number of synthetic games"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducibility"),
    top_k: int = typer.Option(30, "--top-k", "-k", help="Top-K for validation metrics"),
) -> None:
    """Quarterly retrain cycle: generate data, learn weights, evaluate phase transition.

    Local equivalent of the Airflow DAG (spec A.5.3):
    1. Generate synthetic training data (or load existing)
    2. Ridge LOOCV weight learning
    3. Phase transition evaluation
    4. PSI drift detection between quarters
    5. Validation suite report
    """
    import math

    from llmart.data.synthetic import (
        generate_quarter_scores,
        generate_quarterly_feedback,
        generate_training_dataset,
    )
    from llmart.models.ground_truth import classify_tier
    from llmart.models.percentile import PercentileRankManager
    from llmart.models.quantile import QuantileRegressor, extract_features
    from llmart.monitoring.metrics import compute_validation_suite
    from llmart.phase.controller import PhaseController
    from llmart.phase.ridge_learner import Phase2WeightLearner, _spearman_rho

    console.print(
        Panel.fit(
            f"[bold]Games:[/] {n_games} synthetic\n[bold]Seed:[/]  {seed}\n[bold]Top-K:[/] {top_k}",
            title="[bold green]Quarterly Retrain Cycle[/]",
        )
    )

    # Step 1: Generate synthetic training data
    with console.status("[bold green]Generating training data...", spinner="dots"):
        games = generate_training_dataset(n=n_games, seed=seed)
        feedback = generate_quarterly_feedback(games, seed=seed + 1)
        quarter_scores = generate_quarter_scores(games, seed=seed + 2)

    quarters = sorted(quarter_scores.keys())
    from collections import Counter

    tier_dist = Counter(g["hit_tier"] for g in games)

    console.print(f"  Generated {len(games)} games across {len(quarters)} quarters")
    dist_str = ", ".join(f"{t}: {tier_dist.get(t, 0)}" for t in ["Mega", "Hit", "Side", "Hobby"])
    console.print(f"  Tier distribution: {dist_str}")

    # Step 2: Ridge LOOCV weight learning
    console.print("\n[bold]Phase 2: Ridge LOOCV Weight Learning[/]")

    q1_shares: dict[str, float] = {
        "front-loaded": 0.55,
        "balanced": 0.45,
        "live-service": 0.35,
    }

    ml_norms: list[float] = []
    jury_norms: list[float] = []
    y1_log: list[float] = []

    for g, qs in zip(games, quarter_scores_to_per_game(games, quarter_scores), strict=True):
        ml_norms.append(qs["ml"])
        jury_norms.append(qs["jury"] / 4.0)
        y1 = float(g.get("estimated_y1_revenue", 1.0))
        y1_log.append(math.log1p(y1))

    learner = Phase2WeightLearner()
    with console.status("[bold green]Running Ridge LOOCV...", spinner="dots"):
        learner.fit(ml_norms, jury_norms, y1_log)

    result = learner.get_weights()
    ridge_table = Table(title="Ridge Weight Learning")
    ridge_table.add_column("Parameter", style="bold")
    ridge_table.add_column("Value", justify="right")
    ridge_table.add_row("w_ml (learned)", f"{result['w_ml']}")
    ridge_table.add_row("w_jury (learned)", f"{result['w_jury']}")
    ridge_table.add_row("Lambda", f"{result['lambda']}")
    ridge_table.add_row("Nested Rho", f"{result['nested_rho']}")
    ridge_table.add_row("Is Fallback", str(result["is_fallback"]))
    if result["fallback_reason"]:
        ridge_table.add_row("Fallback Reason", str(result["fallback_reason"]))
    console.print(ridge_table)

    # Step 3: Phase transition evaluation
    console.print("\n[bold]Phase Controller: Transition Evaluation[/]")
    controller = PhaseController(initial_phase=0)

    # Simulate progressive phase transitions with feedback
    phase_history: list[dict[str, object]] = []
    for q in quarters:
        q_feedback = [f for f in feedback if f["quarter"] == q]
        if not q_feedback:  # pragma: no cover — generator always produces feedback per quarter
            continue

        # Compute Spearman rho for this quarter
        pred_scores = [f["predicted_score"] for f in q_feedback]
        actual_revs = [f["actual_revenue"] for f in q_feedback]
        rho = _spearman_rho(pred_scores, actual_revs)

        # Accumulate n_samples
        cumulative_n = sum(1 for g in games if g.get("quarter", "") <= q)

        # Feed the controller
        for f in q_feedback:
            controller.get_weights(
                n_samples=cumulative_n,
                agreement_delta=f["prediction_error"],
            )

        decision = controller.evaluate_promotion(
            n=cumulative_n,
            rho=rho,
            w_std=controller.weight_std,
        )
        old_phase = controller.current_phase
        controller.apply_promotion(decision)
        phase_history.append(
            {
                "quarter": q,
                "n_samples": cumulative_n,
                "rho": round(rho, 4),
                "w_std": controller.weight_std,
                "decision": decision,
                "phase": f"{old_phase} -> {controller.current_phase}",
            }
        )

    phase_table = Table(title="Phase Transition History")
    phase_table.add_column("Quarter", style="bold")
    phase_table.add_column("N", justify="right")
    phase_table.add_column("Rho", justify="right")
    phase_table.add_column("W_Std", justify="right")
    phase_table.add_column("Decision")
    phase_table.add_column("Phase")

    for ph in phase_history:
        phase_table.add_row(
            str(ph["quarter"]),
            str(ph["n_samples"]),
            str(ph["rho"]),
            f"{ph['w_std']:.4f}",
            str(ph["decision"]),
            str(ph["phase"]),
        )
    console.print(phase_table)

    # Step 4: PSI drift detection (incremental — compute before eviction)
    console.print("\n[bold]PSI Drift Detection[/]")
    prm = PercentileRankManager()
    psi_results: list[tuple[str, float]] = []
    prev_q: str | None = None
    for q in quarters:
        qd = quarter_scores[q]
        prm.update_quarter(q, qd["ml_scores"], qd["jury_scores"])
        if prev_q is not None and prev_q in prm.quarters:
            psi = prm.compute_psi(prev_q, q, "ml")
            psi_results.append((f"{prev_q} vs {q}", psi))
        prev_q = q

    psi_table = Table(title="PSI Between Quarters")
    psi_table.add_column("Pair", style="bold")
    psi_table.add_column("PSI (ML)", justify="right")
    psi_table.add_column("Status")

    for pair_label, psi in psi_results:
        status = "stable" if psi < 0.10 else ("warning" if psi < 0.25 else "[red]DRIFT[/]")
        psi_table.add_row(pair_label, f"{psi:.4f}", status)

    console.print(psi_table)

    # Step 5: Validation Suite
    console.print("\n[bold]Validation Suite[/]")
    features: list[list[float]] = [extract_features(g) for g in games]
    q1_revenues: list[float] = []
    for g in games:
        y1 = float(g.get("estimated_y1_revenue", 0.0))
        q1_type_val = str(g.get("genre_q1_type", "balanced"))
        q1_rev = y1 * q1_shares.get(q1_type_val, 0.50)
        q1_revenues.append(q1_rev)

    qr = QuantileRegressor()
    qr.fit(features, q1_revenues)

    if qr.is_fitted:
        sel_scores: list[float] = []
        for feat in features:
            pred = qr.predict(feat)
            sel_scores.append(math.log1p(pred.q1_p50) / math.log1p(200_000_000.0))

        metrics = compute_validation_suite(sel_scores, q1_revenues, k=top_k)

        from llmart.monitoring.metrics import classify_metric

        val_table = Table(title="6-Metric Validation Suite")
        val_table.add_column("Metric", style="bold")
        val_table.add_column("Value", justify="right")
        val_table.add_column("Grade")

        for name, val in metrics.items():
            grade = classify_metric(name, val)
            color = {"strong": "green", "conditional": "yellow", "fail": "red"}.get(grade, "white")
            val_table.add_row(name, f"{val:.4f}", f"[{color}]{grade}[/{color}]")

        console.print(val_table)

        from llmart.monitoring.metrics import check_pass_condition

        passed = check_pass_condition(metrics)
        if passed:
            console.print("[green]Validation PASSED (4+ metrics at Conditional+)[/]")
        else:
            console.print("[red]Validation FAILED (< 4 metrics at Conditional+)[/]")
    else:
        console.print("[yellow]Quantile regressor failed to fit[/]")

    # Tier accuracy (synthetic data: tautological match; real data: may diverge)
    correct = sum(
        1
        for g, q1_val in zip(games, q1_revenues, strict=True)
        if classify_tier(q1_val).value == g.get("hit_tier")
    )
    tier_acc = correct / len(games) if games else 0.0
    console.print(f"\n[dim]Tier accuracy (ground truth): {tier_acc:.1%}[/]")
    console.print(f"[dim]Final phase: {controller.current_phase}[/]")


def quarter_scores_to_per_game(
    games: list[dict[str, Any]],
    quarter_scores: dict[str, dict[str, list[float]]],
) -> list[dict[str, float]]:
    """Map quarter-level score arrays back to per-game ml/jury scores."""
    quarter_idx: dict[str, int] = {}
    result: list[dict[str, float]] = []
    for g in games:
        q = g.get("quarter", "2024Q1")
        idx = quarter_idx.get(q, 0)
        qd = quarter_scores.get(q, {"ml_scores": [0.5], "jury_scores": [2.0]})
        ml = qd["ml_scores"][idx] if idx < len(qd["ml_scores"]) else 0.5
        jury = qd["jury_scores"][idx] if idx < len(qd["jury_scores"]) else 2.0
        result.append({"ml": ml, "jury": jury})
        quarter_idx[q] = idx + 1
    return result


@app.command()
def version() -> None:
    """Show version."""
    from llmart import __version__

    console.print(f"llmart {__version__}")
