"""T2 LLM-as-Judge node: GPT-5.2 + Opus-4.5, 500 -> 100."""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

from llmart.models.jury import jury_star_from_passes
from llmart.pipeline.state import GraphState
from llmart.prompts.calibrator import CALIBRATOR_SYSTEM, CALIBRATOR_USER
from llmart.prompts.rubric_5dim import RUBRIC_5DIM_SYSTEM, RUBRIC_5DIM_USER, build_dim_schema

# PDF slide 2: Gameplay(30%) | Innovation(20%) | Monetize(20%) | Polish(15%) | Narrative(15%)
DIMENSIONS = [
    "gameplay",
    "innovation",
    "monetization",
    "polish",
    "narrative",
]
DIM_WEIGHTS = [0.30, 0.20, 0.20, 0.15, 0.15]

# 4-cat grading: L(1), M(2), H(3), E(4)
SCORE_MIN = 1
SCORE_MAX = 4

K_PASSES = 3
MAX_DELTA_CAL = 0.15  # calibration offset clamp ±
JUDGE_TEMPERATURE = 0.3  # Low temperature for consistent structured evaluation

# Resilience: import retry decorator (no-op identity if tenacity unavailable)
try:
    from llmart.resilience import llm_retry as _llm_retry_decorator
except ImportError:  # pragma: no cover

    def _llm_retry_decorator(fn: object) -> object:  # type: ignore[misc]
        return fn


def _deterministic_seed(game_id: str, pass_idx: int) -> int:
    """Produce a repeatable seed from game_id + pass index."""
    h = hashlib.sha256(f"{game_id}:{pass_idx}".encode()).hexdigest()
    return int(h[:8], 16)


# Genre-specific dimension biases for realistic mock scoring.
# Each genre maps to per-dimension offsets from the base quality centre.
# Positive = this genre tends to score higher on this dimension.
_GENRE_DIM_BIAS: dict[str, dict[str, float]] = {
    "Roguelike Deckbuilder": {"gameplay": 0.4, "innovation": 0.2, "narrative": -0.5},
    "Roguelike": {"gameplay": 0.3, "innovation": 0.1, "narrative": -0.3},
    "Action Roguelike": {"gameplay": 0.4, "polish": 0.1, "narrative": -0.4},
    "Survival Craft": {"gameplay": 0.3, "monetization": 0.2, "narrative": -0.3},
    "City Builder": {"gameplay": 0.2, "innovation": -0.1, "monetization": 0.3, "narrative": -0.5},
    "Simulation": {"gameplay": 0.2, "monetization": 0.2, "narrative": -0.4},
    "Co-op Horror": {"gameplay": 0.1, "innovation": 0.2, "narrative": 0.2, "polish": -0.2},
    "Horror": {"narrative": 0.4, "innovation": 0.2, "monetization": -0.2},
    "RPG": {"narrative": 0.5, "gameplay": 0.2, "innovation": -0.1},
    "Metroidvania": {"gameplay": 0.3, "polish": 0.2, "monetization": -0.3, "narrative": -0.1},
    "Tower Defense": {"gameplay": 0.2, "monetization": 0.3, "innovation": -0.3, "narrative": -0.5},
    "Visual Novel": {"narrative": 0.6, "gameplay": -0.4, "innovation": -0.2},
    "Puzzle": {"innovation": 0.2, "gameplay": 0.1, "narrative": -0.3, "monetization": -0.2},
}

# Per-dimension noise sigma: higher for subjective dimensions (narrative, innovation)
_DIM_NOISE_SIGMA: dict[str, float] = {
    "gameplay": 0.35,
    "innovation": 0.55,
    "monetization": 0.40,
    "polish": 0.35,
    "narrative": 0.50,
}

# Circuit breaker: max consecutive LLM failures before falling back to mock
_CIRCUIT_BREAKER_THRESHOLD = 5


def _mock_single_pass(game: dict[str, Any], pass_idx: int) -> dict[str, int]:
    """Generate deterministic 4-cat scores for one pass (mock mode).

    Genre-aware: applies per-dimension bias offsets so that, e.g., roguelikes
    score higher on gameplay but lower on narrative. This produces realistic
    dimension variance patterns that a real LLM would exhibit.
    """
    seed = _deterministic_seed(game["game_id"], pass_idx)
    rng = random.Random(seed)  # noqa: S311

    dims = list(DIMENSIONS)
    rng.shuffle(dims)  # dimension-order shuffle (bias mitigation)

    rating = game.get("steam_rating", 0.5)
    ml_pct = game.get("ml_percentile", rating)  # fall back to rating if unavailable
    # Blend rating (70%) with ml_percentile (30%) to reduce T1/T2 gap in mock mode
    base_quality = 0.7 * rating + 0.3 * ml_pct

    # Genre-specific dimension biases
    genre = game.get("genre", "")
    genre_biases = _GENRE_DIM_BIAS.get(genre, {})

    scores: dict[str, int] = {}
    for dim in dims:
        # Map blended quality [0,1] -> centre [1,4] + genre bias
        centre = SCORE_MIN + base_quality * (SCORE_MAX - SCORE_MIN)
        bias = genre_biases.get(dim, 0.0)
        sigma = _DIM_NOISE_SIGMA.get(dim, 0.4)
        noise = rng.gauss(0, sigma)
        scores[dim] = max(SCORE_MIN, min(SCORE_MAX, round(centre + bias + noise)))
    return scores


def _mock_calibrator(passes: list[dict[str, int]], agree: bool) -> float:
    """Produce a deterministic delta_cal based on pass variance.

    SOT B1: Calibrator invoked on pass^3 disagreement.
    Returns signed offset: positive if later passes trend up, negative if down.

    Dead-zone: when sum-of-scores spread across passes is <= 2, the disagreement
    is considered noise-level and delta_cal returns 0.0 (no calibration applied).
    This avoids micro-adjustments from rounding-level differences.
    """
    if agree:
        # Direct path: no calibration needed (SOT B1 "Y -> Direct")
        return 0.0
    # Calibration path: compute signed direction from pass trend
    all_totals = [sum(p.values()) for p in passes]
    trend = all_totals[-1] - all_totals[0]  # positive if scores increase
    spread = max(all_totals) - min(all_totals)
    if spread <= 2:  # noise-level spread → no calibration
        return 0.0
    sign = 1.0 if trend >= 0 else -1.0
    return round(max(-MAX_DELTA_CAL, min(MAX_DELTA_CAL, sign * (spread - 2) * 0.03)), 4)


def _jury_score(dim_scores: dict[str, float]) -> float:
    """Weighted jury score across 5 dimensions."""
    total = 0.0
    for dim, weight in zip(DIMENSIONS, DIM_WEIGHTS, strict=True):
        total += dim_scores.get(dim, 0.0) * weight
    return round(total, 4)


def _aggregate_passes(passes: list[dict[str, int]]) -> dict[str, float]:
    """Average dimension scores across k passes."""
    agg: dict[str, float] = {}
    for dim in DIMENSIONS:
        agg[dim] = round(sum(p[dim] for p in passes) / len(passes), 2)
    return agg


def _pass3_agreement(passes: list[dict[str, int]], threshold: float = 1.0) -> bool:
    """Check whether all passes agree within threshold per dimension."""
    for dim in DIMENSIONS:
        vals = [p[dim] for p in passes]
        if max(vals) - min(vals) > threshold:
            return False
    return True


@_llm_retry_decorator
def _real_llm_judge(  # pragma: no cover
    game: dict[str, Any],
) -> tuple[dict[str, float], float, float, bool, bool, list[dict[str, int]]]:
    """Call real LLM APIs (OpenAI + Anthropic) for scoring.

    Returns: (dim_scores, jury_score, delta_cal, pass3_agree, flagged)
    """
    import anthropic
    import openai

    tags_str = ", ".join(game.get("tags", []))

    passes: list[dict[str, int]] = []
    oai = openai.OpenAI()
    for pass_idx in range(K_PASSES):
        # Shuffle dimension order per pass for bias mitigation (SOT B1)
        rng = random.Random(_deterministic_seed(game["game_id"], pass_idx))  # noqa: S311
        dim_order = list(DIMENSIONS)
        rng.shuffle(dim_order)

        # Dynamic JSON schema reflects shuffled dimension order
        user_prompt = RUBRIC_5DIM_USER.format(
            title=game["title"],
            genre=game["genre"],
            developer=game["developer"],
            steam_rating=game["steam_rating"],
            review_count=game["review_count"],
            price_usd=game["price_usd"],
            tags=tags_str,
            dim_schema=build_dim_schema(dim_order),
        )

        resp = oai.chat.completions.create(
            model="gpt-5.2",  # GPT-5.2 Thinking (Primary Judge)
            messages=[
                {"role": "system", "content": RUBRIC_5DIM_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=JUDGE_TEMPERATURE,
        )
        content = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {}
        scores: dict[str, int] = {}
        for dim in DIMENSIONS:
            val = parsed.get(dim, {})
            raw = int(val.get("score", 2)) if isinstance(val, dict) else 2
            scores[dim] = max(SCORE_MIN, min(SCORE_MAX, raw))
        passes.append(scores)

    dim_scores = _aggregate_passes(passes)
    agree = _pass3_agreement(passes)

    # Calibrator via Anthropic Opus — only on disagreement (SOT B1 flow)
    delta_cal = 0.0
    flagged = False
    if not agree:
        cal_client = anthropic.Anthropic()
        cal_prompt = CALIBRATOR_USER.format(
            scores_json=json.dumps(dim_scores),
            pass_scores_json=json.dumps([dict(p) for p in passes]),
            game_context=(
                f"{game['title']} ({game['genre']}), rating={game.get('steam_rating', 0):.0%}"
            ),
        )
        cal_resp = cal_client.messages.create(
            model="claude-opus-4-6",  # Opus 4.6 Calibrator
            max_tokens=512,
            temperature=0.2,
            system=CALIBRATOR_SYSTEM,
            messages=[{"role": "user", "content": cal_prompt}],
        )
        first_block = cal_resp.content[0] if cal_resp.content else None
        cal_text = first_block.text if first_block and hasattr(first_block, "text") else "{}"
        # Strip markdown fences if present (e.g. ```json ... ```)
        stripped = cal_text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            cal_data = json.loads(stripped)
        except json.JSONDecodeError:
            cal_data = {}  # Graceful fallback: treat as no calibration
        delta_cal = max(-MAX_DELTA_CAL, min(MAX_DELTA_CAL, float(cal_data.get("delta_cal", 0.0))))
        flagged = bool(cal_data.get("flag", False))

    jury = _jury_score(dim_scores)
    return dim_scores, jury, round(delta_cal, 4), agree, flagged, passes


@_llm_retry_decorator
def _real_primary_only(  # pragma: no cover
    game: dict[str, Any],
) -> tuple[dict[str, float], float, float, bool, bool, list[dict[str, int]]]:
    """Call only GPT Primary (no calibrator) — partial real mode.

    Geode pattern: partial degradation when Anthropic key absent.
    Calibrator is skipped; delta_cal = 0.0 always.
    """
    import openai

    tags_str = ", ".join(game.get("tags", []))
    passes: list[dict[str, int]] = []
    oai = openai.OpenAI()

    for pass_idx in range(K_PASSES):
        rng = random.Random(_deterministic_seed(game["game_id"], pass_idx))  # noqa: S311
        dim_order = list(DIMENSIONS)
        rng.shuffle(dim_order)

        user_prompt = RUBRIC_5DIM_USER.format(
            title=game["title"],
            genre=game["genre"],
            developer=game["developer"],
            steam_rating=game["steam_rating"],
            review_count=game["review_count"],
            price_usd=game["price_usd"],
            tags=tags_str,
            dim_schema=build_dim_schema(dim_order),
        )

        resp = oai.chat.completions.create(
            model="gpt-5.2",  # GPT-5.2 Thinking (Primary Judge)
            messages=[
                {"role": "system", "content": RUBRIC_5DIM_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=JUDGE_TEMPERATURE,
        )
        content = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {}
        scores: dict[str, int] = {}
        for dim in DIMENSIONS:
            val = parsed.get(dim, {})
            raw = int(val.get("score", 2)) if isinstance(val, dict) else 2
            scores[dim] = max(SCORE_MIN, min(SCORE_MAX, raw))
        passes.append(scores)

    dim_scores = _aggregate_passes(passes)
    agree = _pass3_agreement(passes)
    jury = _jury_score(dim_scores)
    # No calibrator — delta_cal is always 0.0
    return dim_scores, jury, 0.0, agree, False, passes


def _resolve_mode(requested_mode: str) -> tuple[str, list[str]]:
    """Resolve requested LLM mode against available API keys.

    OpenClaw pattern: Auth Profile detection before pipeline start.
    Geode pattern: force_dry_run when keys absent.

    Returns:
        (effective_mode, degradation_messages)
        effective_mode: "real" | "real_no_calibrator" | "mock"
    """
    from llmart.readiness import check_readiness

    report = check_readiness(requested_mode)
    messages: list[str] = list(report.guidance)

    if requested_mode != "real":
        return "mock", []

    return report.available_mode, messages


def llm_judge_node(state: GraphState) -> dict[str, Any]:
    """Score each candidate with 5-Dim Value Lens (mock or real)."""
    candidates: list[dict[str, Any]] = state.get("candidates", [])
    mode = state.get("mode", "mock")
    errors: list[str] = []

    # Resolve effective mode via readiness check (Geode/OpenClaw pattern)
    effective_mode, degradation_msgs = _resolve_mode(mode)
    if degradation_msgs:
        errors.extend(degradation_msgs)

    use_real = effective_mode == "real"
    use_partial = effective_mode == "real_no_calibrator"

    # Also check env override
    if os.environ.get("LLMART_REAL_API", "").lower() == "true":
        use_real = True

    # Circuit breaker state: track consecutive LLM failures
    consecutive_failures = 0
    circuit_broken = False

    result: list[dict[str, Any]] = []
    for game in candidates:
        try:
            flagged = False
            pass_dicts: list[dict[str, int | float]] = []

            if (use_real or use_partial) and circuit_broken:
                # Circuit broken: fall back to mock for remaining candidates
                errors.append(
                    f"llm_judge: circuit breaker tripped after "
                    f"{_CIRCUIT_BREAKER_THRESHOLD} consecutive failures, "
                    f"falling back to mock for {game.get('game_id', '?')}"
                )
                passes = [_mock_single_pass(game, i) for i in range(K_PASSES)]
                dim_scores = _aggregate_passes(passes)
                agree = _pass3_agreement(passes)
                jury = _jury_score(dim_scores)
                delta_cal = _mock_calibrator(passes, agree)
                pass_dicts = [dict(p) for p in passes]
            elif use_real:
                dim_scores, jury, delta_cal, agree, flagged, raw_passes = _real_llm_judge(game)
                pass_dicts = [dict(p) for p in raw_passes]
                consecutive_failures = 0  # reset on success
            elif use_partial:
                dim_scores, jury, delta_cal, agree, flagged, raw_passes = _real_primary_only(game)
                pass_dicts = [dict(p) for p in raw_passes]
                consecutive_failures = 0  # reset on success
            else:
                passes = [_mock_single_pass(game, i) for i in range(K_PASSES)]
                dim_scores = _aggregate_passes(passes)
                agree = _pass3_agreement(passes)
                jury = _jury_score(dim_scores)
                delta_cal = _mock_calibrator(passes, agree)
                pass_dicts = [dict(p) for p in passes]
                # Flag simulation: flag if any dimension deviates >= 2.0 from mean
                dim_vals = list(dim_scores.values())
                dim_mean = sum(dim_vals) / len(dim_vals) if dim_vals else 0.0
                flagged = any(abs(v - dim_mean) >= 2.0 for v in dim_vals)

            # jury_star: confidence-weighted consensus (SOT §2.2)
            jury_result = jury_star_from_passes(pass_dicts)

            result.append(
                {
                    **game,
                    "dim_scores": dim_scores,
                    "jury_score": jury,
                    "jury_star": jury_result.jury_final,
                    "jury_delta": jury_result.delta,
                    "jury_conf": jury_result.conf_scalar,
                    "delta_cal": delta_cal,
                    "pass3_agree": agree,
                    "flagged": flagged,
                }
            )
        except Exception as exc:
            errors.append(f"llm_judge: {game.get('game_id', '?')}: {exc}")
            if use_real or use_partial:
                consecutive_failures += 1
                if consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                    circuit_broken = True

    stage_counts = dict(state.get("stage_counts", {}))
    stage_counts["llm_judge"] = len(result)
    stage_timings = dict(state.get("stage_timings", {}))

    return {
        "candidates": result,
        "stage": "llm_judge",
        "errors": errors,
        "stage_counts": stage_counts,
        "stage_timings": stage_timings,
    }
