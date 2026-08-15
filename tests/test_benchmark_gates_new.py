"""Gate mechanics for the 5-tier hierarchy (synthetic scorecards)."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import (
    NULL_KINDS,
    Tier4GateConfig,
    assert_hierarchy_monotone,
    assert_tier0_null_floor,
    assert_tier4_gate,
    score_benchmark_column,
)
from nmr.scorecard import MetricScorecard, evaluate_model


GATE = Tier4GateConfig(
    corr_min=0.0286,
    corr_sharpe_ac_min=1.50,
    fnc_min=0.020,
    deflated_sharpe_min=0.95,
    gain_to_pain_min=1.50,
    cagr_min=0.0,
    turnover_max=0.35,
)


def _synthetic_inputs(n_eras: int = 60, rows_per_era: int = 16, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    for era_num in range(1, n_eras + 1):
        era = f"{era_num:04d}"
        for idx in range(rows_per_era):
            f1 = float(rng.normal())
            latent = 0.8 * f1 + float(rng.normal(0.0, 0.7))
            target = float(np.clip(0.5 + 0.2 * latent, 0.0, 1.0))
            rows.append({
                "era": era, "id": f"{era}_{idx}",
                "prediction": float(rng.random()),
                "numerai_meta_model": float(0.55 * target + 0.45 * rng.random()),
                "target": target,
                "f1": f1,
                "bench": float(0.6 * target + 0.4 * rng.random()),
            })
    full = pl.DataFrame(rows)
    return (
        full.select(["era", "id", "prediction"]),
        full.select(["era", "id", "numerai_meta_model"]),
        full.select(["era", "id", "bench"]),
        full.select(["era", "id", "f1"]),
        full.select(["era", "id", "target"]),
    )


def _make_scorecard(**overrides: float) -> MetricScorecard:
    predictions, meta_model, benchmarks, features, targets = _synthetic_inputs()
    scorecard = evaluate_model(
        predictions, meta_model=meta_model, benchmarks=benchmarks,
        features=features, targets=targets, n_trials=1, seed=77,
        benchmark_col="bench", n_boot=50, min_overlap_eras=20,
        model_id="probe",
    )
    return dataclasses.replace(scorecard, **overrides)


def _null_scorecards() -> dict[str, MetricScorecard]:
    out = {}
    for kind in NULL_KINDS:
        score = _make_scorecard(model_id=kind)
        # Synthetic degeneracy: the fixture's noise metrics are NOT at the
        # null floor (|corr| ~ 0.013, |deflated_sharpe| ~ 0.79 on 60x16
        # rows), and the 0.005/0.05 audit tolerances are calibrated for
        # real-data null baselines. Floor-normalize the two gated fields
        # the fixture cannot bring down; the reject test below still
        # exercises the strict defaults against real synthetic values.
        out[kind] = dataclasses.replace(
            score,
            corr=dataclasses.replace(score.corr, value=0.0),
            deflated_sharpe=0.0,
        )
    return out


def _null_scorecards_unzeroed() -> dict[str, MetricScorecard]:
    """Raw synthetic null scorecards (no floor normalization).

    The fixture's noise metrics (|corr| ~ 0.013, |deflated_sharpe| ~ 0.79
    on 60x16 rows) are realistic for small-sample noise but exceed the
    strict audit defaults, so callers must pass explicit tolerances.
    """
    return {kind: _make_scorecard(model_id=kind) for kind in NULL_KINDS}


def test_score_benchmark_column_wraps_predictions() -> None:
    _, _, benchmarks, _, _ = _synthetic_inputs()
    out = score_benchmark_column(benchmarks, column="bench")
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == benchmarks.height


def test_score_benchmark_column_unknown_column_raises() -> None:
    _, _, benchmarks, _, _ = _synthetic_inputs()
    with pytest.raises(ValueError, match="nope"):
        score_benchmark_column(benchmarks, column="nope")


def test_tier0_null_floor_passes_on_null_scorecards() -> None:
    assert_tier0_null_floor(_null_scorecards())


def test_tier0_null_floor_rejects_high_corr() -> None:
    cards = _null_scorecards()
    cards["null_constant_05"] = dataclasses.replace(
        cards["null_constant_05"],
        corr=dataclasses.replace(cards["null_constant_05"].corr, value=0.05),
    )
    with pytest.raises(ValueError, match="null_constant_05"):
        assert_tier0_null_floor(cards)


def test_tier0_null_floor_requires_all_four() -> None:
    cards = _null_scorecards()
    del cards["null_gaussian_rand"]
    with pytest.raises(ValueError, match="null_gaussian_rand"):
        assert_tier0_null_floor(cards)


def test_tier0_null_floor_defaults_are_pinned() -> None:
    params = inspect.signature(assert_tier0_null_floor).parameters
    assert params["corr_tol"].default == 0.005
    assert params["sharpe_tol"].default == 0.10
    assert params["dsr_tol"].default == 0.05


def test_hierarchy_monotone_atol_default_is_pinned() -> None:
    params = inspect.signature(assert_hierarchy_monotone).parameters
    assert params["atol"].default == 1e-5


def test_tier0_null_floor_passes_unzeroed_cards_at_explicit_tolerances() -> None:
    # Realistic synthetic values: corr ~ -0.0126, corr_sharpe_ac ~ -0.044,
    # deflated_sharpe ~ +0.7885 (fixed seed). Explicit tolerances reflect
    # the small-sample noise floor; the strict defaults stay pinned by the
    # signature test above and exercised by the reject tests below.
    assert_tier0_null_floor(
        _null_scorecards_unzeroed(),
        corr_tol=0.02,
        sharpe_tol=0.10,
        dsr_tol=0.80,
    )


def test_tier0_null_floor_rejects_high_corr_sharpe_at_strict_default() -> None:
    cards = _null_scorecards()
    cards["null_constant_05"] = dataclasses.replace(
        cards["null_constant_05"],
        corr_sharpe_ac=dataclasses.replace(
            cards["null_constant_05"].corr_sharpe_ac, value=0.2
        ),
    )
    with pytest.raises(ValueError, match="corr_sharpe_ac"):
        assert_tier0_null_floor(cards)


def test_tier0_null_floor_rejects_high_deflated_sharpe_at_strict_default() -> None:
    cards = _null_scorecards()
    cards["null_constant_05"] = dataclasses.replace(
        cards["null_constant_05"], deflated_sharpe=0.1
    )
    with pytest.raises(ValueError, match="deflated_sharpe"):
        assert_tier0_null_floor(cards)


def test_tier4_gate_passes_on_strong_scorecard() -> None:
    card = _make_scorecard(
        corr=dataclasses.replace(_make_scorecard().corr, value=0.04),
        corr_sharpe_ac=dataclasses.replace(_make_scorecard().corr_sharpe_ac, value=1.8),
        fnc=0.03,
        deflated_sharpe=1.2,
        gain_to_pain_ratio=2.0,
        cagr_1y=0.1,
        turnover_mean=0.1,
    )
    assert_tier4_gate(card, GATE)


def test_tier4_gate_reports_every_violation() -> None:
    card = _make_scorecard(
        corr=dataclasses.replace(_make_scorecard().corr, value=0.01),
        fnc=0.001,
        turnover_mean=0.9,
    )
    with pytest.raises(ValueError) as excinfo:
        assert_tier4_gate(card, GATE)
    message = str(excinfo.value)
    assert "corr" in message and "fnc" in message and "turnover" in message


def test_tier4_gate_missing_turnover_fails_loudly() -> None:
    card = _make_scorecard(turnover_mean=None, turnover_reason="no id column")
    with pytest.raises(ValueError, match=r"turnover.*no id column"):
        assert_tier4_gate(card, GATE)


def test_monotone_ordering_passes_on_escalating_tiers() -> None:
    cards: dict[str, MetricScorecard] = {}
    tier_of: dict[str, int] = {}
    for tier, scalar in [(0, 0.0), (1, 0.2), (2, 0.4), (3, 0.6), (4, 0.7)]:
        model_id = f"t{tier}_probe"
        cards[model_id] = _make_scorecard(model_id=model_id, rank_scalar=scalar)
        tier_of[model_id] = tier
    assert_hierarchy_monotone(cards, tier_of=tier_of)


def test_monotone_rejects_inverted_tiers() -> None:
    cards: dict[str, MetricScorecard] = {}
    tier_of: dict[str, int] = {}
    for tier, scalar in [(0, 0.5), (1, 0.4), (2, 0.3), (3, 0.2), (4, 0.1)]:
        model_id = f"t{tier}_probe"
        cards[model_id] = _make_scorecard(model_id=model_id, rank_scalar=scalar)
        tier_of[model_id] = tier
    with pytest.raises(ValueError, match="monotone|ordering|tier"):
        assert_hierarchy_monotone(cards, tier_of=tier_of)


def _monotone_fixture() -> tuple[dict[str, MetricScorecard], dict[str, int]]:
    cards: dict[str, MetricScorecard] = {}
    tier_of: dict[str, int] = {}
    for tier, scalar in [(0, 0.0), (1, 0.2), (2, 0.4), (3, 0.6), (4, 0.7)]:
        model_id = f"t{tier}_probe"
        cards[model_id] = _make_scorecard(model_id=model_id, rank_scalar=scalar)
        tier_of[model_id] = tier
    return cards, tier_of


def test_monotone_missing_tier_raises() -> None:
    cards, tier_of = _monotone_fixture()
    del tier_of["t2_probe"]
    with pytest.raises(ValueError, match=r"0\.\.4"):
        assert_hierarchy_monotone(cards, tier_of=tier_of)


def test_monotone_missing_scorecard_raises() -> None:
    cards, tier_of = _monotone_fixture()
    del cards["t3_probe"]
    with pytest.raises(ValueError, match="t3_probe"):
        assert_hierarchy_monotone(cards, tier_of=tier_of)
