"""Tests for scorecard aggregation (E5)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import polars as pl
import pytest
from numerai_tools.scoring import correlation_contribution

from nmr._transforms import rank_gaussianize, tie_kept_rank
from nmr.evaluation import EvaluationEngine
from nmr.inference import block_bootstrap_ci, era_series_stats, resolve_block_len
from nmr.payout import (
    annual_compounded_return,
    gain_to_pain_ratio,
    kelly_fraction,
    payout_report,
    payout_series,
    simulate_overlapping_portfolio,
)
from nmr.scorecard import MetricScorecard, evaluate_cross_check, evaluate_model


def _tiny_inputs() -> (
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
):
    rows: list[dict[str, float | str]] = []
    bench: list[dict[str, float | str]] = []
    for i in range(1, 21):
        era = f"{i:04d}"
        for j in range(3):
            pred = (0.2 * i) + (0.03 * j)
            meta = (0.15 * i) - (0.02 * j)
            f1 = (i + j) % 5
            f2 = (2 * i + j) % 5
            f3 = (3 * i + j) % 5
            rows.append(
                {
                    "era": era,
                    "id": f"{era}_{j:03d}",
                    "prediction": float(pred),
                    "target": float((i + j) % 5) / 4.0,
                    "target_cyrusd_20": float((i + j) % 5) / 4.0,
                    "target_cyrusd_60": float((2 * i + j) % 5) / 4.0,
                    "f1": float(f1),
                    "f2": float(f2),
                    "f3": float(f3),
                    "numerai_meta_model": float(meta),
                }
            )
            bench.append(
                {
                    "era": era,
                    "id": f"{era}_{j:03d}",
                    "v52_lgbm_cyrusd20": float(meta),
                }
            )

    full = pl.DataFrame(rows)
    predictions = full.select(["era", "id", "prediction"])
    meta_model = full.select(["era", "id", "numerai_meta_model"])
    targets = full.select(
        ["era", "id", "target", "target_cyrusd_20", "target_cyrusd_60"]
    )
    features = full.select(["era", "id", "f1", "f2", "f3"])
    benchmarks = pl.DataFrame(bench)
    return predictions, meta_model, benchmarks, features, targets


def test_scorecard_composition_parity_and_cells() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=11,
        benchmark_col="v52_lgbm_cyrusd20",
        n_boot=5,
        alpha=0.05,
        min_overlap_eras=20,
    )
    assert isinstance(score, MetricScorecard)

    base = (
        predictions.join(meta_model, on=["era", "id"], how="inner")
        .join(targets, on=["era", "id"], how="inner")
        .join(features, on=["era", "id"], how="inner")
        .join(benchmarks, on=["era", "id"], how="left")
    )
    evaluator = EvaluationEngine("custom")
    corr_by_era = evaluator.per_era_corr(
        base, pred_col="prediction", target_col="target"
    )
    mmc_by_era = evaluator.per_era_mmc(
        base,
        pred_col="prediction",
        meta_col="numerai_meta_model",
        target_col="target",
    )
    direct_payout = payout_report(
        corr_by_era,
        mmc_by_era,
        horizon="20D",
        n_trials=1,
        seed=11,
        n_boot=5,
        alpha=0.05,
    )

    assert score.rank_scalar == direct_payout.mean_payout
    assert score.deflated_sharpe == direct_payout.deflated_sharpe
    assert score.n_eras == direct_payout.n_eras

    corr_vals = [corr_by_era[k] for k in sorted(corr_by_era, key=int)]
    corr_stats = era_series_stats(corr_vals)
    corr_ci = block_bootstrap_ci(
        corr_vals,
        lambda a: float(era_series_stats(a).mean),
        block_len=resolve_block_len(len(corr_vals), "20D"),
        n_boot=5,
        seed=11,
        alpha=0.05,
    )
    assert score.corr.value == corr_stats.mean
    assert score.corr.ci_low == corr_ci.lo
    assert score.corr.ci_high == corr_ci.hi
    assert score.corr.n_eras == len(corr_vals)


def test_scorecard_bmc_cell_oracle_parity() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=13,
        benchmark_col="v52_lgbm_cyrusd20",
        n_boot=5,
        min_overlap_eras=20,
    )
    assert score.bmc is not None

    base = (
        predictions.join(meta_model, on=["era", "id"], how="inner")
        .join(targets, on=["era", "id"], how="inner")
        .join(features, on=["era", "id"], how="inner")
        .join(benchmarks, on=["era", "id"], how="left")
    )
    one_era = sorted(base.get_column("era").unique().to_list(), key=int)[0]
    pdf = (
        base.filter(pl.col("era") == one_era)
        .select(["prediction", "v52_lgbm_cyrusd20", "target"])
        .to_pandas()
    )
    direct = float(
        correlation_contribution(
            pdf[["prediction"]],
            pdf["v52_lgbm_cyrusd20"].rename("v52_lgbm_cyrusd20"),
            pdf["target"].rename("target"),
        )["prediction"]
    )
    evaluator = EvaluationEngine("custom")
    bmc_map = evaluator.per_era_bmc(
        base,
        pred_col="prediction",
        benchmark_col="v52_lgbm_cyrusd20",
        target_col="target",
        min_overlap_eras=20,
    )
    assert bmc_map[one_era] == pytest.approx(direct, abs=1e-6)


def test_scorecard_to_frame_one_row_and_columns() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=21,
        benchmark_col="v52_lgbm_cyrusd20",
        n_boot=5,
        min_overlap_eras=20,
    )
    frame = score.to_frame()
    assert frame.height == 1
    required = {
        "rank_scalar",
        "deflated_sharpe",
        "mean_payout",
        "mean_payout_ci_low",
        "mean_payout_ci_high",
        "mean_payout_n_eras",
        "corr",
        "corr_ci_low",
        "corr_ci_high",
        "corr_n_eras",
        "bmc",
        "bmc_n_eras",
        "horizon_n_eras",
        "regime_corr_json",
        "bmc_reason",
        "cwmm_reason",
        "horizon_reason",
        "regime_reason",
        "cagr_1y",
        "gain_to_pain_ratio",
        "kelly_fraction",
        "mmc_down",
        "mmc_down_n_eras",
        "mmc_down_reason",
        "turnover_mean",
        "turnover_std",
        "turnover_reason",
        "sim_portfolio_cagr",
        "sim_portfolio_mdd",
        "sim_capital_utilization",
    }
    assert required.issubset(set(frame.columns))


def test_scorecard_thin_coverage_sets_none_with_reason() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=33,
        benchmark_col="v52_lgbm_cyrusd20",
        n_boot=5,
        min_overlap_eras=40,
    )

    assert score.bmc is None
    assert score.cwmm is None
    assert score.horizon_stability is None
    assert score.bmc_reason is not None and "Non-vacuity violation" in score.bmc_reason
    assert (
        score.cwmm_reason is not None and "Non-vacuity violation" in score.cwmm_reason
    )
    assert (
        score.horizon_reason is not None
        and "Non-vacuity violation" in score.horizon_reason
    )


def test_scorecard_noncoverage_valueerror_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()

    def _boom(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(EvaluationEngine, "per_era_bmc", _boom)

    with pytest.raises(ValueError, match="boom"):
        evaluate_model(
            predictions,
            meta_model=meta_model,
            benchmarks=benchmarks,
            features=features,
            targets=targets,
            n_trials=1,
            seed=10,
            benchmark_col="v52_lgbm_cyrusd20",
            n_boot=5,
            min_overlap_eras=20,
        )


def test_scorecard_tier2_tier3_do_not_rerank() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()
    with_tiers = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=5,
        benchmark_col="v52_lgbm_cyrusd20",
        regime_labels=pl.DataFrame(
            {
                "era": [f"{i:04d}" for i in range(1, 21)],
                "regime": ["a" if i <= 10 else "b" for i in range(1, 21)],
            }
        ),
        n_boot=5,
        min_overlap_eras=20,
    )
    without_tiers = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=None,
        features=features,
        targets=targets,
        n_trials=1,
        seed=5,
        benchmark_col=None,
        regime_labels=None,
        n_boot=5,
        min_overlap_eras=20,
    )
    assert with_tiers.rank_scalar == without_tiers.rank_scalar
    assert with_tiers.deflated_sharpe == without_tiers.deflated_sharpe


def test_scorecard_degenerate_predictions_no_nan() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()
    predictions = predictions.with_columns(pl.lit(0.5).alias("prediction"))
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=3,
        benchmark_col="v52_lgbm_cyrusd20",
        n_boot=5,
        min_overlap_eras=20,
    )
    assert score.rank_scalar == pytest.approx(0.0)
    row = score.to_frame().row(0, named=True)
    for value in row.values():
        if isinstance(value, float):
            assert value == value


def test_scorecard_synthetic_determinism_cross_process() -> None:
    """Cross-process determinism: two fresh interpreters over the same
    synthetic payload must produce identical scorecard output. Fully hermetic
    — no real-data files are read, so CI always executes this gate."""
    code = r"""
import json
import polars as pl
from nmr.scorecard import evaluate_model

rows = []
bench = []
for i in range(1, 21):
    era = f"{i:04d}"
    for j in range(2):
        pred = (0.2 * i) + (0.03 * j)
        meta = (0.15 * i) - (0.02 * j)
        f1 = (i + j) % 5
        f2 = (2 * i + j) % 5
        f3 = (3 * i + j) % 5
        rows.append(
            {
                "era": era,
                "id": f"{era}_{j:03d}",
                "prediction": float(pred),
                "target": float((i + j) % 5) / 4.0,
                "target_cyrusd_20": float((i + j) % 5) / 4.0,
                "target_cyrusd_60": float((2 * i + j) % 5) / 4.0,
                "f1": float(f1),
                "f2": float(f2),
                "f3": float(f3),
                "numerai_meta_model": float(meta),
            }
        )
        bench.append(
            {
                "era": era,
                "id": f"{era}_{j:03d}",
                "v52_lgbm_cyrusd20": float(meta),
            }
        )

full = pl.DataFrame(rows)
pred = full.select(["era", "id", "prediction"])
meta = full.select(["era", "id", "numerai_meta_model"])
targets = full.select(["era", "id", "target", "target_cyrusd_20", "target_cyrusd_60"])
features = full.select(["era", "id", "f1", "f2", "f3"])
benchmarks = pl.DataFrame(bench)

card = evaluate_model(
    pred,
    meta_model=meta,
    benchmarks=benchmarks,
    features=features,
    targets=targets,
    n_trials=1,
    seed=77,
    benchmark_col="v52_lgbm_cyrusd20",
    n_boot=2,
    min_overlap_eras=20,
)
row = card.to_frame().to_dicts()[0]
# Timing fields are wall-clock dependent and must not participate in
# cross-process determinism checks.
row = {
    k: v
    for k, v in row.items()
    if k not in {"quality_metric_total_seconds", "quality_metric_timings_json"}
    and not k.startswith("timing_")
}
print(json.dumps(row, sort_keys=True, default=str))
"""
    cmd = [sys.executable, "-c", code]
    run1 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    run2 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert run1.stdout.strip() == run2.stdout.strip()


def test_evaluate_model_input_validation_branches() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()
    base = dict(
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=7,
        min_overlap_eras=5,
    )

    with pytest.raises(ValueError, match="must be a polars DataFrame"):
        evaluate_model(predictions.to_pandas(), **base)
    with pytest.raises(ValueError, match="Missing required columns"):
        evaluate_model(
            predictions.rename({"prediction": "other"}),
            **{**base, "pred_col": "prediction"},
        )
    with pytest.raises(ValueError, match="Missing required columns"):
        evaluate_model(
            predictions.rename({"era": "round"}),
            **{**base, "era_col": "era"},
        )
    with pytest.raises(ValueError, match="at least one feature column"):
        evaluate_model(
            predictions,
            **{
                **base,
                "features": features.select(["era", "id"]),
            },
        )


def test_evaluate_model_regime_labels_populate_scorecard() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()
    regimes = pl.DataFrame(
        {"era": [f"{i:04d}" for i in range(1, 21)], "regime": ["bull"] * 20}
    )
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=None,
        features=features,
        targets=targets,
        n_trials=1,
        seed=7,
        min_overlap_eras=5,
        regime_labels=regimes,
    )
    assert score.regime_corr is not None
    row = score.to_frame().to_dicts()[0]
    assert row["regime_count"] == 1
    assert row["regime_corr_json"] is not None


def test_evaluate_model_auto_selects_benchmark_column() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=7,
        min_overlap_eras=5,
        benchmark_col=None,  # must auto-select the first non-join column
    )
    assert score.bmc is not None
    assert score.bmc.n_eras >= 5
    assert np.isfinite(score.bmc.value)


def test_sorted_numeric_keys_rejects_non_numeric_eras() -> None:
    from nmr.scorecard import _sorted_numeric_keys

    assert _sorted_numeric_keys({"0575": 1.0, "0583": 2.0}) == ["0575", "0583"]
    with pytest.raises(ValueError, match="Non-numeric era keys"):
        _sorted_numeric_keys({"0575": 1.0, "X": 0.0})


def _mmc_down_frames(n_down: int) -> tuple[pl.DataFrame, ...]:
    """30 eras x 4 rows; meta CORR < 0 in exactly the LAST n_down eras.

    Sign guarantee (holds for ANY corr(pred, target) in [-1, 1]):
      up eras:   meta =  target + 0.5*pred -> corr(meta, target) >= 0.5 > 0
      down eras: meta = -target + 0.5*pred -> corr(meta, target) <= -0.5 < 0
    Target is era-varying ((i + j) % 5) so the per-era CORR series is
    non-degenerate — payout_report's deflated_sharpe requires finite skew/kurt,
    and an era-invariant target makes the series constant and raises ValueError.
    """
    rows: list[dict[str, float | str]] = []
    for i in range(1, 31):
        era = f"{i:04d}"
        downside = i > (30 - n_down)
        for j in range(4):
            pred = 0.1 + 0.1 * j
            target = float((i + j) % 5) / 4.0
            meta = -target + 0.5 * pred if downside else target + 0.5 * pred
            rows.append(
                {
                    "era": era,
                    "id": f"id{j}",
                    "prediction": pred,
                    "numerai_meta_model": meta,
                    "target": target,
                    "f1": float(j),
                }
            )
    full = pl.DataFrame(rows)
    return (
        full.select(["era", "id", "prediction"]),
        full.select(["era", "id", "numerai_meta_model"]),
        full.select(["era", "id", "f1"]),
        full.select(["era", "id", "target"]),
    )


def test_mmc_down_filtering() -> None:
    predictions, meta_model, features, targets = _mmc_down_frames(10)
    full = predictions.join(meta_model, on=["era", "id"]).join(
        targets, on=["era", "id"]
    )
    engine = EvaluationEngine("custom")
    mmc_by_era = engine.per_era_mmc(
        full, pred_col="prediction", meta_col="numerai_meta_model",
        target_col="target",
    )
    expected_down = [f"{i:04d}" for i in range(21, 31)]
    expected_value = float(np.mean([mmc_by_era[e] for e in expected_down]))

    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=None,
        features=features,
        targets=targets,
        n_trials=1,
        seed=7,
    )
    assert score.mmc_down == pytest.approx(expected_value)
    assert score.mmc_down_n_eras == 10
    assert score.mmc_down_reason is None


def test_mmc_down_insufficient() -> None:
    predictions, meta_model, features, targets = _mmc_down_frames(2)
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=None,
        features=features,
        targets=targets,
        n_trials=1,
        seed=7,
    )
    assert score.mmc_down is None
    assert score.mmc_down_n_eras == 2
    assert score.mmc_down_reason == "insufficient_downside_eras"


def _turnover_scorecard_frames() -> tuple[pl.DataFrame, ...]:
    rows: list[dict[str, float | str]] = []
    for i in range(1, 26):
        era = f"{i:04d}"
        for j in range(12):
            pred = 0.1 * i + 0.01 * j
            rows.append(
                {
                    "era": era,
                    "id": f"id{j:03d}",
                    "prediction": pred,
                    "numerai_meta_model": pred * 0.5,
                    "target": float((i + j) % 5) / 4.0,
                    "f1": float(j % 5),
                }
            )
    full = pl.DataFrame(rows)
    return (
        full.select(["era", "id", "prediction"]),
        full.select(["era", "id", "numerai_meta_model"]),
        full.select(["era", "id", "f1"]),
        full.select(["era", "id", "target"]),
    )


def test_turnover_flows_into_scorecard() -> None:
    predictions, meta_model, features, targets = _turnover_scorecard_frames()
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=None,
        features=features,
        targets=targets,
        n_trials=1,
        seed=7,
    )
    # pred is a constant per-era shift -> Spearman rho = 1 every transition
    assert score.turnover_mean == 0.0
    assert score.turnover_std == 0.0
    assert score.turnover_reason is None


def test_capital_metrics_flow_from_payout() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=7,
    )
    engine = EvaluationEngine("custom")
    full = (
        predictions.join(meta_model, on=["era", "id"])
        .join(targets, on=["era", "id"])
    )
    corr = engine.per_era_corr(full, pred_col="prediction", target_col="target")
    mmc = engine.per_era_mmc(
        full, pred_col="prediction", meta_col="numerai_meta_model",
        target_col="target",
    )
    series = payout_series(corr, mmc)
    assert score.cagr_1y == pytest.approx(annual_compounded_return(series.clipped))
    assert score.gain_to_pain_ratio == pytest.approx(
        gain_to_pain_ratio(series.clipped)
    )
    assert score.kelly_fraction == pytest.approx(kelly_fraction(series.raw))
    expected_sim = simulate_overlapping_portfolio(series.clipped, horizon_eras=20)
    assert score.sim_portfolio_cagr == pytest.approx(expected_sim.portfolio_cagr)
    assert score.sim_portfolio_mdd == pytest.approx(
        expected_sim.portfolio_max_drawdown
    )
    assert score.sim_capital_utilization == pytest.approx(
        expected_sim.avg_capital_utilization
    )


def test_final_rank_step_is_scorecard_neutral_except_exposure() -> None:
    """D0 parity proof (SEV-1 #14 fix): per-era ``tie_kept_rank`` of the deploy
    output leaves every rank-invariant scorecard cell identical, and moves
    ``max_feature_exposure`` (the one raw-Pearson cell) by design."""
    rng = np.random.default_rng(3)
    rows: list[dict[str, float | str]] = []
    for era_num in range(1, 11):
        era = f"{era_num:04d}"
        latent = rng.normal(size=40)
        for idx in range(40):
            target = float(np.clip(0.5 + 0.2 * latent[idx], 0.0, 1.0))
            rows.append({
                "era": era, "id": f"{era}_{idx:03d}",
                "prediction": float(0.5 * latent[idx] + 0.5 * rng.normal()),
                "target": target,
                "numerai_meta_model": float(0.55 * target + 0.45 * rng.normal()),
                "f1": float(rng.normal()), "f2": float(rng.normal()),
                "f3": float(rng.normal()),
            })
    full = pl.DataFrame(rows)
    predictions = full.select(["era", "id", "prediction"])
    meta_model = full.select(["era", "id", "numerai_meta_model"])
    targets = full.select(["era", "id", "target"])
    features = full.select(["era", "id", "f1", "f2", "f3"])

    def _per_era_rank(frame: pl.DataFrame) -> pl.DataFrame:
        eras = frame.get_column("era").unique().to_list()
        return pl.concat(
            [
                frame.filter(pl.col("era") == era).with_columns(
                    pl.Series(
                        "prediction",
                        tie_kept_rank(
                            frame.filter(pl.col("era") == era)
                            .get_column("prediction")
                            .to_numpy()
                        ),
                    )
                )
                for era in eras
            ]
        ).sort(["era", "id"])

    def _score(preds: pl.DataFrame) -> MetricScorecard:
        return evaluate_model(
            preds,
            meta_model=meta_model,
            benchmarks=None,
            features=features,
            targets=targets,
            n_trials=1,
            seed=11,
            benchmark_col=None,
            n_boot=10,
            min_overlap_eras=10,
        )

    pre = _score(predictions)
    post = _score(_per_era_rank(predictions))

    invariant = [
        "corr", "mmc", "fnc", "corr_sharpe_ac", "cwmm",
        "std_corr", "turnover_mean", "turnover_std", "cagr_1y",
        "gain_to_pain_ratio", "kelly_fraction", "max_drawdown",
        "sortino", "calmar", "burn_rate", "mean_payout",
        "rank_scalar", "deflated_sharpe", "n_eras",
    ]
    for name in invariant:
        assert getattr(pre, name) == getattr(post, name), f"{name} changed"

    # max_feature_exposure is raw Pearson on the prediction column: the rank
    # step changes it by design (the submitted vector is not feature-neutral,
    # inherent to the official neutralize -> rank approach). Fixture is
    # unsaturated (continuous features) so the delta is measurable.
    assert pre.max_feature_exposure != post.max_feature_exposure
    assert post.max_feature_exposure < 0.99  # not saturated

    # Mathematical core: rank_gaussianize is invariant to a prior
    # tie_kept_rank, so FNC (which rank-gaussianizes first) is unaffected.
    sample = rng.normal(size=200)
    assert np.allclose(
        rank_gaussianize(tie_kept_rank(sample)), rank_gaussianize(sample)
    )


def test_degenerate_eras_are_surfaced_not_silent() -> None:
    """A4 (audit SEV-3): degenerate eras score 0.0 at the engine boundary but
    must be distinguishable from genuine zero-IC eras — the scorecard exposes
    them via degenerate_eras / n_degenerate_eras, with values unchanged."""
    rows: list[dict] = []
    for era_num in range(1, 11):
        era = f"{era_num:04d}"
        constant = era_num == 1
        for j in range(4):
            pred = 0.5 if constant else 0.2 * era_num + 0.03 * j
            rows.append({
                "era": era, "id": f"{era}_{j}",
                "prediction": float(pred),
                "numerai_meta_model": float(0.4 + 0.05 * era_num + 0.03 * j),
                "target": float((era_num + j) % 4) / 4.0,
                "f1": float(j % 2), "f2": float(j % 3),
            })
    full = pl.DataFrame(rows)
    predictions = full.select(["era", "id", "prediction"])
    meta_model = full.select(["era", "id", "numerai_meta_model"])
    targets = full.select(["era", "id", "target"])
    features = full.select(["era", "id", "f1", "f2"])

    score = evaluate_model(
        predictions, meta_model=meta_model, benchmarks=None,
        features=features, targets=targets, n_trials=1, seed=3,
        n_boot=5, min_overlap_eras=10,
    )
    assert score.degenerate_eras == ("0001",)  # zero-variance predictions
    assert score.n_degenerate_eras == 1
    assert score.corr.n_eras == 10  # degenerate era still counted, value 0.0
    frame = score.to_frame()
    assert frame.row(0, named=True)["n_degenerate_eras"] == 1


def test_cross_check_result_shape() -> None:
    """Cross-check output contract: scorecard + labeled per-era series + raw Sharpe.

    ``synthetic_frames`` does not exist in this suite, so the payload comes from
    the local ``_tiny_inputs`` builder (same shape, minus benchmarks).
    """
    predictions, meta_model, _benchmarks, features, targets = _tiny_inputs()
    result = evaluate_cross_check(
        predictions,
        meta_model=meta_model,
        features=features,
        targets=targets,
        horizon="20D",
        main_target="target",
        seed=42,
    )
    assert isinstance(result.scorecard, MetricScorecard)
    assert set(result.per_era) == {"corr", "mmc", "fnc"}
    for era_entry in result.per_era["corr"]:
        assert set(era_entry) == {"era", "value"}
    assert isinstance(result.raw_sharpe, float)
    # raw Sharpe is the PLAIN per-era mean/std (not AC-adjusted) and the series
    # must line up with the scorecard built from the same computation path.
    corr_values = [era_entry["value"] for era_entry in result.per_era["corr"]]
    assert result.raw_sharpe == pytest.approx(
        float(np.mean(corr_values)) / float(np.std(corr_values, ddof=0))
    )
    assert len(result.per_era["corr"]) == result.scorecard.n_eras





def test_cross_check_honors_per_era_pf_mapping() -> None:
    """The cross-check applies the same per-era payout-factor series as the
    research validation path.

    ``_tiny_inputs`` payouts are clip-saturated (raw well beyond +-0.05), so a
    factor ABOVE 1 is invisible in the clipped mean; a uniform PF=0.01 pulls
    the series inside the clip band and must reduce the clipped mean payout
    below the base, proving the mapping reaches the payout computation. The
    empty mapping is the explicit 1.0 fallback (identical to base)."""
    predictions, meta_model, _benchmarks, features, targets = _tiny_inputs()
    base = evaluate_cross_check(
        predictions,
        meta_model=meta_model,
        features=features,
        targets=targets,
        horizon="20D",
        main_target="target",
        seed=42,
    )
    eras = [e["era"] for e in base.per_era["corr"]]
    scaled = evaluate_cross_check(
        predictions,
        meta_model=meta_model,
        features=features,
        targets=targets,
        horizon="20D",
        main_target="target",
        seed=42,
        pf={era: 0.01 for era in eras},
    )
    assert scaled.scorecard.mean_payout.value < base.scorecard.mean_payout.value
    fallback = evaluate_cross_check(
        predictions,
        meta_model=meta_model,
        features=features,
        targets=targets,
        horizon="20D",
        main_target="target",
        seed=42,
        pf={},
    )
    assert fallback.scorecard.mean_payout.value == pytest.approx(
        base.scorecard.mean_payout.value
    )
