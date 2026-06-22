"""Tests for scorecard aggregation (E5)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from numerai_tools.scoring import correlation_contribution

from nmr.evaluation import EvaluationEngine
from nmr.inference import block_bootstrap_ci, era_series_stats, resolve_block_len
from nmr.payout import payout_report
from nmr.scorecard import MetricScorecard, evaluate_model


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


@pytest.mark.skipif(
    not (
        Path("data/v5.2/validation.parquet").exists()
        and Path("data/v5.2/validation_benchmark_models.parquet").exists()
        and Path("data/v5.2/meta_model.parquet").exists()
    ),
    reason="v5.2 inputs not on disk; skipped in CI",
)
def test_scorecard_real_v52_determinism_cross_process() -> None:
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
print(json.dumps(card.to_frame().to_dicts()[0], sort_keys=True, default=str))
"""
    cmd = [sys.executable, "-c", code]
    run1 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    run2 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert run1.stdout.strip() == run2.stdout.strip()
