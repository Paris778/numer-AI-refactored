"""Slice 3 (E6) gates: BMC/CWMM uniqueness and overlap hardening."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest
from numerai_tools.scoring import correlation_contribution

from nmr.benchmark import BenchmarkSuite, assert_null_floor, scorecards_sha256
from nmr.evaluation import EvaluationEngine


def _slice3_inputs(
    *,
    n_eras: int = 260,
    rows_per_era: int = 36,
    seed: int = 20260622,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []

    for era_num in range(1, n_eras + 1):
        era = f"{era_num:04d}"
        for idx in range(rows_per_era):
            asset_id = f"{era}_{idx:03d}"
            f1 = float(rng.normal())
            f2 = float(rng.normal())
            f3 = float(rng.normal())
            latent = (
                (0.7 * f1) - (0.32 * f2) + (0.19 * f3) + float(rng.normal(0.0, 0.45))
            )
            target = float(np.clip(0.5 + 0.22 * latent, 0.0, 1.0))
            meta = float(0.58 * target + 0.42 * rng.random())
            benchmark = float(0.54 * target + 0.46 * rng.random())

            rows.append(
                {
                    "era": era,
                    "id": asset_id,
                    "numerai_meta_model": meta,
                    "target": target,
                    "target_cyrusd_20": target,
                    "target_cyrusd_60": float(
                        np.clip(target + rng.normal(0.0, 0.03), 0.0, 1.0)
                    ),
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "v52_lgbm_cyrusd20": benchmark,
                }
            )

    full = pl.DataFrame(rows)
    meta_model = full.select(["era", "id", "numerai_meta_model"])
    benchmarks = full.select(["era", "id", "v52_lgbm_cyrusd20"])
    features = full.select(["era", "id", "f1", "f2", "f3"])
    targets = full.select(
        ["era", "id", "target", "target_cyrusd_20", "target_cyrusd_60"]
    )
    return meta_model, benchmarks, features, targets


def _suite(seed: int = 77) -> BenchmarkSuite:
    meta_model, benchmarks, features, targets = _slice3_inputs()
    return BenchmarkSuite(
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=seed,
        benchmark_col="v52_lgbm_cyrusd20",
        n_boot=100,
        min_overlap_eras=20,
    )


def test_slice3_null_flooring_bmc_cwmm() -> None:
    suite = _suite(seed=11)
    null_scores = suite.run_null_baselines(seed=11)

    assert_null_floor(
        null_scores,
        tolerance=0.02,
        metric_tolerances={"corr_sharpe_ac": 0.12},
    )

    for name, card in null_scores.items():
        assert card.bmc is not None, f"missing bmc for {name}"
        assert card.cwmm is not None, f"missing cwmm for {name}"
        assert abs(float(card.bmc.value)) <= 0.02
        assert abs(float(card.cwmm.value)) <= 0.02


def test_slice3_bmc_oracle_parity() -> None:
    meta_model, benchmarks, features, targets = _slice3_inputs(
        n_eras=40, rows_per_era=20
    )
    cfg_pred = targets.select(["era", "id"]).with_columns(
        (0.7 * pl.col("id").cum_count()).cast(pl.Float64).alias("prediction")
    )

    base = (
        cfg_pred.join(meta_model, on=["era", "id"], how="inner")
        .join(targets.select(["era", "id", "target"]), on=["era", "id"], how="inner")
        .join(features, on=["era", "id"], how="inner")
        .join(benchmarks, on=["era", "id"], how="left")
    )

    evaluator = EvaluationEngine("custom")
    per_era = evaluator.per_era_bmc(
        base,
        pred_col="prediction",
        benchmark_col="v52_lgbm_cyrusd20",
        target_col="target",
        min_overlap_eras=20,
    )

    eras = sorted(per_era, key=int)
    one_era = eras[0]
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

    assert per_era[one_era] == pytest.approx(direct, abs=1e-6)


def test_slice3_hard_non_vacuity_exception_gate() -> None:
    rng = np.random.default_rng(123)
    eras = [f"{i:04d}" for i in range(1, 21)]

    rows: list[dict[str, float | str]] = []
    for era in eras:
        for idx in range(6):
            rows.append(
                {
                    "era": era,
                    "prediction": float(rng.random()),
                    "target": float(rng.random()),
                    "meta": float(rng.random()),
                    "bench": float(rng.random()) if int(era) <= 15 else float("nan"),
                }
            )
    frame = pl.DataFrame(rows)

    evaluator = EvaluationEngine("custom")
    with pytest.raises(ValueError, match="Non-vacuity violation"):
        evaluator.per_era_bmc(
            frame,
            pred_col="prediction",
            benchmark_col="bench",
            target_col="target",
            min_overlap_eras=20,
        )

    with pytest.raises(ValueError, match="Non-vacuity violation"):
        evaluator.per_era_cwmm(
            frame.with_columns(
                pl.when(pl.col("era").cast(pl.Int64) <= 15)
                .then(pl.col("meta"))
                .otherwise(pl.lit(None))
                .alias("meta_sparse")
            ),
            pred_col="prediction",
            meta_col="meta_sparse",
            min_overlap_eras=20,
        )


def test_slice3_cross_process_determinism_bmc_cwmm_bytes() -> None:
    code = r"""
import numpy as np
import polars as pl
from nmr.benchmark import BenchmarkSuite, scorecards_sha256

rng = np.random.default_rng(20260622)
rows = []
for era_num in range(1, 81):
    era = f"{era_num:04d}"
    for idx in range(10):
        f1 = float(rng.normal())
        f2 = float(rng.normal())
        f3 = float(rng.normal())
        latent = (0.7 * f1) - (0.32 * f2) + (0.19 * f3) + float(rng.normal(0.0, 0.45))
        target = float(np.clip(0.5 + 0.22 * latent, 0.0, 1.0))
        meta = float(0.58 * target + 0.42 * rng.random())
        benchmark = float(0.54 * target + 0.46 * rng.random())
        rows.append(
            {
                "era": era,
                "id": f"{era}_{idx:03d}",
                "numerai_meta_model": meta,
                "target": target,
                "target_cyrusd_20": target,
                "target_cyrusd_60": float(np.clip(target + rng.normal(0.0, 0.03), 0.0, 1.0)),
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "v52_lgbm_cyrusd20": benchmark,
            }
        )

full = pl.DataFrame(rows)
suite = BenchmarkSuite(
    meta_model=full.select(["era", "id", "numerai_meta_model"]),
    benchmarks=full.select(["era", "id", "v52_lgbm_cyrusd20"]),
    features=full.select(["era", "id", "f1", "f2", "f3"]),
    targets=full.select(["era", "id", "target", "target_cyrusd_20", "target_cyrusd_60"]),
    n_trials=1,
    seed=77,
    benchmark_col="v52_lgbm_cyrusd20",
    n_boot=60,
    min_overlap_eras=20,
)

scores = suite.run_null_baselines(seed=77)
scores.update(suite.run_classical_baselines(min_train_eras=10))
print(scorecards_sha256(scores))
"""

    cmd = [sys.executable, "-c", code]
    run1 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    run2 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert run1.stdout.strip() == run2.stdout.strip()
