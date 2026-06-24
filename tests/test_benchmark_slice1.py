"""Slice 1 (E6) benchmark gates: null floors, tutorial ingestion, determinism."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from nmr.benchmark import (
    BenchmarkSuite,
    assert_notebook_prediction_contract,
    assert_null_floor,
    assert_slice1_monotone,
    discover_tutorial_notebooks,
    extract_oos_predictions,
    scorecards_sha256,
    scorecards_to_frame,
    write_scorecards_csv,
)


def _slice1_inputs(
    *,
    n_eras: int = 80,
    rows_per_era: int = 20,
    seed: int = 20260622,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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
                (0.8 * f1) - (0.35 * f2) + (0.15 * f3) + float(rng.normal(0.0, 0.7))
            )
            target = float(np.clip(0.5 + 0.2 * latent, 0.0, 1.0))
            meta = float(0.55 * target + 0.45 * rng.random())
            rows.append(
                {
                    "era": era,
                    "id": asset_id,
                    "prediction": float(rng.random()),
                    "numerai_meta_model": meta,
                    "target": target,
                    "target_cyrusd_20": target,
                    "target_cyrusd_60": float(
                        np.clip(target + rng.normal(0.0, 0.04), 0.0, 1.0)
                    ),
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "v52_lgbm_cyrusd20": float(0.6 * target + 0.4 * rng.random()),
                }
            )

    full = pl.DataFrame(rows)
    predictions = full.select(["era", "id", "prediction"])
    meta_model = full.select(["era", "id", "numerai_meta_model"])
    benchmarks = full.select(["era", "id", "v52_lgbm_cyrusd20"])
    features = full.select(["era", "id", "f1", "f2", "f3"])
    targets = full.select(
        ["era", "id", "target", "target_cyrusd_20", "target_cyrusd_60"]
    )
    return predictions, meta_model, benchmarks, features, targets


def _suite(seed: int = 77) -> tuple[BenchmarkSuite, pl.DataFrame, pl.DataFrame]:
    predictions, meta_model, benchmarks, features, targets = _slice1_inputs()
    suite = BenchmarkSuite(
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=seed,
        benchmark_col="v52_lgbm_cyrusd20",
        n_boot=50,
        min_overlap_eras=20,
    )
    return suite, predictions, targets


def test_tutorial_notebook_contracts_present() -> None:
    notebooks = discover_tutorial_notebooks(Path("docs/05-notebooks"))
    assert set(notebooks) == {
        "1_hello_numerai.ipynb",
        "2_feature_neutralization.ipynb",
        "example-model-sunshine.ipynb",
    }
    for path in notebooks.values():
        assert_notebook_prediction_contract(path)


def test_extract_oos_predictions_handles_index_style_csv(tmp_path: Path) -> None:
    raw_path = tmp_path / "validation_predictions_999.csv"
    raw_path.write_text(
        "Unnamed: 0,prediction\n" "id_a,0.11\n" "id_b,0.89\n",
        encoding="utf-8",
    )

    id_to_era = pl.DataFrame(
        {
            "id": ["id_a", "id_b"],
            "era": ["0001", "0001"],
        }
    )

    out = extract_oos_predictions(raw_path, id_to_era=id_to_era)
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == 2
    assert out.get_column("id").to_list() == ["id_a", "id_b"]


def test_slice1_null_floor_gate_and_finite_scorecards() -> None:
    suite, _, _ = _suite(seed=91)
    null_scores = suite.run_null_baselines(seed=202)

    assert_null_floor(
        null_scores,
        tolerance=0.04,
        metric_tolerances={"corr_sharpe_ac": 0.12},
    )

    for model_id, card in null_scores.items():
        row = card.to_frame().row(0, named=True)
        for value in row.values():
            if isinstance(value, float):
                assert math.isfinite(value), f"non-finite at {model_id}: {value}"


def test_slice1_monotone_sanity_null_hello_sunshine() -> None:
    suite, _, targets = _suite(seed=123)
    null_scores = suite.run_null_baselines(seed=123)

    joined = targets.select(["era", "id", "target"])
    rng = np.random.default_rng(999)

    hello = joined.with_columns(
        (pl.col("target") + pl.Series("n", rng.normal(0.0, 0.40, joined.height))).alias(
            "prediction"
        )
    ).select(["era", "id", "prediction"])

    sunshine = joined.with_columns(
        (pl.col("target") + pl.Series("n", rng.normal(0.0, 0.15, joined.height))).alias(
            "prediction"
        )
    ).select(["era", "id", "prediction"])

    scores = dict(null_scores)
    scores["hello"] = suite.evaluate_predictions(hello, model_id="hello", seed=3)
    scores["sunshine"] = suite.evaluate_predictions(
        sunshine, model_id="sunshine", seed=3
    )

    assert_slice1_monotone(
        scores,
        hello_model_id="hello",
        sunshine_model_id="sunshine",
        atol=1e-12,
    )


def test_slice1_cross_process_determinism_hash() -> None:
    code = r"""
import json
import numpy as np
import polars as pl
from nmr.benchmark import BenchmarkSuite, scorecards_sha256

rng = np.random.default_rng(20260622)
rows = []
for era_num in range(1, 41):
    era = f"{era_num:04d}"
    for idx in range(8):
        f1 = float(rng.normal())
        f2 = float(rng.normal())
        f3 = float(rng.normal())
        latent = (0.8 * f1) - (0.35 * f2) + (0.15 * f3) + float(rng.normal(0.0, 0.7))
        target = float(np.clip(0.5 + 0.2 * latent, 0.0, 1.0))
        meta = float(0.55 * target + 0.45 * rng.random())
        rows.append(
            {
                "era": era,
                "id": f"{era}_{idx:03d}",
                "prediction": float(rng.random()),
                "numerai_meta_model": meta,
                "target": target,
                "target_cyrusd_20": target,
                "target_cyrusd_60": float(np.clip(target + rng.normal(0.0, 0.04), 0.0, 1.0)),
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "v52_lgbm_cyrusd20": float(0.6 * target + 0.4 * rng.random()),
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
    n_boot=30,
    min_overlap_eras=20,
)
print(scorecards_sha256(suite.run_null_baselines(seed=77)))
"""

    cmd = [sys.executable, "-c", code]
    run1 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    run2 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert run1.stdout.strip() == run2.stdout.strip()


def test_slice1_scorecards_csv_persistence(tmp_path: Path) -> None:
    suite, predictions, _ = _suite(seed=55)

    scores = suite.run_null_baselines(seed=55)
    scores["candidate"] = suite.evaluate_predictions(
        predictions,
        model_id="candidate",
        seed=55,
    )

    frame = scorecards_to_frame(scores)
    out_path = write_scorecards_csv(
        scores, tmp_path / "artifacts" / "benchmark_scores.csv"
    )

    assert out_path == tmp_path / "artifacts" / "benchmark_scores.csv"
    assert out_path.exists()

    persisted = pl.read_csv(out_path)
    assert persisted.columns == frame.columns
    assert persisted.get_column("model_id").to_list() == sorted(scores)
    assert persisted.height == len(scores)
    assert persisted.sort("model_id").to_dicts() == frame.sort("model_id").to_dicts()
