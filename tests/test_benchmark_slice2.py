"""Slice 2 (E6) benchmark gates: classical ladder and portfolio orthogonality."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import BenchmarkSuite
from nmr.evaluation import MIN_OVERLAP_ERAS


def _slice2_inputs(
    *,
    n_eras: int = 120,
    rows_per_era: int = 20,
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
            nonlinear = (f1 * f2) + np.sin(f3)
            latent = (0.55 * f1) - (0.30 * f2) + (0.20 * f3) + (0.25 * nonlinear)
            latent += float(rng.normal(0.0, 0.35))
            target = float(np.clip(0.5 + 0.23 * latent, 0.0, 1.0))
            meta = float(0.60 * target + 0.40 * rng.random())

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
                    "v52_lgbm_cyrusd20": float(0.55 * target + 0.45 * rng.random()),
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
    meta_model, benchmarks, features, targets = _slice2_inputs()
    return BenchmarkSuite(
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=seed,
        benchmark_col="v52_lgbm_cyrusd20",
        n_boot=80,
        min_overlap_eras=20,
    )


def test_slice2_extended_monotone_sanity_ladder() -> None:
    suite = _suite(seed=17)

    scores = suite.run_null_baselines(seed=17)
    scores.update(suite.run_classical_baselines(min_train_eras=12))

    null_floor = max(
        float(scores[name].rank_scalar)
        for name in ("constant-0.5", "uniform-random", "gaussian-random")
    )
    trivial = float(scores["trivial"].rank_scalar)
    linear = float(scores["linear"].rank_scalar)
    tree = float(scores["tree"].rank_scalar)

    assert (
        null_floor <= trivial + 1e-3
    ), f"Null floor violation: null_floor={null_floor:.6f}, trivial={trivial:.6f}"
    assert (
        trivial <= linear + 1e-3
    ), f"Ladder violation: trivial={trivial:.6f}, linear={linear:.6f}"
    assert (
        linear <= tree + 1e-3
    ), f"Ladder violation: linear={linear:.6f}, tree={tree:.6f}"


def test_slice2_book_orthogonality_non_vacuity_guard() -> None:
    suite = _suite(seed=23)

    n = MIN_OVERLAP_ERAS - 1
    candidate = np.linspace(-0.1, 0.1, n)
    book = np.linspace(0.2, -0.2, n)

    with pytest.raises(ValueError, match="Non-vacuity violation"):
        suite.compute_book_orthogonality(
            candidate,
            book,
            seed=23,
            n_boot=40,
        )


def test_slice2_book_orthogonality_cross_process_determinism() -> None:
    code = r"""
import json
import numpy as np
import polars as pl
from nmr.benchmark import BenchmarkSuite

rng = np.random.default_rng(20260622)
rows = []
for era_num in range(1, 81):
    era = f"{era_num:04d}"
    for idx in range(8):
        f1 = float(rng.normal())
        f2 = float(rng.normal())
        f3 = float(rng.normal())
        nonlinear = (f1 * f2) + np.sin(f3)
        latent = (0.55 * f1) - (0.30 * f2) + (0.20 * f3) + (0.25 * nonlinear) + float(rng.normal(0.0, 0.35))
        target = float(np.clip(0.5 + 0.23 * latent, 0.0, 1.0))
        meta = float(0.60 * target + 0.40 * rng.random())
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
                "v52_lgbm_cyrusd20": float(0.55 * target + 0.45 * rng.random()),
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
    n_boot=40,
    min_overlap_eras=20,
)

n = 80
book = rng.normal(0.0, 1.0, n)
candidate = 0.35 * book + rng.normal(0.0, 0.85, n)

out = suite.compute_book_orthogonality(candidate, book, seed=999, n_boot=150)
print(json.dumps(out, sort_keys=True))
"""

    cmd = [sys.executable, "-c", code]
    run1 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    run2 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert run1.stdout.strip() == run2.stdout.strip()
