"""End-to-end hierarchy orchestration, determinism, and monotonicity."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import (
    BenchmarkCellConfig,
    BenchmarkData,
    BenchmarkHierarchy,
    BenchmarkSuiteSpec,
    Tier4GateConfig,
    gate_report_frame,
    hierarchy_frame,
    scorecards_sha256,
)


def _data_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(20260815)
    # Stable asset ids across eras (>= 10 per era) so per-era prediction-rank
    # turnover is computable for the tier-4 reference. A None turnover is not a
    # hard failure — it is reported as measured=None/pass=None in the gate
    # report and excluded from `assert_tier4_gate` (v5.3 consecutive validation
    # eras share zero ids, so turnover is structurally unavailable there).
    n_eras, rows_per_era = 60, 12
    rows = []
    for era_num in range(1, n_eras + 1):
        era = f"{era_num:04d}"
        # A few eras invert the signal so the synthetic payout path records
        # genuine drawdowns; with all-positive per-era payouts the max
        # drawdown is exactly 0.0 and calmar (mean / max_drawdown) becomes
        # inf, which the tier-4 gate's finiteness pre-check rejects.
        signal_is_target = era_num % 5 != 0
        for idx in range(rows_per_era):
            f1 = float(rng.normal())
            target = float(np.clip(0.5 + 0.2 * f1 + rng.normal(0, 0.3), 0, 1))
            signal = target if signal_is_target else 1.0 - target
            rows.append({
                "era": era, "id": f"asset_{idx}", "f1": f1,
                "target": target,
                "numerai_meta_model": float(0.5 * signal + 0.5 * rng.random()),
                "bench": float(0.6 * signal + 0.4 * rng.random()),
            })
    frame = pl.DataFrame(rows)

    train = frame.filter(pl.col("era").is_in([f"{e:04d}" for e in range(1, 49)]))
    val = frame.filter(pl.col("era").is_in([f"{e:04d}" for e in range(49, 61)]))
    train.write_parquet(tmp_path / "train.parquet")
    val.write_parquet(tmp_path / "validation.parquet")
    val.select(["era", "id", "numerai_meta_model"]).write_parquet(
        tmp_path / "meta_model.parquet"
    )
    val.select(["era", "id", "bench"]).rename({"bench": "v53_lgbm_ender60"}).write_parquet(
        tmp_path / "validation_benchmark_models.parquet"
    )
    (tmp_path / "features.json").write_text(
        '{"feature_sets": {"small": ["f1"], "medium": ["f1"]}}',
        encoding="utf-8",
    )
    return tmp_path


def _spec() -> BenchmarkSuiteSpec:
    gate = Tier4GateConfig(
        corr_min=-1.0, corr_sharpe_ac_min=-10.0, fnc_min=-1.0,
        deflated_sharpe_min=-10.0, gain_to_pain_min=-10.0, cagr_min=-1.0,
        turnover_max=10.0,
    )
    cells = (
        BenchmarkCellConfig(
            benchmark_id="null_constant_05", input_space="none",
            model_kind="null_constant_05", tier=0,
        ),
        BenchmarkCellConfig(
            benchmark_id="linear_ridge_small", input_space="small",
            model_kind="ridge", tier=1,
            targets=("target",), params={"alpha": 1.0},
        ),
        BenchmarkCellConfig(
            benchmark_id="tree_lgbm_shallow_small", input_space="small",
            model_kind="lightgbm", tier=2,
            targets=("target",),
            params={"n_estimators": 5, "learning_rate": 0.1,
                    "max_depth": 2, "num_leaves": 4, "colsample_bytree": 0.5},
        ),
        BenchmarkCellConfig(
            benchmark_id="canon_hello_numerai", input_space="small",
            model_kind="lightgbm", tier=3,
            targets=("target",),
            params={"n_estimators": 5, "learning_rate": 0.1,
                    "max_depth": 2, "num_leaves": 4, "colsample_bytree": 0.5},
        ),
    )
    return BenchmarkSuiteSpec(
        cells=cells, gate=gate, reference_column="v53_lgbm_ender60"
    )


def _run(tmp_path: Path, *, seed: int = 42) -> BenchmarkHierarchy:
    from nmr.benchmark import load_benchmark_data
    data = load_benchmark_data(_data_dir(tmp_path))
    hierarchy = BenchmarkHierarchy(
        spec=_spec(), data=data, seed=seed, n_boot=50,
        min_overlap_eras=5, fast_mode=True,
    )
    return hierarchy


def test_hierarchy_runs_and_emits_frames(tmp_path: Path) -> None:
    result = _run(tmp_path).run()
    expected_ids = {
        "null_constant_05", "linear_ridge_small",
        "tree_lgbm_shallow_small", "canon_hello_numerai",
        "v53_lgbm_ender60",
    }
    assert set(result.scorecards) == expected_ids
    assert result.tier_of["v53_lgbm_ender60"] == 4
    frame = hierarchy_frame(result)
    assert frame.height == 5
    assert "strategy_group" in frame.columns
    assert set(frame.get_column("strategy_group").unique().to_list()) == {
        "tier0", "tier1", "tier2", "tier3", "tier4",
    }
    report = gate_report_frame(result)
    assert report.height == 7
    assert set(report.get_column("field").to_list()) == {
        "corr", "corr_sharpe_ac", "fnc", "deflated_sharpe",
        "gain_to_pain_ratio", "cagr_1y", "turnover_mean",
    }


def test_hierarchy_is_deterministic(tmp_path: Path) -> None:
    result_a = _run(tmp_path, seed=42).run()
    result_b = _run(tmp_path, seed=42).run()
    assert scorecards_sha256(result_a.scorecards) == scorecards_sha256(
        result_b.scorecards
    )


def test_hierarchy_cross_process_determinism(tmp_path: Path) -> None:
    data_dir = _data_dir(tmp_path)
    script = (
        "import os, sys; from pathlib import Path;"
        "from nmr.benchmark import load_benchmark_data, BenchmarkHierarchy, "
        "BenchmarkCellConfig, BenchmarkSuiteSpec, Tier4GateConfig, scorecards_sha256;"
        f"data = load_benchmark_data(Path(r'{data_dir}'));"
        "gate = Tier4GateConfig(corr_min=-1, corr_sharpe_ac_min=-10, fnc_min=-1, "
        "deflated_sharpe_min=-10, gain_to_pain_min=-10, cagr_min=-1, turnover_max=10);"
        "cells = (BenchmarkCellConfig('null_constant_05', 'none', "
        "'null_constant_05', 0), "
        "BenchmarkCellConfig('linear_ridge_small', 'small', 'ridge', 1, "
        "targets=('target',), params={'alpha': 1.0}));"
        "spec = BenchmarkSuiteSpec(cells=cells, gate=gate, "
        "reference_column='v53_lgbm_ender60');"
        "h = BenchmarkHierarchy(spec=spec, data=data, seed=42, n_boot=50, "
        "min_overlap_eras=5, fast_mode=True);"
        "print(scorecards_sha256(h.run().scorecards))"
    )
    env = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    run = lambda: subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=env, cwd=Path.cwd(), check=True,
    ).stdout.strip()
    assert run() == run()


def test_monotone_failure_surfaces_in_result(tmp_path: Path) -> None:
    hierarchy = _run(tmp_path)
    result = hierarchy.run()
    # The four real tiers may or may not order monotonically on this synthetic
    # data; the result must carry verdicts either way (no raise).
    assert isinstance(result.monotone_ok, bool)
    assert isinstance(result.null_floor_ok, bool)
    assert result.tier4_violations == ()
