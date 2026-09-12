"""Tier-1 ridge benchmark generator contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import (
    _feature_block_statistics,
    _standardize_feature_block,
    estimate_ridge_peak_bytes,
    generate_ridge_predictions,
    ridge_memory_budget,
)


def _domain(
    *,
    n_train_eras: int = 30,
    n_val_eras: int = 10,
    rows_per_era: int = 12,
    seed: int = 20260815,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    feature_cols = ["f1", "f2", "f3", "f_const"]

    def make(eras: range) -> list[dict[str, float | str]]:
        rows = []
        for era_num in eras:
            era = f"{era_num:04d}"
            for idx in range(rows_per_era):
                f1, f2, f3 = (float(rng.normal()) for _ in range(3))
                target = float(
                    np.clip(
                        0.5
                        + 0.2 * (0.8 * f1 - 0.4 * f2 + 0.2 * f3)
                        + rng.normal(0.0, 0.3),
                        0.0,
                        1.0,
                    )
                )
                aux = float(np.clip(target + rng.normal(0.0, 0.2), 0.0, 1.0))
                rows.append(
                    {
                        "era": era,
                        "id": f"{era}_{idx}",
                        "f1": f1,
                        "f2": f2,
                        "f3": f3,
                        "f_const": 1.0,
                        "target": target,
                        "aux": aux,
                    }
                )
        return rows

    train = pl.DataFrame(make(range(1, n_train_eras + 1)))
    val = pl.DataFrame(make(range(n_train_eras + 1, n_train_eras + 1 + n_val_eras)))
    val = val.select(["era", "id", *feature_cols])
    return train, val, feature_cols


def test_single_target_ridge_covers_val_and_is_finite() -> None:
    train, val, feats = _domain()
    out = generate_ridge_predictions(
        train,
        val,
        targets=["target"],
        feature_cols=feats,
        alpha=1.0,
        seed=42,
        purge_eras=8,
    )
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == val.height
    assert set(out.get_column("era").unique().to_list()) == set(
        val.get_column("era").unique().to_list()
    )
    assert out.get_column("prediction").is_finite().all()


def test_ridge_is_deterministic_per_seed() -> None:
    train, val, feats = _domain()
    a = generate_ridge_predictions(
        train,
        val,
        targets=["target"],
        feature_cols=feats,
        alpha=1.0,
        seed=42,
        purge_eras=8,
    )
    b = generate_ridge_predictions(
        train,
        val,
        targets=["target"],
        feature_cols=feats,
        alpha=1.0,
        seed=42,
        purge_eras=8,
    )
    assert a.equals(b)


def test_zero_variance_feature_does_not_produce_nan() -> None:
    train, val, feats = _domain()
    out = generate_ridge_predictions(
        train,
        val,
        targets=["target"],
        feature_cols=feats,
        alpha=1.0,
        seed=42,
        purge_eras=8,
    )
    values = out.get_column("prediction").to_numpy()
    assert np.isfinite(values).all()


def test_per_era_output_is_rank_gaussianized() -> None:
    train, val, feats = _domain()
    out = generate_ridge_predictions(
        train,
        val,
        targets=["target"],
        feature_cols=feats,
        alpha=1.0,
        seed=42,
        purge_eras=8,
    )
    per_era = out.group_by("era").agg(pl.col("prediction"))
    for era_df in per_era.iter_rows(named=True):
        values = np.asarray(era_df["prediction"], dtype=float)
        assert abs(float(np.mean(values))) < 1e-8
        assert abs(float(np.std(values, ddof=0)) - 1.0) < 1e-6


def test_multitarget_nan_masking_and_blend() -> None:
    train, val, feats = _domain()
    # Poison some aux-target rows with nulls (watchpoint 1: independent masking)
    poisoned = train.with_columns(
        pl.when(pl.col("era") == "0001")
        .then(None)
        .otherwise(pl.col("aux"))
        .alias("aux")
    )
    out = generate_ridge_predictions(
        poisoned,
        val,
        targets=["target", "aux"],
        feature_cols=feats,
        alpha=1.0,
        seed=42,
        purge_eras=8,
    )
    assert out.height == val.height
    assert np.isfinite(out.get_column("prediction").to_numpy()).all()


def test_tight_gap_raises() -> None:
    train, val, feats = _domain(n_train_eras=30, n_val_eras=10)
    close_val = val.with_columns(pl.lit("0032").alias("era"))
    with pytest.raises(ValueError, match="purge"):
        generate_ridge_predictions(
            train,
            close_val,
            targets=["target"],
            feature_cols=feats,
            alpha=1.0,
            seed=42,
            purge_eras=8,
        )


def test_standardize_feature_block_keeps_float32() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(100, 5)).astype(np.float32)
    val = rng.normal(size=(50, 5)).astype(np.float32)
    out_train, out_val = _standardize_feature_block(train, val)
    assert out_train.dtype == np.float32
    assert out_val.dtype == np.float32


def test_standardize_feature_block_matches_float64_reference() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(100, 5))
    val = rng.normal(size=(50, 5))
    train32, val32 = train.astype(np.float32), val.astype(np.float32)

    mu = np.where(np.isfinite(train.mean(axis=0)), train.mean(axis=0), 0.0)
    sigma = train.std(axis=0)
    scale = np.where((sigma > 0.0) & np.isfinite(sigma), 1.0 / sigma, 0.0)
    expected_train = (train - mu) * scale
    expected_val = (val - mu) * scale

    got_train, got_val = _standardize_feature_block(train32, val32)
    assert np.allclose(got_train, expected_train, rtol=1e-5, atol=1e-6)
    assert np.allclose(got_val, expected_val, rtol=1e-5, atol=1e-6)


def test_standardize_feature_block_zero_variance_column_is_zero() -> None:
    train = np.array([[1.0, 2.0, 5.0], [3.0, 2.0, 7.0]], dtype=np.float32)
    val = np.array([[4.0, 2.0, 9.0]], dtype=np.float32)
    got_train, got_val = _standardize_feature_block(train, val)
    assert np.all(np.isfinite(got_train))
    assert np.all(np.isfinite(got_val))
    assert np.all(got_train[:, 1] == 0.0)
    assert np.all(got_val[:, 1] == 0.0)


def test_standardize_feature_block_uses_train_statistics() -> None:
    # val with a different mean must be centered by the TRAIN mean, not its own.
    train = np.array([[0.0], [10.0]], dtype=np.float32)
    val = np.array([[100.0], [110.0]], dtype=np.float32)
    got_train, got_val = _standardize_feature_block(train, val)
    # train mean 5, std 5 -> standardized train = [-1, 1]
    assert np.allclose(got_train.ravel(), [-1.0, 1.0], atol=1e-6)
    # val centered by TRAIN mean 5 and scaled by 1/5 -> [19, 21]
    assert np.allclose(got_val.ravel(), [19.0, 21.0], atol=1e-5)


# ---------------------------------------------------------------------------
# D1a: memory-safe sklearn-preserving ridge path (2026-09-08)
# ---------------------------------------------------------------------------


def test_estimate_ridge_peak_bytes_terms() -> None:
    terms = estimate_ridge_peak_bytes(
        n_train_rows=1000, n_val_rows=2000, n_val_eras=10, n_features=50
    )
    assert terms["caller_frames_bytes"] == (1000 + 2000) * 50
    assert terms["fit_matrix_bytes"] == 1000 * 50 * 4
    assert terms["val_matrix_bytes"] == 2000 * 50 * 4
    assert terms["val_upcast_bytes"] == 2000 * 50 * 8
    assert terms["fit_block_temp_bytes"] > 0
    assert terms["val_block_temp_bytes"] > 0
    assert terms["stats_transient_bytes"] > 0
    assert terms["overhead_bytes"] > 0
    assert terms["safety_factor"] >= 1.0
    peak_base = (
        terms["caller_frames_bytes"]
        + max(
            terms["fit_matrix_bytes"] + terms["fit_block_temp_bytes"],
            terms["val_matrix_bytes"]
            + terms["val_upcast_bytes"]
            + terms["val_block_temp_bytes"],
        )
        + terms["overhead_bytes"]
    )
    assert terms["peak_commit_bytes"] == int(peak_base * terms["safety_factor"] + 0.5)
    assert terms["peak_ws_bytes"] == terms["peak_commit_bytes"]


def test_estimate_ridge_peak_bytes_rejects_degenerate_shapes() -> None:
    with pytest.raises(ValueError):
        estimate_ridge_peak_bytes(
            n_train_rows=0, n_val_rows=1, n_val_eras=1, n_features=1
        )


def test_ridge_memory_budget_allows_corrected_medium_cell() -> None:
    budget = ridge_memory_budget(
        "linear_ridge_medium",
        n_train_rows=2_701_335,
        n_val_rows=4_107_040,
        n_val_eras=657,
        n_features=780,
        machine_commit_limit_bytes=round(148.8 * 2**30),
        machine_physical_bytes=round(63.7 * 2**30),
    )
    assert budget.verdict == "allow"
    assert budget.reasons == ()
    assert budget.peak_commit_bytes < budget.commit_ceiling_bytes
    assert budget.peak_commit_bytes < budget.machine_commit_limit_bytes
    assert budget.peak_ws_bytes < budget.machine_physical_bytes * budget.ws_fraction
    assert budget.cell_id == "linear_ridge_medium"


def test_ridge_memory_budget_refuses_over_configured_ceiling() -> None:
    budget = ridge_memory_budget(
        "over_ceiling",
        n_train_rows=2_701_335,
        n_val_rows=4_107_040,
        n_val_eras=657,
        n_features=780,
        machine_commit_limit_bytes=round(148.8 * 2**30),
        machine_physical_bytes=round(63.7 * 2**30),
        commit_ceiling_bytes=1,
    )
    assert budget.verdict == "refuse"
    assert any("ceiling" in reason for reason in budget.reasons)


def test_ridge_memory_budget_refuses_over_machine_commit_limit() -> None:
    budget = ridge_memory_budget(
        "over_commit_limit",
        n_train_rows=2_701_335,
        n_val_rows=4_107_040,
        n_val_eras=657,
        n_features=780,
        machine_commit_limit_bytes=1,
        machine_physical_bytes=round(63.7 * 2**30),
    )
    assert budget.verdict == "refuse"
    assert any("commit limit" in reason for reason in budget.reasons)


def test_ridge_memory_budget_refuses_working_set_guard() -> None:
    budget = ridge_memory_budget(
        "over_ws_guard",
        n_train_rows=2_701_335,
        n_val_rows=4_107_040,
        n_val_eras=657,
        n_features=780,
        machine_commit_limit_bytes=round(148.8 * 2**30),
        machine_physical_bytes=1,
    )
    assert budget.verdict == "refuse"
    assert any("working set" in reason for reason in budget.reasons)


def test_ridge_memory_budget_warns_when_machine_limits_unknown() -> None:
    budget = ridge_memory_budget(
        "unknown_limits",
        n_train_rows=1000,
        n_val_rows=2000,
        n_val_eras=10,
        n_features=50,
    )
    assert budget.verdict == "warn"
    assert any("unavailable" in reason for reason in budget.reasons)


def test_generate_ridge_predictions_refuses_before_materializing(
    monkeypatch,
) -> None:
    train, val, feats = _domain()
    monkeypatch.setattr("nmr.benchmark._RIDGE_COMMIT_CEILING_BYTES", 1)
    monkeypatch.delenv("NMR_RIDGE_COMMIT_CEILING_BYTES", raising=False)
    with pytest.raises(ValueError, match="memory budget"):
        generate_ridge_predictions(
            train,
            val,
            targets=["target"],
            feature_cols=feats,
            alpha=1.0,
            seed=42,
            purge_eras=8,
        )


def test_ridge_commit_ceiling_env_override(monkeypatch) -> None:
    from nmr.benchmark import _ridge_commit_ceiling_bytes

    monkeypatch.setenv("NMR_RIDGE_COMMIT_CEILING_BYTES", "123456")
    assert _ridge_commit_ceiling_bytes() == 123456
    monkeypatch.setenv("NMR_RIDGE_COMMIT_CEILING_BYTES", "not-an-int")
    with pytest.raises(ValueError):
        _ridge_commit_ceiling_bytes()


def test_feature_block_statistics_blocked_matches_float64_reference() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(100, 5))
    train32 = train.astype(np.float32)
    mu = np.where(np.isfinite(train.mean(axis=0)), train.mean(axis=0), 0.0)
    sigma = train.std(axis=0)
    scale = np.where((sigma > 0.0) & np.isfinite(sigma), 1.0 / sigma, 0.0)
    expected = (train - mu) * scale
    mu32, scale32 = _feature_block_statistics(train32, block_cols=2)
    got = (train32.astype(np.float64) - mu32.astype(np.float64)) * scale32.astype(
        np.float64
    )
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_feature_block_statistics_transient_is_bounded() -> None:
    # Supporting check only: the authoritative proofs are the pure estimator
    # unit tests above and the process-level peak recorded by
    # freeze_ridge_reference.py on the real medium cell.
    import tracemalloc

    rng = np.random.default_rng(3)
    values = rng.normal(size=(20_000, 64)).astype(np.float32)
    tracemalloc.start()
    _feature_block_statistics(values, block_cols=8)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Generous bound: 3 float64-sized column blocks (copy + deviation + output).
    block_bytes = 20_000 * 8 * 8 * 3
    assert peak < block_bytes


def _frozen_reference(label: str) -> Path | None:
    matches = sorted(Path("artifacts").glob(f"reports/ridge_reference_*_{label}.json"))
    return matches[-1] if matches else None


def test_frozen_ridge_reference_matches_memory_safe_path() -> None:
    pre_path = _frozen_reference("pre")
    post_path = _frozen_reference("post")
    if pre_path is None or post_path is None:
        pytest.skip("frozen ridge reference artifacts missing")
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    post = json.loads(post_path.read_text(encoding="utf-8"))
    assert post["cell_id"] == pre["cell_id"]
    assert post["feature_count"] == pre["feature_count"]
    assert post["train_rows_purged"] == pre["train_rows_purged"]
    assert post["val_rows"] == pre["val_rows"]
    # Standardized feature statistics within the frozen tolerance contract.
    for key in ("mu", "sigma"):
        pre_vals = np.asarray(pre["feature_statistics"][key])
        post_vals = np.asarray(post["feature_statistics"][key])
        assert np.allclose(post_vals, pre_vals, rtol=1e-5, atol=1e-6), key
    # Zero-variance and finite/null filtering behavior: EXACT.
    assert (
        post["feature_statistics"]["zero_variance_features"]
        == pre["feature_statistics"]["zero_variance_features"]
    )
    assert post["finite_y_counts"] == pre["finite_y_counts"]
    assert (
        post["feature_statistics"]["finite_train_rows"]
        == pre["feature_statistics"]["finite_train_rows"]
    )
    # Predictions: bit-exact digest (blocked statistics are value-identical).
    assert post["predictions"]["sha256"] == pre["predictions"]["sha256"]
    # Per-era CORR atol 1e-7 and summary within rtol 1e-6.
    pre_eras = pre["per_era_corr"]["by_era"]
    post_eras = post["per_era_corr"]["by_era"]
    assert sorted(pre_eras) == sorted(post_eras)
    for era in pre_eras:
        assert abs(post_eras[era] - pre_eras[era]) <= 1e-7, era
    for key in ("mean", "std", "sharpe"):
        pre_val = pre["per_era_corr"][key]
        post_val = post["per_era_corr"][key]
        assert abs(post_val - pre_val) <= 1e-6 * max(1.0, abs(pre_val)), key
    # Process-level memory evidence: the fix must reduce the measured peak.
    assert post["peak_memory"]["commit_gib"] < pre["peak_memory"]["commit_gib"]
    assert post["peak_memory"]["ws_gib"] < pre["peak_memory"]["ws_gib"]
