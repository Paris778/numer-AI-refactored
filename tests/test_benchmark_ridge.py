"""Tier-1 ridge benchmark generator contracts."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import _standardize_feature_block, generate_ridge_predictions


def _domain(
    *, n_train_eras: int = 30, n_val_eras: int = 10, rows_per_era: int = 12,
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
                target = float(np.clip(0.5 + 0.2 * (0.8 * f1 - 0.4 * f2 + 0.2 * f3)
                                      + rng.normal(0.0, 0.3), 0.0, 1.0))
                aux = float(np.clip(target + rng.normal(0.0, 0.2), 0.0, 1.0))
                rows.append({
                    "era": era, "id": f"{era}_{idx}",
                    "f1": f1, "f2": f2, "f3": f3, "f_const": 1.0,
                    "target": target, "aux": aux,
                })
        return rows

    train = pl.DataFrame(make(range(1, n_train_eras + 1)))
    val = pl.DataFrame(make(range(n_train_eras + 1, n_train_eras + 1 + n_val_eras)))
    val = val.select(["era", "id", *feature_cols])
    return train, val, feature_cols


def test_single_target_ridge_covers_val_and_is_finite() -> None:
    train, val, feats = _domain()
    out = generate_ridge_predictions(
        train, val, targets=["target"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
    )
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == val.height
    assert set(out.get_column("era").unique().to_list()) == \
        set(val.get_column("era").unique().to_list())
    assert out.get_column("prediction").is_finite().all()


def test_ridge_is_deterministic_per_seed() -> None:
    train, val, feats = _domain()
    a = generate_ridge_predictions(
        train, val, targets=["target"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
    )
    b = generate_ridge_predictions(
        train, val, targets=["target"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
    )
    assert a.equals(b)


def test_zero_variance_feature_does_not_produce_nan() -> None:
    train, val, feats = _domain()
    out = generate_ridge_predictions(
        train, val, targets=["target"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
    )
    values = out.get_column("prediction").to_numpy()
    assert np.isfinite(values).all()


def test_per_era_output_is_rank_gaussianized() -> None:
    train, val, feats = _domain()
    out = generate_ridge_predictions(
        train, val, targets=["target"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
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
        pl.when(pl.col("era") == "0001").then(None).otherwise(pl.col("aux")).alias("aux")
    )
    out = generate_ridge_predictions(
        poisoned, val, targets=["target", "aux"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
    )
    assert out.height == val.height
    assert np.isfinite(out.get_column("prediction").to_numpy()).all()


def test_tight_gap_raises() -> None:
    train, val, feats = _domain(n_train_eras=30, n_val_eras=10)
    close_val = val.with_columns(pl.lit("0032").alias("era"))
    with pytest.raises(ValueError, match="purge"):
        generate_ridge_predictions(
            train, close_val, targets=["target"], feature_cols=feats,
            alpha=1.0, seed=42, purge_eras=8,
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

