"""Tier-2 tree benchmark generator contracts."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import generate_tree_predictions
from nmr.models import construct_tree_model


def _domain(
    *, n_train_eras: int = 24, n_val_eras: int = 6, rows_per_era: int = 10,
    seed: int = 20260815,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    feature_cols = ["f1", "f2", "f3"]

    def make(eras: range) -> list[dict[str, float | str]]:
        rows = []
        for era_num in eras:
            era = f"{era_num:04d}"
            for idx in range(rows_per_era):
                f1, f2, f3 = (float(rng.normal()) for _ in range(3))
                target = float(np.clip(
                    0.5 + 0.25 * (f1 * f2 - 0.3 * f3) + rng.normal(0.0, 0.4),
                    0.0, 1.0,
                ))
                rows.append({
                    "era": era, "id": f"{era}_{idx}",
                    "f1": f1, "f2": f2, "f3": f3, "target": target,
                })
        return rows

    train = pl.DataFrame(make(range(1, n_train_eras + 1)))
    val = pl.DataFrame(make(range(n_train_eras + 1, n_train_eras + 1 + n_val_eras)))
    val = val.select(["era", "id", *feature_cols])
    return train, val, feature_cols


FAST_PARAMS = {"n_estimators": 5, "learning_rate": 0.1, "max_depth": 2, "num_leaves": 4, "colsample_bytree": 0.5}


def test_lgbm_tree_covers_val_and_is_finite() -> None:
    train, val, feats = _domain()
    out = generate_tree_predictions(
        train, val, target="target", feature_cols=feats,
        backend="lightgbm", params=FAST_PARAMS, seed=42, purge_eras=8,
    )
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == val.height
    assert np.isfinite(out.get_column("prediction").to_numpy()).all()


def test_xgb_tree_accepts_max_leaves_mapping() -> None:
    train, val, feats = _domain()
    xgb_params = {
        "n_estimators": 5, "learning_rate": 0.1, "max_depth": 2,
        "max_leaves": 4, "colsample_bytree": 0.5,
    }
    out = generate_tree_predictions(
        train, val, target="target", feature_cols=feats,
        backend="xgboost", params=xgb_params, seed=42, purge_eras=8,
    )
    assert np.isfinite(out.get_column("prediction").to_numpy()).all()


def test_tree_is_deterministic_per_seed() -> None:
    train, val, feats = _domain()
    a = generate_tree_predictions(
        train, val, target="target", feature_cols=feats,
        backend="lightgbm", params=FAST_PARAMS, seed=42, purge_eras=8,
    )
    b = generate_tree_predictions(
        train, val, target="target", feature_cols=feats,
        backend="lightgbm", params=FAST_PARAMS, seed=42, purge_eras=8,
    )
    assert a.equals(b)


def test_colsample_floor_applied_by_constructor() -> None:
    model = construct_tree_model(
        "lightgbm", {"colsample_bytree": 0.001}, seed=42, n_features=10,
        device="cpu",
    )
    resolved = model.get_params()["colsample_bytree"]
    assert resolved >= 0.1
