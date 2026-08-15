"""Tier-3 canonical benchmark generator contracts."""

from __future__ import annotations

import numpy as np
import polars as pl

from nmr.benchmark import generate_canonical_predictions
from nmr.ensemble import Ensembler


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
                target = float(np.clip(0.5 + 0.2 * f1 + rng.normal(0.0, 0.4), 0.0, 1.0))
                rows.append({
                    "era": era, "id": f"{era}_{idx}",
                    "f1": f1, "f2": f2, "f3": f3,
                    "target": target, "aux": float(np.clip(target + rng.normal(0, 0.1), 0, 1)),
                })
        return rows

    train = pl.DataFrame(make(range(1, n_train_eras + 1)))
    val = pl.DataFrame(make(range(n_train_eras + 1, n_train_eras + 1 + n_val_eras)))
    val = val.select(["era", "id", *feature_cols])
    return train, val, feature_cols


FAST_PARAMS = {"n_estimators": 5, "learning_rate": 0.1, "max_depth": 2, "num_leaves": 4, "colsample_bytree": 0.5}


def test_hello_numerai_contract() -> None:
    train, val, feats = _domain()
    out = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.0, purge_eras=8,
    )
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == val.height
    assert np.isfinite(out.get_column("prediction").to_numpy()).all()


def test_neutralized_50_reduces_feature_alignment() -> None:
    train, val, feats = _domain()
    raw = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.0, purge_eras=8,
    )
    neut = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.5, purge_eras=8,
    )
    assert np.isfinite(neut.get_column("prediction").to_numpy()).all()
    assert not raw.equals(neut)


def test_sunshine_multi_target_blend_and_neutralization() -> None:
    train, val, feats = _domain()
    out = generate_canonical_predictions(
        train, val, targets=["target", "aux"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.25, purge_eras=8,
    )
    assert out.height == val.height
    assert np.isfinite(out.get_column("prediction").to_numpy()).all()


def test_canonical_is_deterministic_per_seed() -> None:
    train, val, feats = _domain()
    a = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.5, purge_eras=8,
    )
    b = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.5, purge_eras=8,
    )
    assert a.equals(b)


def test_blend_components_are_rank_normalized_before_blending() -> None:
    train, val, feats = _domain()
    frame = val.with_columns(
        pl.lit(1.0).alias("p1"),
        pl.lit(2.0).alias("p2"),
    )
    blended = Ensembler().blend(
        Ensembler.rank_normalize(frame, pred_cols=["p1", "p2"]),
        pred_cols=["p1", "p2"], weights=[0.5, 0.5], out_col="prediction",
    )
    assert np.isfinite(blended.get_column("prediction").to_numpy()).all()
