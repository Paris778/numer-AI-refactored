"""Tier-0 null prediction generator contracts."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import generate_null_predictions


def _index(n_eras: int = 5, rows_per_era: int = 4) -> pl.DataFrame:
    rows = []
    for era_num in range(1, n_eras + 1):
        for idx in range(rows_per_era):
            rows.append({"era": f"{era_num:04d}", "id": f"{era_num}_{idx}"})
    return pl.DataFrame(rows)


def _features(index: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    return index.with_columns(
        pl.Series("f1", rng.normal(size=index.height)),
        pl.Series("f2", rng.normal(size=index.height)),
    )


def test_constant_is_exactly_half() -> None:
    out = generate_null_predictions(_index(), kind="null_constant_05", seed=42)
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == 20
    assert (out.get_column("prediction") == 0.5).all()


def test_uniform_is_seeded_and_bounded() -> None:
    idx = _index()
    a = generate_null_predictions(idx, kind="null_uniform_rand", seed=42)
    b = generate_null_predictions(idx, kind="null_uniform_rand", seed=42)
    c = generate_null_predictions(idx, kind="null_uniform_rand", seed=43)
    values = a.get_column("prediction").to_numpy()
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert a.equals(b)
    assert not a.equals(c)


def test_gaussian_is_clipped_and_seeded() -> None:
    idx = _index()
    a = generate_null_predictions(idx, kind="null_gaussian_rand", seed=42)
    b = generate_null_predictions(idx, kind="null_gaussian_rand", seed=42)
    values = a.get_column("prediction").to_numpy()
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert a.equals(b)


def test_feature_mean_matches_manual_row_mean() -> None:
    idx = _index()
    feats = _features(idx)
    out = generate_null_predictions(
        idx, kind="null_feature_mean", seed=42,
        features=feats, feature_cols=["f1", "f2"],
    )
    manual = feats.with_columns(
        pl.mean_horizontal([pl.col("f1"), pl.col("f2")]).alias("prediction")
    ).select(["era", "id", "prediction"])
    assert out.sort(["era", "id"]).equals(manual.sort(["era", "id"]))


def test_feature_mean_missing_column_raises() -> None:
    idx = _index()
    feats = _features(idx)
    with pytest.raises(ValueError, match="f3"):
        generate_null_predictions(
            idx, kind="null_feature_mean", seed=42,
            features=feats, feature_cols=["f1", "f3"],
        )


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        generate_null_predictions(_index(), kind="null_bogus", seed=42)


def test_missing_join_keys_raise() -> None:
    bad = pl.DataFrame({"id": ["a"], "prediction": [0.5]})
    with pytest.raises(ValueError, match="era"):
        generate_null_predictions(bad, kind="null_constant_05", seed=42)
