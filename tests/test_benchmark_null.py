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


# Seed found by scanning seeds 0..2000 for n=20 draws of
# np.random.default_rng(seed).normal(0.5, 0.15, 20) containing at least one
# draw outside [0, 1]; seed 275 yields max 1.00098..., so np.clip engages and
# produces an exactly-1.0 value. Without the clip, the in-bounds assertion
# alone would be vacuous for this fixture.
_CLIP_ENGAGING_SEED = 275


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
    a = generate_null_predictions(idx, kind="null_gaussian_rand", seed=_CLIP_ENGAGING_SEED)
    b = generate_null_predictions(idx, kind="null_gaussian_rand", seed=_CLIP_ENGAGING_SEED)
    values = a.get_column("prediction").to_numpy()
    assert np.all((values >= 0.0) & (values <= 1.0))
    # Seed 275 has an unclipped draw at 1.00098...; if np.clip is removed,
    # nothing equals 0.0/1.0 exactly and this fails.
    assert np.any((values == 0.0) | (values == 1.0))
    assert a.equals(b)


def test_gaussian_differs_across_seeds() -> None:
    idx = _index()
    a = generate_null_predictions(idx, kind="null_gaussian_rand", seed=42)
    c = generate_null_predictions(idx, kind="null_gaussian_rand", seed=43)
    assert not a.equals(c)


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


def test_feature_mean_requires_features_frame() -> None:
    with pytest.raises(ValueError, match="features frame"):
        generate_null_predictions(
            _index(), kind="null_feature_mean", seed=42, features=None,
        )


def test_feature_mean_requires_feature_cols() -> None:
    with pytest.raises(ValueError, match="feature column"):
        generate_null_predictions(
            _index(), kind="null_feature_mean", seed=42,
            features=_features(_index()), feature_cols=[],
        )


def test_feature_mean_dropped_index_row_raises() -> None:
    idx = _index()
    feats = _features(idx).head(idx.height - 1)
    with pytest.raises(ValueError, match="dropped"):
        generate_null_predictions(
            idx, kind="null_feature_mean", seed=42,
            features=feats, feature_cols=["f1", "f2"],
        )


def test_duplicate_index_rows_are_deduplicated() -> None:
    idx = _index()
    out = generate_null_predictions(
        pl.concat([idx, idx]), kind="null_constant_05", seed=42,
    )
    expected = generate_null_predictions(idx, kind="null_constant_05", seed=42)
    assert out.height == idx.height
    assert out.equals(expected)


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        generate_null_predictions(_index(), kind="null_bogus", seed=42)


def test_missing_join_keys_raise() -> None:
    bad = pl.DataFrame({"id": ["a"], "prediction": [0.5]})
    with pytest.raises(ValueError, match="era"):
        generate_null_predictions(bad, kind="null_constant_05", seed=42)
