"""Unit tests for NeutralizationEngine."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nmr._transforms import neutralize_array
from nmr.risk import NeutralizationEngine


def _risk_frame() -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for era in ("1", "2"):
        for idx in range(1, 9):
            f1 = float(idx)
            f2 = float((idx % 3) - 1)
            pred = (
                (1.7 * f1)
                - (0.9 * f2)
                + (0.03 * (idx**2))
                + (0.5 if era == "2" else 0.0)
            )
            rows.append(
                {
                    "era": era,
                    "id": f"{era}_{idx}",
                    "pred": pred,
                    "f1": f1,
                    "f2": f2,
                }
            )
    return pl.DataFrame(rows)


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def test_proportion_zero_is_identity(tmp_path) -> None:
    df = _risk_frame()
    engine = NeutralizationEngine(cache_dir=tmp_path)
    result = engine.neutralize(
        df,
        pred_col="pred",
        feature_cols=["f1", "f2"],
        proportion=0.0,
    )
    assert result.equals(df)


def test_proportion_one_drives_feature_exposure_near_zero(tmp_path) -> None:
    df = _risk_frame()
    engine = NeutralizationEngine(cache_dir=tmp_path)
    result = engine.neutralize(
        df,
        pred_col="pred",
        feature_cols=["f1", "f2"],
        proportion=1.0,
    )

    for era in ("1", "2"):
        era_df = result.filter(pl.col("era") == era)
        pred = era_df.get_column("pred").to_numpy()
        assert np.std(pred) > 0.0
        for feature in ("f1", "f2"):
            assert abs(_corr(pred, era_df.get_column(feature).to_numpy())) < 1e-10


def test_intercept_handling_zeroes_pure_linear_plus_offset_signal(tmp_path) -> None:
    df = pl.DataFrame(
        {
            "era": ["1"] * 6,
            "id": [f"id_{i}" for i in range(6)],
            "pred": [11.0, 13.0, 15.0, 17.0, 19.0, 21.0],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    engine = NeutralizationEngine(cache_dir=tmp_path)
    result = engine.neutralize(df, pred_col="pred", feature_cols=["f1"], proportion=1.0)
    assert np.allclose(result.get_column("pred").to_numpy(), 0.0, atol=1e-10)


def test_per_era_independence(tmp_path) -> None:
    df = _risk_frame()
    mutated_df = df.with_columns(
        pl.when(pl.col("era") == "2")
        .then(pl.col("pred") * 100.0)
        .otherwise(pl.col("pred"))
        .alias("pred")
    )
    engine = NeutralizationEngine(cache_dir=tmp_path)

    base = engine.neutralize(
        df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0
    )
    mutated = engine.neutralize(
        mutated_df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0
    )

    base_era1 = base.filter(pl.col("era") == "1").get_column("pred").to_numpy()
    mutated_era1 = mutated.filter(pl.col("era") == "1").get_column("pred").to_numpy()
    assert np.allclose(base_era1, mutated_era1, atol=1e-12)


def test_determinism(tmp_path) -> None:
    df = _risk_frame()
    engine = NeutralizationEngine(cache_dir=tmp_path)
    first = engine.neutralize(
        df, pred_col="pred", feature_cols=["f1", "f2"], proportion=0.6
    )
    second = engine.neutralize(
        df, pred_col="pred", feature_cols=["f1", "f2"], proportion=0.6
    )
    assert first.equals(second)


def test_cache_hit_avoids_recompute(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _risk_frame().filter(pl.col("era") == "1")
    engine = NeutralizationEngine(cache_dir=tmp_path)
    call_count = 0
    original = np.linalg.pinv

    def tracking_pinv(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "pinv", tracking_pinv)

    shifted_df = df.with_columns((pl.col("pred") * 1.7 + 3.0).alias("pred"))

    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0)
    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f2"], proportion=0.25)
    engine.neutralize(
        shifted_df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0
    )

    assert call_count == 1


def test_cache_hit_reuses_same_pinv_for_different_predictions(tmp_path) -> None:
    df = _risk_frame().filter(pl.col("era") == "1")
    second_pred_df = df.with_columns(
        (pl.col("pred") * -0.4 + 2.5 * pl.col("f1") + 0.7).alias("pred")
    )
    engine = NeutralizationEngine(cache_dir=tmp_path)

    first = engine.neutralize(
        df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0
    )
    second = engine.neutralize(
        second_pred_df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0
    )

    assert not np.allclose(
        first.get_column("pred").to_numpy(),
        second.get_column("pred").to_numpy(),
        atol=1e-12,
    )


def test_cache_validation_recomputes_on_mismatched_ids(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _risk_frame().filter(pl.col("era") == "1")
    changed_ids = df.with_columns(pl.lit("1").alias("era")).with_columns(
        pl.Series("id", [f"other_{i}" for i in range(df.height)])
    )
    engine = NeutralizationEngine(cache_dir=tmp_path)
    call_count = 0
    original = np.linalg.pinv

    def tracking_pinv(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "pinv", tracking_pinv)

    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0)
    engine.neutralize(
        changed_ids, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0
    )

    assert call_count == 2


def test_cache_validation_recomputes_on_mismatched_features(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = (
        _risk_frame()
        .filter(pl.col("era") == "1")
        .with_columns((pl.col("f1") * 0.1).alias("f3"))
    )
    engine = NeutralizationEngine(cache_dir=tmp_path)
    call_count = 0
    original = np.linalg.pinv

    def tracking_pinv(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "pinv", tracking_pinv)

    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0)
    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f3"], proportion=1.0)

    assert call_count == 2


def test_invalid_proportion_raises(tmp_path) -> None:
    engine = NeutralizationEngine(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="proportion"):
        engine.neutralize(
            _risk_frame(), pred_col="pred", feature_cols=["f1"], proportion=1.1
        )


def test_neutralize_array_cached_matches_uncached(tmp_path) -> None:
    df = _risk_frame().filter(pl.col("era") == "1")
    engine = NeutralizationEngine(cache_dir=tmp_path)
    # First call populates the cache; second hits it.
    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0)
    result = engine.neutralize(
        df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0
    )
    pred = df.get_column("pred").to_numpy()
    features = df.select(["f1", "f2"]).to_numpy()
    direct = neutralize_array(pred, features, 1.0, pseudo_inverse=None)
    assert np.allclose(result.get_column("pred").to_numpy(), direct, atol=1e-12)


def test_neutralize_array_zero_variance_returns_unchanged() -> None:
    pred = np.full(5, 0.5)
    features = np.arange(10, dtype=float).reshape(5, 2)
    out = neutralize_array(pred, features, 1.0)
    assert np.array_equal(out, pred)


def test_cache_corruption_recomputes(tmp_path) -> None:
    df = _risk_frame().filter(pl.col("era") == "1")
    engine = NeutralizationEngine(cache_dir=tmp_path)
    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0)
    npy_files = list(tmp_path.glob("*.npy"))
    assert len(npy_files) == 1
    npy_files[0].write_bytes(b"\x00" * 16)  # truncate/corrupt
    result = engine.neutralize(
        df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0
    )
    assert np.all(np.isfinite(result.get_column("pred").to_numpy()))


def test_cache_eviction_respects_budget(tmp_path) -> None:
    """Two eras' cache entries; the budget fits only one -> the older is evicted."""
    import os
    import time

    full_df = _risk_frame()  # eras "1" and "2"
    engine = NeutralizationEngine(cache_dir=tmp_path, max_cache_bytes=900)
    engine.neutralize(
        full_df.filter(pl.col("era") == "1"),
        pred_col="pred",
        feature_cols=["f1", "f2"],
        proportion=1.0,
    )
    # Backdate era 1's pair so the mtime-oldest eviction is deterministic even
    # when writes land in the same filesystem clock tick: era 1's npy strictly
    # older than its json, and both strictly older than era 2's pair.
    era1_npy = next(tmp_path.glob("era_1_*.npy"))
    era1_json = next(tmp_path.glob("era_1_*.json"))
    old = time.time() - 3600.0
    os.utime(era1_npy, (old, old))
    os.utime(era1_json, (old + 1.0, old + 1.0))
    engine.neutralize(
        full_df.filter(pl.col("era") == "2"),
        pred_col="pred",
        feature_cols=["f1", "f2"],
        proportion=1.0,
    )
    assert engine.cache_size_bytes() <= 900
    survivors = [p.name for p in tmp_path.glob("*.npy")]
    assert len(survivors) == 1
    assert "era_2_" in survivors[0]  # mtime-oldest (era 1) evicted, era 2 survives


def test_zero_variance_era_keeps_rows_and_is_logged(tmp_path, caplog) -> None:
    df = pl.DataFrame(
        {
            "era": ["1", "1", "2", "2"],
            "id": ["a", "b", "c", "d"],
            "pred": [0.5, 0.5, 0.1, 0.9],
            "f1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    import logging
    engine = NeutralizationEngine(cache_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="nmr.risk"):
        result = engine.neutralize(df, pred_col="pred", feature_cols=["f1"], proportion=1.0)
    assert result.height == 4
    era1 = result.filter(pl.col("era") == "1").get_column("pred").to_numpy()
    assert np.array_equal(era1, np.array([0.5, 0.5]))
    assert any("zero-variance" in record.message for record in caplog.records)
