"""Unit tests for NeutralizationEngine."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

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
    original = np.linalg.lstsq

    def tracking_lstsq(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", tracking_lstsq)

    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0)
    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f2"], proportion=0.25)

    assert call_count == 1


def test_cache_validation_recomputes_on_mismatched_ids(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _risk_frame().filter(pl.col("era") == "1")
    changed_ids = df.with_columns(pl.lit("1").alias("era")).with_columns(
        pl.Series("id", [f"other_{i}" for i in range(df.height)])
    )
    engine = NeutralizationEngine(cache_dir=tmp_path)
    call_count = 0
    original = np.linalg.lstsq

    def tracking_lstsq(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", tracking_lstsq)

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
    original = np.linalg.lstsq

    def tracking_lstsq(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", tracking_lstsq)

    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0)
    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f3"], proportion=1.0)

    assert call_count == 2


def test_invalid_proportion_raises(tmp_path) -> None:
    engine = NeutralizationEngine(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="proportion"):
        engine.neutralize(
            _risk_frame(), pred_col="pred", feature_cols=["f1"], proportion=1.1
        )
