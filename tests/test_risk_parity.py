"""Parity tests for NeutralizationEngine against numerai_tools.scoring.neutralize."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numerai_tools.scoring import neutralize as oracle_neutralize

from nmr.config import DataConfig
from nmr.data import IngestionAgent
from nmr.risk import NeutralizationEngine

NEUTRALIZE_ATOL = 1e-8  # Same least-squares solve; only float roundoff should differ.


def _assert_non_vacuous(
    pred_before: np.ndarray, pred_after: np.ndarray, *, expected_rows: int
) -> None:
    assert len(pred_after) >= expected_rows
    assert np.std(pred_before) > 0.0
    assert np.any(np.abs(pred_after - pred_before) > 1e-6)


def _synthetic_parity_frame() -> pl.DataFrame:
    rng = np.random.default_rng(20260621)
    rows: list[dict[str, float | str]] = []
    for era in range(1, 4):
        for idx in range(200):
            f1 = float(rng.normal(loc=0.2 * era, scale=1.0))
            f2 = float(rng.normal(loc=-0.1 * era, scale=1.1))
            f3 = float(rng.normal(loc=0.05 * idx, scale=0.7))
            pred = 0.8 * f1 - 0.5 * f2 + 0.2 * np.sin(f3) + float(rng.normal(scale=0.3))
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "pred": pred,
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                }
            )
    return pl.DataFrame(rows)


def _oracle_per_era(
    df: pl.DataFrame, *, pred_col: str, feature_cols: list[str]
) -> np.ndarray:
    parts: list[pl.DataFrame] = []
    for era in df.get_column("era").unique(maintain_order=True).to_list():
        era_df = df.filter(pl.col("era") == era)
        pdf = era_df.to_pandas()
        neutralized = oracle_neutralize(
            pdf[[pred_col]], pdf[feature_cols], proportion=1.0
        )
        parts.append(
            era_df.with_columns(
                pl.Series(name=pred_col, values=neutralized[pred_col].to_numpy())
            )
        )
    return pl.concat(parts).sort(["era", "id"]).get_column(pred_col).to_numpy()


def test_custom_matches_oracle_on_synthetic_multi_era(tmp_path) -> None:
    df = _synthetic_parity_frame()
    feature_cols = ["f1", "f2", "f3"]
    engine = NeutralizationEngine(cache_dir=tmp_path)

    result = engine.neutralize(
        df, pred_col="pred", feature_cols=feature_cols, proportion=1.0
    )
    actual = result.sort(["era", "id"]).get_column("pred").to_numpy()
    expected = _oracle_per_era(df, pred_col="pred", feature_cols=feature_cols)

    _assert_non_vacuous(
        pred_before=df.sort(["era", "id"]).get_column("pred").to_numpy(),
        pred_after=actual,
        expected_rows=600,
    )
    assert np.allclose(actual, expected, atol=NEUTRALIZE_ATOL, rtol=0.0, equal_nan=True)


def test_cached_reuse_across_predictions_matches_fresh_oracle(tmp_path) -> None:
    df_a = _synthetic_parity_frame().filter(pl.col("era") == "1")
    df_b = df_a.with_columns(
        (
            (-0.35 * pl.col("pred")) + (1.1 * pl.col("f1")) - (0.6 * pl.col("f2")) + 0.4
        ).alias("pred")
    )
    feature_cols = ["f1", "f2", "f3"]
    engine = NeutralizationEngine(cache_dir=tmp_path)

    engine.neutralize(df_a, pred_col="pred", feature_cols=feature_cols, proportion=1.0)
    cached_b = engine.neutralize(
        df_b, pred_col="pred", feature_cols=feature_cols, proportion=1.0
    )

    actual = cached_b.sort(["era", "id"]).get_column("pred").to_numpy()
    expected = _oracle_per_era(df_b, pred_col="pred", feature_cols=feature_cols)

    _assert_non_vacuous(
        pred_before=df_b.sort(["era", "id"]).get_column("pred").to_numpy(),
        pred_after=actual,
        expected_rows=df_b.height,
    )
    assert np.allclose(actual, expected, atol=NEUTRALIZE_ATOL, rtol=0.0, equal_nan=True)


_REAL_VALIDATION = Path("data/v5.2/validation.parquet")
_REAL_FEATURES = Path("data/v5.2/features.json")


@pytest.mark.skipif(
    not (_REAL_VALIDATION.exists() and _REAL_FEATURES.exists()),
    reason="v5.2 validation/features inputs not on disk; skipped in CI",
)
def test_real_v52_validation_slice_matches_oracle(tmp_path) -> None:
    data_cfg = DataConfig(version="v5.2", feature_set="small", targets=("target",))
    agent = IngestionAgent(data_cfg)
    feature_cols = agent.features("small")[:5]
    eras = (
        agent.load("validation", columns=["era"])
        .get_column("era")
        .unique(maintain_order=True)
        .head(2)
        .to_list()
    )

    df = (
        agent.load(
            "validation",
            columns=["era", "id", "target", *feature_cols],
        )
        .filter(pl.col("era").is_in(eras))
        .with_columns(
            (
                (0.6 * pl.col(feature_cols[0]).cast(pl.Float64))
                + (0.3 * pl.col(feature_cols[1]).cast(pl.Float64))
                - (0.2 * pl.col(feature_cols[2]).cast(pl.Float64))
                + (0.1 * pl.col("target").cast(pl.Float64))
            ).alias("pred")
        )
    )

    assert df.height > 0

    engine = NeutralizationEngine(cache_dir=tmp_path)
    result = engine.neutralize(
        df, pred_col="pred", feature_cols=feature_cols, proportion=1.0
    )
    actual = result.sort(["era", "id"]).get_column("pred").to_numpy()
    expected = _oracle_per_era(df, pred_col="pred", feature_cols=feature_cols)

    _assert_non_vacuous(
        pred_before=df.sort(["era", "id"]).get_column("pred").to_numpy(),
        pred_after=actual,
        expected_rows=df.height,
    )
    assert np.allclose(actual, expected, atol=NEUTRALIZE_ATOL, rtol=0.0, equal_nan=True)
