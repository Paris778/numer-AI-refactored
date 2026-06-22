"""Parity tests: custom backend must track the official oracle closely."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nmr.config import DataConfig
from nmr.data import IngestionAgent
from nmr.evaluation import EvaluationEngine

CORR_ATOL = 1e-6
MMC_ATOL = 1e-6
FNC_ATOL = 1e-5  # Neutralization uses least-squares; tiny numeric drift is expected.


def _synthetic_eval_frame() -> pl.DataFrame:
    rng = np.random.default_rng(20260621)
    rows: list[dict[str, float | str]] = []
    target_support = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    for era in range(1, 4):
        for idx in range(300):
            pred = float(
                np.clip(rng.normal(loc=0.5 + 0.03 * era, scale=0.18), 0.0, 1.0)
            )
            meta = float(
                np.clip(rng.normal(loc=0.45 + 0.02 * era, scale=0.17), 0.0, 1.0)
            )
            target = float(rng.choice(target_support))
            f1 = float(rng.normal(loc=0.1 * era, scale=1.0))
            f2 = float(rng.normal(loc=-0.2 * era, scale=1.2))
            f3 = float(rng.normal(loc=0.05 * idx / 300, scale=0.9))
            rows.append(
                {
                    "era": str(era),
                    "pred": pred,
                    "target": target,
                    "meta": meta,
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                }
            )
    return pl.DataFrame(rows)


def _assert_non_vacuous(scores: dict[str, float], *, expected_eras: int) -> None:
    assert len(scores) >= expected_eras
    assert any(abs(value) > 1e-6 for value in scores.values())


@pytest.mark.parametrize(
    ("metric_name", "kwargs", "atol"),
    [
        ("corr", {"pred_col": "pred", "target_col": "target"}, CORR_ATOL),
        (
            "mmc",
            {"pred_col": "pred", "meta_col": "meta", "target_col": "target"},
            MMC_ATOL,
        ),
        (
            "fnc",
            {
                "pred_col": "pred",
                "feature_cols": ["f1", "f2", "f3"],
                "target_col": "target",
            },
            FNC_ATOL,
        ),
    ],
)
def test_custom_matches_official_on_synthetic_multi_era(
    metric_name: str,
    kwargs: dict,
    atol: float,
) -> None:
    df = _synthetic_eval_frame()
    custom = EvaluationEngine("custom")
    official = EvaluationEngine("official")

    custom_scores = getattr(custom, f"per_era_{metric_name}")(df, **kwargs)
    official_scores = getattr(official, f"per_era_{metric_name}")(df, **kwargs)

    assert list(custom_scores) == list(official_scores)
    _assert_non_vacuous(custom_scores, expected_eras=3)
    for era in custom_scores:
        assert custom_scores[era] == pytest.approx(official_scores[era], abs=atol)


_REAL_VALIDATION = Path("data/v5.2/validation.parquet")
_REAL_META = Path("data/v5.2/meta_model.parquet")
_REAL_FEATURES = Path("data/v5.2/features.json")


@pytest.mark.skipif(
    not (_REAL_VALIDATION.exists() and _REAL_META.exists() and _REAL_FEATURES.exists()),
    reason="v5.2 parity inputs not on disk; skipped in CI",
)
def test_real_v52_sampled_parity() -> None:
    data_cfg = DataConfig(version="v5.2", feature_set="small", targets=("target",))
    agent = IngestionAgent(data_cfg)
    feature_cols = agent.features("small")[:5]

    overlap_eras = (
        pl.scan_parquet(data_cfg.path("validation.parquet"))
        .select("era")
        .unique(maintain_order=True)
        .join(
            pl.scan_parquet(data_cfg.path("meta_model.parquet"))
            .select("era")
            .unique(maintain_order=True),
            on="era",
            how="inner",
        )
        .head(2)
        .collect()
        .get_column("era")
        .to_list()
    )
    assert len(overlap_eras) >= 2

    validation_df = (
        pl.scan_parquet(data_cfg.path("validation.parquet"))
        .select(["era", "id", "target", *feature_cols])
        .filter(pl.col("era").is_in(overlap_eras))
        .group_by("era", maintain_order=True)
        .head(120)
        .collect()
    )

    meta_df = (
        pl.scan_parquet(data_cfg.path("meta_model.parquet"))
        .select(["era", "id", "numerai_meta_model"])
        .filter(pl.col("era").is_in(overlap_eras))
        .group_by("era", maintain_order=True)
        .head(120)
        .collect()
    )

    df = validation_df.join(meta_df, on=["era", "id"], how="inner").with_columns(
        pl.sum_horizontal(
            [pl.col(col).cast(pl.Float64).fill_null(0.0) for col in feature_cols[:3]]
        ).alias("pred")
    )
    assert df.height > 0

    custom = EvaluationEngine("custom")
    official = EvaluationEngine("official")

    corr_custom = custom.per_era_corr(df, pred_col="pred", target_col="target")
    corr_official = official.per_era_corr(df, pred_col="pred", target_col="target")
    _assert_non_vacuous(corr_custom, expected_eras=2)
    for era in corr_custom:
        assert corr_custom[era] == pytest.approx(corr_official[era], abs=CORR_ATOL)

    mmc_custom = custom.per_era_mmc(
        df,
        pred_col="pred",
        meta_col="numerai_meta_model",
        target_col="target",
    )
    mmc_official = official.per_era_mmc(
        df,
        pred_col="pred",
        meta_col="numerai_meta_model",
        target_col="target",
    )
    _assert_non_vacuous(mmc_custom, expected_eras=2)
    for era in mmc_custom:
        assert mmc_custom[era] == pytest.approx(mmc_official[era], abs=MMC_ATOL)

    fnc_custom = custom.per_era_fnc(
        df,
        pred_col="pred",
        feature_cols=feature_cols,
        target_col="target",
    )
    fnc_official = official.per_era_fnc(
        df,
        pred_col="pred",
        feature_cols=feature_cols,
        target_col="target",
    )
    _assert_non_vacuous(fnc_custom, expected_eras=2)
    for era in fnc_custom:
        assert fnc_custom[era] == pytest.approx(fnc_official[era], abs=FNC_ATOL)
