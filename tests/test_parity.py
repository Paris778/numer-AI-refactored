"""Parity tests: custom backend must track the official oracle closely."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numerai_tools.scoring import correlation_contribution

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


_REAL_VALIDATION = Path("data/v5.3/validation.parquet")
_REAL_META = Path("data/v5.3/meta_model.parquet")
_REAL_FEATURES = Path("data/v5.3/features.json")


@pytest.mark.skipif(
    not (_REAL_VALIDATION.exists() and _REAL_META.exists() and _REAL_FEATURES.exists()),
    reason="v5.3 parity inputs not on disk; skipped in CI",
)
def test_real_v53_sampled_parity() -> None:
    data_cfg = DataConfig(version="v5.3", feature_set="small", targets=("target",))
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


# ---------------------------------------------------------------------------
# BMC oracle parity (relocated from the retired benchmark slice-3 suite)
# ---------------------------------------------------------------------------


def _slice3_inputs(
    *,
    n_eras: int = 260,
    rows_per_era: int = 36,
    seed: int = 20260622,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []

    for era_num in range(1, n_eras + 1):
        era = f"{era_num:04d}"
        for idx in range(rows_per_era):
            asset_id = f"{era}_{idx:03d}"
            f1 = float(rng.normal())
            f2 = float(rng.normal())
            f3 = float(rng.normal())
            latent = (
                (0.7 * f1) - (0.32 * f2) + (0.19 * f3) + float(rng.normal(0.0, 0.45))
            )
            target = float(np.clip(0.5 + 0.22 * latent, 0.0, 1.0))
            meta = float(0.58 * target + 0.42 * rng.random())
            benchmark = float(0.54 * target + 0.46 * rng.random())

            rows.append(
                {
                    "era": era,
                    "id": asset_id,
                    "numerai_meta_model": meta,
                    "target": target,
                    "target_cyrusd_20": target,
                    "target_cyrusd_60": float(
                        np.clip(target + rng.normal(0.0, 0.03), 0.0, 1.0)
                    ),
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "v52_lgbm_cyrusd20": benchmark,
                }
            )

    full = pl.DataFrame(rows)
    meta_model = full.select(["era", "id", "numerai_meta_model"])
    benchmarks = full.select(["era", "id", "v52_lgbm_cyrusd20"])
    features = full.select(["era", "id", "f1", "f2", "f3"])
    targets = full.select(
        ["era", "id", "target", "target_cyrusd_20", "target_cyrusd_60"]
    )
    return meta_model, benchmarks, features, targets


def test_slice3_bmc_oracle_parity() -> None:
    meta_model, benchmarks, features, targets = _slice3_inputs(
        n_eras=40, rows_per_era=20
    )
    cfg_pred = targets.select(["era", "id"]).with_columns(
        (0.7 * pl.col("id").cum_count()).cast(pl.Float64).alias("prediction")
    )

    base = (
        cfg_pred.join(meta_model, on=["era", "id"], how="inner")
        .join(targets.select(["era", "id", "target"]), on=["era", "id"], how="inner")
        .join(features, on=["era", "id"], how="inner")
        .join(benchmarks, on=["era", "id"], how="left")
    )

    evaluator = EvaluationEngine("custom")
    per_era = evaluator.per_era_bmc(
        base,
        pred_col="prediction",
        benchmark_col="v52_lgbm_cyrusd20",
        target_col="target",
        min_overlap_eras=20,
    )

    eras = sorted(per_era, key=int)
    one_era = eras[0]
    pdf = (
        base.filter(pl.col("era") == one_era)
        .select(["prediction", "v52_lgbm_cyrusd20", "target"])
        .to_pandas()
    )
    direct = float(
        correlation_contribution(
            pdf[["prediction"]],
            pdf["v52_lgbm_cyrusd20"].rename("v52_lgbm_cyrusd20"),
            pdf["target"].rename("target"),
        )["prediction"]
    )

    assert per_era[one_era] == pytest.approx(direct, abs=1e-6)


def _corr_frame(*, n_eras: int = 2, n_rows: int = 8, seed: int = 20260819) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for era in range(1, n_eras + 1):
        for idx in range(n_rows):
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "pred": float(np.clip(rng.normal(0.5 + 0.05 * era, 0.2), 0.0, 1.0)),
                    "target": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
                }
            )
    return pl.DataFrame(rows)


@pytest.mark.parametrize(
    "mutate",
    [
        "constant_pred",
        "constant_target",
        "nan_pred",
        "ties_pred",
        "single_row",
        "two_rows",
    ],
)
def test_corr_degenerate_eras_match_oracle(mutate: str) -> None:
    """Degenerate inputs never drift the custom path from numerai_tools."""
    df = _corr_frame()
    if mutate == "constant_pred":
        df = df.with_columns(pl.lit(0.5).alias("pred"))
    elif mutate == "constant_target":
        df = df.with_columns(pl.lit(0.5).alias("target"))
    elif mutate == "nan_pred":
        df = df.with_columns(
            pl.when(pl.col("id").str.ends_with("_0")).then(None).otherwise(pl.col("pred")).alias("pred")
        )
    elif mutate == "ties_pred":
        df = df.with_columns(pl.Series("pred", [0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.2, 0.2] * 2, dtype=pl.Float64))
    elif mutate == "single_row":
        df = pl.concat([df.group_by("era", maintain_order=True).head(1)])
    elif mutate == "two_rows":
        df = pl.concat([df.group_by("era", maintain_order=True).head(2)])

    custom = EvaluationEngine("custom").per_era_corr(df, pred_col="pred", target_col="target")
    official = EvaluationEngine("official").per_era_corr(df, pred_col="pred", target_col="target")
    assert list(custom) == list(official)
    for era in custom:
        assert custom[era] == pytest.approx(official[era], abs=1e-6, nan_ok=True)


@pytest.mark.parametrize("mutate", ["constant_meta", "ties_meta", "nan_meta", "nan_pred"])
def test_mmc_degenerate_columns_match_oracle(mutate: str) -> None:
    base = _corr_frame()
    df = base.with_columns(
        pl.Series("meta", np.linspace(0.1, 0.9, base.height), dtype=pl.Float64)
    )
    if mutate == "constant_meta":
        df = df.with_columns(pl.lit(0.5).alias("meta"))
    elif mutate == "ties_meta":
        df = df.with_columns(pl.Series("meta", [0.2, 0.2, 0.5, 0.5, 0.9, 0.9, 0.4, 0.4] * 2, dtype=pl.Float64))
    elif mutate == "nan_meta":
        df = df.with_columns(
            pl.when(pl.col("id").str.ends_with("_0")).then(None).otherwise(pl.col("meta")).alias("meta")
        )
    elif mutate == "nan_pred":
        df = df.with_columns(
            pl.when(pl.col("id").str.ends_with("_0")).then(None).otherwise(pl.col("pred")).alias("pred")
        )

    custom = EvaluationEngine("custom").per_era_mmc(df, pred_col="pred", meta_col="meta", target_col="target")
    official = EvaluationEngine("official").per_era_mmc(df, pred_col="pred", meta_col="meta", target_col="target")
    assert list(custom) == list(official)
    for era in custom:
        assert custom[era] == pytest.approx(official[era], abs=1e-6, nan_ok=True)


@pytest.mark.parametrize("mutate", ["duplicate_feature", "constant_feature", "wide_matrix", "nan_feature"])
def test_fnc_degenerate_features_match_oracle(mutate: str) -> None:
    rng = np.random.default_rng(20260819)
    base = _corr_frame()
    df = base.with_columns(
        pl.Series("f1", rng.normal(size=base.height), dtype=pl.Float64),
        pl.Series("f2", rng.normal(size=base.height), dtype=pl.Float64),
    )
    feats = ["f1", "f2"]
    if mutate == "duplicate_feature":
        df = df.with_columns(pl.col("f1").alias("f1_copy"))
        feats = ["f1", "f1_copy"]
    elif mutate == "constant_feature":
        df = df.with_columns(pl.lit(0.5).alias("fconst"))
        feats = ["f1", "fconst"]
    elif mutate == "wide_matrix":
        df = df.with_columns(
            [pl.Series(f"w{i}", rng.normal(size=df.height), dtype=pl.Float64) for i in range(12)]
        )
        feats = ["f1", "f2", *[f"w{i}" for i in range(12)]]
    elif mutate == "nan_feature":
        df = df.with_columns(
            pl.when(pl.col("id").str.ends_with("_0")).then(None).otherwise(pl.col("f1")).alias("f1")
        )

    custom = EvaluationEngine("custom").per_era_fnc(df, pred_col="pred", feature_cols=feats, target_col="target")
    official = EvaluationEngine("official").per_era_fnc(df, pred_col="pred", feature_cols=feats, target_col="target")
    assert list(custom) == list(official)
    for era in custom:
        assert custom[era] == pytest.approx(official[era], abs=1e-5, nan_ok=True)
