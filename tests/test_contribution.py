"""Tests for E3 contribution and uniqueness metrics (BMC + CWMM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numerai_tools.scoring import correlation_contribution

from nmr._transforms import power_1_5, rank_gaussianize
from nmr.config import DataConfig
from nmr.data import IngestionAgent
from nmr.evaluation import MIN_OVERLAP_ERAS, EvaluationEngine


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denom == 0.0:
        return 0.0
    return float((left_centered @ right_centered) / denom)


def _synthetic_metric_frame(n_eras: int = 25, rows_per_era: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(20260622)
    rows: list[dict[str, float | str]] = []
    for era in range(1, n_eras + 1):
        for idx in range(rows_per_era):
            pred = float(rng.normal(loc=0.2, scale=1.1))
            meta = float(rng.normal(loc=-0.1, scale=0.9))
            target = float(rng.choice(np.array([0.0, 0.25, 0.5, 0.75, 1.0])))
            rows.append(
                {
                    "era": f"{era:04d}",
                    "id": f"{era:04d}_{idx:03d}",
                    "pred": pred,
                    "meta": meta,
                    "target": target,
                }
            )
    return pl.DataFrame(rows)


def _coverage_frame(n_covered: int) -> pl.DataFrame:
    rows: list[dict[str, float | str | None]] = []
    for era in range(1, 26):
        covered = era > (25 - n_covered)
        era_label = f"{era:04d}"
        for i in range(3):
            if covered and i == 0:
                coverage_value: float | None = float(0.1 + era / 100.0)
            else:
                coverage_value = None
            rows.append(
                {
                    "era": era_label,
                    "pred": float(0.2 + 0.01 * i + 0.03 * era),
                    "target": float((i + era) % 5) / 4.0,
                    "benchmark": coverage_value,
                    "meta": coverage_value,
                }
            )
    return pl.DataFrame(rows)


_REAL_VALIDATION = Path("data/v5.2/validation.parquet")
_REAL_BENCHMARKS = Path("data/v5.2/validation_benchmark_models.parquet")


@pytest.mark.skipif(
    not (_REAL_VALIDATION.exists() and _REAL_BENCHMARKS.exists()),
    reason="v5.2 validation+benchmark inputs not on disk; skipped in CI",
)
def test_per_era_bmc_oracle_parity_on_real_v52() -> None:
    data_cfg = DataConfig(version="v5.2", feature_set="small", targets=("target",))
    agent = IngestionAgent(data_cfg)
    feature_cols = agent.features("small")[:3]

    eras = (
        pl.scan_parquet(data_cfg.path("validation.parquet"))
        .select("era")
        .unique(maintain_order=True)
        .head(60)
        .collect()
        .get_column("era")
        .to_list()
    )

    validation_df = (
        pl.scan_parquet(data_cfg.path("validation.parquet"))
        .select(["era", "id", "target", *feature_cols])
        .filter(pl.col("era").is_in(eras))
        .collect()
    )
    bench_df = (
        pl.scan_parquet(data_cfg.path("validation_benchmark_models.parquet"))
        .select(["era", "id", "v52_lgbm_cyrusd20"])
        .filter(pl.col("era").is_in(eras))
        .collect()
    )

    df = validation_df.join(bench_df, on=["era", "id"], how="left").with_columns(
        (
            (0.5 * pl.col(feature_cols[0]).cast(pl.Float64).fill_null(0.0))
            + (0.3 * pl.col(feature_cols[1]).cast(pl.Float64).fill_null(0.0))
            - (0.2 * pl.col(feature_cols[2]).cast(pl.Float64).fill_null(0.0))
            + (0.1 * pl.col("target").cast(pl.Float64).fill_null(0.0))
        ).alias("pred")
    )

    custom = EvaluationEngine("custom")
    official = EvaluationEngine("official")

    custom_scores = custom.per_era_bmc(
        df,
        pred_col="pred",
        benchmark_col="v52_lgbm_cyrusd20",
        target_col="target",
    )
    official_scores = official.per_era_bmc(
        df,
        pred_col="pred",
        benchmark_col="v52_lgbm_cyrusd20",
        target_col="target",
    )

    assert len(custom_scores) >= MIN_OVERLAP_ERAS
    assert list(custom_scores) == list(official_scores)
    for era in custom_scores:
        assert custom_scores[era] == pytest.approx(official_scores[era], abs=1e-6)

    one_era = next(iter(custom_scores))
    era_df = (
        df.filter(pl.col("era") == one_era)
        .select(["pred", "v52_lgbm_cyrusd20", "target"])
        .drop_nulls()
    )
    pdf = era_df.to_pandas()
    direct = float(
        correlation_contribution(
            pdf[["pred"]],
            pdf["v52_lgbm_cyrusd20"].rename("v52_lgbm_cyrusd20"),
            pdf["target"].rename("target"),
        )["pred"]
    )
    assert custom_scores[one_era] == pytest.approx(direct, abs=1e-6)


@pytest.mark.parametrize("backend", ["custom", "official"])
def test_bmc_equals_mmc_when_benchmark_is_meta(backend: str) -> None:
    engine = EvaluationEngine(backend)
    df = _synthetic_metric_frame(n_eras=25, rows_per_era=35)

    bmc = engine.per_era_bmc(
        df,
        pred_col="pred",
        benchmark_col="meta",
        target_col="target",
    )
    mmc = engine.per_era_mmc(
        df,
        pred_col="pred",
        meta_col="meta",
        target_col="target",
    )

    assert list(bmc) == list(mmc)
    for era in bmc:
        assert bmc[era] == pytest.approx(mmc[era], abs=0.0)


def test_cwmm_definition_bounds_and_backend_behavior() -> None:
    rng = np.random.default_rng(123)
    n = 400
    pred_same = rng.normal(size=n)
    meta_same = pred_same.copy()
    pred_ind = rng.normal(size=n)
    meta_ind = rng.normal(size=n)
    pred_neg = np.linspace(-2.0, 2.0, n)
    meta_neg = -pred_neg

    df = pl.DataFrame(
        {
            "era": (["0001"] * n) + (["0002"] * n) + (["0003"] * n),
            "pred": np.concatenate([pred_same, pred_ind, pred_neg]),
            "meta": np.concatenate([meta_same, meta_ind, meta_neg]),
        }
    )

    custom = EvaluationEngine("custom")
    official = EvaluationEngine("official")
    cwmm_custom = custom.per_era_cwmm(
        df,
        pred_col="pred",
        meta_col="meta",
        min_overlap_eras=1,
    )
    cwmm_official = official.per_era_cwmm(
        df,
        pred_col="pred",
        meta_col="meta",
        min_overlap_eras=1,
    )

    assert cwmm_custom == cwmm_official

    for era in ("0001", "0002", "0003"):
        era_df = df.filter(pl.col("era") == era)
        pred = era_df.get_column("pred").to_numpy()
        meta = era_df.get_column("meta").to_numpy()
        reconstructed = _pearson(
            power_1_5(rank_gaussianize(pred)),
            power_1_5(rank_gaussianize(meta)),
        )
        assert cwmm_custom[era] == pytest.approx(reconstructed, abs=1e-12)
        assert -1.0 - 1e-12 <= cwmm_custom[era] <= 1.0 + 1e-12

    assert cwmm_custom["0001"] == pytest.approx(1.0, abs=1e-12)
    assert abs(cwmm_custom["0002"]) < 0.2
    assert cwmm_custom["0003"] == pytest.approx(-1.0, abs=1e-12)


def test_coverage_resolution_and_non_vacuity_guard_for_bmc_and_cwmm() -> None:
    engine = EvaluationEngine("custom")

    frame_20 = _coverage_frame(n_covered=20)
    expected_covered = [f"{i:04d}" for i in range(6, 26)]

    bmc = engine.per_era_bmc(
        frame_20,
        pred_col="pred",
        benchmark_col="benchmark",
        target_col="target",
    )
    cwmm = engine.per_era_cwmm(
        frame_20,
        pred_col="pred",
        meta_col="meta",
    )

    assert len(bmc) == 20
    assert len(cwmm) == 20
    assert list(bmc) == expected_covered
    assert list(cwmm) == expected_covered
    assert "0005" not in bmc

    frame_19 = _coverage_frame(n_covered=19)
    expected_message = (
        "Non-vacuity violation: intersection yielded only 19 eras; "
        "minimum required 20."
    )
    with pytest.raises(ValueError, match=expected_message):
        engine.per_era_bmc(
            frame_19,
            pred_col="pred",
            benchmark_col="benchmark",
            target_col="target",
        )
    with pytest.raises(ValueError, match=expected_message):
        engine.per_era_cwmm(
            frame_19,
            pred_col="pred",
            meta_col="meta",
        )


def test_degenerate_covered_eras_score_zero_and_are_counted() -> None:
    rows: list[dict[str, float | str]] = []
    for era in range(1, 21):
        era_label = f"{era:04d}"
        for idx in range(6):
            pred = 0.5 if era == 1 else float(idx + era / 10.0)
            benchmark = 0.3 if era == 2 else float((-1) ** idx * (0.2 + era / 100.0))
            rows.append(
                {
                    "era": era_label,
                    "pred": pred,
                    "benchmark": benchmark,
                    "meta": benchmark,
                    "target": float((idx + era) % 5) / 4.0,
                }
            )
    df = pl.DataFrame(rows)

    engine = EvaluationEngine("custom")
    bmc = engine.per_era_bmc(
        df,
        pred_col="pred",
        benchmark_col="benchmark",
        target_col="target",
    )
    cwmm = engine.per_era_cwmm(
        df,
        pred_col="pred",
        meta_col="meta",
    )

    assert len(bmc) == 20
    assert len(cwmm) == 20
    assert bmc["0001"] == pytest.approx(0.0)
    assert bmc["0002"] == pytest.approx(0.0)
    assert cwmm["0001"] == pytest.approx(0.0)
    assert cwmm["0002"] == pytest.approx(0.0)
    assert all(np.isfinite(list(bmc.values())))
    assert all(np.isfinite(list(cwmm.values())))


def test_bmc_and_cwmm_deterministic() -> None:
    df = _synthetic_metric_frame(n_eras=25, rows_per_era=20)
    engine = EvaluationEngine("custom")

    bmc_a = engine.per_era_bmc(
        df,
        pred_col="pred",
        benchmark_col="meta",
        target_col="target",
    )
    bmc_b = engine.per_era_bmc(
        df,
        pred_col="pred",
        benchmark_col="meta",
        target_col="target",
    )
    cwmm_a = engine.per_era_cwmm(
        df,
        pred_col="pred",
        meta_col="meta",
    )
    cwmm_b = engine.per_era_cwmm(
        df,
        pred_col="pred",
        meta_col="meta",
    )

    assert bmc_a == bmc_b
    assert cwmm_a == cwmm_b
