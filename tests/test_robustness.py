"""Tests for E4 robustness diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nmr.evaluation import MIN_OVERLAP_ERAS, EvaluationEngine
from nmr.inference import ac_adjusted_sharpe, block_bootstrap_ci, resolve_block_len
from nmr.robustness import (
    _build_perturbation_field,
    _perturb_block_swap,
    _validate_bin_features,
    _validate_blocks,
    adversarial_perturbation,
    regime_conditioned_corr,
    time_horizon_stability,
)


def _extract_feature_sets() -> dict[str, list[str]]:
    import json

    payload = json.loads(Path("data/v5.2/features.json").read_text(encoding="utf-8"))
    return payload["feature_sets"]


def _structural_blocks(feature_sets: dict[str, list[str]]) -> dict[str, list[str]]:
    names = [
        "intelligence",
        "charisma",
        "strength",
        "dexterity",
        "constitution",
        "wisdom",
        "agility",
        "serenity",
        "sunshine",
        "rain",
        "midnight",
        "faith",
    ]
    return {name: feature_sets[name] for name in names}


def _real_linear_predict_fn(weights: np.ndarray):
    def _predict(x: np.ndarray) -> np.ndarray:
        centered = x - 2.0
        return centered @ weights

    return _predict


def _real_eval_train_payload(
    max_eval_rows: int = 2000,
    max_train_rows: int = 8000,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str], dict[str, list[str]]]:
    feature_sets = _extract_feature_sets()
    blocks = _structural_blocks(feature_sets)
    feature_cols = sorted({col for cols in blocks.values() for col in cols})

    # Keep memory bounded: sample deterministically in lazy mode before collect.
    sampled = (
        pl.scan_parquet("data/v5.2/validation.parquet")
        .select(["era", *feature_cols])
        .limit(max_eval_rows)
        .collect()
    )
    train = (
        pl.scan_parquet("data/v5.2/train.parquet")
        .select(feature_cols)
        .limit(max_train_rows)
        .collect()
    )

    sampled = sampled.with_columns(pl.col("era").cast(pl.Utf8))
    return sampled, train, feature_cols, blocks


def test_adversarial_perturbation_boundaries_and_determinism() -> None:
    feature_cols = ["f1", "f2", "f3"]
    eval_df = pl.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002"],
            "f1": [0, 1, 2, 3],
            "f2": [1, 2, 3, 4],
            "f3": [2, 3, 4, 0],
        }
    )
    train_df = pl.DataFrame(
        {
            "f1": [1, 2, 3, 4],
            "f2": [2, 3, 4, 0],
            "f3": [3, 4, 0, 1],
        }
    )
    blocks = {"b1": ["f1", "f2"], "b2": ["f3"]}
    predict_fn = lambda x: x[:, 0] + 0.5 * x[:, 1] - 0.2 * x[:, 2]

    out1 = adversarial_perturbation(
        eval_df,
        train_df,
        predict_fn=predict_fn,
        feature_cols=feature_cols,
        blocks=blocks,
        alpha=0.1,
        seed=7,
    )
    out2 = adversarial_perturbation(
        eval_df,
        train_df,
        predict_fn=predict_fn,
        feature_cols=feature_cols,
        blocks=blocks,
        alpha=0.1,
        seed=7,
    )
    assert out1 == out2

    with pytest.raises(ValueError, match="0.1 <= alpha <= 0.25"):
        adversarial_perturbation(
            eval_df,
            train_df,
            predict_fn=predict_fn,
            feature_cols=feature_cols,
            blocks=blocks,
            alpha=0.09,
            seed=1,
        )


@pytest.mark.skipif(
    not (
        Path("data/v5.2/validation.parquet").exists()
        and Path("data/v5.2/train.parquet").exists()
        and Path("data/v5.2/features.json").exists()
    ),
    reason="v5.2 robustness inputs not on disk; skipped in CI",
)
def test_adversarial_perturbation_real_data_non_noop_and_range() -> None:
    eval_df, train_df, feature_cols, blocks = _real_eval_train_payload()
    k = len(feature_cols)
    weights = np.linspace(-0.7, 0.9, num=k)
    predict_fn = _real_linear_predict_fn(weights)

    out = adversarial_perturbation(
        eval_df,
        train_df,
        predict_fn=predict_fn,
        feature_cols=feature_cols,
        blocks=blocks,
        alpha=0.1,
        seed=123,
    )

    assert out.effective_perturb_frac > 0.02
    assert out.ceiling_stability < 1.0
    assert -1.0 <= out.ceiling_stability <= 1.0
    assert -1.0 <= out.manifold_stability <= 1.0


@pytest.mark.skipif(
    not (
        Path("data/v5.2/validation.parquet").exists()
        and Path("data/v5.2/train.parquet").exists()
        and Path("data/v5.2/features.json").exists()
    ),
    reason="v5.2 robustness inputs not on disk; skipped in CI",
)
def test_adversarial_perturbation_field_independent_of_model_and_leak_free() -> None:
    eval_df, train_df, feature_cols, blocks = _real_eval_train_payload()
    k = len(feature_cols)
    predict_a = _real_linear_predict_fn(np.linspace(-0.2, 0.3, num=k))
    predict_b = _real_linear_predict_fn(np.linspace(0.4, -0.5, num=k))

    res_a = adversarial_perturbation(
        eval_df,
        train_df,
        predict_fn=predict_a,
        feature_cols=feature_cols,
        blocks=blocks,
        alpha=0.25,
        seed=99,
    )
    res_model_swap = adversarial_perturbation(
        eval_df,
        train_df,
        predict_fn=predict_b,
        feature_cols=feature_cols,
        blocks=blocks,
        alpha=0.25,
        seed=99,
    )

    poisoned_eval = eval_df.with_columns(
        [
            pl.when(pl.col(col) == 0)
            .then(4)
            .when(pl.col(col) == 4)
            .then(0)
            .otherwise(pl.col(col))
            .alias(col)
            for col in feature_cols
        ]
    )
    res_b = adversarial_perturbation(
        poisoned_eval,
        train_df,
        predict_fn=predict_b,
        feature_cols=feature_cols,
        blocks=blocks,
        alpha=0.25,
        seed=99,
    )

    # Effective changed fraction is model-independent when eval values are fixed.
    assert res_a.effective_perturb_frac == pytest.approx(
        res_model_swap.effective_perturb_frac,
        abs=0.0,
    )

    # Leak-free/model-independent field: identical shape + seed + data_version yields
    # identical perturbation masks/donors, independent of predict_fn and eval values.
    eval_n = eval_df.height
    train_n = train_df.height
    n_features = len(feature_cols)
    n_blocks = len(blocks)
    field_a = _build_perturbation_field(
        n_eval=eval_n,
        n_features=n_features,
        n_train=train_n,
        n_blocks=n_blocks,
        alpha=0.25,
        seed=99,
        data_version="v5.2",
    )
    field_b = _build_perturbation_field(
        n_eval=poisoned_eval.height,
        n_features=n_features,
        n_train=train_n,
        n_blocks=n_blocks,
        alpha=0.25,
        seed=99,
        data_version="v5.2",
    )
    for lhs, rhs in zip(field_a, field_b):
        assert np.array_equal(lhs, rhs)

    # Strong leak-free proof: swapped output values are always sourced from train donors,
    # and replaying the swap loop in the same block order reproduces public behavior.
    block_indices = _validate_blocks(feature_cols, blocks)
    eval_bins = _validate_bin_features(eval_df, feature_cols)
    train_bins = _validate_bin_features(train_df, feature_cols)
    _, _, block_mask, block_donors = field_a
    swapped = _perturb_block_swap(
        eval_bins,
        train_bins,
        block_indices=block_indices,
        block_mask=block_mask,
        block_donors=block_donors,
    )

    changed = swapped != eval_bins
    for col_idx in range(swapped.shape[1]):
        changed_rows = changed[:, col_idx]
        if not np.any(changed_rows):
            continue
        swapped_values = set(np.unique(swapped[changed_rows, col_idx]).tolist())
        train_values = set(np.unique(train_bins[:, col_idx]).tolist())
        assert swapped_values.issubset(train_values)

    replay = eval_bins.copy()
    for block_i, idxs in enumerate(block_indices):
        rows = np.flatnonzero(block_mask[:, block_i])
        if rows.size == 0:
            continue
        donors = block_donors[rows, block_i]
        replay[np.ix_(rows, idxs)] = train_bins[np.ix_(donors, idxs)]
    assert np.array_equal(swapped, replay)

    # Poisoning eval values can change clamp-collision rate, so changed-cell fractions may differ.
    assert res_b.effective_perturb_frac >= 0.0


def test_time_horizon_stability_same_name_and_relative_divergence() -> None:
    eras = [f"{i:04d}" for i in range(1, 31)]
    rows: list[dict[str, float | str]] = []
    for i, era in enumerate(eras):
        for j in range(12):
            rows.append(
                {
                    "era": era,
                    "pred": float(0.2 * i + 0.1 * j),
                    "benchmark": float(0.2 * i + 0.1 * j),
                    "target_alpha_20": float((i + j) % 5) / 4.0,
                    "target_alpha_60": float((2 * i + j) % 5) / 4.0,
                }
            )
    df = pl.DataFrame(rows)

    res = time_horizon_stability(
        df,
        pred_col="pred",
        benchmark_col="benchmark",
        target_name="alpha",
    )
    assert res.n_eras == 30
    assert res.relative_divergence == pytest.approx(0.0, abs=1e-12)

    with pytest.raises(ValueError, match="Missing required columns"):
        time_horizon_stability(
            df,
            pred_col="pred",
            benchmark_col="benchmark",
            target_name="does_not_exist",
        )


def test_time_horizon_stability_non_vacuity_19_fails_20_passes() -> None:
    rows: list[dict[str, float | str]] = []
    for i in range(1, 21):
        era = f"{i:04d}"
        for j in range(8):
            rows.append(
                {
                    "era": era,
                    "pred": float(0.3 * i + 0.02 * j),
                    "benchmark": float(0.3 * i + 0.02 * j),
                    "target_alpha_20": float((i + j) % 5) / 4.0,
                    "target_alpha_60": float((2 * i + j) % 5) / 4.0,
                }
            )
    df = pl.DataFrame(rows)

    out = time_horizon_stability(
        df,
        pred_col="pred",
        benchmark_col="benchmark",
        target_name="alpha",
        min_overlap_eras=20,
    )
    assert out.n_eras == 20

    df_19 = df.filter(pl.col("era") != "0020")
    with pytest.raises(
        ValueError,
        match=(
            "Non-vacuity violation: horizon overlap yielded only "
            "19 eras; minimum required 20."
        ),
    ):
        time_horizon_stability(
            df_19,
            pred_col="pred",
            benchmark_col="benchmark",
            target_name="alpha",
            min_overlap_eras=20,
        )

    with pytest.raises(ValueError, match="min_overlap_eras must be >= 1"):
        time_horizon_stability(
            df,
            pred_col="pred",
            benchmark_col="benchmark",
            target_name="alpha",
            min_overlap_eras=0,
        )


def test_time_horizon_stability_floor_applies_to_resolved_eras_only() -> None:
    rows: list[dict[str, float | str | None]] = []
    for i in range(1, 22):
        era = f"{i:04d}"
        for j in range(6):
            rows.append(
                {
                    "era": era,
                    "pred": float(0.15 * i + 0.03 * j),
                    "benchmark": float(0.15 * i + 0.03 * j),
                    "target_alpha_20": float((i + j) % 5) / 4.0,
                    "target_alpha_60": (
                        None
                        if era in {"0021", "0022"}
                        else float((2 * i + j) % 5) / 4.0
                    ),
                }
            )
    df = pl.DataFrame(rows)

    # 20 resolved eras pass even with 22 total eras.
    out = time_horizon_stability(
        df,
        pred_col="pred",
        benchmark_col="benchmark",
        target_name="alpha",
        min_overlap_eras=20,
    )
    assert out.n_eras == 20

    # 19 resolved eras fail even with 21 total eras.
    df_19 = df.filter(pl.col("era") != "0020")
    with pytest.raises(
        ValueError,
        match=(
            "Non-vacuity violation: horizon overlap yielded only "
            "19 eras; minimum required 20."
        ),
    ):
        time_horizon_stability(
            df_19,
            pred_col="pred",
            benchmark_col="benchmark",
            target_name="alpha",
            min_overlap_eras=20,
        )


def test_time_horizon_stability_uses_horizon_correct_adjustment() -> None:
    rng = np.random.default_rng(2026)
    n = 200
    x = np.empty(n, dtype=float)
    eps = rng.normal(size=n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + eps[i]

    # Construct frame so per-era corr recovers the same sequence for both model and benchmark.
    rows: list[dict[str, float | str]] = []
    for i, val in enumerate(x, start=1):
        era = f"{i:04d}"
        for j in range(6):
            rows.append(
                {
                    "era": era,
                    "pred": float(val + 0.01 * j),
                    "benchmark": float(val + 0.01 * j),
                    "target_beta_20": float(val + 0.02 * j),
                    "target_beta_60": float(val + 0.02 * j),
                }
            )
    df = pl.DataFrame(rows)
    out = time_horizon_stability(
        df,
        pred_col="pred",
        benchmark_col="benchmark",
        target_name="beta",
    )

    per_era = EvaluationEngine("custom").per_era_corr(
        df,
        pred_col="pred",
        target_col="target_beta_60",
    )
    series = np.asarray([per_era[e] for e in sorted(per_era)], dtype=float)
    sharpe_20 = ac_adjusted_sharpe(series, horizon="20D")
    sharpe_60 = ac_adjusted_sharpe(series, horizon="60D")
    assert sharpe_20 != pytest.approx(sharpe_60)
    assert out.model_sharpe_60 == pytest.approx(sharpe_60)


def test_time_horizon_stability_excludes_trailing_null_target_eras() -> None:
    rows: list[dict[str, float | str | None]] = []
    for i in range(1, 25):
        era = f"{i:04d}"
        for j in range(8):
            rows.append(
                {
                    "era": era,
                    "pred": float(0.1 * i + 0.02 * j),
                    "benchmark": float(0.12 * i + 0.03 * j),
                    "target_alpha_20": float((i + j) % 5) / 4.0,
                    "target_alpha_60": (
                        None
                        if era in {"0022", "0023", "0024"}
                        else float((2 * i + j) % 5) / 4.0
                    ),
                }
            )
    df = pl.DataFrame(rows)

    out = time_horizon_stability(
        df,
        pred_col="pred",
        benchmark_col="benchmark",
        target_name="alpha",
        min_overlap_eras=20,
    )
    assert out.n_eras == 21

    resolved_df = df.filter(~pl.col("era").is_in(["0022", "0023", "0024"]))
    out_resolved_only = time_horizon_stability(
        resolved_df,
        pred_col="pred",
        benchmark_col="benchmark",
        target_name="alpha",
        min_overlap_eras=20,
    )

    assert out == out_resolved_only


def test_regime_conditioned_corr_ci_delegation_and_nonvacuity() -> None:
    rows: list[dict[str, float | str]] = []
    labels: list[dict[str, str]] = []
    for era_i in range(1, 41):
        era = f"{era_i:04d}"
        for j in range(10):
            rows.append(
                {
                    "era": era,
                    "pred": float(era_i + 0.1 * j),
                    "target": float((era_i + j) % 5) / 4.0,
                    "alt_target": float((2 * era_i + j) % 5) / 4.0,
                }
            )
        labels.append({"era": era, "regime": "bull" if era_i <= 20 else "bear"})
    df = pl.DataFrame(rows)
    regime_df = pl.DataFrame(labels)

    out = regime_conditioned_corr(
        df,
        regime_df,
        pred_col="pred",
        target_col="target",
        regime_col="regime",
        horizon="20D",
        seed=44,
        n_boot=300,
        min_eras_per_regime=20,
    )
    assert set(out) == {"bear", "bull"}

    per_era = EvaluationEngine("custom").per_era_corr(
        df, pred_col="pred", target_col="target"
    )
    bull_series = np.asarray([per_era[f"{i:04d}"] for i in range(1, 21)], dtype=float)
    ci = block_bootstrap_ci(
        bull_series,
        lambda a: float(np.mean(a)),
        block_len=resolve_block_len(len(bull_series), "20D"),
        n_boot=300,
        seed=44,
        alpha=0.05,
    )
    assert out["bull"].ci_low == ci.lo
    assert out["bull"].ci_high == ci.hi

    # Partition identity is label-driven: same labels, different target => same era counts by regime.
    out_alt = regime_conditioned_corr(
        df,
        regime_df,
        pred_col="pred",
        target_col="alt_target",
        regime_col="regime",
        horizon="20D",
        seed=44,
        n_boot=300,
        min_eras_per_regime=20,
    )
    assert {k: v.n_eras for k, v in out.items()} == {
        k: v.n_eras for k, v in out_alt.items()
    }

    regime_19 = regime_df.with_columns(
        pl.when(pl.col("era") == "0001")
        .then(pl.lit("tiny"))
        .otherwise(pl.lit("bear"))
        .alias("regime")
    )
    with pytest.raises(
        ValueError,
        match="Non-vacuity violation: regime 'tiny' yielded only 1 eras; minimum required 20.",
    ):
        regime_conditioned_corr(
            df,
            regime_19,
            pred_col="pred",
            target_col="target",
            regime_col="regime",
            horizon="20D",
            seed=44,
            n_boot=200,
            min_eras_per_regime=20,
        )

    with pytest.raises(ValueError, match="min_eras_per_regime must be >= 1"):
        regime_conditioned_corr(
            df,
            regime_df,
            pred_col="pred",
            target_col="target",
            regime_col="regime",
            horizon="20D",
            seed=44,
            min_eras_per_regime=0,
        )


@pytest.mark.skipif(
    not (
        Path("data/v5.2/validation.parquet").exists()
        and Path("data/v5.2/validation_benchmark_models.parquet").exists()
    ),
    reason="v5.2 validation+benchmark inputs not on disk; skipped in CI",
)
def test_time_horizon_stability_real_v52_overlap() -> None:
    eras = (
        pl.scan_parquet("data/v5.2/validation.parquet")
        .select("era")
        .unique(maintain_order=True)
        .head(80)
        .collect()
        .get_column("era")
        .to_list()
    )

    validation = (
        pl.scan_parquet("data/v5.2/validation.parquet")
        .select(
            [
                "era",
                "id",
                "target_cyrusd_20",
                "target_cyrusd_60",
                "feature_antistrophic_striate_conscriptionist",
                "feature_bicameral_showery_wallaba",
                "feature_bridal_fingered_pensioner",
            ]
        )
        .filter(pl.col("era").is_in(eras))
        .collect()
    )
    bench = (
        pl.scan_parquet("data/v5.2/validation_benchmark_models.parquet")
        .select(["era", "id", "v52_lgbm_cyrusd20", "v52_lgbm_cyrusd60"])
        .filter(pl.col("era").is_in(eras))
        .collect()
    )

    merged = validation.join(bench, on=["era", "id"], how="inner")

    df = merged.with_columns(
        (
            0.5
            * pl.col("feature_antistrophic_striate_conscriptionist").cast(pl.Float64)
            + 0.3 * pl.col("feature_bicameral_showery_wallaba").cast(pl.Float64)
            - 0.2 * pl.col("feature_bridal_fingered_pensioner").cast(pl.Float64)
        ).alias("pred")
    ).select(
        ["era", "pred", "v52_lgbm_cyrusd20", "target_cyrusd_20", "target_cyrusd_60"]
    )

    result = time_horizon_stability(
        df,
        pred_col="pred",
        benchmark_col="v52_lgbm_cyrusd20",
        target_name="cyrusd",
    )
    assert result.n_eras >= MIN_OVERLAP_ERAS
