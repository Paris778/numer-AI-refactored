"""Tests for nmr.ensemble.Ensembler."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import nmr.ensemble as ensemble_module
import nmr.evaluation as evaluation_module
from nmr._transforms import rank_gaussianize, rank_gaussianize_unit_variance
from nmr.ensemble import Ensembler
from nmr.evaluation import EvaluationEngine


def _ensemble_frame() -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for era_num in range(1, 4):
        for row_num in range(8):
            base = (era_num * 11 + row_num * 7) / 100.0
            target = 0.7 * base + 0.2 * (row_num / 10.0)
            rows.append(
                {
                    "id": f"{era_num}_{row_num}",
                    "era": str(era_num),
                    "pred_a": base,
                    "pred_b": 3.0 * base + 1.0,
                    "pred_c": -0.8 * base + (row_num / 20.0),
                    "target": target,
                }
            )
    return pl.DataFrame(rows)


def test_rank_normalize_is_per_era_and_deterministic() -> None:
    df = _ensemble_frame()
    ensembler = Ensembler()

    first = ensembler.rank_normalize(df, pred_cols=["pred_a", "pred_c"])
    second = ensembler.rank_normalize(df, pred_cols=["pred_a", "pred_c"])

    assert first.equals(second)
    for era in ["1", "2", "3"]:
        era_df = first.filter(pl.col("era") == era)
        for col in ["pred_a", "pred_c"]:
            values = era_df.get_column(col).to_numpy()
            assert np.std(values, ddof=0) == pytest.approx(1.0)


def test_blend_is_rank_invariant_under_strictly_monotonic_transform() -> None:
    df = _ensemble_frame()
    transformed = df.with_columns((pl.col("pred_a") * 7.0 + 5.0).alias("pred_a"))
    ensembler = Ensembler()

    base = ensembler.blend(df, pred_cols=["pred_a", "pred_c"], weights=[0.6, 0.4])
    moved = ensembler.blend(
        transformed,
        pred_cols=["pred_a", "pred_c"],
        weights=[0.6, 0.4],
    )

    assert np.allclose(
        base.get_column("prediction").to_numpy(),
        moved.get_column("prediction").to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )


def test_single_weight_reduces_to_first_column_rank_gaussianized_form() -> None:
    df = _ensemble_frame()
    ensembler = Ensembler()

    blended = ensembler.blend(df, pred_cols=["pred_a", "pred_c"], weights=[1.0, 0.0])

    expected_parts: list[np.ndarray] = []
    for era in ["1", "2", "3"]:
        era_values = df.filter(pl.col("era") == era).get_column("pred_a").to_numpy()
        expected_parts.append(rank_gaussianize(era_values))
    expected = np.concatenate(expected_parts)

    assert np.allclose(
        blended.get_column("prediction").to_numpy(),
        expected,
        atol=1e-12,
        rtol=0.0,
    )


def test_equal_weights_are_symmetric_in_column_order() -> None:
    df = _ensemble_frame()
    ensembler = Ensembler()

    left = ensembler.blend(df, pred_cols=["pred_a", "pred_c"])
    right = ensembler.blend(df, pred_cols=["pred_c", "pred_a"])

    assert np.allclose(
        left.get_column("prediction").to_numpy(),
        right.get_column("prediction").to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )


def test_transform_helper_is_shared_with_evaluation() -> None:
    assert evaluation_module.rank_gaussianize is rank_gaussianize
    assert (
        ensemble_module.rank_gaussianize_unit_variance is rank_gaussianize_unit_variance
    )


def test_per_era_independence() -> None:
    df = _ensemble_frame()
    changed = df.with_columns(
        pl.when(pl.col("era") == "3")
        .then(pl.col("pred_c") * -50.0)
        .otherwise(pl.col("pred_c"))
        .alias("pred_c")
    )
    ensembler = Ensembler()

    base = ensembler.blend(df, pred_cols=["pred_a", "pred_c"], weights=[0.5, 0.5])
    moved = ensembler.blend(
        changed,
        pred_cols=["pred_a", "pred_c"],
        weights=[0.5, 0.5],
    )

    assert np.allclose(
        base.filter(pl.col("era") == "1").get_column("prediction").to_numpy(),
        moved.filter(pl.col("era") == "1").get_column("prediction").to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )


def test_weight_length_validation() -> None:
    df = _ensemble_frame()
    with pytest.raises(ValueError, match="weights length"):
        Ensembler().blend(df, pred_cols=["pred_a", "pred_b"], weights=[1.0])


def test_learn_weights_is_deterministic_and_improves_constructed_case() -> None:
    df = _ensemble_frame().with_columns(
        [
            (pl.col("target") + 0.02 * pl.col("pred_a")).alias("pred_good"),
            (-pl.col("target") + 0.01 * pl.col("pred_c")).alias("pred_bad"),
        ]
    )
    ensembler = Ensembler()
    engine = EvaluationEngine("custom")

    first = ensembler.learn_weights(
        df,
        pred_cols=["pred_good", "pred_bad"],
        target_col="target",
        method="ridge",
    )
    second = ensembler.learn_weights(
        df,
        pred_cols=["pred_good", "pred_bad"],
        target_col="target",
        method="ridge",
    )
    assert first == pytest.approx(second, abs=1e-12)

    equal_blend = ensembler.blend(df, pred_cols=["pred_good", "pred_bad"])
    learned_blend = ensembler.blend(
        df,
        pred_cols=["pred_good", "pred_bad"],
        weights=first,
    )

    equal_scores = engine.per_era_corr(
        equal_blend.with_columns(pl.col("target")),
        pred_col="prediction",
        target_col="target",
    )
    learned_scores = engine.per_era_corr(
        learned_blend.with_columns(pl.col("target")),
        pred_col="prediction",
        target_col="target",
    )

    assert np.mean(list(learned_scores.values())) >= np.mean(
        list(equal_scores.values())
    )


def test_learn_weights_non_negative_returns_non_negative_tuple() -> None:
    df = _ensemble_frame().with_columns(
        [
            (pl.col("target") + 0.02 * pl.col("pred_a")).alias("pred_good"),
            (0.5 * pl.col("target") + 0.01 * pl.col("pred_b")).alias("pred_ok"),
        ]
    )

    weights = Ensembler().learn_weights(
        df,
        pred_cols=["pred_good", "pred_ok"],
        target_col="target",
        method="non_negative",
    )

    assert len(weights) == 2
    assert all(weight >= -1e-12 for weight in weights)


def test_learn_weights_ridge_satisfies_normal_equations() -> None:
    """The ridge solution must satisfy (X^T X / n + I) w = X^T y / n."""
    df = _ensemble_frame().with_columns(
        [
            (pl.col("target") + 0.02 * pl.col("pred_a")).alias("pred_good"),
            (-pl.col("target") + 0.01 * pl.col("pred_c")).alias("pred_bad"),
        ]
    )
    ensembler = Ensembler()
    weights = ensembler.learn_weights(
        df,
        pred_cols=["pred_good", "pred_bad"],
        target_col="target",
        method="ridge",
    )

    # Rebuild the same rank-normalized design the learner uses.
    clean = df.select(["era", "pred_good", "pred_bad", "target"]).drop_nulls()
    normalized = ensembler.rank_normalize(clean, pred_cols=["pred_good", "pred_bad"])
    design = normalized.select(["pred_good", "pred_bad"]).cast(pl.Float64).to_numpy()
    target = normalized.get_column("target").cast(pl.Float64).to_numpy()

    gram = (design.T @ design) / len(target) + np.eye(design.shape[1])
    rhs = (design.T @ target) / len(target)
    residual = gram @ np.asarray(weights) - rhs
    assert np.allclose(residual, 0.0, atol=1e-8)


def test_learn_weights_ridge_is_invariant_to_replicated_eras() -> None:
    frame = _ensemble_frame().with_columns(
        [
            (pl.col("target") + 0.02 * pl.col("pred_a")).alias("pred_good"),
            (-pl.col("target") + 0.01 * pl.col("pred_c")).alias("pred_bad"),
        ]
    )
    replicated = pl.concat(
        [
            frame,
            frame.with_columns((pl.col("era").cast(pl.Int64) + 10).cast(pl.String)),
        ]
    )

    base_weights = Ensembler().learn_weights(
        frame,
        pred_cols=["pred_good", "pred_bad"],
        target_col="target",
        method="ridge",
    )
    replicated_weights = Ensembler().learn_weights(
        replicated,
        pred_cols=["pred_good", "pred_bad"],
        target_col="target",
        method="ridge",
    )

    assert replicated_weights == pytest.approx(base_weights, abs=1e-12)


def test_learn_weights_ridge_minimizes_ridge_objective() -> None:
    df = _ensemble_frame().with_columns(
        [
            (pl.col("target") + 0.02 * pl.col("pred_a")).alias("pred_good"),
            (-pl.col("target") + 0.01 * pl.col("pred_c")).alias("pred_bad"),
        ]
    )
    ensembler = Ensembler()
    weights = np.asarray(
        ensembler.learn_weights(
            df,
            pred_cols=["pred_good", "pred_bad"],
            target_col="target",
            method="ridge",
        )
    )

    clean = df.select(["era", "pred_good", "pred_bad", "target"]).drop_nulls()
    normalized = ensembler.rank_normalize(clean, pred_cols=["pred_good", "pred_bad"])
    design = normalized.select(["pred_good", "pred_bad"]).cast(pl.Float64).to_numpy()
    target = normalized.get_column("target").cast(pl.Float64).to_numpy()

    def objective(w: np.ndarray) -> float:
        return float(np.mean((design @ w - target) ** 2) + np.sum(w**2))

    assert objective(weights) <= objective(np.array([0.0, 0.0])) + 1e-9
    assert objective(weights) <= objective(np.array([1.0, -1.0])) + 1e-9


def test_learn_weights_nnls_matches_scipy_directly() -> None:
    from scipy.optimize import nnls as scipy_nnls

    df = _ensemble_frame().with_columns(
        [
            (pl.col("target") + 0.02 * pl.col("pred_a")).alias("pred_good"),
            (0.5 * pl.col("target") + 0.01 * pl.col("pred_b")).alias("pred_ok"),
        ]
    )
    ensembler = Ensembler()
    weights = ensembler.learn_weights(
        df,
        pred_cols=["pred_good", "pred_ok"],
        target_col="target",
        method="non_negative",
    )

    clean = df.select(["era", "pred_good", "pred_ok", "target"]).drop_nulls()
    normalized = ensembler.rank_normalize(clean, pred_cols=["pred_good", "pred_ok"])
    design = normalized.select(["pred_good", "pred_ok"]).cast(pl.Float64).to_numpy()
    target = normalized.get_column("target").cast(pl.Float64).to_numpy()

    expected, _ = scipy_nnls(design, target)
    assert np.allclose(weights, expected, atol=1e-8)


def test_blend_single_second_weight_reduces_to_second_column() -> None:
    df = _ensemble_frame()
    ensembler = Ensembler()
    blended = ensembler.blend(df, pred_cols=["pred_a", "pred_c"], weights=[0.0, 1.0])

    expected_parts: list[np.ndarray] = []
    for era in ["1", "2", "3"]:
        era_values = df.filter(pl.col("era") == era).get_column("pred_c").to_numpy()
        expected_parts.append(rank_gaussianize(era_values))
    expected = np.concatenate(expected_parts)
    assert np.allclose(
        blended.get_column("prediction").to_numpy(),
        expected,
        atol=1e-12,
        rtol=0.0,
    )


def test_blend_default_matches_explicit_uniform_weights() -> None:
    df = _ensemble_frame()
    ensembler = Ensembler()
    default = ensembler.blend(df, pred_cols=["pred_a", "pred_c"])
    explicit = ensembler.blend(df, pred_cols=["pred_a", "pred_c"], weights=[0.5, 0.5])
    assert np.allclose(
        default.get_column("prediction").to_numpy(),
        explicit.get_column("prediction").to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )


def test_learn_weights_validation_branches() -> None:
    ensembler = Ensembler()
    df = _ensemble_frame()

    with pytest.raises(ValueError, match="at least one prediction"):
        ensembler.rank_normalize(df, pred_cols=[])
    with pytest.raises(ValueError, match="no usable rows"):
        ensembler.learn_weights(
            df.with_columns(pl.lit(None).alias("pred_a")),
            pred_cols=["pred_a", "pred_b"],
            target_col="target",
        )
    with pytest.raises(ValueError, match="no finite rows"):
        ensembler.learn_weights(
            df.with_columns(pl.lit(float("nan")).alias("target")),
            pred_cols=["pred_a", "pred_b"],
            target_col="target",
        )
    with pytest.raises(ValueError, match="method must be"):
        ensembler.learn_weights(
            df,
            pred_cols=["pred_a", "pred_b"],
            target_col="target",
            method="svm",
        )
    with pytest.raises(ValueError, match="weights must be finite"):
        ensembler.blend(df, pred_cols=["pred_a", "pred_b"], weights=[0.5, float("nan")])
