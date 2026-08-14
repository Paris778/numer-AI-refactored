"""Unit tests for EvaluationEngine summary and boundary behavior."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nmr.evaluation import (
    EvaluationEngine,
    MetricSummary,
    clean_frame,
    downside_era_indices,
    sorted_era_labels,
)


def test_summarize_known_values() -> None:
    engine = EvaluationEngine("custom")
    summary = engine.summarize({"1": 1.0, "2": -2.0, "3": 1.0})

    assert isinstance(summary, MetricSummary)
    assert summary.mean == pytest.approx(0.0)
    assert summary.std == pytest.approx(math.sqrt(2.0))
    assert summary.sharpe == pytest.approx(0.0)
    assert summary.max_drawdown == pytest.approx(2.0)


def test_summarize_zero_std_sets_sharpe_to_zero() -> None:
    engine = EvaluationEngine("custom")
    summary = engine.summarize({"1": 2.0, "2": 2.0, "3": 2.0})
    assert summary.std == pytest.approx(0.0)
    assert summary.sharpe == pytest.approx(0.0)
    assert summary.max_drawdown == pytest.approx(0.0)


def test_summarize_requires_non_empty_mapping() -> None:
    engine = EvaluationEngine("custom")
    with pytest.raises(ValueError, match="at least one era score"):
        engine.summarize({})


def test_per_era_keys_are_numeric_ordered() -> None:
    engine = EvaluationEngine("custom")
    df = pl.DataFrame(
        {
            "era": ["10", "2", "1", "2", "10", "1"],
            "pred": [0.9, 0.3, 0.2, 0.8, 0.1, 0.7],
            "target": [0.2, 0.5, 0.7, 0.1, 0.9, 0.3],
        }
    )

    result = engine.per_era_corr(df, pred_col="pred", target_col="target")
    assert list(result) == ["1", "2", "10"]


@pytest.mark.parametrize("backend", ["custom", "official"])
def test_degenerate_corr_returns_zero(backend: str) -> None:
    engine = EvaluationEngine(backend)
    df = pl.DataFrame(
        {
            "era": ["1", "1", "1", "1"],
            "pred": [0.5, 0.5, 0.5, 0.5],
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    result = engine.per_era_corr(df, pred_col="pred", target_col="target")
    assert result == {"1": 0.0}


@pytest.mark.parametrize("backend", ["custom", "official"])
def test_degenerate_fnc_returns_zero(backend: str) -> None:
    engine = EvaluationEngine(backend)
    df = pl.DataFrame(
        {
            "era": ["1", "1", "1", "1"],
            "pred": [0.5, 0.5, 0.5, 0.5],
            "target": [0.1, 0.2, 0.3, 0.4],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [2.0, 1.0, 4.0, 3.0],
        }
    )
    result = engine.per_era_fnc(
        df,
        pred_col="pred",
        feature_cols=["f1", "f2"],
        target_col="target",
    )
    assert result == {"1": 0.0}


@pytest.mark.parametrize("backend", ["custom", "official"])
def test_degenerate_mmc_returns_zero(backend: str) -> None:
    engine = EvaluationEngine(backend)
    df = pl.DataFrame(
        {
            "era": ["1", "1", "1", "1"],
            "pred": [0.5, 0.5, 0.5, 0.5],
            "meta": [0.2, 0.4, 0.6, 0.8],
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    result = engine.per_era_mmc(
        df,
        pred_col="pred",
        meta_col="meta",
        target_col="target",
    )
    assert result == {"1": 0.0}


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend="):
        EvaluationEngine("bogus")


def test_empty_feature_cols_raise() -> None:
    engine = EvaluationEngine("custom")
    df = pl.DataFrame({"era": ["1"], "pred": [0.1], "target": [0.2]})
    with pytest.raises(ValueError, match="feature_cols"):
        engine.per_era_fnc(df, pred_col="pred", feature_cols=[], target_col="target")


def test_sorted_era_labels_sorts_numerically() -> None:
    assert sorted_era_labels(["10", "2", "1"]) == ["1", "2", "10"]
    with pytest.raises(ValueError):
        sorted_era_labels(["a", "b"])


def test_clean_frame_drops_nulls_and_nonfinite() -> None:
    df = pl.DataFrame(
        {
            "era": ["1", "1", "1"],
            "pred": [0.1, None, float("inf")],
            "target": [1.0, 2.0, 3.0],
        }
    )
    out = clean_frame(df, ["pred", "target"])
    assert out.to_dicts() == [{"pred": 0.1, "target": 1.0}]


def test_per_era_metric_handles_appearance_order_eras() -> None:
    df = pl.DataFrame(
        {
            "era": ["5", "5", "2", "2"],  # appearance order != numeric order
            "pred": [1.0, 2.0, 3.0, 4.0],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    engine = EvaluationEngine("custom")
    out = engine.per_era_corr(df, pred_col="pred", target_col="target")
    assert list(out.keys()) == ["2", "5"]


# ---------------------------------------------------------------------------
# Mathematical property tests (definitional invariants, beyond oracle parity)
# ---------------------------------------------------------------------------


def _corr_frame() -> pl.DataFrame:
    """3 eras, 60 rows each, distinct values (no ties -> exact rank symmetry)."""
    rng = np.random.default_rng(20260805)
    rows: list[dict[str, float | str]] = []
    for era in ("1", "2", "3"):
        for idx in range(60):
            pred = float(rng.normal(loc=0.1 * int(era), scale=1.0))
            meta = float(rng.normal(loc=0.05 * int(era), scale=1.0))
            target = float(0.4 * pred - 0.3 * meta + rng.normal(scale=0.5))
            rows.append(
                {
                    "era": era,
                    "pred": pred,
                    "meta": meta,
                    "target": target,
                    "f1": pred,
                    "f2": meta,
                }
            )
    return pl.DataFrame(rows)


def test_corr_is_invariant_to_strictly_monotone_pred_transform() -> None:
    df = _corr_frame()
    engine = EvaluationEngine("custom")
    base = engine.per_era_corr(df, pred_col="pred", target_col="target")
    transformed = df.with_columns(
        (pl.col("pred") ** 3.0 + 5.0 * pl.col("pred")).alias("pred")
    )
    moved = engine.per_era_corr(transformed, pred_col="pred", target_col="target")
    assert list(base) == list(moved)
    for era in base:
        assert base[era] == pytest.approx(moved[era], abs=1e-12)


def test_corr_negation_flips_sign_exactly() -> None:
    df = _corr_frame()
    engine = EvaluationEngine("custom")
    base = engine.per_era_corr(df, pred_col="pred", target_col="target")
    negated = engine.per_era_corr(
        df.with_columns((-pl.col("pred")).alias("pred")),
        pred_col="pred",
        target_col="target",
    )
    for era in base:
        assert negated[era] == pytest.approx(-base[era], abs=1e-12)


def test_corr_self_is_high_and_bounded() -> None:
    """Numerai CORR is Pearson of two different transforms of pred, so
    CORR(pred, pred) is provably NOT exactly 1.0 (it approaches 1 as the
    sample becomes near-normal); it must be high and bounded by 1."""
    df = _corr_frame().with_columns(pl.col("pred").alias("pred_copy"))
    engine = EvaluationEngine("custom")
    self_corr = engine.per_era_corr(df, pred_col="pred_copy", target_col="pred")
    for era in self_corr:
        assert 0.9 < self_corr[era] <= 1.0 + 1e-12
    corr = engine.per_era_corr(df, pred_col="pred", target_col="target")
    for era in corr:
        assert abs(corr[era]) <= 1.0 + 1e-12


def test_mmc_of_pred_equal_to_meta_is_zero() -> None:
    df = _corr_frame().with_columns(pl.col("meta").alias("pred"))
    engine = EvaluationEngine("custom")
    out = engine.per_era_mmc(
        df, pred_col="pred", meta_col="meta", target_col="target"
    )
    for era in out:
        assert out[era] == pytest.approx(0.0, abs=1e-12)


def test_mmc_negation_flips_sign_exactly() -> None:
    df = _corr_frame()
    engine = EvaluationEngine("custom")
    base = engine.per_era_mmc(
        df, pred_col="pred", meta_col="meta", target_col="target"
    )
    negated = engine.per_era_mmc(
        df.with_columns((-pl.col("pred")).alias("pred")),
        pred_col="pred",
        meta_col="meta",
        target_col="target",
    )
    for era in base:
        assert negated[era] == pytest.approx(-base[era], abs=1e-12)


def test_fnc_is_invariant_to_affine_feature_transform() -> None:
    df = _corr_frame()
    engine = EvaluationEngine("custom")
    base = engine.per_era_fnc(
        df, pred_col="pred", feature_cols=["f1", "f2"], target_col="target"
    )
    # 2*f1 + 3 spans the same column space as f1 (intercept already in design).
    transformed = df.with_columns((2.0 * pl.col("f1") + 3.0).alias("f1"))
    moved = engine.per_era_fnc(
        transformed, pred_col="pred", feature_cols=["f1", "f2"], target_col="target"
    )
    for era in base:
        assert base[era] == pytest.approx(moved[era], abs=1e-8)


def test_cwmm_of_pred_equal_to_meta_is_one() -> None:
    df = _corr_frame().with_columns(pl.col("meta").alias("pred"))
    engine = EvaluationEngine("custom")
    out = engine.per_era_cwmm(
        df, pred_col="pred", meta_col="meta", min_overlap_eras=2
    )
    for era in out:
        assert out[era] == pytest.approx(1.0, abs=1e-12)


def test_bmc_of_pred_equal_to_benchmark_is_zero() -> None:
    df = _corr_frame().with_columns(pl.col("pred").alias("bench"))
    engine = EvaluationEngine("custom")
    out = engine.per_era_bmc(
        df,
        pred_col="pred",
        benchmark_col="bench",
        target_col="target",
        min_overlap_eras=2,
    )
    for era in out:
        assert out[era] == pytest.approx(0.0, abs=1e-12)


def test_single_row_era_scores_zero() -> None:
    df = pl.DataFrame(
        {
            "era": ["1", "2", "2"],
            "pred": [0.5, 0.3, 0.7],
            "target": [0.9, 0.2, 0.8],
        }
    )
    engine = EvaluationEngine("custom")
    out = engine.per_era_corr(df, pred_col="pred", target_col="target")
    assert out == {"1": 0.0, "2": pytest.approx(1.0, abs=1e-12)}


def test_constant_target_scores_zero() -> None:
    df = pl.DataFrame(
        {
            "era": ["1", "1", "1", "1"],
            "pred": [0.1, 0.2, 0.3, 0.4],
            "target": [0.5, 0.5, 0.5, 0.5],
        }
    )
    engine = EvaluationEngine("custom")
    assert engine.per_era_corr(df, pred_col="pred", target_col="target") == {
        "1": 0.0
    }


def test_bmc_skips_all_null_benchmark_era() -> None:
    df = _corr_frame().with_columns(
        pl.when(pl.col("era") == "2")
        .then(None)
        .otherwise(pl.col("meta"))
        .alias("bench")
    )
    engine = EvaluationEngine("custom")
    out = engine.per_era_bmc(
        df,
        pred_col="pred",
        benchmark_col="bench",
        target_col="target",
        min_overlap_eras=2,
    )
    assert "2" not in out
    assert set(out) == {"1", "3"}


def test_bmc_validation_branches() -> None:
    df = _corr_frame().with_columns(pl.col("meta").alias("bench"))
    engine = EvaluationEngine("custom")

    with pytest.raises(ValueError, match="Missing required columns"):
        engine.per_era_bmc(
            df,
            pred_col="pred",
            benchmark_col="missing_bench",
            target_col="target",
            min_overlap_eras=2,
        )
    with pytest.raises(ValueError, match="min_overlap_eras"):
        engine.per_era_bmc(
            df,
            pred_col="pred",
            benchmark_col="bench",
            target_col="target",
            min_overlap_eras=0,
        )
    with pytest.raises(ValueError, match="Non-vacuity"):
        engine.per_era_bmc(
            df.filter(pl.col("era") == "1"),
            pred_col="pred",
            benchmark_col="bench",
            target_col="target",
            min_overlap_eras=2,
        )


def test_should_short_circuit_defensive_branches() -> None:
    engine = EvaluationEngine("custom")
    assert engine._should_short_circuit(np.array([])) is True
    assert engine._should_short_circuit(np.array([0.1]), np.array([0.2])) is True
    assert (
        engine._should_short_circuit(
            np.array([0.1, 0.2, 0.3]), np.array([0.4, np.nan, 0.6])
        )
        is True
    )
    assert (
        engine._should_short_circuit(np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]))
        is False
    )


def test_pearson_corr_constant_input_returns_zero() -> None:
    engine = EvaluationEngine("custom")
    # Reachable only via direct call: _should_short_circuit pre-empts constants
    # in the metric path, but the branch must still be safe.
    assert engine._pearson_corr(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    assert engine._pearson_corr(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0])) == 0.0
    # Normal case still exact.
    left = np.array([1.0, 2.0, 3.0, 4.0])
    right = np.array([2.0, 4.0, 6.0, 8.0])
    assert engine._pearson_corr(left, right) == pytest.approx(1.0, abs=1e-12)


def test_downside_era_indices_strict() -> None:
    meta_corr = {"0002": 0.01, "0001": -0.02, "0003": 0.0, "0004": -0.01}
    assert downside_era_indices(meta_corr) == ["0001", "0004"]
    assert downside_era_indices(meta_corr, threshold=0.0) == ["0001", "0004"]


def test_downside_era_indices_rejects_non_numeric_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="Non-numeric era label"):
        downside_era_indices({"X": 0.1})
    with pytest.raises(ValueError, match="threshold"):
        downside_era_indices({"0001": -0.1}, threshold=np.nan)
