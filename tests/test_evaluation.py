from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.evaluation import EvaluationEngine
from src.risk import NeutralizationEngine


@pytest.fixture(scope="module")
def engine() -> EvaluationEngine:
    return EvaluationEngine(
        era_col="era",
        prediction_col="prediction",
        target_col="target",
        id_col="id",
    )


@pytest.fixture(scope="module")
def era_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [f"row_{index}" for index in range(8)],
            "era": ["0001"] * 4 + ["0002"] * 4,
            "prediction": [0.1, 0.3, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8],
            "target": [0.0, 0.2, 0.8, 1.0, 0.1, 0.3, 0.7, 0.9],
            "benchmark": [0.15, 0.25, 0.75, 0.85, 0.1, 0.35, 0.65, 0.95],
            "feature_a": [0.0, 1.0, 0.0, 1.0, 0.5, 1.5, 0.5, 1.5],
            "feature_b": [1.0, 0.0, 1.0, 0.0, 1.5, 0.5, 1.5, 0.5],
        }
    )


def test_safe_pearson_handles_perfect_correlation(engine: EvaluationEngine) -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    correlation = engine.safe_pearson(values, values)

    assert correlation == pytest.approx(1.0, abs=1e-12)


def test_numerai_corr_returns_zero_for_zero_variance_predictions(
    engine: EvaluationEngine,
) -> None:
    predictions = np.ones(6, dtype=np.float64)
    targets = np.linspace(0.0, 1.0, 6, dtype=np.float64)

    corr = engine.numerai_corr(predictions, targets)

    assert corr == 0.0


def test_evaluate_eras_handles_missing_benchmark_by_returning_none(
    engine: EvaluationEngine,
    era_frame: pl.DataFrame,
) -> None:
    metrics = engine.evaluate_eras(
        era_frame.select(["id", "era", "prediction", "target"])
    )

    assert len(metrics) == 2
    assert all(row.benchmark_corr is None for row in metrics)


def test_feature_exposure_is_vectorized_and_non_negative(
    engine: EvaluationEngine,
    era_frame: pl.DataFrame,
) -> None:
    first_era = era_frame.filter(pl.col("era") == "0001")

    exposure = engine.compute_max_feature_exposure(
        first_era.get_column("prediction").to_numpy(),
        first_era.select(["feature_a", "feature_b"]).to_numpy(),
    )

    assert exposure >= 0.0


def test_compute_max_drawdown_tracks_peak_to_trough(engine: EvaluationEngine) -> None:
    series = np.array([0.3, 0.2, -0.8, 0.1], dtype=np.float64)

    drawdown = engine.compute_max_drawdown(series)

    assert drawdown == pytest.approx(0.8, abs=1e-12)


def test_evaluate_eras_uses_cached_neutralization_when_available(
    engine: EvaluationEngine,
    era_frame: pl.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_era = era_frame.filter(pl.col("era") == "0001")
    neutralization_engine = NeutralizationEngine(cache_root=tmp_path)
    feature_names = ("feature_a", "feature_b")
    feature_matrix = first_era.select(list(feature_names)).to_numpy()
    pseudo_inverse = neutralization_engine.compute_era_pseudo_inverse(feature_matrix)
    cache_path = neutralization_engine.cache_path("small", "0001")
    neutralization_engine._write_cache_artifacts(
        cache_path,
        pseudo_inverse=pseudo_inverse,
        row_ids=tuple(first_era.get_column("id").to_list()),
        feature_names=feature_names,
        era="0001",
        dataset_name="validation",
        subset_name="small",
        add_intercept=neutralization_engine.add_intercept,
    )

    def fail_if_recomputed(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("pseudo-inverse should come from cache")

    monkeypatch.setattr(
        neutralization_engine,
        "compute_era_pseudo_inverse",
        fail_if_recomputed,
    )

    metrics = engine.evaluate_eras(
        first_era,
        feature_columns=feature_names,
        neutralization_engine=neutralization_engine,
        neutralization_subset_name="small",
    )

    assert len(metrics) == 1
    assert metrics[0].fnc is not None


def test_fnc_is_rank_invariant_to_neutralized_residual_magnitude_distortion(
    engine: EvaluationEngine,
    era_frame: pl.DataFrame,
) -> None:
    first_era = era_frame.filter(pl.col("era") == "0001")
    base_residual = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)
    distorted_residual = np.array([-2000.0, -0.5, 0.5, 2000000.0], dtype=np.float64)

    class StubNeutralizationEngine:
        def __init__(self, residual: np.ndarray) -> None:
            self._residual = residual

        def neutralize_tensor(
            self,
            predictions: np.ndarray,
            feature_matrix: np.ndarray,
            *,
            proportion: float = 1.0,
            pseudo_inverse: np.ndarray | None = None,
        ) -> np.ndarray:
            return self._residual.copy()

    base_metric = engine.evaluate_eras(
        first_era,
        feature_columns=("feature_a", "feature_b"),
        neutralization_engine=StubNeutralizationEngine(base_residual),
    )[0].fnc
    distorted_metric = engine.evaluate_eras(
        first_era,
        feature_columns=("feature_a", "feature_b"),
        neutralization_engine=StubNeutralizationEngine(distorted_residual),
    )[0].fnc

    assert base_metric is not None
    assert distorted_metric is not None
    assert distorted_metric == pytest.approx(base_metric, abs=1e-12)


def test_prediction_correlation_uses_tail_transformed_rank_space(
    engine: EvaluationEngine,
) -> None:
    preds_a = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    preds_b = np.array([0.05, 0.3, 0.7, 0.95], dtype=np.float64)

    left = engine.rank_to_gaussian(preds_a, epsilon=engine.epsilon)
    right = engine.rank_to_gaussian(preds_b, epsilon=engine.epsilon)
    expected = engine.safe_pearson(
        np.sign(left) * np.abs(left) ** 1.5,
        np.sign(right) * np.abs(right) ** 1.5,
        epsilon=engine.epsilon,
    )

    actual = engine.prediction_correlation(preds_a, preds_b)

    assert actual == pytest.approx(expected, abs=1e-12)


def test_summary_and_fast_fail_gate(
    engine: EvaluationEngine, era_frame: pl.DataFrame
) -> None:
    metrics = engine.evaluate_eras(
        era_frame,
        feature_columns=("feature_a", "feature_b"),
        benchmark_prediction_col="benchmark",
    )
    summary = engine.summarize(metrics)
    gate = engine.fast_fail_gate(
        summary,
        min_mean_corr=-1.0,
        min_sharpe_corr=-10.0,
        max_drawdown_corr=10.0,
        max_feature_exposure=1.0,
        max_benchmark_corr=1.0,
    )

    assert summary.eras_evaluated == 2
    assert summary.mean_benchmark_corr is not None
    assert summary.max_feature_exposure is not None
    assert gate.passed is True
