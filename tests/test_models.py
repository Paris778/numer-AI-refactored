from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.features import PurgedEraSplitter
from src.models import ModelOrchestrator, WeightedMultiTargetPredictor


@pytest.fixture(scope="module")
def toy_training_frame() -> pl.DataFrame:
    rows_per_era = 12
    eras = [f"{era:04d}" for era in range(1, 13) for _ in range(rows_per_era)]
    row_index = np.arange(len(eras), dtype=np.float64)
    feature_alpha = np.sin(row_index / 7.0)
    feature_beta = np.cos(row_index / 11.0)
    feature_gamma = (row_index % 5) / 5.0
    target = 0.6 * feature_alpha - 0.3 * feature_beta + 0.2 * feature_gamma
    return pl.DataFrame(
        {
            "id": [f"toy_{index:04d}" for index in range(len(eras))],
            "era": eras,
            "feature_alpha": feature_alpha,
            "feature_beta": feature_beta,
            "feature_gamma": feature_gamma,
            "target": target,
        }
    )


@pytest.fixture(scope="module")
def orchestrator() -> ModelOrchestrator:
    return ModelOrchestrator(
        feature_names=["feature_alpha", "feature_beta", "feature_gamma"],
        target_columns=["target"],
        model_library="lightgbm",
        prefer_gpu=False,
        early_stopping_rounds=10,
        model_params={
            "n_estimators": 80,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "min_child_samples": 5,
        },
    )


def test_anchor_fold_returns_finite_predictions(
    toy_training_frame: pl.DataFrame,
    orchestrator: ModelOrchestrator,
) -> None:
    train_df = toy_training_frame.filter(
        pl.col("era").is_in([f"{era:04d}" for era in range(1, 9)])
    )
    val_df = toy_training_frame.filter(
        pl.col("era").is_in([f"{era:04d}" for era in range(9, 11)])
    )

    result = orchestrator.train_anchor_fold(train_df, val_df)

    assert result.model_count == 1
    assert result.validation_predictions.shape[0] == val_df.height
    assert np.isfinite(result.validation_predictions).all()
    assert result.best_iteration >= 1
    assert result.backend == "lightgbm-cpu"


def test_cross_validation_returns_oof_predictions(
    toy_training_frame: pl.DataFrame,
    orchestrator: ModelOrchestrator,
) -> None:
    splitter = PurgedEraSplitter(n_splits=3, purge_buffer=1)

    result = orchestrator.train_cross_validation(toy_training_frame.lazy(), splitter)

    assert result.model_count == 3
    assert len(result.fold_results) == 3
    assert result.oof_predictions.shape[0] == toy_training_frame.height
    assert np.isfinite(result.oof_predictions).all()
    assert result.predictor.model_count == 3


def test_anchor_and_cv_modes_produce_different_prediction_surfaces(
    toy_training_frame: pl.DataFrame,
    orchestrator: ModelOrchestrator,
) -> None:
    train_df = toy_training_frame.filter(
        pl.col("era").is_in([f"{era:04d}" for era in range(1, 9)])
    )
    val_df = toy_training_frame.filter(
        pl.col("era").is_in([f"{era:04d}" for era in range(9, 11)])
    )
    splitter = PurgedEraSplitter(n_splits=3, purge_buffer=1)

    anchor_result = orchestrator.train_anchor_fold(train_df, val_df)
    cv_result = orchestrator.train_cross_validation(toy_training_frame, splitter)
    ensemble_predictions = orchestrator.predict_ensemble(
        val_df, models=cv_result.models
    )

    assert cv_result.model_count > anchor_result.model_count
    assert cv_result.total_fit_seconds > 0.0
    assert anchor_result.fit_seconds > 0.0
    assert not np.allclose(anchor_result.validation_predictions, ensemble_predictions)


def test_rank_average_predictions_matches_percentile_rank_ensemble(
    orchestrator: ModelOrchestrator,
) -> None:
    pred_a = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    pred_b = np.array([10.0, 20.0, 80.0, 90.0], dtype=np.float64)

    averaged = orchestrator.rank_average_predictions([pred_a, pred_b])

    expected = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    assert np.allclose(averaged, expected)


def test_multi_target_predictor_trains_and_predicts(
    toy_training_frame: pl.DataFrame,
) -> None:
    enriched = toy_training_frame.with_columns(
        (0.7 * pl.col("target") + 0.1).alias("target_aux")
    )
    orchestrator = ModelOrchestrator(
        feature_names=["feature_alpha", "feature_beta", "feature_gamma"],
        target_columns=["target", "target_aux"],
        model_library="lightgbm",
        prefer_gpu=False,
        early_stopping_rounds=10,
        model_params={
            "n_estimators": 40,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "min_child_samples": 5,
        },
    )

    result = orchestrator.train_cross_validation(
        enriched,
        PurgedEraSplitter(n_splits=3, purge_buffer=1),
    )
    prediction_frame = enriched.select(
        ["id", "feature_alpha", "feature_beta", "feature_gamma"]
    ).to_pandas()
    predictions = result.predictor.predict(prediction_frame)

    assert result.model_count == 6
    assert predictions.shape[0] == enriched.height
    assert np.isfinite(predictions).all()


def test_weighted_multi_target_predictor_applies_target_weights() -> None:
    class StubModel:
        def __init__(self, outputs: list[float]) -> None:
            self._outputs = np.asarray(outputs, dtype=np.float64)

        def predict(self, frame):
            return self._outputs

    feature_frame = pl.DataFrame({"feature_alpha": [0.0, 1.0, 2.0, 3.0]}).to_pandas()
    predictor = WeightedMultiTargetPredictor(
        feature_names=("feature_alpha",),
        target_models={
            "target_ender_20": (StubModel([0.1, 0.2, 0.8, 0.9]),),
            "target_xerxes_20": (StubModel([0.9, 0.7, 0.3, 0.1]),),
        },
        target_weights={
            "target_ender_20": 0.8,
            "target_xerxes_20": 0.2,
        },
    )

    predictions = predictor.predict(feature_frame)
    expected = ModelOrchestrator.weighted_rank_average_predictions(
        [
            np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
            np.array([0.9, 0.7, 0.3, 0.1], dtype=np.float64),
        ],
        [0.8, 0.2],
    )

    assert np.allclose(predictions, expected)
