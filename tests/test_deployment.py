import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.deployment import AdversarialStressTester, DeploymentHarness
from src.evaluation import EvaluationEngine
from src.features import PurgedEraSplitter
from src.models import ModelOrchestrator
from src.protocols import FeaturePipeline, IdentityTransformer


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
def trained_models(toy_training_frame: pl.DataFrame):
    feature_names = ["feature_alpha", "feature_beta", "feature_gamma"]
    orchestrator = ModelOrchestrator(
        feature_names=feature_names,
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
    cv_result = orchestrator.train_cross_validation(
        toy_training_frame,
        PurgedEraSplitter(n_splits=3, purge_buffer=1),
    )
    return orchestrator, feature_names, cv_result.predictor, cv_result.models


@pytest.fixture(scope="module")
def evaluation_engine() -> EvaluationEngine:
    return EvaluationEngine(
        era_col="era",
        prediction_col="prediction",
        target_col="target",
        id_col="id",
    )


def test_noise_injection_perturbs_predictions_without_crashing(
    toy_training_frame: pl.DataFrame,
    trained_models,
    evaluation_engine: EvaluationEngine,
) -> None:
    _, feature_names, predictor, _ = trained_models
    stress_tester = AdversarialStressTester(
        evaluation_engine,
        degradation_threshold=0.90,
        random_state=7,
    )

    result = stress_tester.evaluate_noise_resilience(
        toy_training_frame,
        predictor,
        feature_names,
        noise_std_ratio=0.5,
    )

    assert result.clean_predictions.shape == result.stressed_predictions.shape
    assert not np.allclose(result.clean_predictions, result.stressed_predictions)
    assert result.degradation_pct >= 0.0


def test_predict_live_reorders_scrambled_feature_columns(
    toy_training_frame: pl.DataFrame,
    trained_models,
    tmp_path: Path,
) -> None:
    _, feature_names, predictor, _ = trained_models
    harness = DeploymentHarness()
    payload = harness.build_numerai_payload(
        feature_pipeline=FeaturePipeline([IdentityTransformer()]),
        predictor_ensemble=predictor,
        feature_names=feature_names,
    )
    payload_bundle = harness.serialize_payload(payload, tmp_path / "bundle.pkl")

    scrambled = toy_training_frame.select(
        ["id", "feature_gamma", "feature_alpha", "feature_beta"]
    )
    predictions = harness.predict_live(scrambled, payload_bundle.payload_path)

    assert predictions.columns == ["id", "prediction"]
    assert predictions.height == scrambled.height
    assert predictions.get_column("prediction").is_null().sum() == 0


def test_predict_live_raises_on_missing_required_feature(
    toy_training_frame: pl.DataFrame,
    trained_models,
    tmp_path: Path,
) -> None:
    _, feature_names, predictor, _ = trained_models
    harness = DeploymentHarness()
    payload = harness.build_numerai_payload(
        feature_pipeline=FeaturePipeline([IdentityTransformer()]),
        predictor_ensemble=predictor,
        feature_names=feature_names,
    )
    payload_bundle = harness.serialize_payload(payload, tmp_path / "bundle_missing.pkl")
    broken = toy_training_frame.select(["id", "feature_alpha", "feature_beta"])

    with pytest.raises(KeyError, match="missing required features"):
        harness.predict_live(broken, payload_bundle.payload_path)


def test_cloudpickle_payload_reload_matches_pre_serialization_predictions(
    toy_training_frame: pl.DataFrame,
    trained_models,
    tmp_path: Path,
) -> None:
    _, feature_names, predictor, _ = trained_models
    harness = DeploymentHarness()
    payload = harness.build_numerai_payload(
        feature_pipeline=FeaturePipeline([IdentityTransformer()]),
        predictor_ensemble=predictor,
        feature_names=feature_names,
    )
    payload_bundle = harness.serialize_payload(payload, tmp_path / "bundle_reload.pkl")

    scrambled = toy_training_frame.select(
        ["id", "feature_gamma", "feature_alpha", "feature_beta"]
    )
    pre_serialization = payload(scrambled.to_pandas().set_index("id"))[
        "prediction"
    ].to_numpy(dtype=np.float64)
    post_serialization = harness.predict_live(scrambled, payload_bundle.payload_path)

    assert np.allclose(
        post_serialization.get_column("prediction").to_numpy(),
        pre_serialization,
    )


def test_serializer_writes_cloudpickle_payload(
    toy_training_frame: pl.DataFrame,
    trained_models,
    tmp_path: Path,
) -> None:
    _, feature_names, predictor, _ = trained_models
    harness = DeploymentHarness()
    payload = harness.build_numerai_payload(
        feature_pipeline=FeaturePipeline([IdentityTransformer()]),
        predictor_ensemble=predictor,
        feature_names=feature_names,
    )
    payload_bundle = harness.serialize_payload(payload, tmp_path / "bundle_files.pkl")

    assert payload_bundle.payload_path.exists()
    reloaded_payload = harness.load_payload(payload_bundle.payload_path)
    live_features = (
        toy_training_frame.select(
            ["id", "feature_alpha", "feature_beta", "feature_gamma"]
        )
        .to_pandas()
        .set_index("id")
    )
    result = reloaded_payload(live_features)
    assert isinstance(result, pd.DataFrame)
    assert "prediction" in result.columns
