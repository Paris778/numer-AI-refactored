import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.deployment import AdversarialStressTester, DeploymentHarness
from src.evaluation import EvaluationEngine
from src.features import PurgedEraSplitter
from src.models import ModelOrchestrator


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
        target_column="target",
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
    return orchestrator, feature_names, cv_result.models


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
    _, feature_names, models = trained_models
    stress_tester = AdversarialStressTester(
        evaluation_engine,
        degradation_threshold=0.90,
        random_state=7,
    )

    result = stress_tester.evaluate_noise_resilience(
        toy_training_frame,
        models,
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
    _, feature_names, models = trained_models
    harness = DeploymentHarness()
    artifact_dir = harness.serialize_candidate(
        tmp_path / "bundle",
        models,
        feature_names,
        {"target_column": "target"},
    )

    scrambled = toy_training_frame.select(
        ["id", "feature_gamma", "feature_alpha", "feature_beta"]
    )
    predictions = harness.predict_live(scrambled, artifact_dir)

    assert predictions.columns == ["id", "prediction"]
    assert predictions.height == scrambled.height
    assert predictions.get_column("prediction").is_null().sum() == 0


def test_predict_live_raises_on_missing_required_feature(
    toy_training_frame: pl.DataFrame,
    trained_models,
    tmp_path: Path,
) -> None:
    _, feature_names, models = trained_models
    harness = DeploymentHarness()
    artifact_dir = harness.serialize_candidate(
        tmp_path / "bundle_missing",
        models,
        feature_names,
        {"target_column": "target"},
    )
    broken = toy_training_frame.select(["id", "feature_alpha", "feature_beta"])

    with pytest.raises(KeyError, match="missing required features"):
        harness.predict_live(broken, artifact_dir)


def test_serializer_writes_manifest_and_native_model_files(
    trained_models,
    tmp_path: Path,
) -> None:
    _, feature_names, models = trained_models
    harness = DeploymentHarness()
    artifact_dir = harness.serialize_candidate(
        tmp_path / "bundle_files",
        models,
        feature_names,
        {"target_column": "target", "mode": "cv"},
    )

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["feature_names"] == feature_names
    assert manifest["model_library"] == "lightgbm"
    assert len(manifest["model_files"]) == len(models)
    for model_file in manifest["model_files"]:
        assert (artifact_dir / model_file).exists()
