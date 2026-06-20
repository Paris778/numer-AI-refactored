import numpy as np
import pandas as pd
import polars as pl
import pytest
from numerai_tools.scoring import feature_neutral_corr as official_feature_neutral_corr
from numerai_tools.scoring import (
    max_feature_correlation as official_max_feature_correlation,
)
from numerai_tools.scoring import numerai_corr as official_numerai_corr

from src.evaluation import EvaluationEngine
from src.risk import NeutralizationEngine


@pytest.fixture(scope="module")
def parity_fixture() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], list[str]]
):
    rng = np.random.default_rng(20260620)
    n_rows = 128
    n_features = 7
    row_ids = [f"id_{index:04d}" for index in range(n_rows)]
    feature_names = tuple(f"feature_{index:02d}" for index in range(n_features))

    features = rng.normal(loc=0.0, scale=1.0, size=(n_rows, n_features))
    features[:, 1] = 0.6 * features[:, 0] + 0.4 * features[:, 1]
    raw_signal = (
        0.7 * features[:, 0]
        - 0.35 * features[:, 2]
        + 0.15 * np.sin(features[:, 3])
        + rng.normal(0.0, 0.2, size=n_rows)
    )
    predictions = raw_signal.copy()
    predictions[::11] = predictions[0]
    predictions[5::17] = predictions[5]

    targets = rng.uniform(0.0, 1.0, size=n_rows)
    targets[::13] = targets[0]

    return predictions, targets, features, feature_names, row_ids


@pytest.fixture(scope="module")
def evaluation_engine() -> EvaluationEngine:
    return EvaluationEngine(
        era_col="era",
        prediction_col="prediction",
        target_col="target",
        id_col="id",
    )


@pytest.fixture(scope="module")
def adversarial_frame() -> pl.DataFrame:
    rng = np.random.default_rng(20260621)
    feature_names = [f"feature_{index:02d}" for index in range(5)]
    eras = ["0001"] * 24 + ["0002"] * 24 + ["0003"] * 24 + ["0004"] * 24
    n_rows = len(eras)
    row_ids = [f"adv_{index:04d}" for index in range(n_rows)]
    feature_matrix = rng.normal(loc=0.0, scale=1.0, size=(n_rows, len(feature_names)))
    feature_matrix[:, 2] = 0.7 * feature_matrix[:, 0] + 0.3 * feature_matrix[:, 2]

    predictions = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    predictions[:29] = 0.125
    predictions[48:72] = 0.5
    predictions[72] = -1e9
    predictions[73] = 1e9

    targets = rng.uniform(0.0, 1.0, size=n_rows)
    targets[48:72] = 0.75

    return pl.DataFrame(
        {
            "id": row_ids,
            "era": eras,
            "prediction": predictions,
            "target": targets,
            **{name: feature_matrix[:, idx] for idx, name in enumerate(feature_names)},
        }
    )


def test_numerai_corr_matches_official_package(
    parity_fixture,
    evaluation_engine: EvaluationEngine,
) -> None:
    predictions, targets, _, _, row_ids = parity_fixture
    prediction_frame = pd.DataFrame({"prediction": predictions}, index=row_ids)
    target_series = pd.Series(targets, index=row_ids, name="target")

    custom = evaluation_engine.numerai_corr(predictions, targets)
    official = float(official_numerai_corr(prediction_frame, target_series).iloc[0])

    assert np.allclose(custom, official, rtol=1e-8, atol=1e-8)


def test_feature_neutral_corr_matches_official_package(
    parity_fixture,
    evaluation_engine: EvaluationEngine,
    tmp_path,
) -> None:
    predictions, targets, features, feature_names, row_ids = parity_fixture
    frame = pl.DataFrame(
        {
            "id": row_ids,
            "era": ["0001"] * len(row_ids),
            "prediction": predictions,
            "target": targets,
            **{name: features[:, idx] for idx, name in enumerate(feature_names)},
        }
    )

    custom = evaluation_engine.evaluate_eras(
        frame,
        feature_columns=feature_names,
        neutralization_engine=NeutralizationEngine(cache_root=tmp_path),
    )[0].fnc
    assert custom is not None

    prediction_frame = pd.DataFrame({"prediction": predictions}, index=row_ids)
    feature_frame = pd.DataFrame(features, index=row_ids, columns=list(feature_names))
    target_series = pd.Series(targets, index=row_ids, name="target")
    official = float(
        official_feature_neutral_corr(
            prediction_frame,
            feature_frame,
            target_series,
        ).iloc[0]
    )

    assert np.allclose(custom, official, rtol=1e-8, atol=1e-8)


def test_max_feature_exposure_matches_official_package(
    parity_fixture,
    evaluation_engine: EvaluationEngine,
) -> None:
    predictions, _, features, feature_names, row_ids = parity_fixture
    prediction_series = pd.Series(predictions, index=row_ids, name="prediction")
    feature_frame = pd.DataFrame(features, index=row_ids, columns=list(feature_names))

    custom = evaluation_engine.compute_max_feature_exposure(predictions, features)
    _, official = official_max_feature_correlation(prediction_series, feature_frame)

    assert np.allclose(custom, float(official), rtol=1e-8, atol=1e-8)


def test_custom_and_official_backends_match_on_adversarial_frame(
    adversarial_frame: pl.DataFrame,
    tmp_path,
) -> None:
    feature_names = tuple(
        column for column in adversarial_frame.columns if column.startswith("feature_")
    )
    custom_engine = EvaluationEngine(
        era_col="era",
        prediction_col="prediction",
        target_col="target",
        id_col="id",
        backend="custom",
    )
    official_engine = EvaluationEngine(
        era_col="era",
        prediction_col="prediction",
        target_col="target",
        id_col="id",
        backend="official",
    )

    custom_summary = custom_engine.summarize(
        custom_engine.evaluate_eras(
            adversarial_frame,
            feature_columns=feature_names,
            neutralization_engine=NeutralizationEngine(cache_root=tmp_path),
        )
    )
    official_summary = official_engine.summarize(
        official_engine.evaluate_eras(
            adversarial_frame,
            feature_columns=feature_names,
        )
    )

    assert np.allclose(
        custom_summary.mean_corr,
        official_summary.mean_corr,
        rtol=1e-8,
        atol=1e-8,
    )
    assert np.allclose(
        custom_summary.mean_fnc,
        official_summary.mean_fnc,
        rtol=1e-8,
        atol=1e-8,
    )
    assert np.allclose(
        custom_summary.max_feature_exposure,
        official_summary.max_feature_exposure,
        rtol=1e-8,
        atol=1e-8,
    )
