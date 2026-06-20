from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from src.data import IngestionAgent
from src.risk import NeutralizationEngine


@pytest.fixture(scope="module")
def data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "v5.2"


@pytest.fixture(scope="module")
def agent(data_root: Path) -> IngestionAgent:
    return IngestionAgent(data_root)


@pytest.fixture()
def engine(tmp_path: Path) -> NeutralizationEngine:
    return NeutralizationEngine(cache_root=tmp_path)


@pytest.fixture(scope="module")
def sample_era(agent: IngestionAgent) -> str:
    engine = NeutralizationEngine(cache_root=Path("artifacts") / "cache")
    return engine.list_eras(agent, "validation")[0]


def test_neutralize_tensor_matches_explicit_formula() -> None:
    engine = NeutralizationEngine(cache_root=Path("artifacts") / "cache")
    feature_matrix = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    predictions = np.array([0.2, 0.4, 0.7, 0.9], dtype=np.float64)
    design_matrix = np.hstack(
        [feature_matrix, np.ones((feature_matrix.shape[0], 1), dtype=np.float64)]
    )

    expected = predictions - design_matrix @ (
        np.linalg.pinv(design_matrix) @ predictions
    )
    actual = engine.neutralize_tensor(predictions, feature_matrix)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_neutralization_removes_constant_intercept_component() -> None:
    engine = NeutralizationEngine(cache_root=Path("artifacts") / "cache")
    feature_matrix = np.array(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.5]],
        dtype=np.float64,
    )
    predictions = np.full(feature_matrix.shape[0], 3.5, dtype=np.float64)

    actual = engine.neutralize_tensor(predictions, feature_matrix)

    np.testing.assert_allclose(actual, np.zeros_like(predictions), atol=1e-10)


def test_cached_matrix_matches_on_the_fly_result(
    agent: IngestionAgent,
    engine: NeutralizationEngine,
    sample_era: str,
) -> None:
    subset_name = "small"
    feature_names = agent.get_feature_names(subset_name)
    era_frame = engine.collect_era_frame(agent, "validation", subset_name, sample_era)
    feature_matrix = era_frame.select(list(feature_names)).to_numpy()
    predictions = np.linspace(0.0, 1.0, feature_matrix.shape[0], dtype=np.float64)

    direct = engine.neutralize_tensor(predictions, feature_matrix)

    engine.cache_subsets(
        agent, "validation", subset_name, eras=[sample_era], overwrite=True
    )
    cached = engine.load_cached_pseudo_inverse(subset_name, sample_era)
    from_cache = engine.neutralize_tensor(
        predictions,
        feature_matrix,
        pseudo_inverse=cached.pseudo_inverse,
    )

    np.testing.assert_allclose(from_cache, direct, rtol=1e-10, atol=1e-10)


def test_cached_full_era_neutralization_completes_in_milliseconds(
    agent: IngestionAgent,
    engine: NeutralizationEngine,
    sample_era: str,
) -> None:
    subset_name = "small"
    feature_names = agent.get_feature_names(subset_name)
    era_frame = engine.collect_era_frame(agent, "validation", subset_name, sample_era)
    feature_matrix = era_frame.select(list(feature_names)).to_numpy()
    predictions = np.linspace(0.0, 1.0, feature_matrix.shape[0], dtype=np.float64)

    engine.cache_subsets(
        agent, "validation", subset_name, eras=[sample_era], overwrite=True
    )
    cached = engine.load_cached_pseudo_inverse(subset_name, sample_era)

    start = perf_counter()
    result = engine.neutralize_tensor(
        predictions,
        feature_matrix,
        pseudo_inverse=cached.pseudo_inverse,
    )
    cached_time_ms = (perf_counter() - start) * 1000.0

    assert result.shape == predictions.shape
    assert cached_time_ms < 1000.0


def test_cache_path_uses_subset_and_era(engine: NeutralizationEngine) -> None:
    cache_path = engine.cache_path("medium", "572")

    assert cache_path.parent.name == "medium"
    assert cache_path.name == "era_0572.npy"
