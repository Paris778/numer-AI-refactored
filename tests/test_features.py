from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.data import IngestionAgent
from src.features import FeatureFactory, PurgedEraSplitter


@pytest.fixture(scope="module")
def toy_frame() -> pl.DataFrame:
    eras = [f"{era:04d}" for era in range(1, 11) for _ in range(2)]
    ids = [f"row_{index:03d}" for index in range(len(eras))]
    signal = np.linspace(0.0, 1.0, len(eras), dtype=np.float64)
    return pl.DataFrame({"id": ids, "era": eras, "signal": signal})


@pytest.fixture(scope="module")
def data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "v5.2"


@pytest.fixture(scope="module")
def agent(data_root: Path) -> IngestionAgent:
    return IngestionAgent(data_root)


def test_purged_splitter_enforces_zero_leakage(toy_frame: pl.DataFrame) -> None:
    splitter = PurgedEraSplitter(n_splits=5, purge_buffer=1)

    splits = splitter.split(toy_frame)

    assert len(splits) == 5
    for train_eras, validation_eras in splits:
        train_ordinals = np.asarray(train_eras, dtype=np.int64)
        validation_ordinals = np.asarray(validation_eras, dtype=np.int64)
        distances = np.abs(train_ordinals[:, None] - validation_ordinals[None, :])
        assert np.all(distances > 1)
        assert set(train_eras).isdisjoint(validation_eras)


def test_boundary_eras_are_purged_correctly(toy_frame: pl.DataFrame) -> None:
    splitter = PurgedEraSplitter(n_splits=5, purge_buffer=1)
    folds = list(splitter.iter_folds(toy_frame))

    first_fold = folds[0]
    last_fold = folds[-1]

    assert list(first_fold.validation_eras) == ["0001", "0002"]
    assert "0003" not in set(first_fold.train_eras)
    assert list(last_fold.validation_eras) == ["0009", "0010"]
    assert "0008" not in set(last_fold.train_eras)


def test_splitter_accepts_lazyframe_and_maps_row_indices(
    toy_frame: pl.DataFrame,
) -> None:
    splitter = PurgedEraSplitter(n_splits=5, purge_buffer=1)

    index_splits = splitter.split_row_indices(toy_frame.lazy())

    assert len(index_splits) == 5
    train_indices, validation_indices = index_splits[2]
    assert train_indices.ndim == 1
    assert validation_indices.ndim == 1
    assert validation_indices.size == 4


def test_feature_factory_adds_rank_and_noise_columns(toy_frame: pl.DataFrame) -> None:
    transformed = (
        FeatureFactory(toy_frame.lazy())
        .add_era_rank("signal")
        .add_noise_baseline(seed=7)
        .collect()
    )

    assert "signal_era_rank" in transformed.columns
    assert "noise_baseline" in transformed.columns
    assert transformed.get_column("noise_baseline").n_unique() > 1


def test_real_train_split_has_zero_leakage(agent: IngestionAgent) -> None:
    splitter = PurgedEraSplitter(n_splits=5, purge_buffer=4)
    train_lazy = agent.scan_dataset("train", include_metadata=True)

    for train_eras, validation_eras in splitter.split(train_lazy):
        train_ordinals = np.asarray(train_eras, dtype=np.int64)
        validation_ordinals = np.asarray(validation_eras, dtype=np.int64)
        distances = np.abs(train_ordinals[:, None] - validation_ordinals[None, :])
        assert np.all(distances > 4)


def test_non_numeric_eras_raise_safe_error() -> None:
    splitter = PurgedEraSplitter(n_splits=2, purge_buffer=1)
    frame = pl.DataFrame(
        {"id": ["a", "b", "c", "d"], "era": ["alpha", "alpha", "beta", "beta"]}
    )

    with pytest.raises(ValueError, match="Non-numeric era format detected"):
        splitter.split(frame)
