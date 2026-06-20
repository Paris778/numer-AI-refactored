import json
from pathlib import Path

import polars as pl
import pytest

from src.data import IngestionAgent


@pytest.fixture(scope="module")
def data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "v5.2"


@pytest.fixture(scope="module")
def agent(data_root: Path) -> IngestionAgent:
    return IngestionAgent(data_root)


def test_dataset_names_include_expected_core_files(agent: IngestionAgent) -> None:
    expected = {"train", "validation", "live", "meta_model"}
    assert expected.issubset(set(agent.dataset_names()))


def test_dataset_path_resolves_existing_file(agent: IngestionAgent) -> None:
    dataset_path = agent.dataset_path("train")

    assert dataset_path.exists()
    assert dataset_path.name == "train.parquet"


def test_unknown_dataset_raises_clear_error(agent: IngestionAgent) -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        agent.dataset_path("unknown")


def test_feature_subset_names_are_loaded_from_metadata(agent: IngestionAgent) -> None:
    subset_names = set(agent.feature_subset_names())

    assert {"small", "medium", "all"}.issubset(subset_names)


def test_scan_dataset_returns_lazyframe_for_small_subset(agent: IngestionAgent) -> None:
    lazy_frame = agent.scan_dataset(
        "train", feature_subset="small", include_targets=True
    )
    schema = lazy_frame.collect_schema()

    assert isinstance(lazy_frame, pl.LazyFrame)
    assert "era" in schema
    assert any(name.startswith("feature_") for name in schema.names())
    assert any(name.startswith("target") for name in schema.names())


def test_unknown_feature_subset_raises_clear_error(agent: IngestionAgent) -> None:
    with pytest.raises(KeyError, match="Unknown feature subset"):
        agent.get_feature_names("mega")


def test_summary_reports_schema_and_row_count(agent: IngestionAgent) -> None:
    summary = agent.summarize_dataset(
        "validation", feature_subset="small", include_targets=True
    )

    assert summary.name == "validation"
    assert summary.row_count > 0
    assert "era" in summary.schema
    assert any(name.startswith("feature_") for name in summary.columns)


def test_available_datasets_reflect_local_files(agent: IngestionAgent) -> None:
    availability = agent.available_datasets()

    assert availability["train"] is True
    assert availability["live_example_preds"] is True


def test_schema_cache_is_lazy_and_instantiation_does_not_touch_missing_datasets(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data" / "v5.2"
    data_root.mkdir(parents=True)
    (data_root / "features.json").write_text(
        json.dumps({"feature_sets": {"small": ["feature_alpha"]}}),
        encoding="utf-8",
    )

    agent = IngestionAgent(
        data_root,
        dataset_files={"train": "missing.parquet"},
    )

    assert agent._schema_cache == {}
    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        agent.get_schema("train")


def test_schema_cache_memoizes_repeated_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    scan_calls = 0
    original_scan_parquet = pl.scan_parquet

    def counting_scan_parquet(*args: object, **kwargs: object) -> pl.LazyFrame:
        nonlocal scan_calls
        scan_calls += 1
        return original_scan_parquet(*args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", counting_scan_parquet)

    agent = IngestionAgent(Path(__file__).resolve().parents[1] / "data" / "v5.2")

    first_schema = agent.get_schema("train")
    second_schema = agent.get_schema("train")

    assert first_schema == second_schema
    assert scan_calls == 1
