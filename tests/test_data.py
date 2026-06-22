"""Tests for nmr.data.IngestionAgent — hermetic, CI-safe, no git-ignored data.

All tests use a synthetic dataset constructed in ``tmp_path``. The one test
that touches ``data/v5.2/`` is guarded by ``@pytest.mark.skipif`` and will be
skipped in CI where the file is absent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import polars as pl
import pytest

from nmr.config import DataConfig
from nmr.data import IngestionAgent

_FEATURES_JSON: dict = {
    "feature_sets": {
        "small": ["feature_a", "feature_b"],
        "medium": ["feature_a", "feature_b", "feature_c"],
        "all": ["feature_a", "feature_b", "feature_c"],
        "group_x": ["feature_b", "feature_c"],
    },
    "targets": ["target", "target_aux"],
}

_VERSION = "v0test"
_TRAIN_ROWS = 4
_LIVE_ROWS = 2


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("module_data")
    version_dir = root / _VERSION
    version_dir.mkdir()

    (version_dir / "features.json").write_text(
        json.dumps(_FEATURES_JSON), encoding="utf-8"
    )

    train_df = pl.DataFrame(
        {
            "era": ["era1", "era1", "era2", "era2"],
            "id": ["id1", "id2", "id3", "id4"],
            "feature_a": [0.1, 0.2, 0.3, 0.4],
            "feature_b": [0.5, 0.6, 0.7, 0.8],
            "feature_c": [0.9, 1.0, 1.1, 1.2],
            "target": [0.0, 0.25, 0.5, 0.75],
            "target_aux": [0.1, 0.2, 0.3, 0.4],
        }
    )
    train_df.write_parquet(version_dir / "train.parquet")
    train_df.write_parquet(version_dir / "validation.parquet")

    live_df = pl.DataFrame(
        {
            "era": ["era3", "era3"],
            "id": ["id5", "id6"],
            "feature_a": [0.5, 0.6],
            "feature_b": [0.7, 0.8],
            "feature_c": [0.9, 1.0],
            "target": [None, None],
        },
        schema={
            "era": pl.String,
            "id": pl.String,
            "feature_a": pl.Float64,
            "feature_b": pl.Float64,
            "feature_c": pl.Float64,
            "target": pl.Float64,
        },
    )
    live_df.write_parquet(version_dir / "live.parquet")

    return root


@pytest.fixture(scope="module")
def cfg(dataset_root: Path) -> DataConfig:
    return DataConfig(
        version=_VERSION,
        feature_set="small",
        targets=("target",),
        data_dir=dataset_root,
    )


@pytest.fixture(scope="module")
def agent(cfg: DataConfig) -> IngestionAgent:
    return IngestionAgent(cfg)


class TestConstruction:
    def test_inert_with_completely_absent_data_dir(self) -> None:
        cfg = DataConfig(
            version="ghost",
            feature_set="small",
            data_dir=Path("/nonexistent/absolutely_missing_path_xyz"),
        )
        ag = IngestionAgent(cfg)
        assert ag._metadata is None
        assert ag._schema_cache == {}

    def test_inert_even_when_split_files_absent(self, tmp_path: Path) -> None:
        (tmp_path / "empty_v").mkdir()
        cfg = DataConfig(version="empty_v", feature_set="small", data_dir=tmp_path)
        IngestionAgent(cfg)


class TestMetadata:
    def test_features_small_membership_and_order(self, agent: IngestionAgent) -> None:
        assert agent.features("small") == ["feature_a", "feature_b"]

    def test_features_medium_membership_and_order(self, agent: IngestionAgent) -> None:
        assert agent.features("medium") == ["feature_a", "feature_b", "feature_c"]

    def test_features_named_group(self, agent: IngestionAgent) -> None:
        assert agent.features("group_x") == ["feature_b", "feature_c"]

    def test_features_default_uses_config_feature_set(
        self, agent: IngestionAgent
    ) -> None:
        assert agent.features() == agent.features("small")

    def test_features_invalid_subset_raises_with_message(
        self, agent: IngestionAgent
    ) -> None:
        with pytest.raises(ValueError, match="Feature subset"):
            agent.features("nonexistent_subset_xyz_abc")

    def test_feature_sets_property_returns_dict(self, agent: IngestionAgent) -> None:
        sets = agent.feature_sets
        assert isinstance(sets, dict)
        assert {"small", "medium", "all"}.issubset(sets)

    def test_feature_sets_property_returns_copy(self, agent: IngestionAgent) -> None:
        sets = agent.feature_sets
        sets["small"].append("corrupt_me")
        assert agent.features("small") == ["feature_a", "feature_b"]

    def test_available_targets_contains_declared_targets(
        self, agent: IngestionAgent
    ) -> None:
        targets = agent.available_targets()
        assert isinstance(targets, list)
        assert "target" in targets
        assert "target_aux" in targets

    def test_feature_metadata_is_dict(self, agent: IngestionAgent) -> None:
        assert isinstance(agent.feature_metadata, dict)

    def test_feature_metadata_returns_copy(self, agent: IngestionAgent) -> None:
        metadata = agent.feature_metadata
        metadata["feature_sets"]["small"].append("corrupt_me")
        assert agent.features("small") == ["feature_a", "feature_b"]

    def test_features_json_absent_raises_on_first_access(self, tmp_path: Path) -> None:
        (tmp_path / "no_json").mkdir()
        cfg = DataConfig(version="no_json", feature_set="small", data_dir=tmp_path)
        ag = IngestionAgent(cfg)
        with pytest.raises(FileNotFoundError):
            _ = ag.feature_metadata


class TestSchema:
    def test_schema_returns_mapping(self, agent: IngestionAgent) -> None:
        assert isinstance(agent.schema("train"), Mapping)

    def test_schema_contains_expected_columns(self, agent: IngestionAgent) -> None:
        s = agent.schema("train")
        for col in ("era", "id", "feature_a", "target"):
            assert col in s, f"Expected column {col!r} missing from schema"

    def test_schema_memoized_single_parquet_call(
        self, agent: IngestionAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_count = 0
        original_scan = pl.scan_parquet

        # Module-scoped fixtures can pre-populate cache in earlier tests.
        agent._schema_cache.pop("train", None)

        def tracking_scan(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_scan(*args, **kwargs)

        monkeypatch.setattr(pl, "scan_parquet", tracking_scan)

        s1 = agent.schema("train")
        s2 = agent.schema("train")

        assert call_count == 1
        assert s1 is s2

    def test_schema_different_splits_cached_independently(
        self, agent: IngestionAgent
    ) -> None:
        s_train = agent.schema("train")
        s_val = agent.schema("validation")
        assert "era" in s_train
        assert "era" in s_val
        assert "train" in agent._schema_cache
        assert "validation" in agent._schema_cache

    def test_missing_split_file_raises_on_schema_access(self, tmp_path: Path) -> None:
        (tmp_path / "v_empty").mkdir()
        cfg = DataConfig(version="v_empty", feature_set="small", data_dir=tmp_path)
        ag = IngestionAgent(cfg)
        with pytest.raises(FileNotFoundError):
            ag.schema("train")

    def test_unknown_split_raises_valueerror_from_schema(
        self, agent: IngestionAgent
    ) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            agent.schema("test_bogus_split")


class TestScan:
    def test_returns_lazyframe(self, agent: IngestionAgent) -> None:
        assert isinstance(agent.scan("train"), pl.LazyFrame)

    def test_column_selection_exact_era_id_features_target(
        self, agent: IngestionAgent
    ) -> None:
        names = agent.scan("train").collect_schema().names()
        assert names == ["era", "id", "feature_a", "feature_b", "target"]

    def test_column_order_deterministic_across_two_scans(
        self, agent: IngestionAgent
    ) -> None:
        cols1 = agent.scan("train").collect_schema().names()
        cols2 = agent.scan("train").collect_schema().names()
        assert cols1 == cols2

    def test_column_pushdown_no_extra_columns_in_collected_data(
        self, agent: IngestionAgent
    ) -> None:
        assert agent.load("train").columns == [
            "era",
            "id",
            "feature_a",
            "feature_b",
            "target",
        ]

    def test_subset_override_medium(self, agent: IngestionAgent) -> None:
        names = agent.scan("train", subset="medium").collect_schema().names()
        assert names == ["era", "id", "feature_a", "feature_b", "feature_c", "target"]

    def test_targets_override(self, agent: IngestionAgent) -> None:
        names = agent.scan("train", targets=["target_aux"]).collect_schema().names()
        assert names == ["era", "id", "feature_a", "feature_b", "target_aux"]

    def test_targets_override_multiple(self, agent: IngestionAgent) -> None:
        names = (
            agent.scan("train", targets=["target", "target_aux"])
            .collect_schema()
            .names()
        )
        assert names == ["era", "id", "feature_a", "feature_b", "target", "target_aux"]

    def test_unknown_target_raises_valueerror(self, agent: IngestionAgent) -> None:
        with pytest.raises(ValueError, match=r"Unknown target\(s\) requested"):
            agent.scan("train", targets=["taget"])

    def test_columns_override_bypasses_subset_and_targets(
        self, agent: IngestionAgent
    ) -> None:
        names = (
            agent.scan("train", columns=["era", "feature_b"]).collect_schema().names()
        )
        assert names == ["era", "feature_b"]

    def test_live_with_absent_targets_does_not_raise(
        self, agent: IngestionAgent
    ) -> None:
        names = agent.live(targets=["target", "target_aux"]).collect_schema().names()
        assert "target" in names
        assert "target_aux" not in names
        assert "era" in names
        assert "feature_a" in names

    def test_live_default_scan_includes_target(self, agent: IngestionAgent) -> None:
        names = agent.live().collect_schema().names()
        assert "target" in names
        assert "era" in names

    def test_live_target_values_are_all_null(self, agent: IngestionAgent) -> None:
        df = agent.load("live")
        assert df["target"].null_count() == _LIVE_ROWS

    def test_missing_split_file_raises_on_scan(self, tmp_path: Path) -> None:
        (tmp_path / "v_no_parquet").mkdir()
        (tmp_path / "v_no_parquet" / "features.json").write_text(
            json.dumps(_FEATURES_JSON), encoding="utf-8"
        )
        cfg = DataConfig(version="v_no_parquet", feature_set="small", data_dir=tmp_path)
        ag = IngestionAgent(cfg)
        with pytest.raises(FileNotFoundError):
            ag.scan("train")

    def test_unknown_split_raises_valueerror_from_scan(
        self, agent: IngestionAgent
    ) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            agent.scan("bogus_split_name")

    def test_invalid_subset_raises_valueerror(self, agent: IngestionAgent) -> None:
        with pytest.raises(ValueError, match="Feature subset"):
            agent.scan("train", subset="not_a_real_subset")


class TestLoad:
    def test_returns_dataframe(self, agent: IngestionAgent) -> None:
        assert isinstance(agent.load("train"), pl.DataFrame)

    def test_row_count_correct(self, agent: IngestionAgent) -> None:
        assert len(agent.load("train")) == _TRAIN_ROWS

    def test_columns_match_scan_schema(self, agent: IngestionAgent) -> None:
        df = agent.load("train")
        assert df.columns == agent.scan("train").collect_schema().names()

    def test_load_live_row_count(self, agent: IngestionAgent) -> None:
        assert len(agent.load("live")) == _LIVE_ROWS


class TestConvenienceDelegates:
    def test_train_returns_lazyframe(self, agent: IngestionAgent) -> None:
        assert isinstance(agent.train(), pl.LazyFrame)

    def test_validation_returns_lazyframe(self, agent: IngestionAgent) -> None:
        assert isinstance(agent.validation(), pl.LazyFrame)

    def test_live_returns_lazyframe(self, agent: IngestionAgent) -> None:
        assert isinstance(agent.live(), pl.LazyFrame)

    def test_train_columns_match_scan_train(self, agent: IngestionAgent) -> None:
        assert (
            agent.train().collect_schema().names()
            == agent.scan("train").collect_schema().names()
        )

    def test_validation_subset_kwarg_forwarded(self, agent: IngestionAgent) -> None:
        names = agent.validation(subset="medium").collect_schema().names()
        assert "feature_c" in names


_REAL_TRAIN = Path("data/v5.2/train.parquet")
_REAL_FEATURES_JSON = Path("data/v5.2/features.json")


@pytest.mark.skipif(
    not (_REAL_TRAIN.exists() and _REAL_FEATURES_JSON.exists()),
    reason="v5.2 dataset not on disk (git-ignored); skipped in CI",
)
def test_real_v52_smoke() -> None:
    cfg = DataConfig(version="v5.2", feature_set="small", targets=("target",))
    ag = IngestionAgent(cfg)

    assert ag._metadata is None
    assert ag._schema_cache == {}

    lf = ag.scan("train")
    assert isinstance(lf, pl.LazyFrame)

    names = lf.collect_schema().names()
    assert "era" in names
    assert "id" in names
    assert "target" in names
    assert "target" in ag.live().collect_schema().names()
    for feat in ag.features("small"):
        assert feat in names, f"feature {feat!r} missing from real train schema"
