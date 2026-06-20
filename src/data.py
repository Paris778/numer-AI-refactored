from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

DEFAULT_DATASET_FILES: dict[str, str] = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "live": "live.parquet",
    "train_benchmark_models": "train_benchmark_models.parquet",
    "validation_benchmark_models": "validation_benchmark_models.parquet",
    "live_benchmark_models": "live_benchmark_models.parquet",
    "validation_example_preds": "validation_example_preds.parquet",
    "live_example_preds": "live_example_preds.parquet",
    "meta_model": "meta_model.parquet",
}

DEFAULT_METADATA_COLUMNS: tuple[str, ...] = ("id", "era", "data_type")


@dataclass(frozen=True)
class DatasetSummary:
    name: str
    path: Path
    columns: tuple[str, ...]
    schema: dict[str, str]
    row_count: int


class IngestionAgent:
    """Lazy parquet access for NumerAI v5.2 datasets."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        metadata_file: str | Path | None = None,
        cache_root: str | Path | None = None,
        dataset_files: dict[str, str] | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.metadata_file = (
            Path(metadata_file).expanduser().resolve()
            if metadata_file is not None
            else self.data_root / "features.json"
        )
        self.cache_root = (
            Path(cache_root).expanduser().resolve()
            if cache_root is not None
            else self.data_root.parent.parent / "artifacts" / "cache"
        )
        self.dataset_files = dict(dataset_files or DEFAULT_DATASET_FILES)
        self._feature_sets = self._load_feature_sets()
        self._schema_cache: dict[str, dict[str, str]] = {}

    @property
    def feature_sets(self) -> dict[str, tuple[str, ...]]:
        return self._feature_sets

    def dataset_names(self) -> tuple[str, ...]:
        return tuple(self.dataset_files)

    def dataset_path(self, dataset_name: str) -> Path:
        try:
            relative_path = self.dataset_files[dataset_name]
        except KeyError as exc:
            available = ", ".join(self.dataset_files)
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
            ) from exc

        dataset_path = (self.data_root / relative_path).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' is missing at {dataset_path}"
            )
        return dataset_path

    def feature_subset_names(self) -> tuple[str, ...]:
        return tuple(self._feature_sets)

    def get_feature_names(self, subset: str = "all") -> tuple[str, ...]:
        try:
            return self._feature_sets[subset]
        except KeyError as exc:
            available = ", ".join(self._feature_sets)
            raise KeyError(
                f"Unknown feature subset '{subset}'. Available subsets: {available}"
            ) from exc

    def get_target_names(self, dataset_name: str = "train") -> tuple[str, ...]:
        return self._target_names_from_schema(self.get_schema(dataset_name))

    def get_metadata_columns(self, dataset_name: str = "train") -> tuple[str, ...]:
        return self._metadata_columns_from_schema(self.get_schema(dataset_name))

    def available_datasets(self) -> dict[str, bool]:
        return {
            dataset_name: (self.data_root / relative_path).exists()
            for dataset_name, relative_path in self.dataset_files.items()
        }

    def scan_dataset(
        self,
        dataset_name: str,
        *,
        feature_subset: str | None = None,
        include_metadata: bool = True,
        include_targets: bool = False,
        extra_columns: Iterable[str] | None = None,
    ) -> pl.LazyFrame:
        dataset_path = self.dataset_path(dataset_name)
        lazy_frame = pl.scan_parquet(dataset_path)
        schema = self.get_schema(dataset_name)

        selected_columns: list[str] = []
        if include_metadata:
            selected_columns.extend(self._metadata_columns_from_schema(schema))
        if feature_subset is not None:
            selected_columns.extend(self.get_feature_names(feature_subset))
        if include_targets:
            selected_columns.extend(self._target_names_from_schema(schema))
        if extra_columns is not None:
            selected_columns.extend(extra_columns)

        if not selected_columns:
            return lazy_frame

        existing_columns = set(schema)
        unique_columns = self._dedupe_columns(selected_columns)
        missing_columns = [
            name for name in unique_columns if name not in existing_columns
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise KeyError(
                f"Requested columns are not available in '{dataset_name}': {missing}"
            )

        return lazy_frame.select(unique_columns)

    def get_schema(self, dataset_name: str) -> dict[str, str]:
        if dataset_name not in self.dataset_files:
            available = ", ".join(self.dataset_files)
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
            )

        if dataset_name not in self._schema_cache:
            dataset_path = self.dataset_path(dataset_name)
            schema = pl.scan_parquet(dataset_path).collect_schema()
            self._schema_cache[dataset_name] = {
                column: str(dtype) for column, dtype in schema.items()
            }

        return self._schema_cache[dataset_name]

    def count_rows(self, dataset_name: str) -> int:
        return (
            pl.scan_parquet(self.dataset_path(dataset_name))
            .select(pl.len().alias("row_count"))
            .collect()
            .item()
        )

    def summarize_dataset(
        self,
        dataset_name: str,
        *,
        feature_subset: str | None = None,
        include_metadata: bool = True,
        include_targets: bool = False,
        extra_columns: Iterable[str] | None = None,
    ) -> DatasetSummary:
        lazy_frame = self.scan_dataset(
            dataset_name,
            feature_subset=feature_subset,
            include_metadata=include_metadata,
            include_targets=include_targets,
            extra_columns=extra_columns,
        )
        schema = lazy_frame.collect_schema()
        return DatasetSummary(
            name=dataset_name,
            path=self.dataset_path(dataset_name),
            columns=tuple(schema.names()),
            schema={column: str(dtype) for column, dtype in schema.items()},
            row_count=self.count_rows(dataset_name),
        )

    def summarize_all(
        self, *, feature_subset: str = "small"
    ) -> dict[str, DatasetSummary]:
        return {
            dataset_name: self.summarize_dataset(
                dataset_name,
                feature_subset=(
                    feature_subset
                    if dataset_name in {"train", "validation", "live"}
                    else None
                ),
                include_targets=dataset_name in {"train", "validation"},
            )
            for dataset_name in self.dataset_names()
        }

    def _load_feature_sets(self) -> dict[str, tuple[str, ...]]:
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Feature metadata file is missing at {self.metadata_file}"
            )

        with self.metadata_file.open("r", encoding="utf-8") as file_handle:
            raw_metadata: dict[str, Any] = json.load(file_handle)

        raw_feature_sets = raw_metadata.get("feature_sets")
        if not isinstance(raw_feature_sets, dict) or not raw_feature_sets:
            raise ValueError(
                "Feature metadata must define a non-empty 'feature_sets' mapping"
            )

        feature_sets: dict[str, tuple[str, ...]] = {}
        for subset_name, feature_names in raw_feature_sets.items():
            if not isinstance(feature_names, list) or not all(
                isinstance(feature_name, str) for feature_name in feature_names
            ):
                raise ValueError(
                    f"Feature subset '{subset_name}' must be a list of strings"
                )
            feature_sets[subset_name] = tuple(feature_names)

        if "all" not in feature_sets:
            ordered_union: list[str] = []
            for subset_name in feature_sets:
                ordered_union.extend(feature_sets[subset_name])
            feature_sets["all"] = tuple(self._dedupe_columns(ordered_union))

        return feature_sets

    @staticmethod
    def _metadata_columns_from_schema(schema: dict[str, str]) -> tuple[str, ...]:
        return tuple(name for name in DEFAULT_METADATA_COLUMNS if name in schema)

    @staticmethod
    def _target_names_from_schema(schema: dict[str, str]) -> tuple[str, ...]:
        return tuple(
            name
            for name in schema
            if name.startswith("target") or name.startswith("aux_target")
        )

    @staticmethod
    def _dedupe_columns(columns: Iterable[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for column in columns:
            if column not in seen:
                deduped.append(column)
                seen.add(column)
        return deduped
