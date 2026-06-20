from __future__ import annotations

"""Data loading and alignment utilities for Numerai research workflows."""

import copy
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "data"


@dataclass(frozen=True)
class ResearchDataBundle:
    """Container for aligned datasets and feature metadata used in experiments."""

    data_dir: Path
    features_path: Path
    feature_metadata: dict
    feature_set: list[str]
    train: pd.DataFrame
    validation: pd.DataFrame
    validation_benchmarks: pd.DataFrame


def resolve_data_dir(
    repo_root: str | Path,
    data_version: str,
) -> Path:
    """Resolve and validate the versioned data directory path."""
    data_dir = Path(repo_root) / "data" / data_version
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected data directory was not found: {data_dir}")
    return data_dir


def ensure_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize scalar/list-like inputs into a concrete list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def unique_columns(*column_groups: list[str]) -> list[str]:
    """Combine column groups while preserving first-seen order and uniqueness."""
    ordered_columns: list[str] = []
    for group in column_groups:
        for column in group:
            if column not in ordered_columns:
                ordered_columns.append(column)
    return ordered_columns


def build_dataset_columns(
    feature_set: list[str],
    target_cols: str | list[str] | tuple[str, ...] | None = None,
    extra_columns: str | list[str] | tuple[str, ...] | None = None,
    include_id: bool = True,
    include_era: bool = True,
) -> list[str]:
    """Construct ordered parquet column selections for Numerai datasets."""
    base_columns: list[str] = []
    if include_id:
        base_columns.append("id")
    if include_era:
        base_columns.append("era")

    return unique_columns(
        base_columns,
        ensure_list(target_cols),
        list(feature_set),
        ensure_list(extra_columns),
    )


def read_parquet_with_id_index(
    path: str | Path,
    columns: list[str] | None = None,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Read parquet data and ensure the returned frame is indexed by id.

    When use_cache=True, this function persists a cache artifact on disk keyed by
    source path, requested columns, and source file metadata (mtime + size), so
    repeated loads across notebooks can reuse preprocessed frames.
    """
    source_path = Path(path).resolve()
    selected_columns = tuple(columns) if columns is not None else None

    if not use_cache:
        frame = pd.read_parquet(source_path, columns=columns)
        if "id" in frame.columns:
            frame = frame.set_index("id")
        elif frame.index.name != "id":
            frame.index.name = "id"
        return frame

    frame = _read_parquet_with_id_index_cached(
        source_path=source_path,
        selected_columns=selected_columns,
        cache_dir=(Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR),
    )

    # Return a copy so callers can safely mutate without polluting cache state.
    return frame.copy()


def _cache_key_for_parquet(
    source_path: Path,
    selected_columns: tuple[str, ...] | None,
) -> str:
    source_stat = source_path.stat()
    columns_blob = "*" if selected_columns is None else "|".join(selected_columns)
    key_payload = (
        f"{source_path}|{source_stat.st_mtime_ns}|{source_stat.st_size}|{columns_blob}"
    )
    return hashlib.sha256(key_payload.encode("utf-8")).hexdigest()[:32]


def _cache_file_for_parquet(
    source_path: Path,
    selected_columns: tuple[str, ...] | None,
    cache_dir: Path,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _cache_key_for_parquet(source_path, selected_columns)
    return cache_dir / f"{cache_key}.parquet"


def _read_parquet_with_id_index_cached(
    source_path: Path,
    selected_columns: tuple[str, ...] | None,
    cache_dir: Path,
) -> pd.DataFrame:
    cache_file = _cache_file_for_parquet(source_path, selected_columns, cache_dir)

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    frame = pd.read_parquet(
        source_path, columns=list(selected_columns) if selected_columns else None
    )

    if "id" in frame.columns:
        frame = frame.set_index("id")
    elif frame.index.name != "id":
        frame.index.name = "id"

    temp_file = cache_file.with_suffix(".tmp.parquet")
    frame.to_parquet(temp_file)
    temp_file.replace(cache_file)

    return frame


@lru_cache(maxsize=8)
def _load_feature_metadata_cached(
    features_path: str,
    file_mtime_ns: int,
    file_size: int,
) -> dict[str, Any]:
    del file_mtime_ns, file_size
    with open(features_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_feature_set(
    features_path: str | Path,
    feature_set_key: str | list[str] | tuple[str, ...] | None = None,
    feature_list: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict, list[str]]:
    """Load feature metadata and return the selected feature set list."""
    features_path = Path(features_path)
    features_stat = features_path.stat()
    feature_metadata = _load_feature_metadata_cached(
        features_path=str(features_path),
        file_mtime_ns=features_stat.st_mtime_ns,
        file_size=features_stat.st_size,
    )
    feature_metadata_copy = copy.deepcopy(feature_metadata)

    if feature_list is not None:
        feature_set = [feature for feature in feature_list if feature != "id"]
        return feature_metadata_copy, feature_set

    selected_feature_set_keys = ensure_list(feature_set_key)
    if not selected_feature_set_keys:
        raise ValueError("Either feature_set_key or feature_list must be provided.")

    available_feature_sets = feature_metadata_copy.get("feature_sets", {})
    missing_feature_set_keys = [
        key for key in selected_feature_set_keys if key not in available_feature_sets
    ]
    if missing_feature_set_keys:
        available = ", ".join(sorted(available_feature_sets))
        raise KeyError(
            "Unknown feature_set_key values: "
            + ", ".join(sorted(missing_feature_set_keys))
            + f". Available keys: {available}"
        )

    feature_set = unique_columns(
        *[
            [feature for feature in available_feature_sets[key] if feature != "id"]
            for key in selected_feature_set_keys
        ]
    )
    return feature_metadata_copy, feature_set


def apply_validation_embargo(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    embargo_eras: int,
) -> pd.DataFrame:
    """Apply an era embargo after train cutoff to reduce temporal leakage."""
    if embargo_eras <= 0:
        return validation.copy()

    train_eras = pd.Series(train["era"].astype(str)).sort_values().unique()
    if len(train_eras) == 0:
        raise ValueError("train must contain at least one era to compute embargo.")

    last_train_era = int(train_eras[-1])
    embargo_values = {
        str(era).zfill(4)
        for era in range(last_train_era + 1, last_train_era + 1 + embargo_eras)
    }
    validation_eras = validation["era"].astype(str)
    return validation.loc[~validation_eras.isin(embargo_values)].copy()


def align_on_id(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-align two frames on id index with strict overlap enforcement."""
    common_ids = left.index.intersection(right.index)
    if len(common_ids) == 0:
        raise ValueError("No overlapping ids found while aligning dataframes.")

    return left.loc[common_ids].copy(), right.loc[common_ids].copy()


def load_benchmark_frame(
    benchmark_path: str | Path,
    benchmark_columns: str | list[str] | tuple[str, ...] | None = None,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load benchmark parquet and optionally project to requested columns."""
    benchmark_frame = read_parquet_with_id_index(
        benchmark_path,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    selected_columns = ensure_list(benchmark_columns)
    if not selected_columns:
        return benchmark_frame

    missing_columns = [
        column for column in selected_columns if column not in benchmark_frame.columns
    ]
    if missing_columns:
        raise KeyError(
            "Benchmark dataframe is missing columns: " + ", ".join(missing_columns)
        )
    return benchmark_frame[selected_columns].copy()


def load_dataset_pair(
    train_path: str | Path,
    validation_path: str | Path,
    columns: list[str],
    embargo_eras: int = 0,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/validation parquet pair with optional validation embargo."""
    train = read_parquet_with_id_index(
        train_path,
        columns=columns,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    validation = read_parquet_with_id_index(
        validation_path,
        columns=columns,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )

    if embargo_eras > 0:
        validation = apply_validation_embargo(
            train=train,
            validation=validation,
            embargo_eras=embargo_eras,
        )

    return train, validation


def load_research_data(
    repo_root: str | Path | None = None,
    data_version: str | None = None,
    feature_set_key: str | list[str] | tuple[str, ...] | None = None,
    target_col: str | list[str] | tuple[str, ...] | None = None,
    benchmark_col: str | list[str] | tuple[str, ...] | None = None,
    embargo_eras: int = 4,
    data_dir: str | Path | None = None,
    features_path: str | Path | None = None,
    train_path: str | Path | None = None,
    validation_path: str | Path | None = None,
    validation_benchmarks_path: str | Path | None = None,
    feature_list: list[str] | tuple[str, ...] | None = None,
    train_extra_columns: str | list[str] | tuple[str, ...] | None = None,
    validation_extra_columns: str | list[str] | tuple[str, ...] | None = None,
    benchmark_extra_columns: str | list[str] | tuple[str, ...] | None = None,
    align_validation_benchmarks: bool = True,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
) -> ResearchDataBundle:
    """Load and align train/validation/benchmark datasets for model evaluation.

    The returned validation and validation_benchmarks frames are id-aligned when
    align_validation_benchmarks=True, matching the downstream expectations used by
    utils.metrics.calculate_metrics.
    """
    if data_dir is None:
        if repo_root is None or data_version is None:
            raise ValueError(
                "Provide either data_dir or both repo_root and data_version."
            )
        data_dir = resolve_data_dir(repo_root=repo_root, data_version=data_version)
    else:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Expected data directory was not found: {data_dir}"
            )

    features_path = (
        Path(features_path) if features_path is not None else data_dir / "features.json"
    )
    if not features_path.exists():
        raise FileNotFoundError(
            f"Expected feature metadata file was not found: {features_path}"
        )

    feature_metadata, feature_set = load_feature_set(
        features_path=features_path,
        feature_set_key=feature_set_key,
        feature_list=feature_list,
    )

    target_columns = ensure_list(target_col)
    benchmark_columns = unique_columns(
        ensure_list(benchmark_col),
        ensure_list(benchmark_extra_columns),
    )
    train_columns = build_dataset_columns(
        feature_set=feature_set,
        target_cols=target_columns,
        extra_columns=train_extra_columns,
    )
    validation_columns = build_dataset_columns(
        feature_set=feature_set,
        target_cols=target_columns,
        extra_columns=validation_extra_columns,
    )

    train_path = (
        Path(train_path) if train_path is not None else data_dir / "train.parquet"
    )
    validation_path = (
        Path(validation_path)
        if validation_path is not None
        else data_dir / "validation.parquet"
    )
    validation_benchmarks_path = (
        Path(validation_benchmarks_path)
        if validation_benchmarks_path is not None
        else data_dir / "validation_benchmark_models.parquet"
    )

    train = read_parquet_with_id_index(
        train_path,
        columns=train_columns,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    validation = read_parquet_with_id_index(
        validation_path,
        columns=validation_columns,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    validation_benchmarks = load_benchmark_frame(
        benchmark_path=validation_benchmarks_path,
        benchmark_columns=benchmark_columns,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )

    primary_benchmark_col = ensure_list(benchmark_col)
    if primary_benchmark_col:
        missing_benchmarks = [
            column
            for column in primary_benchmark_col
            if column not in validation_benchmarks.columns
        ]
        if missing_benchmarks:
            raise KeyError(
                "Benchmark dataframe is missing requested columns: "
                + ", ".join(missing_benchmarks)
            )

    validation = apply_validation_embargo(
        train=train,
        validation=validation,
        embargo_eras=embargo_eras,
    )
    if align_validation_benchmarks:
        validation, validation_benchmarks = align_on_id(
            validation, validation_benchmarks
        )

    benchmark_payload = validation_benchmarks.copy()
    if primary_benchmark_col:
        benchmark_payload = validation_benchmarks[primary_benchmark_col].copy()

    return ResearchDataBundle(
        data_dir=data_dir,
        features_path=features_path,
        feature_metadata=feature_metadata,
        feature_set=feature_set,
        train=train,
        validation=validation,
        validation_benchmarks=benchmark_payload,
    )


def get_available_keys(data_dir: str | Path) -> list[str]:
    """Return sorted data-version subdirectory names under a data root."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    keys = []
    for item in data_dir.iterdir():
        if item.is_dir():
            keys.append(item.name)
    return sorted(keys)
