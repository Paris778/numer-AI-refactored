"""Data layer — lazy, schema-aware Polars ingestion for Numerai v5.2+ data.

``IngestionAgent`` is the single access point for all splits. It enforces:

- construction-time inertness: ``__init__`` touches no files.
- lazy-only scans: :meth:`scan` returns a ``pl.LazyFrame`` via
  ``pl.scan_parquet(...).select(...)``; ``.collect()`` is never called here.
- column pushdown by construction: the explicit ``.select()`` prevents any
  unlisted column from being read from disk.
- schema-level memoization: ``collect_schema()`` reads parquet metadata only and
  is called at most once per split across the agent's lifetime.
- deterministic column order: ``era · id · features(subset) · targets``.
- target validation plus split-local intersection: requested targets are first
  validated against ``features.json`` so typos fail loudly, then intersected
  with the physical split schema so absent target columns for a given split do
  not raise.
- fail loud, fail late: missing files raise ``FileNotFoundError`` on first
  access; invalid subsets and targets raise ``ValueError`` with valid options.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from nmr.config import DataConfig

__all__ = ["IngestionAgent"]


_SPLIT_FILES: dict[str, str] = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "live": "live.parquet",
}

_META_COLUMNS: tuple[str, ...] = ("era", "id")


class IngestionAgent:
    """Lazy, deterministic ingestion agent scoped to a single DataConfig."""

    def __init__(self, data: DataConfig) -> None:
        self._data = data
        self._metadata: dict | None = None
        self._schema_cache: dict[str, pl.Schema] = {}

    def _split_path(self, split: str) -> Path:
        try:
            filename = _SPLIT_FILES[split]
        except KeyError:
            raise ValueError(
                f"Unknown split {split!r}; valid splits: {sorted(_SPLIT_FILES)}"
            ) from None
        return self._data.path(filename)

    def _features_json_path(self) -> Path:
        return self._data.data_dir / self._data.version / "features.json"

    def _metadata_raw(self) -> dict:
        if self._metadata is None:
            path = self._features_json_path()
            try:
                self._metadata = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"features.json not found at {path}. "
                    "Download the Numerai dataset before calling metadata methods."
                ) from None
        return self._metadata

    @property
    def feature_metadata(self) -> dict:
        """Return a defensive copy of parsed ``features.json`` metadata."""
        raw = self._metadata_raw()
        metadata = dict(raw)
        metadata["feature_sets"] = {
            key: list(values) for key, values in raw["feature_sets"].items()
        }
        targets = raw.get("targets")
        if isinstance(targets, list):
            metadata["targets"] = list(targets)
        elif isinstance(targets, dict):
            metadata["targets"] = dict(targets)
        return metadata

    @property
    def feature_sets(self) -> dict[str, list[str]]:
        """Return a defensive copy of all named feature sets."""
        return {
            key: list(values)
            for key, values in self._metadata_raw()["feature_sets"].items()
        }

    def available_targets(self) -> list[str]:
        """Return all target column names declared in ``features.json``."""
        raw = self._metadata_raw()["targets"]
        return list(raw) if isinstance(raw, list) else list(raw.keys())

    def features(self, subset: str | None = None) -> list[str]:
        """Return the ordered feature column names for ``subset``."""
        key = subset if subset is not None else self._data.feature_set
        sets = self._metadata_raw()["feature_sets"]
        if key not in sets:
            raise ValueError(
                f"Feature subset {key!r} not found in features.json; "
                f"valid subsets: {sorted(sets)}"
            )
        return list(sets[key])

    def schema(self, split: str) -> Mapping[str, pl.DataType]:
        """Return parquet schema for ``split`` using metadata-only I/O."""
        if split not in self._schema_cache:
            path = self._split_path(split)
            if not path.exists():
                raise FileNotFoundError(
                    f"Split file not found: {path}. "
                    "Download the Numerai dataset before accessing schema or data."
                )
            self._schema_cache[split] = pl.scan_parquet(path).collect_schema()
        return self._schema_cache[split]

    def scan(
        self,
        split: str,
        *,
        subset: str | None = None,
        targets: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pl.LazyFrame:
        """Return a lazy scan of ``split`` with an explicit minimal selection."""
        path = self._split_path(split)

        if columns is not None:
            if not path.exists():
                raise FileNotFoundError(f"Split file not found: {path}.")
            return pl.scan_parquet(path).select(list(columns))

        file_schema = self.schema(split)
        schema_col_set = set(file_schema.names())

        selected: list[str] = []
        for col in _META_COLUMNS:
            if col in schema_col_set:
                selected.append(col)

        for col in self.features(subset):
            if col in schema_col_set and col not in selected:
                selected.append(col)

        target_list = list(targets) if targets is not None else list(self._data.targets)
        known_targets = set(self.available_targets())
        unknown_targets = [
            target for target in target_list if target not in known_targets
        ]
        if unknown_targets:
            raise ValueError(
                "Unknown target(s) requested: "
                f"{unknown_targets}. Valid targets: {sorted(known_targets)}"
            )

        for col in target_list:
            if col in schema_col_set and col not in selected:
                selected.append(col)

        return pl.scan_parquet(path).select(selected)

    def load(self, split: str, **kwargs) -> pl.DataFrame:
        """Collect the lazy scan for ``split``."""
        return self.scan(split, **kwargs).collect()

    def train(self, **kwargs) -> pl.LazyFrame:
        """Equivalent to ``scan(\"train\", **kwargs)``."""
        return self.scan("train", **kwargs)

    def validation(self, **kwargs) -> pl.LazyFrame:
        """Equivalent to ``scan(\"validation\", **kwargs)``."""
        return self.scan("validation", **kwargs)

    def live(self, **kwargs) -> pl.LazyFrame:
        """Equivalent to ``scan(\"live\", **kwargs)``."""
        return self.scan("live", **kwargs)
