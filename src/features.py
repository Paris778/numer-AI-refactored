from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl

EraFrame = pl.DataFrame | pl.LazyFrame


@dataclass(frozen=True)
class EraFold:
    fold_number: int
    train_eras: np.ndarray
    validation_eras: np.ndarray


class PurgedEraSplitter:
    """Era-aware cross-validation with a symmetric purge buffer."""

    def __init__(self, n_splits: int, purge_buffer: int = 4) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if purge_buffer < 0:
            raise ValueError("purge_buffer must be non-negative")

        self.n_splits = n_splits
        self.purge_buffer = purge_buffer

    def split(
        self,
        df: EraFrame,
        era_col: str = "era",
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        unique_eras = self._collect_unique_eras(df, era_col)
        if unique_eras.size < self.n_splits:
            raise ValueError(
                f"n_splits={self.n_splits} cannot exceed unique era count {unique_eras.size}"
            )

        era_ordinals = self._era_ordinals(unique_eras)
        validation_folds = [
            fold for fold in np.array_split(unique_eras, self.n_splits) if fold.size
        ]

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for validation_eras in validation_folds:
            validation_mask = np.isin(unique_eras, validation_eras)
            validation_ordinals = era_ordinals[validation_mask]
            distances = np.abs(era_ordinals[:, None] - validation_ordinals[None, :])
            purge_mask = np.min(distances, axis=1) <= self.purge_buffer
            train_mask = ~validation_mask & ~purge_mask
            splits.append((unique_eras[train_mask].copy(), validation_eras.copy()))

        return splits

    def iter_folds(
        self,
        df: EraFrame,
        era_col: str = "era",
    ) -> Iterable[EraFold]:
        for fold_number, (train_eras, validation_eras) in enumerate(
            self.split(df, era_col=era_col),
            start=1,
        ):
            yield EraFold(
                fold_number=fold_number,
                train_eras=train_eras,
                validation_eras=validation_eras,
            )

    def split_row_indices(
        self,
        df: EraFrame,
        era_col: str = "era",
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        era_values = self._collect_era_column(df, era_col)
        era_splits = self.split(df, era_col=era_col)

        index_splits: list[tuple[np.ndarray, np.ndarray]] = []
        for train_eras, validation_eras in era_splits:
            train_indices = np.flatnonzero(np.isin(era_values, train_eras))
            validation_indices = np.flatnonzero(np.isin(era_values, validation_eras))
            index_splits.append((train_indices, validation_indices))

        return index_splits

    @staticmethod
    def _collect_unique_eras(df: EraFrame, era_col: str) -> np.ndarray:
        if isinstance(df, pl.LazyFrame):
            era_frame = df.select(era_col).unique().sort(era_col).collect()
        else:
            era_frame = df.select(era_col).unique().sort(era_col)
        return np.asarray(era_frame.get_column(era_col).to_list(), dtype=str)

    @staticmethod
    def _collect_era_column(df: EraFrame, era_col: str) -> np.ndarray:
        if isinstance(df, pl.LazyFrame):
            era_frame = df.select(era_col).collect()
        else:
            era_frame = df.select(era_col)
        return np.asarray(era_frame.get_column(era_col).to_list(), dtype=str)

    @staticmethod
    def _era_ordinals(eras: np.ndarray) -> np.ndarray:
        parsed_ordinals: list[int] = []
        for era in eras:
            digits = re.findall(r"\d+", str(era))
            if not digits:
                raise ValueError(
                    f"Non-numeric era format detected: {era}. Cannot compute safe temporal purge distance."
                )
            parsed_ordinals.append(int(digits[0]))
        return np.asarray(parsed_ordinals, dtype=np.int64)


class FeatureFactory:
    """Composable feature transforms over Polars frames."""

    def __init__(
        self,
        frame: EraFrame,
        *,
        era_col: str = "era",
        id_col: str = "id",
    ) -> None:
        self.era_col = era_col
        self.id_col = id_col
        self._frame = frame.lazy() if isinstance(frame, pl.DataFrame) else frame

    @property
    def frame(self) -> pl.LazyFrame:
        return self._frame

    def select_columns(self, columns: Iterable[str]) -> "FeatureFactory":
        self._frame = self._frame.select(list(columns))
        return self

    def add_era_rank(
        self, column_name: str, *, output_name: str | None = None
    ) -> "FeatureFactory":
        alias = output_name or f"{column_name}_era_rank"
        self._frame = self._frame.with_columns(
            (
                pl.col(column_name)
                .rank(method="average")
                .over(self.era_col)
                .cast(pl.Float64)
                / pl.len().over(self.era_col).cast(pl.Float64)
            ).alias(alias)
        )
        return self

    def add_noise_baseline(
        self,
        *,
        output_name: str = "noise_baseline",
        seed: int = 0,
    ) -> "FeatureFactory":
        self._frame = self._frame.with_row_index("__row_index")
        self._frame = self._frame.with_columns(
            (
                (
                    (
                        (pl.col("__row_index").cast(pl.Float64) + float(seed + 1))
                        * 12.9898
                    ).sin()
                    * 43758.5453
                )
                % 1.0
            ).alias(output_name)
        ).drop("__row_index")
        return self

    def collect(self) -> pl.DataFrame:
        return self._frame.collect()
