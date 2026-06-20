from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import polars as pl

from src.data import IngestionAgent


@dataclass(frozen=True)
class CachedPseudoInverse:
    era: str
    subset_name: str
    path: Path
    feature_names: tuple[str, ...]
    row_ids: tuple[str, ...]
    pseudo_inverse: np.ndarray


@dataclass(frozen=True)
class NeutralizationBenchmark:
    era: str
    subset_name: str
    row_count: int
    feature_count: int
    on_the_fly_ms: float
    cached_ms: float


class NeutralizationEngine:
    """Caches era-level pseudo-inverses for fast feature neutralization."""

    def __init__(
        self,
        cache_root: str | Path = Path("artifacts") / "cache",
        *,
        rcond: float = 1e-12,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.neutralization_root = self.cache_root / "neutralization"
        self.rcond = rcond
        self.neutralization_root.mkdir(parents=True, exist_ok=True)

    def compute_era_pseudo_inverse(self, feature_matrix: np.ndarray) -> np.ndarray:
        matrix = self._coerce_feature_matrix(feature_matrix)
        return np.linalg.pinv(matrix, rcond=self.rcond)

    def neutralize_tensor(
        self,
        predictions: np.ndarray,
        feature_matrix: np.ndarray,
        *,
        proportion: float = 1.0,
        pseudo_inverse: np.ndarray | None = None,
    ) -> np.ndarray:
        matrix = self._coerce_feature_matrix(feature_matrix)
        scores = self._coerce_prediction_vector(
            predictions, expected_rows=matrix.shape[0]
        )
        pinv = (
            self._coerce_pseudo_inverse(
                pseudo_inverse, expected_shape=(matrix.shape[1], matrix.shape[0])
            )
            if pseudo_inverse is not None
            else self.compute_era_pseudo_inverse(matrix)
        )
        return scores - proportion * (matrix @ (pinv @ scores))

    def cache_subsets(
        self,
        agent: IngestionAgent,
        dataset_name: str,
        subset_name: str,
        *,
        eras: Iterable[str] | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        feature_names = agent.get_feature_names(subset_name)
        target_eras = (
            tuple(eras) if eras is not None else self.list_eras(agent, dataset_name)
        )

        written_paths: list[Path] = []
        for era in target_eras:
            cache_path = self.cache_path(subset_name, era)
            if cache_path.exists() and not overwrite:
                written_paths.append(cache_path)
                continue

            era_frame = self.collect_era_frame(agent, dataset_name, subset_name, era)
            feature_matrix = era_frame.select(list(feature_names)).to_numpy()
            pseudo_inverse = self.compute_era_pseudo_inverse(feature_matrix)
            row_ids = np.asarray(era_frame.get_column("id").to_list(), dtype=str)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                pseudo_inverse=pseudo_inverse,
                row_ids=row_ids,
                feature_names=np.asarray(feature_names, dtype=str),
                era=np.asarray([era], dtype=str),
                dataset_name=np.asarray([dataset_name], dtype=str),
                subset_name=np.asarray([subset_name], dtype=str),
            )
            written_paths.append(cache_path)

        return written_paths

    def load_cached_pseudo_inverse(
        self,
        subset_name: str,
        era: str,
    ) -> CachedPseudoInverse:
        cache_path = self.cache_path(subset_name, era)
        if not cache_path.exists():
            raise FileNotFoundError(f"Neutralization cache is missing at {cache_path}")

        with np.load(cache_path, allow_pickle=False) as cached_file:
            return CachedPseudoInverse(
                era=str(cached_file["era"][0]),
                subset_name=str(cached_file["subset_name"][0]),
                path=cache_path,
                feature_names=tuple(cached_file["feature_names"].tolist()),
                row_ids=tuple(cached_file["row_ids"].tolist()),
                pseudo_inverse=np.asarray(
                    cached_file["pseudo_inverse"], dtype=np.float64
                ),
            )

    def neutralize_era(
        self,
        agent: IngestionAgent,
        dataset_name: str,
        subset_name: str,
        era: str,
        predictions: np.ndarray,
        *,
        proportion: float = 1.0,
        save_if_missing: bool = True,
    ) -> np.ndarray:
        feature_names = agent.get_feature_names(subset_name)
        era_frame = self.collect_era_frame(agent, dataset_name, subset_name, era)
        feature_matrix = era_frame.select(list(feature_names)).to_numpy()
        row_ids = tuple(era_frame.get_column("id").to_list())

        cache_path = self.cache_path(subset_name, era)
        if cache_path.exists():
            cached = self.load_cached_pseudo_inverse(subset_name, era)
            if cached.feature_names != feature_names:
                raise ValueError(
                    f"Cached feature names for era '{era}' do not match subset '{subset_name}'"
                )
            if cached.row_ids != row_ids:
                raise ValueError(
                    f"Cached row ids for era '{era}' no longer match dataset '{dataset_name}'"
                )
            pseudo_inverse = cached.pseudo_inverse
        else:
            pseudo_inverse = self.compute_era_pseudo_inverse(feature_matrix)
            if save_if_missing:
                self.cache_subsets(
                    agent,
                    dataset_name,
                    subset_name,
                    eras=[era],
                    overwrite=True,
                )

        return self.neutralize_tensor(
            predictions,
            feature_matrix,
            proportion=proportion,
            pseudo_inverse=pseudo_inverse,
        )

    def benchmark_era(
        self,
        agent: IngestionAgent,
        dataset_name: str,
        subset_name: str,
        era: str,
        predictions: np.ndarray,
        *,
        proportion: float = 1.0,
    ) -> NeutralizationBenchmark:
        feature_names = agent.get_feature_names(subset_name)
        era_frame = self.collect_era_frame(agent, dataset_name, subset_name, era)
        feature_matrix = era_frame.select(list(feature_names)).to_numpy()
        score_vector = self._coerce_prediction_vector(
            predictions,
            expected_rows=feature_matrix.shape[0],
        )

        start = perf_counter()
        self.neutralize_tensor(score_vector, feature_matrix, proportion=proportion)
        on_the_fly_ms = (perf_counter() - start) * 1000.0

        self.cache_subsets(agent, dataset_name, subset_name, eras=[era], overwrite=True)
        cached = self.load_cached_pseudo_inverse(subset_name, era)
        start = perf_counter()
        self.neutralize_tensor(
            score_vector,
            feature_matrix,
            proportion=proportion,
            pseudo_inverse=cached.pseudo_inverse,
        )
        cached_ms = (perf_counter() - start) * 1000.0

        return NeutralizationBenchmark(
            era=era,
            subset_name=subset_name,
            row_count=feature_matrix.shape[0],
            feature_count=feature_matrix.shape[1],
            on_the_fly_ms=on_the_fly_ms,
            cached_ms=cached_ms,
        )

    def list_eras(self, agent: IngestionAgent, dataset_name: str) -> tuple[str, ...]:
        era_column = (
            agent.scan_dataset(dataset_name, include_metadata=True)
            .select("era")
            .unique()
            .sort("era")
            .collect()
            .get_column("era")
            .to_list()
        )
        return tuple(str(value) for value in era_column)

    def collect_era_frame(
        self,
        agent: IngestionAgent,
        dataset_name: str,
        subset_name: str,
        era: str,
    ) -> pl.DataFrame:
        feature_names = list(agent.get_feature_names(subset_name))
        return (
            agent.scan_dataset(
                dataset_name,
                feature_subset=subset_name,
                include_metadata=True,
                include_targets=False,
            )
            .filter(pl.col("era") == era)
            .select(["id", "era", *feature_names])
            .collect()
        )

    def cache_path(self, subset_name: str, era: str) -> Path:
        subset_root = self.neutralization_root / subset_name
        return subset_root / f"{self._normalize_era_label(era)}.npz"

    @staticmethod
    def _normalize_era_label(era: str) -> str:
        digits = re.findall(r"\d+", str(era))
        if digits:
            return f"era_{digits[0].zfill(4)}"
        sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(era)).strip("_")
        return f"era_{sanitized}" if not sanitized.startswith("era_") else sanitized

    @staticmethod
    def _coerce_feature_matrix(feature_matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(feature_matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("feature_matrix must be a 2D array")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("feature_matrix must be non-empty")
        return matrix

    @staticmethod
    def _coerce_prediction_vector(
        predictions: np.ndarray,
        *,
        expected_rows: int,
    ) -> np.ndarray:
        scores = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if scores.shape[0] != expected_rows:
            raise ValueError(
                f"predictions length {scores.shape[0]} does not match expected row count {expected_rows}"
            )
        return scores

    @staticmethod
    def _coerce_pseudo_inverse(
        pseudo_inverse: np.ndarray,
        *,
        expected_shape: Sequence[int],
    ) -> np.ndarray:
        matrix = np.asarray(pseudo_inverse, dtype=np.float64)
        if matrix.shape != tuple(expected_shape):
            raise ValueError(
                f"pseudo_inverse shape {matrix.shape} does not match expected shape {tuple(expected_shape)}"
            )
        return matrix
