"""Risk transforms: per-era feature neutralization with cache-aware least squares.

`NeutralizationEngine` applies Numerai-style linear feature neutralization to a
prediction column on a per-era basis. The expensive least-squares coefficients
are cached per era/content signature so repeated sweeps can reuse solves safely.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from nmr.config import REPO_ROOT

__all__ = ["NeutralizationEngine"]

_INTERCEPT_AWARE = True


class NeutralizationEngine:
    """Per-era, intercept-aware neutralization with validated cache reuse."""

    def __init__(self, *, cache_dir: Path | None = None) -> None:
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else REPO_ROOT / "artifacts" / "cache" / "neutralization"
        )

    def neutralize(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        feature_cols: Sequence[str],
        era_col: str = "era",
        proportion: float = 1.0,
    ) -> pl.DataFrame:
        """Return ``df`` with a per-era neutralized ``pred_col``.

        `proportion=0.0` is an identity transform. `proportion=1.0` applies the
        full least-squares projection subtraction used by `numerai_tools`.
        """
        if not 0.0 <= proportion <= 1.0:
            raise ValueError("proportion must be between 0.0 and 1.0 inclusive")

        feature_list = list(feature_cols)
        if not feature_list:
            raise ValueError("feature_cols must contain at least one feature")

        if proportion == 0.0:
            return df.clone()

        required_cols = [era_col, pred_col, *feature_list]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        work_df = df.with_row_index("__row_idx__")
        eras = work_df.get_column(era_col).unique(maintain_order=True).to_list()
        parts: list[pl.DataFrame] = []

        for era in eras:
            era_df = work_df.filter(pl.col(era_col) == era)
            neutralized = self._neutralize_era(
                era_df,
                era_label=str(era),
                pred_col=pred_col,
                feature_cols=feature_list,
                proportion=proportion,
            )
            parts.append(
                era_df.with_columns(pl.Series(name=pred_col, values=neutralized))
            )

        return pl.concat(parts).sort("__row_idx__").drop("__row_idx__")

    def _neutralize_era(
        self,
        era_df: pl.DataFrame,
        *,
        era_label: str,
        pred_col: str,
        feature_cols: Sequence[str],
        proportion: float,
    ) -> np.ndarray:
        pred = self._column_values(era_df, pred_col)
        features = self._feature_matrix(era_df, feature_cols)

        if np.std(pred) == 0.0:
            return np.full_like(pred, np.nan, dtype=float)

        design = self._design_matrix(features)
        coeffs = self._load_or_compute_coefficients(
            era_df,
            era_label=era_label,
            feature_cols=feature_cols,
            design=design,
            pred=pred,
        )
        adjustment = design.dot(coeffs)
        return pred - (proportion * adjustment)

    def _load_or_compute_coefficients(
        self,
        era_df: pl.DataFrame,
        *,
        era_label: str,
        feature_cols: Sequence[str],
        design: np.ndarray,
        pred: np.ndarray,
    ) -> np.ndarray:
        metadata = self._cache_metadata(
            era_df,
            era_label=era_label,
            feature_cols=feature_cols,
        )
        coeffs_path, metadata_path = self._cache_paths(metadata)

        cached = self._load_cached_coefficients(
            coeffs_path,
            metadata_path,
            expected_metadata=metadata,
        )
        if cached is not None:
            return cached

        coeffs = self._solve_least_squares(design, pred)
        self._store_cached_coefficients(
            coeffs_path,
            metadata_path,
            metadata=metadata,
            coeffs=coeffs,
        )
        return coeffs

    def _solve_least_squares(self, design: np.ndarray, pred: np.ndarray) -> np.ndarray:
        coeffs = np.linalg.lstsq(design, pred.reshape(-1, 1), rcond=1e-6)[0]
        return np.asarray(coeffs).reshape(-1)

    def _design_matrix(self, features: np.ndarray) -> np.ndarray:
        intercept = np.ones((features.shape[0], 1), dtype=float)
        return np.hstack((features, intercept))

    def _column_values(self, df: pl.DataFrame, col: str) -> np.ndarray:
        values = df.get_column(col).cast(pl.Float64).to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Column {col!r} contains null or non-finite values")
        return np.asarray(values, dtype=float)

    def _feature_matrix(
        self, df: pl.DataFrame, feature_cols: Sequence[str]
    ) -> np.ndarray:
        matrix = df.select(list(feature_cols)).cast(pl.Float64).to_numpy()
        if not np.all(np.isfinite(matrix)):
            raise ValueError("feature_cols contain null or non-finite values")
        return np.asarray(matrix, dtype=float)

    def _cache_metadata(
        self,
        era_df: pl.DataFrame,
        *,
        era_label: str,
        feature_cols: Sequence[str],
    ) -> dict[str, object]:
        if "id" in era_df.columns:
            row_ids = [str(value) for value in era_df.get_column("id").to_list()]
        else:
            row_ids = [str(idx) for idx in era_df.get_column("__row_idx__").to_list()]

        row_ids_payload = json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
        return {
            "era": era_label,
            "feature_cols": list(feature_cols),
            "row_count": int(era_df.height),
            "row_ids_sha256": hashlib.sha256(row_ids_payload).hexdigest(),
            "intercept": _INTERCEPT_AWARE,
        }

    def _cache_paths(self, metadata: dict[str, object]) -> tuple[Path, Path]:
        cache_key = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        era_label = str(metadata["era"]).replace("/", "_").replace("\\", "_")
        base = self._cache_dir / f"era_{era_label}_{cache_key}"
        return base.with_suffix(".npy"), base.with_suffix(".json")

    def _load_cached_coefficients(
        self,
        coeffs_path: Path,
        metadata_path: Path,
        *,
        expected_metadata: dict[str, object],
    ) -> np.ndarray | None:
        if not coeffs_path.exists() or not metadata_path.exists():
            return None

        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if cached_metadata != expected_metadata:
            return None

        try:
            coeffs = np.load(coeffs_path)
        except OSError:
            return None
        return np.asarray(coeffs, dtype=float).reshape(-1)

    def _store_cached_coefficients(
        self,
        coeffs_path: Path,
        metadata_path: Path,
        *,
        metadata: dict[str, object],
        coeffs: np.ndarray,
    ) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(coeffs_path, np.asarray(coeffs, dtype=float))
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, indent=2),
            encoding="utf-8",
        )
