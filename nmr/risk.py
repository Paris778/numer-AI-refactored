"""Risk transforms: per-era feature neutralization with cache-aware least squares.

`NeutralizationEngine` applies Numerai-style linear feature neutralization to a
prediction column on a per-era basis. The expensive least-squares coefficients
are cached via the per-era pseudo-inverse of the design matrix so repeated
sweeps can reuse solves safely across different prediction vectors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from nmr.config import REPO_ROOT
from nmr._transforms import neutralize_array

logger = logging.getLogger("nmr.risk")

__all__ = ["NeutralizationEngine"]

_INTERCEPT_AWARE = True

DEFAULT_CACHE_MAX_BYTES = 2 * 2**30  # 2 GiB


class NeutralizationEngine:
    """Per-era, intercept-aware neutralization with validated cache reuse."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        max_cache_bytes: int | None = None,
    ) -> None:
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else REPO_ROOT / "artifacts" / "cache" / "neutralization"
        )
        self._max_cache_bytes = (
            DEFAULT_CACHE_MAX_BYTES
            if max_cache_bytes is None
            else int(max_cache_bytes)
        )
        if self._max_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be >= 0")
        total = self.cache_size_bytes()
        logger.info(
            "[neutralization] cache dir=%s max_bytes=%d current_bytes=%d",
            self._cache_dir, self._max_cache_bytes, total,
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
        logger.info("[neutralize] neutralizing %d eras", len(eras))
        parts: list[pl.DataFrame] = []

        for idx, era in enumerate(eras, start=1):
            if idx == 1 or idx == len(eras) or idx % 50 == 0:
                logger.info("[neutralize] era %d/%d: %s", idx, len(eras), era)
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

        result = pl.concat(parts).sort("__row_idx__").drop("__row_idx__")
        logger.info("[neutralize] complete")
        return result

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
            logger.warning(
                "[neutralize] era %s has zero-variance predictions; returning unchanged",
                era_label,
            )
            return np.asarray(pred, dtype=float).copy()
        if era_df.height <= len(list(feature_cols)) + 1:
            logger.warning(
                "[neutralize] era %s has %d rows <= %d features+intercept; "
                "neutralization fits exactly and the output may be near-zero",
                era_label,
                era_df.height,
                len(list(feature_cols)),
            )

        design = _design_matrix(features)
        pseudo_inverse = self._load_or_compute_pseudo_inverse(
            era_df,
            era_label=era_label,
            feature_cols=feature_cols,
            design=design,
        )
        return neutralize_array(
            pred, features, proportion, pseudo_inverse=pseudo_inverse
        )

    def _load_or_compute_pseudo_inverse(
        self,
        era_df: pl.DataFrame,
        *,
        era_label: str,
        feature_cols: Sequence[str],
        design: np.ndarray,
    ) -> np.ndarray:
        metadata = self._cache_metadata(
            era_df,
            era_label=era_label,
            feature_cols=feature_cols,
        )
        pseudo_inverse_path, metadata_path = self._cache_paths(metadata)

        cached = self._load_cached_array(
            pseudo_inverse_path,
            metadata_path,
            expected_metadata=metadata,
        )
        if cached is not None:
            return cached

        pseudo_inverse = _compute_pseudo_inverse(design)
        self._store_cached_array(
            pseudo_inverse_path,
            metadata_path,
            metadata=metadata,
            array=pseudo_inverse,
        )
        return pseudo_inverse

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

    def cache_size_bytes(self) -> int:
        if not self._cache_dir.exists():
            return 0
        return sum(
            path.stat().st_size for path in self._cache_dir.iterdir() if path.is_file()
        )

    def _evict_to_budget(self) -> None:
        if not self._cache_dir.exists():
            return
        files = sorted(
            (p for p in self._cache_dir.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= self._max_cache_bytes:
                break
            try:
                size = path.stat().st_size
                path.unlink()
                total -= size
            except OSError:
                continue
        if total > self._max_cache_bytes:
            logger.warning(
                "[neutralization] cache still above budget (%d bytes); "
                "raise risk.cache_max_bytes or clear artifacts/cache/neutralization",
                total,
            )

    def _load_cached_array(
        self,
        array_path: Path,
        metadata_path: Path,
        *,
        expected_metadata: dict[str, object],
    ) -> np.ndarray | None:
        if not array_path.exists() or not metadata_path.exists():
            return None

        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if cached_metadata != expected_metadata:
            return None

        try:
            array = np.load(array_path)
        except (OSError, ValueError, EOFError):
            return None
        try:
            os.utime(array_path)
            os.utime(metadata_path)
        except OSError:
            pass
        return np.asarray(array, dtype=float)

    def _store_cached_array(
        self,
        array_path: Path,
        metadata_path: Path,
        *,
        metadata: dict[str, object],
        array: np.ndarray,
    ) -> None:
        from nmr._atomicio import atomic_write_text

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # np.save appends `.npy` when the target name lacks it, so the temp
        # file must keep the `.npy` suffix or the replace below would miss it.
        tmp_array = array_path.with_name(
            f"{array_path.stem}.tmp.{os.getpid()}{array_path.suffix}"
        )
        try:
            np.save(tmp_array, np.asarray(array, dtype=float))
            os.replace(tmp_array, array_path)
            atomic_write_text(
                metadata_path,
                json.dumps(metadata, sort_keys=True, indent=2),
            )
        finally:
            if tmp_array.exists():
                tmp_array.unlink()
        self._evict_to_budget()


def _design_matrix(features: np.ndarray) -> np.ndarray:
    return np.hstack((features, np.ones((features.shape[0], 1), dtype=float)))


def _compute_pseudo_inverse(design: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.pinv(design, rcond=1e-6), dtype=float)
