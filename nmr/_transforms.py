"""Shared rank-domain transforms used by scoring and ensembling.

These helpers are the single source of truth for Numerai-style rank geometry in
this package. Keeping them centralized prevents the evaluation and ensembling
paths from silently drifting apart over time.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "gaussianize",
    "power_1_5",
    "rank_gaussianize",
    "rank_gaussianize_unit_variance",
    "standardize_unit_variance",
    "tie_kept_rank",
]


def tie_kept_rank(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return array
    return (stats.rankdata(array, method="average") - 0.5) / array.size


def gaussianize(values: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(np.asarray(values, dtype=float).reshape(-1))


def standardize_unit_variance(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return array
    std = float(np.std(array, ddof=0))
    if std == 0.0 or not np.isfinite(std):
        return np.zeros_like(array, dtype=float)
    return array / std


def rank_gaussianize(values: np.ndarray) -> np.ndarray:
    return gaussianize(tie_kept_rank(values))


def rank_gaussianize_unit_variance(values: np.ndarray) -> np.ndarray:
    return standardize_unit_variance(rank_gaussianize(values))


def power_1_5(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.abs(array) ** 1.5
