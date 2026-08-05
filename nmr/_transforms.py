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
    "neutralize_array",
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


def neutralize_array(
    pred: np.ndarray,
    features: np.ndarray,
    proportion: float = 1.0,
    *,
    pseudo_inverse: np.ndarray | None = None,
) -> np.ndarray:
    """Per-era intercept-aware linear neutralization (single source of truth).

    The engine passes its cached per-era pseudo-inverse; the deployment closure
    passes ``None`` and the design pseudo-inverse is computed here so both paths
    share identical geometry and ``rcond``. Zero-variance predictions are
    returned unchanged (the era keeps its rows; callers decide on logging).
    """
    pred_array = np.asarray(pred, dtype=float).reshape(-1)
    feature_matrix = np.asarray(features, dtype=float)
    if pred_array.shape[0] != feature_matrix.shape[0]:
        raise ValueError("pred and features must have the same number of rows")
    if not np.all(np.isfinite(pred_array)) or not np.all(np.isfinite(feature_matrix)):
        raise ValueError("pred and features must contain only finite values")
    if np.std(pred_array) == 0.0:
        return pred_array.copy()

    design = np.hstack(
        (feature_matrix, np.ones((feature_matrix.shape[0], 1), dtype=float))
    )
    if pseudo_inverse is None:
        pseudo_inverse = np.asarray(np.linalg.pinv(design, rcond=1e-6), dtype=float)
    coeffs = pseudo_inverse.dot(pred_array)
    adjustment = design.dot(coeffs)
    return pred_array - (proportion * adjustment)
