from __future__ import annotations

"""Post-processing utilities for Numerai prediction ranking and neutralization."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency guard
    tqdm = None


def _to_series(
    values: pd.Series | Sequence[float] | np.ndarray, index: pd.Index
) -> pd.Series:
    """Convert values to an index-aligned series without mutating caller inputs."""
    if isinstance(values, pd.Series):
        return values.reindex(index)
    return pd.Series(np.asarray(values, dtype=float), index=index)


def _resolve_index(
    predictions: pd.Series | Sequence[float] | np.ndarray,
    eras: pd.Series | Sequence[str] | np.ndarray,
    index: pd.Index | None,
) -> pd.Index:
    """Resolve a stable index preference order for downstream alignment."""
    if index is not None:
        return index
    if isinstance(predictions, pd.Series):
        return predictions.index
    if isinstance(eras, pd.Series):
        return eras.index
    return pd.RangeIndex(len(np.asarray(predictions)))


def _to_float_array(
    values: pd.Series | Sequence[float] | np.ndarray,
    index: pd.Index,
) -> np.ndarray:
    """Convert predictions-like inputs into a 1D float array aligned to index."""
    if isinstance(values, pd.Series):
        return values.reindex(index).to_numpy(dtype=np.float64, copy=False)

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        array = np.ravel(array)
    if array.shape[0] != len(index):
        raise ValueError(
            "values length must match index length: "
            f"expected {len(index)}, received {array.shape[0]}."
        )
    return array


def _to_era_array(
    eras: pd.Series | Sequence[str] | np.ndarray,
    index: pd.Index,
) -> np.ndarray:
    """Convert era labels into a 1D string numpy array aligned to index."""
    if isinstance(eras, pd.Series):
        era_series = eras.reindex(index)
    else:
        era_values = np.asarray(eras)
        if era_values.ndim != 1:
            era_values = np.ravel(era_values)
        if era_values.shape[0] != len(index):
            raise ValueError(
                "eras length must match index length: "
                f"expected {len(index)}, received {era_values.shape[0]}."
            )
        era_series = pd.Series(era_values, index=index)
    return era_series.astype(str).to_numpy(copy=False)


def _sorted_era_slices(
    era_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable sorted positions and contiguous era slice boundaries."""
    order = np.argsort(era_values, kind="mergesort")
    if order.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return order, empty, empty

    ordered_eras = era_values[order]
    boundaries = np.flatnonzero(ordered_eras[1:] != ordered_eras[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [order.size]))
    return order, starts, stops


def _progress_iter(
    iterable: Iterable[tuple[np.integer, np.integer]],
    *,
    enabled: bool,
    total: int,
    desc: str,
) -> Iterable[tuple[np.integer, np.integer]]:
    """Wrap iterables in tqdm when progress display is requested."""
    if not enabled:
        return iterable
    if tqdm is None:
        raise ImportError(
            "tqdm is required when show_progress=True. Install it with 'pip install tqdm'."
        )
    return tqdm(iterable, total=total, desc=desc)


def rank_by_era(
    predictions: pd.Series | Sequence[float] | np.ndarray,
    eras: pd.Series | Sequence[str] | np.ndarray,
    index: pd.Index | None = None,
    show_progress: bool = False,
) -> pd.Series:
    """Rank-normalize predictions independently within each era to [0, 1]."""
    index = _resolve_index(predictions, eras, index)
    prediction_values = _to_float_array(predictions, index)
    era_values = _to_era_array(eras, index)

    ranked = np.full(len(index), np.nan, dtype=np.float64)
    order, starts, stops = _sorted_era_slices(era_values)

    for start, stop in _progress_iter(
        zip(starts, stops),
        enabled=show_progress,
        total=len(starts),
        desc="Ranking eras",
    ):
        era_positions = order[start:stop]
        era_predictions = prediction_values[era_positions]
        valid_mask = ~np.isnan(era_predictions)

        if not np.any(valid_mask):
            continue

        valid_positions = era_positions[valid_mask]
        valid_predictions = era_predictions[valid_mask]
        ranked_positions = valid_positions[
            np.argsort(valid_predictions, kind="mergesort")
        ]
        ranked[ranked_positions] = (
            np.arange(1, ranked_positions.size + 1, dtype=np.float64)
            / ranked_positions.size
        )

    return pd.Series(ranked, index=index, dtype=float)


def neutralize_by_era(
    predictions: pd.Series | Sequence[float] | np.ndarray,
    eras: pd.Series | Sequence[str] | np.ndarray,
    features: pd.DataFrame,
    proportion: float = 0.5,
    rank_output: bool = True,
    show_progress: bool = False,
) -> pd.Series:
    """Apply linear feature neutralization per era with optional rank-normalized output."""
    if not 0.0 <= proportion <= 1.0:
        raise ValueError("proportion must be between 0.0 and 1.0.")

    index = features.index
    prediction_values = _to_float_array(predictions, index)
    era_values = _to_era_array(eras, index)
    feature_values = features.to_numpy(dtype=np.float64, copy=False)

    neutralized = np.empty(len(index), dtype=np.float64)
    order, starts, stops = _sorted_era_slices(era_values)

    for start, stop in _progress_iter(
        zip(starts, stops),
        enabled=show_progress,
        total=len(starts),
        desc="Neutralizing eras",
    ):
        era_positions = order[start:stop]
        era_predictions = prediction_values[era_positions]
        centered_predictions = era_predictions - era_predictions.mean()

        era_features = feature_values[era_positions]
        if era_features.shape[1] == 0:
            neutralized[era_positions] = centered_predictions
            continue

        centered_features = era_features - era_features.mean(axis=0, keepdims=True)
        valid_columns = np.ptp(centered_features, axis=0) > 0.0

        if not np.any(valid_columns):
            neutralized[era_positions] = centered_predictions
            continue

        stable_features = centered_features[:, valid_columns]
        coefficients, _, _, _ = np.linalg.lstsq(
            stable_features,
            centered_predictions,
            rcond=1e-6,
        )
        correction = stable_features @ coefficients
        neutralized[era_positions] = centered_predictions - (proportion * correction)

    neutralized_series = pd.Series(neutralized, index=index, dtype=float)

    if rank_output:
        return rank_by_era(
            neutralized_series,
            era_values,
            index=index,
            show_progress=show_progress,
        )
    return neutralized_series


def average_predictions(
    prediction_map: Mapping[str, pd.Series | Sequence[float] | np.ndarray],
) -> pd.Series:
    """Return the row-wise average across named prediction vectors."""
    if not prediction_map:
        raise ValueError("prediction_map must contain at least one prediction series.")

    prediction_frame = pd.DataFrame(prediction_map)
    return prediction_frame.mean(axis=1)


def build_evaluation_frame(
    validation: pd.DataFrame,
    predictions: pd.Series | Sequence[float] | np.ndarray,
    target_col: str,
) -> pd.DataFrame:
    """Create the evaluation dataframe expected by utils.metrics.calculate_metrics."""
    if target_col not in validation.columns:
        raise KeyError(
            f"target_col '{target_col}' was not found in validation dataframe."
        )
    if "era" not in validation.columns:
        raise KeyError("validation dataframe must include an 'era' column.")

    prediction_series = _to_series(predictions, validation.index)
    evaluation_frame = validation.copy()
    evaluation_frame["prediction"] = prediction_series
    return evaluation_frame
