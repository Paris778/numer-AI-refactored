"""Feature-set resolution and stability screening for research campaigns.

Pure functions over ``features.json`` and the train frame; no model logic and
no file state beyond the explicit ``features_json`` argument. Derived subsets
must remain pure functions of their inputs so the run_id fingerprint (config +
data_version + ``nmr/*.py`` + env) is unchanged by subset selection.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

__all__ = [
    "resolve_feature_sets",
    "feature_stability_screen",
    "select_stable_features",
]


def resolve_feature_sets(features_json: Path) -> dict[str, list[str]]:
    """Return every named feature set in ``features.json``, deterministically ordered.

    Includes the canonical sets (small/medium/all) and the obfuscated family
    sets (intelligence, charisma, sunshine, ...) exactly as declared. Pure
    function of the file contents; values are defensive copies.
    """
    path = Path(features_json)
    raw = json.loads(path.read_text(encoding="utf-8"))
    sets = raw.get("feature_sets")
    if not isinstance(sets, dict) or not sets:
        raise ValueError(f"{path}: 'feature_sets' must be a non-empty mapping")
    result: dict[str, list[str]] = {}
    for name, values in sorted(sets.items()):
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(
                f"{path}: feature set {name!r} must be a list of strings"
            )
        result[name] = list(values)
    return result


DEFAULT_MIN_MEAN_CORR = 0.01
DEFAULT_MAX_ABS_DECAY = 0.001

_SCREEN_COLUMNS = (
    "feature", "mean_corr", "corr_std", "decay_slope",
    "cross_regime_variance", "n_eras", "stable",
)


def feature_stability_screen(
    frame: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    era_col: str = "era",
    min_mean_corr: float = DEFAULT_MIN_MEAN_CORR,
    max_abs_decay: float = DEFAULT_MAX_ABS_DECAY,
) -> pl.DataFrame:
    """Per-feature era-window CORR, decay, and cross-regime drift statistics.

    Definition (ARCHITECTURE.md §P): per-era Pearson CORR(feature, target)
    using the same vectorized per-era pattern as ``feature_exposure_report``;
    degenerate eras (zero variance, <2 usable rows, non-finite values)
    contribute 0.0. Aggregates across eras: ``mean_corr`` (mean), ``corr_std``
    (population std), ``decay_slope`` (linear slope of CORR vs era index),
    ``cross_regime_variance`` (variance of first-half vs second-half era-window
    mean CORR — a regime-drift proxy). ``stable`` is True when
    ``mean_corr >= min_mean_corr`` and ``|decay_slope| <= max_abs_decay`` and
    ``n_eras >= 2``.
    """
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")
    required = {era_col, target_col, *feature_list}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing required columns: {sorted(missing)}")

    per_era: dict[str, np.ndarray] = {}
    for part in frame.select([era_col, target_col, *feature_list]).partition_by(
        era_col, maintain_order=True
    ):
        era = str(part.get_column(era_col).to_list()[0])
        clean = part.drop_nulls()
        if clean.height < 2:
            per_era[era] = np.zeros(len(feature_list), dtype=float)
            continue
        target = clean.get_column(target_col).cast(pl.Float64).to_numpy()
        features = clean.select(feature_list).cast(pl.Float64).to_numpy()
        per_era[era] = _feature_target_pearson(features, target)

    if not per_era:
        return pl.DataFrame(
            {name: [] for name in _SCREEN_COLUMNS}
        )

    eras = sorted(per_era, key=int)
    matrix = np.column_stack([per_era[era] for era in eras])
    rows = []
    for i, feature in enumerate(feature_list):
        series = matrix[i]
        era_index = np.arange(len(eras), dtype=float)
        slope = (
            float(np.polyfit(era_index, series, 1)[0]) if len(series) >= 2 else 0.0
        )
        mid = len(series) // 2
        first = float(np.mean(series[:mid])) if mid > 0 else 0.0
        second = float(np.mean(series[mid:])) if len(series) - mid > 0 else 0.0
        cross_regime = 0.25 * (first - second) ** 2
        mean_corr = float(np.mean(series))
        stable = (
            mean_corr >= min_mean_corr
            and abs(slope) <= max_abs_decay
            and len(series) >= 2
        )
        rows.append(
            {
                "feature": feature,
                "mean_corr": mean_corr,
                "corr_std": float(np.std(series, ddof=0)),
                "decay_slope": slope,
                "cross_regime_variance": cross_regime,
                "n_eras": int(len(series)),
                "stable": stable,
            }
        )
    return pl.DataFrame(rows, schema=_SCREEN_COLUMNS)


def select_stable_features(
    screen: pl.DataFrame,
    *,
    min_mean_corr: float,
    max_abs_decay: float,
) -> list[str]:
    """Return the sorted stable feature names passing both thresholds."""
    required = {"feature", "mean_corr", "decay_slope", "stable", "n_eras"}
    missing = required - set(screen.columns)
    if missing:
        raise ValueError(f"screen missing required columns: {sorted(missing)}")
    kept = screen.filter(
        (pl.col("mean_corr") >= min_mean_corr)
        & (pl.col("decay_slope").abs() <= max_abs_decay)
        & (pl.col("n_eras") >= 2)
    )
    return sorted(kept.get_column("feature").to_list())


def _feature_target_pearson(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_centered = target - np.mean(target)
    target_norm = float(np.linalg.norm(target_centered))
    if target_norm == 0.0:
        return np.zeros(features.shape[1], dtype=float)
    feature_centered = features - np.mean(features, axis=0)
    denoms = np.linalg.norm(feature_centered, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = (feature_centered.T @ target_centered) / (denoms * target_norm)
    return np.where(np.isfinite(corrs), corrs, 0.0)
