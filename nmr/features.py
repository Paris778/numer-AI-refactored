"""Feature-set resolution and stability screening for research campaigns.

Pure functions over ``features.json`` and the train frame; no model logic and
no file state beyond the explicit ``features_json`` argument. Subset
derivation adds no inputs to the run_id fingerprint beyond the config itself:
the fingerprint is fully determined by config (including
``data.feature_subset``) + data_version + ``nmr/*.py`` + environment.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
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
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top-level JSON must be an object containing 'feature_sets'"
        )
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


def _era_ic_pair(
    part: pl.DataFrame,
    feature_list: Sequence[str],
    target_col: str,
    era_col: str,
    *,
    spearman: bool,
) -> tuple[str, np.ndarray, np.ndarray, bool]:
    """Per-era (era, Pearson IC vector, Spearman IC vector, degenerate flag).

    The single implementation of per-era feature-target IC. Degenerate eras
    (fewer than 2 usable rows, an all-non-finite target, or a constant target)
    return zero vectors for both metrics with ``degenerate = True``. NaN is
    not null in polars, so non-finite target rows are excluded explicitly;
    rows are complete-case on features via ``drop_nulls``.
    """
    if era_col not in part.columns:
        raise ValueError(f"chunk missing required column {era_col!r}")
    era = str(part.get_column(era_col).to_list()[0])
    zeros = np.zeros(len(feature_list), dtype=float)
    clean = part.drop_nulls()
    if clean.height < 2:
        return era, zeros, zeros, True
    target_all = clean.get_column(target_col).cast(pl.Float64).to_numpy()
    finite = np.isfinite(target_all)
    target = target_all[finite]
    if target.size == 0 or bool(np.all(target == target[0])):
        return era, zeros, zeros, True
    features = clean.select(feature_list).cast(pl.Float64).to_numpy()[finite]
    pearson = _feature_target_pearson(features, target)
    if not spearman:
        return era, pearson, zeros, False
    from nmr import _gpu  # lazy: keeps this module's import graph acyclic

    spearman_vec = _feature_target_pearson(
        _gpu.rankdata(features, axis=0), _gpu.rankdata(target)
    )
    return era, pearson, spearman_vec, False


def _per_era_pearson_chunks(
    chunks: Iterable[pl.DataFrame],
    feature_cols: Sequence[str],
    target_col: str,
    era_col: str,
) -> tuple[dict[str, np.ndarray], set[str]]:
    """Per-era Pearson CORR of each feature vs ``target_col`` from era chunks.

    Each chunk is one era. Returns ``(corrs_by_era, degenerate_eras)``. A
    degenerate era (fewer than 2 usable rows, an all-non-finite target, or a
    constant target) contributes a zero vector and its label is recorded in
    ``degenerate_eras``.
    This is the single implementation of per-era feature-target correlation:
    the frame-based ``_per_era_pearson``, ``feature_stability_screen``, and
    ``nmr.analysis`` all route through it.
    """
    feature_list = list(feature_cols)
    per_era: dict[str, np.ndarray] = {}
    degenerate: set[str] = set()
    for part in chunks:
        era, pearson, _, is_degenerate = _era_ic_pair(
            part, feature_list, target_col, era_col, spearman=False
        )
        per_era[era] = pearson
        if is_degenerate:
            degenerate.add(era)
    return per_era, degenerate


def _per_era_pearson(
    frame: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    era_col: str,
) -> tuple[dict[str, np.ndarray], set[str]]:
    """Frame-based convenience wrapper over :func:`_per_era_pearson_chunks`."""
    feature_list = list(feature_cols)
    return _per_era_pearson_chunks(
        frame.select([era_col, target_col, *feature_list]).partition_by(
            era_col, maintain_order=True
        ),
        feature_list,
        target_col,
        era_col,
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
    degenerate eras (zero variance, <2 usable rows, non-finite values) are
    excluded from every aggregate — they carry no signal information, and
    padding them with 0.0 would bias the mean, std, and decay slope. ``n_eras``
    therefore counts valid eras only. Aggregates across eras: ``mean_corr``
    (mean), ``corr_std`` (population std), ``decay_slope`` (linear slope of
    CORR vs era index), ``cross_regime_variance`` (variance of first-half vs
    second-half era-window mean CORR — a regime-drift proxy). ``stable`` is
    True when ``mean_corr >= min_mean_corr`` and ``|decay_slope| <=
    max_abs_decay`` and ``n_eras >= 2``. When no valid eras exist, numeric
    aggregates are None and ``stable`` is False.
    """
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")
    required = {era_col, target_col, *feature_list}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing required columns: {sorted(missing)}")

    per_era, degenerate = _per_era_pearson(frame, feature_list, target_col, era_col)
    return _aggregate_screen(
        per_era, degenerate, feature_list, min_mean_corr, max_abs_decay
    )


def _aggregate_screen(
    per_era: dict[str, np.ndarray],
    degenerate: set[str],
    feature_list: Sequence[str],
    min_mean_corr: float,
    max_abs_decay: float,
) -> pl.DataFrame:
    """Aggregate per-era CORR vectors into the screen's summary columns.

    Shared by the frame-based :func:`feature_stability_screen` and the
    chunk-based screen path used by ``nmr.analysis``. Degenerate eras (target
    all-NaN, <2 usable rows, or zero-variance target) are excluded from all
    aggregates — see :func:`feature_stability_screen` for column semantics.
    """
    if not per_era:
        return pl.DataFrame(
            {name: [] for name in _SCREEN_COLUMNS}
        )

    eras = [e for e in sorted(per_era, key=int) if e not in degenerate]
    if not eras:
        return pl.DataFrame(
            [
                {
                    "feature": feature,
                    "mean_corr": None,
                    "corr_std": None,
                    "decay_slope": None,
                    "cross_regime_variance": None,
                    "n_eras": 0,
                    "stable": False,
                }
                for feature in feature_list
            ],
            schema=_SCREEN_COLUMNS,
        )
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
    required = {"feature", "mean_corr", "decay_slope", "n_eras"}
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
