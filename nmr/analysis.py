"""Deterministic dataset analysis for research reports.

Era-aware statistics over train/validation frames: split shapes, era
structure, target profiles, feature-target IC, feature moments, feature
correlation structure, regimes, and benchmark context. Pure functions: frames
in, frames/dicts out — no I/O, no wall-clock, no stochastic operations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.stats

__all__ = [
    "SplitStats",
    "describe_splits",
    "era_structure",
    "target_profile",
    "target_correlation_matrix",
    "feature_ic_screen",
    "feature_ic_by_era",
    "feature_summary",
    "FeatureCorrResult",
    "feature_correlation_structure",
    "within_set_redundancy",
    "cross_set_membership",
    "regime_analysis",
    "benchmark_era_corr",
]

REGIME_LOW_PCT = 10.0
REGIME_HIGH_PCT = 90.0
IC_VOL_WINDOW = 20


@dataclass(frozen=True)
class SplitStats:
    """Shape statistics for one dataset split."""

    n_rows: int
    n_eras: int
    min_era: str
    max_era: str
    rows_per_era_min: int
    rows_per_era_median: float
    rows_per_era_max: int
    rows_per_era_mean: float
    rows_per_era_std: float
    n_ids: int


def describe_splits(splits: Mapping[str, pl.DataFrame]) -> dict[str, SplitStats]:
    """Per-split shape statistics. Requires an ``id`` column in each frame."""
    out: dict[str, SplitStats] = {}
    for name, frame in splits.items():
        if "id" not in frame.columns:
            raise ValueError(f"split {name!r} missing required column 'id'")
        per_era = frame.group_by("era").len()
        counts = per_era.get_column("len").to_numpy()
        eras = sorted(per_era.get_column("era").to_list(), key=int)
        out[name] = SplitStats(
            n_rows=frame.height,
            n_eras=len(eras),
            min_era=eras[0],
            max_era=eras[-1],
            rows_per_era_min=int(counts.min()),
            rows_per_era_median=float(np.median(counts)),
            rows_per_era_max=int(counts.max()),
            rows_per_era_mean=float(counts.mean()),
            rows_per_era_std=float(counts.std(ddof=0)),
            n_ids=int(frame.get_column("id").n_unique()),
        )
    return out


def era_structure(frame: pl.DataFrame, era_col: str = "era") -> pl.DataFrame:
    """Per-era row/id counts with era-index gap detection (sorted by int era)."""
    if era_col not in frame.columns:
        raise ValueError(f"frame missing required column {era_col!r}")
    if frame.is_empty():
        raise ValueError("frame is empty: cannot compute era structure")
    n_ids = (
        pl.col("id").n_unique().alias("n_ids")
        if "id" in frame.columns
        else pl.lit(None, dtype=pl.Int64).alias("n_ids")
    )
    per = frame.group_by(era_col).agg(pl.len().alias("n_rows"), n_ids)
    per = per.with_columns(
        pl.col(era_col).cast(pl.Int64).alias("era_index")
    ).sort("era_index")
    gap = (
        per.select(
            (pl.col("era_index") != pl.col("era_index").shift(1) + 1)
            .fill_null(False)
            .alias("gap")
        ).get_column("gap")
    )
    return per.select(
        pl.col(era_col).alias("era"), "era_index", "n_rows", "n_ids", gap
    )
