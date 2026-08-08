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


def target_profile(
    frame: pl.DataFrame,
    target_cols: Sequence[str],
    era_col: str = "era",
) -> pl.DataFrame:
    """Per-target distribution/availability statistics.

    Non-finite target values are dropped before moments; ``missing_rate`` is
    the fraction of non-finite values over all rows; per-era means are
    computed over eras with at least one valid value.
    """
    target_list = list(target_cols)
    if not target_list:
        raise ValueError("target_cols must contain at least one target")
    n_total = frame.height
    era_values: dict[str, list[np.ndarray]] = {t: [] for t in target_list}
    pooled: dict[str, list[np.ndarray]] = {t: [] for t in target_list}
    zero_var_eras: dict[str, int] = {t: 0 for t in target_list}
    present_eras: dict[str, int] = {t: 0 for t in target_list}
    n_finite: dict[str, int] = {t: 0 for t in target_list}

    for part in frame.select([era_col, *target_list]).partition_by(
        era_col, maintain_order=True
    ):
        for t in target_list:
            values = part.get_column(t).cast(pl.Float64).to_numpy()
            finite = values[np.isfinite(values)]
            n_finite[t] += int(finite.size)
            if finite.size > 0:
                present_eras[t] += 1
                era_values[t].append(finite)
                if finite.size >= 2 and np.all(finite == finite[0]):
                    zero_var_eras[t] += 1
            pooled[t].append(finite)

    rows = []
    for t in target_list:
        pooled_arr = np.concatenate(pooled[t]) if pooled[t] else np.array([])
        if pooled_arr.size == 0:
            rows.append(
                {
                    "target": t,
                    "n_eras_present": present_eras[t],
                    "missing_rate": 1.0,
                    "era_mean_mean": None,
                    "era_mean_std": None,
                    "pooled_mean": None,
                    "pooled_std": None,
                    "pooled_skew": None,
                    "pooled_kurtosis": None,
                    "min": None,
                    "max": None,
                    "zero_variance_era_count": 0,
                }
            )
            continue
        era_means = np.array([float(np.mean(v)) for v in era_values[t]])
        mu = float(np.mean(pooled_arr))
        sd = float(np.std(pooled_arr, ddof=0))
        skew = float(scipy.stats.skew(pooled_arr)) if sd > 0 else 0.0
        kurt = float(scipy.stats.kurtosis(pooled_arr, fisher=True)) if sd > 0 else 0.0
        rows.append(
            {
                "target": t,
                "n_eras_present": present_eras[t],
                "missing_rate": 1.0 - n_finite[t] / n_total,
                "era_mean_mean": float(np.mean(era_means)),
                "era_mean_std": float(np.std(era_means, ddof=0)),
                "pooled_mean": mu,
                "pooled_std": sd,
                "pooled_skew": skew,
                "pooled_kurtosis": kurt,
                "min": float(np.min(pooled_arr)),
                "max": float(np.max(pooled_arr)),
                "zero_variance_era_count": zero_var_eras[t],
            }
        )
    return pl.DataFrame(rows)


def target_correlation_matrix(
    frame: pl.DataFrame,
    target_cols: Sequence[str],
    era_col: str = "era",
) -> pl.DataFrame:
    """Equal-era-weighted mean Spearman correlation between target pairs.

    An era is skipped for a pair when either target has <2 valid values or
    zero variance; ``n_eras`` records how many eras contributed.
    """
    target_list = list(target_cols)
    if len(target_list) < 2:
        raise ValueError("target_correlation_matrix needs at least two targets")
    pairs: dict[tuple[str, str], list[float]] = {}
    for i in range(len(target_list)):
        for j in range(i + 1, len(target_list)):
            pairs[(target_list[i], target_list[j])] = []

    for part in frame.select([era_col, *target_list]).partition_by(
        era_col, maintain_order=True
    ):
        for (a, b), values in pairs.items():
            av = part.get_column(a).cast(pl.Float64).to_numpy()
            bv = part.get_column(b).cast(pl.Float64).to_numpy()
            mask = np.isfinite(av) & np.isfinite(bv)
            avc, bvc = av[mask], bv[mask]
            if avc.size < 2 or np.std(avc) == 0.0 or np.std(bvc) == 0.0:
                continue
            ra = scipy.stats.rankdata(avc)
            rb = scipy.stats.rankdata(bvc)
            values.append(float(np.corrcoef(ra, rb)[0, 1]))

    rows = [
        {
            "target_a": a,
            "target_b": b,
            "mean_corr": float(np.mean(vals)) if vals else None,
            "n_eras": len(vals),
        }
        for (a, b), vals in pairs.items()
    ]
    return pl.DataFrame(rows)
