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


def feature_ic_by_era(
    frame: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    era_col: str = "era",
) -> pl.DataFrame:
    """Per-era per-feature IC long-form, via ``_per_era_pearson``.

    Degenerate eras (per the screen convention) carry ``ic = 0.0`` and
    ``degenerate = True``; all other rows carry ``degenerate = False``.
    """
    from nmr.features import _per_era_pearson

    feature_list = list(feature_cols)
    corrs, degenerate = _per_era_pearson(frame, feature_list, target_col, era_col)
    rows = [
        {
            "era": era,
            "feature": feature,
            "ic": float(vec[i]),
            "degenerate": era in degenerate,
        }
        for era, vec in corrs.items()
        for i, feature in enumerate(feature_list)
    ]
    return pl.DataFrame(
        rows,
        schema={
            "era": pl.Utf8,
            "feature": pl.Utf8,
            "ic": pl.Float64,
            "degenerate": pl.Boolean,
        },
    )


def feature_ic_screen(
    frame: pl.DataFrame,
    feature_cols: Sequence[str],
    targets: Sequence[str],
    era_col: str = "era",
) -> pl.DataFrame:
    """Aggregated feature-target screen, one block per reference target.

    Thin wrapper over ``feature_stability_screen`` (the single screen
    implementation) that tags each block with its target.
    """
    from nmr.features import feature_stability_screen

    if not targets:
        raise ValueError("targets must contain at least one target column")
    blocks = [
        feature_stability_screen(
            frame, feature_cols=feature_cols, target_col=t, era_col=era_col
        ).with_columns(pl.lit(t).alias("target"))
        for t in targets
    ]
    return pl.concat(blocks).select(
        [
            "feature",
            "target",
            "mean_corr",
            "corr_std",
            "decay_slope",
            "cross_regime_variance",
            "n_eras",
            "stable",
        ]
    )


def _chunk_moments(values: np.ndarray) -> tuple[float, float, float, float, float]:
    """(n, mean, M2, M3, M4) over a finite 1-D array (raw central moment sums)."""
    n = values.size
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    mean = float(np.mean(values))
    centered = values - mean
    M2 = float(np.sum(centered**2))
    M3 = float(np.sum(centered**3))
    M4 = float(np.sum(centered**4))
    return (float(n), mean, M2, M3, M4)


def _combine(
    a: tuple[float, float, float, float, float],
    b: tuple[float, float, float, float, float],
) -> tuple[float, float, float, float, float]:
    """Terriberry parallel combine of (n, mean, M2, M3, M4) moments.

    ``M2/M3/M4`` are raw central-moment sums; ``mean`` is the arithmetic mean.
    """
    n1, mean_a, M2_a, M3_a, M4_a = a
    n2, mean_b, M2_b, M3_b, M4_b = b
    n = n1 + n2
    if n == 0.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    delta = mean_b - mean_a
    mean = mean_a + delta * n2 / n
    M2 = M2_a + M2_b + delta * delta * n1 * n2 / n
    M3 = (
        M3_a
        + M3_b
        + delta * delta * delta * n1 * n2 * (n1 - n2) / (n * n)
        + 3.0 * delta * (n1 * M2_b - n2 * M2_a) / n
    )
    M4 = (
        M4_a
        + M4_b
        + delta**4 * n1 * n2 * (n1 * n1 - n1 * n2 + n2 * n2) / (n**3)
        + 6.0 * delta * delta * (n1 * n1 * M2_b + n2 * n2 * M2_a) / (n * n)
        + 4.0 * delta * (n1 * M3_b - n2 * M3_a) / n
    )
    return (n, mean, M2, M3, M4)


def feature_summary(
    chunks: Iterable[pl.DataFrame],
    feature_cols: Sequence[str],
    era_col: str = "era",
) -> pl.DataFrame:
    """Per-feature pooled moments via streaming Welford + Terriberry.

    Caller drives chunking (era-sorted ascending). Non-finite values are
    dropped before moments; ``missing_rate = 1 - n_finite / n_total``.
    """
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")
    acc = {
        f: [0.0, 0.0, 0.0, 0.0, 0.0, np.inf, -np.inf, 0.0]
        for f in feature_list
    }  # n, mean, M2, M3, M4, min, max, n_finite
    n_total = 0
    for chunk in chunks:
        if era_col not in chunk.columns:
            raise ValueError(f"chunk missing required column {era_col!r}")
        missing = set(feature_list) - set(chunk.columns)
        if missing:
            raise ValueError(f"chunk missing feature columns: {sorted(missing)}")
        n_total += chunk.height
        for f in feature_list:
            values = chunk.get_column(f).cast(pl.Float64).to_numpy()
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            state = acc[f]
            combined = _combine(tuple(state[:5]), _chunk_moments(finite))
            state[:5] = list(combined)
            state[5] = min(state[5], float(np.min(finite)))
            state[6] = max(state[6], float(np.max(finite)))
            state[7] += float(finite.size)

    rows = []
    for f in feature_list:
        n, mean, M2, M3, M4, cmin, cmax, n_finite = acc[f]
        if n == 0.0:
            rows.append(
                {
                    "feature": f,
                    "pooled_mean": None,
                    "pooled_std": None,
                    "pooled_skew": None,
                    "pooled_kurtosis": None,
                    "min": None,
                    "max": None,
                    "missing_rate": 1.0,
                }
            )
            continue
        std = float(np.sqrt(M2 / n)) if M2 > 0 else 0.0
        skew = float((M3 / n) / ((M2 / n) ** 1.5)) if M2 > 0 else 0.0
        kurt = float((M4 / n) / ((M2 / n) ** 2) - 3.0) if M2 > 0 else 0.0
        rows.append(
            {
                "feature": f,
                "pooled_mean": mean,
                "pooled_std": std,
                "pooled_skew": skew,
                "pooled_kurtosis": kurt,
                "min": cmin,
                "max": cmax,
                "missing_rate": 1.0 - n_finite / n_total,
            }
        )
    return pl.DataFrame(rows)


@dataclass(frozen=True)
class FeatureCorrResult:
    """Era-averaged feature correlation structure."""

    matrix: np.ndarray  # float32 (N, N) symmetric
    feature_order: tuple[str, ...]
    top_pairs: pl.DataFrame
    summary: dict


def _rank_gaussianize_chunk(
    chunk: pl.DataFrame,
    feature_list: Sequence[str],
    era_col: str,
) -> np.ndarray | None:
    """Complete-case per-era rank-gaussianized feature matrix, or None."""
    clean = chunk.select([era_col, *feature_list]).drop_nulls()
    if clean.height < 2:
        return None
    out = np.empty((clean.height, len(feature_list)), dtype=np.float64)
    for j, feature in enumerate(feature_list):
        col = clean.get_column(feature).cast(pl.Float64).to_numpy()
        ranks = scipy.stats.rankdata(col, method="average")
        out[:, j] = scipy.stats.norm.ppf(ranks / (col.size + 1))
    return out


def feature_correlation_structure(
    chunks: Iterable[pl.DataFrame],
    feature_cols: Sequence[str],
    era_col: str = "era",
) -> FeatureCorrResult:
    """Equal-era-weighted mean feature correlation matrix.

    Per era: complete-case rows only, rank-gaussianized per feature, then the
    full correlation matrix; matrices are summed and divided by the era count.
    Degenerate columns (zero variance) contribute 0.0.
    """
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")
    n = len(feature_list)
    acc = np.zeros((n, n), dtype=np.float64)
    n_eras = 0
    for chunk in chunks:
        gauss = _rank_gaussianize_chunk(chunk, feature_list, era_col)
        if gauss is None:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = np.corrcoef(gauss, rowvar=False)
        mat = np.where(np.isfinite(mat), mat, 0.0)
        acc += mat
        n_eras += 1
    if n_eras == 0:
        raise ValueError("no usable eras in feature_correlation_structure input")
    mean_mat = (acc / n_eras).astype(np.float32)

    iu = np.triu_indices(n, k=1)
    abs_vals = np.abs(mean_mat[iu])
    order = np.argsort(abs_vals)[::-1][:100]
    top_rows = [
        {
            "feature_a": feature_list[iu[0][k]],
            "feature_b": feature_list[iu[1][k]],
            "mean_corr": float(mean_mat[iu[0][k], iu[1][k]]),
        }
        for k in order
    ]
    summary = {
        "mean_abs_corr": float(abs_vals.mean()) if abs_vals.size else 0.0,
        "p50_abs_corr": float(np.percentile(abs_vals, 50)) if abs_vals.size else 0.0,
        "p90_abs_corr": float(np.percentile(abs_vals, 90)) if abs_vals.size else 0.0,
        "n_pairs": int(abs_vals.size),
    }
    return FeatureCorrResult(
        matrix=mean_mat,
        feature_order=tuple(feature_list),
        top_pairs=pl.DataFrame(
            top_rows,
            schema={
                "feature_a": pl.Utf8,
                "feature_b": pl.Utf8,
                "mean_corr": pl.Float64,
            },
        ),
        summary=summary,
    )


def within_set_redundancy(
    result: FeatureCorrResult,
    sets: Mapping[str, Sequence[str]],
) -> pl.DataFrame:
    """Per-feature-set pairwise |corr| summary, indexed from the full matrix."""
    index = {f: i for i, f in enumerate(result.feature_order)}
    rows = []
    for name in sorted(sets):
        members = [f for f in sets[name] if f in index]
        if len(members) < 2:
            rows.append(
                {
                    "feature_set": name,
                    "n_features": len(members),
                    "mean_abs_corr": None,
                    "median_abs_corr": None,
                    "max_abs_corr": None,
                    "n_pairs": 0,
                }
            )
            continue
        idx = [index[f] for f in members]
        sub = result.matrix[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        abs_vals = np.abs(sub[iu])
        rows.append(
            {
                "feature_set": name,
                "n_features": len(members),
                "mean_abs_corr": float(abs_vals.mean()),
                "median_abs_corr": float(np.median(abs_vals)),
                "max_abs_corr": float(abs_vals.max()),
                "n_pairs": int(abs_vals.size),
            }
        )
    return pl.DataFrame(rows)


def cross_set_membership(sets: Mapping[str, Sequence[str]]) -> dict:
    """Set sizes and pairwise empirical subset relations."""
    names = sorted(sets)
    set_rows = [
        {"feature_set": name, "n_features": len(set(sets[name]))} for name in names
    ]
    rel_rows = []
    for a in names:
        for b in names:
            if a == b:
                continue
            rel_rows.append(
                {
                    "a": a,
                    "b": b,
                    "a_subset_of_b": set(sets[a]).issubset(set(sets[b])),
                }
            )
    return {
        "sets": pl.DataFrame(set_rows),
        "subset_relations": pl.DataFrame(rel_rows),
    }


def regime_analysis(ic_by_era: pl.DataFrame) -> dict:
    """Deterministic, percentile-based regime analysis of per-era feature IC.

    Crash/hot use decile thresholds (``REGIME_LOW_PCT`` / ``REGIME_HIGH_PCT``);
    the regime column uses quartile bands. ``ic_persistence`` is the mean
    adjacent-era Spearman rank correlation of per-era feature IC vectors.
    """
    required = {"era", "feature", "ic"}
    missing = required - set(ic_by_era.columns)
    if missing:
        raise ValueError(f"ic_by_era missing required columns: {sorted(missing)}")

    sig = (
        ic_by_era.group_by("era")
        .agg(
            pl.col("ic").mean().alias("mean_ic"),
            pl.col("ic").std().alias("ic_std"),
            pl.col("feature").count().alias("n_features"),
        )
        .sort("era")
    )
    mean_ics = sig.get_column("mean_ic").to_numpy()
    n = len(mean_ics)
    ranks = np.argsort(np.argsort(mean_ics))
    pct = 100.0 * ranks / (n - 1) if n > 1 else np.array([50.0])

    q1 = float(np.percentile(mean_ics, 25.0))
    q3 = float(np.percentile(mean_ics, 75.0))
    low_thr = float(np.percentile(mean_ics, REGIME_LOW_PCT))
    high_thr = float(np.percentile(mean_ics, REGIME_HIGH_PCT))

    regime = np.where(pct <= 25.0, "low", np.where(pct >= 75.0, "high", "normal"))
    crash = pct <= REGIME_LOW_PCT
    hot = pct >= REGIME_HIGH_PCT
    sig = sig.with_columns(
        pl.Series("pct_rank", pct),
        pl.Series("regime", regime),
        pl.Series("crash", crash),
        pl.Series("hot", hot),
    )

    eras = sig.get_column("era").to_list()
    crash_eras = [e for e, c in zip(eras, crash) if c]
    hot_eras = [e for e, h in zip(eras, hot) if h]

    # adjacent-era IC-vector Spearman
    pivot = ic_by_era.pivot(on="feature", index="era", values="ic").sort("era")
    feature_names = [c for c in pivot.columns if c != "era"]
    matrix = pivot.select(feature_names).to_numpy()
    matrix = np.nan_to_num(matrix, nan=0.0)
    ranks_mat = np.apply_along_axis(scipy.stats.rankdata, 1, matrix)
    adj = [
        float(np.corrcoef(ranks_mat[t], ranks_mat[t - 1])[0, 1])
        for t in range(1, ranks_mat.shape[0])
    ]
    persistence = {
        "mean": float(np.mean(adj)) if adj else 0.0,
        "std": float(np.std(adj, ddof=0)) if adj else 0.0,
        "n_adjacent": len(adj),
    }

    rolling = sig.select(
        pl.col("era"),
        pl.col("mean_ic")
        .rolling_std(window_size=IC_VOL_WINDOW, min_samples=2)
        .alias("rolling_std"),
    )

    return {
        "regime_thresholds": {
            "low_pct": REGIME_LOW_PCT,
            "high_pct": REGIME_HIGH_PCT,
            "q1": q1,
            "q3": q3,
            "mean_ic_low": low_thr,
            "mean_ic_high": high_thr,
        },
        "era_signal": sig,
        "crash_eras": crash_eras,
        "hot_eras": hot_eras,
        "ic_persistence": persistence,
        "rolling_vol": rolling,
    }


def benchmark_era_corr(
    frame: pl.DataFrame,
    benchmark_cols: Sequence[str],
    target_col: str,
    era_col: str = "era",
) -> dict:
    """Per-era CORR of benchmark models vs target.

    Lightweight context for the report (floors/ceilings), distinct from the
    full ``BenchmarkSuite`` harness. Degenerate eras (fewer than 2 non-null
    rows or constant target) are silently absent — ``n_eras`` reflects the
    actual era overlap.
    """
    from nmr.features import _per_era_pearson

    benchmark_list = list(benchmark_cols)
    if not benchmark_list:
        raise ValueError("benchmark_cols must contain at least one benchmark")
    corrs, degenerate = _per_era_pearson(frame, benchmark_list, target_col, era_col)
    rows = [
        {"era": era, "benchmark": b, "corr": float(vec[i])}
        for era, vec in corrs.items()
        if era not in degenerate
        for i, b in enumerate(benchmark_list)
    ]
    per_era = pl.DataFrame(
        rows,
        schema={"era": pl.Utf8, "benchmark": pl.Utf8, "corr": pl.Float64},
    )
    summary = (
        per_era.group_by("benchmark")
        .agg(
            pl.col("corr").mean().alias("mean_corr"),
            pl.col("corr").std().alias("corr_std"),
            pl.col("era").count().alias("n_eras"),
            pl.col("era").min().alias("first_era"),
            pl.col("era").max().alias("last_era"),
        )
        .sort("benchmark")
    )
    return {"benchmarks": summary, "per_era": per_era}
