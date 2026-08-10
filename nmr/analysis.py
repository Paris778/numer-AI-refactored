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

from nmr.features import DEFAULT_MAX_ABS_DECAY, DEFAULT_MIN_MEAN_CORR

__all__ = [
    "SplitStats",
    "describe_splits",
    "era_structure",
    "target_profile",
    "target_correlation_matrix",
    "feature_ic_screen",
    "feature_ic_by_era",
    "feature_ic_by_split",
    "feature_summary",
    "feature_drift_psi",
    "feature_drift_profile",
    "meta_orthogonality",
    "FeatureCorrResult",
    "feature_correlation_structure",
    "within_set_redundancy",
    "cross_set_membership",
    "regime_analysis",
    "neutralized_ic_profile",
    "benchmark_era_corr",
]

REGIME_LOW_PCT = 10.0
REGIME_HIGH_PCT = 90.0
IC_VOL_WINDOW = 20
PSI_FLAG_THRESHOLD = 0.25
WASSERSTEIN_FLAG_THRESHOLD = 0.25
AUC_FLAG_DELTA = 0.1
_PSI_EPS = 1e-6
_PSI_EDGE_SAMPLE_STRIDE = 100


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
    chunks: Iterable[pl.DataFrame],
    feature_cols: Sequence[str],
    target_col: str,
    era_col: str = "era",
) -> pl.DataFrame:
    """Per-era per-feature IC long-form, via ``_era_ic_pair``.

    Each chunk must be one era (era-partitioned). Degenerate eras carry
    ``ic = 0.0``, ``spearman_ic = 0.0`` and ``degenerate = True``; all other
    rows ``False``.
    """
    from nmr.features import _era_ic_pair

    feature_list = list(feature_cols)
    rows = []
    for part in chunks:
        era, pearson, spearman, is_degenerate = _era_ic_pair(
            part, feature_list, target_col, era_col, spearman=True
        )
        for i, feature in enumerate(feature_list):
            rows.append(
                {
                    "era": era,
                    "feature": feature,
                    "ic": float(pearson[i]),
                    "spearman_ic": float(spearman[i]),
                    "degenerate": is_degenerate,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "era": pl.Utf8,
            "feature": pl.Utf8,
            "ic": pl.Float64,
            "spearman_ic": pl.Float64,
            "degenerate": pl.Boolean,
        },
    )


def feature_ic_by_split(
    ic_by_era: pl.DataFrame,
    train_max_era: int,
    val_min_era: int,
) -> pl.DataFrame:
    """Per-feature mean IC per split, plus the validation-minus-train delta.

    ``train_max_era`` / ``val_min_era`` are inclusive integer era bounds
    (v5.3: train 0001-0574, validation 0575-1231). Eras between the bounds
    and degenerate eras (all-zero IC vectors, or eras flagged by a
    ``degenerate`` column) are excluded from both sides. Output columns:
    feature, train_mean_ic, train_n_eras, val_mean_ic, val_n_eras, delta_ic.
    """
    required = {"era", "feature", "ic"}
    missing = required - set(ic_by_era.columns)
    if missing:
        raise ValueError(f"ic_by_era missing required columns: {sorted(missing)}")
    if train_max_era >= val_min_era:
        raise ValueError("train_max_era must be < val_min_era")

    if "degenerate" in ic_by_era.columns:
        flags = ic_by_era.group_by("era").agg(pl.col("degenerate").any())
        degenerate = set(flags.filter(pl.col("degenerate"))["era"].to_list())
    else:
        degenerate = set(
            ic_by_era.group_by("era")
            .agg(pl.col("ic").abs().max())
            .filter(pl.col("ic") == 0.0)["era"]
            .to_list()
        )

    clean = ic_by_era.filter(~pl.col("era").is_in(degenerate))
    train = clean.filter(pl.col("era").cast(pl.Int64) <= train_max_era)
    val = clean.filter(pl.col("era").cast(pl.Int64) >= val_min_era)
    t = train.group_by("feature").agg(
        pl.col("ic").mean().alias("train_mean_ic"),
        pl.col("era").n_unique().alias("train_n_eras"),
    )
    v = val.group_by("feature").agg(
        pl.col("ic").mean().alias("val_mean_ic"),
        pl.col("era").n_unique().alias("val_n_eras"),
    )
    return (
        t.join(v, on="feature", how="full", coalesce=True)
        .with_columns(
            (pl.col("val_mean_ic") - pl.col("train_mean_ic")).alias("delta_ic")
        )
        .sort("feature")
    )


def _nonlinear_flag(
    mean_corr: np.ndarray, mean_spearman: np.ndarray, threshold: float
) -> np.ndarray:
    """True when |Pearson| <= threshold but |Spearman| > threshold.

    Flags features whose signal is plausibly monotone-nonlinear — invisible
    to the linear Pearson screen but exploitable by rank-based models.
    Non-finite aggregates never flag.
    """
    mc = np.asarray(mean_corr, dtype=float)
    ms = np.asarray(mean_spearman, dtype=float)
    return (
        (np.abs(mc) <= threshold)
        & (np.abs(ms) > threshold)
        & np.isfinite(mc)
        & np.isfinite(ms)
    )


def feature_ic_screen(
    chunks: Iterable[pl.DataFrame],
    feature_cols: Sequence[str],
    targets: Sequence[str],
    era_col: str = "era",
    min_mean_corr: float = DEFAULT_MIN_MEAN_CORR,
    max_abs_decay: float = DEFAULT_MAX_ABS_DECAY,
) -> pl.DataFrame:
    """Aggregated feature-target screen, one block per reference target.

    Chunk-driven (each chunk is one era) so the full feature universe can be
    screened without materializing the whole frame; the aggregation math is
    the shared ``nmr.features._aggregate_screen`` (same as the frame-based
    ``feature_stability_screen``). Adds ``mean_spearman`` (mean per-era
    Spearman rank IC over valid eras) and ``nonlinear`` (|Pearson| <=
    ``min_mean_corr`` but |Spearman| > ``min_mean_corr``) as diagnostics.
    """
    from nmr.features import _aggregate_screen, _era_ic_pair

    if not targets:
        raise ValueError("targets must contain at least one target column")
    feature_list = list(feature_cols)
    blocks = []
    for t in targets:
        pearson_eras: dict[str, np.ndarray] = {}
        spearman_eras: dict[str, np.ndarray] = {}
        degenerate: set[str] = set()
        for part in chunks:
            era, pearson, spearman, is_degenerate = _era_ic_pair(
                part, feature_list, t, era_col, spearman=True
            )
            pearson_eras[era] = pearson
            spearman_eras[era] = spearman
            if is_degenerate:
                degenerate.add(era)
        screen = _aggregate_screen(
            pearson_eras, degenerate, feature_list, min_mean_corr, max_abs_decay
        )
        valid_eras = [
            e for e in sorted(spearman_eras, key=int) if e not in degenerate
        ]
        if valid_eras:
            mean_spearman = np.mean(
                np.column_stack([spearman_eras[e] for e in valid_eras]), axis=1
            )
        else:
            mean_spearman = np.full(len(feature_list), np.nan)
        nonlinear = _nonlinear_flag(
            screen.get_column("mean_corr").to_numpy(),
            mean_spearman,
            min_mean_corr,
        )
        blocks.append(
            screen.with_columns(
                pl.Series("mean_spearman", mean_spearman)
                .fill_nan(None)
                .alias("mean_spearman"),
                pl.Series("nonlinear", nonlinear).alias("nonlinear"),
            ).with_columns(pl.lit(t).alias("target"))
        )
    return pl.concat(blocks).select(
        [
            "feature",
            "target",
            "mean_corr",
            "corr_std",
            "decay_slope",
            "cross_regime_variance",
            "mean_spearman",
            "n_eras",
            "stable",
            "nonlinear",
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


def _collect_strided(
    chunks: Iterable[pl.DataFrame],
    feature_list: Sequence[str],
    era_col: str,
    stride: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """One deterministic strided finite-value sample per feature + full finite
    counts, from an era-chunk stream. Shared by the drift diagnostics."""
    parts: dict[str, list[np.ndarray]] = {f: [] for f in feature_list}
    counts: dict[str, int] = {f: 0 for f in feature_list}
    for chunk in chunks:
        if era_col not in chunk.columns:
            raise ValueError(f"chunk missing required column {era_col!r}")
        for f in feature_list:
            values = chunk.get_column(f).cast(pl.Float64).to_numpy()
            finite = values[np.isfinite(values)]
            counts[f] += int(finite.size)
            parts[f].append(finite[::stride])
    return (
        {
            f: np.concatenate(parts[f]) if parts[f] else np.array([])
            for f in feature_list
        },
        counts,
    )


def _psi_from_samples(
    train: np.ndarray, val: np.ndarray, n_bins: int, eps: float
) -> float | None:
    """PSI between two 1-D samples over train-quantile edges."""
    if train.size == 0 or val.size == 0:
        return None
    if np.ptp(train) == 0.0:
        return 0.0
    edges = np.quantile(train, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    hist_t = np.histogram(train, bins=edges)[0].astype(float) / train.size
    hist_v = np.histogram(val, bins=edges)[0].astype(float) / val.size
    pt = hist_t + eps
    pv = hist_v + eps
    return float(np.sum((pv - pt) * np.log(pv / pt)))


def _wasserstein_from_samples(train: np.ndarray, val: np.ndarray) -> float | None:
    """1-D Wasserstein W1 distance between two samples."""
    if train.size == 0 or val.size == 0:
        return None
    return float(scipy.stats.wasserstein_distance(train, val))


def _auc_from_samples(train: np.ndarray, val: np.ndarray) -> float | None:
    """Univariate adversarial-validation AUC (Mann-Whitney U): how well a
    single feature ranks validation rows above train rows (validation is the
    class to detect). 0.5 = no separation, ~1.0 = perfectly distinguishable."""
    if train.size == 0 or val.size == 0:
        return None
    from nmr import _gpu  # lazy: keeps this module's import graph acyclic

    combined = np.concatenate([train, val])
    ranks = _gpu.rankdata(combined)
    u = float(
        np.sum(ranks[train.size :]) - val.size * (val.size + 1) / 2.0
    )
    return u / (train.size * val.size)


def feature_drift_psi(
    train_chunks: Iterable[pl.DataFrame],
    val_chunks: Iterable[pl.DataFrame],
    feature_cols: Sequence[str],
    era_col: str = "era",
    n_bins: int = 10,
    flag_threshold: float = PSI_FLAG_THRESHOLD,
    edge_sample_stride: int = _PSI_EDGE_SAMPLE_STRIDE,
) -> pl.DataFrame:
    """Per-feature Population Stability Index (PSI) across the split boundary.

    Bin edges are the ``n_bins`` quantiles of the train values — computed on
    a deterministic strided sample (``edge_sample_stride``; ``1`` = exact,
    the production default ``_PSI_EDGE_SAMPLE_STRIDE = 100`` keeps the edge
    computation bounded on the full universe). Both streams are histogrammed
    over those edges (out-of-range values clip into the outer bins; counts
    are exact) and ``psi = sum (p_val - p_train) * ln(p_val / p_train)`` with
    ``_PSI_EPS`` smoothing. ``drifted`` is True when ``psi >
    flag_threshold``. Constant features report 0.0; features with no finite
    values on either side report None. ``n_train``/``n_val`` are full finite
    counts, not sample counts.
    """
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    if edge_sample_stride < 1:
        raise ValueError("edge_sample_stride must be >= 1")

    train, n_train = _collect_strided(
        train_chunks, feature_list, era_col, edge_sample_stride
    )
    val, n_val = _collect_strided(
        val_chunks, feature_list, era_col, edge_sample_stride
    )

    rows = []
    for f in feature_list:
        psi = _psi_from_samples(train[f], val[f], n_bins, _PSI_EPS)
        rows.append(
            {
                "feature": f,
                "psi": psi,
                "n_train": n_train[f],
                "n_val": n_val[f],
                "drifted": psi is not None and psi > flag_threshold,
            }
        )
    return pl.DataFrame(rows)


def feature_drift_profile(
    train_chunks: Iterable[pl.DataFrame],
    val_chunks: Iterable[pl.DataFrame],
    feature_cols: Sequence[str],
    era_col: str = "era",
    n_bins: int = 10,
    flag_threshold: float = PSI_FLAG_THRESHOLD,
    w1_threshold: float = WASSERSTEIN_FLAG_THRESHOLD,
    auc_delta: float = AUC_FLAG_DELTA,
    edge_sample_stride: int = _PSI_EDGE_SAMPLE_STRIDE,
) -> pl.DataFrame:
    """PSI + Wasserstein W1 + univariate adversarial AUC in one sample pass.

    One deterministic strided sample of each stream feeds all three metrics
    (identical sampling to ``feature_drift_psi``); ``n_train``/``n_val`` are
    full finite counts. ``drifted`` = psi > ``flag_threshold`` OR w1 >
    ``w1_threshold`` OR |auc_roc - 0.5| > ``auc_delta``. The AUC is the
    Mann-Whitney separation of train vs validation rows per feature
    (0.5 = no separation, ~1.0 = perfectly distinguishable).
    """
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    if edge_sample_stride < 1:
        raise ValueError("edge_sample_stride must be >= 1")

    train, n_train = _collect_strided(
        train_chunks, feature_list, era_col, edge_sample_stride
    )
    val, n_val = _collect_strided(
        val_chunks, feature_list, era_col, edge_sample_stride
    )

    rows = []
    for f in feature_list:
        t, v = train[f], val[f]
        psi = _psi_from_samples(t, v, n_bins, _PSI_EPS)
        w1 = _wasserstein_from_samples(t, v)
        auc = _auc_from_samples(t, v)
        drifted = (
            (psi is not None and psi > flag_threshold)
            or (w1 is not None and w1 > w1_threshold)
            or (auc is not None and abs(auc - 0.5) > auc_delta)
        )
        rows.append(
            {
                "feature": f,
                "psi": psi,
                "w1": w1,
                "auc_roc": auc,
                "n_train": n_train[f],
                "n_val": n_val[f],
                "drifted": drifted,
            }
        )
    return pl.DataFrame(rows)


def meta_orthogonality(
    chunks: Iterable[pl.DataFrame],
    feature_cols: Sequence[str],
    meta_col: str,
    target_col: str,
    era_col: str = "era",
    corr_threshold: float = DEFAULT_MIN_MEAN_CORR,
) -> pl.DataFrame:
    """Per-feature correlation vs the meta model and the target.

    Computed on the eras where the meta model exists (validation only in
    practice). ``orthogonal`` = |corr_meta| <= ``corr_threshold`` AND
    |corr_target| > ``corr_threshold`` — the feature carries signal the
    consensus does not already price in. Degenerate eras (constant meta,
    constant target, fewer than 2 usable rows) are skipped; ``n_eras``
    reflects valid eras.
    """
    from nmr.features import _feature_target_pearson

    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")
    meta_acc: dict[str, list[float]] = {f: [] for f in feature_list}
    target_acc: dict[str, list[float]] = {f: [] for f in feature_list}
    for part in chunks:
        clean = part.drop_nulls()
        if clean.height < 2:
            continue
        meta_all = clean.get_column(meta_col).cast(pl.Float64).to_numpy()
        target_all = clean.get_column(target_col).cast(pl.Float64).to_numpy()
        mask = np.isfinite(meta_all) & np.isfinite(target_all)
        meta = meta_all[mask]
        target = target_all[mask]
        if (
            meta.size < 2
            or bool(np.all(meta == meta[0]))
            or bool(np.all(target == target[0]))
        ):
            continue
        features = clean.select(feature_list).cast(pl.Float64).to_numpy()[mask]
        meta_vec = _feature_target_pearson(features, meta)
        target_vec = _feature_target_pearson(features, target)
        for i, f in enumerate(feature_list):
            meta_acc[f].append(float(meta_vec[i]))
            target_acc[f].append(float(target_vec[i]))

    rows = []
    for f in feature_list:
        cm = float(np.mean(meta_acc[f])) if meta_acc[f] else None
        ct = float(np.mean(target_acc[f])) if target_acc[f] else None
        orthogonal = (
            cm is not None
            and ct is not None
            and abs(cm) <= corr_threshold
            and abs(ct) > corr_threshold
        )
        rows.append(
            {
                "feature": f,
                "corr_meta": cm,
                "corr_target": ct,
                "n_eras": len(meta_acc[f]),
                "orthogonal": orthogonal,
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
    from nmr import _gpu  # lazy: keeps this module's import graph acyclic

    matrix = clean.select(feature_list).cast(pl.Float64).to_numpy()
    ranks = _gpu.rankdata(matrix, axis=0)
    return scipy.stats.norm.ppf(ranks / (matrix.shape[0] + 1))


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
    k = min(100, abs_vals.size)
    if k == 0:
        order: np.ndarray = np.array([], dtype=np.int64)
    else:
        candidates = np.argpartition(abs_vals, -k)[-k:]
        order = candidates[np.argsort(-abs_vals[candidates])]
    top_rows = [
        {
            "feature_a": feature_list[iu[0][idx]],
            "feature_b": feature_list[iu[1][idx]],
            "mean_corr": float(mean_mat[iu[0][idx], iu[1][idx]]),
        }
        for idx in order
    ]
    summary = {
        "mean_abs_corr": float(abs_vals.mean()) if abs_vals.size else 0.0,
        "p50_abs_corr": float(np.percentile(abs_vals, 50)) if abs_vals.size else 0.0,
        "p90_abs_corr": float(np.percentile(abs_vals, 90)) if abs_vals.size else 0.0,
        "n_pairs": int(abs_vals.size),
        # Defensive PSD guard: the averaged matrix is PSD by construction
        # (per-era zero-filled matrices are block-diagonal PSD), but a
        # negative reading here would signal a regression in NaN handling.
        "min_eigenvalue": float(
            np.linalg.eigvalsh(mean_mat.astype(np.float64)).min()
        ),
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
    the regime column uses quartile bands. Degenerate eras (all-zero IC
    vectors — label-lag, all-NaN targets, or a ``degenerate`` flag column) are
    excluded from the percentile computation and labeled ``unlabeled``;
    ``ic_persistence`` is the mean adjacent-era Spearman rank correlation of
    per-era feature IC vectors.
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
            pl.col("ic").abs().max().alias("max_abs_ic"),
        )
        .sort("era")
    )
    if "degenerate" in ic_by_era.columns:
        flags = (
            ic_by_era.group_by("era")
            .agg(pl.col("degenerate").any().alias("degenerate"))
            .sort("era")
        )
        is_degenerate = flags.get_column("degenerate").to_numpy().astype(bool)
    else:
        # No flag column (synthetic inputs): an all-zero IC vector is the
        # degenerate signature produced by _per_era_pearson_chunks.
        is_degenerate = sig.get_column("max_abs_ic").to_numpy() == 0.0

    mean_ics = sig.get_column("mean_ic").to_numpy()
    n_valid = int((~is_degenerate).sum())
    if n_valid == 0:
        raise ValueError(
            "regime_analysis: no valid (non-degenerate) eras in input"
        )
    pct = np.full(len(mean_ics), np.nan)
    if n_valid > 1:
        ranks = np.argsort(np.argsort(mean_ics[~is_degenerate]))
        pct[~is_degenerate] = 100.0 * ranks / (n_valid - 1)
    else:
        pct[~is_degenerate] = 50.0

    valid_ics = mean_ics[~is_degenerate]
    q1 = float(np.percentile(valid_ics, 25.0))
    q3 = float(np.percentile(valid_ics, 75.0))
    low_thr = float(np.percentile(valid_ics, REGIME_LOW_PCT))
    high_thr = float(np.percentile(valid_ics, REGIME_HIGH_PCT))

    regime = np.where(
        is_degenerate,
        "unlabeled",
        np.where(pct <= 25.0, "low", np.where(pct >= 75.0, "high", "normal")),
    )
    crash = (~is_degenerate) & (pct <= REGIME_LOW_PCT)
    hot = (~is_degenerate) & (pct >= REGIME_HIGH_PCT)
    sig = sig.with_columns(
        pl.Series("degenerate", is_degenerate),
        pl.Series("pct_rank", pct).fill_nan(None).alias("pct_rank"),
        pl.Series("regime", regime),
        pl.Series("crash", crash),
        pl.Series("hot", hot),
    )

    eras = sig.get_column("era").to_list()
    crash_eras = [e for e, c in zip(eras, crash) if c]
    hot_eras = [e for e, h in zip(eras, hot) if h]

    # adjacent-era IC-vector Spearman; eras whose IC vector is entirely zero
    # (degenerate: target all-NaN or <2 rows) are constant under ranking and
    # make corrcoef undefined — exclude them from persistence.
    pivot = ic_by_era.pivot(on="feature", index="era", values="ic").sort("era")
    feature_names = [c for c in pivot.columns if c != "era"]
    matrix = pivot.select(feature_names).to_numpy()
    matrix = np.nan_to_num(matrix, nan=0.0)
    valid_rows = np.max(np.abs(matrix), axis=1) > 0.0
    if int(valid_rows.sum()) < 2:
        persistence = {"mean": 0.0, "std": 0.0, "n_adjacent": 0}
    else:
        ranks_mat = np.apply_along_axis(
            scipy.stats.rankdata, 1, matrix[valid_rows]
        )
        adj = [
            float(np.corrcoef(ranks_mat[t], ranks_mat[t - 1])[0, 1])
            for t in range(1, ranks_mat.shape[0])
        ]
        adj = [v for v in adj if np.isfinite(v)]
        persistence = {
            "mean": float(np.mean(adj)) if adj else 0.0,
            "std": float(np.std(adj, ddof=0)) if adj else 0.0,
            "n_adjacent": len(adj),
        }

    rolling = sig.select(
        pl.col("era"),
        pl.when(pl.col("degenerate"))
        .then(None)
        .otherwise(pl.col("mean_ic"))
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


def neutralized_ic_profile(
    chunks: Iterable[pl.DataFrame],
    signal_cols: Sequence[str],
    feature_cols: Sequence[str],
    target_col: str,
    era_col: str = "era",
    proportions: Sequence[float] = (0.0, 0.5, 1.0),
) -> pl.DataFrame:
    """Per-era IC of signals after feature neutralization (FNE profile).

    For each signal and neutralization proportion, the per-era Pearson IC of
    the neutralized signal vs ``target_col`` is computed on valid eras and
    aggregated as mean/std. ``proportion=0.0`` is the raw signal IC; ``1.0``
    fully orthogonalizes the signal against the feature set (intercept-aware
    linear neutralization via ``nmr._transforms.neutralize_array``). A signal
    whose IC collapses as the proportion grows is largely a linear function
    of the features; one that survives is orthogonal signal.
    """
    from nmr._transforms import neutralize_array
    from nmr.features import _feature_target_pearson

    signal_list = list(signal_cols)
    feature_list = list(feature_cols)
    prop_list = list(proportions)
    if not signal_list:
        raise ValueError("signal_cols must contain at least one signal")
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")
    if not all(0.0 <= p <= 1.0 for p in prop_list):
        raise ValueError("proportions must be within [0, 1]")

    acc: dict[tuple[str, float], list[float]] = {
        (s, p): [] for s in signal_list for p in prop_list
    }
    for part in chunks:
        clean = part.drop_nulls()
        if clean.height < 2:
            continue
        target_all = clean.get_column(target_col).cast(pl.Float64).to_numpy()
        features = clean.select(feature_list).cast(pl.Float64).to_numpy()
        for s in signal_list:
            pred = clean.get_column(s).cast(pl.Float64).to_numpy()
            mask = np.isfinite(target_all) & np.isfinite(pred) & np.all(
                np.isfinite(features), axis=1
            )
            target = target_all[mask]
            if target.size < 2 or bool(np.all(target == target[0])):
                continue
            feat_clean = features[mask]
            pred_clean = pred[mask]
            pinv: np.ndarray | None = None
            for p in prop_list:
                if p == 0.0:
                    neutralized = pred_clean
                else:
                    if pinv is None:
                        # Same geometry as neutralize_array's internal pinv
                        # (rcond=1e-6), computed once per signal per era.
                        design = np.hstack(
                            (
                                feat_clean,
                                np.ones((feat_clean.shape[0], 1), dtype=float),
                            )
                        )
                        pinv = np.linalg.pinv(design, rcond=1e-6)
                    neutralized = neutralize_array(
                        pred_clean, feat_clean, proportion=p, pseudo_inverse=pinv
                    )
                ic = _feature_target_pearson(
                    neutralized[:, None], target
                )[0]
                acc[(s, p)].append(float(ic))

    rows = [
        {
            "signal": s,
            "proportion": p,
            "mean_ic": float(np.mean(vals)) if vals else None,
            "corr_std": float(np.std(vals, ddof=0)) if vals else None,
            "n_eras": len(vals),
        }
        for (s, p), vals in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]
    return pl.DataFrame(rows)


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
