"""Inference primitives for Evaluation Suite v2.

This module is pure statistics over NumPy arrays. It provides descriptive
series moments, horizon-aware bootstrap/autocorrelation tuning, circular
block-bootstrap confidence intervals, Lo autocorrelation-adjusted Sharpe, and
Bailey-Lopez de Prado Deflated Sharpe.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import kurtosis, norm, skew

__all__ = [
    "SeriesStats",
    "BootstrapCI",
    "era_series_stats",
    "resolve_block_len",
    "resolve_bandwidth",
    "block_bootstrap_ci",
    "ac_adjusted_sharpe",
    "deflated_sharpe",
]

Horizon = Literal["20D", "60D"]
ArrayLike1D = Sequence[float] | np.ndarray
ArrayLikeND = Sequence[float] | Sequence[Sequence[float]] | np.ndarray

_BLOCK_FLOOR: dict[Horizon, int] = {"20D": 5, "60D": 13}
_BANDWIDTH_FLOOR: dict[Horizon, int] = {"20D": 4, "60D": 12}


@dataclass(frozen=True)
class SeriesStats:
    n: int
    mean: float
    std: float
    sharpe: float
    skew: float
    kurt: float


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lo: float
    hi: float
    alpha: float
    n_boot: int
    block_len: int


def _as_finite_1d(series: ArrayLike1D, *, name: str) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if x.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must contain only finite values")
    return x


def _as_finite_data(data: ArrayLikeND) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError("data must be 1-D or 2-D")
    if arr.shape[0] == 0:
        raise ValueError("data must have at least one row")
    if not np.isfinite(arr).all():
        raise ValueError("data must contain only finite values")
    return arr


def _validate_horizon(horizon: Horizon) -> Horizon:
    if horizon not in ("20D", "60D"):
        raise ValueError("horizon must be one of {'20D', '60D'}")
    return horizon


def era_series_stats(series: ArrayLike1D) -> SeriesStats:
    x = _as_finite_1d(series, name="series")
    n = int(x.size)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    sharpe = 0.0 if std == 0.0 else float(mean / std)

    if n < 3 or std == 0.0:
        skew_value = 0.0
    else:
        skew_value = float(skew(x, bias=False))

    if n < 4 or std == 0.0:
        kurt_value = 3.0
    else:
        kurt_value = float(kurtosis(x, fisher=False, bias=False))

    return SeriesStats(
        n=n,
        mean=mean,
        std=std,
        sharpe=sharpe,
        skew=skew_value,
        kurt=kurt_value,
    )


def resolve_block_len(n: int, horizon: Horizon, *, override: int | None = None) -> int:
    if n < 1:
        raise ValueError("n must be >= 1")
    horizon = _validate_horizon(horizon)

    if override is not None:
        if not 1 <= override <= n:
            raise ValueError("override block_len must satisfy 1 <= override <= n")
        return int(override)

    heuristic = int(round(n ** (1.0 / 3.0)))
    floor = _BLOCK_FLOOR[horizon]
    cap = n
    if floor > cap:
        raise ValueError("n is too small for the selected horizon block floor")
    return int(min(max(heuristic, floor), cap))


def resolve_bandwidth(
    n: int,
    horizon: Horizon,
    *,
    override: int | None = None,
) -> int:
    if n < 2:
        raise ValueError("n must be >= 2")
    horizon = _validate_horizon(horizon)

    cap = n - 1
    if override is not None:
        if not 1 <= override <= cap:
            raise ValueError("override bandwidth must satisfy 1 <= override <= n-1")
        return int(override)

    heuristic = int(math.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0))))
    floor = _BANDWIDTH_FLOOR[horizon]
    if floor > cap:
        raise ValueError("n is too small for the selected horizon bandwidth floor")
    return int(min(max(heuristic, floor), cap))


def block_bootstrap_ci(
    data: ArrayLikeND,
    stat_fn: Callable[[np.ndarray], float],
    *,
    block_len: int,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
    min_valid_frac: float = 0.5,
) -> BootstrapCI:
    arr = _as_finite_data(data)
    n = int(arr.shape[0])

    if not 1 <= block_len <= n:
        raise ValueError("block_len must satisfy 1 <= block_len <= n")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if not (0.0 < min_valid_frac <= 1.0):
        raise ValueError("min_valid_frac must satisfy 0 < min_valid_frac <= 1")

    point = float(stat_fn(arr))
    if not np.isfinite(point):
        raise ValueError("stat_fn returned a non-finite point estimate")

    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_len))
    block_offsets = np.arange(block_len)
    values: list[float] = []

    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + block_offsets[None, :]) % n).reshape(-1)[:n]
        theta = float(stat_fn(arr[idx]))
        if np.isfinite(theta):
            values.append(theta)

    valid_frac = len(values) / n_boot
    if valid_frac < min_valid_frac:
        raise ValueError("insufficient valid bootstrap replicates")

    valid = np.asarray(values, dtype=float)
    lo, hi = np.percentile(valid, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return BootstrapCI(
        point=point,
        lo=float(lo),
        hi=float(hi),
        alpha=float(alpha),
        n_boot=int(n_boot),
        block_len=int(block_len),
    )


def ac_adjusted_sharpe(
    series: ArrayLike1D,
    *,
    horizon: Horizon | None = None,
    bandwidth: int | None = None,
) -> float:
    x = _as_finite_1d(series, name="series")
    n = int(x.size)
    if n < 2:
        raise ValueError("series must contain at least 2 observations")

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    if std == 0.0:
        return 0.0
    sr = float(mean / std)

    if bandwidth is not None:
        if not 1 <= bandwidth <= n - 1:
            raise ValueError("bandwidth must satisfy 1 <= bandwidth <= n-1")
        k_max = int(bandwidth)
    elif horizon is not None:
        k_max = resolve_bandwidth(n, horizon)
    else:
        raise ValueError("provide horizon or bandwidth")

    centered = x - mean
    denom = float(np.dot(centered, centered))
    if denom == 0.0:
        return 0.0

    weights = 1.0 - (np.arange(1, k_max + 1, dtype=float) / (k_max + 1.0))
    rhos = np.empty(k_max, dtype=float)
    for k in range(1, k_max + 1):
        numer = float(np.dot(centered[:-k], centered[k:]))
        rhos[k - 1] = numer / denom

    d_term = 1.0 + 2.0 * float(np.sum(weights * rhos))
    d_term = max(d_term, 1e-12)
    return float(sr / math.sqrt(d_term))


def deflated_sharpe(
    sharpe: float,
    *,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurt: float,
    trials_sr_var: float | None = None,
    sr0_benchmark: float = 0.0,
) -> float:
    """Compute Deflated Sharpe Ratio (Bailey-Lopez de Prado).

    Caller contract: `sharpe`, `skew`, and `kurt` must come from the unclipped
    raw series when used for payout-proxy evaluation.
    """

    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    sharpe_f = float(sharpe)
    skew_f = float(skew)
    kurt_f = float(kurt)
    sr0_base = float(sr0_benchmark)

    if not np.isfinite([sharpe_f, skew_f, kurt_f, sr0_base]).all():
        raise ValueError("sharpe, skew, kurt, and sr0_benchmark must be finite")

    if n_trials == 1:
        sr0 = sr0_base
    else:
        if trials_sr_var is None:
            raise ValueError("trials_sr_var is required when n_trials > 1")
        trials_var = float(trials_sr_var)
        if not np.isfinite(trials_var) or trials_var <= 0.0:
            raise ValueError("trials_sr_var must be finite and > 0 when n_trials > 1")
        n_trials_f = float(n_trials)
        expected_max = ((1.0 - np.euler_gamma) * norm.ppf(1.0 - (1.0 / n_trials_f))) + (
            np.euler_gamma * norm.ppf(1.0 - (1.0 / (n_trials_f * np.e)))
        )
        sr0 = sr0_base + (math.sqrt(trials_var) * float(expected_max))

    radicand = 1.0 - (skew_f * sharpe_f) + (((kurt_f - 1.0) / 4.0) * (sharpe_f**2))
    if radicand <= 0.0:
        raise ValueError("deflated_sharpe radicand must be > 0")

    z_score = (sharpe_f - sr0) * math.sqrt(float(n_obs - 1)) / math.sqrt(radicand)
    return float(norm.cdf(z_score))
