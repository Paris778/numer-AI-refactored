"""Payout proxy and downside diagnostics for Evaluation Suite v2.

This module converts per-era CORR/MMC series into an economic payout proxy and
downside shape metrics. Inference statistics (bootstrap CI, AC-adjusted Sharpe,
Deflated Sharpe) are delegated to `nmr.inference`.

Terminal-tranche accounting convention (simulator): tranches still locked at
the end of the series are carried at par principal in ``final_equity`` and the
equity curve — no mark-to-market and no unrealized payoff for the final
``horizon_eras`` tranches.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from nmr.inference import (
    BootstrapCI,
    Horizon,
    ac_adjusted_sharpe,
    block_bootstrap_ci,
    deflated_sharpe,
    era_series_stats,
    resolve_block_len,
)

__all__ = [
    "PayoutSeries",
    "PayoutResult",
    "OverlappingSimulationResult",
    "payout_series",
    "annual_compounded_return",
    "gain_to_pain_ratio",
    "kelly_fraction",
    "simulate_overlapping_portfolio",
    "burn_rate",
    "cvar",
    "sortino",
    "max_drawdown",
    "calmar",
    "max_burn_streak",
    "time_to_recovery",
    "payout_report",
]


@dataclass(frozen=True)
class PayoutSeries:
    eras: tuple[str, ...]
    raw: np.ndarray
    clipped: np.ndarray


@dataclass(frozen=True)
class PayoutResult:
    n_eras: int
    pf: float
    mean_payout: float
    payout_ci: BootstrapCI
    deflated_sharpe: float
    burn_rate: float
    cvar5: float
    max_drawdown: float
    sortino: float
    calmar: float
    mmc_sharpe: float
    max_burn_streak: int
    time_to_recovery: int
    cagr_1y: float
    gain_to_pain_ratio: float
    kelly_fraction: float
    overlapping_sim: OverlappingSimulationResult | None = None


def _as_finite_1d(
    series: np.ndarray | list[float] | tuple[float, ...], *, name: str
) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if x.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must contain only finite values")
    return x


_HORIZON_ERAS: dict[str, int] = {"20D": 20, "60D": 60}


@dataclass(frozen=True)
class OverlappingSimulationResult:
    portfolio_cagr: float
    portfolio_max_drawdown: float
    avg_capital_utilization: float
    final_equity: float


def payout_series(
    corr_by_era: Mapping[str, float],
    mmc_by_era: Mapping[str, float],
    *,
    pf: float = 1.0,
    clip: float = 0.05,
) -> PayoutSeries:
    pf_f = float(pf)
    clip_f = float(clip)
    if not np.isfinite(pf_f) or pf_f <= 0.0:
        raise ValueError("pf must be finite and > 0")
    if not np.isfinite(clip_f) or clip_f <= 0.0:
        raise ValueError("clip must be finite and > 0")

    # Era keys are numeric strings — sort numerically (the repo has a
    # documented regression class from lexicographic era ordering).
    eras = tuple(sorted(set(corr_by_era) & set(mmc_by_era), key=int))
    if not eras:
        raise ValueError("corr_by_era and mmc_by_era must share at least one era")

    corr = np.asarray([float(corr_by_era[era]) for era in eras], dtype=float)
    mmc = np.asarray([float(mmc_by_era[era]) for era in eras], dtype=float)
    if not np.isfinite(corr).all() or not np.isfinite(mmc).all():
        raise ValueError("corr_by_era and mmc_by_era must contain only finite values")

    raw = pf_f * ((0.75 * corr) + (2.25 * mmc))
    clipped = np.clip(raw, -clip_f, clip_f)
    return PayoutSeries(eras=eras, raw=raw, clipped=clipped)


def annual_compounded_return(
    clipped: np.ndarray | list[float] | tuple[float, ...],
    *,
    eras_per_year: float = 52.0,
) -> float:
    """Annualized geometric compounded return on stake.

    Computes (prod(1 + r_t)) ** (eras_per_year / n) - 1 in float64 over the
    clipped round-return series. Returns -1.0 when the wealth product is
    <= 0 (total loss) and 0.0 when fewer than 2 observations exist.
    """
    x = _as_finite_1d(clipped, name="clipped")
    n = int(x.size)
    if n < 2:
        return 0.0
    cum_growth = float(np.prod(1.0 + x))
    if cum_growth <= 0.0:
        return -1.0
    return float(cum_growth ** (float(eras_per_year) / float(n)) - 1.0)


def gain_to_pain_ratio(
    clipped: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Gain-to-pain ratio: sum(positive returns) / sum(|negative returns|).

    Zero-loss states return +inf (precedented by ``calmar``; the canonical
    JSON sanitizer maps non-finite floats to strings) or 0.0 when the series
    is entirely flat.
    """
    x = _as_finite_1d(clipped, name="clipped")
    pos_sum = float(np.sum(np.maximum(x, 0.0)))
    neg_sum = float(np.sum(np.abs(np.minimum(x, 0.0))))
    if neg_sum == 0.0:
        return float("inf") if pos_sum > 0.0 else 0.0
    return float(pos_sum / neg_sum)


def kelly_fraction(
    raw: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Bounded discrete Kelly stake fraction: min(1.0, max(0.0, mu / var)).

    Computed on the RAW (unclipped) payout series. The clipped series has
    Popoviciu-bounded variance (sigma^2 <= 0.0025 under the +-5% clip), so
    mu/sigma^2 there saturates at 1.0 for every viable model and carries no
    discrimination. ``payout_report`` passes ``series.raw``.
    """
    x = _as_finite_1d(raw, name="raw")
    mu = float(np.mean(x))
    var = float(np.var(x, ddof=0))
    if var == 0.0 or mu <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, mu / var)))


def simulate_overlapping_portfolio(
    clipped: np.ndarray | list[float] | tuple[float, ...],
    *,
    horizon_eras: int = 20,
    initial_capital: float = 1.0,
    eras_per_year: float = 52.0,
) -> OverlappingSimulationResult:
    """Simulate multi-round concurrent capital lockup and dynamic reinvestment.

    Each era deploys min(cash, total_equity / horizon_eras) into a new tranche
    that matures ``horizon_eras`` eras later with the initiating era's return
    (1 + r_{t - K}) — Numerai round semantics. Equity and utilization are
    recorded BEFORE the era's deployment. Tranches still locked at the end of
    the series are carried at par principal (no unrealized payoff). Series
    shorter than the horizon return a zeroed result.
    """
    x = _as_finite_1d(clipped, name="clipped")
    n = int(x.size)
    horizon = int(horizon_eras)
    if horizon < 1:
        raise ValueError("horizon_eras must be >= 1")
    if n < horizon:
        return OverlappingSimulationResult(
            portfolio_cagr=0.0,
            portfolio_max_drawdown=0.0,
            avg_capital_utilization=0.0,
            final_equity=float(initial_capital),
        )

    cash = float(initial_capital)
    active_stakes: list[tuple[int, float]] = []
    equity_curve = np.zeros(n, dtype=float)
    utilization = np.zeros(n, dtype=float)

    for t in range(n):
        still_active: list[tuple[int, float]] = []
        for maturity_t, principal in active_stakes:
            if maturity_t == t:
                cash += principal * (1.0 + x[maturity_t - horizon])
            else:
                still_active.append((maturity_t, principal))
        active_stakes = still_active

        locked_capital = sum(p for _, p in active_stakes)
        total_equity = cash + locked_capital
        equity_curve[t] = total_equity
        utilization[t] = (
            locked_capital / total_equity if total_equity > 0 else 0.0
        )

        allocated = min(cash, total_equity / float(horizon))
        cash -= allocated
        active_stakes.append((t + horizon, allocated))

    final_eq = float(equity_curve[-1])
    cagr = (
        float(final_eq / initial_capital) ** (float(eras_per_year) / float(n))
        - 1.0
        if final_eq > 0.0
        else -1.0
    )
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = np.where(
        running_max > 0, (running_max - equity_curve) / running_max, 0.0
    )
    return OverlappingSimulationResult(
        portfolio_cagr=cagr,
        portfolio_max_drawdown=float(np.max(drawdowns)),
        avg_capital_utilization=float(np.mean(utilization)),
        final_equity=final_eq,
    )


def burn_rate(clipped: np.ndarray | list[float] | tuple[float, ...]) -> float:
    x = _as_finite_1d(clipped, name="clipped")
    return float(np.mean(x < 0.0))


def cvar(
    series: np.ndarray | list[float] | tuple[float, ...], *, q: float = 0.05
) -> float:
    x = _as_finite_1d(series, name="series")
    q_f = float(q)
    if not (0.0 < q_f < 1.0):
        raise ValueError("q must satisfy 0 < q < 1")
    k = max(1, int(math.floor(q_f * x.size)))
    tail = np.sort(x)[:k]
    return float(np.mean(tail))


def sortino(
    series: np.ndarray | list[float] | tuple[float, ...],
    *,
    target: float = 0.0,
) -> float:
    x = _as_finite_1d(series, name="series")
    target_f = float(target)
    if not np.isfinite(target_f):
        raise ValueError("target must be finite")

    downside = np.minimum(x - target_f, 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    mean = float(np.mean(x))
    if dd == 0.0:
        return 0.0
    return float((mean - target_f) / dd)


def max_drawdown(series: np.ndarray | list[float] | tuple[float, ...]) -> float:
    x = _as_finite_1d(series, name="series")
    cumulative = np.cumsum(x)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    return float(np.max(drawdowns))


def calmar(series: np.ndarray | list[float] | tuple[float, ...]) -> float:
    x = _as_finite_1d(series, name="series")
    mean = float(np.mean(x))
    mdd = max_drawdown(x)
    if mdd == 0.0:
        return float(np.inf) if mean > 0.0 else 0.0
    return float(mean / mdd)


def max_burn_streak(series: np.ndarray | list[float] | tuple[float, ...]) -> int:
    x = _as_finite_1d(series, name="series")
    neg = x < 0.0
    best = 0
    current = 0
    for is_neg in neg:
        if is_neg:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return int(best)


def time_to_recovery(series: np.ndarray | list[float] | tuple[float, ...]) -> int:
    x = _as_finite_1d(series, name="series")
    cumulative = np.cumsum(x)
    running_max = np.maximum.accumulate(cumulative)
    underwater = cumulative < running_max

    best = 0
    current = 0
    for is_underwater in underwater:
        if is_underwater:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return int(best)


def payout_report(
    corr_by_era: Mapping[str, float],
    mmc_by_era: Mapping[str, float],
    *,
    horizon: Horizon,
    n_trials: int,
    seed: int,
    pf: float = 1.0,
    clip: float = 0.05,
    n_boot: int = 1000,
    alpha: float = 0.05,
    trials_sr_var: float | None = None,
    sr0_benchmark: float = 0.0,
    block_len: int | None = None,
) -> PayoutResult:
    series = payout_series(corr_by_era, mmc_by_era, pf=pf, clip=clip)
    n = len(series.eras)
    if n < 2:
        raise ValueError("payout_report requires at least 2 overlapping eras")

    if block_len is None:
        bl = resolve_block_len(n, horizon)
    else:
        bl = resolve_block_len(n, horizon, override=block_len)

    payout_ci = block_bootstrap_ci(
        series.clipped,
        lambda a: float(np.mean(a)),
        block_len=bl,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )

    raw_stats = era_series_stats(series.raw)
    dsr = deflated_sharpe(
        raw_stats.sharpe,
        n_trials=n_trials,
        n_obs=n,
        skew=raw_stats.skew,
        kurt=raw_stats.kurt,
        trials_sr_var=trials_sr_var,
        sr0_benchmark=sr0_benchmark,
    )

    mmc_aligned = np.asarray(
        [float(mmc_by_era[era]) for era in series.eras], dtype=float
    )
    if not np.isfinite(mmc_aligned).all():
        raise ValueError("mmc_by_era must contain only finite values on aligned eras")

    clipped = series.clipped
    horizon_eras = _HORIZON_ERAS[horizon]
    return PayoutResult(
        n_eras=n,
        pf=float(pf),
        mean_payout=float(np.mean(clipped)),
        payout_ci=payout_ci,
        deflated_sharpe=float(dsr),
        burn_rate=burn_rate(clipped),
        cvar5=cvar(clipped, q=0.05),
        max_drawdown=max_drawdown(clipped),
        sortino=sortino(clipped),
        calmar=calmar(clipped),
        mmc_sharpe=ac_adjusted_sharpe(mmc_aligned, horizon=horizon),
        max_burn_streak=max_burn_streak(clipped),
        time_to_recovery=time_to_recovery(clipped),
        cagr_1y=annual_compounded_return(clipped),
        gain_to_pain_ratio=gain_to_pain_ratio(clipped),
        kelly_fraction=kelly_fraction(series.raw),
        overlapping_sim=simulate_overlapping_portfolio(
            clipped, horizon_eras=horizon_eras
        ),
    )
