"""Versioned payout policies and downside diagnostics.

Current Classic Atomic payout is bound to Ender-60 and computed as
``clip(3 * CORR60 + 15 * MMC60, -1, 1)``. The legacy payout-factor formula is
retained only as an explicitly named historical policy.

Weekly validation-era returns cannot reconstruct Atomic's 64 concurrent daily
round positions. Atomic reports therefore leave overlapping capital metrics
unavailable instead of publishing a misleading simulation.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

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
    "PayoutPolicy",
    "CLASSIC_ATOMIC_ENDER60_R1343_V1",
    "CLASSIC_LEGACY_V1",
    "PAYOUT_POLICIES",
    "resolve_payout_policy",
    "PayoutSeries",
    "PayoutResult",
    "OverlappingSimulationResult",
    "PAYOUT_FACTOR_FILENAME",
    "load_payout_factors",
    "era_payout_factors",
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
class PayoutPolicy:
    policy_id: str
    target: str | None
    scoring_horizon: Horizon | None
    corr_multiplier: float
    mmc_multiplier: float
    clip: float
    fixed_payout_factor: float | None
    concurrent_positions: int | None
    effective_from_round: int | None


CLASSIC_ATOMIC_ENDER60_R1343_V1 = PayoutPolicy(
    policy_id="classic_atomic_ender60_r1343_v1",
    target="target_ender_60",
    scoring_horizon="60D",
    corr_multiplier=3.0,
    mmc_multiplier=15.0,
    clip=1.0,
    fixed_payout_factor=1.0,
    concurrent_positions=64,
    effective_from_round=1343,
)

CLASSIC_LEGACY_V1 = PayoutPolicy(
    policy_id="classic_legacy_075_225_clip005_v1",
    target=None,
    scoring_horizon=None,
    corr_multiplier=0.75,
    mmc_multiplier=2.25,
    clip=0.05,
    fixed_payout_factor=None,
    concurrent_positions=None,
    effective_from_round=None,
)

PAYOUT_POLICIES = {
    policy.policy_id: policy
    for policy in (CLASSIC_ATOMIC_ENDER60_R1343_V1, CLASSIC_LEGACY_V1)
}


def resolve_payout_policy(policy: PayoutPolicy | str) -> PayoutPolicy:
    if isinstance(policy, PayoutPolicy):
        return policy
    try:
        return PAYOUT_POLICIES[policy]
    except KeyError as exc:
        raise ValueError(
            f"unknown payout policy {policy!r}; valid policies: {tuple(PAYOUT_POLICIES)}"
        ) from exc


@dataclass(frozen=True)
class PayoutSeries:
    eras: tuple[str, ...]
    raw: np.ndarray
    clipped: np.ndarray


@dataclass(frozen=True)
class PayoutResult:
    policy_id: str
    target: str | None
    scoring_horizon: Horizon
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
    capital_metrics_reason: str | None = None


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

#: Filename of the cleaned historic payout-factor CSV inside the data version dir.
PAYOUT_FACTOR_FILENAME = "payout_factor_historic.csv"


def load_payout_factors(path: Path) -> dict[int, float]:
    """Parse the cleaned historic payout-factor CSV: round -> payout factor.

    The cleaned file keeps ``round, status, close, resolve, pf`` only (metric
    columns are noise). Fails loud on a missing file, blank rows, non-numeric
    factors, non-finite or non-positive factors, or duplicate rounds.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"payout factor CSV missing: {p}")
    factors: dict[int, float] = {}
    with open(p, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            raw_round = (row.get("round") or "").strip()
            raw_pf = (row.get("pf") or "").strip()
            if not raw_round or not raw_pf:
                raise ValueError(f"payout factor CSV {p} has a blank round/pf row")
            round_no = int(raw_round)
            pf = float(raw_pf)
            if not math.isfinite(pf) or pf <= 0.0:
                raise ValueError(
                    f"payout factor for round {round_no} must be finite and > 0, got {pf}"
                )
            if round_no in factors:
                raise ValueError(f"duplicate round {round_no} in payout factor CSV {p}")
            factors[round_no] = pf
    return factors


def era_payout_factors(path: Path | None) -> dict[str, float]:
    """Era-keyed payout factors, joining ``int(era) == round``.

    Returns ``{}`` when ``path`` is None or the file is absent — the explicit
    all-1.0 fallback mode (PF=1.0 is a fallback/synthetic default, never the
    historical assumption). A present-but-malformed file fails loud via
    :func:`load_payout_factors`.
    """
    if path is None or not Path(path).is_file():
        return {}
    return {f"{round_no:04d}": pf for round_no, pf in load_payout_factors(path).items()}


@dataclass(frozen=True)
class OverlappingSimulationResult:
    portfolio_cagr: float
    portfolio_max_drawdown: float
    avg_capital_utilization: float
    final_equity: float


def _normalize_era_key(key: object) -> str:
    """Normalize an era/round key numerically: ``"0001"`` and ``"1"`` both map
    to ``"1"`` so the ``int(era) == round`` join holds regardless of padding."""
    try:
        return str(int(key))
    except (ValueError, TypeError):
        return str(key)


def payout_series(
    corr_by_era: Mapping[str, float],
    mmc_by_era: Mapping[str, float],
    *,
    policy: PayoutPolicy | str,
    pf: Mapping[str, float] | float | None = None,
    clip: float | None = None,
) -> PayoutSeries:
    """Compute a per-era payout series under an explicit versioned policy.

    Historical payout factors and clip overrides are accepted only by the
    legacy policy. Atomic fixes both values as part of its economic contract.
    """
    resolved = resolve_payout_policy(policy)
    if resolved.fixed_payout_factor is not None:
        if pf is not None and pf != resolved.fixed_payout_factor:
            raise ValueError(
                f"policy {resolved.policy_id} has fixed payout factor "
                f"{resolved.fixed_payout_factor}; pf overrides are forbidden"
            )
        if clip is not None and float(clip) != resolved.clip:
            raise ValueError(
                f"policy {resolved.policy_id} has fixed clip {resolved.clip}; "
                "clip overrides are forbidden"
            )
        effective_pf: Mapping[str, float] | float = resolved.fixed_payout_factor
        clip_f = resolved.clip
    else:
        effective_pf = 1.0 if pf is None else pf
        clip_f = resolved.clip if clip is None else float(clip)

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

    if isinstance(effective_pf, Mapping):
        pf_map = {_normalize_era_key(k): float(v) for k, v in effective_pf.items()}
        for era, value in pf_map.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"pf for era {era!r} must be finite and > 0, got {value}"
                )
        pf_values = np.asarray(
            [pf_map.get(_normalize_era_key(era), 1.0) for era in eras], dtype=float
        )
    else:
        pf_f = float(effective_pf)
        if not np.isfinite(pf_f) or pf_f <= 0.0:
            raise ValueError("pf must be finite and > 0")
        pf_values = np.full(len(eras), pf_f, dtype=float)

    raw = pf_values * (
        (resolved.corr_multiplier * corr) + (resolved.mmc_multiplier * mmc)
    )
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
    returns: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Bounded stake fraction maximizing empirical mean log growth.

    The domain is ``0 <= f <= 1`` and is tightened so every observed wealth
    multiplier ``1 + f*r`` remains strictly positive. Callers must pass the
    policy-clipped return series, not an unpayable raw score proxy.
    """
    x = _as_finite_1d(returns, name="returns")
    if float(np.mean(x)) <= 0.0 or np.all(x == x[0]):
        return 0.0
    negative = x[x < 0.0]
    upper = 1.0
    if negative.size:
        upper = min(upper, float(np.min(-1.0 / negative)))
        upper = float(np.nextafter(upper, 0.0))
    if upper <= 0.0:
        return 0.0

    def objective(fraction: float) -> float:
        wealth = 1.0 + fraction * x
        if np.any(wealth <= 0.0):
            return float("inf")
        return -float(np.mean(np.log(wealth)))

    result = minimize_scalar(
        objective,
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": 1e-12},
    )
    candidates = [0.0, upper, float(result.x)]
    best = min(candidates, key=objective)
    return float(best if objective(best) < 0.0 else 0.0)


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
        utilization[t] = locked_capital / total_equity if total_equity > 0 else 0.0

        allocated = min(cash, total_equity / float(horizon))
        cash -= allocated
        active_stakes.append((t + horizon, allocated))

    final_eq = float(equity_curve[-1])
    cagr = (
        float(final_eq / initial_capital) ** (float(eras_per_year) / float(n)) - 1.0
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
    cumulative = np.concatenate(([0.0], np.cumsum(x)))
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
    cumulative = np.concatenate(([0.0], np.cumsum(x)))
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
    policy: PayoutPolicy | str,
    horizon: Horizon,
    n_trials: int,
    seed: int,
    pf: Mapping[str, float] | float | None = None,
    clip: float | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    trials_sr_var: float | None = None,
    sr0_benchmark: float = 0.0,
    block_len: int | None = None,
) -> PayoutResult:
    resolved = resolve_payout_policy(policy)
    if resolved.scoring_horizon is not None and horizon != resolved.scoring_horizon:
        raise ValueError(
            f"policy {resolved.policy_id} requires horizon="
            f"{resolved.scoring_horizon}, got {horizon}"
        )
    series = payout_series(
        corr_by_era,
        mmc_by_era,
        policy=resolved,
        pf=pf,
        clip=clip,
    )
    n = len(series.eras)
    if n < 2:
        raise ValueError("payout_report requires at least 2 overlapping eras")

    if block_len is None:
        bl = resolve_block_len(n, horizon)
    else:
        bl = resolve_block_len(n, horizon, override=block_len)

    # PayoutResult.pf is a scalar summary: the uniform factor when ``pf`` is a
    # scalar, else the mean of the per-era factors applied on the scored eras.
    effective_pf = resolved.fixed_payout_factor if pf is None else pf
    if isinstance(effective_pf, Mapping):
        pf_map = {_normalize_era_key(k): float(v) for k, v in effective_pf.items()}
        pf_summary = float(
            np.mean([pf_map.get(_normalize_era_key(era), 1.0) for era in series.eras])
        )
    else:
        pf_summary = float(1.0 if effective_pf is None else effective_pf)

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
    supports_era_level_capital_sim = resolved.concurrent_positions is None
    return PayoutResult(
        policy_id=resolved.policy_id,
        target=resolved.target,
        scoring_horizon=horizon,
        n_eras=n,
        pf=pf_summary,
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
        kelly_fraction=kelly_fraction(series.clipped),
        overlapping_sim=(
            simulate_overlapping_portfolio(clipped, horizon_eras=_HORIZON_ERAS[horizon])
            if supports_era_level_capital_sim
            else None
        ),
        capital_metrics_reason=(
            None
            if supports_era_level_capital_sim
            else "round_level_returns_unavailable"
        ),
    )
