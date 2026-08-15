"""Tests for E2 payout and downside metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nmr.evaluation import EvaluationEngine
from nmr.inference import deflated_sharpe, era_series_stats
from nmr.payout import (
    OverlappingSimulationResult,
    PayoutResult,
    PayoutSeries,
    annual_compounded_return,
    burn_rate,
    calmar,
    cvar,
    gain_to_pain_ratio,
    kelly_fraction,
    max_burn_streak,
    max_drawdown,
    payout_report,
    payout_series,
    simulate_overlapping_portfolio,
    sortino,
    time_to_recovery,
)


def test_payout_series_arithmetic_parity_and_clip_and_pf_order() -> None:
    corr = {"0002": 0.20, "0001": -0.20, "0003": -0.25}
    mmc = {"0001": 0.03, "0002": -0.04, "0003": 0.10}

    out = payout_series(corr, mmc, pf=2.0, clip=0.05)
    assert isinstance(out, PayoutSeries)
    assert out.eras == ("0001", "0002", "0003")

    expected_raw = np.array(
        [
            2.0 * ((0.75 * -0.20) + (2.25 * 0.03)),
            2.0 * ((0.75 * 0.20) + (2.25 * -0.04)),
            2.0 * ((0.75 * -0.25) + (2.25 * 0.10)),
        ],
        dtype=float,
    )
    expected_clipped = np.clip(expected_raw, -0.05, 0.05)

    assert np.allclose(out.raw, expected_raw, atol=1e-12)
    assert np.allclose(out.clipped, expected_clipped, atol=1e-12)
    assert np.any(out.raw > 0.05)
    assert np.any(out.raw < -0.05)


def test_payout_report_uses_unclipped_raw_for_deflated_sharpe() -> None:
    corr = {
        "0001": -0.40,
        "0002": 0.50,
        "0003": -0.35,
        "0004": 0.45,
        "0005": 0.30,
        "0006": -0.50,
    }
    mmc = {
        "0001": 0.20,
        "0002": -0.30,
        "0003": 0.18,
        "0004": -0.22,
        "0005": 0.25,
        "0006": -0.28,
    }
    series = payout_series(corr, mmc, pf=2.0, clip=0.05)

    report = payout_report(
        corr,
        mmc,
        horizon="20D",
        n_trials=1,
        seed=17,
        pf=2.0,
        clip=0.05,
        n_boot=5,
    )

    raw_stats = era_series_stats(series.raw)
    clipped_stats = era_series_stats(series.clipped)
    expected_raw_dsr = deflated_sharpe(
        raw_stats.sharpe,
        n_trials=1,
        n_obs=len(series.eras),
        skew=raw_stats.skew,
        kurt=raw_stats.kurt,
    )
    clipped_dsr = deflated_sharpe(
        clipped_stats.sharpe,
        n_trials=1,
        n_obs=len(series.eras),
        skew=clipped_stats.skew,
        kurt=clipped_stats.kurt,
    )

    assert report.deflated_sharpe == pytest.approx(expected_raw_dsr, abs=1e-12)
    assert abs(report.deflated_sharpe - clipped_dsr) > 1e-6


def test_downside_metrics_match_hand_calcs() -> None:
    x = np.array([-0.1, 0.2, -0.3, 0.1, -0.2], dtype=float)

    assert burn_rate(x) == pytest.approx(0.6)
    assert cvar(x, q=0.4) == pytest.approx(-0.25)

    downside = np.array([-0.1, 0.0, -0.3, 0.0, -0.2], dtype=float)
    dd = math.sqrt(float(np.mean(downside**2)))
    expected_sortino = float(np.mean(x) / dd)
    assert sortino(x) == pytest.approx(expected_sortino, abs=1e-12)

    assert max_drawdown(x) == pytest.approx(0.4, abs=1e-12)
    assert calmar(x) == pytest.approx(-0.15, abs=1e-12)
    assert max_burn_streak(x) == 1
    assert time_to_recovery(x) == 3


def test_order_independence_for_dict_input_and_order_sensitivity_for_paths() -> None:
    corr_a = {
        "0001": 0.1,
        "0002": -0.2,
        "0003": 0.15,
        "0004": -0.05,
        "0005": 0.07,
        "0006": -0.09,
    }
    mmc_a = {
        "0001": 0.02,
        "0002": -0.01,
        "0003": 0.03,
        "0004": 0.00,
        "0005": -0.02,
        "0006": 0.01,
    }
    corr_b = {
        "0003": 0.15,
        "0001": 0.1,
        "0006": -0.09,
        "0004": -0.05,
        "0002": -0.2,
        "0005": 0.07,
    }
    mmc_b = {
        "0002": -0.01,
        "0004": 0.00,
        "0001": 0.02,
        "0003": 0.03,
        "0005": -0.02,
        "0006": 0.01,
    }

    rep_a = payout_report(
        corr_a,
        mmc_a,
        horizon="20D",
        n_trials=1,
        seed=9,
        n_boot=5,
        block_len=2,
    )
    rep_b = payout_report(
        corr_b,
        mmc_b,
        horizon="20D",
        n_trials=1,
        seed=9,
        n_boot=5,
        block_len=2,
    )
    assert rep_a == rep_b

    s1 = np.array([0.2, -0.2, 0.2, -0.2], dtype=float)
    s2 = np.array([0.2, 0.2, -0.2, -0.2], dtype=float)
    assert max_drawdown(s1) != max_drawdown(s2)
    assert time_to_recovery(s1) != time_to_recovery(s2)


def test_alignment_and_boundary_guards() -> None:
    corr = {
        "0001": 0.1,
        "0002": 0.2,
        "0003": 0.3,
        "0004": -0.1,
        "0005": 0.05,
        "0006": 0.15,
    }
    mmc = {
        "0002": -0.1,
        "0003": 0.1,
        "0004": 0.0,
        "0005": -0.05,
        "0006": 0.02,
        "0007": 0.2,
    }
    report = payout_report(
        corr,
        mmc,
        horizon="20D",
        n_trials=1,
        seed=3,
        n_boot=5,
        block_len=2,
    )
    assert isinstance(report, PayoutResult)
    assert report.n_eras == 5

    with pytest.raises(ValueError, match="share at least one era"):
        payout_series({"0001": 0.1}, {"0002": 0.2})

    with pytest.raises(ValueError, match="at least 2 overlapping eras"):
        payout_report({"0001": 0.1}, {"0001": 0.2}, horizon="20D", n_trials=1, seed=1)

    with pytest.raises(ValueError, match="finite"):
        payout_series({"0001": float("nan")}, {"0001": 0.0})


def test_degenerate_zero_series_is_well_defined() -> None:
    corr = {f"{i:04d}": 0.0 for i in range(1, 11)}
    mmc = {f"{i:04d}": 0.0 for i in range(1, 11)}
    report = payout_report(corr, mmc, horizon="20D", n_trials=1, seed=5, n_boot=5)

    assert report.mean_payout == pytest.approx(0.0)
    assert report.burn_rate == pytest.approx(0.0)
    assert report.cvar5 == pytest.approx(0.0)
    assert report.max_drawdown == pytest.approx(0.0)
    assert report.sortino == pytest.approx(0.0)
    assert report.calmar == pytest.approx(0.0)
    assert report.mmc_sharpe == pytest.approx(0.0)
    assert report.max_burn_streak == 0
    assert report.time_to_recovery == 0
    assert np.isfinite(report.deflated_sharpe)
    assert report.deflated_sharpe == pytest.approx(0.5)


def test_payout_report_determinism_same_seed() -> None:
    corr = {f"{i:04d}": ((-1.0) ** i) * (0.01 * i) for i in range(1, 31)}
    mmc = {f"{i:04d}": ((-1.0) ** (i + 1)) * (0.005 * i) for i in range(1, 31)}

    a = payout_report(
        corr,
        mmc,
        horizon="20D",
        n_trials=1,
        seed=123,
        n_boot=5,
        alpha=0.1,
    )
    b = payout_report(
        corr,
        mmc,
        horizon="20D",
        n_trials=1,
        seed=123,
        n_boot=5,
        alpha=0.1,
    )
    assert a == b


def test_max_drawdown_parity_with_evaluation_engine() -> None:
    per_era = {"0001": 0.2, "0002": -0.4, "0003": 0.1, "0004": -0.3, "0005": 0.2}
    values = np.asarray([per_era[k] for k in sorted(per_era)], dtype=float)

    engine = EvaluationEngine("custom")
    summary = engine.summarize(per_era)

    assert max_drawdown(values) == summary.max_drawdown


def test_ratio_and_input_boundaries() -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        payout_series({"0001": 0.1}, {"0001": 0.2}, pf=0.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        payout_series({"0001": 0.1}, {"0001": 0.2}, clip=0.0)

    with pytest.raises(ValueError, match="non-empty"):
        burn_rate([])
    with pytest.raises(ValueError, match="q"):
        cvar([1.0], q=1.0)
    with pytest.raises(ValueError, match="finite"):
        sortino([0.1, 0.2], target=float("nan"))


def test_as_finite_1d_rejects_2d_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="1-D"):
        burn_rate(np.ones((2, 2)))
    with pytest.raises(ValueError, match="1-D"):
        max_drawdown(np.ones((2, 2)))
    with pytest.raises(ValueError, match="non-empty"):
        calmar(np.array([]))


def test_payout_report_rejects_non_finite_mmc_on_aligned_eras() -> None:
    corr = {"0001": 0.1, "0002": 0.2}
    mmc = {"0001": 0.05, "0002": float("nan")}
    with pytest.raises(ValueError, match="mmc_by_era must contain only finite"):
        payout_report(corr, mmc, horizon="20D", n_trials=1, seed=1)


def test_annual_cagr_math() -> None:
    series = np.full(52, 0.01)
    expected = (1.01) ** 52 - 1.0
    assert annual_compounded_return(series) == pytest.approx(expected, rel=1e-12)


def test_annual_cagr_ruin_and_short_series() -> None:
    # product <= 0 -> -1.0 (total loss)
    assert annual_compounded_return(np.array([-1.0, 0.05])) == -1.0
    assert annual_compounded_return(np.array([-1.5, 0.05])) == -1.0
    # fewer than 2 observations -> 0.0
    assert annual_compounded_return(np.array([0.01])) == 0.0


def test_annual_cagr_input_validation() -> None:
    with pytest.raises(ValueError):
        annual_compounded_return(np.array([]))
    with pytest.raises(ValueError):
        annual_compounded_return(np.array([0.01, np.nan]))
    with pytest.raises(ValueError):
        annual_compounded_return(np.zeros((2, 2)))


def test_gain_to_pain_ratio() -> None:
    series = np.array([0.03, 0.03, 0.03, -0.01])
    assert gain_to_pain_ratio(series) == pytest.approx(9.0)


def test_gain_to_pain_zero_burn_states() -> None:
    # all positive -> +inf
    assert math.isinf(gain_to_pain_ratio(np.array([0.02, 0.01])))
    # all zero -> 0.0
    assert gain_to_pain_ratio(np.array([0.0, 0.0])) == 0.0


def test_kelly_fraction_bounds_and_degenerate() -> None:
    # zero variance -> 0.0
    assert kelly_fraction(np.array([0.01, 0.01, 0.01])) == 0.0
    # non-positive mean -> 0.0
    assert kelly_fraction(np.array([-0.01, 0.01])) == 0.0
    # mid-range: mu=0.02, sigma=0.2 -> 0.5
    series = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.1])
    mu = float(np.mean(series))
    var = float(np.var(series, ddof=0))
    assert kelly_fraction(series) == pytest.approx(min(1.0, mu / var))
    # saturation: mu=0.1, var=1e-6 -> mu/var = 100,000 -> capped at 1.0
    # (a constant array would have zero variance -> 0.0, so use a
    #  low-variance positive-drift sequence)
    assert kelly_fraction(np.array([0.101, 0.099, 0.101, 0.099])) == 1.0


def test_kelly_uses_raw_not_clipped() -> None:
    # Raw series: 19 x +0.03 and one -0.5. Raw Kelly = 0.0035 / 0.01334275
    # ~ 0.2623 (< 1). The clipped variant compresses variance so its Kelly
    # saturates at 1.0. Locks the director-approved raw-series contract.
    raw = np.array([0.03] * 19 + [-0.5])
    clipped = np.clip(raw, -0.05, 0.05)
    kelly_raw = kelly_fraction(raw)
    assert 0.0 < kelly_raw < 1.0
    assert kelly_fraction(clipped) == 1.0


def test_overlapping_sim_zero_return_lockup() -> None:
    # K=20, n=100, all returns zero. Steady-state utilization (pre-deployment)
    # is (K-1)/K = 0.95; 20 warm-up eras average 0.475 -> overall 0.855 exactly.
    result = simulate_overlapping_portfolio(
        np.zeros(100), horizon_eras=20
    )
    assert isinstance(result, OverlappingSimulationResult)
    assert result.final_equity == pytest.approx(1.0)
    assert result.portfolio_cagr == 0.0
    assert result.portfolio_max_drawdown == pytest.approx(0.0, abs=1e-12)
    assert result.avg_capital_utilization == pytest.approx(0.855, abs=1e-9)


def test_overlapping_sim_short_series() -> None:
    result = simulate_overlapping_portfolio(np.full(5, 0.01), horizon_eras=20)
    assert result.portfolio_cagr == 0.0
    assert result.portfolio_max_drawdown == 0.0
    assert result.avg_capital_utilization == 0.0
    assert result.final_equity == 1.0


def test_overlapping_sim_lockup_math() -> None:
    # K=2, n=3, returns [0.1, 0.2, 0.3]: hand-traced equity path.
    # t=0: deploy 0.5 -> tranche maturing at t=2; t=1: deploy 0.5 -> (3, 0.5).
    # t=2: tranche-2 matures paying x[0] = 0.1 -> cash = 0.5 * 1.1 = 0.55;
    # equity = 0.55 + 0.5 (still-locked tranche 3) = 1.05.
    # A payoff-index off-by-one (paying x[2] = 0.3 instead of x[0] = 0.1)
    # would produce equity 1.15, so this pins the x[maturity_t - horizon] index.
    result = simulate_overlapping_portfolio(
        np.array([0.1, 0.2, 0.3]), horizon_eras=2
    )
    assert result.final_equity == pytest.approx(1.05)
    assert result.portfolio_max_drawdown == 0.0
    assert result.portfolio_cagr == pytest.approx(1.05 ** (52.0 / 3.0) - 1.0)


def test_overlapping_sim_drag() -> None:
    # Positive-drift volatile series: cash drag makes the tranched portfolio
    # CAGR strictly below the serial geometric product CAGR.
    # (Verified numerically: port_cagr ~ 0.0348 vs serial_cagr ~ 1.559.)
    series = np.array([0.08, -0.04] * 30)
    result = simulate_overlapping_portfolio(series, horizon_eras=20)
    serial_final = float(np.prod(1.0 + series))
    serial_cagr = serial_final ** (52.0 / 60.0) - 1.0
    assert result.portfolio_cagr == pytest.approx(
        result.final_equity ** (52.0 / 60.0) - 1.0
    )
    assert result.portfolio_cagr < serial_cagr


def test_payout_report_includes_capital_metrics() -> None:
    corr = {f"{i:04d}": 0.02 + 0.01 * ((i % 5) - 2) for i in range(1, 41)}
    mmc = {f"{i:04d}": 0.01 for i in range(1, 41)}
    report = payout_report(corr, mmc, horizon="20D", n_trials=1, seed=7)
    series = payout_series(corr, mmc)
    assert report.cagr_1y == pytest.approx(
        annual_compounded_return(series.clipped)
    )
    assert report.gain_to_pain_ratio == pytest.approx(
        gain_to_pain_ratio(series.clipped)
    )
    assert report.kelly_fraction == pytest.approx(kelly_fraction(series.raw))
    assert report.overlapping_sim is not None
    expected_sim = simulate_overlapping_portfolio(series.clipped, horizon_eras=20)
    assert report.overlapping_sim.final_equity == pytest.approx(
        expected_sim.final_equity
    )
    assert report.overlapping_sim.avg_capital_utilization == pytest.approx(
        expected_sim.avg_capital_utilization
    )


def test_payout_series_sorts_eras_numerically() -> None:
    """Era keys are numeric strings; ordering must be numeric (2 < 10), not
    lexicographic ("10" < "2") — the documented regression class."""
    series = payout_series({"10": 0.1, "2": 0.2}, {"10": 0.05, "2": 0.06})
    assert series.eras == ("2", "10")
