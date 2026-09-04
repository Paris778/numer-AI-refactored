"""Tests for E2 payout and downside metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nmr.evaluation import EvaluationEngine
from nmr.inference import deflated_sharpe, era_series_stats
from nmr.payout import (
    CLASSIC_ATOMIC_ENDER60_R1343_V1,
    CLASSIC_LEGACY_V1,
    OverlappingSimulationResult,
    PayoutResult,
    PayoutSeries,
    annual_compounded_return,
    burn_rate,
    calmar,
    cvar,
    era_payout_factors,
    gain_to_pain_ratio,
    kelly_fraction,
    load_payout_factors,
    max_burn_streak,
    max_drawdown,
    simulate_overlapping_portfolio,
    sortino,
    time_to_recovery,
)
from nmr.payout import payout_report as _payout_report
from nmr.payout import payout_series as _payout_series


def payout_series(*args, **kwargs):
    kwargs.setdefault("policy", CLASSIC_LEGACY_V1)
    return _payout_series(*args, **kwargs)


def payout_report(*args, **kwargs):
    kwargs.setdefault("policy", CLASSIC_LEGACY_V1)
    return _payout_report(*args, **kwargs)


def test_payout_policy_must_be_explicit_at_public_boundary() -> None:
    with pytest.raises(TypeError, match="policy"):
        _payout_series({"1343": 0.1}, {"1343": 0.01})


def test_atomic_policy_matches_current_classic_payout_contract() -> None:
    policy = CLASSIC_ATOMIC_ENDER60_R1343_V1

    assert policy.policy_id == "classic_atomic_ender60_r1343_v1"
    assert policy.target == "target_ender_60"
    assert policy.scoring_horizon == "60D"
    assert policy.concurrent_positions == 64

    out = payout_series(
        {"1343": 0.10, "1344": 0.50, "1345": -0.50},
        {"1343": 0.01, "1344": 0.10, "1345": -0.10},
        policy=policy,
    )
    assert np.allclose(out.raw, [0.45, 3.0, -3.0], atol=1e-12)
    assert np.allclose(out.clipped, [0.45, 1.0, -1.0], atol=1e-12)


def test_atomic_policy_rejects_historical_payout_factors() -> None:
    with pytest.raises(ValueError, match="fixed payout factor"):
        payout_series(
            {"1343": 0.10},
            {"1343": 0.01},
            policy=CLASSIC_ATOMIC_ENDER60_R1343_V1,
            pf={"1343": 0.5},
        )


def test_atomic_report_does_not_fabricate_round_level_capital_metrics() -> None:
    corr = {str(1343 + i): 0.01 + (i % 5) * 0.001 for i in range(20)}
    mmc = {str(1343 + i): 0.005 - (i % 3) * 0.0005 for i in range(20)}

    report = payout_report(
        corr,
        mmc,
        policy=CLASSIC_ATOMIC_ENDER60_R1343_V1,
        horizon="60D",
        n_trials=1,
        seed=7,
        n_boot=5,
    )

    assert report.policy_id == CLASSIC_ATOMIC_ENDER60_R1343_V1.policy_id
    assert report.overlapping_sim is None
    assert report.cagr_1y is None
    assert report.capital_metrics_reason == "round_level_returns_unavailable"


def test_payout_series_arithmetic_parity_and_clip_and_pf_order() -> None:
    corr = {"0002": 0.20, "0001": -0.20, "0003": -0.25}
    mmc = {"0001": 0.03, "0002": -0.04, "0003": 0.10}

    out = payout_series(corr, mmc, policy=CLASSIC_LEGACY_V1, pf=2.0, clip=0.05)
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


def test_drawdown_counts_losses_from_initial_equity() -> None:
    assert max_drawdown(np.array([-0.1])) == pytest.approx(0.1)
    assert max_drawdown(np.array([-0.1, 0.05])) == pytest.approx(0.1)
    assert time_to_recovery(np.array([-0.1, 0.05])) == 2


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
    # Interior optimum: neighboring fractions cannot improve mean log growth.
    series = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.1])
    fraction = kelly_fraction(series)

    def objective(value: float) -> float:
        return float(np.mean(np.log1p(value * series)))

    assert 0.0 < fraction < 1.0
    assert objective(fraction) >= objective(max(0.0, fraction - 1e-4))
    assert objective(fraction) >= objective(min(1.0, fraction + 1e-4))
    assert kelly_fraction(np.array([0.101, 0.099, 0.101, 0.099])) == pytest.approx(1.0)


def test_kelly_fraction_maximizes_admissible_mean_log_growth() -> None:
    returns = np.array([0.1] * 99 + [-2.0])
    fraction = kelly_fraction(returns)

    assert 0.0 <= fraction < 0.5
    assert np.all(1.0 + fraction * returns > 0.0)
    grid = np.linspace(0.0, np.nextafter(0.5, 0.0), 20_001)
    objective = np.mean(np.log1p(grid[:, None] * returns[None, :]), axis=1)
    expected = float(grid[int(np.argmax(objective))])
    assert fraction == pytest.approx(expected, abs=5e-4)


def test_kelly_uses_policy_clipped_returns() -> None:
    raw = np.array([0.03] * 19 + [-0.5])
    clipped = np.clip(raw, -0.05, 0.05)
    kelly_raw = kelly_fraction(raw)
    assert 0.0 < kelly_raw < 1.0
    assert kelly_fraction(clipped) == pytest.approx(1.0)


def test_overlapping_sim_zero_return_lockup() -> None:
    # K=20, n=100, all returns zero. Steady-state utilization (pre-deployment)
    # is (K-1)/K = 0.95; 20 warm-up eras average 0.475 -> overall 0.855 exactly.
    result = simulate_overlapping_portfolio(np.zeros(100), horizon_eras=20)
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
    result = simulate_overlapping_portfolio(np.array([0.1, 0.2, 0.3]), horizon_eras=2)
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
    assert report.cagr_1y == pytest.approx(annual_compounded_return(series.clipped))
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


def test_payout_series_per_era_pf_scales_each_era() -> None:
    corr = {"0001": 0.10, "0002": 0.20, "0003": -0.05}
    mmc = {"0001": 0.02, "0002": -0.03, "0003": 0.01}
    pf_map = {"0001": 2.0, "0003": 3.0}  # "0002" absent -> explicit fallback 1.0
    out = payout_series(corr, mmc, pf=pf_map, clip=0.05)
    expected_raw = np.array(
        [
            2.0 * ((0.75 * 0.10) + (2.25 * 0.02)),
            1.0 * ((0.75 * 0.20) + (2.25 * -0.03)),
            3.0 * ((0.75 * -0.05) + (2.25 * 0.01)),
        ],
        dtype=float,
    )
    assert np.allclose(out.raw, expected_raw, atol=1e-12)
    assert np.allclose(out.clipped, np.clip(expected_raw, -0.05, 0.05), atol=1e-12)


def test_payout_series_pf_mapping_empty_is_fallback_one() -> None:
    corr = {"0001": 0.10}
    mmc = {"0001": 0.02}
    out = payout_series(corr, mmc, pf={}, clip=0.05)
    assert np.allclose(out.raw, [0.75 * 0.10 + 2.25 * 0.02], atol=1e-12)


def test_payout_series_pf_mapping_invalid_values_raise() -> None:
    corr = {"0001": 0.10}
    mmc = {"0001": 0.02}
    for bad in ({"0001": 0.0}, {"0001": -1.0}, {"0001": float("nan")}):
        with pytest.raises(ValueError):
            payout_series(corr, mmc, pf=bad)


def test_payout_report_per_era_pf_summary_and_metrics() -> None:
    corr = {
        "0001": 0.10,
        "0002": 0.20,
        "0003": -0.05,
        "0004": 0.05,
        "0005": 0.15,
        "0006": -0.02,
    }
    mmc = {
        "0001": 0.02,
        "0002": -0.03,
        "0003": 0.01,
        "0004": 0.01,
        "0005": -0.01,
        "0006": 0.02,
    }
    pf_map = {
        "0001": 2.0,
        "0002": 3.0,
        "0003": 0.5,
        "0004": 1.5,
        "0005": 2.5,
        "0006": 0.8,
    }
    report = payout_report(
        corr, mmc, horizon="20D", n_trials=1, seed=5, pf=pf_map, n_boot=5
    )
    series = payout_series(corr, mmc, pf=pf_map)
    assert report.pf == pytest.approx(float(np.mean([2.0, 3.0, 0.5, 1.5, 2.5, 0.8])))
    assert report.mean_payout == pytest.approx(float(np.mean(series.clipped)))
    assert report.cagr_1y == pytest.approx(annual_compounded_return(series.clipped))
    assert report.burn_rate == pytest.approx(burn_rate(series.clipped))
    assert report.max_drawdown == pytest.approx(max_drawdown(series.clipped))


def test_load_payout_factors_round_to_pf(tmp_path) -> None:
    csv_path = tmp_path / "payout_factor_historic.csv"
    csv_path.write_text(
        "round,status,close,resolve,pf\n"
        "1020,Resolved,Jun 03 2025,Jul 04 2025,0.1009\n"
        "1019,Resolved,Jun 02 2025,Jul 03 2025,0.0987\n",
        encoding="utf-8",
    )
    assert load_payout_factors(csv_path) == {1020: 0.1009, 1019: 0.0987}


def test_load_payout_factors_missing_and_malformed(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_payout_factors(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("round,pf\n1019,not-a-number\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_payout_factors(bad)
    zero = tmp_path / "zero.csv"
    zero.write_text("round,pf\n1019,0.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_payout_factors(zero)
    dup = tmp_path / "dup.csv"
    dup.write_text("round,pf\n1019,0.1\n1019,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_payout_factors(dup)


def test_era_payout_factors_join_and_fallback(tmp_path) -> None:
    path = tmp_path / "pf.csv"
    path.write_text("round,pf\n1019,0.0987\n1020,0.1009\n", encoding="utf-8")
    assert era_payout_factors(path) == {"1019": 0.0987, "1020": 0.1009}
    assert era_payout_factors(None) == {}
    assert era_payout_factors(tmp_path / "absent.csv") == {}


def test_payout_series_pf_mapping_numeric_key_normalization() -> None:
    """int(era) == round: the lookup is numeric, so padding never matters."""
    corr = {"1": 0.10}
    mmc = {"1": 0.02}
    out = payout_series(corr, mmc, pf={"0001": 0.5}, clip=0.05)
    expected = 0.5 * ((0.75 * 0.10) + (2.25 * 0.02))
    assert np.allclose(out.raw, [expected], atol=1e-12)
    out2 = payout_series({"0001": 0.10}, {"0001": 0.02}, pf={"1": 0.5}, clip=0.05)
    assert np.allclose(out2.raw, out.raw, atol=1e-12)
