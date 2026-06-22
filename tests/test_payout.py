"""Tests for E2 payout and downside metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nmr.evaluation import EvaluationEngine
from nmr.inference import deflated_sharpe, era_series_stats
from nmr.payout import (
    PayoutResult,
    PayoutSeries,
    burn_rate,
    calmar,
    cvar,
    max_burn_streak,
    max_drawdown,
    payout_report,
    payout_series,
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
