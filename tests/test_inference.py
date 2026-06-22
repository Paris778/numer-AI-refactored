"""Tests for E1 inference core primitives."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nmr.inference import (
    BootstrapCI,
    SeriesStats,
    ac_adjusted_sharpe,
    block_bootstrap_ci,
    deflated_sharpe,
    era_series_stats,
    resolve_bandwidth,
    resolve_block_len,
)


def test_era_series_stats_degenerate_defaults() -> None:
    stats = era_series_stats([1.0, 1.0, 1.0, 1.0])
    assert isinstance(stats, SeriesStats)
    assert stats.n == 4
    assert stats.mean == pytest.approx(1.0)
    assert stats.std == pytest.approx(0.0)
    assert stats.sharpe == pytest.approx(0.0)
    assert stats.skew == pytest.approx(0.0)
    assert stats.kurt == pytest.approx(3.0)


def test_era_series_stats_invalid_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        era_series_stats([])
    with pytest.raises(ValueError, match="finite"):
        era_series_stats([1.0, np.nan])


def test_resolvers_apply_horizon_floors() -> None:
    assert resolve_block_len(574, "20D") >= 5
    assert resolve_block_len(574, "60D") >= 13
    assert resolve_bandwidth(574, "20D") >= 4
    assert resolve_bandwidth(574, "60D") >= 12


def test_resolvers_override_validation_and_small_n_guard() -> None:
    assert resolve_block_len(20, "20D", override=7) == 7
    assert resolve_bandwidth(20, "20D", override=5) == 5

    with pytest.raises(ValueError, match="block floor"):
        resolve_block_len(10, "60D")
    with pytest.raises(ValueError, match="bandwidth floor"):
        resolve_bandwidth(10, "60D")
    with pytest.raises(ValueError, match="override"):
        resolve_block_len(5, "20D", override=0)
    with pytest.raises(ValueError, match="override"):
        resolve_bandwidth(5, "20D", override=5)


def _generate_ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    x = np.empty(n, dtype=float)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = (phi * x[i - 1]) + eps[i]
    return x


def test_ac_adjusted_sharpe_directionality() -> None:
    pos_auto = _generate_ar1(phi=0.6, n=600, seed=7) + 1.0
    naive = era_series_stats(pos_auto).sharpe
    adjusted = ac_adjusted_sharpe(pos_auto, horizon="20D")
    assert adjusted < naive


def test_ac_adjusted_sharpe_iid_near_equal() -> None:
    iid = _generate_ar1(phi=0.0, n=2000, seed=11)
    naive = era_series_stats(iid).sharpe
    adjusted = ac_adjusted_sharpe(iid, horizon="20D")
    assert adjusted == pytest.approx(naive, rel=1e-2, abs=1e-3)


def test_ac_adjusted_sharpe_boundaries() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        ac_adjusted_sharpe([1.0], bandwidth=1)
    with pytest.raises(ValueError, match="provide horizon or bandwidth"):
        ac_adjusted_sharpe([1.0, 2.0])
    with pytest.raises(ValueError, match="bandwidth"):
        ac_adjusted_sharpe([1.0, 2.0, 3.0], bandwidth=0)


def test_deflated_sharpe_monotone_in_trials() -> None:
    kwargs = {
        "sharpe": 0.3,
        "n_obs": 60,
        "skew": 0.0,
        "kurt": 3.0,
        "trials_sr_var": 0.08,
    }
    dsr_1 = deflated_sharpe(n_trials=1, **kwargs)
    dsr_10 = deflated_sharpe(n_trials=10, **kwargs)
    dsr_100 = deflated_sharpe(n_trials=100, **kwargs)
    dsr_1000 = deflated_sharpe(n_trials=1000, **kwargs)

    assert dsr_1 > dsr_10 > dsr_100 > dsr_1000


def test_deflated_sharpe_trials_var_rules() -> None:
    value = deflated_sharpe(
        sharpe=0.5,
        n_trials=1,
        n_obs=120,
        skew=0.1,
        kurt=3.2,
        trials_sr_var=None,
    )
    assert 0.0 <= value <= 1.0

    with pytest.raises(ValueError, match="required when n_trials > 1"):
        deflated_sharpe(
            sharpe=0.5,
            n_trials=2,
            n_obs=120,
            skew=0.1,
            kurt=3.2,
            trials_sr_var=None,
        )


def test_deflated_sharpe_boundary_raises() -> None:
    with pytest.raises(ValueError, match="n_obs"):
        deflated_sharpe(
            sharpe=0.2,
            n_trials=1,
            n_obs=1,
            skew=0.0,
            kurt=3.0,
        )
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe(
            sharpe=0.2,
            n_trials=0,
            n_obs=20,
            skew=0.0,
            kurt=3.0,
        )
    with pytest.raises(ValueError, match="radicand"):
        deflated_sharpe(
            sharpe=1.0,
            n_trials=1,
            n_obs=100,
            skew=4.0,
            kurt=1.0,
        )


def test_block_bootstrap_ci_determinism_same_seed() -> None:
    data = np.linspace(-1.0, 1.0, 200)
    stat_fn = lambda x: float(np.mean(x))

    first = block_bootstrap_ci(data, stat_fn, block_len=5, n_boot=300, seed=123)
    second = block_bootstrap_ci(data, stat_fn, block_len=5, n_boot=300, seed=123)

    assert isinstance(first, BootstrapCI)
    assert first == second


def test_block_bootstrap_ci_2d_row_coherence() -> None:
    n = 120
    col = np.linspace(-2.0, 3.0, n)
    data = np.column_stack([col, col])

    ci = block_bootstrap_ci(
        data,
        lambda arr: float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]),
        block_len=7,
        n_boot=200,
        seed=42,
    )

    assert ci.point == pytest.approx(1.0, abs=1e-12)
    assert ci.lo == pytest.approx(1.0, abs=1e-12)
    assert ci.hi == pytest.approx(1.0, abs=1e-12)


def test_block_bootstrap_ci_invalid_cases() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        block_bootstrap_ci(np.array([]), np.mean, block_len=1, n_boot=10, seed=1)
    with pytest.raises(ValueError, match="block_len"):
        block_bootstrap_ci(
            np.array([1.0, 2.0]), np.mean, block_len=3, n_boot=10, seed=1
        )
    with pytest.raises(ValueError, match="finite"):
        block_bootstrap_ci(
            np.array([1.0, np.nan]),
            np.mean,
            block_len=1,
            n_boot=10,
            seed=1,
        )
    finite_only_on_original = lambda x: (
        float(0.0) if np.array_equal(x, np.array([1.0, 2.0, 3.0])) else float("nan")
    )
    with pytest.raises(ValueError, match="insufficient valid"):
        block_bootstrap_ci(
            np.array([1.0, 2.0, 3.0]),
            finite_only_on_original,
            block_len=1,
            n_boot=100,
            seed=1,
        )


def test_block_bootstrap_ci_coverage_sanity() -> None:
    mu = 0.7
    n_trials = 20
    covered = 0
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        sample = rng.normal(loc=mu, scale=1.0, size=250)
        ci = block_bootstrap_ci(
            sample,
            lambda x: float(np.mean(x)),
            block_len=5,
            n_boot=50,
            seed=seed,
            alpha=0.05,
        )
        if ci.lo <= mu <= ci.hi:
            covered += 1

    coverage = covered / n_trials
    assert 0.80 <= coverage <= 1.0


def test_horizon_floor_is_used_in_adjusted_sharpe() -> None:
    x = np.linspace(-1.0, 1.0, 574)
    expected_k = resolve_bandwidth(len(x), "60D")
    assert expected_k >= 12

    by_horizon = ac_adjusted_sharpe(x, horizon="60D")
    by_explicit_k = ac_adjusted_sharpe(x, bandwidth=expected_k)
    assert by_horizon == pytest.approx(by_explicit_k, abs=1e-12)
