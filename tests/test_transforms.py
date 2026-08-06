"""Direct unit tests for the shared rank-domain transforms (math backbone).

``nmr/_transforms`` is the single source of truth for rank geometry, consumed
by evaluation, ensembling, risk neutralization, and the deployment closure.
These tests pin the mathematical contracts directly (bounds, monotonicity,
tie handling, degenerate inputs) on top of the oracle-parity tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from nmr._transforms import (
    gaussianize,
    neutralize_array,
    power_1_5,
    rank_gaussianize,
    rank_gaussianize_unit_variance,
    standardize_unit_variance,
    tie_kept_rank,
)


def test_tie_kept_rank_values_in_unit_interval_and_monotone() -> None:
    values = np.array([3.0, 1.0, 2.0, 0.0, 5.0])
    ranks = tie_kept_rank(values)
    assert ranks.shape == values.shape
    assert np.all(ranks >= 0.0) and np.all(ranks < 1.0)
    # Rank order must mirror value order (no ties here).
    assert np.all(np.diff(ranks[np.argsort(values)]) > 0.0)


def test_tie_kept_rank_averages_ties() -> None:
    ranks = tie_kept_rank(np.array([2.0, 1.0, 2.0, 1.0]))
    # Values 1.0 occupy rank slots {1,2} -> average 1.5 -> (1.5-0.5)/4 = 0.25.
    # Values 2.0 occupy slots {3,4} -> average 3.5 -> (3.5-0.5)/4 = 0.75.
    assert np.allclose(ranks, [0.75, 0.25, 0.75, 0.25], atol=1e-12)


def test_tie_kept_rank_empty_returns_empty() -> None:
    assert tie_kept_rank(np.array([])).size == 0


def test_gaussianize_matches_norm_ppf_and_is_monotone() -> None:
    from scipy.stats import norm

    x = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    assert np.allclose(gaussianize(x), norm.ppf(x), atol=1e-12)
    assert gaussianize(np.array([0.5])) == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(gaussianize(x)) > 0.0)


def test_rank_gaussianize_of_constant_is_zero() -> None:
    # All ranks collapse to 0.5 -> ppf(0.5) = 0.
    out = rank_gaussianize(np.full(7, 2.5))
    assert np.allclose(out, 0.0, atol=1e-12)


def test_rank_gaussianize_is_invariant_to_strictly_monotone_transform() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=50)
    monotone = np.sign(x) * np.abs(x) ** 3.0  # strictly increasing on R
    assert np.allclose(rank_gaussianize(x), rank_gaussianize(monotone), atol=1e-12)


def test_rank_gaussianize_negation_flips_sign_exactly() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=40)
    # rank(-x) = 1 - rank(x) exactly (no ties), ppf symmetric, so the
    # rank-gaussian of -x is exactly the negation.
    assert np.allclose(rank_gaussianize(-x), -rank_gaussianize(x), atol=1e-12)


def test_standardize_unit_variance_unit_std_and_zeros_on_constant() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = standardize_unit_variance(x)
    assert np.std(out, ddof=0) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(standardize_unit_variance(np.full(5, 2.0)), 0.0, atol=1e-12)
    assert standardize_unit_variance(np.array([])).size == 0


def test_rank_gaussianize_unit_variance_unit_std() -> None:
    x = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8])
    out = rank_gaussianize_unit_variance(x)
    assert np.std(out, ddof=0) == pytest.approx(1.0, abs=1e-12)
    # Constant input has zero variance -> standardized to zeros.
    assert np.allclose(rank_gaussianize_unit_variance(np.full(6, 0.5)), 0.0, atol=1e-12)


def test_power_1_5_preserves_sign_and_magnitude() -> None:
    x = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    out = power_1_5(x)
    expected = np.sign(x) * np.abs(x) ** 1.5
    assert np.allclose(out, expected, atol=1e-12)
    assert power_1_5(np.array([4.0])) == pytest.approx(8.0, abs=1e-12)
    assert power_1_5(np.array([-4.0])) == pytest.approx(-8.0, abs=1e-12)
    assert power_1_5(np.array([0.0])) == pytest.approx(0.0, abs=1e-12)


def test_neutralize_array_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        neutralize_array(np.array([1.0, 2.0]), np.array([[1.0], [2.0], [3.0]]), 1.0)


def test_neutralize_array_non_finite_raises() -> None:
    features = np.arange(8.0).reshape(4, 2)
    with pytest.raises(ValueError, match="finite"):
        neutralize_array(np.array([1.0, np.nan, 3.0, 4.0]), features, 1.0)
    with pytest.raises(ValueError, match="finite"):
        neutralize_array(
            np.array([1.0, 2.0, 3.0, 4.0]),
            features + np.array([[0.0, np.inf]]),
            1.0,
        )


def test_neutralize_array_zero_variance_returns_unchanged() -> None:
    pred = np.full(5, 0.5)
    features = np.arange(10.0).reshape(5, 2)
    out = neutralize_array(pred, features, 1.0)
    assert np.array_equal(out, pred)


def test_neutralize_array_proportion_midpoint_property() -> None:
    """neutralize(p=0.5) is exactly the midpoint of pred and neutralize(p=1)."""
    rng = np.random.default_rng(3)
    pred = rng.normal(size=20)
    features = rng.normal(size=(20, 3))
    full = neutralize_array(pred, features, 1.0)
    half = neutralize_array(pred, features, 0.5)
    assert np.allclose(half, 0.5 * (pred + full), atol=1e-12)


def test_neutralize_array_zero_proportion_is_identity() -> None:
    rng = np.random.default_rng(5)
    pred = rng.normal(size=12)
    features = rng.normal(size=(12, 2))
    assert np.allclose(neutralize_array(pred, features, 0.0), pred, atol=1e-12)
