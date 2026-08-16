"""Tests for nmr._gpu — cupy rankdata must be bit-identical to scipy."""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.stats

from nmr import _gpu


def _rand_with_ties(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.round(rng.normal(size=n), 6)  # float32-style exact ties
    if n > 4:
        values[1] = values[0]
        values[3] = values[2]
    return values


def test_gpu_rankdata_1d_matches_scipy() -> None:
    values = _rand_with_ties(500)
    got = _gpu.rankdata(values)
    expected = scipy.stats.rankdata(values, method="average")
    assert np.array_equal(got, expected)


def test_gpu_rankdata_2d_axis0_matches_scipy() -> None:
    matrix = np.column_stack(
        [_rand_with_ties(200, seed=s) for s in range(5)]
    )
    got = _gpu.rankdata(matrix, axis=0)
    expected = scipy.stats.rankdata(matrix, method="average", axis=0)
    assert np.array_equal(got, expected)


def test_gpu_rankdata_2d_axis1_matches_scipy() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(30, 8))
    got = _gpu.rankdata(matrix, axis=1)
    expected = scipy.stats.rankdata(matrix, method="average", axis=1)
    assert np.array_equal(got, expected)


def test_gpu_rankdata_nan_isolation() -> None:
    # Intentional divergence from scipy 1.17: scipy's rankdata returns an
    # all-NaN result when ANY input is NaN ('propagate'), which would poison
    # an entire era's Spearman/corr computation. The GPU path ranks the
    # finite values correctly and leaves NaN only at the NaN positions.
    values = np.array([3.0, np.nan, 1.0, 2.0, 2.0])
    got = _gpu.rankdata(values)
    assert np.isnan(got[1])
    assert np.array_equal(got[[0, 2, 3, 4]], [4.0, 1.0, 2.5, 2.5])


def test_gpu_rankdata_fallback_without_cupy(monkeypatch) -> None:
    monkeypatch.setattr(_gpu, "_CUPY", None)
    monkeypatch.setattr(_gpu, "_CUPY_LOADED", True)
    values = _rand_with_ties(300)
    assert np.array_equal(
        _gpu.rankdata(values),
        scipy.stats.rankdata(values, method="average"),
    )


def test_gpu_rankdata_edge_cases() -> None:
    # empty, single element, all-equal, all-NaN
    assert _gpu.rankdata(np.array([])).shape == (0,)
    assert _gpu.rankdata(np.array([5.0]))[0] == 1.0
    assert np.array_equal(_gpu.rankdata(np.array([2.0, 2.0, 2.0])), [2.0, 2.0, 2.0])
    assert np.isnan(_gpu.rankdata(np.array([np.nan, np.nan]))[0])


def test_analysis_rank_paths_bit_identical_with_and_without_gpu(
    monkeypatch,
) -> None:
    """The analysis rank hooks (spearman IC, rank-gaussianize, drift AUC)
    must produce byte-identical outputs whether cupy is used or not."""
    from nmr.analysis import feature_drift_profile, feature_ic_by_era
    from nmr.features import _era_ic_pair

    rng = np.random.default_rng(9)
    rows = []
    for e in range(3):
        era = f"{e + 1:04d}"
        for i in range(120):
            x = float(rng.normal())
            rows.append(
                {
                    "era": era,
                    "f1": float(np.round(x, 6)),
                    "f2": float(np.round(x * x, 6)),
                    "target": float(np.round(x + rng.normal(scale=0.1), 6)),
                }
            )
    frame = pl.DataFrame(rows)
    chunks = frame.partition_by("era", maintain_order=True)

    def run() -> tuple[object, object, object]:
        ic = feature_ic_by_era(chunks, ["f1", "f2"], "target")
        dp = feature_drift_profile(
            chunks, chunks, ["f1", "f2"], edge_sample_stride=1
        )
        return ic, dp, _era_ic_pair(chunks[0], ["f1", "f2"], "target", "era", spearman=True)

    gpu_on = run()
    monkeypatch.setattr(_gpu, "_CUPY", None)
    gpu_off = run()
    for name, (a, b) in zip(("ic_by_era", "drift_profile", "era_ic_pair"), zip(gpu_on, gpu_off)):
        if isinstance(a, pl.DataFrame):
            assert a.equals(b), name
        else:
            era_a, pa, sa, da = a
            era_b, pb, sb, db = b
            assert era_a == era_b and da == db
            assert np.array_equal(pa, pb) and np.array_equal(sa, sb)
