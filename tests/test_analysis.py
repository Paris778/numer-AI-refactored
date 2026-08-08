"""Unit tests for nmr.analysis — synthetic frames, seeded where random."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nmr.analysis import SplitStats, describe_splits, era_structure


def _frame(n_eras: int = 4, rows_per_era: int = 8) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for e in range(n_eras):
        era = f"{e + 1:04d}"
        for i in range(rows_per_era):
            rows.append({"era": era, "id": f"n{e:03d}{i:03d}", "x": float(i)})
    return pl.DataFrame(rows)


def test_describe_splits_counts() -> None:
    splits = {"train": _frame(4, 8), "validation": _frame(3, 10)}
    out = describe_splits(splits)
    assert set(out) == {"train", "validation"}
    train = out["train"]
    assert isinstance(train, SplitStats)
    assert train.n_rows == 32
    assert train.n_eras == 4
    assert train.min_era == "0001" and train.max_era == "0004"
    assert train.rows_per_era_min == 8
    assert train.rows_per_era_max == 8
    assert train.rows_per_era_mean == 8.0
    assert train.n_ids == 32


def test_describe_splits_rows_per_era_stats() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002", "0002"],
            "id": ["a", "b", "c", "d", "e"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = describe_splits({"s": frame})["s"]
    assert out.rows_per_era_min == 2
    assert out.rows_per_era_max == 3
    assert out.rows_per_era_mean == 2.5
    assert out.n_eras == 2


def test_describe_splits_requires_id() -> None:
    frame = pl.DataFrame({"era": ["0001"], "x": [1.0]})
    with pytest.raises(ValueError):
        describe_splits({"s": frame})


def test_era_structure_gap_detection() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0002", "0002", "0004"],
            "id": ["a", "b", "c", "d"],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = era_structure(frame)
    assert out["era"].to_list() == ["0001", "0002", "0004"]
    assert out["gap"].to_list() == [False, False, True]  # 0004 jumps from 0002
    assert out["n_rows"].to_list() == [1, 2, 1]


def test_era_structure_empty_raises() -> None:
    with pytest.raises(ValueError):
        era_structure(pl.DataFrame({"era": [], "id": [], "x": []}))


import scipy.stats

from nmr.analysis import target_correlation_matrix, target_profile


def test_target_profile_moments() -> None:
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for e in range(3):
        era = f"{e + 1:04d}"
        for v in rng.normal(size=100):
            rows.append({"era": era, "target": float(v)})
    frame = pl.DataFrame(rows)
    out = target_profile(frame, ["target"])
    assert len(out) == 1
    row = out.row(0, named=True)
    series = frame["target"].to_numpy()
    assert row["n_eras_present"] == 3
    assert np.isclose(row["pooled_mean"], float(series.mean()), atol=1e-12)
    assert np.isclose(row["pooled_std"], float(series.std(ddof=0)), atol=1e-12)
    assert np.isclose(
        row["pooled_skew"], float(scipy.stats.skew(series)), atol=1e-12
    )
    assert np.isclose(
        row["pooled_kurtosis"],
        float(scipy.stats.kurtosis(series, fisher=True)),
        atol=1e-12,
    )
    assert row["missing_rate"] == 0.0
    assert row["zero_variance_era_count"] == 0


def test_target_profile_non_finite_dropped() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0001", "0001", "0002", "0002", "0002"],
            "target": [1.0, float("nan"), 3.0, None, 5.0, 6.0],
        }
    )
    out = target_profile(frame, ["target"])
    row = out.row(0, named=True)
    assert row["n_eras_present"] == 2
    assert np.isclose(row["missing_rate"], 2 / 6)
    assert np.isclose(row["pooled_mean"], (1.0 + 3.0 + 5.0 + 6.0) / 4)


def test_target_profile_zero_variance_era_counted() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002", "0002"],
            "target": [5.0, 5.0, 1.0, 2.0, 3.0],
        }
    )
    out = target_profile(frame, ["target"])
    assert out.row(0, named=True)["zero_variance_era_count"] == 1


def test_target_correlation_matrix_hand_computed() -> None:
    rng = np.random.default_rng(3)
    rows: list[dict[str, object]] = []
    for e in range(4):
        era = f"{e + 1:04d}"
        for i in range(20):
            a = float(rng.normal())
            b = 2.0 * a + float(rng.normal(scale=0.5))
            rows.append({"era": era, "target_alpha": a, "target_beta": b})
    frame = pl.DataFrame(rows)
    out = target_correlation_matrix(frame, ["target_alpha", "target_beta"])
    assert out.columns == ["target_a", "target_b", "mean_corr", "n_eras"]
    assert out.row(0, named=True)["target_a"] == "target_alpha"
    assert out.row(0, named=True)["target_b"] == "target_beta"
    assert out.row(0, named=True)["n_eras"] == 4
    era_corrs = []
    for part in frame.partition_by("era"):
        a = part["target_alpha"].to_numpy()
        b = part["target_beta"].to_numpy()
        ra, rb = scipy.stats.rankdata(a), scipy.stats.rankdata(b)
        era_corrs.append(np.corrcoef(ra, rb)[0, 1])
    assert np.isclose(out.row(0, named=True)["mean_corr"], float(np.mean(era_corrs)), atol=1e-12)


def test_target_correlation_matrix_nan_pair_skipped() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0001", "0001", "0002", "0002", "0002"],
            "target_alpha": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target_beta": [None, None, None, 1.0, 2.0, 3.0],
        }
    )
    out = target_correlation_matrix(frame, ["target_alpha", "target_beta"])
    assert out.row(0, named=True)["n_eras"] == 1  # era 0001 skipped (all-NaN beta)
    assert np.isclose(out.row(0, named=True)["mean_corr"], 1.0)  # perfectly monotone in 0002


def test_target_correlation_matrix_deterministic() -> None:
    rng = np.random.default_rng(5)
    rows = [
        {"era": f"{e + 1:04d}", "a": float(v), "b": float(-v)}
        for e in range(3)
        for v in rng.normal(size=10)
    ]
    frame = pl.DataFrame(rows)
    out1 = target_correlation_matrix(frame, ["a", "b"])
    out2 = target_correlation_matrix(frame, ["a", "b"])
    assert out1.equals(out2)


from nmr.analysis import feature_ic_by_era, feature_ic_screen
from nmr.features import _per_era_pearson


def _ic_frame() -> pl.DataFrame:
    rng = np.random.default_rng(21)
    rows: list[dict[str, float | str]] = []
    for e in range(4):
        era = f"{e + 1:04d}"
        for i in range(12):
            rows.append(
                {
                    "era": era,
                    "feature_alpha": float(rng.normal()),
                    "feature_beta": float(rng.normal()),
                    "target": float(rng.normal()),
                }
            )
    return pl.DataFrame(rows)


def test_feature_ic_by_era_long_form() -> None:
    frame = _ic_frame()
    out = feature_ic_by_era(frame, ["feature_alpha", "feature_beta"], "target")
    assert out.columns == ["era", "feature", "ic", "degenerate"]
    assert out.height == 4 * 2
    assert out["feature"].n_unique() == 2
    assert out["era"].n_unique() == 4
    corrs, _ = _per_era_pearson(frame, ["feature_alpha", "feature_beta"], "target", "era")
    for era, vec in corrs.items():
        rows = out.filter(pl.col("era") == era)
        assert np.array_equal(rows["ic"].to_numpy(), vec)


def test_feature_ic_by_era_degenerate_flag() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0002", "0002", "0002", "0002"],
            "feature_alpha": [1.0, 2.0, 1.0, 1.0, 1.0],
            "feature_beta": [3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [0.1, 1.0, 1.0, 1.0, 1.0],
        }
    )
    out = feature_ic_by_era(frame, ["feature_alpha", "feature_beta"], "target")
    assert out.filter(pl.col("era") == "0001")["degenerate"].all()  # <2 rows
    assert out.filter(pl.col("era") == "0002")["degenerate"].all()  # const target
    assert (out.filter(pl.col("era") == "0001")["ic"] == 0.0).all()  # zero vectors


def test_feature_ic_screen_multi_target() -> None:
    frame = _ic_frame()
    out = feature_ic_screen(frame, ["feature_alpha", "feature_beta"], ["target"])
    assert out.columns == [
        "feature",
        "target",
        "mean_corr",
        "corr_std",
        "decay_slope",
        "cross_regime_variance",
        "n_eras",
        "stable",
    ]
    assert out.height == 2
    assert out["target"].to_list() == ["target", "target"]


def test_feature_ic_screen_empty_targets_raises() -> None:
    with pytest.raises(ValueError):
        feature_ic_screen(_ic_frame(), ["feature_alpha"], [])


from nmr.analysis import feature_summary


def _chunks(n_eras: int = 5, rows: int = 40) -> list[pl.DataFrame]:
    rng = np.random.default_rng(42)
    chunks = []
    for e in range(n_eras):
        era = f"{e + 1:04d}"
        chunks.append(
            pl.DataFrame(
                {
                    "era": [era] * rows,
                    "f1": rng.normal(size=rows),
                    "f2": rng.normal(loc=2.0, scale=0.5, size=rows),
                }
            )
        )
    return chunks


def test_feature_summary_moments_match_scipy() -> None:
    chunks = _chunks()
    out = feature_summary(chunks, ["f1", "f2"])
    assert out.columns == [
        "feature",
        "pooled_mean",
        "pooled_std",
        "pooled_skew",
        "pooled_kurtosis",
        "min",
        "max",
        "missing_rate",
    ]
    full = pl.concat(chunks)
    for feature in ["f1", "f2"]:
        series = full[feature].to_numpy()
        row = out.filter(pl.col("feature") == feature).row(0, named=True)
        assert np.isclose(row["pooled_mean"], float(series.mean()), atol=1e-12)
        assert np.isclose(row["pooled_std"], float(series.std(ddof=0)), atol=1e-12)
        assert np.isclose(row["pooled_skew"], float(scipy.stats.skew(series)), atol=1e-9)
        assert np.isclose(
            row["pooled_kurtosis"],
            float(scipy.stats.kurtosis(series, fisher=True)),
            atol=1e-8,
        )
        assert row["missing_rate"] == 0.0


def test_feature_summary_constant_column() -> None:
    chunks = [
        pl.DataFrame({"era": ["0001"] * 5, "f1": [7.0] * 5}),
        pl.DataFrame({"era": ["0002"] * 5, "f1": [7.0] * 5}),
    ]
    out = feature_summary(chunks, ["f1"])
    row = out.row(0, named=True)
    assert row["pooled_std"] == 0.0
    assert row["pooled_skew"] == 0.0
    assert row["pooled_kurtosis"] == 0.0
    assert row["min"] == 7.0 and row["max"] == 7.0


def test_feature_summary_missing_rate() -> None:
    chunks = [
        pl.DataFrame({"era": ["0001", "0001", "0001"], "f1": [1.0, None, 3.0]}),
        pl.DataFrame({"era": ["0002", "0002", "0002"], "f1": [4.0, 5.0, None]}),
    ]
    out = feature_summary(chunks, ["f1"])
    assert np.isclose(out.row(0, named=True)["missing_rate"], 2 / 6)


def test_feature_summary_chunked_vs_single_pass() -> None:
    chunks = _chunks()
    out_chunked = feature_summary(chunks, ["f1", "f2"])
    out_single = feature_summary([pl.concat(chunks)], ["f1", "f2"])
    for c in ["pooled_mean", "pooled_std", "pooled_skew", "pooled_kurtosis", "min", "max"]:
        assert np.allclose(
            out_chunked[c].to_numpy(), out_single[c].to_numpy(), rtol=1e-9
        )


def test_feature_summary_chunked_bit_identical() -> None:
    chunks = _chunks()
    out1 = feature_summary(chunks, ["f1", "f2"])
    out2 = feature_summary(chunks, ["f1", "f2"])
    assert out1.equals(out2)  # same chunk order => bit-identical on same build


def test_feature_summary_requires_era_and_features() -> None:
    with pytest.raises(ValueError):
        feature_summary([pl.DataFrame({"era": ["0001"], "f1": [1.0]})], ["f1", "missing"])
    with pytest.raises(ValueError):
        feature_summary([pl.DataFrame({"x": [1.0]})], ["f1"])


from nmr.analysis import (
    FeatureCorrResult,
    cross_set_membership,
    feature_correlation_structure,
    within_set_redundancy,
)


def test_feature_correlation_structure_equal_era_weight() -> None:
    # era sizes differ: 0001 has 10 rows, 0002 has 4 rows; both have
    # f1~f2 near-perfectly correlated and f1~f3 anti-correlated
    rng = np.random.default_rng(9)

    def _era(era: str, n: int) -> pl.DataFrame:
        base = rng.normal(size=n)
        return pl.DataFrame(
            {
                "era": [era] * n,
                "f1": base,
                "f2": base + rng.normal(scale=0.01, size=n),
                "f3": -base,
            }
        )

    chunks = [_era("0001", 10), _era("0002", 4)]
    result = feature_correlation_structure(chunks, ["f1", "f2", "f3"])
    assert isinstance(result, FeatureCorrResult)
    assert result.matrix.shape == (3, 3)
    mat = result.matrix
    assert np.allclose(mat[0, 1], 1.0, atol=1e-3)
    assert np.allclose(mat[0, 2], -1.0, atol=1e-3)
    assert np.allclose(mat, mat.T, atol=1e-12)  # symmetric
    assert result.feature_order == ("f1", "f2", "f3")
    assert result.top_pairs.columns == ["feature_a", "feature_b", "mean_corr"]


def test_feature_correlation_structure_zero_variance_era() -> None:
    chunks = [
        pl.DataFrame(
            {"era": ["0001"] * 3, "f1": [1.0, 2.0, 3.0], "f2": [1.0, 1.0, 1.0]}
        ),
        pl.DataFrame(
            {"era": ["0002"] * 3, "f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]}
        ),
    ]
    result = feature_correlation_structure(chunks, ["f1", "f2"])
    # era 0001 f2 has zero variance -> 0.0 correlation; era 0002 -> ~1.0;
    # equal era weight -> ~0.5
    assert np.isclose(result.matrix[0, 1], 0.5, atol=1e-6)


def test_feature_correlation_structure_no_eras_raises() -> None:
    with pytest.raises(ValueError):
        feature_correlation_structure(
            [pl.DataFrame({"era": ["0001"], "f1": [1.0]})], ["f1"]
        )


def test_within_set_redundancy() -> None:
    rng = np.random.default_rng(13)
    chunks = [
        pl.DataFrame(
            {
                "era": [f"{e + 1:04d}"] * 8,
                "fa": rng.normal(size=8),
                "fb": rng.normal(size=8),
                "fc": rng.normal(size=8),
            }
        )
        for e in range(3)
    ]
    result = feature_correlation_structure(chunks, ["fa", "fb", "fc"])
    sets = {"pair": ["fa", "fb"], "solo": ["fa"], "all3": ["fa", "fb", "fc"]}
    out = within_set_redundancy(result, sets)
    assert out["feature_set"].to_list() == ["all3", "pair", "solo"]  # sorted
    row_solo = out.filter(pl.col("feature_set") == "solo").row(0, named=True)
    assert row_solo["n_pairs"] == 0
    assert row_solo["mean_abs_corr"] is None
    row_pair = out.filter(pl.col("feature_set") == "pair").row(0, named=True)
    assert row_pair["n_pairs"] == 1
    assert np.isclose(
        row_pair["mean_abs_corr"], float(np.abs(result.matrix[0, 1])), atol=1e-12
    )


def test_cross_set_membership_subset_relations() -> None:
    sets = {
        "small": ["a", "b"],
        "medium": ["a", "b", "c"],
        "all": ["a", "b", "c", "d"],
    }
    out = cross_set_membership(sets)
    assert out["sets"]["n_features"].to_list() == [4, 3, 2]  # name-sorted
    relations = out["subset_relations"]
    rel = {
        (r["a"], r["b"]): r["a_subset_of_b"] for r in relations.iter_rows(named=True)
    }
    assert rel[("small", "medium")] is True
    assert rel[("small", "all")] is True
    assert rel[("medium", "all")] is True
    assert rel[("all", "small")] is False


from nmr.analysis import (
    IC_VOL_WINDOW,
    REGIME_HIGH_PCT,
    REGIME_LOW_PCT,
    regime_analysis,
)


def _ic_by_era_series(n_eras: int = 30) -> pl.DataFrame:
    # era mean_ic ramps upward so quartile/decile bands are well-separated
    rows = []
    for e in range(n_eras):
        era = f"{e + 1:04d}"
        mean_ic = -0.05 + 0.10 * e / max(n_eras - 1, 1)
        for f, offset in [("fa", 0.01), ("fb", 0.0), ("fc", -0.01)]:
            rows.append({"era": era, "feature": f, "ic": float(mean_ic + offset)})
    return pl.DataFrame(rows)


def test_regime_analysis_bands_and_flags() -> None:
    out = regime_analysis(_ic_by_era_series(30))
    assert REGIME_LOW_PCT == 10.0
    assert REGIME_HIGH_PCT == 90.0
    assert IC_VOL_WINDOW == 20
    sig = out["era_signal"]
    assert "regime" in sig.columns and "crash" in sig.columns and "hot" in sig.columns
    first = sig.row(0, named=True)
    last = sig.row(sig.height - 1, named=True)
    assert first["regime"] == "low"
    assert first["crash"] is True
    assert last["regime"] == "high"
    assert last["hot"] is True
    th = out["regime_thresholds"]
    assert th["mean_ic_low"] <= th["q1"] <= th["q3"] <= th["mean_ic_high"]
    assert out["crash_eras"] == ["0001", "0002", "0003"]
    assert out["hot_eras"] == [f"{e:04d}" for e in range(28, 31)]


def test_regime_analysis_persistence_rank_stable_series() -> None:
    # feature IC ranks constant across eras -> adjacent Spearman = 1.0
    rows = []
    for e in range(5):
        era = f"{e + 1:04d}"
        for f, ic in [("fa", 0.1), ("fb", 0.05), ("fc", 0.0)]:
            rows.append({"era": era, "feature": f, "ic": ic})
    out = regime_analysis(pl.DataFrame(rows))
    assert np.isclose(out["ic_persistence"]["mean"], 1.0, atol=1e-12)
    assert out["ic_persistence"]["n_adjacent"] == 4


def test_regime_analysis_deterministic() -> None:
    ic = _ic_by_era_series(25)
    out1 = regime_analysis(ic)
    out2 = regime_analysis(ic)
    assert out1["era_signal"].equals(out2["era_signal"])
    assert out1["crash_eras"] == out2["crash_eras"]
    assert out1["ic_persistence"] == out2["ic_persistence"]


def test_regime_analysis_requires_columns() -> None:
    with pytest.raises(ValueError):
        regime_analysis(pl.DataFrame({"era": ["0001"], "ic": [0.1]}))


from nmr.analysis import benchmark_era_corr


def test_benchmark_era_corr_known_values() -> None:
    rng = np.random.default_rng(31)
    rows = []
    for e in range(3):
        era = f"{e + 1:04d}"
        for i in range(20):
            pred = float(rng.normal())
            rows.append(
                {
                    "era": era,
                    "id": f"{era}-{i}",
                    "benchmark_small": pred,
                    "benchmark_medium": 2.0 * pred + float(rng.normal(scale=0.1)),
                    "target": pred + float(rng.normal(scale=0.1)),
                }
            )
    frame = pl.DataFrame(rows)
    out = benchmark_era_corr(frame, ["benchmark_small", "benchmark_medium"], "target")
    summary = out["benchmarks"]
    assert summary["benchmark"].to_list() == ["benchmark_medium", "benchmark_small"]
    assert summary["n_eras"].to_list() == [3, 3]
    assert summary["first_era"].to_list() == ["0001", "0001"]
    assert summary["last_era"].to_list() == ["0003", "0003"]
    for row in summary.iter_rows(named=True):
        assert row["mean_corr"] > 0.5


def test_benchmark_era_corr_absent_degenerate_eras() -> None:
    # era 0001 has only 1 row -> degenerate -> absent from output
    frame = pl.DataFrame(
        {
            "era": ["0001", "0002", "0002", "0002", "0002"],
            "id": ["a", "b", "c", "d", "e"],
            "benchmark_small": [1.0, 1.0, 2.0, 3.0, 4.0],
            "target": [0.5, 1.0, 2.0, 3.0, 4.0],
        }
    )
    out = benchmark_era_corr(frame, ["benchmark_small"], "target")
    assert out["benchmarks"]["n_eras"].to_list() == [1]
    assert out["benchmarks"]["first_era"].to_list() == ["0002"]
    assert set(out["per_era"]["era"].to_list()) == {"0002"}


import json

import nmr
from nmr.config import REPO_ROOT


def test_analysis_symbols_exported() -> None:
    for name in [
        "SplitStats",
        "describe_splits",
        "era_structure",
        "target_profile",
        "target_correlation_matrix",
        "feature_ic_screen",
        "feature_ic_by_era",
        "feature_summary",
        "FeatureCorrResult",
        "feature_correlation_structure",
        "within_set_redundancy",
        "cross_set_membership",
        "regime_analysis",
        "benchmark_era_corr",
    ]:
        assert name in nmr.__all__, name
        assert hasattr(nmr, name), name


def test_real_feature_sets_nesting() -> None:
    """Cheap real-data guard (reads features.json only).

    Empirically (v5.3): medium is a subset of all, and small is a curated set
    that is a subset of all but NOT of medium (only 11/42 small features are
    in medium). Assert the true invariants; the report reports relations
    empirically via cross_set_membership.
    """
    features_json = REPO_ROOT / "data" / "v5.3" / "features.json"
    if not features_json.exists():
        pytest.skip("data/v5.3/features.json absent in this checkout")
    raw = json.loads(features_json.read_text(encoding="utf-8"))
    sets = raw["feature_sets"]
    small = set(sets["small"])
    medium = set(sets["medium"])
    all_ = set(sets["all"])
    assert medium <= all_
    assert small <= all_
    assert len(small & medium) < len(small)  # curated set, not a medium subset
