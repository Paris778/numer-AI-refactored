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
    out = feature_ic_by_era(frame.partition_by("era", maintain_order=True), ["feature_alpha", "feature_beta"], "target")
    assert out.columns == ["era", "feature", "ic", "spearman_ic", "degenerate"]
    assert out.height == 4 * 2
    assert out["feature"].n_unique() == 2
    assert out["era"].n_unique() == 4
    corrs, _ = _per_era_pearson(frame, ["feature_alpha", "feature_beta"], "target", "era")
    for era, vec in corrs.items():
        rows = out.filter(pl.col("era") == era)
        assert np.array_equal(rows["ic"].to_numpy(), vec)


def test_feature_ic_by_era_spearman_monotone() -> None:
    # f = t**3 is strictly monotone in t -> per-era Spearman = 1.0 exactly,
    # while Pearson < 1.0.
    rng = np.random.default_rng(23)
    rows = []
    for e in range(3):
        era = f"{e + 1:04d}"
        for t in rng.normal(size=60):
            rows.append({"era": era, "feature_cubic": float(t**3), "target": float(t)})
    frame = pl.DataFrame(rows)
    out = feature_ic_by_era(
        frame.partition_by("era", maintain_order=True),
        ["feature_cubic"],
        "target",
    )
    assert np.allclose(out["spearman_ic"].to_numpy(), 1.0, atol=1e-12)
    assert np.all(out["ic"].to_numpy() < 1.0)
    assert not out["degenerate"].any()


def test_nonlinear_flag_predicate() -> None:
    from nmr.analysis import _nonlinear_flag

    mc = np.array([0.005, 0.02, np.nan, -0.005, 0.0])
    ms = np.array([0.05, 0.5, 0.5, 0.02, 0.0])
    assert _nonlinear_flag(mc, ms, 0.01).tolist() == [True, False, False, True, False]


def test_feature_ic_by_era_degenerate_flag() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0002", "0002", "0002", "0002"],
            "feature_alpha": [1.0, 2.0, 1.0, 1.0, 1.0],
            "feature_beta": [3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [0.1, 1.0, 1.0, 1.0, 1.0],
        }
    )
    out = feature_ic_by_era(frame.partition_by("era", maintain_order=True), ["feature_alpha", "feature_beta"], "target")
    assert out.filter(pl.col("era") == "0001")["degenerate"].all()  # <2 rows
    assert out.filter(pl.col("era") == "0002")["degenerate"].all()  # const target
    assert (out.filter(pl.col("era") == "0001")["ic"] == 0.0).all()  # zero vectors


def test_feature_ic_screen_multi_target() -> None:
    frame = _ic_frame()
    out = feature_ic_screen(frame.partition_by("era", maintain_order=True), ["feature_alpha", "feature_beta"], ["target"])
    assert out.columns == [
        "feature",
        "target",
        "mean_corr",
        "mean_corr_ci_lo",
        "mean_corr_ci_hi",
        "corr_std",
        "decay_slope",
        "cross_regime_variance",
        "mean_spearman",
        "n_eras",
        "stable",
        "nonlinear",
    ]
    assert out.height == 2
    assert out["target"].to_list() == ["target", "target"]
    # random features: no monotone-nonlinear signal -> flag False
    assert out["nonlinear"].to_list() == [False, False]
    assert np.allclose(out["mean_spearman"].to_numpy(), out["mean_corr"].to_numpy(), atol=0.05)


def test_feature_ic_screen_ci_columns_bracket_mean() -> None:
    # 24 eras so the block bootstrap has real resampling variation
    rng = np.random.default_rng(23)
    rows = []
    for e in range(24):
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
    frame = pl.DataFrame(rows)
    out = feature_ic_screen(
        frame.partition_by("era", maintain_order=True),
        ["feature_alpha", "feature_beta"],
        ["target"],
    )
    mean = out["mean_corr"].to_numpy()
    lo = out["mean_corr_ci_lo"].to_numpy()
    hi = out["mean_corr_ci_hi"].to_numpy()
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    # CI brackets the point estimate with strictly positive width
    assert (lo <= mean + 1e-12).all() and (mean - 1e-12 <= hi).all()
    assert (hi - lo > 0.0).all()


def test_feature_ic_screen_ci_deterministic() -> None:
    frame = _ic_frame()
    kwargs = dict(
        chunks=frame.partition_by("era", maintain_order=True),
        feature_cols=["feature_alpha", "feature_beta"],
        targets=["target"],
    )
    out1 = feature_ic_screen(**kwargs)
    out2 = feature_ic_screen(**kwargs)
    assert np.array_equal(
        out1["mean_corr_ci_lo"].to_numpy(), out2["mean_corr_ci_lo"].to_numpy()
    )
    assert np.array_equal(
        out1["mean_corr_ci_hi"].to_numpy(), out2["mean_corr_ci_hi"].to_numpy()
    )


def test_feature_ic_screen_empty_targets_raises() -> None:
    with pytest.raises(ValueError):
        feature_ic_screen(_ic_frame().partition_by("era", maintain_order=True), ["feature_alpha"], [])


from nmr.analysis import feature_ic_by_split


def test_feature_ic_by_split_means_and_delta() -> None:
    # fa: +0.01 in train, -0.02 in validation; fb: flat 0.0
    rows = []
    for era_int in range(1, 9):
        era = f"{era_int:04d}"
        split = "train" if era_int <= 4 else "validation"
        base = 0.01 if split == "train" else -0.02
        for f, offset in [("fa", 0.0), ("fb", 0.02)]:
            rows.append({"era": era, "feature": f, "ic": base + offset})
    out = feature_ic_by_split(pl.DataFrame(rows), train_max_era=4, val_min_era=5)
    fa = out.filter(pl.col("feature") == "fa").row(0, named=True)
    assert fa["train_n_eras"] == 4 and fa["val_n_eras"] == 4
    assert np.isclose(fa["train_mean_ic"], 0.01)
    assert np.isclose(fa["val_mean_ic"], -0.02)
    assert np.isclose(fa["delta_ic"], -0.03)


def test_feature_ic_by_split_excludes_degenerate_and_between_eras() -> None:
    rows = []
    for era_int in [1, 2, 3, 8]:
        era = f"{era_int:04d}"
        for f in ["fa", "fb"]:
            rows.append({"era": era, "feature": f, "ic": 0.001})
    # eras 0004-0005 sit between the split bounds; era 0007 is degenerate
    rows += [
        {"era": "0004", "feature": "fa", "ic": 0.5},
        {"era": "0004", "feature": "fb", "ic": 0.5},
        {"era": "0005", "feature": "fa", "ic": 0.5},
        {"era": "0005", "feature": "fb", "ic": 0.5},
        {"era": "0007", "feature": "fa", "ic": 0.0},
        {"era": "0007", "feature": "fb", "ic": 0.0},
    ]
    out = feature_ic_by_split(pl.DataFrame(rows), train_max_era=3, val_min_era=6)
    fa = out.filter(pl.col("feature") == "fa").row(0, named=True)
    assert fa["train_n_eras"] == 3  # 0001-0003 (0004-0005 between, 0007 degenerate)
    assert fa["val_n_eras"] == 1  # only 0008


def test_feature_ic_by_split_validates_inputs() -> None:
    with pytest.raises(ValueError):
        feature_ic_by_split(pl.DataFrame({"era": ["0001"], "ic": [0.1]}), 4, 5)
    with pytest.raises(ValueError, match="train_max_era"):
        feature_ic_by_split(
            pl.DataFrame({"era": ["0001"], "feature": ["fa"], "ic": [0.1]}), 5, 4
        )


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


from nmr.analysis import feature_drift_psi


def _psi_chunks(values: np.ndarray, n_eras: int = 3) -> list[pl.DataFrame]:
    rng = np.random.default_rng(0)
    chunks = []
    pos = 0
    for e in range(n_eras):
        n = len(values) // n_eras
        era = f"{e + 1:04d}"
        rows = [
            {"era": era, "f_shift": float(v), "f_const": 1.0}
            for v in values[pos : pos + n]
        ]
        pos += n
        chunks.append(pl.DataFrame(rows))
    return chunks


def test_feature_drift_psi_identical_distributions_zero() -> None:
    rng = np.random.default_rng(31)
    values = rng.normal(size=900)
    train = _psi_chunks(values, 3)
    val = _psi_chunks(values, 1)  # identical values -> identical histograms
    out = feature_drift_psi(train, val, ["f_shift", "f_const"], edge_sample_stride=1)
    row = out.filter(pl.col("feature") == "f_shift").row(0, named=True)
    assert row["psi"] == 0.0
    assert row["drifted"] is False


def test_feature_drift_psi_shifted_distribution_flagged() -> None:
    rng = np.random.default_rng(32)
    train = _psi_chunks(rng.normal(size=900), 3)
    val = _psi_chunks(rng.normal(loc=2.0, size=900), 1)
    out = feature_drift_psi(train, val, ["f_shift"], edge_sample_stride=1)
    row = out.row(0, named=True)
    assert row["psi"] > 0.25
    assert row["drifted"] is True


def test_feature_drift_psi_constant_feature_zero() -> None:
    rng = np.random.default_rng(33)
    train = _psi_chunks(rng.normal(size=900), 3)
    val = _psi_chunks(rng.normal(size=900), 1)
    out = feature_drift_psi(train, val, ["f_const"], edge_sample_stride=1)
    row = out.row(0, named=True)
    assert row["psi"] == 0.0
    assert row["drifted"] is False


def test_feature_drift_psi_schema_and_determinism() -> None:
    rng = np.random.default_rng(34)
    train = _psi_chunks(rng.normal(size=900), 3)
    val = _psi_chunks(rng.normal(loc=0.5, size=900), 1)
    out1 = feature_drift_psi(train, val, ["f_shift", "f_const"], edge_sample_stride=1)
    out2 = feature_drift_psi(train, val, ["f_shift", "f_const"], edge_sample_stride=1)
    assert out1.columns == ["feature", "psi", "n_train", "n_val", "drifted"]
    assert out1.equals(out2)
    assert out1["n_train"].to_list() == [900, 900]
    assert out1["n_val"].to_list() == [900, 900]


from nmr.analysis import feature_drift_profile


def test_feature_drift_profile_identical_distributions() -> None:
    rng = np.random.default_rng(41)
    values = rng.normal(size=900)
    train = _psi_chunks(values, 3)
    val = _psi_chunks(values, 1)  # identical values
    out = feature_drift_profile(train, val, ["f_shift", "f_const"], edge_sample_stride=1)
    row = out.filter(pl.col("feature") == "f_shift").row(0, named=True)
    assert row["psi"] == 0.0
    assert row["w1"] == 0.0
    assert abs(row["auc_roc"] - 0.5) < 1e-9  # no separation -> AUC 0.5
    assert row["drifted"] is False


def test_feature_drift_profile_shifted_distribution() -> None:
    rng = np.random.default_rng(42)
    train = _psi_chunks(rng.normal(size=900), 3)
    val = _psi_chunks(rng.normal(loc=2.0, size=900), 1)
    out = feature_drift_profile(train, val, ["f_shift"], edge_sample_stride=1)
    row = out.row(0, named=True)
    assert row["w1"] == pytest.approx(2.0, abs=0.05)  # W1 = mean shift for normals
    assert row["auc_roc"] > 0.9  # theoretical P(val > train) = Phi(2/sqrt(2)) ~ 0.921
    assert row["drifted"] is True


def test_feature_drift_profile_constant_feature_zero() -> None:
    rng = np.random.default_rng(43)
    train = _psi_chunks(rng.normal(size=900), 3)
    val = _psi_chunks(rng.normal(size=900), 1)
    out = feature_drift_profile(train, val, ["f_const"], edge_sample_stride=1)
    row = out.row(0, named=True)
    assert row["psi"] == 0.0
    assert row["w1"] == 0.0
    assert abs(row["auc_roc"] - 0.5) < 1e-9
    assert row["drifted"] is False


def test_feature_drift_profile_schema_and_determinism() -> None:
    rng = np.random.default_rng(44)
    train = _psi_chunks(rng.normal(size=900), 3)
    val = _psi_chunks(rng.normal(loc=0.3, size=900), 1)
    out1 = feature_drift_profile(train, val, ["f_shift", "f_const"], edge_sample_stride=1)
    out2 = feature_drift_profile(train, val, ["f_shift", "f_const"], edge_sample_stride=1)
    assert out1.columns == [
        "feature", "psi", "w1", "w1_norm", "auc_roc", "n_train", "n_val", "drifted",
    ]
    assert out1.equals(out2)


def test_feature_drift_profile_w1_norm_scale_invariance() -> None:
    # Bounded feature: raw W1 = 0.05 (~1 sigma of sigma=0.05) — below the old
    # raw threshold (0.25) but a real 1-sigma shift -> flagged once
    # standardized. Unbounded feature: raw W1 = 0.3 (~0.1 sigma of sigma=3) —
    # above the old raw threshold but statistical noise -> not flagged.
    rng = np.random.default_rng(45)
    base = rng.normal(size=900)
    rows = [
        {
            "era": "0001",
            "f_bounded": float(t * 0.05),
            "f_unbounded": float(t * 3.0),
        }
        for t in base
    ]
    val_rows = [
        {
            "era": "0002",
            "f_bounded": float(t * 0.05 + 0.05),
            "f_unbounded": float(t * 3.0 + 0.3),
        }
        for t in base
    ]
    out = feature_drift_profile(
        [pl.DataFrame(rows)],
        [pl.DataFrame(val_rows)],
        ["f_bounded", "f_unbounded"],
        edge_sample_stride=1,
    )
    bounded = out.filter(pl.col("feature") == "f_bounded").row(0, named=True)
    unbounded = out.filter(pl.col("feature") == "f_unbounded").row(0, named=True)
    assert bounded["w1"] == pytest.approx(0.05, abs=0.02)
    assert bounded["w1_norm"] == pytest.approx(1.0, abs=0.2)
    assert bounded["drifted"] is True
    assert unbounded["w1"] == pytest.approx(0.3, abs=0.1)
    assert unbounded["w1_norm"] == pytest.approx(0.1, abs=0.03)
    assert unbounded["drifted"] is False


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


def test_feature_corr_summary_min_eigenvalue() -> None:
    # f1~f2 near-perfect, f3 = -f1: matrix [[1,1,-1],[1,1,-1],[-1,-1,1]]
    # has eigenvalues 0, 0, 3 -> min_eigenvalue ~= 0.
    rng = np.random.default_rng(15)
    chunks = []
    for e in range(3):
        era = f"{e + 1:04d}"
        base = rng.normal(size=50)
        chunks.append(
            pl.DataFrame(
                {
                    "era": [era] * 50,
                    "f1": base,
                    "f2": base + rng.normal(scale=0.01, size=50),
                    "f3": -base,
                }
            )
        )
    result = feature_correlation_structure(chunks, ["f1", "f2", "f3"])
    assert "min_eigenvalue" in result.summary
    assert abs(result.summary["min_eigenvalue"]) < 1e-6


def test_feature_corr_top100_matches_full_argsort() -> None:
    rng = np.random.default_rng(16)
    chunks = [
        pl.DataFrame(
            {
                "era": [f"{e + 1:04d}"] * 8,
                **{f"f{i}": rng.normal(size=8) for i in range(20)},
            }
        )
        for e in range(3)
    ]
    result = feature_correlation_structure(chunks, [f"f{i}" for i in range(20)])
    mat = result.matrix.astype(np.float64)
    iu = np.triu_indices(20, k=1)
    abs_vals = np.abs(mat[iu])
    reference = np.argsort(abs_vals)[::-1][:100]
    got = [
        (r["feature_a"], r["feature_b"]) for r in result.top_pairs.iter_rows(named=True)
    ]
    expected = [
        (f"f{iu[0][k]}", f"f{iu[1][k]}") for k in reference
    ]
    assert got == expected


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


def test_regime_analysis_degenerate_era_unlabeled() -> None:
    # A zero-signal era (all-zero IC vector, e.g. label-lag) must be marked
    # unlabeled, excluded from crash/hot, and excluded from the percentiles.
    ic = _ic_by_era_series(30)
    zero_rows = pl.DataFrame(
        {
            "era": ["0031"] * 3,
            "feature": ["fa", "fb", "fc"],
            "ic": [0.0, 0.0, 0.0],
        }
    )
    out = regime_analysis(pl.concat([ic, zero_rows]))
    sig = out["era_signal"]
    row = sig.filter(pl.col("era") == "0031").row(0, named=True)
    assert row["regime"] == "unlabeled"
    assert row["crash"] is False and row["hot"] is False
    assert row["pct_rank"] is None
    assert row["degenerate"] is True
    assert "0031" not in out["crash_eras"]
    assert "0031" not in out["hot_eras"]
    # 30 valid eras -> bottom decile is exactly 3 eras, unshifted by the zero era
    assert out["crash_eras"] == ["0001", "0002", "0003"]


def test_regime_analysis_all_degenerate_raises() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001"] * 3,
            "feature": ["fa", "fb", "fc"],
            "ic": [0.0, 0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="no valid"):
        regime_analysis(frame)


def test_regime_analysis_respects_degenerate_column() -> None:
    # The degenerate column is authoritative when present (the script emits it
    # from feature_ic_by_era); a flagged era is unlabeled regardless of ICs.
    ic = _ic_by_era_series(6).with_columns(
        pl.when(pl.col("era") == "0004")
        .then(True)
        .otherwise(False)
        .alias("degenerate")
    )
    out = regime_analysis(ic)
    row = out["era_signal"].filter(pl.col("era") == "0004").row(0, named=True)
    assert row["regime"] == "unlabeled"
    assert row["crash"] is False
    # 5 valid eras -> bottom decile is exactly era 0001
    assert out["crash_eras"] == ["0001"]


def test_regime_analysis_requires_columns() -> None:
    with pytest.raises(ValueError):
        regime_analysis(pl.DataFrame({"era": ["0001"], "ic": [0.1]}))


from nmr.analysis import neutralized_ic_profile


def _fne_chunks(n_eras: int = 4, rows: int = 300) -> list[pl.DataFrame]:
    """Per-era: f1 (feature), linear_sig = f1 + noise, ortho_sig = w (noise),
    target = f1 + eps (correlates with f1 only)."""
    rng = np.random.default_rng(17)
    chunks = []
    for e in range(n_eras):
        era = f"{e + 1:04d}"
        f1 = rng.normal(size=rows)
        w = rng.normal(size=rows)
        chunks.append(
            pl.DataFrame(
                {
                    "era": [era] * rows,
                    "f1": f1,
                    "linear_sig": f1 + rng.normal(scale=0.1, size=rows),
                    "ortho_sig": w,
                    "target": f1 + rng.normal(scale=0.1, size=rows),
                }
            )
        )
    return chunks


def test_neutralized_ic_linear_signal_collapses() -> None:
    out = neutralized_ic_profile(
        _fne_chunks(), ["linear_sig"], ["f1"], "target"
    )
    row = {p: r for r in out.iter_rows(named=True) for p in [r["proportion"]]}
    ic0 = row[0.0]["mean_ic"]
    ic05 = row[0.5]["mean_ic"]
    ic1 = row[1.0]["mean_ic"]
    assert ic0 > 0.8  # raw signal IC is high
    assert abs(ic1) < 0.05  # fully neutralized: linear function of f1
    assert abs(ic05) < ic0  # monotone decay with proportion


def test_neutralized_ic_orthogonal_signal_preserved() -> None:
    rng = np.random.default_rng(19)
    chunks = []
    for e in range(4):
        era = f"{e + 1:04d}"
        f1 = rng.normal(size=300)
        w = rng.normal(size=300)
        chunks.append(
            pl.DataFrame(
                {
                    "era": [era] * 300,
                    "f1": f1,
                    "ortho_sig": w,
                    "target": w + rng.normal(scale=0.1, size=300),
                }
            )
        )
    out = neutralized_ic_profile(chunks, ["ortho_sig"], ["f1"], "target")
    rows = {r["proportion"]: r for r in out.iter_rows(named=True)}
    assert rows[0.0]["mean_ic"] > 0.8
    assert rows[1.0]["mean_ic"] > 0.8  # orthogonal signal survives neutralization
    assert abs(rows[1.0]["mean_ic"] - rows[0.0]["mean_ic"]) < 0.05


def test_neutralized_ic_profile_schema_and_determinism() -> None:
    chunks = _fne_chunks()
    out1 = neutralized_ic_profile(chunks, ["linear_sig", "ortho_sig"], ["f1"], "target")
    out2 = neutralized_ic_profile(chunks, ["linear_sig", "ortho_sig"], ["f1"], "target")
    assert out1.columns == ["signal", "proportion", "mean_ic", "corr_std", "n_eras"]
    assert out1.height == 2 * 3
    assert (out1["n_eras"] == 4).all()
    assert out1.equals(out2)


def test_neutralized_ic_profile_skips_degenerate_eras() -> None:
    chunks = _fne_chunks(3)
    bad = pl.DataFrame(
        {
            "era": ["0004"] * 20,
            "f1": [1.0] * 20,
            "linear_sig": [2.0] * 20,
            "ortho_sig": [3.0] * 20,
            "target": [float("nan")] * 20,
        }
    )
    out = neutralized_ic_profile(
        chunks + [bad], ["linear_sig"], ["f1"], "target"
    )
    assert (out["n_eras"] == 3).all()


def test_neutralized_ic_profile_validates_inputs() -> None:
    with pytest.raises(ValueError):
        neutralized_ic_profile(_fne_chunks(), [], ["f1"], "target")
    with pytest.raises(ValueError, match="proportions"):
        neutralized_ic_profile(
            _fne_chunks(), ["linear_sig"], ["f1"], "target", proportions=(0.0, 1.5)
        )


from nmr.analysis import meta_orthogonality


def _meta_chunks(n_eras: int = 4, rows: int = 300) -> list[pl.DataFrame]:
    """Per era: meta = m, target = m + w, f_ortho = w (orthogonal to meta),
    f_meta = m (fully captured by the consensus)."""
    rng = np.random.default_rng(51)
    chunks = []
    for e in range(n_eras):
        era = f"{e + 1:04d}"
        m = rng.normal(size=rows)
        w = rng.normal(size=rows)
        chunks.append(
            pl.DataFrame(
                {
                    "era": [era] * rows,
                    "meta": m,
                    "target": m + w,
                    "f_ortho": w,
                    "f_meta": m,
                }
            )
        )
    return chunks


def test_meta_orthogonality_orthogonal_feature_flagged() -> None:
    out = meta_orthogonality(
        _meta_chunks(), ["f_ortho", "f_meta"], meta_col="meta", target_col="target"
    )
    ortho = out.filter(pl.col("feature") == "f_ortho").row(0, named=True)
    meta_f = out.filter(pl.col("feature") == "f_meta").row(0, named=True)
    assert abs(ortho["corr_meta"]) < 0.05  # independent of the consensus
    assert ortho["corr_target"] > 0.6  # but carries signal (corr(w, m+w) = 1/sqrt(2))
    assert ortho["orthogonal"] is True
    assert meta_f["corr_meta"] > 0.99  # fully captured by the consensus
    assert meta_f["orthogonal"] is False


def test_meta_orthogonality_schema_and_determinism() -> None:
    chunks = _meta_chunks()
    out1 = meta_orthogonality(chunks, ["f_ortho", "f_meta"], "meta", "target")
    out2 = meta_orthogonality(chunks, ["f_ortho", "f_meta"], "meta", "target")
    assert out1.columns == [
        "feature", "corr_meta", "corr_target", "n_eras", "orthogonal",
    ]
    assert out1.equals(out2)
    assert (out1["n_eras"] == 4).all()


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
        "feature_ic_by_split",
        "feature_summary",
        "feature_drift_psi",
        "feature_drift_profile",
        "meta_orthogonality",
        "FeatureCorrResult",
        "feature_correlation_structure",
        "within_set_redundancy",
        "cross_set_membership",
        "regime_analysis",
        "neutralized_ic_profile",
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


def test_regime_analysis_persistence_ignores_degenerate_eras() -> None:
    # era 0003 is degenerate (all-zero ICs) -> its constant rank vector must
    # not produce NaN persistence between 0002 and 0004.
    frame = pl.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002", "0003", "0003", "0004", "0004"],
            "feature": ["fa", "fb"] * 4,
            "ic": [0.1, 0.0, 0.05, 0.02, 0.0, 0.0, 0.05, 0.02],
        }
    )
    out = regime_analysis(frame)
    pers = out["ic_persistence"]
    assert np.isfinite(pers["mean"])
    assert pers["n_adjacent"] >= 1


def test_sorted_era_labels_strict_padding() -> None:
    from nmr.evaluation import sorted_era_labels

    # consistent (even mixed-width but unique ints) -> chronological
    assert sorted_era_labels(["0583", "0575", "1000"]) == ["0575", "0583", "1000"]
    # same int index with two string representations -> fail loud
    with pytest.raises(ValueError, match="zero-padding"):
        sorted_era_labels(["0575", "575"])
    with pytest.raises(ValueError, match="Non-numeric"):
        sorted_era_labels(["0575", "X"])
