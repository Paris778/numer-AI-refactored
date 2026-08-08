"""Phase 0: _per_era_pearson extraction — single source of truth for per-era IC."""

from __future__ import annotations

import numpy as np
import polars as pl

from nmr.features import _per_era_pearson, feature_stability_screen


def _frame() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    eras = ["0001", "0002", "0003", "0004"]
    rows: list[dict[str, float | str]] = []
    for e in eras:
        for _ in range(10):
            rows.append(
                {
                    "era": e,
                    "feature_alpha": float(rng.normal()),
                    "feature_beta": float(rng.normal()),
                    "target": float(rng.normal()),
                }
            )
    return pl.DataFrame(rows)


def test_per_era_pearson_shapes_and_values() -> None:
    frame = _frame()
    corrs, degenerate = _per_era_pearson(
        frame, ["feature_alpha", "feature_beta"], "target", "era"
    )
    assert set(corrs) == {"0001", "0002", "0003", "0004"}
    assert degenerate == set()
    for era, vec in corrs.items():
        assert vec.shape == (2,)
        part = frame.filter(pl.col("era") == era)
        a = part["feature_alpha"].cast(pl.Float64).to_numpy()
        t = part["target"].cast(pl.Float64).to_numpy()
        assert np.isclose(vec[0], np.corrcoef(a, t)[0, 1], atol=1e-12)


def test_per_era_pearson_degenerate_eras() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0002", "0002", "0002", "0002"],
            "feature_alpha": [1.0, 2.0, 1.0, 1.0, 1.0],
            "feature_beta": [3.0, 4.0, 5.0, 5.0, 5.0],
            "target": [0.1, 1.0, 1.0, 1.0, 1.0],
        }
    )
    corrs, degenerate = _per_era_pearson(
        frame, ["feature_alpha", "feature_beta"], "target", "era"
    )
    assert "0001" in degenerate  # single row -> <2 rows
    assert np.array_equal(corrs["0001"], np.zeros(2))
    assert "0002" in degenerate  # constant target -> zero variance
    assert np.array_equal(corrs["0002"], np.zeros(2))


def test_screen_uses_extracted_helper_single_source_of_truth() -> None:
    frame = _frame()
    screen = feature_stability_screen(
        frame, feature_cols=["feature_alpha", "feature_beta"], target_col="target"
    )
    corrs, _ = _per_era_pearson(
        frame, ["feature_alpha", "feature_beta"], "target", "era"
    )
    eras = sorted(corrs, key=int)
    matrix = np.column_stack([corrs[e] for e in eras])
    for i, feature in enumerate(["feature_alpha", "feature_beta"]):
        row = screen.filter(pl.col("feature") == feature)
        assert np.isclose(
            float(row["mean_corr"][0]), float(np.mean(matrix[i])), atol=1e-12
        )
        assert float(row["n_eras"][0]) == len(eras)
