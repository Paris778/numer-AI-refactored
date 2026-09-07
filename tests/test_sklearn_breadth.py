from __future__ import annotations

import numpy as np

from nmr.sklearn_breadth import (
    EraPurgedSplit,
    build_estimator,
    discover_regressor_names,
)


def test_catalog_enumerates_public_sklearn_regressors() -> None:
    names = discover_regressor_names()

    assert len(names) >= 50
    assert names == tuple(sorted(set(names)))
    assert {"Ridge", "HistGradientBoostingRegressor", "SVR"}.issubset(names)


def test_era_purged_split_excludes_boundary_eras() -> None:
    eras = np.repeat([f"{value:04d}" for value in range(1, 41)], 3)
    splitter = EraPurgedSplit(tuple(eras), purge_eras=8, validation_fraction=0.2)

    train_idx, validation_idx = next(splitter.split(np.zeros((len(eras), 2))))
    train_eras = {eras[index] for index in train_idx}
    validation_eras = {eras[index] for index in validation_idx}

    assert train_eras.isdisjoint(validation_eras)
    assert max(map(int, train_eras)) + 8 < min(map(int, validation_eras))


def test_catalog_factories_cover_nested_and_multivariate_estimators() -> None:
    eras = tuple(f"{value:04d}" for value in range(1, 21) for _ in range(4))
    splitter = EraPurgedSplit(eras, purge_eras=2, validation_fraction=0.2)

    for name in (
        "CCA",
        "MultiOutputRegressor",
        "StackingRegressor",
        "VotingRegressor",
        "IsotonicRegression",
    ):
        estimator = build_estimator(name, seed=42, cv=splitter)
        assert estimator.__class__.__name__ == name
