# ruff: noqa: E402 — thread-pool limits must run before numerical imports.
"""Breadth benchmark for the public scikit-learn regression catalog.

The runner deliberately treats the sklearn catalog as an experiment roster,
not as a production model registry.  Every public sklearn regressor is
attempted on the NumerAI medium feature universe, with a chronological,
era-purged split supplied to estimators that perform internal cross-validation.
Each method runs in its own subprocess so a resource-heavy estimator can be
timed out and recorded without losing the completed results before it.
"""

from __future__ import annotations

# Thread-pool limits must be applied before numpy, polars, or sklearn import.
from nmr.hardware import apply_thread_limits

apply_thread_limits()

import argparse
import dataclasses
import hashlib
import inspect
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import all_estimators

from nmr._atomicio import atomic_write_text
from nmr.benchmark import train_validation_purged_split
from nmr.evaluation import EvaluationEngine
from nmr.features import resolve_feature_sets
from nmr.hardware import discover_hardware

logger = logging.getLogger("nmr.sklearn_breadth")

__all__ = [
    "BreadthConfig",
    "EraPurgedSplit",
    "EstimatorSpec",
    "build_estimator",
    "discover_regressor_names",
    "run_sweep",
]

_DEFAULT_MAX_TRAIN_ROWS = 250_000
_DEFAULT_MAX_VALIDATION_ROWS = 300_000
_DEFAULT_TIMEOUT_SECONDS = 300
_DEFAULT_PURGE_ERAS = 8
_MULTI_TARGET_METHODS = {
    "CCA",
    "PLSCanonical",
    "PLSRegression",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
}
_REQUIRES_MULTI_TARGET = {"MultiOutputRegressor", "RegressorChain"}
_SECONDARY_TARGET = "target_ender_20"
_UNIVARIATE_METHODS = {"IsotonicRegression"}
_METHOD_ROW_CAPS: dict[str, int] = {
    "GaussianProcessRegressor": 2_000,
    "KernelRidge": 8_000,
    "KNeighborsRegressor": 20_000,
    "NuSVR": 8_000,
    "QuantileRegressor": 50_000,
    "RadiusNeighborsRegressor": 20_000,
    "SVR": 8_000,
    "TheilSenRegressor": 10_000,
}
_CV_METHODS = {
    "ElasticNetCV",
    "LarsCV",
    "LassoCV",
    "LassoLarsCV",
    "MultiTaskElasticNetCV",
    "MultiTaskLassoCV",
    "OrthogonalMatchingPursuitCV",
    "RidgeCV",
    "RegressorChain",
}


@dataclasses.dataclass(frozen=True)
class EstimatorSpec:
    """Catalog metadata used to build and scope one sklearn estimator."""

    name: str
    estimator_cls: type[BaseEstimator]
    input_mode: str = "multivariate"
    method_row_cap: int | None = None


@dataclasses.dataclass(frozen=True)
class BreadthConfig:
    """One reproducible breadth-sweep invocation."""

    data_dir: Path
    output: Path
    scope: str = "screen"
    max_train_rows: int | None = _DEFAULT_MAX_TRAIN_ROWS
    max_validation_rows: int | None = _DEFAULT_MAX_VALIDATION_ROWS
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    purge_eras: int = _DEFAULT_PURGE_ERAS
    seed: int = 42
    score_window: str = "meta"
    compute_fnc: bool = False
    methods: tuple[str, ...] = ()
    workspace: Path | None = None
    resume: bool = True

    def __post_init__(self) -> None:
        if self.scope not in ("screen", "full"):
            raise ValueError(f"scope must be 'screen' or 'full', got {self.scope!r}")
        for name in ("max_train_rows", "max_validation_rows"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive int or None, got {value!r}"
                )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds < 1
        ):
            raise ValueError("timeout_seconds must be a positive int")
        if (
            isinstance(self.purge_eras, bool)
            or not isinstance(self.purge_eras, int)
            or self.purge_eras < 0
        ):
            raise ValueError("purge_eras must be a non-negative int")
        if self.score_window not in ("meta", "validation"):
            raise ValueError(
                f"score_window must be 'meta' or 'validation', got {self.score_window!r}"
            )


class EraPurgedSplit:
    """One deterministic chronological CV split with an era purge buffer."""

    def __init__(
        self,
        eras: Sequence[str],
        *,
        purge_eras: int = _DEFAULT_PURGE_ERAS,
        validation_fraction: float = 0.2,
    ) -> None:
        if not eras:
            raise ValueError("eras must be non-empty")
        if not 0.0 < float(validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if (
            isinstance(purge_eras, bool)
            or not isinstance(purge_eras, int)
            or purge_eras < 0
        ):
            raise ValueError("purge_eras must be a non-negative int")
        self._eras = tuple(str(era) for era in eras)
        self._purge_eras = purge_eras
        self._validation_fraction = float(validation_fraction)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        del X, y, groups
        return 1

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        del y, groups
        if len(X) != len(self._eras):
            raise ValueError(
                f"X row count {len(X)} does not match era count {len(self._eras)}"
            )
        ordered = sorted(set(self._eras), key=int)
        validation_count = max(
            1, int(math.ceil(len(ordered) * self._validation_fraction))
        )
        validation_start = len(ordered) - validation_count
        train_end = validation_start - self._purge_eras
        if train_end < 1:
            raise ValueError(
                "not enough eras for a purged CV split: "
                f"n_eras={len(ordered)}, purge={self._purge_eras}"
            )
        train_eras = set(ordered[:train_end])
        validation_eras = set(ordered[validation_start:])
        era_array = np.asarray(self._eras, dtype=object)
        train_index = np.flatnonzero(np.isin(era_array, list(train_eras)))
        validation_index = np.flatnonzero(np.isin(era_array, list(validation_eras)))
        if not len(train_index) or not len(validation_index):
            raise ValueError("purged CV split produced an empty partition")
        yield train_index, validation_index


def _catalog() -> tuple[EstimatorSpec, ...]:
    return tuple(
        EstimatorSpec(
            name=name,
            estimator_cls=cls,
            input_mode="univariate" if name in _UNIVARIATE_METHODS else "multivariate",
            method_row_cap=_METHOD_ROW_CAPS.get(name),
        )
        for name, cls in sorted(
            all_estimators(type_filter="regressor"), key=lambda item: item[0]
        )
    )


def discover_regressor_names() -> tuple[str, ...]:
    """Return all public sklearn regressors visible in the selected environment."""
    return tuple(spec.name for spec in _catalog())


def _ridge(seed: int) -> Ridge:
    return Ridge(alpha=1.0, solver="lsqr", tol=1.0e-3, random_state=seed)


def _hist(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=60,
        max_leaf_nodes=31,
        early_stopping=False,
        random_state=seed,
    )


def _set_if_supported(
    kwargs: dict[str, Any], parameters: set[str], name: str, value: Any
) -> None:
    if name in parameters:
        kwargs[name] = value


def build_estimator(
    name: str,
    *,
    seed: int,
    cv: EraPurgedSplit | None = None,
) -> BaseEstimator:
    """Build a bounded, deterministic estimator configuration by catalog name.

    sklearn CV defaults are never used: any estimator exposing ``cv`` receives
    the era-purged splitter supplied by the worker. Nested regressors use the
    same conservative Ridge/HistGradientBoosting building blocks.
    """
    by_name = {spec.name: spec.estimator_cls for spec in _catalog()}
    if name not in by_name:
        raise ValueError(f"unknown sklearn regressor {name!r}")
    estimator_cls = by_name[name]
    parameters = set(inspect.signature(estimator_cls).parameters)
    kwargs: dict[str, Any] = {}
    _set_if_supported(kwargs, parameters, "random_state", seed)
    _set_if_supported(kwargs, parameters, "n_jobs", -1)
    _set_if_supported(kwargs, parameters, "verbose", False)

    if name == "AdaBoostRegressor":
        kwargs.update(
            estimator=DecisionTreeRegressor(max_depth=4, random_state=seed),
            n_estimators=50,
            learning_rate=0.05,
        )
    elif name == "BaggingRegressor":
        kwargs.update(
            estimator=DecisionTreeRegressor(max_depth=8, random_state=seed),
            n_estimators=25,
            max_samples=0.5,
            max_features=0.5,
        )
    elif name in {"RandomForestRegressor", "ExtraTreesRegressor"}:
        kwargs.update(n_estimators=50, max_depth=12, max_features=0.5)
    elif name == "GradientBoostingRegressor":
        kwargs.update(
            n_estimators=100, learning_rate=0.05, max_depth=3, max_features=0.5
        )
    elif name == "HistGradientBoostingRegressor":
        kwargs.update(max_iter=60, max_leaf_nodes=31, early_stopping=False)
    elif name == "MLPRegressor":
        kwargs.update(
            hidden_layer_sizes=(128,),
            max_iter=30,
            batch_size=1024,
            early_stopping=False,
            shuffle=False,
        )
    elif name == "SGDRegressor":
        kwargs.update(max_iter=30, early_stopping=False, shuffle=False)
    elif name == "PassiveAggressiveRegressor":
        kwargs.update(max_iter=30, early_stopping=False, shuffle=False)
    elif name == "LinearSVR":
        kwargs.update(max_iter=100, dual="auto")
    elif name in {"SVR", "NuSVR"}:
        kwargs.update(kernel="rbf", max_iter=1_000, cache_size=2_048)
    elif name == "KernelRidge":
        kwargs.update(kernel="linear")
    elif name == "GaussianProcessRegressor":
        kwargs.update(optimizer=None, normalize_y=True, copy_X_train=False)
    elif name == "TheilSenRegressor":
        kwargs.update(max_subpopulation=1_000, max_iter=50)
    elif name == "QuantileRegressor":
        kwargs.update(alpha=0.1, quantile=0.5)
    elif name in {"Lasso", "ElasticNet", "MultiTaskLasso", "MultiTaskElasticNet"}:
        kwargs.update(alpha=1.0e-4, max_iter=100)
    elif name in {
        "LassoCV",
        "ElasticNetCV",
        "MultiTaskLassoCV",
        "MultiTaskElasticNetCV",
    }:
        kwargs.update(alphas=np.logspace(-4, 1, 12), max_iter=100)
    elif name == "Lars":
        kwargs.update(n_nonzero_coefs=100, precompute=False)
    elif name == "LassoLars":
        kwargs.update(max_iter=100, precompute=False)
    elif name == "OrthogonalMatchingPursuit":
        kwargs.update(n_nonzero_coefs=100, precompute=False)
    elif name in {"LarsCV", "LassoLarsCV"}:
        kwargs.update(max_iter=100, precompute=False)
    elif name == "OrthogonalMatchingPursuitCV":
        kwargs.update(max_iter=100)
    elif name in {"ARDRegression", "BayesianRidge", "HuberRegressor"}:
        kwargs.update(max_iter=100)
    elif name == "Ridge":
        kwargs.update(alpha=1.0, solver="lsqr", tol=1.0e-3)
    elif name == "RidgeCV":
        kwargs.update(alphas=np.logspace(-3, 3, 9))
    elif name in {"CCA", "PLSCanonical", "PLSRegression"}:
        kwargs.update(n_components=1, max_iter=100)

    if name in _CV_METHODS:
        if cv is None:
            raise ValueError(f"{name} requires an EraPurgedSplit instance")
        kwargs["cv"] = cv

    if name == "MultiOutputRegressor":
        kwargs["estimator"] = _ridge(seed)
    elif name == "RegressorChain":
        kwargs["estimator"] = _ridge(seed)
    elif name == "RANSACRegressor":
        kwargs.update(estimator=_ridge(seed), max_trials=10, min_samples=0.01)
    elif name == "TransformedTargetRegressor":
        kwargs["regressor"] = _ridge(seed)
    elif name in {"StackingRegressor", "VotingRegressor"}:
        base_estimators = [("ridge", _ridge(seed)), ("hist", _hist(seed))]
        kwargs["estimators"] = base_estimators
        if name == "StackingRegressor":
            if cv is None:
                raise ValueError(
                    "StackingRegressor requires an EraPurgedSplit instance"
                )
            kwargs["final_estimator"] = _ridge(seed)
            kwargs["cv"] = cv

    return estimator_cls(**kwargs)


@dataclasses.dataclass(frozen=True)
class _PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    fit_targets: np.ndarray
    eras_train: tuple[str, ...]
    X_validation: np.ndarray
    y_validation: np.ndarray
    eras_validation: tuple[str, ...]
    feature_count: int
    input_feature_count: int
    train_rows_before_filter: int
    validation_rows_before_filter: int
    load_seconds: float
    validation_ids: tuple[str, ...] = ()
    validation_payout_target: np.ndarray | None = None
    validation_feature_names: tuple[str, ...] = ()
    validation_scorecard_features: np.ndarray | None = None
    validation_scorecard_targets: dict[str, np.ndarray] = dataclasses.field(
        default_factory=dict
    )


def _era_labels(path: Path) -> tuple[str, ...]:
    return tuple(
        str(value)
        for value in pl.scan_parquet(path)
        .select("era")
        .unique()
        .collect()
        .get_column("era")
        .to_list()
    )


def _score_eras(
    data_dir: Path, validation_eras: Sequence[str], score_window: str
) -> tuple[str, ...]:
    if score_window == "validation":
        return tuple(sorted(validation_eras, key=int))
    meta_path = data_dir / "meta_model.parquet"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta score window requires {meta_path}")
    meta_eras = set(_era_labels(meta_path))
    overlap = sorted(meta_eras & set(validation_eras), key=int)
    if not overlap:
        raise ValueError("meta score window has no validation-era overlap")
    return tuple(overlap)


def _collect_rows(
    path: Path,
    *,
    columns: Sequence[str],
    eras: Sequence[str],
    row_limit: int | None,
    tail: bool,
) -> pl.DataFrame:
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("era").is_in(list(eras)))
        .select(list(columns))
        .collect()
    )
    if row_limit is not None and frame.height > row_limit:
        frame = frame.tail(row_limit) if tail else frame.head(row_limit)
    return frame


def _standardize(
    train_values: np.ndarray, validation_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train = np.ascontiguousarray(train_values, dtype=np.float32)
    validation = np.ascontiguousarray(validation_values, dtype=np.float32)
    mean = np.mean(train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(train, axis=0, dtype=np.float64)
    scale = np.where((std > 0.0) & np.isfinite(std), 1.0 / std, 0.0).astype(np.float32)
    np.subtract(train, mean, out=train)
    np.multiply(train, scale, out=train)
    np.subtract(validation, mean, out=validation)
    np.multiply(validation, scale, out=validation)
    return train, validation


def _prepare_data(
    *,
    data_dir: Path,
    method: EstimatorSpec,
    max_train_rows: int | None,
    max_validation_rows: int | None,
    purge_eras: int,
    score_window: str,
    target_col: str = "target",
    payout_target_col: str | None = None,
    scorecard_target_cols: Sequence[str] = (),
) -> _PreparedData:
    started = time.perf_counter()
    train_path = data_dir / "train.parquet"
    validation_path = data_dir / "validation.parquet"
    features_path = data_dir / "features.json"
    feature_sets = resolve_feature_sets(features_path)
    feature_cols = list(feature_sets["medium"])
    schema = pl.read_parquet_schema(train_path)
    missing = sorted(set(feature_cols) - set(schema.names()))
    if missing:
        raise ValueError(f"medium feature columns missing from train: {missing[:5]}")
    if not isinstance(target_col, str) or not target_col:
        raise ValueError("target_col must be a non-empty string")
    if payout_target_col is not None and payout_target_col not in schema.names():
        raise ValueError(
            f"payout target {payout_target_col!r} is missing from {train_path}"
        )
    if payout_target_col is not None and "id" not in schema.names():
        raise ValueError(f"payout evaluation requires an id column in {train_path}")
    scorecard_targets = tuple(dict.fromkeys(scorecard_target_cols))
    missing_scorecard_targets = sorted(set(scorecard_targets) - set(schema.names()))
    if missing_scorecard_targets:
        raise ValueError(
            f"scorecard targets missing from {train_path}: {missing_scorecard_targets}"
        )
    needs_scorecard_inputs = payout_target_col is not None or bool(scorecard_targets)
    if needs_scorecard_inputs and "id" not in schema.names():
        raise ValueError(f"scorecard evaluation requires an id column in {train_path}")
    fit_target_cols = [target_col]
    if method.name in _REQUIRES_MULTI_TARGET:
        if _SECONDARY_TARGET not in schema.names():
            raise ValueError(
                f"{method.name} requires secondary target {_SECONDARY_TARGET!r}"
            )
        if _SECONDARY_TARGET not in fit_target_cols:
            fit_target_cols.append(_SECONDARY_TARGET)

    all_train_eras = _era_labels(train_path)
    all_validation_eras = _era_labels(validation_path)
    trimmed_train_eras, validation_eras = train_validation_purged_split(
        all_train_eras,
        all_validation_eras,
        purge_eras=purge_eras,
    )
    selected_validation_eras = _score_eras(data_dir, validation_eras, score_window)
    method_limit = method.method_row_cap
    if max_train_rows is None:
        train_limit = method_limit
    elif method_limit is None:
        train_limit = max_train_rows
    else:
        train_limit = min(max_train_rows, method_limit)
    train = _collect_rows(
        train_path,
        columns=["era", *feature_cols, *fit_target_cols],
        eras=trimmed_train_eras,
        row_limit=train_limit,
        tail=True,
    )
    validation_columns = ["era", *feature_cols, *fit_target_cols]
    if needs_scorecard_inputs:
        validation_columns.insert(1, "id")
    for target_name in (*scorecard_targets, payout_target_col):
        if target_name is not None and target_name not in validation_columns:
            validation_columns.append(target_name)
    validation = _collect_rows(
        validation_path,
        columns=validation_columns,
        eras=selected_validation_eras,
        row_limit=max_validation_rows,
        tail=False,
    )
    train_rows_before_filter = train.height
    validation_rows_before_filter = validation.height

    train_values = train.select(feature_cols).cast(pl.Float32).to_numpy()
    validation_values = validation.select(feature_cols).cast(pl.Float32).to_numpy()
    train_target = train.get_column(target_col).cast(pl.Float64).to_numpy()
    validation_target = validation.get_column(target_col).cast(pl.Float64).to_numpy()
    validation_payout_target = (
        validation.get_column(payout_target_col).cast(pl.Float64).to_numpy()
        if payout_target_col is not None
        else None
    )
    scorecard_target_values = {
        target_name: validation.get_column(target_name).cast(pl.Float64).to_numpy()
        for target_name in scorecard_targets
    }
    fit_targets = np.column_stack(
        [
            train.get_column(column).cast(pl.Float64).to_numpy()
            for column in fit_target_cols
        ]
    )
    train_mask = np.isfinite(fit_targets).all(axis=1) & np.isfinite(train_values).all(
        axis=1
    )
    validation_mask = np.isfinite(validation_target) & np.isfinite(
        validation_values
    ).all(axis=1)
    if train_mask.sum() < 2 or validation_mask.sum() < 2:
        raise ValueError(
            "finite train and validation rows must each contain at least 2 rows"
        )
    train_values = train_values[train_mask]
    validation_values = validation_values[validation_mask]
    train_target = train_target[train_mask]
    fit_targets = fit_targets[train_mask]
    validation_target = validation_target[validation_mask]
    if validation_payout_target is not None:
        validation_payout_target = validation_payout_target[validation_mask]
    scorecard_target_values = {
        target_name: values[validation_mask]
        for target_name, values in scorecard_target_values.items()
    }
    train_era_values = tuple(
        str(value) for value in train.get_column("era").to_numpy()[train_mask]
    )
    validation_era_values = tuple(
        str(value) for value in validation.get_column("era").to_numpy()[validation_mask]
    )
    validation_ids = (
        tuple(
            str(value)
            for value in validation.get_column("id").to_numpy()[validation_mask]
        )
        if payout_target_col is not None
        else ()
    )
    train_values, validation_values = _standardize(train_values, validation_values)
    scorecard_features = validation_values.copy()
    input_feature_count = train_values.shape[1]
    if method.input_mode == "univariate":
        train_values = train_values[:, :1]
        validation_values = validation_values[:, :1]
    return _PreparedData(
        X_train=train_values,
        y_train=train_target,
        fit_targets=fit_targets if len(fit_target_cols) > 1 else fit_targets[:, 0],
        eras_train=train_era_values,
        X_validation=validation_values,
        y_validation=validation_target,
        eras_validation=validation_era_values,
        feature_count=len(feature_cols),
        input_feature_count=input_feature_count,
        train_rows_before_filter=train_rows_before_filter,
        validation_rows_before_filter=validation_rows_before_filter,
        load_seconds=round(time.perf_counter() - started, 6),
        validation_ids=validation_ids,
        validation_payout_target=validation_payout_target,
        validation_feature_names=tuple(feature_cols),
        validation_scorecard_features=scorecard_features,
        validation_scorecard_targets=scorecard_target_values,
    )


def _rss_gib() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024**3), 4)


def _fit_target(method_name: str, target: np.ndarray) -> np.ndarray:
    if method_name == "GammaRegressor":
        return np.maximum(target, np.finfo(float).eps)
    if method_name in _REQUIRES_MULTI_TARGET:
        return target
    if method_name in _MULTI_TARGET_METHODS:
        return target.reshape(-1, 1)
    return target


def _predict_vector(
    method_name: str, estimator: BaseEstimator, X: np.ndarray
) -> np.ndarray:
    if method_name == "IsotonicRegression":
        raw = estimator.predict(X[:, 0])
    else:
        raw = estimator.predict(X)
    values = np.asarray(raw, dtype=float)
    if values.ndim == 2:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(
            f"{method_name} returned predictions with shape {values.shape}"
        )
    return values


def _score_predictions(
    data: _PreparedData,
    predictions: np.ndarray,
    *,
    compute_fnc: bool,
) -> dict[str, Any]:
    engine = EvaluationEngine()
    frame = pl.DataFrame(
        {
            "era": data.eras_validation,
            "prediction": predictions,
            "target": data.y_validation,
        }
    )
    corr_by_era = engine.per_era_corr(
        frame,
        pred_col="prediction",
        target_col="target",
    )
    corr_summary = engine.summarize(corr_by_era)
    result: dict[str, Any] = {
        "corr_mean": corr_summary.mean,
        "corr_std": corr_summary.std,
        "corr_sharpe": corr_summary.sharpe,
        "corr_max_drawdown": corr_summary.max_drawdown,
        "corr_n_eras": len(corr_by_era),
        "fnc_mean": None,
        "fnc_std": None,
        "fnc_sharpe": None,
        "fnc_n_eras": None,
    }
    if not compute_fnc:
        return result
    columns: dict[str, Any] = {
        "era": data.eras_validation,
        "prediction": predictions,
        "target": data.y_validation,
    }
    feature_cols: list[str] = []
    for index in range(data.X_validation.shape[1]):
        name = f"feature_{index:04d}"
        feature_cols.append(name)
        columns[name] = data.X_validation[:, index]
    fnc_frame = pl.DataFrame(columns)
    fnc_by_era = engine.per_era_fnc(
        fnc_frame,
        pred_col="prediction",
        feature_cols=feature_cols,
        target_col="target",
    )
    fnc_summary = engine.summarize(fnc_by_era)
    result.update(
        fnc_mean=fnc_summary.mean,
        fnc_std=fnc_summary.std,
        fnc_sharpe=fnc_summary.sharpe,
        fnc_n_eras=len(fnc_by_era),
    )
    return result


def _serializable_params(estimator: BaseEstimator) -> str:
    try:
        params = estimator.get_params(deep=False)
        return json.dumps(params, sort_keys=True, default=_json_default)
    except Exception as exc:  # pragma: no cover - defensive reporting only
        return json.dumps({"parameter_error": str(exc)}, sort_keys=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def _worker_result_base(method: EstimatorSpec, *, scope: str) -> dict[str, Any]:
    hardware = discover_hardware()
    return {
        "method": method.name,
        "run_identity": None,
        "estimator_class": method.estimator_cls.__name__,
        "status": "running",
        "scope": scope,
        "input_mode": method.input_mode,
        "method_row_cap": method.method_row_cap,
        "gpu_available": bool(hardware.gpus),
        "gpu_used": False,
        "gpu_reason": "scikit-learn regressors have no CUDA execution path",
        "load_seconds": None,
        "fit_seconds": None,
        "predict_seconds": None,
        "score_seconds": None,
        "total_seconds": None,
        "timeout_seconds": None,
        "peak_rss_gib": None,
        "train_rows": None,
        "validation_rows": None,
        "train_rows_before_filter": None,
        "validation_rows_before_filter": None,
        "train_eras": None,
        "validation_eras": None,
        "feature_count": None,
        "input_feature_count": None,
        "corr_mean": None,
        "corr_std": None,
        "corr_sharpe": None,
        "corr_max_drawdown": None,
        "corr_n_eras": None,
        "fnc_mean": None,
        "fnc_std": None,
        "fnc_sharpe": None,
        "fnc_n_eras": None,
        "estimator_params": None,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }


def run_worker(
    *,
    data_dir: Path,
    method_name: str,
    result_path: Path,
    scope: str,
    max_train_rows: int | None,
    max_validation_rows: int | None,
    purge_eras: int,
    seed: int,
    score_window: str,
    compute_fnc: bool,
    run_identity: str,
) -> int:
    methods = {spec.name: spec for spec in _catalog()}
    if method_name not in methods:
        raise ValueError(f"unknown sklearn method {method_name!r}")
    method = methods[method_name]
    started = time.perf_counter()
    result = _worker_result_base(method, scope=scope)
    result["run_identity"] = run_identity
    try:
        data = _prepare_data(
            data_dir=data_dir,
            method=method,
            max_train_rows=max_train_rows,
            max_validation_rows=max_validation_rows,
            purge_eras=purge_eras,
            score_window=score_window,
        )
        result.update(
            load_seconds=data.load_seconds,
            train_rows=int(data.X_train.shape[0]),
            validation_rows=int(data.X_validation.shape[0]),
            train_rows_before_filter=data.train_rows_before_filter,
            validation_rows_before_filter=data.validation_rows_before_filter,
            train_eras=len(set(data.eras_train)),
            validation_eras=len(set(data.eras_validation)),
            feature_count=data.feature_count,
            input_feature_count=data.input_feature_count,
        )
        cv = EraPurgedSplit(data.eras_train, purge_eras=purge_eras)
        estimator = build_estimator(method_name, seed=seed, cv=cv)
        result["estimator_params"] = _serializable_params(estimator)
        fit_started = time.perf_counter()
        fit_X = (
            data.X_train[:, 0] if method.input_mode == "univariate" else data.X_train
        )
        estimator.fit(fit_X, _fit_target(method_name, data.fit_targets))
        result["fit_seconds"] = round(time.perf_counter() - fit_started, 6)
        result["peak_rss_gib"] = _rss_gib()

        predict_started = time.perf_counter()
        predictions = _predict_vector(method_name, estimator, data.X_validation)
        result["predict_seconds"] = round(time.perf_counter() - predict_started, 6)
        if not np.all(np.isfinite(predictions)):
            raise ValueError("estimator returned non-finite predictions")

        score_started = time.perf_counter()
        result.update(_score_predictions(data, predictions, compute_fnc=compute_fnc))
        result["score_seconds"] = round(time.perf_counter() - score_started, 6)
        result["status"] = "completed"
    except Exception as exc:
        result.update(
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        result["total_seconds"] = round(time.perf_counter() - started, 6)
        result["peak_rss_gib"] = max(float(result["peak_rss_gib"] or 0.0), _rss_gib())
        atomic_write_text(
            result_path,
            json.dumps(result, sort_keys=True, indent=2, default=_json_default),
        )
    return 0 if result["status"] == "completed" else 1


def _terminate_process_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except psutil.Error:
        return
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        parent.kill()
    except psutil.Error:
        pass


def _timeout_result(
    method: EstimatorSpec,
    *,
    scope: str,
    timeout_seconds: int,
    elapsed_seconds: float,
    run_identity: str,
) -> dict[str, Any]:
    result = _worker_result_base(method, scope=scope)
    result.update(
        run_identity=run_identity,
        status="timed_out",
        timeout_seconds=timeout_seconds,
        total_seconds=round(elapsed_seconds, 6),
        error_type="TimeoutExpired",
        error_message=f"method exceeded {timeout_seconds}s process budget",
    )
    return result


def _write_frame(rows: Sequence[dict[str, Any]], output: Path) -> pl.DataFrame:
    if not rows:
        raise ValueError("cannot write an empty breadth result")
    frame = pl.DataFrame(list(rows))
    completed = frame.filter(pl.col("status") == "completed")
    if completed.height:
        best_corr = float(completed.get_column("corr_mean").max())
        frame = frame.with_columns(
            pl.when(pl.col("status") == "completed")
            .then(pl.col("corr_mean") == best_corr)
            .otherwise(False)
            .alias("best_corr")
        )
    else:
        frame = frame.with_columns(pl.lit(False).alias("best_corr"))
    frame = frame.with_columns(
        (pl.col("status") == "completed").alias("completed"),
        (pl.col("corr_mean") > 0.0).fill_null(False).alias("positive_corr"),
    )
    dummy = frame.filter(pl.col("method") == "DummyRegressor").get_column("corr_mean")
    dummy_value = float(dummy[0]) if len(dummy) and dummy[0] is not None else None
    if dummy_value is None:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Boolean).alias("beats_dummy"))
    else:
        frame = frame.with_columns(
            (pl.col("corr_mean") > dummy_value).fill_null(False).alias("beats_dummy")
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.sort("method").write_csv(output)
    return frame.sort("method")


def _load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or "method" not in payload
        or "status" not in payload
    ):
        raise ValueError(f"invalid breadth result checkpoint: {path}")
    return payload


def _run_identity(config: BreadthConfig, method_name: str) -> str:
    payload = {
        "data_dir": str(config.data_dir.resolve()),
        "method": method_name,
        "scope": config.scope,
        "max_train_rows": config.max_train_rows,
        "max_validation_rows": config.max_validation_rows,
        "timeout_seconds": config.timeout_seconds,
        "purge_eras": config.purge_eras,
        "seed": config.seed,
        "score_window": config.score_window,
        "compute_fnc": config.compute_fnc,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_sweep(config: BreadthConfig) -> pl.DataFrame:
    """Run or resume the catalog and write a durable CSV scorecard."""
    catalog = {spec.name: spec for spec in _catalog()}
    selected = tuple(config.methods) if config.methods else tuple(sorted(catalog))
    unknown = sorted(set(selected) - set(catalog))
    if unknown:
        raise ValueError(f"unknown sklearn methods: {unknown}")
    workspace = config.workspace or config.output.with_suffix(
        config.output.suffix + ".work"
    )
    results_dir = workspace / "results"
    logs_dir = workspace / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, method_name in enumerate(selected, start=1):
        result_path = results_dir / f"{method_name}.json"
        run_identity = _run_identity(config, method_name)
        if config.resume and result_path.is_file():
            try:
                row = _load_result(result_path)
                if row.get("run_identity") != run_identity:
                    raise ValueError(f"checkpoint identity mismatch: {result_path}")
                rows.append(row)
                logger.info(
                    "[breadth] %d/%d %s resumed status=%s fit=%ss total=%ss",
                    index,
                    len(selected),
                    method_name,
                    row.get("status"),
                    row.get("fit_seconds"),
                    row.get("total_seconds"),
                )
                _write_frame(rows, config.output)
                continue
            except (OSError, ValueError, json.JSONDecodeError):
                result_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "nmr.sklearn_breadth",
            "--worker",
            "--data-dir",
            str(config.data_dir),
            "--method",
            method_name,
            "--result",
            str(result_path),
            "--scope",
            config.scope,
            "--max-train-rows",
            str(config.max_train_rows or 0),
            "--max-validation-rows",
            str(config.max_validation_rows or 0),
            "--purge-eras",
            str(config.purge_eras),
            "--seed",
            str(config.seed),
            "--score-window",
            config.score_window,
            "--run-identity",
            run_identity,
        ]
        if config.compute_fnc:
            command.append("--compute-fnc")
        log_path = logs_dir / f"{method_name}.log"
        started = time.perf_counter()
        logger.info(
            "[breadth] %d/%d starting %s (timeout=%ss, scope=%s)",
            index,
            len(selected),
            method_name,
            config.timeout_seconds,
            config.scope,
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            try:
                return_code = process.wait(timeout=config.timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process.pid)
                elapsed = time.perf_counter() - started
                row = _timeout_result(
                    catalog[method_name],
                    scope=config.scope,
                    timeout_seconds=config.timeout_seconds,
                    elapsed_seconds=elapsed,
                    run_identity=run_identity,
                )
            else:
                if result_path.is_file():
                    row = _load_result(result_path)
                else:
                    row = _worker_result_base(catalog[method_name], scope=config.scope)
                    row.update(
                        status="failed",
                        total_seconds=round(time.perf_counter() - started, 6),
                        error_type="MissingWorkerResult",
                        error_message=f"worker exited {return_code} without a result",
                    )
        if row["status"] == "timed_out":
            atomic_write_text(
                result_path,
                json.dumps(row, sort_keys=True, indent=2, default=_json_default),
            )
        rows.append(row)
        _write_frame(rows, config.output)
        logger.info(
            "[breadth] %d/%d %s status=%s fit=%ss total=%ss",
            index,
            len(selected),
            method_name,
            row.get("status"),
            row.get("fit_seconds"),
            row.get("total_seconds"),
        )

    metadata = {
        "catalog_count": len(catalog),
        "selected_count": len(selected),
        "scope": config.scope,
        "data_dir": str(config.data_dir),
        "max_train_rows": config.max_train_rows,
        "max_validation_rows": config.max_validation_rows,
        "timeout_seconds": config.timeout_seconds,
        "purge_eras": config.purge_eras,
        "score_window": config.score_window,
        "compute_fnc": config.compute_fnc,
        "run_identities": {
            method_name: _run_identity(config, method_name) for method_name in selected
        },
        "gpu_available": bool(discover_hardware().gpus),
        "gpu_used": False,
        "gpu_reason": "scikit-learn regressors have no CUDA execution path",
    }
    atomic_write_text(
        config.output.with_suffix(config.output.suffix + ".json"),
        json.dumps(metadata, sort_keys=True, indent=2, default=_json_default),
    )
    return _write_frame(rows, config.output)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _optional_rows(value: str) -> int | None:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("row limit must be >= 0; zero means unlimited")
    return None if parsed == 0 else parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Breadth benchmark of all public scikit-learn regressors on NumerAI medium."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "v5.3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "reports" / "sklearn_breadth_screen.csv",
    )
    parser.add_argument("--scope", choices=("screen", "full"), default="screen")
    parser.add_argument(
        "--max-train-rows", type=_optional_rows, default=_DEFAULT_MAX_TRAIN_ROWS
    )
    parser.add_argument(
        "--max-validation-rows",
        type=_optional_rows,
        default=_DEFAULT_MAX_VALIDATION_ROWS,
    )
    parser.add_argument(
        "--timeout-seconds", type=_positive_int, default=_DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--purge-eras", type=int, default=_DEFAULT_PURGE_ERAS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-window", choices=("meta", "validation"), default="meta"
    )
    parser.add_argument("--compute-fnc", action="store_true")
    parser.add_argument(
        "--methods",
        default="",
        help="comma-separated estimator names; default is every public sklearn regressor",
    )
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-identity", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.worker:
        if args.method is None or args.result is None:
            raise ValueError("worker mode requires --method and --result")
        return run_worker(
            data_dir=args.data_dir,
            method_name=args.method,
            result_path=args.result,
            scope=args.scope,
            max_train_rows=args.max_train_rows,
            max_validation_rows=args.max_validation_rows,
            purge_eras=args.purge_eras,
            seed=args.seed,
            score_window=args.score_window,
            compute_fnc=args.compute_fnc,
            run_identity=args.run_identity or "",
        )
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    config = BreadthConfig(
        data_dir=args.data_dir,
        output=args.output,
        scope=args.scope,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        timeout_seconds=args.timeout_seconds,
        purge_eras=args.purge_eras,
        seed=args.seed,
        score_window=args.score_window,
        compute_fnc=args.compute_fnc,
        methods=methods,
        workspace=args.workspace,
        resume=not args.no_resume,
    )
    frame = run_sweep(config)
    completed = frame.filter(pl.col("status") == "completed")
    logger.info(
        "[breadth] finished selected=%d completed=%d failed_or_timed_out=%d output=%s",
        frame.height,
        completed.height,
        frame.height - completed.height,
        args.output,
    )
    return 0 if completed.height else 1


if __name__ == "__main__":
    raise SystemExit(main())
