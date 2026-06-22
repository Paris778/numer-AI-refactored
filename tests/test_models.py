"""Tests for nmr.models.ModelOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import pytest

from nmr.config import ModelConfig, SplitConfig
from nmr.models import CVResult, ModelOrchestrator
from nmr.splitter import PurgedEraSplitter


def _model_frame(*, n_eras: int = 16, rows_per_era: int = 6) -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for era_num in range(1, n_eras + 1):
        for row_num in range(rows_per_era):
            f1 = float((era_num * 3 + row_num) % 11) / 10.0
            f2 = float((era_num * 5 - row_num * 2) % 13) / 10.0
            f3 = float((era_num + row_num * 7) % 17) / 10.0
            target = 0.45 * f1 - 0.25 * f2 + 0.15 * f3 + (era_num / 100.0)
            rows.append(
                {
                    "id": f"{era_num}_{row_num}",
                    "era": str(era_num),
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "target": target,
                }
            )
    return pl.DataFrame(rows)


def _walk_forward_splitter() -> PurgedEraSplitter:
    return PurgedEraSplitter(
        SplitConfig(scheme="walk_forward", n_folds=3, purge_eras=1)
    )


def _anchor_splitter() -> PurgedEraSplitter:
    return PurgedEraSplitter(SplitConfig(scheme="anchor", purge_eras=1))


def _tiny_model_params(**extra: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 1,
        "max_depth": 1,
        "min_child_weight": 1,
    }
    params.update(extra)
    return params


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
def test_both_backends_train_anchor_and_emit_polars_predictions(backend: str) -> None:
    df = _model_frame()
    orchestrator = ModelOrchestrator(
        ModelConfig(backend=backend, preset="fast", params=_tiny_model_params()),
        seed=7,
    )

    model, prediction = orchestrator.train_anchor_fold(
        df,
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=_anchor_splitter(),
    )

    assert model is not None
    assert isinstance(prediction, pl.DataFrame)
    assert prediction.columns == ["id", "era", "prediction"]
    assert prediction.height > 0


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
def test_cross_validation_is_deterministic_on_cpu(backend: str) -> None:
    df = _model_frame()
    splitter = _walk_forward_splitter()
    config = ModelConfig(
        backend=backend,
        preset="fast",
        params=_tiny_model_params(),
    )

    first = ModelOrchestrator(config, seed=123).train_cross_validation(
        df,
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=splitter,
    )
    second = ModelOrchestrator(config, seed=123).train_cross_validation(
        df,
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=splitter,
    )

    assert first.oof.equals(second.oof)


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
def test_preset_params_applied_and_overrides_honored(backend: str) -> None:
    df = _model_frame()
    orchestrator = ModelOrchestrator(
        ModelConfig(
            backend=backend,
            preset="fast",
            params=_tiny_model_params(learning_rate=0.05),
        ),
        seed=11,
    )
    model, _ = orchestrator.train_anchor_fold(
        df,
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=_anchor_splitter(),
    )

    params = model.get_params()
    assert params["n_estimators"] == 1
    assert params["learning_rate"] == 0.05
    assert params["max_depth"] == 1
    assert params["min_child_weight"] == 1


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
def test_walk_forward_oof_covers_only_validation_eras_without_overlap(
    backend: str,
) -> None:
    df = _model_frame()
    splitter = _walk_forward_splitter()
    folds = splitter.split(df.get_column("era").to_list())
    orchestrator = ModelOrchestrator(
        ModelConfig(backend=backend, preset="fast", params=_tiny_model_params()),
        seed=19,
    )

    result = orchestrator.train_cross_validation(
        df,
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=splitter,
    )

    expected_val_eras = {era for fold in folds for era in fold.val_eras}
    oof_eras = set(result.oof.get_column("era").to_list())
    expected_rows = df.filter(pl.col("era").is_in(sorted(expected_val_eras))).height

    assert isinstance(result, CVResult)
    assert result.oof.columns == ["id", "era", "prediction"]
    assert oof_eras == expected_val_eras
    assert result.oof.height == expected_rows
    assert len(result.models) == len(folds)

    seen_eras: set[str] = set()
    for fold in folds:
        train_nums = {int(era) for era in fold.train_eras}
        val_nums = {int(era) for era in fold.val_eras}
        purge_buffer = set(range(max(train_nums) + 1, min(val_nums)))

        assert seen_eras.isdisjoint(set(fold.val_eras))
        assert train_nums.isdisjoint(val_nums)
        assert train_nums.isdisjoint(purge_buffer)
        assert val_nums.isdisjoint(purge_buffer)

        fold_predictions = result.oof.filter(pl.col("era").is_in(fold.val_eras))
        assert set(fold_predictions.get_column("era").to_list()) == set(fold.val_eras)
        seen_eras.update(fold.val_eras)


def test_cross_validation_routes_fold_local_train_and_validation_eras(
    monkeypatch,
) -> None:
    df = _model_frame()
    splitter = _walk_forward_splitter()
    orchestrator = ModelOrchestrator(
        ModelConfig(
            backend="lightgbm",
            preset="fast",
            params=_tiny_model_params(),
        ),
        seed=3,
    )
    recorded_pairs: list[tuple[set[str], set[str]]] = []

    def fake_fit_predict_fold(frame, *, fold, feature_cols, target_col, era_col):
        train_eras = set(
            frame.filter(pl.col(era_col).is_in(fold.train_eras))
            .get_column(era_col)
            .to_list()
        )
        val_eras = set(
            frame.filter(pl.col(era_col).is_in(fold.val_eras))
            .get_column(era_col)
            .to_list()
        )
        recorded_pairs.append((train_eras, val_eras))
        prediction = frame.filter(pl.col(era_col).is_in(fold.val_eras)).select(
            ["id", era_col]
        )
        prediction = prediction.rename({era_col: "era"}).with_columns(
            pl.lit(0.0).alias("prediction")
        )
        return object(), prediction

    monkeypatch.setattr(orchestrator, "_fit_predict_fold", fake_fit_predict_fold)

    result = orchestrator.train_cross_validation(
        df,
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=splitter,
    )

    assert result.oof.height > 0
    assert len(recorded_pairs) == len(splitter.split(df.get_column("era").to_list()))
    for train_eras, val_eras in recorded_pairs:
        assert train_eras.isdisjoint(val_eras)


@dataclass
class _FakeModel:
    params: dict[str, Any]

    def fit(self, features, target):
        if self.params.get("device_type") == "gpu":
            raise RuntimeError("GPU unavailable")
        if self.params.get("tree_method") == "gpu_hist":
            raise RuntimeError("GPU unavailable")
        self._rows = len(features)
        return self

    def predict(self, features):
        return np.full(len(features), 0.25)

    def get_params(self, deep: bool = True):
        return dict(self.params)


@dataclass
class _FeatureNameModel:
    seen_fit_columns: list[str] | None = None
    seen_predict_columns: list[str] | None = None

    def fit(self, features, target):
        self.seen_fit_columns = list(features.columns)
        return self

    def predict(self, features):
        self.seen_predict_columns = list(features.columns)
        return np.zeros(len(features), dtype=float)


@pytest.mark.parametrize(
    ("backend", "attribute", "gpu_value", "cpu_value"),
    [
        ("lightgbm", "LGBMRegressor", "gpu", "cpu"),
        ("xgboost", "XGBRegressor", "gpu_hist", "hist"),
    ],
)
def test_gpu_absent_falls_back_to_cpu_without_raising(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    attribute: str,
    gpu_value: str,
    cpu_value: str,
) -> None:
    import nmr.models as models_module

    df = _model_frame()

    def factory(**params):
        return _FakeModel(params=params)

    module = models_module.lgb if backend == "lightgbm" else models_module.xgb
    monkeypatch.setattr(module, attribute, factory)

    orchestrator = ModelOrchestrator(
        ModelConfig(backend=backend, preset="fast", params=_tiny_model_params()),
        seed=29,
    )
    model, prediction = orchestrator.train_anchor_fold(
        df,
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=_anchor_splitter(),
    )

    assert isinstance(prediction, pl.DataFrame)
    params = model.get_params()
    key = "device_type" if backend == "lightgbm" else "tree_method"
    assert params[key] != gpu_value
    assert params[key] == cpu_value


def test_backend_boundary_uses_named_feature_frames_consistently() -> None:
    df = _model_frame()
    orchestrator = ModelOrchestrator(
        ModelConfig(
            backend="lightgbm",
            preset="fast",
            params=_tiny_model_params(),
        ),
        seed=31,
    )
    model = _FeatureNameModel()

    def build_model(_params):
        return model

    orchestrator._build_model = build_model  # type: ignore[method-assign]

    _, prediction = orchestrator.train_anchor_fold(
        df,
        feature_cols=["f3", "f1", "f2"],
        target_col="target",
        splitter=_anchor_splitter(),
    )

    assert prediction.height > 0
    assert model.seen_fit_columns == ["f3", "f1", "f2"]
    assert model.seen_predict_columns == ["f3", "f1", "f2"]
