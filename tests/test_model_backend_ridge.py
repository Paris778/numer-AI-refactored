from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import polars as pl
import pytest

from nmr import BackendRegistry
from nmr.config import DataConfig, ModelConfig, SplitConfig
from nmr.deployment import load_predict
from nmr.models import ModelOrchestrator
from nmr.runner import _build_deploy_pipeline, _serialize_predict_artifact
from nmr.splitter import PurgedEraSplitter


def _ridge_module():
    return importlib.import_module("nmr.model_backend_ridge")


def _ridge_adapter():
    return _ridge_module().RidgeAdapter()


def _training_frame(*, n_eras: int = 12, rows_per_era: int = 5) -> pl.DataFrame:
    rows: list[dict[str, float | str | None]] = []
    for era_num in range(1, n_eras + 1):
        for row_num in range(rows_per_era):
            f1 = float((era_num * 3 + row_num) % 11) / 10.0
            f2 = float((era_num * 5 - row_num * 2) % 13) / 10.0
            f3 = float((era_num + row_num * 7) % 17) / 10.0
            rows.append(
                {
                    "id": f"{era_num}_{row_num}",
                    "era": str(era_num),
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "target": 0.45 * f1 - 0.25 * f2 + 0.15 * f3 + (era_num / 100.0),
                    "target_alt": -0.20 * f1
                    + 0.35 * f2
                    - 0.10 * f3
                    + (era_num / 200.0),
                }
            )
    return pl.DataFrame(rows)


def _anchor_splitter() -> PurgedEraSplitter:
    return PurgedEraSplitter(SplitConfig(scheme="anchor", purge_eras=1))


def _walk_forward_splitter() -> PurgedEraSplitter:
    return PurgedEraSplitter(
        SplitConfig(scheme="walk_forward", n_folds=3, purge_eras=1)
    )


def _ridge_config(*, device: str = "cpu", **params: object) -> ModelConfig:
    return ModelConfig(
        backend="ridge",
        preset="fast",
        device=device,
        params=dict(params),
    )


def _live_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "era": ["21", "21", "21", "22", "22", "22"],
            "f1": [0.0, 0.2, 0.4, 0.3, 0.5, 0.7],
            "f2": [0.1, 0.0, 0.4, 0.2, 0.1, 0.3],
            "f3": [0.2, 0.6, 0.1, 0.8, 0.4, 0.0],
        },
        index=[f"live_{idx}" for idx in range(6)],
    )


def test_ridge_resolve_params_rejects_unknown_and_invalid_values() -> None:
    adapter = _ridge_adapter()

    assert adapter.resolve_params(preset="fast", params={}, n_features=3) == {
        "alpha": 1.0,
        "fit_intercept": True,
        "solver": "lsqr",
    }

    with pytest.raises(ValueError, match="unknown"):
        adapter.resolve_params(
            preset="fast",
            params={"unknown_param": 1},
            n_features=3,
        )

    with pytest.raises(ValueError, match="alpha"):
        adapter.resolve_params(
            preset="fast",
            params={"alpha": 0.0},
            n_features=3,
        )

    with pytest.raises(ValueError, match="alpha"):
        adapter.resolve_params(
            preset="fast",
            params={"alpha": float("inf")},
            n_features=3,
        )

    with pytest.raises(ValueError, match="fit_intercept"):
        adapter.resolve_params(
            preset="fast",
            params={"fit_intercept": "yes"},
            n_features=3,
        )

    with pytest.raises(ValueError, match="solver"):
        adapter.resolve_params(
            preset="fast",
            params={"solver": "not-a-solver"},
            n_features=3,
        )


def test_ridge_cross_validation_is_deterministic_across_processes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import hashlib
import json

import numpy as np
import polars as pl

from nmr.config import ModelConfig, SplitConfig
from nmr.models import ModelOrchestrator
from nmr.splitter import PurgedEraSplitter

rows = []
for era_num in range(1, 13):
    for row_num in range(5):
        f1 = float((era_num * 3 + row_num) % 11) / 10.0
        f2 = float((era_num * 5 - row_num * 2) % 13) / 10.0
        f3 = float((era_num + row_num * 7) % 17) / 10.0
        rows.append(
            {
                "id": f"{era_num}_{row_num}",
                "era": str(era_num),
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "target": 0.45 * f1 - 0.25 * f2 + 0.15 * f3 + (era_num / 100.0),
            }
        )
df = pl.DataFrame(rows)
orchestrator = ModelOrchestrator(
    ModelConfig(backend="ridge", preset="fast", device="cpu", params={"alpha": 0.75}),
    seed=23,
)
result = orchestrator.train_cross_validation(
    df,
    feature_cols=["f1", "f2", "f3"],
    target_col="target",
    splitter=PurgedEraSplitter(SplitConfig(scheme="walk_forward", n_folds=3, purge_eras=1)),
)
payload = {
    "prediction_hash": hashlib.sha256(
        result.oof.sort(["era", "id"]).get_column("prediction").to_numpy().astype(np.float64).tobytes()
    ).hexdigest(),
    "selected_backend_identity": orchestrator.planned_selected_backend_identity(
        n_features=3,
        device_role="cpu",
    ),
}
print(json.dumps(payload, sort_keys=True))
"""

    first = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    second = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert first.stdout.strip() == second.stdout.strip()


def test_ridge_zero_variance_scale_is_one_and_predictions_are_finite() -> None:
    adapter = _ridge_adapter()
    resolved = adapter.resolve_params(
        preset="fast", params={"alpha": 0.5}, n_features=3
    )
    model = adapter.build_model(resolved_params=resolved, seed=7, device="cpu")
    train = np.asarray(
        [
            [0.0, 5.0, 1.0],
            [1.0, 5.0, 0.5],
            [2.0, 5.0, -0.5],
            [3.0, 5.0, -1.0],
        ],
        dtype=float,
    )
    target = np.asarray([0.1, 0.4, 0.8, 1.2], dtype=float)

    fitted = adapter.fit(model, train, target)
    preds = adapter.predict(
        fitted,
        np.asarray(
            [
                [1.5, 5.0, 0.2],
                [2.5, 5.0, -0.2],
            ],
            dtype=float,
        ),
    )

    assert fitted.scaler.scale_[1] == 1.0
    assert np.isfinite(preds).all()


def test_ridge_orchestrator_filters_null_and_non_finite_targets_before_fit() -> None:
    ridge_module = _ridge_module()
    seen_targets: list[np.ndarray] = []

    class SpyRidgeAdapter(ridge_module.RidgeAdapter):
        def fit(self, model, features, target, *, progress=None):
            seen_targets.append(np.asarray(target, dtype=float).copy())
            return super().fit(model, features, target, progress=progress)

    df = _training_frame(n_eras=10, rows_per_era=4).with_columns(
        pl.when((pl.col("era") == "1") & (pl.col("id").str.ends_with("_0")))
        .then(None)
        .when((pl.col("era") == "2") & (pl.col("id").str.ends_with("_1")))
        .then(float("nan"))
        .otherwise(pl.col("target"))
        .alias("target")
    )
    spy = SpyRidgeAdapter()
    registry = BackendRegistry(adapters={"ridge": spy})
    orchestrator = ModelOrchestrator(
        _ridge_config(alpha=0.75),
        seed=17,
        backend_registry=registry,
    )
    train_eras = set(
        _anchor_splitter().split(df.get_column("era").to_list())[0].train_eras
    )
    original_train_rows = df.filter(pl.col("era").is_in(sorted(train_eras))).height

    model, prediction = orchestrator.train_anchor_fold(
        df,
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=_anchor_splitter(),
    )

    assert model is not None
    assert prediction.height > 0
    assert len(seen_targets) == 1
    assert len(seen_targets[0]) == original_train_rows - 2
    assert np.isfinite(seen_targets[0]).all()


def test_ridge_rejects_non_finite_feature_matrices_before_fit_and_predict() -> None:
    adapter = _ridge_adapter()
    resolved = adapter.resolve_params(preset="fast", params={}, n_features=2)

    with pytest.raises(ValueError, match="non-finite"):
        adapter.fit(
            adapter.build_model(resolved_params=resolved, seed=7, device="cpu"),
            np.asarray([[0.0, np.nan], [1.0, 2.0]], dtype=float),
            np.asarray([0.1, 0.2], dtype=float),
        )

    fitted = adapter.fit(
        adapter.build_model(resolved_params=resolved, seed=7, device="cpu"),
        np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float),
        np.asarray([0.1, 0.2, 0.3], dtype=float),
    )

    with pytest.raises(ValueError, match="non-finite"):
        adapter.predict(fitted, np.asarray([[np.inf, 0.0]], dtype=float))


def test_ridge_explicit_gpu_request_raises_before_fit() -> None:
    ridge_module = _ridge_module()

    class SpyRidgeAdapter(ridge_module.RidgeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.fit_calls = 0

        def fit(self, model, features, target, *, progress=None):
            self.fit_calls += 1
            return super().fit(model, features, target, progress=progress)

    spy = SpyRidgeAdapter()
    orchestrator = ModelOrchestrator(
        _ridge_config(device="gpu", alpha=0.75),
        seed=17,
        backend_registry=BackendRegistry(adapters={"ridge": spy}),
    )

    with pytest.raises(ValueError, match="unsupported"):
        orchestrator.train_cross_validation(
            _training_frame(),
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            splitter=_walk_forward_splitter(),
        )

    assert spy.fit_calls == 0


def test_ridge_full_history_model_is_cloudpickle_reloadable() -> None:
    df = _training_frame()
    orchestrator = ModelOrchestrator(_ridge_config(alpha=0.25), seed=9)
    model = orchestrator.train_full_history(
        df,
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        in_process=True,
    )
    features = orchestrator._feature_frame(df, feature_cols=["f1", "f2", "f3"])
    restored = cloudpickle.loads(cloudpickle.dumps(model))

    assert np.array_equal(
        orchestrator._predict_model(model, features=features),
        orchestrator._predict_model(restored, features=features),
    )


def test_ridge_shared_deploy_closure_roundtrips_via_load_predict(
    tmp_path: Path,
) -> None:
    train_df = _training_frame()
    orchestrator = ModelOrchestrator(_ridge_config(alpha=0.5), seed=13)
    predict_fn, model_meta = _build_deploy_pipeline(
        orchestrator=orchestrator,
        train_df=train_df,
        feature_cols=["f1", "f2", "f3"],
        target_cols=["target", "target_alt"],
        weights=[0.6, 0.4],
        proportion=0.25,
        data=DataConfig(version="vtest", data_dir=tmp_path / "data"),
        fit_device="cpu",
    )
    artifact = _serialize_predict_artifact(
        predict_fn=predict_fn,
        model_meta=model_meta,
        artifact_path=tmp_path / "predict.pkl",
    )
    live = _live_features()
    expected = predict_fn(live)
    actual = load_predict(artifact.path)(live)

    assert expected.index.tolist() == actual.index.tolist()
    assert np.allclose(
        expected["prediction"].to_numpy(),
        actual["prediction"].to_numpy(),
        atol=0.0,
        rtol=0.0,
    )


def test_backend_registry_with_builtins_exposes_ridge() -> None:
    registry = BackendRegistry.with_builtins()
    identity = registry.identity()

    assert tuple(sorted(identity["adapters_by_name"])) == (
        "catboost",
        "lightgbm",
        "ridge",
        "xgboost",
    )
    assert registry.resolve("ridge").name == "ridge"
    assert identity["adapters_by_name"]["ridge"]["capabilities"] == {
        "supports_gpu": False,
        "supports_full_history": True,
        "supports_deployment": True,
        "deployment_device": "cpu",
    }
