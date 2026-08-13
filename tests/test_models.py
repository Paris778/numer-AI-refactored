"""Tests for nmr.models.ModelOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import logging
import numpy as np
import polars as pl
import pytest
import xgboost as xgb

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
        assert seen_eras.isdisjoint(set(fold.val_eras))
        assert {int(era) for era in fold.train_eras}.isdisjoint(
            {int(era) for era in fold.val_eras}
        )

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

    def fake_fit_predict_fold(frame, *, fold, feature_cols, target_col, era_col, purge_eras):
        del purge_eras
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
    backend_error: type[Exception] = RuntimeError

    def fit(self, features, target, **kwargs):
        if self.params.get("device_type") == "gpu":
            raise self.backend_error("GPU unavailable")
        if self.params.get("tree_method") == "gpu_hist":
            raise self.backend_error("GPU unavailable")
        if self.params.get("device") == "cuda":
            raise self.backend_error("GPU unavailable")
        self._rows = len(features)
        return self

    def predict(self, features):
        return np.full(len(features), 0.25)

    def get_params(self, deep: bool = True):
        return dict(self.params)


@dataclass
class _FeatureNameModel:
    seen_fit_shape: tuple[int, int] | None = None
    seen_predict_shape: tuple[int, int] | None = None

    def fit(self, features, target, **kwargs):
        self.seen_fit_shape = features.shape
        assert isinstance(features, np.ndarray), "feature frames must be numpy"
        return self

    def predict(self, features):
        self.seen_predict_shape = features.shape
        assert isinstance(features, np.ndarray), "feature frames must be numpy"
        return np.zeros(len(features), dtype=float)


@pytest.mark.parametrize(
    ("backend", "attribute", "key", "gpu_value", "cpu_value", "backend_error"),
    [
        ("lightgbm", "LGBMRegressor", "device_type", "gpu", "cpu", lgb.basic.LightGBMError),
        ("xgboost", "XGBRegressor", "device", "cuda", "cpu", xgb.core.XGBoostError),
    ],
)
def test_gpu_absent_falls_back_to_cpu_without_raising(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    attribute: str,
    key: str,
    gpu_value: str,
    cpu_value: str,
    backend_error: type[Exception],
) -> None:
    import nmr.models as models_module

    df = _model_frame()

    def factory(**params):
        return _FakeModel(params=params, backend_error=backend_error)

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
    assert params[key] != gpu_value
    assert params[key] == cpu_value
    assert orchestrator.resolved_device == "cpu"


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
    # feature frames are Float32 numpy (zero-copy memory discipline); column
    # ORDER follows the requested feature_cols; fit/predict split heights
    # differ (anchor split purges eras between them)
    assert model.seen_fit_shape[1] == 3
    assert model.seen_predict_shape[1] == 3
    assert model.seen_fit_shape[0] > 0 and model.seen_predict_shape[0] > 0


def test_train_full_history_covers_all_eras_and_is_cpu_only() -> None:
    df = _model_frame()
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=5,
    )
    model = orchestrator.train_full_history(
        df, feature_cols=["f1", "f2", "f3"], target_col="target"
    )
    assert model is not None
    assert model.get_params()["device_type"] == "cpu"


def test_train_full_history_drops_null_targets() -> None:
    df = _model_frame(n_eras=4)
    df = df.with_columns(
        pl.when(pl.col("id") == "1_0").then(None).otherwise(pl.col("target")).alias("target")
    )
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=5,
    )
    model = orchestrator.train_full_history(
        df, feature_cols=["f1", "f2", "f3"], target_col="target"
    )
    assert model is not None


def test_fit_predict_fold_rejects_zero_purge_gap() -> None:
    from nmr.splitter import Fold

    df = _model_frame(n_eras=8)
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=3,
    )
    violating = Fold(
        index=0,
        train_eras=tuple(str(e) for e in range(1, 5)),
        val_eras=tuple(str(e) for e in range(5, 7)),  # gap = 1 <= purge_eras=1
    )
    with pytest.raises(ValueError, match="purge"):
        orchestrator._fit_predict_fold(
            df, fold=violating, feature_cols=["f1", "f2", "f3"],
            target_col="target", era_col="era", purge_eras=1,
        )


def test_fit_predict_fold_drops_null_target_rows(caplog: pytest.LogCaptureFixture) -> None:
    df = _model_frame(n_eras=6).with_columns(
        pl.when(pl.col("id") == "1_0")
        .then(None)
        .otherwise(pl.col("target"))
        .alias("target")
    )
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=3,
    )
    with caplog.at_level(logging.WARNING, logger="nmr.models"):
        model, prediction = orchestrator.train_anchor_fold(
            df,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            splitter=_anchor_splitter(),
        )
    assert model is not None
    assert prediction.height > 0
    assert any(
        "dropped 1 rows with null/non-finite targets" in record.message
        for record in caplog.records
    )


def test_fit_model_records_resolved_device() -> None:
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=3,
    )
    df = _model_frame(n_eras=4)
    orchestrator.train_full_history(df, feature_cols=["f1", "f2", "f3"], target_col="target")
    assert orchestrator.resolved_device == "cpu"


from nmr.models import ModelOrchestrator, resolve_model_params


def test_resolve_model_params_merges_preset_and_overrides():
    resolved = resolve_model_params("fast", {"n_estimators": 2500})
    assert resolved["n_estimators"] == 2500          # override wins
    assert resolved["learning_rate"] == 0.01         # preset default present
    assert resolved["num_leaves"] == (2**5) - 1      # preset default present


def test_resolve_model_params_matches_orchestrator_resolution():
    cfg = ModelConfig(backend="lightgbm", preset="fast",
                      params={"n_estimators": 2500, "colsample_bytree": 0.2})
    orch = ModelOrchestrator(cfg, seed=42)
    # _resolved_params(use_gpu=False, n_features=1000) adds backend boilerplate;
    # the preset+params core must equal resolve_model_params for the same inputs.
    # n_features=1000 keeps the colsample floor at its 0.1 no-op, so the
    # resolved core stays identical to the raw resolution.
    resolved = orch._resolved_params(use_gpu=False, n_features=1000)
    for key, value in resolve_model_params("fast", cfg.params).items():
        assert resolved[key] == value


def test_resolve_model_params_unknown_preset_raises():
    import pytest as _pytest

    with _pytest.raises(KeyError):
        resolve_model_params("bogus", {})


def test_xgboost_gpu_params_use_cuda_device() -> None:
    """xgboost >= 3.0 removed tree_method='gpu_hist' (raises Invalid Input);
    GPU acceleration is device='cuda' with tree_method='hist'."""
    cfg = ModelConfig(backend="xgboost", preset="fast")
    orchestrator = ModelOrchestrator(cfg, seed=3)
    gpu = orchestrator._resolved_params(use_gpu=True, n_features=1000)
    cpu = orchestrator._resolved_params(use_gpu=False, n_features=1000)
    assert gpu["tree_method"] == "hist"
    assert gpu["device"] == "cuda"
    assert cpu["tree_method"] == "hist"
    assert cpu["device"] == "cpu"
    assert (
        orchestrator._device_candidate_params(use_gpu=True, n_features=1000)[0][
            "device"
        ]
        == "cuda"
    )


# --- catboost backend (Task 3) -------------------------------------------------


def test_translate_catboost_maps_preset_knobs() -> None:
    from nmr.models import _translate_catboost

    resolved = {
        "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 5,
        "num_leaves": 31, "colsample_bytree": 0.1, "min_data_in_leaf": 100,
    }
    params = _translate_catboost(resolved, seed=42, use_gpu=False)
    assert params["iterations"] == 2000
    assert params["learning_rate"] == 0.01
    assert params["depth"] == 5
    assert params["rsm"] == 0.1
    assert params["min_data_in_leaf"] == 100
    assert "num_leaves" not in params          # dropped: symmetric depth-limited trees


def test_translate_catboost_contract_params_are_fixed_and_win() -> None:
    from nmr.models import _translate_catboost

    resolved = {"random_seed": 1, "thread_count": 8, "n_estimators": 100}
    params = _translate_catboost(resolved, seed=42, use_gpu=False)
    assert params["loss_function"] == "RMSE"
    assert params["random_seed"] == 42         # contract wins over user params
    assert params["thread_count"] == 1         # contract wins over user params
    assert params["verbose"] is False
    assert params["allow_writing_files"] is False
    assert params["task_type"] == "CPU"
    assert params["iterations"] == 100         # non-contract keys still map


def test_translate_catboost_gpu_sets_task_type_and_devices() -> None:
    from nmr.models import _translate_catboost

    params = _translate_catboost({"n_estimators": 100}, seed=42, use_gpu=True)
    assert params["task_type"] == "GPU"
    assert params["devices"] == "0"


def test_catboost_cv_oof_is_deterministic_under_seed(tmp_path) -> None:
    df = _model_frame()
    splitter = _walk_forward_splitter()
    cfg = ModelConfig(backend="catboost", preset="fast", params={"n_estimators": 10})
    orch = ModelOrchestrator(cfg, seed=17)
    first = orch.train_cross_validation(
        df, feature_cols=["f1", "f2", "f3"], target_col="target",
        splitter=splitter, era_col="era",
    )
    second = ModelOrchestrator(cfg, seed=17).train_cross_validation(
        df, feature_cols=["f1", "f2", "f3"], target_col="target",
        splitter=splitter, era_col="era",
    )
    assert first.oof.equals(second.oof)
    assert orch.resolved_device == "cpu"


def test_catboost_is_cpu_only_by_construction() -> None:
    # CatBoost rejects `rsm` on GPU (non-pairwise modes) and every canonical
    # preset ships colsample_bytree -> rsm, so the orchestrator never attempts
    # a GPU candidate for the catboost backend: CPU-only by construction.
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="catboost", preset="fast", params={"n_estimators": 10}),
        seed=29,
    )
    candidates = orchestrator._device_candidate_params(use_gpu=True, n_features=1000)
    assert len(candidates) == 1
    assert candidates[0]["task_type"] == "CPU"


# --- model.device knob (auto|gpu|cpu) -----------------------------------------

def test_orchestrator_device_cpu_never_attempts_gpu(monkeypatch) -> None:
    import nmr.models as models_module

    seen: list[str] = []

    def factory(**params):
        seen.append(params.get("device_type", "?"))
        return _FakeModel(params=params, backend_error=RuntimeError)

    monkeypatch.setattr(models_module.lgb, "LGBMRegressor", factory)
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", device="cpu",
                    params=_tiny_model_params()),
        seed=29,
    )
    model, prediction = orchestrator.train_anchor_fold(
        _model_frame(),
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=_anchor_splitter(),
    )
    assert isinstance(prediction, pl.DataFrame)
    assert seen == ["cpu"]  # a single CPU candidate, no GPU attempt
    assert orchestrator.resolved_device == "cpu"


def test_orchestrator_device_gpu_forced_raises_without_cpu_fallback(monkeypatch) -> None:
    import nmr.models as models_module

    def factory(**params):
        return _FakeModel(params=params, backend_error=lgb.basic.LightGBMError)

    monkeypatch.setattr(models_module.lgb, "LGBMRegressor", factory)
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", device="gpu",
                    params=_tiny_model_params()),
        seed=29,
    )
    with pytest.raises(lgb.basic.LightGBMError):
        orchestrator.train_anchor_fold(
            _model_frame(),
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            splitter=_anchor_splitter(),
        )


def test_orchestrator_device_auto_tries_gpu_then_falls_back(monkeypatch) -> None:
    import nmr.models as models_module

    seen: list[str] = []

    def factory(**params):
        seen.append(params.get("device_type"))
        return _FakeModel(params=params, backend_error=lgb.basic.LightGBMError)

    monkeypatch.setattr(models_module.lgb, "LGBMRegressor", factory)
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", device="auto",
                    params=_tiny_model_params()),
        seed=29,
    )
    model, _ = orchestrator.train_anchor_fold(
        _model_frame(),
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=_anchor_splitter(),
    )
    assert seen == ["gpu", "cpu"]  # GPU attempted, then CPU fallback
    assert orchestrator.resolved_device == "cpu"


def test_train_full_history_stays_cpu_even_with_device_gpu() -> None:
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", device="gpu",
                    params=_tiny_model_params()),
        seed=5,
    )
    model = orchestrator.train_full_history(
        _model_frame(), feature_cols=["f1", "f2", "f3"], target_col="target"
    )
    assert model.get_params()["device_type"] == "cpu"  # deploy artifact invariant


def test_coerce_float32_features_exact_and_all_or_nothing() -> None:
    from nmr.models import coerce_float32_features

    int_frame = pl.DataFrame(
        {
            "f1": pl.Series([0, 1, 2, 3, 4], dtype=pl.Int8),
            "f2": pl.Series([4, 3, 2, 1, 0], dtype=pl.Int8),
        }
    )
    out = coerce_float32_features(int_frame, ["f1", "f2"])
    assert out.schema == {"f1": pl.Float32, "f2": pl.Float32}
    assert out.to_numpy().tolist() == int_frame.to_numpy().tolist()  # exact

    # Float64 stays untouched (precision), all-or-nothing per frame
    float_frame = pl.DataFrame(
        {"f1": pl.Series([0.1, 0.2], dtype=pl.Float64), "f2": pl.Series([1, 2], dtype=pl.Int8)}
    )
    out_mixed = coerce_float32_features(float_frame, ["f1", "f2"])
    assert out_mixed.schema == {"f1": pl.Float64, "f2": pl.Int8}
    out_float = coerce_float32_features(float_frame.select("f1"), ["f1"])
    assert out_float.schema == {"f1": pl.Float64}


def test_feature_frame_is_float32_numpy_for_int_features() -> None:
    from nmr.models import ModelOrchestrator, coerce_float32_features

    df = _model_frame().with_columns(
        pl.col("f1").cast(pl.Int8), pl.col("f2").cast(pl.Int8), pl.col("f3").cast(pl.Int8)
    )
    orchestrator = ModelOrchestrator(ModelConfig(backend="lightgbm"), seed=7)
    frame = orchestrator._feature_frame(df, feature_cols=["f1", "f2", "f3"])
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.float32
    assert frame.shape == (df.height, 3)
    assert np.array_equal(frame, df.select(["f1", "f2", "f3"]).to_numpy())
    assert coerce_float32_features(df, ["f1", "f2", "f3"]).schema == {
        "f1": pl.Float32, "f2": pl.Float32, "f3": pl.Float32
    }


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
def test_fold_predict_chunked_matches_full_predict(backend: str) -> None:
    """Era-batched predict (GPU-VRAM bound) is bit-identical to one call."""
    df = _model_frame(n_eras=12, rows_per_era=6)
    orchestrator = ModelOrchestrator(
        ModelConfig(backend=backend, preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    model = orchestrator.train_full_history(
        df, feature_cols=["f1", "f2", "f3"], target_col="target"
    )
    full = orchestrator._predict_model(
        model, features=orchestrator._feature_frame(df, feature_cols=["f1", "f2", "f3"])
    )
    chunked = orchestrator._predict_model_chunked(model, df, ["f1", "f2", "f3"])
    assert np.array_equal(full, chunked)


def test_predict_model_chunked_empty_frame() -> None:
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    empty = _model_frame().clear()
    out = orchestrator._predict_model_chunked(None, empty, ["f1"])  # type: ignore[arg-type]
    assert out.size == 0


def test_full_history_subprocess_fit_matches_in_process() -> None:
    """The spawned-process fit (bounded commit for full-universe runs) must be
    bit-identical to the in-process fit — same code path, same seed."""
    from nmr.models import ModelOrchestrator, _full_history_fit_worker

    df = _model_frame(n_eras=10, rows_per_era=6)
    cfg = ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params())
    orchestrator = ModelOrchestrator(cfg, seed=7)
    in_process = orchestrator.train_full_history(
        df, feature_cols=["f1", "f2", "f3"], target_col="target", in_process=True
    )

    import json
    import multiprocessing as mp
    from pathlib import Path

    # the worker re-reads the train split from its own data dir
    import tempfile

    data_root = Path(tempfile.mkdtemp()) / "data" / "vtest"
    data_root.mkdir(parents=True)
    (data_root / "features.json").write_text(
        json.dumps({
            "feature_sets": {"small": ["f1", "f2", "f3"],
                             "medium": ["f1", "f2", "f3"],
                             "all": ["f1", "f2", "f3"]},
            "targets": ["target"],
        }),
        encoding="utf-8",
    )
    df.write_parquet(data_root / "train.parquet")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    spec = {
        "data": {
            "version": "vtest", "feature_set": "small", "feature_subset": None,
            "targets": ("target",), "data_dir": str(data_root.parent),
            "supplemental_feature_sets": None,
        },
        "feature_cols": ["f1", "f2", "f3"],
        "target_col": "target",
        "era_col": "era",
        "backend": "lightgbm",
        "preset": "fast",
        "params": _tiny_model_params(),
        "seed": 7,
    }
    proc = ctx.Process(target=_full_history_fit_worker, args=(spec, q))
    proc.start()
    status, payload = q.get()
    proc.join()
    assert status == "ok", payload

    import cloudpickle

    subprocess_model = cloudpickle.loads(payload)
    feats = orchestrator._feature_frame(df, feature_cols=["f1", "f2", "f3"])
    a = in_process.predict(feats)
    b = subprocess_model.predict(feats)
    assert np.array_equal(a, b)


def test_full_history_spawn_path_with_data_config(tmp_path) -> None:
    """The parent-side spawner must accept the DataConfig from the caller
    (regression: it read self._config.data — the orchestrator holds a
    ModelConfig, which has no .data, and lgbm_v1 died with AttributeError
    after 4h of CV)."""
    import json
    from pathlib import Path

    from nmr.config import DataConfig

    data_root = tmp_path / "data" / "vtest"
    data_root.mkdir(parents=True)
    (data_root / "features.json").write_text(
        json.dumps({
            "feature_sets": {"small": ["f1", "f2", "f3"],
                             "medium": ["f1", "f2", "f3"],
                             "all": ["f1", "f2", "f3"]},
            "targets": ["target"],
        }),
        encoding="utf-8",
    )
    df = _model_frame(n_eras=10, rows_per_era=6)
    df.write_parquet(data_root / "train.parquet")

    orch = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    data_cfg = DataConfig(version="vtest", data_dir=tmp_path / "data")
    model = orch._fit_full_history_subprocess(
        df, feature_cols=["f1", "f2", "f3"], target_col="target",
        era_col="era", data=data_cfg,
    )
    feats = orch._feature_frame(df, feature_cols=["f1", "f2", "f3"])
    preds = np.asarray(model.predict(feats), dtype=float)
    assert preds.size == df.height
    assert np.isfinite(preds).all()
    assert orch.resolved_device == "cpu"


def test_full_history_spawn_requires_data_config() -> None:
    orch = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    with pytest.raises(ValueError, match="DataConfig"):
        orch._fit_full_history_subprocess(
            _model_frame(), feature_cols=["f1", "f2", "f3"],
            target_col="target", era_col="era", data=None,  # type: ignore[arg-type]
        )


# --- dynamic colsample_bytree floor (Phase 1, item 3) -------------------------


def test_colsample_floor_values_and_guard() -> None:
    from nmr.models import _colsample_floor

    assert _colsample_floor(1) == pytest.approx(1.0)
    assert _colsample_floor(3) == pytest.approx(1.0)
    assert _colsample_floor(10) == pytest.approx(1.0)
    assert _colsample_floor(42) == pytest.approx(10.0 / 42.0)
    assert _colsample_floor(780) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="n_features"):
        _colsample_floor(0)


def test_resolved_params_raises_colsample_on_small_sets() -> None:
    orch = ModelOrchestrator(ModelConfig(backend="lightgbm", preset="fast"), seed=1)
    cpu = orch._resolved_params(use_gpu=False, n_features=3)
    gpu = orch._resolved_params(use_gpu=True, n_features=3)
    assert cpu["colsample_bytree"] == pytest.approx(1.0)
    assert gpu["colsample_bytree"] == pytest.approx(1.0)  # cross-device parity


def test_resolved_params_preserves_explicit_override_above_floor() -> None:
    cfg = ModelConfig(
        backend="lightgbm", preset="fast", params={"colsample_bytree": 0.5}
    )
    orch = ModelOrchestrator(cfg, seed=1)
    resolved = orch._resolved_params(use_gpu=False, n_features=42)
    assert resolved["colsample_bytree"] == pytest.approx(0.5)


def test_resolved_params_floor_noop_for_large_sets() -> None:
    orch = ModelOrchestrator(ModelConfig(backend="lightgbm", preset="fast"), seed=1)
    resolved = orch._resolved_params(use_gpu=False, n_features=780)
    assert resolved["colsample_bytree"] == pytest.approx(0.1)


def test_resolved_params_floors_xgboost_and_catboost_rsm() -> None:
    xgb_orch = ModelOrchestrator(
        ModelConfig(backend="xgboost", preset="fast"), seed=1
    )
    assert xgb_orch._resolved_params(use_gpu=True, n_features=3)[
        "colsample_bytree"
    ] == pytest.approx(1.0)
    cb_orch = ModelOrchestrator(
        ModelConfig(backend="catboost", preset="fast"), seed=1
    )
    assert cb_orch._resolved_params(use_gpu=False, n_features=3)["rsm"] == pytest.approx(
        1.0
    )


def test_resolved_params_floors_user_native_rsm() -> None:
    # A CatBoost-native `rsm` supplied directly in params must be bounded
    # identically (floor applied post-translation on the final key).
    cfg = ModelConfig(backend="catboost", preset="fast", params={"rsm": 0.05})
    orch = ModelOrchestrator(cfg, seed=1)
    resolved = orch._resolved_params(use_gpu=False, n_features=42)
    assert resolved["rsm"] == pytest.approx(10.0 / 42.0)


def test_colsample_floor_survives_float32_truncation() -> None:
    from nmr.models import _colsample_floor

    # The C++ backends compute static_cast<int>(n_features * fraction) in
    # single precision; the 1e-7 expansion must survive the float32
    # round-trip so the sampled count never truncates below 10.
    assert int(np.float32(_colsample_floor(42)) * 42) == 10
    assert int(np.float32(_colsample_floor(11)) * 11) >= 10
    assert int(np.float32(_colsample_floor(99)) * 99) >= 10
    assert _colsample_floor(780) == pytest.approx(0.1)  # ε stays inside max(0.1, ...)


def test_resolved_params_floors_lightgbm_feature_fraction_alias() -> None:
    cfg = ModelConfig(
        backend="lightgbm", preset="fast", params={"feature_fraction": 0.05}
    )
    orch = ModelOrchestrator(cfg, seed=1)
    resolved = orch._resolved_params(use_gpu=False, n_features=42)
    assert resolved["feature_fraction"] >= 10.0 / 42.0


def test_resolved_params_floors_lightgbm_sub_feature_alias() -> None:
    cfg = ModelConfig(backend="lightgbm", preset="fast", params={"sub_feature": 0.05})
    orch = ModelOrchestrator(cfg, seed=1)
    resolved = orch._resolved_params(use_gpu=False, n_features=3)
    assert resolved["sub_feature"] == pytest.approx(1.0)


def test_resolved_params_floors_all_present_alias_members() -> None:
    # Precedence-proof: with the preset's colsample_bytree AND user-supplied
    # aliases all present, every member of the alias group is floored, so no
    # alias resolution rule can bypass the lower bound.
    cfg = ModelConfig(
        backend="lightgbm",
        preset="fast",
        params={"feature_fraction": 0.05, "sub_feature": 0.05},
    )
    orch = ModelOrchestrator(cfg, seed=1)
    resolved = orch._resolved_params(use_gpu=False, n_features=42)
    assert resolved["colsample_bytree"] >= 10.0 / 42.0
    assert resolved["feature_fraction"] >= 10.0 / 42.0
    assert resolved["sub_feature"] >= 10.0 / 42.0
