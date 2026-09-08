from __future__ import annotations

import importlib
import json
from typing import Any

import numpy as np
import polars as pl
import pytest


def _registry_module():
    return importlib.import_module("nmr.model_backend_registry")


def _protocol_module():
    return importlib.import_module("nmr.model_backend_protocol")


def _valid_capabilities(proto: Any):
    return proto.BackendCapabilities(
        supports_gpu=True,
        supports_full_history=True,
        supports_deployment=True,
        deployment_device="cpu",
    )


def _valid_identity(proto: Any, **overrides: object):
    values = {
        "schema_version": 1,
        "name": "custom_model",
        "adapter_version": "1",
        "implementation_fingerprint": "a" * 64,
        "resolved_params": {},
        "device": "cpu",
        "capabilities": _valid_capabilities(proto),
        "dependency_identity": {"stdlib": "python-3.11"},
    }
    values.update(overrides)
    return proto.BackendIdentity(**values)


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


def _walk_forward_splitter():
    from nmr.config import SplitConfig
    from nmr.splitter import PurgedEraSplitter

    return PurgedEraSplitter(
        SplitConfig(scheme="walk_forward", n_folds=3, purge_eras=1)
    )


def _write_runner_data(root) -> None:
    from pathlib import Path

    version_dir = Path(root) / "vtest"
    version_dir.mkdir(parents=True, exist_ok=True)
    features = {
        "feature_sets": {
            "small": ["f1", "f2", "f3"],
            "medium": ["f1", "f2", "f3"],
            "all": ["f1", "f2", "f3"],
        },
        "targets": ["target"],
    }
    (version_dir / "features.json").write_text(json.dumps(features), encoding="utf-8")
    train = _model_frame(n_eras=12, rows_per_era=6)
    train.write_parquet(version_dir / "train.parquet")
    validation = _model_frame(n_eras=6, rows_per_era=6).with_columns(
        (pl.col("era").cast(pl.Int32) + 12).cast(pl.Utf8).alias("era")
    )
    validation.write_parquet(version_dir / "validation.parquet")


def _runner_config(tmp_path, *, backend: str):
    from nmr.config import (
        DataConfig,
        EvalConfig,
        ExperimentConfig,
        ModelConfig,
        RunConfig,
        SplitConfig,
    )

    data_root = tmp_path / "data"
    _write_runner_data(data_root)
    return ExperimentConfig(
        data=DataConfig(
            version="vtest",
            feature_set="small",
            targets=("target",),
            data_dir=data_root,
        ),
        split=SplitConfig(
            scheme="walk_forward", purge_eras=1, embargo_eras=0, n_folds=2
        ),
        model=ModelConfig(
            backend=backend,
            preset="fast",
            device="cpu",
            params={},
        ),
        evaluation=EvalConfig(
            backend="custom",
            main_target="target",
            metrics=("corr", "sharpe"),
            validation_scorecard=False,
        ),
        run=RunConfig(
            seed=17,
            artifacts_dir=tmp_path / "artifacts",
            name="custom-registry-test",
        ),
    )


class ConstantAdapter:
    def __init__(
        self,
        proto: Any,
        *,
        name: str = "constant_model",
        identity_name: str | None = None,
        adapter_version: str = "1",
        implementation_fingerprint: str = "a" * 64,
        dependency_identity: dict[str, str] | None = None,
        constant_value: float = 0.125,
    ) -> None:
        self.name = name
        self.adapter_version = adapter_version
        self.capabilities = proto.BackendCapabilities(
            supports_gpu=False,
            supports_full_history=True,
            supports_deployment=True,
            deployment_device="cpu",
        )
        self.fit_error_types = (ValueError, TypeError)
        self._proto = proto
        self._identity_name = identity_name or name.lower()
        self._implementation_fingerprint = implementation_fingerprint
        self._dependency_identity = dict(
            {"custom-backend": "1.0"}
            if dependency_identity is None
            else dependency_identity
        )
        self.constant_value = constant_value

    def resolve_params(self, *, preset, params, n_features):
        return {"preset": preset, "params": dict(params), "n_features": n_features}

    def build_model(self, *, resolved_params, seed, device):
        return {
            "resolved_params": dict(resolved_params),
            "seed": seed,
            "device": device,
        }

    def fit(self, model, features, target, *, progress=None):
        if progress is not None:
            progress("fit")
        model["rows"] = int(len(features))
        return model

    def predict(self, model, features):
        del model
        return np.full(len(features), self.constant_value, dtype=float)

    def identity(self, *, resolved_params, device):
        return _valid_identity(
            self._proto,
            name=self._identity_name,
            adapter_version=self.adapter_version,
            implementation_fingerprint=self._implementation_fingerprint,
            resolved_params=dict(resolved_params),
            device=device,
            capabilities=self.capabilities,
            dependency_identity=dict(self._dependency_identity),
        )


class _NonDeepcopyableMutableState:
    def __init__(self) -> None:
        self.values = [1, 2, 3]

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        del memo
        raise TypeError("state does not support deepcopy")


class _DictStateAdapter(ConstantAdapter):
    def __init__(self, proto: Any, *, state: _NonDeepcopyableMutableState) -> None:
        super().__init__(
            proto, name="dict_state_model", identity_name="dict_state_model"
        )
        self.custom_state = state


class _SlotsStateAdapter:
    __slots__ = (
        "name",
        "adapter_version",
        "capabilities",
        "fit_error_types",
        "_proto",
        "_identity_name",
        "_implementation_fingerprint",
        "_dependency_identity",
        "constant_value",
        "state",
    )

    def __init__(self, proto: Any, *, state: _NonDeepcopyableMutableState) -> None:
        self.name = "slots_state_model"
        self.adapter_version = "1"
        self.capabilities = proto.BackendCapabilities(
            supports_gpu=False,
            supports_full_history=True,
            supports_deployment=True,
            deployment_device="cpu",
        )
        self.fit_error_types = (ValueError, TypeError)
        self._proto = proto
        self._identity_name = "slots_state_model"
        self._implementation_fingerprint = "a" * 64
        self._dependency_identity = {"custom-backend": "1.0"}
        self.constant_value = 0.125
        self.state = state

    def resolve_params(self, *, preset, params, n_features):
        return {"preset": preset, "params": dict(params), "n_features": n_features}

    def build_model(self, *, resolved_params, seed, device):
        return {
            "resolved_params": dict(resolved_params),
            "seed": seed,
            "device": device,
        }

    def fit(self, model, features, target, *, progress=None):
        if progress is not None:
            progress("fit")
        model["rows"] = int(len(features))
        return model

    def predict(self, model, features):
        del model
        return np.full(len(features), self.constant_value, dtype=float)

    def identity(self, *, resolved_params, device):
        return _valid_identity(
            self._proto,
            name=self._identity_name,
            adapter_version=self.adapter_version,
            implementation_fingerprint=self._implementation_fingerprint,
            resolved_params=dict(resolved_params),
            device=device,
            capabilities=self.capabilities,
            dependency_identity=dict(self._dependency_identity),
        )


def test_register_duplicate_and_unknown_name_errors() -> None:
    registry_module = _registry_module()
    proto = _protocol_module()
    registry = registry_module.BackendRegistry()

    registry.register(ConstantAdapter(proto, name="custom_model"))

    with pytest.raises(ValueError, match="duplicate"):
        registry.register(ConstantAdapter(proto, name="CUSTOM_MODEL"))

    with pytest.raises(KeyError, match="unknown"):
        registry.resolve("missing_backend")


def test_register_normalizes_name_and_rejects_invalid_syntax() -> None:
    registry_module = _registry_module()
    proto = _protocol_module()
    registry = registry_module.BackendRegistry()

    uppercase = ConstantAdapter(
        proto,
        name="Custom_Model",
        identity_name="custom_model",
    )
    registry.register(uppercase)

    assert registry.resolve("custom_model") is uppercase
    assert registry.resolve("CUSTOM_MODEL") is uppercase

    with pytest.raises(ValueError, match="identifier"):
        registry.register(
            ConstantAdapter(proto, name="bad-name", identity_name="bad_name")
        )


def test_register_rejects_invalid_identity_and_unsupported_shape() -> None:
    registry_module = _registry_module()
    proto = _protocol_module()
    registry = registry_module.BackendRegistry()

    with pytest.raises(ValueError, match="identity"):
        registry.register(
            ConstantAdapter(
                proto,
                name="custom_model",
                identity_name="different_name",
            )
        )

    class IncompleteAdapter:
        name = "custom_model"
        adapter_version = "1"
        capabilities = _valid_capabilities(proto)

    with pytest.raises(TypeError, match="BackendAdapter"):
        registry.register(IncompleteAdapter())  # type: ignore[arg-type]


def test_snapshot_is_isolated_and_sealed() -> None:
    registry_module = _registry_module()
    proto = _protocol_module()
    registry = registry_module.BackendRegistry()
    registry.register(
        ConstantAdapter(proto, name="alpha_model", identity_name="alpha_model")
    )

    snapshot = registry.snapshot()
    registry.register(
        ConstantAdapter(proto, name="beta_model", identity_name="beta_model")
    )

    assert snapshot.resolve("alpha_model").name == "alpha_model"
    with pytest.raises(KeyError, match="unknown"):
        snapshot.resolve("beta_model")
    with pytest.raises(RuntimeError, match="sealed"):
        snapshot.register(
            ConstantAdapter(proto, name="gamma_model", identity_name="gamma_model")
        )


def test_snapshot_is_independent_from_source_adapter_mutation() -> None:
    registry_module = _registry_module()
    proto = _protocol_module()
    adapter = ConstantAdapter(
        proto,
        name="constant_model",
        identity_name="constant_model",
        adapter_version="1",
        implementation_fingerprint="a" * 64,
        dependency_identity={"custom-backend": "1.0"},
        constant_value=0.125,
    )
    registry = registry_module.BackendRegistry()
    registry.register(adapter)

    snapshot = registry.snapshot()
    snapped_adapter = snapshot.resolve("constant_model")
    before = snapped_adapter.identity(resolved_params={}, device="cpu")

    adapter.adapter_version = "2"
    adapter._implementation_fingerprint = "b" * 64
    adapter._dependency_identity = {"custom-backend": "2.0"}
    adapter.constant_value = 0.875

    after = snapped_adapter.identity(resolved_params={}, device="cpu")
    assert snapped_adapter is not adapter
    assert after == before
    assert snapped_adapter.predict({}, np.zeros((3, 1), dtype=float)).tolist() == [
        0.125,
        0.125,
        0.125,
    ]


def test_snapshot_refuses_non_cloneable_dict_state() -> None:
    registry_module = _registry_module()
    proto = _protocol_module()
    adapter = _DictStateAdapter(proto, state=_NonDeepcopyableMutableState())
    registry = registry_module.BackendRegistry()
    registry.register(adapter)

    with pytest.raises(TypeError, match="clone-safe state") as excinfo:
        registry.snapshot()

    assert "custom_state" in str(excinfo.value)


def test_snapshot_refuses_non_cloneable_slots_state() -> None:
    registry_module = _registry_module()
    proto = _protocol_module()
    adapter = _SlotsStateAdapter(proto, state=_NonDeepcopyableMutableState())
    registry = registry_module.BackendRegistry()
    registry.register(adapter)

    with pytest.raises(TypeError, match="clone-safe state") as excinfo:
        registry.snapshot()

    assert "state" in str(excinfo.value)


def test_identity_is_registration_order_independent() -> None:
    registry_module = _registry_module()
    proto = _protocol_module()
    first = registry_module.BackendRegistry()
    second = registry_module.BackendRegistry()

    alpha = ConstantAdapter(
        proto,
        name="alpha_model",
        identity_name="alpha_model",
        implementation_fingerprint="a" * 64,
    )
    beta = ConstantAdapter(
        proto,
        name="beta_model",
        identity_name="beta_model",
        implementation_fingerprint="b" * 64,
    )
    first.register(alpha)
    first.register(beta)
    second.register(beta)
    second.register(alpha)

    assert first.identity() == second.identity()


def test_with_builtins_exposes_builtin_backend_names_including_ridge() -> None:
    registry_module = _registry_module()
    registry = registry_module.BackendRegistry.with_builtins()

    assert tuple(sorted(registry.identity()["adapters_by_name"])) == (
        "catboost",
        "lightgbm",
        "ridge",
        "xgboost",
    )
    assert registry.resolve("ridge").name == "ridge"


def test_with_builtins_returns_fresh_instance_each_time() -> None:
    registry_module = _registry_module()
    first = registry_module.BackendRegistry.with_builtins()
    second = registry_module.BackendRegistry.with_builtins()

    assert first is not second
    assert first.identity() == second.identity()


def test_custom_backend_is_not_process_global() -> None:
    registry_module = _registry_module()
    proto = _protocol_module()

    custom_registry = registry_module.BackendRegistry.with_builtins()
    custom_registry.register(ConstantAdapter(proto))

    with pytest.raises(KeyError, match="unknown"):
        registry_module.BackendRegistry.with_builtins().resolve("constant_model")


def test_custom_backend_runs_purged_oof_with_explicit_registry() -> None:
    from nmr.config import ModelConfig
    from nmr.models import ModelOrchestrator

    registry_module = _registry_module()
    proto = _protocol_module()
    registry = registry_module.BackendRegistry.with_builtins()
    registry.register(ConstantAdapter(proto, constant_value=0.25))
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="constant_model", preset="fast"),
        seed=7,
        backend_registry=registry,
    )
    splitter = _walk_forward_splitter()
    result = orchestrator.train_cross_validation(
        _model_frame(),
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=splitter,
    )
    expected_val_eras = {
        era
        for fold in splitter.split(_model_frame().get_column("era").to_list())
        for era in fold.val_eras
    }

    assert set(result.oof.get_column("era").to_list()) == expected_val_eras
    assert result.oof.get_column("prediction").to_list() == [0.25] * result.oof.height
    assert orchestrator.resolved_device == "cpu"


def test_runner_snapshot_keeps_custom_manifest_identity_stable(tmp_path) -> None:
    from nmr import paths
    from nmr.runner import ExperimentRunner

    registry_module = _registry_module()
    proto = _protocol_module()
    adapter = ConstantAdapter(
        proto,
        adapter_version="1",
        implementation_fingerprint="a" * 64,
        dependency_identity={"custom-backend": "1.0"},
    )
    registry = registry_module.BackendRegistry.with_builtins()
    registry.register(adapter)

    runner = ExperimentRunner(
        _runner_config(tmp_path, backend="constant_model"),
        backend_registry=registry,
    )
    adapter.adapter_version = "2"
    adapter._implementation_fingerprint = "b" * 64
    adapter._dependency_identity = {"custom-backend": "2.0"}

    result = runner.run(deploy=False)
    checkpoint_manifest = json.loads(
        (
            paths.run_dir(result.manifest["config"]["run"]["name"], result.run_id)
            / "oof_checkpoints"
            / "target"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert (
        result.manifest["selected_backend_identity"]["backend_name"] == "constant_model"
    )
    assert result.manifest["selected_backend_identity"]["adapter_version"] == "1"
    assert result.manifest["selected_backend_identity"]["dependency_identity"] == {
        "custom-backend": "1.0"
    }
    assert (
        "constant_model"
        in result.manifest["backend_registry_audit_identity"]["adapter_names"]
    )
    assert (
        checkpoint_manifest["selected_backend_identity"]
        == result.manifest["selected_backend_identity"]
    )
    assert (
        checkpoint_manifest["backend_registry_audit_identity"]
        == result.manifest["backend_registry_audit_identity"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected_value"),
    [
        ("adapter_version", "2"),
        ("implementation_fingerprint", "b" * 64),
        ("dependency_identity", {"custom-backend": "2.0"}),
    ],
)
def test_custom_checkpoint_reuse_refuses_selected_identity_mutation(
    tmp_path, mutation: str, expected_value: object
) -> None:
    from nmr.config import ModelConfig
    from nmr.models import ModelOrchestrator

    registry_module = _registry_module()
    proto = _protocol_module()
    adapter = ConstantAdapter(proto)
    registry = registry_module.BackendRegistry.with_builtins()
    registry.register(adapter)
    checkpoint_dir = tmp_path / "oof_checkpoints"
    splitter = _walk_forward_splitter()

    ModelOrchestrator(
        ModelConfig(backend="constant_model", preset="fast"),
        seed=7,
        backend_registry=registry,
    ).train_oof_with_checkpoints(
        _model_frame(),
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=splitter,
        checkpoint_dir=checkpoint_dir,
    )

    if mutation == "adapter_version":
        adapter.adapter_version = str(expected_value)
    elif mutation == "implementation_fingerprint":
        adapter._implementation_fingerprint = str(expected_value)
    else:
        adapter._dependency_identity = dict(expected_value)

    with pytest.raises(ValueError, match="selected_backend_identity"):
        ModelOrchestrator(
            ModelConfig(backend="constant_model", preset="fast"),
            seed=7,
            backend_registry=registry,
        ).train_oof_with_checkpoints(
            _model_frame(),
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            splitter=splitter,
            checkpoint_dir=checkpoint_dir,
        )
