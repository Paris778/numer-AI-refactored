from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


def _protocol_module():
    return importlib.import_module("nmr.model_backend_protocol")


def _valid_capabilities(proto: object):
    return proto.BackendCapabilities(
        supports_gpu=False,
        supports_full_history=True,
        supports_deployment=False,
        deployment_device="cpu",
    )


def _valid_identity(proto: object, **overrides: object):
    values = {
        "schema_version": 1,
        "name": "test",
        "adapter_version": "1",
        "implementation_fingerprint": "a" * 64,
        "resolved_params": {"alpha": 1.0, "enabled": True},
        "device": "cpu",
        "capabilities": _valid_capabilities(proto),
        "dependency_identity": {"stdlib": "python-3.11"},
    }
    values.update(overrides)
    return proto.BackendIdentity(**values)


def test_backend_adapter_protocol_accepts_minimal_adapter() -> None:
    proto = _protocol_module()

    class TestAdapter:
        name = "test"
        adapter_version = "1"
        capabilities = _valid_capabilities(proto)

        def resolve_params(self, *, preset, params, n_features):
            return {"alpha": 1.0, "preset": preset, "n_features": n_features}

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
            return np.zeros(len(features), dtype=float)

        def identity(self, *, resolved_params, device):
            return _valid_identity(
                proto,
                resolved_params=resolved_params,
                device=device,
            )

    adapter = TestAdapter()
    assert isinstance(adapter, proto.BackendAdapter)

    resolved = adapter.resolve_params(preset="fast", params={}, n_features=42)
    assert resolved["preset"] == "fast"

    identity = adapter.identity(resolved_params=resolved, device="cpu")
    assert identity.name == "test"
    assert identity.schema_version == 1


def test_canonical_json_bytes_sorts_keys_and_normalizes_numpy_scalars() -> None:
    proto = _protocol_module()

    value_a = {
        "zeta": np.int64(7),
        "alpha": {
            "truthy": np.bool_(True),
            "weight": np.float64(1.25),
        },
        "items": [np.int32(3), np.float32(2.5)],
    }
    value_b = {
        "items": [np.int32(3), np.float32(2.5)],
        "alpha": {
            "weight": np.float64(1.25),
            "truthy": np.bool_(True),
        },
        "zeta": np.int64(7),
    }

    normalized = proto.normalize_identity_value(value_a)
    assert normalized == {
        "alpha": {"truthy": True, "weight": 1.25},
        "items": [3, 2.5],
        "zeta": 7,
    }
    assert type(normalized["alpha"]["truthy"]) is bool
    assert type(normalized["alpha"]["weight"]) is float
    assert type(normalized["items"][0]) is int
    assert type(normalized["items"][1]) is float
    assert type(normalized["zeta"]) is int
    assert proto.canonical_json_bytes(value_a) == proto.canonical_json_bytes(value_b)
    assert (
        proto.canonical_json_bytes(value_a)
        == b'{"alpha":{"truthy":true,"weight":1.25},"items":[3,2.5],"zeta":7}'
    )


@pytest.mark.parametrize(
    "value",
    [
        Path("artifact.pkl"),
        float("nan"),
        float("inf"),
        {1: "bad-key"},
        np.float64(np.nan),
        lambda: None,
    ],
    ids=["path", "nan", "inf", "non-string-key", "numpy-nan", "callable"],
)
def test_normalize_identity_value_rejects_unsupported_or_non_finite_values(
    value: object,
) -> None:
    proto = _protocol_module()

    with pytest.raises((TypeError, ValueError)):
        proto.normalize_identity_value(value)

    with pytest.raises((TypeError, ValueError)):
        proto.canonical_json_bytes({"value": value})


def test_normalize_identity_value_rejects_arbitrary_object_instances() -> None:
    proto = _protocol_module()

    class FakeEstimator:
        def fit(self) -> None:
            return None

    with pytest.raises(TypeError):
        proto.normalize_identity_value(FakeEstimator())

    with pytest.raises(TypeError):
        proto.canonical_json_bytes({"estimator": FakeEstimator()})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 0),
        ("schema_version", 2),
        ("name", ""),
        ("name", "Test"),
        ("name", "test-backend"),
        ("implementation_fingerprint", "a" * 63),
        ("implementation_fingerprint", "A" * 64),
        ("device", "auto"),
        ("dependency_identity", {"stdlib": ""}),
        ("dependency_identity", {"": "python-3.11"}),
        ("dependency_identity", {1: "python-3.11"}),
    ],
    ids=[
        "schema-zero",
        "schema-two",
        "empty-name",
        "capitalized-name",
        "hyphenated-name",
        "short-fingerprint",
        "uppercase-fingerprint",
        "unsupported-device",
        "empty-dependency-version",
        "empty-dependency-name",
        "non-string-dependency-name",
    ],
)
def test_backend_identity_validates_invalid_values(field: str, value: object) -> None:
    proto = _protocol_module()

    with pytest.raises((TypeError, ValueError)):
        _valid_identity(proto, **{field: value})


def test_backend_capabilities_validate_device_and_boolean_rules() -> None:
    proto = _protocol_module()

    with pytest.raises(ValueError):
        proto.BackendCapabilities(
            supports_gpu=False,
            supports_full_history=True,
            supports_deployment=True,
            deployment_device="none",
        )

    with pytest.raises(ValueError):
        proto.BackendCapabilities(
            supports_gpu=False,
            supports_full_history=True,
            supports_deployment=False,
            deployment_device="gpu",
        )

    with pytest.raises(TypeError):
        proto.BackendCapabilities(
            supports_gpu=1,
            supports_full_history=True,
            supports_deployment=False,
            deployment_device="cpu",
        )
