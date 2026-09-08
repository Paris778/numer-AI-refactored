"""Protocol types and canonical identity helpers for model backends.

This module defines the narrow contract shared by future backend adapters
without depending on model libraries or training orchestration.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["BackendAdapter", "BackendCapabilities", "BackendIdentity"]

_IDENTITY_SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VALID_DEPLOYMENT_DEVICES = ("cpu", "none")
_VALID_IDENTITY_DEVICES = ("cpu", "gpu")


def _require_bool(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def normalize_identity_value(value: object) -> object:
    """Normalize recursively to canonical JSON-compatible Python values."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        native = float(value)
        if not math.isfinite(native):
            raise ValueError("non-finite floats are not supported in backend identity")
        return native
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not supported in backend identity")
        return value
    if isinstance(value, PurePath):
        raise TypeError("path objects are not supported in backend identity")
    if callable(value):
        raise TypeError("callables are not supported in backend identity")
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("backend identity mappings must use string keys")
            normalized[key] = normalize_identity_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [normalize_identity_value(item) for item in value]
    raise TypeError(f"unsupported backend identity value type: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value into deterministic UTF-8 bytes."""
    normalized = normalize_identity_value(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class BackendCapabilities:
    supports_gpu: bool
    supports_full_history: bool
    supports_deployment: bool
    deployment_device: str

    def __post_init__(self) -> None:
        _require_bool(self.supports_gpu, "supports_gpu")
        _require_bool(self.supports_full_history, "supports_full_history")
        _require_bool(self.supports_deployment, "supports_deployment")
        if self.deployment_device not in _VALID_DEPLOYMENT_DEVICES:
            raise ValueError(
                "deployment_device must be one of "
                f"{_VALID_DEPLOYMENT_DEVICES}, got {self.deployment_device!r}"
            )
        if self.supports_deployment and self.deployment_device != "cpu":
            raise ValueError(
                "supports_deployment=True requires deployment_device='cpu'"
            )


@dataclass(frozen=True)
class BackendIdentity:
    schema_version: int
    name: str
    adapter_version: str
    implementation_fingerprint: str
    resolved_params: Mapping[str, Any]
    device: str
    capabilities: BackendCapabilities
    dependency_identity: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != _IDENTITY_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {_IDENTITY_SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        name = _require_non_empty_string(self.name, "name")
        if _NAME_RE.fullmatch(name) is None:
            raise ValueError("name must match ^[a-z][a-z0-9_]*$, " f"got {self.name!r}")
        _require_non_empty_string(self.adapter_version, "adapter_version")
        if _SHA256_RE.fullmatch(self.implementation_fingerprint) is None:
            raise ValueError(
                "implementation_fingerprint must be a 64-character lowercase SHA-256 string"
            )
        if self.device not in _VALID_IDENTITY_DEVICES:
            raise ValueError(
                f"device must be one of {_VALID_IDENTITY_DEVICES}, got {self.device!r}"
            )
        if not isinstance(self.capabilities, BackendCapabilities):
            raise TypeError("capabilities must be a BackendCapabilities instance")
        if not isinstance(self.resolved_params, Mapping):
            raise TypeError("resolved_params must be a mapping")
        normalized_params = normalize_identity_value(dict(self.resolved_params))
        if not isinstance(normalized_params, dict):
            raise TypeError("resolved_params must normalize to a mapping")
        object.__setattr__(self, "resolved_params", normalized_params)
        if not isinstance(self.dependency_identity, Mapping):
            raise TypeError("dependency_identity must be a mapping")
        normalized_dependencies: dict[str, str] = {}
        for package_name, dependency_version in self.dependency_identity.items():
            key = _require_non_empty_string(package_name, "dependency_identity key")
            version = _require_non_empty_string(
                dependency_version,
                f"dependency_identity[{package_name!r}]",
            )
            normalized_dependencies[key] = version
        object.__setattr__(
            self,
            "dependency_identity",
            {
                key: normalized_dependencies[key]
                for key in sorted(normalized_dependencies)
            },
        )


@runtime_checkable
class BackendAdapter(Protocol):
    name: str
    adapter_version: str
    capabilities: BackendCapabilities

    def resolve_params(
        self,
        *,
        preset: str,
        params: Mapping[str, Any],
        n_features: int,
    ) -> dict[str, Any]: ...

    def build_model(
        self,
        *,
        resolved_params: Mapping[str, Any],
        seed: int,
        device: str,
    ) -> object: ...

    def fit(
        self,
        model: object,
        features: np.ndarray,
        target: np.ndarray,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> object: ...

    def predict(self, model: object, features: np.ndarray) -> np.ndarray: ...

    def identity(
        self,
        *,
        resolved_params: Mapping[str, Any],
        device: str,
    ) -> BackendIdentity: ...
