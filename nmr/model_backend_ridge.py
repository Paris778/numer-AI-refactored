from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .model_backend_protocol import BackendCapabilities, BackendIdentity

_VALID_SOLVERS = ("auto", "svd", "cholesky", "lsqr", "sparse_cg")
_DEFAULT_PARAMS = {
    "alpha": 1.0,
    "fit_intercept": True,
    "solver": "lsqr",
}


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _validated_feature_matrix(features: object) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("ridge features must be a 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("ridge features must be finite; got non-finite feature matrix")
    return matrix


def _validated_target_vector(target: object) -> np.ndarray:
    vector = np.asarray(target, dtype=float).reshape(-1)
    if not np.isfinite(vector).all():
        raise ValueError("ridge targets must be finite")
    return vector


@dataclass
class RidgeModel:
    params: Mapping[str, Any]
    scaler: StandardScaler = field(default_factory=StandardScaler)
    regressor: Ridge = field(init=False)
    fitted: bool = False

    def __post_init__(self) -> None:
        self.regressor = Ridge(**dict(self.params))

    def fit(self, features: object, target: object) -> RidgeModel:
        matrix = _validated_feature_matrix(features)
        vector = _validated_target_vector(target)
        scaled = self.scaler.fit_transform(matrix)
        self.regressor.fit(scaled, vector)
        self.fitted = True
        return self

    def predict(self, features: object) -> np.ndarray:
        if not self.fitted:
            raise ValueError("ridge model must be fitted before predict")
        matrix = _validated_feature_matrix(features)
        scaled = self.scaler.transform(matrix)
        prediction = np.asarray(self.regressor.predict(scaled), dtype=float).reshape(-1)
        if not np.isfinite(prediction).all():
            raise ValueError("ridge predictions must be finite")
        return prediction


class RidgeAdapter:
    name = "ridge"
    adapter_version = "1"
    capabilities = BackendCapabilities(
        supports_gpu=False,
        supports_full_history=True,
        supports_deployment=True,
        deployment_device="cpu",
    )
    fit_error_types = (ValueError, TypeError)

    def resolve_params(
        self,
        *,
        preset: str,
        params: Mapping[str, Any],
        n_features: int,
    ) -> dict[str, Any]:
        del preset, n_features
        unknown = sorted(set(params) - set(_DEFAULT_PARAMS))
        if unknown:
            raise ValueError(f"ridge got unknown params: {unknown}")
        resolved = dict(_DEFAULT_PARAMS)
        resolved.update(dict(params))
        alpha = float(resolved["alpha"])
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("ridge alpha must be finite and > 0")
        fit_intercept = resolved["fit_intercept"]
        if not isinstance(fit_intercept, bool):
            raise ValueError("ridge fit_intercept must be a bool")
        solver = resolved["solver"]
        if solver not in _VALID_SOLVERS:
            raise ValueError(
                f"ridge solver must be one of {_VALID_SOLVERS}, got {solver!r}"
            )
        return {
            "alpha": alpha,
            "fit_intercept": fit_intercept,
            "solver": solver,
        }

    def build_model(
        self,
        *,
        resolved_params: Mapping[str, Any],
        seed: int,
        device: str,
    ) -> object:
        del seed
        if device != "cpu":
            raise ValueError(
                "requested_device='gpu' is unsupported for the ridge backend"
            )
        return RidgeModel(params=dict(resolved_params))

    def fit(
        self,
        model: object,
        features: np.ndarray,
        target: np.ndarray,
        *,
        progress=None,
    ) -> object:
        del progress
        if not isinstance(model, RidgeModel):
            raise TypeError("ridge model must be a RidgeModel instance")
        return model.fit(features, target)

    def predict(self, model: object, features: np.ndarray) -> np.ndarray:
        if not isinstance(model, RidgeModel):
            raise TypeError("ridge model must be a RidgeModel instance")
        return model.predict(features)

    def identity(
        self,
        *,
        resolved_params: Mapping[str, Any],
        device: str,
    ) -> BackendIdentity:
        return BackendIdentity(
            schema_version=1,
            name=self.name,
            adapter_version=self.adapter_version,
            implementation_fingerprint=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            resolved_params=dict(resolved_params),
            device=device,
            capabilities=self.capabilities,
            dependency_identity={"scikit-learn": _package_version("scikit-learn")},
        )
