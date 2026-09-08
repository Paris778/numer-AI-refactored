from __future__ import annotations

import hashlib
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import catboost
import numpy as np

from .model_backend_protocol import BackendCapabilities, BackendIdentity
from .models import (
    _FIT_PROGRESS_PERIOD,
    _raise_to_colsample_floor,
    _translate_catboost,
    resolve_model_params,
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


class CatBoostAdapter:
    name = "catboost"
    adapter_version = "1"
    capabilities = BackendCapabilities(
        supports_gpu=False,
        supports_full_history=True,
        supports_deployment=True,
        deployment_device="cpu",
    )
    fit_error_types = (ValueError, TypeError, catboost.CatBoostError)

    def resolve_params(
        self,
        *,
        preset: str,
        params: Mapping[str, Any],
        n_features: int,
    ) -> dict[str, Any]:
        resolved = resolve_model_params(preset, dict(params))
        if "rsm" in resolved:
            resolved["rsm"] = _raise_to_colsample_floor(
                float(resolved["rsm"]), n_features
            )
        return resolved

    def resolve_device_params(
        self,
        *,
        preset: str,
        params: Mapping[str, Any],
        n_features: int,
        seed: int,
        device: str,
    ) -> dict[str, Any]:
        translated = _translate_catboost(
            resolve_model_params(preset, dict(params)),
            seed=seed,
            use_gpu=device == "gpu",
        )
        translated["rsm"] = _raise_to_colsample_floor(
            float(translated["rsm"]), n_features
        )
        return translated

    def build_model(
        self,
        *,
        resolved_params: Mapping[str, Any],
        seed: int,
        device: str,
    ) -> object:
        del seed, device
        return catboost.CatBoostRegressor(**dict(resolved_params))

    def fit(
        self,
        model: object,
        features: np.ndarray,
        target: np.ndarray,
        *,
        progress=None,
    ) -> object:
        del progress
        return model.fit(features, target, verbose=_FIT_PROGRESS_PERIOD)

    def predict(self, model: object, features: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(features), dtype=float).reshape(-1)

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
            dependency_identity={"catboost": _package_version("catboost")},
        )
