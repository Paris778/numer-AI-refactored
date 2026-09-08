from __future__ import annotations

import hashlib
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from .model_backend_protocol import BackendCapabilities, BackendIdentity
from .models import (
    _FIT_PROGRESS_PERIOD,
    _LGBM_COLSAMPLE_ALIASES,
    _raise_to_colsample_floor,
    resolve_model_params,
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


class LightGBMAdapter:
    name = "lightgbm"
    adapter_version = "1"
    capabilities = BackendCapabilities(
        supports_gpu=True,
        supports_full_history=True,
        supports_deployment=True,
        deployment_device="cpu",
    )
    fit_error_types = (ValueError, TypeError, lgb.basic.LightGBMError)

    def resolve_params(
        self,
        *,
        preset: str,
        params: Mapping[str, Any],
        n_features: int,
    ) -> dict[str, Any]:
        resolved = {
            "objective": "regression",
            "n_jobs": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            **resolve_model_params(preset, dict(params)),
        }
        for alias in _LGBM_COLSAMPLE_ALIASES:
            if alias in resolved:
                resolved[alias] = _raise_to_colsample_floor(
                    float(resolved[alias]), n_features
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
        resolved = self.resolve_params(
            preset=preset,
            params=params,
            n_features=n_features,
        )
        resolved["random_state"] = seed
        resolved["device_type"] = "gpu" if device == "gpu" else "cpu"
        return resolved

    def build_model(
        self,
        *,
        resolved_params: Mapping[str, Any],
        seed: int,
        device: str,
    ) -> object:
        del seed, device
        return lgb.LGBMRegressor(**dict(resolved_params))

    def fit(
        self,
        model: object,
        features: np.ndarray,
        target: np.ndarray,
        *,
        progress=None,
    ) -> object:
        del progress

        def _lgb_progress(env: Any) -> None:
            iteration = env.iteration + 1
            if iteration == 1 or iteration % _FIT_PROGRESS_PERIOD == 0:
                print(f"[fit] lightgbm iteration {iteration}", flush=True)

        return model.fit(features, target, callbacks=[_lgb_progress])

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
            dependency_identity={"lightgbm": _package_version("lightgbm")},
        )
