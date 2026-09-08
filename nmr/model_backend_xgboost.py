from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from .model_backend_protocol import BackendCapabilities, BackendIdentity
from .models import _raise_to_colsample_floor, resolve_model_params


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


class XGBoostAdapter:
    name = "xgboost"
    adapter_version = "1"
    capabilities = BackendCapabilities(
        supports_gpu=True,
        supports_full_history=True,
        supports_deployment=True,
        deployment_device="cpu",
    )
    fit_error_types = (ValueError, TypeError, xgb.core.XGBoostError)

    def resolve_params(
        self,
        *,
        preset: str,
        params: Mapping[str, Any],
        n_features: int,
    ) -> dict[str, Any]:
        resolved = {
            "objective": "reg:squarederror",
            "n_jobs": 1,
            "verbosity": 0,
            "subsample": 1.0,
            "colsample_bylevel": 1.0,
            **resolve_model_params(preset, dict(params)),
        }
        num_leaves = resolved.pop("num_leaves", None)
        min_data_in_leaf = resolved.pop("min_data_in_leaf", None)
        if num_leaves is not None:
            resolved.setdefault("grow_policy", "lossguide")
            resolved.setdefault("max_leaves", num_leaves)
        elif "max_leaves" in resolved:
            resolved.setdefault("grow_policy", "lossguide")
        if min_data_in_leaf is not None:
            resolved.setdefault("min_child_weight", float(min_data_in_leaf))
        resolved["colsample_bytree"] = _raise_to_colsample_floor(
            float(resolved["colsample_bytree"]), n_features
        )
        resolved["tree_method"] = "hist"
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
        resolved["seed"] = seed
        resolved["device"] = "cuda" if device == "gpu" else "cpu"
        return resolved

    def build_model(
        self,
        *,
        resolved_params: Mapping[str, Any],
        seed: int,
        device: str,
    ) -> object:
        del seed, device
        return xgb.XGBRegressor(**dict(resolved_params))

    def fit(
        self,
        model: object,
        features: np.ndarray,
        target: np.ndarray,
        *,
        progress=None,
    ) -> object:
        del progress
        started = time.monotonic()
        print("[fit] xgboost training started", flush=True)
        fitted = model.fit(features, target)
        print(
            f"[fit] xgboost training done ({time.monotonic() - started:.1f}s)",
            flush=True,
        )
        return fitted

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
            dependency_identity={"xgboost": _package_version("xgboost")},
        )
