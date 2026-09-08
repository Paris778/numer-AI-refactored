"""Sealed per-run registry for backend adapters and built-in audit identity."""

from __future__ import annotations

import copy
import json
import re
import types
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .model_backend_protocol import (
    BackendAdapter,
    BackendIdentity,
    canonical_json_bytes,
)

__all__ = ["BackendRegistry"]

_REGISTRY_SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SNAPSHOT_CLONE_HOOK = "__backend_registry_clone__"


@dataclass(frozen=True)
class _RegisteredAdapter:
    adapter: BackendAdapter
    audit_identity: BackendIdentity


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("backend adapter name must be a non-empty identifier")
    normalized = name.lower()
    if _NAME_RE.fullmatch(normalized) is None:
        raise ValueError(
            "backend adapter name must be a lowercase identifier matching "
            "^[a-z][a-z0-9_]*$"
        )
    return normalized


def _default_builtin_adapters() -> dict[str, BackendAdapter]:
    from .model_backend_catboost import CatBoostAdapter
    from .model_backend_lightgbm import LightGBMAdapter
    from .model_backend_ridge import RidgeAdapter
    from .model_backend_xgboost import XGBoostAdapter

    return {
        "lightgbm": LightGBMAdapter(),
        "xgboost": XGBoostAdapter(),
        "catboost": CatBoostAdapter(),
        "ridge": RidgeAdapter(),
    }


def _canonical_mapping(value: object) -> dict[str, Any]:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _is_safe_shared_state_value(value: object) -> bool:
    if value is None or isinstance(
        value,
        (
            str,
            bytes,
            int,
            float,
            bool,
            complex,
            range,
            slice,
            type,
            types.ModuleType,
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
        ),
    ):
        return True
    if isinstance(value, tuple):
        return all(_is_safe_shared_state_value(item) for item in value)
    if isinstance(value, frozenset):
        return all(_is_safe_shared_state_value(item) for item in value)
    return False


def _clone_failure_message(*, adapter_name: str, state_name: str) -> str:
    return (
        f"backend adapter {adapter_name!r} attribute {state_name!r} must provide "
        "clone-safe state for BackendRegistry.snapshot() "
        "(support copy.deepcopy or define __backend_registry_clone__())"
    )


def _clone_adapter_state_value(
    value: object,
    *,
    adapter_name: str,
    state_name: str,
) -> object:
    if _is_safe_shared_state_value(value):
        return value
    clone_hook = getattr(value, _SNAPSHOT_CLONE_HOOK, None)
    if callable(clone_hook):
        try:
            cloned = clone_hook()
        except Exception as exc:
            raise TypeError(
                f"{_clone_failure_message(adapter_name=adapter_name, state_name=state_name)}: "
                f"clone hook raised {type(exc).__name__}: {exc}"
            ) from exc
        if cloned is value:
            raise TypeError(
                f"{_clone_failure_message(adapter_name=adapter_name, state_name=state_name)}: "
                "clone hook returned the original mutable object"
            )
        return cloned
    try:
        cloned = copy.deepcopy(value)
    except Exception as exc:
        raise TypeError(
            f"{_clone_failure_message(adapter_name=adapter_name, state_name=state_name)}: "
            f"deepcopy raised {type(exc).__name__}: {exc}"
        ) from exc
    if cloned is value:
        raise TypeError(
            f"{_clone_failure_message(adapter_name=adapter_name, state_name=state_name)}: "
            "deepcopy returned the original mutable object"
        )
    return cloned


def _clone_adapter(adapter: BackendAdapter) -> BackendAdapter:
    clone = adapter.__class__.__new__(adapter.__class__)
    adapter_name = getattr(adapter, "name", adapter.__class__.__name__)
    state = getattr(adapter, "__dict__", None)
    if isinstance(state, dict):
        clone.__dict__.update(
            {
                key: _clone_adapter_state_value(
                    value,
                    adapter_name=adapter_name,
                    state_name=key,
                )
                for key, value in state.items()
            }
        )

    slots = getattr(adapter.__class__, "__slots__", ())
    if isinstance(slots, str):
        slot_names = (slots,)
    else:
        slot_names = tuple(slots)
    for slot_name in slot_names:
        if slot_name == "__dict__" or not hasattr(adapter, slot_name):
            continue
        setattr(
            clone,
            slot_name,
            _clone_adapter_state_value(
                getattr(adapter, slot_name),
                adapter_name=adapter_name,
                state_name=slot_name,
            ),
        )
    return clone


class BackendRegistry:
    def __init__(
        self,
        *,
        adapters: Mapping[str, BackendAdapter] | None = None,
        sealed: bool = False,
    ) -> None:
        self._entries: dict[str, _RegisteredAdapter] = {}
        self._sealed = False
        if adapters is not None:
            self._register_many(adapters)
        self._sealed = sealed

    @classmethod
    def with_builtins(
        cls,
        *,
        builtin_adapters: Mapping[str, BackendAdapter] | None = None,
    ) -> BackendRegistry:
        return cls(
            adapters=(
                _default_builtin_adapters()
                if builtin_adapters is None
                else builtin_adapters
            )
        )

    def register(self, adapter: BackendAdapter) -> None:
        if self._sealed:
            raise RuntimeError("backend registry is sealed")
        if not isinstance(adapter, BackendAdapter):
            raise TypeError("adapter must satisfy the BackendAdapter protocol")
        normalized_name = _normalize_name(adapter.name)
        if normalized_name in self._entries:
            raise ValueError(f"duplicate backend adapter name: {normalized_name!r}")
        audit_identity = adapter.identity(resolved_params={}, device="cpu")
        if not isinstance(audit_identity, BackendIdentity):
            raise TypeError("adapter identity must return a BackendIdentity")
        if audit_identity.name != normalized_name:
            raise ValueError(
                "adapter identity name must match the normalized adapter name"
            )
        if audit_identity.capabilities != adapter.capabilities:
            raise ValueError(
                "adapter identity capabilities must match the adapter capabilities"
            )
        self._entries[normalized_name] = _RegisteredAdapter(
            adapter=adapter,
            audit_identity=audit_identity,
        )

    def resolve(self, name: str) -> BackendAdapter:
        normalized_name = _normalize_name(name)
        try:
            return self._entries[normalized_name].adapter
        except KeyError as exc:
            raise KeyError(f"unknown backend adapter: {normalized_name!r}") from exc

    def snapshot(self) -> BackendRegistry:
        return BackendRegistry(
            adapters={
                name: _clone_adapter(entry.adapter)
                for name, entry in self._entries.items()
            },
            sealed=True,
        )

    def identity(self) -> Mapping[str, Any]:
        adapters_by_name = {
            name: asdict(self._entries[name].audit_identity)
            for name in sorted(self._entries)
        }
        return _canonical_mapping(
            {
                "schema_version": _REGISTRY_SCHEMA_VERSION,
                "adapter_names": sorted(adapters_by_name),
                "adapters_by_name": adapters_by_name,
            }
        )

    def _register_many(self, adapters: Mapping[str, BackendAdapter]) -> None:
        for expected_name, adapter in adapters.items():
            normalized_expected = _normalize_name(expected_name)
            normalized_actual = _normalize_name(adapter.name)
            if normalized_expected != normalized_actual:
                raise ValueError(
                    "built-in adapter mapping keys must match adapter names after normalization"
                )
            self.register(adapter)
