"""Deployment artifact serialization with provenance and integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cloudpickle

__all__ = ["DeploymentArtifact", "load_predict", "serialize_predict"]


@dataclass(frozen=True)
class DeploymentArtifact:
    path: Path
    manifest: dict[str, Any]


def serialize_predict(
    predict_fn,
    *,
    path: str | Path,
    feature_names: Sequence[str],
    models=None,
) -> DeploymentArtifact:
    """Serialize `predict_fn` and write an integrity manifest.

    Security model: the sibling manifest SHA-256 protects against accidental
    corruption, not authenticity. An attacker who can edit files can modify both
    payload and manifest hash.
    """
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "predict_fn": predict_fn,
        "models": models,
    }
    payload_bytes = cloudpickle.dumps(payload)
    artifact_path.write_bytes(payload_bytes)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(feature_names),
        "sha256": _sha256_bytes(payload_bytes),
        "environment": _environment_fingerprint(),
    }
    _manifest_path(artifact_path).write_text(
        json.dumps(manifest, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return DeploymentArtifact(path=artifact_path, manifest=manifest)


def load_predict(path: str | Path) -> Callable:
    """Load a serialized `predict` callable after integrity check.

    Security model: this verifies payload integrity against the local manifest,
    but it does not establish trusted origin. `cloudpickle.loads` executes
    arbitrary code, so only load artifacts from trusted sources.
    """
    artifact_path = Path(path)
    manifest = json.loads(_manifest_path(artifact_path).read_text(encoding="utf-8"))
    payload_bytes = artifact_path.read_bytes()
    actual_hash = _sha256_bytes(payload_bytes)
    expected_hash = manifest.get("sha256")
    if actual_hash != expected_hash:
        raise ValueError(
            f"Artifact SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )

    payload = cloudpickle.loads(payload_bytes)
    predict_fn = payload["predict_fn"]
    if not callable(predict_fn):
        raise TypeError("Serialized artifact does not contain a callable predict_fn")
    return predict_fn


def _manifest_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(f"{artifact_path.name}.manifest.json")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _environment_fingerprint() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os_name": os.name,
        "packages": {
            name: _package_version(name)
            for name in [
                "cloudpickle",
                "numpy",
                "pandas",
                "polars",
                "numerai-tools",
            ]
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
