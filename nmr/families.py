"""Read-only model-family / full-version discovery layer.

A model "family" is identified by its research runs' ``run.name`` (e.g.
``brb1-xgb-v6``). Promotion to a full version (trained on train+validation)
is marked by a valid manifest at ``artifacts/models/<family>/full/manifest.json``.
This module is strictly read-only: it never writes to ``artifacts/models/``
or ``artifacts/registry/``; the promotion writer is a future workstream.
Nothing here enters a canonical hash — it is display metadata only.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nmr.config import REPO_ROOT

logger = logging.getLogger("nmr.families")

FAMILY_DIR_NAME = "models"
FULL_DIR_NAME = "full"
FULL_MANIFEST_NAME = "manifest.json"
DEFAULT_MODELS_DIR = REPO_ROOT / "artifacts" / FAMILY_DIR_NAME

# Lowercase-only: prevents case-collision overwrites on case-insensitive
# filesystems (Windows NTFS / macOS APFS) and matches the run.name convention.
_FAMILY_NAME_RE = re.compile(r"^[a-z0-9_-]+$")

__all__ = [
    "DEFAULT_MODELS_DIR",
    "FAMILY_DIR_NAME",
    "FULL_DIR_NAME",
    "FULL_MANIFEST_NAME",
    "FullVersion",
    "family_has_full_version",
    "full_manifest_path",
    "load_full_version",
    "scan_full_versions",
]


@dataclass(frozen=True)
class FullVersion:
    """A validated full-version promotion of a model family."""

    family: str
    manifest_path: Path
    artifact_path: str | None
    promoted_from_run_id: str | None
    promoted_at: str | None
    config: dict[str, Any]


def _require_valid_family(family: str) -> str:
    if not _FAMILY_NAME_RE.fullmatch(family):
        raise ValueError(
            f"invalid family name {family!r}: must match {_FAMILY_NAME_RE.pattern}"
        )
    return family


def full_manifest_path(models_dir: Path, family: str) -> Path:
    """Resolve ``models_dir/<family>/full/manifest.json`` (family validated)."""
    _require_valid_family(family)
    return Path(models_dir) / family / FULL_DIR_NAME / FULL_MANIFEST_NAME


_DRIVE_LETTER_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_portable_relative(candidate: Path) -> bool:
    """True when ``candidate`` is relative on every platform.

    ``is_absolute()`` alone is platform-specific: ``C:\\...`` is not absolute
    on POSIX and ``/abs/...`` is not absolute on Windows, so root/drive and
    drive-letter forms are rejected explicitly.
    """
    return (
        not candidate.is_absolute()
        and not candidate.root
        and not candidate.drive
        and not _DRIVE_LETTER_RE.match(str(candidate))
        and ".." not in candidate.parts
    )


def _validate_artifact(manifest_dir: Path, artifact_path: object) -> str | None:
    """Artifact must be a non-empty relative path (no /, drive, or ..) whose
    file exists beside the manifest. None on any violation."""
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return None
    candidate = Path(artifact_path)
    if not _is_portable_relative(candidate):
        return None
    if not (manifest_dir / candidate).is_file():
        return None
    return candidate.as_posix()


def load_full_version(models_dir: Path, family: str) -> FullVersion | None:
    """Load and validate the family's full-version manifest, or None.

    None when: manifest missing, JSON invalid, or any validation rule fails
    (family != dir name, scope != "full", missing run id, missing/invalid
    artifact). A warning is logged on structural violations.
    """
    _require_valid_family(family)
    path = full_manifest_path(models_dir, family)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("nmr.families: corrupt manifest at %s", path)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("family") != family:
        logger.warning(
            "nmr.families: manifest family %r != dir %r",
            payload.get("family"), family,
        )
        return None
    if payload.get("training_scope") != "full":
        return None
    run_id = payload.get("promoted_from_run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return None
    artifact = _validate_artifact(path.parent, payload.get("artifact_path"))
    if artifact is None:
        logger.warning(
            "nmr.families: hollow or invalid artifact for %s (artifact_path=%r)",
            family, payload.get("artifact_path"),
        )
        return None
    config = payload.get("config")
    return FullVersion(
        family=family,
        manifest_path=path,
        artifact_path=artifact,
        promoted_from_run_id=run_id,
        promoted_at=payload.get("promoted_at"),
        config=config if isinstance(config, dict) else {},
    )


def scan_full_versions(models_dir: Path) -> dict[str, FullVersion]:
    """Return ``{family: FullVersion}`` for every valid manifest under models_dir."""
    base = Path(models_dir)
    if not base.is_dir():
        return {}
    found: dict[str, FullVersion] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not _FAMILY_NAME_RE.fullmatch(entry.name):
            continue
        version = load_full_version(base, entry.name)
        if version is not None:
            found[version.family] = version
    return found


def family_has_full_version(models_dir: Path, family: str) -> bool:
    """True when the family has a valid full-version manifest."""
    return load_full_version(models_dir, family) is not None
