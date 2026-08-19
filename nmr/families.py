"""Read-only model-family / full-version discovery layer.

A model "family" is identified by its research runs' ``run.name`` (e.g.
``brb1-xgb-v6``). Promotion to a full version (trained on train+validation)
writes one immutable slot per promoted run at
``artifacts/models/<family>/full/<run_id>/manifest.json`` plus an atomic
``current.json`` pointer naming the active slot (written by ``nmr/promote.py``,
the promotion writer). This module is strictly read-only: it never writes to
``artifacts/models/`` or ``artifacts/registry/``. Resolution is pointer-driven
and fails loud on a missing/dangling pointer — it never guesses slots by mtime.
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
CURRENT_POINTER_NAME = "current.json"
DEFAULT_MODELS_DIR = REPO_ROOT / "artifacts" / FAMILY_DIR_NAME

# Lowercase-only: prevents case-collision overwrites on case-insensitive
# filesystems (Windows NTFS / macOS APFS) and matches the run.name convention.
_FAMILY_NAME_RE = re.compile(r"^[a-z0-9_-]+$")
# Full-version slots are keyed by the 64-hex run_id they were promoted from.
_SLOT_DIR_RE = re.compile(r"^[0-9a-f]{64}$")

__all__ = [
    "CURRENT_POINTER_NAME",
    "DEFAULT_MODELS_DIR",
    "FAMILY_DIR_NAME",
    "FULL_DIR_NAME",
    "FULL_MANIFEST_NAME",
    "FullVersion",
    "available_slots",
    "family_has_full_version",
    "full_manifest_path",
    "load_full_version",
    "scan_full_versions",
    "validate_family_name",
]


@dataclass(frozen=True)
class FullVersion:
    """A validated full-version promotion of a model family.

    ``rehearsal=True`` marks a D7 truncated rehearsal artifact: trained on a
    subset, never the family's current pointer, and excluded from
    :func:`scan_full_versions` / :func:`family_has_full_version` — it can
    never be read as a genuine full version at a glance.
    """

    family: str
    manifest_path: Path
    artifact_path: str | None
    promoted_from_run_id: str | None
    promoted_at: str | None
    config: dict[str, Any]
    rehearsal: bool = False
    training_rows: int | None = None
    training_era_range: tuple[int, int] | None = None


def _require_valid_family(family: str) -> str:
    if not _FAMILY_NAME_RE.fullmatch(family):
        raise ValueError(
            f"invalid family name {family!r}: must match {_FAMILY_NAME_RE.pattern}"
        )
    return family


def validate_family_name(family: str) -> str:
    """Validate a family name for the promotion writer; raises ValueError."""
    return _require_valid_family(family)


def full_manifest_path(models_dir: Path, family: str) -> Path:
    """Resolve the CURRENT full-version slot's manifest path, or fail loud.

    Layout: ``models_dir/<family>/full/<run_id>/manifest.json`` (one immutable
    slot per promoted run) plus an atomic ``current.json`` pointer naming the
    active slot. Reads the pointer and returns the pointed slot's manifest
    path. Raises ``FileNotFoundError`` — listing the available slots — when the
    pointer is missing, corrupt, or dangles. Never falls back to scanning slots
    by mtime (determinism discipline): an absent pointer is an explicit
    absence, not a guess.
    """
    _require_valid_family(family)
    full_dir = Path(models_dir) / family / FULL_DIR_NAME
    pointer = full_dir / CURRENT_POINTER_NAME
    if not pointer.is_file():
        raise FileNotFoundError(
            f"no current full version for family {family!r}: "
            f"{pointer} missing; available slots: "
            f"{available_slots(models_dir, family) or 'none'}"
        )
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise FileNotFoundError(
            f"corrupt current.json pointer for family {family!r}: {exc}"
        ) from exc
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    if not isinstance(run_id, str) or not run_id.strip():
        raise FileNotFoundError(
            f"current.json for family {family!r} has no run_id; "
            f"available slots: {available_slots(models_dir, family) or 'none'}"
        )
    manifest = full_dir / run_id / FULL_MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(
            f"current.json for family {family!r} points to missing slot "
            f"{run_id!r}; available slots: "
            f"{available_slots(models_dir, family) or 'none'}"
        )
    return manifest


def available_slots(models_dir: Path, family: str) -> list[str]:
    """Sorted run_ids of full-version slots that hold a manifest."""
    _require_valid_family(family)
    full_dir = Path(models_dir) / family / FULL_DIR_NAME
    if not full_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in full_dir.iterdir()
        if entry.is_dir()
        and _SLOT_DIR_RE.fullmatch(entry.name)
        and (entry / FULL_MANIFEST_NAME).is_file()
    )


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
    """Load and validate the family's CURRENT full-version manifest, or None.

    None when: no current pointer (fail-loud resolution is available via
    ``full_manifest_path`` / ``available_slots``), manifest missing, JSON
    invalid, or any validation rule fails (family != dir name, scope != "full",
    missing run id, missing/invalid artifact). A warning is logged on
    structural violations.
    """
    _require_valid_family(family)
    try:
        path = full_manifest_path(models_dir, family)
    except FileNotFoundError:
        return None
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
    era_range = payload.get("training_era_range")
    return FullVersion(
        family=family,
        manifest_path=path,
        artifact_path=artifact,
        promoted_from_run_id=run_id,
        promoted_at=payload.get("promoted_at"),
        config=config if isinstance(config, dict) else {},
        rehearsal=bool(payload.get("rehearsal", False)),
        training_rows=(
            int(payload["training_rows"])
            if isinstance(payload.get("training_rows"), (int, float))
            else None
        ),
        training_era_range=(
            (int(era_range[0]), int(era_range[1]))
            if isinstance(era_range, (list, tuple)) and len(era_range) == 2
            else None
        ),
    )


def scan_full_versions(models_dir: Path) -> dict[str, FullVersion]:
    """Return ``{family: FullVersion}`` for every valid GENUINE full version.

    Rehearsal artifacts (``rehearsal: true`` in the manifest) are excluded:
    they are D7 truncated-subset artifacts, never the family's current full
    version, and must not drive ``has_full_version`` or the dashboard's FULL
    stamp. They remain discoverable via ``available_slots`` and the slot
    manifest itself.
    """
    base = Path(models_dir)
    if not base.is_dir():
        return {}
    found: dict[str, FullVersion] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not _FAMILY_NAME_RE.fullmatch(entry.name):
            continue
        version = load_full_version(base, entry.name)
        if version is not None and not version.rehearsal:
            found[version.family] = version
    return found


def family_has_full_version(models_dir: Path, family: str) -> bool:
    """True when the family has a valid GENUINE (non-rehearsal) full version."""
    version = load_full_version(models_dir, family)
    return version is not None and not version.rehearsal
