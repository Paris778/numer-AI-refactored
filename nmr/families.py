"""Read-only model-family / full-version discovery — thin wrapper over nmr.lifecycle.

A model "family" is identified by its research runs' ``run.name`` (e.g.
``brb1-xgb-v6``). Promotion to a full version (trained on train+validation)
publishes one immutable slot per promoted run at
``experiments/<family>/exports/full/<run_id>/`` plus an atomic ``current.json``
pointer naming the active slot (written by ``nmr/promote.py``, the promotion
writer). This module is a compatibility wrapper: it re-exports
``lifecycle.ExportVersion`` as ``FullVersion`` and delegates all resolution to
``nmr.lifecycle`` / ``nmr.paths``. Resolution is pointer-driven and fails loud
on a missing/dangling pointer — it never guesses slots by mtime. Nothing here
enters a canonical hash — it is display metadata only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nmr import lifecycle, paths

logger = logging.getLogger("nmr.families")

# Legacy constants retained for call-site compatibility (Task 11 may drop
# the ones no longer referenced once the dashboard and CLI are retargeted).
FAMILY_DIR_NAME = "models"
FULL_DIR_NAME = "full"
FULL_MANIFEST_NAME = "manifest.json"
CURRENT_POINTER_NAME = "current.json"
DEFAULT_MODELS_DIR = paths.DEFAULT_ARTIFACTS_DIR / FAMILY_DIR_NAME

# The full-version record type IS the lifecycle export version (scope "full").
FullVersion = lifecycle.ExportVersion

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


def validate_family_name(family: str) -> str:
    """Validate a family name for the promotion writer; raises ValueError.

    Delegates the slug rule to ``paths.validate_slug`` but preserves the
    promotion-domain error message (pinned by ``tests/test_promote.py``).
    """
    try:
        return paths.validate_slug(family)
    except ValueError as exc:
        raise ValueError(
            f"invalid family name {family!r}: must match {paths.SLUG_RE.pattern}"
        ) from exc


def full_manifest_path(models_dir: Path, family: str) -> Path:
    """DEPRECATED compat shim (removed in Task 11): the current full slot's
    ``export.json`` path, fail loud on a missing/dangling pointer.

    The pre-rebuild layout stored the slot record as ``manifest.json`` under
    ``artifacts/models/<family>/full/<run_id>/``; both moved (the record is
    ``export.json`` under ``experiments/<family>/exports/full/<run_id>/``).
    Resolution stays pointer-driven; prefer ``lifecycle.valid_export`` +
    ``paths.current_pointer_path`` over this shim.
    """
    logger.warning(
        "nmr.families.full_manifest_path is deprecated; resolve via "
        "nmr.lifecycle.valid_export / nmr.paths.current_pointer_path"
    )
    if lifecycle.current_full_status(family) != "full":
        raise FileNotFoundError(
            f"no current full version for family {family!r}: "
            f"{paths.current_pointer_path(family)} missing or dangling; "
            f"available slots: {available_slots(models_dir, family) or 'none'}"
        )
    pointer = json.loads(paths.current_pointer_path(family).read_text(encoding="utf-8"))
    return paths.export_json_path(family, "full", pointer["run_id"])


def available_slots(models_dir: Path, family: str) -> list[str]:
    """Sorted run_ids of all full-version slot dirs (valid or not), excluding
    ``.tmp-`` staging dirs — the diagnostics for pointer-error messages.
    ``models_dir`` is accepted for call-site compatibility; the layout is
    ``paths.EXPERIMENTS_ROOT``.
    """
    base = paths.export_dir(family, "full", "x").parent
    if not base.is_dir():
        return []
    return sorted(
        entry.name
        for entry in base.iterdir()
        if entry.is_dir() and not entry.name.startswith(".tmp-")
    )


def load_full_version(models_dir: Path, family: str) -> FullVersion | None:
    """Load and validate the family's CURRENT full-version export, or None.

    None when no valid full export exists, the ``current.json`` pointer is
    missing/dangling/corrupt (a degraded family), or the pointed slot fails
    ``lifecycle.valid_export``. ``models_dir`` is accepted for call-site
    compatibility; the layout is ``paths.EXPERIMENTS_ROOT``.
    """
    if lifecycle.current_full_status(family) != "full":
        return None
    pointer = json.loads(paths.current_pointer_path(family).read_text(encoding="utf-8"))
    return lifecycle.valid_export(family, "full", pointer["run_id"])


def scan_full_versions(models_dir: Path) -> dict[str, FullVersion]:
    """Return ``{family: FullVersion}`` for every family with a GENUINE full version.

    Each family's entry is its pointer'd current export (``load_full_version``),
    falling back to the newest valid export when the pointer is missing or
    dangling (a degraded family). Rehearsal artifacts (``rehearsal: true``) are
    excluded: they are D7 truncated-subset artifacts, never the family's current
    full version, and must not drive ``has_full_version`` or the dashboard's
    FULL stamp. They remain discoverable via ``available_slots`` and the slot
    record itself. ``models_dir`` is accepted for call-site compatibility; the
    layout is ``paths.EXPERIMENTS_ROOT``.
    """
    base = paths.EXPERIMENTS_ROOT
    if not base.is_dir():
        return {}
    found: dict[str, FullVersion] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not paths.SLUG_RE.fullmatch(entry.name):
            continue
        family = entry.name
        version = load_full_version(models_dir, family)
        if version is None:
            version = next(
                (
                    v
                    for v in lifecycle.scan_valid_exports(family, "full")
                    if not v.rehearsal
                ),
                None,
            )
        if version is not None and not version.rehearsal:
            found[family] = version
    return found


def family_has_full_version(models_dir: Path, family: str) -> bool:
    """True when the family has a valid GENUINE (non-rehearsal) full version."""
    version = load_full_version(models_dir, family)
    return version is not None and not version.rehearsal
