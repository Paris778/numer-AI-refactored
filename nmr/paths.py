"""Pure path derivation for the experiment layout.

Single place that knows where anything lives under ``experiments/`` or the
shared machine cache. No module hardcodes ``experiments`` or ``artifacts``
strings. Nothing here reads/writes the filesystem or enters a canonical hash
except as config-provided values (shared helpers take ``artifacts_dir``).
"""
from __future__ import annotations

import re
from pathlib import Path

from nmr.config import REPO_ROOT

EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Lowercase-only: prevents case-collision overwrites on case-insensitive
# filesystems and matches the run.name / family convention.
SLUG_RE = re.compile(r"^[a-z0-9_-]+$")

# Run ids are 64-char lowercase hex — the record directory name and the
# pointer payload's run_id. A corrupt pointer with a non-hex run_id must never
# resolve an unexpected path.
RID_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_slug(slug: str) -> str:
    """Validate a family slug; raises ValueError."""
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"invalid family slug {slug!r}: must match {SLUG_RE.pattern}"
        )
    return slug


def validate_run_id(run_id: str) -> str:
    """Validate a run_id (64-char lowercase hex); raises ValueError."""
    if not isinstance(run_id, str) or not RID_RE.fullmatch(run_id):
        raise ValueError(
            f"run_id={run_id!r} is not a 64-char lowercase hex string"
        )
    return run_id


def experiment_dir(slug: str) -> Path:
    return EXPERIMENTS_ROOT / validate_slug(slug)


def run_dir(slug: str, run_id: str) -> Path:
    return experiment_dir(slug) / "runs" / run_id


def run_json_path(slug: str, run_id: str) -> Path:
    return run_dir(slug, run_id) / "run.json"


def export_dir(slug: str, scope: str, run_id: str) -> Path:
    if scope not in ("partial", "full"):
        raise ValueError(f"scope must be 'partial' or 'full', got {scope!r}")
    # A non-empty run_id must be a 64-hex run id — a corrupt pointer or a
    # stray directory name must never resolve an unexpected path. Empty is
    # accepted for legacy probe callers; the pointer path is built directly
    # (current_pointer_path) so no caller needs the empty form.
    if run_id:
        validate_run_id(run_id)
    return experiment_dir(slug) / "exports" / scope / run_id


def export_json_path(slug: str, scope: str, run_id: str) -> Path:
    return export_dir(slug, scope, run_id) / "export.json"


def current_pointer_path(slug: str) -> Path:
    # Built directly — the pointer is not an export slot, so it never goes
    # through export_dir's run_id validation.
    return experiment_dir(slug) / "exports" / "full" / "current.json"


def champion_path() -> Path:
    return EXPERIMENTS_ROOT / "champion.json"


def shared_cache_dir(artifacts_dir: Path | None = None) -> Path:
    return (artifacts_dir or DEFAULT_ARTIFACTS_DIR) / "cache"


def shared_reports_dir(artifacts_dir: Path | None = None) -> Path:
    return (artifacts_dir or DEFAULT_ARTIFACTS_DIR) / "reports"
