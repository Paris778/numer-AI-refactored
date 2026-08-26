"""Run/export persistence and atomic publication for the experiment layout.

``record_run`` creates the family scaffold atomically with the first run.json
(spec §2 family-creation rule). Export slots are staged under ``.tmp-<run_id>``
and published by a single directory rename; discovery ignores ``.tmp-`` names.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from nmr import paths
from nmr._atomicio import atomic_write_text

logger = logging.getLogger("nmr.experiment_store")

_RID_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RID_RE.fullmatch(run_id):
        raise ValueError(f"run_id must be 64-char lowercase hex, got {run_id!r}")
    return run_id


def _write_scaffold(slug: str, *, display_name: str, base_config: dict[str, Any]) -> Path:
    exp = paths.experiment_dir(slug)
    exp.mkdir(parents=True, exist_ok=True)
    meta_path = exp / "meta.json"
    if not meta_path.exists():
        atomic_write_text(meta_path, json.dumps({"display_name": display_name, "staked": None}, sort_keys=True))
    base_cfg = exp / "base_config.yaml"
    if not base_cfg.exists():
        import yaml

        atomic_write_text(base_cfg, yaml.safe_dump(base_config, sort_keys=False))
    readme = exp / "README.md"
    if not readme.exists():
        atomic_write_text(readme, f"# {slug}\n\n<!-- human record: what was done, decisions, results -->\n")
    return exp


def ensure_family(slug: str, *, display_name: str, base_config: dict[str, Any]) -> Path:
    return _write_scaffold(slug, display_name=display_name, base_config=base_config)


def record_run(slug: str, run_id: str, payload: dict[str, Any]) -> Path:
    """Persist run.json (atomic) — creating the family scaffold on first run."""
    _validate_run_id(run_id)
    display_name = str((payload.get("manifest") or {}).get("display_name") or slug)
    base_config = (payload.get("manifest") or {}).get("config") or {}
    _write_scaffold(slug, display_name=display_name, base_config=base_config)
    run_dir = paths.run_dir(slug, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = paths.run_json_path(slug, run_id)
    atomic_write_text(target, json.dumps(payload, sort_keys=True))
    return target


def read_run(slug: str, run_id: str) -> dict[str, Any]:
    path = paths.run_json_path(slug, run_id)
    if not path.is_file():
        raise FileNotFoundError(f"no run record at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def stage_export(slug: str, scope: str, run_id: str) -> Path:
    _validate_run_id(run_id)
    parent = paths.export_dir(slug, scope, run_id).parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".tmp-{run_id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def discard_staged_export(slug: str, scope: str, run_id: str) -> None:
    _validate_run_id(run_id)
    staging = paths.export_dir(slug, scope, run_id).parent / f".tmp-{run_id}"
    if staging.exists():
        shutil.rmtree(staging)


def publish_staged_export(slug: str, scope: str, run_id: str) -> Path:
    """Atomically rename the staged dir into the final slot. Raises ValueError
    when the slot already exists (exports are immutable — spec §6)."""
    _validate_run_id(run_id)
    final = paths.export_dir(slug, scope, run_id)
    staging = final.parent / f".tmp-{run_id}"
    if not staging.is_dir():
        raise FileNotFoundError(f"no staged export at {staging}")
    if final.exists():
        discard_staged_export(slug, scope, run_id)
        raise ValueError(f"export slot {final} already exists; exports are immutable")
    os.replace(staging, final)  # single directory rename; target is absent
    return final
