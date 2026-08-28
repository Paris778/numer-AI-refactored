"""Run/export persistence and atomic publication for the experiment layout.

``record_run`` creates the family scaffold atomically with the first run.json
(spec §2 family-creation rule). Export slots are staged under ``.tmp-<run_id>``
and published by a single directory rename; discovery ignores ``.tmp-`` names.
``record_run_result`` is the script-facing recorder: it persists the run's
parquet outputs (``oof.parquet`` + ``validation_preds.parquet`` when the
validation scorecard ran) alongside the run.json built by ``record_run`` —
the run-immutability preflight runs BEFORE any artifact write, so a rejected
re-record never mutates the heavy parquets.
"""
from __future__ import annotations

import dataclasses
import io
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from nmr import paths
from nmr._atomicio import atomic_write_bytes, atomic_write_text

if TYPE_CHECKING:
    from nmr.runner import RunResult

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


def _check_run_immutable(slug: str, run_id: str, canonical: str, target: Path) -> None:
    """Raise when run.json exists with a payload different from ``canonical``.

    The single immutability predicate for run records (spec §2): re-recording
    an existing run.json with a DIFFERENT payload raises ``ValueError``; the
    byte-identical payload stays idempotent. Shared by ``record_run`` and the
    ``record_run_result`` preflight so a rejected re-record raises before any
    artifact write.
    """
    if target.is_file() and target.read_text(encoding="utf-8") != canonical:
        raise ValueError(
            f"run {run_id!r} already recorded with a different payload "
            f"({target}); runs are immutable once recorded"
        )


def record_run(slug: str, run_id: str, payload: dict[str, Any]) -> Path:
    """Persist run.json (atomic) — creating the family scaffold on first run.

    Runs are immutable once recorded (spec §2): re-recording an existing
    run.json with a DIFFERENT payload raises ``ValueError``; re-recording the
    byte-identical payload stays idempotent (the canonical JSON is simply
    rewritten). The immutability preflight runs BEFORE the scaffold is
    written (2026-08-26 review, SECONDARY 7): a rejected re-record must leave
    the tree byte-identical — the rejected payload's display_name/base_config
    must never leak into a fresh scaffold.
    """
    _validate_run_id(run_id)
    display_name = str((payload.get("manifest") or {}).get("display_name") or slug)
    base_config = (payload.get("manifest") or {}).get("config") or {}
    target = paths.run_json_path(slug, run_id)
    canonical = json.dumps(payload, sort_keys=True)
    _check_run_immutable(slug, run_id, canonical, target)
    _write_scaffold(slug, display_name=display_name, base_config=base_config)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, canonical)
    return target


def read_run(slug: str, run_id: str) -> dict[str, Any]:
    path = paths.run_json_path(slug, run_id)
    if not path.is_file():
        raise FileNotFoundError(f"no run record at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet_atomic(path: Path, frame: pl.DataFrame) -> None:
    """Serialize ``frame`` to parquet bytes and write them atomically.

    The run-artifact frame writer: bytes go through
    ``nmr._atomicio.atomic_write_bytes`` (temp file + fsync + os.replace) —
    the same atomic-bytes contract as the checkpoint frame writer
    (``nmr._oof.write_frame_atomic``), never ``write_parquet`` directly to
    the final path.
    """
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    atomic_write_bytes(path, buffer.getvalue())


def _result_payload(result: RunResult) -> dict[str, Any]:
    """Serialize a runner result into the persisted run.json record.

    The scorecard block drops timing/instrumentation columns — they capture
    wall-clock durations that differ across processes and are excluded from
    canonical serialization (AGENTS.md timing-hazard).
    """
    scorecard_block = None
    if result.scorecard is not None:
        row = result.scorecard.to_frame().to_dicts()[0]
        scorecard_block = {
            key: value
            for key, value in row.items()
            if not key.startswith(("timing_", "quality_metric"))
        }
    return {
        "run_id": result.run_id,
        "metrics": dataclasses.asdict(result.metrics),
        "manifest": result.manifest,
        "scorecard": scorecard_block,
        "oof_path": "oof.parquet",
        "artifact_path": str(result.artifact.path) if result.artifact else None,
        "artifact_manifest": result.artifact.manifest if result.artifact else None,
    }


def record_run_result(slug: str, result: RunResult) -> Path:
    """Record a runner result under ``experiments/<slug>/runs/<run_id>/``.

    Persists ``oof.parquet`` and (when the validation scorecard ran)
    ``validation_preds.parquet`` atomically, then writes ``run.json`` via
    :func:`record_run` — the parquets land before the record marker so a
    partial failure never leaves a run.json without its artifacts. The
    run-immutability check runs FIRST (:func:`_check_run_immutable`): a
    re-record whose payload differs from the recorded run.json raises before
    any artifact write, so a rejected re-record never mutates the heavy
    parquets (the byte-identical same-payload re-record stays idempotent).
    Returns the run directory (the script-facing analogue of the retired
    ``registry.record``). ``slug`` is the family slug (``config.run.name``).
    """
    paths.validate_slug(slug)
    _validate_run_id(result.run_id)
    payload = _result_payload(result)
    _check_run_immutable(
        slug, result.run_id, json.dumps(payload, sort_keys=True),
        paths.run_json_path(slug, result.run_id),
    )
    run_dir = paths.run_dir(slug, result.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(run_dir / "oof.parquet", result.oof)
    if result.validation_predictions is not None:
        _write_parquet_atomic(
            run_dir / "validation_preds.parquet", result.validation_predictions
        )
    record_run(slug, result.run_id, payload)
    return run_dir


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
