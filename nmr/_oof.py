"""Shared multi-target OOF training — single source for runner + research.

The leakage-critical OOF construction path lives here exactly once: the runner
(deploy/evaluation) and research (HPO/held-out) both delegate to it. The
duplicated copies this replaces (audit SEV-2 #5) could silently drift — the
runner tunes models the researcher evaluates, so the OOF path must be one
implementation.

This module also owns the shared checkpoint identity/atomic-write helpers
extracted from the OOF fold checkpoint path (spec 2026-08-23-checkpoint-
coverage-extension §2.5): they are the single implementation for the OOF,
deploy, and validation checkpoint stages.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from nmr._atomicio import atomic_write_bytes
from nmr.models import ModelOrchestrator
from nmr.splitter import PurgedEraSplitter

__all__ = [
    "train_multi_target_oof",
    "fitting_code_sha256",
    "checkpoint_manifest",
    "verify_checkpoint_manifest",
    "ensure_no_torn_tree",
    "write_frame_atomic",
    "write_bytes_atomic",
]

_CODE_IDENTITY_FILES = ("nmr/models.py", "nmr/splitter.py", "nmr/runner.py")

# The only devices ModelOrchestrator._fit_model can resolve (models.py):
# checkpoint manifests must record one of these — anything else is rejected
# loudly on resume, never accepted vacuously.
_KNOWN_RESOLVED_DEVICES = ("cpu", "gpu")


def fitting_code_sha256() -> str:
    """SHA-256 over the fitting-code source bytes (models + splitter + runner).

    The modules that define fold geometry, fit behavior, and the staged
    pipeline. This is the code identity recorded in checkpoint manifests:
    run_id binds config and data, never code, so checkpoints must not
    silently survive a fitting-code change (spec 2026-08-20-oof-checkpoint-
    resume §2.5; ``nmr/runner.py`` added by spec 2026-08-23-checkpoint-
    coverage-extension §2.4).
    """
    digest = hashlib.sha256()
    for relative in _CODE_IDENTITY_FILES:
        path = Path(__file__).resolve().parents[1] / relative
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_manifest(device: str) -> dict[str, str]:
    """Identity manifest for a checkpoint root: code sha256 + fit device."""
    return {"code_sha256": fitting_code_sha256(), "device": device}


def verify_checkpoint_manifest(
    manifest_path: Path, current_device: str | None
) -> None:
    """Verify an existing manifest: code exact-compare, device three-way.

    Code identity is exact-compared always. The device guard compares exactly
    when ``current_device`` is known (post-fit reuse); when it is None (a
    fresh orchestrator, device unknown pre-fit) the stored device must be a
    real fit device (``_KNOWN_RESOLVED_DEVICES``) — anything else is rejected
    loudly, never accepted vacuously. Mismatches raise ``ValueError`` with
    delete-to-refit guidance.
    """
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stored.get("code_sha256") != fitting_code_sha256():
        raise ValueError(
            "OOF checkpoint code_sha256 mismatch: fitting code changed "
            f"since the checkpoints were written ({manifest_path}). "
            "Delete the oof_checkpoints directory to force a full refit."
        )
    stored_device = stored.get("device")
    if current_device is not None:
        if stored_device != str(current_device):
            raise ValueError(
                "OOF checkpoint device mismatch: checkpoints were "
                f"fitted on device {stored_device!r}, current device "
                f"is {str(current_device)!r}. Delete the "
                "oof_checkpoints directory to force a full refit."
            )
    elif stored_device not in _KNOWN_RESOLVED_DEVICES:
        raise ValueError(
            "OOF checkpoint device mismatch: manifest records "
            f"unknown device {stored_device!r} ({manifest_path}). "
            "Delete the oof_checkpoints directory to force a full refit."
        )


def ensure_no_torn_tree(manifest_path: Path) -> None:
    """Reject a checkpoint root with fold parts but no manifest (torn tree)."""
    existing_parts = any(
        manifest_path.parent.rglob("fold_*.parquet")
    ) if manifest_path.parent.exists() else False
    if existing_parts:
        raise ValueError(
            "OOF checkpoint tree has fold parts but no manifest.json "
            f"({manifest_path}) — inconsistent state. Delete the "
            "oof_checkpoints directory to force a full refit."
        )


def write_frame_atomic(frame: pl.DataFrame, path: Path) -> None:
    """Serialize ``frame`` to parquet bytes and write them atomically.

    The only checkpoint frame writer: bytes go through
    ``nmr/_atomicio.atomic_write_bytes`` (temp file + fsync + os.replace) —
    never ``write_parquet`` directly to the final path.
    """
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    atomic_write_bytes(path, buffer.getvalue())


def write_bytes_atomic(data: bytes, path: Path) -> None:
    """Write ``data`` to ``path`` atomically (temp + fsync + os.replace)."""
    atomic_write_bytes(path, data)


def train_multi_target_oof(
    modeler: ModelOrchestrator,
    df: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    splitter: PurgedEraSplitter,
    targets: Sequence[str],
    checkpoint_dir: Path | None = None,
) -> pl.DataFrame:
    """Train per-target cross-validated OOF and stack the predictions.

    Each target's OOF is renamed ``pred_<target>`` and inner-joined on
    ``(id, era)``, so only rows present in every target's OOF survive. The
    splitter is the sole fold authority (era-purged, leakage-safe); no
    random row-level CV is ever constructed here.

    With ``checkpoint_dir`` set, each target routes to the checkpoint-aware
    OOF-only path (``train_oof_with_checkpoints``): existing fold parquets
    are loaded instead of refit, missing folds are fitted and persisted
    atomically. Without it, the legacy ``train_cross_validation`` path runs
    unchanged (spec 2026-08-20-oof-checkpoint-resume).
    """
    stacked: pl.DataFrame | None = None
    for target in targets:
        if checkpoint_dir is not None:
            part = modeler.train_oof_with_checkpoints(
                df,
                feature_cols=feature_cols,
                target_col=target,
                splitter=splitter,
                era_col="era",
                checkpoint_dir=checkpoint_dir,
            )
        else:
            result = modeler.train_cross_validation(
                df,
                feature_cols=feature_cols,
                target_col=target,
                splitter=splitter,
                era_col="era",
            )
            part = result.oof
        part = part.rename({"prediction": f"pred_{target}"})
        if stacked is None:
            stacked = part
        else:
            stacked = stacked.join(part, on=["id", "era"], how="inner")
    assert stacked is not None
    return stacked
