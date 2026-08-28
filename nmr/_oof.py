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
    "feature_list_fingerprint",
    "splitter_geometry_fingerprint",
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


def feature_list_fingerprint(feature_cols: Sequence[str]) -> str:
    """SHA-256 over the sorted feature names — a canonical feature-schema
    fingerprint recorded in checkpoint manifests (2026-08-26 review, SECONDARY
    1): a checkpoint tree written for one feature list must refuse resume for
    a different one (order-independent — sorting makes the fingerprint
    canonical)."""
    canonical = ",".join(sorted(feature_cols))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def splitter_geometry_fingerprint(
    splitter: PurgedEraSplitter, eras: Sequence[str]
) -> str:
    """SHA-256 over the splitter's canonical fold boundaries.

    ``splitter.split(eras)`` yields the exact per-fold train/val era tuples
    (the splitter is the sole fold authority — ``nmr.splitter.PurgedEraSplitter``
    exposes ``split`` and ``purge_eras``); a canonical string of those
    boundaries hashed binds a checkpoint to the fold geometry that produced it.
    A resume against different fold boundaries refuses (no silent reuse of
    folds from another geometry).
    """
    folds = splitter.split(list(eras))
    canonical = "|".join(
        f"{fold.index}:{','.join(fold.train_eras)};{','.join(fold.val_eras)}"
        for fold in folds
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def checkpoint_manifest(
    device: str,
    *,
    data_fingerprint: str | None = None,
    environment: str | None = None,
    target_col: str | None = None,
    feature_fingerprint: str | None = None,
    splitter_fingerprint: str | None = None,
) -> dict[str, str]:
    """Identity manifest for a checkpoint root: code sha256 + fit device, plus
    the rebuild-identity terms (spec §3.1) — ``data_fingerprint`` and the
    portable ``environment`` — and the fit-identity terms (2026-08-26 review,
    SECONDARY 1) — ``target_col``, the canonical feature-list fingerprint, and
    the splitter-geometry fingerprint — when the caller knows them (the runner
    always does; direct unit callers may omit them). Fields are present only
    when provided, so callers without the runner's identity context keep the
    legacy code+device form.
    """
    manifest: dict[str, str] = {"code_sha256": fitting_code_sha256(), "device": device}
    if data_fingerprint is not None:
        manifest["data_fingerprint"] = data_fingerprint
    if environment is not None:
        manifest["environment"] = environment
    if target_col is not None:
        manifest["target_col"] = target_col
    if feature_fingerprint is not None:
        manifest["feature_fingerprint"] = feature_fingerprint
    if splitter_fingerprint is not None:
        manifest["splitter_fingerprint"] = splitter_fingerprint
    return manifest


def verify_checkpoint_manifest(
    manifest_path: Path,
    current_device: str | None,
    *,
    data_fingerprint: str | None = None,
    environment: str | None = None,
    target_col: str | None = None,
    feature_fingerprint: str | None = None,
    splitter_fingerprint: str | None = None,
    checkpoint_kind: str = "oof_checkpoints",
) -> None:
    """Verify an existing manifest: code exact-compare, device three-way,
    rebuild-identity (data/environment) exact-compare, fit-identity
    (target/features/splitter) exact-compare.

    Code identity is exact-compared always. The device guard compares exactly
    when ``current_device`` is known (post-fit reuse); when it is None (a
    fresh orchestrator, device unknown pre-fit) the stored device must be a
    real fit device (``_KNOWN_RESOLVED_DEVICES``) — anything else is rejected
    loudly, never accepted vacuously. The rebuild-identity guards (spec §3.1)
    exact-compare ``data_fingerprint`` and ``environment`` when the current
    values are provided (the runner always provides them); a stored manifest
    missing a guarded field is treated as a mismatch — refuse loudly, never
    resume a checkpoint whose data snapshot or dependency environment drifted.
    The fit-identity guards (2026-08-26 review, SECONDARY 1) exact-compare
    ``target_col``, ``feature_fingerprint``, and ``splitter_fingerprint`` when
    provided — a checkpoint dir copied from another target (or fitted on
    another feature list / fold geometry) refuses resume instead of silently
    reusing the wrong target's folds. Mismatches raise ``ValueError`` with
    delete-to-refit guidance. ``checkpoint_kind`` names the checkpoint stage
    in the error text (``oof_checkpoints``, ``deploy_checkpoints``,
    ``validation_checkpoints``).
    """
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stored.get("code_sha256") != fitting_code_sha256():
        raise ValueError(
            f"{checkpoint_kind} code_sha256 mismatch: fitting code changed "
            f"since the checkpoints were written ({manifest_path}). "
            f"Delete the {checkpoint_kind} directory to force a full refit."
        )
    if data_fingerprint is not None and stored.get("data_fingerprint") != data_fingerprint:
        raise ValueError(
            f"{checkpoint_kind} data_fingerprint mismatch: the data snapshot "
            f"changed since the checkpoints were written ({manifest_path}). "
            f"Delete the {checkpoint_kind} directory to force a full refit."
        )
    if environment is not None and stored.get("environment") != environment:
        raise ValueError(
            f"{checkpoint_kind} environment mismatch: the dependency "
            f"environment changed since the checkpoints were written "
            f"({manifest_path}). Delete the {checkpoint_kind} directory to "
            f"force a full refit."
        )
    if target_col is not None and stored.get("target_col") != target_col:
        raise ValueError(
            f"{checkpoint_kind} target mismatch: checkpoints were fitted for "
            f"target {stored.get('target_col')!r}, current target is "
            f"{target_col!r} ({manifest_path}). Delete the {checkpoint_kind} "
            f"directory to force a full refit."
        )
    if (
        feature_fingerprint is not None
        and stored.get("feature_fingerprint") != feature_fingerprint
    ):
        raise ValueError(
            f"{checkpoint_kind} feature-list mismatch: the feature schema "
            f"changed since the checkpoints were written ({manifest_path}). "
            f"Delete the {checkpoint_kind} directory to force a full refit."
        )
    if (
        splitter_fingerprint is not None
        and stored.get("splitter_fingerprint") != splitter_fingerprint
    ):
        raise ValueError(
            f"{checkpoint_kind} splitter-geometry mismatch: the fold "
            f"boundaries changed since the checkpoints were written "
            f"({manifest_path}). Delete the {checkpoint_kind} directory to "
            f"force a full refit."
        )
    stored_device = stored.get("device")
    if current_device is not None:
        if stored_device != str(current_device):
            raise ValueError(
                f"{checkpoint_kind} device mismatch: checkpoints were "
                f"fitted on device {stored_device!r}, current device "
                f"is {str(current_device)!r}. Delete the "
                f"{checkpoint_kind} directory to force a full refit."
            )
    elif stored_device not in _KNOWN_RESOLVED_DEVICES:
        raise ValueError(
            f"{checkpoint_kind} device mismatch: manifest records "
            f"unknown device {stored_device!r} ({manifest_path}). "
            f"Delete the {checkpoint_kind} directory to force a full refit."
        )


def ensure_no_torn_tree(
    manifest_path: Path,
    *,
    checkpoint_kind: str = "oof_checkpoints",
    part_glob: str = "fold_*.parquet",
) -> None:
    """Reject a checkpoint root with parts but no manifest (torn tree).

    ``part_glob`` is the checkpoint unit's file pattern (``fold_*.parquet``
    for OOF folds, ``*.pkl`` for deploy models); ``checkpoint_kind`` names the
    checkpoint stage in the error text.
    """
    existing_parts = any(
        manifest_path.parent.rglob(part_glob)
    ) if manifest_path.parent.exists() else False
    if existing_parts:
        raise ValueError(
            f"{checkpoint_kind} tree has parts but no manifest.json "
            f"({manifest_path}) — inconsistent state. Delete the "
            f"{checkpoint_kind} directory to force a full refit."
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
    data_fingerprint: str | None = None,
    environment: str | None = None,
) -> pl.DataFrame:
    """Train per-target cross-validated OOF and stack the predictions.

    Each target's OOF is renamed ``pred_<target>`` and inner-joined on
    ``(id, era)``, so only rows present in every target's OOF survive. The
    splitter is the sole fold authority (era-purged, leakage-safe); no
    random row-level CV is ever constructed here.

    With ``checkpoint_dir`` set, each target routes to the checkpoint-aware
    OOF-only path (``train_oof_with_checkpoints``): existing fold parquets
    are loaded instead of refit, missing folds are fitted and persisted
    atomically. ``data_fingerprint`` and ``environment`` are the rebuild-
    identity terms (spec §3.1) recorded in the checkpoint manifest — the
    runner passes them; callers without them keep the legacy code+device
    manifest. Without it, the legacy ``train_cross_validation`` path runs
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
                data_fingerprint=data_fingerprint,
                environment=environment,
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
