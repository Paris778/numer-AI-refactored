"""Shared multi-target OOF training — single source for runner + research.

The leakage-critical OOF construction path lives here exactly once: the runner
(deploy/evaluation) and research (HPO/held-out) both delegate to it. The
duplicated copies this replaces (audit SEV-2 #5) could silently drift — the
runner tunes models the researcher evaluates, so the OOF path must be one
implementation.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from nmr._atomicio import atomic_write_bytes
from nmr.models import ModelOrchestrator
from nmr.splitter import PurgedEraSplitter

__all__ = ["train_multi_target_oof"]

_CODE_IDENTITY_FILES = ("nmr/models.py", "nmr/splitter.py")

# The only devices ModelOrchestrator._fit_model can resolve (models.py):
# checkpoint manifests must record one of these — anything else is rejected
# loudly on resume, never accepted vacuously.
_KNOWN_RESOLVED_DEVICES = ("cpu", "gpu")


def _fitting_code_sha256() -> str:
    """SHA-256 over the fitting-code source bytes (models + splitter).

    The two modules that define fold geometry and fit behavior. This is the
    code identity recorded in OOF checkpoint manifests: run_id binds config
    and data, never code, so checkpoints must not silently survive a
    fitting-code change (spec 2026-08-20-oof-checkpoint-resume §2.5).
    """
    digest = hashlib.sha256()
    for relative in _CODE_IDENTITY_FILES:
        path = Path(__file__).resolve().parents[1] / relative
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_frame_atomic(frame: pl.DataFrame, path: Path) -> None:
    """Serialize ``frame`` to parquet bytes and write them atomically.

    The only OOF checkpoint writer: bytes go through
    ``nmr/_atomicio.atomic_write_bytes`` (temp file + fsync + os.replace) —
    never ``write_parquet`` directly to the final path.
    """
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    atomic_write_bytes(path, buffer.getvalue())


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
