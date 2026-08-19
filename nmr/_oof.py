"""Shared multi-target OOF training — single source for runner + research.

The leakage-critical OOF construction path lives here exactly once: the runner
(deploy/evaluation) and research (HPO/held-out) both delegate to it. The
duplicated copies this replaces (audit SEV-2 #5) could silently drift — the
runner tunes models the researcher evaluates, so the OOF path must be one
implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from nmr.models import ModelOrchestrator
from nmr.splitter import PurgedEraSplitter

__all__ = ["train_multi_target_oof"]


def train_multi_target_oof(
    modeler: ModelOrchestrator,
    df: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    splitter: PurgedEraSplitter,
    targets: Sequence[str],
) -> pl.DataFrame:
    """Train per-target cross-validated OOF and stack the predictions.

    Each target's OOF is renamed ``pred_<target>`` and inner-joined on
    ``(id, era)``, so only rows present in every target's OOF survive. The
    splitter is the sole fold authority (era-purged, leakage-safe); no
    random row-level CV is ever constructed here.
    """
    stacked: pl.DataFrame | None = None
    for target in targets:
        result = modeler.train_cross_validation(
            df,
            feature_cols=feature_cols,
            target_col=target,
            splitter=splitter,
            era_col="era",
        )
        part = result.oof.rename({"prediction": f"pred_{target}"})
        if stacked is None:
            stacked = part
        else:
            stacked = stacked.join(part, on=["id", "era"], how="inner")
    assert stacked is not None
    return stacked
