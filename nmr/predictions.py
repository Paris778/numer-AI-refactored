"""Prediction artifact contract: validated frames plus provenance.

Training (or a foreign producer) emits a ``PredictionSet``. Composition and
evaluation consume it. This module does not import ``nmr.models``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nmr._transforms import tie_kept_rank
from nmr.ensemble import Ensembler
from nmr.risk import NeutralizationEngine

__all__ = [
    "PREDICTION_STAGES",
    "PredictionProvenance",
    "PredictionSet",
    "compose_predictions",
    "evaluate_prediction_set",
    "prediction_set_from_frame",
    "read_prediction_set",
]

CAPITAL_STAGES: tuple[str, ...] = ("validation", "submission")

PREDICTION_STAGES: tuple[str, ...] = (
    "raw",
    "blended",
    "neutralized",
    "validation",
    "submission",
)

_REQUIRED_FRAME_COLS = ("era", "id", "prediction")


@dataclass(frozen=True)
class PredictionProvenance:
    """Identity of a prediction frame — what it is, not how long it took.

    Wall-clock and absolute paths are forbidden here: this block may enter
    hashes. ``None`` fields stay present so a reader can see what was unknown.
    """

    stage: str
    trained_targets: tuple[str, ...]
    ensemble_target: str | None = None
    scoring_target: str | None = None
    training_horizon: str | None = None
    scoring_horizon: str | None = None
    era_partition: str | None = None
    data_fingerprint: str | None = None
    feature_fingerprint: str | None = None
    fit_role: str | None = None
    device: str | None = None
    source_run_id: str | None = None
    selection_bias: bool = False
    split_estimand: bool = False

    def __post_init__(self) -> None:
        if self.stage not in PREDICTION_STAGES:
            raise ValueError(
                f"stage must be one of {PREDICTION_STAGES}, got {self.stage!r}"
            )


@dataclass(frozen=True)
class PredictionSet:
    frame: pl.DataFrame
    provenance: PredictionProvenance


def prediction_set_from_frame(
    frame: pl.DataFrame,
    provenance: PredictionProvenance,
    *,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> PredictionSet:
    """Validate, normalize column names, sort, and wrap ``frame``."""
    if not isinstance(frame, pl.DataFrame):
        raise ValueError("frame must be a polars DataFrame")
    missing = [
        name
        for name, col in (
            ("era", era_col),
            ("id", id_col),
            ("prediction", pred_col),
        )
        if col not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.height < 1:
        raise ValueError("prediction frame must contain at least one row")

    normalized = frame.select(
        [
            pl.col(era_col).alias("era"),
            pl.col(id_col).alias("id"),
            pl.col(pred_col).alias("prediction"),
        ]
    ).drop_nulls(["era", "id"])
    if normalized.height != frame.height:
        raise ValueError("era and id must not contain nulls")
    if normalized.select(["era", "id"]).n_unique() != normalized.height:
        raise ValueError("prediction frame must have unique (era, id) keys")

    pred_values = normalized.get_column("prediction").cast(pl.Float64).to_numpy()
    if not np.all(np.isfinite(pred_values)):
        raise ValueError("prediction values must be finite")
    if provenance.stage == "submission":
        if not np.all((pred_values > 0.0) & (pred_values < 1.0)):
            raise ValueError("submission predictions must be in (0, 1)")

    ordered = normalized.sort(["era", "id"]).select(list(_REQUIRED_FRAME_COLS))
    return PredictionSet(frame=ordered, provenance=provenance)


def read_prediction_set(path: Path, provenance: PredictionProvenance) -> PredictionSet:
    """Load a three-column prediction parquet and attach caller provenance."""
    return prediction_set_from_frame(pl.read_parquet(path), provenance)


def compose_predictions(
    frame: pl.DataFrame,
    *,
    pred_cols: Sequence[str],
    provenance: PredictionProvenance,
    weights: Sequence[float] | None = None,
    era_col: str = "era",
    id_col: str = "id",
    feature_cols: Sequence[str] = (),
    neutralization_proportion: float = 0.0,
    neutralization_cache_dir: Path | None = None,
    as_submission: bool = False,
) -> PredictionSet:
    """Rank-blend components, optionally neutralize, optionally submission-rank.

    Wraps ``Ensembler.blend`` and ``NeutralizationEngine.neutralize``. Does not
    change their math. The returned stage is ``submission`` if
    ``as_submission``, else ``neutralized`` when ``neutralization_proportion > 0``,
    else ``blended``.
    """
    pred_list = list(pred_cols)
    if not pred_list:
        raise ValueError("pred_cols must contain at least one prediction column")
    if era_col not in frame.columns or id_col not in frame.columns:
        raise ValueError(f"frame must contain {era_col!r} and {id_col!r}")

    blended = Ensembler().blend(
        frame,
        pred_cols=pred_list,
        weights=weights,
        era_col=era_col,
        out_col="prediction",
    )
    stage = "blended"
    work = blended
    if neutralization_proportion > 0.0:
        feature_list = list(feature_cols)
        if not feature_list:
            raise ValueError(
                "feature_cols must contain at least one feature when "
                "neutralization_proportion > 0"
            )
        work = NeutralizationEngine(
            cache_dir=neutralization_cache_dir,
            max_cache_bytes=0,
        ).neutralize(
            work,
            pred_col="prediction",
            feature_cols=feature_list,
            era_col=era_col,
            proportion=neutralization_proportion,
        )
        stage = "neutralized"
    if as_submission:
        work = _per_era_submission_rank(work, era_col=era_col, pred_col="prediction")
        stage = "submission"
    return prediction_set_from_frame(
        work,
        replace(provenance, stage=stage),
        era_col=era_col,
        id_col=id_col,
        pred_col="prediction",
    )


def _per_era_submission_rank(
    frame: pl.DataFrame, *, era_col: str, pred_col: str
) -> pl.DataFrame:
    indexed = frame.with_row_index("__row_idx")
    parts: list[pl.DataFrame] = []
    for era_df in indexed.partition_by(era_col, as_dict=False, maintain_order=True):
        ranked = tie_kept_rank(era_df.get_column(pred_col).cast(pl.Float64).to_numpy())
        parts.append(era_df.with_columns(pl.Series(pred_col, ranked)))
    return pl.concat(parts, how="vertical").sort("__row_idx").drop("__row_idx")


def evaluate_prediction_set(
    prediction_set: PredictionSet,
    *,
    meta_model: pl.DataFrame,
    benchmarks: pl.DataFrame | None,
    features: pl.DataFrame,
    targets: pl.DataFrame,
    allow_research_stage: bool = False,
    **kwargs: Any,
):
    """Score a ``PredictionSet`` through ``evaluate_model``.

    Capital stages are ``validation`` and ``submission``. Other stages raise
    unless ``allow_research_stage=True``. Does not import ``nmr.models``.
    """
    from nmr.scorecard import evaluate_model

    stage = prediction_set.provenance.stage
    if stage not in CAPITAL_STAGES and not allow_research_stage:
        raise ValueError(
            f"evaluate_prediction_set refuses stage={stage!r}; "
            f"capital stages are {CAPITAL_STAGES} "
            "(pass allow_research_stage=True for research frames)"
        )
    return evaluate_model(
        prediction_set.frame,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        **kwargs,
    )
