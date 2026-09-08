"""Prediction artifact contract: validated frames plus provenance.

Training (or a foreign producer) emits a ``PredictionSet``. Composition and
evaluation consume it. This module does not import ``nmr.models``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nmr._transforms import tie_kept_rank
from nmr.config import VALID_HORIZONS, VALID_MODEL_DEVICES
from nmr.ensemble import Ensembler
from nmr.payout import resolve_payout_policy
from nmr.risk import NeutralizationEngine

__all__ = [
    "PREDICTION_STAGES",
    "ERA_PARTITIONS",
    "FIT_ROLES",
    "CAPITAL_EVIDENCE_VERSION",
    "CapitalContext",
    "PredictionProvenance",
    "PredictionSet",
    "ResearchEvaluation",
    "compose_predictions",
    "evaluate_prediction_set",
    "prediction_set_from_frame",
    "read_prediction_set",
    "validation_key_fingerprint",
]

PREDICTION_STAGES: tuple[str, ...] = (
    "raw",
    "blended",
    "neutralized",
    "validation",
    "submission",
)
ERA_PARTITIONS: tuple[str, ...] = ("oof", "validation", "live", "held_out")
FIT_ROLES: tuple[str, ...] = ("cv_oof", "full_history", "foreign")
_RESEARCH_PARTITIONS: frozenset[str] = frozenset({"oof", "held_out"})
_REQUIRED_FRAME_COLS = ("era", "id", "prediction")
_DEFAULT_MAIN_TARGET = "target"
_DEFAULT_HORIZON = "20D"
# Schema version of the persisted capital-evidence block (run.json manifest).
# Promotion refuses any other version — bump deliberately, never silently.
CAPITAL_EVIDENCE_VERSION = 1
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def validation_key_fingerprint(
    frame: pl.DataFrame, *, era_col: str = "era", id_col: str = "id"
) -> str:
    """SHA-256 over the canonical ``(era, id)`` key universe of a frame.

    Order-independent and extra-column-independent: two frames carry the same
    fingerprint iff they cover exactly the same keys. This is the identity
    term that lets capital evaluation refuse sparse or extra prediction rows.
    """
    if era_col not in frame.columns or id_col not in frame.columns:
        raise ValueError(
            f"frame must contain {era_col!r} and {id_col!r} for key fingerprinting"
        )
    keys = sorted(
        (str(era), str(key_id))
        for era, key_id in frame.select([era_col, id_col]).iter_rows()
    )
    return hashlib.sha256(
        json.dumps(keys, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _assert_capital_frame_key_universe(
    name: str, frame: pl.DataFrame, context: CapitalContext
) -> None:
    """Require one auxiliary evaluation frame to equal the trusted universe."""
    if not isinstance(frame, pl.DataFrame):
        raise ValueError(f"capital evaluation {name} must be a polars DataFrame")
    if "era" not in frame.columns or "id" not in frame.columns:
        raise ValueError(f"capital evaluation {name} must contain exact (era, id) keys")
    if frame.select(["era", "id"]).n_unique() != frame.height:
        raise ValueError(f"capital evaluation {name} contains duplicate (era, id) keys")
    if (
        frame.height != context.validation_row_count
        or validation_key_fingerprint(frame) != context.validation_key_fingerprint
    ):
        raise ValueError(
            f"capital evaluation {name} keys do not match the authoritative "
            "validation key universe"
        )


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
    validation_window: tuple[str, ...] | None = None
    payout_policy_id: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in PREDICTION_STAGES:
            raise ValueError(
                f"stage must be one of {PREDICTION_STAGES}, got {self.stage!r}"
            )
        targets = self.trained_targets
        if not isinstance(targets, tuple):
            object.__setattr__(self, "trained_targets", tuple(targets))
            targets = self.trained_targets
        if not targets or any(not isinstance(t, str) or not t for t in targets):
            raise ValueError("trained_targets must be a non-empty tuple of strings")
        if self.era_partition is not None and self.era_partition not in ERA_PARTITIONS:
            raise ValueError(
                f"era_partition must be one of {ERA_PARTITIONS}, "
                f"got {self.era_partition!r}"
            )
        if self.fit_role is not None and self.fit_role not in FIT_ROLES:
            raise ValueError(
                f"fit_role must be one of {FIT_ROLES}, got {self.fit_role!r}"
            )
        if self.device is not None and self.device not in VALID_MODEL_DEVICES:
            raise ValueError(
                f"device must be one of {VALID_MODEL_DEVICES}, got {self.device!r}"
            )
        for name, value in (
            ("training_horizon", self.training_horizon),
            ("scoring_horizon", self.scoring_horizon),
        ):
            if value is not None and value not in VALID_HORIZONS:
                raise ValueError(
                    f"{name} must be one of {VALID_HORIZONS}, got {value!r}"
                )
        window = self.validation_window
        if window is not None:
            if not isinstance(window, tuple):
                object.__setattr__(self, "validation_window", tuple(window))
                window = self.validation_window
            if any(not isinstance(era, str) or not era for era in window):
                raise ValueError("validation_window must be a tuple of era labels")
        if self.payout_policy_id is not None and not self.payout_policy_id:
            raise ValueError("payout_policy_id must be a non-empty string or None")


@dataclass(frozen=True)
class PredictionSet:
    frame: pl.DataFrame
    provenance: PredictionProvenance


@dataclass(frozen=True)
class ResearchEvaluation:
    """Scorecard produced from a research-stage frame. Never capital evidence."""

    scorecard: Any
    is_capital: bool = False

    def __post_init__(self) -> None:
        if self.is_capital:
            raise ValueError("ResearchEvaluation cannot be capital evidence")


@dataclass(frozen=True)
class CapitalContext:
    """Authoritative capital identity for one validation evaluation.

    Produced by a trusted validation loader (the runner derives every field
    from the loaded, purged ``validation.parquet`` and the run config — never
    from the prediction artifact). A caller can still construct the dataclass,
    but the predicate verifies it against BOTH the prediction frame and the
    persisted evidence chain; promotion additionally re-verifies the key
    universe against the data on disk. Self-consistency with a prediction
    frame is necessary but never sufficient: the context is bound to the data
    snapshot (``data_fingerprint``) and the feature schema
    (``feature_fingerprint``), which promotion re-checks against the current
    data files.

    Threat-model boundary: this binding is anti-DRIFT, not anti-forgery. An
    operator with filesystem write access to ``experiments/`` and read access
    to the data can compute every fingerprint and fabricate a consistent
    evidence chain; nothing in this module can detect that. The chain detects
    stale, partial, or mislabeled evidence and hand-edited inconsistencies —
    it does not authenticate the producer.
    """

    validation_window: tuple[str, ...]
    scoring_target: str
    scoring_horizon: str
    payout_policy_id: str
    data_fingerprint: str
    feature_fingerprint: str
    validation_key_fingerprint: str
    validation_row_count: int
    trained_targets: tuple[str, ...]
    ensemble_target: str
    training_horizon: str
    split_estimand: bool

    def __post_init__(self) -> None:
        window = self.validation_window
        if not isinstance(window, tuple):
            object.__setattr__(self, "validation_window", tuple(window))
            window = self.validation_window
        if not window or any(not isinstance(era, str) or not era for era in window):
            raise ValueError("CapitalContext requires a non-empty validation_window")
        if not self.scoring_target:
            raise ValueError("CapitalContext requires scoring_target")
        if self.scoring_horizon not in VALID_HORIZONS:
            raise ValueError(
                "scoring_horizon must be one of "
                f"{VALID_HORIZONS}, got {self.scoring_horizon!r}"
            )
        if not self.payout_policy_id:
            raise ValueError("CapitalContext requires payout_policy_id")
        if not self.data_fingerprint:
            raise ValueError("CapitalContext requires data_fingerprint")
        if not self.feature_fingerprint:
            raise ValueError("CapitalContext requires feature_fingerprint")
        if not _HEX64_RE.fullmatch(self.validation_key_fingerprint):
            raise ValueError(
                "CapitalContext validation_key_fingerprint must be a 64-char "
                "lowercase hex digest"
            )
        if (
            not isinstance(self.validation_row_count, int)
            or self.validation_row_count < 1
        ):
            raise ValueError("CapitalContext requires validation_row_count >= 1")
        targets = self.trained_targets
        if not isinstance(targets, tuple):
            object.__setattr__(self, "trained_targets", tuple(targets))
            targets = self.trained_targets
        if not targets or any(not isinstance(t, str) or not t for t in targets):
            raise ValueError("CapitalContext requires a non-empty trained_targets")
        if not self.ensemble_target:
            raise ValueError("CapitalContext requires ensemble_target")
        if self.training_horizon not in VALID_HORIZONS:
            raise ValueError(
                "training_horizon must be one of "
                f"{VALID_HORIZONS}, got {self.training_horizon!r}"
            )


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

    pred_series = normalized.get_column("prediction").cast(pl.Float64, strict=True)
    pred_values = pred_series.to_numpy()
    if pred_series.null_count() > 0 or not np.all(np.isfinite(pred_values)):
        raise ValueError("prediction values must be finite")
    if provenance.stage == "submission":
        if not np.all((pred_values > 0.0) & (pred_values < 1.0)):
            raise ValueError("submission predictions must be in (0, 1)")

    ordered = (
        normalized.with_columns(pl.Series("prediction", pred_values, dtype=pl.Float64))
        .sort(["era", "id"])
        .select(list(_REQUIRED_FRAME_COLS))
    )
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
    features: pl.DataFrame | None = None,
    neutralization_proportion: float = 0.0,
    neutralization_cache_dir: Path | None = None,
    as_submission: bool = False,
) -> PredictionSet:
    """Rank-blend components, optionally neutralize, optionally submission-rank.

    Wraps ``Ensembler.blend`` and ``NeutralizationEngine.neutralize``. Does not
    change their math. The returned stage is ``submission`` if
    ``as_submission``, else ``neutralized`` when ``neutralization_proportion > 0``,
    else ``blended``. ``as_submission`` is a rank transform, not a capital
    promotion: OOF / held-out partitions are rejected.
    """
    pred_list = list(pred_cols)
    if not pred_list:
        raise ValueError("pred_cols must contain at least one prediction column")
    if era_col not in frame.columns or id_col not in frame.columns:
        raise ValueError(f"frame must contain {era_col!r} and {id_col!r}")
    proportion = float(neutralization_proportion)
    if not np.isfinite(proportion) or not 0.0 <= proportion <= 1.0:
        raise ValueError("neutralization_proportion must be a finite value in [0, 1]")
    if as_submission and provenance.era_partition in _RESEARCH_PARTITIONS:
        raise ValueError(
            "as_submission cannot relabel a research era_partition "
            f"({provenance.era_partition!r}) into a submission artifact"
        )

    work = frame
    feature_list = list(feature_cols)
    if proportion > 0.0:
        if not feature_list:
            raise ValueError(
                "feature_cols must contain at least one feature when "
                "neutralization_proportion > 0"
            )
        missing = [col for col in feature_list if col not in work.columns]
        if missing:
            if features is None:
                raise ValueError(
                    "features frame is required to join missing neutralization "
                    f"columns {missing}"
                )
            required_feat = [era_col, id_col, *feature_list]
            missing_feat = [c for c in required_feat if c not in features.columns]
            if missing_feat:
                raise ValueError(f"features missing required columns: {missing_feat}")
            feat = features.select(required_feat)
            if feat.select([era_col, id_col]).n_unique() != feat.height:
                raise ValueError("features must have unique (era, id) keys")
            pred_height = work.height
            pred_keys = work.select([era_col, id_col])
            if pred_keys.n_unique() != pred_height:
                raise ValueError("prediction frame must have unique (era, id) keys")
            work = work.join(feat, on=[era_col, id_col], how="inner")
            if work.height != pred_height:
                raise ValueError(
                    "feature join must preserve prediction keys exactly "
                    f"(predictions={pred_height}, joined={work.height})"
                )

    blended = Ensembler().blend(
        work,
        pred_cols=pred_list,
        weights=weights,
        era_col=era_col,
        out_col="prediction",
    )
    stage = "blended"
    composed = blended
    if proportion > 0.0:
        composed = NeutralizationEngine(
            cache_dir=neutralization_cache_dir,
            max_cache_bytes=0,
        ).neutralize(
            composed,
            pred_col="prediction",
            feature_cols=feature_list,
            era_col=era_col,
            proportion=proportion,
        )
        stage = "neutralized"
    if as_submission:
        composed = _per_era_submission_rank(
            composed, era_col=era_col, pred_col="prediction"
        )
        stage = "submission"
    return prediction_set_from_frame(
        composed,
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


def _assert_capital_eligible(
    prediction_set: PredictionSet,
    *,
    context: CapitalContext,
    main_target: str,
    horizon: str,
) -> None:
    provenance = prediction_set.provenance
    if provenance.stage != "validation":
        raise ValueError(
            "capital evaluation requires stage='validation' "
            f"(got {provenance.stage!r}); submission is not capital-eligible"
        )
    if provenance.era_partition != "validation":
        raise ValueError(
            "capital evaluation requires era_partition='validation' "
            f"(got {provenance.era_partition!r})"
        )
    if provenance.selection_bias:
        raise ValueError("capital evaluation refuses selection_bias=True")
    if not provenance.scoring_target:
        raise ValueError("capital evaluation requires scoring_target identity")
    if not provenance.scoring_horizon:
        raise ValueError("capital evaluation requires scoring_horizon identity")
    if not provenance.data_fingerprint:
        raise ValueError("capital evaluation requires data_fingerprint")
    if not provenance.feature_fingerprint:
        raise ValueError("capital evaluation requires feature_fingerprint")
    if not provenance.payout_policy_id:
        raise ValueError("capital evaluation requires payout_policy_id identity")
    if not provenance.ensemble_target:
        raise ValueError("capital evaluation requires ensemble_target identity")
    if not provenance.training_horizon:
        raise ValueError("capital evaluation requires training_horizon identity")
    window = provenance.validation_window
    if not window:
        raise ValueError("capital evaluation requires validation_window identity")
    frame_eras = tuple(
        sorted(prediction_set.frame.get_column("era").unique().to_list(), key=str)
    )
    locked = tuple(sorted((str(era) for era in window), key=str))
    expected = tuple(sorted((str(era) for era in context.validation_window), key=str))
    if frame_eras != locked or locked != expected:
        raise ValueError("validation_window does not match prediction eras")
    frame_key_fp = validation_key_fingerprint(prediction_set.frame)
    if (
        frame_key_fp != context.validation_key_fingerprint
        or prediction_set.frame.height != context.validation_row_count
    ):
        raise ValueError(
            "prediction keys do not match the authoritative validation key "
            f"universe (expected {context.validation_row_count} rows, "
            f"got {prediction_set.frame.height})"
        )
    if provenance.payout_policy_id != context.payout_policy_id:
        raise ValueError(
            "payout_policy_id provenance "
            f"{provenance.payout_policy_id!r} does not match CapitalContext "
            f"{context.payout_policy_id!r}"
        )
    if provenance.scoring_target != context.scoring_target:
        raise ValueError(
            "scoring_target provenance "
            f"{provenance.scoring_target!r} does not match CapitalContext "
            f"{context.scoring_target!r}"
        )
    if provenance.scoring_horizon != context.scoring_horizon:
        raise ValueError(
            "scoring_horizon provenance "
            f"{provenance.scoring_horizon!r} does not match CapitalContext "
            f"{context.scoring_horizon!r}"
        )
    if provenance.trained_targets != context.trained_targets:
        raise ValueError(
            "trained_targets provenance "
            f"{provenance.trained_targets!r} does not match CapitalContext "
            f"{context.trained_targets!r}"
        )
    if provenance.ensemble_target != context.ensemble_target:
        raise ValueError(
            "ensemble_target provenance "
            f"{provenance.ensemble_target!r} does not match CapitalContext "
            f"{context.ensemble_target!r}"
        )
    if provenance.training_horizon != context.training_horizon:
        raise ValueError(
            "training_horizon provenance "
            f"{provenance.training_horizon!r} does not match CapitalContext "
            f"{context.training_horizon!r}"
        )
    if provenance.split_estimand != context.split_estimand:
        raise ValueError(
            "split_estimand provenance "
            f"{provenance.split_estimand!r} does not match CapitalContext "
            f"{context.split_estimand!r}"
        )
    if provenance.data_fingerprint != context.data_fingerprint:
        raise ValueError("data_fingerprint does not match CapitalContext")
    if provenance.feature_fingerprint != context.feature_fingerprint:
        raise ValueError("feature_fingerprint does not match CapitalContext")
    if provenance.scoring_target != main_target:
        raise ValueError(
            "scoring_target provenance "
            f"{provenance.scoring_target!r} does not match evaluation "
            f"main_target={main_target!r}"
        )
    if provenance.scoring_horizon != horizon:
        raise ValueError(
            "scoring_horizon provenance "
            f"{provenance.scoring_horizon!r} does not match evaluation "
            f"horizon={horizon!r}"
        )


def evaluate_prediction_set(
    prediction_set: PredictionSet,
    *,
    meta_model: pl.DataFrame,
    benchmarks: pl.DataFrame | None,
    features: pl.DataFrame,
    targets: pl.DataFrame,
    allow_research_stage: bool = False,
    capital_context: CapitalContext | None = None,
    **kwargs: Any,
):
    """Score a ``PredictionSet`` through ``evaluate_model``.

    Capital evaluation requires a :class:`CapitalContext` produced by a
    trusted validation loader (the runner derives it from the loaded, purged
    validation data) plus ``stage='validation'``,
    ``era_partition='validation'``, ``selection_bias=False``, required
    scoring/payout/fingerprint identity, training-estimand identity, and
    exact ``(era, id)`` key coverage — sparse or extra prediction rows are
    rejected. ``submission`` is a rank-domain artifact, not a capital
    scorecard. Research stages return :class:`ResearchEvaluation` only when
    ``allow_research_stage=True``. Does not import ``nmr.models``.
    """
    from nmr.scorecard import evaluate_model

    payout_policy = kwargs.get("payout_policy")
    if payout_policy is None:
        raise ValueError("payout_policy is required")
    resolved_policy = resolve_payout_policy(payout_policy)
    main_target = kwargs.get(
        "main_target", resolved_policy.target or _DEFAULT_MAIN_TARGET
    )
    horizon = kwargs.get("horizon", resolved_policy.scoring_horizon or _DEFAULT_HORIZON)

    if allow_research_stage:
        scorecard = evaluate_model(
            prediction_set.frame,
            meta_model=meta_model,
            benchmarks=benchmarks,
            features=features,
            targets=targets,
            **kwargs,
        )
        return ResearchEvaluation(scorecard=scorecard, is_capital=False)

    if capital_context is None:
        raise ValueError("capital evaluation requires CapitalContext")
    if capital_context.scoring_target != str(main_target):
        raise ValueError(
            "CapitalContext scoring_target "
            f"{capital_context.scoring_target!r} does not match evaluation "
            f"main_target={main_target!r}"
        )
    if capital_context.scoring_horizon != str(horizon):
        raise ValueError(
            "CapitalContext scoring_horizon "
            f"{capital_context.scoring_horizon!r} does not match evaluation "
            f"horizon={horizon!r}"
        )
    if capital_context.payout_policy_id != resolved_policy.policy_id:
        raise ValueError(
            "CapitalContext payout_policy_id "
            f"{capital_context.payout_policy_id!r} does not match evaluation "
            f"payout_policy={resolved_policy.policy_id!r}"
        )
    capital_frames = [
        ("predictions", prediction_set.frame),
        ("meta_model", meta_model),
        ("features", features),
        ("targets", targets),
    ]
    if benchmarks is not None:
        capital_frames.append(("benchmarks", benchmarks))
    for name, frame in capital_frames:
        _assert_capital_frame_key_universe(name, frame, capital_context)
    _assert_capital_eligible(
        prediction_set,
        context=capital_context,
        main_target=str(main_target),
        horizon=str(horizon),
    )
    return evaluate_model(
        prediction_set.frame,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        **kwargs,
        _expected_key_fingerprint=capital_context.validation_key_fingerprint,
        _expected_row_count=capital_context.validation_row_count,
        _expected_era_window=capital_context.validation_window,
    )
