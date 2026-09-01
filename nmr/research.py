"""Research helpers: deterministic sweeps and diagnostics over existing components."""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nmr import paths
from nmr._oof import train_multi_target_oof
from nmr.config import ExperimentConfig, set_global_seeds
from nmr.data import IngestionAgent
from nmr.ensemble import Ensembler
from nmr.evaluation import EvaluationEngine, MetricSummary
from nmr.inference import ac_adjusted_sharpe, era_series_stats
from nmr.models import ModelOrchestrator, coerce_float32_features
from nmr.risk import NeutralizationEngine
from nmr.splitter import PurgedEraSplitter

__all__ = [
    "SweepResult",
    "HyperparameterSweep",
    "NeutralizationFrontier",
    "neutralization_frontier",
    "feature_exposure_report",
    "metric_direction",
]

_METRIC_DIRECTIONS = {
    "mean": "maximize",
    "sharpe": "maximize",
    "corr_sharpe_ac": "maximize",
    "std": "minimize",
    "max_drawdown": "minimize",
}


def metric_direction(metric: str) -> str:
    try:
        return _METRIC_DIRECTIONS[metric]
    except KeyError as exc:
        raise ValueError(
            f"metric={metric!r} not in {sorted(_METRIC_DIRECTIONS)}"
        ) from exc


@dataclass(frozen=True)
class SweepResult:
    trials: pl.DataFrame
    best_params: dict[str, Any]
    best_value: float


@dataclass(frozen=True)
class NeutralizationFrontier:
    proportions: list[float]
    metrics: list[MetricSummary]


class HyperparameterSweep:
    def __init__(self, base_config: ExperimentConfig, *, metric: str = "sharpe"):
        self._base_config = base_config
        self._metric = metric
        self._direction = metric_direction(metric)

    def run(self, space: dict, *, n_trials: int, seed: int) -> SweepResult:
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if not space:
            raise ValueError("space must contain at least one parameter")

        set_global_seeds(seed)
        rng = np.random.default_rng(seed)
        keys = sorted(space.keys())
        candidate_lists = [_normalize_options(space[key]) for key in keys]

        all_candidates = [
            dict(zip(keys, values)) for values in itertools.product(*candidate_lists)
        ]
        if not all_candidates:
            raise ValueError("search space has no valid candidates")

        indices = np.arange(len(all_candidates))
        rng.shuffle(indices)
        chosen = [
            all_candidates[int(idx)] for idx in indices[: min(n_trials, len(indices))]
        ]

        trials: list[dict[str, Any]] = []
        for trial_idx, params in enumerate(chosen):
            cfg = _override_config(self._base_config, params)
            metric_value, moments = _held_out_metric_full(cfg, metric_name=self._metric)
            trials.append(
                {
                    "trial_id": trial_idx,
                    "params_json": json.dumps(params, sort_keys=True),
                    "metric_value": float(metric_value),
                    "metric": self._metric,
                    "ic_sharpe": moments.ic_sharpe,
                    "ic_skew": moments.ic_skew,
                    "ic_kurt": moments.ic_kurt,
                    "ic_n_eras": moments.ic_n_eras,
                    "ic_std": moments.ic_std,
                }
            )

        trial_df = pl.DataFrame(trials).sort(
            ["metric_value", "trial_id"],
            descending=[self._direction == "maximize", False],
        )
        best_row = trial_df.row(0, named=True)
        best_params = json.loads(best_row["params_json"])
        best_value = float(best_row["metric_value"])
        return SweepResult(
            trials=trial_df, best_params=best_params, best_value=best_value
        )


def neutralization_frontier(
    result_oof: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    proportions: Sequence[float],
    target_col: str,
    era_col: str = "era",
    pred_col: str = "prediction",
    backend: str = "custom",
    cache_dir: Path | None = None,
) -> NeutralizationFrontier:
    if not proportions:
        raise ValueError("proportions must contain at least one value")

    risk_engine = NeutralizationEngine(cache_dir=cache_dir)
    evaluator = EvaluationEngine(backend)
    metrics: list[MetricSummary] = []
    normalized_props = [float(p) for p in proportions]

    for proportion in normalized_props:
        neutralized = risk_engine.neutralize(
            result_oof,
            pred_col=pred_col,
            feature_cols=feature_cols,
            era_col=era_col,
            proportion=proportion,
        )
        per_era = evaluator.per_era_corr(
            neutralized,
            pred_col=pred_col,
            target_col=target_col,
            era_col=era_col,
        )
        metrics.append(evaluator.summarize(per_era))

    return NeutralizationFrontier(proportions=normalized_props, metrics=metrics)


def feature_exposure_report(
    oof: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    era_col: str = "era",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Per-era Pearson correlation of predictions vs each feature (vectorized).

    Definition (documented in ARCHITECTURE.md §L): plain Pearson correlation of
    the raw prediction and feature columns per era, then aggregated. This is the
    community-standard exposure definition (it is NOT the power-1.5 Numerai
    CORR used by per_era_corr). Values changed vs the pre-2026-08-05
    implementation; recorded exposure numbers are not comparable across that
    boundary.
    """
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")

    per_era: dict[str, np.ndarray] = {}
    parts = oof.select([era_col, pred_col, *feature_list]).partition_by(
        era_col, maintain_order=True
    )
    for part in parts:
        era = str(part.get_column(era_col).to_list()[0])
        clean = part.drop_nulls()
        if clean.is_empty():
            per_era[era] = np.zeros(len(feature_list), dtype=float)
            continue
        pred = clean.get_column(pred_col).cast(pl.Float64).to_numpy()
        features = clean.select(feature_list).cast(pl.Float64).to_numpy()
        per_era[era] = _pred_feature_pearson(pred, features)

    if not per_era:
        return pl.DataFrame(
            {
                "feature": feature_list,
                "mean_abs_exposure": [0.0] * len(feature_list),
                "max_abs_exposure": [0.0] * len(feature_list),
            }
        )

    eras = sorted(per_era, key=int)
    matrix = np.column_stack([per_era[era] for era in eras])
    rows = [
        {
            "feature": feature,
            "mean_abs_exposure": float(np.mean(np.abs(matrix[i]))),
            "max_abs_exposure": float(np.max(np.abs(matrix[i]))),
        }
        for i, feature in enumerate(feature_list)
    ]
    return pl.DataFrame(rows).sort("max_abs_exposure", descending=True)


def _pred_feature_pearson(pred: np.ndarray, features: np.ndarray) -> np.ndarray:
    pred_centered = pred - np.mean(pred)
    pred_norm = float(np.linalg.norm(pred_centered))
    if pred_norm == 0.0:
        return np.zeros(features.shape[1], dtype=float)
    feature_centered = features - np.mean(features, axis=0)
    denoms = np.linalg.norm(feature_centered, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = (feature_centered.T @ pred_centered) / (denoms * pred_norm)
    return np.where(np.isfinite(corrs), corrs, 0.0)


def _per_era_ac_sharpe(per_era: dict[str, float], *, horizon: str = "20D") -> float:
    """Autocorrelation-adjusted Sharpe of a per-era metric series.

    Chronological order is mandatory for autocorrelation: ``per_era``'s
    insertion order follows the frame's lexicographic era sort ("1","10","11",...)
    which would corrupt the AC computation. Sort era keys numerically
    (scorecard._sorted_numeric_keys idiom); era labels are numeric strings.
    """
    sorted_keys = sorted(per_era, key=int)
    series = [per_era[k] for k in sorted_keys]
    return ac_adjusted_sharpe(series, horizon=horizon)


def _held_out_metric(config: ExperimentConfig, *, metric_name: str) -> float:
    """Held-out metric scalar (the public contract used by sweeps).

    Delegates to :func:`_held_out_metric_full`; the per-era series moments are
    computed in the same training pass and discarded here.
    """
    return _held_out_metric_full(config, metric_name=metric_name)[0]


@dataclass(frozen=True)
class _HeldOutMoments:
    """Per-era IC series moments of the held-out partition (same series that
    produced the metric — the same-distribution invariant for fleet DSR)."""

    ic_sharpe: float
    ic_skew: float
    ic_kurt: float
    ic_n_eras: int
    ic_std: float


def _held_out_metric_full(
    config: ExperimentConfig, *, metric_name: str
) -> tuple[float, _HeldOutMoments]:
    set_global_seeds(config.run.seed)
    agent = IngestionAgent(config.data)
    feature_cols = agent.features(config.data.resolved_feature_set)
    main_target = config.evaluation.main_target
    targets = list(dict.fromkeys([*config.data.targets, main_target]))

    frame = agent.load("train", columns=["era", "id", *feature_cols, *targets]).sort(
        ["era", "id"]
    )
    train_eras, purge_eras, held_out_eras = _held_out_partition(
        frame.get_column("era").to_list(),
        frac=0.2,
        purge_eras=config.split.purge_eras,
    )
    train_df = frame.filter(pl.col("era").is_in(train_eras))
    held_out_df = frame.filter(pl.col("era").is_in(held_out_eras))
    _ = purge_eras

    if train_df.is_empty() or held_out_df.is_empty():
        raise ValueError("Held-out split is empty; increase era history")

    splitter = PurgedEraSplitter(config.split)
    modeler = ModelOrchestrator(config.model, seed=config.run.seed)
    cv_oof = _train_multi_target_oof(
        modeler,
        train_df,
        feature_cols=feature_cols,
        splitter=splitter,
        targets=config.data.targets,
    )

    joined_train = train_df.select(["id", "era", main_target, *feature_cols]).join(
        cv_oof, on=["id", "era"], how="inner"
    )
    pred_cols = [col for col in cv_oof.columns if col.startswith("pred_")]

    ensembler = Ensembler()
    weights = ensembler.learn_weights(
        joined_train.select(["era", *pred_cols, main_target]),
        pred_cols=pred_cols,
        target_col=main_target,
        era_col="era",
        method=config.ensemble.method,
    )

    anchor_splitter = PurgedEraSplitter(
        dataclasses.replace(config.split, scheme="anchor", n_folds=1)
    )
    anchor_predictions: list[pl.DataFrame] = []
    for target in config.data.targets:
        model, _ = modeler.train_anchor_fold(
            train_df,
            feature_cols=feature_cols,
            target_col=target,
            splitter=anchor_splitter,
            era_col="era",
        )
        # numpy feature matrix (zero-copy float32) — the pandas path goes
        # through pyarrow and doubles memory (OOM at 3,555 features)
        feature_frame = coerce_float32_features(held_out_df, feature_cols).to_numpy()
        raw_pred = np.asarray(model.predict(feature_frame), dtype=float)
        anchor_predictions.append(
            held_out_df.select(["id", "era"]).with_columns(
                pl.Series(f"pred_{target}", raw_pred)
            )
        )

    merged_pred = anchor_predictions[0]
    for frame_part in anchor_predictions[1:]:
        merged_pred = merged_pred.join(frame_part, on=["id", "era"], how="inner")

    held_out_joined = held_out_df.select(
        ["id", "era", main_target, *feature_cols]
    ).join(
        merged_pred,
        on=["id", "era"],
        how="inner",
    )
    blended = ensembler.blend(
        held_out_joined,
        pred_cols=pred_cols,
        weights=weights,
        era_col="era",
        out_col="prediction",
    )
    neutralized = NeutralizationEngine(
        cache_dir=paths.shared_cache_dir(config.run.artifacts_dir) / "neutralization",
        max_cache_bytes=config.risk.cache_max_bytes,
    ).neutralize(
        blended,
        pred_col="prediction",
        feature_cols=feature_cols,
        era_col="era",
        proportion=config.risk.neutralization_proportion,
    )
    evaluator = EvaluationEngine(config.evaluation.backend)
    per_era = evaluator.per_era_corr(
        neutralized,
        pred_col="prediction",
        target_col=main_target,
        era_col="era",
    )
    stats = era_series_stats([per_era[k] for k in sorted(per_era, key=int)])
    moments = _HeldOutMoments(
        ic_sharpe=stats.sharpe,
        ic_skew=stats.skew,
        ic_kurt=stats.kurt,
        ic_n_eras=stats.n,
        ic_std=stats.std,
    )
    if metric_name == "corr_sharpe_ac":
        return _per_era_ac_sharpe(per_era, horizon="20D"), moments
    summary = evaluator.summarize(per_era)
    if not hasattr(summary, metric_name):
        raise ValueError(f"Unknown metric {metric_name!r}")
    return float(getattr(summary, metric_name)), moments


def _train_multi_target_oof(
    modeler: ModelOrchestrator,
    df: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    splitter: PurgedEraSplitter,
    targets: Sequence[str],
) -> pl.DataFrame:
    """C10 (audit SEV-2 #5): thin alias over the shared OOF implementation.

    The duplicated copy this replaced could silently drift from the runner's
    OOF path; the single source now lives in ``nmr._oof``.
    """
    return train_multi_target_oof(
        modeler, df, feature_cols=feature_cols, splitter=splitter, targets=targets
    )


def _held_out_partition(
    eras: Sequence[str],
    *,
    frac: float,
    purge_eras: int,
) -> tuple[list[str], list[str], list[str]]:
    if purge_eras < 0:
        raise ValueError("purge_eras must be >= 0")

    # Preserve the ORIGINAL era labels (zero-padded "0575"): converting to
    # str(int) produced "575", which matches nothing in is_in() filters on
    # padded data — the HPO held-out evaluation silently dropped every era
    # below 1000 (regression 2026-08-11, same class as the runner purge bug).
    numeric_to_label: dict[int, str] = {}
    for era in eras:
        numeric_to_label.setdefault(int(era), era)

    unique = sorted(numeric_to_label)
    hold_count = max(1, int(round(len(unique) * frac)))
    held_out_nums = unique[-hold_count:]
    held_out_set = set(held_out_nums)
    held_out_min = min(held_out_nums)

    purge_set = {
        value for value in unique if held_out_min - purge_eras <= value < held_out_min
    }

    train_nums = [
        value
        for value in unique
        if value not in held_out_set and value not in purge_set
    ]
    purge_nums = [value for value in unique if value in purge_set]

    return (
        [numeric_to_label[value] for value in train_nums],
        [numeric_to_label[value] for value in purge_nums],
        [numeric_to_label[value] for value in held_out_nums],
    )


def _normalize_options(raw: Any) -> list[Any]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values = list(raw)
    else:
        values = [raw]
    if not values:
        raise ValueError("search space entry cannot be empty")
    return values


def _override_config(
    base: ExperimentConfig, params: dict[str, Any]
) -> ExperimentConfig:
    config = copy.deepcopy(base)
    model = dataclasses.replace(config.model, params={**config.model.params, **params})
    return dataclasses.replace(config, model=model)
