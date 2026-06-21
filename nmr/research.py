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

from nmr.config import ExperimentConfig, set_global_seeds
from nmr.data import IngestionAgent
from nmr.ensemble import Ensembler
from nmr.evaluation import EvaluationEngine, MetricSummary
from nmr.models import ModelOrchestrator
from nmr.risk import NeutralizationEngine
from nmr.splitter import PurgedEraSplitter

__all__ = [
    "SweepResult",
    "HyperparameterSweep",
    "NeutralizationFrontier",
    "neutralization_frontier",
    "feature_exposure_report",
]


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
            metric_value = _held_out_metric(cfg, metric_name=self._metric)
            trials.append(
                {
                    "trial_id": trial_idx,
                    "params_json": json.dumps(params, sort_keys=True),
                    "metric_value": float(metric_value),
                    "metric": self._metric,
                }
            )

        trial_df = pl.DataFrame(trials).sort(
            ["metric_value", "trial_id"], descending=[True, False]
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
) -> NeutralizationFrontier:
    if not proportions:
        raise ValueError("proportions must contain at least one value")

    risk_engine = NeutralizationEngine()
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
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")

    evaluator = EvaluationEngine("custom")
    eras = sorted({str(value) for value in oof.get_column(era_col).to_list()}, key=int)
    rows: list[dict[str, Any]] = []
    for feature in feature_list:
        per_era_values: list[float] = []
        for era in eras:
            era_df = oof.filter(pl.col(era_col) == era)
            eval_df = era_df.select([pred_col, feature, era_col]).rename(
                {feature: "target"}
            )
            corr_map = evaluator.per_era_corr(
                eval_df,
                pred_col=pred_col,
                target_col="target",
                era_col=era_col,
            )
            per_era_values.extend(float(value) for value in corr_map.values())

        values = np.asarray(per_era_values, dtype=float)
        rows.append(
            {
                "feature": feature,
                "mean_abs_exposure": (
                    float(np.mean(np.abs(values))) if values.size else 0.0
                ),
                "max_abs_exposure": (
                    float(np.max(np.abs(values))) if values.size else 0.0
                ),
            }
        )

    return pl.DataFrame(rows).sort("max_abs_exposure", descending=True)


def _held_out_metric(config: ExperimentConfig, *, metric_name: str) -> float:
    set_global_seeds(config.run.seed)
    agent = IngestionAgent(config.data)
    feature_cols = agent.features(config.data.feature_set)
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
        method="ridge",
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
        feature_frame = held_out_df.select(feature_cols).to_pandas()
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
    neutralized = NeutralizationEngine().neutralize(
        blended,
        pred_col="prediction",
        feature_cols=feature_cols,
        era_col="era",
        proportion=1.0,
    )
    evaluator = EvaluationEngine(config.evaluation.backend)
    per_era = evaluator.per_era_corr(
        neutralized,
        pred_col="prediction",
        target_col=main_target,
        era_col="era",
    )
    summary = evaluator.summarize(per_era)
    if not hasattr(summary, metric_name):
        raise ValueError(f"Unknown metric {metric_name!r}")
    return float(getattr(summary, metric_name))


def _train_multi_target_oof(
    modeler: ModelOrchestrator,
    df: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    splitter: PurgedEraSplitter,
    targets: Sequence[str],
) -> pl.DataFrame:
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


def _held_out_partition(
    eras: Sequence[str],
    *,
    frac: float,
    purge_eras: int,
) -> tuple[list[str], list[str], list[str]]:
    if purge_eras < 0:
        raise ValueError("purge_eras must be >= 0")

    unique = sorted({int(era) for era in eras})
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
        [str(value) for value in train_nums],
        [str(value) for value in purge_nums],
        [str(value) for value in held_out_nums],
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
