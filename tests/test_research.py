"""Tests for research enablement helpers."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from nmr.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    SplitConfig,
)
from nmr.evaluation import EvaluationEngine
from nmr.research import (
    HyperparameterSweep,
    _held_out_partition,
    feature_exposure_report,
    neutralization_frontier,
)
from nmr.risk import NeutralizationEngine


def _train_frame() -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for era in range(1, 21):
        for idx in range(8):
            f1 = 0.02 * era + 0.01 * idx
            f2 = -0.01 * era + 0.015 * idx
            target = 0.7 * f1 - 0.4 * f2 + 0.05 * np.sin(idx)
            target_alt = 0.4 * f1 + 0.6 * f2 - 0.03 * np.cos(idx)
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": float(f1),
                    "f2": float(f2),
                    "target": float(target),
                    "target_alt": float(target_alt),
                }
            )
    return pl.DataFrame(rows)


def _write_data(tmp_path) -> ExperimentConfig:
    data_root = tmp_path / "data"
    vdir = data_root / "vresearch"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "features.json").write_text(
        json.dumps(
            {
                "feature_sets": {
                    "small": ["f1", "f2"],
                    "medium": ["f1", "f2"],
                    "all": ["f1", "f2"],
                },
                "targets": ["target", "target_alt"],
            }
        ),
        encoding="utf-8",
    )
    _train_frame().write_parquet(vdir / "train.parquet")
    return ExperimentConfig(
        data=DataConfig(
            version="vresearch",
            feature_set="small",
            targets=("target", "target_alt"),
            data_dir=data_root,
        ),
        split=SplitConfig(
            scheme="walk_forward", purge_eras=1, embargo_eras=0, n_folds=3
        ),
        model=ModelConfig(
            backend="lightgbm",
            preset="fast",
            params={"n_estimators": 10, "learning_rate": 0.05},
        ),
        evaluation=EvalConfig(backend="custom", main_target="target"),
        run=RunConfig(name="research", seed=19, artifacts_dir=tmp_path / "artifacts"),
    )


def test_sweep_is_deterministic_and_held_out(tmp_path) -> None:
    cfg = _write_data(tmp_path)
    space = {"n_estimators": [6, 8], "learning_rate": [0.03, 0.07]}
    first = HyperparameterSweep(cfg, metric="sharpe").run(space, n_trials=4, seed=123)
    second = HyperparameterSweep(cfg, metric="sharpe").run(space, n_trials=4, seed=123)

    assert first.trials.equals(second.trials)
    assert first.best_params == second.best_params
    assert first.best_value == second.best_value


def test_held_out_partition_enforces_purge_gap(tmp_path) -> None:
    cfg = _write_data(tmp_path)
    eras = _train_frame().get_column("era").to_list()
    train_eras, _, held_out_eras = _held_out_partition(
        eras,
        frac=0.2,
        purge_eras=cfg.split.purge_eras,
    )

    train_max = max(map(int, train_eras))
    held_out_min = min(map(int, held_out_eras))
    assert held_out_min - train_max - 1 >= cfg.split.purge_eras


def test_neutralization_frontier_matches_endpoints(tmp_path) -> None:
    cfg = _write_data(tmp_path)
    df = _train_frame()
    proportions = [0.0, 1.0]
    frontier = neutralization_frontier(
        df,
        feature_cols=["f1", "f2"],
        proportions=proportions,
        target_col="target",
        pred_col="target_alt",
    )

    evaluator = EvaluationEngine("custom")
    risk = NeutralizationEngine()

    raw_scores = evaluator.per_era_corr(df, pred_col="target_alt", target_col="target")
    full_neutral = risk.neutralize(
        df,
        pred_col="target_alt",
        feature_cols=["f1", "f2"],
        proportion=1.0,
    )
    full_scores = evaluator.per_era_corr(
        full_neutral, pred_col="target_alt", target_col="target"
    )

    assert frontier.proportions == proportions
    assert frontier.metrics[0] == evaluator.summarize(raw_scores)
    assert frontier.metrics[1] == evaluator.summarize(full_scores)


def test_feature_exposure_report_is_deterministic_and_sorted() -> None:
    df = _train_frame().with_columns(pl.col("target_alt").alias("prediction"))
    first = feature_exposure_report(
        df, feature_cols=["f1", "f2"], pred_col="prediction"
    )
    second = feature_exposure_report(
        df, feature_cols=["f1", "f2"], pred_col="prediction"
    )

    assert first.equals(second)
    assert first.columns == ["feature", "mean_abs_exposure", "max_abs_exposure"]
    values = first.get_column("max_abs_exposure").to_list()
    assert values == sorted(values, reverse=True)
