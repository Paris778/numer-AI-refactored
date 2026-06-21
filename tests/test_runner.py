"""Tests for deterministic experiment orchestration."""

from __future__ import annotations

import json

import pandas as pd
import polars as pl

from nmr.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    SplitConfig,
)
from nmr.deployment import load_predict
from nmr.runner import ExperimentRunner
from nmr.splitter import PurgedEraSplitter


def _build_train_frame() -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for era in range(1, 13):
        for idx in range(6):
            f1 = (era * 0.03) + (idx * 0.02)
            f2 = (era * -0.02) + (idx * 0.01)
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": 0.6 * f1 - 0.3 * f2 + 0.05 * era,
                    "target_alt": 0.2 * f1 + 0.7 * f2 - 0.04 * era,
                }
            )
    return pl.DataFrame(rows)


def _write_synthetic_data(root) -> None:
    version_dir = root / "vtest"
    version_dir.mkdir(parents=True, exist_ok=True)
    features = {
        "feature_sets": {
            "small": ["f1", "f2"],
            "medium": ["f1", "f2"],
            "all": ["f1", "f2"],
        },
        "targets": ["target", "target_alt"],
    }
    (version_dir / "features.json").write_text(json.dumps(features), encoding="utf-8")
    _build_train_frame().write_parquet(version_dir / "train.parquet")


def _config(tmp_path) -> ExperimentConfig:
    data_root = tmp_path / "data"
    _write_synthetic_data(data_root)
    return ExperimentConfig(
        data=DataConfig(
            version="vtest",
            feature_set="small",
            targets=("target", "target_alt"),
            data_dir=data_root,
        ),
        split=SplitConfig(
            scheme="walk_forward", purge_eras=1, embargo_eras=0, n_folds=2
        ),
        model=ModelConfig(
            backend="lightgbm",
            preset="fast",
            params={"n_estimators": 8, "learning_rate": 0.05},
        ),
        evaluation=EvalConfig(backend="custom", main_target="target"),
        run=RunConfig(
            seed=17, artifacts_dir=tmp_path / "artifacts", name="runner-test"
        ),
    )


def test_runner_is_deterministic_and_leakage_safe(tmp_path) -> None:
    cfg = _config(tmp_path)
    runner = ExperimentRunner(cfg)
    first = runner.run(deploy=False)
    second = runner.run(deploy=False)

    assert first.run_id == second.run_id
    assert first.oof.equals(second.oof)
    assert first.metrics == second.metrics
    assert first.artifact is None and second.artifact is None

    eras = _build_train_frame().get_column("era").to_list()
    folds = PurgedEraSplitter(cfg.split).split(eras)
    expected_val_eras = {era for fold in folds for era in fold.val_eras}
    assert set(first.oof.get_column("era").to_list()) == expected_val_eras


def test_runner_deploy_serializes_reloadable_predict(tmp_path) -> None:
    cfg = _config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=True)

    assert result.artifact is not None
    loaded_predict = load_predict(result.artifact.path)
    live_features = pd.DataFrame(
        {"f1": [0.1, 0.2], "f2": [0.3, 0.4]},
        index=["id_a", "id_b"],
    )
    prediction = loaded_predict(live_features)
    assert list(prediction.columns) == ["prediction"]
    assert prediction.index.tolist() == ["id_a", "id_b"]


def test_run_id_is_path_independent_and_seed_sensitive(tmp_path) -> None:
    cfg_a = _config(tmp_path / "a")
    cfg_b = _config(tmp_path / "b")

    runner_a = ExperimentRunner(cfg_a)
    runner_b = ExperimentRunner(cfg_b)
    assert runner_a._run_id == runner_b._run_id

    cfg_seed_flip = ExperimentConfig(
        data=cfg_a.data,
        split=cfg_a.split,
        model=cfg_a.model,
        evaluation=cfg_a.evaluation,
        run=RunConfig(
            seed=cfg_a.run.seed + 1,
            artifacts_dir=cfg_a.run.artifacts_dir,
            name=cfg_a.run.name,
        ),
    )
    runner_seed_flip = ExperimentRunner(cfg_seed_flip)
    assert runner_seed_flip._run_id != runner_a._run_id
