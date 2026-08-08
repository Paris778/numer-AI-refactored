"""Tests for deterministic experiment orchestration."""

from __future__ import annotations

import json

import numpy as np
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

    val_rows = []
    for era in range(13, 19):
        for idx in range(6):
            # NOTE: features are shifted back into the training envelope
            # ((era - 11) => train-era-2..7 feature values). The brief's raw
            # `era * 0.03` formula puts every validation row outside the model's
            # training range, so the (correct) full-history model extrapolates to
            # a single constant leaf per era and full neutralization maps that to
            # exactly 0.0 — making the F-019 fidelity test's Spearman rho
            # undefined. Shifted features keep eras/assertions unchanged while
            # yielding non-degenerate predictions.
            f1 = ((era - 11) * 0.03) + (idx * 0.02)
            f2 = ((era - 11) * -0.02) + (idx * 0.01)
            val_rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": 0.6 * f1 - 0.3 * f2 + 0.05 * era,
                    "target_alt": 0.2 * f1 + 0.7 * f2 - 0.04 * era,
                }
            )
    val = pl.DataFrame(val_rows)
    val.write_parquet(version_dir / "validation.parquet")
    val.select(["era", "id"]).with_columns(
        pl.lit(0.35).alias("numerai_meta_model")
    ).write_parquet(version_dir / "meta_model.parquet")
    val.select(["era", "id"]).with_columns(
        pl.lit(0.2).alias("bench_cyrusd_20")
    ).write_parquet(version_dir / "validation_benchmark_models.parquet")


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
            params={"n_estimators": 10, "learning_rate": 0.05, "min_data_in_leaf": 2},
        ),
        evaluation=EvalConfig(
            backend="custom", main_target="target",
            metrics=("corr", "fnc", "sharpe"),  # mmc is validation-only; guard rejects it without the validation stage
            validation_scorecard=False,
        ),
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
        {"f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "f2": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        index=[f"id_{i}" for i in range(6)],
    )
    prediction = loaded_predict(live_features)
    assert list(prediction.columns) == ["prediction"]
    assert prediction.index.tolist() == [f"id_{i}" for i in range(6)]
    assert prediction["prediction"].notna().all()
    assert prediction["prediction"].nunique() > 1  # non-constant pipeline output


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


def _validation_config(tmp_path) -> ExperimentConfig:
    cfg = _config(tmp_path)
    return ExperimentConfig(
        data=cfg.data,
        split=cfg.split,
        model=cfg.model,
        evaluation=EvalConfig(
            backend="custom", main_target="target", validation_scorecard=True
        ),
        run=cfg.run,
    )


def test_validation_stage_produces_scorecard_and_purges_first_eras(tmp_path) -> None:
    cfg = _validation_config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=True)

    assert result.scorecard is not None
    assert result.scorecard.model_id == result.run_id
    assert result.manifest["validation_purge_dropped_first_eras"] == cfg.split.purge_eras
    assert result.validation_predictions is not None
    scored_eras = set(result.validation_predictions.get_column("era").to_list())
    assert "13" not in scored_eras  # purge_eras=1 -> era 13 dropped
    assert "14" in scored_eras
    assert result.artifact is not None


def test_deployed_artifact_matches_validation_stage_predictions(tmp_path) -> None:
    """F-019 fidelity: load_predict reproduces the scored validation pipeline."""
    import pandas as pd
    from scipy.stats import spearmanr

    cfg = _validation_config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=True)
    assert result.artifact is not None and result.validation_predictions is not None

    val = pl.read_parquet(cfg.data.path("validation.parquet")).filter(
        pl.col("era") != "13"
    )
    features_pd = val.select(["id", "era", "f1", "f2"]).to_pandas().set_index("id")
    loaded = load_predict(result.artifact.path)
    out = loaded(features_pd)

    expected = result.validation_predictions.sort("id")
    actual = pl.from_pandas(out.reset_index().rename(columns={"index": "id"})).sort("id")
    rho, _ = spearmanr(expected.get_column("prediction").to_numpy(),
                       actual.get_column("prediction").to_numpy())
    assert rho > 0.999
    assert np.allclose(
        expected.get_column("prediction").to_numpy(),
        actual.get_column("prediction").to_numpy(),
        atol=1e-12,
    )


def test_single_fold_falls_back_to_uniform_weights(tmp_path, caplog) -> None:
    cfg = _config(tmp_path)
    single_fold = ExperimentConfig(
        data=cfg.data, split=SplitConfig(scheme="anchor", purge_eras=1, n_folds=1),
        model=cfg.model,
        evaluation=EvalConfig(
            backend="custom", main_target="target",
            metrics=("corr", "fnc", "sharpe"),  # mmc is validation-only; guard rejects it without the validation stage
            validation_scorecard=False,
        ),
        run=cfg.run,
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="nmr.runner"):
        result = ExperimentRunner(single_fold).run(deploy=False)
    assert result.manifest["weights"] == [0.5, 0.5]  # 2 components, uniform
    assert any("uniform" in record.message for record in caplog.records)


def test_mmc_metric_requires_validation_scorecard(tmp_path) -> None:
    cfg = _config(tmp_path)
    bad = ExperimentConfig(
        data=cfg.data, split=cfg.split, model=cfg.model,
        evaluation=EvalConfig(
            backend="custom", main_target="target",
            validation_scorecard=False, metrics=("corr", "mmc", "sharpe"),
        ),
        run=cfg.run,
    )
    import pytest as _pytest
    with _pytest.raises(ValueError, match="mmc"):
        ExperimentRunner(bad).run(deploy=False)


def test_fold_held_out_weight_learning_and_scoring_eras(tmp_path) -> None:
    cfg = _config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=False)
    folds = PurgedEraSplitter(cfg.split).split(
        _build_train_frame().get_column("era").to_list()
    )
    assert set(result.manifest["weight_learning_eras"]) == {
        era for fold in folds[:-1] for era in fold.val_eras
    }
    assert set(result.manifest["scoring_eras"]) == set(folds[-1].val_eras)


def test_feature_subset_changes_run_id_and_uses_subset_features(tmp_path) -> None:
    """feature_subset must change the run fingerprint and reach the data layer."""
    import json as _json

    cfg = _config(tmp_path)
    # vtest features.json has small == medium == all == [f1, f2]; add a family
    # set via the data dir used by _config and re-run with feature_subset.
    version_dir = cfg.data.data_dir / "vtest"
    features = _json.loads((version_dir / "features.json").read_text(encoding="utf-8"))
    features["feature_sets"]["sunshine"] = ["f1", "f2"]
    (version_dir / "features.json").write_text(_json.dumps(features), encoding="utf-8")

    plain = ExperimentRunner(cfg)
    subset_cfg = ExperimentConfig(
        data=DataConfig(
            version=cfg.data.version, feature_set=cfg.data.feature_set,
            feature_subset="sunshine", targets=cfg.data.targets,
            data_dir=cfg.data.data_dir,
        ),
        split=cfg.split, model=cfg.model, evaluation=cfg.evaluation, run=cfg.run,
    )
    subset = ExperimentRunner(subset_cfg)
    assert plain._run_id != subset._run_id
    assert subset.run(deploy=False).manifest["feature_cols"] == ["f1", "f2"]
