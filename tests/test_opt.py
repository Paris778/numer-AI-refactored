# tests/test_opt.py
from __future__ import annotations

import json

import polars as pl
import pytest

from nmr.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    SplitConfig,
)
from nmr.opt import _SpaceParam, _parse_space


def test_parse_space_accepts_all_kinds() -> None:
    space = {
        "learning_rate": {"kind": "float", "low": 0.005, "high": 0.05, "log": True},
        "n_estimators": {"kind": "int", "low": 100, "high": 10000, "log": True},
        "num_leaves": {"kind": "int", "low": 16, "high": 256},
        "boosting": {"kind": "categorical", "choices": ["gbdt", "dart"]},
    }
    parsed = {p.name: p for p in _parse_space(space)}
    assert parsed["learning_rate"].kind == "float"
    assert parsed["learning_rate"].log is True
    assert parsed["n_estimators"].log is True
    assert parsed["n_estimators"].step is None
    assert parsed["num_leaves"].step is None
    assert parsed["boosting"].choices == ["gbdt", "dart"]


def test_parse_space_int_step() -> None:
    parsed = {p.name: p for p in _parse_space(
        {"n_estimators": {"kind": "int", "low": 100, "high": 10000, "step": 100}})}
    assert parsed["n_estimators"].step == 100


@pytest.mark.parametrize(
    "space, match",
    [
        ({}, "empty"),
        ({"a": {"kind": "bogus", "low": 0, "high": 1}}, "kind"),
        ({"a": {"kind": "float", "low": 1.0, "high": 0.5}}, "low"),
        ({"a": {"kind": "float", "low": 0.0, "high": 0.1, "log": True}}, "low"),
        ({"a": {"kind": "int", "low": 1, "high": 10, "log": True, "step": 2}}, "step"),
        ({"a": {"kind": "int", "low": 1, "high": 10, "step": 0}}, "step"),
        ({"a": {"kind": "categorical", "choices": []}}, "choices"),
        ({"a": {"kind": "categorical", "choices": [(1, 2)]}}, "choices"),
        ({"a": {"kind": "categorical", "choices": [None]}}, "choices"),
        ({"a": {"kind": "float", "low": 1.0, "high": 2.0, "bogus": 1}}, "unknown"),
        ({"a": "not-a-dict"}, "spec"),
        ({"a": {"kind": "float", "high": 1.0}}, "low"),
        ({"a": {"kind": "int", "low": 1}}, "low"),
        ({"a": {"kind": "int", "low": 0, "high": 10, "log": True}}, "low"),
        ({"a": {"kind": "int", "low": 1, "high": 10, "step": 2.5}}, "step"),
    ],
)
def test_parse_space_validation_errors(space, match) -> None:
    with pytest.raises(ValueError, match=match):
        _parse_space(space)


# --- Task 5: bayesian_sweep. ``bayesian_sweep`` is imported function-locally so
# the Task 4 parse_space tests above keep passing during the RED phase
# (``from nmr.opt import bayesian_sweep`` at module level would abort collection
# of the whole file before the symbol exists). ---


def _sweep_config(tmp_path, *, n_train_eras: int = 12):
    """Synthetic-data builder identical to tests/test_runner.py's vtest pattern
    (small frames, fast preset, n_estimators=10, n_folds=2), parameterized by the
    number of TRAIN eras. Held-out = round(0.2 * n) eras; ``corr_sharpe_ac`` needs
    >= 5 held-out eras (the 20D AC bandwidth floor is 4, cap n-1), so the
    corr_sharpe_ac test uses n_train_eras=30 (held-out=6). Inlined locally — do
    NOT import across test modules (tests/ is not a package)."""
    from nmr.config import (
        DataConfig,
        EvalConfig,
        ExperimentConfig,
        ModelConfig,
        RunConfig,
        SplitConfig,
    )

    data_root = tmp_path / "data"
    version_dir = data_root / "vtest"
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

    train_rows = []
    for era in range(1, n_train_eras + 1):
        for idx in range(6):
            f1 = (era * 0.03) + (idx * 0.02)
            f2 = (era * -0.02) + (idx * 0.01)
            train_rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": 0.6 * f1 - 0.3 * f2 + 0.05 * era,
                    "target_alt": 0.2 * f1 + 0.7 * f2 - 0.04 * era,
                }
            )
    pl.DataFrame(train_rows).write_parquet(version_dir / "train.parquet")

    # Validation/meta/benchmark frames: unused by _held_out_metric (it loads only
    # "train"), kept for parity with the test_runner vtest layout. Features are
    # shifted back into the training envelope ((era - (n_train_eras - 1)) maps the
    # first validation era to train-era-2 feature values, matching test_runner).
    val_rows = []
    shift = n_train_eras - 1
    for era in range(n_train_eras + 1, n_train_eras + 7):
        for idx in range(6):
            f1 = ((era - shift) * 0.03) + (idx * 0.02)
            f2 = ((era - shift) * -0.02) + (idx * 0.01)
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
            metrics=("corr", "fnc", "sharpe"),
            validation_scorecard=False,
        ),
        run=RunConfig(
            seed=17, artifacts_dir=tmp_path / "artifacts", name="opt-test"
        ),
    )


def test_bayesian_sweep_is_deterministic_under_seed(tmp_path) -> None:
    from nmr.opt import bayesian_sweep

    cfg = _sweep_config(tmp_path)
    space = {
        "learning_rate": {"kind": "float", "low": 0.01, "high": 0.1, "log": True},
        "num_leaves": {"kind": "int", "low": 4, "high": 32},
    }
    first = bayesian_sweep(cfg, space, n_trials=4, seed=7, n_startup_trials=2)
    second = bayesian_sweep(cfg, space, n_trials=4, seed=7, n_startup_trials=2)
    assert first.trials.equals(second.trials)
    assert first.best_params == second.best_params
    assert first.best_value == second.best_value


def test_bayesian_sweep_anchors_baseline_as_trial_zero(tmp_path) -> None:
    from nmr.opt import bayesian_sweep

    cfg = _sweep_config(tmp_path)
    space = {
        "learning_rate": {"kind": "float", "low": 0.01, "high": 0.1, "log": True},
        "n_estimators": {"kind": "int", "low": 10, "high": 20},
    }
    result = bayesian_sweep(cfg, space, n_trials=3, seed=7, n_startup_trials=2)
    trial0 = json.loads(
        result.trials.filter(pl.col("trial_id") == 0).get_column("params_json")[0]
    )
    from nmr.models import resolve_model_params

    resolved = resolve_model_params(cfg.model.preset, cfg.model.params)
    for key in ("learning_rate", "n_estimators"):
        if key in resolved:
            assert trial0[key] == resolved[key]
    # n_estimators=10 preset/override is in the space and must be anchored:
    assert trial0["n_estimators"] == 10


def test_bayesian_sweep_rejects_parallel_trials(tmp_path) -> None:
    from nmr.opt import bayesian_sweep

    cfg = _sweep_config(tmp_path)
    with pytest.raises(ValueError, match="n_jobs"):
        bayesian_sweep(cfg, {"num_leaves": {"kind": "int", "low": 4, "high": 32}},
                       n_trials=2, seed=7, n_jobs=2)


def test_bayesian_sweep_supports_corr_sharpe_ac_metric(tmp_path) -> None:
    from nmr.opt import bayesian_sweep

    # 30 train eras -> held-out = round(0.2*30) = 6 eras; the 20D AC bandwidth
    # floor (4) needs >= 5 eras, so the default 12-era fixture would raise.
    cfg = _sweep_config(tmp_path, n_train_eras=30)
    space = {"learning_rate": {"kind": "float", "low": 0.01, "high": 0.1, "log": True}}
    result = bayesian_sweep(cfg, space, n_trials=2, seed=7, metric="corr_sharpe_ac")
    assert result.trials.get_column("metric").to_list() == ["corr_sharpe_ac"] * 2
    assert result.trials.get_column("metric_value").is_finite().all()


def test_bayesian_sweep_failed_trial_recorded_and_continues(tmp_path) -> None:
    from nmr.opt import bayesian_sweep

    cfg = _sweep_config(tmp_path)
    # num_leaves < 0 makes LightGBM raise inside _held_out_metric -> TrialPruned.
    # enqueue_base_config=False is REQUIRED here: the resolved baseline
    # (num_leaves=31) lies outside the [-8, -1] space, and Optuna 4.9's
    # fixed-param path returns out-of-range enqueued values with only a warning
    # (probe-verified), letting the anchored trial 0 SUCCEED. Disabling the
    # anchor keeps every trial inside the invalid range.
    space = {"num_leaves": {"kind": "int", "low": -8, "high": -1}}
    result = bayesian_sweep(cfg, space, n_trials=3, seed=7, n_startup_trials=2,
                            enqueue_base_config=False)
    assert result.trials.height == 3                     # synchronized with study.trials
    assert result.trials.get_column("metric_value").null_count() == 3
    assert result.best_params == {} or result.best_value is not None  # best may be empty


def test_bayesian_sweep_metrics_reject_unknown(tmp_path) -> None:
    from nmr.opt import bayesian_sweep

    cfg = _sweep_config(tmp_path)
    with pytest.raises(ValueError, match="metric"):
        bayesian_sweep(cfg, {"num_leaves": {"kind": "int", "low": 4, "high": 32}},
                       n_trials=2, seed=7, metric="corr")
