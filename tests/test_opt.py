# tests/test_opt.py
from __future__ import annotations

import json

import numpy as np
import optuna
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
        ({"a": {"kind": "float", "low": 1.0, "high": 2.0, "log": "yes"}}, "log"),
        ({"a": {"kind": "int", "low": 1, "high": 10, "log": 0}}, "log"),
        ({"a": {"kind": "float", "low": 1.0, "high": 2.0, "step": 2}}, "step"),
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
            f1 = 0.35 + 0.12 * np.sin(0.3 * era) + 0.02 * idx
            f2 = 0.10 + 0.08 * np.cos(0.25 * era) + 0.015 * idx
            train_rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": 0.6 * f1 - 0.3 * f2 + (0.01 + 0.02 * (era % 3)) * np.sin(idx),
                    "target_alt": 0.2 * f1 + 0.7 * f2 - 0.03 * np.cos(idx),
                }
            )
    pl.DataFrame(train_rows).write_parquet(version_dir / "train.parquet")

    # Validation/meta/benchmark frames: unused by _held_out_metric (it loads only
    # "train"), kept for parity with the test_runner vtest layout. Features are
    # bounded periodic functions of era, so val features evaluated at
    # (era - (n_train_eras - 1)) stay inside the train feature envelope. The
    # (era % 3)-scaled sin(idx) target term makes the per-era held-out CORR vary
    # across eras — a constant per-era CORR would vacuously pass the
    # corr_sharpe_ac end-to-end test via the std==0 short-circuit.
    val_rows = []
    shift = n_train_eras - 1
    for era in range(n_train_eras + 1, n_train_eras + 7):
        for idx in range(6):
            f1 = 0.35 + 0.12 * np.sin(0.3 * (era - shift)) + 0.02 * idx
            f2 = 0.10 + 0.08 * np.cos(0.25 * (era - shift)) + 0.015 * idx
            val_rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": 0.6 * f1 - 0.3 * f2
                    + (0.01 + 0.02 * ((era - shift) % 3)) * np.sin(idx),
                    "target_alt": 0.2 * f1 + 0.7 * f2 - 0.03 * np.cos(idx),
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


def test_bayesian_sweep_supports_corr_sharpe_ac_metric(tmp_path, monkeypatch) -> None:
    from nmr import research
    from nmr.opt import bayesian_sweep

    captured: dict[str, dict[str, float]] = {}
    orig = research._per_era_ac_sharpe

    def recording(per_era, *, horizon="20D"):
        captured["per_era"] = per_era
        return orig(per_era, horizon=horizon)

    monkeypatch.setattr(research, "_per_era_ac_sharpe", recording)
    # 30 train eras -> held-out = round(0.2*30) = 6 eras; the 20D AC bandwidth
    # floor (4) needs >= 5 eras, so the default 12-era fixture would raise.
    cfg = _sweep_config(tmp_path, n_train_eras=30)
    space = {"learning_rate": {"kind": "float", "low": 0.01, "high": 0.1, "log": True}}
    result = bayesian_sweep(cfg, space, n_trials=2, seed=7, metric="corr_sharpe_ac")
    assert result.trials.get_column("metric").to_list() == ["corr_sharpe_ac"] * 2
    values = result.trials.get_column("metric_value")
    assert values.is_finite().all()
    assert (values != 0.0).all()          # every trial took the real AC path
    series = np.asarray(list(captured["per_era"].values()), dtype=float)
    assert series.size >= 5               # 20D bandwidth floor: >= 5 held-out eras
    assert np.std(series) > 0.0           # per-era corr genuinely varies


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


def test_bayesian_sweep_disables_baseline_anchor(tmp_path, monkeypatch) -> None:
    # Spec contract: with enqueue_base_config=False the resolved baseline is NOT
    # enqueued — trial 0 must be TPE-suggested, not the anchor. Guards the
    # `if enqueue_base_config:` wiring (a regression here would silently make
    # every sweep trial 0 the baseline again).
    from nmr.opt import bayesian_sweep
    from nmr.models import resolve_model_params

    cfg = _sweep_config(tmp_path)
    # Baseline num_leaves (fast preset = 31) lies outside this space, so any
    # enqueued anchor would be visible in trial 0; a TPE-suggested trial 0 can
    # never equal 31.
    space = {"num_leaves": {"kind": "int", "low": 4, "high": 8}}
    enqueue_calls: list[dict] = []
    orig_enqueue = optuna.Study.enqueue_trial

    def recording(self, params, *args, **kwargs):
        enqueue_calls.append(dict(params))
        orig_enqueue(self, params, *args, **kwargs)

    monkeypatch.setattr(optuna.Study, "enqueue_trial", recording)
    result = bayesian_sweep(cfg, space, n_trials=3, seed=7, n_startup_trials=2,
                            enqueue_base_config=False)
    assert enqueue_calls == []                     # anchor never enqueued
    resolved = resolve_model_params(cfg.model.preset, cfg.model.params)
    trial0 = json.loads(
        result.trials.filter(pl.col("trial_id") == 0).get_column("params_json")[0]
    )
    assert trial0["num_leaves"] != resolved["num_leaves"]   # not the baseline


def test_bayesian_sweep_sampled_params_respect_space(tmp_path) -> None:
    # Spec contract: every SUGGESTED trial param must lie inside the space —
    # floats within [low, high], ints within [low, high] and on the step
    # lattice, categoricals within choices. enqueue_base_config=False keeps
    # every trial sampled (an anchored baseline can legitimately fall off the
    # step lattice, so it is excluded here).
    from nmr.opt import bayesian_sweep

    cfg = _sweep_config(tmp_path)
    space = {
        "learning_rate": {"kind": "float", "low": 0.01, "high": 0.1, "log": True},
        "num_leaves": {"kind": "int", "low": 4, "high": 32, "step": 4},
        "boosting": {"kind": "categorical", "choices": ["gbdt", "dart"]},
    }
    result = bayesian_sweep(cfg, space, n_trials=8, seed=7, n_startup_trials=2,
                            enqueue_base_config=False)
    for row in result.trials.get_column("params_json"):
        params = json.loads(row)
        assert 0.01 <= params["learning_rate"] <= 0.1
        num_leaves = params["num_leaves"]
        assert 4 <= num_leaves <= 32
        assert (num_leaves - 4) % 4 == 0           # step=4 respected
        assert params["boosting"] in ("gbdt", "dart")


def test_bayesian_sweep_forwards_n_startup_trials_to_sampler(
    tmp_path, monkeypatch
) -> None:
    # Pins the wiring between bayesian_sweep's n_startup_trials parameter and
    # TPESampler: a no-op here (silent default of 10) violates AGENTS §2.5
    # fail-loud/no-hidden-defaults. Wrapper delegates to the real __init__ so
    # the sampler still works; plain pytest monkeypatch, no unittest.mock.
    from nmr.opt import bayesian_sweep

    seen: dict[str, int] = {}
    orig_init = optuna.samplers.TPESampler.__init__

    def recording_init(self, *args, **kwargs):
        seen["n_startup_trials"] = kwargs.get("n_startup_trials")
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(optuna.samplers.TPESampler, "__init__", recording_init)
    cfg = _sweep_config(tmp_path)
    bayesian_sweep(
        cfg,
        {"num_leaves": {"kind": "int", "low": 4, "high": 32}},
        n_trials=1,
        seed=7,
        n_startup_trials=2,
    )
    assert seen["n_startup_trials"] == 2

def test_hyperparameter_sweep_trials_carry_held_out_moments(tmp_path) -> None:
    from nmr.research import HyperparameterSweep

    cfg = _sweep_config(tmp_path)
    result = HyperparameterSweep(cfg, metric="sharpe").run(
        {"n_estimators": [10, 12]}, n_trials=2, seed=3
    )
    for col in ("ic_sharpe", "ic_skew", "ic_kurt", "ic_n_eras", "ic_std"):
        assert col in result.trials.columns, col
    assert result.trials["ic_n_eras"].min() >= 1
    assert result.trials["ic_std"].min() > 0.0

def test_sweep_dsr_computes_fleet_deflation() -> None:
    from nmr.inference import deflated_sharpe
    from nmr.opt import sweep_dsr

    trials = pl.DataFrame({
        "trial_id": [0, 1, 2],
        "metric_value": [0.4, 0.6, 0.5],
        "ic_sharpe": [0.4, 0.6, 0.5],
        "ic_skew": [0.0, 0.1, -0.1],
        "ic_kurt": [3.0, 3.2, 2.9],
        "ic_n_eras": [600, 649, 620],
        "ic_std": [0.1, 0.1, 0.1],
        "metric": ["sharpe", "sharpe", "sharpe"],
    })
    out = sweep_dsr(trials)
    var = np.var([0.4, 0.6, 0.5], ddof=1)
    expected = deflated_sharpe(
        0.4, n_trials=3, n_obs=600, skew=0.0, kurt=3.0, trials_sr_var=var
    )
    assert out["dsr_sweep_aware"][0] == pytest.approx(expected)
    assert out["dsr_pass_sweep"].dtype == pl.Boolean
    assert "dsr_reason" in out.columns
    assert out["dsr_n_trials"][0] == 3

def test_sweep_dsr_zero_variance_guard() -> None:
    from nmr.opt import sweep_dsr

    trials = pl.DataFrame({
        "trial_id": [0, 1, 2],
        "metric_value": [0.5, 0.5, 0.5],
        "ic_sharpe": [0.5, 0.5, 0.5],
        "ic_skew": [0.0, 0.0, 0.0],
        "ic_kurt": [3.0, 3.0, 3.0],
        "ic_n_eras": [600, 600, 600],
        "ic_std": [0.1, 0.1, 0.1],
        "metric": ["sharpe"] * 3,
    })
    out = sweep_dsr(trials)
    assert out["dsr_sweep_aware"].null_count() == 3
    assert (out["dsr_reason"] == "zero_cross_trial_sharpe_variance").all()

def test_bayesian_sweep_trials_carry_held_out_moments(tmp_path) -> None:
    from nmr.opt import bayesian_sweep

    cfg = _sweep_config(tmp_path)
    result = bayesian_sweep(
        cfg, {"num_leaves": {"kind": "int", "low": 8, "high": 16}},
        n_trials=2, seed=7, n_startup_trials=1,
    )
    for col in ("ic_sharpe", "ic_skew", "ic_kurt", "ic_n_eras", "ic_std"):
        assert col in result.trials.columns, col
