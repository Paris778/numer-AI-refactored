"""Tests for nmr.config: loading, defaults, validation, paths, and determinism."""

from __future__ import annotations

import random

import numpy as np
import pytest

from nmr.config import (
    REPO_ROOT,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    SplitConfig,
    load_config,
    set_global_seeds,
)


def test_load_example_config(example_config_path):
    cfg = load_config(example_config_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.data.version == "v5.3"
    assert cfg.data.feature_set == "small"
    assert cfg.data.targets == ("target",)
    assert cfg.split.scheme == "walk_forward"
    assert cfg.split.purge_eras == 8
    assert cfg.model.backend == "lightgbm"
    assert cfg.evaluation.backend == "custom"
    assert cfg.run.seed == 42


def test_defaults_when_empty(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.data.feature_set == "small"
    assert cfg.model.preset == "fast"
    assert cfg.evaluation.metrics == ("corr", "mmc", "fnc", "sharpe")


def test_targets_and_metrics_coerced_to_tuple():
    assert DataConfig(targets=["a", "b"]).targets == ("a", "b")
    assert EvalConfig(metrics=["corr"]).metrics == ("corr",)


def test_invalid_feature_set_raises():
    with pytest.raises(ValueError):
        DataConfig(feature_set="huge")


def test_invalid_model_backend_raises():
    with pytest.raises(ValueError):
        ModelConfig(backend="bogus")


def test_invalid_split_scheme_raises():
    with pytest.raises(ValueError):
        SplitConfig(scheme="kfold")


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("data:\n  feature_set: small\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_unknown_section_raises(tmp_path):
    p = tmp_path / "bad_section.yaml"
    p.write_text("nonsense:\n  foo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_data_path_resolution():
    path = DataConfig().path("train.parquet")
    assert path.name == "train.parquet"
    assert "v5.3" in str(path)
    assert path.is_relative_to(REPO_ROOT)


def test_seed_determinism():
    set_global_seeds(123)
    first = (random.random(), float(np.random.rand()))
    set_global_seeds(123)
    second = (random.random(), float(np.random.rand()))
    assert first == second


def test_risk_section_validates_proportion() -> None:
    from nmr.config import RiskConfig

    assert RiskConfig().neutralization_proportion == 1.0
    assert RiskConfig(neutralization_proportion=0.0).neutralization_proportion == 0.0
    with pytest.raises(ValueError):
        RiskConfig(neutralization_proportion=1.5)


def test_ensemble_section_validates_method() -> None:
    from nmr.config import EnsembleConfig

    assert EnsembleConfig().method == "ridge"
    assert EnsembleConfig(method="non_negative").method == "non_negative"
    with pytest.raises(ValueError):
        EnsembleConfig(method="svm")


def test_set_global_seeds_does_not_touch_hash_env() -> None:
    import os

    os.environ.pop("PYTHONHASHSEED", None)
    set_global_seeds(42)
    assert "PYTHONHASHSEED" not in os.environ


def test_feature_subset_overrides_feature_set_in_resolution():
    from nmr.config import DataConfig

    cfg = DataConfig(feature_set="small", feature_subset="sunshine")
    assert cfg.resolved_feature_set == "sunshine"
    plain = DataConfig(feature_set="small")
    assert plain.resolved_feature_set == "small"


def test_feature_subset_must_be_non_empty_when_provided():
    import pytest as _pytest

    from nmr.config import DataConfig

    with _pytest.raises(ValueError, match="feature_subset"):
        DataConfig(feature_subset="")


def test_catboost_backend_is_valid():
    from nmr.config import ModelConfig

    assert ModelConfig(backend="catboost").backend == "catboost"


def test_invalid_backend_still_raises():
    import pytest as _pytest

    from nmr.config import ModelConfig

    with _pytest.raises(ValueError, match="backend"):
        ModelConfig(backend="bogus")


def test_model_config_device_validation() -> None:
    for device in ("auto", "gpu", "cpu"):
        assert ModelConfig(device=device).device == device
    with pytest.raises(ValueError, match="device"):
        ModelConfig(device="quantum")
    # the default preserves the legacy GPU-first behavior
    assert ModelConfig().device == "auto"


def test_full_history_fit_device_policy() -> None:
    assert ModelConfig().validation_fit_device == "cpu"
    assert ModelConfig().deploy_fit_device == "cpu"
    assert ModelConfig(validation_fit_device="gpu").validation_fit_device == "gpu"
    with pytest.raises(ValueError, match="validation_fit_device"):
        ModelConfig(validation_fit_device="auto")
    with pytest.raises(ValueError, match="deploy_fit_device"):
        ModelConfig(deploy_fit_device="gpu")
    with pytest.raises(ValueError, match="CatBoost is CPU-only"):
        ModelConfig(backend="catboost", validation_fit_device="gpu")


def test_workflow_phase_contracts() -> None:
    from nmr.config import config_from_dict

    base = _cfg_dict(
        data={"horizon": "60D", "targets": ["target_ender_60"]},
        split={"purge_eras": 16},
    )
    base["model"] = {
        "backend": "xgboost",
        "device": "gpu",
        "validation_fit_device": "gpu",
    }
    base["evaluation"] = {
        "main_target": "target_ender_60",
        "validation_scorecard": False,
        "metrics": ["corr", "fnc", "sharpe"],
    }
    base["run"] = {"workflow": "gpu_screen"}
    screen = config_from_dict(base)
    assert screen.run.workflow == "gpu_screen"

    base["run"] = {"workflow": "cpu_confirm"}
    base["model"]["validation_fit_device"] = "cpu"
    base["evaluation"]["validation_scorecard"] = True
    base["evaluation"]["metrics"] = ["corr", "mmc", "fnc", "sharpe"]
    confirm = config_from_dict(base)
    assert confirm.run.workflow == "cpu_confirm"

    base["evaluation"]["validation_scorecard"] = True
    with pytest.raises(ValueError, match="gpu_screen"):
        base["run"] = {"workflow": "gpu_screen"}
        config_from_dict(base)


def test_eval_metrics_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="metrics"):
        EvalConfig(
            metrics=("cor",)
        )  # typo must fail loudly, not silently compute nothing


def test_eval_metrics_accepts_known_names() -> None:
    for names in (("corr",), ("corr", "mmc", "fnc", "sharpe")):
        assert EvalConfig(metrics=names).metrics == tuple(names)


def test_atomic_payout_policy_is_the_explicit_evaluation_default() -> None:
    config = ExperimentConfig()

    assert config.evaluation.payout_policy == "classic_atomic_ender60_r1343_v1"
    assert config.evaluation.main_target == "target"
    assert config.data.targets == ("target",)


def test_unknown_payout_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="payout policy"):
        EvalConfig(payout_policy="future_unverified_policy")


def test_ensemble_main_target_must_be_a_trained_component() -> None:
    from nmr.config import config_from_dict

    with pytest.raises(ValueError, match="main_target"):
        config_from_dict(
            {
                "data": {"targets": ["target_cyrusd_20"], "horizon": "20D"},
                "evaluation": {"main_target": "target_ender_20"},
            }
        )


def _cfg_dict(*, data: dict | None = None, split: dict | None = None) -> dict:
    base: dict = {
        "data": {"feature_set": "small", "targets": ["target"]},
        "split": {"purge_eras": 8},
    }
    base["data"].update(data or {})
    base["split"].update(split or {})
    return base


def test_horizon_default_and_validation() -> None:
    from nmr.config import config_from_dict

    assert DataConfig().horizon == "20D"
    assert config_from_dict(_cfg_dict()).data.horizon == "20D"
    with pytest.raises(ValueError, match="horizon"):
        config_from_dict(_cfg_dict(data={"horizon": "30D"}))


def test_purge_horizon_law_20d_ok_60d_insufficient() -> None:
    from nmr.config import config_from_dict, enforce_purge_horizon_law

    # 20D + purge 8 is the law's minimum — accepted.
    assert config_from_dict(_cfg_dict()).split.purge_eras == 8
    # 60D + purge 8 loads (the floor is data-aware) but a real-size dataset
    # (574 eras) rejects it at run time.
    cfg60 = config_from_dict(_cfg_dict(data={"horizon": "60D"}))
    with pytest.raises(ValueError, match="purge_eras"):
        enforce_purge_horizon_law(574, cfg60)
    # 60D + purge 16 on real data — accepted.
    cfg60ok = config_from_dict(
        _cfg_dict(data={"horizon": "60D"}, split={"purge_eras": 16})
    )
    enforce_purge_horizon_law(574, cfg60ok)
    # Small synthetic datasets are governed by the splitter's geometry.
    enforce_purge_horizon_law(12, config_from_dict(_cfg_dict(split={"purge_eras": 1})))
    # stricter-than-law purges are fine.
    enforce_purge_horizon_law(
        574, config_from_dict(_cfg_dict(split={"purge_eras": 16}))
    )


def test_target_name_horizon_agreement() -> None:
    from nmr.config import config_from_dict

    # target_cyrusd_60 with declared 20D — contradiction, rejected.
    with pytest.raises(ValueError, match="encodes horizon"):
        config_from_dict(
            _cfg_dict(
                data={"horizon": "20D", "targets": ["target", "target_cyrusd_60"]}
            )
        )
    # agreement: 60D target with 60D horizon + purge 16 — accepted.
    cfg = config_from_dict(
        _cfg_dict(
            data={"horizon": "60D", "targets": ["target", "target_cyrusd_60"]},
            split={"purge_eras": 16},
        )
    )
    assert cfg.data.horizon == "60D"
    # un-encoded names impose no constraint.
    assert (
        config_from_dict(_cfg_dict(data={"targets": ["target"]})).data.horizon == "20D"
    )


def test_embargo_eras_must_be_zero() -> None:
    """A2 (audit SEV-3): embargo_eras was an inert, documented knob — now
    rejected at load; purge_eras is the active leakage buffer."""
    assert SplitConfig().embargo_eras == 0
    with pytest.raises(ValueError, match="embargo_eras"):
        SplitConfig(embargo_eras=4)
    from nmr.config import config_from_dict

    with pytest.raises(ValueError, match="embargo_eras"):
        config_from_dict(_cfg_dict(split={"embargo_eras": 4}))


def test_catboost_quick_ender60_config_loads():
    cfg = load_config(REPO_ROOT / "configs" / "catboost-quick-ender60.yaml")
    assert cfg.data.feature_set == "medium"
    assert cfg.data.targets == ("target_ender_60",)
    assert cfg.data.horizon == "60D"
    assert cfg.split.purge_eras == 16
    assert cfg.model.backend == "catboost"
    assert cfg.model.preset == "fast"
    assert cfg.model.device == "cpu"
    assert cfg.model.params["iterations"] == 300
    assert cfg.model.params["depth"] == 3
    assert cfg.model.params["rsm"] == 1.0
    assert cfg.evaluation.main_target == "target_ender_60"
    assert cfg.evaluation.validation_scorecard is True


def test_catboost_ender60_fast_config_loads():
    cfg = load_config(REPO_ROOT / "configs" / "catboost-ender60-fast.yaml")
    assert cfg.run.name == "catboost-ender60-fast"
    assert cfg.data.feature_set == "medium"
    assert cfg.data.targets == ("target_ender_60",)
    assert cfg.data.horizon == "60D"
    assert cfg.split.purge_eras == 16
    assert cfg.model.backend == "catboost"
    assert cfg.model.preset == "fast"
    assert cfg.model.device == "cpu"
    assert cfg.model.params["iterations"] == 2000
    assert cfg.model.params["depth"] == 5
    assert cfg.model.params["rsm"] == 1.0
    assert cfg.evaluation.main_target == "target_ender_60"
    assert cfg.evaluation.payout_policy == "classic_atomic_ender60_r1343_v1"
    assert cfg.evaluation.validation_scorecard is True


@pytest.mark.parametrize(
    ("filename", "expected_targets", "expected_preset", "expected_estimators"),
    [
        (
            "xgb-gpu-calibration-ender60.yaml",
            ("target_ender_60",),
            "fast",
            50,
        ),
        (
            "xgb-gpu-ender60-standard.yaml",
            ("target_ender_60",),
            "standard",
            20_000,
        ),
        (
            "xgb-gpu-60d-ensemble-standard.yaml",
            (
                "target_ender_60",
                "target_cyrusd_60",
                "target_sam_60",
                "target_teager2b_60",
            ),
            "standard",
            5_000,
        ),
    ],
)
def test_gpu_xgboost_campaign_configs_are_gate_aligned(
    filename: str,
    expected_targets: tuple[str, ...],
    expected_preset: str,
    expected_estimators: int,
) -> None:
    cfg = load_config(REPO_ROOT / "configs" / filename)

    assert cfg.data.feature_set == "small"
    assert cfg.data.feature_set != "all"
    assert cfg.data.targets == expected_targets
    assert cfg.data.horizon == "60D"
    assert cfg.split.purge_eras == 16
    assert cfg.split.n_folds in (2, 4)
    assert cfg.model.backend == "xgboost"
    assert cfg.model.device == "gpu"
    assert cfg.model.validation_fit_device == "gpu"
    assert cfg.model.deploy_fit_device == "cpu"
    assert cfg.model.preset == expected_preset
    assert cfg.model.params["n_estimators"] == expected_estimators
    assert cfg.model.params["colsample_bytree"] == 0.25
    assert cfg.model.params["subsample"] == 0.8
    assert cfg.evaluation.main_target == "target_ender_60"
    assert cfg.evaluation.validation_scorecard is True
    assert cfg.evaluation.payout_policy == "classic_atomic_ender60_r1343_v1"


@pytest.mark.parametrize(
    ("screen_name", "confirm_name"),
    [
        (
            "xgb-gpu-screen-ender60.yaml",
            "xgb-cpu-confirm-ender60.yaml",
        ),
        (
            "xgb-gpu-screen-60d-ensemble.yaml",
            "xgb-cpu-confirm-60d-ensemble.yaml",
        ),
    ],
)
def test_gpu_screen_and_cpu_confirmation_configs_form_pairs(
    screen_name: str, confirm_name: str
) -> None:
    screen = load_config(REPO_ROOT / "configs" / screen_name)
    confirm = load_config(REPO_ROOT / "configs" / confirm_name)

    assert screen.data == confirm.data
    assert screen.split == confirm.split
    assert screen.model.backend == confirm.model.backend
    assert screen.model.preset == confirm.model.preset
    assert screen.model.params == confirm.model.params
    assert screen.evaluation.backend == confirm.evaluation.backend
    assert screen.evaluation.main_target == confirm.evaluation.main_target
    assert screen.evaluation.payout_policy == confirm.evaluation.payout_policy
    assert screen.ensemble == confirm.ensemble
    assert screen.risk == confirm.risk
    assert screen.run.workflow == "gpu_screen"
    assert confirm.run.workflow == "cpu_confirm"
    assert screen.model.device == confirm.model.device == "gpu"
    assert screen.evaluation.validation_scorecard is False
    assert confirm.evaluation.validation_scorecard is True
    assert screen.model.validation_fit_device == "gpu"
    assert confirm.model.validation_fit_device == "cpu"
    assert screen.model.deploy_fit_device == confirm.model.deploy_fit_device == "cpu"
