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
