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
    assert cfg.data.version == "v5.2"
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
        ModelConfig(backend="catboost")


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
    assert "v5.2" in str(path)
    assert path.is_relative_to(REPO_ROOT)


def test_seed_determinism():
    set_global_seeds(123)
    first = (random.random(), float(np.random.rand()))
    set_global_seeds(123)
    second = (random.random(), float(np.random.rand()))
    assert first == second
