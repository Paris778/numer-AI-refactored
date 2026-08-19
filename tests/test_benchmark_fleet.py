"""Untiered benchmark fleet: config schema, generators, runner, placement."""

from __future__ import annotations

from pathlib import Path

import pytest

from nmr.benchmark_fleet import (
    load_fleet_config,
    load_fleet_suite_config,
)


def _write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "fleet.yaml"
    path.write_text(text, encoding="utf-8")
    return path


MINIMAL = """
cells:
  - benchmark_id: silly_target_lag_mean
    source: docs/05-notebooks
    input_space: none
    model_kind: target_lag_mean
    targets: [target]
    params: {window: 1}
"""


def test_minimal_cell_roundtrip(tmp_path):
    cfg = load_fleet_config(_write_yaml(tmp_path, MINIMAL))
    assert len(cfg.cells) == 1
    cell = cfg.cells[0]
    assert cell.benchmark_id == "silly_target_lag_mean"
    assert cell.input_space == "none"
    assert cell.model_kind == "target_lag_mean"
    assert cell.targets == ("target",)
    assert cell.params["window"] == 1
    assert cell.seed == 42
    assert cell.neutralization is None


def test_unknown_cell_key_rejected(tmp_path):
    text = MINIMAL.replace("    params: {window: 1}",
                           "    params: {window: 1}\n    bogus_key: 1")
    with pytest.raises(ValueError, match="Unknown FleetCellConfig keys"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_tier_key_is_rejected(tmp_path):
    text = MINIMAL.replace("    model_kind: target_lag_mean",
                           "    tier: 3\n    model_kind: target_lag_mean")
    with pytest.raises(ValueError, match="Unknown FleetCellConfig keys"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_invalid_kind_rejected(tmp_path):
    text = MINIMAL.replace("target_lag_mean", "null_constant_05")
    with pytest.raises(ValueError, match="model_kind"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_invalid_neutralization_rejected(tmp_path):
    text = MINIMAL.replace(
        "    params: {window: 1}",
        "    params: {window: 1}\n    neutralization: 0.6",
    )
    with pytest.raises(ValueError, match="neutralization"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_neutralization_none_parses(tmp_path):
    text = MINIMAL.replace(
        "    params: {window: 1}",
        "    params: {window: 1}\n    neutralization: none",
    )
    cell = load_fleet_config(_write_yaml(tmp_path, text)).cells[0]
    assert cell.neutralization is None


def test_lag_mean_requires_none_input_space(tmp_path):
    text = MINIMAL.replace("input_space: none", "input_space: small")
    with pytest.raises(ValueError, match="target_lag_mean requires"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_lag_mean_requires_single_target(tmp_path):
    text = MINIMAL.replace("targets: [target]", "targets: [target, target_ender_20]")
    with pytest.raises(ValueError, match="single target"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_ridge_stack_requires_main_target_and_specialists(tmp_path):
    text = """
cells:
  - benchmark_id: rs
    source: legacy
    input_space: small
    model_kind: ridge_stack
    params: {mode: fixed, alpha: 1e-6}
"""
    with pytest.raises(ValueError, match="ridge_stack requires"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_suite_config_aggregates_and_dedupes(tmp_path):
    (tmp_path / "a.yaml").write_text(MINIMAL, encoding="utf-8")
    (tmp_path / "b.yaml").write_text(
        MINIMAL.replace("silly_target_lag_mean", "other_cell"), encoding="utf-8"
    )
    cells = load_fleet_suite_config(tmp_path)
    assert [c.benchmark_id for c in cells] == ["other_cell", "silly_target_lag_mean"]
    (tmp_path / "b.yaml").write_text(MINIMAL, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate benchmark ids"):
        load_fleet_suite_config(tmp_path)
