"""Config-layer tests for the 5-tier benchmark hierarchy."""

from __future__ import annotations

from pathlib import Path

import pytest

from nmr.benchmark import (
    VALID_BENCHMARK_TIERS,
    load_benchmark_file,
    load_benchmark_suite_config,
)
from nmr.config import REPO_ROOT

BENCHMARK_CONFIG_DIR = REPO_ROOT / "configs" / "benchmarks"


def _write_yaml(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_all_shipped_config_files_load() -> None:
    spec = load_benchmark_suite_config(BENCHMARK_CONFIG_DIR)
    assert {cell.benchmark_id for cell in spec.cells} == {
        "null_constant_05",
        "null_uniform_rand",
        "null_gaussian_rand",
        "null_feature_mean",
        "linear_ridge_small",
        "linear_ridge_medium",
        "linear_ridge_multitarget",
        "tree_lgbm_shallow_small",
        "tree_xgb_shallow_medium",
        "tree_lgbm_fast_medium",
        "canon_hello_numerai",
        "canon_neutralized_50",
        "canon_sunshine_ensemble",
    }
    assert spec.reference_column == "v53_lgbm_ender60"
    assert spec.gate is not None
    assert spec.gate.corr_min == 0.0286
    tiers = [cell.tier for cell in spec.cells]
    assert tiers == sorted(tiers)
    assert set(tiers) == set(VALID_BENCHMARK_TIERS[:4])


def test_tier4_gate_thresholds() -> None:
    spec = load_benchmark_suite_config(BENCHMARK_CONFIG_DIR)
    gate = spec.gate
    assert gate is not None
    assert gate.corr_min == 0.0286
    assert gate.corr_sharpe_ac_min == 0.78
    assert gate.fnc_min == 0.020
    assert gate.deflated_sharpe_min == 0.95
    assert gate.gain_to_pain_min == 1.50
    assert gate.cagr_min == 0.0
    assert gate.turnover_max == 0.35


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
tier: 0
cells:
  - benchmark_id: null_constant_05
    input_space: none
    model_kind: null_constant_05
    bogus_key: 1
""",
    )
    with pytest.raises(ValueError, match="[Uu]nknown"):
        load_benchmark_file(path)


def test_invalid_enum_values_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
tier: 9
cells:
  - benchmark_id: x
    input_space: none
    model_kind: null_constant_05
""",
    )
    with pytest.raises(ValueError, match="tier"):
        load_benchmark_file(path)


def test_tier0_input_space_validation(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
tier: 0
cells:
  - benchmark_id: null_feature_mean
    input_space: medium
    model_kind: null_feature_mean
""",
    )
    with pytest.raises(ValueError, match="null_feature_mean"):
        load_benchmark_file(path)


def test_neutralization_fraction_validated(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
tier: 3
cells:
  - benchmark_id: canon_neutralized_50
    input_space: medium
    model_kind: lightgbm
    neutralization: 1.5
""",
    )
    with pytest.raises(ValueError, match="neutralization"):
        load_benchmark_file(path)


def test_tier4_file_requires_gate(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
tier: 4
reference_column: v53_lgbm_ender60
""",
    )
    with pytest.raises(ValueError, match="gate"):
        load_benchmark_file(path)


def test_duplicate_benchmark_ids_rejected(tmp_path: Path) -> None:
    d = tmp_path
    _write_yaml(
        d,
        "a.yaml",
        """
tier: 0
cells:
  - benchmark_id: dup
    input_space: none
    model_kind: null_constant_05
""",
    )
    _write_yaml(
        d,
        "b.yaml",
        """
tier: 1
cells:
  - benchmark_id: dup
    input_space: small
    model_kind: ridge
""",
    )
    with pytest.raises(ValueError, match="dup"):
        load_benchmark_suite_config(d)
