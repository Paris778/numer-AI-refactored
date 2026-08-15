# 5-Tier Benchmark Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing S11 benchmark ladder with a deterministic, config-driven 5-tier escalating benchmark hierarchy (null → ridge → trees → canonical → production gate) in `nmr/benchmark.py`, `benchmark_runner.py`, and `tests/test_benchmark_*.py`.

**Architecture:** `nmr/benchmark.py` becomes a `BenchmarkHierarchy` engine: YAML configs under `configs/benchmarks/` (validated by a dedicated frozen-dataclass loader) declare benchmark cells per tier; per-tier prediction generators produce `[era, id, prediction]` frames; every cell is scored through the existing `evaluate_model` pipeline; three gate functions (Tier-0 null floor, Tier-4 thresholds, tier monotonicity) produce hard verdicts. The runner orchestrates all tiers and exits non-zero on gate violation.

**Tech Stack:** Python 3.11, Polars, NumPy/SciPy, scikit-learn (Ridge), LightGBM/XGBoost, `nmr` internals (`evaluate_model`, `Ensembler`, `NeutralizationEngine`, `_colsample_floor`).

**Design spec:** `docs/superpowers/specs/2026-08-15-benchmark-hierarchy-design.md` (approved). All decisions and thresholds come from there.

## Global Constraints

- **Determinism is sacred:** every stochastic operation seeded from config (`seed`); LGBM/XGBoost use `n_jobs=1` + fixed `random_state`; canonical hashes exclude wall-clock timing and absolute paths.
- **Leakage is a correctness bug:** Tier 1–3 fits use the purged train→validation split (exact 8-era buffer, strict ordering); never weaken assertions.
- **No fabricated numbers:** pytest enforces gate *mechanics* on synthetic fixtures only; absolute thresholds are config values evaluated at runtime on real data.
- **`nmr/` is the only tested boundary:** `benchmark_runner.py` is a thin control plane — no formulas.
- **No hidden defaults / no magic values:** closed sets as module-level tuples; `ValueError` on unknown keys, invalid enums, degenerate inputs.
- **SSOT same-commit:** any change to benchmark behavior updates `docs/06-evaluation/benchmark-line-in-the-sand.md`, `ARCHITECTURE.md`, and `AGENTS.md` in the same commit (self-update directive).
- **Windows venv:** run Python as `./.venv/Scripts/python` (never the pip shim); test command: `./.venv/Scripts/python -m pytest <path> -q`.
- **Dirty working tree:** 33 unrelated files are already modified. Every `git add` in this plan lists files explicitly — never `git add -A` or `git add .`.
- **Watchpoints (audit-mandated):** (1) per-target NaN masking in multi-target Ridge; (2) zero-variance standardization outputs 0.0; (3) monotonicity `atol = 1e-5`.
- **Real-data smoke:** `benchmark_runner.py --fast-mode` is the pre-sign-off gate; run it as a background task with a log file.

---

### Task 1: Benchmark config schema + loader + YAML inventory

**Files:**
- Create: `configs/benchmarks/tier0_null.yaml`
- Create: `configs/benchmarks/tier1_ridge_small.yaml`
- Create: `configs/benchmarks/tier1_ridge_medium.yaml`
- Create: `configs/benchmarks/tier1_ridge_multitarget.yaml`
- Create: `configs/benchmarks/tier2_tree_shallow.yaml`
- Create: `configs/benchmarks/tier2_tree_fast.yaml`
- Create: `configs/benchmarks/tier3_sunshine.yaml`
- Create: `configs/benchmarks/tier4_gate.yaml`
- Modify: `nmr/benchmark.py` (append new section; leave all legacy code intact — old tests still pass)
- Test: `tests/test_benchmark_config.py`

**Interfaces:**
- Produces (used by Tasks 2–9):
  - `VALID_BENCHMARK_TIERS: tuple[int, ...] = (0, 1, 2, 3, 4)`
  - `VALID_INPUT_SPACES: tuple[str, ...] = ("none", "small", "medium")`
  - `VALID_BENCHMARK_MODEL_KINDS: tuple[str, ...] = ("null_constant_05", "null_uniform_rand", "null_gaussian_rand", "null_feature_mean", "ridge", "lightgbm", "xgboost")`
  - `DEFAULT_BENCHMARK_SEED: int = 42`
  - `DEFAULT_BENCHMARK_PURGE_ERAS: int = 8`
  - `Tier4GateConfig` (frozen dataclass): `corr_min: float`, `corr_sharpe_ac_min: float`, `fnc_min: float`, `deflated_sharpe_min: float`, `gain_to_pain_min: float`, `cagr_min: float`, `turnover_max: float`
  - `BenchmarkCellConfig` (frozen dataclass): `benchmark_id: str`, `input_space: str`, `model_kind: str`, `tier: int`, `targets: tuple[str, ...] = ("target",)`, `params: Mapping[str, Any]` (frozen dict), `seed: int = DEFAULT_BENCHMARK_SEED`, `neutralization: float = 0.0`, `anchors: Mapping[str, float] | None = None`, `fast_mode_params: Mapping[str, Any] | None = None` — `tier` is injected by the loader from the file-level `tier` (an explicitly conflicting `tier` inside a cell raises)
  - `BenchmarkFileConfig` (frozen dataclass): `tier: int`, `cells: tuple[BenchmarkCellConfig, ...]`, `reference_column: str | None = None`, `gate: Tier4GateConfig | None = None`
  - `BenchmarkSuiteSpec` (frozen dataclass): `cells: tuple[BenchmarkCellConfig, ...]` (sorted by `(tier, benchmark_id)`), `gate: Tier4GateConfig | None`, `reference_column: str | None`
  - `load_benchmark_file(path: str | Path) -> BenchmarkFileConfig`
  - `load_benchmark_suite_config(config_dir: str | Path) -> BenchmarkSuiteSpec`

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_config.py`:

```python
"""Config-layer tests for the 5-tier benchmark hierarchy."""

from __future__ import annotations

from pathlib import Path

import pytest

from nmr.benchmark import (
    BenchmarkCellConfig,
    BenchmarkFileConfig,
    BenchmarkSuiteSpec,
    Tier4GateConfig,
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
    assert gate.corr_sharpe_ac_min == 1.50
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'BenchmarkCellConfig' from 'nmr.benchmark'` (and missing `configs/benchmarks/`).

- [ ] **Step 3: Write the 8 YAML files**

Create `configs/benchmarks/tier0_null.yaml`:

```yaml
# Tier 0 — Null & statistical invariants (the sanity gate).
tier: 0
cells:
  - benchmark_id: null_constant_05
    input_space: none
    model_kind: null_constant_05
    seed: 42

  - benchmark_id: null_uniform_rand
    input_space: none
    model_kind: null_uniform_rand
    seed: 42

  - benchmark_id: null_gaussian_rand
    input_space: none
    model_kind: null_gaussian_rand
    seed: 42

  - benchmark_id: null_feature_mean
    input_space: small
    model_kind: null_feature_mean
    seed: 42
```

Create `configs/benchmarks/tier1_ridge_small.yaml`:

```yaml
# Tier 1 — Convex linear baseline, small universe (42 features).
tier: 1
cells:
  - benchmark_id: linear_ridge_small
    input_space: small
    model_kind: ridge
    targets: ["target"]
    params:
      alpha: 1.0
    seed: 42
    anchors:
      corr: 0.0145
      sharpe: 1.05
```

Create `configs/benchmarks/tier1_ridge_medium.yaml`:

```yaml
# Tier 1 — Convex linear baseline, medium universe (780 features).
tier: 1
cells:
  - benchmark_id: linear_ridge_medium
    input_space: medium
    model_kind: ridge
    targets: ["target"]
    params:
      alpha: 10.0
    seed: 42
    anchors:
      corr: 0.0145
      sharpe: 1.05
```

Create `configs/benchmarks/tier1_ridge_multitarget.yaml`:

```yaml
# Tier 1 — Four independent Ridge fits, equal-weight rank-Gaussian blend.
tier: 1
cells:
  - benchmark_id: linear_ridge_multitarget
    input_space: medium
    model_kind: ridge
    targets: ["target", "target_cyrusd_20", "target_sam_20", "target_victor_20"]
    params:
      alpha: 10.0
    seed: 42
    anchors:
      corr: 0.0145
      sharpe: 1.05
```

Create `configs/benchmarks/tier2_tree_shallow.yaml`:

```yaml
# Tier 2 — Shallow non-linear trees.
tier: 2
cells:
  - benchmark_id: tree_lgbm_shallow_small
    input_space: small
    model_kind: lightgbm
    targets: ["target"]
    params:
      max_depth: 3
      num_leaves: 7
      learning_rate: 0.02
      n_estimators: 500
      colsample_bytree: 0.1
      subsample: 0.8
    seed: 42
    fast_mode_params:
      n_estimators: 50

  - benchmark_id: tree_xgb_shallow_medium
    input_space: medium
    model_kind: xgboost
    targets: ["target"]
    params:
      max_depth: 4
      max_leaves: 15
      n_estimators: 1000
      learning_rate: 0.01
      colsample_bytree: 0.1
    seed: 42
    fast_mode_params:
      n_estimators: 50
```

Create `configs/benchmarks/tier2_tree_fast.yaml`:

```yaml
# Tier 2 — Canonical fast-preset LightGBM on medium.
tier: 2
cells:
  - benchmark_id: tree_lgbm_fast_medium
    input_space: medium
    model_kind: lightgbm
    targets: ["target"]
    params:
      max_depth: 5
      num_leaves: 31
      n_estimators: 2000
      learning_rate: 0.01
      colsample_bytree: 0.1
    seed: 42
    fast_mode_params:
      n_estimators: 50
    anchors:
      corr: 0.0210
      sharpe: 1.30
```

Create `configs/benchmarks/tier3_sunshine.yaml`:

```yaml
# Tier 3 — Canonical community & tutorial baselines, in-process re-fits.
tier: 3
cells:
  - benchmark_id: canon_hello_numerai
    input_space: small
    model_kind: lightgbm
    targets: ["target"]
    params:
      max_depth: 5
      num_leaves: 31
      n_estimators: 2000
      learning_rate: 0.01
      colsample_bytree: 0.1
    seed: 42
    fast_mode_params:
      n_estimators: 50
    anchors:
      corr: 0.0130

  - benchmark_id: canon_neutralized_50
    input_space: medium
    model_kind: lightgbm
    targets: ["target"]
    params:
      max_depth: 5
      num_leaves: 31
      n_estimators: 2000
      learning_rate: 0.01
      colsample_bytree: 0.1
    seed: 42
    neutralization: 0.5
    fast_mode_params:
      n_estimators: 50
    anchors:
      corr: 0.0220

  - benchmark_id: canon_sunshine_ensemble
    input_space: medium
    model_kind: lightgbm
    targets: ["target", "target_cyrusd_20", "target_sam_20", "target_victor_20"]
    params:
      max_depth: 5
      num_leaves: 31
      n_estimators: 2000
      learning_rate: 0.01
      colsample_bytree: 0.1
    seed: 42
    neutralization: 0.25
    fast_mode_params:
      n_estimators: 50
    anchors:
      corr: 0.0245
```

Create `configs/benchmarks/tier4_gate.yaml`:

```yaml
# Tier 4 — Production capital gate (the line in the sand).
tier: 4
reference_column: v53_lgbm_ender60
gate:
  corr_min: 0.0286
  corr_sharpe_ac_min: 1.50
  fnc_min: 0.020
  deflated_sharpe_min: 0.95
  gain_to_pain_min: 1.50
  cagr_min: 0.0
  turnover_max: 0.35
```

- [ ] **Step 4: Implement loader + dataclasses**

Append to `nmr/benchmark.py` (after existing imports):

```python
import dataclasses
from types import MappingProxyType

import yaml
```

Then add (anywhere after the existing `NULL_BASELINES` block; keep legacy code intact):

```python
# ---------------------------------------------------------------------------
# 5-tier benchmark hierarchy: config schema (spec:
# docs/superpowers/specs/2026-08-15-benchmark-hierarchy-design.md)
# ---------------------------------------------------------------------------

VALID_BENCHMARK_TIERS: tuple[int, ...] = (0, 1, 2, 3, 4)
VALID_INPUT_SPACES: tuple[str, ...] = ("none", "small", "medium")
VALID_BENCHMARK_MODEL_KINDS: tuple[str, ...] = (
    "null_constant_05",
    "null_uniform_rand",
    "null_gaussian_rand",
    "null_feature_mean",
    "ridge",
    "lightgbm",
    "xgboost",
)
NULL_KINDS: tuple[str, ...] = (
    "null_constant_05",
    "null_uniform_rand",
    "null_gaussian_rand",
    "null_feature_mean",
)
DEFAULT_BENCHMARK_SEED: int = 42
DEFAULT_BENCHMARK_PURGE_ERAS: int = 8


def _reject_unknown_keys(cls: type, data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError(
            f"{cls.__name__} section must be a mapping, got {type(data).__name__}"
        )
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")


def _freeze_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    out = dict(value)
    for key in out:
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings, got {key!r}")
    return MappingProxyType(out)


@dataclasses.dataclass(frozen=True)
class Tier4GateConfig:
    corr_min: float
    corr_sharpe_ac_min: float
    fnc_min: float
    deflated_sharpe_min: float
    gain_to_pain_min: float
    cagr_min: float
    turnover_max: float

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Tier4GateConfig.{field.name} must be numeric, got {value!r}"
                )
            if not float(value) == float(value):  # NaN check
                raise ValueError(f"Tier4GateConfig.{field.name} must be finite")
        if not (-1.0 <= self.corr_min <= 1.0):
            raise ValueError(f"corr_min out of range: {self.corr_min!r}")
        if self.turnover_max < 0.0:
            raise ValueError(f"turnover_max must be >= 0: {self.turnover_max!r}")


@dataclasses.dataclass(frozen=True)
class BenchmarkCellConfig:
    benchmark_id: str
    input_space: str
    model_kind: str
    tier: int
    targets: tuple[str, ...] = ("target",)
    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    seed: int = DEFAULT_BENCHMARK_SEED
    neutralization: float = 0.0
    anchors: Mapping[str, float] | None = None
    fast_mode_params: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.benchmark_id or not isinstance(self.benchmark_id, str):
            raise ValueError(f"benchmark_id must be a non-empty string: {self.benchmark_id!r}")
        if self.tier not in VALID_BENCHMARK_TIERS:
            raise ValueError(f"tier={self.tier!r} not in {VALID_BENCHMARK_TIERS}")
        if self.input_space not in VALID_INPUT_SPACES:
            raise ValueError(
                f"input_space={self.input_space!r} not in {VALID_INPUT_SPACES}"
            )
        if self.model_kind not in VALID_BENCHMARK_MODEL_KINDS:
            raise ValueError(
                f"model_kind={self.model_kind!r} not in {VALID_BENCHMARK_MODEL_KINDS}"
            )
        if self.model_kind == "null_feature_mean" and self.input_space != "small":
            raise ValueError(
                "null_feature_mean requires input_space='small', "
                f"got {self.input_space!r}"
            )
        if self.model_kind in NULL_KINDS and self.input_space != "none" \
                and self.model_kind != "null_feature_mean":
            raise ValueError(
                f"{self.model_kind} requires input_space='none', "
                f"got {self.input_space!r}"
            )
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("targets must be a non-empty tuple")
        if not all(isinstance(t, str) and t for t in self.targets):
            raise ValueError(f"targets must be non-empty strings: {self.targets!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if not 0.0 <= float(self.neutralization) <= 1.0:
            raise ValueError(
                f"neutralization must be in [0, 1], got {self.neutralization!r}"
            )
        object.__setattr__(self, "params", _freeze_mapping(self.params, name="params"))
        if self.anchors is not None:
            anchors = _freeze_mapping(self.anchors, name="anchors")
            for key, value in anchors.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"anchor {key!r} must be numeric, got {value!r}")
            object.__setattr__(self, "anchors", anchors)
        if self.fast_mode_params is not None:
            object.__setattr__(
                self,
                "fast_mode_params",
                _freeze_mapping(self.fast_mode_params, name="fast_mode_params"),
            )


@dataclasses.dataclass(frozen=True)
class BenchmarkFileConfig:
    tier: int
    cells: tuple[BenchmarkCellConfig, ...] = ()
    reference_column: str | None = None
    gate: Tier4GateConfig | None = None

    def __post_init__(self) -> None:
        if self.tier not in VALID_BENCHMARK_TIERS:
            raise ValueError(
                f"tier={self.tier!r} not in {VALID_BENCHMARK_TIERS}"
            )
        if self.tier == 4:
            if self.gate is None:
                raise ValueError("tier 4 config requires a 'gate' section")
            if not self.reference_column:
                raise ValueError("tier 4 config requires a non-empty reference_column")
        else:
            if not self.cells:
                raise ValueError(
                    f"tier {self.tier} config requires non-empty cells"
                )
            if self.gate is not None:
                raise ValueError(f"gate section only allowed for tier 4, got tier {self.tier}")
        ids = [cell.benchmark_id for cell in self.cells]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate benchmark ids in file: {ids}")


def _build_benchmark_cell(data: Any, tier: int) -> BenchmarkCellConfig:
    if not isinstance(data, dict):
        raise ValueError(
            f"benchmark cell must be a mapping, got {type(data).__name__}"
        )
    if "benchmark_id" not in data:
        raise ValueError(f"benchmark cell missing 'benchmark_id': {data!r}")
    if "tier" in data and int(data["tier"]) != int(tier):
        raise ValueError(
            f"cell tier {data['tier']!r} conflicts with file tier {tier!r}"
        )
    data["tier"] = int(tier)
    _reject_unknown_keys(BenchmarkCellConfig, data)
    if isinstance(data.get("targets"), list):
        data["targets"] = tuple(data["targets"])
    return BenchmarkCellConfig(**data)


def load_benchmark_file(path: str | Path) -> BenchmarkFileConfig:
    """Load and validate a single benchmark tier config file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"benchmark config must be a mapping, got {type(raw).__name__}")
    _reject_unknown_keys(BenchmarkFileConfig, raw)
    if not isinstance(raw.get("cells", []), list):
        raise ValueError("cells must be a list")
    gate_raw = raw.get("gate")
    gate = None
    if gate_raw is not None:
        _reject_unknown_keys(Tier4GateConfig, gate_raw)
        gate = Tier4GateConfig(**gate_raw)
    return BenchmarkFileConfig(
        tier=int(raw["tier"]),
        cells=tuple(_build_benchmark_cell(c, int(raw["tier"])) for c in raw["cells"]),
        reference_column=raw.get("reference_column"),
        gate=gate,
    )


@dataclasses.dataclass(frozen=True)
class BenchmarkSuiteSpec:
    cells: tuple[BenchmarkCellConfig, ...]
    gate: Tier4GateConfig | None
    reference_column: str | None


def load_benchmark_suite_config(config_dir: str | Path) -> BenchmarkSuiteSpec:
    """Load every *.yaml file in config_dir and aggregate into a suite spec."""
    directory = Path(config_dir)
    files = sorted(p for p in directory.glob("*.yaml"))
    if not files:
        raise ValueError(f"no benchmark config files found in {directory}")
    all_cells: list[BenchmarkCellConfig] = []
    gate: Tier4GateConfig | None = None
    reference_column: str | None = None
    for path in files:
        file_cfg = load_benchmark_file(path)
        if file_cfg.gate is not None:
            if gate is not None:
                raise ValueError("multiple tier-4 gate configs found")
            gate = file_cfg.gate
            reference_column = file_cfg.reference_column
        all_cells.extend(file_cfg.cells)
    ids = [cell.benchmark_id for cell in all_cells]
    if len(set(ids)) != len(ids):
        seen = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate benchmark ids across configs: {seen}")
    all_cells.sort(key=lambda c: (c.tier, c.benchmark_id))
    return BenchmarkSuiteSpec(
        cells=tuple(all_cells),
        gate=gate,
        reference_column=reference_column,
    )
```

- [ ] **Step 4b: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_config.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add configs/benchmarks nmr/benchmark.py tests/test_benchmark_config.py
git commit -m "feat: benchmark hierarchy config schema, loader, and tier YAMLs"
```

---

### Task 2: Purged train→validation split helper

**Files:**
- Modify: `nmr/benchmark.py` (append after Task 1 section)
- Test: `tests/test_benchmark_purge.py`

**Interfaces:**
- Produces (used by Tasks 4–6):
  - `train_validation_purged_split(train_eras: Sequence[str], val_eras: Sequence[str], *, purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS) -> tuple[tuple[str, ...], tuple[str, ...]]` — returns `(trimmed_train_eras, val_eras)`; raises `ValueError` on non-numeric labels, zero-padding inconsistency, overlap, non-strict ordering, or a purge buffer that is not exactly `purge_eras` eras wide.

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_purge.py`:

```python
"""Purged train->validation split invariants for benchmark fits."""

from __future__ import annotations

import pytest

from nmr.benchmark import train_validation_purged_split


def _eras(start: int, stop: int) -> list[str]:
    return [f"{i:04d}" for i in range(start, stop)]


def test_valid_split_returns_trimmed_train_and_val() -> None:
    train, val = train_validation_purged_split(
        _eras(1, 100), _eras(108, 200), purge_eras=8
    )
    assert train == tuple(_eras(1, 92))
    assert val == tuple(_eras(108, 200))


def test_zero_purge_keeps_all_train_eras() -> None:
    train, val = train_validation_purged_split(
        _eras(1, 100), _eras(101, 200), purge_eras=0
    )
    assert train == tuple(_eras(1, 100))


def test_gap_too_small_raises() -> None:
    with pytest.raises(ValueError, match="purge"):
        train_validation_purged_split(_eras(1, 100), _eras(102, 200), purge_eras=8)


def test_gap_too_large_raises() -> None:
    with pytest.raises(ValueError, match="purge"):
        train_validation_purged_split(_eras(1, 100), _eras(120, 200), purge_eras=8)


def test_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap"):
        train_validation_purged_split(_eras(1, 100), _eras(50, 200), purge_eras=8)


def test_non_numeric_label_raises() -> None:
    with pytest.raises(ValueError, match="[Nn]umeric"):
        train_validation_purged_split(["era_1", "0002"], _eras(10, 20), purge_eras=8)


def test_zero_padding_inconsistency_raises() -> None:
    with pytest.raises(ValueError, match="[Zz]ero-padding"):
        train_validation_purged_split(["1", "02", "0003"], _eras(11, 20), purge_eras=8)


def test_degenerate_train_raises() -> None:
    with pytest.raises(ValueError, match="purge|train"):
        train_validation_purged_split(_eras(1, 5), _eras(20, 30), purge_eras=8)


def test_empty_eras_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        train_validation_purged_split([], _eras(20, 30), purge_eras=8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_purge.py -q`
Expected: FAIL — `ImportError` (function not defined).

- [ ] **Step 3: Implement**

Append to `nmr/benchmark.py`:

```python
def _ordered_numeric_eras(eras: Sequence[str]) -> list[str]:
    """Dedupe, validate, and numerically sort era labels."""
    if not eras:
        raise ValueError("era universe is empty")
    mapping: dict[int, str] = {}
    for era in eras:
        if not isinstance(era, str):
            raise ValueError(
                f"Era labels must be strings, got {type(era).__name__}"
            )
        try:
            era_num = int(era)
        except ValueError as exc:
            raise ValueError(f"Non-numeric era label {era!r}") from exc
        if era_num in mapping and mapping[era_num] != era:
            raise ValueError(
                "Inconsistent zero-padding in era labels: "
                f"{mapping[era_num]!r} vs {era!r}"
            )
        mapping[era_num] = era
    return [mapping[num] for num in sorted(mapping)]


def train_validation_purged_split(
    train_eras: Sequence[str],
    val_eras: Sequence[str],
    *,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the purged train->validation era partition for benchmark fits.

    Mirrors ``PurgedEraSplitter`` invariants for the fixed one-shot split:
    train eras strictly precede validation eras, and exactly ``purge_eras``
    eras are excluded between them (the trimmed train tail).
    """
    if isinstance(purge_eras, bool) or not isinstance(purge_eras, int) or purge_eras < 0:
        raise ValueError(f"purge_eras must be a non-negative int, got {purge_eras!r}")

    ordered_train = _ordered_numeric_eras(train_eras)
    ordered_val = _ordered_numeric_eras(val_eras)

    overlap = set(ordered_train) & set(ordered_val)
    if overlap:
        raise ValueError(f"train/validation era overlap: {sorted(overlap)[:5]}")

    if len(ordered_train) <= purge_eras:
        raise ValueError(
            "Not enough train eras after purge: "
            f"train={len(ordered_train)}, purge={purge_eras}"
        )

    trimmed = ordered_train[:-purge_eras]
    train_max = int(trimmed[-1])
    val_min = int(ordered_val[0])
    if train_max >= val_min:
        raise ValueError(
            "train eras must be strictly earlier than validation eras: "
            f"max(train)={train_max} >= min(val)={val_min}"
        )

    gap_width = val_min - train_max - 1
    if gap_width != purge_eras:
        raise ValueError(
            f"Purge buffer is not exactly {purge_eras} eras wide: got {gap_width} "
            f"(max(train)={train_max}, min(val)={val_min})"
        )

    return tuple(trimmed), tuple(ordered_val)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_purge.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark.py tests/test_benchmark_purge.py
git commit -m "feat: purged train->validation split helper for benchmark fits"
```

---

### Task 3: Tier-0 null prediction generators

**Files:**
- Modify: `nmr/benchmark.py` (append after Task 2 section)
- Test: `tests/test_benchmark_null.py`

**Interfaces:**
- Produces (used by Task 8):
  - `generate_null_predictions(prediction_index: pl.DataFrame, *, kind: str, seed: int, features: pl.DataFrame | None = None, feature_cols: Sequence[str] = (), era_col: str = "era", id_col: str = "id", pred_col: str = "prediction") -> pl.DataFrame` — returns sorted, unique `[era, id, prediction]` on the prediction index rows; deterministic per `(kind, seed)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_null.py`:

```python
"""Tier-0 null prediction generator contracts."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import generate_null_predictions


def _index(n_eras: int = 5, rows_per_era: int = 4) -> pl.DataFrame:
    rows = []
    for era_num in range(1, n_eras + 1):
        for idx in range(rows_per_era):
            rows.append({"era": f"{era_num:04d}", "id": f"{era_num}_{idx}"})
    return pl.DataFrame(rows)


def _features(index: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    return index.with_columns(
        pl.Series("f1", rng.normal(size=index.height)),
        pl.Series("f2", rng.normal(size=index.height)),
    )


def test_constant_is_exactly_half() -> None:
    out = generate_null_predictions(_index(), kind="null_constant_05", seed=42)
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == 20
    assert (out.get_column("prediction") == 0.5).all()


def test_uniform_is_seeded_and_bounded() -> None:
    idx = _index()
    a = generate_null_predictions(idx, kind="null_uniform_rand", seed=42)
    b = generate_null_predictions(idx, kind="null_uniform_rand", seed=42)
    c = generate_null_predictions(idx, kind="null_uniform_rand", seed=43)
    values = a.get_column("prediction").to_numpy()
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert a.equals(b)
    assert not a.equals(c)


def test_gaussian_is_clipped_and_seeded() -> None:
    idx = _index()
    a = generate_null_predictions(idx, kind="null_gaussian_rand", seed=42)
    b = generate_null_predictions(idx, kind="null_gaussian_rand", seed=42)
    values = a.get_column("prediction").to_numpy()
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert a.equals(b)


def test_feature_mean_matches_manual_row_mean() -> None:
    idx = _index()
    feats = _features(idx)
    out = generate_null_predictions(
        idx, kind="null_feature_mean", seed=42,
        features=feats, feature_cols=["f1", "f2"],
    )
    manual = feats.with_columns(
        pl.mean_horizontal([pl.col("f1"), pl.col("f2")]).alias("prediction")
    ).select(["era", "id", "prediction"])
    assert out.sort(["era", "id"]).equals(manual.sort(["era", "id"]))


def test_feature_mean_missing_column_raises() -> None:
    idx = _index()
    feats = _features(idx)
    with pytest.raises(ValueError, match="f3"):
        generate_null_predictions(
            idx, kind="null_feature_mean", seed=42,
            features=feats, feature_cols=["f1", "f3"],
        )


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        generate_null_predictions(_index(), kind="null_bogus", seed=42)


def test_missing_join_keys_raise() -> None:
    bad = pl.DataFrame({"id": ["a"], "prediction": [0.5]})
    with pytest.raises(ValueError, match="era"):
        generate_null_predictions(bad, kind="null_constant_05", seed=42)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_null.py -q`
Expected: FAIL — `ImportError` (function not defined).

- [ ] **Step 3: Implement**

Append to `nmr/benchmark.py`:

```python
def generate_null_predictions(
    prediction_index: pl.DataFrame,
    *,
    kind: str,
    seed: int,
    features: pl.DataFrame | None = None,
    feature_cols: Sequence[str] = (),
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Generate deterministic tier-0 null predictions on the prediction index."""
    if kind not in NULL_KINDS:
        raise ValueError(f"Unknown null kind {kind!r}; expected one of {NULL_KINDS}")
    missing_keys = [c for c in (era_col, id_col) if c not in prediction_index.columns]
    if missing_keys:
        raise ValueError(f"prediction_index missing required columns: {missing_keys}")

    index = (
        prediction_index.select([era_col, id_col])
        .unique()
        .sort([era_col, id_col])
    )
    n = index.height
    rng = np.random.default_rng(seed)

    if kind == "null_constant_05":
        values = np.full(n, 0.5, dtype=float)
    elif kind == "null_uniform_rand":
        values = rng.uniform(0.0, 1.0, n)
    elif kind == "null_gaussian_rand":
        values = np.clip(rng.normal(0.5, 0.15, n), 0.0, 1.0)
    else:  # null_feature_mean
        if features is None:
            raise ValueError("null_feature_mean requires a features frame")
        if not feature_cols:
            raise ValueError("null_feature_mean requires at least one feature column")
        missing_feats = [c for c in feature_cols if c not in features.columns]
        if missing_feats:
            raise ValueError(f"features missing columns: {missing_feats}")
        joined = index.join(
            features.select([era_col, id_col, *feature_cols]),
            on=[era_col, id_col],
            how="inner",
        )
        if joined.height != n:
            raise ValueError(
                f"null_feature_mean join dropped {n - joined.height} rows"
            )
        values = (
            joined.select(
                pl.mean_horizontal(
                    [pl.col(c).cast(pl.Float64, strict=False) for c in feature_cols]
                )
            )
            .to_series()
            .to_numpy()
        )

    return index.with_columns(pl.Series(pred_col, values))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_null.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark.py tests/test_benchmark_null.py
git commit -m "feat: tier-0 null prediction generators (constant/uniform/gaussian/feature-mean)"
```

---

### Task 4: Tier-1 Ridge generators (standardization + multi-target blend)

**Files:**
- Modify: `nmr/benchmark.py` (append after Task 3 section)
- Test: `tests/test_benchmark_ridge.py`

**Interfaces:**
- Produces (used by Tasks 6–8):
  - `generate_ridge_predictions(train: pl.DataFrame, val: pl.DataFrame, *, targets: Sequence[str], feature_cols: Sequence[str], alpha: float, seed: int, purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS, era_col: str = "era", id_col: str = "id", pred_col: str = "prediction") -> pl.DataFrame`
  - Contract: `train` contains `[era, id, *feature_cols, *targets]`; `val` contains `[era, id, *feature_cols]`. Returns `[era, id, prediction]` covering all val rows, per-era rank-gaussianized. Fits per target with independent NaN masking (watchpoint 1); standardization uses trimmed-train stats with zero-variance → 0.0 (watchpoint 2); multi-target blend is equal-weight in rank-Gaussian domain via `Ensembler`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_ridge.py`:

```python
"""Tier-1 ridge benchmark generator contracts."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import generate_ridge_predictions


def _domain(
    *, n_train_eras: int = 30, n_val_eras: int = 10, rows_per_era: int = 12,
    seed: int = 20260815,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    feature_cols = ["f1", "f2", "f3", "f_const"]

    def make(eras: range) -> list[dict[str, float | str]]:
        rows = []
        for era_num in eras:
            era = f"{era_num:04d}"
            for idx in range(rows_per_era):
                f1, f2, f3 = (float(rng.normal()) for _ in range(3))
                target = float(np.clip(0.5 + 0.2 * (0.8 * f1 - 0.4 * f2 + 0.2 * f3)
                                      + rng.normal(0.0, 0.3), 0.0, 1.0))
                aux = float(np.clip(target + rng.normal(0.0, 0.2), 0.0, 1.0))
                rows.append({
                    "era": era, "id": f"{era}_{idx}",
                    "f1": f1, "f2": f2, "f3": f3, "f_const": 1.0,
                    "target": target, "aux": aux,
                })
        return rows

    train = pl.DataFrame(make(range(1, n_train_eras + 1)))
    val = pl.DataFrame(make(range(n_train_eras + 1, n_train_eras + 1 + n_val_eras)))
    val = val.select(["era", "id", *feature_cols])
    return train, val, feature_cols


def test_single_target_ridge_covers_val_and_is_finite() -> None:
    train, val, feats = _domain()
    out = generate_ridge_predictions(
        train, val, targets=["target"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
    )
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == val.height
    assert set(out.get_column("era").unique().to_list()) == \
        set(val.get_column("era").unique().to_list())
    assert out.get_column("prediction").is_finite().all()


def test_ridge_is_deterministic_per_seed() -> None:
    train, val, feats = _domain()
    a = generate_ridge_predictions(
        train, val, targets=["target"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
    )
    b = generate_ridge_predictions(
        train, val, targets=["target"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
    )
    assert a.equals(b)


def test_zero_variance_feature_does_not_produce_nan() -> None:
    train, val, feats = _domain()
    out = generate_ridge_predictions(
        train, val, targets=["target"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
    )
    values = out.get_column("prediction").to_numpy()
    assert np.isfinite(values).all()


def test_per_era_output_is_rank_gaussianized() -> None:
    train, val, feats = _domain()
    out = generate_ridge_predictions(
        train, val, targets=["target"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
    )
    per_era = out.group_by("era").agg(pl.col("prediction"))
    for era_df in per_era.iter_rows(named=True):
        values = np.asarray(era_df["prediction"], dtype=float)
        assert abs(float(np.mean(values))) < 1e-8
        assert abs(float(np.std(values, ddof=0)) - 1.0) < 1e-6


def test_multitarget_nan_masking_and_blend() -> None:
    train, val, feats = _domain()
    # Poison some aux-target rows with nulls (watchpoint 1: independent masking)
    poisoned = train.with_columns(
        pl.when(pl.col("era") == "0001").then(None).otherwise(pl.col("aux")).alias("aux")
    )
    out = generate_ridge_predictions(
        poisoned, val, targets=["target", "aux"], feature_cols=feats,
        alpha=1.0, seed=42, purge_eras=8,
    )
    assert out.height == val.height
    assert np.isfinite(out.get_column("prediction").to_numpy()).all()


def test_tight_gap_raises() -> None:
    train, val, feats = _domain(n_train_eras=30, n_val_eras=10)
    close_val = val.with_columns(pl.lit("0032").alias("era"))
    with pytest.raises(ValueError, match="purge"):
        generate_ridge_predictions(
            train, close_val, targets=["target"], feature_cols=feats,
            alpha=1.0, seed=42, purge_eras=8,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_ridge.py -q`
Expected: FAIL — `ImportError` (function not defined).

- [ ] **Step 3: Implement**

Append to `nmr/benchmark.py`:

```python
def _standardize_feature_block(
    train_values: np.ndarray, val_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize with train statistics; zero-variance features -> 0.0."""
    mu = np.mean(train_values, axis=0)
    sigma = np.std(train_values, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    scale = np.where((sigma > 0.0) & np.isfinite(sigma), 1.0 / sigma, 0.0)
    return (train_values - mu) * scale, (val_values - mu) * scale


def generate_ridge_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    targets: Sequence[str],
    feature_cols: Sequence[str],
    alpha: float,
    seed: int,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Fit purged Ridge models per target and blend in rank-Gaussian domain."""
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha < 0:
        raise ValueError(f"alpha must be a non-negative number, got {alpha!r}")
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")
    if not targets:
        raise ValueError("targets must be non-empty")

    trimmed_train_eras, val_eras = train_validation_purged_split(
        train.get_column(era_col).unique().to_list(),
        val.get_column(era_col).unique().to_list(),
        purge_eras=purge_eras,
    )

    train_rows = train.filter(pl.col(era_col).is_in(trimmed_train_eras))
    val_rows = val.sort([era_col, id_col])
    missing_feats = [c for c in feature_cols if c not in train.columns or c not in val.columns]
    if missing_feats:
        raise ValueError(f"missing feature columns: {missing_feats}")

    x_train_raw = train_rows.select(feature_cols).cast(pl.Float64).to_numpy()
    x_val_raw = val_rows.select(feature_cols).cast(pl.Float64).to_numpy()
    x_train, x_val = _standardize_feature_block(x_train_raw, x_val_raw)

    component_preds: dict[str, np.ndarray] = {}
    for target in targets:
        if target not in train.columns:
            raise ValueError(f"missing target column: {target!r}")
        y = train_rows.get_column(target).cast(pl.Float64).to_numpy()
        mask = np.isfinite(y)
        if mask.sum() < 2:
            raise ValueError(
                f"target {target!r} has fewer than 2 finite train rows after purge"
            )
        model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=seed)
        model.fit(x_train[mask], y[mask])
        component_preds[target] = np.asarray(model.predict(x_val), dtype=float)

    frame = val_rows.select([era_col, id_col]).with_columns(
        [pl.Series(target, component_preds[target]) for target in targets]
    )
    weights = [1.0 / len(targets)] * len(targets)
    ensembler = Ensembler()
    blended = ensembler.blend(
        Ensembler.rank_normalize(frame, pred_cols=list(targets), era_col=era_col),
        pred_cols=list(targets),
        weights=weights,
        era_col=era_col,
        out_col=pred_col,
    )
    return blended.select([era_col, id_col, pred_col]).sort([era_col, id_col])
```

Add `from nmr.ensemble import Ensembler` to the imports at the top of `nmr/benchmark.py` (extend the existing import block).

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_ridge.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark.py tests/test_benchmark_ridge.py
git commit -m "feat: tier-1 ridge benchmark generator (purged, standardized, rank-gaussian blend)"
```

---

### Task 5: Tier-2 tree generators (+ shared backend-param constructor in models.py)

**Files:**
- Modify: `nmr/models.py` (extract module-level param-resolution helper; behavior unchanged)
- Modify: `nmr/benchmark.py` (append `generate_tree_predictions`)
- Test: `tests/test_benchmark_trees.py`

**Interfaces:**
- Produces (used by Tasks 6–8):
  - `nmr.models.construct_tree_model(backend: str, params: Mapping[str, Any], *, seed: int, n_features: int, device: str = "cpu") -> object` — resolves params (colsample floor, `num_leaves`→`max_leaves` for XGBoost, determinism flags, `n_jobs=1`) and returns the estimator.
  - `generate_tree_predictions(train: pl.DataFrame, val: pl.DataFrame, *, target: str, feature_cols: Sequence[str], backend: str, params: Mapping[str, Any], seed: int, purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS, era_col: str = "era", id_col: str = "id", pred_col: str = "prediction") -> pl.DataFrame` — raw features, CPU-only, per-era rank-gaussianized output.

- [ ] **Step 1: Read the current param-resolution code**

Read `nmr/models.py` lines 100–200 and 490–600. Identify the exact signatures of `_resolved_params` (or equivalent) and `_build_model`, including how `ModelOrchestrator` calls them. The refactor below delegates to the same logic; adapt names only if they differ from `_resolved_params` / `_build_model` (do not change behavior — `tests/test_models.py` is the guard).

- [ ] **Step 2: Extract the shared constructor**

In `nmr/models.py`, add a module-level function next to `_build_model`:

```python
def construct_tree_model(
    backend: str,
    params: Mapping[str, Any],
    *,
    seed: int,
    n_features: int,
    device: str = "cpu",
) -> object:
    """Build a deterministic, CPU-default tree estimator from raw params.

    Applies the same backend param mapping, colsample flooring, and
    determinism flags as ``ModelOrchestrator``. Used by the benchmark
    hierarchy so benchmark cells never hand-duplicate param resolution.
    """
    config = ModelConfig(
        backend=backend,
        preset="fast",
        params=dict(params),
        device=device,
    )
    orchestrator = ModelOrchestrator(config, seed=seed)
    resolved = orchestrator._resolved_params(
        use_gpu=device != "cpu", n_features=int(n_features)
    )
    return orchestrator._build_model(resolved)
```

If `_resolved_params` / `_build_model` / `ModelConfig` have different shapes than shown, first read their definitions (lines ~100–200 and ~490–600 of `nmr/models.py`) and adapt the wrapper while preserving behavior. After this edit, run the existing model tests to confirm no behavior change:

Run: `./.venv/Scripts/python -m pytest tests/test_models.py -q`
Expected: PASS (existing tests unchanged).

- [ ] **Step 3: Write the failing benchmark test**

Create `tests/test_benchmark_trees.py`:

```python
"""Tier-2 tree benchmark generator contracts."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import generate_tree_predictions
from nmr.models import construct_tree_model


def _domain(
    *, n_train_eras: int = 24, n_val_eras: int = 6, rows_per_era: int = 10,
    seed: int = 20260815,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    feature_cols = ["f1", "f2", "f3"]

    def make(eras: range) -> list[dict[str, float | str]]:
        rows = []
        for era_num in eras:
            era = f"{era_num:04d}"
            for idx in range(rows_per_era):
                f1, f2, f3 = (float(rng.normal()) for _ in range(3))
                target = float(np.clip(
                    0.5 + 0.25 * (f1 * f2 - 0.3 * f3) + rng.normal(0.0, 0.4),
                    0.0, 1.0,
                ))
                rows.append({
                    "era": era, "id": f"{era}_{idx}",
                    "f1": f1, "f2": f2, "f3": f3, "target": target,
                })
        return rows

    train = pl.DataFrame(make(range(1, n_train_eras + 1)))
    val = pl.DataFrame(make(range(n_train_eras + 1, n_train_eras + 1 + n_val_eras)))
    val = val.select(["era", "id", *feature_cols])
    return train, val, feature_cols


FAST_PARAMS = {"n_estimators": 5, "learning_rate": 0.1, "max_depth": 2, "num_leaves": 4, "colsample_bytree": 0.5}


def test_lgbm_tree_covers_val_and_is_finite() -> None:
    train, val, feats = _domain()
    out = generate_tree_predictions(
        train, val, target="target", feature_cols=feats,
        backend="lightgbm", params=FAST_PARAMS, seed=42, purge_eras=8,
    )
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == val.height
    assert np.isfinite(out.get_column("prediction").to_numpy()).all()


def test_xgb_tree_accepts_max_leaves_mapping() -> None:
    train, val, feats = _domain()
    xgb_params = {
        "n_estimators": 5, "learning_rate": 0.1, "max_depth": 2,
        "max_leaves": 4, "colsample_bytree": 0.5,
    }
    out = generate_tree_predictions(
        train, val, target="target", feature_cols=feats,
        backend="xgboost", params=xgb_params, seed=42, purge_eras=8,
    )
    assert np.isfinite(out.get_column("prediction").to_numpy()).all()


def test_tree_is_deterministic_per_seed() -> None:
    train, val, feats = _domain()
    a = generate_tree_predictions(
        train, val, target="target", feature_cols=feats,
        backend="lightgbm", params=FAST_PARAMS, seed=42, purge_eras=8,
    )
    b = generate_tree_predictions(
        train, val, target="target", feature_cols=feats,
        backend="lightgbm", params=FAST_PARAMS, seed=42, purge_eras=8,
    )
    assert a.equals(b)


def test_colsample_floor_applied_by_constructor() -> None:
    model = construct_tree_model(
        "lightgbm", {"colsample_bytree": 0.001}, seed=42, n_features=10,
        device="cpu",
    )
    resolved = model.get_params()["colsample_bytree"]
    assert resolved >= 0.1
```

- [ ] **Step 4: Run test to verify the benchmark test fails (before the generator exists)**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_trees.py::test_lgbm_tree_covers_val_and_is_finite -q`
Expected: FAIL — `ImportError` for `generate_tree_predictions` (the `construct_tree_model` test may pass immediately; that is fine — it guards the refactor).

- [ ] **Step 5: Implement the generator**

Append to `nmr/benchmark.py`:

```python
def generate_tree_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    target: str,
    feature_cols: Sequence[str],
    backend: str,
    params: Mapping[str, Any],
    seed: int,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Fit one shallow tree on purged train eras and predict validation rows."""
    if backend not in ("lightgbm", "xgboost"):
        raise ValueError(f"Unsupported tree backend: {backend!r}")
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")

    trimmed_train_eras, _ = train_validation_purged_split(
        train.get_column(era_col).unique().to_list(),
        val.get_column(era_col).unique().to_list(),
        purge_eras=purge_eras,
    )

    train_rows = train.filter(pl.col(era_col).is_in(trimmed_train_eras))
    val_rows = val.sort([era_col, id_col])
    if target not in train.columns:
        raise ValueError(f"missing target column: {target!r}")
    missing_feats = [c for c in feature_cols if c not in train.columns or c not in val.columns]
    if missing_feats:
        raise ValueError(f"missing feature columns: {missing_feats}")

    x_train = train_rows.select(feature_cols).cast(pl.Float64).to_pandas()
    y = train_rows.get_column(target).cast(pl.Float64).to_numpy()
    mask = np.isfinite(y)
    if mask.sum() < 2:
        raise ValueError(f"target {target!r} has fewer than 2 finite train rows after purge")
    x_val = val_rows.select(feature_cols).cast(pl.Float64).to_pandas()

    model = construct_tree_model(
        backend, dict(params), seed=seed, n_features=len(feature_cols),
        device="cpu",
    )
    model.fit(x_train[mask], y[mask])
    raw = np.asarray(model.predict(x_val), dtype=float)

    frame = val_rows.select([era_col, id_col]).with_columns(pl.Series(pred_col, raw))
    blended = Ensembler().blend(
        Ensembler.rank_normalize(frame, pred_cols=[pred_col], era_col=era_col),
        pred_cols=[pred_col],
        weights=[1.0],
        era_col=era_col,
        out_col=pred_col,
    )
    return blended.select([era_col, id_col, pred_col]).sort([era_col, id_col])
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_trees.py tests/test_models.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add nmr/models.py nmr/benchmark.py tests/test_benchmark_trees.py
git commit -m "feat: tier-2 tree benchmark generator + shared backend-param constructor"
```

---

### Task 6: Tier-3 canonical generators (hello / neutralized-50 / sunshine)

**Files:**
- Modify: `nmr/benchmark.py` (append after Task 5 section)
- Test: `tests/test_benchmark_canonical.py`

**Interfaces:**
- Produces (used by Task 8):
  - `generate_canonical_predictions(train: pl.DataFrame, val: pl.DataFrame, *, targets: Sequence[str], feature_cols: Sequence[str], params: Mapping[str, Any], seed: int, neutralization: float, purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS, era_col: str = "era", id_col: str = "id", pred_col: str = "prediction") -> pl.DataFrame`
  - Single target → one LightGBM fit; multi-target → one LightGBM per target, equal-weight rank-Gaussian blend (reuses `generate_tree_predictions` per target); then post-hoc `NeutralizationEngine.neutralize(proportion=neutralization)` against `feature_cols` when `neutralization > 0`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_canonical.py`:

```python
"""Tier-3 canonical benchmark generator contracts."""

from __future__ import annotations

import numpy as np
import polars as pl

from nmr.benchmark import generate_canonical_predictions
from nmr.ensemble import Ensembler


def _domain(
    *, n_train_eras: int = 24, n_val_eras: int = 6, rows_per_era: int = 10,
    seed: int = 20260815,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    feature_cols = ["f1", "f2", "f3"]

    def make(eras: range) -> list[dict[str, float | str]]:
        rows = []
        for era_num in eras:
            era = f"{era_num:04d}"
            for idx in range(rows_per_era):
                f1, f2, f3 = (float(rng.normal()) for _ in range(3))
                target = float(np.clip(0.5 + 0.2 * f1 + rng.normal(0.0, 0.4), 0.0, 1.0))
                rows.append({
                    "era": era, "id": f"{era}_{idx}",
                    "f1": f1, "f2": f2, "f3": f3,
                    "target": target, "aux": float(np.clip(target + rng.normal(0, 0.1), 0, 1)),
                })
        return rows

    train = pl.DataFrame(make(range(1, n_train_eras + 1)))
    val = pl.DataFrame(make(range(n_train_eras + 1, n_train_eras + 1 + n_val_eras)))
    val = val.select(["era", "id", *feature_cols])
    return train, val, feature_cols


FAST_PARAMS = {"n_estimators": 5, "learning_rate": 0.1, "max_depth": 2, "num_leaves": 4, "colsample_bytree": 0.5}


def test_hello_numerai_contract() -> None:
    train, val, feats = _domain()
    out = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.0, purge_eras=8,
    )
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == val.height
    assert np.isfinite(out.get_column("prediction").to_numpy()).all()


def test_neutralized_50_reduces_feature_alignment() -> None:
    train, val, feats = _domain()
    raw = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.0, purge_eras=8,
    )
    neut = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.5, purge_eras=8,
    )
    assert np.isfinite(neut.get_column("prediction").to_numpy()).all()
    assert not raw.equals(neut)


def test_sunshine_multi_target_blend_and_neutralization() -> None:
    train, val, feats = _domain()
    out = generate_canonical_predictions(
        train, val, targets=["target", "aux"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.25, purge_eras=8,
    )
    assert out.height == val.height
    assert np.isfinite(out.get_column("prediction").to_numpy()).all()


def test_canonical_is_deterministic_per_seed() -> None:
    train, val, feats = _domain()
    a = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.5, purge_eras=8,
    )
    b = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=feats,
        params=FAST_PARAMS, seed=42, neutralization=0.5, purge_eras=8,
    )
    assert a.equals(b)


def test_blend_components_are_rank_normalized_before_blending() -> None:
    train, val, feats = _domain()
    frame = val.with_columns(
        pl.lit(1.0).alias("p1"),
        pl.lit(2.0).alias("p2"),
    )
    blended = Ensembler().blend(
        Ensembler.rank_normalize(frame, pred_cols=["p1", "p2"]),
        pred_cols=["p1", "p2"], weights=[0.5, 0.5], out_col="prediction",
    )
    assert np.isfinite(blended.get_column("prediction").to_numpy()).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_canonical.py -q`
Expected: FAIL — `ImportError` (function not defined).

- [ ] **Step 3: Implement**

Append to `nmr/benchmark.py`:

```python
def generate_canonical_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    targets: Sequence[str],
    feature_cols: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
    neutralization: float,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Tier-3 canonical baselines: LightGBM fits + optional neutralization."""
    if not targets:
        raise ValueError("targets must be non-empty")
    if not 0.0 <= float(neutralization) <= 1.0:
        raise ValueError(f"neutralization must be in [0, 1], got {neutralization!r}")

    if len(targets) == 1:
        out = generate_tree_predictions(
            train, val, target=targets[0], feature_cols=feature_cols,
            backend="lightgbm", params=params, seed=seed,
            purge_eras=purge_eras, era_col=era_col, id_col=id_col,
            pred_col=pred_col,
        )
    else:
        parts: list[pl.DataFrame] = []
        for index, target in enumerate(targets):
            parts.append(
                generate_tree_predictions(
                    train, val, target=target, feature_cols=feature_cols,
                    backend="lightgbm", params=params, seed=seed + index,
                    purge_eras=purge_eras, era_col=era_col, id_col=id_col,
                    pred_col=pred_col,
                ).rename({pred_col: f"__component_{index}"})
            )
        stacked = parts[0]
        for part in parts[1:]:
            stacked = stacked.join(part, on=[era_col, id_col], how="inner")
        component_cols = [f"__component_{index}" for index in range(len(targets))]
        weights = [1.0 / len(targets)] * len(targets)
        ensembler = Ensembler()
        out = ensembler.blend(
            Ensembler.rank_normalize(
                stacked, pred_cols=component_cols, era_col=era_col
            ),
            pred_cols=component_cols,
            weights=weights,
            era_col=era_col,
            out_col=pred_col,
        ).select([era_col, id_col, pred_col]).sort([era_col, id_col])

    if float(neutralization) > 0.0:
        # NeutralizationEngine requires the feature columns present in-frame.
        with_features = out.join(
            val.select([era_col, id_col, *feature_cols]),
            on=[era_col, id_col],
            how="inner",
        )
        engine = NeutralizationEngine()
        out = engine.neutralize(
            with_features,
            pred_col=pred_col,
            feature_cols=list(feature_cols),
            era_col=era_col,
            proportion=float(neutralization),
        ).select([era_col, id_col, pred_col]).sort([era_col, id_col])
    return out
```

Add `from nmr.risk import NeutralizationEngine` to the imports at the top of `nmr/benchmark.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_canonical.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark.py tests/test_benchmark_canonical.py
git commit -m "feat: tier-3 canonical benchmark generators (hello/neutralized-50/sunshine)"
```

---

### Task 7: Tier-4 reference scorer + the three gate functions

**Files:**
- Modify: `nmr/benchmark.py` (append after Task 6 section)
- Test: `tests/test_benchmark_gates_new.py`

**Interfaces:**
- Produces (used by Task 8):
  - `score_benchmark_column(benchmarks: pl.DataFrame, *, column: str, era_col: str = "era", id_col: str = "id", pred_col: str = "prediction") -> pl.DataFrame` — wraps a benchmark-model column as a predictions frame (null/non-finite rows dropped).
  - `assert_tier0_null_floor(scorecards: Mapping[str, MetricScorecard], *, corr_tol: float = 0.005, sharpe_tol: float = 0.10, dsr_tol: float = 0.05) -> None` — requires all four `NULL_KINDS` scorecards; asserts `|corr| <= corr_tol`, `|corr_sharpe_ac| <= sharpe_tol`, `|deflated_sharpe| <= dsr_tol`; all scorecard values finite.
  - `assert_tier4_gate(scorecard: MetricScorecard, gate: Tier4GateConfig) -> None` — the 7 production thresholds; raises `ValueError` listing every violated field; a `None` `turnover_mean` fails loudly with an explicit message.
  - `assert_hierarchy_monotone(scorecards: Mapping[str, MetricScorecard], *, tier_of: Mapping[str, int], atol: float = 1e-5) -> None` — per-tier scalar = max `rank_scalar` of that tier's members; requires `scalar(T0) + atol <= scalar(T1) + ... ` ordering `T0 < T1 < T2 < T3 <= T4` (`atol` on the lower bound of each pair; watchpoint 3).

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_gates_new.py`:

```python
"""Gate mechanics for the 5-tier hierarchy (synthetic scorecards)."""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import (
    NULL_KINDS,
    Tier4GateConfig,
    assert_hierarchy_monotone,
    assert_tier0_null_floor,
    assert_tier4_gate,
    score_benchmark_column,
)
from nmr.scorecard import MetricScorecard, evaluate_model


GATE = Tier4GateConfig(
    corr_min=0.0286,
    corr_sharpe_ac_min=1.50,
    fnc_min=0.020,
    deflated_sharpe_min=0.95,
    gain_to_pain_min=1.50,
    cagr_min=0.0,
    turnover_max=0.35,
)


def _synthetic_inputs(n_eras: int = 60, rows_per_era: int = 16, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    for era_num in range(1, n_eras + 1):
        era = f"{era_num:04d}"
        for idx in range(rows_per_era):
            f1 = float(rng.normal())
            latent = 0.8 * f1 + float(rng.normal(0.0, 0.7))
            target = float(np.clip(0.5 + 0.2 * latent, 0.0, 1.0))
            rows.append({
                "era": era, "id": f"{era}_{idx}",
                "prediction": float(rng.random()),
                "numerai_meta_model": float(0.55 * target + 0.45 * rng.random()),
                "target": target,
                "f1": f1,
                "bench": float(0.6 * target + 0.4 * rng.random()),
            })
    full = pl.DataFrame(rows)
    return (
        full.select(["era", "id", "prediction"]),
        full.select(["era", "id", "numerai_meta_model"]),
        full.select(["era", "id", "bench"]),
        full.select(["era", "id", "f1"]),
        full.select(["era", "id", "target"]),
    )


def _make_scorecard(**overrides: float) -> MetricScorecard:
    predictions, meta_model, benchmarks, features, targets = _synthetic_inputs()
    scorecard = evaluate_model(
        predictions, meta_model=meta_model, benchmarks=benchmarks,
        features=features, targets=targets, n_trials=1, seed=77,
        benchmark_col="bench", n_boot=50, min_overlap_eras=20,
        model_id="probe",
    )
    return dataclasses.replace(scorecard, **overrides)


def _null_scorecards() -> dict[str, MetricScorecard]:
    out = {}
    for kind in NULL_KINDS:
        out[kind] = _make_scorecard(model_id=kind)
    return out


def test_score_benchmark_column_wraps_predictions() -> None:
    _, _, benchmarks, _, _ = _synthetic_inputs()
    out = score_benchmark_column(benchmarks, column="bench")
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == benchmarks.height


def test_score_benchmark_column_unknown_column_raises() -> None:
    _, _, benchmarks, _, _ = _synthetic_inputs()
    with pytest.raises(ValueError, match="nope"):
        score_benchmark_column(benchmarks, column="nope")


def test_tier0_null_floor_passes_on_null_scorecards() -> None:
    assert_tier0_null_floor(_null_scorecards())


def test_tier0_null_floor_rejects_high_corr() -> None:
    cards = _null_scorecards()
    cards["null_constant_05"] = dataclasses.replace(
        cards["null_constant_05"],
        corr=dataclasses.replace(cards["null_constant_05"].corr, value=0.05),
    )
    with pytest.raises(ValueError, match="null_constant_05"):
        assert_tier0_null_floor(cards)


def test_tier0_null_floor_requires_all_four() -> None:
    cards = _null_scorecards()
    del cards["null_gaussian_rand"]
    with pytest.raises(ValueError, match="null_gaussian_rand"):
        assert_tier0_null_floor(cards)


def test_tier4_gate_passes_on_strong_scorecard() -> None:
    card = _make_scorecard(
        corr=dataclasses.replace(_make_scorecard().corr, value=0.04),
        corr_sharpe_ac=dataclasses.replace(_make_scorecard().corr_sharpe_ac, value=1.8),
        fnc=0.03,
        deflated_sharpe=1.2,
        gain_to_pain_ratio=2.0,
        cagr_1y=0.1,
        turnover_mean=0.1,
    )
    assert_tier4_gate(card, GATE)


def test_tier4_gate_reports_every_violation() -> None:
    card = _make_scorecard(
        corr=dataclasses.replace(_make_scorecard().corr, value=0.01),
        fnc=0.001,
        turnover_mean=0.9,
    )
    with pytest.raises(ValueError) as excinfo:
        assert_tier4_gate(card, GATE)
    message = str(excinfo.value)
    assert "corr" in message and "fnc" in message and "turnover" in message


def test_tier4_gate_missing_turnover_fails_loudly() -> None:
    card = _make_scorecard(turnover_mean=None, turnover_reason="no id column")
    with pytest.raises(ValueError, match="turnover"):
        assert_tier4_gate(card, GATE)


def test_monotone_ordering_passes_on_escalating_tiers() -> None:
    cards: dict[str, MetricScorecard] = {}
    tier_of: dict[str, int] = {}
    for tier, scalar in [(0, 0.0), (1, 0.2), (2, 0.4), (3, 0.6), (4, 0.7)]:
        model_id = f"t{tier}_probe"
        cards[model_id] = _make_scorecard(model_id=model_id, rank_scalar=scalar)
        tier_of[model_id] = tier
    assert_hierarchy_monotone(cards, tier_of=tier_of)


def test_monotone_rejects_inverted_tiers() -> None:
    cards: dict[str, MetricScorecard] = {}
    tier_of: dict[str, int] = {}
    for tier, scalar in [(0, 0.5), (1, 0.4), (2, 0.3), (3, 0.2), (4, 0.1)]:
        model_id = f"t{tier}_probe"
        cards[model_id] = _make_scorecard(model_id=model_id, rank_scalar=scalar)
        tier_of[model_id] = tier
    with pytest.raises(ValueError, match="monotone|ordering|tier"):
        assert_hierarchy_monotone(cards, tier_of=tier_of)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_gates_new.py -q`
Expected: FAIL — `ImportError` (functions not defined).

- [ ] **Step 3: Implement**

Append to `nmr/benchmark.py`:

```python
def score_benchmark_column(
    benchmarks: pl.DataFrame,
    *,
    column: str,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Wrap a benchmark-model column as a predictions frame."""
    if column not in benchmarks.columns:
        raise ValueError(f"Unknown benchmark column {column!r}")
    missing = [c for c in (era_col, id_col) if c not in benchmarks.columns]
    if missing:
        raise ValueError(f"benchmarks missing required columns: {missing}")
    return (
        benchmarks.select([era_col, id_col, pl.col(column).alias(pred_col)])
        .drop_nulls()
        .with_columns(pl.col(pred_col).cast(pl.Float64, strict=False))
        .filter(pl.col(pred_col).is_finite())
        .sort([era_col, id_col])
    )


def assert_tier0_null_floor(
    scorecards: Mapping[str, MetricScorecard],
    *,
    corr_tol: float = 0.005,
    sharpe_tol: float = 0.10,
    dsr_tol: float = 0.05,
) -> None:
    """Tier-0 sanity gate: null baselines must score at the statistical floor."""
    for name in NULL_KINDS:
        if name not in scorecards:
            raise ValueError(f"Missing null baseline scorecard {name!r}")

    for name in NULL_KINDS:
        score = scorecards[name]
        _assert_scorecard_finite(score, model_id=name)
        checks = (
            ("corr", float(score.corr.value), float(corr_tol)),
            ("corr_sharpe_ac", float(score.corr_sharpe_ac.value), float(sharpe_tol)),
            ("deflated_sharpe", float(score.deflated_sharpe), float(dsr_tol)),
        )
        for metric_name, observed, tolerance in checks:
            if abs(observed) > tolerance:
                raise ValueError(
                    "Null floor violation for "
                    f"{name}.{metric_name}: |{observed:.8f}| > {tolerance:.8f}"
                )


def assert_tier4_gate(scorecard: MetricScorecard, gate: Tier4GateConfig) -> None:
    """Production capital gate: reject candidates below the 7 hard thresholds."""
    _assert_scorecard_finite(scorecard, model_id=scorecard.model_id)
    violations: list[str] = []

    def _check(field: str, observed: float, threshold: float, strict: bool) -> None:
        if strict:
            if observed <= threshold:
                violations.append(
                    f"{field}: observed={observed:.8f}, need > {threshold:.8f}"
                )
        elif observed < threshold:
            violations.append(
                f"{field}: observed={observed:.8f}, need >= {threshold:.8f}"
            )

    if scorecard.turnover_mean is None:
        violations.append(
            f"turnover_mean: unavailable (reason={scorecard.turnover_reason!r}); "
            f"cannot verify <= {gate.turnover_max:.4f}"
        )
    else:
        if float(scorecard.turnover_mean) > float(gate.turnover_max):
            violations.append(
                "turnover_mean: "
                f"observed={float(scorecard.turnover_mean):.8f}, "
                f"need <= {gate.turnover_max:.4f}"
            )

    _check("corr", float(scorecard.corr.value), float(gate.corr_min), strict=False)
    _check(
        "corr_sharpe_ac",
        float(scorecard.corr_sharpe_ac.value),
        float(gate.corr_sharpe_ac_min),
        strict=False,
    )
    _check("fnc", float(scorecard.fnc), float(gate.fnc_min), strict=False)
    _check(
        "deflated_sharpe",
        float(scorecard.deflated_sharpe),
        float(gate.deflated_sharpe_min),
        strict=False,
    )
    _check(
        "gain_to_pain_ratio",
        float(scorecard.gain_to_pain_ratio),
        float(gate.gain_to_pain_min),
        strict=False,
    )
    _check("cagr_1y", float(scorecard.cagr_1y), float(gate.cagr_min), strict=True)

    if violations:
        raise ValueError(
            f"Tier-4 gate violations for {scorecard.model_id!r}: "
            + "; ".join(violations)
        )


def assert_hierarchy_monotone(
    scorecards: Mapping[str, MetricScorecard],
    *,
    tier_of: Mapping[str, int],
    atol: float = 1e-5,
) -> None:
    """Assert escalating tier ordering on the rank scalar (T0 < T1 < T2 < T3 <= T4)."""
    tiers_present = sorted(set(tier_of.values()))
    if tiers_present != [0, 1, 2, 3, 4]:
        raise ValueError(f"tier_of must cover all tiers 0..4, got {tiers_present}")

    scalar_by_tier: dict[int, float] = {}
    for tier in (0, 1, 2, 3, 4):
        members = [mid for mid, t in tier_of.items() if t == tier]
        if not members:
            raise ValueError(f"No scorecards for tier {tier}")
        missing = [mid for mid in members if mid not in scorecards]
        if missing:
            raise ValueError(f"Missing scorecards for tier {tier}: {missing}")
        scalar_by_tier[tier] = max(
            float(scorecards[mid].rank_scalar) for mid in members
        )

    for lower in (0, 1, 2):
        if scalar_by_tier[lower] + atol > scalar_by_tier[lower + 1]:
            raise ValueError(
                "Monotone violation: tier "
                f"{lower}={scalar_by_tier[lower]:.8f} not < tier "
                f"{lower + 1}={scalar_by_tier[lower + 1]:.8f} (atol={atol:.2e})"
            )
    if scalar_by_tier[3] > scalar_by_tier[4] + atol:
        raise ValueError(
            "Monotone violation: tier "
            f"3={scalar_by_tier[3]:.8f} not <= tier "
            f"4={scalar_by_tier[4]:.8f} (atol={atol:.2e})"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_gates_new.py -q`
Expected: PASS (10 tests). If `test_tier4_gate_passes_on_strong_scorecard` fails on `gain_to_pain_ratio` because `_make_scorecard` produces degenerate values on synthetic data, adjust the override values (they are explicit floats — raising them is a legitimate fix, not a code change).

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark.py tests/test_benchmark_gates_new.py
git commit -m "feat: benchmark gate functions (tier-0 floor, tier-4 thresholds, monotonicity)"
```

---

### Task 8: `BenchmarkHierarchy` orchestration + legacy removal + test-suite rewrite

**Files:**
- Modify: `nmr/benchmark.py` (remove all legacy S11 symbols; add hierarchy orchestration)
- Modify: `nmr/__init__.py` (exports)
- Delete: `tests/test_benchmark_baselines.py`, `tests/test_benchmark_gates.py`, `tests/test_benchmark_slice1.py`, `tests/test_benchmark_slice2.py`, `tests/test_benchmark_slice3.py`
- Create: `tests/test_benchmark_hierarchy.py`
- Modify: `tests/test_parity.py` (relocate BMC oracle parity test)
- Modify: `tests/test_scripts.py` (update the benchmark_runner surface stub)
- Modify: `tests/test_package_api.py` if it asserts benchmark exports (read it first)

**Interfaces:**
- Produces (used by Task 9):
  - `@dataclasses.dataclass(frozen=True) class BenchmarkData:` fields `meta_model: pl.DataFrame`, `benchmarks: pl.DataFrame`, `features_json: Path`, `train_path: Path`, `validation_path: Path`
  - `load_benchmark_data(data_dir: str | Path) -> BenchmarkData` (lazy: reads only `meta_model.parquet` + `validation_benchmark_models.parquet`; stores paths)
  - `resolve_benchmark_feature_cols(features_json: Path, input_space: str, available: Sequence[str]) -> list[str]`
  - `@dataclasses.dataclass(frozen=True) class BenchmarkHierarchyResult:` fields `scorecards: Mapping[str, MetricScorecard]`, `tier_of: Mapping[str, int]`, `gate: Tier4GateConfig | None`, `null_floor_ok: bool`, `null_floor_errors: tuple[str, ...]`, `tier4_violations: tuple[str, ...]`, `monotone_ok: bool`, `monotone_error: str | None`
  - `class BenchmarkHierarchy` — `__init__(self, *, spec: BenchmarkSuiteSpec, data: BenchmarkData, seed: int = DEFAULT_BENCHMARK_SEED, horizon: str = "20D", n_boot: int = 1000, min_overlap_eras: int = 20, fast_mode: bool = False)`; `run() -> BenchmarkHierarchyResult`
  - `hierarchy_frame(result: BenchmarkHierarchyResult) -> pl.DataFrame` — scorecard rows + `tier` + `strategy_group` (`f"tier{tier}"`) columns
  - `gate_report_frame(result: BenchmarkHierarchyResult) -> pl.DataFrame` — one row per tier-4 field: `model_id, field, threshold, measured, pass`

**Scoring details (fixed decisions for this task):**
- `evaluate_model` is called with `n_trials=1`, `seed=cell.seed`, `horizon=horizon`, `n_boot=n_boot`, `min_overlap_eras=min_overlap_eras`, `benchmark_col=spec.reference_column`, `model_id=cell.benchmark_id`.
- The `features` frame passed to `evaluate_model` is the **cell's own validation feature frame** (`[era, id, *feature_cols]`); for `input_space: none` cells the **small** feature frame is used so FNC stays well-defined; for the tier-4 reference the **medium** frame is used. FNE is therefore `FNC@medium` (full 3,555 is prohibited by the feature-universe policy — document this in Task 11).
- Null cells score on the validation prediction index. Tier 1–3 cells load per-cell column projections of `train_path`/`validation_path` via `pl.read_parquet(..., columns=[...])` (memory guard: one cell at a time, no full-frame residency).
- `run()` collects gate outcomes into the result without raising (so the runner can write outputs before exiting non-zero).

- [ ] **Step 1: Read the files that constrain the rewrite**

Read `tests/test_scripts.py` and `tests/test_package_api.py` (untracked). Note exactly what surfaces they assert. Then read `nmr/benchmark.py` from the top to confirm the legacy symbol list to delete:
`NULL_BASELINES`, `TUTORIAL_NOTEBOOK_TO_MODEL_ID`, `_TUTORIAL_NOTEBOOK_ANCHORS`, `_EvalConfig`, `BenchmarkSuite`, `iter_baseline_predictions`, `run_classical_baselines`, `compute_book_orthogonality`, `run_null_baselines`, `evaluate_predictions`, `evaluate_normalized_predictions`, `normalized_era_labels`, `normalize_predictions`, `evaluate_tutorial_predictions`, `null_prediction_frame`, `_resolve_join_keys`, `_normalize_predictions`, `_trivial_prediction_frame`, `_walk_forward_model_predictions`, `_build_classical_model`, `_as_finite_vector`, `_safe_corr`, `_orthogonality_stat`, `ingest_tutorial_prediction`, `ingest_tutorial_prediction_batch`, `discover_tutorial_notebooks`, `assert_notebook_prediction_contract`, `extract_oos_predictions`, `assert_null_floor`, `assert_slice1_monotone`, `_verify_notebook_contract`, `_resolve_notebook_path`, `_read_prediction_file`, `_infer_id_column`, `_infer_prediction_column`.
**Keep:** `scorecards_to_frame`, `write_scorecards_csv`, `canonical_scorecards_bytes`, `scorecards_sha256`, `_assert_scorecard_finite`, `_json_default`, `_sanitize_json_payload`, and everything added by Tasks 1–7.

- [ ] **Step 2: Remove the legacy symbols and add the orchestration layer**

Rewrite `nmr/benchmark.py` as: docstring + imports + Task 1 config section + Task 2 purge section + Task 3 null generator + Task 4 ridge generator + Task 5 tree generator + Task 6 canonical generator + Task 7 gate functions + kept scorecard helpers + the new code below. Imports needed at the top (prune unused ones — `Iterator`, `hashlib`, `json` stay for canonical bytes; `MappingProxyType`, `dataclasses`, `yaml` from Task 1; `numpy`, `polars`, `Ridge`, `Ensembler`, `NeutralizationEngine`, `evaluate_model`, `MetricScorecard`, `MIN_OVERLAP_ERAS`, `resolve_small_feature_set`, `resolve_feature_sets`, `construct_tree_model`):

```python
# ---------------------------------------------------------------------------
# Hierarchy orchestration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BenchmarkData:
    meta_model: pl.DataFrame
    benchmarks: pl.DataFrame
    features_json: Path
    train_path: Path
    validation_path: Path


def load_benchmark_data(data_dir: str | Path) -> BenchmarkData:
    """Load the lightweight shared domains; heavy parquets stay lazy."""
    directory = Path(data_dir)
    for name in ("meta_model.parquet", "validation_benchmark_models.parquet",
                 "features.json", "train.parquet", "validation.parquet"):
        if not (directory / name).exists():
            raise FileNotFoundError(f"Missing benchmark data asset: {directory / name}")
    meta_model = pl.read_parquet(directory / "meta_model.parquet").select(
        ["era", "id", "numerai_meta_model"]
    )
    benchmarks = pl.read_parquet(directory / "validation_benchmark_models.parquet")
    return BenchmarkData(
        meta_model=meta_model,
        benchmarks=benchmarks,
        features_json=directory / "features.json",
        train_path=directory / "train.parquet",
        validation_path=directory / "validation.parquet",
    )


def resolve_benchmark_feature_cols(
    features_json: Path,
    input_space: str,
    available: Sequence[str],
) -> list[str]:
    """Resolve feature columns for a benchmark input space, fail-loud."""
    if input_space not in VALID_INPUT_SPACES:
        raise ValueError(f"input_space={input_space!r} not in {VALID_INPUT_SPACES}")
    if input_space == "none":
        return []
    if input_space == "small":
        return resolve_small_feature_set(features_json, available)
    sets = resolve_feature_sets(features_json)
    if "medium" not in sets:
        raise ValueError("features.json has no 'medium' feature set")
    cols = [c for c in sets["medium"] if c in available]
    missing = sorted(set(sets["medium"]) - set(available))
    if missing:
        raise ValueError(
            f"{len(missing)} medium features missing from data columns: "
            f"{missing[:5]}..."
        )
    return cols


@dataclasses.dataclass(frozen=True)
class BenchmarkHierarchyResult:
    scorecards: Mapping[str, MetricScorecard]
    tier_of: Mapping[str, int]
    gate: Tier4GateConfig | None
    null_floor_ok: bool
    null_floor_errors: tuple[str, ...]
    tier4_violations: tuple[str, ...]
    monotone_ok: bool
    monotone_error: str | None


class BenchmarkHierarchy:
    """Config-driven 5-tier benchmark ladder (the line in the sand)."""

    def __init__(
        self,
        *,
        spec: BenchmarkSuiteSpec,
        data: BenchmarkData,
        seed: int = DEFAULT_BENCHMARK_SEED,
        horizon: str = "20D",
        n_boot: int = 1000,
        min_overlap_eras: int = 20,
        fast_mode: bool = False,
    ) -> None:
        if not spec.cells:
            raise ValueError("BenchmarkSuiteSpec has no cells")
        self._spec = spec
        self._data = data
        self._seed = int(seed)
        self._horizon = horizon
        self._n_boot = int(n_boot)
        self._min_overlap_eras = int(min_overlap_eras)
        self._fast_mode = bool(fast_mode)
        self._schema_cols = pl.read_parquet_schema(data.validation_path).names()

    def _feature_cols(self, cell: BenchmarkCellConfig) -> list[str]:
        return resolve_benchmark_feature_cols(
            self._data.features_json, cell.input_space, self._schema_cols
        )

    def _cell_params(self, cell: BenchmarkCellConfig) -> dict[str, Any]:
        params = dict(cell.params)
        if self._fast_mode and cell.fast_mode_params:
            params.update(dict(cell.fast_mode_params))
        return params

    def _domain_frames(
        self, cell: BenchmarkCellConfig, feature_cols: list[str]
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        id_era = ["era", "id"]
        train = pl.read_parquet(
            self._data.train_path,
            columns=[*id_era, *feature_cols, *cell.targets],
        )
        val = pl.read_parquet(
            self._data.validation_path,
            columns=[*id_era, *feature_cols],
        )
        return train, val

    def _predictions_for_cell(
        self, cell: BenchmarkCellConfig
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return (predictions, val_feature_frame) for one benchmark cell."""
        feature_cols = self._feature_cols(cell)
        val_id = pl.read_parquet(
            self._data.validation_path, columns=["era", "id"]
        )
        params = self._cell_params(cell)

        if cell.model_kind in NULL_KINDS:
            if cell.model_kind == "null_feature_mean":
                val_features = pl.read_parquet(
                    self._data.validation_path,
                    columns=["era", "id", *feature_cols],
                )
                preds = generate_null_predictions(
                    val_id, kind=cell.model_kind, seed=cell.seed,
                    features=val_features, feature_cols=feature_cols,
                )
            else:
                preds = generate_null_predictions(
                    val_id, kind=cell.model_kind, seed=cell.seed
                )
                small_cols = resolve_benchmark_feature_cols(
                    self._data.features_json, "small", self._schema_cols
                )
                val_features = pl.read_parquet(
                    self._data.validation_path,
                    columns=["era", "id", *small_cols],
                )
            return preds, val_features

        train, val = self._domain_frames(cell, feature_cols)
        if cell.model_kind == "ridge":
            preds = generate_ridge_predictions(
                train, val, targets=list(cell.targets),
                feature_cols=feature_cols,
                alpha=float(params.get("alpha", 1.0)),
                seed=cell.seed,
            )
        elif cell.model_kind == "lightgbm":
            preds = generate_canonical_predictions(
                train, val, targets=list(cell.targets),
                feature_cols=feature_cols, params=params, seed=cell.seed,
                neutralization=cell.neutralization,
            )
        elif cell.model_kind == "xgboost":
            preds = generate_tree_predictions(
                train, val, target=cell.targets[0],
                feature_cols=feature_cols, backend="xgboost",
                params=params, seed=cell.seed,
            )
        else:
            raise ValueError(f"Unsupported benchmark model kind: {cell.model_kind!r}")
        return preds, val

    def run(self) -> BenchmarkHierarchyResult:
        """Score every cell, the tier-4 reference, and all hard gates."""
        scorecards: dict[str, MetricScorecard] = {}
        tier_of: dict[str, int] = {}

        for cell in self._spec.cells:
            logger.info(
                "[hierarchy] tier %d: %s (kind=%s)", cell.tier,
                cell.benchmark_id, cell.model_kind,
            )
            preds, val_features = self._predictions_for_cell(cell)
            scorecards[cell.benchmark_id] = evaluate_model(
                preds,
                meta_model=self._data.meta_model,
                benchmarks=self._data.benchmarks,
                features=val_features,
                targets=pl.read_parquet(
                    self._data.validation_path,
                    columns=["era", "id", "target"],
                ),
                n_trials=1,
                seed=cell.seed,
                horizon=self._horizon,
                main_target="target",
                benchmark_col=self._spec.reference_column,
                n_boot=self._n_boot,
                min_overlap_eras=self._min_overlap_eras,
                model_id=cell.benchmark_id,
            )
            tier_of[cell.benchmark_id] = cell.tier

        reference_id = "v53_lgbm_ender60"
        if self._spec.reference_column:
            reference_id = self._spec.reference_column
            medium_cols = resolve_benchmark_feature_cols(
                self._data.features_json, "medium", self._schema_cols
            )
            ref_features = pl.read_parquet(
                self._data.validation_path,
                columns=["era", "id", *medium_cols],
            )
            ref_preds = score_benchmark_column(
                self._data.benchmarks, column=self._spec.reference_column
            )
            scorecards[reference_id] = evaluate_model(
                ref_preds,
                meta_model=self._data.meta_model,
                benchmarks=self._data.benchmarks,
                features=ref_features,
                targets=pl.read_parquet(
                    self._data.validation_path,
                    columns=["era", "id", "target"],
                ),
                n_trials=1,
                seed=self._seed,
                horizon=self._horizon,
                main_target="target",
                benchmark_col=self._spec.reference_column,
                n_boot=self._n_boot,
                min_overlap_eras=self._min_overlap_eras,
                model_id=reference_id,
            )
            tier_of[reference_id] = 4

        null_cards = {
            mid: scorecards[mid] for mid in NULL_KINDS if mid in scorecards
        }
        null_floor_ok, null_floor_errors = True, ()
        try:
            assert_tier0_null_floor(null_cards)
        except ValueError as exc:
            null_floor_ok, null_floor_errors = False, (str(exc),)

        tier4_violations: tuple[str, ...] = ()
        if self._spec.gate is not None and reference_id in scorecards:
            try:
                assert_tier4_gate(scorecards[reference_id], self._spec.gate)
            except ValueError as exc:
                tier4_violations = (str(exc),)

        monotone_ok, monotone_error = True, None
        try:
            assert_hierarchy_monotone(scorecards, tier_of=tier_of)
        except ValueError as exc:
            monotone_ok, monotone_error = False, str(exc)

        return BenchmarkHierarchyResult(
            scorecards=scorecards,
            tier_of=tier_of,
            gate=self._spec.gate,
            null_floor_ok=null_floor_ok,
            null_floor_errors=null_floor_errors,
            tier4_violations=tier4_violations,
            monotone_ok=monotone_ok,
            monotone_error=monotone_error,
        )


def hierarchy_frame(result: BenchmarkHierarchyResult) -> pl.DataFrame:
    """Scorecard rows with tier metadata (dashboard-compatible)."""
    frame = scorecards_to_frame(result.scorecards)
    tier_rows = pl.DataFrame(
        {
            "model_id": list(result.tier_of.keys()),
            "tier": [result.tier_of[mid] for mid in result.tier_of.keys()],
        }
    )
    frame = frame.join(tier_rows, on="model_id", how="left").with_columns(
        pl.col("tier").cast(pl.Int64).map_elements(
            lambda t: f"tier{int(t)}", return_dtype=pl.String
        ).alias("strategy_group")
    )
    return frame.sort(["tier", "model_id"])


def gate_report_frame(result: BenchmarkHierarchyResult) -> pl.DataFrame:
    """One row per tier-4 field: threshold vs measured."""
    gate = result.gate
    if gate is None:
        return pl.DataFrame(
            {"model_id": [], "field": [], "threshold": [], "measured": [], "pass": []}
        )
    reference_id = "v53_lgbm_ender60"
    for mid in result.tier_of:
        if result.tier_of[mid] == 4:
            reference_id = mid
            break
    card = result.scorecards[reference_id]
    rows = [
        ("corr", gate.corr_min, float(card.corr.value), False),
        ("corr_sharpe_ac", gate.corr_sharpe_ac_min,
         float(card.corr_sharpe_ac.value), False),
        ("fnc", gate.fnc_min, float(card.fnc), False),
        ("deflated_sharpe", gate.deflated_sharpe_min,
         float(card.deflated_sharpe), False),
        ("gain_to_pain_ratio", gate.gain_to_pain_min,
         float(card.gain_to_pain_ratio), False),
        ("cagr_1y", gate.cagr_min, float(card.cagr_1y), True),
        ("turnover_mean", gate.turnover_max,
         float(card.turnover_mean) if card.turnover_mean is not None else None,
         False),
    ]
    out_rows = []
    for field, threshold, measured, strict in rows:
        passed = (
            (measured is not None)
            and ((measured > threshold) if strict else (measured >= threshold))
        )
        out_rows.append({
            "model_id": reference_id,
            "field": field,
            "threshold": threshold,
            "measured": measured,
            "pass": passed,
        })
    return pl.DataFrame(out_rows)
```

- [ ] **Step 3: Update `nmr/__init__.py` exports**

Replace the `from .benchmark import (...)` block and its `__all__` entries with:

```python
from .benchmark import (
    BenchmarkCellConfig,
    BenchmarkData,
    BenchmarkFileConfig,
    BenchmarkHierarchy,
    BenchmarkHierarchyResult,
    BenchmarkSuiteSpec,
    Tier4GateConfig,
    VALID_BENCHMARK_TIERS,
    assert_hierarchy_monotone,
    assert_tier0_null_floor,
    assert_tier4_gate,
    canonical_scorecards_bytes,
    gate_report_frame,
    generate_canonical_predictions,
    generate_null_predictions,
    generate_ridge_predictions,
    generate_tree_predictions,
    hierarchy_frame,
    load_benchmark_data,
    load_benchmark_file,
    load_benchmark_suite_config,
    resolve_benchmark_feature_cols,
    score_benchmark_column,
    scorecards_sha256,
    scorecards_to_frame,
    train_validation_purged_split,
    write_scorecards_csv,
)
```

And mirror the same names in `__all__`.

- [ ] **Step 4: Rewrite the benchmark test files**

Delete `tests/test_benchmark_baselines.py`, `tests/test_benchmark_gates.py`, `tests/test_benchmark_slice1.py`, `tests/test_benchmark_slice2.py`, `tests/test_benchmark_slice3.py` (superseded by Tasks 1–7 test files + the new hierarchy file below). Then rename the new gates file so the final layout is clean:

```bash
git mv tests/test_benchmark_gates_new.py tests/test_benchmark_gates.py
```

Create `tests/test_benchmark_hierarchy.py`:

```python
"""End-to-end hierarchy orchestration, determinism, and monotonicity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import (
    BenchmarkCellConfig,
    BenchmarkData,
    BenchmarkHierarchy,
    BenchmarkSuiteSpec,
    Tier4GateConfig,
    gate_report_frame,
    hierarchy_frame,
    scorecards_sha256,
)


def _data_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(20260815)
    n_eras, rows_per_era = 60, 8
    rows = []
    for era_num in range(1, n_eras + 1):
        era = f"{era_num:04d}"
        for idx in range(rows_per_era):
            f1 = float(rng.normal())
            target = float(np.clip(0.5 + 0.2 * f1 + rng.normal(0, 0.3), 0, 1))
            rows.append({
                "era": era, "id": f"{era}_{idx}", "f1": f1,
                "target": target,
                "numerai_meta_model": float(0.5 * target + 0.5 * rng.random()),
                "bench": float(0.6 * target + 0.4 * rng.random()),
            })
    frame = pl.DataFrame(rows)

    train = frame.filter(pl.col("era").is_in([f"{e:04d}" for e in range(1, 41)]))
    val = frame.filter(pl.col("era").is_in([f"{e:04d}" for e in range(49, 61)]))
    (tmp_path / "train.parquet").mkdir(exist_ok=True)
    train.write_parquet(tmp_path / "train.parquet")
    val.write_parquet(tmp_path / "validation.parquet")
    val.select(["era", "id", "numerai_meta_model"]).write_parquet(
        tmp_path / "meta_model.parquet"
    )
    val.select(["era", "id", "bench"]).rename({"bench": "v53_lgbm_ender60"}).write_parquet(
        tmp_path / "validation_benchmark_models.parquet"
    )
    (tmp_path / "features.json").write_text(
        '{"feature_sets": {"small": ["f1"], "medium": ["f1"]}}',
        encoding="utf-8",
    )
    return tmp_path


def _spec() -> BenchmarkSuiteSpec:
    gate = Tier4GateConfig(
        corr_min=-1.0, corr_sharpe_ac_min=-10.0, fnc_min=-1.0,
        deflated_sharpe_min=-10.0, gain_to_pain_min=-10.0, cagr_min=-1.0,
        turnover_max=10.0,
    )
    cells = (
        BenchmarkCellConfig(
            benchmark_id="null_constant_05", input_space="none",
            model_kind="null_constant_05", tier=0,
        ),
        BenchmarkCellConfig(
            benchmark_id="linear_ridge_small", input_space="small",
            model_kind="ridge", tier=1,
            targets=("target",), params={"alpha": 1.0},
        ),
        BenchmarkCellConfig(
            benchmark_id="tree_lgbm_shallow_small", input_space="small",
            model_kind="lightgbm", tier=2,
            targets=("target",),
            params={"n_estimators": 5, "learning_rate": 0.1,
                    "max_depth": 2, "num_leaves": 4, "colsample_bytree": 0.5},
        ),
        BenchmarkCellConfig(
            benchmark_id="canon_hello_numerai", input_space="small",
            model_kind="lightgbm", tier=3,
            targets=("target",),
            params={"n_estimators": 5, "learning_rate": 0.1,
                    "max_depth": 2, "num_leaves": 4, "colsample_bytree": 0.5},
        ),
    )
    return BenchmarkSuiteSpec(
        cells=cells, gate=gate, reference_column="v53_lgbm_ender60"
    )


def _run(tmp_path: Path, *, seed: int = 42) -> BenchmarkHierarchy:
    from nmr.benchmark import load_benchmark_data
    data = load_benchmark_data(_data_dir(tmp_path))
    hierarchy = BenchmarkHierarchy(
        spec=_spec(), data=data, seed=seed, n_boot=50,
        min_overlap_eras=5, fast_mode=True,
    )
    return hierarchy


def test_hierarchy_runs_and_emits_frames(tmp_path: Path) -> None:
    result = _run(tmp_path).run()
    expected_ids = {
        "null_constant_05", "linear_ridge_small",
        "tree_lgbm_shallow_small", "canon_hello_numerai",
        "v53_lgbm_ender60",
    }
    assert set(result.scorecards) == expected_ids
    assert result.tier_of["v53_lgbm_ender60"] == 4
    frame = hierarchy_frame(result)
    assert frame.height == 5
    assert "strategy_group" in frame.columns
    assert set(frame.get_column("strategy_group").unique().to_list()) == {
        "tier0", "tier1", "tier2", "tier3", "tier4",
    }
    report = gate_report_frame(result)
    assert report.height == 7
    assert set(report.get_column("field").to_list()) == {
        "corr", "corr_sharpe_ac", "fnc", "deflated_sharpe",
        "gain_to_pain_ratio", "cagr_1y", "turnover_mean",
    }


def test_hierarchy_is_deterministic(tmp_path: Path) -> None:
    result_a = _run(tmp_path, seed=42).run()
    result_b = _run(tmp_path, seed=42).run()
    assert scorecards_sha256(result_a.scorecards) == scorecards_sha256(
        result_b.scorecards
    )


def test_hierarchy_cross_process_determinism(tmp_path: Path) -> None:
    data_dir = _data_dir(tmp_path)
    script = (
        "import os, sys; from pathlib import Path;"
        "from nmr.benchmark import load_benchmark_data, BenchmarkHierarchy, "
        "BenchmarkCellConfig, BenchmarkSuiteSpec, Tier4GateConfig, scorecards_sha256;"
        f"data = load_benchmark_data(Path(r'{data_dir}'));"
        "gate = Tier4GateConfig(corr_min=-1, corr_sharpe_ac_min=-10, fnc_min=-1, "
        "deflated_sharpe_min=-10, gain_to_pain_min=-10, cagr_min=-1, turnover_max=10);"
        "cells = (BenchmarkCellConfig('null_constant_05', 'none', "
        "'null_constant_05', 0), "
        "BenchmarkCellConfig('linear_ridge_small', 'small', 'ridge', 1, "
        "targets=('target',), params={'alpha': 1.0}));"
        "spec = BenchmarkSuiteSpec(cells=cells, gate=gate, "
        "reference_column='v53_lgbm_ender60');"
        "h = BenchmarkHierarchy(spec=spec, data=data, seed=42, n_boot=50, "
        "min_overlap_eras=5, fast_mode=True);"
        "print(scorecards_sha256(h.run().scorecards))"
    )
    env = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    run = lambda: subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=env, cwd=Path.cwd(), check=True,
    ).stdout.strip()
    assert run() == run()


def test_monotone_failure_surfaces_in_result(tmp_path: Path) -> None:
    hierarchy = _run(tmp_path)
    result = hierarchy.run()
    # The four real tiers may or may not order monotonically on this synthetic
    # data; the result must carry verdicts either way (no raise).
    assert isinstance(result.monotone_ok, bool)
    assert isinstance(result.null_floor_ok, bool)
    assert result.tier4_violations == ()
```

Note: `min_overlap_eras=5` is test-only (synthetic data has 12 validation eras); production stays at 20.

- [ ] **Step 5: Relocate the BMC oracle parity test**

Move `test_slice3_bmc_oracle_parity` from the deleted `tests/test_benchmark_slice3.py` into `tests/test_parity.py`: copy the test function and its `_slice3_inputs`/`_suite`-style helpers verbatim (adapting only imports so they reference `nmr.scorecard.EvaluationEngine` directly and drop `BenchmarkSuite` usage). Run:

Run: `./.venv/Scripts/python -m pytest tests/test_parity.py -q`
Expected: PASS.

- [ ] **Step 6: Update `tests/test_scripts.py` and `tests/test_package_api.py`**

`tests/test_scripts.py`: replace the `iter_baseline_predictions` stub test with an import-surface test for the new runner (Task 9 defines the runner's `main`/`_parse_args`; for now assert `import benchmark_runner` succeeds and `benchmark_runner._parse_args` exists — final CLI assertions land in Task 9).

`tests/test_package_api.py`: read it; replace any benchmark-symbol assertions with the new `nmr/__init__.py` surface from Step 3.

- [ ] **Step 7: Run the full fast gate**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS (full suite; note any failures and fix before committing — do not proceed with red tests).

- [ ] **Step 8: Commit**

```bash
git add nmr/benchmark.py nmr/__init__.py tests/test_benchmark_hierarchy.py tests/test_benchmark_gates.py tests/test_parity.py tests/test_scripts.py tests/test_package_api.py
git rm tests/test_benchmark_baselines.py tests/test_benchmark_slice1.py tests/test_benchmark_slice2.py tests/test_benchmark_slice3.py
git commit -m "feat: BenchmarkHierarchy engine replaces legacy S11 suite (tier orchestration + gates)"
```

---

### Task 9: Runner rewrite (`benchmark_runner.py`)

**Files:**
- Modify: `benchmark_runner.py` (full rewrite — thin control plane only)
- Modify: `tests/test_scripts.py` (CLI surface tests)

**Interfaces:**
- Produces (used by Tasks 10–11): CLI `--data-dir`, `--configs`, `--output`, `--gate-report`, `--seed`, `--n-boot`, `--min-overlap-eras`, `--horizon`, `--log-level`, `--fast-mode`; `main() -> int` (exit 1 on null-floor or tier-4 violation; monotonicity is a hard gate in full mode, a logged warning in `--fast-mode` because fast params degrade tiers 2–3 by design).

- [ ] **Step 1: Write the failing CLI test**

Append to `tests/test_scripts.py`:

```python
def test_benchmark_runner_cli_defaults() -> None:
    import benchmark_runner

    args = benchmark_runner._parse_args_with(
        ["--data-dir", "data/v5.3", "--seed", "42", "--n-boot", "1000"]
    )
    assert args.seed == 42
    assert args.n_boot == 1000
    assert args.output.name == "benchmark_hierarchy_scorecard.csv"
    assert "reports" in args.output.parts
    assert args.configs.name == "benchmarks"
    assert args.fast_mode is False


def test_benchmark_runner_cli_fast_mode_and_horizon() -> None:
    import benchmark_runner

    args = benchmark_runner._parse_args_with(["--fast-mode", "--horizon", "60D"])
    assert args.fast_mode is True
    assert args.horizon == "60D"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_scripts.py -q`
Expected: FAIL — `AttributeError: module 'benchmark_runner' has no attribute '_parse_args_with'`.

- [ ] **Step 3: Rewrite `benchmark_runner.py`**

Replace the whole file with:

```python
"""Control plane for the 5-tier benchmark hierarchy (the line in the sand).

Thin wrapper only: argument parsing, data wiring, output writing, exit codes.
All benchmark logic lives in ``nmr.benchmark``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from nmr.benchmark import (
    BenchmarkHierarchy,
    gate_report_frame,
    hierarchy_frame,
    load_benchmark_data,
    load_benchmark_suite_config,
)


def _min_one_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("n-boot must be >= 1")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic 5-tier benchmark hierarchy runner."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "v5.3")
    parser.add_argument(
        "--configs", type=Path, default=Path("configs") / "benchmarks"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts")
        / "reports"
        / "benchmark_hierarchy_scorecard.csv",
    )
    parser.add_argument(
        "--gate-report",
        type=Path,
        default=Path("artifacts") / "reports" / "benchmark_gate_report.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=_min_one_int, default=1000)
    parser.add_argument("--min-overlap-eras", type=int, default=20)
    parser.add_argument("--horizon", choices=("20D", "60D"), default="20D")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument("--fast-mode", action="store_true")
    return parser


def _parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


def _parse_args_with(argv: list[str]) -> argparse.Namespace:
    """Test hook: parse an explicit argument vector."""
    return _build_parser().parse_args(argv)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("benchmark_runner")

    log.info("Loading benchmark suite config from %s", args.configs)
    spec = load_benchmark_suite_config(args.configs)
    log.info("Loading benchmark data from %s", args.data_dir)
    data = load_benchmark_data(args.data_dir)

    hierarchy = BenchmarkHierarchy(
        spec=spec,
        data=data,
        seed=args.seed,
        horizon=args.horizon,
        n_boot=1 if args.fast_mode else args.n_boot,
        min_overlap_eras=args.min_overlap_eras,
        fast_mode=args.fast_mode,
    )

    t0 = time.perf_counter()
    log.info(
        "Running %d benchmark cells%s",
        len(spec.cells),
        " (fast mode)" if args.fast_mode else "",
    )
    result = hierarchy.run()
    log.info("Hierarchy scored in %.1fs", time.perf_counter() - t0)

    for path in (args.output, args.gate_report):
        path.parent.mkdir(parents=True, exist_ok=True)

    hierarchy_frame(result).write_csv(args.output)
    gate_report_frame(result).write_csv(args.gate_report)
    log.info("Scorecard frame written to %s", args.output)
    log.info("Gate report written to %s", args.gate_report)

    for row in gate_report_frame(result).iter_rows(named=True):
        log.info(
            "tier4 gate %s: measured=%s threshold=%s pass=%s",
            row["field"], row["measured"], row["threshold"], row["pass"],
        )

    hard_failures: list[str] = []
    if not result.null_floor_ok:
        hard_failures.extend(result.null_floor_errors)
    hard_failures.extend(result.tier4_violations)
    if not result.monotone_ok:
        if args.fast_mode:
            log.warning(
                "Monotonicity not enforced in fast mode (degraded tier params): %s",
                result.monotone_error,
            )
        else:
            hard_failures.append(result.monotone_error or "monotone failure")

    if hard_failures:
        for message in hard_failures:
            log.error("GATE FAILURE: %s", message)
        return 1

    log.info("All hard gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_scripts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmark_runner.py tests/test_scripts.py
git commit -m "feat: rewrite benchmark_runner.py as hierarchy control plane"
```

---

### Task 10: Verification — full fast gate + real-data smoke

**Files:**
- No code changes expected. Possibly `configs/benchmarks/tier4_gate.yaml` (evidence-backed re-pin only).

- [ ] **Step 1: Run the full test suite and record the count**

Run: `./.venv/Scripts/python -m pytest -q 2>&1 | tail -5`
Expected: PASS, and record the collected test count (`N passed`) — this number goes into `AGENTS.md` in Task 11.

- [ ] **Step 2: Launch the real-data smoke in the background**

Run (background, `disable_timeout`, log file):

```bash
nohup ./.venv/Scripts/python benchmark_runner.py \
  --data-dir data/v5.3 \
  --configs configs/benchmarks \
  --seed 42 \
  --n-boot 1000 \
  --fast-mode \
  --output artifacts/reports/benchmark_hierarchy_scorecard_smoke.csv \
  --gate-report artifacts/reports/benchmark_gate_report_smoke.csv \
  > artifacts/benchmark_hierarchy_smoke.log 2>&1 &
```

- [ ] **Step 3: Poll the log until completion**

Check: `tail -20 artifacts/benchmark_hierarchy_smoke.log` — wait for either `All hard gates passed.` or `GATE FAILURE:` lines plus the final exit. Expect ~10–40 minutes (fast-mode tiers 1–3; tier-4 reference scoring is data-only, unaffected by fast params).

- [ ] **Step 4: Verify outputs**

```bash
head -3 artifacts/reports/benchmark_hierarchy_scorecard_smoke.csv
cat artifacts/reports/benchmark_gate_report_smoke.csv
```

Expected: scorecard CSV has one row per benchmark id (13 cells + `v53_lgbm_ender60`) with `tier` + `strategy_group` columns; gate report has 7 rows with measured vs threshold and pass flags.

- [ ] **Step 5: Evidence-backed threshold re-pin (only if needed)**

Compare the gate report's measured values for `v53_lgbm_ender60` against `configs/benchmarks/tier4_gate.yaml`. The tier-4 point estimates are valid in fast mode (only tier 1–3 fits degrade; CI widths shrink with `n_boot=1` but point values are unchanged). Per design decision #2:
- If every field passes → no change.
- If a field fails (measured < threshold) → the spec's absolute number does not match v5.3 reality. Edit only the failing threshold(s) in `tier4_gate.yaml` to the measured value, then commit:

```bash
git add configs/benchmarks/tier4_gate.yaml artifacts/reports/benchmark_gate_report_smoke.csv
git commit -m "bench: re-pin tier-4 gate thresholds to measured v5.3 values (evidence: benchmark_hierarchy_smoke.log)"
```

- [ ] **Step 6: Record results for the final report**

Copy these into the completion summary: smoke runtime, measured tier-4 values per field, null-floor verdict, monotonicity verdict (logged warning in fast mode), and whether any threshold was re-pinned.

---

### Task 11: Documentation, SSOT, dashboard paths, and stale-artifact removal

**Files:**
- Modify: `docs/06-evaluation/benchmark-line-in-the-sand.md` (full rewrite)
- Modify: `docs/06-evaluation/evaluation-suite-bible.md` (§11 E6, §15 ledger)
- Modify: `ARCHITECTURE.md` (§M, §O benchmark row, module dep graph, config/assets section)
- Modify: `AGENTS.md` (identity line, toolkit rows, gates 2–3, hazards, test count)
- Modify: `dashboard_app.py`, `generate_dashboard.py` (artifact path + strategy groups)
- Modify: `nmr/analysis.py` (docstring reference)
- Modify: `.kimi-code/skills/**` (any benchmark command references — grep first)
- Modify: `README.md`, `CONTRIBUTING.md`, `docs/DOCS_README.md` (only if grep finds benchmark references)
- Delete: `configs/campaigns/benchmark-rebuild-v1/` and stale benchmark artifacts (exact list below)

- [ ] **Step 1: Grep for every reference that must move**

```bash
grep -rn "benchmark_scores\|BenchmarkSuite\|slice1\|S11\|iter_baseline_predictions\|benchmark-rebuild" \
  --include="*.py" --include="*.md" --include="*.yaml" \
  .kimi-code README.md CONTRIBUTING.md docs/DOCS_README.md docs/06-evaluation \
  ARCHITECTURE.md AGENTS.md dashboard_app.py generate_dashboard.py nmr | grep -v ".git/" | grep -v "docs/superpowers"
```

Update every hit per the sections below.

- [ ] **Step 2: Rewrite `docs/06-evaluation/benchmark-line-in-the-sand.md`**

Replace the file content with:

```markdown
# Benchmark "Line in the Sand" — The 5-Tier Hierarchy

> **Purpose of this file:** a standing memory aid for the tiered benchmark floor every model must clear before capital deployment. The authoritative spec is the evaluation bible (`evaluation-suite-bible.md`, §11 E6 gate) and the design spec `docs/superpowers/specs/2026-08-15-benchmark-hierarchy-design.md`. If a tier, gate, or threshold changes, change it in the bible first, then here.

## 1) The one idea

Tiers 0–3 exist so a candidate's scorecard can be read as a rung on a ladder; Tier 4 is the production gate. Every rung emits a complete scorecard through `evaluate_model()`, so the ladder is directly comparable row-for-row with a real candidate.

| Tier | Rungs | Role |
| --- | --- | --- |
| 0 | constant-0.5, uniform-random, gaussian-random (clipped), feature-mean (small) | statistical zero-floor; a candidate indistinguishable from tier 0 is defective |
| 1 | Ridge small / medium / 4-target blend (purged, standardized) | linear factor frontier; non-linear models must beat it |
| 2 | shallow LightGBM/XGBoost + canonical fast preset | depth/interaction hurdle |
| 3 | hello-numerai, neutralized-50, sunshine 4×20D ensemble (in-process re-fits) | canonical community references |
| 4 | `v53_lgbm_ender60` benchmark column | the line in the sand for capital |

## 2) Hard gates (enforced by `nmr/benchmark.py`)

- **G — Tier-0 null floor:** |CORR| ≤ 0.005, |AC-Sharpe| ≤ 0.10, |DSR| ≤ 0.05 for every tier-0 rung. A null scoring above its floor means a broken metric.
- **G — Tier-4 production gate:** measured on `v53_lgbm_ender60` over the shared meta-model overlap window — CORR ≥ 0.0286, AC-Sharpe ≥ 1.50, FNC@medium ≥ 0.020, DSR ≥ 0.95, GPR ≥ 1.50, CAGR > 0, turnover ≤ 0.35. Thresholds live in `configs/benchmarks/tier4_gate.yaml` and are re-pinned to measured v5.3 values with evidence when they deviate.
- **G — Monotonicity:** rank scalar orders Tier0 < Tier1 < Tier2 < Tier3 ≤ Tier4 (per-tier max, atol 1e-5). Enforced in full runs; logged-only in `--fast-mode` (fast tree params degrade tiers 2–3 by design).
- **G — Determinism:** same data-version + seed + configs ⇒ identical scorecard hashes across processes (`scorecards_sha256`).

## 3) Fit topology (leakage rules)

Tiers 1–3 fit on `train.parquet` eras minus the final 8 (purge buffer) and predict `validation.parquet` eras. The split is asserted by `train_validation_purged_split()`: strict chronological ordering, exact 8-era buffer, numeric era labels only. Ridge features are standardized with trimmed-train statistics (zero-variance features → 0.0). Multi-target blends are equal-weight in the per-era rank-Gaussian domain (`Ensembler`), then re-gaussianized.

## 4) Tier anchors (report-only reference lines)

Tier 1–3 `anchors` in the YAMLs (e.g. CORR ≈ 0.0145 ridge-medium, ≈ 0.0210 fast-medium tree) are sanity reference lines logged against measured values — they are **not** enforced gates. The only enforced absolute numbers are the tier-0 floor and the tier-4 thresholds.

## 5) Notes & deviations

- **FNE is FNC@medium (780),** not the full 3,555 universe: full-universe validation FNC is memory-prohibited by the feature-universe policy. The tier-4 gate field `fnc_min` is measured against medium.
- Tier-4 point estimates are identical between fast and full modes (the reference is a data column); only tier 1–3 rungs degrade in fast mode.
- Run: `python benchmark_runner.py --data-dir data/v5.3 --seed 42 --n-boot 1000` (full) or `--fast-mode` (smoke). Outputs: `artifacts/reports/benchmark_hierarchy_scorecard.csv` + `benchmark_gate_report.csv`.
```

- [ ] **Step 3: Update the evaluation bible**

Read `docs/06-evaluation/evaluation-suite-bible.md` lines 424–431 (§11 E6) and 493–504 (§15 ledger). Replace the E6 description text with:

```markdown
### E6 — Benchmark Hierarchy Gate *(needs E5)*
The 5-tier benchmark ladder (`nmr/benchmark.py` `BenchmarkHierarchy`, configs in `configs/benchmarks/`): tier-0 null floor, tier-1 purged Ridge, tier-2 shallow trees, tier-3 canonical community models, tier-4 `v53_lgbm_ender60` production gate. Hard gates: tier-0 null floor, tier-4 thresholds, tier monotonicity; determinism via `scorecards_sha256`. Spec: `docs/06-evaluation/benchmark-line-in-the-sand.md`.
```

And in §15 replace any rows referencing the S11 ladder, tutorial ingestion, or slice1–3 tests with a single row:

```markdown
- **Benchmark hierarchy:** replaced the S11 ladder; tutorial-notebook ingestion and walk-forward classical baselines removed; tier ordering and tier-4 thresholds enforced by `BenchmarkHierarchy` (see §11 E6).
```

- [ ] **Step 4: Update `ARCHITECTURE.md`**

Replace section `### M. Benchmark Harness — nmr/benchmark.py` (lines 227–238) with:

```markdown
### M. Benchmark Hierarchy — nmr/benchmark.py

The 5-tier escalating benchmark ladder ("the line in the sand"). Config-driven:
`configs/benchmarks/*.yaml` → `load_benchmark_suite_config()` → frozen
`BenchmarkCellConfig` / `Tier4GateConfig` dataclasses (unknown keys rejected,
enum-validated). `BenchmarkHierarchy.run()` scores every cell plus the
`v53_lgbm_ender60` reference column through `evaluate_model()` and evaluates
three hard gates: `assert_tier0_null_floor` (|CORR| ≤ 0.005, |AC-Sharpe| ≤ 0.10,
|DSR| ≤ 0.05), `assert_tier4_gate` (7 production thresholds), and
`assert_hierarchy_monotone` (rank-scalar tier ordering, atol 1e-5). Tier 1–3 fits
use `train_validation_purged_split()` (exact 8-era buffer, strict ordering);
multi-target blends are equal-weight in the rank-Gaussian domain (`Ensembler`);
tier-3 neutralization reuses `NeutralizationEngine`; tree params resolve through
`nmr.models.construct_tree_model` (colsample floor, determinism flags).
Determinism: `scorecards_sha256` (timing fields stripped). FNE is FNC@medium
(full 3,555 is prohibited by the feature-universe policy).
```

Update the `benchmark_runner.py` row in `### O. Control-Plane Scripts` (line 276) to:

```markdown
| `benchmark_runner.py` | 5-tier hierarchy control plane: `--data-dir`, `--configs`, `--seed`, `--n-boot`, `--fast-mode`; writes `artifacts/reports/benchmark_hierarchy_scorecard.csv` + `benchmark_gate_report.csv`; exit 1 on hard-gate failure |
```

Update the module dependency graph row (line 406) to:

```markdown
benchmark.py ──> scorecard, models, features, risk, ensemble, inference
```

In `## 4. Configuration & Data Registry`, update the benchmark command (line 450) to `python benchmark_runner.py [--fast-mode] --output artifacts/reports/benchmark_hierarchy_scorecard.csv` and add a `configs/benchmarks/` inventory bullet: 8 tier YAMLs (tier0_null … tier4_gate) validated by `load_benchmark_suite_config`.

- [ ] **Step 5: Update `AGENTS.md`**

Apply these exact edits (line numbers refer to the current file; re-grep if shifted):
- §1 identity: `pytest (669 tests)` → `pytest (N tests)` with the count measured in Task 10 Step 1.
- Toolkit row (was line 142): `| Change benchmark baselines / gates | ... |` → `| Change the benchmark hierarchy (cells, gates, thresholds) | `nmr/benchmark.py` + `configs/benchmarks/` + `benchmark_runner.py` |`
- Knowledge-base row (was line 164): `(null floor + S11 ladder)` → `(5-tier hierarchy: tiers 0–4, hard gates)`
- Gate 2 (was line 194): `determinism hashes (`tests/test_benchmark_slice1.py`)` → `determinism hashes (`tests/test_benchmark_hierarchy.py`)`
- Gate 3 (was line 195): rewrite to `3. **Pre-sign-off gate** (mandatory before delivering work) — full N-test suite plus the real-data benchmark smoke (`benchmark_runner.py --fast-mode` → `artifacts/reports/benchmark_hierarchy_scorecard_smoke.csv` + `benchmark_gate_report_smoke.csv`).`
- Timing-field hazard (was line 208): replace `(test_benchmark_slice1.py, test_benchmark_slice3.py, test_scorecard.py)` with `(tests/test_benchmark_hierarchy.py, tests/test_scorecard.py)`.
- Benchmark parquet gap hazard (was line 211): append `The benchmark hierarchy reads validation_benchmark_models.parquet only (tier 4 reference); tiers 1–3 fit their own models on train.parquet.`
- Add a new hazard bullet under §8: `**Benchmark hierarchy runtime:** full hierarchy is multi-hour (medium tree fits on ~2.1M train rows). Use --fast-mode for smoke; FNE gate is FNC@medium per the feature-universe policy.`
- §10 timeouts (was line 267): keep `--fast-mode` wording (still accurate).

- [ ] **Step 6: Dashboard + script paths**

In `dashboard_app.py` and `generate_dashboard.py`, replace the default scorecard path `artifacts/benchmark_scores.csv` with `artifacts/reports/benchmark_hierarchy_scorecard.csv` (fall back to the legacy path if missing, so old artifacts still load), and replace strategy-group literals (`null`, `classical`, `benchmark_model`, `tutorial`) with `tier0`–`tier4`. In `nmr/analysis.py` line ~1437, replace the `BenchmarkSuite` docstring mention with `BenchmarkHierarchy`.

- [ ] **Step 7: Skills and remaining docs**

Apply Step 1's grep results: update any `.kimi-code/skills/**` command references (e.g. the verification-before-claim smoke command) to the new runner flags/outputs; update `README.md`/`CONTRIBUTING.md`/`docs/DOCS_README.md` only where the grep found benchmark references.

- [ ] **Step 8: Remove stale benchmark configs and artifacts**

```bash
git rm -r configs/campaigns/benchmark-rebuild-v1
```

Then inspect what is tracked before deleting artifacts:

```bash
git ls-files artifacts | grep -i "benchmark\|rebuild" 
ls artifacts/campaigns/
```

Delete only benchmark-related files: `artifacts/benchmark_scores.csv`, `artifacts/benchmark_scores_smoke.csv`, `artifacts/benchmark_full_run.log`, any `artifacts/benchmark_test_era_labels*.csv`, and the `rebuild_v53*.log` / `_step2_evidence.*` files under `artifacts/campaigns/`. For the campaign JSON files, open each and confirm its `model_id`/cell ids are `lgbm_v1..v6`/`xgb_v1..v6` (the rebuild campaign) before removing. Do **not** touch `artifacts/registry/` or `artifacts/runs/`.

- [ ] **Step 9: Final full-suite re-run and commit**

Run: `./.venv/Scripts/python -m pytest -q 2>&1 | tail -3`
Expected: PASS with the same count as Task 10. Then:

```bash
git add docs/06-evaluation/benchmark-line-in-the-sand.md docs/06-evaluation/evaluation-suite-bible.md ARCHITECTURE.md AGENTS.md dashboard_app.py generate_dashboard.py nmr/analysis.py .kimi-code/skills README.md CONTRIBUTING.md docs/DOCS_README.md
git commit -m "docs: SSOT updates for the 5-tier benchmark hierarchy + stale benchmark removal"
```

If Task 10 re-pinned `tier4_gate.yaml`, include that file reference in the commit message body.
