# Untiered Benchmark Fleet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an untiered fleet of 19 deterministic benchmark models (silly heuristics → Finance Arena recreations) to the `nmr` benchmark system, scored through the existing evaluation pipeline and placed against the 5-tier ladder by measured performance.

**Architecture:** A new `nmr/benchmark_fleet.py` module holds the fleet config schema, five prediction generators (`target_lag_mean`, fleet `lightgbm`, fleet `xgboost`, `mlp`, `ridge_stack` fixed/search), and a `BenchmarkFleet` runner mirroring `BenchmarkHierarchy`. The existing hierarchy is untouched except for two additive refactors: `tier4_gate_verdict`/`tier_max_corrs` helpers and an optional `fleet_scorecards` parameter on `canonical_scorecards_bytes`. Configs live in `configs/benchmarks/fleet/` (no `tier` field).

**Tech Stack:** Python 3.11+, Polars, NumPy, scikit-learn (Ridge, MLPRegressor, StandardScaler path via `_standardize_feature_block`), LightGBM/XGBoost via `construct_tree_model`, `NeutralizationEngine`, `Ensembler`, `evaluate_model`, pytest + ruff.

**Spec:** `docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md` — the single source of truth for roster, params, and semantics.

## Global Constraints

- All business logic in `nmr/`; `benchmark_runner.py` stays a thin control plane. Tests in `tests/`; scripts/notebooks contain zero logic.
- Determinism: same config + data + seed ⇒ identical outputs; no wall-clock or absolute paths in canonical hashes; tree fits `n_jobs=1` (via `construct_tree_model`); per-device determinism only (everything here is CPU).
- Oracle parity: neutralization ONLY via `nmr.risk.NeutralizationEngine` (never hand-rolled).
- Leakage rules: benchmark cells use `train_validation_purged_split` (8-era purge) for the train→validation boundary; `ridge_stack` uses a 16-era purge at its internal specialist/meta boundary when any 60D specialist is present, else 8.
- Rank-domain ensembling: per-component rank-Gaussianize → blend → re-Gaussianize (`Ensembler`).
- Fail loudly: unknown config keys/sections rejected; degenerate inputs raise `ValueError`; no silent fallbacks; no hidden defaults.
- No magic values: closed sets in module-level tuples (`VALID_FLEET_MODEL_KINDS`, `VALID_FLEET_NEUTRALIZATION`, `VALID_FLEET_NEUTRALIZER_SELECTIONS`).
- Never touch `../numer-AI/` (read-only source, mined for params already captured in the spec).
- Test commands: `./.venv/Scripts/python -m pytest ...` (the `Scripts/pip` shim is poisoned; use `python -m pip` if ever needed).
- Coverage commands use package-level specs only (`--cov=nmr`).
- `embargo_eras` must stay 0 in every config we write; `purge_eras` is the active buffer.

## File Structure

**Create:**
- `nmr/benchmark_fleet.py` — fleet config schema (`FleetCellConfig`, `FleetFileConfig`, `load_fleet_config`, `load_fleet_suite_config`), generators (`generate_lagged_target_predictions`, `generate_fleet_lightgbm_predictions`, `generate_fleet_xgb_predictions`, `generate_mlp_predictions`, `generate_ridge_stack_predictions`), runner (`BenchmarkFleet`, `FleetResult`, `fleet_frame`, `fleet_placement`, `write_fleet_csv`)
- `configs/benchmarks/fleet/fleet_silly.yaml` — 1 cell
- `configs/benchmarks/fleet/fleet_tutorials.yaml` — 5 cells
- `configs/benchmarks/fleet/fleet_community.yaml` — 6 cells
- `configs/benchmarks/fleet/fleet_finance_arena.yaml` — 7 cells
- `tests/test_benchmark_fleet.py` — all fleet tests (config, generators, runner, placement, determinism)

**Modify:**
- `nmr/benchmark.py` — add `tier4_gate_verdict`, `tier_max_corrs`; refactor `assert_tier4_gate`/`assert_hierarchy_monotone`/`gate_report_frame` to reuse them; extend `canonical_scorecards_bytes(scorecards, fleet_scorecards=None)`
- `nmr/models.py` — `construct_tree_model(..., extra_params=None)` (merged into resolved params after colsample flooring; needed for `early_stopping_rounds`)
- `benchmark_runner.py` — `--fleet-configs`, `--fleet-output`, `--no-fleet` CLI wiring
- `nmr/__init__.py` — export the new public symbols
- `tests/test_benchmark_gates.py` — verdict/rungs tests
- `tests/test_benchmark_hierarchy.py` — canonical-bytes test extended with fleet scorecards
- `docs/06-evaluation/benchmark-line-in-the-sand.md`, `docs/superpowers/specs/2026-08-15-benchmark-hierarchy-design.md` (one-line amendment), `ARCHITECTURE.md`, `AGENTS.md` — spec §9

---

### Task 1: Gate helpers — `tier4_gate_verdict` + `tier_max_corrs`

**Files:**
- Modify: `nmr/benchmark.py` (gate area, ~line 752; monotonicity, ~line 810; gate_report_frame, ~line 1272)
- Modify: `nmr/__init__.py` (import + `__all__`)
- Test: `tests/test_benchmark_gates.py`

**Interfaces:**
- Produces:
  - `tier4_gate_verdict(scorecard: MetricScorecard, gate: Tier4GateConfig) -> dict[str, bool | None]` — per-threshold verdict; `None` = structurally unavailable or display-only
  - `tier_max_corrs(scorecards: Mapping[str, MetricScorecard], tier_of: Mapping[str, int]) -> dict[int, float]` — per-tier max of `corr.value`
  - (internal) `_tier4_gate_rows(scorecard, gate) -> list[tuple[str, float | None, float, bool | None]]` — `(field, observed, threshold, strict)`; `strict=None` marks display-only
- Consumes: `MetricScorecard` fields `corr.value`, `corr_sharpe_ac.value`, `fnc`, `gain_to_pain_ratio`, `cagr_1y`, `turnover_mean`, `deflated_sharpe` (all read as floats)

- [ ] **Step 1: Write failing tests** — append to `tests/test_benchmark_gates.py`:

```python
from nmr.benchmark import Tier4GateConfig, tier4_gate_verdict, tier_max_corrs


def _gate() -> Tier4GateConfig:
    return Tier4GateConfig(
        corr_min=0.0286, corr_sharpe_ac_min=0.78, fnc_min=0.020,
        deflated_sharpe_min=0.95, gain_to_pain_min=1.50,
        cagr_min=0.0, turnover_max=0.35,
    )


def test_tier4_gate_verdict_shape_and_pass():
    card = make_scorecard(corr=0.03, sharpe=0.8, fnc=0.02, gpr=1.5, cagr=0.01, turnover=0.1)
    verdict = tier4_gate_verdict(card, _gate())
    assert verdict == {
        "corr": True, "corr_sharpe_ac": True, "fnc": True,
        "deflated_sharpe": None, "gain_to_pain_ratio": True,
        "cagr_1y": True, "turnover_mean": True,
    }


def test_tier4_gate_verdict_cagr_strict_and_turnover_none():
    card = make_scorecard(corr=0.03, sharpe=0.8, fnc=0.02, gpr=1.5, cagr=0.0, turnover=None)
    verdict = tier4_gate_verdict(card, _gate())
    assert verdict["cagr_1y"] is False          # strict >, not >=
    assert verdict["turnover_mean"] is None     # structurally unavailable


def test_tier4_gate_verdict_turnover_high_fails():
    card = make_scorecard(corr=0.03, sharpe=0.8, fnc=0.02, gpr=1.5, cagr=0.01, turnover=0.9)
    verdict = tier4_gate_verdict(card, _gate())
    assert verdict["turnover_mean"] is False   # <= max, not >=


def test_tier_max_corrs_orders_by_tier():
    cards = {
        "t0": make_scorecard(corr=0.002),
        "t1": make_scorecard(corr=0.005),
        "t4": make_scorecard(corr=0.029),
    }
    rungs = tier_max_corrs(cards, {"t0": 0, "t1": 1, "t4": 4})
    assert rungs == {0: 0.002, 1: 0.005, 4: 0.029}
```

`make_scorecard(...)` is a module-local helper the implementer adds above these tests, building a `MetricScorecard` via `evaluate_model` on a tiny synthetic frame (the existing gate tests in this file already build scorecards — reuse that pattern; see the file's current helpers).

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_gates.py -k "verdict or max_corrs" -v`
Expected: FAIL — `ImportError: cannot import name 'tier4_gate_verdict'`.

- [ ] **Step 3: Implement helpers and refactor** — in `nmr/benchmark.py`, insert before `assert_tier4_gate` (line 752):

```python
def _tier4_gate_rows(
    scorecard: MetricScorecard, gate: Tier4GateConfig
) -> list[tuple[str, float | None, float, bool | None]]:
    """(field, observed, threshold, strict) rows for the 7 tier-4 fields.

    ``strict=None`` marks display-only fields (deflated_sharpe, A6) that are
    never pass/fail; ``observed=None`` marks structurally unavailable fields
    (turnover on v5.3).
    """
    card = scorecard
    return [
        ("corr", float(card.corr.value), float(gate.corr_min), False),
        ("corr_sharpe_ac", float(card.corr_sharpe_ac.value),
         float(gate.corr_sharpe_ac_min), False),
        ("fnc", float(card.fnc), float(gate.fnc_min), False),
        ("deflated_sharpe", float(card.deflated_sharpe),
         float(gate.deflated_sharpe_min), None),
        ("gain_to_pain_ratio", float(card.gain_to_pain_ratio),
         float(gate.gain_to_pain_min), False),
        ("cagr_1y", float(card.cagr_1y), float(gate.cagr_min), True),
        ("turnover_mean",
         None if card.turnover_mean is None else float(card.turnover_mean),
         float(gate.turnover_max), False),
    ]


def tier4_gate_verdict(
    scorecard: MetricScorecard, gate: Tier4GateConfig
) -> dict[str, bool | None]:
    """Per-threshold pass/fail booleans; None = unavailable/display-only."""
    _assert_scorecard_finite(scorecard, model_id=scorecard.model_id)
    verdict: dict[str, bool | None] = {}
    for field, observed, threshold, strict in _tier4_gate_rows(scorecard, gate):
        if observed is None or strict is None:
            verdict[field] = None
        elif field == "turnover_mean":
            verdict[field] = observed <= threshold
        elif strict:
            verdict[field] = observed > threshold
        else:
            verdict[field] = observed >= threshold
    return verdict
```

Replace the body of `assert_tier4_gate` (keep its docstring and signature):

```python
def assert_tier4_gate(scorecard: MetricScorecard, gate: Tier4GateConfig) -> None:
    """Production capital gate: reject candidates below the 7 hard thresholds.

    ``turnover_mean`` is structurally unavailable on v5.3 (disjoint era
    universes — consecutive validation eras share zero ids), so a ``None``
    turnover is reported by ``gate_report_frame`` but is not a hard failure.
    """
    _assert_scorecard_finite(scorecard, model_id=scorecard.model_id)
    violations: list[str] = []
    for field, observed, threshold, strict in _tier4_gate_rows(scorecard, gate):
        if observed is None or strict is None:
            continue
        if field == "turnover_mean":
            if observed > threshold:
                violations.append(
                    "turnover_mean: "
                    f"observed={observed:.8f}, need <= {threshold:.4f}"
                )
        elif strict:
            if observed <= threshold:
                violations.append(
                    f"{field}: observed={observed:.8f}, need > {threshold:.8f}"
                )
        elif observed < threshold:
            violations.append(
                f"{field}: observed={observed:.8f}, need >= {threshold:.8f}"
            )
    if violations:
        raise ValueError(
            f"Tier-4 gate violations for {scorecard.model_id!r}: "
            + "; ".join(violations)
        )
```

Insert before `assert_hierarchy_monotone` and refactor it:

```python
def tier_max_corrs(
    scorecards: Mapping[str, MetricScorecard],
    tier_of: Mapping[str, int],
) -> dict[int, float]:
    """Per-tier max of mean CORR (the monotonicity ladder metric)."""
    tiers = sorted(set(tier_of.values()))
    out: dict[int, float] = {}
    for tier in tiers:
        members = [mid for mid, t in tier_of.items() if t == tier]
        missing = [mid for mid in members if mid not in scorecards]
        if missing:
            raise ValueError(f"Missing scorecards for tier {tier}: {missing}")
        out[tier] = max(float(scorecards[mid].corr.value) for mid in members)
    return out
```

In `assert_hierarchy_monotone`, replace the scalar computation block (`scalar_by_tier` construction) with `scalar_by_tier = tier_max_corrs(scorecards, tier_of)` when `metric == "corr"`; keep the existing `rank_scalar` branch as-is.

Replace the body of `gate_report_frame` (from `card = result.scorecards[reference_id]` through the return) with:

```python
    card = result.scorecards[reference_id]
    rows = _tier4_gate_rows(card, gate)
    out_rows = []
    for field, measured, threshold, strict in rows:
        if measured is None or strict is None:
            passed = None
        elif strict:
            passed = measured > threshold
        else:
            passed = measured >= threshold
        out_rows.append({
            "model_id": reference_id,
            "field": field,
            "threshold": threshold,
            "measured": measured,
            "pass": passed,
        })
    return pl.DataFrame(out_rows)
```

Add `tier4_gate_verdict` and `tier_max_corrs` to `__all__` (alphabetical order).

- [ ] **Step 4: Update `nmr/__init__.py`** — add `tier4_gate_verdict` and `tier_max_corrs` to the `from .benchmark import (...)` block and to `__all__` (alphabetical positions).

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_gates.py -v`
Expected: PASS — including all pre-existing gate tests (the refactor must not change hard-gate behavior).

- [ ] **Step 6: Commit**

```bash
git add nmr/benchmark.py nmr/__init__.py tests/test_benchmark_gates.py
git commit -m "feat(benchmark): add tier4_gate_verdict and tier_max_corrs helpers"
```

---

### Task 2: Fleet config schema

**Files:**
- Create: `nmr/benchmark_fleet.py` (module skeleton + schema)
- Modify: `nmr/__init__.py`
- Test: `tests/test_benchmark_fleet.py`

**Interfaces:**
- Produces:
  - `VALID_FLEET_MODEL_KINDS: tuple[str, ...] = ("target_lag_mean", "lightgbm", "xgboost", "mlp", "ridge_stack")`
  - `VALID_FLEET_NEUTRALIZATION: tuple[float | None, ...] = (None, 0.25, 0.35, 0.5, 1.0)`
  - `VALID_FLEET_NEUTRALIZER_SELECTIONS: tuple[str, ...] = ("none", "riskiest_50")`
  - `FleetCellConfig` (frozen dataclass): `benchmark_id: str`, `source: str`, `input_space: str`, `model_kind: str`, `targets: tuple[str, ...] = ("target",)`, `target_weights: Mapping[str, float] | None = None`, `params: Mapping[str, Any] = {}`, `seed: int = 42`, `neutralization: float | None = None`, `neutralizer_selection: str = "none"`, `neutralizer_count: int = 50`, `fast_mode_params: Mapping[str, Any] | None = None`, `anchors: Mapping[str, float] | None = None`
  - `FleetFileConfig` (frozen): `cells: tuple[FleetCellConfig, ...]`
  - `load_fleet_config(path: str | Path) -> FleetFileConfig`
  - `load_fleet_suite_config(config_dir: str | Path) -> tuple[FleetCellConfig, ...]` — glob `*.yaml`, aggregate, duplicate-id check, sorted by `benchmark_id`
- Consumes: `nmr.benchmark._reject_unknown_keys`, `nmr.benchmark._freeze_mapping`, `nmr.benchmark.VALID_INPUT_SPACES`, `nmr.benchmark.DEFAULT_BENCHMARK_SEED`

- [ ] **Step 1: Write failing tests** — create `tests/test_benchmark_fleet.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nmr.benchmark_fleet'`.

- [ ] **Step 3: Implement the module** — create `nmr/benchmark_fleet.py`:

```python
"""Untiered benchmark fleet: community & tutorial model recreation layer.

Fleet cells are benchmark models without a tier assignment: they are scored
through the same ``evaluate_model`` pipeline as the 5-tier hierarchy and
their measured scorecards place them against the tier ladder indirectly.
Spec: docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from nmr.benchmark import (
    DEFAULT_BENCHMARK_SEED,
    VALID_INPUT_SPACES,
    _freeze_mapping,
    _reject_unknown_keys,
)

logger = logging.getLogger("nmr.benchmark_fleet")
# NOTE: Tasks 3-9 re-add the imports their generators need
# (numpy, polars, _standardize_feature_block, generate_canonical_predictions,
#  Ensembler, feature_stability_screen, construct_tree_model,
#  NeutralizationEngine, MetricScorecard/evaluate_model, BenchmarkData,
#  Tier4GateConfig, scorecards_to_frame, tier4_gate_verdict, Ridge, MLPRegressor,
#  lightgbm.early_stopping) and extend __all__ accordingly — one task at a time,
# so the lint gate stays green between tasks.

__all__ = [
    "FleetCellConfig",
    "FleetFileConfig",
    "VALID_FLEET_MODEL_KINDS",
    "VALID_FLEET_NEUTRALIZATION",
    "VALID_FLEET_NEUTRALIZER_SELECTIONS",
    "load_fleet_config",
    "load_fleet_suite_config",
]
# NOTE: this task imports only what the schema needs (lint gate: no unused
# imports, no undefined __all__ names). Tasks 3-9 add generator/runner
# symbols and their imports (`generate_canonical_predictions`,
# `construct_tree_model`, `NeutralizationEngine`, `Ensembler`, etc.) back
# to this file as they implement them.

VALID_FLEET_MODEL_KINDS: tuple[str, ...] = (
    "target_lag_mean", "lightgbm", "xgboost", "mlp", "ridge_stack",
)
VALID_FLEET_NEUTRALIZATION: tuple[float | None, ...] = (None, 0.25, 0.35, 0.5, 1.0)
VALID_FLEET_NEUTRALIZER_SELECTIONS: tuple[str, ...] = ("none", "riskiest_50")
DEFAULT_NEUTRALIZER_COUNT: int = 50


@dataclasses.dataclass(frozen=True)
class FleetCellConfig:
    benchmark_id: str
    source: str
    input_space: str
    model_kind: str
    targets: tuple[str, ...] = ("target",)
    target_weights: Mapping[str, float] | None = None
    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    seed: int = DEFAULT_BENCHMARK_SEED
    neutralization: float | None = None
    neutralizer_selection: str = "none"
    neutralizer_count: int = DEFAULT_NEUTRALIZER_COUNT
    fast_mode_params: Mapping[str, Any] | None = None
    anchors: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.benchmark_id or not isinstance(self.benchmark_id, str):
            raise ValueError(f"benchmark_id must be a non-empty string: {self.benchmark_id!r}")
        if not self.source or not isinstance(self.source, str):
            raise ValueError(f"source must be a non-empty string: {self.source!r}")
        if self.input_space not in VALID_INPUT_SPACES:
            raise ValueError(
                f"input_space={self.input_space!r} not in {VALID_INPUT_SPACES}"
            )
        if self.model_kind not in VALID_FLEET_MODEL_KINDS:
            raise ValueError(
                f"model_kind={self.model_kind!r} not in {VALID_FLEET_MODEL_KINDS}"
            )
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("targets must be a non-empty tuple")
        if not all(isinstance(t, str) and t for t in self.targets):
            raise ValueError(f"targets must be non-empty strings: {self.targets!r}")
        if self.model_kind == "target_lag_mean":
            if self.input_space != "none":
                raise ValueError(
                    "target_lag_mean requires input_space='none', "
                    f"got {self.input_space!r}"
                )
            if len(self.targets) != 1:
                raise ValueError(
                    "target_lag_mean requires exactly one single target, "
                    f"got {self.targets!r}"
                )
        if self.model_kind == "ridge_stack":
            for key in ("mode", "main_target", "specialists"):
                if key not in self.params:
                    raise ValueError(
                        f"ridge_stack requires params.{key}"
                    )
            if self.params["mode"] not in ("fixed", "search"):
                raise ValueError(
                    f"ridge_stack params.mode must be 'fixed' or 'search', "
                    f"got {self.params['mode']!r}"
                )
        if self.neutralization not in VALID_FLEET_NEUTRALIZATION:
            raise ValueError(
                f"neutralization={self.neutralization!r} not in "
                f"{VALID_FLEET_NEUTRALIZATION}"
            )
        if self.neutralizer_selection not in VALID_FLEET_NEUTRALIZER_SELECTIONS:
            raise ValueError(
                f"neutralizer_selection={self.neutralizer_selection!r} not in "
                f"{VALID_FLEET_NEUTRALIZER_SELECTIONS}"
            )
        if not isinstance(self.neutralizer_count, int) or isinstance(self.neutralizer_count, bool) \
                or self.neutralizer_count < 1:
            raise ValueError(
                f"neutralizer_count must be a positive int, got {self.neutralizer_count!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        object.__setattr__(self, "params", _freeze_mapping(self.params, name="params"))
        if self.target_weights is not None:
            weights = dict(self.target_weights)
            for target in weights:
                if target not in self.targets:
                    raise ValueError(
                        f"target_weights key {target!r} not in targets {self.targets!r}"
                    )
            if not all(isinstance(w, (int, float)) and not isinstance(w, bool)
                       and float(w) > 0.0 for w in weights.values()):
                raise ValueError("target_weights must be positive numbers")
            object.__setattr__(
                self, "target_weights",
                _freeze_mapping(self.target_weights, name="target_weights"),
            )
        if self.fast_mode_params is not None:
            object.__setattr__(
                self, "fast_mode_params",
                _freeze_mapping(self.fast_mode_params, name="fast_mode_params"),
            )
        if self.anchors is not None:
            object.__setattr__(
                self, "anchors",
                _freeze_mapping(self.anchors, name="anchors"),
            )


@dataclasses.dataclass(frozen=True)
class FleetFileConfig:
    cells: tuple[FleetCellConfig, ...] = ()

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("fleet config requires non-empty cells")
        ids = [cell.benchmark_id for cell in self.cells]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate benchmark ids in file: {ids}")


def load_fleet_config(path: str | Path) -> FleetFileConfig:
    """Load and validate a single fleet config file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"fleet config must be a mapping, got {type(raw).__name__}")
    _reject_unknown_keys(FleetFileConfig, raw)
    if not isinstance(raw.get("cells", []), list):
        raise ValueError("cells must be a list")
    cells: list[FleetCellConfig] = []
    for data in raw.get("cells", []):
        if not isinstance(data, dict):
            raise ValueError(f"fleet cell must be a mapping, got {type(data).__name__}")
        data = dict(data)
        if "neutralization" in data and data["neutralization"] == "none":
            data["neutralization"] = None
        if isinstance(data.get("targets"), list):
            data["targets"] = tuple(data["targets"])
        _reject_unknown_keys(FleetCellConfig, data)
        cells.append(FleetCellConfig(**data))
    return FleetFileConfig(cells=tuple(cells))


def load_fleet_suite_config(config_dir: str | Path) -> tuple[FleetCellConfig, ...]:
    """Load every *.yaml in config_dir, dedupe ids, sort by benchmark_id."""
    directory = Path(config_dir)
    files = sorted(p for p in directory.glob("*.yaml"))
    if not files:
        raise ValueError(f"no fleet config files found in {directory}")
    all_cells: list[FleetCellConfig] = []
    for path in files:
        all_cells.extend(load_fleet_config(path).cells)
    ids = [cell.benchmark_id for cell in all_cells]
    if len(set(ids)) != len(ids):
        seen = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate benchmark ids across configs: {seen}")
    return tuple(sorted(all_cells, key=lambda c: c.benchmark_id))
```

- [ ] **Step 4: Update `nmr/__init__.py`** — add `from .benchmark_fleet import (...)` with the schema symbols (`FleetCellConfig`, `FleetFileConfig`, `load_fleet_config`, `load_fleet_suite_config`, `VALID_FLEET_MODEL_KINDS`, `VALID_FLEET_NEUTRALIZATION`, `VALID_FLEET_NEUTRALIZER_SELECTIONS`) and the same names to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -v`
Expected: PASS (schema tests only).

- [ ] **Step 6: Commit**

```bash
git add nmr/benchmark_fleet.py nmr/__init__.py tests/test_benchmark_fleet.py
git commit -m "feat(benchmark): add fleet config schema (untiered benchmark cells)"
```

---

### Task 3: `target_lag_mean` generator

**Files:**
- Modify: `nmr/benchmark_fleet.py`
- Test: `tests/test_benchmark_fleet.py`

**Interfaces:**
- Produces:
  - `generate_lagged_target_predictions(train: pl.DataFrame, val_index: pl.DataFrame, *, target: str, window: int = 1, era_col: str = "era", id_col: str = "id", pred_col: str = "prediction") -> pl.DataFrame` — per validation era, constant prediction = mean of `target` over all rows pooled across the trailing `window` train eras. `train` must contain `[era_col, target]`; `val_index` must contain `[era_col, id_col]` and no target columns are read from it.
- Consumes: nothing beyond polars/numpy.

- [ ] **Step 1: Write failing tests** — append to `tests/test_benchmark_fleet.py`:

```python
import numpy as np
import polars as pl

from nmr.benchmark_fleet import generate_lagged_target_predictions


def _lag_train() -> pl.DataFrame:
    rows = []
    for era in (1, 2, 3, 4):
        for i in range(3):
            rows.append({"era": f"{era:04d}", "target": float(era)})
    return pl.DataFrame(rows)


def _val_index() -> pl.DataFrame:
    return pl.DataFrame(
        [{"era": f"{era:04d}", "id": f"{era}_{i}"}
         for era in (10, 11) for i in range(4)]
    )


def test_lag_mean_window_one_uses_last_train_era():
    out = generate_lagged_target_predictions(
        _lag_train(), _val_index(), target="target", window=1
    )
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == 8
    # last train era is "0004" with target mean 4.0 -> constant for every val row
    assert (out.get_column("prediction") == 4.0).all()


def test_lag_mean_window_two_pools_rows():
    out = generate_lagged_target_predictions(
        _lag_train(), _val_index(), target="target", window=2
    )
    # trailing two train eras: 3.0 and 4.0 -> pooled mean 3.5
    assert (out.get_column("prediction") == 3.5).all()


def test_lag_mean_never_reads_val_targets():
    val = _val_index()  # deliberately has no target column at all
    out = generate_lagged_target_predictions(_lag_train(), val, target="target")
    assert out.height == 8


def test_lag_mean_rejects_bad_window():
    with pytest.raises(ValueError, match="window"):
        generate_lagged_target_predictions(_lag_train(), _val_index(), target="target", window=0)
    with pytest.raises(ValueError, match="window"):
        generate_lagged_target_predictions(_lag_train(), _val_index(), target="target", window=5)


def test_lag_mean_rejects_val_era_overlap():
    bad_train = _lag_train().with_columns(pl.lit("0010").alias("era"))
    with pytest.raises(ValueError, match="strictly earlier"):
        generate_lagged_target_predictions(bad_train, _val_index(), target="target")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k lag_mean -v`
Expected: FAIL — `ImportError: cannot import name 'generate_lagged_target_predictions'`.

- [ ] **Step 3: Implement** — append to `nmr/benchmark_fleet.py`:

```python
def generate_lagged_target_predictions(
    train: pl.DataFrame,
    val_index: pl.DataFrame,
    *,
    target: str,
    window: int = 1,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Silly baseline: per validation era, predict the trailing-train target mean.

    All rows pooled across the trailing ``window`` train eras; train targets
    only, so the prediction is leak-safe by construction.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be a positive int, got {window!r}")
    if target not in train.columns:
        raise ValueError(f"train missing target column: {target!r}")
    for col in (era_col, id_col):
        if col not in val_index.columns:
            raise ValueError(f"val_index missing required column: {col!r}")

    train_eras = sorted(train.get_column(era_col).unique().to_list())
    val_eras = sorted(val_index.get_column(era_col).unique().to_list())
    if not train_eras or not val_eras:
        raise ValueError("train and val_index must each contain at least one era")
    if max(int(e) for e in train_eras) >= min(int(e) for e in val_eras):
        raise ValueError("train eras must be strictly earlier than validation eras")
    if window > len(train_eras):
        raise ValueError(
            f"window={window} exceeds available train eras ({len(train_eras)})"
        )

    trailing = train_eras[-window:]
    pooled = (
        train.filter(pl.col(era_col).is_in(trailing))
        .select(pl.col(target).cast(pl.Float64))
        .drop_nulls()
    )
    if pooled.is_empty():
        raise ValueError(f"trailing train eras have no finite {target!r} rows")
    value = float(pooled.get_column(target).mean())

    return (
        val_index.select([era_col, id_col])
        .sort([era_col, id_col])
        .with_columns(pl.lit(value, dtype=pl.Float64).alias(pred_col))
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k lag_mean -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark_fleet.py tests/test_benchmark_fleet.py
git commit -m "feat(benchmark-fleet): lagged train-target-mean generator"
```

---

### Task 4: Fleet `lightgbm` with riskiest-50 neutralizer selection

**Files:**
- Modify: `nmr/benchmark_fleet.py`
- Test: `tests/test_benchmark_fleet.py`

**Interfaces:**
- Produces:
  - `_select_riskiest_features(train: pl.DataFrame, *, feature_cols: Sequence[str], target_col: str, count: int, era_col: str = "era") -> list[str]` — screen features via `feature_stability_screen`, rank by `cross_regime_variance` descending (nulls last), tie-break by feature name ascending, take top `count`
  - `generate_fleet_lightgbm_predictions(train, val, *, targets, feature_cols, params, seed, neutralization=0.0, neutralizer_selection="none", neutralizer_count=DEFAULT_NEUTRALIZER_COUNT, purge_eras=DEFAULT_BENCHMARK_PURGE_ERAS, era_col="era", id_col="id", pred_col="prediction") -> pl.DataFrame`
- Consumes: `generate_canonical_predictions`, `feature_stability_screen`, `NeutralizationEngine`, `train_validation_purged_split`

- [ ] **Step 1: Write failing tests** — append to `tests/test_benchmark_fleet.py`:

```python
from nmr.benchmark_fleet import (
    _select_riskiest_features,
    generate_fleet_lightgbm_predictions,
)


def _tiny_train_val(eras: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)):
    """Synthetic train (with target) + val (no targets) with 3 features.

    Val eras are derived as (max(train)+1, max(train)+2) so the exact 8-era
    purge gap holds for every ``eras`` argument (train_validation_purged_split
    enforces max(trimmed_train) + 9 == min(val)).
    """
    rng = np.random.default_rng(11)
    train_rows, val_rows = [], []
    for i, era in enumerate(eras):
        for row in range(5):
            train_rows.append({
                "era": f"{era:04d}", "id": f"t{era}_{row}",
                "f1": float(i) + rng.normal(0, 0.01),
                "f2": rng.normal(0, 1),
                "f3": float(era % 2),
                "target": float(i % 3),
            })
    for era in (max(eras) + 1, max(eras) + 2):
        for row in range(5):
            val_rows.append({
                "era": f"{era:04d}", "id": f"v{era}_{row}",
                # deterministic per-row drift: tiny fixtures can otherwise
                # yield constant per-era predictions and fail the per-era
                # rank-gaussian mean/std assertions
                "f1": float(row), "f2": rng.normal(0, 1), "f3": float(row % 2),
            })
    return pl.DataFrame(train_rows), pl.DataFrame(val_rows)


def test_select_riskiest_ranks_by_cross_regime_variance():
    train, _ = _tiny_train_val()
    # f1 drifts monotonically across eras -> highest cross_regime_variance
    out = _select_riskiest_features(
        train, feature_cols=["f1", "f2", "f3"], target_col="target", count=1
    )
    assert out == ["f1"]


def test_fleet_lgbm_selection_none_matches_canonical():
    train, val = _tiny_train_val()
    params = {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2, "num_leaves": 4}
    fleet_out = generate_fleet_lightgbm_predictions(
        train, val, targets=["target"], feature_cols=["f1", "f2", "f3"],
        params=params, seed=42, neutralization=0.5, neutralizer_selection="none",
    )
    canonical = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=["f1", "f2", "f3"],
        params=params, seed=42, neutralization=0.5,
    )
    assert fleet_out.equals(canonical)


def test_fleet_lgbm_riskiest_50_matches_manual_pipeline():
    train, val = _tiny_train_val()
    params = {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2, "num_leaves": 4}
    neutralizers = _select_riskiest_features(
        train, feature_cols=["f1", "f2", "f3"], target_col="target", count=2
    )
    fleet_out = generate_fleet_lightgbm_predictions(
        train, val, targets=["target"], feature_cols=["f1", "f2", "f3"],
        params=params, seed=42, neutralization=1.0,
        neutralizer_selection="riskiest_50", neutralizer_count=2,
    )
    raw = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=["f1", "f2", "f3"],
        params=params, seed=42, neutralization=0.0,
    )
    with_features = raw.join(
        val.select(["era", "id", *neutralizers]), on=["era", "id"], how="inner"
    )
    manual = NeutralizationEngine().neutralize(
        with_features, pred_col="prediction",
        feature_cols=neutralizers, era_col="era", proportion=1.0,
    ).select(["era", "id", "prediction"]).sort(["era", "id"])
    assert fleet_out.equals(manual)


def test_fleet_lgbm_rejects_unknown_selection():
    train, val = _tiny_train_val()
    with pytest.raises(ValueError, match="neutralizer_selection"):
        generate_fleet_lightgbm_predictions(
            train, val, targets=["target"], feature_cols=["f1"],
            params={"n_estimators": 5}, seed=42,
            neutralizer_selection="bogus",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k "riskiest or fleet_lgbm" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement** — append to `nmr/benchmark_fleet.py`:

```python
def _select_riskiest_features(
    train: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    count: int,
    era_col: str = "era",
) -> list[str]:
    """Top-``count`` features by cross-regime drift (framework risk screen).

    Ranked by ``cross_regime_variance`` descending, nulls last, feature name
    ascending as the deterministic tie-break. Documented deviation from the
    notebooks' ``get_biggest_change_features`` (same intent — most unstable
    features — via the framework-tested screen).
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"count must be a positive int, got {count!r}")
    screen = feature_stability_screen(
        train, feature_cols=list(feature_cols), target_col=target_col,
        era_col=era_col,
    )
    ranked = screen.sort(
        by=["cross_regime_variance", "feature"],
        descending=[True, False],
        nulls_last=[True, False],
    )  # polars >= 1.41: kwargs form (Expr.nulls_last()/desc() removed)
    return ranked.get_column("feature").to_list()[:count]


def generate_fleet_lightgbm_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    targets: Sequence[str],
    feature_cols: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
    neutralization: float = 0.0,
    neutralizer_selection: str = "none",
    neutralizer_count: int = DEFAULT_NEUTRALIZER_COUNT,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Fleet LightGBM: canonical fits + optional riskiest-feature neutralization."""
    if neutralizer_selection not in VALID_FLEET_NEUTRALIZER_SELECTIONS:
        raise ValueError(
            f"neutralizer_selection={neutralizer_selection!r} not in "
            f"{VALID_FLEET_NEUTRALIZER_SELECTIONS}"
        )
    trimmed_train_eras, _ = train_validation_purged_split(
        train.get_column(era_col).unique().to_list(),
        val.get_column(era_col).unique().to_list(),
        purge_eras=purge_eras,
    )
    train_rows = train.filter(pl.col(era_col).is_in(trimmed_train_eras))

    if neutralizer_selection == "riskiest_50":
        neutralizer_cols = _select_riskiest_features(
            train_rows, feature_cols=feature_cols,
            target_col=list(targets)[0], count=neutralizer_count, era_col=era_col,
        )
    else:
        neutralizer_cols = list(feature_cols)

    out = generate_canonical_predictions(
        train, val, targets=list(targets), feature_cols=list(feature_cols),
        params=params, seed=seed, neutralization=0.0,
        purge_eras=purge_eras, era_col=era_col, id_col=id_col, pred_col=pred_col,
    )

    if float(neutralization) > 0.0:
        with_features = out.join(
            val.select([era_col, id_col, *neutralizer_cols]),
            on=[era_col, id_col], how="inner",
        )
        out = NeutralizationEngine().neutralize(
            with_features, pred_col=pred_col, feature_cols=neutralizer_cols,
            era_col=era_col, proportion=float(neutralization),
        ).select([era_col, id_col, pred_col]).sort([era_col, id_col])
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k "riskiest or fleet_lgbm" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark_fleet.py tests/test_benchmark_fleet.py
git commit -m "feat(benchmark-fleet): lightgbm generator with riskiest-50 neutralization"
```

---

### Task 5: Fleet `xgboost` — multi-target weights + early stopping

**Files:**
- Modify: `nmr/models.py` (`construct_tree_model` gains `extra_params`)
- Modify: `nmr/benchmark_fleet.py`
- Test: `tests/test_benchmark_fleet.py`

**Interfaces:**
- Produces:
  - `construct_tree_model(backend, params, *, seed, n_features, device="cpu", extra_params: Mapping[str, Any] | None = None)` — `extra_params` merged into the resolved params AFTER colsample flooring (so `early_stopping_rounds` reaches the XGBoost constructor untouched)
  - `generate_fleet_xgb_predictions(train, val, *, targets, feature_cols, params, seed, target_weights=None, purge_eras=DEFAULT_BENCHMARK_PURGE_ERAS, era_col="era", id_col="id", pred_col="prediction") -> pl.DataFrame` — per-target XGBoost fits (NaN-masked), optional deterministic tail-of-train holdout for `early_stopping_rounds`, rank-Gaussian weighted blend (weights normalized)
- Consumes: `train_validation_purged_split`, `construct_tree_model`, `Ensembler`

- [ ] **Step 1: Write failing tests** — append to `tests/test_benchmark_fleet.py`:

```python
from nmr.benchmark_fleet import generate_fleet_xgb_predictions


def test_xgb_weighted_blend_normalizes_weights_and_orders_ranks():
    train, val = _tiny_train_val()
    params = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1}
    train = train.with_columns(pl.col("target").alias("target_ender_20"))
    out = generate_fleet_xgb_predictions(
        train, val, targets=["target", "target_ender_20"],
        feature_cols=["f1", "f2", "f3"], params=params, seed=42,
        target_weights={"target": 0.35, "target_ender_20": 0.65},
    )
    assert out.columns == ["era", "id", "prediction"]
    # rank-gaussianized per era: mean ~0, std ~1 within each era
    for era in out.get_column("era").unique().to_list():
        vals = out.filter(pl.col("era") == era).get_column("prediction").to_numpy()
        assert abs(float(vals.mean())) < 1e-6
        assert abs(float(vals.std(ddof=0)) - 1.0) < 1e-6


def test_xgb_weights_default_to_equal():
    train, val = _tiny_train_val()
    params = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1}
    a = generate_fleet_xgb_predictions(
        train, val, targets=["target"], feature_cols=["f1"],
        params=params, seed=42,
    )
    b = generate_fleet_xgb_predictions(
        train, val, targets=["target"], feature_cols=["f1"],
        params=params, seed=42, target_weights={"target": 2.0},
    )
    assert a.equals(b)  # single target: weight value is irrelevant


def test_xgb_early_stopping_engages_on_holdout():
    train, val = _tiny_train_val(eras=tuple(range(1, 25)))
    params = {
        "n_estimators": 50, "max_depth": 2, "learning_rate": 0.1,
        "early_stopping_rounds": 5, "holdout_era_frac": 0.25,
    }
    out = generate_fleet_xgb_predictions(
        train, val, targets=["target"], feature_cols=["f1", "f2", "f3"],
        params=params, seed=42,
    )
    assert out.height == 10


def test_xgb_rejects_weight_for_unknown_target():
    train, val = _tiny_train_val()
    with pytest.raises(ValueError, match="target_weights"):
        generate_fleet_xgb_predictions(
            train, val, targets=["target"], feature_cols=["f1"],
            params={"n_estimators": 5}, seed=42,
            target_weights={"bogus": 1.0},
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k xgb -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3a: Extend `construct_tree_model`** — in `nmr/models.py`:

```python
def construct_tree_model(
    backend: str,
    params: Mapping[str, Any],
    *,
    seed: int,
    n_features: int,
    device: str = "cpu",
    extra_params: Mapping[str, Any] | None = None,
) -> object:
    """Build a deterministic, CPU-default tree estimator from raw params.

    Applies the same backend param mapping, colsample flooring, and
    determinism flags as ``ModelOrchestrator``. Used by the benchmark
    hierarchy so benchmark cells never hand-duplicate param resolution.
    ``extra_params`` are merged AFTER resolution so constructor-only kwargs
    (e.g. XGBoost ``early_stopping_rounds``) bypass param validation.
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
    if extra_params:
        resolved.update(dict(extra_params))
    return orchestrator._build_model(resolved)
```

- [ ] **Step 3b: Implement the generator** — append to `nmr/benchmark_fleet.py`:

```python
_ES_KEYS = ("early_stopping_rounds", "holdout_era_frac")


def generate_fleet_xgb_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    targets: Sequence[str],
    feature_cols: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
    target_weights: Mapping[str, float] | None = None,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Fleet XGBoost: per-target fits, optional early stopping, weighted blend."""
    if not targets or not feature_cols:
        raise ValueError("targets and feature_cols must be non-empty")
    weights = dict(target_weights) if target_weights else {}
    for target in weights:
        if target not in targets:
            raise ValueError(
                f"target_weights key {target!r} not in targets {list(targets)!r}"
            )
    trimmed_train_eras, _ = train_validation_purged_split(
        train.get_column(era_col).unique().to_list(),
        val.get_column(era_col).unique().to_list(),
        purge_eras=purge_eras,
    )
    train_rows = train.filter(pl.col(era_col).is_in(trimmed_train_eras))
    val_rows = val.sort([era_col, id_col])
    missing_feats = [c for c in feature_cols if c not in train.columns or c not in val.columns]
    if missing_feats:
        raise ValueError(f"missing feature columns: {missing_feats}")

    es_rounds = params.get("early_stopping_rounds")
    holdout_frac = float(params.get("holdout_era_frac", 0.1))
    fit_eras = list(trimmed_train_eras)
    holdout_eras: list[str] = []
    if es_rounds is not None and len(fit_eras) >= 4:
        n_hold = max(1, int(round(holdout_frac * len(fit_eras))))
        if n_hold >= len(fit_eras):
            n_hold = len(fit_eras) // 2
        holdout_eras = fit_eras[-n_hold:]
        fit_eras = fit_eras[:-n_hold]
    model_params = {k: v for k, v in params.items() if k not in _ES_KEYS}

    x_fit = train_rows.filter(pl.col(era_col).is_in(fit_eras)) \
        .select(feature_cols).cast(pl.Float32).to_pandas()
    x_val = val_rows.select(feature_cols).cast(pl.Float32).to_pandas()
    x_hold = None
    if holdout_eras:
        hold_rows = train_rows.filter(pl.col(era_col).is_in(holdout_eras))
        x_hold = hold_rows.select(feature_cols).cast(pl.Float32).to_pandas()

    component_preds: dict[str, np.ndarray] = {}
    for index, target in enumerate(targets):
        if target not in train.columns:
            raise ValueError(f"missing target column: {target!r}")
        y = train_rows.filter(pl.col(era_col).is_in(fit_eras)) \
            .get_column(target).cast(pl.Float64).to_numpy()
        mask = np.isfinite(y)
        if mask.sum() < 2:
            raise ValueError(
                f"target {target!r} has fewer than 2 finite train rows after purge"
            )
        extra = (
            {"early_stopping_rounds": int(es_rounds)} if es_rounds is not None else None
        )
        model = construct_tree_model(
            "xgboost", model_params, seed=seed + index,
            n_features=len(feature_cols), device="cpu", extra_params=extra,
        )
        if x_hold is not None:
            y_hold = train_rows.filter(pl.col(era_col).is_in(holdout_eras)) \
                .get_column(target).cast(pl.Float64).to_numpy()
            mask_h = np.isfinite(y_hold)
            model.fit(
                x_fit[mask], y[mask],
                eval_set=[(x_hold[mask_h], y_hold[mask_h])],
                verbose=False,
            )
        else:
            model.fit(x_fit[mask], y[mask])
        component_preds[target] = np.asarray(model.predict(x_val), dtype=float)

    val_index = val_rows.select([era_col, id_col])
    frame = val_index.with_columns(
        [pl.Series(target, component_preds[target]) for target in targets]
    )
    if weights:
        total = sum(weights.get(t, 0.0) for t in targets)
        blend_weights = [weights.get(t, 0.0) / total for t in targets]
    else:
        blend_weights = [1.0 / len(targets)] * len(targets)
    ensembler = Ensembler()
    blended = ensembler.blend(
        Ensembler.rank_normalize(frame, pred_cols=list(targets), era_col=era_col),
        pred_cols=list(targets), weights=blend_weights,
        era_col=era_col, out_col=pred_col,
    )
    gaussianized = Ensembler.rank_normalize(
        blended, pred_cols=[pred_col], era_col=era_col
    )
    return gaussianized.select([era_col, id_col, pred_col]).sort([era_col, id_col])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k xgb -v`
Expected: PASS. Also run `./.venv/Scripts/python -m pytest tests/test_benchmark_trees.py tests/test_benchmark_canonical.py -q` to confirm `construct_tree_model`'s signature extension broke nothing.

- [ ] **Step 5: Commit**

```bash
git add nmr/models.py nmr/benchmark_fleet.py tests/test_benchmark_fleet.py
git commit -m "feat(benchmark-fleet): xgboost weighted multi-target generator with early stopping"
```

---

### Task 6: `mlp` generator

**Files:**
- Modify: `nmr/benchmark_fleet.py`
- Test: `tests/test_benchmark_fleet.py`

**Interfaces:**
- Produces: `generate_mlp_predictions(train, val, *, target, feature_cols, params, seed, purge_eras=DEFAULT_BENCHMARK_PURGE_ERAS, era_col="era", id_col="id", pred_col="prediction") -> pl.DataFrame`
- Consumes: `_standardize_feature_block`, `train_validation_purged_split`, `Ensembler`, `sklearn.neural_network.MLPRegressor`

- [ ] **Step 1: Write failing tests** — append to `tests/test_benchmark_fleet.py`:

```python
from nmr.benchmark_fleet import generate_mlp_predictions


_MLP_KEYS = (
    "hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate_init",
    "batch_size", "max_iter", "early_stopping", "n_iter_no_change",
    "validation_fraction",
)


def test_mlp_same_seed_is_deterministic():
    train, val = _tiny_train_val()
    params = {"hidden_layer_sizes": (8, 4), "max_iter": 30, "batch_size": 4,
              "learning_rate_init": 0.01}
    a = generate_mlp_predictions(
        train, val, target="target", feature_cols=["f1", "f2", "f3"],
        params=params, seed=42,
    )
    b = generate_mlp_predictions(
        train, val, target="target", feature_cols=["f1", "f2", "f3"],
        params=params, seed=42,
    )
    assert a.equals(b)


def test_mlp_constant_feature_stays_finite():
    train, val = _tiny_train_val()
    train = train.with_columns(pl.lit(1.0).alias("f4"))
    val = val.with_columns(pl.lit(1.0).alias("f4"))
    out = generate_mlp_predictions(
        train, val, target="target", feature_cols=["f1", "f2", "f3", "f4"],
        params={"hidden_layer_sizes": (8, 4), "max_iter": 30, "batch_size": 4},
        seed=42,
    )
    assert out.get_column("prediction").is_finite().all()


def test_mlp_rejects_unknown_param_key():
    train, val = _tiny_train_val()
    with pytest.raises(ValueError, match="unknown mlp param"):
        generate_mlp_predictions(
            train, val, target="target", feature_cols=["f1"],
            params={"hidden_layer_sizes": (4,), "bogus": 1}, seed=42,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k mlp -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement** — append to `nmr/benchmark_fleet.py` (import `MLPRegressor` from `sklearn.neural_network` at the top of the file):

```python
_VALID_MLP_PARAM_KEYS: tuple[str, ...] = (
    "hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate_init",
    "batch_size", "max_iter", "early_stopping", "n_iter_no_change",
    "validation_fraction",
)


def generate_mlp_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    target: str,
    feature_cols: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Fleet MLP (sklearn MLPRegressor): standardized features, fixed seed."""
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")
    unknown = sorted(set(params) - set(_VALID_MLP_PARAM_KEYS))
    if unknown:
        raise ValueError(f"unknown mlp param keys: {unknown}")
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

    x_train = train_rows.select(feature_cols).cast(pl.Float32).to_numpy(writable=True)
    x_val = val_rows.select(feature_cols).cast(pl.Float32).to_numpy(writable=True)
    y = train_rows.get_column(target).cast(pl.Float64).to_numpy()
    mask = np.isfinite(y)
    if mask.sum() < 2:
        raise ValueError(
            f"target {target!r} has fewer than 2 finite train rows after purge"
        )
    x_train, x_val = _standardize_feature_block(x_train, x_val)

    model = MLPRegressor(**dict(params), random_state=seed)
    model.fit(x_train[mask], y[mask])
    raw = np.asarray(model.predict(x_val), dtype=float)

    frame = val_rows.select([era_col, id_col]).with_columns(pl.Series(pred_col, raw))
    blended = Ensembler().blend(
        Ensembler.rank_normalize(frame, pred_cols=[pred_col], era_col=era_col),
        pred_cols=[pred_col], weights=[1.0], era_col=era_col, out_col=pred_col,
    )
    return blended.select([era_col, id_col, pred_col]).sort([era_col, id_col])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k mlp -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark_fleet.py tests/test_benchmark_fleet.py
git commit -m "feat(benchmark-fleet): sklearn MLP generator"
```


---

### Task 7: `ridge_stack` generator — fixed mode

**Files:**
- Modify: `nmr/benchmark_fleet.py`
- Test: `tests/test_benchmark_fleet.py`

**Interfaces:**
- Produces:
  - `_stack_partitions(trimmed: Sequence[str], *, meta_tail_pct: float, specialists: Sequence[str]) -> tuple[list[str], list[str]]` — `(specialist_eras, meta_eras)`; meta = trailing `max(1, round(meta_tail_pct * n))` eras; boundary purge 16 if any specialist ends `_60`, else 8; raises `ValueError` if fewer than 2 specialist eras remain
  - `generate_ridge_stack_predictions(train, val, *, main_target, specialists, feature_cols, params, seed, neutralization=0.0, val_targets=None, benchmarks=None, purge_eras=DEFAULT_BENCHMARK_PURGE_ERAS, era_col="era", id_col="id", pred_col="prediction") -> pl.DataFrame` — `mode: fixed` reads `params.alpha`, `params.meta_alpha`, `params.meta_tail_pct`; `mode: search` is Task 8
- Consumes: `train_validation_purged_split`, `Ensembler`, `NeutralizationEngine`, `Ridge` (sklearn)

- [ ] **Step 1: Write failing tests** — append to `tests/test_benchmark_fleet.py`:

```python
from nmr.benchmark_fleet import _stack_partitions, generate_ridge_stack_predictions


def test_stack_partitions_tail_and_purge_8():
    eras = [f"{e:04d}" for e in range(1, 31)]
    spec, meta = _stack_partitions(eras, meta_tail_pct=0.10, specialists=["target_x_20"])
    assert meta == [f"{e:04d}" for e in range(28, 31)]   # trailing 3
    assert spec[-1] == "0019"                             # 8-era purge (0020..0027)
    assert set(spec) & set(meta) == set()


def test_stack_partitions_purge_16_when_60d_present():
    eras = [f"{e:04d}" for e in range(1, 40)]
    spec, meta = _stack_partitions(eras, meta_tail_pct=0.10, specialists=["target_ender_60"])
    assert spec[-1] == "0019"  # meta = 0036..0039, 16-era purge = 0020..0035
    assert set(spec) & set(meta) == set()


def test_stack_partitions_raises_when_not_enough_eras():
    eras = [f"{e:04d}" for e in range(1, 8)]
    with pytest.raises(ValueError, match="stack split"):
        _stack_partitions(eras, meta_tail_pct=0.5, specialists=["target_x_20"])


def _stack_train_val():
    rng = np.random.default_rng(23)
    train_rows = []
    for era in range(1, 41):
        for row in range(6):
            train_rows.append({
                "era": f"{era:04d}", "id": f"t{era}_{row}",
                "f1": rng.normal(0, 1), "f2": rng.normal(0, 1),
                "main": rng.normal(0, 1),
                "aux1": rng.normal(0, 1),
                "aux2": rng.normal(0, 1),
            })
    for i, row in enumerate(train_rows):  # NaN some aux2 rows: per-target masking
        if i % 7 == 0:
            row["aux2"] = None
    val_rows = [
        {"era": f"{era:04d}", "id": f"v{era}_{row}", "f1": rng.normal(0, 1), "f2": rng.normal(0, 1)}
        for era in (41, 42) for row in range(6)
    ]
    return pl.DataFrame(train_rows), pl.DataFrame(val_rows)


def test_ridge_stack_fixed_never_reads_val_targets_and_is_ranked():
    train, val = _stack_train_val()
    out = generate_ridge_stack_predictions(  # val has no target cols — purity contract
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"],
        params={"mode": "fixed", "alpha": 1e-6, "meta_alpha": 1e-6, "meta_tail_pct": 0.10},
        seed=42,
    )
    assert out.columns == ["era", "id", "prediction"]
    for era in out.get_column("era").unique().to_list():
        vals = out.filter(pl.col("era") == era).get_column("prediction").to_numpy()
        assert abs(float(vals.mean())) < 1e-6


def test_ridge_stack_fixed_is_seed_deterministic():
    train, val = _stack_train_val()
    params = {"mode": "fixed", "alpha": 1e-6, "meta_alpha": 1e-6, "meta_tail_pct": 0.10}
    a = generate_ridge_stack_predictions(
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"], params=params, seed=42,
    )
    b = generate_ridge_stack_predictions(
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"], params=params, seed=42,
    )
    assert a.equals(b)


def test_ridge_stack_rejects_unknown_mode():
    train, val = _stack_train_val()
    with pytest.raises(ValueError, match="mode"):
        generate_ridge_stack_predictions(
            train, val, main_target="main", specialists=["aux1"],
            feature_cols=["f1"], params={"mode": "bogus", "alpha": 1.0}, seed=42,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k "stack" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement** — append to `nmr/benchmark_fleet.py`:

```python
def _stack_partitions(
    trimmed: Sequence[str],
    *,
    meta_tail_pct: float,
    specialists: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split purged train eras into specialist-train and meta-tail partitions.

    The boundary gets a horizon-aware purge buffer mirroring the splitter
    convention: 16 eras when any 60D specialist is present, else 8.
    """
    if not 0.0 < float(meta_tail_pct) < 1.0:
        raise ValueError(f"meta_tail_pct must be in (0, 1), got {meta_tail_pct!r}")
    eras = list(trimmed)
    n_meta = max(1, int(round(float(meta_tail_pct) * len(eras))))
    stack_purge = 16 if any(str(t).endswith("_60") for t in specialists) else 8
    if len(eras) - n_meta - stack_purge < 2:
        raise ValueError(
            "not enough train eras for stack split: "
            f"eras={len(eras)}, meta_tail={n_meta}, purge={stack_purge}"
        )
    meta = eras[-n_meta:]
    spec = eras[: len(eras) - n_meta - stack_purge]
    return spec, meta


def generate_ridge_stack_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    main_target: str,
    specialists: Sequence[str],
    feature_cols: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
    neutralization: float = 0.0,
    val_targets: pl.DataFrame | None = None,
    benchmarks: pl.DataFrame | None = None,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Two-layer ridge stacking: per-target specialists -> meta ridge.

    Fixed mode: one Ridge per specialist (``params.alpha``), meta Ridge
    (``params.meta_alpha``) on per-era-ranked tail OOF predictions.
    Search mode (v1.5.1) delegates to :func:`_ridge_stack_search`.
    """
    mode = params.get("mode")
    if mode not in ("fixed", "search"):
        raise ValueError(f"params.mode must be 'fixed' or 'search', got {mode!r}")
    if not specialists or not feature_cols:
        raise ValueError("specialists and feature_cols must be non-empty")
    if main_target not in train.columns:
        raise ValueError(f"train missing main target column: {main_target!r}")

    trimmed_train_eras, _ = train_validation_purged_split(
        train.get_column(era_col).unique().to_list(),
        val.get_column(era_col).unique().to_list(),
        purge_eras=purge_eras,
    )
    train_rows = train.filter(pl.col(era_col).is_in(trimmed_train_eras))
    val_rows = val.sort([era_col, id_col])
    missing_feats = [c for c in feature_cols if c not in train.columns or c not in val.columns]
    if missing_feats:
        raise ValueError(f"missing feature columns: {missing_feats}")

    # v1.5.1 NaN strategy: fill features with 0.5 (neutral) before any fit.
    if bool(params.get("nan_fill", False)):
        train_rows = train_rows.with_columns(
            [pl.col(c).fill_null(0.5) for c in feature_cols]
        )
        val_rows = val_rows.with_columns(
            [pl.col(c).fill_null(0.5) for c in feature_cols]
        )

    if mode == "search":
        return _ridge_stack_search(
            train_rows, val_rows, main_target=main_target,
            specialists=list(specialists), feature_cols=list(feature_cols),
            params=params, seed=seed, neutralization=float(neutralization),
            val_targets=val_targets, benchmarks=benchmarks,
            era_col=era_col, id_col=id_col, pred_col=pred_col,
        )

Transitional note for Task 7 (ruff F821): `_ridge_stack_search` is defined in Task 8. To keep the lint gate green in Task 7, add this module-level transitional stub right after the dispatch above, with a comment saying Task 8 replaces it with the real implementation:

```python
def _ridge_stack_search(*args: object, **kwargs: object) -> pl.DataFrame:
    """Transitional stub — the search-mode implementation arrives in Task 8."""
    raise NotImplementedError("ridge_stack search mode is implemented in the next task")
```

    spec_eras, meta_eras = _stack_partitions(
        trimmed_train_eras,
        meta_tail_pct=float(params["meta_tail_pct"]),
        specialists=list(specialists),
    )
    spec_rows = train_rows.filter(pl.col(era_col).is_in(spec_eras))
    meta_rows = train_rows.filter(pl.col(era_col).is_in(meta_eras))
    alpha = float(params["alpha"])
    meta_alpha = float(params["meta_alpha"])

    x_spec = spec_rows.select(feature_cols).cast(pl.Float32).to_numpy()
    x_meta = meta_rows.select(feature_cols).cast(pl.Float32).to_numpy()
    x_val = val_rows.select(feature_cols).cast(pl.Float32).to_numpy()

    meta_pred_frames: list[pl.DataFrame] = []
    val_pred_frames: list[pl.DataFrame] = []
    for target in specialists:
        if target not in train.columns:
            raise ValueError(f"train missing specialist target column: {target!r}")
        y = spec_rows.get_column(target).cast(pl.Float64).to_numpy()
        mask = np.isfinite(y)
        if mask.sum() < 2:
            raise ValueError(f"specialist {target!r} has <2 finite train rows")
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=seed)
        model.fit(x_spec[mask], y[mask])
        meta_pred_frames.append(
            meta_rows.select([era_col, id_col]).with_columns(
                pl.Series(target, np.asarray(model.predict(x_meta), dtype=float))
            )
        )
        val_pred_frames.append(
            val_rows.select([era_col, id_col]).with_columns(
                pl.Series(target, np.asarray(model.predict(x_val), dtype=float))
            )
        )

    meta_ranked = meta_pred_frames[0]
    for part in meta_pred_frames[1:]:
        meta_ranked = meta_ranked.join(part, on=[era_col, id_col], how="inner")
    meta_ranked = Ensembler.rank_normalize(
        meta_ranked, pred_cols=list(specialists), era_col=era_col
    )
    val_ranked = val_pred_frames[0]
    for part in val_pred_frames[1:]:
        val_ranked = val_ranked.join(part, on=[era_col, id_col], how="inner")
    val_ranked = Ensembler.rank_normalize(
        val_ranked, pred_cols=list(specialists), era_col=era_col
    )

    meta_y = meta_rows.select([era_col, id_col, main_target]).drop_nulls()
    meta_X = meta_ranked.join(meta_y, on=[era_col, id_col], how="inner")
    if meta_X.height < 2:
        raise ValueError("fewer than 2 aligned meta-train rows")
    meta_model = Ridge(alpha=meta_alpha, fit_intercept=True, random_state=seed)
    meta_model.fit(
        meta_X.select(specialists).cast(pl.Float32).to_numpy(),
        meta_X.get_column(main_target).cast(pl.Float64).to_numpy(),
    )
    raw = np.asarray(
        meta_model.predict(val_ranked.select(specialists).cast(pl.Float32).to_numpy()),
        dtype=float,
    )
    frame = val_ranked.select([era_col, id_col]).with_columns(pl.Series(pred_col, raw))
    out = Ensembler().blend(
        Ensembler.rank_normalize(frame, pred_cols=[pred_col], era_col=era_col),
        pred_cols=[pred_col], weights=[1.0], era_col=era_col, out_col=pred_col,
    ).select([era_col, id_col, pred_col]).sort([era_col, id_col])

    if float(neutralization) > 0.0:
        with_features = out.join(
            val.select([era_col, id_col, *feature_cols]),
            on=[era_col, id_col], how="inner",
        )
        out = NeutralizationEngine().neutralize(
            with_features, pred_col=pred_col, feature_cols=list(feature_cols),
            era_col=era_col, proportion=float(neutralization),
        ).select([era_col, id_col, pred_col]).sort([era_col, id_col])
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k "stack" -v`
Expected: PASS (partition + fixed-mode tests; search-mode tests arrive in Task 8).

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark_fleet.py tests/test_benchmark_fleet.py
git commit -m "feat(benchmark-fleet): ridge-stacking generator (fixed mode)"
```

---

### Task 8: `ridge_stack` generator — search mode (FA v1.5.1)

**Files:**
- Modify: `nmr/benchmark_fleet.py`
- Test: `tests/test_benchmark_fleet.py`

**Interfaces:**
- Produces:
  - `_era_sharpe(preds: np.ndarray, eras: Sequence[str], target: np.ndarray) -> float` — mean/std(ddof=0) of per-era Pearson CORR (degenerate eras skipped; `-inf` when none)
  - `_ridge_stack_search(...) -> pl.DataFrame` (private, called by `generate_ridge_stack_predictions` in search mode)
- Consumes: `Ridge`, `construct_tree_model("lightgbm", ...)`, `lightgbm.early_stopping`, `NeutralizationEngine`, `Ensembler`
- Search params (all read from `params`): `snnr_weights` (mapping), `top_k` (int, 12), `min_coverage` (0.50), `min_abs_main_corr` (0.01), `priority_hints` (list), `specialist_alpha_grid` (list[float]), `specialist_sharpe_floor` (0.50), `min_specialists` (6), `meta_alpha_grid` (list[float]), `meta_lgbm_params` (mapping), `decorr_grid` (list[float]), `neutralization_grid` (list[float]), `benchmark_col` (str, `"v53_lgbm_ender20"`), `meta_tail_pct` (0.10), `nan_fill` (bool)

- [ ] **Step 1: Write failing tests** — append to `tests/test_benchmark_fleet.py`:

```python
from nmr.benchmark_fleet import _era_sharpe


def test_era_sharpe_matches_manual_mean_over_std():
    eras = ["0001", "0001", "0002", "0002"]
    preds = np.array([0.1, 0.2, 0.9, 0.4])
    target = np.array([0.2, 0.4, 0.8, 0.3])
    corr1 = float(np.corrcoef([0.1, 0.2], [0.2, 0.4])[0, 1])
    corr2 = float(np.corrcoef([0.9, 0.4], [0.8, 0.3])[0, 1])
    manual = (corr1 + corr2) / 2.0 / (np.std([corr1, corr2], ddof=0) + 1e-12)
    assert abs(_era_sharpe(preds, eras, target) - manual) < 1e-9


def _search_train_val():
    rng = np.random.default_rng(31)
    train_rows = []
    for era in range(1, 61):
        signal = float(era % 5)
        for row in range(8):
            train_rows.append({
                "era": f"{era:04d}", "id": f"t{era}_{row}",
                "f1": rng.normal(0, 1), "f2": rng.normal(0, 1),
                "main": signal + rng.normal(0, 0.5),
                "aux1": signal + rng.normal(0, 0.5),
                "aux2": rng.normal(0, 1),  # useless specialist
            })
    val_rows = [
        {"era": f"{era:04d}", "id": f"v{era}_{row}",
         "f1": rng.normal(0, 1), "f2": rng.normal(0, 1),
         "main": float(era % 5) + rng.normal(0, 0.5)}
        for era in (61, 62) for row in range(8)
    ]
    bench_rows = [
        {"era": f"{era:04d}", "id": f"v{era}_{row}", "v53_lgbm_ender20": rng.normal(0, 1)}
        for era in (61, 62) for row in range(8)
    ]
    return pl.DataFrame(train_rows), pl.DataFrame(val_rows), pl.DataFrame(bench_rows)


_SEARCH_PARAMS = {
    "mode": "search",
    "meta_tail_pct": 0.10,
    "snnr_weights": {"aux1": 0.9, "aux2": 0.1},
    "top_k": 2,
    "min_coverage": 0.5,
    "min_abs_main_corr": 0.01,
    "priority_hints": [],
    "specialist_alpha_grid": [0.01, 1.0, 100.0],
    "specialist_sharpe_floor": -10.0,   # permissive on synthetic noise
    "min_specialists": 1,
    "meta_alpha_grid": [0.01, 1.0],
    "meta_lgbm_params": {
        "max_depth": 2, "n_estimators": 10, "learning_rate": 0.1,
        "colsample_bytree": 0.8, "subsample": 0.8, "reg_lambda": 1.0,
        "early_stopping_rounds": 5, "valid_tail_pct": 0.2, "min_valid_eras": 1,
    },
    "decorr_grid": [0.0],
    "neutralization_grid": [0.0],
    "benchmark_col": "v53_lgbm_ender20",
    "nan_fill": True,
}


def test_ridge_stack_search_runs_and_ranks():
    train, val, bench = _search_train_val()
    out = generate_ridge_stack_predictions(
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"], params=_SEARCH_PARAMS, seed=42,
        val_targets=val.select(["era", "id", "main"]),
        benchmarks=bench,
    )
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == 16
    for era in out.get_column("era").unique().to_list():
        vals = out.filter(pl.col("era") == era).get_column("prediction").to_numpy()
        assert abs(float(vals.mean())) < 1e-6


def test_ridge_stack_search_requires_val_targets_and_benchmarks():
    train, val, bench = _search_train_val()
    with pytest.raises(ValueError, match="val_targets"):
        generate_ridge_stack_predictions(
            train, val, main_target="main", specialists=["aux1"],
            feature_cols=["f1"], params=_SEARCH_PARAMS, seed=42,
            benchmarks=bench,
        )
    with pytest.raises(ValueError, match="benchmarks"):
        generate_ridge_stack_predictions(
            train, val, main_target="main", specialists=["aux1"],
            feature_cols=["f1"], params=_SEARCH_PARAMS, seed=42,
            val_targets=val.select(["era", "id", "main"]),
        )


def test_ridge_stack_search_pruning_floor_raises():
    train, val, bench = _search_train_val()
    params = dict(_SEARCH_PARAMS)
    params["specialist_sharpe_floor"] = 1000.0
    with pytest.raises(ValueError, match="min_specialists"):
        generate_ridge_stack_predictions(
            train, val, main_target="main", specialists=["aux1", "aux2"],
            feature_cols=["f1", "f2"], params=params, seed=42,
            val_targets=val.select(["era", "id", "main"]),
            benchmarks=bench,
        )


def test_ridge_stack_search_is_seed_deterministic():
    train, val, bench = _search_train_val()
    a = generate_ridge_stack_predictions(
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"], params=_SEARCH_PARAMS, seed=42,
        val_targets=val.select(["era", "id", "main"]), benchmarks=bench,
    )
    b = generate_ridge_stack_predictions(
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"], params=_SEARCH_PARAMS, seed=42,
        val_targets=val.select(["era", "id", "main"]), benchmarks=bench,
    )
    assert a.equals(b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k "search or era_sharpe" -v`
Expected: FAIL — `ImportError: cannot import name '_era_sharpe'`.

- [ ] **Step 3: Implement** — append to `nmr/benchmark_fleet.py` (add `from lightgbm import early_stopping` to the imports):

```python
def _era_sharpe(preds: np.ndarray, eras: Sequence[str], target: np.ndarray) -> float:
    """Mean/std(ddof=0) of per-era Pearson CORR (notebook `_compute_era_sharpe`)."""
    frame = pl.DataFrame(
        {"prediction": preds, "era": list(eras), "target": target}
    ).drop_nulls()
    era_corrs: list[float] = []
    for _, era_frame in frame.group_by("era", maintain_order=True):
        if era_frame["prediction"].n_unique() < 2 or era_frame["target"].n_unique() < 2:
            continue
        p = era_frame.get_column("prediction").to_numpy()
        t = era_frame.get_column("target").to_numpy()
        era_corrs.append(float(np.corrcoef(p, t)[0, 1]))
    if not era_corrs:
        return -np.inf
    arr = np.asarray(era_corrs, dtype=float)
    return float(arr.mean() / (arr.std(ddof=0) + 1.0e-12))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _rank_values_per_era(values: np.ndarray, eras: Sequence[str]) -> np.ndarray:
    """Per-era rank-gaussianize a raw vector (Ensembler semantics)."""
    frame = pl.DataFrame({"__v": values, "era": list(eras)})
    ranked = Ensembler.rank_normalize(frame, pred_cols=["__v"], era_col="era")
    return ranked.get_column("__v").to_numpy()


def _per_era_corrs(values: np.ndarray, eras: Sequence[str], target: np.ndarray) -> np.ndarray:
    frame = pl.DataFrame(
        {"__v": values, "era": list(eras), "target": target}
    ).drop_nulls()
    out: list[float] = []
    for _, era_frame in frame.group_by("era", maintain_order=True):
        if era_frame["__v"].n_unique() < 2 or era_frame["target"].n_unique() < 2:
            continue
        out.append(float(np.corrcoef(
            era_frame.get_column("__v").to_numpy(),
            era_frame.get_column("target").to_numpy(),
        )[0, 1]))
    return np.asarray(out, dtype=float)


def _ridge_stack_search(
    train_rows: pl.DataFrame,
    val_rows: pl.DataFrame,
    *,
    main_target: str,
    specialists: list[str],
    feature_cols: list[str],
    params: Mapping[str, Any],
    seed: int,
    neutralization: float,
    val_targets: pl.DataFrame | None,
    benchmarks: pl.DataFrame | None,
    era_col: str,
    id_col: str,
    pred_col: str,
) -> pl.DataFrame:
    """v1.5.1-style config-driven specialist/meta search (selection-biased).

    Candidate selection uses validation (as the notebook did) — the runner
    flags the resulting scorecard with ``selection_bias: true``.
    """
    if val_targets is None:
        raise ValueError("search mode requires val_targets (selection uses validation)")
    if benchmarks is None:
        raise ValueError("search mode requires benchmarks (decorr sweep)")
    benchmark_col = str(params["benchmark_col"])
    if benchmark_col not in benchmarks.columns:
        raise ValueError(f"benchmarks missing column: {benchmark_col!r}")

    # 1. Target quality filter (coverage, corr to main, priority hints, top-k).
    weights = dict(params["snnr_weights"])
    quality: list[tuple[float, float, float, str]] = []
    for target in specialists:
        if target not in train_rows.columns:
            continue
        series = train_rows.get_column(target)
        coverage = float(series.drop_nulls().len() / max(1, series.len()))
        if coverage < float(params["min_coverage"]):
            continue
        aligned = train_rows.select([main_target, target]).drop_nulls()
        corr = _pearson(
            aligned.get_column(main_target).to_numpy(),
            aligned.get_column(target).to_numpy(),
        )
        if abs(corr) < float(params["min_abs_main_corr"]):
            continue
        hint_bonus = 1.0 if target in list(params.get("priority_hints", [])) else 0.0
        quality.append((hint_bonus, float(weights.get(target, 0.0)), corr, target))
    quality.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    selected = [row[3] for row in quality[: int(params["top_k"])]]
    if not selected:
        raise ValueError("no auxiliary targets survive the quality filter")

    spec_eras, meta_eras = _stack_partitions(
        sorted(train_rows.get_column(era_col).unique().to_list()),
        meta_tail_pct=float(params["meta_tail_pct"]),
        specialists=selected,
    )
    spec_rows = train_rows.filter(pl.col(era_col).is_in(spec_eras))
    meta_rows = train_rows.filter(pl.col(era_col).is_in(meta_eras))
    x_spec = spec_rows.select(feature_cols).cast(pl.Float32).to_numpy()
    x_meta = meta_rows.select(feature_cols).cast(pl.Float32).to_numpy()
    x_val = val_rows.select(feature_cols).cast(pl.Float32).to_numpy()

    # 2. Specialist alpha search: Sharpe on the meta tail.
    meta_main = meta_rows.get_column(main_target).cast(pl.Float64).to_numpy()
    meta_era_list = meta_rows.get_column(era_col).to_list()
    kept: list[tuple[str, float, object]] = []  # (target, alpha, model)
    for target in selected:
        y = spec_rows.get_column(target).cast(pl.Float64).to_numpy()
        mask = np.isfinite(y)
        if mask.sum() < 2:
            continue
        best: tuple[float, float, object] | None = None
        for alpha in params["specialist_alpha_grid"]:
            model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=seed)
            model.fit(x_spec[mask], y[mask])
            meta_raw = np.asarray(model.predict(x_meta), dtype=float)
            meta_ranked = _rank_values_per_era(meta_raw, meta_era_list)
            sharpe = _era_sharpe(meta_ranked, meta_era_list, meta_main)
            if best is None or sharpe > best[0]:
                best = (sharpe, float(alpha), model)
        if best is not None and best[0] >= float(params["specialist_sharpe_floor"]):
            kept.append((target, best[1], best[2]))
    if len(kept) < int(params["min_specialists"]):
        raise ValueError(
            f"only {len(kept)} specialists survive the Sharpe floor, "
            f"need >= min_specialists={params['min_specialists']}"
        )

    # 3. Meta features: per-era-ranked specialist predictions on meta tail + val.
    meta_X = meta_rows.select([era_col, id_col])
    val_X = val_rows.select([era_col, id_col])
    for target, _, model in kept:
        meta_X = meta_X.with_columns(
            pl.Series(target, np.asarray(model.predict(x_meta), dtype=float))
        )
        val_X = val_X.with_columns(
            pl.Series(target, np.asarray(model.predict(x_val), dtype=float))
        )
    kept_cols = [t for t, _, _ in kept]
    meta_X = Ensembler.rank_normalize(meta_X, pred_cols=kept_cols, era_col=era_col)
    val_X = Ensembler.rank_normalize(val_X, pred_cols=kept_cols, era_col=era_col)
    meta_y = meta_rows.select([era_col, id_col, main_target]).drop_nulls()
    meta_fit = meta_X.join(meta_y, on=[era_col, id_col], how="inner")
    if meta_fit.height < 2:
        raise ValueError("fewer than 2 aligned meta-train rows")
    meta_fit_X = meta_fit.select(kept_cols).cast(pl.Float32).to_numpy()
    meta_fit_y = meta_fit.get_column(main_target).cast(pl.Float64).to_numpy()
    val_meta_X = val_X.select(kept_cols).cast(pl.Float32).to_numpy()

    # 4. Meta candidates: non-negative ridge grid + shallow LGBM (internal es split).
    candidates: dict[str, np.ndarray] = {}
    for alpha in params["meta_alpha_grid"]:
        try:
            model = Ridge(alpha=float(alpha), positive=True, random_state=seed)
            model.fit(meta_fit_X, meta_fit_y)
        except TypeError:
            model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=seed)
            model.fit(meta_fit_X, meta_fit_y)
        candidates[f"ridge|alpha={float(alpha):.4g}"] = np.asarray(
            model.predict(val_meta_X), dtype=float
        )
    lgbm_params = dict(params["meta_lgbm_params"])
    es_rounds = int(lgbm_params.pop("early_stopping_rounds", 50))
    valid_tail_pct = float(lgbm_params.pop("valid_tail_pct", 0.2))
    min_valid_eras = int(lgbm_params.pop("min_valid_eras", 5))
    meta_era_sorted = meta_fit.get_column(era_col).unique().sort().to_list()
    if len(meta_era_sorted) > 1:
        n_valid = max(min_valid_eras, int(round(len(meta_era_sorted) * valid_tail_pct)))
        n_valid = min(n_valid, max(1, len(meta_era_sorted) - 1))
        valid_eras = set(meta_era_sorted[-n_valid:])
        is_valid = meta_fit.get_column(era_col).is_in(list(valid_eras)).to_numpy()
    else:
        is_valid = np.zeros(meta_fit.height, dtype=bool)
    lgbm_model = construct_tree_model(
        "lightgbm", lgbm_params, seed=seed,
        n_features=len(kept_cols), device="cpu",
    )
    if is_valid.any():
        lgbm_model.fit(
            meta_fit_X[~is_valid], meta_fit_y[~is_valid],
            eval_set=[(meta_fit_X[is_valid], meta_fit_y[is_valid])],
            callbacks=[early_stopping(es_rounds, verbose=False)],
        )
    else:
        lgbm_model.fit(meta_fit_X, meta_fit_y)
    candidates["lgbm"] = np.asarray(lgbm_model.predict(val_meta_X), dtype=float)

    # 5. Post-processing sweeps: benchmark decorr x neutralization; selection on
    #    validation mean CORR vs the main target (documented selection bias).
    val_main = val_targets.join(
        val_rows.select([era_col, id_col]), on=[era_col, id_col], how="inner"
    ).sort([era_col, id_col])
    val_main_y = val_main.get_column(main_target).cast(pl.Float64).to_numpy()
    val_era_list = val_main.get_column(era_col).to_list()
    bench_sorted = benchmarks.sort([era_col, id_col])
    bench_ranked = _rank_values_per_era(
        bench_sorted.get_column(benchmark_col).to_numpy(),
        bench_sorted.get_column(era_col).to_list(),
    )
    best: tuple[float, str, float, float, np.ndarray] | None = None
    for key in sorted(candidates):  # deterministic iteration order
        base = candidates[key]
        for decorr in params["decorr_grid"]:
            decorrelated = base - float(decorr) * bench_ranked
            for neu in params["neutralization_grid"]:
                ranked = _rank_values_per_era(decorrelated, val_era_list)
                era_corrs = _per_era_corrs(ranked, val_era_list, val_main_y)
                score = float(np.mean(era_corrs)) if era_corrs.size else -np.inf
                if best is None or score > best[0]:
                    best = (score, key, float(decorr), float(neu), decorrelated)
    if best is None:
        raise ValueError("no post-processing candidate survived")
    selected_raw = best[4]

    frame = val_main.select([era_col, id_col]).with_columns(
        pl.Series(pred_col, selected_raw)
    )
    if best[3] > 0.0:
        with_features = frame.join(
            val_rows.select([era_col, id_col, *feature_cols]),
            on=[era_col, id_col], how="inner",
        )
        frame = NeutralizationEngine().neutralize(
            with_features, pred_col=pred_col, feature_cols=feature_cols,
            era_col=era_col, proportion=best[3],
        ).select([era_col, id_col, pred_col])
    out = Ensembler().blend(
        Ensembler.rank_normalize(frame, pred_cols=[pred_col], era_col=era_col),
        pred_cols=[pred_col], weights=[1.0], era_col=era_col, out_col=pred_col,
    )
    return out.select([era_col, id_col, pred_col]).sort([era_col, id_col])
```

Note: the cell-level `neutralization` field is not applied by search mode — the sweep's selected proportion subsumes it (the `fa_v151_ridge_ensemble` config leaves the field unset).

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k "search or era_sharpe" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark_fleet.py tests/test_benchmark_fleet.py
git commit -m "feat(benchmark-fleet): ridge-stacking search mode (FA v1.5.1 recreation)"
```

---

### Task 9: `BenchmarkFleet` runner, placement, canonical bytes

**Files:**
- Modify: `nmr/benchmark_fleet.py` (runner + frame + placement + csv writer)
- Modify: `nmr/benchmark.py` (`canonical_scorecards_bytes` fleet parameter)
- Modify: `nmr/__init__.py`
- Test: `tests/test_benchmark_fleet.py`, `tests/test_benchmark_hierarchy.py`

**Interfaces:**
- Produces:
  - `fleet_placement(corr: float, rungs: Mapping[int, float]) -> str`
  - `FleetResult` (frozen): `scorecards: Mapping[str, MetricScorecard]`, `sources: Mapping[str, str]`, `placements: Mapping[str, str]`, `gate_verdicts: Mapping[str, Mapping[str, bool | None]]`, `selection_bias: Mapping[str, bool]`
  - `BenchmarkFleet(*, spec: tuple[FleetCellConfig, ...], data: BenchmarkData, seed=42, horizon="20D", n_boot=1000, min_overlap_eras=20, fast_mode=False)` with `run(*, tier_rungs: Mapping[int, float], gate: Tier4GateConfig | None) -> FleetResult`
  - `fleet_frame(result: FleetResult) -> pl.DataFrame`, `write_fleet_csv(result, output_path) -> Path`
  - `canonical_scorecards_bytes(scorecards, fleet_scorecards: Mapping[str, MetricScorecard] | None = None) -> bytes` — combined mapping, raises on id collision
- Consumes: `BenchmarkData`, `resolve_benchmark_feature_cols`, `evaluate_model`, `scorecards_to_frame`, `tier4_gate_verdict`, all five fleet generators

- [ ] **Step 1: Write failing tests** — append to `tests/test_benchmark_fleet.py`:

```python
import json as _json
from pathlib import Path as _Path

from nmr.benchmark import BenchmarkData, load_benchmark_data
from nmr.benchmark_fleet import (
    BenchmarkFleet,
    fleet_frame,
    fleet_placement,
    write_fleet_csv,
)


def test_fleet_placement_edges_and_intervals():
    rungs = {0: 0.002, 1: 0.005, 2: 0.007, 3: 0.0095, 4: 0.029}
    assert fleet_placement(0.001, rungs) == "below tier 0"
    assert fleet_placement(0.030, rungs) == "above tier 4"
    assert fleet_placement(0.005, rungs) == "tier1..tier2"
    assert fleet_placement(0.0095, rungs) == "tier3..tier4"
    with pytest.raises(ValueError, match="rungs"):
        fleet_placement(0.01, {})


def test_fleet_placement_rejects_out_of_range_corr():
    with pytest.raises(ValueError, match="corr"):
        fleet_placement(1.5, {0: 0.002, 1: 0.005})


def _write_synthetic_benchmark_data(tmp_path: Path) -> BenchmarkData:
    """Minimal on-disk v5.3-like assets for a full fleet run."""
    rng = np.random.default_rng(9)
    eras_train = [f"{e:04d}" for e in range(1, 41)]
    eras_val = [f"{e:04d}" for e in range(41, 47)]
    rows_train = [
        {"era": era, "id": f"t{era}_{i}",
         "f1": rng.normal(0, 1), "f2": rng.normal(0, 1),
         "target": rng.normal(0, 1), "target_aux": rng.normal(0, 1)}
        for era in eras_train for i in range(6)
    ]
    rows_val = [
        {"era": era, "id": f"v{era}_{i}", "f1": rng.normal(0, 1), "f2": rng.normal(0, 1),
         "target": rng.normal(0, 1)}
        for era in eras_val for i in range(6)
    ]  # val carries `target` because evaluate_model(main_target="target") needs the join
    pl.DataFrame(rows_train).write_parquet(tmp_path / "train.parquet")
    pl.DataFrame(rows_val).write_parquet(tmp_path / "validation.parquet")
    pl.DataFrame([
        {"era": era, "id": f"v{era}_{i}", "numerai_meta_model": rng.normal(0, 1)}
        for era in eras_val for i in range(6)
    ]).write_parquet(tmp_path / "meta_model.parquet")
    pl.DataFrame([
        {"era": era, "id": f"v{era}_{i}", "v53_lgbm_ender20": rng.normal(0, 1)}
        for era in eras_val for i in range(6)
    ]).write_parquet(tmp_path / "validation_benchmark_models.parquet")
    features = {
        "feature_sets": {"small": ["f1", "f2"], "medium": ["f1", "f2"]},
        "targets": ["target"],
    }
    (tmp_path / "features.json").write_text(_json.dumps(features), encoding="utf-8")
    return load_benchmark_data(tmp_path)


def test_benchmark_fleet_run_scores_and_places(tmp_path):
    data = _write_synthetic_benchmark_data(tmp_path)
    cells = (
        FleetCellConfig(
            benchmark_id="silly_target_lag_mean", source="test",
            input_space="none", model_kind="target_lag_mean",
            targets=("target",), params={"window": 1},
        ),
        FleetCellConfig(
            benchmark_id="tutorial_hello_deep", source="test",
            input_space="small", model_kind="lightgbm",
            targets=("target",),
            params={"n_estimators": 10, "learning_rate": 0.1,
                    "max_depth": 2, "num_leaves": 4},
            fast_mode_params={"n_estimators": 5},
        ),
    )
    fleet = BenchmarkFleet(
        spec=cells, data=data, seed=42, horizon="20D", n_boot=1,
        min_overlap_eras=2, fast_mode=True,
    )
    result = fleet.run(
        tier_rungs={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, gate=None
    )
    assert set(result.scorecards) == {"silly_target_lag_mean", "tutorial_hello_deep"}
    assert result.gate_verdicts["silly_target_lag_mean"] == {}
    assert result.selection_bias == {
        "silly_target_lag_mean": False, "tutorial_hello_deep": False
    }
    for placement in result.placements.values():
        assert placement.startswith(("below", "above", "tier"))
    frame = fleet_frame(result)
    assert "placement" in frame.columns and "selection_bias" in frame.columns
    path = write_fleet_csv(result, tmp_path / "fleet.csv")
    assert path.exists()


def test_benchmark_fleet_search_cell_marks_selection_bias(tmp_path):
    data = _write_synthetic_benchmark_data(tmp_path)
    cell = FleetCellConfig(
        benchmark_id="fa_v151_ridge_ensemble", source="test",
        input_space="small", model_kind="ridge_stack",
        targets=("target",),
        # specialists must NOT include main_target (polars duplicate-column
        # projection would raise), and the search params must be complete.
        params={
            "mode": "search", "main_target": "target",
            "specialists": ["target_aux"],
            "meta_tail_pct": 0.10, "nan_fill": True,
            "snnr_weights": {"target_aux": 1.0},
            "top_k": 1, "min_coverage": 0.5, "min_abs_main_corr": 0.0,
            "priority_hints": [],
            "specialist_alpha_grid": [0.01, 1.0],
            "specialist_sharpe_floor": -10.0, "min_specialists": 1,
            "meta_alpha_grid": [0.01],
            "meta_lgbm_params": {
                "max_depth": 2, "n_estimators": 10, "learning_rate": 0.1,
                "colsample_bytree": 0.8, "subsample": 0.8, "reg_lambda": 1.0,
                "early_stopping_rounds": 5, "valid_tail_pct": 0.2,
                "min_valid_eras": 1,
            },
            "decorr_grid": [0.0], "neutralization_grid": [0.0],
            "benchmark_col": "v53_lgbm_ender20",
        },
    )
    fleet = BenchmarkFleet(
        spec=(cell,), data=data, seed=42, horizon="20D", n_boot=1,
        min_overlap_eras=2, fast_mode=True,
    )
    result = fleet.run(tier_rungs={}, gate=None)
    assert result.selection_bias["fa_v151_ridge_ensemble"] is True
```

Append to `tests/test_benchmark_hierarchy.py` (adapt helper names to that file's existing synthetic-scorecard helpers — read the file first):

```python
def test_canonical_bytes_include_fleet_scorecards_and_reject_collisions():
    base = make_two_scorecards()   # existing helper in this file
    fleet = {"fleet_extra": make_one_scorecard(model_id="fleet_extra")}
    solo = canonical_scorecards_bytes(base)
    combined = canonical_scorecards_bytes(base, fleet_scorecards=fleet)
    assert solo != combined
    with pytest.raises(ValueError, match="collision"):
        canonical_scorecards_bytes(
            base, fleet_scorecards={"t0": make_one_scorecard(model_id="t0")}
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k "placement or fleet_run or selection_bias" tests/test_benchmark_hierarchy.py -k canonical -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3a: Extend `canonical_scorecards_bytes`** — in `nmr/benchmark.py`, change the signature and the first lines (keep the rest of the body verbatim):

```python
def canonical_scorecards_bytes(
    scorecards: Mapping[str, MetricScorecard],
    fleet_scorecards: Mapping[str, MetricScorecard] | None = None,
) -> bytes:
    """Canonical, timing-stripped scorecard serialization for determinism.

    ``fleet_scorecards`` are merged into the same canonical payload so fleet
    determinism is covered by the same cross-process hash. Id collisions
    between the hierarchy and fleet mappings raise (both are scored domains).
    """
    if fleet_scorecards:
        collision = set(scorecards) & set(fleet_scorecards)
        if collision:
            raise ValueError(
                f"benchmark id collision between hierarchy and fleet: {sorted(collision)}"
            )
        scorecards = {**scorecards, **fleet_scorecards}
    frame = scorecards_to_frame(scorecards).sort("model_id")
    # (existing body unchanged from here on)
```

- [ ] **Step 3b: Implement runner + placement + frame** — append to `nmr/benchmark_fleet.py`:

```python
@dataclasses.dataclass(frozen=True)
class FleetResult:
    scorecards: Mapping[str, MetricScorecard]
    sources: Mapping[str, str]
    placements: Mapping[str, str]
    gate_verdicts: Mapping[str, Mapping[str, bool | None]]
    selection_bias: Mapping[str, bool]


def fleet_placement(corr: float, rungs: Mapping[int, float]) -> str:
    """Place a measured CORR against the per-tier max-corr ladder rungs."""
    if not rungs:
        raise ValueError("rungs must be non-empty")
    if (
        not isinstance(corr, (int, float)) or isinstance(corr, bool)
        or not -1.0 <= float(corr) <= 1.0
    ):
        raise ValueError(f"corr must be a float in [-1, 1], got {corr!r}")
    ordered = sorted(rungs.items())
    if corr < ordered[0][1]:
        return "below tier 0"
    if corr > ordered[-1][1]:
        return f"above tier {ordered[-1][0]}"
    for index in range(len(ordered) - 1):
        tier_low, value_low = ordered[index]
        tier_high, value_high = ordered[index + 1]
        if value_low <= corr < value_high:
            return f"tier{tier_low}..tier{tier_high}"
    return f"tier{ordered[-1][0]}"


class BenchmarkFleet:
    """Untiered fleet of benchmark models (spec 2026-08-19-benchmark-fleet-design)."""

    def __init__(
        self,
        *,
        spec: tuple[FleetCellConfig, ...],
        data: BenchmarkData,
        seed: int = DEFAULT_BENCHMARK_SEED,
        horizon: str = "20D",
        n_boot: int = 1000,
        min_overlap_eras: int = 20,
        fast_mode: bool = False,
    ) -> None:
        if not spec:
            raise ValueError("fleet spec has no cells")
        self._spec = spec
        self._data = data
        self._seed = int(seed)
        self._horizon = horizon
        self._n_boot = int(n_boot)
        self._min_overlap_eras = int(min_overlap_eras)
        self._fast_mode = bool(fast_mode)
        self._schema_cols = pl.read_parquet_schema(data.validation_path).names()
        self._target_cols = ["era", "id"] + [
            c for c in self._schema_cols if c == "target" or c.startswith("target_")
        ]

    def _feature_cols(self, cell: FleetCellConfig) -> list[str]:
        return resolve_benchmark_feature_cols(
            self._data.features_json, cell.input_space, self._schema_cols
        )

    def _cell_params(self, cell: FleetCellConfig) -> dict[str, Any]:
        params = dict(cell.params)
        if self._fast_mode and cell.fast_mode_params:
            params.update(dict(cell.fast_mode_params))
        return params

    def _predictions_for_cell(
        self, cell: FleetCellConfig
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        feature_cols = self._feature_cols(cell)
        params = self._cell_params(cell)
        val_id = pl.read_parquet(self._data.validation_path, columns=["era", "id"])

        if cell.input_space == "none":
            small_cols = resolve_benchmark_feature_cols(
                self._data.features_json, "small", self._schema_cols
            )
            val_features = pl.read_parquet(
                self._data.validation_path, columns=["era", "id", *small_cols]
            )
        else:
            val_features = pl.read_parquet(
                self._data.validation_path,
                columns=["era", "id", *feature_cols],
            )

        if cell.model_kind == "target_lag_mean":
            train_targets = pl.read_parquet(
                self._data.train_path, columns=["era", cell.targets[0]]
            )
            preds = generate_lagged_target_predictions(
                train_targets, val_id, target=cell.targets[0],
                window=int(params.get("window", 1)),
            )
            return preds, val_features

        if cell.model_kind == "ridge_stack":
            stack_targets = [
                str(params["main_target"]), *map(str, params["specialists"])
            ]
            train = pl.read_parquet(
                self._data.train_path,
                columns=["era", "id", *feature_cols, *stack_targets],
            )
            is_search = params.get("mode") == "search"
            val_cols = ["era", "id", *feature_cols]
            if is_search:
                val_cols.append(str(params["main_target"]))
            val = pl.read_parquet(
                self._data.validation_path, columns=val_cols
            )
            preds = generate_ridge_stack_predictions(
                train, val,
                main_target=str(params["main_target"]),
                specialists=[str(t) for t in params["specialists"]],
                feature_cols=feature_cols, params=params, seed=cell.seed,
                neutralization=float(cell.neutralization or 0.0),
                val_targets=(
                    val.select(["era", "id", str(params["main_target"])])
                    if is_search else None
                ),
                benchmarks=self._data.benchmarks,
            )
            return preds, val_features

        train = pl.read_parquet(
            self._data.train_path,
            columns=["era", "id", *feature_cols, *cell.targets],
        )
        val = pl.read_parquet(
            self._data.validation_path, columns=["era", "id", *feature_cols]
        )
        if cell.model_kind == "lightgbm":
            preds = generate_fleet_lightgbm_predictions(
                train, val, targets=list(cell.targets), feature_cols=feature_cols,
                params=params, seed=cell.seed,
                neutralization=float(cell.neutralization or 0.0),
                neutralizer_selection=cell.neutralizer_selection,
                neutralizer_count=cell.neutralizer_count,
            )
        elif cell.model_kind == "xgboost":
            preds = generate_fleet_xgb_predictions(
                train, val, targets=list(cell.targets), feature_cols=feature_cols,
                params=params, seed=cell.seed,
                target_weights=(
                    dict(cell.target_weights) if cell.target_weights else None
                ),
            )
        elif cell.model_kind == "mlp":
            preds = generate_mlp_predictions(
                train, val, target=cell.targets[0], feature_cols=feature_cols,
                params=params, seed=cell.seed,
            )
        else:
            raise ValueError(f"Unsupported fleet model kind: {cell.model_kind!r}")
        return preds, val_features

    def run(
        self,
        *,
        tier_rungs: Mapping[int, float],
        gate: Tier4GateConfig | None,
    ) -> FleetResult:
        scorecards: dict[str, MetricScorecard] = {}
        sources: dict[str, str] = {}
        selection_bias: dict[str, bool] = {}
        val_targets = pl.read_parquet(
            self._data.validation_path, columns=self._target_cols
        )
        for cell in self._spec:
            logger.info("[fleet] %s (kind=%s)", cell.benchmark_id, cell.model_kind)
            preds, val_features = self._predictions_for_cell(cell)
            scorecards[cell.benchmark_id] = evaluate_model(
                preds,
                meta_model=self._data.meta_model,
                benchmarks=self._data.benchmarks,
                features=val_features,
                targets=val_targets,
                n_trials=1,
                seed=cell.seed,
                horizon=self._horizon,
                main_target="target",
                benchmark_col=None,
                n_boot=self._n_boot,
                min_overlap_eras=self._min_overlap_eras,
                model_id=cell.benchmark_id,
            )
            sources[cell.benchmark_id] = cell.source
            selection_bias[cell.benchmark_id] = (
                cell.model_kind == "ridge_stack"
                and cell.params.get("mode") == "search"
            )
            if cell.anchors:
                measured = float(scorecards[cell.benchmark_id].corr.value)
                for key, anchor in cell.anchors.items():
                    logger.info(
                        "    anchor %s=%.4f (measured=%.6f)",
                        key, float(anchor), measured,
                    )
        placements = (
            {
                mid: fleet_placement(float(card.corr.value), tier_rungs)
                for mid, card in scorecards.items()
            }
            if tier_rungs else {}
        )  # empty ladder => no placements (placement is report-only)
        gate_verdicts: dict[str, Mapping[str, bool | None]] = {}
        for mid, card in scorecards.items():
            gate_verdicts[mid] = (
                tier4_gate_verdict(card, gate) if gate is not None else {}
            )
        return FleetResult(
            scorecards=scorecards,
            sources=sources,
            placements=placements,
            gate_verdicts=gate_verdicts,
            selection_bias=selection_bias,
        )


def fleet_frame(result: FleetResult) -> pl.DataFrame:
    """Scorecard rows + placement/selection-bias/gate-verdict columns."""
    frame = scorecards_to_frame(result.scorecards)
    extra = pl.DataFrame({
        "model_id": list(result.scorecards.keys()),
        "source": [result.sources[mid] for mid in result.scorecards],
        "placement": [result.placements[mid] for mid in result.scorecards],
        "selection_bias": [
            result.selection_bias[mid] for mid in result.scorecards
        ],
    })
    out = frame.join(extra, on="model_id", how="left")
    verdict_fields = (
        "corr", "corr_sharpe_ac", "fnc", "deflated_sharpe",
        "gain_to_pain_ratio", "cagr_1y", "turnover_mean",
    )
    for field in verdict_fields:
        out = out.with_columns(
            pl.Series(
                f"gate_{field}",
                [result.gate_verdicts[mid].get(field) for mid in sorted(result.scorecards)],
            )
        )  # scorecards_to_frame sorts by model_id — verdicts must follow that order
    return out.sort("model_id")


def write_fleet_csv(result: FleetResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"output_path must be a .csv file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fleet_frame(result).write_csv(path)
    return path
```

- [ ] **Step 3c: Update `nmr/__init__.py`** — add `BenchmarkFleet`, `FleetResult`, `fleet_frame`, `fleet_placement`, `write_fleet_csv` and the five generator names to the `from .benchmark_fleet import (...)` block and `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py tests/test_benchmark_hierarchy.py -v`
Expected: PASS. The synthetic `BenchmarkFleet.run` test covers `target_lag_mean`, `lightgbm`, and the search-bias flag; `mlp`/`xgboost`/`ridge_stack` full-run paths are covered by their generator tests plus the real-data smoke in Task 13.

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark_fleet.py nmr/benchmark.py nmr/__init__.py tests/test_benchmark_fleet.py tests/test_benchmark_hierarchy.py
git commit -m "feat(benchmark-fleet): BenchmarkFleet runner, placement, canonical-bytes coverage"
```

---

### Task 10: Fleet config YAMLs (19 cells)

**Files:**
- Create: `configs/benchmarks/fleet/fleet_silly.yaml`, `fleet_tutorials.yaml`, `fleet_community.yaml`, `fleet_finance_arena.yaml`
- Test: `tests/test_benchmark_fleet.py` (suite-load test over the real config dir)

**Interfaces:** consumes `load_fleet_suite_config` (Task 2); pure data otherwise.

- [ ] **Step 1: Write the four config files** — exact content per spec §6 (defaults omitted: seed 42, `neutralizer_selection: none`).

`configs/benchmarks/fleet/fleet_silly.yaml`:

```yaml
# Wave 0 — silly baselines (audit §1: rolling historic target mean).
cells:
  - benchmark_id: silly_target_lag_mean
    source: docs/05-notebooks (audit §1)
    input_space: none
    model_kind: target_lag_mean
    targets: [target]
    params: {window: 1}
```

`configs/benchmarks/fleet/fleet_tutorials.yaml`:

```yaml
# Wave 1 — tutorial models, small + deep (docs/05-notebooks/1..3). Shallow =
# notebook defaults (2k trees); deep = the notebooks' commented 30k-tree params.
# Notebook 3 applies no neutralization. canon_hello_numerai (small) already
# exists in tier 3, so only the deep hello variant is recreated here.
cells:
  - benchmark_id: tutorial_hello_deep
    source: docs/05-notebooks/1_hello_numerai.ipynb
    input_space: small
    model_kind: lightgbm
    targets: [target]
    params: {n_estimators: 30000, learning_rate: 0.001, max_depth: 10, num_leaves: 31, colsample_bytree: 0.1}
    fast_mode_params: {n_estimators: 50}

  - benchmark_id: tutorial_neutralized_small
    source: docs/05-notebooks/2_feature_neutralization.ipynb
    input_space: small
    model_kind: lightgbm
    targets: [target]
    params: {n_estimators: 2000, learning_rate: 0.01, max_depth: 5, num_leaves: 15, colsample_bytree: 0.1}
    neutralization: 0.5
    fast_mode_params: {n_estimators: 50}

  - benchmark_id: tutorial_neutralized_deep
    source: docs/05-notebooks/2_feature_neutralization.ipynb
    input_space: small
    model_kind: lightgbm
    targets: [target]
    params: {n_estimators: 30000, learning_rate: 0.001, max_depth: 10, num_leaves: 15, colsample_bytree: 0.1}
    neutralization: 0.5
    fast_mode_params: {n_estimators: 50}

  - benchmark_id: tutorial_ensemble_small
    source: docs/05-notebooks/3_target_ensemble.ipynb
    input_space: small
    model_kind: lightgbm
    targets: [target_ender_20, target_victor_20, target_xerxes_20, target_teager2b_20]
    params: {n_estimators: 2000, learning_rate: 0.01, max_depth: 5, num_leaves: 31, colsample_bytree: 0.1}
    fast_mode_params: {n_estimators: 50}

  - benchmark_id: tutorial_ensemble_deep
    source: docs/05-notebooks/3_target_ensemble.ipynb
    input_space: small
    model_kind: lightgbm
    targets: [target_ender_20, target_victor_20, target_xerxes_20, target_teager2b_20]
    params: {n_estimators: 30000, learning_rate: 0.001, max_depth: 10, num_leaves: 31, colsample_bytree: 0.1}
    fast_mode_params: {n_estimators: 50}
```

`configs/benchmarks/fleet/fleet_community.yaml`:

```yaml
# Wave 2 — community example scripts (docs/05-notebooks/community_notebooks/),
# shallow + deep. Deep = the scripts' recommended params (20k trees). v4/v4.1
# target mapping: nomi_v4_60 -> ender_60, jerome_v4_60 -> ender_60,
# nomi_v4_20 -> ender_20, jerome_v4_20 -> jeremy_20 (flagged assumption).
cells:
  - benchmark_id: community_example_shallow
    source: docs/05-notebooks/community_notebooks/example_model.py
    input_space: medium
    model_kind: lightgbm
    targets: [target]
    params: {n_estimators: 2000, learning_rate: 0.01, max_depth: 5, num_leaves: 31, colsample_bytree: 0.1}
    neutralization: 1.0
    neutralizer_selection: riskiest_50
    neutralizer_count: 50
    fast_mode_params: {n_estimators: 50}

  - benchmark_id: community_example_deep
    source: docs/05-notebooks/community_notebooks/example_model.py
    input_space: medium
    model_kind: lightgbm
    targets: [target]
    params: {n_estimators: 20000, learning_rate: 0.001, max_depth: 6, num_leaves: 64, colsample_bytree: 0.1}
    neutralization: 1.0
    neutralizer_selection: riskiest_50
    neutralizer_count: 50
    fast_mode_params: {n_estimators: 50}

  - benchmark_id: community_advanced_shallow
    source: docs/05-notebooks/community_notebooks/example_model_advanced.py
    input_space: medium
    model_kind: lightgbm
    targets: [target, target_ender_60, target_jeremy_20]
    params: {n_estimators: 2000, learning_rate: 0.01, max_depth: 5, num_leaves: 31, colsample_bytree: 0.1}
    neutralization: 1.0
    neutralizer_selection: riskiest_50
    neutralizer_count: 50
    fast_mode_params: {n_estimators: 50}

  - benchmark_id: community_advanced_deep
    source: docs/05-notebooks/community_notebooks/example_model_advanced.py
    input_space: medium
    model_kind: lightgbm
    targets: [target, target_ender_60, target_jeremy_20]
    params: {n_estimators: 20000, learning_rate: 0.001, max_depth: 6, num_leaves: 64, colsample_bytree: 0.1}
    neutralization: 1.0
    neutralizer_selection: riskiest_50
    neutralizer_count: 50
    fast_mode_params: {n_estimators: 50}

  - benchmark_id: community_sunshine_shallow
    source: docs/05-notebooks/community_notebooks/example_model_sunshine.py
    input_space: medium
    model_kind: lightgbm
    targets: [target_ender_20, target_ender_60, target_ralph_20, target_tyler_20, target_victor_20, target_waldo_20]
    params: {n_estimators: 2000, learning_rate: 0.01, max_depth: 5, num_leaves: 32, colsample_bytree: 0.1}
    neutralization: 0.5
    fast_mode_params: {n_estimators: 50}

  - benchmark_id: community_sunshine_deep
    source: docs/05-notebooks/community_notebooks/example_model_sunshine.py
    input_space: medium
    model_kind: lightgbm
    targets: [target_ender_20, target_ender_60, target_ralph_20, target_tyler_20, target_victor_20, target_waldo_20]
    params: {n_estimators: 20000, learning_rate: 0.001, max_depth: 6, num_leaves: 64, colsample_bytree: 0.1}
    neutralization: 0.5
    fast_mode_params: {n_estimators: 50}
```

`configs/benchmarks/fleet/fleet_finance_arena.yaml`:

```yaml
# Wave 3+4 — Finance Arena recreations (mined read-only from
# ../numer-AI/models/version_0/v0.x and version_1/v1.5). Deviations per spec:
# purged trimmed-train fit (8-era), NeutralizationEngine, single seed, CPU.
# The 17 SNNR specialists are pinned from
# ../numer-AI/exploratory_notebooks/outputs/snnr_weights_vs_correlation_v5.2.csv.
cells:
  - benchmark_id: fa_v02_xgb
    source: ../numer-AI/models/version_0/v0.2/finance_arena_v0.2.ipynb
    input_space: small
    model_kind: xgboost
    targets: [target]
    params: {n_estimators: 2000, learning_rate: 0.01, max_depth: 6, subsample: 0.8, colsample_bytree: 0.1, early_stopping_rounds: 50, holdout_era_frac: 0.1}
    fast_mode_params: {n_estimators: 50, early_stopping_rounds: 5}

  - benchmark_id: fa_v03_lgbm_mt
    source: ../numer-AI/models/version_0/v0.3/finance_arena_v0.3.ipynb
    input_space: small
    model_kind: lightgbm
    targets: [target, target_ender_20, target_victor_20]
    params: {n_estimators: 20000, learning_rate: 0.01, max_depth: 6, num_leaves: 64}
    neutralization: 0.5
    fast_mode_params: {n_estimators: 50}

  - benchmark_id: fa_v04_xgb_weighted
    source: ../numer-AI/models/version_0/v0.4/finance_arena_v0.4.ipynb
    input_space: small
    model_kind: xgboost
    targets: [target, target_jasper_20, target_teager2b_20, target_claudia_20]
    target_weights: {target: 0.35, target_jasper_20: 0.30, target_teager2b_20: 0.23, target_claudia_20: 0.12}
    params: {n_estimators: 2000, learning_rate: 0.01, max_depth: 6, subsample: 0.8, colsample_bytree: 0.1}
    neutralization: 0.35
    fast_mode_params: {n_estimators: 50}

  - benchmark_id: fa_v05_ridge_stack
    source: ../numer-AI/models/version_0/v0.5/finance_arena_v0.5.ipynb
    input_space: small
    model_kind: ridge_stack
    targets: [target_ender_20]
    params:
      mode: fixed
      main_target: target_ender_20
      meta_tail_pct: 0.30
      alpha: 1.0e-6
      meta_alpha: 1.0e-6
      specialists: [target_jasper_20, target_teager2b_20, target_claudia_20, target_rowan_20, target_waldo_20, target_ender_60, target_xerxes_20, target_jeremy_20, target_cyrusd_20, target_agnes_20, target_victor_20, target_ralph_20, target_caroline_20, target_delta_20, target_tyler_20, target_sam_20, target_echo_20]
    neutralization: 0.5

  - benchmark_id: fa_v060_mlp
    source: ../numer-AI/models/version_0/v0.6/finance_arena_v0.6.0.ipynb
    input_space: small
    model_kind: mlp
    targets: [target_ender_20]
    params:
      hidden_layer_sizes: [256, 128, 64]
      activation: relu
      solver: adam
      alpha: 0.001
      learning_rate_init: 0.001
      batch_size: 1024
      max_iter: 150
      early_stopping: true
      n_iter_no_change: 15
      validation_fraction: 0.05
    neutralization: 0.5
    fast_mode_params: {max_iter: 20, hidden_layer_sizes: [8, 4], batch_size: 32}

  - benchmark_id: fa_v150_ridge_stack_tail10
    source: ../numer-AI/models/version_1/v1.5/fa_v1.5.0_ridge_ensemble.ipynb
    input_space: small
    model_kind: ridge_stack
    targets: [target_ender_20]
    params:
      mode: fixed
      main_target: target_ender_20
      meta_tail_pct: 0.10
      alpha: 1.0e-6
      meta_alpha: 1.0e-6
      specialists: [target_jasper_20, target_teager2b_20, target_claudia_20, target_rowan_20, target_waldo_20, target_ender_60, target_xerxes_20, target_jeremy_20, target_cyrusd_20, target_agnes_20, target_victor_20, target_ralph_20, target_caroline_20, target_delta_20, target_tyler_20, target_sam_20, target_echo_20]

  - benchmark_id: fa_v151_ridge_ensemble
    source: ../numer-AI/models/version_1/v1.5/fa_v1.5.1_ridge_ensemble.ipynb
    input_space: small
    model_kind: ridge_stack
    targets: [target_ender_20]
    params:
      mode: search
      main_target: target_ender_20
      meta_tail_pct: 0.10
      nan_fill: true
      snnr_weights:
        target_jasper_20: 0.300272
        target_teager2b_20: 0.226785
        target_claudia_20: 0.090945
        target_rowan_20: 0.077177
        target_waldo_20: 0.061258
        target_ender_60: 0.041783
        target_xerxes_20: 0.034451
        target_jeremy_20: 0.034373
        target_cyrusd_20: 0.026915
        target_agnes_20: 0.020739
        target_victor_20: 0.019592
        target_ralph_20: 0.014911
        target_caroline_20: 0.014023
        target_delta_20: 0.013578
        target_tyler_20: 0.012408
        target_sam_20: 0.005609
        target_echo_20: 0.005179
      specialists: [target_jasper_20, target_teager2b_20, target_claudia_20, target_rowan_20, target_waldo_20, target_ender_60, target_xerxes_20, target_jeremy_20, target_cyrusd_20, target_agnes_20, target_victor_20, target_ralph_20, target_caroline_20, target_delta_20, target_tyler_20, target_sam_20, target_echo_20]
      top_k: 12
      min_coverage: 0.50
      min_abs_main_corr: 0.01
      priority_hints: [target_victor_20, target_xerxes_20, target_teager_20]
      specialist_alpha_grid: [0.01, 0.03162277660168379, 0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0, 316.22776601683796, 1000.0, 3162.2776601683795, 10000.0]
      specialist_sharpe_floor: 0.50
      min_specialists: 6
      meta_alpha_grid: [0.01, 0.05623413251903491, 0.31622776601683794, 1.7782794100389228, 10.0, 56.23413251903491, 316.22776601683796, 1778.2794100389228, 10000.0]
      meta_lgbm_params:
        max_depth: 3
        n_estimators: 500
        learning_rate: 0.03
        colsample_bytree: 0.8
        subsample: 0.8
        reg_lambda: 1.0
        early_stopping_rounds: 50
        valid_tail_pct: 0.20
        min_valid_eras: 5
      decorr_grid: [0.00, 0.05, 0.10]
      neutralization_grid: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
      benchmark_col: v53_lgbm_ender20
    fast_mode_params:
      specialist_alpha_grid: [0.01, 1.0, 10000.0]
      meta_alpha_grid: [0.01, 1.0, 10000.0]
      neutralization_grid: [0.0]
      decorr_grid: [0.00]
      meta_lgbm_params:
        max_depth: 3
        n_estimators: 50
        learning_rate: 0.1
        colsample_bytree: 0.8
        subsample: 0.8
        reg_lambda: 1.0
        early_stopping_rounds: 5
        valid_tail_pct: 0.20
        min_valid_eras: 1
```

- [ ] **Step 2: Add a config suite test** — append to `tests/test_benchmark_fleet.py`:

```python
_REPO_FLEET_DIR = Path(__file__).resolve().parents[1] / "configs" / "benchmarks" / "fleet"


def test_real_fleet_configs_load_with_all_19_cells():
    cells = load_fleet_suite_config(_REPO_FLEET_DIR)
    ids = [c.benchmark_id for c in cells]
    assert len(ids) == 19
    assert len(set(ids)) == 19
    assert "fa_v151_ridge_ensemble" in ids and "silly_target_lag_mean" in ids
    search = next(c for c in cells if c.benchmark_id == "fa_v151_ridge_ensemble")
    assert search.params["mode"] == "search"
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k "real_fleet or suite" -v`
Expected: PASS (4 files, 19 cells, no duplicates, schema clean).

- [ ] **Step 4: Commit**

```bash
git add configs/benchmarks/fleet tests/test_benchmark_fleet.py
git commit -m "feat(benchmark-fleet): 19 fleet cell configs (silly, tutorials, community, FA)"
```

---

### Task 11: Runner CLI wiring

**Files:**
- Modify: `benchmark_runner.py`
- Test: `tests/test_benchmark_fleet.py` (parser tests via `_parse_args_with`)

**Interfaces:**
- Consumes: `load_fleet_suite_config`, `BenchmarkFleet`, `tier_max_corrs`, `write_fleet_csv`
- Produces: CLI flags `--fleet-configs`, `--fleet-output`, `--no-fleet`

- [ ] **Step 1: Write failing tests** — append to `tests/test_benchmark_fleet.py`:

```python
from pathlib import Path

from benchmark_runner import _parse_args_with


def test_runner_parser_fleet_defaults():
    args = _parse_args_with([])
    assert args.fleet_configs == Path("configs") / "benchmarks" / "fleet"
    assert args.fleet_output == (
        Path("artifacts") / "reports" / "benchmark_fleet_scorecard.csv"
    )
    assert args.no_fleet is False


def test_runner_parser_no_fleet_flag():
    args = _parse_args_with(["--no-fleet"])
    assert args.no_fleet is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k runner_parser -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'fleet_configs'`.

- [ ] **Step 3: Implement** — in `benchmark_runner.py`:

Extend the imports:

```python
from nmr.benchmark_fleet import BenchmarkFleet, load_fleet_suite_config, write_fleet_csv
from nmr.benchmark import tier_max_corrs
```

Add parser arguments after `--gate-report`:

```python
    parser.add_argument(
        "--fleet-configs",
        type=Path,
        default=Path("configs") / "benchmarks" / "fleet",
    )
    parser.add_argument(
        "--fleet-output",
        type=Path,
        default=Path("artifacts") / "reports" / "benchmark_fleet_scorecard.csv",
    )
    parser.add_argument("--no-fleet", action="store_true")
```

In `main()`, after the gate-report logging block and before the hard-failure check:

```python
    if not args.no_fleet:
        try:
            fleet_cells = load_fleet_suite_config(args.fleet_configs)
        except ValueError as exc:
            log.error("FLEET CONFIG FAILURE: %s", exc)
            return 1
        rungs = tier_max_corrs(result.scorecards, result.tier_of)
        fleet = BenchmarkFleet(
            spec=fleet_cells,
            data=data,
            seed=args.seed,
            horizon=args.horizon,
            n_boot=1 if args.fast_mode else args.n_boot,
            min_overlap_eras=args.min_overlap_eras,
            fast_mode=args.fast_mode,
        )
        t1 = time.perf_counter()
        log.info(
            "Running %d fleet cells%s",
            len(fleet_cells), " (fast mode)" if args.fast_mode else "",
        )
        fleet_result = fleet.run(tier_rungs=rungs, gate=spec.gate)
        log.info("Fleet scored in %.1fs", time.perf_counter() - t1)
        write_fleet_csv(fleet_result, args.fleet_output)
        log.info("Fleet scorecard written to %s", args.fleet_output)
        for mid in fleet_result.scorecards:
            log.info(
                "fleet %s: placement=%s selection_bias=%s",
                mid, fleet_result.placements[mid],
                fleet_result.selection_bias[mid],
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_fleet.py -k runner_parser -v`
Expected: PASS. Also run `./.venv/Scripts/python -m pytest tests/test_benchmark_config.py -q` (existing runner tests stay green).

- [ ] **Step 5: Commit**

```bash
git add benchmark_runner.py tests/test_benchmark_fleet.py
git commit -m "feat(benchmark-fleet): runner CLI fleet wiring"
```

---

### Task 12: Documentation & SSOT (same commit)

**Files:**
- Modify: `docs/06-evaluation/benchmark-line-in-the-sand.md`
- Modify: `docs/superpowers/specs/2026-08-15-benchmark-hierarchy-design.md` (§10 one-line amendment)
- Modify: `ARCHITECTURE.md` (benchmark module area)
- Modify: `AGENTS.md` (toolkit row, hazard, test count)

- [ ] **Step 1: `benchmark-line-in-the-sand.md`** — append after the tier-4 section:

```markdown
## Untiered Benchmark Fleet

A fourth config layer, `configs/benchmarks/fleet/`, holds benchmark models
with **no tier assignment**: silly heuristics, tutorial small/deep variants,
community example scripts (shallow/deep), and the Finance Arena v0.2-v1.5.1
recreations — 19 cells. They are scored through the identical
`evaluate_model` pipeline and reported in
`artifacts/reports/benchmark_fleet_scorecard.csv` with a `placement` column
(measured CORR vs the per-tier max-corr rungs), informational tier-4 gate
verdicts, and a `selection_bias` flag (true only for the v1.5.1 search cell,
whose candidate selection uses validation — never compare it naively).

Fleet results never participate in the hard gates (null floor, tier-4 gate,
monotonicity). Anchors are report-only and re-pinned after measurement.
Full design: `docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md`.
```

- [ ] **Step 2: Hierarchy spec amendment** — append to §10 of `docs/superpowers/specs/2026-08-15-benchmark-hierarchy-design.md`:

```markdown
5. **Untiered benchmark fleet (2026-08-19):** a new untiered config layer
   (`configs/benchmarks/fleet/`, `nmr/benchmark_fleet.py`) adds 19 recreated
   community/tutorial/Finance-Arena benchmark models scored through the same
   evaluation pipeline, with report-only placement against the tier rungs.
   Tiers, gates, and monotonicity semantics are unchanged. Design:
   `docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md`.
```

- [ ] **Step 3: `ARCHITECTURE.md`** — in the benchmark module area, add a subsection mirroring the existing style:

```markdown
### Untiered benchmark fleet (`nmr/benchmark_fleet.py`)

Fleet config schema (`FleetCellConfig` — the tiered cell fields minus `tier`,
plus `source`, `target_weights`, `neutralizer_selection`, `neutralizer_count`),
five generators (`target_lag_mean` — trailing-train target mean; fleet
`lightgbm` — canonical fits + optional riskiest-50 neutralizer selection via
`feature_stability_screen`; fleet `xgboost` — multi-target weighted rank
blend + optional tail-holdout early stopping; `mlp` — sklearn MLPRegressor
with `_standardize_feature_block`; `ridge_stack` — fixed/search two-layer
ridge stacking, horizon-aware 8/16-era internal purge, search mode =
config-driven grids with validation-based candidate selection), and the
`BenchmarkFleet` runner (scored via `evaluate_model`, report-only placement
vs per-tier max-corr rungs, tier-4 verdict columns). Runner CLI:
`--fleet-configs` (default `configs/benchmarks/fleet`), `--fleet-output`
(default `artifacts/reports/benchmark_fleet_scorecard.csv`), `--no-fleet`.
Fleet scorecards join `canonical_scorecards_bytes`. Spec:
`docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md`.
```

- [ ] **Step 4: `AGENTS.md`** — add a toolkit row after the benchmark-hierarchy row:

```markdown
| Change the untiered benchmark fleet (configs, generators, runner) | `nmr/benchmark_fleet.py` + `configs/benchmarks/fleet/` (spec: `docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md`) |
```

In §8 operational hazards, after the benchmark-hierarchy runtime hazard:

```markdown
### Fleet deep-cell runtime & selection bias (2026-08-19)
Fleet deep cells (20k/30k-tree LightGBM fits on ~2.1M train rows) are multi-hour CPU jobs; a full 19-cell fleet run is tens of CPU-hours across waves — use `nohup` + log polling; fast-mode overrides keep the smoke gate minutes-scale. `fa_v151_ridge_ensemble` is **selection-biased by design** (candidate selection uses validation, as the notebook did) — its scorecard row carries `selection_bias: true`; never compare it naively against unbiased cells. Fleet results never participate in the hard gates.
```

Then update the pytest test-count claim in §1 (and CI comments if named) to the new measured total — run `pytest -q` first and use the actual number.

- [ ] **Step 5: Commit**

```bash
git add docs/06-evaluation/benchmark-line-in-the-sand.md docs/superpowers/specs/2026-08-15-benchmark-hierarchy-design.md ARCHITECTURE.md AGENTS.md
git commit -m "docs: fleet layer SSOT updates (line-in-the-sand, hierarchy spec, architecture, agents)"
```

---

### Task 13: Verification gates

- [ ] **Step 1: Lint**

Run: `./.venv/Scripts/python -m ruff check .`
Expected: clean (E/F/I/UP @120). Fix findings, re-run until clean, commit fixes.

- [ ] **Step 2: Fast functional gate**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: all pass (927 pre-existing + new fleet tests). Report the actual count; re-verify the AGENTS.md claim matches (Task 12 Step 4).

- [ ] **Step 3: Real-data smoke (fleet on)**

Run: `./.venv/Scripts/python benchmark_runner.py --fast-mode`
Expected: `artifacts/reports/benchmark_hierarchy_scorecard_smoke.csv`, `benchmark_gate_report_smoke.csv`, and `benchmark_fleet_scorecard.csv` all written; every fleet row has `placement` and `selection_bias`; hard gates pass (fast-mode monotonicity is warning-only). A crashing cell exits non-zero with the cell id logged — fix before proceeding.

- [ ] **Step 4: Wave-by-wave full fleet runs**

Run in the background (multi-hour; use `nohup` so the session can close):

```bash
nohup ./.venv/Scripts/python benchmark_runner.py > artifacts/reports/fleet_full_run.log 2>&1 &
```

Poll with `tail -f artifacts/reports/fleet_full_run.log` (stage markers + `[fleet]` lines). Expected total: tens of CPU-hours, dominated by the 20k/30k-tree cells (`tutorial_*_deep`, `community_*_deep`, `fa_v03_lgbm_mt`) and the v1.5.1 search cell. Record each cell's measured CORR, placement, and gate verdicts; re-pin fleet `anchors` in a follow-up commit if adopted (evidence-driven, hierarchy-spec decision #2 procedure).

- [ ] **Step 5: End-of-session gate**

Run: `./.venv/Scripts/python -m ruff check .` then `./.venv/Scripts/python -m pytest -q`
Expected: both green on the final state. Report actual results truthfully, including any skips (e.g. missing `data/v5.3` parquets skip real-data tests — say so).

---

## Plan Self-Review Notes

- Spec coverage: §1-§5 (architecture, schema, engine, determinism) → Tasks 1-9; §6 roster → Task 10; §7 runner/reporting → Task 11; §8 tests → each task's Step 1; §9 docs → Task 12; §12 verification → Task 13. No spec section lacks a task.
- Deliberate inline decisions: search mode ignores the cell-level `neutralization` field (the sweep's selected proportion subsumes it; config leaves it unset); `_select_riskiest_features` ranks by `cross_regime_variance` desc with a name tie-break; `_era_sharpe` uses `ddof=0` per the notebook.
- Type consistency: `FleetCellConfig` field names are used verbatim across Tasks 2, 9, 10; generator signatures (`main_target`, `specialists`, `feature_cols`, `params`, `purge_eras`, `era_col`, `id_col`, `pred_col`) are identical across Tasks 3-8; `BenchmarkFleet.run(*, tier_rungs, gate)` matches the Task 11 caller.
- `tests/test_benchmark_hierarchy.py` helper names (`make_two_scorecards`, `make_one_scorecard`) are illustrative — the implementer must adapt to that file's actual synthetic-scorecard helpers.
