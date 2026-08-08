# Bayesian HPO via Optuna Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved BO-HPO spec: a new `nmr/opt.py` module exposing `bayesian_sweep(...)` (Optuna TPE, deterministic, declarative space dict), an additive `corr_sharpe_ac` metric branch in `research._held_out_metric`, a public `models.resolve_model_params` helper, the pinned `optuna==4.9.0` dependency, and same-change-set docs + S2-skill updates.

**Architecture:** `nmr/opt.py` is the only Optuna-importing module. The objective is harness-internal (materialize trial params via `_override_config` → evaluate via `_held_out_metric`). Sweeps are seeded (`TPESampler(seed, deterministic=True)`), single-threaded (`n_jobs=1` asserted), in-memory storage, and return the existing `SweepResult` contract built post-hoc from `study.trials`. Preset resolution stays in `models.py` via the new public `resolve_model_params`; `opt.py` intersects it with the space for the baseline anchor.

**Tech Stack:** Python 3.11+ (venv 3.12), Polars, NumPy/SciPy, LightGBM/XGBoost, **Optuna 4.9.0** (user-granted exception to the no-new-deps rule, 2026-08-08).

## Global Constraints

- `nmr/` is the only tested boundary; scripts and skills contain zero business logic (AGENTS §2.1).
- TDD: no production code without a failing test first (AGENTS §7). Every new function has a test that was watched fail.
- **Dependency exception (user-granted 2026-08-08):** Optuna may be added, pinned to `optuna==4.9.0`, imported ONLY in `nmr/opt.py`. No other new third-party dependencies.
- Determinism: seeded TPE; `n_jobs=1` hard invariant (asserted); trials reproducible cross-process under pinned versions; per-environment caveat (GPU vs CPU) documented, same as runs.
- `SweepResult` schema unchanged (four columns: `trial_id`, `params_json`, `metric_value`, `metric`; params serialized, NOT expanded). `HyperparameterSweep.run` contract unchanged.
- No metric-formula changes, no purge-geometry changes, nothing enters `canonical_scorecards_bytes`. `optuna` NOT added to the run_id environment fingerprint.
- Categorical choices restricted to JSON primitives (str/int/float/bool); `log=True` requires `low > 0`; both fail early.
- `corr_sharpe_ac` in `_held_out_metric` sorts era keys chronologically (numeric) before `ac_adjusted_sharpe(..., horizon="20D")`.
- Failed trials raise `optuna.exceptions.TrialPruned` (never return dummy numerics); trials log built post-hoc from `study.trials` (COMPLETE → value, else None); failures loud via `logger.error`.
- Doc SSOT same-change-set; AGENTS.md ≤ 32 KB; test-count claims in AGENTS.md/README.md/CONTRIBUTING.md synced numeric-only (precedent).
- Git flow (user-authorized pattern): work on branch `bo-hpo` (created), commits per task, `main` untouched until the user chooses integration. No push.
- Verification honesty: run commands, read output, report truthfully.

## File Structure

| File | Responsibility |
|---|---|
| `nmr/opt.py` (new) | Space validation + `bayesian_sweep` (only Optuna-importing module) |
| `nmr/models.py` (modify) | Public `resolve_model_params(preset, params)`; `_resolved_params` delegates |
| `nmr/research.py` (modify) | `_held_out_metric` `corr_sharpe_ac` branch (chronological sort) |
| `nmr/__init__.py` (modify) | Export `bayesian_sweep` |
| `requirements.txt` (modify) | Pin `optuna==4.9.0` |
| `.venv` (modify) | Install optuna (project venv) |
| `tests/test_opt.py` (new) | All `nmr/opt.py` tests |
| `tests/test_models.py`, `tests/test_research.py`, `tests/test_contribution.py` (modify) | Additive cases |
| `ARCHITECTURE.md`, `AGENTS.md`, `README.md` (modify) | §S, carve-out + toolkit row, tree entry |
| `.kimi-code/skills/hpo-narrowing/SKILL.md` (modify) | S2 stage-2 → `bayesian_sweep` |

---

### Task 1: `nmr/models.py` — public `resolve_model_params`

**Files:**
- Modify: `nmr/models.py:36-50` (after `_CANONICAL_PRESETS`) and `nmr/models.py:301-303` (`_resolved_params`)
- Test: `tests/test_models.py` (append)

**Interfaces:**
- Consumes: `_CANONICAL_PRESETS` (existing).
- Produces: `resolve_model_params(preset: str, params: dict[str, Any]) -> dict[str, Any]` — module-level, `_CANONICAL_PRESETS[preset].copy(); update(params)`. Consumed by Task 5 (`bayesian_sweep` anchor) and `_resolved_params`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_models.py
from nmr.models import ModelOrchestrator, resolve_model_params


def test_resolve_model_params_merges_preset_and_overrides():
    resolved = resolve_model_params("fast", {"n_estimators": 2500})
    assert resolved["n_estimators"] == 2500          # override wins
    assert resolved["learning_rate"] == 0.01         # preset default present
    assert resolved["num_leaves"] == (2**5) - 1      # preset default present


def test_resolve_model_params_matches_orchestrator_resolution():
    cfg = ModelConfig(backend="lightgbm", preset="fast",
                      params={"n_estimators": 2500, "colsample_bytree": 0.2})
    orch = ModelOrchestrator(cfg, seed=42)
    # _resolved_params(use_gpu=False) adds backend boilerplate; the preset+params
    # core must equal resolve_model_params for the same inputs.
    resolved = orch._resolved_params(use_gpu=False)
    for key, value in resolve_model_params("fast", cfg.params).items():
        assert resolved[key] == value


def test_resolve_model_params_unknown_preset_raises():
    import pytest as _pytest

    with _pytest.raises(KeyError):
        resolve_model_params("bogus", {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -q`
Expected: FAIL with `ImportError: cannot import name 'resolve_model_params'`.

- [ ] **Step 3: Write minimal implementation**

```python
# nmr/models.py — after the _CANONICAL_PRESETS block
def resolve_model_params(preset: str, params: dict[str, Any]) -> dict[str, Any]:
    """Resolve preset defaults overridden by explicit ``params``.

    Single source of truth for preset+params resolution (ARCHITECTURE.md §S):
    ``model.params`` wins over ``_CANONICAL_PRESETS[preset]``. Used by
    :meth:`ModelOrchestrator._resolved_params` and the Bayesian sweep anchor.
    """
    resolved = dict(_CANONICAL_PRESETS[preset])
    resolved.update(params)
    return resolved
```

```python
# nmr/models.py — _resolved_params, replace the first two lines:
    base = resolve_model_params(self._config.preset, self._config.params)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -q`
Expected: PASS (new + existing). Also confirm the parity test's key loop passes (backend boilerplate keys like `objective`/`random_state` are not preset keys, so only the preset+params keys are compared).

- [ ] **Step 5: Commit** — `feat(models): add public resolve_model_params` on `bo-hpo`.

---

### Task 2: `nmr/research.py` — `corr_sharpe_ac` metric branch in `_held_out_metric`

**Files:**
- Modify: `nmr/research.py:297-307` (the `evaluator.summarize` + `hasattr` guard block in `_held_out_metric`)
- Test: `tests/test_research.py` (append)

**Interfaces:**
- Consumes: `nmr.inference.ac_adjusted_sharpe` (add import).
- Produces: `_per_era_ac_sharpe(per_era: dict[str, float], *, horizon: str = "20D") -> float` — numerically sorts era keys then computes `ac_adjusted_sharpe`; and `_held_out_metric(config, *, metric_name)` now accepts `corr_sharpe_ac` (delegating to the helper) in addition to `mean/std/sharpe/max_drawdown`. Consumed by Task 5 (objective).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_research.py
import numpy as np

from nmr.inference import ac_adjusted_sharpe
from nmr.research import _per_era_ac_sharpe


def test_per_era_ac_sharpe_sorts_eras_chronologically() -> None:
    # Eras 1..12: insertion order deliberately shuffled; the numeric sort must
    # recover the chronological series ("1","2",...,"12", NOT lexicographic
    # "1","10","11","12","2",...).
    values = {str(era): 0.05 * (era % 7) - 0.1 for era in range(1, 13)}
    items = list(values.items())
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(items))
    shuffled = dict(items[int(i)] for i in idx)

    got = _per_era_ac_sharpe(shuffled, horizon="20D")
    chronological = [values[str(era)] for era in sorted(range(1, 13))]
    expected = ac_adjusted_sharpe(chronological, horizon="20D")
    assert got == expected


def test_per_era_ac_sharpe_differs_from_lexicographic_order() -> None:
    # Prove the sort matters: a lexicographic series gives a different AC value.
    values = {str(era): 0.05 * (era % 7) - 0.1 for era in range(1, 13)}
    lexicographic = [values[k] for k in sorted(values)]  # "1","10","11","12","2",...
    assert _per_era_ac_sharpe(values, horizon="20D") != ac_adjusted_sharpe(
        lexicographic, horizon="20D"
    )


def test_per_era_ac_sharpe_requires_two_eras() -> None:
    from nmr.research import _per_era_ac_sharpe

    with pytest.raises(ValueError, match="at least 2"):
        _per_era_ac_sharpe({"1": 0.1}, horizon="20D")


def test_held_out_metric_supports_corr_sharpe_ac(tmp_path) -> None:
    from nmr.research import _held_out_metric

    cfg = _write_data(tmp_path)  # existing fixture: vtest, fast preset, small trees
    value = _held_out_metric(cfg, metric_name="corr_sharpe_ac")
    assert np.isfinite(value)


def test_held_out_metric_still_rejects_unknown_metric(tmp_path) -> None:
    from nmr.research import _held_out_metric

    cfg = _write_data(tmp_path)
    with pytest.raises(ValueError, match="Unknown metric"):
        _held_out_metric(cfg, metric_name="bogus_metric")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_research.py -q`
Expected: FAIL — `corr_sharpe_ac` currently raises `ValueError("Unknown metric 'corr_sharpe_ac'")` (the `hasattr` guard).

- [ ] **Step 3: Write minimal implementation**

```python
# nmr/research.py — add to imports:
from nmr.inference import ac_adjusted_sharpe

# nmr/research.py — add near _held_out_metric:
def _per_era_ac_sharpe(per_era: dict[str, float], *, horizon: str = "20D") -> float:
    """Autocorrelation-adjusted Sharpe of a per-era metric series.

    Chronological order is mandatory for autocorrelation: ``per_era``'s
    insertion order follows the frame's lexicographic era sort ("1","10","11",...)
    which would corrupt the AC computation. Sort era keys numerically
    (scorecard._sorted_numeric_keys idiom); era labels are numeric strings.
    """
    sorted_keys = sorted(per_era, key=int)
    series = [per_era[k] for k in sorted_keys]
    return ac_adjusted_sharpe(series, horizon=horizon)

# nmr/research.py — replace the tail of _held_out_metric (research.py:297-307):
    evaluator = EvaluationEngine(config.evaluation.backend)
    per_era = evaluator.per_era_corr(
        neutralized,
        pred_col="prediction",
        target_col=main_target,
        era_col="era",
    )
    if metric_name == "corr_sharpe_ac":
        return _per_era_ac_sharpe(per_era, horizon="20D")
    summary = evaluator.summarize(per_era)
    if not hasattr(summary, metric_name):
        raise ValueError(f"Unknown metric {metric_name!r}")
    return float(getattr(summary, metric_name))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_research.py -q`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit** — `feat(research): support corr_sharpe_ac in held-out metric` on `bo-hpo`.

---

### Task 3: Pin + install `optuna==4.9.0`

**Files:**
- Modify: `requirements.txt` (append `optuna==4.9.0`)

**Interfaces:**
- Consumes: nothing.
- Produces: Optuna available in `.venv` for Tasks 4–5 (configuration change — no test required per the TDD exception for config files; verify import works).

- [ ] **Step 1: Pin the dependency**

Append to `requirements.txt`:
```
optuna==4.9.0
```

- [ ] **Step 2: Install into the project venv**

Run: `.venv/Scripts/python -m pip install optuna==4.9.0`
Expected: installs; then verify: `.venv/Scripts/python -c "import optuna; print(optuna.__version__)"` → `4.9.0`. Confirm `optuna.logging.set_verbosity`, `optuna.storages.InMemoryStorage`, `optuna.trial.TrialState.COMPLETE`, `optuna.exceptions.TrialPruned`, and `TPESampler(seed=..., deterministic=True)` all exist in 4.9.0 (quick `hasattr` checks; do NOT guess — verify in the installed package).

- [ ] **Step 3: Commit** — `build(deps): pin optuna==4.9.0 (user-granted exception)` on `bo-hpo`.

---

### Task 4: `nmr/opt.py` — space validation (pure functions)

**Files:**
- Create: `nmr/opt.py` (only `_parse_space`/validation for now; the module imports optuna + models.resolve_model_params at top)
- Test: `tests/test_opt.py` (new file, first batch)

**Interfaces:**
- Consumes: nothing from earlier tasks (validation is pure).
- Produces: `_parse_space(space: dict) -> list[_SpaceParam]` with frozen `_SpaceParam(kind, name, low, high, log, step, choices)`; raises `ValueError` per the spec's validation matrix. Consumed by Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_opt.py
from __future__ import annotations

import pytest

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
    ],
)
def test_parse_space_validation_errors(space, match) -> None:
    with pytest.raises(ValueError, match=match):
        _parse_space(space)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_opt.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nmr.opt'`.

- [ ] **Step 3: Write minimal implementation**

```python
# nmr/opt.py
"""Bayesian hyperparameter optimization via Optuna (user-granted dependency).

``bayesian_sweep`` is the single Optuna-integration point. Space definitions are
declarative dicts (ARCHITECTURE.md §S); the objective is harness-internal
(``research._held_out_metric``); sweeps are seeded, single-threaded, and
deterministic per environment.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import optuna

from nmr.config import ExperimentConfig
from nmr.models import resolve_model_params
from nmr.research import SweepResult, _held_out_metric, _override_config

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("nmr.opt")

__all__ = ["bayesian_sweep"]

_VALID_METRICS = ("mean", "std", "sharpe", "max_drawdown", "corr_sharpe_ac")
_JSON_PRIMITIVES = (str, int, float, bool)


@dataclass(frozen=True)
class _SpaceParam:
    kind: Literal["float", "int", "categorical"]
    name: str
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    step: int | None = None
    choices: tuple[Any, ...] = ()


def _parse_space(space: dict[str, dict[str, Any]]) -> list[_SpaceParam]:
    if not space:
        raise ValueError("search space must contain at least one parameter")
    parsed: list[_SpaceParam] = []
    for name, spec in space.items():
        if not isinstance(spec, dict):
            raise ValueError(f"parameter {name!r}: spec must be a dict")
        unknown = set(spec) - {"kind", "low", "high", "log", "step", "choices"}
        if unknown:
            raise ValueError(f"parameter {name!r}: unknown keys {sorted(unknown)}")
        kind = spec.get("kind")
        if kind not in ("float", "int", "categorical"):
            raise ValueError(f"parameter {name!r}: kind must be float/int/categorical")
        if kind in ("float", "int"):
            low, high = spec.get("low"), spec.get("high")
            if low is None or high is None or low > high:
                raise ValueError(f"parameter {name!r}: low/high bounds invalid")
            log = bool(spec.get("log", False))
            if log and low <= 0:
                raise ValueError(
                    f"parameter {name!r}: 'low' must be > 0 when log=True, got {low}"
                )
            step = spec.get("step")
            if step is not None and (not isinstance(step, int) or step < 1):
                raise ValueError(f"parameter {name!r}: step must be a positive int")
            if log and step is not None:
                raise ValueError(
                    f"parameter {name!r}: log=True and step are mutually exclusive"
                )
            parsed.append(
                _SpaceParam(kind=kind, name=name, low=low, high=high,
                            log=log, step=step)
            )
        else:
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"parameter {name!r}: choices must be a non-empty list")
            if not all(isinstance(c, _JSON_PRIMITIVES) for c in choices):
                raise ValueError(
                    f"parameter {name!r}: categorical choices must be str/int/float/bool"
                )
            parsed.append(_SpaceParam(kind="categorical", name=name, choices=tuple(choices)))
    return parsed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_opt.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat(opt): add declarative space parsing and validation` on `bo-hpo`.

---

### Task 5: `nmr/opt.py` — `bayesian_sweep`

**Files:**
- Modify: `nmr/opt.py` (append `bayesian_sweep` + objective + `_suggest`)
- Test: `tests/test_opt.py` (append)

**Interfaces:**
- Consumes: `_parse_space` (Task 4), `research._held_out_metric`/`_override_config`/`SweepResult` (existing), `models.resolve_model_params` (Task 1).
- Produces: `bayesian_sweep(base_config, space, *, n_trials, seed, metric="sharpe", n_startup_trials=10, enqueue_base_config=True, n_jobs=1) -> SweepResult` — the public HPO entry consumed by the S2 skill.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_opt.py
import json

import polars as pl

from nmr.opt import bayesian_sweep


def _sweep_config(tmp_path):
    # Reuse the synthetic-data pattern from tests/test_runner.py (vtest fixture).
    # Implement a local builder identical to test_runner._config/_write_synthetic_data
    # (copy the fixture code; small frames, fast preset, n_estimators=10, n_folds=2).
    from tests.test_runner import _config
    return _config(tmp_path)


def test_bayesian_sweep_is_deterministic_under_seed(tmp_path) -> None:
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
    cfg = _sweep_config(tmp_path)
    with pytest.raises(ValueError, match="n_jobs"):
        bayesian_sweep(cfg, {"num_leaves": {"kind": "int", "low": 4, "high": 32}},
                       n_trials=2, seed=7, n_jobs=2)


def test_bayesian_sweep_supports_corr_sharpe_ac_metric(tmp_path) -> None:
    cfg = _sweep_config(tmp_path)
    space = {"learning_rate": {"kind": "float", "low": 0.01, "high": 0.1, "log": True}}
    result = bayesian_sweep(cfg, space, n_trials=2, seed=7, metric="corr_sharpe_ac")
    assert result.trials.get_column("metric").to_list() == ["corr_sharpe_ac"] * 2
    assert result.trials.get_column("metric_value").is_finite().all()


def test_bayesian_sweep_failed_trial_recorded_and_continues(tmp_path) -> None:
    cfg = _sweep_config(tmp_path)
    # num_leaves < 0 makes LightGBM raise inside _held_out_metric -> TrialPruned.
    space = {"num_leaves": {"kind": "int", "low": -8, "high": -1}}
    result = bayesian_sweep(cfg, space, n_trials=3, seed=7, n_startup_trials=2)
    assert result.trials.height == 3                     # synchronized with study.trials
    assert result.trials.get_column("metric_value").null_count() == 3
    assert result.best_params == {} or result.best_value is not None  # best may be empty


def test_bayesian_sweep_metrics_reject_unknown(tmp_path) -> None:
    cfg = _sweep_config(tmp_path)
    with pytest.raises(ValueError, match="metric"):
        bayesian_sweep(cfg, {"num_leaves": {"kind": "int", "low": 4, "high": 32}},
                       n_trials=2, seed=7, metric="corr")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_opt.py -q`
Expected: FAIL with `ImportError: cannot import name 'bayesian_sweep'` (or `AttributeError` for missing pieces).

- [ ] **Step 3: Write minimal implementation**

```python
# append to nmr/opt.py
def _suggest(trial: optuna.Trial, param: _SpaceParam) -> Any:
    if param.kind == "float":
        return trial.suggest_float(param.name, param.low, param.high, log=param.log)
    if param.kind == "int":
        kwargs: dict[str, Any] = {"log": param.log}
        if param.step is not None:
            kwargs["step"] = param.step
        return trial.suggest_int(param.name, param.low, param.high, **kwargs)
    return trial.suggest_categorical(param.name, list(param.choices))


def bayesian_sweep(
    base_config: ExperimentConfig,
    space: dict[str, dict[str, Any]],
    *,
    n_trials: int,
    seed: int,
    metric: str = "sharpe",
    n_startup_trials: int = 10,
    enqueue_base_config: bool = True,
    n_jobs: int = 1,
) -> SweepResult:
    """Bayesian hyperparameter sweep over ``space`` around ``base_config``.

    Seeded TPE sampler (``TPESampler(seed=...)`` — deterministic-by-default
    since Optuna 4.x, which removed the 3.x ``deterministic`` flag; verified
    on 4.9.0), single-threaded (``n_jobs`` must be 1 — parallel trials break
    TPE determinism), in-memory storage.
    Trial 0 evaluates the resolved baseline (preset defaults + ``model.params``,
    intersected with the space) when ``enqueue_base_config`` is true.
    Returns the standard :class:`SweepResult` (ARCHITECTURE.md §S).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_startup_trials < 1:
        raise ValueError("n_startup_trials must be >= 1")
    if n_jobs != 1:
        raise ValueError(
            f"n_jobs must be 1 (parallel trials break TPE determinism); got {n_jobs}"
        )
    if metric not in _VALID_METRICS:
        raise ValueError(f"metric={metric!r} not in {sorted(_VALID_METRICS)}")

    parsed = _parse_space(space)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=optuna.storages.InMemoryStorage(),
    )
    if enqueue_base_config:
        resolved = resolve_model_params(base_config.model.preset, base_config.model.params)
        anchor = {p.name: resolved[p.name] for p in parsed if p.name in resolved}
        if anchor:
            study.enqueue_trial(anchor)

    def objective(trial: optuna.Trial) -> float:
        params = {p.name: _suggest(trial, p) for p in parsed}
        cfg = _override_config(base_config, params)
        try:
            value = _held_out_metric(cfg, metric_name=metric)
        except Exception as exc:
            logger.error("[bayesian_sweep] trial %s failed: %s", trial.number, exc)
            raise optuna.exceptions.TrialPruned(f"trial failed: {exc}") from exc
        finally:
            gc.collect()
        return float(value)

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    rows = []
    for t in study.trials:
        value = t.value if t.state == optuna.trial.TrialState.COMPLETE else None
        rows.append(
            {
                "trial_id": t.number,
                "params_json": json.dumps(t.params, sort_keys=True),
                "metric_value": value,
                "metric": metric,
            }
        )
    trial_df = pl.DataFrame(rows).sort(
        ["metric_value", "trial_id"], descending=[True, False], nulls_last=True
    )
    best = study.best_trial if len(study.best_trials) > 0 else None
    return SweepResult(
        trials=trial_df,
        best_params=best.params if best is not None else {},
        best_value=float(best.value) if best is not None else float("nan"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_opt.py -q`
Expected: PASS. If `_sweep_config` reuse from `tests.test_runner` fails (imports across test modules), inline the fixture builder in `tests/test_opt.py` instead — same code, no cross-test-module import.

- [ ] **Step 5: Commit** — `feat(opt): add deterministic bayesian_sweep` on `bo-hpo`.

---

### Task 6: Exports + SSOT docs + S2 skill + count sync

**Files:**
- Modify: `nmr/__init__.py` (import + `__all__`), `tests/test_contribution.py` (append export test)
- Modify: `ARCHITECTURE.md` (§S + §3 graph + `models.py`/`research.py` notes), `AGENTS.md` (carve-out + toolkit row), `README.md` (tree), `.kimi-code/skills/hpo-narrowing/SKILL.md` (S2 stage 2)
- Modify: AGENTS.md/README.md/CONTRIBUTING.md count claims (numeric-only)

**Interfaces:**
- Consumes: Tasks 1–5 deliverables.
- Produces: public `nmr.bayesian_sweep`; docs per the spec; S2 skill stage-2 BO protocol.

- [ ] **Step 1: Write the failing test (exports contract)**

```python
# append to tests/test_contribution.py
def test_public_api_includes_bayesian_sweep():
    import nmr

    assert "bayesian_sweep" in nmr.__all__
    assert nmr.bayesian_sweep is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_contribution.py -q`
Expected: FAIL (`bayesian_sweep` not exported).

- [ ] **Step 3: Implement exports**

```python
# nmr/__init__.py — add to the research imports block:
from .opt import bayesian_sweep
# and to __all__ (alphabetical):
    "bayesian_sweep",
```

- [ ] **Step 4: Docs (same change set, SSOT)**

- `ARCHITECTURE.md` — add **§S `nmr/opt.py`** after §R: space schema (dict form, log/step, categorical JSON-primitive rule, `low>0` when log), metric resolution (incl. `corr_sharpe_ac` with mandatory numeric-era sort, `horizon="20D"`), baseline anchor (`resolve_model_params` ∩ space), parameter-resolution rule (`model.params` wins over preset — cross-ref §G), determinism contract (seeded TPE, `n_jobs=1` invariant, per-environment caveat, pinned-version reproducibility), error handling (`TrialPruned`, never dummy numerics), post-hoc `SweepResult` construction (COMPLETE → value else None). Update §3 dependency graph (opt → config/models/research) and the §G `_resolved_params` note (delegates to `resolve_model_params`). Update the §L `_held_out_metric` description (metric set now incl. `corr_sharpe_ac`).
- `AGENTS.md` — §3 Absolute Prohibitions: append the carve-out — "🚫 Never add third-party dependencies when the stdlib, NumPy/SciPy, or Polars can do the job. **EXCEPTION (user-granted 2026-08-08): Optuna** (pinned in `requirements.txt`) for the HPO path — imported only in `nmr/opt.py`; parallel trial execution is forbidden (`n_jobs=1`)." Toolkit table row: `Change HPO search strategy | nmr/opt.py — bayesian_sweep (Optuna, user-granted dep)`.
- `README.md` — annotated tree: add `nmr/opt.py` line.
- `.kimi-code/skills/hpo-narrowing/SKILL.md` — stage 2 protocol: narrow around the top-k from stage 1 via `bayesian_sweep(base_config, space, *, n_trials, seed, metric, enqueue_base_config=True, n_jobs=1)` with a declarative dict space; metrics `{mean, std, sharpe, max_drawdown, corr_sharpe_ac}`; note the per-environment reproducibility caveat and that Trial 0 is the resolved baseline.
- Count sync: compute the new total (355 + tests added across Tasks 1–6) and update the numeric claims in AGENTS.md/README.md/CONTRIBUTING.md (numeric-only, precedent).

- [ ] **Step 5: Run the focused + full suites**

Run: `.venv/Scripts/python -m pytest tests/test_contribution.py tests/test_opt.py tests/test_models.py tests/test_research.py -q` then `.venv/Scripts/python -m pytest -q` then `.venv/Scripts/python -m pytest tests/test_docs_hygiene.py -q`
Expected: all green; counts synced.

- [ ] **Step 6: Commit** — `feat: export bayesian_sweep and update SSOT docs + S2 skill` on `bo-hpo`.

---

### Task 7: Full verification gate

**Files:** none (verification only).

- [ ] **Step 1: Full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: PASS, all tests (355 baseline + new). Report the exact count.

- [ ] **Step 2: Benchmark smoke (`_held_out_metric` is an evaluation-path change)**

Run: `.venv/Scripts/python benchmark_runner.py --fast-mode --output artifacts/benchmark_scores_smoke.csv --labels-output artifacts/benchmark_test_era_labels_smoke.csv`
Expected: exits 0.

- [ ] **Step 3: Doc-SSOT scan** — AGENTS ≤ 32 KB; §S content matches `nmr/opt.py` signatures/behavior; no duplicated facts across the four files; `feature_subset`/`bayesian_sweep` fact homes correct.

- [ ] **Step 4: Record** — report results truthfully; do not claim anything not run.

---

## Self-Review

**Spec coverage** (spec section → task):
- Architecture/baseline anchor → Task 5 (+ models helper Task 1, anchor-resolution rule). ✔
- Metric resolution incl. `corr_sharpe_ac` + chronological sort → Task 2. ✔
- Parameter-resolution rule (documented) → Task 6 ARCHITECTURE §S. ✔
- Optuna hygiene (verbosity, InMemoryStorage, n_jobs) → Tasks 3/5. ✔
- Space schema validation (structural, log>0, categorical primitives) → Task 4. ✔
- Determinism contract → Task 5 tests. ✔
- Error handling (TrialPruned, post-hoc trials) → Task 5. ✔
- Output contract (4 columns, params_json, post-hoc) → Task 5. ✔
- gc.collect → Task 5. ✔
- Docs + S2 skill + count sync → Task 6. ✔
- Verification gate → Task 7. ✔
- Review amendments 2.1–2.5 all present (anchor resolution Task 1/5; log bounds + categorical primitives Task 4; chronological sort Task 2; post-hoc trials Task 5). ✔

**Placeholder scan:** the only deferral is the Optuna-version resolution, now pinned (`optuna==4.9.0`, verified available via pip index). All code steps carry real code.

**Type consistency:** `resolve_model_params(preset, params)` (Task 1) consumed by Task 5 anchor; `_parse_space → list[_SpaceParam]` (Task 4) consumed by Task 5; `_held_out_metric(..., metric_name="corr_sharpe_ac")` (Task 2) consumed by Task 5 objective; `bayesian_sweep` signature matches the spec and Task 5/6 tests; `SweepResult` fields match the existing contract.
