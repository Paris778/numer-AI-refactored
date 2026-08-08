# Bayesian HPO via Optuna — Design

> Status: approved 2026-08-08 (user), revised 2026-08-08 per architecture review #1 (9 findings: 7 accepted, 2 rejected with evidence), revised 2026-08-08 per edge-case review #2 (5 hazards: all accepted). Sub-project 1 of the external-library exception (BO → CatBoost → dashboard, in that sequence).

**Goal:** Add a Bayesian-optimization HPO path to the `nmr` research harness using Optuna, expressed as a declarative search-space dict, deterministic under seed, and wired into the S2 (`hpo-narrowing`) skill's stage-2 narrowing loop.

## Decisions (user-confirmed + review-adjudicated)

| Decision | Choice | Rationale |
|---|---|---|
| Library | **Optuna** (user-granted exception to the no-new-deps rule, 2026-08-08) | TPE sampler seeded via `seed` (deterministic-by-default since Optuna 4.x — the 3.x `deterministic` flag was removed; verified empirically on 4.9.0); first-class LightGBM/XGBoost integration; actively maintained |
| Space expression | **Declarative dict form (canonical), no positional tuples** | Serializable, auditable, config-driven; avoids tuple-length ambiguity for agents generating raw configs |
| API surface | **New module `nmr/opt.py`** with `bayesian_sweep(...)` | Isolates the Optuna import to one module; `HyperparameterSweep.run` (deterministic random/Cartesian) stays backward-compatible |
| Review 1.3 | **`corr_sharpe_ac` supported** | Primary promotion/ranking metric (`RunRegistry.promote_if_better` defaults to it); proxy-only objectives misalign with champion gates |
| Review 3.2 | **`n_jobs=1` non-negotiable** | Seeded TPE determinism breaks under parallel trials (async completion ordering); models already force `n_jobs: 1` internally (`models.py:309,322`) — parallel Optuna on top would oversubscribe |

## Architecture

- New tested module `nmr/opt.py` — the ONLY module that imports `optuna`.
- Public API: `bayesian_sweep(base_config: ExperimentConfig, space: dict, *, n_trials: int, seed: int, metric: str = "sharpe", n_startup_trials: int = 10, enqueue_base_config: bool = True, n_jobs: int = 1) -> SweepResult`.
- The objective is harness-internal: each trial's params are materialized via `research._override_config` (merging into `model.params` only — see Parameter Resolution Rule) and evaluated via `research._held_out_metric`.
- Returns the existing `SweepResult` contract (`trials: pl.DataFrame`, `best_params: dict`, `best_value: float`).
- `HyperparameterSweep.run()` in `nmr/research.py` is NOT modified — brute-force/random path preserved for small spaces.

### Baseline anchor (review 2.1 — resolution defect)

`study.enqueue_trial` only fixes the keys present in the enqueued dict; space keys missing from it fall through to (random/TPE) sampling — so the anchor must cover **every resolvable space key**, and must NOT include keys outside the space. Preset resolution is `models.py`'s source of truth; `nmr/opt.py` never re-implements it and never imports `_CANONICAL_PRESETS` directly. Instead:

- `nmr/models.py` gains a small public helper, e.g. `resolve_model_params(preset: str, params: dict[str, Any]) -> dict[str, Any]`, implemented as `preset_defaults.copy(); params.update(...)` — the exact logic `_resolved_params` (`models.py:301-303`) already uses; `_resolved_params` delegates to it (additive, tested; no behavior change).
- When `enqueue_base_config=True`, `bayesian_sweep` computes:
  ```python
  resolved = resolve_model_params(base_config.model.preset, base_config.model.params)
  anchor_params = {k: resolved[k] for k in space if k in resolved}
  if anchor_params:
      study.enqueue_trial(anchor_params)
  ```
  Space keys absent from the resolved defaults are intentionally not anchored (they get sampled — expected for novel params). `enqueue_trial` runs before `study.optimize(...)`, so Trial 0 tests the resolved baseline (preset defaults + `model.params` overrides) for every anchored key.

### Metric resolution (review 1.3 + 2.4)

`_held_out_metric` in `nmr/research.py` is extended (additive, tested) to resolve:
- `mean` / `std` / `sharpe` / `max_drawdown` → `getattr(MetricSummary, metric_name)` (existing behavior).
- `corr_sharpe_ac` → **explicitly chronological** era series, then `ac_adjusted_sharpe`:
  ```python
  # per_era_corr insertion order follows the frame's lexicographic era sort
  # ("1","10","11",...), which would corrupt autocorrelation — sort numerically.
  sorted_keys = sorted(per_era, key=int)          # matches scorecard._sorted_numeric_keys
  metric_series = [per_era[k] for k in sorted_keys]
  return ac_adjusted_sharpe(metric_series, horizon="20D")
  ```
  The `horizon="20D"` default matches `ExperimentRunner._run_validation_stage`'s hardcoded validation horizon; era labels are numeric strings (same assumption as `feature_exposure_report`). Horizon inference from target-name suffix is a possible future refinement, out of scope now.
- Any other metric name → `ValueError` (unchanged).

### Parameter Resolution Rule (review 1.3 — documented, verified against code)

`ModelConfig` has exactly three fields (`backend`, `preset`, `params`; `config.py`). Presets (`_CANONICAL_PRESETS`) own the defaults; resolution is `preset.copy(); params.update(...)` (`models.py:301-303`) — **`model.params` is the single override channel and wins over preset defaults**. There are no top-level hyperparameter attributes to route to; `_override_config` merging trial params into `model.params` is the complete, correct resolution. No other mapping exists or is needed.

### Optuna hygiene (review 1.3)

- `optuna.logging.set_verbosity(optuna.logging.WARNING)` at module import — suppress per-trial INFO noise on stderr.
- `optuna.create_study(..., storage=optuna.storages.InMemoryStorage())` — explicit in-memory storage; no ghost `.db`/SQLite files in the repo root or `artifacts/`.
- `study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)` with a runtime guard: `n_jobs != 1 → ValueError` (parallel trial execution is forbidden; determinism invariant + CPU oversubscription).

## Space schema

Canonical dict form, validated fail-loud at call time (`ValueError` naming the offending key):

```
space = {
    "learning_rate": {"kind": "float", "low": 0.005, "high": 0.05, "log": True},
    "n_estimators":  {"kind": "int", "low": 100, "high": 10000, "log": True, "step": 100},
    "num_leaves":    {"kind": "int", "low": 16, "high": 256},
    "boosting":      {"kind": "categorical", "choices": ["gbdt", "dart"]},
}
```

Mapping to Optuna:
- `float`: `trial.suggest_float(name, low, high, log=log)` — `log` optional (default False).
- `int`: `trial.suggest_int(name, low, high, log=log, step=step)` — `log` and `step` optional; `step` must be a positive int; `log=True` and `step` are mutually exclusive (Optuna constraint).
- `categorical`: `trial.suggest_categorical(name, choices)` — non-empty list of JSON-primitive values (see below).

Validation errors (each raises `ValueError` with a matchable message, ALL before any trial runs):
- Structural: unknown kind; missing/unknown keys; `low > high`; `step < 1` or non-int; `log` not boolean; empty space; `n_trials < 1`; `n_startup_trials < 1`; `n_jobs != 1`; metric not in `{mean, std, sharpe, max_drawdown, corr_sharpe_ac}`.
- **Log-scale positivity (review 2.2):** if `log=True`, require `low > 0` for both `float` and `int` — fail early, before any trial starts (`suggest_*` would otherwise raise inside the first trial): `if spec.get("log", False) and spec["low"] <= 0: raise ValueError(f"Key '{key}': 'low' must be > 0 when log=True")`.
- **Categorical primitives (review 2.3):** choices must be JSON-serializable primitives to preserve the `params_json` round-trip contract (`json.dumps`/`json.loads` would otherwise mutate tuples into lists, breaking equality): `if not all(isinstance(c, (str, int, float, bool)) for c in choices): raise ValueError(f"Key '{key}': categorical choices must be str/int/float/bool")`. Empty choices also raise.

## Determinism & reproducibility contract

- Study created with `TPESampler(seed=seed)` — seeded trial generation; deterministic-by-default since Optuna 4.x (the 3.x `deterministic` flag was removed; verified on 4.9.0: identical seeds ⇒ identical trial sequences). Trial 0 (baseline anchor) is deterministic by construction.
- Evaluations are the harness's bit-deterministic `_held_out_metric` (custom backend, CPU). Same config + space + seed + pinned dependencies ⇒ identical trial sequence, identical best params, cross-process.
- **`n_jobs=1` is a hard invariant** (enforced by assertion): parallel trials break seeded-TPE determinism and oversubscribe CPU (models already run `n_jobs: 1` internally).
- Documented caveats (mirroring the existing GPU/CPU run caveat):
  1. Sweep reproducibility is per-environment — GPU-vs-CPU evaluation differences cascade through BO adaptivity.
  2. Optuna's TPE internals can change across versions — sweeps are reproducible under pinned versions (`requirements.txt` pins Optuna).
- `optuna` is deliberately NOT added to the run_id environment fingerprint (`ExperimentRunner._environment_fingerprint`): it never affects a single run's model results. Fingerprint stays `{numpy, polars, pandas, lightgbm, xgboost}`.

## Error handling (review 1.1)

- **Failed trials NEVER return dummy numerics to Optuna** (a `-inf`/`NaN` return value would corrupt the TPE Parzen-window KDE or be silently mishandled). Inside the objective:
  ```python
  try:
      value = _held_out_metric(cfg, metric_name=metric)
  except Exception as exc:
      logger.error("[bayesian_sweep] trial %s failed: %s", trial.number, exc)
      raise optuna.exceptions.TrialPruned(f"trial failed: {exc}") from exc
  ```
  `TrialPruned` marks the trial PRUNED in the study; TPE excludes pruned trials from its density estimation, and `study.best_trial` only selects completed trials with finite values.
- The sweep's own trials log records failed trials with `metric_value = None` and the four shared `SweepResult` columns — identical schema to `HyperparameterSweep.run()` (see Output Contract); failures are loud (`logger.error` per failed trial) and never silent.
- All validation errors raise immediately (fail early), before any trial runs.

## Output contract (review 1.4 + 2.5 — post-hoc construction)

`SweepResult.trials` is a Polars DataFrame with columns `trial_id`, `params_json`, `metric_value`, `metric`, sorted by `metric_value` desc — **byte-identical schema to `HyperparameterSweep.run()`** (which stores params serialized in `params_json`, `research.py:80-96`; NOT expanded columns). Consumers (S2 skill, config materialization) read params via `json.loads(params_json)`. No schema expansion: changing the shared contract would break existing consumers.

**Post-hoc construction (review 2.5):** `SweepResult.trials` is built AFTER `study.optimize()` returns, by iterating `study.trials` — never accumulated inside the objective (a `TrialPruned`/exception aborts the objective body, so in-objective accumulation would miss rows and desync from Optuna's state):

```python
rows = []
for t in study.trials:
    value = t.value if t.state == optuna.trial.TrialState.COMPLETE else None
    rows.append({"trial_id": t.number, "params_json": json.dumps(t.params, sort_keys=True),
                 "metric_value": value, "metric": metric})
```

`best_params`/`best_value` come from `study.best_trial` (completed trials only); failed/pruned trials appear as `metric_value = None` rows. This guarantees 100% state synchronization with Optuna.

## Resource cleanup (review 1.3)

The objective wrapper calls `gc.collect()` at the end of each trial evaluation to keep memory bounded across long sweeps (Polars frames, NumPy arrays, LightGBM/XGBoost C-objects; refcounting frees most, `gc` handles cycles).

## Testing (`tests/test_opt.py`)

- Same-seed determinism: two `bayesian_sweep` calls with identical args produce identical `trials`, `best_params`, `best_value`.
- Baseline anchor (2.1): with `enqueue_base_config=True`, trial 0's params equal `resolve_model_params(preset, model.params)` intersected with `space.keys()` for every anchored key; a space key NOT in the resolved defaults is not anchored (sampled). With `enqueue_base_config=False`, no anchoring.
- `corr_sharpe_ac` metric (2.4): produces the `ac_adjusted_sharpe` of the held-out per-era corr series; on synthetic data with eras `"1".."12"` (lexicographically misordered dict input), the result equals the value computed from numerically-sorted eras — proving chronological ordering.
- Metric validation: `mean/std/sharpe/max_drawdown/corr_sharpe_ac` accepted; anything else raises.
- Space validation (2.2, 2.3): `log=True` with `low <= 0` raises; categorical with a tuple/list/None choice raises; empty categorical raises; plus the structural classes (unknown kind, inverted bounds, `step < 1`, `log+step` on int, empty space).
- `n_jobs=2` raises `ValueError`.
- Failed-trial handling (2.5): a space value that makes training raise → trial marked failed (null `metric_value` row in `trials`), sweep continues, `best` comes from finite trials, and the trials frame is synchronized with `study.trials` (count matches).
- Distribution mapping: float log/linear, int log/step, categorical — values within bounds/choices.
- `SweepResult` contract: four columns; `best_params` round-trips through `params_json` (incl. categorical primitives).
- `models.resolve_model_params` unit tests: preset defaults + params overrides win; matches `_resolved_params` output for the same inputs (parity within models tests).
- Test data: reuse the `tests/test_runner.py` synthetic `vtest` fixture pattern (small frames, `fast` preset, few trees).

## Docs & skill updates (same change set)

- `ARCHITECTURE.md`: new §S (`nmr/opt.py`) — space schema, metric resolution (incl. `corr_sharpe_ac` and chronological sorting), baseline-anchor resolution rule, parameter-resolution rule, determinism contract + `n_jobs=1` invariant, the two caveats; §3 dependency graph node (opt → models/research); update the `models.py` `_resolved_params` note for the new public helper; update the `research.py` `_held_out_metric` description for the new metric branch.
- `AGENTS.md`: §3 prohibition carve-out — "Never add third-party dependencies… EXCEPT Optuna (user-granted exception 2026-08-08) for the HPO path (`nmr/opt.py` only)." Toolkit table row: `Change HPO search strategy | nmr/opt.py — bayesian_sweep`. Note the `n_jobs=1` invariant pointer.
- `README.md`: annotated tree entry for `nmr/opt.py`; requirements note updated.
- `requirements.txt`: pin `optuna==<latest>` (resolved at plan time; verify current version in the environment).
- `.kimi-code/skills/hpo-narrowing/SKILL.md` (S2): stage 2 narrows via `bayesian_sweep` (declarative dict space, `enqueue_base_config=True`, `n_jobs=1`); keep stage 1's coarse `HyperparameterSweep.run` for small spaces and stage 3's full-run confirmation flow unchanged; encode the metric set (now incl. `corr_sharpe_ac`) and the per-environment reproducibility caveat.
- Doc-SSOT count sync in AGENTS.md/README.md/CONTRIBUTING.md when tests are added (established precedent).

## Out of scope (this cycle)

- CatBoost backend and Streamlit+Plotly dashboard (sub-projects 2–3, later cycles per the approved sequence).
- No change to the run_id environment fingerprint; `HyperparameterSweep.run` contract unchanged; `SweepResult` schema unchanged.
- No new metric formulas, no purge-geometry changes, no changes to `canonical_scorecards_bytes`.
- BO is not wired into `ExperimentRunner` or the run_id payload — sweeps are research-side only.
- Horizon inference from target-name suffix (deferred; `horizon="20D"` default for `corr_sharpe_ac`).
