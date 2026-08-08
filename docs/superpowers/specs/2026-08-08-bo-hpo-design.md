# Bayesian HPO via Optuna — Design

> Status: approved 2026-08-08 (user), revised 2026-08-08 per independent architecture review (9 findings: 7 accepted, 2 rejected with evidence). Sub-project 1 of the external-library exception (BO → CatBoost → dashboard, in that sequence).

**Goal:** Add a Bayesian-optimization HPO path to the `nmr` research harness using Optuna, expressed as a declarative search-space dict, deterministic under seed, and wired into the S2 (`hpo-narrowing`) skill's stage-2 narrowing loop.

## Decisions (user-confirmed + review-adjudicated)

| Decision | Choice | Rationale |
|---|---|---|
| Library | **Optuna** (user-granted exception to the no-new-deps rule, 2026-08-08) | TPE sampler with `sampler_seed` + `deterministic=True`; first-class LightGBM/XGBoost integration; actively maintained |
| Space expression | **Declarative dict form (canonical), no positional tuples** | Serializable, auditable, config-driven; avoids tuple-length ambiguity for agents generating raw configs |
| API surface | **New module `nmr/opt.py`** with `bayesian_sweep(...)` | Isolates the Optuna import to one module; `HyperparameterSweep.run` (deterministic random/Cartesian) stays backward-compatible |
| Review item 1.3 | **`corr_sharpe_ac` supported** | Primary promotion/ranking metric (`RunRegistry.promote_if_better` defaults to it); restricting HPO to proxy metrics misaligns with champion gates |
| Review item 3.2 | **`n_jobs=1` non-negotiable** | TPE `deterministic=True` breaks under parallel trials (async completion ordering); models already force `n_jobs: 1` internally (`models.py:309,322`) — parallel Optuna on top would oversubscribe |

## Architecture

- New tested module `nmr/opt.py` — the ONLY module that imports `optuna`.
- Public API: `bayesian_sweep(base_config: ExperimentConfig, space: dict, *, n_trials: int, seed: int, metric: str = "sharpe", n_startup_trials: int = 10, enqueue_base_config: bool = True, n_jobs: int = 1) -> SweepResult`.
- **Baseline anchor:** when `enqueue_base_config=True`, `study.enqueue_trial(base_config.model.params)` runs BEFORE `study.optimize(...)`, so Trial 0 evaluates the existing configured baseline (e.g. `configs/first_model.yaml`) before random startup / TPE exploration.
- The objective is harness-internal: each trial's params are materialized via `research._override_config` (merging into `model.params` only — see Parameter Resolution Rule) and evaluated via `research._held_out_metric`.
- Returns the existing `SweepResult` contract (`trials: pl.DataFrame`, `best_params: dict`, `best_value: float`).
- `HyperparameterSweep.run()` in `nmr/research.py` is NOT modified — brute-force/random path preserved for small spaces.

### Metric resolution (review 1.3)

`_held_out_metric` in `nmr/research.py` is extended (additive, tested) to resolve:
- `mean` / `std` / `sharpe` / `max_drawdown` → `getattr(MetricSummary, metric_name)` (existing behavior).
- `corr_sharpe_ac` → `ac_adjusted_sharpe(sorted_era_values, horizon="20D")` from `nmr.inference` — the same construction as `scorecard._cell_from_sharpe_series` and the runner's hardcoded validation horizon.
- Any other metric name → `ValueError` (unchanged).

### Parameter Resolution Rule (review 3.3 — documented, verified against code)

`ModelConfig` has exactly three fields (`backend`, `preset`, `params`; `config.py`). Presets (`_CANONICAL_PRESETS`) own the defaults; `ModelOrchestrator._resolved_params` computes `preset.copy(); params.update(...)` (`models.py:301-303`) — **`model.params` is the single override channel and wins over preset defaults**. There are no top-level hyperparameter attributes to route to; `_override_config` merging trial params into `model.params` is the complete, correct resolution. No other mapping exists or is needed.

### Optuna hygiene (review 3.1)

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
- `categorical`: `trial.suggest_categorical(name, choices)` — non-empty list of hashable values.

Validation errors (each raises `ValueError` with a matchable message): unknown kind; missing/unknown keys; `low > high`; empty categorical choices; non-hashable choice; `step < 1` or non-int; `log` not boolean; empty space; `n_trials < 1`; `n_startup_trials < 1`; `n_jobs != 1`; metric not in `{mean, std, sharpe, max_drawdown, corr_sharpe_ac}`.

## Determinism & reproducibility contract

- Study created with `TPESampler(seed=seed, deterministic=True)` — seeded trial generation. Trial 0 (baseline anchor) is deterministic by construction.
- Evaluations are the harness's bit-deterministic `_held_out_metric` (custom backend, CPU). Same config + space + seed + pinned dependencies ⇒ identical trial sequence, identical best params, cross-process.
- **`n_jobs=1` is a hard invariant** (enforced by assertion): parallel trials break `deterministic=True` TPE and oversubscribe CPU (models already run `n_jobs: 1` internally).
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
  `TrialPruned` marks the trial PRUNED in the study; TPE excludes pruned trials from its density estimation, and `study.best_trial` only ever selects completed trials with finite values.
- The sweep's own trials log records failed trials with `metric_value = None` (null) and the four shared `SweepResult` columns — identical schema to `HyperparameterSweep.run()` (see Output Contract); failures are loud (logger.error per failed trial) and never silent.
- All validation errors raise immediately (fail early), before any trial runs.

## Output contract (review 4.1 — verified, no change needed)

`SweepResult.trials` is a Polars DataFrame with columns `trial_id`, `params_json`, `metric_value`, `metric`, sorted by `metric_value` desc — **byte-identical schema to `HyperparameterSweep.run()`** (which stores params serialized in `params_json`, `research.py:80-96`; NOT expanded columns — the reviewer's premise was incorrect). Consumers (S2 skill, config materialization) read params via `json.loads(params_json)`. Failed trials appear as rows with `metric_value = None`. No schema expansion: changing the shared contract would break existing consumers, the opposite of backward compatibility.

## Resource cleanup (review 3.4)

The objective wrapper calls `gc.collect()` at the end of each trial evaluation to keep memory bounded across long sweeps (Polars frames, NumPy arrays, LightGBM/XGBoost C-objects; refcounting frees most, `gc` handles cycles).

## Testing (`tests/test_opt.py`)

- Same-seed determinism: two `bayesian_sweep` calls with identical args produce identical `trials`, `best_params`, `best_value`.
- Baseline anchor: with `enqueue_base_config=True`, trial 0's params equal `base_config.model.params`.
- `corr_sharpe_ac` metric: runs and produces the `ac_adjusted_sharpe` of the held-out per-era corr series; cross-checked against the scorecard path on synthetic data where feasible.
- Metric validation: `mean/std/sharpe/max_drawdown/corr_sharpe_ac` accepted; anything else raises.
- Space validation: each error class raises `ValueError` with a matchable message (unknown kind, inverted bounds, empty categorical, non-hashable choice, `step < 1`, `log+step` on int, empty space).
- `n_jobs=2` raises `ValueError`.
- Failed-trial handling: a space value that makes training raise → trial marked failed (null `metric_value` in the log), sweep continues, `best` comes from finite trials.
- Distribution mapping: float log/linear, int log/step, categorical — values within bounds/choices.
- `SweepResult` contract: four columns; `best_params` round-trips through `params_json`.
- Test data: reuse the `tests/test_runner.py` synthetic `vtest` fixture pattern (small frames, `fast` preset, few trees).

## Docs & skill updates (same change set)

- `ARCHITECTURE.md`: new §S (`nmr/opt.py`) — space schema, metric resolution (incl. `corr_sharpe_ac`), parameter-resolution rule, determinism contract + `n_jobs=1` invariant, the two caveats; §3 dependency graph node (opt → research); update the `research.py` `_held_out_metric` description for the new metric branch.
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
- No expanded-params columns in `SweepResult` (contract matches `HyperparameterSweep.run`).
