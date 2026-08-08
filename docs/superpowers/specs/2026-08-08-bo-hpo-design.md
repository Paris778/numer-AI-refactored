# Bayesian HPO via Optuna — Design

> Status: approved 2026-08-08 (user). Sub-project 1 of the external-library exception (BO → CatBoost → dashboard, in that sequence).

**Goal:** Add a Bayesian-optimization HPO path to the `nmr` research harness using Optuna, expressed as a declarative search-space dict, deterministic under seed, and wired into the S2 (`hpo-narrowing`) skill's stage-2 narrowing loop.

## Decisions (user-confirmed)

| Decision | Choice | Rationale |
|---|---|---|
| Library | **Optuna** (user-granted exception to the no-new-deps rule, 2026-08-08) | TPE sampler with `sampler_seed` + `deterministic=True`; first-class LightGBM/XGBoost integration; actively maintained |
| Space expression | **Declarative dict** mapped internally to `trial.suggest_*` | Serializable, auditable, config-driven; no business logic in agent/user code (AGENTS §2.1) |
| API surface | **New module `nmr/opt.py`** with `bayesian_sweep(...)` | Isolates the Optuna import to one module; `HyperparameterSweep.run` (deterministic random/Cartesian) stays backward-compatible |

## Architecture

- New tested module `nmr/opt.py` — the ONLY module that imports `optuna`.
- Public API: `bayesian_sweep(base_config: ExperimentConfig, space: dict, *, n_trials: int, seed: int, metric: str = "sharpe", n_startup_trials: int = 10) -> SweepResult`.
- The objective is harness-internal: it materializes each trial's params via the existing `_override_config` (merging into `model.params` only) and evaluates via the existing `research._held_out_metric` (held-out = final ~20% of eras with purge; metrics limited to `MetricSummary` fields `mean/std/sharpe/max_drawdown` — `corr_sharpe_ac` raises).
- Returns the existing `SweepResult` contract (`trials: pl.DataFrame`, `best_params: dict`, `best_value: float`), so downstream consumers (S2 skill, config materialization) are unchanged.
- `HyperparameterSweep.run()` in `nmr/research.py` is NOT modified — brute-force/random path preserved for small spaces.

## Space schema

Declarative dict, validated fail-loud at call time (`ValueError` with the offending key):

```
space = {
    "learning_rate": ("float", 0.005, 0.05, True),   # (kind, low, high, log)
    "num_leaves":    ("int", 16, 256),               # (kind, low, high)
    "boosting":      ("categorical", ["gbdt", "dart"]),
}
```

Mapping to Optuna:
- `("float", lo, hi, log)` → `trial.suggest_float(name, lo, hi, log=log)`; `log=False` allowed.
- `("int", lo, hi)` → `trial.suggest_int(name, lo, hi)`.
- `("categorical", [a, b, ...])` → `trial.suggest_categorical(name, [a, b, ...])` (non-empty list of hashable values).

Validation rules (each raises `ValueError`): space empty; unknown kind; `lo > hi` for int/float; empty categorical list; non-hashable categorical value; `n_trials < 1`; `n_startup_trials < 1`; unknown metric (must be a `MetricSummary` field).

## Determinism & reproducibility contract

- Study created with `optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed, deterministic=True))` — seeded trial generation.
- Evaluations are the harness's bit-deterministic `_held_out_metric` (custom backend, CPU). Same config + space + seed + pinned dependencies ⇒ identical trial sequence, identical best params, cross-process.
- Documented caveats (mirroring the existing GPU/CPU run caveat in AGENTS.md):
  1. Sweep reproducibility is per-environment — GPU-vs-CPU evaluation differences cascade through BO adaptivity.
  2. Optuna's TPE internals can change across versions — sweeps are reproducible under pinned versions (`requirements.txt` pins Optuna).
- `optuna` is deliberately NOT added to the run_id environment fingerprint (`ExperimentRunner._environment_fingerprint`): it never affects a single run's model results. Fingerprint stays `{numpy, polars, pandas, lightgbm, xgboost}`.

## Error handling

- A trial whose evaluation raises is caught and recorded in the trials log with a non-finite `metric_value` marker and the error message (fail loud in the sweep log), and the sweep continues — mirroring `run_campaign.py`'s batch semantics. Trial failures do not silently vanish; the final report lists them.
- All validation errors raise immediately (fail early), before any trial runs.

## Testing (`tests/test_opt.py`)

- Same-seed determinism: two `bayesian_sweep` calls with identical args produce identical `trials` DataFrame, `best_params`, `best_value` (cross-process contract).
- Seed sensitivity: different seeds (may) produce different trials — asserted only weakly where provable (e.g., different `n_startup_trials` random draws), never a flaky inequality.
- Space validation: each error class (empty, unknown kind, inverted range, empty categorical, non-hashable value, `n_trials < 1`, `n_startup_trials < 1`, unknown metric) raises `ValueError` with a matchable message.
- Distribution mapping: float log/linear, int, categorical all produce values within bounds / from the list.
- Metric validation: only `mean/std/sharpe/max_drawdown` accepted.
- `SweepResult` contract: trials columns (`trial_id`, `params_json`, `metric_value`, `metric`), sorted by value desc, `best_params` round-trips through `params_json`.
- Test data: reuse the `tests/test_runner.py` synthetic `vtest` fixture pattern (small frames, `fast` preset, few trees) so trials are cheap.

## Docs & skill updates (same change set)

- `ARCHITECTURE.md`: new §S (`nmr/opt.py`) — space schema, determinism contract, the two caveats; §3 dependency graph node (opt → research/_held_out_metric); note in the known-gaps/tech-debt section if relevant.
- `AGENTS.md`: §3 prohibition carve-out — "Never add third-party dependencies… EXCEPT Optuna (user-granted exception 2026-08-08) for the HPO path (`nmr/opt.py` only)." Toolkit table row: `Change HPO search strategy | nmr/opt.py — bayesian_sweep`.
- `README.md`: annotated tree entry for `nmr/opt.py`; requirements note updated.
- `requirements.txt`: pin `optuna==<latest>` (resolved at plan time; verify current version in the environment).
- `.kimi-code/skills/hpo-narrowing/SKILL.md` (S2): stage 2 narrows via `bayesian_sweep` around the top-k from stage 1; keep stage 1's coarse `HyperparameterSweep.run` for small spaces and stage 3's full-run confirmation flow unchanged; encode the metric restriction and the per-environment reproducibility caveat.
- Doc-SSOT count sync in AGENTS.md/README.md/CONTRIBUTING.md when tests are added (established precedent).

## Out of scope (this cycle)

- CatBoost backend and Streamlit+Plotly dashboard (sub-projects 2–3, later cycles per the approved sequence).
- No change to the run_id environment fingerprint; `HyperparameterSweep.run` contract unchanged.
- No new metric formulas, no purge-geometry changes, no changes to `canonical_scorecards_bytes`.
- BO is not wired into `ExperimentRunner` or the run_id payload — sweeps are research-side only.
