---
name: hpo-narrowing
description: Use when tuning nmr model hyperparameters — planning a multi-stage sweep, narrowing a search space around top candidates, or proposing a tuned candidate for promotion
type: prompt
disableModelInvocation: false
---

# Multi-Stage HPO (S2)

**Core principle:** coarse → narrow → confirm, inside the deterministic core; routine sweeps at the `fast` preset; promotion stays with the human.

## When to Use
- Tuning `model.params` when `feature_set`/targets are fixed.
- Shrinking a search space around a promising region.

## Protocol

**Stage 1 — coarse.** For small spaces: `HyperparameterSweep(base_config, *, metric="sharpe").run(space, *, n_trials, seed)` at the `fast` preset; `space` maps a param to a list of options (or a scalar); candidates drawn deterministically from `seed`.
- Held-out = **final ~20% of eras with purge** (`frac=0.2`, `purge_eras` buffer excluded) — not "20 eras".
- Overrides `model.params` only — cannot sweep `feature_set`/targets.

**Stage 2 — narrow.** Shrink the space around the top-k from stage 1, then `bayesian_sweep(base_config, space, *, n_trials, seed, metric, enqueue_base_config=True, n_jobs=1)`.
- Declarative dict space: each param is `{"kind": "float"|"int"|"categorical", ...}` — float/int take `low`/`high` (optional `log` requires `low > 0`; `step` for ints); categorical takes `choices` (str/int/float/bool). Invalid specs raise `ValueError` upfront.
- `metric` ∈ {`mean`, `std`, `sharpe`, `max_drawdown`, `corr_sharpe_ac`} — the full `_held_out_metric` set.
- With `enqueue_base_config=True` (default), **Trial 0 is the resolved baseline** (preset defaults + `model.params`, space-intersected).
- Seeded TPE, but deterministic **per environment** — results can differ across machines (like GPU vs CPU OOF); rerun on the environment you report.
- Pick top candidates from `SweepResult.trials` (columns `trial_id`, `params_json`, `metric_value`, `metric`, sorted by `metric_value` desc).

**Stage 3 — confirm.** Materialize best configs into `configs/hpo/*.yaml`; run full `ExperimentRunner(cfg).run(deploy=False)` + validation scorecard. Compare via `promotion_verdict(candidate, champion, *, metric="corr_sharpe_ac")`: propose **"promote"** only when the candidate's CI clears the champion's; "caution" on overlap or missing CI. Advisory only — hand the proposal to the human; **never call `RunRegistry.promote` yourself**.

## Budget Rule
- Routine sweeps: `fast` preset only; `deep` reserved for confirmed winners (AGENTS.md §10).

## Hard Rules
- [ ] `metric` ∈ {`mean`, `std`, `sharpe`, `max_drawdown`, `corr_sharpe_ac`} — nothing else.
- [ ] Held-out = last ~20% of eras with purge, never a fixed 20.
- [ ] Sweep space touches `model.params` only.
- [ ] `n_jobs=1` always — parallel trials break TPE determinism.
- [ ] `fast` for routine sweeps; `deep` only for confirmed winners.
- [ ] Promotion is a human decision — verdicts are advisory, never executed.

## Common Mistakes
| Mistake | Fix |
|---|---|
| Dict space passed to `HyperparameterSweep.run` | Stage 1 takes option lists; stage 2 takes the dict space |
| "20-era held-out" assumption | Final ~20% of eras, purged |
| Sweeping `feature_set`/targets | Only `model.params` is overridable |
| `n_jobs > 1` "to go faster" | Breaks TPE determinism; keep 1 |
| Promoting on a verdict | Propose to human; registry untouched |
| `deep` sweep "just to be sure" | Runs for hours; confirm winners only |
