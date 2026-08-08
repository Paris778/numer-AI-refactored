---
name: hpo-narrowing
description: Use when tuning model hyperparameters for the nmr framework — planning a multi-stage sweep, narrowing a search space around top candidates, or deciding whether a tuned candidate is worth proposing for promotion
type: prompt
disableModelInvocation: false
---

# Multi-Stage HPO (S2)

**Core principle:** coarse → narrow → confirm, all inside the deterministic core; routine sweeps run at the `fast` preset only; promotion stays with the human.

## When to Use
- Tuning `model.params` when `feature_set`/targets are already fixed.
- Shrinking a search space around a promising region.
- Preparing a promotion proposal for human review.

## Protocol

**Stage 1 — coarse.** `HyperparameterSweep(base_config, *, metric="sharpe").run(space, *, n_trials, seed)` at the `fast` preset. `space` maps a param to a list of options (or a scalar); candidates are drawn deterministically from `seed`.
- **CRITICAL — `metric` must be a `MetricSummary` field: `mean`, `std`, `sharpe`, `max_drawdown` only.** `_held_out_metric` does `getattr(summary, metric_name)`; anything else (e.g. `corr_sharpe_ac`) raises `ValueError: Unknown metric`.
- Held-out geometry: the **final ~20% of eras with purge** (`frac=0.2` of unique eras, `purge_eras` eras before the window excluded) — not "20 eras".
- The sweep overrides only `model.params` — it cannot sweep `feature_set`/targets.

**Stage 2 — narrow.** Read `SweepResult.trials` (polars DataFrame: `trial_id`, `params_json`, `metric_value`, `metric`, sorted by `metric_value` desc). Shrink the space around the top-k param combinations and re-run with the same seed discipline.

**Stage 3 — confirm.** Materialize the best configs into `configs/hpo/*.yaml`; run the full `ExperimentRunner(cfg).run(deploy=False)` + validation scorecard. Compare via `promotion_verdict(candidate, champion, *, metric="corr_sharpe_ac")`: propose **"promote"** only when the candidate's CI clears the champion's; "caution" on CI overlap or missing CI. The verdict is advisory — hand the proposal to the human. **Never call `RunRegistry.promote`/`promote_if_better` yourself.**

## Budget Rule
- Routine sweeps: `fast` preset only. `deep` is reserved for confirmed winners (AGENTS.md §10: full presets run hours).

## Hard Rules
- [ ] `metric` ∈ {`mean`, `std`, `sharpe`, `max_drawdown`} — nothing else.
- [ ] Held-out = last ~20% of eras with purge, never a fixed 20.
- [ ] Sweep space touches `model.params` only.
- [ ] `fast` for routine sweeps; `deep` only for confirmed winners.
- [ ] Promotion is a human decision — verdicts are advisory, never executed.

## Common Mistakes
| Mistake | Fix |
|---|---|
| `metric="corr_sharpe_ac"` → "Unknown metric" | Use a `MetricSummary` field |
| "20-era held-out" assumption | Final ~20% of eras, purged |
| Sweeping `feature_set`/targets | Only `model.params` is overridable |
| Promoting on a verdict | Propose to human; registry stays untouched |
| `deep` sweep "just to be sure" | Runs for hours; confirm winners only |
