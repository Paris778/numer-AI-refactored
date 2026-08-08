---
name: run-meta-analysis
description: Use when comparing recorded runs from the nmr registry — paired era-level comparisons of two OOFs, fleet-wide ranking with multiple-trial DSR, or grouping runs into robust config families
type: prompt
disableModelInvocation: false
---

# Run Meta-Analysis (S3)

**Core principle:** read-only, era-paired, device-aware comparison over the registry. If it would mutate state, it doesn't belong here.

## When to Use
- Comparing two runs' OOFs on aligned eras.
- Ranking a fleet with a defensible multiple-trial DSR policy.
- Emitting robust config families for a follow-up hpo-narrowing campaign.

## Protocol

1. **Ingest** — `RunRegistry(root).list()` entries; each `run.json` carries `run_id`, `metrics` (`mean`/`std`/`sharpe`/`max_drawdown`), `manifest` (incl. `oof_device` and `config`), and a `scorecard` block with `<metric>_ci_low/_ci_high/_n_eras`, `perturb_*`, `regime_*`, `horizon_*` (timing/quality fields are stripped at record time).
2. **Pair on aligned era windows and matching `oof_device`** — GPU vs CPU OOF values are not comparable. `paired_era_comparison(oof_a, oof_b, *, metric_fn, era_col="era", horizon="20D", n_boot=1000, seed, alpha=0.05, min_overlap_eras=MIN_OVERLAP_ERAS, block_len=None, device_a=None, device_b=None)`; `metric_fn` maps an OOF frame → `{era: metric}` (e.g. a closure over `EvaluationEngine().per_era_corr`). Positive `mean_diff` ⇒ A is better. Fewer than `MIN_OVERLAP_ERAS` (20) overlapping eras raises `NonVacuityError`; a `device_mismatch=True` result is reported, never silently corrected.
3. **DSR with an explicit `n_trials` policy** — `deflated_sharpe(sharpe, *, n_trials, n_obs, skew, kurt, trials_sr_var, sr0_benchmark=0.0)`. `trials_sr_var` is **required when `n_trials > 1`**. The scorecard's stored `deflated_sharpe` was computed with `n_trials=1`; `fleet_summary` records `policy_n_trials`/`policy_dsr_confidence` as context — campaign-aware DSR needs era-level recompute via the paired tooling.
4. **Group + flag outliers** — `fleet_summary(runs, *, metric="corr_sharpe_ac", n_trials, dsr_confidence=0.95)` produces per-run rows: preset, feature_set/feature_subset, neutralization_proportion, `oof_device`, `max_feature_exposure`, `dsr_pass`, `has_bmc`/`has_horizon`/`has_perturb`/`has_regime`. Flag runs with good headline CORR but poor `perturb_*`/`regime_*`/`horizon_*` stability. Emit "robust config families" + recommended base configs for hpo-narrowing (S2).

## Hard Rules
- [ ] Never compare across `oof_device` — check `manifest.oof_device` before any numeric comparison.
- [ ] Never pair below `MIN_OVERLAP_ERAS` (20) — `NonVacuityError` is a floor, not a suggestion.
- [ ] Every DSR claim carries an explicit `n_trials` (with `trials_sr_var` when > 1).
- [ ] Read-only: never call `record`/`promote`/`promote_if_better` from an analysis path.

## Common Mistakes
| Mistake | Fix |
|---|---|
| Comparing GPU vs CPU OOF | Match `oof_device` or flag `device_mismatch` |
| Pooled (non-era) metric comparison | Per-era first, then aggregate |
| Treating stored DSR as campaign-aware | It was computed with `n_trials=1` |
| Silently dropping scorecard-less (legacy) runs | `fleet_summary` flags them; keep the flag |
| "Just this once" registry write | Analysis never writes |
