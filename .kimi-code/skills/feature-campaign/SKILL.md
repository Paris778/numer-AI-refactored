---
name: feature-campaign
description: Use when designing a feature-subset experiment campaign for the nmr framework — comparing data.feature_subset candidates, screening features for cross-era/cross-regime stability, or Pareto-selecting feature sets against the champion
type: prompt
disableModelInvocation: false
---

# Feature Campaigns (S1)

**Core principle:** one campaign = one named batch of configs sharing a hypothesis, differing only in `data.feature_subset`; every subset is a pure function of `features.json` + the stability screen; the human picks the winner.

## When to Use
- Comparing named feature sets (small/medium/all, personality families, v2/v3 equivalents) on real data.
- Screening candidate features for stability before committing a config.
- Fleet runs via `run_campaign.py`, then Pareto selection against the champion.

## Protocol

1. **Discover** — `resolve_feature_sets(Path("data/v5.3/features.json"))` returns `dict[str, list[str]]` in deterministic order. `small`=42, `medium`=780, `all`=3555, plus personality families (intelligence, charisma, strength, dexterity, constitution, wisdom, agility, serenity, sunshine, rain, midnight, faith, ...) and v2/v3 equivalents. Enumerate from `features.json` — never hardcode the list.
2. **Screen** — `feature_stability_screen(frame, *, feature_cols, target_col, era_col="era", min_mean_corr=DEFAULT_MIN_MEAN_CORR, max_abs_decay=DEFAULT_MAX_ABS_DECAY)` computes per-feature `mean_corr`, `corr_std`, `decay_slope`, `cross_regime_variance`, `n_eras`, `stable`; keep winners via `select_stable_features(screen, *, min_mean_corr, max_abs_decay) -> list[str]` (sorted). Defaults `0.01` / `0.001`. Formulas: ARCHITECTURE.md §P. This is nmr business logic — call it, never re-implement in scripts/notebooks. (The `DEFAULT_*` constants are module-level in `nmr/features.py`, not package-root exports.)
3. **Generate candidates** — subset definitions must be pure functions of `features.json` + screen output: same inputs ⇒ same subset. Screen-derived sets (`screen_stable`, `screen_nonlinear`, `screen_linear_or_nonlinear`, `screen_drift_filtered`) are materialized by the `derived_sets` analysis stage from the **train-only** screen (`screens_train` stage → `feature_ic_screen_train.parquet`, eras 0001..0574 — subset derivation must never see validation labels) into `derived_feature_sets.json` and consumed via `data.supplemental_feature_sets` (merged by `IngestionAgent`; key collision raises). The full-span `screens` stage is descriptive only. **The `stable` column here is NOT the same predicate as step 2's `feature_stability_screen.stable`** (that one is only the classic |mean_corr|/|decay_slope|/n_eras point predicate): `feature_ic_screen`'s `stable` is the full gate — classic predicate ∧ CI strictly excludes zero ∧ BH-FDR q ≤ 0.05 (per target). Screening locally with `feature_stability_screen` skips the CI/FDR gate silently — use `feature_ic_screen` when you need the full gate. An **empty `screen_stable` is a valid scientific result** — training on it fails loudly at ingestion (`IngestionAgent.features` raises). A different subset is a different experiment by design: the run_id fingerprint covers config (incl. `data.feature_subset` + a SHA256 of the supplemental file's **contents**, with the path itself stripped — editing the file changes run identity) + data_version + `nmr/*.py` + environment (ARCHITECTURE.md §P).
4. **Materialize + evaluate** — one YAML per candidate (e.g. `configs/campaigns/<name>/...`), each with `data.feature_subset: <set>`; subset wins over `feature_set` (`resolved_feature_set` = subset if set). Dry-run first: `python run_campaign.py --config a.yaml --config b.yaml --name <campaign> --dry-run` (prints run_ids, writes nothing), then run for real. The log lands atomically at `artifacts/campaigns/<campaign_id>.json` with per-config run_ids and statuses (`recorded`/`skipped`/`error`).
5. **Select** — `fleet_summary(RunRegistry("artifacts/registry").list(), *, n_trials, dsr_confidence=0.95)`; Pareto-filter Sharpe vs feature count, penalize high `max_feature_exposure`; present ranked configs for **human review**. Never auto-promote.

## Hard Rules
- [ ] `purge_eras >= 8` (20D targets) / `>= 16` (60D) — protocol-enforced; the code only checks the configured gap, it does not block weakening.
- [ ] Subsets are pure functions of `features.json` + screen output.
- [ ] No business logic in configs/scripts/notebooks — call the nmr functions.
- [ ] Never call `RunRegistry.promote`/`promote_if_better` — promotion is a human decision.

## Common Mistakes
| Mistake | Fix |
|---|---|
| Hardcoding set names | `resolve_feature_sets(features.json)` |
| Pooled (non-era) stability screening | Use `feature_stability_screen` (per-era internally) |
| Forgetting subset wins over `feature_set` | Set `data.feature_subset`; keep `feature_set` as fallback |
| Skipping `--dry-run` | Run_ids are cheap; training is not |
| Auto-promoting the fleet winner | Verdict → human, always |
