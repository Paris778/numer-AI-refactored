# Endeavour Report: Benchmark Fleet, Checkpointing, and the `mt-std-v1` Campaign

> Written 2026-08-23, after the successful completion of the `mt-std-v1` campaign. Covers the full arc of the work: the original ask, design rationale, implementation methodology, operational incidents, the campaign itself, and everything that remains open. Companion documents: the fleet spec (`docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md`), the fleet plan (`docs/superpowers/plans/2026-08-19-benchmark-fleet.md`), the checkpoint spec/plan (`docs/superpowers/specs/2026-08-20-oof-checkpoint-resume-design.md`, `docs/superpowers/plans/2026-08-20-oof-checkpoint-resume.md`), and the SDD ledger (`.superpowers/sdd/2026-08-19-benchmark-fleet/progress.md`).

---

## 1. Executive Summary

The endeavour had three waves:

1. **The Benchmark Fleet** — a new untiered layer of 19 deterministic benchmark models (silly heuristics → tutorial recreations → community scripts → the Finance Arena v0.2–v1.5.1 series), scored through the framework's existing evaluation pipeline and placed against the 5-tier ladder by measured performance. Fully implemented, reviewed, and CI-green.
2. **Operational hardening** — runner composability (`--only-fleet`, CSV-sourced tier rungs, cell selection) and fold-granularity OOF checkpointing with code/device identity, so a multi-day campaign crash no longer erases every fit. Fully implemented, reviewed, CI-green.
3. **The `mt-std-v1` campaign** — a 4-target × 4-fold standard-preset LightGBM run on v5.3 medium (780 features), started before the checkpointing existed and therefore completely uninsured. It survived 66.5 hours, completed all 16 CV folds + 4 full-history deploy fits + the validation scorecard, and recorded:

| Metric | Value |
|---|---|
| OOF mean per-era CORR | 0.0344 |
| OOF Sharpe | 1.72 |
| Validation CORR (86-era meta overlap) | 0.0142 |
| Validation CORR Sharpe (AC) | 0.588 |
| Validation FNC | 0.0141 |
| Validation MMC | 0.0026 |
| Ladder placement | between tier 3 and tier 4 (below the production gate) |

---

## 2. Origins and Goals

The user asked to extend the benchmark system with models "ranging from super simple and silly heuristics to proper advanced and competitive models," specifically:

1. The three Numerai tutorial notebooks (`1_hello_numerai`, `2_feature_neutralization`, `3_target_ensemble`) — **small and deep** versions.
2. All community notebooks (`example_model.py`, `example_model_advanced.py`, `example_model_sunshine.py`) — **shallow and deep** versions, re-fit on v5.3 (the sources used v4/v4.1).
3. An audit list: constant-0.5, rolling previous-era target mean, simple ridge, and the **Finance Arena series** (v0.2 XGBoost, v0.3 multi-target LightGBM, v0.4 weighted XGBoost, v0.5 ridge-specialist stacking, v0.6.0 MLP, v1.5.0 ridge-stack tail-10, v1.5.1 grid-searched ridge ensemble).

Initial exploration established what already existed: `null_constant_05` and three ridge cells in the hierarchy, and the three tutorials already present as tier-3 canonical cells at fast params. The Finance Arena sources were located in the read-only legacy repo (`C:/dev/numer-AI/models/version_0/`, `version_1/`), and the SNNR auxiliary-target lists were pinned from `C:/dev/numer-AI/exploratory_notebooks/outputs/snnr_weights_vs_correlation_v5.2.csv`.

---

## 3. Design Decisions and Rationale

All decisions were made through the brainstorming → spec → plan workflow, with the user answering structured questions and overruling the initial tier proposal:

| # | Decision | Rationale |
|---|---|---|
| 1 | **Untiered fleet** (user directive) | New models live in `configs/benchmarks/fleet/` with no `tier` field; their measured scorecards place them against the ladder indirectly (report-only `placement` column). The 5-tier "line in the sand" stays untouched — no gate re-semanticization, no monotonicity risk from unknown model strengths. |
| 2 | Extend the benchmark system, not standalone runs | Everything scored through the identical `evaluate_model` pipeline; determinism covered by `canonical_scorecards_bytes`. |
| 3 | Roster lock | Only the listed models + missing variants. No duplication of existing cells (`null_constant_05`, `linear_ridge_*`, `canon_hello_numerai`); legacy extras (FA v0.1, `simple_lgbm_shallow`) excluded. |
| 4 | Fidelity policy | Architecture/params/targets/neutralization proportions faithful to sources; ALL processing via framework machinery — purged trimmed-train fits (exact 8-era purge), rank-Gaussian ensembling, `NeutralizationEngine` only. Where notebooks conflict with invariants (4-era embargoes, hand-rolled neutralize, multi-seed retraining, CV loops), the framework wins and the deviation is documented per cell. |
| 5 | "small/deep", "shallow/deep" | small/shallow = notebook defaults (2k trees, lr 0.01, depth 5); deep = the notebooks' commented recommended params (30k trees for tutorials; 20k/lr 0.001/depth 6/64 leaves for community). |
| 6 | SNNR aux targets pinned | 17 targets from the legacy CSV (all present in v5.3), weights pinned into the v1.5.1 config. No runtime SNNR computation. |
| 7 | v1.5.1 selection bias made explicit | The search cell selects candidates on validation (as the notebook did) — scorecard rows carry `selection_bias: true`; never compared naively against unbiased cells. |
| 8 | v4→v5.3 target mapping | `nomi_v4_60 → target_ender_60`, `jerome_v4_60 → target_ender_60`, `nomi_v4_20 → target_ender_20`, `jerome_v4_20 → target_jeremy_20` (name-adjacency assumption, flagged, one-line config edits if wrong). |
| 9 | Work on `main` | Explicit user consent (data/venv/artifacts live only in this checkout). |

---

## 4. What Was Built

### 4.1 The Untiered Benchmark Fleet

- **New module `nmr/benchmark_fleet.py`**: `FleetCellConfig` schema (same cell fields as tiered cells minus `tier`, plus `source`, `target_weights`, `neutralizer_selection`, `neutralizer_count`), loaders, five generators, and the `BenchmarkFleet` runner.
- **Generators**: `target_lag_mean` (trailing-train target mean, leak-safe by construction); fleet `lightgbm` (canonical fits + riskiest-50 neutralizer selection via `feature_stability_screen`); fleet `xgboost` (weighted multi-target rank blend, tail-holdout early stopping via a new `construct_tree_model(extra_params=...)`); `mlp` (sklearn MLPRegressor + `_standardize_feature_block`); `ridge_stack` (two-layer ridge stacking, horizon-aware 8/16-era internal purge; search mode = the full v1.5.1 pipeline: quality filter → 13-point alpha grid with Sharpe pruning → non-negative-ridge/LGBM meta candidates → decorr × neutralization sweeps → validation-based selection).
- **Runner**: `BenchmarkFleet.run(tier_rungs, gate)` → scorecards + `fleet_placement` (vs per-tier max-corr rungs) + informational tier-4 gate verdicts + derived `selection_bias`. Fleet scorecards join `canonical_scorecards_bytes`.
- **Configs**: 19 cells across `fleet_silly.yaml` (1), `fleet_tutorials.yaml` (5), `fleet_community.yaml` (6), `fleet_finance_arena.yaml` (7).
- **Gate helpers**: `tier4_gate_verdict` / `tier_max_corrs` extracted from the hard gates (shared rows, hard-gate behavior unchanged — verified by the pre-existing tests).
- **Final-review fix (C1)**: xgb/mlp cells initially ignored their `neutralization` config — the "config knob that lies" bug class. Wired `NeutralizationEngine` through both generators with oracle tests.

### 4.2 Runner Composability

`--only-fleet` (skip the hierarchy; placement rungs from the last hierarchy scorecard CSV via `load_tier_rungs_from_csv`), `--rungs-csv`, `--fleet-ids` (comma-separated cell selection with fail-loud unknown-id rejection). Verified end-to-end: `silly_target_lag_mean` scored in 4.7 s with zero hierarchy.

### 4.3 OOF Fold-Checkpointing & Resume

Built after a parallel review session established that a 21-hour campaign run persisted **nothing** until completion (the registry row is written at the very end). Two review cycles:

- **v1** was rejected with 5 blockers: the bit-for-bit guarantee was untested in the one case that can break it (mixed load+fit within a target); `CVResult.models` would be silently truncated on resume; checkpoints keyed on run_id only would silently survive code changes; two defective tests; the plan forbade the AGENTS.md update the repo mandates.
- **v2/v2.1** (review-approved) fixed all five: a new `ModelOrchestrator.train_oof_with_checkpoints` isolates resume from the public `train_cross_validation` (models can never truncate); a `manifest.json` records `code_sha256` (SHA-256 of `nmr/models.py` + `nmr/splitter.py`) and the fit device, written atomically **at the first fitted fold** (device is only known post-fit — an earlier write would record `"None"` and vacate the guard); mismatches, corrupt parquets, and torn trees all raise with "delete the directory to refit" guidance; the mixed-resume test (delete one fold mid-target → refit → bit-for-bit equality + load/fit log mix) is the determinism proof; atomic writes reuse `atomic_write_bytes`; AGENTS.md got the §8 hazard with a compensating trim (32,719 → 32,620 B).

---

## 5. Methodology

- **Process**: Superpowers brainstorming → written spec (committed) → user review → written implementation plan (committed, code-verbatim) → subagent-driven development: a fresh coder subagent per task, TDD (red → green → commit), an independent task reviewer per task (spec compliance + quality verdicts), a fix loop, and a final whole-branch review. Progress ledger kept at `.superpowers/sdd/2026-08-19-benchmark-fleet/progress.md`.
- **Verification gates**: `ruff check` (E/F/I/UP @120) + pytest as the functional gate; CI green at `1da1478`; docs-hygiene tests kept green (test-count claims synced to measured counts: 979 → 988 → 1007); full suite 1007 collected / green.
- **Review culture**: every task diff went through an independent reviewer. The final review found exactly one Critical (C1) — fixed and re-reviewed. All other findings were parked in the ledger with rulings; nothing was silently discarded.

---

## 6. Operational Saga (What Nearly Went Wrong)

1. **Smoke attempt 1** "died with the session" — in fact the process survived for hours; Git Bash `ps` doesn't show Windows processes. Lesson: `wmic process ... get CommandLine` is the only trustworthy liveness check.
2. **The duplicate-runner incident**: the misdiagnosis caused a second runner to be launched alongside the first — two processes on the same log/outputs, likely contributing to the next failure. One was killed once detected.
3. **Smoke attempt 2 died with a genuine OOM**: `ArrowMemoryError: malloc of 12.8 GB failed` during `canon_sunshine_ensemble` (4× medium LightGBM fits) while the box was under external memory pressure.
4. **Smoke attempt 3** was progressing (tier 1) when the user paused the work to run the campaign — killed cleanly at their request. Real-data fleet verification therefore remains **unfinished** (see §10).
5. **The file-visibility mystery**: while building the live campaign monitor, the campaign's log/status files proved to be periodically unreadable (absent for stretches of seconds-to-minutes) — traced to a non-atomic rewriter external to the repo (a parallel session's watcher). The monitor was hardened (retries, last-good-state fallback, watched-path messaging) and later superseded by that session's own tooling (`a19df13` removed `campaign_status.py` as obsolete).

---

## 7. The Campaign

**Config** (`configs/mt-std-v1.yaml`): v5.3, medium (780 features), 4 targets (`cyrusd_20`, `ender_20`, `jasper_20`, `teager2b_20`), standard LightGBM preset (20k trees), walk-forward 4-fold, 8-era purge, ridge ensemble weights, neutralization proportion 1.0, seed 20260819. `validation_scorecard: true` (which is why 4 full-history deploy fits ran), deploy artifact disabled.

**Timeline** (started 2026-08-20 13:40, finished 2026-08-23 08:09 ≈ 66.5 h):

| Stage | Result |
|---|---|
| CV folds, `cyrusd_20` (4) | ✅ avg 52.1 min/fold (27.1–93.4) |
| CV folds, `ender_20` (4) | ✅ avg 207.3 min/fold — one fold stretched to 685.6 min by a machine-sleep event (47.8 min avg excluding it) |
| CV folds, `jasper_20` (4) | ✅ avg 55.8 min/fold |
| CV folds, `teager2b_20` (4) | ✅ avg 34.4 min/fold |
| OOF blend → neutralize → evaluate | ✅ |
| Full-history deploy fits (4 targets, 20k trees each, train-only ~2.7M rows) | ✅ (through ~19:48 on 08-22) |
| Validation scorecard (4.1M rows, 649 eras) | ✅ quiet-compute stage, ~12 h, one core pegged |
| Registry write + clean exit | ✅ run_id `2c5e5f39…`, campaign log JSON written |

**The risk window**: the entire campaign ran on pre-checkpoint code. Everything — 16 CV models, 4 deploy models, OOF frames — lived only in process memory until the final registry write. The checkpointing work merged mid-campaign applies to the **next** launch, and covers the CV folds (not the deploy/validation stages).

**Results** (registry `artifacts/registry/2c5e5f39…/run.json`):

- OOF (last-fold scoring eras): mean CORR 0.03443, std 0.02003, Sharpe 1.719, max drawdown 0.0352.
- Validation (86-era meta overlap): CORR 0.01416, CORR Sharpe (AC) 0.588, FNC 0.01415, MMC 0.00260, rank scalar 0.01401.
- Ladder placement: **tier3..tier4** — above every community baseline (tier-3 max corr 0.00952), below the production gate (corr ≥ 0.0286, sharpe ≥ 0.78, fnc ≥ 0.020).
- `artifact_path: None` — no `predict.pkl` produced (deploy disabled); the full-history fits served the validation scorecard.
- OOF→validation gap (0.0344 → 0.0142) is the expected train-history optimism; validation is the tournament-relevant number.

---

## 8. Commit Chronology (This Session's Work)

| Date | Commits | What |
|---|---|---|
| 08-19 | `7bcc1a8` → `60a2a16` | Fleet spec + plan + all 12 implementation tasks (gate helpers → schema → 5 generators → runner → 19 configs → CLI → docs) |
| 08-20 | `b51fe60`, `b584791` | Plan synced with reviewed deviations; final-review C1 fix (xgb/mlp neutralization) |
| 08-20 | `59a893e` | Cold-start handoff document (for the parallel session) |
| 08-21 | `0956953`, `0329996` | Handoff corrections; runner composability (`--only-fleet` + CSV rungs + `--fleet-ids`) |
| 08-21 | `12d9e0a`, `e01fbb9`, `db40717` | Checkpoint spec+plan v1 → v2 (5 review blockers fixed) → v2.1 (manifest-at-first-fit) |
| 08-21 | `ca6883c`, `6f17bd4`, `c466f8e`, `1da1478` | Checkpoint core + runner wiring + docs; hygiene count sync (CI green at `1da1478`) |
| 08-21 | `b10141d`, `1bc7c85` | Live campaign monitor + hardening (later removed by the parallel session, `a19df13`) |

Parallel-session commits interleaved throughout (mutation-gate work, dataless CI, lint fixes, durability guidelines) — treated as outside this endeavour's scope, noted where they interact.

---

## 9. Lessons Learned

**Operational:**
- Git Bash `ps` cannot see Windows python processes — `wmic`/`tasklist` only. A "lost" background task may be alive; check before relaunching. One runner at a time, ever.
- Long jobs: `nohup ./.venv/Scripts/python -u … > log 2>&1 &` with a NEW log file per attempt; `>` from a second process destroys the first's log view.
- The box's memory ceiling is real; the medium fits' `to_pandas` allocations (~12.8 GB) die under external pressure. Check headroom before launching; never run heavy jobs concurrently with the campaign.
- Machine-sleep events can stretch a single fold to 11.4 h; wall-clock ETAs are only valid while the box stays awake.

**Process:**
- The external-review loop (spec/plan → independent reviewer → fix → re-review) caught a Critical bug class (config knob that lies) that per-task reviews structurally cannot see — cross-task invariants need a whole-branch pass.
- "Config + data fingerprint = identity" is wrong for cached artifacts; code identity must be in the cache key, or the cache will eventually hand back a wrong answer with a straight face.
- Doc-hygiene tests are real gates (test-count claims, nav coverage, module graph) — keep them green in the same commit, per the repo's own law.

---

## 10. Open Items (Honest State)

1. **Fleet real-data verification (Task 13) — NOT DONE.** The 19 fleet cells have never been scored on real v5.3 data; no placements/gate verdicts measured; anchors not re-pinned. Resumable from the handoff doc: smoke → full run (tens of CPU-hours for the deep cells) → end-of-session gate.
2. **Thread caps** — nothing in `nmr/` caps polars/OpenMP (`n_jobs=1` covers tree fits only). Joint design decision deferred until after the campaign; the machine is now free to have it.
3. **Checkpoint coverage** — DONE (2026-08-23). All three uninsured stages now checkpoint under the code/device identity manifest: CV folds, deploy fits (`e5a038e`), and validation era-batch predicts (this commit, `feat(runner): validation predict-batch checkpoints + docs`), with the shared helpers extracted in `b8d635b`. The final `evaluate_model` scorecard call stays uncheckpointed (single call, no clean granularity).
4. **Parked findings** — full list with rulings in the SDD ledger (error-message cosmetics, `fleet_frame` empty-placements corner, search-mode test hardening, the all-loaded-resume device-swap note, etc.). All ruled defer; triage before any future change touches those lines.
5. **`e01fbb9` is a red ancestor** — docs-only CI failure (stale test-count claim) fixed by `1da1478`; harmless unless bisecting through that range.
6. **Backlog from the parallel session**: `_transforms` re-measurement, mutmut harness patch, upstream mutmut issue, Monday's mutation-gate run review.
