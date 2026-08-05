# Design Spec — Codebase Sanity Check (`numer-AI-refactored`)

**Date:** 2026-08-05 · **Branch:** `sanity-check` · **Status:** Approved by user (3 design reviews, all amendments folded in)
**Source:** `docs/reviews/2026-08-05-codebase-review.md` (Principal Engineering audit, 28 findings F-001..F-028)

---

## 1. Objective

Implement **all 28 findings** from the audit with the audit's recommended *proper* fixes, keeping the SSOT suite (`AGENTS.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `README.md`) in sync in the same commits. Deferred (explicitly out of scope): FEAT-003 submission pipeline, FEAT-008 multi-seed HPO, FEAT-007 mypy gate, and the Phase-1 ModelAdapter architecture. In scope by extension of the proper fixes: FEAT-001 (full-pipeline artifact), FEAT-002 (validation scorecard stage), FEAT-005 (guarded promotion), FEAT-006 (vectorized evaluation), FEAT-009 (neutralization cache budget), and the F-005 exposure vectorization.

## 2. Locked scope decisions (user-approved)

| # | Decision | Consequence |
|---|---|---|
| 1 | All 28 findings, proper fixes | No new features beyond the proper fixes listed above |
| 2 | Unified validation leaderboard | FEAT-002 in scope; trained runs ranked on the same scorecard definitions as benchmarks; legacy runs in a secondary table |
| 3 | CI = pytest + script smoke tests, no mypy | No AGENTS.md §8 tooling-policy change |
| 4 | Deployed model trains on **all eras** | New `ModelOrchestrator.train_full_history`; no anchor-fold holdout |
| A | Sequential clusters, test-first | P0 → P1 → P2 → P3/docs; full suite green after every commit |
| — | One commit per design sub-section | A1, A2, A3+A4, B1..B7, C1, C2(final) — each with same-commit SSOT doc updates |

## 3. Architecture shape (post-fix)

```
config → data → splitter → models (CV OOF | full-history) → ensemble (weights) → risk (neutralize)
                                                                                │
   ExperimentRunner.run():                                                     │
     • OOF CV per target → rank-normalize → learn weights (folds 0..K-2)        │
     • blend + neutralize(final-fold OOF only) → MetricSummary (provenance)     │
     • full-history models → predict validation → blend+neutralize (one weight  │
       set) → drop first purge_eras val eras → evaluate_model → MetricScorecard │  ← ranking metric
     • deploy: serialize closure embedding same transforms + models + weights ──┘
   Dashboard: ranks trained (scorecard) + benchmark rows on identical definitions
```

**Invariant:** one weight set, learned once (folds `0..K-2`), recorded once in the manifest, consumed by OOF scoring, the validation stage, and the deployed closure. `n_folds < 2` → uniform weights with a logged warning.

**`RunResult.oof` contract:** `oof` and `oof.parquet` keep the **full stacked OOF** (all folds). Only `metrics` narrows to final-fold rows; manifest records `weight_learning_eras` (folds 0..K-2) and `scoring_eras` (final fold). (Prevents breaking `tests/test_runner.py` and preserves provenance.)

## 4. Section A — P0 cluster

### A1. Benchmark runner integrity (F-001, F-016)
- New public `BenchmarkSuite.iter_baseline_predictions(*, include_classical: bool, min_train_eras: int) -> Iterator[tuple[str, str, pl.DataFrame]]` yielding `(model_id, group, raw_preds)` — wraps `null_prediction_frame`, `_trivial_prediction_frame`, `_walk_forward_model_predictions` behind one public surface.
- Existing `run_classical_baselines()` consumes the same generator (one code path, not two).
- `benchmark_runner._candidate_strategies` consumes only the public generator; seeds assigned by position (`NULL_BASELINES` iterated instead of the hardcoded name tuple); classical entries get `seed+4`/`seed+5` per current convention.

### A2. Deployable artifact = evaluated strategy (F-002, F-019, F-026, F-013 fold-in, all-eras anchor)
- `ModelOrchestrator.train_full_history(df, *, feature_cols, target_col, era_col) -> object`: fits one seeded model on **all eras** (no fold, no purge) via the existing `_fit_model`; applies the F-007 null-target filter (see B3).
- `_serialize_predict_artifact(model, train_df, feature_cols, target_cols, weights, proportion, artifact_path)` (no `splitter` param — F-026):
  - trains one full-history model per `data.targets` component;
  - builds a closure `predict(live_features, live_benchmark_models=None)` that: selects ordered features → per-target model predictions → **per-era rank-gaussianize** (per `Ensembler` geometry; if `live_features` carries an `era` column, group by it, else treat all rows as one era — documented contract) → weighted blend → re-gaussianize → neutralize with `proportion` via `neutralize_array`.
  - **Deployment runtime constraint:** the closure's code path references only `numpy`/`scipy` (verified: `nmr/_transforms.py` imports numpy + scipy only). `cloudpickle.register_pickle_by_value(nmr._transforms)` is called before `dumps` so the *actual shared* implementations (`rank_gaussianize`, `rank_gaussianize_unit_variance`, `neutralize_array`) are embedded by value — **no duplicated transform math, no `nmr` import at load time**. No polars/pandas in the embedded code path (pandas receives the input frame at the boundary only).
- `neutralize_array(pred, features, proportion) -> np.ndarray`: new pure-numpy module-level helper in `nmr/_transforms.py`; single implementation used by both `NeutralizationEngine._neutralize_era` and the closure. Zero-variance → returns input unchanged (B4 contract).
- `serialize_predict` writes **atomically**: payload via temp file + fsync + `os.replace`, then the manifest the same way (F-013). AGENTS.md §9 updated in this commit.
- Manifest gains: `weights`, `targets`, `proportion`, `feature_set`, `anchor_geometry: "all_eras"`, resolved device.
- Fidelity regression test (F-019): on a synthetic fixture, `loaded_predict(val_features)` must rank ≈ 1.0 (Spearman) vs. the **validation-stage predictions** (same full-history models, in-repo transforms) on a shared era. (Not vs. CV OOF — the models differ by construction under all-eras training.)

### A3. Validation scorecard stage (FEAT-002, F-004 root)
- `ExperimentRunner.run()` gains a validation stage (config-gated): full-history models → predict on `validation.parquet` → rank-normalize → blend with the **single** OOF-learned weight set → neutralize → **drop the first `split.purge_eras` validation eras** (20D-target overlap with the last train eras — leakage) → `evaluate_model(...)` with meta model, benchmark models, features, targets, `backend=config.evaluation.backend` → `MetricScorecard` stored in `run.json` + manifest.
- `run.json` gains a `scorecard` block (flattened `MetricScorecard.to_frame()` row minus timing fields — timing fields poison canonical hashes, and scorecards here are not hashed, but exclude them from the registry payload anyway for cleanliness).
- Manifest records `validation_purge_dropped_first_eras: purge_eras`.

### A4. Unified dashboard (F-004, F-023)
- `generate_dashboard` trained rows read `run.json["scorecard"]` (`corr` → mean, `corr_sharpe_ac` → sharpe, `std_corr` → std, scorecard `max_drawdown`) — same definitions as benchmark rows → one ranked leaderboard is valid.
- Runs without a scorecard → `source="trained_legacy"`, secondary table below the ranked leaderboard, excluded from ranking.
- `html.escape()` on all interpolated cells (F-023).

## 5. Section B — module-level changes

### B1. Registry & guarded promotion (F-003, F-021, F-022)
- `RunRegistry.promote_if_better(run_id, metric="corr_sharpe_ac") -> tuple[Path, bool]`:
  - regex-validates `run_id` as `[0-9a-f]{64}` (same as `promote` — F-022 in both);
  - validates `metric` against a `_SCORECARD_METRIC_FIELDS` tuple (scorecard fields: `corr_sharpe_ac`, `rank_scalar`, `corr`, `mmc`, `fnc`, `std_corr`, `max_drawdown`, `deflated_sharpe`) with a clear `ValueError`;
  - compares against the current champion's recorded **scorecard** metric; promotes only if strictly better, or if no champion exists (corrupted champion pointer → missing run dir → treated as no champion);
  - legacy runs without a scorecard are **refused** auto-promotion (require manual `promote()`); logs the refusal.
- `best()`: deterministic `(metric value, run_id)` sort; `ValueError` on unknown metric. `list()`: `(mtime, run_id)` stable sort.
- `record()`: OOF parquet via temp + `os.replace` (registry half of F-013); stores `scorecard` block.
- `train_first_model.py`: `promote_if_better`; prints verdict + champion comparison.

### B2. Honest evaluation (F-006, F-015)
- Config (A2 commit): new `risk` section with `neutralization_proportion: float = 1.0` (validated in [0,1]) — active immediately in `run()` (replaces hardcoded `1.0`) and the closure. Extended with `cache_max_bytes` in the B4 commit (active when the engine budget lands; never inert).
- Config (B2 commit): new `ensemble` section with `method: str = "ridge"` (`ridge` | `non_negative`, validated) — active immediately in `learn_weights`.
- Weight-learning honesty: learn on OOF rows of folds `0..K-2`; blend/neutralize/score only the final fold's OOF rows (see §3 invariant + §4 A3).
- **MMC scoping (technical blocker):** the meta model covers validation eras only — train-era OOF MMC is unimplementable. `evaluation.metrics` in the runner OOF path supports only `corr`, `fnc`, `sharpe`. Requesting `mmc` without the validation stage enabled raises `ValueError` at run start. `mmc`/`bmc`/`cwmm` exist only in the validation scorecard. Documented in ARCHITECTURE.md §2A.
- `evaluate_model` gains optional `backend: str = "custom"`; validation stage passes `config.evaluation.backend`. `official` stays opt-in (pandas round-trips per era, slower) — never a default.

### B3. Model-layer guards (F-007, F-009, F-014)
- F-007: in `_fit_predict_fold` and `train_full_history`, filter training rows to finite non-null targets; log dropped-row count; `ValueError` if nothing remains.
- F-009: `_fit_model` logs each failed device attempt (`type(exc).__name__: exc`); `except` narrowed to `(ValueError, TypeError, lgb.basic.LightGBMError, xgb.core.XGBoostError)`. Resolved device recorded in the manifest (not in the run_id hash — machine-dependent).
- F-014: `PurgedEraSplitter` gains a `purge_eras` property; `_fit_predict_fold` asserts `min(val_nums) - max(train_nums) > purge_eras` (real, fireable assertion). `tests/test_models.py:163` tautological mirror replaced with a violating-fold test (adjacent eras, zero purge → `ValueError`).

### B4. Risk cache & I/O (F-008, F-011, F-012)
- F-012: cache load catches `(OSError, ValueError, EOFError)`; both cache files (`.npy` + `.json`) written via temp + `os.replace`, metadata last.
- F-008: `NeutralizationEngine(..., max_cache_bytes: int | None = None)` — default `DEFAULT_CACHE_MAX_BYTES = 2 GiB` (named constant); size log at init; **total-size LRU eviction (mtime-oldest first) on store**; no orphan inference/deletion (unsound under concurrency — consciously rejected). Config field `risk.cache_max_bytes` (optional).
- F-011: zero-variance predictions → `neutralize_array` returns input unchanged; engine logs a warning with the era label; era count preserved downstream. Contract documented in ARCHITECTURE.md §2F + pinned by test.

### B5. Vectorization (F-005, F-010, F-027 + risk.py loop)
- `_per_era_metric`, `per_era_bmc`, `per_era_cwmm`, `_resolve_overlap_eras`: single `df.partition_by(era_col, maintain_order=True)` pass, era→part lookup, identical dict outputs (203 tests are the oracle).
- `NeutralizationEngine.neutralize` era loop (risk.py:76): same `partition_by` treatment.
- `feature_exposure_report` (F-005, pulled into the A3 commit): partition once; per-era **Pearson** correlation of prediction vs all features as one matrix op. **Definition change** (previously Numerai-CORR): documented in ARCHITECTURE.md; pre/post `max_feature_exposure` numbers incomparable → changelog note; determinism fixtures (`test_benchmark_slice1/3`) updated **deliberately** — commit message states the old→new hash pair (a silently regenerated hash fixture is indistinguishable from a regression); `benchmark_scores*.csv` regenerated under the new definition (at minimum a `--fast-mode` run in this branch).
- F-027: `_sorted_labels` → public `sorted_era_labels`, `_clean_frame` → public `clean_frame` (module-level in evaluation.py); update `robustness.py` + tests; add to `__all__`.

### B6. Benchmark module polish (F-017, F-024, F-028)
- Module logger: INFO era progress in `_walk_forward_model_predictions`; WARNING on id-column inference fallback and any other inference fallback.
- Remove the `GradientBoostingRegressor` sklearn fallback (F-017) — lightgbm is a hard dependency; fail loudly.
- F-024: log the inferred id column at WARNING.

### B7. CI + script contract tests (F-018)
- `.github/workflows/ci.yml`: Python **3.12** (matches local venv 3.12.4), `pip install -r requirements.txt`, `pytest -q`. Real-data tests self-skip without v5.2 assets.
- `tests/test_scripts.py` (new): imports `benchmark_runner`/`generate_dashboard`/`train_first_model`; dry-runs `_candidate_strategies` against a stub suite (catches F-001-class contract breaks); dashboard escaping + unified ranking; promotion-guard smoke.

## 6. Section C — P3 polish, docs re-sync, verification

### C1. Quick wins (final P3 commit)
- F-020: remove `os.environ["PYTHONHASHSEED"] = str(seed)` from `set_global_seeds`; fix docstring (subprocess-only).
- F-025: remove unused `python-dotenv` from `requirements.txt`; reword AGENTS.md §9 (credentials notebook-only, never hardcoded).

### C2. Docs / SSOT re-sync (same commits as invalidating code)
- **AGENTS.md**: §1 "pytest is the sole automated gate" → "pytest, enforced by CI"; §2.4 assertion claim (now real purge-width assertion); §7 verification gates (CI workflow); §8 GPU hazard reworded (fallback now logged + device in manifest) + new closure/`register_pickle_by_value` hazard + "no lint/type-check tooling" stays true; §9 atomicity claim scoped to its true exhaustive set (registry JSON, artifacts + manifest, OOF parquet, risk-cache pair) + dotenv wording.
- **ARCHITECTURE.md**: §2A config schema (`risk`/`ensemble` sections, `evaluation.metrics` wiring with mmc-validation-only split, `backend` plumbed); §2F neutralization NaN→unchanged contract; §2M/§2N deployment rewrite (all-eras full-pipeline artifact, single weight set, manifest schema, validation purge-drop); exposure = per-era Pearson; `promote_if_better`; cache budget.
- **CONTRIBUTING.md**: CI workflow reference.
- **README.md**: config surface (`risk`/`ensemble`), dashboard description.
- **`configs/example.yaml`**: `risk`/`ensemble` sections.
- **Test-count sweep (final cluster):** the hardcoded "203 tests" in AGENTS.md, README.md, CONTRIBUTING.md updated once from an actual `pytest --collect-only -q` count.
- **Export sweep checklist:** `nmr/__init__.py` imports + `__all__` for every new public symbol (`sorted_era_labels`, `clean_frame`; methods `iter_baseline_predictions`, `train_full_history`, `promote_if_better` need no export; `neutralize_array` stays internal in `_transforms`).

## 7. Findings coverage ledger

| Findings | Where |
|---|---|
| F-001, F-016 | A1 |
| F-002, F-019, F-026, F-013 (deployment) | A2 |
| F-004, F-023 | A4 (+ FEAT-002 = A3) |
| F-003, F-021, F-022, F-013 (registry) | B1 |
| F-006, F-015 | B2 |
| F-007, F-009, F-014 | B3 |
| F-008, F-011, F-012 | B4 |
| F-005, F-010, F-027 | B5 |
| F-017, F-024, F-028 | B6 |
| F-018 | B7 |
| F-020, F-025 | C1 |
| Docs drift (F-013/14/15/25 secondary) | C2 |

No orphans. All 28 findings covered.

## 8. Config schema changes (concrete)

```yaml
# new optional sections (unknown keys still rejected by load_config)
risk:
  neutralization_proportion: 1.0      # float in [0, 1]
  cache_max_bytes: null               # null -> DEFAULT_CACHE_MAX_BYTES (2 GiB)
ensemble:
  method: ridge                       # ridge | non_negative
# evaluation.metrics semantics: runner OOF path honors corr/fnc/sharpe;
# 'mmc' requires the validation stage (else ValueError at run start).
```

## 9. Test plan (new / updated)

- F-019 fidelity: `loaded_predict` vs validation-stage predictions (Spearman ≈ 1.0).
- B1: `promote_if_better` (better/worse/no-champion/corrupted-champion/legacy-refusal/unknown-metric/run-id-regex).
- B3: null-target filter (rows dropped, fit succeeds; all-null → ValueError); purge-width assertion violation.
- B4: cache corruption → recompute; eviction under tiny budget; zero-variance → unchanged + era preserved.
- B5: vectorized outputs identical to oracle contract (full suite); exposure values under Pearson definition; `sorted_era_labels`/`clean_frame` public helpers.
- A4/B7: dashboard escaping + ranking; script contract smoke.
- Determinism fixtures `test_benchmark_slice1/3`: deliberate old→new hash diff, stated in commit message.
- F-020: `set_global_seeds` no longer touches env (assert env unchanged).

## 10. Verification gates (pre-sign-off)

1. Full `pytest -q` green after every commit.
2. `benchmark_runner.py --fast-mode` (real-data smoke; regenerates `artifacts/benchmark_scores_smoke.csv` under the new exposure definition).
3. Regenerate primary `benchmark_scores.csv` (fast-mode at minimum) + `generate_dashboard`.
4. Real-data runner round-trip on `fast` preset incl. validation stage + deploy + `load_predict` sanity (time-boxed; `train_first_model.py`).
5. Non-fast benchmark backfill (`linear`/`tree` rows) — only if wall-clock budget allows (walk-forward per-era training is long); otherwise documented as a follow-up.
6. Update the "203 tests" count from `pytest --collect-only -q` in the final cluster.

## 11. Risks & open items

- Exposure-definition change alters recorded scorecards/hashes — managed via deliberate fixture diffs + CSV regeneration (B5).
- Validation-stage runtime cost until B5 lands — mitigated by pulling F-005 vectorization into the A3 commit.
- Numerai hosted runtime library availability (numpy/scipy assumed; `_transforms` verified numpy/scipy-only) — the fidelity test + `register_pickle_by_value` keep drift impossible, but a runtime without scipy would fail at predict time; documented hazard.
- `official` backend speed — opt-in only.
