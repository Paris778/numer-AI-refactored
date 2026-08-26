# Architecture: numer-AI-refactored (`nmr`)

> **Status:** This document describes the **current implementation** as it exists in `nmr/`. It is not aspirational — every formula, schema, and stage order described here matches the running code. When the code changes, update this document in the same commit (see [`AGENTS.md`](AGENTS.md) Self-Update Directive).

---

## 1. System Topology & Data Flow

A deterministic, filesystem-backed research pipeline for the Numerai Classic tournament. No services, no databases — pure Python from YAML config to deployable artifact.

```
configs/*.yaml
     |
     v  load_config() — frozen dataclasses, unknown-key rejection
ExperimentConfig
     |
     v
+----------------------------------------------------------------+
| ExperimentRunner.run(deploy=False)          nmr/runner.py      |
|                                                                |
|  1. set_global_seeds(run.seed)      random/np                  |
|  2. IngestionAgent.load("train")    lazy Polars, col pushdown  |
|  3. per target in data.targets:                                |
|       ModelOrchestrator.train_cross_validation()               |
|         └── PurgedEraSplitter folds (walk_forward, 8-era purge)|
|       → OOF pred_{target} columns joined on [id, era]          |
|  4. Ensembler.learn_weights(ensemble.method) on folds 0..K-2    |
|     Ensembler.blend()               rank-domain, re-gaussianize|
|  5. NeutralizationEngine.neutralize(proportion from config)    |
|         └── per-era pinv cache: artifacts/cache/neutralization |
|  6. per_era_corr() on scoring eras (final fold) → summarize()  |
|  7. [deploy] full-history pipeline → serialize_predict()       |
|         └── experiments/{slug}/runs/{run_id}/predict.pkl (+man.) |
+----------------------------------------------------------------+
     |
     v  RunResult(run_id, oof, metrics, artifact, manifest)
RunRegistry.record() ──> root/{run_id}/{run.json, oof.parquet, validation_preds.parquet?}   (legacy compat layout; removed Task 11)
RunRegistry.promote() ─> experiments/champion.json   (atomic pointer {run_id, experiment_slug, promoted_at})
RunRegistry.promote_if_better() ─> experiments/champion.json  (cross-family, guarded: scorecard metric + direction)
     |
     v
generate_dashboard.py (thin wrapper) ─> dashboard_ui.report ─> artifacts/dashboard.html  (executive report — offline single-file vanilla HTML/CSS/SVG, < 112 KiB)

Parallel harness:
benchmark_runner.py ──> BenchmarkHierarchy (nmr/benchmark.py)
   tiers 0-3 generator cells + tier-4 reference columns (v53_lgbm_ender60 gate + v53_lgbm_ender20)
   ──> evaluate_model() scorecards ──> artifacts/reports/benchmark_hierarchy_scorecard.csv
                                     + artifacts/reports/benchmark_gate_report.csv
```

**Key design decisions:**
- **Deterministic end to end:** `run_id` is a SHA256 of config + data version + code fingerprint + environment fingerprint (path fields stripped). Same inputs ⇒ same run_id, same OOF, same scorecard hash.
- **Lazy, columnar data:** Polars `scan_parquet` with explicit `.select()` pushdown; nothing is materialized until needed.
- **Filesystem as database:** atomic JSON writes (temp + fsync + `os.replace`) make the registry crash-safe without any server.
- **Dual-backend metrics:** fast custom NumPy implementations for research, `numerai_tools` official path for audit — bound together by parity tests.

---

## 2. Component Specifications

### A. Configuration — `nmr/config.py`

Frozen dataclasses; `__post_init__` validates enums, non-negativity, and non-emptiness. `load_config(path)` parses YAML, rejects unknown keys/sections. `set_global_seeds(seed)` seeds `random` and `np.random` (it does not set `PYTHONHASHSEED` — hash randomization is fixed at interpreter startup).

| Section | Fields (defaults) | Valid values |
|---|---|---|
| `data: DataConfig` | `version="v5.3"`, `feature_set="small"`, `feature_subset=None`, `supplemental_feature_sets=None`, `targets=("target",)`, `horizon="20D"`, `data_dir=REPO_ROOT/"data"` | feature_set ∈ `("small", "medium", "all")`; feature_subset: any `features.json` set name or `None` (validated at ingestion, §P); supplemental_feature_sets: JSON path merged into the set registry (collision ⇒ `ValueError`, §P); horizon ∈ `("20D", "60D")` with the purge law (below) |
| `split: SplitConfig` | `scheme="walk_forward"`, `purge_eras=8`, `embargo_eras=0`, `n_folds=4` | scheme ∈ `("walk_forward", "anchor")`; **`embargo_eras` must be 0 — non-zero raises (A2, 2026-08-18)** |
| `model: ModelConfig` | `backend="lightgbm"`, `preset="fast"`, `params={}` | backend ∈ `("lightgbm", "xgboost", "catboost")`, preset ∈ `("fast", "standard", "deep")` |
| `evaluation: EvalConfig` | `backend="custom"`, `main_target="target"`, `metrics=("corr","mmc","fnc","sharpe")`, `validation_scorecard=True` | backend ∈ `("custom", "official")`; metrics ⊆ `("corr","mmc","fnc","sharpe")` (unknown names ⇒ `ValueError` at load — a typo must not silently compute nothing) |
| `risk: RiskConfig` | `neutralization_proportion=1.0`, `cache_max_bytes=None` | proportion ∈ [0, 1]; `cache_max_bytes` ≥ 0 or None |
| `ensemble: EnsembleConfig` | `method="ridge"` | method ∈ `("ridge", "non_negative")` |
| `run: RunConfig` | `name="default"`, `seed=42`, `artifacts_dir=REPO_ROOT/"artifacts"` | — |

`metrics` semantics: CORR/FNC/sharpe-family are computed on the train OOF in `run()`; MMC/BMC/CWMM are validation-only (they need the meta model / benchmark columns that only the validation stage provides). `validation_scorecard=False` skips the validation stage entirely (no meta/benchmark assets required). `metrics` gates `run()` at start: requesting `mmc` while `validation_scorecard=False` raises `ValueError` before any training (MMC covers validation eras only, via the meta model).

`REPO_ROOT = Path(__file__).resolve().parent.parent`. Relative paths resolve against it. See [configs/example.yaml](configs/example.yaml) for the annotated schema and [configs/first_model.yaml](configs/first_model.yaml) for the current competitive config (4×20D targets, ridge ensemble, full neutralization, seed 20260713).

### B. Data Layer — `nmr/data.py`

`IngestionAgent(data: DataConfig)` — construction-time inert (no I/O).

- `scan(split, *, subset=None, targets=None, columns=None) -> pl.LazyFrame` — lazy scan with column pushdown; `load()` collects; `train()/validation()/live()` shortcuts.
- Split files: `{"train": "train.parquet", "validation": "validation.parquet", "live": "live.parquet"}`; meta columns `("era", "id")`.
- `feature_sets` / `features(subset)` / `available_targets()` read `data/v5.3/features.json` (defensive copies).
- Deterministic column order: `era · id · features(subset) · targets`. Requested targets are validated against `features.json` then intersected with the physical schema. Schema reads are metadata-only and cached per split. Missing split file ⇒ `FileNotFoundError` on first access.

### C. Validation Splitting — `nmr/splitter.py`

`PurgedEraSplitter(split: SplitConfig).split(eras) -> list[Fold]`; `Fold(index, train_eras, val_eras)` frozen. `PurgedEraSplitter.purge_eras -> int` exposes `split.purge_eras` so consumers can re-assert the purge width at train time.

Geometry: eras deduped and sorted numerically (non-numeric ⇒ `ValueError`). With `n = era_count`, `k = n_folds`:
- `val_size = n // (k + 1)`; `prefix_size = n - k * val_size`; requires `prefix_size - purge_eras ≥ 1`.
- **walk_forward** fold *i*: val = `eras[prefix + i·val_size : prefix + (i+1)·val_size]`, train = `eras[: val_start - purge_eras]`.
- **anchor**: single fold with `k=1` geometry — one train prefix, one validation window.

Invariants validated on every fold: `max(train) < min(val)`, exactly `purge_eras` eras excluded between them, no era reuse, disjoint validation windows. `embargo_eras` is **rejected at load when non-zero** (A2, 2026-08-18 — it was structurally inert). Purge/horizon leakage law (A1, 2026-08-18): `data.horizon` (`20D`/`60D`) requires `purge_eras ≥ 8`/`≥ 16` — enforced data-aware via `enforce_purge_horizon_law(era_count, config)` at run time when the dataset has ≥ 2× the floor's eras (real-data regime); target names encoding a horizon (`target_<name>_20/60`) must agree with the declared horizon at config load. Purge/embargo convention: [docs/DOCS_README.md](docs/DOCS_README.md) §3; official benchmark walk-forward table (156-era blocks): [docs/01-canon/models.md](docs/01-canon/models.md).

### D. Transforms — `nmr/_transforms.py`

Shared by evaluation, ensemble, and submission:

| Function | Definition |
|---|---|
| `tie_kept_rank(v)` | `(rankdata(v, method="average") − 0.5) / n` → [0, 1) |
| `gaussianize(v)` | `norm.ppf(v)` |
| `rank_gaussianize(v)` | `gaussianize(tie_kept_rank(v))` |
| `standardize_unit_variance(v)` | `v / std(v, ddof=0)`; zeros if std is 0/non-finite |
| `rank_gaussianize_unit_variance(v)` | composition of the above |
| `neutralize_array(pred, features, proportion=1.0, *, pseudo_inverse=None)` | `pred − proportion · design @ pinv(design, rcond=1e-6) @ pred`, design = `[features | 1]`; zero-variance pred returned unchanged |
| `power_1_5(v)` | `sign(v) · |v|^1.5` |

All transforms follow the canonical definitions in [docs/01-canon/scoring/00-definitions.md](docs/01-canon/scoring/00-definitions.md).

### E. Evaluation Engine — `nmr/evaluation.py`

`EvaluationEngine(backend="custom")` — per-era metric dicts (`{era: score}`); `custom` = NumPy implementations below, `official` = `numerai_tools.scoring` delegation. `MIN_OVERLAP_ERAS = 20`; insufficient overlap raises `NonVacuityError(ValueError)`.

Per-era engines partition the input frame exactly once (`partition_by(era_col, maintain_order=True)`) and iterate `sorted_era_labels(...)` order, keying partitions by era label — output dicts remain numerically era-ordered regardless of row appearance order. `sorted_era_labels(labels)` and `clean_frame(df, columns)` are public module-level helpers (no private cross-module access).

Capital-readiness helpers (v2.5): `downside_era_indices(meta_corr, *, threshold=0.0) -> list[str]` returns chronological era labels where the meta model's CORR is **strictly** below the threshold (strict `<` per the director-locked MMC-down contract; fail-loud via `sorted_era_labels` on non-numeric labels); `per_era_turnover(df, *, pred_col, era_col="era", id_col="id") -> dict[str, float]` returns `{era: 1 − ρ_k}` Spearman rank turnover for each consecutive chronological era transition on the shared stock-ID intersection — transitions with fewer than `_TURNOVER_MIN_SHARED_IDS` (10) shared IDs are skipped, non-finite ρ maps to 0.0 (turnover 1.0, bounded in [0, 2]), and missing `era_col`/`id_col`/`pred_col` raise `ValueError`.

| Metric | Custom algorithm (per era) |
|---|---|
| **CORR** `per_era_corr` | `Pearson( power_1_5(rank_gaussianize(pred)), power_1_5(target − mean(target)) )` |
| **MMC** `per_era_mmc` | rank-gaussianize pred & meta; orthogonalize pred against meta (`p − m·(p@m / m@m)`); bucket targets in [0,1] rescaled to [0,4]; return `(target − mean) @ neutral_pred / n` |
| **FNC** `per_era_fnc` | least-squares residual of `rank_gaussianize(pred)` against `[features | intercept]`, std-normalized, then CORR vs target |
| **BMC** `per_era_bmc` | MMC-form vs a benchmark column (`min_overlap_eras=20`) |
| **CWMM** `per_era_cwmm` | MMC-form pred-vs-meta (no oracle counterpart) |

`summarize(per_era) -> MetricSummary(mean, std, sharpe, max_drawdown)` — std ddof=0, `sharpe = mean/std` (0 if std=0), drawdown on cumulative sum. Degenerate eras (<2 rows, zero variance, non-finite) short-circuit to score 0.0 after `clean_frame()` null/finite filtering.

Metric definitions follow the canonical tournament spec: CORR → [docs/01-canon/scoring/01-correlation.md](docs/01-canon/scoring/01-correlation.md); MMC/BMC → [docs/01-canon/scoring/02-mmc-bmc.md](docs/01-canon/scoring/02-mmc-bmc.md); FNC → [docs/01-canon/scoring/03-fnc.md](docs/01-canon/scoring/03-fnc.md). The repo's judging rules are the evaluation spec of record: [docs/06-evaluation/evaluation-suite-bible.md](docs/06-evaluation/evaluation-suite-bible.md).

### F. Neutralization — `nmr/risk.py`

`NeutralizationEngine(cache_dir=REPO_ROOT/"artifacts"/"cache"/"neutralization")`.

`neutralize(df, *, pred_col, feature_cols, era_col="era", proportion=1.0)` — per era: design `[features | 1]` (intercept-aware), `coeffs = pinv(design, rcond=1e-6) @ pred`, output `pred − proportion · (design @ coeffs)`. `proportion ∈ [0, 1]` (0 = identity, 1 = full). All values must be finite (else `ValueError`). The engine delegates the per-era solve to `_transforms.neutralize_array` (single source of truth, shared with the deployment closure); eras with zero-variance predictions are returned unchanged with a logged warning, and eras with `n_rows ≤ n_features + 1` warn that the fit is exact.

Per-era pseudo-inverse cache: key = SHA256 of `{era, sorted feature_cols, row_count, row_ids_sha256, intercept: true}`; files `era_{label}_{key}.npy` + `.json` (metadata revalidated before loading). The cache is bounded by `risk.cache_max_bytes` (default `DEFAULT_CACHE_MAX_BYTES = 2 GiB`) with mtime-oldest-first LRU eviction on store; every cache hit `os.utime`s both files so mtime reflects last use, not just write time, and a warning is logged if the cache stays over budget after a sweep. Cache writes are atomic (temp `.npy` + `os.replace`, metadata via `atomic_write_text`); a corrupt or truncated entry (`OSError`/`ValueError`/`EOFError` on load) is discarded and recomputed, so corruption self-heals.

### G. Modeling — `nmr/models.py`

`ModelOrchestrator(config: ModelConfig, *, seed=42)`:
- `train_cross_validation(df, *, feature_cols, target_col, splitter, era_col) -> CVResult(oof, models)` — per fold: fit on train eras, predict val eras, stack OOF.
- `train_anchor_fold(...) -> (model, val_predictions)` — single anchor fold (research use; no longer used for deployment).
- `train_full_history(df, *, feature_cols, target_col, era_col="era", include_validation=False) -> model` — one CPU-only model fit on every era, with null/non-finite targets dropped (logged count; `ValueError` if nothing remains). Used by the deployment pipeline so the artifact reproduces identically on any hosted runtime. `include_validation` (promotion writer): when the fit spawns, the child re-reads train+validation concatenated so the full version sees the validation eras.
- **Spawned full-history fits:** fits whose float32 feature matrix exceeds `_FULL_HISTORY_SUBPROCESS_MIN_BYTES` (8 GiB; env override `NMR_FULL_HISTORY_SPAWN_MIN_BYTES` lets the rehearsal force the spawn path at small scale) run in a fresh spawn process with bounded commit; the parent receives `(status, (model_bytes, peak_ws, peak_commit))` via `_receive_subprocess_result`, which polls child liveness (5 s poll interval) while awaiting the result queue and drains 5 s after child exit — a child that dies before reporting raises `RuntimeError` promptly instead of hanging the parent forever; a still-alive child is waited on indefinitely (a legitimate full-history fit runs for hours, so no overall timeout). The worker reports its measured peak working set AND commit charge (`_peak_memory_counters`, stdlib-only) — the promotion RAM guard gates on commit + WS.
- **Memory discipline (recorded 2026-08-10):** `coerce_float32_features(df, feature_cols)` casts exactly-representable schemas (Int*/UInt*/Float32 only — the v5.x integer bins) to a single Float32 polars block; `_feature_frame` returns its **zero-copy numpy view** (polars→numpy verified 0-copy; pandas is skipped — polars→pandas goes through pyarrow and allocates a second full copy, ~36 GiB at 3,555 × 2.1M, the `lgbm_v1` full-history OOM). Float64/mixed schemas pass through untouched. Fold and full-history predicts run in era-batches (`_predict_model_chunked`, 20 eras/chunk) — a full fold-val matrix at 3,555 features (~4.9 GiB float32) exceeds the 4 GiB GPU VRAM (`xgb_v1` CUDA OOM). The **validation stage** predicts in era-batches (`runner._predict_validation_era_batches`, 40 eras/chunk — same boundaries as `_predict_in_era_batches`) so the deploy closure's float64 `to_numpy` stays ≤ ~1.7 GiB. All paths are bit-identical to the pre-optimization code (Int8 bins exact in float32; per-era closure ops order-independent) — determinism tests cover them.
- Fold leakage-safety re-asserted at train time (`_assert_fold_is_leakage_safe(fold, purge_eras=...)`): before fitting each fold it enforces no era reuse, non-empty sides, strict time-ordering, and `min(val) − max(train) > purge_eras` (gap ≤ `purge_eras` ⇒ `ValueError`).
- `_fit_predict_fold` drops null/non-finite target rows from the train slice before fitting (logged dropped count; `ValueError` if nothing remains).
- OOF-CV is GPU-first for lightgbm/xgboost with automatic CPU fallback: LightGBM via `device_type="gpu"`, XGBoost (>= 3.0) via `device="cuda"` + `tree_method="hist"` (`gpu_hist` was removed in 3.x and raises `Invalid Input`). A failed device attempt is logged with `type(exc).__name__` + message (only backend errors — `ValueError`/`TypeError`/`LightGBMError`/`XGBoostError`/`CatBoostError` — trigger fallback; anything else fails loudly), and `resolved_device` records which device actually fit (`"gpu"`/`"cpu"`, `None` before the first successful fit). `model.device` (`auto` | `gpu` | `cpu`, default `auto`) controls CV/experimentation: `gpu` returns only the GPU candidate (a failure raises — no silent fallback), `cpu` never attempts GPU. The run manifest records the config device as `pipeline_device` and the actual fit device as `oof_device`. CatBoost is CPU-only by design — it never attempts a GPU candidate (see below). `train_full_history` is CPU-only by design.
- **Checkpoints & resume (specs 2026-08-20-oof-checkpoint-resume, 2026-08-23-checkpoint-coverage-extension):** three stages checkpoint under `experiments/<slug>/runs/<run_id>/` — `oof_checkpoints/` (CV fold parts), `deploy_checkpoints/` (per-target pickled full-history models), `validation_checkpoints/` (per-era-batch prediction frames). Each root carries a `manifest.json` recording `code_sha256` (SHA-256 of `nmr/models.py` + `nmr/splitter.py` + `nmr/runner.py` source bytes — run_id binds config and data, never code) and `device` (the post-fit `resolved_device`), written atomically at the **first completed unit** (the device is only known post-fit, so an earlier write would record a vacuous `None`). A code/device mismatch, a torn tree (parts without `manifest.json`), or a corrupt part raises `ValueError` with "delete the directory" guidance — never silently reuse stale checkpoints. Shared helpers live in `nmr/_oof.py` (`fitting_code_sha256`, `checkpoint_manifest`, `verify_checkpoint_manifest`, `ensure_no_torn_tree`, `write_frame_atomic`, `write_bytes_atomic`); all writes go through `nmr/_atomicio.py::atomic_write_bytes` (frames serialized to parquet bytes first; never `write_parquet` to the final path). Retention: deleted with the run dir; concurrent duplicate run_ids are unsupported.
  - **OOF folds (2026-08-20):** `train_oof_with_checkpoints(df, *, feature_cols, target_col, splitter, era_col="era", checkpoint_dir) -> pl.DataFrame` — checkpoint-aware OOF-only fold loop over `oof_checkpoints/<target>/fold_NN.parquet`; `train_cross_validation` delegates to the same private `_cv_fold_parts` loop (`checkpoint_dir=None` — every fold fitted), while the checkpoint path loads existing fold parts (models discarded, never exposed). `nmr/_oof.py::train_multi_target_oof(..., checkpoint_dir=None)` routes each target here when set; `ExperimentRunner.run()` passes `experiments/<slug>/runs/<run_id>/oof_checkpoints`. The mixed load+refit resume is bit-for-bit identical to a fresh fit (tested); fold-disjointness is still enforced.
  - **Deploy fits (2026-08-23):** `_build_deploy_pipeline(..., deploy_checkpoint_dir=None)` persists each fitted full-history model with cloudpickle to `deploy_checkpoints/<target>.pkl` right after its fit; a resume loads present targets instead of refitting (the predict closure is rebuilt from the loaded model — cheap and deterministic). The recorded device is the post-fit `resolved_device` (`"cpu"` — full-history fits are CPU-only).
  - **Validation predicts (2026-08-23):** `_run_validation_stage(..., validation_checkpoint_dir=None, checkpoint_device=None)` predicts through the checkpoint-aware `_predict_validation_era_batches` — identical era-batch boundaries to `_predict_in_era_batches` (shared `_era_batch_frames` helper, `_VAL_PREDICT_ERA_BATCH`), persisting each batch to `validation_checkpoints/preds_batch_NN.parquet`; a resume loads present batches and predicts missing ones (byte-identical, tested). `run()` passes `checkpoint_device=str(orchestrator.resolved_device)` — the deploy fits that precede the stage resolve it; a `None` device at manifest init raises loudly (a predict never resolves one). The final `evaluate_model` scorecard call is NOT checkpointed (single call, no clean granularity).

Canonical presets (`_CANONICAL_PRESETS`, mirroring Numerai's published benchmark params):

| Preset | n_estimators | lr | max_depth | num_leaves | colsample_bytree | extra |
|---|---|---|---|---|---|---|
| fast | 2 000 | 0.01 | 5 | 31 | 0.1 | — |
| standard | 20 000 | 0.001 | 6 | 64 | 0.1 | — |
| deep | 30 000 | 0.001 | 10 | 1 024 | 0.1 | `min_data_in_leaf=10000` |

LightGBM adds `objective="regression"`, `random_state=seed`, `n_jobs=1`, `deterministic=True`, `force_col_wise=True`. XGBoost translates `num_leaves→max_leaves`, `min_data_in_leaf→min_child_weight`, adds `reg:squarederror` + `seed`. CatBoost (`_translate_catboost`) renames `n_estimators→iterations`, `colsample_bytree→rsm`, `max_depth→depth`, keeps `min_data_in_leaf` as-is, drops `num_leaves` (symmetric depth-limited trees only), passes every other key through unchanged, then overlays a fixed contract that **wins over user params**: `loss_function="RMSE"`, `random_seed=seed`, `thread_count=1`, `verbose=False`, `allow_writing_files=False` (CatBoost writes files by default; disabled), `task_type` = `"GPU"`/`"CPU"` (+ `devices="0"` on GPU). CatBoost is **CPU-only by design**: `rsm` is incompatible with GPU (non-pairwise modes) and every canonical preset ships `colsample_bytree→rsm`, so `_device_candidate_params` returns a single CPU candidate even when `use_gpu=True` — a GPU fit is never attempted. Determinism: CPU + `thread_count=1` + `random_seed` ⇒ identical OOF across processes (verified); GPU determinism is not guaranteed (same per-device caveat as §6). The `catboost` package version enters the run_id environment fingerprint **only for `backend="catboost"` configs** — lightgbm/xgboost fingerprints stay byte-identical to the legacy shape (§N, §S). `ModelConfig.params` overrides presets key-by-key. `resolve_model_params(preset, params)` is the single source of truth for that resolution (`model.params` wins over `_CANONICAL_PRESETS[preset]`); `ModelOrchestrator._resolved_params` delegates to it, as does the Bayesian sweep baseline anchor (§S).

Presets mirror Numerai's published benchmark params and walk-forward purge convention in [docs/01-canon/models.md](docs/01-canon/models.md).

**Dynamic colsample floor (2026-08-14):** the feature-sampling fraction is raise-only floored per feature count at fit time — `c_effective = min(1.0, max(c_resolved, min(1.0, max(0.1, min(10, |S|)/|S| + 1e-7))))`. Small sets can no longer be crippled by sampling ~1 feature per tree (campaign v2–v4 had `3 × 0.1 → 1`); the 1e-7 expansion guards the float32 truncation hazard in the C++ backends (`static_cast<int>(n_features · fraction)` can land infinitesimally below an integer boundary) and sits inside the `max(0.1, …)` bound so `|S| ≥ 100` configs are bit-identical to pre-floor behavior. Applied in `_resolved_params` on the backend-final sampling key(s): `colsample_bytree` for XGBoost; **every present member** of the LightGBM alias group `{colsample_bytree, feature_fraction, sub_feature}` (one `_ConfigAliases` group in the installed wrapper — unknown kwargs flow through `**kwargs` into the native engine, so flooring a single alias is not precedence-proof); `rsm` post-translation for CatBoost (a user-native `rsm` is bounded identically). `n_features` is threaded `_fit_model → _device_candidate_params → _resolved_params` (both device candidates get the identical floored value); the Optuna baseline anchor still enqueues the raw resolved value (§S, documented divergence).

### H. Ensembling — `nmr/ensemble.py`

`Ensembler`:
- `rank_normalize(df, *, pred_cols, era_col)` — per-era `rank_gaussianize_unit_variance` on each component.
- `blend(df, *, pred_cols, weights=None, out_col="prediction")` — rank-normalize → weighted dot product (default uniform 1/n) → `rank_gaussianize` the combination.
- `learn_weights(oof_df, *, pred_cols, target_col, method="ridge") -> tuple[float, ...]` — rank-normalized design matrix, finite-row masking, then:
  - **ridge** (α=1.0): `solve(XᵀX + I, Xᵀy)`
  - **nnls**: `scipy.optimize.nnls(X, y)`

The weight-learning `method` is driven by `EnsembleConfig.method` in both `run()` (§N) and the HPO path `_held_out_metric` (§L), so sweeps evaluate the same blend the runner deploys. `neutralization_frontier` sweeps proportions explicitly and does not use it.

### I. Statistical Inference — `nmr/inference.py`

| Function | Mechanism |
|---|---|
| `era_series_stats(series) -> SeriesStats(n, mean, std, sharpe, skew, kurt)` | ddof=0; kurtosis Fisher=False, bias=False |
| `resolve_block_len(n, horizon, *, override)` | `round(n^(1/3))`, floors `{20D: 5, 60D: 13}`, capped at n |
| `resolve_bandwidth(n, horizon, *, override)` | `floor(4·(n/100)^(2/9))`, floors `{20D: 4, 60D: 12}`, capped at n−1 |
| `block_bootstrap_ci(data, stat_fn, *, block_len, n_boot, seed, alpha=0.05, min_valid_frac=0.5) -> BootstrapCI` | circular block bootstrap; percentile CI; fails if valid replicate fraction < 0.5 |
| `ac_adjusted_sharpe(series, *, horizon\|bandwidth)` | Lo (1998): `SR / sqrt(1 + 2·Σ wₖ·ρₖ)`, Bartlett weights `wₖ = 1 − k/(k_max+1)`, d-term clipped ≥ 1e-12 |
| `deflated_sharpe(sharpe, *, n_trials, n_obs, skew, kurt, trials_sr_var, sr0_benchmark)` | Bailey–López de Prado DSR: expected-max SR via Gumbel/Euler–Mascheroni; z = `(SR − SR₀)·sqrt(n−1) / sqrt(1 − skew·SR + (kurt−1)/4·SR²)`; returns Φ(z) |

`Horizon = Literal["20D", "60D"]`.

### J. Payout Proxy — `nmr/payout.py`

`payout_series(corr_by_era, mmc_by_era, *, pf=1.0, clip=0.05)`:

```
raw     = pf · (0.75·corr + 2.25·mmc)      # current Numerai Classic weighting
clipped = clip(raw, −0.05, +0.05)
```

Diagnostics on the series: `burn_rate` (fraction < 0), `cvar(q=0.05)` (mean of bottom 5%), `sortino` (downside-deviation ratio), `max_drawdown` (cumsum vs running max), `calmar` (mean/MDD), `max_burn_streak` (longest consecutive negative run), `time_to_recovery` (longest underwater period). `payout_report(...) -> PayoutResult` bundles all of these plus `BootstrapCI` on mean payout, deflated Sharpe, and AC-adjusted MMC Sharpe.

Capital-readiness extensions (v2.5, pure NumPy, float64 throughout): `annual_compounded_return(clipped, *, eras_per_year=52.0)` — `(∏(1 + r_t))^(52/n) − 1`, `−1.0` when the wealth product ≤ 0 (ruin), `0.0` when `n < 2`; `gain_to_pain_ratio(clipped)` — `Σ max(0, r_t) / Σ |min(0, r_t)|`, `+inf` on a zero-pain positive series (precedented by `calmar`; the canonical JSON sanitizer `_sanitize_json_payload` maps non-finite floats to `"Infinity"`/`"-Infinity"`/`"NaN"` strings in canonical bytes while parquet/CSV carry `inf` natively), `0.0` on an all-flat series; `kelly_fraction(raw)` — `min(1.0, max(0.0, μ / σ²))` with `σ² = var(ddof=0)` computed on the **raw** (unclipped) series (`0.0` when `σ² = 0` or `μ ≤ 0`): the clipped series has Popoviciu-bounded variance (≤ 0.0025 under the ±5% clip) so μ/σ² there saturates at 1.0 for every viable model and carries no discrimination; `simulate_overlapping_portfolio(clipped, *, horizon_eras=20, initial_capital=1.0, eras_per_year=52.0) -> OverlappingSimulationResult` — multi-round lockup simulator: at each era, tranches maturing at `t` pay `principal·(1 + r_{t−K})` (the initiating era's return — Numerai round semantics), then `min(cash, total_equity/K)` deploys as a new tranche maturing at `t + K`; equity and utilization are recorded **before** the era's deployment; tranches still locked at series end are carried at par principal (at-cost convention — no mark-to-market, no unrealized payoff); `n < horizon_eras` returns a zeroed result. Horizon eras derive from the report's `horizon` argument via `_HORIZON_ERAS = {"20D": 20, "60D": 60}`. `PayoutResult` gains `cagr_1y`, `gain_to_pain_ratio`, `kelly_fraction` (CAGR/GPR on `series.clipped`, Kelly on `series.raw`) and `overlapping_sim: OverlappingSimulationResult | None = None` (default preserves existing direct constructions).

Payout weights, ±5% clip, and stake thresholds follow [docs/01-canon/staking.md](docs/01-canon/staking.md).

### K. Scorecard — `nmr/scorecard.py`

`evaluate_model(predictions, *, meta_model, benchmarks, features, targets, n_trials, seed, horizon="20D", main_target="target", benchmark_col=None, backend="custom", regime_labels=None, perturbation=None, pf=1.0, clip=0.05, n_boot=1000, alpha=0.05, min_overlap_eras=20, model_id="model", ...) -> MetricScorecard`

Flow: join predictions ∩ meta ∩ targets ∩ features on `[era]` or `[era, id]` (optional left-join benchmarks) → per-era CORR/MMC/FNC (+BMC/CWMM when benchmark/meta available) → payout report → `MetricCell(value, ci_low, ci_high, n_eras)` bootstrap cells → feature-exposure, horizon-stability, regime, and perturbation diagnostics. Horizon targets inferred by regex `_([a-zA-Z0-9]+)(?:20|60)$` on `benchmark_col`, requiring both `target_{name}_20` and `target_{name}_60`.

`MetricScorecard` (frozen, 44 fields) includes `rank_scalar`, `deflated_sharpe`, `mean_payout/corr/mmc/corr_sharpe_ac/bmc/cwmm` cells, `fnc`, `cvar5`, `max_drawdown`, `burn_rate`, `sortino`, `calmar`, `max_feature_exposure`, `degenerate_eras`, robustness sub-results, the capital-readiness block (v2.5), and `metric_timing_seconds` + `eval_total_seconds` instrumentation. Capital-readiness fields: `cagr_1y`, `gain_to_pain_ratio`, `kelly_fraction`, `sim_portfolio_cagr`, `sim_portfolio_mdd`, `sim_capital_utilization` (floats, from `PayoutResult`/`OverlappingSimulationResult`); `mmc_down` (mean MMC over the eras where the meta model's per-era CORR < 0 — `None` + `mmc_down_reason="insufficient_downside_eras"` when fewer than `_MMC_DOWN_MIN_ERAS` (5) such eras; `mmc_down_n_eras` always records the count, `mmc_down_reason` is `None` otherwise); `turnover_mean`/`turnover_std` (mean and population std ddof=0 of `1 − ρ_k` transitions — both `None` when the join lacks the id column (`turnover_reason="id column unavailable"`) or fewer than 2 valid transitions exist (`"insufficient_transitions"`), `None` reason otherwise). `degenerate_eras: tuple[str, ...]` (A4, 2026-08-18) lists the era labels whose per-era CORR was normalized to `0.0` at the engine boundary (<2 usable rows, zero variance, or non-finite values — §E): metric values are unchanged, but the eras are now surfaced rather than silently pooled into the aggregates. `to_frame()` flattens to a single-row Polars frame (cells expand to `{name}`, `{name}_ci_low`, `{name}_ci_high`, `{name}_n_eras`; timings become `timing_*` columns + `quality_metric_timings_json` / `quality_metric_total_seconds`). **Timing columns are excluded from canonical hashing** (§M).

The evaluation spec of record (metrics, gates, build slices E1–E6) is [docs/06-evaluation/evaluation-suite-bible.md](docs/06-evaluation/evaluation-suite-bible.md).

### L. Research & Robustness — `nmr/research.py`, `nmr/robustness.py`

- `HyperparameterSweep(base_config, *, metric="sharpe").run(space, *, n_trials, seed) -> SweepResult(trials, best_params, best_value)` — Cartesian product of the space, shuffled, sampled to `n_trials`; each trial overrides `model.params` and evaluates on a purged held-out split (final ~20% of eras).
- `_held_out_metric(config, *, metric_name) -> float` — the shared held-out evaluation behind `HyperparameterSweep` and `bayesian_sweep` (§S): trains multi-target OOF on the train partition, learns ensemble weights, anchor-fits on train, then blends + neutralizes the held-out partition and computes per-era CORR. Metric set: `mean`/`std`/`sharpe`/`max_drawdown` via `MetricSummary` attributes, plus `corr_sharpe_ac` via `_per_era_ac_sharpe(per_era, horizon="20D")` (metric-resolution contract: §S); unknown names raise `ValueError`.
- `neutralization_frontier(oof, *, feature_cols, proportions, ...) -> NeutralizationFrontier(proportions, metrics)` — sweeps neutralization proportion, per-era CORR + `MetricSummary` at each point.
- `feature_exposure_report(oof, *, feature_cols, ...)` — per-feature mean/max absolute **Pearson correlation** with predictions, vectorized via one `partition_by(era)` pass + per-era matrix op (`_pred_feature_pearson`). Definition change dated 2026-08-05 (previously power-1.5 Numerai CORR per feature) — recorded exposure numbers are **not comparable** across that boundary.
- `adversarial_perturbation(...) -> PerturbationResult(alpha, n_eras, ceiling_stability, manifold_stability, gap, effective_perturb_frac)` — cell-level ±1 bin flips (features are Int8 ∈ [0,4]) plus circular block swaps from train; per-era Spearman stability of predictions.
- `time_horizon_stability(...) -> HorizonStabilityResult` — model vs benchmark AC-adjusted Sharpe on `_20` vs `_60` targets; decay and relative divergence.
- `regime_conditioned_corr(...) -> dict[str, RegimeCorr]` — per-regime per-era CORR with block-bootstrap CIs.

### M. Benchmark Hierarchy — nmr/benchmark.py

The 5-tier escalating benchmark ladder ("the line in the sand"). Config-driven:
`configs/benchmarks/*.yaml` → `load_benchmark_suite_config()` → frozen
`BenchmarkCellConfig` / `Tier4GateConfig` dataclasses (unknown keys rejected,
enum-validated). `BenchmarkHierarchy.run()` scores every cell plus the
`v53_lgbm_ender60` reference column (plus additional `reference_columns`,
e.g. `v53_lgbm_ender20`, scored as informational tier-4 rows) through
`evaluate_model()` and evaluates
three hard gates: `assert_tier0_null_floor` (|CORR| ≤ 0.005, |AC-Sharpe| ≤ 0.15
over the three structural nulls; no DSR check — null DSRs span 0.11–1.0),
`assert_tier4_gate` (6 production thresholds in `configs/benchmarks/tier4_gate.yaml`;
turnover is structurally unavailable on v5.3 — reported as measured=None/pass=None,
excluded from hard failure; **`deflated_sharpe` is display-only — pass=None, A6 2026-08-18**:
at n_trials=1 no deflation occurs and no search history exists at gate time, so gating
on it was false assurance; search-aware DSR lives in `sweep_dsr`/`campaign_evidence`),
and `assert_hierarchy_monotone` (per-tier max of
`corr.value`, T0 < T1 < T2 < T3 ≤ T4, atol 1e-5; `rank_scalar` selectable via
`metric=`). Tier 1–3 fits use `train_validation_purged_split()` (exact 8-era
buffer, strict ordering); multi-target blends are equal-weight in the rank-Gaussian
domain (`Ensembler`); tier-3 neutralization reuses `NeutralizationEngine`; tree
params resolve through `nmr.models.construct_tree_model` (colsample floor,
determinism flags). Determinism: `scorecards_sha256` (timing fields stripped).
FNE is FNC@medium (full 3,555 is prohibited by the feature-universe policy).

### Untiered benchmark fleet (`nmr/benchmark_fleet.py`)

Fleet config schema (`FleetCellConfig` — the tiered cell fields minus `tier`,
plus `source`, `target_weights`, `neutralizer_selection`, `neutralizer_count`),
five generators (`target_lag_mean` — trailing-train target mean; fleet
`lightgbm` — canonical fits + optional riskiest-50 neutralizer selection via
`feature_stability_screen`; fleet `xgboost` — multi-target weighted rank
blend + optional tail-holdout early stopping; `mlp` — sklearn MLPRegressor
with `_standardize_feature_block`; `ridge_stack` — fixed/search two-layer
ridge stacking, horizon-aware 8/16-era internal purge, search mode =
config-driven grids with validation-based candidate selection), and the
`BenchmarkFleet` runner (scored via `evaluate_model`, report-only placement
vs per-tier max-corr rungs, tier-4 verdict columns). Runner CLI:
`--fleet-configs` (default `configs/benchmarks/fleet`), `--fleet-output`
(default `artifacts/reports/benchmark_fleet_scorecard.csv`), `--no-fleet`.
Fleet scorecards join `canonical_scorecards_bytes`. Spec:
`docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md`.

### N. Runner, Registry, Submission, Deployment

**`nmr/runner.py`** — stage order in §1 diagram. `RunResult(run_id, oof, metrics, artifact, manifest, scorecard=None, validation_predictions=None)`. `run_id` = SHA256 of `{config (data_dir/artifacts_dir/supplemental_feature_sets paths stripped; when supplemental_feature_sets is configured, a supplemental_feature_sets_sha256 of the resolved file's CRLF-normalized contents is included — identical files at different roots hash identically, and editing the file changes run identity), data_version, data_fingerprint, code_fingerprint, environment_fingerprint}` where code fingerprint = SHA256 over sorted `nmr/*.py` names+contents and environment = Python + versions of numpy/polars/pandas/lightgbm/xgboost/optuna (plus `catboost` when `model.backend == "catboost"` — config-aware, §G/§S). **data_fingerprint** (B1, 2026-08-18) = SHA256 over per-file snapshots of `train.parquet` + `validation.parquet` (+ `meta_model.parquet`/`validation_benchmark_models.parquet` when `evaluation.validation_scorecard` — config-aware): `{name, footer schema, footer row count, era min, era max, era count}` + `features.json` content SHA256; byte size excluded from the hash (cache key only, cached under `artifacts/cache/`); detection limits documented (restated feature values within unchanged schema/row-count/era-stats are NOT detected); missing data files raise (run_id requires the data snapshot). The run_id scheme bumped once on 2026-08-18 (data term + optuna): future run_ids differ from pre-bump legacy rows; registry rows stay immutable. Ensemble weights are learned on the validation eras of folds `0..K-2` via `EnsembleConfig.method`; when `n_folds < 2` they fall back to uniform `1/n_components` with a logged warning. OOF metrics are computed on the **final fold's** validation eras only (`scoring_eras`), so the OOF scorecard carries no in-sample weight-fitting bias; the returned OOF frame itself still spans every fold. The manifest records `weights`, `weight_learning_eras`, `scoring_eras`, and `summary_metrics` (OOF aggregates for each requested non-MMC metric), plus the rebuild identity (spec §3.1): `data_fingerprint` (the exact value hashed into the run_id, computed once in `__init__`), `code_fingerprint` (the portable full-package sha256, same as the run-id code term), `environment` (sorted `name==version` list over the pinned deps), `pipeline_device` (config knob), `oof_device` (post-fit `resolved_device`). The deploy pipeline is built **at most once** per run, when `deploy or evaluation.validation_scorecard` (`_build_deploy_pipeline`: per-target all-eras CPU-only models + rank-gaussianize + learned weights + neutralize; no `splitter`), and that single closure is shared by the validation stage and the deploy block — never retrained. The **validation stage** (`_run_validation_stage`) loads `validation.parquet` plus `meta_model.parquet` (required — missing ⇒ `FileNotFoundError`) and `validation_benchmark_models.parquet` (optional — BMC/horizon disabled when absent), with target columns = config targets ∪ `main_target` ∪ **every `target`/`target_*` column in the validation schema** (so horizon target pairs reach the scorecard’s inference — loading only config targets silently disabled horizon stability on every runner scorecard), drops the first `split.purge_eras` validation eras (20D-target overlap), scores the shared pipeline, and produces a full `MetricScorecard` with `benchmark_col` = first non-join benchmark column (same convention as `benchmark_runner`); the run manifest records `validation_purge_dropped_first_eras`. Then `_serialize_predict_artifact(predict_fn, model_meta, artifact_path)` serializes (never retrains) to `experiments/{slug}/runs/{run_id}/predict.pkl` + manifest. The artifact's `models` metadata carries `targets`/`weights`/`proportion`/`geometry="all_eras"`/`device="cpu"`/`feature_names`; the run manifest adds `pipeline_device="cpu"`.

**`nmr/registry.py`** — cross-family `RunRegistry(root)` — **comparison + champion pointer only** (run persistence lives in `nmr/experiment_store.py`, §Z). Runs live under the experiments root:

```
experiments/
├── champion.json                 # {"run_id", "experiment_slug", "promoted_at"}  (atomic pointer)
└── {slug}/runs/{run_id}/
    ├── run.json                  # {run_id, metrics{mean,std,sharpe,max_drawdown},
    │                             #  manifest, scorecard{flat scalars}|null, oof_path,
    │                             #  artifact_path|null, artifact_manifest|null}
    ├── oof.parquet
    └── validation_preds.parquet  # [era, id, prediction] on validation eras,
                                  # only when evaluation.validation_scorecard

artifacts/models/<family>/full/manifest.json  # promoted full-version marker (family == run.name; read-only via nmr/families.py)
```

All JSON writes: temp file in parent dir → fsync → `os.replace()`; the OOF parquet likewise writes temp + `os.replace` (no fsync). `list() -> list[str]` returns every run_id across families (sorted); `best(metric="corr_sharpe_ac") -> (run_id, slug) | None` picks the highest scorecard metric across families (runs lacking the metric are skipped); `promote(run_id, slug)` and `promote_if_better(run_id, slug, metric="corr_sharpe_ac") -> (Path, bool)` write `experiments/champion.json` atomically — `promote_if_better` promotes only when the candidate's scorecard metric strictly beats the champion's, honoring direction (`_SCORECARD_METRIC_DIRECTION`: `max_drawdown`/`std_corr` are lower-is-better); a scorecard-bearing candidate may displace a scorecard-less champion; a candidate lacking the metric or an unknown metric raises `ValueError`; `resolve_champion() -> (run_id, slug)` fails loud on a missing/dangling/corrupt pointer (never silently treats it as no champion). **Compat shims (removed in Task 11):** `record(RunResult)` keeps the legacy single-pool layout (`root/<run_id>/` — `tests/test_campaign.py` pins it; stub manifests carry no `config.run.name`); iteration and `promote`/`promote_if_better` additionally accept legacy rows (slug derived from `manifest.config.run.name` when a champion pointer needs one); `promote(run_id, slug=None)` / `promote_if_better(run_id, slug=None, ...)` resolve the slug by scanning the registry root (ambiguous/not-found ⇒ `ValueError`); `nmr/promote.py::resolve_champion_run_id` still reads the legacy pointer file. Champion writes are single-writer (CLI/runner entry points only — design spec §9).

**Retired (2026-08-26):** the pre-rebuild `artifacts/registry/<run_id>/` rows are gone (clean slate by design — design spec §14); their era-range manifest quirk is historical — never trust era-range manifest fields for "what this run was scored on"; use scorecard `*_n_eras` cells and the stored parquet. Registry files stay immutable: document, never backfill.


**`nmr/submission.py`** — `build_submission(predictions, *, id_col="id", pred_col="prediction")`: validates id non-null/unique and predictions finite, converts to open-interval percentile ranks via `tie_kept_rank`, casts (`id` Utf8, `prediction` Float64), sorts by id. `validate_submission(submission, *, live_ids)`: delegates to `numerai_tools.submissions.validate_submission_numerai`, raises `ValueError` listing first 5 extra/missing ids. `write_submission` → CSV.

**`nmr/deployment.py`** — `serialize_predict(predict_fn, *, path, feature_names, models=None) -> DeploymentArtifact(path, manifest)`: cloudpickles `{"predict_fn", "models"}`, writes sibling `{path}.manifest.json` with `created_at`, `feature_names`, `sha256`, environment fingerprint (Python version/implementation/platform + package versions). `load_predict(path)`: verifies SHA256 against manifest (raises `ValueError` on mismatch), unpickles, returns the callable.

**Predict callable contract** (Numerai hosted runtime):

```python
def predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame | None = None) -> pd.DataFrame:
    ...  # returns a 'prediction' column indexed to live ids; deterministic; feature_names order respected
```

### O. Control-Plane Scripts (zero business logic)

| Script | What it does |
|---|---|
| [train_first_model.py](train_first_model.py) | `load_config("configs/first_model.yaml")` → `ExperimentRunner.run(deploy=True)` → `RunRegistry.record` + `promote_if_better` (prints promotion verdict) → prints summary, writes `summary.json` |
| [benchmark_runner.py](benchmark_runner.py) | 5-tier hierarchy control plane: `--data-dir`, `--configs`, `--seed`, `--n-boot`, `--fast-mode`; writes `artifacts/reports/benchmark_hierarchy_scorecard.csv` + `benchmark_gate_report.csv`; exit 1 on hard-gate failure |
| [generate_dashboard.py](generate_dashboard.py) | **Thin entry wrapper** — all logic in `dashboard_ui/report.py`; compiles the deterministic offline Model Tournament `artifacts/dashboard.html` (112 KiB artifact budget, enforced by `dashboard_ui.report.MAX_ARTIFACT_BYTES`) on the `nmr/dashboard.py` engine; CLI surface unchanged (`python generate_dashboard.py`) |
| [dashboard_app.py](dashboard_app.py) | **Thin entry wrapper** — calls `dashboard_ui.app.main`; embeds the same vanilla Model Tournament renderer with `st.components.v1.html`; read-only; launch: `streamlit run dashboard_app.py` |
| `dashboard_ui/charts.py` | pure SVG geometry and compact payload helpers (`data_to_svg_path`, `svg_area_path`, `cumulative_series`, `drawdown_series`, `build_dashboard_payload`, `compact_timeseries_payload`); tested in `tests/test_dashboard_ui.py`; static assets mirror the geometry client-side |
| `dashboard_ui/report.py` | deterministic HTML compiler — `build_dashboard_html` / `generate_dashboard` / `main`; inlines generated `static/{style.min.css, app.min.js, layout.html}` from readable `style.css`/`app.js` renderer sources |
| `dashboard_ui/app.py` | Streamlit app — calls `dashboard_ui.app.main`; the shared vanilla Model Tournament renderer is embedded with `st.components.v1.html`; read-only; launch: `streamlit run dashboard_app.py` |
| [run_campaign.py](run_campaign.py) | Run a named batch of configs and record trial lineage (see §R) |
| [analyze_dataset.py](analyze_dataset.py) | Modular dataset analysis: 17 named stages (`overview`, `targets`, `ic_by_era`, `screens`, `screens_train`, `summary`, `psi`, `drift`, `derived_sets`, `corr_medium`, `corr_all`, `set_membership`, `ic_by_split`, `regimes`, `benchmarks`, `meta_ortho`, `manifest`). Flags: `--only a,b` / `--skip a,b` run a subset (dependencies auto-included; `manifest` always runs), `--features small\|medium\|all`, `--max-eras N`, `--full-all-matrix`. `screens` writes the **descriptive full-span** screen (`feature_ic_screen.parquet`, eras 0001..1231 — never an input to subset derivation); `screens_train` writes the **train-only** screen (`feature_ic_screen_train.parquet`, eras 0001..0574); `derived_sets` reads **only** the train-only screen and writes `derived_feature_sets.json` (`screen_stable`, `screen_nonlinear`, `screen_linear_or_nonlinear`, `screen_drift_filtered` — pure functions of the train-only screen + drift dumps, sorted; see §P); `drift` writes the PSI + W1 + adversarial-AUC profile (`feature_drift_profile.parquet`, `w1_norm = w1 / σ_train`); `meta_ortho` writes per-feature meta-model orthogonality; the FNE profile uses an 11-point neutralization grid. Stage boundaries and per-era ticks print progress to stdout/stderr (never into artifacts); the manifest records `stages_run` + a machine-hardware summary (informational — never hashed). |
| [hardware_status.py](hardware_status.py) | Print machine specs + live resource status (`--record` writes `artifacts/reports/hardware_specs.json`); all logic in `nmr/hardware.py` (stdlib only) |
| [promote_model.py](promote_model.py) | Promote a registry run to a full version (Model Uploads `predict.pkl`): `--run-id` / `--family` / `--models-dir` / `--override-gate` / `--force`; all logic in `nmr/promote.py`; writes `artifacts/models/<family>/full/<run_id>/{predict.pkl, manifest.json}` and, on a tier-4 gate pass, the atomic `current.json` pointer; prints the Model Uploads upload instructions |
| [rehearse_promotion.py](rehearse_promotion.py) | Truncated-window promotion rehearsal (D7 Stage 1): exercises the whole promotion path — including the spawned full-history fit (`NMR_FULL_HISTORY_SPAWN_MIN_BYTES`) — at small scale, measures peak commit + working set for the RAM guard, and writes a `rehearsal: true` artifact that is excluded from `scan_full_versions` and never becomes `current.json` |
| [measure_ram_curve.py](measure_ram_curve.py) | Three-point full-history **commit** curve (measured, not estimated): runs real worker fits at increasing row counts in fresh subprocesses, fits `peak = a + b·rows`, and writes `artifacts/reports/ram_curve.json` (intercept, slope, R², extrapolation factor, both anchors). All logic in `nmr/promote.py` / `nmr/models.py`. **Measured (2026-08-18):** `commit = 2.57 + 8.87e-6·rows` GiB (R² = 0.993), `ws = 1.10 + 8.18e-6·rows` GiB; full-version (6.85M rows) extrapolates to combined commit ≈ 61–65 GiB, working set ≈ 55–58 GiB (86–90% of physical). **Open hypothesis (next cycle):** the per-row slope implies ~2.5 float32-equivalents held simultaneously — suspects sklearn `check_array` copying the polars→numpy view and LightGBM `Dataset` construction (`lgb.Dataset(..., free_raw_data=True)`); target slope ~5e-6 GiB/row → full-scale commit ≈ 40 GiB (own determinism proof required). |
| [refresh_data.py](refresh_data.py) | Round-aware dataset refresh + era-ledger update (`--dry-run`, `--check-only`, `--strict`, `--live-only`); policy in `nmr/refresh.py`, this script only wires `numerapi` calls and file I/O |
| [render_dataset_report.py](render_dataset_report.py) | Renders the LLM-optimized pre-modelling dataset & feature study from the `analyze_dataset.py` dumps + campaign logs (consumes `nmr.config`, `nmr.meta`, `nmr.refresh`) |

### P. Feature-Set Resolution & Stability Screening — `nmr/features.py`

Pure functions over `features.json` and the train frame; no model logic and no file state beyond the explicit `features_json` argument. Subset derivation adds exactly one input to the `run_id` fingerprint beyond the config itself: when `data.supplemental_feature_sets` is configured, the resolved file's CRLF-normalized SHA256 is included with the path stripped (§N) — so the fingerprint is fully determined by config (including `data.feature_subset`) + the supplemental file's contents + data_version + `nmr/*.py` + environment.

`resolve_feature_sets(features_json: Path) -> dict[str, list[str]]` — returns every named set in `features.json` (`feature_sets` must be a non-empty mapping whose values are lists of strings, else `ValueError`), deterministically ordered by set name; values are defensive copies.

`resolve_small_feature_set(features_json: Path, available: Sequence[str]) -> list[str]` — resolves the tutorial-style `small` set restricted to `available` columns, in declared order. Fails loudly on missing/corrupt metadata, a missing `small` set, or an empty intersection (`ValueError`) — never substitutes an arbitrary feature list (the benchmark runner’s baselines and downstream scorecards depend on it; no hidden defaults).

`feature_stability_screen(frame, *, feature_cols, target_col, era_col="era", min_mean_corr=DEFAULT_MIN_MEAN_CORR, max_abs_decay=DEFAULT_MAX_ABS_DECAY) -> pl.DataFrame` — per-feature era-window statistics:

- Per era, Pearson `CORR(feature, target)` computed in one vectorized per-era pass over `[era, target, *feature_cols]` (same pattern as `feature_exposure_report`, §L). Degenerate eras — fewer than 2 usable rows, an all-non-finite target, or a constant target — are excluded from every aggregate: `n_eras` counts valid eras only, and when no valid eras exist the numeric aggregates are `None` with `stable = False`. (Zero-IC padding of degenerate eras would bias the mean, std, and decay slope — label-lag eras carry no signal information, not zero signal.)
- Aggregates across eras (`_SCREEN_COLUMNS` = `feature, mean_corr, corr_std, decay_slope, cross_regime_variance, n_eras, stable`): `mean_corr` (mean), `corr_std` (population std, ddof=0), `decay_slope` (linear slope of CORR vs numeric era index via `np.polyfit(..., 1)`; `0.0` when fewer than 2 eras), `cross_regime_variance` (`0.25 · (first − second)²` — the variance of the first-half vs second-half era-window mean CORR, a regime-drift proxy), `n_eras`.
- `stable` predicate: `mean_corr ≥ min_mean_corr` AND `|decay_slope| ≤ max_abs_decay` AND `n_eras ≥ 2`. Defaults: `DEFAULT_MIN_MEAN_CORR = 0.01`, `DEFAULT_MAX_ABS_DECAY = 0.001`.

`select_stable_features(screen, *, min_mean_corr, max_abs_decay) -> list[str]` — filters the screen frame on `mean_corr ≥ min_mean_corr`, `|decay_slope| ≤ max_abs_decay`, `n_eras ≥ 2` and returns the passing feature names sorted.

`DataConfig.feature_subset` (`str | None`, default `None`) — optional override naming any feature set in `features.json`. Config load rejects empty-string values; the name itself is validated against `features.json` at ingestion time (`IngestionAgent.features` — fail loud, fail late). The `resolved_feature_set` property returns `feature_subset` when set, else `feature_set`; the runner and `HyperparameterSweep` resolve features through it (see also §2A, §B).

`DataConfig.supplemental_feature_sets` (`Path | None`, default `None`) — optional JSON file (`{"feature_sets": {...}}`) whose sets are **merged** into the `features.json` sets by `IngestionAgent` (used by the `feature_sets` property and `features()`). Merge is a pure function of the two files; a missing file, a malformed payload, non-string set values, or a **key collision** with `features.json` raises `ValueError` (fail loud — supplemental keys can never silently shadow packaged sets). Screen-derived campaign subsets (`derived_feature_sets.json` from the `derived_sets` analysis stage) are consumed through this key. The `run_id` fingerprint strips this path from the canonical payload (absolute paths must never enter hashes) and instead includes a SHA256 of the resolved file’s CRLF-normalized contents under `supplemental_feature_sets_sha256` (only when the field is configured, so configs without a supplemental set keep legacy run_ids byte-identical) — identical files at different repo roots hash identically, and editing the file changes run identity.

`feature_ic_screen(...) -> pl.DataFrame` — the chunked analysis twin of `feature_stability_screen` (§P), adding `mean_spearman` (mean per-era Spearman rank IC over valid eras), `nonlinear` (`|Pearson| ≤ min_mean_corr` AND `|Spearman| > min_mean_corr`), the **UQ columns** `mean_corr_ci_lo` / `mean_corr_ci_hi` — 95% stationary block-bootstrap CI (20D/60D horizon-aware block floors via `resolve_block_len`, seeded `IC_CI_SEED`) on the era-mean Pearson IC over valid eras — and the **FDR columns** `ci_excludes_zero`, `p_value`, `fdr_q`, `fdr_pass`. `stable` is the full gate: classic point predicate (mean/slope/n_eras from `_aggregate_screen`) AND `ci_excludes_zero` (both CI bounds strictly on the same side of zero) AND Benjamini–Hochberg `fdr_q ≤ 0.05` (`SCREEN_FDR_Q`), with BH computed per target over that target's p-values only (never pooled across 20D/60D horizons). `p_value` is Hall's null-shifted two-sided block-bootstrap p (same seeded machinery and budget as the CI); degenerate series (constant up to 1e-12, or < 2 valid eras) yield p = 1.0 / null and can never be stable (fail-safe — see the degenerate-data doctrine, AGENTS.md §2.5). Output is cast to the explicit 16-column `SCREEN_PARQUET_SCHEMA` dtype contract at the function boundary. Subset derivation consumes the **train-only** screen (`screens_train` stage, `feature_ic_screen_train.parquet`); the full-span screen is descriptive only (§O).

Drift diagnostics (`feature_drift_psi`, `feature_drift_profile`): PSI over train-decile bins (`psi > 0.25`); raw 1-Wasserstein `w1` (kept for reference) plus **scale-standardized `w1_norm = w1 / σ_train`** (train-sample sigma) so one threshold works across bounded and unbounded features; adversarial AUC (Mann-Whitney separation, `|auc − 0.5| > 0.1`). `drifted = psi > 0.25 OR w1_norm > 0.50 OR |auc − 0.5| > 0.1` (constants `PSI_FLAG_THRESHOLD`, `WASSERSTEIN_NORM_FLAG_THRESHOLD`, `AUC_FLAG_DELTA`).

`neutralized_ic_series(chunks, signal_cols, feature_cols, target_col, *, era_col="era", proportion=1.0) -> pl.DataFrame` — per-era long-form of the FNE math (same intercept-aware pinv neutralization, `rcond=1e-6`); mean over eras matches `neutralized_ic_profile` at the same proportion (parity-tested). Feeds bootstrap CIs and paired comparisons on post-neutralization signal.

### Q. Cross-Run Meta-Analysis — `nmr/meta.py`

Decision layer on top of `nmr.inference` and `nmr.evaluation`; reuses the repo's seeded block-bootstrap machinery and never mutates the registry.

`paired_era_comparison(oof_a, oof_b, *, metric_fn, era_col="era", horizon="20D", n_boot=1000, seed, alpha=0.05, min_overlap_eras=MIN_OVERLAP_ERAS, block_len=None, device_a=None, device_b=None) -> PairedResult` — `metric_fn` maps an OOF frame to `{era: metric}` (e.g. a closure over `EvaluationEngine().per_era_corr`). A missing `era_col` in either frame raises `ValueError` naming the frame(s); eras are intersected on the numeric era index, and fewer than `min_overlap_eras` overlapping eras raises `NonVacuityError` (default `MIN_OVERLAP_ERAS`, §E). The era-level diffs (`A − B`; positive `mean_diff` means A is better) are block-bootstrapped via `block_bootstrap_ci` (§I) with `block_len` resolved by `resolve_block_len(n, horizon)` unless overridden. `PairedResult(mean_diff, ci_low, ci_high, n_eras, device_mismatch, alpha, n_boot, block_len)` is frozen. `device_mismatch` is `True` when both `device_a` and `device_b` are supplied and differ — GPU vs CPU OOF values are not comparable ([`AGENTS.md`](AGENTS.md#8-critical-operational-hazards)) — and is reported, never silently corrected.

`promotion_verdict(candidate, champion, *, metric="corr_sharpe_ac", alpha=0.05) -> "promote" | "hold" | "caution"` — significance-aware promotion decision on registry entries via CI-bearing scorecard cells. Directions (`_VERDICT_DIRECTIONS`) mirror `RunRegistry._SCORECARD_METRIC_DIRECTION` (higher-is-better: corr, mmc, fnc, corr_sharpe_ac, deflated_sharpe; lower-is-better: std_corr, max_drawdown) and are parity-tested in `test_meta.py`. Higher-is-better: `candidate.ci_low > champion.ci_high` → `"promote"`; `candidate.ci_high < champion.ci_low` → `"hold"`; mirror logic for lower-is-better; any CI overlap → `"caution"`. No champion, or a champion lacking the metric, → `"promote"`; a candidate lacking the metric raises `ValueError`; a missing CI on either side → `"caution"`. Advisory only — never writes the registry. `alpha` is accepted for API symmetry and does not change the CI-separability rule.

`fleet_summary(runs, *, metric="corr_sharpe_ac", n_trials, dsr_confidence=0.95) -> pl.DataFrame` — flattens registry entries into a per-run fleet table: the requested scorecard metric (value + CI + n_eras), stored `deflated_sharpe` with a `dsr_pass` flag against `dsr_confidence`, `max_feature_exposure`, `oof_device`, manifest-config grouping columns (`preset`, `feature_set`, `feature_subset`, `neutralization_proportion`), robustness presence flags (`has_bmc`, `has_horizon`, `has_perturb`, `has_regime`), and policy context columns (`policy_n_trials`, `policy_dsr_confidence`). Runs without a scorecard are flagged (legacy), never silently dropped. Deterministic: sorted by metric desc, run_id tiebreak. **DSR policy note:** the stored `deflated_sharpe` was computed with `n_trials=1` at scorecard time (standalone contract); campaign-aware and sweep-aware deflation are computed post-hoc by `campaign_evidence` (below) and `opt.sweep_dsr` (§S) — never here.

`campaign_evidence(campaign_log_path, registry_root, *, data, main_target="target", fne_reference_set="medium", n_boot=200, seed=0, min_overlap_eras=MIN_OVERLAP_ERAS) -> CampaignEvidence` — assembles per-variant validation evidence for every recorded run of a campaign log: validation mean IC with the run scorecard's 95% CI, IC Sharpe and max drawdown (scorecard), feature count and backend/device (manifest), and FNE at 100% neutralization against `fne_reference_set` with its own bootstrap CI (per-era residual ICs via `neutralized_ic_series`). **Campaign-aware DSR (2026-08-14):** per recorded cell with `n_eras >= 4` and `ic_std > 0`, the full-window IC Sharpe is deflated post-hoc against the empirical cross-cell Sharpe distribution — `n_trials` = valid cell count, `trials_sr_var` = sample variance of the cells' Sharpes (ddof=1, no analytic fallback), via `inference.deflated_sharpe_fleet`. Columns: `dsr_campaign_aware`, `dsr_pass_campaign` (≥ 0.95), `dsr_reason` (`zero_cross_trial_sharpe_variance` / `degenerate_series` / `insufficient_trials` / `radicand`), `dsr_n_trials`, `dsr_trials_sr_var`; evidence assembly never crashes on degenerate fleets (Guard A). `pairwise` block-bootstraps the per-era validation-IC difference for the screen-defining pairs (v2 vs v3, v2 vs v4, v3 vs v4) per config-prefix (e.g. `lgbm_v2`), positive diff = first variant better; `NonVacuityError` pairs surface as error rows. Runs with status ≠ `recorded` or missing artifacts become error rows — never silently dropped. Requires `validation_preds.parquet` per run (persisted by `RunRegistry.record` when the validation scorecard is enabled). Headline metrics (`mean_ic`, CI, Sharpe, max drawdown) are computed on the **full validation window** from the per-era IC series (numeric-ordered — lexicographic era sorts scramble the block bootstrap), with the scorecard's 86-era meta-overlap cells kept as explicit `scorecard_*_86era` columns. The assembled evidence for the 2026-08 benchmark-rebuild campaign is persisted at `artifacts/reports/dataset_analysis/campaign_{variants,pairwise}.parquet` and rendered into §7 of the joined pre-modeling document `docs/04-research/pre-modelling-dataset-feature-study-2026-08.md` (also §8 operational findings: purge-bug fix, HPO padding fix, hardware ceilings).

### R. Campaign Orchestration — `nmr/campaign.py` + `run_campaign.py`

A campaign is a named batch of experiment configs whose runs share a hypothesis; the module provides deterministic trial-lineage attribution on top of the registry. No wall-clock fields are stored in the log (canonical-determinism friendly; file mtime carries chronology).

`campaign_id(name, config_paths) -> str` — SHA256 over the JSON `{"name": name, "configs": sorted(per-file content SHA256 digests)}` (sorted keys). Order-independent (digests are sorted) and path-independent (content hashes, so moving or renaming config files does not change identity). Empty name or empty config list raises `ValueError`.

`build_campaign_log(name, config_paths, runs) -> CampaignLog` — validates the name, that every config path exists (`FileNotFoundError` otherwise), and that each `CampaignRun` status is in `("recorded", "skipped", "error")`; non-error runs must carry a `run_id`. Returns the frozen `CampaignLog(campaign_id, name, configs: tuple[CampaignConfig(path, sha256), ...], runs: tuple[CampaignRun(config_path, run_id, status, error=None), ...])`.

`write_campaign_log(log, campaigns_dir) -> Path` — writes `campaigns_dir/{campaign_id}.json` atomically via `nmr._atomicio.atomic_write_text` (temp + fsync + `os.replace`). Payload schema (`CampaignLog.to_payload()`, JSON `indent=2`, sorted keys):

```json
{
  "campaign_id": "<sha256 hex>",
  "name": "<campaign name>",
  "configs": [{"path": "<config path>", "sha256": "<content sha256>"}],
  "runs": [{"config_path": "<config path>", "run_id": "<sha256 hex>|null", "status": "recorded|skipped|error", "error": "<message>|null"}]
}
```

CLI contract (`run_campaign.py`, zero business logic): `--config` (repeatable, required), `--name` (required), `--registry` (default `artifacts/registry`), `--campaigns-dir` (default `artifacts/campaigns`), `--deploy` (passes `deploy=True` to `ExperimentRunner.run`), `--dry-run` (prints computed `run_id`s without training or writing — the registry is not even constructed). Per config: an invalid config is logged and recorded as `status="error"` with the message; an already-recorded `run_id` is `"skipped"`; a successful run is recorded via `RunRegistry.record` as `"recorded"`. Prints one `status<TAB>config_path<TAB>run_id` line per run; exits `0` when no run failed, `1` otherwise (`--dry-run` always exits `0`).

### S. Bayesian HPO — `nmr/opt.py`

`bayesian_sweep` is the single Optuna-integration point (user-granted dependency, pinned `optuna==4.9.0` in `requirements.txt`, imported only here). Spaces are declarative dicts; the objective is harness-internal (`research._held_out_metric`, §L); sweeps are seeded, single-threaded, and deterministic per environment.

`bayesian_sweep(base_config, space, *, n_trials, seed, metric="sharpe", n_startup_trials=10, enqueue_base_config=True, n_jobs=1) -> SweepResult`

- **Space schema.** `space` maps a parameter name to a spec dict with `kind` ∈ `("float", "int", "categorical")`. Float/int specs take `low`/`high` (both required, `low ≤ high`) plus optional `log` (must be a boolean; requires `low > 0`). `step` is **int-only**: valid on int specs (positive int, mutually exclusive with `log`) and rejected with `ValueError` on float specs. Categorical specs take `choices` — a non-empty list of JSON primitives (`str`/`int`/`float`/`bool`). Unknown spec keys, invalid kinds or bounds, non-boolean `log`, `step` on a float spec, or non-primitive choices raise `ValueError` **before any trial** (an empty space raises too).
- **Metric resolution.** `metric` ∈ `("mean", "std", "sharpe", "max_drawdown", "corr_sharpe_ac")`; anything else raises `ValueError` at call time. `mean`/`std`/`sharpe`/`max_drawdown` come from `MetricSummary` attributes of the per-era CORR summary; `corr_sharpe_ac` comes from `research._per_era_ac_sharpe(per_era, horizon="20D")`, which sorts era keys numerically (`sorted(per_era, key=int)`) before AC adjustment — the frame's lexicographic era order (`"1","10","11",…`) would corrupt the autocorrelation.
- **Baseline anchor.** With `enqueue_base_config=True` (default), the study enqueues the resolved baseline before `optimize`: `resolve_model_params(base_config.model.preset, base_config.model.params)` (§G) intersected with the space keys — so **Trial 0 is the resolved baseline** (preset defaults overridden by `model.params`), unless no space key overlaps the resolved params.
- **Parameter resolution.** Each trial overrides `model.params` via `_override_config(base_config, params)`; `model.params` wins over `_CANONICAL_PRESETS[preset]` key-by-key, and `resolve_model_params` is the single source of truth for that resolution (§G — this satisfies its ARCHITECTURE.md §S docstring reference).
- **Determinism contract.** Seeded TPE sampler (`TPESampler(seed=..., n_startup_trials=...)` — deterministic-by-default since Optuna 4.x, which removed the 3.x `deterministic` flag; verified on 4.9.0) with in-memory storage. `n_jobs` must be `1` (`ValueError` otherwise): parallel trials break TPE determinism. Reproducibility is **per environment** (same caveat as GPU vs CPU OOF divergence, §6) and rests on the `optuna==4.9.0` pin — `optuna` is not part of the run_id environment fingerprint (`_environment_fingerprint` covers numpy/polars/pandas/lightgbm/xgboost, plus `catboost` for catboost-backend configs), so changing the pin changes the search without the fingerprint flagging it.
- **Error handling.** Any objective exception (from `_held_out_metric`) raises `optuna.exceptions.TrialPruned` — never dummy numerics — and `gc.collect()` runs per trial. `SweepResult` is constructed post-hoc from `study.trials`: `trial_id`, `params_json` (sorted keys), `metric_value` (`t.value` for `COMPLETE` trials, else `None`), `metric`; rows sorted by `metric_value` desc with `trial_id` tiebreak, nulls last. If no trial completed, `best_params={}` and `best_value=nan`.

`SweepResult` is the shared frozen dataclass (§L): `trials: pl.DataFrame`, `best_params: dict[str, Any]`, `best_value: float`.

**Sweep-aware DSR (2026-08-14):** both sweeps capture the held-out per-era IC moments per trial (`ic_sharpe`/`ic_skew`/`ic_kurt`/`ic_n_eras`/`ic_std` via `research._held_out_metric_full` — same training pass, same series as the metric) into `SweepResult.trials`. The pure post-hoc helper `opt.sweep_dsr(trials)` appends `dsr_sweep_aware`, `dsr_pass_sweep` (≥ 0.95), `dsr_reason`, `dsr_n_trials`, `dsr_trials_sr_var` — `n_trials` = valid trial count, `trials_sr_var` = empirical sample variance (ddof=1) of the trials' IC Sharpes, via `inference.deflated_sharpe_fleet`; degenerate/zero-variance fleets yield None + a reason code (Guard A), never a crash or an analytic fallback. Per-run scorecards are untouched (`n_trials = 1` standalone contract).

### T. Project Skills — `.kimi-code/skills/` (Kimi research protocols)

Project-scope Kimi skills encode the research-orchestration protocols built on the modules above. Each protocol lives in its `SKILL.md` (that file is the source of truth for the protocol); this section is only the map:

| Skill | What it drives |
|---|---|
| `feature-campaign` | Feature subsetting + cross-regime stability screening (§P) → `run_campaign.py` batches (§R) → human-reviewed selection |
| `hpo-narrowing` | Multi-stage HPO: coarse `HyperparameterSweep.run` (§L) → `bayesian_sweep` (§S) → full-run confirmation via `promotion_verdict` (§Q); promotion is always a human decision |
| `run-meta-analysis` | Paired fleet comparison + robust-family selection (`nmr/meta.py`, §Q) — read-only over the registry |
| `verification-before-claim` | QA gate for every agent output: full suite, canonical-hash purity, parity tests, 8/16-era purge floor, doc SSOT same-change-set, `oof_device` check |

All four are committed at `.kimi-code/skills/<name>/SKILL.md` (project-level Kimi skills — discovered automatically by the CLI; the directory is tracked, not gitignored).

### U. Hardware Discovery & Status — `nmr/hardware.py`

Stdlib-only system probing (no new dependencies): CUDA device discovery via the `nvidia-smi` CLI, RAM via ctypes `GlobalMemoryStatusEx` (Windows) or `/proc/meminfo` (Linux), CPU usage via `GetSystemTimes` (Windows) or `/proc/stat` (Linux). `discover_hardware()` returns the machine-constant `HardwareSpec` (safe to record — the dataset-analysis manifest embeds a summary); `hardware_status()` returns the instantaneous `HardwareStatus` and **must never enter canonical hashes or run_id payloads**. Pure parsing helpers (`parse_gpu_devices`, `parse_gpu_status`, `parse_meminfo`, `parse_cpu_times`) are the tested boundary. GPU acceleration policy: `ModelOrchestrator` (§G) is GPU-first with CPU fallback. The analysis pipeline uses `nmr/_gpu.py`: cupy-accelerated `rankdata` (bit-identical to scipy on finite data, ~5.8× on era-sized matrices) with automatic scipy fallback when cupy is absent — cupy and the `nvidia-*` runtime wheels are user-granted, optional dependencies (see `requirements.txt`); the analysis remains fully functional without them.

**Thread-pool limits (2026-08-23):** `apply_thread_limits(max_threads: int | None = None) -> int` caps polars/OpenMP/BLAS threading from process start. Resolution order: explicit `max_threads` > `NMR_MAX_THREADS` env > `min(os.cpu_count() or 1, 8)`; an invalid `NMR_MAX_THREADS` (non-int or < 1) raises `ValueError` (fail loud). It sets `POLARS_MAX_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` only when not already present — user-set values win. Call it at process start; note that importing `nmr.hardware` runs the package `__init__`, so polars/numpy may already be in `sys.modules` — the cap still governs because polars reads `POLARS_MAX_THREADS` lazily at pool creation and LightGBM/XGBoost read `OMP_NUM_THREADS` at fit time; numpy's BLAS DLL is the residual (loaded before the call and not retroactively capped). The heavy CLIs (`benchmark_runner.py`, `run_campaign.py`, `analyze_dataset.py`, `train_first_model.py`, `promote_model.py`, `rehearse_promotion.py`) run it as their first executable statement.

Measured on the dev box (RTX A1000 Laptop, 4 GiB VRAM, driver 580.97; recorded 2026-08-09): xgboost `device="cuda"` trains **9.1×** faster than CPU (`hist`, `n_jobs=1`) on 300k×780, 300 trees — 13.5 s vs 123.3 s — and the full 3,555-feature universe fits the 4 GiB device; cupy `rankdata` is **5.8×** faster than scipy at 3555×7000 (0.40 s vs 2.33 s). scipy 1.17 `rankdata` returns an all-NaN array when any input is NaN (`nan_policy='propagate'`); `_gpu.rankdata` isolates NaN at the NaN positions instead (intentionally more correct; v5.3 features contain no NaN, so both paths agree on real data).

### V. Training & Analysis Progress Markers

Long-running paths print console progress that never enters artifacts: `analyze_dataset.py` logs `[stage i/n] name ... done (Xs)` per stage and `[label] era k/N` ticks per 100 eras (stderr); `ModelOrchestrator` prints `[fit] lightgbm iteration N` every `_FIT_PROGRESS_PERIOD` (100) iterations (CatBoost: period `verbose`; xgboost 3.x's sklearn wrapper has no callback hook — start/elapsed markers only); `benchmark_runner.py` logs per-strategy start/memory/elapsed. Progress is wall-clock output only — excluded from all canonical hashes by construction (it never reaches artifacts).

### W. Executive Dashboard Engine — `nmr/dashboard.py` + `dashboard_ui/`

`nmr/dashboard.py` — Model Tournament engine — unified leaderboard, explicit metric directions with default `mmc` ranking, source/tier-derived cohorts, deterministic rank maps, ML ADVANTAGE comparisons, compact immutable model dossiers, tier-4 gate projection, stored-first capital recompute, 7-metric multimetric timeseries, and pairwise rank-gaussian similarity. `UNIFIED_SCHEMA` is unchanged. `dashboard_ui/charts.py` provides tested SVG geometry plus compact columnar payload encoding; `dashboard_ui/static/{layout.html,app.js,style.css}` is the single leaderboard-first renderer for search, rank switching, cohort filters, shortcuts, landscape/profile charts, and the model dossier drawer. `dashboard_ui/report.py` compiles the deterministic single-file HTML using generated `app.min.js` and `style.min.css`; `dashboard_ui/app.py` embeds the same HTML with `st.components.v1.html`. The report is explicitly offline evaluation and contains no live or production performance. Presentation tests live in `tests/test_dashboard_ui.py`.

Dashboard cohorts are presentation-derived, not a new registry field: trained rows are `source in {trained, trained_legacy}`; heuristic rows are benchmark tiers 0–2 (null, Ridge, and shallow trees); benchmark rows are tiers 3–4 (canonical/community and official references); and `source == full` rows are lineage-only with null comparable metrics. Every rank metric declares whether higher or lower is better; nulls sort last and ties use `model_id`. The ML ADVANTAGE strip compares the best trained row with the best available heuristic and benchmark rows on the active metric. RAPS and win-rate are not active scorecard fields and are omitted rather than inferred.

The browser renderer is pointer-driven across all chart surfaces: mouse hover and touch `pointermove`/`pointerdown` events show a local contextual tooltip, with a crosshair on the era chart. The alpha trajectory includes a dynamic title, y-axis metric ticks, x-axis era ticks, and explicit `Evaluation era` labeling. Benchmark rows receive a distinct visual background/stripe, and the first three trained rows in the active ranking receive gold, silver, and bronze `(1)`, `(2)`, and `(3)` medal markers while retaining their true leaderboard ranks.

**Standardized window & regeneration rule:** all dashboard rows (trained runs and benchmark tiers) are compared on the **meta-overlap window** = `validation.parquet ∩ meta_model.parquet` — currently eras `1133..1218` (86 eras); it is meta coverage, not an arbitrary choice. `meta_model.parquet` (v5.3) only exists from era 1133 onward, and the window moves forward as the local data snapshot is refreshed (`refresh_data.py` + `nmr/refresh.py`; the live file expands weekly). **After every data refresh, regenerate the report** (`./.venv/Scripts/python generate_dashboard.py`) — a stale `artifacts/dashboard.html` would compare rows on a window that no longer matches the refreshed data. The capital-cell recompute derives its era axis from the same meta overlap at generation time.

### X. Experiment Path Derivation — `nmr/paths.py`

Pure path derivation for the experiment layout: the single place that knows
where anything lives under `experiments/` (repo root) and the shared machine
cache. No other module hardcodes the `experiments` / `artifacts` strings.
Consumes `nmr/config.py` (`REPO_ROOT`) only; reads/writes nothing and never
enters a canonical hash (the shared helpers take a config-provided
`artifacts_dir` override). API: `EXPERIMENTS_ROOT`, `SLUG_RE` /
`validate_slug` (lowercase `^[a-z0-9_-]+$` family slugs), `experiment_dir`,
`run_dir`, `run_json_path`, `export_dir` (scope ∈ `partial`/`full`),
`export_json_path`, `current_pointer_path`, `champion_path`,
`shared_cache_dir`, `shared_reports_dir`.

### Y. Experiment Lifecycle — `nmr/lifecycle.py`

Read-only lifecycle derivation over the experiment layout (§X): export
validity, total stage derivation, and deterministic ordering. Consumes
`nmr.paths` (layout) and `nmr.deployment.load_predict` (hash-verified
loadability as the export-validity predicate — trusted-source rule; imported
lazily so importing this module stays light). API: `SCOPES`,
`LIFECYCLE_STAGES`, `StakedRecord`, `ExportVersion`, `load_staked_record`,
`valid_export`, `scan_valid_exports`, `current_full_status`, `derive_stage`,
`sort_exports`. Six lifecycle stages in badge precedence: `uninitialized` →
`research` → `partial` → `degraded` → `full` → `staked`. Export identity
binding: slot-dir `run_id` == `export.json.promoted_from_run_id` == family
slug match; a `partial` export additionally requires `scorecard.json`.
`derive_stage` is a total function over filesystem state returning
`(lifecycle_stage, current_full_status)`.

### Z. Experiment Persistence & Atomic Publication — `nmr/experiment_store.py`

Write-path companion to §X/§Y: run persistence, family-scaffold creation, and
atomic export publication. Consumes `nmr.paths` (layout) and
`nmr._atomicio.atomic_write_text` (temp + fsync + replace). API:
`ensure_family`, `record_run`, `read_run`, `stage_export`,
`discard_staged_export`, `publish_staged_export`. `record_run` creates the
family scaffold (`meta.json` + `base_config.yaml` + `README.md`) atomically
with the first `run.json` (spec §2 family-creation rule); `base_config.yaml`
is the NON-authoritative reference copy — the per-run `run.json` config is
authoritative. Export slots are staged at `exports/<scope>/.tmp-<run_id>/` and
published by a single `os.replace` directory rename into the immutable slot
`exports/<scope>/<run_id>/`; discovery ignores `.tmp-` names. Re-publishing an
existing slot raises `ValueError` (exports immutable — spec §6); staged
residue is discarded, never half-published. `run_id` is regex-validated
(`^[0-9a-f]{64}$`, path-traversal guard).

---

## Model Families & Full Versions (nmr/families.py)

A model **family** is the set of registry runs sharing `manifest.config.run.name`
(e.g. `brb1-xgb-v6`; duplicate reruns belong to one family). Promotion to a
**full version** (trained on train+validation, deployed) writes one immutable
slot per promoted run at `artifacts/models/<family>/full/<run_id>/manifest.json`
plus an atomic `current.json` pointer (`{"run_id": <64-hex>, "promoted_at": ...}`,
temp + fsync + `os.replace`) naming the active slot. The pointer + valid slot
manifest IS the marker. `nmr/families.py` is the read-only discovery layer
(writes live in `nmr/promote.py`, the promotion writer): resolution is
pointer-driven — a missing/corrupt/dangling `current.json` fails loud via
`full_manifest_path` (listing `available_slots`) and yields `None` from the
tolerant `load_full_version`/`scan_full_versions` scans. Slots are never
selected by mtime. Old slots remain for rollback; repointing `current.json`
is a deliberate write.

Manifest schema (`manifest.json`, per-slot):

| Field | Requirement |
|---|---|
| `family` | equals the directory name (lowercase `^[a-z0-9_-]+$`) |
| `training_scope` | `"full"` |
| `promoted_from_run_id` | non-empty registry run id (dangling lineage warns, never invalidates) |
| `promoted_at` | display metadata only — never in a canonical hash |
| `artifact_path` | non-empty relative path — no leading `/`, no drive letter, no `..`; resolved against the manifest's own slot dir; file must exist (hollow promotions rejected) |
| `config` | snapshot of the promoted research config |
| `rehearsal` | `true` for a D7 truncated-subset rehearsal artifact — first-class discriminator: excluded from `scan_full_versions`/`family_has_full_version` and NEVER the `current.json` pointer, so it can never be read as a genuine full version at a glance (review directive 2026-08-18) |
| `training_rows` / `training_era_range` | actual rows + `[min, max]` era range the artifact was fit on — a rehearsal (~68k rows on a subset) is distinguishable from a genuine full version (6.85M rows, `[0001..1231]`) without reading `config_normalizations` |
| `tier4_gate_passed` / `tier4_receipts` / `override_used` / `config_normalizations` | promotion verdict block — always written by the writer; a failed-gate rehearsal artifact carries `tier4_gate_passed: false` in its own manifest |

Public API: `full_manifest_path` (fail-loud pointer resolution),
`available_slots`, `load_full_version`, `scan_full_versions`,
`family_has_full_version`, `FullVersion`, constants `FAMILY_DIR_NAME` /
`FULL_DIR_NAME` / `FULL_MANIFEST_NAME` / `CURRENT_POINTER_NAME` /
`DEFAULT_MODELS_DIR`.

Leaderboard integration (`nmr/dashboard.py`): `UNIFIED_SCHEMA` carries
`family`, `training_scope` (`"research"` / `"full"`), `has_full_version`.
`load_unified_leaderboard` scans `artifacts/models/` ONCE, stamps trained rows
via set membership, and appends one `source="full"` row per valid manifest
(`model_id = "<family>::full"`, all metric cells null — in-sample metrics are
never shown as comparable OOF numbers). `evaluate_gate_status` stamps full
rows `FULL` (all gate receipts null). `EVALUABLE_ROWS = pl.col("source") != "full"`
is the single chart-inclusion predicate (source-based so benchmark rows with
null `training_scope` remain visible).

---

## 3. Module Dependency Graph

Verified against the source imports (`tests/test_docs_hygiene.py::test_architecture_documents_every_module`
asserts every `nmr/*.py` appears here).

```
_atomicio.py  (leaf — temp + fsync + os.replace writers)
_gpu.py       (leaf — optional cupy rankdata, automatic scipy fallback)
_transforms.py(leaf — rank/gaussianize/power-1.5/neutralize)
config.py     (leaf — no nmr imports)
hardware.py   (leaf — stdlib system probing)
inference.py  (leaf — NumPy/SciPy only)
refresh.py    (leaf — stdlib only; pure refresh policy, no I/O/numerapi)
features.py   (leaf — stdlib/NumPy/Polars only)

data.py      ──> config (DataConfig)
splitter.py  ──> config (SplitConfig)
families.py  ──> config (REPO_ROOT)
paths.py     ──> config (REPO_ROOT)
lifecycle.py ──> paths, deployment (load_predict — export-validity predicate)
evaluation.py──> _transforms (power_1_5, rank_gaussianize)
ensemble.py  ──> _transforms (rank_gaussianize, rank_gaussianize_unit_variance)
payout.py    ──> inference
campaign.py  ──> _atomicio
deployment.py──> _atomicio (cloudpickle artifact + integrity manifest)
risk.py      ──> _atomicio, _transforms, config
submission.py──> _transforms (tie_kept_rank), deployment, numerai_tools.submissions
models.py    ──> config (ModelConfig), data, splitter (Fold, PurgedEraSplitter)
_oof.py      ──> _atomicio, models, splitter (shared multi-target OOF + checkpoints: runner + research)
robustness.py──> _transforms, evaluation, inference
analysis.py  ──> _transforms, features, inference
research.py  ──> _oof, config, data, ensemble, evaluation, inference, models, risk, splitter
scorecard.py ──> evaluation, inference, payout, research, robustness
opt.py       ──> config (ExperimentConfig), inference, models (resolve_model_params), research (_held_out_metric, _override_config, SweepResult)
benchmark.py ──> ensemble, features, models, risk, scorecard
benchmark_fleet.py ──> benchmark, ensemble, features, models, risk, scorecard
meta.py      ──> analysis, config, data, evaluation, features, inference
runner.py    ──> _oof, _transforms, config, data, deployment, ensemble, evaluation, models, risk, scorecard, splitter
registry.py  ──> _atomicio, experiment_store, paths, runner (RunResult)
dashboard.py ──> benchmark, config, ensemble, evaluation, families, payout, scorecard
explainers.py ──> dashboard_ui.service (read-only dynamic model labels)
scenarios.py  ──> payout, evaluation (allocation scenario research helpers)
promote.py   ──> _atomicio, benchmark, config, data, families, models, runner, submission

nmr/__init__.py re-exports the public API of all modules (keep imports and __all__ in sync).
```

### Refresh ledger (`data/numerai_era_data.csv`)

Round-aware refresh (policy in `nmr/refresh.py`, wiring in the root script
`refresh_data.py`) maintains the era ledger with these columns:

| Column | Type | Notes |
|---|---|---|
| `date` | ISO date | Date the record was written |
| `dataset` | `train` / `validation` / `live` | One row per dataset per refresh |
| `start_era` / `end_era` | zero-padded string | Read from the parquet `era` column; `"X"` for `live` (unlabeled rounds) |
| `round_id` | float when present, empty otherwise | Tournament round, set only for `live` |

Refresh triggers: `live.*` files every round advance; weekly-expanding files
(`validation.parquet`, `validation_benchmark_models.parquet`,
`validation_example_preds.*`, `meta_model.parquet`) on round advance; static files
(`train.parquet`, `train_benchmark_models.parquet`, `features.json`) only when
missing. `--live-only` skips expanding files. Writes are atomic via `_atomicio`.

---

## 4. Configuration & Data Registry

- One typed config object (`ExperimentConfig`) parameterizes everything; nothing else reads YAML. Schema in §2A; live examples in [configs/](configs/).
- Dataset: Numerai **v5.3** parquet assets under `data/v5.3/` (train/validation/live, meta_model, benchmark models, example preds, `features.json`). Asset inventory and download expectations live in [`README.md`](README.md#data-assets).
- Feature sets from `features.json`: `small` (benchmark tutorial convention: 42-column `feature_sets.small`), `medium`, `all`; every named set is resolvable via `nmr.features.resolve_feature_sets` (§P). `data.feature_subset` optionally overrides `feature_set` — schema in §2A, semantics in §P.
- Benchmark tiers: `configs/benchmarks/` — 8 tier YAMLs (`tier0_null` … `tier4_gate`) validated by `load_benchmark_suite_config` (§M); tier 1–3 `anchors` are report-only reference lines, tier-0/tier-4 thresholds are hard gates.

---

## 5. Tool & Function Registry

| Trigger | Module / entry | Execution target |
|---|---|---|
| Run an experiment | `nmr.runner.ExperimentRunner.run(deploy=...)` | Full pipeline §1 |
| Compare / promote across families | `nmr.registry.RunRegistry.list / best / promote / promote_if_better` | `experiments/*/runs/*/run.json` + `experiments/champion.json` |
| Score a prediction set | `nmr.scorecard.evaluate_model` | `MetricScorecard` |
| Benchmark everything | `python benchmark_runner.py [--fast-mode] --output artifacts/reports/benchmark_hierarchy_scorecard.csv` | `artifacts/reports/benchmark_hierarchy_scorecard.csv` + `benchmark_gate_report.csv` |
| First-model train + promote | `python train_first_model.py` | registry + champion |
| Leaderboard | `python generate_dashboard.py` | `artifacts/dashboard.html` |
| Build/validate a submission | `nmr.submission.build_submission / validate_submission / write_submission` | CSV in (0,1) |
| Package for hosted upload | `nmr.deployment.serialize_predict` | `predict.pkl` + manifest |
| HPO sweep | `nmr.research.HyperparameterSweep.run` | `SweepResult` |
| Neutralization tuning | `nmr.research.neutralization_frontier` | proportion → `MetricSummary` curve |
| Mutation gate | `scripts/mutation_gate.py` (CI-only; mutmut refuses native Windows) | `configs/mutation_receipt.json` |

---

## 6. Technical Debt & Known Gaps

- **`embargo_eras` is structurally inert** — see §C (schema) and [`AGENTS.md`](AGENTS.md#8-critical-operational-hazards) (agent directive).
- **Expression-level feature engineering is deferred:** feature-set resolution + stability screening are now supported (`nmr/features.py` §P); derived/expression-level transforms are still deferred — do not reference a FeatureFactory.
- **Benchmark train parquet early-era gap:** `train_benchmark_models.parquet` lacks rows for the first ~30 train eras (agent policy in [`AGENTS.md`](AGENTS.md#8-critical-operational-hazards)).
- **GPU/CPU numeric divergence:** determinism is guaranteed per device, not across GPU↔CPU fallback boundaries.
- **No packaging metadata:** the repo has no `pyproject.toml`; imports rely on the pytest setup documented in [`CONTRIBUTING.md`](CONTRIBUTING.md) (Critical footguns).
- **Timing instrumentation is hash-hazardous by construction:** every new scorecard field must be triaged into canonical-vs-excluded (see `canonical_scorecards_bytes`).
