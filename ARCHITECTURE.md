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
|         └── artifacts/runs/{run_id}/predict.pkl (+manifest)    |
+----------------------------------------------------------------+
     |
     v  RunResult(run_id, oof, metrics, artifact, manifest)
RunRegistry.record() ──> artifacts/registry/{run_id}/{run.json, oof.parquet, validation_preds.parquet?}
RunRegistry.promote() ─> artifacts/registry/champion.json   (atomic pointer)
RunRegistry.promote_if_better() ─> champion.json  (guarded: scorecard metric + direction)
     |
     v
generate_dashboard.py ─> artifacts/dashboard.html  (ranked leaderboard)

Parallel harness:
benchmark_runner.py ──> BenchmarkSuite (nmr/benchmark.py)
   null baselines + classical baselines + Numerai benchmark cols
   ──> evaluate_model() scorecards ──> artifacts/benchmark_scores.csv
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
| `data: DataConfig` | `version="v5.3"`, `feature_set="small"`, `feature_subset=None`, `supplemental_feature_sets=None`, `targets=("target",)`, `data_dir=REPO_ROOT/"data"` | feature_set ∈ `("small", "medium", "all")`; feature_subset: any `features.json` set name or `None` (validated at ingestion, §P); supplemental_feature_sets: JSON path merged into the set registry (collision ⇒ `ValueError`, §P) |
| `split: SplitConfig` | `scheme="walk_forward"`, `purge_eras=8`, `embargo_eras=4`, `n_folds=4` | scheme ∈ `("walk_forward", "anchor")` |
| `model: ModelConfig` | `backend="lightgbm"`, `preset="fast"`, `params={}` | backend ∈ `("lightgbm", "xgboost", "catboost")`, preset ∈ `("fast", "standard", "deep")` |
| `evaluation: EvalConfig` | `backend="custom"`, `main_target="target"`, `metrics=("corr","mmc","fnc","sharpe")`, `validation_scorecard=True` | backend ∈ `("custom", "official")` |
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

Invariants validated on every fold: `max(train) < min(val)`, exactly `purge_eras` eras excluded between them, no era reuse, disjoint validation windows. `embargo_eras` is accepted but **structurally inert** (see [`AGENTS.md`](AGENTS.md#8-critical-operational-hazards)). Purge/embargo convention (8/16 operational vs 4/16 minimum): [docs/DOCS_README.md](docs/DOCS_README.md) §3; official benchmark walk-forward table (156-era blocks): [docs/01-canon/models.md](docs/01-canon/models.md).

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
- `train_full_history(df, *, feature_cols, target_col, era_col="era") -> model` — one CPU-only model fit on every era, with null/non-finite targets dropped (logged count; `ValueError` if nothing remains). Used by the deployment pipeline so the artifact reproduces identically on any hosted runtime.
- **Memory discipline (recorded 2026-08-10):** `coerce_float32_features(df, feature_cols)` casts exactly-representable schemas (Int*/UInt*/Float32 only — the v5.x integer bins) to a single Float32 polars block; `_feature_frame` returns its **zero-copy numpy view** (polars→numpy verified 0-copy; pandas is skipped — polars→pandas goes through pyarrow and allocates a second full copy, ~36 GiB at 3,555 × 2.1M, the `lgbm_v1` full-history OOM). Float64/mixed schemas pass through untouched. Fold and full-history predicts run in era-batches (`_predict_model_chunked`, 20 eras/chunk) — a full fold-val matrix at 3,555 features (~4.9 GiB float32) exceeds the 4 GiB GPU VRAM (`xgb_v1` CUDA OOM). The **validation stage** predicts in era-batches (`runner._predict_in_era_batches`, 40 eras/chunk) so the deploy closure's float64 `to_numpy` stays ≤ ~1.7 GiB. All paths are bit-identical to the pre-optimization code (Int8 bins exact in float32; per-era closure ops order-independent) — determinism tests cover them.
- Fold leakage-safety re-asserted at train time (`_assert_fold_is_leakage_safe(fold, purge_eras=...)`): before fitting each fold it enforces no era reuse, non-empty sides, strict time-ordering, and `min(val) − max(train) > purge_eras` (gap ≤ `purge_eras` ⇒ `ValueError`).
- `_fit_predict_fold` drops null/non-finite target rows from the train slice before fitting (logged dropped count; `ValueError` if nothing remains).
- OOF-CV is GPU-first for lightgbm/xgboost with automatic CPU fallback: LightGBM via `device_type="gpu"`, XGBoost (>= 3.0) via `device="cuda"` + `tree_method="hist"` (`gpu_hist` was removed in 3.x and raises `Invalid Input`). A failed device attempt is logged with `type(exc).__name__` + message (only backend errors — `ValueError`/`TypeError`/`LightGBMError`/`XGBoostError`/`CatBoostError` — trigger fallback; anything else fails loudly), and `resolved_device` records which device actually fit (`"gpu"`/`"cpu"`, `None` before the first successful fit). `model.device` (`auto` | `gpu` | `cpu`, default `auto`) controls CV/experimentation: `gpu` returns only the GPU candidate (a failure raises — no silent fallback), `cpu` never attempts GPU. The run manifest records the config device as `pipeline_device` and the actual fit device as `oof_device`. CatBoost is CPU-only by design — it never attempts a GPU candidate (see below). `train_full_history` is CPU-only by design.

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

Capital-readiness extensions (v2.5, pure NumPy, float64 throughout): `annual_compounded_return(clipped, *, eras_per_year=52.0)` — `(∏(1 + r_t))^(52/n) − 1`, `−1.0` when the wealth product ≤ 0 (ruin), `0.0` when `n < 2`; `gain_to_pain_ratio(clipped)` — `Σ max(0, r_t) / Σ |min(0, r_t)|`, `+inf` on a zero-pain positive series (precedented by `calmar`; the canonical JSON sanitizer `_sanitize_json_payload` maps non-finite floats to `"Infinity"`/`"-Infinity"`/`"NaN"` strings in canonical bytes while parquet/CSV carry `inf` natively), `0.0` on an all-flat series; `kelly_fraction(raw)` — `min(1.0, max(0.0, μ / σ²))` with `σ² = var(ddof=0)` computed on the **raw** (unclipped) series: the clipped series has Popoviciu-bounded variance (≤ 0.0025 under the ±5% clip) so μ/σ² there saturates at 1.0 for every viable model and carries no discrimination; `simulate_overlapping_portfolio(clipped, *, horizon_eras=20, initial_capital=1.0, eras_per_year=52.0) -> OverlappingSimulationResult` — multi-round lockup simulator: at each era, tranches maturing at `t` pay `principal·(1 + r_{t−K})` (the initiating era's return — Numerai round semantics), then `min(cash, total_equity/K)` deploys as a new tranche maturing at `t + K`; equity and utilization are recorded **before** the era's deployment; tranches still locked at series end are carried at par principal (at-cost convention — no mark-to-market, no unrealized payoff); `n < horizon_eras` returns a zeroed result. Horizon eras derive from the report's `horizon` argument via `_HORIZON_ERAS = {"20D": 20, "60D": 60}`. `PayoutResult` gains `cagr_1y`, `gain_to_pain_ratio`, `kelly_fraction` (CAGR/GPR on `series.clipped`, Kelly on `series.raw`) and `overlapping_sim: OverlappingSimulationResult | None = None` (default preserves existing direct constructions).

Payout weights, ±5% clip, and stake thresholds follow [docs/01-canon/staking.md](docs/01-canon/staking.md).

### K. Scorecard — `nmr/scorecard.py`

`evaluate_model(predictions, *, meta_model, benchmarks, features, targets, n_trials, seed, horizon="20D", main_target="target", benchmark_col=None, backend="custom", regime_labels=None, perturbation=None, pf=1.0, clip=0.05, n_boot=1000, alpha=0.05, min_overlap_eras=20, model_id="model", ...) -> MetricScorecard`

Flow: join predictions ∩ meta ∩ targets ∩ features on `[era]` or `[era, id]` (optional left-join benchmarks) → per-era CORR/MMC/FNC (+BMC/CWMM when benchmark/meta available) → payout report → `MetricCell(value, ci_low, ci_high, n_eras)` bootstrap cells → feature-exposure, horizon-stability, regime, and perturbation diagnostics. Horizon targets inferred by regex `_([a-zA-Z0-9]+)(?:20|60)$` on `benchmark_col`, requiring both `target_{name}_20` and `target_{name}_60`.

`MetricScorecard` (frozen, 43 fields) includes `rank_scalar`, `deflated_sharpe`, `mean_payout/corr/mmc/corr_sharpe_ac/bmc/cwmm` cells, `fnc`, `cvar5`, `max_drawdown`, `burn_rate`, `sortino`, `calmar`, `max_feature_exposure`, robustness sub-results, the capital-readiness block (v2.5), and `metric_timing_seconds` + `eval_total_seconds` instrumentation. Capital-readiness fields: `cagr_1y`, `gain_to_pain_ratio`, `kelly_fraction`, `sim_portfolio_cagr`, `sim_portfolio_mdd`, `sim_capital_utilization` (floats, from `PayoutResult`/`OverlappingSimulationResult`); `mmc_down` (mean MMC over the eras where the meta model's per-era CORR < 0 — `None` + `mmc_down_reason="insufficient_downside_eras"` when fewer than `_MMC_DOWN_MIN_ERAS` (5) such eras; `mmc_down_n_eras` always records the count, `mmc_down_reason` is `None` otherwise); `turnover_mean`/`turnover_std` (mean and population std ddof=0 of `1 − ρ_k` transitions — both `None` when the join lacks the id column (`turnover_reason="id column unavailable"`) or fewer than 2 valid transitions exist (`"insufficient_transitions"`), `None` reason otherwise). `to_frame()` flattens to a single-row Polars frame (cells expand to `{name}`, `{name}_ci_low`, `{name}_ci_high`, `{name}_n_eras`; timings become `timing_*` columns + `quality_metric_timings_json` / `quality_metric_total_seconds`). **Timing columns are excluded from canonical hashing** (§M).

The evaluation spec of record (metrics, gates, build slices E1–E6) is [docs/06-evaluation/evaluation-suite-bible.md](docs/06-evaluation/evaluation-suite-bible.md).

### L. Research & Robustness — `nmr/research.py`, `nmr/robustness.py`

- `HyperparameterSweep(base_config, *, metric="sharpe").run(space, *, n_trials, seed) -> SweepResult(trials, best_params, best_value)` — Cartesian product of the space, shuffled, sampled to `n_trials`; each trial overrides `model.params` and evaluates on a purged held-out split (final ~20% of eras).
- `_held_out_metric(config, *, metric_name) -> float` — the shared held-out evaluation behind `HyperparameterSweep` and `bayesian_sweep` (§S): trains multi-target OOF on the train partition, learns ensemble weights, anchor-fits on train, then blends + neutralizes the held-out partition and computes per-era CORR. Metric set: `mean`/`std`/`sharpe`/`max_drawdown` via `MetricSummary` attributes, plus `corr_sharpe_ac` via `_per_era_ac_sharpe(per_era, horizon="20D")` (metric-resolution contract: §S); unknown names raise `ValueError`.
- `neutralization_frontier(oof, *, feature_cols, proportions, ...) -> NeutralizationFrontier(proportions, metrics)` — sweeps neutralization proportion, per-era CORR + `MetricSummary` at each point.
- `feature_exposure_report(oof, *, feature_cols, ...)` — per-feature mean/max absolute **Pearson correlation** with predictions, vectorized via one `partition_by(era)` pass + per-era matrix op (`_pred_feature_pearson`). Definition change dated 2026-08-05 (previously power-1.5 Numerai CORR per feature) — recorded exposure numbers are **not comparable** across that boundary.
- `adversarial_perturbation(...) -> PerturbationResult(alpha, n_eras, ceiling_stability, manifold_stability, gap, effective_perturb_frac)` — cell-level ±1 bin flips (features are Int8 ∈ [0,4]) plus circular block swaps from train; per-era Spearman stability of predictions.
- `time_horizon_stability(...) -> HorizonStabilityResult` — model vs benchmark AC-adjusted Sharpe on `_20` vs `_60` targets; decay and relative divergence.
- `regime_conditioned_corr(...) -> dict[str, RegimeCorr]` — per-regime per-era CORR with block-bootstrap CIs.

### M. Benchmark Harness — `nmr/benchmark.py`

`BenchmarkSuite` evaluates prediction sources through the same `evaluate_model` pipeline:

- **Null baselines** — `NULL_BASELINES = ("constant-0.5", "uniform-random", "gaussian-random")` (seeded RNG).
- **Classical baselines** (`run_classical_baselines`, `min_train_eras=10`, walk-forward era t−1 → t): trivial (row-mean of features), linear (`Ridge(alpha=1.0)`), tree (`LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=15, colsample_bytree=0.1, subsample=0.8)`, lightgbm-only — a missing lightgbm propagates `ImportError`, no sklearn fallback). Walk-forward iteration logs INFO progress per era (`[walk_forward] ...`).
- **Tutorial ingestion** — `TUTORIAL_NOTEBOOK_TO_MODEL_ID = {"1_hello_numerai.ipynb": "hello-numerai", "2_feature_neutralization.ipynb": "feature-neutralization", "example-model-sunshine.ipynb": "sunshine"}`; notebook anchor-string contract checks; `ingest_tutorial_prediction[_batch]` / `extract_oos_predictions` normalize arbitrary artifacts to `[era, id, prediction]`. Id-column inference falls back to the first non-metric column with a WARNING log (`[tutorial] ...`).
- **Gates** — `assert_null_floor(scorecards, tolerance=0.05)`: every null baseline must have |value| ≤ tolerance on rank_scalar, mean_payout, corr, mmc, fnc, corr_sharpe_ac (+bmc/cwmm if present). `assert_slice1_monotone`: null ≤ hello-numerai ≤ sunshine.
- **Determinism** — `canonical_scorecards_bytes()`: drops all timing columns, JSON with sorted keys, `separators=(",", ":")`, NaN/Inf as string sentinels, keyed by model_id; `scorecards_sha256()` digests it.
- **Output** — `scorecards_to_frame` / `write_scorecards_csv` (column inventory = `MetricScorecard.to_frame()` §K).

Benchmark ladder rationale (null floor, S11 rungs, hard gates): [docs/06-evaluation/benchmark-line-in-the-sand.md](docs/06-evaluation/benchmark-line-in-the-sand.md).

### N. Runner, Registry, Submission, Deployment

**`nmr/runner.py`** — stage order in §1 diagram. `RunResult(run_id, oof, metrics, artifact, manifest, scorecard=None, validation_predictions=None)`. `run_id` = SHA256 of `{config (data_dir/artifacts_dir stripped), data_version, code_fingerprint, environment_fingerprint}` where code fingerprint = SHA256 over sorted `nmr/*.py` names+contents and environment = Python + versions of numpy/polars/pandas/lightgbm/xgboost (plus `catboost` when `model.backend == "catboost"` — config-aware, §G/§S). Ensemble weights are learned on the validation eras of folds `0..K-2` via `EnsembleConfig.method`; when `n_folds < 2` they fall back to uniform `1/n_components` with a logged warning. OOF metrics are computed on the **final fold's** validation eras only (`scoring_eras`), so the OOF scorecard carries no in-sample weight-fitting bias; the returned OOF frame itself still spans every fold. The manifest records `weights`, `weight_learning_eras`, `scoring_eras`, and `summary_metrics` (OOF aggregates for each requested non-MMC metric). The deploy pipeline is built **at most once** per run, when `deploy or evaluation.validation_scorecard` (`_build_deploy_pipeline`: per-target all-eras CPU-only models + rank-gaussianize + learned weights + neutralize; no `splitter`), and that single closure is shared by the validation stage and the deploy block — never retrained. The **validation stage** (`_run_validation_stage`) loads `validation.parquet` plus `meta_model.parquet` (required — missing ⇒ `FileNotFoundError`) and `validation_benchmark_models.parquet` (optional — BMC/horizon disabled when absent), drops the first `split.purge_eras` validation eras (20D-target overlap), scores the shared pipeline, and produces a full `MetricScorecard` with `benchmark_col` = first non-join benchmark column (same convention as `benchmark_runner`); the run manifest records `validation_purge_dropped_first_eras`. Then `_serialize_predict_artifact(predict_fn, model_meta, artifact_path)` serializes (never retrains) to `artifacts/runs/{run_id}/predict.pkl` + manifest. The artifact's `models` metadata carries `targets`/`weights`/`proportion`/`geometry="all_eras"`/`device="cpu"`/`feature_names`; the run manifest adds `pipeline_device="cpu"`.

**`nmr/registry.py`** — `RunRegistry(root)`:

```
artifacts/registry/
├── champion.json                 # {"run_id": "<sha256 hex>"}  (atomic pointer)
└── {run_id}/
    ├── run.json                  # {run_id, metrics{mean,std,sharpe,max_drawdown},
    │                             #  manifest, scorecard{flat scalars}|null, oof_path,
    │                             #  artifact_path|null, artifact_manifest|null}
    ├── oof.parquet
    └── validation_preds.parquet  # [era, id, prediction] on validation eras,
                                  # only when evaluation.validation_scorecard
```

All JSON writes: temp file in parent dir → fsync → `os.replace()`; the OOF parquet likewise writes temp + `os.replace` (no fsync). `record(result)` writes OOF then run.json; when `result.scorecard` is present, run.json carries a `scorecard` block of flat scalar keys per `MetricScorecard.to_frame()` (values filtered to drop `timing_*`/`quality_metric*` keys). `best(metric="sharpe")` validates the metric against `MetricSummary` fields (`mean`, `std`, `sharpe`, `max_drawdown`; unknown ⇒ `ValueError`) and returns the max with a run_id tiebreak (deterministic); `list()` sorts by (mtime, run_id) — stable. `promote(run_id)` regex-validates `[0-9a-f]{64}` (path-traversal guard), validates existence, then atomically rewrites champion.json. `promote_if_better(run_id, metric="corr_sharpe_ac") -> tuple[Path, bool]` promotes only when the candidate's scorecard metric strictly beats the champion's, honoring direction (`_SCORECARD_METRIC_DIRECTION`: `max_drawdown`/`std_corr` are lower-is-better); a scorecard-bearing candidate may displace a scorecard-less (legacy) champion; legacy candidates (no scorecard) and unknown metrics raise `ValueError`; a missing/corrupted champion pointer is treated as no champion.

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
| [benchmark_runner.py](benchmark_runner.py) | Flags: `--data-dir` (data/v5.3), `--output`, `--labels-output`, `--seed` (77), `--n-boot` (300), `--min-overlap-eras` (20), `--horizon` (20D/60D), `--min-train-eras` (10), `--log-level`, `--fast-mode` (n_boot=1, skips linear/tree). Loads validation/meta/benchmarks → `BenchmarkSuite` → scorecards CSV + per-era label profile CSV. Low-variance predictions (<1e-9) get a fallback scorecard. |
| [generate_dashboard.py](generate_dashboard.py) | Aggregates registry runs + benchmark CSV → Sharpe-ranked dark-theme leaderboard at `artifacts/dashboard.html` |
| [dashboard_app.py](dashboard_app.py) | Streamlit+Plotly interactive dashboard over registry scorecards, benchmark CSV, campaign logs, and `fleet_summary` (§Q); read-only; launch: `streamlit run dashboard_app.py`. Pure shaping helpers are unit-tested (tests/test_scripts.py). |
| [run_campaign.py](run_campaign.py) | Run a named batch of configs and record trial lineage (see §R) |
| [analyze_dataset.py](analyze_dataset.py) | Modular dataset analysis: 17 named stages (`overview`, `targets`, `ic_by_era`, `screens`, `screens_train`, `summary`, `psi`, `drift`, `derived_sets`, `corr_medium`, `corr_all`, `set_membership`, `ic_by_split`, `regimes`, `benchmarks`, `meta_ortho`, `manifest`). Flags: `--only a,b` / `--skip a,b` run a subset (dependencies auto-included; `manifest` always runs), `--features small\|medium\|all`, `--max-eras N`, `--full-all-matrix`. `screens` writes the **descriptive full-span** screen (`feature_ic_screen.parquet`, eras 0001..1231 — never an input to subset derivation); `screens_train` writes the **train-only** screen (`feature_ic_screen_train.parquet`, eras 0001..0574); `derived_sets` reads **only** the train-only screen and writes `derived_feature_sets.json` (`screen_stable`, `screen_nonlinear`, `screen_linear_or_nonlinear`, `screen_drift_filtered` — pure functions of the train-only screen + drift dumps, sorted; see §P); `drift` writes the PSI + W1 + adversarial-AUC profile (`feature_drift_profile.parquet`, `w1_norm = w1 / σ_train`); `meta_ortho` writes per-feature meta-model orthogonality; the FNE profile uses an 11-point neutralization grid. Stage boundaries and per-era ticks print progress to stdout/stderr (never into artifacts); the manifest records `stages_run` + a machine-hardware summary (informational — never hashed). |
| [hardware_status.py](hardware_status.py) | Print machine specs + live resource status (`--record` writes `artifacts/reports/hardware_specs.json`); all logic in `nmr/hardware.py` (stdlib only) |

### P. Feature-Set Resolution & Stability Screening — `nmr/features.py`

Pure functions over `features.json` and the train frame; no model logic and no file state beyond the explicit `features_json` argument. Subset derivation adds no inputs to the `run_id` fingerprint beyond the config itself: the fingerprint is fully determined by config (including `data.feature_subset`) + data_version + `nmr/*.py` + environment.

`resolve_feature_sets(features_json: Path) -> dict[str, list[str]]` — returns every named set in `features.json` (`feature_sets` must be a non-empty mapping whose values are lists of strings, else `ValueError`), deterministically ordered by set name; values are defensive copies.

`feature_stability_screen(frame, *, feature_cols, target_col, era_col="era", min_mean_corr=DEFAULT_MIN_MEAN_CORR, max_abs_decay=DEFAULT_MAX_ABS_DECAY) -> pl.DataFrame` — per-feature era-window statistics:

- Per era, Pearson `CORR(feature, target)` computed in one vectorized per-era pass over `[era, target, *feature_cols]` (same pattern as `feature_exposure_report`, §L). Degenerate eras — fewer than 2 usable rows, an all-non-finite target, or a constant target — are excluded from every aggregate: `n_eras` counts valid eras only, and when no valid eras exist the numeric aggregates are `None` with `stable = False`. (Zero-IC padding of degenerate eras would bias the mean, std, and decay slope — label-lag eras carry no signal information, not zero signal.)
- Aggregates across eras (`_SCREEN_COLUMNS` = `feature, mean_corr, corr_std, decay_slope, cross_regime_variance, n_eras, stable`): `mean_corr` (mean), `corr_std` (population std, ddof=0), `decay_slope` (linear slope of CORR vs numeric era index via `np.polyfit(..., 1)`; `0.0` when fewer than 2 eras), `cross_regime_variance` (`0.25 · (first − second)²` — the variance of the first-half vs second-half era-window mean CORR, a regime-drift proxy), `n_eras`.
- `stable` predicate: `mean_corr ≥ min_mean_corr` AND `|decay_slope| ≤ max_abs_decay` AND `n_eras ≥ 2`. Defaults: `DEFAULT_MIN_MEAN_CORR = 0.01`, `DEFAULT_MAX_ABS_DECAY = 0.001`.

`select_stable_features(screen, *, min_mean_corr, max_abs_decay) -> list[str]` — filters the screen frame on `mean_corr ≥ min_mean_corr`, `|decay_slope| ≤ max_abs_decay`, `n_eras ≥ 2` and returns the passing feature names sorted.

`DataConfig.feature_subset` (`str | None`, default `None`) — optional override naming any feature set in `features.json`. Config load rejects empty-string values; the name itself is validated against `features.json` at ingestion time (`IngestionAgent.features` — fail loud, fail late). The `resolved_feature_set` property returns `feature_subset` when set, else `feature_set`; the runner and `HyperparameterSweep` resolve features through it (see also §2A, §B).

`DataConfig.supplemental_feature_sets` (`Path | None`, default `None`) — optional JSON file (`{"feature_sets": {...}}`) whose sets are **merged** into the `features.json` sets by `IngestionAgent` (used by the `feature_sets` property and `features()`). Merge is a pure function of the two files; a missing file, a malformed payload, non-string set values, or a **key collision** with `features.json` raises `ValueError` (fail loud — supplemental keys can never silently shadow packaged sets). Screen-derived campaign subsets (`derived_feature_sets.json` from the `derived_sets` analysis stage) are consumed through this key. The `run_id` fingerprint covers the config (including this path), so a changed derived-set file changes run identity by design.

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

Measured on the dev box (RTX A1000 Laptop, 4 GiB VRAM, driver 580.97; recorded 2026-08-09): xgboost `device="cuda"` trains **9.1×** faster than CPU (`hist`, `n_jobs=1`) on 300k×780, 300 trees — 13.5 s vs 123.3 s — and the full 3,555-feature universe fits the 4 GiB device; cupy `rankdata` is **5.8×** faster than scipy at 3555×7000 (0.40 s vs 2.33 s). scipy 1.17 `rankdata` returns an all-NaN array when any input is NaN (`nan_policy='propagate'`); `_gpu.rankdata` isolates NaN at the NaN positions instead (intentionally more correct; v5.3 features contain no NaN, so both paths agree on real data).

### V. Training & Analysis Progress Markers

Long-running paths print console progress that never enters artifacts: `analyze_dataset.py` logs `[stage i/n] name ... done (Xs)` per stage and `[label] era k/N` ticks per 100 eras (stderr); `ModelOrchestrator` prints `[fit] lightgbm iteration N` every `_FIT_PROGRESS_PERIOD` (100) iterations (CatBoost: period `verbose`; xgboost 3.x's sklearn wrapper has no callback hook — start/elapsed markers only); `benchmark_runner.py` logs per-strategy start/memory/elapsed. Progress is wall-clock output only — excluded from all canonical hashes by construction (it never reaches artifacts).

---

## 3. Module Dependency Graph

```
config.py        (leaf — no nmr imports)
_transforms.py   (leaf)
features.py      (leaf — stdlib/NumPy/Polars only)
refresh.py       (leaf — stdlib only; pure refresh policy, no I/O/numerapi)

data.py      ──> config (DataConfig)
splitter.py  ──> config (SplitConfig)
risk.py      ──> config (REPO_ROOT)
models.py    ──> config (ModelConfig), splitter (Fold, PurgedEraSplitter)
evaluation.py──> _transforms (power_1_5, rank_gaussianize)
ensemble.py  ──> _transforms (rank_gaussianize, rank_gaussianize_unit_variance)
submission.py──> _transforms (tie_kept_rank), numerai_tools.submissions
inference.py (leaf — NumPy/SciPy only)
meta.py      ──> evaluation, inference
payout.py    ──> inference
research.py  ──> config, data, models, splitter, risk, evaluation
opt.py       ──> config (ExperimentConfig), models (resolve_model_params), research (_held_out_metric, _override_config, SweepResult)
robustness.py──> inference, _transforms
scorecard.py ──> evaluation, inference, payout, research, robustness
benchmark.py ──> scorecard, evaluation, data
runner.py    ──> config, data, splitter, models, ensemble, risk, evaluation, deployment
registry.py  ──> runner (RunResult)
campaign.py  ──> _atomicio
deployment.py (leaf — cloudpickle/stdlib)

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

---

## 5. Tool & Function Registry

| Trigger | Module / entry | Execution target |
|---|---|---|
| Run an experiment | `nmr.runner.ExperimentRunner.run(deploy=...)` | Full pipeline §1 |
| Record / promote a run | `nmr.registry.RunRegistry.record / best / promote / promote_if_better` | `artifacts/registry/` |
| Score a prediction set | `nmr.scorecard.evaluate_model` | `MetricScorecard` |
| Benchmark everything | `python benchmark_runner.py [--fast-mode]` | `artifacts/benchmark_scores.csv` |
| First-model train + promote | `python train_first_model.py` | registry + champion |
| Leaderboard | `python generate_dashboard.py` | `artifacts/dashboard.html` |
| Build/validate a submission | `nmr.submission.build_submission / validate_submission / write_submission` | CSV in (0,1) |
| Package for hosted upload | `nmr.deployment.serialize_predict` | `predict.pkl` + manifest |
| HPO sweep | `nmr.research.HyperparameterSweep.run` | `SweepResult` |
| Neutralization tuning | `nmr.research.neutralization_frontier` | proportion → `MetricSummary` curve |

---

## 6. Technical Debt & Known Gaps

- **`embargo_eras` is structurally inert** — see §C (schema) and [`AGENTS.md`](AGENTS.md#8-critical-operational-hazards) (agent directive).
- **Expression-level feature engineering is deferred:** feature-set resolution + stability screening are now supported (`nmr/features.py` §P); derived/expression-level transforms are still deferred — do not reference a FeatureFactory.
- **Benchmark train parquet early-era gap:** `train_benchmark_models.parquet` lacks rows for the first ~30 train eras (agent policy in [`AGENTS.md`](AGENTS.md#8-critical-operational-hazards)).
- **GPU/CPU numeric divergence:** determinism is guaranteed per device, not across GPU↔CPU fallback boundaries.
- **No packaging metadata:** the repo has no `pyproject.toml`; imports rely on the pytest setup documented in [`CONTRIBUTING.md`](CONTRIBUTING.md) (Critical footguns).
- **Timing instrumentation is hash-hazardous by construction:** every new scorecard field must be triaged into canonical-vs-excluded (see `canonical_scorecards_bytes`).
