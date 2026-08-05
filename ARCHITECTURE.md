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
+---------------------------------------------------------------+
| ExperimentRunner.run(deploy=False)          nmr/runner.py      |
|                                                                |
|  1. set_global_seeds(run.seed)      PYTHONHASHSEED/random/np   |
|  2. IngestionAgent.load("train")    lazy Polars, col pushdown  |
|  3. per target in data.targets:                                |
|       ModelOrchestrator.train_cross_validation()               |
|         └── PurgedEraSplitter folds (walk_forward, 8-era purge)|
|       → OOF pred_{target} columns joined on [id, era]          |
|  4. Ensembler.learn_weights(method="ridge")                    |
|     Ensembler.blend()               rank-domain, re-gaussianize|
|  5. NeutralizationEngine.neutralize(proportion=1.0)            |
|         └── per-era pinv cache: artifacts/cache/neutralization |
|  6. EvaluationEngine.per_era_corr() → summarize()              |
|  7. [deploy] anchor-fold model → serialize_predict()           |
|         └── artifacts/runs/{run_id}/predict.pkl (+manifest)    |
+---------------------------------------------------------------+
     |
     v  RunResult(run_id, oof, metrics, artifact, manifest)
RunRegistry.record() ──> artifacts/registry/{run_id}/{run.json, oof.parquet}
RunRegistry.promote() ─> artifacts/registry/champion.json   (atomic pointer)
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

Frozen dataclasses; `__post_init__` validates enums, non-negativity, and non-emptiness. `load_config(path)` parses YAML, rejects unknown keys/sections. `set_global_seeds(seed)` sets `PYTHONHASHSEED`, `random.seed`, `np.random.seed`.

| Section | Fields (defaults) | Valid values |
|---|---|---|
| `data: DataConfig` | `version="v5.2"`, `feature_set="small"`, `targets=("target",)`, `data_dir=REPO_ROOT/"data"` | feature_set ∈ `("small", "medium", "all")` |
| `split: SplitConfig` | `scheme="walk_forward"`, `purge_eras=8`, `embargo_eras=4`, `n_folds=4` | scheme ∈ `("walk_forward", "anchor")` |
| `model: ModelConfig` | `backend="lightgbm"`, `preset="fast"`, `params={}` | backend ∈ `("lightgbm", "xgboost")`, preset ∈ `("fast", "standard", "deep")` |
| `evaluation: EvalConfig` | `backend="custom"`, `main_target="target"`, `metrics=("corr","mmc","fnc","sharpe")` | backend ∈ `("custom", "official")` |
| `run: RunConfig` | `name="default"`, `seed=42`, `artifacts_dir=REPO_ROOT/"artifacts"` | — |

`REPO_ROOT = Path(__file__).resolve().parent.parent`. Relative paths resolve against it. See [configs/example.yaml](configs/example.yaml) for the annotated schema and [configs/first_model.yaml](configs/first_model.yaml) for the current competitive config (4×20D targets, ridge ensemble, full neutralization, seed 20260713).

### B. Data Layer — `nmr/data.py`

`IngestionAgent(data: DataConfig)` — construction-time inert (no I/O).

- `scan(split, *, subset=None, targets=None, columns=None) -> pl.LazyFrame` — lazy scan with column pushdown; `load()` collects; `train()/validation()/live()` shortcuts.
- Split files: `{"train": "train.parquet", "validation": "validation.parquet", "live": "live.parquet"}`; meta columns `("era", "id")`.
- `feature_sets` / `features(subset)` / `available_targets()` read `data/v5.2/features.json` (defensive copies).
- Deterministic column order: `era · id · features(subset) · targets`. Requested targets are validated against `features.json` then intersected with the physical schema. Schema reads are metadata-only and cached per split. Missing split file ⇒ `FileNotFoundError` on first access.

### C. Validation Splitting — `nmr/splitter.py`

`PurgedEraSplitter(split: SplitConfig).split(eras) -> list[Fold]`; `Fold(index, train_eras, val_eras)` frozen.

Geometry: eras deduped and sorted numerically (non-numeric ⇒ `ValueError`). With `n = era_count`, `k = n_folds`:
- `val_size = n // (k + 1)`; `prefix_size = n - k * val_size`; requires `prefix_size - purge_eras ≥ 1`.
- **walk_forward** fold *i*: val = `eras[prefix + i·val_size : prefix + (i+1)·val_size]`, train = `eras[: val_start - purge_eras]`.
- **anchor**: single fold with `k=1` geometry — one train prefix, one validation window.

Invariants validated on every fold: `max(train) < min(val)`, exactly `purge_eras` eras excluded between them, no era reuse, disjoint validation windows. `embargo_eras` is accepted but **structurally inert** (see [`AGENTS.md`](AGENTS.md#8-critical-operational-hazards)).

### D. Transforms — `nmr/_transforms.py`

Shared by evaluation, ensemble, and submission:

| Function | Definition |
|---|---|
| `tie_kept_rank(v)` | `(rankdata(v, method="average") − 0.5) / n` → [0, 1) |
| `gaussianize(v)` | `norm.ppf(v)` |
| `rank_gaussianize(v)` | `gaussianize(tie_kept_rank(v))` |
| `standardize_unit_variance(v)` | `v / std(v, ddof=0)`; zeros if std is 0/non-finite |
| `rank_gaussianize_unit_variance(v)` | composition of the above |
| `power_1_5(v)` | `sign(v) · |v|^1.5` |

### E. Evaluation Engine — `nmr/evaluation.py`

`EvaluationEngine(backend="custom")` — per-era metric dicts (`{era: score}`); `custom` = NumPy implementations below, `official` = `numerai_tools.scoring` delegation. `MIN_OVERLAP_ERAS = 20`; insufficient overlap raises `NonVacuityError(ValueError)`.

| Metric | Custom algorithm (per era) |
|---|---|
| **CORR** `per_era_corr` | `Pearson( power_1_5(rank_gaussianize(pred)), power_1_5(target − mean(target)) )` |
| **MMC** `per_era_mmc` | rank-gaussianize pred & meta; orthogonalize pred against meta (`p − m·(p@m / m@m)`); bucket targets in [0,1] rescaled to [0,4]; return `(target − mean) @ neutral_pred / n` |
| **FNC** `per_era_fnc` | least-squares residual of `rank_gaussianize(pred)` against `[features | intercept]`, std-normalized, then CORR vs target |
| **BMC** `per_era_bmc` | MMC-form vs a benchmark column (`min_overlap_eras=20`) |
| **CWMM** `per_era_cwmm` | MMC-form pred-vs-meta (no oracle counterpart) |

`summarize(per_era) -> MetricSummary(mean, std, sharpe, max_drawdown)` — std ddof=0, `sharpe = mean/std` (0 if std=0), drawdown on cumulative sum. Degenerate eras (<2 rows, zero variance, non-finite) short-circuit to score 0.0 after `_clean_frame()` null/finite filtering.

### F. Neutralization — `nmr/risk.py`

`NeutralizationEngine(cache_dir=REPO_ROOT/"artifacts"/"cache"/"neutralization")`.

`neutralize(df, *, pred_col, feature_cols, era_col="era", proportion=1.0)` — per era: design `[features | 1]` (intercept-aware), `coeffs = pinv(design, rcond=1e-6) @ pred`, output `pred − proportion · (design @ coeffs)`. `proportion ∈ [0, 1]` (0 = identity, 1 = full). All values must be finite (else `ValueError`).

Per-era pseudo-inverse cache: key = SHA256 of `{era, sorted feature_cols, row_count, row_ids_sha256, intercept: true}`; files `era_{label}_{key}.npy` + `.json` (metadata revalidated before loading).

### G. Modeling — `nmr/models.py`

`ModelOrchestrator(config: ModelConfig, *, seed=42)`:
- `train_cross_validation(df, *, feature_cols, target_col, splitter, era_col) -> CVResult(oof, models)` — per fold: fit on train eras, predict val eras, stack OOF.
- `train_anchor_fold(...) -> (model, val_predictions)` — single anchor fold, used for deployment.
- Fold leakage-safety re-asserted at train time (`_assert_fold_is_leakage_safe`).
- GPU-first (`device_type="gpu"` / `tree_method="gpu_hist"`) with automatic CPU fallback.

Canonical presets (`_CANONICAL_PRESETS`, mirroring Numerai's published benchmark params):

| Preset | n_estimators | lr | max_depth | num_leaves | colsample_bytree | extra |
|---|---|---|---|---|---|---|
| fast | 2 000 | 0.01 | 5 | 31 | 0.1 | — |
| standard | 20 000 | 0.001 | 6 | 64 | 0.1 | — |
| deep | 30 000 | 0.001 | 10 | 1 024 | 0.1 | `min_data_in_leaf=10000` |

LightGBM adds `objective="regression"`, `random_state=seed`, `n_jobs=1`, `deterministic=True`, `force_col_wise=True`. XGBoost translates `num_leaves→max_leaves`, `min_data_in_leaf→min_child_weight`, adds `reg:squarederror` + `seed`. `ModelConfig.params` overrides presets key-by-key.

### H. Ensembling — `nmr/ensemble.py`

`Ensembler`:
- `rank_normalize(df, *, pred_cols, era_col)` — per-era `rank_gaussianize_unit_variance` on each component.
- `blend(df, *, pred_cols, weights=None, out_col="prediction")` — rank-normalize → weighted dot product (default uniform 1/n) → `rank_gaussianize` the combination.
- `learn_weights(oof_df, *, pred_cols, target_col, method="ridge") -> tuple[float, ...]` — rank-normalized design matrix, finite-row masking, then:
  - **ridge** (α=1.0): `solve(XᵀX + I, Xᵀy)`
  - **nnls**: `scipy.optimize.nnls(X, y)`

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

### K. Scorecard — `nmr/scorecard.py`

`evaluate_model(predictions, *, meta_model, benchmarks, features, targets, n_trials, seed, horizon="20D", main_target="target", benchmark_col=None, regime_labels=None, perturbation=None, pf=1.0, clip=0.05, n_boot=1000, alpha=0.05, min_overlap_eras=20, model_id="model", ...) -> MetricScorecard`

Flow: join predictions ∩ meta ∩ targets ∩ features on `[era]` or `[era, id]` (optional left-join benchmarks) → per-era CORR/MMC/FNC (+BMC/CWMM when benchmark/meta available) → payout report → `MetricCell(value, ci_low, ci_high, n_eras)` bootstrap cells → feature-exposure, horizon-stability, regime, and perturbation diagnostics. Horizon targets inferred by regex `_([a-zA-Z0-9]+)(?:20|60)$` on `benchmark_col`, requiring both `target_{name}_20` and `target_{name}_60`.

`MetricScorecard` (frozen, 31 fields) includes `rank_scalar`, `deflated_sharpe`, `mean_payout/corr/mmc/corr_sharpe_ac/bmc/cwmm` cells, `fnc`, `cvar5`, `max_drawdown`, `burn_rate`, `sortino`, `calmar`, `max_feature_exposure`, robustness sub-results, and `metric_timing_seconds` + `eval_total_seconds` instrumentation. `to_frame()` flattens to a single-row Polars frame (cells expand to `{name}`, `{name}_ci_low`, `{name}_ci_high`, `{name}_n_eras`; timings become `timing_*` columns + `quality_metric_timings_json` / `quality_metric_total_seconds`). **Timing columns are excluded from canonical hashing** (§M).

### L. Research & Robustness — `nmr/research.py`, `nmr/robustness.py`

- `HyperparameterSweep(base_config, *, metric="sharpe").run(space, *, n_trials, seed) -> SweepResult(trials, best_params, best_value)` — Cartesian product of the space, shuffled, sampled to `n_trials`; each trial overrides `model.params` and evaluates on a purged held-out split (final ~20% of eras).
- `neutralization_frontier(oof, *, feature_cols, proportions, ...) -> NeutralizationFrontier(proportions, metrics)` — sweeps neutralization proportion, per-era CORR + `MetricSummary` at each point.
- `feature_exposure_report(oof, *, feature_cols, ...)` — per-feature mean/max absolute correlation with predictions.
- `adversarial_perturbation(...) -> PerturbationResult(alpha, n_eras, ceiling_stability, manifold_stability, gap, effective_perturb_frac)` — cell-level ±1 bin flips (features are Int8 ∈ [0,4]) plus circular block swaps from train; per-era Spearman stability of predictions.
- `time_horizon_stability(...) -> HorizonStabilityResult` — model vs benchmark AC-adjusted Sharpe on `_20` vs `_60` targets; decay and relative divergence.
- `regime_conditioned_corr(...) -> dict[str, RegimeCorr]` — per-regime per-era CORR with block-bootstrap CIs.

### M. Benchmark Harness — `nmr/benchmark.py`

`BenchmarkSuite` evaluates prediction sources through the same `evaluate_model` pipeline:

- **Null baselines** — `NULL_BASELINES = ("constant-0.5", "uniform-random", "gaussian-random")` (seeded RNG).
- **Classical baselines** (`run_classical_baselines`, `min_train_eras=10`, walk-forward era t−1 → t): trivial (row-mean of features), linear (`Ridge(alpha=1.0)`), tree (`LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=15, colsample_bytree=0.1, subsample=0.8)`).
- **Tutorial ingestion** — `TUTORIAL_NOTEBOOK_TO_MODEL_ID = {"1_hello_numerai.ipynb": "hello-numerai", "2_feature_neutralization.ipynb": "feature-neutralization", "example-model-sunshine.ipynb": "sunshine"}`; notebook anchor-string contract checks; `ingest_tutorial_prediction[_batch]` / `extract_oos_predictions` normalize arbitrary artifacts to `[era, id, prediction]`.
- **Gates** — `assert_null_floor(scorecards, tolerance=0.05)`: every null baseline must have |value| ≤ tolerance on rank_scalar, mean_payout, corr, mmc, fnc, corr_sharpe_ac (+bmc/cwmm if present). `assert_slice1_monotone`: null ≤ hello-numerai ≤ sunshine.
- **Determinism** — `canonical_scorecards_bytes()`: drops all timing columns, JSON with sorted keys, `separators=(",", ":")`, NaN/Inf as string sentinels, keyed by model_id; `scorecards_sha256()` digests it.
- **Output** — `scorecards_to_frame` / `write_scorecards_csv` (column inventory = `MetricScorecard.to_frame()` §K).

### N. Runner, Registry, Submission, Deployment

**`nmr/runner.py`** — stage order in §1 diagram. `RunResult(run_id, oof, metrics, artifact, manifest)`. `run_id` = SHA256 of `{config (data_dir/artifacts_dir stripped), data_version, code_fingerprint, environment_fingerprint}` where code fingerprint = SHA256 over sorted `nmr/*.py` names+contents and environment = Python + versions of numpy/polars/pandas/lightgbm/xgboost. `deploy=True` trains an anchor fold, wraps a `predict()` closure with ordered feature names, and writes `artifacts/runs/{run_id}/predict.pkl` + manifest.

**`nmr/registry.py`** — `RunRegistry(root)`:

```
artifacts/registry/
├── champion.json                 # {"run_id": "<sha256 hex>"}  (atomic pointer)
└── {run_id}/
    ├── run.json                  # {run_id, metrics{mean,std,sharpe,max_drawdown},
    │                             #  manifest, oof_path, artifact_path|null, artifact_manifest|null}
    └── oof.parquet
```

All JSON writes: temp file in parent dir → fsync → `os.replace()`. `record(result)` writes OOF then run.json; `best(metric="sharpe")` scans run.json files and returns the max; `promote(run_id)` validates existence then atomically rewrites champion.json.

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
| [train_first_model.py](train_first_model.py) | `load_config("configs/first_model.yaml")` → `ExperimentRunner.run(deploy=True)` → `RunRegistry.record` + `promote` → prints summary, writes `summary.json` |
| [benchmark_runner.py](benchmark_runner.py) | Flags: `--data-dir` (data/v5.2), `--output`, `--labels-output`, `--seed` (77), `--n-boot` (300), `--min-overlap-eras` (20), `--horizon` (20D/60D), `--min-train-eras` (10), `--log-level`, `--fast-mode` (n_boot=1, skips linear/tree). Loads validation/meta/benchmarks → `BenchmarkSuite` → scorecards CSV + per-era label profile CSV. Low-variance predictions (<1e-9) get a fallback scorecard. |
| [generate_dashboard.py](generate_dashboard.py) | Aggregates registry runs + benchmark CSV → Sharpe-ranked dark-theme leaderboard at `artifacts/dashboard.html` |

---

## 3. Module Dependency Graph

```
config.py        (leaf — no nmr imports)
_transforms.py   (leaf)

data.py      ──> config (DataConfig)
splitter.py  ──> config (SplitConfig)
risk.py      ──> config (REPO_ROOT)
models.py    ──> config (ModelConfig), splitter (Fold, PurgedEraSplitter)
evaluation.py──> _transforms (power_1_5, rank_gaussianize)
ensemble.py  ──> _transforms (rank_gaussianize, rank_gaussianize_unit_variance)
submission.py──> _transforms (tie_kept_rank), numerai_tools.submissions
inference.py (leaf — NumPy/SciPy only)
payout.py    ──> inference
research.py  ──> config, data, models, splitter, risk, evaluation
robustness.py──> inference, _transforms
scorecard.py ──> evaluation, inference, payout, research, robustness
benchmark.py ──> scorecard, evaluation, data
runner.py    ──> config, data, splitter, models, ensemble, risk, evaluation, deployment
registry.py  ──> runner (RunResult)
deployment.py (leaf — cloudpickle/stdlib)

nmr/__init__.py re-exports the public API of all modules (keep imports and __all__ in sync).
```

---

## 4. Configuration & Data Registry

- One typed config object (`ExperimentConfig`) parameterizes everything; nothing else reads YAML. Schema in §2A; live examples in [configs/](configs/).
- Dataset: Numerai **v5.2** parquet assets under `data/v5.2/` (train/validation/live, meta_model, benchmark models, example preds, `features.json`). Asset inventory and download expectations live in [`README.md`](README.md#data-assets).
- Feature sets from `features.json`: `small` (benchmark tutorial convention: 42-column `feature_sets.small`), `medium`, `all`.

---

## 5. Tool & Function Registry

| Trigger | Module / entry | Execution target |
|---|---|---|
| Run an experiment | `nmr.runner.ExperimentRunner.run(deploy=...)` | Full pipeline §1 |
| Record / promote a run | `nmr.registry.RunRegistry.record / best / promote` | `artifacts/registry/` |
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

- **`embargo_eras` is inert:** validated, stored, and unused by fold geometry. Reserved for future two-sided schemes.
- **`features.py` (slice 2b) does not exist:** feature engineering beyond `features.json` subsets is deferred; do not reference a FeatureFactory.
- **Benchmark train parquet early-era gap:** `train_benchmark_models.parquet` lacks rows for the first ~30 train eras (agent policy in [`AGENTS.md`](AGENTS.md#8-critical-operational-hazards)).
- **GPU/CPU numeric divergence:** determinism is guaranteed per device, not across GPU↔CPU fallback boundaries.
- **No packaging metadata:** the repo has no `pyproject.toml`; imports rely on `pythonpath = .` in [pytest.ini](pytest.ini) and running from the repo root.
- **Timing instrumentation is hash-hazardous by construction:** every new scorecard field must be triaged into canonical-vs-excluded (see `canonical_scorecards_bytes`).
