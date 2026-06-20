# NumerAI Refactored Pipeline

This repository contains a completed six-slice research and deployment pipeline for NumerAI v5.2 built around a fast custom execution path and an official oracle parity path.

The core design rule is simple:

- Use custom infrastructure for performance, memory efficiency, and deployment control.
- Use official Numerai tooling as the scoring oracle.
- Do not trust custom math unless parity tests prove it.

## Current Status

The infrastructure phase is complete.

Implemented slices:

1. Repository structure, lazy ingestion, and notebook harness
2. Era-level neutralization caching
3. Era-purged validation splitting and feature factory base
4. GPU-aware anchor and cross-validation model orchestration
5. Dual-backend evaluation with official oracle parity
6. Deployment stress testing, provenance-aware serialization, and deterministic promotion runner

Current validation status at the time of this handoff:

```text
43 passed in 3.71s
```

## Repository Layout

Top-level directories and their roles:

- `data/` - local NumerAI data assets
- `artifacts/` - cache, deployment bundles, and promotion outputs
- `notebooks/` - researcher-facing notebook control plane
- `src/` - tested system boundary
- `tests/` - unit, parity, deployment, and runner verification

Important source modules:

- `src/data.py` - lazy Polars ingestion via `IngestionAgent`
- `src/risk.py` - intercept-aware neutralization and cached pseudo-inverses
- `src/features.py` - `PurgedEraSplitter` and `FeatureFactory`
- `src/models.py` - `ModelOrchestrator` for anchor and full-CV modes
- `src/evaluation.py` - dual-backend evaluation engine (`custom` and `official`)
- `src/deployment.py` - stress testing, artifact serialization, hash verification, live prediction
- `src/runner.py` - deterministic promotion DAG
- `src/notebook_utils.py` - notebook presentation layer

Important tests:

- `tests/test_data.py`
- `tests/test_risk.py`
- `tests/test_features.py`
- `tests/test_models.py`
- `tests/test_evaluation.py`
- `tests/test_deployment.py`
- `tests/test_parity.py`
- `tests/test_runner.py`

## Design Boundaries

### What is custom on purpose

These modules exist because the official packages do not solve the systems problem:

- `src/data.py`
- `src/features.py`
- `src/risk.py`
- `src/models.py`
- `src/deployment.py`
- `src/runner.py`

Reasons:

- Polars-first lazy parquet access
- schema memoization
- era-purged CV
- era-level neutralization matrix caching
- GPU-aware research loops
- immutable deployment artifacts
- deterministic promotion workflow

### What is tied to the official oracle

Canonical scoring math must remain tethered to `numerai_tools`.

The evaluation layer uses two backends:

- `backend="custom"` for fast NumPy/Polars research loops
- `backend="official"` for audit and CI parity against `numerai_tools.scoring`

The parity rule is non-negotiable:

- if parity fails, the custom implementation is suspect
- if parity passes, the custom implementation is justified for performance use

## Cold-Start Architecture Map

### 1. Ingestion

`IngestionAgent` in `src/data.py`:

- loads `features.json`
- exposes NumerAI feature subsets dynamically
- lazily scans parquet files with Polars
- memoizes dataset schemas lazily on first access
- supports metadata, target, and subset-aware column selection

Key principle:

- construction must not depend on all parquet files existing
- schema access is cached lazily, not eagerly preloaded

### 2. Neutralization

`NeutralizationEngine` in `src/risk.py`:

- builds intercept-aware pseudo-inverses
- caches era-level neutralization artifacts under `artifacts/cache/neutralization/`
- stores uncompressed `.npy` matrices plus JSON metadata
- validates row ids, feature names, and intercept configuration before reuse

Key principle:

- neutralization without an intercept is wrong unless centering invariants are proven

### 3. Validation Splitting

`PurgedEraSplitter` in `src/features.py`:

- operates on unique eras instead of row-level CV state
- purges eras within a symmetric buffer around validation folds
- rejects non-numeric era formats instead of silently inventing chronology

Key principle:

- temporal leakage is a correctness failure, not a tuning detail

### 4. Training

`ModelOrchestrator` in `src/models.py`:

- supports `lightgbm` and `xgboost`
- prefers GPU and falls back to CPU if needed
- exposes `train_anchor_fold(...)`
- exposes `train_cross_validation(...)`
- produces OOF predictions and model collections

Key principle:

- anchor mode is for fast directional iteration
- full CV mode is for promotion-grade evaluation

### 5. Evaluation

`EvaluationEngine` in `src/evaluation.py`:

- computes NumerAI-style CORR
- computes FNC through the canonical rank -> gaussianize -> neutralize -> variance-normalize -> `numerai_corr` pipeline
- computes benchmark correlation and max feature exposure
- summarizes per-era metrics
- applies fast-fail gates
- supports `backend="custom"` and `backend="official"`

Important warning:

- the official backend can emit degenerate warnings or NaN edge states on flatline eras
- those are normalized at the engine boundary, not globally suppressed in notebooks
- do not add a global notebook warning filter

### 6. Deployment and Promotion

`DeploymentHarness` in `src/deployment.py`:

- serializes native LightGBM or XGBoost model binaries
- writes `manifest.json`
- records feature ordering
- records model SHA-256 hashes
- records environment fingerprint information
- verifies hashes before live prediction
- reorders live features strictly to the manifest order

`AdversarialStressTester` in `src/deployment.py`:

- performs the current v1 stress test
- injects per-feature Gaussian noise based on feature standard deviation
- compares degraded summary metrics against a threshold

`PromotionRunner` in `src/runner.py`:

- executes the deterministic promotion DAG
- halts on gate failure
- runs custom evaluation
- runs official parity on a sampled era slice
- runs stress testing
- serializes artifacts
- reloads artifacts and verifies exact prediction parity in the smoke test

## Promotion DAG

The promotion DAG in `PromotionRunner.run()` is:

1. Ingest and split
2. Train full CV
3. Evaluate on the custom backend
4. Apply fast-fail gate
5. Run oracle parity on a sampled era slice
6. Run stress test v1
7. Serialize artifact bundle with hashes and environment fingerprint
8. Reload artifact and assert exact prediction match on a control sample

This is the main operational boundary of the repo.

## Notebook Control Plane

The primary user interface is `notebooks/check_harness.ipynb`.

It is intentionally notebook-native. There is no CLI because a CLI would be operational packaging, not research necessity.

Notebook structure:

1. Cell 1 - initialization and singleton setup
2. Cell 2 - research scratchpad
3. Cell 3 - declarative promotion config
4. Cell 4 - promotion runner invocation and report rendering

Presentation belongs in `src/notebook_utils.py`, not in the notebook body.

## Artifact Contract

Serialized promotion bundles currently contain:

- native model files, for example `model_01.txt`
- `manifest.json`

Manifest fields currently include:

- `manifest_version`
- `model_library`
- `feature_names`
- `model_files`
- `model_hashes`
- `environment`
- `config_metadata`

Environment fingerprint currently includes:

- Python version
- OS platform
- exact versions of `polars`, `numpy`, `lightgbm`, `xgboost`, `scipy`
- `requirements.txt` path and SHA-256 if available

Live inference rules:

- missing manifest is fatal
- missing required features is fatal
- model hash mismatch is fatal
- feature order is enforced strictly from the manifest

## Validation Commands

Run the full suite:

```powershell
.\.venv\Scripts\python -m pytest tests/test_data.py tests/test_risk.py tests/test_features.py tests/test_models.py tests/test_evaluation.py tests/test_deployment.py tests/test_parity.py tests/test_runner.py -q
```

Run only parity:

```powershell
.\.venv\Scripts\python -m pytest tests/test_parity.py -q
```

Run only deployment and runner checks:

```powershell
.\.venv\Scripts\python -m pytest tests/test_deployment.py tests/test_runner.py -q
```

## Known Limitations

These are real limitations, not cosmetic ones:

1. Stress testing is still v1.
   The current perturbation is independent Gaussian feature noise. That is useful, but not the final institutional battery. The next serious upgrade is empirical rank jitter, feature dropout, and blockwise perturbation methods that respect the discrete NumerAI feature topology.

2. Official-backend warnings are not suppressed.
   This is intentional. Do not add global warning filters in notebooks. If warning suppression is needed later, scope it surgically around oracle calls only.

3. The official parity tests are strong but not exhaustive.
   Current parity covers direct function parity plus an adversarial multi-era frame. A future hardening pass should add a small real-data parity sample in CI.

4. `src/runner.py` is a library entrypoint, not a separate operations wrapper.
   This is fine for notebook-driven research. Do not build a CLI unless unattended or multi-operator execution becomes necessary.

## Rules for Future Agents

If you pick this repo up cold, follow these rules:

1. Do not rewrite canonical scoring math casually.
2. If you modify `src/evaluation.py`, run `tests/test_parity.py` immediately.
3. Do not replace native model serialization with Python pickle.
4. Do not add a global warning filter to the notebook.
5. Do not remove hash validation from live inference.
6. Do not reintroduce eager schema preload in `src/data.py`.
7. Do not relax era purge logic without proving no leakage.
8. If a new metric or deployment behavior is added, extend the promotion DAG tests.

## Immediate Next Work

The infrastructure tranche is closed. The next work is alpha generation, not more chassis polishing.

Highest-value next directions:

1. New feature families and feature subset experiments
2. Auxiliary target specialization and blends
3. Ensemble diversification across model classes and target definitions
4. Better stress testing tailored to NumerAI feature topology
5. Real-data oracle parity slice in CI

## Minimal Cold-Start Recipe

If a new agent needs to pick this up immediately:

1. Read this file.
2. Read `src/runner.py`.
3. Read `src/evaluation.py`.
4. Read `tests/test_parity.py`.
5. Open `notebooks/check_harness.ipynb` and start from Cell 3 if you want a promotion run.
6. Do not touch the scoring path without re-running parity.

That is the actual system boundary.