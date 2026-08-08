# numer-AI-refactored — Numerai Quantitative Research Framework

A **lean, deterministic research framework** for the [Numerai Classic tournament](https://numer.ai), built as a single tested Python package (`nmr/`). It takes a typed YAML config through data ingestion, era-purged cross-validation, multi-target LightGBM/XGBoost training, rank-domain ensembling, feature neutralization, oracle-parity evaluation, and out the other end produces a registry-tracked, cloudpickled `predict()` artifact ready for hosted upload.

**Stack:** Python 3.11+ · Polars · LightGBM / XGBoost · NumPy / SciPy / scikit-learn · `numerai-tools` · `numerapi` · cloudpickle · pytest (387 tests)

> **For AI coding agents:** [`AGENTS.md`](AGENTS.md) is the authoritative source of truth for principles, invariants, and operational hazards — read it first. System internals live in [`ARCHITECTURE.md`](ARCHITECTURE.md). Humans contributing code should read [`CONTRIBUTING.md`](CONTRIBUTING.md). This README is a human-facing overview and setup guide; when documents disagree, trust `AGENTS.md` and the code.

---

## What it does

- **Deterministic experiments** — one frozen `ExperimentConfig` drives everything; identical config + data + code produces an identical SHA256 `run_id`, OOF predictions, and scorecard hash.
- **Leakage-safe validation** — era-grouped walk-forward / anchor splits with an 8-era purge for 20D targets (16 for 60D). Random row-level CV is structurally impossible.
- **Oracle-parity metrics** — fast custom CORR / MMC / FNC / BMC / CWMM implementations, each pinned to `numerai_tools.scoring` by parity tests.
- **Rank-domain ensembling** — per-era rank-gaussianized component blending with ridge/NNLS weight learning.
- **Feature neutralization** — intercept-aware per-era least squares with a content-addressed pseudo-inverse cache.
- **Institution-grade evaluation** — block-bootstrap CIs, AC-adjusted Sharpe, Deflated Sharpe, payout proxy with burn/drawdown/CVaR/sortino diagnostics, perturbation / horizon / regime robustness.
- **Benchmark harness** — null + classical baselines and Numerai benchmark models scored through the same pipeline, with monotone-sanity and null-floor gates.
- **Run registry & deployment** — atomic filesystem registry with champion promotion; cloudpickled `predict()` artifacts with SHA256 integrity manifests.

---

## Project Structure

```
.
├── nmr/                       # the framework package — the ONLY tested boundary
│   ├── __init__.py            # public API surface (keep imports + __all__ in sync)
│   ├── config.py              # typed frozen YAML config, seeding, path resolution
│   ├── data.py                # IngestionAgent — lazy Polars scans over data/v5.2
│   ├── splitter.py            # PurgedEraSplitter — leakage-safe era folds
│   ├── _transforms.py         # shared rank / gaussianize / power-1.5 transforms
│   ├── evaluation.py          # dual-backend CORR/MMC/FNC/BMC/CWMM metric engine
│   ├── features.py            # feature-set resolution + stability screening
│   ├── risk.py                # NeutralizationEngine — per-era, intercept-aware, cached
│   ├── models.py              # ModelOrchestrator — LightGBM/XGBoost, CV OOF + anchor
│   ├── ensemble.py            # rank-domain blending, ridge/NNLS weight learning
│   ├── inference.py           # bootstrap CI, AC-adjusted Sharpe, Deflated Sharpe
│   ├── meta.py                # cross-run meta-analysis + promotion verdicts
│   ├── payout.py              # payout proxy + downside diagnostics
│   ├── scorecard.py           # MetricScorecard aggregator (evaluate_model)
│   ├── research.py            # HPO sweeps, neutralization frontier, exposure report
│   ├── robustness.py          # perturbation, horizon-stability, regime diagnostics
│   ├── benchmark.py           # benchmark suite: null/classical baselines + gates
│   ├── campaign.py            # campaign orchestration — trial-lineage logs
│   ├── runner.py              # ExperimentRunner — deterministic end-to-end pipeline
│   ├── registry.py            # RunRegistry — atomic run store + champion promotion
│   ├── submission.py          # submission build / numerai_tools validation / CSV
│   └── deployment.py          # cloudpickle predict artifact + integrity manifest
├── configs/                   # experiment configs (YAML)
│   ├── example.yaml           # annotated full schema
│   └── first_model.yaml       # current competitive config (4×20D-target ensemble)
├── tests/                     # 387 tests (unit / parity / determinism / real-data tests)
├── data/                      # local Numerai v5.2 assets (parquets git-ignored)
├── artifacts/                 # runs, registry, caches, campaigns, benchmark CSVs (generated)
├── docs/                      # curated Numerai knowledge base — start at docs/DOCS_README.md
├── notebooks/                 # researcher control plane (thin, zero business logic)
├── benchmark_runner.py        # CLI: score null/classical/benchmark baselines → CSV
├── run_campaign.py            # CLI: run a named batch of configs → artifacts/campaigns/
├── train_first_model.py       # CLI: train, register, and promote the first model
├── generate_dashboard.py      # CLI: validation-scorecard leaderboard → artifacts/dashboard.html
├── pytest.ini                 # pythonpath = . (no install step needed)
├── requirements.txt           # runtime + dev dependencies
├── AGENTS.md                  # authoritative reference for AI coding agents
├── ARCHITECTURE.md            # pipeline topology, formulas, schemas
└── CONTRIBUTING.md            # human contributor workflow + test commands
```

---

## Configuration

Experiments are parameterized by a single typed `ExperimentConfig` — nothing else reads YAML, and invalid configs (unknown keys, bad enum values) fail loudly at load time.

```python
from nmr import load_config, set_global_seeds

cfg = load_config("configs/example.yaml")
set_global_seeds(cfg.run.seed)
```

Sections: `run` (name, seed, artifacts dir) · `data` (version, feature set, targets) · `split` (scheme, purge/embargo, folds) · `model` (backend, preset, param overrides) · `evaluation` (backend, main target, metrics, validation scorecard) · `ensemble` (weight-learning method) · `risk` (neutralization proportion, cache budget). The annotated schema lives in [configs/example.yaml](configs/example.yaml); exact fields, defaults, and valid values are specified in [`ARCHITECTURE.md`](ARCHITECTURE.md#a-configuration--nmrconfigpy).

Numerai API credentials (for `numerapi` download/upload) are used only in notebooks and loaded from a git-ignored `.env`; no credentials are needed to run tests or train on already-downloaded data.

---

## Data Assets

The framework expects Numerai **v5.2** assets in `data/v5.2/` (downloadable via `numerapi`; see [data/refresh_data.ipynb](data/refresh_data.ipynb)):

| File | Purpose |
|---|---|
| `train.parquet` / `validation.parquet` / `live.parquet` | Feature + target data per split |
| `features.json` | Feature-set definitions (`small` / `medium` / `all`) + target inventory |
| `meta_model.parquet` | Stake-weighted meta-model predictions (MMC/CWMM) |
| `train_benchmark_models.parquet` / `validation_benchmark_models.parquet` / `live_benchmark_models.parquet` | Numerai-provided benchmark model predictions (BMC, horizon diagnostics) |
| `live_example_preds.parquet` / `validation_example_preds.parquet` | Example prediction artifacts |

Real-data tests and `benchmark_runner.py` require these files; pure-unit tests do not.

---

## Quickstart

```powershell
# 1. Activate the venv and install dependencies
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python -m pip install -r requirements.txt

# 2. Train, register, and promote the first competitive model
.\.venv\Scripts\python train_first_model.py

# 3. Score the baseline field and build the leaderboard
.\.venv\Scripts\python benchmark_runner.py --fast-mode
.\.venv\Scripts\python generate_dashboard.py     # → artifacts/dashboard.html
```

The dashboard ranks trained runs and benchmarks on the same validation-scorecard definitions (CORR/Sharpe from the `scorecard` block in `run.json`); runs without a validation scorecard are shown separately in a legacy (train-OOF metrics) section.

Library usage:

```python
from nmr import ExperimentRunner, RunRegistry, load_config

cfg = load_config("configs/first_model.yaml")
result = ExperimentRunner(cfg).run(deploy=True)   # RunResult(run_id, oof, metrics, artifact, manifest)

registry = RunRegistry(cfg.run.artifacts_dir / "registry")
registry.record(result)
registry.promote(result.run_id)                   # → artifacts/registry/champion.json
```

---

## Testing

See [`CONTRIBUTING.md`](CONTRIBUTING.md#testing--verification) for the full workflow, targeted subsets, and the pre-sign-off gate.

```powershell
# Quick verification (from the repo root)
.\.venv\Scripts\python -m pytest -q
```

---

## Repository Guide

### Core docs (start here)

- [`AGENTS.md`](AGENTS.md) — principles, invariants, verification gates, hazards (for AI agents and contributors)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — pipeline topology, metric formulas, artifact schemas
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development workflow, test commands, review checklist

### Domain knowledge base ([docs/](docs/DOCS_README.md))

A curated, tiered Numerai knowledge base with a deterministic reading path:

- [docs/01-canon/](docs/01-canon/overview.md) — canonical tournament truth: data, scoring (CORR/MMC/BMC/FNC), submissions, staking
- [docs/02-strategy/](docs/02-strategy/strategy-bible.md) — strategy bible, community wisdom, target-ensembling math
- [docs/03-reference/](docs/03-reference/numerai-tools.md) — `numerapi` and `numerai-tools` API references
- [docs/04-research/](docs/04-research/research-program.md) — research program + consolidated research-ideas file (incl. neural-network directions), deep tabular-DL survey
- [docs/05-notebooks/](docs/05-notebooks/) — onboarding notebooks (hello-numerai, neutralization, target ensembles, sunshine example)
- [docs/06-evaluation/](docs/06-evaluation/evaluation-suite-bible.md) — **the evaluation spec of record**: how this repo judges a model; the benchmark null-floor / S11 ladder is [benchmark-line-in-the-sand.md](docs/06-evaluation/benchmark-line-in-the-sand.md)
- [docs/99-archive/](docs/99-archive/) — archived, low-priority reference (bounty/security pointer, grandmaster-seasons summary, super-research prompt); raw source originals preserved unmodified under `docs/99-archive/raw-source/`

The authoritative map — importance tiers, per-file table, and reading recipes — lives in [docs/DOCS_README.md](docs/DOCS_README.md); the bullets above are a directory summary, not the map itself.

### Other

- [notebooks/](notebooks/) — researcher scratch space (thin control plane only)
- `../numer-AI/` — V1 legacy repo: **read-only**; mined for logic, never imported
