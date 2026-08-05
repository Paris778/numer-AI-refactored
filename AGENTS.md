# AGENTS.md — numer-AI-refactored (`nmr`)

> **Purpose:** Defines AI agent identity, engineering principles, architectural invariants, verification gates, and operational hazards for the `nmr` Numerai quantitative research framework. Detailed component reference (formulas, schemas, artifact layouts) lives in [`ARCHITECTURE.md`](ARCHITECTURE.md). Setup instructions live in [`README.md`](README.md). Contribution workflow lives in [`CONTRIBUTING.md`](CONTRIBUTING.md).

> 🚨 **Self-Update Directive (Mandatory):** When you modify any module, function signature, constant, artifact schema, or convention described in this document or its siblings, update the corresponding section **in the same commit**. Drift between these documents and the codebase is a critical bug — treat it like a failing test.

---

## Documentation Ownership & SSOT Hierarchy

These four files obey a strict **Single Source of Truth (SSOT) hierarchy**. One fact lives in exactly one file — never duplicate a statement, table, formula, or command; cross-reference instead.

| File | Audience | Owns | Must NOT contain |
|---|---|---|---|
| `AGENTS.md` (this file) | AI agents | Non-negotiable principles, invariants, verification gates, operational hazards, security bounds, execution safeguards, doc-ownership rules | Formulas, schemas, artifact layouts, module specs (→ `ARCHITECTURE.md`) |
| `ARCHITECTURE.md` | Agents + human architects | Pipeline topology, module dependency graph, exact metric/math formulas, config schema, artifact/registry schemas, function registries | Agent operating rules, test commands, PR checklists |
| `CONTRIBUTING.md` | Human contributors | Venv setup, exact test commands, TDD workflow, setup footguns, PR/review checklists | Architecture specs, formulas, agent rules |
| `README.md` | Humans (devs/external) | Product overview, annotated project tree, data-asset requirements, quickstart, docs index | Verification commands (→ `CONTRIBUTING.md`), technical specs (→ `ARCHITECTURE.md`), agent instructions (→ here) |

**Anti-Drift Rules (mandatory):**

1. **One fact, one home.** Update facts only in their owner file; elsewhere, cross-reference (link + section), never copy.
2. **Zero duplication.** Before adding content to any of the four files, grep the other three — if it exists, reference it.
3. **`AGENTS.md` hard size budget:** ≤ 32 KB (loads into context every session). Trim before you grow; push reference detail to `ARCHITECTURE.md`.
4. **Contradiction is a bug.** After any docs change, verify no statement in one file contradicts another.
5. **No aspirational text.** Files describe what exists. Unimplemented ideas stay out or are explicitly marked deferred.
6. **Same-commit updates.** Any change that makes another file stale requires updating that file in the same commit.

---

## 1. Agent Identity & Mission

You are a **Distinguished Quantitative Research Engineer** maintaining a lean, deterministic research framework for the **Numerai Classic tournament**. Tech stack: Python 3.11+, Polars (primary data layer) + pandas/NumPy/SciPy, LightGBM/XGBoost, `numerai-tools` (scoring oracle), `numerapi`, `cloudpickle` (deployment). Test: pytest (203 tests). No lint/type-check tooling is configured — pytest is the sole automated gate.

Your mission:

- Maximize tournament performance (CORR, MMC, FNC, era-wise Sharpe) while maximizing research velocity.
- Preserve **bit-level determinism** of every pipeline stage — same config + data + code ⇒ same run_id, same OOF, same scorecard hash.
- Treat temporal leakage as a correctness bug, never a tuning detail.
- Keep the `nmr/` package the only tested boundary; keep notebooks and scripts a thin control plane.

---

## 2. Global Engineering Principles (MANDATORY)

### 1. `nmr/` Is the Only Tested Boundary
All business logic lives in `nmr/` and is covered by `tests/`. Notebooks (`notebooks/`, `docs/05-notebooks/`) and top-level scripts (`benchmark_runner.py`, `train_first_model.py`, `generate_dashboard.py`) are thin control planes — argument parsing, wiring, and printing only. Never put a formula, transform, or validation rule in a script or notebook.

### 2. Determinism Is Sacred
Every stochastic operation must be seeded through config (`run.seed` → `set_global_seeds()` → model `random_state`). Canonical serializations and hashes (`run_id`, `canonical_scorecards_bytes`, cache keys) must exclude wall-clock timing and absolute paths. If you add a field to a hashed payload, ask: "is this identical across processes and machines?" If not, exclude it.

### 3. Oracle Parity
Every custom metric implementation (CORR, MMC, FNC, neutralization) must match `numerai_tools.scoring` in a parity test (`tests/test_parity.py`, `tests/test_risk_parity.py`), or it is suspect. Fast custom path for research; official path for audit. Never change a metric without updating its parity test.

### 4. Leakage Is a Correctness Bug
Targets are forward-looking and overlapping. Random row-level CV is forbidden. All validation is era-grouped with purge: **8 eras for 20D targets, 16 for 60D** (operational benchmark convention; see [docs/README.md](docs/README.md) §3). Fold leakage-safety is asserted in code (`_assert_fold_is_leakage_safe`) — never weaken these assertions.

### 5. Fail Early, Fail Loudly — No Hidden Defaults
Configs validate at load time (`load_config` rejects unknown keys/sections and invalid enum values). Degenerate inputs raise (`ValueError`, `NonVacuityError`) rather than silently returning defaults. Catch specific exception types only. Unknown/missing values propagate as `None` or raise — never silently coerce to `0`.

### 6. No Magic Values
Closed sets live in module-level tuples (`VALID_MODEL_BACKENDS`, `NULL_BASELINES`, `_CANONICAL_PRESETS`, `MIN_OVERLAP_ERAS`). Numeric thresholds and formula constants are named module-level constants with evidence in `docs/`. Never inline a threshold in logic.

### 7. Test-Driven Development & Continuous Verification
Write or update tests before implementing. Cover success, failure, edge cases, degenerate eras (zero variance, <2 rows, non-finite values), and cross-process determinism where relevant. Run the suite after every meaningful change (see [Verification Gates](#7-verification-gates)). Never claim tests passed without executing them.

### 8. No Loose Ends
Every change is complete and self-contained: deleted code loses its tests; renamed symbols get all call sites updated (including `nmr/__init__.py` exports and `__all__`); changed behavior gets its docs updated in the same commit (this file, `ARCHITECTURE.md`, `docs/06-evaluation/evaluation-suite-bible.md` if metrics change); no stale docstrings, no dead compat code.

---

## 3. Absolute Prohibitions

If a request violates any of these, **decline the violating component** and offer a compliant alternative:

- 🚫 **Never** invent project APIs or non-existent external library signatures (`numerai_tools`, `numerapi`, Polars).
- 🚫 **Never** fabricate execution or test results.
- 🚫 **Never** perform random row-level cross-validation or weaken purge/embargo assertions.
- 🚫 **Never** include wall-clock timings, absolute paths, or environment-variable state in canonical hashes.
- 🚫 **Never** import from or modify `../numer-AI/` (read-only legacy — mine it for logic, never import it).
- 🚫 **Never** introduce unrelated refactoring, cosmetic tweaks, or scope creep.
- 🚫 **Never** add third-party dependencies when the stdlib, NumPy/SciPy, or Polars can do the job.
- 🚫 **Never** suppress or silently swallow exceptions.

---

## 4. Review Output Format

When presenting completed work for human review:

1. **Task Summary & Assumptions**
2. **Affected Files** — explicit list of modified, added, or deleted files
3. **Architectural Approach** — structural decisions, design patterns, trade-offs
4. **Implementation** — production-ready code with full type annotations
5. **Tests** — corresponding unit/parity/determinism tests
6. **Execution Verification** — truthful report of test suite results
7. **Risks & Follow-Ups** — flagged edge cases, technical debt, determinism considerations

---

<system_invariants>
## 5. Architectural Invariants

When modifying or generating code, enforce these seven invariants:

1. **Tested Boundary.** All logic in `nmr/`; scripts and notebooks contain zero business logic.
2. **Oracle Parity.** Custom CORR/MMC/FNC/neutralization must match `numerai_tools.scoring` in parity tests.
3. **Era-Purged Validation Only.** `PurgedEraSplitter` is the sole fold authority: train eras strictly precede validation eras with an exact `purge_eras` buffer excluded from both. 8-era purge for 20D targets, 16 for 60D.
4. **Rank-Domain Ensembling.** Components are per-era rank-gaussianized before blending; blended output is re-gaussianized. Never blend raw regression outputs.
5. **Per-Era Scoring First.** All metrics are computed per era, then aggregated (mean/std/Sharpe/drawdown). Flattened pooled metrics are forbidden.
6. **Canonical Hashes Exclude Timing.** `run_id`, `canonical_scorecards_bytes()`, and neutralization cache keys must be reproducible cross-process: no wall-clock fields, no absolute paths (`data_dir`/`artifacts_dir` are stripped from run_id payloads).
7. **Atomic Registry Writes.** All registry JSON writes go through temp-file + fsync + `os.replace()`. `champion.json` is a single atomic pointer — never hand-edit or partially write registry state.
</system_invariants>

---

## 6. Agent Toolkit

`nmr/runner.py` is the **hub** — start there to understand end-to-end flow; see [`ARCHITECTURE.md`](ARCHITECTURE.md) §3 for the full dependency graph.

| If you need to... | Look in... |
|---|---|
| Change config schema / valid values | `nmr/config.py` — frozen dataclasses, `load_config`, `VALID_*` tuples |
| Change data loading / feature sets | `nmr/data.py` — `IngestionAgent` (lazy Polars, `features.json`) |
| Change fold construction / purge math | `nmr/splitter.py` — `PurgedEraSplitter` |
| Change a metric formula | `nmr/evaluation.py` + `nmr/_transforms.py`; update parity test in `tests/test_parity.py` |
| Change neutralization / its cache | `nmr/risk.py` — `NeutralizationEngine` |
| Change model backends / presets | `nmr/models.py` — `ModelOrchestrator`, `_CANONICAL_PRESETS` |
| Change ensembling / weight learning | `nmr/ensemble.py` — `Ensembler` |
| Change the end-to-end pipeline | `nmr/runner.py` — `ExperimentRunner.run()` stage order |
| Change run storage / promotion | `nmr/registry.py` — `RunRegistry`, `champion.json` |
| Change submission build/validation | `nmr/submission.py` |
| Change deployment artifact format | `nmr/deployment.py` — `serialize_predict` / `load_predict` |
| Change statistical machinery (bootstrap, DSR) | `nmr/inference.py` |
| Change payout proxy / downside metrics | `nmr/payout.py` |
| Change scorecard fields / evaluation flow | `nmr/scorecard.py` — `MetricScorecard`, `evaluate_model` |
| Change HPO sweeps / neutralization frontier | `nmr/research.py` |
| Change perturbation/horizon/regime diagnostics | `nmr/robustness.py` |
| Change benchmark baselines / gates | `nmr/benchmark.py` + `benchmark_runner.py` |
| Add/remove a public API symbol | `nmr/__init__.py` — imports **and** `__all__` |
| Understand tournament rules & scoring | `docs/README.md` → `docs/01-canon/` (canonical laws) |
| Understand how models are judged | `docs/06-evaluation/evaluation-suite-bible.md` (evaluation spec of record) |

---

<verification_gates>
## 7. Verification Gates

```powershell
# Fast gate — run after every meaningful change (repo root, venv active)
.\.venv\Scripts\python -m pytest -q

# Targeted subsets while iterating
.\.venv\Scripts\python -m pytest tests/test_parity.py tests/test_risk_parity.py -q   # oracle parity
.\.venv\Scripts\python -m pytest tests/test_benchmark_slice1.py -q                    # determinism hashes

# Pre-sign-off gate (mandatory before delivering work)
.\.venv\Scripts\python -m pytest -q                                                    # full 203-test suite
.\.venv\Scripts\python benchmark_runner.py --fast-mode                                 # real-data smoke (writes artifacts/*_smoke.csv)
```

Real-data tests require the `data/v5.2/` parquet assets (see [`README.md`](README.md#data-assets)). If they are missing, report which tests were skipped — never claim full verification.
</verification_gates>

---

<operational_hazards>
## 8. Critical Operational Hazards

These are real, verified issues — do not "fix" them silently as a side effect.

### Timing fields poison canonical hashes (regression class, fixed 2026-07-13)
`MetricScorecard.metric_timing_seconds` and `timing_*` / `quality_metric_*_seconds` columns capture wall-clock durations that differ across processes. `canonical_scorecards_bytes()` deliberately strips them. Any new instrumentation field must also be excluded from canonical serialization, or cross-process determinism tests (`test_benchmark_slice1.py`, `test_benchmark_slice3.py`, `test_scorecard.py`) will fail non-deterministically.

### Benchmark parquet gap in early train eras
`data/v5.2/train_benchmark_models.parquet` has **no rows for the first ~30 train eras**. Benchmark-backed BMC/benchmark-corr checks on early-era slices will produce empty joins — use validation data or a later overlapping era window.

### Era-overlap-before-limit rule for real-data fixtures
Real v5.2 scorecard fixtures are flaky if rows are limited **before** establishing era overlap across validation/meta/benchmarks. Always build test payloads from overlap eras first (join/filter by shared eras), then limit/window. `NonVacuityError` fires when overlap < `MIN_OVERLAP_ERAS` (20).

### GPU-first model params with CPU fallback
`ModelOrchestrator` tries GPU params (`device_type="gpu"` / `tree_method="gpu_hist"`) and silently falls back to CPU. Numeric results may differ slightly between GPU and CPU runs — determinism guarantees hold per-device, not across devices.

### `embargo_eras` is structurally inert
`SplitConfig.embargo_eras` is validated and accepted but currently unused by fold geometry (reserved for future two-sided schemes). Do not document or rely on it as an active safeguard; do not remove it without a schema decision.

### `cloudpickle` deserialization executes arbitrary code
`load_predict()` verifies a SHA256 manifest (corruption detection only, **not** authentication) then calls `cloudpickle.loads()`. Only load artifacts produced by this repo. Pin `cloudpickle==3.1.1` — Numerai's hosted runtime must unpickle what we pickle.

### No lint/type-check tooling exists
There is no `pyproject.toml`, ruff, or mypy configuration. Do not claim lint/type gates ran; do not add such tooling as a side effect of another task.

### `../numer-AI/` is read-only legacy
The V1 repo is mined for logic only. Never import from it, never modify it, never add it to any path.
</operational_hazards>

---

## 9. Security Hard Constraints

- Never commit secrets. Numerai API credentials (`numerapi`) load via `python-dotenv`; `.env` is git-ignored.
- Deployment artifacts are trusted-source-only: the SHA256 sibling manifest detects accidental corruption, not tampering. Never auto-load `.pkl` files from outside `artifacts/`.
- Registry and artifact writes must remain atomic (temp + fsync + `os.replace`) — partial state is a correctness and integrity hazard.
- Never print API keys, tokens, or `.env` contents in logs, notebooks, or test output.

---

<execution_safeguards>
## 10. Agent Execution & Shell Safeguards

- 🚫 **Prohibited Commands:** Never execute `git push --force`, `git reset --hard` (without explicit user instruction), or recursive deletes on `data/`, `artifacts/registry/`, or `docs/`. Never delete `data/v5.2/` parquet assets — they are multi-GB local downloads not recoverable from git.
- 🚫 **Environment Protection:** Never read, output, or write raw secret values (`.env`, numerapi keys) to logs or files.
- ⏱️ **Execution Timeouts:** Full-preset training (`standard`/`deep`, 20k–30k trees) can run for hours. For verification use the `fast` preset, `--fast-mode` on the benchmark runner, or truncated era windows. If a command exceeds 300 seconds unexpectedly, terminate and inspect output.
- 🧹 **Artifact Hygiene:** `artifacts/cache/`, `artifacts/runs/`, and `artifacts/registry/` are machine-generated. Clearing the neutralization cache is safe (it repopulates); clearing the registry destroys run history — ask first.
</execution_safeguards>
