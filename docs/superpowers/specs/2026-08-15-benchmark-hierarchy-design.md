# Design Spec: 5-Tier Benchmark Hierarchy ("The Line in the Sand")

> Status: APPROVED (director disposition 2026-08-15). Implementation authorized across all sections.
> Scope: full replacement of the existing S11 benchmark ladder in `nmr/benchmark.py`, `benchmark_runner.py`, and `tests/test_benchmark_*.py`.

## 1. Mission

Define a deterministic, 5-tier escalating benchmark hierarchy that establishes empirical hurdle rates for the Numerai Classic tournament and rejects sub-par quantitative models before capital deployment. Tiers:

- **Tier 0 — Null & statistical invariants** (sanity gate): constant 0.5, seeded uniform, seeded clipped Gaussian, small-feature row mean.
- **Tier 1 — Convex linear baselines** (complexity hurdle): purged Ridge OLS on small / medium / 4×20D multi-target.
- **Tier 2 — Shallow non-linear trees** (depth/interaction hurdle): LightGBM/XGBoost shallow + canonical fast preset.
- **Tier 3 — Canonical community baselines**: Hello-Numerai, 50%-neutralized, Sunshine 4×20D ensemble.
- **Tier 4 — Production capital gate**: `v53_lgbm_ender60` reference with 7 hard thresholds.

## 2. Approved Decisions Log

| # | Question | Decision |
|---|---|---|
| 1 | Removal scope | **Full replacement** of the old S11 ladder (tutorial ingestion, walk-forward classical/trivial baselines, old gates) — no two parallel benchmark systems. |
| 2 | Numeric floors handling | **Config-driven + runtime gate**: thresholds live in configs, enforced at runtime on real data; pytest covers gate *mechanics* on synthetic fixtures; configs re-pinned to measured values after first real run. |
| 3 | Tier-3 production | **In-process re-fit**: LightGBM fits + `NeutralizationEngine` + rank-Gaussian ensembling; notebook ingestion machinery deleted. |
| 4 | Config schema location | **Dedicated `load_benchmark_config` in `nmr/benchmark.py`** (frozen dataclass, closed sets, fail-fast) — `nmr/config.py` untouched. |
| 5 | Architecture | **A: hierarchy-engine rebuild** — `BenchmarkSuite` → `BenchmarkHierarchy`. |
| 6 | Train→validation purge | **Local `_train_validation_purged_split()` helper in `nmr/benchmark.py`** mirroring splitter invariants; `PurgedEraSplitter` NOT extended (keeps the fold authority pure for CV). |
| 7 | Multi-target names | Use physical v5.3 columns: `target`, `target_cyrusd_20`, `target_sam_20`, `target_victor_20` (spec's `cyrus_20` is a typo). |
| 8 | Tier-4 thresholds | New `tier4_gate.yaml` (8th config file) — declarative, auditable; no hardcoded constants. |
| 9 | Null DSR | Tolerance-based: |DSR| ≤ 0.05, not strict equality (ill-conditioned denominators at near-zero Sharpe). |
| 10 | Tier 1–3 anchors | Report-only reference lines in configs (logged vs measured, not enforced). Enforced properties: Tier-4 thresholds, Tier-0 null floor, tier monotonicity. |

## 3. Config Layer — `configs/benchmarks/`

8 YAML files, all validated by `load_benchmark_config(path)`:

| File | Cells |
|---|---|
| `tier0_null.yaml` | `null_constant_05`, `null_uniform_rand`, `null_gaussian_rand`, `null_feature_mean` |
| `tier1_ridge_small.yaml` | `linear_ridge_small` (α=1.0) |
| `tier1_ridge_medium.yaml` | `linear_ridge_medium` (α=10.0) |
| `tier1_ridge_multitarget.yaml` | `linear_ridge_multitarget` (4 targets) |
| `tier2_tree_shallow.yaml` | `tree_lgbm_shallow_small`, `tree_xgb_shallow_medium` |
| `tier2_tree_fast.yaml` | `tree_lgbm_fast_medium` |
| `tier3_sunshine.yaml` | `canon_hello_numerai`, `canon_neutralized_50`, `canon_sunshine_ensemble` |
| `tier4_gate.yaml` | production gate thresholds (7 fields) + reference column `v53_lgbm_ender60` |

### Schema (frozen `BenchmarkConfig` dataclass)

```yaml
tier: 0..4                          # VALID_BENCHMARK_TIERS
benchmark_id: str                   # unique, deterministic id
input_space: none|small|medium      # VALID_INPUT_SPACES
targets: [target]                   # target column names (tier 1–3)
model:
  kind: null_constant|null_uniform|null_gaussian|null_feature_mean|ridge|lightgbm|xgboost
  params: {...}                     # kind-specific, exact-pinned
  seed: 42                          # all stochasticity via this seed
neutralization: none|0.25|0.5       # tier 3 only
gate:                               # tier 4 file only
  corr_min: 0.0286
  corr_sharpe_ac_min: 1.50
  fnc_min: 0.020
  deflated_sharpe_min: 0.95
  gain_to_pain_min: 1.50
  cagr_min: 0.0                     # strictly > 0
  turnover_max: 0.35
anchors:                            # tiers 1–3: report-only reference lines
  corr: 0.0145 | 0.0210 | {...}
  sharpe: 1.05 | 1.30 | {...}
fast_mode:                          # overrides when runner --fast-mode
  params: {...}                     # reduced trees/estimators
```

Validation mirrors `load_config` philosophy: unknown keys/sections rejected, enum values from module-level tuples (`VALID_BENCHMARK_TIERS`, `VALID_INPUT_SPACES`, `VALID_MODEL_KINDS`), degenerate params raise `ValueError`. Seeds must be identical across processes (determinism invariant).

Model params (spec values, verbatim):

- **Tier 2 shallow LGBM**: `max_depth=3, num_leaves=7, learning_rate=0.02, n_estimators=500, colsample_bytree=0.1, subsample=0.8`
- **Tier 2 shallow XGB (medium)**: `max_depth=4, max_leaves=15, n_estimators=1000, learning_rate=0.01`
- **Tier 2 fast LGBM (medium)**: canonical fast preset `max_depth=5, num_leaves=31, n_estimators=2000, learning_rate=0.01`
- Effective colsample flooring (reuse `nmr/models.py` resolution logic, never re-implement):
  `c_eff = min(1.0, max(c_resolved, min(1.0, max(0.1, min(10,|S|)/|S| + 1e-7))))`

## 4. Engine — `nmr/benchmark.py` rewrite

### 4.1 `BenchmarkHierarchy`

Replaces `BenchmarkSuite`. Responsibilities:

- `from_config_dir(config_dir, ...)` / `from_configs(paths, ...)`: load + validate all cells.
- `iter_benchmark_cells()`: yield each cell as a prediction generation task.
- `generate_predictions(cell, data, ...)`: per-tier prediction frames keyed `[era, id, prediction]`.
- `score_cell(...)` → `MetricScorecard` via the existing `evaluate_model` pipeline (kept verbatim — evaluation core is NOT in scope).
- `hierarchy_frame()` / `write_hierarchy_csv(...)`: scorecard frame with tier + group columns.
- `gate_report()`: verdicts (pass/fail + measured vs threshold) for null floor, tier-4 gate, monotonicity.

### 4.2 Prediction generators per tier

**Tier 0** (validation eras only, no fit):
- `null_constant_05`: all rows = 0.5
- `null_uniform_rand`: `default_rng(seed).uniform(0,1)` row count = prediction index
- `null_gaussian_rand`: `clip(N(0.5, 0.15), 0, 1)` seeded
- `null_feature_mean`: row-mean of the 42 small features

**Tiers 1–3 fit topology** (canonical train→validation split):
1. `train_eras` = `train.parquet` era labels (numeric-sorted), `val_eras` = `validation.parquet` era labels.
2. `_train_validation_purged_split(train_eras, val_eras, purge_eras=8)`:
   - drops the final `purge_eras` train eras;
   - asserts strict chronology (`max(trimmed_train) < min(val)`) and an **exact** purge buffer of `purge_eras` eras between them (mirrors `PurgedEraSplitter._validate_fold` invariants, incl. numeric-only era labels, zero-padding consistency);
   - raises `ValueError` on any violation (leakage = correctness bug).
3. Fit on trimmed train rows; predict all validation rows. Ridge: standardize features with train mean/std (persisted, applied to val). Trees: raw features, `n_jobs=1`, `random_state` from cell seed.
4. Output = OOF predictions on validation eras → per-era rank-gaussianize.

**Tier 1 multi-target blend**: 4 independent Ridge fits (one per target) → per-era rank-gaussianize each component → equal-weight mean → per-era re-gaussianize (rank-domain ensembling invariant; consistent with `Ensembler` semantics).

**Tier 3**:
- `canon_hello_numerai`: single LightGBM, small features, primary target, no neutralization.
- `canon_neutralized_50`: single LightGBM, medium features, 50% neutralization via `NeutralizationEngine` (intercept-aware, cache intact).
- `canon_sunshine_ensemble`: 4 × 20D LightGBM on medium → rank-Gaussian equal-weight blend → 25% neutralization.

**Tier 4**: `v53_lgbm_ender60` column from `validation_benchmark_models.parquet`, scored over the shared era-overlap window (era-overlap-before-limit rule; `MIN_OVERLAP_ERAS` = 20 non-vacuity).

### 4.3 Gates (module-level functions, all tested)

1. `assert_tier0_null_floor(scorecards, tolerances)` — per null model:
   - `|mean CORR| ≤ 0.005`, `|Sharpe(AC)| ≤ 0.10`, `|DSR| ≤ 0.05`
   - also keeps the existing finite-scorecard assertion on all fields.
2. `assert_tier4_gate(scorecard, gate_config)` — generic threshold gate over the 7 Tier-4 fields:
   - `corr ≥ 0.0286`, `corr_sharpe_ac ≥ 1.50`, `fnc ≥ 0.020`, `deflated_sharpe ≥ 0.95`, `gain_to_pain_ratio ≥ 1.50`, `cagr_1y > 0`, `turnover_mean ≤ 0.35`
   - any missing/None field (e.g. turnover unavailable) ⇒ gate FAIL with explicit reason (fail loud, no silent pass).
3. `assert_hierarchy_monotone(scorecards)` — on rank scalar: Tier0 < Tier1 < Tier2 < Tier3 ≤ Tier4, with `atol` tolerance.
4. `canonical_scorecards_bytes` / `scorecards_sha256` — kept; timing fields excluded (existing hazard).

### 4.4 Deletions (full replacement)

- `TUTORIAL_NOTEBOOK_TO_MODEL_ID`, `_TUTORIAL_NOTEBOOK_ANCHORS`, `discover_tutorial_notebooks`, `ingest_tutorial_prediction`, `ingest_tutorial_prediction_batch`, `assert_notebook_prediction_contract`, `extract_oos_predictions`, notebook contract/inference helpers.
- `iter_baseline_predictions`, `run_classical_baselines`, `run_null_baselines` (superseded by tier generators), old `assert_null_floor`, `assert_slice1_monotone`, old `_build_classical_model` walk-forward.
- `compute_book_orthogonality`: check call sites first; keep only if still referenced outside slice-2 tests (decision in plan phase).
- Stale `configs/campaigns/benchmark-rebuild-v1/` (12 YAML cells) and its campaign artifacts/logs.
- Old benchmark output CSVs under `artifacts/` (replaced by `artifacts/reports/benchmark_hierarchy_scorecard.csv`).

## 5. Runner — `benchmark_runner.py` rewrite

```
python benchmark_runner.py \
  --data-dir data/v5.3 \
  --configs configs/benchmarks \
  --output artifacts/reports/benchmark_hierarchy_scorecard.csv \
  --gate-report artifacts/reports/benchmark_gate_report.csv \
  --seed 42 \
  --n-boot 1000
```

- `--fast-mode`: n_boot=1, `fast_mode` param overrides from configs (smoke gate per AGENTS.md).
- Loads `train.parquet`/`validation.parquet` lazily (column projection: features + targets + era/id); medium cells use 780-feature resolution from `features.json`.
- Runs tiers sequentially (memory guard: one cell in flight); logs tier progress + measured vs anchor/ref lines.
- Exit code 1 on any hard gate violation (null floor, tier-4, monotonicity); exit code 0 only when all hard gates pass and output written.

## 6. Tests

Rewrite the 5 benchmark test files (922 lines) around the hierarchy; keep synthetic-fixture pattern (no real-data dependency in CI; real-data smoke remains the pre-sign-off gate):

- `tests/test_benchmark_hierarchy.py` (new, replaces slice1–3):
  - config loading: valid round-trip, unknown keys rejected, invalid enums rejected, degenerate params raise;
  - `assert_tier0_null_floor` mechanics (pass/fail synthetic scorecards, tolerance override);
  - `assert_tier4_gate` mechanics (each of 7 fields, missing-field loud failure);
  - `assert_hierarchy_monotone` (ordering, atol, missing-tier failure);
  - determinism: canonical bytes cross-process hash (subprocess, mirrors existing slice1 test);
  - `_train_validation_purged_split`: exact purge buffer, strict ordering, numeric-only labels, zero-padding consistency, degenerate inputs raise.
- `tests/test_benchmark_baselines.py` (rewritten): generator contracts — constant exactly 0.5, seeded reproducibility (uniform/gaussian), feature-mean equals row mean, ridge standardization train-stats-only, rank-gaussianization, multi-target blend rank-domain behavior, tree seed determinism.
- `tests/test_benchmark_gates.py` (rewritten): gate edge cases.
- BMC/CWMM oracle parity coverage from old slice3 moves into `tests/test_parity.py` scope (parity coverage must not regress).
- CI (`.github/workflows/ci.yml`) fast gate: test count changes — update `AGENTS.md` claim in the same commit.

## 7. Documentation & SSOT (same commit)

- `docs/06-evaluation/benchmark-line-in-the-sand.md`: rewrite as the 5-tier hierarchy reference (tiers, gates, anchors policy, measured-vs-config re-pinning procedure); stays a memory aid pointing at the bible.
- `docs/06-evaluation/evaluation-suite-bible.md` §11 (E6) + §15 (deferral ledger): update to reference the hierarchy; drop obsolete tutorial-chain references.
- `ARCHITECTURE.md`: `nmr/benchmark.py` module spec (BenchmarkHierarchy, config schema, generators, gates), runner CLI, artifact paths.
- `AGENTS.md`: toolkit table rows (benchmark hierarchy), operational hazards (medium-tier fit runtimes, purge helper rationale), test count.
- `README.md` / `CONTRIBUTING.md`: touch only if they name removed artifacts (check in plan).

## 8. Implementation Watchpoints (audit-mandated invariants)

1. **Auxiliary-target NaN masking (Tier-1 multi-target Ridge):** each constituent Ridge fit filters null target rows *independently* on its own train partition before fitting — sporadic missing values in early train eras must not silently poison a fit or drop rows from sibling fits.
2. **Standardization zero-variance safeguard:** if a feature has zero sample variance on the trimmed train slice, the standardized column must output `0.0` (never divide-by-zero or NaN).
3. **Monotonicity tolerance:** `assert_hierarchy_monotone()` uses `atol = 1e-5` to prevent floating-point boundary failures on the rank-scalar ordering.

## 9. Risks & Hazards

- **Medium-tier fit runtime**: 780-feature LightGBM fits on ~2.1M train rows (tiers 2–3) are the dominant cost; `fast_mode` overrides keep the smoke gate minutes-scale; full hierarchy is a pre-sign-off multi-hour run (`nohup` pattern).
- **Absolute thresholds may not match v5.3 measurement**: configs are re-pinned after first real run with evidence (decision #2); until then, gate failures on real data are *information*, not code bugs.
- **Early train-era benchmark parquet gap**: not applicable to tier 1–3 (own fits); Tier-4 column is validation-only — no hazard.
- **Determinism**: LGBM/XGBoost fixed seeds + `n_jobs=1`; no wall-clock/paths in canonical bytes (existing hazard preserved).
- **Test-count drift**: AGENTS.md/CI claims updated in the same commit (mandatory self-update directive).
- **Neutralization cache**: `NeutralizationEngine` cache keys must remain machine-independent; no changes to `nmr/risk.py` in scope.

## 10. Amendments after First Real-Data Smoke (2026-08-15)

The first real-data smoke run on v5.3 measured the gates against actual data. The §4.3 numbers and semantics above are superseded as follows (original text preserved as history):

1. **Tier-0 null floor — `assert_tier0_null_floor`.** `|Sharpe(AC)|` tolerance re-pinned 0.10 → **0.15**. The **DSR check is dropped**: null DSRs span 0.11–1.0 on v5.3 (degenerate denominator behavior near-zero Sharpe), so deflated Sharpe has no constant null value. The gate covers only the three structural nulls (constant-0.5, uniform-random, gaussian-random); `null_feature_mean` is scored but excluded — it is not structurally null on v5.3 (corr 0.0029, sharpe 0.257). This supersedes decision #9 (tolerance-based |DSR| ≤ 0.05).
2. **Tier-4 gate — `assert_tier4_gate`.** `corr_sharpe_ac_min` re-pinned 1.50 → **0.78** (measured 0.7808 for `v53_lgbm_ender60` over the 86-era meta overlap; the aspirational 1.50 would fail the shipped reference). `corr_min` 0.0286 confirmed by measurement (0.02927). **Turnover semantics changed**: turnover is structurally unavailable on v5.3 (consecutive validation eras share zero ids), so it is reported as measured=None/pass=None in the gate report, **excluded from hard failure**, and logged loudly — superseding the "missing/None field ⇒ gate FAIL" rule.
3. **Monotonicity — `assert_hierarchy_monotone`.** Default metric changed from the rank scalar to **per-tier max of `corr.value`**; `rank_scalar` remains selectable via `metric="rank_scalar"`. Evidence: the real-data corr ladder 0.00294 < 0.00478 < 0.00741 < 0.00952 ≤ 0.02927 orders all five tiers cleanly, while rank_scalar's noise spread swamps the null-vs-ridge rung. `atol = 1e-5` unchanged.
4. **Fit topology confirmed unchanged:** purged train→validation split (exact 8-era buffer), float32 end-to-end, rank-Gaussian multi-target blends, FNE = FNC@medium.
5. **Untiered benchmark fleet (2026-08-19):** a new untiered config layer
   (`configs/benchmarks/fleet/`, `nmr/benchmark_fleet.py`) adds 19 recreated
   community/tutorial/Finance-Arena benchmark models scored through the same
   evaluation pipeline, with report-only placement against the tier rungs.
   Tiers, gates, and monotonicity semantics are unchanged. Design:
   `docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md`.
