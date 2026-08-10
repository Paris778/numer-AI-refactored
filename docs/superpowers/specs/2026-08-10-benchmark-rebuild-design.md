# Benchmark & Evidence Rebuild (v5.3) — Design

**Date:** 2026-08-10
**Status:** Approved by user (2026-08-10)
**Scope:** Close the 5 statistical blind spots from the institutional quant audit (83/100 report score), purge all legacy benchmarks/results/models, and rebuild the benchmark layer from scratch on v5.3 with empirical model evidence.

## 1. Objective

Transition the framework from a descriptive data report to an empirically validated model-intelligence suite:

1. Fix the audit's statistical gaps: scale-unstandardized W1, missing UQ, unwatermarked small samples, unvalidated linear screen.
2. Purge `artifacts/registry`, `artifacts/runs`, benchmark CSVs, `artifacts/bundles` (user-approved).
3. Rebuild benchmarks on v5.3: null baselines + S11 + benchmark models + a 6-variant GBDT feature campaign.
4. Calibrate the `stable` screen gate from campaign evidence (evidence-driven update, user-approved).

## 2. Approved Decisions

| ID | Decision |
|---|---|
| D1 | W1 standardized by train σ: `w1_norm = w1 / σ_train`; drift flag = `psi > 0.25 OR w1_norm > 0.50 OR |auc−0.5| > 0.1` |
| D2 | Block-bootstrap 95% CIs (via `nmr/inference.py`) on headline stats: per-feature mean IC, benchmark-era correlations, campaign variant metrics |
| D3 | Renderer watermark `[SMALL SAMPLE: 86 ERAS — HIGH SAMPLING VARIANCE]` on all `meta_model.parquet`-derived tables |
| D4 | Screen-derived feature sets registered via new `data.supplemental_feature_sets` config key (merged by `IngestionAgent`; key collision → `ValueError`) |
| D5 | Campaign: 6 variants (v1 all / v2 stable / v3 stable∪nonlinear / v4 drift-filtered / v5 small / v6 medium) × LightGBM fast + XGBoost GPU; standard preset arbitrates ranking disagreement (Stage B) |
| D6 | FNE reference feature set = `medium` (780) |
| D7 | `benchmark_runner.py` default data-dir moved v5.2 → v5.3 |
| D8 | Guardrail: `DEFAULT_MIN_MEAN_CORR` / `DEFAULT_MAX_ABS_DECAY` stay frozen until campaign CIs show statistically separated gains for v3/v4 over v2 |
| D9 | Guardrail: background campaign run logs stdout+stderr to `artifacts/campaigns/rebuild_v53.log` |

## 3. Implementation Parts

### Part 1 — Statistical fixes

- **1.1 W1 standardization** — `nmr/analysis.py`:
  - New module constant `WASSERSTEIN_NORM_FLAG_THRESHOLD = 0.50`.
  - `feature_drift_profile`: rename param `w1_threshold` → `w1_norm_threshold` (default = new constant); add `w1_norm` output column (`w1 / std(train sample)`; `None` when either side non-finite); `drifted` uses `w1_norm`.
  - Raw `w1` column retained for reference.
- **1.2 Block-bootstrap UQ** — `nmr/analysis.py` + `nmr/inference.py`:
  - `feature_ic_screen`: add `mean_corr_ci_lo` / `mean_corr_ci_hi` (stationary block bootstrap over **valid** eras; `block_bootstrap_ci` with horizon-consistent block length; default `n_boot=200`).
  - `benchmark_era_corr`: add CI columns on the mean per-era correlation aggregate.
  - Constants: `IC_CI_BOOT = 200`.
- **1.3 Watermark** — `render_dataset_report.py`: `_META_WARNING` banner paragraph inserted above every meta-derived section (meta_orthogonality, neutralized IC frontier).

### Part 2 — Purge (approved)

Delete: `artifacts/registry/*`, `artifacts/runs/*`, `artifacts/benchmark_scores*.csv`, `artifacts/benchmark_test_era_labels*.csv`, `artifacts/bundles/*`. Code, `data/`, docs untouched.

### Part 3 — Screen-derived feature sets

- `nmr/config.py` `DataConfig`: new field `supplemental_feature_sets: Path | None = None`; validation: non-empty path when set.
- `nmr/data.py` `IngestionAgent._metadata_raw()`: merge supplemental file (`{"feature_sets": {...}}` shape) into `feature_sets`; **key collision with features.json → `ValueError`** listing colliding keys; missing file → `ValueError`.
- `analyze_dataset.py`: new stage `derived_sets` (no deps; ordered after `drift`, before `manifest`):
  - Reads `feature_ic_screen.parquet` + `feature_drift_profile.parquet` from the out dir (missing → `RuntimeError` with explicit message).
  - Primary reference target = first target column present in the screen file.
  - Writes `derived_feature_sets.json` atomically: `screen_stable` (stable), `screen_nonlinear` (stable=False AND nonlinear), `screen_linear_or_nonlinear` (stable OR nonlinear), `screen_drift_filtered` (v3 members AND NOT drifted). Sorted lists; keys always present.

### Part 4 — Feature campaign (6 variants)

- Configs: `configs/campaigns/benchmark-rebuild-v1/<backend>_v<n>.yaml` (12 files: lightgbm-fast + xgboost-gpu × 6 subsets).
  - `data.feature_subset`: all / screen_stable / screen_linear_or_nonlinear / screen_drift_filtered / small / medium.
  - `data.supplemental_feature_sets` set for v2/v3/v4 only.
  - Fixed seed; `split.purge_eras: 8`; 20D; walk-forward; same model params across variants per backend.
- Dry-run → `run_campaign.py --name benchmark-rebuild-v1` → background `nohup` with log `artifacts/campaigns/rebuild_v53.log` (stdout+stderr).
- Stage B: standard preset on variants whose cross-backend ranking disagrees.

### Part 5 — Benchmark rebuild (v5.3)

- `benchmark_runner.py`: default `--data-dir` → `data/v5.3`; full run (`n_boot=300`).
- Campaign evidence scored per-variant from OOF: val mean IC, IC Sharpe, max drawdown of cumulative IC, FNE@100% vs medium, bootstrap 95% CI — into report §7 (not merged into `benchmark_scores.csv`).

### Part 6 — Evidence synthesis & docs

- Report §7 "Modeling Implications": campaign table + paired era comparisons (v2 vs v3 vs v4) + top-gain features per variant + FNE frontier.
- Screen gate decision per D8; if triggered, update `DEFAULT_*` constants + ARCHITECTURE.md §P + tests in same commit.
- Docs: AGENTS.md (benchmark conventions, supplemental sets, runner v5.3 default), ARCHITECTURE.md §P/§R, count guards. Full `pytest -q` green (549 + new).

## 4. Test Plan

| Change | Test |
|---|---|
| W1 norm | `tests/test_analysis.py`: scale-invariance (bounded vs unbounded features); `w1_norm` column correctness; flag threshold shift |
| Screen CIs | synthetic eras: CI brackets mean; degenerate eras excluded from bootstrap; `n_boot=1` smoke path |
| Watermark | `tests/test_docs_hygiene.py` / renderer test: banner present above meta sections |
| Supplemental sets | `tests/test_data.py`: merge, collision `ValueError`, missing file `ValueError`, determinism (sorted) |
| Derived sets stage | `tests/test_analyze_dataset.py`: fixture dumps → expected keys/sets; missing inputs → `RuntimeError` |
| Runner default | smoke: default data-dir resolves v5.3 |

## 5. Verification Gates

1. `./.venv/Scripts/python -m pytest -q` (549 + new tests).
2. `./.venv/Scripts/python analyze_dataset.py --only drift,derived_sets` → updated `feature_drift_profile.parquet` (w1_norm) + `derived_feature_sets.json`.
3. Campaign dry-run prints 12 run_ids; real run completes; log at `artifacts/campaigns/rebuild_v53.log`.
4. `benchmark_runner.py` full run on v5.3 → `benchmark_scores.csv` with nulls + S11 + benchmark models.
5. Report re-rendered; §7 contains campaign table with CIs.

## 6. Risks

- Purge irreversible (approved; registry rebuilt from new runs).
- Fast-preset evidence noisy → Stage B escalation path.
- Screen threshold change alters future campaigns → documented + human-visible in report.
- GPU vs CPU determinism: campaign runs per-device; final champion training remains CPU (deploy invariant).
