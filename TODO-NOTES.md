# TODO / Session Notes (LLM agent state map)

> Record for future LLM agents. Last updated: 2026-08-10 (benchmark-rebuild session).

## Current state

- **Benchmark & Evidence Rebuild (v5.3)** in progress. Spec: `docs/superpowers/specs/2026-08-10-benchmark-rebuild-design.md` (approved by user).
- **Purge done (user-approved):** `artifacts/registry/*` (2 runs + champion), `artifacts/runs/*`, `artifacts/benchmark_scores*.csv`, era labels, `artifacts/bundles/*` — all empty. No champion until a new run is promoted.
- **Code shipped (tests green, suite 564):**
  - `data.supplemental_feature_sets` config key — `IngestionAgent` merges `{"feature_sets": {...}}` (collision ⇒ `ValueError`).
  - `analyze_dataset.py` 16th stage `derived_sets` → `derived_feature_sets.json` (`screen_stable`, `screen_nonlinear`, `screen_linear_or_nonlinear`, `screen_drift_filtered`; primary target = `target`).
  - `feature_drift_profile` → `w1_norm = w1 / σ_train`; drift flag `psi > 0.25 OR w1_norm > 0.50 OR |auc−0.5| > 0.1`.
  - `feature_ic_screen` → `mean_corr_ci_lo/hi` (seeded block bootstrap, valid eras only).
  - `nmr/analysis.neutralized_ic_series` (per-era FNE long form) + `nmr/meta.campaign_evidence` (§7 evidence assembler).
  - `RunRegistry.record` persists `validation_preds.parquet` when validation scorecard enabled.
  - `render_dataset_report.py` §4.2 CI columns, §4.3 `w1_norm`, meta small-sample watermark, §7.1/§7.2 campaign tables (`--campaign-log` + `--registry` args).
  - `benchmark_runner.py` default data-dir `data/v5.2` → `data/v5.3`.
- **Background jobs (logs):**
  - Campaign: `nohup run_campaign.py` 12 configs (`configs/campaigns/benchmark-rebuild-v1/{lgbm,xgb}_v{1..6}.yaml`) → `artifacts/campaigns/rebuild_v53.log`.
  - Benchmark rebuild: `nohup benchmark_runner.py` → `artifacts/benchmark_full_run.log`.
  - Screens re-run (CI columns): `nohup analyze_dataset.py --only screens` → `artifacts/reports/dataset_analysis/screens_rerun.log`.
- **Pending (P6):** campaign completion → `campaign_evidence` → re-render report (`render_dataset_report.py --campaign-log artifacts/campaigns/benchmark-rebuild-v1.json`) → **screen gate decision** (user pre-approved evidence-driven update of `DEFAULT_MIN_MEAN_CORR`/gate if v3/v4 beat v2 with CI-excluding-zero on both backends) → final `pytest -q` (564) + smoke gates → commit (ASK USER FIRST — no commits without explicit instruction).
- **Key data facts (v5.3):** screen on `target` (20D) = only **3 stable** features (423 stable + 98 nonlinear for `target_agnes_60`); meta FNE flat ≈ 0.022; 2/780 meta-orthogonal features; min eigenvalue 0.0042 (PSD).

## Standing flags / rules

- No git mutations without explicit user instruction (pending commits must be asked).
- `./.venv/Scripts/pip` is a shim into legacy `../numer-AI/.venv` — always `./.venv/Scripts/python -m pip`.
- Long jobs: `nohup ... > log 2>&1 &`; poll logs. Test count: 564 (AGENTS/README/CONTRIBUTING + `pytest --collect-only`).
- GPU: `model.device` (`auto|gpu|cpu`); xgboost 3.x needs `device="cuda"` + `tree_method="hist"` (no `gpu_hist`).
- Registry/champion currently EMPTY by design (purged) — do not treat as data loss.
